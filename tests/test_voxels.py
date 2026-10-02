"""
Tests for ``ordito.voxels``.

Four references, each where it is the closest thing to an oracle: **open3d** for the two
voxelizers, the down-sample and the membership query (it computes the same tri-box accept set and
the same half-voxel anchor); **scipy.ndimage** through trimesh's ``voxel.morphology`` for dilation,
erosion and hole filling, which are exactly its binary morphology; **trimesh** for
``fill_orthographic`` and ``multibox``; and **igl** for ``grid`` and
``unique_sparse_voxel_corners``.

Two conventions matter throughout:

- **Every fixture is translated by a non-round offset.** open3d floors cell coordinates in
  ``float64`` and ordito in ``float32``, so a vertex sitting exactly on a cell plane can land in
  different cells. Nothing here is allowed to sit on that edge.
- **Row order is leaf-major**, not lexicographic, so every set comparison sorts both sides. The
  order is nonetheless deterministic and identical on CPU and CUDA, which
  ``test_cell_order_is_deterministic`` and ``test_cell_order_matches_across_devices`` assert
  directly — the device-agnostic design rests on it.

The module runs on the ordinary ``device`` fixture like every other one: ``allocate_by_voxels``
gained a CPU path in Warp 1.16, so nothing here is CUDA-only.
"""

from __future__ import annotations

from typing import Literal, cast

import igl
import numpy as np
import numpy.typing as npt
import open3d as o3d
import pytest
import pytorch3d.ops as p3d_ops
import pyvista as pv
import scipy.ndimage as ndi
import torch
import trimesh as tm
import trimesh.voxel.ops as tm_voxel_ops
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from tests.comparisons import assert_nonconstant, lexsort_rows
from tests.conversions import (
    meshlib_bitset_to_numpy,
    numpy_to_meshlib,
    numpy_to_meshlib_bitset,
    numpy_to_warp,
    points_to_meshlib,
    points_to_torch,
    points_to_warp,
    pytorch3d_to_numpy,
    trimesh_to_open3d,
    trimesh_to_pyvista,
    warp_empty,
    warp_to_trimesh,
)

# A translation with no round coordinate, so no vertex of any fixture lands on a cell plane.
_OFFSET = np.array([0.137, -0.219, 0.331])

# igl's ``unique_sparse_voxel_corners`` numbers a cell's corners in yxz binary-counting order and
# reads column 0 of its subscripts as *y*; this is the resulting permutation of ordito's
# ``c = 4 dx + 2 dy + dz`` columns (it is its own inverse).
_IGL_CORNER_ORDER = [1, 0, 2, 3, 5, 4, 6, 7]


@pytest.fixture
def sphere(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Move the shared ``icosphere`` to a non-round offset, as trimesh plus its Warp buffers.

    The translation is the point of the local fixture: a voxel grid built around a sphere centred
    on the origin can pass while every index is off by half a cell, because the offsets cancel.
    """
    mesh_tm, _mesh_wp = icosphere
    mesh_tm.apply_translation(_OFFSET)
    return (mesh_tm, *numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device))


def _cloud(
    mesh_tm: tm.Trimesh, device: str, n: int = 20000
) -> tuple[np.ndarray, wp.array[wp.vec3]]:
    """Sample the surface of ``mesh_tm`` deterministically, as NumPy and Warp."""
    points_np = tm.sample.sample_surface(mesh_tm, n, seed=7)[0]
    return points_np, points_to_warp(points_np, device)


def _dense(
    grid: wp.Volume, origin_cell: tuple[int, int, int], shape: tuple[int, int, int]
) -> np.ndarray:
    """Dense occupancy of ``grid`` over an explicit cell box, as NumPy."""
    return od.voxels.to_dense(grid, origin_cell=origin_cell, shape=shape)[0].numpy()


# ---------------------------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------------------------


@pytest.mark.parity("voxelize_mesh", "open3d")
def test_voxelize_mesh_matches_open3d(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """
    Class A: the accepted cell sets are identical, compared after ``lexsort_rows`` on both sides.

    open3d walks a ``round((max - min) / voxel_size) + 2`` candidate window and ordito the exact
    inclusive AABB window, but a cell outside a triangle's AABB cannot overlap the triangle, so the
    windows differ only in rejected candidates. Both then run Moller's 13-axis test with no epsilon.
    """
    mesh_tm, vertices_wp, faces_wp = sphere
    voxel_size = 0.13
    origin = mesh_tm.vertices.min(axis=0) - 0.5 * voxel_size
    grid = od.voxels.voxelize_mesh(
        vertices_wp, faces_wp, voxel_size, origin=wp.vec3(*origin.tolist())
    )

    grid_o3d = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        trimesh_to_open3d(mesh_tm),
        voxel_size=voxel_size,
        min_bound=origin,
        max_bound=origin
        + np.ceil((mesh_tm.vertices.max(axis=0) - origin) / voxel_size) * voxel_size,
    )
    cells_o3d = np.array([voxel.grid_index for voxel in grid_o3d.get_voxels()], dtype=np.int32)

    assert cells_o3d.shape[0] > 0
    assert np.array_equal(lexsort_rows(od.voxels.cells(grid).numpy()), lexsort_rows(cells_o3d))


@pytest.mark.parity("voxelize_mesh", "meshlib")
def test_voxelize_mesh_matches_meshlib(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]], device: str
):
    """
    Class C (two representations): MeshLib returns a **distance field**, not an occupancy set.

    ``meshToVolume`` builds an OpenVDB narrow band -- unsigned distances in *voxel* units, clamped
    at ``surfaceOffset`` -- so there is no cell set to compare with. What is comparable is the
    geometry both encode, and the two directions of that are exact statements rather than
    tolerances:

    - a cell ordito accepts contains a piece of the surface, so its **centre** is within half a
      cell diagonal of it: ``distance <= sqrt(3)/2`` voxels;
    - a grid node whose distance is under **half** a voxel has its closest surface point inside the
      surrounding cell, so ordito must have accepted that cell.

    Measured on the translated ``icosphere(3)`` at a 0.1 voxel: all 1 898 accepted cells sample at
    most **0.806** against the 0.866 bound, and all 1 226 nodes under 0.5 land inside an accepted
    cell -- both at 100 %, in both directions.

    Three facts about the grid, none of them documented and all of them load-bearing. The field is
    read through ``vdbVolumeToSimpleVolume`` and ``mn.getNumpy3Darray``; its samples sit on grid
    **nodes**, not cell centres; and its origin is ``bbox.min - surfaceOffset * voxel``, which is
    also why ``dims`` comes out as the mesh's extent in cells plus ``2 * surfaceOffset``.

    **Bug class excluded:** a grid anchored differently -- the failure mode every voxel comparison
    in this module is written against. **Mutation probe, measured:** displacing the sample points by
    half a voxel puts **19.3 %** of the cells outside the bound and takes the maximum from 0.806 to
    1.661, so the assert bites at exactly the error it is for.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    voxel_size, surface_offset = 0.1, 3.0

    grid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, voxel_size)
    centres_np = od.voxels.cell_centers(grid).numpy().astype(np.float64)

    vertices_np = vertices_wp.numpy().astype(np.float64)
    mesh_ml = numpy_to_meshlib(vertices_np, faces_wp.numpy().reshape(-1, 3))
    params_ml = mm.MeshToVolumeParams()
    params_ml.voxelSize = mm.Vector3f(voxel_size, voxel_size, voxel_size)
    params_ml.surfaceOffset = surface_offset
    field_np = cast(
        npt.NDArray[np.float64],
        mn.getNumpy3Darray(
            mm.vdbVolumeToSimpleVolume(mm.meshToVolume(mm.MeshPart(mesh_ml), params_ml))
        ),
    )
    origin_np = vertices_np.min(axis=0) - surface_offset * voxel_size

    # Non-vacuity: a narrow band around the surface, not a constant field.
    assert field_np.min() < 0.1 < field_np.max()
    assert len(centres_np) > 100

    # 1. Every accepted cell's centre is within half a cell diagonal of the surface.
    sampled_np = ndi.map_coordinates(
        field_np, ((centres_np - origin_np) / voxel_size).T, order=1, mode="nearest"
    )
    assert sampled_np.max() <= np.sqrt(3.0) / 2.0

    # 2. Every node closer than half a voxel lies inside a cell ordito accepted.
    near_np = np.argwhere(field_np < 0.5)
    assert near_np.shape[0] > 100
    node_positions_np = origin_np + near_np * voxel_size
    occupied_wp = od.voxels.occupancy_at_points(grid, points_to_warp(node_positions_np, device))
    assert occupied_wp.numpy().all()


@pytest.mark.parity("voxelize_mesh", "pyvista")
def test_voxelize_mesh_solid_contains_the_pyvista_mask(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """
    Class B, and the named transform is the **sample convention**, which is the whole finding.

    ``voxelize_binary_mask`` is *solid*, so it maps to ``mode="solid"`` and not to the default
    surface mode -- measured 17 256 filled of 32 768 on a 32-cube against ordito's 4 496 surface
    cells. But it is not the same set even then: VTK writes a **point** mask, testing each lattice
    *point* against the closed surface, while ordito accepts a **cell** the closed triangle meets.
    So VTK's set is contained in ordito's and the difference is exactly the boundary shell, which
    is what the asserts below state: **0** cells are pyvista-only, 2 480 are ordito-only, and every
    one of those 2 480 is in ordito's own surface voxelization (of 4 760 surface cells).

    Reading that containment as a disagreement is the trap; a comparison that expected equality
    would fail by 12.7% of the cells on a correct implementation. The grid is also cell-centred on
    VTK's side -- origin ``-0.96875`` at spacing ``0.0625`` for a unit sphere -- so ordito's origin
    is shifted by half a voxel to put the two lattices in register.
    """
    mesh_tm, vertices_wp, faces_wp = sphere
    dimensions = (32, 32, 32)

    mask_pv = trimesh_to_pyvista(mesh_tm).voxelize_binary_mask(dimensions=dimensions)
    spacing_np = np.asarray(mask_pv.spacing)
    origin_np = np.asarray(mask_pv.origin)
    mask_np = np.asarray(mask_pv.point_data["mask"]).astype(bool).reshape(dimensions, order="F")
    assert 0 < int(mask_np.sum()) < mask_np.size, "the reference filled some cells, not all"

    origin_wp = wp.vec3(*(origin_np - 0.5 * spacing_np).tolist())
    solid = od.voxels.voxelize_mesh(
        vertices_wp, faces_wp, float(spacing_np[0]), origin=origin_wp, mode="solid"
    )
    surface = od.voxels.voxelize_mesh(vertices_wp, faces_wp, float(spacing_np[0]), origin=origin_wp)

    def dense(grid: wp.Volume) -> np.ndarray:
        cells_np = od.voxels.cells(grid).numpy()
        inside_np = np.all((cells_np >= 0) & (cells_np < np.array(dimensions)), axis=1)
        occupied_np = np.zeros(dimensions, dtype=bool)
        kept_np = cells_np[inside_np]
        occupied_np[kept_np[:, 0], kept_np[:, 1], kept_np[:, 2]] = True
        return occupied_np

    solid_np, surface_np = dense(solid), dense(surface)
    assert not (mask_np & ~solid_np).any(), "VTK's point mask must be contained in the solid set"
    excess_np = solid_np & ~mask_np
    assert excess_np.any(), "and the two conventions must actually differ, or the row is vacuous"
    assert not (excess_np & ~surface_np).any(), "every extra cell is a boundary-shell cell"


def test_voxelize_mesh_solid_is_sealed(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]], device: str
):
    """``mode="solid"`` fills the interior, and no 6-connected path leaves it."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    surface = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.12)
    solid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.12, mode="solid")
    assert int(od.voxels.cells(solid).shape[0]) > int(od.voxels.cells(surface).shape[0])

    # The padded shell of the solid's own bounding box must stay empty, and the interior must be
    # separated from it: filling again changes nothing.
    refilled = od.voxels.fill_cavities(solid)
    lower, extent = _bounds_of(solid)
    assert np.array_equal(_dense(solid, lower, extent), _dense(refilled, lower, extent))
    assert od.voxels.occupancy_at_points(
        solid, wp.array([wp.vec3(*_OFFSET.tolist())], dtype=wp.vec3, device=device)
    ).numpy()[0]


def test_voxelize_mesh_rejects_too_many_candidates(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """The candidate guard raises before allocating, and names ``voxel_size``."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    with pytest.raises(ValueError, match="voxel_size"):
        od.voxels.voxelize_mesh(vertices_wp, faces_wp, 1e-4, max_candidates=1000)


@pytest.mark.parity("voxelize_points", "open3d")
def test_voxelize_points_matches_open3d(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]], device: str
):
    """
    Class A: same occupied cells for the same cloud and pitch, both sides ``lexsort``ed.

    Also pins the default anchor: ordito's ``origin`` default is open3d's
    ``min_bound - 0.5 * voxel_size``, so the two agree cell for cell without either being told the
    other's transform.
    """
    mesh_tm, _vertices_wp, _faces_wp = sphere
    points_np, points_wp = _cloud(mesh_tm, device)
    voxel_size = 0.13

    grid = od.voxels.voxelize_points(points_wp, voxel_size)
    cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np))
    grid_o3d = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud_o3d, voxel_size=voxel_size)
    cells_o3d = np.array([voxel.grid_index for voxel in grid_o3d.get_voxels()], dtype=np.int32)

    assert cells_o3d.shape[0] > 0
    assert np.array_equal(lexsort_rows(od.voxels.cells(grid).numpy()), lexsort_rows(cells_o3d))
    assert np.allclose(np.asarray(grid_o3d.origin), od.voxels.grid_transform(grid)[1], atol=1e-6)


@pytest.mark.parity("voxel_down_sample", "open3d")
def test_voxel_down_sample_matches_open3d(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]], device: str
):
    """
    Class B: both sides re-keyed by the integer cell before their positions are compared.

    The positions themselves are never ``lexsort``ed — two cell means that tie in ``float32`` and
    differ in the 16th digit in ``float64`` would order differently and fail by the full coordinate
    range. The integer cell is the correspondence; the positions are then compared in place.
    """
    mesh_tm, _vertices_wp, _faces_wp = sphere
    points_np, points_wp = _cloud(mesh_tm, device)
    voxel_size = 0.13

    grid = od.voxels.voxelize_points(points_wp, voxel_size)
    _size, origin = od.voxels.grid_transform(grid)
    pooled_wp = od.voxels.voxel_down_sample(points_wp, voxel_size)
    cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np))
    pooled_o3d = np.asarray(cloud_o3d.voxel_down_sample(voxel_size).points)

    assert pooled_o3d.shape[0] > 0
    assert pooled_wp.size == pooled_o3d.shape[0]
    key_o3d = od.voxels.cell_indices(
        points_to_warp(pooled_o3d, device), voxel_size, origin=origin
    ).numpy()
    key_wp = od.voxels.cells(grid).numpy()
    order_o3d = np.lexsort(key_o3d.T[::-1])
    order_wp = np.lexsort(key_wp.T[::-1])
    assert np.array_equal(key_o3d[order_o3d], key_wp[order_wp])
    assert np.allclose(pooled_wp.numpy()[order_wp], pooled_o3d[order_o3d], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("voxel_size", [0.13, 0.26])
@pytest.mark.parity("voxel_down_sample", "meshlib")
def test_voxel_down_sample_matches_meshlib(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]], device: str, voxel_size: float
):
    """
    Class C (a count statistic): ``pointGridSampling`` *selects* a point, it does not pool one.

    The difference is the operation, not the parameter: MeshLib returns a ``VertBitSet`` picking one
    surviving input point per occupied cell, where ``voxel_down_sample`` returns the cell **mean**,
    a position that need not be an input point at all. So the positions are not comparable and the
    shared quantity is how many cells the cloud occupies.

    Measured on the translated icosphere cloud, ordito against MeshLib: 429 / 464 at a 5 % voxel
    and 136 / 128 at 10 %. The residual is grid *anchoring* -- neither library documents where cell
    zero starts -- so the bound is 1.3x, the same one ``cluster_decimate`` carries against
    ``verticesGridSampling`` for the same reason.

    **Mutation probe, measured:** the count falls ~3x per doubling of the voxel size on both sides,
    so a factor-of-two error in the cell size lands far outside the 1.3x band that anchoring
    accounts for. open3d carries the exact positional comparison for this group.
    """
    mesh_tm, _vertices_wp, _faces_wp = sphere
    points_np, points_wp = _cloud(mesh_tm, device)

    pooled_wp = od.voxels.voxel_down_sample(points_wp, voxel_size)
    cloud_ml = points_to_meshlib(points_np)
    sampled_ml = mm.pointGridSampling(mm.PointCloudPart(cloud_ml), voxel_size)

    n_cells_ml = sampled_ml.count()
    n_cells_wp = pooled_wp.size
    assert 0 < n_cells_ml < points_np.shape[0]  # non-vacuity: it really sampled down
    assert 0 < n_cells_wp < points_np.shape[0]
    assert 1.0 / 1.3 < n_cells_wp / n_cells_ml < 1.3


def test_voxel_down_sample_pooling_modes(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]], device: str
):
    """``min`` / ``max`` bracket the cell mean, and ``sum`` is that mean times the count."""
    mesh_tm, _vertices_wp, _faces_wp = sphere
    _points_np, points_wp = _cloud(mesh_tm, device, n=4000)
    voxel_size = 0.2

    lowest = od.voxels.voxel_down_sample(points_wp, voxel_size, pooling="min").numpy()
    highest = od.voxels.voxel_down_sample(points_wp, voxel_size, pooling="max").numpy()
    average, inverse = od.voxels.voxel_down_sample(
        points_wp, voxel_size, pooling="mean", return_inverse=True
    )
    total = od.voxels.voxel_down_sample(points_wp, voxel_size, pooling="sum").numpy()

    assert (lowest <= average.numpy() + 1e-6).all()
    assert (average.numpy() <= highest + 1e-6).all()
    counts = np.bincount(inverse.numpy(), minlength=average.size)
    assert counts.min() > 0
    assert np.allclose(total, average.numpy() * counts[:, None], rtol=1e-5, atol=1e-5)


def test_voxel_down_sample_mean_is_reproducible(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]], device: str
):
    """The deterministic branch is bitwise stable: no float atomics in the ``mean`` reduce."""
    mesh_tm, _vertices_wp, _faces_wp = sphere
    _points_np, points_wp = _cloud(mesh_tm, device, n=8000)
    first = od.voxels.voxel_down_sample(points_wp, 0.15).numpy()
    second = od.voxels.voxel_down_sample(points_wp, 0.15).numpy()
    assert np.array_equal(first, second)


def test_pool_by_voxel_ignores_outside_points(device: str):
    """Points outside the grid contribute nothing, and voxels with no point pool to zero."""
    cells_np = np.array([[0, 0, 0], [5, 5, 5]], dtype=np.int32)
    grid = od.voxels.from_cells(
        wp.array(cells_np, dtype=wp.int32, device=device), 1.0, wp.vec3(0.0, 0.0, 0.0)
    )
    points_np = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [-9.0, -9.0, -9.0]])
    points_wp = points_to_warp(points_np, device)
    pooled = od.voxels.pool_by_voxel(grid, points_wp, points_wp).numpy()

    slot = od.voxels.cells(grid).numpy().tolist().index([0, 0, 0])
    other = 1 - slot
    assert np.allclose(pooled[slot], [0.5, 0.5, 0.5])
    assert np.array_equal(pooled[other], np.zeros(3))


# ---------------------------------------------------------------------------------------------
# Grid <-> world
# ---------------------------------------------------------------------------------------------


def test_cells_round_trip_through_from_cells(device: str):
    """
    The half-voxel translation, pinned directly.

    NanoVDB centres voxel ``i`` on index-space coordinate ``i``, so a transform anchored at
    ``origin`` rather than ``origin + 0.5 * voxel_size`` returns every cell shifted by ``+1`` on
    every axis. The set below includes negatives and a coordinate past the first 8-cubed leaf, so
    that failure mode cannot hide.
    """
    cells_np = np.array([[0, 0, 0], [3, -4, 5], [-7, 2, 9], [63, 63, 63]], dtype=np.int32)
    grid = od.voxels.from_cells(
        wp.array(cells_np, dtype=wp.int32, device=device), 0.25, wp.vec3(1.3, -2.7, 0.4)
    )
    assert np.array_equal(lexsort_rows(od.voxels.cells(grid).numpy()), lexsort_rows(cells_np))

    voxel_size, origin = od.voxels.grid_transform(grid)
    assert voxel_size == pytest.approx(0.25)
    assert np.allclose(origin, [1.3, -2.7, 0.4], atol=1e-6)


def test_cell_slots_are_the_row_indices(device: str):
    """
    The volume's linear index *is* the row index of [`cells`][ordito.voxels.cells].

    Every payload array in the module depends on it, so probing the cells back must give
    ``0..n-1`` in order.
    """
    rng = np.random.default_rng(11)
    cells_np = np.unique(rng.integers(-6, 7, size=(400, 3)).astype(np.int32), axis=0)
    grid = od.voxels.from_cells(
        wp.array(cells_np, dtype=wp.int32, device=device), 0.5, wp.vec3(0.0, 0.0, 0.0)
    )
    voxels = od.voxels.cells(grid)
    assert int(voxels.shape[0]) == cells_np.shape[0]
    assert od.voxels.occupancy_at_cells(grid, voxels).numpy().all()

    centers = od.voxels.cell_centers(grid)
    assert od.voxels.occupancy_at_points(grid, centers).numpy().all()
    # The centre of row k must probe back to row k, which is the statement above.
    pooled = od.voxels.pool_by_voxel(grid, centers, centers).numpy()
    assert np.allclose(pooled, centers.numpy(), rtol=1e-6, atol=1e-6)


def test_cell_order_is_deterministic(device: str):
    """
    Class of bug this excludes: a build order that depends on the input order.

    Two builds from differently shuffled cell rows return byte-identical
    [`cells`][ordito.voxels.cells]. The order is *leaf-major*, checked here on a multi-leaf set by
    an explicit example rather than by global sortedness, which is false for it.
    """
    rng = np.random.default_rng(3)
    cells_np = np.unique(rng.integers(-9, 10, size=(500, 3)).astype(np.int32), axis=0)
    first = od.voxels.cells(
        od.voxels.from_cells(
            wp.array(cells_np, dtype=wp.int32, device=device), 0.5, wp.vec3(0.0, 0.0, 0.0)
        )
    ).numpy()
    shuffled = cells_np[rng.permutation(cells_np.shape[0])]
    second = od.voxels.cells(
        od.voxels.from_cells(
            wp.array(np.ascontiguousarray(shuffled), dtype=wp.int32, device=device),
            0.5,
            wp.vec3(0.0, 0.0, 0.0),
        )
    ).numpy()
    assert np.array_equal(first, second)

    # Leaf-major: an 8-cubed leaf's cells come out together, ahead of the next leaf's, even though
    # a globally lexicographic order would interleave them.
    block = np.array([[0, 0, 0], [1, 1, 1], [8, 0, 0], [0, 0, 8]], dtype=np.int32)
    rows = od.voxels.cells(
        od.voxels.from_cells(
            wp.array(block, dtype=wp.int32, device=device), 1.0, wp.vec3(0.0, 0.0, 0.0)
        )
    ).numpy()
    leaves = rows // 8
    assert np.array_equal(leaves, lexsort_rows(leaves))
    assert not np.array_equal(rows, lexsort_rows(rows))


def test_cell_order_matches_across_devices():
    """
    The claim the whole device-agnostic design rests on: CPU and CUDA return the same permutation.

    Reported as a skip rather than passing silently when only one device is available — a
    one-device run would leave the claim untested while looking green.
    """
    if not wp.is_cuda_available():
        pytest.skip("needs both a CPU and a CUDA device to compare the two row orders")
    rng = np.random.default_rng(5)
    cells_np = np.ascontiguousarray(
        np.unique(rng.integers(-9, 10, size=(499, 3)).astype(np.int32), axis=0)
    )
    rows = [
        od.voxels.cells(
            od.voxels.from_cells(
                wp.array(cells_np, dtype=wp.int32, device=device), 0.5, wp.vec3(0.0, 0.0, 0.0)
            )
        ).numpy()
        for device in ("cpu", "cuda:0")
    ]
    assert np.array_equal(rows[0], rows[1])


def test_cells_sorted_order_is_the_unique_rows_order(device: str):
    """``order="sorted"`` reproduces ``grouping.unique_rows``'s row order exactly."""
    rng = np.random.default_rng(17)
    cells_np = np.ascontiguousarray(rng.integers(0, 12, size=(600, 3)).astype(np.int32))
    cells_wp = odt.as_array2d(wp.array(cells_np, dtype=wp.int32, device=device), wp.int32)
    grid = od.voxels.from_cells(cells_wp, 0.5, wp.vec3(0.0, 0.0, 0.0))
    assert np.array_equal(
        od.voxels.cells(grid, order="sorted").numpy(), od.grouping.unique_rows(cells_wp).numpy()
    )


def test_cells_sorted_order_handles_negative_cells(device: str):
    """The shift that makes the packed key well defined is order-preserving, negatives included."""
    cells_np = np.array([[0, 0, 0], [3, -4, 5], [-7, 2, 9], [-7, 2, 8]], dtype=np.int32)
    grid = od.voxels.from_cells(
        wp.array(cells_np, dtype=wp.int32, device=device), 0.5, wp.vec3(0.0, 0.0, 0.0)
    )
    rows = od.voxels.cells(grid, order="sorted").numpy()
    # Ascending in the packed key means lexicographic with x as the *last* tiebreak.
    assert np.array_equal(rows, rows[np.lexsort((rows[:, 0], rows[:, 1], rows[:, 2]))])


@pytest.mark.parity("occupancy_at_points", "open3d")
def test_occupancy_at_points_matches_open3d(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]], device: str
):
    """
    Class A: identical boolean masks against ``check_if_included``.

    The query set holds points both inside and outside, so a constant answer cannot pass.
    """
    mesh_tm, _vertices_wp, _faces_wp = sphere
    points_np, points_wp = _cloud(mesh_tm, device, n=5000)
    voxel_size = 0.15
    grid = od.voxels.voxelize_points(points_wp, voxel_size)

    queries_np = np.concatenate([points_np[:400], points_np[:400] + 5.0])
    queries_wp = points_to_warp(queries_np, device)
    cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np))
    grid_o3d = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud_o3d, voxel_size=voxel_size)
    inside_o3d = np.array(
        grid_o3d.check_if_included(o3d.utility.Vector3dVector(queries_np)), dtype=bool
    )

    assert inside_o3d.any()
    assert not inside_o3d.all()
    assert np.array_equal(od.voxels.occupancy_at_points(grid, queries_wp).numpy(), inside_o3d)


@pytest.mark.parity("grid_points", "igl", "pyvista")
def test_grid_points_matches_igl(device: str):
    """
    Class B: ``igl.grid`` and VTK's ``ImageData`` emit the same lattice over the unit cube.

    Compared as sets against igl, because its flattening order is its own.

    pyvista's is knowable, so it is compared **exactly** through the permutation rather than as a
    set: ``ImageData(...).points`` runs **x fastest** where ordito runs z fastest, so reading
    pyvista's buffer in Fortran order and flattening in C order is ordito's buffer element for
    element. That is a stronger claim than a set comparison and it is what pins the ordering both
    ways -- a set comparison passes for any permutation, including a transposed one.
    """
    shape = (4, 5, 6)
    lattice_wp = od.voxels.grid_points(
        shape, bounds=(wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 1.0, 1.0)), device=device
    ).numpy()
    lattice_igl = np.asarray(igl.grid(np.array(shape)))

    assert lattice_igl.shape == lattice_wp.shape
    assert np.allclose(
        lexsort_rows(np.round(lattice_igl, 9)), lexsort_rows(np.round(lattice_wp, 9)), atol=1e-6
    )
    # pyvista: x fastest, so its Fortran reading is ordito's C reading, element for element.
    lattice_pv = np.asarray(
        pv.ImageData(
            dimensions=shape,
            origin=(0.0, 0.0, 0.0),
            spacing=tuple(1.0 / max(n - 1, 1) for n in shape),
        ).points
    )
    assert lattice_pv.shape == lattice_wp.shape
    assert np.allclose(
        lattice_pv.reshape(*shape, 3, order="F").reshape(-1, 3), lattice_wp, atol=1e-6
    )

    # C order, z fastest, and the two corners land on ``bounds``.
    assert np.allclose(lattice_wp[0], [0.0, 0.0, 0.0])
    assert np.allclose(lattice_wp[-1], [1.0, 1.0, 1.0])
    assert np.allclose(lattice_wp[1], [0.0, 0.0, 0.2])


@pytest.mark.parity("splat_onto_grid", "pytorch3d")
def test_splat_onto_grid_matches_pytorch3d(device: str):
    """
    Class B: ``add_points_features_to_volume_densities_features`` under one **axis transpose**.

    Both outputs agree at exactly **0.0** on the host and at 4.8e-07 / 3.6e-07 on CUDA, where the
    two sides' atomic accumulations commit in different orders -- and that pins the whole
    convention: the trilinear weights, the density accumulation and the
    ``max(density, min_weight)`` division are the reference's, bit for bit where the order allows.

    The transpose is the only transform and it is not a tolerance: pytorch3d stores its volume as
    ``(minibatch, channels, D, H, W)`` and reads a point's ``(x, y, z)`` into ``(W, H, D)``, so its
    lattice is indexed ``[z, y, x]`` where ordito's is ``[x, y, z]``. Its local coordinates are
    the ``[-1, 1]`` cube with ``align_corners=True``, which is ordito's
    ``bounds=(-1, -1, -1), (1, 1, 1)`` exactly, so the coordinate map contributes nothing here --
    a fixture on a different box would need it and would be measuring the map rather than the
    splat.

    ``rescale_features=True`` is pytorch3d's default and is what makes both sides an *average*; at
    ``False`` its features would be the raw accumulation and ordito has no such switch.
    """
    resolution = 8
    rng = np.random.default_rng(4)
    points_np = (rng.random((300, 3)) * 1.6 - 0.8).astype(np.float32)
    values_np = rng.normal(size=(300, 3)).astype(np.float32)
    shape_p3d = (1, 3, resolution, resolution, resolution)
    features_p3d, densities_p3d = p3d_ops.add_points_features_to_volume_densities_features(
        points_to_torch(points_np, device),
        points_to_torch(values_np, device),
        torch.zeros((1, 1, resolution, resolution, resolution), device=device),
        torch.zeros(shape_p3d, device=device),
        mode="trilinear",
        min_weight=1e-4,
        align_corners=True,
    )
    field_wp, density_wp = od.voxels.splat_onto_grid(
        points_to_warp(points_np, device),
        points_to_warp(values_np, device),
        (resolution, resolution, resolution),
        bounds=(wp.vec3(-1.0, -1.0, -1.0), wp.vec3(1.0, 1.0, 1.0)),
    )

    assert features_p3d.shape == shape_p3d
    assert float(densities_p3d.sum()) == pytest.approx(points_np.shape[0], rel=1e-6)
    assert np.allclose(
        density_wp.numpy(),
        densities_p3d[0, 0].cpu().numpy().transpose(2, 1, 0),
        rtol=1e-5,
        atol=1e-6,
    )
    assert np.allclose(
        field_wp.numpy(), features_p3d[0].cpu().numpy().transpose(3, 2, 1, 0), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parity("sample_grid_trilinear", "pytorch3d")
def test_sample_grid_trilinear_matches_pytorch3d(device: str):
    """
    Class B: ``torch.nn.functional.grid_sample`` is the gather half, under the same axis transpose.

    The reference is torch's own trilinear sampler rather than a pytorch3d entry point, because
    that *is* what pytorch3d's volume sampling is -- ``Volumes`` hands ``grid_sample`` its
    ``[-1, 1]`` local coordinates -- and it is the independent implementation of the stencil this
    kernel writes. Two conventions and nothing else: the volume is ``(N, C, D, H, W)`` so a point's
    ``(x, y, z)`` goes in as ``(x, y, z)`` and indexes ``[z, y, x]``, and ``align_corners=True``
    puts ``-1`` on the first sample rather than on the cell edge, which is ordito's ``bounds``.

    ``padding_mode="border"`` is the one that matches: ordito clamps its base cell, so a query
    outside the lattice reads the boundary stencil rather than zero. Measured 4.77e-07 -- float32,
    not exact, because torch and Warp sum the eight corners in different orders.

    Also a **ordito-against-ordito** round trip, and the reference comparison above carries the
    oracle for it: points placed *on* the lattice corners make each stencil degenerate to its own
    corner, so a splat followed by a sample is the identity at **0.0** on any field.
    """
    resolution = 8
    rng = np.random.default_rng(6)
    field_np = rng.normal(size=(resolution, resolution, resolution)).astype(np.float32)
    queries_np = (rng.random((200, 3)) * 1.8 - 0.9).astype(np.float32)

    # (N, C, D, H, W) with [z, y, x] indexing against ordito's [x, y, z].
    volume_p3d = torch.as_tensor(field_np.transpose(2, 1, 0), device=device)[None, None]
    sampled_p3d = torch.nn.functional.grid_sample(
        volume_p3d,
        points_to_torch(queries_np, device).reshape(1, 1, 1, -1, 3),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(-1)
    field_wp = odt.as_array3d(wp.array(field_np, dtype=wp.float32, device=device), wp.float32)
    sampled_wp = od.voxels.sample_grid_trilinear(
        field_wp,
        points_to_warp(queries_np, device),
        bounds=(wp.vec3(-1.0, -1.0, -1.0), wp.vec3(1.0, 1.0, 1.0)),
    )

    assert sampled_p3d.shape == (queries_np.shape[0],)
    assert_nonconstant(sampled_p3d.cpu().numpy(), tol=1.0)
    assert np.allclose(sampled_wp.numpy(), sampled_p3d.cpu().numpy(), rtol=1e-5, atol=1e-6)

    # The exact round trip: on-lattice points, so each stencil is its own corner.
    lattice_np = np.stack(
        np.meshgrid(*[np.arange(resolution, dtype=np.float32)] * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    values_np = rng.normal(size=lattice_np.shape[0]).astype(np.float32)
    lattice_wp = points_to_warp(lattice_np, device)
    splatted_wp, density_wp = od.voxels.splat_onto_grid(
        lattice_wp,
        wp.array(values_np, dtype=wp.float32, device=device),
        (resolution, resolution, resolution),
    )
    round_trip_wp = od.voxels.sample_grid_trilinear(splatted_wp, lattice_wp)

    assert np.array_equal(density_wp.numpy(), np.ones((resolution,) * 3, dtype=np.float32))
    assert np.array_equal(round_trip_wp.numpy(), values_np)


def test_splat_onto_grid_invalid_arguments(device: str):
    """Not a parity assert: the guards ``splat_onto_grid`` and its inverse document."""
    points_wp = points_to_warp(np.zeros((4, 3), dtype=np.float32), device)
    values_wp = wp.zeros(4, dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="three positive integers"):
        od.voxels.splat_onto_grid(points_wp, values_wp, (4, 0, 4))
    with pytest.raises(ValueError, match="same length"):
        od.voxels.splat_onto_grid(
            points_wp, wp.zeros(3, dtype=wp.float32, device=device), (4, 4, 4)
        )
    with pytest.raises(ValueError, match="min_weight must be positive"):
        od.voxels.splat_onto_grid(points_wp, values_wp, (4, 4, 4), min_weight=0.0)
    with pytest.raises(ValueError, match="rank-3 lattice"):
        od.voxels.sample_grid_trilinear(values_wp, points_wp)  # pyright: ignore[reportArgumentType]  # deliberately rank 1

    # An empty cloud is a zero field and a zero density, not an error.
    empty_wp = points_to_warp(np.zeros((0, 3), dtype=np.float32), device)
    field_wp, density_wp = od.voxels.splat_onto_grid(
        empty_wp, wp.zeros(0, dtype=wp.float32, device=device), (4, 4, 4)
    )
    assert not field_wp.numpy().any()
    assert not density_wp.numpy().any()
    assert od.voxels.sample_grid_trilinear(field_wp, empty_wp).size == 0


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_splat_and_sample_handle_a_single_slice_axis(device: str, axis: int):
    """
    Not a library comparison: pytorch3d's volume admits no degenerate axis.

    There is nothing to compare against, so the oracle is the non-degenerate lattice beside it.

    A lattice with one sample on an axis is documented as legal ("each at least 1"), and the
    trilinear stencil still addresses two corners per axis. The base corner is clamped into
    ``[0, shape - 2]``, which is ``0`` here, so a stencil written ``base + 1`` steps one slice off
    the end at four of its eight corners. The weight there is exactly zero, so no *value* is ever
    wrong and only the address is out of range -- which is why this needs a bounds check to see it
    and why the asserts below cannot: run under ``wp.config.mode = "debug"`` to reproduce the
    original failure, or on the cpu device, where an out-of-bounds Warp write is heap corruption.

    The values are asserted against the same field on a two-slice lattice, which must agree
    exactly: a degenerate axis contributes weight 1 at its only corner either way.
    """
    rng = np.random.default_rng(11)
    shape = [3, 3, 3]
    shape[axis] = 1
    n_points = 32
    points_np = rng.uniform(0.0, 2.0, size=(n_points, 3)).astype(np.float32)
    # Every position collapses onto the single slice, so pin the coordinate there too: that is what
    # ``_lattice_transform`` does by mapping the degenerate axis's inverse spacing to zero.
    points_np[:, axis] = 0.0
    values_np = rng.normal(size=n_points).astype(np.float32)
    points_wp = points_to_warp(points_np, device)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)

    field_wp, density_wp = od.voxels.splat_onto_grid(
        points_wp, values_wp, cast("tuple[int, int, int]", tuple(shape))
    )
    assert field_wp.shape == tuple(shape)
    # The eight weights sum to one per point, so the density is the point count exactly -- the
    # invariant `splat_onto_grid`'s own Notes states, and it fails if a corner escaped the lattice.
    assert float(density_wp.numpy().sum()) == pytest.approx(float(n_points), rel=1e-5)

    sampled_wp = od.voxels.sample_grid_trilinear(field_wp, points_wp)
    assert np.all(np.isfinite(sampled_wp.numpy()))

    # The oracle: the same lattice with two slices on that axis. The degenerate axis's stencil
    # weight is 1 at slice 0 and 0 at slice 1, so slice 0 must match element for element.
    thick = list(shape)
    thick[axis] = 2
    thick_field_wp, thick_density_wp = od.voxels.splat_onto_grid(
        points_wp, values_wp, cast("tuple[int, int, int]", tuple(thick))
    )
    slice_0 = [slice(None)] * 3
    slice_0[axis] = slice(0, 1)
    assert np.allclose(field_wp.numpy(), thick_field_wp.numpy()[tuple(slice_0)], atol=1e-6)
    assert np.allclose(density_wp.numpy(), thick_density_wp.numpy()[tuple(slice_0)], atol=1e-6)
    assert np.allclose(
        sampled_wp.numpy(),
        od.voxels.sample_grid_trilinear(thick_field_wp, points_wp).numpy(),
        atol=1e-6,
    )


def test_grid_points_defaults_to_index_space(device: str):
    """``bounds=None`` gives lattice indices, matching ``levelset.marching_cubes``."""
    lattice = od.voxels.grid_points((2, 3, 4), device=device).numpy()
    expected = np.stack(np.meshgrid(*(np.arange(n) for n in (2, 3, 4)), indexing="ij"), -1)
    assert np.array_equal(lattice, expected.reshape(-1, 3).astype(np.float32))


# ---------------------------------------------------------------------------------------------
# Set algebra and resampling
# ---------------------------------------------------------------------------------------------


def _shifted(grid: wp.Volume, offset: tuple[int, int, int]) -> wp.Volume:
    """Move a voxel set by whole cells, so the two grids share a lattice but not a set."""
    voxel_size, origin = od.voxels.grid_transform(grid)
    rows = od.voxels.cells(grid).numpy() + np.array(offset, dtype=np.int32)
    return od.voxels.from_cells(
        wp.array(rows, dtype=wp.int32, device=grid.device), voxel_size, origin
    )


@pytest.mark.parity(
    "set_algebra",
    "meshlib",
    benchmarked=False,
    reason="MeshLib answers this over a dense VoxelBitSet addressed by a VolumeIndexer, so its "
    "cost is a bitwise fold over the whole box while ordito's is a rebuild of a sparse set; "
    "timing them together would report the density of the fixture rather than either algorithm. "
    "trimesh's ops.boolean_sparse is the other candidate row and needs the optional `sparse` "
    "package, which is not a test dependency here.",
)
def test_set_algebra_matches_meshlib(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """
    Class A on the dense occupancy: MeshLib's ``VoxelBitSet`` ``|``, ``&`` and ``-``.

    ``TypedBitSet`` derives from ``MR::BitSet``, so the three operators are the set algebra itself
    with no geometry in the way — which is exactly what makes it a clean oracle here: the only
    thing under test is whether ordito's sparse rebuild picks the same cells a bitmask would.

    The two grids are one voxel set and a copy of it moved three cells along ``x``, so all three
    answers are non-vacuous and distinct: the shift is smaller than the sphere, so the overlap is
    large, and it is non-zero, so neither difference is empty. Both sides see one padded box.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    left = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.16, mode="solid")
    right = _shifted(left, (3, 0, 0))
    lower, extent = _bounds_of(od.voxels.union(left, right))
    padded_lower = cast("tuple[int, int, int]", tuple(c - 1 for c in lower))
    padded_shape = cast("tuple[int, int, int]", tuple(n + 2 for n in extent))
    left_np = _dense(left, padded_lower, padded_shape)
    right_np = _dense(right, padded_lower, padded_shape)
    assert left_np.any()
    assert right_np.any()
    assert (left_np & right_np).any()  # non-vacuous: the two sets genuinely overlap
    assert (left_np & ~right_np).any()  # and genuinely differ

    for operation, expected_ml in (
        (od.voxels.union, _voxel_mask_ml(left_np)[0] | _voxel_mask_ml(right_np)[0]),
        (od.voxels.intersection, _voxel_mask_ml(left_np)[0] & _voxel_mask_ml(right_np)[0]),
        (od.voxels.difference, _voxel_mask_ml(left_np)[0] - _voxel_mask_ml(right_np)[0]),
    ):
        result_np = _dense(operation(left, right), padded_lower, padded_shape)
        assert np.array_equal(result_np, _dense_from_mask_ml(expected_ml, padded_shape))


def test_set_algebra_needs_one_lattice(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """Ordito against ordito: a cell coordinate is meaningless across two transforms."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    coarse = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.2)
    finer = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.16)
    for operation in (od.voxels.union, od.voxels.intersection, od.voxels.difference):
        with pytest.raises(ValueError, match="one lattice"):
            operation(coarse, finer)


@pytest.mark.parity(
    "revoxelize",
    "trimesh",
    benchmarked=False,
    reason="the operation is one occupancy probe per cell of the new lattice and is dominated by "
    "the dense allocation it fills, so a row would price to_dense and from_dense over again; both "
    "already carry the fill_cavities and fill_orthographic groups. trimesh's own revoxelized is "
    "additionally not a comparable row, for the sampling reason this test records.",
)
def test_revoxelize_matches_trimesh(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """
    Class B: ``trimesh.voxel.VoxelGrid.is_filled`` evaluated at the new cell centres.

    That is the membership test ``VoxelGrid.revoxelized`` runs internally, and the reason the
    comparison goes through it rather than through ``revoxelized`` itself is that ``revoxelized``
    samples on ``grid_linspace(self.bounds, shape)`` — ``shape`` points spanning the box
    *inclusive*, so a step of ``extents / (shape - 1)`` — and then attaches a transform whose scale
    is ``extents / shape``. The two disagree, and the visible consequence is that a completely
    filled 2x2x2 grid resampled to ``(8, 8, 8)`` comes back with 343 = 7^3 cells filled rather than
    512: the far face of samples lands exactly on the boundary and reads as empty. So there is no
    pitch a caller can ask ``revoxelized`` for that lands on the lattice ordito resamples onto.

    Three settings, so the claim is not a coarsening claim alone: unchanged (exact round trip),
    halved (refinement, which must be hole-free — this is why the rule is "the new centre lands in
    an old cell" rather than "voxelize the old centres") and tripled.

    Tripled rather than doubled on purpose: an **even** integer coarsening puts every new cell
    centre exactly on a face of the old lattice (``origin + (2c + 1) * old_size``), where ordito's
    float32 floor and trimesh's float64 round are free to disagree. Measured — at ``2.0`` this
    comparison fails on cells at the ties. Odd and fractional factors land strictly inside an old
    cell and the two agree exactly.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    grid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.16, mode="solid")
    voxel_size, origin = od.voxels.grid_transform(grid)
    lower, extent = _bounds_of(grid)
    voxel_tm = tm.voxel.VoxelGrid(
        _dense(grid, lower, extent),
        transform=tm.transformations.scale_and_translate(
            voxel_size,
            [float(origin[axis]) + (lower[axis] + 0.5) * voxel_size for axis in range(3)],
        ),
    )
    assert voxel_tm.filled_count == int(od.voxels.cells(grid).shape[0])

    for factor in (1.0, 0.5, 3.0):
        resampled = od.voxels.revoxelize(grid, factor * voxel_size)
        centers_np = od.voxels.cell_centers(resampled).numpy()
        assert len(centers_np) > 0
        assert np.array_equal(voxel_tm.is_filled(centers_np), np.ones(len(centers_np), bool))

        # And the other direction, which ``is_filled`` on the kept cells cannot see: no occupied
        # centre of the new lattice was missed. The lattice is regular, so its own resampling at
        # the same pitch enumerates it.
        every_center_np = od.voxels.cell_centers(
            od.voxels.revoxelize(od.voxels.dilate(resampled), factor * voxel_size)
        ).numpy()
        filled_tm = every_center_np[voxel_tm.is_filled(every_center_np)]
        assert np.array_equal(lexsort_rows(filled_tm), lexsort_rows(centers_np))


def test_revoxelize_round_trips_and_refines(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """
    Ordito against ordito: the oracle is ``test_revoxelize_matches_trimesh`` above.

    Two structural claims that comparison does not make. An unchanged ``voxel_size`` returns the
    *identical* lattice, which is what lets the result compose with the set algebra — the default
    origin is the input's own, not the occupied box's corner. And halving the pitch multiplies the
    voxel count by exactly eight, i.e. the refinement leaves no gaps between the old centres.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    grid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.16, mode="solid")
    voxel_size, origin = od.voxels.grid_transform(grid)
    n_voxels = int(od.voxels.cells(grid).shape[0])

    same = od.voxels.revoxelize(grid, voxel_size)
    assert od.voxels.grid_transform(same) == (voxel_size, origin)
    assert np.array_equal(
        lexsort_rows(od.voxels.cells(same).numpy()), lexsort_rows(od.voxels.cells(grid).numpy())
    )
    assert int(od.voxels.cells(od.voxels.union(grid, same)).shape[0]) == n_voxels

    assert (
        int(od.voxels.cells(od.voxels.revoxelize(grid, 0.5 * voxel_size)).shape[0]) == 8 * n_voxels
    )


def test_revoxelize_samples_only_the_occupied_box(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """
    Ordito against ordito: the oracle is ``test_revoxelize_matches_trimesh`` above.

    A grid whose voxels sit ten thousand cells from its own origin resamples to the same answer as
    one at the origin, and does not raise the budget guard. Anchoring the sampling lattice with
    ``from_dense``'s ``origin_cell`` rather than with the grid origin is what makes that true; the
    naive form allocates the whole box between the two and dies here.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    grid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.16, mode="solid")
    voxel_size, _origin = od.voxels.grid_transform(grid)
    offset = np.array([10000, 0, 0], dtype=np.int32)

    far = _shifted(grid, cast("tuple[int, int, int]", tuple(int(x) for x in offset)))
    resampled = od.voxels.revoxelize(far, voxel_size, max_cells=1 << 20)
    assert np.array_equal(
        lexsort_rows(od.voxels.cells(resampled).numpy()),
        lexsort_rows(od.voxels.cells(grid).numpy() + offset),
    )


def test_revoxelize_rejects_an_unaffordable_lattice(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """The budget guard raises before allocating, and names ``voxel_size`` like its sibling."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    grid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.2)
    with pytest.raises(ValueError, match="voxel_size"):
        od.voxels.revoxelize(grid, 1e-4, max_cells=1000)


# ---------------------------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------------------------


@pytest.mark.parity("dilate", "trimesh")
def test_dilate_matches_scipy(sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]]):
    """
    Class A: dense occupancy against ``scipy.ndimage.binary_dilation``.

    Its default structure is what trimesh's ``morphology.binary_dilation`` passes.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    grid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.16)
    lower, extent = _bounds_of(grid)
    occupancy_np = _dense(grid, lower, extent)
    assert occupancy_np.any()

    padded_lower = cast("tuple[int, int, int]", tuple(c - 1 for c in lower))
    padded_shape = cast("tuple[int, int, int]", tuple(n + 2 for n in extent))
    dilated = _dense(od.voxels.dilate(grid), padded_lower, padded_shape)
    assert np.array_equal(dilated, ndi.binary_dilation(np.pad(occupancy_np, 1)))


@pytest.mark.parametrize("connectivity", [6, 18, 26])
def test_erode_matches_scipy(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
    connectivity: Literal[6, 18, 26],
):
    """Class A: erosion against ``scipy.ndimage.binary_erosion`` at the matching structure rank."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    solid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.14, mode="solid")
    lower, extent = _bounds_of(solid)
    occupancy_np = _dense(solid, lower, extent)
    rank = {6: 1, 18: 2, 26: 3}[connectivity]

    eroded = _dense(od.voxels.erode(solid, connectivity=connectivity), lower, extent)
    reference = ndi.binary_erosion(
        occupancy_np, structure=ndi.generate_binary_structure(3, rank), border_value=0
    )
    assert reference.any()
    assert np.array_equal(eroded, reference)


def _voxel_mask_ml(occupancy_np: np.ndarray) -> tuple[mm.VoxelBitSet, mm.VolumeIndexer]:
    """
    Load a dense occupancy array into a ``VoxelBitSet`` plus the ``VolumeIndexer`` addressing it.

    Both MeshLab morphology calls take the pair and **mutate the bitset in place**, returning
    ``None``; the indexer is what carries the dimensions, and it fixes the bit order --
    ``VolumeIndexer::toPos`` decodes ``x + dims.x * y + dims.x * dims.y * z``, so ``x`` runs
    fastest and the dense array flattens with ``order="F"``.

    The load itself is one ``np.packbits`` through
    [`numpy_to_meshlib_bitset`][tests.conversions.numpy_to_meshlib_bitset]; ``VoxelBitSet``'s
    converting constructor then copies the bits and the size out of the untyped set.
    """
    dims_ml = mm.Vector3i(*(int(n) for n in occupancy_np.shape))
    indexer_ml = mm.VolumeIndexer(dims_ml)
    mask_ml = mm.VoxelBitSet(numpy_to_meshlib_bitset(occupancy_np.ravel(order="F")))
    return mask_ml, indexer_ml


def _dense_from_mask_ml(mask_ml: mm.VoxelBitSet, shape: tuple[int, int, int]) -> np.ndarray:
    """
    Read a ``VoxelBitSet`` back as a dense bool array of ``shape``.

    ``mn.getNumpyBitSet`` is declared over ``const MR::BitSet&`` and ``VoxelBitSet`` derives from
    it, so pybind11 upcasts and the whole set comes back in one call, flat in ``VoxelId`` order --
    the inverse of the ``order="F"`` flattening above, hence ``reshape(shape[::-1]).T``.
    """
    return meshlib_bitset_to_numpy(mask_ml, int(np.prod(shape))).reshape(shape[::-1]).T


@pytest.mark.parity("dilate", "meshlib")
@pytest.mark.parity(
    "erode",
    "meshlib",
    benchmarked=False,
    reason="erosion has no benchmark group of its own: it is dilation's complement over the same "
    "candidate buffer, and a second row would price the same pass under another name. The dilate "
    "group carries the timed MeshLib row, and test_erode_matches_scipy is the scipy oracle.",
)
def test_dilate_and_erode_match_meshlib(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """
    Class A on the occupancy: ``expandVoxelsMask`` / ``shrinkVoxelsMask`` are the 6-neighbour forms.

    MeshLib exposes no connectivity switch -- its neighbourhood is the six face-adjacent voxels,
    which is ordito's *default* and not its 18- or 26-connected settings, so the comparison is
    pinned at ``connectivity=6`` and the other two are excluded by construction rather than by
    tolerance. Verified on a 2x2x2 block: 8 voxels dilate to 32 under both, where ordito's
    26-connected setting gives 64.

    Two interface facts the helpers above encode: both calls **mutate the bitset and return
    ``None``**, and the bitset is addressed in ``VoxelId`` order (``x`` fastest) rather than in the
    dense array's C order. Neither direction needs a per-cell Python loop -- ``BitSet.fromBlocks``
    loads the packed ``uint64`` blocks and ``mn.getNumpyBitSet`` upcasts the ``VoxelBitSet`` back to
    a flat bool array -- which is what lets ``benchmarks/test_voxels.py`` time the morphology itself
    with the load hoisted into ``pedantic``'s untimed ``setup``.

    The dense box is padded by one cell on every side so a dilation has somewhere to go; both sides
    see the identical box, so the comparison is element-wise over the whole array.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    solid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.2, mode="solid")
    lower, extent = _bounds_of(solid)
    padded_lower = cast("tuple[int, int, int]", tuple(c - 1 for c in lower))
    padded_shape = cast("tuple[int, int, int]", tuple(n + 2 for n in extent))
    occupancy_np = _dense(solid, padded_lower, padded_shape)
    assert occupancy_np.any()

    dilated_np = _dense(od.voxels.dilate(solid), padded_lower, padded_shape)
    eroded_np = _dense(od.voxels.erode(solid), padded_lower, padded_shape)

    mask_ml, indexer_ml = _voxel_mask_ml(occupancy_np)
    assert mask_ml.count() == int(occupancy_np.sum())  # non-vacuity: the load round-trips
    assert mm.expandVoxelsMask(mask_ml, indexer_ml, 1) is None  # mutates, returns nothing
    assert np.array_equal(_dense_from_mask_ml(mask_ml, padded_shape), dilated_np)

    mask_ml, indexer_ml = _voxel_mask_ml(occupancy_np)
    mm.shrinkVoxelsMask(mask_ml, indexer_ml, 1)
    assert np.array_equal(_dense_from_mask_ml(mask_ml, padded_shape), eroded_np)
    assert eroded_np.sum() < occupancy_np.sum() < dilated_np.sum()


@pytest.mark.parity(
    "closing",
    "trimesh",
    benchmarked=False,
    reason="closing is dilate followed by erode and opening is the reverse, so a row would time "
    "the two passes the dilate group already times, under a third name; trimesh's own "
    "morphology.binary_closing is scipy.ndimage on a dense array, which the dilate group's "
    "reference row already prices at the same connectivity.",
)
@pytest.mark.parity(
    "opening",
    "scipy",
    benchmarked=False,
    reason="scipy.ndimage.binary_opening has no trimesh wrapper and no benchmark row for the same "
    "reason closing has none: it is erode then dilate, both of which the dilate group times "
    "already, and the dense reference measures the padded box rather than the voxel set.",
)
@pytest.mark.parametrize("connectivity", [6, 18, 26])
def test_closing_and_opening_match_scipy(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
    connectivity: Literal[6, 18, 26],
):
    """
    Class A: ``scipy.ndimage.binary_closing`` / ``binary_opening`` at the matching structure rank.

    The first is what trimesh's ``morphology.binary_closing`` wraps; the second trimesh does not
    expose, so scipy is the reference directly.

    The dense box is padded by two cells on every side, which is what makes this a comparison
    rather than a divergence: the sparse form dilates onto whatever cells it needs, where the dense
    reference's intermediate dilation is clipped at the array border and its erosion then eats a
    shell off every face. With the padding both sides see the same infinite lattice.

    Non-vacuity is asserted rather than assumed: the input is a *hollow* voxelization with a
    one-cell notch cut out of it, so the closing genuinely fills something and the opening
    genuinely removes something at every connectivity.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    shell = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.16)
    lower, extent = _bounds_of(shell)
    padded_lower = cast("tuple[int, int, int]", tuple(c - 2 for c in lower))
    padded_shape = cast("tuple[int, int, int]", tuple(n + 4 for n in extent))
    occupancy_np = _dense(shell, padded_lower, padded_shape)
    rank = {6: 1, 18: 2, 26: 3}[connectivity]
    structure_np = ndi.generate_binary_structure(3, rank)

    closed_np = _dense(
        od.voxels.closing(shell, connectivity=connectivity), padded_lower, padded_shape
    )
    opened_np = _dense(
        od.voxels.opening(shell, connectivity=connectivity), padded_lower, padded_shape
    )
    assert np.array_equal(closed_np, ndi.binary_closing(occupancy_np, structure=structure_np))
    assert np.array_equal(opened_np, ndi.binary_opening(occupancy_np, structure=structure_np))
    # A hollow shell is thin, so opening removes and closing adds: neither is the identity here.
    assert opened_np.sum() < occupancy_np.sum() < closed_np.sum()


def test_closing_and_opening_are_the_two_compositions(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """
    Ordito against ordito: the oracle is ``test_closing_and_opening_match_scipy`` above.

    The pair is defined by the *order* of the two passes, which is the one thing a comparison
    against a single reference call cannot catch — a closing that eroded first would still be a
    legal morphological filter and would still be idempotent. Both compositions are asserted
    explicitly, and so are the containments that separate them.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    shell = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.16)
    lower, extent = _bounds_of(shell)
    padded_lower = cast("tuple[int, int, int]", tuple(c - 2 for c in lower))
    padded_shape = cast("tuple[int, int, int]", tuple(n + 4 for n in extent))
    occupancy_np = _dense(shell, padded_lower, padded_shape)

    closed_np = _dense(od.voxels.closing(shell), padded_lower, padded_shape)
    opened_np = _dense(od.voxels.opening(shell), padded_lower, padded_shape)
    assert np.array_equal(
        closed_np, _dense(od.voxels.erode(od.voxels.dilate(shell)), padded_lower, padded_shape)
    )
    assert np.array_equal(
        opened_np, _dense(od.voxels.dilate(od.voxels.erode(shell)), padded_lower, padded_shape)
    )
    assert np.array_equal(closed_np & occupancy_np, occupancy_np)  # closing contains the input
    assert np.array_equal(opened_np & occupancy_np, opened_np)  # opening is contained in it


def test_surface_voxels_is_the_erosion_complement(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """``surface_voxels`` is exactly ``grid`` minus ``erode(grid)``, by definition of the shell."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    solid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.14, mode="solid")
    lower, extent = _bounds_of(solid)
    occupancy_np = _dense(solid, lower, extent)
    eroded = _dense(od.voxels.erode(solid), lower, extent)
    shell = _dense(od.voxels.surface_voxels(solid), lower, extent)
    assert shell.any()
    assert eroded.any()
    assert np.array_equal(shell, occupancy_np & ~eroded)


@pytest.mark.parity("fill_cavities", "trimesh")
def test_fill_cavities_matches_scipy(cave_cube: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: dense occupancy against ``scipy.ndimage.binary_fill_holes``.

    Run on ``cave_cube``, a box with a box-shaped void, so the fill is emphatically not a no-op --
    a fixture with nothing to fill would pass while testing nothing.
    """
    mesh_tm, mesh_wp = cave_cube
    grid = od.voxels.voxelize_mesh(mesh_wp.points, mesh_wp.indices, 0.02)
    lower, extent = _bounds_of(grid)
    occupancy_np = _dense(grid, lower, extent)

    filled = _dense(od.voxels.fill_cavities(grid), lower, extent)
    reference = ndi.binary_fill_holes(occupancy_np)
    assert reference.sum() > occupancy_np.sum()
    assert np.array_equal(filled, reference)
    assert mesh_tm.is_watertight


@pytest.mark.parity("fill_orthographic", "trimesh")
def test_fill_orthographic_matches_trimesh(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """Class A: dense occupancy against ``trimesh.voxel.ops.fill_orthographic``."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    grid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.16)
    lower, extent = _bounds_of(grid)
    occupancy_np = _dense(grid, lower, extent)

    filled = _dense(od.voxels.fill_orthographic(grid), lower, extent)
    reference = tm_voxel_ops.fill_orthographic(occupancy_np)
    assert reference.sum() > occupancy_np.sum()
    assert np.array_equal(filled, reference)


@pytest.mark.parametrize("fill", ["fill_cavities", "fill_orthographic"])
def test_fills_budget_the_box_they_densify(device: str, fill: str):
    """
    Not a library comparison: a budget guard has no counterpart in trimesh or scipy.

    Both densify unconditionally, because both are *handed* a dense array rather than a sparse grid.

    Both fills answer a question about the empty complement, so both densify the occupied bounding
    box. That box is set by how far apart the voxels are and has no relation to how many there are,
    so a two-voxel grid is unbounded -- at a separation of 800 cells it is 803**3 nodes, and the
    five lattices built over it come to 5.3 GiB. The sibling ``revoxelize`` already guards its own
    lattice for the same reason; these two did not.

    Asserted on a grid small enough to be harmless (two voxels, 128 apart) with the budget lowered
    to match, so the test never allocates what it is guarding against.
    """
    fill_fn = getattr(od.voxels, fill)
    cells_np = np.array([[0, 0, 0], [128, 128, 128]], dtype=np.int32)
    grid = od.voxels.from_cells(
        wp.array(cells_np, dtype=wp.int32, device=device), voxel_size=1.0, origin=wp.vec3()
    )
    assert int(grid.get_active_stats().voxel_count) == 2

    # The box is ~130**3 = 2.2e6 cells for a two-voxel grid: the guard is about the box, not the
    # count, and a budget above the *count* but below the *box* is exactly what must raise.
    with pytest.raises(ValueError, match="above max_cells"):
        fill_fn(grid, max_cells=1000)
    with pytest.raises(ValueError, match="max_cells must be positive"):
        fill_fn(grid, max_cells=0)

    # Generous budget: the same call runs, and a fill of two isolated voxels adds nothing.
    assert int(fill_fn(grid, max_cells=1 << 28).get_active_stats().voxel_count) == 2


# ---------------------------------------------------------------------------------------------
# resolve_voxel_grid
# ---------------------------------------------------------------------------------------------


def test_resolve_voxel_grid_defaults_are_one_percent_and_a_half_cell_below(device: str) -> None:
    """
    Class A: the package's single definition of an unspecified voxel grid, against the arithmetic.

    A cell of ``1 %`` of the bounding-box diagonal, anchored half a cell below the lower corner --
    Open3D's anchor, and the reason ``voxelize_points``, ``voxel_down_sample`` and
    ``cluster_decimate`` agree cell for cell. Both defaults resolve independently, so each is
    checked with the other supplied: a rule applied only when *both* are ``None`` would pass a test
    that always omits both.

    ``atol`` rather than exact: ``origin`` is a ``wp.vec3``, so the returned corner is ``float32``
    where the arithmetic here is ``float64`` (measured gap 7e-10).
    """
    points_np = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    points_wp = points_to_warp(points_np, device)
    diagonal = float(np.linalg.norm(points_np[1] - points_np[0]))

    voxel_size, origin = od.voxels.resolve_voxel_grid(points_wp)

    assert voxel_size == pytest.approx(0.01 * diagonal)
    assert np.allclose(list(origin), -0.5 * voxel_size, rtol=0, atol=1e-7)

    # Each default resolves on its own, not only when both are missing.
    size_only, given_origin = od.voxels.resolve_voxel_grid(points_wp, origin=wp.vec3(9.0, 9.0, 9.0))
    assert size_only == pytest.approx(0.01 * diagonal)
    assert list(given_origin) == [9.0, 9.0, 9.0]
    given_size, origin_only = od.voxels.resolve_voxel_grid(points_wp, 0.25)
    assert given_size == 0.25
    assert np.allclose(list(origin_only), -0.125, rtol=0, atol=1e-7)


def test_resolve_voxel_grid_passes_both_arguments_through(device: str) -> None:
    """With both supplied it reads no bounds at all, so it must return them unchanged."""
    points_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32), dtype=wp.vec3, device=device
    )

    voxel_size, origin = od.voxels.resolve_voxel_grid(points_wp, 0.25, wp.vec3(1.0, 2.0, 3.0))

    assert voxel_size == 0.25
    assert list(origin) == [1.0, 2.0, 3.0]


def test_resolve_voxel_grid_empty_input_takes_a_unit_diagonal(device: str) -> None:
    """An empty set has no bounding box, so the documented fallback is a diagonal of one."""
    voxel_size, origin = od.voxels.resolve_voxel_grid(warp_empty(0, wp.vec3, device))

    assert voxel_size == pytest.approx(0.01)
    assert np.allclose(list(origin), -0.005, rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    ("n_points", "voxel_size"), [(0, None), (1, None), (4000, None), (4000, 1e-3), (4000, 0.37)]
)
def test_resolve_voxel_grid_cell_bound_covers_every_cell(
    device: str, n_points: int, voxel_size: float | None
) -> None:
    """
    Not a library comparison: the bound is ordito's own hash radix, which no reference exposes.

    ``return_cell_bound`` hands a cell-row hash the ``max_index`` it would otherwise infer, so it
    must exceed every coordinate ``cell_indices`` writes -- by at most the documented two, so it
    stays a *tight* radix -- and packing against it with ``validate=False`` must group the cells
    exactly as the inferring, validating hash does. The cloud is offset and anisotropic so the
    three axes' extents differ, and a millimetre cell makes the counts large enough for float32
    rounding in the cell kernel to matter.
    """
    rng = np.random.default_rng(5)
    points_np = (rng.random((n_points, 3)) * [3.0, 1.0, 0.2] - [7.0, 0.5, 2.0]).astype(np.float32)
    points_wp = points_to_warp(points_np, device)

    size, origin, bound = od.voxels.resolve_voxel_grid(
        points_wp, voxel_size, return_cell_bound=True
    )
    assert (size, list(origin)) == (
        od.voxels.resolve_voxel_grid(points_wp, voxel_size)[0],
        list(od.voxels.resolve_voxel_grid(points_wp, voxel_size)[1]),
    )
    if n_points == 0:
        assert bound >= 1
        return
    cells = od.voxels.cell_indices(points_wp, size, origin=origin)
    cells_np = cells.numpy()
    assert cells_np.min() >= 0
    assert cells_np.max() < bound <= cells_np.max() + 3

    inferred = od.grouping.unique_1d(od.grouping.hash_indices_rows(cells), return_inverse=True)
    bounded = od.grouping.unique_1d(
        od.grouping.hash_indices_rows(cells, bound, validate=False), return_inverse=True
    )
    assert inferred[0].shape == bounded[0].shape
    assert np.array_equal(inferred[1].numpy(), bounded[1].numpy())


@pytest.mark.parametrize("n_points", [1, 5])
def test_resolve_voxel_grid_zero_extent_input_takes_a_unit_diagonal(
    device: str, n_points: int
) -> None:
    """
    Not a library comparison: this is ordito's own default convention, which no reference shares.

    A single point, or any all-coincident cloud, has a zero-extent bounding box and so no scale to
    read a cell width off. It takes the same unit diagonal the empty set takes -- the alternative
    was a derived ``0.01 * 0.0`` that tripped the positivity guard and reported
    ``requires voxel_size > 0, got 0.0``, naming the one argument the caller left as ``None``.

    Both arities matter: ``n_points=1`` is the obvious case and ``n_points=5`` is the one a check
    written as ``shape[0] == 1`` would miss, since coincidence rather than count is the condition.
    """
    points_wp = points_to_warp(np.full((n_points, 3), 2.5, dtype=np.float32), device)

    voxel_size, origin = od.voxels.resolve_voxel_grid(points_wp)

    assert voxel_size == pytest.approx(0.01)
    assert np.allclose(list(origin), 2.5 - 0.005, rtol=0, atol=1e-6)

    # The whole point is that the derived grid is usable, not merely that nothing raised.
    grid = od.voxels.voxelize_points(points_wp)
    assert int(grid.get_active_stats().voxel_count) == 1


def test_resolve_voxel_grid_rejects_a_non_positive_size_and_names_its_caller(device: str) -> None:
    """
    The ``caller`` argument exists so the message names the public entry point, not this helper.

    Both halves are asserted, because a message that silently ignored ``caller`` would still look
    right at the one call site that does not pass it.
    """
    points_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32), dtype=wp.vec3, device=device
    )

    with pytest.raises(ValueError, match="resolve_voxel_grid requires voxel_size > 0"):
        od.voxels.resolve_voxel_grid(points_wp, 0.0)
    with pytest.raises(ValueError, match="cluster_decimate requires voxel_size > 0"):
        od.voxels.resolve_voxel_grid(points_wp, -1.0, caller="cluster_decimate")


# ---------------------------------------------------------------------------------------------
# Conversion and meshing
# ---------------------------------------------------------------------------------------------


def test_dense_round_trip_with_negative_cells(device: str):
    """``from_dense(to_dense(g))`` is the identity, and ``origin_cell`` is what makes it one."""
    cells_np = np.array([[-3, -3, -3], [0, 1, 2], [-1, 4, -2], [5, -5, 5]], dtype=np.int32)
    grid = od.voxels.from_cells(
        wp.array(cells_np, dtype=wp.int32, device=device), 0.4, wp.vec3(-1.0, 2.0, 0.5)
    )
    occupancy, origin_cell = od.voxels.to_dense(grid)
    assert origin_cell == (-3, -5, -3)

    voxel_size, origin = od.voxels.grid_transform(grid)
    rebuilt = od.voxels.from_dense(occupancy, voxel_size, origin, origin_cell=origin_cell)
    assert np.array_equal(lexsort_rows(od.voxels.cells(rebuilt).numpy()), lexsort_rows(cells_np))


def test_to_field_round_trips_through_marching_cubes(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
    cave_cube: tuple[tm.Trimesh, wp.Mesh],
):
    """
    The ``bounds`` handoff, end to end and on two topologies.

    This is the mistake the module is most exposed to: the same tuple meaning two different things
    to producer and consumer, i.e. an ``(n-1)``-vs-``n`` spacing error or a C-order transposition.
    Either one moves the surface bodily, which the two-way surface-distance bound catches; the
    volume check pins the scale.

    ``cave_cube`` is voxelized in ``"surface"`` mode rather than ``"solid"``, because it is
    precisely a mesh with an enclosed cavity and ``"solid"`` fills that cavity by definition — its
    inner shell is what makes it worth testing here, so it has to survive.
    """
    mesh_tm, vertices_wp, faces_wp = sphere
    voxel_size = 0.1
    solid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, voxel_size, mode="solid")
    field, bounds = od.voxels.to_field(solid)
    vertices_out, faces_out = od.levelset.marching_cubes(field, 0.5, bounds=bounds)
    surface = tm.Trimesh(vertices_out.numpy(), faces_out.numpy().reshape(-1, 3), process=True)
    tm.repair.fix_normals(surface)

    n_voxels = int(od.voxels.cells(solid).shape[0])
    assert surface.is_watertight
    # The iso-0.5 crossing of a 0/1 field on cell centres lands on the cells' shared faces, so the
    # extracted volume is the occupied cell volume (a little under it, because marching cubes
    # bevels the convex corners). Against the *mesh* volume the bound would have to be 30 %;
    # against its own cells it is tight, which is what makes it a check on ``bounds``.
    assert surface.volume == pytest.approx(n_voxels * voxel_size**3, rel=0.12)
    _assert_surfaces_within(surface, mesh_tm, 2.0 * voxel_size)

    # The cavity fixture, surface mode, both shells present.
    mesh_tm, mesh_wp = cave_cube
    voxel_size = 0.03
    shell = od.voxels.voxelize_mesh(mesh_wp.points, mesh_wp.indices, voxel_size)
    field, bounds = od.voxels.to_field(shell)
    vertices_out, faces_out = od.levelset.marching_cubes(field, 0.5, bounds=bounds)
    shell_out = tm.Trimesh(vertices_out.numpy(), faces_out.numpy().reshape(-1, 3), process=True)
    assert shell_out.faces.shape[0] > 0
    _assert_surfaces_within(shell_out, mesh_tm, 2.0 * voxel_size)


def test_grid_points_round_trips_through_marching_cubes(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], device: str
):
    """Not a library comparison: the ``bounds`` handoff, sampling an SDF and re-extracting."""
    mesh_tm, mesh_wp = icosahedron
    lower, upper = od.bounds.aabb(mesh_wp.points)
    pad = 0.1 * float(wp.length(upper - lower))
    lower_floats, upper_floats = odt.vec3_floats(lower), odt.vec3_floats(upper)
    bounds = (
        wp.vec3(*(lower_floats[axis] - pad for axis in range(3))),
        wp.vec3(*(upper_floats[axis] + pad for axis in range(3))),
    )
    shape = (40, 40, 40)
    samples = od.voxels.grid_points(shape, bounds=bounds, device=device)
    distance = od.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, samples, sign_mode="winding"
    )
    vertices_out, faces_out = od.levelset.marching_cubes(
        odt.as_array3d(distance.reshape(shape), wp.float32), 0.0, bounds=bounds
    )
    surface = tm.Trimesh(vertices_out.numpy(), faces_out.numpy().reshape(-1, 3), process=True)
    tm.repair.fix_normals(surface)

    assert surface.is_watertight
    assert abs(surface.volume - mesh_tm.volume) / mesh_tm.volume < 0.05


@pytest.mark.parity("to_boxes", "trimesh", "pyvista")
def test_to_boxes_matches_trimesh_multibox(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """
    Class B: ``cull_internal=False`` against ``trimesh.voxel.ops.multibox`` and VTK's glyph filter.

    Not the triangle centroids: the two split each cube face into triangles along different
    diagonals, so those differ by a third of a cell while the *surface* is identical. And not
    ``lexsort`` on the corner positions either — that is the measured false negative on float rows
    with ties; a ``cKDTree`` bijection is used instead.

    pyvista's ``glyph(geom=pv.Cube(), scale=False, orient=False)`` is VTK's template instancing and
    belongs with ``multibox`` on the unwelded side -- **exactly 12 triangles per centre** after
    ``.triangulate()``, which is the same ``12 n`` and is asserted as an equality rather than a
    bound. Two of its arguments are load-bearing: left at their defaults ``scale`` and ``orient``
    read a scalar and a vector array off the cloud and would size and rotate each cube, and without
    ``.triangulate()`` the output is quads, so the face count is not comparable with either other
    side.
    """
    from scipy.spatial import cKDTree

    _mesh_tm, vertices_wp, faces_wp = sphere
    voxel_size = 0.2
    grid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, voxel_size)
    vertices_out, faces_out = od.voxels.to_boxes(grid, cull_internal=False)
    boxes_wp = warp_to_trimesh(vertices_out, faces_out)
    boxes_tm = tm_voxel_ops.multibox(od.voxels.cell_centers(grid).numpy(), pitch=voxel_size)

    assert boxes_tm.faces.shape[0] > 0
    assert boxes_wp.faces.shape[0] == boxes_tm.faces.shape[0]
    assert boxes_wp.area == pytest.approx(boxes_tm.area, rel=1e-5)

    # multibox translates a template cube per centre, so its shared corners are only equal to
    # within float rounding and ``np.unique`` will not collapse them. Set equality is therefore
    # stated as a two-way nearest-neighbour bound against ordito's exactly-shared corners.
    corners_wp = np.unique(boxes_wp.vertices[boxes_wp.faces.reshape(-1)], axis=0)
    corners_tm = boxes_tm.vertices[boxes_tm.faces.reshape(-1)]
    assert np.max(cKDTree(corners_wp).query(corners_tm)[0]) < 1e-5

    # pyvista instances the same template cube, and its corners land on the same lattice.
    centers_np = od.voxels.cell_centers(grid).numpy()
    cube_pv = pv.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
    boxes_pv = (
        pv.PolyData(np.ascontiguousarray(centers_np))
        .glyph(geom=cube_pv, scale=False, orient=False)
        .triangulate()
    )
    assert boxes_pv.n_cells == 12 * len(centers_np) == boxes_wp.faces.shape[0]
    corners_pv = np.asarray(boxes_pv.points)
    assert np.max(cKDTree(corners_wp).query(corners_pv)[0]) < 1e-5
    assert np.max(cKDTree(corners_tm).query(corners_wp)[0]) < 1e-5
    assert boxes_wp.volume == pytest.approx(boxes_tm.volume, rel=1e-5)


def test_to_boxes_culled_is_a_closed_outward_shell(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """The culled shell is watertight and its enclosed volume is the occupied cell volume."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    voxel_size = 0.2
    solid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, voxel_size, mode="solid")
    vertices_out, faces_out = od.voxels.to_boxes(solid)
    shell = warp_to_trimesh(vertices_out, faces_out)
    shell.remove_unreferenced_vertices()

    n_voxels = int(od.voxels.cells(solid).shape[0])
    assert shell.is_watertight
    # Positive volume is the winding claim: an inward-wound shell would report the negative.
    assert shell.volume == pytest.approx(n_voxels * voxel_size**3, rel=1e-5)


def _bbox_normalized(points_np: np.ndarray) -> np.ndarray:
    """Map a point set into the unit cube by its own bounding box, so two addressings compare."""
    points_np = np.asarray(points_np, dtype=np.float64)
    lower_np, upper_np = points_np.min(axis=0), points_np.max(axis=0)
    return (points_np - lower_np) / (upper_np - lower_np)


@pytest.mark.parity(
    "to_boxes",
    "pytorch3d",
    benchmarked=False,
    reason="the to_boxes group is pinned to cull_internal=False so that trimesh's multibox and "
    "VTK's glyph filter are like-for-like unwelded rows, and cubify culls and compacts -- timing "
    "it there would race a 33x smaller face buffer against ordito's all-faces row (23 136 "
    "triangles against 762 816 on a 64-cubed sphere). The like-for-like pair was measured anyway "
    "and is not the interesting kind of gap: cubify 2.238 ms against to_boxes(cull_internal=True) "
    "at 1.850 ms on 63 568 voxels, CUDA both sides.",
)
def test_to_boxes_matches_pytorch3d(device: str):
    """
    Class B: ``ops.cubify`` is ``to_boxes(cull_internal=True)`` under one affine coordinate map.

    Both cull the faces between two occupied voxels and both compact the interior corners away, so
    the counts agree **exactly** -- 74 vertices and 144 faces at resolution 6 -- and what remains
    is the addressing. ordito takes ``(voxel_size, origin)`` and pytorch3d normalizes into its own
    grid, so the comparison is on bounding-box-normalized coordinates, sorted (nothing pins two
    emission orders). Measured 5.96e-08.

    ``align`` is **not** one of the transforms, and that is worth recording because it looks like
    it should be: all three of ``"topleft"`` / ``"corner"`` / ``"center"`` return the *identical*
    face buffer and vertex count, differing only by a uniform scale and a translation -- bounding
    boxes ``[-0.6, 1.0]``, ``[-0.667, 0.667]`` and ``[-0.8, 0.8]`` on this fixture. So all three
    are already reachable through ``from_cells(cells, voxel_size, origin)`` and the normalization
    below absorbs the difference; asserting that here is what keeps the next reader from adding an
    ``align=`` keyword that would be a second spelling of ``voxel_size`` and ``origin``.

    The unreferenced-vertex assert is the other half: ``to_boxes`` used to return the whole corner
    lattice, 117 corners where 74 are referenced at this resolution and 4 370 680 where 190 640
    are at 4.3 M voxels, and that padding was what stood between the two answers.
    """
    resolution = 6
    axis_np = (np.arange(resolution) + 0.5) / resolution * 2.0 - 1.0
    x_np, y_np, z_np = np.meshgrid(axis_np, axis_np, axis_np, indexing="ij")
    occupancy_np = ((x_np**2 + y_np**2 + z_np**2) < 0.6).astype(np.float32)
    occupancy_p3d = torch.as_tensor(occupancy_np, device=device)[None]

    boxes_p3d = p3d_ops.cubify(occupancy_p3d, thresh=0.5, align="topleft")
    vertices_p3d, faces_p3d = pytorch3d_to_numpy(boxes_p3d)
    # All three alignments are one uniform scale plus a translation apart, so the normalization
    # below makes them the same answer -- there is nothing for an ``align=`` keyword to express.
    for align in ("corner", "center"):
        other_p3d, other_faces_p3d = pytorch3d_to_numpy(
            p3d_ops.cubify(occupancy_p3d, thresh=0.5, align=align)
        )
        assert np.array_equal(other_faces_p3d, faces_p3d)
        assert np.allclose(_bbox_normalized(other_p3d), _bbox_normalized(vertices_p3d), atol=1e-6)

    cells_np = np.ascontiguousarray(np.argwhere(occupancy_np > 0.5), dtype=np.int32)
    grid = od.voxels.from_cells(
        wp.array(cells_np, dtype=wp.int32, device=device), 1.0, wp.vec3(0.0, 0.0, 0.0)
    )
    vertices_wp, faces_wp = od.voxels.to_boxes(grid, cull_internal=True)
    vertices_np, faces_np = vertices_wp.numpy(), faces_wp.numpy().reshape(-1, 3)

    assert vertices_p3d.shape[0] > 0
    assert np.unique(faces_np).size == vertices_wp.size, "to_boxes left a corner unreferenced"
    assert vertices_wp.size == vertices_p3d.shape[0]
    assert faces_np.shape[0] == faces_p3d.shape[0]
    assert np.allclose(
        lexsort_rows(np.round(_bbox_normalized(vertices_np), 5)),
        lexsort_rows(np.round(_bbox_normalized(vertices_p3d), 5)),
        atol=1e-6,
    )


@pytest.mark.parity("voxel_corners", "igl")
def test_voxel_corners_matches_igl(
    sphere: tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]],
):
    """
    Class B: ``igl.unique_sparse_voxel_corners`` after two named transforms.

    It reads column 0 of a subscript as *y*, and it numbers a cell's corners in yxz
    binary-counting order.

    Its packing radix is ``2 ** depth + 1``, so ``depth`` must be large enough to hold the largest
    cell index; at the documented ``depth=0`` every subscript aliases and it returns a handful of
    corners for hundreds of cells.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    voxel_size = 0.2
    grid = od.voxels.voxelize_mesh(vertices_wp, faces_wp, voxel_size)
    _size, origin = od.voxels.grid_transform(grid)
    cells_np = od.voxels.cells(grid).numpy()
    corners_wp, cell_corners_wp = od.voxels.voxel_corners(grid)

    depth = int(np.ceil(np.log2(cells_np.max() + 2)))
    corners_igl, indices_igl, positions_igl = igl.unique_sparse_voxel_corners(
        np.asarray(origin, dtype=np.float64),
        voxel_size * (1 << depth),
        depth,
        cells_np[:, [1, 0, 2]].astype(np.int64),
    )
    corners_igl = np.asarray(corners_igl)[:, [1, 0, 2]].astype(np.int32)

    assert corners_igl.shape[0] > 0
    assert np.array_equal(lexsort_rows(corners_igl), lexsort_rows(corners_wp.numpy()))
    # Per-cell indices agree once igl's corner numbering is permuted onto ordito's.
    ours = corners_wp.numpy()[cell_corners_wp.numpy()]
    theirs = corners_igl[np.asarray(indices_igl)]
    assert np.array_equal(ours, theirs[:, _IGL_CORNER_ORDER, :])
    assert np.allclose(
        positions_igl, corners_igl * voxel_size + np.asarray(origin, dtype=np.float64), atol=1e-6
    )


# ---------------------------------------------------------------------------------------------
# Contracts: empty inputs, rebuildable grids, the lazy fem import
# ---------------------------------------------------------------------------------------------


def test_every_entry_point_survives_an_empty_input(device: str):
    """
    An empty input returns an empty result rather than raising.

    Warp 1.17 raises ``Failed to create volume`` on a zero-point build (1.15 aborted the process),
    so this is a contract every entry point has to hold up on its own.
    """
    no_points = wp.zeros(0, dtype=wp.vec3, device=device)
    no_faces = wp.zeros(0, dtype=wp.int32, device=device)
    empty_cells = odt.as_array2d(wp.zeros((0, 3), dtype=wp.int32, device=device), wp.int32)

    grids = [
        od.voxels.voxelize_points(no_points),
        od.voxels.voxelize_mesh(no_points, no_faces),
        od.voxels.from_cells(empty_cells, 1.0, wp.vec3(0.0, 0.0, 0.0)),
    ]
    for grid in grids:
        assert int(od.voxels.cells(grid).shape[0]) == 0
        assert od.voxels.cell_centers(grid).size == 0
        assert int(od.voxels.dilate(grid).get_active_stats().voxel_count) == 0
        assert int(od.voxels.erode(grid).get_active_stats().voxel_count) == 0
        assert int(od.voxels.surface_voxels(grid).get_active_stats().voxel_count) == 0
        assert int(od.voxels.fill_cavities(grid).get_active_stats().voxel_count) == 0
        assert int(od.voxels.fill_orthographic(grid).get_active_stats().voxel_count) == 0
        assert od.voxels.pool_by_voxel(grid, no_points, no_points).size == 0
        assert int(od.voxels.voxel_corners(grid)[0].shape[0]) == 0
        assert od.voxels.to_boxes(grid)[1].size == 0
    assert od.voxels.voxel_down_sample(no_points).size == 0


def test_cells_reports_the_active_length_of_a_rebuildable_grid(device: str):
    """
    A rebuildable grid reports its length through ``get_active_stats``, not ``get_voxel_count``.

    The trap that would otherwise produce a plausible wrong answer: ``get_voxel_count`` reports the
    reserved **capacity**, and the tail of ``get_voxels`` is ``[0, 0, 0]`` padding. Nothing in this
    module builds such a grid, but a caller can hand one in.
    """
    cells_np = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6], [-1, -2, -3]], dtype=np.int32)
    grid = wp.Volume.allocate_by_voxels(
        wp.array(cells_np, dtype=wp.int32, device=device),
        voxel_size=1.0,
        translation=(0.5, 0.5, 0.5),
        max_active_voxels=64,
        max_leaf_nodes=64,
        max_lower_nodes=64,
        max_upper_nodes=64,
        device=device,
    )
    assert grid.is_rebuildable
    grid.rebuild(wp.array(cells_np[:2], dtype=wp.int32, device=device))

    assert grid.get_voxel_count() > 2  # the capacity, padded with [0, 0, 0]
    rows = od.voxels.cells(grid).numpy()
    assert len(rows) == 2
    assert np.array_equal(lexsort_rows(rows), lexsort_rows(cells_np[:2]))


def test_voxels_imports_warp_fem_lazily():
    """
    ``warp.fem`` is imported inside [`voxel_corners`][ordito.voxels.voxel_corners], not at scope.

    That import is a real fixed cost every caller of the module would otherwise pay.

    Checked statically rather than through ``sys.modules``: ``ordito/kernels/curvature.py`` imports
    ``warp.fem.linalg`` at module scope, which drags the whole ``warp.fem`` package in regardless of
    what this module does, so a runtime probe could never fail.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(od.voxels))
    module_scope_imports = [
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    ] + [node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module]
    assert not any(name.startswith("warp.fem") for name in module_scope_imports)
    # ...and it really is imported somewhere, so this is not vacuous.
    assert "import warp.fem" in inspect.getsource(od.voxels)


def test_invalid_arguments_raise(device: str):
    """Each documented ``ValueError`` / ``TypeError`` is reachable."""
    points = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    empty_cells = odt.as_array2d(wp.zeros((0, 3), dtype=wp.int32, device=device), wp.int32)
    grid = od.voxels.voxelize_points(points, 1.0)

    with pytest.raises(ValueError, match="voxel_size"):
        od.voxels.voxelize_points(points, -1.0)
    with pytest.raises(ValueError, match="mode"):
        od.voxels.voxelize_mesh(points, wp.zeros(0, dtype=wp.int32, device=device), 1.0, mode="x")  # pyright: ignore[reportArgumentType]  # deliberately off-menu
    with pytest.raises(ValueError, match="order"):
        od.voxels.cells(grid, order="lexicographic")  # pyright: ignore[reportArgumentType]  # deliberately off-menu
    with pytest.raises(ValueError, match="pooling"):
        od.voxels.pool_by_voxel(grid, points, points, pooling="median")  # pyright: ignore[reportArgumentType]  # deliberately off-menu
    with pytest.raises(ValueError, match="connectivity"):
        od.voxels.dilate(grid, connectivity=7)  # pyright: ignore[reportArgumentType]  # deliberately off-menu
    with pytest.raises(ValueError, match="three columns"):
        od.voxels.from_cells(
            odt.as_array2d(wp.zeros((0, 2), dtype=wp.int32, device=device), wp.int32),
            1.0,
            wp.vec3(),
        )
    with pytest.raises(ValueError, match="voxel_size"):
        od.voxels.from_cells(empty_cells, 0.0, wp.vec3())
    with pytest.raises(ValueError, match="shape"):
        od.voxels.grid_points((0, 1, 1), device=device)
    with pytest.raises(ValueError, match="pad"):
        od.voxels.to_field(grid, pad=-1)
    with pytest.raises(TypeError, match="index"):
        od.voxels.cells(wp.Volume.load_from_numpy(np.ones((2, 2, 2), dtype=np.float32)))


def _assert_surfaces_within(surface_a: tm.Trimesh, surface_b: tm.Trimesh, tolerance: float) -> None:
    """Both one-sided surface distances between two meshes are under ``tolerance``."""
    samples_a = tm.sample.sample_surface(surface_a, 2000, seed=1)[0]
    samples_b = tm.sample.sample_surface(surface_b, 2000, seed=1)[0]
    assert tm.proximity.closest_point(surface_b, samples_a)[1].max() < tolerance
    assert tm.proximity.closest_point(surface_a, samples_b)[1].max() < tolerance


def _bounds_of(grid: wp.Volume) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Tight ``(lower_cell, extent)`` cell box of a grid, for the dense comparisons."""
    rows = od.voxels.cells(grid).numpy()
    lower = rows.min(axis=0)
    upper = rows.max(axis=0)
    extent = upper - lower + 1
    return (
        (int(lower[0]), int(lower[1]), int(lower[2])),
        (int(extent[0]), int(extent[1]), int(extent[2])),
    )
