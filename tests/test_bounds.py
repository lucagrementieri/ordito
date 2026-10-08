"""Regression tests for ``ordito.bounds`` against trimesh, open3d and libigl."""

from __future__ import annotations

from typing import Literal, cast

import igl
import numpy as np
import open3d as o3d
import pytest
import pyvista as pv
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm
from scipy.spatial import cKDTree

import ordito as od
import ordito.typing as odt
from tests.comparisons import lexsort_rows
from tests.conftest import MESHES
from tests.conversions import (
    numpy_to_warp,
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pyvista,
    warp_empty,
)

_Objective = Literal["volume", "surface_area", "diagonal"]
_OBJECTIVES: list[_Objective] = ["volume", "surface_area", "diagonal"]


def _bounds_np(lower_wp: wp.vec3, upper_wp: wp.vec3) -> np.ndarray:
    """Stack ordito's two corner ``wp.vec3`` into the ``(2, 3)`` layout every reference uses."""
    return np.stack(
        [
            np.array([lower_wp.x, lower_wp.y, lower_wp.z]),
            np.array([upper_wp.x, upper_wp.y, upper_wp.z]),
        ]
    )


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("aabb", "trimesh", "open3d", "igl", "meshlib", "pyvista")
@pytest.mark.parity("enclosing_diagonal", "pyvista", "igl")
def test_aabb_and_diagonal_match_the_references(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The axis-aligned bounding box against five references, and its diagonal against two.

    Box: Class A on trimesh, open3d, MeshLib and pyvista, Class B on igl. Trivial to compute and
    trivial to get subtly wrong -- a reduction that seeds its accumulator at zero rather than at
    +/-inf returns a box clamped to the origin, which is correct for any mesh straddling it and
    wrong for every mesh that does not. Every fixture here is translated away from the origin, so
    that bug would show.

    The benchmark's trimesh row is the uncached ``vstack((v.min(0), v.max(0)))`` formula behind
    ``Trimesh.bounds`` rather than the cached property, and that formula is what is compared here.

    **igl returns the box as geometry, not as two corners.** ``igl.bounding_box(V)`` gives the
    ``2**3 = 8`` corner *vertices* plus the 12 triangles of the box hull, so the named transform is
    to reduce those 8 corners back to a min/max pair. That is a genuinely different output shape for
    the same answer, and reducing it is exact.

    MeshLib's ``computeBoundingBox`` returns a ``Box3f`` -- ``.min`` / ``.max``, the two corners
    directly. Its ``region`` argument is passed ``None`` for the whole mesh; note that it takes the
    *topology* as well as the points, so on a mesh with unreferenced vertices it would box only the
    referenced ones, where ordito's takes the point buffer alone. Every fixture here references
    every vertex, so the two coincide.

    pyvista's ``DataSet.bounds`` needs one layout transform: it is **interleaved per axis** as
    ``(xmin, xmax, ymin, ymax, zmin, zmax)``, so it is reshaped to ``(3, 2)`` and transposed.
    Reading it as two corners without that step gives a plausible-looking box that is wrong on any
    mesh whose extents differ. **``DataSet.center`` is not a centroid** -- it is the bbox
    midpoint, asserted as such, so it belongs to this quantity and not to
    ``measures.surface_centroid``.

    Diagonal: Class A against ``DataSet.length`` and ``igl.bounding_box_diagonal``, both computing
    the box internally from one point set: ``enclosing_diagonal`` with no ``other`` is the
    single-cloud form. The fixtures span a cube-like solid and two thin open surfaces, so no single
    axis dominates the answer on all of them: an implementation returning the longest *extent*
    rather than the diagonal would match on none.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    lower_wp, upper_wp = od.bounds.aabb(mesh_wp.points)
    bounds_wp = _bounds_np(lower_wp, upper_wp)

    bounds_np = np.vstack((mesh_tm.vertices.min(axis=0), mesh_tm.vertices.max(axis=0)))
    assert np.allclose(bounds_wp, bounds_np, rtol=1e-5, atol=1e-5)

    box_o3d = trimesh_to_open3d(mesh_tm).get_axis_aligned_bounding_box()
    bounds_o3d = np.stack([box_o3d.get_min_bound(), box_o3d.get_max_bound()])
    assert np.allclose(bounds_wp, bounds_o3d, rtol=1e-5, atol=1e-5)

    corners_igl, faces_igl = map(
        np.asarray, igl.bounding_box(np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64))
    )
    assert corners_igl.shape == (8, 3), "the box comes back as its 8 corner vertices"
    assert faces_igl.shape == (12, 3), "and the 12 triangles of its hull"
    bounds_igl = np.stack([corners_igl.min(axis=0), corners_igl.max(axis=0)])
    assert np.allclose(bounds_wp, bounds_igl, rtol=1e-5, atol=1e-5)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    # MeshLib reads a ``None`` region as the whole mesh; its stub types the argument non-optional.
    box_ml = mm.computeBoundingBox(mesh_ml.topology, mesh_ml.points, None)  # pyright: ignore[reportArgumentType]
    bounds_ml = np.stack([[*box_ml.min], [*box_ml.max]])
    assert np.allclose(bounds_wp, bounds_ml, rtol=1e-5, atol=1e-5)

    mesh_pv = trimesh_to_pyvista(mesh_tm)
    bounds_pv = np.asarray(mesh_pv.bounds).reshape(3, 2).T
    assert np.allclose(bounds_wp, bounds_pv, rtol=1e-5, atol=1e-5)
    # The bbox midpoint, which is what pyvista's ``center`` is -- not the surface centroid.
    assert np.allclose(np.asarray(mesh_pv.center), bounds_pv.mean(axis=0), rtol=1e-5, atol=1e-5)

    diagonal_wp = od.bounds.enclosing_diagonal(mesh_wp.points)
    assert np.isclose(diagonal_wp, float(mesh_pv.length), rtol=1e-5, atol=1e-5)
    diagonal_igl = igl.bounding_box_diagonal(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    )
    assert np.isclose(diagonal_wp, diagonal_igl, rtol=1e-5, atol=1e-5)
    # Not merely the longest extent: on these fixtures the two differ by more than the tolerance.
    extents_np = mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)
    assert diagonal_wp > float(extents_np.max()) * (1.0 + 1e-3)


def test_aabb_single_point(device: str) -> None:
    """One point is a degenerate box: both corners land on it and the diagonal is zero."""
    points_wp = wp.array(
        np.array([[1.5, -2.5, 3.5]], dtype=np.float32), dtype=wp.vec3, device=device
    )
    lower_wp, upper_wp = od.bounds.aabb(points_wp)

    assert np.allclose(_bounds_np(lower_wp, upper_wp), [[1.5, -2.5, 3.5]] * 2, rtol=1e-6)
    assert od.bounds.enclosing_diagonal(points_wp) == 0.0


def test_aabb_union_encloses_both_boxes() -> None:
    """The union takes the component-wise min of the mins and max of the maxes, nothing more."""
    a_min, a_max = wp.vec3(-1.0, 0.0, 2.0), wp.vec3(1.0, 3.0, 4.0)
    b_min, b_max = wp.vec3(0.0, -5.0, 3.0), wp.vec3(2.0, 1.0, 3.5)

    union_min, union_max = od.bounds.aabb_union(a_min, a_max, b_min, b_max)

    assert np.allclose(odt.vec3_floats(union_min), [-1.0, -5.0, 2.0])
    assert np.allclose(odt.vec3_floats(union_max), [2.0, 3.0, 4.0])
    # Idempotent, and each input box is contained in the result.
    for box_min, box_max in ((a_min, a_max), (b_min, b_max)):
        again_min, again_max = od.bounds.aabb_union(union_min, union_max, box_min, box_max)
        assert np.allclose(odt.vec3_floats(again_min), [-1.0, -5.0, 2.0])
        assert np.allclose(odt.vec3_floats(again_max), [2.0, 3.0, 4.0])


def test_aabb_union_matches_a_pooled_reduction(device: str) -> None:
    """
    Unioning two clouds' boxes must equal the box of the concatenated cloud.

    That is the property callers rely on when they accumulate bounds incrementally, and the one
    thing a per-component slip would break while leaving both boxes individually plausible.
    """
    rng = np.random.default_rng(5)
    cloud_a = rng.normal(size=(200, 3)).astype(np.float32) + 4.0
    cloud_b = rng.normal(size=(300, 3)).astype(np.float32) - 2.0

    boxes = [od.bounds.aabb(points_to_warp(cloud, device)) for cloud in (cloud_a, cloud_b)]
    union_min, union_max = od.bounds.aabb_union(*boxes[0], *boxes[1])

    pooled_min, pooled_max = od.bounds.aabb(points_to_warp(np.vstack([cloud_a, cloud_b]), device))
    assert np.allclose(_bounds_np(union_min, union_max), _bounds_np(pooled_min, pooled_max))


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("enclosing_diagonal", "igl")
def test_enclosing_diagonal_matches_igl_on_the_pooled_cloud(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: equal to ``igl.bounding_box_diagonal`` of the two clouds *stacked*.

    The named transform is the stacking -- igl takes one point set, so the two-set signature maps
    onto it by concatenating, which is exactly the quantity ``enclosing_diagonal`` is defined as.
    Non-vacuous by construction: the queries are pushed outside the mesh's own box along every axis,
    so the union box is strictly larger than either input's and a one-sided implementation
    (measuring ``points`` alone, or ``other`` alone) fails.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    rng = np.random.default_rng(11)
    # Offset by the full extent so no query lands inside the mesh's box.
    extent_np = vertices_np.max(axis=0) - vertices_np.min(axis=0)
    queries_np = rng.random((64, 3)) * extent_np + vertices_np.max(axis=0)

    queries_wp = points_to_warp(queries_np, mesh_wp.device)
    diagonal_wp = od.bounds.enclosing_diagonal(mesh_wp.points, queries_wp)
    diagonal_igl = igl.bounding_box_diagonal(np.vstack([vertices_np, queries_np]))

    assert diagonal_igl > 0.0
    # Strictly larger than either side alone, or the union is not being taken.
    assert diagonal_wp > od.bounds.enclosing_diagonal(mesh_wp.points) + 1e-4
    assert diagonal_wp > od.bounds.enclosing_diagonal(queries_wp) + 1e-4
    assert np.allclose(diagonal_wp, diagonal_igl, rtol=1e-5, atol=1e-5)


def test_enclosing_diagonal_ignores_an_empty_second_set(device: str) -> None:
    """``other=None`` and ``other=<empty>`` both measure ``points`` alone."""
    rng = np.random.default_rng(12)
    cloud_wp = points_to_warp(rng.normal(size=(128, 3)), device)
    empty_wp = warp_empty(0, wp.vec3, device)

    alone = od.bounds.enclosing_diagonal(cloud_wp)
    assert alone == od.bounds.enclosing_diagonal(cloud_wp, None)
    assert alone == od.bounds.enclosing_diagonal(cloud_wp, empty_wp)


def _corner_box_np(lower_wp: wp.vec3, upper_wp: wp.vec3) -> np.ndarray:
    """
    Cut the low corner out of a box: 60% of each side, padded 1% outward on the low side.

    Every comparison below runs on this rather than on the fixture's own bounds, for two reasons
    that both had to be measured. It must select a **strict** subset, or the assertion has no
    teeth: against the full box every point is inside, so an implementation returning ``arange(n)``
    would pass. And it cannot be the box *shrunk about its centre*, which is the obvious
    construction and selects **nothing** on ``icosahedron`` and ``hemisphere`` -- every vertex of a
    convex polyhedron is at an extreme on some axis, so a window excluding all six extremes is
    empty and the comparison becomes ``[] == []``. Keeping the low corner leaves 3 / 9 / 14 / 35
    vertices of 12 / 16 / 97 / 544 across the four fixtures, checked here by an assert rather than
    by this sentence.

    The 1% outward pad is the float32 half of it: ordito tests the ``float32`` vertex buffer where
    the reference tests trimesh's ``float64`` one, so a plane placed exactly at a coordinate would
    let the two disagree at the boundary on rounding alone. Pushed off the data, the inclusive rule
    is pinned by ``test_points_in_aabb_boundary_and_non_finite_match_open3d`` instead, where both
    sides get exactly representable corners.
    """
    box_np = _bounds_np(lower_wp, upper_wp)
    extent_np = np.ptp(box_np, axis=0)
    return np.stack([box_np[0] - 0.01 * extent_np, box_np[0] + 0.6 * extent_np])


def _corners_wp(box_np: np.ndarray) -> tuple[wp.vec3, wp.vec3]:
    """Return a ``(2, 3)`` box as its two ``wp.vec3`` corners, for the ordito side."""
    return wp.vec3(*box_np[0].tolist()), wp.vec3(*box_np[1].tolist())


def _aabb_o3d(box_np: np.ndarray) -> o3d.geometry.AxisAlignedBoundingBox:
    """Open3D's axis-aligned box over the *same* two corners ordito is given."""
    return o3d.geometry.AxisAlignedBoundingBox(box_np[0], box_np[1])


def _obb_o3d(box_np: np.ndarray, frame_np: np.ndarray) -> o3d.geometry.OrientedBoundingBox:
    """
    Open3D's oriented box built **from** ordito's ``(rotation, min_bound, max_bound)`` triple.

    Constructing the reference's input from ordito's output is what leaves the *predicate* as the
    only thing under test: Open3D takes a world-space centre, a box-to-world rotation and an
    extent, where ordito takes a world-to-box frame and the extent in box coordinates, so the
    conversion is a transpose plus one corner rebuild. Had the box been searched independently on
    both sides, a disagreement could not be attributed to either half.
    """
    return o3d.geometry.OrientedBoundingBox(
        frame_np.T @ box_np.mean(axis=0), frame_np.T, np.ptp(box_np, axis=0)
    )


def _original_face_rows(
    sub_vertices_wp: wp.array[wp.vec3], sub_faces_wp: wp.array[wp.int32], vertices_np: np.ndarray
) -> np.ndarray:
    """
    Canonicalize a submesh's faces back into the *input's* vertex numbering.

    A crop compacts and renumbers its vertices, and two implementations are free to number them
    differently, so the face buffers are not comparable as returned. Matching each output vertex to
    the input vertex it came from puts both sides back in one numbering; sorting within each row and
    then lexsorting the rows drops the winding and the face order, neither of which a containment
    rule fixes. The row canonicalization is exact because it runs on integers -- the
    lexsort hazard is about float coordinates, and none survive to here.

    The match is nearest-neighbour with a bijection check rather than an equality, because a crop
    moves no vertex but ordito *stores* them in ``float32`` where the reference table is trimesh's
    ``float64``: measured 4.3e-07 at worst on these fixtures, against a vertex spacing of order
    0.05 on the finest of them, so the 1e-5 bound sits ~20x above the rounding and ~5 000x below
    the nearest wrong answer. The bijection is what makes the bound safe: two output vertices
    collapsing onto one input row would fail it even inside the tolerance.
    """
    sub_np = sub_vertices_wp.numpy()
    distance_np, original_np = cKDTree(vertices_np).query(sub_np)
    assert np.max(distance_np) < 1e-5, "a cropped vertex is not one of the input's"
    assert np.unique(original_np).size == original_np.size, "the vertex match is not 1:1"
    rows_np = np.asarray(original_np)[sub_faces_wp.numpy().reshape(-1, 3)]
    return lexsort_rows(np.sort(rows_np, axis=1))


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("points_in_aabb", "open3d")
@pytest.mark.parity("crop_points", "open3d")
def test_points_in_aabb_and_crop_points_match_open3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: the selected index list, element-wise against Open3D's, plus the mask/index round trip.

    Both sides are handed the identical two corners, so the only thing that can differ is the
    comparison. The mask assert is the second half of the same claim -- the index form is defined
    as [`flatnonzero`][ordito.array.flatnonzero] of the mask, and this pins that the two agree
    rather than trusting the composition.

    Class A for ``crop_points`` too: the cropped positions element-wise against
    ``PointCloud.crop``, and their indices. Both sides emit the survivors in ascending input order
    -- Open3D because every legacy selection routes through ``SelectByIndex``, which walks a mask
    -- so the positions compare row by row with no reordering. The returned indices are checked
    against the same box's index list rather than assumed: they are what lets a caller crop a
    parallel attribute, so a silently misaligned second return would be worse than none.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp = mesh_wp.points
    box_np = _corner_box_np(*od.bounds.aabb(vertices_wp))

    indices_o3d = np.sort(
        np.asarray(
            _aabb_o3d(box_np).get_point_indices_within_bounding_box(
                o3d.utility.Vector3dVector(mesh_tm.vertices)
            )
        )
    )
    # Non-vacuous on both sides: a full box or an empty one would make this test pass on nothing.
    assert 0 < indices_o3d.size < mesh_tm.vertices.shape[0]

    lower_wp, upper_wp = _corners_wp(box_np)
    indices_wp = od.bounds.points_in_aabb(vertices_wp, lower_wp, upper_wp)
    assert np.array_equal(indices_wp.numpy(), indices_o3d)

    mask_wp = od.bounds.points_in_aabb_mask(vertices_wp, lower_wp, upper_wp)
    assert np.array_equal(np.flatnonzero(mask_wp.numpy()), indices_o3d)

    cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh_tm.vertices))
    kept_o3d = np.asarray(cloud_o3d.crop(_aabb_o3d(box_np)).points)
    assert 0 < kept_o3d.shape[0] < mesh_tm.vertices.shape[0]
    kept_wp, crop_indices_wp = od.bounds.crop_points(vertices_wp, lower_wp, upper_wp)
    assert np.allclose(kept_wp.numpy(), kept_o3d, rtol=1e-5, atol=1e-5)
    assert np.array_equal(crop_indices_wp.numpy(), indices_o3d)


def test_points_in_aabb_boundary_and_non_finite_match_open3d(device: str) -> None:
    """
    Class A: the two conventions a random cloud cannot exercise -- the closed boundary and ``nan``.

    A containment rule's edge cases are exactly the points a random fixture never produces, and
    both of these were decided by measurement rather than by preference: the boundary is
    **inclusive** on Open3D's side (a point on a corner, an edge midpoint and a face centre are all
    selected) and a coordinate that is ``nan`` or infinite is **not**. The kernel's shorter
    vector spelling, ``wp.min(point, lower) == lower``, agrees on every finite row here and reports
    all three ``nan`` rows as inside, which is what this test would catch.
    """
    points_np = np.array(
        [
            [0.0, 0.0, 0.0],  # interior
            [-1.0, -1.0, -1.0],  # the min corner
            [1.0, 1.0, 1.0],  # the max corner
            [1.0, 0.0, 0.0],  # a face centre
            [1.0, 1.0, 0.0],  # an edge midpoint
            [1.0 + 1e-6, 0.0, 0.0],  # just outside one face
            [np.nan, 0.0, 0.0],
            [0.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.inf, 0.0, 0.0],
            [-np.inf, 0.0, 0.0],
        ]
    )
    box_np = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

    indices_o3d = np.sort(
        np.asarray(
            _aabb_o3d(box_np).get_point_indices_within_bounding_box(
                o3d.utility.Vector3dVector(points_np)
            )
        )
    )
    assert np.array_equal(indices_o3d, np.arange(5)), "the reference's own convention moved"

    indices_wp = od.bounds.points_in_aabb(points_to_warp(points_np, device), *_corners_wp(box_np))
    assert np.array_equal(indices_wp.numpy(), indices_o3d)


@pytest.mark.parametrize("mesh_name", MESHES)
def test_points_in_obb_matches_open3d(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: the same index-list equality through an oriented box, on a tilted cloud.

    The box comes from [`oriented_bounding_box`][ordito.bounds.oriented_bounding_box] and is
    handed to Open3D through ``_obb_o3d``, so ordito's ``(rotation, min_bound, max_bound)``
    convention is under test alongside the predicate: a transposed frame or a corner read in world
    coordinates instead of box coordinates selects a different set, not a differently-numbered one.
    The cloud is tilted so that the frame is far from the identity and the assert cannot pass
    through the axis-aligned path by accident.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np, points_wp = _tilted_cloud(mesh_tm, mesh_wp.device)

    rotation_wp, lower_wp, upper_wp = od.bounds.oriented_bounding_box(points_wp, 256)
    frame_np = _frame_np(rotation_wp)
    box_np = _corner_box_np(lower_wp, upper_wp)
    assert not np.allclose(frame_np, np.eye(3), atol=1e-3), "the frame is the identity"

    indices_o3d = np.sort(
        np.asarray(
            _obb_o3d(box_np, frame_np).get_point_indices_within_bounding_box(
                o3d.utility.Vector3dVector(points_np)
            )
        )
    )
    assert 0 < indices_o3d.size < points_np.shape[0]

    indices_wp = od.bounds.points_in_obb(points_wp, rotation_wp, *_corners_wp(box_np))
    assert np.array_equal(indices_wp.numpy(), indices_o3d)

    mask_wp = od.bounds.points_in_obb_mask(points_wp, rotation_wp, *_corners_wp(box_np))
    assert np.array_equal(np.flatnonzero(mask_wp.numpy()), indices_o3d)


def test_points_in_obb_with_the_identity_frame_is_the_aabb_form(device: str) -> None:
    """
    Ordito against ordito: the oriented query at ``rotation = I`` is the axis-aligned one.

    Not a parity assert -- the two entry points share one predicate, and the Open3D comparisons
    above carry the oracle for both. What this pins is that the *rotation* is applied as a
    world-to-box map and not as its inverse, which the identity is precisely the frame that cannot
    show: it runs on the non-finite and boundary rows too, where the two forms could otherwise
    diverge without any random point noticing.
    """
    points_np = np.array(
        [[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]
    )
    points_wp = points_to_warp(points_np, device)
    lower_wp, upper_wp = _corners_wp(np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    identity_wp = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    aligned_np = od.bounds.points_in_aabb(points_wp, lower_wp, upper_wp).numpy()
    assert np.array_equal(aligned_np, np.arange(3)), "the axis-aligned answer moved"
    assert np.array_equal(
        od.bounds.points_in_obb(points_wp, identity_wp, lower_wp, upper_wp).numpy(), aligned_np
    )


@pytest.mark.parametrize("oriented", [False, True])
def test_index_crop_and_mask_forms_agree(device: str, oriented: bool) -> None:
    """
    Ordito against ordito: the index and crop forms select exactly the mask form's points.

    Not a parity assert -- the Open3D comparisons above carry the oracle. The index and crop forms
    test containment in a flag kernel of their own rather than compacting the mask, so this pins
    the two spellings of the predicate to each other on the rows a random cloud never produces:
    ``nan`` and infinite coordinates, and the closed boundary.
    """
    rng = np.random.default_rng(23)
    points_np = rng.uniform(-2.0, 2.0, size=(4000, 3))
    points_np[:6] = [
        [np.nan, 0.0, 0.0],
        [0.0, np.inf, 0.0],
        [-np.inf, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
    points_wp = points_to_warp(points_np, device)
    lower_wp, upper_wp = _corners_wp(np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    angle = 0.3
    rotation_wp = wp.mat33(
        np.cos(angle), -np.sin(angle), 0.0, np.sin(angle), np.cos(angle), 0.0, 0.0, 0.0, 1.0
    )
    if oriented:
        mask_np = od.bounds.points_in_obb_mask(points_wp, rotation_wp, lower_wp, upper_wp).numpy()
        indices_wp = od.bounds.points_in_obb(points_wp, rotation_wp, lower_wp, upper_wp)
        kept_wp, crop_indices_wp = od.bounds.crop_points(
            points_wp, lower_wp, upper_wp, rotation=rotation_wp
        )
    else:
        mask_np = od.bounds.points_in_aabb_mask(points_wp, lower_wp, upper_wp).numpy()
        indices_wp = od.bounds.points_in_aabb(points_wp, lower_wp, upper_wp)
        kept_wp, crop_indices_wp = od.bounds.crop_points(points_wp, lower_wp, upper_wp)
    expected_np = np.flatnonzero(mask_np)
    assert 0 < expected_np.size < points_np.shape[0]
    assert not mask_np[:3].any(), "a non-finite row was selected"
    assert np.array_equal(indices_wp.numpy(), expected_np)
    assert np.array_equal(crop_indices_wp.numpy(), expected_np)
    assert np.array_equal(kept_wp.numpy(), points_np.astype(np.float32)[expected_np])


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("crop_mesh", "open3d")
def test_crop_mesh_matches_open3d(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: the kept triangles against ``TriangleMesh.crop``, remapped into one vertex numbering.

    The transform is ``_original_face_rows``: both crops renumber their compacted vertices in their
    own order, so the face buffers are pushed back through the input's numbering before being
    compared as sets. The vertex and face *counts* are class A on top of that, and the invariant --
    every surviving vertex inside the box -- is what a shared misreading of the containment rule
    could not satisfy, since it is checked against the box rather than against the reference.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    box_np = _corner_box_np(*od.bounds.aabb(vertices_wp))

    mesh_o3d = trimesh_to_open3d(mesh_tm).crop(_aabb_o3d(box_np))
    faces_o3d = np.asarray(mesh_o3d.triangles)
    assert 0 < faces_o3d.shape[0] < mesh_tm.faces.shape[0]

    sub_vertices_wp, sub_faces_wp = od.bounds.crop_mesh(vertices_wp, faces_wp, *_corners_wp(box_np))
    assert sub_faces_wp.size // 3 == faces_o3d.shape[0]
    assert sub_vertices_wp.size == np.asarray(mesh_o3d.vertices).shape[0]
    assert np.array_equal(
        _original_face_rows(sub_vertices_wp, sub_faces_wp, mesh_tm.vertices),
        _original_face_rows(
            points_to_warp(np.asarray(mesh_o3d.vertices), mesh_wp.device),
            wp.array(
                np.ascontiguousarray(faces_o3d.ravel(), dtype=np.int32),
                dtype=wp.int32,
                device=mesh_wp.device,
            ),
            mesh_tm.vertices,
        ),
    )

    inside_np = od.bounds.points_in_aabb_mask(sub_vertices_wp, *_corners_wp(box_np)).numpy()
    assert inside_np.all(), "a cropped vertex lies outside the box"


def test_crop_mesh_drops_the_faces_that_straddle_the_box(device: str) -> None:
    """
    Not a library comparison: the all-corners rule, on a mesh built so that one face straddles.

    The reference comparison above cannot isolate this -- both libraries apply the same rule, so a
    shared misreading would agree -- and it is the whole difference between a crop and a boolean:
    the straddling triangle is **dropped**, leaving a ragged edge, rather than clipped into new
    geometry. The counts are asserted exactly because the mesh is small enough to enumerate: two
    triangles fully inside, one straddling, one fully outside.
    """
    vertices_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],  # the four inside the box
            [5.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
            [6.0, 0.0, 0.0],  # three well outside it
        ]
    )
    faces_np = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 6]], dtype=np.int32)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np.ravel(), device)
    lower_wp, upper_wp = _corners_wp(np.array([[-0.5, -0.5, -0.5], [1.5, 1.5, 0.5]]))

    sub_vertices_wp, sub_faces_wp = od.bounds.crop_mesh(vertices_wp, faces_wp, lower_wp, upper_wp)
    assert sub_faces_wp.size // 3 == 2
    assert sub_vertices_wp.size == 4

    # ``face_mode="any"`` is the documented escape hatch, and it keeps the straddling one.
    mask_wp = od.bounds.points_in_aabb_mask(vertices_wp, lower_wp, upper_wp)
    _, any_faces_wp = od.selection.submesh_from_vertex_mask(
        vertices_wp, faces_wp, mask_wp, face_mode="any"
    )
    assert any_faces_wp.size // 3 == 3


def test_points_in_aabb_empty_cloud_and_empty_box(device: str) -> None:
    """An empty cloud and an inverted box both select nothing, through all four entry points."""
    rng = np.random.default_rng(21)
    cloud_wp = points_to_warp(rng.normal(size=(64, 3)), device)
    empty_wp = warp_empty(0, wp.vec3, device)
    lower_wp, upper_wp = wp.vec3(-1.0, -1.0, -1.0), wp.vec3(1.0, 1.0, 1.0)
    identity_wp = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    assert od.bounds.points_in_aabb(empty_wp, lower_wp, upper_wp).size == 0
    assert od.bounds.points_in_aabb_mask(empty_wp, lower_wp, upper_wp).size == 0
    assert od.bounds.points_in_obb(empty_wp, identity_wp, lower_wp, upper_wp).size == 0
    assert od.bounds.crop_points(empty_wp, lower_wp, upper_wp)[0].size == 0

    # An inverted box is empty rather than universal: no point satisfies both comparisons.
    assert od.bounds.points_in_aabb(cloud_wp, upper_wp, lower_wp).size == 0
    assert od.bounds.points_in_obb(cloud_wp, identity_wp, upper_wp, lower_wp).size == 0


def _tilted_cloud(
    mesh_tm: tm.Trimesh, device: wp.DeviceLike
) -> tuple[np.ndarray, wp.array[wp.vec3]]:
    """
    Stretch a fixture's vertices anisotropically and rotate them off the coordinate axes.

    Every oriented-box comparison below runs on this rather than on the raw fixture, and it is not
    cosmetic: on a shape whose extent is the same in every direction, *all* candidate orientations
    score nearly the same, so an agreement on the achieved objective would hold for an
    implementation that returned any orientation at all. Stretching by ``(1, 2, 3)`` makes the
    minimizer both unique and far from the identity, which is what gives the assertions teeth --
    and the rotation is what stops the axis-aligned box from already being the answer.
    """
    tilt_np = tm.transformations.random_rotation_matrix(np.random.default_rng(7).random(3))[:3, :3]
    points_np = np.ascontiguousarray(mesh_tm.vertices * np.array([1.0, 2.0, 3.0]) @ tilt_np.T)
    points_wp = points_to_warp(points_np, device)
    return points_np, points_wp


def _frame_np(rotation_wp: wp.mat33) -> np.ndarray:
    """Return ordito's world-to-box frame as ``(3, 3)`` NumPy, its rows still the box axes."""
    return np.array([[rotation_wp[i, j] for j in range(3)] for i in range(3)])


def _achieved_loss(
    points_np: np.ndarray, frame_np: np.ndarray, objective: _Objective = "volume"
) -> float:
    """
    Recompute, in NumPy, the objective a world-to-box frame achieves on ``points_np``.

    Both libraries are scored through this, from the matrix each one returns, so the comparison is
    of the *boxes* rather than of two self-reported numbers.
    """
    sides_np = np.ptp(points_np @ frame_np.T, axis=0)
    if objective == "volume":
        return float(np.prod(sides_np))
    if objective == "surface_area":
        return float(2.0 * (sides_np * np.roll(sides_np, 1)).sum())
    return float(np.square(sides_np).sum())


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parametrize("objective", _OBJECTIVES)
@pytest.mark.parity("oriented_bounding_box", "igl")
def test_oriented_bounding_box_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str, objective: _Objective
) -> None:
    """
    Class A on the achieved objective, for all three of them: the two searches find the same box.

    This is a *search*, and the comparison is only meaningful because both libraries search the
    identical candidate set -- the Super-Fibonacci spiral of [Alexa 2022] with the identity
    appended, which ordito adopts precisely so that the two are comparable. So this is not "two
    heuristics landed near each other": with the same candidates and the same three objectives the
    argmin is the same element, and the assertion is tight rather than statistical (measured
    agreement 1e-7 relative, asserted at 1e-5).

    Both sides are scored by ``_achieved_loss`` from the matrix they return, not by trusting the
    number each reports, so a library that returned a good loss with the wrong frame would fail.

    ``refine_iterations=0``: igl has no refinement, and the identical-candidate-set argument is
    only about the sampled phase -- refined, ordito is strictly better than this comparison.

    The ``volume`` arm (igl's default objective) is also Class B on the frame itself: the same
    matrix as igl, element-wise, once transposed for igl's row-vector convention. That pins the one
    thing the objective comparison cannot -- which of the two transpose conventions ordito returns.
    Reading it the wrong way round still gives an orthonormal matrix and a plausible box, so nothing
    else here would catch it. Element-wise equality of the *frames* is only well posed because the
    minimizer is unique on a stretched, tilted cloud; that is why this does not run on the raw
    fixtures. Every arm checks the frame is a proper rotation, not a reflection.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np, points_wp = _tilted_cloud(mesh_tm, mesh_wp.device)
    igl_objective = {
        "volume": igl.ORIENTED_BOUNDING_BOX_MINIMIZE_VOLUME,
        "surface_area": igl.ORIENTED_BOUNDING_BOX_MINIMIZE_SURFACE_AREA,
        "diagonal": igl.ORIENTED_BOUNDING_BOX_MINIMIZE_DIAGONAL_LENGTH,
    }[objective]

    rotation_wp, lower_wp, upper_wp = od.bounds.oriented_bounding_box(
        points_wp, 512, objective, refine_iterations=0
    )

    # igl applies its matrix on the right of a row vector, so its frame is ordito's transposed.
    frame_igl = np.asarray(igl.oriented_bounding_box(points_np, 512, igl_objective)).T
    frame_wp = _frame_np(rotation_wp)
    loss_wp = _achieved_loss(points_np, frame_wp, objective)
    loss_igl = _achieved_loss(points_np, frame_igl, objective)
    assert np.isclose(loss_wp, loss_igl, rtol=1e-5)
    if objective == "volume":
        assert np.allclose(frame_wp, frame_igl, atol=1e-5)
    # A proper rotation, not a reflection: the box axes are right-handed.
    assert np.allclose(frame_wp @ frame_wp.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(frame_wp), 1.0, atol=1e-5)

    # The search is worth running on this fixture: the identity is one of the 512 candidates, so a
    # tie with the axis-aligned box would mean the other 511 never improved on it.
    assert loss_wp < _achieved_loss(points_np, np.eye(3), objective) * 0.99

    # And ordito's own bounds are the extent in its own frame -- the convention the docstring
    # states, and the only part of the answer igl does not return.
    projected_np = points_np @ frame_wp.T
    assert np.allclose(
        _bounds_np(lower_wp, upper_wp),
        np.stack([projected_np.min(0), projected_np.max(0)]),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("mesh_name", "compare_open3d"), [("sphere_irregular", True), ("saddle_graded", False)]
)
@pytest.mark.parity("oriented_bounding_box", "trimesh", "pyvista", "open3d")
def test_oriented_bounding_box_volume_bands(
    request: pytest.FixtureRequest, mesh_name: str, compare_open3d: bool
) -> None:
    """
    Class C: one refined 32 768-candidate box against three other boxes, by volume.

    A derived scalar -- box volume -- because no correspondence exists between the frames: the
    libraries minimize (or, for pyvista, do not minimize) over *different* candidate sets, and
    different orientations can realise the same volume. The bug class excluded is a search that
    does not search -- a mis-scored objective, candidates that fail to cover ``SO(3)``, or a frame
    paired with extents it did not produce -- all of which leave the volume far above a real
    minimum. The reported extents are checked to be the box the reported frame achieves, so the
    volume is the quantity that was minimized and not an unrelated pair of numbers. Mutation probe,
    asserted against pyvista below: the axis-aligned box (``rotations=1, refine_iterations=0``, what
    a search that silently scored nothing would return) reads 1.58x to 5.37x the trimesh and open3d
    volumes and 1.65x to 5.37x pyvista's, breaking every ``1.02`` ceiling on every fixture.

    **trimesh** searches the orientations flush with a convex-hull face. **Neither side bounds the
    other, and that was measured rather than assumed.** trimesh's answer reads like an exact minimum
    and is not one: refined, ordito returns **3.4% less** volume on the tilted half torus and 2.1%
    less on the icosahedron, so the hull-face restriction can miss the optimum -- hence a two-sided
    band. ``volume_wp / volume_tm`` runs from 0.966 (half_torus) through 0.979 (icosahedron) and
    1.0001 (hemisphere) to 1.0005 (cave_cube -- refinement recovers the cube's exact orientation to
    0.05%), so the ``[0.88, 1.02]`` band clears the worst reading by 3.5x below and 40x above.

    **open3d**'s ``get_minimal_oriented_bounding_box`` is a third independent hull-based minimizer.
    Not ``get_oriented_bounding_box``: that one is PCA of the hull and minimizes nothing (measured
    12.9% above ordito on the tilted half_torus, and *exact* on cave_cube where axis-snapping is
    what PCA happens to do). Neither side bounds the other: ``volume_wp / volume_o3d`` runs 0.979
    (icosahedron) through 1.0000 (half_torus and hemisphere, ties to 4 digits) to 1.0005
    (cave_cube). The ``[0.90, 1.02]`` band clears the worst reading by 4.7x below and 40x above.

    **pyvista** -- VTK's ``oriented_bounding_box`` -- is PCA of the **points**, so it minimizes
    nothing and is the one reference ordito's search should *dominate*: a search over ``SO(3)``
    cannot lose to a single fixed orientation by more than its own sampling error. Measured
    ``volume_wp / volume_pv``: **1.00001 (icosahedron), 1.00015 (cave_cube), 0.9428 (hemisphere),
    0.8353 (half_torus)** -- up to 17% tighter and never more than 0.02% looser, so the ``1.02``
    ceiling clears the worst reading by 130x on that side. The ``0.75`` floor is a sanity bound
    rather than a tight one: PCA is not arbitrarily bad on these shapes.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np, points_wp = _tilted_cloud(mesh_tm, mesh_wp.device)

    rotation_wp, lower_wp, upper_wp = od.bounds.oriented_bounding_box(points_wp, 32768)
    volume_wp = float(np.prod(np.ptp(_bounds_np(lower_wp, upper_wp), axis=0)))
    assert np.isclose(
        volume_wp, _achieved_loss(points_np, _frame_np(rotation_wp), "volume"), rtol=1e-5
    )

    _, extents_tm = tm.bounds.oriented_bounds(tm.PointCloud(points_np))
    volume_tm = float(np.prod(extents_tm))
    assert volume_tm > 0.0, "the reference produced a box before it is compared to"
    assert volume_tm * 0.88 <= volume_wp <= volume_tm * 1.02

    # open3d's minimal box collapses on ``saddle_graded``: extent ``[0, 0, 0]`` (``robust=True``
    # too, and on the cloud's convex-hull vertices alone), where trimesh and pyvista both return
    # 13.439 and ordito 13.439. A reference failure, so that fixture keeps the other two.
    if compare_open3d:
        cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np))
        volume_o3d = cloud_o3d.get_minimal_oriented_bounding_box().volume()
        assert volume_o3d > 0.0, "the reference produced a box before it is compared to"
        assert volume_o3d * 0.90 <= volume_wp <= volume_o3d * 1.02

    box_pv = cast("pv.PolyData", pv.PolyData(points_np).oriented_bounding_box(as_composite=False))
    volume_pv = float(box_pv.volume)
    assert volume_pv > 0.0, "the reference produced a box before it is compared to"
    assert volume_pv * 0.75 <= volume_wp <= volume_pv * 1.02

    # The axis-aligned box is what a search that scored nothing would return, and it fails.
    _rotation_np, lower_np, upper_np = od.bounds.oriented_bounding_box(
        points_wp, 1, refine_iterations=0
    )
    assert float(np.prod(np.ptp(_bounds_np(lower_np, upper_np), axis=0))) > volume_pv * 1.02


def test_oriented_bounding_box_prefilter_returns_the_identical_box(device: str) -> None:
    """
    Above ``CONVEX_PREFILTER_MIN_POINTS`` the convex-superset prefilter must change nothing.

    The dense cloud (prefiltered internally) and its explicit hull-candidate subset (below the
    threshold, so searched directly) score the same support points, and min/max extents are
    order-independent, so the two boxes must agree to float32 exactness — not merely in volume.
    """
    rng = np.random.default_rng(23)
    n = od.bounds.CONVEX_PREFILTER_MIN_POINTS + 10_000
    cloud_np = (rng.standard_normal((n, 3)) @ np.diag([3.0, 1.0, 0.5])).astype(np.float32)
    cloud_wp = points_to_warp(cloud_np, device)

    kept_wp = od.array.gather(
        cloud_wp, od.array.flatnonzero(od.points.convex_superset_mask(cloud_wp))
    )
    assert kept_wp.size < n // 10, "the prefilter must actually discard interior points"

    rotation_dense, lower_dense, upper_dense = od.bounds.oriented_bounding_box(cloud_wp)
    rotation_kept, lower_kept, upper_kept = od.bounds.oriented_bounding_box(kept_wp)

    assert np.allclose(_frame_np(rotation_dense), _frame_np(rotation_kept), rtol=0, atol=0)
    assert np.array_equal(_bounds_np(lower_dense, upper_dense), _bounds_np(lower_kept, upper_kept))


def test_oriented_bounding_box_single_rotation_is_the_aabb(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    ``rotations=1`` scores only the identity, so it must reproduce ``aabb`` exactly.

    The identity is deliberately the *last* candidate rather than the first, which is what makes
    this an edge case worth pinning: an off-by-one in the spiral's ``n - 1`` split would drop it and
    return some arbitrary orientation here. ``refine_iterations=0``, because refinement exists to
    *improve* on the sampled answer -- refined ``rotations=1`` legitimately beats the AABB.
    """
    _, mesh_wp = sphere_irregular
    rotation_wp, lower_wp, upper_wp = od.bounds.oriented_bounding_box(
        mesh_wp.points, 1, refine_iterations=0
    )

    assert np.array_equal(_frame_np(rotation_wp), np.eye(3))
    assert np.array_equal(
        _bounds_np(lower_wp, upper_wp), _bounds_np(*od.bounds.aabb(mesh_wp.points))
    )


def test_oriented_bounding_box_never_loses_to_the_aabb(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Including the identity in the candidate set makes the result monotone in ``rotations``.

    Adding candidates can only lower the minimum, and every count contains the axis-aligned box, so
    the volume sequence must be non-increasing and never above the AABB's. That is a property of the
    candidate construction rather than of any one answer, and it is what lets a caller raise
    ``rotations`` without checking the result got better. ``refine_iterations=0`` for the sequence:
    refinement converges to whichever basin each count's winner lands in, so *refined* volumes are
    not monotone in ``rotations`` -- the refined guarantee is the separate one pinned in
    [`test_oriented_bounding_box_refinement_is_monotone`][tests.test_bounds.test_oriented_bounding_box_refinement_is_monotone].
    """
    mesh_tm, mesh_wp = saddle_graded
    _, points_wp = _tilted_cloud(mesh_tm, mesh_wp.device)

    volumes = [
        float(
            np.prod(
                np.ptp(
                    _bounds_np(
                        *od.bounds.oriented_bounding_box(points_wp, n, refine_iterations=0)[1:]
                    ),
                    axis=0,
                )
            )
        )
        for n in (1, 16, 256, 4096)
    ]

    assert volumes == sorted(volumes, reverse=True)
    assert volumes[-1] < volumes[0] * 0.95, "the tilted cloud's box is materially tighter"
    assert np.isclose(
        volumes[0], float(np.prod(np.ptp(_bounds_np(*od.bounds.aabb(points_wp)), axis=0)))
    )


@pytest.mark.parametrize("mesh_name", MESHES)
def test_oriented_bounding_box_refinement_is_monotone(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Refinement never loses to sampling, and the refined default ties or beats the sampled 32k box.

    The monotonicity is by construction -- every round's candidate ball includes the identity
    perturbation, so each chain re-scores its own base -- but the construction is exactly the kind
    of thing a refactor breaks silently, so it is pinned per fixture. The second assert is the
    reason the refinement exists: eight rounds on top of the *default* 4 096 candidates reach at
    least the quality of an unrefined 32 768-candidate search (measured: strictly better on every
    fixture here, from 0.3% on the icosahedron to 4.9% on the cube shell).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np, points_wp = _tilted_cloud(mesh_tm, mesh_wp.device)

    def volume(rotations: int, refine_iterations: int) -> float:
        rotation_wp, lower_wp, upper_wp = od.bounds.oriented_bounding_box(
            points_wp, rotations, refine_iterations=refine_iterations
        )
        volume_box = float(np.prod(np.ptp(_bounds_np(lower_wp, upper_wp), axis=0)))
        assert np.isclose(
            volume_box, _achieved_loss(points_np, _frame_np(rotation_wp), "volume"), rtol=1e-5
        )
        return volume_box

    sampled = volume(4096, 0)
    refined = volume(4096, 8)
    assert refined <= sampled * (1.0 + 1e-6)
    assert refined <= volume(32768, 0) * (1.0 + 1e-6)


@pytest.mark.parametrize("rotations", [1, 2, 3, 4, 33])
def test_oriented_bounding_box_refines_from_fewer_candidates_than_chains(
    rotations: int, device: str
) -> None:
    """
    Refinement is reachable below ``_REFINE_CHAINS`` candidates, and still never loses to the AABB.

    Ordito against ordito -- no reference refines a sampled box, and the oracle here is the
    axis-aligned box the candidate set is built to contain. The counts straddle two boundaries the
    rest of the file never reaches: the chain seeding has fewer distinct candidates than chains to
    give them below four, so it pads by repeating a pick, and ``_REFINE_WINDOW`` bounds the walk at
    33. Every other refinement test runs the 4 096 default, where neither branch is taken.

    What this covers is that those counts produce a *valid, refined* box, not the padding line
    itself: a chain seeded from a bad frame loses the final argmin, so the answer survives it
    (verified by disabling the pad, which leaves every test in this file passing). The padding is
    there so the seeding never hands the refinement an unwritten frame, and the kernel clamps that
    read besides.
    """
    rng = np.random.default_rng(31)
    cloud_np = (rng.standard_normal((2_000, 3)) @ np.diag([4.0, 1.0, 0.3])).astype(np.float32)
    cloud_wp = points_to_warp(cloud_np, device)

    rotation_wp, lower_wp, upper_wp = od.bounds.oriented_bounding_box(
        cloud_wp, rotations, refine_iterations=8
    )

    assert np.allclose(_frame_np(rotation_wp) @ _frame_np(rotation_wp).T, np.eye(3), atol=1e-5)
    volume_wp = float(np.prod(np.ptp(_bounds_np(lower_wp, upper_wp), axis=0)))
    volume_aabb = float(np.prod(np.ptp(_bounds_np(*od.bounds.aabb(cloud_wp)), axis=0)))
    assert volume_wp <= volume_aabb * (1.0 + 1e-6)
    # Non-vacuity: this cloud is tilted, so refinement has something to find even from one
    # candidate -- otherwise every count would pass by returning the axis-aligned box.
    assert volume_wp < volume_aabb * 0.95
    assert np.isclose(
        volume_wp, _achieved_loss(cloud_np, _frame_np(rotation_wp), "volume"), rtol=1e-5
    )


@pytest.mark.parametrize("seed_device", ["cpu", "cuda:0"])
def test_oriented_bounding_box_chain_seeding_agrees_across_devices(seed_device: str) -> None:
    """
    Ordito against ordito: the single-block chain seeding must pick the same frames on the CPU.

    ``oriented_box_seed_chains`` is the module's only ``wp.launch_tiled`` kernel, and the ``device``
    fixture returns ``cuda:0`` whenever CUDA is present, so without an explicit parametrize its CPU
    path is never executed. There ``wp.launch_tiled`` runs one lane per block and ``wp.block_dim()``
    reads 1, so the lone lane walks the whole loss table and every ``block_argmin`` folds a
    one-element tile -- which is correct precisely because the walk strides by ``wp.block_dim()``
    rather than by a constant.

    CUDA carries the oracle: it is the device every other test in this file runs on.
    """
    if seed_device == "cuda:0" and wp.get_cuda_device_count() == 0:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(37)
    cloud_np = (rng.standard_normal((1_500, 3)) @ np.diag([3.0, 1.5, 0.4])).astype(np.float32)

    rotation_wp, lower_wp, upper_wp = od.bounds.oriented_bounding_box(
        points_to_warp(cloud_np, seed_device), 512, refine_iterations=0
    )

    # The sampled phase is a pure argmin over a loss table, so the claim is that both devices pick
    # the *same candidate*, not merely a similar box -- two different candidates of a 512-frame
    # spiral are O(1) apart, so a tolerance this tight can only pass for one of them. It cannot be
    # bit-equality: ``oriented_box_candidate_axes`` fuses its ``quat_to_matrix`` multiply-adds on
    # CUDA and not on the CPU, so the candidate set itself differs by 1.19e-07 before any search
    # runs (measured, and the same on both sides of this change).
    reference_wp = points_to_warp(cloud_np, "cpu")
    rotation_cpu, lower_cpu, upper_cpu = od.bounds.oriented_bounding_box(
        reference_wp, 512, refine_iterations=0
    )
    assert np.allclose(_frame_np(rotation_wp), _frame_np(rotation_cpu), rtol=0, atol=1e-6)
    assert np.allclose(
        _bounds_np(lower_wp, upper_wp), _bounds_np(lower_cpu, upper_cpu), rtol=0, atol=1e-5
    )
    # Non-vacuity: the search must have improved on the axis-aligned box it also contains.
    assert (
        _achieved_loss(cloud_np, _frame_np(rotation_wp), "volume")
        < _achieved_loss(cloud_np, np.eye(3), "volume") * 0.99
    )


def test_oriented_bounding_box_empty_cloud(device: str) -> None:
    """An empty cloud gives the identity frame and the same inverted box ``aabb`` returns."""
    empty_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    rotation_wp, lower_wp, upper_wp = od.bounds.oriented_bounding_box(empty_wp)

    assert np.array_equal(_frame_np(rotation_wp), np.eye(3))
    assert np.all(np.isposinf(_bounds_np(lower_wp, upper_wp)[0]))
    assert np.all(np.isneginf(_bounds_np(lower_wp, upper_wp)[1]))


def test_oriented_bounding_box_rejects_bad_arguments(device: str) -> None:
    """All three documented ``ValueError``s are reachable, and none is raised for a valid call."""
    points_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32), dtype=wp.vec3, device=device
    )
    with pytest.raises(ValueError, match="rotations must be >= 1"):
        od.bounds.oriented_bounding_box(points_wp, 0)
    with pytest.raises(ValueError, match="objective must be"):
        od.bounds.oriented_bounding_box(points_wp, 8, "perimeter")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="refine_iterations must be >= 0"):
        od.bounds.oriented_bounding_box(points_wp, 8, refine_iterations=-1)
