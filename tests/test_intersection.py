"""Regression tests for ``ordito.intersection`` against Trimesh and potpourri3d."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

import igl
import numpy as np
import numpy.typing as npt
import potpourri3d as pp3d
import pytest
import pyvista as pv
import trimesh as tm
import trimesh.intersections as tm_intersections
import warp as wp
from meshlib import mrmeshpy as mm
from scipy.spatial import KDTree
from vtkmodules.vtkCommonDataModel import vtkPlane, vtkPlaneCollection
from vtkmodules.vtkFiltersGeneral import vtkClipClosedSurface

import ordito as od
from ordito.constants import TOLERANCE_MERGE
from ordito.kernels import graph as kernel_graph
from tests.comparisons import hausdorff_two_sided
from tests.conftest import MESHES, OPEN_MESHES
from tests.conversions import (
    meshlib_bitset_to_numpy,
    meshlib_to_trimesh,
    numpy_to_pymeshfix,
    numpy_to_warp,
    points_to_warp,
    pymeshfix_face_remap,
    pymeshfix_intersecting_faces,
    trimesh_to_meshlib,
    trimesh_to_pyvista,
    trimesh_to_warp,
    warp_empty,
    warp_to_trimesh,
)

_SPLIT_MESHES = ["icosphere", "unit_box", "torus"]


def _plane_ml(normal_np: np.ndarray, origin_np: np.ndarray) -> mm.Plane3f:
    """
    Build a ``Plane3f`` from ordito's ``(normal, point)`` pair.

    MeshLib's plane is the ``n . x == d`` form, so the named transform on every plane pairing in
    this file is ``d = normal . origin``. Passing the origin itself would place the plane through
    the coordinate origin instead, which on a centred mesh is a plausible-looking wrong answer.
    """
    return mm.Plane3f(
        mm.Vector3f(*np.asarray(normal_np, dtype=float).tolist()),
        float(np.dot(normal_np, origin_np)),
    )


def _section_points_ml(mesh_ml: mm.Mesh, section_ml: Iterable[mm.EdgePoint]) -> np.ndarray:
    """
    Decode one ``EdgePoint`` section into ``(n, 3)`` positions.

    ``extractPlaneSections`` and ``extractIsolines`` both return *barycentric* points on edges
    rather than coordinates, so each has to go through ``Mesh.edgePoint``; reading their fields
    directly gives an edge id and a parameter, not a position.
    """
    return np.array(
        [
            [
                mesh_ml.edgePoint(point_ml).x,
                mesh_ml.edgePoint(point_ml).y,
                mesh_ml.edgePoint(point_ml).z,
            ]
            for point_ml in section_ml
        ]
    )


def _canonical_segments(lines_np: np.ndarray) -> np.ndarray:
    """Sort segments for order-independent comparison. ``lines_np`` shape ``(m, 2, 3)``."""
    if lines_np.shape[0] == 0:
        return lines_np.reshape(0, 2, 3)
    pairs = []
    for seg in lines_np:
        a, b = np.sort(seg, axis=0)
        pairs.append(np.concatenate([a, b]))
    ordered = np.array(pairs)
    return ordered[
        np.lexsort(
            (
                ordered[:, 3],
                ordered[:, 4],
                ordered[:, 5],
                ordered[:, 0],
                ordered[:, 1],
                ordered[:, 2],
            )
        )
    ].reshape(-1, 2, 3)


def _segments_equal(
    got_np: np.ndarray, exp_np: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-5
) -> bool:
    got_segments = _canonical_segments(got_np.reshape(-1, 2, 3))
    expected_segments = _canonical_segments(exp_np.reshape(-1, 2, 3))
    if got_segments.shape != expected_segments.shape:
        return False
    if got_segments.shape[0] == 0:
        return True
    return bool(np.allclose(got_segments, expected_segments, rtol=rtol, atol=atol))


@pytest.mark.parity("segments_with_plane", "trimesh")
def test_segments_with_plane_axis_aligned(device: str) -> None:
    """
    Class A: crossing points and the validity mask against ``trimesh.intersections.plane_lines``.

    Three hand-built segments cover the three cases the mask distinguishes: one crossing, one
    entirely on the far side, one lying in the plane. Comparing the mask as well as the points
    is what makes the last two testable at all, since they produce no point.
    """
    plane_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    endpoints_np = np.array(
        [
            [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [1.0, 0.0, 2.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    endpoints_np = np.transpose(endpoints_np, (1, 0, 2))
    intersections_tm, valid_tm = tm_intersections.plane_lines(
        plane_origin, plane_normal, endpoints_np, line_segments=True
    )

    start_points_wp = points_to_warp(endpoints_np[0], device)
    end_points_wp = points_to_warp(endpoints_np[1], device)
    intersections_wp, valid_wp = od.intersection.segments_with_plane(
        start_points_wp,
        end_points_wp,
        wp.vec3(*plane_normal.tolist()),
        wp.vec3(*plane_origin.tolist()),
        line_segments=True,
    )

    assert np.array_equal(valid_wp.numpy(), valid_tm)
    assert np.allclose(
        intersections_wp.numpy()[valid_wp.numpy()], intersections_tm, rtol=1e-5, atol=1e-5
    )


def test_segments_with_plane_parallel(device: str) -> None:
    """
    Class A on the mask alone: a segment parallel to the plane has no crossing to compare.

    Split out from the test above because the *point* is undefined here -- only the ``False``
    in the validity mask is a claim, and asserting it beside real crossings would let a wrong
    point hide.
    """
    plane_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    endpoints_np = np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], dtype=np.float32)
    endpoints_np = np.transpose(endpoints_np, (1, 0, 2))
    _, valid_tm = tm_intersections.plane_lines(
        plane_origin, plane_normal, endpoints_np, line_segments=True
    )

    start_points_wp = points_to_warp(endpoints_np[0], device)
    end_points_wp = points_to_warp(endpoints_np[1], device)
    _, valid_wp = od.intersection.segments_with_plane(
        start_points_wp,
        end_points_wp,
        wp.vec3(*plane_normal.tolist()),
        wp.vec3(*plane_origin.tolist()),
        line_segments=True,
    )
    assert np.array_equal(valid_wp.numpy(), valid_tm)


def _axis_planes(mesh_tm: tm.Trimesh) -> list[tuple[np.ndarray, np.ndarray]]:
    """Three axis-aligned planes through the centroid; none can miss the mesh."""
    mid_np = 0.5 * (mesh_tm.bounds[0] + mesh_tm.bounds[1])
    return [
        (np.array([0.0, 0.0, 1.0]), mid_np),
        (np.array([1.0, 0.0, 0.0]), mid_np),
        (np.array([0.0, 1.0, 0.0]), mid_np),
    ]


def _tilted_plane(mesh_tm: tm.Trimesh) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build a plane aligned with no axis and no face.

    An axis-aligned plane can pass through vertices and edges of a symmetric fixture, which
    exercises the degenerate branches rather than the general one; an 11-degree tilt makes
    every crossing a clean edge interior.
    """
    axis_np = tm.unitize(np.array([1.0, 2.0, 0.3], dtype=np.float32))
    angle = np.radians(11)
    base = tm.transformations.rotation_matrix(angle=angle, direction=axis_np)
    plane_normal_np = tm.transform_points([[0.0, 0.0, 1.0]], base, translate=False)[0]
    plane_origin_np = tm.transform_points([mesh_tm.centroid], base)[0]
    return [(plane_normal_np, plane_origin_np)]


def _miss_plane(mesh_tm: tm.Trimesh) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build a plane translated well past the mesh's bounds, so it misses the mesh entirely."""
    return [(np.array([0.0, 0.0, 1.0]), mesh_tm.bounds[1] + np.array([0.0, 0.0, 10.0]))]


_MESH_WITH_PLANE_CASES = [
    *[
        pytest.param(mesh_name, _axis_planes, id=f"axis_planes-{mesh_name}")
        for mesh_name in ("icosahedron", "half_torus", "hemisphere")
    ],
    *[
        pytest.param(mesh_name, _tilted_plane, id=f"tilted_plane-{mesh_name}")
        for mesh_name in ("icosahedron", "hemisphere")
    ],
    pytest.param("icosahedron", _miss_plane, id="miss_plane"),
]


@pytest.mark.parametrize(("mesh_name", "plane_scenario"), _MESH_WITH_PLANE_CASES)
@pytest.mark.parity("mesh_with_plane", "trimesh")
def test_mesh_with_plane_matches_trimesh(
    request: pytest.FixtureRequest,
    mesh_name: str,
    plane_scenario: Callable[[tm.Trimesh], list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """
    Class B (segment canonicalization): the cross-section as an unordered set of segments.

    Neither library defines the segment order or which end of a segment comes first, so both
    sides go through ``_segments_equal``, which sorts endpoints within a segment and then
    segments within the set. ``miss_plane``'s expected answer is the empty set on both sides --
    ``_segments_equal`` already handles that, and this is verified directly against trimesh
    rather than only checking ordito's own output shape, which is what the unmerged test did.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    for plane_normal, plane_origin in plane_scenario(mesh_tm):
        lines_tm = tm_intersections.mesh_plane(
            mesh=mesh_tm, plane_normal=plane_normal, plane_origin=plane_origin
        )
        lines_wp = od.intersection.mesh_with_plane(
            mesh_wp.points,
            mesh_wp.indices,
            wp.vec3(*plane_normal.tolist()),
            wp.vec3(*plane_origin.tolist()),
        )
        assert _segments_equal(lines_wp.numpy(), cast("np.ndarray", lines_tm))


def test_mesh_with_plane_return_faces(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B: the ``return_faces`` half, where the face index must stay paired with its segment.

    The segments are compared as a canonicalized set as above, and the face indices as a sorted
    multiset -- pairing them elementwise is not possible once the segment order is
    canonicalized, which is the honest limit of this comparison.
    """
    mesh_tm, mesh_wp = icosahedron
    plane_normal = np.array([0.0, 0.0, 1.0])
    plane_origin = mesh_tm.centroid

    lines_tm, faces_tm = tm_intersections.mesh_plane(
        mesh=mesh_tm, plane_normal=plane_normal, plane_origin=plane_origin, return_faces=True
    )
    lines_wp, faces_wp = od.intersection.mesh_with_plane(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(*plane_normal.tolist()),
        wp.vec3(*plane_origin.tolist()),
        return_faces=True,
    )

    assert _segments_equal(lines_wp.numpy(), lines_tm)
    assert np.array_equal(np.sort(faces_wp.numpy()), np.sort(faces_tm))


@pytest.mark.parity("mesh_with_plane", "meshlib")
@pytest.mark.parity("mesh_with_mesh", "meshlib")
def test_plane_and_mesh_sections_match_meshlib(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B on the *curve*: both references order the contour where ordito returns segments.

    Neither pairing can be compared element-wise, and for the same reason in both directions:
    ``extractPlaneSections`` and ``findIntersectionContours`` walk the intersection into an ordered
    polyline, while ``mesh_with_plane`` and ``mesh_with_mesh`` emit an unordered ``(m, 2, 3)``
    segment soup. So the comparable quantities are the curve's **total length** and the point set it
    passes through, which is what a caller of either actually consumes.

    Measured on ``icosphere(3)`` cut at ``z = 0.13``: MeshLib returns **one** closed section of 95
    points against ordito's 94 segments -- the same 94, with the first point repeated to close --
    and the perimeters are **6.217219 against 6.217220**. Every decoded point sits at exactly the
    plane's ``z``, which is the assert that catches a plane built from the wrong ``d``.

    For the mesh-mesh half, two overlapping spheres give one contour of 153 points against 201
    segments -- different counts, since neither library promises a particular sampling of the same
    curve -- with total lengths **5.008801 against 5.008798**.
    """
    mesh_tm, mesh_wp = icosphere
    normal_np = np.array([0.0, 0.0, 1.0])
    origin_np = mesh_tm.vertices.mean(axis=0) + np.array([0.0, 0.0, 0.13])
    mesh_ml = trimesh_to_meshlib(mesh_tm)

    segments_wp = od.intersection.mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, wp.vec3(*normal_np.tolist()), wp.vec3(*origin_np.tolist())
    ).numpy()
    sections_ml = mm.extractPlaneSections(mm.MeshPart(mesh_ml), _plane_ml(normal_np, origin_np))

    assert len(sections_ml) == 1  # one closed rim, so the length comparison is not over fragments
    points_ml = _section_points_ml(mesh_ml, sections_ml[0])
    assert np.allclose(points_ml[:, 2], origin_np[2], atol=1e-5)  # the plane really is where it is
    assert points_ml.shape[0] == len(segments_wp) + 1  # closed: the first point repeats

    length_wp = float(np.linalg.norm(segments_wp[:, 1] - segments_wp[:, 0], axis=1).sum())
    length_ml = float(np.linalg.norm(np.diff(points_ml, axis=0), axis=1).sum())
    assert np.isclose(length_wp, length_ml, rtol=1e-5)

    # Mesh against mesh, on two overlapping spheres.
    other_tm = mesh_tm.copy()
    other_tm.apply_translation([1.2, 0.0, 0.0])
    other_wp = trimesh_to_warp(other_tm, str(mesh_wp.points.device))
    crossing_wp = od.intersection.mesh_with_mesh(
        mesh_wp.points, mesh_wp.indices, other_wp.points, other_wp.indices
    ).numpy()
    contours_ml = mm.findIntersectionContours(mesh_ml, trimesh_to_meshlib(other_tm))

    assert len(contours_ml) == 1
    contour_np = np.array([[point.x, point.y, point.z] for point in contours_ml[0]])
    assert contour_np.shape[0] > 10  # non-vacuity: the two spheres really do intersect
    assert np.isclose(
        float(np.linalg.norm(crossing_wp[:, 1] - crossing_wp[:, 0], axis=1).sum()),
        float(np.linalg.norm(np.diff(contour_np, axis=0), axis=1).sum()),
        rtol=1e-4,
    )


def _curves_pp(
    vertices_np: np.ndarray, faces_np: np.ndarray, values_np: np.ndarray, isovalue: float
) -> tuple[list[np.ndarray], list[bool]]:
    """Decode potpourri3d's barycentric contour output into point arrays and closed flags."""
    edges_pp = np.asarray(pp3d.edges(vertices_np, faces_np))
    curves: list[np.ndarray] = []
    closed: list[bool] = []
    for curve in pp3d.marching_triangles(vertices_np, faces_np, values_np, isovalue):
        points = []
        for element, barycentric in curve:
            if len(barycentric) == 0:  # a vertex
                points.append(vertices_np[element])
            elif len(barycentric) == 1:  # a point along an edge
                start, end = edges_pp[element]
                points.append(
                    (1.0 - barycentric[0]) * vertices_np[start] + barycentric[0] * vertices_np[end]
                )
            else:  # a point inside a face
                weights = np.array(
                    [barycentric[0], barycentric[1], 1.0 - barycentric[0] - barycentric[1]]
                )
                points.append(weights @ vertices_np[faces_np[element]])
        is_closed = len(points) > 2 and np.allclose(points[0], points[-1])
        curves.append(np.array(points[:-1] if is_closed else points))
        closed.append(is_closed)
    return curves, closed


def _total_length(curves: list[np.ndarray], closed: list[bool]) -> float:
    """Sum the curve lengths, counting the closing chord of every closed curve."""
    total = 0.0
    for points, is_closed in zip(curves, closed, strict=True):
        total += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if is_closed:
            total += float(np.linalg.norm(points[0] - points[-1]))
    return total


# ---------------------------------------------------------------------------
# marching_triangles
# ---------------------------------------------------------------------------


@pytest.mark.parity("marching_triangles", "meshlib")
@pytest.mark.parity("marching_triangles_curves", "meshlib")
def test_marching_triangles_matches_meshlib(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B: ``extractIsolines`` returns the same level set, closed by a repeated point.

    Two transforms, both named. Its ``vertValues`` is a ``VertScalars`` filled per vertex -- there
    is no array constructor, so the fill is a Python loop over ``VertId`` keys -- and its output is
    a list of ``EdgePoint`` contours that must go through ``Mesh.edgePoint`` to become positions.
    Its closed contours repeat their first point where ordito returns the cycle once and flags it
    ``closed``, which is the same convention potpourri3d's ``marching_triangles`` uses.

    Measured on the ``z`` field of ``icosphere(3)`` at 0.13: one closed curve, 94 points against 95,
    lengths **6.217219 against 6.217220**. That is the same curve
    [`test_plane_and_mesh_sections_match_meshlib`] gets from the plane section, which is the
    consistency worth having -- the level set of a coordinate *is* a plane section, and the two
    entry points reach it by different code.

    The second half covers the ``marching_triangles_curves`` group, whose question is the opposite
    one: a sinusoidal field whose level set is **many** short loops rather than one long one, where
    the linking is the work. Both libraries return **18** closed curves totalling 52.3072 against
    52.3072 -- agreeing to **1.0e-08** -- so the curve *count* is asserted as well as the length,
    which is what would break if either side split or merged a loop.
    """
    mesh_tm, mesh_wp = icosphere
    isovalue = float(mesh_tm.vertices[:, 2].mean() + 0.13)
    field_np = np.ascontiguousarray(mesh_tm.vertices[:, 2], dtype=np.float32)
    field_wp = wp.array(field_np, dtype=wp.float32, device=mesh_wp.points.device)

    curves_wp, closed_wp = od.intersection.marching_triangles(
        mesh_wp.points, mesh_wp.indices, field_wp, isovalue
    )

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    values_ml = mm.VertScalars()
    values_ml.resize(mesh_ml.points.size(), 0.0)
    for index, value in enumerate(field_np):
        values_ml[mm.VertId(index)] = float(value)
    isolines_ml = mm.extractIsolines(mesh_ml.topology, values_ml, isovalue)

    assert len(isolines_ml) == len(curves_wp) == 1
    assert closed_wp == [True]
    points_ml = _section_points_ml(mesh_ml, isolines_ml[0])
    curve_np = curves_wp[0].numpy()

    assert points_ml.shape[0] == len(curve_np) + 1  # its closed contour repeats the first point
    assert np.allclose(points_ml[0], points_ml[-1], atol=1e-6)
    length_wp = float(
        np.linalg.norm(np.diff(np.vstack([curve_np, curve_np[:1]]), axis=0), axis=1).sum()
    )
    assert np.isclose(
        length_wp, float(np.linalg.norm(np.diff(points_ml, axis=0), axis=1).sum()), rtol=1e-5
    )

    # The many-curve field: same count, same total length, and the linking is what could differ.
    wave_np = np.ascontiguousarray(
        np.sin(6.0 * mesh_tm.vertices[:, 0])
        * np.cos(6.0 * mesh_tm.vertices[:, 1])
        * np.sin(6.0 * mesh_tm.vertices[:, 2]),
        dtype=np.float32,
    )
    wave_curves_wp, wave_closed_wp = od.intersection.marching_triangles(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(wave_np, dtype=wp.float32, device=mesh_wp.points.device),
        0.0,
    )
    wave_values_ml = mm.VertScalars()
    wave_values_ml.resize(mesh_ml.points.size(), 0.0)
    for index, value in enumerate(wave_np):
        wave_values_ml[mm.VertId(index)] = float(value)
    wave_lines_ml = mm.extractIsolines(mesh_ml.topology, wave_values_ml, 0.0)

    assert len(wave_curves_wp) > 5  # non-vacuity: this field really is multi-curve
    assert len(wave_lines_ml) == len(wave_curves_wp)
    total_wp = sum(
        float(
            np.linalg.norm(
                np.diff(
                    np.vstack([curve.numpy(), curve.numpy()[:1]]) if is_closed else curve.numpy(),
                    axis=0,
                ),
                axis=1,
            ).sum()
        )
        for curve, is_closed in zip(wave_curves_wp, wave_closed_wp, strict=True)
    )
    total_ml = sum(
        float(np.linalg.norm(np.diff(_section_points_ml(mesh_ml, line_ml), axis=0), axis=1).sum())
        for line_ml in wave_lines_ml
    )
    assert np.isclose(total_wp, total_ml, rtol=1e-5)


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parametrize("axis", [0, 2])
@pytest.mark.parity("marching_triangles", "potpourri3d")
def test_marching_triangles_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, axis: int
) -> None:
    """
    Class B (barycentric decoding): potpourri3d reports hits in its own element numbering.

    Section 6 records the decode: ``(element_index, coords)`` pairs dispatched on
    ``len(coords)`` through ``pp3d.edges``, and its closed curves repeat their first point
    where ordito's do not. Both transforms are on the reference side; the positions are then
    compared directly.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    # A coordinate function, contoured a little off centre so the level set misses the vertices: an
    # exact vertex hit is a genuine convention difference (see the dedicated test below).
    values_np = np.ascontiguousarray(vertices_np[:, axis])
    isovalue = float(0.5137 * values_np.min() + 0.4863 * values_np.max())
    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)

    curves_wp, closed_wp = od.intersection.marching_triangles(
        mesh_wp.points, mesh_wp.indices, values_wp, isovalue, n_vertices=len(vertices_np)
    )
    curves_pp, closed_pp = _curves_pp(vertices_np, faces_np, values_np, isovalue)

    assert len(curves_wp) == len(curves_pp)
    assert sorted(closed_wp) == sorted(closed_pp)
    assert np.isclose(
        _total_length([curve.numpy() for curve in curves_wp], closed_wp),
        _total_length(curves_pp, closed_pp),
        rtol=1e-5,
        atol=1e-5,
    )
    # The level sets must coincide as point sets, not merely in total length.
    bounding_diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
    assert (
        hausdorff_two_sided(
            np.concatenate([curve.numpy() for curve in curves_wp]), np.concatenate(curves_pp)
        )
        < 1e-6 * bounding_diagonal
    )


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("marching_triangles", "igl")
@pytest.mark.parity("marching_triangles_curves", "igl")
def test_marching_triangles_matches_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: ``igl.isolines`` returns a segment **soup**, so the linking must be undone first.

    igl gives ``(points, segments, segment_values)`` with no curve structure at all -- every
    crossing is an independent 2-point segment -- where ``marching_triangles`` returns polylines.
    The named transform is therefore to reduce ordito's curves to the same soup: each consecutive
    pair of a curve is a segment, plus the closing pair for a closed curve. Two quantities are then
    directly comparable and both are asserted: the total segment length, and the point sets through
    a two-sided Hausdorff distance.

    Segment *count* is not compared, and that is deliberate: a polyline of ``n`` points contributes
    ``n - 1`` segments (``n`` closed), so ordito's count is derived from its linking while igl's is
    the raw crossing count -- they agree here but the equality is a property of this input rather
    than of the two algorithms, and asserting it would be asserting the wrong thing.

    Both this group and ``marching_triangles_curves`` are covered because the transform is the same
    for one long contour and for a thousand short ones; the linking count is what differs, and it is
    exactly what this comparison steps around.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)
    values_np = np.ascontiguousarray(vertices_np[:, 2])
    isovalue = float(0.5137 * values_np.min() + 0.4863 * values_np.max())
    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)

    points_igl, segments_igl, _values_igl = igl.isolines(
        vertices_np, faces_np, values_np, np.array([isovalue])
    )
    points_igl, segments_igl = np.asarray(points_igl), np.asarray(segments_igl)
    curves_wp, closed_wp = od.intersection.marching_triangles(
        mesh_wp.points, mesh_wp.indices, values_wp, isovalue, n_vertices=len(vertices_np)
    )

    assert segments_igl.shape[0] > 0
    length_igl = float(
        np.linalg.norm(
            np.asarray(points_igl)[segments_igl[:, 0]] - np.asarray(points_igl)[segments_igl[:, 1]],
            axis=1,
        ).sum()
    )
    assert np.isclose(
        _total_length([curve.numpy() for curve in curves_wp], closed_wp),
        length_igl,
        rtol=1e-5,
        atol=1e-5,
    )

    bounding_diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
    assert (
        hausdorff_two_sided(np.concatenate([curve.numpy() for curve in curves_wp]), points_igl)
        < 1e-5 * bounding_diagonal
    )


@pytest.mark.parity("marching_triangles_curves", "potpourri3d")
def test_marching_triangles_many_components_matches_potpourri3d(device: str) -> None:
    """
    Class B, same decoding, on a field whose level set breaks into many small loops.

    The single-contour test above exercises the per-face crossing arithmetic; this exercises
    the segment *linking*, which is where a many-component field can drop or merge a loop while
    every individual crossing stays right.
    """
    # An oscillating field breaks the level set into many small loops, which is what exercises the
    # segment linking rather than the per-face crossing arithmetic.
    mesh_tm = od.creation.icosphere(subdivisions=3)
    vertices_np = np.ascontiguousarray(mesh_tm[0].numpy(), dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm[1].numpy().reshape(-1, 3), dtype=np.int32)
    values_np = np.ascontiguousarray(
        np.sin(8.0 * vertices_np[:, 0]) * np.cos(8.0 * vertices_np[:, 1])
    )
    isovalue = 0.1370

    vertices_wp = points_to_warp(vertices_np, device)
    faces_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    values_wp = wp.array(values_np, dtype=wp.float64, device=device)

    curves_wp, closed_wp = od.intersection.marching_triangles(
        vertices_wp, faces_wp, values_wp, isovalue, n_vertices=len(vertices_np)
    )
    curves_pp, closed_pp = _curves_pp(vertices_np, faces_np, values_np, isovalue)

    assert len(curves_wp) > 10
    assert len(curves_wp) == len(curves_pp)
    assert all(closed_wp)
    assert np.isclose(
        _total_length([curve.numpy() for curve in curves_wp], closed_wp),
        _total_length(curves_pp, closed_pp),
        rtol=1e-5,
        atol=1e-5,
    )


def test_marching_triangles_open_curve_ends_on_the_boundary(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    mesh_tm, mesh_wp = hemisphere
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    # The hemisphere's rim is a single loop, so a level set of x has to run into it.
    values_np = np.ascontiguousarray(vertices_np[:, 0])
    isovalue = float(values_np.mean())
    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)

    curves_wp, closed_wp = od.intersection.marching_triangles(
        mesh_wp.points, mesh_wp.indices, values_wp, isovalue, n_vertices=len(vertices_np)
    )

    assert not all(closed_wp)
    boundary_vertices = mesh_wp.points.numpy()[
        od.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices).numpy()
    ]
    for points, is_closed in zip([curve.numpy() for curve in curves_wp], closed_wp, strict=True):
        if is_closed:
            continue
        # Both ends of an open curve sit on a boundary edge, hence within one edge length of a
        # boundary vertex.
        edge_length = od.edges.mean_edge_length(mesh_wp.points, mesh_wp.indices)
        for end in (points[0], points[-1]):
            assert np.linalg.norm(boundary_vertices - end, axis=1).min() <= edge_length


def test_marching_triangles_exact_vertex_hit_is_reported_once(device: str) -> None:
    # Two triangles sharing edge (1, 2), with the field vanishing exactly at both shared vertices,
    # so the level set *is* that edge. Counting a value equal to the isovalue as positive leaves the
    # all-positive face with nothing to report and the mixed-sign face with one segment along the
    # shared edge: the curve appears exactly once rather than twice or not at all.
    vertices_wp = wp.array(
        np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        dtype=wp.vec3,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2, 3, 2, 1], dtype=np.int32), dtype=wp.int32, device=device)
    values_wp = wp.array(np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32), device=device)

    curves_wp, closed_wp = od.intersection.marching_triangles(
        vertices_wp, faces_wp, values_wp, 0.0, n_vertices=4
    )

    assert len(curves_wp) == 1
    assert closed_wp == [False]
    assert curves_wp[0].shape == (2,)
    # The two endpoints are the shared vertices themselves.
    assert np.allclose(
        np.sort(curves_wp[0].numpy(), axis=0),
        np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_marching_triangles_level_set_is_the_piecewise_linear_one(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    mesh_tm, mesh_wp = icosahedron
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces)
    values_np = np.ascontiguousarray(vertices_np[:, 2])
    isovalue = 0.1234
    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)

    curves_wp, _ = od.intersection.marching_triangles(
        mesh_wp.points, mesh_wp.indices, values_wp, isovalue, n_vertices=len(vertices_np)
    )

    # One segment per face whose vertex values straddle the isovalue, and every segment lands in a
    # curve: the total point count equals the cut-face count for an all-closed level set.
    positive = values_np[faces_np] >= isovalue
    cut_faces = int((~(positive.all(axis=1) | (~positive).all(axis=1))).sum())
    assert sum(curve.size for curve in curves_wp) == cut_faces


def test_marching_triangles_no_crossing(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    values_wp = wp.array(
        np.ascontiguousarray(np.asarray(mesh_tm.vertices)[:, 2]),
        dtype=wp.float64,
        device=mesh_wp.device,
    )
    assert od.intersection.marching_triangles(mesh_wp.points, mesh_wp.indices, values_wp, 1e6) == (
        [],
        [],
    )


def test_marching_triangles_empty(device: str) -> None:
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    values_wp = warp_empty(0, wp.float32, device)
    assert od.intersection.marching_triangles(vertices_wp, faces_wp, values_wp) == ([], [])


@pytest.mark.parametrize("field", ["plane", "wave"])
def test_marching_triangles_is_its_packed_form_split(
    icosphere: tuple[tm.Trimesh, wp.Mesh], field: str
) -> None:
    """
    Ordito against ordito: the list form is the packed form, curve by curve.

    ``marching_triangles`` carries the potpourri3d, igl and MeshLib comparisons above; this pins
    ``marching_triangles_with_offsets`` to it -- the same points in the same order, total-terminated
    offsets bounding each curve, and the same closed flags. ``wave`` gives many curves, ``plane``
    one.
    """
    mesh_tm, mesh_wp = icosphere
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    if field == "plane":
        values_np = vertices_np[:, 2]
    else:
        values_np = np.sin(9.0 * vertices_np[:, 0]) * np.cos(7.0 * vertices_np[:, 1])
    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)

    curves_wp, closed_wp = od.intersection.marching_triangles(
        mesh_wp.points, mesh_wp.indices, values_wp, 0.05
    )
    points_wp, offsets_wp, closed_packed_wp = od.intersection.marching_triangles_with_offsets(
        mesh_wp.points, mesh_wp.indices, values_wp, 0.05
    )

    offsets_np = offsets_wp.numpy()
    assert len(curves_wp) == 1 if field == "plane" else len(curves_wp) > 10
    assert offsets_np.size == len(curves_wp) + 1
    assert offsets_np[0] == 0
    assert offsets_np[-1] == points_wp.size
    assert closed_packed_wp.numpy().tolist() == closed_wp
    for c, curve_wp in enumerate(curves_wp):
        assert np.array_equal(
            curve_wp.numpy(), points_wp.numpy()[offsets_np[c] : offsets_np[c + 1]]
        )


def test_marching_triangles_with_offsets_empty(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """An empty level set is no points, ``[0]`` offsets and no flags."""
    _, mesh_wp = icosahedron
    values_wp = wp.zeros(mesh_wp.points.size, dtype=wp.float32, device=mesh_wp.device)
    points_wp, offsets_wp, closed_wp = od.intersection.marching_triangles_with_offsets(
        mesh_wp.points, mesh_wp.indices, values_wp, 1e6
    )
    assert points_wp.size == 0
    assert closed_wp.size == 0
    assert offsets_wp.numpy().tolist() == [0]


def _marching_triangles_both_links(
    monkeypatch: pytest.MonkeyPatch,
    vertices_wp: wp.array[wp.vec3],
    faces_wp: wp.array[wp.int32],
    values_wp: wp.array[wp.float32] | wp.array[wp.float64],
    isovalue: float,
) -> list[tuple[list[np.ndarray], list[bool]]]:
    """Run ``marching_triangles`` once through the host link and once through the device link."""
    results = []
    for on_device in (False, True):
        monkeypatch.setattr(
            od.intersection, "_links_on_device", lambda device, n, on_device=on_device: on_device
        )
        curves_wp, closed_wp = od.intersection.marching_triangles(
            vertices_wp, faces_wp, values_wp, isovalue
        )
        results.append(([curve.numpy() for curve in curves_wp], closed_wp))
    return results


@pytest.mark.parametrize("hops", [2, 3, 16])
@pytest.mark.parametrize("field", ["random", "wave"])
@pytest.mark.parametrize("mesh_name", [*MESHES, "icosphere"])
def test_marching_triangles_device_link_matches_host_link(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    mesh_name: str,
    field: str,
    hops: int,
) -> None:
    """
    Ordito against ordito: the device link returns the host link's curves exactly.

    The host link (``_link_segments``) carries the oracle, through the potpourri3d, igl and
    MeshLib comparisons above; this pins the device link to it -- the same curves in the same
    order, each from the same first point, with the same closed flags -- on both devices, with the
    gate forced each way. The open meshes put open contours in the set, which rank from their
    unique first segment rather than from their lowest one, and ``hops`` -- the widest jump
    ``graph.pointer_jump_schedule`` may take -- walks the pointer-jumping rounds through several
    round counts, 2 and 3 putting the curve lengths on both sides of many round boundaries.
    """
    monkeypatch.setattr(kernel_graph, "POINTER_JUMP_MAX_HOPS", hops)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    rng = np.random.default_rng(23)
    if field == "random":
        values_np = rng.standard_normal(len(vertices_np))
    else:
        values_np = np.sin(9.0 * vertices_np[:, 0]) * np.cos(7.0 * vertices_np[:, 1])
    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)

    (curves_host, closed_host), (curves_device, closed_device) = _marching_triangles_both_links(
        monkeypatch, mesh_wp.points, mesh_wp.indices, values_wp, 0.1
    )

    assert len(curves_host) > 0
    if mesh_name in OPEN_MESHES:
        assert not all(closed_host)
    assert closed_device == closed_host
    assert len(curves_device) == len(curves_host)
    for points_device, points_host in zip(curves_device, curves_host, strict=True):
        assert np.array_equal(points_device, points_host)


def test_marching_triangles_rejects_inconsistent_winding(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """
    Not a library comparison: a face wound against its neighbours makes both links raise.

    A reversed face on the contour starts its segment on the edge its neighbour's also starts on,
    which the host link reports as inconsistent winding; the device link hands such a level set
    to the host link rather than chaining it.
    """
    vertices_np, faces_np = (array.numpy() for array in od.creation.icosphere(subdivisions=2))
    faces_np = faces_np.reshape(-1, 3).copy()
    values_np = vertices_np[:, 2].astype(np.float64) - 0.1
    positive = values_np[faces_np] >= 0.0
    straddles = np.flatnonzero(positive.any(axis=1) & ~positive.all(axis=1))
    faces_np[straddles[0]] = faces_np[straddles[0], ::-1]
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np.reshape(-1), device)
    values_wp = wp.array(values_np, dtype=wp.float64, device=device)

    for on_device in (False, True):
        monkeypatch.setattr(
            od.intersection, "_links_on_device", lambda device, n, on_device=on_device: on_device
        )
        with pytest.raises(ValueError, match="not consistently oriented"):
            od.intersection.marching_triangles(vertices_wp, faces_wp, values_wp, 0.0)


def _pyvista_intersection_segments(mesh1_pv: pv.PolyData, mesh2_pv: pv.PolyData) -> np.ndarray:
    intersection_pv = cast(
        pv.PolyData, mesh1_pv.intersection(mesh2_pv, split_first=False, split_second=False)[0]
    )
    if intersection_pv.n_cells == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    pairs_np = np.reshape(intersection_pv.lines, (-1, 3))[:, 1:]
    return intersection_pv.points[pairs_np]


def _segment_total_length(segments_np: np.ndarray) -> float:
    """Total length of a segment soup -- the one scalar an unordered curve comparison can share."""
    segments_np = segments_np.reshape(-1, 2, 3)
    return float(np.sum(np.linalg.norm(segments_np[:, 1] - segments_np[:, 0], axis=1)))


def _intersection_curves_match(
    got_segments_np: np.ndarray,
    ref_segments_np: np.ndarray,
    *,
    ref_atol: float = 1e-5,
    got_atol: float = 1e-5,
) -> bool:
    """Check both segment sets describe the same intersection curves."""
    got_segments_np = got_segments_np.reshape(-1, 2, 3)
    ref_segments_np = ref_segments_np.reshape(-1, 2, 3)
    if ref_segments_np.shape[0] == 0:
        return got_segments_np.shape[0] == 0
    if got_segments_np.shape[0] == 0:
        return False
    got_pts_np = got_segments_np.reshape(-1, 3)
    ref_pts_np = ref_segments_np.reshape(-1, 3)
    ref_distances_np = KDTree(got_pts_np).query(ref_pts_np, distance_upper_bound=ref_atol)[0]
    got_distances_np = KDTree(ref_pts_np).query(got_pts_np, distance_upper_bound=got_atol)[0]
    return bool(np.all(np.isfinite(ref_distances_np)) and np.all(np.isfinite(got_distances_np)))


def test_mesh_with_mesh_empty(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], cave_cube: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    _, ico_wp = icosahedron
    _, cave_wp = cave_cube

    lines_wp = od.intersection.mesh_with_mesh(
        ico_wp.points, ico_wp.indices, cave_wp.points, cave_wp.indices
    )
    assert lines_wp.shape == (0, 2)


@pytest.mark.parity("mesh_with_mesh", "pyvista")
def test_mesh_with_mesh_icosahedron_cave_cube(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], cave_cube: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class B: the same crossing curve as ``vtkIntersectionPolyDataFilter``, as an unordered set.

    Both sides emit a segment soup over the same crossing, so the transform is only that neither
    ordering is meaningful -- matched by two-sided nearest-neighbour containment plus the two
    quantities an ordering cannot affect: the segment **count** and the **total curve length**.
    Measured on this pairing: 36 = 36 segments, length 4.195738316 against 4.195736885 (relative
    3.4e-07) and a segment-midpoint Hausdorff of 5.96e-08 both ways. On two ``icosphere(3)``s offset
    by 0.6 the same three numbers read 170 = 170, a **bit-identical** 5.984692574 and 2.77e-07.

    MeshLib's ``findIntersectionContours`` covers the same group and does strictly more (it links
    the crossing into ordered contours); VTK's filter returns the soup ordito returns, which is why
    this is Class B where that pairing is Class C.
    """
    ico_tm, ico_wp = icosahedron
    cave_tm, _ = cave_cube
    cave_at_ico_tm = cave_tm.copy()
    cave_at_ico_tm.apply_translation(ico_tm.centroid)
    cave_wp = trimesh_to_warp(cave_at_ico_tm, ico_wp.device)

    ref_segments_np = _pyvista_intersection_segments(
        trimesh_to_pyvista(ico_tm), trimesh_to_pyvista(cave_at_ico_tm)
    )
    lines_wp = od.intersection.mesh_with_mesh(
        ico_wp.points, ico_wp.indices, cave_wp.points, cave_wp.indices
    )
    segments_np = lines_wp.numpy().reshape(-1, 2, 3)

    assert ref_segments_np.shape[0] > 0  # non-vacuity: VTK found the crossing
    assert segments_np.shape[0] == ref_segments_np.shape[0]
    assert _intersection_curves_match(segments_np, ref_segments_np)
    assert np.isclose(_segment_total_length(segments_np), _segment_total_length(ref_segments_np))


def test_mesh_with_mesh_near_tangent_pair_is_not_dropped(device: str) -> None:
    """
    A pair crossing at a very small dihedral angle is not silently dropped by the narrow phase.

    ``triangle_intersection_segment`` computes ``line_direction = cross(normal, other_normal)`` in
    float32; for two triangles related by a tiny tilt about a *generic* (non-axis-aligned) axis,
    this cross product used to underflow to the exact zero vector via catastrophic cancellation --
    two float32 products agreeing to more digits than float32 carries -- even though the two
    triangles genuinely cross (confirmed independently by the float64 broad-phase test that admits
    the pair). The fix does the whole computation in float64, the same precedent
    ``predicates.triangles_intersect`` already sets one level up. This exact vertex pair was found
    by searching random rigid rotations for one that reproduces the failure; it is pinned here
    rather than re-searched.
    """
    a = np.array(
        [
            [-4.6970606, -2.4794655, -4.6679626],
            [2.8626564, -6.442955, 0.54178125],
            [0.917202, 4.4612103, 2.0630906],
        ],
        dtype=np.float32,
    )
    b = np.array(
        [
            [-4.69706, -2.4794652, -4.667963],
            [2.8626568, -6.442955, 0.5417808],
            [0.9172017, 4.4612103, 2.063091],
        ],
        dtype=np.float32,
    )
    verts_np = np.concatenate([a, b], axis=0)
    faces_a_np = np.array([0, 1, 2], dtype=np.int32)
    faces_b_np = np.array([3, 4, 5], dtype=np.int32)
    verts_wp = wp.array(verts_np, dtype=wp.vec3, device=device)
    faces_a_wp = wp.array(faces_a_np, dtype=wp.int32, device=device)
    faces_b_wp = wp.array(faces_b_np, dtype=wp.int32, device=device)

    crossing = od.intersection._colliding_face_pairs(  # pyright: ignore[reportPrivateUsage]
        verts_wp, faces_a_wp, verts_wp, faces_b_wp, 16, "probe"
    )
    assert crossing is not None, "the float64 broad phase must confirm this pair genuinely crosses"

    segments_wp = od.intersection.mesh_with_mesh(verts_wp, faces_a_wp, verts_wp, faces_b_wp)
    assert int(segments_wp.shape[0]) == 1
    p0, p1 = segments_wp.numpy()[0]
    assert np.linalg.norm(p1 - p0) > 1e-5, "the recovered segment must not itself be degenerate"


# --- marching_triangles: isocontours of a scalar field (potpourri3d reference) ---------
@pytest.mark.parity("mesh_collision_pairs", "meshlib")
def test_mesh_collision_pairs_matches_meshlib(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class A: the same colliding faces as ``findCollidingTriangleBitsets``, both masks.

    A sphere against a small box pushed into its side, which is also the configuration that
    exercises the **query/target swap**: the broad phase queries the larger mesh's faces against
    the smaller one's BVH, so the pair columns come out in face-count order and have to be put back
    into the caller's. Both argument orders are tested for that reason -- measured (26, 8) faces
    one way and (8, 26) the other, matching MeshLib element for element in both.

    A pair list is also checked against the masks it reduces to, since the two entry points share a
    helper and could disagree only by the reduction.
    """
    mesh_tm, _ = icosphere
    box_tm = tm.creation.box(extents=[0.5, 0.5, 0.5])
    box_tm.apply_translation([0.9, 0.0, 0.0])

    sphere_wp = trimesh_to_warp(mesh_tm, device)
    box_wp = trimesh_to_warp(box_tm, device)
    colliding_ml = mm.findCollidingTriangleBitsets(
        mm.MeshPart(trimesh_to_meshlib(mesh_tm)), mm.MeshPart(trimesh_to_meshlib(box_tm))
    )
    sphere_ml = meshlib_bitset_to_numpy(colliding_ml[0], mesh_tm.faces.shape[0])
    box_ml = meshlib_bitset_to_numpy(colliding_ml[1], box_tm.faces.shape[0])
    assert int(sphere_ml.sum()) > 0  # non-vacuity: the reference found the collision

    sphere_mask_wp, box_mask_wp = od.intersection.collision_masks(
        sphere_wp.points, sphere_wp.indices, box_wp.points, box_wp.indices
    )
    assert np.array_equal(sphere_mask_wp.numpy(), sphere_ml)
    assert np.array_equal(box_mask_wp.numpy(), box_ml)

    # The reversed argument order, where the swap fires the other way.
    box_first_wp, sphere_second_wp = od.intersection.collision_masks(
        box_wp.points, box_wp.indices, sphere_wp.points, sphere_wp.indices
    )
    assert np.array_equal(box_first_wp.numpy(), box_ml)
    assert np.array_equal(sphere_second_wp.numpy(), sphere_ml)

    # The pairs reduce to those masks, and every index is in range for its own mesh.
    pairs_np = od.intersection.mesh_collision_pairs(
        sphere_wp.points, sphere_wp.indices, box_wp.points, box_wp.indices
    ).numpy()
    assert len(pairs_np) > 0
    assert pairs_np[:, 0].max() < mesh_tm.faces.shape[0]
    assert pairs_np[:, 1].max() < box_tm.faces.shape[0]
    assert set(pairs_np[:, 0].tolist()) == set(np.flatnonzero(sphere_ml).tolist())
    assert set(pairs_np[:, 1].tolist()) == set(np.flatnonzero(box_ml).tolist())


@pytest.mark.parity(
    "mesh_collision_pairs",
    "pymeshfix",
    benchmarked=False,
    reason="pymeshfix has no two-mesh collision entry point at all -- the answer comes from "
    "running its single-mesh select_intersecting_triangles over the concatenation, so a row "
    "would time the wrong operation on the wrong input shape, behind a load that is 49-50 % of "
    "the round. "
    "meshlib's findCollidingTriangleBitsets is the two-mesh call and carries the timed row.",
)
@pytest.mark.parametrize("kind", ["sphere_and_box", "two_spheres"])
def test_mesh_collision_pairs_matches_pymeshfix(device: str, kind: str) -> None:
    """
    Class B: the two masks, unioned into one face set over the concatenated buffer.

    pymeshfix answers a *self*-intersection question, so the transform is to concatenate the two
    meshes and offset the second mesh's face indices -- which is only equivalent because neither
    input self-intersects, asserted here by checking each mesh alone flags nothing. The second half
    of the transform is [`pymeshfix_face_remap`][tests.conversions.pymeshfix_face_remap]:
    ``load_array`` returns a *reordering* of the face buffer even when it repairs nothing, so a raw
    index comparison fails on a mesh whose counts match. Measured on the sphere pair, that failure
    looks like a real disagreement rather than a plumbing one -- 42 faces on each side both ways,
    with only **7 of 42** indices in common. Through the remap the sets are **equal**.

    Both fixtures agree exactly, on both devices: (26, 8) faces for the sphere against the box and
    (42, 42) for the two spheres, i.e. 34 and 84 faces of the concatenation.

    Worth knowing if this ever fails alone: pymeshfix counts a *touching* pair where ordito and
    MeshLib (at ``touchIsIntersection=False``) do not, so its answer is a superset in general and
    equal here only because both configurations interpenetrate cleanly. ``tests/test_repair.py``
    carries a measured case where the two conventions differ by 97 faces.
    """
    if kind == "sphere_and_box":
        first_tm = tm.creation.icosphere(subdivisions=3)
        second_tm = tm.creation.box(extents=[0.5, 0.5, 0.5])
        second_tm.apply_translation([0.9, 0.0, 0.0])
        n_expected = (26, 8)
    else:
        first_tm = tm.creation.icosphere(subdivisions=2)
        second_tm = tm.creation.icosphere(subdivisions=2)
        second_tm.apply_translation([0.7, 0.0, 0.0])
        n_expected = (42, 42)

    first_wp = trimesh_to_warp(first_tm, device)
    second_wp = trimesh_to_warp(second_tm, device)
    first_mask_np, second_mask_np = (
        mask_wp.numpy()
        for mask_wp in od.intersection.collision_masks(
            first_wp.points, first_wp.indices, second_wp.points, second_wp.indices
        )
    )
    assert (int(first_mask_np.sum()), int(second_mask_np.sum())) == n_expected

    # The transform is only valid because neither input crosses itself.
    for mesh_tm in (first_tm, second_tm):
        alone_pmf = numpy_to_pymeshfix(mesh_tm.vertices, mesh_tm.faces)
        assert pymeshfix_intersecting_faces(alone_pmf, tris_per_cell=50, justproper=False).size == 0

    n_first = first_tm.faces.shape[0]
    vertices_np = np.vstack([first_tm.vertices, second_tm.vertices])
    faces_np = np.vstack(
        [np.asarray(first_tm.faces), np.asarray(second_tm.faces) + first_tm.vertices.shape[0]]
    )
    tin_pmf = numpy_to_pymeshfix(vertices_np, faces_np)
    assert tin_pmf.n_faces == faces_np.shape[0]  # the remap below needs an untouched load
    faces_pmf = pymeshfix_face_remap(tin_pmf, faces_np)[
        pymeshfix_intersecting_faces(tin_pmf, tris_per_cell=50, justproper=False)
    ]

    expected = set(np.flatnonzero(first_mask_np).tolist()) | set(
        (np.flatnonzero(second_mask_np) + n_first).tolist()
    )
    assert set(faces_pmf.tolist()) == expected


def test_mesh_collision_pairs_beats_meshlib_on_axis_aligned_boxes(device: str) -> None:
    """
    Not a parity assert: the input class where the two disagree, arbitrated rather than tolerated.

    Two unit boxes offset by ``(0.5, 0.5, 0.5)`` interpenetrate at a corner, and every crossing
    there is between axis-aligned triangles whose edges are parallel -- the configuration a
    separating-axis narrow phase gets wrong and an interval test does not. An exact ``float64``
    Moller test over all 144 face pairs says **(6, 6)**; ordito reports (6, 6) and MeshLib says
    **(5, 4)**, missing one face on the first mesh and two on the second.

    So this is recorded as a place ordito is *more* accurate, which is worth pinning for two
    reasons: the equality above must not be generalized into "the two always agree", and a future
    change that made ordito match MeshLib here would be a regression rather than a fix.
    """
    first_tm = tm.creation.box()
    second_tm = tm.creation.box()
    second_tm.apply_translation([0.5, 0.5, 0.5])
    first_wp = trimesh_to_warp(first_tm, device)
    second_wp = trimesh_to_warp(second_tm, device)

    first_mask_wp, second_mask_wp = od.intersection.collision_masks(
        first_wp.points, first_wp.indices, second_wp.points, second_wp.indices
    )
    assert int(first_mask_wp.numpy().sum()) == 6
    assert int(second_mask_wp.numpy().sum()) == 6

    colliding_ml = mm.findCollidingTriangleBitsets(
        mm.MeshPart(trimesh_to_meshlib(first_tm)), mm.MeshPart(trimesh_to_meshlib(second_tm))
    )
    first_ml = meshlib_bitset_to_numpy(colliding_ml[0], first_tm.faces.shape[0])
    # MeshLib finds a strict subset here, which is the divergence being pinned.
    assert int(first_ml.sum()) < 6
    assert np.all(first_mask_wp.numpy()[first_ml])


def test_mesh_collision_pairs_degenerate(device: str) -> None:
    """
    Not a library comparison: separated meshes, an empty mesh, and the candidate-cap guard.

    Two boxes three units apart must report no collision at all -- the case a broad phase that
    forgot its narrow phase would fail -- and an empty face buffer must give empty answers rather
    than raising, since ``collision_masks`` still owes a mask per mesh.
    """
    first_tm = tm.creation.box()
    far_tm = tm.creation.box()
    far_tm.apply_translation([3.0, 0.0, 0.0])
    first_wp = trimesh_to_warp(first_tm, device)
    far_wp = trimesh_to_warp(far_tm, device)

    pairs_wp = od.intersection.mesh_collision_pairs(
        first_wp.points, first_wp.indices, far_wp.points, far_wp.indices
    )
    assert pairs_wp.shape == (0, 2)
    first_mask_wp, far_mask_wp = od.intersection.collision_masks(
        first_wp.points, first_wp.indices, far_wp.points, far_wp.indices
    )
    assert not bool(first_mask_wp.numpy().any())
    assert not bool(far_mask_wp.numpy().any())

    empty_faces_wp = warp_empty(0, wp.int32, device)
    empty_pairs_wp = od.intersection.mesh_collision_pairs(
        first_wp.points, first_wp.indices, first_wp.points, empty_faces_wp
    )
    assert empty_pairs_wp.shape == (0, 2)
    _mask_wp, empty_mask_wp = od.intersection.collision_masks(
        first_wp.points, first_wp.indices, first_wp.points, empty_faces_wp
    )
    assert empty_mask_wp.shape == (0,)

    with pytest.raises(ValueError, match="max_triangle_collisions"):
        od.intersection.mesh_collision_pairs(
            first_wp.points,
            first_wp.indices,
            far_wp.points,
            far_wp.indices,
            max_triangle_collisions=0,
        )


def _sliced_meshes_equivalent(
    vertices_a_np: np.ndarray,
    faces_a_np: np.ndarray,
    vertices_b_np: np.ndarray,
    faces_b_np: np.ndarray,
    plane_normal: np.ndarray,
    plane_origin: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    if len(faces_a_np) != len(faces_b_np):
        return False
    if len(faces_a_np) == 0:
        # A plane that touches without cutting (test_slice_mesh_with_plane_on_plane) leaves both
        # sides with no faces and nothing left to compare -- trimesh's own .bounds is None for a
        # zero-face mesh, so the checks below cannot run on this case at all.
        return len(vertices_a_np) == 0 and len(vertices_b_np) == 0
    mesh_a_tm = tm.Trimesh(vertices_a_np, faces_a_np, process=False)
    mesh_b_tm = tm.Trimesh(vertices_b_np, faces_b_np, process=False)
    if not np.allclose(mesh_a_tm.bounds, mesh_b_tm.bounds, rtol=rtol, atol=atol):
        return False
    if not np.isclose(mesh_a_tm.area, mesh_b_tm.area, rtol=1e-4, atol=1e-4):
        return False
    dots_b_np = np.dot(plane_normal, (mesh_b_tm.vertices - plane_origin).T)
    return bool(np.min(dots_b_np) >= -max(TOLERANCE_MERGE, 1e-5))


def _box_corner_plane(mesh_tm: tm.Trimesh) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build a corner cut crossing three faces at once, so the retriangulation has a real choice."""
    return [(mesh_tm.bounds[1], mesh_tm.bounds[1] - 0.05)]


def _box_top_plane(mesh_tm: tm.Trimesh) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build a face-parallel cut: whole faces fall on one side rather than being split."""
    return [(np.array([0.0, 0.0, 1.0]), mesh_tm.bounds[1] - 0.05)]


def _on_plane(mesh_tm: tm.Trimesh) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build a plane exactly at the bounding box's top: touches the mesh without cutting it."""
    return [(np.array([0.0, 0.0, 1.0]), mesh_tm.bounds[1])]


_SLICE_MESH_WITH_PLANE_CASES = [
    pytest.param("unit_box", _box_corner_plane, 5, id="box_corner"),
    pytest.param("unit_box", _box_top_plane, 14, id="box_top"),
    *[
        pytest.param(mesh_name, _axis_planes, None, id=f"axis_planes-{mesh_name}")
        for mesh_name in ("icosahedron", "hemisphere")
    ],
    pytest.param("icosahedron", _tilted_plane, None, id="tilted_plane"),
    pytest.param("icosahedron", _on_plane, None, id="on_plane"),
]


@pytest.mark.parametrize(
    ("mesh_name", "plane_scenario", "expected_face_count"), _SLICE_MESH_WITH_PLANE_CASES
)
@pytest.mark.parity("slice_mesh_with_plane", "trimesh")
def test_slice_mesh_with_plane_matches_trimesh(
    request: pytest.FixtureRequest,
    mesh_name: str,
    plane_scenario: Callable[[tm.Trimesh], list[tuple[np.ndarray, np.ndarray]]],
    expected_face_count: int | None,
) -> None:
    """
    Class B: cross-sections against ``slice_faces_plane``, compared as canonical winding rows.

    ``box_corner`` and ``box_top`` share the ``unit_box`` fixture and pin the retriangulation on
    flat faces with clean corners -- one crossing three faces at once, the case where the
    retriangulation has a real choice, the other face-parallel, where whole faces should stay
    untouched rather than being split; each carries its own expected face count, which no other
    scenario needs. ``axis_planes``/``tilted_plane`` run the same comparison on curved and open
    fixtures, where the cut meets many small faces and, on ``hemisphere``, an existing boundary.
    ``on_plane`` is the boundary between *cut* and *miss*, where a strict-versus-inclusive side
    test changes the answer -- trimesh's convention is the reference, and both sides end up
    empty, which ``_sliced_meshes_equivalent`` now handles directly rather than needing its own
    separate assertion.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    for plane_normal_np, plane_origin_np in plane_scenario(mesh_tm):
        vertices_tm, faces_tm, _ = tm_intersections.slice_faces_plane(
            mesh_tm.vertices, mesh_tm.faces, plane_normal_np, plane_origin_np
        )
        vertices_wp, faces_wp = od.intersection.slice_mesh_with_plane(
            mesh_wp.points,
            mesh_wp.indices,
            wp.vec3(*plane_normal_np.tolist()),
            wp.vec3(*plane_origin_np.tolist()),
        )
        vertices_wp_np = vertices_wp.numpy()
        faces_wp_np = faces_wp.numpy().reshape(-1, 3)
        assert _sliced_meshes_equivalent(
            vertices_tm, faces_tm, vertices_wp_np, faces_wp_np, plane_normal_np, plane_origin_np
        )
        if expected_face_count is not None:
            assert len(faces_tm) == expected_face_count


# ---------------------------------------------------------------------------
# split_mesh_with_plane
# ---------------------------------------------------------------------------


@pytest.mark.parity("slice_mesh_with_plane", "meshlib")
@pytest.mark.parity("split_mesh_with_plane", "meshlib")
def test_slice_and_split_with_plane_match_meshlib(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B on the retriangulation: the same face counts and the same area, from two mutating calls.

    ``trimWithPlane`` keeps the **positive** side, which is ordito's convention, and
    ``subdivideWithPlane`` inserts the section as real edges and returns the ``FaceBitSet`` of that
    side -- exactly the ``(vertices, faces, side_mask)`` triple ``split_mesh_with_plane`` returns.
    Both mutate the mesh they are given and return something else, so each gets its own.

    Measured on ``icosphere(3)`` at ``z = 0.13``, and the agreement is stronger than the class
    suggests: the slice is **670 faces** on both sides with an area of **5.438285** on both, and the
    split is **1 468 faces, 736 vertices and 670 positive faces** on both, with an area of
    **12.506493**. What is *not* shared is the vertex count of the slice -- 477 against 383 -- since
    ordito emits a cut vertex per crossing edge where MeshLib reuses its half-edge topology, which
    is why this is a count-and-area comparison rather than a buffer one.

    The plane's ``d`` is the transform, and the assert that catches it getting lost is the z-range:
    both results start exactly at the cut.
    """
    mesh_tm, mesh_wp = icosphere
    normal_np = np.array([0.0, 0.0, 1.0])
    origin_np = mesh_tm.vertices.mean(axis=0) + np.array([0.0, 0.0, 0.13])
    plane_ml = _plane_ml(normal_np, origin_np)

    sliced_vertices_wp, sliced_faces_wp = od.intersection.slice_mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, wp.vec3(*normal_np.tolist()), wp.vec3(*origin_np.tolist())
    )
    sliced_tm = warp_to_trimesh(sliced_vertices_wp, sliced_faces_wp)

    trimmed_ml = trimesh_to_meshlib(mesh_tm)
    trim_params_ml = mm.TrimWithPlaneParams()
    trim_params_ml.plane = plane_ml
    assert mm.trimWithPlane(trimmed_ml, trim_params_ml) is None  # mutates, returns nothing
    trimmed_tm = meshlib_to_trimesh(trimmed_ml)

    assert 0 < trimmed_tm.faces.shape[0] < mesh_tm.faces.shape[0]  # it really trimmed
    assert sliced_tm.faces.shape[0] == trimmed_tm.faces.shape[0]
    assert np.isclose(sliced_tm.area, trimmed_tm.area, rtol=1e-5)
    assert np.isclose(sliced_tm.vertices[:, 2].min(), origin_np[2], atol=1e-5)
    assert np.isclose(trimmed_tm.vertices[:, 2].min(), origin_np[2], atol=1e-5)

    # The splitting form: the same cut, both sides kept, with the positive side as a mask.
    split_vertices_wp, split_faces_wp, side_wp = od.intersection.split_mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, wp.vec3(*normal_np.tolist()), wp.vec3(*origin_np.tolist())
    )
    split_tm = warp_to_trimesh(split_vertices_wp, split_faces_wp)

    subdivided_ml = trimesh_to_meshlib(mesh_tm)
    positive_ml = mm.subdivideWithPlane(subdivided_ml, plane_ml)
    subdivided_tm = meshlib_to_trimesh(subdivided_ml)

    assert split_tm.faces.shape[0] == subdivided_tm.faces.shape[0]
    assert split_tm.vertices.shape[0] == subdivided_tm.vertices.shape[0]
    assert int(side_wp.numpy().sum()) == positive_ml.count()
    assert int(side_wp.numpy().sum()) == sliced_tm.faces.shape[0]  # and it is the sliced side
    assert np.isclose(split_tm.area, subdivided_tm.area, rtol=1e-5)
    assert np.isclose(split_tm.area, mesh_tm.area, rtol=1e-5)  # splitting conserves the surface


@pytest.mark.parity("split_mesh_with_plane", "pyvista")
def test_split_mesh_with_plane_matches_pyvista(device: str) -> None:
    """
    Class B against ``PolyData.clip(return_clipped=True)``, VTK's both-sides plane clip.

    The named transform is the **side convention**: pyvista's ``clip`` keeps the side the normal
    points *away* from, so its ``kept`` is ordito's ``~above`` and its ``clipped`` is ``above``
    (measured ``kept z in [-1, 0.1]`` against a plane at ``z = 0.1`` with normal ``+z``). Take the
    names at face value and the two areas are swapped, which the per-side asserts below catch.

    Compared as the *union's* geometry plus the label partition rather than cell-for-cell: VTK
    numbers its output points in its own traversal order and splits the two-crossing quad on its own
    diagonal, so there is no face correspondence to assert. What is asserted is stronger than a
    total: each side's area separately, which pins the partition, and both point sets
    bidirectionally (measured 5.4e-08).

    VTK 9.7's clip emits a triangle cut with two corners kept as **one quad** (its linear-cell
    clip now reads ``vtkMarchingCellsClipCases``), where earlier releases and ordito emit two
    triangles, so the reference is ``triangulate()``-d before any count is compared.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    mesh_wp = trimesh_to_warp(mesh_tm, device)
    height = 0.1

    vertices_wp, faces_wp, above_wp = od.intersection.split_mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, height)
    )
    kept_pv, clipped_pv = trimesh_to_pyvista(mesh_tm).clip(
        normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, height), return_clipped=True
    )
    kept_pv, clipped_pv = kept_pv.triangulate(), clipped_pv.triangulate()

    # Anti-vacuity: a plane that missed, or that kept one side only, would pass everything below.
    above_np = above_wp.numpy()
    assert 0 < int(above_np.sum()) < above_np.size
    assert kept_pv.n_cells > 0
    assert clipped_pv.n_cells > 0

    faces_np = faces_wp.numpy().reshape(-1, 3)
    points_np = vertices_wp.numpy().astype(np.float64)
    assert faces_np.shape[0] == kept_pv.n_cells + clipped_pv.n_cells

    # ``clipped`` is the +normal side, so it pairs with ``above``.
    assert np.isclose(
        tm.Trimesh(points_np, faces_np[above_np], process=False).area, clipped_pv.area, rtol=1e-5
    )
    assert np.isclose(
        tm.Trimesh(points_np, faces_np[~above_np], process=False).area, kept_pv.area, rtol=1e-5
    )

    union_pv = np.vstack([np.asarray(kept_pv.points), np.asarray(clipped_pv.points)])
    assert np.max(KDTree(union_pv).query(points_np)[0]) < 1e-5
    assert np.max(KDTree(points_np).query(union_pv)[0]) < 1e-5


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "hemisphere", "half_torus"])
def test_split_mesh_with_plane_refines_without_cracking(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Not a library comparison: crack-free, on-plane, side-pure and area-preserving invariants.

    Every one of these would fail for a per-face cut like
    [`slice_mesh_with_plane`][ordito.intersection.slice_mesh_with_plane]'s, which is the point of
    the function: a closed input stays closed, the Euler characteristic is unchanged (inserting a
    curve of edges into a triangulation adds equal numbers of vertices, edges and faces), and no
    output face straddles the plane, which is what makes the ``above`` label well defined.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    normal_np = np.array([0.3, -0.5, 1.0])
    normal_np = normal_np / np.linalg.norm(normal_np)
    origin_np = mesh_tm.vertices.mean(axis=0)
    n_vertices_in = mesh_wp.points.size
    closed_in = od.validation.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=False)

    vertices_wp, faces_wp, above_wp = od.intersection.split_mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, wp.vec3(*normal_np.tolist()), wp.vec3(*origin_np.tolist())
    )
    points_np = vertices_wp.numpy().astype(np.float64)
    faces_np = faces_wp.numpy().reshape(-1, 3)
    above_np = above_wp.numpy()

    # Anti-vacuity: a plane through the centroid must actually cut.
    assert len(points_np) > n_vertices_in
    assert 0 < int(above_np.sum()) < above_np.size

    # Every inserted vertex lies on the plane.
    inserted = points_np[n_vertices_in:]
    assert np.abs((inserted - origin_np) @ normal_np).max() < 1e-5

    # No face straddles, so the label is exact rather than a majority vote.
    dots = (points_np[faces_np] - origin_np) @ normal_np
    assert not ((dots > 1e-6).any(axis=1) & (dots < -1e-6).any(axis=1)).any()
    assert (above_np == (dots.max(axis=1) > 1e-6)).all()

    # Crack-free: manifoldness and area survive, and so does the Euler characteristic.
    assert od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=not closed_in) is True
    if closed_in:
        assert od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False) is True
    assert np.isclose(tm.Trimesh(points_np, faces_np, process=False).area, mesh_tm.area, rtol=1e-5)
    assert od.measures.euler_characteristic(faces_wp) == od.measures.euler_characteristic(
        mesh_wp.indices
    )


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_split_mesh_with_plane_above_block_is_the_slice(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The ``above`` submesh is exactly what ``slice_mesh_with_plane`` returns.

    Not ordito-compared-with-itself for its own sake: the two share no code path — the slice cuts
    per face into three compacted classes, this splits per *edge* and labels afterwards — so
    agreeing on face count and area to ``1e-6`` is a real cross-check of the label convention, which
    is the one thing a caller has to get right. It also pins the documented promise that the two
    agree, including the in-plane-face tie-break.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    normal = wp.vec3(0.0, 0.0, 1.0)
    origin = wp.vec3(*mesh_tm.vertices.mean(axis=0).tolist())

    split_v, split_f, above = od.intersection.split_mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, normal, origin
    )
    above_v, above_f = od.selection.submesh_from_face_mask(split_v, split_f, above)
    slice_v, slice_f = od.intersection.slice_mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, normal, origin
    )

    assert above_f.size > 0
    assert above_f.size == slice_f.size
    assert np.isclose(
        warp_to_trimesh(above_v, above_f).area, warp_to_trimesh(slice_v, slice_f).area, rtol=1e-6
    )


def test_split_mesh_with_plane_through_a_vertex_inserts_nothing_there(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    A vertex already on the plane is used as the crossing rather than duplicated beside it.

    This is what the ``tolerance`` parameter buys, and the assert that bites is the *count*: without
    the strict-opposite-signs test every edge incident to the on-plane vertex would also be
    "crossed" at a point coinciding with it, giving a fan of zero-length edges and degenerate
    faces. Checked by the count of inserted vertices and by the absence of a degenerate face.
    """
    mesh_tm, mesh_wp = icosahedron
    apex = int(np.argmax(mesh_tm.vertices[:, 2]))
    normal_np = np.array([0.0, 0.0, 1.0])
    origin_np = mesh_tm.vertices[apex]
    n_vertices_in = mesh_wp.points.size

    vertices_wp, faces_wp, _ = od.intersection.split_mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, wp.vec3(*normal_np.tolist()), wp.vec3(*origin_np.tolist())
    )
    points_np = vertices_wp.numpy().astype(np.float64)

    # The apex is the unique highest vertex of an icosahedron, so a plane through it touches the
    # surface at that point alone: nothing is crossed and nothing is inserted.
    assert len(points_np) == n_vertices_in
    assert faces_wp.size == mesh_wp.indices.size
    # No zero-area face was introduced anywhere.
    assert (
        tm.Trimesh(points_np, faces_wp.numpy().reshape(-1, 3), process=False).area_faces.min() > 0
    )


def test_split_mesh_with_plane_misses_the_mesh(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """A plane clear of the mesh returns it unchanged with a constant label."""
    mesh_tm, mesh_wp = icosahedron
    vertices_wp, faces_wp, above_wp = od.intersection.split_mesh_with_plane(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, float(mesh_tm.bounds[1][2]) + 1.0),
    )
    assert np.array_equal(faces_wp.numpy(), mesh_wp.indices.numpy())
    assert np.allclose(vertices_wp.numpy(), mesh_wp.points.numpy())
    assert not above_wp.numpy().any()


# ---------------------------------------------------------------------------
# clip_mesh_with_field
# ---------------------------------------------------------------------------


def _height_field(mesh_tm: tm.Trimesh, device: str) -> wp.array[wp.float32]:
    """Take the z coordinate as a per-vertex ``float32`` field: a horizontal plane's distance."""
    return wp.array(
        np.ascontiguousarray(mesh_tm.vertices[:, 2], dtype=np.float32),
        dtype=wp.float32,
        device=device,
    )


@pytest.mark.parity("clip_mesh_with_field", "pyvista")
def test_clip_mesh_with_field_matches_pyvista_clip_scalar(device: str) -> None:
    """
    Class A on the kept surface, against ``PolyData.clip_scalar`` over the identical field.

    VTK cuts the same triangles at the same crossings, so the face counts are equal and the
    positions agree as point sets — measured 5.4e-08 on ``icosphere(3)``. The vertex *order* differs
    because each side appends its crossing points in its own traversal order, hence the
    nearest-neighbour comparison rather than an element-wise one.

    ``invert=False`` is not optional and is the trap in this row: ``clip_scalar``'s **default keeps
    the low side** (measured 798 faces below ``z = 0.1`` against 670 above), where ordito keeps
    ``values >= isovalue``. Take the default and the two answers are different regions of the same
    mesh, which the face-count assert catches only because they happen to differ in size.

    VTK 9.7's clip emits a triangle cut with two corners kept as **one quad** (its linear-cell
    clip now reads ``vtkMarchingCellsClipCases``), where earlier releases and ordito emit two
    triangles, so the reference is ``triangulate()``-d before any count is compared.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    mesh_wp = trimesh_to_warp(mesh_tm, device)
    isovalue = 0.1

    clipped_v, clipped_f = od.intersection.clip_mesh_with_field(
        mesh_wp.points, mesh_wp.indices, _height_field(mesh_tm, device), isovalue
    )
    mesh_pv = trimesh_to_pyvista(mesh_tm)
    mesh_pv.point_data["height"] = np.ascontiguousarray(mesh_tm.vertices[:, 2])
    clipped_pv = mesh_pv.clip_scalar(scalars="height", value=isovalue, invert=False).triangulate()

    # Anti-vacuity: a clip that kept nothing, or everything, would pass the comparisons below.
    assert 0 < clipped_pv.n_faces < len(mesh_tm.faces)
    assert clipped_f.size // 3 == clipped_pv.n_faces
    points_np = clipped_v.numpy().astype(np.float64)
    points_pv = np.asarray(clipped_pv.points)
    assert np.max(KDTree(points_pv).query(points_np)[0]) < 1e-5
    assert np.max(KDTree(points_np).query(points_pv)[0]) < 1e-5
    assert np.isclose(
        tm.Trimesh(points_np, clipped_f.numpy().reshape(-1, 3), process=False).area,
        clipped_pv.area,
        rtol=1e-5,
    )
    # Nothing below the isovalue survived.
    assert points_np[:, 2].min() >= isovalue - 1e-5


@pytest.mark.parity("clip_mesh_with_field", "pyvista")
@pytest.mark.parametrize("subdivisions", [3, 4])
@pytest.mark.parametrize("height_fraction", [0.3, 0.5, 0.81])
def test_clip_mesh_with_field_capped_matches_pyvista_clip_closed_surface(
    device: str, subdivisions: int, height_fraction: float
) -> None:
    """
    Class A on the solid, against ``clip_closed_surface`` -- VTK's capped plane clip.

    The two cappers triangulate the section differently (a min-weight interval DP here, VTK's own
    there), so the comparison is the vertex set, the face count and the *solid*: the same points
    (matched both ways by nearest neighbour), the same count and the same volume to seven digits at
    every level probed. Watertightness and edge-manifoldness are asserted on
    both sides, which is what the cap exists to restore and what a cracked section rim would break.

    The level is a height quantile, so it falls between vertices at irregular offsets: the case that
    broke the earlier cap, which welded the clip's output with a relative-bucket tolerance and so
    merged crossing points into input vertices a bucket away -- ``icosphere(4)`` at the 0.81
    quantile came back with 136 open and 12 non-manifold edges and 142 faces short of VTK's.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh_wp = trimesh_to_warp(mesh_tm, device)
    isovalue = float(np.quantile(mesh_tm.vertices[:, 2], height_fraction))

    capped_v, capped_f = od.intersection.clip_mesh_with_field(
        mesh_wp.points, mesh_wp.indices, _height_field(mesh_tm, device), isovalue, cap=True
    )
    capped_tm = warp_to_trimesh(capped_v, capped_f)
    mesh_pv = trimesh_to_pyvista(mesh_tm)
    closed_pv = mesh_pv.clip_closed_surface(
        normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, isovalue)
    ).triangulate()

    assert closed_pv.n_open_edges == 0
    assert capped_tm.is_watertight
    assert od.validation.is_edge_manifold(capped_f, allow_boundary_edges=False)
    assert len(capped_tm.faces) == closed_pv.n_faces
    # Both caps reuse the section's own points, so the vertex sets are the same set: as many points
    # on each side, and every point of each within float32 rounding of one of the other's.
    points_np = capped_tm.vertices.astype(np.float64)
    points_pv = np.asarray(closed_pv.points, dtype=np.float64)
    assert len(points_np) == len(points_pv)
    assert np.max(KDTree(points_pv).query(points_np)[0]) < 1e-5
    assert np.max(KDTree(points_np).query(points_pv)[0]) < 1e-5
    assert np.isclose(capped_tm.volume, closed_pv.volume, rtol=1e-5)
    # The cap is not free: without it the same clip is open.
    _, uncapped_f = od.intersection.clip_mesh_with_field(
        mesh_wp.points, mesh_wp.indices, _height_field(mesh_tm, device), isovalue
    )
    assert capped_f.size > uncapped_f.size


def _vtk_clip_closed_surface(
    mesh_tm: tm.Trimesh, origin: npt.ArrayLike, normal: npt.ArrayLike
) -> pv.PolyData:
    """
    Run ``vtkClipClosedSurface`` itself, keeping the ``normal`` side of the plane, triangulated.

    pyvista's ``clip_closed_surface`` refuses any input with an open edge before VTK sees it; the
    VTK class accepts one, caps the cut's own loops and leaves every other boundary open, which is
    the behaviour the open-input comparisons below are against.
    """
    plane = vtkPlane()
    plane.SetOrigin(*np.asarray(origin, dtype=np.float64).tolist())
    plane.SetNormal(*np.asarray(normal, dtype=np.float64).tolist())
    planes = vtkPlaneCollection()
    planes.AddItem(plane)
    clipper = vtkClipClosedSurface()
    clipper.SetClippingPlanes(planes)
    clipper.SetInputData(trimesh_to_pyvista(mesh_tm))
    clipper.Update()
    return pv.wrap(clipper.GetOutput()).triangulate()


@pytest.mark.parity("clip_mesh_with_field", "pyvista")
def test_clip_mesh_with_field_capped_leaves_an_input_boundary_open(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class A against ``vtkClipClosedSurface`` on an open input: only the section is sealed.

    The cut is across the dome's axis, keeping the band next to the rim, so the kept region holds
    the input's own boundary as well as the section. VTK caps the cut and leaves the rim open; this
    must do the same -- same vertex and face counts, the same number of open edges (the rim's, all
    of them), the same area (an open surface has no volume to compare). Sealing every boundary
    loop instead would close the rim and fail the open-edge count.
    """
    mesh_tm, mesh_wp = hemisphere
    device = str(mesh_wp.device)
    # The fixture's dome axis and rim centre, as it rotates and translates a z-up hemisphere.
    rim_np = mesh_tm.vertices[
        np.unique(mesh_tm.edges[tm.grouping.group_rows(mesh_tm.edges_sorted, require_count=1)])
    ]
    center_np = rim_np.mean(axis=0)
    axis_np = mesh_tm.vertices.mean(axis=0) - center_np
    axis_np /= np.linalg.norm(axis_np)
    height = 0.5
    # Kept where the height above the rim plane is at most ``height``.
    field_wp = wp.array(
        np.ascontiguousarray(-(mesh_tm.vertices - center_np) @ axis_np, dtype=np.float32),
        dtype=wp.float32,
        device=device,
    )
    capped_v, capped_f = od.intersection.clip_mesh_with_field(
        mesh_wp.points, mesh_wp.indices, field_wp, -height, cap=True
    )
    capped_tm = warp_to_trimesh(capped_v, capped_f)
    clipped_pv = _vtk_clip_closed_surface(mesh_tm, center_np + height * axis_np, -axis_np)

    rim_edges = int(od.boundary.boundary_edges(capped_v, capped_f).shape[0])
    assert rim_edges > 0  # non-vacuity: the input's rim survives the clip
    assert rim_edges == clipped_pv.n_open_edges
    assert len(capped_tm.vertices) == clipped_pv.n_points
    assert len(capped_tm.faces) == clipped_pv.n_faces
    assert np.isclose(capped_tm.area, clipped_pv.area, rtol=1e-5)


@pytest.mark.parity("clip_mesh_with_field", "pyvista")
def test_clip_mesh_with_field_capped_keeps_coincident_input_vertices(device: str) -> None:
    """
    Class A against ``vtkClipClosedSurface``: coincident input vertices are not merged.

    Two closed icospheres placed so one vertex of each lands on the same point, each with its own
    copy. VTK merges only the points the cut creates, so both copies survive and the parts stay
    disjoint; this must return the same vertex count, face count and volume, and two components.
    Merging the copies -- which the earlier cap's weld did -- would join the parts at a pinch vertex
    and lose one vertex.
    """
    sphere_a = tm.creation.icosphere(subdivisions=2)
    sphere_b = sphere_a.copy()
    touching = sphere_a.vertices[:, 0].argmax()
    sphere_b.apply_translation(
        sphere_a.vertices[touching] - sphere_b.vertices[sphere_b.vertices[:, 0].argmin()]
    )
    mesh_tm = tm.Trimesh(
        np.vstack([sphere_a.vertices, sphere_b.vertices]),
        np.vstack([sphere_a.faces, sphere_b.faces + len(sphere_a.vertices)]),
        process=False,
    )
    mesh_wp = trimesh_to_warp(mesh_tm, device)
    isovalue = -0.1
    assert mesh_tm.vertices[touching, 2] > isovalue  # the coincident pair is kept

    capped_v, capped_f = od.intersection.clip_mesh_with_field(
        mesh_wp.points, mesh_wp.indices, _height_field(mesh_tm, device), isovalue, cap=True
    )
    capped_tm = warp_to_trimesh(capped_v, capped_f)
    clipped_pv = _vtk_clip_closed_surface(mesh_tm, (0.0, 0.0, isovalue), (0.0, 0.0, 1.0))

    assert len(capped_tm.vertices) == clipped_pv.n_points
    assert len(capped_tm.faces) == clipped_pv.n_faces
    assert np.isclose(capped_tm.volume, clipped_pv.volume, rtol=1e-5)
    assert capped_tm.is_watertight
    assert len(capped_tm.split(only_watertight=False)) == 2


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_clip_mesh_with_field_reproduces_slice_mesh_with_plane(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The plane clip *is* this function over the plane's signed distance, so the two must agree.

    No reference: this pins the delegation itself. The one documented difference is the face lying
    in the level set, which the plane resolves from its normal and the field cannot — so the plane
    used here misses every vertex, keeping the two paths comparable face for face.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    plane_origin_np = 0.5 * (mesh_tm.bounds[0] + mesh_tm.bounds[1])
    plane_normal_np = np.array([0.0, 0.0, 1.0])
    field_wp = wp.array(
        np.ascontiguousarray(
            (mesh_tm.vertices - plane_origin_np) @ plane_normal_np, dtype=np.float32
        ),
        dtype=wp.float32,
        device=mesh_wp.device,
    )

    sliced_v, sliced_f = od.intersection.slice_mesh_with_plane(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(*plane_normal_np.tolist()),
        wp.vec3(*plane_origin_np.tolist()),
    )
    clipped_v, clipped_f = od.intersection.clip_mesh_with_field(
        mesh_wp.points, mesh_wp.indices, field_wp
    )
    assert sliced_f.size > 0
    assert np.array_equal(clipped_f.numpy(), sliced_f.numpy())
    assert np.allclose(clipped_v.numpy(), sliced_v.numpy(), rtol=1e-5, atol=1e-5)


def test_clip_mesh_with_field_through_a_saddle_vertex_reuses_it(device: str) -> None:
    """
    A saddle vertex within tolerance of the level set is reused exactly, not interpolated nearby.

    ``test_split_mesh_with_plane_through_a_vertex_inserts_nothing_there`` covers the *local
    extremum* case (a plane through the icosahedron's apex, whose whole ring is on one side), which
    ``classify_faces_for_split``'s ``SPLIT_CLASS_CUT_CORNER`` and this function's slice/clip path
    both handle safely without reaching the cut-triangle kernel at all. This is the other case that
    class exists for: a *saddle* on-plane vertex, whose ring straddles the level set, which used to
    land in ``SLICE_CLASS_CUT_TRI`` alongside a genuine two-crossing cut and get a new,
    near-duplicate vertex interpolated next to it instead of reusing it.
    """
    # One triangle: corner 0 sits within TOLERANCE_MERGE of the level set (but not exactly on it,
    # so a naive interpolation would land a hair's breadth away rather than exactly on it), corner
    # 1 is strictly kept, corner 2 is strictly dropped.
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces_np = np.array([0, 1, 2], dtype=np.int32)
    values_np = np.array([0.5 * TOLERANCE_MERGE, 1.0, -1.0], dtype=np.float32)

    vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)

    new_v, new_f = od.intersection.clip_mesh_with_field(vertices_wp, faces_wp, values_wp, 0.0)
    positions_np = new_v.numpy()

    # Exactly one genuine crossing (corner 1 to corner 2) plus the reused on-plane corner: three
    # vertices, one triangle -- not four vertices from two independently-interpolated corners.
    assert len(positions_np) == 3
    assert new_f.size == 3
    assert np.any(np.all(positions_np == vertices_np[0], axis=1)), (
        "the on-plane vertex must be reused bit-exactly, not interpolated near it"
    )
    mesh_out = tm.Trimesh(positions_np, new_f.numpy().reshape(-1, 3), process=False)
    assert mesh_out.area_faces.min() > 0


def test_clip_mesh_with_field_section_is_the_marching_triangles_curve(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Round trip for the region/level-set pair: the clip's rim is the contour, edge for edge.

    [`marching_triangles`][ordito.intersection.marching_triangles] returns the level set and this
    returns the region on one side of it, so the boundary of the region has to *be* the level set.
    Compared as total length plus a two-sided point-set distance, because the clip's rim carries one
    vertex per crossing where the contour carries one per segment endpoint.

    The isovalue deliberately misses every vertex, asserted below. Four of the icosahedron's twelve
    sit at exactly the centroid's height, and a contour through a vertex is where the two functions
    legitimately differ: the contour reports a zero-length segment there (documented) while the clip
    has a single rim vertex, so the counts stop matching for a reason that is not a bug.
    """
    mesh_tm, mesh_wp = icosahedron
    isovalue = float(mesh_tm.centroid[2]) + 0.17
    field_np = mesh_tm.vertices[:, 2]
    assert np.abs(field_np - isovalue).min() > 1e-3, "the isovalue must miss every vertex"
    field_wp = wp.array(
        np.ascontiguousarray(field_np, dtype=np.float32), dtype=wp.float32, device=mesh_wp.device
    )

    curves, closed = od.intersection.marching_triangles(
        mesh_wp.points, mesh_wp.indices, field_wp, isovalue
    )
    assert len(curves) == 1
    contour_np = curves[0].numpy().astype(np.float64)

    clipped_v, clipped_f = od.intersection.clip_mesh_with_field(
        mesh_wp.points, mesh_wp.indices, field_wp, isovalue
    )
    welded_v, _unique, _inverse, welded_f = od.repair.remove_duplicated_vertices(
        clipped_v, clipped_f
    )
    rim_edges_np = od.boundary.boundary_edges(welded_v, welded_f).numpy()
    positions_np = welded_v.numpy().astype(np.float64)
    rim_np = positions_np[np.unique(rim_edges_np)]
    assert len(rim_np) == len(contour_np)
    assert hausdorff_two_sided(rim_np, contour_np) < 1e-5

    rim_segments_np = positions_np[rim_edges_np]
    rim_length = float(np.linalg.norm(rim_segments_np[:, 1] - rim_segments_np[:, 0], axis=1).sum())
    contour_length = _total_length([curve.numpy().astype(np.float64) for curve in curves], closed)
    assert np.isclose(rim_length, contour_length, rtol=1e-4)


def test_clip_mesh_with_field_accepts_a_float64_field(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """A ``float64`` field — what ``heat_geodesic`` returns — clips the same region as its cast."""
    mesh_tm, mesh_wp = icosahedron
    field_np = mesh_tm.vertices[:, 2] - mesh_tm.centroid[2]
    isovalue = 0.05
    clipped_64 = od.intersection.clip_mesh_with_field(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.ascontiguousarray(field_np), dtype=wp.float64, device=mesh_wp.device),
        isovalue,
    )
    clipped_32 = od.intersection.clip_mesh_with_field(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(
            np.ascontiguousarray(field_np, dtype=np.float32),
            dtype=wp.float32,
            device=mesh_wp.device,
        ),
        isovalue,
    )
    assert clipped_64[1].size > 0
    assert np.array_equal(clipped_64[1].numpy(), clipped_32[1].numpy())
    assert np.allclose(clipped_64[0].numpy(), clipped_32[0].numpy(), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# empty mesh is a no-op, across mesh_with_plane / slice_mesh_with_plane /
# split_mesh_with_plane / clip_mesh_with_field
# ---------------------------------------------------------------------------

# (name, callable, expected shapes) triples; each callable takes (vertices, faces) and returns
# the tuple of arrays the wrapper produces, paired with the exact shape each must have on an
# empty mesh -- kept per-function rather than normalized to "shape[0] == 0", since
# mesh_with_plane's line-segment array is 2D and the original test pinned both dimensions.
_INTERSECTION_EMPTY_MESH_CASES = [
    (
        "mesh_with_plane",
        lambda v, f: (
            od.intersection.mesh_with_plane(v, f, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0)),
        ),
        ((0, 2),),
    ),
    (
        "slice_mesh_with_plane",
        lambda v, f: od.intersection.slice_mesh_with_plane(
            v, f, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0)
        ),
        ((0,), (0,)),
    ),
    (
        "split_mesh_with_plane",
        lambda v, f: od.intersection.split_mesh_with_plane(
            v, f, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0)
        ),
        ((0,), (0,), (0,)),
    ),
    (
        "clip_mesh_with_field",
        lambda v, f: od.intersection.clip_mesh_with_field(
            v, f, warp_empty(0, wp.float32, f.device)
        ),
        ((0,), (0,)),
    ),
]


@pytest.mark.parametrize(
    ("intersection_fn", "expected_shapes"),
    [case[1:] for case in _INTERSECTION_EMPTY_MESH_CASES],
    ids=[case[0] for case in _INTERSECTION_EMPTY_MESH_CASES],
)
def test_intersection_empty_mesh_is_a_noop(
    device: str,
    intersection_fn: Callable[[wp.array[wp.vec3], wp.array[wp.int32]], tuple[wp.array[Any], ...]],
    expected_shapes: tuple[tuple[int, ...], ...],
) -> None:
    """Not a library comparison: every operator returns an all-empty result on an empty mesh."""
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = warp_empty(0, wp.int32, device)
    result = intersection_fn(vertices_wp, faces_wp)
    for got_wp, expected_shape in zip(result, expected_shapes, strict=True):
        assert got_wp.shape == expected_shape


def _off_vertex_isovalue(mesh_tm: tm.Trimesh) -> float:
    """
    Pick a height strictly between two vertex heights, so no corner lies *on* the level set.

    A quantile of the data is often a data point, and an isovalue that coincides with a vertex is a
    different case with a different answer: ordito splits such a face into two triangles where VTK
    emits three, so the parity row would fail on a convention rather than on a defect.

    The midpoint of the **largest gap** rather than of an arbitrary neighbouring pair, because
    ``np.unique`` on a float64 height column separates values that differ at 1e-17 -- on the
    ``torus`` fixture a ring's heights are equal to within that, so a midpoint taken there lands
    back on a vertex and cuts nothing.
    """
    heights_np = np.unique(np.asarray(mesh_tm.vertices[:, 2], dtype=np.float64))
    widest = int(np.argmax(np.diff(heights_np)))
    return float(0.5 * (heights_np[widest] + heights_np[widest + 1]))


def _split_sides(
    mesh_wp: wp.Mesh, field_wp: wp.array[wp.float32], isovalue: float
) -> tuple[tm.Trimesh, tm.Trimesh, tm.Trimesh]:
    """Split along the level set and return the whole result and its two sides, as trimeshes."""
    vertices_wp, faces_wp, positive_wp = od.intersection.split_faces_along_field(
        mesh_wp.points, mesh_wp.indices, field_wp, isovalue
    )
    positive_np = positive_wp.numpy()
    negative_wp = wp.array(~positive_np, dtype=wp.bool, device=faces_wp.device)
    return (
        warp_to_trimesh(vertices_wp, faces_wp),
        warp_to_trimesh(*od.selection.submesh_from_face_mask(vertices_wp, faces_wp, positive_wp)),
        warp_to_trimesh(*od.selection.submesh_from_face_mask(vertices_wp, faces_wp, negative_wp)),
    )


@pytest.mark.parity("split_faces_along_field", "pyvista")
@pytest.mark.parametrize("mesh_name", _SPLIT_MESHES)
def test_split_faces_along_field_matches_pyvista_clip_scalar_both(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: equal after one named transform -- VTK returns its two halves *low side first*.

    ``clip_scalar(both=True)`` is the two-sided form of the filter the clip is already pinned
    against, and it returns a **tuple** ``(below, above)`` where this function's mask marks the
    ``>= isovalue`` side. So block 1 pairs with the mask and block 0 with its negation; taking the
    tuple in order compares each side against the other one, which on a symmetric mesh would still
    pass on area. Both orders are asserted here for that reason.

    At that pairing it is exact: face counts equal and areas equal to six decimals on all three
    fixtures. Areas rather than positions, because each side appends its crossing points in its own
    order -- the same reason the clip's own row uses a nearest-neighbour compare.

    VTK 9.7's clip emits a triangle cut with two corners kept as **one quad** (its linear-cell
    clip now reads ``vtkMarchingCellsClipCases``), where earlier releases and ordito emit two
    triangles, so the reference is ``triangulate()``-d before any count is compared.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    isovalue = _off_vertex_isovalue(mesh_tm)
    # The row's precondition: no corner sits on the level set, so every cut is edge-to-edge.
    assert np.abs(np.asarray(mesh_tm.vertices[:, 2]) - isovalue).min() > 1e-4
    _whole_tm, positive_tm, negative_tm = _split_sides(
        mesh_wp, _height_field(mesh_tm, str(mesh_wp.device)), isovalue
    )

    mesh_pv = trimesh_to_pyvista(mesh_tm)
    mesh_pv.point_data["height"] = np.ascontiguousarray(mesh_tm.vertices[:, 2])
    below_pv, above_pv = mesh_pv.clip_scalar(scalars="height", value=isovalue, both=True)
    assert above_pv is not None
    assert below_pv is not None
    below_pv, above_pv = below_pv.triangulate(), above_pv.triangulate()

    # Non-vacuity: an isovalue outside the field's range would leave one side empty and every
    # comparison below trivially true. Note a clipped side can carry *more* cells than the whole
    # input, since cutting a face emits two or three, so the input's count is not an upper bound.
    assert above_pv.n_cells > 0
    assert below_pv.n_cells > 0
    assert len(positive_tm.faces) > 0
    assert len(negative_tm.faces) > 0

    assert len(positive_tm.faces) == above_pv.n_cells
    assert len(negative_tm.faces) == below_pv.n_cells
    assert np.isclose(positive_tm.area, above_pv.area, rtol=1e-5)
    assert np.isclose(negative_tm.area, below_pv.area, rtol=1e-5)
    # And the transform is load-bearing wherever the two sides differ in size, so the swapped
    # pairing fails rather than passing by accident. ``unit_box`` is exactly the case where it
    # cannot bite -- its widest height gap is the midplane, which halves it symmetrically into two
    # areas of 3.0 -- which is why the other two fixtures are in this row.
    if not np.isclose(positive_tm.area, negative_tm.area, rtol=1e-5):
        assert not np.isclose(positive_tm.area, below_pv.area, rtol=1e-5)


@pytest.mark.parametrize("mesh_name", _SPLIT_MESHES)
def test_split_faces_along_field_partitions_the_surface(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Not a library comparison: these are the properties that make the result *one* mesh.

    A reference agreeing on both sides' areas -- which the row above checks -- says nothing about
    whether they are joined. Four things say that, and none of them is visible in an area: the
    output is still **watertight** and edge-manifold, which needs the crossing vertices to be shared
    by the faces on both sides of every cut edge rather than duplicated per face; the total area is
    unchanged, so nothing was dropped or double-counted; the two sides' areas **sum** to it, so the
    mask is a partition and not two overlapping selections; and every input vertex is still at its
    own position and its own index, which is what lets a caller carry per-vertex data across.

    The per-face crossing is the failure this excludes, and it is the natural implementation: it
    gives the right areas, the right face counts and a **non**-watertight result, which is why
    ``clip_mesh_with_field(cap=True)`` has to merge its crossings before it can fill.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    isovalue = _off_vertex_isovalue(mesh_tm)
    field_wp = _height_field(mesh_tm, str(mesh_wp.device))
    vertices_wp, faces_wp, _positive_wp = od.intersection.split_faces_along_field(
        mesh_wp.points, mesh_wp.indices, field_wp, isovalue
    )
    whole_tm, positive_tm, negative_tm = _split_sides(mesh_wp, field_wp, isovalue)

    assert faces_wp.size // 3 > len(mesh_tm.faces)  # non-vacuity: faces were cut
    assert od.validation.is_edge_manifold(faces_wp)
    assert whole_tm.is_watertight == mesh_tm.is_watertight
    assert np.isclose(whole_tm.area, mesh_tm.area, rtol=1e-5)
    assert np.isclose(positive_tm.area + negative_tm.area, mesh_tm.area, rtol=1e-5)
    n_vertices = mesh_wp.points.size
    assert vertices_wp.size > n_vertices
    assert np.array_equal(vertices_wp.numpy()[:n_vertices], mesh_wp.points.numpy())


def test_split_faces_along_field_positive_side_is_the_clip(
    icosphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a parity assert: this pins the split to ``clip_mesh_with_field``, which carries the oracle.

    The clip is compared against ``clip_scalar`` and ``clip_closed_surface`` in this file; the split
    reuses its classifier and its windings but keeps both sides. So the positive side must be the
    clip's answer **exactly** -- same face count, same area -- and any divergence is the split's,
    since the clip's is the tested one.
    """
    mesh_tm, mesh_wp = icosphere
    isovalue = 0.1
    field_wp = _height_field(mesh_tm, str(mesh_wp.device))
    _whole_tm, positive_tm, _negative_tm = _split_sides(mesh_wp, field_wp, isovalue)
    clipped_tm = warp_to_trimesh(
        *od.intersection.clip_mesh_with_field(mesh_wp.points, mesh_wp.indices, field_wp, isovalue)
    )
    assert 0 < len(clipped_tm.faces) < len(mesh_tm.faces)  # non-vacuity
    assert len(positive_tm.faces) == len(clipped_tm.faces)
    assert np.isclose(positive_tm.area, clipped_tm.area, rtol=1e-6)


def test_split_faces_along_field_degenerate_level_sets(
    torus: tuple[tm.Trimesh, wp.Mesh], icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: the three level sets that cut nothing, each for a different reason.

    A level set that already **lies on mesh edges** cuts no face -- the ``torus`` fixture's vertex
    rings sit at exactly ``z = 0``, so the surface comes back byte-identical and only the labels are
    new. An isovalue **outside the field's range** puts every face on one side. And a field with a
    vertex *exactly* at the isovalue exercises the two-triangle class: a face with one corner on the
    level set splits along the segment from that corner to the single opposite crossing, so it emits
    two triangles rather than three, and emitting three would leave a zero-area sliver.
    """
    torus_tm, torus_wp = torus
    field_wp = _height_field(torus_tm, str(torus_wp.device))
    assert np.count_nonzero(np.abs(torus_tm.vertices[:, 2]) < 1e-9) > 0  # the rings really are at 0
    vertices_wp, faces_wp, positive_wp = od.intersection.split_faces_along_field(
        torus_wp.points, torus_wp.indices, field_wp, 0.0
    )
    assert np.array_equal(faces_wp.numpy(), torus_wp.indices.numpy())
    assert np.array_equal(vertices_wp.numpy(), torus_wp.points.numpy())
    assert 0 < int(positive_wp.numpy().sum()) < faces_wp.size // 3

    sphere_tm, sphere_wp = icosphere
    sphere_field_wp = _height_field(sphere_tm, str(sphere_wp.device))
    for isovalue in (float(sphere_tm.vertices[:, 2].max()) + 1.0, -10.0):
        _v, out_faces_wp, out_positive_wp = od.intersection.split_faces_along_field(
            sphere_wp.points, sphere_wp.indices, sphere_field_wp, isovalue
        )
        assert np.array_equal(out_faces_wp.numpy(), sphere_wp.indices.numpy())
        assert len(np.unique(out_positive_wp.numpy())) == 1

    # A vertex exactly on the level set: its incident faces take the two-triangle class, so the
    # face count grows by less than two per crossed face and no output triangle is degenerate.
    on_vertex = float(sphere_tm.vertices[17, 2])
    corner_v_wp, corner_f_wp, _ = od.intersection.split_faces_along_field(
        sphere_wp.points, sphere_wp.indices, sphere_field_wp, on_vertex
    )
    _normals_wp, areas_wp = od.triangles.face_normals_and_areas(corner_v_wp, corner_f_wp)
    areas_np = areas_wp.numpy()
    assert areas_np.min() > 0.0
    assert np.isclose(areas_np.sum(), sphere_tm.area, rtol=1e-5)
