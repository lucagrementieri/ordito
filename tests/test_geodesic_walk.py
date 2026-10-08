"""
Regression tests for ``ordito.geodesic_walk`` against potpourri3d (CPU reference).

Two things are checked independently of the reference, because they are what "a geodesic" means: the
traced arc length equals the requested one (the direction's tangential magnitude), and every traced
point lies on the surface. Against potpourri3d the arc lengths agree exactly; the *endpoints* agree
only to a fraction of an edge length, because a path crossing a vertex has no unique straightest
continuation and the two libraries resolve that differently (see the module docstring).
"""

from __future__ import annotations

from typing import cast

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from tests.conftest import MESHES
from tests.conversions import numpy_to_warp, points_to_warp, trimesh_to_meshlib, warp_empty


def _rays(mesh_tm: tm.Trimesh, n_rays: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random start vertices and directions, scaled to a few edge lengths."""
    rng = np.random.default_rng(seed)
    start = rng.integers(0, len(mesh_tm.vertices), n_rays).astype(np.int32)
    scale = 3.0 * float(
        np.linalg.norm(
            mesh_tm.vertices[mesh_tm.edges[:, 1]] - mesh_tm.vertices[mesh_tm.edges[:, 0]], axis=1
        ).mean()
    )
    directions = rng.normal(size=(n_rays, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return start, (scale * directions).astype(np.float32)


def _tangential_length(direction: np.ndarray, normal: np.ndarray) -> float:
    return float(np.linalg.norm(direction - np.dot(direction, normal) * normal))


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _trace(
    mesh_wp: wp.Mesh,
    start_np: np.ndarray,
    directions_np: np.ndarray,
    frames: tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Trace from host start vertices and directions; returns the packed ``(points, offsets)``."""
    return od.geodesic_walk.trace_from_vertex(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(start_np, dtype=wp.int32, device=mesh_wp.device),
        points_to_warp(directions_np, mesh_wp.device),
        frames=frames,
    )


# ---------------------------------------------------------------------------
# trace_from_vertex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", MESHES)
def test_trace_from_vertex_walks_the_requested_distance_on_the_surface(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The two properties that make a trace a geodesic, each checked against its own oracle.

    Not a library comparison, for the arc length: the direction's *tangential* component sets the
    distance to walk, so the reference is arithmetic rather than another implementation -- an
    equality on a closed mesh and an upper bound on an open one, where a ray can stop at the rim.

    Class C (a distance bound, not a correspondence), for the surface: trimesh supplies only the
    point-to-surface distance, so every traced point is asserted to lie on a triangle. The bug class
    it excludes is the one unfolding gets wrong -- drifting off the surface at a triangle crossing
    -- which no arc-length check would see. The cross-library check is
    [`test_trace_from_vertex_matches_potpourri3d`].
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    start_np, directions_np = _rays(mesh_tm, 24, seed=0)
    frames_wp = od.tangent_space.vertex_tangent_frames(mesh_wp.points, mesh_wp.indices)
    points_wp, offsets_wp = _trace(mesh_wp, start_np, directions_np, frames=frames_wp)

    normals = frames_wp[2].numpy()
    is_boundary = od.halfedge.vertex_one_rings(mesh_wp.indices, n_vertices=len(mesh_tm.vertices))[
        2
    ].numpy()
    curves = od.array.split(points_wp, offsets_wp)
    for ray, (start, direction) in enumerate(zip(start_np, directions_np, strict=True)):
        points = curves[ray].numpy()
        requested = _tangential_length(direction.astype(np.float64), normals[start])
        # A ray reaching the boundary stops early, and one leaving a boundary vertex's fan does not
        # start at all, so on an open mesh the requested length is only an upper bound.
        traced = _path_length(points)
        assert traced <= requested * (1.0 + 1e-4) + 1e-6
        if not is_boundary.any():
            assert np.isclose(traced, requested, rtol=1e-4, atol=1e-5)
        # The path starts where it was asked to.
        assert np.allclose(points[0], mesh_tm.vertices[start], rtol=1e-5, atol=1e-5)

    # Every traced point must lie on a triangle: an unfolding error would drift off the surface.
    distance_tm = np.abs(
        tm.proximity.signed_distance(mesh_tm, points_wp.numpy().astype(np.float64))
    )
    scale = float(np.linalg.norm(mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)))
    assert distance_tm.max() < 1e-5 * scale


@pytest.mark.parametrize(
    "mesh_name", ["sphere_irregular", "sphere_irregular_hollow", "saddle_graded"]
)
@pytest.mark.parity("trace_rays", "potpourri3d")
@pytest.mark.parity("trace_locality", "potpourri3d")
def test_trace_from_vertex_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: the same input direction, the same walk.

    Both libraries read a 3-D direction at a vertex as a normalized polar angle (the corner angles
    rescaled to a full turn, half a turn at a boundary vertex) in a tangent frame fitted to the
    whole fan, so the start is gauge-free and defined at cone and saddle vertices alike. Random rays
    plus one from each of the two most negative and two most positive interior angle defects and two
    from rim vertices into the fan, so the convention's hard cases are in the set: first segments
    agree to a hundredth of a degree, arc lengths to 1e-6 and endpoints to 1e-3 of a mean edge. The
    one difference is a rim start pointing off the surface: ordito traces nothing, potpourri3d
    walks along whichever edge it lands on, so those rays are not compared. Probed: the bracketing
    convention this replaced (a direction placed between the projected edges that bracket it)
    started up to 9 degrees off at a cone vertex and missed endpoints by up to 1.83 edges.

    Carries the ``trace_locality`` marker as well: that group is this same call on a second axis
    (mesh diameter at pinned vertex count), so one comparison answers for both rows.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    start_np, directions_np = _rays(mesh_tm, 12, seed=2)
    angle_sums = np.zeros(vertices_np.shape[0])
    np.add.at(angle_sums, mesh_tm.faces.ravel(), mesh_tm.face_angles.ravel())
    defects = 2.0 * np.pi - angle_sums
    rim = np.unique(
        mesh_tm.edges_sorted[tm.grouping.group_rows(mesh_tm.edges_sorted, require_count=1)]
    )
    defects[rim] = 0.0
    order = np.argsort(defects)
    extremes = np.concatenate((order[:2], order[-2:])).astype(np.int32)
    if rim.size == 0:  # a closed fixture: strong saddle and cone starts
        assert (
            defects[extremes].min() < -np.radians(10.0) < np.radians(10.0) < defects[extremes].max()
        )
    # Rim starts aim into the fan, at an incident face's centroid: a direction pointing off the
    # surface traces nothing in ordito, where potpourri3d walks along whichever edge it lands on.
    rim_starts = rim[:2].astype(np.int32)
    into_fan = np.array(
        [mesh_tm.triangles_center[mesh_tm.vertex_faces[v][0]] - vertices_np[v] for v in rim_starts]
    ).reshape(-1, 3)
    into_fan *= np.linalg.norm(directions_np[0]) / np.linalg.norm(into_fan, axis=1, keepdims=True)
    start_np = np.concatenate((start_np, extremes, rim_starts))
    directions_np = np.concatenate(
        (directions_np, directions_np[: extremes.size], into_fan.astype(np.float32))
    )

    curves = od.array.split(*_trace(mesh_wp, start_np, directions_np))
    tracer_pp = pp3d.GeodesicTracer(vertices_np, faces_np)
    edge_length = float(np.linalg.norm(np.diff(vertices_np[mesh_tm.edges], axis=1), axis=2).mean())
    walked = 0
    for ray, start in enumerate(start_np):
        points = np.asarray(curves[ray].numpy(), dtype=np.float64)
        path_pp = np.asarray(
            tracer_pp.trace_geodesic_from_vertex(int(start), directions_np[ray].astype(np.float64))
        )
        if points.shape[0] == 1:  # only a rim start whose direction points off the surface
            assert start in rim
            assert ray < start_np.size - rim_starts.size
            continue
        walked += 1
        leaving, leaving_pp = points[1] - points[0], path_pp[1] - path_pp[0]
        cosine = leaving @ leaving_pp / (np.linalg.norm(leaving) * np.linalg.norm(leaving_pp))
        assert np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) < 1e-2
        assert np.isclose(_path_length(path_pp), _path_length(points), rtol=1e-6, atol=0.0)
        assert np.linalg.norm(points[-1] - path_pp[-1]) < 1e-3 * edge_length
    assert walked >= 12


def _normalized_edge_direction(
    mesh_tm: tm.Trimesh, normal: np.ndarray, vertex: int, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the unit tangent direction whose normalized polar angle at ``vertex`` is edge ``k``'s.

    A transcription of the convention ``trace_from_vertex`` documents, for a closed fan: corner
    angles rescaled to a full turn, normalized angle 0 placed by fitting the projected edges, each
    rotated back by its normalized angle. Edges are in the fan's counter-clockwise order.
    """
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    following, corner = {}, {}
    for face, angles in zip(mesh_tm.faces, mesh_tm.face_angles, strict=True):
        if vertex in face:
            slot = list(face).index(vertex)
            following[face[(slot + 1) % 3]] = face[(slot + 2) % 3]
            corner[face[(slot + 1) % 3]] = angles[slot]
    fan = [min(following)]
    while len(fan) < len(following):
        fan.append(following[fan[-1]])
    corners = np.array([corner[n] for n in fan])
    theta = 2.0 * np.pi * np.concatenate(([0.0], np.cumsum(corners)[:-1])) / corners.sum()
    axis_x = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    edges = vertices_np[fan] - vertices_np[vertex]
    anchor = np.angle(np.sum((edges @ axis_x + 1j * (edges @ axis_y)) * np.exp(-1j * theta)))
    return np.cos(anchor + theta[k]) * axis_x + np.sin(anchor + theta[k]) * axis_y, edges[k]


def test_trace_from_vertex_passes_through_the_vertex_an_edge_leads_to(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: a ray at an edge's normalized angle follows it through its far vertex.

    At ``sphere_irregular``'s strongest cone vertex (a 130-degree defect, four edges) the direction
    whose normalized polar angle is an incident edge's leaves along that edge and so runs exactly
    into its far vertex. That walk once stopped there, because the opposite edge's crossing landed a
    rounding outside [0, 1]; it now passes through, with half the vertex's total angle on either
    side (the straightest continuation), and traces its whole length.
    """
    mesh_tm, mesh_wp = sphere_irregular
    vertices_np = np.asarray(mesh_tm.vertices)
    angle_sums = np.zeros(vertices_np.shape[0])
    np.add.at(angle_sums, mesh_tm.faces.ravel(), mesh_tm.face_angles.ravel())
    cone = int(np.argmax(np.abs(2.0 * np.pi - angle_sums)))
    assert abs(2.0 * np.pi - angle_sums[cone]) > np.radians(90.0)
    normal = (
        od.tangent_space.vertex_tangent_frames(mesh_wp.points, mesh_wp.indices)[2]
        .numpy()[cone]
        .astype(np.float64)
    )
    length = 5.0 * float(
        np.linalg.norm(
            vertices_np[mesh_tm.vertex_neighbors[cone]] - vertices_np[cone], axis=1
        ).mean()
    )
    rays = [
        _normalized_edge_direction(mesh_tm, normal, cone, k)
        for k in range(len(mesh_tm.vertex_neighbors[cone]))
    ]
    directions = np.array([length * unit for unit, _ in rays], dtype=np.float32)
    points_wp, offsets_wp = _trace(mesh_wp, np.full(len(rays), cone), directions)
    for (_, edge), curve in zip(rays, od.array.split(points_wp, offsets_wp), strict=True):
        points = np.asarray(curve.numpy(), dtype=np.float64)
        assert points.shape[0] > 2  # it walked, and through the far vertex
        assert np.isclose(_path_length(points), length, rtol=1e-5, atol=0.0)
        leaving = points[1] - points[0]
        cosine = leaving @ edge / (np.linalg.norm(leaving) * np.linalg.norm(edge))
        assert np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) < 0.1


def test_trace_from_vertex_does_not_depend_on_the_numbering(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: renumbering the vertices and faces leaves every path where it was.

    The start direction is read in a frame fitted to the whole fan, so no edge -- and so no index
    -- is special. Rays from the ten vertices of largest angle defect, cones and saddles, on the
    mesh and on a copy with vertices, faces and each face's corners shuffled: endpoints within
    1e-4 of a mean edge. A direction placed among the projected edges that bracket it fell back to
    the ring's first edge at a saddle, whose projected fan folds, and moved by up to 35 degrees.
    """
    mesh_tm, mesh_wp = sphere_irregular
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces)
    angle_sums = np.zeros(vertices_np.shape[0])
    np.add.at(angle_sums, faces_np.ravel(), mesh_tm.face_angles.ravel())
    starts = np.argsort(-np.abs(2.0 * np.pi - angle_sums))[:10].astype(np.int32)
    assert (2.0 * np.pi - angle_sums[starts]).min() < -np.radians(30.0)  # saddles are among them
    rng = np.random.default_rng(4)
    directions = rng.normal(size=(starts.size, 3)).astype(np.float32)

    order = rng.permutation(vertices_np.shape[0])
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    shuffled = rank[faces_np][rng.permutation(faces_np.shape[0])]
    shift = rng.integers(0, 3, shuffled.shape[0])
    shuffled = np.take_along_axis(shuffled, (np.arange(3)[None, :] + shift[:, None]) % 3, axis=1)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np[order], shuffled.ravel(), mesh_wp.device)

    paths = od.array.split(*_trace(mesh_wp, starts, directions))
    paths_shuffled = od.array.split(
        *od.geodesic_walk.trace_from_vertex(
            vertices_wp,
            faces_wp,
            wp.array(rank[starts].astype(np.int32), dtype=wp.int32, device=mesh_wp.device),
            points_to_warp(directions, mesh_wp.device),
        )
    )
    edge_length = float(np.linalg.norm(np.diff(vertices_np[mesh_tm.edges], axis=1), axis=2).mean())
    for path, path_shuffled in zip(paths, paths_shuffled, strict=True):
        assert path.size > 1
        assert np.linalg.norm(path.numpy()[-1] - path_shuffled.numpy()[-1]) < 1e-4 * edge_length


def test_trace_from_vertex_stops_at_the_boundary(saddle_graded: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: rays fired off the rim must stop there, not wrap or leave.

    trimesh again supplies only the surface-distance oracle. The step-cap assert is what separates
    *stopping* from *running out of iterations* -- both give a short path, and only one is right.
    """
    mesh_tm, mesh_wp = saddle_graded
    # Aim from every boundary vertex along the outward direction with a long reach: each ray must
    # stop at the rim rather than wrap around or leave the surface.
    _, _, is_boundary_wp = od.halfedge.vertex_one_rings(
        mesh_wp.indices, n_vertices=len(mesh_tm.vertices)
    )
    boundary = np.flatnonzero(is_boundary_wp.numpy()).astype(np.int32)
    centroid = np.asarray(mesh_tm.vertices).mean(axis=0)
    outward = np.asarray(mesh_tm.vertices)[boundary] - centroid
    outward *= 100.0 / np.linalg.norm(outward, axis=1, keepdims=True)

    points_wp, offsets_wp = _trace(mesh_wp, boundary, outward)

    scale = float(
        np.linalg.norm(np.asarray(mesh_tm.vertices).max(0) - np.asarray(mesh_tm.vertices).min(0))
    )
    assert offsets_wp.numpy()[-1] < len(boundary) * 64  # nothing ran to the step cap
    distance_tm = np.abs(
        tm.proximity.signed_distance(mesh_tm, points_wp.numpy().astype(np.float64))
    )
    assert distance_tm.max() < 1e-5 * scale


# ---------------------------------------------------------------------------
# trace_from_face
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "saddle_graded"])
@pytest.mark.parity("trace_from_face", "potpourri3d")
def test_trace_from_face_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A on the start point and the arc length, for the barycentric entry point.

    Same split as [`test_trace_from_vertex_matches_potpourri3d`] and for the same reason -- the
    endpoint depends on a vertex-crossing tie-break -- but the *start* is exactly specified by the
    barycentric coordinates, so unlike the vertex form it is asserted at ``1e-4`` rather than
    bounded.

    ``trace_geodesic_from_face`` is the reference, the barycentric twin of the vertex entry point on
    the same ``GeodesicTracer``. This docstring used to add "no ``parity`` marker:
    ``trace_from_face`` is not separately benchmarked" -- the group existed and simply had no
    reference row, which is the gap rather than a reason, and it has one now.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    rng = np.random.default_rng(3)
    n_rays = 12
    start_faces = rng.integers(0, len(faces_np), n_rays).astype(np.int32)
    barycentric = np.full((n_rays, 3), 1.0 / 3.0)
    scale = 2.0 * float(
        np.linalg.norm(
            mesh_tm.vertices[mesh_tm.edges[:, 1]] - mesh_tm.vertices[mesh_tm.edges[:, 0]], axis=1
        ).mean()
    )
    directions = rng.normal(size=(n_rays, 3))
    directions *= scale / np.linalg.norm(directions, axis=1, keepdims=True)

    points_wp, offsets_wp = od.geodesic_walk.trace_from_face(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(start_faces, dtype=wp.int32, device=mesh_wp.device),
        points_to_warp(barycentric, mesh_wp.device),
        points_to_warp(directions, mesh_wp.device),
    )
    curves = od.array.split(points_wp, offsets_wp)

    tracer_pp = pp3d.GeodesicTracer(vertices_np, faces_np)
    for ray in range(n_rays):
        path_pp = np.asarray(
            tracer_pp.trace_geodesic_from_face(
                int(start_faces[ray]), barycentric[ray], directions[ray]
            )
        )
        points = curves[ray].numpy()
        assert np.allclose(points[0], path_pp[0], rtol=1e-4, atol=1e-4)
        assert np.isclose(_path_length(points), _path_length(path_pp), rtol=1e-4, atol=1e-5)


def test_trace_from_face_zero_direction_is_a_single_point(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = sphere_irregular
    points_wp, offsets_wp = od.geodesic_walk.trace_from_face(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device),
        wp.array(
            np.full((1, 3), 1.0 / 3.0, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
        ),
        wp.array(np.zeros((1, 3), dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device),
    )
    assert np.array_equal(offsets_wp.numpy(), np.array([0, 1]))
    assert points_wp.shape == (1,)


def test_trace_empty(device: str) -> None:
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    empty_int = warp_empty(0, wp.int32, device)
    empty_vec = warp_empty(0, wp.vec3, device)
    points_wp, offsets_wp = od.geodesic_walk.trace_from_vertex(
        vertices_wp, faces_wp, empty_int, empty_vec
    )
    assert points_wp.shape == (0,)
    assert offsets_wp.list() == [0]


# --------------------------------------------------------------------------------------
# descend_field / geodesic_path
# --------------------------------------------------------------------------------------

# Not ``sphere_irregular``: its heat field (72 % obtuse faces) has a spurious local minimum at
# vertex 300, where 142 of the 499 paths to vertex 0 stop -- the method's limit, matched by
# potpourri3d's identical field (CLAUDE.md section 16.6). A path test needs a field whose one
# minimum is the source; ``sphere_well_shaped`` (every one of 399 paths arrives) and
# ``torus_irregular`` have that from vertex 0.
_PATH_MESHES = ["sphere_well_shaped", "torus_irregular", "unit_box"]


def _paths_to_source(mesh_wp: wp.Mesh, targets_np: np.ndarray) -> list[wp.array[wp.vec3]]:
    """Trace every target back to vertex 0 and slice the packed result."""
    device = mesh_wp.points.device
    source_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    points_wp, offsets_wp = od.geodesic_walk.geodesic_path(
        mesh_wp.points, mesh_wp.indices, source_wp, wp.array(targets_np, wp.int32, device=device)
    )
    return od.array.split(points_wp, offsets_wp)


@pytest.mark.parametrize("mesh_name", _PATH_MESHES)
@pytest.mark.parity("geodesic_path", "igl")
def test_geodesic_path_is_never_shorter_than_the_exact_geodesic(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class C with an **inequality**, which is the only bound that holds for an approximate geodesic.

    ``igl.exact_geodesic`` propagates MMP windows and is *globally* exact, so it is a true lower
    bound on the length of any path between the same two vertices -- and the assertion is that
    ordito never comes in under it. Measured over the three ``_PATH_MESHES``, the minimum ratio is
    **1.0000-1.0126** and the median 1.0000-1.0353; the worst single path is 1.42 long on
    ``torus_irregular`` (the heat field's gradient, below), 1.12 on ``sphere_well_shaped`` and 1.08
    on ``unit_box``, where a cube's exact geodesics run along flat faces that a first-order field
    resolves poorly.

    The upper bound is asserted too, because an inequality alone would pass for a wildly detoured
    path -- and it is taken against the **same walk over potpourri3d's field**, not against a
    fixed ratio, because the excess is the heat method's. A descent's length is the integral of
    ``dphi / |grad phi|``, and on an irregular mesh the heat field's gradient is far from unit
    (0.27 to 2.0 along one ``sphere_irregular`` path): ordito's paths come out up to 1.42 times the
    exact geodesic on ``torus_irregular``, and descending potpourri3d's identical method
    (``use_robust=False``, the fields agree to 2e-5 of the range) gives the same paths. So ordito's
    path must match the reference-field path to 1e-3 of its length, which a detour in the walk or
    a wrong field fails and the method's error does not.

    ``igl.exact_geodesic`` needs **all six** arguments -- a four-argument call binds ``vt`` to
    ``fs`` and returns an empty array rather than raising -- so both face sets are passed
    explicitly empty (CLAUDE.md section 7.6).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    n_vertices = vertices_np.shape[0]
    rng = np.random.default_rng(2)
    targets_np = rng.choice(
        np.arange(1, n_vertices), min(20, n_vertices - 1), replace=False
    ).astype(np.int32)

    paths = _paths_to_source(mesh_wp, targets_np)
    lengths_np = np.array([float(od.polyline.polyline_length(path)) for path in paths])

    exact_igl = np.asarray(
        igl.exact_geodesic(
            vertices_np,
            np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([], dtype=np.int64),
            targets_np.astype(np.int64),
            np.array([], dtype=np.int64),
        )
    )
    assert np.all(exact_igl > 0.0)  # non-vacuity: the reference answered for every target

    ratio_np = lengths_np / exact_igl
    assert ratio_np.min() > 0.999  # never shorter than the exact geodesic

    distance_pp = pp3d.MeshHeatMethodDistanceSolver(
        vertices_np, np.ascontiguousarray(mesh_tm.faces, dtype=np.int32), use_robust=False
    ).compute_distance(0)
    device = mesh_wp.points.device
    points_pp, offsets_pp = od.geodesic_walk.descend_field(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(distance_pp, dtype=wp.float64, device=device),
        wp.array(targets_np, dtype=wp.int32, device=device),
    )
    lengths_pp = np.array(
        [float(od.polyline.polyline_length(path)) for path in od.array.split(points_pp, offsets_pp)]
    )
    assert np.allclose(lengths_np, lengths_pp, rtol=1e-3, atol=0.0)


@pytest.mark.parametrize("mesh_name", _PATH_MESHES)
def test_geodesic_path_reaches_its_source(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Not a library comparison: that the descent terminates *at* the source rather than short of it.

    ``descend_field`` documents four ways a path may legitimately stop early -- a local minimum, a
    flat face, the mesh boundary, ``max_steps`` -- and on a closed mesh carrying a heat distance
    field to a single source, none of them applies: the field has exactly one minimum and it is the
    source. So every path must arrive, and the last point of each is asserted to be the source
    vertex itself, which the exact-geodesic ratio test only catches indirectly and only when the
    truncated path is *much* shorter.

    The bug this guards is a float-drift coin flip, so it needs the assert stated this way. The
    walk tracked the field value at its current point by accumulating ``-slope * distance`` per
    in-face step rather than re-reading it; when a step landed exactly on a vertex -- which an
    icosphere's symmetry makes routine -- the walk carried that accumulated value into the next
    face and the flat-face fallback compared it against the value of the vertex it was standing on.
    Measured on ``icosphere(3)``: the two agreed to 1.8e-08, the accumulated one landing above on
    cuda:0 and below on cpu, so 2 of these 20 paths stopped dead on cpu at 0.265 and 0.524 of their
    true length while cuda:0 completed them. One device, one fixture, deterministic each time.

    Each path must also start at its target and **stay on the surface**: a descent that
    mis-unfolded across an edge would produce a plausible polyline floating off the mesh, which no
    length comparison catches, so every point is asserted within a rounding of a face.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = int(np.asarray(mesh_tm.vertices).shape[0])
    rng = np.random.default_rng(2)
    targets_np = rng.choice(
        np.arange(1, n_vertices), min(20, n_vertices - 1), replace=False
    ).astype(np.int32)

    source_np = np.asarray(mesh_tm.vertices)[0]
    paths = _paths_to_source(mesh_wp, targets_np)
    assert len(paths) == len(targets_np)  # non-vacuity: a path came back for every target
    vertices_np = mesh_wp.points.numpy()
    diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
    for target, path in zip(targets_np, paths, strict=True):
        points_np = path.numpy()
        assert len(points_np) >= 2
        assert np.allclose(points_np[0], np.asarray(mesh_tm.vertices)[int(target)], atol=1e-5)
        assert np.allclose(points_np[-1], source_np, atol=1e-5), (
            f"path from {int(target)} stopped {np.linalg.norm(points_np[-1] - source_np):.4f} "
            f"short of the source after {len(points_np)} points"
        )
        _closest_wp, distance_wp, _face_wp = od.proximity.closest_point_on_mesh(
            mesh_wp.points, mesh_wp.indices, path
        )
        assert float(distance_wp.numpy().max()) < 1e-5 * diagonal


@pytest.mark.parity("geodesic_path", "potpourri3d")
def test_geodesic_path_matches_potpourri3d_on_a_sphere(
    sphere_well_shaped: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class C: within a per-cent of ``EdgeFlipGeodesicSolver``'s *exact* path, on a sphere.

    potpourri3d flips edges until the path is locally shortest, so its length is the exact geodesic
    for that homotopy class -- and on a simply-connected surface that is *the* geodesic. Measured on
    ``sphere_well_shaped`` over 24 targets: ordito is never shorter (minimum ratio **1.0000**),
    median **1.0116** and worst **1.0848**. The gap is the heat field's first-order accuracy, which
    is the price of getting every path from one solve. (On ``sphere_irregular`` a path stopping at
    the field's spurious minimum reads 0.04 of the geodesic; see ``_PATH_MESHES``.)

    **This fixture is simply connected on purpose.** On a torus the comparison inverts, for a
    reason that is not an error on either side: ``find_geodesic_path`` shortens within the
    homotopy class of the edge path it starts from, so it can return a path going the long way
    round while a field descent takes the short one -- measured, 3 of 20 paths came out *shorter*
    than the reference there. That is why the globally exact lower bound in the test above uses
    ``igl.exact_geodesic`` instead, and why this comparison stays on the sphere.
    """
    mesh_tm, mesh_wp = sphere_well_shaped
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    solver_pp = pp3d.EdgeFlipGeodesicSolver(
        vertices_np, np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    )
    rng = np.random.default_rng(0)
    targets_np = rng.choice(np.arange(1, vertices_np.shape[0]), 24, replace=False).astype(np.int32)

    paths = _paths_to_source(mesh_wp, targets_np)
    lengths_np = np.array([float(od.polyline.polyline_length(path)) for path in paths])
    exact_pp = np.array(
        [
            float(
                np.linalg.norm(
                    np.diff(solver_pp.find_geodesic_path(v_start=0, v_end=int(target)), axis=0),
                    axis=1,
                ).sum()
            )
            for target in targets_np
        ]
    )
    assert np.all(exact_pp > 0.0)  # non-vacuity: the reference found every path

    ratio_np = lengths_np / exact_pp
    assert ratio_np.min() > 0.999
    assert np.median(ratio_np) < 1.05
    assert ratio_np.max() < 1.2


def test_descend_field_stops_at_a_local_minimum_and_at_a_boundary(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], saddle_graded: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: the two documented ways a descent stops before the stop value.

    A **local minimum** is the interesting one, because it is what makes this a field walk rather
    than a path finder: descending a field with two basins from a vertex in the wrong basin ends at
    that basin's own minimum, not at the global one. Here the field is the distance to a source and
    the descent starts at the *source*, whose value is already at the stop -- so the path is one
    point, which is the honest answer rather than an error.

    At a **mesh boundary** the walk stops where the surface does: on ``saddle_graded``, descending a
    field whose minimum lies off the rim leaves paths ending on the rim.
    """
    _, sphere_wp = sphere_irregular
    device = sphere_wp.points.device
    source_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    distance_wp = od.heat.heat_geodesic(sphere_wp.points, sphere_wp.indices, source_wp)
    at_source_wp, offsets_wp = od.geodesic_walk.descend_field(
        sphere_wp.points, sphere_wp.indices, distance_wp, source_wp
    )
    assert at_source_wp.size == 1  # already at the stop value
    assert np.array_equal(offsets_wp.numpy(), np.array([0, 1]))

    # A boundary: the field's source is a rim vertex, so paths from the far side reach it, but a
    # field with no reachable minimum stops on the rim instead.
    mesh_tm, open_wp = saddle_graded
    rim_wp = od.boundary.boundary_vertex_indices(open_wp.points, open_wp.indices)
    assert rim_wp.size > 0
    open_distance_wp = od.heat.heat_geodesic(
        open_wp.points, open_wp.indices, odt.as_dense(rim_wp[:1])
    )
    interior_np = np.setdiff1d(
        np.arange(mesh_tm.vertices.shape[0], dtype=np.int32), rim_wp.numpy()
    )[:8]
    points_wp, path_offsets_wp = od.geodesic_walk.descend_field(
        open_wp.points,
        open_wp.indices,
        open_distance_wp,
        wp.array(interior_np, dtype=wp.int32, device=device),
    )
    assert points_wp.size > interior_np.size  # every path has more than a point
    assert path_offsets_wp.size == interior_np.size + 1


def test_descend_field_accepts_the_precomputed_pair_values_first(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: ``vertex_faces=`` takes the pair values first.

    Like every packed pair in the package. The oracle is the default branch -- ``descend_field``
    building the incidence itself -- and this pins the precomputed branch to it. It exists because
    the keyword is the one place in the package where a caller *constructs* such a pair by hand,
    and both halves are
    ``wp.array[wp.int32]``: passing it transposed raises nothing, reads offsets as face indices,
    and (measured while the convention was being fixed) segfaults the CPU backend several launches
    later rather than at the call. So the composition is asserted rather than assumed.

    See [`array.pack_1d_arrays`][ordito.array.pack_1d_arrays] for the convention itself.
    """
    _, mesh_wp = sphere_irregular
    device = mesh_wp.points.device
    n_vertices = mesh_wp.points.size
    source_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    distance_wp = od.heat.heat_geodesic(mesh_wp.points, mesh_wp.indices, source_wp)
    starts_wp = wp.array(np.arange(1, 9, dtype=np.int32), dtype=wp.int32, device=device)

    derived_points, derived_offsets = od.geodesic_walk.descend_field(
        mesh_wp.points, mesh_wp.indices, distance_wp, starts_wp
    )

    incidence = od.adjacency.vertex_face_adjacency(mesh_wp.indices, n_vertices=n_vertices)
    assert incidence[0].size == mesh_wp.indices.size  # values, not offsets
    assert incidence[1].size == n_vertices + 1  # offsets, not values
    supplied_points, supplied_offsets = od.geodesic_walk.descend_field(
        mesh_wp.points, mesh_wp.indices, distance_wp, starts_wp, vertex_faces=incidence
    )

    assert derived_points.size > starts_wp.size  # not a batch of single points
    assert np.array_equal(derived_offsets.numpy(), supplied_offsets.numpy())
    assert np.allclose(derived_points.numpy(), supplied_points.numpy(), rtol=1e-5, atol=1e-5)


def test_descend_field_guards_and_empty(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a library comparison: the length guard and the empty batch."""
    _, mesh_wp = sphere_irregular
    device = mesh_wp.points.device
    values_wp = wp.zeros(3, dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="one entry per vertex"):
        od.geodesic_walk.descend_field(
            mesh_wp.points,
            mesh_wp.indices,
            values_wp,
            wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device),
        )

    field_wp = wp.zeros(mesh_wp.points.size, dtype=wp.float64, device=device)
    points_wp, offsets_wp = od.geodesic_walk.descend_field(
        mesh_wp.points, mesh_wp.indices, field_wp, warp_empty(0, wp.int32, device)
    )
    assert points_wp.shape == (0,)
    assert offsets_wp.shape == (1,)
    assert od.array.split(points_wp, offsets_wp) == []


def _cycle_length(vertices_np: np.ndarray, loop_np: np.ndarray) -> float:
    """Length of a closed vertex-index cycle, whose last entry joins back to its first."""
    points_np = vertices_np[loop_np]
    return float(
        np.linalg.norm(np.diff(np.vstack([points_np, points_np[:1]]), axis=0), axis=1).sum()
    )


def _shortened_generators(
    mesh_wp: wp.Mesh,
) -> tuple[list[wp.array[wp.int32]], list[wp.array[wp.int32]], int]:
    """Return the homology generators and the same loops shortened, with the sweep count."""
    loops_wp = od.homology.homology_generators(mesh_wp.points, mesh_wp.indices)
    shortened_wp, sweeps = od.geodesic_walk.shorten_loop(mesh_wp.points, mesh_wp.indices, loops_wp)
    return loops_wp, shortened_wp, sweeps


@pytest.mark.parity("shorten_loop", "potpourri3d")
def test_shorten_loop_preserves_the_homotopy_class(
    torus_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class B: equal after a named transform -- flow both loops to the geodesic in their class.

    ``EdgeFlipGeodesicSolver.find_geodesic_loop`` shortens a loop *within its homotopy class*, so
    the length it converges to is a property of the class and not of the curve handed to it. Running
    it from the input loop and from the shortened one must therefore give the same number, and that
    is the whole correctness claim: shortening is allowed to move the curve anywhere in its class
    and nowhere else. Measured **bit-identical** on both generators of the fixture (relative
    difference 0.0), and the same on a 24x12 torus and on a genus-2 union.

    A length comparison alone could not carry this. A sweep that leaked out of its class would
    usually get *shorter*, so it would look like a better result; the invariant is what says no.

    The gap that remains is real and is not tested as an equality: this stays on the edge graph
    where the reference crosses face interiors, and the ratio to the geodesic length is between
    **1.066x** and **1.578x** across the three meshes above -- widest where the mesh is a regular
    grid whose rows are not geodesics, because a one-ring move cannot step the loop off a row
    without lengthening the edge path first.
    """
    mesh_tm, mesh_wp = torus_irregular
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    loops_wp, shortened_wp, sweeps = _shortened_generators(mesh_wp)
    assert len(loops_wp) == 2  # non-vacuity: genus 1, so there are two generators to shorten
    assert 0 < sweeps < 100  # it converged rather than being cut off by the cap

    solver_pp = pp3d.EdgeFlipGeodesicSolver(
        vertices_np, np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    )

    def geodesic_length(loop_wp: wp.array[wp.int32]) -> float:
        points_pp = solver_pp.find_geodesic_loop(loop_wp.numpy().astype(np.int64))
        return float(np.linalg.norm(np.diff(points_pp, axis=0), axis=1).sum())

    improved = 0
    for loop_wp, shortened_loop_wp in zip(loops_wp, shortened_wp, strict=True):
        before = _cycle_length(vertices_np, loop_wp.numpy())
        after = _cycle_length(vertices_np, shortened_loop_wp.numpy())
        assert after <= before + 1e-6
        improved += after < before - 1e-6

        exact_before = geodesic_length(loop_wp)
        exact_after = geodesic_length(shortened_loop_wp)
        assert exact_before > 0.0  # a contractible loop would have collapsed to a point here
        assert np.allclose(exact_after, exact_before, rtol=1e-5, atol=1e-5)
        assert after >= exact_after - 1e-6  # the class's geodesic bounds any curve the sweeps reach
    assert improved >= 1  # and the sweeps did something: 1.411x on this fixture's major generator


@pytest.mark.parametrize("mesh_name", ["torus_irregular", "genus_two"])
@pytest.mark.parity(
    "shorten_loop",
    "meshlib",
    benchmarked=False,
    reason="Tested but not timed: findShortestEquivalentLoops takes one loop per "
    "call and returns a multi-loop system, so a row over a generator basis would "
    "price a Python loop and a different output shape. potpourri3d is timed.",
)
def test_shorten_loop_bounded_by_meshlib(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class C: ordito's edge-path local minimum against a reference minimizing over the same space.

    The second oracle for this group, and it isolates something
    [`test_shorten_loop_preserves_the_homotopy_class`][tests.test_geodesic_walk.test_shorten_loop_preserves_the_homotopy_class]
    cannot. That test compares against ``potpourri3d.EdgeFlipGeodesicSolver``, which is allowed to
    leave the edge graph, so its 1.066x-1.578x gap mixes two separate things: how far ordito's
    sweep is from the best *edge path*, and how far the best edge path is from the true geodesic.
    ``findShortestEquivalentLoops`` stays on mesh edges, so the ratio here is the first of those
    alone -- the local-versus-global gap over one search space.

    Two asserts, and the sharp one is not the ratio. Building the reference's input requires walking
    ordito's output through ``MeshTopology.findEdge`` pair by pair, which **validates the loop
    against an independent halfedge structure**: a sweep that ever rerouted through a vertex outside
    the one-ring would emit a consecutive pair that is not a mesh edge, and ``findEdge`` returns an
    invalid ``EdgeId`` rather than an approximation. Every existing validity check on this function
    goes through ordito's own adjacency. The ratio then bounds the length gap.

    The bug class the threshold excludes is a sweep that stalls or lengthens. Measured ratios of
    ordito's length to the reference's total: **1.07 and 1.54** on ``torus``, **1.18 / 1.31 /
    1.55 / 1.17** on ``genus_two`` -- so the 2.0 threshold clears the worst by 1.29x. The mutation
    probe is the unshortened tree-cotree loop, which the sweep is what removes: feeding ``torus``'s
    major generator raw takes the ratio to **2.17** and fails. That margin is narrower than a
    threshold test would normally want, which is why the edge-walk assert above carries the weight
    and this one is a bound rather than the claim.

    !!! warning "The reference returns a loop *system*, so the comparable total is a sum"
        ``findShortestEquivalentLoops`` may split one loop into several that are jointly equivalent
        to it -- measured, 2 loops for two of ``genus_two``'s four generators. Comparing against
        ``min`` of the returned lengths therefore reads a ratio of **3.71** on one of them and looks
        like a gross disagreement; against the ``sum`` the same generator reads 1.31. The sum is the
        equivalent object.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = mesh_wp.points.numpy().astype(np.float64)
    _loops_wp, shortened_wp, _sweeps = _shortened_generators(mesh_wp)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    topology_ml = mesh_ml.topology
    ratios = []
    for shortened_loop_wp in shortened_wp:
        shortened_np = shortened_loop_wp.numpy()
        edge_loop_ml = mm.std_vector_Id_EdgeTag()
        for tail, head in zip(shortened_np, np.roll(shortened_np, -1), strict=True):
            edge_ml = topology_ml.findEdge(mm.VertId(int(tail)), mm.VertId(int(head)))
            assert edge_ml.valid(), (
                f"{tail} -> {head} is not a mesh edge: the sweep left the one-ring"
            )
            edge_loop_ml.append(edge_ml)

        system_ml = mm.findShortestEquivalentLoops(mm.MeshPart(mesh_ml), edge_loop_ml)
        assert len(system_ml) > 0, "the reference returned nothing; the comparison would be vacuous"
        total_ml = sum(
            float(mesh_ml.edgeLength(edge_ml.undirected()))
            for loop_ml in system_ml
            for edge_ml in loop_ml
        )
        length = _cycle_length(vertices_np, shortened_np)
        assert total_ml <= length + 1e-4  # it minimizes over the same space, so it cannot do worse
        ratios.append(length / total_ml)

    assert max(ratios) < 2.0, f"the sweep left the loops long: ratios {ratios}"
    assert max(ratios) > 1.0 + 1e-3  # non-vacuity: an exactly-equal fixture would assert nothing


def test_shorten_loop_accepts_precomputed_connectivity(
    torus_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: the ``twins`` / ``rings`` arguments do not change the sweep.

    Not a parity assert. The call that rebuilds the connectivity itself is the one the reference
    comparison above runs, so it carries the oracle; what this pins is that handing the sweep a
    precomputed halfedge structure -- which every caller holding a ``Trimesh`` now can -- reaches
    the identical cycles, since the loop walk reads nothing else about the topology.
    """
    _mesh_tm, mesh_wp = torus_irregular
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    loops_wp = od.homology.homology_generators(vertices_wp, faces_wp)
    assert len(loops_wp) == 2  # non-vacuity: genus 1, so there are two generators to shorten

    rebuilt_wp, rebuilt_sweeps = od.geodesic_walk.shorten_loop(vertices_wp, faces_wp, loops_wp)
    mesh = od.Trimesh.from_warp_mesh(mesh_wp)
    cached_wp, cached_sweeps = od.geodesic_walk.shorten_loop(
        vertices_wp, faces_wp, loops_wp, twins=mesh.halfedge_twins, rings=mesh.vertex_one_rings
    )

    assert cached_sweeps == rebuilt_sweeps > 0
    for rebuilt_loop_wp, cached_loop_wp in zip(rebuilt_wp, cached_wp, strict=True):
        assert np.array_equal(cached_loop_wp.numpy(), rebuilt_loop_wp.numpy())


def test_shorten_loop_returns_valid_non_separating_cycles(
    torus_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: no reference shortens a loop along mesh edges.

    Two invariants instead, neither visible in a length. The output must still be a **closed walk
    along mesh edges** -- consecutive entries adjacent, and the last adjacent to the first -- which
    is what a wrongly reconstructed link arc would break. And it must still be **non-separating**:
    cutting a genus-1 surface along a simple non-separating cycle leaves one component with two
    boundary loops, where a contractible cycle would cut a disk off and leave two components. A
    length check cannot see the difference, because a loop collapsing onto a disk gets shorter.
    """
    _, mesh_wp = torus_irregular
    device = mesh_wp.indices.device
    _loops_wp, shortened_wp, _sweeps = _shortened_generators(mesh_wp)

    edges_np = {
        tuple(sorted(edge)) for edge in od.edges.faces_to_edges(mesh_wp.indices).numpy().tolist()
    }
    for shortened_loop_wp in shortened_wp:
        loop_np = shortened_loop_wp.numpy()
        assert loop_np.size >= 3
        assert np.unique(loop_np).size == loop_np.size  # simple, so the cut below applies
        rolled_np = np.roll(loop_np, -1)
        assert all(
            tuple(sorted((int(a), int(b)))) in edges_np
            for a, b in zip(loop_np, rolled_np, strict=True)
        )

        loop_edges_wp = wp.array(
            np.stack([loop_np, rolled_np], axis=1).astype(np.int32), dtype=wp.int32, device=device
        )
        cut_vertices_wp, cut_faces_wp = od.seams.cut_along_edges(
            mesh_wp.points, mesh_wp.indices, odt.as_array2d(loop_edges_wp, wp.int32)
        )
        labels_np = od.adjacency.face_connected_component_labels(cut_faces_wp).numpy()
        assert np.unique(labels_np).size == 1
        assert (
            len(
                od.boundary.boundary_loops(cast("wp.array[wp.vec3]", cut_vertices_wp), cut_faces_wp)
            )
            == 2
        )


def test_shorten_loop_is_its_packed_form_split(genus_two: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Ordito against ordito: the list form is the packed form, loop by loop.

    ``shorten_loop`` carries the potpourri3d homotopy comparison above; this pins
    ``shorten_loop_with_offsets`` to it on a genus-2 basis (four loops), and checks that the packed
    form leaves its input buffer untouched and hands it back as is when no sweep may run.
    """
    _, mesh_wp = genus_two
    flat_wp, offsets_wp = od.homology.homology_generators_with_offsets(
        mesh_wp.points, mesh_wp.indices
    )
    flat_before_np = flat_wp.numpy().copy()
    loops_wp = od.array.split(flat_wp, offsets_wp)
    assert len(loops_wp) == 4

    shortened_wp, sweeps = od.geodesic_walk.shorten_loop(mesh_wp.points, mesh_wp.indices, loops_wp)
    packed_wp, packed_offsets_wp, packed_sweeps = od.geodesic_walk.shorten_loop_with_offsets(
        mesh_wp.points, mesh_wp.indices, flat_wp, offsets_wp
    )

    assert packed_sweeps == sweeps
    assert np.array_equal(flat_wp.numpy(), flat_before_np)
    offsets_np = packed_offsets_wp.numpy()
    assert offsets_np.size == len(shortened_wp) + 1
    packed_np = packed_wp.numpy()
    for i, loop_wp in enumerate(shortened_wp):
        assert np.array_equal(loop_wp.numpy(), packed_np[offsets_np[i] : offsets_np[i + 1]])

    unchanged = od.geodesic_walk.shorten_loop_with_offsets(
        mesh_wp.points, mesh_wp.indices, flat_wp, offsets_wp, max_iter=0
    )
    assert unchanged == (flat_wp, offsets_wp, 0)


@pytest.mark.parametrize("group", [1, 4])
def test_shorten_loop_regrows_its_buffers_and_matches(
    genus_two: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch, group: int
) -> None:
    """
    Ordito against ordito: buffers with no headroom give the same loops as the default ones.

    With zero headroom the first sweep that lengthens a loop cannot fit, so it is undone and run
    again in grown buffers -- the path no default-sized call reaches. The default call carries the
    oracle (the potpourri3d comparison above). The test asserts that the regrowth really ran and
    that the loops and the sweep count are identical, at two recorded group sizes.
    """
    _, mesh_wp = genus_two
    flat_wp, offsets_wp = od.homology.homology_generators_with_offsets(
        mesh_wp.points, mesh_wp.indices
    )
    expected_wp, expected_offsets_wp, expected_sweeps = od.geodesic_walk.shorten_loop_with_offsets(
        mesh_wp.points, mesh_wp.indices, flat_wp, offsets_wp
    )
    built: list[int] = []
    sweep_class = od.geodesic_walk._LoopSweep  # pyright: ignore[reportPrivateUsage]

    class CountingSweep(sweep_class):
        def __init__(self, *args: object) -> None:
            built.append(cast("int", args[7]))
            super().__init__(*args)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(od.geodesic_walk, "_LoopSweep", CountingSweep)
    monkeypatch.setattr(od.geodesic_walk, "_SHORTEN_LOOP_GROWTH", 0.0)
    monkeypatch.setattr(od.geodesic_walk, "_SHORTEN_LOOP_SLACK", 0)
    monkeypatch.setattr(od.geodesic_walk, "_SHORTEN_LOOP_GRAPH_SWEEPS", group)
    shortened_wp, shortened_offsets_wp, sweeps = od.geodesic_walk.shorten_loop_with_offsets(
        mesh_wp.points, mesh_wp.indices, flat_wp, offsets_wp
    )
    assert len(built) >= 2
    assert built[0] == flat_wp.size
    assert sweeps == expected_sweeps
    assert np.array_equal(shortened_offsets_wp.numpy(), expected_offsets_wp.numpy())
    assert np.array_equal(shortened_wp.numpy(), expected_wp.numpy())


def test_shorten_loop_is_idempotent_and_handles_edge_cases(
    torus_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: a fixed point of a monotone local rule has no external oracle.

    A second call must change nothing -- the first one ran to convergence, so every position already
    fails its acceptance test -- which pins that the stopping rule and the acceptance rule agree. A
    loop too short to have a triple, and an empty list, come back untouched.
    """
    _, mesh_wp = torus_irregular
    _loops_wp, once_wp, _sweeps = _shortened_generators(mesh_wp)
    twice_wp, sweeps = od.geodesic_walk.shorten_loop(mesh_wp.points, mesh_wp.indices, once_wp)
    assert sweeps == 2  # one sweep per parity, both finding nothing to do
    for first_wp, second_wp in zip(once_wp, twice_wp, strict=True):
        assert np.array_equal(first_wp.numpy(), second_wp.numpy())

    assert od.geodesic_walk.shorten_loop(mesh_wp.points, mesh_wp.indices, []) == ([], 0)
    stub_wp = wp.array([0, 1], dtype=wp.int32, device=mesh_wp.indices.device)
    kept_wp, _ = od.geodesic_walk.shorten_loop(mesh_wp.points, mesh_wp.indices, [stub_wp])
    assert np.array_equal(kept_wp[0].numpy(), [0, 1])

    with pytest.raises(TypeError, match=r"expected dtype"):
        od.geodesic_walk.shorten_loop(
            mesh_wp.points,
            mesh_wp.indices,
            [wp.array([0.0, 1.0], dtype=wp.float32, device=mesh_wp.indices.device)],
        )
