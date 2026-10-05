"""Regression tests for ``ordito.holes``."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import numpy as np
import open3d as o3d
import pytest
import trimesh as tm
import trimesh.repair as tm_repair
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import ordito as od
from ordito.holes import _non_increasing_indices  # pyright: ignore[reportPrivateUsage]
from ordito.kernels import holes as kernel_holes
from tests.comparisons import (
    boundary_loop_sizes,
    canonical_winding,
    hausdorff_surface_two_sided,
    lexsort_rows,
    same_partition,
)
from tests.conftest import OPEN_MESHES
from tests.conversions import (
    meshlib_bitset_to_numpy,
    meshlib_to_trimesh,
    numpy_to_meshlib,
    numpy_to_meshlib_bitset,
    numpy_to_warp,
    pymeshfix_to_numpy,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshfix,
    trimesh_to_pymeshlab,
    warp_empty,
    warp_to_trimesh,
)

if TYPE_CHECKING:
    from typing_extensions import Buffer


# Open-surface fixtures that actually have a boundary to fill.
def _fillable_loops(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> list[wp.array[wp.int32]]:
    return [loop for loop in od.boundary.boundary_loops(vertices, faces) if loop.size >= 3]


def _loop_sizes_of(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> list[int]:
    return [loop.size for loop in _fillable_loops(vertices, faces)]


def _loop_perimeters_of(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> list[float]:
    return [
        od.polyline.polyline_length(od.array.gather(vertices, loop), closed=True)
        for loop in _fillable_loops(vertices, faces)
    ]


def _loop_sizes(mesh_wp: wp.Mesh) -> list[int]:
    return _loop_sizes_of(mesh_wp.points, mesh_wp.indices)


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity(
    "fill_fan",
    "meshlib",
    benchmarked=False,
    reason="MeshLib has no fan fill. fillHoleTrivially adds an apex vertex, which makes it "
    "fill_cone's operation and it is timed in that group -- a row here would price a cone under "
    "the fan's name and buy an extra vertex per loop that the fan does not pay for.",
)
def test_fill_fan_covers_the_same_rim_as_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class C (two derived scalars): the same rim triangulated two ways covers the same surface.

    MeshLib has no fan: ``fillHoleTrivially`` puts a new vertex at the rim's centroid and fans from
    *that*, which is [`fill_cone`][ordito.holes.fill_cone]'s operation and is compared to it
    element-wise one test below. So the claim available for the fan is the one property two
    different triangulations of one **planar** rim must share -- they cover the same region -- and
    the comparable quantities are the added area and the enclosed volume.

    That makes it a sharper test than it sounds, because the agreement is at the float32 floor and
    nowhere near the tolerance: measured on a 97-vertex rim, the areas agree to better than
    **1e-9 relative** and the volumes to **5.9e-09 relative** -- the residual being the converter's
    float32 storage on MeshLib's side, since with both fed one identical float32 buffer the two
    areas land 2e-15 apart -- while the face counts differ by exactly 2 per loop (``B - 2`` fan
    triangles against ``B``) and the vertex counts by exactly 1. The mutation probe: dropping a
    single fan triangle changes the area by 0.0104, **1.1e-03 relative**, which is 2e5 x the bound
    asserted below.

    What it excludes is a fan that skips a rim vertex, emits a triangle twice, or walks the loop in
    an order that makes the cap self-overlap -- every one of which changes the covered area while
    leaving the triangle count right. What it cannot see is *which* triangulation was chosen, which
    is what the count assertions are for.

    Not a library comparison, the count half: a fan over a ``B``-vertex loop is ``B - 2`` triangles
    and adds no vertices -- arithmetic, not another implementation -- cross-checked against the
    count ``trimesh.repair.fill_holes(use_fan=True)`` adds. The input is asserted *not* watertight
    first, so the sealing claim cannot pass on a mesh that was already closed.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = mesh_wp.points.size
    loop_sizes = _loop_sizes(mesh_wp)

    assert not od.validation.is_watertight(mesh_wp.points, mesh_wp.indices)
    fan_faces_wp = od.holes.fill_fan(mesh_wp.points, mesh_wp.indices)
    fan_tm = warp_to_trimesh(mesh_wp.points, fan_faces_wp)

    assert od.validation.is_watertight(mesh_wp.points, fan_faces_wp)
    assert od.validation.is_winding_consistent(fan_faces_wp)
    n_new_faces = (fan_faces_wp.size - mesh_wp.indices.size) // 3
    assert n_new_faces == sum(size - 2 for size in loop_sizes)
    fan_ref_tm = mesh_tm.copy()
    tm_repair.fill_holes(fan_ref_tm, use_fan=True)
    assert n_new_faces == len(fan_ref_tm.faces) - len(mesh_tm.faces)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    apex_ids_ml = [
        mm.fillHoleTrivially(mesh_ml, edge_ml).get()
        for edge_ml in mesh_ml.topology.findHoleRepresentiveEdges()
    ]
    mesh_ml.pack()  # mandatory before reading topology back
    cone_ref_tm = tm.Trimesh(
        mn.getNumpyVerts(mesh_ml), mn.getNumpyFaces(mesh_ml.topology), process=False
    )

    # Non-vacuity: MeshLib filled every rim, and both closures are real surfaces.
    assert len(apex_ids_ml) == len(loop_sizes) > 0
    assert fan_tm.is_watertight
    assert cone_ref_tm.is_watertight
    assert fan_tm.area > 1.0

    # One apex and two extra triangles per loop, on MeshLib's side only.
    assert len(cone_ref_tm.vertices) == n_vertices + len(loop_sizes)
    assert len(cone_ref_tm.faces) == len(fan_tm.faces) + 2 * len(loop_sizes)

    # And with the rims planar, the two caps cover the same region.
    assert np.isclose(fan_tm.area, cone_ref_tm.area, rtol=1e-7, atol=1e-9)
    assert np.isclose(fan_tm.volume, cone_ref_tm.volume, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("fill_cone", "meshlib")
def test_fill_cone_matches_meshlib(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: ``fillHoleTrivially`` is the same cone -- same apex, same triangles, same winding.

    MeshLib's name says "trivially" where ordito's says "cone", but the operation is identical:
    one new vertex at the rim's centroid and one triangle per boundary edge. It returns the apex's
    ``VertId`` and writes the new faces into an optional ``FaceBitSet``, so both halves of the
    answer are directly readable rather than inferred from a count.

    Two conventions are named rather than assumed. The apex is compared at ``1e-6`` because
    MeshLib stores float32 and reads back float64, the same floor ordito's own buffer sets
    (measured 1.19e-07 here). And the triangles are compared through
    [`tests.comparisons.canonical_winding`][], which rotates each one onto its lowest index --
    the two libraries pick different *starting corners* but the same orientation, and rotation
    cannot hide a flip, so the winding stays under test.

    MeshLib fills one hole per call, so a multi-rim fixture is looped on that side; ``half_torus``
    exercises that with two rims, which is what makes the loop non-vacuous.

    Not a library comparison, the structural half: the result is watertight and consistently
    wound, adds one apex and ``B`` triangles per loop, references only valid vertices, and puts
    each apex at the mean of its rim's vertices (at ``1e-4``).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = mesh_wp.points.size
    n_faces = mesh_wp.indices.size
    loop_sizes = _loop_sizes(mesh_wp)
    rim_centroids_np = np.array(
        [
            mesh_wp.points.numpy()[loop_wp.numpy()].mean(axis=0)
            for loop_wp in _fillable_loops(mesh_wp.points, mesh_wp.indices)
        ]
    )

    new_vertices_wp, filled_faces_wp = od.holes.fill_cone(mesh_wp.points, mesh_wp.indices)

    assert od.validation.is_watertight(new_vertices_wp, filled_faces_wp)
    assert od.validation.is_winding_consistent(filled_faces_wp)
    assert new_vertices_wp.size == n_vertices + len(loop_sizes)
    assert (filled_faces_wp.size - n_faces) // 3 == sum(loop_sizes)
    filled_np = filled_faces_wp.numpy()
    assert filled_np.min() >= 0
    assert filled_np.max() < new_vertices_wp.size
    assert np.allclose(
        np.sort(new_vertices_wp.numpy()[n_vertices:], axis=0),
        np.sort(rim_centroids_np, axis=0),
        rtol=1e-4,
        atol=1e-4,
    )

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    new_faces_ml = mm.FaceBitSet()
    apex_ids_ml = [
        mm.fillHoleTrivially(mesh_ml, edge_ml, new_faces_ml).get()
        for edge_ml in mesh_ml.topology.findHoleRepresentiveEdges()
    ]
    faces_ml = mn.getNumpyFaces(mesh_ml.topology)[np.flatnonzero(mn.getNumpyBitSet(new_faces_ml))]

    # Non-vacuity, and the fixture check: a rim-less input would make every assert below trivial.
    assert len(apex_ids_ml) == new_vertices_wp.size - n_vertices > 0
    assert np.allclose(
        np.sort(mn.getNumpyVerts(mesh_ml)[apex_ids_ml], axis=0),
        np.sort(new_vertices_wp.numpy()[n_vertices:], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.array_equal(
        lexsort_rows(canonical_winding(filled_faces_wp.numpy()[n_faces:])),
        lexsort_rows(canonical_winding(faces_ml)),
    )


@pytest.mark.parametrize("method", ["fill_fan", "fill_cone", "fill_min_weight"])
def test_fill_preserve_largest(half_torus: tuple[tm.Trimesh, wp.Mesh], method: str) -> None:
    """
    Not a library comparison: ``preserve_largest_hole`` leaves exactly the longest rim open.

    The preserved loop is the one with the greatest *perimeter*, not the most vertices; every other
    hole is filled with the method's own count (``B - 2`` triangles for a fan or the min-weight
    DP, ``B`` triangles and one apex for a cone), and the one remaining rim has the preserved
    loop's perimeter.
    """
    _, mesh_wp = half_torus
    loop_sizes = _loop_sizes(mesh_wp)
    perimeters = _loop_perimeters_of(mesh_wp.points, mesh_wp.indices)
    # The fixture must have several holes for "preserve the largest" to be meaningful.
    assert len(loop_sizes) >= 2
    preserved = int(np.argmax(perimeters))
    preserved_size = loop_sizes[preserved]
    n_vertices_before = mesh_wp.points.size

    if method == "fill_cone":
        vertices_wp, filled_faces = od.holes.fill_cone(
            mesh_wp.points, mesh_wp.indices, preserve_largest_hole=True
        )
        # One centroid per filled hole; B triangles per filled hole.
        assert vertices_wp.size == n_vertices_before + len(loop_sizes) - 1
        expected_new_faces = sum(loop_sizes) - preserved_size
    else:
        fill = od.holes.fill_fan if method == "fill_fan" else od.holes.fill_min_weight
        vertices_wp = mesh_wp.points
        filled_faces = fill(mesh_wp.points, mesh_wp.indices, preserve_largest_hole=True)
        expected_new_faces = sum(size - 2 for size in loop_sizes) - (preserved_size - 2)

    assert (filled_faces.size - mesh_wp.indices.size) // 3 == expected_new_faces
    remaining_perimeters = _loop_perimeters_of(vertices_wp, filled_faces)
    assert len(remaining_perimeters) == 1
    assert np.isclose(remaining_perimeters[0], perimeters[preserved], rtol=1e-5, atol=1e-5)


def test_fill_holes_preserve_largest_single_hole_unchanged(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = hemisphere

    # With one hole, preserving the largest leaves nothing to fill.
    assert len(_loop_sizes(mesh_wp)) == 1

    filled_faces = od.holes.fill_fan(mesh_wp.points, mesh_wp.indices, preserve_largest_hole=True)
    assert np.array_equal(filled_faces.numpy(), mesh_wp.indices.numpy())

    new_vertices, cone_faces = od.holes.fill_cone(
        mesh_wp.points, mesh_wp.indices, preserve_largest_hole=True
    )
    assert np.array_equal(cone_faces.numpy(), mesh_wp.indices.numpy())
    assert new_vertices.size == mesh_wp.points.size


# --- Minimum-weight hole triangulation (``fill_min_weight``) ---------------------------

_BAD = 1e10  # kernel ``BAD_METRIC``


def _circumcircle_diameter(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = float(np.dot(b - a, b - a))
    ca = float(np.dot(a - c, a - c))
    bc = float(np.dot(c - b, c - b))
    if ab <= 0.0:
        return float(np.sqrt(ca))
    if ca <= 0.0:
        return float(np.sqrt(bc))
    if bc <= 0.0:
        return float(np.sqrt(ab))
    n = np.cross(b - a, c - a)
    f = float(np.dot(n, n))
    if f <= 0.0:
        return np.inf
    return float(np.sqrt(ab * ca * bc / f))


def _triangle_aspect_ratio(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    bc = float(np.linalg.norm(c - b))
    ca = float(np.linalg.norm(a - c))
    ab = float(np.linalg.norm(b - a))
    half_perimeter = (bc + ca + ab) * 0.5
    den = 8.0 * (half_perimeter - bc) * (half_perimeter - ca) * (half_perimeter - ab)
    if den <= 0.0:
        return np.inf
    return bc * ca * ab / den


FILL_METRICS = [
    "plane_normalized",
    "min_area",
    "circumscribed",
    "plane",
    "min_tri_angle",
    "edge_length",
    "universal",
    "max_dihedral",
    "complex_fill",
]
_COMBINE_MAX = {"max_dihedral"}  # metrics that accumulate with max instead of sum


def _min_triangle_angle_sin(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = float(np.linalg.norm(b - a))
    ca = float(np.linalg.norm(a - c))
    bc = float(np.linalg.norm(c - b))
    if ab <= 0.0 or ca <= 0.0 or bc <= 0.0:
        return 0.0
    return float(np.linalg.norm(np.cross(b - a, c - a)) * min(ab, ca, bc) / (ab * ca * bc))


def _dihedral(left: np.ndarray, right: np.ndarray, edge: np.ndarray) -> float:
    edge_dir = edge / np.linalg.norm(edge)
    return float(np.arctan2(np.dot(edge_dir, np.cross(left, right)), np.dot(left, right)))


def _tri_term(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, normal: np.ndarray, char_area: float, metric: str
) -> float:
    """Per-triangle term, pure-NumPy mirror of ``kernels.holes.triangle_fill_metric``."""
    a, b, c = (x.astype(np.float32) for x in (a, b, c))
    if metric == "min_area":
        return float(np.linalg.norm(np.cross(b - a, c - a)))
    if metric == "circumscribed" or metric == "universal":
        return _circumcircle_diameter(a, b, c)
    if metric == "min_tri_angle":
        return float(np.exp(25.0 * (0.86602540378443864676 - _min_triangle_angle_sin(a, b, c))))
    if metric == "edge_length" or metric == "max_dihedral":
        return 0.0
    if metric == "complex_fill":
        aspect_ratio = _triangle_aspect_ratio(a, b, c)
        if aspect_ratio > _BAD:
            return _BAD
        return aspect_ratio + 100.0 * float(np.linalg.norm(np.cross(b - a, c - a))) * char_area
    if metric == "plane":
        if float(np.dot(normal.astype(np.float32), np.cross(b - a, c - a))) < 0.0:
            return _BAD
        return _circumcircle_diameter(a, b, c)
    # plane_normalized
    face_norm = np.cross(b - a, c - a)
    face_dbl_area_sq = float(np.dot(face_norm, face_norm))
    if face_dbl_area_sq == 0.0:
        return _BAD
    dot_res = float(np.dot(normal.astype(np.float32), face_norm))
    if dot_res < 0.0 or dot_res * dot_res * 4.0 < face_dbl_area_sq:
        return _BAD
    aspect_ratio = _triangle_aspect_ratio(a, b, c)
    if aspect_ratio > _BAD:
        return _BAD
    return _circumcircle_diameter(a, b, c) * aspect_ratio


def _edge_term(
    a: np.ndarray, b: np.ndarray, lft: np.ndarray, rgt: np.ndarray, metric: str
) -> float:
    """Per-edge term, pure-NumPy mirror of ``kernels.holes.fill_edge_term``."""
    a, b, lft, rgt = (x.astype(np.float32) for x in (a, b, lft, rgt))
    if metric == "edge_length":
        return float(np.linalg.norm(b - a))
    ab = b - a
    if metric == "universal":
        norm_l = np.cross(lft - a, ab)
        norm_r = np.cross(ab, rgt - a)
        dbl_area = float(np.linalg.norm(norm_l) + np.linalg.norm(norm_r))
        return float(np.sqrt(dbl_area) * np.exp(5.0 * abs(_dihedral(norm_l, norm_r, ab))))
    if metric == "max_dihedral":
        return abs(_dihedral(np.cross(lft - a, ab), np.cross(ab, rgt - a), ab))
    if metric == "complex_fill":
        norm_a = np.cross(rgt - b, -ab)
        norm_c = np.cross(lft - a, ab)
        denom = float(np.linalg.norm(norm_a) * np.linalg.norm(norm_c))
        if denom == 0.0:
            return _BAD
        cos_ac = float(np.dot(norm_a, norm_c) / denom)
        if cos_ac <= -1.0:
            return _BAD
        return ((1.0 - cos_ac) / (1.0 + cos_ac)) ** 4
    return 0.0


def _total_fill_metric(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], fill_flat: np.ndarray, metric: str
) -> float:
    """
    Score a fill triangulation exactly as ``fill_dp_span`` accumulates it.

    Validated against MeshLib's own ``calcCombinedFillMetric``: per-triangle terms plus per-edge
    (dihedral) terms — interior edges use both adjacent fill apexes, rim edges the existing face's
    opposite vertex (``smoothBd``).
    """
    from collections import defaultdict

    vertices_np = vertices.numpy()
    fill = fill_flat.reshape(-1, 3)
    is_max = metric in _COMBINE_MAX

    loops = _fillable_loops(vertices, faces)
    loop_of: dict[int, int] = {}
    normals: list[np.ndarray] = []
    char_areas: list[float] = []
    for li, loop_wp in enumerate(loops):
        loop = loop_wp.numpy()
        for vtx in loop:
            loop_of[int(vtx)] = li
        loop_pos = od.array.gather(vertices, loop_wp)
        normals.append(np.asarray(od.polyline.polyline_normal(loop_pos)))
        pos = vertices_np[loop]
        rim = np.roll(pos, -1, axis=0) - pos
        max_sq = float(np.max(np.einsum("ij,ij->i", rim, rim)))
        char_areas.append(1.0 / max_sq if max_sq > 0.0 else 1.0)

    def edge_map(triangles: np.ndarray) -> dict[tuple[int, int], list[int]]:
        out: dict[tuple[int, int], list[int]] = defaultdict(list)
        for tri in triangles:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for u, v, w in ((a, b, c), (b, c, a), (a, c, b)):
                out[(u, v) if u < v else (v, u)].append(w)
        return out

    fill_apex = edge_map(fill)
    orig_third = edge_map(faces.numpy().reshape(-1, 3))

    total = 0.0
    for tri in fill:
        li = loop_of[int(tri[0])]
        a, b, c = (vertices_np[int(x)] for x in tri)
        # Emitted triangles are reverse-wound vs. the DP scoring; take the accepted orientation.
        term = min(
            _tri_term(a, b, c, normals[li], char_areas[li], metric),
            _tri_term(a, c, b, normals[li], char_areas[li], metric),
        )
        total = max(total, term) if is_max else total + term
    for edge, apexes in fill_apex.items():
        pa, pb = vertices_np[edge[0]], vertices_np[edge[1]]
        if len(apexes) == 2:
            left, right = vertices_np[apexes[0]], vertices_np[apexes[1]]
        elif len(apexes) == 1:
            others = orig_third.get(edge, [])
            if len(others) != 1:
                continue
            left, right = vertices_np[others[0]], vertices_np[apexes[0]]
        else:
            continue
        term = _edge_term(pa, pb, left, right, metric)
        total = max(total, term) if is_max else total + term
    return total


# MeshLib's factory for each ``FILL_METRICS`` entry, called with the mesh and a hole edge.
_MESHLIB_FILL_METRICS: dict[str, Callable[[mm.Mesh, mm.EdgeId], mm.FillHoleMetric]] = {
    "plane_normalized": lambda m, e: mm.getPlaneNormalizedFillMetric(m, e),
    "min_area": lambda m, e: mm.getMinAreaMetric(m),
    "circumscribed": lambda m, e: mm.getCircumscribedMetric(m),
    "plane": lambda m, e: mm.getPlaneFillMetric(m, e),
    "min_tri_angle": lambda m, e: mm.getMinTriAngleMetric(m),
    "edge_length": lambda m, e: mm.getEdgeLengthFillMetric(m),
    "universal": lambda m, e: mm.getUniversalMetric(m),
    "max_dihedral": lambda m, e: mm.getMaxDihedralAngleMetric(m),
    "complex_fill": lambda m, e: mm.getComplexFillMetric(m, e),
}


def _meshlib_fill_triangles(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], metric: str
) -> np.ndarray:
    """
    Fill triangles produced by MeshLib's ``fillHole`` for the same metric, as a flat vertex buffer.

    ``maxPolygonSubdivisions`` is raised so MeshLib runs the exhaustive DP (no large-hole
    sub-sampling), matching [`fill_min_weight`]'s full search. fillHole reuses existing
    vertices, so the new faces are those not present in the original triangle set.
    """
    make_metric = _MESHLIB_FILL_METRICS[metric]
    vertices_np = np.ascontiguousarray(vertices.numpy(), dtype=np.float32)
    faces_np = np.ascontiguousarray(faces.numpy().reshape(-1, 3), dtype=np.int32)
    mesh = numpy_to_meshlib(vertices_np, faces_np)
    params = mm.FillHoleParams()
    params.maxPolygonSubdivisions = 1000
    for edge in mesh.topology.findHoleRepresentiveEdges():
        params.metric = make_metric(mesh, edge)
        mm.fillHole(mesh, edge, params)
    faces_out = mn.getNumpyFaces(mesh.topology)
    original = {tuple(sorted(int(x) for x in tri)) for tri in faces_np}
    fill = [tri for tri in faces_out if tuple(sorted(int(x) for x in tri)) not in original]
    return np.asarray(fill, dtype=np.int32).reshape(-1)


@pytest.mark.parity("fillable_loop_mask", "meshlib")
def test_fillable_loop_mask_pinch_matches_meshlib(device: str) -> None:
    """
    Class B: equal after reducing MeshLib's per-*vertex* answer to a per-*loop* one.

    ``findRepeatedVertsOnHoleBd`` marks the vertices a boundary walk visits twice, which is the
    pinch condition seen from the other end: a loop is pinched exactly when it contains one of
    them. So the transform is "unfillable if the loop meets the marked set", and the two agree.

    The fixture is the smallest mesh with a pinched rim -- two triangles meeting at one vertex and
    nowhere else -- because on a clean mesh both answers are empty and the comparison would be
    vacuous. Measured: MeshLib marks vertex 0 alone, and ordito's two loops ``[0, 1, 2]`` and
    ``[0, 3, 4]`` both come back ``False``, since each passes through the marked vertex; the clean
    control comes back ``True`` with an empty marked set, which is what rules out a mask that is
    simply always ``False``. (The vertex walk used to return one garbage loop ``[0, 0, 0, 1, 2]``
    here; the halfedge walk returns the two rims the triangles actually have.)
    """
    pinched_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    pinched_faces_np = np.array([0, 1, 2, 0, 3, 4], dtype=np.int32)
    clean_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    clean_faces_np = np.array([0, 1, 2], dtype=np.int32)

    for vertices_np, faces_np, expected, n_loops in (
        (pinched_np, pinched_faces_np, False, 2),
        (clean_np, clean_faces_np, True, 1),
    ):
        vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
        loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)
        assert len(loops_wp) == n_loops  # non-vacuity: there is a rim to judge
        fillable_np = od.holes.fillable_loop_mask(vertices_wp, faces_wp, loops_wp).numpy()

        mesh_ml = numpy_to_meshlib(vertices_np, faces_np.reshape(-1, 3))
        repeated_np = meshlib_bitset_to_numpy(
            mm.findRepeatedVertsOnHoleBd(mesh_ml.topology), len(vertices_np)
        )
        assert repeated_np.any() == (not expected)
        for fillable, loop_wp in zip(fillable_np, loops_wp, strict=True):
            assert bool(fillable) is expected
            assert (not repeated_np[loop_wp.numpy()].any()) is expected


def test_fillable_loop_mask_chord_is_conservative(device: str) -> None:
    """
    Not a library comparison: MeshLib's ``findHoleComplicatingFaces`` measures something else.

    The chord condition is ordito's own, and the reference is recorded as *not* matching it: on a
    square split by one diagonal, ``findHoleComplicatingFaces`` flags **no** face while this mask
    returns ``False``, because the rim's two opposite corners are already joined by that diagonal.
    So there is nothing to compare and the test states the behaviour instead.

    Both directions matter. A hexagonal fan has no chord and comes back ``True``, which is what
    stops the mask being trivially ``False``; the square comes back ``False``. And the docstring's
    conservatism claim is checked rather than asserted away: the square *does* fill cleanly, because
    the dynamic program picks the other diagonal -- so ``False`` means "check the result", and a
    test that treated it as "cannot be filled" would be wrong.
    """
    square_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    square_faces_np = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    vertices_wp, faces_wp = numpy_to_warp(square_np, square_faces_np, device)
    assert not od.holes.fillable_loop_mask(vertices_wp, faces_wp).numpy()[0]
    mesh_ml = numpy_to_meshlib(square_np, square_faces_np.reshape(-1, 3))
    assert mm.findHoleComplicatingFaces(mesh_ml).count() == 0  # the recorded divergence

    sides = 6
    fan_np = np.vstack(
        [[0.0, 0.0, 0.0]]
        + [
            [float(np.cos(2 * np.pi * k / sides)), float(np.sin(2 * np.pi * k / sides)), 0.0]
            for k in range(sides)
        ]
    )
    fan_faces_np = np.array(
        [index for k in range(sides) for index in (0, 1 + k, 1 + (k + 1) % sides)], dtype=np.int32
    )
    fan_vertices_wp, fan_faces_wp = numpy_to_warp(fan_np, fan_faces_np, device)
    assert od.holes.fillable_loop_mask(fan_vertices_wp, fan_faces_wp).numpy()[0]


def test_fillable_loop_mask_shared_vertex_disqualifies_both_loops(device: str) -> None:
    """
    Not a library comparison: the guarantee is ordito's own wording, not a reference's.

    ``fillable_loop_mask`` documents ``True`` as a *guarantee* -- "a vertex shared by two different
    rims is the same pinch spread across two loops, and disqualifies both". The pair below is that
    configuration with one twist: the loop that shares the vertex is itself clean, and the loop it
    shares with is self-pinched somewhere else. Both are unfillable, because the shared vertex is
    non-manifold whoever else touches it.

    The loops are supplied rather than derived, and deliberately: ``boundary_loops``' walk merges a
    shared boundary vertex into a single loop, so the two-loop form is reachable exactly through
    the ``loops=`` keyword -- which every caller that filters or derives its own rims uses. The
    control pair, identical except that the second loop is a disjoint triangle's rim, keeps the
    test from passing on a mask that is simply always ``False``.

    **Bug class excluded:** counting rim ownership over only the loops that survived the self-pinch
    test, which makes a clean loop sharing a vertex with a pinched one answer ``True``. **Mutation
    probe, measured against a detached pre-fix worktree:** the shared pair returns
    ``[False, True]`` there and ``[False, False]`` here, with the control ``[False, True]`` on
    both.
    """
    vertices_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 2.0, 0.0],
            [-1.0, 2.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
        ]
    )
    faces_np = np.array([[0, 1, 2], [0, 3, 4], [2, 5, 6], [7, 8, 9]], dtype=np.int64)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    def loop(indices: list[int]) -> wp.array[wp.int32]:
        return wp.array(np.array(indices, dtype=np.int32), dtype=wp.int32, device=device)

    pinched = loop([0, 1, 2, 0, 3, 4])  # visits vertex 0 twice: a pinch within itself
    sharing = loop([2, 5, 6])  # clean, but meets ``pinched`` at vertex 2
    disjoint = loop([7, 8, 9])  # clean and meets nothing

    shared_np = od.holes.fillable_loop_mask(vertices_wp, faces_wp, [pinched, sharing]).numpy()
    control_np = od.holes.fillable_loop_mask(vertices_wp, faces_wp, [pinched, disjoint]).numpy()

    assert np.array_equal(shared_np, np.array([False, False]))
    assert np.array_equal(control_np, np.array([False, True]))


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus", "icosphere_coarse"])
def test_fillable_loop_mask_on_the_fixtures(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Not a library comparison: the mask must agree with what the fill actually does.

    Every rim of these fixtures is simple and chord-free, so the mask is all-``True`` -- and the
    claim that ``True`` carries is checkable: ``fill_min_weight(resolve_multiple_edges=False)``,
    which is the unprotected path, must leave an edge-manifold mesh. That is the guarantee, and
    running the fill with the protection *off* is the only way to test it.

    ``icosphere_coarse`` is the closed case, where there are no loops and the mask is empty.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)
    fillable_np = od.holes.fillable_loop_mask(vertices_wp, faces_wp, loops_wp).numpy()
    assert fillable_np.shape == (len(loops_wp),)
    if not len(loops_wp):
        return  # the closed fixture: nothing to fill, and nothing to claim
    assert fillable_np.all()
    filled_wp = od.holes.fill_min_weight(vertices_wp, faces_wp, resolve_multiple_edges=False)
    assert filled_wp.size > faces_wp.size
    assert od.validation.is_edge_manifold(filled_wp)


@pytest.mark.parity(
    "fill_min_weight_chords",
    "meshlib",
    benchmarked=False,
    reason="resolve_multiple_edges defaults to True, so the marked comparison below runs "
    "this group's chords configuration without naming it. Its plain row is the same DP "
    "with the forbidden-chord pass off, which no reference has a counterpart for -- "
    "banning chords that already exist as mesh edges is ordito's own guarantee.",
)
@pytest.mark.parity("fill_min_weight", "meshlib")
@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parametrize("metric", FILL_METRICS)
def test_fill_min_weight_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str, metric: str
) -> None:
    """
    Class C: the triangle count and the *achieved optimum*, there being no canonical triangulation.

    The interval DP has one minimum-weight cost and, in general, several triangulations that reach
    it -- a rim with cocircular vertices ties, and the two libraries break ties differently -- so
    the face buffers need not agree even when both solutions are optimal. What must agree is the
    triangle count and the total metric each answer achieves, scored here by the *same* function
    for both.

    That scorer is the part a class-C claim has to justify, and it is anchored separately:
    [`test_fill_metric_scorer_matches_meshlib`] pins ``_total_fill_metric`` against MeshLib's own
    ``calcCombinedFillMetric``, so a scorer that agreed with neither library's definition could not
    make this test pass. Excludes the bug class "the DP reaches a worse optimum" -- a greedy fan, a
    fill that misses the true minimum, an off-by-one in the interval recursion -- for each of the
    metrics in ``FILL_METRICS``, which is why the metric is an axis rather than a single choice.

    The tolerance is ``rtol=2e-3`` rather than the usual ``1e-5`` because both sides accumulate
    their metric in ``float32`` across the whole patch.

    Not a library comparison, the count identity: whichever triangulation a metric picks, it must
    be ``B - 2`` triangles over the existing vertices, and the result is closed and consistently
    wound. ordito's own ``is_watertight`` false-positives on coplanar or curved caps through its
    self-intersection test, so trimesh is the watertight oracle beside the edge-manifold check.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    n_orig = mesh_wp.indices.size
    loop_sizes = _loop_sizes(mesh_wp)

    filled_faces = od.holes.fill_min_weight(mesh_wp.points, mesh_wp.indices, metric=metric)
    fill_wp = filled_faces.numpy()[n_orig:]
    fill_ml = _meshlib_fill_triangles(mesh_wp.points, mesh_wp.indices, metric)

    assert len(fill_wp) // 3 == sum(size - 2 for size in loop_sizes)
    assert od.validation.is_edge_manifold(filled_faces, allow_boundary_edges=False)
    assert od.validation.is_winding_consistent(filled_faces)
    assert len(_loop_sizes_of(mesh_wp.points, filled_faces)) == 0
    filled_tm = warp_to_trimesh(mesh_wp.points, filled_faces)
    assert filled_tm.is_watertight
    assert filled_tm.is_winding_consistent

    # Same triangle count and same achieved optimum as MeshLib's exhaustive fillHole (the exact
    # triangulation can differ under ties / MeshLib's tie-breaking, so compare the cost).
    assert len(fill_wp) // 3 == len(fill_ml) // 3
    total_wp = _total_fill_metric(mesh_wp.points, mesh_wp.indices, fill_wp, metric)
    total_ml = _total_fill_metric(mesh_wp.points, mesh_wp.indices, fill_ml, metric)
    assert np.isclose(total_wp, total_ml, rtol=2e-3, atol=1e-3)


def test_fill_metric_scorer_matches_meshlib(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Anchor ``_total_fill_metric`` to MeshLib's own ``calcCombinedFillMetric`` (single-hole mesh).

    Validates that the test-side scorer used by ``test_fill_min_weight_matches_meshlib`` truly
    computes each MeshLib metric, for the metrics that expose a triangle term
    (``calcCombinedFillMetric`` cannot score the edge-only metrics — it always calls
    ``triangleMetric``, which is empty for them).
    """
    _, mesh_wp = hemisphere
    faces_np = np.ascontiguousarray(mesh_wp.indices.numpy().reshape(-1, 3), dtype=np.int32)
    verts_np = np.ascontiguousarray(mesh_wp.points.numpy(), dtype=np.float32)
    # The edge-only metrics have no triangle term for ``calcCombinedFillMetric`` to score.
    scored = {
        metric: factory
        for metric, factory in _MESHLIB_FILL_METRICS.items()
        if metric not in {"edge_length", "max_dihedral"}
    }
    for metric, factory in scored.items():
        fill = (
            od.holes.fill_min_weight(mesh_wp.points, mesh_wp.indices, metric=metric)
            .numpy()[faces_np.size :]
            .reshape(-1, 3)
        )
        mesh_orig = numpy_to_meshlib(verts_np, faces_np)
        metric_obj = factory(mesh_orig, mesh_orig.topology.findHoleRepresentiveEdges()[0])
        full = np.vstack([faces_np, fill]).astype(np.int32)
        mesh_full = numpy_to_meshlib(verts_np, full)
        region_bools = np.zeros(len(full), dtype=bool)
        region_bools[len(faces_np) :] = True
        region = mn.faceBitSetFromBools(cast("Buffer", region_bools))
        mr_cost = mm.calcCombinedFillMetric(mesh_full, region, metric_obj)
        my_cost = _total_fill_metric(mesh_wp.points, mesh_wp.indices, fill.reshape(-1), metric)
        assert np.isclose(my_cost, mr_cost, rtol=2e-3, atol=1e-3), metric


def _sealed_volume(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    """
    Enclosed volume of a sealed mesh, after canonicalising the winding.

    The ``fix_winding`` pass is load-bearing for Open3D: ``fill_holes`` emits its cap triangles
    wound
    *against* the rest of the mesh, so a raw signed-volume read of its output is meaningless (-1.06
    on ``hemisphere``, where the true volume is 2.02). Repairing the winding first is the named
    transform that makes the three libraries' volumes comparable.
    """
    mesh_tm = tm.Trimesh(vertices_np, np.asarray(faces_np).reshape(-1, 3), process=False)
    tm_repair.fix_winding(mesh_tm)
    return float(mesh_tm.volume)


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity(
    "fill_min_weight_chords",
    "open3d",
    "pymeshlab",
    benchmarked=False,
    reason="resolve_multiple_edges defaults to True, so the marked comparison below runs "
    "this group's chords configuration without naming it. Its plain row is the same DP "
    "with the forbidden-chord pass off, which no reference has a counterpart for -- "
    "banning chords that already exist as mesh edges is ordito's own guarantee.",
)
@pytest.mark.parity("fill_min_weight", "open3d", "pymeshlab")
def test_fill_min_weight_matches_open3d_and_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class C: three hole fillers pick three different triangulations of the same loop.

    All three seal the identical boundary with the identical budget -- ``B - 2`` triangles over the
    existing vertices only -- but no two agree triangle for triangle, so there is no correspondence
    to recover and the comparison has to be on the sealed *surface* plus the achieved optimum.
    Open3D's is read through the tensor API (``o3d.t`` ``fill_holes``) and MeshLab's through
    ``meshing_close_holes``, whose default ``maxholesize=30`` would close nothing on these rims, so
    the cap is lifted exactly as the benchmark lifts it.

    Three asserts. The counts are Class A: same vertex count, same face count, and every added
    triangle drawn only from the boundary loops. The **volume** is Class B under the named
    [`_sealed_volume`][tests.test_holes._sealed_volume] transform -- these rims are planar,
    so the enclosed volume is triangulation-invariant and all three must agree exactly. And the
    **cost** is the optimality claim: ordito runs the exact minimum-weight DP, so its
    ``plane_normalized`` cost must be no worse than either reference's triangulation of the same
    loop.

    **Bug class excluded:** a DP that finds a valid but *suboptimal* triangulation -- the failure
    the volume and count asserts are both blind to, since every triangulation of a planar rim has
    the same volume and the same triangle count.

    **Mutation probe, measured:** on ``hemisphere`` the costs are ordito **290.7**, MeshLab 1 821.1
    (6.3x) and Open3D 827.3 (2.8x); on ``half_torus`` **686.0** against 2 740.6 (4.0x) and 1 127.9
    (1.6x). The bound is not one any answer passes: this module's own
    [`fill_fan`][ordito.holes.fill_fan] emits the same ``B - 2`` triangles over
    the same vertices and scores 860.1 and 3 509.5 -- it *fails* against Open3D on both fixtures.
    ``zeros_like`` and "return the input faces" fail at the face-count assert.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh_tm = warp_to_trimesh(mesh_wp.points, mesh_wp.indices)
    vertices_np = mesh_tm.vertices
    n_original = mesh_wp.indices.size

    filled_wp = od.holes.fill_min_weight(mesh_wp.points, mesh_wp.indices).numpy()

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    statistics_pml = meshset_pml.meshing_close_holes(maxholesize=1_000_000)
    assert statistics_pml["closed_holes"] > 0
    faces_pml = np.asarray(meshset_pml.current_mesh().face_matrix())
    assert np.allclose(meshset_pml.current_mesh().vertex_matrix(), vertices_np, atol=1e-6)

    # The tensor mesh must be held in a name: chaining ``from_legacy(...).fill_holes()`` lets the
    # temporary be collected and the result's ``positions`` then read freed memory (observed as
    # 2052.1 and 4.4e-41 in the first rows) rather than raising.
    tensor_o3d = o3d.t.geometry.TriangleMesh.from_legacy(trimesh_to_open3d(mesh_tm))
    mesh_o3d = tensor_o3d.fill_holes()
    faces_o3d = mesh_o3d.triangle["indices"].numpy()
    assert np.allclose(mesh_o3d.vertex["positions"].numpy(), vertices_np, atol=1e-6)

    loop_vertices = {
        int(vertex)
        for loop_wp in _fillable_loops(mesh_wp.points, mesh_wp.indices)
        for vertex in loop_wp.numpy()
    }
    cost_wp = _total_fill_metric(
        mesh_wp.points, mesh_wp.indices, filled_wp[n_original:], "plane_normalized"
    )
    for faces_ref in (faces_pml, faces_o3d):
        assert faces_ref.shape[0] == filled_wp.size // 3
        added_ref = np.asarray(faces_ref).reshape(-1)[n_original:].astype(np.int32)
        assert set(np.unique(added_ref).tolist()) <= loop_vertices
        assert np.isclose(
            _sealed_volume(vertices_np, filled_wp),
            _sealed_volume(vertices_np, faces_ref),
            rtol=1e-5,
        )
        cost_ref = _total_fill_metric(
            mesh_wp.points, mesh_wp.indices, added_ref, "plane_normalized"
        )
        assert cost_wp <= cost_ref, f"the minimum-weight DP scored worse: {cost_wp} > {cost_ref}"


def _punched_sphere(n_holes: int, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Icosphere with ``n_holes`` scattered single-triangle holes: many independent small loops."""
    mesh_tm = tm.creation.icosphere(subdivisions=3)
    rng = np.random.default_rng(seed)
    # Only drop faces whose one-rings are disjoint, so each hole stays a separate 3-vertex loop.
    blocked: set[int] = set()
    dropped: list[int] = []
    adjacency = mesh_tm.face_adjacency
    neighbors: dict[int, set[int]] = {face: set() for face in range(len(mesh_tm.faces))}
    for left, right in adjacency:
        neighbors[int(left)].add(int(right))
        neighbors[int(right)].add(int(left))
    for face in rng.permutation(len(mesh_tm.faces)):
        face = int(face)
        if len(dropped) == n_holes or face in blocked:
            continue
        dropped.append(face)
        blocked.add(face)
        blocked.update(neighbors[face])
        for near in neighbors[face]:
            blocked.update(neighbors[near])
    keep = np.setdiff1d(np.arange(len(mesh_tm.faces)), np.asarray(dropped))
    return mesh_tm.vertices, mesh_tm.faces[keep]


@pytest.mark.parametrize("metric", ["plane_normalized", "min_area", "universal"])
def test_fill_min_weight_batched_equals_per_loop(device: str, metric: str) -> None:
    """
    Filling many holes in one call must match filling them one at a time.

    The interval DP is solved for every loop in the same launches over one ragged table, so this is
    the regression that would catch a loop bleeding into its neighbour's block -- an off-by-one in
    ``dp_offsets``, a chord marked against the wrong loop, or a min-area fallback firing for the
    whole batch instead of the loops that needed it.
    """
    vertices_np, faces_np = _punched_sphere(24)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    loops = _fillable_loops(vertices_wp, faces_wp)
    assert len(loops) >= 8

    batched_np = od.holes.fill_loops_min_weight(vertices_wp, faces_wp, loops, metric, True).numpy()
    per_loop = [
        od.holes.fill_loops_min_weight(vertices_wp, faces_wp, [loop], metric, True).numpy()[
            faces_wp.size :
        ]
        for loop in loops
    ]
    expected_np = np.concatenate([faces_wp.numpy(), *per_loop])
    assert np.array_equal(batched_np, expected_np)


def _star_tube(n: int = 64, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Open tube whose free rim is a non-convex, non-planar star of ``n`` vertices.

    A rim this long (spans past ``HOLE_DP_BLOCK``, so a lane covers several apexes) and this
    irregular is what gives the interval DP genuine ties to break; a convex planar rim does not.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = np.where(np.arange(n) % 2 == 0, 1.0, 0.45)
    rng = np.random.default_rng(seed)
    rim = np.stack(
        [radius * np.cos(angle), radius * np.sin(angle), rng.normal(scale=0.08, size=n)], axis=1
    )
    skirt = np.stack([1.6 * np.cos(angle), 1.6 * np.sin(angle), np.full(n, -0.5)], axis=1)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, n + i, n + j])
        faces.append([i, n + j, j])
    return np.vstack([rim, skirt]), np.asarray(faces, dtype=np.int64)


@pytest.mark.parametrize("metric", FILL_METRICS)
@pytest.mark.parametrize("smooth_boundary", [True, False])
def test_fill_dp_span_tiled_matches_serial(
    monkeypatch: pytest.MonkeyPatch, device: str, metric: str, smooth_boundary: bool
) -> None:
    """
    The tiled per-span engine must emit the **identical face set** as the serial one.

    This is the gate the rest of the file cannot be: the interval DP's ``update_argmin`` takes the
    *smallest* apex ``k`` at equal cost, and that choice picks the triangles, so a reduction that
    resolves a tie to a different ``k`` produces an equal-count, equal-cost, **different**
    triangulation. Measured on this very fixture set: inverting the tie-break to the largest ``k``
    changes 76 of 144 (fixture, metric, flag) cases and every other test in this file still passes.
    So byte-equality against the serial reference is the only thing that pins the tiled reduction,
    and ``fill_dp_span`` stays in the module as that reference (it is also the default CPU engine).

    Runs on **both** devices. It used to skip on CPU, because the tiled kernel strided its apex loop
    by the ``HOLE_DP_BLOCK`` constant while ``wp.launch_tiled`` gives the CPU one lane per block --
    so lane 0 stepped by 32 and the DP minimized over every 32nd apex. Striding by
    ``wp.block_dim()`` instead makes the single CPU lane cover every apex, and this comparison is
    what verifies that: on CPU it is now a genuine tiled-vs-serial check rather than a skip.
    """
    vertices_np, faces_np = _star_tube()
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    n_orig = faces_wp.size

    original = od.holes._run_hole_dp  # pyright: ignore[reportPrivateUsage]
    fills = {}
    for tiled in (True, False):
        monkeypatch.setattr(
            od.holes,
            "_run_hole_dp",
            lambda *args, _tiled=tiled, **kwargs: original(*args, **kwargs, tiled=_tiled),
        )
        fills[tiled] = od.holes.fill_min_weight(
            vertices_wp, faces_wp, metric=metric, smooth_boundary=smooth_boundary
        ).numpy()[n_orig:]

    assert fills[True].size > 3 * 3, "fixture produced no fill to compare"
    assert np.array_equal(fills[True], fills[False])


@pytest.mark.parametrize("metric", FILL_METRICS)
@pytest.mark.parametrize("n_rim", [64, 66])
def test_fill_dp_captured_matches_plain(
    monkeypatch: pytest.MonkeyPatch, device: str, metric: str, n_rim: int
) -> None:
    """
    Ordito against ordito: the recorded span sweep must emit the **identical face set**.

    The plain per-span loop carries the oracle -- it is what the CPU device runs and what every
    reference comparison in this file is written against. The recorded form differs only in where
    the span comes from (``HoleFillTables.span_base`` plus a per-launch offset, stepped on the
    device between groups) and in launching each group at the *first* group's grid width, so any
    drift in either shows up here as a different triangulation rather than a different cost. As in
    ``test_fill_dp_span_tiled_matches_serial``, equal totals would not be enough: the DP's
    tie-break picks the triangles.

    ``n_rim`` is the mechanism parameter. 64 leaves ``62 % HOLE_DP_GRAPH_SPANS`` spans over, so the
    last replay runs past the final span and is absorbed by the kernel's own ``span >= b`` guard;
    66 divides exactly and never reaches that branch. Both must give the same answer as the loop.

    Skipped on the CPU device, which has no stream to record and always takes the plain loop.
    """
    if not wp.get_device(device).is_cuda:
        pytest.skip("graph capture needs a CUDA stream; the CPU device runs the plain span loop")
    vertices_np, faces_np = _star_tube(n_rim)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    n_orig = faces_wp.size
    # The shipped gate must actually choose the recorded path here, or the arms below agree
    # because they are the same path twice.
    assert kernel_holes.hole_dp_captures(n_rim, 2)

    original = od.holes._run_hole_dp  # pyright: ignore[reportPrivateUsage]
    fills = {}
    for captured in (True, False):
        monkeypatch.setattr(
            od.holes,
            "_run_hole_dp",
            lambda *args, _c=captured, **kwargs: original(*args, **kwargs, captured=_c),
        )
        fills[captured] = od.holes.fill_min_weight(vertices_wp, faces_wp, metric=metric).numpy()[
            n_orig:
        ]

    assert fills[False].size > 3 * 3, "fixture produced no fill to compare"
    assert np.array_equal(fills[True], fills[False])


def test_hole_dp_captures_is_bounded_at_both_ends(device: str) -> None:
    """
    Not a library comparison: the sweep's launch schedule has no counterpart to compare against.

    Pins the *shape* of the gate rather than its constants. A short sweep must decline, because
    recording costs one ordinary launch per span in the group and is wasted beside a sequence of
    the same order. A sweep whose spans each carry heavy device work must decline too, and that is
    the half a reader is likely to drop: it is not a bound on rim length but on
    ``n_loops * max_size ** 2``, so growing the rim *count* at a fixed length has to turn the gate
    off exactly as growing the length does.
    """
    del device  # the gate is host-side arithmetic and reads no device
    short, light = 8, 64
    assert not kernel_holes.hole_dp_captures(short, 1), "a sweep of a few spans must decline"
    assert kernel_holes.hole_dp_captures(light, 1), "the fixture the capture test uses must pass"
    heavy_length = kernel_holes.HOLE_DP_CAPTURE_MAX_WORK
    assert not kernel_holes.hole_dp_captures(heavy_length, 1), "one very long rim must decline"
    # Same rim length, more rims: the work bound has to see this and a length bound cannot.
    assert not kernel_holes.hole_dp_captures(light, kernel_holes.HOLE_DP_CAPTURE_MAX_WORK)


def test_fill_min_weight_optimal_vs_fan(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = hemisphere
    # Single-loop fixture: the DP minimizes the plane-normalized objective over all triangulations,
    # and the fan is one such triangulation, so the min-weight total must not exceed the fan's.
    n_orig = mesh_wp.indices.size
    fan_fill = od.holes.fill_fan(mesh_wp.points, mesh_wp.indices).numpy()[n_orig:]
    mw_fill = od.holes.fill_min_weight(mesh_wp.points, mesh_wp.indices).numpy()[n_orig:]

    fan_total = _total_fill_metric(mesh_wp.points, mesh_wp.indices, fan_fill, "plane_normalized")
    mw_total = _total_fill_metric(mesh_wp.points, mesh_wp.indices, mw_fill, "plane_normalized")
    assert mw_total <= fan_total + 1e-4


def test_fill_min_weight_avoids_multiple_edges(device: str) -> None:
    # Two triangles sharing edge (0, 2); the boundary loop 0-1-2-3 has 0-2 as a pre-existing chord.
    vertices = wp.array(
        [[0.0, 0.0, 0.0], [0.5, 2.0, 0.0], [1.0, 0.0, 0.0], [0.5, -2.0, 0.0]],
        dtype=wp.vec3,
        device=device,
    )
    faces = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device=device)

    def edge_face_count(faces_flat: np.ndarray, u: int, v: int) -> int:
        return int(sum({u, v} <= set(tri) for tri in faces_flat.reshape(-1, 3).tolist()))

    resolved = od.holes.fill_min_weight(vertices, faces, resolve_multiple_edges=True)
    # The forbidden diagonal (0, 2) keeps its two original faces; the other diagonal is used.
    assert edge_face_count(resolved.numpy(), 0, 2) == 2
    assert od.validation.is_edge_manifold(resolved, allow_boundary_edges=True)

    unresolved = od.holes.fill_min_weight(vertices, faces, resolve_multiple_edges=False)
    # Free to reuse the geometrically preferred diagonal (0, 2), creating a non-manifold edge.
    assert edge_face_count(unresolved.numpy(), 0, 2) == 4
    assert not od.validation.is_edge_manifold(unresolved, allow_boundary_edges=True)


# A quad rim (0, 1, 2, 3) whose two diagonals are both mesh edges: vertices 4 and 5 each join all
# four rim vertices, giving the patch one cross-cap (see the test below for why it must be one).
_CROSS_CAP_VERTICES = np.array(
    [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0.3, 0.1, 0.7], [-0.2, 0.2, -0.7]],
    dtype=np.float64,
)
_CROSS_CAP_FACES = np.array(
    [[0, 1, 5], [1, 2, 4], [2, 3, 5], [3, 0, 4], [0, 2, 4], [0, 2, 5], [1, 3, 4], [1, 3, 5]],
    dtype=np.int32,
)


def test_fill_min_weight_leaves_a_hole_open_when_both_diagonals_are_forbidden(device: str) -> None:
    """
    A quad hole whose both diagonals are already forbidden must come back unfilled.

    A quad hole (0, 1, 2, 3) whose *both* diagonals -- 0-2 and 1-3 -- already exist as mesh edges
    elsewhere. A single forbidden diagonal (the sibling test above) still leaves one legal
    triangulation; here neither of the two possible triangulations of a 4-gon has an admissible
    apex, so the hole must come back untouched rather than filled with a triangle that reuses one
    of the forbidden chords.

    This configuration needs a non-orientable surrounding to exist at all (a planar/orientable
    quad's two diagonals always cross, so at most one can be a pre-existing edge without already
    making some rim vertex non-manifold) -- vertices 4 and 5 below are each connected to all four
    rim vertices, giving the patch a single cross-cap. Verified independently: every incident edge
    has the expected face count (rim edges 1, everything else 2) and every vertex's link is one
    connected fan, i.e. this is a genuine 2-manifold with one boundary loop -- (0, 1, 2, 3) -- and
    diagonals (0, 2) and (1, 3) both present with two faces each, not touching the rim.
    """
    vertices, faces = numpy_to_warp(_CROSS_CAP_VERTICES, _CROSS_CAP_FACES, device)

    loops = od.boundary.boundary_loops(vertices, faces)
    assert len(loops) == 1
    assert loops[0].numpy().tolist() == [0, 1, 2, 3]

    for metric in ["plane_normalized", "min_area", "universal"]:
        filled = od.holes.fill_min_weight(
            vertices, faces, metric=metric, resolve_multiple_edges=True
        )
        assert np.array_equal(filled.numpy(), faces.numpy())

    result_loops = od.boundary.boundary_loops(vertices, faces)
    assert len(result_loops) == 1
    assert result_loops[0].numpy().tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("metric", ["plane_normalized", "min_area"])
def test_fill_min_weight_compacts_around_a_blocked_hole(device: str, metric: str) -> None:
    """
    Not a library comparison: no reference forbids a chord and reports what it left open.

    One blocked quad (the cross-cap patch of the test above) beside an icosahedron missing one
    face. The traceback writes each rim's ``B - 2`` triangles into a padded block at the tail of
    the returned buffer, and only a rim that falls short sends the call down the compaction path --
    the only one that re-places the triangles. The fully blocked test above falls short by
    *everything* and never reaches it, so this fixture is built to fall short by part: the quad's
    two slots are empty and the triangle hole's one slot is not. The claim is that the output is
    exactly the input faces followed by the one closing triangle, with the quad's rim still open.
    """
    quad_vertices, quad_faces = _CROSS_CAP_VERTICES, _CROSS_CAP_FACES
    icosahedron = tm.creation.icosahedron()
    offset = quad_vertices.shape[0]
    vertices_np = np.vstack([quad_vertices, icosahedron.vertices + 5.0])
    faces_np = np.vstack([quad_faces, icosahedron.faces[1:] + offset])
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    loops = od.boundary.boundary_loops(vertices_wp, faces_wp)
    assert sorted(len(loop) for loop in loops) == [3, 4], "fixture must hold a quad and a triangle"

    filled_np = od.holes.fill_min_weight(vertices_wp, faces_wp, metric=metric).numpy()

    n_orig = faces_np.size
    assert np.array_equal(filled_np[:n_orig], faces_np.reshape(-1))
    assert filled_np.size == n_orig + 3, "exactly the triangle hole's one face is appended"
    assert set(filled_np[n_orig:].tolist()) == set((icosahedron.faces[0] + offset).tolist())
    filled_wp = wp.array(filled_np, dtype=wp.int32, device=device)
    remaining = od.boundary.boundary_loops(vertices_wp, filled_wp)
    assert [sorted(loop.numpy().tolist()) for loop in remaining] == [[0, 1, 2, 3]]


def test_fill_min_weight_rejects_unknown_metric(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: the documented ``ValueError`` at all four metric-taking entry points.

    ``fill_loops_min_weight`` is the one that had no guard and surfaced a bare
    ``KeyError('bogus')`` -- naming neither the argument nor the alternatives -- so it is checked
    here beside the three that did, through the one shared validator they now all call. The
    exception type is the claim: a ``pytest.raises(ValueError)`` fails on a ``KeyError``.
    """
    _, mesh_wp = hemisphere
    loops_wp = od.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)
    assert loops_wp  # non-vacuity: the loop-taking form gets a real loop, not an empty list
    face_mask_wp = wp.zeros(mesh_wp.indices.size // 3, dtype=wp.bool, device=mesh_wp.device)

    for call in (
        lambda: od.holes.fill_min_weight(mesh_wp.points, mesh_wp.indices, metric="bogus"),
        lambda: od.holes.fill_loops_min_weight(
            mesh_wp.points, mesh_wp.indices, loops_wp, "bogus", False
        ),
        lambda: od.holes.fill_smooth(mesh_wp.points, mesh_wp.indices, metric="bogus"),
        lambda: od.holes.refill_region(
            mesh_wp.points, mesh_wp.indices, face_mask_wp, metric="bogus"
        ),
    ):
        with pytest.raises(ValueError, match="metric must be one of"):
            call()


# ---------------------------------------------------------------------------
# fill_small
# ---------------------------------------------------------------------------


def _two_holes_of_different_size(device: str):
    """
    Cut a wide cap and a two-face pinhole into an icosphere, reporting both perimeters.

    The pinhole is two faces sharing an *edge*, so its rim is a simple 4-gon, and the cap is a
    16-gon. The south polar region is widened sixfold in ``x`` and ``y`` so the 4-gon's perimeter
    (about 3.6) exceeds the 16-gon's (2.41) -- the inversion the threshold tests turn on. An
    earlier version took the two lowest faces, which share only a corner: the rim was pinched, and
    its apparent 5-vertex, long-perimeter loop was the vertex walk's garbage (fake jumps to vertex
    0), not a hole the mesh has.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    centers_np = sphere_tm.triangles_center
    keep_np = np.ones(sphere_tm.faces.shape[0], dtype=bool)
    keep_np[np.argsort(-centers_np[:, 2])[:20]] = False
    bottom = int(np.argmin(centers_np[:, 2]))
    adjacency_np = sphere_tm.face_adjacency
    pair_np = adjacency_np[(adjacency_np == bottom).any(axis=1)][0]
    keep_np[pair_np] = False
    vertices_np = sphere_tm.vertices.copy()
    vertices_np[vertices_np[:, 2] < -0.8, :2] *= 6.0
    holed_tm = tm.Trimesh(vertices_np, sphere_tm.faces[keep_np], process=False)
    holed_tm.remove_unreferenced_vertices()
    vertices_wp, faces_wp = numpy_to_warp(holed_tm.vertices, holed_tm.faces, device)
    loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)
    perimeters = [
        float(od.polyline.polyline_length(od.array.gather(vertices_wp, loop_wp), closed=True))
        for loop_wp in loops_wp
    ]
    return vertices_wp, faces_wp, [loop_wp.size for loop_wp in loops_wp], perimeters


def test_fill_small_fills_exactly_the_loops_under_the_threshold(device: str) -> None:
    """
    Class A: the face count grows by ``n_loop - 2`` for each loop at or under ``max_perimeter``.

    A fan triangulation of an ``n``-gon is ``n - 2`` triangles, so the count is an exact oracle for
    *which* loops were filled rather than just that something was. The fixture carries a 16-vertex
    loop of perimeter 2.41 and a 4-vertex one of perimeter about 3.6 -- the shorter perimeter
    belongs to the loop with *more* vertices, so a threshold between them cannot be satisfied by a
    vertex-count rule by accident.
    """
    vertices_wp, faces_wp, loop_sizes, perimeters = _two_holes_of_different_size(device)
    n_faces = faces_wp.size // 3
    assert len(perimeters) == 2
    smaller, larger = min(perimeters), max(perimeters)
    smaller_size = loop_sizes[perimeters.index(smaller)]

    below_both_wp = od.holes.fill_small(vertices_wp, faces_wp, smaller * 0.5)
    between_wp = od.holes.fill_small(vertices_wp, faces_wp, (smaller + larger) / 2.0)
    above_both_wp = od.holes.fill_small(vertices_wp, faces_wp, larger * 1.1)

    assert below_both_wp.size // 3 == n_faces
    assert between_wp.size // 3 == n_faces + smaller_size - 2
    assert above_both_wp.size // 3 == n_faces + sum(loop_sizes) - 4
    # The prefix is the input face buffer: fill triangles are appended, never interleaved.
    assert np.array_equal(above_both_wp.numpy()[: faces_wp.size], faces_wp.numpy())


def test_fill_small_leaves_a_watertight_mesh_alone(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """With no boundary loop at all the result is a copy of the input, not the input itself."""
    _mesh_tm, mesh_wp = icosahedron

    filled_wp = od.holes.fill_small(mesh_wp.points, mesh_wp.indices, 1e9)

    assert np.array_equal(filled_wp.numpy(), mesh_wp.indices.numpy())
    assert filled_wp.ptr != mesh_wp.indices.ptr


def _two_rims_of_different_edge_count(device: str):
    """
    Cut two well-separated caps of different size out of an icosphere, reporting the rim sizes.

    Distinct from [`_two_holes_of_different_size`] in what a reference comparison needs: both rims
    are ordinary manifold loops over an *undeformed* sphere, so pymeshfix's connectivity-repairing
    loader leaves the mesh **untouched** (639 v / 1 250 f in and out, 2 boundaries), and nothing
    about the geometry was arranged to invert an ordering. The other fixture widens its south pole
    to make a 4-gon outlast a 16-gon in perimeter, which is what the threshold tests need and a
    count comparison does not. (Its pinhole was once two faces meeting only at a vertex -- a pinched
    rim that loader cuts, shifting every count by a triangle -- which is how this one came to be.)
    """
    sphere_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    centers_np = sphere_tm.triangles_center
    keep_np = np.ones(sphere_tm.faces.shape[0], dtype=bool)
    keep_np[np.argsort(-centers_np[:, 2])[:20]] = False
    keep_np[np.argsort(centers_np[:, 2])[:10]] = False
    holed_tm = tm.Trimesh(sphere_tm.vertices, sphere_tm.faces[keep_np], process=False)
    holed_tm.remove_unreferenced_vertices()
    vertices_wp, faces_wp = numpy_to_warp(holed_tm.vertices, holed_tm.faces, device)
    sizes = [loop_wp.size for loop_wp in od.boundary.boundary_loops(vertices_wp, faces_wp)]
    return holed_tm, vertices_wp, faces_wp, sizes


@pytest.mark.parity(
    "fill_small",
    "pymeshfix",
    "pymeshlab",
    benchmarked=False,
    reason="fill_small has no benchmark group: the interval DP it drives is what costs anything "
    "and that is already timed as fill_min_weight over the same loops, so a group here would "
    "re-measure the DP under a second name. Neither reference could carry a row anyway -- "
    "fill_small_boundaries is 8-10 % of a pymeshfix round (5.8 ms against 67.9 ms of load on "
    "bunny_decimated, 51.4 against 439.6 on bunny) and a PyTMesh takes exactly one load_array, so "
    "the build cannot leave the timed callable; meshing_close_holes sits behind pymeshlab's own "
    "~0.47 us/vertex MeshSet build. The threshold semantics are pinned here instead.",
)
@pytest.mark.parametrize("max_edges", [9, 10, 15, 16, 24])
def test_fill_small_max_edges_matches_the_edge_count_references(
    device: str, max_edges: int
) -> None:
    """
    Class B: the same loops are filled, after one named transform on pymeshlab's bound.

    ``max_edges`` exists because a **length** threshold is not expressible by two of the three
    references: pymeshfix's ``fill_small_boundaries(nbe, ...)`` and pymeshlab's
    ``meshing_close_holes(maxholesize=...)`` both count boundary edges, and no conversion between
    the two units exists without knowing the rim's sampling.

    The transform is the off-by-one, and it is measured rather than assumed, because the two
    references disagree with each other about it: on a 16-edge rim **pymeshfix fills at ``nbe=16``
    and pymeshlab only at ``maxholesize=17``**. So pymeshfix's bound is inclusive, pymeshlab's is
    exclusive, ordito follows pymeshfix (which is also what its own docstring gets wrong -- it says
    "less than"), and pymeshlab is called at ``max_edges + 1``.

    What is compared is the added-triangle count, which is an exact oracle for *which* loops were
    filled: an ``n``-gon patched without new vertices is ``n - 2`` triangles whatever the
    triangulation, so 8 means the 10-edge rim alone, 22 means both, 0 means neither. The
    triangulations themselves differ -- that is a separate, class-C claim -- and this row says
    nothing about them.

    Non-vacuous at both ends and in the middle: the parametrization straddles both rim sizes (10
    and 16), so two thresholds fill nothing, one fills one rim and two fill both, and a threshold
    rule that ignored its argument would fail at least one.
    """
    holed_tm, vertices_wp, faces_wp, sizes = _two_rims_of_different_edge_count(device)
    n_faces = faces_wp.size // 3
    assert sorted(sizes) == [10, 16]  # the fixture, so the thresholds above straddle both rims
    expected = sum(size - 2 for size in sizes if size <= max_edges)

    filled_wp = od.holes.fill_small(vertices_wp, faces_wp, max_edges=max_edges)

    tin_pmf = trimesh_to_pymeshfix(holed_tm)
    assert tin_pmf.n_faces == n_faces  # the loader left the mesh alone, so counts compare
    assert tin_pmf.n_boundaries == 2
    tin_pmf.fill_small_boundaries(nbe=max_edges, refine=False)
    added_pmf = tin_pmf.n_faces - n_faces

    meshset_pml = trimesh_to_pymeshlab(holed_tm)
    meshset_pml.meshing_close_holes(maxholesize=max_edges + 1, selfintersection=False)
    added_pml = meshset_pml.current_mesh().face_number() - n_faces

    assert filled_wp.size // 3 - n_faces == expected
    assert added_pmf == expected
    assert added_pml == expected


def test_fill_small_thresholds_select_opposite_loops(device: str) -> None:
    """
    Not a parity assert: ordito against ordito, pinning the two thresholds apart.

    The oracle for both branches is
    [`test_fill_small_max_edges_matches_the_edge_count_references`] (edges) and
    [`test_fill_small_fills_exactly_the_loops_under_the_threshold`] (perimeter); this test carries
    neither. Its job is to show the two thresholds are not two spellings of one thing, which no
    reference comparison can show because no reference implements both.

    The fixture is the one whose orderings are **inverted**: a 16-vertex rim of perimeter 2.406
    beside a 4-vertex rim of perimeter about 3.6, so "the small loop" is the 16-gon by length and
    the 4-gon by count. A threshold in the middle of each range therefore fills a *different* loop
    depending on which unit it is in, and the added-triangle counts (14 against 2) tell them apart.
    """
    vertices_wp, faces_wp, loop_sizes, perimeters = _two_holes_of_different_size(device)
    n_faces = faces_wp.size // 3
    by_size = dict(zip(loop_sizes, perimeters, strict=True))
    assert by_size[16] < by_size[4]  # the inversion this test turns on

    by_edges_wp = od.holes.fill_small(vertices_wp, faces_wp, max_edges=(4 + 16) // 2)
    by_length_wp = od.holes.fill_small(vertices_wp, faces_wp, sum(perimeters) / 2.0)

    assert by_edges_wp.size // 3 - n_faces == 4 - 2
    assert by_length_wp.size // 3 - n_faces == 16 - 2


class _FillSmallThresholds(TypedDict, total=False):
    max_perimeter: float
    max_edges: int


@pytest.mark.parametrize(
    "kwargs", [{}, {"max_perimeter": 1.0, "max_edges": 8}], ids=["neither", "both"]
)
def test_fill_small_requires_exactly_one_threshold(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], kwargs: _FillSmallThresholds
) -> None:
    """The documented ``ValueError``, in both directions: no threshold and two."""
    _mesh_tm, mesh_wp = hemisphere
    with pytest.raises(ValueError, match="exactly one"):
        od.holes.fill_small(mesh_wp.points, mesh_wp.indices, **kwargs)


@pytest.mark.parity(
    "fill_min_weight",
    "pymeshfix",
    benchmarked=False,
    reason="fill_small_boundaries is 8-10 % of a pymeshfix round -- 5.8 ms against 67.9 ms of load "
    "on bunny_decimated, 51.4 against 439.6 on bunny -- and a PyTMesh accepts exactly one "
    "load_array, so the build cannot leave the timed callable. A row would report a 90 % load as a "
    "hole fill, which is the failure mode the stitch/meshlib exemption already describes.",
)
@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus", "two_rims"])
def test_fill_min_weight_matches_pymeshfix(
    request: pytest.FixtureRequest, device: str, mesh_name: str
) -> None:
    """
    Class C (a surface distance): the same patch *surface*, from a genuinely different algorithm.

    It cannot be better than C, and the reason is the algorithm rather than a tolerance. MeshFix
    patches a hole by clipping ears smallest-angle-first -- a greedy in the Barequet-Sharir family
    -- where this runs the minimum-weight interval DP to optimality. Measured on the two-rim
    fixture: both produce 22 triangles and **0 of the 22 are shared**, so no relabelling makes the
    triangulations equal. What *is* comparable is everything else, and all of it is asserted:

    - the patch triangle count is exactly ``B - 2`` per rim on both sides, with no new vertices,
      which excludes a filler that fans through an added apex, that misses a rim, or that patches
      one twice;
    - both results are watertight with Euler characteristic 2, which excludes a self-intersecting
      or non-manifold patch;
    - the two surfaces agree to under 5 % of the mean edge length.

    That last bound is where the numbers matter. On ``hemisphere`` and ``half_torus`` the rim is a
    planar section, so *every* new-vertex-free triangulation of it spans the identical surface and
    the measurement is float noise (3.1e-08 and 4.3e-08) -- true, and weak. The ``two_rims`` case is
    the live one: its rims are non-planar, the two triangulations genuinely differ in space, and the
    distance reads **0.00165**, 1.1 % of the 0.1508 mean edge. Mutation probes on that same input,
    replacing the DP with a different filler: [`fill_fan`][ordito.holes.fill_fan] measures 0.0309
    (20.5 % of the mean edge) and [`fill_cone`][ordito.holes.fill_cone] 0.0257 (17.1 %), so the 5 %
    threshold sits 4.5x above the agreement and 4x below the nearest wrong answer.
    """
    if mesh_name == "two_rims":
        mesh_tm, vertices_wp, faces_wp, _sizes = _two_rims_of_different_edge_count(device)
    else:
        mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
        vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    n_faces = faces_wp.size // 3
    rim_sizes = [loop.size for loop in od.boundary.boundary_loops(vertices_wp, faces_wp)]
    expected = sum(size - 2 for size in rim_sizes)
    mean_edge = float(
        np.linalg.norm(
            mesh_tm.vertices[mesh_tm.edges_unique[:, 0]]
            - mesh_tm.vertices[mesh_tm.edges_unique[:, 1]],
            axis=1,
        ).mean()
    )

    filled_wp = od.holes.fill_min_weight(vertices_wp, faces_wp)
    filled_tm = warp_to_trimesh(vertices_wp, filled_wp)

    tin_pmf = trimesh_to_pymeshfix(mesh_tm)
    assert tin_pmf.n_faces == n_faces  # the loader left the mesh alone, so the counts compare
    assert tin_pmf.n_boundaries == len(rim_sizes)
    tin_pmf.fill_small_boundaries(nbe=0, refine=False)
    vertices_pmf, faces_pmf = pymeshfix_to_numpy(tin_pmf)
    filled_pmf = tm.Trimesh(vertices_pmf, faces_pmf, process=False)

    assert expected > 0  # non-vacuity: there was a hole to fill
    assert faces_pmf.shape[0] - n_faces == expected
    assert vertices_pmf.shape[0] == mesh_tm.vertices.shape[0]
    assert filled_wp.size // 3 - n_faces == expected
    assert filled_tm.is_watertight
    assert filled_pmf.is_watertight
    assert filled_tm.euler_number == 2
    assert filled_pmf.euler_number == 2
    distance = hausdorff_surface_two_sided(
        np.asarray(filled_tm.vertices), filled_tm.faces, vertices_pmf, faces_pmf
    )
    assert distance < 0.05 * mean_edge


@pytest.mark.parity(
    "fill_smooth_target_edge",
    "pymeshfix",
    benchmarked=False,
    reason="this test calls fill_smooth with no max_edge, so it runs that group's derived "
    "id -- the per-rim target measurement -- and holds the result to pymeshfix's refined "
    "patch. A row would re-time the fill_smooth group, since deriving the target is a stage "
    "of the same call rather than a separable operation.",
)
@pytest.mark.parity(
    "fill_smooth",
    "pymeshfix",
    benchmarked=False,
    reason="the refinement is 8-10 % of a pymeshfix round behind a load that cannot leave the "
    "timed callable (5.8 ms against 67.9 ms on bunny_decimated, 51.4 against 439.6 on bunny), so a "
    "row would price the load. The comparison is a density criterion rather than a cost anyway, "
    "which is what this test records.",
)
@pytest.mark.parity(
    "refine_region_to_density",
    "pymeshfix",
    benchmarked=False,
    reason="pymeshfix performs this refinement only as a stage inside fill_small_boundaries, "
    "behind a load that is ~90 % of the round and cannot be hoisted out of it, so a row would "
    "price the load. The criterion is what is comparable and the refine='density' arm below is "
    "where it lands -- fill_smooth dispatches that arm straight to refine_region_to_density.",
)
@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_fill_smooth_refinement_matches_pymeshfix(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class C: the same refined surface, and -- under ``refine="density"`` -- a comparable density.

    This row isolates the *refinement* stage, so ordito runs with ``smooth_curvature=False``:
    pymeshfix's ``refine=True`` densifies and Delaunay-flips the patch but never moves it off the
    triangulation it filled, and comparing against ordito's default -- which additionally smooths
    the patch into the surrounding curvature -- would confound two stages. The flag is named for the
    same reason potpourri3d's solvers are constructed with ``use_robust=False``: both sides have to
    be discretizing the same thing.

    With it off the two surfaces are **identical to float noise** -- 3.1e-08 on ``hemisphere`` and
    4.3e-08 on ``half_torus`` -- both watertight with Euler characteristic 2. That the numbers are
    that small rather than merely small is itself the finding: these rims are planar sections, so
    every unmoved refinement of the patch stays in the rim plane. The surface assert therefore
    excludes a refinement that leaves the plane, folds, or breaks watertightness, and says nothing
    about the sampling.

    The sampling is what ``refine`` selects, and it is the reason the density criterion exists.
    Measured as inserted vertices against pymeshfix's (47 on ``hemisphere``, 16 on ``half_torus``):

    | ``refine`` | hemisphere | half_torus |
    |---|---|---|
    | ``"max_edge"`` (default) | 201, **4.28x** | 192, **12.0x** |
    | ``"density"`` | 58, **1.23x** | 24, **1.50x** |

    ``"max_edge"`` bisects against one global target length and over-refines by an order of
    magnitude on the torus, where the rim is much finer than the mesh's mean edge; ``"density"``
    splits each patch triangle at its centroid only while its own sampling is coarser than the
    surrounding mesh's, which is MeshFix's criterion and lands within 1.5x of its count. The bound
    asserted here is 2x for ``"density"`` and only 20x for ``"max_edge"``, which is what makes the
    pair a comparison rather than two independent tolerances -- the default's row is recorded, not
    endorsed.

    Mutation probe for the surface bound, and it is a large one: ordito's own curvature smoothing
    moves the patch to **123 %** of the mean edge on ``hemisphere`` and 60 % on ``half_torus``,
    seven orders of magnitude past the 1e-6 threshold. So the assert is not something any patch of
    roughly the right shape would pass.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = mesh_wp.points.size

    tin_pmf = trimesh_to_pymeshfix(mesh_tm)
    assert tin_pmf.n_points == n_vertices  # the loader left the mesh alone
    tin_pmf.fill_small_boundaries(nbe=0, refine=True)
    vertices_pmf, faces_pmf = pymeshfix_to_numpy(tin_pmf)
    refined_pmf = tm.Trimesh(vertices_pmf, faces_pmf, process=False)
    inserted_pmf = vertices_pmf.shape[0] - n_vertices

    assert inserted_pmf > 0  # non-vacuity: the reference really refined
    assert refined_pmf.is_watertight
    assert refined_pmf.euler_number == 2

    refines: tuple[tuple[Literal["max_edge", "density"], float], ...] = (
        ("density", 2.0),
        ("max_edge", 20.0),
    )
    for refine, ratio_bound in refines:
        refined_wp, refined_faces_wp = od.holes.fill_smooth(
            mesh_wp.points, mesh_wp.indices, smooth_curvature=False, refine=refine
        )
        refined_tm = warp_to_trimesh(refined_wp, refined_faces_wp)
        inserted_wp = refined_wp.size - n_vertices

        assert inserted_wp > 0
        assert inserted_wp / inserted_pmf < ratio_bound, (refine, inserted_wp, inserted_pmf)
        assert refined_tm.is_watertight
        assert refined_tm.euler_number == 2
        assert (
            hausdorff_surface_two_sided(
                np.asarray(refined_tm.vertices), refined_tm.faces, vertices_pmf, faces_pmf
            )
            < 1e-6
        )


# ---------------------------------------------------------------------------
# fill_smooth (MeshLib fillHoleNicely)
# ---------------------------------------------------------------------------


def _mesh_volume_area(vertices_np: np.ndarray, faces_np: np.ndarray) -> tuple[float, float]:
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return float(mesh.volume), float(mesh.area)


def _meshlib_fill_nicely_volume(
    vertices_np: np.ndarray, faces_np: np.ndarray, max_edge: float
) -> float:
    mesh = numpy_to_meshlib(vertices_np, faces_np)
    settings = mm.FillHoleNicelySettings()
    settings.subdivideSettings.maxEdgeLen = max_edge
    for edge in mesh.topology.findHoleRepresentiveEdges():
        mm.fillHoleNicely(mesh, edge, settings)
    verts = mn.getNumpyVerts(mesh)
    faces = mn.getNumpyFaces(mesh.topology)
    return float(tm.Trimesh(verts, faces, process=False).volume)


def test_fill_smooth_invariants(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    Not a library comparison: what ``fill_smooth`` must hold whatever it chooses to add.

    No boundary loop left, winding consistent, and the input's vertices unmoved in the prefix
    -- that last separates *filling* from remeshing the whole surface. MeshLab supplies the
    volume comparison in [`test_fill_smooth_statistics_vs_meshlib`]; it cannot supply these,
    because it closes nothing at its own default (section 6).
    """
    _, mesh_wp = hemisphere
    n_v0 = mesh_wp.points.size

    new_vertices, new_faces, patch = od.holes.fill_smooth(
        mesh_wp.points, mesh_wp.indices, return_patch=True
    )
    verts_np = new_vertices.numpy()
    faces_np = new_faces.numpy().reshape(-1, 3)

    assert len(_loop_sizes_of(new_vertices, new_faces)) == 0
    mesh_tm = tm.Trimesh(verts_np, faces_np, process=False)
    assert mesh_tm.is_watertight
    assert mesh_tm.is_winding_consistent
    assert od.validation.is_edge_manifold(new_faces)
    assert int(patch.numpy().sum()) > 0
    assert np.allclose(verts_np[:n_v0], mesh_wp.points.numpy(), atol=1e-6)


def test_fill_smooth_triangulate_only(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = hemisphere
    new_vertices, new_faces = od.holes.fill_smooth(
        mesh_wp.points, mesh_wp.indices, triangulate_only=True
    )
    expected_faces = od.holes.fill_min_weight(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(new_faces.numpy(), expected_faces.numpy())
    assert np.array_equal(new_vertices.numpy(), mesh_wp.points.numpy())


@pytest.mark.parity(
    "fill_smooth_target_edge",
    "meshlib",
    benchmarked=False,
    reason="this test passes max_edge=0.3, which is that group's explicit id, and MeshLib is "
    "handed the identical number -- so the pair covers the half the derived row is measured "
    "against. A row would re-time the fill_smooth group under a second name.",
)
@pytest.mark.parity("fill_smooth", "meshlib")
def test_fill_smooth_statistics_vs_meshlib(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class C (a derived scalar): the filled volume, because the two patches share no vertices.

    MeshLab's ``fillHoleNicely`` subdivides and smooths on its own schedule, so no
    correspondence exists between the two caps -- the enclosed volume is the strongest
    comparable quantity, and it excludes a cap that bulges or collapses while still being
    watertight.
    """
    _, mesh_wp = hemisphere
    vertices_np = mesh_wp.points.numpy().astype(np.float64)
    faces_np = mesh_wp.indices.numpy().reshape(-1, 3)
    max_edge = 0.3

    new_vertices, new_faces = od.holes.fill_smooth(
        mesh_wp.points, mesh_wp.indices, max_edge=max_edge
    )
    volume_od, _ = _mesh_volume_area(new_vertices.numpy(), new_faces.numpy().reshape(-1, 3))
    volume_ml = _meshlib_fill_nicely_volume(vertices_np, faces_np, max_edge)
    assert np.isclose(volume_od, volume_ml, rtol=0.05)


@pytest.mark.parity("refill_region", "meshlib")
def test_refill_region_matches_meshlib(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A on the triangulation, Class C on the refinement: the same patch as ``patchMesh``.

    MeshLib's ``patchMesh`` is this operation -- delete a face region, fill the hole nicely -- and
    in ``triangulateOnly`` mode the two agree **exactly**: measured 593 vertices, 1 182 faces and an
    enclosed volume of 4.0771 against 4.0770 on a subdivision-3 icosphere with a cap of 132 faces
    removed. That is as strong as this comparison can be, because the minimum-weight DP has one
    answer and both implementations find it.

    With refinement on, the two diverge by *density* rather than by shape -- each subdivides on its
    own schedule (766 vertices here against MeshLib's 644) -- so that half is the enclosed volume,
    which both bring back close to the original 4.1527: 4.1539 and 4.1372. The refined patch being
    *nearer* the original than the flat one (4.0771) is what matters, and is asserted.

    ``patchMesh`` mutates its mesh, so each mode gets a fresh one.
    """
    mesh_tm, mesh_wp = icosphere
    n_faces = mesh_tm.faces.shape[0]
    region_np = np.asarray(mesh_tm.triangles_center)[:, 2] > 0.8
    assert 0 < int(region_np.sum()) < n_faces
    region_wp = wp.array(region_np, dtype=wp.bool, device=mesh_wp.points.device)
    volume_before = float(od.measures.volume(mesh_wp.points, mesh_wp.indices))

    flat_vertices_wp, flat_faces_wp = od.holes.refill_region(
        mesh_wp.points, mesh_wp.indices, region_wp, triangulate_only=True
    )
    mesh_flat_ml = trimesh_to_meshlib(mesh_tm)
    region_flat_ml = mm.FaceBitSet(numpy_to_meshlib_bitset(region_np))
    region_flat_ml.resize(n_faces)
    flat_settings_ml = mm.FillHoleNicelySettings()
    flat_settings_ml.triangulateOnly = True
    patch_flat_ml = mm.patchMesh(mesh_flat_ml, region_flat_ml, flat_settings_ml)
    flat_ml = meshlib_to_trimesh(mesh_flat_ml)

    assert patch_flat_ml.count() > 0  # non-vacuity: the reference filled something
    assert flat_vertices_wp.size == flat_ml.vertices.shape[0]
    assert flat_faces_wp.size // 3 == flat_ml.faces.shape[0]
    assert np.isclose(
        float(od.measures.volume(flat_vertices_wp, flat_faces_wp)),
        flat_ml.volume,
        rtol=1e-4,
        atol=1e-6,
    )
    assert od.validation.is_watertight(flat_vertices_wp, flat_faces_wp)

    refined_vertices_wp, refined_faces_wp = od.holes.refill_region(
        mesh_wp.points, mesh_wp.indices, region_wp
    )
    mesh_refined_ml = trimesh_to_meshlib(mesh_tm)
    region_refined_ml = mm.FaceBitSet(numpy_to_meshlib_bitset(region_np))
    region_refined_ml.resize(n_faces)
    mm.patchMesh(mesh_refined_ml, region_refined_ml, mm.FillHoleNicelySettings())
    refined_ml = meshlib_to_trimesh(mesh_refined_ml)

    volume_flat = float(od.measures.volume(flat_vertices_wp, flat_faces_wp))
    volume_refined = float(od.measures.volume(refined_vertices_wp, refined_faces_wp))
    assert np.isclose(volume_refined, refined_ml.volume, rtol=0.02)
    # The refinement is worth having: it recovers curvature the flat cap loses.
    assert abs(volume_refined - volume_before) < abs(volume_flat - volume_before)
    assert od.validation.is_watertight(refined_vertices_wp, refined_faces_wp)


def test_refill_region_leaves_an_untouched_mesh_alone(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: an empty region, and one whose removal opens no *new* rim.

    Both must return the surviving mesh with an all-``False`` patch rather than filling something.
    The second case is the one worth pinning: deleting a face that lies on the hemisphere's existing
    rim extends that rim, and the extension *is* reported as a loop -- so this asserts the opposite
    case, an empty mask, where nothing is deleted and nothing may be filled. A version that filled
    the input's own rim here would look like a working hole-filler and be the wrong function.
    """
    mesh_tm, mesh_wp = hemisphere
    n_faces = mesh_tm.faces.shape[0]
    empty_wp = wp.zeros(n_faces, dtype=wp.bool, device=mesh_wp.points.device)

    vertices_wp, faces_wp, patch_wp = od.holes.refill_region(
        mesh_wp.points, mesh_wp.indices, empty_wp, return_patch=True
    )
    assert faces_wp.size // 3 == n_faces
    assert not bool(patch_wp.numpy().any())
    assert np.allclose(vertices_wp.numpy(), mesh_wp.points.numpy())
    # Still open: the rim it arrived with is untouched.
    assert int(od.boundary.boundary_edges(vertices_wp, faces_wp).shape[0]) > 0

    with pytest.raises(ValueError, match="metric must be one of"):
        od.holes.refill_region(mesh_wp.points, mesh_wp.indices, empty_wp, metric="nonsense")
    with pytest.raises(ValueError, match="one entry per face"):
        od.holes.refill_region(
            mesh_wp.points,
            mesh_wp.indices,
            wp.zeros(3, dtype=wp.bool, device=mesh_wp.points.device),
        )


def test_fill_smooth_natural_smooth(device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    Not a library comparison: ``natural_smooth`` must blend the patch into the rim it meets.

    The claim is about the *rim*, which no reference exposes separately: with the flag on,
    curvature continues across the boundary instead of creasing there. trimesh supplies only
    the surface geometry the assertion is computed from.
    """
    sphere, _sphere_wp = icosphere
    hemi = sphere.slice_plane(
        plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]), cap=False
    )
    hemi.merge_vertices()
    vertices_np = np.ascontiguousarray(hemi.vertices.astype(np.float64))
    faces_np = np.ascontiguousarray(hemi.faces.astype(np.int32).reshape(-1))
    v_wp, f_wp = numpy_to_warp(vertices_np, faces_np, device)
    n_v0 = len(vertices_np)

    verts_off = od.holes.fill_smooth(v_wp, f_wp, natural_smooth=False)[0].numpy()
    verts_on = od.holes.fill_smooth(v_wp, f_wp, natural_smooth=True)[0].numpy()

    # naturalSmooth grows a collar past the rim, so some original vertices move.
    disp = np.linalg.norm(verts_off[:n_v0] - verts_on[:n_v0], axis=1)
    assert int((disp > 1e-5).sum()) > 0

    _, new_faces = od.holes.fill_smooth(v_wp, f_wp, natural_smooth=True)
    mesh_tm = tm.Trimesh(verts_on, new_faces.numpy().reshape(-1, 3), process=False)
    assert mesh_tm.is_watertight
    assert mesh_tm.is_winding_consistent


# ---------------------------------------------------------------------------
# empty mesh / already-watertight mesh, across every hole-filling operator
# ---------------------------------------------------------------------------

# (name, callable) pairs; each callable takes (vertices, faces) and returns the tuple of arrays
# the wrapper produces -- fill_fan and fill_min_weight return faces only, fill_cone and
# fill_smooth return (vertices, faces).
_HOLE_FILL_CASES = [
    ("fill_fan", lambda v, f: (od.holes.fill_fan(v, f),)),
    ("fill_cone", lambda v, f: od.holes.fill_cone(v, f)),
    ("fill_min_weight", lambda v, f: (od.holes.fill_min_weight(v, f),)),
    ("fill_smooth", lambda v, f: od.holes.fill_smooth(v, f)),
]

# The same callables paired with what each must equal on an already-watertight mesh -- faces
# alone for the single-output wrappers, (vertices, faces) for the pair-output ones -- so the
# comparison never has to infer which reference goes with which result from its length.
_HOLE_FILL_WATERTIGHT_CASES = [
    ("fill_fan", lambda v, f: (od.holes.fill_fan(v, f),), lambda v, f: (f,)),
    ("fill_cone", lambda v, f: od.holes.fill_cone(v, f), lambda v, f: (v, f)),
    ("fill_min_weight", lambda v, f: (od.holes.fill_min_weight(v, f),), lambda v, f: (f,)),
    ("fill_smooth", lambda v, f: od.holes.fill_smooth(v, f), lambda v, f: (v, f)),
]


@pytest.mark.parametrize(
    ("hole_fill_fn", "reference_fn"),
    [case[1:] for case in _HOLE_FILL_WATERTIGHT_CASES],
    ids=[case[0] for case in _HOLE_FILL_WATERTIGHT_CASES],
)
def test_holes_watertight_mesh_is_unchanged(
    hole_fill_fn: Callable[[wp.array[wp.vec3], wp.array[wp.int32]], tuple[wp.array[Any], ...]],
    reference_fn: Callable[[wp.array[wp.vec3], wp.array[wp.int32]], tuple[wp.array[Any], ...]],
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Not a library comparison: an already-closed mesh has no hole for any of these to fill."""
    _, mesh_wp = icosahedron
    result = hole_fill_fn(mesh_wp.points, mesh_wp.indices)
    expected = reference_fn(mesh_wp.points, mesh_wp.indices)
    for got_wp, expected_wp in zip(result, expected, strict=True):
        assert np.array_equal(got_wp.numpy(), expected_wp.numpy())


# fill_smooth carries no empty-mesh case: unlike the other three, nothing in its own test file
# ever exercised it on a zero-face mesh, and this merge is a consolidation, not new coverage.
@pytest.mark.parametrize(
    "hole_fill_fn",
    [case[1] for case in _HOLE_FILL_CASES if case[0] != "fill_smooth"],
    ids=[case[0] for case in _HOLE_FILL_CASES if case[0] != "fill_smooth"],
)
def test_holes_empty_mesh_is_a_noop(
    device: str,
    hole_fill_fn: Callable[[wp.array[wp.vec3], wp.array[wp.int32]], tuple[wp.array[Any], ...]],
) -> None:
    """Not a library comparison: every output array stays empty on a zero-face mesh."""
    vertices = warp_empty(0, wp.vec3, device)
    faces = warp_empty(0, wp.int32, device)
    for result_wp in hole_fill_fn(vertices, faces):
        assert result_wp.size == 0


# --- Stitching two open meshes across one boundary loop each ----------------------------------


def _cone(
    n: int,
    apex_z: float,
    rim_z: float,
    radius: float = 1.0,
    phase: float = 0.0,
    center_x: float = 0.0,
):
    """
    Open triangle-fan cone: an apex plus one rim circle. Its boundary is the rim loop.

    Returns ``(vertices_np, faces_np)`` with vertex 0 the apex and vertices ``1..n`` the rim,
    wound so the surface is consistently oriented. ``center_x`` shifts the rim laterally, which
    makes the two-rim correspondence non-monotone and exercises the LIS correction.
    """
    angles = phase + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rim = np.column_stack(
        [center_x + radius * np.cos(angles), radius * np.sin(angles), np.full(n, rim_z)]
    )
    vertices = np.vstack([[center_x, 0.0, apex_z], rim]).astype(np.float64)
    apex_above_rim = apex_z > rim_z
    faces = np.empty((n, 3), dtype=np.int32)
    for i in range(n):
        first, second = 1 + i, 1 + (i + 1) % n
        faces[i] = (0, first, second) if apex_above_rim else (0, second, first)
    return vertices, faces.reshape(-1)


def _cone_wp(
    device: wp.DeviceLike,
    n: int,
    apex_z: float,
    rim_z: float,
    radius: float = 1.0,
    phase: float = 0.0,
    center_x: float = 0.0,
):
    vertices_np, faces_np = _cone(n, apex_z, rim_z, radius, phase, center_x)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    return vertices_np, faces_np, vertices_wp, faces_wp


def _capsule_halves(device: str, n_a: int, n_b: int, phase: float = 0.0, offset: float = 0.0):
    """
    Two open cones whose rims are separated in ``z`` (a real frustum band, no overlap).

    ``offset`` shifts the top rim laterally so the rim-to-rim correspondence is non-monotone,
    exercising the LIS correction; the band stays non-degenerate.
    """
    bottom = _cone_wp(device, n=n_a, apex_z=-1.0, rim_z=0.0)
    top = _cone_wp(device, n=n_b, apex_z=1.5, rim_z=0.5, phase=phase, center_x=offset)
    return bottom, top


@pytest.mark.parametrize(
    ("n_a", "n_b", "phase", "offset"),
    [
        (8, 8, 0.0, 0.0),
        (16, 11, 0.3, 0.0),
        (7, 13, 0.7, 0.0),
        (24, 5, 1.1, 0.0),
        (17, 11, 0.9, 1.2),
    ],
)
def test_stitch_watertight(device: str, n_a: int, n_b: int, phase: float, offset: float) -> None:
    (_, _, va, fa), (_, _, vb, fb) = _capsule_halves(device, n_a, n_b, phase, offset)

    assert not od.validation.is_watertight(va, fa)
    assert not od.validation.is_watertight(vb, fb)

    new_vertices, new_faces = od.holes.stitch(va, fa, vb, fb)

    assert od.validation.is_watertight(new_vertices, new_faces)
    assert od.validation.is_winding_consistent(new_faces)

    # A + B faces plus one bridge triangle per rim edge on each side.
    n_new_faces = (new_faces.size - fa.size - fb.size) // 3
    assert n_new_faces == n_a + n_b
    assert new_vertices.size == va.size + vb.size


def test_stitch_argument_order_invariant(device: str) -> None:
    (_, _, va, fa), (_, _, vb, fb) = _capsule_halves(device, 16, 11, phase=0.3)

    vertices_ab, faces_ab = od.holes.stitch(va, fa, vb, fb)
    vertices_ba, faces_ba = od.holes.stitch(vb, fb, va, fa)

    # The larger loop is always A, so swapping the arguments yields the same mesh.
    assert np.array_equal(vertices_ab.numpy(), vertices_ba.numpy())
    assert np.array_equal(
        lexsort_rows(np.sort(faces_ab.numpy().reshape(-1, 3), axis=1)),
        lexsort_rows(np.sort(faces_ba.numpy().reshape(-1, 3), axis=1)),
    )


@pytest.mark.parametrize(
    ("n_a", "n_b", "phase", "offset"), [(16, 11, 0.3, 0.0), (17, 11, 0.9, 0.4)]
)
@pytest.mark.parity(
    "stitch",
    "meshlib",
    benchmarked=False,
    reason="MeshLib has no zipper. Its two-argument stitchHoles finds the holes itself and then "
    "runs the same minimum-weight DP its three-argument form runs, which is already timed in the "
    "stitch_min_weight group -- a row here would price that DP under the zipper's name and read as "
    "the zipper being 100x slower than it is.",
)
def test_stitch_reaches_meshlibs_optimum(
    device: str, n_a: int, n_b: int, phase: float, offset: float
) -> None:
    """
    Class C (a derived scalar): the greedy zipper's band, scored by MeshLib's own metric.

    ``stitch`` walks the two rims from the cheapest starting correspondence and never reconsiders,
    where MeshLib's two-argument ``stitchHoles`` -- the overload that finds the holes itself, so
    ordito's loop *pairing* is compared too and not only its triangulation -- searches the whole
    space. The zipper is therefore an upper bound on the optimum by construction, and the question
    a comparison can answer is how loose a bound.

    Measured over eight rim configurations: **the zipper reaches the optimum exactly (ratio 1.0000)
    on seven of them**, and costs 18.9% more on the one whose top rim is shifted sideways, where the
    rim-to-rim correspondence stops being monotone and greed is provably not enough. Those are the
    two cases parametrized here, so the test covers both the tight branch and the loose one rather
    than only the flattering half.

    What it excludes is a zipper that emits a *valid but badly shaped* band -- a slipped
    correspondence costs several times the optimum, not 19% -- and what it cannot see is a different
    tie-break at equal cost, which is real: 4 of the 8 bands differ from MeshLib's triangle for
    triangle at an identical score, so comparing triangles rather than cost would fail on a
    symmetric rim. The cost helper is shared with the minimum-weight section below.
    """
    (va_np, fa_np, va, fa), (vb_np, fb_np, vb, fb) = _capsule_halves(
        device, n_a, n_b, phase, offset
    )

    new_vertices, new_faces = od.holes.stitch(va, fa, vb, fb)
    n_orig = fa.size + fb.size
    band_od = new_faces.numpy()[n_orig:].reshape(-1, 3)
    # ``stitch`` puts the larger-boundary mesh first, so rebuild the prefix in the order it used.
    fa_rows, fb_rows = fa_np.reshape(-1, 3), fb_np.reshape(-1, 3)
    if n_a >= n_b:
        orig_od = np.vstack([fa_rows, fb_rows + len(va_np)]).astype(np.int32)
    else:
        orig_od = np.vstack([fb_rows, fa_rows + len(vb_np)]).astype(np.int32)

    verts_ml = np.ascontiguousarray(np.vstack([va_np, vb_np]), dtype=np.float32)
    orig_ml = np.ascontiguousarray(np.vstack([fa_rows, fb_rows + len(va_np)]), dtype=np.int32)
    mesh_ml = numpy_to_meshlib(verts_ml, orig_ml)
    assert len(mesh_ml.topology.findHoleRepresentiveEdges()) == 2  # the auto-detect's input
    assert mm.stitchHoles(mesh_ml, mm.StitchHolesParams())  # the two-argument overload, and it ran
    faces_out_ml = mn.getNumpyFaces(mesh_ml.topology)
    original = {tuple(sorted(int(x) for x in row)) for row in orig_ml}
    band_ml = np.array(
        [row for row in faces_out_ml if tuple(sorted(int(x) for x in row)) not in original],
        dtype=np.int32,
    )

    # Both close the two rims with one triangle per rim edge and no new vertex.
    assert len(band_od) == len(band_ml) == n_a + n_b
    assert new_vertices.size == len(va_np) + len(vb_np)
    assert od.validation.is_watertight(new_vertices, new_faces)

    cost_od = _meshlib_stitch_cost(new_vertices.numpy(), orig_od, band_od, "complex_stitch")
    cost_ml = _meshlib_stitch_cost(verts_ml, orig_ml, band_ml, "complex_stitch")
    assert cost_ml > 0.0  # non-vacuity: a zero optimum would make any ratio pass
    assert cost_od >= cost_ml * (1.0 - 1e-6)  # the greedy band cannot beat the exhaustive one
    assert cost_od < 1.25 * cost_ml  # measured 1.0000 and 1.1889 on these two configurations


# --- Minimum-weight stitching (``stitch_min_weight``) ------------------------------------------

STITCH_METRICS = ["complex_stitch", "edge_length_stitch", "vertical"]
# complex_stitch (aspect + dihedral) and vertical (area/normal) are winding-invariant, so MeshLib's
# calcCombinedFillMetric re-scores them exactly; edge_length_stitch's |c-a| term is winding-order
# sensitive, so it is checked structurally only.
STITCH_COST_METRICS = ["complex_stitch", "vertical"]


# MeshLib's factory for each ``STITCH_METRICS`` entry, called with the mesh.
_MESHLIB_STITCH_METRICS: dict[str, Callable[[mm.Mesh], mm.FillHoleMetric]] = {
    "complex_stitch": lambda m: mm.getComplexStitchMetric(m),
    "edge_length_stitch": lambda m: mm.getEdgeLengthStitchMetric(m),
    "vertical": lambda m: mm.getVerticalStitchMetric(m, mm.Vector3f(0.0, 0.0, 1.0)),
}


def _meshlib_stitch_band(
    va_np: np.ndarray, fa_np: np.ndarray, vb_np: np.ndarray, fb_np: np.ndarray, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MeshLib ``stitchHoles`` band for the same metric; returns ``(verts, orig_faces, band)``."""
    make_metric = _MESHLIB_STITCH_METRICS[metric]
    verts = np.ascontiguousarray(np.vstack([va_np, vb_np]), dtype=np.float32)
    orig = np.ascontiguousarray(np.vstack([fa_np, fb_np + len(va_np)]), dtype=np.int32)
    mesh = numpy_to_meshlib(verts, orig)
    edges = mesh.topology.findHoleRepresentiveEdges()
    params = mm.StitchHolesParams()
    params.metric = make_metric(mesh)
    mm.stitchHoles(mesh, edges[0], edges[1], params)
    faces_out = mn.getNumpyFaces(mesh.topology)
    original = {tuple(sorted(int(x) for x in t)) for t in orig}
    band = np.array(
        [t for t in faces_out if tuple(sorted(int(x) for x in t)) not in original], np.int32
    )
    return verts, orig, band


def _meshlib_stitch_cost(
    verts: np.ndarray, orig: np.ndarray, band: np.ndarray, metric: str
) -> float:
    make_metric = _MESHLIB_STITCH_METRICS[metric]
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    mesh_orig = numpy_to_meshlib(verts, orig)
    metric_obj = make_metric(mesh_orig)
    full = np.ascontiguousarray(np.vstack([orig, band]), dtype=np.int32)
    mesh_full = numpy_to_meshlib(verts, full)
    region_bools = np.zeros(len(full), dtype=bool)
    region_bools[len(orig) :] = True
    region = mn.faceBitSetFromBools(cast("Buffer", region_bools))
    return mm.calcCombinedFillMetric(mesh_full, region, metric_obj)


@pytest.mark.parametrize(("n_a", "n_b"), [(9, 13), (16, 11), (8, 8)])
@pytest.mark.parametrize("metric", STITCH_METRICS)
def test_stitch_min_weight_watertight(device: str, n_a: int, n_b: int, metric: str) -> None:
    """
    Not a library comparison: the band's exact triangle count, with trimesh as the oracle.

    Two rims of ``n_a`` and ``n_b`` vertices close with exactly ``n_a + n_b`` triangles and no new
    vertices, whatever the metric chooses -- an arithmetic reference, not another implementation.
    The metric's *choice* among those triangulations is what
    [`test_stitch_min_weight_matches_meshlib`] compares.
    """
    (_, _, va, fa), (_, _, vb, fb) = _capsule_halves(device, n_a, n_b)

    new_vertices, new_faces = od.holes.stitch_min_weight(va, fa, vb, fb, metric=metric)

    # A band of exactly n_a + n_b triangles over the existing vertices, closing the two rims.
    n_band = (new_faces.size - fa.size - fb.size) // 3
    assert n_band == n_a + n_b
    assert new_vertices.size == va.size + vb.size
    assert od.validation.is_winding_consistent(new_faces)
    filled_tm = warp_to_trimesh(new_vertices, new_faces)
    assert filled_tm.is_watertight


@pytest.mark.parametrize(("n_a", "n_b"), [(8, 8), (40, 37), (64, 33)])
@pytest.mark.parametrize("metric", STITCH_METRICS)
@pytest.mark.parametrize("tile", [32, 64])
def test_stitch_dp_tile_matches_diagonal(
    monkeypatch: pytest.MonkeyPatch, device: str, n_a: int, n_b: int, metric: str, tile: int
) -> None:
    """
    Ordito against ordito: the tiled band DP must emit the **identical band** as the diagonal one.

    ``stitch_dp_diag`` carries the oracle here -- it is the straight reading of the recurrence, one
    launch per anti-diagonal -- and ``stitch_dp_tile`` is the fast schedule that has to agree with
    it. Both call the same ``stitch_dp_cell``, so the only thing that can differ is *ordering*: the
    tiled kernel replaces all but ``2 * ceil(n / tile)`` of the launch barriers with a block
    barrier, and a missing or mis-placed one would let a cell read a predecessor that a lane in
    another warp had not written yet. Nothing else in this file would see that -- the band would
    still be watertight, still ``n_a + n_b`` triangles, and still plausible.

    **``tile`` is parametrized because the shipped value cannot fail.** At 32 the block is a single
    warp and needs no barrier at all: deleting the ``wp.tile_sum`` leaves this green at 32 and
    fails it on every run at 64. So the 64 arm is the one that pins the barrier, and the 32 arm
    pins the value actually shipped. The three ``(n_a, n_b)`` pairs cover a grid inside one tile, a
    grid of several whole-and-partial tiles, and one that is partial on one axis only.
    """
    monkeypatch.setattr(kernel_holes, "STITCH_DP_TILE", tile)
    (_, _, va, fa), (_, _, vb, fb) = _capsule_halves(device, n_a, n_b, phase=0.3)

    original = od.holes._run_stitch_dp  # pyright: ignore[reportPrivateUsage]
    bands = {}
    for tiled in (True, False):
        monkeypatch.setattr(
            od.holes,
            "_run_stitch_dp",
            lambda *args, _tiled=tiled, **kwargs: original(*args, **kwargs, tiled=_tiled),
        )
        _, faces = od.holes.stitch_min_weight(va, fa, vb, fb, metric=metric)
        bands[tiled] = faces.numpy()[fa.size + fb.size :]

    assert bands[True].size == 3 * (n_a + n_b), "fixture produced no band to compare"
    assert np.array_equal(bands[True], bands[False])


@pytest.mark.parity("stitch_min_weight", "meshlib")
@pytest.mark.parametrize(("n_a", "n_b"), [(9, 13), (16, 11)])
@pytest.mark.parametrize("metric", STITCH_COST_METRICS)
def test_stitch_min_weight_matches_meshlib(device: str, n_a: int, n_b: int, metric: str) -> None:
    """
    Class C (a derived scalar): the band's *cost* matches MeshLib's optimum, not its triangles.

    Both sides minimize the same objective over the same rims, and several triangulations can reach
    the optimum -- so the comparable quantity is the cost, evaluated by one shared function on both
    answers. What this excludes is ordito settling for a worse triangulation; what it cannot see is
    a different tie-break at equal cost, which is the point of comparing costs.
    """
    (va_np, fa_np, va, fa), (vb_np, fb_np, vb, fb) = _capsule_halves(device, n_a, n_b)

    new_vertices, new_faces = od.holes.stitch_min_weight(va, fa, vb, fb, metric=metric)
    n_orig = fa.size + fb.size  # flat length of the two original face buffers
    band_od = new_faces.numpy()[n_orig:].reshape(-1, 3)
    fa_rows, fb_rows = fa_np.reshape(-1, 3), fb_np.reshape(-1, 3)
    orig_od = np.vstack([fa_rows, fb_rows + len(va_np)]).astype(np.int32)

    verts_ml, orig_ml, band_ml = _meshlib_stitch_band(va_np, fa_rows, vb_np, fb_rows, metric)

    # ordito reaches MeshLib's exhaustive stitchHoles optimum (compare cost, not exact triangles).
    cost_od = _meshlib_stitch_cost(new_vertices.numpy(), orig_od, band_od, metric)
    cost_ml = _meshlib_stitch_cost(verts_ml, orig_ml, band_ml, metric)
    assert np.isclose(cost_od, cost_ml, rtol=3e-3, atol=1e-3)


@pytest.mark.parametrize(("n_a", "n_b", "offset"), [(9, 13, 0.0), (7, 11, 1.2), (16, 5, 0.6)])
def test_stitch_min_weight_edge_length_reaches_the_band_optimum(
    device: str, n_a: int, n_b: int, offset: float
) -> None:
    """
    Not a library comparison: no reference exposes this objective, so the oracle is a host DP.

    ``edge_length_stitch`` is documented as *"summed connection-edge length"*, and that objective
    has a closed form the test can brute-force in five lines: whichever rim a step advances, the
    connection edge it introduces is ``(a[i], b[j])`` -- the other two corners of the new band
    triangle were already joined at the cell it came from -- so
    ``dp[i, j] = min(dp[i - 1, j], dp[i, j - 1]) + |a_i - b_j|`` over the same aligned, rolled rims
    the wrapper builds. The band's own total is read back off its triangles: every connection edge
    is shared by exactly two consecutive band triangles and each triangle has exactly two of them,
    so half the sum over triangles is the path cost.

    **Bug class excluded:** a DP branch scoring the wrong edge, which is invisible to a
    watertightness or triangle-count check and to a cost comparison that evaluates *ordito's own*
    metric on both answers. **Mutation probe, measured against a detached pre-fix worktree:** with
    the advance-A branch scoring the A rim edge (a path-independent constant) instead of the
    connection edge, the band comes back 1.03-1.17x above the optimum and this fails; the same
    fixtures stay watertight and keep the exact ``n_a + n_b`` triangle count either way.
    """
    (va_np, _fa_np, va, fa), (vb_np, _fb_np, vb, fb) = _capsule_halves(
        device, n_a, n_b, phase=0.7, offset=offset
    )
    loop_a = od.boundary.boundary_loops(va, fa)[0]
    loop_b = od.boundary.boundary_loops(vb, fb)[0]

    new_vertices, new_faces = od.holes.stitch_loops_min_weight(
        va, fa, loop_a, vb, fb, loop_b, metric="edge_length_stitch"
    )

    offset_b = va.size
    band = new_faces.numpy()[fa.size + fb.size :].reshape(-1, 3)
    assert band.shape[0] == n_a + n_b  # non-vacuity: a full band, not a degenerate one
    positions = new_vertices.numpy()
    total = 0.0
    for triangle in band:
        for u, v in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            if (u < offset_b) != (v < offset_b):
                total += float(np.linalg.norm(positions[u] - positions[v]))
    total *= 0.5

    assert np.isclose(total, _band_edge_length_optimum(va_np, loop_a, vb_np, loop_b), rtol=1e-5)


def _band_edge_length_optimum(
    vertices_a_np: np.ndarray,
    loop_a: wp.array[wp.int32],
    vertices_b_np: np.ndarray,
    loop_b: wp.array[wp.int32],
) -> float:
    """
    Host optimum of the ``edge_length_stitch`` objective over the two rims, by exhaustive DP.

    Mirrors ``holes.stitch_loops_min_weight``'s alignment exactly -- rim A reversed, both rolled to
    their closest vertex pair (``numpy.argmin`` on the flattened distance matrix is that
    function's documented tie-break) -- and then walks every monotone staircase. The two cells that
    would consume a whole rim before touching the other are the kernel's own forbidden cells.
    """
    la = loop_a.numpy()[::-1].copy()
    lb = loop_b.numpy().copy()
    rim_a, rim_b = vertices_a_np[la], vertices_b_np[lb]
    distance = np.linalg.norm(rim_a[:, None, :] - rim_b[None, :, :], axis=2)
    start_a, start_b = np.unravel_index(int(np.argmin(distance)), distance.shape)
    a_pos = np.roll(rim_a, -int(start_a), axis=0).astype(np.float64)
    b_pos = np.roll(rim_b, -int(start_b), axis=0).astype(np.float64)

    n_a, n_b = len(la), len(lb)
    dp = np.full((n_a + 1, n_b + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(n_a + 1):
        for j in range(n_b + 1):
            if (i == 0 and j == 0) or (i == n_a and j == 0) or (j == n_b and i == 0):
                continue
            step = float(np.linalg.norm(a_pos[i % n_a] - b_pos[j % n_b]))
            best = np.inf
            if i >= 1:
                best = min(best, dp[i - 1, j] + step)
            if j >= 1:
                best = min(best, dp[i, j - 1] + step)
            dp[i, j] = best
    return float(dp[n_a, n_b])


# ---------------------------------------------------------------------------
# stitch_smooth (MeshLib stitchHolesNicely)
# ---------------------------------------------------------------------------


def _hemisphere_pair(device: str):
    """Two facing hemispheres (single boundary loop each) for stitch tests."""
    meshes = []
    for z_sign, z_off in ((1.0, 0.6), (-1.0, -0.6)):
        sphere = tm.creation.icosphere(subdivisions=2, radius=1.0)
        cap = sphere.slice_plane(
            plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, z_sign]), cap=False
        )
        cap.merge_vertices()
        cap.apply_translation([0.0, 0.0, z_off])
        v, f = numpy_to_warp(cap.vertices, cap.faces, device)
        meshes.append((v, f))
    return meshes[0], meshes[1]


def test_stitch_smooth_watertight(device: str):
    """
    Not a library comparison: the invariants ``stitch_smooth`` must hold, with trimesh as oracle.

    Watertight, winding-consistent, no boundary loop left, and the two inputs' vertices unmoved in
    the prefix -- that last is what separates *stitching* from *remeshing the whole thing*.
    """
    (va, fa), (vb, fb) = _hemisphere_pair(device)
    n_v0 = va.size + vb.size

    new_vertices, new_faces = od.holes.stitch_smooth(va, fa, vb, fb)
    verts_np = new_vertices.numpy()
    mesh_tm = tm.Trimesh(verts_np, new_faces.numpy().reshape(-1, 3), process=False)

    assert boundary_loop_sizes(new_faces.numpy().reshape(-1, 3)) == []
    assert mesh_tm.is_watertight
    assert mesh_tm.is_winding_consistent
    # Original vertices are the concatenated prefix, unchanged.
    assert np.allclose(verts_np[:n_v0], np.concatenate([va.numpy(), vb.numpy()]), atol=1e-6)


@pytest.mark.parity(
    "stitch_smooth",
    "meshlib",
    benchmarked=False,
    reason="stitch_smooth has no benchmark group: the band it refines is the one "
    "stitch_min_weight already times, plus fill_smooth's finisher, and a group here "
    "would re-measure both under a third name. stitchHolesNicely is nonetheless the "
    "only reference that performs the whole three-stage operation, which is why the "
    "comparison lives here.",
)
def test_stitch_smooth_statistics_vs_meshlib(device: str):
    """
    Class C (a derived scalar): the enclosed volume, because the two bands share no vertices.

    ``stitchHolesNicely`` is the same three stages in the same order -- minimum-weight band, refine
    to a target edge length, smooth the new interior into both surrounding surfaces -- and it takes
    the same knobs (``subdivideSettings.maxEdgeLen``, ``maxEdgeSplits``, ``smoothCurvature``), which
    is what makes this comparable at all. But each side subdivides on its own schedule, so no vertex
    correspondence exists and the volume is the strongest shared quantity, exactly as
    [`test_fill_smooth_statistics_vs_meshlib`] argues for the single-hole case.

    Measured on two facing hemispheres at ``max_edge=0.15``: volume within **1.9%** and area within
    **1.2%**, with 2 384 faces against MeshLib's 2 368 -- so the refinement lands within 0.7% of the
    same triangle budget from the same target length. At ``max_edge=0.3`` the volumes are 3.8%
    apart, which is the refinement schedule diverging where there are fewer splits to average over,
    and is why the finer target is the one asserted.

    What the volume excludes is a band that bulges, collapses or pinches while still being
    watertight; what it cannot see is a band that is smooth in the wrong place.

    **Mutation probe** for that claim, since a 5 % scalar bound is exactly the shape that a
    plausible-but-wrong answer can slip through. Displacing the equatorial band of a unit sphere --
    the geometry a stitched pair of hemispheres is -- and reading the same two statistics: a 2 %
    band-radius error moves the volume 1.4 % and the area 1.1 % (**passes**), 5 % moves them 3.6 %
    and 3.3 % (**passes**), and **10 % moves them 7.4 % and 7.9 %, which fails both bounds**. So the
    bound separates a band misplaced by a tenth of the radius and tolerates one misplaced by a
    twentieth -- which is the right order, because the two libraries' own refinement schedules
    already differ by the 0.7 % recorded above and by 3.8 % at the coarser target. Tightening it
    would be measuring the schedule, not the band.
    """
    (va, fa), (vb, fb) = _hemisphere_pair(device)
    max_edge = 0.15

    new_vertices, new_faces = od.holes.stitch_smooth(va, fa, vb, fb, max_edge=max_edge)
    mesh_od = warp_to_trimesh(new_vertices, new_faces)
    assert mesh_od.is_watertight

    verts_ml = np.ascontiguousarray(np.vstack([va.numpy(), vb.numpy()]), dtype=np.float32)
    orig_ml = np.ascontiguousarray(
        np.vstack([fa.numpy().reshape(-1, 3), fb.numpy().reshape(-1, 3) + va.size]), dtype=np.int32
    )
    mesh_ml = numpy_to_meshlib(verts_ml, orig_ml)
    holes_ml = mesh_ml.topology.findHoleRepresentiveEdges()
    assert len(holes_ml) == 2
    settings_ml = mm.StitchHolesNicelySettings()
    settings_ml.triangulateParams.metric = mm.getComplexStitchMetric(mesh_ml)
    settings_ml.subdivideSettings.maxEdgeLen = max_edge
    settings_ml.subdivideSettings.maxEdgeSplits = 1000
    settings_ml.smoothCurvature = True
    patch_ml = mm.stitchHolesNicely(mesh_ml, holes_ml[0], holes_ml[1], settings_ml)
    n_patch_ml = int(mn.getNumpyBitSet(patch_ml).sum())  # before pack(): the returned ids are stale
    mesh_ml.pack()  # mandatory before reading topology back
    mesh_ref = tm.Trimesh(
        mn.getNumpyVerts(mesh_ml), mn.getNumpyFaces(mesh_ml.topology), process=False
    )

    # Non-vacuity: both sides really refined the band rather than returning the plain stitch, whose
    # band is one triangle per rim edge -- an order of magnitude fewer faces than this.
    n_input_faces = fa.size // 3 + fb.size // 3
    assert n_patch_ml > 100
    assert mesh_ref.is_watertight
    assert len(mesh_ref.faces) > n_input_faces + 100
    assert len(mesh_od.faces) > n_input_faces + 100
    assert np.isclose(mesh_od.volume, mesh_ref.volume, rtol=0.05)
    assert np.isclose(mesh_od.area, mesh_ref.area, rtol=0.05)


# ---------------------------------------------------------------------------
# The loop-level engines (``stitch_loops`` / ``stitch_loops_min_weight``)
# ---------------------------------------------------------------------------


def _stitch_loops_np(
    vertices_a: np.ndarray,
    faces_a: np.ndarray,
    loop_a: np.ndarray,
    vertices_b: np.ndarray,
    faces_b: np.ndarray,
    loop_b: np.ndarray,
) -> np.ndarray:
    """
    Pure-NumPy port of the boundary zippering, used as the CPU reference for the kernels.

    Mirrors ``ordito.holes.stitch_loops``'s greedy minimal-perimeter zippering, step for step.
    Perimeters are computed in ``float32`` so the argmin tie-breaks match the Warp kernels.
    Returns the flat ``(3 * n_faces,)`` face buffer.
    """
    vertices_a = vertices_a.astype(np.float32)
    vertices_b = vertices_b.astype(np.float32)
    n, m = loop_a.size, loop_b.size
    if n < m:
        vertices_a, vertices_b = vertices_b, vertices_a
        faces_a, faces_b = faces_b, faces_a
        loop_a, loop_b = loop_b, loop_a
        n, m = m, n

    flipped_a = loop_a[::-1]
    loop_b_shifted = loop_b + len(vertices_a)
    a_pos = vertices_a[flipped_a]
    b_pos = vertices_b[loop_b]

    difference = a_pos[:, None, :] - b_pos[None, :, :]
    distances = np.sqrt((difference**2).sum(-1)).astype(np.float32)
    perimeters = distances + np.roll(distances, -1, axis=0)

    shift_a, shift_b = np.unravel_index(int(np.argmin(perimeters)), perimeters.shape)
    flipped_a = np.roll(flipped_a, -shift_a)
    loop_b_shifted = np.roll(loop_b_shifted, -shift_b)
    perimeters = np.roll(np.roll(perimeters, -shift_a, axis=0), -shift_b, axis=1)
    edge = np.argmin(perimeters, axis=1)

    if edge[-1] == edge[0]:
        trailing = int(np.argmin(np.flip(edge) == edge[0]))
        flipped_a = np.roll(flipped_a, trailing)
        edge = np.roll(edge, trailing)
        perimeters = np.roll(perimeters, trailing, axis=0)

    if not np.all(np.diff(edge) >= 0):
        edge = np.append(edge, loop_b.size)
        perimeters = np.vstack([perimeters, perimeters[0]])
        unsorted_indices = _non_increasing_indices(edge)
        stable = np.delete(np.arange(edge.size), unsorted_indices)
        next_indices = stable[np.searchsorted(stable, unsorted_indices)]
        for index, next_index in zip(unsorted_indices, next_indices, strict=True):
            edge[index] = (
                int(np.argmin(perimeters[index, edge[index - 1] : edge[next_index] + 1]))
                + edge[index - 1]
            )
        edge = edge[:-1]

    window_a = np.lib.stride_tricks.sliding_window_view(
        np.append(flipped_a, flipped_a[0]), window_shape=2
    )
    bridge_a = np.column_stack([window_a, loop_b_shifted[edge]])
    window_b = np.lib.stride_tricks.sliding_window_view(
        np.append(loop_b_shifted, loop_b_shifted[0]), window_shape=2
    )
    apex = flipped_a[np.searchsorted(edge, np.arange(loop_b.size), side="right") % flipped_a.size]
    bridge_b = np.column_stack([np.fliplr(window_b), apex])

    faces = np.vstack(
        [faces_a.reshape(-1, 3), faces_b.reshape(-1, 3) + len(vertices_a), bridge_a, bridge_b]
    )
    return faces.reshape(-1).astype(np.int32)


@pytest.mark.parametrize(
    ("n_a", "n_b", "phase", "offset"),
    [
        (16, 11, 0.3, 0.0),
        (7, 13, 0.7, 0.0),
        (24, 5, 1.1, 0.0),
        (17, 11, 0.9, 1.2),
        # Equal-length rims take the branch where the larger-loop swap does *not* fire (it is on a
        # strict inequality), so the caller's own argument order decides which mesh is A. Every
        # other case here swaps or not by size; only this one pins the tie.
        (13, 13, 0.5, 0.0),
    ],
)
def test_stitch_loops_matches_numpy(
    device: str, n_a: int, n_b: int, phase: float, offset: float
) -> None:
    bottom, top = _capsule_halves(device, n_a, n_b, phase, offset)
    va_np, fa_np, va, fa = bottom
    vb_np, fb_np, vb, fb = top

    loop_a = od.boundary.longest_boundary_loop(va, fa)
    loop_b = od.boundary.longest_boundary_loop(vb, fb)

    _, faces_wp = od.holes.stitch_loops(va, fa, loop_a, vb, fb, loop_b)
    faces_np = _stitch_loops_np(va_np, fa_np, loop_a.numpy(), vb_np, fb_np, loop_b.numpy())

    assert np.array_equal(
        lexsort_rows(np.sort(faces_wp.numpy().reshape(-1, 3), axis=1)),
        lexsort_rows(np.sort(faces_np.reshape(-1, 3), axis=1)),
    )


def test_non_increasing_indices() -> None:
    # The longest non-decreasing subsequence keeps the repeated 1s and 4s; only 5 (at index 4)
    # falls outside it, so its index is flagged for correction.
    numbers = np.array([0, 1, 1, 2, 5, 3, 4, 4, 7], dtype=np.int64)
    assert np.array_equal(_non_increasing_indices(numbers), np.array([4]))

    # A strictly sorted sequence needs no correction.
    assert _non_increasing_indices(np.arange(6, dtype=np.int64)).size == 0


@pytest.mark.parity("extend_hole", "meshlib")
@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_extend_hole_matches_meshlib(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on the counts and on where the new rim lands, against ``extendAllHoles``.

    Both add two triangles and one vertex per rim edge, so the face and vertex counts agree exactly
    -- measured 238 to 330 faces and 143 to 189 vertices on a sliced ``icosphere(2)``, identical on
    both sides. The *positions* are also checked rather than assumed: every appended vertex must lie
    **in** the plane, which is the one thing an orthogonal projection guarantees and a bevel or an
    offset would not.

    Comparing positions element-wise would need the two libraries to enumerate the rim in the same
    order, which they do not, so the geometry is compared through invariants that do not depend on
    it: the plane residual, the surviving loop count, and edge-manifoldness.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    n_vertices = vertices_wp.size
    n_faces = faces_wp.size // 3
    loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)
    rim_total = sum(loop.size for loop in loops_wp)
    assert rim_total > 0  # non-vacuity: an open fixture

    height = float(np.asarray(mesh_tm.vertices)[:, 2].max()) + 1.0
    extended_vertices_wp, extended_faces_wp = od.holes.extend_hole(
        vertices_wp, faces_wp, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, height)
    )
    assert extended_vertices_wp.size == n_vertices + rim_total
    assert extended_faces_wp.size // 3 == n_faces + 2 * rim_total
    appended_np = extended_vertices_wp.numpy()[n_vertices:]
    assert np.allclose(appended_np[:, 2], height, atol=1e-5)
    assert np.array_equal(extended_vertices_wp.numpy()[:n_vertices], vertices_wp.numpy())
    assert od.validation.is_edge_manifold(extended_faces_wp)
    assert len(od.boundary.boundary_loops(extended_vertices_wp, extended_faces_wp)) == len(loops_wp)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    mm.extendAllHoles(mesh_ml, mm.Plane3f(mm.Vector3f(0.0, 0.0, 1.0), height))
    extended_ml = meshlib_to_trimesh(mesh_ml)
    assert len(extended_ml.faces) == extended_faces_wp.size // 3
    assert len(extended_ml.vertices) == extended_vertices_wp.size


def test_extend_hole_then_fill_is_watertight(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: the composition this exists for, and the empty case.

    Extending to a plane and then filling must give a **closed solid** -- that is the reason the
    function is not itself a cap. Asserted on the result rather than on the extension, because a
    bridge with a reversed quad would still be edge-manifold and would fail here.

    The check is watertightness plus **winding consistency**, not the volume's sign: this fixture is
    inward-wound, so its solid has a negative signed volume and a ``> 0`` assert would fail on a
    perfectly correct bridge. Consistency is what the bridge is actually responsible for.

    A closed input has no rim, so the call is the identity; and an explicit empty loop list is the
    same case reached differently.
    """
    mesh_tm, mesh_wp = hemisphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    height = float(np.asarray(mesh_tm.vertices)[:, 2].max()) + 0.5
    extended_vertices_wp, extended_faces_wp = od.holes.extend_hole(
        vertices_wp, faces_wp, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, height)
    )
    capped_faces_wp = od.holes.fill_min_weight(extended_vertices_wp, extended_faces_wp)
    capped_tm = warp_to_trimesh(extended_vertices_wp, capped_faces_wp)
    assert capped_tm.is_watertight
    # Winding rather than the volume's *sign*: the fixture's own orientation decides that (this one
    # comes out inward, so a positive-volume assert would fail on a correct bridge), while
    # consistency is the property the bridge actually has to have.
    assert capped_tm.is_winding_consistent
    assert abs(capped_tm.volume) > 0.0

    same_vertices_wp, same_faces_wp = od.holes.extend_hole(
        vertices_wp, faces_wp, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, height), []
    )
    assert np.array_equal(same_faces_wp.numpy(), faces_wp.numpy())
    assert np.array_equal(same_vertices_wp.numpy(), vertices_wp.numpy())
    with pytest.raises(TypeError, match=r"expected dtype"):
        od.holes.extend_hole(
            vertices_wp,
            faces_wp,
            wp.vec3(0.0, 0.0, 1.0),
            wp.vec3(0.0, 0.0, height),
            [wp.zeros(3, dtype=wp.float32, device=faces_wp.device)],
        )


def _meshlib_hole_edge(mesh_ml: mm.Mesh, edge: tuple[int, int]) -> mm.EdgeId:
    """
    Map a ordito boundary edge ``(v0, v1)`` to the meshlib ``EdgeId`` with the hole on its left.

    ordito reports a boundary edge wound the way its own face winds it, so the *face* traverses
    ``v0 -> v1`` and the hole side is the reverse. meshlib's bridge and fill entry points all want
    the edge with **no left face**, which is therefore ``v1 -> v0``. The mapping is asserted rather
    than trusted: both ends are read back off the topology and the missing left face is checked.
    """
    edge_ml = mesh_ml.topology.findEdge(mm.VertId(int(edge[1])), mm.VertId(int(edge[0])))
    assert edge_ml.valid()
    assert mesh_ml.topology.org(edge_ml).get() == int(edge[1])
    assert mesh_ml.topology.dest(edge_ml).get() == int(edge[0])
    assert not mesh_ml.topology.left(edge_ml).valid()
    return edge_ml


@pytest.mark.parity("build_bottom", "meshlib")
@pytest.mark.parametrize("hole_extension", [0.0, 0.3])
def test_build_bottom_matches_meshlib(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], hole_extension: float
) -> None:
    """
    Class A against ``buildBottom``: the base plane's placement, and the band's counts.

    Both sides place the plane at the rim's own extreme along ``-direction``, pushed a further
    ``hole_extension``, and bridge to it with two triangles per rim edge -- so the counts agree
    exactly and every appended vertex must land at the *same* height, which is the one number the
    choice of plane decides. That height is compared against meshlib's own extreme, not against a
    recomputation, so a disagreement about which vertex is lowest would fail here.

    Element-wise position comparison is not available: the two libraries enumerate the rim in
    different orders. The band's geometry is pinned instead by the plane residual (exact by
    construction for an orthogonal projection) plus the face and vertex counts and
    edge-manifoldness.
    """
    mesh_tm, mesh_wp = hemisphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    direction = wp.vec3(0.0, 0.0, 1.0)
    loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)
    assert len(loops_wp) == 1  # non-vacuity: one rim, so both sides bottom the same hole
    rim_total = loops_wp[0].size

    bottomed_vertices_wp, bottomed_faces_wp = od.holes.build_bottom(
        vertices_wp, faces_wp, direction, hole_extension
    )
    n_vertices = vertices_wp.size
    assert bottomed_vertices_wp.size == n_vertices + rim_total
    assert bottomed_faces_wp.size // 3 == faces_wp.size // 3 + 2 * rim_total
    assert np.array_equal(bottomed_vertices_wp.numpy()[:n_vertices], vertices_wp.numpy())
    assert od.validation.is_edge_manifold(bottomed_faces_wp)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    rim_ml = mesh_ml.topology.findHoleRepresentiveEdges()
    assert rim_ml.size() == 1
    lowest_rim = float(vertices_wp.numpy()[loops_wp[0].numpy(), 2].min())
    mm.buildBottom(mesh_ml, rim_ml[0], mm.Vector3f(0.0, 0.0, 1.0), hole_extension)
    bottomed_ml = meshlib_to_trimesh(mesh_ml)
    assert len(bottomed_ml.faces) == bottomed_faces_wp.size // 3
    assert len(bottomed_ml.vertices) == bottomed_vertices_wp.size

    appended_np = bottomed_vertices_wp.numpy()[n_vertices:]
    expected_height = lowest_rim - hole_extension
    assert np.allclose(appended_np[:, 2], expected_height, atol=1e-5)
    appended_ml = np.asarray(bottomed_ml.vertices)[len(mesh_tm.vertices) :]
    assert np.allclose(appended_ml[:, 2], expected_height, atol=1e-5)


def test_build_bottom_fits_each_rim_separately(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: meshlib's ``buildBottom`` takes one hole, so it cannot show this.

    The invariant that distinguishes this function from
    [`extend_hole`][ordito.holes.extend_hole] is that each rim gets its *own* plane. ``half_torus``
    has two rims at different heights, so a shared plane would put both rings at one height and a
    per-rim plane puts them at two -- the assert is that the appended ring heights form exactly two
    groups, one per rim, each at that rim's own minimum.

    The closed case and the empty-loop-list case are the identity, and are checked here for the
    same reason ``extend_hole``'s are.
    """
    _, mesh_wp = half_torus
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    loops_wp = od.boundary.boundary_loops(vertices_wp, faces_wp)
    assert len(loops_wp) == 2  # non-vacuity: two rims is what makes the per-rim plane visible

    vertices_np = vertices_wp.numpy()
    rim_minima = sorted(float(vertices_np[loop.numpy(), 2].min()) for loop in loops_wp)
    assert rim_minima[1] - rim_minima[0] > 1e-3  # the two rims genuinely sit at different heights

    bottomed_vertices_wp, _ = od.holes.build_bottom(vertices_wp, faces_wp, wp.vec3(0.0, 0.0, 1.0))
    appended_np = bottomed_vertices_wp.numpy()[vertices_wp.size :]
    heights = sorted(np.unique(np.round(appended_np[:, 2], 5)).tolist())
    assert len(heights) == 2
    assert np.allclose(heights, rim_minima, atol=1e-5)

    same_vertices_wp, same_faces_wp = od.holes.build_bottom(
        vertices_wp, faces_wp, wp.vec3(0.0, 0.0, 1.0), 0.0, []
    )
    assert np.array_equal(same_faces_wp.numpy(), faces_wp.numpy())
    assert np.array_equal(same_vertices_wp.numpy(), vertices_wp.numpy())


def _rim_successors(vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]) -> dict[int, int]:
    """Map each rim vertex to the next one along the boundary, in face-winding direction."""
    return {
        int(u): int(v)
        for u, v in od.boundary.oriented_boundary_edges(vertices_wp, faces_wp).numpy()
    }


@pytest.mark.parity("bridge_edges", "meshlib")
@pytest.mark.parametrize("step", [6, 12, 5])
def test_bridge_edges_matches_meshlib(hemisphere: tuple[tm.Trimesh, wp.Mesh], step: int) -> None:
    """
    Class A against ``makeBridge``: the same two triangles, including which diagonal splits them.

    The comparison is on the appended face block as a canonically wound, lexicographically sorted
    row set -- both libraries add exactly two triangles over the same four existing vertices, so
    there is nothing to reindex and no tolerance involved. That it agrees on the *diagonal* is the
    substantive part: the quadrilateral admits two triangulations and only one of them matches, so
    a patch wound the other way round would fail here rather than merely look different.

    The edge pairs are taken at three different separations along one rim, because a bridge between
    nearly opposite edges and one between near neighbours take different branches in meshlib.
    """
    mesh_tm, mesh_wp = hemisphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    rim_np = od.boundary.oriented_boundary_edges(vertices_wp, faces_wp).numpy()
    assert len(rim_np) > step  # non-vacuity: the rim is long enough for this separation
    edge_a = (int(rim_np[0][0]), int(rim_np[0][1]))
    edge_b = (int(rim_np[step][0]), int(rim_np[step][1]))

    n_faces = faces_wp.size // 3
    bridged_faces_wp = od.holes.bridge_edges(vertices_wp, faces_wp, edge_a, edge_b)
    patch_np = bridged_faces_wp.numpy().reshape(-1, 3)[n_faces:]
    assert len(patch_np) == 2
    assert np.array_equal(bridged_faces_wp.numpy()[: 3 * n_faces], faces_wp.numpy())
    assert od.validation.is_edge_manifold(bridged_faces_wp)
    assert warp_to_trimesh(vertices_wp, bridged_faces_wp).is_winding_consistent
    # Bridging one rim to itself splits it in two: the topological point of the operation.
    assert len(od.boundary.boundary_loops(vertices_wp, bridged_faces_wp)) == 2

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    result_ml = mm.makeBridge(
        mesh_ml.topology, _meshlib_hole_edge(mesh_ml, edge_a), _meshlib_hole_edge(mesh_ml, edge_b)
    )
    assert result_ml.newFaces == 2  # non-vacuity: meshlib built the bridge rather than refusing
    patch_ml = np.asarray(meshlib_to_trimesh(mesh_ml).faces)[n_faces:]
    assert np.array_equal(
        lexsort_rows(canonical_winding(patch_np)), lexsort_rows(canonical_winding(patch_ml))
    )


@pytest.mark.parity("bridge_edges", "meshlib")
@pytest.mark.parametrize("side", ["successor", "predecessor"])
def test_bridge_edges_shared_vertex_is_one_triangle(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], side: str
) -> None:
    """
    Class A on meshlib's other branch: two rim-consecutive edges give **one** triangle, not two.

    Both orders are covered because meshlib reaches them differently -- one is the
    ``prev(a.sym()) == b`` branch and the other swaps the two edges first -- and a port that
    handled only one would still pass the general-case test. The rim also stays a *single* loop
    here, where a general bridge splits it in two, so that count is asserted as well.
    """
    mesh_tm, mesh_wp = hemisphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    rim_np = od.boundary.oriented_boundary_edges(vertices_wp, faces_wp).numpy()
    successors = _rim_successors(vertices_wp, faces_wp)
    edge_a = (int(rim_np[0][0]), int(rim_np[0][1]))
    if side == "successor":
        edge_b = (edge_a[1], successors[edge_a[1]])
    else:
        predecessors = {v: u for u, v in successors.items()}
        edge_b = (predecessors[edge_a[0]], edge_a[0])
    assert len(set(edge_a) | set(edge_b)) == 3  # non-vacuity: the two edges really do share one end

    n_faces = faces_wp.size // 3
    bridged_faces_wp = od.holes.bridge_edges(vertices_wp, faces_wp, edge_a, edge_b)
    patch_np = bridged_faces_wp.numpy().reshape(-1, 3)[n_faces:]
    assert len(patch_np) == 1
    assert od.validation.is_edge_manifold(bridged_faces_wp)
    assert warp_to_trimesh(vertices_wp, bridged_faces_wp).is_winding_consistent
    assert len(od.boundary.boundary_loops(vertices_wp, bridged_faces_wp)) == 1

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    result_ml = mm.makeBridge(
        mesh_ml.topology, _meshlib_hole_edge(mesh_ml, edge_a), _meshlib_hole_edge(mesh_ml, edge_b)
    )
    assert result_ml.newFaces == 1
    patch_ml = np.asarray(meshlib_to_trimesh(mesh_ml).faces)[n_faces:]
    assert np.array_equal(
        lexsort_rows(canonical_winding(patch_np)), lexsort_rows(canonical_winding(patch_ml))
    )


@pytest.mark.parametrize(
    ("shared", "edge_a", "edge_b", "expected"),
    [("a0 == b0", (0, 1), (0, 3), [1, 0, 3]), ("a1 == b1", (2, 0), (4, 0), [0, 2, 4])],
)
def test_bridge_edges_shared_endpoint_on_the_split_diagonal(
    device: str, shared: str, edge_a: tuple[int, int], edge_b: tuple[int, int], expected: list[int]
) -> None:
    """
    Not a library comparison: meshlib's ``makeBridge`` declines this input.

    It needs a manifold topology, so the collapsed quad's own traversal is the only available
    oracle -- which is exact, the answer being a single named triangle rather than a tolerance.

    The quad ``(a1, a0, b1, b0)`` is split along its ``a0``-``b0`` diagonal, so a shared endpoint
    *on that diagonal* is the one case the split cannot express: at ``a0 == b0`` both halves fold
    onto a line and the patch came out **empty** (``bridge_edges`` returned the mesh unchanged, with
    no error, despite documenting a 3-entry growth), and at ``a1 == b1`` the two halves were the
    same triangle wound opposite ways and the patch came out **doubled**. The rim-consecutive test
    above cannot reach either: it exercises the *crossed* endpoints (``a0 == b1`` / ``a1 == b0``),
    where the plain degeneracy filter is already correct.

    Reaching the aligned case needs two boundary loops sharing a vertex, which a single simple rim
    cannot have -- hence the explicit bowtie rather than a fixture. What the invariants exclude is
    the empty and the doubled patch; they say nothing about the general two-triangle case, which
    the meshlib comparisons above cover.
    """
    vertices_np = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=np.float64
    )
    faces_np = np.array([0, 1, 2, 0, 3, 4], dtype=np.int32)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    # Non-vacuity: the two edges really are boundary edges that share exactly one endpoint.
    rim_np = od.boundary.oriented_boundary_edges(vertices_wp, faces_wp).numpy()
    rim_rows = {(int(u), int(v)) for u, v in rim_np}
    assert edge_a in rim_rows
    assert edge_b in rim_rows
    assert len(set(edge_a) | set(edge_b)) == 3

    bridged_faces_wp = od.holes.bridge_edges(vertices_wp, faces_wp, edge_a, edge_b)
    patch_np = bridged_faces_wp.numpy().reshape(-1, 3)[2:]
    assert len(patch_np) == 1, f"{shared} must give one triangle, got {len(patch_np)}"
    assert np.array_equal(patch_np[0], expected)
    # The documented growth is 3 entries for the shared-vertex case; an empty patch would leave the
    # buffer at its input length and a doubled one would add 6.
    assert bridged_faces_wp.size == faces_wp.size + 3


def test_bridge_edges_rim_check_agrees_with_the_rim_table(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: the validation's one scan of the faces is the rim table's membership.

    ``bridge_edges`` decides both halves of ``validate`` from per-pair edge counts over the faces
    rather than from an edge table, so this pins the rim half to
    [`oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges], which carries the
    definition: every row of that table is accepted as ``edge_a``, while the same edge reversed and
    an interior edge are refused. The chord-duplicate half must fire as well -- two consecutive rim
    edges joined across their shared rim vertex's neighbour -- since that is the failure
    ``validate`` exists for and ``join_closest_components`` relies on it on every join.

    Not a library comparison either: meshlib returns a falsy result where this raises. The same
    edge twice is refused too, and ``validate=False`` is shown to skip every one of these checks,
    which is what the switch is for.
    """
    _, mesh_wp = hemisphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    rim_np = od.boundary.oriented_boundary_edges(vertices_wp, faces_wp).numpy()
    rim = {(int(u), int(v)) for u, v in rim_np}
    assert len(rim) >= 16
    edge_far = (int(rim_np[len(rim_np) // 2][0]), int(rim_np[len(rim_np) // 2][1]))
    accepted = 0
    for u, v in rim_np:
        edge = (int(u), int(v))
        if edge == edge_far:
            continue
        # A pair near ``edge_far`` may be refused for its chord, never for being off the rim.
        try:
            od.holes.bridge_edges(vertices_wp, faces_wp, edge, edge_far)
        except ValueError as error:
            message = str(error)
        else:
            message = ""
            accepted += 1
        assert message == "" or "non-manifold" in message
    assert accepted > len(rim) // 2
    edge_a = (int(rim_np[0][0]), int(rim_np[0][1]))
    with pytest.raises(ValueError, match=r"edge_a=.*is not a boundary edge"):
        od.holes.bridge_edges(vertices_wp, faces_wp, (edge_a[1], edge_a[0]), edge_far)
    faces_np = faces_wp.numpy().reshape(-1, 3)
    interior = next(
        (int(a), int(b))
        for a, b in faces_np[:, :2]
        if (int(a), int(b)) not in rim and (int(b), int(a)) not in rim
    )
    with pytest.raises(ValueError, match=r"edge_b=.*is not a boundary edge"):
        od.holes.bridge_edges(vertices_wp, faces_wp, edge_a, interior)
    with pytest.raises(ValueError, match="different edges"):
        od.holes.bridge_edges(vertices_wp, faces_wp, edge_a, edge_a)

    # Two rim edges one apart share no vertex, but the vertex between them already joins both ends,
    # so the patch's own side would be a second edge there.
    successors = _rim_successors(vertices_wp, faces_wp)
    middle = successors[edge_a[1]]
    edge_next = (middle, successors[middle])
    with pytest.raises(ValueError, match="non-manifold"):
        od.holes.bridge_edges(vertices_wp, faces_wp, edge_a, edge_next)

    assert (
        od.holes.bridge_edges(vertices_wp, faces_wp, interior, edge_far, False).size
        == faces_wp.size + 6
    )


def test_bridge_edges_smooth_opposed_edges_stay_finite(device: str) -> None:
    """
    Not a library comparison: meshlib fairs a different curve, so it has no opinion on this input.

    The strip's width direction is interpolated from ``edge_a``'s to ``edge_b``'s and renormalized.
    When the two edges are wound so those directions are **anti-parallel** the blend passes through
    the zero vector at the halfway sample, and the strip has to make a half turn whose axis is
    arbitrary -- a jump discontinuity, not a value. The two coplanar triangles below are that
    configuration: both edges run along ``+x``, ``edge_a``'s face is below its edge and ``edge_b``'s
    above, so ``pa0 - pa1`` and ``pb1 - pb0`` are exactly opposed.

    The documented behaviour is a twisted strip, not an error -- the function's own note already
    says a strip across nearly opposed edges can fold -- so what is pinned here is that every
    position it returns is finite and that the strip is the full one, not a collapsed fan.

    **Bug class excluded:** a division by the zero-length blend. **Mutation probe, measured against
    a detached pre-fix worktree:** the unguarded normalization returns **2 NaN rows of 10**, with
    nothing raised and only a NumPy ``RuntimeWarning`` on stderr.
    """
    vertices_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.0, 3.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.5, 4.0, 0.0],
        ]
    )
    faces_np = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    strip_vertices_wp, strip_faces_wp = od.holes.bridge_edges_smooth(
        vertices_wp, faces_wp, (0, 1), (3, 4), 0.5
    )

    positions_np = strip_vertices_wp.numpy()
    assert len(positions_np) > vertices_np.shape[0]  # non-vacuity: it really did subdivide
    assert np.isfinite(positions_np).all()
    # A full strip, not the collapsed one-segment fan: two triangles per segment.
    n_new_faces = (strip_faces_wp.size - faces_wp.size) // 3
    assert n_new_faces > 2
    _normals_wp, areas_wp = od.triangles.face_normals_and_areas(strip_vertices_wp, strip_faces_wp)
    assert np.isfinite(areas_wp.numpy()).all()


@pytest.mark.parity("bridge_edges_smooth", "meshlib")
def test_bridge_edges_smooth_matches_meshlib(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C against ``makeSmoothBridge``: the two strips as surfaces, at two-sided Hausdorff.

    No correspondence exists to compare element-wise, and the two curves are not the same
    construction: ordito spans the gap with one cubic Hermite segment while meshlib fairs a
    resampled control polyline with an iterative stabilized solver, so the strips carry different
    vertex counts (measured 16 against 18 at ``sampling_step=0.25``) and no shared
    parameterization. What *is* comparable is where the strip goes, so both are extracted as
    surfaces and compared with
    [`hausdorff_surface_two_sided`][tests.comparisons.hausdorff_surface_two_sided].

    The bug class this excludes is a strip that spans the gap along the **chord** rather than
    leaving both surfaces tangentially -- the whole reason the function exists over
    [`bridge_edges`][ordito.holes.bridge_edges]. Mutation probe: the flat two-triangle patch over
    the same pair measures **0.4942** against the smooth strip's **0.0755** agreement with meshlib,
    a margin of **6.5x**. The bound is set at 3x the measured agreement, which the flat patch
    misses by 2.2x -- so a chord-spanning implementation fails here rather than passing loosely.

    Measured across three edge separations and two sampling steps, the agreement ranges 0.059 to
    0.228 and the flat-patch margin 2.8x to 8.4x; the point pinned here is the finest of them.
    """
    mesh_tm, mesh_wp = hemisphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    rim_np = od.boundary.oriented_boundary_edges(vertices_wp, faces_wp).numpy()
    edge_a = (int(rim_np[0][0]), int(rim_np[0][1]))
    edge_b = (int(rim_np[12][0]), int(rim_np[12][1]))
    sampling_step = 0.25

    n_vertices, n_faces = vertices_wp.size, faces_wp.size // 3
    strip_vertices_wp, strip_faces_wp = od.holes.bridge_edges_smooth(
        vertices_wp, faces_wp, edge_a, edge_b, sampling_step
    )
    assert strip_vertices_wp.size > n_vertices  # non-vacuity: it really did subdivide
    assert od.validation.is_edge_manifold(strip_faces_wp)
    assert warp_to_trimesh(strip_vertices_wp, strip_faces_wp).is_winding_consistent
    assert len(od.boundary.boundary_loops(strip_vertices_wp, strip_faces_wp)) == 2

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    result_ml = mm.makeSmoothBridge(
        mesh_ml,
        _meshlib_hole_edge(mesh_ml, edge_a),
        _meshlib_hole_edge(mesh_ml, edge_b),
        sampling_step,
    )
    assert result_ml.newFaces > 2  # non-vacuity: meshlib subdivided rather than emitting one quad
    smooth_ml = meshlib_to_trimesh(mesh_ml)

    strip_tm = warp_to_trimesh(strip_vertices_wp, strip_faces_wp).submesh(
        [np.arange(n_faces, strip_faces_wp.size // 3)], append=True
    )
    strip_ml = smooth_ml.submesh([np.arange(n_faces, len(smooth_ml.faces))], append=True)
    assert isinstance(strip_tm, tm.Trimesh)
    assert isinstance(strip_ml, tm.Trimesh)
    agreement = hausdorff_surface_two_sided(
        np.asarray(strip_tm.vertices),
        np.asarray(strip_tm.faces),
        np.asarray(strip_ml.vertices),
        np.asarray(strip_ml.faces),
    )
    assert agreement < 3.0 * 0.0755

    flat_faces_wp = od.holes.bridge_edges(vertices_wp, faces_wp, edge_a, edge_b)
    flat_tm = warp_to_trimesh(vertices_wp, flat_faces_wp).submesh(
        [np.arange(n_faces, flat_faces_wp.size // 3)], append=True
    )
    assert isinstance(flat_tm, tm.Trimesh)
    # The probe that makes the bound above mean something: a chord-spanning patch is 3x further.
    assert (
        hausdorff_surface_two_sided(
            np.asarray(flat_tm.vertices),
            np.asarray(flat_tm.faces),
            np.asarray(strip_ml.vertices),
            np.asarray(strip_ml.faces),
        )
        > 3.0 * agreement
    )


# ---------------------------------------------------------------------------
# join_closest_components
# ---------------------------------------------------------------------------


def _open_shells_tm(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], count: int, gap: float = 3.0
) -> tm.Trimesh:
    """``count`` copies of the hemisphere fixture in a row, each ``gap`` apart along ``x``."""
    mesh_tm, _mesh_wp = hemisphere
    shells_tm = []
    for index in range(count):
        shell_tm = mesh_tm.copy()
        shell_tm.apply_translation([gap * index, 0.0, 0.0])
        shells_tm.append(shell_tm)
    return cast("tm.Trimesh", tm.util.concatenate(shells_tm))


@pytest.mark.parity("join_closest_components", "pymeshfix")
@pytest.mark.parametrize("count", [2, 3, 5])
def test_join_closest_components_matches_pymeshfix(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], device: str, count: int
) -> None:
    """
    Class B: the same joins, compared through the counts because the chosen seam is not shared.

    ``join_closest_components`` picks the globally closest pair of boundary *vertices* and bridges
    an oriented edge at each; MeshFix picks a vertex pair too but bridges its own choice of incident
    edge, so the two patches need not be the same two triangles even when the same shells were
    joined. What is comparable, and asserted: the face count, the boundary loop count, the component
    count, and that no vertex was added. Measured on three shells, both sides go from
    291 v / 504 f / 3 loops to **291 v / 508 f / 1 loop / 1 component**.

    The transform is the counts, and the identity ``2 * (count - 1)`` faces added is what makes them
    an oracle for *how many* joins happened rather than merely that something did: each bridge is
    one quadrilateral split along a diagonal, so a filler that fanned a whole rim or added a vertex
    would fail on the count before anything else.

    The merged rim is left **open** on purpose -- one loop out, not zero -- because closing it is
    [`fill_min_weight`][ordito.holes.fill_min_weight]'s job, and asserting that is what pins the
    division of labour between the two.
    """
    mesh_tm = _open_shells_tm(hemisphere, count)
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    n_faces = faces_wp.size // 3
    assert len(od.boundary.boundary_loops(vertices_wp, faces_wp)) == count

    joined_wp = od.holes.join_closest_components(vertices_wp, faces_wp)
    joined_tm = warp_to_trimesh(vertices_wp, joined_wp)

    tin_pmf = trimesh_to_pymeshfix(mesh_tm)
    assert tin_pmf.n_boundaries == count  # the loader left the shells alone
    tin_pmf.join_closest_components()
    vertices_pmf, faces_pmf = pymeshfix_to_numpy(tin_pmf)
    joined_pmf = tm.Trimesh(vertices_pmf, faces_pmf, process=False)

    assert faces_pmf.shape[0] == n_faces + 2 * (count - 1)  # the reference really joined
    assert tin_pmf.n_boundaries == 1
    assert joined_wp.size // 3 == faces_pmf.shape[0]
    assert vertices_pmf.shape[0] == mesh_tm.vertices.shape[0]
    assert len(od.boundary.boundary_loops(vertices_wp, joined_wp)) == 1
    assert len(joined_tm.split(only_watertight=False)) == 1
    assert len(joined_pmf.split(only_watertight=False)) == 1
    assert od.validation.is_edge_manifold(joined_wp)
    # The prefix is the input face buffer: bridge triangles are appended, never interleaved.
    assert np.array_equal(joined_wp.numpy()[: faces_wp.size], faces_wp.numpy())


@pytest.mark.parametrize("count", [3, 6])
def test_join_closest_components_rim_tracks_the_regrouped_boundary(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], device: str, count: int
) -> None:
    """
    Ordito against ordito: the rim edited join by join is the rim a regroup would find.

    ``join_closest_components`` no longer regroups the face buffer between joins; it edits its
    boundary table and component labels as each patch dictates. The pymeshfix comparison above
    carries the oracle for the joins themselves. This pins the edit: after every join, the table
    (rows and their order, which is the pairing's tie-break) equals
    ``oriented_boundary_edges`` of the faces so far, and the labels partition the rows as
    ``face_connected_component_labels`` does. The shells are a regular row, so the distances
    carry exact ties and the row order is live.
    """
    mesh_tm = _open_shells_tm(hemisphere, count, gap=2.2)
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    rim = od.holes._JoinRim.of(vertices_wp, faces_wp)  # pyright: ignore[reportPrivateUsage]
    assert rim is not None
    faces_np = faces_wp.numpy()
    max_distance_sq = wp.float32(float("inf"))
    for _ in range(count - 1):
        pair = rim.closest_cross_component_edges(max_distance_sq)
        assert pair is not None
        triangles = od.holes._bridge_triangles(*pair)  # pyright: ignore[reportPrivateUsage]
        rim.bridge(triangles)
        faces_np = np.concatenate([faces_np, np.asarray(triangles, dtype=np.int32).reshape(-1)])
        current_wp = wp.array(faces_np, dtype=wp.int32, device=device)
        rows_np = od.boundary.oriented_boundary_edges(vertices_wp, current_wp).numpy()
        assert np.array_equal(rim.rows(), rows_np)
        face_labels_np = od.adjacency.face_connected_component_labels(current_wp).numpy()
        tail_faces_np = np.array(
            [
                np.flatnonzero((faces_np.reshape(-1, 3) == tail).any(axis=1))[0]
                for tail in rows_np[:, 0]
            ]
        )
        expected_np = face_labels_np[tail_faces_np]
        labels_np = rim.row_components()
        assert same_partition(labels_np, expected_np)
    assert rim.closest_cross_component_edges(max_distance_sq) is None


def test_join_closest_components_joins_the_nearest_pair(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Not a library comparison: *which* shells get joined, which the count identity cannot see.

    Three shells in a row are an ambiguous test of the pairing -- every greedy order ends with one
    component -- so this places the third shell far off the line, so that the only two-join sequence
    a nearest-link rule can produce is (near pair first, distant shell second). With ``max_joins=1``
    the answer is visible directly: exactly the two close shells become one component and the far
    one is still on its own.
    """
    mesh_tm, _mesh_wp = hemisphere
    near_tm = mesh_tm.copy()
    near_tm.apply_translation([2.5, 0.0, 0.0])
    far_tm = mesh_tm.copy()
    far_tm.apply_translation([0.0, 40.0, 0.0])
    combined_tm = tm.util.concatenate([mesh_tm, near_tm, far_tm])
    assert isinstance(combined_tm, tm.Trimesh)
    vertices_wp, faces_wp = numpy_to_warp(combined_tm.vertices, combined_tm.faces, device)
    n_faces = faces_wp.size // 3

    once_wp = od.holes.join_closest_components(vertices_wp, faces_wp, max_joins=1)
    components = warp_to_trimesh(vertices_wp, once_wp).split(only_watertight=False)

    assert once_wp.size // 3 == n_faces + 2
    assert len(components) == 2
    # The joined component is the two near shells; the untouched one is the far shell alone.
    sizes = sorted(len(component.faces) for component in components)
    assert sizes == [n_faces // 3, 2 * (n_faces // 3) + 2]
    assert max(component.vertices[:, 1].max() for component in components) > 39.0


class _JoinBounds(TypedDict, total=False):
    max_distance: float
    max_joins: int


@pytest.mark.parametrize(
    ("kwargs", "expected_joins"),
    [
        ({"max_distance": 0.1}, 0),
        ({"max_distance": 10.0}, 2),
        ({"max_joins": 1}, 1),
        ({"max_joins": 0}, 0),
    ],
)
def test_join_closest_components_respects_its_bounds(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], device: str, kwargs: _JoinBounds, expected_joins: int
) -> None:
    """
    Not a library comparison: ``max_distance`` and ``max_joins`` have no pymeshfix counterpart.

    MeshFix joins unconditionally until the mesh is connected, which on a mesh holding genuinely
    separate objects welds them; both bounds are ordito's addition, and their defaults reproduce
    the reference (see [`test_join_closest_components_matches_pymeshfix`]). The shells here are 3.0
    apart, so a 0.1 bound admits nothing and a 10.0 bound admits everything -- straddling the gap
    rather than testing one side of it.
    """
    mesh_tm = _open_shells_tm(hemisphere, 3)
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    n_faces = faces_wp.size // 3

    joined_wp = od.holes.join_closest_components(vertices_wp, faces_wp, **kwargs)

    assert joined_wp.size // 3 == n_faces + 2 * expected_joins


def test_join_closest_components_leaves_closed_and_single_meshes_alone(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], hemisphere: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Not a library comparison: the three inputs with nothing to join, each for a different reason.

    Two *closed* shells have no boundary to bridge to, one open shell has no second component, and
    a closed mesh has neither. All three come back as a copy of the input rather than raising or
    welding something -- which is what makes the function safe to put at a fixed point in a
    pipeline.
    """
    mesh_tm, _mesh_wp = icosahedron
    second_tm = mesh_tm.copy()
    second_tm.apply_translation([5.0, 0.0, 0.0])
    for source_tm in (tm.util.concatenate([mesh_tm, second_tm]), _open_shells_tm(hemisphere, 1)):
        assert isinstance(source_tm, tm.Trimesh)
        vertices_wp, faces_wp = numpy_to_warp(source_tm.vertices, source_tm.faces, device)
        joined_wp = od.holes.join_closest_components(vertices_wp, faces_wp)
        assert np.array_equal(joined_wp.numpy(), faces_wp.numpy())
        assert joined_wp.ptr != faces_wp.ptr


_INVALID_ARGUMENT_CASES = [
    "fill_smooth_edge_weights",
    "stitch_watertight",
    "stitch_two_rims",
    "stitch_min_weight_watertight",
    "stitch_min_weight_metric",
    "stitch_smooth_watertight",
    "stitch_loops_two_vertex_loop",
    "join_closest_components_negative_max_joins",
]


@pytest.mark.parametrize("case", _INVALID_ARGUMENT_CASES)
def test_invalid_arguments_raise(request: pytest.FixtureRequest, device: str, case: str) -> None:
    """
    Not a library comparison: the documented ``ValueError`` of each guard, by its message.

    The stitchers need exactly one boundary loop per side, so a watertight mesh (no loop) and
    ``half_torus`` (two) are both refused; the loop-level engine needs loops of at least three
    vertices; ``fill_smooth`` names its ``edge_weights`` options and ``stitch_min_weight`` its
    metrics; and ``join_closest_components`` refuses a negative join budget.
    """
    if case == "fill_smooth_edge_weights":
        _, mesh_wp = request.getfixturevalue("hemisphere")
        with pytest.raises(ValueError, match="edge_weights must be"):
            od.holes.fill_smooth(mesh_wp.points, mesh_wp.indices, edge_weights="nope")
    elif case in {"stitch_watertight", "stitch_two_rims"}:
        _, mesh_wp = request.getfixturevalue(
            "icosahedron" if case == "stitch_watertight" else "half_torus"
        )
        if case == "stitch_two_rims":
            assert len(boundary_loop_sizes(mesh_wp.indices.numpy().reshape(-1, 3))) >= 2
        _, _, va, fa = _cone_wp(device=device, n=10, apex_z=-1.0, rim_z=0.0)
        with pytest.raises(ValueError, match="exactly one boundary loop"):
            od.holes.stitch(mesh_wp.points, mesh_wp.indices, va, fa)
    elif case == "stitch_min_weight_watertight":
        _, mesh_wp = request.getfixturevalue("icosahedron")
        _, _, vb, fb = _cone_wp(device=device, n=8, apex_z=1.0, rim_z=0.5)
        with pytest.raises(ValueError, match="exactly one boundary loop"):
            od.holes.stitch_min_weight(mesh_wp.points, mesh_wp.indices, vb, fb)
    elif case == "stitch_min_weight_metric":
        (_, _, va, fa), (_, _, vb, fb) = _capsule_halves(device, 8, 8)
        with pytest.raises(ValueError, match="metric must be one of"):
            od.holes.stitch_min_weight(va, fa, vb, fb, metric="bogus")
    elif case == "stitch_smooth_watertight":
        _, mesh_wp = request.getfixturevalue("icosahedron")
        with pytest.raises(ValueError, match="exactly one boundary loop"):
            od.holes.stitch_smooth(mesh_wp.points, mesh_wp.indices, mesh_wp.points, mesh_wp.indices)
    elif case == "stitch_loops_two_vertex_loop":
        _, _, va, fa = _cone_wp(device=device, n=8, apex_z=-1.0, rim_z=0.0)
        _, _, vb, fb = _cone_wp(device=device, n=8, apex_z=1.0, rim_z=0.5)
        loop_a = od.boundary.longest_boundary_loop(va, fa)
        tiny_loop = wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=device)
        with pytest.raises(ValueError, match="at least 3 vertices"):
            od.holes.stitch_loops(va, fa, loop_a, vb, fb, tiny_loop)
    else:
        mesh_tm = _open_shells_tm(request.getfixturevalue("hemisphere"), 2)
        vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
        with pytest.raises(ValueError, match="max_joins"):
            od.holes.join_closest_components(vertices_wp, faces_wp, max_joins=-1)
