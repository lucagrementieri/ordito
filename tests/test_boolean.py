"""Regression tests for ``ordito.boolean``."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol

import manifold3d as m3
import numpy as np
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.boolean as od_boolean
from tests.comparisons import assert_same_as_manifold3d, assert_same_cut_faces
from tests.conftest import CLOSED_MESHES
from tests.conversions import (
    float32_trimesh,
    meshlib_to_trimesh,
    numpy_to_warp,
    trimesh_to_manifold3d,
    trimesh_to_meshlib,
    trimesh_to_pymeshlab,
    warp_empty,
)

MeshPair = tuple[tm.Trimesh, wp.Mesh]
WarpMesh = tuple[wp.array[wp.vec3], wp.array[wp.int32]]


class Operation(Protocol):
    """``union``, ``intersection`` or ``difference``."""

    def __call__(
        self,
        vertices_a: wp.array[wp.vec3],
        faces_a: wp.array[wp.int32],
        vertices_b: wp.array[wp.vec3],
        faces_b: wp.array[wp.int32],
        *,
        collapse_collinear: bool = True,
    ) -> WarpMesh:
        """Return the Boolean of the two meshes."""
        ...


OPERATIONS: list[tuple[str, Operation]] = [
    ("union", od.boolean.union),
    ("intersection", od.boolean.intersection),
    ("difference", od.boolean.difference),
]
_OPERATION_IDS = [name for name, _ in OPERATIONS]

# The closed fixtures the generic comparisons sweep: genus 0 and genus 1, both with needles.
_GENERIC_MESHES = [*CLOSED_MESHES, "torus_irregular"]


def _moved_copy(mesh_tm: tm.Trimesh, seed: int) -> tm.Trimesh:
    """Return ``mesh_tm`` rotated about its centroid and shifted by a third of its extent."""
    rng = np.random.default_rng(seed)
    moved = mesh_tm.copy()
    centre = mesh_tm.vertices.mean(axis=0)
    moved.apply_translation(-centre)
    moved.apply_transform(tm.transformations.random_rotation_matrix(rng.random(3)))
    moved.apply_translation(centre + mesh_tm.extents * np.array([0.31, 0.17, 0.11]))
    return float32_trimesh(moved)


def _box(lower: list[float], upper: list[float]) -> tm.Trimesh:
    return tm.creation.box(bounds=[lower, upper])


# Configurations where the operands touch, overlap in a plane or coincide: every tie Manifold's
# symbolic perturbation breaks, and the cleanup it then runs.
_DEGENERATE_CASES = {
    "identical": (_box([0, 0, 0], [1, 1, 1]), _box([0, 0, 0], [1, 1, 1])),
    "face_contact": (_box([0, 0, 0], [1, 1, 1]), _box([1, 0, 0], [2, 1, 1])),
    "corner_contact": (_box([0, 0, 0], [1, 1, 1]), _box([1, 1, 1], [2, 2, 2])),
    "coplanar_overlap": (_box([0, 0, 0], [1, 1, 1]), _box([0.5, 0, 0], [1.5, 1, 1])),
    "partial_face_contact": (
        _box([-0.25, -0.5, -0.5], [0.25, 0.5, 0.5]),
        _box([0.25, 0.25, -0.75], [0.75, 1.75, -0.25]),
    ),
    "nested": (_box([0, 0, 0], [1, 1, 1]), _box([0.2, 0.2, 0.2], [0.8, 0.8, 0.8])),
    "disjoint": (_box([0, 0, 0], [1, 1, 1]), _box([2, 2, 2], [3, 3, 3])),
}


def _m3_operation(name: str, a_m3: m3.Manifold, b_m3: m3.Manifold) -> m3.Manifold:
    if name == "union":
        return a_m3 + b_m3
    if name == "intersection":
        return a_m3 ^ b_m3
    return a_m3 - b_m3


@pytest.mark.parity("boolean_union", "manifold3d")
@pytest.mark.parity("boolean_intersection", "manifold3d")
@pytest.mark.parity("boolean_difference", "manifold3d")
@pytest.mark.parametrize("mesh_name", _GENERIC_MESHES)
@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=_OPERATION_IDS)
def test_boolean_matches_manifold3d(
    request: pytest.FixtureRequest, mesh_name: str, name: str, operation: Operation
) -> None:
    """
    The Boolean of a mesh and a rotated, shifted copy is Manifold's, array for array.

    Class A, the strictest form: the same vertices in the same order, the same triangles in the
    same order and rotation, as ``manifold3d``'s ``to_mesh()``. The reference is non-vacuous: its
    volume is neither operand's, so the crossing curve is really there.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    a_tm = float32_trimesh(mesh_tm)
    b_tm = _moved_copy(a_tm, seed=7)
    result_m3 = _m3_operation(name, trimesh_to_manifold3d(a_tm), trimesh_to_manifold3d(b_tm))
    assert result_m3.num_tri() > 0
    assert not np.isclose(result_m3.volume(), a_tm.volume, rtol=1e-3)
    device = mesh_wp.device
    assert_same_as_manifold3d(
        operation(
            *numpy_to_warp(a_tm.vertices, a_tm.faces, device),
            *numpy_to_warp(b_tm.vertices, b_tm.faces, device),
        ),
        result_m3,
    )


@pytest.mark.parity("boolean_union", "manifold3d")
@pytest.mark.parity("boolean_intersection", "manifold3d")
@pytest.mark.parity("boolean_difference", "manifold3d")
@pytest.mark.parametrize("case", list(_DEGENERATE_CASES))
@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=_OPERATION_IDS)
def test_boolean_degenerate_contacts_match_manifold3d(
    device: str, case: str, name: str, operation: Operation
) -> None:
    """
    Touching, coplanar, coincident, nested and disjoint solids give Manifold's mesh exactly.

    Class A, array for array, the empty result included. These are the ties the symbolic
    perturbation decides (``partial_face_contact``'s turn on the last bit of the vertex normals'
    angle weights), the slivers and duplicate edges the topology cleanup removes, and the union of
    apart meshes that Manifold composes rather than intersects (``disjoint``; ``corner_contact``
    touches, so it is intersected).
    """
    a_tm, b_tm = _DEGENERATE_CASES[case]
    result_m3 = _m3_operation(name, trimesh_to_manifold3d(a_tm), trimesh_to_manifold3d(b_tm))
    result_wp = operation(
        *numpy_to_warp(a_tm.vertices, a_tm.faces, device),
        *numpy_to_warp(b_tm.vertices, b_tm.faces, device),
    )
    assert_same_as_manifold3d(result_wp, result_m3)


@pytest.mark.parity("boolean_union", "trimesh")
@pytest.mark.parity("boolean_intersection", "trimesh")
@pytest.mark.parity("boolean_difference", "trimesh")
@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=_OPERATION_IDS)
def test_boolean_matches_trimesh(device: str, name: str, operation: Operation) -> None:
    """
    The Boolean equals ``trimesh.boolean``'s with its default engine (Manifold).

    Class A, up to vertex numbering, face order and rotation (trimesh merges and reorders the
    engine's output).
    """
    a_tm = float32_trimesh(tm.creation.icosphere(subdivisions=3))
    b_tm = _moved_copy(a_tm, seed=11)
    reference_tm = getattr(tm.boolean, name)([a_tm, b_tm], engine="manifold")
    assert len(reference_tm.faces) > 0
    vertices_wp, faces_wp = operation(
        *numpy_to_warp(a_tm.vertices, a_tm.faces, device),
        *numpy_to_warp(b_tm.vertices, b_tm.faces, device),
    )
    reference_wp = numpy_to_warp(reference_tm.vertices, reference_tm.faces, device)
    assert od.equality.meshes_allclose(vertices_wp, faces_wp, *reference_wp, atol=1e-6)


# Operands with flat faces crossed along straight lines, where Manifold's collinear collapse
# removes vertices the exact-arithmetic Booleans keep, beside the curved generic fixtures.
_FLAT_SHAPES = {
    "cylinder": lambda: tm.creation.cylinder(radius=0.5, height=2.0, sections=32),
    "box": lambda: tm.creation.box(),
}
_EXACT_SHAPES = [*_GENERIC_MESHES, *_FLAT_SHAPES]


@pytest.mark.parity("boolean_union", "pymeshlab")
@pytest.mark.parity("boolean_intersection", "pymeshlab")
@pytest.mark.parity("boolean_difference", "pymeshlab")
@pytest.mark.parametrize("shape", _EXACT_SHAPES)
@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=_OPERATION_IDS)
def test_boolean_matches_pymeshlab(
    request: pytest.FixtureRequest, device: str, shape: str, name: str, operation: Operation
) -> None:
    """
    Without the collinear collapse, the Boolean is pymeshlab's (libigl's CGAL Boolean).

    Class B, through [`assert_same_cut_faces`][tests.comparisons.assert_same_cut_faces]: the same
    vertices (within ``1e-6``), the same triangle count, and on each operand plane the same
    vertices and vector area, so every polygon of the arrangement is cut, kept and wound alike. The
    transform forgets the triangulation of each polygon, each library's own choice: Manifold
    ear-clips, libigl's CGAL Boolean and MeshLib split each face their own ways, and those two
    disagree with each other as well.

    ``collapse_collinear=False`` is the variant that reaches it: Manifold's cleanup collapses
    crossing vertices on straight lines through flat regions, which CGAL keeps (114 of the
    cylinder's 402 vertices; the curved fixtures have none and match at either setting,
    ``test_collapse_collinear_only_changes_flat_crossings``). Operands in general position only:
    where surfaces touch or overlap in a plane each library breaks the tie its own way, and
    pymeshlab raises on an empty result.
    """
    a_tm, b_tm = _exact_operands(request, shape)
    meshset_pml = trimesh_to_pymeshlab(a_tm)
    # A mesh read off a temporary ``MeshSet`` dangles once that set is collected: keep it bound.
    second_pml = trimesh_to_pymeshlab(b_tm)
    meshset_pml.add_mesh(second_pml.current_mesh())
    getattr(meshset_pml, f"generate_boolean_{name}")(first_mesh=0, second_mesh=1)
    result_pml = meshset_pml.current_mesh()
    _assert_same_exact_boolean(
        device, operation, a_tm, b_tm, result_pml.vertex_matrix(), result_pml.face_matrix()
    )


@pytest.mark.parity("boolean_union", "meshlib")
@pytest.mark.parity("boolean_intersection", "meshlib")
@pytest.mark.parity("boolean_difference", "meshlib")
@pytest.mark.parametrize("shape", _EXACT_SHAPES)
@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=_OPERATION_IDS)
def test_boolean_matches_meshlib(
    request: pytest.FixtureRequest, device: str, shape: str, name: str, operation: Operation
) -> None:
    """
    Without the collinear collapse, the Boolean is MeshLib's ``boolean``.

    Class B, the transform of ``test_boolean_matches_pymeshlab``, whose answer MeshLib's equals
    on these operands but for its own triangulation. Where operands touch or overlap in a plane,
    MeshLib keeps zero-volume sheets and duplicate vertices that Manifold's cleanup removes.
    """
    a_tm, b_tm = _exact_operands(request, shape)
    operation_ml = {
        "union": mm.BooleanOperation.Union,
        "intersection": mm.BooleanOperation.Intersection,
        "difference": mm.BooleanOperation.DifferenceAB,
    }[name]
    result_ml = mm.boolean(trimesh_to_meshlib(a_tm), trimesh_to_meshlib(b_tm), operation_ml)
    assert result_ml.valid()
    mesh_ml = meshlib_to_trimesh(result_ml.mesh)
    _assert_same_exact_boolean(device, operation, a_tm, b_tm, mesh_ml.vertices, mesh_ml.faces)


def _exact_operands(request: pytest.FixtureRequest, shape: str) -> tuple[tm.Trimesh, tm.Trimesh]:
    """Return a fixture's mesh, or a flat shape, and its rotated, shifted copy, in ``float32``."""
    if shape in _FLAT_SHAPES:
        a_tm = float32_trimesh(_FLAT_SHAPES[shape]())
    else:
        mesh_tm, _ = request.getfixturevalue(shape)
        a_tm = float32_trimesh(mesh_tm)
    return a_tm, _moved_copy(a_tm, seed=7)


def _assert_same_exact_boolean(
    device: str,
    operation: Operation,
    a_tm: tm.Trimesh,
    b_tm: tm.Trimesh,
    vertices_ref: np.ndarray,
    faces_ref: np.ndarray,
) -> None:
    """Assert ``operation(..., collapse_collinear=False)`` is the reference up to triangulation."""
    reference_tm = tm.Trimesh(vertices_ref, faces_ref, process=False)
    assert len(reference_tm.faces) > 0
    assert not np.isclose(reference_tm.volume, a_tm.volume, rtol=1e-3)
    result_wp = operation(
        *numpy_to_warp(a_tm.vertices, a_tm.faces, device),
        *numpy_to_warp(b_tm.vertices, b_tm.faces, device),
        collapse_collinear=False,
    )
    assert_same_cut_faces(result_wp, reference_tm.vertices, reference_tm.faces, [a_tm, b_tm])


@pytest.mark.parametrize(
    "operation", [pytest.param(operation, id=name) for name, operation in OPERATIONS]
)
def test_collapse_collinear_only_changes_flat_crossings(
    request: pytest.FixtureRequest, device: str, operation: Operation
) -> None:
    """
    The collinear collapse removes crossing vertices on flat faces, and nothing on curved ones.

    Ordito against ordito: the two settings of one function, whose ``False`` side
    ``test_boolean_matches_pymeshlab`` and ``test_boolean_matches_meshlib`` compare with the
    exact Booleans and whose default
    ``test_boolean_matches_manifold3d`` compares with Manifold. On the curved fixture the meshes
    are equal array for array; on the cylinder the collapse removes vertices and leaves the solid.
    """
    results = {}
    for shape in ("sphere_irregular", "cylinder"):
        a_tm, b_tm = _exact_operands(request, shape)
        operands = (
            *numpy_to_warp(a_tm.vertices, a_tm.faces, device),
            *numpy_to_warp(b_tm.vertices, b_tm.faces, device),
        )
        results[shape] = [
            operation(*operands, collapse_collinear=collapse) for collapse in (True, False)
        ]
    (curved_vertices, curved_faces), (kept_vertices, kept_faces) = results["sphere_irregular"]
    assert np.array_equal(curved_vertices.numpy(), kept_vertices.numpy())
    assert np.array_equal(curved_faces.numpy(), kept_faces.numpy())
    collapsed, kept = results["cylinder"]
    assert kept[0].size > collapsed[0].size
    assert np.isclose(od.measures.volume(*collapsed), od.measures.volume(*kept), rtol=1e-5)


@pytest.mark.parity("split_by_mesh", "manifold3d")
@pytest.mark.parametrize("case", ["spheres", "coplanar_overlap", "nested", "disjoint"])
def test_split_by_mesh_matches_manifold3d(device: str, case: str) -> None:
    """
    Both parts of a split are Manifold's ``Manifold.split``, array for array.

    Class A. The generic pair is two icospheres crossing; the boxes are the coplanar, nested and
    disjoint ties, where one part is the whole mesh or empty.
    """
    if case == "spheres":
        a_tm = float32_trimesh(tm.creation.icosphere(subdivisions=3))
        b_tm = _moved_copy(a_tm, seed=3)
    else:
        a_tm, b_tm = _DEGENERATE_CASES[case]
    inside_m3, outside_m3 = trimesh_to_manifold3d(a_tm).split(trimesh_to_manifold3d(b_tm))
    inside_wp, outside_wp = od.boolean.split_by_mesh(
        *numpy_to_warp(a_tm.vertices, a_tm.faces, device),
        *numpy_to_warp(b_tm.vertices, b_tm.faces, device),
    )
    assert_same_as_manifold3d(inside_wp, inside_m3)
    assert_same_as_manifold3d(outside_wp, outside_m3)


# Planes through the box and the sphere: axis-aligned through vertices and edges (every tie),
# diagonal, oblique, nearly axis-aligned, and clear of the mesh.
_PLANES = [
    ((0.0, 0.0, 1.0), 0.25),
    ((1.0, 0.0, 0.0), 0.0),
    ((0.0, 1.0, 0.0), 0.5),
    ((1.0, 1.0, 0.0), 0.0),
    ((0.3, -0.5, 0.8), 0.1),
    ((-1.0, 2.0, 3.0), -0.2),
    ((0.0, 0.0, -1.0), -0.25),
    ((-1.0, 0.0, 0.0), -0.5),
    ((0.0, -1.0, 0.0), 0.0),
    ((-1.0, -1.0, 0.2), 0.0),
    ((0.3, -0.5, -0.8), 0.1),
    ((1.0, 0.2, 0.0), 0.1),
    ((1.0, 1e-9, 1e-9), 0.1),
    ((0.0, 0.0, 1.0), 5.0),
]


@pytest.mark.parity("split_by_plane", "manifold3d")
@pytest.mark.parity("trim_by_plane", "manifold3d")
@pytest.mark.parametrize("shape", ["box", "sphere", "torus"])
@pytest.mark.parametrize(("normal", "offset"), _PLANES)
def test_plane_cuts_match_manifold3d(
    device: str, shape: str, normal: tuple[float, float, float], offset: float
) -> None:
    """
    ``split_by_plane`` and ``trim_by_plane`` are Manifold's, array for array.

    Class A. Manifold cuts with a cube it scales, moves and turns onto the plane in ``float64``,
    its rotation from its own ``sin`` / ``cos`` of degrees; ordito builds the same cube the same
    way, so planes through the box's vertices and edges tie exactly as Manifold's do. Normals
    along each negative axis and in every quadrant take each special case of that rotation's
    ``asin`` and ``atan2``; a shallow azimuth takes ``atan``'s unreduced polynomial, and a normal
    a nanoradian off the x axis its tiny-argument returns (and ``cos`` of a tiny angle). The last
    plane misses the mesh: everything on one side.
    """
    mesh_tm = {
        "box": _box([0, 0, 0], [1, 1, 1]),
        "sphere": float32_trimesh(tm.creation.icosphere(subdivisions=3)),
        "torus": float32_trimesh(tm.creation.torus(1.0, 0.3, 24, 12)),
    }[shape]
    mesh_m3 = trimesh_to_manifold3d(mesh_tm)
    above_m3, below_m3 = mesh_m3.split_by_plane(normal, offset)
    trimmed_m3 = mesh_m3.trim_by_plane(normal, offset)
    mesh_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    above_wp, below_wp = od.boolean.split_by_plane(*mesh_wp, normal, offset)
    assert_same_as_manifold3d(above_wp, above_m3)
    assert_same_as_manifold3d(below_wp, below_m3)
    assert_same_as_manifold3d(od.boolean.trim_by_plane(*mesh_wp, normal, offset), trimmed_m3)


def test_cuts_of_an_empty_mesh_are_empty(device: str) -> None:
    """
    An empty mesh cut leaves empty parts; an empty cutter leaves the whole mesh outside it.

    Not a library comparison: Manifold builds no solid from an empty mesh to compare with.
    """
    box_tm = _box([0, 0, 0], [1, 1, 1])
    box_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    empty_wp = (warp_empty(0, wp.vec3, device), warp_empty(0, wp.int32, device))
    above, below = od.boolean.split_by_plane(*empty_wp, (0.0, 0.0, 1.0), 0.5)
    assert above[1].size == 0
    assert below[1].size == 0
    assert od.boolean.trim_by_plane(*empty_wp, (0.0, 0.0, 1.0), 0.5)[1].size == 0
    inside, outside = od.boolean.split_by_mesh(*empty_wp, *box_wp)
    assert inside[1].size == 0
    assert outside[1].size == 0
    inside, outside = od.boolean.split_by_mesh(*box_wp, *empty_wp)
    assert inside[1].size == 0
    assert od.equality.meshes_allclose(*outside, *box_wp)


def test_plane_cut_rejects_a_zero_normal(device: str) -> None:
    """A zero normal raises ``ValueError``. Not a library comparison: the guard."""
    box_tm = _box([0, 0, 0], [1, 1, 1])
    mesh_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    with pytest.raises(ValueError, match="normal"):
        od.boolean.split_by_plane(*mesh_wp, (0.0, 0.0, 0.0))


_ATAN2_CASES = [
    (math.nan, 1.0),
    (1.0, math.nan),
    (0.5, 1.0),
    (0.0, -1.0),
    (-0.0, -1.0),
    (0.0, 1.0),
    (-0.0, 1.0),
    (1.0, 0.0),
    (-1.0, -0.0),
    (math.inf, math.inf),
    (math.inf, -math.inf),
    (-math.inf, math.inf),
    (-math.inf, -math.inf),
    (1.0, math.inf),
    (-1.0, math.inf),
    (1.0, -math.inf),
    (-1.0, -math.inf),
    (math.inf, 2.0),
    (1e300, 1e-300),
    (1e-300, -1e300),
    (1.0, 2.0),
    (1.0, -2.0),
    (-1.0, -2.0),
    (-1.0, 2.0),
    (3.0, 0.7),
    (0.2, 0.15),
    (1e20, 1.0),
    (1e-10, 1.0),
]
_ASIN_CASES = [math.nan, 2.0, -2.0, math.inf, 1.0, -1.0, 0.0, 1e-20, 0.3, -0.45, 0.7, -0.9]
_ATAN_CASES = [math.nan, math.inf, -math.inf, 1e300, -1e30, 1e-10, 0.2, -0.3, 0.6, 1.0, 2.0, -50.0]
_SIND_CASES = [0.0, 1e-7, 30.0, 45.0, -60.0, 90.0, 123.4, 180.0, -270.0, 720.0]


def test_manifold_math_matches_libm() -> None:
    """
    The ports of Manifold's ``math::atan2``, ``atan``, ``acos`` and ``asin`` agree with libm's.

    Not a library comparison: Manifold's own functions are not exposed to Python, so the
    reference is Python's ``math`` (the platform libm), which the first two reimplement. Special
    values (NaN, signed zeros, infinities, both arguments infinite, ratios past ``2**64``) must
    match exactly, sign of zero included; every other value within an ulp. Manifold defines
    ``asin(x)`` as ``pi / 2 - acos(x)``, so it is held to an ulp of ``pi / 2`` instead (it returns
    zero for ``1e-20``). The plane cuts build their half-space's rotation from ``asin`` and
    ``atan2``, then Manifold's ``sind`` / ``cosd`` of the angles in degrees: those are exact at
    multiples of 90 and within two ulps of ``sin(radians(x))`` elsewhere (the reduction to
    ``[-45, 45]`` degrees rounds differently), and NaN off the finite numbers.
    """

    def same(value: float, reference: float, tolerance: float = 0.0) -> bool:
        if math.isnan(reference):
            return math.isnan(value)
        if math.isinf(reference) or (reference == 0.0 and tolerance == 0.0):
            return value == reference and math.copysign(1.0, value) == math.copysign(1.0, reference)
        return abs(value - reference) <= (tolerance or math.ulp(reference))

    for y, x in _ATAN2_CASES:
        assert same(od_boolean._m_atan2(y, x), math.atan2(y, x)), (y, x)  # pyright: ignore[reportPrivateUsage]
    half_pi_ulp = math.ulp(math.pi / 2)
    for x in _ASIN_CASES:
        inside = -1.0 <= x <= 1.0
        acos_reference = math.acos(x) if inside else math.nan
        assert same(od_boolean._m_acos(x), acos_reference), x  # pyright: ignore[reportPrivateUsage]
        asin_reference = math.asin(x) if inside else math.nan
        assert same(od_boolean._m_asin(x), asin_reference, half_pi_ulp), x  # pyright: ignore[reportPrivateUsage]
    for x in _ATAN_CASES:
        assert same(od_boolean._m_atan(x), math.atan(x)), x  # pyright: ignore[reportPrivateUsage]
    for x in _SIND_CASES:
        sine = od_boolean._sind(x)  # pyright: ignore[reportPrivateUsage]
        cosine = od_boolean._cosd(x)  # pyright: ignore[reportPrivateUsage]
        if x % 90.0 == 0.0:
            assert (sine, cosine) == (
                [0.0, 1.0, 0.0, -1.0][int(x // 90.0) % 4],
                [1.0, 0.0, -1.0, 0.0][int(x // 90.0) % 4],
            ), x
        else:
            assert abs(sine - math.sin(math.radians(x))) <= 2 * math.ulp(1.0), x
            assert abs(cosine - math.cos(math.radians(x))) <= 2 * math.ulp(1.0), x
    for x in (math.inf, -math.inf, math.nan):
        assert math.isnan(od_boolean._sind(x)), x  # pyright: ignore[reportPrivateUsage]


def _batch_meshes(case: str) -> list[tm.Trimesh]:
    """
    Return a batch case's meshes: one, touching in a row, scattered, apart beside a bar, or many.

    ``many`` is nine coarse spheres all overlapping one another, enough for the pairing heap to
    combine four pairs in one round.
    """
    if case == "single":
        return [float32_trimesh(tm.creation.icosphere(subdivisions=2))]
    if case == "many":
        rng = np.random.default_rng(11)
        spheres = [tm.creation.icosphere(subdivisions=1) for _ in range(9)]
        for sphere in spheres:
            sphere.apply_translation(rng.uniform(-0.3, 0.3, 3))
        return [float32_trimesh(sphere) for sphere in spheres]
    if case == "row":
        return [_box([0.75 * i, 0, 0], [0.75 * i + 1, 1, 1]) for i in range(3)]
    rng = np.random.default_rng(3 if case == "scattered" else 5)
    meshes: list[tm.Trimesh] = []
    for i in range(6):
        sphere = tm.creation.icosphere(subdivisions=2, radius=float(rng.uniform(0.4, 0.8)))
        offset = rng.uniform(-1.0, 1.0, 3) if case == "scattered" else np.array([3.0 * i, 0, 0])
        sphere.apply_translation(offset)
        meshes.append(float32_trimesh(sphere))
    if case == "apart":
        meshes.append(_box([0.5, 0, 0], [4, 0.5, 0.5]))
    return meshes


@pytest.mark.parity("union_all", "manifold3d")
@pytest.mark.parity("intersection_all", "manifold3d")
@pytest.mark.parity("difference_all", "manifold3d")
@pytest.mark.parametrize("case", ["single", "row", "scattered", "apart", "many"])
@pytest.mark.parametrize(
    ("function", "op_m3"),
    [
        (od.boolean.union_all, m3.OpType.Add),
        (od.boolean.intersection_all, m3.OpType.Intersect),
        (od.boolean.difference_all, m3.OpType.Subtract),
    ],
    ids=["union", "intersection", "difference"],
)
def test_batch_booleans_match_manifold3d(
    device: str, case: str, function: Callable[[Sequence[WarpMesh]], WarpMesh], op_m3: m3.OpType
) -> None:
    """
    The batch Booleans are Manifold's ``batch_boolean``, array for array.

    Class A. ``apart`` holds spheres whose boxes do not touch, which Manifold composes rather
    than intersects, then joins to the bar across them; ``scattered`` overlaps in a chain the
    pairing order decides; ``row`` touches face to face; ``single`` is one mesh, returned as it is;
    ``many`` overlaps nine ways, so a round of the pairing heap combines four pairs.
    """
    meshes_tm = _batch_meshes(case)
    result_m3 = m3.Manifold.batch_boolean(
        [trimesh_to_manifold3d(mesh) for mesh in meshes_tm], op_m3
    )
    result_wp = function([numpy_to_warp(mesh.vertices, mesh.faces, device) for mesh in meshes_tm])
    assert_same_as_manifold3d(result_wp, result_m3)


@pytest.mark.parity("union_all", "meshlib")
@pytest.mark.parametrize("case", ["scattered", "apart"])
def test_union_all_matches_meshlib(device: str, case: str) -> None:
    """
    ``union_all`` is MeshLib's ``uniteManyMeshes`` up to the triangulation of each cut face.

    Class B, the transform of ``test_boolean_matches_meshlib``: MeshLib does not simplify its
    result, and on these curved operands neither does the collinear collapse, so both results
    keep every crossing vertex. ``uniteManyMeshes`` unites its meshes in its own order (nested
    components removed, no random shifts by default), which leaves the same arrangement.
    ``scattered`` overlaps in a chain. ``apart`` is the spheres of that case whose boxes do not
    touch, which ``union_all`` joins without a Boolean, without its bar: the bar is a box, whose
    flat faces Manifold's collapse clears of collinear cut vertices that MeshLib keeps (with the
    bar: 1 047 vertices and 2 074 faces against 1 010 and 2 000, the same volume). ``row``'s
    boxes touch face to face, where MeshLib also keeps the contact sheets Manifold removes; both
    are left to ``test_batch_booleans_match_manifold3d``.
    """
    meshes_tm = _batch_meshes(case)[:6]
    meshes_ml = mm.std_vector_Mesh_const_ptr([trimesh_to_meshlib(mesh) for mesh in meshes_tm])
    result_tm = meshlib_to_trimesh(mm.uniteManyMeshes(meshes_ml, mm.UniteManyMeshesParams()))
    volume_sum = sum(float(mesh.volume) for mesh in meshes_tm)
    if case == "scattered":  # non-vacuity: the spheres overlap (0.97 of the summed volume)
        assert float(result_tm.volume) < 0.99 * volume_sum
    else:  # the disjoint shortcut: a concatenation
        assert np.isclose(float(result_tm.volume), volume_sum, rtol=1e-5)
    result_wp = od.boolean.union_all(
        [numpy_to_warp(mesh.vertices, mesh.faces, device) for mesh in meshes_tm]
    )
    assert_same_cut_faces(result_wp, result_tm.vertices, result_tm.faces, meshes_tm)


def _plane_tm(normal: tuple[float, float, float], offset: float, half_width: float) -> tm.Trimesh:
    """Return a square of two triangles on the plane ``normal . x == offset`` (unit normal)."""
    normal_np = np.asarray(normal, dtype=np.float64) / np.linalg.norm(normal)
    first_np = np.cross(normal_np, [1.0, 0.0, 0.0] if abs(normal_np[0]) < 0.9 else [0.0, 1.0, 0.0])
    first_np /= np.linalg.norm(first_np)
    second_np = np.cross(normal_np, first_np)
    centre_np = normal_np * offset
    corners_np = [
        centre_np + half_width * (s * first_np + t * second_np)
        for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return tm.Trimesh(np.array(corners_np), [[0, 1, 2], [0, 2, 3]], process=False)


# Planes through the sphere fixtures, offset from the vertex mean: oblique, axis-aligned (through
# the icosphere's equator vertices), diagonal.
_TRIMESH_PLANES = [((0.3, -0.5, 0.8), 0.1), ((0.0, 0.0, 1.0), 0.0), ((1.0, 1.0, 0.0), 0.0)]


@pytest.mark.parity("split_by_plane", "trimesh")
@pytest.mark.parity("trim_by_plane", "trimesh")
@pytest.mark.parametrize("mesh_name", ["icosphere", *CLOSED_MESHES])
@pytest.mark.parametrize(("normal", "offset"), _TRIMESH_PLANES)
def test_plane_cuts_match_trimesh(
    request: pytest.FixtureRequest,
    device: str,
    mesh_name: str,
    normal: tuple[float, float, float],
    offset: float,
) -> None:
    """
    The plane cuts are trimesh's capped ``slice_mesh_plane`` up to the triangulation of each face.

    Class B through ``assert_same_cut_faces``, with the plane itself as a third operand so the cap
    has a plane to be assigned to: the side faces, cut into polygons along the plane, and the cap,
    one polygon per section loop, must match polygon by polygon. trimesh keeps the side of the
    normal it is handed, so ``-normal`` gives ``split_by_plane``'s second half.

    Curved fixtures only. On a box or a torus, whose grid quads are planar, Manifold removes the
    cut vertices that fall collinear on a flat polygon and trimesh keeps them (the box cut by
    ``z = 0.25``: 8 against 12 vertices, the same volume), and on the torus cut by the oblique
    plane trimesh's cap misses one triangle (365 faces against 366, not watertight).
    """
    mesh_tm, _ = request.getfixturevalue(mesh_name)
    mesh_tm = float32_trimesh(mesh_tm)
    normal_np = np.asarray(normal) / np.linalg.norm(normal)
    offset += float(mesh_tm.vertices.mean(axis=0) @ normal_np)
    origin_np = offset * normal_np
    above_tm = tm.intersections.slice_mesh_plane(mesh_tm, normal_np, origin_np, cap=True)
    below_tm = tm.intersections.slice_mesh_plane(mesh_tm, -normal_np, origin_np, cap=True)
    assert above_tm is not None
    assert below_tm is not None
    for half_tm in (above_tm, below_tm):
        half_tm.remove_unreferenced_vertices()
        assert 0.0 < float(half_tm.volume) < float(mesh_tm.volume)  # non-vacuity: a real cut

    mesh_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    operands = [mesh_tm, _plane_tm(normal, offset, 10.0 * float(mesh_tm.scale) + abs(offset))]
    trimmed_wp = od.boolean.trim_by_plane(*mesh_wp, normal, offset)
    assert_same_cut_faces(trimmed_wp, above_tm.vertices, above_tm.faces, operands)
    above_wp, below_wp = od.boolean.split_by_plane(*mesh_wp, normal, offset)
    assert_same_cut_faces(above_wp, above_tm.vertices, above_tm.faces, operands)
    assert_same_cut_faces(below_wp, below_tm.vertices, below_tm.faces, operands)


def test_batch_booleans_reject_an_empty_list() -> None:
    """No mesh at all raises ``ValueError``. Not a library comparison: the guard."""
    with pytest.raises(ValueError, match="at least one mesh"):
        od.boolean.union_all([])


def _defective_box(case: str) -> tm.Trimesh:
    """
    Return a unit box with one defect Manifold's solid constructor repairs, unprocessed.

    ``pinched``: a second box sharing one corner vertex; ``collinear``: an edge split at its
    midpoint on one side only, closed by a zero-area triangle; ``short_edge``: an edge split in
    both its faces at a point on one of its ends; ``repeated_index``: a face naming a vertex twice,
    listed first; ``unused_vertex``: a vertex no face names.
    """
    box_tm = _box([0, 0, 0], [1, 1, 1])
    vertices_np = np.asarray(box_tm.vertices, dtype=np.float32)
    faces_np = np.asarray(box_tm.faces, dtype=np.int64)
    a = int(np.flatnonzero((vertices_np == 0.0).all(axis=1))[0])
    b = int(np.flatnonzero((vertices_np == [1.0, 0.0, 0.0]).all(axis=1))[0])
    if case == "pinched":
        other_tm = _box([1, 1, 1], [2, 2, 2])
        other_np = np.asarray(other_tm.vertices, dtype=np.float32)
        shared = int(np.flatnonzero((other_np == 1.0).all(axis=1))[0])
        keep = np.arange(len(other_np)) != shared
        index_np = np.cumsum(keep) - 1 + len(vertices_np)
        index_np[shared] = int(np.flatnonzero((vertices_np == 1.0).all(axis=1))[0])
        vertices_np = np.vstack([vertices_np, other_np[keep]])
        faces_np = np.vstack([faces_np, index_np[np.asarray(other_tm.faces)]])
    elif case in ("collinear", "short_edge"):
        m = len(vertices_np)
        point = [0.5, 0.0, 0.0] if case == "collinear" else vertices_np[a]
        vertices_np = np.vstack([vertices_np, [point]]).astype(np.float32)
        rows: list[list[int]] = []
        for face in faces_np.tolist():
            for k in range(3):
                p, q, r = face[k], face[(k + 1) % 3], face[(k + 2) % 3]
                if (p, q) == (a, b):
                    rows += [[a, m, r], [m, b, r]]
                    break
                if (p, q) == (b, a) and case == "short_edge":
                    rows += [[b, m, r], [m, a, r]]
                    break
                if (p, q) == (b, a):
                    rows += [face, [b, m, a]]
                    break
            else:
                rows.append(face)
        faces_np = np.asarray(rows)
    elif case == "repeated_index":
        faces_np = np.vstack([[[a, a, b]], faces_np])
    else:
        vertices_np = np.vstack([vertices_np, [[5.0, 5.0, 5.0]]]).astype(np.float32)
    return tm.Trimesh(vertices_np, faces_np, process=False)


_DEFECTS = ["pinched", "collinear", "short_edge", "repeated_index", "unused_vertex"]


@pytest.mark.parity("union_all", "manifold3d")
@pytest.mark.parity("boolean_difference", "manifold3d")
@pytest.mark.parametrize("case", _DEFECTS)
def test_solids_are_cleaned_as_manifold3d_cleans_them(device: str, case: str) -> None:
    """
    A mesh with a defect Manifold's solid constructor repairs gives Manifold's answers.

    Class A, array for array. Manifold drops a triangle naming a vertex twice, splits a pinched
    vertex, collapses a zero-length edge, swaps degenerate triangles and drops unused vertices as
    it builds a solid, and ordito builds its solids the same way: the solid alone (a one-mesh
    ``union_all``) and its difference with a ball. The case's own counts are asserted so each
    repair is seen to happen (or, for the collinear triangle, Manifold to keep it).
    """
    mesh_tm = _defective_box(case)
    ball_tm = float32_trimesh(tm.creation.icosphere(subdivisions=2, radius=0.6))
    ball_tm.apply_translation([0.5, 0.0, 0.5])
    built_m3 = trimesh_to_manifold3d(mesh_tm)
    difference_m3 = built_m3 - trimesh_to_manifold3d(ball_tm)
    expected = {
        "pinched": (16, 24),
        "collinear": (9, 14),
        "short_edge": (8, 12),
        "repeated_index": (8, 12),
        "unused_vertex": (8, 12),
    }[case]
    assert (built_m3.num_vert(), built_m3.num_tri()) == expected
    mesh_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    ball_wp = numpy_to_warp(ball_tm.vertices, ball_tm.faces, device)
    assert_same_as_manifold3d(od.boolean.union_all([mesh_wp]), built_m3)
    assert_same_as_manifold3d(od.boolean.difference(*mesh_wp, *ball_wp), difference_m3)


@pytest.mark.parametrize(("name", "operation"), OPERATIONS, ids=_OPERATION_IDS)
def test_boolean_with_an_empty_operand(device: str, name: str, operation: Operation) -> None:
    """
    An empty operand leaves the other mesh, or nothing, as a solid's set algebra says.

    Not a library comparison: Manifold builds no solid from an empty mesh to compare with, so the
    cases are checked against their definitions: the union is the other operand, the
    intersection empty, the difference the first operand (or empty when that is the empty one).
    """
    box_tm = _box([0, 0, 0], [1, 1, 1])
    vertices, faces = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    empty_vertices = warp_empty(0, wp.vec3, device)
    empty_faces = warp_empty(0, wp.int32, device)
    _, faces_left = operation(empty_vertices, empty_faces, vertices, faces)
    _, faces_right = operation(vertices, faces, empty_vertices, empty_faces)
    expected_left = {"union": 12, "intersection": 0, "difference": 0}[name]
    expected_right = {"union": 12, "intersection": 0, "difference": 12}[name]
    assert faces_left.size // 3 == expected_left
    assert faces_right.size // 3 == expected_right


@pytest.mark.parity("union_all", "manifold3d")
@pytest.mark.parametrize("corner", [0.0, 2.0])
def test_union_all_with_an_empty_mesh_matches_manifold3d(device: str, corner: float) -> None:
    """
    An empty mesh in a batch union leaves the other mesh: Manifold's ``batch_boolean``.

    Class A against ``batch_boolean`` with an empty ``Manifold()`` beside a box, array for array.
    The empty solid's box is Manifold's empty ``Box``, which meets no other, so the box and the
    empty mesh are composed (the concatenation of an empty part, and the tolerance of a solid of no
    scale) and sorted in the box's own frame. Its box was once the origin point, which a box clear
    of the origin stretched the composed Morton frame to, reordering the faces: the ``corner=2``
    case.
    """
    box_tm = _box([corner] * 3, [corner + 1.0] * 3)
    result_m3 = m3.Manifold.batch_boolean(
        [trimesh_to_manifold3d(box_tm), m3.Manifold()], m3.OpType.Add
    )
    assert result_m3.num_tri() == 12  # non-vacuity: the box survives
    empty_wp = (warp_empty(0, wp.vec3, device), warp_empty(0, wp.int32, device))
    result_wp = od.boolean.union_all(
        [numpy_to_warp(box_tm.vertices, box_tm.faces, device), empty_wp]
    )
    assert_same_as_manifold3d(result_wp, result_m3)


def test_boolean_with_only_degenerate_triangles_is_an_empty_operand(device: str) -> None:
    """
    An operand whose every triangle names a vertex twice is dropped to an empty solid.

    Class A against Manifold's solid of the box alone, for the union and the difference; the
    intersection is empty. Manifold drops such triangles as it builds a solid (see
    ``test_solids_are_cleaned_as_manifold3d_cleans_them``'s ``repeated_index``), but given only
    those it refuses the mesh (``NotManifold``), so the operand is held to the set algebra of an
    empty one. Excludes keeping a degenerate triangle (the union would gain faces) and failing the
    pairing on an operand left with no faces.
    """
    box_tm = _box([0, 0, 0], [1, 1, 1])
    box_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces_np = np.array([[0, 0, 1], [1, 2, 2], [2, 0, 0]], dtype=np.int32)
    degenerate_m3 = m3.Manifold(m3.Mesh(vertices_np, faces_np.astype(np.uint32)))
    assert degenerate_m3.status() == m3.Error.NotManifold
    degenerate_wp = numpy_to_warp(vertices_np, faces_np, device)
    box_m3 = trimesh_to_manifold3d(box_tm)
    assert_same_as_manifold3d(od.boolean.union(*box_wp, *degenerate_wp), box_m3)
    assert_same_as_manifold3d(od.boolean.difference(*box_wp, *degenerate_wp), box_m3)
    assert od.boolean.intersection(*box_wp, *degenerate_wp)[1].size == 0


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "boy_surface"])
def test_boolean_rejects_open_or_non_orientable_meshes(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    A mesh with a rim, or one that cannot be wound consistently, raises ``ValueError``.

    Not a library comparison: the contract is the error. ``saddle_graded`` has a rim and
    ``boy_surface`` no consistent winding; both operand positions are checked.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    box_tm = _box([0, 0, 0], [1, 1, 1])
    box_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, mesh_wp.device)
    with pytest.raises(ValueError, match="closed, edge-manifold and consistently wound"):
        od.boolean.union(mesh_wp.points, mesh_wp.indices, *box_wp)
    with pytest.raises(ValueError, match="closed, edge-manifold and consistently wound"):
        od.boolean.difference(*box_wp, mesh_wp.points, mesh_wp.indices)


def test_boolean_requires_one_device() -> None:
    """Operands on two devices raise ``RuntimeError``. Not a library comparison: the guard."""
    if not wp.is_cuda_available():
        pytest.skip("needs a CUDA device beside the CPU")
    box_tm = _box([0, 0, 0], [1, 1, 1])
    with pytest.raises(RuntimeError):
        od.boolean.union(
            *numpy_to_warp(box_tm.vertices, box_tm.faces, "cpu"),
            *numpy_to_warp(box_tm.vertices, box_tm.faces, "cuda:0"),
        )
