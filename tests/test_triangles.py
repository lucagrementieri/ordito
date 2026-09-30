"""Regression tests for ``triwarp.triangles`` against ``trimesh.triangles`` (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import pytorch3d.ops as p3d_ops
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import triwarp as tw
import triwarp.typing as twt
from tests.comparisons import assert_nonconstant
from tests.conversions import (
    faces_igl,
    meshlib_corner_normals_to_numpy,
    numpy_to_meshlib_undirected_edges,
    numpy_to_warp,
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pytorch3d,
    trimesh_to_pyvista,
)


@pytest.mark.parity("face_normals_and_areas", "trimesh")
def test_face_normals_and_areas(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the unit normal and the area per face, both against trimesh.

    The two come out of one cross product, so comparing both is what separates a normalization
    bug from a winding one -- a flipped face has the right area and the wrong normal.
    """
    mesh_tm, mesh_wp = icosahedron
    normal_wp, area_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(normal_wp.numpy(), mesh_tm.face_normals, rtol=1e-5, atol=1e-5)
    assert np.allclose(area_wp.numpy(), mesh_tm.area_faces, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("face_normals_and_areas", "pytorch3d")
def test_face_normals_and_areas_matches_pytorch3d(icosphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: ``mesh_face_areas_normals`` is bit-identical to triwarp's, on both outputs.

    Not merely within tolerance -- **exactly** 0.0 on this fixture, because the two implementations
    are the same three lines: ``(v1 - v0) x (v2 - v0)``, its norm halved for the area, and the same
    cross product normalized for the normal. Worth having as a separate test from the trimesh one
    for that reason: a reference that agrees to 1e-7 leaves room for a different summation order,
    and one that agrees to 0.0 does not.

    Note the areas come back **float32** whatever the ``Meshes`` was built from -- the C++ kernel
    casts -- so a float64 comparison here would be measuring pytorch3d's own downcast.

    The ``Meshes`` is built on the **triwarp side's own device**, which is what makes the exact
    claim hold on both: pytorch3d has separate CPU and CUDA kernels, and each agrees bit-for-bit
    with triwarp's on the same device while a ``pytorch3d``-on-host against ``triwarp``-on-CUDA
    comparison lands at 1.86e-09 on the areas and 1.19e-07 on the normals. So this is also one of
    the tests section 6's device rule asks for -- it exercises the reference's *own* two backends
    rather than trusting the CPU pass.
    """
    mesh_tm, mesh_wp = icosphere
    mesh_p3d = trimesh_to_pytorch3d(mesh_tm, str(mesh_wp.points.device))
    areas_p3d, normals_p3d = p3d_ops.mesh_face_areas_normals(
        mesh_p3d.verts_packed(), mesh_p3d.faces_packed()
    )
    normals_wp, areas_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    assert areas_p3d.shape == (mesh_tm.faces.shape[0],)
    assert_nonconstant(areas_p3d.cpu().numpy(), tol=1e-5)
    assert np.array_equal(areas_wp.numpy(), areas_p3d.cpu().numpy())
    assert np.array_equal(normals_wp.numpy(), normals_p3d.cpu().numpy())


@pytest.mark.parity("face_normals_and_areas", "igl", "potpourri3d", "pymeshlab")
def test_face_normals_and_areas_against_the_partial_references(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
):
    """
    The three references that each answer *half* of this function, so the benchmark can be read.

    ``face_normals_and_areas`` returns both quantities from one cross product, and the benchmark
    reads these three rows as a floor rather than a fair race because each computes less. That makes
    them no *less* valid as oracles, only partial. All three are Class B -- exact transforms, full
    ``1e-5`` tolerance:

    - ``igl.doublearea`` is literally twice the area, so the transform is a factor of two;
    - ``potpourri3d.face_areas`` is the area directly, and is pure numpy rather than
      geometry-central, so it is the weakest of the three as independent evidence;
    - MeshLab's ``compute_normal_per_face`` turns out to check **both** halves, not one.
      ``face_normal_matrix()`` returns the *unnormalised* cross product: measured on this fixture
      its magnitude is ``2 * area`` to within 1e-15 across every face, and its direction matches
      trimesh's unit normals exactly. So dividing by the magnitude gives the normal and halving the
      magnitude gives the area, and this row is a full oracle rather than a partial one. Do not
      "fix" a failure here by normalising triwarp's side -- triwarp's normals are already unit, and
      the magnitude is the area check.

    Uses ``half_torus`` rather than ``icosahedron`` so the areas actually vary across faces: on a
    mesh of 20 congruent triangles a factor-of-two error in one reference and a wrong *constant* in
    triwarp are indistinguishable.
    """
    mesh_tm, mesh_wp = half_torus
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = faces_igl(mesh_tm)

    normals_wp, areas_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    areas_igl = np.asarray(igl.doublearea(vertices_np, faces_np)) / 2.0
    assert np.allclose(areas_wp.numpy(), areas_igl, rtol=1e-5, atol=1e-5)

    areas_pp = pp3d.face_areas(vertices_np, mesh_tm.faces.astype(np.int32))
    assert np.allclose(areas_wp.numpy(), areas_pp, rtol=1e-5, atol=1e-5)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_normal_per_face()
    crosses_pml = meshset_pml.current_mesh().face_normal_matrix()
    magnitudes_pml = np.linalg.norm(crosses_pml, axis=1)
    assert np.allclose(areas_wp.numpy(), magnitudes_pml / 2.0, rtol=1e-5, atol=1e-5)
    assert np.allclose(
        normals_wp.numpy(), crosses_pml / magnitudes_pml[:, None], rtol=1e-5, atol=1e-5
    )


@pytest.mark.parity("corner_normals", "meshlib")
@pytest.mark.parametrize("mesh_name", ["unit_box", "icosphere_coarse", "cave_cube"])
@pytest.mark.parametrize("crease_angle", [None, 0.5])
def test_corner_normals_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str, crease_angle: float | None
) -> None:
    """
    Class A against ``computePerCornerNormals``, at the **area** weighting and with creases.

    The weighting is the finding this row exists to pin. MeshLib's per-corner normals are
    area-weighted, not angle-weighted -- measured, they agree with ``weighting="area"`` to 1.2e-07
    and sit **0.244** from ``weighting="angle"`` on ``unit_box`` and 0.521 on a cylinder. Which is
    the same split MeshLib already forces on the *vertex* normals (``computePerVertNormals`` pairs
    with area weighting, ``computePerVertPseudoNormals`` with angle), so it is a convention this
    reference distinguishes and the others do not.

    Both crease cases are run because they exercise different code: with ``None`` every rotation
    about a vertex completes and the answer collapses to the vertex normal, while at 0.5 rad the
    walk stops at hard edges and the per-corner answer is the point of the function. Measured 0.0 on
    ``unit_box`` with its 12 creases and 6e-08 on a cylinder with 96 of 192.

    The fixtures are **closed and undistorted** deliberately. On ``half_torus``, whose conftest
    fixture scales its vertices by ``exp(-y)``, the same comparison reads **1.8e-05** -- a float32
    floor rather than a disagreement, because at 0.5 rad that mesh has 1 099 creases and a
    normalized sum of a thousand near-cancelling weights amplifies the two libraries' 2.6e-08
    storage difference. Loosening the tolerance to admit it would weaken the row everywhere else.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    n_faces = faces_wp.size // 3
    mesh_ml = trimesh_to_meshlib(mesh_tm)

    creases_wp = None
    creases_ml = mm.UndirectedEdgeBitSet()
    if crease_angle is not None:
        creases_wp = tw.seams.crease_edges(vertices_wp, faces_wp, angle=crease_angle)
        # Non-vacuity: at this angle the fixture must actually have hard edges, or the case below
        # is the no-crease one again under a different name.
        assert int(creases_wp.shape[0]) > 0
        creases_ml = numpy_to_meshlib_undirected_edges(mesh_ml.topology, creases_wp.numpy())

    normals_ml = meshlib_corner_normals_to_numpy(
        mm.computePerCornerNormals(mesh_ml, creases_ml), n_faces
    )
    normals_wp = tw.triangles.corner_normals(vertices_wp, faces_wp, creases_wp, weighting="area")
    assert normals_wp.shape == (n_faces, 3)
    assert np.allclose(normals_wp.numpy(), normals_ml, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["unit_box", "icosphere_coarse", "hemisphere"])
@pytest.mark.parametrize("weighting", ["angle", "area"])
def test_corner_normals_degenerate_crease_sets_are_exact(
    request: pytest.FixtureRequest, mesh_name: str, weighting: str
) -> None:
    """
    Not a library comparison: the two crease sets whose answer is another triwarp function, exactly.

    These are the strongest available checks on the rotation walk, because both sides are computed
    by different code and must agree to float32 and not to a tolerance:

    * **no creases** -- every rotation completes, so each corner gets its *vertex's* normal under
      the same weighting, compared against ``triwarp.vertices``' own function;
    * **every edge a crease** -- no rotation moves at all, so each corner gets its own *face's*
      normal, whatever the weighting.

    ``hemisphere`` is in the list on purpose: a boundary vertex's fan is a path rather than a
    cycle, so the walk has to terminate on the rim in both directions and still cover the whole
    fan. Getting that wrong shows up here and nowhere else.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    faces_np = faces_wp.numpy().reshape(-1, 3)

    smooth_np = tw.triangles.corner_normals(vertices_wp, faces_wp, weighting=weighting).numpy()
    vertex_normals_wp = (
        tw.vertices.vertex_normals(vertices_wp, faces_wp, weighting="angle")
        if weighting == "angle"
        else tw.vertices.vertex_normals(vertices_wp, faces_wp)
    )
    assert np.allclose(smooth_np, vertex_normals_wp.numpy()[faces_np], rtol=1e-5, atol=1e-5)

    every_edge_wp = tw.edges.faces_to_edges(faces_wp, sorted=True)
    hard_np = tw.triangles.corner_normals(
        vertices_wp, faces_wp, every_edge_wp, weighting=weighting
    ).numpy()
    face_normals_np = tw.triangles.face_normals_and_areas(vertices_wp, faces_wp)[0].numpy()
    assert np.allclose(hard_np, face_normals_np[:, None, :], rtol=1e-5, atol=1e-5)
    # And the two extremes must differ, or neither comparison above is testing the walk.
    assert not np.allclose(smooth_np, hard_np, rtol=1e-3, atol=1e-3)


def test_corner_normals_ccw_first_step_crease_does_not_skip_clockwise_walk(device: str) -> None:
    """
    Not a library comparison: a regression pin for a loop-closure sentinel collision.

    Four faces ring a single shared vertex ``v``, with exactly one crease edge at ``{v, p0}``.
    Every corner but one reaches the crease partway through its counter-clockwise walk, which
    correctly leaves ``halfedge != corner`` afterwards and runs the clockwise half too. The fourth
    corner's *own* ``halfedge_prev`` edge -- the very first one tested -- **is** the crease, so its
    counter-clockwise loop breaks on the first iteration, before ``halfedge`` is ever reassigned
    from its initial value of ``corner``. Conflating that with "the fan closed" (which also leaves
    ``halfedge == corner``) skipped the clockwise walk for exactly that one corner, collapsing its
    answer to its own face's normal instead of the four-face smooth-group average every other
    corner at ``v`` gets -- removing a single edge from a 4-face ring around a vertex still leaves
    every face reachable from every corner by going around the other way.
    """
    vertices_np = np.array(
        [
            [0.0, 0.0, 0.0],  # v
            [1.0, 0.0, 0.0],  # p0
            [0.0, 1.0, 0.0],  # p1
            [-1.0, 0.0, 0.0],  # p2
            [0.0, -1.0, 0.0],  # p3
        ],
        dtype=np.float32,
    )
    # F0=(v,p0,p1), F1=(v,p1,p2), F2=(v,p2,p3), F3=(v,p3,p0) -- each face lists v first, so v's
    # corner within face i is flat index 3*i, and F3's own ``halfedge_prev`` edge is {p0, v}.
    faces_np = np.array([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1], dtype=np.int32)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    crease_wp = wp.array(np.array([[0, 1]], dtype=np.int32), device=device)

    normals_wp = tw.triangles.corner_normals(
        vertices_wp, faces_wp, crease_wp, weighting="area"
    ).numpy()
    # All four faces are wound the same way in the xy-plane, so every one of them has the flat
    # +z normal -- and every corner at v must agree with it, since none of the four is excluded by
    # the single crease.
    corners_at_v = normals_wp[:, 0, :]
    assert np.allclose(corners_at_v, np.array([0.0, 0.0, 1.0]), rtol=1e-5, atol=1e-5)


def test_corner_normals_edge_cases(device: str, unit_box: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a library comparison: an empty mesh, and the two argument errors."""
    empty_vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    empty_faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.triangles.corner_normals(empty_vertices_wp, empty_faces_wp).shape == (0, 3)

    _mesh_tm, mesh_wp = unit_box
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    with pytest.raises(ValueError, match=r"shape \(k, 2\)"):
        tw.triangles.corner_normals(
            vertices_wp,
            faces_wp,
            twt.as_array2d(wp.zeros((2, 3), dtype=wp.int32, device=device), wp.int32),
        )
    with pytest.raises(ValueError, match="weighting must be"):
        tw.triangles.corner_normals(vertices_wp, faces_wp, weighting="sine")


@pytest.mark.parity("face_normals_and_areas", "open3d")
def test_face_normals_matches_open3d(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on the normals half; the areas reduce to open3d's one total.

    ``compute_triangle_normals`` returns *unit* normals, unlike MeshLab's raw cross product above,
    so no transform is needed. The area half has no per-face open3d counterpart
    (``get_surface_area`` is the total), so the sum is compared as its class-B reduction.
    ``half_torus`` for the same varying-area reason as the partial-references test.
    """
    mesh_tm, mesh_wp = half_torus
    normals_wp, areas_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    mesh_o3d = trimesh_to_open3d(mesh_tm)
    mesh_o3d.compute_triangle_normals()
    assert np.allclose(normals_wp.numpy(), np.asarray(mesh_o3d.triangle_normals), atol=1e-5)
    assert np.isclose(float(areas_wp.numpy().sum()), mesh_o3d.get_surface_area(), rtol=1e-5)


@pytest.mark.parity("face_normals_and_areas", "pyvista")
def test_face_normals_and_areas_match_pyvista(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on both halves, from the one reference that answers both -- in two calls.

    VTK splits the cross product across two filters: ``compute_normals`` gives the *unit* cell
    normal (float32, per this module's dtype note) and ``compute_cell_sizes`` the area. The three
    ``compute_normals`` flags matter and are passed explicitly: ``consistent_normals`` and
    ``auto_orient_normals`` would let VTK re-wind the mesh before differentiating it, which would
    compare triwarp's normals against a *different* orientation, and ``split_vertices`` would change
    the point count. ``half_torus`` for the varying-area reason the partial-references test gives.
    """
    mesh_tm, mesh_wp = half_torus
    normals_wp, areas_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    mesh_pv = trimesh_to_pyvista(mesh_tm)
    normals_pv = mesh_pv.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=False,
        auto_orient_normals=False,
        split_vertices=False,
    )
    assert np.allclose(
        normals_wp.numpy(), np.asarray(normals_pv.cell_data["Normals"]), rtol=1e-5, atol=1e-5
    )
    sizes_pv = mesh_pv.compute_cell_sizes(length=False, area=True, volume=False)
    assert np.allclose(
        areas_wp.numpy(), np.asarray(sizes_pv.cell_data["Area"]), rtol=1e-5, atol=1e-5
    )


def test_angles(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the three corner angles per face against ``Trimesh.face_angles``, in corner order.

    Column order is part of the claim -- angle ``i`` is at corner ``i`` -- because the
    cotangent Laplacian and the angle defect both index it that way.
    """
    mesh_tm, mesh_wp = half_torus
    angles_wp = tw.triangles.face_angles(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(angles_wp.numpy(), mesh_tm.face_angles, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("metric", "filter_metric"),
    [
        ("radius_ratio", "inradius/circumradius"),
        ("area_max_side", "area/max side"),
        ("mean_ratio", "Mean ratio"),
        ("area", "Area"),
    ],
)
@pytest.mark.parity("face_quality", "pymeshlab")
def test_face_quality_against_pymeshlab(
    half_torus: tuple[tm.Trimesh, wp.Mesh], metric: str, filter_metric: str
):
    """Class B (per-measure naming): the four VCG measures, against the filter they came from."""
    mesh_tm, mesh_wp = half_torus
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_scalar_by_aspect_ratio_per_face(metric=filter_metric)
    quality_pml = meshset_pml.current_mesh().face_scalar_array()

    quality_wp = tw.triangles.face_quality(mesh_wp.points, mesh_wp.indices, metric=metric)
    assert np.allclose(quality_wp.numpy(), quality_pml, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("metric", "measure", "reciprocal"),
    [
        ("area", "area", False),
        ("aspect_ratio", "radius_ratio", False),
        ("mean_ratio", "shape", False),
        ("mean_ratio", "aspect_frobenius", True),
    ],
)
@pytest.mark.parity("face_quality", "pyvista")
def test_face_quality_against_the_verdict_measures(
    half_torus: tuple[tm.Trimesh, wp.Mesh], metric: str, measure: str, reciprocal: bool
):
    """
    Decode VTK's Verdict measure names onto triwarp's, three Class A and one Class B.

    The mapping is the trap, not the arithmetic, and the two inversions in it will mislead anyone
    reading pyvista's docs instead of this table:

    - Verdict's ``radius_ratio`` is ``R / (2 r_in)``, which is triwarp's **aspect_ratio**;
    - triwarp's own ``radius_ratio`` is its *reciprocal* (asserted below so the inversion is pinned
      rather than described);
    - ``shape`` is ``4 sqrt(3) A / (a^2 + b^2 + c^2)``, triwarp's **mean_ratio**;
    - ``aspect_frobenius`` is one over that -- the Class B row, one named reciprocal -- and
      ``condition`` duplicates it exactly, so it gets no row of its own.

    ``area_max_side`` has no Verdict counterpart at all, and Verdict's ``aspect_ratio``
    (``max_edge / (2 sqrt(3) r_in)``) has no triwarp counterpart; neither is compared.

    Anti-vacuity, which this comparison is unusually exposed to: of ``cell_quality``'s 28 measures
    only 12 are defined on a triangle and the other 16 come back as the constant ``-1.0`` null
    value, while ``distortion`` is a constant ``1.0`` on ordinary input. So the measure is asserted
    to *vary* across faces before it is compared -- on ``half_torus`` it does, by construction.
    """
    mesh_tm, mesh_wp = half_torus
    quality_pv = np.asarray(trimesh_to_pyvista(mesh_tm).cell_quality(measure).cell_data[measure])
    # Neither a null (-1.0) nor a constant: both would pass an allclose against a broken port.
    assert quality_pv.min() > 0.0
    assert_nonconstant(quality_pv, tol=1e-3)

    quality_wp = tw.triangles.face_quality(mesh_wp.points, mesh_wp.indices, metric=metric)
    expected_pv = 1.0 / quality_pv if reciprocal else quality_pv
    assert np.allclose(quality_wp.numpy(), expected_pv, rtol=1e-5, atol=1e-5)

    if measure == "radius_ratio":  # triwarp's like-named metric is the other way up
        radius_ratio_wp = tw.triangles.face_quality(
            mesh_wp.points, mesh_wp.indices, metric="radius_ratio"
        )
        assert np.allclose(radius_ratio_wp.numpy() * quality_pv, 1.0, rtol=1e-4, atol=1e-4)


@pytest.mark.parity("face_angles", "trimesh", "igl")
def test_face_angles(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on both references: ``(n_faces, 3)`` interior angles, element-wise, no transform.

    All three libraries index the angle by the corner it sits at -- column ``j`` is the angle at
    ``faces[f, j]`` -- so ``igl.internal_angles``, ``trimesh``'s ``face_angles`` property and
    ``triangles.face_angles`` are directly comparable. That is worth pinning rather than assuming:
    the natural alternative convention indexes an angle by the *edge* opposite it, which is a
    cyclic shift of the row and would still pass a per-face sum check.

    ``half_torus`` rather than ``icosahedron`` because every corner of an icosahedron's faces
    carries the same angle, which is exactly the fixture a shifted row would survive.
    """
    mesh_tm, mesh_wp = half_torus
    angles_igl = igl.internal_angles(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), faces_igl(mesh_tm)
    )

    angles_wp = tw.triangles.face_angles(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(angles_wp.numpy(), mesh_tm.face_angles, rtol=1e-5, atol=1e-5)
    assert np.allclose(angles_wp.numpy(), angles_igl, rtol=1e-5, atol=1e-5)
    # The row is not merely a permutation: the angle in column j sits at vertex faces[f, j].
    assert np.allclose(angles_wp.numpy().sum(axis=1), np.pi, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("face_angles", "pyvista")
def test_face_angles_extremes_against_pyvista(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class B: VTK gives the two *extremes* only, in **degrees**.

    ``cell_quality('min_angle')`` and ``('max_angle')`` are the same corner angles this module
    returns, reduced per face and converted -- so the named transform is ``np.degrees`` plus a
    min/max along the row, and the comparison cannot see the third angle or which corner each
    belongs to. That is what keeps the igl / trimesh row above the element-wise oracle.
    """
    mesh_tm, mesh_wp = half_torus
    quality_pv = trimesh_to_pyvista(mesh_tm).cell_quality(["min_angle", "max_angle"])

    angles_np = np.degrees(tw.triangles.face_angles(mesh_wp.points, mesh_wp.indices).numpy())

    assert np.allclose(
        angles_np.min(axis=1), np.asarray(quality_pv.cell_data["min_angle"]), rtol=1e-5, atol=1e-4
    )
    assert np.allclose(
        angles_np.max(axis=1), np.asarray(quality_pv.cell_data["max_angle"]), rtol=1e-5, atol=1e-4
    )
    # Non-vacuous: on a mesh of congruent equilateral faces both columns would read 60 everywhere.
    assert_nonconstant(np.asarray(quality_pv.cell_data["min_angle"]), tol=1.0)


@pytest.mark.parity("face_normals_and_areas", "meshlib")
@pytest.mark.parity(
    "face_centroids",
    "meshlib",
    benchmarked=False,
    reason="MeshLib's triCenter is per *face*, so a batched row would be a Python loop over the "
    "face buffer and would time the loop rather than MeshLib -- 49-67x the batched cost where a "
    "batched form exists at all (section 6). It is a sound correctness oracle at fixture size, "
    "which is what this test uses it as. igl and pyvista carry the timed rows for this group.",
)
@pytest.mark.parity(
    "face_quality",
    "meshlib",
    benchmarked=False,
    reason="triangleAspectRatio is per *face*, same as triCenter above: batching it means a Python "
    "loop, and that row would price the loop. It is the strongest correctness oracle this group "
    "has -- exactly 0.0 difference against metric='aspect_ratio', where pyvista names the same "
    "quantity radius_ratio and igl gives it only as a ratio of two other arrays -- so the "
    "comparison belongs here and the timing stays with igl / pymeshlab / pyvista.",
)
@pytest.mark.parity(
    "face_angles",
    "meshlib",
    benchmarked=False,
    reason="MeshLib has no per-face angle table: mm.angle is a two-vector primitive and sumAngles "
    "is per *vertex*, so both are looped on the reference side and a row here would price the loop "
    "rather than MeshLib -- the per-element rule from section 6. trimesh, igl and pyvista carry "
    "the timed rows. What sumAngles adds is a second, independent route to the same numbers: the "
    "per-vertex sum of the table, which is the quantity vertex_defects is built from.",
)
def test_per_face_quantities_match_meshlib(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A for three per-face families at once, all element-wise in face order.

    MeshLib is the fourth independent implementation of these, and it is the one that pins two
    conventions the others leave open. Its ``computePerFaceNormals`` is **normalized** (|n| = 1 to
    6e-08 measured), unlike pymeshlab's ``face_normal_matrix()``, which is the raw cross product at
    magnitude ``2 * area``; and its ``triangleAspectRatio`` is exactly the measure
    [`face_quality`][triwarp.triangles.face_quality] calls ``aspect_ratio`` -- **0.0** difference,
    where pyvista names the same quantity ``radius_ratio`` and igl gives it only as a ratio of two
    other arrays.

    One named transform, the same one ``igl.doublearea`` needs: ``dblArea`` is twice the area.
    Everything else is direct. Three of MeshLib's four entry points here are **per-face** rather
    than batched, so they are looped on the reference side -- fine in a test at this size, and the
    reason ``benchmarks/`` reads those rows as an upper bound (see section 6).

    The fourth family is the corner angles, and MeshLib reaches them two ways: ``mm.angle(a, b)``
    on the two corner vectors (Class B -- the transform is building those vectors, and MeshLib's
    ``atan2(|cross|, dot)`` form is a different formulation from an ``acos`` of the normalized dot,
    which is what makes it worth comparing) and ``sumAngles`` per vertex, which must equal the
    table's per-vertex sum (Class A, and the quantity ``vertex_defects`` subtracts from 2pi).
    Measured on ``half_torus``: **2.38e-07** on the corners and **1.07e-06** on the vertex sums.

    Testing the four together is deliberate: they come out of the same corner load, so a
    fixture-level disagreement (a converter dropping a vertex, a face buffer reshaped wrong) shows
    up in all of them at once and is distinguishable from a real per-quantity bug.
    """
    mesh_tm, mesh_wp = half_torus
    mesh_ml = trimesh_to_meshlib(mesh_tm)
    topology_ml, points_ml = mesh_ml.topology, mesh_ml.points
    n_faces = mesh_wp.indices.size // 3
    assert topology_ml.numValidFaces() == n_faces > 0  # non-vacuity, and the converter's own check

    normals_wp, areas_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)
    normals_ml = mn.toNumpyArray(mm.computePerFaceNormals(mesh_ml))
    assert np.allclose(np.linalg.norm(normals_ml, axis=1), 1.0, atol=1e-6)
    assert np.allclose(normals_wp.numpy(), normals_ml, rtol=1e-5, atol=1e-5)

    faces_ml = [mm.FaceId(f) for f in range(n_faces)]
    areas_ml = np.array([mm.dblArea(topology_ml, points_ml, f) / 2.0 for f in faces_ml])
    assert np.allclose(areas_wp.numpy(), areas_ml, rtol=1e-5, atol=1e-5)

    centroids_ml = np.array([[*mm.triCenter(topology_ml, points_ml, f)] for f in faces_ml])
    centroids_wp = tw.triangles.face_centroids(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(centroids_wp.numpy(), centroids_ml, rtol=1e-5, atol=1e-5)

    aspect_ml = np.array([mm.triangleAspectRatio(topology_ml, points_ml, f) for f in faces_ml])
    aspect_wp = tw.triangles.face_quality(mesh_wp.points, mesh_wp.indices, metric="aspect_ratio")
    assert_nonconstant(aspect_ml, tol=0.1)  # non-vacuity: a constant would pass any tolerance
    assert np.allclose(aspect_wp.numpy(), aspect_ml, rtol=1e-5, atol=1e-5)

    faces_np = np.asarray(mesh_tm.faces)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float32)
    corner_ml = np.array(
        [
            [
                mm.angle(
                    mm.Vector3f(*(vertices_np[row[(k + 1) % 3]] - vertices_np[row[k]]).tolist()),
                    mm.Vector3f(*(vertices_np[row[(k + 2) % 3]] - vertices_np[row[k]]).tolist()),
                )
                for k in range(3)
            ]
            for row in faces_np
        ]
    )
    angles_wp = tw.triangles.face_angles(mesh_wp.points, mesh_wp.indices)
    # non-vacuity: equilateral faces would read 60 degrees flat
    assert_nonconstant(corner_ml, tol=0.5)
    assert np.allclose(angles_wp.numpy(), corner_ml, rtol=1e-5, atol=1e-5)

    # The same table read the other way: MeshLib's per-vertex angle sum.
    sums_ml = np.array(
        [mm.sumAngles(topology_ml, points_ml, mm.VertId(v)) for v in range(mesh_wp.points.size)]
    )
    sums_wp = np.bincount(
        faces_np.reshape(-1), weights=angles_wp.numpy().reshape(-1), minlength=mesh_wp.points.size
    )
    assert np.allclose(sums_wp, sums_ml, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("face_quality", "igl")
def test_face_quality_aspect_ratio_against_igl(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """Class B (a derived ratio): igl gives the circumradius and inradius as two arrays."""
    mesh_tm, mesh_wp = half_torus
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)
    aspect_igl = np.asarray(igl.circumradius(vertices_np, faces_np)[0]) / (
        2.0 * np.asarray(igl.inradius(vertices_np, faces_np))
    )

    quality_wp = tw.triangles.face_quality(mesh_wp.points, mesh_wp.indices, metric="aspect_ratio")
    assert np.allclose(quality_wp.numpy(), aspect_igl, rtol=1e-4, atol=1e-5)


def test_face_quality_radius_ratio_equilateral(device: str):
    """An equilateral triangle is the optimum of all three normalized measures."""
    side = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0]], dtype=np.float32
    )
    vertices_wp = points_to_warp(side, device)
    faces_wp = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    assert np.allclose(
        tw.triangles.face_quality(vertices_wp, faces_wp, metric="radius_ratio").numpy(), 1.0
    )
    assert np.allclose(
        tw.triangles.face_quality(vertices_wp, faces_wp, metric="mean_ratio").numpy(), 1.0
    )
    assert np.allclose(
        tw.triangles.face_quality(vertices_wp, faces_wp, metric="area_max_side").numpy(),
        np.sqrt(3.0) / 2.0,
    )
    assert np.allclose(
        tw.triangles.face_quality(vertices_wp, faces_wp, metric="aspect_ratio").numpy(), 1.0
    )


def test_face_quality_degenerate(sliver_patch: tuple[np.ndarray, np.ndarray, wp.array, wp.array]):
    """The sliver reads ``+inf`` under ``aspect_ratio`` and ~0 under the bounded measures."""
    _vertices_np, _faces_np, vertices_wp, faces_wp = sliver_patch
    aspect_wp = tw.triangles.face_quality(vertices_wp, faces_wp, metric="aspect_ratio")
    assert np.isinf(aspect_wp.numpy()[0])
    for metric in ("radius_ratio", "area_max_side", "mean_ratio"):
        quality_wp = tw.triangles.face_quality(vertices_wp, faces_wp, metric=metric)
        assert quality_wp.numpy()[0] < 1e-5


def test_face_quality_unknown_metric(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="unknown metric"):
        tw.triangles.face_quality(mesh_wp.points, mesh_wp.indices, metric="skewness")


@pytest.mark.parametrize("with_degenerate", [False, True], ids=["clean", "with_degenerate"])
def test_face_nondegenerate_mask(hemisphere: tuple[tm.Trimesh, wp.Mesh], with_degenerate: bool):
    """
    Class A on a boolean mask, against ``trimesh.triangles.nondegenerate``.

    Parametrized so the comparison sees **both** answers. On the fixture alone every face is
    nondegenerate, so an implementation returning all-``True`` unconditionally passed -- and the
    docstring here used to say the degenerate branch was "covered separately by the zero-area tests
    in this file", which do not exist. The two asserts below carry that claim now instead of a
    sentence: each parametrization is checked to produce the answer it is named for before the mask
    is compared. Both of trimesh's stated degeneracy causes are present in the second case, an
    exactly collinear triangle and one with a repeated corner.

    The appended vertices are exact powers of two and the mesh is scaled to ~1e-2 deliberately
    (CLAUDE.md section 12.4): both libraries test an *absolute* 1e-8 altitude, and at unit scale FMA
    fusion gives a repeated-corner triangle an area of ~1e-8 on CUDA against exactly 0 on the CPU,
    so an inexactly-collinear face would disagree across devices for a reason that is not the
    code's.
    """
    mesh_tm, mesh_wp = hemisphere
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64) * 1e-2
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1)
    if with_degenerate:
        base = vertices_np.shape[0]
        step = 2.0**-7  # exact in float32, so ``2 * step`` is too and the triple is truly collinear
        vertices_np = np.vstack(
            [vertices_np, np.array([[0.0, 0.0, 0.0], [step, 0.0, 0.0], [2.0 * step, 0.0, 0.0]])]
        )
        faces_np = np.concatenate(
            [faces_np, np.array([base, base + 1, base + 2, base, base + 1, base], dtype=np.int32)]
        )

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, mesh_wp.points.device)
    # Compare on the *uploaded* float32 positions, so the two sides see identical coordinates.
    triangles_np = vertices_wp.numpy()[faces_np.reshape(-1, 3)]
    nondegenerate_tm = tm.triangles.nondegenerate(triangles_np)
    assert np.count_nonzero(~nondegenerate_tm) == (2 if with_degenerate else 0)

    nondegenerate_wp = tw.triangles.face_nondegenerate_mask(vertices_wp, faces_wp)
    assert np.array_equal(nondegenerate_wp.numpy().astype(bool), nondegenerate_tm)


def test_barycentric_to_points(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: barycentric-to-Cartesian against trimesh, on random unnormalized coordinates.

    The coordinates are not normalized to sum to one, which is deliberate: both libraries treat
    them as affine weights, and normalizing would hide a divide the other does not do.
    """
    mesh_tm, mesh_wp = hemisphere
    barycentric_np = np.random.default_rng(31).random((mesh_tm.triangles.shape[0], 3))
    points_tm = tm.triangles.barycentric_to_points(mesh_tm.triangles, barycentric_np)

    barycentric_wp = points_to_warp(barycentric_np, mesh_wp.points.device)
    points_wp = tw.triangles.barycentric_to_points(mesh_wp.points, mesh_wp.indices, barycentric_wp)
    assert np.allclose(points_wp.numpy(), points_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["cramer", "cross"])
@pytest.mark.parity("points_to_barycentric", "trimesh", "igl")
def test_points_to_barycentric(hemisphere: tuple[tm.Trimesh, wp.Mesh], method: str):
    """
    Class A on both references, against *both* of trimesh's solver methods.

    triwarp exposes one formulation where trimesh exposes two, so the parametrize now varies the
    **reference** rather than the code under test -- which is the stronger comparison, since it
    asserts the single triwarp answer against both of the alternatives a caller might have expected.
    They agree here because this fixture's triangles are well shaped; the formulations part company
    on slivers, which the sweep below covers.

    ``igl.barycentric_coordinates(P, A, B, C)`` takes the three corner arrays rather than a mesh, so
    the only difference from a call convention standpoint is that the triangle soup is unpacked into
    three ``(n, 3)`` blocks -- the values are compared with no transform at all.

    The points are generated *from* barycentric weights, so each one lies exactly in its triangle's
    plane and the coordinates are the ones that produced it; a solver that got the plane projection
    wrong rather than the in-plane solve would still pass a "sums to one" check, which is why the
    comparison is against two independent solvers instead.
    """
    mesh_tm, mesh_wp = hemisphere

    barycentric_np = np.random.default_rng(32).random((mesh_tm.triangles.shape[0], 3))
    points_np = tm.triangles.barycentric_to_points(mesh_tm.triangles, barycentric_np)
    barycentric_tm = tm.triangles.points_to_barycentric(mesh_tm.triangles, points_np, method=method)
    barycentric_igl = igl.barycentric_coordinates(
        np.ascontiguousarray(points_np),
        np.ascontiguousarray(mesh_tm.triangles[:, 0]),
        np.ascontiguousarray(mesh_tm.triangles[:, 1]),
        np.ascontiguousarray(mesh_tm.triangles[:, 2]),
    )

    points_wp = points_to_warp(points_np, mesh_wp.points.device)
    barycentric_wp = tw.triangles.points_to_barycentric(mesh_wp.points, mesh_wp.indices, points_wp)
    assert np.allclose(barycentric_wp.numpy(), barycentric_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(barycentric_wp.numpy(), barycentric_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("height", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
@pytest.mark.parity("points_to_barycentric", "trimesh")
def test_points_to_barycentric_survives_slivers(device: str, height: float) -> None:
    """
    Class B: against trimesh evaluated in ``float64``, which is exact over this whole sweep.

    The named transform is the precision: trimesh promotes its input to ``float64`` whatever it is
    handed (measured), so its cross-product formulation is exact over this whole sweep and cannot
    exhibit the failure itself. The question is whether triwarp's ``float32`` solve tracks it, and
    the sweep is the test -- it pins the formulation's accuracy across five decades of degeneracy,
    so a change that reintroduced the Gram-determinant form would fail the lower cells rather than
    a tolerance. ``method="cross"`` is named explicitly on the reference side because trimesh's
    *other* formulation is not exact here either: it disagrees with its own cross-product form by
    5.6e-06 at ``height = 1e-6``, in ``float64``, which is why triwarp ships one formulation and not
    two (the measured table is in ``kernels.triangles.point_barycentric``).

    ``test_points_to_barycentric`` covers the same comparison on well-shaped triangles; this covers
    the class where the two algebraically-equal formulations part company, which is why the package
    ships only one (the measured table is in ``kernels.triangles.point_barycentric``).
    """
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, height, 0.0]], dtype=np.float32)
    faces_np = np.array([0, 1, 2], dtype=np.int32)
    point_np = np.array([[0.4, height * 0.5, 0.0]], dtype=np.float32)

    triangles_np = vertices_np[faces_np].reshape(1, 3, 3)
    barycentric_tm = tm.triangles.points_to_barycentric(triangles_np, point_np, method="cross")
    assert_nonconstant(barycentric_tm[0], 0.1)  # a constant answer would compare vacuously

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    point_wp = points_to_warp(point_np, device)
    barycentric_wp = tw.triangles.points_to_barycentric(vertices_wp, faces_wp, point_wp)
    assert np.allclose(barycentric_wp.numpy(), barycentric_tm, rtol=1e-4, atol=1e-4)


def test_points_to_barycentric_on_a_zero_area_triangle(device: str) -> None:
    """
    Triwarp against triwarp: a triangle with exactly zero area has no barycentric frame.

    Not a library comparison: trimesh and igl both divide by zero here and return a non-finite row,
    so there is no reference answer to compare against -- triwarp deliberately differs, answering
    along the triangle's longest edge, which is not an approximation because a zero-area triangle
    *is* that segment. The invariants are what pin it: the coordinates stay finite, sum to one, and
    reconstruct the query point when it lies on the segment.
    """
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=np.float32)
    faces_np = np.array([0, 1, 2], dtype=np.int32)
    point_np = np.array([[0.4, 0.0, 0.0]], dtype=np.float32)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    point_wp = points_to_warp(point_np, device)

    barycentric_np = tw.triangles.points_to_barycentric(vertices_wp, faces_wp, point_wp).numpy()[0]
    assert np.all(np.isfinite(barycentric_np))
    assert np.isclose(barycentric_np.sum(), 1.0, rtol=1e-5, atol=1e-5)
    rebuilt_np = barycentric_np @ vertices_np.astype(np.float64)
    assert np.allclose(rebuilt_np, point_np[0], rtol=1e-5, atol=1e-5)

    # The reference the package diverges from, asserted so the divergence stays deliberate. The
    # errstate is not defensive: dividing by the zero area is exactly what is being demonstrated,
    # so its warning is the expected result rather than a signal about the input (section 7.7).
    triangles_np = vertices_np[faces_np].reshape(1, 3, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        barycentric_tm = tm.triangles.points_to_barycentric(triangles_np, point_np)
    assert not np.all(np.isfinite(barycentric_tm))


def test_closest_point(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the per-triangle closest point against ``trimesh.triangles.closest_point``.

    Unlike the whole-mesh query in ``test_proximity``, each query is matched to *its own*
    triangle, so there is no tie between faces and the point itself is comparable rather than
    only its distance.
    """
    mesh_tm, mesh_wp = hemisphere
    points_np = np.random.default_rng(33).random((mesh_tm.triangles.shape[0], 3))
    closest_points_tm = tm.triangles.closest_point(mesh_tm.triangles, points_np)

    points_wp = points_to_warp(points_np, mesh_wp.points.device)
    closest_points_wp = tw.triangles.closest_point(mesh_wp.points, mesh_wp.indices, points_wp)
    assert np.allclose(closest_points_wp.numpy(), closest_points_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parity(
    "triangle_closest_point",
    "meshlib",
    benchmarked=False,
    reason="the soup form of closest_point has no benchmark group: the whole-mesh query is what "
    "costs anything and it is timed as closest_point_on_mesh, where scipy and open3d carry the "
    "rows. MeshLib's closestPointInTriangle is a four-vector primitive anyway, so a row would time "
    "a Python loop over the face buffer -- the per-element rule from section 6.",
)
@pytest.mark.parity(
    "barycentric_to_points",
    "meshlib",
    benchmarked=False,
    reason="barycentric_to_points has no benchmark group of its own: it is the inverse of "
    "points_to_barycentric, which carries the group, and one interpolation under two names would "
    "price the same pass twice. MeshLib's triPoint is per-point besides, so the row would time the "
    "loop.",
)
def test_soup_quantities_match_meshlib(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on the projection and Class B on the interpolation, for the module's two soup ops.

    ``closestPointInTriangle(p, a, b, c)`` is the same query triangle by triangle and needs no
    transform -- measured **1.19e-07**, the float32 floor. It returns a ``(point, TriPointf)`` pair
    and the point is the half compared here; its barycentric half is the *other* function's answer
    and is checked through ``triPoint`` below rather than decoded twice.

    ``triPoint`` needs one named transform, and it is not guessable: MeshLib addresses a point in a
    face by an **edge**, with the weights running ``(1 - a - b, a, b)`` over
    ``(org(e), dest(e), dest(next(e)))``. So the barycentric triple has to be permuted into
    MeshLib's own corner order for the face's representative edge, which the loop does by looking up
    each vertex id rather than assuming the rotation -- getting it wrong yields a point inside the
    right triangle, which is exactly the failure a loose tolerance would hide. Measured
    **1.19e-07** once the permutation is right.

    The weights are normalized here, unlike [`test_barycentric_to_points`], because a
    ``MeshTriPoint`` whose coordinates do not sum to one is outside the face and MeshLib is entitled
    to a different answer; the affine-weights claim stays with the trimesh oracle.
    """
    mesh_tm, mesh_wp = hemisphere
    faces_np = np.asarray(mesh_tm.faces)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float32)
    n_faces = faces_np.shape[0]
    rng = np.random.default_rng(34)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    topology_ml, points_ml = mesh_ml.topology, mesh_ml.points
    assert topology_ml.numValidFaces() == n_faces > 0  # non-vacuity, and the converter's own check

    # closest_point: one query per triangle, deliberately off the surface so the projection bites.
    queries_np = np.ascontiguousarray(
        mesh_tm.triangles.mean(axis=1) + rng.standard_normal((n_faces, 3)) * 0.35, dtype=np.float32
    )
    queries_wp = points_to_warp(queries_np, mesh_wp.points.device)
    closest_wp = tw.triangles.closest_point(mesh_wp.points, mesh_wp.indices, queries_wp)
    closest_ml = np.array(
        [
            [
                *mm.closestPointInTriangle(
                    mm.Vector3f(*queries_np[f].tolist()),
                    mm.Vector3f(*vertices_np[faces_np[f, 0]].tolist()),
                    mm.Vector3f(*vertices_np[faces_np[f, 1]].tolist()),
                    mm.Vector3f(*vertices_np[faces_np[f, 2]].tolist()),
                )[0]
            ]
            for f in range(n_faces)
        ]
    )
    # Non-vacuity: the queries must really be off the triangles, or this compares two copies of p.
    assert np.linalg.norm(closest_ml - queries_np, axis=1).max() > 0.1
    assert np.allclose(closest_wp.numpy(), closest_ml, rtol=1e-5, atol=1e-5)

    # barycentric_to_points: MeshLib's edge-relative convention, permuted per face.
    barycentric_np = rng.random((n_faces, 3)) + 0.05
    barycentric_np /= barycentric_np.sum(axis=1, keepdims=True)
    barycentric_wp = points_to_warp(barycentric_np, mesh_wp.points.device)
    points_wp = tw.triangles.barycentric_to_points(mesh_wp.points, mesh_wp.indices, barycentric_wp)
    interpolated_ml = np.empty((n_faces, 3))
    for f in range(n_faces):
        edge_ml = topology_ml.edgeWithLeft(mm.FaceId(f))
        corners_ml = [int(v) for v in topology_ml.getLeftTriVerts(edge_ml)]
        weights = {int(v): float(w) for v, w in zip(faces_np[f], barycentric_np[f], strict=True)}
        interpolated_ml[f] = [
            *mm.triPoint(
                topology_ml,
                points_ml,
                mm.MeshTriPoint(
                    edge_ml, mm.TriPointf(weights[corners_ml[1]], weights[corners_ml[2]])
                ),
            )
        ]
    assert np.allclose(points_wp.numpy(), interpolated_ml, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("face_centroids", "igl", "pyvista")
def test_face_centroids(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A on both references: one barycentre per face, element-wise, no transform.

    ``igl.barycenter`` and VTK's ``cell_centers`` both return the corner mean in face order — the
    latter is a *parametric* centre in general, but on a triangle that is the barycentre, which is
    why the row is Class A rather than a documented approximation.

    The assert that matters beyond the comparison is the last one: the barycentre must lie *in* its
    own triangle, which the barycentric coordinates ``(1/3, 1/3, 1/3)`` state exactly. A function
    that returned the mesh centroid broadcast, or the first corner, would match neither.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    centroids_igl = igl.barycenter(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), faces_igl(mesh_tm)
    )
    centroids_pv = np.asarray(trimesh_to_pyvista(mesh_tm).cell_centers().points)

    centroids_wp = tw.triangles.face_centroids(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(centroids_wp.numpy(), centroids_igl, rtol=1e-5, atol=1e-5)
    assert np.allclose(centroids_wp.numpy(), centroids_pv, rtol=1e-5, atol=1e-5)
    barycentric_wp = tw.triangles.points_to_barycentric(
        mesh_wp.points, mesh_wp.indices, centroids_wp
    )
    assert np.allclose(barycentric_wp.numpy(), 1.0 / 3.0, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "half_torus"])
def test_face_signed_volumes(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A against the NumPy oracle, element-wise, plus Class B on the sum.

    ``dot(v0, cross(v1, v2)) / 6`` per face, which is a one-liner in NumPy and therefore an exact
    oracle rather than an approximate one. The class-B half is that the sum over a closed mesh is
    ``mesh_tm.volume`` -- the transform being the reduction -- and it is the assert that would catch
    a per-face sign error the element-wise comparison could only catch if the oracle had the same
    bug. ``half_torus`` is open, so it is here for the element-wise half only, which is exactly the
    point: the per-face quantity is defined whether or not the surface bounds anything.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    triangles_np = np.asarray(mesh_tm.triangles, dtype=np.float64)
    v0, v1, v2 = triangles_np[:, 0], triangles_np[:, 1], triangles_np[:, 2]
    volumes_np = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0

    volumes_wp = tw.triangles.face_signed_volumes(mesh_wp.points, mesh_wp.indices)

    scale = float(np.abs(volumes_np).max())
    assert np.allclose(volumes_wp.numpy(), volumes_np, rtol=1e-5, atol=1e-5 * scale)
    if mesh_tm.is_watertight:
        assert np.isclose(volumes_wp.numpy().sum(), mesh_tm.volume, rtol=1e-5, atol=1e-5)


def test_face_signed_volumes_apex_shifts_each_face_but_not_the_sum(
    device: str, icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]
):
    """
    Moving the apex changes every per-face volume and leaves the closed-mesh total alone.

    Both halves are needed. The invariance of the sum is the property ``measures.volume`` relies on
    when it passes no apex at all; the *variance* of the individual entries is what makes ``apex``
    a real parameter rather than a decoration, and it is the axis ``sample.sample_volume`` uses --
    it fans from the surface centroid precisely so that no entry comes out negative.
    """
    mesh_tm, _mesh_tm_wp = icosphere_coarse
    vertices_wp = points_to_warp(mesh_tm.vertices, device)
    faces_wp = wp.array(
        np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )

    at_origin_np = tw.triangles.face_signed_volumes(vertices_wp, faces_wp).numpy()
    shifted_np = tw.triangles.face_signed_volumes(
        vertices_wp, faces_wp, wp.vec3(2.0, -1.0, 0.5)
    ).numpy()

    assert not np.allclose(at_origin_np, shifted_np, atol=1e-4)
    assert np.isclose(at_origin_np.sum(), shifted_np.sum(), rtol=1e-4)


def test_face_signed_volumes_follows_the_input_dtype(
    device: str, icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]
):
    """``vec3d`` in, ``float64`` out -- the axis ``smoothing``'s volume constraint needs."""
    mesh_tm, _mesh_tm_wp = icosphere_coarse
    faces_wp = wp.array(
        np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    vertices_f64_wp = wp.array(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), dtype=wp.vec3d, device=device
    )
    vertices_f32_wp = points_to_warp(mesh_tm.vertices, device)

    volumes_f64_wp = tw.triangles.face_signed_volumes(vertices_f64_wp, faces_wp)
    volumes_f32_wp = tw.triangles.face_signed_volumes(vertices_f32_wp, faces_wp)

    assert volumes_f64_wp.dtype is wp.float64
    assert volumes_f32_wp.dtype is wp.float32
    assert np.allclose(volumes_f64_wp.numpy(), volumes_f32_wp.numpy(), rtol=1e-6, atol=1e-7)


def test_face_signed_volumes_empty(device: str):
    vertices_wp = wp.zeros(1, dtype=wp.vec3, device=device)
    faces_wp = wp.array([], dtype=wp.int32, device=device)
    assert tw.triangles.face_signed_volumes(vertices_wp, faces_wp).shape == (0,)
