"""Regression tests for ``ordito.curvature`` against ``trimesh.curvature`` (CPU reference)."""

import subprocess
import sys
from pathlib import Path
from typing import Literal

import igl
import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn

import ordito as od
import ordito.typing as odt
from tests.comparisons import assert_nonconstant
from tests.conftest import (
    MESHES,
    ROUND_FRAME_ROTATION,
    ROUND_FRAME_SCALE,
    ROUND_FRAME_SHIFT,
    round_frame_coordinates,
)
from tests.conversions import (
    faces_igl,
    numpy_to_warp,
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
)


@pytest.mark.parity("principal_curvature", "igl")
@pytest.mark.parametrize("supplied", ["both", "normals_only"])
def test_principal_curvature_with_supplied_face_tables(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], supplied: str
) -> None:
    """
    Class A: with the face normals and areas handed over, the curvatures are still libigl's.

    The pair is used only when both halves are given; a lone ``face_normals`` is ignored and the
    function derives both, so either way the answer must match ``igl.principal_curvature``.
    Measured 1.3e-06 at worst, both devices.
    """
    mesh_tm, mesh_wp = sphere_irregular
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    _, _, pv1_igl, pv2_igl, _ = igl.principal_curvature(vertices_np, faces_np, useKring=False)

    face_normals_wp, face_areas_wp = od.triangles.face_normals_and_areas(
        mesh_wp.points, mesh_wp.indices
    )
    _, _, pv1_wp, pv2_wp = od.curvature.principal_curvature(
        mesh_wp.points,
        mesh_wp.indices,
        frame_independent=False,
        face_normals=face_normals_wp,
        face_areas=face_areas_wp if supplied == "both" else None,
    )
    assert np.allclose(pv1_wp.numpy(), pv1_igl, atol=1e-5, rtol=1e-5)
    assert np.allclose(pv2_wp.numpy(), pv2_igl, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
def test_principal_curvature_graded(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Values and directions against libigl where curvature varies, on both shape-operator paths.

    **Class A, ``frame_independent=False``**: reproduces ``igl::principal_curvature``'s symmetrized
    shape operator, so values and directions match at the standard tolerance. Measured, both
    devices: values to **7.1e-06** absolute (3.3e-07 relative), directions to ``|cos| = 1.000000``
    at every vertex. The only freedom is the eigenvector's sign, so directions compare by
    ``|dot|``; that is the convention, not a transform.

    **Class C (a bound), ``frame_independent=True``**: the default solves the true generalized
    eigenproblem (a surface invariant) rather than libigl's frame-dependent symmetrized operator.
    The two share the trace, so the mean curvature ``(PV1 + PV2) / 2`` agrees at the standard
    tolerance (measured **3.7e-06**). The values themselves differ only where the fitted frame is
    anisotropic: on this fixture by **1.13e-04** at worst, bounded at 3x that. Not a sharp test
    of the formulation: here the two operators nearly coincide (on ``sphere_irregular`` they part
    by 0.33, which is why that fixture runs only the frame-dependent comparison above); it pins
    that the default stays the same answer wherever the frame does not matter.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    pd1_igl, pd2_igl, pv1_igl, pv2_igl, bad_igl = map(
        np.asarray, igl.principal_curvature(vertices_np, faces_np, useKring=False)
    )

    # Exclude vertices igl marked bad (degenerate) and umbilics where PV1 ~ PV2 (dirs undefined)
    bad = np.array(bad_igl, dtype=np.int32)
    gap = np.abs(np.asarray(pv1_igl) - np.asarray(pv2_igl))
    mask = np.ones(len(pv1_igl), dtype=bool)
    if len(bad) > 0:
        mask[bad] = False
    assert mask.sum() > 0.9 * mask.size  # non-vacuity: the comparison covers the mesh
    directional = mask & (gap >= 1e-2)

    pd1_wp, pd2_wp, pv1_wp, pv2_wp = od.curvature.principal_curvature(
        mesh_wp.points, mesh_wp.indices, frame_independent=False
    )
    assert np.allclose(pv1_wp.numpy()[mask], pv1_igl[mask], atol=1e-5, rtol=1e-5)
    assert np.allclose(pv2_wp.numpy()[mask], pv2_igl[mask], atol=1e-5, rtol=1e-5)
    pd1_dot = np.abs(np.einsum("ij,ij->i", pd1_wp.numpy()[directional], pd1_igl[directional]))
    pd2_dot = np.abs(np.einsum("ij,ij->i", pd2_wp.numpy()[directional], pd2_igl[directional]))
    assert np.allclose(pd1_dot, 1.0, rtol=0.0, atol=1e-5)
    assert np.allclose(pd2_dot, 1.0, rtol=0.0, atol=1e-5)

    # Default (frame_independent=True): true Weingarten map, independent of the tangent frame.
    _, _, pv1_wp, pv2_wp = od.curvature.principal_curvature(mesh_wp.points, mesh_wp.indices)
    pv1_indep = pv1_wp.numpy()
    pv2_indep = pv2_wp.numpy()

    # Mean curvature (the shared trace invariant) must match libigl at the standard tolerance.
    mean_indep = 0.5 * (pv1_indep + pv2_indep)
    mean_igl = 0.5 * (np.asarray(pv1_igl) + np.asarray(pv2_igl))
    assert np.allclose(mean_indep[mask], mean_igl[mask], atol=1e-5, rtol=1e-5)
    assert np.abs(pv1_indep - pv1_igl)[mask].max() < 3.4e-4
    assert np.abs(pv2_indep - pv2_igl)[mask].max() < 3.4e-4


@pytest.mark.parity("principal_curvature", "pymeshlab")
def test_principal_curvature_directions_match_pymeshlab(
    torus_round: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class B on the principal *directions*, which are the only quantity MeshLab exposes comparably.

    ``compute_curvature_principal_directions_per_vertex`` writes two direction matrices of **unit
    vectors**, so the curvature magnitudes are not in them; its scalar output is a mean curvature
    over a neighbourhood MeshLab derives itself, not a value oracle. The transform is the usual
    eigenvector sign freedom: the comparison is ``|dot| == 1``.

    **Fixture choice is the substance here.** Principal directions are only defined where the two
    principal curvatures differ, and checking them needs a surface whose answer is known:
    ``torus_round`` (an exact torus of revolution, irregularly sampled) has its meridian as the
    first direction at every vertex (curvature 2.5 against at most 0.71 along the parallel).

    **The method is load-bearing.** MeshLab's plain ``"Quadric Fitting"`` (the filter's default) is
    not an oracle on an irregular sampling: its first direction is within 0.99 of the meridian at
    only 56 % of ``torus_round``'s vertices and about 45 degrees from both lines of curvature at
    the worst (``|cos|`` 0.708 against the better line; Taubin, PCA and normal cycles likewise reach
    0.707-0.726). ``"Scale Dependent Quadric Fitting"`` is: worst 0.9927 against the meridian, all
    400 vertices. ordito's ``PD1`` is the meridian at every vertex too (worst 0.9625, the fit's
    accuracy, matched by libigl's own fit; ``test_principal_directions_match_the_analytic_torus``).

    **Measured, and the mutation probes.** Against that method ordito's first direction reads worst
    **0.9468**, median 0.9967, and the second worst 0.9855, in both modes. The bars are 0.84 and
    0.95 (3x the measured deviation from 1) plus a median above 0.99. Pairing ordito's first
    direction with MeshLab's *second* gives a mean of 0.035 and no vertex above 0.88; an arbitrary
    tangent vector (normal cross the z axis) a mean of 0.65 and 24 % of vertices above 0.88. So
    neither an axis swap nor "return any tangent vector" survives.

    ``autoclean=False`` is load-bearing: the filter defaults to deleting unreferenced vertices,
    which would silently renumber the output against ordito's.
    """
    mesh_tm, mesh_wp = torus_round

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_curvature_principal_directions_per_vertex(
        method="Scale Dependent Quadric Fitting", autoclean=False
    )
    assert meshset_pml.current_mesh().vertex_number() == mesh_tm.vertices.shape[0]
    first_pml = np.asarray(
        meshset_pml.current_mesh().vertex_curvature_principal_dir1_matrix(), dtype=np.float64
    )
    second_pml = np.asarray(
        meshset_pml.current_mesh().vertex_curvature_principal_dir2_matrix(), dtype=np.float64
    )
    first_pml /= np.linalg.norm(first_pml, axis=1, keepdims=True)
    second_pml /= np.linalg.norm(second_pml, axis=1, keepdims=True)

    first_wp, second_wp, pv1_wp, pv2_wp = od.curvature.principal_curvature(
        mesh_wp.points, mesh_wp.indices
    )
    # The fixture must actually have distinct principal curvatures, or the directions are arbitrary:
    # the fitted gap is at least 0.31 (true minimum 1.79 / 1.7 = 1.05 in the placed frame).
    assert np.abs(pv1_wp.numpy() - pv2_wp.numpy()).min() > 0.1

    first_np = np.abs(np.einsum("ij,ij->i", first_pml, first_wp.numpy()))
    second_np = np.abs(np.einsum("ij,ij->i", second_pml, second_wp.numpy()))
    assert first_np.min() > 0.84, f"worst |dot| {first_np.min():.4f}"
    assert np.median(first_np) > 0.99
    assert second_np.min() > 0.95, f"worst |dot| {second_np.min():.4f}"


@pytest.mark.parametrize("mesh_name", [*MESHES, "torus_irregular_holes"])
@pytest.mark.parity("principal_curvature_pyvista", "pyvista")
def test_principal_curvature_pyvista_method_matches_pyvista(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: ``method="pyvista"`` is ``curvature("maximum")`` / ``("minimum")``, vertex for vertex.

    VTK's extremes are ``H +- sqrt(H^2 - K)`` over its 1-ring ``K`` (the angle defect over the
    lumped area) and ``H`` (each interior edge adds ``3 |e| theta / (A_f + A_g)`` to both ends,
    ``theta`` the dihedral signed by ``(n_f x n_g) . e``, half the mean over the vertex's interior
    edges), ``H`` alone where ``H^2 < K``. Formed in ``float64``, so the graded saddle's needles
    (``K`` up to 9e5) hold too: measured 2.1e-07 of the field's largest magnitude on
    ``saddle_graded``, 6e-08 on the others, both devices; the tolerance is ``1e-5`` of that
    magnitude, the field's scale. The directions are zero: VTK forms none.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh_pv = trimesh_to_pyvista(mesh_tm)
    maximum_pv = np.asarray(mesh_pv.curvature("maximum"))
    minimum_pv = np.asarray(mesh_pv.curvature("minimum"))
    pd1_wp, pd2_wp, pv1_wp, pv2_wp = od.curvature.principal_curvature(
        mesh_wp.points, mesh_wp.indices, method="pyvista"
    )

    scale = float(np.abs(maximum_pv).max())
    assert np.ptp(maximum_pv - minimum_pv) > 1e-2
    assert np.allclose(pv1_wp.numpy(), maximum_pv, rtol=1e-5, atol=1e-5 * scale)
    assert np.allclose(pv2_wp.numpy(), minimum_pv, rtol=1e-5, atol=1e-5 * scale)
    assert not pd1_wp.numpy().any()
    assert not pd2_wp.numpy().any()


@pytest.mark.parametrize(
    "mesh_name", ["sphere_irregular", "torus_irregular_holes", "sphere_irregular_band"]
)
@pytest.mark.parity("principal_curvature_meyer", "pymeshlab")
@pytest.mark.parity("vertex_curvatures", "pymeshlab")
def test_principal_curvature_meyer_method_matches_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B (rim vertices dropped): ``method="meyer"`` is MeshLab's ``H +- sqrt(H^2 - K)``.

    MeshLab exposes Meyer et al.'s mean and Gaussian curvature, not principal values, so the named
    transform is forming the pair from its two fields, ``H`` alone where ``H^2 < K``. On a rim the
    sign of MeshLab's ``H`` is its own rounding (see
    [`test_vertex_curvatures_match_pymeshlab`][tests.test_curvature.test_vertex_curvatures_match_pymeshlab]),
    and the sign decides which value is the larger, so rim vertices are left out. Tolerance as
    there, ``1e-5`` of the field's scale; the directions are zero. Ordito against ordito too: the
    pair must be the one [`vertex_curvatures`][ordito.curvature.vertex_curvatures] implies, to the
    ``float32`` rounding of its two fields.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    curvatures_pml: dict[str, np.ndarray] = {}
    for kind in ("Gaussian Curvature", "Mean Curvature"):
        meshset_pml = trimesh_to_pymeshlab(mesh_tm)
        meshset_pml.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=kind)
        curvatures_pml[kind] = meshset_pml.current_mesh().vertex_scalar_array()

    def extremes(mean: np.ndarray, gaussian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spread = np.sqrt(np.maximum(mean * mean - gaussian, 0.0))
        return mean + spread, mean - spread

    pv1_pml, pv2_pml = extremes(
        curvatures_pml["Mean Curvature"], curvatures_pml["Gaussian Curvature"]
    )
    pd1_wp, pd2_wp, pv1_wp, pv2_wp = od.curvature.principal_curvature(
        mesh_wp.points, mesh_wp.indices, method="meyer"
    )
    mean_wp, gaussian_wp = od.curvature.vertex_curvatures(mesh_wp.points, mesh_wp.indices)
    pv1_np, pv2_np = extremes(
        mean_wp.numpy().astype(np.float64), gaussian_wp.numpy().astype(np.float64)
    )

    interior = ~_rim_vertex_mask(mesh_tm)
    scale = float(np.abs(pv1_pml).max())
    assert np.ptp(pv1_pml - pv2_pml) > 1e-2
    assert np.allclose(pv1_wp.numpy()[interior], pv1_pml[interior], rtol=1e-5, atol=1e-5 * scale)
    assert np.allclose(pv2_wp.numpy()[interior], pv2_pml[interior], rtol=1e-5, atol=1e-5 * scale)
    assert np.allclose(pv1_wp.numpy(), pv1_np, rtol=1e-5, atol=1e-5 * scale)
    assert np.allclose(pv2_wp.numpy(), pv2_np, rtol=1e-5, atol=1e-5 * scale)
    assert not pd1_wp.numpy().any()
    assert not pd2_wp.numpy().any()


@pytest.mark.parity("principal_curvature_taubin", "pymeshlab")
def test_principal_curvature_taubin_method_matches_pymeshlab(device: str) -> None:
    """
    Class B (rim vertices dropped, unordered pair): ``method="taubin"`` is MeshLab's Taubin.

    vcg's ``UpdateCurvature::PrincipalDirections`` forms Taubin's tensor with the same normal
    curvatures, tangent projections and angle-weighted normal, and its Givens rotation diagonalizes
    it exactly, but it hands back ``K1`` / ``K2`` in no order and weights each edge by the wrong
    pair of faces: its ring walk adds the face *before* an edge's, not the one after, and at the
    first entry of the walk the index ``(i - 1) % size`` wraps through ``size_t`` to an arbitrary
    face. Ported verbatim that weighting reproduces it to 2e-7 of the field on ``sphere_irregular``,
    ``torus_irregular`` and ``saddle_graded``; the paper's weights (ordito's) leave 1-2 % in the
    median there. So the comparison runs on a mesh where both weightings are one: a sheared lattice
    on a cylinder, every face congruent, so every face in a star has the same area. The shear
    puts the lattice's edges obliquely to the lines of curvature, so the tensor is not diagonal in
    any edge's frame, and the fixture is not degenerate (``PV1 - PV2`` is 2.57). Rim vertices are
    dropped (vcg weighs a rim edge by its face over the ring's area, not half of it).

    The transforms: the pair sorted, and MeshLab's first direction taken from whichever of its two
    is the larger value's. Measured 3.3e-6 against values of 2.06 and -0.50, and ``|cos|`` above
    0.9999999; tolerance as elsewhere, ``1e-5`` of the field's scale and ``|cos| > 0.9999``.
    """
    vertices_np, faces_np = _sheared_cylinder_arrays()
    mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
    assert np.ptp(mesh_tm.area_faces) < 1e-5 * mesh_tm.area_faces.max()
    first_pml, second_pml, directions_pml = _pymeshlab_principal_curvature(
        mesh_tm, "Taubin approximation"
    )
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    pd1_wp, _, pv1_wp, pv2_wp = od.curvature.principal_curvature(
        vertices_wp, faces_wp, method="taubin"
    )

    interior = ~_rim_vertex_mask(mesh_tm)
    larger_pml = np.maximum(first_pml, second_pml)
    smaller_pml = np.minimum(first_pml, second_pml)
    scale = float(np.abs(larger_pml).max())
    assert (larger_pml - smaller_pml)[interior].min() > 1.0
    assert np.allclose(pv1_wp.numpy()[interior], larger_pml[interior], atol=1e-5 * scale)
    assert np.allclose(pv2_wp.numpy()[interior], smaller_pml[interior], atol=1e-5 * scale)
    direction_pml = np.where((first_pml >= second_pml)[:, None], *directions_pml)
    alignment_np = np.abs(np.einsum("ij,ij->i", direction_pml, pd1_wp.numpy()))
    assert alignment_np[interior].min() > 0.9999


def _sheared_cylinder_arrays(
    sections: int = 24, rows: int = 10, shear: float = 0.4
) -> tuple[np.ndarray, np.ndarray]:
    """Tile a unit cylinder with one triangle shape: a lattice whose rows turn ``shear`` a step."""
    row_np, column_np = np.meshgrid(np.arange(rows), np.arange(sections), indexing="ij")
    angle_np = (column_np + shear * row_np) * (2.0 * np.pi / sections)
    vertices_np = np.stack(
        [np.cos(angle_np), np.sin(angle_np), 0.12 * row_np.astype(np.float64)], axis=-1
    ).reshape(-1, 3)
    low_np = row_np[:-1].ravel() * sections + column_np[:-1].ravel()
    right_np = row_np[:-1].ravel() * sections + (column_np[:-1].ravel() + 1) % sections
    faces_np = np.concatenate(
        [
            np.stack([low_np, right_np, right_np + sections], axis=1),
            np.stack([low_np, right_np + sections, low_np + sections], axis=1),
        ]
    )
    return vertices_np.astype(np.float32), faces_np.astype(np.int32)


@pytest.mark.parametrize(
    "mesh_name", ["sphere_irregular", "torus_irregular_holes", "saddle_graded"]
)
@pytest.mark.parity("principal_curvature_normal_cycles", "pymeshlab")
def test_principal_curvature_normal_cycles_method_matches_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B (rim and fold vertices dropped): ``method="normal_cycles"`` is MeshLab's normal cycle.

    vcg's ``PrincipalDirectionsNormalCycle`` sums the same ``beta_e |e| e e^T`` over the edges at a
    vertex, with three differences that the transforms undo:

    - it divides by nothing, so its values are ordito's times twice the barycentric cell's area
      (a third of the ring's);
    - it returns the tensor's two *largest* eigenvalues, where ordito returns the two whose
      eigenvectors are not the normal's. They differ where the normal's eigenvalue is not the
      smallest (7 vertices of ``sphere_irregular``, one of ``sphere_well_shaped``), so each pair is
      completed to the whole spectrum with the trace, ``sum_e beta_e |e|`` from trimesh's dihedrals
      signed by its adjacency projections (its ``face_adjacency_convex`` flag is a tolerance test,
      and flips the sign of the saddle's near-flat edges), and the sorted spectra are compared;
    - its first direction is the eigenvector whose eigenvalue is larger in magnitude, the tensor's
      own, where ordito's ``PD1`` is the smaller eigenvalue's (the curvature belongs to the
      direction across its eigenvector), so MeshLab's first direction is matched against ordito's
      ``PD2`` where ``|PV1| >= |PV2|`` and its ``PD1`` elsewhere; and the normal each picks,
      ``PD1 x PD2``, must agree.

    Dropped: rim vertices (vcg's ring walk bounces off a rim and writes zero at a rank-one tensor),
    and vertices on an edge folded past a right angle, where vcg's ``beta_e`` is an arcsine that
    folds the angle back (a 164-degree needle fold reads 16 degrees); ordito takes the ``atan2``.
    That leaves 83-89 % of the vertices. Measured: spectra to 2.9e-08 of the field on the closed
    fixtures and 3.1e-07 on the saddle; directions and normals ``|cos|`` above 0.9999998 (the
    directions where both eigenvalue gaps exceed 5 % of the field: 36-1024 vertices). Probes:
    halving the edge share reads 0.36-0.50 of the field, and taking ``PD1`` as the larger
    eigenvalue's own eigenvector reads ``|cos|`` at most 5e-08.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    first_pml, second_pml, directions_pml = _pymeshlab_principal_curvature(mesh_tm, "Normal Cycles")
    pd1_wp, pd2_wp, pv1_wp, pv2_wp = od.curvature.principal_curvature(
        mesh_wp.points, mesh_wp.indices, method="normal_cycles"
    )
    pv1_np = pv1_wp.numpy().astype(np.float64)
    pv2_np = pv2_wp.numpy().astype(np.float64)

    edges_np = mesh_tm.face_adjacency_edges
    signed_np = -np.sign(mesh_tm.face_adjacency_projections) * mesh_tm.face_adjacency_angles
    lengths_np = np.linalg.norm(
        mesh_tm.vertices[edges_np[:, 0]] - mesh_tm.vertices[edges_np[:, 1]], axis=1
    )
    trace_np = np.zeros(mesh_tm.vertices.shape[0])
    np.add.at(trace_np, edges_np.ravel(), np.repeat(signed_np * lengths_np, 2))
    cell_np = np.zeros(mesh_tm.vertices.shape[0])
    np.add.at(cell_np, mesh_tm.faces.ravel(), np.repeat(mesh_tm.area_faces, 3))
    first_np, second_np = pv1_np * cell_np * (2.0 / 3.0), pv2_np * cell_np * (2.0 / 3.0)
    spectrum_np = np.sort(np.stack([first_np, second_np, trace_np - first_np - second_np]), axis=0)
    spectrum_pml = np.sort(
        np.stack([first_pml, second_pml, trace_np - first_pml - second_pml]), axis=0
    )

    folded = np.zeros(mesh_tm.vertices.shape[0], dtype=bool)
    folded[edges_np[mesh_tm.face_adjacency_angles > np.pi / 2].ravel()] = True
    kept = ~folded & ~_rim_vertex_mask(mesh_tm)
    assert kept.mean() > 0.8
    scale = float(np.abs(spectrum_pml).max())
    assert_nonconstant(first_pml[kept], tol=1e-3 * scale)
    assert np.allclose(spectrum_np[:, kept], spectrum_pml[:, kept], rtol=0.0, atol=1e-5 * scale)

    normal_np = np.cross(pd1_wp.numpy(), pd2_wp.numpy())
    normal_pml = np.cross(*directions_pml)
    assert np.abs(np.einsum("ij,ij->i", normal_np, normal_pml))[kept].min() > 0.9999
    magnitude = float(np.abs(pv1_np).max())
    distinct = (
        kept
        & (pv1_np - pv2_np > 0.05 * magnitude)
        & (np.abs(np.abs(pv1_np) - np.abs(pv2_np)) > 0.05 * magnitude)
    )
    assert distinct.sum() > 30
    direction_np = np.where(
        (np.abs(pv1_np) >= np.abs(pv2_np))[:, None], pd2_wp.numpy(), pd1_wp.numpy()
    )
    alignment_np = np.abs(np.einsum("ij,ij->i", direction_np, directions_pml[0]))
    assert alignment_np[distinct].min() > 0.9999


def _pymeshlab_principal_curvature(
    mesh_tm: tm.Trimesh, method: str
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    MeshLab's per-vertex ``K1``, ``K2`` and unit ``(PD1, PD2)`` from one ``method``.

    The filter exposes ``K1`` and ``K2`` only as its scalar output (``"Min Curvature"`` is ``K1``
    and ``"Max Curvature"`` ``K2``, whatever their order), so it runs once for each.
    ``autoclean=False`` keeps the vertex numbering.
    """
    values_pml: list[np.ndarray] = []
    directions_pml: list[np.ndarray] = []
    for kind in ("Min Curvature", "Max Curvature"):
        meshset_pml = trimesh_to_pymeshlab(mesh_tm)
        meshset_pml.compute_curvature_principal_directions_per_vertex(
            method=method, curvcolormethod=kind, autoclean=False
        )
        mesh_pml = meshset_pml.current_mesh()
        assert mesh_pml.vertex_number() == mesh_tm.vertices.shape[0]
        values_pml.append(np.asarray(mesh_pml.vertex_scalar_array()))
        directions_pml = [
            np.asarray(matrix, dtype=np.float64)
            / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-30)
            for matrix in (
                mesh_pml.vertex_curvature_principal_dir1_matrix(),
                mesh_pml.vertex_curvature_principal_dir2_matrix(),
            )
        ]
    return values_pml[0], values_pml[1], (directions_pml[0], directions_pml[1])


@pytest.mark.parametrize("method", ["taubin", "normal_cycles"])
def test_one_ring_principal_directions_match_the_analytic_torus(
    torus_round: tuple[tm.Trimesh, wp.Mesh], method: Literal["taubin", "normal_cycles"]
) -> None:
    """
    Not a library comparison: the 1-ring tensors against the torus's closed form.

    The pymeshlab comparisons above pin each tensor to MeshLab's, which on its own cannot tell an
    estimator from a consistent mistake in both, and MeshLab orders neither its values nor (for
    Taubin) its directions. A torus of revolution can: its meridians carry the larger curvature,
    ``1 / r``, at every vertex of ``torus_round`` (see
    [`test_principal_directions_match_the_analytic_torus`][tests.test_curvature.test_principal_directions_match_the_analytic_torus]).
    A 1-ring estimate on a dart-thrown sampling is noisy vertex by vertex (MeshLab's own reach
    ``|cos|`` 0.707-0.726 at the worst there), so the claim is statistical: measured median
    ``|cos|`` between ``PD1`` and the meridian 0.990 (Taubin) and 0.977 (normal cycles), 86 % and
    75 % of the vertices above 0.9, and median ``PV1`` 0.1 % and 1.9 % off ``1 / r``. A swapped
    pair reads median 0.14 and 0.06 with 2 % above 0.9, and the unnormalized tensor (MeshLab's
    values) a sixth of ``1 / r``. Bars: median above 0.95, 65 % above 0.9, ``PV1`` within 5 %.
    """
    mesh_tm, mesh_wp = torus_round
    major_radius, minor_radius = 1.0, 0.4
    vertices_np = round_frame_coordinates(mesh_tm.vertices)
    angle_np = np.arctan2(vertices_np[:, 1], vertices_np[:, 0])
    axis_np = np.stack([np.cos(angle_np), np.sin(angle_np), np.zeros_like(angle_np)], axis=1)
    normal_np = vertices_np - major_radius * axis_np
    normal_np /= np.linalg.norm(normal_np, axis=1, keepdims=True)
    parallel_np = np.stack([-np.sin(angle_np), np.cos(angle_np), np.zeros_like(angle_np)], axis=1)
    meridian_np = np.cross(normal_np, parallel_np)

    pd1_wp, _, pv1_wp, _ = od.curvature.principal_curvature(
        mesh_wp.points, mesh_wp.indices, method=method
    )
    alignment_np = np.abs(np.einsum("ij,ij->i", pd1_wp.numpy() @ ROUND_FRAME_ROTATION, meridian_np))
    assert np.median(alignment_np) > 0.95, f"median |cos| {np.median(alignment_np):.4f}"
    assert (alignment_np > 0.9).mean() > 0.65
    meridian_curvature = 1.0 / (minor_radius * ROUND_FRAME_SCALE)
    assert abs(np.median(pv1_wp.numpy()) / meridian_curvature - 1.0) < 0.05


def test_principal_curvature_rejects_an_unknown_method(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """``method`` is validated at the boundary, and the message names the value and the menu."""
    _, mesh_wp = sphere_irregular
    with pytest.raises(ValueError, match=r"method must be .*'normal_cycles'.* got 'pca'"):
        od.curvature.principal_curvature(mesh_wp.points, mesh_wp.indices, method="pca")  # pyright: ignore[reportArgumentType]


def _rim_vertex_mask(mesh_tm: tm.Trimesh) -> np.ndarray:
    """``(n_vertices,)`` whether each vertex lies on a boundary edge."""
    rows, counts = np.unique(mesh_tm.edges_sorted, axis=0, return_counts=True)
    mask = np.zeros(mesh_tm.vertices.shape[0], dtype=bool)
    mask[rows[counts == 1].ravel()] = True
    return mask


# ``saddle_graded`` is not here: MeshLab stores its corner angles and mixed areas in ``float``, and
# over a needle's 1e-6 area that rounding moves ``H`` by 8.5 % of the field and ``K`` by 4.8e-05.
@pytest.mark.parametrize(
    "mesh_name", ["sphere_irregular", "torus_irregular_holes", "sphere_irregular_band"]
)
@pytest.mark.parity("vertex_curvatures", "pymeshlab")
def test_vertex_curvatures_match_pymeshlab(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (a rim vertex's sign of ``H`` dropped): MeshLab's pointwise Meyer curvatures.

    ``compute_scalar_by_discrete_curvature_per_vertex`` is vcg's ``MeanAndGaussian``: the defect
    (minus, on a rim, the angle between the rim edges) over the mixed area, and the cotangent
    Laplacian's length over it, signed by the area-weighted normal. ordito forms the same in
    ``float64``; MeshLab rounds its angles and areas to ``float``, so the tolerance is ``1e-5`` of
    each field's largest magnitude. Measured: off the rims ``K`` to 6.4e-06 and ``H`` to 3.8e-06 of
    the scale; on the rims ``|H|`` to 1.45e-05 (bound 5e-05), where the Laplacian lies almost in
    the tangent plane and its sign against the normal is MeshLab's rounding (1 and 2 rim vertices
    flip on ``torus_irregular_holes`` and an open sphere). The fixtures take a closed surface, three
    rims and a band. Mutation probe: dividing by the barycentric (lumped) area instead moves ``K``
    by far more than the bound on these obtuse-heavy meshes (asserted below).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    curvatures_pml: dict[str, np.ndarray] = {}
    for kind in ("Gaussian Curvature", "Mean Curvature"):
        meshset_pml = trimesh_to_pymeshlab(mesh_tm)
        meshset_pml.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=kind)
        curvatures_pml[kind] = meshset_pml.current_mesh().vertex_scalar_array()
    gaussian_pml = curvatures_pml["Gaussian Curvature"]
    mean_pml = curvatures_pml["Mean Curvature"]
    mean_wp, gaussian_wp = od.curvature.vertex_curvatures(mesh_wp.points, mesh_wp.indices)
    mean_np = mean_wp.numpy()
    gaussian_np = gaussian_wp.numpy()

    rim = _rim_vertex_mask(mesh_tm)
    gaussian_scale = float(np.abs(gaussian_pml).max())
    mean_scale = float(np.abs(mean_pml).max())
    assert_nonconstant(gaussian_pml, tol=1e-1)
    assert np.allclose(gaussian_np, gaussian_pml, rtol=1e-5, atol=1e-5 * gaussian_scale)
    assert np.allclose(mean_np[~rim], mean_pml[~rim], rtol=1e-5, atol=1e-5 * mean_scale)
    assert np.allclose(
        np.abs(mean_np[rim]), np.abs(mean_pml[rim]), rtol=1e-5, atol=5e-5 * mean_scale
    )

    defects_np = od.curvature.vertex_defects(
        mesh_tm.vertices.shape[0],
        mesh_wp.indices,
        od.triangles.face_angles(mesh_wp.points, mesh_wp.indices),
    ).numpy()
    lumped_np = np.zeros(mesh_tm.vertices.shape[0])
    np.add.at(lumped_np, mesh_tm.faces.ravel(), np.repeat(mesh_tm.area_faces / 3.0, 3))
    interior = ~rim
    assert not np.allclose(
        defects_np[interior] / lumped_np[interior],
        gaussian_pml[interior],
        rtol=1e-5,
        atol=1e-5 * gaussian_scale,
    )


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular"])
@pytest.mark.parity("vertex_defects", "trimesh", "igl", "meshlib", "pyvista")
def test_vertex_defects(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on trimesh, igl and MeshLib, Class B on pyvista, on open fixtures and a closed one.

    ``igl.gaussian_curvature`` and MeshLib's ``mn.getNumpyGaussianCurvature`` are both the
    *pointwise* angle defect ``2π - Σθ`` -- the same quantity ``tm.curvature.vertex_defects``
    returns and emphatically **not** the ball-integrated Cohen-Steiner/Morvan measure
    ``curvature.discrete_gaussian_curvature`` computes, which is exempted from parity against
    MeshLab for exactly that reason. Sharing a name with a different measure is the whole hazard
    here, so the comparison is worth having several times over -- and MeshLib is the one to be
    careful with, because it is the reference whose *name* says curvature while its value is the
    defect.

    MeshLib's batched ``mn.getNumpyGaussianCurvature`` is used rather than the per-vertex
    ``mm.discreteGaussianCurvature``; the two are bit-identical (measured 0.0) and the batched form
    is 49-67x faster, which is why its per-vertex entry points are not looped.

    Open and closed fixtures are both needed because the interesting disagreement would be at the
    **boundary**: a reference could reasonably use ``π - Σθ`` there. Measured, none of them does --
    ``half_torus``'s 56 boundary vertices agree element-wise with the closed ``icosahedron``'s
    interior ones -- so the assert is a plain ``allclose`` over every vertex.

    pyvista is Class B: VTK's ``curvature('gaussian')`` is this defect divided by the lumped area.
    ``vtkCurvatures`` returns a *density* -- the angle defect over the barycentric lumped area,
    ``sum of incident face areas / 3`` -- so the named transform is a multiplication by that area,
    which is read off ``compute_cell_sizes`` on the same mesh rather than recomputed. Measured
    element-wise agreement 2.7e-07 / 1.2e-06 / 5.1e-07 on the three fixtures. ``atol`` carries
    that comparison rather than ``rtol``: a flat vertex has zero defect, so a relative tolerance is
    meaningless there. The residual is ordito's ``float32`` vertex buffer and not the reference's
    -- pyvista's curvature comes back float64, and the same comparison against vedo's float32
    points measures the same 5e-05 on a larger mesh.

    The call runs with the CPU as Warp's current device, so on a CUDA run it must still answer on
    its inputs' device: the ordinary tests run with the arrays' device already current, which hid
    a launch that forwarded no ``device=`` in this function once.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    n_vertices = mesh_tm.vertices.shape[0]
    vertex_defects_tm = tm.curvature.vertex_defects(mesh_tm)
    vertex_defects_igl = np.asarray(
        igl.gaussian_curvature(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), faces_igl(mesh_tm)
        )
    ).ravel()
    vertex_defects_ml = mn.getNumpyGaussianCurvature(trimesh_to_meshlib(mesh_tm))

    mesh_pv = trimesh_to_pyvista(mesh_tm)
    gaussian_pv = np.asarray(mesh_pv.curvature("gaussian"))
    areas_pv = np.asarray(
        mesh_pv.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"]
    )
    lumped_pv = np.zeros(n_vertices)
    np.add.at(lumped_pv, mesh_tm.faces.ravel(), np.repeat(areas_pv / 3.0, 3))

    face_angles_wp = _face_angles_wp(mesh_tm, mesh_wp.device)
    with wp.ScopedDevice("cpu"):
        vertex_defects_wp = od.curvature.vertex_defects(n_vertices, mesh_wp.indices, face_angles_wp)
    assert str(vertex_defects_wp.device) == str(mesh_wp.device)

    # Non-vacuity: an implementation returning zeros passes any allclose against another one, and
    # on a nearly-flat patch that is what the true answer looks like. Neither the spread nor the
    # minimum is the check -- an icosahedron is regular so all 12 read pi/3, and half_torus has
    # genuinely near-flat vertices at 4e-05 -- so the claim is that the answer is not all zero.
    assert np.abs(vertex_defects_tm).max() > 1e-2
    assert np.abs(vertex_defects_wp.numpy()).max() > 1e-2
    assert np.allclose(vertex_defects_wp.numpy(), vertex_defects_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(vertex_defects_wp.numpy(), vertex_defects_igl, rtol=1e-5, atol=1e-5)
    assert np.allclose(vertex_defects_wp.numpy(), vertex_defects_ml, rtol=1e-5, atol=1e-5)
    assert np.allclose(vertex_defects_wp.numpy(), gaussian_pv * lumped_pv, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("mesh_name", "chi"), [("sphere_irregular", 2), ("boy_surface", 1), ("bohemian_dome", 0)]
)
def test_vertex_defects_satisfy_gauss_bonnet(
    request: pytest.FixtureRequest, mesh_name: str, chi: int
) -> None:
    """
    Not a library comparison: the defects of a closed mesh sum to ``2π χ``, at any genus.

    No reference: this is the discrete Gauss-Bonnet theorem, and it is a stronger statement about
    the defects than a per-vertex comparison because it couples every vertex at once. The point of
    running it on these three fixtures is the **odd** characteristic: Boy's surface is the only
    input in the suite with χ = 1, so it is the only one that can catch a defect convention that is
    right up to a factor of two, or a sign that is right only on an orientable mesh.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    assert mesh_tm.is_watertight
    assert od.measures.euler_characteristic(mesh_wp.indices) == chi

    defects_wp = od.curvature.vertex_defects(
        mesh_tm.vertices.shape[0], mesh_wp.indices, _face_angles_wp(mesh_tm, mesh_wp.device)
    )
    assert np.isclose(defects_wp.numpy().sum(), 2.0 * np.pi * chi, rtol=1e-4, atol=1e-3)


def test_vertex_defects_are_reproducible_when_deterministic(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a parity assert: under ``ordito.config``'s deterministic mode two calls agree bit for bit.

    The angle sums are formed exactly in ``float64``, so the atomics' commit order changes no bit;
    the off mode's ``float32`` arrival-order sums agree with them to ``float32`` round-off.
    """
    mesh_tm, mesh_wp = sphere_irregular
    face_angles_wp = _face_angles_wp(mesh_tm, mesh_wp.device)
    n_vertices = mesh_tm.vertices.shape[0]
    with od.config.deterministic(True):
        first_np = od.curvature.vertex_defects(n_vertices, mesh_wp.indices, face_angles_wp).numpy()
        second_np = od.curvature.vertex_defects(n_vertices, mesh_wp.indices, face_angles_wp).numpy()
    assert np.array_equal(first_np, second_np)
    with od.config.deterministic(False):
        arrival_np = od.curvature.vertex_defects(
            n_vertices, mesh_wp.indices, face_angles_wp
        ).numpy()
    assert np.allclose(arrival_np, first_np, rtol=1e-5, atol=1e-5)


def _face_angles_wp(mesh_tm: tm.Trimesh, device: wp.DeviceLike) -> odt.Array2dFloat32:
    """Upload trimesh's ``(n_faces, 3)`` corner angles as the ``face_angles`` table ordito takes."""
    return odt.as_array2d(
        wp.array(mesh_tm.face_angles, dtype=wp.float32, device=device), wp.float32
    )


@pytest.mark.parametrize("radius", [0.1, 0.5])
@pytest.mark.parity("discrete_gaussian_curvature", "trimesh")
def test_discrete_gaussian_curvature(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], radius: float
) -> None:
    """
    Class A: the Cohen-Steiner/Morvan ball measure against trimesh's, at the same radius.

    Both sides are fed *trimesh's* face angles, so the comparison isolates the ball integration
    rather than re-testing [`triangles.face_angles`], which has its own oracle. The call runs with
    the CPU as Warp's current device, so on a CUDA run it must still answer on its inputs' device: a
    launch that forwarded no ``device=`` once went unseen, because the ordinary tests run with the
    arrays' device already current (companion to ``test_vertex_defects``).
    """
    mesh_tm, mesh_wp = saddle_graded
    points_tm = mesh_tm.vertices
    gauss_curvature_tm = tm.curvature.discrete_gaussian_curvature_measure(
        mesh_tm, points_tm, radius
    )

    points_wp = points_to_warp(points_tm, mesh_wp.device)
    face_angles_wp = odt.as_array2d(
        wp.array(mesh_tm.face_angles, dtype=wp.float32, device=mesh_wp.device), wp.float32
    )
    with wp.ScopedDevice("cpu"):
        gauss_curvature_wp = od.curvature.discrete_gaussian_curvature(
            points_wp, mesh_wp.points, mesh_wp.indices, face_angles_wp, radius
        )
    assert str(gauss_curvature_wp.device) == str(mesh_wp.device)
    assert_nonconstant(gauss_curvature_tm, tol=1e-3)
    assert np.allclose(gauss_curvature_wp.numpy(), gauss_curvature_tm, rtol=1e-5, atol=1e-5)


def test_discrete_gaussian_curvature_edge_cases(saddle_graded: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: the face-angle table check, and the empty inputs.

    A face-angle table with the wrong row count raises before any launch; no query point gives no
    value, and a mesh with no vertex gives every query a zero measure.
    """
    _, mesh_wp = saddle_graded
    device = mesh_wp.device
    angles_wp = od.triangles.face_angles(mesh_wp.points, mesh_wp.indices)
    points_wp = odt.as_dense(mesh_wp.points[:3])
    with pytest.raises(ValueError, match="face_angles must have one row per triangle"):
        od.curvature.discrete_gaussian_curvature(
            points_wp,
            mesh_wp.points,
            mesh_wp.indices,
            odt.as_array2d(odt.as_dense(angles_wp[:2]), wp.float32),
            0.5,
        )
    no_points_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    no_value = od.curvature.discrete_gaussian_curvature(
        no_points_wp, mesh_wp.points, mesh_wp.indices, angles_wp, 0.5
    )
    assert no_value.numpy().tolist() == []
    no_faces_wp = wp.zeros(0, dtype=wp.int32, device=device)
    no_angles_wp = odt.empty_2d((0, 3), wp.float32, device=device)
    zero = od.curvature.discrete_gaussian_curvature(
        points_wp, wp.zeros(0, dtype=wp.vec3, device=device), no_faces_wp, no_angles_wp, 0.5
    )
    assert np.array_equal(zero.numpy(), np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize(
    ("radius", "supplied"),
    [
        pytest.param(2.0, False, id="2.0"),
        pytest.param(0.5, False, id="0.5"),
        pytest.param(0.5, True, id="0.5-supplied"),
    ],
)
@pytest.mark.parity("discrete_mean_curvature", "trimesh")
def test_discrete_mean_curvature(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], radius: float, supplied: bool
) -> None:
    """
    Class A: the ball mean-curvature measure against trimesh's, over every vertex.

    **Two radii, because they exercise different halves of the ball.** At 0.5 (1.3 mean edges) a
    ball holds a vertex's few rings; at 2.0 it spans half the mesh (extent 4.0-4.3), so every query
    clips a large part of the surface against the ball. Both answers vary per vertex on
    ``sphere_irregular`` -- 500 distinct values each, spread 2.57 (signs mixed) and 7.30 -- so a
    permuted result, an off-by-one in the gather or a query/vertex index swap is visible, and the
    assert below checks that the reference really did vary before comparing to it. (On a regular
    icosahedron the whole-surface radius gave one value repeated, which could see only a global
    scale error.) The ``supplied`` arm hands over a caller's ``face_adjacency`` pair, which must
    give the same measure as the derived one.

    MeshLab's pointwise curvature is a different operator, compared as a statistic in
    ``test_discrete_curvatures_match_pymeshlab_pointwise``.
    """
    mesh_tm, mesh_wp = sphere_irregular
    points_tm = mesh_tm.vertices
    mean_curvature_tm = tm.curvature.discrete_mean_curvature_measure(mesh_tm, points_tm, radius)

    adjacency_wp, adjacency_edges_wp = (
        od.adjacency.face_adjacency(mesh_wp.indices, return_edges=True)
        if supplied
        else (None, None)
    )
    mean_curvature_wp = od.curvature.discrete_mean_curvature(
        points_to_warp(points_tm, mesh_wp.device),
        mesh_wp.points,
        mesh_wp.indices,
        radius,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
    )
    # Non-vacuous on the curved fixture: a constant reference would pass any per-vertex bug.
    assert_nonconstant(mean_curvature_tm, tol=1e-3)
    assert np.allclose(mean_curvature_wp.numpy(), mean_curvature_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("discrete_gaussian_curvature", "pymeshlab")
@pytest.mark.parity("discrete_mean_curvature", "pymeshlab")
@pytest.mark.parametrize("kind", ["gaussian", "mean"])
def test_discrete_curvatures_match_pymeshlab_pointwise(device: str, kind: str) -> None:
    """
    Class C: the ball measure over the ball's area against MeshLab's pointwise curvature.

    MeshLab's ``compute_scalar_by_discrete_curvature_per_vertex`` is the Meyer et al. 1-ring
    curvature per unit area; ordito's (trimesh's) is the Cohen-Steiner/Morvan measure integrated
    over a ball, so the measure divided by ``pi r^2`` estimates the same pointwise value wherever
    the surface is smooth on the ball's scale. On an exact torus (``R = 1``, ``r = 0.4``, 64 x 32
    sections, Gaussian curvature from -4.15 to 1.79) at a radius of two mean edges, the median
    difference is 1.5 % of MeshLab's largest magnitude for both curvatures, both devices; the bound
    is 5 %, 3.3x that. Excluded bug classes: a measure at the wrong vertex or with the wrong sign,
    and a wrong scale. Mutation probes, asserted: MeshLab's values shuffled over the vertices read
    47 % (Gaussian) and 25 % (mean), and the measure over ``2 pi r^2`` (a halved normalization)
    21 % and 38 %. The irregular fixtures are too coarse for the premise: on the 400-point
    ``torus_round`` a mean edge is the minor radius and the margin falls to 3-5x.
    """
    mesh_tm = tm.creation.torus(1.0, 0.4, major_sections=64, minor_sections=32)
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_scalar_by_discrete_curvature_per_vertex(
        curvaturetype="Gaussian Curvature" if kind == "gaussian" else "Mean Curvature"
    )
    curvature_pml = meshset_pml.current_mesh().vertex_scalar_array()
    scale = float(np.abs(curvature_pml).max())
    assert_nonconstant(curvature_pml, tol=1e-1)

    vertices_wp = points_to_warp(mesh_tm.vertices, device)
    faces_wp = wp.array(mesh_tm.faces.reshape(-1).astype(np.int32), dtype=wp.int32, device=device)
    radius = 2.0 * float(mesh_tm.edges_unique_length.mean())
    if kind == "gaussian":
        angles_wp = od.triangles.face_angles(vertices_wp, faces_wp)
        measure_wp = od.curvature.discrete_gaussian_curvature(
            vertices_wp, vertices_wp, faces_wp, angles_wp, radius
        )
    else:
        measure_wp = od.curvature.discrete_mean_curvature(
            vertices_wp, vertices_wp, faces_wp, radius
        )
    pointwise_wp = measure_wp.numpy() / (np.pi * radius**2)

    def median_gap(reference: np.ndarray, estimate: np.ndarray) -> float:
        return float(np.median(np.abs(estimate - reference))) / scale

    assert median_gap(curvature_pml, pointwise_wp) < 0.05
    shuffled_pml = np.random.default_rng(0).permutation(curvature_pml)
    assert median_gap(shuffled_pml, pointwise_wp) > 0.05
    assert median_gap(curvature_pml, 0.5 * pointwise_wp) > 0.05


@pytest.mark.parametrize("case", ["no_points", "no_faces", "no_adjacent_pair"])
def test_discrete_mean_curvature_without_a_dihedral(device: str, case: str) -> None:
    """
    Not a library comparison: no shared edge means no bending, so the measure is zero.

    The mean-curvature measure sums dihedral angles over shared edges inside the ball. A lone
    triangle has no shared edge, a mesh with no face has none either, and no query point gives no
    value. trimesh's measure is no oracle here: its edge tree raises on an empty edge set.
    """
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    points_np = np.array([[0.2, 0.2, 0.0], [0.0, 0.0, 0.0]])
    faces_np = np.zeros(0, dtype=np.int32) if case == "no_faces" else np.arange(3, dtype=np.int32)
    queries_np = np.zeros((0, 3)) if case == "no_points" else points_np
    measure_wp = od.curvature.discrete_mean_curvature(
        points_to_warp(queries_np, device),
        points_to_warp(vertices_np, device),
        wp.array(faces_np, dtype=wp.int32, device=device),
        2.0,
    )
    assert np.array_equal(measure_wp.numpy(), np.zeros(len(queries_np), dtype=np.float32))


@pytest.mark.parametrize("frame_independent", [True, False])
def test_principal_directions_stay_orthogonal_on_an_axis_aligned_field(
    parabolic_lattice: tuple[tm.Trimesh, wp.Mesh], frame_independent: bool
) -> None:
    """
    Not a library comparison: ``PD1`` and ``PD2`` must span the tangent plane, never coincide.

    The two principal directions are eigenvectors of a 2x2 shape operator, and on a surface whose
    curvature aligns with the tangent frame the fit built, that operator comes out **diagonal**.
    That is the case the curved fixtures never reach and the one where reading the eigenvector off
    a single fixed row of ``m - lam*I`` fails: for the eigenvalue equal to ``m00`` the top row is
    the zero row, so it has to be read off the second row instead. ``parabolic_lattice`` reaches it
    at every vertex whose reference tangent lands on a grid direction, which is what that fixture
    exists for.

    Measured, and it is the discriminator: before the second-row fallback, **41 of 441** vertices
    here returned ``PD1 == (0, 1, 0)`` and ``PD2 == (0, -1, 0)`` -- parallel, so the max-curvature
    direction was 90 degrees off -- and 22 of 441 did on ``cuda:0``, identically for both
    ``frame_independent`` modes.

    **The orthogonality assert is no longer what catches that**, and the analytic-direction asserts
    below are, which is worth stating because it is not obvious: the second direction is now a
    cross product with the normal, so the pair is orthonormal by construction whether or not the
    *first* direction is right. Mutation-probed by restoring the single-row eigenvector -- both
    parametrized arms then fail at "PD1 must run along the flat ruling" and nothing else. That is
    the assert this fixture exists for; orthogonality is the sibling test's job.

    The fixture is checked for non-vacuity two ways: every vertex must produce a frame at all (a
    failed fit returns zeros and would pass an orthogonality test trivially), and the field must
    actually be anisotropic, or the directions would be arbitrary at an umbilic point -- which is
    the sibling case ``test_principal_directions_are_a_frame_at_an_umbilic_point`` covers.

    The directions are also checked against the answer this surface has by hand, which is what
    says the fix picked the *right* pair of axes rather than merely two different ones. A parabolic
    cylinder is developable: it bends across the ruling (x) and is flat along it (y). Under the
    ``PV1 >= PV2`` ordering that makes ``PD1`` the *flat* direction, because the surface curves
    away from its ``+z`` normal so the bending curvature is the negative one -- the assertions
    below are the way round they look, not transposed.
    """
    _mesh_tm, mesh_wp = parabolic_lattice

    pd1_wp, pd2_wp, pv1_wp, pv2_wp = od.curvature.principal_curvature(
        mesh_wp.points, mesh_wp.indices, frame_independent=frame_independent
    )
    pd1_np, pd2_np = pd1_wp.numpy(), pd2_wp.numpy()

    # Non-vacuity: zeros are the failed-fit signal and would satisfy orthogonality for free.
    assert np.all(np.linalg.norm(pd1_np, axis=1) > 0.5)
    assert np.all(np.linalg.norm(pd2_np, axis=1) > 0.5)
    # Non-vacuity: at an umbilic point any tangent pair is a valid answer.
    assert np.abs(pv1_wp.numpy() - pv2_wp.numpy()).min() > 0.1

    dots_np = np.abs(np.einsum("ij,ij->i", pd1_np, pd2_np))
    assert dots_np.max() < 1e-3, f"worst |PD1 . PD2| {dots_np.max():.3e}"

    # The analytic answer for a developable parabolic cylinder, and it pins which axis is which:
    # the ruling (y) is flat and the cross-ruling direction (x) carries all the bending. The
    # ordering is PV1 >= PV2 and this surface curves *away* from its +z normal, so the bending
    # curvature is the negative one -- PD1 is the flat ruling and PD2 is across it, not the
    # reverse. The y axis lies in the surface everywhere, so PD1 is exactly it.
    assert np.abs(pd1_np[:, 1]).min() > 0.99, "PD1 must run along the flat ruling"
    assert np.abs(pd2_np[:, 1]).max() < 0.05, "PD2 must run across the ruling"
    assert np.abs(pv1_wp.numpy()).max() < 0.05, "the ruling direction is flat"
    assert pv2_wp.numpy().max() < -0.4, "the cross-ruling direction carries the bending"
    # A frame, not just a pair: both directions are unit and both lie in the tangent plane. The
    # normal is the one the fit itself uses -- the area-weighted vertex normal, which is what
    # ``principal_curvature`` builds internally -- not trimesh's, which weights differently and
    # sits 0.010 away on this lattice's boundary vertices. Tangency is a claim about the plane the
    # function fitted in, so it has to be read against that plane.
    normal_np = od.normals.vertex_normals(mesh_wp.points, mesh_wp.indices).numpy()
    assert np.allclose(np.linalg.norm(pd1_np, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(pd2_np, axis=1), 1.0, atol=1e-5)
    assert np.abs(np.einsum("ij,ij->i", normal_np, pd1_np)).max() < 1e-5
    assert np.abs(np.einsum("ij,ij->i", normal_np, pd2_np)).max() < 1e-5


@pytest.mark.parametrize("radius", [2, 5])
def test_principal_directions_are_a_frame_at_an_umbilic_point(
    sphere_round: tuple[tm.Trimesh, wp.Mesh], radius: int
) -> None:
    """
    Not a library comparison: at an umbilic point no reference fixes *which* pair is returned.

    A sphere is umbilic everywhere -- the two principal curvatures are equal, so every tangent
    direction is a principal direction and the shape operator is a multiple of the identity. No
    oracle can pin ``PD1`` there, which is exactly why
    ``test_principal_curvature_directions_match_pymeshlab`` takes ``torus_round`` and not a
    sphere (on ``icosphere`` the agreement read a meaningless 0.62). What is still a
    contract, and what nothing asserted before, is that the pair is a **frame**: two orthonormal
    vectors spanning the tangent plane. The helper that answers a degenerate 2x2 returns the
    reference frame's own two axes, so which frame it is depends on the vertex numbering, but that
    it is *a* frame does not.

    That makes this the sibling of
    ``test_principal_directions_stay_orthogonal_on_an_axis_aligned_field``, which covers the
    opposite end of the same helper: there the two eigenvalues are maximally separated and the
    eigenvectors are determined, here they coincide and only the invariant survives. The curvature
    magnitudes are still checked against the sphere's own ``1 / r``, which is the part an umbilic
    point does determine.

    Mutation-probed by restoring the two independent eigen-solves: both radii then fail, at the
    orthogonality assert and nowhere else. **Both radii are kept because they are not equally
    degenerate** -- how much anisotropy the solve sees depends on how much surface the ball covers,
    and an earlier single-radius version of this test caught that mutation only through the
    tangency assert, which is a weaker and more incidental guard.

    A quadric fitted over a wide spherical cap also overestimates curvature, and the bias grows
    with the ball -- scaled mean ``PV1`` reads 1.032 at radius 2 and 1.227 at radius 5 on
    ``sphere_round`` (400 dart-thrown vertices), against a true ``1 / r`` of 1. That is a property
    of the method, not a defect, so the magnitude assert is a one-sided bracket rather than a
    tolerance: the fit never reads *under* a sphere's curvature.

    ``sphere_round`` is the premise: an umbilic surface, irregularly sampled (no bumpy fixture is
    umbilic anywhere).
    """
    mesh_tm, mesh_wp = sphere_round

    pd1_wp, pd2_wp, pv1_wp, pv2_wp = od.curvature.principal_curvature(
        mesh_wp.points, mesh_wp.indices, radius=radius
    )
    pd1_np, pd2_np, pv1_np, pv2_np = (
        pd1_wp.numpy(),
        pd2_wp.numpy(),
        pv1_wp.numpy(),
        pv2_wp.numpy(),
    )

    # Non-vacuity: the fixture must actually be umbilic, or this is the anisotropic test again.
    assert np.abs(pv1_np - pv2_np).max() < 0.05, "the sphere must be umbilic to the fit's accuracy"
    # Non-vacuity: zeros are the failed-fit signal and satisfy every invariant below for free.
    assert np.all(np.linalg.norm(pd1_np, axis=1) > 0.5)

    # Orthonormal...
    assert np.allclose(np.linalg.norm(pd1_np, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(pd2_np, axis=1), 1.0, atol=1e-5)
    assert np.abs(np.einsum("ij,ij->i", pd1_np, pd2_np)).max() < 1e-3
    # ...and tangent to the plane the fit used, which is the area-weighted vertex normal.
    normal_np = od.normals.vertex_normals(mesh_wp.points, mesh_wp.indices).numpy()
    assert np.abs(np.einsum("ij,ij->i", normal_np, pd1_np)).max() < 1e-5
    assert np.abs(np.einsum("ij,ij->i", normal_np, pd2_np)).max() < 1e-5
    # The discrete normal is itself the exact radial one on a sphere, to within the tessellation:
    # that is what says the frame sits in the *surface's* tangent plane and not merely in a plane
    # of ordito's own choosing. On this irregular sampling the area-weighted normal sits up to 6.4
    # degrees off radial (``|cos|`` 0.9937; the angle-weighted one 2.4), so the bar is 0.98, 3x the
    # measured deviation; an arbitrary tangent-plane normal reads near 0.
    radial_np = np.asarray(mesh_tm.vertices, dtype=np.float64) - ROUND_FRAME_SHIFT
    radial_np /= np.linalg.norm(radial_np, axis=1, keepdims=True)
    assert np.abs(np.einsum("ij,ij->i", radial_np, normal_np)).min() > 0.98

    # The magnitudes an umbilic point does determine: both principal curvatures are 1 / r, the
    # same at every vertex. Spread measured 0.0142 at radius 2 and 0.0238 at 5 (curvature 1 / 1.7),
    # against a 0.05 bar; the scaled means are 1.024-1.032 and 1.209-1.227, inside the bracket the
    # fit's own cap bias sets (the smallest scaled value 1.011 and 1.186: never under).
    sphere_radius = float(np.linalg.norm(mesh_tm.vertices - ROUND_FRAME_SHIFT, axis=1).mean())
    assert np.ptp(pv1_np) < 0.05, "a sphere's curvature is the same at every vertex"
    assert np.ptp(pv2_np) < 0.05
    assert 1.0 <= float(pv1_np.mean()) * sphere_radius <= 1.25
    assert 1.0 <= float(pv2_np.mean()) * sphere_radius <= 1.25


@pytest.mark.parametrize("frame_independent", [True, False])
def test_principal_directions_match_the_analytic_torus(
    torus_round: tuple[tm.Trimesh, wp.Mesh], frame_independent: bool
) -> None:
    """
    Class A against a closed form: on a torus the principal directions are the parameter curves.

    Not a library comparison, and that is the point rather than a shortfall. The reference
    libraries cover this function unevenly: igl is an oracle for ``frame_independent=False`` only
    (it *is* the symmetrized operator that flag reproduces), and pymeshlab's
    ``vertex_curvature_principal_dir1_matrix`` is asserted for ``PD1`` alone, because its two
    directions are ordered differently from ordito's at 5% of vertices. That left ``PD2`` in the
    default ``frame_independent=True`` branch -- the output most sensitive to how the second
    eigenvector is obtained -- with no reference comparison at all. A torus has one.

    A torus of revolution is a principal-coordinate surface: its meridians (around the tube) and
    its parallels (around the axis) are the lines of curvature everywhere, with curvatures ``1/r``
    and ``cos(theta) / (R + r cos(theta))``. Those two families are recovered here from the vertex
    positions alone -- no fit, no library -- so this is an exact oracle for the *directions*, which
    is what the assert reads. The magnitudes are left to the igl and pymeshlab comparisons above,
    since the quadric fit's cap bias makes them a weaker claim than the directions.

    Measured on ``torus_round`` (400 dart-thrown vertices on ``R = 1``, ``r = 0.4``, two principal
    curvatures at least 1.79 apart everywhere, so no umbilic vertex): ``|cos|`` worst **0.9625**
    for the meridians and 0.9917 for the parallels, median 0.9977 / 0.9994, in both modes. That is
    the quadric fit's accuracy on an irregular sampling, not a defect: libigl's own fit on the same
    mesh reads 0.9626 / 0.9884 at radius 2 and 0.9577 / 0.9832 at radius 5. Before the second
    direction was derived as a cross product the parallels read **0.0000** -- the returned pair
    failed to contain one of the two lines of curvature at all on some vertices -- which is the
    regression this pins. The bars are a worst ``|cos|`` above 0.88 (3x the measured deviation)
    and a median above 0.99, which a systematic tilt of the frame would fail.
    """
    mesh_tm, mesh_wp = torus_round
    major_radius, minor_radius = 1.0, 0.4

    # Recover each vertex's (meridian, parallel) frame from its position. The tube's centre circle
    # has radius ``major_radius``, so the vector from the nearest point on it is the surface normal
    # direction, and the two tangents follow from it.
    # The analytic frame lives in the torus's canonical coordinates; ordito's directions are rotated
    # into them (a direction ignores the shift and the scale).
    vertices_np = round_frame_coordinates(mesh_tm.vertices)
    angle_np = np.arctan2(vertices_np[:, 1], vertices_np[:, 0])
    axis_np = np.stack(
        [np.cos(angle_np), np.sin(angle_np), np.zeros_like(angle_np)], axis=1
    )  # outward radial direction of the centre circle
    normal_np = vertices_np - major_radius * axis_np
    normal_np /= np.linalg.norm(normal_np, axis=1, keepdims=True)
    parallel_np = np.stack([-np.sin(angle_np), np.cos(angle_np), np.zeros_like(angle_np)], axis=1)
    meridian_np = np.cross(normal_np, parallel_np)
    meridian_np /= np.linalg.norm(meridian_np, axis=1, keepdims=True)

    # Non-vacuity: no umbilic vertices, so both directions are genuinely determined.
    cos_theta_np = np.einsum("ij,ij->i", normal_np, axis_np)
    curvature_gap_np = np.abs(
        1.0 / minor_radius - cos_theta_np / (major_radius + minor_radius * cos_theta_np)
    )
    assert curvature_gap_np.min() > 1.0, "a torus fixture with an umbilic vertex is the wrong one"

    pd1_wp, pd2_wp, _, _ = od.curvature.principal_curvature(
        mesh_wp.points, mesh_wp.indices, frame_independent=frame_independent
    )
    pd1_np, pd2_np = pd1_wp.numpy() @ ROUND_FRAME_ROTATION, pd2_wp.numpy() @ ROUND_FRAME_ROTATION
    assert np.all(np.linalg.norm(pd1_np, axis=1) > 0.5), "every fit must have produced a frame"

    # The returned pair must *contain* both lines of curvature. Which of PD1/PD2 carries which is
    # the PV1 >= PV2 ordering's business and flips with the sign of the parallel curvature across
    # the inner and outer halves of the tube, so each analytic direction is matched against the
    # better of the two -- an eigenvector is defined up to sign, hence the absolute value.
    for name, exact_np in (("meridian", meridian_np), ("parallel", parallel_np)):
        alignment_np = np.maximum(
            np.abs(np.einsum("ij,ij->i", pd1_np, exact_np)),
            np.abs(np.einsum("ij,ij->i", pd2_np, exact_np)),
        )
        assert alignment_np.min() > 0.88, f"{name}: worst |cos| {alignment_np.min():.4f}"
        assert np.median(alignment_np) > 0.99, f"{name}: median |cos| {np.median(alignment_np):.4f}"


def test_principal_curvature_is_reproducible(saddle_graded: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Ordito against ordito: the consumer that made ``vertex_normals``' summation order visible.

    The oracle for the values is ``test_principal_curvature_graded``; this pins
    repeatability. The quadric fit is ill conditioned at a near-flat vertex, so it amplified the
    one-ULP (1.19e-07) run-to-run movement of a ``float32`` atomic accumulator into curvature
    swings of up to **7.96e-04** absolute and **77 % relative** on ``half_torus`` -- at 19 of 544
    vertices, and not as a swap of ``PV1`` with ``PV2`` (the *sorted* pair moved by the same
    amount). ``saddle_graded`` keeps the near-flat vertices that amplify it (its saddle centre).
    ``normals.vertex_normals`` accumulates in ``float64`` now, which is where the fix is;
    ``test_vertex_normals_are_reproducible`` guards that layer directly.
    """
    _, mesh_wp = saddle_graded

    runs = []
    for _ in range(8):
        _, _, pv1_wp, pv2_wp = od.curvature.principal_curvature(
            mesh_wp.points, mesh_wp.indices, radius=2
        )
        runs.append((pv1_wp.numpy().copy(), pv2_wp.numpy().copy()))

    # Non-vacuity: a constant or all-zero field would compare equal to itself for free (range 0.61).
    assert np.ptp(runs[0][0]) > 0.2
    for pv1_np, pv2_np in runs[1:]:
        assert np.array_equal(runs[0][0], pv1_np)
        assert np.array_equal(runs[0][1], pv2_np)


def test_principal_curvature_fits_each_chunk_of_balls_alike(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: the fit over a walk of several chunks is the fit over one.

    Each chunk of geodesic balls is fitted as the walk leaves it, so a mesh larger than one chunk
    launches the fit once a chunk, reading that chunk's queue rows. No fixture is that large, so the
    chunk is forced down to an uneven split; the one-chunk call carries the libigl oracle
    (``test_principal_curvature_graded``) and all four outputs must equal it bit for bit.
    """
    _, mesh_wp = saddle_graded
    whole = od.curvature.principal_curvature(mesh_wp.points, mesh_wp.indices, radius=3)
    chunk = mesh_wp.points.shape[0] // 3 + 7
    monkeypatch.setattr(od.neighbors, "_GEODESIC_BALL_CHUNK", chunk)
    chunked = od.curvature.principal_curvature(mesh_wp.points, mesh_wp.indices, radius=3)
    assert np.ptp(whole[2].numpy()) > 0.2
    for whole_wp, chunked_wp in zip(whole, chunked, strict=True):
        assert np.array_equal(whole_wp.numpy(), chunked_wp.numpy())


@pytest.mark.parametrize("scale", [3e-4, 1e-9])
def test_principal_curvature_is_scale_equivariant(
    sphere_round: tuple[tm.Trimesh, wp.Mesh], scale: float
) -> None:
    """
    Ordito against ordito: curvature has units of 1/length, so scaling the mesh scales it back.

    The oracle sits on the unit-scale side, which
    ``test_principal_curvature_with_supplied_face_tables`` / ``test_principal_curvature_graded``
    pin against libigl; this
    only asks that shrinking the mesh does not change the answer it reports in the mesh's own
    units. It did: the quadric fit's normal matrix has a diagonal spanning ``h^8`` to ``h^2`` at
    mesh scale ``h``, so the absolute singularity threshold in
    ``kernels.linalg.solve_normal_equations`` rejected well-conditioned fits and the kernel's
    fallback wrote zero curvature -- 42 of 642 vertices at ``1e-3`` and all 642 at ``3e-4`` on
    ``icosphere(3)``, with nothing raised.

    ``sphere_round`` is the premise: a fallback zero is caught by every vertex carrying a curvature
    far from zero, which a bumpy fixture's (crossing zero) does not.

    The smallest scale pins a *second*, independent break that lived one layer down and is
    fixed in ``kernels.triangles.face_normals_and_area``: an absolute floor on ``|cross|`` there
    zeroed every vertex normal at ``h <= 3e-6``, and a zero normal is a zero frame and so a zero
    curvature. ``test_normals.py::test_vertex_normals_area_matches_igl_at_any_scale`` is that
    layer's own guard; this one is the consumer that found it.
    """
    mesh_tm, mesh_wp = sphere_round
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)

    _, _, pv1_unit_wp, pv2_unit_wp = od.curvature.principal_curvature(
        points_to_warp(vertices_np, mesh_wp.device), mesh_wp.indices, radius=2
    )
    _, _, pv1_small_wp, pv2_small_wp = od.curvature.principal_curvature(
        points_to_warp(vertices_np * scale, mesh_wp.device), mesh_wp.indices, radius=2
    )
    pv1_unit_np, pv2_unit_np = pv1_unit_wp.numpy(), pv2_unit_wp.numpy()

    # Non-vacuity: the unit-scale answer is the unit sphere's, so every vertex must carry a real
    # curvature -- a zero here would make the comparison below one between two fallbacks.
    assert np.abs(pv1_unit_np).min() > 0.5
    assert np.allclose(pv1_small_wp.numpy() * scale, pv1_unit_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(pv2_small_wp.numpy() * scale, pv2_unit_np, rtol=1e-4, atol=1e-4)


def test_principal_curvature_mean_edge_length_changes_no_answer(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Ordito against ordito: ``mean_edge_length=`` only spares the reduction that measures it."""
    _mesh_tm, mesh_wp = saddle_graded
    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)
    plain = od.curvature.principal_curvature(mesh.vertices, mesh.faces)
    given = od.curvature.principal_curvature(
        mesh.vertices, mesh.faces, mean_edge_length=mesh.mean_edge_length
    )
    for given_wp, plain_wp in zip(given, plain, strict=True):
        assert np.array_equal(given_wp.numpy(), plain_wp.numpy())


def _analytic_torus_cloud(
    device: str, n: int
) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], np.ndarray, np.ndarray, np.ndarray]:
    """
    ``n`` random points on the torus ``R = 1``, ``r = 0.4`` placed in the round frame.

    Returns the points, their exact outward normals, the larger and smaller exact principal
    curvature, and the exact parallel direction, all in the placed frame. The meridian curvature
    is ``1 / r`` and the parallel one ``cos(theta) / (R + r cos(theta))``, negative on the inner
    half of the tube, so the pair changes order and sign across it.
    """
    rng = np.random.default_rng(31)
    major_radius, minor_radius = 1.0, 0.4
    theta_np, phi_np = rng.uniform(0.0, 2.0 * np.pi, (2, n))
    axis_np = np.stack([np.cos(phi_np), np.sin(phi_np), np.zeros(n)], axis=1)
    normal_np = np.cos(theta_np)[:, None] * axis_np + np.sin(theta_np)[:, None] * [0.0, 0.0, 1.0]
    points_np = major_radius * axis_np + minor_radius * normal_np
    placed_np = ROUND_FRAME_SCALE * points_np @ ROUND_FRAME_ROTATION.T + ROUND_FRAME_SHIFT
    meridian_np = np.full(n, 1.0 / minor_radius)
    parallel_np = np.cos(theta_np) / (major_radius + minor_radius * np.cos(theta_np))
    parallel_direction_np = np.stack([-np.sin(phi_np), np.cos(phi_np), np.zeros(n)], axis=1)
    return (
        points_to_warp(placed_np, device),
        points_to_warp(normal_np @ ROUND_FRAME_ROTATION.T, device),
        np.maximum(meridian_np, parallel_np) / ROUND_FRAME_SCALE,
        np.minimum(meridian_np, parallel_np) / ROUND_FRAME_SCALE,
        parallel_direction_np @ ROUND_FRAME_ROTATION.T,
    )


def test_point_principal_curvature_matches_the_analytic_torus(device: str) -> None:
    """
    Not a library comparison: no reference fits curvature on a bare cloud with given normals.

    pymeshlab's point-set curvature fits an algebraic sphere (APSS), a different estimator, and
    every other reference needs a mesh. The oracle is the closed form instead: 20 000 random
    points on the torus ``R = 1``, ``r = 0.4`` (rotated, scaled 1.7, off the origin) with their
    exact normals, whose principal curvatures and lines of curvature are known everywhere and
    whose smaller curvature is negative on the inner half of the tube, so a sign or an ordering
    error cannot pass.

    Measured over 16-nearest rows, in units of the tube's curvature ``1 / r``: the error of
    ``PV1`` median 0.0048, worst 0.097, of ``PV2`` median 0.0035, worst 0.058; the parallel
    direction within ``|cos|`` 0.9995 of one of ``(PD1, PD2)`` everywhere (other seeds alike).
    The bars are about 3x those. Flipping the normals negates both curvatures (``PV1`` error
    median 0.99), and swapping ``PV1`` with ``PV2`` reads a median error of 1.0.

    [`test_point_principal_curvature_matches_pymeshlab_apss`]
    [tests.test_curvature.test_point_principal_curvature_matches_pymeshlab_apss] is the library
    comparison.
    """
    points, normals, larger_np, smaller_np, parallel_np = _analytic_torus_cloud(device, 20_000)
    # Non-vacuity: the smaller curvature spreads and takes both signs.
    assert np.ptp(smaller_np) > 1.0
    assert smaller_np.min() < 0.0
    neighbor_idx, _ = od.neighbors.query_nearest(points, points, 16)
    pd1, pd2, pv1, pv2 = od.curvature.point_principal_curvature(points, normals, neighbor_idx)

    tube_scale = 0.4 * ROUND_FRAME_SCALE  # 1 / (the tube's curvature)
    for name, error_np in (
        ("PV1", np.abs(pv1.numpy() - larger_np) * tube_scale),
        ("PV2", np.abs(pv2.numpy() - smaller_np) * tube_scale),
    ):
        assert np.median(error_np) < 0.015, f"{name}: median error {np.median(error_np):.4f}"
        assert error_np.max() < 0.3, f"{name}: worst error {error_np.max():.4f}"
    alignment_np = np.maximum(
        np.abs(np.einsum("ij,ij->i", pd1.numpy(), parallel_np)),
        np.abs(np.einsum("ij,ij->i", pd2.numpy(), parallel_np)),
    )
    assert alignment_np.min() > 0.9985, f"worst |cos| {alignment_np.min():.4f}"
    assert np.allclose(np.einsum("ij,ij->i", pd1.numpy(), pd2.numpy()), 0.0, atol=1e-5)


@pytest.mark.parity("point_principal_curvature", "pymeshlab")
def test_point_principal_curvature_matches_pymeshlab_apss(device: str) -> None:
    """
    Class C: MeshLab's APSS curvature of the same cloud, a different estimator of one quantity.

    No reference fits a Monge jet on a bare point cloud from Python (CGAL's ``Jet_fitting_3`` has
    no binding that takes one). ``compute_curvature_and_color_apss_per_vertex`` estimates the same
    principal curvatures from the same points and normals by projecting onto an algebraic-sphere
    MLS surface, so the two agree to the accuracy of both rather than bit for bit: on the
    analytic torus of the test above (20 000 points) APSS at ``filterscale=4`` reads the exact
    curvatures to a median of 0.012 and a 95th percentile of 0.064, in units of the tube's
    curvature ``1 / r``, where ordito's 16-nearest fit reads 0.005.

    The statistic is the median and the 90th percentile of the difference, per curvature, in
    those units: measured 0.0137 / 0.047 for ``PV1`` and 0.0126 / 0.047 for ``PV2``; the bars
    are 0.04 / 0.15, about 3x. The tails are APSS's (its worst reads 0.37-0.43 off the exact
    value), which is why no maximum is asserted. ``PV1`` is the constant tube curvature here, so
    shuffling cannot catch it; ``PV2`` varies, and against a shuffled copy its median difference
    is 0.31, eight times the bar. APSS's two values are not ordered, so they are sorted first.
    """
    points, normals, _, _, _ = _analytic_torus_cloud(device, 20_000)
    neighbor_idx, _ = od.neighbors.query_nearest(points, points, 16)
    _pd1, _pd2, pv1, pv2 = od.curvature.point_principal_curvature(points, normals, neighbor_idx)

    points_np = points.numpy().astype(np.float64)
    normals_np = normals.numpy().astype(np.float64)
    normals_np /= np.linalg.norm(normals_np, axis=1, keepdims=True)
    curvatures_pml = []
    for curvature_type in ("K1", "K2"):
        meshset_pml = ml.MeshSet()
        meshset_pml.add_mesh(ml.Mesh(vertex_matrix=points_np, v_normals_matrix=normals_np))
        meshset_pml.compute_curvature_and_color_apss_per_vertex(
            filterscale=4.0, curvaturetype=curvature_type
        )
        curvatures_pml.append(meshset_pml.current_mesh().vertex_scalar_array().copy())
    larger_pml = np.maximum(*curvatures_pml)
    smaller_pml = np.minimum(*curvatures_pml)
    assert np.ptp(smaller_pml) > 1.0  # non-vacuity: the compared curvature varies

    tube_scale = 0.4 * ROUND_FRAME_SCALE  # 1 / (the tube's curvature)
    for name, difference_np in (
        ("PV1", np.abs(pv1.numpy() - larger_pml) * tube_scale),
        ("PV2", np.abs(pv2.numpy() - smaller_pml) * tube_scale),
    ):
        assert np.median(difference_np) < 0.04, f"{name}: median {np.median(difference_np):.4f}"
        p90 = np.percentile(difference_np, 90)
        assert p90 < 0.15, f"{name}: 90th percentile {p90:.4f}"


def test_point_principal_curvature_skips_padding_and_declines_small_rows(device: str) -> None:
    """
    Ordito against ordito: ``-1`` slots are absent, and a row of under six entries declines.

    Two trailing ``-1`` columns leave every value unchanged bit for bit, which a kernel reading
    the slot as a point would not. Rows of the point and four neighbours cannot determine the
    five-parameter fit, so every output is zero there, the documented "no fit" answer.
    """
    points, normals, _, _, _ = _analytic_torus_cloud(device, 2_000)
    neighbor_idx, _ = od.neighbors.query_nearest(points, points, 12)
    plain = od.curvature.point_principal_curvature(points, normals, neighbor_idx)
    rows_np = neighbor_idx.numpy()
    padded_np = np.concatenate([rows_np, np.full((len(rows_np), 2), -1, np.int32)], axis=1)
    padded_idx = odt.as_array2d(wp.array(padded_np, dtype=wp.int32, device=device), wp.int32)
    padded = od.curvature.point_principal_curvature(points, normals, padded_idx)
    assert np.all(plain[2].numpy() != 0.0)
    for padded_wp, plain_wp in zip(padded, plain, strict=True):
        assert np.array_equal(padded_wp.numpy(), plain_wp.numpy())

    short_idx = odt.as_array2d(
        wp.array(np.ascontiguousarray(rows_np[:, :5]), dtype=wp.int32, device=device), wp.int32
    )
    for output_wp in od.curvature.point_principal_curvature(points, normals, short_idx):
        assert not np.any(output_wp.numpy())


def test_point_principal_curvature_rejects_mismatched_tables(device: str) -> None:
    """A normal table or neighbour table without one row per point raises ``ValueError``."""
    points, normals, _, _, _ = _analytic_torus_cloud(device, 50)
    neighbor_idx, _ = od.neighbors.query_nearest(points, points, 8)
    with pytest.raises(ValueError, match="one entry per point"):
        od.curvature.point_principal_curvature(points, odt.as_dense(normals[:40]), neighbor_idx)
    with pytest.raises(ValueError, match="one entry per point"):
        od.curvature.point_principal_curvature(
            points, normals, odt.as_array2d(odt.as_dense(neighbor_idx[:40]), wp.int32)
        )


@pytest.mark.parametrize(
    "mesh_name", ["sphere_irregular", "torus_irregular_holes", "saddle_graded"]
)
@pytest.mark.parity("mls_principal_curvature_apss", "pymeshlab")
def test_mls_principal_curvature_apss_matches_pymeshlab(
    request: pytest.FixtureRequest, tmp_path: Path, mesh_name: str
) -> None:
    """
    Class B (values through their sum and product): ``method="apss"`` is MeshLab's APSS curvature.

    The cloud is the fixture's vertices with trimesh's vertex normals, both as ``float32``.
    ``compute_curvature_and_color_apss_per_vertex`` projects each point onto the same APSS surface
    and reads the curvature off the same gradient and Hessian; it exposes the values only through
    one scalar per call, and its ``"K1"`` / ``"K2"`` outputs kill the process wherever the pair is
    complex in rounding (``sqrt(H^2 - 4K)`` of a near-umbilic point is NaN, and its histogram pass
    then corrupts the heap), so the transform reads its ``"Mean"`` (the *sum* ``PV1 + PV2``: vcg
    returns the Weingarten map's trace) and ``"Gauss"`` (the product), which give the pair.
    Measured 1.9e-08 to 7.6e-08 of each field's largest magnitude; tolerance ``1e-5`` of it.

    **MeshLab runs in a fresh process.** Its projection stops on ``ProjectionAccuracy`` times a
    point spacing that nothing initializes, so the answer depends on what the heap held there: in
    a fresh process zero (every point takes all 15 steps; ordito stops at 1e-10 of the point's
    reach, which changes nothing measurable), after other MeshLab work in the same process
    anything -- the same input then moved ``sphere_irregular``'s fields by 4.6e-04, and stopping
    at 1e-4 of the spacing (the documented intent) by 8e-06.

    The directions are compared on ``saddle_graded`` only. vcg's ``WeingartenMap`` solves the
    non-symmetric ``(I - n n^T) H / |g|`` with Eigen's self-adjoint solver, which reads its lower
    triangle alone: a NumPy replication of that solve reproduces MeshLab's first direction to
    1e-15 on ``sphere_irregular``, where it is as far as ``|cos|`` 0.16 from the symmetric shape
    operator's. On the saddle the operator is nearly symmetric (``|cos|`` above 0.9995). There
    MeshLab's first direction is the one of the smaller ``|PV|``; pairing it with the other reads
    at most 0.0094.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np = np.asarray(mesh_tm.vertices, dtype=np.float32)
    normals_np = np.asarray(mesh_tm.vertex_normals, dtype=np.float32)
    fields_pml = _pymeshlab_apss_fields_in_a_fresh_process(points_np, normals_np, tmp_path)
    pd1_wp, pd2_wp, pv1_wp, pv2_wp = od.curvature.mls_principal_curvature(
        wp.array(points_np, dtype=wp.vec3, device=mesh_wp.device),
        wp.array(normals_np, dtype=wp.vec3, device=mesh_wp.device),
    )
    pv1_np = pv1_wp.numpy().astype(np.float64)
    pv2_np = pv2_wp.numpy().astype(np.float64)

    sum_scale = float(np.abs(fields_pml["Mean"]).max())
    product_scale = float(np.abs(fields_pml["Gauss"]).max())
    assert_nonconstant(fields_pml["Mean"], tol=1e-2 * sum_scale)
    assert_nonconstant(fields_pml["Gauss"], tol=1e-2 * product_scale)
    assert np.allclose(pv1_np + pv2_np, fields_pml["Mean"], rtol=0.0, atol=1e-5 * sum_scale)
    assert np.allclose(pv1_np * pv2_np, fields_pml["Gauss"], rtol=0.0, atol=1e-5 * product_scale)
    if mesh_name == "saddle_graded":
        smaller_np = np.where(
            (np.abs(pv1_np) <= np.abs(pv2_np))[:, None], pd1_wp.numpy(), pd2_wp.numpy()
        )
        distinct = np.abs(np.abs(pv1_np) - np.abs(pv2_np)) > 0.05 * np.abs(pv1_np).max()
        assert distinct.sum() > 100
        alignment_np = np.abs(np.einsum("ij,ij->i", smaller_np, fields_pml["first"]))
        assert alignment_np[distinct].min() > 0.999


_APSS_IN_A_FRESH_PROCESS = """
import os
import sys

import numpy as np
import pymeshlab as ml

data = np.load(sys.argv[1])
fields = {}
for kind in ("Mean", "Gauss"):
    meshset = ml.MeshSet()
    meshset.add_mesh(ml.Mesh(data["points"], v_normals_matrix=data["normals"]))
    meshset.compute_curvature_and_color_apss_per_vertex(curvaturetype=kind)
    fields[kind] = meshset.current_mesh().vertex_scalar_array()
    fields["first"] = meshset.current_mesh().vertex_curvature_principal_dir1_matrix()
np.savez(sys.argv[2], **fields)
sys.stdout.flush()
os._exit(0)
"""


def _pymeshlab_apss_fields_in_a_fresh_process(
    points_np: np.ndarray, normals_np: np.ndarray, tmp_path: Path
) -> dict[str, np.ndarray]:
    """
    Compute MeshLab's APSS ``"Mean"`` and ``"Gauss"`` fields and first directions in a new process.

    The test above says why. The child exits without tearing pymeshlab down (its teardown can
    abort after a filter has run), so its result is read off the file it wrote.
    """
    source, result = tmp_path / "cloud.npz", tmp_path / "fields.npz"
    np.savez(source, points=points_np.astype(np.float64), normals=normals_np.astype(np.float64))
    subprocess.run(
        [sys.executable, "-c", _APSS_IN_A_FRESH_PROCESS, str(source), str(result)], check=True
    )
    with np.load(result) as fields:
        return {name: fields[name] for name in fields.files}


@pytest.mark.parametrize("method", ["apss", "rimls"])
def test_mls_principal_curvature_matches_the_analytic_torus(
    device: str, method: Literal["apss", "rimls"]
) -> None:
    """
    Not a library comparison for RIMLS; the closed form for both, on a dense analytic torus.

    No reference computes RIMLS curvature: MeshLab's RIMLS colorize filter writes zero curvature and
    uninitialized directions (``RIMLS::gradient`` never sets the error flag the filter tests, so
    every point reads as failed) and its MLS projection filters segfault. The torus of
    revolution sampled on its parameter grid (4 608 points, ``R = 1``, ``r = 0.4``) with its
    exact normals, whose larger principal curvature is ``1 / r`` along the meridian everywhere.
    Measured median ``PV1 r`` 0.9992 (APSS) and 1.0187 (RIMLS, which does not reproduce a sphere
    exactly: 1.09 on a coarser grid), and ``|cos|`` of ``PD1`` with the meridian 1.0000 at every
    point for both (the grid is symmetric about each meridian). A swapped pair reads ``|cos|``
    0.0000 and the parallel's curvature, at most 0.72 against 2.5. Bars: median ``PV1 r`` within
    1 % (APSS) and 5 % (RIMLS), median ``|cos|`` above 0.99, and an answer at every point.
    """
    mesh_tm = tm.creation.torus(
        major_radius=1.0, minor_radius=0.4, major_sections=96, minor_sections=48
    )
    points_np = np.asarray(mesh_tm.vertices, dtype=np.float32)
    angle_np = np.arctan2(points_np[:, 1], points_np[:, 0])
    axis_np = np.stack([np.cos(angle_np), np.sin(angle_np), np.zeros_like(angle_np)], axis=1)
    normals_np = points_np - axis_np
    normals_np /= np.linalg.norm(normals_np, axis=1, keepdims=True)
    parallel_np = np.stack([-np.sin(angle_np), np.cos(angle_np), np.zeros_like(angle_np)], axis=1)
    meridian_np = np.cross(normals_np, parallel_np)

    pd1_wp, _, pv1_wp, _ = od.curvature.mls_principal_curvature(
        wp.array(points_np, dtype=wp.vec3, device=device),
        wp.array(normals_np.astype(np.float32), dtype=wp.vec3, device=device),
        method=method,
    )
    ratio_np = pv1_wp.numpy() * 0.4
    assert ratio_np.min() > 0.0
    assert abs(np.median(ratio_np) - 1.0) < (0.01 if method == "apss" else 0.05)
    alignment_np = np.abs(np.einsum("ij,ij->i", pd1_wp.numpy(), meridian_np))
    assert np.median(alignment_np) > 0.99, f"median |cos| {np.median(alignment_np):.4f}"


def test_mls_principal_curvature_rimls_keeps_a_cube_edge_sharp(device: str) -> None:
    """
    Not a library comparison: the property RIMLS exists for, against APSS on a cube.

    No reference computes RIMLS curvature (see the torus test above). 6 000 points sampled evenly on
    a unit cube's faces with the faces' normals; within 0.02-0.05 of an edge, inside every
    sample's reach, APSS's sphere fits round the edge off (median ``|PV1|`` 3.77) and RIMLS's
    refitting drops the other face's samples (0.12); both read zero farther in. Bars: RIMLS
    under a tenth of APSS there, and both flat (median below 0.01) beyond 0.1.
    """
    box_tm = tm.creation.box()
    points_np, face_index_np = tm.sample.sample_surface_even(box_tm, 6000, seed=0)
    normals_np = box_tm.face_normals[face_index_np]
    edge_distance_np = np.sort(0.5 - np.abs(points_np), axis=1)[:, 1]
    near = (edge_distance_np > 0.02) & (edge_distance_np < 0.05)
    far = edge_distance_np > 0.1
    assert near.sum() > 300
    assert far.sum() > 1000
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    normals_wp = wp.array(normals_np.astype(np.float32), dtype=wp.vec3, device=device)
    _, _, apss_wp, _ = od.curvature.mls_principal_curvature(points_wp, normals_wp, method="apss")
    _, _, rimls_wp, _ = od.curvature.mls_principal_curvature(points_wp, normals_wp, method="rimls")
    apss_np, rimls_np = np.abs(apss_wp.numpy()), np.abs(rimls_wp.numpy())
    assert np.median(rimls_np[near]) < 0.1 * np.median(apss_np[near])
    assert np.median(apss_np[far]) < 0.01
    assert np.median(rimls_np[far]) < 0.01


@pytest.mark.parametrize("count", [0, 1])
def test_mls_principal_curvature_of_too_few_points_is_zero(device: str, count: int) -> None:
    """Ordito against itself: a surface needs two samples, so fewer give the zero answer."""
    points_wp = wp.array(np.zeros((count, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    normals_wp = wp.array(np.ones((count, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    for array_wp in od.curvature.mls_principal_curvature(points_wp, normals_wp):
        assert array_wp.shape == (count,)
        assert not array_wp.numpy().any()


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({"method": "imls"}, r"method must be 'apss' or 'rimls', got 'imls'"),
        ({"filter_scale": 0.0}, r"filter_scale must be positive, got 0.0"),
        ({"short_normals": True}, r"normals must have one entry per point, got 3 for 4"),
    ],
)
def test_mls_principal_curvature_validates_its_arguments(
    device: str, keywords: dict[str, object], message: str
) -> None:
    """Each documented ``ValueError``, raised before any launch."""
    points_wp = wp.array(np.eye(4, 3, dtype=np.float32), dtype=wp.vec3, device=device)
    short = bool(keywords.pop("short_normals", False))
    normals_wp = wp.array(
        np.ones((3 if short else 4, 3), dtype=np.float32), dtype=wp.vec3, device=device
    )
    with pytest.raises(ValueError, match=message):
        od.curvature.mls_principal_curvature(points_wp, normals_wp, **keywords)  # pyright: ignore[reportArgumentType]
