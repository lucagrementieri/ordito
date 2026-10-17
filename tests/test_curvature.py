"""Regression tests for ``ordito.curvature`` against ``trimesh.curvature`` (CPU reference)."""

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
import ordito.typing as odt
from tests.comparisons import assert_nonconstant, fraction_within
from tests.conftest import ROUND_FRAME_ROTATION, ROUND_FRAME_SHIFT, round_frame_coordinates
from tests.conversions import points_to_warp, trimesh_to_pymeshlab


@pytest.mark.parity("principal_curvature", "igl")
def test_principal_curvature(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Class A: curvature values against libigl, frame-dependent path."""
    mesh_tm, mesh_wp = sphere_irregular

    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    _, _, pv1_igl, pv2_igl, _ = igl.principal_curvature(vertices_np, faces_np, useKring=False)

    # frame_independent=False reproduces igl::principal_curvature's symmetrized shape operator.
    _, _, pv1_wp, pv2_wp = od.curvature.principal_curvature(
        mesh_wp.points, mesh_wp.indices, frame_independent=False
    )

    assert np.allclose(pv1_wp.numpy(), pv1_igl, atol=1e-3, rtol=1e-3)
    assert np.allclose(pv2_wp.numpy(), pv2_igl, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
def test_principal_curvature_graded(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Values and directions against libigl where curvature varies, on both shape-operator paths.

    **Class B (directions up to sign), ``frame_independent=False``**: reproduces
    ``igl::principal_curvature``'s symmetrized shape operator, so values and directions match.

    **Class C (a fraction bound), ``frame_independent=True``**: the default solves the true
    generalized eigenproblem (a surface invariant) rather than libigl's frame-dependent symmetrized
    operator. The two formulations share the trace of the shape operator, so the mean curvature
    ``(PV1 + PV2) / 2`` is preserved exactly; only the eigenvalue *spread* differs, and only
    appreciably at high-anisotropy vertices where ``PV1 - PV2`` is large. The bulk of vertices
    therefore stay close to libigl.
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
    mask[gap < 1e-2] = False

    pd1_wp, pd2_wp, pv1_wp, pv2_wp = od.curvature.principal_curvature(
        mesh_wp.points, mesh_wp.indices, frame_independent=False
    )
    # Tolerance is relaxed relative to the closed-mesh test: float32 input vs libigl float64,
    # plus slight radius difference from avg_edge_length rounding.
    assert np.allclose(pv1_wp.numpy()[mask], pv1_igl[mask], atol=5e-2, rtol=5e-2)
    assert np.allclose(pv2_wp.numpy()[mask], pv2_igl[mask], atol=5e-2, rtol=5e-2)
    # Directions defined up to sign — compare |cos angle| ≈ 1 at non-umbilic vertices
    pd1_dot = np.abs(np.einsum("ij,ij->i", pd1_wp.numpy()[mask], pd1_igl[mask]))
    pd2_dot = np.abs(np.einsum("ij,ij->i", pd2_wp.numpy()[mask], pd2_igl[mask]))
    assert np.allclose(pd1_dot, 1.0, atol=1e-1)
    assert np.allclose(pd2_dot, 1.0, atol=1e-1)

    # Default (frame_independent=True): true Weingarten map, independent of the tangent frame.
    _, _, pv1_wp, pv2_wp = od.curvature.principal_curvature(mesh_wp.points, mesh_wp.indices)
    pv1_indep = pv1_wp.numpy()
    pv2_indep = pv2_wp.numpy()

    # Mean curvature (the shared trace invariant) must match libigl tightly.
    mean_indep = 0.5 * (pv1_indep + pv2_indep)
    mean_igl = 0.5 * (np.asarray(pv1_igl) + np.asarray(pv2_igl))
    assert np.allclose(mean_indep[mask], mean_igl[mask], atol=5e-2, rtol=5e-2)

    # The principal values themselves stay close for the vast majority of vertices; genuine
    # divergence is confined to the few highest-anisotropy vertices, which the mask already drops.
    # Class C, so it carries the shuffle probe its helper asks for: both fractions measure
    # **1.0000** over the 544 surviving vertices, and permuting the reference drops them to 0.105
    # and 0.074 -- 9x under the bar, so the threshold is testing the correspondence and not the
    # marginal distributions.
    assert fraction_within(pv1_indep[mask], pv1_igl[mask]) > 0.95
    assert fraction_within(pv2_indep[mask], pv2_igl[mask]) > 0.95


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


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity("discrete_gaussian_curvature", "trimesh")
def test_discrete_gaussian_curvature(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A: the Cohen-Steiner/Morvan ball measure against trimesh's, at the same radius.

    Both sides are fed *trimesh's* face angles, so the comparison isolates the ball integration
    rather than re-testing [`triangles.face_angles`], which has its own oracle. Only four query
    points, which is enough because the measure is local and each one integrates an independent
    1-ring.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    face_angles_tm = mesh_tm.face_angles
    points_tm = mesh_tm.vertices[:4]
    radius = 0.1
    gauss_curvature_tm = tm.curvature.discrete_gaussian_curvature_measure(
        mesh_tm, points_tm, radius
    )

    points_wp = points_to_warp(points_tm, mesh_wp.device)
    face_angles_wp = odt.as_array2d(
        wp.array(face_angles_tm, dtype=wp.float32, device=mesh_wp.device), wp.float32
    )
    gauss_curvature_wp = od.curvature.discrete_gaussian_curvature(
        points_wp, mesh_wp.points, mesh_wp.indices, face_angles_wp, radius
    )
    assert np.allclose(gauss_curvature_wp.numpy(), gauss_curvature_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("radius", [2.0, 0.5])
@pytest.mark.parity("discrete_mean_curvature", "trimesh")
def test_discrete_mean_curvature(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], radius: float
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
    scale error.)

    ``benchmarks/test_curvature.py`` records why pymeshlab cannot be the oracle here (a
    different operator, 0.982 correlation with a 7 % offset).
    """
    mesh_tm, mesh_wp = sphere_irregular
    points_tm = mesh_tm.vertices
    mean_curvature_tm = tm.curvature.discrete_mean_curvature_measure(mesh_tm, points_tm, radius)

    points_wp = points_to_warp(points_tm, mesh_wp.device)
    mean_curvature_wp = od.curvature.discrete_mean_curvature(
        points_wp, mesh_wp.points, mesh_wp.indices, radius
    )
    # Non-vacuous on the curved fixture: a constant reference would pass any per-vertex bug.
    assert_nonconstant(mean_curvature_tm, tol=1e-3)
    assert np.allclose(mean_curvature_wp.numpy(), mean_curvature_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not wp.is_cuda_available(),
    reason="needs a second device to make the current device differ from the arrays' device",
)
def test_discrete_gaussian_curvature_ignores_the_current_device(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
):
    """
    Class A: ``discrete_gaussian_curvature`` answers on its inputs' device, not Warp's current one.

    Companion to ``test_vertices.py``'s scatter-wrapper case: a launch of this function that
    forwarded no ``device=`` once went unseen, because the ordinary tests run with the arrays'
    device already current.
    """
    mesh_tm, mesh_wp = saddle_graded
    radius = 0.5
    points_tm = mesh_tm.vertices
    face_angles_tm = mesh_tm.face_angles
    gauss_curvature_tm = tm.curvature.discrete_gaussian_curvature_measure(
        mesh_tm, points_tm, radius
    )

    points_wp = points_to_warp(points_tm, mesh_wp.device)
    face_angles_wp = odt.as_array2d(
        wp.array(face_angles_tm, dtype=wp.float32, device=mesh_wp.device), wp.float32
    )

    with wp.ScopedDevice("cpu"):
        gauss_curvature_wp = od.curvature.discrete_gaussian_curvature(
            points_wp, mesh_wp.points, mesh_wp.indices, face_angles_wp, radius
        )

    assert str(gauss_curvature_wp.device) == str(mesh_wp.device)
    assert np.allclose(gauss_curvature_wp.numpy(), gauss_curvature_tm, rtol=1e-5, atol=1e-5)


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
    normal_np = od.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices).numpy()
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
    normal_np = od.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices).numpy()
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
    ``vertices.vertex_normals`` accumulates in ``float64`` now, which is where the fix is;
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


@pytest.mark.parametrize("scale", [1e-3, 3e-4, 1e-6, 1e-9])
def test_principal_curvature_is_scale_equivariant(
    sphere_round: tuple[tm.Trimesh, wp.Mesh], scale: float
) -> None:
    """
    Ordito against ordito: curvature has units of 1/length, so scaling the mesh scales it back.

    The oracle sits on the unit-scale side, which
    ``test_principal_curvature`` / ``test_principal_curvature_graded`` pin against libigl; this
    only asks that shrinking the mesh does not change the answer it reports in the mesh's own
    units. It did: the quadric fit's normal matrix has a diagonal spanning ``h^8`` to ``h^2`` at
    mesh scale ``h``, so the absolute singularity threshold in
    ``kernels.linalg.solve_normal_equations`` rejected well-conditioned fits and the kernel's
    fallback wrote zero curvature -- 42 of 642 vertices at ``1e-3`` and all 642 at ``3e-4`` on
    ``icosphere(3)``, with nothing raised.

    ``sphere_round`` is the premise: a fallback zero is caught by every vertex carrying a curvature
    far from zero, which a bumpy fixture's (crossing zero) does not.

    The two smallest scales pin a *second*, independent break that lived one layer down and is
    fixed in ``kernels.triangles.face_normals_and_area``: an absolute floor on ``|cross|`` there
    zeroed every vertex normal at ``h <= 3e-6``, and a zero normal is a zero frame and so a zero
    curvature. ``test_vertices.py::test_vertex_normals_area_matches_igl_at_any_scale`` is that
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
