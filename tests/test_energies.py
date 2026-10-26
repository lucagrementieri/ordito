"""
Regression tests for ``ordito.energies``: the quadratic forms assembled from a Laplacian.

Every comparison here is against libigl, which is the only reference that exposes these operators at
all -- ``harmonic_integrated_from_laplacian_and_mass``, ``hessian_energy``,
``curved_hessian_energy``, the Crouzeix-Raviart pair and ``lscm``'s ``Q``. Mirrors the module's
source order: k-harmonic, the two Hessian energies, the Crouzeix-Raviart pair, the LSCM operators.

``vector_area_matrix`` has no binding of its own, so it is derived from the two things that do:
``A = (-repdiag(L, 2) - Q) / 2``. That is a class-B transform, named here so the assert is not
mistaken for a direct one.
"""

from __future__ import annotations

import itertools
from typing import Literal, cast

import igl
import numpy as np
import pytest
import pytorch3d.loss as p3d_loss
import scipy.sparse as sp
import trimesh as tm
import warp as wp

import ordito as od
from tests.comparisons import sparse_allclose
from tests.conftest import BOUNDARY_MESHES, MESHES
from tests.conversions import (
    bsr_to_csr,
    mesh_igl,
    numpy_to_warp,
    scipy_to_bsr,
    trimesh_to_pytorch3d,
)


@pytest.mark.parametrize("target_length", [0.0, 0.3])
@pytest.mark.parity("edge_length_loss", "pytorch3d")
def test_edge_length_loss_matches_pytorch3d(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], target_length: float
) -> None:
    """
    Class A: ``mesh_edge_loss`` at two resting lengths, one of them non-zero.

    Both points matter. At ``target_length = 0.0`` the loss is the mean squared edge length and a
    sign error in the deviation would be invisible; at 0.3 the two sides are 240x smaller and only
    agree if the subtraction happens before the squaring. Measured 0.0899725 against pytorch3d's
    0.0899726 and 3.73347e-04 against 3.73347e-04 over 480 unique edges.

    pytorch3d's per-mesh ``1 / E`` weighting collapses to a plain mean for a single mesh, which is
    ordito's only case -- so this is a direct comparison rather than a class-B one.
    """
    mesh_tm, mesh_wp = sphere_irregular
    loss_p3d = float(
        p3d_loss.mesh_edge_loss(trimesh_to_pytorch3d(mesh_tm), target_length=target_length)
    )
    loss_wp = od.energies.edge_length_loss(
        mesh_wp.points, mesh_wp.indices, target_length=target_length
    )

    assert loss_p3d > 1e-6
    assert np.allclose(loss_wp, loss_p3d, rtol=1e-5, atol=0.0)


def test_edge_length_loss_edges_change_no_answer(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Ordito against ordito: ``unique_edges=`` only spares the edge sort; the loss is unchanged."""
    _mesh_tm, mesh_wp = sphere_irregular
    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)
    plain = od.energies.edge_length_loss(mesh.vertices, mesh.faces, 0.3)
    given = od.energies.edge_length_loss(
        mesh.vertices, mesh.faces, 0.3, unique_edges=mesh.edges_unique
    )
    assert plain > 0.0
    assert given == plain


@pytest.mark.parametrize("mesh_name", [*MESHES, "mobius", "boy_surface"])
@pytest.mark.parity("normal_consistency_loss", "pytorch3d")
def test_normal_consistency_loss_matches_pytorch3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: ``mesh_normal_consistency``'s per-pair mean, on oriented and non-orientable meshes.

    The reference orients every pair by the shared edge, ``(v1 - v0) x (apex - v0)`` per face, and
    so does ordito; ``mobius`` and ``boy_surface`` have no consistent winding at all, so a pair
    oriented by the faces' windings instead would read some seams as folds there (the old dihedral
    form read 0.0062 against 0.0232 on ``mobius``).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    loss_p3d = float(p3d_loss.mesh_normal_consistency(trimesh_to_pytorch3d(mesh_tm)))
    loss_wp = od.energies.normal_consistency_loss(mesh_wp.points, mesh_wp.indices)

    assert loss_p3d > 1e-6
    assert np.allclose(loss_wp, loss_p3d, rtol=1e-5, atol=0.0)


@pytest.mark.parity("normal_consistency_loss", "pytorch3d")
def test_normal_consistency_loss_matches_pytorch3d_on_flips_and_fans(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Class A on the two inputs a winding-oriented, one-pair-per-edge form gets wrong.

    ``sphere_irregular`` with every seventh face flipped: the reference is winding-free, so its
    loss equals the unflipped mesh's, and so must ordito's (a face-winding dihedral read 0.581
    against 0.166 here). Three faces on one edge: the reference counts all ``C(3, 2) = 3`` pairs
    and reports 0.777, where a pairing that kept only two-face edges would report no pair at all.
    """
    mesh_tm, _mesh_wp = sphere_irregular
    flipped_np = np.asarray(mesh_tm.faces).copy()
    flipped_np[::7] = flipped_np[::7, ::-1]
    flipped_tm = tm.Trimesh(mesh_tm.vertices, flipped_np, process=False)
    assert not flipped_tm.is_winding_consistent
    flipped_p3d = float(p3d_loss.mesh_normal_consistency(trimesh_to_pytorch3d(flipped_tm)))
    loss_p3d = float(p3d_loss.mesh_normal_consistency(trimesh_to_pytorch3d(mesh_tm)))
    flipped_vertices_wp, flipped_faces_wp = numpy_to_warp(mesh_tm.vertices, flipped_np, device)
    flipped_wp = od.energies.normal_consistency_loss(flipped_vertices_wp, flipped_faces_wp)

    assert np.allclose(flipped_p3d, loss_p3d, rtol=1e-5, atol=0.0)
    assert np.allclose(flipped_wp, flipped_p3d, rtol=1e-5, atol=0.0)

    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.3], [0.0, 0.2, 1.0]],
        dtype=np.float64,
    )
    faces_np = np.array([[0, 1, 2], [1, 0, 3], [1, 0, 4]], dtype=np.int64)
    fan_p3d = float(
        p3d_loss.mesh_normal_consistency(
            trimesh_to_pytorch3d(tm.Trimesh(vertices_np, faces_np, process=False))
        )
    )
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    fan_wp = od.energies.normal_consistency_loss(vertices_wp, faces_wp)

    assert np.allclose(fan_p3d, 0.777404, rtol=1e-4, atol=0.0)
    assert np.allclose(fan_wp, fan_p3d, rtol=1e-5, atol=0.0)


def test_normal_consistency_loss_sorted_edge_keys_change_no_answer(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Ordito against ordito: ``sorted_edge_keys=`` only spares the edge sort."""
    _mesh_tm, mesh_wp = sphere_irregular
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    plain = od.energies.normal_consistency_loss(vertices_wp, faces_wp)
    given = od.energies.normal_consistency_loss(
        vertices_wp, faces_wp, sorted_edge_keys=od.adjacency.sorted_face_edge_keys(faces_wp)
    )
    assert plain > 0.0
    assert given == plain


@pytest.mark.parametrize(
    ("method", "mesh_name"),
    [
        ("uniform", "sphere_irregular"),
        ("cot", "sphere_irregular"),
        ("cotcurv", "sphere_well_shaped"),
    ],
)
@pytest.mark.parity("laplacian_smoothing_loss", "pytorch3d")
def test_laplacian_smoothing_loss_matches_pytorch3d(
    request: pytest.FixtureRequest, method: Literal["uniform", "cot", "cotcurv"], mesh_name: str
) -> None:
    """
    Class B: all three of ``mesh_laplacian_smoothing``'s methods, each under its own rescaling.

    The three are three different quantities and not a tuning knob, which is what makes the
    parametrize worth having: measured **0.14793 / 0.02894** on ``sphere_irregular`` and
    **0.27200** on ``sphere_well_shaped``, so a branch answering with the wrong normalization
    cannot pass. Agreement 1.6e-07 / 4.0e-07 / 7.0e-08 relative. ``cotcurv`` divides by
    pytorch3d's vertex areas, which it forms by Heron's formula in ``float32``: on
    ``sphere_irregular``'s needles those are 3.9e-05 off (the loss 2.0e-05), so that method runs
    on the needle-free fixture; ``cot`` normalizes by the row sum, where the area cancels.

    Class B rather than A because the reference reads a cotangent Laplacian whose off-diagonal is
    twice ordito's half-cotangent table and whose diagonal is identically zero; the two ratios
    ``(L v) / rowsum`` and ``(L v) / (6 M)`` are invariant to that factor, which is the named
    transform and is why the wrapper can assemble from ordito's own ``cotmatrix``.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    loss_p3d = float(
        p3d_loss.mesh_laplacian_smoothing(trimesh_to_pytorch3d(mesh_tm), method=method)
    )
    loss_wp = od.energies.laplacian_smoothing_loss(mesh_wp.points, mesh_wp.indices, method)

    assert loss_p3d > 1e-3
    assert np.allclose(loss_wp, loss_p3d, rtol=1e-5, atol=0.0)


def test_laplacian_smoothing_loss_methods_are_three_quantities(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: the three methods are far apart, and an empty mesh is 0.0.

    Not a parity assert -- the reference comparison above carries the oracle. This is the guard
    that keeps the parametrized test above non-vacuous: if two methods ever collapsed onto one
    answer, that test would still pass on the wrong branch. ``curvature.mean_curvature`` is the
    ``cotcurv`` variant's per-vertex sibling and is the reason it is an order of magnitude larger:
    it carries units of one over length where the other two are lengths.
    """
    mesh_tm, mesh_wp = sphere_irregular
    del mesh_tm
    losses = [
        od.energies.laplacian_smoothing_loss(mesh_wp.points, mesh_wp.indices, method)
        for method in ("uniform", "cot", "cotcurv")
    ]
    assert losses[0] > losses[1] > 0.0
    assert losses[2] > 5.0 * losses[0]
    assert all(abs(a - b) > 1e-3 for a, b in itertools.pairwise(losses))

    empty_vertices_wp = wp.zeros(0, dtype=wp.vec3, device=mesh_wp.points.device)
    empty_faces_wp = wp.zeros(0, dtype=wp.int32, device=mesh_wp.points.device)
    assert od.energies.laplacian_smoothing_loss(empty_vertices_wp, empty_faces_wp) == 0.0
    assert od.energies.edge_length_loss(empty_vertices_wp, empty_faces_wp) == 0.0
    assert od.energies.normal_consistency_loss(empty_vertices_wp, empty_faces_wp) == 0.0
    with pytest.raises(ValueError, match="method must be"):
        od.energies.laplacian_smoothing_loss(
            mesh_wp.points,
            mesh_wp.indices,
            "cotan",  # pyright: ignore[reportArgumentType]  # the off-menu value under test
        )


@pytest.mark.parametrize("method", ["uniform", "cot", "cotcurv"])
def test_laplacian_smoothing_loss_pattern_changes_no_answer(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], method: Literal["uniform", "cot", "cotcurv"]
) -> None:
    """
    Ordito against ordito: ``pattern=`` only spares the operator's sort, so the loss is unchanged.

    Each method takes the pattern of the operator it assembles, and the other one is refused.
    """
    _mesh_tm, mesh_wp = sphere_irregular
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    operators = ("laplacian_directed", "cotmatrix")
    right, wrong = operators if method == "uniform" else operators[::-1]
    pattern = od.laplacian.mesh_operator_pattern(faces_wp, vertices_wp.size, operator=right)

    plain = od.energies.laplacian_smoothing_loss(vertices_wp, faces_wp, method)
    given = od.energies.laplacian_smoothing_loss(vertices_wp, faces_wp, method, pattern=pattern)
    assert plain > 0.0
    assert given == plain
    with pytest.raises(ValueError, match="pattern was built for"):
        od.energies.laplacian_smoothing_loss(
            vertices_wp,
            faces_wp,
            method,
            pattern=od.laplacian.mesh_operator_pattern(faces_wp, vertices_wp.size, operator=wrong),
        )


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
@pytest.mark.parity("k_harmonic", "igl")
def test_k_harmonic_matches_igl(request: pytest.FixtureRequest, mesh_name: str, k: int) -> None:
    """
    Class A: identical Laplacian and mass on both sides isolate the k-harmonic composition.

    igl's cotangent Laplacian and barycentric mass diagonal are handed to
    ``igl.harmonic_integrated_from_laplacian_and_mass`` and (uploaded unchanged) to
    ``k_harmonic``; the assembled ``Q`` must then agree to assembly rounding at every
    power, including the ``k == 3`` case whose entries igl itself flags as not numerically robust.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)

    laplacian_igl = igl.cotmatrix(vertices_np, faces_np)
    mass_igl = igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    q_igl = igl.harmonic_integrated_from_laplacian_and_mass(laplacian_igl, mass_igl, k)

    laplacian_wp = scipy_to_bsr(laplacian_igl, mesh_wp.device)
    mass_wp = wp.array(mass_igl.diagonal(), dtype=wp.float64, device=mesh_wp.device)
    q_wp = bsr_to_csr(od.energies.k_harmonic(laplacian_wp, mass_wp, k=k))

    scale = abs(q_igl).max()
    assert sparse_allclose(q_wp, q_igl, rtol=1e-9, atol=1e-9 * scale)


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
def test_k_harmonic_identity_mass_and_power_guard(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    With ``mass=None`` the operator is the plain power ``(-L)^k``, the ``tutte`` flavor.

    Checked against scipy's own sparse product; ``k < 1`` must raise.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)

    laplacian_igl = igl.cotmatrix(vertices_np, faces_np).tocsr()
    laplacian_wp = scipy_to_bsr(laplacian_igl, mesh_wp.device)
    q_wp = bsr_to_csr(od.energies.k_harmonic(laplacian_wp, k=2))
    q_sp = laplacian_igl @ laplacian_igl
    assert sparse_allclose(q_wp, q_sp, rtol=1e-9, atol=1e-9 * abs(q_sp).max())

    with pytest.raises(ValueError, match="k must be >= 1"):
        od.energies.k_harmonic(laplacian_wp, k=0)


_HESSIAN_ENERGIES = {
    "hessian_energy": (od.energies.hessian_energy, igl.hessian_energy),
    "curved_hessian_energy": (od.energies.curved_hessian_energy, igl.curved_hessian_energy),
}


@pytest.mark.parametrize(
    ("energy", "mesh_name"),
    [
        *(("hessian_energy", name) for name in ("sphere_irregular", "saddle_graded")),
        *(("curved_hessian_energy", name) for name in ("saddle_graded", "torus_irregular")),
    ],
)
@pytest.mark.parity("hessian_energy", "igl")
@pytest.mark.parity("curved_hessian_energy", "igl")
def test_hessian_energies_match_igl(
    request: pytest.FixtureRequest, energy: str, mesh_name: str
) -> None:
    """
    Class A on the assembled ``(n_vertices, n_vertices)`` matrix, for both Hessian energies.

    igl is handed the float32-rounded vertices ordito actually computes from, so the comparison
    isolates the operator assembly (the two-ring contraction, the Voronoi mass, the boundary
    kill) from input precision; entries scale as the inverse fourth power of the mesh size, which
    would otherwise let vertex rounding dominate the tolerance. The open fixtures are the ones
    that exercise the killed boundary degrees of freedom.

    ``curved_hessian_energy``'s per-face contraction of ``D^T Mi (L + K) Mi D`` must agree with
    igl's chained sparse products. ordito numbers and orients its unique edges differently from
    ``igl::orient_halfedges`` (min-vertex-first instead of first-occurrence-first), and the
    agreement here is what shows the energy is invariant to that gauge. ``torus`` is the
    curvature-rich closed case where the ``K`` correction actually contributes. Its closed arm
    also checks, as an invariant rather than a comparison, that constants have zero curved energy:
    every Crouzeix-Raviart gradient row sums to zero by construction.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_wp.points.numpy(), dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_wp.indices.numpy().reshape(-1, 3), dtype=np.int64)
    energy_od, energy_igl = _HESSIAN_ENERGIES[energy]

    q_igl = energy_igl(vertices_np, faces_np)
    q_wp = bsr_to_csr(energy_od(mesh_wp.points, mesh_wp.indices))

    scale = abs(q_igl).max()
    assert scale > 0.0
    assert sparse_allclose(q_wp, q_igl, rtol=1e-7, atol=1e-7 * scale)
    if energy == "curved_hessian_energy" and mesh_tm.is_watertight:
        assert np.abs(q_wp @ np.ones(q_wp.shape[0])).max() < 1e-9 * abs(q_wp).max()


def test_hessian_energy_annihilates_linear_fields_where_biharmonic_does_not(device: str) -> None:
    """
    The natural-boundary property the energy exists for, checked without igl.

    On a **flat** open mesh every affine field ``a + b.x`` has zero Hessian, so it must be exactly
    in the energy's null space — while the clamped biharmonic operator (``k_harmonic`` at
    ``k == 2``) penalizes the same fields at the boundary, which is the distortion Stein et
    al. 2018 diagnose. The contrast is asserted so the null-space check cannot pass vacuously.
    Flatness matters: on the curved fixtures the piecewise-linear Hessian of ``b.x`` is genuinely
    nonzero across bent edges (measured 5.06 against a 9.0 matrix scale on ``icosahedron``, for
    igl and ordito alike), so this property is only testable on a planar patch.
    """
    vertices_wp, faces_wp = od.creation.grid(count=(7, 7), device=device)
    vertices_np = np.ascontiguousarray(vertices_wp.numpy(), dtype=np.float64)
    n_vertices = len(vertices_np)
    linear_fields = np.c_[np.ones(n_vertices), vertices_np]

    q_hessian = bsr_to_csr(od.energies.hessian_energy(vertices_wp, faces_wp)).toarray()
    residual_hessian = np.abs(q_hessian @ linear_fields).max()

    laplacian_wp = od.laplacian.cotmatrix(vertices_wp, faces_wp, dtype=wp.float64)
    mass_wp = od.laplacian.mass_matrix_entries(vertices_wp, faces_wp, dtype=wp.float64)
    q_biharmonic = bsr_to_csr(od.energies.k_harmonic(laplacian_wp, mass_wp, k=2)).toarray()
    residual_biharmonic = np.abs(q_biharmonic @ linear_fields).max()

    scale = np.abs(q_hessian).max()
    assert residual_hessian < 1e-9 * scale
    assert residual_biharmonic > 1e-3 * scale


def _igl_edge_arguments(mesh_wp: wp.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Ordito's ``edges_unique`` numbering in the ``(E, EMAP)`` layout igl's CR bindings take.

    igl's ``EMAP`` is column-major over ``igl::oriented_facets`` — entry ``c * n_faces + f`` is
    the edge opposite corner ``c`` of face ``f`` — while ordito's ``inverse`` is row-major over
    halfedges ``(f[s], f[s+1])``, where slot ``s`` spans the edge opposite corner ``(s + 2) % 3``.
    """
    unique_edges_wp, inverse_wp = od.edges.edges_unique(
        mesh_wp.indices, n_vertices=mesh_wp.points.size
    )
    inverse_np = inverse_wp.numpy().reshape(-1, 3)
    edge_map_igl = np.concatenate([inverse_np[:, (c + 1) % 3] for c in range(3)], dtype=np.int64)
    return unique_edges_wp.numpy().astype(np.int64), edge_map_igl


_CROUZEIX_RAVIART = {
    "cotmatrix": (od.energies.crouzeix_raviart_cotmatrix, igl.crouzeix_raviart_cotmatrix),
    "massmatrix": (od.energies.crouzeix_raviart_massmatrix, igl.crouzeix_raviart_massmatrix),
}


@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "saddle_graded"])
@pytest.mark.parametrize("operator", list(_CROUZEIX_RAVIART))
@pytest.mark.parity("crouzeix_raviart_cotmatrix", "igl")
@pytest.mark.parity("crouzeix_raviart_massmatrix", "igl")
def test_crouzeix_raviart_operators_match_igl(
    request: pytest.FixtureRequest, operator: str, mesh_name: str
) -> None:
    """
    Class B; the named transform is the edge numbering, handed *to* igl.

    ``igl.crouzeix_raviart_cotmatrix`` and ``igl.crouzeix_raviart_massmatrix`` accept an explicit
    ``(E, EMAP)``, so feeding them ordito's ``edges_unique`` numbering makes the two
    ``(n_edges, n_edges)`` matrices directly comparable -- no row permutation is applied to either
    side's output. Asserted on the dense form so a stray off-diagonal triplet in the mass matrix is
    visible, exactly like ``test_mass_matrix_assembled_matches_igl``.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)
    operator_od, operator_igl = _CROUZEIX_RAVIART[operator]

    edges_igl, edge_map_igl = _igl_edge_arguments(mesh_wp)
    matrix_igl = operator_igl(vertices_np, faces_np, edges_igl, edge_map_igl).tocsr()
    matrix_wp = bsr_to_csr(operator_od(mesh_wp.points, mesh_wp.indices))

    assert matrix_wp.shape == matrix_igl.shape == (len(edges_igl), len(edges_igl))
    # ``np.allclose`` of the dense forms, entry for entry, without densifying: every entry where
    # the two differ is stored in their difference, and everywhere else both agree exactly. On
    # ``saddle_graded``'s 13 333 edges the dense pair was 2.8 GB and 3.9 s a call.
    difference = (matrix_wp - matrix_igl).tocoo()
    reference = np.asarray(matrix_igl[difference.row, difference.col]).ravel()
    assert np.all(np.abs(difference.data) <= 1e-5 + 1e-5 * np.abs(reference))


def test_crouzeix_raviart_shared_edge_numbering(saddle_graded: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Precomputed ``(unique_edges, edge_map)`` must not change the answer; half a pair raises."""
    _mesh_tm, mesh_wp = saddle_graded
    unique_edges, edge_map = od.edges.edges_unique(mesh_wp.indices, n_vertices=mesh_wp.points.size)
    derived = bsr_to_csr(
        od.energies.crouzeix_raviart_cotmatrix(mesh_wp.points, mesh_wp.indices)
    ).toarray()
    supplied = bsr_to_csr(
        od.energies.crouzeix_raviart_cotmatrix(
            mesh_wp.points, mesh_wp.indices, unique_edges=unique_edges, edge_map=edge_map
        )
    ).toarray()
    assert np.array_equal(derived, supplied)
    curved_derived = bsr_to_csr(
        od.energies.curved_hessian_energy(mesh_wp.points, mesh_wp.indices)
    ).toarray()
    curved_supplied = bsr_to_csr(
        od.energies.curved_hessian_energy(
            mesh_wp.points, mesh_wp.indices, unique_edges=unique_edges, edge_map=edge_map
        )
    ).toarray()
    assert np.array_equal(curved_derived, curved_supplied)

    with pytest.raises(ValueError, match="together"):
        od.energies.crouzeix_raviart_massmatrix(
            mesh_wp.points, mesh_wp.indices, unique_edges=unique_edges
        )
    with pytest.raises(ValueError, match="together"):
        od.energies.curved_hessian_energy(mesh_wp.points, mesh_wp.indices, edge_map=edge_map)


def test_operator_family_empty_mesh(device: str) -> None:
    """Empty face buffers return square zero-nnz operators of the right dimension."""
    vertices_wp = wp.array(np.zeros((3, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)

    hessian = od.energies.hessian_energy(vertices_wp, faces_wp)
    assert (int(hessian.nrow), int(hessian.ncol)) == (3, 3)
    curved = od.energies.curved_hessian_energy(vertices_wp, faces_wp)
    assert (int(curved.nrow), int(curved.ncol)) == (3, 3)
    cr_cot = od.energies.crouzeix_raviart_cotmatrix(vertices_wp, faces_wp)
    assert (int(cr_cot.nrow), int(cr_cot.ncol)) == (0, 0)
    cr_mass = od.energies.crouzeix_raviart_massmatrix(vertices_wp, faces_wp)
    assert (int(cr_mass.nrow), int(cr_mass.ncol)) == (0, 0)


def test_curved_hessian_and_crouzeix_raviart_cotmatrix_reject_non_edge_manifold(
    device: str,
) -> None:
    """
    Both raise on the edge-manifold precondition their docstrings document.

    Like the igl originals, which assert it, instead of silently disagreeing about what a third
    incident face means. Three faces sharing one edge: the same non-edge-manifold fixture
    ``test_is_edge_manifold_nonmanifold_fan_matches_pyvista`` uses.
    """
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces_np = np.array([[0, 1, 2], [0, 3, 1], [0, 1, 4]])
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    with pytest.raises(ValueError, match="edge-manifold"):
        od.energies.curved_hessian_energy(vertices_wp, faces_wp)
    with pytest.raises(ValueError, match="edge-manifold"):
        od.energies.crouzeix_raviart_cotmatrix(vertices_wp, faces_wp)


def _lscm_q_igl(mesh_tm: tm.Trimesh) -> sp.csr_matrix:
    """
    Return igl's LSCM Hessian ``Q``, which it exposes only as ``igl.lscm``'s second return.

    ``Q`` does not depend on the pins (they only constrain the solve), so two arbitrary ones do.
    """
    vertices_np, faces_np = mesh_igl(mesh_tm)
    pins_np = np.array([0, 1], dtype=np.int64)
    pins_uv_np = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    _, hessian_igl = igl.lscm(vertices_np, faces_np, pins_np, pins_uv_np)
    return hessian_igl.tocsr()


@pytest.mark.parametrize("mesh_name", BOUNDARY_MESHES)
@pytest.mark.parametrize("pattern_build", ["directed", "undirected"])
def test_lscm_hessian_matches_igl(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    mesh_name: str,
    pattern_build: str,
) -> None:
    """
    Class B: igl exposes the Hessian only as ``igl.lscm``'s second return, so it comes from there.

    The named transform is the extraction, not a value change: igl's ``Q`` is exactly
    ``-repdiag(L, 2) - 2A``, the same matrix ordito assembles, compared over the union of the two
    sparsity patterns because the two builds order their CSR entries differently. This builds the
    Hessian only, no conjugate-gradient solve. The Hessian from a mesh's cached ``cot_entries``
    and ``pattern`` is compared too. The boundary term is read off the Laplacian's pattern, so
    both of its builds are run (the undirected one by lowering its face-count switch), on one rim
    and on four.
    """
    if pattern_build == "undirected":
        monkeypatch.setattr(od.laplacian, "_UNDIRECTED_PATTERN_FROM_FACES", 0)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    hessian_igl = _lscm_q_igl(mesh_tm)
    # The boundary term is not vacuous here: ``-2A`` holds cross-quadrant entries.
    n_vertices = mesh_wp.points.shape[0]
    assert hessian_igl[:n_vertices, n_vertices:].count_nonzero() > 0
    hessian_wp = bsr_to_csr(od.energies.lscm_hessian(mesh_wp.points, mesh_wp.indices))
    assert sparse_allclose(hessian_wp, hessian_igl, rtol=1e-5, atol=1e-5)
    mesh = od.Trimesh(mesh_wp.points, mesh_wp.indices)
    hessian_cached_wp = od.energies.lscm_hessian(
        mesh_wp.points,
        mesh_wp.indices,
        cot_entries=mesh.cotmatrix_entries,
        pattern=mesh.mesh_operator_pattern,
    )
    assert sparse_allclose(bsr_to_csr(hessian_cached_wp), hessian_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", BOUNDARY_MESHES)
def test_vector_area_matrix_matches_igl_derived(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: ``vector_area_matrix`` is unbound, so it is solved for from two functions that are.

    ``A = (-repdiag(L, 2) - Q) / 2`` inverts the definition of the LSCM Hessian, giving an
    independent reference out of ``igl.cotmatrix`` and ``igl.lscm`` -- both of which ordito is
    compared against separately, so the derivation does not smuggle in ordito's own answer.
    It is among the C++ functions with no Python binding. On ``BOUNDARY_MESHES``, one rim and four,
    since the matrix is a sum over boundary edges and is zero on a closed mesh.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)
    laplacian_igl = igl.cotmatrix(vertices_np, faces_np)
    area_igl = cast(
        sp.csr_matrix, (-sp.block_diag([laplacian_igl, laplacian_igl]) - _lscm_q_igl(mesh_tm)) / 2.0
    )

    assert abs(area_igl).max() > 1e-3  # non-vacuity: the rims contribute
    area_wp = bsr_to_csr(od.energies.vector_area_matrix(mesh_wp.points, mesh_wp.indices))
    assert sparse_allclose(area_wp, area_igl, rtol=1e-5, atol=1e-5)


def test_vector_area_matrix_mates_change_no_answer(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: ``mates=`` only spares the boundary search its halfedge pairing.

    The saddle has a rim, so the matrix has entries, and they must match the plain call's.
    """
    _mesh_tm, mesh_wp = saddle_graded
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    mates_wp = od.halfedge.halfedge_mates(faces_wp, vertices_wp.size)

    plain_bsr = od.energies.vector_area_matrix(vertices_wp, faces_wp)
    given_bsr = od.energies.vector_area_matrix(vertices_wp, faces_wp, mates=mates_wp)
    n_stored = int(plain_bsr.offsets.numpy()[-1])
    assert n_stored > 0
    assert np.array_equal(given_bsr.offsets.numpy(), plain_bsr.offsets.numpy())
    for given_wp, plain_wp in (
        (given_bsr.columns, plain_bsr.columns),
        (given_bsr.values, plain_bsr.values),
    ):
        assert np.array_equal(given_wp.numpy()[:n_stored], plain_wp.numpy()[:n_stored])


@pytest.mark.parametrize(
    "function", ["edge_length_loss", "normal_consistency_loss", "laplacian_smoothing_loss"]
)
def test_loss_arrival_order_agrees_with_the_default(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], function: str
) -> None:
    """
    Ordito against ordito: determinism off returns the default's mean to rounding.

    The mode only swaps the mean's fixed-order block commit for one atomic per block; the
    reference comparisons above pin the default.
    """
    _, mesh_wp = sphere_irregular
    loss = getattr(od.energies, function)
    fixed = loss(mesh_wp.points, mesh_wp.indices)
    with od.config.deterministic(False):
        arrival = loss(mesh_wp.points, mesh_wp.indices)
    assert fixed > 0.0
    assert np.isclose(fixed, arrival, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("energy", ["hessian_energy", "curved_hessian_energy"])
def test_hessian_energies_are_reproducible_when_deterministic(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], energy: str
) -> None:
    """
    Not a parity assert: under ``ordito.config``'s deterministic mode two calls agree bit for bit.

    Both energies sum ``float64`` per-vertex quantities (the Voronoi mass, the angle sums) that an
    atomic scatter commits in arrival order on CUDA, and ``hessian_energy`` also emits its triplets
    in the vertex-face rows' order; the deterministic path sums both in a fixed order. The off
    mode is checked against the on mode to round-off, not to the bit.
    """
    _, mesh_wp = sphere_irregular
    energy_od = getattr(od.energies, energy)
    with od.config.deterministic(True):
        first = bsr_to_csr(energy_od(mesh_wp.points, mesh_wp.indices)).toarray()
        second = bsr_to_csr(energy_od(mesh_wp.points, mesh_wp.indices)).toarray()
    assert np.array_equal(first, second)
    with od.config.deterministic(False):
        arrival = bsr_to_csr(energy_od(mesh_wp.points, mesh_wp.indices)).toarray()
    assert np.allclose(arrival, first, rtol=1e-9, atol=1e-9 * np.abs(first).max())
