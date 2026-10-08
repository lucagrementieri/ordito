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


@pytest.mark.parity("normal_consistency_loss", "pytorch3d")
def test_normal_consistency_loss_matches_pytorch3d(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Class A on edge-manifold input, with the non-manifold divergence **pinned** rather than avoided.

    ``mesh_normal_consistency`` enumerates every *pair* of faces sharing an edge -- ``C(k, 2)``
    pairs at an edge with ``k`` incident faces, through its own
    ``_C.mesh_normal_consistency_find_verts`` -- where
    [`face_adjacency_angles`][ordito.adjacency.face_adjacency_angles] reports one pair per
    adjacency. The two coincide exactly wherever every edge has at most two faces, which is what
    the first half measures: 0.1660283 against 0.1660283 over ``sphere_irregular``'s 1 494 pairs.

    The second half is the divergence itself, on three faces sharing one edge, and it is sharper
    than a factor: pytorch3d sees ``C(3, 2) = 3`` pairs there and reports **0.777**, while
    [`face_adjacency`][ordito.adjacency.face_adjacency] keeps only edges with *exactly* two
    incident faces and so reports **no pairs at all** and a loss of ``0.0``. The test pins both
    numbers rather than papering over them with a tolerance; without that half the class-A label
    would read as a claim about all input.
    """
    mesh_tm, mesh_wp = sphere_irregular
    loss_p3d = float(p3d_loss.mesh_normal_consistency(trimesh_to_pytorch3d(mesh_tm)))
    loss_wp = od.energies.normal_consistency_loss(mesh_wp.points, mesh_wp.indices)

    assert loss_p3d > 1e-6
    assert np.allclose(loss_wp, loss_p3d, rtol=1e-4, atol=0.0)

    # Three faces on one edge: 3 reference pairs against ordito's 2 adjacencies.
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.3], [0.0, 0.2, 1.0]],
        dtype=np.float64,
    )
    faces_np = np.array([[0, 1, 2], [1, 0, 3], [1, 0, 4]], dtype=np.int64)
    fan_p3d = float(
        p3d_loss.mesh_normal_consistency(trimesh_to_pytorch3d(tm.Trimesh(vertices_np, faces_np)))
    )
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    fan_wp = od.energies.normal_consistency_loss(vertices_wp, faces_wp)

    assert np.allclose(fan_p3d, 0.777404, rtol=1e-4, atol=0.0)
    assert fan_wp == 0.0


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


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "saddle_graded"])
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
        *(
            ("curved_hessian_energy", name)
            for name in ("sphere_irregular", "saddle_graded", "torus_irregular")
        ),
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
    curvature-rich closed case where the ``K`` correction actually contributes. Its closed arms
    also check, as an invariant rather than a comparison, that constants have zero curved energy:
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

    with pytest.raises(ValueError, match="together"):
        od.energies.crouzeix_raviart_massmatrix(
            mesh_wp.points, mesh_wp.indices, unique_edges=unique_edges
        )


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


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
def test_lscm_hessian_matches_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: igl exposes the Hessian only as ``igl.lscm``'s second return, so it comes from there.

    The named transform is the extraction, not a value change: igl's ``Q`` is exactly
    ``-repdiag(L, 2) - 2A``, the same matrix ordito assembles, compared over the union of the two
    sparsity patterns because the two builds order their CSR entries differently. This builds the
    Hessian only, no conjugate-gradient solve.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    hessian_wp = bsr_to_csr(od.energies.lscm_hessian(mesh_wp.points, mesh_wp.indices))
    assert sparse_allclose(hessian_wp, _lscm_q_igl(mesh_tm), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
def test_vector_area_matrix_matches_igl_derived(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: ``vector_area_matrix`` is unbound, so it is solved for from two functions that are.

    ``A = (-repdiag(L, 2) - Q) / 2`` inverts the definition of the LSCM Hessian, giving an
    independent reference out of ``igl.cotmatrix`` and ``igl.lscm`` -- both of which ordito is
    compared against separately, so the derivation does not smuggle in ordito's own answer.
    Section 6 lists this among the C++ functions with no Python binding.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)
    laplacian_igl = igl.cotmatrix(vertices_np, faces_np)
    area_igl = cast(
        sp.csr_matrix, (-sp.block_diag([laplacian_igl, laplacian_igl]) - _lscm_q_igl(mesh_tm)) / 2.0
    )

    area_wp = bsr_to_csr(od.energies.vector_area_matrix(mesh_wp.points, mesh_wp.indices))
    assert sparse_allclose(area_wp, area_igl, rtol=1e-5, atol=1e-5)
