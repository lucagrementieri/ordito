from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import igl
import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg
import trimesh as tm
import warp as wp

import ordito as od
import ordito.typing as odt
from tests.comparisons import sparse_allclose
from tests.conftest import saddle_graded_arrays
from tests.conversions import (
    mesh_igl,
    numpy_to_warp,
    numpy_to_warp_uv,
    points_to_warp_uv,
    warp_empty,
)


def _face_flipped_indices_np(vertices_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    """NumPy reference for libigl ``flipped_triangles``: 2D signed area strictly negative."""
    tri = vertices_np[faces_np]  # (n_faces, 3, 2)
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    signed_area2 = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    return np.flatnonzero(signed_area2 < 0.0).astype(np.int64)


def _random_2d_mesh(rng: np.random.Generator, n_faces: int):
    """Random 2D triangle soup with mixed orientations as (vertices_2d, flat_faces)."""
    vertices_np = rng.standard_normal((n_faces * 3, 2)).astype(np.float64)
    faces_np = np.arange(n_faces * 3, dtype=np.int64).reshape(n_faces, 3)
    return vertices_np, faces_np


def _circle_boundary(mesh_wp: wp.Mesh) -> tuple[wp.array[wp.int32], wp.array[wp.vec2]]:
    """Return the longest boundary loop and its arc-length map onto the unit circle."""
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    return boundary_wp, od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)


def _harmonic_warm_start(
    mesh_wp: wp.Mesh,
) -> tuple[wp.array[wp.int32], wp.array[wp.vec2], wp.array[wp.vec2]]:
    """Return the circle boundary plus the harmonic map it pins, ARAP's warm start."""
    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    uv_init_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp
    )
    return boundary_wp, boundary_uv_wp, uv_init_wp


def _two_rim_pins(mesh_wp: wp.Mesh) -> tuple[np.ndarray, wp.array[wp.int32]]:
    """Two pins half a loop apart on the longest rim, as ``int32`` host and device indices."""
    loop_np = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices).numpy()
    pins_np = np.array([loop_np[0], loop_np[len(loop_np) // 2]], dtype=np.int32)
    return pins_np, wp.array(pins_np, dtype=wp.int32, device=mesh_wp.device)


def _unit_quad(device: str) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Return the two-triangle unit square in the ``z = 0`` plane."""
    vertices = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        dtype=wp.vec3,
        device=device,
    )
    faces = wp.array(np.array([0, 1, 2, 1, 3, 2], dtype=np.int32), dtype=wp.int32, device=device)
    return vertices, faces


def test_flipped_faces_random_mixed(device: str):
    rng = np.random.default_rng(0)
    vertices_np, faces_np = _random_2d_mesh(rng, n_faces=64)
    vertices_wp, faces_wp = numpy_to_warp_uv(vertices_np, faces_np, device)

    tri = vertices_np[faces_np]
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    mask_np = (e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]) < 0.0

    assert np.array_equal(
        od.parametrization.face_flipped_mask(vertices_wp, faces_wp).numpy(), mask_np
    )
    assert np.array_equal(
        od.parametrization.face_flipped_indices(vertices_wp, faces_wp).numpy(),
        _face_flipped_indices_np(vertices_np, faces_np),
    )


_SINGLE_TRIANGLE_NP = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)


@pytest.mark.parametrize(
    ("vertices_np", "faces_np", "expected_np"),
    [
        (_SINGLE_TRIANGLE_NP, np.array([[0, 1, 2]]), np.array([False])),
        (_SINGLE_TRIANGLE_NP, np.array([[2, 1, 0]]), np.array([True])),
        # Zero area: the strict ``< 0`` means a collinear triangle is not flagged.
        (np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), np.array([[0, 1, 2]]), np.array([False])),
        (np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=bool)),
    ],
    ids=["ccw", "cw", "collinear", "empty"],
)
def test_flipped_faces_on_known_windings(
    device: str, vertices_np: np.ndarray, faces_np: np.ndarray, expected_np: np.ndarray
):
    """Not a library comparison: the winding of each triangle is known by construction."""
    vertices_wp, faces_wp = numpy_to_warp_uv(vertices_np, faces_np, device)
    assert np.array_equal(
        od.parametrization.face_flipped_mask(vertices_wp, faces_wp).numpy(), expected_np
    )
    assert np.array_equal(
        od.parametrization.face_flipped_indices(vertices_wp, faces_wp).numpy(),
        np.flatnonzero(expected_np),
    )


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_band"])
@pytest.mark.parity("map_vertices_to_circle", "igl")
def test_map_vertices_to_circle_matches_igl(request: pytest.FixtureRequest, mesh_name: str):
    """Class A: the arc-length circle map against ``igl.map_vertices_to_circle``, no transform."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, _ = mesh_igl(mesh_tm)

    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_np = boundary_wp.numpy().astype(np.int64)

    circle_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    circle_igl = igl.map_vertices_to_circle(vertices_np, boundary_np)

    assert np.allclose(circle_wp.numpy(), circle_igl, rtol=1e-5, atol=1e-5)


def test_map_vertices_to_circle_single_vertex_loop(device: str):
    """A single-vertex loop has zero perimeter; the arc-length map must not divide 0/0 into NaN."""
    vertices_wp = wp.array(np.array([[0.0, 0.0, 0.0]]), dtype=wp.vec3, device=device)
    boundary_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    circle_wp = od.parametrization.map_vertices_to_circle(vertices_wp, boundary_wp)
    assert np.isfinite(circle_wp.numpy()).all()


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_band"])
@pytest.mark.parity(
    "graph_laplacian",
    "igl",
    benchmarked=False,
    reason="libigl builds A - diag(rowsum(A)) inline inside igl::harmonic rather than binding "
    "it, so the reference here is igl.adjacency_matrix plus a NumPy diagonal -- timing that "
    "would price a hand-rolled composition under a library's name. scipy binds the operation "
    "outright (csgraph.laplacian, measured an exact sign-flipped match at 0.0 on icosphere(2)) "
    "but takes an adjacency matrix, which ordito's face buffer is not, so its row would time "
    "the same composition one step earlier. benchmarks/test_laplacian.py carries the decline.",
)
def test_graph_laplacian_matches_igl(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class B: igl has no ``graph_laplacian``, so the reference is assembled from its adjacency.

    The named transform is ``A - diag(rowsum(A))`` over ``igl.adjacency_matrix``: the definition of
    the umbrella operator, built on the reference side. Note igl's adjacency is sized by ``F.max() +
    1`` rather than ``len(V)``, which is why it runs on fixtures where every
    vertex is referenced.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, faces_np = mesh_igl(mesh_tm)
    n_vertices = mesh_wp.points.size

    operator_wp = od.laplacian.graph_laplacian(mesh_wp.points, mesh_wp.indices)
    operator_csr = scipy.sparse.csr_matrix(
        (operator_wp.values.numpy(), operator_wp.columns.numpy(), operator_wp.offsets.numpy()),
        shape=(n_vertices, n_vertices),
    )

    adjacency_igl = igl.adjacency_matrix(faces_np).astype(np.float64)
    laplacian_igl = adjacency_igl - scipy.sparse.diags(
        np.asarray(adjacency_igl.sum(axis=1)).ravel()
    )

    assert sparse_allclose(operator_csr, laplacian_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_band"])
@pytest.mark.parity("harmonic", "igl")
@pytest.mark.parity("harmonic_conditioning", "igl")
def test_harmonic_matches_igl(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A: the same boundary constraints in, the same interior UVs out, at ``1e-4``.

    Both sides receive the identical circle map, so nothing about the boundary is under test here --
    the comparison isolates the interior solve. The tolerance is the CG stop, not a disagreement:
    igl factorizes where ordito iterates.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)

    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)

    uv_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp
    )
    uv_igl = igl.harmonic(vertices_np, faces_np, boundary_np, boundary_uv_np, 1)

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-4, atol=1e-4)


def _polyharmonic_reference(
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    boundary_np: np.ndarray,
    boundary_uv_np: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    ``harmonic(k)`` solved directly in SciPy over igl's cotangent and barycentric mass matrices.

    ``Q = -L (M^-1 (-L))^(k - 1)``, the interior block factored by ``spsolve``. igl.harmonic's own
    default mass is Voronoi, where ordito uses the barycentric lumped mass, so the reference is
    assembled here rather than taken from ``igl.harmonic`` at ``k > 1``.
    """
    laplacian = igl.cotmatrix(vertices_np, faces_np)
    mass_inv = scipy.sparse.diags(
        1.0
        / np.asarray(
            igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
        )
    )
    product = -laplacian
    for _ in range(k - 1):
        product = product @ mass_inv @ (-laplacian)
    # scipy-stubs type the product as ``_spbase | ArrayND``, neither of which it gives ``tocsc``.
    operator = scipy.sparse.csr_matrix(product)
    n_vertices = vertices_np.shape[0]
    interior = np.setdiff1d(np.arange(n_vertices), boundary_np)
    solution = scipy.sparse.linalg.spsolve(
        operator[interior][:, interior].tocsc(),
        -(operator[interior][:, boundary_np] @ boundary_uv_np),
    )
    uv_ref = np.zeros((n_vertices, 2))
    uv_ref[boundary_np] = boundary_uv_np
    uv_ref[interior] = solution
    return uv_ref


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("mesh_name", ["sphere_irregular_cap", "sphere_irregular_band"])
def test_polyharmonic_matches_reference(request: pytest.FixtureRequest, mesh_name: str, k: int):
    """
    Class B: igl's operators with the barycentric mass, solved directly, at ``1e-4``.

    The named transform is the mass: igl.harmonic's default is Voronoi. ``k == 3`` is solved by a
    factorization; its conjugate gradient used to stop at its tolerance 1e-5 to 1e-2 of the range
    away from the answer on these fixtures.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)

    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)

    uv_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, k=k
    )
    uv_ref = _polyharmonic_reference(vertices_np, faces_np, boundary_np, boundary_uv_np, k)

    assert np.allclose(uv_wp.numpy(), uv_ref, rtol=1e-4, atol=1e-4)


def test_biharmonic_on_a_graded_patch_matches_a_direct_solve(device: str):
    """
    Class B against ``_polyharmonic_reference`` on the graded saddle, at 1 % of the UV range.

    The graded patch's biharmonic system is past what conjugate gradient reaches in ``float64``: it
    stalled for its whole iteration cap and returned a map 0.76 of the range away on this fixture
    (0.29 on the 68 x 68 one, 0.54 on ``benchmarks``' 133 x 133), with no warning. The solve is
    verified now and falls back to a factorization: measured 4.4e-5 of the range, so the bound is
    over 200x off the agreement and 76x under the defect. On the benchmark mesh the answer is
    itself ambiguous at the percent level (direct solves under four SuperLU orderings disagree by
    0.6 to 2.2 %), which is why this runs on the smaller fixture.
    """
    vertices_np, faces_np = saddle_graded_arrays()
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(vertices_np), np.asarray(faces_np, dtype=np.int32).ravel(), device
    )
    boundary_wp = od.boundary.longest_boundary_loop(vertices_wp, faces_wp)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(vertices_wp, boundary_wp)

    uv_wp = od.parametrization.harmonic(vertices_wp, faces_wp, boundary_wp, boundary_uv_wp, k=2)
    uv_ref = _polyharmonic_reference(
        vertices_wp.numpy().astype(np.float64),
        np.asarray(faces_np, dtype=np.int64),
        boundary_wp.numpy().astype(np.int64),
        boundary_uv_wp.numpy().astype(np.float64),
        2,
    )

    assert np.ptp(uv_ref) > 1.0
    assert np.abs(uv_wp.numpy() - uv_ref).max() < 0.01 * np.ptp(uv_ref)


def test_biharmonic_is_deterministic(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    # Regression guard for operator-assembly nondeterminism: the float64-native biharmonic operator
    # must give the same result across repeated calls. The original defect sized a rebuild's triplet
    # buffers by ``BsrMatrix.nnz`` (the capacity, not the entry count) and so fed the uninitialized
    # tail to ``bsr_from_triplets``, manifesting as ~1e22 / NaN corruption; a tight tolerance (well
    # above conjugate-gradient's ~1e-8 atomic last-ULP jitter) reliably catches a regression.
    _, mesh_wp = saddle_graded
    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    first = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, k=2
    ).numpy()
    for _ in range(5):
        again = od.parametrization.harmonic(
            mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, k=2
        ).numpy()
        assert np.allclose(again, first, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_band"])
def test_tutte_matches_igl_reference(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class B: igl has no ``tutte``, so the reference is its fixed-value minimizer on ``D - A``.

    Two named transforms, both on the reference side: assemble the uniform Laplacian from
    ``igl.adjacency_matrix`` and hand it to ``igl.min_quad_with_fixed`` under ordito's own boundary
    constraints. That isolates the interior solve, which is the only part not already covered by
    [`test_graph_laplacian_matches_igl`] and [`test_map_vertices_to_circle_matches_igl`].
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, faces_np = mesh_igl(mesh_tm)
    n_vertices = mesh_wp.points.size

    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)

    uv_wp = od.parametrization.tutte(mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp)

    # Reference: uniform Laplacian D - A solved with igl's fixed-value quadratic minimizer, given
    # the identical boundary constraints, so the comparison isolates the interior solve.
    # igl's binding types the matrix ``csc_matrix[int]``, which scipy-stubs' ``sum`` rejects.
    adjacency = cast("scipy.sparse.csc_matrix[np.int64]", igl.adjacency_matrix(faces_np))
    laplacian = (
        scipy.sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)) - adjacency
    ).tocsc()
    if laplacian.shape[0] != n_vertices:
        laplacian.resize((n_vertices, n_vertices))  # pyright: ignore[reportArgumentType]  # stubs: Never
    uv_igl = np.asarray(
        igl.min_quad_with_fixed(
            laplacian, np.zeros((n_vertices, 2), np.float64), boundary_np, boundary_uv_np
        )
    )

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-4, atol=1e-4)


def test_tutte_disk_is_fold_free(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    # A disk-topology mesh with a convex (circle) boundary yields a bijective, fold-free Tutte map.
    _, mesh_wp = saddle_graded
    boundary_wp, boundary_uv_wp = _circle_boundary(mesh_wp)
    uv_wp = od.parametrization.tutte(mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp)
    assert od.parametrization.face_flipped_indices(uv_wp, mesh_wp.indices).numpy().size == 0


def _arap_igl(
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    fixed_np: np.ndarray,
    fixed_uv_np: np.ndarray,
    uv_init_np: np.ndarray,
    max_iterations: int,
):
    """Libigl ARAP reference (dim=2, elements energy); returns the (n, 2) UV."""
    data = igl.ARAPData()
    data.max_iter = max_iterations
    igl.arap_precomputation(vertices_np, faces_np, 2, fixed_np.astype(np.int32), data)
    return igl.arap_solve(
        fixed_uv_np.astype(np.float64), data, np.ascontiguousarray(uv_init_np.astype(np.float64))
    )


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_band"])
@pytest.mark.parity("arap", "igl")
def test_arap_matches_igl(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A at a fixed iteration count, with the *same warm start* fed to both sides.

    ARAP is a local-global iteration, so its answer depends on where it started and how many rounds
    it ran: both are pinned here (ordito's own harmonic solve, 10 iterations), or the comparison
    would be measuring two different points along two different trajectories.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)

    # Full boundary loop pinned to the unit circle; identical harmonic warm start fed to both sides.
    boundary_wp, boundary_uv_wp, uv_init_wp = _harmonic_warm_start(mesh_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)
    uv_init_np = uv_init_wp.numpy().astype(np.float64)

    uv_wp = od.parametrization.arap(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, uv_init_wp, max_iterations=10
    )
    uv_igl = _arap_igl(vertices_np, faces_np, boundary_np, boundary_uv_np, uv_init_np, 10)

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-4, atol=1e-4)


def test_arap_free_boundary_matches_igl(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on the free-boundary branch: only two vertices pinned, the rest of the rim moving.

    The fully-pinned case above cannot see a bug in the boundary rows of the system, since there are
    none to solve; this is the same comparison with 4 iterations and almost the whole rim free.
    """
    # Pin only two boundary vertices to their harmonic UV; the rest of the boundary is free.
    mesh_tm, mesh_wp = saddle_graded
    vertices_np, faces_np = mesh_igl(mesh_tm)

    _, _, uv_init_wp = _harmonic_warm_start(mesh_wp)
    uv_init_np = uv_init_wp.numpy().astype(np.float64)

    fixed_np, fixed_wp = _two_rim_pins(mesh_wp)
    fixed_uv_np = uv_init_np[fixed_np]
    fixed_uv_wp = points_to_warp_uv(fixed_uv_np, mesh_wp.device)

    uv_wp = od.parametrization.arap(
        mesh_wp.points, mesh_wp.indices, fixed_wp, fixed_uv_wp, uv_init_wp, max_iterations=4
    )
    uv_igl = _arap_igl(vertices_np, faces_np, fixed_np, fixed_uv_np, uv_init_np, 4)

    # float32-UV drift vs igl's float64 grows slowly; four iterations stays well under 1e-3.
    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-3, atol=1e-3)


def test_arap_default_tolerance_tracks_a_tight_solve(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    # ``arap`` defaults its inner CG to 1e-7 rather than the 1e-8 the other solvers use: its global
    # solves are inner steps of a truncated outer iteration. Guard that the looser default still
    # tracks a tight solve two orders below it, far inside the 1e-4 gate the igl oracles use. The
    # pinned rows must equal the prescribed UV exactly (they are re-enforced every iteration).
    _, mesh_wp = saddle_graded
    boundary_wp, boundary_uv_wp, uv_init_wp = _harmonic_warm_start(mesh_wp)

    uv_default_wp = od.parametrization.arap(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, uv_init_wp, max_iterations=10
    )
    uv_tight_wp = od.parametrization.arap(
        mesh_wp.points,
        mesh_wp.indices,
        boundary_wp,
        boundary_uv_wp,
        uv_init_wp,
        max_iterations=10,
        tolerance=1e-9,
    )
    assert np.allclose(uv_default_wp.numpy(), uv_tight_wp.numpy(), rtol=1e-5, atol=1e-5)
    assert np.array_equal(uv_default_wp.numpy()[boundary_wp.numpy()], boundary_uv_wp.numpy())


def test_arap_all_vertices_fixed(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    # Every vertex pinned: the prescribed UV is returned with no solve, so this runs on CPU too.
    _, mesh_wp = saddle_graded
    n_vertices = mesh_wp.points.size
    rng = np.random.default_rng(7)
    fixed_uv_np = rng.standard_normal((n_vertices, 2)).astype(np.float32)
    all_indices_np = np.arange(n_vertices, dtype=np.int32)
    fixed_wp = wp.array(all_indices_np, dtype=wp.int32, device=mesh_wp.device)
    fixed_uv_wp = points_to_warp_uv(fixed_uv_np, mesh_wp.device)
    uv_init_wp = wp.zeros(n_vertices, dtype=wp.vec2, device=mesh_wp.device)

    uv_wp = od.parametrization.arap(
        mesh_wp.points, mesh_wp.indices, fixed_wp, fixed_uv_wp, uv_init_wp, max_iterations=10
    )
    assert np.array_equal(uv_wp.numpy(), fixed_uv_np)


@pytest.mark.parametrize("mesh_name", ["saddle_graded", "sphere_irregular_band"])
@pytest.mark.parity("lscm", "igl")
def test_lscm_matches_igl(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A against ``igl.lscm`` with the same two pins, the libigl tutorial-502 convention.

    LSCM is defined only up to the pins, so pinning both sides identically is what makes an
    elementwise comparison meaningful at all -- with a free gauge there would be nothing to compare.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)

    # Pin two boundary vertices to (0, 0) and (1, 0), the libigl tutorial-502 convention.
    pins_np, pins_wp = _two_rim_pins(mesh_wp)
    pins_uv_np = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    pins_uv_wp = points_to_warp_uv(pins_uv_np, mesh_wp.device)

    uv_wp = od.parametrization.lscm(mesh_wp.points, mesh_wp.indices, pins_wp, pins_uv_wp)
    uv_igl, _ = igl.lscm(
        vertices_np, faces_np, pins_np.astype(np.int64), pins_uv_np.astype(np.float64)
    )

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-4, atol=1e-4)


def test_lscm_closed_mesh_matches_igl(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on the degenerate closed-mesh branch, where the area term vanishes.

    With no boundary, ``A = 0`` and the system reduces to ``-repdiag(L, 2)``; igl accepts that
    input, so the branch has a real oracle rather than only an invariant. It is its own test
    because the boundary fixtures never exercise it.
    """
    # Closed mesh: A = 0, Q = -repdiag(L, 2). igl.lscm accepts closed input.
    mesh_tm, mesh_wp = sphere_irregular
    vertices_np, faces_np = mesh_igl(mesh_tm)

    pins_np = np.array([0, 7], dtype=np.int32)
    pins_uv_np = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    pins_wp = wp.array(pins_np, dtype=wp.int32, device=mesh_wp.device)
    pins_uv_wp = points_to_warp_uv(pins_uv_np, mesh_wp.device)

    uv_wp = od.parametrization.lscm(mesh_wp.points, mesh_wp.indices, pins_wp, pins_uv_wp)
    uv_igl, _ = igl.lscm(
        vertices_np, faces_np, pins_np.astype(np.int64), pins_uv_np.astype(np.float64)
    )

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-4, atol=1e-4)


def test_lscm_is_fold_free(saddle_graded: tuple[tm.Trimesh, wp.Mesh]):
    # LSCM of a disk-topology open surface with two pins is conformal and fold-free.
    _, mesh_wp = saddle_graded
    _, pins_wp = _two_rim_pins(mesh_wp)
    pins_uv_wp = wp.array(
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
    )
    uv_wp = od.parametrization.lscm(mesh_wp.points, mesh_wp.indices, pins_wp, pins_uv_wp)
    assert od.parametrization.face_flipped_indices(uv_wp, mesh_wp.indices).numpy().size == 0


def test_lscm_waives_pin_count_below_two_vertices(device: str):
    # The docstring's "unless the mesh has fewer than two vertices" exemption: a single-vertex,
    # zero-face mesh with no pins at all must not raise.
    vertices_wp = wp.array(np.array([[0.0, 0.0, 0.0]]), dtype=wp.vec3, device=device)
    faces_wp = warp_empty(0, wp.int32, device)
    pins_wp = warp_empty(0, wp.int32, device)
    pins_uv_wp = warp_empty(0, wp.vec2, device)
    uv_wp = od.parametrization.lscm(vertices_wp, faces_wp, pins_wp, pins_uv_wp)
    assert np.isfinite(uv_wp.numpy()).all()


def _harmonic_on_quad(device: str) -> wp.array[wp.vec2]:
    vertices, faces = _unit_quad(device)
    boundary = wp.array(np.array([0, 1, 3], dtype=np.int32), dtype=wp.int32, device=device)
    boundary_uv = points_to_warp_uv(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), device)
    return od.parametrization.harmonic(vertices, faces, boundary, boundary_uv)


def _arap_on_quad(device: str) -> wp.array[wp.vec2]:
    vertices, faces = _unit_quad(device)
    fixed = wp.array(np.array([0, 1, 3], dtype=np.int32), dtype=wp.int32, device=device)
    fixed_uv = points_to_warp_uv(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), device)
    uv_init = wp.zeros(4, dtype=wp.vec2, device=device)
    return od.parametrization.arap(vertices, faces, fixed, fixed_uv, uv_init)


def _lscm_on_quad(device: str) -> wp.array[wp.vec2]:
    vertices, faces = _unit_quad(device)
    pins = wp.array(np.array([0, 3], dtype=np.int32), dtype=wp.int32, device=device)
    pins_uv = points_to_warp_uv(np.array([[0.0, 0.0], [1.0, 1.0]]), device)
    return od.parametrization.lscm(vertices, faces, pins, pins_uv)


@pytest.mark.parametrize(
    "solve",
    [_harmonic_on_quad, _arap_on_quad, _lscm_on_quad],
    ids=["harmonic_three_pins", "arap_three_pins", "lscm_two_pins"],
)
def test_solvers_cpu_match_cuda(solve: Callable[[str], wp.array[wp.vec2]]):
    """
    Class A: each CPU solve is the CUDA one, on the two-triangle unit quad.

    Covers the harmonic interior solve, the ARAP local/global alternation and the LSCM free-vertex
    solve, each under the pins its arm names.
    """
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to compare them")

    uv = {device: solve(device).numpy() for device in ("cpu", "cuda:0")}

    assert np.isfinite(uv["cpu"]).all()
    assert np.allclose(uv["cpu"], uv["cuda:0"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("solver", ["arap", "lscm"])
def test_solvers_on_an_empty_mesh(device: str, solver: str):
    """Not a library comparison: no vertices, no pins, an empty UV buffer rather than a raise."""
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = warp_empty(0, wp.int32, device)
    pins_wp = warp_empty(0, wp.int32, device)
    pins_uv_wp = warp_empty(0, wp.vec2, device)
    if solver == "arap":
        uv_init_wp = warp_empty(0, wp.vec2, device)
        uv_wp = od.parametrization.arap(vertices_wp, faces_wp, pins_wp, pins_uv_wp, uv_init_wp)
    else:
        uv_wp = od.parametrization.lscm(vertices_wp, faces_wp, pins_wp, pins_uv_wp)
    assert uv_wp.numpy().size == 0


def _short_boundary_uv(mesh_wp: wp.Mesh) -> tuple[wp.array[wp.int32], wp.array[wp.vec2]]:
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    return boundary_wp, wp.zeros(boundary_wp.size - 1, dtype=wp.vec2, device=mesh_wp.device)


def _uv_init(mesh_wp: wp.Mesh) -> wp.array[wp.vec2]:
    return wp.zeros(mesh_wp.points.size, dtype=wp.vec2, device=mesh_wp.device)


def _pins_and_zero_uv(
    mesh_wp: wp.Mesh, n_pins: int
) -> tuple[wp.array[wp.int32], wp.array[wp.vec2]]:
    pins_wp = wp.array(np.arange(n_pins, dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    return pins_wp, wp.zeros(n_pins, dtype=wp.vec2, device=mesh_wp.device)


_GUARD_CASES: list[tuple[str, Callable[[wp.Mesh], object], str]] = [
    # The index/UV pairs feed a scatter kernel indexed by the indices' own length; a shorter UV
    # buffer would otherwise be an out-of-bounds read.
    (
        "harmonic_short_boundary_uv",
        lambda m: od.parametrization.harmonic(m.points, m.indices, *_short_boundary_uv(m)),
        "same length",
    ),
    # Interior vertices but no pins: the ARAP global system is singular; raised pre-solve.
    (
        "arap_no_fixed",
        lambda m: od.parametrization.arap(
            m.points, m.indices, *_pins_and_zero_uv(m, 0), _uv_init(m)
        ),
        "at least one fixed vertex",
    ),
    (
        "arap_short_fixed_uv",
        lambda m: od.parametrization.arap(m.points, m.indices, *_short_boundary_uv(m), _uv_init(m)),
        "same length",
    ),
    (
        "arap_zero_iterations",
        lambda m: od.parametrization.arap(
            m.points, m.indices, *_circle_boundary(m), _uv_init(m), max_iterations=0
        ),
        "max_iterations",
    ),
    (
        "arap_zero_tolerance",
        lambda m: od.parametrization.arap(
            m.points, m.indices, *_circle_boundary(m), _uv_init(m), tolerance=0.0
        ),
        "tolerance",
    ),
    # Fewer than two pins leaves the similarity-transform null space; raised pre-solve.
    (
        "lscm_no_pins",
        lambda m: od.parametrization.lscm(m.points, m.indices, *_pins_and_zero_uv(m, 0)),
        "at least two pinned vertices",
    ),
    (
        "lscm_one_pin",
        lambda m: od.parametrization.lscm(m.points, m.indices, *_pins_and_zero_uv(m, 1)),
        "at least two pinned vertices",
    ),
    (
        "lscm_short_pinned_uv",
        lambda m: od.parametrization.lscm(
            m.points, m.indices, _two_rim_pins(m)[1], wp.zeros(1, dtype=wp.vec2, device=m.device)
        ),
        "same length",
    ),
]


@pytest.mark.parametrize(
    ("call", "message"),
    [(call, message) for _, call, message in _GUARD_CASES],
    ids=[name for name, _, _ in _GUARD_CASES],
)
def test_solvers_reject_malformed_constraints(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], call: Callable[[wp.Mesh], object], message: str
):
    """Not a parity assert: every guard raises before any solve, so each runs on the CPU too."""
    _, mesh_wp = saddle_graded
    with pytest.raises(ValueError, match=message):
        call(mesh_wp)


def _count_factorizations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the sparse Cholesky factorizations built from here on through ``ordito.linalg``."""
    built: list[int] = []
    factor = od.linalg.sparse_cholesky

    def counted(*args: Any, **kwargs: Any) -> od.cholesky.SparseCholesky:
        result = factor(*args, **kwargs)
        built.append(result.n)
        return result

    monkeypatch.setattr(od.linalg, "sparse_cholesky", counted)
    return built


def _fixed_vertex_call(
    method: str, mesh_wp: wp.Mesh, scale: float
) -> tuple[wp.array[wp.int32], wp.array[wp.vec2]]:
    """Return the fixed set and values a ``method`` call takes: the loop, or two of its vertices."""
    vertices, faces = mesh_wp.points, mesh_wp.indices
    loop = od.boundary.longest_boundary_loop(vertices, faces)
    if method != "lscm":
        circle = od.parametrization.map_vertices_to_circle(vertices, loop).numpy()
        return loop, wp.array(scale * circle, dtype=wp.vec2, device=vertices.device)
    loop_np = loop.numpy()
    pins = wp.array(
        np.array([loop_np[0], loop_np[loop_np.size // 2]], dtype=np.int32),
        dtype=wp.int32,
        device=vertices.device,
    )
    pins_uv = np.array([[0.0, 0.0], [scale, 0.0]], dtype=np.float32)
    return pins, wp.array(pins_uv, dtype=wp.vec2, device=vertices.device)


@pytest.mark.parametrize(
    ("method", "k"), [("harmonic", 1), ("harmonic", 2), ("harmonic", 3), ("tutte", 1), ("lscm", 1)]
)
def test_fixed_vertex_solver_factors_once_for_every_call_fixing_the_same_vertices(
    sphere_irregular_cap: tuple[tm.Trimesh, wp.Mesh],
    method: str,
    k: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ordito against ordito: a prepared factorization serves later calls with no further build.

    ``FixedVertexSolver.factor`` builds one factorization; two calls on the mesh with different
    fixed *values* build none of their own and agree with the ``vertices, faces`` form (which
    iterates below ``k == 3``, verified) to the solves' tolerance. The oracles are the igl tests
    above (``test_harmonic_matches_igl``, ``test_tutte_matches_igl_reference``,
    ``test_lscm_matches_igl``, ``test_polyharmonic_matches_reference``); this pins that the
    prepared path solves the same system.
    """
    _, mesh_wp = sphere_irregular_cap
    vertices, faces = mesh_wp.points, mesh_wp.indices
    function = getattr(od.parametrization, method)
    keywords = {} if method == "lscm" else {"k": k}
    built = _count_factorizations(monkeypatch)
    mesh = od.Trimesh(vertices, faces)
    fixed, _ = _fixed_vertex_call(method, mesh_wp, 1.0)
    assert mesh.fixed_vertex_solver().factor(method, fixed, k=k)
    assert len(built) == 1
    assert mesh.fixed_vertex_solver().nbytes > 0
    for scale in (1.0, 2.5):
        fixed, values = _fixed_vertex_call(method, mesh_wp, scale)
        before = len(built)
        prepared = function(mesh, fixed, values, **keywords).numpy()
        assert len(built) == before
        # The ``vertices, faces`` form factors on its own at ``k >= 3``; that build is not counted.
        direct = function(vertices, faces, fixed, values, **keywords).numpy()
        assert np.allclose(prepared, direct, rtol=1e-5, atol=1e-5 * np.ptp(direct))
    mesh.release_factorizations()
    assert mesh.fixed_vertex_solver().nbytes == 0


def test_fixed_vertex_solver_ignores_a_factorization_of_another_fixed_set(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: a call fixing other vertices than the kept factorization's solves anew.

    The kept system's reduced operator is that of its own fixed set; a call fixing the loop less
    one vertex must not be answered by it (it would solve a different system), so it iterates --
    builds nothing at ``k == 1`` -- and agrees with the ``vertices, faces`` form. The kept
    factorization stays and still serves its own set.
    """
    _, mesh_wp = saddle_graded
    vertices, faces = mesh_wp.points, mesh_wp.indices
    mesh = od.Trimesh(vertices, faces)
    loop, circle = _fixed_vertex_call("harmonic", mesh_wp, 1.0)
    mesh.fixed_vertex_solver().factor("harmonic", loop, k=1)
    built = _count_factorizations(monkeypatch)
    fewer = wp.array(loop.numpy()[1:], dtype=wp.int32, device=vertices.device)
    fewer_uv = wp.array(circle.numpy()[1:], dtype=wp.vec2, device=vertices.device)
    other = od.parametrization.harmonic(mesh, fewer, fewer_uv).numpy()
    direct = od.parametrization.harmonic(vertices, faces, fewer, fewer_uv).numpy()
    assert built == []
    assert np.allclose(other, direct, rtol=1e-5, atol=1e-5 * np.ptp(direct))
    own = od.parametrization.harmonic(mesh, loop, circle).numpy()
    assert built == []
    assert np.allclose(
        own, od.parametrization.harmonic(vertices, faces, loop, circle).numpy(), atol=1e-5
    )


@pytest.mark.parametrize(("k", "discards"), [(1, False), (3, True)])
def test_fixed_vertex_calls_say_when_they_discard_a_factorization(
    sphere_irregular_cap: tuple[tm.Trimesh, wp.Mesh],
    caplog: pytest.LogCaptureFixture,
    k: int,
    discards: bool,
) -> None:
    """
    Not a parity assert: the ``vertices, faces`` form logs at ``INFO`` exactly when it drops one.

    ``k == 3`` is factored from the start, so the call builds a factorization it cannot keep and
    says so; ``k == 1`` iterates on this mesh and builds none. Given a ``Trimesh``, the same
    ``k == 3`` call keeps it on the mesh and logs nothing.
    """
    _, mesh_wp = sphere_irregular_cap
    vertices, faces = mesh_wp.points, mesh_wp.indices
    loop, circle = _fixed_vertex_call("harmonic", mesh_wp, 1.0)
    with caplog.at_level(logging.INFO, logger="ordito.parametrization"):
        od.parametrization.harmonic(vertices, faces, loop, circle, k=k)
    infos = [r for r in caplog.records if r.levelno == logging.INFO and "Trimesh" in r.message]
    assert len(infos) == (1 if discards else 0)
    caplog.clear()
    mesh = od.Trimesh(vertices, faces)
    with caplog.at_level(logging.INFO, logger="ordito.parametrization"):
        od.parametrization.harmonic(mesh, loop, circle, k=k)
    assert [r for r in caplog.records if r.levelno == logging.INFO] == []
    assert (mesh.fixed_vertex_solver().nbytes > 0) == discards


@pytest.mark.parametrize(("method", "k"), [("arap", 1), ("harmonic", 0), ("lscm", 2)])
def test_fixed_vertex_solver_rejects_an_off_menu_method_or_power(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], method: str, k: int
) -> None:
    """Not a parity assert: ``factor`` and ``solve`` raise ``ValueError`` naming the argument."""
    _, mesh_wp = saddle_graded
    solver = od.Trimesh(mesh_wp.points, mesh_wp.indices).fixed_vertex_solver()
    loop, _ = _fixed_vertex_call("harmonic", mesh_wp, 1.0)
    with pytest.raises(ValueError, match=r"method|k="):
        solver.factor(method, loop, k=k)
    mask = wp.zeros(mesh_wp.points.size, dtype=wp.bool, device=mesh_wp.points.device)
    values = odt.as_array2d(
        wp.zeros((2, mesh_wp.points.size), dtype=wp.float64, device=mesh_wp.points.device),
        wp.float64,
    )
    with pytest.raises(ValueError, match=r"method|k="):
        solver.solve(method, mask, values, k=k)


@pytest.mark.parametrize(
    ("method", "k"), [("harmonic", 1), ("harmonic", 2), ("tutte", 1), ("lscm", 1)]
)
def test_solver_direct_factors_at_once_and_keeps_it_on_the_mesh(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh],
    method: str,
    k: int,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ordito against ordito: ``solver="direct"`` is ``factor`` folded into the first call.

    On a ``Trimesh`` the first call factors (one build) and keeps it, so later default-``solver``
    calls fixing the same vertices build nothing; each agrees with the ``vertices, faces`` form's
    verified iteration to the solves' tolerance (the igl oracles are the tests above). The
    ``vertices, faces`` form under ``"direct"`` factors too, and says at ``INFO`` that it drops it.
    """
    _, mesh_wp = saddle_graded
    vertices, faces = mesh_wp.points, mesh_wp.indices
    function = getattr(od.parametrization, method)
    keywords = {} if method == "lscm" else {"k": k}
    fixed, values = _fixed_vertex_call(method, mesh_wp, 1.0)
    iterated = function(vertices, faces, fixed, values, **keywords).numpy()
    built = _count_factorizations(monkeypatch)
    mesh = od.Trimesh(vertices, faces)
    direct = function(mesh, fixed, values, solver="direct", **keywords).numpy()
    assert len(built) == 1
    again = function(mesh, fixed, values, **keywords).numpy()
    assert len(built) == 1
    for result in (direct, again):
        assert np.allclose(result, iterated, rtol=1e-5, atol=1e-5 * np.ptp(iterated))
    with caplog.at_level(logging.INFO, logger="ordito.parametrization"):
        function(vertices, faces, fixed, values, solver="direct", **keywords)
    assert len(built) == 2
    assert [r for r in caplog.records if r.levelno == logging.INFO and "Trimesh" in r.message]


@pytest.mark.parametrize("method", ["harmonic", "tutte", "lscm"])
def test_fixed_vertex_maps_reject_an_off_menu_solver(
    saddle_graded: tuple[tm.Trimesh, wp.Mesh], method: str
) -> None:
    """Not a parity assert: ``solver`` outside the menu raises ``ValueError`` naming it."""
    _, mesh_wp = saddle_graded
    fixed, values = _fixed_vertex_call(method, mesh_wp, 1.0)
    function = getattr(od.parametrization, method)
    with pytest.raises(ValueError, match="solver"):
        function(mesh_wp.points, mesh_wp.indices, fixed, values, solver="cholesky")
    with pytest.raises(ValueError, match="solver"):
        function(od.Trimesh(mesh_wp.points, mesh_wp.indices), fixed, values, solver="cholesky")
