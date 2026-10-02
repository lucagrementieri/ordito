from __future__ import annotations

from typing import cast

import igl
import numpy as np
import pytest
import scipy.sparse
import trimesh as tm
import warp as wp

import ordito as od
from tests.conversions import mesh_igl, numpy_to_warp_uv, points_to_warp_uv, warp_empty


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


def test_face_flipped_mask_index_consistency(device: str):
    rng = np.random.default_rng(1)
    vertices_np, faces_np = _random_2d_mesh(rng, n_faces=32)
    vertices_wp, faces_wp = numpy_to_warp_uv(vertices_np, faces_np, device)

    mask_wp = od.parametrization.face_flipped_mask(vertices_wp, faces_wp)
    assert np.array_equal(
        od.parametrization.face_flipped_indices(vertices_wp, faces_wp).numpy(),
        np.flatnonzero(mask_wp.numpy()),
    )


def test_flipped_faces_all_and_none(device: str):
    # A single CCW (positive-area) triangle: not flipped.
    ccw_np = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, 2]], dtype=np.int64)
    vertices_wp, faces_wp = numpy_to_warp_uv(ccw_np, faces_np, device)
    assert od.parametrization.face_flipped_indices(vertices_wp, faces_wp).numpy().size == 0
    assert not od.parametrization.face_flipped_mask(vertices_wp, faces_wp).numpy().any()

    # Reverse the winding of every triangle: all flipped.
    cw_faces_np = faces_np[:, ::-1].copy()
    vertices_wp, cw_faces_wp = numpy_to_warp_uv(ccw_np, cw_faces_np, device)
    assert np.array_equal(
        od.parametrization.face_flipped_indices(vertices_wp, cw_faces_wp).numpy(),
        np.arange(cw_faces_np.shape[0], dtype=np.int64),
    )
    assert od.parametrization.face_flipped_mask(vertices_wp, cw_faces_wp).numpy().all()


def test_flipped_faces_degenerate_not_flagged(device: str):
    # Collinear (zero-area) triangle: strict "< 0" means it is not flagged.
    collinear_np = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, 2]], dtype=np.int64)
    vertices_wp, faces_wp = numpy_to_warp_uv(collinear_np, faces_np, device)
    assert not od.parametrization.face_flipped_mask(vertices_wp, faces_wp).numpy().any()
    assert od.parametrization.face_flipped_indices(vertices_wp, faces_wp).numpy().size == 0


def test_flipped_faces_empty_mesh(device: str):
    vertices_wp = warp_empty(0, wp.vec2, device)
    faces_wp = warp_empty(0, wp.int32, device)
    assert od.parametrization.face_flipped_mask(vertices_wp, faces_wp).numpy().size == 0
    assert od.parametrization.face_flipped_indices(vertices_wp, faces_wp).numpy().size == 0


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
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
def test_graph_laplacian_matches_igl(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class B: igl has no ``graph_laplacian``, so the reference is assembled from its adjacency.

    The named transform is ``A - diag(rowsum(A))`` over ``igl.adjacency_matrix``: the definition of
    the umbrella operator, built on the reference side. Note igl's adjacency is sized by
    ``F.max() + 1`` rather than ``len(V)`` (section 6), which is why this runs on ``hemisphere``,
    where every vertex is referenced.
    """
    mesh_tm, mesh_wp = hemisphere
    _, faces_np = mesh_igl(mesh_tm)
    n_vertices = mesh_wp.points.size

    operator_wp = od.laplacian.graph_laplacian(mesh_wp.points, mesh_wp.indices)
    operator_dense = scipy.sparse.csr_matrix(
        (operator_wp.values.numpy(), operator_wp.columns.numpy(), operator_wp.offsets.numpy()),
        shape=(n_vertices, n_vertices),
    ).toarray()

    adjacency_igl = igl.adjacency_matrix(faces_np).toarray().astype(np.float64)
    laplacian_igl = adjacency_igl - np.diag(adjacency_igl.sum(axis=1))

    assert np.allclose(operator_dense, laplacian_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
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

    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)

    uv_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp
    )
    uv_igl = igl.harmonic(vertices_np, faces_np, boundary_np, boundary_uv_np, 1)

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_biharmonic_matches_reference(request: pytest.FixtureRequest, mesh_name: str):
    # k=2 (biharmonic). igl.harmonic's default mass is Voronoi, but ordito uses the barycentric
    # lumped mass, so compare against a barycentric-mass biharmonic solved directly in SciPy.
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = mesh_igl(mesh_tm)
    n_vertices = mesh_wp.points.size

    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)

    uv_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, k=2
    )

    laplacian = igl.cotmatrix(vertices_np, faces_np)
    mass_inv = scipy.sparse.diags(
        1.0
        / np.asarray(
            igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
        )
    )
    # scipy-stubs type the product as ``_spbase | ArrayND``, neither of which it gives ``tocsc``.
    biharmonic = ((-laplacian) @ mass_inv @ (-laplacian)).tocsc()  # pyright: ignore[reportAttributeAccessIssue]
    interior = np.setdiff1d(np.arange(n_vertices), boundary_np)
    solution = scipy.sparse.linalg.spsolve(
        biharmonic[np.ix_(interior, interior)],
        -(biharmonic[np.ix_(interior, boundary_np)] @ boundary_uv_np),
    )
    uv_ref = np.zeros((n_vertices, 2))
    uv_ref[boundary_np] = boundary_uv_np
    uv_ref[interior] = solution

    assert np.allclose(uv_wp.numpy(), uv_ref, rtol=1e-4, atol=1e-4)


def test_biharmonic_is_deterministic(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    # Regression guard for operator-assembly nondeterminism: the float64-native biharmonic operator
    # must give the same result across repeated calls. The original defect sized a rebuild's triplet
    # buffers by ``BsrMatrix.nnz`` (the capacity, not the entry count) and so fed the uninitialized
    # tail to ``bsr_from_triplets``, manifesting as ~1e22 / NaN corruption; a tight tolerance (well
    # above conjugate-gradient's ~1e-8 atomic last-ULP jitter) reliably catches a regression.
    _, mesh_wp = hemisphere
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    first = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, k=2
    ).numpy()
    for _ in range(5):
        again = od.parametrization.harmonic(
            mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, k=2
        ).numpy()
        assert np.allclose(again, first, rtol=1e-5, atol=1e-5)


def test_harmonic_cpu_matches_cuda():
    """Class A: the CPU interior solve is the CUDA one (two-triangle quad, three pinned corners)."""
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to compare them")

    uv = {}
    for device in ("cpu", "cuda:0"):
        vertices = wp.array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
            dtype=wp.vec3,
            device=device,
        )
        faces = wp.array(
            np.array([0, 1, 2, 1, 3, 2], dtype=np.int32), dtype=wp.int32, device=device
        )
        boundary = wp.array(np.array([0, 1, 3], dtype=np.int32), dtype=wp.int32, device=device)
        boundary_uv = wp.array(
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            dtype=wp.vec2,
            device=device,
        )
        uv[device] = od.parametrization.harmonic(vertices, faces, boundary, boundary_uv).numpy()

    assert np.isfinite(uv["cpu"]).all()
    assert np.allclose(uv["cpu"], uv["cuda:0"], rtol=1e-5, atol=1e-5)


def test_harmonic_boundary_uv_length_mismatch_raises(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    # boundary_indices/boundary_uv feed a scatter kernel indexed by boundary_indices' own length;
    # a shorter boundary_uv would otherwise be an out-of-bounds read.
    _, mesh_wp = hemisphere
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    short_uv = wp.zeros(boundary_wp.size - 1, dtype=wp.vec2, device=mesh_wp.device)
    with pytest.raises(ValueError, match="same length"):
        od.parametrization.harmonic(mesh_wp.points, mesh_wp.indices, boundary_wp, short_uv)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
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

    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
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


def test_tutte_disk_is_fold_free(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    # A disk-topology mesh with a convex (circle) boundary yields a bijective, fold-free Tutte map.
    _, mesh_wp = hemisphere
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
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


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
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
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)
    uv_init_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp
    )
    uv_init_np = uv_init_wp.numpy().astype(np.float64)

    uv_wp = od.parametrization.arap(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, uv_init_wp, max_iterations=10
    )
    uv_igl = _arap_igl(vertices_np, faces_np, boundary_np, boundary_uv_np, uv_init_np, 10)

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-4, atol=1e-4)


def test_arap_free_boundary_matches_igl(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on the free-boundary branch: only two vertices pinned, the rest of the rim moving.

    The fully-pinned case above cannot see a bug in the boundary rows of the system, since there are
    none to solve; this is the same comparison with 4 iterations and almost the whole rim free.
    """
    # Pin only two boundary vertices to their harmonic UV; the rest of the boundary is free.
    mesh_tm, mesh_wp = hemisphere
    vertices_np, faces_np = mesh_igl(mesh_tm)

    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    uv_init_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp
    )
    uv_init_np = uv_init_wp.numpy().astype(np.float64)

    loop_np = boundary_wp.numpy()
    fixed_np = np.array([loop_np[0], loop_np[len(loop_np) // 2]], dtype=np.int32)
    fixed_uv_np = uv_init_np[fixed_np]
    fixed_wp = wp.array(fixed_np, dtype=wp.int32, device=mesh_wp.device)
    fixed_uv_wp = points_to_warp_uv(fixed_uv_np, mesh_wp.device)

    uv_wp = od.parametrization.arap(
        mesh_wp.points, mesh_wp.indices, fixed_wp, fixed_uv_wp, uv_init_wp, max_iterations=4
    )
    uv_igl = _arap_igl(vertices_np, faces_np, fixed_np, fixed_uv_np, uv_init_np, 4)

    # float32-UV drift vs igl's float64 grows slowly; four iterations stays well under 1e-3.
    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-3, atol=1e-3)


def test_arap_default_tolerance_tracks_a_tight_solve(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    # ``arap`` defaults its inner CG to 1e-7 rather than the 1e-8 the other solvers use: its global
    # solves are inner steps of a truncated outer iteration. Guard that the looser default still
    # tracks a tight solve two orders below it, far inside the 1e-4 gate the igl oracles use.
    _, mesh_wp = hemisphere
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    uv_init_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp
    )

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


def test_arap_fixed_vertices_pinned(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    # The pinned rows must equal the prescribed UV exactly (they are re-enforced every iteration).
    _, mesh_wp = hemisphere
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    uv_init_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp
    )

    uv_wp = od.parametrization.arap(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, uv_init_wp, max_iterations=10
    )
    assert np.array_equal(uv_wp.numpy()[boundary_wp.numpy()], boundary_uv_wp.numpy())


def test_arap_disk_is_finite(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = hemisphere
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    uv_init_wp = od.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp
    )
    uv_wp = od.parametrization.arap(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, uv_init_wp, max_iterations=10
    )
    assert np.isfinite(uv_wp.numpy()).all()


def test_arap_all_vertices_fixed(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    # Every vertex pinned: the prescribed UV is returned with no solve, so this runs on CPU too.
    _, mesh_wp = hemisphere
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


def test_arap_empty_fixed_raises(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    # Interior vertices but no pins: the ARAP global system is singular; raised pre-solve (CPU-ok).
    _, mesh_wp = hemisphere
    empty_fixed = warp_empty(0, wp.int32, mesh_wp.device)
    empty_uv = warp_empty(0, wp.vec2, mesh_wp.device)
    uv_init = wp.zeros(mesh_wp.points.size, dtype=wp.vec2, device=mesh_wp.device)
    with pytest.raises(ValueError, match="at least one fixed vertex"):
        od.parametrization.arap(mesh_wp.points, mesh_wp.indices, empty_fixed, empty_uv, uv_init)


def test_arap_fixed_uv_length_mismatch_raises(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    # fixed_indices/fixed_uv feed a scatter kernel indexed by fixed_indices' own length; a shorter
    # fixed_uv would otherwise be an out-of-bounds read.
    _, mesh_wp = hemisphere
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    short_uv = wp.zeros(boundary_wp.size - 1, dtype=wp.vec2, device=mesh_wp.device)
    uv_init = wp.zeros(mesh_wp.points.size, dtype=wp.vec2, device=mesh_wp.device)
    with pytest.raises(ValueError, match="same length"):
        od.parametrization.arap(mesh_wp.points, mesh_wp.indices, boundary_wp, short_uv, uv_init)


def test_arap_bad_iterations_raises(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = hemisphere
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    uv_init = wp.zeros(mesh_wp.points.size, dtype=wp.vec2, device=mesh_wp.device)
    with pytest.raises(ValueError, match="max_iterations"):
        od.parametrization.arap(
            mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, uv_init, max_iterations=0
        )


def test_arap_bad_tolerance_raises(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = hemisphere
    boundary_wp = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = od.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    uv_init = wp.zeros(mesh_wp.points.size, dtype=wp.vec2, device=mesh_wp.device)
    with pytest.raises(ValueError, match="tolerance"):
        od.parametrization.arap(
            mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, uv_init, tolerance=0.0
        )


def test_arap_cpu_matches_cuda():
    """Class A: the CPU local/global alternation is the CUDA one (quad, three pinned corners)."""
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to compare them")

    uv = {}
    for device in ("cpu", "cuda:0"):
        vertices = wp.array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
            dtype=wp.vec3,
            device=device,
        )
        faces = wp.array(
            np.array([0, 1, 2, 1, 3, 2], dtype=np.int32), dtype=wp.int32, device=device
        )
        fixed = wp.array(np.array([0, 1, 3], dtype=np.int32), dtype=wp.int32, device=device)
        fixed_uv = wp.array(
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            dtype=wp.vec2,
            device=device,
        )
        uv_init = wp.zeros(4, dtype=wp.vec2, device=device)
        uv[device] = od.parametrization.arap(vertices, faces, fixed, fixed_uv, uv_init).numpy()

    assert np.isfinite(uv["cpu"]).all()
    assert np.allclose(uv["cpu"], uv["cuda:0"], rtol=1e-5, atol=1e-5)


def test_arap_empty_mesh(device: str):
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = warp_empty(0, wp.int32, device)
    fixed_wp = warp_empty(0, wp.int32, device)
    fixed_uv_wp = warp_empty(0, wp.vec2, device)
    uv_init_wp = warp_empty(0, wp.vec2, device)
    uv_wp = od.parametrization.arap(vertices_wp, faces_wp, fixed_wp, fixed_uv_wp, uv_init_wp)
    assert uv_wp.numpy().size == 0


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
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
    loop_np = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices).numpy()
    pins_np = np.array([loop_np[0], loop_np[len(loop_np) // 2]], dtype=np.int32)
    pins_uv_np = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    pins_wp = wp.array(pins_np, dtype=wp.int32, device=mesh_wp.device)
    pins_uv_wp = points_to_warp_uv(pins_uv_np, mesh_wp.device)

    uv_wp = od.parametrization.lscm(mesh_wp.points, mesh_wp.indices, pins_wp, pins_uv_wp)
    uv_igl, _ = igl.lscm(
        vertices_np, faces_np, pins_np.astype(np.int64), pins_uv_np.astype(np.float64)
    )

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-4, atol=1e-4)


def test_lscm_closed_mesh_matches_igl(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on the degenerate closed-mesh branch, where the area term vanishes.

    With no boundary, ``A = 0`` and the system reduces to ``-repdiag(L, 2)``; igl accepts that
    input, so the branch has a real oracle rather than only an invariant. It is its own test
    because the boundary fixtures never exercise it.
    """
    # Closed mesh: A = 0, Q = -repdiag(L, 2). igl.lscm accepts closed input.
    mesh_tm, mesh_wp = icosahedron
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


def test_lscm_is_fold_free(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    # LSCM of a disk-topology open surface with two pins is conformal and fold-free.
    _, mesh_wp = hemisphere
    loop_np = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices).numpy()
    pins_wp = wp.array(
        np.array([loop_np[0], loop_np[len(loop_np) // 2]], dtype=np.int32),
        dtype=wp.int32,
        device=mesh_wp.device,
    )
    pins_uv_wp = wp.array(
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
    )
    uv_wp = od.parametrization.lscm(mesh_wp.points, mesh_wp.indices, pins_wp, pins_uv_wp)
    assert od.parametrization.face_flipped_indices(uv_wp, mesh_wp.indices).numpy().size == 0


@pytest.mark.parametrize("n_pins", [0, 1])
def test_lscm_too_few_pins_raises(hemisphere: tuple[tm.Trimesh, wp.Mesh], n_pins: int):
    # Fewer than two pins leaves the similarity-transform null space; raised pre-solve (CPU-safe).
    _, mesh_wp = hemisphere
    pins_wp = wp.array(np.arange(n_pins, dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    pins_uv_wp = wp.array(
        np.zeros((n_pins, 2), dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
    )
    with pytest.raises(ValueError, match="at least two pinned vertices"):
        od.parametrization.lscm(mesh_wp.points, mesh_wp.indices, pins_wp, pins_uv_wp)


def test_lscm_waives_pin_count_below_two_vertices(device: str):
    # The docstring's "unless the mesh has fewer than two vertices" exemption: a single-vertex,
    # zero-face mesh with no pins at all must not raise.
    vertices_wp = wp.array(np.array([[0.0, 0.0, 0.0]]), dtype=wp.vec3, device=device)
    faces_wp = warp_empty(0, wp.int32, device)
    pins_wp = warp_empty(0, wp.int32, device)
    pins_uv_wp = warp_empty(0, wp.vec2, device)
    uv_wp = od.parametrization.lscm(vertices_wp, faces_wp, pins_wp, pins_uv_wp)
    assert np.isfinite(uv_wp.numpy()).all()


def test_lscm_pinned_uv_length_mismatch_raises(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    # pinned_indices/pinned_uv feed a scatter kernel indexed by pinned_indices' own length; a
    # shorter pinned_uv would otherwise be an out-of-bounds read.
    _, mesh_wp = hemisphere
    loop_np = od.boundary.longest_boundary_loop(mesh_wp.points, mesh_wp.indices).numpy()
    pins_wp = wp.array(
        np.array([loop_np[0], loop_np[len(loop_np) // 2]], dtype=np.int32),
        dtype=wp.int32,
        device=mesh_wp.device,
    )
    short_uv = wp.zeros(1, dtype=wp.vec2, device=mesh_wp.device)
    with pytest.raises(ValueError, match="same length"):
        od.parametrization.lscm(mesh_wp.points, mesh_wp.indices, pins_wp, short_uv)


def test_lscm_cpu_matches_cuda():
    """Class A: the CPU free-vertex solve is the CUDA one (two-triangle quad, two pins)."""
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to compare them")

    uv = {}
    for device in ("cpu", "cuda:0"):
        vertices = wp.array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
            dtype=wp.vec3,
            device=device,
        )
        faces = wp.array(
            np.array([0, 1, 2, 1, 3, 2], dtype=np.int32), dtype=wp.int32, device=device
        )
        pins = wp.array(np.array([0, 3], dtype=np.int32), dtype=wp.int32, device=device)
        pins_uv = wp.array(
            np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), dtype=wp.vec2, device=device
        )
        uv[device] = od.parametrization.lscm(vertices, faces, pins, pins_uv).numpy()

    assert np.isfinite(uv["cpu"]).all()
    assert np.allclose(uv["cpu"], uv["cuda:0"], rtol=1e-5, atol=1e-5)


def test_lscm_empty_mesh(device: str):
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = warp_empty(0, wp.int32, device)
    pins_wp = warp_empty(0, wp.int32, device)
    pins_uv_wp = warp_empty(0, wp.vec2, device)
    uv_wp = od.parametrization.lscm(vertices_wp, faces_wp, pins_wp, pins_uv_wp)
    assert uv_wp.numpy().size == 0
