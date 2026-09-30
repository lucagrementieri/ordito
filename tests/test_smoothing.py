"""Regression tests for ``triwarp.smoothing`` against trimesh / igl (CPU reference)."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import igl
import numpy as np
import numpy.typing as npt
import pymeshlab as ml
import pytest
import pytorch3d.ops as p3d_ops
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh as tm
import trimesh.smoothing as tms
import warp as wp
import warp.sparse as wps
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import triwarp as tw
import triwarp.typing as twt
from tests.conversions import (
    meshlib_bitset_to_numpy,
    meshlib_to_trimesh,
    numpy_to_meshlib,
    numpy_to_meshlib_bitset,
    numpy_to_warp,
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pytorch3d,
    trimesh_to_warp,
    warp_empty,
    warp_to_trimesh,
)
from triwarp.kernels import smoothing as kernel_smoothing

if TYPE_CHECKING:
    from typing_extensions import Buffer


def _meshlab_umbrella(mesh_tm: tm.Trimesh, device: str) -> twt.BsrMatrix[wp.float32]:
    """
    Assemble MeshLab's face-count-weighted umbrella as a row-stochastic operator.

    Each neighbour is weighted by the number of faces its edge shares (2 for an interior edge, 1 on
    a boundary) and the vertex itself by 1, then the row is normalized. Feeding this to triwarp's
    ``laplacian_operator=`` parameter is what makes the comparison Class B rather than an exemption.
    """
    shared: dict[tuple[int, int], int] = {}
    for face_np in mesh_tm.faces:
        for start, end in ((0, 1), (1, 2), (2, 0)):
            key = (int(face_np[start]), int(face_np[end]))
            shared[key] = shared.get(key, 0) + 1
            shared[key[1], key[0]] = shared.get((key[1], key[0]), 0) + 1

    n_vertices = mesh_tm.vertices.shape[0]
    row_sum_np = np.ones(n_vertices)
    for (row, _column), count in shared.items():
        row_sum_np[row] += count
    rows_np = np.arange(n_vertices, dtype=np.int32)
    columns_np = rows_np.copy()
    values_np = 1.0 / row_sum_np
    entries_np = np.array(list(shared.items()), dtype=object)
    neighbor_rows = np.array([row for (row, _column), _count in entries_np], dtype=np.int32)
    neighbor_columns = np.array([column for (_row, column), _count in entries_np], dtype=np.int32)
    neighbor_values = (
        np.array([count for _key, count in entries_np], dtype=np.float64)
        / row_sum_np[neighbor_rows]
    )

    umbrella = wps.bsr_from_triplets(
        n_vertices,
        n_vertices,
        wp.array(np.concatenate((rows_np, neighbor_rows)), dtype=wp.int32, device=device),
        wp.array(np.concatenate((columns_np, neighbor_columns)), dtype=wp.int32, device=device),
        wp.array(
            np.concatenate((values_np, neighbor_values)).astype(np.float32),
            dtype=wp.float32,
            device=device,
        ),
    )
    assert twt.has_blocks(umbrella, wp.float32)
    return umbrella


def _noisy_icosphere(subdivisions: int = 2, sigma: float = 0.01, seed: int = 0) -> tm.Trimesh:
    """Build a closed, evenly tessellated mesh carrying noise for a smoother to remove."""
    mesh_tm = tm.creation.icosphere(subdivisions=subdivisions)
    mesh_tm.vertices = mesh_tm.vertices + np.random.default_rng(seed).normal(
        0.0, sigma, mesh_tm.vertices.shape
    )
    return mesh_tm


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_filter_laplacian_explicit(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points, mesh_wp.indices, iterations=8, volume_constraint=False
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_laplacian(mesh_ref, iterations=8, volume_constraint=False)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


def test_filter_laplacian_volume_constraint(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron

    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points, mesh_wp.indices, iterations=8, volume_constraint=True
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_laplacian(mesh_ref, iterations=8, volume_constraint=True)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("iterations", [1, 3, 10])
@pytest.mark.parity("filter_laplacian_integration", "pymeshlab")
def test_filter_laplacian_matches_pymeshlab(device: str, iterations: int) -> None:
    """
    Class B: exact, once MeshLab's own umbrella is passed through ``laplacian_operator=``.

    Three named transforms. ``cotangentweight=False`` selects the uniform scheme (the benchmark
    passes it too); ``lamb=1.0`` matches MeshLab's full step, which replaces the vertex by the
    weighted mean rather than interpolating toward it; and the operator is
    [`_meshlab_umbrella`][tests.test_smoothing._meshlab_umbrella] rather than triwarp's default,
    because MeshLab's uniform Laplacian is *not* the 1-ring mean.

    Measured agreement **6.5e-08 / 1.2e-07 / 2.6e-07** at 1 / 3 / 10 passes -- float32 accumulation
    noise -- against **5.0e-03** at one pass with the plain 1-ring mean. So this is the assert that
    pins which of MeshLab's two umbrellas this filter uses; with the wrong one it fails by 4 orders
    of magnitude. This also exercises the pluggable-operator path the benchmark's
    ``filter_laplacian_integration`` group times.

    A closed fixture, because the ``1`` self-weight and per-face neighbour counts were solved for on
    one; the boundary generalization (a shared count of 1) is written but not asserted here.
    """
    mesh_tm = _noisy_icosphere()
    mesh_wp = trimesh_to_warp(mesh_tm, device)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.apply_coord_laplacian_smoothing(
        stepsmoothnum=iterations, cotangentweight=False, boundary=True
    )

    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points,
        mesh_wp.indices,
        lamb=1.0,
        iterations=iterations,
        volume_constraint=False,
        laplacian_operator=_meshlab_umbrella(mesh_tm, device),
    )
    assert np.allclose(
        smoothed_wp.numpy(), meshset_pml.current_mesh().vertex_matrix(), rtol=1e-5, atol=1e-5
    )


def test_filter_laplacian_implicit(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron

    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points,
        mesh_wp.indices,
        lamb=0.5,
        iterations=6,
        implicit_time_integration=True,
        volume_constraint=False,
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_laplacian(
        mesh_ref, lamb=0.5, iterations=6, implicit_time_integration=True, volume_constraint=False
    )

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus"])
@pytest.mark.parametrize("lamb", [0.5, 5.0, 50.0])
def test_filter_laplacian_implicit_on_open_meshes(
    request: pytest.FixtureRequest, mesh_name: str, lamb: float
) -> None:
    """
    Class A: the implicit filter against trimesh, on closed *and* open meshes, at every ``lamb``.

    The uniform operator is built from directed ``mesh.edges`` (trimesh's own convention), so on a
    mesh with a boundary the backward-Euler system is not symmetric. A conjugate-gradient solve of
    it -- what this used to be -- matched trimesh on the closed icosahedron the older test above
    uses and returned vertices ~1e4 off on ``hemisphere`` at ``lamb = 5``; the open arms are the
    point. ``lamb = 50`` pushes the fixed-point step count past its cap, so it exercises the
    BiCGSTAB fallback.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points,
        mesh_wp.indices,
        lamb=lamb,
        iterations=3,
        implicit_time_integration=True,
        volume_constraint=False,
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_laplacian(
        mesh_ref, lamb=lamb, iterations=3, implicit_time_integration=True, volume_constraint=False
    )
    # Non-vacuity: the filter must actually move the mesh.
    assert np.abs(mesh_ref.vertices - mesh_tm.vertices).max() > 1e-3
    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-4, atol=1e-4)


def test_filter_laplacian_pluggable_operator(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = half_torus
    operator = tw.laplacian.laplacian(mesh_wp.points, mesh_wp.indices, equal_weight=False)

    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points,
        mesh_wp.indices,
        iterations=6,
        volume_constraint=False,
        laplacian_operator=operator,
    )
    mesh_ref = mesh_tm.copy()
    operator_tm = tms.laplacian_calculation(mesh_ref, equal_weight=False)
    tms.filter_laplacian(
        mesh_ref, iterations=6, volume_constraint=False, laplacian_operator=operator_tm
    )

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


def test_filter_laplacian_implicit_duplicate_built_operator(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    A duplicate-built operator gives the same result as its compact form.

    ``cotmatrix`` sizes its storage for six keys per face plus one per vertex, so ``operator.nnz``
    -- that *capacity*, recorded until a sync -- overshoots ``nnz_sync()`` by ~1.85x. The implicit
    system used to size its ``wp.empty`` triplet buffers by ``nnz``, leaving that gap uninitialized
    for the triplet build to read back as triplets: out-of-range garbage indices are dropped
    silently, but any landing in ``[0, n)`` accumulate a junk value into a real entry. Measured
    ``‖values‖ = 1.1e13`` against a correct 84.3, and every vertex ``NaN`` end to end. Rebuilt
    sliced to ``nnz_sync()`` the same operator is compact, so the two must agree.
    """
    _, mesh_wp = half_torus
    # Two independent builds of the same operator. ``nnz_sync()`` repairs the stale ``nnz`` cache
    # *in place*, so measuring the capacity on one build would hand the filter a repaired matrix and
    # the test would pass whatever the implementation does -- the operator under test has to be a
    # build nothing has synced.
    unsynced = tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices, dtype=wp.float32)
    reference = tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices, dtype=wp.float32)
    capacity = int(reference.nnz)
    nnz = reference.nnz_sync()
    # Guard the guard: without a real capacity gap this test compares a matrix with itself.
    assert capacity > nnz
    compact = wps.bsr_from_triplets(
        int(reference.nrow),
        int(reference.ncol),
        reference.uncompress_rows()[:nnz],
        reference.columns[:nnz],
        reference.values[:nnz],
        prune_numerical_zeros=False,
    )
    assert twt.has_blocks(compact, wp.float32)

    # Random garbage is almost always out of range, and ``bsr_from_triplets`` drops an out-of-range
    # index silently -- which is exactly why the defect stayed invisible. Leaving plausible in-range
    # indices in the memory pool makes the unwritten tail reachable, so the assertions below have
    # something to catch.
    n_vertices = mesh_wp.points.size
    for _ in range(6):
        _junk = (
            wp.full(capacity + n_vertices, 7, dtype=wp.int32, device=mesh_wp.points.device),
            wp.full(capacity + n_vertices, 11, dtype=wp.int32, device=mesh_wp.points.device),
            wp.full(capacity + n_vertices, 1.0e9, dtype=wp.float64, device=mesh_wp.points.device),
        )
        del _junk
    wp.synchronize()

    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points,
        mesh_wp.indices,
        iterations=2,
        implicit_time_integration=True,
        volume_constraint=False,
        laplacian_operator=unsynced,
    )
    compact_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points,
        mesh_wp.indices,
        iterations=2,
        implicit_time_integration=True,
        volume_constraint=False,
        laplacian_operator=compact,
    )

    assert np.all(np.isfinite(smoothed_wp.numpy()))
    assert np.allclose(smoothed_wp.numpy(), compact_wp.numpy(), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Region smoothing solves vs MeshLib (positionVertsSmoothly / SharpBd)
# ---------------------------------------------------------------------------


def test_inflate_grows_the_volume_along_the_normals(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: MeshLib's ``inflate`` is a different operation, measured.

    It is the only reference that has one, and it cannot be compared. With **every** vertex selected
    -- the whole-mesh inflation this function performs -- it collapses the sphere to the origin at
    every pressure probed (1e-4, 1e-3, 1e-2, 0.05, 0.5: max radius 0.0000 each time), because its
    implicit solve takes the unselected vertices as its Dirichlet condition, and selecting all of
    them leaves the system with no anchor. Given a **region** it runs, but solves a different
    problem: on a 223-vertex cap of ``icosphere(3)`` the volume goes 4.153 to 2.843 -- *below* the
    input, since the implicit Laplacian flattens the cap onto its pinned rim before the pressure
    pushes back -- and then rises with pressure (2.843 / 2.847 / 2.865 at 0.001 / 0.01 / 0.05). Only
    the monotonicity is shared, and that is too weak to be a parity claim.

    So the properties carry it, and each excludes something a volume alone would not:

    * **volume monotone in the pressure**, which excludes a flow that smooths without inflating --
      measured 4.153 unchanged at zero, 4.540 at 0.1 mean-edge and 6.328 at 0.5;
    * **displacement normal-aligned**, mean cosine 0.959 and 0.998 at those pressures, ruling out
      growing the volume by shearing;
    * **watertight and free of self-intersections**, which is what a caller depends on;
    * **pressure zero is the identity in volume**, which is the tightest of the four: any change
      there would be the displacement leaking, since ``filter_laplacian``'s volume constraint is on.
    """
    mesh_tm, mesh_wp = icosphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    mean_edge = float(
        np.linalg.norm(
            mesh_tm.vertices[mesh_tm.edges[:, 0]] - mesh_tm.vertices[mesh_tm.edges[:, 1]], axis=1
        ).mean()
    )
    normals_np = tw.vertices.vertex_normals(vertices_wp, faces_wp).numpy()

    volumes = []
    for scale in (0.0, 0.1, 0.5):
        inflated_wp = tw.smoothing.inflate(vertices_wp, faces_wp, scale * mean_edge)
        inflated_tm = warp_to_trimesh(inflated_wp, faces_wp)
        volumes.append(float(inflated_tm.volume))
        assert inflated_tm.is_watertight
        assert not tw.validation.is_self_intersecting(wp.Mesh(inflated_wp, faces_wp))
        if scale == 0.0:
            assert np.isclose(volumes[-1], mesh_tm.volume, rtol=1e-3)
            continue
        displacement_np = inflated_wp.numpy() - vertices_wp.numpy()
        alignment_np = (displacement_np * normals_np).sum(axis=1) / np.linalg.norm(
            displacement_np, axis=1
        )
        assert alignment_np.mean() > 0.9
    assert volumes[0] < volumes[1] < volumes[2]


def test_inflate_deflates_and_handles_edge_cases(device: str) -> None:
    """
    Not a library comparison: the negative-pressure branch and the degenerate arguments.

    A negative pressure must *shrink* the volume, which is the same code with the sign flipped and
    is worth pinning because a normal-direction bug would show up as growth either way.
    ``pre_smooth`` and ``gradual`` are asserted only to change the answer, not to change it in a
    particular direction -- they are knobs on a flow, and a test claiming more would be inventing a
    specification.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=2)
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces).ravel().astype(np.int32), device
    )
    deflated_wp = tw.smoothing.inflate(vertices_wp, faces_wp, -0.05)
    assert warp_to_trimesh(deflated_wp, faces_wp).volume < mesh_tm.volume

    plain_np = tw.smoothing.inflate(vertices_wp, faces_wp, 0.05, pre_smooth=False).numpy()
    smoothed_np = tw.smoothing.inflate(vertices_wp, faces_wp, 0.05, pre_smooth=True).numpy()
    assert not np.allclose(plain_np, smoothed_np)
    assert not np.allclose(
        plain_np, tw.smoothing.inflate(vertices_wp, faces_wp, 0.05, gradual=False).numpy()
    )

    assert np.array_equal(
        tw.smoothing.inflate(vertices_wp, faces_wp, 0.05, iterations=0).numpy(), vertices_wp.numpy()
    )
    empty_vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    empty_faces_wp = warp_empty(0, wp.int32, device)
    assert tw.smoothing.inflate(empty_vertices_wp, empty_faces_wp, 0.1).shape == (0,)
    with pytest.raises(ValueError, match="iterations must be non-negative"):
        tw.smoothing.inflate(vertices_wp, faces_wp, 0.1, iterations=-1)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("filter_humphrey", "trimesh")
def test_filter_humphrey(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: HC filtering against trimesh, the only oracle this filter has.

    **pymeshlab is not a second oracle here**, recorded so it is not re-tried:
    ``apply_coord_hc_laplacian_smoothing`` implements the same Vollmer et al. paper but exposes no
    parameters at all -- no step count, no ``alpha``, no ``beta`` -- and its single pass matches
    ``filter_humphrey`` at *none* of the 8 x 11 x 11 ``(iterations, alpha, beta)`` combinations
    probed (best max-coordinate deviation 0.019 on a mesh carrying 0.016 of noise). It is a
    different formulation, not this one with other constants, and it is used only as a per-pass
    timing reference in ``benchmarks/test_smoothing.py``.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    smoothed_wp = tw.smoothing.filter_humphrey(
        mesh_wp.points, mesh_wp.indices, alpha=0.1, beta=0.5, iterations=8
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_humphrey(mesh_ref, alpha=0.1, beta=0.5, iterations=8)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("filter_spikes", "meshlib")
@pytest.mark.parametrize("threshold_turns", [0.5, 0.75])
def test_filter_spikes_matches_meshlib(
    torus_spikes: tuple[tm.Trimesh, wp.Mesh], threshold_turns: float
) -> None:
    """
    Class A twice: the same vertices are flagged, and the same number are moved.

    ``findSpikeVertices(topology, points, minSumAngle)`` returns the mask this repair acts on, and
    the two agree **element for element** on the spiky torus -- 5 flagged at ``pi`` and 12 at
    ``1.5 * pi``, with the same vertex indices. Then ``removeSpikes`` moves exactly as many
    vertices as this does, at the same thresholds, so the *extent* of the repair matches even though
    the move itself need not: MeshLib's relaxation and a closed-1-ring average are different
    displacements, and the test does not claim otherwise.

    What it does claim, and this is the part no comparison gives, is that **no spike survives**: the
    angle sums are recomputed afterwards and none is still below the threshold. A repair that moved
    the right vertices to the wrong places would pass a count comparison and fail this.

    Two thresholds because the answer must not be a constant: at ``0.5`` of a turn 5 vertices
    qualify and at ``0.75`` twelve do, so a mask that ignored the parameter would fail one of them.
    """
    mesh_tm, mesh_wp = torus_spikes
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    n_vertices = vertices_wp.size
    min_angle_sum = threshold_turns * 2.0 * math.pi

    defects_np = tw.vertices.vertex_defects(
        n_vertices, faces_wp, tw.triangles.face_angles(vertices_wp, faces_wp)
    ).numpy()
    spikes_np = (2.0 * np.pi - defects_np) < min_angle_sum
    assert 0 < int(spikes_np.sum()) < n_vertices  # non-vacuity: some but not all

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    spikes_ml = meshlib_bitset_to_numpy(
        mm.findSpikeVertices(mesh_ml.topology, mesh_ml.points, min_angle_sum), n_vertices
    )
    assert np.array_equal(spikes_np, spikes_ml)

    repaired_wp, flattened = tw.smoothing.filter_spikes(
        vertices_wp, faces_wp, min_angle_sum, return_count=True
    )
    moved_np = np.abs(repaired_wp.numpy() - vertices_wp.numpy()).max(axis=1) > 1e-7
    assert flattened == int(spikes_np.sum())
    assert int(moved_np.sum()) == int(spikes_np.sum())
    assert not np.any(moved_np & ~spikes_np)  # nothing but the spikes moved

    repaired_ml = trimesh_to_meshlib(mesh_tm)
    mm.removeSpikes(repaired_ml, 10, min_angle_sum)
    moved_ml = (
        np.abs(
            np.asarray(meshlib_to_trimesh(repaired_ml).vertices) - np.asarray(mesh_tm.vertices)
        ).max(axis=1)
        > 1e-7
    )
    assert int(moved_ml.sum()) == int(moved_np.sum())

    # And the repair worked: recomputed on the new positions, nothing is a spike any more.
    after_np = tw.vertices.vertex_defects(
        n_vertices, faces_wp, tw.triangles.face_angles(repaired_wp, faces_wp)
    ).numpy()
    assert not np.any((2.0 * np.pi - after_np) < min_angle_sum)


def test_filter_spikes_return_count_shapes(torus_spikes: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: the two return shapes of the ``return_count`` keyword.

    The default is the bare position buffer, which is what makes this composable with every other
    filter in this module -- it was the one member whose return was not a ``wp.array[wp.vec3]``.
    Both forms must describe the same call, so the positions are compared as well as the shapes --
    at a tolerance rather than exactly, because the 1-ring average is accumulated with atomic adds
    and its float32 summation order varies between launches.
    """
    _mesh_tm, mesh_wp = torus_spikes
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices

    positions_only_wp = tw.smoothing.filter_spikes(vertices_wp, faces_wp, math.pi)
    assert isinstance(positions_only_wp, wp.array)

    positions_wp, flattened = tw.smoothing.filter_spikes(
        vertices_wp, faces_wp, math.pi, return_count=True
    )
    assert flattened > 0  # non-vacuity: the fixture really has spikes
    assert np.allclose(positions_only_wp.numpy(), positions_wp.numpy(), rtol=1e-5, atol=1e-5)


def test_filter_spikes_leaves_a_clean_mesh_alone(
    icosphere: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Not a library comparison: the do-nothing branch, which is the safety claim.

    A sphere has no spike at any sensible threshold, so the buffers must come back **identical** --
    not merely close. That is what rules out a repair that smooths everything a little, which a
    displacement-magnitude assert on a spiky mesh would not catch.
    """
    _, mesh_wp = icosphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    repaired_wp, flattened = tw.smoothing.filter_spikes(
        vertices_wp, faces_wp, math.pi, return_count=True
    )
    assert flattened == 0
    assert np.array_equal(repaired_wp.numpy(), vertices_wp.numpy())

    empty_vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    empty_faces_wp = warp_empty(0, wp.int32, device)
    assert tw.smoothing.filter_spikes(empty_vertices_wp, empty_faces_wp, math.pi).shape == (0,)
    with pytest.raises(ValueError, match="max_iter must be non-negative"):
        tw.smoothing.filter_spikes(vertices_wp, faces_wp, math.pi, max_iter=-1)


@pytest.mark.parity("equalize_triangle_areas", "meshlib")
@pytest.mark.parametrize("no_shrinkage", [False, True])
def test_equalize_triangle_areas_matches_meshlib(device: str, no_shrinkage: bool) -> None:
    """
    Class A against ``equalizeTriAreas``: the same positions, to ``float32``.

    Both minimize the summed squared areas of each vertex's incident triangles by the same 3x3
    solve, so this is an element-wise position comparison and not a statistic. Measured on a noisy
    ``icosphere(3)`` at three passes and ``force=0.5``: **exactly 0.0** maximum difference without
    the shrinkage constraint, and **3.6e-07** with it -- the residual there is the vertex normal,
    which both sides recompute from the current positions every pass.

    ``no_shrinkage`` is parametrized because it is a different linear system (a 2x2 in the tangent
    plane rather than the full 3x3), not a flag on the same one, and the unconstrained branch would
    pass a test that never set it.

    The invariant asserted alongside is that the areas actually became more even: the standard
    deviation of the triangle areas has to fall, which no position comparison implies -- the two
    libraries could agree on a wrong answer.
    """
    mesh_tm = _noisy_icosphere(3, 0.02)
    mesh_wp = trimesh_to_warp(mesh_tm, device)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices

    relaxed_wp = tw.smoothing.equalize_triangle_areas(
        vertices_wp, faces_wp, 3, 0.5, no_shrinkage=no_shrinkage
    )
    mesh_ml = trimesh_to_meshlib(mesh_tm)
    params_ml = mm.MeshEqualizeTriAreasParams()
    params_ml.iterations = 3
    params_ml.force = 0.5
    params_ml.noShrinkage = no_shrinkage
    mm.equalizeTriAreas(mesh_ml, params_ml)
    relaxed_ml = cast("np.ndarray", mn.toNumpyArray(mesh_ml.points))
    # Non-vacuity: the reference moved the mesh, so an identity implementation would fail below.
    assert np.abs(relaxed_ml - np.asarray(mesh_tm.vertices)).max() > 1e-3
    assert np.allclose(relaxed_wp.numpy(), relaxed_ml, rtol=1e-5, atol=1e-5)

    before = tw.triangles.face_normals_and_areas(vertices_wp, faces_wp)[1].numpy()
    after = tw.triangles.face_normals_and_areas(relaxed_wp, faces_wp)[1].numpy()
    assert after.std() < before.std()


def test_equalize_triangle_areas_respects_its_region_and_bound(device: str) -> None:
    """
    Not a library comparison: the two axes meshlib's parameter struct exposes but its port narrows.

    ``region`` and ``max_displacement`` are asserted against the *input* rather than a reference,
    because what they claim is exactly a statement about the input: nothing outside the region may
    move at all, and nothing anywhere may move further than the bound. Both are parametrized over a
    value that binds and one that does not, so neither passes vacuously -- the loose bound has to
    reproduce the unbounded answer bit for bit, and the tight one has to actually clip.
    """
    mesh_wp = trimesh_to_warp(_noisy_icosphere(3, 0.02), device)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    n_vertices = vertices_wp.size
    positions_np = vertices_wp.numpy()

    unbounded_wp = tw.smoothing.equalize_triangle_areas(vertices_wp, faces_wp, 3, 0.5)
    reach = np.linalg.norm(unbounded_wp.numpy() - positions_np, axis=1).max()
    assert reach > 1e-3  # non-vacuity: there is a displacement for the bound to bite into

    loose_wp = tw.smoothing.equalize_triangle_areas(
        vertices_wp, faces_wp, 3, 0.5, max_displacement=10.0 * reach
    )
    assert np.array_equal(loose_wp.numpy(), unbounded_wp.numpy())
    tight_wp = tw.smoothing.equalize_triangle_areas(
        vertices_wp, faces_wp, 3, 0.5, max_displacement=0.2 * reach
    )
    assert np.linalg.norm(tight_wp.numpy() - positions_np, axis=1).max() <= 0.2 * reach + 1e-6

    region_np = positions_np[:, 2] > 0.0
    region_wp = wp.array(region_np, dtype=wp.bool, device=device)
    assert 0 < int(region_np.sum()) < n_vertices  # non-vacuity: a real partition
    partial_wp = tw.smoothing.equalize_triangle_areas(
        vertices_wp, faces_wp, 3, 0.5, region=region_wp
    )
    assert np.array_equal(partial_wp.numpy()[~region_np], positions_np[~region_np])
    assert not np.array_equal(partial_wp.numpy()[region_np], positions_np[region_np])

    all_true_wp = wp.full(n_vertices, True, dtype=wp.bool, device=device)
    assert np.array_equal(
        tw.smoothing.equalize_triangle_areas(
            vertices_wp, faces_wp, 3, 0.5, region=all_true_wp
        ).numpy(),
        unbounded_wp.numpy(),
    )
    assert np.array_equal(
        tw.smoothing.equalize_triangle_areas(vertices_wp, faces_wp, 0).numpy(), positions_np
    )
    with pytest.raises(ValueError, match="iterations must be non-negative"):
        tw.smoothing.equalize_triangle_areas(vertices_wp, faces_wp, -1)
    with pytest.raises(ValueError, match="region must be a length-"):
        tw.smoothing.equalize_triangle_areas(
            vertices_wp, faces_wp, 1, region=wp.zeros(3, dtype=wp.bool, device=device)
        )


@pytest.mark.parity("relax_keep_volume", "meshlib")
def test_relax_keep_volume_matches_meshlib(device: str) -> None:
    """
    Class A against ``relaxKeepVolume``, plus the property the name claims.

    Element-wise positions agree to **1.9e-09** on a noisy ``icosphere(3)`` at three passes -- the
    two-pass formulation (build the displacement field, then subtract its ring average) is the same
    arithmetic on both sides.

    The invariant is the point of the function and no comparison implies it: a plain
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian] over the same mesh must lose
    noticeably more volume than this does. Asserted as a ratio rather than an absolute so it does
    not depend on the noise amplitude.
    """
    mesh_tm = _noisy_icosphere(3, 0.02)
    mesh_wp = trimesh_to_warp(mesh_tm, device)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices

    relaxed_wp = tw.smoothing.relax_keep_volume(vertices_wp, faces_wp, 3, 0.5)
    mesh_ml = trimesh_to_meshlib(mesh_tm)
    params_ml = mm.MeshRelaxParams()
    params_ml.iterations = 3
    params_ml.force = 0.5
    mm.relaxKeepVolume(mesh_ml, params_ml)
    relaxed_ml = cast("np.ndarray", mn.toNumpyArray(mesh_ml.points))
    assert np.abs(relaxed_ml - np.asarray(mesh_tm.vertices)).max() > 1e-3  # non-vacuity
    assert np.allclose(relaxed_wp.numpy(), relaxed_ml, rtol=1e-5, atol=1e-5)

    start = abs(warp_to_trimesh(vertices_wp, faces_wp).volume)
    kept = abs(warp_to_trimesh(relaxed_wp, faces_wp).volume)
    plain_wp = tw.smoothing.filter_laplacian(
        vertices_wp, faces_wp, lamb=0.5, iterations=3, volume_constraint=False
    )
    shrunk = abs(warp_to_trimesh(plain_wp, faces_wp).volume)
    assert abs(kept - start) < abs(shrunk - start)


@pytest.mark.parity("relax_approx", "meshlib")
@pytest.mark.parametrize("fit", ["planar", "quadric"])
def test_relax_approx_matches_meshlib(device: str, fit: Literal["planar", "quadric"]) -> None:
    """
    Class C against ``relaxApprox``: the surfaces agree, the neighbourhoods are not the same set.

    The two differ by construction in *which* vertices each fit sees -- triwarp uses
    [`geodesic_ball`][triwarp.neighbors.geodesic_ball]'s breadth-first ball and meshlib dilates a
    bitset by an edge-length budget, which are the same idea and not the same set -- so an
    element-wise comparison is not available and the displacement fields are compared instead.

    Measured on a noisy ``icosphere(3)`` at ``dilate_radius=0.3``, one pass, ``force=0.5``:
    max position difference **0.0094** planar and **0.0198** quadric, against reference
    displacements of **0.0345** and **0.0194** -- so the two answers are within a quarter of the
    move for planar and within one move for quadric. The bug class this excludes is a fit taken over
    the wrong neighbourhood or in the wrong frame, which would put the vertex somewhere unrelated:
    mutation probe, replacing the fit with the input positions (no move at all) scores **0.0345**
    and **0.0194** against the same reference, so the bound below separates the real answer from
    doing nothing by **3.7x** planar and **1.0x** quadric. Only the planar row therefore carries a
    ratio bound; the quadric row is pinned by the *shared* claim instead, that both flatten the
    noise by a similar amount.

    ``fit`` is parametrized because the two are different fits, not a tuning: the quadric keeps
    curvature the plane removes, and on a sphere that difference is the whole answer.
    """
    mesh_tm = _noisy_icosphere(3, 0.02)
    mesh_wp = trimesh_to_warp(mesh_tm, device)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    positions_np = vertices_wp.numpy()

    relaxed_wp = tw.smoothing.relax_approx(vertices_wp, faces_wp, 0.3, 1, 0.5, fit)
    mesh_ml = trimesh_to_meshlib(mesh_tm)
    params_ml = mm.MeshApproxRelaxParams()
    params_ml.iterations = 1
    params_ml.force = 0.5
    params_ml.surfaceDilateRadius = 0.3
    params_ml.type = mm.RelaxApproxType.Planar if fit == "planar" else mm.RelaxApproxType.Quadric
    mm.relaxApprox(mesh_ml, params_ml)
    relaxed_ml = cast("np.ndarray", mn.toNumpyArray(mesh_ml.points))

    reference_move = np.abs(relaxed_ml - np.asarray(mesh_tm.vertices)).max()
    assert reference_move > 1e-3  # non-vacuity: the default radius is a documented no-op
    disagreement = np.abs(relaxed_wp.numpy() - relaxed_ml).max()
    if fit == "planar":
        assert disagreement < reference_move / 3.0

    # Both take the noise out: the deviation from the underlying unit sphere must fall, by amounts
    # within 25% of each other. This is what the quadric row is pinned by.
    def roughness(points_np: np.ndarray) -> float:
        return float(np.std(np.linalg.norm(points_np, axis=1)))

    before = roughness(positions_np)
    assert roughness(relaxed_wp.numpy()) < before
    assert roughness(relaxed_ml) < before
    smoothed_tw = before - roughness(relaxed_wp.numpy())
    smoothed_ml = before - roughness(relaxed_ml)
    assert abs(smoothed_tw - smoothed_ml) < 0.25 * max(smoothed_tw, smoothed_ml)


@pytest.mark.parametrize("scale", [3e-4, 1e-4])
def test_relax_approx_quadric_is_scale_equivariant(device: str, scale: float) -> None:
    """
    Triwarp against triwarp: the same relaxation, in units of the mesh, at two mesh scales.

    The oracle sits on the unit-scale side, which ``test_relax_approx_matches_meshlib`` pins
    against ``relaxApprox``; this only asks that shrinking the mesh does not change the relative
    move. It did, silently and only for the quadric fit: that fit's design row spans
    ``[u^2, u v, v^2, u, v, 1]``, so at mesh scale ``h`` the normal matrix's diagonal spans ``h^8``
    down to ``1`` and the absolute singularity threshold in
    ``kernels.linalg.solve_normal_equations`` reported well-conditioned neighbourhoods as
    singular -- the vertex then falls back to the *planar* answer, which is a plausible-looking
    position and not an error. Measured before the fix, on this fixture and at this radius: the
    quadric and planar answers were bit-identical at 16 of 642 vertices at ``3e-4`` and at 511 of
    642 at ``1e-4``, against 0 of 642 at unit scale.

    The planar arm is the control: it solves no normal equations, so it was scale-equivariant all
    along and stays so here.
    """
    mesh_tm = _noisy_icosphere(3, 0.02)
    unit_wp = trimesh_to_warp(mesh_tm, device)
    small_tm = mesh_tm.copy()
    small_tm.vertices = small_tm.vertices * scale
    small_wp = trimesh_to_warp(small_tm, device)

    moves = {}
    for fit in ("planar", "quadric"):
        unit_np = tw.smoothing.relax_approx(
            unit_wp.points, unit_wp.indices, 0.3, 1, 0.5, fit
        ).numpy()
        small_np = tw.smoothing.relax_approx(
            small_wp.points, small_wp.indices, 0.3 * scale, 1, 0.5, fit
        ).numpy()
        moves[fit] = unit_np - np.asarray(mesh_tm.vertices)
        assert np.allclose(small_np / scale, unit_np, rtol=1e-3, atol=1e-3)

    # Non-vacuity: the two fits must disagree at unit scale, or "the quadric answer" and "the
    # planar fallback this used to produce" would be the same thing and the assert above could not
    # tell them apart.
    assert np.abs(moves["quadric"] - moves["planar"]).max() > 1e-3


def test_relax_approx_needs_a_radius_that_reaches(device: str) -> None:
    """
    Not a library comparison: the documented silent no-op, and the argument guards.

    A ball smaller than one edge holds the vertex alone, the fit is skipped, and the call returns
    the input **exactly** -- which is worth pinning because it is the failure a caller meets first
    and it raises nothing. meshlib's own ``surfaceDilateRadius`` default of ``0.0`` lands there
    (measured: zero displacement on a noisy ``icosphere(3)``), which is why this port makes the
    radius a required argument with no default rather than copying that one.
    """
    mesh_wp = trimesh_to_warp(_noisy_icosphere(3, 0.02), device)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    tiny_wp = tw.smoothing.relax_approx(vertices_wp, faces_wp, 1e-6, 1, 0.5)
    assert np.array_equal(tiny_wp.numpy(), vertices_wp.numpy())
    assert np.array_equal(
        tw.smoothing.relax_approx(vertices_wp, faces_wp, 0.3, 0).numpy(), vertices_wp.numpy()
    )
    with pytest.raises(ValueError, match="dilate_radius must be positive"):
        tw.smoothing.relax_approx(vertices_wp, faces_wp, 0.0)
    with pytest.raises(ValueError, match="fit must be"):
        tw.smoothing.relax_approx(
            vertices_wp,
            faces_wp,
            0.3,
            fit="cubic",  # pyright: ignore[reportArgumentType]  # the off-menu value under test
        )
    with pytest.raises(ValueError, match="max_displacement must be non-negative"):
        tw.smoothing.relax_approx(vertices_wp, faces_wp, 0.3, max_displacement=-1.0)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("filter_taubin", "trimesh")
def test_filter_taubin(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: lambda-mu filtering against ``trimesh.smoothing``, at matching iteration counts.

    trimesh counts half-steps where MeshLab counts lambda-mu pairs (section 6), so the
    iteration argument is passed in trimesh's convention here and the pymeshlab comparison
    doubles it separately.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    smoothed_wp = tw.smoothing.filter_taubin(
        mesh_wp.points, mesh_wp.indices, lamb=0.5, nu=0.53, iterations=9
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_taubin(mesh_ref, lamb=0.5, nu=0.53, iterations=9)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


# --- MeshLab's two coordinate umbrellas -------------------------------------------------
#
# MeshLab does not use one Laplacian for its coordinate smoothers, it uses two, and which one a
# filter picks was recovered numerically rather than read off any documentation: one pass of each
# filter is a linear map, so running it over 12 random position sets on one connectivity and solving
# least-squares for the per-vertex stencil determines the weights exactly (residual 2e-16).
#
#   * ``apply_coord_laplacian_smoothing`` and ``apply_coord_unsharp_mask`` weight each neighbour by
#     the number of faces the edge to it shares and count the vertex *itself* once, so on a closed
#     mesh a degree-d vertex gets ``1 / (2d + 1)`` on itself and ``2 / (2d + 1)`` on each neighbour.
#     That is `_meshlab_umbrella` below, and it is why these two filters are NOT the plain 1-ring
#     mean -- the difference is 8% of the displacement, far too large to read as a tolerance.
#   * ``apply_coord_taubin_smoothing`` uses the plain 1-ring mean instead, which is triwarp's own
#     default operator.


@pytest.mark.parametrize("steps", [1, 3, 5])
@pytest.mark.parity("filter_taubin", "pymeshlab")
def test_filter_taubin_matches_pymeshlab(device: str, steps: int) -> None:
    """
    Class B: exact, once the *pass count* is mapped -- and the mapping is a factor of two.

    MeshLab's ``stepsmoothnum`` counts lambda-mu **pairs**, where triwarp follows trimesh and does
    one half-step per ``iterations``, alternating the shrinking and inflating passes. So
    ``iterations = 2 * stepsmoothnum`` is the named transform, and ``mu=-0.53`` is triwarp's
    ``nu=0.53`` (the sign is in the convention, not the value). Unlike the two filters above, this
    one uses the plain 1-ring mean, so triwarp's default operator is the right one.

    Measured **5.2e-08 / 4.4e-08 / 4.6e-08** at 1 / 3 / 5 MeshLab steps under the doubled count,
    against **2.7e-02 / 2.6e-02 / 2.6e-02** at the naive equal count -- a 5-orders-of-magnitude
    separation, so this assert is what holds the pass mapping in place.
    ``benchmarks/test_smoothing`` originally gave MeshLab ``stepsmoothnum=_ITERATIONS`` against
    triwarp's ``iterations=_ITERATIONS`` and so timed it doing twice the passes; that row now halves
    it.
    """
    mesh_tm = _noisy_icosphere()
    mesh_wp = trimesh_to_warp(mesh_tm, device)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=steps)

    smoothed_wp = tw.smoothing.filter_taubin(
        mesh_wp.points, mesh_wp.indices, lamb=0.5, nu=0.53, iterations=2 * steps
    )
    assert np.allclose(
        smoothed_wp.numpy(), meshset_pml.current_mesh().vertex_matrix(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("iterations", [1, 3, 10])
@pytest.mark.parity(
    "filter_taubin",
    "pytorch3d",
    benchmarked=False,
    reason="the filter_taubin group times the default fixed-operator filter, which is trimesh's "
    "and disagrees with pytorch3d by the size of the displacement -- that pair is the noparity "
    "entry in benchmarks/test_smoothing.py and stays one. recompute=True is what agrees, and it "
    "is a separate filter at ~23x the cost (0.49 -> 11.06 ms at 10 passes on CUDA), so timing it "
    "in that group would race two different amounts of work under one name.",
)
def test_filter_taubin_recompute_matches_pytorch3d(device: str, iterations: int) -> None:
    """
    Class B: ``ops.taubin_smoothing`` under the pass-count doubling, with ``recompute=True``.

    This is the pair section 4's ``recompute`` keyword exists for, and the measurement is what
    justified building it. pytorch3d rebuilds its inverse-distance operator from the *current*
    positions before every half-pass; against one fixed operator the two sit **4.1e-03 / 6.5e-03 /
    9.9e-03** apart at 1 / 3 / 10 of its iterations -- the size of the displacement itself, which
    is why the default is a ``noparity`` entry rather than a loose tolerance -- and recomputing
    closes it to **2.4e-07 / 4.2e-07 / 7.2e-07**.

    Two named transforms, both conventions rather than arithmetic: ``num_iter`` counts lambda-mu
    **pairs** where triwarp does one half-step per ``iterations``, so the count doubles (the same
    factor the pymeshlab pair above needs); and ``mu=-0.53`` is triwarp's ``nu=0.53``, the sign
    living in the convention.

    The fixed-operator gap is asserted here too, on the same input -- without it this test would
    read as a claim that the two libraries agree, when what it shows is that they agree *only*
    under this keyword.
    """
    mesh_tm = _noisy_icosphere()
    mesh_wp = trimesh_to_warp(mesh_tm, device)
    smoothed_p3d = p3d_ops.taubin_smoothing(
        trimesh_to_pytorch3d(mesh_tm), lambd=0.53, mu=-0.53, num_iter=iterations
    )
    verts_t = smoothed_p3d.verts_packed()
    assert verts_t is not None
    vertices_p3d = verts_t.cpu().numpy()

    recomputed_wp = tw.smoothing.filter_taubin(
        mesh_wp.points,
        mesh_wp.indices,
        lamb=0.53,
        nu=0.53,
        iterations=2 * iterations,
        recompute=True,
    )
    assert np.allclose(recomputed_wp.numpy(), vertices_p3d, rtol=1e-5, atol=1e-5)

    # The default is a *different* filter, and the whole reason for the keyword.
    fixed_wp = tw.smoothing.filter_taubin(
        mesh_wp.points,
        mesh_wp.indices,
        lamb=0.53,
        nu=0.53,
        iterations=2 * iterations,
        laplacian_operator=tw.laplacian.laplacian(
            mesh_wp.points, mesh_wp.indices, equal_weight=False
        ),
    )
    assert float(np.abs(fixed_wp.numpy() - vertices_p3d).max()) > 1e-3

    with pytest.raises(ValueError, match="do not also pass one"):
        tw.smoothing.filter_taubin(
            mesh_wp.points,
            mesh_wp.indices,
            recompute=True,
            laplacian_operator=tw.laplacian.laplacian(mesh_wp.points, mesh_wp.indices),
        )


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_filter_neighborhood_average(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: the plain 1-ring mean against a numpy reference over trimesh's adjacency.

    The reference is written here rather than taken from a filter, because every library's
    "Laplacian smoothing" weights its neighbours differently -- section 6 records MeshLab's two
    undocumented umbrellas and Open3D's inverse-distance one.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    iterations = 5

    smoothed_wp = tw.smoothing.filter_neighborhood_average(
        mesh_wp.points, mesh_wp.indices, iterations=iterations
    )

    mesh_o3d = trimesh_to_open3d(mesh_tm).filter_smooth_simple(number_of_iterations=iterations)

    assert np.allclose(smoothed_wp.numpy(), np.asarray(mesh_o3d.vertices), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
@pytest.mark.parity("filter_mut_dif_laplacian", "trimesh")
def test_filter_mut_dif_laplacian_volume_constraint(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A against trimesh, on watertight fixtures only -- the constraint needs a volume.

    The volume-preserving variant inflates along vertex normals, so an open mesh has nothing to
    preserve; restricting the fixtures is what makes the comparison meaningful rather than a
    looser tolerance.
    """
    # Watertight meshes only: the volume constraint inflates along vertex normals, so mesh.volume
    # must be meaningful.
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    smoothed_wp = tw.smoothing.filter_mut_dif_laplacian(
        mesh_wp.points, mesh_wp.indices, lamb=0.5, iterations=8, volume_constraint=True
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_mut_dif_laplacian(mesh_ref, lamb=0.5, iterations=8, volume_constraint=True)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


def test_filter_mut_dif_laplacian_no_volume_constraint(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    # Open mesh, unconstrained path. Note: the per-vertex adil = 1/|N.(V - L.V)| reciprocal is
    # coupled globally through its mean, so on strongly-saddled meshes (e.g. half_torus) the filter
    # is chaotically sensitive to input precision (the float64 trimesh reference itself diverges by
    # ~1e-2 under a float32 input round-trip). The hemisphere has no such near-zero normal residual,
    # so float32 warp matches the float64 reference tightly.
    mesh_tm, mesh_wp = hemisphere

    smoothed_wp = tw.smoothing.filter_mut_dif_laplacian(
        mesh_wp.points, mesh_wp.indices, lamb=0.5, iterations=8, volume_constraint=False
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_mut_dif_laplacian(mesh_ref, lamb=0.5, iterations=8, volume_constraint=False)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


@pytest.mark.parity(
    "filter_implicit_fairing",
    "igl",
    benchmarked=False,
    reason="libigl binds no implicit-fairing driver, so the reference is one igl.cotmatrix plus "
    "one igl.massmatrix plus a scipy spsolve per pass, assembled here. A row would time a "
    "direct sparse factorization against Warp's conjugate gradient over a composition neither "
    "library exposes as a function, which is a solver comparison rather than this group's.",
)
def test_filter_implicit_fairing(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A against libigl on a closed mesh, which is where the flow is defined.

    The reference is the flow itself, assembled per pass out of ``igl.cotmatrix`` and
    ``igl.massmatrix`` and stepped with a direct ``scipy`` solve, because libigl binds no
    implicit-fairing driver -- which is also why the pair is claimed untimed above.

    On an open boundary the unconstrained flow degrades boundary triangles and the CG solve
    diverges -- igl's direct solver tolerates that and Warp offers only CG, so the fixture
    choice is a real limitation rather than a convenience.
    """
    # Implicit curvature flow is defined for closed meshes; on open boundaries the unconstrained
    # flow degrades boundary triangles and the conjugate-gradient solve diverges (igl's direct
    # solver tolerates it, Warp only offers CG), so the regression uses the watertight icosahedron.
    mesh_tm, mesh_wp = icosahedron

    lamb = 0.1
    iterations = 6
    vertices_igl = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_igl = np.array(mesh_tm.faces, dtype=np.int64)
    for _ in range(iterations):
        stiffness_igl = cast("sp.csc_matrix[np.float64]", igl.cotmatrix(vertices_igl, faces_igl))
        mass_igl = igl.massmatrix(vertices_igl, faces_igl, igl.MASSMATRIX_TYPE_BARYCENTRIC)
        system_igl = (mass_igl - lamb * stiffness_igl).tocsc()
        vertices_igl = cast("np.ndarray", spla.spsolve(system_igl, mass_igl.dot(vertices_igl)))

    smoothed_wp = tw.smoothing.filter_implicit_fairing(
        mesh_wp.points, mesh_wp.indices, lamb=lamb, iterations=iterations
    )

    assert np.allclose(smoothed_wp.numpy(), vertices_igl, rtol=1e-4, atol=1e-4)


def test_filter_implicit_fairing_pins_the_boundary(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Boundary vertices are held exactly; the interior is the part that moves."""
    _, mesh_wp = hemisphere
    original_np = mesh_wp.points.numpy()
    boundary_np = tw.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices).numpy()
    interior_np = np.setdiff1d(np.arange(len(original_np)), boundary_np)
    assert len(boundary_np) > 0  # the fixture must actually be open for this to mean anything

    smoothed_np = tw.smoothing.filter_implicit_fairing(
        mesh_wp.points, mesh_wp.indices, iterations=5
    ).numpy()

    assert np.array_equal(smoothed_np[boundary_np], original_np[boundary_np])
    assert np.abs(smoothed_np[interior_np] - original_np[interior_np]).max() > 1e-6


def test_filter_implicit_fairing_pinned_stays_stable_on_an_open_mesh(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Many passes on an open mesh converge instead of diverging.

    The unconstrained flow pulls the free boundary inward until the triangles there collapse, after
    which the system is effectively singular and the conjugate gradient runs to its iteration cap.
    Pinning makes each pass a Dirichlet problem over the interior, which is well posed however many
    times it is applied — so this asserts *no* non-convergence warning, not merely finiteness.
    """
    _, mesh_wp = hemisphere
    extent = float(np.abs(mesh_wp.points.numpy()).max())

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        smoothed_np = tw.smoothing.filter_implicit_fairing(
            mesh_wp.points, mesh_wp.indices, iterations=25
        ).numpy()

    assert np.isfinite(smoothed_np).all()
    # Smoothing cannot inflate the patch beyond its pinned rim.
    assert np.abs(smoothed_np).max() <= extent * 1.01


def test_filter_implicit_fairing_pins_a_mesh_with_no_interior_vertex(device: str) -> None:
    """
    Not a library comparison: no reference exposes a boundary-pinned implicit fairing.

    There is nothing to compare a *no-op* against, and the claim is exactly that nothing moves.

    A mesh whose every vertex is on the boundary -- a single triangle, a fan, a strip, a small hole
    patch -- leaves the pinned flow with no unknown, so the pass is the identity. ``None`` from the
    partition builder means "no boundary, run unconstrained", and folding this case into it ran the
    *unconstrained* flow and moved every vertex the caller asked to pin. What the invariant excludes
    is only that substitution; the pinned-versus-free contrast is what makes it non-vacuous, since
    an unconditional early return would pass the first assert alone.
    """
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.4]])
    faces_np = np.array([0, 1, 2, 1, 3, 2], dtype=np.int32)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    # Non-vacuity: every vertex really is on the boundary, so there is no interior to solve over.
    boundary_np = tw.boundary.boundary_vertex_indices(vertices_wp, faces_wp).numpy()
    assert len(boundary_np) == len(vertices_np)

    pinned_np = tw.smoothing.filter_implicit_fairing(
        vertices_wp, faces_wp, iterations=4, pin_boundary=True
    ).numpy()
    assert np.array_equal(pinned_np, vertices_wp.numpy())
    # The unconstrained flow on the same input *does* move, which is what the pinned path was
    # silently doing before -- so this is the arm that makes the equality above mean something.
    free_np = tw.smoothing.filter_implicit_fairing(
        vertices_wp, faces_wp, iterations=4, pin_boundary=False
    ).numpy()
    assert np.abs(free_np - vertices_wp.numpy()).max() > 1e-6


def test_filter_implicit_fairing_pin_boundary_is_a_no_op_on_a_closed_mesh(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    A watertight mesh has no boundary, so the flag routes down the same unreduced solve either way.

    Compared at ``float64`` round-off rather than bitwise: the sparse mat-vec accumulates with
    atomics, so *any* two runs of this function differ in the last bits, flag or no flag.
    """
    _, mesh_wp = icosahedron
    assert tw.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices).size == 0

    pinned_np = tw.smoothing.filter_implicit_fairing(
        mesh_wp.points, mesh_wp.indices, iterations=6, pin_boundary=True
    ).numpy()
    unpinned_np = tw.smoothing.filter_implicit_fairing(
        mesh_wp.points, mesh_wp.indices, iterations=6, pin_boundary=False
    ).numpy()

    assert np.allclose(pinned_np, unpinned_np, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# refine_and_smooth_region (shared finisher of fill_smooth and stitch_smooth)
# ---------------------------------------------------------------------------


def test_implicit_filters_cpu_match_cuda(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: both implicit filters give the CUDA answer on CPU.

    Pins the removal of the CUDA-only guard: ``warp.optim.linear.cg`` returned NaN on the Warp CPU
    device through 1.15, so these two refused to run there at all. Fixed in 1.16.
    """
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to compare them")
    mesh_tm, _ = icosahedron

    for name, call in (
        (
            "filter_laplacian",
            lambda v, f: tw.smoothing.filter_laplacian(
                v, f, iterations=2, implicit_time_integration=True
            ),
        ),
        (
            "filter_implicit_fairing",
            lambda v, f: tw.smoothing.filter_implicit_fairing(v, f, iterations=2),
        ),
    ):
        positions = {}
        for device in ("cpu", "cuda:0"):
            mesh_wp = trimesh_to_warp(mesh_tm, device)
            positions[device] = call(mesh_wp.points, mesh_wp.indices).numpy()
        assert np.isfinite(positions["cpu"]).all(), name
        assert np.allclose(positions["cpu"], positions["cuda:0"], rtol=1e-5, atol=1e-5), name


def _sphere_region(subdivisions: int = 2, z_cut: float = 0.5):
    sph = tm.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    vertices = sph.vertices.astype(np.float64)
    faces = sph.faces.astype(np.int32)
    free = vertices[:, 2] > z_cut
    return vertices, faces, free


@pytest.mark.parity("smooth_region_fixed_rim", "meshlib")
def test_smooth_region_fixed_rim_matches_meshlib(device: str):
    """
    Class A on the positions: ``positionVertsSmoothlySharpBd`` solves the same system, directly.

    The rim is pinned on both sides and the free set is identical, so the two solve the same
    Dirichlet problem and agree to ``1e-5`` element-wise even though meshlib factorizes where this
    iterates -- a solve to a fixpoint has one answer, which is what makes this Class A rather than a
    displacement bound.

    The fixed half is asserted exactly (``array_equal``, not ``allclose``): a solver that moved a
    pinned vertex by a hair would still pass the tolerance on the free set.
    """
    vertices_np, faces_np, free_np = _sphere_region()
    v_wp = points_to_warp(vertices_np, device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    free_wp = wp.array(free_np, dtype=wp.bool, device=device)

    result_wp = tw.smoothing.smooth_region_fixed_rim(v_wp, f_wp, free_wp)

    mesh_ml = numpy_to_meshlib(vertices_np, faces_np)
    params_ml = mm.PositionVertsSmoothlyParams()
    params_ml.region = mn.vertBitSetFromBools(cast("Buffer", free_np))
    mm.positionVertsSmoothlySharpBd(mesh_ml, params_ml)
    verts_ml = mn.getNumpyVerts(mesh_ml)

    assert np.allclose(result_wp.numpy(), verts_ml, rtol=1e-5, atol=1e-5)
    assert np.array_equal(result_wp.numpy()[~free_np], v_wp.numpy()[~free_np])


def test_smooth_region_fixed_rim_dirichlet_residual(device: str):
    vertices_np, faces_np, free_np = _sphere_region()
    v_wp = points_to_warp(vertices_np, device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    free_wp = wp.array(free_np, dtype=wp.bool, device=device)

    result = tw.smoothing.smooth_region_fixed_rim(v_wp, f_wp, free_wp).numpy()

    # Umbrella (unit-weight) residual: deg(v) * p_v - sum_neighbors p_d == 0 for every free vertex.
    adjacency: dict[int, set[int]] = {}
    for tri in faces_np:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            adjacency.setdefault(int(a), set()).add(int(b))
            adjacency.setdefault(int(b), set()).add(int(a))
    max_residual = 0.0
    for v in np.flatnonzero(free_np):
        neighbors = list(adjacency[int(v)])
        residual = len(neighbors) * result[v] - result[neighbors].sum(axis=0)
        max_residual = max(max_residual, float(np.linalg.norm(residual)))
    assert max_residual < 1e-4


@pytest.mark.parametrize("edge_weights", ["cotan", "unit"])
@pytest.mark.parity("smooth_region", "meshlib")
def test_smooth_region_matches_meshlib(device: str, edge_weights: str):
    """
    Class A on the positions, over both edge weightings: ``positionVertsSmoothly``'s two modes.

    Same reasoning as the fixed-rim test above, at ``1e-4``. Both weightings are run because each
    builds a different system rather than a differently-scaled one, and because the pairing of
    ``EdgeWeights.Cotan`` / ``EdgeWeights.Unit`` onto triwarp's ``edge_weights`` is a claim about
    which discretization each name means -- the same kind of claim MeshLib settles for the vertex
    normals. ``VertexMass.Unit`` is passed explicitly for that reason: the mass matrix is the third
    axis, and triwarp has no counterpart to its area-weighted setting.
    """
    vertices_np, faces_np, free_np = _sphere_region()
    v_wp = points_to_warp(vertices_np, device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    free_wp = wp.array(free_np, dtype=wp.bool, device=device)

    result_wp = tw.smoothing.smooth_region(v_wp, f_wp, free_wp, edge_weights=edge_weights)

    mesh_ml = numpy_to_meshlib(vertices_np, faces_np)
    ew_ml = mm.EdgeWeights.Cotan if edge_weights == "cotan" else mm.EdgeWeights.Unit
    mm.positionVertsSmoothly(
        mesh_ml, mn.vertBitSetFromBools(cast("Buffer", free_np)), ew_ml, mm.VertexMass.Unit
    )
    verts_ml = mn.getNumpyVerts(mesh_ml)

    assert np.allclose(result_wp.numpy(), verts_ml, rtol=1e-4, atol=1e-4)
    assert np.array_equal(result_wp.numpy()[~free_np], v_wp.numpy()[~free_np])


def test_smooth_region_empty_region(device: str):
    vertices_np, faces_np, _ = _sphere_region()
    v_wp = points_to_warp(vertices_np, device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    empty = wp.zeros(len(vertices_np), dtype=wp.bool, device=device)
    result = tw.smoothing.smooth_region_fixed_rim(v_wp, f_wp, empty)
    assert np.array_equal(result.numpy(), v_wp.numpy())


@pytest.mark.parametrize(
    "smoother", [tw.smoothing.smooth_region, tw.smoothing.smooth_region_fixed_rim]
)
def test_region_smoothers_leave_an_unreferenced_free_vertex_where_it_is(
    device: str, smoother: Callable[..., wp.array[wp.vec3]]
) -> None:
    """
    An unreferenced vertex has no smoothing equation, so it must not move.

    Not a library comparison: every reference here drops unreferenced vertices on import
    (``meshFromFacesVerts`` sizes by ``F.max() + 1``, ``load_array`` deletes them), so none can be
    asked what happens to one -- which is exactly why this went unnoticed. The invariant is the
    contract instead, and it excludes the failure mode it was written for: both solvers take the
    free vertices' *new* positions as unknowns, and a vertex touched by no face contributes no row
    to the system, so conjugate gradient never writes its entry. Seeded from ``wp.zeros`` that
    entry stays zero and the vertex is silently teleported to the origin -- measured at 7 of
    ``bunny_decimated``'s free vertices and **297 of ``bunny``'s 1 113**, up to 60 % of the
    bounding-box diagonal away. Seeding from the current positions leaves it alone.

    The referenced vertices are asserted to have *moved*, so a smoother that returned its input
    unchanged could not pass this.
    """
    vertices_np, faces_np, free_np = _sphere_region()
    # One vertex no face refers to, placed inside the free region and well away from the origin.
    stray_np = np.array([[3.0, 4.0, 5.0]])
    vertices_np = np.vstack([vertices_np, stray_np])
    free_np = np.append(free_np, True)
    stray = len(vertices_np) - 1

    v_wp = points_to_warp(vertices_np, device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    free_wp = wp.array(free_np, dtype=wp.bool, device=device)
    smoothed_np = smoother(v_wp, f_wp, free_wp).numpy()

    assert np.allclose(smoothed_np[stray], stray_np[0], rtol=1e-5, atol=1e-5)
    moved_np = np.linalg.norm(smoothed_np[:stray] - vertices_np[:stray], axis=1)
    assert moved_np[free_np[:stray]].max() > 1e-4


@pytest.mark.parametrize(
    ("smoother", "edge_weights"),
    [
        (tw.smoothing.smooth_region_fixed_rim, None),
        (tw.smoothing.smooth_region, "unit"),
        (tw.smoothing.smooth_region, "cotan"),
    ],
)
def test_region_smoothers_ignore_a_face_repeating_a_vertex(
    device: str, smoother: Callable[..., wp.array[wp.vec3]], edge_weights: str | None
) -> None:
    """
    A face ``(a, a, b)`` adds no neighbour: the region solves answer as if it were absent.

    Not a library comparison: every reference here either rejects a repeated-index face or cleans
    it away on import, so none can be asked. The invariant is the contract the region solves' row
    walks are built on -- a vertex's neighbours are the other endpoints of its unique edges, and the
    self-loop ``(a, a)`` such a face contributes is no neighbour (it would put a second entry in the
    diagonal's column). ``(a, b)`` is an edge of the mesh already, so the answer must not move.
    """
    vertices_np, faces_np, free_np = _sphere_region()
    v_wp = points_to_warp(vertices_np, device)
    free_wp = wp.array(free_np, dtype=wp.bool, device=device)
    a, b = (int(x) for x in faces_np[np.flatnonzero(free_np[faces_np].all(axis=1))[0], :2])
    degenerate_np = np.vstack([faces_np, [a, a, b]])
    kwargs = {} if edge_weights is None else {"edge_weights": edge_weights}

    def smooth(faces: np.ndarray) -> np.ndarray:
        f_wp = wp.array(faces.reshape(-1).astype(np.int32), dtype=wp.int32, device=device)
        return smoother(v_wp, f_wp, free_wp, **kwargs).numpy()

    clean_np = smooth(faces_np)
    assert np.linalg.norm(clean_np - vertices_np, axis=1).max() > 1e-4
    assert np.allclose(smooth(degenerate_np), clean_np, rtol=1e-5, atol=1e-5)


def _patch_to_refine(device: str):
    """Hole an icosphere, fill it, and mark the fill as the patch, with the pre-fill counts."""
    sphere_tm = tm.creation.icosphere(subdivisions=2, radius=1.0)
    centers_np = sphere_tm.triangles_center
    keep_np = np.ones(sphere_tm.faces.shape[0], dtype=bool)
    keep_np[np.argsort(-centers_np[:, 2])[:6]] = False
    holed_tm = tm.Trimesh(sphere_tm.vertices, sphere_tm.faces[keep_np], process=False)
    holed_tm.remove_unreferenced_vertices()
    vertices_wp, faces_wp = numpy_to_warp(holed_tm.vertices, holed_tm.faces, device)

    n_vertices_before = vertices_wp.size
    n_faces_before = faces_wp.size // 3
    filled_wp = tw.holes.fill_min_weight(vertices_wp, faces_wp)
    n_faces_after = filled_wp.size // 3
    patch_wp = tw.array.indices_to_mask(
        wp.array(
            np.arange(n_faces_before, n_faces_after, dtype=np.int32), dtype=wp.int32, device=device
        ),
        n_faces_after,
        device=device,
    )
    return vertices_wp, filled_wp, patch_wp, n_vertices_before


def test_refine_and_smooth_region_without_curvature_is_just_the_subdivision(device: str) -> None:
    """
    Class A: ``smooth_curvature=False`` returns ``subdivide_region_to_size``'s output bit for bit.

    That early return is the branch its two callers take when smoothing is off, so the claim worth
    pinning is that nothing else happens on the way -- all three returns compare equal with
    ``np.array_equal``, not a tolerance.
    """
    vertices_wp, faces_wp, patch_wp, n_vertices_before = _patch_to_refine(device)
    max_edge = float(tw.edges.mean_edge_length(vertices_wp, faces_wp)) * 0.6

    refined_wp, refined_faces_wp, refined_patch_wp = tw.smoothing.refine_and_smooth_region(
        vertices_wp,
        faces_wp,
        n_vertices_before,
        patch_wp,
        max_edge=max_edge,
        smooth_curvature=False,
        **_REFINE_KWARGS,
    )
    subdivided_wp, subdivided_faces_wp, subdivided_patch_wp = tw.remesh.subdivide_region_to_size(
        vertices_wp, faces_wp, patch_wp, max_edge=max_edge, max_splits=3, max_angle_change=0.5
    )

    assert refined_wp.size > n_vertices_before
    assert np.array_equal(refined_wp.numpy(), subdivided_wp.numpy())
    assert np.array_equal(refined_faces_wp.numpy(), subdivided_faces_wp.numpy())
    assert np.array_equal(refined_patch_wp.numpy(), subdivided_patch_wp.numpy())


def test_refine_and_smooth_region_moves_only_the_vertices_subdivision_added(device: str) -> None:
    """
    With curvature smoothing on, every new interior vertex moves and no pre-existing one does.

    That is the whole contract of the ``n_vertices_before`` argument: the free set is the tail
    subdivision appended, minus the mesh boundary. Measured on this patch -- 3 new vertices, all
    three displaced (max 0.034), and the 161 pre-existing ones displaced by **exactly 0.0**.
    """
    vertices_wp, faces_wp, patch_wp, n_vertices_before = _patch_to_refine(device)
    max_edge = float(tw.edges.mean_edge_length(vertices_wp, faces_wp)) * 0.6

    smoothed_wp, smoothed_faces_wp, _patch_wp = tw.smoothing.refine_and_smooth_region(
        vertices_wp,
        faces_wp,
        n_vertices_before,
        patch_wp,
        max_edge=max_edge,
        smooth_curvature=True,
        **_REFINE_KWARGS,
    )
    subdivided_wp, subdivided_faces_wp, _subdivided_patch_wp = tw.remesh.subdivide_region_to_size(
        vertices_wp, faces_wp, patch_wp, max_edge=max_edge, max_splits=3, max_angle_change=0.5
    )

    # Smoothing moves positions only: the connectivity is the subdivision's.
    assert np.array_equal(smoothed_faces_wp.numpy(), subdivided_faces_wp.numpy())
    displacement_np = np.abs(smoothed_wp.numpy() - subdivided_wp.numpy()).max(axis=1)
    assert displacement_np.size > n_vertices_before
    assert np.array_equal(
        displacement_np[:n_vertices_before],
        np.zeros(n_vertices_before, dtype=displacement_np.dtype),
    )
    assert np.all(displacement_np[n_vertices_before:] > 1e-9)


# ---------------------------------------------------------------------------
# Scalar-field filters vs pymeshlab (apply_scalar_smoothing / _saturation_per_vertex)
# ---------------------------------------------------------------------------


@pytest.mark.parity("smooth_region_boundary", "meshlib")
@pytest.mark.parametrize("subdivisions", [3, 4])
def test_smooth_region_boundary_matches_meshlib(device: str, subdivisions: int) -> None:
    """
    Class C against ``smoothRegionBoundary``: identical moved sets, agreeing rim curves.

    Class C rather than A because the two are not the same algorithm end to end. Both pin a field
    to -1 on the region and +1 outside, solve its harmonic interpolation over the band that touches
    both, and slide those vertices onto the zero level set -- but meshlib first **flips** the band's
    interior edges to improve the configuration, which changes the connectivity and therefore the
    level set. This port deliberately does not (it promises the face buffer back untouched), so the
    positions cannot match element-wise.

    What does match, and is asserted: the set of vertices that move is **exactly equal** (44 of 642
    at ``subdivisions=3``, 112 of 2 562 at 4), the rim's total length agrees within 1 % (6.073
    against 6.107, and 6.024 against 6.049), the per-vertex displacement magnitudes correlate at
    **0.984-0.987**, and the largest position disagreement is **0.0052** against displacements
    reaching 0.15 -- a 29x margin. The bug class excluded is a band that slides the wrong way or
    off the surface: both sides keep every moved vertex within **0.0021** of the unit sphere, which
    a projection onto the wrong level set would not.

    The invariant asserted alongside is the function's actual purpose and no comparison implies it:
    the rim gets **shorter**. Two mesh resolutions because the level set is a property of the
    surface, not of the triangulation, so the claim has to survive refining it.

    !!! note "The divergence is real on a ragged selection"
        Where the region is speckled with isolated single-face islands on a *coarse* mesh, meshlib's
        edge flips are doing most of the work and this port's rim can end up slightly longer than
        the input (measured 40.32 against meshlib's 38.86 on a 15 %-speckled ``icosphere(3)``,
        against an input of 39.46). Refining the mesh removes the gap. The fixtures here are a
        clean threshold selection, which is the case the function is for.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    vertices_np = np.asarray(mesh_tm.vertices)
    faces_np = np.asarray(mesh_tm.faces)
    region_np = vertices_np[faces_np].mean(axis=1)[:, 2] > 0.3
    assert 0 < int(region_np.sum()) < len(faces_np)  # non-vacuity: a real region with a rim
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np.ravel(), device)
    region_wp = wp.array(region_np, dtype=wp.bool, device=device)

    smoothed_wp = tw.smoothing.smooth_region_boundary(vertices_wp, faces_wp, region_wp, 4)
    smoothed_np = smoothed_wp.numpy()

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    region_ml = mm.FaceBitSet(numpy_to_meshlib_bitset(region_np))
    region_ml.resize(len(faces_np))
    before_ml = cast(npt.NDArray[np.float64], mn.toNumpyArray(mesh_ml.points)).copy()
    mm.smoothRegionBoundary(mesh_ml, region_ml, 4)
    smoothed_ml = cast(npt.NDArray[np.float64], mn.toNumpyArray(mesh_ml.points))

    moved_wp = np.linalg.norm(smoothed_np - vertices_wp.numpy(), axis=1) > 1e-6
    moved_ml = np.linalg.norm(smoothed_ml - before_ml, axis=1) > 1e-6
    assert int(moved_ml.sum()) > 0  # non-vacuity: the reference did something
    assert np.array_equal(moved_wp, moved_ml)
    assert np.abs(smoothed_np - smoothed_ml).max() < 0.05

    rim_np = tw.selection.region_boundary_edges(faces_wp, region_wp).numpy()

    def rim_length(points_np: np.ndarray) -> float:
        return float(
            np.linalg.norm(points_np[rim_np[:, 0]] - points_np[rim_np[:, 1]], axis=1).sum()
        )

    start = rim_length(vertices_np)
    assert rim_length(smoothed_np) < start
    assert abs(rim_length(smoothed_np) - rim_length(smoothed_ml)) < 0.01 * start
    # Neither side lifted the band off the sphere it slid along.
    assert np.abs(np.linalg.norm(smoothed_np[moved_wp], axis=1) - 1.0).max() < 0.01


def test_smooth_region_boundary_leaves_connectivity_and_the_rest_alone(device: str) -> None:
    """
    Not a library comparison: the promises the signature makes, and the degenerate regions.

    Only the band may move -- neither the region's interior nor anything outside it -- and the face
    buffer the caller passed in is still the one its ``region`` mask indexes. An empty region and a
    full one both have no band at all, so both are the identity; that is the branch a caller hits
    when a selection threshold misses.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    vertices_np = np.asarray(mesh_tm.vertices)
    faces_np = np.asarray(mesh_tm.faces)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np.ravel(), device)
    n_faces = len(faces_np)

    region_np = vertices_np[faces_np].mean(axis=1)[:, 2] > 0.3
    region_wp = wp.array(region_np, dtype=wp.bool, device=device)
    smoothed_np = tw.smoothing.smooth_region_boundary(vertices_wp, faces_wp, region_wp, 4).numpy()
    moved = np.linalg.norm(smoothed_np - vertices_wp.numpy(), axis=1) > 1e-6
    # Every moved vertex belongs to both a selected and an unselected face: that is the band.
    inside = np.zeros(len(vertices_np), dtype=bool)
    inside[faces_np[region_np].ravel()] = True
    outside = np.zeros(len(vertices_np), dtype=bool)
    outside[faces_np[~region_np].ravel()] = True
    assert not np.any(moved & ~(inside & outside))

    for degenerate in (np.zeros(n_faces, dtype=bool), np.ones(n_faces, dtype=bool)):
        identity_np = tw.smoothing.smooth_region_boundary(
            vertices_wp, faces_wp, wp.array(degenerate, dtype=wp.bool, device=device), 4
        ).numpy()
        assert np.array_equal(identity_np, vertices_wp.numpy())
    assert np.array_equal(
        tw.smoothing.smooth_region_boundary(vertices_wp, faces_wp, region_wp, 0).numpy(),
        vertices_wp.numpy(),
    )
    with pytest.raises(ValueError, match="iterations must be non-negative"):
        tw.smoothing.smooth_region_boundary(vertices_wp, faces_wp, region_wp, -1)
    with pytest.raises(ValueError, match="region must be a length-"):
        tw.smoothing.smooth_region_boundary(
            vertices_wp, faces_wp, wp.zeros(3, dtype=wp.bool, device=device), 4
        )


def test_smooth_region_boundary_later_passes_match_a_rebuild(device: str) -> None:
    """
    Triwarp against triwarp: the in-place rewrite of the band's system against a fresh one.

    Every pass after the first rewrites the first pass's Dirichlet system from the new cotangents
    (``kernels/smoothing.band_dirichlet_values``) into the same operator, whose solver state -- and
    its first pass's Jacobi diagonal -- carries over. A call of one pass builds its operator fresh,
    so chaining single-pass calls is the rebuild, and the meshlib comparison above carries the
    oracle for it; ``test_smooth_region_boundary_system_matches_the_cotmatrix_extraction`` pins
    the system itself. The two differ by the warm start and the solver tolerance, far under float32
    rounding of the positions -- a stale value or right-hand side in the rewrite would not.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=4, radius=1.0)
    vertices_np = np.asarray(mesh_tm.vertices)
    faces_np = np.asarray(mesh_tm.faces)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np.ravel(), device)
    centers_np = vertices_np[faces_np].mean(axis=1)
    region_np = centers_np[:, 0] + 0.3 * np.sin(5.0 * centers_np[:, 1]) > 0.1
    region_wp = wp.array(region_np, dtype=wp.bool, device=device)

    rewritten_np = tw.smoothing.smooth_region_boundary(vertices_wp, faces_wp, region_wp, 3).numpy()
    rebuilt_wp = vertices_wp
    for _ in range(3):
        rebuilt_wp = tw.smoothing.smooth_region_boundary(rebuilt_wp, faces_wp, region_wp, 1)
    displacement_np = np.linalg.norm(rewritten_np - vertices_np, axis=1)
    assert displacement_np.max() > 1e-2
    assert np.allclose(rewritten_np, rebuilt_wp.numpy(), rtol=0.0, atol=1e-5)


def test_smooth_region_boundary_system_matches_the_cotmatrix_extraction(device: str) -> None:
    """
    Triwarp against triwarp: the band-sized Dirichlet system against the mesh-wide extraction.

    ``smooth_region_boundary`` assembles its band's system from the free vertices' vertex-face
    rings (``kernels/smoothing.band_pattern`` and ``band_dirichlet_values``) rather than
    extracting it from a mesh-wide ``-cotmatrix`` with
    [`assemble_interior_system`][triwarp.linalg.assemble_interior_system], which carries the
    oracle here. The pattern -- offsets and sorted columns -- must match exactly, and the values
    and right-hand side to summation order. The band holds a vertex whose ring meets another free
    vertex through two faces, so a neighbour counted twice would show in the offsets.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    vertices_np = np.asarray(mesh_tm.vertices)
    faces_np = np.asarray(mesh_tm.faces)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np.ravel(), device)
    centers_np = vertices_np[faces_np].mean(axis=1)
    region_np = centers_np[:, 0] + 0.3 * np.sin(5.0 * centers_np[:, 1]) > 0.1
    inside_np = np.zeros(len(vertices_np), dtype=bool)
    inside_np[faces_np[region_np].ravel()] = True
    outside_np = np.zeros(len(vertices_np), dtype=bool)
    outside_np[faces_np[~region_np].ravel()] = True
    free_np = inside_np & outside_np
    fixed_wp = wp.array(~free_np, dtype=wp.bool, device=device)
    field_np = np.where(inside_np, -1.0, 1.0)[None, :]
    field_wp = twt.as_array2d(wp.array(field_np, dtype=wp.float64, device=device), wp.float64)
    free_map_wp, n_free = tw.linalg.free_partition(fixed_wp)
    assert n_free > 10

    operator = wps.bsr_scale(tw.laplacian.cotmatrix(vertices_wp, faces_wp, dtype=wp.float64), -1.0)
    assert twt.has_blocks(operator, wp.float64)
    system_wp, rhs_wp = tw.linalg.assemble_interior_system(
        operator, fixed_wp, free_map_wp, field_wp, n_free
    )

    vertex_faces = tw.adjacency.vertex_face_adjacency(faces_wp)
    offsets_wp, columns_wp = tw.smoothing._band_pattern(  # pyright: ignore[reportPrivateUsage]
        faces_wp, vertex_faces, fixed_wp, free_map_wp, n_free
    )
    values_wp = warp_empty(columns_wp.size, wp.float64, device)
    band_rhs_wp = warp_empty((1, n_free), wp.float64, device)
    wp.launch(
        kernel_smoothing.band_dirichlet_values,
        dim=len(vertices_np),
        inputs=[
            fixed_wp,
            free_map_wp,
            vertex_faces[1],
            vertex_faces[0],
            faces_wp,
            vertices_wp,
            field_wp,
            offsets_wp,
            columns_wp,
        ],
        outputs=[values_wp, band_rhs_wp],
        device=device,
    )
    nnz = int(system_wp.nnz_sync())
    row_lengths_np = np.diff(offsets_wp.numpy())
    assert row_lengths_np.max() > row_lengths_np.min()
    assert np.array_equal(offsets_wp.numpy(), system_wp.offsets.numpy()[: n_free + 1])
    assert np.array_equal(columns_wp.numpy(), system_wp.columns.numpy()[:nnz])
    assert np.allclose(values_wp.numpy(), system_wp.values.numpy()[:nnz], rtol=1e-12, atol=1e-12)
    assert np.allclose(band_rhs_wp.numpy(), rhs_wp.numpy(), rtol=1e-12, atol=1e-12)


def test_project_to_zero_isoline_handles_an_exact_field_tie(device: str) -> None:
    """
    Regression: an exact field tie used to drop a triangle's candidate crossing outright.

    ``project_to_zero_isoline``'s apex is the corner whose sign differs from the other two; when
    that apex and one of its neighbours happen to share the exact same field value (both exactly
    zero, which the harmonic field genuinely can hit at a vertex the region boundary passes through
    symmetrically), the two used to fall through to ``continue`` rather than recognising that, by
    linearity, the whole edge between the tied pair lies on the level set.

    Reaching an exact ``0.0`` tie through ``smooth_region_boundary``'s harmonic solve needs an
    elaborately mirror-symmetric mesh and region (the solve is otherwise never bit-exact), and even
    then the tied vertex itself never visibly moves -- it is already sitting on the level set, so
    "stay put" is the right answer whether or not the bug is fixed. What is observable is a *third*
    free vertex that shares this same triangle without being part of the tied pair: this builds the
    kernel's inputs directly, the way its docstring specifies them, so the tie is exact by
    construction rather than by a fragile numerical coincidence.
    """
    # Triangle (tied_a, other, tied_b): the first and third corners share field 0.0 exactly, and
    # the second is some nonzero value -- the ordering that routes through
    # ``project_to_zero_isoline``'s final branch, where the apex (the third corner) ties with the
    # first rather than differing from both. ``other`` sits far off the tied edge, so a fixed value
    # is a poor answer next to the segment's actual closest point, and the difference is a mutation
    # probe: reverting the fix brings ``continue`` back and ``other`` stops moving.
    positions_np = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    positions_wp = wp.array(positions_np, dtype=wp.vec3, device=device)
    faces_wp = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    field_wp = wp.array([0.0, 1.0, 0.0], dtype=wp.float64, device=device)
    free_wp = wp.array([False, True, False], dtype=wp.bool, device=device)
    # Only vertex 1 ("other") is free, so only its CSR row is ever read; face 0 is its one
    # incident face.
    offsets_wp = wp.array([0, 0, 1, 1], dtype=wp.int32, device=device)
    vertex_faces_wp = wp.array([0], dtype=wp.int32, device=device)
    # The free vertex's field value arrives as the reduced solve's answer, at its rank 0.
    free_map_wp = wp.array([0, 0, 1], dtype=wp.int32, device=device)
    solution_wp = wp.array([1.0], dtype=wp.float64, device=device)
    out_wp = warp_empty(3, wp.vec3, device)
    wp.launch(
        kernel_smoothing.project_to_zero_isoline,
        dim=3,
        inputs=[
            positions_wp,
            faces_wp,
            offsets_wp,
            vertex_faces_wp,
            field_wp,
            free_wp,
            free_map_wp,
            solution_wp,
            1.0,
        ],
        outputs=[out_wp],
        device=device,
    )
    out_np = out_wp.numpy()
    # The tied edge is the segment from (0, 0, 0) to (0, 1, 0); its closest point to (2, 2, 2) is
    # the far endpoint (0, 1, 0), not the untouched input position.
    assert np.allclose(out_np[1], [0.0, 1.0, 0.0], atol=1e-6)


def _scalar_spike(mesh_tm: tm.Trimesh) -> np.ndarray:
    """Build a delta at vertex 0: the field with the steepest possible gradient, in float64."""
    values_np = np.zeros(mesh_tm.vertices.shape[0], dtype=np.float64)
    values_np[0] = 10.0
    return values_np


@pytest.mark.parametrize("mesh_name", ["icosahedron", "torus", "cave_cube"])
@pytest.mark.parity("filter_scalar_laplacian", "pymeshlab")
def test_filter_scalar_laplacian_matches_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: one full-step pass is exactly MeshLab's ``apply_scalar_smoothing_per_vertex``.

    **Closed fixtures only**, and deliberately so: VCG's ``VertexQualityLaplacian`` smooths a
    *boundary* vertex along the boundary curve alone — averaging only its two boundary neighbours
    and discarding the rest of its ring — while this port averages every ring for every vertex.
    Measured on ``hemisphere``, a boundary vertex two boundary-steps from a spike reads ``5.0``
    there against ``3.33`` here, because its divisor is 2 rather than its degree 3. That is a
    different operator, not a tolerance, so the oracle is only claimed where no vertex has a
    boundary.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    values_np = _scalar_spike(mesh_tm)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm, values_np)
    meshset_pml.apply_scalar_smoothing_per_vertex()

    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.device)
    smoothed_wp = tw.smoothing.filter_scalar_laplacian(
        values_wp, mesh_wp.points, mesh_wp.indices, lamb=1.0, iterations=1
    )
    assert np.allclose(
        smoothed_wp.numpy(), meshset_pml.current_mesh().vertex_scalar_array(), rtol=1e-5, atol=1e-6
    )


def test_filter_scalar_laplacian_conserves_the_mean_on_a_closed_mesh(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Repeated passes flatten the field without moving its total, on a mesh of uniform valence.

    The averaging operator is row-stochastic but not column-stochastic in general, so the sum is
    only conserved when every vertex has the same degree — which an icosahedron does (valence 5
    everywhere). That makes it the one fixture where this is an exact invariant.
    """
    mesh_tm, mesh_wp = icosahedron
    values_np = _scalar_spike(mesh_tm)
    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.device)

    smoothed_wp = tw.smoothing.filter_scalar_laplacian(
        values_wp, mesh_wp.points, mesh_wp.indices, lamb=0.5, iterations=40
    )
    assert np.isclose(smoothed_wp.numpy().sum(), values_np.sum(), rtol=1e-4)
    assert smoothed_wp.numpy().std() < values_np.std()
    assert np.array_equal(values_wp.numpy(), values_np.astype(np.float32))  # input untouched


def test_filter_scalar_laplacian_zero_iterations(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    values_np = _scalar_spike(mesh_tm).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=mesh_wp.device)
    out_wp = tw.smoothing.filter_scalar_laplacian(
        values_wp, mesh_wp.points, mesh_wp.indices, iterations=0
    )
    assert np.array_equal(out_wp.numpy(), values_np)


def test_filter_scalar_laplacian_length_mismatch(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    values_wp = wp.zeros(3, dtype=wp.float32, device=mesh_wp.device)
    with pytest.raises(ValueError, match="one entry per vertex"):
        tw.smoothing.filter_scalar_laplacian(values_wp, mesh_wp.points, mesh_wp.indices)


# ---------------------------------------------------------------------------
# Feature-preserving smoothing vs pymeshlab (two-step / normal filtering / unsharp)
# ---------------------------------------------------------------------------


def _noisy_cube(scale: float = 0.01, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a subdivided unit cube plus noise: flat faces, 90-degree creases, and grit."""
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0]).subdivide().subdivide().subdivide()
    clean_np = np.ascontiguousarray(box_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(box_tm.faces, dtype=np.int32)
    rng = np.random.default_rng(seed)
    noisy_np = clean_np + rng.normal(scale=scale, size=clean_np.shape)
    return clean_np, faces_np, noisy_np


def _dihedral_percentile(vertices_np: np.ndarray, faces_np: np.ndarray, percentile: float) -> float:
    """Percentile of the unsigned dihedral in degrees: how sharp the sharpest edges still are."""
    mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
    return float(np.percentile(np.rad2deg(np.abs(mesh_tm.face_adjacency_angles)), percentile))


def _rms_error(vertices_np: np.ndarray, clean_np: np.ndarray) -> float:
    return float(np.sqrt(((vertices_np - clean_np) ** 2).sum(axis=1)).mean())


def test_filter_normals_are_unit_and_crease_gated(device: str) -> None:
    """The gate is the whole algorithm: at 0 degrees nothing averages, at 180 everything does."""
    clean_np, faces_np, noisy_np = _noisy_cube()
    vertices_wp = points_to_warp(noisy_np, device)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    raw_wp, _areas = tw.triangles.face_normals_and_areas(vertices_wp, faces_wp)

    gated_wp = tw.smoothing.filter_normals(vertices_wp, faces_wp, iterations=20, threshold=60.0)
    assert np.allclose(np.linalg.norm(gated_wp.numpy(), axis=1), 1.0, rtol=1e-4, atol=1e-4)

    # Threshold 0 lets nothing average, so the field is the geometric one (up to renormalization).
    untouched_wp = tw.smoothing.filter_normals(vertices_wp, faces_wp, threshold=0.0)
    assert np.allclose(untouched_wp.numpy(), raw_wp.numpy(), rtol=1e-4, atol=1e-4)

    # Threshold 180 averages across the cube's edges too, so the normals collapse toward each other.
    isotropic_wp = tw.smoothing.filter_normals(vertices_wp, faces_wp, threshold=180.0)
    assert (
        np.abs(isotropic_wp.numpy() - raw_wp.numpy()).max()
        > np.abs(gated_wp.numpy() - raw_wp.numpy()).max()
    )
    assert clean_np.shape[0] > 0


@pytest.mark.parity("filter_normals", "meshlib")
def test_filter_normals_matches_meshlib(device: str) -> None:
    """
    Class C (a recovery statistic): two normal denoisers, both recovering the same clean field.

    ``denoiseNormals`` is a different formulation -- an L1/total-variation minimization over the
    face graph, weighted per **undirected edge** and regularized by ``gamma`` -- where
    ``filter_normals`` runs crease-gated diffusion passes. Neither parameter maps onto the other's,
    so there is no correspondence and the comparison is what a denoiser is *for*: how close each
    gets to the normals of the mesh before the noise was added.

    Measured on ``icosphere(3)`` displaced by Gaussian noise at 2 % of the radius, as mean and worst
    ``|dot|`` against the clean face normals:

    | | mean | worst |
    |---|---|---|
    | noisy input | 0.9604 | **0.665** |
    | triwarp, 20 passes | 0.99943 | 0.9942 |
    | meshlib, ``gamma=20`` | 0.99867 | 0.9879 |
    | triwarp against meshlib | 0.99856 | 0.9843 |

    So both recover the field, and **they agree with each other more closely than either agrees
    with the input** -- which is the claim, and the mutation probe is the input row itself: a
    filter that did nothing would score 0.9604 / 0.665 and fail every assert below by a wide
    margin.

    Two interface facts. ``denoiseNormals`` mutates the ``FaceNormals`` it is handed and returns
    nothing, so the field is recomputed with ``computePerFaceNormals`` first; and its ``v`` argument
    is a per-**undirected-edge** weight array that must be sized to
    ``topology.undirectedEdgeSize()`` -- there is no default, and a wrongly sized one is not
    checked.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=3)
    clean_vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    clean_normals_wp, _areas_wp = tw.triangles.face_normals_and_areas(clean_vertices_wp, faces_wp)
    clean_normals_np = clean_normals_wp.numpy()

    rng = np.random.default_rng(0)
    noisy_np = mesh_tm.vertices + rng.normal(scale=0.02, size=mesh_tm.vertices.shape)
    noisy_vertices_wp, _faces_wp = numpy_to_warp(noisy_np, mesh_tm.faces, device)
    raw_normals_wp, _areas_wp = tw.triangles.face_normals_and_areas(noisy_vertices_wp, faces_wp)

    smoothed_wp = tw.smoothing.filter_normals(
        noisy_vertices_wp, faces_wp, iterations=20, threshold=60.0
    )

    mesh_ml = numpy_to_meshlib(noisy_np, mesh_tm.faces)
    normals_ml = mm.computePerFaceNormals(mesh_ml)  # mutated in place by the call below
    weights_ml = mm.UndirectedEdgeScalars()
    weights_ml.resize(mesh_ml.topology.undirectedEdgeSize(), 1.0)
    assert mm.denoiseNormals(mesh_ml, normals_ml, weights_ml, 20.0) is None
    denoised_ml = cast("np.ndarray", mn.toNumpyArray(normals_ml))

    def agreement(a_np: np.ndarray, b_np: np.ndarray) -> tuple[float, float]:
        dots_np = np.abs(np.einsum("ij,ij->i", a_np, b_np))
        return float(dots_np.mean()), float(dots_np.min())

    raw_mean, raw_worst = agreement(raw_normals_wp.numpy(), clean_normals_np)
    wp_mean, wp_worst = agreement(smoothed_wp.numpy(), clean_normals_np)
    ml_mean, ml_worst = agreement(denoised_ml, clean_normals_np)
    pair_mean, pair_worst = agreement(smoothed_wp.numpy(), denoised_ml)

    assert raw_worst < 0.8  # non-vacuity: the input really is noisy
    assert min(wp_worst, ml_worst) > 0.95  # both recovered the field
    assert min(wp_mean, ml_mean) > raw_mean
    # And they agree with each other better than either agrees with what they were given.
    assert pair_mean > raw_mean
    assert pair_worst > raw_worst


def test_filter_normals_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match=r"threshold must be in \[0, 180\]"):
        tw.smoothing.filter_normals(mesh_wp.points, mesh_wp.indices, threshold=-1.0)


def test_filter_two_step_denoises_without_rounding_the_creases(device: str) -> None:
    """
    The claim that separates this filter from every other one in the module, tested both ways.

    On a noisy cube two-step must (a) get *closer* to the clean mesh than the noise was, and (b)
    leave the 90-degree creases at 90 degrees. Isotropic Laplacian smoothing fails both: measured,
    it moves the mesh **further** from clean (RMS 0.016 to 0.047, and 0.016 to 0.097 on the crease
    vertices alone) and drops the 95th-percentile dihedral from 88 to 25 degrees. Taubin, which
    exists to fix Laplacian shrinkage, still lands at 56 — shrinkage was never the problem here.
    """
    clean_np, faces_np, noisy_np = _noisy_cube()
    vertices_wp = points_to_warp(noisy_np, device)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )

    two_step_np = tw.smoothing.filter_two_step(vertices_wp, faces_wp).numpy().astype(np.float64)
    laplacian_np = (
        tw.smoothing.filter_laplacian(vertices_wp, faces_wp, iterations=10)
        .numpy()
        .astype(np.float64)
    )

    assert _rms_error(two_step_np, clean_np) < _rms_error(noisy_np, clean_np)
    assert _rms_error(two_step_np, clean_np) < _rms_error(laplacian_np, clean_np)

    # Creases survive: the sharp end of the dihedral distribution stays where the clean cube has it.
    assert _dihedral_percentile(two_step_np, faces_np, 95.0) > 85.0
    assert _dihedral_percentile(laplacian_np, faces_np, 95.0) < 50.0

    # And the improvement is concentrated on the crease vertices, which is the point.
    on_edge_np = (np.abs(np.abs(clean_np) - 0.5) < 1e-6).sum(axis=1) >= 2
    assert on_edge_np.any()
    assert _rms_error(two_step_np[on_edge_np], clean_np[on_edge_np]) < _rms_error(
        noisy_np[on_edge_np], clean_np[on_edge_np]
    )


@pytest.mark.parity("filter_two_step", "pymeshlab")
def test_filter_two_step_matches_pymeshlab_on_crease_preservation(device: str) -> None:
    """
    Class C (a crease measure): the same two-stage scheme at the same four parameters.

    The measure both must pass is crease preservation, and both do: 95th-percentile dihedral 87
    degrees for MeshLab against this port's 90, where plain Laplacian smoothing collapses to 25.

    **Denoising is where they part company, and not in MeshLab's favour.** At the same four
    parameters its output is *further* from the clean cube than the noise was — RMS 0.0244 against
    the noise's 0.0156, against this port's 0.0114 — because its fitting step rounds the corners
    inward (a corner at ``(-0.5, -0.5, -0.5)`` comes back at ``(-0.43, -0.49, -0.42)``). So the
    assertion below is one-sided: this port must be at least as close to clean as MeshLab, not equal
    to it.
    """
    clean_np, faces_np, noisy_np = _noisy_cube()
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(ml.Mesh(noisy_np, faces_np))
    meshset_pml.apply_coord_two_steps_smoothing(
        stepsmoothnum=3, normalthr=60.0, stepnormalnum=20, stepfitnum=20
    )
    pml_np = np.ascontiguousarray(meshset_pml.current_mesh().vertex_matrix(), dtype=np.float64)

    vertices_wp = points_to_warp(noisy_np, device)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    two_step_np = tw.smoothing.filter_two_step(vertices_wp, faces_wp).numpy().astype(np.float64)

    assert _rms_error(two_step_np, clean_np) <= _rms_error(pml_np, clean_np)
    assert _dihedral_percentile(pml_np, faces_np, 95.0) > 85.0
    assert _dihedral_percentile(two_step_np, faces_np, 95.0) > 85.0


@pytest.mark.parity(
    "filter_two_step",
    "meshlib",
    benchmarked=False,
    reason="Tested but not timed: meshDenoiseViaNormals is a different scheme at "
    "unmappable parameters, so a timed row would compare two different amounts of "
    "work per call. pymeshlab, which exposes the same four parameters, is timed.",
)
def test_filter_two_step_matches_meshlib_on_crease_preserving_denoising(device: str) -> None:
    """
    Class C (denoising *and* a crease measure): a second crease-preserving filter, both halves.

    The second oracle for this group, and it closes the half the first one leaves open.
    [`test_filter_two_step_matches_pymeshlab_on_crease_preservation`][tests.test_smoothing.test_filter_two_step_matches_pymeshlab_on_crease_preservation]
    can only assert the crease half, because MeshLab at the same four parameters ends *further* from
    the clean cube than the noise was -- so nothing there confirms that a crease-preserving filter
    should also denoise. ``meshDenoiseViaNormals`` does both, which makes the conjunction testable:
    measured RMS to clean **0.012712** against the noise's **0.015620**, with the 95th-percentile
    dihedral at **86.43** degrees.

    The two are different algorithms and their parameters do not correspond -- the reference is
    parametrized by ``beta`` and ``gamma`` and has no crease threshold in degrees at all -- so this
    compares outcomes at each side's own defaults rather than positions. They land close all the
    same: mutual RMS **0.004979**, max coordinate deviation **0.019910**, which is under half the
    noise amplitude. triwarp is the closer of the two to clean (0.011415).

    The bug class excluded is the one this filter exists to avoid: a scheme that blurs a crease
    while reporting a lower residual. Both sides must clear *both* thresholds, and isotropic
    Laplacian smoothing is the probe -- at 10 iterations it reads dihedral **25.18** against the
    85.0 bar (a **3.4x** margin) and crease RMS **0.097202** against the reference's 0.011670
    (**8.3x**), while moving total RMS the wrong way to 0.046993.
    """
    clean_np, faces_np, noisy_np = _noisy_cube()
    vertices_wp = points_to_warp(noisy_np, device)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    two_step_np = tw.smoothing.filter_two_step(vertices_wp, faces_wp).numpy().astype(np.float64)
    laplacian_np = (
        tw.smoothing.filter_laplacian(vertices_wp, faces_wp, iterations=10)
        .numpy()
        .astype(np.float64)
    )

    mesh_ml = numpy_to_meshlib(noisy_np, faces_np)
    mm.meshDenoiseViaNormals(mesh_ml, mm.DenoiseViaNormalsSettings())
    denoised_np = np.ascontiguousarray(meshlib_to_trimesh(mesh_ml).vertices, dtype=np.float64)
    assert denoised_np.shape == clean_np.shape  # the topology is fixed, so nothing was packed away

    noise_error = _rms_error(noisy_np, clean_np)
    # Both denoise, and neither rounds the creases off -- the conjunction is the claim.
    assert _rms_error(denoised_np, clean_np) < noise_error
    assert _rms_error(two_step_np, clean_np) < noise_error
    assert _dihedral_percentile(denoised_np, faces_np, 95.0) > 85.0
    assert _dihedral_percentile(two_step_np, faces_np, 95.0) > 85.0

    # Two different algorithms, so outcomes rather than positions -- but they land close.
    assert _rms_error(two_step_np, denoised_np) < 0.5 * noise_error

    # The probe: an isotropic filter fails both halves on the same input.
    assert _rms_error(laplacian_np, clean_np) > noise_error
    assert _dihedral_percentile(laplacian_np, faces_np, 95.0) < 50.0


def test_filter_two_step_leaves_a_flat_patch_alone(device: str) -> None:
    """A plane is a fixed point of both halves: filtered normals are already the geometric ones."""
    grid_vertices_wp, grid_faces_wp = tw.creation.grid(count=(12, 12), device=device)
    smoothed_wp = tw.smoothing.filter_two_step(grid_vertices_wp, grid_faces_wp)
    assert np.allclose(smoothed_wp.numpy(), grid_vertices_wp.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("iterations", [1, 5])
@pytest.mark.parity("filter_sharpen", "pymeshlab")
def test_filter_sharpen_matches_pymeshlab(device: str, iterations: int) -> None:
    """
    Class B: exact under the same operator substitution as the Laplacian test above.

    See [`test_filter_laplacian_matches_pymeshlab`]
    [tests.test_smoothing.test_filter_laplacian_matches_pymeshlab] for how the operator is derived.

    ``apply_coord_unsharp_mask`` takes ``weight`` and ``iterations`` under triwarp's own names and
    meanings, and ``weightorig=1.0`` is its default (the original-position weight triwarp fixes at
    1), so the only transform is passing
    [`_meshlab_umbrella`][tests.test_smoothing._meshlab_umbrella] as the smoothing operator.

    Measured **1.4e-07** at both 1 and 5 iterations, against **1.5e-03 / 4.4e-03** with the plain
    1-ring mean -- so, like the Laplacian test, this doubles as the check on which umbrella the
    filter uses.
    """
    mesh_tm = _noisy_icosphere()
    mesh_wp = trimesh_to_warp(mesh_tm, device)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.apply_coord_unsharp_mask(weight=0.3, weightorig=1.0, iterations=iterations)

    sharpened_wp = tw.smoothing.filter_sharpen(
        mesh_wp.points,
        mesh_wp.indices,
        weight=0.3,
        iterations=iterations,
        laplacian_operator=_meshlab_umbrella(mesh_tm, device),
    )
    assert np.allclose(
        sharpened_wp.numpy(), meshset_pml.current_mesh().vertex_matrix(), rtol=1e-5, atol=1e-5
    )


def test_filter_sharpen_amplifies_detail(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """Sharpening inverts smoothing, so it must move the mesh *away* from its smooth self."""
    sphere_tm, _sphere_tm_wp = icosphere
    rng = np.random.default_rng(4)
    bumpy_np = np.asarray(sphere_tm.vertices) * (
        1.0 + rng.normal(scale=0.02, size=(sphere_tm.vertices.shape[0], 1))
    )
    vertices_wp = points_to_warp(bumpy_np, device)
    faces_wp = wp.array(
        np.ascontiguousarray(sphere_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )

    smoothed_np = (
        tw.smoothing.filter_laplacian(
            vertices_wp, faces_wp, lamb=1.0, iterations=5, volume_constraint=False
        )
        .numpy()
        .astype(np.float64)
    )
    sharpened_np = (
        tw.smoothing.filter_sharpen(vertices_wp, faces_wp, weight=0.5, iterations=5)
        .numpy()
        .astype(np.float64)
    )
    detail_np = bumpy_np - smoothed_np
    # The sharpened mesh is the original plus half the detail, so it is exactly that much further
    # out.
    assert np.allclose(sharpened_np, bumpy_np + 0.5 * detail_np, rtol=1e-3, atol=1e-4)
    assert np.abs(sharpened_np - smoothed_np).max() > np.abs(bumpy_np - smoothed_np).max()


def test_filter_sharpen_zero_weight_is_the_identity(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _mesh_tm, mesh_wp = icosahedron
    out_wp = tw.smoothing.filter_sharpen(mesh_wp.points, mesh_wp.indices, weight=0.0)
    assert np.allclose(out_wp.numpy(), mesh_wp.points.numpy(), rtol=1e-6, atol=1e-6)


# (name, callable) pairs; each callable takes (vertices, faces) and returns the smoothed vertex
# array a vertex-position filter produces. filter_scalar_laplacian is not among these: it smooths
# a per-vertex scalar field rather than positions and takes a values array first, a genuinely
# different signature rather than a boilerplate variation.
_SMOOTHING_VERTEX_FILTER_CASES = [
    (
        "filter_taubin",
        lambda v, f, iterations: tw.smoothing.filter_taubin(v, f, iterations=iterations),
    ),
    (
        "filter_neighborhood_average",
        lambda v, f, iterations: tw.smoothing.filter_neighborhood_average(
            v, f, iterations=iterations
        ),
    ),
    (
        "filter_two_step",
        lambda v, f, iterations: tw.smoothing.filter_two_step(v, f, iterations=iterations),
    ),
]


@pytest.mark.parametrize(
    "smoothing_fn",
    [case[1] for case in _SMOOTHING_VERTEX_FILTER_CASES],
    ids=[case[0] for case in _SMOOTHING_VERTEX_FILTER_CASES],
)
def test_smoothing_zero_iterations_returns_a_copy(
    smoothing_fn: Callable[[wp.array[wp.vec3], wp.array[wp.int32], int], wp.array[wp.vec3]],
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Not a library comparison: zero iterations is the identity for a vertex-position filter."""
    _, mesh_wp = icosahedron
    smoothed_wp = smoothing_fn(mesh_wp.points, mesh_wp.indices, 0)
    assert np.array_equal(smoothed_wp.numpy(), mesh_wp.points.numpy())


# filter_taubin's own case additionally requests real iterations (4), confirming the empty-mesh
# guard fires before the solve rather than merely being what a 0-iteration no-op would look like
# anyway; the other two filters keep their default iteration counts, as the original tests did.
@pytest.mark.parametrize(
    "smoothing_fn",
    [
        lambda v, f: tw.smoothing.filter_taubin(v, f, iterations=4),
        lambda v, f: tw.smoothing.filter_two_step(v, f),
        lambda v, f: tw.smoothing.filter_sharpen(v, f),
    ],
    ids=["filter_taubin", "filter_two_step", "filter_sharpen"],
)
def test_smoothing_empty_mesh_is_a_noop(
    device: str, smoothing_fn: Callable[[wp.array[wp.vec3], wp.array[wp.int32]], wp.array[wp.vec3]]
) -> None:
    """Not a library comparison: every vertex-position filter returns nothing on an empty mesh."""
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = warp_empty(0, wp.int32, device)
    assert smoothing_fn(vertices_wp, faces_wp).shape == (0,)


class _RefineKwargs(TypedDict):
    max_edge_splits: int
    max_angle_change_after_flip: float
    smooth_boundary: bool
    natural_smooth: bool
    edge_weights: str


_REFINE_KWARGS: _RefineKwargs = {
    "max_edge_splits": 3,
    "max_angle_change_after_flip": 0.5,
    "smooth_boundary": False,
    "natural_smooth": False,
    "edge_weights": "cotangent",
}


def test_filter_laplacian_leaves_a_faceless_mesh_alone(device: str) -> None:
    """
    Triwarp against triwarp: the volume constraint must not touch a mesh it cannot measure.

    Not a library comparison: trimesh's ``filter_laplacian`` requires a real mesh, so there is no
    reference for the degenerate input. The invariant is that a face buffer with no faces leaves
    every position exactly where it was.

    The regression this pins is specific and was introduced by moving the rescale onto the device:
    with no faces the mesh has no centre of mass, so the rescale's ``center`` is ``NaN``, and a
    "skip" spelled as a scale of ``1.0`` is ``(p - NaN) * 1 + NaN`` -- which propagates instead of
    cancelling. The skip has to decline to write at all.
    """
    positions_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    vertices_wp = points_to_warp(positions_np, device)
    faces_wp = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32, device=device)

    smoothed_np = tw.smoothing.filter_laplacian(vertices_wp, faces_wp, iterations=3).numpy()
    assert np.isfinite(smoothed_np).all()
    assert np.array_equal(smoothed_np, positions_np)
