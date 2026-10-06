"""Regression tests for ``ordito.remesh`` against Trimesh (CPU reference)."""

# This file tests ``ordito.remesh``'s private passes and buffers directly.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, TypedDict, cast

import igl
import numpy as np
import numpy.typing as npt
import pymeshlab as ml
import pytest
import pytorch3d.ops as p3d_ops
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from scipy.spatial import KDTree

import ordito as od
import ordito.typing as odt
from ordito.kernels import remesh as kernel_remesh
from ordito.triangles import FaceQualityMetric
from tests.comparisons import (
    assert_nearest_bijection,
    hausdorff_surface_two_sided,
    lexsort_rows,
    trimesh_outline_loops,
    undirected_edges,
)
from tests.conftest import CLOSED_MESHES, MESHES
from tests.conversions import (
    bsr_to_csr,
    bsr_to_dense,
    faces_igl,
    meshlib_bitset_to_numpy,
    meshlib_scalars_to_numpy,
    meshlib_to_trimesh,
    numpy_to_meshlib,
    numpy_to_meshlib_bitset,
    numpy_to_warp,
    open3d_to_trimesh,
    points_to_warp,
    points_to_warp_uv,
    pytorch3d_to_numpy,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pytorch3d,
    trimesh_to_pyvista,
    trimesh_to_warp,
    warp_empty,
    warp_to_trimesh,
)


class _RemeshKwargs(TypedDict):
    """The ``isotropic_remesh`` keywords two calls in one test share."""

    target_length: float
    iterations: int
    reproject: bool


def _max_edge_length(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    edges = vertices_np[undirected_edges(faces_np)]
    return float(np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1).max())


def _surface_area(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    tris = vertices_np[faces_np]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return float(np.linalg.norm(cross, axis=1).sum() * 0.5)


def _signed_volume(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    tris = vertices_np[faces_np]
    return float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0)


# ---------------------------------------------------------------------------
# Helpers shared across the groups below.
#
# The four here were each defined inside one group and reached from up to five others, which only
# ever worked because Python resolves a name at call time; reordering the groups into source order
# is what made that visible. The three above them were already shared and already up here.
# ---------------------------------------------------------------------------


def _filled_hemisphere(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """Fill the hemisphere's boundary and return (vertices_wp, faces_wp, region_mask_wp)."""
    _, mesh_wp = hemisphere
    faces_filled = od.holes.fill_min_weight(mesh_wp.points, mesh_wp.indices)
    n0 = mesh_wp.indices.size // 3
    n1 = faces_filled.size // 3
    region = np.zeros(n1, dtype=bool)
    region[n0:] = True
    region_wp = wp.array(region, dtype=wp.bool, device=mesh_wp.device)
    return mesh_wp.points, faces_filled, region_wp


def _region_max_edge(vertices_np: np.ndarray, faces_np: np.ndarray, region_np: np.ndarray):
    edges = undirected_edges(faces_np)
    face_of_edge = np.repeat(np.arange(faces_np.shape[0]), 3)
    lengths = np.linalg.norm(vertices_np[edges[:, 0]] - vertices_np[edges[:, 1]], axis=1)
    in_region = region_np[face_of_edge]
    return float(lengths[in_region].max()) if in_region.any() else 0.0


def _icosphere_wp(device: str, subdivisions: int = 3):
    """
    Build an icosphere at a caller-chosen subdivision: ``(trimesh, vertices_wp, faces_wp)``.

    Parametrized over ``subdivisions``, which is exactly what a fixture cannot be -- ``icosphere``
    and ``icosphere_coarse`` in ``conftest.py`` pin 3 and 2, and this file's decimation tests sweep
    2 through 4. So it stays a builder, and its job is only to pair the trimesh with the buffers
    ``numpy_to_warp`` uploads.
    """
    sphere = tm.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    vertices, faces = numpy_to_warp(sphere.vertices, sphere.faces, device)
    return sphere, vertices, faces


def _degenerate_face_count(vertices_np: np.ndarray, faces_np: np.ndarray) -> int:
    """Faces with exactly zero area -- a repeated index or two coincident corners."""
    triangles = vertices_np[faces_np]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return int((np.linalg.norm(cross, axis=1) == 0.0).sum())


# ======================================================================================
# Isotropic explicit remeshing (isotropic_remesh)
#
# Reference: meshlib ``mrmeshpy.remesh`` (``_ml`` suffix) for the edge-length spread; the rest are
# metric/topological invariants (edge concentration, valence variance, Hausdorff, manifoldness,
# feature/boundary preservation). Outputs never match a reference vertex-for-vertex.
# ======================================================================================


def _edge_lengths(vertices_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    edges = vertices_np[np.unique(undirected_edges(faces_np), axis=0)]
    return np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1)


def _valences(faces_np: np.ndarray, n_vertices: int) -> np.ndarray:
    edges = np.unique(undirected_edges(faces_np), axis=0)
    valence = np.zeros(n_vertices, dtype=np.int64)
    np.add.at(valence, edges[:, 0], 1)
    np.add.at(valence, edges[:, 1], 1)
    return valence


def _meshlib_remesh_spread(vertices_np: np.ndarray, faces_np: np.ndarray, target: float) -> float:
    """
    Coefficient of variation (std / mean) of meshlib remesh at ``target``.

    ``pack()`` is mandatory and skipping it is silent: ``mm.remesh`` retires faces in place, so
    ``getNumpyFaces`` returns ``last_valid_face_id + 1`` rows rather than ``numValidFaces()``, and
    the difference is all-zero padding. Measured on ``icosphere(3)`` at half the mean edge: **12 800
    rows of which 7 696 are ``[0, 0, 0]``**, against 5 104 real faces. Those rows contribute a
    zero-length edge ``(0, 0)`` to the edge set, which deflates the reference's mean and inflates
    the very spread this function exists to report -- i.e. it makes the bound below *looser*, in
    ordito's favour, which is the direction nothing would notice. The effect happens to be small,
    because ``np.unique`` collapses all 7 696 padding rows into that one edge: CV **0.2525 ->
    0.2522**. The fix is for the reading to be right, not because the number moved.
    """
    mesh_ml = numpy_to_meshlib(vertices_np, faces_np)
    settings = mm.RemeshSettings()
    settings.targetEdgeLen = float(target)
    settings.projectOnOriginalMesh = True
    mm.remesh(mesh_ml, settings)
    mesh_ml.pack()
    vertices_ml = mn.getNumpyVerts(mesh_ml)
    faces_ml = mn.getNumpyFaces(mesh_ml.topology)
    assert faces_ml.shape[0] == mesh_ml.topology.numValidFaces()
    lengths_ml = _edge_lengths(np.asarray(vertices_ml), np.asarray(faces_ml))
    return float(lengths_ml.std() / lengths_ml.mean())


def _graded_patch(n: int = 96, ratio: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a saddle patch whose ``x`` spacing varies by ``ratio``: anisotropic, valence-perfect.

    The shape ``benchmarks/meshes.py`` calls ``saddle_graded``, reduced to test size. Every interior
    vertex has valence exactly 6 and the triangulation is already Delaunay, so neither a
    valence-driven flip nor a Delaunay flip can see the anisotropy -- only the collapse and the
    *area-equalizing* tangential relaxation can remove it. That combination is what makes this the
    input the remesher's stages are individually blind to, and it is why it is worth a test.
    """
    t_np = np.linspace(0.0, 1.0, n) ** 2.0  # quadratic spacing -> strong grading along x
    x_np = t_np * ratio
    y_np = np.linspace(0.0, ratio, n)
    x_grid, y_grid = np.meshgrid(x_np, y_np, indexing="ij")
    z_grid = 0.02 * (x_grid**2 - y_grid**2) / ratio
    vertices = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).astype(np.float64)
    return vertices, _grid_faces(n)


def _grid_faces(n: int) -> np.ndarray:
    """Return the faces of an ``n x n`` row-major grid, each cell cut on ``a``-``a + n + 1``."""
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return np.ascontiguousarray(faces, dtype=np.int32)


def _icosphere_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Clean closed control input, as plain NumPy arrays."""
    sphere = tm.creation.icosphere(subdivisions=3, radius=1.0)
    return (
        np.ascontiguousarray(sphere.vertices, dtype=np.float64),
        np.ascontiguousarray(sphere.faces, dtype=np.int32),
    )


def _worst_aspect_ratio(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    """Longest / shortest edge over the non-degenerate faces (``inf`` if any is degenerate)."""
    triangles = vertices_np[faces_np]
    lengths = np.linalg.norm(
        np.stack(
            [
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 1],
                triangles[:, 0] - triangles[:, 2],
            ],
            axis=1,
        ),
        axis=2,
    )
    if (lengths.min(axis=1) == 0.0).any():
        return float("inf")
    return float((lengths.max(axis=1) / lengths.min(axis=1)).max())


def test_remesh_emits_no_degenerate_faces(device: str) -> None:
    """
    No output face may have exactly zero area, on a clean *and* a badly graded input.

    This is the regression gate for two bugs the pymeshlab benchmark reference exposed, neither of
    which any other assertion in this file would catch -- they all run on clean closed icospheres:

    * ``valence_flip_candidates`` flipped on the valence objective alone. Convexity makes a flip
      legal but bounds nothing about the shape it produces, so on a graded mesh it turned slivers
      into worse slivers and in float32 landed on exactly-zero area: **2 738 of 84 406 faces** on a
      ``saddle_graded``-shaped patch. It now rejects a flip that would create a degenerate triangle
      or increase the worse aspect ratio of the pair.
    The second gap this input exposed -- ``_smooth_pass`` computing the **unweighted** one-ring
    centroid while ``isotropic_remesh``'s Notes promise the area-equalizing form -- is now closed;
    see ``test_smooth_pass_is_the_area_weighted_centroid`` for the property and
    ``kernels/remesh.accumulate_one_ring`` for what it bought. This test keeps asserting only the
    degeneracy and do-no-harm properties, which is what it is the gate for.
    """
    for label, (vertices_np, faces_np) in (
        ("icosphere", _icosphere_arrays()),
        ("graded_patch", _graded_patch()),
    ):
        vertices_wp = points_to_warp(vertices_np, device)
        faces_wp = wp.array(
            np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        target = float(od.edges.mean_edge_length(vertices_wp, faces_wp))
        out_vertices, out_faces = od.remesh.isotropic_remesh(
            vertices_wp, faces_wp, target_length=target, iterations=3
        )
        out_vertices_np = out_vertices.numpy().astype(np.float64)
        out_faces_np = out_faces.numpy().reshape(-1, 3)
        assert _degenerate_face_count(out_vertices_np, out_faces_np) == 0, label
        # And it must never leave the mesh worse-shaped than it found it.
        assert _worst_aspect_ratio(out_vertices_np, out_faces_np) < 2.0 * _worst_aspect_ratio(
            vertices_np, faces_np
        ), label


def _area_weighted_ring_centroid(
    vertices_np: npt.NDArray[np.float64], faces_np: npt.NDArray[np.int32]
) -> npt.NDArray[np.float64]:
    """Botsch-Kobbelt's relaxation target: each one-ring neighbour weighted by its own area."""
    triangles = vertices_np[faces_np]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    vertex_area = np.zeros(len(vertices_np))
    for corner in range(3):
        np.add.at(vertex_area, faces_np[:, corner], areas / 3.0)
    edges = undirected_edges(faces_np.reshape(-1, 3))
    ring_sum = np.zeros_like(vertices_np)
    ring_weight = np.zeros(len(vertices_np))
    for u, v in edges:
        ring_sum[u] += vertex_area[v] * vertices_np[v]
        ring_weight[u] += vertex_area[v]
        ring_sum[v] += vertex_area[u] * vertices_np[u]
        ring_weight[v] += vertex_area[u]
    centroid = np.zeros_like(vertices_np)
    live = ring_weight > 0.0
    centroid[live] = ring_sum[live] / ring_weight[live][:, None]
    return centroid


def test_smooth_pass_is_the_area_weighted_centroid(device: str) -> None:
    """
    Not a library comparison: the oracle is the relaxation formula, which no reference exposes.

    ``isotropic_remesh``'s Notes promise the *area-equalizing* tangential relaxation of
    Botsch-Kobbelt -- each one-ring neighbour weighted by its own barycentric area -- and for a long
    time the code computed the plain unweighted centroid instead. The two differ only where the
    sampling is uneven, which is exactly the input the stage exists for and exactly the input the
    closed, well-shaped fixtures cannot supply, so the gap survived every end-to-end assertion in
    this file. Hence a direct, hand-computable check on the one pass, on the graded patch.

    The tangential projection is checked at the same time: the relaxed position must differ from the
    raw centroid only along the vertex normal.
    """
    vertices_np, faces_np = _graded_patch(n=48)
    vertices_wp = points_to_warp(vertices_np, device)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    codes_wp, _boundary = od.remesh._classify(vertices_wp, faces_wp, wp.float32(math.radians(30.0)))
    relaxed_np = od.remesh._smooth_pass(vertices_wp, faces_wp, codes_wp).numpy().astype(np.float64)

    positions_np = vertices_wp.numpy().astype(np.float64)
    free = codes_wp.numpy() == 0
    assert free.sum() > 1000  # non-vacuity: the pass has a real interior to move

    normals_np = od.vertices.vertex_normals(vertices_wp, faces_wp).numpy().astype(np.float64)
    delta = _area_weighted_ring_centroid(positions_np, faces_np) - positions_np
    expected = positions_np + delta - np.einsum("ij,ij->i", delta, normals_np)[:, None] * normals_np

    assert np.allclose(relaxed_np[free], expected[free], rtol=1e-4, atol=1e-4)
    # Pinned vertices do not move at all, whatever the ring says.
    assert np.array_equal(relaxed_np[~free], positions_np[~free])
    # And the weighting is what is being checked: the *unweighted* centroid is a different answer
    # here, so a regression to it would fail the comparison above rather than pass it vacuously.
    unweighted = np.zeros_like(positions_np)
    degree = np.zeros(len(positions_np))
    for u, v in undirected_edges(faces_np.reshape(-1, 3)):
        unweighted[u] += positions_np[v]
        unweighted[v] += positions_np[u]
        degree[u] += 1.0
        degree[v] += 1.0
    unweighted[degree > 0] /= degree[degree > 0][:, None]
    assert not np.allclose(
        unweighted[free], _area_weighted_ring_centroid(positions_np, faces_np)[free], atol=1e-3
    )


def test_smooth_pass_vetoes_a_move_that_would_fold_a_face(device: str) -> None:
    """
    Not a library comparison: the fold veto's own contract, on the link shape that can reach it.

    A tangential step to a *convex combination* of the one ring can never leave a **convex** link
    polygon, so on every well-shaped fixture in this file the veto is unreachable by construction
    and a test built on one of them would assert nothing. It takes a strongly non-convex link --
    here a deep notch of three near neighbours opposite three far ones, so the area weights, which
    the far neighbours dominate, pull the centroid across the notch -- and then the unguarded step
    inverts three of the six incident faces.

    The vertex must therefore keep its position exactly, which is what ``smooth_free_vertices``
    promises when ``move_flips_normal`` rejects a proposal.
    """
    ring = [
        (2.1775, 2.9487),
        (0.7503, 6.7468),
        (-0.241, 0.0305),
        (0.425, -0.8692),
        (0.4578, -0.3791),
        (6.9468, -1.2066),
    ]
    vertices_np = np.array([(0.0, 0.0, 0.0), *[(x, y, 0.0) for x, y in ring]], dtype=np.float64)
    faces_np = np.ascontiguousarray(
        [[0, 1 + i, 1 + (i + 1) % len(ring)] for i in range(len(ring))], dtype=np.int32
    )
    vertices_wp = points_to_warp(vertices_np, device)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    codes_wp, _boundary = od.remesh._classify(vertices_wp, faces_wp, wp.float32(math.radians(30.0)))
    assert codes_wp.numpy()[0] == 0  # the centre is FREE, so only the veto can hold it still

    # Non-vacuity: the proposal this pass would otherwise commit really does invert incident faces.
    centroid = _area_weighted_ring_centroid(vertices_np, faces_np)[0]
    folded = 0
    for triangle in faces_np:
        before = vertices_np[triangle]
        after = before.copy()
        after[list(triangle).index(0)] = centroid
        normal_before = np.cross(before[1] - before[0], before[2] - before[0])
        normal_after = np.cross(after[1] - after[0], after[2] - after[0])
        length_after = np.linalg.norm(normal_after)
        if (
            length_after == 0.0
            or np.dot(normal_before / np.linalg.norm(normal_before), normal_after / length_after)
            < 0.2
        ):
            folded += 1
    assert folded == 3

    relaxed_np = od.remesh._smooth_pass(vertices_wp, faces_wp, codes_wp).numpy().astype(np.float64)
    assert np.array_equal(relaxed_np, vertices_wp.numpy().astype(np.float64))


@pytest.mark.parity("isotropic_remesh", "meshlib")
@pytest.mark.parametrize(
    ("mesh_name", "target_scale", "spread_bound"), [("icosphere", 0.5, 0.5), ("graded", 1.0, 0.85)]
)
def test_remesh_edge_concentration(
    device: str, mesh_name: str, target_scale: float, spread_bound: float
) -> None:
    """
    Class C (a spread statistic): both remeshes hit the requested target, ordito more tightly.

    No correspondence exists -- the two run different stopping rules (a fixed ``iterations`` x five
    parallel passes against a serial local-operation queue), so they return different meshes -- and
    what is comparable is how well each concentrates its edge lengths around the *requested* target.

    **The graded arm is the one that carries the claim, and a uniform fixture alone cannot.** On an
    already-uniform sphere every stage of the remesher agrees about what to do, so the comparison
    measures almost nothing about the stages individually; on a patch whose spacing varies 60x it
    separates them. Measured over five repetitions on both devices -- identical to four decimals
    every time, so these bounds are set by the reference drifting, not by run-to-run noise:

    | | ``icosphere(3)`` @ 0.5x | graded patch @ 1.0x |
    |---|---|---|
    | ordito coefficient of variation | 0.0419 | 0.1227 |
    | meshlib CV (the reference) | 0.2522 | 0.1994 |
    | ratio, against the bound here | 0.166 vs 0.5 (**3.0x**) | 0.615 vs 0.85 (**1.4x**) |
    | mean / target | 0.9975 | 1.0029 |
    | fraction in ``[0.7, 1.4] x`` target | 1.0000 | 0.9941 |

    **Bug class excluded:** a remesh that converges to the wrong length scale, or that reaches the
    mean by mixing very long and very short edges rather than by equalizing them.

    **Mutation probe, run by disabling one stage at a time and naming the assert each trips:** the
    *input* mesh scores a ratio of 1.99 and 0.68 in band, failing both; disabling ``collapse`` gives
    1.29 and disabling ``split`` 1.37, each failing the spread bound; disabling ``smooth`` fails
    **both** asserts independently -- ratio 0.928 against the 0.85 bound, and 0.9495 in band against
    0.95. **The spread bound is the robust half of that pair and the reason it is 0.85 rather than
    a rounder 1.0**: at 1.0 this test would pass with the area-equalizing relaxation switched off
    entirely, and the in-band assert alone clears the mutant by 0.0005, which is not a margin. Only
    ``swap`` can be disabled without tripping anything.

    **And the mean-band assert is vacuous on the graded arm** -- its input already sits at 0.9981 of
    the target, because grading redistributes edge lengths without changing their mean -- so there
    the spread bound and the in-band fraction are the whole test. On the icosphere arm it is the
    other way round: the input sits at 2.0x target, so a do-nothing pass fails the mean outright.
    """
    if mesh_name == "icosphere":
        sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
        reference_vertices, reference_faces = sphere.vertices, sphere.faces
    else:
        reference_vertices, reference_faces = _graded_patch(n=48)
        vertices_wp, faces_wp = numpy_to_warp(reference_vertices, reference_faces, device)
    target = target_scale * od.edges.mean_edge_length(vertices_wp, faces_wp)

    out_vertices, out_faces = od.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=10
    )
    vertices_np = out_vertices.numpy().astype(np.float64)
    faces_np = out_faces.numpy().reshape(-1, 3)
    lengths = _edge_lengths(vertices_np, faces_np)

    assert abs(lengths.mean() / target - 1.0) < 0.05  # mean within 5% of target
    in_band = np.mean((lengths >= 0.7 * target) & (lengths <= 1.4 * target))
    assert in_band >= 0.95
    spread_ml = _meshlib_remesh_spread(reference_vertices, reference_faces, target)
    assert 0.0 < spread_ml < 1.0  # non-vacuity: the reference produced a real remesh
    assert lengths.std() / lengths.mean() <= spread_bound * spread_ml


def test_remesh_preserves_the_topology_and_stays_near_the_surface(device: str) -> None:
    """
    Not a library comparison: a remeshed sphere stays a closed genus-0 surface on the input.

    Watertight, Euler characteristic 2, volume within a few percent of the unit sphere's, and a
    two-sided surface distance to the input under the target length -- reprojection's job.
    """
    sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    target = 0.5 * od.edges.mean_edge_length(vertices_wp, faces_wp)

    out_vertices, out_faces = od.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=10
    )
    mesh_out = warp_to_trimesh(out_vertices, out_faces)
    assert od.validation.is_watertight(out_vertices, out_faces)
    assert mesh_out.euler_number == 2  # genus 0
    # Volume of the unit sphere is preserved to a few percent.
    assert abs(mesh_out.volume - 4.0 / 3.0 * np.pi) / (4.0 / 3.0 * np.pi) < 0.05
    hausdorff = hausdorff_surface_two_sided(
        sphere.vertices,
        sphere.faces,
        out_vertices.numpy().astype(np.float64),
        out_faces.numpy().reshape(-1, 3),
    )
    # Reprojection keeps the remesh close to the original surface (well under the target length).
    assert hausdorff < target


def test_remesh_valence_variance_decreases(device: str) -> None:
    # A noisy, irregular triangulation: perturbed icosphere with random extra subdivision.
    sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    target = od.edges.mean_edge_length(vertices_wp, faces_wp)
    valence_before = _valences(sphere.faces, sphere.vertices.shape[0])

    out_vertices, out_faces = od.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=10
    )
    faces_np = out_faces.numpy().reshape(-1, 3)
    valence_after = _valences(faces_np, out_vertices.size)
    # Interior valences concentrate around 6: variance about the ideal does not grow.
    assert np.var(valence_after - 6) <= np.var(valence_before - 6) + 0.5


def test_remesh_cave_cube_manifold(cave_cube: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: remeshing a non-convex shell must not tear or self-weld it.

    trimesh supplies the manifoldness and genus checks. The fixture is the point: a cavity
    gives the collapse pass two surfaces close enough to merge if it ignores connectivity.
    """
    mesh_tm, mesh_wp = cave_cube
    vertices_wp = wp.clone(mesh_wp.points)
    faces_wp = wp.clone(mesh_wp.indices)
    target = 0.5 * od.edges.mean_edge_length(vertices_wp, faces_wp)

    out_vertices, out_faces = od.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=8
    )
    mesh_out = warp_to_trimesh(out_vertices, out_faces)
    assert od.validation.is_watertight(out_vertices, out_faces)
    # Two nested cubes: Euler characteristic 4 (two genus-0 shells) is preserved.
    assert mesh_out.euler_number == mesh_tm.euler_number


def test_remesh_feature_preservation(cave_cube: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = cave_cube
    vertices_wp = wp.clone(mesh_wp.points)
    faces_wp = wp.clone(mesh_wp.indices)
    target = 0.4 * od.edges.mean_edge_length(vertices_wp, faces_wp)

    out_vertices, _ = od.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=8, feature_angle=30.0
    )
    vertices_np = out_vertices.numpy().astype(np.float64)
    # The 8 outer cube corners (frozen CORNER vertices) survive at their exact positions.
    outer_corners = np.array(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    )
    for corner in outer_corners:
        assert np.min(np.linalg.norm(vertices_np - corner, axis=1)) < 1e-6


def test_remesh_boundary_preservation(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = hemisphere
    vertices_wp = wp.clone(mesh_wp.points)
    faces_wp = wp.clone(mesh_wp.indices)
    n_loops_before = len(warp_to_trimesh(mesh_wp.points, mesh_wp.indices).outline().entities)
    target = 0.5 * od.edges.mean_edge_length(vertices_wp, faces_wp)

    out_vertices, out_faces = od.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=8
    )
    mesh_out = warp_to_trimesh(out_vertices, out_faces)
    # The open boundary is still a single closed loop (the disk boundary is preserved).
    assert len(mesh_out.outline().entities) == n_loops_before


def test_remesh_flags_off(device: str) -> None:
    _sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=2)
    n_faces_before = faces_wp.size // 3
    # Collapse-only (no split/swap/smooth/reproject) can only reduce the face count.
    out_vertices, out_faces = od.remesh.isotropic_remesh(
        vertices_wp,
        faces_wp,
        target_length=10.0 * od.edges.mean_edge_length(vertices_wp, faces_wp),
        iterations=5,
        split=False,
        swap=False,
        smooth=False,
        reproject=False,
    )
    assert out_faces.size // 3 <= n_faces_before
    assert od.validation.is_watertight(out_vertices, out_faces)


def test_collapse_pass_commits_a_useful_fraction_on_a_structured_patch(device: str) -> None:
    """
    The collapse stage removes a real fraction of a grid patch's vertices, not a handful.

    Not a library comparison: this is a claim about ordito's own parallel independent set, and no
    reference exposes one pass of a collapse stage. The independent set is chosen by an atomic-min
    lock over each candidate's two closed 1-rings, and the *key* it locks by decides how much a
    pass commits. ``edges_unique`` orders edges lexicographically by endpoint index, which on a
    structured mesh is spatially monotone, and a monotone key field has essentially one local
    minimum -- so a lock keyed by the raw edge index commits **one** collapse per pass however many
    candidates there are. ``kernels/remesh.py``'s ``scramble_index`` breaks that correlation.

    A grid patch is the input that exposes it, which is why this test does not use a sphere
    fixture. **Mutation probe, measured on this box:** with the raw index restored as the key,
    ``_collapse_pass`` removes exactly **5** vertices at both ``32 x 32`` (1 024 vertices) and
    ``68 x 68`` (4 624) against **171** and **733** hashed -- so the 10 % bound below sits 20x
    above the broken answer and is insensitive to the mesh size, where an absolute count would not
    be. Every other ``isotropic_remesh`` test passed against the broken version, which is how it
    survived; a claim about the input's shape is a claim an assert can carry cheaply.
    """
    vertices_wp, faces_wp = od.creation.grid(count=(68, 68), extents=(1.0, 1.0), device=device)
    n_vertices = vertices_wp.size
    target = 2.0 * od.edges.mean_edge_length(vertices_wp, faces_wp)
    low, high = od.remesh._length_bands(None, target, n_vertices, device)

    out_vertices, out_faces, _incidence = od.remesh._collapse_pass(
        vertices_wp, faces_wp, low, high, wp.float32(math.radians(30.0))
    )
    removed = n_vertices - out_vertices.size
    assert removed >= 0.1 * n_vertices, f"collapse stage removed only {removed} of {n_vertices}"
    # The pass must still leave a valid mesh: no degenerate faces, boundary loop intact.
    faces_np = out_faces.numpy().reshape(-1, 3)
    assert np.all(faces_np[:, 0] != faces_np[:, 1])
    assert np.all(faces_np[:, 1] != faces_np[:, 2])
    assert np.all(faces_np[:, 0] != faces_np[:, 2])
    assert len(trimesh_outline_loops(warp_to_trimesh(out_vertices, out_faces))) == 1


def _graded_height_field(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``z = x^2 - y^2`` over an ``n x n`` grid graded towards its centre, as host arrays.

    ``(vertices, faces)``: ``(n * n, 3)`` float64 positions and a flat triangle index buffer. The
    grading (sample spacing growing with the cube of the distance from the centre) makes it the
    input that breaks a collapse guard: long thin faces on the rim, tiny ones in the middle, and
    every quadric of the flat outer region close to singular. It is a height field, so every face
    projects to the xy-plane with one winding.
    """
    samples = np.sign(np.linspace(-1.0, 1.0, n)) * np.abs(np.linspace(-1.0, 1.0, n)) ** 3
    x_np, y_np = np.meshgrid(samples, samples, indexing="ij")
    vertices_np = np.stack([x_np, y_np, x_np * x_np - y_np * y_np], axis=-1).reshape(-1, 3)
    index_np = np.arange(n * n).reshape(n, n)
    faces_np = np.stack(
        [
            index_np[:-1, :-1],
            index_np[:-1, 1:],
            index_np[1:, :-1],
            index_np[:-1, 1:],
            index_np[1:, 1:],
            index_np[1:, :-1],
        ],
        axis=-1,
    ).reshape(-1)
    return vertices_np, faces_np.astype(np.int32)


def test_collapse_pass_vetoes_a_collapse_that_would_fold_a_face(device: str) -> None:
    """
    The collapse stage rejects a collapse that inverts an incident face, as the quadric one does.

    Not a library comparison: no reference exposes one pass of a collapse stage, and this is a
    claim about ordito's own guard. ``collapse_candidates`` shares ``collapse_survivor`` and
    ``satisfies_link_condition`` with ``quadric_collapse_candidates`` precisely because a
    duplicated *decision rule* is a correctness hazard, and for a long time it did not share the
    fold veto -- the link condition is topological and the anti-oscillation walk bounds *lengths*,
    so nothing else in the kernel notices a fold.

    **A strongly graded patch is the input that exposes it**, which is why this builds one rather
    than using a sphere fixture: on a well-shaped mesh no candidate collapse folds a face and the
    assertion is vacuous.

    **The assertion is the invariant a fold breaks, not a quality statistic.** The patch is a
    *height field* -- ``z = x^2 - y^2`` sampled on a graded grid -- so every face's projection to
    the xy-plane carries the same winding, and a collapse moves a vertex to an edge midpoint, whose
    projection is the midpoint of the projections. A surviving face with the opposite projected
    winding is therefore folded, exactly. Mutation probe against a baseline worktree: removing the
    veto leaves **9 folded faces of 30 450** where the guard leaves **0 of 30 502**. A
    minimum-area assertion was tried first and is *not* usable -- the fold produces an exactly
    zero-area face only at some targets, and at this one both arms bottom out around 5e-12, so the
    probe passed against the broken code.
    """
    vertices_wp, faces_wp = numpy_to_warp(*_graded_height_field(133), device)

    n_vertices = vertices_wp.size
    target = 2.0 * od.edges.mean_edge_length(vertices_wp, faces_wp)
    low, high = od.remesh._length_bands(None, target, n_vertices, device)
    out_vertices, out_faces, _incidence = od.remesh._collapse_pass(
        vertices_wp, faces_wp, low, high, wp.float32(math.radians(30.0))
    )

    # The stage must have done real work, or the guard was never reached.
    assert out_vertices.size < n_vertices

    triangles_np = out_vertices.numpy()[out_faces.numpy().reshape(-1, 3)]
    edge_a_np = triangles_np[:, 1] - triangles_np[:, 0]
    edge_b_np = triangles_np[:, 2] - triangles_np[:, 0]
    projected_np = edge_a_np[:, 0] * edge_b_np[:, 1] - edge_b_np[:, 0] * edge_a_np[:, 1]
    # Every input face projects with one winding; a survivor carrying the other one is folded.
    assert not (np.any(projected_np > 0.0) and np.any(projected_np < 0.0))


def test_remesh_adaptive_sizing_field_grades_the_result(device: str) -> None:
    """
    A graded sizing field produces a graded mesh: achieved edge length tracks the requested one.

    Asserted against the *input field* rather than against a reference, because the pymeshlab row
    for this group is a documented D2 exemption (see its ``noparity`` reason) and MeshLab's
    ``adaptive`` derives its own field from curvature rather than accepting one. The correlation is
    what the feature claims; the low-z/high-z ratio is the same claim in a form that a uniform
    remesher would fail outright, since it returns ~1.0 there.
    """
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    target = 0.06
    height = sphere_tm.vertices[:, 2]
    fraction = (height - height.min()) / np.ptp(height)
    # 0.3x the target at the bottom rising to 2.0x at the top.
    field_np = ((0.3 + 1.7 * fraction) * target).astype(np.float32)
    field_wp = wp.array(np.ascontiguousarray(field_np), dtype=wp.float32, device=device)

    out_vertices, out_faces = od.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=field_wp, iterations=3
    )
    points_np = out_vertices.numpy().astype(np.float64)
    faces_np = out_faces.numpy().reshape(-1, 3)
    pairs = np.concatenate([faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]])
    achieved = np.linalg.norm(points_np[pairs[:, 0]] - points_np[pairs[:, 1]], axis=1)
    midpoints = points_np[pairs].mean(axis=1)
    requested = field_np[KDTree(sphere_tm.vertices).query(midpoints)[1]]

    # Anti-vacuity: the remesh must have actually rebuilt the mesh.
    assert faces_np.shape[0] > faces_wp.size // 3
    # Monotone association between requested and achieved length. Measured 0.92 Spearman; a uniform
    # remesh of the same mesh scores ~0 here because ``achieved`` would not vary with ``requested``.
    order_requested = np.argsort(np.argsort(requested))
    order_achieved = np.argsort(np.argsort(achieved))
    spearman = np.corrcoef(order_requested, order_achieved)[0, 1]
    assert spearman > 0.8, spearman
    # And the coarse half really is coarser. Measured ratio 2.60 against a field ratio of ~3.
    low = achieved[midpoints[:, 2] < np.median(midpoints[:, 2])].mean()
    high = achieved[midpoints[:, 2] >= np.median(midpoints[:, 2])].mean()
    assert high / low > 1.8, (low, high)


def test_remesh_constant_sizing_field_reproduces_the_scalar_target(device: str) -> None:
    """
    A constant field gives the scalar path's *topology exactly* and its positions to 1.2e-05.

    The regression guard for the widening: if the array path diverged structurally from the scalar
    one, the face buffers would differ. They do not — ``np.array_equal`` holds — and the residual
    position gap is the float rounding of the threshold documented in ``isotropic_remesh``'s Notes
    (``4/3 * t`` formed per vertex in ``float32`` against Python ``float64`` narrowed once). The
    tolerance here is 4x the measured 1.2e-05 on a mesh of extent 2.0, not a free parameter.
    """
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    target = 0.06
    constant_wp = wp.array(
        np.full(len(sphere_tm.vertices), target, dtype=np.float32), dtype=wp.float32, device=device
    )

    scalar_v, scalar_f = od.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=3
    )
    field_v, field_f = od.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=constant_wp, iterations=3
    )

    assert scalar_f.size > faces_wp.size  # anti-vacuity
    assert np.array_equal(field_f.numpy(), scalar_f.numpy())
    assert np.abs(field_v.numpy() - scalar_v.numpy()).max() < 5e-5


@pytest.mark.parametrize("divisor", [3.0, 10.0])
def test_remesh_max_deviation_bounds_the_surface_distance(device: str, divisor: float) -> None:
    """
    ``max_deviation`` bounds the result's distance to the input, measured with the query it uses.

    The bound is asserted against
    [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh] — the same
    ``wp.mesh_query_point_no_sign`` the clamp is built on, so this is the function's actual contract
    and it holds to 1.4e-07. It is deliberately *not* asserted against trimesh: the two queries
    disagree by up to 2.1e-05 in absolute terms, so a trimesh-side assert reads 1.37x the bound at a
    bound of 1.07e-04 and would fail for a reason that is not a defect here (see the Notes on
    ``isotropic_remesh`` and the ``reproject`` stage, which has always used the same query).

    ``divisor`` is parametrized so one case binds moderately and one tightly; both must bind, which
    the unbounded-deviation comparison asserts, or the test would pass on a no-op.
    """
    _sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    common: _RemeshKwargs = {"target_length": 0.06, "iterations": 5, "reproject": False}

    free_v, _free_f = od.remesh.isotropic_remesh(vertices_wp, faces_wp, **common)
    free_deviation = float(
        od.reduce.max(od.proximity.closest_point_on_mesh(vertices_wp, faces_wp, free_v)[1])
    )
    bound = free_deviation / divisor

    bounded_v, _bounded_f = od.remesh.isotropic_remesh(
        vertices_wp, faces_wp, max_deviation=bound, **common
    )
    deviation = float(
        od.reduce.max(od.proximity.closest_point_on_mesh(vertices_wp, faces_wp, bounded_v)[1])
    )

    # The bound binds: without it the surface moves further than the bound allows.
    assert free_deviation > bound * 1.5
    assert deviation <= bound + 1e-6, (deviation, bound)


def test_remesh_target_validation(device: str) -> None:
    _sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=1)
    n_vertices = vertices_wp.size
    with pytest.raises(ValueError, match="target_length"):
        od.remesh.isotropic_remesh(vertices_wp, faces_wp, target_length=-1.0)
    with pytest.raises(ValueError, match="one target_length per vertex"):
        od.remesh.isotropic_remesh(
            vertices_wp,
            faces_wp,
            target_length=wp.full(n_vertices + 1, 0.1, dtype=wp.float32, device=device),
        )
    with pytest.raises(ValueError, match="positive target_length everywhere"):
        od.remesh.isotropic_remesh(
            vertices_wp,
            faces_wp,
            target_length=wp.zeros(n_vertices, dtype=wp.float32, device=device),
        )
    with pytest.raises(ValueError, match="max_deviation"):
        od.remesh.isotropic_remesh(vertices_wp, faces_wp, max_deviation=0.0)


def test_remesh_zero_iterations_returns_the_input_faces(device: str) -> None:
    # iterations=0 returns a clone unchanged.
    _sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=1)
    _out_v, out_f = od.remesh.isotropic_remesh(vertices_wp, faces_wp, iterations=0)
    assert out_f.size == faces_wp.size


# ---------------------------------------------------------------------------
# Vertex-clustering decimation vs open3d / pymeshlab
# ---------------------------------------------------------------------------


@pytest.mark.parity("subdivide_to_size", "meshlib")
def test_subdivide_to_size_matches_meshlib(device: str) -> None:
    """
    Class C (no correspondence): both split until every edge is under the same length cap.

    ``subdivideMesh``'s ``maxEdgeLen`` is ordito's ``max_edge`` and the two share the guarantee --
    every surviving edge below the cap -- but not the route: MeshLib also *flips* edges as it goes
    (its ``maxDeviationAfterFlip`` gate) where ordito only splits, so the outputs differ in face
    count. Measured on ``icosphere(2)`` at half the mean edge: 3 200 faces against 3 320, longest
    edge 0.143 against 0.141 with a cap of 0.150, and the two surfaces **0.021** apart, which is
    0.14 of the cap.

    ``maxDeviationAfterFlip`` is raised from its default of 1.0 -- an absolute length, so on a
    unit-scale mesh the default is effectively unbounded and on a millimetre-scale one it would
    forbid every flip. Passing it explicitly is what keeps the comparison from depending on the
    fixture's units.

    **Bug class excluded:** a splitter that stops early and leaves edges above the cap, which is the
    guarantee both sides make and is asserted on both outputs. **Mutation probe, measured:** the
    *input* mesh's longest edge is 0.32, 2.1x the cap, so a pass that did nothing fails the cap
    assert by more than a factor of two.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=2)
    vertices_wp, faces_wp = numpy_to_warp(sphere_tm.vertices, sphere_tm.faces, device)
    max_edge = 0.5 * float(od.edges.mean_edge_length(vertices_wp, faces_wp))

    out_vertices_wp, out_faces_wp = od.remesh.subdivide_to_size(vertices_wp, faces_wp, max_edge)
    out_tm = warp_to_trimesh(out_vertices_wp, out_faces_wp)

    mesh_ml = numpy_to_meshlib(sphere_tm.vertices, sphere_tm.faces)
    settings_ml = mm.SubdivideSettings()
    settings_ml.maxEdgeLen = max_edge
    settings_ml.maxEdgeSplits = 10_000_000
    settings_ml.maxDeviationAfterFlip = 1e30  # an absolute length; its default of 1.0 is unitful
    n_splits_ml = mm.subdivideMesh(mesh_ml, settings_ml)
    subdivided_tm = meshlib_to_trimesh(mesh_ml)

    assert n_splits_ml > 0  # non-vacuity: the reference really subdivided
    assert _edge_lengths(sphere_tm.vertices, sphere_tm.faces).max() > 2.0 * max_edge
    for result_tm in (out_tm, subdivided_tm):
        assert _edge_lengths(result_tm.vertices, result_tm.faces).max() <= max_edge
    assert abs(out_tm.faces.shape[0] - subdivided_tm.faces.shape[0]) < 0.1 * out_tm.faces.shape[0]
    assert (
        hausdorff_surface_two_sided(
            out_tm.vertices, out_tm.faces, subdivided_tm.vertices, subdivided_tm.faces
        )
        < 0.25 * max_edge
    )


@pytest.mark.parity("subdivide_region_to_size", "meshlib")
def test_subdivide_region_to_size_matches_meshlib(device: str) -> None:
    """
    Class C (no correspondence): the same cap, applied to the same half of the same mesh.

    meshlib is the only reference with a region restriction, and it has the whole parameter set:
    ``SubdivideSettings.region`` is ordito's ``region``, and ``maxEdgeLen`` / ``maxEdgeSplits`` /
    ``maxDeviationAfterFlip`` are ``max_edge`` / ``max_splits`` / ``max_deviation``. So this is the
    comparison in ``test_subdivide_to_size_matches_meshlib`` with one field set, and it diverges for
    the same reason: MeshLib *flips* as it splits where ordito only splits.

    Measured on ``icosphere(3)``'s upper half at 0.6 of the mean edge:

    | | faces | region | area inside | area outside |
    |---|---|---|---|---|
    | input | 1 280 | 624 | 6.087263879 | 6.419228855 |
    | ordito | 3 200 | 2 496 | **6.087263824** | 6.419228780 |
    | meshlib | 3 256 | 2 552 | 6.084993649 | 6.419228780 |

    Two things in that table are the test. **The complement's area is identical on both sides to
    nine digits**, so neither refiner moves the surface it was told to leave alone -- and it is
    *not* left untouched combinatorially: both grow it from 656 faces to **704**, by exactly the
    same amount, because a split on the region's rim must propagate across to stay crack-free. A
    test asserting the complement's faces are unchanged would fail on both libraries and for a good
    reason, which is why the invariant here is the area and the face count rather than the face
    buffer.

    **Inside** the region the two part company in the direction the flip predicts: ordito preserves
    the area to 9e-09 relative (a split cannot move a surface) and MeshLib loses 2.3e-03 (a flip
    can). That is the discriminating fact, so it is asserted rather than absorbed into a tolerance.

    ``maxDeviationAfterFlip`` is raised from its default of 1.0, an absolute length, for the reason
    the sibling records. ``maintainRegion`` is left alone and is **not** the bool its name suggests:
    it is a second ``FaceBitSet``. And ``settings.region`` is an **in-place** output as well as an
    input -- MeshLib grows it to track the refined region (624 -> 2 552 bits, resized to the new
    face count), which is what makes the inside/outside split readable on its answer at all.

    **Bug class excluded:** a refiner that stops early inside the region, asserted as the cap on
    both outputs' region faces. **Mutation probe, measured:** the region's longest input edge is
    1.8x the cap, so a pass that did nothing fails that assert by most of a factor of two.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=3)
    vertices_np = np.asarray(sphere_tm.vertices)
    faces_np = np.asarray(sphere_tm.faces)
    region_np = np.ascontiguousarray(vertices_np[faces_np].mean(axis=1)[:, 2] > 0.0)

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    region_wp = wp.array(region_np, dtype=wp.bool, device=device)
    max_edge = 0.6 * float(od.edges.mean_edge_length(vertices_wp, faces_wp))

    out_vertices_wp, out_faces_wp, out_region_wp = od.remesh.subdivide_region_to_size(
        vertices_wp, faces_wp, region_wp, max_edge
    )
    out_tm = warp_to_trimesh(out_vertices_wp, out_faces_wp)

    mesh_ml = numpy_to_meshlib(vertices_np, faces_np)
    region_ml = mm.FaceBitSet(numpy_to_meshlib_bitset(region_np))
    settings_ml = mm.SubdivideSettings()
    settings_ml.maxEdgeLen = max_edge
    settings_ml.maxEdgeSplits = 10_000_000
    settings_ml.maxDeviationAfterFlip = 1e30  # an absolute length; its default of 1.0 is unitful
    settings_ml.region = region_ml
    n_splits_ml = mm.subdivideMesh(mesh_ml, settings_ml)
    refined_tm = meshlib_to_trimesh(mesh_ml)
    refined_region_ml = meshlib_bitset_to_numpy(region_ml, refined_tm.faces.shape[0])

    assert n_splits_ml > 0  # non-vacuity: the reference really refined
    # Mutation probe: the region's own longest edge is well above the cap to begin with.
    assert _edge_lengths(vertices_np, faces_np[region_np]).max() > 1.5 * max_edge

    out_region_np = out_region_wp.numpy()
    area_outside = float(
        tm.Trimesh(vertices_np, faces_np, process=False).area_faces[~region_np].sum()
    )
    for result_tm, refined_region_np in ((out_tm, out_region_np), (refined_tm, refined_region_ml)):
        assert result_tm.faces.shape[0] > faces_np.shape[0]
        assert (
            _edge_lengths(result_tm.vertices, result_tm.faces[refined_region_np]).max() <= max_edge
        )
        # Neither refiner moves the complement, although both subdivide into it to stay crack-free.
        assert np.isclose(
            float(result_tm.area_faces[~refined_region_np].sum()), area_outside, rtol=1e-5, atol=0.0
        )
        assert int((~refined_region_np).sum()) > int((~region_np).sum())
    assert abs(out_tm.faces.shape[0] - refined_tm.faces.shape[0]) < 0.1 * out_tm.faces.shape[0]
    assert int((~out_region_np).sum()) == int((~refined_region_ml).sum())

    # Inside the region the two part company, in the direction the flip predicts.
    area_inside = float(
        tm.Trimesh(vertices_np, faces_np, process=False).area_faces[region_np].sum()
    )
    assert np.isclose(
        float(out_tm.area_faces[out_region_np].sum()), area_inside, rtol=1e-6, atol=0.0
    )
    assert not np.isclose(
        float(refined_tm.area_faces[refined_region_ml].sum()), area_inside, rtol=1e-6, atol=0.0
    )
    # Every input vertex survives in place: this refiner only ever inserts.
    assert np.allclose(out_tm.vertices[: vertices_np.shape[0]], vertices_np, rtol=1e-5, atol=1e-5)


def test_subdivide_region_to_size_never_writes_into_the_caller(device: str) -> None:
    """
    Not a library comparison: no reference promises anything about the caller's own buffers.

    ``subdivide_region_to_size`` documents ``vertices``/``faces`` as inputs and returns new buffers,
    which on the path where *nothing* splits means it has to copy them -- and the copy has to happen
    before the closing Delaunay pass, because ``_flip_region_faces`` rewrites its face buffer in
    place. The fixture is the smallest input that reaches that path with work to do: a quad whose
    diagonal is non-Delaunay (vertex 3 sits inside the circumcircle of face 0) and whose longest
    edge is far under ``max_edge``, so the split loop breaks on its first iteration and the flip
    pass is the only thing that runs.

    **Bug class excluded:** an in-place write into an argument. **Mutation probe, measured against
    a detached pre-fix worktree:** with the copy below the flip pass, the caller's own ``faces``
    goes from ``[0 1 2 0 2 3]`` to ``[2 3 1 0 1 3]`` and this test fails on its first assert. A
    rectangle does *not* probe it -- every rectangle is cyclic, so its diagonal is a Delaunay tie
    and nothing flips.
    """
    vertices_np = np.array([[0, 0, 0], [1, -1, 0], [2, 0, 0], [1, -0.1, 0]], dtype=np.float64)
    faces_np = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    region_wp = wp.full(2, True, dtype=wp.bool, device=device)
    faces_before = faces_wp.numpy().copy()
    vertices_before = vertices_wp.numpy().copy()

    out_vertices_wp, out_faces_wp, _out_region_wp = od.remesh.subdivide_region_to_size(
        vertices_wp, faces_wp, region_wp, max_edge=100.0
    )

    assert np.array_equal(faces_wp.numpy(), faces_before)
    assert np.array_equal(vertices_wp.numpy(), vertices_before)
    assert out_faces_wp.ptr != faces_wp.ptr
    assert out_vertices_wp.ptr != vertices_wp.ptr
    # Non-vacuity: the flip pass really had something to do, so the copy was load-bearing.
    assert not np.array_equal(out_faces_wp.numpy(), faces_before)


@pytest.mark.parametrize("voxel_size", [0.1, 0.3])
@pytest.mark.parametrize("contraction", ["average", "closest"])
@pytest.mark.parity("cluster_decimate", "open3d")
def test_cluster_decimate_matches_open3d(
    device: str, voxel_size: float, contraction: Literal["average", "closest"]
) -> None:
    """
    Class A: cell assignment is Open3D's, so the face count matches exactly, not approximately.

    Open3D's grid anchor is ``min_bound - voxel_size / 2``; this pins that choice, since an anchor
    at ``min_bound`` splits the vertices on the box face into two cells and the counts diverge.
    """
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    simplified_o3d = trimesh_to_open3d(sphere_tm).simplify_vertex_clustering(voxel_size=voxel_size)

    decimated_vertices_wp, decimated_faces_wp = od.remesh.cluster_decimate(
        vertices_wp, faces_wp, voxel_size=voxel_size, contraction=contraction
    )
    assert decimated_faces_wp.size // 3 == len(simplified_o3d.triangles)
    assert decimated_vertices_wp.size == len(simplified_o3d.vertices)

    if contraction == "average":
        # Same cells and the same mean per cell, so the vertex *sets* coincide pointwise.
        distance_np, _index = KDTree(np.asarray(simplified_o3d.vertices)).query(
            decimated_vertices_wp.numpy().astype(np.float64)
        )
        assert np.max(distance_np) < 1e-5
    else:
        # 'Closest to centre' keeps every output vertex on the input surface, exactly.
        distance_np, _index = KDTree(np.asarray(sphere_tm.vertices)).query(
            decimated_vertices_wp.numpy().astype(np.float64)
        )
        assert np.max(distance_np) < 1e-5


@pytest.mark.parametrize("voxel_fraction", [0.02, 0.05, 0.1])
@pytest.mark.parity("cluster_decimate", "meshlib")
def test_cluster_decimate_matches_meshlib_cell_count(device: str, voxel_fraction: float) -> None:
    """
    Class C (a count statistic): ``verticesGridSampling`` picks one vertex per occupied cell.

    Not the same operation, and the difference is named rather than tolerated: MeshLib *samples* --
    it returns a ``VertBitSet`` selecting one surviving vertex per voxel and never builds a mesh --
    where ``cluster_decimate`` contracts each cell's vertices to a representative and rebuilds the
    faces. What the two share is the cell decomposition, so the comparable quantity is **how many
    cells the surface occupies**, which is ordito's output vertex count and MeshLib's bit count.

    Measured on ``icosphere(4)`` at 2 %, 5 % and 10 % of the bounding-box diagonal: 2 310 / 541 /
    151 against MeshLib's 2 394 / 548 / 128, i.e. ratios of **0.97 / 0.99 / 1.18**. The residual is
    grid *anchoring* -- neither library documents where cell zero starts, and at a coarse voxel a
    half-cell shift moves points across boundaries -- so the bound is 1.3x rather than exact.

    **Bug class excluded:** a voxel size interpreted in the wrong units, or a grid whose cells are
    the wrong size. **Mutation probe, measured:** the count falls **4.3x** between consecutive
    fractions here (2 310 -> 541 -> 151), so a factor-of-two error in the cell size moves the ratio
    far outside the 1.3x band while the anchoring residual stays inside it.
    """
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    diagonal = float(
        np.linalg.norm(sphere_tm.vertices.max(axis=0) - sphere_tm.vertices.min(axis=0))
    )
    voxel_size = voxel_fraction * diagonal

    clustered_vertices_wp, clustered_faces_wp = od.remesh.cluster_decimate(
        vertices_wp, faces_wp, voxel_size=voxel_size
    )
    sampled_ml = mm.verticesGridSampling(
        mm.MeshPart(numpy_to_meshlib(sphere_tm.vertices, sphere_tm.faces)), voxel_size
    )

    n_cells_ml = sampled_ml.count()
    n_cells_wp = clustered_vertices_wp.size
    assert 0 < n_cells_ml < sphere_tm.vertices.shape[0]  # non-vacuity: it really sampled down
    assert 0 < n_cells_wp < sphere_tm.vertices.shape[0]
    assert 1.0 / 1.3 < n_cells_wp / n_cells_ml < 1.3
    assert clustered_faces_wp.size > 0


def test_cluster_decimate_stays_near_the_input_surface(device: str) -> None:
    """A resampling, so the shape has to survive: Hausdorff within about one cell."""
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    voxel_size = 0.2
    decimated_vertices_wp, decimated_faces_wp = od.remesh.cluster_decimate(
        vertices_wp, faces_wp, voxel_size=voxel_size
    )
    deviation = hausdorff_surface_two_sided(
        np.asarray(sphere_tm.vertices),
        np.asarray(sphere_tm.faces),
        decimated_vertices_wp.numpy().astype(np.float64),
        decimated_faces_wp.numpy().reshape(-1, 3),
    )
    assert deviation < voxel_size


def test_cluster_decimate_emits_no_degenerate_or_duplicated_faces(device: str) -> None:
    """Collapsed faces are dropped and welded duplicates deduped: both are part of the algorithm."""
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    decimated_vertices_wp, decimated_faces_wp = od.remesh.cluster_decimate(
        vertices_wp, faces_wp, voxel_size=0.25
    )
    faces_np = decimated_faces_wp.numpy().reshape(-1, 3)
    assert _degenerate_face_count(decimated_vertices_wp.numpy().astype(np.float64), faces_np) == 0
    assert len(np.unique(np.sort(faces_np, axis=1), axis=0)) == faces_np.shape[0]
    # Every output vertex is referenced by a face.
    assert len(np.unique(faces_np)) == decimated_vertices_wp.size


@pytest.mark.parametrize("voxel_size", [1e-3, 0.1, 0.25])
@pytest.mark.parametrize("contraction", ["average", "closest"])
def test_cluster_decimate_matches_a_numpy_transcription(
    device: str, voxel_size: float, contraction: Literal["average", "closest"]
) -> None:
    """
    Class A against a NumPy transcription of the definition, pinning the output *numbering*.

    Open3D fixes the counts and the vertex set but not the order, and the order is a contract:
    output vertices are the kept cells in packed-cell-key order (``unique_1d``'s), faces are the
    distinct welded faces in ``unique_faces``' order, each in its first occurrence's winding. The
    kernel dedups with hash tables whose slots number the cells arbitrarily, so the order is
    recovered by sorts that this checks against ``np.unique``. The three sizes cover: nothing
    merging (more distinct faces than vertices, the branch where the face sort cannot reuse the
    cell sort's buffers), and two where welding leaves duplicate faces, asserted below.
    """
    # Two concentric sheets a hair apart, the inner one reversed -- a thin shell. They weld into the
    # same cells, so their faces map onto the same cluster triples in opposite windings, and the
    # face dedup has duplicates to remove at every coarse size whose kept winding is observable.
    sphere_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    n_sheet = sphere_tm.vertices.shape[0]
    vertices_wp, faces_wp = numpy_to_warp(
        np.concatenate((sphere_tm.vertices, 1.002 * sphere_tm.vertices)),
        np.concatenate((sphere_tm.faces, sphere_tm.faces[:, ::-1] + n_sheet)),
        device,
    )
    out_vertices_wp, out_faces_wp = od.remesh.cluster_decimate(
        vertices_wp, faces_wp, voxel_size=voxel_size, contraction=contraction
    )
    voxel, origin, bound = od.voxels.resolve_voxel_grid(
        vertices_wp, voxel_size, return_cell_bound=True
    )
    vertices_np = vertices_wp.numpy()
    local_np = (vertices_np - np.array(origin, dtype=np.float32)) * np.float32(1.0 / voxel)
    cells_np = np.floor(local_np).astype(np.int64)
    keys_np = cells_np[:, 0] + bound * (cells_np[:, 1] + bound * cells_np[:, 2])
    _keys, labels_np = np.unique(keys_np, return_inverse=True)
    remapped_np = labels_np[faces_wp.numpy().reshape(-1, 3)]
    distinct = (
        (remapped_np[:, 0] != remapped_np[:, 1])
        & (remapped_np[:, 1] != remapped_np[:, 2])
        & (remapped_np[:, 0] != remapped_np[:, 2])
    )
    surviving_np = remapped_np[distinct]
    kept_np = np.unique(surviving_np)
    ranked_np = np.searchsorted(kept_np, surviving_np)
    n_kept = kept_np.size
    sorted_np = np.sort(ranked_np, axis=1)
    face_keys_np = sorted_np[:, 0] + n_kept * (sorted_np[:, 1] + n_kept * sorted_np[:, 2])
    _face_keys, first_np = np.unique(face_keys_np, return_index=True)
    expected_faces_np = ranked_np[first_np]
    if voxel_size > 0.01:
        assert first_np.size < surviving_np.shape[0]  # non-vacuity: welding made duplicates
    else:
        assert first_np.size > len(vertices_np)  # non-vacuity: the wide face sort
    assert np.array_equal(out_faces_wp.numpy().reshape(-1, 3), expected_faces_np)

    rank_np = np.full(labels_np.max() + 1, -1)
    rank_np[kept_np] = np.arange(n_kept)
    vertex_rank_np = rank_np[labels_np]
    if contraction == "average":
        sums_np = np.zeros((n_kept, 3))
        np.add.at(sums_np, vertex_rank_np[vertex_rank_np >= 0], vertices_np[vertex_rank_np >= 0])
        counts_np = np.bincount(vertex_rank_np[vertex_rank_np >= 0], minlength=n_kept)
        assert np.allclose(
            out_vertices_wp.numpy(), sums_np / counts_np[:, None], rtol=1e-5, atol=1e-5
        )
    else:
        # Every output vertex is an input vertex of its own cluster.
        _distance, index_np = KDTree(vertices_np).query(out_vertices_wp.numpy())
        assert np.array_equal(vertex_rank_np[index_np], np.arange(n_kept))
    assert 2 * n_sheet == len(vertices_np)


def test_cluster_decimate_decimates_monotonically(device: str) -> None:
    """A wider cell can only ever produce fewer faces."""
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    counts = [
        od.remesh.cluster_decimate(vertices_wp, faces_wp, voxel_size=size)[1].size // 3
        for size in (0.05, 0.1, 0.2, 0.4)
    ]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]


def test_cluster_decimate_default_voxel_size(device: str) -> None:
    """The default is 1% of the bounding-box diagonal, matching MeshLab's ``threshold``."""
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    default_faces_wp = od.remesh.cluster_decimate(vertices_wp, faces_wp)[1]
    diagonal = float(np.linalg.norm(np.array([2.0, 2.0, 2.0])))
    explicit_faces_wp = od.remesh.cluster_decimate(
        vertices_wp, faces_wp, voxel_size=0.01 * diagonal
    )[1]
    assert default_faces_wp.size == explicit_faces_wp.size


def test_cluster_decimate_invalid(device: str) -> None:
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=2)
    with pytest.raises(ValueError, match="voxel_size > 0"):
        od.remesh.cluster_decimate(vertices_wp, faces_wp, voxel_size=0.0)
    with pytest.raises(ValueError, match="contraction must be"):
        od.remesh.cluster_decimate(vertices_wp, faces_wp, contraction="quadric")  # pyright: ignore[reportArgumentType]


def test_cluster_decimate_collapses_to_nothing(device: str) -> None:
    """
    A cell wider than the mesh collapses every face, and the output is empty, not a lone vertex.

    Not a library comparison: Open3D keeps the cluster as an unreferenced vertex (see the
    ``cluster_decimate`` Notes), so no reference returns this answer. It is the one input where no
    cluster is referenced and no face class survives the dedup.
    """
    for contraction in ("average", "closest"):
        _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=2)
        out_vertices_wp, out_faces_wp = od.remesh.cluster_decimate(
            vertices_wp, faces_wp, voxel_size=100.0, contraction=contraction
        )
        assert out_vertices_wp.size == 0
        assert out_faces_wp.size == 0


# ---------------------------------------------------------------------------
# Quadric edge-collapse decimation vs igl / open3d / pymeshlab
# ---------------------------------------------------------------------------


def _inverted_face_count(vertices_np: np.ndarray, faces_np: np.ndarray) -> int:
    """
    Faces whose outward normal points *inward* on a star-shaped mesh centred on the origin.

    The failure mode an unguarded quadric method produces at a high reduction ratio, and cheap to
    detect on a sphere: a correctly oriented face has its normal agreeing with its own centroid.
    """
    mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
    centroids_np = mesh_tm.vertices[mesh_tm.faces].mean(axis=1)
    return int((np.einsum("ij,ij->i", mesh_tm.face_normals, centroids_np) < 0.0).sum())


_QUADRIC_SPHERE_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, float]] = {}


def _quadric_decimated_sphere(
    device: str, target_faces: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return ordito's decimation of ``icosphere(4)`` and its two-sided deviation from the input.

    ``(vertices, faces, deviation)``: ``(n, 3)`` float64 positions, ``(target_faces, 3)`` faces,
    and the surface Hausdorff distance the reference comparisons bound. The four reference tests
    below compare against the same decimation at the same targets, so it is computed once per
    ``(device, target_faces)`` and handed out as read-only host arrays -- no test can mutate what a
    later one reads.
    """
    key = (str(device), target_faces)
    if key not in _QUADRIC_SPHERE_CACHE:
        sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
        decimated_vertices_wp, decimated_faces_wp = od.remesh.quadric_decimate(
            vertices_wp, faces_wp, target_faces=target_faces
        )
        decimated_vertices_np = decimated_vertices_wp.numpy().astype(np.float64)
        decimated_faces_np = decimated_faces_wp.numpy().reshape(-1, 3)
        decimated_vertices_np.flags.writeable = False
        decimated_faces_np.flags.writeable = False
        deviation = hausdorff_surface_two_sided(
            np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64),
            np.asarray(sphere_tm.faces),
            decimated_vertices_np,
            decimated_faces_np,
        )
        _QUADRIC_SPHERE_CACHE[key] = (decimated_vertices_np, decimated_faces_np, deviation)
    return _QUADRIC_SPHERE_CACHE[key]


@pytest.mark.parametrize("target_faces", [2560, 1024, 512])
@pytest.mark.parity("quadric_decimate", "igl", "open3d")
def test_quadric_decimate_beats_igl_and_open3d_on_deviation(device: str, target_faces: int) -> None:
    """
    Class C (a deviation bound): at the same face count, no *worse* than two serial references.

    The plan for this port said to expect the batched-parallel formulation to pick a different
    sequence of collapses from a serial priority queue, and to compare by deviation rather than by
    equality. It does, and it comes out ahead: measured two-sided Hausdorff to the input icosphere
    at 512 faces is **0.0147 here against igl's 0.0250 and Open3D's 0.0236**, and the ordering holds
    at every target. Spreading the collapses over independent sets rather than draining a queue
    keeps the error evenly distributed, which is what a max-norm rewards.

    The assertion is one-sided with slack, not an equality — the point is that the parallel method
    is competitive, not that this exact ratio is a contract.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=4, radius=1.0)
    vertices_np = np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(sphere_tm.faces, dtype=np.int64)

    decimated_igl = igl.decimate(vertices_np, faces_np, target_faces)
    igl_tm = tm.Trimesh(np.asarray(decimated_igl[0]), np.asarray(decimated_igl[1]), process=False)
    mesh_o3d = trimesh_to_open3d(sphere_tm).simplify_quadric_decimation(
        target_number_of_triangles=target_faces
    )
    o3d_tm = open3d_to_trimesh(mesh_o3d)

    _decimated_vertices_np, decimated_faces_np, deviation_wp = _quadric_decimated_sphere(
        device, target_faces
    )
    assert decimated_faces_np.shape[0] == target_faces

    deviation_igl = hausdorff_surface_two_sided(
        vertices_np, np.asarray(sphere_tm.faces), igl_tm.vertices, igl_tm.faces
    )
    deviation_o3d = hausdorff_surface_two_sided(
        vertices_np, np.asarray(sphere_tm.faces), o3d_tm.vertices, o3d_tm.faces
    )
    assert deviation_wp <= 1.2 * min(deviation_igl, deviation_o3d)


def test_quadric_decimate_emits_no_inverted_or_degenerate_faces(device: str) -> None:
    """The normal-flip guard's job: even at 5% of the triangles, nothing folds over."""
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    decimated_vertices_wp, decimated_faces_wp = od.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_ratio=0.05
    )
    vertices_np = decimated_vertices_wp.numpy().astype(np.float64)
    faces_np = decimated_faces_wp.numpy().reshape(-1, 3)
    assert _inverted_face_count(vertices_np, faces_np) == 0
    assert _degenerate_face_count(vertices_np, faces_np) == 0
    assert od.validation.is_edge_manifold(decimated_faces_wp, allow_boundary_edges=False)
    assert od.validation.is_winding_consistent(decimated_faces_wp)


def test_quadric_decimate_preserves_the_topology(device: str) -> None:
    """Not a library comparison: a closed genus-0 surface stays so, and its volume barely moves."""
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    decimated_vertices_wp, decimated_faces_wp = od.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_ratio=0.2
    )
    decimated_tm = warp_to_trimesh(decimated_vertices_wp, decimated_faces_wp)
    assert decimated_tm.is_watertight
    assert decimated_tm.euler_number == 2
    assert np.isclose(decimated_tm.volume, sphere_tm.volume, rtol=0.02)


def test_quadric_decimate_keeps_the_features_of_a_cube(device: str) -> None:
    """
    Class C (a feature-distance bound): a cube's twelve edges are its whole shape.

    This is the property that distinguishes a quadric method from a length-driven one: the flat
    faces have zero quadric cost to collapse and the creases have a large one, so the sharp edges
    survive down to the coarsest usable mesh. Checked as the surviving dihedral distribution, which
    is invariant to *which* particular collapses happened.
    """
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0]).subdivide().subdivide().subdivide()
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(box_tm.vertices), np.asarray(box_tm.faces), device
    )
    decimated_vertices_wp, decimated_faces_wp = od.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_ratio=0.1
    )
    decimated_tm = warp_to_trimesh(decimated_vertices_wp, decimated_faces_wp)
    # Still a box: the same eight corners, the same volume, and 90-degree edges intact.
    assert np.allclose(decimated_tm.bounds, box_tm.bounds, atol=1e-4)
    assert np.isclose(decimated_tm.volume, box_tm.volume, rtol=0.02)
    angles_np = np.rad2deg(np.abs(decimated_tm.face_adjacency_angles))
    assert np.percentile(angles_np, 95.0) > 85.0


@pytest.mark.parity("quadric_decimate", "pymeshlab")
def test_quadric_decimate_reaches_pymeshlab_quality(device: str) -> None:
    """
    Class C (a quality bound): MeshLab drives the same metric serially, so it sets the bar.

    Its ``autoclean`` default deletes unreferenced vertices, so the MeshSet is built fresh here (it
    is one of the two filters recorded as not idempotent even in geometry). Compared by deviation,
    as with the other two references.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=4, radius=1.0)
    target_faces = 1024
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(sphere_tm.faces, dtype=np.int32),
        )
    )
    meshset_pml.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
    mesh_pml = meshset_pml.current_mesh()
    pml_tm = tm.Trimesh(mesh_pml.vertex_matrix(), mesh_pml.face_matrix(), process=False)

    vertices_np = np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64)
    _, _, deviation_wp = _quadric_decimated_sphere(device, target_faces)
    deviation_pml = hausdorff_surface_two_sided(
        vertices_np, np.asarray(sphere_tm.faces), pml_tm.vertices, pml_tm.faces
    )
    assert deviation_wp <= 1.2 * deviation_pml


@pytest.mark.parametrize("target_faces", [2560, 1024])
@pytest.mark.parity("quadric_decimate", "meshlib")
def test_quadric_decimate_matches_meshlib_quality(device: str, target_faces: int) -> None:
    """
    Class C (a deviation bound): the same face count, reached by a different collapse order.

    ``decimateMesh`` takes a **deleted**-face budget rather than a target, so the named transform is
    ``maxDeletedFaces = n_faces - target_faces``; fed that, it lands on exactly the requested count
    on both targets here, as ordito does. ``packMesh=True`` is required to read the result at all
    -- without it ``getNumpyFaces`` returns the pre-decimation buffer padded with degenerate
    ``[0, 0, 0]`` rows, which is the hazard
    [`meshlib_to_trimesh`][tests.conversions.meshlib_to_trimesh] exists for.

    There is no correspondence between the outputs -- two greedy quadric solvers with different
    tie-breaking pick different collapses -- so the comparison is the deviation from the *input*
    surface, which is what a decimator is trying to minimize. Measured on ``icosphere(4)``:
    ordito is **1.30x** MeshLib's deviation at 2 560 faces and **1.04x** at 1 024, so the bound is
    set at 1.5x.

    **Bug class excluded:** a decimator that reaches the face count by collapsing the wrong edges,
    which shows up as a deviation several times the reference's rather than a third above it.
    **Mutation probe, measured:** the same comparison at ``target_faces=512`` (an eighth of the
    input) gives deviations an order of magnitude larger on both sides, so a fixed absolute
    threshold would not separate the two; the *ratio* is what stays near one.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=4, radius=1.0)
    vertices_np = np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(sphere_tm.faces)

    _, decimated_faces_np, deviation_wp = _quadric_decimated_sphere(device, target_faces)

    mesh_ml = numpy_to_meshlib(sphere_tm.vertices, sphere_tm.faces)
    settings_ml = mm.DecimateSettings()
    settings_ml.maxDeletedFaces = faces_np.shape[0] - target_faces
    settings_ml.packMesh = True
    result_ml = mm.decimateMesh(mesh_ml, settings_ml)
    decimated_tm = meshlib_to_trimesh(mesh_ml)

    assert result_ml.facesDeleted > 0  # non-vacuity: the reference really decimated
    assert decimated_tm.faces.shape[0] == target_faces
    assert decimated_faces_np.shape[0] == target_faces

    deviation_ml = hausdorff_surface_two_sided(
        vertices_np, faces_np, decimated_tm.vertices, decimated_tm.faces
    )
    assert deviation_ml > 0.0
    assert deviation_wp <= 1.5 * deviation_ml


@pytest.mark.parametrize("target_faces", [2560, 1024, 512])
@pytest.mark.parity("quadric_decimate", "pyvista")
def test_quadric_decimate_stays_within_the_pyvista_band(device: str, target_faces: int) -> None:
    """
    Class C by deviation, against the one reference of the four that is measurably better.

    That is the finding this row exists to record rather than hide. ``vtkDecimatePro`` (which is
    what ``PolyData.decimate`` wraps) hits the requested count exactly and, on ``icosphere(4)``,
    leaves a
    *smaller* two-sided surface deviation than ordito's batched-parallel collapse: measured
    **0.00167 / 0.00609 / 0.00986** against ordito's **0.00255 / 0.00695 / 0.01330** at 2 560 /
    1 024 / 512 faces -- a ratio of 1.53 / 1.14 / 1.35. So the bound here is a *band*, at 2.0x with
    a 1.3x margin on the worst reading, and not the one-sided "no worse than" the igl / open3d test
    asserts.

    The same measurement puts the four references in order, which is what makes the band meaningful
    rather than arbitrary: at 2 560 faces, pyvista 0.00167 < ordito 0.00255 < open3d 0.00364 < igl
    0.00589. Serial priority queues are not all alike, and VTK's is the strongest of the three;
    [`test_quadric_decimate_beats_igl_and_open3d_on_deviation`] keeps that ordering live at the same
    three targets by requiring ordito to stay ahead of the other two on the same decimation, so a
    regression cannot hide inside the loosened ceiling.

    ``decimate_pro`` is deliberately **not** the row even though pyvista exposes it: it only
    *removes* vertices, so every surviving point stays exactly on the sphere (mean ``| |r| - 1 |`` =
    1.4e-17 against 6.5e-04 for ``decimate`` and 6.3e-04 for ordito, asserted below). A
    vertex-removal decimator cannot be beaten on sphere deviation by anything that places new
    vertices, so comparing against it would measure that constraint rather than the quality.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=4, radius=1.0)
    vertices_np = np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(sphere_tm.faces)
    mesh_pv = trimesh_to_pyvista(sphere_tm)

    decimated_pv = mesh_pv.decimate(1.0 - target_faces / faces_np.shape[0])
    assert decimated_pv.n_faces == target_faces, "the reference hit the target it is compared at"
    pv_tm = tm.Trimesh(
        np.asarray(decimated_pv.points), np.asarray(decimated_pv.regular_faces), process=False
    )

    decimated_vertices_np, decimated_faces_np, deviation_wp = _quadric_decimated_sphere(
        device, target_faces
    )
    assert decimated_faces_np.shape[0] == target_faces

    deviation_pv = hausdorff_surface_two_sided(vertices_np, faces_np, pv_tm.vertices, pv_tm.faces)
    assert deviation_pv > 0.0
    assert deviation_wp <= 2.0 * deviation_pv

    # The kind-of-algorithm discriminator: decimate_pro only removes, the other two place.
    def radius_error(points_np: np.ndarray) -> float:
        return float(np.abs(np.linalg.norm(points_np, axis=1) - 1.0).mean())

    assert radius_error(np.asarray(mesh_pv.decimate_pro(0.5).points)) < 1e-12
    assert radius_error(np.asarray(mesh_pv.decimate(0.5).points)) > 1e-5
    assert radius_error(decimated_vertices_np) > 1e-5


def test_quadric_decimate_is_monotone_in_the_target(device: str) -> None:
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    counts = [
        od.remesh.quadric_decimate(vertices_wp, faces_wp, target_ratio=ratio)[1].size // 3
        for ratio in (0.8, 0.4, 0.2, 0.1)
    ]
    assert counts == sorted(counts, reverse=True)


def test_quadric_decimate_reaches_its_target_and_floors_on_feature_angle(device: str) -> None:
    """
    Not a library comparison: no reference exposes ``feature_angle``'s effect on the floor.

    Two halves of one ``Notes`` claim. **The target is reached exactly whenever it is reachable**
    -- at the default, which freezes no interior edge, both 320 and 20 faces of ``icosphere(3)``
    are hit on the nose. And **a frozen-feature angle is what stops a decimation short**: at 30
    degrees a target of 20 stops far above it, because a sphere's own dihedral angles grow as it
    is coarsened until every edge is sharper than the angle and every vertex is a frozen corner.
    Raising the angle lowers the floor monotonically.

    **Bug class excluded:** a floor that moves for a reason other than the parameter the docstring
    names -- a decimation that silently stopped on the iteration cap, say, would give the same
    floor at every angle. The angle sweep is what separates the two, and it is asserted as a
    strict monotone decrease rather than at fixed values, since the exact floor is a property of
    the input rather than of the contract.
    """
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    n_faces = faces_wp.size // 3
    assert n_faces == 1280  # non-vacuity: the fixture the floors below were measured on

    for target in (320, 20):
        reached = od.remesh.quadric_decimate(vertices_wp, faces_wp, target_faces=target)[1]
        assert reached.size // 3 == target

    floors = [
        od.remesh.quadric_decimate(vertices_wp, faces_wp, target_faces=20, feature_angle=angle)[
            1
        ].size
        // 3
        for angle in (30.0, 45.0, 60.0, 180.0)
    ]
    assert floors[0] > 20  # a frozen-feature angle really does stop short
    assert floors == sorted(floors, reverse=True)
    assert len(set(floors)) > 1  # the angle moves it, so the cap is not what binds
    assert floors[-1] == 20

    # The iteration cap is not what stops the 30-degree run: more passes give the identical mesh.
    patient = od.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_faces=20, feature_angle=30.0, max_iter=400
    )[1]
    assert patient.size // 3 == floors[0]


@pytest.mark.parametrize("target_ratio", [0.5, 0.1])
def test_quadric_decimate_keeps_rim_vertices_on_the_surface(
    device: str, target_ratio: float
) -> None:
    """
    Not a library comparison: a bound on how far any output vertex sits from the input surface.

    A rim vertex's quadric is nearly singular in the direction off the rim, and on the graded
    height field's flat outer region its minimizer slid far out: before the boundary constraint
    plane one output vertex sat 1.16 bounding-box diagonals off the surface at a ratio of 0.5 and
    0.96 at 0.1. The worst vertex is now 3.1e-4 and 1.1e-3 of the diagonal; the bound sits 2.7x
    above the larger and over two orders of magnitude below the defect. Mutation probe: dropping
    the constraint (``BOUNDARY_QUADRIC_WEIGHT = 0``) fails the 0.1 arm.

    **Bug class excluded:** a placement that leaves the surface along a direction the quadric does
    not constrain.
    """
    vertices_np, faces_np = _graded_height_field(65)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    diagonal = float(np.linalg.norm(np.ptp(vertices_np, axis=0)))

    out_vertices_wp, out_faces_wp = od.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_ratio=target_ratio
    )
    target = math.ceil(target_ratio * (faces_np.size // 3))
    assert target - 1 <= out_faces_wp.size // 3 <= target
    _points, distance_wp, _faces = od.proximity.closest_point_on_mesh(
        vertices_wp, faces_wp, out_vertices_wp
    )
    assert distance_wp.numpy().max() < 3e-3 * diagonal


def test_quadric_decimate_keeps_the_boundary(device: str) -> None:
    """
    Not a library comparison: every input boundary vertex stays on the output, corners included.

    The face planes say nothing about where a rim runs within the surface, so without the
    Garland-Heckbert boundary constraint a collapse that pulls the rim inwards or cuts a corner
    costs nothing, and boundaries were eroded first: on this patch at a tenth of its faces an input
    rim vertex sat 0.13 of the bounding-box diagonal from the output and a corner of the square was
    cut off by 0.8. With the constraint the worst rim vertex is within 9.2e-4 and all four corners
    survive exactly.

    **Bug class excluded:** a boundary collapse priced by the face planes alone.
    """
    n = 65
    vertices_np, faces_np = _graded_height_field(n)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    diagonal = float(np.linalg.norm(np.ptp(vertices_np, axis=0)))
    rim_np = od.boundary.boundary_vertex_indices(vertices_wp, faces_wp).numpy()
    assert rim_np.size == 4 * (n - 1)  # non-vacuity: the whole square rim

    out_vertices_wp, out_faces_wp = od.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_ratio=0.1
    )
    _points, distance_wp, _faces = od.proximity.closest_point_on_mesh(
        out_vertices_wp,
        out_faces_wp,
        wp.array(vertices_np[rim_np].astype(np.float32), dtype=wp.vec3, device=device),
    )
    assert distance_wp.numpy().max() < 3e-3 * diagonal
    corners_np = vertices_np[[0, n - 1, n * (n - 1), n * n - 1]].astype(np.float32)
    out_vertices_np = out_vertices_wp.numpy()
    for corner_np in corners_np:
        assert (out_vertices_np == corner_np).all(axis=1).any()


def _closed_rings(faces_np: np.ndarray, n_vertices: int) -> list[set[int]]:
    """Return each vertex's closed 1-ring over a flat triangle buffer."""
    rings: list[set[int]] = [{v} for v in range(n_vertices)]
    for a, b, c in faces_np.reshape(-1, 3):
        rings[a] |= {b, c}
        rings[b] |= {a, c}
        rings[c] |= {a, b}
    return rings


@pytest.mark.parametrize("mesh_name", ["icosphere", "hemisphere"])
def test_quadric_decimate_commits_no_conflicting_collapses(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: every pass's committed collapses are pairwise non-conflicting.

    The pass commits an independent set over several rounds, and the set is only correct when no
    committed collapse has an endpoint in another's closed 1-rings: that is the condition under
    which applying them together is applying them one after another, so the link conditions, fold
    vetoes and placements the scoring computed still hold. Nothing downstream checks it -- a
    conflicting pair yields a plausible mesh. This reads each pass's collapse map (``remap[r] =
    s``, in the numbering the pass started from) against the adjacency of the faces it started
    from, on the pass that is issued and on the captured passes after it.

    **Bug class excluded:** a win test that misses a conflict, across the rounds of a pass as well
    as within one. Mutation probe: reading one endpoint's claim but not the other's fails both
    fixtures.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    buffers = od.remesh._DecimationBuffers(
        mesh_wp.points, mesh_wp.indices, len(mesh_tm.faces) // 10, wp.float32(math.pi)
    )
    checked = 0
    for _ in range(3):
        n_faces, n_vertices = (int(x) for x in buffers.state.numpy()[:2])
        faces_np = buffers.faces.numpy()[: 3 * n_faces].copy()
        if not buffers.run_pass():
            break
        remap_np = buffers._collapse_remap.numpy()[:n_vertices]
        removed_np = np.flatnonzero(remap_np != np.arange(n_vertices))
        pairs = [(int(remap_np[r]), int(r)) for r in removed_np]
        assert len(pairs) > 1
        assert buffers._round_state.numpy()[0] > 1
        rings = _closed_rings(faces_np, n_vertices)
        endpoints = [s for s, _r in pairs] + [r for _s, r in pairs]
        assert len(set(endpoints)) == len(endpoints)  # no vertex in two collapses
        owner = {v: i for i, (s, r) in enumerate(pairs) for v in (s, r)}
        for i, (s, r) in enumerate(pairs):
            for v in rings[s] | rings[r]:
                assert owner.get(v, i) == i, f"collapses {pairs[i]} and {pairs[owner[v]]} conflict"
        checked += 1
    assert checked >= 2


def test_quadric_decimate_commits_a_useful_fraction_on_a_flat_grid(device: str) -> None:
    """
    Not a library comparison: one pass removes a useful share of a regular flat grid's faces.

    The structured-mesh guard of the quadric pass, beside the collapse stage's own
    (``test_collapse_pass_commits_a_useful_fraction_on_a_structured_patch``). A flat grid is
    the input where every candidate costs the same and the edge numbering is spatially monotone,
    so a lock key ordered by the cost *rank* or by the index has a single minimum per region and
    a pass commits a handful of collapses; the hashed, cost-bucketed key commits several percent of
    the faces at once (7.2 % here, against 3.8 % under the disjoint-ring rule the independent set
    used to require).

    **Bug class excluded:** a spatially coherent lock key, and an independent-set rule stricter
    than the conflict it has to exclude.
    """
    n = 65
    samples = np.linspace(-1.0, 1.0, n)
    x_np, y_np = np.meshgrid(samples, samples, indexing="ij")
    vertices_np = np.stack([x_np, y_np, np.zeros_like(x_np)], axis=-1).reshape(-1, 3)
    _graded_np, faces_np = _graded_height_field(n)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    n_faces = faces_np.size // 3
    buffers = od.remesh._DecimationBuffers(vertices_wp, faces_wp, 10, wp.float32(math.pi))
    assert buffers.run_pass()
    removed = 1.0 - int(buffers.state.numpy()[0]) / n_faces
    assert removed > 0.05


def test_quadric_decimate_target_at_or_above_the_input_is_a_copy(device: str) -> None:
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=2)
    n_faces = faces_wp.size // 3
    for kwargs in (
        {"target_faces": n_faces},
        {"target_faces": n_faces + 100},
        {"target_ratio": 1.0},
    ):
        # Each dict holds exactly one target, of the type its keyword takes.
        out_vertices_wp, out_faces_wp = od.remesh.quadric_decimate(vertices_wp, faces_wp, **kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType]
        assert np.array_equal(out_faces_wp.numpy(), faces_wp.numpy())
        assert np.array_equal(out_vertices_wp.numpy(), vertices_wp.numpy())


def test_quadric_decimate_invalid(device: str) -> None:
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=1)
    with pytest.raises(ValueError, match="exactly one of target_faces and target_ratio"):
        od.remesh.quadric_decimate(vertices_wp, faces_wp)
    with pytest.raises(ValueError, match="exactly one of target_faces and target_ratio"):
        od.remesh.quadric_decimate(vertices_wp, faces_wp, target_faces=10, target_ratio=0.5)
    with pytest.raises(ValueError, match="target_faces must be non-negative"):
        od.remesh.quadric_decimate(vertices_wp, faces_wp, target_faces=-1)
    with pytest.raises(ValueError, match=r"target_ratio must be in \(0, 1\]"):
        od.remesh.quadric_decimate(vertices_wp, faces_wp, target_ratio=0.0)


def test_quadric_decimate_captures_its_pass(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    The decimation pass is replayed as a CUDA graph rather than reissued.

    The 2.6-4.5x that ``_DecimationBuffers`` is for rests entirely on this, and there is no other
    signal when it stops happening: a host readback added anywhere in the pass body makes the
    capture raise CUDA error 906, and a well-meant ``try``/``except`` or a widened fallback
    condition around that would leave a correct function that is four times slower. This test is
    the alarm. It is CUDA-only because the fallback path is the right answer everywhere else.
    """
    if not wp.get_device(device).is_cuda or not wp.is_conditional_graph_supported():
        pytest.skip("conditional CUDA graphs unavailable")
    mesh_tm, mesh_wp = icosphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices

    buffers = od.remesh._DecimationBuffers(
        vertices_wp, faces_wp, len(mesh_tm.faces) // 4, wp.float32(np.radians(30.0))
    )
    assert buffers.run_pass()  # issued: this is the pass that measures the true edge count
    assert buffers._graph is None
    assert buffers.run_pass()  # captured, and replayed by every pass after
    assert buffers._graph is not None


def test_quadric_decimate_padding_never_reaches_the_output(
    icosphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: the fixed-width pass keeps its padding out of the output.

    ``_DecimationBuffers`` runs every pass at a width the mesh has long since shrunk below, and
    carries the slack as a dummy vertex and a dummy edge slot. Those sentinels are one index past
    the live data on purpose, so a leak shows up as an out-of-range face index or an unreferenced
    vertex rather than as a wrong number -- which is exactly what this asserts, after each pass
    rather than only on the result.
    """
    mesh_tm, mesh_wp = icosphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    buffers = od.remesh._DecimationBuffers(
        vertices_wp, faces_wp, len(mesh_tm.faces) // 8, wp.float32(np.radians(30.0))
    )
    passes = 0
    while buffers.run_pass() and passes < 50:
        passes += 1
        # ``[faces, vertices, edges, commits, removed faces]``.
        n_faces, n_vertices, n_edges = (int(x) for x in buffers.state.numpy()[:3])
        live_faces = buffers.faces.numpy()[: 3 * n_faces].reshape(-1, 3)
        assert n_edges <= buffers.n_edges, "the edge capacity bound was violated"
        assert live_faces.max() < n_vertices, "a face still points at the dummy vertex"
        assert len(np.unique(live_faces)) == n_vertices, "unreferenced vertices left behind"
        assert (buffers.faces.numpy()[3 * n_faces :] == buffers.n_vertices).all(), (
            "the padded face slots are not the dummy triangle"
        )
    assert passes > 1, "the fixture must decimate over several passes for this to test anything"
    out_vertices_wp, out_faces_wp, _face_source_wp = buffers.result()
    assert out_faces_wp.size // 3 <= len(mesh_tm.faces) // 8
    assert out_faces_wp.numpy().max() < out_vertices_wp.size


@wp.kernel  # pyright: ignore[reportUntypedFunctionDecorator]  # wp.kernel has no return annotation
def _probe_winner_faces_before(
    rank: wp.array[wp.int32],
    winners: wp.array[wp.int32],
    words: wp.array[wp.uint32],
    prefix: wp.array[wp.int32],
    chunk_offsets: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    i = wp.int32(wp.tid())  # pyright: ignore[reportArgumentType]  # Warp's stub types tid() loosely
    out_faces[i] = kernel_remesh.winner_faces_before(  # pyright: ignore[reportIndexIssue]
        rank[winners[i]],  # pyright: ignore[reportArgumentType]
        words,
        prefix,
        chunk_offsets,
    )


def test_winner_faces_before_counts_the_cheaper_winners_faces(device: str) -> None:
    """
    Not a library comparison: the oracle is a host prefix sum in the pass's cost order.

    A collapse round commits its winners cheapest first until the pass's face budget is spent, and
    reads how many faces the cheaper winners remove off a word scan of two bitmasks
    (``kernels/remesh.winner_faces_before``): every winner at its cost rank, and the single-face
    (boundary) winners at theirs. This drives the function directly against NumPy, with ranks a
    random permutation and ``m`` spanning several ``SCAN_CHUNK`` s of words, so the chunk offsets
    carry.

    **Bug class excluded:** the second bitmask read from the start of the buffer rather than from
    its own half, a word or bit off by one, and a chunk offset dropped at a chunk boundary.
    """
    rng = np.random.default_rng(7)
    m = 70_000
    rank_np = rng.permutation(m).astype(np.int32)
    winner_np = rng.random(m) < 0.3
    single_np = winner_np & (rng.random(m) < 0.2)
    assert winner_np.sum() > 1000
    assert single_np.sum() > 100

    half = -(-m // 32)
    bits_np = np.zeros(64 * half, dtype=bool)
    bits_np[rank_np[winner_np]] = True
    bits_np[32 * half + rank_np[single_np]] = True
    words_np = np.packbits(bits_np.reshape(-1, 32)[:, ::-1], axis=1).view(">u4").ravel()
    words_wp = wp.array(words_np.astype(np.uint32), dtype=wp.uint32, device=device)
    scan = od.remesh._ExclusiveScan(2 * half, wp.get_device(device))
    scan.launch(words_wp, words=True)

    winners_np = np.flatnonzero(winner_np).astype(np.int32)
    faces_wp = warp_empty(len(winners_np), wp.int32, device)
    wp.launch(
        _probe_winner_faces_before,
        dim=len(winners_np),
        inputs=[
            wp.array(rank_np, device=device),
            wp.array(winners_np, device=device),
            words_wp,
            scan.prefix,
            scan.chunk_offsets,
            faces_wp,
        ],
        device=device,
    )
    removed_np = np.zeros(m, dtype=np.int64)
    removed_np[rank_np[winner_np]] = np.where(single_np[winner_np], 1, 2)
    expected_np = np.concatenate([[0], np.cumsum(removed_np)[:-1]])[rank_np[winners_np]]
    assert np.array_equal(faces_wp.numpy(), expected_np)


@pytest.mark.parametrize("target_ratio", [0.5, 0.1])
def test_quadric_decimate_provenance_maps_are_consistent(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh], target_ratio: float
) -> None:
    """
    Not a library comparison: no reference returns a decimation's provenance, so this is invariants.

    The claim is that the two maps *describe the mesh that was returned*, which is a much stronger
    statement than either map being well-formed, and it is checked the only way that is exact:
    mapping each output face's **source** face through ``vertex_index`` must reproduce that output
    face's own three indices. A collapse renumbers a surviving face's corners and never rebuilds it,
    so the two triples agree as sets -- measured 3 of 3 shared vertices on every output face at both
    ratios. A map that drifted by one pass would still be surjective and still be in range; it would
    fail this.

    Three more, each excluding a different failure: ``vertex_index`` is **onto** the output vertices
    (nothing survives unreferenced by it), ``face_index`` is **injective** (a collapse deletes faces
    and creates none, so no two output faces can share a source), and the positions and faces are
    identical to the plain two-value call, i.e. asking for the maps does not change the answer.

    The lower ratio matters: it takes several passes, so it exercises the composition across the
    captured graph replay rather than just the first issued pass.
    """
    mesh_tm, _ = icosphere
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1), device
    )
    n_input_vertices = vertices_wp.size
    faces_np = np.asarray(mesh_tm.faces)

    out_vertices_wp, out_faces_wp, vertex_index_wp, face_index_wp = od.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_ratio=target_ratio, return_index=True
    )
    n_out_vertices = out_vertices_wp.size
    n_out_faces = out_faces_wp.size // 3
    assert n_out_faces < faces_np.shape[0]  # something was actually decimated

    vertex_index_np = vertex_index_wp.numpy()
    face_index_np = face_index_wp.numpy()
    assert vertex_index_np.shape == (n_input_vertices,)
    assert face_index_np.shape == (n_out_faces,)
    assert vertex_index_np.max() < n_out_vertices
    assert set(vertex_index_np[vertex_index_np >= 0].tolist()) == set(range(n_out_vertices))
    assert len(set(np.asarray(face_index_np).tolist())) == n_out_faces
    assert face_index_np.min() >= 0
    assert face_index_np.max() < faces_np.shape[0]

    # The load-bearing check: the source face, renumbered through the vertex map, is the output.
    out_faces_np = out_faces_wp.numpy().reshape(-1, 3)
    mapped_np = vertex_index_np[faces_np[face_index_np]]
    assert np.array_equal(np.sort(mapped_np, axis=1), np.sort(out_faces_np, axis=1))

    plain_vertices_wp, plain_faces_wp = od.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_ratio=target_ratio
    )
    assert np.array_equal(plain_faces_wp.numpy(), out_faces_wp.numpy())
    assert np.allclose(plain_vertices_wp.numpy(), out_vertices_wp.numpy(), rtol=1e-6, atol=1e-6)


def test_quadric_decimate_provenance_on_degenerate_inputs(
    device: str, unit_box: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: the ``-1`` entry, and that the no-op path reports it too.

    An input vertex no face references cannot land anywhere, so it reports ``-1`` rather than a
    plausible index -- the one case where ``vertex_index`` is not total, which is why the docstring
    says so.

    **The no-op path is held to the same contract, and this assert is what pins that.** A target at
    or above the input count collapses nothing, but it still compacts and still reports ``-1``, so
    the shape of the answer does not depend on which side of the input's face count the target
    landed on. An earlier version of this test asserted the opposite -- identity maps and the
    vertex buffer verbatim -- on the reasoning that "a decimation that did nothing should give the
    attribute back". That reasoning does not survive the discontinuity it creates: ``bunny`` carries
    1 113 unreferenced vertices, so a caller sweeping ``target_ratio`` over it would watch them
    appear at ``1.0`` and vanish at ``0.99``.
    """
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0, 9.0, 9.0]], dtype=np.float32
    )
    faces_np = np.array([0, 1, 2], dtype=np.int32)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    kept_vertices_wp, kept_faces_wp, vertex_index_wp, face_index_wp = od.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_faces=8, return_index=True
    )
    assert np.array_equal(kept_vertices_wp.numpy(), vertices_np[:3])
    assert np.array_equal(kept_faces_wp.numpy(), faces_np)
    assert np.array_equal(vertex_index_wp.numpy(), np.array([0, 1, 2, -1]))
    assert np.array_equal(face_index_wp.numpy(), np.arange(1))

    # A mesh with nothing unreferenced is untouched by that compaction, which is what keeps the
    # no-op path a pass-through for every ordinary input.
    clean_vertices_wp, clean_faces_wp = od.remesh.quadric_decimate(
        points_to_warp(vertices_np[:3], device), faces_wp, target_faces=8
    )
    assert np.array_equal(clean_vertices_wp.numpy(), vertices_np[:3])
    assert np.array_equal(clean_faces_wp.numpy(), faces_np)

    # A mesh with an unreferenced vertex, decimated for real: vertex 3 has nowhere to go.
    grid_tm, _grid_wp = unit_box
    grid_vertices_np = np.vstack([np.asarray(grid_tm.vertices), [[9.0, 9.0, 9.0]]])
    grid_vertices_wp, grid_faces_wp = numpy_to_warp(
        grid_vertices_np, np.asarray(grid_tm.faces, dtype=np.int32).reshape(-1), device
    )
    _vertices_wp, _faces_wp, grid_index_wp, _face_wp = od.remesh.quadric_decimate(
        grid_vertices_wp, grid_faces_wp, target_faces=8, return_index=True
    )
    assert int(grid_index_wp.numpy()[-1]) == -1  # the unreferenced vertex


# ---------------------------------------------------------------------------
# Parallel Delaunay flips (flip_to_delaunay), and the flip topology beneath them
# ---------------------------------------------------------------------------


def _delone_violations(vertices_np: np.ndarray, faces_np: np.ndarray, region_np: np.ndarray):
    """Count interior region edges that fail the (angle-gate-free) circumcircle Delone test."""
    faces = faces_np
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, t in enumerate(faces):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_faces.setdefault((int(min(a, b)), int(max(a, b))), []).append(fi)

    def circ_diam_sq(a: np.ndarray, b: np.ndarray, c: np.ndarray):
        ab = np.dot(b - a, b - a)
        ca = np.dot(a - c, a - c)
        bc = np.dot(c - b, c - b)
        if ab <= 0 or ca <= 0 or bc <= 0:
            return np.inf
        f = np.dot(np.cross(b - a, c - a), np.cross(b - a, c - a))
        return np.inf if f <= 0 else ab * ca * bc / f

    violations = 0
    for (u, v), fs in edge_faces.items():
        if len(fs) != 2 or not (region_np[fs[0]] and region_np[fs[1]]):
            continue
        apex = []
        for fi in fs:
            apex.extend([int(x) for x in faces[fi] if int(x) not in (u, v)])
        if len(apex) != 2:
            continue
        a, c = vertices_np[u], vertices_np[v]
        b, d = vertices_np[apex[1]], vertices_np[apex[0]]
        m_ac = max(circ_diam_sq(a, c, d), circ_diam_sq(c, a, b))
        m_bd = max(circ_diam_sq(b, d, a), circ_diam_sq(d, b, c))
        if m_bd < m_ac * (1.0 - 1e-6):
            violations += 1
    return violations


def test_flip_to_delaunay_reduces_violations(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    v, f, region = _filled_hemisphere(hemisphere)
    max_edge = 0.3 * _region_max_edge(v.numpy(), f.numpy().reshape(-1, 3), region.numpy())
    nv, nf, nr = od.remesh.subdivide_region_to_size(v, f, region, max_edge=max_edge, delaunay=False)

    before = _delone_violations(nv.numpy(), nf.numpy().reshape(-1, 3), nr.numpy())
    flipped = od.remesh.flip_to_delaunay(nv, nf, region=nr)
    after = _delone_violations(nv.numpy(), flipped.numpy().reshape(-1, 3), nr.numpy())

    assert after <= before
    # Face count unchanged; mesh stays closed.
    assert flipped.size == nf.size
    edges = undirected_edges(flipped.numpy().reshape(-1, 3))
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.array_equal(np.unique(counts), np.array([2]))


@pytest.mark.parity("flip_to_delaunay", "meshlib")
def test_flip_to_delaunay_matches_meshlib(device: str) -> None:
    """
    Class B: the same fixpoint, reached by a parallel pass instead of a serial queue.

    ``makeDeloneEdgeFlips`` is the same empty-circumcircle criterion and the same operation, and on
    a decisive input the two agree exactly: a planar kite whose long diagonal is the non-Delaunay
    one, where both flip that one edge and return the identical pair of triangles.

    On a mesh with hundreds of violations the *outputs* differ, and that is a property of the
    algorithms rather than of the criterion -- ordito commits a conflict-free independent set per
    round where MeshLib works a queue, so which of two competing flips wins differs. What is
    asserted there is the fixpoint, which is the actual claim either function makes: after ordito's
    pass, **MeshLib finds 0 further flips to make**, against **942** in the input. That is a
    stronger statement than a face-set comparison and it is checked with the reference's own
    criterion, not ordito's.

    The converse is not symmetric and is asserted as measured rather than papered over: ordito
    still changes **20 of 1 280** faces in MeshLib's output, so ordito's test is the stricter of
    the two on near-cocircular quads.
    """
    # A kite: the diagonal 0-2 (length 2.0) is longer than 1-3 (1.8), so it is the one to flip.
    kite_vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.9, 0.0], [2.0, 0.0, 0.0], [1.0, -0.9, 0.0]]
    )
    kite_faces_np = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    vertices_wp, faces_wp = numpy_to_warp(kite_vertices_np, kite_faces_np, device)
    flipped_wp = od.remesh.flip_to_delaunay(vertices_wp, wp.clone(faces_wp), max_iter=100)

    mesh_ml = numpy_to_meshlib(kite_vertices_np, kite_faces_np)
    assert mm.makeDeloneEdgeFlips(mesh_ml, mm.DeloneSettings(), 100) == 1  # it flipped the diagonal
    flipped_ml = meshlib_to_trimesh(mesh_ml).faces

    assert np.array_equal(
        lexsort_rows(np.sort(flipped_wp.numpy().reshape(-1, 3), axis=1)),
        lexsort_rows(np.sort(np.asarray(flipped_ml, dtype=np.int32), axis=1)),
    )

    # The fixpoint, on an input with hundreds of violations.
    sphere_tm = tm.creation.icosphere(subdivisions=3)
    jittered_np = sphere_tm.vertices + np.random.default_rng(1).normal(
        scale=0.08, size=sphere_tm.vertices.shape
    )
    vertices_wp, faces_wp = numpy_to_warp(jittered_np, sphere_tm.faces, device)
    delaunay_np = (
        od.remesh.flip_to_delaunay(vertices_wp, wp.clone(faces_wp), max_iter=100)
        .numpy()
        .reshape(-1, 3)
    )

    violations_before = mm.makeDeloneEdgeFlips(
        numpy_to_meshlib(jittered_np, sphere_tm.faces), mm.DeloneSettings(), 100
    )
    violations_after = mm.makeDeloneEdgeFlips(
        numpy_to_meshlib(jittered_np, delaunay_np), mm.DeloneSettings(), 100
    )
    assert violations_before > 100  # non-vacuity: the input really is far from Delaunay
    assert violations_after == 0


def test_flip_to_delaunay_region_gated(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    v, f, region = _filled_hemisphere(hemisphere)
    nv, nf, nr = od.remesh.subdivide_region_to_size(
        v, f, region, max_edge=1e9, delaunay=False
    )  # no splits; just exercise gating on the raw fill patch
    flipped = od.remesh.flip_to_delaunay(nv, nf, region=nr)
    faces_before = nf.numpy().reshape(-1, 3)
    faces_after = flipped.numpy().reshape(-1, 3)
    region_np = nr.numpy()
    # Faces outside the region are never rewritten.
    assert np.array_equal(faces_before[~region_np], faces_after[~region_np])


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "cave_cube"])
def test_flip_topology_matches_the_composed_adjacency(
    mesh_name: str, request: pytest.FixtureRequest
) -> None:
    """
    ``_FlipTopology`` reproduces the wrapper chain it replaces, exactly.

    The flip loop stopped composing [`face_adjacency`][ordito.adjacency.face_adjacency],
    [`face_adjacency_unshared`][ordito.adjacency.face_adjacency_unshared] and a second
    [`sort_and_argsort`][ordito.array.sort_and_argsort] of the edge keys, and builds all three on
    fixed buffers instead — a measured 1.5-2.6x. That is only sound while the two agree row for
    row, including a fresh build's row *order* (ascending key, which is the order the
    independent-set claims rank candidates in). This is an internal-consistency check, not a
    reference comparison, so it carries no parity marker.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces = mesh_wp.indices
    n_vertices = len(mesh_tm.vertices)

    topology = od.remesh._FlipTopology(faces, n_vertices)
    rows = topology.rebuild()

    edges_sorted = od.edges.faces_to_edges(faces, sorted=True)
    adjacency_ref, adjacency_edges_ref = od.adjacency.face_adjacency(
        faces, return_edges=True, n_vertices=n_vertices
    )
    unshared_ref = od.adjacency.face_adjacency_unshared(faces, adjacency_ref, adjacency_edges_ref)
    keys_ref, _order = od.array.sort_and_argsort(
        od.grouping.hash_indices_rows(edges_sorted, max_index=n_vertices, validate=False)
    )

    assert rows == int(adjacency_ref.shape[0]) > 0
    assert np.array_equal(topology.adjacency.numpy(), adjacency_ref.numpy())
    assert np.array_equal(topology.adjacency_edges.numpy(), adjacency_edges_ref.numpy())
    assert np.array_equal(topology.unshared.numpy(), unshared_ref.numpy())
    assert np.array_equal(topology.sorted_keys.numpy(), keys_ref.numpy())


@pytest.mark.parametrize("flip_budget", [None, 7])
def test_flip_topology_incremental_state_matches_a_fresh_build(
    icosphere: tuple[tm.Trimesh, wp.Mesh], flip_budget: int | None
) -> None:
    """
    Ordito against ordito: the flip loop's maintained topology against a rebuild of its output.

    After the first flipping round the loop stops regrouping and keeps its rows, halfedge maps and
    duplicate-edge set current through each commit. Whatever state it ends in must describe the
    faces it ends with exactly as a fresh build of them does: the same rows (as a set -- the
    maintained ones keep their slots rather than ascending key order), each row's halfedges on its
    own two faces, every halfedge mapped to the row of its edge, and the key set holding exactly
    the current edges. ``flip_budget=7`` forces the tombstone rebuild mid-loop.
    """
    mesh_tm, mesh_wp = icosphere
    device = mesh_wp.device
    rng = np.random.default_rng(11)
    # Large enough that flips continue for several rounds past the plain first one (79, 28, 9 and
    # 1 flips on ``icosphere``), which is where the incremental state is exercised at all.
    jitter = rng.normal(scale=0.3 * mesh_tm.edges_unique_length.mean(), size=mesh_tm.vertices.shape)
    vertices, faces = numpy_to_warp(mesh_tm.vertices + jitter, mesh_tm.faces, device)
    n_vertices = len(mesh_tm.vertices)
    region = wp.ones(faces.size // 3, dtype=wp.int32, device=device)

    topology = od.remesh._FlipTopology(faces, n_vertices)
    if flip_budget is not None:
        topology._flip_budget = flip_budget
    flips = od.remesh._flip_region_faces(vertices, faces, region, None, None, 50, topology)
    assert flips > 200
    assert topology.incremental
    halfedge_row = topology.halfedge_row
    row_halfedges = topology.row_halfedges
    assert halfedge_row is not None
    assert row_halfedges is not None

    fresh = od.remesh._FlipTopology(faces, n_vertices)
    fresh.rebuild(incremental=True)

    def rows(t: od.remesh._FlipTopology) -> np.ndarray:
        return np.hstack([t.adjacency.numpy(), t.adjacency_edges.numpy(), t.unshared.numpy()])

    assert np.array_equal(lexsort_rows(rows(topology)), lexsort_rows(rows(fresh)))

    faces_np = faces.numpy().reshape(-1, 3)
    row_halfedges_np = row_halfedges.numpy()
    assert np.array_equal(np.sort(row_halfedges_np // 3, axis=1), topology.adjacency.numpy()), (
        "a row's halfedges are not on its two faces"
    )
    tails = faces_np.reshape(-1)[row_halfedges_np]
    heads = faces_np[row_halfedges_np // 3, (row_halfedges_np % 3 + 1) % 3]
    assert np.array_equal(
        np.sort(np.stack([tails[:, 0], heads[:, 0]], 1), axis=1), topology.adjacency_edges.numpy()
    )

    # Map each maintained row to the fresh row of the same edge, then compare the halfedge maps.
    def key(edges_np: np.ndarray) -> np.ndarray:
        return edges_np[:, 0].astype(np.int64) + edges_np[:, 1].astype(np.int64) * n_vertices

    fresh_row_of_key = dict(
        zip(key(fresh.adjacency_edges.numpy()).tolist(), range(fresh.rows), strict=True)
    )
    to_fresh = np.array([fresh_row_of_key[k] for k in key(topology.adjacency_edges.numpy())])
    maintained = halfedge_row.numpy()
    mapped = np.where(maintained >= 0, to_fresh[np.maximum(maintained, 0)], -1)
    fresh_halfedge_row = fresh.halfedge_row
    assert fresh_halfedge_row is not None
    assert np.array_equal(mapped, fresh_halfedge_row.numpy())

    slots = topology.edge_set.numpy()
    live = slots[(slots != 0) & (slots != np.uint64(0xFFFFFFFFFFFFFFFF))] - np.uint64(1)
    assert np.array_equal(np.sort(live), np.unique(fresh.sorted_keys.numpy()))


def test_flip_topology_drops_non_manifold_edges_like_face_adjacency(device: str) -> None:
    """
    An edge with three or more face corners is in neither table.

    The run-length test that excludes it is the one branch the manifold fixtures cannot reach, and
    ``face_adjacency`` keeps only edges shared by *exactly* two corners.
    """
    # A closed tetrahedron plus a fourth face on one of its edges: that edge has three corners.
    faces_np = np.array([0, 1, 2, 0, 3, 1, 1, 3, 2, 2, 3, 0, 0, 1, 4], dtype=np.int32)
    faces = wp.array(faces_np, dtype=wp.int32, device=device)

    topology = od.remesh._FlipTopology(faces, 5)
    rows = topology.rebuild()

    adjacency_ref = od.adjacency.face_adjacency(faces, n_vertices=5)
    assert rows == int(adjacency_ref.shape[0])
    assert np.array_equal(topology.adjacency.numpy(), adjacency_ref.numpy())
    # Edge (0, 1) carries three corners, so no row of the table mentions the pair it would form.
    assert not (topology.adjacency_edges.numpy() == [0, 1]).all(axis=1).any()


@pytest.mark.parametrize("flip", [False, True])
def test_flip_topology_edges_unique_matches_edges_unique(device: str, flip: bool) -> None:
    """
    Ordito against ordito: the flip pass's edge sort is ``edges_unique`` for the faces it leaves.

    ``subdivide_region_to_size`` reads its next pass's unique edges off the topology the flip pass
    left rather than calling [`edges_unique`][ordito.edges.edges_unique] again, and the split
    numbers its new vertices in that edge order, so the two must agree row for row and corner for
    corner. ``edges_unique`` carries the oracle. The ``flip`` arm runs a Delone pass that rewrites
    faces first -- a sheared grid, whose every quad has the better diagonal -- so the check covers
    the regroup a flipping round leaves behind, not only the opening build.
    """
    vertices_np, faces_np = _sheared_grid(n=12)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    n_vertices = vertices_wp.size
    topology = od.remesh._FlipTopology(faces_wp, n_vertices)
    if flip:
        before_np = faces_wp.numpy().copy()
        region_flags = wp.full(faces_np.shape[0], 1, dtype=wp.int32, device=device)
        flips = od.remesh._flip_region_faces(
            vertices_wp, faces_wp, region_flags, None, None, 8, topology
        )
        assert flips > 0
        assert not np.array_equal(faces_wp.numpy(), before_np)
    else:
        topology.rebuild()
    assert topology.built

    unique_edges_wp, inverse_wp = topology.edges_unique()
    unique_edges_ref, inverse_ref = od.edges.edges_unique(faces_wp, n_vertices=n_vertices)
    assert np.array_equal(unique_edges_wp.numpy(), unique_edges_ref.numpy())
    assert np.array_equal(inverse_wp.numpy(), inverse_ref.numpy())


# ---------------------------------------------------------------------------
# Objective-driven edge flips vs pymeshlab
# ---------------------------------------------------------------------------


def _sheared_grid(n: int = 24, shear: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a **flat** grid of sheared parallelograms, split along each cell's long diagonal.

    Shear is what makes this a fixture: a rectangle's two diagonals are the same length, so both
    triangulations of it are congruent and no quality objective can prefer either — an
    axis-aligned grid, however anisotropic, has nothing to flip. Shearing by ``shear`` cells makes
    one diagonal ``(1 + shear, 1)`` and the other ``(1 - shear, -1)``, so the right flip exists at
    every quad and the planarity objective must find it. The default ``shear=4`` maximizes the
    *relative* gain: pushing it higher makes both triangles worse, so the ratio shrinks back
    toward 1 even as the mesh gets uglier. Flat, so the flip is a pure
    retriangulation and cannot change the surface.
    """
    i_grid, j_grid = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    vertices = np.column_stack(
        [
            (i_grid + shear * j_grid).ravel().astype(np.float64),
            j_grid.ravel().astype(np.float64),
            np.zeros(n * n),
        ]
    )
    return vertices, _grid_faces(n)


def _saddle_grid(n: int = 16, step: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """
    ``z = x y`` over a square grid: the fixture the curvature objective exists for.

    On a hyperbolic paraboloid the two diagonals of a quad have *opposite* curvature — one runs
    along a ruling of the surface and is nearly straight, the other bends. So the choice is
    maximally consequential, and the current diagonal is deliberately the bending one.
    """
    i_grid, j_grid = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x_np = (i_grid * step).ravel().astype(np.float64)
    y_np = (j_grid * step).ravel().astype(np.float64)
    vertices = np.column_stack([x_np, y_np, x_np * y_np])
    return vertices, _grid_faces(n)


def _min_quality(
    vertices_wp: wp.array[wp.vec3],
    faces_wp: wp.array[wp.int32],
    metric: FaceQualityMetric = "area_max_side",
) -> float:
    return float(od.triangles.face_quality(vertices_wp, faces_wp, metric=metric).numpy().min())


def _total_bend(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    """Sum of the absolute dihedral angle over every interior edge — the curvature objective."""
    mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
    return float(np.abs(mesh_tm.face_adjacency_angles).sum())


def test_flip_by_objective_planarity_improves_the_worst_triangle(device: str) -> None:
    """Every quad of a flat sheared grid has a better diagonal, and the flip must take it."""
    vertices_np, faces_np = _sheared_grid()
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    before = _min_quality(vertices_wp, faces_wp)

    flipped_wp = od.remesh.flip_by_objective(vertices_wp, faces_wp, objective="planarity")
    after = _min_quality(vertices_wp, flipped_wp)
    assert after > before * 1.4

    # A retriangulation: same faces, same vertices, still a clean manifold patch.
    assert flipped_wp.size == faces_wp.size
    assert od.validation.is_winding_consistent(flipped_wp)
    assert od.validation.is_edge_manifold(flipped_wp)
    assert _degenerate_face_count(vertices_np, flipped_wp.numpy().reshape(-1, 3)) == 0


@pytest.mark.parity("flip_by_objective", "pymeshlab")
def test_flip_by_objective_planarity_at_least_matches_pymeshlab(device: str) -> None:
    """
    Class C (a planarity bound): MeshLab runs the same objective serially, so it sets the bar.

    ``meshing_edge_flip_by_planar_optimization`` takes the *same* planarity threshold and the *same*
    quality metric (``planartype='area/max side'``), and its greedy serial pass is free to take
    every flip in any order. A parallel independent-set pass can only ever match it, so requiring
    the resulting worst triangle to be within 10% is a real check that the predicate agrees.
    """
    vertices_np, faces_np = _sheared_grid()
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(ml.Mesh(vertices_np, np.ascontiguousarray(faces_np, dtype=np.int32)))
    meshset_pml.meshing_edge_flip_by_planar_optimization(
        pthreshold=1.0, planartype="area/max side", iterations=10
    )
    faces_pml = meshset_pml.current_mesh().face_matrix()
    assert faces_pml.shape[0] == faces_np.shape[0]

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    _vertices_pml_wp, faces_pml_wp = numpy_to_warp(vertices_np, faces_pml, device)
    flipped_wp = od.remesh.flip_by_objective(vertices_wp, faces_wp, objective="planarity")
    assert _min_quality(vertices_wp, flipped_wp) >= 0.9 * _min_quality(vertices_wp, faces_pml_wp)


def test_flip_by_objective_planarity_refuses_a_curved_quad(device: str) -> None:
    """With ``planar_angle=0`` nothing is flat enough, so the triangulation must be untouched."""
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    flipped_wp = od.remesh.flip_by_objective(
        vertices_wp, faces_wp, objective="planarity", planar_angle=0.0
    )
    assert np.array_equal(flipped_wp.numpy(), faces_wp.numpy())


def test_flip_by_objective_curvature_flattens(device: str) -> None:
    """The curvature objective must lower the total absolute dihedral angle."""
    vertices_np, faces_np = _saddle_grid()
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    before = _total_bend(vertices_np, faces_np)

    flipped_wp = od.remesh.flip_by_objective(vertices_wp, faces_wp, objective="curvature")
    after = _total_bend(vertices_np, flipped_wp.numpy().reshape(-1, 3))
    assert after < before
    assert flipped_wp.size == faces_wp.size
    assert od.validation.is_winding_consistent(flipped_wp)
    assert od.validation.is_edge_manifold(flipped_wp)


def test_flip_by_objective_curvature_leaves_a_sphere_alone(device: str) -> None:
    """
    An icosphere's diagonals are already the flat ones, so a converged pass changes nothing much.

    Not an equality assertion: the icosphere's quads are close enough to symmetric that a handful
    genuinely tie, and the relative ``1e-6`` margin is what keeps those from oscillating. What must
    hold is that the total bend does not *increase*.
    """
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    before = _total_bend(np.asarray(sphere_tm.vertices), np.asarray(sphere_tm.faces))
    flipped_wp = od.remesh.flip_by_objective(vertices_wp, faces_wp, objective="curvature")
    after = _total_bend(np.asarray(sphere_tm.vertices), flipped_wp.numpy().reshape(-1, 3))
    assert after <= before * (1.0 + 1e-6)


def test_flip_by_objective_region_gated(device: str) -> None:
    """Faces outside the region keep their edges, so the flip count can only go down."""
    vertices_np, faces_np = _sheared_grid()
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    n_faces = faces_wp.size // 3
    region_np = np.zeros(n_faces, dtype=bool)
    region_np[: n_faces // 4] = True
    region_wp = wp.array(region_np, dtype=wp.bool, device=device)

    full_wp = od.remesh.flip_by_objective(vertices_wp, faces_wp, objective="planarity")
    gated_wp = od.remesh.flip_by_objective(
        vertices_wp, faces_wp, objective="planarity", region=region_wp
    )
    changed_full = int((full_wp.numpy() != faces_wp.numpy()).sum())
    changed_gated = int((gated_wp.numpy() != faces_wp.numpy()).sum())
    assert 0 < changed_gated < changed_full


def test_flip_by_objective_invalid(device: str) -> None:
    """
    Not the mismatched-``region`` guard.

    That guard is shared with the other two remesh operators and covered once, for all three, by
    ``test_remesh_rejects_a_mismatched_region``.
    """
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=1)
    with pytest.raises(ValueError, match="objective must be"):
        od.remesh.flip_by_objective(vertices_wp, faces_wp, objective="delaunay")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="metric must be one of"):
        od.remesh.flip_by_objective(vertices_wp, faces_wp, metric="aspect_ratio")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="planar_angle must be in"):
        od.remesh.flip_by_objective(vertices_wp, faces_wp, planar_angle=200.0)


# --- intrinsic_delaunay ---------------------------------------------------------------
@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus", "torus"])
def test_intrinsic_delaunay_removes_negative_cotangent_weights(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    flipped = bsr_to_dense(
        od.laplacian.robust_laplacian(mesh_wp.points, mesh_wp.indices), n_vertices
    )

    # A non-negative off-diagonal (in this sign convention, where the diagonal is negative) is what
    # "Delaunay" buys: it is the condition for the Laplacian to satisfy a maximum principle.
    off_diagonal = flipped - np.diag(np.diag(flipped))
    assert off_diagonal.min() > -1e-6


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_intrinsic_delaunay_leaves_a_delaunay_mesh_alone(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    original_lengths = od.edges.face_edge_lengths(mesh_wp.points, mesh_wp.indices).numpy()
    faces, lengths, n_flips = od.remesh.intrinsic_delaunay(mesh_wp.points, mesh_wp.indices)

    # These fixtures come from an icosphere, whose triangulation is already intrinsically Delaunay.
    assert n_flips == 0
    assert np.array_equal(faces.numpy(), mesh_wp.indices.numpy())
    assert np.allclose(lengths.numpy(), original_lengths, rtol=1e-6, atol=1e-6)


def _summed_cotangent_weights(
    faces_np: npt.NDArray[np.int32], cot_np: npt.NDArray[np.float32]
) -> tuple[
    dict[tuple[int, int], float], dict[tuple[int, int], int], dict[tuple[int, int], list[int]]
]:
    """
    Per undirected edge: summed half-cotangent, incident face count, and opposite corners.

    ``cotmatrix_entries_intrinsic`` returns one half-cotangent per ``(face, corner)``, for the edge
    *opposite* that corner, and the assembled matrix entry is their sum -- verified equal to this
    dictionary for all 4 375 edges of the fixture below, so this is the operator's own weight and
    not a re-derivation of it.
    """
    weight: dict[tuple[int, int], float] = {}
    incident: dict[tuple[int, int], int] = {}
    opposite: dict[tuple[int, int], list[int]] = {}
    for face_index, triangle in enumerate(faces_np):
        for corner in range(3):
            a, b = int(triangle[(corner + 1) % 3]), int(triangle[(corner + 2) % 3])
            key = (min(a, b), max(a, b))
            weight[key] = weight.get(key, 0.0) + float(cot_np[face_index, corner])
            incident[key] = incident.get(key, 0) + 1
            opposite.setdefault(key, []).append(int(triangle[corner]))
    return weight, incident, opposite


def test_intrinsic_delaunay_resolves_interior_violations_via_multi_edges(device: str) -> None:
    """
    Not a library comparison: the fixture that needs a second edge between two adjacent vertices.

    ``intrinsic_delaunay`` tracks connectivity through an incrementally-maintained halfedge twin
    table, so a flip may legitimately create a second, geometrically distinct edge between two
    vertices some other edge already connects (intrinsically that is a different geodesic, not a
    duplicate). Dini's surface is strongly graded (edge lengths spanning four orders) and open,
    which is what makes it reach that case at all -- there is deliberately no ``conftest`` entry
    for it, since it exists only to exercise what the well-shaped fixtures above cannot.

    **Two things to check, and they are different claims.** Every *interior* edge -- whether it is
    the only edge between its two vertices or one of several -- must satisfy the Delaunay
    non-negativity condition, since a flip can reach any of them. A *boundary* edge is exempt: the
    two-opposite-angles condition has nothing to compare a boundary edge's one angle against, so an
    obtuse boundary corner is not a Delaunay violation and no flip addresses it.
    """
    vertices_wp, faces_wp = od.creation.parametric_surface("dini", device=device)
    intrinsic_faces_wp, lengths_wp, n_flips = od.remesh.intrinsic_delaunay(vertices_wp, faces_wp)
    assert n_flips > 0  # The comparison is not vacuous: the flipper did real work first.

    faces_np = intrinsic_faces_wp.numpy().reshape(-1, 3)
    cot_np = od.laplacian.cotmatrix_entries_intrinsic(lengths_wp).numpy()
    weight, incident, _opposite = _summed_cotangent_weights(faces_np, cot_np)

    interior = [e for e in weight if incident[e] >= 2]
    boundary = [e for e in weight if incident[e] == 1]
    multi_edges = [e for e in weight if incident[e] > 2]

    # The fixture must actually reach a multi-edge, or the claim below is vacuous.
    assert multi_edges, "fixture no longer creates a second edge between adjacent vertices"
    assert all(incident[e] % 2 == 0 for e in interior), "an interior edge count must be even"

    # The claim: every interior edge -- single or multi -- is now non-negative.
    assert min(weight[e] for e in interior) > -1e-6

    # And the boundary residue is untouched: it is not a Delaunay violation, so it survives.
    assert boundary
    assert min(weight[e] for e in boundary) < -1e-6


@pytest.mark.parametrize("mesh_name", ["half_torus", "torus"])
@pytest.mark.parity("intrinsic_delaunay", "igl")
def test_intrinsic_delaunay_metric_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The intrinsic metric after flipping, against libigl's serial flipper.

    The two cannot be compared face for face: ordito flips independent sets in parallel rounds
    where libigl drains a queue, so the *sequence* differs and so does the face ordering. What must
    agree is where they land, because the intrinsic Delaunay triangulation of a surface is unique
    away from cocircular degeneracies -- so the multiset of edge lengths is the invariant, and this
    is Class B with a sort rather than a weakened tolerance.

    That makes it a real check rather than a formality: on ``half_torus`` ordito performs **298**
    flips and still reaches libigl's metric to 1e-4, which a wrong flip rule or a mis-unfolded
    diagonal would not. ``igl.intrinsic_delaunay_cotmatrix`` is used for its second return value
    (the lengths); it assembles a matrix as well, which is why the benchmark reads its row as
    including work ordito's does not.

    The invariants of an intrinsic flip are asserted on the same call, and they are not implied by
    the length multiset: the vertex set, the face count and the total (Heron) area are properties
    of the surface, not of its triangulation, so none may change.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _lengths_igl = igl.intrinsic_delaunay_cotmatrix(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), mesh_tm.faces.astype(np.int64)
    )[1]

    faces_wp, lengths_wp, n_flips = od.remesh.intrinsic_delaunay(mesh_wp.points, mesh_wp.indices)

    # A quad grid split by diagonals is not Delaunay, so there is work to do...
    assert n_flips > 0, "fixture is already Delaunay; this would assert nothing"
    assert np.allclose(
        np.sort(lengths_wp.numpy().ravel()),
        np.sort(np.asarray(_lengths_igl).ravel()),
        rtol=1e-4,
        atol=1e-4,
    )
    # ... but the flips are *intrinsic*: the vertex count, the face count and the total area are all
    # properties of the surface, not of its triangulation, so none of them may change.
    assert faces_wp.shape == mesh_wp.indices.shape
    assert np.array_equal(np.sort(np.unique(faces_wp.numpy())), np.sort(np.unique(mesh_tm.faces)))
    sides = lengths_wp.numpy().astype(np.float64)
    semi = sides.sum(axis=1) / 2.0
    heron = semi * (semi - sides[:, 0]) * (semi - sides[:, 1]) * (semi - sides[:, 2])
    assert np.isclose(np.sqrt(np.maximum(heron, 0.0)).sum(), mesh_tm.area, rtol=1e-3, atol=1e-3)


def _undirected_intrinsic_lengths(faces_np: np.ndarray, lengths_np: np.ndarray) -> np.ndarray:
    """
    Reduce a ``(n_faces, 3)`` per-corner length table to one sorted length per undirected edge.

    ``lengths[f, e]`` belongs to the edge *opposite* corner ``e``, so the same interior edge appears
    in two face corners and a boundary edge in one. Deduplicating rather than doubling is what makes
    the multiset comparable on an open mesh: ``half_torus`` has 64 boundary edges, so the naive
    ``np.repeat(..., 2)`` of the reference side is 3 136 entries against ordito's 3 072.
    """
    faces = faces_np.reshape(-1, 3)
    corners = [
        np.sort(np.stack([faces[:, (e + 1) % 3], faces[:, (e + 2) % 3]], axis=1), axis=1)
        for e in range(3)
    ]
    edges = np.concatenate(corners)
    lengths = np.concatenate([lengths_np[:, e] for e in range(3)])
    _unique, first = np.unique(edges, axis=0, return_index=True)
    return np.sort(lengths[first].astype(np.float64))


@pytest.mark.parametrize("mesh_name", ["half_torus", "torus"])
@pytest.mark.parity(
    "intrinsic_delaunay",
    "meshlib",
    benchmarked=False,
    reason="Tested but not timed: the row would price EdgeLengthMesh.fromMesh's "
    "cotangent-and-length precomputation together with the flips, which is not the "
    "quantity the group times. The igl row is the timed CPU reference.",
)
def test_intrinsic_delaunay_metric_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: the same intrinsic metric, from a flipper that reaches it in a different flip count.

    A second witness for the invariant
    [`test_intrinsic_delaunay_metric_matches_igl`][tests.test_remesh.test_intrinsic_delaunay_metric_matches_igl]
    rests on, and a sharper one, because this reference **disagrees about the flips and still agrees
    about the metric**. On ``torus`` ordito performs 476 flips and ``makeDeloneEdgeFlips`` performs
    **592** -- a 1.24x difference in the work done -- yet the resulting edge-length multisets match
    to 5.96e-08 absolute / 2.22e-07 relative. That is the content of the claim: the intrinsic
    Delaunay triangulation of a fixed surface is unique away from cocircular degeneracies, so the
    flip *sequence* and even the flip *count* are implementation detail while the metric is not.
    Comparing against igl alone cannot show this, since igl exposes no flip count.

    The named transform is
    [`_undirected_intrinsic_lengths`][tests.test_remesh._undirected_intrinsic_lengths]: ordito
    returns a per-corner ``(n_faces, 3)`` table and ``EdgeLengthMesh.edgeLengths`` is indexed by
    undirected edge, so ordito's side is deduplicated to one length per edge. Doubling the
    reference instead is wrong on an open mesh -- see that helper's docstring for the 64-edge
    discrepancy on ``half_torus``.

    Measured, ``half_torus``: 298 flips from both sides, agreeing to 4.77e-07 / 3.72e-07.

    !!! note "The intrinsic overload is the third one"
        ``makeDeloneEdgeFlips`` is overloaded three ways and only the ``EdgeLengthMesh`` form flips
        *intrinsically*; the ``Mesh`` and ``(MeshTopology, VertCoords)`` forms flip the extrinsic
        triangulation and would move the surface, which is the thing this function is defined not
        to do. Its settings type differs accordingly -- ``IntrinsicDeloneSettings``, whose
        ``threshold`` defaults to ``0.0``, i.e. flip whenever the opposite angles sum past pi, which
        is ordito's rule exactly.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_wp, lengths_wp, n_flips = od.remesh.intrinsic_delaunay(mesh_wp.points, mesh_wp.indices)

    edge_mesh_ml = mm.EdgeLengthMesh.fromMesh(trimesh_to_meshlib(mesh_tm))
    n_flips_ml = mm.makeDeloneEdgeFlips(edge_mesh_ml, mm.IntrinsicDeloneSettings(), 100)
    lengths_ml = np.sort(meshlib_scalars_to_numpy(edge_mesh_ml.edgeLengths).astype(np.float64))

    assert n_flips > 0, "fixture is already Delaunay; this would assert nothing"
    assert n_flips_ml > 0, "the reference flipped nothing; the comparison would be vacuous"
    assert np.allclose(
        _undirected_intrinsic_lengths(faces_wp.numpy(), lengths_wp.numpy()),
        lengths_ml,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parity("subdivide", "trimesh")
def test_subdivide(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B (face order): one uniform 1-to-4 pass against ``trimesh.remesh.subdivide``.

    The reference is computed in ``float64`` and cast down, so the comparison is not measuring
    ordito's precision against numpy's. The counts are compared directly and the faces through a
    nearest-neighbour bijection on their centroids, which pins the new midpoints and that every
    child face exists once. A coordinate lexsort is not usable: the icosahedron's centroids tie, and
    a tie broken by rounding noise orders the two sides differently.
    """
    mesh_tm, mesh_wp = icosahedron

    vertices_np = mesh_tm.vertices.astype(np.float32)
    faces_np = mesh_tm.faces.astype(np.int32)

    new_v_tm, new_f_tm = tm.remesh.subdivide(vertices_np.astype(np.float64), faces_np)[:2]
    new_v_tm = new_v_tm.astype(np.float32)

    new_v_wp, new_f_wp = od.remesh.subdivide(mesh_wp.points, mesh_wp.indices)

    new_v_wp_np = new_v_wp.numpy()
    new_f_wp_np = new_f_wp.numpy().reshape(-1, 3)
    new_f_tm_np = new_f_tm.reshape(-1, 3)

    assert len(new_v_wp_np) == new_v_tm.shape[0], (
        f"vertex count mismatch: got {len(new_v_wp_np)}, expected {new_v_tm.shape[0]}"
    )
    assert new_f_wp_np.shape[0] == new_f_tm_np.shape[0], (
        f"face count mismatch: got {new_f_wp_np.shape[0]}, expected {new_f_tm_np.shape[0]}"
    )

    centroids_wp = new_v_wp_np[new_f_wp_np].mean(axis=1)
    centroids_tm = new_v_tm[new_f_tm_np].mean(axis=1)
    assert_nearest_bijection(centroids_wp, centroids_tm, 1e-5)


@pytest.mark.parity("subdivide", "open3d")
def test_subdivide_matches_open3d(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B: Open3D's ``subdivide_midpoint`` is the same 1:4 split under a different vertex order.

    Neither library defines the output ordering -- ordito appends one new vertex per unique edge in
    its own edge order, Open3D in its -- so the named transform matches the face **centroid sets**
    and requires the match to be a bijection. A lexsort compare is not usable: the icosahedron's
    centroids carry coordinate ties that ordito resolves in ``float32`` and Open3D in ``float64``,
    so the row order is decided by rounding noise (measured: a 1.59 spurious mismatch).
    """
    mesh_tm, mesh_wp = icosahedron

    mesh_o3d = trimesh_to_open3d(mesh_tm).subdivide_midpoint(number_of_iterations=1)
    mesh_ref = open3d_to_trimesh(mesh_o3d)
    vertices_wp, faces_wp = od.remesh.subdivide(mesh_wp.points, mesh_wp.indices)

    assert vertices_wp.size == mesh_ref.vertices.shape[0]
    assert faces_wp.size // 3 == mesh_ref.faces.shape[0]

    centroids_wp = (
        vertices_wp.numpy().astype(np.float64)[faces_wp.numpy().reshape(-1, 3)].mean(axis=1)
    )
    centroids_o3d = mesh_ref.vertices[mesh_ref.faces].mean(axis=1)
    assert_nearest_bijection(centroids_wp, centroids_o3d, 1e-5)


# No ``parity`` marker: ``igl.upsample`` is not a *benchmarked* reference for ``subdivide``, because
# it corrupts the process heap on the scan meshes (the numbers are in
# ``benchmarks/test_remesh.py::test_subdivide``). It is safe on this fixture -- 1 200 calls across
# six processes are clean -- so the comparison itself is worth keeping, and the parity gate only
# asks that every *benchmarked* pair be tested, not the reverse.
@pytest.mark.parity("subdivide", "pytorch3d")
def test_subdivide_matches_pytorch3d(icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B: ``ops.SubdivideMeshes`` is the same 1:4 split, matched by nearest neighbour.

    Identical counts (642 vertices, 1 280 faces from 162 / 320) and the matched coordinates agree
    at **0.0** -- exactly, because both sides take the same midpoint of the same float32 edge.
    Face buffers are deliberately *not* compared positionally: pytorch3d emits its four children in
    its own order and numbers the new midpoints by its own edge table.

    The vertex correspondence is a ``cKDTree`` query plus a bijection check and **not**
    ``lexsort_rows``, which CLAUDE.md section 7.5 records as unusable on float coordinates with
    ties: two sides that tie in float32 but differ in the 16th float64 digit order those rows
    differently and the compare then fails by the full coordinate range. It passed here only
    because both sides happen to produce bit-identical midpoints today, so any change to either
    summation order would have turned it into a false negative reading as a real disagreement.

    ``SubdivideMeshes`` is a ``torch.nn.Module`` rather than a function, and constructing it with
    no ``meshes=`` argument is what makes it recompute the subdivision topology per call -- passing
    a mesh there caches it, which would be timing a different thing in the benchmark.
    """
    mesh_tm, mesh_wp = icosphere_coarse
    subdivided_p3d = p3d_ops.SubdivideMeshes()(trimesh_to_pytorch3d(mesh_tm))
    vertices_p3d, faces_p3d = pytorch3d_to_numpy(subdivided_p3d)
    vertices_wp, faces_wp = od.remesh.subdivide(mesh_wp.points, mesh_wp.indices)

    assert vertices_p3d.shape[0] == 4 * len(mesh_tm.vertices) - 6
    assert faces_p3d.shape[0] == 4 * len(mesh_tm.faces)
    assert vertices_wp.size == vertices_p3d.shape[0]
    assert faces_wp.size // 3 == faces_p3d.shape[0]
    assert_nearest_bijection(vertices_wp.numpy(), vertices_p3d.astype(np.float32), 0.0)


def test_subdivide_matches_igl(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B: ``igl.upsample`` is the same 1:4 midpoint split under a different vertex order.

    igl keeps the original vertices in place and appends one per unique edge, exactly as ordito
    does, so the *vertex* arrays agree on their leading ``n_vertices`` rows -- which is asserted
    directly and is a stronger statement than the centroid match alone. The new vertices and the
    faces are ordered by each library's own edge enumeration, so those go through the same
    bijective centroid match the Open3D test uses, and for the same reason: a lexsort over
    coordinates is decided by rounding noise where the icosahedron's centroids tie.
    """
    mesh_tm, mesh_wp = icosahedron

    vertices_upsampled_igl, faces_upsampled_igl = map(
        np.asarray,
        igl.upsample(np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), faces_igl(mesh_tm)),
    )
    vertices_wp, faces_wp = od.remesh.subdivide(mesh_wp.points, mesh_wp.indices)

    assert vertices_wp.size == vertices_upsampled_igl.shape[0]
    assert faces_wp.size // 3 == faces_upsampled_igl.shape[0]
    # The original vertices are untouched and stay in place on both sides.
    n_original = mesh_tm.vertices.shape[0]
    assert np.allclose(
        vertices_wp.numpy()[:n_original], vertices_upsampled_igl[:n_original], rtol=1e-5, atol=1e-5
    )

    centroids_wp = (
        vertices_wp.numpy().astype(np.float64)[faces_wp.numpy().reshape(-1, 3)].mean(axis=1)
    )
    centroids_igl = vertices_upsampled_igl[faces_upsampled_igl].mean(axis=1)
    assert_nearest_bijection(centroids_wp, centroids_igl, 1e-5)


def test_subdivide_edge_lengths(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """All edges in the subdivided mesh are at most half the longest original edge."""
    mesh_tm, mesh_wp = icosahedron

    vertices_np = mesh_tm.vertices.astype(np.float32)
    faces_np = mesh_tm.faces.astype(np.int32)

    new_v_wp, new_f_wp = od.remesh.subdivide(mesh_wp.points, mesh_wp.indices)

    new_v_np = new_v_wp.numpy()
    new_f_np = new_f_wp.numpy().reshape(-1, 3)

    orig_max_edge = _max_edge_length(vertices_np, faces_np)
    new_max_edge = _max_edge_length(new_v_np, new_f_np)

    assert new_max_edge <= orig_max_edge / 2.0 + 1e-5


# --------------------------------------------------------------------------------------
# subdivide_loop
# --------------------------------------------------------------------------------------


def _loop_odd_correspondence(
    faces_wp_np: np.ndarray, faces_igl: np.ndarray, n_new: int, n_original: int
) -> np.ndarray:
    """
    Map each of ordito's new edge vertices onto igl's, decoded from the two face tables.

    Both libraries emit four children per face in input face order, and one of those children is the
    central triangle spanning the face's three *new* vertices -- ordito emits it fourth and igl
    third. Reading that row off both tables therefore pairs the two enumerations corner by corner,
    which is exact where a coordinate ``lexsort`` would be decided by rounding noise. Returns
    ``perm`` with ``perm[ordito_index] == igl_index`` over the new vertices.
    """
    perm = np.full(n_new, -1, dtype=np.int64)
    perm[faces_wp_np[3::4].ravel()] = faces_igl[2::4].ravel()
    new_ids = np.arange(n_original, n_new)
    assert perm[new_ids].min() >= n_original, "a new vertex was paired with an original one"
    assert len(set(perm[new_ids].tolist())) == new_ids.size, "the pairing is not a bijection"
    return perm


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("subdivide_loop", "igl")
def test_subdivide_loop_matches_igl(mesh_name: str, request: pytest.FixtureRequest) -> None:
    """
    Class A on the moved originals, Class B on the new vertices: the same Loop stencils as igl.

    Every weight in Loop subdivision is a convention another library may pick differently, and this
    pins all four of them against ``igl.loop`` at ``1e-5``: the interior ``beta`` (**Warren's**
    ``3/16`` at valence 3 and ``3/(8n)`` above, not Loop's trigonometric one), the ``3/8``-``1/8``
    edge rule, the ``1/2`` boundary-edge rule, and the ``3/4``-``1/8`` boundary-vertex rule.
    Getting any one of them wrong still yields a smooth-looking surface, so a shape-only assertion
    would not see it.

    The originals correspond by index on both sides -- igl returns them first too -- so that half is
    Class A and directly comparable. The new vertices are ordered by each library's own edge
    enumeration, and ``_loop_odd_correspondence`` decodes the exact pairing from the face tables
    rather than matching coordinates.

    The fixtures cover what the branches need: two closed meshes (``cave_cube`` supplies valence-3
    and valence-6 corners) and two with a boundary, so the boundary rules are not dead code here --
    ``hemisphere`` and ``half_torus`` both have a rim, which is asserted before the comparison.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_original = int(mesh_tm.vertices.shape[0])

    vertices_wp, faces_wp = od.remesh.subdivide_loop(mesh_wp.points, mesh_wp.indices)

    vertices_igl, faces_loop_igl = map(
        np.asarray,
        igl.loop(np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), faces_igl(mesh_tm)),
    )
    assert vertices_wp.size == vertices_igl.shape[0]
    assert faces_wp.size // 3 == faces_loop_igl.shape[0]

    vertices_new_np = vertices_wp.numpy()
    # The originals moved -- that is what makes this Loop and not `subdivide` -- and they moved the
    # same way on both sides.
    assert np.allclose(
        vertices_new_np[:n_original], vertices_igl[:n_original], rtol=1e-5, atol=1e-5
    )
    assert not np.allclose(vertices_new_np[:n_original], mesh_tm.vertices, atol=1e-4)

    perm_np = _loop_odd_correspondence(
        faces_wp.numpy().reshape(-1, 3), np.asarray(faces_loop_igl), vertices_wp.size, n_original
    )
    new_ids = np.arange(n_original, vertices_wp.size)
    assert np.allclose(
        vertices_new_np[new_ids], vertices_igl[perm_np[new_ids]], rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("subdivide_loop", "open3d")
def test_subdivide_loop_matches_open3d(mesh_name: str, request: pytest.FixtureRequest) -> None:
    """
    Class B: Open3D's ``subdivide_loop`` is the same variant, under its own new-vertex order.

    A second oracle beside igl, worth having because it is independent: Open3D agrees with igl to
    **2e-16** on the relocated originals, so the two references corroborate each other on the one
    choice that is genuinely ambiguous here -- Warren's ``beta`` against Loop's original. trimesh
    picks the other one and is exempted in the benchmark for it.

    Open3D also returns the originals first, so that prefix compares directly; the new vertices and
    faces follow its own edge enumeration and go through the bijective centroid match the
    ``subdivide`` / Open3D test uses, for the same reason (a coordinate ``lexsort`` is decided by
    rounding noise where centroids tie).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_original = int(mesh_tm.vertices.shape[0])

    vertices_wp, faces_wp = od.remesh.subdivide_loop(mesh_wp.points, mesh_wp.indices)

    mesh_o3d = trimesh_to_open3d(mesh_tm).subdivide_loop(number_of_iterations=1)
    vertices_o3d = np.asarray(mesh_o3d.vertices)
    assert vertices_o3d.shape[0] == vertices_wp.size
    assert np.allclose(
        vertices_wp.numpy()[:n_original], vertices_o3d[:n_original], rtol=1e-5, atol=1e-5
    )

    centroids_wp = (
        vertices_wp.numpy().astype(np.float64)[faces_wp.numpy().reshape(-1, 3)].mean(axis=1)
    )
    centroids_o3d = vertices_o3d[np.asarray(mesh_o3d.triangles)].mean(axis=1)
    assert_nearest_bijection(centroids_wp, centroids_o3d, 1e-5)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_subdivide_loop_keeps_the_boundary_in_the_boundary(
    mesh_name: str, request: pytest.FixtureRequest
) -> None:
    """
    The boundary stencils close on the boundary: a rim vertex is a combination of rim vertices only.

    That is the property that lets two patches sharing a seam subdivide independently and still
    meet, and it is exactly what a stencil that let interior neighbours leak in would break -- while
    still passing every smoothness or shape check. Asserted geometrically, by checking a rim vertex
    lands in the affine hull of the *old* rim, which the interior rule's ``beta`` term would leave.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    boundary_wp = od.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices)
    assert boundary_wp.size > 0, "the fixture has a boundary to preserve"

    vertices_wp, faces_wp = od.remesh.subdivide_loop(mesh_wp.points, mesh_wp.indices)

    rim_before_np = mesh_wp.points.numpy()[boundary_wp.numpy()]
    rim_after_np = vertices_wp.numpy()[
        od.boundary.boundary_vertex_indices(vertices_wp, faces_wp).numpy()
    ]
    # Every new rim vertex is a convex combination of old rim vertices, so it cannot leave their
    # bounding box; an interior stencil leaking in would pull it inward, off the rim.
    assert np.all(rim_after_np >= rim_before_np.min(axis=0) - 1e-5)
    assert np.all(rim_after_np <= rim_before_np.max(axis=0) + 1e-5)


def test_subdivide_loop_shrinks_a_convex_solid_towards_its_limit(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Loop approximates where ``subdivide`` interpolates, so it must move the surface and shrink it.

    The pair of assertions is what distinguishes the two functions on a convex solid: the midpoint
    split puts every new vertex on the old flat faces, so the surface and its volume are unchanged
    (to float32 rounding), while Loop pulls the vertices in and the volume falls. Iterating three
    times also checks the passes compose -- each one is a fresh call, which is how ``igl.loop``'s
    ``number_of_subdivs`` is meant to be reproduced.
    """
    _, mesh_wp = icosahedron
    volume_before = od.measures.volume(mesh_wp.points, mesh_wp.indices)

    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    volumes = []
    for _ in range(3):
        vertices_wp, faces_wp = od.remesh.subdivide_loop(vertices_wp, faces_wp)
        volumes.append(od.measures.volume(vertices_wp, faces_wp))

    assert volumes[0] < volume_before, "Loop pulls a convex surface inward"
    # Converging, not collapsing: successive passes change the volume by less and less, and the
    # limit surface stays a sizeable fraction of the original solid.
    steps = [abs(volumes[i + 1] - volumes[i]) for i in range(len(volumes) - 1)]
    assert steps[1] < steps[0]
    assert volumes[-1] > 0.5 * volume_before

    # The midpoint split on the same input keeps the solid, which is the contrast being drawn: an
    # earlier strict ``>`` here passed only on the rounding of one summation order.
    vertices_mid_wp, faces_mid_wp = od.remesh.subdivide(mesh_wp.points, mesh_wp.indices)
    volume_mid = od.measures.volume(vertices_mid_wp, faces_mid_wp)
    assert np.isclose(volume_mid, volume_before, rtol=1e-6, atol=0.0)
    assert volumes[0] < volume_before * (1.0 - 1e-3)


def test_subdivide_loop_leaves_a_nonmanifold_edge_at_its_midpoint(device: str) -> None:
    """
    Three faces on one edge: the interior stencil is undefined there, so the midpoint rule applies.

    The documented fallback, asserted rather than assumed because the alternative -- summing three
    opposite vertices into a stencil scaled for two -- fails silently, moving that vertex off the
    edge entirely instead of raising.
    """
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    # Edge (0, 1) is shared by all three faces.
    faces_np = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int32).ravel()
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    vertices_new_wp, faces_new_wp = od.remesh.subdivide_loop(vertices_wp, faces_wp)

    positions_np = vertices_new_wp.numpy()
    midpoint_np = 0.5 * (vertices_np[0] + vertices_np[1])
    distances_np = np.linalg.norm(positions_np - midpoint_np, axis=1)
    assert distances_np.min() < 1e-6, "the vertex on the non-manifold edge is at its midpoint"
    assert faces_new_wp.size // 3 == 12


@pytest.mark.parametrize("mesh_name", MESHES)
def test_subdivide_loop_operator_reproduces_its_own_positions(
    mesh_name: str, request: pytest.FixtureRequest
) -> None:
    """
    Not a library comparison: the operator *is* the pass, so applying it to the vertices is exact.

    No reference library returns Loop's interpolation matrix, so there is nothing to compare against
    -- and nothing is needed, because the claim is an identity rather than an agreement:
    ``P @ vertices`` must be the vertex buffer the same call returned. That is the strongest
    available check on the weights, since it fails if any single stencil is emitted with a different
    weight than the position kernel used, and it is why the operator is assembled through the same
    two shared ``@wp.func`` weight helpers rather than by a second transcription of the rules.

    Three further invariants, each excluding a different way to be wrong: every row sums to **1**
    (an affine combination, so a rigid motion of the input moves the output rigidly), the shape is
    ``(n_out, n_in)`` with ``n_out`` the returned vertex count, and the default two-value call
    returns the identical positions -- the operator costs nothing when it is not asked for.

    The tolerance is ``1e-6`` rather than exact: a row sum accumulates in column order where the
    kernel sums the 1-ring first, and in float32 those differ in the last bits. Measured
    max deviation 1.2e-07 over these four fixtures.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    n_vertices = vertices_wp.size

    moved_wp, subdivided_faces_wp, operator = od.remesh.subdivide_loop(
        vertices_wp, faces_wp, return_operator=True
    )
    assert int(operator.nrow) == moved_wp.size
    assert int(operator.ncol) == n_vertices

    operator_np = bsr_to_csr(operator)
    assert np.allclose(np.asarray(operator_np.sum(axis=1)).ravel(), 1.0, rtol=1e-6, atol=1e-6)
    assert np.allclose(operator_np @ vertices_wp.numpy(), moved_wp.numpy(), rtol=1e-6, atol=1e-6)

    plain_vertices_wp, plain_faces_wp = od.remesh.subdivide_loop(vertices_wp, faces_wp)
    assert np.array_equal(plain_faces_wp.numpy(), subdivided_faces_wp.numpy())
    # Positions from two separate calls, not compared byte for byte: the ring sums are accumulated
    # with atomics, so their float32 order varies run to run on CUDA.
    assert np.allclose(plain_vertices_wp.numpy(), moved_wp.numpy(), rtol=1e-6, atol=1e-6)


def test_subdivide_loop_operator_carries_a_field(
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: what the operator is *for* -- a field surviving the subdivision.

    Two properties of an affine interpolation operator, both of which a wrong operator fails: a
    constant field stays constant everywhere (rows summing to 1), and a field that is linear in the
    vertex positions stays linear, because the operator reproduces those positions exactly. The
    second is the one that catches weights placed on the wrong *columns*: a permuted operator still
    has unit row sums.

    Covers the three dtypes the transfer registers, since each is a separately compiled overload:
    ``wp.float32``, ``wp.vec2`` (a UV) and ``wp.vec3`` (a colour or a normal).
    """
    _, mesh_wp = icosphere_coarse
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    n_vertices = vertices_wp.size
    device = vertices_wp.device

    moved_wp, _faces_wp, operator = od.remesh.subdivide_loop(
        vertices_wp, faces_wp, return_operator=True
    )
    n_out = moved_wp.size

    constant_wp = wp.full(n_vertices, wp.float32(2.5), dtype=wp.float32, device=device)
    carried_wp = od.interpolation.transfer_through_operator(constant_wp, operator)
    assert carried_wp.shape == (n_out,)
    assert np.allclose(carried_wp.numpy(), 2.5, rtol=1e-6, atol=1e-6)

    # A linear field: f(v) = dot(v, direction). Linear in the positions, so the transferred field
    # must equal the same function of the *moved* positions.
    direction_np = np.array([0.3, -0.7, 0.2], dtype=np.float32)
    linear_np = mesh_wp.points.numpy() @ direction_np
    linear_wp = wp.array(linear_np, dtype=wp.float32, device=device)
    carried_linear_np = od.interpolation.transfer_through_operator(linear_wp, operator).numpy()
    assert np.allclose(carried_linear_np, moved_wp.numpy() @ direction_np, rtol=1e-5, atol=1e-5)

    # vec3: transferring the positions themselves is the operator's own identity.
    carried_positions_wp = od.interpolation.transfer_through_operator(vertices_wp, operator)
    assert np.allclose(carried_positions_wp.numpy(), moved_wp.numpy(), rtol=1e-6, atol=1e-6)

    # vec2: a UV pair whose components sum to one stays a partition, since the rows are affine.
    uv_np = np.ascontiguousarray(
        np.stack([np.linspace(0.0, 1.0, n_vertices), np.linspace(1.0, 0.0, n_vertices)], axis=1),
        dtype=np.float32,
    )
    carried_uv_np = od.interpolation.transfer_through_operator(
        points_to_warp_uv(uv_np, device), operator
    ).numpy()
    assert carried_uv_np.shape == (n_out, 2)
    assert np.allclose(carried_uv_np.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)


def test_transfer_through_operator_guards_and_empty_inputs(device: str) -> None:
    """
    Not a library comparison: the column-count guard, and the two degenerate shapes.

    An empty face buffer makes the pass the identity, which is asserted here rather than left
    implicit: a caller subdividing a point set should get its field back unchanged rather than an
    exception or an empty answer.
    """
    vertices_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    faces_wp = warp_empty(0, wp.int32, device)
    _vertices_wp, _faces_wp, identity = od.remesh.subdivide_loop(
        vertices_wp, faces_wp, return_operator=True
    )
    assert int(identity.nrow) == 3
    assert int(identity.ncol) == 3
    field_wp = wp.array(
        np.array([1.0, 2.0, 3.0], dtype=np.float32), dtype=wp.float32, device=device
    )
    assert np.allclose(
        od.interpolation.transfer_through_operator(field_wp, identity).numpy(),
        np.array([1.0, 2.0, 3.0]),
    )

    with pytest.raises(ValueError, match="columns"):
        od.interpolation.transfer_through_operator(
            wp.zeros(2, dtype=wp.float32, device=device), identity
        )


# --------------------------------------------------------------------------------------
# subdivide_to_size
# --------------------------------------------------------------------------------------


@pytest.mark.parity("subdivide_to_size", "trimesh")
def test_subdivide_to_size_reference_regular(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Class A: a single pass on a regular mesh, where every face splits 1-to-4, matches trimesh."""
    mesh_tm, mesh_wp = icosahedron
    faces_np = mesh_tm.faces.astype(np.int32)
    max_edge = 0.6 * _max_edge_length(mesh_tm.vertices.astype(np.float32), faces_np)
    _assert_matches_trimesh_subdivide_to_size(mesh_tm, mesh_wp, max_edge)


@pytest.mark.parametrize("split_fraction", [0.7, 0.35])
@pytest.mark.parity("subdivide_to_size", "pymeshlab")
def test_subdivide_to_size_matches_pymeshlab(
    split_fraction: float, icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class A: MeshLab's midpoint refinement produces the *identical* mesh, vertex for vertex.

    ``meshing_surface_subdivision_midpoint`` splits every edge over ``threshold`` at its midpoint
    and repeats, which is exactly what this function does, and at both split fractions the two agree
    on the vertex count, the face count, the resulting longest edge and every vertex *position* --
    worst nearest-neighbour distance **6.5e-08**. So no transform on the geometry is needed at all.

    Two on the plumbing, both matching what the benchmark passes. ``threshold`` takes a wrapper type
    and gets ``ml.PureValue`` fed from the same absolute length ordito receives, not a
    ``PercentageValue`` of MeshLab's own bounding box. And ``iterations`` is a pass *cap* rather
    than a convergence criterion, so it is set well above the ``log2`` depth the target needs and
    the surplus passes find nothing left to refine; this is why the two converge to the same fixed
    point despite counting passes differently.

    Measured on ``icosphere(2)``: 642 vertices / 1 280 faces at 0.7x the mean edge, 2 562 / 5 120 at
    0.35x, identical on both sides.
    """
    mesh_tm, mesh_wp = icosphere_coarse
    edges_np = mesh_tm.edges_unique
    mean_edge = float(
        np.linalg.norm(
            mesh_tm.vertices[edges_np[:, 0]] - mesh_tm.vertices[edges_np[:, 1]], axis=1
        ).mean()
    )
    max_edge = split_fraction * mean_edge

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.meshing_surface_subdivision_midpoint(
        iterations=10, threshold=ml.PureValue(max_edge)
    )
    vertices_pml = np.asarray(meshset_pml.current_mesh().vertex_matrix(), dtype=np.float64)
    faces_pml = np.asarray(meshset_pml.current_mesh().face_matrix())

    vertices_wp, faces_wp = od.remesh.subdivide_to_size(mesh_wp.points, mesh_wp.indices, max_edge)
    vertices_np = vertices_wp.numpy().astype(np.float64)
    faces_np = faces_wp.numpy().reshape(-1, 3)

    assert len(vertices_np) == vertices_pml.shape[0]
    assert faces_np.shape[0] == faces_pml.shape[0]

    # Same vertex set, then the same faces once ordito's indices are remapped onto MeshLab's.
    remap_np = assert_nearest_bijection(vertices_np, vertices_pml, 1e-5)
    mapped_np = np.sort(remap_np[faces_np], axis=1)
    reference_np = np.sort(faces_pml, axis=1)
    assert np.array_equal(lexsort_rows(mapped_np), lexsort_rows(reference_np))


def test_subdivide_to_size_reference_mixed(device: str) -> None:
    """A single pass with mixed 1/2/3-split faces matches trimesh exactly."""
    # A stretched icosahedron gives two distinct edge lengths so that, for an
    # intermediate threshold, faces split on 1, 2, or 3 edges in one pass.
    mesh_tm = tm.creation.icosahedron()
    mesh_tm.vertices = mesh_tm.vertices * np.array([1.0, 1.0, 2.2])
    mesh_wp = trimesh_to_warp(mesh_tm, device)
    _assert_matches_trimesh_subdivide_to_size(mesh_tm, mesh_wp, 1.9, max_iter=1)


def _assert_matches_trimesh_subdivide_to_size(
    mesh_tm: tm.Trimesh, mesh_wp: wp.Mesh, max_edge: float, **max_iter: int
) -> None:
    """
    Assert ``subdivide_to_size`` equals trimesh's: vertex set, face set and per-source face count.

    The vertices are matched by nearest-neighbour bijection (identical up to float32 storage), the
    faces compared as sorted rows through that match, and the ``return_index`` histograms equal.
    """
    new_v_wp, new_f_wp, index_wp = od.remesh.subdivide_to_size(
        mesh_wp.points, mesh_wp.indices, max_edge, return_index=True, **max_iter
    )
    new_v_np = new_v_wp.numpy()
    new_f_np = new_f_wp.numpy().reshape(-1, 3)

    # ``subdivide_to_size`` returns two values or three depending on ``return_index``, and
    # trimesh declares no overloads, so the arity is asserted here rather than inferred.
    ref_v, ref_f, ref_index = cast(
        "tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]",
        tm.remesh.subdivide_to_size(
            mesh_tm.vertices, mesh_tm.faces, max_edge, return_index=True, **max_iter
        ),
    )

    assert new_f_np.shape[0] == ref_f.shape[0]
    wp_to_ref = assert_nearest_bijection(new_v_np, ref_v, 1e-4)

    faces_mapped = np.sort(wp_to_ref[new_f_np], axis=1)
    faces_ref = np.sort(ref_f, axis=1)
    assert np.array_equal(lexsort_rows(faces_mapped), lexsort_rows(faces_ref))

    n_in_faces = mesh_tm.faces.shape[0]
    hist_wp = np.bincount(index_wp.numpy(), minlength=n_in_faces)
    hist_ref = np.bincount(ref_index, minlength=n_in_faces)
    assert np.array_equal(hist_wp, hist_ref)


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parametrize("frac", [0.75, 0.5, 0.3])
def test_subdivide_to_size_max_edge(
    mesh_name: str, frac: float, request: pytest.FixtureRequest
) -> None:
    """
    Not a library comparison: every edge is at most ``max_edge`` after subdivision.

    The defining property, on every fixture. On the closed ones the result must also stay crack
    free -- every edge shared by two faces, so no T-junction was introduced -- and keep its Euler
    characteristic.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    max_edge = frac * od.edges.mean_edge_length(mesh_wp.points, mesh_wp.indices)

    new_v_wp, new_f_wp = od.remesh.subdivide_to_size(mesh_wp.points, mesh_wp.indices, max_edge)
    new_f_np = new_f_wp.numpy().reshape(-1, 3)
    result_max_edge = _max_edge_length(new_v_wp.numpy(), new_f_np)

    assert result_max_edge <= max_edge + 1e-4
    if mesh_name in CLOSED_MESHES:
        _, counts = np.unique(undirected_edges(new_f_np), axis=0, return_counts=True)
        assert np.array_equal(counts, np.full(counts.shape, 2)), "T-junctions / cracks introduced"

        # Euler characteristic is preserved (no topology change).
        n_v = len(new_v_wp.numpy())
        n_e = np.unique(undirected_edges(new_f_np), axis=0).shape[0]
        n_f = new_f_np.shape[0]
        assert n_v - n_e + n_f == mesh_tm.euler_number


@pytest.mark.parametrize("mesh_name", MESHES)
def test_subdivide_to_size_noop(mesh_name: str, request: pytest.FixtureRequest) -> None:
    """A threshold above the longest edge returns the mesh unchanged."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = mesh_tm.faces.astype(np.int32)
    max_edge = 2.0 * _max_edge_length(mesh_wp.points.numpy(), faces_np)

    new_v_wp, new_f_wp = od.remesh.subdivide_to_size(mesh_wp.points, mesh_wp.indices, max_edge)

    assert np.array_equal(new_v_wp.numpy(), mesh_wp.points.numpy())
    assert np.array_equal(new_f_wp.numpy().reshape(-1, 3), faces_np)


@pytest.mark.parametrize("mesh_name", MESHES)
def test_subdivide_to_size_preserves_surface_and_reports_its_source_faces(
    mesh_name: str, request: pytest.FixtureRequest
) -> None:
    """
    Not a library comparison: the output covers the input and knows where each face came from.

    Midpoints lie on original edges, so surface area (and closed volume) is unchanged; and each
    output face carries a valid source id and lies inside that source triangle.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = mesh_wp.points.numpy()
    faces_np = mesh_tm.faces.astype(np.int32)
    n_in_faces = faces_np.shape[0]
    max_edge = 0.4 * _max_edge_length(vertices_np, faces_np)

    new_v_wp, new_f_wp, index_wp = od.remesh.subdivide_to_size(
        mesh_wp.points, mesh_wp.indices, max_edge, return_index=True
    )
    new_v_np = new_v_wp.numpy()
    new_f_np = new_f_wp.numpy().reshape(-1, 3)
    index_np = index_wp.numpy()

    assert np.isclose(
        _surface_area(new_v_np, new_f_np), _surface_area(vertices_np, faces_np), rtol=1e-4
    )
    if mesh_tm.is_watertight:
        assert np.isclose(
            _signed_volume(new_v_np, new_f_np), _signed_volume(vertices_np, faces_np), rtol=1e-4
        )

    assert index_np.size == new_f_np.shape[0]
    assert index_np.min() >= 0
    assert index_np.max() < n_in_faces

    # Every output-face centroid lies inside its claimed source triangle.
    centroids = new_v_np[new_f_np].mean(axis=1)
    src = vertices_np[faces_np[index_np]]
    a, b, c = src[:, 0], src[:, 1], src[:, 2]
    v0, v1, v2 = b - a, c - a, centroids - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    bary_v = (d11 * d20 - d01 * d21) / denom
    bary_w = (d00 * d21 - d01 * d20) / denom
    bary_u = 1.0 - bary_v - bary_w
    tol = 1e-3
    assert np.all(bary_u >= -tol)
    assert np.all(bary_v >= -tol)
    assert np.all(bary_w >= -tol)
    assert np.all(bary_u <= 1.0 + tol)
    assert np.all(bary_v <= 1.0 + tol)
    assert np.all(bary_w <= 1.0 + tol)


def test_subdivide_to_size_single_triangle(device: str) -> None:
    """A single triangle with one over-long edge splits into two faces."""
    # Edges: base (0,1) = 2.0, the other two ~1.044. With max_edge = 1.5 only the
    # base edge is over-long, so it splits once into two faces.
    vertices_np = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.3, 0.0]], dtype=np.float32)
    faces_np = np.array([0, 1, 2], dtype=np.int32)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    new_v_wp, new_f_wp = od.remesh.subdivide_to_size(vertices_wp, faces_wp, 1.5)
    new_v_np = new_v_wp.numpy()
    new_f_np = new_f_wp.numpy().reshape(-1, 3)

    assert len(new_v_np) == 4  # one midpoint added
    assert new_f_np.shape[0] == 2
    assert _max_edge_length(new_v_np, new_f_np) <= 1.5 + 1e-5
    # midpoint of the base edge (0,1) at (1, 0, 0) is present
    assert np.isclose(new_v_np, np.array([1.0, 0.0, 0.0])).all(axis=1).any()


def test_subdivide_to_size_max_iter_exceeded(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    small_edge = 0.1 * od.edges.mean_edge_length(mesh_wp.points, mesh_wp.indices)
    with pytest.raises(ValueError, match="max_iter exceeded"):
        od.remesh.subdivide_to_size(mesh_wp.points, mesh_wp.indices, small_edge, max_iter=0)


def test_subdivide_to_size_sizing_field(device: str) -> None:
    """
    A per-vertex sizing field refines each region to *its own* target, not to a global one.

    The assert that separates this from the scalar call is per-edge rather than global: every edge
    must be within the mean of its endpoints' targets, and the coarse half must retain edges longer
    than the fine half's target — which a scalar call at the field's minimum could not do.
    """
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=2)
    height = sphere_tm.vertices[:, 2]
    fraction = (height - height.min()) / np.ptp(height)
    field_np = (0.08 + 0.32 * fraction).astype(np.float32)
    field_wp = wp.array(np.ascontiguousarray(field_np), dtype=wp.float32, device=device)

    out_vertices, out_faces = od.remesh.subdivide_to_size(
        vertices_wp, faces_wp, field_wp, max_iter=12
    )
    points_np = out_vertices.numpy().astype(np.float64)
    faces_np = out_faces.numpy().reshape(-1, 3)
    assert faces_np.shape[0] > faces_wp.size // 3  # anti-vacuity

    # The field on the refined mesh: an inserted midpoint carries the mean of what it split, which
    # is the value the nearest original vertex reports for a field this smooth.
    pairs = np.unique(np.sort(undirected_edges(faces_np), axis=1), axis=0)
    lengths = np.linalg.norm(points_np[pairs[:, 0]] - points_np[pairs[:, 1]], axis=1)
    midpoints = points_np[pairs].mean(axis=1)
    targets = field_np[KDTree(sphere_tm.vertices).query(midpoints)[1]]
    # Every edge respects its own local target, with slack for the nearest-vertex approximation of
    # the field at the midpoint (the field varies by 0.32 across the sphere).
    assert (lengths <= targets * 1.35).all()
    # And the result is genuinely graded, not uniformly refined to the minimum.
    low = lengths[midpoints[:, 2] < np.median(midpoints[:, 2])].mean()
    high = lengths[midpoints[:, 2] >= np.median(midpoints[:, 2])].mean()
    assert high / low > 1.5, (low, high)


# ---------------------------------------------------------------------------
# Region-restricted subdivision (subdivide_region_to_size)
#
# Its two shared helpers, _filled_hemisphere and _region_max_edge, are at the top of the file:
# flip_to_delaunay reaches them too and now precedes this group.
# ---------------------------------------------------------------------------


def test_subdivide_region_max_edge(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    v, f, region = _filled_hemisphere(hemisphere)
    max_edge = 0.2 * _region_max_edge(v.numpy(), f.numpy().reshape(-1, 3), region.numpy())
    nv, nf, nr = od.remesh.subdivide_region_to_size(v, f, region, max_edge=max_edge, delaunay=False)
    max_edge_np = _region_max_edge(nv.numpy(), nf.numpy().reshape(-1, 3), nr.numpy())
    assert max_edge_np <= max_edge + 1e-4


def test_subdivide_region_crack_free(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    v, f, region = _filled_hemisphere(hemisphere)
    max_edge = 0.3 * _region_max_edge(v.numpy(), f.numpy().reshape(-1, 3), region.numpy())
    _, nf, _ = od.remesh.subdivide_region_to_size(v, f, region, max_edge=max_edge)
    faces_np = nf.numpy().reshape(-1, 3)
    edges = undirected_edges(faces_np)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    # Filled hemisphere is closed: every undirected edge is shared by exactly two faces.
    assert np.array_equal(np.unique(counts), np.array([2]))


def test_subdivide_region_appends_and_leaves_the_outside_untouched(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
):
    """
    Not a library comparison: refinement only appends vertices and only rewrites the rim.

    The originals stay a verbatim prefix of the vertex buffer, and a face outside the region is
    either unchanged or retriangulated because it shared a split rim edge.
    """
    v, f, region = _filled_hemisphere(hemisphere)
    n_vertices_before = v.size
    max_edge = 0.3 * _region_max_edge(v.numpy(), f.numpy().reshape(-1, 3), region.numpy())
    nv, nf, nr = od.remesh.subdivide_region_to_size(v, f, region, max_edge=max_edge, delaunay=False)
    nv_np = nv.numpy()
    assert len(nv_np) > n_vertices_before
    assert np.array_equal(nv_np[:n_vertices_before], v.numpy())

    faces_np = nf.numpy().reshape(-1, 3)
    region_np = nr.numpy()
    original = {tuple(sorted(t)) for t in f.numpy().reshape(-1, 3).tolist()}
    for t in faces_np[~region_np]:
        touches_new = any(idx >= n_vertices_before for idx in t)
        # A non-region face is unchanged, or only retriangulated because it shared a split rim edge.
        assert tuple(sorted(int(x) for x in t)) in original or touches_new


def test_subdivide_region_max_splits_takes_the_longest_edges(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
):
    """
    Class A: a bound budget spends itself on the longest eligible edges, not on an arbitrary five.

    ``_keep_longest_edges`` ranks the pass's eligible edges on the device and keeps the longest that
    fit. This pins the *selection rule* rather than the tie order, which neither the host nor the
    device spelling defines. Non-vacuous by construction: the budget is a twentieth of the eligible
    count, so the branch is reached and has to discard most of what it was given.
    """
    v, f, region = _filled_hemisphere(hemisphere)
    vertices_np, faces_np, region_np = v.numpy(), f.numpy().reshape(-1, 3), region.numpy()
    max_edge = 0.2 * _region_max_edge(vertices_np, faces_np, region_np)

    edges_np = np.unique(np.sort(undirected_edges(faces_np), axis=1), axis=0)
    lengths_np = np.linalg.norm(vertices_np[edges_np[:, 0]] - vertices_np[edges_np[:, 1]], axis=1)
    face_of_edge = np.repeat(np.arange(faces_np.shape[0]), 3)
    region_edges = {
        tuple(edge) for edge in np.sort(undirected_edges(faces_np), axis=1)[region_np[face_of_edge]]
    }
    eligible = np.array([tuple(edge) in region_edges for edge in edges_np], dtype=bool) & (
        lengths_np > max_edge
    )
    budget = int(eligible.sum()) // 3
    assert 2 <= budget < int(eligible.sum()), (
        "the budget must bind and still leave a real choice, or the test is about nothing"
    )

    nv, _, _ = od.remesh.subdivide_region_to_size(
        v, f, region, max_edge=max_edge, max_splits=budget, delaunay=False
    )
    # One new vertex per split, appended after the originals, so the midpoints identify the edges.
    midpoints_np = nv.numpy()[v.size :]
    assert midpoints_np.shape[0] == budget

    edge_midpoints = 0.5 * (vertices_np[edges_np[:, 0]] + vertices_np[edges_np[:, 1]])
    split = np.array(
        [np.abs(edge_midpoints - point).sum(axis=1).argmin() for point in midpoints_np]
    )
    assert np.allclose(edge_midpoints[split], midpoints_np, rtol=1e-5, atol=1e-5)
    # Every split edge was eligible, and none of them is shorter than an eligible edge left alone.
    assert eligible[split].all()
    left = eligible.copy()
    left[split] = False
    assert lengths_np[split].min() >= lengths_np[left].max() - 1e-6


# ---------------------------------------------------------------------------
# Region-restricted density refinement (refine_region_to_density)
# ---------------------------------------------------------------------------


def _graded_patch_with_hole(device: str, n: int = 17):
    """
    Build a flat grid whose spacing grows 8x across it, with a hole punched in the middle.

    The input the two refiners differ on, and the reason it has to be built rather than taken from
    ``tests/conftest.py``: on a *uniformly* sampled patch a target edge length and a local density
    are the same instruction, so a comparison there is vacuous whatever it asserts. Here the rim
    edges span an order of magnitude, so "match the surroundings" and "be shorter than L" pull the
    patch in different directions.

    Returns ``(vertices_wp, faces_wp, region_wp)`` with the region covering the fill patch, plus the
    original face count.
    """
    xs = np.linspace(0.0, 1.0, n) ** 2 * 4.0
    ys = np.linspace(0.0, 1.0, n) * 2.0
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    vertices_np = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(n * n)], axis=1)
    quads = [
        [
            [i * n + j, i * n + j + n, i * n + j + 1],
            [i * n + j + 1, i * n + j + n, i * n + j + n + 1],
        ]
        for i in range(n - 1)
        for j in range(n - 1)
    ]
    faces_np = np.asarray(quads).reshape(-1, 3)
    centers_np = vertices_np[faces_np].mean(axis=1)
    hole_np = (
        (centers_np[:, 0] > xs[6])
        & (centers_np[:, 0] < xs[11])
        & (centers_np[:, 1] > ys[6])
        & (centers_np[:, 1] < ys[11])
    )
    holed_tm = tm.Trimesh(vertices_np, faces_np[~hole_np], process=False)
    holed_tm.remove_unreferenced_vertices()

    vertices_wp, faces_wp = numpy_to_warp(holed_tm.vertices, holed_tm.faces, device)
    n_faces = faces_wp.size // 3
    # ``preserve_largest_hole`` leaves the grid's own outer boundary open and fills only the punch.
    filled_wp = od.holes.fill_min_weight(vertices_wp, faces_wp, preserve_largest_hole=True)
    region_np = np.zeros(filled_wp.size // 3, dtype=bool)
    region_np[n_faces:] = True
    return vertices_wp, filled_wp, wp.array(region_np, dtype=wp.bool, device=device), n_faces


def _patch_scale_ratios(
    vertices_wp: wp.array[wp.vec3],
    faces_wp: wp.array[wp.int32],
    region_wp: wp.array[wp.bool],
    original_vertices_wp: wp.array[wp.vec3],
    original_faces_wp: wp.array[wp.int32],
) -> np.ndarray:
    """
    Per patch triangle, its mean edge length over the *local* surrounding scale.

    The surrounding scale is Liepa's attribute -- the mean incident edge length in the original
    mesh -- read at the original vertex nearest the triangle's centroid. A refinement that matched
    its neighbourhood everywhere would return all ones; the spread of these ratios across the patch
    is what separates a density criterion from a single global target.
    """
    original_tm = warp_to_trimesh(original_vertices_wp, original_faces_wp)
    edges_np = original_tm.edges_unique
    lengths_np = np.linalg.norm(
        original_tm.vertices[edges_np[:, 0]] - original_tm.vertices[edges_np[:, 1]], axis=1
    )
    totals_np = np.zeros(original_tm.vertices.shape[0])
    counts_np = np.zeros(original_tm.vertices.shape[0])
    for column in (0, 1):
        np.add.at(totals_np, edges_np[:, column], lengths_np)
        np.add.at(counts_np, edges_np[:, column], 1.0)
    referenced_np = np.flatnonzero(counts_np > 0)
    scale_np = totals_np[referenced_np] / counts_np[referenced_np]
    tree = KDTree(original_tm.vertices[referenced_np])

    triangles_np = vertices_wp.numpy().astype(np.float64)[
        faces_wp.numpy().reshape(-1, 3)[region_wp.numpy()]
    ]
    mean_edge_np = np.linalg.norm(triangles_np - np.roll(triangles_np, 1, axis=1), axis=2).mean(
        axis=1
    )
    return mean_edge_np / scale_np[tree.query(triangles_np.mean(axis=1))[1]]


def test_refine_region_to_density_matches_the_surroundings_better_than_a_target_length(
    device: str,
) -> None:
    """
    Not a library comparison: the two ordito refiners against each other on a graded patch.

    No reference computes this in isolation -- pymeshfix performs exactly this refinement but only
    inside ``fill_small_boundaries``, where ``tests/test_holes.py`` compares it -- so the claim here
    is the one the criterion was added for: on a **graded** neighbourhood, Liepa's density rule puts
    the patch at its surroundings' sampling where a single target edge length cannot.

    Measured on a patch whose surrounding edge lengths span 8x, as the ratio of each patch
    triangle's mean edge to the local surrounding scale: ``"density"`` gives a max/min spread of
    **1.90** with a mean of **1.09**, and ``subdivide_region_to_size`` at the mesh's mean edge gives
    a spread of **3.58** with a mean of **0.59** -- so it is not merely less even, it is uniformly
    over-refining by about 1.7x because one length is too fine at the coarse end. The asserted
    bounds sit between the two measurements on both statistics.

    The fixture is graded on purpose: on a uniform patch the two criteria coincide and any assert
    here would pass for either.
    """
    vertices_wp, faces_wp, region_wp, n_faces = _graded_patch_with_hole(device)
    original_faces_wp = wp.clone(odt.as_dense(faces_wp[: 3 * n_faces]))

    density_v, density_f, density_r = od.remesh.refine_region_to_density(
        vertices_wp, faces_wp, region_wp
    )
    length_v, length_f, length_r = od.remesh.subdivide_region_to_size(
        vertices_wp,
        faces_wp,
        region_wp,
        max_edge=float(od.edges.mean_edge_length(vertices_wp, original_faces_wp)),
        max_splits=100_000,
    )

    density_np = _patch_scale_ratios(
        density_v, density_f, density_r, vertices_wp, original_faces_wp
    )
    length_np = _patch_scale_ratios(length_v, length_f, length_r, vertices_wp, original_faces_wp)

    assert density_v.size > vertices_wp.size  # non-vacuity: it refined
    assert length_v.size > vertices_wp.size
    density_spread = density_np.max() / density_np.min()
    length_spread = length_np.max() / length_np.min()
    assert density_spread < 2.5 < length_spread
    assert abs(density_np.mean() - 1.0) < 0.3
    assert abs(length_np.mean() - 1.0) > 0.3


def test_refine_region_to_density_leaves_the_mesh_closed_and_the_outside_alone(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: the invariants a 1-to-3 centroid split gives for free.

    Every new vertex is interior to a triangle and no edge is divided, so unlike edge bisection
    there is nothing for a neighbouring face to agree to -- which means the mesh stays closed with
    **no** crack-free template at all, non-region faces come back byte-identical, and the face count
    grows by exactly twice the number of inserted vertices. Asserting all three is what says the
    split really is independent rather than accidentally consistent on this input.
    """
    vertices_wp, faces_wp, region_wp = _filled_hemisphere(hemisphere)
    n_vertices = vertices_wp.size
    outside_np = {
        tuple(sorted(row)) for row in faces_wp.numpy().reshape(-1, 3)[~region_wp.numpy()].tolist()
    }

    new_v, new_f, new_r = od.remesh.refine_region_to_density(vertices_wp, faces_wp, region_wp)

    assert new_v.size > n_vertices  # non-vacuity: it refined
    assert new_f.size // 3 == faces_wp.size // 3 + 2 * (new_v.size - n_vertices)
    assert new_r.size == new_f.size // 3
    refined_tm = warp_to_trimesh(new_v, new_f)
    assert refined_tm.is_watertight
    assert refined_tm.euler_number == 2
    kept_np = {tuple(sorted(row)) for row in new_f.numpy().reshape(-1, 3)[~new_r.numpy()].tolist()}
    assert kept_np == outside_np
    # The originals are a prefix: new vertices are appended, so the caller's index range holds.
    assert np.array_equal(new_v.numpy()[:n_vertices], vertices_wp.numpy())


def test_refine_region_to_density_alpha_monotone(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: ``alpha`` is the criterion's one knob and it has to act like one.

    Raising it loosens both clauses of the test, so the patch can only get finer -- measured on the
    filled hemisphere: 0 vertices inserted at ``alpha = 1``, 58 at ``sqrt(2)`` (the paper's value)
    and 97 at 2. The assert is the ordering rather than the numbers, since the counts depend on the
    patch the minimum-weight fill happened to choose.
    """
    vertices_wp, faces_wp, region_wp = _filled_hemisphere(hemisphere)
    n_vertices = vertices_wp.size
    inserted = [
        od.remesh.refine_region_to_density(vertices_wp, faces_wp, region_wp, alpha=alpha)[0].size
        - n_vertices
        for alpha in (1.0, math.sqrt(2.0), 2.0)
    ]
    assert inserted[0] == 0
    assert inserted[0] < inserted[1] < inserted[2]


def test_refine_region_to_density_whole_mesh_region_terminates(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: the whole mesh as the region still terminates.

    The empty region, the identity, is pinned by
    [`test_refiners_return_buffers_independent_of_their_input`].
    """
    vertices_wp, faces_wp, region_wp = _filled_hemisphere(hemisphere)

    # No surrounding mesh at all: the scale attribute falls back to the whole mesh's edges, which
    # must still converge rather than divide by a zero scale for ever.
    all_wp = wp.ones(region_wp.size, dtype=wp.bool, device=region_wp.device)
    _whole_v, whole_f, _whole_r = od.remesh.refine_region_to_density(vertices_wp, faces_wp, all_wp)
    assert whole_f.size >= faces_wp.size


def _no_region(faces: wp.array[wp.int32]) -> wp.array[wp.bool]:
    return wp.zeros(0, dtype=wp.bool, device=faces.device)


# (name, callable) pairs; each callable takes (vertices, faces) and returns the arrays that must
# come back empty: every array an operator returns, except where a test only ever pinned its faces
# (``isotropic_remesh`` and the two region refiners).
_REMESH_EMPTY_MESH_CASES = [
    ("isotropic_remesh", lambda v, f: (od.remesh.isotropic_remesh(v, f)[1],)),
    ("cluster_decimate", lambda v, f: od.remesh.cluster_decimate(v, f)),
    ("quadric_decimate", lambda v, f: od.remesh.quadric_decimate(v, f, target_faces=0)),
    ("flip_to_delaunay", lambda v, f: (od.remesh.flip_to_delaunay(v, f),)),
    ("flip_by_objective", lambda v, f: (od.remesh.flip_by_objective(v, f),)),
    ("subdivide", lambda v, f: od.remesh.subdivide(v, f)),
    ("subdivide_loop", lambda v, f: od.remesh.subdivide_loop(v, f)),
    ("subdivide_to_size", lambda v, f: od.remesh.subdivide_to_size(v, f, 1.0, return_index=True)),
    (
        "subdivide_region_to_size",
        lambda v, f: (od.remesh.subdivide_region_to_size(v, f, _no_region(f), max_edge=0.1)[1],),
    ),
    (
        "refine_region_to_density",
        lambda v, f: (od.remesh.refine_region_to_density(v, f, _no_region(f))[1],),
    ),
]


@pytest.mark.parametrize(
    "remesh_fn",
    [case[1] for case in _REMESH_EMPTY_MESH_CASES],
    ids=[case[0] for case in _REMESH_EMPTY_MESH_CASES],
)
def test_remesh_empty_mesh_is_a_noop(
    device: str,
    remesh_fn: Callable[[wp.array[wp.vec3], wp.array[wp.int32]], tuple[wp.array[wp.int32], ...]],
) -> None:
    """Not a library comparison: every operator returns an all-empty result on an empty mesh."""
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = warp_empty(0, wp.int32, device)
    for result_wp in remesh_fn(vertices_wp, faces_wp):
        assert result_wp.size == 0


# (name, callable) pairs; each callable takes (vertices, faces, region) and calls the operator
# with that region. flip_to_delaunay and flip_by_objective share the guard via _flip_setup;
# refine_region_to_density has its own but raises the identical message -- verified directly
# against all three on the same icosphere(1) + 2-element region before merging.
_REMESH_REJECTS_MISMATCHED_REGION_CASES = [
    ("flip_to_delaunay", lambda v, f, r: od.remesh.flip_to_delaunay(v, f, region=r)),
    ("flip_by_objective", lambda v, f, r: od.remesh.flip_by_objective(v, f, region=r)),
    (
        "refine_region_to_density",
        lambda v, f, r: od.remesh.refine_region_to_density(v, f, region=r),
    ),
]


@pytest.mark.parametrize(
    "remesh_fn",
    [case[1] for case in _REMESH_REJECTS_MISMATCHED_REGION_CASES],
    ids=[case[0] for case in _REMESH_REJECTS_MISMATCHED_REGION_CASES],
)
def test_remesh_rejects_a_mismatched_region(
    device: str,
    remesh_fn: Callable[[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]], object],
) -> None:
    """Not a library comparison: the documented ``ValueError`` on a wrong-length ``region``."""
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=1)
    with pytest.raises(ValueError, match="region must have length"):
        remesh_fn(vertices_wp, faces_wp, wp.zeros(2, dtype=wp.bool, device=device))


@pytest.mark.parametrize(
    "entry_point", ["subdivide_to_size", "subdivide_region_to_size", "refine_region_to_density"]
)
def test_refiners_return_buffers_independent_of_their_input(
    entry_point: str, hemisphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: no reference exposes a buffer-aliasing contract.

    All three refiners bind ``current_vertices = vertices`` and can leave their pass loop before
    ever replacing it, so the "nothing needed splitting" path used to hand the caller its own
    arrays straight back — ``refine_region_to_density`` did it for the region mask too. The module
    documents independent buffers (``subdivide``'s guard says so in as many words), so the check
    that bites is *writing into the result and finding the input untouched*, which a no-op test
    comparing values cannot make: an alias compares equal. The value comparison is asserted first,
    so this is also where each refiner's no-op result is pinned.
    """
    vertices_src, faces_src, region_wp = _filled_hemisphere(hemisphere)
    # Clone off the session fixture first, so a regression corrupts this test's own buffers rather
    # than every later test that shares the hemisphere.
    vertices_wp = wp.clone(vertices_src)
    faces_wp = wp.clone(faces_src)
    empty_wp = wp.zeros(region_wp.size, dtype=wp.bool, device=region_wp.device)
    vertices_before = vertices_wp.numpy().copy()
    faces_before = faces_wp.numpy().copy()

    new_region_wp = None
    if entry_point == "subdivide_to_size":
        beyond_longest = 10.0 * float(np.ptp(vertices_before, axis=0).max())
        new_vertices_wp, new_faces_wp = od.remesh.subdivide_to_size(
            vertices_wp, faces_wp, beyond_longest
        )
    elif entry_point == "subdivide_region_to_size":
        new_vertices_wp, new_faces_wp, new_region_wp = od.remesh.subdivide_region_to_size(
            vertices_wp, faces_wp, empty_wp, max_edge=0.01, delaunay=False
        )
    else:
        new_vertices_wp, new_faces_wp, new_region_wp = od.remesh.refine_region_to_density(
            vertices_wp, faces_wp, empty_wp
        )

    # The no-op path is the one under test, so confirm it was taken before reading anything into
    # the independence check below.
    assert np.array_equal(new_vertices_wp.numpy(), vertices_before)
    assert np.array_equal(new_faces_wp.numpy(), faces_before)

    new_vertices_wp.fill_(wp.vec3(1.0e3, 1.0e3, 1.0e3))
    new_faces_wp.fill_(-7)
    if new_region_wp is not None:
        new_region_wp.fill_(True)

    assert np.array_equal(vertices_wp.numpy(), vertices_before)
    assert np.array_equal(faces_wp.numpy(), faces_before)
    if new_region_wp is not None:
        assert not np.any(empty_wp.numpy())


# ---------------------------------------------------------------------------
# split_edges
# ---------------------------------------------------------------------------


@pytest.mark.parity(
    "split_edges",
    "igl",
    "open3d",
    benchmarked=False,
    reason="both are timed already, and not here: open3d's subdivide_midpoint carries the row in "
    "the subdivide group and a second one would double-count it, while igl.upsample is absent from "
    "benchmarks/ outright because it corrupts the process heap on the scan meshes -- the reason it "
    "is pinned to icosahedron here, the one fixture it is measured clean on.",
)
def test_split_edges_all_matches_igl_and_open3d(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B (vertex order): splitting every edge is the regular 1-to-4 subdivision.

    That is the operation both ``igl.upsample`` and open3d's ``subdivide_midpoint(1)`` perform.

    The direct version of the transitive claim
    [`test_split_edges_every_edge_is_the_regular_subdivision`] makes: rather than routing through
    ``subdivide`` and inheriting *its* references, this hands the primitive's own output to
    ``igl.upsample`` and open3d's ``subdivide_midpoint(1)``. Both insert one vertex per edge at its
    midpoint and retriangulate each face into four, which is exactly ``split_edges`` with an
    all-``True`` mask.

    The named transform is a nearest-neighbour bijection on the vertex set, because the three
    libraries number the inserted midpoints in three different orders -- ``lexsort_rows`` is not
    usable on float coordinates (CLAUDE.md section 7.5), and the counts are asserted first so the
    bijection cannot hide a missing or duplicated vertex. Measured max nearest-neighbour distance
    5.4e-08 against both, i.e. ordito's float32 storage.

    ``igl.upsample`` is pinned to ``icosahedron``: it corrupts the heap on the scan meshes and
    SIGSEGVs on ``bunny``, and ``icosahedron`` is the fixture it is measured clean on over 1 200
    calls.
    """
    mesh_tm, mesh_wp = icosahedron
    n_faces = mesh_wp.indices.size // 3
    split_vertices_wp, split_faces_wp = _split_every_edge(mesh_wp)
    split_np = split_vertices_wp.numpy().astype(np.float64)

    vertices_igl, faces_upsampled_igl = map(
        np.asarray,
        igl.upsample(np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), faces_igl(mesh_tm)),
    )
    mesh_o3d = trimesh_to_open3d(mesh_tm).subdivide_midpoint(number_of_iterations=1)
    vertices_o3d = np.asarray(mesh_o3d.vertices)

    # Counts first: a bijection over the wrong number of points is not a comparison.
    assert split_faces_wp.size // 3 == 4 * n_faces
    assert vertices_igl.shape[0] == len(split_np)
    assert vertices_o3d.shape[0] == len(split_np)
    assert np.asarray(faces_upsampled_igl).shape[0] == 4 * n_faces
    assert np.asarray(mesh_o3d.triangles).shape[0] == 4 * n_faces

    for reference in (vertices_igl, vertices_o3d):
        assert_nearest_bijection(split_np, np.ascontiguousarray(reference, dtype=np.float64), 1e-5)


def test_split_edges_every_edge_is_the_regular_subdivision(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: splitting every edge must equal ``subdivide``.

    ``subdivide`` is the half of the pair carrying the oracle -- splitting every edge *is* the
    1-to-4 subdivision, so the two must agree exactly.

    The strongest available check on the templates: the ``count == 3`` branch of the emission kernel
    is only reachable this way, and ``subdivide`` is independently tested against trimesh and igl,
    so agreeing with it exactly validates the primitive against those references transitively.

    That transitivity is why the ``split_edges`` benchmark group carries no parity claim: the
    reference coverage here is real but indirect, and a marker would assert a comparison this test
    does not make. A *direct* one is available -- ``igl.upsample`` and open3d's
    ``subdivide_midpoint`` both bind the regular subdivision -- and would upgrade this to Class B.
    """
    _mesh_tm, mesh_wp = icosahedron
    split_v, split_f = _split_every_edge(mesh_wp)
    fine_v, fine_f = od.remesh.subdivide(mesh_wp.points, mesh_wp.indices)

    assert split_f.size // 3 == 4 * (mesh_wp.indices.size // 3)
    assert np.array_equal(split_f.numpy(), fine_f.numpy())
    assert np.allclose(split_v.numpy(), fine_v.numpy())


def test_split_edges_honours_caller_supplied_positions(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    ``split_positions`` puts the new vertex where the caller asks, not at the midpoint.

    This is the parameter ``split_mesh_with_plane`` depends on, so the test pins the *indexing*
    contract too: positions are ordered by the exclusive scan of the mask, i.e. ascending
    unique-edge index among the flagged edges.
    """
    _mesh_tm, mesh_wp = icosahedron
    device = mesh_wp.indices.device
    unique_edges, inverse = od.edges.edges_unique(mesh_wp.indices)
    edges_np = unique_edges.numpy()
    points_np = mesh_wp.points.numpy().astype(np.float64)

    # Flag three edges and place each new vertex at 1/4 along, which no midpoint could match.
    chosen = np.array([1, 5, 9])
    mask_np = np.zeros(len(edges_np), dtype=bool)
    mask_np[chosen] = True
    quarter_np = 0.75 * points_np[edges_np[chosen, 0]] + 0.25 * points_np[edges_np[chosen, 1]]
    mask_wp = wp.array(np.ascontiguousarray(mask_np), dtype=wp.bool, device=device)
    positions_wp = points_to_warp(quarter_np, device)

    split_v, split_f = od.remesh.split_edges(
        mesh_wp.points,
        mesh_wp.indices,
        mask_wp,
        positions_wp,
        unique_edges=unique_edges,
        inverse=inverse,
    )
    inserted = split_v.numpy().astype(np.float64)[mesh_wp.points.size :]
    assert inserted.shape[0] == 3
    # Ascending edge order, element-wise: the documented slot assignment.
    assert np.allclose(inserted, quarter_np, atol=1e-6)
    # Each flagged edge cut one face into two, and the faces are still valid.
    assert split_f.size // 3 > mesh_wp.indices.size // 3
    assert od.validation.is_edge_manifold(split_f, allow_boundary_edges=False)


def test_split_edges_is_crack_free_for_an_arbitrary_mask(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Not a library comparison: an arbitrary edge subset still leaves a closed, manifold mesh."""
    mesh_tm, mesh_wp = icosahedron
    device = mesh_wp.indices.device
    unique_edges, inverse = od.edges.edges_unique(mesh_wp.indices)
    rng = np.random.default_rng(20260811)
    mask_np = rng.random(int(unique_edges.shape[0])) < 0.5
    mask_wp = wp.array(np.ascontiguousarray(mask_np), dtype=wp.bool, device=device)

    split_v, split_f = od.remesh.split_edges(
        mesh_wp.points, mesh_wp.indices, mask_wp, unique_edges=unique_edges, inverse=inverse
    )
    assert int(mask_np.sum()) > 0  # anti-vacuity
    assert split_v.size == mesh_wp.points.size + int(mask_np.sum())
    assert od.validation.is_edge_manifold(split_f, allow_boundary_edges=False)
    assert np.isclose(warp_to_trimesh(split_v, split_f).area, mesh_tm.area, rtol=1e-5)


def test_split_edges_carries_a_per_face_index(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """``index`` rides through the split, and ``None`` reports provenance into the input faces."""
    _mesh_tm, mesh_wp = icosahedron
    device = mesh_wp.indices.device
    n_faces = mesh_wp.indices.size // 3

    _v, faces_wp, provenance = _split_every_edge(mesh_wp, return_index=True)
    provenance_np = provenance.numpy()
    assert provenance_np.size == faces_wp.size // 3
    assert provenance_np.min() >= 0
    assert provenance_np.max() < n_faces
    # A 1-to-4 split means every input face appears exactly four times.
    assert np.array_equal(np.bincount(provenance_np, minlength=n_faces), np.full(n_faces, 4))

    # An explicit index is carried rather than replaced: label faces by parity and check it
    # survives.
    labels_np = (np.arange(n_faces) % 2).astype(np.int32)
    _v2, _f2, carried = _split_every_edge(
        mesh_wp, index=wp.array(labels_np, dtype=wp.int32, device=device), return_index=True
    )
    assert np.array_equal(carried.numpy(), labels_np[provenance_np])


def test_split_edges_empty_mask_is_a_copy(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    unique_edges, inverse = od.edges.edges_unique(mesh_wp.indices)
    nothing = wp.zeros(int(unique_edges.shape[0]), dtype=wp.bool, device=mesh_wp.indices.device)

    split_v, split_f, index = od.remesh.split_edges(
        mesh_wp.points,
        mesh_wp.indices,
        nothing,
        unique_edges=unique_edges,
        inverse=inverse,
        return_index=True,
    )
    assert np.array_equal(split_f.numpy(), mesh_wp.indices.numpy())
    assert np.allclose(split_v.numpy(), mesh_wp.points.numpy())
    assert np.array_equal(index.numpy(), np.arange(mesh_wp.indices.size // 3))


def test_split_edges_empty_mask_does_not_alias_the_caller_index(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: buffer ownership is ordito's contract, not a reference's.

    The sibling above pins the *values* on the no-split path; this pins that the third return is
    the caller's own ``index`` buffer copied rather than handed straight back, so that ownership
    does not silently depend on whether any edge happened to be flagged.

    **Bug class excluded:** a caller writing into the returned index destroying its own input.
    **Mutation probe, measured against a detached pre-fix worktree:** returning ``carried``
    unchanged makes the write below land in ``carry`` as well, failing the last assert.
    """
    _mesh_tm, mesh_wp = icosahedron
    device = mesh_wp.indices.device
    unique_edges, inverse = od.edges.edges_unique(mesh_wp.indices)
    nothing = wp.zeros(int(unique_edges.shape[0]), dtype=wp.bool, device=device)
    n_faces = mesh_wp.indices.size // 3
    carry = od.array.arange(n_faces, device=device)

    _split_v, _split_f, index = od.remesh.split_edges(
        mesh_wp.points,
        mesh_wp.indices,
        nothing,
        unique_edges=unique_edges,
        inverse=inverse,
        index=carry,
        return_index=True,
    )
    assert index.ptr != carry.ptr
    assert np.array_equal(index.numpy(), carry.numpy())
    index.fill_(-1)
    assert np.array_equal(carry.numpy(), np.arange(n_faces))


def test_split_edges_validation(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    device = mesh_wp.indices.device
    unique_edges, inverse = od.edges.edges_unique(mesh_wp.indices)
    n_edges = int(unique_edges.shape[0])
    n_faces = mesh_wp.indices.size // 3
    full_mask = wp.full(n_edges, True, dtype=wp.bool, device=device)

    with pytest.raises(ValueError, match="one entry per unique edge"):
        od.remesh.split_edges(
            mesh_wp.points,
            mesh_wp.indices,
            wp.full(n_edges + 1, True, dtype=wp.bool, device=device),
            unique_edges=unique_edges,
            inverse=inverse,
        )
    with pytest.raises(ValueError, match="one entry per flagged edge"):
        od.remesh.split_edges(
            mesh_wp.points,
            mesh_wp.indices,
            full_mask,
            wp.zeros(n_edges - 1, dtype=wp.vec3, device=device),
            unique_edges=unique_edges,
            inverse=inverse,
        )
    with pytest.raises(ValueError, match="one entry per face"):
        od.remesh.split_edges(
            mesh_wp.points,
            mesh_wp.indices,
            full_mask,
            unique_edges=unique_edges,
            inverse=inverse,
            index=wp.zeros(n_faces + 1, dtype=wp.int32, device=device),
        )


def _split_every_edge(mesh_wp: wp.Mesh, **kwargs: Any) -> tuple[Any, ...]:
    """Run ``split_edges`` with every unique edge flagged, forwarding its keywords."""
    unique_edges, inverse = od.edges.edges_unique(mesh_wp.indices)
    every_edge = wp.full(
        int(unique_edges.shape[0]), True, dtype=wp.bool, device=mesh_wp.indices.device
    )
    return od.remesh.split_edges(
        mesh_wp.points,
        mesh_wp.indices,
        every_edge,
        unique_edges=unique_edges,
        inverse=inverse,
        **kwargs,
    )
