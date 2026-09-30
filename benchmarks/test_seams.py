"""
Benchmarks for ``triwarp.seams``: crease detection and the topological cut along a seam set.

Two groups, on two different mesh sets, because they have two different preconditions:

* ``crease_edges`` runs the **scan sweep**. It is a face-adjacency build plus one threshold — pure
  throughput, and the group to read as a floor for anything that consumes a feature set. Face
  adjacency tolerates a non-manifold edge (it drops it), so every registry mesh is fair game.
* ``cut_along_edges`` runs on the synthetic **icospheres** instead, and the reason is a hard
  precondition rather than a preference: the cut is defined through halfedge twins, so it needs an
  edge-manifold mesh, and ``bunny_decimated`` has **150 edges shared by three or more faces**.
  triwarp raises there and so does MeshLab (``this filter require manifoldness``), so neither side
  has a number to report — the same boundary the midpoint-subdivision references run into (see the
  benchmarks README hazard table).

The cut's cost is halfedge twins, a key set for the marked edges, a corner-graph build and a
**connected-components pass over ``3F`` nodes**. The sweep that matters for it is therefore not only
size but *how much* is cut, since that sets how far the components pass has to contract: the
``cut_fraction`` parameter takes it from a quarter of the interior edges to all of them.

References
----------
**pymeshlab** covers both: ``compute_selection_crease_per_edge`` is the same dihedral threshold
(reported as a vertex selection, so it is selection-only and shares the MeshSet) and
``meshing_cut_along_crease_edges`` is the cut — which finds the creases *and* cuts them in one call,
so that row is an upper bound on triwarp's cut alone and should be read against the sum of the two
triwarp rows. It rewrites the topology, so its MeshSet is rebuilt inside the timed callable. Its cut
row exists only at ``cut_fraction=1.0``: the filter takes a dihedral threshold rather than an edge
set, so ``angledeg=0`` (cut everything non-coplanar) is the only setting comparable to a triwarp
fraction.

One structural difference is worth knowing before comparing outputs rather than times: on a cube cut
at every crease, triwarp emits the **minimal** 24 vertices and MeshLab 32 (see
``tests/test_seams.py``). Both give 6 components and the same area, so the extra copies are
redundant rather than wrong — but they are extra memory in every downstream pass.

Neither trimesh, igl nor open3d has a cut along a marked edge set at all: trimesh's
``unmerge_vertices`` splits *every* edge (a full soup) rather than a chosen set, which is the
degenerate case rather than the operation.
"""

from __future__ import annotations

import igl
import numpy as np
import pymeshlab as ml
import pytest
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
import triwarp.typing as twt
from conftest import BenchCase, skip_larger_than

# Crease threshold in degrees. 30 is MeshLab's own documentation default for a "hard" edge and picks
# out a real feature set on every scan mesh rather than everything or nothing.
_CREASE_ANGLE = 30.0

_cut_cache: dict[tuple[str, str, float], twt.Array2dInt32] = {}
_atlas_cache: dict[str, np.ndarray] = {}
_seam_meshset_cache: dict[str, ml.MeshSet] = {}


def _cut_edges(bench_case: BenchCase, fraction: float) -> twt.Array2dInt32:
    """
    Take a deterministic ``fraction`` of the mesh's interior edges -- an *input* of the cut.

    Strided rather than sampled, so the marked set is spread evenly over the surface and the corner
    graph it leaves is the same shape on every mesh. Cached per (mesh, device, fraction).
    """
    key = (bench_case.mesh_name, str(bench_case.device), fraction)
    if key not in _cut_cache:
        # ``angle=0`` on an icosphere is every interior edge, since no two of its faces are
        # coplanar.
        all_edges = tw.seams.crease_edges(
            bench_case.vertices_wp, bench_case.faces_wp, angle=0.0
        ).numpy()
        stride = max(1, round(1.0 / fraction))
        _cut_cache[key] = twt.as_array2d(
            wp.array(
                np.ascontiguousarray(all_edges[::stride], dtype=np.int32),
                dtype=wp.int32,
                device=bench_case.device,
            ),
            wp.int32,
        )
    return _cut_cache[key]


@pytest.mark.benchmark(group="crease_edges")
@pytest.mark.benchlibs("triwarp", "pymeshlab", "pyvista", "meshlib")
def test_crease_edges(bench_case: BenchCase) -> None:
    """
    Face adjacency plus a dihedral threshold: the cheap half of the seam workflow.

    VTK's ``extract_feature_edges`` is the same threshold; the three other edge classes it can emit
    are switched off so it does the same work, and it returns the edges as a line-cell ``PolyData``
    rather than as a selection.

    meshlib's ``findCreaseEdges`` is the closest of the three in output shape -- an
    ``UndirectedEdgeBitSet``, one bit per edge, which is triwarp's answer without the pair decode --
    and the tightest in result: the same 12 edges on a unit box at both thresholds tested
    (``tests/test_seams.py``). Its angle is in **radians** and is measured from planar, so the
    module's degree constant is converted rather than passed. It reads the topology and mutates
    nothing, so one mesh serves every round.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        angle = float(np.radians(_CREASE_ANGLE))
        creases_ml = bench_case.run(
            lambda: mm.findCreaseEdges(mesh_ml.topology, mesh_ml.points, angle)
        )
        assert creases_ml.size() <= mesh_ml.topology.undirectedEdgeSize()
        return
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny", "vtkFeatureEdges is a single-threaded edge walk")
        mesh_pv = bench_case.mesh_pv
        creases_pv = bench_case.run(
            lambda: mesh_pv.extract_feature_edges(
                feature_angle=_CREASE_ANGLE,
                feature_edges=True,
                boundary_edges=False,
                non_manifold_edges=False,
                manifold_edges=False,
            )
        )
        assert creases_pv.n_cells >= 0
        return
    if bench_case.kind == "pymeshlab":
        skip_larger_than(bench_case, "bunny", "MeshLab's crease selection is a serial edge walk")
        meshset_pml = bench_case.meshset_pml  # selection-only, so the geometry survives
        bench_case.run(
            lambda: meshset_pml.compute_selection_crease_per_edge(
                angledegneg=-_CREASE_ANGLE, angledegpos=_CREASE_ANGLE
            )
        )
        assert meshset_pml.current_mesh().vertex_selection_array().shape == (bench_case.n_vertices,)
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    creases = bench_case.run(lambda: tw.seams.crease_edges(vertices, faces, angle=_CREASE_ANGLE))
    assert int(creases.shape[1]) == 2


@pytest.mark.benchmark(group="cut_along_edges")
@pytest.mark.benchmeshes("sphere_small", "sphere_med", "sphere_large")
@pytest.mark.benchlibs("triwarp", "igl", "pymeshlab")
@pytest.mark.parametrize("cut_fraction", [0.25, 1.0])
def test_cut_along_edges(bench_case: BenchCase, cut_fraction: float) -> None:
    """
    Twins, key set and a union-find over ``3F`` corners, each halfedge naming one join.

    The two fractions are the two ends of the components pass: at ``0.25`` most corners still merge,
    so the pass contracts ``3F`` nodes down toward ``V``, while at ``1.0`` every join is a self-loop
    and every node is already its own component. Both issue the same launches, allocations and
    readbacks -- the corner graph is formed in the union-find's own threads, never materialised or
    compacted -- so any gap between them is the hooks' device work.

    ``igl.cut_mesh`` is the reference that takes the same *edge set* triwarp does, once it is
    rewritten as the ``(n_faces, 3)`` per-corner bool mask the binding wants -- that rewrite is an
    input transform and is cached outside the timed callable, like ``_cut_edges`` itself. It is the
    only reference here that can run at both fractions, since MeshLab takes a dihedral threshold
    instead of a set. It also **agrees with triwarp on the output size** where MeshLab does not (24
    vertices against 32 on a cut cube; see ``tests/test_seams.py``), which is what makes the third
    row worth having.

    igl gets slower from ``0.25`` to ``1.0``: its per-corner walk pays for each new vertex it
    emits.
    """
    if bench_case.kind == "igl":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        mask_igl = _cut_corner_mask_np(bench_case, cut_fraction)
        vertices_cut_igl, faces_cut_igl = bench_case.run(
            lambda: igl.cut_mesh(vertices_np, faces_np, mask_igl)[:2], rounds=3
        )
        assert faces_cut_igl.shape[0] == bench_case.n_faces
        assert vertices_cut_igl.shape[0] >= bench_case.n_vertices
        return
    if bench_case.kind == "pymeshlab":
        if cut_fraction != 1.0:
            pytest.skip("MeshLab's cut takes a dihedral threshold, not an edge set")
        skip_larger_than(bench_case, "sphere_med", "MeshLab's cut is a serial per-face rewrite")
        # Rewrites the topology, so the MeshSet is rebuilt inside the timed callable. This filter
        # also *finds* the creases, so read it against both triwarp rows summed.
        new_meshset_pml = bench_case.new_meshset_pml

        def cut_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.meshing_cut_along_crease_edges(angledeg=0.0)
            return meshset_pml.current_mesh().vertex_number()

        assert bench_case.run(cut_pml, rounds=3) >= bench_case.n_vertices
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    edges = _cut_edges(bench_case, cut_fraction)
    cut_vertices, cut_faces = bench_case.run(
        lambda: tw.seams.cut_along_edges(vertices, faces, edges), rounds=3
    )
    assert cut_faces.size == faces.size
    assert np.isfinite(cut_vertices.numpy()[:1]).all()


_cut_mask_cache: dict[tuple[str, float], np.ndarray] = {}


def _cut_corner_mask_np(bench_case: BenchCase, fraction: float) -> np.ndarray:
    """
    Re-encode the same cut set as igl's ``(n_faces, 3)`` per-corner bool mask.

    ``igl.cut_mesh``'s ``C`` marks *corners*, not edges: ``C[f, i]`` is set when edge
    ``(F[f, i], F[f, (i + 1) % 3])`` is cut, so every cut edge is marked twice, once from each
    incident face. Note the ``(i, i + 1)`` numbering -- ``igl.ears`` uses the opposite-vertex one
    instead, and the two conventions coexist inside libigl (see ``tests/test_seams.py``).

    The edge set is rebuilt on the **cpu** rather than read from ``_cut_edges``: a reference case
    carries no Warp device, so ``bench_case.vertices_wp`` is unavailable there. Same mesh, same
    ``angle=0`` crease set, same stride, so the two paths receive the identical cut. Cached per
    (mesh, fraction), because it is a re-encoding of an *input* rather than part of the operation.
    """
    key = (bench_case.mesh_name, fraction)
    if key not in _cut_mask_cache:
        faces_np = bench_case.faces_np
        vertices_cpu = wp.array(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
            dtype=wp.vec3,
            device="cpu",
        )
        faces_cpu = wp.array(
            np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device="cpu"
        )
        all_edges_np = tw.seams.crease_edges(vertices_cpu, faces_cpu, angle=0.0).numpy()
        edges_np = all_edges_np[:: max(1, round(1.0 / fraction))]
        cut_keys = np.sort(edges_np.astype(np.int64), axis=1)
        cut_hashes = cut_keys[:, 0] * (int(faces_np.max()) + 1) + cut_keys[:, 1]
        mask = np.zeros(faces_np.shape, dtype=bool)
        for corner in range(3):
            pairs = np.sort(
                np.stack([faces_np[:, corner], faces_np[:, (corner + 1) % 3]], axis=1), axis=1
            )
            hashes = pairs[:, 0] * (int(faces_np.max()) + 1) + pairs[:, 1]
            mask[:, corner] = np.isin(hashes, cut_hashes)
        _cut_mask_cache[key] = np.ascontiguousarray(mask)
    return _cut_mask_cache[key]


def _wedge_atlas_np(bench_case: BenchCase) -> np.ndarray:
    """
    Build a ``(3 * n_faces, 2)`` per-corner atlas whose only seam is the ``+-pi`` wrap.

    Both sides need *some* atlas -- the registry meshes carry no UVs -- and this one is the honest
    input to time on: a continuous map would leave the seam mask empty and a per-triangle one would
    make every interior edge a seam, so neither exercises the compaction the way a real atlas does.
    Cached per mesh, since it is an input rather than part of the measurement.
    """
    if bench_case.mesh_name not in _atlas_cache:
        vertices_np = bench_case.vertices_np - bench_case.vertices_np.mean(axis=0)
        corners_np = vertices_np[bench_case.faces_np]
        u_np = np.arctan2(corners_np[..., 1], corners_np[..., 0]) / (2.0 * np.pi) + 0.5
        u_np = u_np - np.round(u_np - u_np[:, :1])
        radius_np = np.maximum(np.linalg.norm(corners_np, axis=-1), 1e-12)
        v_np = np.arccos(np.clip(corners_np[..., 2] / radius_np, -1.0, 1.0)) / np.pi
        _atlas_cache[bench_case.mesh_name] = np.stack([u_np, v_np], axis=-1).reshape(-1, 2)
    return _atlas_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="uv_seam_edges")
@pytest.mark.benchmeshes("sphere_small", "sphere_med", "sphere_large")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
def test_uv_seam_edges(bench_case: BenchCase) -> None:
    """
    Halfedge twins plus a per-edge texcoord comparison and three compactions.

    On the icospheres for the same reason the cut is: the classification is defined through twins,
    so it needs an edge-manifold mesh and ``bunny_decimated`` is not one.

    The cost split is worth knowing before optimizing this: twins is a radix sort over ``3F`` keys
    and the classification is one pass, so this group should track ``cut_along_edges``' first half
    almost exactly and any gap between them is the components pass, not the seam test.
    """
    if bench_case.kind == "pymeshlab":
        # Selection-only, so the wedge-UV MeshSet survives its own filter and is cached.
        if bench_case.mesh_name not in _seam_meshset_cache:
            meshset_pml = ml.MeshSet()
            meshset_pml.add_mesh(
                ml.Mesh(
                    vertex_matrix=np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64),
                    face_matrix=np.ascontiguousarray(bench_case.faces_np, dtype=np.int32),
                    w_tex_coords_matrix=np.ascontiguousarray(
                        _wedge_atlas_np(bench_case), dtype=np.float64
                    ),
                )
            )
            _seam_meshset_cache[bench_case.mesh_name] = meshset_pml
        meshset_pml = _seam_meshset_cache[bench_case.mesh_name]
        bench_case.run(meshset_pml.compute_selection_by_texture_seams_per_vertex)
        # MeshLab reports only the vertex set, and unions boundaries into it -- so it does strictly
        # less than the triwarp row, which also splits boundaries out and finds foldovers.
        assert meshset_pml.current_mesh().vertex_selection_array().shape == (bench_case.n_vertices,)
        return
    faces = bench_case.faces_wp
    texcoords = wp.array(
        np.ascontiguousarray(_wedge_atlas_np(bench_case), dtype=np.float32),
        dtype=wp.vec2,
        device=bench_case.device,
    )
    n_vertices = bench_case.n_vertices
    seams, boundaries, foldovers = bench_case.run(
        lambda: tw.seams.uv_seam_edges(faces, texcoords, n_vertices=n_vertices)
    )
    assert int(seams.shape[0]) > 0
    assert int(seams.shape[1]) == 4
    assert int(boundaries.shape[1]) == 2
    assert int(foldovers.shape[1]) == 4
