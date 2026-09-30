"""
Benchmarks for ``triwarp.combine``: assembling meshes from parts and splitting them apart.

Axis: **components** for ``split`` and ``concatenate``, **loops_dp** for the stitching pair.
Neither family scales with face count, and this pair is the sharpest example in the package of
functions whose cost lives entirely somewhere else -- in the number of pieces going in or coming
out.

``split`` is one labelling pass plus one stable radix sort -- both O(F) and both fast -- and it is
nearly flat across the component axis and a win at every point. The axis is what keeps it that
way: a **host loop calling ``submesh_from_face_indices`` once per component** spreads by two orders
of magnitude across it and ends as a *loss* to trimesh at a thousand components, and no face-count
sweep would see that -- the scan registry's meshes differ in component count only by accident. The
residual slope is what is left of the per-component cost.

``stitch_min_weight`` runs a grid dynamic program over the two rims: an O(La x Lb) table filled by
O(La + Lb) *sequential* anti-diagonal launches, then a host traceback. So it is sized by rim
length, which is what ``loops_dp`` provides -- ``rim_short``'s two 512-vertex rims give a
512 x 512 table, and the meshes are deliberately tiny (1 024 faces) because the face count is
irrelevant to the cost.

References
----------
**trimesh**'s ``Trimesh.split(only_watertight=False)`` and **open3d**'s
``cluster_connected_triangles`` plus a ``select_by_index`` per cluster are the two equivalents of
``split``; both do the same labelling-then-compaction, and the open3d branch's ``np.unique`` over
each cluster's faces is part of what an open3d user pays, exactly as scipy is for the trimesh path.

**pymeshlab**'s ``generate_splitting_by_connected_components`` is the third, and the cheapest to
state: one filter call does both halves and *pushes one new mesh per component* onto the MeshSet, so
the component count is read straight off ``mesh_number()``. It is also the group's sharpest
reference, because it carries a per-component host cost: it spreads by more than an order of
magnitude across the axis where triwarp's batched compaction is nearly flat. That is the shape the
axis exists to detect, reproduced independently.

Neither trimesh, open3d nor pymeshlab has an equivalent of ``stitch`` / ``stitch_min_weight``:
joining two open meshes along their boundary loops with a minimum-weight triangulation is not in any
of the three APIs (trimesh's ``util.concatenate`` merges without stitching, open3d's boolean
operations need closed inputs, and MeshLab's ``meshing_snap_mismatched_borders`` snaps *coincident*
borders together rather than triangulating a gap between them). Those two groups are before/after
self-comparisons.
"""

from __future__ import annotations

import numpy as np
import pytest
import pytorch3d.structures as p3d_structures
import torch
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase, mesh_ml_from_numpy

# Copies for the concatenate sweep: the function still issues one packing copy per input buffer, so
# the input *count* is the driver and the total face count is held roughly fixed between the points.
_CONCAT_COPIES = [8, 512]

# The min-weight stitch DP and split on a thousand components run to tens of milliseconds a call.
_ROUNDS = 3

_split_cache: dict[tuple[str, str], tuple] = {}
_parts_cache: dict[tuple[str, str, int], list] = {}


def _split_inputs(bench_case: BenchCase) -> tuple:
    """Return the whole mesh as one soup, on the device (or host) the case needs."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _split_cache:
        if bench_case.kind == "triwarp":
            _split_cache[key] = (bench_case.vertices_wp, bench_case.faces_wp)
        else:
            _split_cache[key] = (bench_case.vertices_np, bench_case.faces_np)
    return _split_cache[key]


_mesh_ml_cache: dict[str, mm.Mesh] = {}


def _mesh_ml(bench_case: BenchCase) -> mm.Mesh:
    """Cache one ``meshlib.Mesh`` per mesh: the split row reads it and never mutates it."""
    if bench_case.mesh_name not in _mesh_ml_cache:
        _mesh_ml_cache[bench_case.mesh_name] = bench_case.new_mesh_ml()
    return _mesh_ml_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="split")
@pytest.mark.benchaxis("components")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab", "meshlib")
def test_split(bench_case: BenchCase) -> None:
    """
    Label, sort, then one batched compaction of every component: nearly flat across the axis.

    meshlib's row stops at the labelling: ``getAllComponents`` returns one ``FaceBitSet`` per
    component and ``cloneRegion`` -- which needs an ``ObjectMesh`` wrapper -- is what would extract
    them, so this is a lower bound on the group rather than the same work. That makes it the useful
    row to read against triwarp's across the components axis: the gap between them is the
    compaction, which is the half this group was optimized for. ``FaceIncidence.PerEdge`` is
    triwarp's rule; the mesh is read-only here, so one serves every round.
    """
    expected = {"sphere_med": 1, "parts_64": 64, "parts_1024": 1024}[bench_case.mesh_name]
    if bench_case.kind == "meshlib":
        mesh_part_ml = mm.MeshPart(_mesh_ml(bench_case))
        components_ml = bench_case.run(
            lambda: mm.getAllComponents(mesh_part_ml, mm.MeshComponents.FaceIncidence.PerEdge),
            rounds=_ROUNDS,
        )
        assert len(components_ml) == expected
        return
    if bench_case.kind == "pymeshlab":
        # One filter, but it *pushes* one new mesh per component onto the MeshSet, so it mutates the
        # set and must be rebuilt per round. The mesh count is asserted on, which is the same
        # component-count check the other three branches make.
        def split_pml() -> int:
            meshset_pml = bench_case.new_meshset_pml()
            meshset_pml.generate_splitting_by_connected_components()
            return meshset_pml.mesh_number() - 1

        assert bench_case.run(split_pml, rounds=_ROUNDS) == expected
        return
    if bench_case.kind == "triwarp":
        vertices, faces = _split_inputs(bench_case)
        parts = bench_case.run(lambda: tw.combine.split(vertices, faces), rounds=_ROUNDS)
    elif bench_case.kind == "trimesh":
        vertices_np, faces_np = _split_inputs(bench_case)
        mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
        parts = bench_case.run(lambda: mesh_tm.split(only_watertight=False), rounds=_ROUNDS)
    else:
        # ``tw.combine.split`` returns compact per-component ``(vertices, faces)`` submeshes, so the
        # open3d equivalent is ``cluster_connected_triangles`` (the labelling) followed by
        # ``select_by_index`` per cluster (the compaction). ``select_by_index`` takes *vertex*
        # indices, hence the ``np.unique`` over each cluster's faces -- numpy is part of what an
        # open3d user pays here, exactly as scipy is for the trimesh path.
        mesh_o3d = bench_case.mesh_o3d
        faces_i32 = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)

        def run_o3d() -> list:
            labels_np = np.asarray(mesh_o3d.cluster_connected_triangles()[0])
            return [
                mesh_o3d.select_by_index(np.unique(faces_i32[labels_np == label]))
                for label in range(int(labels_np.max()) + 1)
            ]

        parts = bench_case.run(run_o3d, rounds=_ROUNDS)
    assert len(parts) == expected


def _face_slice(bench_case: BenchCase, lo: int, hi: int) -> wp.array[wp.int32]:
    """Contiguous face-index range as a device buffer (untimed setup, so numpy is fine)."""
    return wp.array(np.arange(lo, hi, dtype=np.int32), dtype=wp.int32, device=bench_case.device)


def _submesh(bench_case: BenchCase, lo: int, hi: int) -> tuple:
    """Faces ``[lo, hi)`` of the case mesh as a compact standalone ``(vertices, faces)`` pair."""
    return tw.selection.submesh_from_face_indices(
        bench_case.vertices_wp, bench_case.faces_wp, _face_slice(bench_case, lo, hi)
    )


def _parts(bench_case: BenchCase, copies: int) -> list:
    """Build ``copies`` independent ``(vertices, faces)`` pairs to feed ``concatenate``."""
    key = (bench_case.mesh_name, str(bench_case.device), copies)
    if key not in _parts_cache:
        n_faces = bench_case.n_faces
        stride = max(1, n_faces // copies)
        bounds = [(lo, min(lo + stride, n_faces)) for lo in range(0, n_faces, stride)]
        _parts_cache[key] = [_submesh(bench_case, lo, hi) for lo, hi in bounds if lo < hi]
    return _parts_cache[key]


_parts_p3d_cache: dict[tuple[str, str, int], list] = {}


def _parts_p3d(bench_case: BenchCase, copies: int) -> list:
    """Build the same pieces as a list of ``Meshes``, cached: the input, not the operation."""
    key = (bench_case.mesh_name, bench_case.torch_device, copies)
    if key not in _parts_p3d_cache:
        vertices_p3d = torch.as_tensor(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
            device=bench_case.torch_device,
        )
        faces_np, n_faces = bench_case.faces_np, bench_case.n_faces
        stride = max(1, n_faces // copies)
        _parts_p3d_cache[key] = [
            p3d_structures.Meshes(
                verts=[vertices_p3d],
                faces=[
                    torch.as_tensor(
                        np.ascontiguousarray(
                            faces_np[lo : min(lo + stride, n_faces)], dtype=np.int64
                        ),
                        device=bench_case.torch_device,
                    )
                ],
            )
            for lo in range(0, n_faces, stride)
            if lo < n_faces
        ]
    return _parts_p3d_cache[key]


_parts_ml_cache: dict[tuple[str, int], mm.std_vector_std_shared_ptr_Mesh] = {}


def _parts_ml(bench_case: BenchCase, copies: int) -> mm.std_vector_std_shared_ptr_Mesh:
    """Build the same pieces as a MeshLib mesh vector, cached: the input, not the operation."""
    key = (bench_case.mesh_name, copies)
    if key not in _parts_ml_cache:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        n_faces = bench_case.n_faces
        stride = max(1, n_faces // copies)
        meshes_ml = mm.std_vector_std_shared_ptr_Mesh()
        for lo in range(0, n_faces, stride):
            block_np = faces_np[lo : min(lo + stride, n_faces)]
            if block_np.shape[0]:
                meshes_ml.append(mesh_ml_from_numpy(vertices_np, block_np))
        _parts_ml_cache[key] = meshes_ml
    return _parts_ml_cache[key]


@pytest.mark.benchmark(group="concatenate")
@pytest.mark.benchlibs("triwarp", "trimesh", "meshlib", "pytorch3d")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.parametrize("copies", _CONCAT_COPIES)
def test_concatenate(bench_case: BenchCase, copies: int) -> None:
    """
    The inverse of ``split``, and the same shape of cost: work proportional to the *input count*.

    Total face count is held at ``sphere_med``'s 81 920 and only the number of pieces changes, so
    any slope here is per-mesh overhead rather than data movement. The index renumbering is a
    single launch over the packed buffer rather than one ``wp.map`` per input mesh, which is most of
    the slope; what remains is two ``wp.copy`` calls per piece, because Warp has no gather across
    separate allocations and the packing itself cannot be batched.

    **pytorch3d**'s ``join_meshes_as_scene`` is this operation exactly -- concatenate the vertex
    buffers, shift each piece's indices by the running count -- and
    ``tests/test_combine.py::test_concatenate_matches_pytorch3d`` pins it to byte equality on both
    halves. Like triwarp's row its pieces are cached (they are the input), and like triwarp's it
    has no gather across separate allocations to exploit, so its slope in the piece count is the
    same shape of per-piece overhead this group exists to measure.
    """
    if bench_case.kind == "pytorch3d":
        pieces_p3d = _parts_p3d(bench_case, copies)
        joined_p3d = bench_case.run(lambda: p3d_structures.join_meshes_as_scene(pieces_p3d))
        assert joined_p3d.faces_packed().shape[0] == bench_case.n_faces
        return
    if bench_case.kind == "meshlib":
        # ``mergeMeshes`` takes a vector of *shared pointers* and returns a new mesh without
        # touching its inputs, so unlike almost everything else in this library the pieces are
        # cached: they are the input, and rebuilding them per round would time the converter.
        pieces_ml = _parts_ml(bench_case, copies)
        merged_ml = bench_case.run(lambda: mm.mergeMeshes(pieces_ml))
        assert merged_ml.topology.numValidFaces() == bench_case.n_faces
        return
    if bench_case.kind == "triwarp":
        pieces = _parts(bench_case, copies)
        vertices, faces = bench_case.run(lambda: tw.combine.concatenate(pieces))
        assert faces.size // 3 == bench_case.n_faces
        assert vertices.size > 0
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        stride = max(1, bench_case.n_faces // copies)
        meshes_tm = [
            tm.Trimesh(vertices_np, faces_np[lo : lo + stride], process=False)
            for lo in range(0, bench_case.n_faces, stride)
        ]
        combined = bench_case.run(lambda: tm.util.concatenate(meshes_tm))
        assert len(combined.faces) == bench_case.n_faces
