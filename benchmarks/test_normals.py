"""
Benchmarks for ``ordito.normals``: the vertex-normal weightings and their accumulators.

Two axes, because the module has two different kinds of function:

* **scan sweep** for ``n_vertices`` and ``mean_vertex_normals`` -- one pass over ``3F`` indices,
  pure throughput, and the place ``lucy`` (28M faces) earns its keep: a host-side ``n_vertices``
  copies the whole face buffer across the bus just to take a max.
* **valence** for everything that *accumulates* per vertex. All of these do ``3F`` atomic adds
  into ``V`` slots, so in principle a mesh with a few very-high-valence hubs serializes where a
  regular one does not. ``fan_hub`` is the extreme: identical vertex *and* face count to
  ``sphere_med``, but its cone apex and base centre have valence 40 960 against a uniform 6.

``average_onto_vertices`` and ``transfer_onto_vertices`` live in
[`test_interpolation.py`](test_interpolation.py), where ``ordito.interpolation`` does, under group
names that name *this* module — the group name is the cross-suite key every ``parity`` marker cites,
so it does not move with the file.

Across the contention axis the hub cost ``vertex_normals`` 9x and ``curvature.vertex_defects`` 3x
the uniform mesh while each lane committed its own atomics: trimesh's cone puts a hub in every
other face, so half of each warp hit one slot, and the ``float64`` accumulator serialized there. The
scatters now pool the lanes of a warp that share a vertex (``kernels/scatter.peer_summed_add``),
which took the hub rows to 0.10-0.27x and leaves them about 1.4x the uniform mesh on CUDA. The axis
is what would catch that regressing.

References
----------
**trimesh**'s area-weighted ``vertex_normals``, rebuilt inside the timed callable because trimesh
caches it. **open3d**'s ``compute_vertex_normals`` is also area-weighted and is ordito's closest
analogue; it writes the result into the mesh but recomputes on every call rather than caching, so
the shared mesh stays honest across rounds. ``igl.per_vertex_normals`` was dropped: it segfaults
flakily when invoked late in a session that mixes Warp CUDA/CPU JIT with the other native
libraries.

``n_vertices`` has no open3d equivalent worth timing: open3d stores the vertex count explicitly, so
``len(mesh.vertices)`` is O(1) and does not measure the max-reduce ordito performs.

**pymeshlab**'s ``weightmode`` enum is what makes it useful here: ``compute_normal_per_vertex``
implements four weighting schemes behind one filter, two of which are exactly ordito's --
``'Simple Average'`` is ``mean_vertex_normals`` and ``'By Area'`` is
``vertex_normals``, so the pair also isolates what the area weight costs on the reference's side.
Both write only the vertex-normal attribute and are idempotent,
so they run against the shared MeshSet with no build inside the timed region.

The reference is flat across the valence axis, and if anything marginally *faster* on the hub
mesh: a serial scatter has nothing to contend on.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import pytorch3d.structures as p3d_structures
import torch
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
from conftest import BenchCase

_face_data_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], wp.array[wp.float32]]] = {}


def _face_normals_and_areas(
    bench_case: BenchCase,
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32]]:
    """Precomputed per-face normals and areas -- optional *inputs*, not part of the operation."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _face_data_cache:
        _face_data_cache[key] = od.triangles.face_normals_and_areas(
            bench_case.vertices_wp, bench_case.faces_wp
        )
    return _face_data_cache[key]


@pytest.mark.benchmark(group="mean_vertex_normals")
@pytest.mark.benchlibs("ordito", "pymeshlab", "pyvista")
def test_mean_vertex_normals(bench_case: BenchCase) -> None:
    """
    The unweighted scatter, on the scan sweep: the throughput baseline for the group below.

    VTK's ``compute_normals`` is the unweighted scheme too, but it computes the *face* normals in
    the same pass where ordito is handed them, so its row carries the cross products as well --
    read the pair against ``face_normals_and_areas`` in
    [`test_triangles.py`](test_triangles.py) to separate the two halves.
    """
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        normals_pv = bench_case.run(
            lambda: mesh_pv.compute_normals(
                cell_normals=False,
                point_normals=True,
                consistent_normals=False,
                auto_orient_normals=False,
                split_vertices=False,
            )
        )
        assert np.asarray(normals_pv.point_data["Normals"]).shape == (bench_case.n_vertices, 3)
        return
    if bench_case.kind == "pymeshlab":  # 'Simple Average' is exactly the unweighted scheme
        meshset_pml = bench_case.meshset_pml
        bench_case.run(lambda: meshset_pml.compute_normal_per_vertex(weightmode="Simple Average"))
        assert meshset_pml.current_mesh().vertex_normal_matrix().shape[0] == bench_case.n_vertices
        return
    face_normals, _areas = _face_normals_and_areas(bench_case)
    n_vertices = bench_case.n_vertices
    faces = bench_case.faces_wp
    result = bench_case.run(lambda: od.normals.mean_vertex_normals(n_vertices, faces, face_normals))
    assert result.shape == (n_vertices,)


@pytest.mark.benchmark(group="weighted_vertex_normals")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_weighted_vertex_normals(bench_case: BenchCase) -> None:
    """
    The corner-weighted scatter, at ``mean_vertex_normals``' operating point: the weights' price.

    The weights are the corner angles, the scheme ``vertex_normals(weighting="angle")`` derives,
    and like the face normals they are *inputs* built outside the timed region on both sides.
    ``trimesh.geometry.weighted_vertex_normals`` is the reference ``tests/test_normals.py``
    compares against, fed trimesh's own face normals and angles there as here.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "trimesh":
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        faces_np, normals_np, angles_np = mesh_tm.faces, mesh_tm.face_normals, mesh_tm.face_angles
        normals_tm = bench_case.run(
            lambda: tm.geometry.weighted_vertex_normals(n_vertices, faces_np, normals_np, angles_np)
        )
        assert normals_tm.shape == (n_vertices, 3)
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    face_normals, _areas = _face_normals_and_areas(bench_case)
    face_angles = od.triangles.face_angles(vertices, faces)
    result = bench_case.run(
        lambda: od.normals.weighted_vertex_normals(n_vertices, faces, face_normals, face_angles)
    )
    assert result.shape == (n_vertices,)


@pytest.mark.benchmark(group="vertex_normals")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("ordito", "trimesh", "open3d", "pymeshlab", "meshlib", "pytorch3d")
def test_vertex_normals(bench_case: BenchCase) -> None:
    """
    Area-weighted scatter, uniform valence 6 against two 40 960-valence hubs.

    meshlib's ``computePerVertNormals`` is this weighting exactly -- 1.19e-07, and its
    angle-weighted sibling ``computePerVertPseudoNormals`` is 6.8e-03 away, which is what
    tests/test_normals.py::test_vertex_normal_weightings_match_meshlib pins. It is also the only
    **multi-threaded** row in this group, so read the ratio against ``ordito-cuda`` and not against
    ``ordito-cpu``. Pure, so the mesh is built once outside the timed callable.

    **pytorch3d**'s ``verts_normals_packed`` is the same area weighting (1.19e-07, pinned in
    tests/test_normals.py::test_vertex_normals_match_pytorch3d), but it is a **memoized accessor
    on an immutable container**, so unlike every other row here the ``Meshes`` has to be built
    *inside* the timed callable -- reading it twice on a shared object prices nothing. That means
    the row carries the container construction as well -- a small share on the host and about a
    third of the round on CUDA -- so read the CUDA row in particular as an upper bound on the
    scatter rather than as the scatter.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pytorch3d":
        vertices_p3d = torch.as_tensor(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
            device=bench_case.torch_device,
        )
        faces_p3d = torch.as_tensor(
            np.ascontiguousarray(bench_case.faces_np, dtype=np.int64),
            device=bench_case.torch_device,
        )
        normals_p3d = bench_case.run(
            lambda: p3d_structures.Meshes(
                verts=[vertices_p3d], faces=[faces_p3d]
            ).verts_normals_packed()
        )
        assert normals_p3d.shape == (n_vertices, 3)
        return
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        normals_ml = bench_case.run(lambda: mm.computePerVertNormals(mesh_ml))
        assert normals_ml.size() == n_vertices
        return
    if bench_case.kind == "pymeshlab":  # 'By Area' is ordito's weighting exactly
        meshset_pml = bench_case.meshset_pml
        bench_case.run(lambda: meshset_pml.compute_normal_per_vertex(weightmode="By Area"))
        assert meshset_pml.current_mesh().vertex_normal_matrix().shape == (n_vertices, 3)
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: od.normals.vertex_normals(vertices, faces))
        assert result.shape == (n_vertices,)
    elif bench_case.kind == "trimesh":
        # trimesh vertex_normals are area-weighted; rebuild inside (cached property)
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).vertex_normals)
        assert result.shape == (n_vertices, 3)
        assert np.isfinite(result).any()
    else:
        # open3d writes the normals into the mesh, but recomputes them on every call rather than
        # caching (identical cost on the second call), so the shared mesh is reusable.
        mesh_o3d = bench_case.mesh_o3d
        result_o3d = bench_case.run(mesh_o3d.compute_vertex_normals)
        assert np.asarray(result_o3d.vertex_normals).shape == (n_vertices, 3)


@pytest.mark.benchmark(group="vertex_normals_precomputed")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("ordito")
def test_vertex_normals_precomputed(bench_case: BenchCase) -> None:
    """
    The same scatter with ``face_normals`` / ``face_areas`` supplied: the warm half of the cost.

    Passing ``None`` for either makes the function recompute a full per-face pass first. Neither
    reference library has an equivalent -- both always recompute -- so this is ordito-only, and
    its gap against the group above is what a caller saves by keeping the face data around.
    """
    face_normals, face_areas = _face_normals_and_areas(bench_case)
    n_vertices = bench_case.n_vertices
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    result = bench_case.run(
        lambda: od.normals.vertex_normals(
            vertices, faces, face_normals=face_normals, face_weights=face_areas
        )
    )
    assert result.shape == (n_vertices,)


@pytest.mark.benchmark(group="vertex_normals_one_table")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("weighting", ["area", "angle"])
@pytest.mark.parametrize("supplied", ["face_normals", "face_weights"])
def test_vertex_normals_one_table(bench_case: BenchCase, weighting: str, supplied: str) -> None:
    """
    The scatter with one of the two per-face tables supplied: what ``Trimesh.vertex_normals`` does.

    The property passes whichever of its ``face_normals`` / ``face_angles`` caches is filled, and
    one of the two often is. The missing quantity is formed inside the scatter, so this should sit
    at ``vertex_normals_precomputed``'s cost; a gap above it is a per-face pass written first.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    face_normals, face_areas = _face_normals_and_areas(bench_case)
    mode: Literal["area", "angle"] = "angle" if weighting == "angle" else "area"
    if supplied == "face_normals":
        result = bench_case.run(
            lambda: od.normals.vertex_normals(
                vertices, faces, weighting=mode, face_normals=face_normals
            )
        )
    else:
        weights = od.triangles.face_angles(vertices, faces) if mode == "angle" else face_areas
        result = bench_case.run(
            lambda: od.normals.vertex_normals(vertices, faces, weighting=mode, face_weights=weights)
        )
    assert result.shape == (bench_case.n_vertices,)
