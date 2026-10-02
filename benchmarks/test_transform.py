"""
Benchmarks for ``ordito.transform``: applying an affine matrix to a point buffer and to a mesh.

Axis: the **scan sweep**, and nothing else. An affine map is one fused-multiply-add chain per
element with no topology, no parameters and no data-dependent branching, so vertex count is the
whole story -- this is the purest bandwidth row in the suite and the one to read when a more
complicated row is suspected of being memory-bound.

The rows exist for a second reason, which is that they are the **floor** every cached-mesh
workflow is measured against. [`Trimesh.transform`][ordito.mesh.Trimesh.transform] carries most
of its cache through a rigid motion, so what it costs is this row plus a BVH rebuild; a claim that
the cache is worth having is a claim about the *difference* between that and reassembling the
operators, and neither half means anything without this one.

No group here sweeps the transform *class*. Classification is a handful of host float comparisons
on a 4x4 -- [`classify_transform`][ordito.transform.classify_transform] does no device work at
all -- so a rigid matrix and a shear cost the same to apply, and a row per class would measure the
same kernel five times. The class decides what is *recomputed afterwards*, which belongs to the
caller's workflow rather than to this call.

References
----------
**trimesh**'s ``transformations.transform_points`` is the closest match in the suite: a pure
function, arrays in and arrays out, with no mesh object involved. It is also ``float64`` against
ordito's ``float32``, so it moves twice the bytes -- on a bandwidth row that is most of any gap
and the ratio should be read against it.

**pyvista**'s ``DataSet.transform(matrix, inplace=False)`` is VTK's ``vtkTransformFilter``. It
returns a new dataset and therefore also *copies the topology*, which ordito's row does not, so
read it as an upper bound. Its default ``transform_all_input_vectors=False`` is left alone: with it
on, VTK would additionally push every normal and vector array through the inverse transpose, which
is [`transform_normals`][ordito.transform.transform_normals]'s job and not this group's.

**open3d**'s ``PointCloud.transform`` **mutates**, so the cloud is rebuilt inside the timed
callable and the row prices that construction along with the transform -- the same rule its
``remove_*`` rows follow. On this operation the construction is the larger half at every size, so
the row is an upper bound with a wide margin rather than a close comparison.

**libigl** binds no transform: the C++ examples apply a matrix with Eigen directly, which would
make an igl row a NumPy row under another name. **meshlib**'s ``Mesh.transform`` mutates and needs
a fresh mesh per round like open3d's, and its ``AffineXf3f`` has to be assembled per call from
``Matrix3f`` rows; both are priced in ``new_mesh_ml`` rather than in the transform, so it is left
out rather than reported as a slow transform.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest
import trimesh as tm
import warp as wp

import ordito as od
from conftest import BenchCase

# The matrix every row applies: a rotation about an off-origin centre, so no component of the
# multiply is trivially zero and none of the libraries can shortcut a translation-only path.
# ``np.asarray`` narrows trimesh's ``ndarray | sympy.Matrix`` return, the numeric branch here.
_MATRIX_NP = np.asarray(tm.transformations.rotation_matrix(1.1, [0.3, 0.5, 0.81], [0.2, 0.1, 0.0]))
_MATRIX_WP = wp.mat44(*_MATRIX_NP.flatten().tolist())


@pytest.mark.benchmark(group="transform_points")
@pytest.mark.benchlibs("ordito", "trimesh", "open3d", "pyvista")
def test_transform_points(bench_case: BenchCase) -> None:
    """
    One 4x4 applied to every vertex: the suite's purest bandwidth row.

    ordito's side is a single ``wp.map`` over a ``@wp.func``, so the row is the launch floor at
    the small end and pure device bandwidth at the large end, with nothing in between to attribute
    a regression to. The three references each do strictly more work than the multiply -- see the
    module docstring for what each adds -- so all three are upper bounds.
    """
    if bench_case.kind == "trimesh":
        points_np = bench_case.vertices_np.astype(np.float64)
        moved_np = bench_case.run(
            lambda: tm.transformations.transform_points(points_np, _MATRIX_NP)
        )
        assert moved_np.shape == points_np.shape
        return
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        moved_pv = bench_case.run(lambda: mesh_pv.transform(_MATRIX_NP, inplace=False))
        assert moved_pv.n_points == bench_case.n_vertices
        return
    if bench_case.kind == "open3d":
        # `transform` mutates, so the cloud is rebuilt per round and the row prices that too.
        points_np = bench_case.vertices_np.astype(np.float64)
        moved_o3d = bench_case.run(
            lambda: o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np)).transform(
                _MATRIX_NP
            )
        )
        assert len(moved_o3d.points) == bench_case.n_vertices
        return
    vertices_wp = bench_case.vertices_wp
    moved_wp = bench_case.run(lambda: od.transform.transform_points(vertices_wp, _MATRIX_WP))
    assert moved_wp.shape == vertices_wp.shape
    assert np.isfinite(moved_wp.numpy()).all()


@pytest.mark.benchmark(group="transform_points_inplace")
@pytest.mark.benchlibs("ordito")
def test_transform_points_inplace(bench_case: BenchCase) -> None:
    """
    The same map with ``out=points``, which is what the allocation is worth.

    ordito-only on purpose: this is not a different algorithm from ``transform_points`` and no
    reference exposes the choice, so it is not a comparison -- it is the measurement behind the
    advice that [`Trimesh.transform`][ordito.mesh.Trimesh.transform] offers no in-place mode. The
    difference between the two rows is one buffer allocation, and it is what an in-place transform
    on the cached class could save while costing the whole cache.
    """
    scratch = wp.clone(bench_case.vertices_wp)
    moved_wp = bench_case.run(
        lambda: od.transform.transform_points(scratch, _MATRIX_WP, out=scratch)
    )
    assert moved_wp is scratch
    assert np.isfinite(scratch.numpy()).all()


@pytest.mark.benchmark(group="transform_mesh")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_transform_mesh(bench_case: BenchCase) -> None:
    """
    Vertices plus the winding flip a mirroring transform needs.

    The matrix is a **reflection**, so both sides take their orientation-reversing branch and the
    row prices the face-buffer rewrite rather than skipping it. trimesh's ``apply_transform``
    mutates and additionally invalidates and partly rebuilds its own cache, so it is an upper
    bound; the comparable claim is that both produce the same mesh, which
    ``tests/test_transform.py`` makes.
    """
    mirror_np = tm.transformations.reflection_matrix([0.2, 0.1, 0.0], [0.3, 0.5, 0.81])
    if bench_case.kind == "trimesh":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        moved_tm = bench_case.run(
            lambda: tm.Trimesh(vertices_np, faces_np, process=False).apply_transform(mirror_np)
        )
        assert len(moved_tm.faces) == bench_case.n_faces
        return
    vertices_wp, faces_wp = bench_case.vertices_wp, bench_case.faces_wp
    mirror_wp = wp.mat44(*mirror_np.flatten().tolist())
    moved_v, moved_f = bench_case.run(
        lambda: od.transform.transform_mesh(vertices_wp, faces_wp, mirror_wp)
    )
    assert moved_v.shape == vertices_wp.shape
    assert moved_f.shape == faces_wp.shape


@pytest.mark.benchmark(group="transform_normals")
@pytest.mark.benchlibs("ordito", "pyvista")
def test_transform_normals(bench_case: BenchCase) -> None:
    """
    The inverse-transpose map, which is a different quantity from the point map and not a variant.

    pyvista reaches it through ``transform(transform_all_input_vectors=True)``, which pushes the
    mesh's normal arrays through the inverse transpose alongside the points -- so its row includes
    the point transform and the topology copy on top of the normal map, and is an upper bound by a
    wide margin. It is here because it is the only reference in the suite that applies the
    covector map at all.
    """
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv.compute_normals(point_normals=True, cell_normals=False)
        moved_pv = bench_case.run(
            lambda: mesh_pv.transform(_MATRIX_NP, transform_all_input_vectors=True, inplace=False)
        )
        assert moved_pv.n_points == bench_case.n_vertices
        return
    normals_wp = od.vertices.vertex_normals(bench_case.vertices_wp, bench_case.faces_wp)
    moved_wp = bench_case.run(lambda: od.transform.transform_normals(normals_wp, _MATRIX_WP))
    assert moved_wp.shape == normals_wp.shape
    # Unit *or* zero: the scan meshes carry unreferenced vertices whose normal is zero, and the
    # map preserves zero by design rather than producing a direction out of nothing.
    lengths = np.linalg.norm(moved_wp.numpy(), axis=1)
    assert np.isclose(lengths[lengths > 0.0], 1.0, rtol=1e-4, atol=1e-4).all()
