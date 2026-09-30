"""
Benchmarks for ``triwarp.interpolation``: averaging scatters, barycentric pull, cloud interpolation.

Three axes, because the module holds three different kinds of function:

* **valence** for the three ``average_*`` scatters. Each is ``3F`` atomic adds into ``V`` slots (or
  ``3F`` gathers into ``F`` for the face direction), so in principle a mesh with a few very
  high-valence hubs serialises where a regular one does not. ``fan_hub`` is the extreme: identical
  vertex *and* face count to ``sphere_med``, but its cone apex and base centre have valence 40 960
  against a uniform 6. The finding these rows exist to keep visible is that it does **not** cost
  much -- see ``test_vertices.py``, whose normals scatters read under 2x across the same axis.
* **scale** for ``transfer_onto_vertices``, which is not a scatter at all: it is a closest-point
  query per target vertex plus a barycentric blend, so its cost is the BVH's and nothing this module
  owns.
* **neighbourhood size** for ``interpolate_from_points``, whose kernel reduces each query's mean
  during the hash-grid walk -- so the row is really the number of neighbours visited, and the
  radius is derived from the mean edge length to hold the neighbour count fixed across meshes;
  triwarp leads pyvista by well over an order of magnitude at both widths. No neighbour list is
  materialized, so both widths run on every mesh: the 64-neighbour case over lucy used to be a
  ball-query CSR of several gigabytes.

``average_onto_vertices`` and ``transfer_onto_vertices`` were previously timed in
[`test_vertices.py`](test_vertices.py). They moved here with their group names unchanged, because
the ``benchmark(group=)`` name is a cross-suite key every ``parity`` marker cites -- renaming one
would break the gate rather than relocate a row.

References
----------
**libigl** covers all three averaging directions and is the reference this module was written
against (``tests/test_interpolation.py`` has used it as the oracle from the start; these rows add
the missing timings):

* ``igl.average_onto_faces(F, S)`` is the vertex-to-face mean;
* ``igl.average_onto_vertices(V, F, S)`` is the face-to-vertex mean -- note its ``S`` is a
  per-face **scalar**, not a per-face vector, which is the one signature trap here;
* ``igl.average_from_edges_onto_vertices(F, E, oE, uE)`` needs the ``(E, oE)`` halfedge numbering
  from ``igl.orient_halfedges(F)``, which is also what triwarp's edge-based overload consumes, so
  both sides receive the identical edge indexing rather than each building its own.

**pymeshlab**'s ``compute_scalar_transfer_face_to_vertex(areaweight=False)`` is the plain corner
mean of ``average_onto_vertices``, and ``transfer_attributes_per_vertex(qualitytransfer=True)`` is
the barycentric pull. Both read and write mesh *attributes* rather than taking arrays, so each row
seeds the input attribute and reads the output one; the transfer needs **two** meshes in the set and
writes into the second, so that row builds a fresh two-mesh MeshSet inside the timed callable --
twice the per-vertex build cost, which on a scan mesh is most of the row before any transfer
happens.

**pyvista**'s ``DataSet.interpolate`` is ``interpolate_from_points``'s reference: a
``vtkPointInterpolator`` with a ``vtkGaussianKernel``, given the identical radius and sharpness, so
the two compute the same weighted mean and differ only in the locator (a
``vtkStaticPointLocator`` against triwarp's hash grid). It is the module's only pyvista row -- VTK
has no per-element averaging filter, so the three ``average_*`` groups keep libigl.

trimesh and open3d have no equivalent for any of the four mesh-side functions: attribute averaging
over a mesh's incidence structure is not something either exposes as a function.
"""

from __future__ import annotations

import igl
import numpy as np
import pymeshlab as ml
import pytest
import pyvista as pv
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from conftest import BenchCase, skip_larger_than

_FIELD_SEED = 5

_face_field_cache: dict[tuple[str, str], wp.array[wp.float32]] = {}
_vertex_field_cache: dict[tuple[str, str], wp.array[wp.float32]] = {}
_edge_field_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_edge_length_cache: dict[tuple[str, str], float] = {}


def _face_field_np(bench_case: BenchCase) -> np.ndarray:
    """One scalar per face: the barycentre's z, so the values are deterministic and vary."""
    return np.ascontiguousarray(
        bench_case.vertices_np[bench_case.faces_np][..., 2].mean(axis=1), dtype=np.float64
    )


def _face_field_wp(bench_case: BenchCase) -> wp.array[wp.float32]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _face_field_cache:
        _face_field_cache[key] = wp.array(
            np.ascontiguousarray(_face_field_np(bench_case), dtype=np.float32),
            dtype=wp.float32,
            device=bench_case.device,
        )
    return _face_field_cache[key]


def _mean_edge_length(bench_case: BenchCase) -> float:
    """Mean undirected edge length, the scale every radius in this module is derived from."""
    key = (bench_case.mesh_name, "edge-length")
    if key not in _edge_length_cache:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        corners_np = vertices_np[faces_np]
        lengths_np = np.linalg.norm(corners_np - np.roll(corners_np, 1, axis=1), axis=2)
        _edge_length_cache[key] = float(lengths_np.mean())
    return _edge_length_cache[key]


def _vertex_field_np(bench_case: BenchCase) -> np.ndarray:
    """One scalar per vertex: its own z."""
    return np.ascontiguousarray(bench_case.vertices_np[:, 2], dtype=np.float64)


def _vertex_field_wp(bench_case: BenchCase) -> wp.array[wp.float32]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _vertex_field_cache:
        _vertex_field_cache[key] = wp.array(
            np.ascontiguousarray(_vertex_field_np(bench_case), dtype=np.float32),
            dtype=wp.float32,
            device=bench_case.device,
        )
    return _vertex_field_cache[key]


def _edge_numbering(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``igl.orient_halfedges``' ``(E, oE)`` plus a per-unique-edge field -- all *inputs*.

    Both sides consume this numbering, so it is built once with igl and shared rather than each
    library indexing the edges its own way: the row is about the averaging, not the edge sort.
    """
    key = (bench_case.mesh_name, "edges")
    if key not in _edge_field_cache:
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        edges_np, orientation_np = igl.orient_halfedges(faces_np)
        edges_np = np.asarray(edges_np)
        rng = np.random.default_rng(_FIELD_SEED)
        values_np = np.ascontiguousarray(rng.uniform(size=int(edges_np.max()) + 1))
        _edge_field_cache[key] = (edges_np, np.asarray(orientation_np), values_np)
    return _edge_field_cache[key]


@pytest.mark.benchmark(group="average_onto_faces")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("triwarp", "igl", "pyvista")
def test_average_onto_faces(bench_case: BenchCase) -> None:
    """
    Vertex-to-face mean: a ``3F`` **gather**, so the one direction valence cannot hurt.

    It is on the valence axis anyway, as the control for the two scatters below: this row reads
    three vertices per face and writes one value with no contention at all, so any spread here is
    noise and any spread *there* is the atomics. Without it the scatters' flatness has nothing to be
    flat against.

    VTK spells it ``point_data_to_cell_data``; the field is seeded on the shared ``PolyData`` once,
    outside the timed region, since what a gather costs depends on the indices and not the values.
    """
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        mesh_pv.point_data["field"] = _vertex_field_np(bench_case).astype(np.float64)
        transferred_pv = bench_case.run(mesh_pv.point_data_to_cell_data)
        assert np.asarray(transferred_pv.cell_data["field"]).shape == (bench_case.n_faces,)
        return
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        values = _vertex_field_wp(bench_case)
        result = bench_case.run(lambda: tw.interpolation.average_onto_faces(faces, values))
        assert result.shape == (bench_case.n_faces,)
        return
    faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
    values_np = _vertex_field_np(bench_case)
    result_igl = bench_case.run(lambda: igl.average_onto_faces(faces_np, values_np))
    assert result_igl.shape == (bench_case.n_faces,)


@pytest.mark.benchmark(group="average_onto_vertices")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("triwarp", "igl", "pymeshlab", "pyvista")
def test_average_onto_vertices(bench_case: BenchCase) -> None:
    """
    The face-to-vertex scatter: the same contention as the normals rows, without the math.

    VTK's ``cell_data_to_point_data`` needs no weighting flag -- it is the unweighted incident-cell
    mean, which is this function.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        mesh_pv.cell_data["field"] = _face_field_np(bench_case).astype(np.float64)
        transferred_pv = bench_case.run(mesh_pv.cell_data_to_point_data)
        assert np.asarray(transferred_pv.point_data["field"]).shape == (n_vertices,)
        return
    if bench_case.kind == "pymeshlab":
        # MeshLab transfers whatever is in the face scalar attribute, so it is seeded once (values
        # do not change what a scatter costs, only the indices do) and ``areaweight=False`` gives
        # the plain corner mean this function computes.
        meshset_pml = bench_case.meshset_pml
        meshset_pml.compute_scalar_by_function_per_face(q="z0")
        bench_case.run(lambda: meshset_pml.compute_scalar_transfer_face_to_vertex(areaweight=False))
        assert meshset_pml.current_mesh().vertex_scalar_array().shape == (n_vertices,)
        return
    if bench_case.kind == "igl":
        # ``S`` is a per-face *scalar*, not a per-face vector -- the signature trap for this call.
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        values_np = _face_field_np(bench_case)
        result_igl = bench_case.run(
            lambda: igl.average_onto_vertices(vertices_np, faces_np, values_np)
        )
        assert result_igl.ravel().shape == (n_vertices,)
        return
    faces = bench_case.faces_wp
    values = _face_field_wp(bench_case)
    result = bench_case.run(
        lambda: tw.interpolation.average_onto_vertices(n_vertices, faces, values)
    )
    assert result.shape == (n_vertices,)


@pytest.mark.benchmark(group="average_from_edges_onto_vertices")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("triwarp", "igl")
def test_average_from_edges_onto_vertices(bench_case: BenchCase) -> None:
    """
    The edge-to-vertex scatter, over ``igl.orient_halfedges``' edge numbering.

    Both sides are handed the same ``(E, oE)`` tables and the same per-edge field, so this times the
    averaging alone -- the edge numbering is an input, built once outside the timed callable. It is
    the third of the module's three directions and the only one whose *source* is neither vertices
    nor faces, which is why it needs a row of its own rather than being read off the other two.
    """
    n_vertices = bench_case.n_vertices
    edges_np, orientation_np, values_np = _edge_numbering(bench_case)
    if bench_case.kind == "igl":
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        result_igl = bench_case.run(
            lambda: igl.average_from_edges_onto_vertices(
                faces_np, edges_np, orientation_np, values_np
            )
        )
        assert result_igl.ravel().shape == (n_vertices,)
        return
    device = bench_case.device
    faces = bench_case.faces_wp
    edges_wp = twt.as_array2d(
        wp.array(edges_np.astype(np.int32), dtype=wp.int32, device=device), wp.int32
    )
    orientation_wp = twt.as_array2d(
        wp.array(orientation_np.astype(np.int32), dtype=wp.int32, device=device), wp.int32
    )
    values_wp = wp.array(
        np.ascontiguousarray(values_np, dtype=np.float32), dtype=wp.float32, device=device
    )
    result = bench_case.run(
        lambda: tw.interpolation.average_from_edges_onto_vertices(
            n_vertices, faces, edges_wp, orientation_wp, values_wp
        )
    )
    assert result.shape == (n_vertices,)


@pytest.mark.benchmark(group="transfer_onto_vertices")
@pytest.mark.benchlibs("triwarp", "pymeshlab", "pyvista")
def test_transfer_onto_vertices(bench_case: BenchCase) -> None:
    """
    Closest-point plus barycentric blend, transferring a field from a mesh onto itself.

    pyvista's ``sample`` interpolates a source field onto a target's points through a cell locator,
    which is this operation -- and on a target that coincides with the source it is exact: measured
    642 of 642 points valid and 5.96e-08 from triwarp's answer (``tests/test_interpolation.py``).
    Two things it does that this group's other rows do not: it marks misses in
    ``vtkValidPointMask`` rather than extrapolating, and it interpolates only where the query lands
    *inside* a source cell, so on a target offset from the surface its valid fraction falls sharply.
    The self-transfer here is the input class where the three rows measure the same work.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny", "VTK's probe filter is a serial cell-locator walk")
        source_pv = bench_case.mesh_pv.copy()
        source_pv.point_data["field"] = _vertex_field_np(bench_case).astype(np.float64)
        target_pv = pv.PolyData(bench_case.vertices_np)
        sampled_pv = bench_case.run(lambda: target_pv.sample(source_pv))
        assert sampled_pv.point_data["field"].shape[0] == n_vertices
        return
    if bench_case.kind == "pymeshlab":
        skip_larger_than(bench_case, "bunny", "MeshLab's transfer is a serial closest-point walk")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        faces_i32 = np.ascontiguousarray(faces_np, dtype=np.int32)
        values_np = _vertex_field_np(bench_case)

        def transfer_pml() -> int:
            meshset_pml = ml.MeshSet()
            meshset_pml.add_mesh(ml.Mesh(vertices_np, faces_i32, v_scalar_array=values_np))
            meshset_pml.add_mesh(ml.Mesh(vertices_np, faces_i32))
            meshset_pml.transfer_attributes_per_vertex(
                sourcemesh=0,
                targetmesh=1,
                qualitytransfer=True,
                colortransfer=False,
                upperbound=ml.PercentageValue(50),
            )
            meshset_pml.set_current_mesh(1)
            return meshset_pml.current_mesh().vertex_scalar_array().shape[0]

        assert bench_case.run(transfer_pml) == n_vertices
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    values = _vertex_field_wp(bench_case)
    transferred, _distance = bench_case.run(
        lambda: tw.interpolation.transfer_onto_vertices(vertices, faces, values, vertices)
    )
    assert transferred.shape == (n_vertices,)


@pytest.mark.benchmark(group="transfer_through_operator")
@pytest.mark.benchlibs("triwarp")
def test_transfer_through_operator(bench_case: BenchCase) -> None:
    """
    Carry a per-vertex field through one Loop pass, using the pass's own interpolation operator.

    Two numbers worth having next to each other, and the group times only the second: assembling the
    operator (``subdivide_loop(return_operator=True)``, done in ``setup`` so it is *not* timed) and
    applying it to a field. Read the timed number against ``transfer_onto_vertices`` above, which
    answers the same question -- "the field, on the subdivided mesh" -- through a closest-point
    projection instead, and is both lossy and far more expensive because it walks a BVH per vertex.

    **No reference row.** No installed library returns a subdivision's interpolation matrix at all,
    which is the gap the operator closes; ``tests/test_remesh.py`` therefore carries an
    invariant-only claim (the operator reproduces the pass's own positions) rather than a library
    comparison, and this group contributes no parity pair by construction.

    Flat across the mesh pair, because the row is one CSR pass and the field is small next to the
    launch floor. ``transfer_onto_vertices`` on the same field is several times dearer, so the
    operator is cheaper *and* exact where the projection is lossy; the projection's advantage is
    that it needs no operator, which is the trade the two docstrings state.
    """
    skip_larger_than(bench_case, "bunny", "one Loop pass quadruples the face count")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    values = _vertex_field_wp(bench_case)

    def build() -> twt.BsrMatrix[wp.float32]:
        _new_vertices, _new_faces, operator = tw.remesh.subdivide_loop(
            vertices, faces, return_operator=True
        )
        return operator

    n_out = int(build().nrow)
    transferred = bench_case.run(
        lambda operator: tw.interpolation.transfer_through_operator(values, operator), setup=build
    )
    assert transferred.shape == (n_out,)


@pytest.mark.benchmark(group="interpolate_from_points")
@pytest.mark.benchlibs("triwarp", "pyvista")
@pytest.mark.parametrize("neighborhood", [8, 64])
def test_interpolate_from_points(bench_case: BenchCase, neighborhood: int) -> None:
    """
    Scattered-cloud interpolation, on the axis that governs it: the neighbourhood size.

    The mesh's own vertices are both the source cloud and the queries, so the row scales with the
    mesh; the radius is set from the mean edge length so that a query sees roughly ``neighborhood``
    sources on every mesh, which is what makes the two points comparable across the sweep rather
    than measuring the cloud's density. The kernel reduces each query's mean during the hash-grid
    walk, so the sweep is really over the number of neighbours visited. The grid's default
    resolution grows with the cloud past about a million points: lucy occupies about twice as many
    cells as a 128-bin grid has buckets, so at that resolution every probe also walked the points
    of an unrelated cell.

    pyvista's ``DataSet.interpolate`` is ``vtkPointInterpolator`` with the identical Gaussian kernel
    and the same radius, its locator being a ``vtkStaticPointLocator`` where triwarp uses a hash
    grid.
    """
    radius = float(np.sqrt(neighborhood / np.pi)) * _mean_edge_length(bench_case)
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny", "VTK's point interpolator is a serial locator walk")
        source_pv = pv.PolyData(np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64))
        source_pv.point_data["v"] = _vertex_field_np(bench_case)
        query_pv = pv.PolyData(np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64))
        interpolated_pv = bench_case.run(
            lambda: query_pv.interpolate(source_pv, radius=radius, sharpness=2.0)
        )
        assert interpolated_pv.n_points == bench_case.n_vertices
        return
    points, values = bench_case.vertices_wp, _vertex_field_wp(bench_case)
    interpolated_wp = bench_case.run(
        lambda: tw.interpolation.interpolate_from_points(points, values, points, radius)
    )
    assert interpolated_wp.shape == (bench_case.n_vertices,)
