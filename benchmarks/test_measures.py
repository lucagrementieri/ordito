"""
Benchmarks for ``ordito.measures``: the whole-mesh reductions, which are readback-bound.

Every group here ends in a host-side value, so each pays at least one device-to-host crossing on top
of its device pass -- and at the sizes in the scan sweep that latency is a large share of the total.
That is the axis these rows measure at the small end: each group is now one launch and one
crossing, so ``volume`` and ``surface_centroid`` read as the floor of a host-side reduction and
``moments`` as what its ``float64`` integrand adds on top.

What this axis is **not** is a licence to copy arrays: reducing per-face integrand buffers with
``.numpy().sum(axis=0)`` moves 72 bytes *per face* to be added on the host, where the kernels here
fold their sums on the device and send the host 8 to 80 bytes. Read the **min** in this group,
not the median: with the igl and trimesh rows sharing the process the ordito medians here run an
order of magnitude above its own min, the same instability
[`test_holes.py`](test_holes.py) documents for its DP rows.

``get_geometric_measures`` is pymeshlab's reference for the centroid and it does *more*: one
read-only call returns ``shell_barycenter`` (the area-weighted centroid ordito computes),
``barycenter`` (the plain vertex mean), the surface area, the mesh volume, the average edge length
and the inertia tensor. So that row is an **upper** bound on the centroid alone -- and the same
number appears as the ``mean_edge_length`` reference in [`test_edges.py`](test_edges.py), which is
worth knowing before reading either as a per-quantity cost. It is read-only, so it shares the
MeshSet.

``volume`` has **no group**; see the known-gaps table in ``README.md``.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
from meshlib import mrmeshpy as mm

import ordito as od
from conftest import BenchCase, skip_larger_than


@pytest.mark.benchmark(group="surface_centroid")
@pytest.mark.benchlibs("ordito", "trimesh", "pymeshlab", "meshlib")
def test_surface_centroid(bench_case: BenchCase) -> None:
    """
    The area-weighted shell centroid: one pass over the faces plus one host readback.

    One launch on both devices, ``surface_centroid_totals``, whose ``float32`` block sums are
    widened and committed into ``float64`` slots (the last block folds them in a fixed order with
    the deterministic setting on), and one read of the four sums -- the floor for a function whose
    return type is a host-side ``wp.vec3``.

    meshlib's ``findCenterFromFaces`` is this exact quantity and nothing more -- unlike the
    pymeshlab row above, which returns five measures at once. Note it has a sibling,
    ``findCenterFromPoints``, which is the plain vertex mean and belongs to
    [`points.centroid`][ordito.points.centroid]; tests/test_measures.py holds the two apart.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        centre_ml = bench_case.run(lambda: mm.findCenterFromFaces(mesh_ml.topology, mesh_ml.points))
        assert np.isfinite([*centre_ml]).all()
        return
    if bench_case.kind == "pymeshlab":
        # ``get_geometric_measures``' ``barycenter`` is the vertex mean and ``shell_barycenter`` the
        # area-weighted centroid ordito computes; the call returns both plus the area, volume and
        # inertia tensor, so it is an upper bound rather than an equivalent. Capped at ``bunny``:
        # it costs 1.12 s a call on ``dragon``, for a ratio the two medium meshes already establish.
        skip_larger_than(bench_case, "bunny", "get_geometric_measures is 1.12 s a call on dragon")
        meshset_pml = bench_case.meshset_pml
        assert bench_case.run(meshset_pml.get_geometric_measures)["shell_barycenter"].shape == (3,)
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: od.measures.surface_centroid(vertices, faces))
        assert np.isfinite(list(result)).all()
    else:  # numpy reference: the uncached formula behind ``trimesh.Trimesh.centroid``
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> np.ndarray:
            triangles = vertices[faces]
            crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            areas = 0.5 * np.linalg.norm(crosses, axis=1)
            return (triangles.mean(axis=1) * areas[:, None]).sum(axis=0) / areas.sum()

        result = bench_case.run(run)
        assert result.shape == (3,)


@pytest.mark.benchmark(group="moments")
@pytest.mark.benchlibs("ordito", "igl", "trimesh", "pyvista", "meshlib")
def test_moments(bench_case: BenchCase) -> None:
    """
    Volume, centre of mass and inertia tensor: ten ``float64`` sums over the faces.

    One launch and one read of the ten sums, like ``surface_centroid`` above; the gap between the
    two is the ``float64`` integrand (1/64 of the ``float32`` rate on the RTX 5090), which sets the
    row from the mid-size scan meshes up.

    ``igl.moments`` returns the first moment un-normalised and the inertia already about the centre
    of mass; ``trimesh``'s ``mass_properties`` computes the same three from the same integrals on
    the host. Both are timed on the whole call, since neither exposes the integrals separately.

    **pyvista answers the volume alone**, so its row does strictly less: ``PolyData.volume`` is the
    same divergence integral over the faces with no centre of mass and no inertia tensor. Unlike
    open3d's it does *not* validate first, which is what makes it safe to time here at all.

    There is no open3d row, measured rather than assumed: ``get_volume`` validates before it
    integrates, and the validation is the same brute-force ``IsWatertight`` composition its
    ``is_watertight`` row times -- seconds, on a mesh whose integral is microseconds. A row here
    would re-time ``is_watertight`` under this group's name (the same trap the validation module
    documents for ``is_volume``), and it raises outright on the non-watertight sweep meshes.

    The lesson the group is kept for: **a plausible cost model is not a measurement.** "Readback-
    bound" was true of this row and still hid a bytes-moved bug -- three ``.numpy().sum(axis=0)``
    reductions over the per-face ``vec3d`` integrands, 72 bytes per face rather than per return --
    for as long as nobody checked which of the two the number was. A caller wanting only the volume
    should still call [`volume`][ordito.measures.volume], which pays one crossing.
    """
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        assert np.isfinite(bench_case.run(lambda: float(mesh_pv.volume)))
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        volume, _center, inertia = bench_case.run(lambda: od.measures.moments(vertices, faces))
        assert np.isfinite(volume)
        assert np.asarray(inertia).reshape(3, 3).shape == (3, 3)
        return
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    if bench_case.kind == "igl":
        volume_igl, first_igl, inertia_igl = bench_case.run(
            lambda: igl.moments(vertices_np, faces_np)
        )
        assert np.isfinite(volume_igl)
        assert np.asarray(first_igl).shape == (3,)
        assert np.asarray(inertia_igl).shape == (3, 3)
        return
    properties_tm = bench_case.run(
        lambda: tm.Trimesh(vertices_np, faces_np, process=False).mass_properties
    )
    assert np.asarray(properties_tm["inertia"]).shape == (3, 3)


@pytest.mark.benchmark(group="volume")
@pytest.mark.benchlibs("ordito")
def test_volume(bench_case: BenchCase) -> None:
    """
    The signed volume alone: one fused pass over the faces and one readback.

    Its references sit in the ``moments`` group, where the integral family is timed against them;
    this row is the cheapest call with a block commit, on its arrival-order form.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    assert np.isfinite(bench_case.run(lambda: od.measures.volume(vertices, faces)))


@pytest.mark.benchmark(group="euler_characteristic")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_euler_characteristic(bench_case: BenchCase) -> None:
    """
    ``V - E + F`` over the referenced vertices and the unique edges: one launch, one readback.

    The vertex marks and an open-addressing edge set homed at each edge's lower vertex count
    their first arrivals in one pass (no sort), and both counts come back in one read; the row
    passes the vertex count, so the index-bound read of the inferred path is not timed. The
    trimesh row is the uncached ``euler_number`` on a fresh ``Trimesh`` (its own ``edges_unique``
    grouping and referenced-vertex count).
    """
    if bench_case.kind == "ordito":
        faces = bench_case.faces_wp
        result = bench_case.run(
            lambda: od.measures.euler_characteristic(faces, n_vertices=bench_case.n_vertices)
        )
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(
            lambda: tm.Trimesh(vertices_np, faces_np, process=False).euler_number
        )
    assert isinstance(int(result), int)
