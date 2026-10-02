"""
Benchmarks for ``ordito.tangent_space``: frames, polar angles and transport.

Three groups over two axes:

* **valence** (``sphere_med`` -> ``fan_hub``, ``V`` and ``F`` pinned) for
  ``halfedge_tangent_angles``, which walks each vertex's ring twice in one thread -- once to total
  the corner angles, once to lay them out. It inherits ``vertex_one_rings``'s exposure to a single
  wide hub, and this is where that shows.
* **scale** for the frames and the transport angles, which are per-vertex and per-halfedge
  arithmetic with no walk in them.

``halfedge_tangent_angles`` spreads by more than an order of magnitude on the valence axis -- more
exposed than ``vertex_one_rings`` on the same two meshes, because it walks the ring twice (once to
total the corner angles, once to lay them out) and the hub's walk is serial both times.
``vertex_tangent_frames`` is flat over the scale axis where potpourri3d is superlinear, and
``halfedge_transport_angles`` is flat too: given the polar angles it is one read per halfedge and
nothing else.

potpourri3d is the reference, but note what its row includes: its tangent frames are a by-product of
constructing ``MeshVectorHeatSolver``, which also builds the halfedge mesh, the cotangent Laplacian
and the connection Laplacian, and factors them. There is no way to ask it for the frames alone. So
its number is an **upper bound** on the frame cost, and a fair number only for the question "what
does it cost to get to the point of having tangent frames". ``use_intrinsic_delaunay=False`` keeps
its discretization the same as ordito's.

That same construction is the only route to its connection Laplacian, which is what
``halfedge_transport_angles`` corresponds to (ordito will assemble that matrix itself in a later
phase); until then the transport-angle row is ordito-only, with potpourri3d's construction cost
shown in the frames group rather than double-counted here.

**trimesh**, **libigl** and **open3d** have no tangent-space machinery at all: none exposes a
per-vertex 2D basis, a rotational halfedge order or a connection. libigl's ``igl.local_basis`` is
per-*face* (a frame per triangle, from its own edge vectors) and needs no ring walk or intrinsic
flattening, so it answers a different question and is not comparable.
"""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest

import ordito as od
from conftest import BenchCase

# Constructing potpourri3d's vector-heat solver factors two sparse systems; it runs into hundreds of
# milliseconds on the larger meshes, so fewer rounds.
_ROUNDS = 3


@pytest.mark.benchmark(group="vertex_tangent_frames")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "potpourri3d")
def test_vertex_tangent_frames(bench_case: BenchCase) -> None:
    """Angle-weighted normals plus a reference direction per vertex."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        basis_x, _, _ = bench_case.run(
            lambda: od.tangent_space.vertex_tangent_frames(vertices, faces), rounds=_ROUNDS
        )
        assert basis_x.shape == (n_vertices,)
    else:
        # The frames come out of the vector-heat solver's construction, which also assembles and
        # factors the cotangent and connection Laplacians: an upper bound, not a like-for-like row.
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)

        def frames_pp() -> np.ndarray:
            solver = pp3d.MeshVectorHeatSolver(vertices_np, faces_np, use_intrinsic_delaunay=False)
            return np.asarray(solver.get_tangent_frames()[0])

        assert bench_case.run(frames_pp, rounds=_ROUNDS).shape == (n_vertices, 3)


@pytest.mark.benchmark(group="face_tangent_frames")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "igl")
def test_face_tangent_frames(bench_case: BenchCase) -> None:
    """
    A frame per face: one edge normalize and one cross product, plus the face normals.

    Read it against ``vertex_tangent_frames`` above, which is the same idea one dimension up and an
    order of magnitude more work: a vertex frame needs the angle-weighted normal and the one-ring
    walk that picks its reference halfedge, where a face frame needs only the face's own first edge.
    That contrast is the reason both exist -- and why this one is not gauge-dependent.

    ``igl.local_basis`` uses the identical convention (first edge, then ``normal x basis_x``), so
    unlike every other row in this module the two sides are element-wise comparable rather than
    comparable up to a rotation; ``tests/test_tangent_space.py`` asserts exactly that.
    """
    n_faces = bench_case.n_faces
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        basis_x, _basis_y, _normals = bench_case.run(
            lambda: od.tangent_space.face_tangent_frames(vertices, faces)
        )
        assert basis_x.shape == (n_faces,)
        return
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    basis_x_igl, _basis_y_igl, _normal_igl = bench_case.run(
        lambda: igl.local_basis(vertices_np, faces_np)
    )
    assert basis_x_igl.shape == (n_faces, 3)


@pytest.mark.benchmark(group="halfedge_tangent_angles")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("ordito")
def test_halfedge_tangent_angles(bench_case: BenchCase) -> None:
    """Two serial ring walks per vertex: the valence-sensitive group of this module."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    angles = bench_case.run(lambda: od.tangent_space.halfedge_tangent_angles(vertices, faces))
    assert angles.shape == (faces.size,)


@pytest.mark.benchmark(group="halfedge_transport_angles")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito")
def test_halfedge_transport_angles(bench_case: BenchCase) -> None:
    """The connection phases, given the polar angles: one flat pass over the halfedges."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    n_vertices = bench_case.n_vertices
    twins = od.halfedge.halfedge_twins(faces, n_vertices=n_vertices)
    rings = od.halfedge.vertex_one_rings(faces, twins=twins, n_vertices=n_vertices)
    angles = od.tangent_space.halfedge_tangent_angles(vertices, faces, rings=rings)
    rho = bench_case.run(
        lambda: od.tangent_space.halfedge_transport_angles(
            vertices, faces, twins=twins, tangent_angles=angles
        )
    )
    assert rho.shape == (faces.size,)
