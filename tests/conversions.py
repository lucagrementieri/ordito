"""Shared mesh format conversions for tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol, SupportsIndex, TypeVar, cast

import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytorch3d.structures as p3d_structures
import pyvista as pv
import scipy.sparse as sp
import torch
import trimesh as tm
import warp as wp
import warp.sparse as wps
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from pymeshfix import _meshfix
from scipy.spatial import cKDTree

import ordito.typing as odt

if TYPE_CHECKING:
    from typing_extensions import Buffer

DType = TypeVar("DType")


class _MeshlibScalars(Protocol):
    """A MeshLib scalar container: iterable, with a ``size()`` method and no ``len()``."""

    def __iter__(self) -> Iterator[float]: ...
    def size(self) -> int: ...


class _MeshlibIndices(Protocol):
    """A MeshLib id container, whose elements implement ``__index__``."""

    def __iter__(self) -> Iterator[SupportsIndex]: ...
    def size(self) -> int: ...


def warp_empty(
    shape: int | tuple[int, ...], dtype: type[DType], device: wp.DeviceLike
) -> wp.array[DType, Any]:
    """
    ``wp.empty`` with its ``dtype`` carried into the return type.

    Warp leaves ``wp.empty``'s ``dtype`` parameter unannotated, so a checker types it from its
    ``float`` default and rejects every other element type. This is the one place that says so.
    """
    return wp.empty(shape, dtype=cast("type[float]", dtype), device=device)


def trimesh_to_warp(mesh: tm.Trimesh, device: wp.DeviceLike) -> wp.Mesh:
    vertices = wp.array(
        np.ascontiguousarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces = wp.array(
        np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    return wp.Mesh(points=vertices, indices=faces)


def numpy_to_warp(
    vertices_np: np.ndarray, faces_np: np.ndarray, device: wp.DeviceLike
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Upload a NumPy mesh as ordito's ``(wp.array[wp.vec3], flat wp.array[wp.int32])`` pair.

    The single most duplicated helper in this suite: six private copies across six modules and 54
    call sites, differing only in where the ``float32`` cast sat. It is separate from
    [`trimesh_to_warp`][tests.conversions.trimesh_to_warp], which returns a ``wp.Mesh`` (a BVH
    build), because most tests want the raw buffers a ordito wrapper takes and never touch a
    ``wp.Mesh``.

    Positions land as **float32**: that is what ``wp.vec3`` holds, and it is the reason a comparison
    against a float64 reference bottoms out around 1e-7 rather than at machine epsilon.

    Parameters
    ----------
    vertices_np
        ``(n, 3)`` positions, any float dtype.
    faces_np
        Vertex indices, ``(n_faces, 3)`` or already flat -- reshaped to ordito's flat buffer
        either way.
    device
        Warp device for both arrays.

    See Also
    --------
    [`warp_to_trimesh`][tests.conversions.warp_to_trimesh]
        The inverse, for reading a ordito result back out.
    [`numpy_to_warp_uv`][tests.conversions.numpy_to_warp_uv]
        The ``wp.vec2`` form, for the parametrization tests' 2-D vertex buffers.
    """
    return (
        wp.array(np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec3, device=device),
        wp.array(
            np.ascontiguousarray(np.asarray(faces_np).reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        ),
    )


def numpy_to_warp_uv(
    uv_np: np.ndarray, faces_np: np.ndarray, device: wp.DeviceLike
) -> tuple[wp.array[wp.vec2], wp.array[wp.int32]]:
    """
    Upload a 2-D vertex buffer and its faces as ``(wp.array[wp.vec2], flat wp.array[wp.int32])``.

    The [`numpy_to_warp`][tests.conversions.numpy_to_warp] of the parametrization tests, whose
    "vertices" are UV coordinates in the plane. Separate rather than a ``dtype=`` switch because the
    two are never interchangeable at a call site: a function taking a UV atlas will not accept
    positions, and a silently-wrong vector width is exactly the kind of mistake a shared helper
    should make impossible.
    """
    return (
        wp.array(np.ascontiguousarray(uv_np, dtype=np.float32), dtype=wp.vec2, device=device),
        wp.array(
            np.ascontiguousarray(np.asarray(faces_np).reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        ),
    )


def points_to_warp(points_np: np.ndarray, device: wp.DeviceLike) -> wp.array[wp.vec3]:
    """
    Upload an ``(n, 3)`` point cloud as ``wp.array[wp.vec3]``.

    The vertices-only half of [`numpy_to_warp`][tests.conversions.numpy_to_warp], and the sibling of
    [`points_to_open3d`][tests.conversions.points_to_open3d],
    [`points_to_pymeshlab`][tests.conversions.points_to_pymeshlab],
    [`points_to_pyvista`][tests.conversions.points_to_pyvista] and
    [`points_to_meshlib`][tests.conversions.points_to_meshlib] -- every reference library had one
    and Warp did not, so the suite hand-rolled it **403** times in four different spellings across
    35 files, plus six one-line private copies carrying another 181 calls. Reach for this wherever
    a bare cloud goes to a ordito wrapper: query points, normals, ray origins and directions,
    polyline vertices, a sampled surface.

    Positions land as **float32**: that is what ``wp.vec3`` holds, and it is the reason a comparison
    against a float64 reference bottoms out around 1e-7 rather than at machine epsilon. The
    ``np.ascontiguousarray`` is not ceremony *here* even though Warp accepts a non-contiguous
    payload (probed on 1.16) -- it costs nothing on an already-contiguous array and it keeps one
    spelling where there were four. Note the contrast with the *index*-array hazard, which
    is real and unrelated: a stride is silently ignored on a gather **index**, never on a payload
    upload.

    Parameters
    ----------
    points_np
        ``(n, 3)`` positions or directions, any float dtype.
    device
        Warp device for the result.

    See Also
    --------
    [`points_to_warp_uv`][tests.conversions.points_to_warp_uv]
        The ``wp.vec2`` form, for planar clouds and UV buffers.
    [`numpy_to_warp`][tests.conversions.numpy_to_warp]
        The ``(vertices, faces)`` form, when the cloud is a mesh.
    """
    return wp.array(np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=device)


def points_to_warp_uv(points_np: np.ndarray, device: wp.DeviceLike) -> wp.array[wp.vec2]:
    """
    Upload an ``(n, 2)`` planar point set as ``wp.array[wp.vec2]``.

    The ``wp.vec2`` counterpart of [`points_to_warp`][tests.conversions.points_to_warp], for a UV
    atlas, a 2-D polygon ring or a planar Delaunay input. Separate rather than a ``dtype=`` switch
    for the reason [`numpy_to_warp_uv`][tests.conversions.numpy_to_warp_uv] gives: the two are
    never interchangeable at a call site, and a silently-wrong vector width is exactly what a shared
    helper should make impossible.
    """
    return wp.array(np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec2, device=device)


def trimesh_to_pymeshlab(mesh: tm.Trimesh, scalars: np.ndarray | None = None) -> ml.MeshSet:
    """
    Wrap a ``tm.Trimesh`` in a fresh single-mesh ``pymeshlab.MeshSet``.

    MeshLab wants float64 positions; the ``(n_faces, 3)`` index array goes in as-is. The returned
    MeshSet is **not** reusable across filters: almost every one of them mutates ``current_mesh()``
    in place, so build a new one per comparison rather than threading one through a test.

    ``scalars`` seeds the per-vertex scalar attribute, which the ``*_per_vertex`` scalar filters
    read and write in place: they take no array argument, so a comparison against one has to put
    its input here rather than pass it.
    """
    meshset = ml.MeshSet()
    meshset.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh.faces, dtype=np.int32),
            **(
                {}
                if scalars is None
                else {"v_scalar_array": np.ascontiguousarray(scalars, dtype=np.float64)}
            ),
        )
    )
    return meshset


def warp_to_pymeshlab(vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]) -> ml.MeshSet:
    """
    Read ordito's ``(vertices, flat faces)`` pair back into a ``pymeshlab.MeshSet``.

    The face buffer is ordito's flat one, so it is reshaped to ``(n_faces, 3)`` here; use this when
    the mesh under test is a ordito *output* rather than one of the ``tests/conftest.py`` fixtures
    (which already carry a ``tm.Trimesh`` for ``trimesh_to_pymeshlab``).

    !!! note "Deliberately unexercised, and not dead"
        No test calls this today, and that is a decision rather than an oversight -- the
        inventory of helpers points readers at it, so deleting it invites the next person to
        hand-roll the wrapper and rediscover the hazards above. It also has a caller that is
        invisible from here: ``tests/parity.py`` keys its reference detection off these
        converter *names*, so removing one would silently narrow the parity scan.
    """
    meshset = ml.MeshSet()
    meshset.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(vertices_wp.numpy(), dtype=np.float64),
            np.ascontiguousarray(faces_wp.numpy().reshape(-1, 3), dtype=np.int32),
        )
    )
    return meshset


def wedge_uv_to_pymeshlab(
    vertices_np: np.ndarray, faces_np: np.ndarray, wedge_uv_np: np.ndarray
) -> ml.MeshSet:
    """
    Build a MeshSet carrying a **per-wedge** (per-corner) texture atlas.

    MeshLab stores UVs on face corners rather than on vertices, which is why it has no texcoord
    *index* buffer at all and why its seam predicate compares coordinates. ``w_tex_coords_matrix``
    takes exactly ordito's per-corner layout, ``(3 * n_faces, 2)`` in ``3 * f + k`` order, so any
    numpy atlas can drive ``compute_selection_by_texture_seams_per_vertex`` without going through a
    file.
    """
    meshset = ml.MeshSet()
    meshset.add_mesh(
        ml.Mesh(
            vertex_matrix=np.ascontiguousarray(vertices_np, dtype=np.float64),
            face_matrix=np.ascontiguousarray(faces_np, dtype=np.int32),
            w_tex_coords_matrix=np.ascontiguousarray(wedge_uv_np, dtype=np.float64),
        )
    )
    return meshset


def points_to_pymeshlab(points_np: np.ndarray, normals_np: np.ndarray | None = None) -> ml.MeshSet:
    """
    Wrap a bare point cloud in a **face-less** single-mesh ``pymeshlab.MeshSet``.

    A handful of MeshLab filters require a mesh with vertices and no faces --
    ``compute_normal_for_point_clouds`` refuses anything else, and the query layer of
    ``compute_scalar_by_distance_from_another_mesh_per_vertex`` wants the sample points on their own
    -- which neither [`trimesh_to_pymeshlab`][tests.conversions.trimesh_to_pymeshlab] nor
    [`warp_to_pymeshlab`][tests.conversions.warp_to_pymeshlab] can build. Same per-filter freshness
    rule as those two: one MeshSet, one filter call.
    """
    meshset = ml.MeshSet()
    vertices = np.ascontiguousarray(points_np, dtype=np.float64)
    if normals_np is None:
        meshset.add_mesh(ml.Mesh(vertices))
    else:
        meshset.add_mesh(
            ml.Mesh(vertices, v_normals_matrix=np.ascontiguousarray(normals_np, dtype=np.float64))
        )
    return meshset


def trimesh_to_open3d(mesh: tm.Trimesh) -> o3d.geometry.TriangleMesh:
    """
    Wrap a ``tm.Trimesh`` in a legacy ``open3d.geometry.TriangleMesh``.

    Open3D wants float64 positions and **int32** faces (``Vector3iVector`` silently misreads a
    wider dtype). Every array goes through ``np.array`` rather than ``np.ascontiguousarray``, which
    is load-bearing: the pybind11 ``Vector3dVector`` cast raises ``ValueError: array is not
    writeable`` on a read-only input, and ``ascontiguousarray`` returns an already-contiguous array
    untouched -- so trimesh's cached ``vertex_normals`` would fail where ``vertices`` succeeded.

    Unlike the pymeshlab helpers this result is safe to reuse across calls *in tests*: the
    per-filter freshness rule that forces ``benchmarks/conftest.py`` to rebuild its MeshSet exists
    because a benchmark applies the same mutating call ten times in a row, which no test does. A
    test that calls a genuinely mutating method twice should still build twice.
    """
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.array(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.array(mesh.faces, dtype=np.int32)),
    )


def points_to_open3d(
    points_np: np.ndarray, normals_np: np.ndarray | None = None
) -> o3d.geometry.PointCloud:
    """
    Wrap a point cloud, and optionally its normals, in an ``open3d.geometry.PointCloud``.

    ``np.array`` rather than ``np.ascontiguousarray`` for the same writeability reason as
    [`trimesh_to_open3d`][tests.conversions.trimesh_to_open3d].
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(points_np, dtype=np.float64))
    if normals_np is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.array(normals_np, dtype=np.float64))
    return cloud


def open3d_to_trimesh(mesh_o3d: o3d.geometry.TriangleMesh) -> tm.Trimesh:
    """Read a legacy open3d mesh back out, unprocessed so the topology survives the round trip."""
    return tm.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices), faces=np.asarray(mesh_o3d.triangles), process=False
    )


def trimesh_to_open3d_t(mesh: tm.Trimesh) -> o3d.t.geometry.TriangleMesh:
    """
    Wrap a ``tm.Trimesh`` in a tensor-API ``open3d.t.geometry.TriangleMesh``.

    The result must be **bound to a name** by the caller before anything is chained off it:
    ``o3d.t.geometry.TriangleMesh.from_legacy(x).fill_holes()`` lets the temporary be collected
    mid-expression and the result reads freed memory -- garbage floats rather than an exception.
    Routing every tensor-mesh construction through this helper keeps that binding explicit. For
    ``RaycastingScene`` work, bind this result, then ``scene.add_triangles(mesh_t)``.
    """
    return o3d.t.geometry.TriangleMesh.from_legacy(trimesh_to_open3d(mesh))


def faces_igl(mesh: tm.Trimesh) -> np.ndarray:
    """
    Faces as the ``(n_faces, 3)`` int64 array the libigl bindings expect.

    libigl's Eigen templates are instantiated for 64-bit indices, so handing them trimesh's native
    dtype works by luck rather than contract; several functions crash on int32.

    See Also
    --------
    [`mesh_igl`][tests.conversions.mesh_igl]
        Both halves at once, for the majority of igl calls that take ``(V, F)``.
    """
    return mesh.faces.astype(np.int64)


def mesh_igl(mesh: tm.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """
    ``(vertices float64 (n, 3), faces int64 (n_faces, 3))``: the libigl calling convention.

    Most of the bound surface takes both arrays together, so the pair is the useful unit and
    [`faces_igl`][tests.conversions.faces_igl] is the F-only special case (``igl.adjacency_matrix``,
    ``igl.vertex_components``, ``igl.is_vertex_manifold``, which size their output by
    ``F.max() + 1`` rather than by ``len(V)``).

    !!! danger "Never pair these arrays with a *different* mesh's faces"
        libigl bounds-checks nothing: ``igl.cotmatrix(V, F)`` with one index past the end of ``V``
        is a SIGSEGV with no traceback, not an exception. Taking both halves from one mesh in one
        call is the point of this helper.
    """
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def trimesh_to_pyvista(mesh: tm.Trimesh) -> pv.PolyData:
    """
    Wrap a ``tm.Trimesh`` as a ``pyvista.PolyData``, preserving float64 positions exactly.

    Goes through ``PolyData.from_regular_faces`` rather than the padded ``[3, i, j, k]`` cell array
    the ``PolyData(points, faces)`` constructor wants: building that padding is an order of
    magnitude dearer than the classmethod, and it is the floor under every
    pyvista benchmark row.

    pyvista round-trips float64 exactly (measured error ``0.0`` on ``[1/3, pi, e]``), so where a
    comparison against ordito shows a ~1e-7 residual the float32 floor is *ordito's* ``wp.vec3``
    vertex buffer, not the reference's storage.
    """
    return pv.PolyData.from_regular_faces(
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int32),
    )


def points_to_pyvista(points_np: np.ndarray) -> pv.PolyData:
    """
    Wrap an ``(n, 3)`` point array as a ``pyvista.PolyData`` cloud.

    The bare constructor gives one vertex cell per point, which is what the cloud filters
    (``select_interior_points``, ``compute_implicit_distance``, ``interpolate``, ``delaunay_2d``)
    expect; passing a face array would make them read cell centres instead.
    """
    return pv.PolyData(np.ascontiguousarray(points_np, dtype=np.float64))


def polyline_to_pyvista(polyline_np: np.ndarray, *, closed: bool = False) -> pv.PolyData:
    """
    Wrap an ordered ``(n, 3)`` point array as a ``pyvista.PolyData`` holding **one** line cell.

    The single cell is the whole point: ``pv.lines_from_points`` builds one two-point cell per
    segment instead, and every polyline filter then restarts at each of them --
    ``compute_arc_length`` reports **0.0638** for a 200-point helix whose length is 12.7049, and
    ``decimate_polyline`` is a no-op at every reduction. ``find_closest_cell`` on this form is the
    point-to-*segment* distance (measured 2.49e-07 against
    [`polyline_point_distance`][ordito.polyline.polyline_point_distance]), where on the
    per-segment form it is the same answer at 199x the cell count.

    ``closed=True`` repeats the first index at the end, which is what ``triangulate_contours``
    needs to read the line as a polygon boundary; the point buffer itself is not duplicated.
    """
    points_np = np.ascontiguousarray(polyline_np, dtype=np.float64)
    indices_np = np.arange(points_np.shape[0])
    if closed:
        indices_np = np.append(indices_np, 0)
    return pv.PolyData(points_np, lines=np.hstack([[indices_np.size], indices_np]).astype(np.int64))


def pyvista_edges_to_indices(edges_pv: pv.PolyData, vertices_np: np.ndarray) -> np.ndarray:
    """
    Read a pyvista edge extraction's line cells back as ``(n, 2)`` indices, min-first per row.

    ``extract_feature_edges`` returns a **new** ``PolyData`` carrying only the points its lines
    touch, renumbered -- so its ``lines`` array indexes that new set and not the mesh it came from.
    Measured: on a unit box the extraction happens to keep all 8 points and still orders them
    differently, so a comparison that skips this step fails on a mesh where the *counts* match,
    which reads as a real disagreement. ``extract_all_edges`` does keep the numbering, but it goes
    through the same path here so no caller has to remember which filter is which.

    The mapping is by position and exact. VTK 9.7's multithreaded ``vtkExtractEdges`` stores its
    points as ``float32`` whatever the input precision, and has no output-precision setting, so
    the mesh vertices are rounded to the extraction's own dtype before the match: the
    nearest-neighbour distance is then ``0.0`` and the assert below stays a bijection check
    rather than becoming a tolerance.
    """
    lines_np = np.asarray(edges_pv.lines).reshape(-1, 3)[:, 1:]
    points_np = np.asarray(edges_pv.points)
    reference_np = np.ascontiguousarray(vertices_np, dtype=points_np.dtype)
    distances_np, indices_np = cKDTree(reference_np).query(points_np)
    indices_np = np.asarray(indices_np)
    assert float(np.max(distances_np, initial=0.0)) == 0.0
    # One-to-one, or the rounding merged two vertices and the match picked one of them.
    assert np.unique(indices_np).size == indices_np.size
    return np.sort(indices_np[lines_np], axis=1)


def numpy_to_meshlib(vertices_np: np.ndarray, faces_np: np.ndarray) -> mm.Mesh:
    """
    Build a ``meshlib.mrmeshpy.Mesh`` from a NumPy ``(vertices, faces)`` pair.

    **Vertices first**, like every other converter in this module -- which is the point of routing
    through it, because ``mn.meshFromFacesVerts`` itself takes *faces* first. That swap is the one
    thing to get wrong by hand: the two arrays differ in shape only when the counts differ, so a
    reversed call on a mesh with as many faces as vertices builds silent nonsense rather than
    raising. Both dtypes are permissive on this wheel -- int32 and int64 faces, float32 and float64
    positions all accepted -- so the arrays go through as the caller holds them, and ``faces_np``
    may be ``(n_faces, 3)`` or already flat.

    The result is **not** reusable across calls: almost every MeshLib free function mutates its
    ``Mesh`` in place and returns a status, a count or a bitset of new elements, so build a fresh
    one per comparison -- the ``trimesh_to_pymeshlab`` rule rather than the ``trimesh_to_open3d``
    one. A mutating call also invalidates the lazily-built AABB tree cached on the mesh.

    !!! warning "The vertex buffer is sized by the faces, not by ``len(V)``"
        A **trailing** unreferenced vertex is dropped outright (163 positions in, 162 back), while
        an **interior** one is kept in the buffer and excluded from ``numValidVerts`` (163 in, 163
        back, ``numValidVerts`` 162). So never assume ``getNumpyVerts(...).shape[0] == len(V)``, and
        never hand MeshLib a compacted ``V`` with the original ``F``.

    See Also
    --------
    [`meshlib_to_trimesh`][tests.conversions.meshlib_to_trimesh]
        The inverse, which packs before reading the topology back.
    [`trimesh_to_meshlib`][tests.conversions.trimesh_to_meshlib]
        The same conversion from a ``tests/conftest.py`` fixture.
    [`warp_to_meshlib`][tests.conversions.warp_to_meshlib]
        The same conversion from a ordito output.
    """
    # MeshLib's stubs type these as ``Buffer``, which numpy's stubs implement only from Python 3.12.
    return mn.meshFromFacesVerts(
        cast("Buffer", np.ascontiguousarray(np.asarray(faces_np).reshape(-1, 3))),
        cast("Buffer", np.ascontiguousarray(vertices_np)),
    )


def trimesh_to_meshlib(mesh: tm.Trimesh) -> mm.Mesh:
    """
    Wrap a ``tm.Trimesh`` in a ``meshlib.mrmeshpy.Mesh``.

    Thin front end on [`numpy_to_meshlib`][tests.conversions.numpy_to_meshlib]; its argument-order,
    freshness and vertex-buffer-sizing notes all apply here.
    """
    return numpy_to_meshlib(mesh.vertices, mesh.faces)


def warp_to_meshlib(vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]) -> mm.Mesh:
    """
    Read ordito's ``(vertices, flat faces)`` pair into a ``meshlib.mrmeshpy.Mesh``.

    Use this when the mesh under test is a ordito *output* rather than one of the
    ``tests/conftest.py`` fixtures (which already carry a ``tm.Trimesh`` for
    [`trimesh_to_meshlib`][tests.conversions.trimesh_to_meshlib]). Same freshness rule as
    [`numpy_to_meshlib`][tests.conversions.numpy_to_meshlib], which does the work.

    !!! note "Deliberately unexercised, and not dead"
        No test calls this today, and that is a decision rather than an oversight -- the
        inventory of helpers points readers at it, so deleting it invites the next person to
        hand-roll the wrapper and rediscover the hazards above. It also has a caller that is
        invisible from here: ``tests/parity.py`` keys its reference detection off these
        converter *names*, so removing one would silently narrow the parity scan.
    """
    return numpy_to_meshlib(vertices_wp.numpy(), faces_wp.numpy())


def points_to_meshlib(points_np: np.ndarray, normals_np: np.ndarray | None = None) -> mm.PointCloud:
    """
    Wrap a bare point cloud, and optionally its normals, in a ``meshlib.mrmeshpy.PointCloud``.

    Several oracles read the normals and quietly do something else without them --
    ``makeOrientedNormals``, ``triangulatePointCloud`` and ``findOutliers`` all consult
    ``cloud.normals`` -- so pass them whenever the ordito side had them. ``findOutliers`` does
    worse than quietly: at its default ``mask`` of ``All`` it **segfaults** on a cloud with no
    normals, because that set includes the ``AwayNormal`` criterion.
    """
    # ``Buffer``: see numpy_to_meshlib.
    points_ml = cast("Buffer", np.ascontiguousarray(points_np))
    if normals_np is None:
        return mn.pointCloudFromPoints(points_ml)
    return mn.pointCloudFromPoints(points_ml, cast("Buffer", np.ascontiguousarray(normals_np)))


def numpy_to_meshlib_bitset(flags_np: np.ndarray) -> mm.BitSet:
    """
    Load a flat ``bool`` array into a ``mm.BitSet`` of exactly ``flags_np.size`` bits.

    The bulk inverse of [`meshlib_bitset_to_numpy`][tests.conversions.meshlib_bitset_to_numpy], and
    the reason neither direction needs a Python loop. ``BitSet.fromBlocks`` is bound and takes the
    raw ``uint64`` blocks, so ``np.packbits`` fills the whole set in one call -- measured 258x
    faster than a per-cell ``set()`` loop by more than two orders of magnitude, which is the
    difference between a benchmarkable reference and one whose row would time the load.

    Two mechanical points. ``bitorder="little"`` is not optional: the block's bit *i* is index
    ``i``, which is NumPy's non-default order. And ``fromBlocks`` rejects a NumPy ``uint64`` array
    with ``TypeError`` -- the bound argument is ``std_vector_unsigned_long``, which accepts a Python
    list -- then rounds the size up to whole 64-bit blocks, so the ``resize`` trims the padding back
    to the caller's domain.

    ``flags_np`` must already be flattened in the target element's own order; for a
    ``VoxelBitSet`` addressed by a ``VolumeIndexer`` that is ``x`` fastest, i.e.
    ``occupancy_np.ravel(order="F")`` for a dense ``(nx, ny, nz)`` array. Wrap the result in the
    typed set the call wants -- ``mm.VoxelBitSet(bitset_ml)``, ``mm.FaceBitSet(bitset_ml)`` -- whose
    converting constructor copies the bits and the size.

    See Also
    --------
    [`meshlib_bitset_to_numpy`][tests.conversions.meshlib_bitset_to_numpy]
    """
    flat_np = np.ascontiguousarray(flags_np, dtype=bool).ravel()
    packed_np = np.packbits(flat_np, bitorder="little")
    packed_np = np.pad(packed_np, (0, (-packed_np.size) % 8)).view(np.uint64)
    bitset_ml = mm.BitSet.fromBlocks(mm.std_vector_unsigned_long(packed_np.tolist()))
    bitset_ml.resize(flat_np.size)
    return bitset_ml


def meshlib_to_trimesh(mesh_ml: mm.Mesh, *, pack: bool = True) -> tm.Trimesh:
    """
    Read a MeshLib mesh back into a ``tm.Trimesh``, packing the deleted elements away first.

    ``pack()`` is **mandatory** after anything that deletes, and skipping it is silent. Measured
    after ``decimateMesh(maxDeletedFaces=200)`` on a 320-face mesh: ``numValidFaces`` reads 120,
    ``topology.faceSize()`` reads 320, and ``getNumpyFaces`` returns **319** rows --
    ``last_valid_face_id + 1`` -- of which **199 are ``[0, 0, 0]``**, degenerate triangles on vertex
    0. ``getNumpyVerts`` still returns all 162 positions while ``numValidVerts`` is 62. No
    exception and no warning. After ``pack()`` the same reads give 120 faces and 62 vertices.

    Pass ``pack=False`` only to observe that state deliberately; note it *mutates* ``mesh_ml``
    either way, since packing renumbers in place.

    ``process=False`` for the same reason
    [`warp_to_trimesh`][tests.conversions.warp_to_trimesh] uses it: trimesh's default processing
    welds coincident vertices, which would undo the identification the comparison is testing.
    """
    if pack:
        mesh_ml.pack()
    return tm.Trimesh(
        vertices=mn.getNumpyVerts(mesh_ml).astype(np.float64),
        faces=mn.getNumpyFaces(mesh_ml.topology).astype(np.int64),
        process=False,
    )


def meshlib_scalars_to_numpy(scalars_ml: _MeshlibScalars) -> np.ndarray:
    """
    Read a MeshLib scalar container (``VertScalars`` / ``FaceScalars`` / ...) as a float64 array.

    Neither of the two obvious routes works. ``mn.toNumpyArray`` binds only ``VertCoords`` /
    ``FaceNormals`` / ``std_vector_Vector3_float`` and raises a clear ``TypeError`` for anything
    else, which is safe; but ``np.asarray(vert_scalars)`` returns a **0-d ``object`` array** rather
    than raising, so a comparison written that way fails several lines later in whatever NumPy call
    comes next, with nothing pointing at the cause.

    Iteration is the working route. Bitsets have their own reader,
    [`meshlib_bitset_to_numpy`][tests.conversions.meshlib_bitset_to_numpy], because their size is
    *not* always the element domain.
    """
    return np.fromiter(iter(scalars_ml), np.float64, scalars_ml.size())


def meshlib_indices_to_numpy(indices_ml: _MeshlibIndices) -> np.ndarray:
    """
    Read a MeshLib index container (``Buffer_VertId`` / ``VertMap`` / ...) as an int64 array.

    The sibling of [`meshlib_scalars_to_numpy`][tests.conversions.meshlib_scalars_to_numpy] for the
    containers whose element is an *id* rather than a number, and it needs its own body because the
    scalar reader raises on them: a ``VertId`` implements ``__index__`` but not ``__float__``, so
    ``np.fromiter(..., np.float64)`` fails with ``float() argument must be a string or a real
    number``. ``np.asarray`` is the usual silent trap -- a 0-d ``object`` array -- and
    ``mn.toNumpyArray`` rejects the type outright.

    An invalid id comes back as ``-1``, which is what ``VertId()`` holds by default, so the result
    is signed rather than unsigned on purpose.
    """
    return np.fromiter((int(index_ml) for index_ml in indices_ml), np.int64, indices_ml.size())


def meshlib_bitset_to_numpy(bitset_ml: mm.BitSet, size: int) -> np.ndarray:
    """
    Read a MeshLib bitset as a ``bool`` array of exactly ``size`` entries.

    ``mn.getNumpyBitSet`` returns the bitset at *its own* length, and whether that is the element
    domain depends on which function produced it. A bitset MeshLib sized against the mesh comes
    back domain-sized -- ``getBoundaryVerts`` on a 162-vertex mesh gives 162 entries even when one
    bit is set -- but a bitset built by *insertion* is only as long as its highest set bit needs:
    ``findSelfCollidingTrianglesBS`` on a 640-face pair of overlapping spheres returns **608**
    entries (last colliding face 607) and an *empty* array on a clean mesh, where the comparison
    wants 640 and 640.

    Neither shape raises, and both break a ``np.array_equal`` against a ordito mask by shape
    rather than by value, so every bitset in this suite is read through here with the domain size
    stated by the caller. A bitset *longer* than ``size`` is a genuine domain mismatch (the wrong
    element type, or a converter that dropped elements) and raises.
    """
    flags_np = mn.getNumpyBitSet(bitset_ml)
    if flags_np.size > size:
        raise ValueError(f"bitset holds {flags_np.shape[0]} bits, more than the {size} elements")
    return np.pad(flags_np, (0, size - flags_np.size))


def numpy_to_pymeshfix(vertices_np: np.ndarray, faces_np: np.ndarray) -> _meshfix.PyTMesh:
    """
    Build a ``pymeshfix._meshfix.PyTMesh`` from a NumPy ``(vertices, faces)`` pair.

    **Vertices first**, like every other builder here, and quiet by default -- ``set_quiet(True)``
    goes in before the load because the kernel narrates to stderr and one of its messages is
    reported *inverted* (see below).

    !!! warning "``load_array`` is not a load; it is already a repair, and it renumbers"
        It runs the kernel's connectivity fix and Euler update before returning, so the mesh that
        comes back is not the mesh that went in. Measured on a 42-vertex / 80-face icosphere: a
        trailing *or* interior unreferenced vertex is dropped (43 -> 42, surviving positions and
        their relative order intact); an exactly duplicated face is **kept** and the non-manifold
        edges it creates are cut instead (80 f -> 81 f, 42 v -> **45 v**), while a *reversed*
        duplicate is refused and the vertices are still cut; two coincident referenced vertices are
        **not** merged; one backwards face is rewound, and *every* face backwards is left alone,
        because consistent is not the same as outward (volume -3.6587 in and out).

        On the scan meshes: ``bunny_decimated`` 8 171 v / 16 301 f loads as **8 372 v / 16 220 f**
        and ``bunny`` 35 947 v / 69 451 f as **34 834 v / 69 451 f** (its 1 113 unreferenced
        vertices). On a non-orientable closed surface it cuts the orientation-reversing seam --
        ``boy`` 1 483 v -> 1 559 v at an unchanged 2 964 faces -- which leaves two coincident sheets
        where the surface had one.

        So **every comparison against this reference must be index-free**: positions, canonically
        sorted face rows, sets and counts. Where a face index is unavoidable, assert the load
        changed nothing first (``n_points == len(v) and n_faces == len(f)``) and build the map from
        the *returned* buffer -- which is what
        [`pymeshfix_intersecting_faces`][tests.conversions.pymeshfix_intersecting_faces] does.

    One ``PyTMesh`` serves **one load and one mutating call**: a second ``load_array`` raises
    ``RuntimeError``, and every algorithm mutates in place and returns a status, a count or an
    array rather than the mesh. So this returns a fresh object every call and there is no cached
    form -- the ``numpy_to_meshlib`` rule, not the ``trimesh_to_open3d`` one.

    See Also
    --------
    [`pymeshfix_to_numpy`][tests.conversions.pymeshfix_to_numpy]
        The inverse, whose face buffer is a reordering even when nothing was repaired.
    [`trimesh_to_pymeshfix`][tests.conversions.trimesh_to_pymeshfix]
    [`warp_to_pymeshfix`][tests.conversions.warp_to_pymeshfix]
    """
    tin_pmf = _meshfix.PyTMesh()
    tin_pmf.set_quiet(True)
    tin_pmf.load_array(
        np.ascontiguousarray(vertices_np, dtype=np.float64),
        np.ascontiguousarray(np.asarray(faces_np).reshape(-1, 3), dtype=np.int32),
    )
    return tin_pmf


def trimesh_to_pymeshfix(mesh: tm.Trimesh) -> _meshfix.PyTMesh:
    """
    Wrap a ``tm.Trimesh`` in a fresh ``pymeshfix._meshfix.PyTMesh``.

    Thin front end on [`numpy_to_pymeshfix`][tests.conversions.numpy_to_pymeshfix]; its renumbering
    and one-call-per-object notes both apply.

    !!! warning "A ``trimesh.slice_plane`` output is a poor input"
        The same hazard the MeshLib converters carry, and worse here. A hemisphere sliced from
        ``icosphere(2)`` without ``merge_vertices()`` loads as 121 v -> **137 v** and reports **17**
        boundary loops where the surface has one; after ``merge_vertices()`` it loads unchanged at
        97 v and reports **1**. Use the ``tests/conftest.py`` fixtures, which already merge, and
        assert ``n_boundaries`` before comparing a per-hole answer.
    """
    return numpy_to_pymeshfix(mesh.vertices, mesh.faces)


def warp_to_pymeshfix(
    vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]
) -> _meshfix.PyTMesh:
    """
    Read ordito's ``(vertices, flat faces)`` pair into a fresh ``pymeshfix._meshfix.PyTMesh``.

    Use this when the mesh under test is a ordito *output* rather than one of the
    ``tests/conftest.py`` fixtures (which carry a ``tm.Trimesh`` for
    [`trimesh_to_pymeshfix`][tests.conversions.trimesh_to_pymeshfix]). Same renumbering and
    one-call-per-object rules as [`numpy_to_pymeshfix`][tests.conversions.numpy_to_pymeshfix].

    !!! note "Deliberately unexercised, and not dead"
        No test calls this today, and that is a decision rather than an oversight -- the
        inventory of helpers points readers at it, so deleting it invites the next person to
        hand-roll the wrapper and rediscover the hazards above. It also has a caller that is
        invisible from here: ``tests/parity.py`` keys its reference detection off these
        converter *names*, so removing one would silently narrow the parity scan.
    """
    return numpy_to_pymeshfix(vertices_wp.numpy(), faces_wp.numpy())


def pymeshfix_to_numpy(tin_pmf: _meshfix.PyTMesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a ``PyTMesh`` back as ``(vertices, faces)``, and say why the buffers may not line up.

    ``return_arrays()`` unchanged -- this exists so the warning has one home rather than being
    restated at every call site.

    !!! warning "The face buffer is a reordering, even when nothing was repaired"
        ``icosphere(2)`` round trips with **byte-identical float64 vertices** (max abs difference
        ``0.0``) and an identical *triangle set* under ``np.sort(rows, axis=1)`` plus a lexsort, but
        the rows come back in a different order and each row starts at a different corner. Never
        compare face buffers positionally; and remember the vertex buffer itself may be a different
        length than the input's -- see
        [`numpy_to_pymeshfix`][tests.conversions.numpy_to_pymeshfix].
    """
    return tin_pmf.return_arrays()


def pymeshfix_intersecting_faces(
    tin_pmf: _meshfix.PyTMesh, *, tris_per_cell: int, justproper: bool
) -> np.ndarray:
    """
    Face indices from ``select_intersecting_triangles``, with the uninitialised tail dropped.

    The call allocates an ``(n, 3)`` ``int32`` array and writes its ``n`` face indices into the
    **flat** prefix, leaving ``2n`` entries of heap garbage. Measured on two ``icosphere(2)``s
    translated 1.2 apart (640 faces): shape ``(72, 3)``, a flat prefix of 72 ascending indices all
    below 640, and ``arr.max()`` reading **30 751** -- a value that varies between processes. So
    ``out.ravel()[: out.shape[0]]`` is the only defined read, and a naive
    ``np.array_equal(out1, out2)`` over two calls reports nondeterminism that is not there: the
    prefix is identical across repeat calls and across a fresh object, only the tail is not.

    Indices address the **returned** face buffer, which is a reordering of the input's
    ([`pymeshfix_to_numpy`][tests.conversions.pymeshfix_to_numpy]), so a comparison against a
    ordito per-face mask has to remap through the canonical sorted rows. The result is sorted, so
    it compares directly against ``np.flatnonzero`` of a mask once remapped.
    """
    out_pmf = tin_pmf.select_intersecting_triangles(
        tris_per_cell=tris_per_cell, justproper=justproper
    )
    return np.sort(out_pmf.ravel()[: out_pmf.shape[0]])


def pymeshfix_face_remap(tin_pmf: _meshfix.PyTMesh, faces_np: np.ndarray) -> np.ndarray:
    """
    Map each *returned* face of a ``PyTMesh`` back to its index in ``faces_np``.

    The one sanctioned way to compare a pymeshfix per-face answer against a ordito mask, and it
    raises rather than guessing when that is not possible. Two things stand between the two index
    spaces: ``load_array`` may add or drop faces and vertices before anything else runs, and even
    when it does not, the buffer that comes back is a *reordering* whose rows also start at
    different corners ([`pymeshfix_to_numpy`][tests.conversions.pymeshfix_to_numpy]).

    So this checks the face count first, then keys both buffers by their canonically sorted rows --
    which also catches a load that renumbered the *vertices* at an unchanged face count, since the
    keys then match nothing (``boy`` loads as 1 559 vertices from 1 483 with all 2 964 faces
    intact). A caller does ``remap[pymeshfix_intersecting_faces(tin_pmf, ...)]`` and compares that
    against ``np.flatnonzero(mask_wp.numpy())``.

    Raises
    ------
    ValueError
        If the load changed the face count, if the input faces are not distinct as unordered rows,
        or if a returned face is not one of the input's.

    See Also
    --------
    [`pymeshfix_intersecting_faces`][tests.conversions.pymeshfix_intersecting_faces]
    """
    faces_in = np.ascontiguousarray(np.asarray(faces_np).reshape(-1, 3))
    _, faces_out = tin_pmf.return_arrays()
    if len(faces_out) != len(faces_in):
        raise ValueError(
            f"load_array changed the mesh ({len(faces_in)} faces in, {len(faces_out)} back); "
            "no face-index remap exists -- compare counts, sets or positions instead"
        )
    index_of = {tuple(sorted(row)): i for i, row in enumerate(faces_in.tolist())}
    if len(index_of) != len(faces_in):
        raise ValueError("input faces are not distinct as unordered rows; no remap exists")
    try:
        return np.array(
            [index_of[tuple(sorted(row))] for row in faces_out.tolist()], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(f"returned face {error.args[0]} is not an input face") from error


def points_to_torch(points_np: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Upload an ``(n, 3)`` point cloud as the ``(1, n, 3)`` float32 tensor ``pytorch3d.ops`` wants.

    **Every** ``pytorch3d.ops`` entry point is batched with a leading minibatch axis, and the
    padding is expressed as a separate ``lengths`` argument rather than inferred -- so a single
    cloud goes in as ``x[None]`` and its answer comes back out as ``result[0]``. That wrap is the
    shape of trap that is easy to miss: it is one line, it is needed at every
    call site, and getting it wrong does not raise. ``knn_points(p, q)`` handed a bare ``(P, 3)``
    reads it as ``(N=P, P1=3, D)`` and cheerfully compares three points, which is why the
    pytorch3d comparisons assert the reference's output *shape* before its values.

    float32 because that is what ordito's ``wp.vec3`` holds; pytorch3d preserves float64 where it
    is handed it, so matching the storage keeps the residual attributable to one side.

    Parameters
    ----------
    points_np
        ``(n, 3)`` positions or directions, any float dtype.
    device
        Torch device string; warp's own ``"cpu"`` / ``"cuda:0"`` spellings are accepted verbatim.

    See Also
    --------
    [`points_to_pytorch3d`][tests.conversions.points_to_pytorch3d]
        The ``Pointclouds`` container form, for the ``loss`` entry points.
    """
    return torch.as_tensor(
        np.ascontiguousarray(points_np, dtype=np.float32), device=device
    ).unsqueeze(0)


def numpy_to_pytorch3d(
    vertices_np: np.ndarray, faces_np: np.ndarray, device: str = "cpu"
) -> p3d_structures.Meshes:
    """
    Wrap a NumPy mesh in a ``pytorch3d.structures.Meshes``, **float32 positions and int64 faces**.

    A ``Meshes`` is an immutable caching container, which makes it the opposite of a
    ``pymeshlab.MeshSet``: it derives ``verts_packed`` / ``edges_packed`` /
    ``faces_packed_to_edges_packed`` / ``verts_normals_packed`` on first request and memoizes them,
    and every ``pytorch3d.ops`` and ``pytorch3d.loss`` entry point is pure. So **one object serves
    many comparisons** and there is no freshness rule to observe. The corollary is that the answer
    lives on the accessors and not in the constructor arguments -- ``verts_packed()`` is what a
    comparison reads, never the tensors handed in.

    Positions land as float32 because that is what ``wp.vec3`` holds. pytorch3d does *not* cast for
    you: a float64 ``Meshes`` keeps float64 through ``verts_packed()``, so an unconverted reference
    would compare a float64 answer against ordito's float32 one and read as ordito being wrong by
    ~1e-7. Faces land as int64, which is what ``Meshes`` stores regardless (an int32 face tensor is
    silently widened), so the cast is documentation rather than a requirement.

    Parameters
    ----------
    vertices_np
        ``(n, 3)`` positions, any float dtype.
    faces_np
        Vertex indices, ``(n_faces, 3)`` or already flat -- reshaped either way.
    device
        Torch device string; warp's own ``"cpu"`` / ``"cuda:0"`` spellings are accepted verbatim.

    See Also
    --------
    [`pytorch3d_to_numpy`][tests.conversions.pytorch3d_to_numpy]
        The inverse, for reading a pytorch3d result back out.
    """
    return p3d_structures.Meshes(
        verts=[torch.as_tensor(np.ascontiguousarray(vertices_np, dtype=np.float32), device=device)],
        faces=[
            torch.as_tensor(
                np.ascontiguousarray(np.asarray(faces_np).reshape(-1, 3), dtype=np.int64),
                device=device,
            )
        ],
    )


def trimesh_to_pytorch3d(mesh: tm.Trimesh, device: str = "cpu") -> p3d_structures.Meshes:
    """Wrap a ``tm.Trimesh`` in a ``pytorch3d.structures.Meshes`` on ``device``."""
    return numpy_to_pytorch3d(mesh.vertices, mesh.faces, device)


def warp_to_pytorch3d(
    vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32], device: str | None = None
) -> p3d_structures.Meshes:
    """
    Wrap a ordito ``(vertices, flat faces)`` pair in a ``Meshes``, on the buffers' own device.

    ``device`` overrides that, which is what the CPU-reference comparisons want: a ordito answer
    computed on ``cuda:0`` is still compared against a pytorch3d one built on the host, since
    pytorch3d has separate CPU and CUDA kernels and only the CPU pass is a stable oracle. Pass the
    warp device explicitly to exercise the CUDA kernels instead.
    """
    return numpy_to_pytorch3d(
        vertices_wp.numpy(), faces_wp.numpy(), str(vertices_wp.device) if device is None else device
    )


def points_to_pytorch3d(
    points_np: np.ndarray, normals_np: np.ndarray | None = None, device: str = "cpu"
) -> p3d_structures.Pointclouds:
    """
    Wrap a point cloud, and optionally its normals, in a ``pytorch3d.structures.Pointclouds``.

    The container form of [`points_to_torch`][tests.conversions.points_to_torch]: the ``loss``
    entry points (``chamfer_distance``, ``point_mesh_face_distance``) take a ``Pointclouds`` where
    the ``ops`` ones take bare batched tensors. Same caching-container rules as
    [`numpy_to_pytorch3d`][tests.conversions.numpy_to_pytorch3d] -- immutable, safe to share.
    """
    return p3d_structures.Pointclouds(
        points=[torch.as_tensor(np.ascontiguousarray(points_np, dtype=np.float32), device=device)],
        normals=(
            None
            if normals_np is None
            else [
                torch.as_tensor(np.ascontiguousarray(normals_np, dtype=np.float32), device=device)
            ]
        ),
    )


def pytorch3d_to_numpy(meshes_p3d: p3d_structures.Meshes) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a single-mesh ``Meshes`` back out as ``(vertices_f64, faces_i64)``.

    Goes through the ``*_packed()`` accessors rather than ``verts_list()``, because that is where a
    pytorch3d *result* lives -- ``SubdivideMeshes`` and ``ops.cubify`` both return a ``Meshes``
    whose constructor arguments the caller never saw. ``detach()`` because several entry points
    return tensors carrying a grad graph, which ``numpy()`` refuses.
    """
    verts_t = meshes_p3d.verts_packed()
    faces_t = meshes_p3d.faces_packed()
    assert verts_t is not None
    assert faces_t is not None
    return (
        verts_t.detach().cpu().numpy().astype(np.float64),
        faces_t.detach().cpu().numpy().astype(np.int64),
    )


def warp_to_trimesh(vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]) -> tm.Trimesh:
    """
    Read a ordito ``(vertices, flat faces)`` pair back into a ``tm.Trimesh``.

    ``process=False`` is not optional: trimesh's default processing merges coincident vertices, and
    the meshes that most need this conversion are the ones whose *identification* is the thing under
    test -- a Moebius band's seam is a single shared index by construction, and a fixture that gets
    re-welded on the way in is no longer the fixture that was built.
    """
    return tm.Trimesh(
        vertices=vertices_wp.numpy().astype(np.float64),
        faces=faces_wp.numpy().reshape(-1, 3).astype(np.int64),
        process=False,
    )


def scipy_to_bsr(
    matrix_sp: sp.coo_matrix | sp.csr_matrix | sp.csc_matrix, device: wp.DeviceLike
) -> odt.BsrMatrix[wp.float64]:
    """
    Upload a scipy sparse matrix as a ``float64`` BSR of 1x1 blocks on ``device``.

    The inverse of [`bsr_to_csr`][tests.conversions.bsr_to_csr]. The matrix goes through its COO
    form as triplets, so a ``coo_matrix`` built with repeated ``(row, col)`` pairs reaches
    ``warp.sparse.bsr_from_triplets`` with its duplicates intact -- they are summed there, and the
    result's ``nnz`` is the stale triplet count a duplicate build leaves. Explicit zeros are kept
    (``prune_numerical_zeros=False``), so the uploaded pattern is the scipy matrix's own.
    """
    coo = matrix_sp.tocoo()
    matrix_wp = wps.bsr_from_triplets(
        int(coo.shape[0]),
        int(coo.shape[1]),
        wp.array(coo.row.astype(np.int32), dtype=wp.int32, device=device),
        wp.array(coo.col.astype(np.int32), dtype=wp.int32, device=device),
        wp.array(np.ascontiguousarray(coo.data, dtype=np.float64), dtype=wp.float64, device=device),
        prune_numerical_zeros=False,
    )
    assert odt.has_blocks(matrix_wp, wp.float64)
    return matrix_wp


def bsr_to_dense(matrix: odt.SparseMatrix, n_vertices: int) -> np.ndarray:
    """
    Densify a ``BsrMatrix``, reading only the entries its offsets actually address.

    ``BsrMatrix.values`` is allocated at the *triplet* count and its ``nnz`` is an upper bound until
    synchronized, so the tail of that buffer is uninitialized scratch. Comparing two matrices'
    ``values`` arrays directly reads that scratch and is flaky by construction; the row offsets are
    the only safe way in.
    """
    offsets = matrix.offsets.numpy()
    columns = matrix.columns.numpy()
    values = matrix.values.numpy()
    dense = np.zeros((n_vertices, n_vertices))
    for row in range(n_vertices):
        for slot in range(offsets[row], offsets[row + 1]):
            dense[row, columns[slot]] = values[slot]
    return dense


def bsr_to_csr(matrix: odt.SparseMatrix) -> sp.csr_matrix:
    """
    Convert a ``BsrMatrix`` to a scipy CSR, at its own declared shape.

    The scipy-side counterpart of [`bsr_to_dense`][tests.conversions.bsr_to_dense], for the operator
    comparisons that want a sparse matrix rather than a dense block -- the Crouzeix-Raviart pair is
    ``(n_edges, n_edges)`` and the LSCM Hessian ``(2n, 2n)``, both too large to densify comfortably
    on the larger fixtures. Reads through the row offsets for the same reason ``bsr_to_dense`` does:
    ``values`` is allocated at the triplet count and its tail is uninitialized scratch.
    """
    nrow = int(matrix.nrow)
    ncol = int(matrix.ncol)
    return sp.csr_matrix(
        (matrix.values.numpy(), matrix.columns.numpy(), matrix.offsets.numpy()), shape=(nrow, ncol)
    )


def meshlib_corner_normals_to_numpy(
    corner_normals_ml: mm.Vector_std_array_Vector3f_3_FaceId, n_faces: int
) -> np.ndarray:
    """
    Read ``computePerCornerNormals``' result into an ``(n_faces, 3, 3)`` array.

    Its return type is ``Vector_std_array_Vector3f_3_FaceId``, which has no ``len()`` and is not
    accepted by ``mrmeshnumpy``'s readers -- it is indexed by ``FaceId`` and yields a fixed
    three-element array of ``Vector3f`` per face. So the readback is a double loop and there is no
    bulk path; keep the meshes small where this is the oracle.
    """
    return np.array(
        [
            [
                (row[k].x, row[k].y, row[k].z)
                for k, row in ((k, corner_normals_ml[mm.FaceId(face)]) for k in range(3))
            ]
            for face in range(n_faces)
        ],
        dtype=np.float64,
    )


def numpy_to_meshlib_undirected_edges(
    topology_ml: mm.MeshTopology, edges_np: np.ndarray
) -> mm.UndirectedEdgeBitSet:
    """
    Vertex-index pairs as MeshLib's ``UndirectedEdgeBitSet``, which its crease arguments take.

    ``findEdge`` resolves a directed ``EdgeId``, and ``.undirected()`` drops the direction, so a row
    given in either order sets the same bit. The set has to be ``resize``d to the topology's edge
    count first: an unsized one silently accepts nothing.
    """
    bits_ml = mm.UndirectedEdgeBitSet()
    bits_ml.resize(topology_ml.undirectedEdgeSize())
    for start, end in np.asarray(edges_np).tolist():
        bits_ml.set(
            topology_ml.findEdge(mm.VertId(int(start)), mm.VertId(int(end))).undirected(), True
        )
    return bits_ml
