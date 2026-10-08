"""Regression tests for ``ordito.combine``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
import pytorch3d.structures as p3d_structures
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm
from scipy.spatial import cKDTree

import ordito as od
from tests.conftest import CLOSED_MESHES
from tests.conversions import (
    meshlib_bitset_to_numpy,
    meshlib_to_trimesh,
    numpy_to_warp,
    open3d_to_trimesh,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pytorch3d,
    warp_empty,
)

MeshPair = tuple[tm.Trimesh, wp.Mesh]


@pytest.fixture
def three_parts(
    sphere_irregular: MeshPair, saddle_graded: MeshPair, torus_irregular: MeshPair
) -> list[MeshPair]:
    """Return the three disjoint fixtures the multi-part tests combine."""
    return [sphere_irregular, saddle_graded, torus_irregular]


def _concatenate_parts(parts: Sequence[MeshPair]) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    return od.combine.concatenate([(mesh_wp.points, mesh_wp.indices) for _, mesh_wp in parts])


@pytest.mark.parity("concatenate", "trimesh")
@pytest.mark.parametrize("n_parts", [1, 3], ids=["one_mesh", "three_meshes"])
def test_concatenate_meshes(three_parts: list[MeshPair], n_parts: int) -> None:
    """
    Class A: meshes packed against ``trimesh.util.concatenate``, positions and indices.

    The index *offsetting* is the whole operation, so comparing elementwise rather than as a
    set is the point: a wrong offset would still give a valid-looking mesh with the right
    counts. The one-mesh arm checks that a one-element list comes back with the input's own
    values, not merely an equivalent mesh.
    """
    parts = three_parts[:n_parts]
    concat_tm = tm.util.concatenate([mesh_tm for mesh_tm, _ in parts])
    assert isinstance(concat_tm, tm.Trimesh)
    concat_vertices_wp, concat_faces_wp = _concatenate_parts(parts)
    assert np.allclose(concat_vertices_wp.numpy(), concat_tm.vertices)
    assert np.array_equal(concat_faces_wp.numpy(), concat_tm.faces.reshape(-1))


@pytest.mark.parity("concatenate", "meshlib")
def test_concatenate_matches_meshlib(request: pytest.FixtureRequest) -> None:
    """
    Class A on both buffers: ``mergeMeshes`` offsets the same indices into the same order.

    Its argument is a ``std_vector_std_shared_ptr_Mesh`` -- a vector of *shared pointers*, which
    pybind11 fills from plain ``Mesh`` objects, so the meshes have to be built first and appended
    rather than constructed inline. It returns a new mesh and mutates none of its inputs, which is
    unusual enough in this library to be worth stating.

    The comparison is element-wise rather than set-wise for the reason the trimesh pairing above
    gives: the index *offsetting* is the whole operation, and a wrong offset still yields a
    valid-looking mesh with the right counts. Both sides land on 1 656 vertices and 3 174 faces from
    ``sphere_irregular`` and ``saddle_graded`` here, with identical positions and identical faces.

    The fixtures are disjoint on purpose -- ``mergeMeshes`` does not weld, and neither does
    ``concatenate``; overlapping inputs would compare two different de-duplication policies rather
    than two concatenations.
    """
    mesh_a_tm, mesh_a_wp = request.getfixturevalue("sphere_irregular")
    mesh_b_tm, mesh_b_wp = request.getfixturevalue("saddle_graded")

    concat_vertices_wp, concat_faces_wp = od.combine.concatenate(
        [(mesh_a_wp.points, mesh_a_wp.indices), (mesh_b_wp.points, mesh_b_wp.indices)]
    )

    meshes_ml = mm.std_vector_std_shared_ptr_Mesh()
    for mesh_tm in (mesh_a_tm, mesh_b_tm):
        meshes_ml.append(trimesh_to_meshlib(mesh_tm))
    merged_tm = meshlib_to_trimesh(mm.mergeMeshes(meshes_ml))

    n_vertices = mesh_a_tm.vertices.shape[0] + mesh_b_tm.vertices.shape[0]
    assert merged_tm.vertices.shape[0] == n_vertices  # non-vacuity: nothing was welded away
    assert merged_tm.faces.shape[0] == mesh_a_tm.faces.shape[0] + mesh_b_tm.faces.shape[0]
    assert np.allclose(concat_vertices_wp.numpy(), merged_tm.vertices, rtol=1e-5, atol=1e-5)
    assert np.array_equal(concat_faces_wp.numpy().reshape(-1, 3), merged_tm.faces)


@pytest.mark.parity("concatenate", "pytorch3d")
def test_concatenate_matches_pytorch3d(request: pytest.FixtureRequest, device: str) -> None:
    """
    Class A: ``join_meshes_as_scene`` is ``concatenate`` -- positions exact, faces byte-equal.

    "As a scene" is the operation that matters: it concatenates the vertex buffers and shifts each
    mesh's face indices by the running vertex count, which is ordito's packing verbatim. The
    faces comparison is positional and passes, so the two agree on the *order* of the meshes and
    not merely on the resulting soup -- the sibling ``join_meshes_as_batch`` would keep them as a
    minibatch instead and is not this operation.
    """
    meshes_tm = [request.getfixturevalue(name)[0] for name in CLOSED_MESHES]
    joined_p3d = p3d_structures.join_meshes_as_scene(
        [trimesh_to_pytorch3d(mesh_tm) for mesh_tm in meshes_tm]
    )
    meshes_wp = [numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device) for mesh_tm in meshes_tm]
    vertices_wp, faces_wp = od.combine.concatenate(meshes_wp)

    verts_p3d = joined_p3d.verts_packed()
    faces_p3d = joined_p3d.faces_packed()
    assert verts_p3d is not None
    assert faces_p3d is not None
    assert verts_p3d.shape[0] == sum(len(m.vertices) for m in meshes_tm)
    assert np.array_equal(vertices_wp.numpy(), verts_p3d.numpy())
    assert np.array_equal(faces_wp.numpy().reshape(-1, 3), faces_p3d.numpy())


def test_concatenate_empty() -> None:
    vertices_wp, faces_wp = od.combine.concatenate([])
    assert vertices_wp.shape == (0,)
    assert faces_wp.shape == (0,)


@pytest.mark.parity("split", "trimesh")
def test_split_meshes(three_parts: list[MeshPair]) -> None:
    """
    Ordito against ordito: ``split`` inverts ``concatenate`` on three known components.

    The component *count* is the reference-checkable part and is pinned separately by
    [`test_split_matches_open3d_and_pymeshlab`]; what only a round trip can check is that each
    component comes back with its own vertices renumbered consistently. ``concatenate`` has its
    own trimesh oracle above, so the loop is not closed on an untested function.
    """
    concat_vertices_wp, concat_faces_wp = _concatenate_parts(three_parts)

    split_wp = od.combine.split(concat_vertices_wp, concat_faces_wp)
    assert len(split_wp) == 3

    roundtrip_vertices_wp, roundtrip_faces_wp = od.combine.concatenate(split_wp)
    assert np.allclose(roundtrip_vertices_wp.numpy(), concat_vertices_wp.numpy())
    assert np.array_equal(roundtrip_faces_wp.numpy(), concat_faces_wp.numpy())

    split_wp_sorted = sorted(split_wp, key=lambda mesh: mesh[1].size)
    meshes_tm_sorted = sorted(
        [mesh_tm for mesh_tm, _ in three_parts], key=lambda mesh: len(mesh.faces)
    )
    for (vertices_wp, faces_wp), mesh_tm in zip(split_wp_sorted, meshes_tm_sorted, strict=True):
        assert np.allclose(vertices_wp.numpy(), mesh_tm.vertices, rtol=1e-5, atol=1e-5)
        assert np.array_equal(faces_wp.numpy(), mesh_tm.faces.reshape(-1))


@pytest.mark.parity("split", "meshlib")
def test_split_finds_all_eight_components_of_one_generated_mesh(
    torus_components: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class B on the partition, on a mesh that arrived as **one buffer** with eight components in it.

    Every other ``split`` test builds its input with ``concatenate``, so the components were
    separate buffers a moment earlier and their faces are already contiguous and index-disjoint.
    This fixture is one generated mesh whose eight open pieces interleave in neither respect, which
    is the input a labelling that leaned on contiguity would get wrong -- and there are eight, where
    the concatenated tests use three.

    Compared against ``getAllComponents`` on face counts and against trimesh's own ``split``, plus
    the covering assert: every face in exactly one component.
    """
    mesh_tm, mesh_wp = torus_components
    n_faces = mesh_tm.faces.shape[0]
    parts_wp = od.combine.split(mesh_wp.points, mesh_wp.indices)

    components_ml = mm.getAllComponents(
        mm.MeshPart(trimesh_to_meshlib(mesh_tm)), mm.MeshComponents.FaceIncidence.PerEdge
    )
    counts_ml = sorted(component_ml.count() for component_ml in components_ml)
    counts_tm = sorted(part_tm.faces.shape[0] for part_tm in mesh_tm.split(only_watertight=False))

    assert len(counts_ml) == 8  # non-vacuity: the reference really sees eight pieces
    assert counts_ml == counts_tm
    assert sorted(faces_wp.size // 3 for _vertices_wp, faces_wp in parts_wp) == counts_ml

    covered_np = np.zeros(n_faces, dtype=int)
    for component_ml in components_ml:
        covered_np += meshlib_bitset_to_numpy(component_ml, n_faces)
    assert (covered_np == 1).all()


@pytest.mark.parity("split", "open3d", "pymeshlab")
def test_split_matches_open3d_and_pymeshlab(three_parts: list[MeshPair]) -> None:
    """
    Class B against both references, each of which returns the components a different way.

    **Open3D** has no single split call: ``cluster_connected_triangles`` returns the *labelling*, so
    the named transform is the compaction the benchmark also pays -- ``select_by_index`` per label,
    fed the label's vertex indices via ``np.unique``. **MeshLab** does compact, but
    ``generate_splitting_by_connected_components`` *pushes* the components onto the MeshSet after
    the original, so the transform is to skip mesh 0 and read ``vertex_matrix`` / ``face_matrix``
    off each of the rest.

    None of the three defines a component order, so all three lists are sorted by face count -- the
    fixtures have 20, 168 and 1 024 faces, so that pairing is unambiguous -- and each paired
    component is compared by its face-centroid set through a bijective nearest-neighbour match.
    """
    combined_tm = tm.util.concatenate([mesh_tm for mesh_tm, _ in three_parts])
    assert isinstance(combined_tm, tm.Trimesh)
    concat_vertices_wp, concat_faces_wp = _concatenate_parts(three_parts)

    mesh_o3d = trimesh_to_open3d(combined_tm)
    labels_o3d = np.asarray(mesh_o3d.cluster_connected_triangles()[0])
    faces_i32 = np.ascontiguousarray(combined_tm.faces, dtype=np.int32)
    parts_o3d = [
        open3d_to_trimesh(
            mesh_o3d.select_by_index(
                cast("Sequence[int]", np.unique(faces_i32[labels_o3d == label]))
            )
        )
        for label in range(int(labels_o3d.max()) + 1)
    ]

    meshset_pml = trimesh_to_pymeshlab(combined_tm)
    meshset_pml.generate_splitting_by_connected_components()
    parts_pml = []
    for index in range(1, meshset_pml.mesh_number()):
        meshset_pml.set_current_mesh(index)
        parts_pml.append(
            (
                np.asarray(meshset_pml.current_mesh().vertex_matrix(), dtype=np.float64),
                np.asarray(meshset_pml.current_mesh().face_matrix()),
            )
        )

    parts_wp = [
        (vertices_wp.numpy().astype(np.float64), faces_wp.numpy().reshape(-1, 3))
        for vertices_wp, faces_wp in od.combine.split(concat_vertices_wp, concat_faces_wp)
    ]
    assert len(parts_wp) == len(parts_o3d) == len(parts_pml) == 3

    def by_faces(
        parts: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return sorted(parts, key=lambda part: part[1].shape[0])

    reference_parts = [(part.vertices, part.faces) for part in parts_o3d]
    for part_wp, part_o3d, part_pml in zip(
        by_faces(parts_wp), by_faces(reference_parts), by_faces(parts_pml), strict=True
    ):
        for vertices_ref, faces_ref in (part_o3d, part_pml):
            assert part_wp[0].shape[0] == vertices_ref.shape[0]
            assert part_wp[1].shape[0] == faces_ref.shape[0]
            centroids_wp = part_wp[0][part_wp[1]].mean(axis=1)
            centroids_ref = vertices_ref[faces_ref].mean(axis=1)
            distance_np, match_np = cKDTree(centroids_ref).query(centroids_wp)
            assert np.max(distance_np) < 1e-5
            assert len(set(np.asarray(match_np).tolist())) == np.asarray(match_np).size


def test_split_with_offsets_matches_split(three_parts: list[MeshPair]) -> None:
    """Ordito against ordito: ``split`` is ``split_with_offsets`` split, slice for slice."""
    concat_vertices_wp, concat_faces_wp = _concatenate_parts(three_parts)

    vertices_all_wp, vertex_offsets_wp, faces_all_wp, face_offsets_wp = (
        od.combine.split_with_offsets(concat_vertices_wp, concat_faces_wp)
    )
    split_wp = od.combine.split(concat_vertices_wp, concat_faces_wp)
    assert vertex_offsets_wp.size == len(split_wp) + 1 == 4

    vertex_bounds_np = vertex_offsets_wp.list()
    face_bounds_np = face_offsets_wp.list()
    assert vertex_bounds_np[-1] == vertices_all_wp.size
    assert face_bounds_np[-1] == faces_all_wp.size // 3
    for index, (vertices_wp, faces_wp) in enumerate(split_wp):
        v_begin, v_end = vertex_bounds_np[index], vertex_bounds_np[index + 1]
        f_begin, f_end = face_bounds_np[index], face_bounds_np[index + 1]
        assert np.array_equal(vertices_all_wp.numpy()[v_begin:v_end], vertices_wp.numpy())
        assert np.array_equal(faces_all_wp.numpy()[3 * f_begin : 3 * f_end], faces_wp.numpy())

    # ``copy=True`` returns the same data in independent buffers.
    for (view_vertices_wp, view_faces_wp), (copy_vertices_wp, copy_faces_wp) in zip(
        split_wp, od.combine.split(concat_vertices_wp, concat_faces_wp, copy=True), strict=True
    ):
        assert np.array_equal(view_vertices_wp.numpy(), copy_vertices_wp.numpy())
        assert np.array_equal(view_faces_wp.numpy(), copy_faces_wp.numpy())
        assert copy_vertices_wp.ptr != view_vertices_wp.ptr


def test_split_copies_many_components(device: str) -> None:
    """
    Ordito against ordito: forty components copy out equal to their views, into owned buffers.

    Forty is past the count at which ``array.split(copy=True)`` fills its copies in one launch, the
    path a handful of components never reaches. The views carry the oracle through
    ``test_split_meshes``; this pins the copies to them.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=0)
    parts_tm = [sphere_tm.copy().apply_translation([3.0 * i, 0.0, 0.0]) for i in range(40)]
    mesh_tm = tm.util.concatenate(parts_tm)
    assert isinstance(mesh_tm, tm.Trimesh)
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces.reshape(-1), device)

    views_wp = od.combine.split(vertices_wp, faces_wp)
    copies_wp = od.combine.split(vertices_wp, faces_wp, copy=True)

    assert len(views_wp) == len(copies_wp) == 40
    for (view_vertices_wp, view_faces_wp), (copy_vertices_wp, copy_faces_wp) in zip(
        views_wp, copies_wp, strict=True
    ):
        assert np.array_equal(view_vertices_wp.numpy(), copy_vertices_wp.numpy())
        assert np.array_equal(view_faces_wp.numpy(), copy_faces_wp.numpy())
        assert copy_faces_wp.size == 3 * sphere_tm.faces.shape[0]
        assert copy_vertices_wp._ref is None  # pyright: ignore[reportPrivateUsage]
        assert copy_faces_wp._ref is None  # pyright: ignore[reportPrivateUsage]


def test_split_single_component(request: pytest.FixtureRequest) -> None:
    """Ordito against ordito: the ``k == 1`` fast path equals the batched key packing."""
    mesh_tm, mesh_wp = request.getfixturevalue("sphere_irregular")
    split_wp = od.combine.split(mesh_wp.points, mesh_wp.indices)
    assert len(split_wp) == 1
    assert np.allclose(split_wp[0][0].numpy(), mesh_tm.vertices, rtol=1e-5, atol=1e-5)
    assert np.array_equal(split_wp[0][1].numpy(), mesh_tm.faces.reshape(-1))


def test_split_empty(device: str) -> None:
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = warp_empty(0, wp.int32, device)
    assert od.combine.split(vertices_wp, faces_wp) == []
    vertices_all_wp, vertex_offsets_wp, faces_all_wp, face_offsets_wp = (
        od.combine.split_with_offsets(vertices_wp, faces_wp)
    )
    assert vertices_all_wp.shape == (0,)
    assert np.array_equal(vertex_offsets_wp.numpy(), np.zeros(1, dtype=np.int32))
    assert faces_all_wp.shape == (0,)
    assert np.array_equal(face_offsets_wp.numpy(), np.zeros(1, dtype=np.int32))


def test_concatenate_rejects_mismatched_devices() -> None:
    """
    Ordito against ordito: a cross-device ``(vertices, faces)`` pair must be rejected.

    Not a library comparison: no reference library shares Warp's device model. This one is not
    covered by the family test in ``tests/test_array.py`` because the mismatch here is *inside* a
    sequence argument rather than between two named ones -- ``concatenate`` takes one
    ``meshes_data`` parameter, so the guard has to descend into it. The failure it prevents is not
    an exception: the launch device is taken from the first pair's ``vertices`` while the buffer it
    writes is packed onto its ``faces``' device, so a CPU launch against CUDA pointers segfaults
    with no Python traceback (CLAUDE.md section 12.1).
    """
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to construct a mismatch")
    vertices_wp = wp.zeros(3, dtype=wp.vec3, device="cpu")
    faces_wp = wp.zeros(3, dtype=wp.int32, device="cuda:0")
    with pytest.raises(RuntimeError, match="one device"):
        od.combine.concatenate([(vertices_wp, faces_wp)])
    # A mismatch between two *pieces*, which the same descent has to catch.
    with pytest.raises(RuntimeError, match="one device"):
        od.combine.concatenate(
            [
                (vertices_wp, wp.zeros(3, dtype=wp.int32, device="cpu")),
                (wp.zeros(3, dtype=wp.vec3, device="cuda:0"), faces_wp),
            ]
        )
