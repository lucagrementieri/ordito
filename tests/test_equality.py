from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from scipy.spatial import cKDTree

import ordito as od
import ordito.typing as odt
from tests.comparisons import canonical_winding, lexsort_rows, same_partition
from tests.conftest import MESHES
from tests.conversions import numpy_to_warp, points_to_warp, warp_empty


def _int_rows(rows_np: np.ndarray, device: str) -> odt.Array2dInt32:
    """Upload host integer rows as a rank-2 ``int32`` array on ``device``."""
    return odt.as_array2d(
        wp.array(np.ascontiguousarray(rows_np, dtype=np.int32), dtype=wp.int32, device=device),
        wp.int32,
    )


def _shuffled(
    vertices_np: np.ndarray, faces_np: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mesh with its vertices renumbered, its faces reordered and each face rotated."""
    rng = np.random.default_rng(seed)
    order_np = rng.permutation(len(vertices_np))
    new_index_np = np.argsort(order_np)
    faces_np = new_index_np[faces_np][rng.permutation(len(faces_np))]
    roll_np = rng.integers(0, 3, len(faces_np))
    faces_np = np.take_along_axis(faces_np, (roll_np[:, None] + np.arange(3)) % 3, axis=1)
    return vertices_np[order_np], faces_np


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parametrize("n_vertices_bound", [None, 1 << 22], ids=["one_sort", "two_sorts"])
def test_canonical_faces_matches_numpy(
    request: pytest.FixtureRequest, mesh_name: str, n_vertices_bound: int | None
) -> None:
    """
    Not a library comparison: no reference names a canonical face order. Class A against NumPy.

    The oracle is the per-row rotation to the smallest corner and a ``lexsort`` of the rows. A bound
    of ``2 ** 22`` vertices no longer packs three indices into one 64-bit key, so it takes the
    two-sort path on the same faces; both must equal the oracle element for element.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _vertices_np, faces_np = _shuffled(np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces), 0)
    faces_wp = wp.array(faces_np.ravel().astype(np.int32), dtype=wp.int32, device=mesh_wp.device)
    canonical_np = lexsort_rows(canonical_winding(faces_np))
    assert not np.array_equal(canonical_np, faces_np)  # the input really was out of order

    canonical_wp = od.equality.canonical_faces(faces_wp, n_vertices=n_vertices_bound)
    assert np.array_equal(canonical_wp.numpy().reshape(-1, 3), canonical_np)


def test_canonical_faces_validates_its_input(device: str) -> None:
    """Not a parity assert: the empty input, a partial triangle and a negative index."""
    empty_wp = warp_empty(0, wp.int32, device)
    assert od.equality.canonical_faces(empty_wp).size == 0
    with pytest.raises(ValueError, match="whole number of triangles"):
        od.equality.canonical_faces(wp.array([0, 1], dtype=wp.int32, device=device))
    with pytest.raises(ValueError, match="non-negative"):
        od.equality.canonical_faces(wp.array([0, 1, -2], dtype=wp.int32, device=device))


@pytest.mark.parametrize("mesh_name", MESHES)
def test_faces_equal(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Not a library comparison: no reference compares face sets. Ordito against a constructed answer.

    Both answers are exercised on every fixture: a reordered and rotated copy is equal, and each of
    one flipped face, one face moved onto another vertex, one duplicated face in place of another
    and one corner past ``n_vertices`` is not.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    device = mesh_wp.device
    faces_np = np.asarray(mesh_tm.faces)
    n_vertices = len(mesh_tm.vertices)
    rng = np.random.default_rng(1)
    shuffled_np = faces_np[rng.permutation(len(faces_np))]
    shuffled_np = np.take_along_axis(
        shuffled_np, (rng.integers(0, 3, len(shuffled_np))[:, None] + np.arange(3)) % 3, axis=1
    )

    def upload(rows_np: np.ndarray) -> wp.array[wp.int32]:
        return wp.array(rows_np.ravel().astype(np.int32), dtype=wp.int32, device=device)

    assert od.equality.faces_equal(mesh_wp.indices, upload(shuffled_np))
    assert od.equality.faces_equal(mesh_wp.indices, upload(shuffled_np), n_vertices=n_vertices)

    flipped_np = shuffled_np.copy()
    flipped_np[7] = flipped_np[7, ::-1]
    moved_np = shuffled_np.copy()
    moved_np[7, 0] = next(v for v in range(n_vertices) if v not in moved_np[7])
    duplicated_np = shuffled_np.copy()
    duplicated_np[7] = duplicated_np[8]
    for changed_np in (flipped_np, moved_np, duplicated_np):
        assert not od.equality.faces_equal(mesh_wp.indices, upload(changed_np))

    beyond_np = shuffled_np.copy()
    beyond_np[7, 0] = n_vertices
    assert not od.equality.faces_equal(mesh_wp.indices, upload(beyond_np), n_vertices=n_vertices)
    assert not od.equality.faces_equal(mesh_wp.indices, upload(shuffled_np[1:]))


@pytest.mark.parametrize("mesh_name", MESHES)
def test_meshes_allclose(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Not a library comparison: no reference compares meshes up to numbering; a constructed answer.

    A renumbered, reordered and rotated copy is equal, also moved by a tenth of the tolerance; each
    of one flipped face, one vertex moved by ten times the tolerance, two vertices swapped in the
    faces and a missing face is not. The tolerance is a third of the closest vertex pair's
    distance, the largest the nearest-neighbour match allows (``saddle_graded``'s needles put two
    vertices 1.4e-5 apart).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    device = mesh_wp.device
    vertices_np = mesh_wp.points.numpy()
    shuffled_vertices_np, shuffled_faces_np = _shuffled(vertices_np, np.asarray(mesh_tm.faces), 2)
    distances_np, _ = cKDTree(vertices_np).query(vertices_np, k=2)
    closest = float(np.asarray(distances_np)[:, 1].min())
    atol = closest / 3.0

    def allclose(vertices_np: np.ndarray, faces_np: np.ndarray) -> bool:
        vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
        return od.equality.meshes_allclose(
            mesh_wp.points, mesh_wp.indices, vertices_wp, faces_wp, rtol=0.0, atol=atol
        )

    assert allclose(shuffled_vertices_np, shuffled_faces_np)
    nudged_np = shuffled_vertices_np + np.float32(0.1 * atol / np.sqrt(3.0))
    assert allclose(nudged_np, shuffled_faces_np)

    flipped_np = shuffled_faces_np.copy()
    flipped_np[3] = flipped_np[3, ::-1]
    assert not allclose(shuffled_vertices_np, flipped_np)
    moved_np = shuffled_vertices_np.copy()
    moved_np[5] += np.float32(10.0 * atol)
    assert not allclose(moved_np, shuffled_faces_np)
    # The positions are untouched; only the faces name two vertices the other way round.
    a, b = shuffled_faces_np[0, 0], shuffled_faces_np[0, 1]
    swapped_np = np.where(
        shuffled_faces_np == a, b, np.where(shuffled_faces_np == b, a, shuffled_faces_np)
    )
    assert not allclose(shuffled_vertices_np, swapped_np)
    assert not allclose(shuffled_vertices_np, shuffled_faces_np[1:])


def test_meshes_allclose_needs_a_bijection(device: str) -> None:
    """
    Not a parity assert: two vertices of ``a`` near one vertex of ``b`` do not compare equal.

    ``a`` holds a vertex twice where ``b`` holds two distinct vertices, one of them within
    tolerance of both copies: each copy's nearest is that one, so the match is not a bijection,
    though every distance passes. The same meshes with their roles exchanged fail the distance.
    """
    vertices_a_np = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    vertices_b_np = np.array([[0, 0, 0], [0, 0, 0.5], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces_np = np.array([[0, 2, 3], [1, 3, 2]])
    vertices_a_wp, faces_wp = numpy_to_warp(vertices_a_np, faces_np, device)
    vertices_b_wp, _faces_b_wp = numpy_to_warp(vertices_b_np, faces_np, device)
    assert not od.equality.meshes_allclose(vertices_a_wp, faces_wp, vertices_b_wp, faces_wp)
    assert not od.equality.meshes_allclose(vertices_b_wp, faces_wp, vertices_a_wp, faces_wp)
    assert od.equality.meshes_allclose(vertices_b_wp, faces_wp, vertices_b_wp, faces_wp)


def test_meshes_allclose_edge_cases(device: str) -> None:
    """Not a parity assert: empty meshes, different counts, an out-of-range corner, bad sizes."""
    mesh_tm = tm.creation.box()
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    empty_vertices_wp = warp_empty(0, wp.vec3, device)
    empty_faces_wp = warp_empty(0, wp.int32, device)
    assert od.equality.meshes_allclose(
        empty_vertices_wp, empty_faces_wp, empty_vertices_wp, empty_faces_wp
    )
    assert not od.equality.meshes_allclose(vertices_wp, faces_wp, empty_vertices_wp, empty_faces_wp)
    # The same faces over a vertex buffer one vertex longer.
    longer_wp, _ = numpy_to_warp(
        np.vstack([mesh_tm.vertices, [[5.0, 5.0, 5.0]]]), mesh_tm.faces, device
    )
    assert not od.equality.meshes_allclose(vertices_wp, faces_wp, longer_wp, faces_wp)
    # Vertices with no face are compared too, and faces past the vertex count are unequal.
    assert od.equality.meshes_allclose(vertices_wp, empty_faces_wp, vertices_wp, empty_faces_wp)
    beyond_np = np.asarray(mesh_tm.faces).copy()
    beyond_np[0, 0] = len(mesh_tm.vertices)
    _, beyond_wp = numpy_to_warp(mesh_tm.vertices, beyond_np, device)
    assert not od.equality.meshes_allclose(vertices_wp, beyond_wp, vertices_wp, faces_wp)
    assert not od.equality.meshes_allclose(vertices_wp, faces_wp, vertices_wp, beyond_wp)
    partial_wp = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="whole number of triangles"):
        od.equality.meshes_allclose(vertices_wp, partial_wp, vertices_wp, partial_wp)


@pytest.mark.parametrize("mesh_name", MESHES)
def test_points_allclose(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Not a library comparison: no reference compares point sets up to order; a constructed answer.

    A shuffled copy is equal, also moved by a tenth of the tolerance; one point moved by ten times
    the tolerance, one point replaced by a copy of another and a missing point are not.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    device = mesh_wp.device
    points_np = mesh_wp.points.numpy()
    distances_np, _ = cKDTree(points_np).query(points_np, k=2)
    atol = float(np.asarray(distances_np)[:, 1].min()) / 3.0
    shuffled_np = points_np[np.random.default_rng(3).permutation(len(points_np))]

    def allclose(other_np: np.ndarray) -> bool:
        return od.equality.points_allclose(
            mesh_wp.points, points_to_warp(other_np, device), rtol=0.0, atol=atol
        )

    assert allclose(shuffled_np)
    assert allclose(shuffled_np + np.float32(0.1 * atol / np.sqrt(3.0)))
    moved_np = shuffled_np.copy()
    moved_np[5] += np.float32(10.0 * atol)
    assert not allclose(moved_np)
    duplicated_np = shuffled_np.copy()
    duplicated_np[5] = duplicated_np[6]
    assert not allclose(duplicated_np)
    assert not allclose(shuffled_np[1:])


@pytest.mark.parametrize("width", [1, 2, 3])
def test_rows_equal(device: str, width: int) -> None:
    """
    Not a library comparison: no reference compares row multisets; Class A against NumPy.

    Rows drawn with repeats and negative entries: a row-shuffled copy is equal, as the ``lexsort``
    oracle says; one entry changed, two entries of a row swapped and a missing row are not.
    """
    rng = np.random.default_rng(4)
    rows_np = rng.integers(-5, 5, size=(400, width)).astype(np.int32)
    assert len(np.unique(rows_np, axis=0)) < len(rows_np)  # repeated rows really occur
    shuffled_np = rows_np[rng.permutation(len(rows_np))]
    assert np.array_equal(lexsort_rows(rows_np), lexsort_rows(shuffled_np))

    def equal(other_np: np.ndarray) -> bool:
        return od.equality.rows_equal(_int_rows(rows_np, device), _int_rows(other_np, device))

    assert equal(shuffled_np)
    changed_np = shuffled_np.copy()
    changed_np[7, 0] = 9
    assert not equal(changed_np)
    if width > 1:
        swapped_np = shuffled_np.copy()
        row = int(np.flatnonzero(swapped_np[:, 0] != swapped_np[:, 1])[0])
        swapped_np[row, :2] = swapped_np[row, 1::-1]
        assert not equal(swapped_np)
    assert not equal(shuffled_np[1:])
    empty_np = np.zeros((0, width), dtype=np.int32)
    assert od.equality.rows_equal(_int_rows(empty_np, device), _int_rows(empty_np, device))


def test_partitions_equal(device: str) -> None:
    """
    Not a library comparison: no reference compares partitions; Class A against the test oracle.

    Random labellings agree with ``tests.comparisons.same_partition`` on both answers: a renamed
    labelling is equal; two classes merged, one class split and one item moved are not.
    """
    rng = np.random.default_rng(5)
    labels_np = rng.integers(-3, 20, size=500).astype(np.int32)
    names_np = rng.permutation(np.arange(100, 123))
    renamed_np = names_np[labels_np + 3].astype(np.int32)
    merged_np = np.where(labels_np == 4, 5, labels_np).astype(np.int32)
    split_np = labels_np.copy()
    split_np[np.flatnonzero(labels_np == 4)[0]] = 99
    moved_np = labels_np.copy()
    moved_np[0] = labels_np[0] + 1

    labels_wp = wp.array(labels_np, dtype=wp.int32, device=device)
    for other_np, expected in (
        (renamed_np, True),
        (merged_np, False),
        (split_np, False),
        (moved_np, False),
    ):
        assert same_partition(labels_np, other_np) == expected
        other_wp = wp.array(other_np, dtype=wp.int32, device=device)
        assert od.equality.partitions_equal(labels_wp, other_wp) == expected
        assert od.equality.partitions_equal(other_wp, labels_wp) == expected
    assert not od.equality.partitions_equal(labels_wp, wp.array(labels_np[:10], device=device))
