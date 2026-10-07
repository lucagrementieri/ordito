from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito.kernels.grouping import VEC3_PACK_PRECISION, VEC3_PACK_SHIFT
from tests.comparisons import lexsort_rows, same_partition
from tests.conversions import points_to_warp, warp_empty

# Host data, uploaded per test onto the fixture's device -- not ``wp.array`` at module scope. A
# module-level ``wp.array`` with no ``device=`` lands on Warp's *current* device, which is
# ``cuda:0`` here, and re-wrapping it onto another device raises ``Item indexing is not supported on
# wp.array objects`` from deep inside ``wp.array.__init__``. That was invisible while the ``device``
# fixture returned one device: the allocation and the test always agreed. It is the defect
# ``api_conventions`` check 14 exists for, in the one tree that check does not scan.
group_test_data = (
    (np.array([1, 3, 2, 3, 4, 4, 7, 5, -1, 5, 5], dtype=np.int32), 2, [[1, 3], [4, 5]]),
    (
        np.array([0, 1, 2, 1, 5, 6, 1, 0, 0, 0, 6, 4, 6], dtype=np.uint64),
        3,
        [[1, 3, 6], [5, 10, 12]],
    ),
    (np.array([-1, 3, 2, -3, 4, 2, -1, 2, 2, 2], dtype=np.int64), 4, []),
    # High-bit uint64 keys sort natively as unsigned (after low keys) in Warp 1.18.
    (np.array([2**63 + 5, 1, 2**63 + 5, 1], dtype=np.uint64), 2, [[1, 3], [0, 2]]),
)


def _int_rows(rows_np: np.ndarray | list[list[int]], device: str) -> odt.Array2dInt32:
    """Upload host integer rows as a rank-2 ``int32`` array on ``device``."""
    return odt.as_array2d(
        wp.array(np.ascontiguousarray(rows_np, dtype=np.int32), dtype=wp.int32, device=device),
        wp.int32,
    )


@pytest.mark.parametrize(("values_np", "length", "expected_rows"), group_test_data)
def test_group(
    device: str, values_np: np.ndarray, length: int, expected_rows: list[list[int]]
) -> None:
    """Class A against a hand-written expectation, over the four key dtypes the radix sort takes."""
    groups_np = np.array(expected_rows, dtype=np.int32).reshape(-1, length)

    values_wp = wp.array(values_np, device=device)
    groups_wp = od.grouping.group(values_wp, length)
    assert values_wp.dtype.__name__ == values_np.dtype.name  # the dtype under test really landed
    assert np.array_equal(groups_wp.numpy(), groups_np)


@pytest.mark.parity("group", "trimesh")
@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "saddle_graded"])
def test_group_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (row and group order): ``trimesh.grouping.group`` over the same edge inverse.

    Both sides answer "which index sets share a value, at exactly this multiplicity", and on a
    mesh's unique-edge inverse the ``length=2`` groups are the adjacent face pairs -- the workload
    the benchmark row times. Neither library promises an order: ordito emits groups in
    radix-sorted key order and trimesh in ``argsort`` order, and within a group neither fixes
    which member comes first, so the named transform is a sort along both axes. The count assert
    is what makes that sound -- a sort cannot rescue two answers that disagree about *how many*
    groups there are.

    The boundary case is the point of running ``half_torus`` as well: an open mesh's rim edges
    appear once, so they must be absent from both answers rather than padded into either.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    inverse_wp = od.edges.edges_unique_inverse(mesh_wp.indices, n_vertices=n_vertices)
    inverse_np = inverse_wp.numpy()

    groups_wp = od.grouping.group(inverse_wp, 2).numpy()
    groups_tm = np.asarray(tm.grouping.group(inverse_np, min_len=2, max_len=2))

    # Non-vacuity, and the invariant that fixes the expected count: every interior edge is shared by
    # exactly two faces, so there is one pair per interior edge and none per boundary edge.
    n_interior = int((np.bincount(inverse_np, minlength=inverse_np.max() + 1) == 2).sum())
    assert n_interior > 1
    assert groups_wp.shape == (n_interior, 2)

    assert np.array_equal(
        lexsort_rows(np.sort(groups_wp, axis=1)), lexsort_rows(np.sort(groups_tm, axis=1))
    )


def test_group_int_rows(device: str) -> None:
    """
    Class B (order within a group): ``trimesh.grouping.group_rows`` at the same multiplicity.

    Ordito against ordito for the unvalidated arm: with a ``max_value`` bound, ``validate=False``
    skips the range check and must return the validated groups unchanged.
    """
    data_np = np.array([[1, 2], [3, 4], [1, 2], [2, 1], [3, 4], [0, 1], [3, 4]], dtype=np.int32)
    length = 2
    groups_np = np.sort(tm.grouping.group_rows(data_np, require_count=length), axis=1)

    data_wp = _int_rows(data_np, device)
    groups_wp = od.grouping.group_int_rows(data_wp, length)
    assert np.array_equal(np.sort(groups_wp.numpy(), axis=1), groups_np)

    bounded_wp = od.grouping.group_int_rows(data_wp, length, max_value=7)
    unvalidated_wp = od.grouping.group_int_rows(data_wp, length, max_value=7, validate=False)
    assert np.array_equal(np.sort(bounded_wp.numpy(), axis=1), groups_np)
    assert np.array_equal(unvalidated_wp.numpy(), bounded_wp.numpy())


@pytest.mark.parametrize(
    ("dtype", "data_list"),
    [
        pytest.param(wp.int32, [0, 1, 20, 3, 1, 3, 10, 20], id="int32"),
        pytest.param(wp.float32, [20.0, 10.0, 2.0, 3.0, 1.0, 3.0, 10.0, 20.0], id="float32"),
        pytest.param(wp.int64, [20, 10, 2, 3, 1, 3, 10, 20], id="int64"),
        pytest.param(wp.uint64, [20, 10, 20, 3, 1, 3, 10, 20], id="uint64"),
    ],
)
def test_unique_1d(device: str, dtype: type, data_list: list[float]) -> None:
    """
    Class A against ``numpy.unique``: values alone, then with the inverse and the counts.

    One arm per dtype the radix sort takes, the ``float32`` arm being the one that keeps the hash
    table rather than the sort.
    """
    data_np = np.array(data_list, dtype=wp.dtype_to_numpy(dtype))
    unique_np, inverse_np, counts_np = np.unique(data_np, return_inverse=True, return_counts=True)

    data_wp = wp.array(data_np, dtype=dtype, device=device)
    assert np.array_equal(od.grouping.unique_1d(data_wp).numpy(), unique_np)
    unique_wp, inverse_wp, counts_wp = od.grouping.unique_1d(
        data_wp, return_inverse=True, return_counts=True
    )
    assert np.array_equal(unique_wp.numpy(), unique_np)
    assert np.array_equal(inverse_wp.numpy(), inverse_np.ravel())
    assert np.array_equal(counts_wp.numpy(), counts_np)


def test_unique_1d_inverse_float_with_nan(device: str):
    """
    Class A against ``numpy.unique``.

    Regression: the inverse used a ``side="right"`` binary search, which ``NaN`` poisons --
    every comparison against the sorted array's ``NaN`` tail is false, so the search walked into
    it and reported the last slot for every *finite* value
    (measured ``[2, 0, 2, 2, 0]`` against numpy's ``[1, 0, 1, 2, 0]``). Non-vacuous by
    construction: the two asserts below pin that the finite values do not collapse onto one slot
    and that ``NaN`` still gets its own.
    """
    data_np = np.array([3.5, 1.25, 3.5, np.nan, 1.25, np.nan], dtype=np.float32)
    unique_np, inverse_np = np.unique(data_np, return_inverse=True)
    assert np.unique(inverse_np).size == 3

    data_wp = wp.array(data_np, dtype=wp.float32, device=device)
    unique_wp, inverse_wp = od.grouping.unique_1d(data_wp, return_inverse=True)
    assert np.array_equal(unique_wp.numpy(), unique_np, equal_nan=True)
    assert np.array_equal(inverse_wp.numpy(), inverse_np.ravel())


@pytest.mark.parametrize("dtype", [wp.int32, wp.uint64])
def test_unique_1d_max_value(device: str, dtype: type) -> None:
    """
    Class A against ``numpy.unique``, at the tightest ``max_value`` the data allows.

    The bound only narrows the final sort to the bits it needs, so the answer must be the unbounded
    one. The values span a few thousand distinct keys drawn to leave the top bit of the bound set,
    so a sort that dropped one bit too many would misorder them; the ``int32`` arm reaches the
    signed sort path, where the narrowed range stops short of the sign bit.
    """
    rng = np.random.default_rng(20)
    data_np = rng.integers(0, 5000, 20000).astype(np.uint64 if dtype == wp.uint64 else np.int32)
    unique_np, inverse_np, counts_np = np.unique(data_np, return_inverse=True, return_counts=True)
    assert int(unique_np.max()).bit_length() == 13

    data_wp = wp.array(data_np, dtype=dtype, device=device)
    unique_wp, inverse_wp, counts_wp = od.grouping.unique_1d(
        data_wp, return_inverse=True, return_counts=True, max_value=int(unique_np.max())
    )
    assert np.array_equal(unique_wp.numpy(), unique_np)
    assert np.array_equal(inverse_wp.numpy(), inverse_np.ravel())
    assert np.array_equal(counts_wp.numpy(), counts_np)


def test_unique_1d_max_value_invalid(device: str) -> None:
    """Not a library comparison: the two inputs ``max_value`` rejects."""
    ints_wp = wp.array(np.arange(4, dtype=np.int32), dtype=wp.int32, device=device)
    floats_wp = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="non-negative"):
        od.grouping.unique_1d(ints_wp, max_value=-1)
    with pytest.raises(ValueError, match="integer data only"):
        od.grouping.unique_1d(floats_wp, max_value=3)


def test_unique_rows_inverse_counts(device: str):
    data_np = np.array([[0, 1], [2, 3], [0, 1], [2, 3], [4, 5]], dtype=np.int32)
    data_wp = wp.array(data_np, dtype=wp.int32, device=device)
    unique_wp, inverse_wp, counts_wp = od.grouping.unique_rows(
        data_wp, return_inverse=True, return_counts=True
    )
    _, _inverse_np, counts_np = np.unique(data_np, axis=0, return_inverse=True, return_counts=True)
    assert np.array_equal(np.sort(counts_wp.numpy()), np.sort(counts_np))
    for i in range(data_np.shape[0]):
        assert np.array_equal(unique_wp.numpy()[inverse_wp.numpy()[i]], data_np[i])


@pytest.mark.parametrize("unique_fraction", [1.0, 0.1], ids=["allunique", "tenth"])
@pytest.mark.parity("unique_rows", "trimesh")
def test_unique_rows_matches_trimesh(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], device: str, unique_fraction: float
) -> None:
    """
    Class B: ``trimesh.grouping.unique_rows`` returns row *indices*, ordito returns the rows.

    Two named transforms. ``unique_rows_tm[0]`` indexes back into the input to get the rows
    themselves, and neither library defines the output order, so both sides go through
    [`tests.comparisons.lexsort_rows`][]. The inverse map is compared through its defining property
    -- ``unique[inverse[i]] == data[i]`` -- rather than elementwise, because the label *numbering*
    is a function of each library's own output order.

    Parametrized over the same two duplicate densities the benchmark sweeps: every row distinct, and
    a tenth as many distinct rows repeated ten times.
    """
    mesh_tm, _mesh_wp = icosahedron
    faces_np = mesh_tm.faces.astype(np.int32)
    n_unique = max(1, int(faces_np.shape[0] * unique_fraction))
    rows_np = np.ascontiguousarray(faces_np[np.arange(faces_np.shape[0]) % n_unique])
    rows_wp = wp.array(rows_np.reshape(-1), dtype=wp.int32, device=device).reshape(rows_np.shape)

    unique_tm, inverse_tm = tm.grouping.unique_rows(rows_np)
    unique_wp, inverse_wp = od.grouping.unique_rows(rows_wp, return_inverse=True)

    assert np.array_equal(lexsort_rows(unique_wp.numpy()), lexsort_rows(rows_np[unique_tm]))
    assert np.array_equal(unique_wp.numpy()[inverse_wp.numpy()], rows_np[unique_tm][inverse_tm])


def test_unique_rows_vec3(device: str):
    data_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    data_wp = points_to_warp(data_np, device)
    unique_wp, inverse_wp = od.grouping.unique_rows(data_wp, return_inverse=True)
    assert unique_wp.size == 2
    for i in range(data_np.shape[0]):
        assert np.allclose(
            unique_wp.numpy()[inverse_wp.numpy()[i]], data_np[i], rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize(
    ("dtype", "row_shape", "error", "message"),
    [
        pytest.param(wp.int64, (3,), ValueError, "unsupported dtype", id="unsupported-dtype"),
        pytest.param(wp.int32, (), TypeError, None, id="rank-1-not-vec3"),
    ],
)
def test_unique_rows_rejects_empty_and_non_empty_alike(
    device: str,
    dtype: type,
    row_shape: tuple[int, ...],
    error: type[Exception],
    message: str | None,
) -> None:
    """
    An input ``unique_rows`` cannot hash raises the same whether or not it is empty.

    The ``n == 0`` branch must validate the way ``hash_rows`` does on non-empty input: an
    unsupported dtype, and a rank-1 array that is not ``vec3``, each in both sizes.
    """
    empty_wp = warp_empty((0, *row_shape), dtype, device)
    non_empty_wp = wp.ones((1, *row_shape), dtype=dtype, device=device)
    for data_wp in (empty_wp, non_empty_wp):
        with pytest.raises(error, match=message):
            od.grouping.unique_rows(data_wp)


@pytest.mark.parity("unique_faces", "igl", "trimesh")
def test_unique_faces(device: str):
    """
    Class B twice: igl returns the sorted rows and trimesh returns a mask; ordito the winding.

    Both dedup faces up to vertex permutation and both return the inverse map. The difference is the
    *representative*: igl documents ``FF == sort(F(IA, :), 2)``, so its rows come out ascending,
    while ordito keeps the first occurrence's original winding -- the property the last two asserts
    below pin, and the reason the igl comparison sorts ordito's rows first.

    ``Trimesh.unique_faces`` is the third implementation and does strictly less: it returns a
    per-face **bool mask** marking the survivors rather than rebuilding the buffer, so its transform
    is a ``flatnonzero`` and the comparison is on the surviving *set*. It is also
    orientation-agnostic, which is the property this group turns on and the reason it belongs here
    rather than beside ``unique_rows`` -- measured on a sphere plus a flipped copy of 20 of its
    faces, 320 kept of 340.

    The input deliberately contains both a rotation (``[2, 0, 1]``) and a reflection (``[2, 1, 0]``)
    of face 0, so a dedup that collapsed only rotations -- i.e. an orientation-*sensitive* one --
    would report four unique faces instead of three and fail against all three references.
    """
    # Faces sharing the same three vertices (any orientation) collapse to one representative.
    faces_np = np.array(
        [[0, 1, 2], [2, 0, 1], [3, 4, 5], [2, 1, 0], [3, 5, 4], [6, 7, 8]], dtype=np.int32
    )
    faces_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    unique_wp, inverse_wp = od.grouping.unique_faces(faces_wp, return_inverse=True)

    unique_faces_np = unique_wp.numpy().reshape(-1, 3)
    inverse = inverse_wp.numpy()

    sorted_input = np.sort(faces_np, axis=1)
    n_unique_np = np.unique(sorted_input, axis=0).shape[0]
    assert unique_faces_np.shape[0] == n_unique_np == 3

    # Each input face maps to a unique representative sharing its vertex set.
    for i in range(faces_np.shape[0]):
        assert np.array_equal(np.sort(unique_faces_np[inverse[i]]), np.sort(faces_np[i]))
    # Representatives are the first occurrence with original vertex order preserved.
    assert np.array_equal(unique_faces_np[inverse[0]], faces_np[0])
    assert np.array_equal(unique_faces_np[inverse[2]], faces_np[2])

    unique_igl, _representatives_igl, inverse_igl = igl.unique_simplices(
        np.ascontiguousarray(faces_np, dtype=np.int64)
    )[:3]
    assert np.array_equal(lexsort_rows(np.sort(unique_faces_np, axis=1)), lexsort_rows(unique_igl))

    # trimesh returns a survivor *mask* over the input faces, so compare the sets it selects.
    vertices_np = np.zeros((int(faces_np.max()) + 1, 3), dtype=np.float64)
    vertices_np[:, 0] = np.arange(vertices_np.shape[0])
    mask_tm = tm.Trimesh(vertices_np, faces_np, process=False).unique_faces()
    assert int(np.count_nonzero(mask_tm)) == n_unique_np
    assert np.array_equal(
        lexsort_rows(np.sort(faces_np[mask_tm], axis=1)),
        lexsort_rows(np.sort(unique_faces_np, axis=1)),
    )
    # The two inverse maps agree as *partitions* of the input, whatever the slot numbering.
    assert same_partition(inverse, np.asarray(inverse_igl).ravel())


def test_row_operations_on_empty_input(device: str) -> None:
    """
    Empty faces and empty rows reach each function's own empty-input path.

    ``hash_indices_rows`` must not reach ``reduce.minmax``, which raises on an empty array, and
    ``group_int_rows`` must not crash inside that default validation on its way to ``group``.
    """
    unique_wp, inverse_wp = od.grouping.unique_faces(
        warp_empty(0, wp.int32, device), return_inverse=True
    )
    assert unique_wp.size == 0
    assert inverse_wp.size == 0

    empty_wp = odt.as_array2d(warp_empty((0, 3), wp.int32, device), wp.int32)
    packed_wp = od.grouping.hash_indices_rows(empty_wp)
    assert packed_wp.shape == (0,)
    assert packed_wp.dtype == wp.uint64
    packed_unvalidated_wp = od.grouping.hash_indices_rows(empty_wp, max_index=5, validate=False)
    assert packed_unvalidated_wp.shape == (0,)

    assert od.grouping.group_int_rows(empty_wp, 2).shape == (0, 2)


def test_first_occurrence_indices_matches_numpy_return_index(device: str) -> None:
    """
    Class A: the representative of each class is ``np.unique(inverse, return_index=True)``.

    This is the step that makes ``unique_faces`` keep the original winding and ``edges_unique`` keep
    the first-seen edge, so *first* rather than *any* occurrence is the whole contract. The
    ``n_unique=None`` path -- a device reduction plus a host sync -- is checked to agree with the
    explicit one, and an oversized ``n_unique`` is checked to fill the documented ``n`` sentinel.
    """
    values_np = np.array([5, 3, 5, 1, 3, 3, 9], dtype=np.int32)
    unique_wp, inverse_wp = od.grouping.unique_1d(
        wp.array(values_np, dtype=wp.int32, device=device), return_inverse=True
    )
    n_unique = unique_wp.size

    first_wp = od.grouping.first_occurrence_indices(inverse_wp, n_unique)

    _classes_np, expected_np = np.unique(inverse_wp.numpy(), return_index=True)
    assert expected_np.size == n_unique
    assert np.array_equal(first_wp.numpy(), expected_np)
    # Deriving n_unique from the inverse costs a reduction and a sync, and must give the same map.
    assert np.array_equal(od.grouping.first_occurrence_indices(inverse_wp).numpy(), expected_np)

    padded_wp = od.grouping.first_occurrence_indices(inverse_wp, n_unique + 2)
    assert np.array_equal(padded_wp.numpy()[:n_unique], expected_np)
    assert np.array_equal(padded_wp.numpy()[n_unique:], np.full(2, values_np.size, dtype=np.int32))


def test_first_occurrence_indices_picks_representatives_of_duplicate_rows(device: str) -> None:
    """Gathering by the result reproduces the unique array that came back beside the inverse."""
    rows_np = np.array([[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]], dtype=np.int32)
    rows_wp = wp.array(np.ascontiguousarray(rows_np), dtype=wp.int32, device=device)
    unique_wp, inverse_wp = od.grouping.unique_rows(rows_wp, return_inverse=True)

    first_wp = od.grouping.first_occurrence_indices(inverse_wp, int(unique_wp.shape[0]))

    assert int(unique_wp.shape[0]) == 3
    assert np.array_equal(rows_np[first_wp.numpy()], unique_wp.numpy())


@pytest.mark.parametrize("kind", ["vec3", "int32_rows", "float32_rows", "strided_float32_rows"])
def test_hash_rows_dispatches_to_the_typed_hashers(device: str, kind: str) -> None:
    """
    Class A: the dispatcher returns exactly what the function it forwards to returns.

    The strided arm is a column-step view of a wider table, which a ``view`` cannot retype and
    which used to raise from a flatten inside the conversion.
    """
    positions_np = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    if kind == "vec3":
        data_wp = points_to_warp(positions_np, device)
        expected_wp = od.grouping.hash_vector_rows(data_wp)
    elif kind == "int32_rows":
        rows_np = np.array([[1, 2], [3, 4], [1, 2]], dtype=np.int32)
        data_wp = _int_rows(rows_np, device)
        expected_wp = od.grouping.hash_indices_rows(data_wp)
    elif kind == "float32_rows":
        data_wp = wp.array(np.ascontiguousarray(positions_np), dtype=wp.float32, device=device)
        expected_wp = od.grouping.hash_vector_rows(points_to_warp(positions_np, device))
    else:
        wide_np = np.zeros((3, 6), dtype=np.float32)
        wide_np[:, ::2] = positions_np
        data_wp = odt.as_dense(wp.array(wide_np, dtype=wp.float32, device=device)[:, ::2])
        assert not data_wp.is_contiguous
        expected_wp = od.grouping.hash_vector_rows(points_to_warp(positions_np, device))

    keys_np = od.grouping.hash_rows(data_wp).numpy()

    assert np.array_equal(keys_np, expected_wp.numpy())
    # Equal rows must collide and unequal ones must not, or the dispatch proves nothing.
    assert keys_np[0] == keys_np[2]
    assert keys_np[0] != keys_np[1]


def test_hash_rows_rejects_a_dtype_and_a_width_it_cannot_pack(device: str) -> None:
    """The two documented ``ValueError`` paths: an unsupported dtype and a non-width-3 float32."""
    with pytest.raises(ValueError, match="unsupported dtype"):
        od.grouping.hash_rows(wp.zeros((2, 3), dtype=wp.float64, device=device))
    with pytest.raises(ValueError, match="width 3"):
        od.grouping.hash_rows(wp.zeros((2, 2), dtype=wp.float32, device=device))


def test_hash_vector_rows(device: str) -> None:
    rng = np.random.default_rng(17)
    n = 256
    vectors_np = rng.standard_normal((n, 3), dtype=np.float64)
    packed_np = _pack_vec3_np(vectors_np)

    vectors_wp = points_to_warp(vectors_np, device)
    packed_wp = od.grouping.hash_vector_rows(vectors_wp)
    packed = packed_wp.numpy()

    assert np.array_equal(packed, packed_np)

    vectors_double_wp = wp.array(vectors_np, dtype=wp.vec3d, device=device)
    with pytest.raises(ValueError, match=r"data must be a wp\.array\[wp\.vec3\]"):
        _ = od.grouping.hash_vector_rows(vectors_double_wp)


def test_hash_vector_rows_folds_signed_zero(device: str) -> None:
    # IEEE-754's two zeros compare equal, so they have to share a key. Left unfolded, the sign bit
    # is the most significant bit of the bucket and survives the shift, which is how a revolved
    # pole (``cos(theta) * 0.0`` is -0.0 for half the slices) fails to match itself.
    vectors_wp = wp.array(
        np.array([[0.0, 0.0, 5.0], [-0.0, -0.0, 5.0], [-0.0, 0.0, 5.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    assert len(set(od.grouping.hash_vector_rows(vectors_wp).list())) == 1


def test_hash_vector_rows_epsilon_allows_negative_coordinates(device: str) -> None:
    # The rounded grid indices are negative for any mesh spanning the origin. Packing them requires
    # a positive radix, so the grid is translated by a whole number of cells first, which must not
    # move a cell boundary: the +-1e-9 pairs below still merge and the two sites stay distinct.
    vertices_np = np.array(
        [[-1.0, -2.0, -3.0], [-1.0 + 1e-9, -2.0, -3.0], [4.0, 5.0, 6.0], [4.0, 5.0 + 1e-9, 6.0]],
        dtype=np.float32,
    )
    vertices_wp = points_to_warp(vertices_np, device)
    keys_np = od.grouping.hash_vector_rows(vertices_wp, epsilon=1e-6).numpy()
    assert keys_np[0] == keys_np[1]
    assert keys_np[2] == keys_np[3]
    assert keys_np[0] != keys_np[2]

    # Translating the input must not change how the rows group, only the keys themselves.
    shifted_wp = points_to_warp(vertices_np + np.float32(100.0), device)
    shifted_np = od.grouping.hash_vector_rows(shifted_wp, epsilon=1e-6).numpy()
    assert shifted_np[0] == shifted_np[1]
    assert shifted_np[2] == shifted_np[3]
    assert shifted_np[0] != shifted_np[2]


def test_hash_vector_rows_epsilon_far_from_origin(device: str) -> None:
    # Cells are measured from the data's own minimum corner, not from the coordinate origin. That
    # is what keeps a small epsilon meaningful far from zero: scaling a coordinate near 1e4 by 1e6
    # would exceed float32's ~7 digits, quantising away the cell index, and it keeps the packing
    # radix at the size of the extent so the row keys stay injective.
    offset_np = np.float32(1.0e4)
    rng = np.random.default_rng(5)
    sites_np = (rng.random((64, 3)).astype(np.float32) + offset_np).astype(np.float32)
    # Each site duplicated exactly, so the 128 rows must collapse to 64 distinct keys.
    vertices_wp = points_to_warp(np.vstack((sites_np, sites_np)), device)
    keys_np = od.grouping.hash_vector_rows(vertices_wp, epsilon=1e-6).numpy()
    assert np.array_equal(keys_np[:64], keys_np[64:])
    assert len(set(keys_np.tolist())) == 64


def test_hash_indices_rows_valid(device: str) -> None:
    rng = np.random.default_rng(23)
    n_rows, n_cols = 64, 5
    actual_max_index = 17
    max_index = actual_max_index + 3
    indices_np = rng.integers(0, actual_max_index, size=(n_rows, n_cols), dtype=np.int32)
    packed_np = _pack_indices_rows_np(indices_np, max_index)
    packed_default_np = _pack_indices_rows_np(indices_np)

    indices_wp = _int_rows(indices_np, device)
    packed_wp = od.grouping.hash_indices_rows(indices_wp, max_index=max_index)
    packed_default_wp = od.grouping.hash_indices_rows(indices_wp)
    assert np.array_equal(packed_wp.numpy(), packed_np)
    assert np.array_equal(packed_default_wp.numpy(), packed_default_np)


def test_hash_indices_rows_invalid(device: str) -> None:
    max_index = 8
    indices_wp = _int_rows([[0, 1, 2], [-3, 4, 1]], device)

    with pytest.raises(ValueError, match="data must be non-negative, got a minimum of -3"):
        _ = od.grouping.hash_indices_rows(indices_wp, max_index=max_index)

    indices_oob = _int_rows([[0, 1, 2], [3, 8, 1]], device)
    with pytest.raises(ValueError, match="data must be less than max_index 8, got a maximum of 8"):
        _ = od.grouping.hash_indices_rows(indices_oob, max_index=max_index)

    with pytest.raises(ValueError, match="max_index must be positive, got 0"):
        _ = od.grouping.hash_indices_rows(indices_wp, max_index=0)

    indices_np_ok = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    indices_ok = _int_rows(indices_np_ok, device)
    packed_np = _pack_indices_rows_np(indices_np_ok, max_index)
    packed_wp = od.grouping.hash_indices_rows(indices_ok, max_index=max_index)
    assert np.array_equal(packed_wp.numpy(), packed_np)


def test_hash_indices_rows_unvalidated(device: str) -> None:
    rng = np.random.default_rng(11)
    max_index = 23
    indices_np = rng.integers(0, max_index, size=(64, 2), dtype=np.int32)
    indices_wp = _int_rows(indices_np, device)

    # Skipping validation must not change the keys, only the range check that produces them.
    validated_wp = od.grouping.hash_indices_rows(indices_wp, max_index=max_index)
    unvalidated_wp = od.grouping.hash_indices_rows(indices_wp, max_index=max_index, validate=False)
    assert np.array_equal(unvalidated_wp.numpy(), validated_wp.numpy())
    assert np.array_equal(unvalidated_wp.numpy(), _pack_indices_rows_np(indices_np, max_index))

    # Two columns need no radix at all: every int32 reinterpreted as uint32 is below
    # INDEX_RADIX_PAIR, so packing against it is injective without reducing the rows. The keys are
    # different numbers from the tight radix's, but they have to induce the same *order* -- both
    # are monotone lexicographic in (row[1], row[0]) -- which is what every caller relies on.
    pair_wp = od.grouping.hash_indices_rows(indices_wp, validate=False)
    assert np.array_equal(
        pair_wp.numpy(), _pack_indices_rows_np(indices_np, od.constants.INDEX_RADIX_PAIR)
    )
    assert np.array_equal(
        np.argsort(pair_wp.numpy(), kind="stable"), np.argsort(validated_wp.numpy(), kind="stable")
    )
    # Non-vacuity: the fixture really does contain rows that differ only in the second column, so
    # the order assertion above is testing the radix and not a coincidence of a 64-row sample.
    assert len(np.unique(indices_np[:, 1])) > 1

    # A wider row still needs one, because its radix has to keep ``radix ** w`` inside a uint64.
    wide_wp = _int_rows(rng.integers(0, max_index, size=(64, 3), dtype=np.int32), device)
    with pytest.raises(ValueError, match="validate=False requires an explicit max_index"):
        _ = od.grouping.hash_indices_rows(wide_wp, validate=False)


def _pack_vec3_np(vectors_np: np.ndarray) -> np.ndarray:
    if vectors_np.dtype != np.float32:
        vectors_np = vectors_np.astype(np.float32)
    bits = vectors_np.view(np.uint32)
    # IEEE-754's two zeros compare equal, so -0.0 folds onto +0.0 rather than keeping its sign bit.
    bits = np.where(vectors_np == 0.0, np.uint32(0), bits)
    ix = bits[:, 0].astype(np.uint64) >> VEC3_PACK_SHIFT.value
    iy = bits[:, 1].astype(np.uint64) >> VEC3_PACK_SHIFT.value
    iz = bits[:, 2].astype(np.uint64) >> VEC3_PACK_SHIFT.value
    return ix | (iy << VEC3_PACK_PRECISION.value) | (iz << (2 * VEC3_PACK_PRECISION.value))


def _pack_indices_rows_np(indices_np: np.ndarray, max_index: int | None = None) -> np.ndarray:
    """CPU reference for ``pack_indices``: mixed-radix sum with wrapping ``uint64`` math."""
    radix = np.max(indices_np) + 1 if max_index is None else max_index
    return np.sum(indices_np * np.power(radix, np.arange(indices_np.shape[1])), axis=1)
