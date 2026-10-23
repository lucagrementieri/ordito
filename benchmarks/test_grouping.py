"""
Benchmarks for ``ordito.grouping``: fixed-multiplicity grouping and row deduplication.

``group`` runs the ``face_adjacency`` workload on the scan sweep -- interior edges appear exactly
twice, so ``length=2`` groups are the adjacent face pairs, and the cost is one radix sort over
``3F`` keys plus a segment pass.

``unique_rows`` instead sweeps the **duplicate density** at a fixed input length, because that is
the only thing that can move: the sort is oblivious to how many rows collide, so the density only
changes the size of the output compaction. A gap wider than the output-size ratio would mean the
segment scan is not oblivious, which is the thing worth catching.

There is no open3d equivalent for either: grouping and deduplicating rows of an arbitrary id array
is an array primitive, not a mesh operation, and open3d exposes nothing at that level. **trimesh
covers both**: ``grouping.unique_rows`` for the second and ``grouping.group(values, min_len,
max_len)`` for the first, which is the same operation reached by an ``argsort`` plus a NumPy segment
walk. Both reference branches are capped at ``bunny`` and are handed a host-built input, since a
benchmark input must not come from the code under test.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp

import ordito as od
import ordito.typing as odt
from conftest import BenchCase, skip_larger_than

# Fraction of the rows that are distinct: everything unique, against a tenth as many distinct
# values repeated ten times. The input length is identical, so only the collision density moves.
_UNIQUE_FRACTIONS = [1.0, 0.1]

_rows_cache: dict[tuple[str, str, float], tuple[odt.Array2dInt32, np.ndarray]] = {}


def _duplicate_rows(
    bench_case: BenchCase, unique_fraction: float
) -> tuple[odt.Array2dInt32, np.ndarray]:
    """Build an ``(n, 3)`` int32 row block whose distinct-row count is ``unique_fraction * n``."""
    key = (bench_case.mesh_name, str(bench_case.device), unique_fraction)
    if key not in _rows_cache:
        faces = bench_case.faces_np
        n_unique = max(1, int(faces.shape[0] * unique_fraction))
        rows_np = np.ascontiguousarray(faces[np.arange(faces.shape[0]) % n_unique], dtype=np.int32)
        rows_wp = wp.array(rows_np.reshape(-1), dtype=wp.int32, device=bench_case.device).reshape(
            rows_np.shape
        )
        _rows_cache[key] = (odt.as_array2d(rows_wp, wp.int32), rows_np)
    return _rows_cache[key]


_inverse_cache: dict[tuple[str, str], wp.array[wp.int32]] = {}


def _edge_inverse(bench_case: BenchCase) -> wp.array[wp.int32]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _inverse_cache:
        _inverse_cache[key] = od.edges.edges_unique_inverse(
            bench_case.faces_wp, n_vertices=bench_case.n_vertices
        )
    return _inverse_cache[key]


_inverse_np_cache: dict[str, np.ndarray] = {}


def _edge_inverse_np(bench_case: BenchCase) -> np.ndarray:
    """
    Return the same key array on the host, for the reference branches, built without ordito.

    A reference case carries no device, so it cannot be handed ``_edge_inverse``'s buffer -- and it
    should not be, because a benchmark input must not come from the code under test (see
    ``meshes.py``). ``np.unique`` over the row-sorted edges is the host derivation of the same
    thing. The two inverses label their classes in different orders, which is irrelevant here: the
    grouping workload is set by the multiset of key multiplicities, identical either way.
    """
    if bench_case.mesh_name not in _inverse_np_cache:
        faces_np = bench_case.faces_np
        edges_np = np.sort(faces_np[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).astype(np.int64), axis=1)
        _inverse_np_cache[bench_case.mesh_name] = (
            np.unique(edges_np, axis=0, return_inverse=True)[1].astype(np.int32).ravel()
        )
    return _inverse_np_cache[bench_case.mesh_name]


_edge_rows_cache: dict[tuple[str, str], odt.Array2dInt32] = {}


def _edge_rows(bench_case: BenchCase) -> odt.Array2dInt32:
    """Return the ``(3F, 2)`` face-edge rows, each sorted, uploaded to the case's device."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _edge_rows_cache:
        rows_np = np.ascontiguousarray(_sorted_edges_np(bench_case), dtype=np.int32)
        rows_wp = wp.array(rows_np.reshape(-1), dtype=wp.int32, device=bench_case.device)
        _edge_rows_cache[key] = odt.as_array2d(rows_wp.reshape(rows_np.shape), wp.int32)
    return _edge_rows_cache[key]


def _sorted_edges_np(bench_case: BenchCase) -> np.ndarray:
    """Return the ``(3F, 2)`` face-edge rows with each row sorted, built on the host."""
    faces_np = bench_case.faces_np
    return np.sort(faces_np[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)


@pytest.mark.benchmark(group="group")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_group(bench_case: BenchCase) -> None:
    """
    Fixed-multiplicity index grouping over the edge inverse: a radix sort plus a segment pass.

    ``trimesh.grouping.group(values, min_len=2, max_len=2)`` is the same operation and the same
    answer -- ``tests/test_grouping.py::test_group_matches_trimesh`` shows the two group *sets* are
    equal -- reached by an ``argsort`` plus a NumPy segment walk instead. Both sides are handed the
    edge inverse -- built outside the timed callable on both branches, and on the host for the
    reference so that its input does not come from the code under test -- so this row is the
    grouping alone. Capped at ``bunny``, past which the reference's segment walk is unaffordable.
    """
    if bench_case.kind == "ordito":
        inverse = _edge_inverse(bench_case)
        pairs = bench_case.run(lambda: od.grouping.group(inverse, 2))
        assert pairs.shape[1] == 2
        assert pairs.shape[0] > 0
    else:
        skip_larger_than(bench_case, "bunny", "the reference sorts and walks segments on one core")
        inverse_np = _edge_inverse_np(bench_case)
        pairs_tm = bench_case.run(
            lambda: np.asarray(tm.grouping.group(inverse_np, min_len=2, max_len=2))
        )
        assert pairs_tm.shape[1] == 2
        assert pairs_tm.shape[0] > 0


@pytest.mark.benchmark(group="group_int_rows")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_group_int_rows(bench_case: BenchCase) -> None:
    """
    Row grouping at multiplicity two over the ``3F`` sorted face-edge rows: hash, then ``group``.

    The same ``face_adjacency`` workload as ``group`` above, reached from the rows instead of a
    precomputed inverse, so the two read side by side as the price of the row hash.
    ``trimesh.grouping.group_rows(require_count=2)`` is the same operation
    (``tests/test_grouping.py::test_group_int_rows``); both sides get the host-built rows, and the
    reference is capped at ``bunny`` like ``group``.
    """
    if bench_case.kind == "ordito":
        rows_wp = _edge_rows(bench_case)
        n_vertices = bench_case.n_vertices
        pairs = bench_case.run(lambda: od.grouping.group_int_rows(rows_wp, 2, max_value=n_vertices))
        assert pairs.shape[1] == 2
        assert pairs.shape[0] > 0
    else:
        skip_larger_than(bench_case, "bunny", "the reference sorts and walks segments on one core")
        rows_np = _sorted_edges_np(bench_case)
        pairs_tm = bench_case.run(
            lambda: np.asarray(tm.grouping.group_rows(rows_np, require_count=2))
        )
        assert pairs_tm.shape[1] == 2
        assert pairs_tm.shape[0] > 0


@pytest.mark.benchmark(group="unique_1d")
@pytest.mark.benchlibs("ordito")
def test_unique_1d(bench_case: BenchCase) -> None:
    """
    Sorted unique values of the edge inverse, with inverse and counts: the hash-table dedup.

    Every interior edge's class id appears twice, so the input is ``3F`` keys and the output about
    ``1.5F``: the hash inserts scale with the input and the final sort with the output. Ordito only:
    the test oracle is ``numpy.unique``, not a mesh library.
    """
    inverse = _edge_inverse(bench_case)
    unique, inverse_out, counts = bench_case.run(
        lambda: od.grouping.unique_1d(inverse, return_inverse=True, return_counts=True)
    )
    assert 0 < unique.shape[0] < inverse.shape[0]
    assert inverse_out.shape == inverse.shape
    assert counts.shape == unique.shape


_edge_keys_cache: dict[tuple[str, str], wp.array[wp.uint64]] = {}


def _edge_keys(bench_case: BenchCase) -> wp.array[wp.uint64]:
    """Return the ``(3F,)`` packed undirected edge keys, built on the host and uploaded."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _edge_keys_cache:
        edges_np = _sorted_edges_np(bench_case).astype(np.uint64)
        keys_np = edges_np[:, 0] * np.uint64(bench_case.n_vertices) + edges_np[:, 1]
        _edge_keys_cache[key] = wp.array(keys_np, dtype=wp.uint64, device=bench_case.device)
    return _edge_keys_cache[key]


@pytest.mark.benchmark(group="hashed_occurrence_counts")
@pytest.mark.benchlibs("ordito")
def test_hashed_occurrence_counts(bench_case: BenchCase) -> None:
    """
    Per-slot occurrence counts of the ``3F`` packed edge keys: ``unique_1d``'s table alone.

    The counting half of ``unique_1d`` with no compaction, sort or host read, so the two rows read
    side by side as what those stages cost. Ordito only: the test oracle is ``numpy.unique``.
    """
    keys = _edge_keys(bench_case)
    counts = bench_case.run(lambda: od.grouping.hashed_occurrence_counts(keys))
    assert counts.shape[0] > keys.shape[0]


@pytest.mark.benchmark(group="unique_faces")
@pytest.mark.benchlibs("ordito", "igl", "trimesh")
@pytest.mark.parametrize("duplicate_fraction", [0.0, 0.5], ids=["allunique", "half"])
def test_unique_faces(bench_case: BenchCase, duplicate_fraction: float) -> None:
    """
    Orientation-agnostic face deduplication: sort each row, then dedup.

    The axis is the duplicate density, as in ``unique_rows`` above: at ``half`` the input carries a
    *flipped* copy of half its faces, which is the case that separates an orientation-agnostic dedup
    from a plain row dedup -- the flipped copies must collapse.

    ``igl.unique_simplices`` is the same operation and returns ``(FF, IA, IC)`` where ``IC`` is
    ordito's inverse. One difference to know before comparing: **igl returns the sorted rows**
    (``FF == sort(F(IA, :), 2)``) where ordito returns the first occurrence with its original
    winding intact, so the parity comparison sorts ordito's rows first.

    ``Trimesh.unique_faces`` is the third implementation and the one that makes this group's axis
    legible: it is a *mask* rather than a rebuild, so it does strictly less than the other two rows
    and it carries the property that separates this group from ``unique_rows`` -- it is
    orientation-agnostic, measured 320 kept from 340 where 20 are flipped copies. It is a cached
    property on a fresh
    ``Trimesh``, so the mesh is rebuilt inside the timed callable; that build is why the row is
    capped where igl's is.

    All three asserts bound the survivor count rather than pinning it to the input's, because two
    registry meshes carry duplicate faces of their own -- ``bunny_decimated`` has **87** (16 214
    unique of 16 301) and ``lucy`` likewise -- so a ``== n_faces`` assert was failing on them for
    every library, which is what a bound rather than an equality is for on a *shape* check.
    """
    faces_np = bench_case.faces_np
    if duplicate_fraction:
        n_duplicated = int(duplicate_fraction * faces_np.shape[0])
        faces_np = np.concatenate([faces_np, faces_np[:n_duplicated, ::-1]])
    if bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "bunny", "the reference sorts and dedups on one core")
        vertices_np = bench_case.vertices_np

        def unique_faces_tm() -> int:
            mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
            return int(np.count_nonzero(mesh_tm.unique_faces()))

        assert 0 < bench_case.run(unique_faces_tm) <= bench_case.n_faces
        return
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "the reference sorts and dedups on one core")
        faces_igl = np.ascontiguousarray(faces_np, dtype=np.int64)
        unique_igl = bench_case.run(lambda: igl.unique_simplices(faces_igl))
        assert 0 < unique_igl[0].shape[0] <= bench_case.n_faces
        return
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=bench_case.device,
    )
    unique_wp = bench_case.run(lambda: od.grouping.unique_faces(faces_wp))
    assert 0 < int(unique_wp.shape[0]) // 3 <= bench_case.n_faces


@pytest.mark.benchmark(group="unique_rows")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("unique_fraction", _UNIQUE_FRACTIONS, ids=["allunique", "tenth"])
def test_unique_rows(bench_case: BenchCase, unique_fraction: float) -> None:
    """
    Row deduplication, at two duplicate densities.

    Both radix passes cost the same either way -- the sort does not care how many rows collide --
    so any gap between these two rows is the *output* stage: fewer unique rows means a smaller
    compaction, and a larger gap than that would mean the segment scan is doing more work than it
    needs to. Total input length is held fixed so only the duplicate density varies.
    """
    rows_wp, rows_np = _duplicate_rows(bench_case, unique_fraction)
    if bench_case.kind == "ordito":
        unique = bench_case.run(lambda: od.grouping.unique_rows(rows_wp))
        assert unique[0].size > 0
    else:
        unique_tm = bench_case.run(lambda: tm.grouping.unique_rows(rows_np))
        assert len(unique_tm[0]) > 0


@pytest.mark.benchmark(group="first_occurrence_indices")
@pytest.mark.benchlibs("ordito")
def test_first_occurrence_indices(bench_case: BenchCase) -> None:
    """
    First index of each class of the edge inverse, with the class count passed (no readback).

    One scatter-min over ``3F`` entries into ``1.5F`` slots, the representative-picking half of
    every dedup. Ordito only: the test oracle is ``numpy.unique(return_index=True)``.
    """
    inverse = _edge_inverse(bench_case)
    n_unique = int(inverse.numpy().max()) + 1  # setup: the class count a caller already holds
    first = bench_case.run(lambda: od.grouping.first_occurrence_indices(inverse, n_unique))
    assert first.shape == (n_unique,)


@pytest.mark.benchmark(group="hash_rows")
@pytest.mark.benchlibs("ordito")
def test_hash_rows(bench_case: BenchCase) -> None:
    """
    Packed ``uint64`` key per face row through the dispatching ``hash_rows``, radix inferred.

    The inferred radix costs a min/max reduction and its readback before the packing pass, which is
    what separates this row from ``hash_indices_rows`` with ``max_index`` passed. Ordito only: no
    reference packs rows into keys.
    """
    faces_2d = odt.as_array2d(bench_case.faces_wp.reshape((bench_case.n_faces, 3)), wp.int32)
    keys = bench_case.run(lambda: od.grouping.hash_rows(faces_2d))
    assert keys.shape == (bench_case.n_faces,)


@pytest.mark.benchmark(group="hash_vector_rows")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("epsilon", [0.0, 1e-4], ids=["relative", "absolute"])
def test_hash_vector_rows(bench_case: BenchCase, epsilon: float) -> None:
    """
    Packed ``uint64`` key per vertex position, under both bucketing schemes.

    ``absolute`` snaps to a grid from the data's minimum corner, which needs a bounds reduction
    first; ``relative`` packs the float bits directly, one pass. Ordito only: no reference packs
    positions into keys.
    """
    vertices = bench_case.vertices_wp
    keys = bench_case.run(lambda: od.grouping.hash_vector_rows(vertices, epsilon))
    assert keys.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="hash_indices_rows")
@pytest.mark.benchlibs("ordito")
def test_hash_indices_rows(bench_case: BenchCase) -> None:
    """
    Packed ``uint64`` key per face row with the radix passed as ``n_vertices``.

    The validated path: the range check's min/max reduction still runs, so this row against
    ``hash_rows`` (radix inferred from the same reduction) isolates the dispatch. Ordito only.
    """
    faces_2d = odt.as_array2d(bench_case.faces_wp.reshape((bench_case.n_faces, 3)), wp.int32)
    n_vertices = bench_case.n_vertices
    keys = bench_case.run(lambda: od.grouping.hash_indices_rows(faces_2d, n_vertices))
    assert keys.shape == (bench_case.n_faces,)


@pytest.mark.benchmark(group="sorted_undirected_edge_keys")
@pytest.mark.benchlibs("ordito")
def test_sorted_undirected_edge_keys(bench_case: BenchCase) -> None:
    """
    Sorted packed keys of the ``3F`` face-edge rows: one packing pass and one radix sort.

    Ordito only: the test oracle is a NumPy sort of the packed keys.
    """
    rows_wp = _edge_rows(bench_case)
    n_vertices = bench_case.n_vertices
    keys = bench_case.run(lambda: od.grouping.sorted_undirected_edge_keys(rows_wp, n_vertices))
    assert keys.shape == (rows_wp.shape[0],)
