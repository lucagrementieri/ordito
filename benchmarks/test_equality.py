"""
Benchmarks for ``ordito.equality``: equality of meshes, point sets, rows and partitions.

Every row compares a scan mesh with a renumbered, reordered and rotated copy of itself, built on
the host outside the timed callable, so each call does the whole comparison and answers ``True``.
``canonical_faces`` is the radix sort of the face rows underneath both comparisons;
``faces_equal`` is two of those and an element-wise pass; ``meshes_allclose`` adds a nearest-vertex
query and the bijection check before them.

Ordito only: no reference library compares meshes up to numbering. The tests' NumPy oracle (a
per-row rotation and a ``lexsort``) is not a mesh library.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import ordito as od
import ordito.typing as odt
from conftest import BenchCase

_shuffled_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], wp.array[wp.int32]]] = {}


def _shuffled(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Return the case's mesh renumbered, its faces reordered and rotated, built on the host."""
    assert bench_case.device is not None
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _shuffled_cache:
        rng = np.random.default_rng(0)
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        order_np = rng.permutation(len(vertices_np))
        faces_np = np.argsort(order_np)[faces_np][rng.permutation(len(faces_np))]
        roll_np = rng.integers(0, 3, len(faces_np))
        faces_np = np.take_along_axis(faces_np, (roll_np[:, None] + np.arange(3)) % 3, axis=1)
        _shuffled_cache[key] = (
            wp.array(vertices_np[order_np], dtype=wp.vec3, device=bench_case.device),
            wp.array(faces_np.ravel().astype(np.int32), dtype=wp.int32, device=bench_case.device),
        )
    return _shuffled_cache[key]


@pytest.mark.benchmark(group="meshes_allclose")
@pytest.mark.benchlibs("ordito")
def test_meshes_allclose(bench_case: BenchCase) -> None:
    """
    The whole comparison: a nearest-vertex query, the bijection check and two face sorts.

    The scan meshes have no coincident vertices, which the nearest-neighbour match needs.
    """
    vertices_wp, faces_wp = _shuffled(bench_case)
    vertices_ref, faces_ref = bench_case.vertices_wp, bench_case.faces_wp
    equal = bench_case.run(
        lambda: od.equality.meshes_allclose(vertices_ref, faces_ref, vertices_wp, faces_wp)
    )
    assert equal


@pytest.mark.benchmark(group="faces_equal")
@pytest.mark.benchlibs("ordito")
def test_faces_equal(bench_case: BenchCase) -> None:
    """Two face sorts and an element-wise pass over one vertex numbering, the bound passed in."""
    assert bench_case.device is not None
    rng = np.random.default_rng(1)
    faces_np = bench_case.faces_np[rng.permutation(bench_case.n_faces)]
    roll_np = rng.integers(0, 3, len(faces_np))
    faces_np = np.take_along_axis(faces_np, (roll_np[:, None] + np.arange(3)) % 3, axis=1)
    reordered = wp.array(
        faces_np.ravel().astype(np.int32), dtype=wp.int32, device=bench_case.device
    )
    faces_ref = bench_case.faces_wp
    n_vertices = bench_case.n_vertices
    equal = bench_case.run(
        lambda: od.equality.faces_equal(faces_ref, reordered, n_vertices=n_vertices)
    )
    assert equal


@pytest.mark.benchmark(group="canonical_faces")
@pytest.mark.benchlibs("ordito")
def test_canonical_faces(bench_case: BenchCase) -> None:
    """One radix sort of the rotated face rows, with the bound passed in."""
    _vertices_wp, faces_wp = _shuffled(bench_case)
    n_vertices = bench_case.n_vertices
    canonical = bench_case.run(lambda: od.equality.canonical_faces(faces_wp, n_vertices=n_vertices))
    assert canonical.size == faces_wp.size


@pytest.mark.benchmark(group="points_allclose")
@pytest.mark.benchlibs("ordito")
def test_points_allclose(bench_case: BenchCase) -> None:
    """The vertex half of ``meshes_allclose``: the nearest-point query and the bijection check."""
    vertices_wp, _faces_wp = _shuffled(bench_case)
    vertices_ref = bench_case.vertices_wp
    equal = bench_case.run(lambda: od.equality.points_allclose(vertices_ref, vertices_wp))
    assert equal


_edge_rows_cache: dict[tuple[str, str], tuple[odt.Array2dInt32, odt.Array2dInt32]] = {}


def _edge_rows(bench_case: BenchCase) -> tuple[odt.Array2dInt32, odt.Array2dInt32]:
    """Return the ``(3 * n_faces, 2)`` directed face edges and a row-shuffled copy, host-built."""
    assert bench_case.device is not None
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _edge_rows_cache:
        edges_np = bench_case.faces_np[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).astype(np.int32)
        shuffled_np = edges_np[np.random.default_rng(2).permutation(len(edges_np))]
        edges, shuffled = (
            odt.as_array2d(
                wp.array(np.ascontiguousarray(rows_np), dtype=wp.int32, device=bench_case.device),
                wp.int32,
            )
            for rows_np in (edges_np, shuffled_np)
        )
        _edge_rows_cache[key] = (edges, shuffled)
    return _edge_rows_cache[key]


@pytest.mark.benchmark(group="rows_equal")
@pytest.mark.benchlibs("ordito")
def test_rows_equal(bench_case: BenchCase) -> None:
    """The directed face edges against a row-shuffled copy: two radix sorts a side and a compare."""
    edges, shuffled = _edge_rows(bench_case)
    equal = bench_case.run(lambda: od.equality.rows_equal(edges, shuffled))
    assert equal


@pytest.mark.benchmark(group="partitions_equal")
@pytest.mark.benchlibs("ordito")
def test_partitions_equal(bench_case: BenchCase) -> None:
    """
    Each face's smallest corner as its label, against the same labels renamed by a permutation.

    Three hash-table ``unique_1d`` calls, each with its readback.
    """
    assert bench_case.device is not None
    rng = np.random.default_rng(3)
    labels_np = bench_case.faces_np.min(axis=1).astype(np.int32)
    renamed_np = rng.permutation(bench_case.n_vertices).astype(np.int32)[labels_np]
    labels = wp.array(labels_np, dtype=wp.int32, device=bench_case.device)
    renamed = wp.array(renamed_np, dtype=wp.int32, device=bench_case.device)
    equal = bench_case.run(lambda: od.equality.partitions_equal(labels, renamed))
    assert equal
