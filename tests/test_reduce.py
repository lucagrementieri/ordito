from __future__ import annotations

from typing import Literal

import numpy as np
import pymeshlab as ml
import pytest
import pytorch3d.ops.utils as p3d_ops_utils
import torch
import trimesh as tm
import warp as wp

import ordito.reduce as od_reduce
import ordito.typing as odt
from tests.conversions import points_to_torch, points_to_warp, trimesh_to_pyvista, warp_empty


def _random_values(shape: tuple[int, ...] | int, seed: int = 42) -> np.ndarray:
    """Fixed-seed float32 standard-normal test data."""
    return np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)


@pytest.mark.parity("min_scalar", "numpy")
def test_min_1d(device: str) -> None:
    """Class A: direct comparison against ``numpy.min``."""
    rng = np.random.default_rng(42)
    n = 100
    values_np = rng.integers(-1000, 1000, (n,), dtype=np.int32)
    min_np = values_np.min()

    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    min_wp = od_reduce.min(values_wp)
    assert np.allclose(min_wp, min_np)


def test_min_2d(device: str) -> None:
    n = 200
    m = 100
    values_np = _random_values((n, m))
    min_np = values_np.min()
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    min_wp = od_reduce.min(values_wp)
    assert np.allclose(min_wp, min_np)


@pytest.mark.parametrize("axis", [0, 1])
def test_min_2d_axis(device: str, axis: Literal[0, 1]) -> None:
    rng = np.random.default_rng(42)
    values_np = rng.integers(-1000, 1000, (32, 10), dtype=np.int32)
    values_wp = odt.as_array2d(wp.array(values_np, dtype=wp.int32, device=device), wp.int32)
    got_wp = od_reduce.min(values_wp, axis=axis)
    exp_np = values_np.min(axis=axis)
    assert np.array_equal(got_wp.numpy(), exp_np)


def test_max_1d(device: str) -> None:
    rng = np.random.default_rng(42)
    n = 100
    values_np = rng.integers(-1000, 1000, (n,), dtype=np.int32)
    max_np = values_np.max()

    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    max_wp = od_reduce.max(values_wp)
    assert np.allclose(max_wp, max_np)


def test_max_2d(device: str) -> None:
    n = 200
    m = 100
    values_np = _random_values((n, m))
    max_np = values_np.max()
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    max_wp = od_reduce.max(values_wp)
    assert np.allclose(max_wp, max_np)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parity("max_axis1", "numpy")
def test_max_2d_axis(device: str, axis: Literal[0, 1]) -> None:
    """Class A: direct comparison against ``numpy.max(axis=...)`` over both axes."""
    rng = np.random.default_rng(42)
    values_np = rng.integers(-1000, 1000, (32, 10), dtype=np.int32)
    values_wp = odt.as_array2d(wp.array(values_np, dtype=wp.int32, device=device), wp.int32)
    got_wp = od_reduce.max(values_wp, axis=axis)
    exp_np = values_np.max(axis=axis)
    assert np.array_equal(got_wp.numpy(), exp_np)


@pytest.mark.parity("minmax_scalar", "numpy")
def test_minmax_1d(device: str) -> None:
    """Class A: both extrema against ``numpy.min`` / ``numpy.max``."""
    rng = np.random.default_rng(42)
    n = 100
    values_np = rng.integers(-1000, 1000, (n,), dtype=np.int32)
    min_np = values_np.min()
    max_np = values_np.max()

    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    min_wp, max_wp = od_reduce.minmax(values_wp)
    assert np.allclose(min_wp, min_np)
    assert np.allclose(max_wp, max_np)


@pytest.mark.parametrize("shape", [(200, 100), (5000, 2), (5000, 3), (3, 5000)])
@pytest.mark.parity("minmax_global_2d", "numpy")
def test_minmax_2d(device: str, shape: tuple[int, int]) -> None:
    """
    Class A: rank-2 ``axis=None`` extrema against ``numpy.min`` / ``numpy.max``.

    Parametrized over narrow *and* wide trailing extents on purpose: ``(m, 2)`` is the edge-table
    shape [`ordito.graph.connected_components`][] validates, and it clips the ``TILE_2D`` square
    so the tile branch never runs — a wide-only fixture would leave that path untested.
    """
    values_np = _random_values(shape)
    min_np = values_np.min()
    max_np = values_np.max()
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    min_wp, max_wp = od_reduce.minmax(values_wp)
    assert np.allclose(min_wp, min_np)
    assert np.allclose(max_wp, max_np)


def test_minmax_vec3(device: str) -> None:
    """Component-wise corner pair of a ``wp.vec3`` array (the ``aabb`` reduction)."""
    rng = np.random.default_rng(42)
    points_np = rng.standard_normal((500, 3)).astype(np.float32)
    points_wp = points_to_warp(points_np, device)

    lower_wp, upper_wp = od_reduce.minmax(points_wp)

    assert np.allclose(np.array(list(lower_wp)), points_np.min(axis=0), rtol=1e-6, atol=1e-6)
    assert np.allclose(np.array(list(upper_wp)), points_np.max(axis=0), rtol=1e-6, atol=1e-6)


def test_minmax_vec3_rejects_axis_and_empty(device: str) -> None:
    points_wp = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="axis=None"):
        od_reduce.minmax(points_wp, axis=0)  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(ValueError, match="non-empty"):
        od_reduce.minmax(warp_empty(0, wp.vec3, device))


@pytest.mark.parametrize("axis", [0, 1])
def test_minmax_2d_axis(device: str, axis: Literal[0, 1]) -> None:
    values_np = _random_values((32, 10))
    values_wp = odt.as_array2d(wp.array(values_np, dtype=wp.float32, device=device), wp.float32)
    got_min_wp, got_max_wp = od_reduce.minmax(values_wp, axis=axis)
    exp_min_np = values_np.min(axis=axis)
    exp_max_np = values_np.max(axis=axis)
    assert np.allclose(got_min_wp.numpy(), exp_min_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(got_max_wp.numpy(), exp_max_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("any_bool", "numpy")
def test_any_1d(device: str) -> None:
    """Class A, against ``numpy.any``, on a mask that is neither all-``True`` nor all-``False``."""
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(100,), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    any_wp = od_reduce.any(mask_wp)
    any_ref_np = np.any(mask_np)
    assert any_wp == any_ref_np


@pytest.mark.parity("all_bool", "numpy")
def test_all_1d(device: str) -> None:
    """Class A, against ``numpy.all``, on a mask that is neither all-``True`` nor all-``False``."""
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(100,), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    all_wp = od_reduce.all(mask_wp)
    all_ref_np = np.all(mask_np)
    assert all_wp == all_ref_np


@pytest.mark.parametrize("axis", [0, 1])
def test_any_2d_axis(device: str, axis: Literal[0, 1]) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(32, 4), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    any_wp = od_reduce.any(mask_wp, axis=axis)
    any_ref_np = np.any(mask_np, axis=axis)
    assert np.array_equal(any_wp.numpy(), any_ref_np)


def test_any_2d_global(device: str) -> None:
    rng = np.random.default_rng(42)
    for mask_np in [
        rng.choice([False, True], size=(32, 4), replace=True),
        np.zeros((32, 4), dtype=bool),
        np.ones((32, 4), dtype=bool),
    ]:
        mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
        any_wp = od_reduce.any(mask_wp, axis=None)
        any_np = bool(np.any(mask_np))
        assert any_wp == any_np, f"any global mismatch: ordito {any_wp}, numpy {any_np}"


@pytest.mark.parametrize("axis", [0, 1])
def test_all_2d_axis(device: str, axis: Literal[0, 1]) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(32, 4), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    all_wp = od_reduce.all(mask_wp, axis=axis)
    all_ref_np = np.all(mask_np, axis=axis)
    assert np.array_equal(all_wp.numpy(), all_ref_np)


def test_all_2d_global(device: str) -> None:
    rng = np.random.default_rng(42)
    for mask_np in [
        rng.choice([False, True], size=(32, 4), replace=True),
        np.zeros((32, 4), dtype=bool),
        np.ones((32, 4), dtype=bool),
    ]:
        mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
        all_wp = od_reduce.all(mask_wp, axis=None)
        all_np = bool(np.all(mask_np))
        assert all_wp == all_np, f"all global mismatch: ordito {all_wp}, numpy {all_np}"


def test_scalar_reduce_1d_axis_raises(device: str) -> None:
    values_wp = wp.array([1, 2, 3], dtype=wp.int32, device=device)
    for fn in (od_reduce.min, od_reduce.max, od_reduce.minmax, od_reduce.sum):
        with pytest.raises(ValueError, match="requires axis=None for a 1D array"):
            fn(values_wp, axis=0)  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_sum_bool_1d_axis_raises(device: str) -> None:
    mask_wp = wp.array([True, False, True], dtype=wp.bool, device=device)
    with pytest.raises(ValueError, match="requires axis=None for a 1D array"):
        od_reduce.sum(mask_wp, axis=0)


def test_scalar_reduce_2d_invalid_axis_raises(device: str) -> None:
    """An axis outside {0, 1, None} must raise rather than silently reading as axis=0."""
    values_wp = wp.array([[1, 2], [3, 4]], dtype=wp.int32, device=device)
    for fn in (od_reduce.min, od_reduce.max, od_reduce.minmax, od_reduce.sum):
        with pytest.raises(ValueError, match="requires axis to be 0, 1, or None"):
            fn(values_wp, axis=2)  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_bool_reduce_2d_invalid_axis_raises(device: str) -> None:
    """Same invalid-axis contract as the scalar family, for ``any``/``all``."""
    mask_wp = wp.array([[True, False], [False, True]], dtype=wp.bool, device=device)
    for fn in (od_reduce.any, od_reduce.all):
        with pytest.raises(ValueError, match="requires axis to be 0, 1, or None"):
            fn(mask_wp, axis=2)  # pyright: ignore[reportCallIssue, reportArgumentType]


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_min_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    values_np = rng.integers(-1000, 1000, shape, dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    min_wp = od_reduce.min(values_wp)
    assert np.allclose(min_wp, values_np.min())


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_max_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    values_np = rng.standard_normal(shape).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    max_wp = od_reduce.max(values_wp)
    assert np.allclose(max_wp, values_np.max(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_minmax_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    values_np = rng.standard_normal(shape).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    got_min, got_max = od_reduce.minmax(values_wp)
    assert np.allclose(got_min, values_np.min(), rtol=1e-5, atol=1e-5)
    assert np.allclose(got_max, values_np.max(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(9, 9), (65, 10), (16, 16), (3, 5), (200, 100)])
def test_scalar_reduce_global_noncontiguous(device: str, shape: tuple[int, int]) -> None:
    """
    The last remaining exerciser of ``kernels/reduce.py``'s rank-2 ``axis=None`` kernels.

    A non-contiguous rank-2 array can't flatten onto the 1-D kernel, so this is the only path left
    to reach them: every contiguous array now flattens there instead
    (``reduce._flattened_for_global``). Shapes span the full-tile case, both single- and
    double-boundary-short cases, and a wide table -- the same coverage
    ``test_min_2d``/``test_minmax_2d``/``test_min_partial_tiles`` used to give the rank-2 kernel
    before it stopped being reachable for a contiguous array.
    """
    rng = np.random.default_rng(99)
    wide_np = rng.integers(-1000, 1000, (shape[0], shape[1] * 2), dtype=np.int32)
    values_np = wide_np[:, ::2]
    values_wp = odt.as_dense(wp.array(wide_np, dtype=wp.int32, device=device)[:, ::2])
    assert not values_wp.is_contiguous
    assert np.array_equal(od_reduce.min(values_wp), values_np.min())
    assert np.array_equal(od_reduce.max(values_wp), values_np.max())
    assert np.array_equal(od_reduce.sum(values_wp), values_np.sum())
    got_min, got_max = od_reduce.minmax(values_wp)
    assert np.array_equal(got_min, values_np.min())
    assert np.array_equal(got_max, values_np.max())


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_any_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    mask_np = rng.choice([False, True], size=shape, replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert od_reduce.any(mask_wp) == bool(np.any(mask_np))


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_all_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    mask_np = rng.choice([False, True], size=shape, replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert od_reduce.all(mask_wp) == bool(np.all(mask_np))


@pytest.mark.parametrize(
    ("shape", "axis"), [((9, 9), 0), ((9, 9), 1), ((65, 10), 0), ((65, 10), 1)]
)
def test_scalar_reduce_partial_tiles_axis(
    device: str, shape: tuple[int, int], axis: Literal[0, 1]
) -> None:
    rng = np.random.default_rng(99)
    values_np = rng.integers(-1000, 1000, shape, dtype=np.int32)
    values_wp = odt.as_array2d(wp.array(values_np, dtype=wp.int32, device=device), wp.int32)
    assert np.array_equal(od_reduce.min(values_wp, axis=axis).numpy(), values_np.min(axis=axis))
    assert np.array_equal(od_reduce.max(values_wp, axis=axis).numpy(), values_np.max(axis=axis))
    got_min, got_max = od_reduce.minmax(values_wp, axis=axis)
    assert np.array_equal(got_min.numpy(), values_np.min(axis=axis))
    assert np.array_equal(got_max.numpy(), values_np.max(axis=axis))
    assert np.array_equal(od_reduce.sum(values_wp, axis=axis).numpy(), values_np.sum(axis=axis))


@pytest.mark.parametrize(
    ("shape", "axis"), [((9, 9), 0), ((9, 9), 1), ((65, 10), 0), ((65, 10), 1)]
)
def test_bool_reduce_partial_tiles_axis(
    device: str, shape: tuple[int, int], axis: Literal[0, 1]
) -> None:
    rng = np.random.default_rng(99)
    mask_np = rng.choice([False, True], size=shape, replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert np.array_equal(od_reduce.any(mask_wp, axis=axis).numpy(), np.any(mask_np, axis=axis))
    assert np.array_equal(od_reduce.all(mask_wp, axis=axis).numpy(), np.all(mask_np, axis=axis))


@pytest.mark.parity("sum_scalar", "numpy")
def test_sum_1d(device: str) -> None:
    """Class A: direct comparison against ``numpy.sum``."""
    rng = np.random.default_rng(42)
    n = 100
    values_np = rng.integers(-1000, 1000, (n,), dtype=np.int32)
    sum_np = values_np.sum()

    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    sum_wp = od_reduce.sum(values_wp)
    assert np.allclose(sum_wp, sum_np)


@pytest.mark.parametrize("n", [1_000, 50_000, 3_000_000])
def test_float_sums_are_bit_identical_across_calls(device: str, n: int) -> None:
    """
    Ordito against ordito: repeated float sums agree bit for bit, and match NumPy in float64.

    The NumPy comparison is the oracle for the value; the repeat is the claim. One ``atomic_add``
    per block adds the blocks in arrival order on CUDA, which moved the last bit between calls; the
    sizes span one block, one fold stage and two (``ITEMS_PER_BLOCK_1D = 1024``).
    """
    rng = np.random.default_rng(7)
    values_np = rng.random(n, dtype=np.float32)
    weights_np = rng.random(n, dtype=np.float32)
    values_wp: wp.array[wp.float32] = wp.array(values_np, dtype=wp.float32, device=device)
    weights_wp: wp.array[wp.float32] = wp.array(weights_np, dtype=wp.float32, device=device)
    points_wp = wp.array(rng.random((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)

    sums = {od_reduce.sum(values_wp) for _ in range(6)}
    means = {od_reduce.mean(values_wp) for _ in range(6)}
    weighted = {od_reduce.weighted_sum(values_wp, weights_wp) for _ in range(6)}
    vectors = {tuple(od_reduce.sum(points_wp)) for _ in range(6)}
    assert len(sums) == len(means) == len(weighted) == len(vectors) == 1
    assert np.isclose(sums.pop(), values_np.astype(np.float64).sum(), rtol=1e-5)
    assert np.isclose(weighted.pop(), (values_np.astype(np.float64) * weights_np).sum(), rtol=1e-5)


def test_sum_2d(device: str) -> None:
    n = 200
    m = 100
    values_np = _random_values((n, m))
    sum_np = values_np.sum()
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    sum_wp = od_reduce.sum(values_wp)
    assert np.allclose(sum_wp, sum_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parity("sum_axis0", "numpy")
def test_sum_2d_axis(device: str, axis: Literal[0, 1]) -> None:
    """Class A: direct comparison against ``numpy.sum(axis=...)`` over both axes."""
    rng = np.random.default_rng(42)
    values_np = rng.integers(-1000, 1000, (32, 10), dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    got_wp = od_reduce.sum(values_wp, axis=axis)
    exp_np = values_np.sum(axis=axis)
    assert np.array_equal(got_wp.numpy(), exp_np)


@pytest.mark.parity("sum_bool", "numpy")
def test_sum_bool_1d(device: str) -> None:
    """
    Class A, against ``numpy.ndarray.sum`` over the same mask.

    Counts a ``wp.bool`` mask without widening it to ``int32`` first, so this is the value gate on
    ``kernels.reduce._reduce_bool_1d_tiled``. The mask is half set, so neither an all-zero nor an
    all-one shortcut would pass it.
    """
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(100,), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    sum_wp = od_reduce.sum(mask_wp)
    sum_np = int(mask_np.sum())
    assert sum_wp == sum_np


@pytest.mark.parametrize("axis", [0, 1])
def test_sum_bool_2d_axis(device: str, axis: Literal[0, 1]) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(32, 4), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    got_wp = od_reduce.sum(mask_wp, axis=axis)
    exp_np = mask_np.sum(axis=axis).astype(np.int32)
    assert np.array_equal(got_wp.numpy(), exp_np)


def test_sum_bool_2d_global(device: str) -> None:
    rng = np.random.default_rng(42)
    for mask_np in [
        rng.choice([False, True], size=(32, 4), replace=True),
        np.zeros((32, 4), dtype=bool),
        np.ones((32, 4), dtype=bool),
    ]:
        mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
        sum_wp = od_reduce.sum(mask_wp, axis=None)
        sum_np = int(mask_np.sum())
        assert sum_wp == sum_np, f"sum global mismatch: ordito {sum_wp}, numpy {sum_np}"


@pytest.mark.parity("weighted_sum", "numpy")
def test_weighted_sum_1d(device: str) -> None:
    """Class A: ``sum(values * weights)`` against the NumPy expression the benchmark times."""
    rng = np.random.default_rng(42)
    n = 100
    values_np = rng.standard_normal(n, dtype=np.float32)
    weights_np = rng.random(n, dtype=np.float32)
    exp_np = float(np.sum(values_np * weights_np))

    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    weights_wp = wp.array(weights_np, dtype=wp.float32, device=device)
    got_wp = od_reduce.weighted_sum(values_wp, weights_wp)
    assert np.allclose(got_wp, exp_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("weighted_sum", "pyvista")
def test_weighted_sum_integrates_a_surface_field(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: ``sum(values * areas)`` is what VTK's ``integrate_data`` computes for a cell array.

    The weights are the triangle areas, so the reduction *is* the surface integral of a piecewise
    constant field, and VTK reports it as a one-cell grid carrying the integral under the same array
    name. Measured 8 significant digits on this fixture; the float32 accumulator is the limit.

    The field is deliberately **asymmetric** (the first corner's ``z``, plus an offset so it does
    not change sign). ``integrate_data`` of a symmetric field on a symmetric fixture reads
    ``-1.2e-15``
    -- a comparison against that number passes for any implementation that returns roughly zero, so
    it would be testing the fixture's symmetry rather than the reduction.
    """
    mesh_tm, mesh_wp = half_torus
    mesh_pv = trimesh_to_pyvista(mesh_tm)
    areas_np = np.asarray(
        mesh_pv.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"]
    )
    values_np = np.ascontiguousarray(mesh_tm.vertices[mesh_tm.faces[:, 0], 2] + 3.0)

    mesh_pv.cell_data["field"] = values_np
    integral_pv = float(np.asarray(mesh_pv.integrate_data().cell_data["field"])[0])
    assert abs(integral_pv) > 1.0  # non-vacuous: a near-zero integral would pass trivially

    total_wp = od_reduce.weighted_sum(
        wp.array(values_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.device),
        wp.array(areas_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.device),
    )
    assert np.isclose(total_wp, integral_pv, rtol=1e-5, atol=1e-5)


def test_weighted_sum_length_mismatch_raises(device: str) -> None:
    values_wp = wp.array([1.0, 2.0], dtype=wp.float32, device=device)
    weights_wp = wp.array([1.0], dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="equal length"):
        od_reduce.weighted_sum(values_wp, weights_wp)


def test_weighted_sum_rank2_raises(device: str) -> None:
    """A rank-2 argument must raise rather than silently reading its leading dimension."""
    values_wp = wp.array([[1.0, 2.0]], dtype=wp.float32, device=device)
    weights_wp = wp.array([1.0], dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="requires rank-1"):
        od_reduce.weighted_sum(values_wp, weights_wp)


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_sum_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    values_np = rng.integers(-1000, 1000, shape, dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    sum_wp = od_reduce.sum(values_wp)
    assert np.allclose(sum_wp, values_np.sum())


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_sum_bool_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    mask_np = rng.choice([False, True], size=shape, replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert od_reduce.sum(mask_wp) == int(mask_np.sum())


@pytest.mark.parametrize(
    ("shape", "axis"), [((9, 9), 0), ((9, 9), 1), ((65, 10), 0), ((65, 10), 1)]
)
def test_sum_partial_tiles_axis(device: str, shape: tuple[int, int], axis: Literal[0, 1]) -> None:
    rng = np.random.default_rng(99)
    values_np = rng.integers(-1000, 1000, shape, dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    assert np.array_equal(od_reduce.sum(values_wp, axis=axis).numpy(), values_np.sum(axis=axis))


def test_mean_1d_float(device: str) -> None:
    values_np = _random_values(100)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    mean_wp = od_reduce.mean(values_wp)
    assert np.allclose(mean_wp, values_np.mean(), rtol=1e-5, atol=1e-5)


def test_mean_1d_int(device: str) -> None:
    rng = np.random.default_rng(42)
    values_np = rng.integers(-1000, 1000, (100,), dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    mean_wp = od_reduce.mean(values_wp)
    assert np.allclose(mean_wp, values_np.mean(), rtol=1e-5, atol=1e-5)


def test_mean_2d(device: str) -> None:
    values_np = _random_values((200, 100))
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    mean_wp = od_reduce.mean(values_wp)
    assert np.allclose(mean_wp, values_np.mean(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
def test_mean_2d_axis(device: str, axis: Literal[0, 1]) -> None:
    values_np = _random_values((32, 10))
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    mean_wp = od_reduce.mean(values_wp, axis=axis)
    assert np.allclose(mean_wp.numpy(), values_np.mean(axis=axis), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
def test_mean_2d_axis_int(device: str, axis: Literal[0, 1]) -> None:
    rng = np.random.default_rng(42)
    values_np = rng.integers(-1000, 1000, (32, 10), dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    mean_wp = od_reduce.mean(values_wp, axis=axis)
    assert np.allclose(mean_wp.numpy(), values_np.mean(axis=axis), rtol=1e-5, atol=1e-5)


def test_mean_bool_global(device: str) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(32, 4), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert np.allclose(od_reduce.mean(mask_wp), mask_np.mean(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
def test_mean_bool_2d_axis(device: str, axis: Literal[0, 1]) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(32, 4), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    mean_wp = od_reduce.mean(mask_wp, axis=axis)
    assert np.allclose(mean_wp.numpy(), mask_np.mean(axis=axis), rtol=1e-5, atol=1e-5)


def test_mean_1d_axis_raises(device: str) -> None:
    values_wp = wp.array([1, 2, 3], dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="requires axis=None for a 1D array"):
        od_reduce.mean(values_wp, axis=0)


@pytest.mark.parity("mean_vec3", "numpy")
def test_mean_vec3_1d(device: str) -> None:
    """Class A: component-wise mean against ``numpy.mean(axis=0)``."""
    rng = np.random.default_rng(20)
    values_np = rng.standard_normal((300, 3)).astype(np.float32)
    values_wp = points_to_warp(values_np, device)
    mean_wp = od_reduce.mean(values_wp)
    assert np.allclose(np.array(mean_wp), values_np.mean(axis=0), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [63, 64, 65, 197])
def test_mean_vec3_partial_tiles(device: str, n: int) -> None:
    rng = np.random.default_rng(n)
    values_np = rng.standard_normal((n, 3)).astype(np.float32)
    values_wp = points_to_warp(values_np, device)
    mean_wp = od_reduce.mean(values_wp)
    assert np.allclose(np.array(mean_wp), values_np.mean(axis=0), rtol=1e-4, atol=1e-4)


def test_mean_vec3_axis_raises(device: str) -> None:
    values_wp = wp.zeros(4, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="axis"):
        od_reduce.mean(values_wp, axis=0)


def test_mean_vec3_empty_raises(device: str) -> None:
    values_wp = warp_empty(0, wp.vec3, device)
    with pytest.raises(ValueError, match="non-empty"):
        od_reduce.mean(values_wp)


@pytest.mark.parity("sum_vec3", "numpy")
def test_sum_vec3_1d(device: str) -> None:
    """Class A: component-wise sum against ``numpy.sum(axis=0)``."""
    rng = np.random.default_rng(20)
    values_np = rng.standard_normal((300, 3)).astype(np.float32)
    values_wp = points_to_warp(values_np, device)
    sum_wp = od_reduce.sum(values_wp)
    assert np.allclose(np.array(sum_wp), values_np.sum(axis=0), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [63, 64, 65, 197])
def test_sum_vec3_partial_tiles(device: str, n: int) -> None:
    rng = np.random.default_rng(n)
    values_np = rng.standard_normal((n, 3)).astype(np.float32)
    values_wp = points_to_warp(values_np, device)
    sum_wp = od_reduce.sum(values_wp)
    assert np.allclose(np.array(sum_wp), values_np.sum(axis=0), rtol=1e-4, atol=1e-4)


def test_sum_vec3_axis_raises(device: str) -> None:
    values_wp = wp.zeros(4, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="axis"):
        od_reduce.sum(values_wp, axis=0)


def test_sum_vec3_empty_raises(device: str) -> None:
    values_wp = warp_empty(0, wp.vec3, device)
    with pytest.raises(ValueError, match="non-empty"):
        od_reduce.sum(values_wp)


def test_weighted_sum_vec3_1d(device: str) -> None:
    rng = np.random.default_rng(21)
    values_np = rng.standard_normal((300, 3)).astype(np.float32)
    weights_np = rng.random(300, dtype=np.float32)
    values_wp = points_to_warp(values_np, device)
    weights_wp = wp.array(weights_np, dtype=wp.float32, device=device)
    sum_wp = od_reduce.weighted_sum(values_wp, weights_wp)
    exp_np = (weights_np[:, None] * values_np).sum(axis=0)
    assert np.allclose(np.array(sum_wp), exp_np, rtol=1e-4, atol=1e-4)


@pytest.mark.parity("weighted_sum", "pytorch3d")
def test_weighted_sum_matches_pytorch3d(device: str) -> None:
    """
    Class B: ``ops.utils.wmean`` is ``weighted_sum`` divided by ``sum(weights)``.

    pytorch3d's is the weighted *mean* -- ``sum(w * x) / sum(w)`` with an ``eps=1e-9`` floor under
    the denominator -- so dividing ordito's weighted sum by the plain sum of the same weights is
    the whole transform, and the two agree to 2.79e-08 over 50 ``vec3`` samples. Both reductions
    are exercised, which is what makes the pair a check on ``weighted_sum`` rather than on the
    division: a wrong numerator and a wrong denominator would have to cancel.

    The ``eps`` never bites here (the weights are ``rng.random``, so the sum is far from zero) and
    ordito has no counterpart for it, which is why the fixture avoids the case rather than
    asserting on it.
    """
    rng = np.random.default_rng(9)
    values_np = rng.normal(size=(50, 3)).astype(np.float32)
    weights_np = rng.random(50).astype(np.float32)
    mean_p3d = p3d_ops_utils.wmean(
        points_to_torch(values_np, device), torch.as_tensor(weights_np, device=device).unsqueeze(0)
    )[0, 0]

    values_wp = points_to_warp(values_np, device)
    weights_wp = wp.array(weights_np, dtype=wp.float32, device=device)
    total_wp = od_reduce.weighted_sum(values_wp, weights_wp)

    assert float(np.abs(mean_p3d.cpu().numpy()).max()) > 1e-3
    assert np.allclose(
        np.array(list(total_wp)) / od_reduce.sum(weights_wp),
        mean_p3d.cpu().numpy(),
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("n", [100, 101])
@pytest.mark.parity("median", "pymeshlab", "numpy")
def test_scalar_statistics_match_pymeshlab(device: str, n: int) -> None:
    """
    Class B: ``get_scalar_statistics_per_vertex`` answers four of these reductions in one call.

    The named transform is the dict index -- ``"min"``, ``"max"``, ``"avg"`` -- and those three are
    exact. Reading all of them off the one call is also what makes them *mutually* consistent, which
    no single-reduction comparison can check.

    **``"med"`` is not ordito's median, and the difference is a definition rather than a
    tolerance.** MeshLab reports the sorted element at index ``n // 2 - 1``, one *below* the middle,
    for both parities: measured at n = 100, 101 and 1001 it returns ranks 49, 49 and 499 where the
    middle is 49.5, 50 and 500. So it is compared against that named order statistic instead, which
    still checks that the two see the same sorted distribution, while
    [`tests.test_polyline.test_reduce_median_matches_numpy`][] is the oracle for
    [`ordito.reduce.median`][] itself.
    """
    rng = np.random.default_rng(0)
    values_np = rng.standard_normal(n)
    # A face-less MeshSet carrying the values as its vertex scalar attribute: the reduction is over
    # a bare array, so the positions are arbitrary and only the scalars matter.
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(rng.standard_normal((n, 3))),
            v_scalar_array=np.ascontiguousarray(values_np),
        )
    )
    statistics_pml = meshset_pml.get_scalar_statistics_per_vertex()

    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=device)
    assert np.isclose(od_reduce.min(values_wp), statistics_pml["min"], rtol=1e-5, atol=1e-5)
    assert np.isclose(od_reduce.max(values_wp), statistics_pml["max"], rtol=1e-5, atol=1e-5)
    assert np.isclose(od_reduce.mean(values_wp), statistics_pml["avg"], rtol=1e-5, atol=1e-5)

    assert np.isclose(od_reduce.median(values_wp), np.median(values_np), rtol=1e-5, atol=1e-5)
    assert np.isclose(statistics_pml["med"], np.sort(values_np)[n // 2 - 1], rtol=1e-5, atol=1e-5)
