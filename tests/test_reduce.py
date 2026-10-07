from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pymeshlab as ml
import pytest
import pytorch3d.ops.utils as p3d_ops_utils
import torch
import warp as wp

import ordito.reduce as od_reduce
import ordito.typing as odt
from tests.conversions import points_to_torch, points_to_warp, trimesh_to_pyvista, warp_empty

# Bound once as untyped callables: the tests below pass ``axis`` as a ``Literal[0, 1] | None``
# variable, which no single overload of the public functions accepts.
_MIN: Callable[..., Any] = od_reduce.min
_MAX: Callable[..., Any] = od_reduce.max
_MINMAX: Callable[..., Any] = od_reduce.minmax
_SUM: Callable[..., Any] = od_reduce.sum
_MEAN: Callable[..., Any] = od_reduce.mean
_ANY: Callable[..., Any] = od_reduce.any
_ALL: Callable[..., Any] = od_reduce.all

_SCALAR_DTYPES = [
    pytest.param(np.int32, wp.int32, id="int32"),
    pytest.param(np.float32, wp.float32, id="float32"),
]


def _scalar_values(shape: tuple[int, ...], dtype_np: type, seed: int) -> np.ndarray:
    """Fixed-seed test data: integers in ``[-1000, 1000)`` or float32 standard normals."""
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype_np, np.integer):
        return rng.integers(-1000, 1000, shape, dtype=np.int32)
    return rng.standard_normal(shape, dtype=np.float32)


def _random_mask(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Fixed-seed mask with each element drawn ``True`` or ``False`` with equal probability."""
    return np.random.default_rng(seed).choice([False, True], size=shape, replace=True)


def _host(value: Any) -> np.ndarray:
    """Read a reduction's result as NumPy: an axis result read back, a scalar wrapped."""
    return value.numpy() if isinstance(value, wp.array) else np.asarray(value)


def _assert_scalar_reductions(
    values_np: np.ndarray, values_wp: wp.array[Any], axis: Literal[0, 1] | None = None
) -> None:
    """
    Assert ``min``, ``max``, ``minmax``, ``sum`` and ``mean`` of ``values_wp`` equal NumPy's.

    The extrema and an integer sum are exact; a float sum and every mean are compared at
    ``1e-5``.
    """
    lower, upper = _MINMAX(values_wp, axis=axis)
    for name, got, expected in (
        ("min", _MIN(values_wp, axis=axis), values_np.min(axis=axis)),
        ("max", _MAX(values_wp, axis=axis), values_np.max(axis=axis)),
        ("minmax lower", lower, values_np.min(axis=axis)),
        ("minmax upper", upper, values_np.max(axis=axis)),
    ):
        assert np.array_equal(_host(got), expected), name
    sum_wp = _host(_SUM(values_wp, axis=axis))
    if np.issubdtype(values_np.dtype, np.integer):
        assert np.array_equal(sum_wp, values_np.sum(axis=axis)), "sum"
    else:
        assert np.allclose(sum_wp, values_np.sum(axis=axis), rtol=1e-5, atol=1e-5), "sum"
    mean_wp = _host(_MEAN(values_wp, axis=axis))
    assert np.allclose(mean_wp, values_np.mean(axis=axis), rtol=1e-5, atol=1e-5), "mean"


def _assert_bool_reductions(
    mask_np: np.ndarray, mask_wp: wp.array[wp.bool], axis: Literal[0, 1] | None = None
) -> None:
    """Assert ``any``, ``all``, ``sum`` (a count) and ``mean`` of ``mask_wp`` equal NumPy's."""
    assert np.array_equal(_host(_ANY(mask_wp, axis=axis)), np.any(mask_np, axis=axis)), "any"
    assert np.array_equal(_host(_ALL(mask_wp, axis=axis)), np.all(mask_np, axis=axis)), "all"
    count_np = mask_np.sum(axis=axis)
    assert np.array_equal(_host(_SUM(mask_wp, axis=axis)), count_np), "sum"
    mean_wp = _host(_MEAN(mask_wp, axis=axis))
    assert np.allclose(mean_wp, mask_np.mean(axis=axis), rtol=1e-5, atol=1e-5), "mean"


@pytest.mark.parametrize(("dtype_np", "dtype_wp"), _SCALAR_DTYPES)
@pytest.mark.parity("min_scalar", "numpy")
@pytest.mark.parity("minmax_scalar", "numpy")
@pytest.mark.parity("sum_scalar", "numpy")
def test_scalar_reductions_1d(device: str, dtype_np: type, dtype_wp: type) -> None:
    """
    Class A: ``min``, ``max``, ``minmax``, ``sum`` and ``mean`` of a rank-1 array against NumPy.

    One call each on 100 values, at an integer and a float dtype.
    """
    values_np = _scalar_values((100,), dtype_np, seed=42)
    _assert_scalar_reductions(values_np, wp.array(values_np, dtype=dtype_wp, device=device))


@pytest.mark.parametrize("shape", [(200, 100), (5000, 2), (5000, 3), (3, 5000)])
@pytest.mark.parity("minmax_global_2d", "numpy")
def test_scalar_reductions_2d_global(device: str, shape: tuple[int, int]) -> None:
    """
    Class A: rank-2 ``axis=None`` reductions against NumPy.

    Parametrized over narrow *and* wide trailing extents on purpose: ``(m, 2)`` is the edge-table
    shape [`ordito.graph.connected_components`][] validates, and it clips the ``TILE_2D`` square
    so the tile branch never runs — a wide-only fixture would leave that path untested.
    """
    values_np = _scalar_values(shape, np.float32, seed=42)
    _assert_scalar_reductions(values_np, wp.array(values_np, dtype=wp.float32, device=device))


@pytest.mark.parametrize("shape", [(9, 9), (65, 10), (16, 16), (3, 5), (200, 100)])
def test_scalar_reduce_global_noncontiguous(device: str, shape: tuple[int, int]) -> None:
    """
    The last remaining exerciser of ``kernels/reduce.py``'s rank-2 ``axis=None`` kernels.

    A non-contiguous rank-2 array can't flatten onto the 1-D kernel, so this is the only path left
    to reach them: every contiguous array now flattens there instead
    (``reduce._flattened_for_global``). Shapes span the full-tile case, both single- and
    double-boundary-short cases, and a wide table.
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
def test_reductions_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    """
    Global reductions over arrays that end mid-tile, against NumPy.

    Every scalar reduction at an integer and a float dtype, and every mask reduction, on a rank-1
    length one past a tile and on rank-2 shapes short of a tile in one or both dimensions.
    """
    for dtype_np, dtype_wp in ((np.int32, wp.int32), (np.float32, wp.float32)):
        values_np = _scalar_values(shape, dtype_np, seed=99)
        _assert_scalar_reductions(values_np, wp.array(values_np, dtype=dtype_wp, device=device))
    mask_np = _random_mask(shape, seed=99)
    _assert_bool_reductions(mask_np, wp.array(mask_np, dtype=wp.bool, device=device))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("shape", [(32, 10), (9, 9), (65, 10)])
@pytest.mark.parametrize(("dtype_np", "dtype_wp"), _SCALAR_DTYPES)
@pytest.mark.parity("max_axis1", "numpy")
@pytest.mark.parity("sum_axis0", "numpy")
def test_scalar_reductions_along_an_axis(
    device: str, dtype_np: type, dtype_wp: type, shape: tuple[int, int], axis: Literal[0, 1]
) -> None:
    """
    Class A: ``min`` / ``max`` / ``minmax`` / ``sum`` / ``mean`` with ``axis=...`` against NumPy.

    Both axes, at a shape wider than a tile and two that end mid-tile.
    """
    values_np = _scalar_values(shape, dtype_np, seed=42)
    _assert_scalar_reductions(
        values_np, wp.array(values_np, dtype=dtype_wp, device=device), axis=axis
    )


@pytest.mark.parametrize(
    "mask_np",
    [
        pytest.param(_random_mask((100,), seed=42), id="random_1d"),
        pytest.param(_random_mask((32, 4), seed=42), id="random_2d"),
        pytest.param(np.zeros((32, 4), dtype=bool), id="all_false_2d"),
        pytest.param(np.ones((32, 4), dtype=bool), id="all_true_2d"),
    ],
)
@pytest.mark.parity("any_bool", "numpy")
@pytest.mark.parity("all_bool", "numpy")
@pytest.mark.parity("sum_bool", "numpy")
def test_mask_reductions_global(device: str, mask_np: np.ndarray) -> None:
    """
    Class A: ``any``, ``all``, ``sum`` and ``mean`` of a ``wp.bool`` mask against NumPy.

    The random masks are about half set, so neither an all-``True`` nor an all-``False`` shortcut
    passes them; the constant masks give ``any`` and ``all`` their other answers. ``sum`` counts the
    mask without widening it to ``int32`` first, so this is the value gate on
    ``kernels.reduce._reduce_bool_1d_tiled``.
    """
    _assert_bool_reductions(mask_np, wp.array(mask_np, dtype=wp.bool, device=device))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("shape", [(32, 4), (9, 9), (65, 10)])
def test_mask_reductions_along_an_axis(
    device: str, shape: tuple[int, int], axis: Literal[0, 1]
) -> None:
    """``any`` / ``all`` / ``sum`` / ``mean`` of a mask with ``axis=...`` against NumPy."""
    mask_np = _random_mask(shape, seed=42)
    _assert_bool_reductions(mask_np, wp.array(mask_np, dtype=wp.bool, device=device), axis=axis)


@pytest.mark.parametrize("n", [63, 64, 65, 197, 300, 500])
@pytest.mark.parity("sum_vec3", "numpy")
@pytest.mark.parity("mean_vec3", "numpy")
def test_vec3_reductions(device: str, n: int) -> None:
    """
    Class A: component-wise ``sum``, ``mean``, ``weighted_sum`` and ``minmax`` of ``wp.vec3``s.

    Against ``numpy.sum`` / ``mean`` / ``min`` / ``max`` with ``axis=0`` and the weighted NumPy
    sum. ``minmax`` returns the corner pair the ``aabb`` reduction uses. Lengths straddle a tile
    boundary on both sides.
    """
    rng = np.random.default_rng(n)
    values_np = rng.standard_normal((n, 3)).astype(np.float32)
    weights_np = rng.random(n, dtype=np.float32)
    values_wp = points_to_warp(values_np, device)
    weights_wp = wp.array(weights_np, dtype=wp.float32, device=device)

    assert np.allclose(
        np.array(od_reduce.sum(values_wp)), values_np.sum(axis=0), rtol=1e-4, atol=1e-4
    )
    assert np.allclose(
        np.array(od_reduce.mean(values_wp)), values_np.mean(axis=0), rtol=1e-4, atol=1e-4
    )
    weighted_np = (weights_np[:, None] * values_np).sum(axis=0)
    weighted_wp = od_reduce.weighted_sum(values_wp, weights_wp)
    assert np.allclose(np.array(weighted_wp), weighted_np, rtol=1e-4, atol=1e-4)
    lower_wp, upper_wp = od_reduce.minmax(values_wp)
    assert np.allclose(np.array(list(lower_wp)), values_np.min(axis=0), rtol=1e-6, atol=1e-6)
    assert np.allclose(np.array(list(upper_wp)), values_np.max(axis=0), rtol=1e-6, atol=1e-6)


def _ints_1d(device: str) -> wp.array[wp.int32]:
    return wp.array([1, 2, 3], dtype=wp.int32, device=device)


def _ints_2d(device: str) -> wp.array[wp.int32]:
    return wp.array([[1, 2], [3, 4]], dtype=wp.int32, device=device)


def _mask_2d(device: str) -> wp.array[wp.bool]:
    return wp.array([[True, False], [False, True]], dtype=wp.bool, device=device)


# One row per rejected call: an id, the call on a device, and the message it must raise.
_REJECTED_CALLS: list[tuple[str, Callable[[str], object], str]] = [
    *(
        (
            f"{name}_1d_axis",
            lambda device, fn=fn: fn(_ints_1d(device), axis=0),
            "requires axis=None for a 1D array",
        )
        for name, fn in (
            ("min", _MIN),
            ("max", _MAX),
            ("minmax", _MINMAX),
            ("sum", _SUM),
            ("mean", _MEAN),
        )
    ),
    (
        "sum_bool_1d_axis",
        lambda device: _SUM(wp.array([True, False, True], dtype=wp.bool, device=device), axis=0),
        "requires axis=None for a 1D array",
    ),
    *(
        (
            f"{name}_2d_axis_2",
            lambda device, fn=fn: fn(_ints_2d(device), axis=2),
            "requires axis to be 0, 1, or None",
        )
        for name, fn in (("min", _MIN), ("max", _MAX), ("minmax", _MINMAX), ("sum", _SUM))
    ),
    *(
        (
            f"{name}_bool_2d_axis_2",
            lambda device, fn=fn: fn(_mask_2d(device), axis=2),
            "requires axis to be 0, 1, or None",
        )
        for name, fn in (("any", _ANY), ("all", _ALL))
    ),
    *(
        (
            f"{name}_vec3_axis",
            lambda device, fn=fn: fn(wp.zeros(4, dtype=wp.vec3, device=device), axis=0),
            "axis=None",
        )
        for name, fn in (("minmax", _MINMAX), ("sum", _SUM), ("mean", _MEAN))
    ),
    *(
        (
            f"{name}_vec3_empty",
            lambda device, fn=fn: fn(warp_empty(0, wp.vec3, device)),
            "non-empty",
        )
        for name, fn in (("minmax", _MINMAX), ("sum", _SUM), ("mean", _MEAN))
    ),
    (
        "weighted_sum_length_mismatch",
        lambda device: od_reduce.weighted_sum(
            wp.array([1.0, 2.0], dtype=wp.float32, device=device),
            wp.array([1.0], dtype=wp.float32, device=device),
        ),
        "equal length",
    ),
    (
        "weighted_sum_rank2",
        lambda device: od_reduce.weighted_sum(
            wp.array([[1.0, 2.0]], dtype=wp.float32, device=device),
            wp.array([1.0], dtype=wp.float32, device=device),
        ),
        "requires rank-1",
    ),
]


@pytest.mark.parametrize(
    ("call", "match"), [pytest.param(call, match, id=name) for name, call, match in _REJECTED_CALLS]
)
def test_invalid_reductions_raise(device: str, call: Callable[[str], object], match: str) -> None:
    """
    Each rejected argument raises ``ValueError`` rather than being read some other way.

    An ``axis`` on a rank-1 array, an axis outside ``{0, 1, None}`` on a rank-2 one (instead of
    silently reading as ``axis=0``), any ``axis`` or an empty input on a ``wp.vec3`` array, and
    ``weighted_sum`` arguments of unequal length or rank 2 (instead of reading the leading
    dimension).
    """
    with pytest.raises(ValueError, match=match):
        call(device)


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


@pytest.mark.parametrize("mesh_name", ["half_torus", "saddle_graded"])
@pytest.mark.parity("weighted_sum", "pyvista")
def test_weighted_sum_integrates_a_surface_field(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
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
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
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


@pytest.mark.parametrize("n", [7, 8])
@pytest.mark.parametrize(
    ("dtype_wp", "dtype_np"),
    [
        (wp.float32, np.float32),
        (wp.float64, np.float64),
        (wp.int32, np.int32),
        (wp.int64, np.int64),
        (wp.uint32, np.uint32),
        (wp.uint64, np.uint64),
    ],
)
def test_reduce_median_matches_numpy(device: str, n: int, dtype_wp: type, dtype_np: type) -> None:
    """Class A: ``reduce.median`` equals ``np.median`` at both parities and every key dtype."""
    rng = np.random.default_rng(90 + n)
    if np.issubdtype(dtype_np, np.floating):
        values_np = rng.standard_normal(n).astype(dtype_np)
    else:
        values_np = rng.integers(0, 1000, size=n).astype(dtype_np)
    values_wp = wp.array(values_np, dtype=dtype_wp, device=device)
    assert np.allclose(od_reduce.median(values_wp), np.median(values_np), rtol=1e-5, atol=1e-5)


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
    [`tests.test_reduce.test_reduce_median_matches_numpy`][] is the oracle for
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
