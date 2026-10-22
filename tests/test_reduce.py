from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import pymeshlab as ml
import pytest
import pytorch3d.ops.utils as p3d_ops_utils
import torch
import warp as wp

import ordito.reduce as od_reduce
import ordito.typing as odt
from ordito import _launch
from ordito.kernels import energies as kernel_energies
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import scatter as kernel_scatter
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


@pytest.mark.parametrize("shape", [(200, 100), (5000, 2)])
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


@pytest.mark.parametrize("shape", [(65, 10), (16, 16), (3, 5), (200, 100)])
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


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("shape", [(32, 10), (65, 10)])
@pytest.mark.parametrize(("dtype_np", "dtype_wp"), _SCALAR_DTYPES)
@pytest.mark.parity("max_axis1", "numpy")
@pytest.mark.parity("sum_axis0", "numpy")
def test_scalar_reductions_along_an_axis(
    device: str, dtype_np: type, dtype_wp: type, shape: tuple[int, int], axis: Literal[0, 1]
) -> None:
    """
    Class A: ``min`` / ``max`` / ``minmax`` / ``sum`` / ``mean`` with ``axis=...`` against NumPy.

    Both axes, at a shape one row past a tile and one under a tile.
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
@pytest.mark.parametrize("shape", [(32, 4), (65, 10)])
def test_mask_reductions_along_an_axis(
    device: str, shape: tuple[int, int], axis: Literal[0, 1]
) -> None:
    """``any`` / ``all`` / ``sum`` / ``mean`` of a mask with ``axis=...`` against NumPy."""
    mask_np = _random_mask(shape, seed=42)
    _assert_bool_reductions(mask_np, wp.array(mask_np, dtype=wp.bool, device=device), axis=axis)


@pytest.mark.parametrize("n", [63, 64, 65, 197])
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
            f"{name}_empty",
            lambda device, fn=fn: fn(warp_empty(0, wp.int32, device)),
            f"{name} requires a non-empty array",
        )
        for name, fn in (("min", _MIN), ("max", _MAX))
    ),
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
    (
        "weighted_sum_empty",
        lambda device: od_reduce.weighted_sum(
            warp_empty(0, wp.float32, device), warp_empty(0, wp.float32, device)
        ),
        "non-empty arrays",
    ),
    *(
        (
            f"{name}_vec3_rank2",
            lambda device, fn=fn: fn(wp.zeros((2, 2), dtype=wp.vec3, device=device)),
            "1D array|rank-1 wp.vec3",
        )
        for name, fn in (("minmax", _MINMAX), ("sum", _SUM))
    ),
    (
        "sum_bool_empty",
        lambda device: _SUM(warp_empty(0, wp.bool, device)),
        "sum requires a non-empty array",
    ),
    *(
        (
            f"{name}_bool_empty",
            lambda device, fn=fn: fn(warp_empty(0, wp.bool, device)),
            f"{name} requires a non-empty array",
        )
        for name, fn in (("any", _ANY), ("all", _ALL))
    ),
    *(
        (
            f"{name}_bool_rank3",
            lambda device, fn=fn: fn(wp.zeros((2, 2, 2), dtype=wp.bool, device=device)),
            "requires a 1D or 2D array",
        )
        for name, fn in (("any", _ANY), ("all", _ALL))
    ),
    *(
        (
            f"{name}_rank3",
            lambda device, fn=fn: fn(wp.zeros((2, 2, 2), dtype=wp.int32, device=device)),
            "requires a 1D or 2D array",
        )
        for name, fn in (("min", _MIN), ("max", _MAX))
    ),
    (
        "median_rank2",
        lambda device: od_reduce.median(_ints_2d(device)),
        "median requires a 1D array",
    ),
    (
        "median_empty",
        lambda device: od_reduce.median(warp_empty(0, wp.float32, device)),
        "median requires a non-empty array",
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
    dimension). Empty and rank-3 inputs, a rank-2 ``wp.vec3`` array, and a ``median`` of an empty
    or rank-2 array are each rejected by name.
    """
    with pytest.raises(ValueError, match=match):
        call(device)


@pytest.mark.parametrize("n", [1_000, 50_000, 3_000_000])
@pytest.mark.parity("weighted_sum", "numpy")
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


@pytest.mark.parametrize("deterministic", [True, False], indirect=True)
@pytest.mark.parametrize(
    ("shape", "axis"),
    [((1_000_000, 3), 0), ((1000, 1000), 0), ((1000, 1000), 1), ((5000, 10), 0), ((50_000,), None)],
)
def test_float_axis_sums_on_both_commits(
    device: str, shape: tuple[int, ...], axis: Literal[0, 1] | None, deterministic: bool
) -> None:
    """
    Not a library comparison: float sums and means against NumPy in float64, on both commits.

    The per-axis shapes reduce over thousands of tiles per slot, where one atomic commit per tile
    (determinism off) moved the last bits from call to call on CUDA; the fixed-order commit
    must repeat bit for bit. Both arms must match NumPy to float32 rounding.
    """
    values_np = np.random.default_rng(8).random(shape, dtype=np.float32) + np.float32(0.5)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    expected_sum = values_np.astype(np.float64).sum(axis=axis)
    expected_mean = values_np.astype(np.float64).mean(axis=axis)
    if axis is None:
        sum_np = np.asarray(od_reduce.sum(values_wp))
        mean_np = np.asarray(od_reduce.mean(values_wp))
    else:
        sum_np = od_reduce.sum(values_wp, axis=axis).numpy()
        mean_np = od_reduce.mean(values_wp, axis=axis).numpy()
    assert np.allclose(sum_np, expected_sum, rtol=1e-5, atol=0.0)
    assert np.allclose(mean_np, expected_mean, rtol=1e-5, atol=0.0)
    if deterministic and axis is not None:
        assert np.array_equal(od_reduce.sum(values_wp, axis=axis).numpy(), sum_np)


@pytest.mark.parametrize("mesh_name", ["saddle_graded"])
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


@pytest.mark.parametrize("deterministic", [True, False], indirect=True)
@pytest.mark.parametrize("n", [1, 100, 1_024, 5_000, 1_100_000])
def test_sum_of_terms_is_the_sum(device: str, n: int, deterministic: bool) -> None:
    """
    Ordito against ordito: ``sum_of_terms`` totals are ``sum`` and ``weighted_sum``, bit for bit.

    ``sum`` has the NumPy oracle (``test_float_sums_are_bit_identical_across_calls``); this pins
    the device-side entry to it, for ``sum``'s own kernels, the weighted ones and a fused term
    (``edge_length_loss``'s ``(x - 0)^2``, against ``sum`` of the materialized squares), each
    written into one slot of a shared buffer whose other slot it must not touch. The sizes span
    one block, one fold stage and two. The atomic form (determinism off) adds blocks in arrival
    order on CUDA, so there the totals agree to rounding instead.
    """
    rng = np.random.default_rng(11)
    values_np = rng.random(n, dtype=np.float32)
    weights_np = rng.random(n, dtype=np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    weights_wp = wp.array(weights_np, dtype=wp.float32, device=device)
    terms = kernel_energies.EdgeDeviationTerms()
    terms.n = n
    terms.lengths = values_wp
    terms.target = 0.0
    squares_wp = wp.array(values_np * values_np, dtype=wp.float32, device=device)

    # The atomic form adds into its slot, so that buffer starts at zero; the fourth slot is never
    # written and keeps its initial value.
    initial = -1.0 if deterministic else 0.0
    totals_wp = wp.full(4, initial, dtype=wp.float32, device=device)
    od_reduce.sum_of_terms(
        kernel_reduce.ARRAY_SUM, [values_wp], n, out=odt.as_dense(totals_wp[0:1])
    )
    od_reduce.sum_of_terms(
        kernel_reduce.WEIGHTED_SUM, [values_wp, weights_wp], n, out=odt.as_dense(totals_wp[1:2])
    )
    od_reduce.sum_of_terms(
        kernel_energies.EDGE_DEVIATION_SUM,
        [terms],
        n,
        out=odt.as_dense(totals_wp[2:3]),
        device=device,
    )
    alone_wp = od_reduce.sum_of_terms(kernel_reduce.ARRAY_SUM, [values_wp], n)

    expected_np = np.array(
        [
            od_reduce.sum(values_wp),
            od_reduce.weighted_sum(values_wp, weights_wp),
            od_reduce.sum(squares_wp),
            initial,
        ],
        dtype=np.float32,
    )
    assert expected_np[0] > 0.0
    assert alone_wp.shape == (1,)
    if deterministic or not wp.get_device(device).is_cuda:
        assert np.array_equal(totals_wp.numpy(), expected_np)
        assert np.array_equal(alone_wp.numpy(), expected_np[:1])
    else:
        assert np.allclose(totals_wp.numpy(), expected_np, rtol=1e-5, atol=0.0)


_GROUP_SUM_DTYPES = [
    pytest.param(wp.float32, np.float32, 1, id="float32"),
    pytest.param(wp.float64, np.float64, 1, id="float64"),
    pytest.param(wp.vec2, np.float32, 2, id="vec2"),
    pytest.param(wp.vec3, np.float32, 3, id="vec3"),
    pytest.param(wp.vec4, np.float32, 4, id="vec4"),
    pytest.param(wp.vec2d, np.float64, 2, id="vec2d"),
    pytest.param(wp.vec3d, np.float64, 3, id="vec3d"),
    pytest.param(wp.vec4d, np.float64, 4, id="vec4d"),
]


def _group_values(n: int, width: int, dtype_np: type, seed: int) -> np.ndarray:
    """``(n,)`` or ``(n, width)`` standard normals of ``dtype_np``, from a fixed seed."""
    shape = (n,) if width == 1 else (n, width)
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype_np)


@pytest.mark.parametrize("deterministic", [True, False], indirect=True)
@pytest.mark.parametrize(("dtype_wp", "dtype_np", "width"), _GROUP_SUM_DTYPES)
@pytest.mark.parity("scatter_sum", "numpy")
def test_scatter_sum_matches_numpy(
    device: str, dtype_wp: type, dtype_np: type, width: int, deterministic: bool
) -> None:
    """
    Class A: per-group sums against ``numpy.bincount`` in ``float64``, on both settings.

    50 000 values into 3 000 groups, a group no index names included (it sums to zero), with
    ``n_groups`` passed and with it inferred from the largest index. Vectors are compared per
    component (``bincount`` of each column).
    """
    n, n_groups = 50_000, 3_000
    values_np = _group_values(n, width, dtype_np, seed=12)
    rng = np.random.default_rng(13)
    indices_np = rng.integers(0, n_groups - 1, n).astype(np.int32)
    indices_np[indices_np == 7] = 8  # group 7 receives nothing
    columns_np = values_np.reshape(n, width).astype(np.float64)
    expected_np = np.stack(
        [np.bincount(indices_np, columns_np[:, c], minlength=n_groups) for c in range(width)], 1
    ).reshape((n_groups,) if width == 1 else (n_groups, width))
    assert np.ptp(expected_np) > 1.0
    values_wp = wp.array(values_np, dtype=dtype_wp, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    sums_wp = od_reduce.scatter_sum(values_wp, indices_wp, n_groups)
    inferred_wp = od_reduce.scatter_sum(values_wp, indices_wp)
    assert sums_wp.dtype == dtype_wp
    assert sums_wp.shape == (n_groups,)
    assert inferred_wp.shape == (n_groups - 1,)
    assert np.allclose(sums_wp.numpy(), expected_np, rtol=1e-5, atol=1e-4)
    assert np.allclose(inferred_wp.numpy(), expected_np[:-1], rtol=1e-5, atol=1e-4)
    assert np.all(sums_wp.numpy()[7] == 0.0)
    if deterministic:
        # Each group's sum depends on its own terms alone, not on how many groups there are.
        assert np.array_equal(inferred_wp.numpy(), sums_wp.numpy()[:-1])


@pytest.mark.parametrize(("dtype_wp", "dtype_np", "width"), _GROUP_SUM_DTYPES)
def test_scatter_sum_repeats_bit_for_bit_on_both_devices(
    device: str, dtype_wp: type, dtype_np: type, width: int
) -> None:
    """
    Ordito against ordito: under the setting the sums repeat bit for bit and equal the CPU's.

    One group holds a third of the values (every lane of a warp adds to one slot, the warp-summed
    commit), the rest spread over many; float atomics in arrival order moved the last bits of such
    sums from call to call on CUDA. ``test_scatter_sum_matches_numpy`` holds the value's oracle.
    """
    n = 200_000
    values_np = _group_values(n, width, dtype_np, seed=14)
    indices_np = np.random.default_rng(15).integers(0, 5_000, n).astype(np.int32)
    indices_np[: n // 3] = 0
    sums = [
        od_reduce.scatter_sum(
            wp.array(values_np, dtype=dtype_wp, device=device),
            wp.array(indices_np, dtype=wp.int32, device=device),
        ).numpy()
        for _ in range(4)
    ]
    on_cpu = od_reduce.scatter_sum(
        wp.array(values_np, dtype=dtype_wp, device="cpu"),
        wp.array(indices_np, dtype=wp.int32, device="cpu"),
    ).numpy()
    for repeat in [*sums[1:], on_cpu]:
        assert np.array_equal(repeat.view(np.uint8), sums[0].view(np.uint8))


def _scatter_sum_positive(dtype: Any, width: int) -> Any:
    """Build a kernel adding each ``dtype`` value whose first component is positive."""

    def _scalars(values: Any, indices: Any, sums: Any, out_sums: Any) -> None:
        i = wp.int32(wp.tid())  # pyright: ignore[reportArgumentType]
        if values[i] > type(values[i])(0.0):
            kernel_scatter.scatter_add(out_sums, sums, indices[i], values[i])

    def _vectors(values: Any, indices: Any, sums: Any, out_sums: Any) -> None:
        i = wp.int32(wp.tid())  # pyright: ignore[reportArgumentType]
        if values[i][0] > type(values[i][0])(0.0):
            kernel_scatter.scatter_add_vector(out_sums, sums, indices[i], values[i])

    _k = _scalars if width == 1 else _vectors
    _k.__annotations__.update(
        values=wp.array[dtype],
        indices=wp.array[wp.int32],
        sums=kernel_reduce.GroupSumPass,
        out_sums=wp.array[dtype],
        **{"return": None},
    )
    return wp.kernel(_k, name=f"test_scatter_sum_positive_{dtype.__name__}")


_SCATTER_SUM_POSITIVE = {
    dtype_wp: _scatter_sum_positive(dtype_wp, width)
    for dtype_wp, width in ((wp.float32, 1), (wp.vec3, 3), (wp.vec3d, 3))
}


@pytest.mark.parametrize("deterministic", [True, False], indirect=True)
@pytest.mark.parametrize(("dtype_wp", "dtype_np", "width"), _GROUP_SUM_DTYPES[::3])
@pytest.mark.parametrize("run", [1, 7, 10_000])
def test_group_sums_of_hot_groups_under_divergence(
    device: str, dtype_wp: type, dtype_np: type, width: int, run: int, deterministic: bool
) -> None:
    """
    Class A: a scatter taken by about half of each warp's lanes, against ``numpy.bincount``.

    Groups are runs of ``run`` consecutive items (1: every lane its own group; 7: warps shared
    by a few groups; 10 000: every warp on one group), and only the items whose first component
    is positive add, so a warp's active lanes are a scattered subset of it. With the setting off
    that is the warp-uniform pre-sum over a partial mask (``kernels/reduce.warp_lanes_sum``'s
    rank walk); a lane summed twice, or one dropped, moves its group by a whole term. The item
    count is not a multiple of 32, so the last warp is short too. Under the setting the sums also
    repeat bit for bit.
    """
    n = 100_003
    values_np = _group_values(n, width, dtype_np, seed=20)
    indices_np = (np.arange(n) // run).astype(np.int32)
    n_groups = int(indices_np[-1]) + 1
    columns_np = values_np.reshape(n, width).astype(np.float64)
    taken_np = columns_np[:, 0] > 0.0
    assert 0.4 < taken_np.mean() < 0.6
    expected_np = np.stack(
        [
            np.bincount(indices_np[taken_np], columns_np[taken_np, c], minlength=n_groups)
            for c in range(width)
        ],
        1,
    ).reshape((n_groups,) if width == 1 else (n_groups, width))
    values_wp = wp.array(values_np, dtype=dtype_wp, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    sums = [_launch.group_sums(n_groups, dtype_wp, n, device) for _ in range(2)]
    for pass_sums in sums:
        _launch.launch_group_sums(
            _SCATTER_SUM_POSITIVE[dtype_wp], n, [values_wp, indices_wp], pass_sums, device=device
        )
    sums_np, again_np = (_launch.finish_group_sums(pass_sums).numpy() for pass_sums in sums)
    assert np.allclose(sums_np, expected_np, rtol=1e-5, atol=1e-3)
    if deterministic:
        assert np.array_equal(again_np.view(np.uint8), sums_np.view(np.uint8))


@pytest.mark.parametrize(
    ("dtype_wp", "dtype_np"), [(wp.float32, np.float32), (wp.float64, np.float64)]
)
def test_scatter_sum_is_exact_to_its_bound(device: str, dtype_wp: type, dtype_np: type) -> None:
    """
    Not a library comparison: each group's sum against its exact rational sum.

    Values spanning 120 binades with both signs, and a group of exactly cancelling pairs; NumPy's
    ``float64`` sum has no such guarantee, so the oracle is ``fractions.Fraction``. The documented
    bound is each term rounded to ``2^-(124 - 2h)`` of its group's largest magnitude (two words,
    ``float64``) or ``2^-(62 - h)`` (one word, ``float32``), then the result's own rounding: the
    sums must sit within it, and the cancelling group must be exactly zero.
    """
    n, n_groups = 4_000, 20
    rng = np.random.default_rng(16)
    values_np = (rng.standard_normal(n) * np.exp2(rng.integers(-60, 60, n))).astype(dtype_np)
    indices_np = rng.integers(1, n_groups, n).astype(np.int32)
    pairs_np = rng.standard_normal(200).astype(dtype_np) * dtype_np(1e20)
    values_np = np.concatenate([values_np, pairs_np, -pairs_np])
    indices_np = np.concatenate([indices_np, np.zeros(400, np.int32)])
    sums_np = od_reduce.scatter_sum(
        wp.array(values_np, dtype=dtype_wp, device=device),
        wp.array(indices_np, dtype=wp.int32, device=device),
    ).numpy()
    assert sums_np[0] == 0.0
    headroom = (values_np.size - 1).bit_length()
    unit = 2.0 ** -(124 - 2 * headroom) if dtype_np is np.float64 else 2.0 ** -(62 - headroom)
    for g in range(1, n_groups):
        members_np = values_np[indices_np == g]
        exact = sum((Fraction(float(v)) for v in members_np), Fraction(0))
        largest = float(np.abs(members_np).max())
        allowed = members_np.size * unit * 2.0 * largest + 2.0 * np.spacing(dtype_np(abs(exact)))
        assert abs(Fraction(float(sums_np[g])) - exact) <= Fraction(float(allowed))


@pytest.mark.parametrize(
    ("dtype_wp", "dtype_np"), [(wp.float32, np.float32), (wp.float64, np.float64)]
)
def test_scatter_sum_edge_values(device: str, dtype_wp: type, dtype_np: type) -> None:
    """
    Not a library comparison: zeros, subnormals, equal values, extremes and non-finite groups.

    Group 0 holds zeros, group 1 subnormals (summed exactly: ``float(sum)`` of the integer
    multiples of the smallest subnormal), group 2 a thousand equal values, group 3 the largest
    finite value twice and its negation once (no overflow in the words), group 4 a NaN and group 5
    an infinity (both NaN under the setting, ``inf`` stays out of the words).
    """
    tiny = np.finfo(dtype_np).smallest_subnormal
    big = np.finfo(dtype_np).max
    multiples_np = np.arange(1, 101)
    values_np = np.concatenate(
        [
            np.zeros(5),
            multiples_np * tiny,
            np.full(1_000, 0.1),
            [big, big, -big],
            [1.0, np.nan],
            [1.0, np.inf],
        ]
    ).astype(dtype_np)
    indices_np = np.repeat(np.arange(6, dtype=np.int32), [5, 100, 1_000, 3, 2, 2])
    sums_np = od_reduce.scatter_sum(
        wp.array(values_np, dtype=dtype_wp, device=device),
        wp.array(indices_np, dtype=wp.int32, device=device),
    ).numpy()
    equal_exact = float(sum((Fraction(float(dtype_np(0.1))) for _ in range(1_000)), Fraction(0)))
    assert sums_np[0] == 0.0
    assert sums_np[1] == dtype_np(int(multiples_np.sum()) * tiny)
    assert sums_np[2] == dtype_np(equal_exact)
    assert sums_np[3] == big
    assert np.isnan(sums_np[4])
    assert np.isnan(sums_np[5])


def test_group_sum_headroom_limits(device: str) -> None:
    """
    Not a library comparison: the largest headroom sums exactly, and one more term raises.

    ``max_terms = 2**31`` is the most a slot can take with both words' sums inside ``int64``; a
    sum at that headroom must still be exact for well-scaled values, and ``2**31 + 1`` must be
    refused rather than wrap.
    """
    values_np = np.array([1.5, -0.25, 3.0, 2.0**-30], dtype=np.float64)
    values_wp = wp.array(values_np, dtype=wp.float64, device=device)
    indices_wp = wp.array([0, 0, 1, 1], dtype=wp.int32, device=device)
    sums = _launch.group_sums(2, wp.float64, 2**31, device)
    _launch.launch_group_sums(
        kernel_scatter.SCATTER_SUM_SCALARS[wp.float64],
        4,
        [values_wp, indices_wp],
        sums,
        device=device,
    )
    assert _launch.finish_group_sums(sums).numpy().tolist() == [1.25, 3.0 + 2.0**-30]
    with pytest.raises(ValueError, match="2\\*\\*31"):
        _launch.group_sums(2, wp.float64, 2**31 + 1, device)


@wp.kernel  # pyright: ignore[reportUntypedFunctionDecorator]  # wp.kernel has no return annotation
def _read_group_sums(
    sums_in: wp.array[wp.float32],
    sums: kernel_reduce.GroupSumPass,  # pyright: ignore[reportGeneralTypeIssues]  # a struct value
    out: wp.array[wp.float32],
) -> None:
    g = wp.int32(wp.tid())  # pyright: ignore[reportArgumentType]
    out[g] = kernel_reduce.read_group_sum(sums_in, sums, g)  # pyright: ignore[reportIndexIssue]


@pytest.mark.parametrize("deterministic", [True, False], indirect=True)
def test_read_group_sum_equals_the_finish(device: str, deterministic: bool) -> None:
    """
    Ordito against ordito: a consumer reading ``read_group_sum`` gets the finished sums' bits.

    ``_launch.group_sums(fused=True)`` allocates no result under the setting (``out`` is
    ``None``) and the consumer reads the words; off, it reads the accumulator. Either way it must
    equal ``finish_group_sums`` of the same scatter bit for bit, whose values
    ``test_scatter_sum_matches_numpy`` checks against NumPy.
    """
    n, n_groups = 30_000, 700
    rng = np.random.default_rng(20)
    values_wp = wp.array(rng.standard_normal(n).astype(np.float32), device=device)
    indices_np = rng.integers(0, n_groups, n).astype(np.int32)
    indices_np[: n // 4] = 3
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    kernel = kernel_scatter.SCATTER_SUM_SCALARS[wp.float32]
    finished = _launch.group_sums(n_groups, wp.float32, n, device)
    _launch.launch_group_sums(kernel, n, [values_wp, indices_wp], finished, device=device)
    expected_np = _launch.finish_group_sums(finished).numpy()
    fused = _launch.group_sums(n_groups, wp.float32, n, device, fused=True)
    assert (fused.out is None) == deterministic
    _launch.launch_group_sums(kernel, n, [values_wp, indices_wp], fused, device=device)
    read_wp = warp_empty(n_groups, wp.float32, device)
    wp.launch(
        _read_group_sums,
        dim=n_groups,
        inputs=[fused.out, fused.launch_pass, read_wp],
        device=device,
    )
    if deterministic or not wp.get_device(device).is_cuda:
        assert np.array_equal(read_wp.numpy().view(np.uint8), expected_np.view(np.uint8))
    else:
        assert np.allclose(read_wp.numpy(), expected_np, rtol=1e-5, atol=1e-5)


def _scatter_sum_matrices(dtype: Any) -> Any:
    """Build a kernel adding each ``dtype`` matrix into its group (``scatter_add_matrix``)."""

    def _k(values: Any, indices: Any, sums: Any, out_sums: Any) -> None:
        i = wp.int32(wp.tid())  # pyright: ignore[reportArgumentType]
        kernel_scatter.scatter_add_matrix(out_sums, sums, indices[i], values[i])

    _k.__annotations__.update(
        values=wp.array[dtype],
        indices=wp.array[wp.int32],
        sums=kernel_reduce.GroupSumPass,
        out_sums=wp.array[dtype],
        **{"return": None},
    )
    return wp.kernel(_k, name=f"test_scatter_sum_matrices_{dtype.__name__}")


_SCATTER_SUM_MATRICES = {dtype: _scatter_sum_matrices(dtype) for dtype in (wp.mat33, wp.mat33d)}


def _matrix_group_sums(
    values_np: np.ndarray, indices_np: np.ndarray, n_groups: int, dtype_wp: type, device: str
) -> np.ndarray:
    """``(n_groups, 3, 3)`` group sums of ``values_np`` through the matrix scatter and finish."""
    values_wp = wp.array(values_np, dtype=dtype_wp, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    sums = _launch.group_sums(n_groups, dtype_wp, values_np.shape[0], device)
    _launch.launch_group_sums(
        _SCATTER_SUM_MATRICES[dtype_wp],
        values_np.shape[0],
        [values_wp, indices_wp],
        sums,
        device=device,
    )
    return _launch.finish_group_sums(sums).numpy()


@pytest.mark.parametrize("deterministic", [True, False], indirect=True)
@pytest.mark.parametrize(
    ("dtype_wp", "dtype_np"), [(wp.mat33, np.float32), (wp.mat33d, np.float64)]
)
def test_matrix_group_sums_are_exact_and_repeat(
    device: str, dtype_wp: type, dtype_np: type, deterministic: bool
) -> None:
    """
    Not a library comparison: matrix group sums against each entry's exact rational sum.

    No public entry sums matrices, so the scatter (``kernels/scatter.scatter_add_matrix``) and the
    matrix finish (``FINISH_GROUP_SUMS[mat33 / mat33d]``) are driven directly. Entries span 80
    binades with both signs, one group holds a third of the terms (the warp-summed commit) and one
    receives nothing. Under the setting each entry must sit within the documented bound of its
    ``fractions.Fraction`` sum (``test_scatter_sum_is_exact_to_its_bound``'s, with the group's
    largest entry over all nine components, since they share one field), the sums must repeat bit
    for bit and equal the CPU's; off, they are ``numpy.add.at`` to ``1e-5``. A transposed slot
    (``(group * rows + c) * cols + r``) fails the comparison on every off-diagonal entry.
    """
    n, n_groups = 6_000, 40
    rng = np.random.default_rng(19)
    values_np = rng.standard_normal((n, 3, 3)) * np.exp2(rng.integers(-40, 40, (n, 3, 3)))
    values_np = values_np.astype(dtype_np)
    indices_np = rng.integers(0, n_groups - 1, n).astype(np.int32)
    indices_np[: n // 3] = 5
    sums_np = _matrix_group_sums(values_np, indices_np, n_groups, dtype_wp, device)
    assert sums_np.shape == (n_groups, 3, 3)
    assert np.all(sums_np[n_groups - 1] == 0.0)
    if not deterministic:
        expected_np = np.zeros((n_groups, 3, 3))
        np.add.at(expected_np, indices_np, values_np.astype(np.float64))
        scale_np = np.abs(values_np).max() * 1e-5
        assert np.allclose(sums_np, expected_np, rtol=1e-5, atol=float(scale_np))
        return
    headroom = (n - 1).bit_length()
    unit = 2.0 ** -(124 - 2 * headroom) if dtype_np is np.float64 else 2.0 ** -(62 - headroom)
    for g in range(n_groups - 1):
        members_np = values_np[indices_np == g]
        largest = float(np.abs(members_np).max())
        for r in range(3):
            for c in range(3):
                exact = sum((Fraction(float(v)) for v in members_np[:, r, c]), Fraction(0))
                allowed = members_np.shape[0] * unit * 2.0 * largest
                allowed += 2.0 * float(np.spacing(dtype_np(abs(exact))))
                error = abs(Fraction(float(sums_np[g, r, c])) - exact)
                assert error <= Fraction(allowed), (g, r, c)
    again_np = _matrix_group_sums(values_np, indices_np, n_groups, dtype_wp, device)
    on_cpu_np = _matrix_group_sums(values_np, indices_np, n_groups, dtype_wp, "cpu")
    assert np.array_equal(again_np.view(np.uint8), sums_np.view(np.uint8))
    assert np.array_equal(on_cpu_np.view(np.uint8), sums_np.view(np.uint8))


@pytest.mark.parametrize("deterministic", [True, False], indirect=True)
@pytest.mark.parametrize(("dtype_wp", "dtype_np", "width"), _GROUP_SUM_DTYPES[::3])
@pytest.mark.parity("segment_sum", "numpy")
def test_segment_sum_matches_numpy(
    device: str, dtype_wp: type, dtype_np: type, width: int, deterministic: bool
) -> None:
    """
    Class B: per-segment sums against ``numpy.add.reduceat``, with empty segments set to zero.

    ``reduceat`` repeats the value at an empty segment's start where ``segment_sum`` returns zero,
    so the transform is to zero the reference at the empty segments (asserted present). Segment
    lengths run from 0 to 2 000, so a long segment's warps share one slot.
    """
    rng = np.random.default_rng(17)
    lengths_np = rng.integers(0, 40, 300)
    lengths_np[::25] = 0
    lengths_np[5] = 2_000
    offsets_np = np.concatenate([[0], np.cumsum(lengths_np)]).astype(np.int32)
    n = int(offsets_np[-1])
    values_np = _group_values(n, width, dtype_np, seed=18)
    expected_np = np.add.reduceat(values_np.astype(np.float64), offsets_np[:-1].clip(max=n - 1))
    expected_np[lengths_np == 0] = 0.0
    assert np.count_nonzero(lengths_np == 0) >= 12
    values_wp = wp.array(values_np, dtype=dtype_wp, device=device)
    offsets_wp = wp.array(offsets_np, dtype=wp.int32, device=device)
    sums_wp = od_reduce.segment_sum(values_wp, offsets_wp)
    assert sums_wp.shape == (lengths_np.size,)
    assert np.allclose(sums_wp.numpy(), expected_np, rtol=1e-5, atol=1e-4)
    if deterministic:
        assert np.array_equal(od_reduce.segment_sum(values_wp, offsets_wp).numpy(), sums_wp.numpy())


_REJECTED_GROUP_SUMS: list[tuple[str, Callable[[str], object], type, str]] = [
    (
        "rank-2 values",
        lambda device: od_reduce.scatter_sum(
            warp_empty((3, 2), wp.float32, device), warp_empty(3, wp.int32, device)
        ),
        ValueError,
        "rank-1 values",
    ),
    (
        "length mismatch",
        lambda device: od_reduce.scatter_sum(
            wp.zeros(3, dtype=wp.float32, device=device), wp.zeros(4, dtype=wp.int32, device=device)
        ),
        ValueError,
        "one index per value",
    ),
    (
        "index past n_groups",
        lambda device: od_reduce.scatter_sum(
            wp.zeros(2, dtype=wp.float32, device=device),
            wp.array([0, 2], dtype=wp.int32, device=device),
            2,
        ),
        ValueError,
        "outside",
    ),
    (
        "negative index",
        lambda device: od_reduce.scatter_sum(
            wp.zeros(2, dtype=wp.float32, device=device),
            wp.array([-1, 0], dtype=wp.int32, device=device),
        ),
        ValueError,
        "outside",
    ),
    (
        "negative n_groups",
        lambda device: od_reduce.scatter_sum(
            wp.zeros(0, dtype=wp.float32, device=device), warp_empty(0, wp.int32, device), -1
        ),
        ValueError,
        "n_groups >= 0",
    ),
    (
        "integer values",
        lambda device: od_reduce.scatter_sum(
            wp.zeros(2, dtype=wp.int32, device=device), wp.zeros(2, dtype=wp.int32, device=device)
        ),
        TypeError,
        "sums wp.float32",
    ),
    (
        "int64 indices",
        lambda device: od_reduce.scatter_sum(
            wp.zeros(2, dtype=wp.float32, device=device), wp.zeros(2, dtype=wp.int64, device=device)
        ),
        TypeError,
        "int32 indices",
    ),
    (
        "rank-2 indices",
        lambda device: od_reduce.scatter_sum(
            wp.zeros(4, dtype=wp.float32, device=device),
            wp.zeros((2, 2), dtype=wp.int32, device=device),
        ),
        ValueError,
        "rank-1 indices",
    ),
    (
        "int64 offsets",
        lambda device: od_reduce.segment_sum(
            wp.zeros(3, dtype=wp.float32, device=device),
            wp.array([0, 3], dtype=wp.int64, device=device),
        ),
        TypeError,
        "int32 offsets, got int64",
    ),
    (
        "offsets not closed by n",
        lambda device: od_reduce.segment_sum(
            wp.zeros(3, dtype=wp.float32, device=device),
            wp.array([0, 2], dtype=wp.int32, device=device),
        ),
        ValueError,
        "not at the 3 values",
    ),
    (
        "empty offsets",
        lambda device: od_reduce.segment_sum(
            wp.zeros(0, dtype=wp.float32, device=device), warp_empty(0, wp.int32, device)
        ),
        ValueError,
        "non-empty rank-1 offsets",
    ),
]


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        pytest.param(call, error, match, id=name)
        for name, call, error, match in _REJECTED_GROUP_SUMS
    ],
)
def test_invalid_group_sums_raise(
    device: str, call: Callable[[str], object], error: type[Exception], match: str
) -> None:
    """Each rejected argument of ``scatter_sum`` / ``segment_sum`` raises its documented error."""
    with pytest.raises(error, match=match):
        call(device)


def test_group_sums_of_nothing(device: str) -> None:
    """Not a library comparison: no values give zero groups, or zero sums in the groups asked."""
    empty_values = wp.zeros(0, dtype=wp.float32, device=device)
    empty_indices = warp_empty(0, wp.int32, device)
    assert od_reduce.scatter_sum(empty_values, empty_indices).shape == (0,)
    assert od_reduce.scatter_sum(empty_values, empty_indices, 3).numpy().tolist() == [0.0] * 3
    offsets = wp.zeros(3, dtype=wp.int32, device=device)
    assert od_reduce.segment_sum(empty_values, offsets).numpy().tolist() == [0.0, 0.0]


def test_float_sum_of_a_strided_view_is_reproducible(device: str) -> None:
    """
    Ordito against ordito: a non-contiguous rank-2 float sum repeats and equals the dense one.

    A column slice cannot flatten, and its rank-2 kernel commits in arrival order; under the
    setting it takes the fixed-order sum of a dense copy, so it equals the sum of the same values
    held densely, bit for bit, on every call. NumPy holds the value.
    """
    table_np = np.random.default_rng(19).random((300_000, 4), dtype=np.float32)
    table_wp = wp.array(table_np, dtype=wp.float32, device=device)
    view_wp = table_wp[:, 1:3]
    assert not view_wp.is_contiguous
    dense_wp = wp.array(np.ascontiguousarray(table_np[:, 1:3]), dtype=wp.float32, device=device)
    sums = {_SUM(view_wp) for _ in range(4)}
    assert sums == {_SUM(dense_wp)}
    assert np.isclose(sums.pop(), table_np[:, 1:3].astype(np.float64).sum(), rtol=1e-5)


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


@pytest.mark.parity("median", "pymeshlab", "numpy")
def test_scalar_statistics_match_pymeshlab(device: str) -> None:
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
    n = 100
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
