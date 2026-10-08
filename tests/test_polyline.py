from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest
import shapely.geometry as sg
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm
from trimesh.path import traversal as tm_traversal

import ordito as od
from ordito.kernels import polyline as kernel_polyline
from tests.comparisons import hausdorff_two_sided, lexsort_rows
from tests.conversions import (
    meshlib_to_trimesh,
    points_to_warp,
    points_to_warp_uv,
    polyline_to_pyvista,
    warp_empty,
)

if TYPE_CHECKING:
    import pyvista as pv


def _random_open_polyline(seed: int, n: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3))


def _helix() -> np.ndarray:
    """Build a 128-point, two-turn helix of unit radius rising ``4 pi / 6``."""
    steps_np = np.linspace(0.0, 4.0 * np.pi, 128)
    return np.stack([np.cos(steps_np), np.sin(steps_np), steps_np / 6.0], axis=1)


def _polyline_ml(pts_np: np.ndarray) -> mm.Polyline3:
    """
    Wrap a NumPy polyline in a ``meshlib.Polyline3``.

    Through the **constructor**, not ``addFromPoints``: that method's bound signature takes a raw
    ``Vector3f*`` plus a count rather than a vector, so a natural call raises. The constructor's
    single-contour overload is the usable route, and it produces ``n`` points with ``n - 1`` edges.
    """
    return mm.Polyline3(_contour_ml(pts_np))


def _contour_ml(pts_np: np.ndarray) -> mm.std_vector_Vector3_float:
    """Build the same points as the bare contour ``calcLength`` takes."""
    contour_ml = mm.std_vector_Vector3_float()
    for point_np in np.asarray(pts_np, dtype=np.float64):
        contour_ml.append(mm.Vector3f(*point_np.tolist()))
    return contour_ml


def _ordered_ml(polyline_ml: mm.Polyline3) -> np.ndarray:
    """
    Read a ``Polyline3`` back in **path** order, through ``contours()``.

    Not through ``points``, and this is a silent wrong answer rather than an inconvenience:
    ``points`` is the *storage* buffer, so after a subdivision or a decimation its consecutive
    entries are no longer consecutive along the curve. Measured on a 128-point helix subdivided to
    407 points, the largest gap between adjacent ``points`` entries is **2.548** against a curve
    whose longest segment is 0.050 -- a spacing assertion built on it reads 50x too large and looks
    like the reference failing.

    ``contours()`` walks the topology and needs no ``pack()``; a ``points`` read would need one as
    well (``vertsDeleted`` 104 with ``points.size()`` still 128 and ``numValidVerts`` 24).
    """
    return np.asarray([[point.x, point.y, point.z] for point in polyline_ml.contours()[0]])


def _max_deviation_np(points_np: np.ndarray, polyline_np: np.ndarray) -> float:
    """Largest distance from any of ``points_np`` to the polyline through ``polyline_np``."""
    starts, ends = polyline_np[:-1], polyline_np[1:]
    spans = ends - starts
    fractions = np.clip(
        ((points_np[:, None, :] - starts) * spans).sum(-1)
        / np.maximum((spans * spans).sum(-1), 1e-30),
        0.0,
        1.0,
    )
    closest = starts + fractions[..., None] * spans
    return float(np.linalg.norm(points_np[:, None, :] - closest, axis=-1).min(axis=1).max())


def _closed_from(pts_np: np.ndarray) -> np.ndarray:
    return np.concatenate([pts_np, pts_np[:1]], axis=0)


# --- NumPy reference implementations (mirroring the source PyTorch logic) ---


def _centroid_np(pts: np.ndarray) -> np.ndarray:
    segments = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(segments, axis=-1, keepdims=True)
    midpoints = pts[:-1] + segments / 2
    return (midpoints * seg_len).sum(axis=0) / seg_len.sum()


def _closed_normal_np(pts: np.ndarray) -> np.ndarray:
    # Closed Newell's method: consecutive-vertex cross products including the wrap-around edge.
    normal = np.cross(pts, np.roll(pts, -1, axis=0)).sum(axis=0)
    return normal / np.linalg.norm(normal)


def _angles_np(pts: np.ndarray) -> np.ndarray:
    segments = np.diff(pts, axis=0)
    rolled = np.roll(segments, -1, axis=0)
    cos = np.sum(segments * rolled, axis=-1) / (
        np.linalg.norm(segments, axis=-1) * np.linalg.norm(rolled, axis=-1)
    )
    angles = np.arccos(np.clip(cos, -1.0, 1.0))
    if np.allclose(pts[0], pts[-1]):
        # ``angles[i]`` is the angle between segment ``i`` and its cyclic successor, which is the
        # turning angle at vertex ``i + 1`` (mod n_segments), not at vertex ``i`` -- roll it back
        # by one so it is indexed by vertex, matching ``polyline_angles``' own per-vertex output.
        rotated = np.roll(angles, 1)
        return np.concatenate([rotated, rotated[:1]])
    angles[-1] = 0.0
    return np.concatenate([angles[-1:], angles])


def _upsample_np(pts: np.ndarray, step: float) -> np.ndarray:
    segments = np.diff(pts, axis=0)
    steps = np.clip(np.linalg.norm(segments, axis=-1) // step, 1, None).astype(np.int64)
    cumulative = np.roll(np.cumsum(steps), 1)
    total = int(cumulative[0])
    cumulative[0] = 0
    indices = np.repeat(np.arange(len(steps)), steps)
    weights = (np.arange(total) - cumulative[indices]) / steps[indices]
    return pts[indices] + weights[:, None] * segments[indices]


def _endpoint_normals_np(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n1 = np.cross(b, a + c)
    n2 = np.cross(b, a - c)
    plane = n1 if n1 @ n1 >= n2 @ n2 else n2
    if plane @ plane < 1e-6:
        return np.zeros(3), np.zeros(3)
    unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
    nod = unit(np.cross(plane, b))
    return unit(nod + unit(np.cross(plane, a))), unit(nod + unit(np.cross(plane, c)))


def _arc_point_np(
    po: np.ndarray, pd: np.ndarray, no: np.ndarray, nd: np.ndarray, t: float
) -> np.ndarray:
    b = pd - po
    chord = np.linalg.norm(b)
    linear = po + t * b
    if chord < 1e-6:
        return po
    if no @ no < 0.5 or nd @ nd < 0.5:
        return linear
    theta = abs(np.arccos(np.clip(no @ nd, -1.0, 1.0)))
    if theta < 1e-6:
        return linear
    tangent = b / chord
    sign = 1.0 if b @ (nd - no) >= 0.0 else -1.0
    bulge = sign * (no + nd)
    m = bulge - (bulge @ tangent) * tangent
    if m @ m < 1e-6:
        return linear
    m = m / np.linalg.norm(m)
    alpha = 0.5 * theta
    radius = chord / (2.0 * np.sin(alpha))
    center = 0.5 * (po + pd) - radius * np.cos(alpha) * m
    phi = (2.0 * t - 1.0) * alpha
    return center + radius * (np.cos(phi) * m + np.sin(phi) * tangent)


def _smooth_upsample_np(pts: np.ndarray, step: float, closed: bool) -> np.ndarray:
    n = len(pts)
    m = n - 1  # distinct vertices when closed (pts[-1] == pts[0])
    segments = np.diff(pts, axis=0)
    steps = np.clip(np.linalg.norm(segments, axis=-1) // step, 1, None).astype(np.int64)
    out = []
    for s in range(n - 1):
        po, pd = pts[s], pts[s + 1]
        no = nd = np.zeros(3)
        if closed:
            a, c = po - pts[(s - 1 + m) % m], pts[(s + 2) % m] - pd
            no, nd = _endpoint_normals_np(a, segments[s], c)
            interior = True
        elif s >= 1 and s + 2 <= n - 1:
            a, c = po - pts[s - 1], pts[s + 2] - pd
            no, nd = _endpoint_normals_np(a, segments[s], c)
            interior = True
        else:
            interior = False
        for k in range(int(steps[s])):
            t = k / steps[s]
            out.append(_arc_point_np(po, pd, no, nd, t) if interior else po + t * (pd - po))
    return np.array(out)


def _downsample_np(pts: np.ndarray, step: float) -> np.ndarray:
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    keep = [0]
    last = 0.0
    for i in range(1, len(pts)):
        if cumulative[i] - last >= step:
            keep.append(i)
            last = cumulative[i]
    return pts[keep]


def _resample_np(pts: np.ndarray, num_points: int) -> np.ndarray:
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    arc = np.linspace(0.0, cumulative[-1], num_points)
    return np.stack([np.interp(arc, cumulative, pts[:, k]) for k in range(3)], axis=1)


def _distance_np(points: np.ndarray, pts: np.ndarray) -> np.ndarray:
    starts = pts[:-1][None]
    segments = np.diff(pts, axis=0)[None]
    seg_sq = (segments**2).sum(axis=-1, keepdims=True).clip(1e-8)
    vectors = points[:, None] - starts
    t = np.clip((vectors * segments).sum(axis=-1, keepdims=True) / seg_sq, 0.0, 1.0)
    closest = starts + t * segments
    return np.linalg.norm(points[:, None] - closest, axis=-1).min(axis=1)


def _radius_np(
    pts: np.ndarray,
    reduction: str,
    center: np.ndarray | None = None,
    normal: np.ndarray | None = None,
) -> float:
    if center is None:
        center = _centroid_np(pts)
    if normal is None:
        normal = _closed_normal_np(pts)
    else:
        normal = normal / np.linalg.norm(normal)
    projected = pts - normal * np.sum((pts - center) * normal, axis=-1, keepdims=True)
    segments = np.diff(projected, axis=0)
    seg_sq = np.sum(segments**2, axis=-1).clip(1e-8)
    double_areas = -np.sum((projected[:-1] - center) * segments, axis=-1)
    t = np.clip(double_areas / seg_sq, 0.0, 1.0)
    closest = projected[:-1] + t[:, None] * segments
    distances = np.linalg.norm(closest - center, axis=-1)
    return float(getattr(np, reduction)(distances))


# --- open / close ---


@pytest.mark.parametrize(
    ("operation", "seed", "input_closed", "expected_closed"),
    [
        pytest.param("open", 0, True, False, id="open-drops-the-duplicate-endpoint"),
        pytest.param("open", 1, False, False, id="open-leaves-an-open-input"),
        pytest.param("close", 2, False, True, id="close-appends-the-first-point"),
        pytest.param("close", 3, True, True, id="close-leaves-a-closed-input"),
    ],
)
def test_open_and_close_polyline(
    device: str, operation: str, seed: int, input_closed: bool, expected_closed: bool
) -> None:
    """Not a library comparison: ``polyline_open`` / ``polyline_close`` against the NumPy form."""
    pts_np = _random_open_polyline(seed)
    input_np = _closed_from(pts_np) if input_closed else pts_np
    function = od.polyline.polyline_open if operation == "open" else od.polyline.polyline_close
    result_wp = function(points_to_warp(input_np, device))
    expected_np = (_closed_from(pts_np) if expected_closed else pts_np).astype(np.float32)
    assert np.allclose(result_wp.numpy(), expected_np, rtol=1e-5, atol=1e-5)


# --- length (meshlib / pyvista oracles) ---


@pytest.mark.parity("polyline_length", "meshlib")
def test_polyline_length_matches_meshlib(device: str) -> None:
    """
    Class A: ``calcLength`` sums the same segments over the same ``float32`` contour.

    Both open and closed forms, the closed one through a named transform on the *reference* side --
    append the first point, since MeshLib's ``calcLength`` takes a bare contour and has no closed
    flag. That is what ``closed=True`` is defined to mean, so the transform is the definition rather
    than an accommodation.

    **This compared ``==`` and now compares a relative tolerance, deliberately.** ``calcLength``
    accumulates left to right and ``polyline_length`` folds a ``wp.tile_sum`` tree over
    lane-strided partials, so the two agree only up to ``float32`` summation order -- measured
    ~1e-7 relative on this fixture, which is one or two ulps of a 50-segment sum and *below* the
    error either accumulation carries against the exact answer. The tree is in fact the more
    accurate of the two, since it halves the depth over which rounding compounds. The bar here is
    therefore ``rtol=1e-6``: tight enough that a wrong segment, a dropped one or a doubled one
    fails it by orders of magnitude, and loose enough not to pin an implementation detail of how
    the partial sums are ordered.

    ``calcLength`` is an overload set over 2-D and 3-D, float and double contours; the ``Vector3f``
    one is what a ``float32`` polyline maps onto, and picking the ``double`` overload instead would
    silently compare a different accumulation.
    """
    pts_np = _random_open_polyline(2, n=50)

    length_wp = od.polyline.polyline_length(points_to_warp(pts_np, device))
    length_ml = mm.calcLength(_contour_ml(pts_np))
    assert length_ml > 0.0  # non-vacuity
    assert length_wp == pytest.approx(length_ml, rel=1e-6)

    closed_wp = od.polyline.polyline_length(points_to_warp(pts_np, device), closed=True)
    closed_ml = mm.calcLength(_contour_ml(_closed_from(pts_np)))
    assert closed_ml > length_ml  # the closing segment is real
    assert closed_wp == pytest.approx(closed_ml, rel=1e-6)


@pytest.mark.parity("polyline_length", "pyvista")
def test_polyline_length_matches_pyvista(device: str) -> None:
    """
    Class A: VTK's ``compute_arc_length`` accumulates the same segments, to 7.35e-08 relative.

    The residual is ordito's ``float32`` buffer against pyvista's exact float64 point storage, and
    it is the same floor every pyvista row in the suite bottoms out at.

    **The input has to be one line cell.** ``polyline_to_pyvista`` builds it that way because
    ``pv.lines_from_points`` gives one two-point cell per segment and ``compute_arc_length``
    restarts at every one of them -- measured max **0.0638** against a true 12.7049 on a 200-point
    helix, which reads as a factor-of-200 disagreement rather than as the wrong input. The field is
    *cumulative* per point, so the length is its last/maximum entry; ``compute_cell_sizes``'
    ``Length`` sums to the identical value and either is admissible.

    The closed form goes through the same named transform the meshlib pairing uses -- append the
    first point, since VTK has no closed flag either.
    """
    pts_np = _random_open_polyline(2, n=50)

    length_wp = od.polyline.polyline_length(points_to_warp(pts_np, device))
    arc_pv = np.asarray(
        cast("pv.PolyData", polyline_to_pyvista(pts_np).compute_arc_length())["arc_length"]
    )
    length_pv = float(arc_pv.max())
    assert length_pv > 0.0  # non-vacuity
    assert np.allclose(length_wp, length_pv, rtol=1e-5, atol=1e-5)

    closed_wp = od.polyline.polyline_length(points_to_warp(pts_np, device), closed=True)
    closed_pv = float(
        np.asarray(
            cast("pv.PolyData", polyline_to_pyvista(_closed_from(pts_np)).compute_arc_length())[
                "arc_length"
            ]
        ).max()
    )
    assert closed_pv > length_pv  # the closing segment is real
    assert np.allclose(closed_wp, closed_pv, rtol=1e-5, atol=1e-5)


# --- centroid / normal (NumPy reference) ---


def test_polyline_centroid_matches_reference(device: str) -> None:
    pts_np = _random_open_polyline(20)
    centroid_wp = od.polyline.polyline_centroid(points_to_warp(pts_np, device))
    assert np.allclose(list(centroid_wp), _centroid_np(pts_np), rtol=1e-4, atol=1e-4)


def test_polyline_centroid_closed_matches_reference(device: str) -> None:
    """
    ``closed=True`` weights the seam segment like any other, which moves the centroid.

    Both halves matter. The first assert is the value, against the same NumPy reference fed the
    explicitly-closed points -- so ``closed=True`` and ``polyline_close`` agree. The second is that
    the two answers *differ*: this is a length-weighted mean, so adding a segment reweights every
    other one, and a collapse that quietly ignored the keyword would pass the first assert alone.
    """
    pts_np = _random_open_polyline(20)
    polyline_wp = points_to_warp(pts_np, device)

    centroid_wp = od.polyline.polyline_centroid(polyline_wp, closed=True)

    assert np.allclose(list(centroid_wp), _centroid_np(_closed_from(pts_np)), rtol=1e-4, atol=1e-4)
    assert not np.allclose(
        list(centroid_wp), list(od.polyline.polyline_centroid(polyline_wp)), atol=1e-3
    )


def test_polyline_normal_matches_reference(device: str) -> None:
    pts_np = _random_open_polyline(22)
    normal_wp = od.polyline.polyline_normal(points_to_warp(pts_np, device))
    # polyline_normal treats the polyline as a closed loop (Newell's method); the normal sign is
    # well-defined by the summed cross products, so compare directly.
    assert np.allclose(list(normal_wp), _closed_normal_np(pts_np), rtol=1e-4, atol=1e-4)


def test_polyline_normal_far_from_the_origin(device: str) -> None:
    """
    Not a library comparison: a 1e-2 loop at coordinates near 700 against its exact normal.

    Newell's sum is the same about any point of a closed ring only in exact arithmetic. Taken about
    the coordinate origin, in ``float32``, it subtracts cross products of size ``|p|^2 ~ 5e5`` to
    leave a loop area of ~3e-5, and the normal came out 118.8 degrees wrong; about the loop's first
    point (``predicates.newell_term``) it agrees with the float64 sum of the same float32 points
    to within a 1e-3 rad bound. The exact reference is that float64 sum about the same point.
    """
    rng = np.random.default_rng(1)
    angles = np.sort(rng.random(9)) * 2.0 * np.pi
    loop_np = np.column_stack(
        (np.cos(angles), 0.3 * np.sin(angles), 0.05 * np.sin(2.0 * angles))
    ) * 1e-2 + np.array([700.0, -400.0, 250.0])
    loop_np = loop_np.astype(np.float32).astype(np.float64)
    exact_np = np.cross(loop_np - loop_np[0], np.roll(loop_np, -1, axis=0) - loop_np[0]).sum(axis=0)
    exact_np /= np.linalg.norm(exact_np)

    normal_np = np.array(list(od.polyline.polyline_normal(points_to_warp(loop_np, device))))
    assert np.arccos(np.clip(normal_np @ exact_np, -1.0, 1.0)) < 1e-3


# --- angles (NumPy reference; trimesh vector_angle cross-check) ---


@pytest.mark.parametrize(
    ("seed", "input_closed"),
    [pytest.param(30, False, id="open"), pytest.param(31, True, id="closed")],
)
def test_polyline_angles_matches_reference(device: str, seed: int, input_closed: bool) -> None:
    """Not a library comparison: turning angles against ``_angles_np``, open and closed input."""
    pts_np = _random_open_polyline(seed)
    if input_closed:
        pts_np = _closed_from(pts_np)
    angles_wp = od.polyline.polyline_angles(points_to_warp(pts_np, device))
    assert np.allclose(angles_wp.numpy(), _angles_np(pts_np), rtol=1e-4, atol=1e-4)


def test_polyline_angles_closed_is_indexed_by_vertex_not_by_segment(device: str) -> None:
    """
    A hand-computed right triangle pins the per-vertex indexing, independent of ``_angles_np``.

    The turning angle at vertex ``i`` is the angle between segment ``i - 1`` and segment ``i``,
    and the angle between segment ``i`` and its cyclic successor belongs to vertex ``i + 1`` -- an
    off-by-one rotation that a fuzz comparison against a reference sharing the same convention
    would not catch. ``closed=True`` returns one angle per input vertex, so the comparison against
    three expected values also pins the length. The triangle's three turning angles (90 degrees at
    the right-angle corner, 135 degrees at the other two, i.e. 180 minus each interior angle) are
    distinct enough that a rotated result cannot pass by coincidence.
    """
    triangle_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    angles_wp = od.polyline.polyline_angles(points_to_warp(triangle_np, device), closed=True)
    expected = np.radians([90.0, 135.0, 135.0])
    assert np.allclose(angles_wp.numpy(), expected, rtol=1e-4, atol=1e-4)


# --- distance (NumPy reference) ---


def test_distance_to_polyline_matches_reference(device: str) -> None:
    rng = np.random.default_rng(40)
    pts_np = _random_open_polyline(40)
    points_np = rng.standard_normal((25, 3))
    points_wp = points_to_warp(points_np, device)
    distances_wp = od.polyline.polyline_point_distance(points_wp, points_to_warp(pts_np, device))
    assert np.allclose(distances_wp.numpy(), _distance_np(points_np, pts_np), rtol=1e-4, atol=1e-4)


@pytest.mark.parity("polyline_point_distance", "meshlib")
def test_distance_to_polyline_matches_meshlib(device: str) -> None:
    """
    Class A after one named transform: ``findProjectionOnPolyline`` reports a **squared** distance.

    The same convention open3d's k-NN and MeshLib's own ``findProjection`` use, so the square root
    is the whole of the transform; max difference **3.7e-07** over 50 queries against a 40-segment
    polyline, which is the float32 floor.

    It is a per-query call with no batched form, so this loops -- fine at test size, and the reason
    the benchmark row for this group stays with the vectorized references.
    """
    rng = np.random.default_rng(40)
    pts_np = _random_open_polyline(40, n=40)
    points_np = rng.standard_normal((50, 3)) * 3.0 + pts_np.mean(axis=0)

    distances_wp = od.polyline.polyline_point_distance(
        points_to_warp(points_np, device), points_to_warp(pts_np, device)
    )

    polyline_ml = _polyline_ml(pts_np)
    assert polyline_ml.points.size() == pts_np.shape[0]  # the constructor kept every point
    distances_ml = np.array(
        [
            np.sqrt(
                mm.findProjectionOnPolyline(mm.Vector3f(*point_np.tolist()), polyline_ml).distSq
            )
            for point_np in points_np
        ]
    )

    assert distances_ml.min() > 0.0  # non-vacuity: no query sits on the polyline
    assert np.allclose(distances_wp.numpy(), distances_ml, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("polyline_point_distance", "pyvista")
def test_distance_to_polyline_matches_pyvista(device: str) -> None:
    """
    Class A: ``find_closest_cell`` on a one-cell polyline is the point-to-*segment* distance.

    Max abs difference **2.49e-07** over 1 500 queries against a 200-point helix, median 3.72e-08,
    correlation 1.000000000 -- the float32 floor again. Unlike MeshLib's
    ``findProjectionOnPolyline`` this form is batched, which is why the benchmark row for this group
    can carry it.

    Two wrong routes, both measured. The distance must be recomputed from the returned closest
    *point*: ``find_closest_cell`` reports the cell, not the length. And
    ``compute_implicit_distance`` -- the exact SDF that serves ``signed_distance_on_mesh`` -- needs
    **polygons**: on a line set VTK logs ``No polygons to evaluate function!`` once per query and
    returns a field **3.35** away from the truth rather than raising.
    """
    rng = np.random.default_rng(40)
    pts_np = _random_open_polyline(40, n=40)
    points_np = rng.standard_normal((50, 3)) * 3.0 + pts_np.mean(axis=0)

    distances_wp = od.polyline.polyline_point_distance(
        points_to_warp(points_np, device), points_to_warp(pts_np, device)
    )

    line_pv = polyline_to_pyvista(pts_np)
    assert line_pv.n_cells == 1  # one cell, or every filter restarts per segment
    _cells_pv, closest_pv = cast(
        "tuple[np.ndarray, np.ndarray]",
        line_pv.find_closest_cell(points_np, return_closest_point=True),
    )
    distances_pv = np.linalg.norm(points_np - np.asarray(closest_pv), axis=1)

    assert distances_pv.min() > 0.0  # non-vacuity: no query sits on the polyline
    assert np.allclose(distances_wp.numpy(), distances_pv, rtol=1e-5, atol=1e-5)


def test_distance_to_single_point_polyline(device: str) -> None:
    rng = np.random.default_rng(41)
    points_np = rng.standard_normal((10, 3))
    target_np = rng.standard_normal((1, 3))
    distances_wp = od.polyline.polyline_point_distance(
        points_to_warp(points_np, device), points_to_warp(target_np, device)
    )
    expected = np.linalg.norm(points_np - target_np[0], axis=-1)
    assert np.allclose(distances_wp.numpy(), expected, rtol=1e-4, atol=1e-4)


def test_distance_to_empty_polyline_is_infinite(device: str) -> None:
    """An empty curve is infinitely far, rather than whatever the allocator last held."""
    rng = np.random.default_rng(42)
    points_wp = points_to_warp(rng.standard_normal((5, 3)), device)
    empty_wp = warp_empty(0, wp.vec3, device)

    distances_wp = od.polyline.polyline_point_distance(points_wp, empty_wp)

    assert np.array_equal(distances_wp.numpy(), np.full(5, np.inf, dtype=np.float32))


@pytest.mark.parametrize("closed", [False, True])
def test_polyline_point_distance_slices_match_one_thread_per_query(
    device: str, closed: bool
) -> None:
    """
    The segment-sliced search answers what one thread walking every segment does, bit for bit.

    Ordito against ordito: the parity tests above carry the oracle, this pins the two schedules
    ``polyline_point_distance`` chooses between. Fifty queries against a 3 000-point walk slice the
    segments on CUDA (and not on the CPU device, where both sides are the one-thread kernel); a
    minimum is order-free, so any split of the segments must give the same floats.
    """
    rng = np.random.default_rng(43)
    polyline_wp = points_to_warp(np.cumsum(rng.standard_normal((3000, 3)), axis=0), device)
    queries_wp = points_to_warp(rng.standard_normal((50, 3)) * 20.0, device)
    unsliced_wp = warp_empty(50, wp.float32, device)
    wp.launch(
        kernel_polyline.distance_to_segments,
        dim=50,
        inputs=[queries_wp, polyline_wp, wp.int32(closed), unsliced_wp],
        device=device,
    )
    distances_wp = od.polyline.polyline_point_distance(queries_wp, polyline_wp, closed=closed)
    assert np.array_equal(distances_wp.numpy(), unsliced_wp.numpy())


# --- upsample (NumPy reference) ---


@pytest.mark.parametrize("step", [0.3, 0.75])
def test_upsample_polyline_matches_reference(device: str, step: float) -> None:
    pts_np = _random_open_polyline(50)
    upsampled_wp = od.polyline.polyline_upsample(points_to_warp(pts_np, device), step)
    assert np.allclose(upsampled_wp.numpy(), _upsample_np(pts_np, step), rtol=1e-4, atol=1e-4)


@pytest.mark.parity("polyline_upsample", "meshlib")
def test_upsample_polyline_matches_meshlib(device: str) -> None:
    """
    Class C (no correspondence): the same curve resampled two ways, both of them *on* the curve.

    ``subdividePolyline`` takes the same ``maxEdgeLen``, and it is the only reference that resamples
    a 3-D polyline at all. What the two do not share is the rule: ordito splits each segment into
    ``max(floor(length / step), 1)`` **equal** pieces, so it lands near the step and can overshoot
    it by up to 2x; MeshLib **bisects** until every edge is under the cap, so it lands under the
    step and can undershoot by up to 2x. Measured on a 128-point helix (spacing 0.100273) at a step
    of 0.040109 -- ordito 254 points at spacing 0.050136, MeshLib 509 at 0.025068. Both are inside
    the factor of two, from opposite sides, which is what the count bound below asserts.

    The claim that does hold exactly is the one that matters: **every point either library emits
    lies on the input polyline**, so neither is smoothing. Measured 1.3e-07 and 2.2e-07, i.e. the
    float32 buffer's own noise.

    ``maxEdgeSplits`` is raised from its default of **1 000**, which would otherwise stop the
    reference in the first few percent of the work and make it look fast.

    **Bug class excluded:** a resampler that interpolates off the curve (a spline fit, a smoothing
    pass) or that leaves the spacing where it found it. **Mutation probe, measured:** feeding
    ``polyline_smooth_upsample``'s output to the deviation assert -- ordito's own curvature-aware
    variant, which is *meant* to leave the chord -- gives 1.1e-03, four orders past the 1e-05 bound.
    """
    pts_np = _helix()
    spacing = float(np.linalg.norm(np.diff(pts_np, axis=0), axis=1).mean())
    step = 0.4 * spacing

    dense_wp = od.polyline.polyline_upsample(points_to_warp(pts_np, device), step)
    dense_np = dense_wp.numpy().astype(np.float64)

    polyline_ml = _polyline_ml(pts_np)
    settings_ml = mm.PolylineSubdivideSettings()
    settings_ml.maxEdgeLen = step
    settings_ml.maxEdgeSplits = 10_000_000
    n_added_ml = mm.subdividePolyline(polyline_ml, settings_ml)
    dense_ml = _ordered_ml(polyline_ml)

    assert n_added_ml > 0  # non-vacuity: the reference really subdivided
    for resampled_np in (dense_np, dense_ml):
        assert len(resampled_np) > pts_np.shape[0]
        # Every emitted point is on the input polyline: neither library smooths.
        assert _max_deviation_np(resampled_np, pts_np) < 1e-5
        # Both land within a factor of two of the step, from opposite sides.
        emitted = np.linalg.norm(np.diff(resampled_np, axis=0), axis=1)
        assert emitted.max() < 2.0 * step
        assert emitted.mean() > 0.5 * step

    # Mutation probe for the deviation bound: the curvature-aware variant leaves the chord.
    smooth_np = od.polyline.polyline_smooth_upsample(points_to_warp(pts_np, device), step).numpy()
    assert _max_deviation_np(smooth_np.astype(np.float64), pts_np) > 1e-4


# --- smooth upsample (curvature-aware, NumPy reference + circle oracle) ---


@pytest.mark.parametrize("step", [0.3, 0.75])
def test_smooth_upsample_matches_reference(device: str, step: float) -> None:
    pts_np = _random_open_polyline(50)
    smoothed_wp = od.polyline.polyline_smooth_upsample(points_to_warp(pts_np, device), step)
    assert np.allclose(
        smoothed_wp.numpy(), _smooth_upsample_np(pts_np, step, closed=False), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("step", [0.3, 0.75])
def test_smooth_upsample_closed_matches_reference(device: str, step: float) -> None:
    pts_np = _closed_from(_random_open_polyline(52))
    smoothed_wp = od.polyline.polyline_smooth_upsample(
        points_to_warp(pts_np, device), step, closed=True
    )
    assert np.allclose(
        smoothed_wp.numpy(), _smooth_upsample_np(pts_np, step, closed=True), rtol=1e-4, atol=1e-4
    )


def test_smooth_upsample_straight_line_reduces_to_linear(device: str) -> None:
    # Collinear vertices have zero curvature everywhere, so the arc collapses onto the chord and
    # the result must coincide with plain linear upsampling.
    pts_np = np.stack([np.linspace(0.0, 3.0, 7), np.zeros(7), np.zeros(7)], axis=1)
    step = 0.25
    linear = od.polyline.polyline_upsample(points_to_warp(pts_np, device), step).numpy()
    smoothed = od.polyline.polyline_smooth_upsample(points_to_warp(pts_np, device), step).numpy()
    assert np.allclose(smoothed, linear, rtol=1e-5, atol=1e-5)


def test_smooth_upsample_preserves_original_vertices(device: str) -> None:
    # Each segment's first sample (t == 0) is its start vertex, so all but the final vertex survive.
    pts_np = _random_open_polyline(54, n=6)
    smoothed = od.polyline.polyline_smooth_upsample(points_to_warp(pts_np, device), 0.5).numpy()
    for vertex in pts_np[:-1]:
        assert np.any(np.all(np.isclose(smoothed, vertex, rtol=1e-4, atol=1e-4), axis=1))


@pytest.mark.parametrize("radius", [2.0, 3e-2, 1e-4, 1e-5])
def test_smooth_upsample_closed_recovers_circle_at_any_scale(device: str, radius: float) -> None:
    """
    The circumscribed-arc fit must not silently fall back to the straight chord at small scale.

    ``endpoint_normals``' collinearity guard used to compare an absolute cross-product magnitude to
    a fixed epsilon, and that magnitude scales as the *4th power* of the coordinate scale, so an
    ordinary (non-collinear) octagon bend was reclassified as degenerate once its radius dropped
    below ~4e-2 -- the arc fit collapsed to the straight chord there even though nothing about the
    octagon's *shape* (an angle, which is scale-invariant) had changed. `radius=2.0` is the
    original, always-passing scale; the smaller ones reproduce the failure this guards against.

    A regular polygon inscribed in a circle: curvature fitting reconstructs the circumscribed arcs,
    so every inserted point lies on the circle, where the straight-chord upsample cuts inside it.
    """
    polygon_np = _planar_circle(8, radius)
    smoothed = od.polyline.polyline_smooth_upsample(
        points_to_warp(polygon_np, device), 0.35 * radius, closed=True
    ).numpy()
    assert np.allclose(np.linalg.norm(smoothed, axis=-1), radius, rtol=1e-3, atol=1e-3 * radius)
    # The added points genuinely bulge outward relative to the straight-chord upsample.
    linear = od.polyline.polyline_upsample(
        points_to_warp(polygon_np, device), 0.35 * radius, closed=True
    ).numpy()
    assert np.linalg.norm(linear, axis=-1).min() < radius * (1.0 - 5e-3)


def test_smooth_upsample_detects_an_already_closed_input(device: str) -> None:
    """
    An explicitly closed input is smoothed at the seam even without passing ``closed=True``.

    Without the auto-detection the two segments touching the seam fall back to linear
    interpolation instead of the fitted arc every other segment gets, which is visible as the same
    radial deviation a plain ``polyline_upsample`` would leave.
    """
    radius = 2.0
    ring_np = _planar_circle(8, radius)
    closed_np = np.concatenate([ring_np, ring_np[:1]], axis=0)
    smoothed = od.polyline.polyline_smooth_upsample(points_to_warp(closed_np, device), 0.35).numpy()
    assert np.allclose(np.linalg.norm(smoothed[:, :2], axis=-1), radius, rtol=1e-3, atol=1e-3)


def test_smooth_upsample_short_polyline_unchanged(device: str) -> None:
    single = points_to_warp(np.array([[1.0, 2.0, 3.0]]), device)
    assert od.polyline.polyline_smooth_upsample(single, 0.5).size == 1


# --- cumulative_arc_length (NumPy reference) ---


def test_cumulative_arc_length_matches_reference(device: str) -> None:
    pts_np = _random_open_polyline(40)
    cumulative_wp = od.polyline.cumulative_arc_length(points_to_warp(pts_np, device))
    segment_lengths_np = np.linalg.norm(np.diff(pts_np, axis=0), axis=1)
    expected_np = np.concatenate([[0.0], np.cumsum(segment_lengths_np)])
    assert np.allclose(cumulative_wp.numpy(), expected_np, rtol=1e-5, atol=1e-5)


def test_cumulative_arc_length_last_entry_is_total_length(device: str) -> None:
    pts_np = _random_open_polyline(25)
    polyline_wp = points_to_warp(pts_np, device)
    cumulative_wp = od.polyline.cumulative_arc_length(polyline_wp)
    assert np.isclose(
        float(cumulative_wp.numpy()[-1]), od.polyline.polyline_length(polyline_wp), rtol=1e-5
    )


# --- downsample (NumPy reference) ---


@pytest.mark.parametrize("step", [0.5, 1.5])
def test_downsample_polyline_matches_reference(device: str, step: float) -> None:
    pts_np = _random_open_polyline(60)
    downsampled_wp = od.polyline.polyline_downsample(points_to_warp(pts_np, device), step)
    assert np.allclose(downsampled_wp.numpy(), _downsample_np(pts_np, step), rtol=1e-4, atol=1e-4)


def test_downsample_polyline_branches_agree(device: str) -> None:
    """
    The pointer-doubling greedy selection is the serial walk's, exactly.

    **Ordito against ordito**, not a parity assert: the oracle is the *serial* branch, which
    ``test_downsample_polyline_matches_reference`` pins to a NumPy transcription of the same greedy
    rule and ``test_downsample_polyline_matches_meshlib`` compares against ``decimatePolyline``.
    This test carries no oracle of its own; it exists because
    ``polyline_downsample`` picks between the two branches on the point count *and the device*
    (``_DOWNSAMPLE_DOUBLING_FROM``, CUDA only), so on any single run one of them is never exercised
    by the tests above.

    The claim is **exact equality of the masks**, not a tolerance: the successor search evaluates
    the same ``cumulative[j] - cumulative[i] >= step`` comparison the walk does, on the same
    device-computed ``cumulative`` array, so nothing rounds differently. A tolerance here would hide
    precisely the boundary disagreement the shared spelling exists to prevent.

    Four spacings, because the interesting inputs are the ones at the comparison's boundary:
    uniform-random; heavily clustered (a sixth power, so most segments are far below the step and a
    few are far above); duplicated points (runs of zero-length segments, where several points sit at
    the same arc length and the two forms must pick the same one); and an **exact** grid, where
    ``cumulative[j] - cumulative[i]`` equals the step to the last bit and the two forms must agree
    on whether that counts.

    That last one is what makes this test bite. **Mutation probe:** loosening the successor search's
    ``>=`` to ``>`` leaves every other spacing passing and fails on ``exact`` -- so without it the
    test would not distinguish the predicate the whole shared-spelling argument rests on.
    """
    rng = np.random.default_rng(4)
    for spacing in ("uniform", "clustered", "duplicates", "exact"):
        n = 3000
        if spacing == "uniform":
            steps_np = rng.random(n - 1) * 0.01 + 1e-4
        elif spacing == "clustered":
            steps_np = rng.random(n - 1) ** 6 * 0.05
        elif spacing == "duplicates":
            steps_np = np.zeros(n - 1)
            steps_np[::3] = 0.01
        else:
            # 0.25 and 1.0 are exact in float32, so cumulative[i + 4] - cumulative[i] == step.
            steps_np = np.full(n - 1, 0.25)
        cumulative_np = np.concatenate([[0.0], np.cumsum(steps_np)]).astype(np.float32)
        cumulative_wp = wp.array(cumulative_np, dtype=wp.float32, device=device)
        step = 1.0 if spacing == "exact" else float(4.0 * steps_np.mean())

        # An open table: the polyline is read only for a ``closed=True`` seam, so none is passed.
        serial_wp = wp.zeros(n, dtype=wp.int32, device=device)
        wp.launch(
            kernel_polyline.greedy_downsample_mask,
            dim=1,
            inputs=[cumulative_wp, wp.float32(step), None, 0, serial_wp],
            device=device,
        )
        doubling_wp = wp.zeros(n, dtype=wp.int32, device=device)
        od.polyline._greedy_downsample_doubling(cumulative_wp, step, None, False, doubling_wp)  # pyright: ignore[reportPrivateUsage]

        assert int(serial_wp.numpy().sum()) > 1, f"{spacing}: the walk kept only the first point"
        assert np.array_equal(serial_wp.numpy(), doubling_wp.numpy()), spacing


def test_downsample_polyline_above_the_doubling_threshold(device: str) -> None:
    """
    ``polyline_downsample`` obeys its own spacing rule on an input past the branch point.

    Not a library comparison: no reference downsamples to a *step*
    (``decimatePolyline`` reaches a count, which is why
    ``test_downsample_polyline_matches_meshlib`` has to convert one into the other), and at this
    size a NumPy oracle would be comparing a ``float64`` sequential ``cumsum`` against Warp's
    ``float32`` tree scan -- their difference over 8 192 segments is ~2e-4, which is the same order
    as the gaps between decisions, so it would flag rounding as disagreement. The exactness claim
    lives in ``test_downsample_polyline_branches_agree`` instead, against the branch that *is*
    pinned to an oracle.

    What the invariants exclude: a selection that keeps points closer together than the step (the
    rule itself), that reorders or moves them (subsequence and membership), or that drops the ends
    of the polyline it should start from.
    """
    n = 2 * od.polyline._DOWNSAMPLE_DOUBLING_FROM  # pyright: ignore[reportPrivateUsage]
    rng = np.random.default_rng(5)
    points_np = np.cumsum(rng.standard_normal((n, 3)) * 0.01, axis=0)
    step = 0.05
    sparse_wp = od.polyline.polyline_downsample(points_to_warp(points_np, device), step)

    kept_np = sparse_wp.numpy()
    assert 2 < len(kept_np) < n, "vacuous: the selection kept everything or nothing"
    assert np.allclose(kept_np[0], points_np[0], rtol=1e-5, atol=1e-5)
    # Every kept point is an input point, in input order.
    matches = np.array([int(np.argmin(np.linalg.norm(points_np - p, axis=1))) for p in kept_np])
    assert np.all(np.diff(matches) > 0)
    assert np.allclose(kept_np, points_np[matches], rtol=1e-5, atol=1e-5)
    # Consecutive kept points are at least a step apart in arc length.
    arc_np = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points_np, axis=0), axis=1))])
    assert np.all(np.diff(arc_np[matches]) >= step - 1e-4)


@pytest.mark.parity("polyline_downsample", "meshlib")
def test_downsample_polyline_matches_meshlib(device: str) -> None:
    """
    Class C (a derived scalar): the same number of points removed, both answers on the curve.

    ``decimatePolyline`` reaches a *count* where ``polyline_downsample`` reaches a *step*, so the
    named transform is to hand it ordito's own deletion count as ``maxDeletedVertices`` and open
    ``maxError`` up so the count is what binds. Without that they would stop for different reasons
    and the comparison would be two unrelated reductions.

    **Two of its defaults have to be turned off, and neither is a tuning choice.**
    ``optimizeVertexPos`` defaults *on* and moves each surviving vertex to a fitted position, which
    would put the output off the input point set. ``touchBdVertices`` defaults *on* and collapses
    the endpoints away: measured on this fixture it returns (0.99511, 0.09879, 0.01649) where the
    input starts at (1, 0, 0), so a first-point assertion fails by a whole segment for a reason that
    is a parameter rather than a disagreement. With both off the endpoints come back exactly.

    With those set, both keep a subset of the input points at a matched count -- 56 of 128 at twice
    the mean spacing, 30 of 128 at four times -- and both stay on the original curve. The point
    *sets* differ, because a collapse queue and an arc-length walk choose differently, which is why
    this is a scalar comparison rather than an equality.

    One convention is left as a measured divergence rather than asserted: ordito keeps the
    **first** point by contract and the last only when the walk lands on it (at four times the
    spacing its last kept point is one short of the end), where MeshLib at ``touchBdVertices=False``
    keeps both.

    **Bug class excluded:** a downsampler that moves the points it keeps, or that loses the start of
    the curve. Both are asserted on both answers.
    """
    pts_np = _helix()
    spacing = float(np.linalg.norm(np.diff(pts_np, axis=0), axis=1).mean())

    for factor, n_bound in ((2.0, 60), (4.0, 34)):
        sparse_wp = od.polyline.polyline_downsample(
            points_to_warp(pts_np, device), factor * spacing
        )
        sparse_np = sparse_wp.numpy().astype(np.float64)
        n_expected = len(sparse_np)
        # Non-vacuity: the reduction is real, and close enough to its target to be the right one.
        assert n_expected < n_bound
        assert n_expected > n_bound // 2

        polyline_ml = _polyline_ml(pts_np)
        settings_ml = mm.DecimatePolylineSettings_Vector3f()
        settings_ml.maxDeletedVertices = pts_np.shape[0] - n_expected
        settings_ml.maxError = 1e30  # let the count bind, not the collapse cost
        settings_ml.optimizeVertexPos = False  # defaults on, and would move the kept points
        settings_ml.touchBdVertices = False  # defaults on, and would collapse the endpoints
        result_ml = mm.decimatePolyline(polyline_ml, settings_ml)
        sparse_ml = _ordered_ml(polyline_ml)

        assert int(result_ml.vertsDeleted) == pts_np.shape[0] - n_expected
        assert sparse_ml.shape[0] == n_expected
        for reduced_np in (sparse_np, sparse_ml):
            # A kept point is an input point: neither library moved what it kept.
            assert _max_deviation_np(reduced_np, pts_np) < 1e-5
            # The start of the curve survives, so this shortened rather than trimmed.
            assert np.allclose(reduced_np[0], pts_np[0], rtol=1e-5, atol=1e-5)
        # MeshLib additionally keeps the far endpoint; ordito's greedy walk need not land on it.
        assert np.allclose(sparse_ml[-1], pts_np[-1], rtol=1e-5, atol=1e-5)


# --- resample (NumPy interp reference + trimesh oracle) ---


@pytest.mark.parametrize("num_points", [5, 50])
def test_resample_polyline_matches_numpy_interp(device: str, num_points: int) -> None:
    pts_np = _random_open_polyline(70)
    resampled_wp = od.polyline.polyline_resample(points_to_warp(pts_np, device), num_points)
    assert np.allclose(resampled_wp.numpy(), _resample_np(pts_np, num_points), rtol=1e-4, atol=1e-4)


def test_resample_polyline_matches_trimesh(device: str) -> None:
    """
    Class A: arc-length resampling against ``trimesh.path.traversal.resample_path``.

    Tolerance is ``1e-3`` rather than ``1e-5`` because ordito interpolates in ``float32`` and
    trimesh in ``float64`` over a cumulative arc length, so the error grows along the path;
    [`test_resample_polyline_matches_numpy_interp`] is the tight version of the same claim against
    an oracle in matching precision.
    """
    pts_np = _random_open_polyline(71)
    num_points = 40
    resampled_tm = tm_traversal.resample_path(pts_np, count=num_points)
    resampled_wp = od.polyline.polyline_resample(points_to_warp(pts_np, device), num_points)
    assert np.allclose(resampled_wp.numpy(), resampled_tm, rtol=1e-3, atol=1e-3)


def test_resample_polyline_closed_shape_and_endpoints(device: str) -> None:
    pts_np = _random_open_polyline(72)
    num_points = 16
    resampled_wp = od.polyline.polyline_resample(
        points_to_warp(pts_np, device), num_points, closed=True
    )
    resampled = resampled_wp.numpy()
    assert resampled.shape == (num_points, 3)
    # first sample is the closed-polyline start; there is no duplicated closing point
    assert np.allclose(resampled[0], pts_np[0].astype(np.float32), rtol=1e-4, atol=1e-4)


def test_resample_single_point_repeats(device: str) -> None:
    point_np = np.array([[1.0, 2.0, 3.0]])
    resampled_wp = od.polyline.polyline_resample(points_to_warp(point_np, device), 5)
    assert np.allclose(resampled_wp.numpy(), np.repeat(point_np, 5, axis=0), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("closed", [False, True])
def test_resample_empty_polyline_is_returned_unchanged(device: str, closed: bool) -> None:
    """
    An empty polyline comes back empty, ``closed`` or not, as the docstring states.

    Not a library comparison: an edge-case contract. ``closed=True`` used to slice ``num_points``
    samples out of the empty result and raise Warp's zero-length-slice ``RuntimeError``.
    """
    empty_wp = warp_empty(0, wp.vec3, device)
    assert od.polyline.polyline_resample(empty_wp, 5, closed=closed).shape == (0,)


# --- radius (NumPy reference) ---


def _planar_circle(n: int, radius: float) -> np.ndarray:
    """Return the regular ``n``-gon inscribed in a circle of ``radius``, CCW in the xy-plane."""
    angle = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros(n)], axis=1)


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("reduction", ["min", "max", "mean", "median"])
def test_polyline_radius_explicit_plane_matches_reference(
    device: str, reduction: Literal["min", "max", "mean", "median"], closed: bool
) -> None:
    """
    Not a library comparison: ``_radius_np`` on a sampled circle, open and ``closed=True``.

    ``closed=True`` adds the seam segment: on a *sampled circle* stored without its closing point
    the open form misses the arc between the last and first sample entirely, so the closed form is
    compared against the reference fed the explicitly closed points.
    """
    pts_np = _planar_circle(24, radius=2.0)
    center_np = np.zeros(3)
    normal_np = np.array([0.0, 0.0, 1.0])
    radius_wp = od.polyline.polyline_radius(
        points_to_warp(pts_np, device),
        reduction,
        wp.vec3(*center_np.tolist()),
        wp.vec3(*normal_np.tolist()),
        closed=closed,
    )
    radius_np = _radius_np(
        _closed_from(pts_np) if closed else pts_np, reduction, center_np, normal_np
    )
    assert np.allclose(radius_wp, radius_np, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduction", ["min", "max", "mean", "median"])
def test_polyline_radius_default_plane_matches_reference(
    device: str, reduction: Literal["min", "max", "mean", "median"]
) -> None:
    pts_np = _random_open_polyline(80)
    radius_wp = od.polyline.polyline_radius(points_to_warp(pts_np, device), reduction)
    radius_np = _radius_np(pts_np, reduction)
    assert np.allclose(radius_wp, radius_np, rtol=1e-3, atol=1e-3)


def test_polyline_radius_closed_default_plane_differs_from_open(device: str) -> None:
    """The default ``center`` / ``normal`` come from the closed polyline, so the keyword reaches."""
    pts_np = _random_open_polyline(80)
    polyline_wp = points_to_warp(pts_np, device)
    assert not np.isclose(
        od.polyline.polyline_radius(polyline_wp, "mean", closed=True),
        od.polyline.polyline_radius(polyline_wp, "mean"),
        rtol=1e-3,
    )


@pytest.mark.parametrize("reduction", ["min", "max", "mean", "median"])
def test_polyline_radius_closed_on_a_closed_input_adds_no_segment(
    device: str, reduction: Literal["min", "max", "mean", "median"]
) -> None:
    """
    ``closed=True`` on a loop already ending on its first point is the open call, bit for bit.

    Ordito against ordito: ``test_polyline_radius_explicit_plane_matches_reference`` carries the
    oracle for the loop stored open. Past three points the fused reductions decide on the device
    whether the closing segment is already there, so this is the branch where a wrong decision
    would add a stand-in segment and move the reduction and the default plane.
    """
    pts_np = _planar_circle(24, radius=2.0) + np.array([0.0, 0.0, 0.3]) * np.arange(24)[:, None]
    polyline_wp = points_to_warp(np.concatenate([pts_np, pts_np[:1]]), device)
    assert od.polyline.polyline_radius(polyline_wp, reduction, closed=True) == (
        od.polyline.polyline_radius(polyline_wp, reduction)
    )


@pytest.mark.parametrize(
    ("points", "center", "normal", "expected"),
    [
        pytest.param(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            (0.5, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            id="two-points",
        ),
        pytest.param(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            None,
            (0.0, 0.0, 1.0),
            0.0,
            id="closed-three-points",
        ),
    ],
)
def test_polyline_radius_with_an_explicit_plane_needs_no_fit(
    device: str,
    points: list[list[float]],
    center: tuple[float, float, float] | None,
    normal: tuple[float, float, float],
    expected: float,
) -> None:
    """
    Not a library comparison: inputs too short to fit a plane measure once the plane is given.

    With both defaults supplied a two-point polyline needs no minimum beyond two. A three-point
    closed loop has no plane, but with an explicit normal the centroid is the segment's midpoint,
    which lies on it, so the ``max`` radius is zero.
    """
    radius = od.polyline.polyline_radius(
        points_to_warp(np.array(points), device),
        "max",
        center=None if center is None else wp.vec3(*center),
        normal=wp.vec3(*normal),
    )
    assert np.isclose(radius, expected, atol=1e-5)


# --- edge cases ---


@pytest.mark.parametrize(
    ("call", "match"),
    [
        pytest.param(
            lambda device: od.polyline.polyline_normal(
                points_to_warp(_random_open_polyline(0, n=2), device)
            ),
            "three points",
            id="normal-two-points",
        ),
        pytest.param(
            lambda device: od.polyline.polyline_radius(
                points_to_warp(_random_open_polyline(81), device),
                "sum",  # pyright: ignore[reportArgumentType]
            ),
            "unsupported reduction",
            id="radius-unknown-reduction",
        ),
        pytest.param(
            lambda device: od.polyline.polyline_radius(
                points_to_warp(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), device)
            ),
            "polyline_radius requires at least three points",
            id="radius-two-points-default-plane",
        ),
        pytest.param(
            lambda device: od.polyline.polyline_radius(
                points_to_warp(
                    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), device
                )
            ),
            "polyline_normal requires at least three points",
            id="radius-closed-three-points-default-plane",
        ),
    ],
)
def test_invalid_inputs_raise(device: str, call: Callable[[str], object], match: str) -> None:
    """
    Not a library comparison: each documented ``ValueError``, with its message.

    Two points pass ``polyline_radius``'s "at least two" floor but cannot derive a default plane on
    their own: ``polyline_normal`` needs three points to fit a plane and used to be reached
    unguarded whenever ``center`` or ``normal`` was left at its default, raising a message about a
    different function's minimum for an input this one's own docs call valid. A three-point loop
    whose last point repeats its first has no plane either, and the default normal rejects it.
    """
    with pytest.raises(ValueError, match=match):
        call(device)


def test_length_short_polyline_is_zero(device: str) -> None:
    single = points_to_warp(np.array([[0.0, 0.0, 0.0]]), device)
    assert od.polyline.polyline_length(single) == 0.0


def test_angles_short_polyline_is_zeros(device: str) -> None:
    single = points_to_warp(np.array([[0.0, 0.0, 0.0]]), device)
    angles = od.polyline.polyline_angles(single).numpy()
    assert np.array_equal(angles, np.zeros(1, dtype=np.float32))


# --- triangulate (ear clipping) ---


def _l_shape() -> np.ndarray:
    """Return a non-convex (one reflex corner) simple polygon in the xy-plane, CCW."""
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
    return np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)


def _star(points: int = 5, outer: float = 2.0, inner: float = 0.8) -> np.ndarray:
    """Return a star polygon (alternating reflex corners), CCW, in the xy-plane."""
    angle = np.linspace(0.0, 2 * np.pi, 2 * points, endpoint=False)
    radius = np.where(np.arange(2 * points) % 2 == 0, outer, inner)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros(2 * points)], axis=1)


def _rotate_into_3d(pts_xy: np.ndarray, seed: int) -> np.ndarray:
    """Apply a random rotation + translation so the polygon lies in a tilted 3D plane."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return pts_xy @ q.T + rng.standard_normal(3)


def _polygon_area(pts: np.ndarray) -> float:
    """Area of a planar polygon in 3D via the summed cross products (Newell)."""
    rolled = np.roll(pts, -1, axis=0)
    return float(np.linalg.norm(np.cross(pts, rolled).sum(axis=0)) / 2.0)


def _triangle_areas(pts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    return np.linalg.norm(np.cross(b - a, c - a), axis=-1) / 2.0


def _assert_valid_triangulation(pts: np.ndarray, faces: np.ndarray) -> None:
    n = pts.shape[0]
    assert faces.shape == (n - 2, 3)
    assert faces.min() >= 0
    assert faces.max() < n
    assert set(faces.ravel().tolist()) == set(range(n))  # no orphan vertices
    tri_areas = _triangle_areas(pts, faces)
    assert np.all(tri_areas > 1e-6)  # no degenerate triangles
    assert np.allclose(tri_areas.sum(), _polygon_area(pts), rtol=1e-5, atol=1e-5)


@pytest.mark.parity("polyline_triangulate", "meshlib")
def test_triangulate_polyline_matches_meshlib(device: str) -> None:
    """
    Class C (count and area): two ear-clippings of the same polygon, with different diagonals.

    ``triangulateContours`` takes **2-D** closed contours -- ``std_vector_Vector2_float`` with the
    first point repeated -- where ``polyline_triangulate`` takes an open 3-D loop, so the transform
    is dropping z and closing the ring. Omitting the repeat is the silent failure mode: an open
    square comes back as **one** triangle over three vertices rather than two over four.

    The diagonals are free, so the outputs differ face for face -- on the L-shape ordito returns
    ``[[4,5,0],[0,1,2],[0,2,3],[4,0,3]]`` and MeshLib ``[[3,5,0],[5,3,4],[1,3,0],[3,1,2]]`` -- and
    what a triangulation of a simple polygon must agree on is the *count* (``n - 2``) and the total
    area. Measured exact on both an L-shape (4 triangles, area 3.00000) and a 10-vertex star
    (8 triangles, area 1.32252).

    Both shapes are **non-convex**, which is the point: a fan over vertex 0 triangulates any convex
    polygon correctly and would pass a convex comparison while producing triangles outside an
    L-shape. The area assert is what catches that, since a fan over a reflex vertex covers more or
    less than the polygon.
    """
    for name, polygon_np in (
        ("L", _l_shape()[:, :2]),
        ("star", _star(5, outer=1.0, inner=0.45)[:, :2]),
    ):
        points_np = np.column_stack([polygon_np, np.zeros(polygon_np.shape[0])])
        faces_wp = od.polyline.polyline_triangulate(points_to_warp(points_np, device)).numpy()

        contour_ml = mm.std_vector_Vector2_float()
        for point_np in np.vstack([polygon_np, polygon_np[:1]]):  # closed: the repeat is required
            contour_ml.append(mm.Vector2f(float(point_np[0]), float(point_np[1])))
        contours_ml = mm.std_vector_std_vector_Vector2_float()
        contours_ml.append(contour_ml)
        mesh_ml = meshlib_to_trimesh(mm.triangulateContours(contours_ml))

        assert mesh_ml.faces.shape[0] == polygon_np.shape[0] - 2, name
        assert len(faces_wp) == mesh_ml.faces.shape[0], name
        area_wp = _triangle_areas(points_np, faces_wp).sum()
        area_ml = _triangle_areas(
            np.asarray(mesh_ml.vertices), np.asarray(mesh_ml.faces, dtype=np.int32)
        ).sum()
        assert np.isclose(area_wp, area_ml, rtol=1e-5), name
        assert np.isclose(area_wp, _polygon_area(points_np), rtol=1e-5), name


@pytest.mark.parity("polyline_triangulate", "pyvista")
def test_triangulate_polyline_matches_pyvista(device: str) -> None:
    """
    Class C (count and area): VTK's ``triangulate_contours`` ear-clips the same polygon.

    Same standard as the MeshLib pairing above and for the same reason -- the diagonals of a simple
    polygon's triangulation are free, so only ``n - 2`` and the total area are shared. Measured
    exact on the L-shape (4 triangles, 3.00000) and the 10-vertex star (8, 1.32252), and on a
    40-point star the areas agree to nine digits (3.264193743 against a 3.264193743 shoelace).

    Two things about the input, both measured. The line cell must be **closed** -- the first index
    repeated, which is what ``polyline_to_pyvista(closed=True)`` does -- and it introduces **zero**
    Steiner points, so the filled polygon reuses the loop's own vertices exactly as ordito does.
    And ``delaunay_2d(edge_source=loop)`` is *not* the alternative route: it ignores the loop as a
    boundary and triangulates the convex hull, measured 63 cells covering area **4.465** against
    the star's 3.264.
    """
    for name, polygon_np in (("L", _l_shape()[:, :2]), ("star", _star(5)[:, :2])):
        points_np = np.column_stack([polygon_np, np.zeros(polygon_np.shape[0])])
        faces_wp = od.polyline.polyline_triangulate(points_to_warp(points_np, device)).numpy()

        filled_pv = polyline_to_pyvista(points_np, closed=True).triangulate_contours()
        assert filled_pv.is_all_triangles, name
        assert filled_pv.n_points == polygon_np.shape[0], name  # no Steiner points
        assert filled_pv.n_cells == polygon_np.shape[0] - 2, name  # non-vacuity
        assert len(faces_wp) == filled_pv.n_cells, name

        area_wp = _triangle_areas(points_np, faces_wp).sum()
        assert np.isclose(area_wp, float(filled_pv.area), rtol=1e-5), name
        assert np.isclose(area_wp, _polygon_area(points_np), rtol=1e-5), name


def test_triangulate_convex_is_fan(device: str) -> None:
    pts_np = _planar_circle(8, 1.5)
    faces_wp = od.polyline.polyline_triangulate(points_to_warp(pts_np, device))
    n = pts_np.shape[0]
    expected = np.stack([np.zeros(n - 2), np.arange(1, n - 1), np.arange(2, n)], axis=1)
    assert np.array_equal(faces_wp.numpy(), expected.astype(np.int32))
    _assert_valid_triangulation(pts_np, faces_wp.numpy())


@pytest.mark.parametrize(
    "polygon",
    [
        pytest.param(_l_shape, id="l-shape"),
        pytest.param(lambda: _star(6), id="star"),
        pytest.param(lambda: _rotate_into_3d(_star(6), seed=7), id="tilted-plane"),
        pytest.param(lambda: _l_shape()[::-1].copy(), id="clockwise"),
    ],
)
def test_triangulate_is_valid(device: str, polygon: Callable[[], np.ndarray]) -> None:
    """
    Not a library comparison: ``n - 2`` non-degenerate faces, every vertex used, the exact area.

    Non-convex rings (one reflex corner, alternating reflex corners), the star moved into a tilted
    3-D plane, and the L-shape wound clockwise; a fan over a reflex vertex fails the area assert.
    """
    pts_np = polygon()
    faces_wp = od.polyline.polyline_triangulate(points_to_warp(pts_np, device))
    _assert_valid_triangulation(pts_np, faces_wp.numpy())


def test_triangulate_closed_input_matches_open(device: str) -> None:
    pts_np = _star(5)
    closed_np = _closed_from(pts_np)
    faces_open = od.polyline.polyline_triangulate(points_to_warp(pts_np, device)).numpy()
    faces_closed = od.polyline.polyline_triangulate(points_to_warp(closed_np, device)).numpy()
    assert np.array_equal(faces_open, faces_closed)


def test_triangulate_single_triangle(device: str) -> None:
    pts_np = _planar_circle(3, 1.5)
    faces_wp = od.polyline.polyline_triangulate(points_to_warp(pts_np, device))
    assert faces_wp.shape == (1, 3)
    assert set(faces_wp.numpy().ravel().tolist()) == {0, 1, 2}


def test_triangulate_too_few_points(device: str) -> None:
    pts_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    faces_wp = od.polyline.polyline_triangulate(points_to_warp(pts_np, device))
    assert faces_wp.shape == (0, 3)


@pytest.mark.parametrize("one_block_max", [512, 0])
def test_triangulate_from_offsets_matches_each_loop(
    device: str, one_block_max: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: the packed triangulation against one call per loop.

    `polyline_triangulate` carries the oracle for each loop (its meshlib and pyvista comparisons);
    the packed form must give every loop the same triangle set, offset to packed indices. The
    loops cover the block path's convex fan, a clockwise L, a star in a tilted plane, a loop
    repeating its first point, one of two points (no faces) and a random star longer than the
    one-block cap, which takes the round loop; ``one_block_max = 0`` sends every loop there.
    """
    monkeypatch.setattr(kernel_polyline, "EAR_ONE_BLOCK_MAX", one_block_max)
    rng = np.random.default_rng(7)
    angle_np = np.sort(rng.uniform(0.0, 2.0 * np.pi, 700))
    radius_np = rng.uniform(0.2, 1.0, 700)
    long_np = np.column_stack(
        (radius_np * np.cos(angle_np), radius_np * np.sin(angle_np), np.full(700, 0.5))
    )
    tilt_np = np.asarray(tm.transformations.rotation_matrix(0.6, [1.0, 1.0, 0.0]))[:3, :3]
    loops_np = [
        _planar_circle(7, 1.5),
        _l_shape()[::-1].copy(),
        _star(9) @ tilt_np.T + 3.0,
        _closed_from(_star(5)),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        long_np,
    ]
    offsets_np = np.concatenate(([0], np.cumsum([loop.shape[0] for loop in loops_np])))
    faces_wp, face_offsets_wp = od.polyline.polyline_triangulate_from_offsets(
        points_to_warp(np.concatenate(loops_np), device),
        wp.array(offsets_np.astype(np.int32), dtype=wp.int32, device=device),
    )
    faces_np = faces_wp.numpy()
    face_offsets_np = face_offsets_wp.numpy()
    assert face_offsets_np.shape == (len(loops_np) + 1,)
    assert face_offsets_np[-1] == faces_wp.shape[0]
    for r, loop_np in enumerate(loops_np):
        alone_np = od.polyline.polyline_triangulate(points_to_warp(loop_np, device)).numpy()
        packed_np = faces_np[face_offsets_np[r] : face_offsets_np[r + 1]] - offsets_np[r]
        assert np.array_equal(lexsort_rows(packed_np), lexsort_rows(alone_np))
    assert face_offsets_np[5] - face_offsets_np[4] == 0
    assert face_offsets_np[6] - face_offsets_np[5] == 698


def test_triangulate_from_offsets_without_loops(device: str) -> None:
    """Not a library comparison: no loops give no faces and the one-entry offsets."""
    faces_wp, face_offsets_wp = od.polyline.polyline_triangulate_from_offsets(
        points_to_warp(np.zeros((0, 3)), device), wp.zeros(1, dtype=wp.int32, device=device)
    )
    assert faces_wp.shape == (0, 3)
    assert face_offsets_wp.numpy().tolist() == [0]


# --- simplify (Ramer-Douglas-Peucker, NumPy reference) ---


def _simplify_np(pts_np: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    # Mirrors ``igl::ramer_douglas_peucker``; returns (S, J).
    n = len(pts_np)
    keep = np.ones(n, dtype=bool)
    stol = tol * tol

    def rec(ixs: int, ixe: int) -> None:
        sdmax, ixc = 0.0, -1
        if ixe - ixs > 1:
            s, d = pts_np[ixs], pts_np[ixe]
            dms = d - s
            sdes = float(dms @ dms)
            for k in range(ixs + 1, ixe):
                p = pts_np[k]
                if sdes <= 1e-7:
                    sd = float((p - s) @ (p - s))
                else:
                    t = -(dms @ (s - p)) / sdes
                    proj = (1.0 - t) * s + t * d
                    sd = float((p - proj) @ (p - proj))
                if sd > sdmax:
                    sdmax, ixc = sd, k
        if sdmax <= stol:
            keep[ixs + 1 : ixe] = False
        else:
            rec(ixs, ixc)
            rec(ixc, ixe)

    if n >= 2:
        rec(0, n - 1)
    indices = np.flatnonzero(keep)
    return pts_np[indices], indices


@pytest.mark.parametrize("tol", [0.1, 0.5])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_simplify_matches_reference(device: str, tol: float, seed: int) -> None:
    pts_np = _random_open_polyline(seed, n=40)
    simplified_wp, indices_wp = od.polyline.polyline_simplify(points_to_warp(pts_np, device), tol)
    simplified_np, indices_np = _simplify_np(pts_np, tol)
    assert np.array_equal(indices_wp.numpy(), indices_np.astype(np.int32))
    assert np.allclose(
        simplified_wp.numpy(), simplified_np.astype(np.float32), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize(
    "builder",
    [
        pytest.param(
            lambda: [[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [2.0, 0.0, 0.0]], id="nan_interior"
        ),
        pytest.param(
            lambda: [[np.nan, 0.0, 0.0], [1.0, 5.0, 0.0], [2.0, 0.0, 0.0]], id="nan_endpoint"
        ),
        pytest.param(
            lambda: [[0.0, 0.0, 0.0], [1.0, np.inf, 0.0], [2.0, 0.0, 0.0]], id="inf_interior"
        ),
        pytest.param(lambda: np.full((5, 3), np.nan).tolist(), id="all_nan"),
    ],
)
def test_simplify_non_finite_terminates(
    device: str, builder: Callable[[], list[list[float]]]
) -> None:
    """
    A non-finite coordinate leaves the round loop terminating and the answer well-formed.

    Not a library comparison: no reference simplifier defines an answer here. What is asserted is
    that the loop ends and the result is still a sorted index set keeping both endpoints -- the
    round loop is driven by a device-side condition, so a span that never settles would hang rather
    than return a wrong answer.

    The mechanism is worth recording because it is not the obvious one: ``wp.atomic_max`` does
    **not** propagate ``NaN`` (measured ``atomic_max(-1, NaN, 25) == 25.0`` on both devices), so a
    ``NaN`` distance simply loses its span's reduction. Either a finite point wrote the maximum and
    resolves the argmax, or every interior distance was ``NaN``, the accumulator keeps its armed
    ``-1.0``, and the span settles as within tolerance. That is why ``rdp_split_spans``'
    unresolved-argmax guard is unreachable rather than load-bearing here.
    """
    pts_np = np.array(builder(), dtype=np.float64)
    _, indices_wp = od.polyline.polyline_simplify(points_to_warp(pts_np, device), 0.1)
    kept = indices_wp.numpy()
    assert kept[0] == 0
    assert kept[-1] == pts_np.shape[0] - 1
    assert np.all(np.diff(kept) > 0)  # still a sorted index set into the input


@pytest.mark.parametrize(
    ("points", "tol", "expected"),
    [
        pytest.param(
            np.stack([np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)], axis=1),
            1e-3,
            [0, 10],
            id="collinear-collapses-to-endpoints",
        ),
        pytest.param(
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]]),
            0.1,
            [0, 1, 2],
            id="sharp-corner-kept",
        ),
        pytest.param(np.zeros((0, 3)), 0.5, [], id="empty"),
        pytest.param(np.array([[0.3, -0.4, 1.2]]), 0.5, [0], id="single-point"),
        pytest.param(np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]), 0.5, [0, 1], id="two-points"),
    ],
)
def test_simplify_small_inputs(
    device: str, points: np.ndarray, tol: float, expected: list[int]
) -> None:
    """
    Not a library comparison: inputs whose kept set is known by hand.

    A straight run keeps only its endpoints, a tent's apex deviates far from the base chord and is
    kept, and the inputs with fewer than three points come back whole (or empty).
    """
    simplified_wp, indices_wp = od.polyline.polyline_simplify(points_to_warp(points, device), tol)
    assert np.array_equal(indices_wp.numpy(), np.array(expected, dtype=np.int32))
    assert simplified_wp.shape == (len(expected),)
    assert np.allclose(simplified_wp.numpy(), points[expected].astype(np.float32))


@pytest.mark.parametrize("cap", [0, 10**9])
@pytest.mark.parametrize("shape", ["spiral", "walk"])
def test_simplify_block_and_round_loop_match_reference(
    device: str, shape: str, cap: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Class A, against the recursive NumPy port, through both forms of the round loop.

    Up to ``RDP_ONE_BLOCK_MAX`` points on CUDA -- and at every length on the CPU device -- the whole
    loop runs as one block (``rdp_simplify_block``), longer polylines on CUDA as the captured
    four-launch round. Every other simplify test here sits below the cap, so it is forced both
    ways: ``0`` sends a CUDA call through the launches, ``10**9`` through the block. The deep
    spiral runs dozens of rounds and the random walk is 1 500 points, more than the block's lanes,
    so a lane walks several points per step and the barriers between the steps are exercised.
    """
    monkeypatch.setattr(kernel_polyline, "RDP_ONE_BLOCK_MAX", cap)
    if shape == "spiral":
        angle = np.linspace(0.0, 20.0 * 2.0 * np.pi, 512)
        radius = np.linspace(0.05, 1.0, 512)
        pts_np = np.stack(
            [radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)], axis=1
        )
        tol = 1e-2 * float(np.ptp(pts_np, axis=0).max())
    else:
        pts_np = np.cumsum(np.random.default_rng(5).normal(size=(1500, 3)), axis=0)
        tol = 0.5
    _, indices_wp = od.polyline.polyline_simplify(points_to_warp(pts_np, device), tol)
    _, indices_np = _simplify_np(pts_np, tol)
    # Non-vacuous: a real share of the points is dropped and a real share kept.
    assert 10 < indices_np.size < pts_np.shape[0] - 100
    assert np.array_equal(indices_wp.numpy(), indices_np.astype(np.int32))


@pytest.mark.parametrize("tol", [0.1, 0.5])
def test_simplify_closed_matches_reference(device: str, tol: float) -> None:
    pts_np = _random_open_polyline(3, n=30)
    closed_np = _closed_from(pts_np)
    simplified_wp, indices_wp = od.polyline.polyline_simplify(
        points_to_warp(pts_np, device), tol, closed=True
    )
    simplified_np, indices_np = _simplify_np(closed_np, tol)
    assert np.array_equal(indices_wp.numpy(), indices_np.astype(np.int32))
    assert np.allclose(
        simplified_wp.numpy(), simplified_np.astype(np.float32), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("cap", [0, 10**9])
@pytest.mark.parametrize("repeated", [False, True])
def test_simplify_closed_through_both_round_forms(
    device: str, cap: int, repeated: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Class A, against the recursive NumPy port of the closed loop, through both round forms.

    ``closed=True`` reaches its closing point by wrapping the index over ``n + 1`` entries; an
    input that already repeats its first point (``repeated``) leaves the last entry a dead slot
    the seed must settle unkept, where a missed slot would add a spurious index ``n``. ``cap``
    forces the one-block form (``10**9``) or the captured launches (``0``) as in
    ``test_simplify_block_and_round_loop_match_reference``.
    """
    monkeypatch.setattr(kernel_polyline, "RDP_ONE_BLOCK_MAX", cap)
    angle = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    radius = 1.0 + 0.3 * np.sin(7.0 * angle)
    pts_np = np.stack([radius * np.cos(angle), radius * np.sin(angle), 0.1 * angle], axis=1)
    closed_np = _closed_from(pts_np)
    input_np = closed_np if repeated else pts_np
    simplified_wp, indices_wp = od.polyline.polyline_simplify(
        points_to_warp(input_np, device), 0.02, closed=True
    )
    simplified_np, indices_np = _simplify_np(closed_np, 0.02)
    # Non-vacuous: the loop keeps both ends of the seam and drops a real share of the rest.
    assert 10 < indices_np.size < pts_np.shape[0] - 100
    assert indices_np[-1] == pts_np.shape[0]
    assert np.array_equal(indices_wp.numpy(), indices_np.astype(np.int32))
    assert np.allclose(
        simplified_wp.numpy(), simplified_np.astype(np.float32), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parity("polyline_simplify", "meshlib", "pyvista")
@pytest.mark.parametrize("tolerance", [0.8, 2.4])
def test_simplify_matches_the_two_decimators(device: str, tolerance: float) -> None:
    """
    Class C (a derived scalar): three decimators removing the same number of points.

    Neither reference is Ramer-Douglas-Peucker and **neither takes ordito's tolerance**, which is
    the finding this test exists to pin rather than a limitation to apologise for.
    ``decimatePolyline``'s ``maxError`` is a *collapse cost*, not a deviation bound: driven by it at
    0.8134 on this fixture, its output sits **1.9187** from the input, 2.4x the number it was
    handed. So a tolerance-matched comparison would compare two different quantities under one
    parameter name. Both references are therefore driven by the **count** ordito's tolerance
    produces, and what is compared is where each one puts the points it keeps.

    Measured at the two settings below, all three keeping the same count:

    | tolerance | kept | ordito | meshlib | pyvista |
    |---|---|---|---|---|
    | 0.8 | 39 of 40 | 0.373016 | **0.151980** | 0.373016 |
    | 2.4 | 10 of 40 | 1.760643 | 2.402235 | 1.918705 |

    Two things in that table. **pyvista is exact at the light setting** -- ``decimate_polyline``
    retains ordito's own point set, a two-sided Hausdorff of **0.0** -- and diverges at the heavy
    one (1.9187), both legal since neither claims a unique answer at a 4x reduction. And **MeshLib
    is the most accurate of the three at the light setting and the only one over budget at the heavy
    one** (2.402235 against 2.4), which is the same fact as the paragraph above: its reduction is
    driven by a quadric cost, so the deviation is an outcome rather than a promise. That is why the
    tolerance bound below is asserted on ordito and pyvista and not on MeshLib.

    pyvista's ``PolyData`` must hold **one** line cell. ``pv.lines_from_points`` gives one two-point
    cell per segment and ``decimate_polyline`` is then a **no-op at every reduction**, so a
    comparison built that way asserts nothing and looks like agreement --
    [`polyline_to_pyvista`][tests.conversions.polyline_to_pyvista] is the builder for that reason,
    and its cell count is asserted here rather than assumed.

    ``optimizeVertexPos`` and ``touchBdVertices`` are turned off on the MeshLib side for the reasons
    ``test_downsample_polyline_matches_meshlib`` records; without the first, the "keeps a subset of
    the input points" assert below would fail on every setting.

    **Bug class excluded:** a simplifier that moves the points it keeps (all three), and one that
    exceeds its own tolerance (the two that promise one). **Mutation probe, measured:** MeshLib's
    ``maxError``-driven answer deviates 1.9187 against a 0.8134 bound, 2.4x, so the bound is not
    something any reduction of roughly the right size passes.
    """
    pts_np = _random_open_polyline(50, n=40)
    simplified_wp, _indices_wp = od.polyline.polyline_simplify(
        points_to_warp(pts_np, device), tolerance
    )
    simplified_np = simplified_wp.numpy().astype(np.float64)
    n_kept = len(simplified_np)
    assert 2 < n_kept < pts_np.shape[0]  # non-vacuity: it simplified, and did not collapse

    polyline_ml = _polyline_ml(pts_np)
    settings_ml = mm.DecimatePolylineSettings_Vector3f()
    settings_ml.maxDeletedVertices = pts_np.shape[0] - n_kept
    settings_ml.maxError = 1e30  # a collapse cost, not a deviation bound -- see the docstring
    settings_ml.optimizeVertexPos = False
    settings_ml.touchBdVertices = False
    result_ml = mm.decimatePolyline(polyline_ml, settings_ml)
    simplified_ml = _ordered_ml(polyline_ml)

    line_pv = polyline_to_pyvista(pts_np)
    assert line_pv.n_cells == 1  # a per-segment cell set would make decimate_polyline a no-op
    decimated_pv = line_pv.decimate_polyline(1.0 - n_kept / pts_np.shape[0])
    simplified_pv = np.asarray(decimated_pv.points, dtype=np.float64)

    assert int(result_ml.vertsDeleted) == pts_np.shape[0] - n_kept
    assert simplified_ml.shape[0] == n_kept
    assert simplified_pv.shape[0] == n_kept

    # All three keep a subset of the input points rather than moving them.
    for reduced_np in (simplified_np, simplified_ml, simplified_pv):
        assert _max_deviation_np(reduced_np, pts_np) < 1e-5
    # Only the two that bound the deviation are held to the tolerance.
    assert _max_deviation_np(pts_np, simplified_np) <= tolerance
    assert _max_deviation_np(pts_np, simplified_pv) <= tolerance


@pytest.mark.parametrize("ending", ["open", "repeated", "within_tolerance"])
def test_closed_keyword_equals_closing_the_polyline_explicitly(device: str, ending: str) -> None:
    """
    ``closed=True`` is exactly the open computation on ``polyline_close(polyline)``.

    Ordito against ordito: the convention, asserted once for every function that carries the
    keyword rather than seven times in seven near-identical tests. Three of the ten additionally
    drop the duplicated seam point on the way out, so they are checked against the sliced form
    instead -- which is the whole difference between the two groups and the thing most likely to
    drift.

    ``ending`` is the part the kernels decide on the device: an open-ended input gains the
    closing segment, while one whose last point already repeats the first -- exactly, or within
    ``allclose``'s tolerance -- gains nothing, ``polyline_close`` returning it unchanged.
    **Mutation probe:** making the kernels' closure test always add the segment fails both
    repeated arms, at the ``polyline_upsample`` assert.
    """
    pts_np = _random_open_polyline(33, n=20)
    if ending != "open":
        seam_np = pts_np[:1] * (1.0 + (5e-6 if ending == "within_tolerance" else 0.0))
        pts_np = np.concatenate([pts_np, seam_np])
    polyline_wp = points_to_warp(pts_np, device)
    stored_np = polyline_wp.numpy()
    assert od.polyline.is_closed(polyline_wp) == (ending != "open")
    assert np.array_equal(stored_np[0], stored_np[-1]) == (ending == "repeated")
    closed_wp = od.polyline.polyline_close(polyline_wp)
    queries_wp = points_to_warp(_random_open_polyline(34, n=15), device)
    n_original = polyline_wp.size

    # Pure delegation: closed=True adds the segment and changes nothing else.
    assert np.isclose(
        od.polyline.polyline_length(polyline_wp, closed=True),
        od.polyline.polyline_length(closed_wp),
    )
    assert np.allclose(
        list(od.polyline.polyline_centroid(polyline_wp, closed=True)),
        list(od.polyline.polyline_centroid(closed_wp)),
    )
    assert np.isclose(
        od.polyline.polyline_radius(polyline_wp, "mean", closed=True),
        od.polyline.polyline_radius(closed_wp, "mean"),
    )
    assert np.array_equal(
        od.polyline.polyline_point_distance(queries_wp, polyline_wp, closed=True).numpy(),
        od.polyline.polyline_point_distance(queries_wp, closed_wp).numpy(),
    )
    assert np.array_equal(
        od.polyline.polyline_upsample(polyline_wp, 0.3, closed=True).numpy(),
        od.polyline.polyline_upsample(closed_wp, 0.3).numpy(),
    )
    assert np.array_equal(
        od.polyline.polyline_downsample(polyline_wp, 0.3, closed=True).numpy(),
        od.polyline.polyline_downsample(closed_wp, 0.3).numpy(),
    )
    simplified_closed, indices_closed = od.polyline.polyline_simplify(
        polyline_wp, 0.05, closed=True
    )
    simplified_explicit, indices_explicit = od.polyline.polyline_simplify(closed_wp, 0.05)
    assert np.array_equal(simplified_closed.numpy(), simplified_explicit.numpy())
    assert np.array_equal(indices_closed.numpy(), indices_explicit.numpy())

    # ...and the three that drop the duplicated seam point, so their result is a cyclic ring.
    assert np.array_equal(
        od.polyline.polyline_resample(polyline_wp, 17, closed=True).numpy(),
        od.polyline.polyline_resample(closed_wp, 18).numpy()[0:17],
    )
    assert np.array_equal(
        od.polyline.polyline_angles(polyline_wp, closed=True).numpy(),
        od.polyline.polyline_angles(closed_wp).numpy()[0:n_original],
    )
    assert np.array_equal(
        od.polyline.polyline_smooth_upsample(polyline_wp, 0.3, closed=True).numpy(),
        od.polyline.polyline_smooth_upsample(closed_wp, 0.3, closed=True).numpy(),
    )


# --- triangulate_polygon: the 2D entry point to the same ear clipper ---------------------------

_SQUARE_RING = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
_L_RING = _l_shape()[:, :2]


def _signed_ring_area(ring_np: np.ndarray) -> float:
    """Shoelace area of a 2D ring, positive counter-clockwise."""
    return 0.5 * float(
        np.dot(ring_np[:, 0], np.roll(ring_np[:, 1], -1))
        - np.dot(np.roll(ring_np[:, 0], -1), ring_np[:, 1])
    )


def _tiled_area(ring_np: np.ndarray, faces_np: np.ndarray) -> float:
    triangles_np = ring_np.astype(np.float64)[faces_np.reshape(-1, 3)]
    edge_a = triangles_np[:, 1] - triangles_np[:, 0]
    edge_b = triangles_np[:, 2] - triangles_np[:, 0]
    return 0.5 * float(np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]).sum())


@pytest.mark.parametrize("ring_name", ["square", "L"])
def test_triangulate_polygon(device: str, ring_name: str) -> None:
    ring_np = _SQUARE_RING if ring_name == "square" else _L_RING
    vertices_wp, faces_wp = od.polyline.triangulate_polygon(points_to_warp_uv(ring_np, device))
    assert vertices_wp.size == ring_np.shape[0]
    assert faces_wp.size // 3 == ring_np.shape[0] - 2
    # No Steiner points, and the triangles must tile the polygon exactly.
    exact_area = 2.0 if ring_name == "square" else 3.0
    assert np.isclose(_tiled_area(vertices_wp.numpy(), faces_wp.numpy()), exact_area, rtol=1e-5)


def _star_ring_wp(n: int, inner: float = 0.45) -> np.ndarray:
    """Alternating-radius star: every other vertex is reflex, so no ear has an ear-free ring-2."""
    angle_np = 2.0 * np.pi * np.arange(n) / n
    radius_np = np.where(np.arange(n) % 2 == 0, 1.0, inner)
    return np.column_stack((radius_np * np.cos(angle_np), radius_np * np.sin(angle_np)))


@pytest.mark.parametrize("n", [16, 64, 512])
def test_triangulate_polygon_star(device: str, n: int) -> None:
    """
    A star ring is the worst case for the ear clipper's independent-set rule.

    Half its vertices are reflex and the convex ones alternate, so competing ears sit exactly two
    apart around the ring -- the configuration that made a raw-index rank clip one ear per round.
    """
    ring_np = _star_ring_wp(n)
    vertices_wp, faces_wp = od.polyline.triangulate_polygon(points_to_warp_uv(ring_np, device))
    assert vertices_wp.size == n
    assert faces_wp.size // 3 == n - 2

    triangles_np = vertices_wp.numpy().astype(np.float64)[faces_wp.numpy().reshape(-1, 3)]
    edge_a = triangles_np[:, 1] - triangles_np[:, 0]
    edge_b = triangles_np[:, 2] - triangles_np[:, 0]
    signed_np = 0.5 * (edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    # Consistent winding: every triangle turns the same way as the ring, so no signed area flips.
    assert np.all(signed_np > 0.0) or np.all(signed_np < 0.0)
    # And they tile the star exactly (shoelace over the ring).
    assert np.isclose(np.abs(signed_np).sum(), abs(_signed_ring_area(ring_np)), rtol=1e-5)


def _triangle_cover_count(
    vertices_np: np.ndarray, faces_np: np.ndarray, points_np: np.ndarray, margin: float
) -> np.ndarray:
    """
    Count, per query point, how many of the triangles strictly contain it.

    ``margin`` is a barycentric slack that excludes points lying on a triangle edge, so a point
    shared by two triangles of a valid tiling is not double-counted -- with random queries such a
    point is measure-zero anyway, and the slack makes that robust rather than lucky.
    """
    triangles_np = vertices_np[faces_np]
    edge_a = triangles_np[:, 1] - triangles_np[:, 0]
    edge_b = triangles_np[:, 2] - triangles_np[:, 0]
    offset_np = points_np[None, :, :] - triangles_np[:, None, 0, :]
    twice_area = edge_a[:, 0] * edge_b[:, 1] - edge_b[:, 0] * edge_a[:, 1]
    weight_b = (
        offset_np[..., 0] * edge_b[:, None, 1] - offset_np[..., 1] * edge_b[:, None, 0]
    ) / twice_area[:, None]
    weight_c = (
        edge_a[:, None, 0] * offset_np[..., 1] - edge_a[:, None, 1] * offset_np[..., 0]
    ) / twice_area[:, None]
    weight_a = 1.0 - weight_b - weight_c
    inside_np = (weight_a > margin) & (weight_b > margin) & (weight_c > margin)
    return inside_np.sum(axis=0)


@pytest.mark.parametrize("n", [16, 64])
@pytest.mark.parity("triangulate_polygon", "trimesh")
def test_triangulate_polygon_covers_same_region_as_trimesh(device: str, n: int) -> None:
    """
    Class C: two valid ear clippings, so only the tiled region is comparable.

    ``trimesh.creation.triangulate_polygon`` cuts *different* diagonals from ordito's clipper.

    There is no elementwise correspondence to recover -- two valid ear clippings of one polygon are
    genuinely different triangle sets -- so the comparison is the tiled region itself, sampled at
    4 000 uniform points over the bounding box and reduced to a per-point cover count.

    **Bug class excluded:** an ear clipper that emits a triangle *outside* the ring, or that lets
    two ears overlap. Either shows up as a cover count of 0 or 2 where the reference says 1, and
    neither is visible to the area sum in
    [`test_triangulate_polygon_star`][tests.test_polyline.test_triangulate_polygon_star] when the
    surplus and the deficit happen to cancel. The cover count is deliberately *blind* to winding
    (the barycentric weights are scale-invariant, so reversing a triangle changes nothing); the
    signed-area assert in that same star test is what covers orientation.

    **Mutation probe, measured on ``n=64``, 1 275 of the 4 000 samples interior:** the two agree on
    **4 000 / 4 000** points, and the assert is exact equality, so every probe below clears it by
    its full count. Dropping one ordito triangle disagrees on 25; translating the ring by 1% of its
    radius, on 206. Both degenerate implementations fail too: an all-zero face buffer keeps the
    ``n - 2`` count and still disagrees on all 1 275 interior points, and the naive single fan --
    valid only for a convex ring -- disagrees on 1 777.

    Both sides are additionally checked to introduce no Steiner points, which is what makes the
    vertex arrays directly comparable as sets.
    """
    ring_np = _star_ring_wp(n)
    vertices_wp, faces_wp = od.polyline.triangulate_polygon(points_to_warp_uv(ring_np, device))
    vertices_tm, faces_tm = tm.creation.triangulate_polygon(sg.Polygon(ring_np))

    assert faces_wp.size // 3 == faces_tm.shape[0] == n - 2
    assert vertices_wp.size == vertices_tm.shape[0] == n
    # Same vertex set: equal counts plus a two-sided Hausdorff distance at float32 resolution. A
    # lexsort compare is not usable here -- the star has coordinate pairs that tie to 1e-16, so the
    # row order is decided by rounding noise rather than by the values.
    assert hausdorff_two_sided(vertices_wp.numpy().astype(np.float64), vertices_tm) < 1e-6

    rng = np.random.default_rng(11)
    points_np = rng.uniform(-1.05, 1.05, size=(4000, 2))
    count_wp = _triangle_cover_count(
        vertices_wp.numpy().astype(np.float64), faces_wp.numpy().reshape(-1, 3), points_np, 1e-9
    )
    count_tm = _triangle_cover_count(vertices_tm, faces_tm, points_np, 1e-9)
    assert np.array_equal(count_wp, count_tm)


@pytest.mark.parity("triangulate_polygon", "trimesh")
def test_triangulate_polygon_near_collinear(device: str) -> None:
    """
    Class C, against ``trimesh.creation.triangulate_polygon``: the tiled region on a near-flat ring.

    A ring whose interior vertices are almost on the line back to the start, so every ear test is
    decided by a near-zero cross product -- where a ranking change could stall the clipper, and
    where the chosen diagonals depend on exactly how the orientation test is evaluated (evaluating
    it on centred, rotated coordinates picked different ears from the input coordinates). No
    reference pins the diagonals, so the claim is the one
    [`test_triangulate_polygon_covers_same_region_as_trimesh`][tests.test_polyline.test_triangulate_polygon_covers_same_region_as_trimesh]
    makes: a complete ``n - 2``-face tiling whose per-point cover count equals the reference's, so
    no triangle lies outside the ring and none overlap. Measured agreement 20 000 / 20 000 samples
    on both devices, cover count never above 1.
    """
    n = 64
    x_np = np.linspace(0.0, 1.0, n - 1)
    ring_np = np.vstack(
        (np.column_stack((x_np, 1e-7 * np.sin(np.pi * x_np))), np.array([[0.5, -0.25]]))
    )
    vertices_wp, faces_wp = od.polyline.triangulate_polygon(points_to_warp_uv(ring_np, device))
    vertices_tm, faces_tm = tm.creation.triangulate_polygon(sg.Polygon(ring_np))
    assert vertices_wp.size == n
    assert faces_wp.size // 3 == faces_tm.shape[0] == n - 2

    points_np = np.random.default_rng(1).uniform([-0.05, -0.3], [1.05, 0.05], size=(20000, 2))
    count_wp = _triangle_cover_count(
        vertices_wp.numpy().astype(np.float64), faces_wp.numpy().reshape(-1, 3), points_np, 1e-12
    )
    count_tm = _triangle_cover_count(vertices_tm, faces_tm, points_np, 1e-12)
    # Non-vacuous: the ring's interior holds a real share of the samples.
    assert (count_tm == 1).sum() > 1000
    assert np.array_equal(count_wp, count_tm)


def test_triangulate_polygon_drops_repeated_closing_point(device: str) -> None:
    closed_np = np.vstack((_SQUARE_RING, _SQUARE_RING[:1]))
    vertices_wp, faces_wp = od.polyline.triangulate_polygon(points_to_warp_uv(closed_np, device))
    assert vertices_wp.size == _SQUARE_RING.shape[0]
    assert faces_wp.size // 3 == _SQUARE_RING.shape[0] - 2


def test_triangulate_polygon_too_few_points(device: str) -> None:
    vertices_wp, faces_wp = od.polyline.triangulate_polygon(
        points_to_warp_uv(np.zeros((2, 2)), device)
    )
    assert vertices_wp.size == 2
    assert faces_wp.size == 0


def test_triangulate_polygon_leaves_a_clockwise_input_unchanged(device: str) -> None:
    """
    Not a library comparison: a clockwise ring is mirrored before the ear tests, the input is not.

    ``triangulate_polygon`` clips the caller's own buffer rather than a projected copy of it, so
    the in-place mirror a clockwise ring needs must land on a copy. Pins that the input and the
    returned ring both still hold the caller's points, and that the faces tile the polygon.
    """
    ring_np = _star_ring_wp(64)[::-1].copy()
    assert _signed_ring_area(ring_np) < 0.0  # the fixture really is clockwise
    ring_wp = points_to_warp_uv(ring_np, device)
    before_np = ring_wp.numpy().copy()
    vertices_wp, faces_wp = od.polyline.triangulate_polygon(ring_wp)
    assert np.array_equal(ring_wp.numpy(), before_np)
    assert np.array_equal(vertices_wp.numpy(), before_np)
    assert faces_wp.size // 3 == 62
    assert np.isclose(_tiled_area(before_np, faces_wp.numpy()), -_signed_ring_area(ring_np))


@pytest.mark.parametrize("n", [64, 1500, 4096])
@pytest.mark.parametrize("clockwise", [False, True])
def test_triangulate_polygon_single_block_matches_round_loop(
    device: str, n: int, clockwise: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: the single-block ear loop against the captured round loop.

    A ring up to ``EAR_ONE_BLOCK_MAX`` corners is clipped by one block that runs every round
    itself, walking the ring for each ear test, a longer one by the four-launch round loop, whose
    ear test walks the ear grid; the trimesh cover test carries the oracle for the first at its
    sizes. Forcing the cap both ways must give the same triangle set (row order is atomic-append
    order on CUDA) on a random star, which has ears at every scale and, in a clockwise ring, reads
    the grid through the mirror: walking the cells of the mirrored triangle instead of the raw one
    changes the faces. The grid walk's rounding pad is not reached by any random ring (deleting it
    leaves these faces unchanged); it is a margin, not a tested branch.
    """
    rng = np.random.default_rng(n)
    angle_np = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
    radius_np = rng.uniform(0.2, 1.0, n)
    ring_np = np.column_stack((radius_np * np.cos(angle_np), radius_np * np.sin(angle_np)))
    if clockwise:
        ring_np = ring_np[::-1].copy()
    ring_wp = points_to_warp_uv(ring_np, device)
    rows = {}
    for cap in (0, 10**9):
        monkeypatch.setattr(kernel_polyline, "EAR_ONE_BLOCK_MAX", cap)
        _, faces_wp = od.polyline.triangulate_polygon(ring_wp)
        faces_np = faces_wp.numpy().reshape(-1, 3)
        rows[cap] = lexsort_rows(faces_np)
    assert rows[0].shape == (n - 2, 3)
    assert np.array_equal(rows[0], rows[10**9])
    assert np.isclose(_tiled_area(ring_np, rows[0]), abs(_signed_ring_area(ring_np)), rtol=1e-5)


def _slanted_ring(points_per_side: int) -> np.ndarray:
    """
    Build a convex hexagon far from the origin with many points per side, rounded to ``float32``.

    Each side's interior points lie on a slanted line only up to ``float32`` rounding, so every
    three consecutive ones are collinear to the rounding: the ring on which an ear clipper (or a
    fan) emits rounding-level slivers.
    """
    angles_np = np.linspace(0.0, 2.0 * np.pi, 7)[:-1] + 0.3
    corners_np = 1000.0 + 37.0 * np.column_stack((np.cos(angles_np), np.sin(angles_np)))
    t_np = np.linspace(0.0, 1.0, points_per_side, endpoint=False)[:, None]
    sides_np = [corners_np[k] + t_np * (corners_np[(k + 1) % 6] - corners_np[k]) for k in range(6)]
    return np.vstack(sides_np).astype(np.float32)


def _sliver_count(ring_np: np.ndarray, faces_np: np.ndarray, ulps: float) -> int:
    """Faces whose height over their longest side is at most ``ulps`` float32 ulps of the ring."""
    triangles_np = ring_np.astype(np.float64)[faces_np.reshape(-1, 3)]
    edges_np = np.roll(triangles_np, -1, axis=1) - triangles_np
    twice_area_np = np.abs(
        edges_np[:, 0, 0] * edges_np[:, 1, 1] - edges_np[:, 0, 1] * edges_np[:, 1, 0]
    )
    height_np = twice_area_np / np.linalg.norm(edges_np, axis=2).max(axis=1)
    bound = ulps * float(np.finfo(np.float32).eps) * float(np.abs(ring_np).max())
    return int((height_np <= bound).sum())


@pytest.mark.parametrize("cap", [0, 10**9], ids=["round_loop", "single_block"])
def test_triangulate_polygon_ring_with_a_repeated_corner_is_complete(
    device: str, cap: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Not a library comparison: a ring holding a zero-length side still gets all ``n - 2`` faces.

    Two consecutive equal points are neither convex nor removable by the strict ear test, and the
    triangle of each neighbouring ear holds the duplicate on its boundary, so a clipper without a
    rule for them stalls with the ring unfinished (a long planar section ring did, two faces
    short). Both clipping paths are forced.
    """
    ring_np = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    monkeypatch.setattr(kernel_polyline, "EAR_ONE_BLOCK_MAX", cap)
    _, faces_wp = od.polyline.triangulate_polygon(points_to_warp_uv(ring_np, device))
    faces_np = faces_wp.numpy()
    assert faces_np.size // 3 == ring_np.shape[0] - 2
    assert np.isclose(_tiled_area(ring_np, faces_np), 1.0)


@pytest.mark.parametrize("cap", [0, 10**9], ids=["round_loop", "single_block"])
def test_triangulate_polygon_leaves_no_rounding_level_slivers(
    device: str, cap: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Not a library comparison: no face's height is at the ring's rounding level.

    On ``_slanted_ring`` every interior side point makes a corner whose orientation is a rounding
    accident. The ring is convex, so without the flat-corner rule it is fanned from vertex 0 and
    the two sides at that vertex become zero-area faces; without the thin-ear deferral the ear
    loop clips those corners as slivers. The tiling must still be complete and exact in area.
    """
    ring_np = _slanted_ring(40)
    monkeypatch.setattr(kernel_polyline, "EAR_ONE_BLOCK_MAX", cap)
    _, faces_wp = od.polyline.triangulate_polygon(points_to_warp_uv(ring_np, device))
    faces_np = faces_wp.numpy()
    assert faces_np.size // 3 == ring_np.shape[0] - 2
    assert np.isclose(
        _tiled_area(ring_np, faces_np), abs(_signed_ring_area(ring_np.astype(np.float64)))
    )
    assert _sliver_count(ring_np, faces_np, 1.0) == 0


def test_polyline_triangulate_is_reproducible_far_from_the_origin(device: str) -> None:
    """
    Ordito against ordito: repeated calls on one long, tilted ring give one triangle set.

    The plane frame decides every ear, so it must not depend on the order a reduction's partials
    arrive in; a ring past the single-block cap and far from the origin is where a per-block,
    ``float32`` frame moved between runs. Rows are compared as sets (their order is atomic-append
    order on CUDA).
    """
    ring_np = _slanted_ring(400).astype(np.float64)
    rotation_np = np.asarray(tm.transformations.rotation_matrix(0.7, [1.0, 2.0, 3.0]))[:3, :3]
    loop_np = np.column_stack((ring_np, np.full(ring_np.shape[0], 250.0))) @ rotation_np.T
    loop_wp = points_to_warp(loop_np, device)
    sets = []
    for _ in range(4):
        faces_np = od.polyline.polyline_triangulate(loop_wp).numpy()
        assert faces_np.shape == (ring_np.shape[0] - 2, 3)
        sets.append(lexsort_rows(faces_np))
    assert all(np.array_equal(sets[0], other) for other in sets[1:])
