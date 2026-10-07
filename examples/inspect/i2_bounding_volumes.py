from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="I2",
    title="Bounding boxes, principal axes and a fitted plane",
    summary="""
    A bunny point cloud turned about a skew axis. The axis-aligned box from
    [`aabb`][ordito.bounds.aabb] grows with the tilt; the oriented box from
    [`oriented_bounding_box`][ordito.bounds.oriented_bounding_box] searches orientations for the
    smallest volume and does not. [`principal_axes`][ordito.points.principal_axes] gives the
    covariance frame (drawn at three times the standard deviation along each axis), and
    [`fit_plane`][ordito.points.fit_plane] the least-squares plane, whose normal is the least
    spread axis.
    """,
    credits=(
        (
            "Open3D: bounding volumes",
            "https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html",
        ),
        ("trimesh: examples", "https://trimesh.org/examples.html"),
        ("libigl 910", "https://libigl.github.io/tutorial/#bounding-boxes"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    points = data.load_points("tilted_bunny_cloud", device)

    aabb_lower, aabb_upper = od.bounds.aabb(points)
    rotation, obb_lower, obb_upper = od.bounds.oriented_bounding_box(points)
    axes, scatter, centroid = od.points.principal_axes(points)  # scatter = n - 1 times variance
    plane_normal, plane_origin = od.points.fit_plane(points)

    aabb_volume = np.prod(np.array(aabb_upper) - np.array(aabb_lower))
    obb_volume = np.prod(np.array(obb_upper) - np.array(obb_lower))
    print(f"axis-aligned box volume {aabb_volume:.3e}, oriented box volume {obb_volume:.3e}")
    deviations = np.sqrt(np.array(scatter) / (points.size - 1))
    print(f"standard deviations along the principal axes {deviations}")
    # --8<-- [end:code]
    return {
        "points": points.numpy(),
        "aabb": (np.array(aabb_lower), np.array(aabb_upper)),
        "obb": (np.array(rotation).reshape(3, 3), np.array(obb_lower), np.array(obb_upper)),
        "axes": np.array(axes).reshape(3, 3),
        "deviations": deviations,
        "centroid": np.array(centroid),
        "plane": (np.array(plane_origin), np.array(plane_normal)),
    }


def _corners(
    lower: np.ndarray, upper: np.ndarray, rotation: np.ndarray | None = None
) -> np.ndarray:
    """Return the eight corners in ``itertools.product`` order, mapped from box frame to world."""
    corners = np.array(list(itertools.product(*zip(lower, upper, strict=True))))
    return corners if rotation is None else corners @ rotation  # rows of rotation are the axes


def figure(result: dict[str, Any]) -> r.Figure:
    points = result["points"]
    cloud = r.Points(points, color=r.LIGHT_GREEN, size=3.0)
    rotation, lower, upper = result["obb"]
    aabb = _corners(*result["aabb"])
    obb = _corners(lower, upper, rotation)
    deviations = result["deviations"]
    centroid = result["centroid"]
    origin, normal = result["plane"]
    size = float(np.ptp(points, axis=0).max())
    both = np.concatenate([aabb, obb])
    bounds = np.stack([both.min(axis=0), both.max(axis=0)])
    return r.Figure(
        [
            r.Panel(
                [cloud, r.WireBox(aabb, color=r.BLUE), r.WireBox(obb, color=r.RED)],
                title="Axis-aligned (blue) and oriented (red) boxes",
                bounds=bounds,
                camera=r.Camera(direction=(0.25, 0.25, 1.0), up=(0.0, 1.0, 0.0), zoom=1.3),
            ),
            r.Panel(
                [
                    r.Points(points, color=r.LIGHT_GREEN, size=2.0, opacity=0.35),
                    r.Arrows(
                        np.repeat(centroid[None], 3, axis=0),
                        result["axes"] * (3.0 * deviations)[:, None],
                        color=r.ORANGE,
                    ),
                    r.Plane(origin, normal, size=1.1 * size, color=r.BLUE),
                ],
                title="Principal axes and fitted plane",
                bounds=bounds,
                camera=r.Camera(direction=(1.0, 0.45, 0.55), up=(0.0, 1.0, 0.0), zoom=1.3),
            ),
        ],
        camera=data.camera("tilted_bunny_cloud"),
        panel_size=(720, 620),
    )
