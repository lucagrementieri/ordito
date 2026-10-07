from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC4",
    title="Outlier removal: statistical, radius and probabilistic",
    summary="""
    A jittered scan of the bunny with 900 stray points scattered through its bounding box. Three
    detectors, all from neighbourhood statistics:

    - [`statistical_outlier_mask`][ordito.points.statistical_outlier_mask] flags points whose
      mean distance to their `k` nearest neighbours is far above the cloud's average;
    - [`radius_outlier_mask`][ordito.points.radius_outlier_mask] flags points with too few
      neighbours within a fixed radius;
    - [`outlier_probability`][ordito.points.outlier_probability] scores each point by how
      stretched its neighbourhood is relative to its neighbours' (Local Outlier Probability).
    """,
    credits=(
        (
            "Open3D: point cloud outlier removal",
            "https://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html",
        ),
        (
            "PyMeshLab: select outliers",
            "https://pymeshlab.readthedocs.io/en/latest/filter_list.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    points = data.load_points("noisy_cloud", device)
    neighbours, distances = od.neighbors.query_nearest(points, points, k=20)

    statistical = od.points.statistical_outlier_mask(distances, std_ratio=2.0).numpy()
    radius = od.points.radius_outlier_mask(points, radius=0.004, min_neighbors=4).numpy()
    probability = od.points.outlier_probability(neighbours, distances).numpy()

    stray = np.arange(points.shape[0]) >= 30_000  # the input's last 900 points are the strays
    for name, flagged in (
        ("statistical", statistical),
        ("radius", radius),
        ("LoOP > 0.8", probability > 0.8),
    ):
        print(f"{name:>11}: flags {flagged.sum()}, of which {(flagged & stray).sum()} strays")
    # --8<-- [end:code]
    return {
        "points": points.numpy(),
        "statistical": statistical,
        "radius": radius,
        "probability": probability,
    }


def figure(result: dict[str, Any]) -> r.Figure:
    points = result["points"]
    panels: list[r.Panel] = [r.Panel([r.Points(points, color=r.GREY, size=4)], title="Noisy cloud")]
    for key, title in (("statistical", "Statistical"), ("radius", "Radius")):
        flagged = result[key]
        panels.append(
            r.Panel(
                [
                    r.Points(points[~flagged], color=r.GREEN, size=3),
                    r.Points(points[flagged], color=r.RED, size=7),
                ],
                title=f"{title}: {flagged.sum()} flagged",
            )
        )
    order = np.argsort(result["probability"])
    panels.append(
        r.Panel(
            [
                r.Points(
                    points[order],
                    scalars=result["probability"][order],
                    cmap="viridis",
                    clim=(0.0, 1.0),
                    scalar_bar="outlier probability",
                    size=5,
                )
            ],
            title="Local outlier probability",
        )
    )
    return r.Figure(panels, camera=data.camera("noisy_cloud"), ncols=4, panel_size=(560, 520))
