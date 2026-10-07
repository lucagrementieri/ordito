from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC5",
    title="Cloud-to-cloud distance",
    summary="""
    Two scans of the same bunny, the second pushed outwards by a bump on its flank and a gentle
    ripple. [`query_nearest`][ordito.neighbors.query_nearest] with `k=1` finds, for every point
    of the second cloud, its nearest point in the first and the distance to it: a per-point
    deviation map with no mesh in sight. For one cloud against itself,
    [`nearest_neighbor_distance`][ordito.neighbors.nearest_neighbor_distance] gives each point's
    spacing to its closest other point, the noise floor such a comparison sits on.
    """,
    credits=(
        (
            "Open3D: point cloud distance",
            "https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html",
        ),
        (
            "PyVista: distance between point clouds",
            "https://docs.pyvista.org/examples/01-filter/distance_between_surfaces",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    reference = data.load_points("bunny_cloud", device)
    displaced = data.load_points("bunny_cloud_displaced", device)

    _, distance = od.neighbors.query_nearest(reference, displaced, k=1)
    spacing = od.neighbors.nearest_neighbor_distance(reference)
    print(f"deviation: median {np.median(distance.numpy()):.5f}, max {distance.numpy().max():.5f}")
    print(f"spacing of the reference cloud: median {np.median(spacing.numpy()):.5f}")
    # --8<-- [end:code]
    return {
        "reference": reference.numpy(),
        "displaced": displaced.numpy(),
        "distance": distance.numpy(),
        "spacing": spacing.numpy(),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    distance, spacing = result["distance"], result["spacing"]
    clim = (0.0, float(np.percentile(distance, 99.5)))  # one scale for both maps
    return r.Figure(
        [
            r.Panel(
                [
                    r.Points(result["reference"], color=r.GREY, size=4),
                    r.Points(result["displaced"], color=r.BLUE, size=4, opacity=0.6),
                ],
                title="Reference (grey) and displaced (blue)",
            ),
            r.Panel(
                [
                    r.Points(
                        result["displaced"],
                        scalars=distance,
                        cmap="magma_r",
                        clim=clim,
                        scalar_bar="distance to the reference",
                        size=5,
                    )
                ],
                title="Deviation of the displaced cloud",
            ),
            r.Panel(
                [
                    r.Points(
                        result["reference"],
                        scalars=spacing,
                        cmap="magma_r",
                        clim=clim,
                        scalar_bar="nearest-neighbour spacing",
                        size=5,
                    )
                ],
                title="Spacing within the reference",
            ),
        ],
        camera=data.camera("bunny_cloud"),
    )
