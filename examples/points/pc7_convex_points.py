from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC7",
    title="Convex-hull points and half-space tests",
    summary="""
    Which points of the cloud lie on its convex hull?
    [`convex_subset_mask`][ordito.points.convex_subset_mask] marks the points that are extreme
    along some sampled direction: every one is on the hull, and more directions find more of
    them. [`convex_superset_mask`][ordito.points.convex_superset_mask] answers the other way
    round, discarding only points provably inside, so every hull vertex survives.
    [`half_space_mask`][ordito.points.half_space_mask] is the building block: the points strictly
    on one side of a plane. ordito selects hull *points* only; it does not build the hull's
    triangle mesh.
    """,
    credits=(
        (
            "Open3D: convex hull",
            "https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html",
        ),
        ("trimesh: convex", "https://trimesh.org/trimesh.convex.html"),
    ),
    notes="""
    Checked against an exact hull (`scipy.spatial.ConvexHull`, 1564 vertices on this cloud): all
    1104 points of the subset are hull vertices, and every hull vertex is among the superset's
    candidates. Convex-hull meshing is not part of ordito.
    """,
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    points = data.load_points("bunny_cloud", device)
    on_hull = od.points.convex_subset_mask(points, n_directions=4096).numpy()
    candidates = od.points.convex_superset_mask(points).numpy()
    upper = od.points.half_space_mask(
        points, wp.vec3(0.3, 1.0, 0.0), plane_origin=wp.vec3(0.0, 0.11, 0.0)
    ).numpy()
    print(f"{on_hull.sum()} points certified on the hull (subset)")
    print(f"{candidates.sum()} points that may be on it (superset), of {points.shape[0]}")
    print(f"{upper.sum()} points above the plane")
    # --8<-- [end:code]
    return {"points": points.numpy(), "on_hull": on_hull, "candidates": candidates, "upper": upper}


def figure(result: dict[str, Any]) -> r.Figure:
    points = result["points"]
    on_hull, candidates, upper = result["on_hull"], result["candidates"], result["upper"]
    return r.Figure(
        [
            r.Panel(
                [
                    r.Points(points, color=r.GREY, size=3),
                    r.Points(points[on_hull], color=r.RED, size=10),
                ],
                title=f"Hull points: {on_hull.sum()}",
            ),
            r.Panel(
                [
                    r.Points(points[~candidates], color=r.GREY, size=3),
                    r.Points(points[candidates], color=r.ORANGE, size=6),
                ],
                title=f"Possible hull points: {candidates.sum()}",
            ),
            r.Panel(
                [
                    r.Points(points[~upper], color=r.GREY, size=4),
                    r.Points(points[upper], color=r.GREEN, size=4),
                    r.Plane(np.array([0.0, 0.11, 0.0]), np.array([0.3, 1.0, 0.0]), size=0.2),
                ],
                title="Half space above a plane",
            ),
        ],
        camera=data.camera("bunny_cloud"),
    )
