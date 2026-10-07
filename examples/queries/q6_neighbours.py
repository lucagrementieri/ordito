from __future__ import annotations

from typing import Any

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="Q6",
    title="Nearest neighbours and ball queries",
    summary="""
    Two neighbourhood queries on a 30 000-point cloud, from five query points.
    [`query_nearest`][ordito.neighbors.query_nearest] returns the `k` nearest points of each
    query (a fixed count, whose patch size varies with the local density), and
    [`query_ball`][ordito.neighbors.query_ball] returns every point within a radius (a fixed size,
    whose count varies). Both are exact; the spatial index is built internally, or passed in to
    reuse it across calls.
    """,
    credits=(
        ("Open3D: KD-tree", "https://www.open3d.org/docs/release/tutorial/geometry/kdtree.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    points = data.load_points("bunny_cloud", device)
    picked = points.numpy()[[1200, 5000, 9100, 17000, 26000]]
    queries = wp.array(picked, dtype=wp.vec3, device=device)

    knn_index, knn_distance = od.neighbors.query_nearest(points, queries, k=400)
    ball_index, ball_distance = od.neighbors.query_ball(points, queries, r=0.012)
    print("k nearest: farthest neighbour at", knn_distance.numpy()[:, -1].round(4))
    print("ball of radius 0.012: counts", [index.shape[0] for index in ball_index])
    # --8<-- [end:code]
    return {
        "points": points.numpy(),
        "queries": queries.numpy(),
        "knn": knn_index.numpy(),
        "ball": [index.numpy() for index in ball_index],
        "ball_distance": [distance.numpy() for distance in ball_distance],
    }


_COLORS = [r.RED, r.BLUE, r.ORANGE, r.MAGENTA, "#00a0a0"]


def figure(result: dict[str, Any]) -> r.Figure:
    points, queries = result["points"], result["queries"]
    knn_layers: list[r.Layer] = [r.Points(points, color=r.GREY, size=3)]
    ball_layers: list[r.Layer] = [r.Points(points, color=r.GREY, size=3)]
    for k, color in enumerate(_COLORS):
        knn_layers.append(r.Points(points[result["knn"][k]], color=color, size=6))
        ball = result["ball"][k]
        ball_layers.append(r.Points(points[ball], color=color, size=6))
    for layers in (knn_layers, ball_layers):
        layers.append(r.Points(queries, color="#202020", size=12))
    return r.Figure(
        [
            r.Panel(knn_layers, title="400 nearest neighbours"),
            r.Panel(ball_layers, title="Ball of radius 0.012"),
        ],
        camera=r.Camera(direction=(0.25, 0.25, 1.0), zoom=1.35),
    )
