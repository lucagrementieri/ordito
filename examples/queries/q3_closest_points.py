from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="Q3",
    title="Closest points on a surface",
    summary="""
    [`closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh] projects every query point
    onto the nearest point of the surface and returns that point, its distance and the face it
    lies on. Here it runs on random points around the bunny (the box comes from
    [`aabb`][ordito.bounds.aabb]) and on a dense grid in a plane through it, whose distances
    form the unsigned distance field of the surface.
    """,
    credits=(
        ("trimesh: nearest", "https://github.com/mikedh/trimesh/blob/main/examples/nearest.ipynb"),
        (
            "Open3D: distance queries",
            "https://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    lower, upper = od.bounds.aabb(vertices)
    lower, upper = np.array(lower), np.array(upper)
    margin = 0.15 * (upper - lower)

    rng = np.random.default_rng(4)
    queries = rng.uniform(lower - margin, upper + margin, (300, 3))
    queries = wp.array(queries, dtype=wp.vec3, device=device)
    closest, distance, _ = od.proximity.closest_point_on_mesh(vertices, faces, queries)

    # The same query on a 400 x 400 grid in the plane z = mid-depth: an unsigned distance field.
    x, y = np.meshgrid(
        np.linspace(lower[0] - margin[0], upper[0] + margin[0], 400),
        np.linspace(lower[1] - margin[1], upper[1] + margin[1], 400),
    )
    grid = np.stack([x, y, np.full_like(x, 0.5 * (lower[2] + upper[2]))], axis=-1)
    _, field, _ = od.proximity.closest_point_on_mesh(
        vertices, faces, wp.array(grid.reshape(-1, 3), dtype=wp.vec3, device=device)
    )
    print(f"query distances: {distance.numpy().min():.4f} to {distance.numpy().max():.4f}")
    # --8<-- [end:code]
    return {
        "queries": queries.numpy(),
        "closest": closest.numpy(),
        "distance": distance.numpy(),
        "extent": (x.min(), x.max(), y.min(), y.max()),
        "field": field.numpy().reshape(x.shape),
        "z": float(grid[0, 0, 2]),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    queries, closest, distance = result["queries"], result["closest"], result["distance"]
    segments = np.stack([queries, closest], axis=1)
    field, extent = result["field"], result["extent"]

    def draw_field(ax: Any) -> None:
        ax.imshow(field, origin="lower", extent=extent, cmap=r.striped("viridis", bands=16))
        ax.contour(field, levels=[2e-4], origin="lower", extent=extent, colors=r.RED)
        ax.set_xticks([])
        ax.set_yticks([])

    bounds = np.stack([queries.min(axis=0), queries.max(axis=0)])
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN),
                    r.Segments(segments, color=r.GREY, width=1.2),
                    r.Points(
                        queries, scalars=distance, cmap="viridis", scalar_bar="distance", size=9
                    ),
                    r.Points(closest, color=r.RED, size=7),
                ],
                title="Queries (by distance) and their closest points",
                bounds=bounds,
            ),
            r.Plot(draw_field, title=f"Distance field in the plane z = {result['z']:.3f}"),
        ],
        camera=data.camera("bunny"),
    )
