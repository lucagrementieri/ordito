from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import to_rgb

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="G5",
    title="Extending values from a few points (geodesic Voronoi cells)",
    summary="""
    [`extend_scalar`][ordito.heat.extend_scalar] spreads values given at a few source vertices
    over the whole surface: every vertex takes the value of its geodesically nearest source, with
    a short smooth blend where two sources meet. It diffuses the values and an indicator of the
    sources for the same short time and divides one by the other. Extending each source's
    indicator in turn and keeping the largest partitions the surface into geodesic Voronoi cells;
    the [`heat_operators`][ordito.heat.heat_operators] are built once for all eight. The
    sources are spread with [`farthest_point_sample`][ordito.points.farthest_point_sample].
    """,
    credits=(("potpourri3d: extend scalar", "https://github.com/nmwsharp/potpourri3d"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    operators = od.heat.heat_operators(vertices, faces)
    sources = od.points.farthest_point_sample(vertices, 8)

    # Extend each source's indicator (1 there, 0 at the others); the largest one wins.
    weights = np.stack(
        [
            od.heat.extend_scalar(
                vertices, faces, sources, wp.array(np.eye(8)[k], device=device), operators=operators
            ).numpy()
            for k in range(8)
        ]
    )
    cells = weights.argmax(axis=0)
    print(f"sources: {sources.numpy().tolist()}")
    print(f"vertices per cell: {np.bincount(cells, minlength=8).tolist()}")
    print(f"weights sum to 1 within {np.abs(weights.sum(axis=0) - 1).max():.1e}")
    # --8<-- [end:code]
    return {"sources": sources.numpy(), "weights": weights, "cells": cells}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    palette = [
        "#76b900",
        "#2c7be8",
        "#f39c12",
        "#c2189b",
        "#e8412c",
        "#17becf",
        "#8c564b",
        "#f7d038",
    ]
    rgb = (np.array([to_rgb(c) for c in palette]) * 255).astype(np.uint8)
    face_cells = result["cells"][faces[:, 0]]  # one corner's cell: hard borders along edges
    seeds = r.Points(vertices[result["sources"]], color="#202020", size=16)
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(vertices, faces, face_colors=rgb[face_cells]), seeds],
                title="Geodesic Voronoi cells",
            ),
            r.Panel(
                [r.Mesh(vertices, faces, face_colors=rgb[face_cells]), seeds],
                title="Cells (back)",
                camera=r.Camera(direction=(-0.4, 0.3, -1.0), up=(0.0, 1.0, 0.0), zoom=1.15),
            ),
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=result["weights"][int(result["cells"][5088])],
                        cmap="viridis",
                        clim=(0.0, 1.0),
                        scalar_bar="weight",
                    ),
                    seeds,
                ],
                title="One source's extended indicator",
            ),
        ],
        camera=data.camera("bunny"),
    )
