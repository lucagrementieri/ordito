from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="D7",
    title="Isolines of a scalar field",
    summary="""
    [`marching_triangles_with_offsets`][ordito.intersection.marching_triangles_with_offsets] cuts
    the level set `f = c` of a per-vertex field out of every triangle and links the pieces into
    curves, returned packed: all points, the offsets where each curve starts, and whether it
    closes. [`split`][ordito.array.split] unpacks them into one array per curve. Two fields on the
    bunny: its height, whose level sets are horizontal slices, and the geodesic distance from the
    tip of an ear by [`heat_geodesic`][ordito.heat.heat_geodesic], whose level sets are geodesic
    circles.
    """,
    credits=(
        ("libigl 905", "https://libigl.github.io/tutorial/#isolines"),
        ("potpourri3d", "https://github.com/nmwsharp/potpourri3d#mesh-utilities"),
        ("PyVista: contouring", "https://docs.pyvista.org/examples/01-filter/contouring"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    height = wp.array(vertices.numpy()[:, 1], dtype=wp.float32, device=device)
    ear = wp.array([22820], dtype=wp.int32, device=device)
    distance = od.heat.heat_geodesic(vertices, faces, ear)

    curves = {}
    for name, field in (("height", height), ("distance", distance)):
        values = field.numpy()
        curves[name] = []
        for level in np.linspace(values.min(), values.max(), 22)[1:-1]:
            points, offsets, _closed = od.intersection.marching_triangles_with_offsets(
                vertices, faces, field, float(level)
            )
            curves[name] += od.array.split(points, offsets)
        print(f"{name}: {len(curves[name])} curves over 20 levels")
    # --8<-- [end:code]
    return {
        "height": height.numpy(),
        "distance": distance.numpy(),
        "curves": {k: [c.numpy() for c in v] for k, v in curves.items()},
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    panels = []
    for key, title in (("height", "Height"), ("distance", "Geodesic distance from an ear")):
        values = result[key]
        panels.append(
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=values,
                        cmap="viridis",
                        clim=(float(values.min()), float(values.max())),
                    ),
                    r.Lines(result["curves"][key], color="#303030", width=1.8),
                ],
                title=title,
            )
        )
    return r.Figure(panels, camera=data.camera("bunny"))
