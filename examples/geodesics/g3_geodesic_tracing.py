from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="G3",
    title="Tracing straightest geodesics (the exponential map)",
    summary="""
    [`trace_from_vertex`][ordito.geodesic_walk.trace_from_vertex] walks a batch of rays across the
    surface, each starting at a vertex in a tangent direction and going straight ahead through
    every face it enters, for an arc length equal to the direction's length. A fan of equal-length
    rays in every direction is the image of a disk under the exponential map: on the torus the
    rays along the outer equator stay together while those climbing over the tube spread and wind
    around it. The tangent directions are built in each vertex's frame from
    [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames].
    """,
    notes="""
    A ray that runs exactly into a vertex stops there (the walk has no rule for which face to
    continue into). On the torus's regular grid that happens to rays leaving along an edge, so
    the fan starts half a step off the frame's axes.
    """,
    credits=(("potpourri3d: geodesic tracer", "https://github.com/nmwsharp/potpourri3d"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    def fan(name: str, start: int, length: float, n_rays: int = 32):
        vertices, faces = data.load(name, device)
        basis_x, basis_y, _ = od.tangent_space.vertex_tangent_frames(vertices, faces)
        angle = (np.arange(n_rays)[:, None] + 0.5) * (2.0 * np.pi / n_rays)
        directions = length * (
            np.cos(angle) * basis_x.numpy()[start] + np.sin(angle) * basis_y.numpy()[start]
        )
        points, offsets = od.geodesic_walk.trace_from_vertex(
            vertices,
            faces,
            wp.full(n_rays, start, dtype=wp.int32, device=device),
            wp.array(directions, dtype=wp.vec3, device=device),
        )
        return od.array.split(points, offsets)

    torus_rays = fan("torus", start=0, length=4.0, n_rays=24)  # a vertex on the outer equator
    bunny_rays = fan("bunny", start=16151, length=0.1)  # a vertex on the back
    lengths = [od.polyline.polyline_length(ray) for ray in torus_rays]
    print(f"torus ray lengths: {min(lengths):.3f} to {max(lengths):.3f}")
    print(f"edge crossings per torus ray: up to {max(len(ray) for ray in torus_rays) - 2}")
    # --8<-- [end:code]
    return {"torus": [p.numpy() for p in torus_rays], "bunny": [p.numpy() for p in bunny_rays]}


def _colored(rays: list[np.ndarray]) -> list[r.Lines]:
    colors = plt.get_cmap("hsv")(np.arange(len(rays)) / len(rays))
    return [
        r.Lines([ray], color=to_hex(color), width=2.5)
        for ray, color in zip(rays, colors, strict=True)
    ]


def figure(result: dict[str, Any]) -> r.Figure:
    torus_v, torus_f = data.arrays("torus")
    bunny_v, bunny_f = data.arrays("bunny")
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(torus_v, torus_f, color=r.LIGHT_GREEN),
                    *_colored(result["torus"]),
                    r.Points(torus_v[[0]], color=r.RED, size=16),
                ],
                title="Torus: 24 rays of length 4",
                camera=r.Camera(direction=(0.6, -0.5, 0.9), up=(0.0, 0.0, 1.0), zoom=1.1),
            ),
            r.Panel(
                [
                    r.Mesh(bunny_v, bunny_f, color=r.LIGHT_GREEN),
                    *_colored(result["bunny"]),
                    r.Points(bunny_v[[16151]], color=r.RED, size=16),
                ],
                title="Bunny: 32 rays from the back",
                camera=r.Camera(direction=(0.5, 0.9, 0.7), up=(0.0, 1.0, 0.0), zoom=1.1),
            ),
        ],
        link_bounds=False,
    )
