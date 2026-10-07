from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="G8",
    title="Shortening non-contractible loops",
    summary="""
    [`homology_generators`][ordito.homology.homology_generators] returns a basis of the loops
    that cannot be shrunk to a point: two per handle. They come out of a spanning-tree
    construction, so they are long and jagged.
    [`shorten_loop`][ordito.geodesic_walk.shorten_loop] then shortens each one without changing
    which way it wraps around the surface, rerouting it one vertex at a time through the shorter
    side of the vertex's ring, until it settles on a short loop along the mesh edges.
    """,
    notes="""
    The shortened loops stay on mesh edges, so they are locally shortest *edge* paths rather than
    true geodesics. On the torus's regular grid the loop around the hole cannot step off a row of
    edges without first getting longer, which is why it does not settle exactly on the inner
    equator.
    """,
    credits=(("potpourri3d: geodesic loops", "https://github.com/nmwsharp/potpourri3d"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    result = {}

    def total_length(vertices: wp.array[wp.vec3], loops: list[wp.array[wp.int32]]) -> float:
        points = [od.array.gather(vertices, loop) for loop in loops]
        return sum(od.polyline.polyline_length(p, closed=True) for p in points)

    for name in ("torus", "handles"):
        vertices, faces = data.load(name, device)
        loops = od.homology.homology_generators(vertices, faces)
        short, sweeps = od.geodesic_walk.shorten_loop(vertices, faces, loops, max_iter=1000)
        before, after = total_length(vertices, loops), total_length(vertices, short)
        print(f"{name}: {len(loops)} loops, length {before:.2f} -> {after:.2f} ({sweeps} sweeps)")
        result[name] = ([loop.numpy() for loop in loops], [loop.numpy() for loop in short])
    # --8<-- [end:code]
    return result


def _loops(vertices: np.ndarray, loops: list[np.ndarray]) -> list[r.Lines]:
    colors = plt.get_cmap("tab10")(np.arange(len(loops)) % 10)
    return [
        r.Lines([vertices[loop]], color=to_hex(color), width=4.0, closed=True)
        for loop, color in zip(loops, colors, strict=True)
    ]


def figure(result: dict[str, Any]) -> r.Figure:
    torus_v, torus_f = data.arrays("torus")
    handles_v, handles_f = data.arrays("handles")
    before, after = result["torus"]
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(torus_v, torus_f, color=r.LIGHT_GREEN), *_loops(torus_v, before)],
                title="Torus: generators",
                camera=r.Camera(direction=(0.0, -0.3, 1.0), up=(0.0, 1.0, 0.0), zoom=1.1),
            ),
            r.Panel(
                [r.Mesh(torus_v, torus_f, color=r.LIGHT_GREEN), *_loops(torus_v, after)],
                title="Torus: shortened",
                camera=r.Camera(direction=(0.0, -0.3, 1.0), up=(0.0, 1.0, 0.0), zoom=1.1),
            ),
            r.Panel(
                [
                    r.Mesh(handles_v, handles_f, color=r.LIGHT_GREEN),
                    *_loops(handles_v, result["handles"][1]),
                ],
                title="Genus 9: 18 shortened loops",
                camera=data.camera("handles"),
            ),
        ],
        link_bounds=False,
    )
