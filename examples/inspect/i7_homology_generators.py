from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="I7",
    title="Homology generators: handles and tunnels",
    summary="""
    [`homology_generators`][ordito.homology.homology_generators] returns a basis of the loops
    that cannot be shrunk to a point on a closed surface: two per handle, so a torus has 2 and the
    slab with nine tunnels has 18. The basis comes from a tree-cotree construction, so its loops
    follow the spanning trees and wander; [`shorten_loop`][ordito.geodesic_walk.shorten_loop]
    pulls each one tighter along the mesh edges without changing its homotopy class. A basis
    loop that winds around several tunnels at once stays long: shortening cannot turn it into a
    loop around one tunnel, because that is a different class.
    """,
    credits=(("MeshLib: tunnel detection", "https://meshlib.io/documentation/index.html"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    torus_vertices, torus_faces = data.load("torus", device)
    torus_loops = od.homology.homology_generators(torus_vertices, torus_faces)
    print(f"torus: {len(torus_loops)} generators (genus {len(torus_loops) // 2})")

    vertices, faces = data.load("handles", device)
    loops = od.homology.homology_generators(vertices, faces)
    short_loops, sweeps = od.geodesic_walk.shorten_loop(vertices, faces, loops, max_iter=500)
    print(f"slab: {len(loops)} generators (genus {len(loops) // 2})")

    def length(loop: wp.array[wp.int32]) -> float:  # closed polyline length
        points = vertices.numpy()[loop.numpy()]
        return np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1).sum()

    before, after = sum(map(length, loops)), sum(map(length, short_loops))
    print(f"total loop length {before:.1f} before shortening, {after:.1f} after ({sweeps} sweeps)")
    # --8<-- [end:code]
    return {
        "torus": [loop.numpy() for loop in torus_loops],
        "loops": [loop.numpy() for loop in loops],
        "short": [loop.numpy() for loop in short_loops],
    }


def _lines(vertices: np.ndarray, loops: list[np.ndarray]) -> list[r.Lines]:
    colors = plt.get_cmap("tab20")(np.arange(len(loops)) % 20)
    return [
        r.Lines(
            [vertices[loop]],
            closed=True,
            width=4.0,
            color="#{:02x}{:02x}{:02x}".format(*(int(255 * c) for c in color[:3])),
        )
        for loop, color in zip(loops, colors, strict=True)
    ]


def figure(result: dict[str, Any]) -> r.Figure:
    torus_vertices, torus_faces = data.arrays("torus")
    vertices, faces = data.arrays("handles")
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(torus_vertices, torus_faces, color=r.GREY),
                    *_lines(torus_vertices, result["torus"]),
                ],
                title="Torus: 2 generators",
                camera=data.camera("torus"),
            ),
            r.Panel(
                [r.Mesh(vertices, faces, color=r.GREY), *_lines(vertices, result["loops"])],
                title="Genus 9: 18 generators",
                camera=data.camera("handles"),
            ),
            r.Panel(
                [r.Mesh(vertices, faces, color=r.GREY), *_lines(vertices, result["short"])],
                title="Shortened",
                camera=data.camera("handles"),
            ),
        ],
        link_bounds=False,
    )
