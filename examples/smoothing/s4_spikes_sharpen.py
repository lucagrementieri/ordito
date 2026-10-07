from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="S4",
    title="Removing spikes and sharpening detail",
    summary="""
    Two targeted filters. [`filter_spikes`][ordito.smoothing.filter_spikes] finds vertices whose
    corner angles add up to much less than a full turn (needles, a common scan artifact), moves
    only those onto the average of their neighbours, and repeats until none is left: everything
    else stays exactly where it was. [`filter_sharpen`][ordito.smoothing.filter_sharpen] is
    unsharp masking for meshes: it adds back a multiple of the difference between the surface
    and a smoothed copy of it, which exaggerates the bunny's fur-like detail.
    """,
    credits=(
        ("MeshLib: remove spikes", "https://meshlib.io/documentation/Examples.html"),
        ("PyMeshLab: unsharp mask", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import math

    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("spiky_torus", device)
    despiked, moves = od.smoothing.filter_spikes(
        vertices, faces, min_angle_sum=math.pi, return_count=True
    )
    moved = np.linalg.norm(despiked.numpy() - vertices.numpy(), axis=1) > 0
    print(f"spike filter: {moves} moves, {moved.sum()} of {moved.size} vertices changed")

    bunny, bunny_faces = data.load("bunny", device)
    sharpened = od.smoothing.filter_sharpen(bunny, bunny_faces, weight=1.0, iterations=5)
    # --8<-- [end:code]
    return {
        "spiky": vertices.numpy(),
        "despiked": despiked.numpy(),
        "moved": moved,
        "sharpened": sharpened.numpy(),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    _, torus_f = data.arrays("spiky_torus")
    bunny_v, bunny_f = data.arrays("bunny")
    head = bunny_v[11842] + np.array([0.035, 0.015, 0.0])
    close = np.stack([head - 0.035, head + 0.035])
    torus_cam = r.Camera(direction=(0.0, -0.7, 1.0), up=(0.0, 0.0, 1.0), zoom=1.35)
    bunny_cam = data.camera("bunny")
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(result["spiky"], torus_f, color=r.LIGHT_GREEN)],
                title="Spikes",
                camera=torus_cam,
            ),
            r.Panel(
                [
                    r.Mesh(result["despiked"], torus_f, color=r.GREEN),
                    r.Points(result["despiked"][result["moved"]], color=r.RED, size=10),
                ],
                title="After filter_spikes (moved in red)",
                camera=torus_cam,
            ),
            r.Panel(
                [r.Mesh(bunny_v, bunny_f, color=r.GREEN)],
                title="Bunny",
                camera=bunny_cam,
                bounds=close,
            ),
            r.Panel(
                [r.Mesh(result["sharpened"], bunny_f, color=r.GREEN)],
                title="Sharpened",
                camera=bunny_cam,
                bounds=close,
            ),
        ],
        link_bounds=False,
        panel_size=(540, 500),
    )
