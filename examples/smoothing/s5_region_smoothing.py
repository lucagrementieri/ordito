from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

CENTER = 5088

META = Meta(
    id="S5",
    title="Fairing a region and smoothing its rim",
    summary="""
    A patch of the bunny's flank is selected (the vertices near a point, grown by two rings with
    [`expand_vertex_mask`][ordito.selection.expand_vertex_mask]).
    [`smooth_region`][ordito.smoothing.smooth_region] then repositions only the selected vertices
    so that the surface is as smooth as possible *including across the rim*: the bumps vanish and
    the patch blends tangentially into the fixed surface around it.
    The rim of a selection follows triangle edges and zigzags;
    [`smooth_region_boundary`][ordito.smoothing.smooth_region_boundary] slides the vertices on it
    along the surface onto a smooth curve without changing the selection. The rim is drawn from
    [`region_boundary_edges`][ordito.selection.region_boundary_edges].
    """,
    credits=(
        (
            "libigl 401: surface fairing",
            "https://libigl.github.io/tutorial/#biharmonic-deformation",
        ),
        ("MeshLib: smoothRegionBoundary", "https://meshlib.io/documentation/Examples.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    points = vertices.numpy()
    near = np.linalg.norm(points - points[5088], axis=1) < 0.03  # a point on the flank
    selected = od.selection.expand_vertex_mask(faces, wp.array(near, device=device), hops=2)
    faired = od.smoothing.smooth_region(vertices, faces, selected)

    # The face region the selection covers, and its rim before and after smoothing it.
    region = wp.array(np.all(selected.numpy()[faces.numpy().reshape(-1, 3)], axis=1), device=device)
    smooth_rim = od.smoothing.smooth_region_boundary(faired, faces, region, iterations=8)
    rim = od.selection.region_boundary_edges(faces, region)
    moved = np.linalg.norm(faired.numpy() - points, axis=1)
    print(f"{selected.numpy().sum()} free vertices, largest move {moved.max():.4f}")
    print(f"{rim.shape[0]} rim edges")
    # --8<-- [end:code]
    return {
        "selected": selected.numpy(),
        "region": region.numpy(),
        "faired": faired.numpy(),
        "smooth_rim": smooth_rim.numpy(),
        "rim": rim.numpy(),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    colors = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (faces.shape[0], 1))
    colors[result["region"]] = (0xF3, 0x9C, 0x12)
    rim = result["rim"]
    center = vertices[CENTER]
    wide = np.stack([center - 0.06, center + 0.06])
    on_rim = result["faired"][rim[np.argmax(result["faired"][rim[:, 0], 0]), 0]]
    close = np.stack([on_rim - 0.007, on_rim + 0.007])
    camera = r.Camera(direction=(0.4, 0.15, 1.0), up=(0.0, 1.0, 0.0))

    def view(positions: np.ndarray, title: str, bounds: np.ndarray, edges: bool) -> r.Panel:
        return r.Panel(
            [
                r.Mesh(positions, faces, face_colors=colors, show_edges=edges, line_width=0.3),
                r.Segments(positions[rim], color=r.RED, width=3.0),
            ],
            title=title,
            bounds=bounds,
        )

    return r.Figure(
        [
            view(vertices, "Selection", wide, False),
            view(result["faired"], "smooth_region", wide, False),
            view(result["faired"], "Rim after smooth_region (close-up)", close, True),
            view(result["smooth_rim"], "After smooth_region_boundary", close, True),
        ],
        camera=camera,
        link_bounds=False,
        panel_size=(540, 520),
    )
