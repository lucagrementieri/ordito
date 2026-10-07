from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="C4",
    title="Polygon triangulation",
    summary="""
    [`triangulate_polygon`][ordito.polyline.triangulate_polygon] fills a simple 2-D polygon by ear
    clipping on the device, using only the polygon's own vertices: an `n`-gon becomes `n - 2`
    triangles. [`extrude_triangulation`][ordito.creation.extrude_triangulation] raises the
    triangulation into a watertight solid, and
    [`polyline_triangulate`][ordito.polyline.polyline_triangulate] does the same filling for a
    closed planar loop in 3-D, whatever plane it lies in. Interior rings (holes) are not
    supported, so the gear's bore is left out.
    """,
    credits=(
        ("libigl 604", "https://libigl.github.io/tutorial/#triangulation-of-closed-polygons"),
        (
            "MeshLib: contour triangulation",
            "https://meshlib.io/documentation/ExampleTriangulation.html",
        ),
        ("trimesh: creation", "https://trimesh.org/trimesh.creation.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od

    # A 16-tooth gear outline, counter-clockwise.
    teeth, steps = 16, np.array([0.0, 0.18, 0.5, 0.68])
    angle = 2 * np.pi * (np.arange(teeth)[:, None] + steps).ravel() / teeth
    radius = np.tile([0.8, 1.0, 1.0, 0.8], teeth)
    outline = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)

    ring, faces = od.polyline.triangulate_polygon(wp.array(outline, dtype=wp.vec2, device=device))
    print(f"{ring.shape[0]} vertices -> {faces.shape[0] // 3} triangles")
    solid = od.creation.extrude_triangulation(ring, faces, height=0.3)
    print(f"extruded: watertight {od.validation.is_watertight(*solid)}")

    # The same outline on a tilted plane in 3-D.
    tilt = od.transform.matrix_to_numpy(od.transform.rotation_matrix((1.0, 1.0, 0.0), 0.9))[:3, :3]
    loop = np.c_[outline, np.zeros(len(outline))] @ tilt.T
    loop_faces = od.polyline.polyline_triangulate(wp.array(loop, dtype=wp.vec3, device=device))
    print(f"tilted loop: {loop_faces.shape[0]} triangles")
    # --8<-- [end:code]
    return {
        "outline": outline,
        "faces": faces.numpy().reshape(-1, 3),
        "solid": (solid[0].numpy(), solid[1].numpy()),
        "loop": loop,
        "loop_faces": loop_faces.numpy(),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    outline, faces = result["outline"], result["faces"]
    closed = np.vstack([outline, outline[:1]])

    def draw_outline(ax: Any) -> None:
        ax.fill(closed[:, 0], closed[:, 1], color=r.LIGHT_GREEN)
        ax.plot(closed[:, 0], closed[:, 1], color=r.RED, lw=2)
        ax.plot(outline[:, 0], outline[:, 1], "o", color=r.RED, ms=3)
        ax.axis("off")

    def draw_triangles(ax: Any) -> None:
        ax.tripcolor(
            outline[:, 0],
            outline[:, 1],
            faces,
            facecolors=np.random.default_rng(0).random(len(faces)),
            cmap="YlGn",
            edgecolors="#303030",
            lw=0.6,
        )
        ax.plot(closed[:, 0], closed[:, 1], color=r.RED, lw=2)
        ax.axis("off")

    loop = result["loop"]
    return r.Figure(
        [
            r.Plot(draw_outline, title="Outline (64 vertices)"),
            r.Plot(draw_triangles, title="triangulate_polygon"),
            r.Panel(
                [r.Mesh(*result["solid"], show_edges=True, smooth=False, line_width=0.6)],
                title="extrude_triangulation",
                camera=r.Camera(direction=(0.6, -1.0, 1.3), up=(0.0, 0.0, 1.0)),
            ),
            r.Panel(
                [
                    r.Mesh(loop, result["loop_faces"], show_edges=True, smooth=False),
                    r.Lines([loop], closed=True, width=4.0),
                ],
                title="polyline_triangulate (tilted, 3-D)",
                camera=r.Camera(direction=(1.0, -1.0, 0.1), up=(0.0, 0.0, 1.0)),
            ),
        ],
        link_bounds=False,
        panel_size=(520, 520),
    )
