from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="X1",
    title="Plane sections and slice stacks",
    summary="""
    [`mesh_with_plane`][ordito.intersection.mesh_with_plane] intersects a mesh with a plane and
    returns the section as a set of line segments, one per crossed triangle. Twenty horizontal
    planes through the 871 000-face dragon give a stack of contours, like a slicer preparing a
    3-D print.
    """,
    credits=(
        ("trimesh: section", "https://github.com/mikedh/trimesh/blob/main/examples/section.ipynb"),
        ("PyVista: slicing", "https://docs.pyvista.org/examples/01-filter/slice"),
        (
            "PyMeshLab: planar section",
            "https://pymeshlab.readthedocs.io/en/latest/filter_list.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("dragon", device)
    (_, y_min, _), (_, y_max, _) = od.bounds.aabb(vertices)
    heights = np.linspace(y_min, y_max, 22)[1:-1]

    sections = []
    for height in heights:
        segments = od.intersection.mesh_with_plane(
            vertices, faces, wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, float(height), 0.0)
        )
        sections.append(segments.numpy())  # (m, 2) rows of segment endpoints
    print(f"{len(sections)} sections, {sum(s.shape[0] for s in sections)} segments in all")
    # --8<-- [end:code]
    return {"heights": heights, "sections": sections}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("dragon")
    sections = result["sections"]
    colors = [
        "#{:02x}{:02x}{:02x}".format(*(int(255 * c) for c in plt.get_cmap("viridis")(t)[:3]))
        for t in np.linspace(0.0, 1.0, len(sections))
    ]
    stack = [
        r.Segments(np.asarray(s).reshape(-1, 2, 3), color=c, width=1.2)
        for s, c in zip(sections, colors, strict=True)
    ]
    middle = np.asarray(sections[len(sections) // 2]).reshape(-1, 2, 3)

    def draw_section(ax: Any) -> None:
        for a, b in middle:
            ax.plot([a[0], b[0]], [a[2], b[2]], color=colors[len(sections) // 2], lw=1.0)
        ax.set_xticks([])
        ax.set_yticks([])

    return r.Figure(
        [
            r.Panel([r.Mesh(vertices, faces, color=r.GREY, opacity=0.3), *stack], title="Dragon"),
            r.Panel(
                stack,
                title="Twenty sections, seen from above",
                camera=r.Camera(direction=(0.2, 1.2, 0.8), zoom=1.2),
            ),
            r.Plot(
                draw_section, title=f"Section at y = {result['heights'][len(sections) // 2]:.3f}"
            ),
        ],
        camera=data.camera("dragon"),
    )
