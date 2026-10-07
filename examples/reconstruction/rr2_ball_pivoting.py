from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="RR2",
    title="Ball pivoting",
    summary="""
    [`ball_pivoting`][ordito.reconstruction.ball_pivoting] rolls a ball of fixed radius over the
    cloud: every three points the ball can rest on without containing another become a triangle,
    and the ball pivots across each new edge to find the next. Unlike Poisson reconstruction it
    *interpolates* the points (every vertex is an input point) and leaves holes where the
    sampling is sparser than the ball, which
    [`boundary_loops`][ordito.boundary.boundary_loops] finds (red).
    """,
    credits=(
        (
            "Open3D: ball pivoting",
            "https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html",
        ),
        ("PyMeshLab: ball pivoting", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    points = data.load_points("bunny_cloud", device)
    normals = wp.array(data.cloud_normals("bunny_cloud"), dtype=wp.vec3, device=device)
    # clustering=0 keeps candidates near an edge's ends: a random sample has many close pairs.
    vertices, faces = od.reconstruction.ball_pivoting(points, normals, radius=0.003, clustering=0.0)

    holes = od.boundary.boundary_loops(vertices, faces)
    print(f"{faces.shape[0] // 3} triangles over {points.shape[0]} points")
    print(f"{len(holes)} boundary loops left open")
    # --8<-- [end:code]
    return {
        "points": points.numpy(),
        "vertices": vertices.numpy(),
        "faces": faces.numpy().reshape(-1, 3),
        "holes": [hole.numpy() for hole in holes],
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = result["vertices"], result["faces"]
    rims = [vertices[hole] for hole in result["holes"]]
    center = np.array([-0.021, 0.114, 0.037])  # a hole on the bunny's chest
    close = np.stack([center - 0.008, center + 0.008])
    return r.Figure(
        [
            r.Panel([r.Points(result["points"], color=r.GREY, size=3)], title="Cloud"),
            r.Panel(
                [r.Mesh(vertices, faces), r.Lines(rims, closed=True, width=2.5)],
                title="Ball-pivoting mesh, holes in red",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, show_edges=True, line_width=0.5, smooth=False),
                    r.Lines(rims, closed=True, width=4),
                ],
                title="Close-up",
                bounds=close,
            ),
        ],
        camera=data.camera("bunny_cloud"),
    )
