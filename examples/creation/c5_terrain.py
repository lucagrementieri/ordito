from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="C5",
    title="Terrain triangulation",
    summary="""
    Scattered survey points of a height field are triangulated in the plane by
    [`delaunay_triangulation`][ordito.reconstruction.delaunay_triangulation], which returns the
    triangulation that maximizes the smallest angle. Its faces index the input points, so lifting
    them back to their heights turns the 2-D triangulation into the terrain surface (a 2.5-D
    mesh).
    """,
    credits=(
        ("PyVista: delaunay_2d", "https://docs.pyvista.org/examples/00-load/create_tri_surface"),
        (
            "MeshLib: terrain triangulation",
            "https://meshlib.io/documentation/ExampleTerrainTriangulation.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    points = data.load_points("terrain_points", device)
    plan = wp.array(points.numpy()[:, :2], dtype=wp.vec2, device=device)
    faces = od.reconstruction.delaunay_triangulation(plan)
    terrain = od.Trimesh(points, faces)
    print(f"{points.shape[0]} points -> {terrain.n_faces} triangles")
    print(f"boundary loops: {len(terrain.boundary_loops)}, Euler {terrain.euler_characteristic}")
    # --8<-- [end:code]
    return {"points": points.numpy(), "faces": faces.numpy()}


def figure(result: dict[str, Any]) -> r.Figure:
    points, faces = result["points"], result["faces"]
    height = points[:, 2]
    flat = points.copy()
    flat[:, 2] = height.min()
    clim = (float(height.min()), float(height.max()))
    return r.Figure(
        [
            r.Panel(
                [r.Points(points, scalars=height, cmap="terrain", clim=clim, size=5)],
                title="Survey points, by height",
            ),
            r.Panel(
                [
                    r.Mesh(flat, faces, color=r.LIGHT_GREEN, show_edges=True, line_width=0.8),
                    r.Points(flat, color=r.RED, size=4),
                ],
                title="Delaunay triangulation in the plane (close-up)",
                bounds=np.array([[-10.0, -10.0, flat[0, 2]], [-4.0, -4.0, flat[0, 2]]]),
                camera=r.Camera(direction=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0), zoom=0.9),
            ),
            r.Panel(
                [r.Mesh(points, faces, scalars=height, cmap="terrain", clim=clim, smooth=True)],
                title="Lifted to the heights",
            ),
        ],
        camera=data.camera("terrain_points"),
        panel_size=(600, 520),
    )
