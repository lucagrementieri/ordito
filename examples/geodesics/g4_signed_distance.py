from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="G4",
    title="Signed distance to curves on the surface",
    summary="""
    [`heat_signed_distance`][ordito.heat.heat_signed_distance] measures the distance along the
    surface to a set of closed, oriented curves, positive on one side and negative on the other.
    It diffuses the curves' normals with the vector heat method and integrates the normalized
    result, so it needs no inside/outside test and handles several curves at once. The curves
    here are the rims of geodesic disks: [`heat_geodesic`][ordito.heat.heat_geodesic] gives the
    distance to a centre and [`boundary_loops`][ordito.boundary.boundary_loops] of the faces
    within a radius orders the rim.
    """,
    credits=(("potpourri3d: signed heat method", "https://github.com/nmwsharp/potpourri3d"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    triangles = faces.numpy().reshape(-1, 3)

    def rim(center: int, radius: float) -> np.ndarray:
        """Return the boundary loop of the geodesic disk of ``radius`` around ``center``."""
        source = wp.array([center], dtype=wp.int32, device=device)
        distance = od.heat.heat_geodesic(vertices, faces, source).numpy()
        disk = triangles[(distance[triangles] < radius).all(axis=1)].ravel()
        loops = od.boundary.boundary_loops(vertices, wp.array(disk, device=device))
        return max(loops, key=lambda loop: loop.shape[0]).numpy()

    body, cheek = rim(5088, 0.05), rim(6710, 0.02)
    one = od.heat.heat_signed_distance(vertices, faces, wp.array(body, device=device))
    both = od.heat.heat_signed_distance(
        vertices,
        faces,
        wp.array(np.concatenate([body, cheek]), device=device),
        curve_offsets=wp.array(
            [0, body.size, body.size + cheek.size], dtype=wp.int32, device=device
        ),
    )
    print(f"one curve: {one.numpy().min():.4f} to {one.numpy().max():.4f}")
    print(f"value on the curve: |d| <= {np.abs(one.numpy()[body]).max():.1e}")
    # --8<-- [end:code]
    return {"curves": [body, cheek], "one": one.numpy(), "both": both.numpy()}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    clim = r.symmetric_clim(result["both"], 100.0)
    cmap = r.striped("RdBu_r", bands=20, dark=0.85)
    panels = []
    for key, curves, title in (
        ("one", result["curves"][:1], "One curve"),
        ("both", result["curves"], "Two curves"),
    ):
        panels.append(
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=result[key],
                        cmap=cmap,
                        clim=clim,
                        scalar_bar="signed distance",
                    ),
                    r.Lines([vertices[c] for c in curves], color="#202020", width=3.0, closed=True),
                ],
                title=title,
            )
        )
    return r.Figure(panels, camera=data.camera("bunny"))
