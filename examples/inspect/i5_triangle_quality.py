from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="I5",
    title="Triangle quality",
    summary="""
    [`face_quality`][ordito.triangles.face_quality] scores the shape of every triangle of the bunny
    scan. The aspect ratio (circumradius over twice the inradius) is 1 for an equilateral triangle
    and grows without bound for a sliver; the mean ratio and the radius ratio run the other way,
    from 0 for a degenerate triangle to 1. The scan is mostly well shaped, with a scatter of
    slivers where the scanner's patches were stitched.
    """,
    credits=(
        ("PyVista: mesh quality", "https://docs.pyvista.org/examples/01-filter/mesh_quality"),
        ("PyMeshLab: filters", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
        ("libigl 701", "https://libigl.github.io/tutorial/#statistics"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    aspect = od.triangles.face_quality(vertices, faces, "aspect_ratio").numpy()
    mean_ratio = od.triangles.face_quality(vertices, faces, "mean_ratio").numpy()
    radius_ratio = od.triangles.face_quality(vertices, faces, "radius_ratio").numpy()
    print(f"aspect ratio: median {np.median(aspect):.3f}, worst {aspect.max():.1f}")
    print(f"{(aspect > 4).sum()} of {aspect.size} faces have an aspect ratio above 4")
    # --8<-- [end:code]
    return {"aspect": aspect, "mean_ratio": mean_ratio, "radius_ratio": radius_ratio}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    aspect = result["aspect"]
    bad = aspect > 4.0
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=aspect,
                        cmap="magma_r",
                        clim=(1.0, 2.5),
                        scalar_bar="aspect ratio",
                        smooth=False,
                    )
                ],
                title="Aspect ratio",
            ),
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=result["mean_ratio"],
                        cmap="viridis",
                        clim=(0.3, 1.0),
                        scalar_bar="mean ratio",
                        smooth=False,
                    )
                ],
                title="Mean ratio",
            ),
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=result["radius_ratio"],
                        cmap="viridis",
                        clim=(0.3, 1.0),
                        scalar_bar="radius ratio",
                        smooth=False,
                    )
                ],
                title="Radius ratio",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY),
                    r.Points(vertices[faces[bad]].mean(axis=1), color=r.RED, size=9),
                ],
                title="Aspect ratio > 4",
            ),
        ],
        camera=data.camera("bunny"),
        panel_size=(560, 540),
    )
