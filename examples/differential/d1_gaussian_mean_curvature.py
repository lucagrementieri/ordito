from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="D1",
    title="Gaussian and mean curvature",
    summary="""
    Three curvature fields on the Stanford dragon.
    [`vertex_defects`][ordito.vertices.vertex_defects] is the pointwise angle defect, `2pi` minus
    the corner angles at a vertex, the discrete Gaussian curvature integrated over the vertex's
    neighbourhood; it is sharp and noisy on a scan.
    [`discrete_gaussian_curvature`][ordito.curvature.discrete_gaussian_curvature] and
    [`discrete_mean_curvature`][ordito.curvature.discrete_mean_curvature] are the Cohen-Steiner and
    Morvan curvature measures of a ball around each query point: the defects inside the ball, and
    the dihedral angles of the edges inside it weighted by their length. A wider ball averages over
    more surface. Colour ranges clip at the 90th percentile of the magnitude: the scan's holes and
    stray patches put a few large values in the tails.
    """,
    credits=(
        ("libigl 202", "https://libigl.github.io/tutorial/#gaussian-curvature"),
        (
            "trimesh: curvature",
            "https://github.com/mikedh/trimesh/blob/main/examples/curvature.ipynb",
        ),
        (
            "PyVista: curvature",
            "https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyDataFilters.curvature",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("dragon", device)
    angles = od.triangles.face_angles(vertices, faces)
    defects = od.vertices.vertex_defects(vertices.shape[0], faces, angles)

    radius = 4.0 * od.edges.mean_edge_length(vertices, faces)
    gaussian = od.curvature.discrete_gaussian_curvature(vertices, vertices, faces, angles, radius)
    mean = od.curvature.discrete_mean_curvature(vertices, vertices, faces, radius)
    print(f"{vertices.shape[0]} vertices, ball radius {radius:.5f}")
    print(f"median |mean curvature measure| {np.median(np.abs(mean.numpy())):.3g}")
    # --8<-- [end:code]
    return {"defects": defects.numpy(), "gaussian": gaussian.numpy(), "mean": mean.numpy()}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("dragon")
    panels = []
    for key, title in (
        ("defects", "Angle defect"),
        ("gaussian", "Gaussian curvature measure"),
        ("mean", "Mean curvature measure"),
    ):
        values = result[key]
        panels.append(
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=values,
                        cmap="RdBu_r",
                        clim=r.symmetric_clim(values, 90.0),
                        scalar_bar=key,
                    )
                ],
                title=title,
            )
        )
    return r.Figure(panels, camera=data.camera("dragon"), panel_size=(720, 520))
