from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC6",
    title="Interpolating sparse samples onto a surface",
    summary="""
    A field measured at only 60 probe points, spread over the bunny by
    [`farthest_point_sample`][ordito.points.farthest_point_sample], is carried onto every vertex
    by [`interpolate_from_points`][ordito.interpolation.interpolate_from_points]: a
    Gaussian-weighted mean of the nearby probes. With a radius footprint, vertices farther than
    the radius from every probe get the `null_value` (grey); with `k` nearest probes every vertex
    gets a value, however far its probes are.
    """,
    credits=(("PyVista: interpolate", "https://docs.pyvista.org/examples/01-filter/interpolate"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, _ = data.load("bunny", device)
    probes = vertices.numpy()[od.points.farthest_point_sample(vertices, 60).numpy()]
    readings = np.sin(40.0 * probes[:, 0]) + np.cos(40.0 * probes[:, 1])  # the "measurement"

    source = wp.array(probes, dtype=wp.vec3, device=device)
    values = wp.array(readings, dtype=wp.float32, device=device)
    by_radius = od.interpolation.interpolate_from_points(
        source, values, vertices, radius=0.015, null_value=float("nan")
    )
    by_neighbours = od.interpolation.interpolate_from_points(
        source, values, vertices, radius=0.015, k=4
    )
    print(f"vertices with no probe within the radius: {np.isnan(by_radius.numpy()).sum()}")
    # --8<-- [end:code]
    truth = np.sin(40.0 * vertices.numpy()[:, 0]) + np.cos(40.0 * vertices.numpy()[:, 1])
    return {
        "probes": probes,
        "readings": readings,
        "by_radius": by_radius.numpy(),
        "by_neighbours": by_neighbours.numpy(),
        "truth": truth,
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    clim = (float(result["truth"].min()), float(result["truth"].max()))

    def field(values: np.ndarray, title: str, bar: str | None = None) -> r.Panel:
        missing = np.isnan(values)
        layers: list[r.Layer] = [
            r.Mesh(
                vertices,
                faces,
                scalars=np.where(missing, clim[0], values),
                cmap="coolwarm",
                clim=clim,
                scalar_bar=bar,
            )
        ]
        if missing.any():
            gaps = missing[faces].any(axis=1)
            layers.append(r.Mesh(vertices, faces[gaps], color=r.GREY))
        layers.append(r.Points(result["probes"], color="#202020", size=8))
        return r.Panel(layers, title=title)

    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY),
                    r.Points(
                        result["probes"],
                        scalars=result["readings"],
                        cmap="coolwarm",
                        clim=clim,
                        size=16,
                    ),
                ],
                title="60 probe readings",
            ),
            field(result["by_radius"], "Within a radius"),
            field(result["by_neighbours"], "4 nearest probes"),
            field(result["truth"], "The field the probes sampled", bar="value"),
        ],
        camera=data.camera("bunny"),
        ncols=4,
        panel_size=(560, 520),
    )
