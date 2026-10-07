from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="V2",
    title="Signed distance field and its level sets",
    summary="""
    [`signed_distance_grid`][ordito.proximity.signed_distance_grid] samples the signed distance
    to the bunny on a regular lattice (negative inside; the winding-number sign copes with the
    open base). One slice of the field is shown on the left with its zero contour, and
    [`marching_cubes`][ordito.levelset.marching_cubes] extracts three of its level sets: inside,
    on and outside the surface.
    """,
    credits=(
        ("libigl 704", "https://libigl.github.io/tutorial/#signed-distances"),
        (
            "Open3D: distance queries",
            "https://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html",
        ),
        ("MeshLib: signed distance", "https://meshlib.io/documentation/ExampleSignedDistance.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    field, bounds = od.proximity.signed_distance_grid(
        vertices, faces, voxel_size=0.0015, pad=8, sign_mode="winding"
    )
    distances = field.numpy()
    print(f"lattice {distances.shape}, from {distances.min():.4f} to {distances.max():.4f}")

    levels = {}
    for iso in (-0.004, 0.0, 0.008):
        levels[iso] = od.levelset.marching_cubes(field, iso, bounds=bounds)
        print(f"iso {iso:+.3f}: {levels[iso][1].shape[0] // 3} faces")
    # --8<-- [end:code]
    lower, upper = (np.array(b) for b in bounds)
    return {
        "field": field.numpy(),
        "lower": lower,
        "upper": upper,
        "levels": {iso: (v.numpy(), f.numpy().reshape(-1, 3)) for iso, (v, f) in levels.items()},
    }


def figure(result: dict[str, Any]) -> r.Figure:
    field, lower, upper = result["field"], result["lower"], result["upper"]
    k = field.shape[2] // 2
    sliced = field[:, :, k].T  # rows y, columns x

    colors = {-0.004: r.BLUE, 0.0: r.GREEN, 0.008: r.ORANGE}

    def draw(ax: Any) -> None:
        extent = (lower[0], upper[0], lower[1], upper[1])
        bound = float(np.abs(sliced).max())
        image = ax.imshow(
            sliced, origin="lower", extent=extent, cmap="RdBu_r", vmin=-bound, vmax=bound
        )
        for iso, color in colors.items():
            ax.contour(
                sliced, levels=[iso], origin="lower", extent=extent, colors=color, linewidths=2.0
            )
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.04)
        bar.ax.tick_params(colors=r.TEXT)
        ax.set_xticks([])
        ax.set_yticks([])

    panels: list[r.Panel | r.Plot] = [r.Plot(draw, title="Middle slice and the three contours")]
    outer = result["levels"][0.008][0]
    frame = np.stack([outer.min(axis=0), outer.max(axis=0)])
    for iso, (v, f) in result["levels"].items():
        panels.append(
            r.Panel([r.Mesh(v, f, color=colors[iso])], title=f"iso = {iso:+.3f}", bounds=frame)
        )
    return r.Figure(panels, camera=data.camera("bunny"), link_bounds=False, panel_size=(520, 500))
