from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="V7",
    title="Generalized winding number",
    summary="""
    [`winding_number`][ordito.proximity.winding_number] sums the signed solid angles that every
    triangle subtends at a query point: 1 inside a closed surface, 0 outside, and a smooth,
    still nearly binary field when the surface has holes. That makes inside/outside meaningful
    for broken input. On a slice through the holey bunny the field stays near 1 inside and only
    blurs where the slice passes near a hole. With a third of the triangles thrown away at
    random (a soup with gaps everywhere) the interior reads about 2/3 instead of 1, but it is
    still flat and well separated from the outside, so a threshold at 0.5 (red contour) still
    recovers it.
    """,
    credits=(("libigl 702", "https://libigl.github.io/tutorial/#generalized-winding-number"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("holey_bunny", device)
    rng = np.random.default_rng(0)
    triangles = faces.numpy().reshape(-1, 3)
    soup = wp.array(
        triangles[rng.random(len(triangles)) > 1 / 3].ravel(), dtype=wp.int32, device=device
    )

    # Query points on the plane z = 0, a little wider than the bunny.
    xs, ys = np.linspace(-0.11, 0.075, 280), np.linspace(0.02, 0.2, 270)
    x, y = np.meshgrid(xs, ys)
    queries = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    points = wp.array(queries, dtype=wp.vec3, device=device)

    holey = od.proximity.winding_number(vertices, faces, points).numpy().reshape(x.shape)
    sparse = od.proximity.winding_number(vertices, soup, points).numpy().reshape(x.shape)
    for name, w in (("holey bunny", holey), ("soup", sparse)):
        print(f"{name}: {(w > 0.5).mean():.1%} of the slice inside, max w {w.max():.2f}")
    # --8<-- [end:code]
    return {
        "xs": xs,
        "ys": ys,
        "holey": holey,
        "soup": sparse,
        "soup_faces": soup.numpy().reshape(-1, 3),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("holey_bunny")
    xs, ys = result["xs"], result["ys"]
    gx, gy = np.meshgrid(xs, ys)
    grid_v = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    n, m = len(ys), len(xs)
    ids = np.arange(n * m).reshape(n, m)
    quads = np.stack([ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]], axis=-1).reshape(
        -1, 4
    )
    grid_f = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])

    def slice_plot(key: str) -> r.Plot:
        def draw(ax: Any) -> None:
            extent = (xs[0], xs[-1], ys[0], ys[-1])
            image = ax.imshow(
                result[key], origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1
            )
            ax.contour(xs, ys, result[key], levels=[0.5], colors=r.RED, linewidths=1.5)
            bar = ax.figure.colorbar(image, ax=ax, fraction=0.04)
            bar.ax.tick_params(colors=r.TEXT)
            ax.set_xticks([])
            ax.set_yticks([])

        return r.Plot(draw, title=f"{'Holey bunny' if key == 'holey' else 'Soup'}: w on z = 0")

    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY, opacity=0.35),
                    r.Mesh(
                        grid_v,
                        grid_f,
                        scalars=result["holey"].ravel(),
                        clim=(0.0, 1.0),
                        smooth=False,
                    ),
                ],
                title="Slice through the holey bunny",
            ),
            slice_plot("holey"),
            r.Panel(
                [r.Mesh(vertices, result["soup_faces"], color=r.LIGHT_GREEN, smooth=False)],
                title="Soup: a third of the faces dropped",
            ),
            slice_plot("soup"),
        ],
        camera=data.camera("holey_bunny"),
        link_bounds=False,
        panel_size=(520, 480),
    )
