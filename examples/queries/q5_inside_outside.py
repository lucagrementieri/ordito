from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="Q5",
    title="Inside / outside classification",
    summary="""
    Which points of a grid lie inside the bunny? The bunny here has holes punched in its side, so
    "inside" is ill-posed for a ray test:
    [`contains_points`][ordito.ray.contains_points] votes over the parity of a few rays, and a ray
    that escapes through a hole flips its vote. The generalized winding number,
    [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh] with
    `sign_mode="winding"`, degrades gracefully instead: it counts how many times the surface wraps
    around a point, and a small hole only changes that count slightly.
    """,
    credits=(
        (
            "PyVista: extract cells inside a surface",
            "https://docs.pyvista.org/examples/01-filter/extract_cells_inside_surface",
        ),
        (
            "trimesh: contains",
            "https://trimesh.org/trimesh.base.html#trimesh.base.Trimesh.contains",
        ),
        ("libigl 702", "https://libigl.github.io/tutorial/#generalized-winding-number"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("holey_bunny", device)
    lower, upper = (np.array(corner) for corner in od.bounds.aabb(vertices))
    axes = [np.linspace(lower[i], upper[i], 48) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points = wp.array(grid, dtype=wp.vec3, device=device)

    mesh = wp.Mesh(points=vertices, indices=faces)
    by_parity = od.ray.contains_points(mesh, points).numpy()
    signed = od.proximity.signed_distance_on_mesh(vertices, faces, points, sign_mode="winding")
    by_winding = signed.numpy() < 0.0
    print(f"inside by ray parity:     {by_parity.sum()} of {grid.shape[0]} grid points")
    print(f"inside by winding number: {by_winding.sum()}")
    print(f"the two disagree on {(by_parity != by_winding).sum()}")
    # --8<-- [end:code]
    return {"grid": grid, "parity": by_parity, "winding": by_winding}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("holey_bunny")
    grid, parity, winding = result["grid"], result["parity"], result["winding"]
    disagree = parity != winding
    # One layer of the grid (constant z), the one where the two tests disagree the most.
    _, layer_of = np.unique(grid[:, 2], return_inverse=True)
    layer = layer_of == np.argmax(np.bincount(layer_of, weights=disagree))
    front = r.Camera(direction=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0), zoom=1.1, parallel=True)

    def slab(inside: np.ndarray, title: str) -> r.Panel:
        return r.Panel(
            [
                r.Points(grid[layer & inside], color=r.GREEN, size=7),
                r.Points(grid[layer & ~inside], color=r.GREY, size=4),
                r.Points(grid[layer & disagree], color=r.RED, size=3),
            ],
            title=title,
            camera=front,
        )

    return r.Figure(
        [
            r.Panel([r.Mesh(vertices, faces, color=r.LIGHT_GREEN)], title="Holey bunny"),
            slab(parity, "One grid layer: ray parity"),
            slab(winding, "Same layer: winding number"),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY, opacity=0.35),
                    r.Points(grid[disagree], color=r.RED, size=7),
                ],
                title="All points where they disagree",
            ),
        ],
        camera=data.camera("holey_bunny"),
        ncols=4,
        panel_size=(560, 520),
    )
