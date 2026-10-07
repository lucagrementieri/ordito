from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="T4",
    title="Inside tests and surface samples",
    summary="""
    [`contains`][ordito.mesh.Trimesh.contains] classifies query points as inside or outside by ray
    parity against the mesh's cached BVH, so it needs a closed surface: the bunny is first closed
    by [`make_solid`][ordito.repair.make_solid]. [`sample`][ordito.mesh.Trimesh.sample] draws
    area-uniform points on the surface and returns the face each point landed on, which indexes
    [`face_normals`][ordito.mesh.Trimesh.face_normals].
    """,
    credits=(
        ("trimesh: quick start", "https://trimesh.org/quick_start.html"),
        (
            "Open3D: mesh sampling",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    mesh = od.Trimesh(*od.repair.make_solid(*data.load("bunny", device)))
    lower, upper = mesh.bounds
    axes = [np.linspace(low, high, 48) for low, high in zip(lower, upper, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = mesh.contains(wp.array(grid, dtype=wp.vec3, device=device)).numpy()
    print(f"{inside.sum()} of {grid.shape[0]} grid points are inside")

    points, face_index = mesh.sample(3000, seed=0)
    normals = mesh.face_normals.numpy()[face_index.numpy()]
    print(f"{points.shape[0]} samples on a surface of area {mesh.area:.4f}")
    # --8<-- [end:code]
    return {
        "vertices": mesh.vertices.numpy(),
        "faces": mesh.faces.numpy(),
        "grid": grid,
        "inside": inside,
        "points": points.numpy(),
        "normals": normals,
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = result["vertices"], result["faces"]
    grid, inside = result["grid"].reshape(48, 48, 48, 3), result["inside"].reshape(48, 48, 48)
    k = 24
    layer, mask = grid[:, :, k].reshape(-1, 3), inside[:, :, k].ravel()
    cube = result["grid"]
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY, opacity=0.25),
                    r.Points(layer[~mask], color=r.GREY, size=6),
                    r.Points(layer[mask], color=r.GREEN, size=9),
                ],
                title="One slice of the grid",
            ),
            r.Panel(
                [r.Points(cube[result["inside"]], color=r.GREEN, size=5)],
                title="Every inside grid point",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN),
                    r.Points(result["points"], color=r.ORANGE, size=5),
                    r.Arrows(
                        result["points"][:800], result["normals"][:800], color=r.BLUE, scale=0.007
                    ),
                ],
                title="Surface samples and their normals",
            ),
        ],
        camera=data.camera("bunny"),
        panel_size=(600, 560),
    )
