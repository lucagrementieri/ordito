from __future__ import annotations

from typing import Any

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="V3",
    title="Meshing an implicit surface: the gyroid",
    summary="""
    Any scalar field sampled on a lattice can be meshed by
    [`marching_cubes`][ordito.levelset.marching_cubes]. The gyroid
    `sin x cos y + sin y cos z + sin z cos x = 0` is a triply periodic minimal surface; on the
    left it fills a box, on the right the field is combined with a sphere's signed distance
    (their maximum is the intersection of the two solids) to cut a closed gyroid solid out of a
    ball.
    """,
    credits=(
        ("PyVista: gyroid", "https://docs.pyvista.org/examples/99-advanced/gyroid"),
        ("libigl 715", "https://libigl.github.io/tutorial/#marching-cubes"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od

    n, extent = 160, 2.0 * np.pi
    x, y, z = np.meshgrid(*[np.linspace(-extent, extent, n)] * 3, indexing="ij")
    gyroid = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    in_ball = np.maximum(gyroid, np.sqrt(x**2 + y**2 + z**2) - 0.95 * extent)

    bounds = (wp.vec3(-extent, -extent, -extent), wp.vec3(extent, extent, extent))
    surfaces = {}
    for name, field in {"box": gyroid, "ball": in_ball}.items():
        lattice = od.typing.as_array3d(
            wp.array(field.astype(np.float32), dtype=wp.float32, device=device), wp.float32
        )
        surfaces[name] = od.levelset.marching_cubes(lattice, 0.0, bounds=bounds)
        v, f = surfaces[name]
        print(f"{name}: {f.shape[0] // 3} faces, watertight {od.validation.is_watertight(v, f)}")
    # --8<-- [end:code]
    return {name: (v.numpy(), f.numpy().reshape(-1, 3)) for name, (v, f) in surfaces.items()}


def figure(result: dict[str, Any]) -> r.Figure:
    camera = r.Camera(direction=(1.0, 0.7, 0.6), up=(0.0, 0.0, 1.0), zoom=1.15)
    bv, bf = result["box"]
    sv, sf = result["ball"]
    return r.Figure(
        [
            r.Panel([r.Mesh(bv, bf, scalars=bv[:, 2], cmap="viridis")], title="In a box"),
            r.Panel(
                [r.Mesh(sv, sf, scalars=sv[:, 2], cmap="viridis")], title="Intersected with a ball"
            ),
        ],
        camera=camera,
        link_bounds=False,
        panel_size=(640, 600),
    )
