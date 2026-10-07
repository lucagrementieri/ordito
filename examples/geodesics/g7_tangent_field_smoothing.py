from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="G7",
    title="Smoothing a tangent vector field",
    summary="""
    A random unit vector at every vertex is a tangent field with no structure.
    [`diffuse_tangent_field`][ordito.heat.diffuse_tangent_field] runs one implicit step of the
    connection Laplacian on it, which averages neighbouring vectors *after* transporting them into
    a common frame, so the field straightens out across the surface instead of cancelling against
    the change of tangent plane. The diffusion time of
    [`vector_heat_operators`][ordito.heat.vector_heat_operators] sets how far the averaging
    reaches. The smoothed fields are normalized to unit length and turned into 3-D vectors by
    [`tangent_to_world`][ordito.heat.tangent_to_world]; the printed alignment compares the two
    ends of every [`edges_unique`][ordito.edges.edges_unique] edge.
    """,
    credits=(("libigl 901", "https://libigl.github.io/tutorial/"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    angle = np.random.default_rng(0).uniform(0.0, 2.0 * np.pi, vertices.shape[0])
    noisy = wp.array(np.stack([np.cos(angle), np.sin(angle)], 1), dtype=wp.vec2d, device=device)
    edges = od.edges.edges_unique(faces)[0].numpy()

    world = {}
    for name, t in (("random", None), ("t = 1e-3", 1e-3), ("t = 1e-2", 1e-2)):
        system, _, (basis_x, basis_y, _), preconditioner = od.heat.vector_heat_operators(
            vertices, faces, t
        )
        field = noisy.numpy()
        if t is not None:
            field = od.heat.diffuse_tangent_field(system, noisy, preconditioner=preconditioner)
            field = field.numpy() / np.linalg.norm(field.numpy(), axis=1, keepdims=True)
        tangent = wp.array(field, dtype=wp.vec2, device=device)
        world[name] = od.heat.tangent_to_world(tangent, basis_x, basis_y).numpy()
        cosine = np.einsum("ij,ij->i", world[name][edges[:, 0]], world[name][edges[:, 1]])
        print(f"{name:>9}: mean cosine between neighbours {cosine.mean():.3f}")
    # --8<-- [end:code]
    return {"world": world}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    normals = data.vertex_normals(vertices, faces)
    rng = np.random.default_rng(1)
    pick = rng.choice(vertices.shape[0], 1500, replace=False)
    lift = vertices + 0.001 * normals
    panels = []
    for name, field in result["world"].items():
        panels.append(
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN),
                    r.Segments(
                        np.stack(
                            [lift[pick] - 0.004 * field[pick], lift[pick] + 0.004 * field[pick]], 1
                        ),
                        color=r.BLUE,
                        width=2.0,
                    ),
                ],
                title="Random field" if name == "random" else f"Smoothed, {name}",
            )
        )
    return r.Figure(panels, camera=data.camera("bunny"))
