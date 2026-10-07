from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

SOURCE = 5088

META = Meta(
    id="G6",
    title="Parallel transport and the logarithmic map",
    summary="""
    The vector heat method carries a tangent vector from a source vertex to every other vertex
    along the shortest path. [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors]
    returns the result in each vertex's own tangent frame, and
    [`tangent_to_world`][ordito.heat.tangent_to_world] turns it into 3-D arrows.
    [`log_map`][ordito.heat.log_map] combines the same transport with the geodesic distance into
    2-D coordinates around the source, a local "unwrapping" of the surface: a checkerboard drawn in
    those coordinates stays square near the source and bends where the surface curves. Both share
    one [`vector_heat_operators`][ordito.heat.vector_heat_operators] bundle.
    """,
    credits=(
        ("potpourri3d: vector heat", "https://github.com/nmwsharp/potpourri3d"),
        ("libigl 902", "https://libigl.github.io/tutorial/"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    operators = od.heat.vector_heat_operators(vertices, faces)
    basis_x, basis_y, _ = operators[2]

    source = 5088  # a vertex on the flank
    transported, resolved = od.heat.transport_tangent_vectors(
        vertices,
        faces,
        wp.array([source], dtype=wp.int32, device=device),
        wp.array([wp.vec2(1.0, 0.0)], dtype=wp.vec2, device=device),
        operators=operators,
    )
    arrows = od.heat.tangent_to_world(transported, basis_x, basis_y)
    coordinates = od.heat.log_map(vertices, faces, source, operators=operators)
    print(f"resolved vertices: {resolved.numpy().mean():.1%}")
    radius = np.linalg.norm(coordinates.numpy(), axis=1)
    print(f"largest log-map radius (geodesic distance): {radius.max():.4f}")
    # --8<-- [end:code]
    return {"arrows": arrows.numpy(), "resolved": resolved.numpy(), "log": coordinates.numpy()}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    rng = np.random.default_rng(0)
    pick = rng.choice(vertices.shape[0], 450, replace=False)
    pick = pick[result["resolved"][pick]]
    normals = data.vertex_normals(vertices, faces)
    lift = vertices + 0.0015 * normals
    log = result["log"]
    face_log = log[faces].mean(axis=1)
    cell = 0.012
    checker = (np.floor(face_log[:, 0] / cell) + np.floor(face_log[:, 1] / cell)) % 2
    light = np.array([0xF3, 0x9C, 0x12]) / 255.0
    dark = np.array([0x30, 0x30, 0x30]) / 255.0
    colors = np.where(checker[:, None] > 0, light, dark)
    fade = np.clip((np.linalg.norm(face_log, axis=1) - 0.07) / 0.03, 0.0, 1.0)[:, None]
    colors = (1 - fade) * colors + fade * (np.array([0xC8, 0xC8, 0xC8]) / 255.0)
    face_colors = (colors * 255).astype(np.uint8)
    angle = np.arctan2(log[:, 1], log[:, 0])
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN),
                    r.Arrows(lift[pick], result["arrows"][pick], color=r.BLUE, scale=0.011),
                    r.Points(vertices[[SOURCE]], color=r.RED, size=18),
                ],
                title="Transported vector",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, scalars=angle, cmap="twilight", clim=(-np.pi, np.pi)),
                    r.Points(vertices[[SOURCE]], color=r.RED, size=18),
                ],
                title="Log map: angle",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, face_colors=face_colors),
                    r.Points(vertices[[SOURCE]], color=r.RED, size=18),
                ],
                title="Log map: checkerboard",
            ),
        ],
        camera=data.camera("bunny"),
    )
