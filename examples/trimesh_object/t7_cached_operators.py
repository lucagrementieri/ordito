from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="T7",
    title="Reusing cached operators",
    summary="""
    The discrete operators of a mesh depend on the mesh alone, so a
    [`Trimesh`][ordito.mesh.Trimesh] assembles each of them once and hands the same object to
    every solve. Here [`heat_operators`][ordito.mesh.Trimesh.heat_operators] serves four
    [`heat_geodesic`][ordito.heat.heat_geodesic] calls from four different sources, and the
    mesh's [`heat_solver`][ordito.mesh.Trimesh.heat_solver] keeps the factorizations built on the
    second, so the last two run no iteration.
    [`vector_heat_operators`][ordito.mesh.Trimesh.vector_heat_operators] holds that same scalar
    bundle and the [`vertex_tangent_frames`][ordito.mesh.Trimesh.vertex_tangent_frames], and feeds
    [`log_map`][ordito.heat.log_map]: every vertex's position in the tangent plane of a source
    vertex, drawn with hue for its direction and stripes for its distance. The cotangent matrix
    [`cotmatrix`][ordito.mesh.Trimesh.cotmatrix] and lumped mass
    [`mass_matrix_entries`][ordito.mesh.Trimesh.mass_matrix_entries] are cached the same way, for
    callers that assemble their own systems.
    """,
    credits=(
        ("potpourri3d", "https://github.com/nmwsharp/potpourri3d#mesh-distance"),
        ("libigl 716", "https://libigl.github.io/tutorial/#heat-method"),
    ),
)

SOURCES = {"ear": 22820, "nose": 11842, "tail": 12217, "front foot": 34264}


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    mesh = od.Trimesh(*data.load("bunny", device))
    operators = mesh.heat_operators  # assembled here, once

    distances = {}
    for name, vertex in {"ear": 22820, "nose": 11842, "tail": 12217, "front foot": 34264}.items():
        source = wp.array([vertex], dtype=wp.int32, device=device)
        distances[name] = od.heat.heat_geodesic(mesh, source).numpy()
        print(f"from the {name:<10}: farthest point at {distances[name].max():.4f}")

    vector_operators = mesh.vector_heat_operators
    print("vector bundle reuses the scalar one:", vector_operators[1] is operators)
    log = od.heat.log_map(mesh, 16308)
    # --8<-- [end:code]
    return {"distances": distances, "log": log.numpy()}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    cmap = r.striped("viridis", bands=20)
    panels = []
    for name, d in result["distances"].items():
        panels.append(
            r.Panel(
                [
                    r.Mesh(vertices, faces, scalars=d, cmap=cmap, clim=(0.0, float(d.max()))),
                    r.Points(vertices[[SOURCES[name]]], color=r.RED, size=24),
                ],
                title=f"Distance from the {name}",
            )
        )
    log = result["log"]
    angle = np.arctan2(log[:, 1], log[:, 0])
    radius = np.linalg.norm(log, axis=1)
    rgb = plt.get_cmap("hsv")((angle + np.pi) / (2 * np.pi))[:, :3]
    rgb *= np.where((radius / 0.012).astype(int) % 2 == 1, 0.7, 1.0)[:, None]
    panels.append(
        r.Panel(
            [
                r.Mesh(vertices, faces, scalars=(rgb * 255).astype(np.uint8), rgb=True),
                r.Points(vertices[[16308]], color=r.RED, size=24),
            ],
            title="Log map from the back",
        )
    )
    return r.Figure(panels, camera=data.camera("bunny"), panel_size=(520, 520))
