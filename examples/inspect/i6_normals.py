from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="I6",
    title="Normals: per face, per vertex, per corner",
    summary="""
    Four ways to put normals on a machined part, coloured by direction (red, green, blue for
    the x, y, z components). [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]
    gives one flat normal per face. [`vertex_normals`][ordito.vertices.vertex_normals] averages
    them per vertex, weighted by face area or by the corner angle: area weighting leans towards
    the long triangles of a flat side, angle weighting does not depend on how a flat side was
    triangulated. Both smear a crease. [`corner_normals`][ordito.triangles.corner_normals] keeps
    one normal per face corner and averages only over faces not separated by a
    [`crease_edges`][ordito.seams.crease_edges] edge, which is what a renderer with hard edges
    needs.
    """,
    credits=(
        ("libigl 201", "https://libigl.github.io/tutorial/#normals"),
        (
            "Open3D: vertex normals",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
        ("PyVista: compute normals", "https://docs.pyvista.org/examples/01-filter/compute_normals"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("cad_part", device)
    face_normals, _ = od.triangles.face_normals_and_areas(vertices, faces)
    area_weighted = od.vertices.vertex_normals(vertices, faces, weighting="area")
    angle_weighted = od.vertices.vertex_normals(vertices, faces, weighting="angle")
    creases = od.seams.crease_edges(vertices, faces, angle=30.0)
    corner = od.triangles.corner_normals(vertices, faces, creases)  # (n_faces, 3) normals

    gap = np.degrees(
        np.arccos(np.clip((area_weighted.numpy() * angle_weighted.numpy()).sum(1), -1, 1))
    )
    print(f"area- and angle-weighted vertex normals differ by up to {gap.max():.1f} degrees")
    # --8<-- [end:code]
    return {
        "face": face_normals.numpy(),
        "area": area_weighted.numpy(),
        "angle": angle_weighted.numpy(),
        "corner": corner.numpy().reshape(-1, 3),
    }


def _rgb(normals: np.ndarray) -> np.ndarray:
    return np.clip(0.5 * (normals + 1.0), 0.0, 1.0)


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("cad_part")
    scale = 0.12
    # Unweld every face so a per-corner value becomes a per-vertex one.
    soup_vertices = vertices[faces].reshape(-1, 3)
    soup_faces = np.arange(soup_vertices.shape[0]).reshape(-1, 3)
    centroids = vertices[faces].mean(axis=1)
    panels = [
        r.Panel(
            [
                r.Mesh(
                    vertices,
                    faces,
                    face_colors=(_rgb(result["face"]) * 255).astype(np.uint8),
                    smooth=False,
                ),
                r.Arrows(centroids, result["face"] * scale, color="#505050"),
            ],
            title="Face normals",
        )
    ]
    for key, title in (
        ("area", "Vertex normals, area-weighted"),
        ("angle", "Vertex normals, angle-weighted"),
    ):
        panels.append(
            r.Panel(
                [
                    r.Mesh(vertices, faces, scalars=_rgb(result[key]), rgb=True, smooth=False),
                    r.Arrows(vertices, result[key] * scale, color="#505050"),
                ],
                title=title,
            )
        )
    panels.append(
        r.Panel(
            [
                r.Mesh(
                    soup_vertices,
                    soup_faces,
                    scalars=_rgb(result["corner"]),
                    rgb=True,
                    smooth=False,
                )
            ],
            title="Corner normals (creases > 30°)",
        )
    )
    return r.Figure(panels, camera=data.camera("cad_part"), panel_size=(560, 500))
