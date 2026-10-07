from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="X3",
    title="Clipping with a scalar field",
    summary="""
    Any per-vertex field can cut a mesh along one of its level sets. Here the field is the
    geodesic distance from the bunny's nose ([`heat_geodesic`][ordito.heat.heat_geodesic]).
    [`clip_mesh_with_field`][ordito.intersection.clip_mesh_with_field] keeps the region on one
    side of the level set, re-triangulating every crossed face so the edge is clean rather than
    staircased; [`split_faces_along_field`][ordito.intersection.split_faces_along_field] keeps
    both sides, makes the level set a curve of real mesh edges and labels each face by its side.
    """,
    credits=(
        (
            "PyVista: clip with a surface",
            "https://docs.pyvista.org/examples/01-filter/clip_with_surface",
        ),
        ("PyVista: threshold", "https://docs.pyvista.org/examples/01-filter/using_filters"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    nose = wp.array([11842], dtype=wp.int32, device=device)
    distance = od.heat.heat_geodesic(vertices, faces, nose)

    # Keep the geodesic disk of radius 0.06 around the nose: the field 0.06 - d is >= 0 there.
    radius = 0.06
    inside = wp.array(radius - distance.numpy(), dtype=wp.float32, device=device)
    disk_vertices, disk_faces = od.intersection.clip_mesh_with_field(vertices, faces, inside)

    split_vertices, split_faces, positive = od.intersection.split_faces_along_field(
        vertices, faces, inside
    )
    print(f"clipped disk: {disk_faces.shape[0] // 3} faces")
    inside_count = int(positive.numpy().sum())
    print(f"split mesh: {inside_count} faces inside, {split_faces.shape[0] // 3} in all")
    # --8<-- [end:code]
    return {
        "distance": distance.numpy(),
        "disk_vertices": disk_vertices.numpy(),
        "disk_faces": disk_faces.numpy().reshape(-1, 3),
        "split_vertices": split_vertices.numpy(),
        "split_faces": split_faces.numpy().reshape(-1, 3),
        "positive": positive.numpy(),
        "radius": radius,
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    distance = result["distance"]
    disk_v, disk_f = result["disk_vertices"], result["disk_faces"]
    split_v, split_f = result["split_vertices"], result["split_faces"]
    colors = np.tile(np.array([0xC8, 0xC8, 0xC8], dtype=np.uint8), (split_f.shape[0], 1))
    colors[result["positive"]] = (0x76, 0xB9, 0x00)
    nose = vertices[11842]
    # Close-up on one stretch of the cut: a crossing point the split appended.
    crossings = split_v[vertices.shape[0] :]
    spot = crossings[np.argmax(crossings[:, 2])]
    half = 0.008
    close = np.stack([spot - half, spot + half])
    view = r.Camera(direction=tuple(spot - vertices.mean(axis=0)), zoom=1.0)
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=distance,
                        cmap=r.striped("viridis", bands=20),
                        scalar_bar="geodesic distance",
                    ),
                    r.Points(nose[None], color=r.RED, size=16),
                ],
                title="Distance from the nose",
                camera=data.camera("bunny"),
            ),
            r.Panel(
                [r.Mesh(disk_v, disk_f, color=r.GREEN)],
                title="clip_mesh_with_field: the disk d <= 0.06",
                bounds=np.stack([disk_v.min(axis=0), disk_v.max(axis=0)]),
                camera=data.camera("bunny"),
            ),
            r.Panel(
                [r.Mesh(split_v, split_f, face_colors=colors, show_edges=True, smooth=False)],
                title="split_faces_along_field (close-up)",
                bounds=close,
                camera=view,
            ),
        ]
    )
