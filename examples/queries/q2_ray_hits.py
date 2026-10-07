from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="Q2",
    title="Rays against a mesh: hits and misses",
    summary="""
    A fan of rays from one point is shot at an icosphere.
    [`intersects_any`][ordito.ray.intersects_any] answers hit or miss for each ray, and
    [`intersects_location`][ordito.ray.intersects_location] returns the first hit of every ray
    that strikes: its position, the ray it belongs to and the face it struck.
    """,
    credits=(
        ("trimesh: ray", "https://github.com/mikedh/trimesh/blob/main/examples/ray.ipynb"),
        ("PyVista: ray_trace", "https://docs.pyvista.org/examples/01-filter/ray_trace"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2, device=device)
    mesh = wp.Mesh(points=vertices, indices=faces)

    # 60 rays from one eye point, aimed at random targets around the sphere.
    rng = np.random.default_rng(0)
    eye = np.array([3.0, 1.8, 2.4])
    targets = rng.uniform(-1.2, 1.2, (60, 3))
    origins = wp.array(np.broadcast_to(eye, targets.shape), dtype=wp.vec3, device=device)
    directions = wp.array(targets - eye, dtype=wp.vec3, device=device)

    hit = od.ray.intersects_any(mesh, origins, directions)
    locations, ray_index, face_index = od.ray.intersects_location(mesh, origins, directions)
    print(f"{int(hit.numpy().sum())} of {targets.shape[0]} rays hit the sphere")
    print(f"{np.unique(face_index.numpy()).size} distinct faces struck")
    # --8<-- [end:code]
    return {
        "vertices": vertices.numpy(),
        "faces": faces.numpy().reshape(-1, 3),
        "eye": eye,
        "targets": targets,
        "hit": hit.numpy(),
        "locations": locations.numpy(),
        "ray_index": ray_index.numpy(),
        "face_index": face_index.numpy(),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces, eye = result["vertices"], result["faces"], result["eye"]
    directions = result["targets"] - eye
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    hit = result["hit"]
    end = np.empty_like(directions)
    end[result["ray_index"]] = result["locations"]
    misses = ~hit
    end[misses] = eye + directions[misses] * (
        np.linalg.norm(result["targets"][misses] - eye, axis=1)[:, None] + 1.0
    )
    colors = np.tile(np.array([0xC8, 0xC8, 0xC8], dtype=np.uint8), (faces.shape[0], 1))
    colors[result["face_index"]] = (0xE8, 0x41, 0x2C)
    rays_hit = np.stack([np.broadcast_to(eye, end[hit].shape), end[hit]], axis=1)
    rays_miss = np.stack([np.broadcast_to(eye, end[misses].shape), end[misses]], axis=1)
    camera = r.Camera(direction=(1.0, -0.5, -1.0), up=(0.0, 0.0, 1.0), zoom=1.1)
    sphere = r.Mesh(vertices, faces, face_colors=colors, show_edges=True, smooth=False)
    return r.Figure(
        [
            r.Panel(
                [
                    sphere,
                    r.Segments(rays_hit, color=r.ORANGE, width=1.5),
                    r.Segments(rays_miss, color=r.BLUE, width=1.0),
                    r.Points(result["locations"], color=r.RED, size=10),
                    r.Points(eye[None], color=r.ORANGE, size=16),
                ],
                title="Rays: hits orange, misses blue",
                bounds=np.array([[-2.0, -2.0, -2.0], [3.0, 1.8, 2.4]]),
            ),
            r.Panel(
                [sphere, r.Points(result["locations"], color=r.ORANGE, size=12)],
                title="Struck faces and hit points",
                bounds=np.array([[-1.1, -1.1, -1.1], [1.1, 1.1, 1.1]]),
                camera=r.Camera(direction=tuple(eye), up=(0.0, 0.0, 1.0), zoom=1.1),
            ),
        ],
        camera=camera,
    )
