from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="Q1",
    title="Ray casting a depth and a normal image",
    summary="""
    One ray per pixel of a pinhole camera is cast against the scene's BVH with
    [`intersects_location`][ordito.ray.intersects_location]. Each hit gives the depth of its pixel,
    and the normal of the face it struck from
    [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas] gives the pixel's normal:
    a ray tracer's first bounce in a few lines.
    """,
    credits=(
        (
            "Open3D: ray casting",
            "https://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html",
        ),
        ("trimesh: raytrace", "https://github.com/mikedh/trimesh/blob/main/examples/raytrace.py"),
        ("libigl 608", "https://libigl.github.io/tutorial/#off-screen-ray-tracing-with-embree"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("scene", device)
    mesh = wp.Mesh(points=vertices, indices=faces)

    # A pinhole camera at `eye` looking at `target`, one ray per pixel.
    height, width, fov = 300, 480, np.radians(45.0)
    eye, target, up = (
        np.array([0.0, 1.4, 3.2]),
        np.array([0.0, 0.35, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, up) / np.linalg.norm(np.cross(forward, up))
    v, u = np.mgrid[0:height, 0:width]
    scale = np.tan(fov / 2) / height * 2
    directions = (
        forward
        + (u - width / 2)[..., None] * scale * right
        - (v - height / 2)[..., None] * scale * np.cross(right, forward)
    ).reshape(-1, 3)
    origins = wp.array(np.broadcast_to(eye, directions.shape), dtype=wp.vec3, device=device)
    directions = wp.array(directions, dtype=wp.vec3, device=device)

    locations, ray_index, face_index = od.ray.intersects_location(mesh, origins, directions)
    face_normals, _ = od.triangles.face_normals_and_areas(vertices, faces)

    depth = np.full(height * width, np.nan)
    depth[ray_index.numpy()] = np.linalg.norm(locations.numpy() - eye, axis=1)
    normal = np.zeros((height * width, 3))
    normal[ray_index.numpy()] = face_normals.numpy()[face_index.numpy()]
    depth, normal = depth.reshape(height, width), normal.reshape(height, width, 3)
    print(f"{ray_index.shape[0]} of {height * width} rays hit the scene")
    # --8<-- [end:code]
    return {"depth": depth, "normal": normal}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("scene")
    depth, normal = result["depth"], result["normal"]
    hit = np.isfinite(depth)
    rgba = np.zeros((*normal.shape[:2], 4))
    rgba[..., :3] = 0.5 * (normal + 1.0)
    rgba[..., 3] = hit

    def draw_depth(ax: Any) -> None:
        ax.imshow(np.ma.masked_invalid(depth), cmap="magma_r")
        ax.axis("off")

    def draw_normal(ax: Any) -> None:
        ax.imshow(rgba)
        ax.axis("off")

    return r.Figure(
        [
            r.Panel(
                [r.Mesh(vertices, faces)],
                title="Scene",
                camera=r.Camera(direction=(0.0, 1.05, 3.2), zoom=1.3),
            ),
            r.Plot(draw_depth, title="Depth"),
            r.Plot(draw_normal, title="Normals"),
        ],
        panel_size=(640, 460),
    )
