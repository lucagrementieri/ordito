from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="X4",
    title="Cropping to a box",
    summary="""
    [`crop_mesh`][ordito.bounds.crop_mesh] keeps the faces whose three corners lie inside a box
    and renumbers them from zero; [`crop_points`][ordito.bounds.crop_points] keeps the points
    inside one and returns their original indices too, so per-point attributes can follow. Both
    take an optional rotation, which turns the axis-aligned box into an oriented one.
    """,
    credits=(
        (
            "Open3D: crop point cloud",
            "https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html",
        ),
        (
            "PyVista: clip with a box",
            "https://docs.pyvista.org/examples/01-filter/clip_with_plane_box",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    head_vertices, head_faces = od.bounds.crop_mesh(
        vertices, faces, wp.vec3(-0.1, 0.09, -0.06), wp.vec3(-0.03, 0.2, 0.06)
    )

    # An oriented box: rows of `rotation` are the box axes, the bounds are in box coordinates.
    points = data.load_points("bunny_cloud", device)
    angle = np.radians(35.0)
    rotation = wp.mat33(
        np.cos(angle), np.sin(angle), 0.0, -np.sin(angle), np.cos(angle), 0.0, 0.0, 0.0, 1.0
    )
    box_min, box_max = wp.vec3(-0.02, 0.02, -0.08), wp.vec3(0.11, 0.08, 0.08)
    kept, _ = od.bounds.crop_points(points, box_min, box_max, rotation=rotation)
    print(f"cropped mesh: {head_faces.shape[0] // 3} of {faces.shape[0] // 3} faces")
    print(f"cropped cloud: {kept.shape[0]} of {points.shape[0]} points")
    # --8<-- [end:code]
    return {
        "head_vertices": head_vertices.numpy(),
        "head_faces": head_faces.numpy().reshape(-1, 3),
        "points": points.numpy(),
        "kept": kept.numpy(),
        "rotation": np.array(rotation).reshape(3, 3),
        "box": (np.array(box_min), np.array(box_max)),
        "aabb": (np.array([-0.1, 0.09, -0.06]), np.array([-0.03, 0.2, 0.06])),
    }


def _corners(lower: np.ndarray, upper: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    local = np.array(list(itertools.product(*zip(lower, upper, strict=True))))
    return local @ rotation  # box-to-world is the transpose of the world-to-box rows


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    aabb_lower, aabb_upper = result["aabb"]
    box_lower, box_upper = result["box"]
    aabb = _corners(aabb_lower, aabb_upper, np.eye(3))
    obb = _corners(box_lower, box_upper, result["rotation"])
    camera = data.camera("bunny")
    everything = np.concatenate([vertices, aabb, obb])
    bounds = np.stack([everything.min(axis=0), everything.max(axis=0)])
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(vertices, faces, color=r.LIGHT_GREEN), r.WireBox(aabb, color=r.BLUE)],
                title="Mesh and an axis-aligned box",
                bounds=bounds,
            ),
            r.Panel(
                [r.Mesh(result["head_vertices"], result["head_faces"]), r.WireBox(aabb)],
                title="crop_mesh",
                bounds=bounds,
            ),
            r.Panel(
                [r.Points(result["points"], color=r.GREY, size=3), r.WireBox(obb, color=r.BLUE)],
                title="Cloud and an oriented box",
                bounds=bounds,
            ),
            r.Panel(
                [r.Points(result["kept"], color=r.GREEN, size=4), r.WireBox(obb)],
                title="crop_points",
                bounds=bounds,
            ),
        ],
        camera=camera,
        ncols=4,
        link_bounds=False,
        panel_size=(560, 520),
    )
