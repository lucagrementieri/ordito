from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="V4",
    title="Voxelizing meshes and point clouds",
    summary="""
    [`voxelize_mesh`][ordito.voxels.voxelize_mesh] marks every cell a triangle actually touches
    (an exact triangle-box test), which seals a closed surface (the scan's open base is closed
    first with [`fill_min_weight`][ordito.holes.fill_min_weight]);
    [`fill_cavities`][ordito.voxels.fill_cavities] then fills every empty cell that cannot reach
    the outside, turning the shell into a solid (`mode="solid"` does both; the filled interior
    is orange). [`voxelize_points`][ordito.voxels.voxelize_points] marks the cells that contain
    a sample of a point cloud. [`to_boxes`][ordito.voxels.to_boxes] meshes a voxel
    set as cubes for display; the solid is shown cut in half through
    [`cell_centers`][ordito.voxels.cell_centers].
    """,
    credits=(
        (
            "Open3D: voxelization",
            "https://www.open3d.org/docs/release/tutorial/geometry/voxelization.html",
        ),
        ("trimesh: voxel", "https://trimesh.org/trimesh.voxel.html"),
        ("PyVista: voxelize", "https://docs.pyvista.org/examples/01-filter/voxelize"),
        ("PyTorch3D: cubify", "https://pytorch3d.org/tutorials"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    faces = od.holes.fill_min_weight(vertices, faces)  # seal the scan's open base first
    shell = od.voxels.voxelize_mesh(vertices, faces, voxel_size=0.004)
    solid = od.voxels.fill_cavities(shell)
    cloud = od.voxels.voxelize_points(data.load_points("bunny_cloud", device), voxel_size=0.004)
    for name, grid in {"surface": shell, "solid": solid, "point cloud": cloud}.items():
        print(f"{name}: {grid.get_active_stats().voxel_count} voxels")

    shell_boxes = od.voxels.to_boxes(shell)
    cloud_boxes = od.voxels.to_boxes(cloud)
    solid_centers = od.voxels.cell_centers(solid)
    # --8<-- [end:code]
    return {
        "shell": tuple(a.numpy() for a in shell_boxes),
        "cloud": tuple(a.numpy() for a in cloud_boxes),
        "solid": solid_centers.numpy(),
        "shell_centers": od.voxels.cell_centers(shell).numpy(),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    sv, sf = result["shell"]
    cv, cf = result["cloud"]
    centers = result["solid"]
    back = centers[:, 2] < np.median(centers[:, 2])
    shell = {tuple(c) for c in np.round(result["shell_centers"] / 0.004).astype(int).tolist()}
    on_shell = np.array([tuple(c) in shell for c in np.round(centers / 0.004).astype(int).tolist()])
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(sv, sf, show_edges=True, smooth=False, line_width=0.3)], title="Surface"
            ),
            r.Panel(
                [
                    r.Boxes(centers[back & on_shell], 0.004, color=r.GREEN),
                    r.Boxes(centers[back & ~on_shell], 0.004, color=r.ORANGE),
                ],
                title="Solid, cut in half",
            ),
            r.Panel(
                [r.Mesh(cv, cf, color=r.BLUE, show_edges=True, smooth=False, line_width=0.3)],
                title="Point cloud",
            ),
        ],
        camera=data.camera("bunny"),
        panel_size=(640, 600),
    )
