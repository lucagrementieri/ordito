from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC1",
    title="Down-sampling: voxel grid, farthest point and blue noise",
    summary="""
    Three ways to thin the dragon's 437 000 vertices to a few thousand points.
    [`voxel_down_sample`][ordito.voxels.voxel_down_sample] averages the points in each occupied
    cell of a grid; [`farthest_point_sample`][ordito.points.farthest_point_sample] greedily picks
    the point farthest from all picked so far, for an exact count with even coverage; and
    [`sample_surface_blue_noise`][ordito.sample.sample_surface_blue_noise] draws new points on
    the mesh surface, no two closer than a radius.
    """,
    credits=(
        (
            "Open3D: voxel down-sampling",
            "https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html",
        ),
        (
            "PyVista: farthest point sampling",
            "https://docs.pyvista.org/examples/01-filter/farthest_point_sampling",
        ),
        (
            "pytorch3d: sample_farthest_points",
            "https://pytorch3d.readthedocs.io/en/latest/modules/ops.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("dragon", device)

    voxel = od.voxels.voxel_down_sample(vertices, 0.004)
    farthest = vertices.numpy()[od.points.farthest_point_sample(vertices, voxel.shape[0]).numpy()]
    blue, _ = od.sample.sample_surface_blue_noise(vertices, faces, radius=0.0027, seed=1)
    print(f"{vertices.shape[0]} input points")
    print(f"voxel grid: {voxel.shape[0]}, farthest point: {farthest.shape[0]}")
    print(f"blue noise: {blue.shape[0]}")
    # --8<-- [end:code]
    return {
        "points": vertices.numpy(),
        "voxel": voxel.numpy(),
        "farthest": farthest,
        "blue": blue.numpy(),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    points = result["points"]
    head = np.array([[-0.11, 0.12, -0.05], [-0.04, 0.2, 0.04]])
    panels = [r.Panel([r.Points(points, color=r.GREY, size=2)], title="Dragon vertices")]
    for key, title in (
        ("voxel", "Voxel grid"),
        ("farthest", "Farthest point"),
        ("blue", "Blue noise on the surface"),
    ):
        panels.append(
            r.Panel(
                [r.Points(result[key], color=r.GREEN, size=4)],
                title=f"{title} ({result[key].shape[0]})",
            )
        )
    close = [
        r.Panel(
            [r.Points(result[key], color=r.GREEN, size=7)], title=f"{title}: the head", bounds=head
        )
        for key, title in (
            ("voxel", "Voxel grid"),
            ("farthest", "Farthest point"),
            ("blue", "Blue noise"),
        )
    ]
    close.insert(
        0, r.Panel([r.Points(points, color=r.GREY, size=3)], title="The head", bounds=head)
    )
    return r.Figure(panels + close, camera=data.camera("dragon"), ncols=4, panel_size=(560, 440))
