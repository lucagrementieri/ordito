from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="RR4",
    title="Rigid ICP: point-to-point vs point-to-plane",
    summary="""
    Two overlapping partial scans of the bunny, the second turned by 15 degrees and shifted.
    Iterative closest point alternates between matching each point of the moving scan to its
    nearest neighbour in the target and solving for the rigid motion that best fits the matches.
    [`icp`][ordito.registration.icp] minimizes point-to-point distances; here it stalls in a
    local minimum a third of the way there, because the scans only partly overlap and the
    unmatched rims pull back.
    [`icp_point_to_plane`][ordito.registration.icp_point_to_plane] minimizes the distance to the
    target's tangent planes (normals from [`estimate_normals`][ordito.points.estimate_normals]),
    which lets the scans slide along each other, and lands on the true pose.
    """,
    credits=(
        (
            "Open3D: ICP registration",
            "https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html",
        ),
        ("MeshLib: ICP", "https://meshlib.io/documentation/ExampleMeshICP.html"),
        ("libigl 808", "https://libigl.github.io/tutorial/#iterative-closest-point"),
        (
            "trimesh: scan registration",
            "https://github.com/mikedh/trimesh/blob/main/examples/scan_register.py",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    target = data.load_points("scan_a", device)
    moving = data.load_points("scan_b", device)
    neighbours, _ = od.neighbors.query_nearest(target, target, k=16)
    target_normals = od.points.estimate_normals(target, neighbours)

    _, point_to_point, _ = od.registration.icp(moving, target, max_iterations=50, max_distance=0.02)
    matrix, point_to_plane, _ = od.registration.icp_point_to_plane(
        moving, target, target_normals=target_normals, max_iterations=50, max_distance=0.02
    )
    for name, aligned in (("point-to-point", point_to_point), ("point-to-plane", point_to_plane)):
        _, residual = od.neighbors.query_nearest(target, aligned, k=1)
        print(f"{name}: median distance to the target {np.median(residual.numpy()):.5f}")
    angle = np.degrees(np.arccos((np.trace(matrix.numpy()[0][:3, :3]) - 1.0) / 2.0))
    print(f"point-to-plane recovered a rotation of {angle:.1f} degrees")
    # --8<-- [end:code]
    residuals = {}
    for key, aligned in (("point", point_to_point), ("plane", point_to_plane)):
        _, residual = od.neighbors.query_nearest(target, aligned, k=1)
        residuals[key] = residual.numpy()
    return {
        "target": target.numpy(),
        "moving": moving.numpy(),
        "point": point_to_point.numpy(),
        "plane": point_to_plane.numpy(),
        "residual_point": residuals["point"],
        "residual_plane": residuals["plane"],
    }


def figure(result: dict[str, Any]) -> r.Figure:
    target = result["target"]
    clim = (0.0, 0.004)

    def pair(moving: np.ndarray, title: str) -> r.Panel:
        return r.Panel(
            [r.Points(target, color=r.GREY, size=3), r.Points(moving, color=r.BLUE, size=3)],
            title=title,
        )

    def residual(key: str, title: str) -> r.Panel:
        return r.Panel(
            [
                r.Points(
                    result[key],
                    scalars=result[f"residual_{key}"],
                    cmap="magma_r",
                    clim=clim,
                    scalar_bar="distance to the target",
                    size=4,
                )
            ],
            title=title,
        )

    return r.Figure(
        [
            pair(result["moving"], "Before: target grey, moving blue"),
            pair(result["point"], "Point-to-point"),
            pair(result["plane"], "Point-to-plane"),
            residual("point", "Residual, point-to-point"),
            residual("plane", "Residual, point-to-plane"),
        ],
        camera=data.camera("scan_a"),
        ncols=5,
        panel_size=(480, 520),
    )
