from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="RR5",
    title="Robust ICP with outliers",
    summary="""
    The moving scan of RR4 again, now cluttered with 15 % stray points around it, and no
    distance cut-off on the matches. Plain least squares lets every stray pull on the fit, and
    [`icp_point_to_plane`][ordito.registration.icp_point_to_plane] converges to a wrong pose. A
    robust kernel down-weights large residuals: `robust_kernel="huber"` caps their influence,
    `"tukey"` ignores them beyond a scale estimated from the residuals' median absolute
    deviation, and both recover the pose.
    """,
    credits=(
        (
            "Open3D: robust kernels",
            "https://www.open3d.org/docs/release/tutorial/pipelines/robust_kernels.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od

    target = data.load_points("scan_a", device)
    moving = data.load_points("scan_b_outliers", device)
    neighbours, _ = od.neighbors.query_nearest(target, target, k=16)
    target_normals = od.points.estimate_normals(target, neighbours)

    aligned = {}
    for kernel in ("none", "huber", "tukey"):
        matrix, aligned[kernel], _ = od.registration.icp_point_to_plane(
            moving, target, target_normals=target_normals, robust_kernel=kernel, max_iterations=50
        )
        rotation = matrix.numpy()[0][:3, :3]
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
        print(f"{kernel:>5}: recovered a rotation of {angle:.1f} degrees (true: 15.0)")
    # --8<-- [end:code]
    n_scan = data.arrays("scan_b")[0].shape[0]
    return {
        "target": target.numpy(),
        "moving": moving.numpy(),
        "n_scan": n_scan,
        **{kernel: points.numpy() for kernel, points in aligned.items()},
    }


def figure(result: dict[str, Any]) -> r.Figure:
    target, n = result["target"], result["n_scan"]

    def pair(moving: np.ndarray, title: str) -> r.Panel:
        return r.Panel(
            [
                r.Points(target, color=r.GREY, size=3),
                r.Points(moving[:n], color=r.BLUE, size=3),
                r.Points(moving[n:], color=r.RED, size=3),
            ],
            title=title,
        )

    return r.Figure(
        [
            pair(result["moving"], "Before (strays red)"),
            pair(result["none"], "Least squares"),
            pair(result["huber"], "Huber"),
            pair(result["tukey"], "Tukey"),
        ],
        camera=r.Camera(direction=(0.25, 0.25, 1.0), zoom=0.8),
        ncols=4,
        panel_size=(520, 520),
    )
