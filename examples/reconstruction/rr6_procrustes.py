from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="RR6",
    title="Known correspondences: Procrustes alignment",
    summary="""
    When point `i` of one set is known to match point `i` of the other (landmarks, a tracked
    mesh, a deformed copy), no iteration is needed: [`procrustes`][ordito.registration.procrustes]
    solves for the best rotation, translation and, optionally, uniform scale in closed form (the
    Kabsch-Umeyama solution). Here a copy of the bunny is scaled by 1.6, turned 70 degrees, moved
    and jittered, and the transform is recovered from the vertex correspondence alone.
    """,
    credits=(
        (
            "pytorch3d: corresponding_points_alignment",
            "https://pytorch3d.readthedocs.io/en/latest/modules/ops.html",
        ),
        ("trimesh: procrustes", "https://trimesh.org/trimesh.registration.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, _ = data.load("bunny", device)
    rng = np.random.default_rng(0)
    turn = data.rotation((1.0, 0.4, 0.2), 70.0)
    copy = 1.6 * vertices.numpy() @ turn.T + [0.25, 0.05, -0.1]
    copy += rng.normal(0.0, 0.0005, copy.shape)
    target = wp.array(copy, dtype=wp.vec3, device=device)

    matrix, aligned, cost = od.registration.procrustes(vertices, target, reflection=False)
    linear = matrix.numpy()[0][:3, :3]
    print(f"recovered scale {np.cbrt(np.linalg.det(linear)):.4f} (true 1.6)")
    # `cost` is the mean squared distance, so its root is the RMS residual: the jitter alone.
    print(f"rms residual {np.sqrt(cost):.5f}, jitter {0.0005 * np.sqrt(3):.5f}")
    # --8<-- [end:code]
    return {"target": copy, "aligned": aligned.numpy()}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    target, aligned = result["target"], result["aligned"]
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(vertices, faces, color=r.GREEN), r.Mesh(target, faces, color=r.GREY)],
                title="Source (green) and target (grey)",
                bounds=np.stack(
                    [
                        np.minimum(vertices.min(0), target.min(0)),
                        np.maximum(vertices.max(0), target.max(0)),
                    ]
                ),
            ),
            r.Panel(
                [
                    r.Mesh(aligned, faces, color=r.GREEN),
                    r.Points(target[::12], color=r.GREY, size=4),
                ],
                title="Source moved onto the target (grey dots)",
                bounds=np.stack([target.min(0), target.max(0)]),
            ),
        ],
        camera=r.Camera(direction=(0.25, 0.25, 1.0), zoom=1.1),
        link_bounds=False,
    )
