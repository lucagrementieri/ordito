from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC3",
    title="Normal estimation",
    summary="""
    A bare point cloud has no normals; [`estimate_normals`][ordito.points.estimate_normals] fits
    a plane to each point's neighbourhood (here its 16 nearest neighbours from
    [`query_nearest`][ordito.neighbors.query_nearest]) and takes the plane's normal. The sign is
    a convention, not a measurement: by default each normal is turned away from the cloud's
    centroid, which is right on most of the bunny and wrong where the surface folds back towards
    the centre (red, compared with the exact normals of the surface the cloud was sampled from).
    """,
    credits=(
        (
            "Open3D: vertex normal estimation",
            "https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html",
        ),
        (
            "pytorch3d: estimate_pointcloud_normals",
            "https://pytorch3d.readthedocs.io/en/latest/modules/ops.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    points = data.load_points("bunny_cloud", device)
    neighbours, _ = od.neighbors.query_nearest(points, points, k=16)
    normals = od.points.estimate_normals(points, neighbours)

    # Compare with the exact normals of the surface the cloud was sampled from.
    cosine = np.sum(normals.numpy() * data.cloud_normals("bunny_cloud"), axis=1)
    angle = np.degrees(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0)))
    print(f"median angle to the true normal (ignoring sign): {np.median(angle):.2f} degrees")
    print(f"flipped relative to the true normal: {(cosine < 0).mean():.1%} of points")
    # --8<-- [end:code]
    return {"points": points.numpy(), "normals": normals.numpy(), "cosine": cosine, "angle": angle}


def figure(result: dict[str, Any]) -> r.Figure:
    points, normals, cosine = result["points"], result["normals"], result["cosine"]
    shading = np.clip(0.5 * (normals + 1.0), 0.0, 1.0)
    center = np.array([0.0, 0.1, 0.045])
    head = np.stack([center - 0.022, center + 0.022])
    framed = np.all((points >= head[0]) & (points <= head[1]), axis=1)
    every = np.flatnonzero(framed)[::4]
    vertices, faces = data.arrays("bunny")
    flipped = cosine < 0
    return r.Figure(
        [
            r.Panel(
                [r.Points(points, scalars=shading, rgb=True, size=4)],
                title="Points coloured by their normal",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY),
                    r.Arrows(points[every], normals[every], color=r.BLUE, scale=0.0045),
                ],
                title="Normals (close-up)",
                bounds=head,
            ),
            r.Panel(
                [
                    r.Points(points[~flipped], color=r.GREEN, size=4),
                    r.Points(points[flipped], color=r.RED, size=5),
                ],
                title="Oriented against the true normal (red)",
            ),
        ],
        camera=data.camera("bunny_cloud"),
    )
