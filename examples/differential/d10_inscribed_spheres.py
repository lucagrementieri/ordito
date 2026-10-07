from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="D10",
    title="Maximal inscribed spheres and the medial axis",
    summary="""
    After [`filter_laplacian`][ordito.smoothing.filter_laplacian] removes the scan's small bumps,
    [`max_tangent_sphere`][ordito.visibility.max_tangent_sphere] shrinks a ball touching the surface
    at each face centroid, from the inside, until it touches the surface nowhere else. Its radius is
    a local thickness, and its centre lies on the medial axis: the centres of the larger balls trace
    the bunny's skeleton, a sheet down its body and curves along its ears. Balls touching a small
    crease stay tiny, so the skeleton view keeps balls above a tenth of the largest radius.
    """,
    credits=(
        ("trimesh: examples", "https://trimesh.org/examples.html"),
        ("MeshLib: thickness", "https://meshlib.io/documentation/index.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    # The scan's millimetre bumps cap every ball touching them; smooth them away first.
    vertices = od.smoothing.filter_laplacian(vertices, faces, iterations=20)
    mesh = wp.Mesh(points=vertices, indices=faces)
    # Touch every face at its centroid, where the surface is flat: at a vertex the ball must
    # also clear the faces around it, which pins it small wherever the scan is bumpy.
    face_normals, _ = od.triangles.face_normals_and_areas(vertices, faces)
    corners = vertices.numpy()[faces.numpy().reshape(-1, 3)]
    centroids = wp.array(corners.mean(axis=1), dtype=wp.vec3, device=device)
    centers, radii = od.visibility.max_tangent_sphere(mesh, centroids, normals=face_normals)

    radii = radii.numpy()
    medial = radii > 0.1 * radii.max()
    print(f"largest inscribed ball: radius {radii.max():.4f}, median {np.median(radii):.4f}")
    print(f"{medial.sum()} of {radii.size} balls above a tenth of the largest")
    # --8<-- [end:code]
    return {
        "centers": centers.numpy(),
        "radii": radii,
        "medial": medial,
        "touch": centroids.numpy(),
        "vertices": vertices.numpy(),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    _, faces = data.arrays("bunny")
    vertices = result["vertices"]
    radii, centers, medial = result["radii"], result["centers"], result["medial"]
    clim = (0.0, float(np.percentile(radii, 98)))
    # A dozen balls spread over the shape: greedy farthest-first over the medial ones.
    candidates = np.flatnonzero(medial)
    chosen = [int(candidates[np.argmax(radii[candidates])])]
    for _ in range(11):
        gap = np.min(
            np.linalg.norm(centers[candidates][:, None] - centers[chosen][None], axis=2), axis=1
        )
        chosen.append(int(candidates[np.argmax(gap)]))
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=radii,
                        cmap="viridis",
                        clim=clim,
                        scalar_bar="radius",
                    )
                ],
                title="Inscribed ball radius",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY, opacity=0.15),
                    r.Points(
                        centers[medial], scalars=radii[medial], cmap="viridis", clim=clim, size=3.0
                    ),
                ],
                title="Ball centres (medial axis)",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY, opacity=0.25),
                    r.Spheres(centers[chosen], radii[chosen], color=r.ORANGE, opacity=0.7),
                    r.Points(result["touch"][chosen], color=r.RED, size=12),
                ],
                title="A dozen balls",
            ),
        ],
        camera=data.camera("bunny"),
    )
