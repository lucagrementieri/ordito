from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="D3",
    title="Gradient of a scalar field",
    summary="""
    A smooth pattern of hills and valleys defined on the bunny's vertices, and its gradient from
    [`face_gradients`][ordito.laplacian.face_gradients]: on each triangle the linear interpolant
    of the vertex values has one constant gradient, which lies in the triangle's plane. The arrows
    point uphill; the magnitude vanishes on hilltops, valley floors and saddles, and the scan's
    small bumps show through it because the gradient is taken along the surface.
    """,
    credits=(
        ("libigl 204", "https://libigl.github.io/tutorial/#gradient"),
        ("PyVista: gradients", "https://docs.pyvista.org/examples/01-filter/gradients"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    x, y, _ = vertices.numpy().T.astype(np.float64)
    field = np.sin(60.0 * x) + np.sin(60.0 * y)
    values = wp.array(field, dtype=wp.float64, device=device)

    gradients = od.laplacian.face_gradients(vertices, faces, values)  # (n_faces,) wp.vec3d
    magnitude = np.linalg.norm(gradients.numpy(), axis=1)
    print(f"{gradients.shape[0]} face gradients")
    print(f"|grad f| from {magnitude.min():.2f} to {magnitude.max():.1f}")
    # --8<-- [end:code]
    return {"field": field, "gradients": gradients.numpy(), "magnitude": magnitude}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    centroids = vertices[faces].mean(axis=1)
    # A random subset of the faces facing the camera on the bunny's side carries the arrows.
    center, half = np.array([0.0, 0.1, 0.03]), 0.03
    near = np.flatnonzero(np.all(np.abs(centroids - center) < 1.5 * half, axis=1))
    picked = np.random.default_rng(0).choice(near, size=min(450, near.size), replace=False)
    gradients = result["gradients"][picked]
    scale = 0.006 / np.percentile(result["magnitude"], 95)
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(vertices, faces, scalars=result["field"], cmap="coolwarm", scalar_bar="f")],
                title="Scalar field",
            ),
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=result["magnitude"],
                        cmap="viridis",
                        clim=r.percentile_clim(result["magnitude"]),
                        scalar_bar="|grad f|",
                    )
                ],
                title="Gradient magnitude",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, scalars=result["field"], cmap="coolwarm"),
                    r.Arrows(centroids[picked], gradients * scale, color="#303030"),
                ],
                title="Gradient (close-up)",
                bounds=np.stack([center - half, center + half]),
            ),
        ],
        camera=data.camera("bunny"),
    )
