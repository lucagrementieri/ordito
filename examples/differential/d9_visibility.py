from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="D9",
    title="Ambient occlusion, obscurance, shape diameter and thickness",
    summary="""
    Four ray-traced per-vertex descriptors of the happy Buddha, all cast against one `wp.Mesh`
    BVH.

    - [`ambient_occlusion`][ordito.visibility.ambient_occlusion]: the share of the outward
      hemisphere blocked by the model, shown as the light that gets through.
    - [`volumetric_obscurance`][ordito.visibility.volumetric_obscurance]: the same, with each
      occluder counted `exp(-tau * t)` by its distance `t`, so only nearby geometry darkens a point;
      it brings out the folds of the robe that a far wall hides in plain occlusion.
    - [`shape_diameter`][ordito.visibility.shape_diameter]: the robust mean length of a cone of
      rays fired inwards, the local diameter of the volume.
    - [`thickness`][ordito.visibility.thickness]: twice the radius of the largest ball inside
      the volume touching the point. One ball per point is cheap but follows every small bump of
      the scan, hence the speckle where the shape diameter is smooth.
    """,
    credits=(
        ("libigl 606", "https://libigl.github.io/tutorial/#ambient-occlusion"),
        ("PyMeshLab: filters", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
        ("MeshLib: thickness", "https://meshlib.io/documentation/index.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("buddha", device)
    mesh = wp.Mesh(points=vertices, indices=faces)
    normals = od.vertices.vertex_normals(vertices, faces)
    diagonal = od.bounds.enclosing_diagonal(vertices)

    occlusion = od.visibility.ambient_occlusion(mesh, vertices, normals=normals)
    obscurance = od.visibility.volumetric_obscurance(
        mesh, vertices, normals=normals, tau=40.0 / diagonal
    )
    diameter = od.visibility.shape_diameter(mesh, vertices, normals=normals)
    thickness = od.visibility.thickness(mesh, vertices, normals=normals)
    print(f"{vertices.shape[0]} vertices, mean occlusion {occlusion.numpy().mean():.3f}")
    finite = np.isfinite(diameter.numpy())
    print(
        f"shape diameter: {finite.sum()} finite, median {np.median(diameter.numpy()[finite]):.4f}"
    )
    # --8<-- [end:code]
    return {
        "occlusion": occlusion.numpy(),
        "obscurance": obscurance.numpy(),
        "diameter": diameter.numpy(),
        "thickness": thickness.numpy(),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("buddha")
    panels = [
        r.Panel(
            [
                r.Mesh(
                    vertices,
                    faces,
                    scalars=1.0 - result["occlusion"],
                    cmap="gray",
                    clim=(0.15, 1.0),
                )
            ],
            title="Ambient occlusion",
        ),
        r.Panel(
            [
                r.Mesh(
                    vertices,
                    faces,
                    scalars=1.0 - result["obscurance"],
                    cmap="gray",
                    clim=(0.15, 1.0),
                )
            ],
            title="Volumetric obscurance",
        ),
    ]
    for key, title in (("diameter", "Shape diameter"), ("thickness", "Thickness")):
        values = np.where(np.isfinite(result[key]), result[key], np.nan)
        panels.append(
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=values,
                        cmap="viridis",
                        clim=r.percentile_clim(values, 2.0, 98.0),
                        scalar_bar=key,
                    )
                ],
                title=title,
            )
        )
    return r.Figure(
        panels,
        camera=r.Camera(direction=(0.0, 0.1, 1.0), up=(0.0, 1.0, 0.0), zoom=1.3),
        panel_size=(440, 640),
    )
