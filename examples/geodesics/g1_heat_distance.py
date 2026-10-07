from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="G1",
    title="Geodesic distance with the heat method",
    summary="""
    [`heat_geodesic`][ordito.heat.heat_geodesic] approximates the distance along the surface to
    the nearest source vertex. It diffuses heat from the sources for a short time, normalizes the
    gradient of the result into a unit field, and integrates that field back into a distance.
    Given a [`Trimesh`][ordito.mesh.Trimesh], the two sparse operators this needs are built once
    and kept on the mesh -- with their factorizations, built on the second call -- so the second
    set of sources here reuses them.
    """,
    credits=(
        ("libigl 716", "https://libigl.github.io/tutorial/#heat-method"),
        ("potpourri3d", "https://github.com/nmwsharp/potpourri3d#mesh-distance"),
        ("PyMeshLab", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    mesh = od.Trimesh(*data.load("bunny", device))

    ear = wp.array([22820], dtype=wp.int32, device=device)
    landmarks = wp.array([22820, 11842, 12217], dtype=wp.int32, device=device)  # ear, nose, tail
    from_ear = od.heat.heat_geodesic(mesh, ear)
    from_landmarks = od.heat.heat_geodesic(mesh, landmarks)  # the mesh's operators, factored
    print(f"farthest point from the ear: {from_ear.numpy().max():.4f}")
    # --8<-- [end:code]
    return {"from_ear": from_ear.numpy(), "from_landmarks": from_landmarks.numpy()}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    cmap = r.striped("viridis", bands=20)
    panels = []
    for key, sources, title in (
        ("from_ear", [22820], "One source"),
        ("from_landmarks", [22820, 11842, 12217], "Three sources"),
    ):
        panels.append(
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=result[key],
                        cmap=cmap,
                        scalar_bar="distance",
                        clim=(0.0, float(result[key].max())),
                    ),
                    r.Points(vertices[sources], color=r.RED, size=18),
                ],
                title=title,
            )
        )
    return r.Figure(panels, camera=data.camera("bunny"))
