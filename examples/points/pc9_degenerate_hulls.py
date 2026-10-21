from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="PC9",
    title="Convex hulls of degenerate clouds",
    summary="""
    The clouds a tolerance-based hull gets wrong are the ones made of exact coincidences. A
    9 x 9 x 9 lattice has every point of a face in one plane and every point of an edge on one
    line: [`convex_hull`][ordito.points.convex_hull] keeps the 8 corners and nothing else. An
    octahedron's 6 corners, with 2 000 more points placed *exactly* on its faces (multiples of
    1/64, which `float32` stores without rounding), keep the corners alone. A disc 2e-6 thick, six
    orders of magnitude thinner than it is wide, is still a solid: its hull runs round the rim and
    over the few interior points that stand proud of their neighbours by a fraction of a micron.
    Each hull's vertex set is checked against `scipy.spatial.ConvexHull`.
    """,
    credits=(("trimesh: convex", "https://trimesh.org/trimesh.convex.html"),),
    notes="""
    The rules for degenerate input:

    - **Coplanar points on a facet** (the lattice's faces): the facet is triangulated arbitrarily,
      and the points inside it are not vertices.
    - **Collinear points along a hull edge**: only the two ends are vertices.
    - **Coincident points**: one of them is kept as the vertex.
    - **A cloud a fraction of a micron thick**: still a solid, hulled like any other.
    - **Fewer than 4 points, or no 4 spanning a tetrahedron** (an exactly flat cloud): `ValueError`,
      as there is no volume to bound.
    """,
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import scipy.spatial
    import warp as wp

    import ordito as od

    rng = np.random.default_rng(0)
    axis = np.linspace(-1.0, 1.0, 9)
    lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)

    # Points on the octahedron |x| + |y| + |z| = 1 with coordinates in multiples of 1/64.
    steps = rng.integers(0, 65, size=(4000, 2))
    steps = steps[steps.sum(axis=1) <= 64][:2000]
    signs = rng.choice([-1.0, 1.0], size=(steps.shape[0], 3))
    on_faces = signs * np.column_stack([steps, 64 - steps.sum(axis=1)]) / 64.0
    octahedron = np.vstack([np.eye(3), -np.eye(3), on_faces])

    disc = rng.uniform(-1.0, 1.0, size=(6000, 2))
    disc = disc[np.hypot(disc[:, 0], disc[:, 1]) < 1.0]
    slab = np.column_stack([disc, rng.uniform(-1e-6, 1e-6, size=disc.shape[0])])

    clouds = {"lattice": lattice, "octahedron": octahedron, "slab": slab}
    hulls = {}
    for name, cloud in clouds.items():
        points = wp.array(cloud, dtype=wp.vec3, device=device)
        faces = od.points.convex_hull(points)
        hull_vertices = np.unique(faces.numpy())
        qhull = scipy.spatial.ConvexHull(points.numpy())
        print(
            f"{name}: {points.shape[0]} points, {hull_vertices.size} hull vertices, "
            f"same as qhull: {np.array_equal(hull_vertices, np.sort(qhull.vertices))}"
        )
        hulls[name] = (points.numpy(), faces.numpy().reshape(-1, 3))
    # --8<-- [end:code]
    return hulls


def figure(result: dict[str, Any]) -> r.Figure:
    titles = {
        "lattice": "9³ lattice: 8 vertices",
        "octahedron": "Points on an octahedron: 6 vertices",
        "slab": "Disc 2e-6 thick: {} vertices",
    }
    cameras = {
        "lattice": r.Camera(direction=(1.0, 0.7, 1.4), up=(0.0, 1.0, 0.0)),
        "octahedron": r.Camera(direction=(1.0, 0.6, 1.6), up=(0.0, 1.0, 0.0)),
        "slab": r.Camera(direction=(0.2, -1.0, 0.9), up=(0.0, 0.0, 1.0)),
    }
    panels = []
    for name, (points, faces) in result.items():
        used = np.zeros(points.shape[0], dtype=bool)
        used[faces.ravel()] = True
        panels.append(
            r.Panel(
                [
                    r.Points(points[~used], color=r.GREY, size=3),
                    r.Mesh(
                        points,
                        faces,
                        color=r.GREEN,
                        opacity=0.3,
                        show_edges=True,
                        line_width=1.0,
                        smooth=False,
                    ),
                    r.Points(points[used], color=r.RED, size=8),
                ],
                title=titles[name].format(int(used.sum())),
                camera=cameras[name],
            )
        )
    return r.Figure(panels, link_bounds=False)
