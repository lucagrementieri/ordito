from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="I4",
    title="Feature edges: creases, convexity and boundaries",
    summary="""
    [`face_adjacency`][ordito.adjacency.face_adjacency] pairs the faces that share an edge;
    [`face_adjacency_angles`][ordito.adjacency.face_adjacency_angles] measures the dihedral angle
    across each pair and
    [`face_adjacency_convex`][ordito.adjacency.face_adjacency_convex] says which way it folds.
    [`crease_edges`][ordito.seams.crease_edges] keeps the edges that bend by more than a threshold
    angle in one call, and [`boundary_loops`][ordito.boundary.boundary_loops] orders the edges used
    by a single face into closed rims.
    """,
    credits=(
        ("PyVista: extract edges", "https://docs.pyvista.org/examples/01-filter/extract_edges"),
        ("trimesh: examples", "https://trimesh.org/examples.html"),
        ("PyMeshLab: filters", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("cad_part", device)
    adjacency, shared_edges = od.adjacency.face_adjacency(faces, return_edges=True)
    angles = od.adjacency.face_adjacency_angles(vertices, faces, adjacency)
    convex = od.adjacency.face_adjacency_convex(vertices, faces, adjacency, shared_edges)
    creases = od.seams.crease_edges(vertices, faces, angle=30.0)
    sharp = angles.numpy() > np.radians(30.0)
    print(
        f"{creases.shape[0]} crease edges: {(sharp & convex.numpy()).sum()} convex, "
        f"{(sharp & ~convex.numpy()).sum()} concave"
    )

    rim_vertices, rim_faces = data.load("hemisphere", device)
    loops = od.boundary.boundary_loops(rim_vertices, rim_faces)
    print(f"hemisphere: {len(loops)} boundary loop of {loops[0].shape[0]} vertices")
    # --8<-- [end:code]
    return {
        "angles": angles.numpy(),
        "edges": shared_edges.numpy(),
        "convex": convex.numpy(),
        "creases": creases.numpy(),
        "loops": [loop.numpy() for loop in loops],
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("cad_part")
    sharp = result["angles"] > np.radians(30.0)
    edges = result["edges"]
    rim_vertices, rim_faces = data.arrays("hemisphere")
    angles = np.degrees(result["angles"])

    def histogram(ax: Any) -> None:
        ax.hist(angles, bins=np.linspace(0.0, 180.0, 37), color=r.GREEN)
        ax.axvline(30.0, color=r.RED, linestyle="--")
        ax.set_xlabel("dihedral angle (degrees)")
        ax.set_ylabel("face pairs")

    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY, smooth=False),
                    r.Segments(vertices[result["creases"]], color=r.RED, width=4.0),
                ],
                title="Crease edges (> 30°)",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY, smooth=False),
                    r.Segments(
                        vertices[edges[sharp & result["convex"]]], color=r.ORANGE, width=4.0
                    ),
                    r.Segments(vertices[edges[sharp & ~result["convex"]]], color=r.BLUE, width=4.0),
                ],
                title="Convex (orange) and concave (blue)",
            ),
            r.Plot(histogram, title="Dihedral angles", aspect=""),
            r.Panel(
                [
                    r.Mesh(rim_vertices, rim_faces, color=r.LIGHT_GREEN, show_edges=True),
                    r.Lines([rim_vertices[loop] for loop in result["loops"]], closed=True),
                ],
                title="Boundary loop",
                camera=r.Camera(direction=(0.4, -0.9, -0.45), up=(0.0, 0.0, 1.0), zoom=1.2),
                bounds=np.stack([rim_vertices.min(axis=0), rim_vertices.max(axis=0)]),
            ),
        ],
        camera=data.camera("cad_part"),
        panel_size=(560, 520),
    )
