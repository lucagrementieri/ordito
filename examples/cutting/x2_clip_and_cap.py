from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="X2",
    title="Clipping and capping",
    summary="""
    [`split_mesh_with_plane`][ordito.intersection.split_mesh_with_plane] inserts the plane's
    section into the mesh as real edges and labels every face by its side;
    [`submesh_from_face_mask`][ordito.selection.submesh_from_face_mask] keeps one side, leaving
    an open cut whose rim [`boundary_loops`][ordito.boundary.boundary_loops] finds.
    [`fill_min_weight`][ordito.holes.fill_min_weight] then caps every rim with a minimum-weight
    triangulation (here the body and both ears), so the clipped bunny is closed again.
    """,
    credits=(
        (
            "PyVista: clip a closed surface",
            "https://docs.pyvista.org/examples/01-filter/clip_with_plane_box",
        ),
        ("trimesh: slice_plane", "https://trimesh.org/trimesh.intersections.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    normal, origin = wp.vec3(0.3, 0.1, -1.0), wp.vec3(-0.02, 0.1, 0.0)

    split_vertices, split_faces, above = od.intersection.split_mesh_with_plane(
        vertices, faces, normal, origin
    )
    half_vertices, half_faces = od.selection.submesh_from_face_mask(
        split_vertices, split_faces, above
    )
    rims = od.boundary.boundary_loops(half_vertices, half_faces)
    capped_faces = od.holes.fill_min_weight(half_vertices, half_faces)
    print(f"{len(rims)} rims, the cut is {max(rim.shape[0] for rim in rims)} vertices long")
    print("watertight after capping:", od.validation.is_watertight(half_vertices, capped_faces))
    # --8<-- [end:code]
    return {
        "normal": np.array(normal),
        "origin": np.array(origin),
        "vertices": half_vertices.numpy(),
        "faces": half_faces.numpy().reshape(-1, 3),
        "rims": [rim.numpy() for rim in rims],
        "capped": capped_faces.numpy().reshape(-1, 3),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    half_v, half_f, capped = result["vertices"], result["faces"], result["capped"]
    colors = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (capped.shape[0], 1))
    colors[half_f.shape[0] :] = (0xF3, 0x9C, 0x12)
    rims = [half_v[rim] for rim in result["rims"]]
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN),
                    r.Plane(result["origin"], result["normal"], size=0.8 * diagonal),
                ],
                title="Bunny and the cutting plane",
            ),
            r.Panel(
                [r.Mesh(half_v, half_f, color=r.GREEN), r.Lines(rims, closed=True, width=3)],
                title="Clipped: an open cut",
            ),
            r.Panel([r.Mesh(half_v, capped, face_colors=colors)], title="Capped"),
        ],
        camera=r.Camera(direction=(0.6, 0.3, 1.0), zoom=1.15),
    )
