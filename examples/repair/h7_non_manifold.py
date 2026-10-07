from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="H7",
    title="Non-manifold repair",
    summary="""
    Two cubes meet at a single corner (a non-manifold vertex: two fans of faces touch only
    there), and a fin is glued to one cube's edge (a non-manifold edge with three faces).
    [`vertex_manifold_mask`][ordito.validation.vertex_manifold_mask] and
    [`edge_manifold_mask`][ordito.validation.edge_manifold_mask] find them. Two repairs with
    different trade-offs:
    [`remove_non_manifold_faces`][ordito.repair.remove_non_manifold_faces] deletes every face
    on a non-manifold edge, while
    [`split_non_manifold_vertices`][ordito.repair.split_non_manifold_vertices] keeps every face
    and duplicates vertices instead, so the cubes come apart at the corner and the fin is cut
    loose along its edge (pulled apart slightly in the image). Deleting faces cannot fix the
    shared corner, so only the split result is manifold.
    """,
    credits=(
        ("PyMeshLab", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
        (
            "Open3D: mesh properties",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("broken_box", device)
    bad_vertices = ~od.validation.vertex_manifold_mask(vertices, faces).numpy()
    bad_faces = ~od.validation.edge_manifold_mask(faces).numpy()
    print(f"non-manifold vertices: {bad_vertices.sum()}")
    print(f"faces on a non-manifold edge: {bad_faces.sum()}")

    removed_vertices, removed_faces = od.repair.remove_non_manifold_faces(vertices, faces)
    split_vertices, split_faces, source = od.repair.split_non_manifold_vertices(vertices, faces)
    for name, (v, f) in {
        "removed": (removed_vertices, removed_faces),
        "split": (split_vertices, split_faces),
    }.items():
        manifold = od.validation.is_vertex_manifold(f) and od.validation.is_edge_manifold(f)
        print(f"{name}: {f.shape[0] // 3} faces, {v.shape[0]} vertices, manifold: {manifold}")
    # --8<-- [end:code]
    return {
        "bad_vertices": bad_vertices,
        "bad_faces": bad_faces,
        "removed": (removed_vertices.numpy(), removed_faces.numpy().reshape(-1, 3)),
        "split": (split_vertices.numpy(), split_faces.numpy().reshape(-1, 3), source.numpy()),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("broken_box")
    colors = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (faces.shape[0], 1))
    colors[result["bad_faces"]] = (0xC2, 0x18, 0x9B)
    rv, rf = result["removed"]
    sv, sf, source = result["split"]
    # Pull the split copies apart a little so the duplicated vertices can be seen.
    centroid = sv[sf].mean(axis=1)
    shift = np.zeros_like(sv)
    counts = np.zeros(sv.shape[0])
    for k in range(3):
        np.add.at(shift, sf[:, k], centroid)
        np.add.at(counts, sf[:, k], 1)
    shift = shift / np.maximum(counts, 1)[:, None] - sv
    duplicated = np.bincount(source, minlength=vertices.shape[0])[source] > 1
    exploded = sv + 0.15 * shift * duplicated[:, None]
    edges: r.MeshStyle = {"show_edges": True, "smooth": False}
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, face_colors=colors, **edges),
                    r.Points(vertices[result["bad_vertices"]], color=r.RED, size=22),
                ],
                title="Non-manifold vertex and edge",
            ),
            r.Panel([r.Mesh(rv, rf, **edges)], title="remove_non_manifold_faces"),
            r.Panel(
                [
                    r.Mesh(exploded, sf, **edges),
                    r.Points(exploded[duplicated], color=r.ORANGE, size=16),
                ],
                title="split_non_manifold_vertices",
            ),
        ],
        camera=r.Camera(direction=(1.0, -0.8, 0.45), up=(0.0, 0.0, 1.0), zoom=1.1),
        panel_size=(620, 600),
    )
