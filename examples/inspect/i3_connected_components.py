from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="I3",
    title="Connected components: label, split, remove debris",
    summary="""
    The bunny scan is surrounded by small floating blobs, the debris a scanner leaves behind.
    [`face_connected_component_labels`][ordito.adjacency.face_connected_component_labels] gives
    every face the label of its edge-connected component, [`split`][ordito.combine.split] cuts the
    mesh into one mesh per component, and
    [`remove_small_components`][ordito.repair.remove_small_components] drops every component
    below a face count (or area, or diameter) in one call.
    """,
    credits=(
        (
            "Open3D: connected components",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
        ("PyVista: connectivity", "https://docs.pyvista.org/examples/01-filter/connectivity"),
        ("PyMeshLab: filters", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny_debris", device)
    labels = od.adjacency.face_connected_component_labels(faces)
    parts = od.combine.split(vertices, faces)
    sizes = sorted((part_faces.shape[0] // 3 for _, part_faces in parts), reverse=True)
    print(f"{len(parts)} components; the largest has {sizes[0]} faces, the next {sizes[1]}")

    clean_vertices, clean_faces = od.repair.remove_small_components(vertices, faces, min_faces=1000)
    print(f"kept {clean_faces.shape[0] // 3} of {faces.shape[0] // 3} faces")
    # --8<-- [end:code]
    _, rank = np.unique(labels.numpy(), return_inverse=True)
    return {
        "component": rank,
        "clean_vertices": clean_vertices.numpy(),
        "clean_faces": clean_faces.numpy(),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny_debris")
    rng = np.random.default_rng(0)
    palette = (rng.uniform(0.25, 0.95, (int(result["component"].max()) + 1, 3)) * 255).astype(
        np.uint8
    )
    colors = palette[result["component"]]
    largest = np.bincount(result["component"]).argmax()
    colors[result["component"] == largest] = (0x76, 0xB9, 0x00)
    return r.Figure(
        [
            r.Panel([r.Mesh(vertices, faces, face_colors=colors)], title="Components"),
            r.Panel(
                [r.Mesh(result["clean_vertices"], result["clean_faces"])],
                title="Small components removed",
            ),
        ],
        camera=data.camera("bunny_debris"),
    )
