from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="H4",
    title="Joining nearby open components",
    summary="""
    A surface torn into four pieces by two narrow cracks is four components.
    [`join_closest_components`][ordito.holes.join_closest_components] repeatedly bridges the
    closest pair of boundary edges on two different open components with a two-triangle patch
    (see [`bridge_edges`][ordito.holes.bridge_edges]) until one component is left. Nothing moves
    and no vertex is added; `max_distance` would refuse joins across a wider gap.
    [`face_connected_component_labels`][ordito.adjacency.face_connected_component_labels] counts
    the components before and after.
    """,
    credits=(("MeshLib examples", "https://meshlib.io/documentation/Examples.html"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("torn_hemisphere", device)
    joined = od.holes.join_closest_components(vertices, faces)

    before = od.adjacency.face_connected_component_labels(faces).numpy()
    after = od.adjacency.face_connected_component_labels(joined).numpy()
    print(f"components: {np.unique(before).size} -> {np.unique(after).size}")
    print(f"bridge triangles: {(joined.shape[0] - faces.shape[0]) // 3}")
    # --8<-- [end:code]
    return {"labels": before, "joined": joined.numpy().reshape(-1, 3)}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("torn_hemisphere")
    labels = np.unique(result["labels"], return_inverse=True)[1]
    joined = result["joined"]
    colors = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (joined.shape[0], 1))
    colors[faces.shape[0] :] = (0xF3, 0x9C, 0x12)
    center = vertices[joined[faces.shape[0] :].ravel()].mean(axis=0)
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(vertices, faces, scalars=labels.astype(float), cmap="tab10", clim=(0, 9))],
                title="Four pieces",
            ),
            r.Panel([r.Mesh(vertices, joined, face_colors=colors)], title="Joined: one component"),
            r.Panel(
                [r.Mesh(vertices, joined, face_colors=colors, show_edges=True, line_width=0.6)],
                title="Close-up of the bridges",
                bounds=np.stack([center - 0.45, center + 0.45]),
            ),
        ],
        camera=data.camera("torn_hemisphere"),
        panel_size=(620, 600),
    )
