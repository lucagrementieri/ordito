from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="X5",
    title="Growing and shrinking selections",
    summary="""
    A vertex selection grows by one ring of neighbours per hop with
    [`expand_vertex_mask`][ordito.selection.expand_vertex_mask]: from three seed vertices,
    eighteen hops give three patches, two of which have merged. Colouring each vertex by the hop
    that first reached it shows the rings.
    [`shrink_vertex_mask`][ordito.selection.shrink_vertex_mask] erodes the selection again by six
    rings, peeling a band off its whole outline, and
    [`submesh_from_vertex_mask`][ordito.selection.submesh_from_vertex_mask] cuts the result out
    as its own mesh.
    """,
    credits=(
        (
            "PyMeshLab: selection dilate / erode",
            "https://pymeshlab.readthedocs.io/en/latest/filter_list.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    seeds = np.zeros(vertices.shape[0], dtype=bool)
    seeds[[18577, 9802, 4939]] = True
    mask = wp.array(seeds, dtype=wp.bool, device=device)

    # Grow one hop at a time and remember when each vertex joined.
    hop = np.where(seeds, 0, -1)
    for k in range(1, 19):
        mask = od.selection.expand_vertex_mask(faces, mask, 1)
        hop[(hop < 0) & mask.numpy()] = k

    shrunk = od.selection.shrink_vertex_mask(faces, mask, 6)
    patch_vertices, patch_faces = od.selection.submesh_from_vertex_mask(vertices, faces, shrunk)
    print(f"grown: {int(mask.numpy().sum())} vertices, shrunk: {int(shrunk.numpy().sum())}")
    print(f"extracted patch: {patch_faces.shape[0] // 3} faces")
    # --8<-- [end:code]
    return {
        "hop": hop,
        "grown": mask.numpy(),
        "shrunk": shrunk.numpy(),
        "patch_vertices": patch_vertices.numpy(),
        "patch_faces": patch_faces.numpy().reshape(-1, 3),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    hop = result["hop"]
    selected = result["grown"]
    region = selected[faces].all(axis=1)
    # Per-face colour: grey outside, then by hop count.
    face_hop = hop[faces].max(axis=1)
    cmap = r.striped("viridis", bands=19)
    colors = np.tile(np.array([200, 200, 200], dtype=np.uint8), (faces.shape[0], 1))
    colors[region] = (255 * np.asarray(cmap(face_hop[region] / 18.0))[:, :3]).astype(np.uint8)
    grown_faces = selected[faces].all(axis=1)
    shrunk_faces = result["shrunk"][faces].all(axis=1)
    morph = np.tile(np.array([200, 200, 200], dtype=np.uint8), (faces.shape[0], 1))
    morph[grown_faces] = (0xF3, 0x9C, 0x12)
    morph[shrunk_faces] = (0x76, 0xB9, 0x00)
    seeds = vertices[[18577, 9802, 4939]]
    center = seeds.mean(axis=0)
    close = np.stack([center - 0.035, center + 0.035])
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, face_colors=colors),
                    r.Points(seeds, color=r.RED, size=14),
                ],
                title="Grown 18 hops, by hop count",
                bounds=close,
            ),
            r.Panel(
                [r.Mesh(vertices, faces, face_colors=morph)],
                title="Shrunk 6 hops (green), removed (orange)",
                bounds=close,
            ),
            r.Panel(
                [r.Mesh(result["patch_vertices"], result["patch_faces"])],
                title="submesh_from_vertex_mask",
                bounds=close,
            ),
        ],
        camera=r.Camera(direction=(0.1, 0.15, 1.0)),
    )
