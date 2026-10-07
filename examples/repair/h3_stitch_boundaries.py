from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="H3",
    title="Stitching two boundaries",
    summary="""
    Two open cups face each other across a gap, with rims of different radius, resolution and
    tilt. [`stitch_min_weight`][ordito.holes.stitch_min_weight] joins the rims with the band of
    triangles that minimizes a stitching metric;
    [`stitch_smooth`][ordito.holes.stitch_smooth] then refines that band and fairs it into both
    surfaces. [`bridge_edges`][ordito.holes.bridge_edges] is the local operation underneath: it
    tacks one boundary edge of each rim together with two triangles, turning the two rims into
    one.
    """,
    credits=(
        ("MeshLib: stitch holes", "https://meshlib.io/documentation/ExampleMeshStitchHole.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices_a, faces_a = data.load("cup_bottom", device)
    vertices_b, faces_b = data.load("cup_top", device)
    band_vertices, band_faces = od.holes.stitch_min_weight(vertices_a, faces_a, vertices_b, faces_b)
    smooth_vertices, smooth_faces, band = od.holes.stitch_smooth(
        vertices_a, faces_a, vertices_b, faces_b, return_patch=True
    )
    print("stitched watertight:", od.validation.is_watertight(smooth_vertices, smooth_faces))

    # Bridge the closest pair of boundary edges, one on each rim.
    vertices, faces = od.combine.concatenate([(vertices_a, faces_a), (vertices_b, faces_b)])
    rim = od.boundary.oriented_boundary_edges(vertices, faces).numpy()
    on_a = rim[:, 0] < vertices_a.shape[0]
    mid = vertices.numpy()[rim].mean(axis=1)
    gap = np.linalg.norm(mid[on_a][:, None] - mid[~on_a][None], axis=2)
    i, j = np.unravel_index(gap.argmin(), gap.shape)
    bridged = od.holes.bridge_edges(vertices, faces, tuple(rim[on_a][i]), tuple(rim[~on_a][j]))
    print("rims after the bridge:", len(od.boundary.boundary_loops(vertices, bridged)))
    # --8<-- [end:code]
    return {
        "vertices": vertices.numpy(),
        "faces": faces.numpy().reshape(-1, 3),
        "band": (band_vertices.numpy(), band_faces.numpy().reshape(-1, 3)),
        "smooth": (smooth_vertices.numpy(), smooth_faces.numpy().reshape(-1, 3), band.numpy()),
        "bridged": bridged.numpy().reshape(-1, 3),
    }


def _colors(n_faces: int, added: np.ndarray) -> np.ndarray:
    colors = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (n_faces, 1))
    colors[added] = (0xF3, 0x9C, 0x12)
    return colors


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = result["vertices"], result["faces"]
    n = faces.shape[0]
    bv, bf = result["band"]
    sv, sf, band = result["smooth"]
    bridged = result["bridged"]
    tail = np.arange(bridged.shape[0]) >= n
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(vertices, faces, color=r.LIGHT_GREEN, show_edges=True, line_width=0.4)],
                title="Two cups",
            ),
            r.Panel(
                [
                    r.Mesh(
                        bv,
                        bf,
                        face_colors=_colors(bf.shape[0], np.arange(bf.shape[0]) >= n),
                        show_edges=True,
                    )
                ],
                title="stitch_min_weight",
            ),
            r.Panel(
                [
                    r.Mesh(
                        sv,
                        sf,
                        face_colors=_colors(sf.shape[0], band),
                        show_edges=True,
                        line_width=0.4,
                    )
                ],
                title="stitch_smooth",
            ),
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        bridged,
                        face_colors=_colors(bridged.shape[0], tail),
                        show_edges=True,
                    )
                ],
                title="bridge_edges",
            ),
        ],
        camera=data.camera("cup_bottom"),
        panel_size=(480, 620),
    )
