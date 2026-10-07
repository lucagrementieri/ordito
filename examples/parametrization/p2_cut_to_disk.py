from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import ListedColormap, to_rgb

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="P2",
    title="Cutting a closed surface open along seams",
    summary="""
    A torus cannot be flattened as it is: it has no boundary and two independent loops around it.
    Cutting along one loop of each kind opens it into a disk. The loops come from
    [`homology_generators`][ordito.homology.homology_generators], straightened by
    [`shorten_loop`][ordito.geodesic_walk.shorten_loop];
    [`cut_along_edges`][ordito.seams.cut_along_edges] duplicates the vertices along them so the
    two sides come apart, and [`lscm`][ordito.parametrization.lscm] flattens the result into a
    rectangle-like chart. Reading the chart back,
    [`uv_seam_edges`][ordito.seams.uv_seam_edges] finds exactly the edges where the two sides of
    the cut land in different places in UV, and
    [`seam_edge_vertices`][ordito.seams.seam_edge_vertices] turns them back into vertex pairs.
    """,
    credits=(
        ("libigl: seam edges", "https://libigl.github.io/tutorial/"),
        (
            "PyMeshLab: cut along crease edges",
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

    vertices, faces = data.load("torus", device)
    loops = od.homology.homology_generators(vertices, faces)
    loops, _ = od.geodesic_walk.shorten_loop(vertices, faces, loops, max_iter=1000)
    ring = [loop.numpy() for loop in loops]
    cut = np.concatenate([np.stack([loop, np.roll(loop, -1)], axis=1) for loop in ring])

    cut_vertices, cut_faces = od.seams.cut_along_edges(
        vertices, faces, wp.array(cut, dtype=wp.int32, device=device)
    )
    rim = od.boundary.boundary_loops(cut_vertices, cut_faces)
    print(
        f"{len(loops)} loops, {cut.shape[0]} cut edges; the cut mesh has {len(rim)} boundary loop"
    )

    b = rim[0].numpy()
    pins = wp.array([b[0], b[b.size // 2]], dtype=wp.int32, device=device)
    pinned_uv = wp.array([(0.0, 0.0), (1.0, 0.0)], dtype=wp.vec2, device=device)
    uv = od.parametrization.lscm(cut_vertices, cut_faces, pins, pinned_uv)

    seams, _, _ = od.seams.uv_seam_edges(faces, uv, face_texcoords=cut_faces)
    seam_pairs = od.seams.seam_edge_vertices(faces, seams)
    print(f"seams found from the UV chart: {seam_pairs.shape[0]} edges")
    # --8<-- [end:code]
    return {
        "loops": [loop.numpy() for loop in loops],
        "cut_vertices": cut_vertices.numpy(),
        "cut_faces": cut_faces.numpy().reshape(-1, 3),
        "uv": uv.numpy(),
        "rim": b,
        "seam_pairs": seam_pairs.numpy(),
    }


def _checker(faces: np.ndarray, uv: np.ndarray, cells: float) -> np.ndarray:
    center = uv[faces].mean(axis=1)
    size = np.ptp(uv, axis=0).max() / cells
    return ((np.floor(center[:, 0] / size) + np.floor(center[:, 1] / size)) % 2).astype(int)


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("torus")
    cut_v, cut_f, uv = result["cut_vertices"], result["cut_faces"], result["uv"]
    parity = _checker(cut_f, uv, 14)
    rgb = (np.array([to_rgb("#404040"), to_rgb(r.ORANGE)]) * 255).astype(np.uint8)
    seams = vertices[result["seam_pairs"]]
    rim = uv[result["rim"]]

    def layout(ax: Any) -> None:
        ax.tripcolor(
            uv[:, 0],
            uv[:, 1],
            cut_f,
            facecolors=parity,
            cmap=ListedColormap(["#404040", r.ORANGE]),
            edgecolors="none",
        )
        ax.plot(*np.vstack([rim, rim[:1]]).T, color=r.RED, linewidth=1.5)
        ax.set_xticks([])
        ax.set_yticks([])

    camera = r.Camera(direction=(0.0, -0.6, 1.0), up=(0.0, 1.0, 0.0), zoom=1.1)
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN),
                    r.Segments(seams, color=r.RED, width=4.0),
                ],
                title="Cut along two loops",
            ),
            r.Panel(
                [r.Mesh(cut_v, cut_f, face_colors=rgb[parity], smooth=False)],
                title="Checkerboard from the chart",
            ),
            r.Plot(layout, title="UV chart (rim in red)"),
        ],
        camera=camera,
    )
