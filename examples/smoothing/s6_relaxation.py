from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="S6",
    title="Relaxation: even triangle areas and a local surface fit",
    summary="""
    Two filters that move vertices for reasons other than diffusion.
    [`equalize_triangle_areas`][ordito.smoothing.equalize_triangle_areas] moves each vertex to
    the point that minimizes the summed squared areas of its triangles, which evens out the
    scan's triangle sizes without changing its connectivity (colour: each triangle's area over
    the mean on a log scale from 1/3 to 3, blue smaller and red larger, from
    [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]).
    [`relax_approx`][ordito.smoothing.relax_approx] fits a quadric to every vertex's geodesic
    neighbourhood and moves the vertex onto it, so noise smaller than the neighbourhood is
    removed in a few passes while the curvature of the shape is kept; its radius here is two
    mean edge lengths from [`edges_unique_length`][ordito.edges.edges_unique_length].
    """,
    credits=(
        ("MeshLib: relax", "https://meshlib.io/documentation/Examples.html"),
        ("PyMeshLab: smoothing", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    even = od.smoothing.equalize_triangle_areas(vertices, faces, iterations=5)
    for name, positions in (("scan", vertices), ("equalized", even)):
        _, areas = od.triangles.face_normals_and_areas(positions, faces)
        spread = areas.numpy().std() / areas.numpy().mean()
        print(f"{name:>9}: triangle area spread (std / mean) {spread:.3f}")

    noisy, _ = data.load("noisy_bunny", device)
    edge = od.edges.edges_unique_length(vertices, faces).numpy().mean()
    relaxed = od.smoothing.relax_approx(noisy, faces, 2 * edge, iterations=3, fit="quadric")
    clean = vertices.numpy()
    for name, positions in (("noisy", noisy), ("relaxed", relaxed)):
        error = np.linalg.norm(positions.numpy() - clean, axis=1).mean()
        print(f"{name:>9}: mean distance from the clean scan {error:.2e}")
    # --8<-- [end:code]
    return {"even": even.numpy(), "noisy": noisy.numpy(), "relaxed": relaxed.numpy()}


def _area_ratio(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return np.log2(np.maximum(area / area.mean(), 1e-3))


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    center = vertices[5088]
    close = np.stack([center - 0.012, center + 0.012])
    bunny = data.camera("bunny")

    def areas(positions: np.ndarray, title: str) -> r.Panel:
        return r.Panel(
            [
                r.Mesh(
                    positions,
                    faces,
                    scalars=_area_ratio(positions, faces),
                    cmap="RdBu_r",
                    clim=(-1.5, 1.5),
                    show_edges=True,
                    line_width=0.4,
                    smooth=False,
                )
            ],
            title=title,
            bounds=close,
        )

    return r.Figure(
        [
            areas(vertices, "Scan: triangle areas"),
            areas(result["even"], "After equalize_triangle_areas"),
            r.Panel([r.Mesh(result["noisy"], faces, color=r.LIGHT_GREEN)], title="Noisy"),
            r.Panel(
                [r.Mesh(result["relaxed"], faces, color=r.GREEN)], title="relax_approx (quadric)"
            ),
        ],
        camera=bunny,
        link_bounds=False,
        panel_size=(540, 520),
    )
