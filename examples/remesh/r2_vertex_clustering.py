from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta
from examples.remesh.r1_quadric_decimation import HEAD

META = Meta(
    id="R2",
    title="Vertex clustering",
    summary="""
    [`cluster_decimate`][ordito.remesh.cluster_decimate] snaps every vertex to a uniform voxel
    grid, welds each occupied cell to one vertex and drops the faces that collapsed. It has no
    priority queue, so it is fully parallel, but it is a resampling rather than a simplification:
    the triangles come out uniform in size whatever the detail underneath, and thin features
    narrower than a cell can weld together.
    """,
    credits=(
        (
            "Open3D: vertex clustering",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
        ("PyMeshLab", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("dragon", device)
    points = vertices.numpy()
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    results = {}
    for fraction in (0.005, 0.015):
        results[fraction] = od.remesh.cluster_decimate(
            vertices, faces, voxel_size=fraction * diagonal
        )
        print(
            f"cells of {fraction:.1%} of the diagonal: {results[fraction][1].shape[0] // 3} faces"
        )
    # --8<-- [end:code]
    return {
        "levels": [
            (fraction, v.numpy(), f.numpy().reshape(-1, 3)) for fraction, (v, f) in results.items()
        ]
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("dragon")
    levels = result["levels"]
    panels = [r.Panel([r.Mesh(vertices, faces)], title=f"Input ({faces.shape[0]:,} faces)")]
    panels += [
        r.Panel([r.Mesh(v, f, smooth=False)], title=f"Cell {100 * s:g} % ({f.shape[0]:,} faces)")
        for s, v, f in levels
    ]
    panels.append(
        r.Panel(
            [r.Mesh(vertices, faces, show_edges=True, line_width=0.4)],
            title="Snout, input",
            bounds=HEAD,
        )
    )
    panels += [
        r.Panel(
            [r.Mesh(v, f, show_edges=True, line_width=0.4, smooth=False)],
            title=f"Snout, cell {100 * s:g} %",
            bounds=HEAD,
        )
        for s, v, f in levels
    ]
    return r.Figure(panels, camera=data.camera("dragon"), ncols=3)
