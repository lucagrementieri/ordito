from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="R1",
    title="Quadric decimation",
    summary="""
    [`quadric_decimate`][ordito.remesh.quadric_decimate] simplifies the dragon to a face budget by
    Garland-Heckbert edge collapses: each collapse is priced by how far it moves the surface, so
    flat regions are thinned first and the scales, teeth and creases survive longest. The
    wireframe close-ups of the snout show the same three levels. Nothing is frozen by default
    (``feature_angle=180``): the quadric error itself is what keeps creases, so even a scan full of
    sharp edges reaches a 1 % budget.
    """,
    credits=(
        ("MeshLib: decimate", "https://meshlib.io/documentation/ExampleMeshDecimate.html"),
        (
            "Open3D: mesh simplification",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
        ("PyVista: decimation", "https://docs.pyvista.org/examples/01-filter/decimate"),
        ("PyMeshLab", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
        ("libigl 703", "https://libigl.github.io/tutorial/#mesh-decimation"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("dragon", device)
    results = {}
    for ratio in (0.1, 0.01):
        results[ratio] = od.remesh.quadric_decimate(vertices, faces, target_ratio=ratio)
        print(f"{ratio:.0%}: {faces.shape[0] // 3} -> {results[ratio][1].shape[0] // 3} faces")
    # --8<-- [end:code]
    return {
        "levels": [
            (ratio, v.numpy(), f.numpy().reshape(-1, 3)) for ratio, (v, f) in results.items()
        ]
    }


HEAD = np.array([[-0.112, 0.11, -0.03], [-0.062, 0.16, 0.03]])
"""Bounds of the dragon's snout, for the wireframe close-ups."""


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("dragon")
    levels = [(1.0, vertices, faces), *result["levels"]]
    panels = [
        r.Panel([r.Mesh(v, f)], title=f"{100 * ratio:g} % ({f.shape[0]:,} faces)")
        for ratio, v, f in levels
    ]
    panels += [
        r.Panel(
            [r.Mesh(v, f, show_edges=True, line_width=0.4, smooth=False)],
            title=f"Snout, {100 * ratio:g} %",
            bounds=HEAD,
        )
        for ratio, v, f in levels
    ]
    return r.Figure(panels, camera=data.camera("dragon"), ncols=3)
