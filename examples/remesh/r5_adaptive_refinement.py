from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="R5",
    title="Refining to a size or to the surrounding density",
    summary="""
    Filling the bunny's holes with [`fill_min_weight`][ordito.holes.fill_min_weight] leaves
    patches of long triangles. Three refinements bring them to the scan's sampling:

    - [`subdivide_to_size`][ordito.remesh.subdivide_to_size] bisects every edge of the mesh
      longer than a bound, crack-free;
    - [`subdivide_region_to_size`][ordito.remesh.subdivide_region_to_size] does the same inside
      a face region only, with Delaunay flips between passes;
    - [`refine_region_to_density`][ordito.remesh.refine_region_to_density] needs no length at
      all: it splits a patch triangle until its size matches its corners' surroundings (Liepa's
      criterion).
    """,
    credits=(
        (
            "MeshLib: mesh modification",
            "https://meshlib.io/documentation/ExampleMeshModification.html",
        ),
        ("PyMeshLab", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
        ("trimesh: subdivide_to_size", "https://trimesh.org/trimesh.remesh.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("holey_bunny", device)
    filled = od.holes.fill_min_weight(vertices, faces)
    is_patch = np.arange(filled.shape[0] // 3) >= faces.shape[0] // 3  # fill faces come last
    patch = wp.array(is_patch, dtype=wp.bool, device=device)
    max_edge = 1.5 * od.edges.mean_edge_length(vertices, faces)

    everywhere = od.remesh.subdivide_to_size(vertices, filled, max_edge, return_index=True)
    in_region = od.remesh.subdivide_region_to_size(vertices, filled, patch, max_edge)
    to_density = od.remesh.refine_region_to_density(vertices, filled, patch)
    for name, (_, new_faces, *_) in {
        "everywhere": everywhere,
        "in the patches": in_region,
        "to density": to_density,
    }.items():
        print(f"{name}: {filled.shape[0] // 3} -> {new_faces.shape[0] // 3} faces")
    # --8<-- [end:code]
    loops = od.boundary.boundary_loops(vertices, faces)
    rim = vertices.numpy()[max(loops, key=lambda loop: loop.shape[0]).numpy()]
    return {
        "filled": filled.numpy().reshape(-1, 3),
        "is_patch": is_patch,
        "everywhere": tuple(a.numpy() for a in everywhere),
        "in_region": tuple(a.numpy() for a in in_region),
        "to_density": tuple(a.numpy() for a in to_density),
        "rim": rim,
    }


def _colors(is_patch: np.ndarray) -> np.ndarray:
    colors = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (is_patch.size, 1))
    colors[is_patch] = (0xF3, 0x9C, 0x12)
    return colors


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, _ = data.arrays("holey_bunny")
    rim = result["rim"]
    center, half = rim.mean(axis=0), 0.9 * np.ptp(rim, axis=0).max()
    close = np.stack([center - half, center + half])

    def panel(v: np.ndarray, f: np.ndarray, is_patch: np.ndarray, title: str) -> r.Panel:
        mesh = r.Mesh(
            v, f.reshape(-1, 3), face_colors=_colors(is_patch), show_edges=True, line_width=0.6
        )
        return r.Panel([mesh], title=title, bounds=close)

    filled, is_patch = result["filled"], result["is_patch"]
    v1, f1, parent = result["everywhere"]
    v2, f2, region2 = result["in_region"]
    v3, f3, region3 = result["to_density"]
    return r.Figure(
        [
            panel(vertices, filled, is_patch, "Filled"),
            panel(v1, f1, is_patch[parent], "subdivide_to_size"),
            panel(v2, f2, region2, "subdivide_region_to_size"),
            panel(v3, f3, region3, "refine_region_to_density"),
        ],
        camera=data.camera("holey_bunny"),
        ncols=4,
        panel_size=(560, 560),
    )
