from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="V5",
    title="Voxel morphology",
    summary="""
    Binary morphology on the solid voxelized bunny of the previous example, two steps each with
    the 6-neighbourhood: [`dilate`][ordito.voxels.dilate] grows the set by a shell of cells,
    [`erode`][ordito.voxels.erode] peels one off (the thin ears go first),
    [`opening`][ordito.voxels.opening] (erode then dilate) removes features thinner than the
    structuring element while keeping the bulk, and [`closing`][ordito.voxels.closing] (dilate
    then erode) fills narrow gaps and dents. Added cells are orange; the cells opening removes
    are faint red.
    """,
    credits=(("trimesh: voxel morphology", "https://trimesh.org/trimesh.voxel.morphology.html"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    faces = od.holes.fill_min_weight(vertices, faces)
    solid = od.voxels.voxelize_mesh(vertices, faces, voxel_size=0.004, mode="solid")

    results = {"input": solid}
    for operation in (od.voxels.dilate, od.voxels.erode, od.voxels.opening, od.voxels.closing):
        results[operation.__name__] = operation(solid, iterations=2)
    for name, grid in results.items():
        print(f"{name}: {grid.get_active_stats().voxel_count} voxels")
    # --8<-- [end:code]
    return {name: od.voxels.cell_centers(g).numpy() for name, g in results.items()}


def _keys(centers: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(c) for c in np.round(centers / 0.004).astype(int).tolist()}


def figure(result: dict[str, Any]) -> r.Figure:
    original = _keys(result["input"])
    panels = []
    for name, centers in result.items():
        keys = _keys(centers)
        layers: list[r.Layer] = [
            r.Boxes(centers, 0.004, color=r.LIGHT_GREEN if name == "input" else r.GREEN)
        ]
        added = np.array([c for c in keys - original]) * 0.004
        removed = np.array([c for c in original - keys]) * 0.004
        if added.size:
            layers.append(r.Boxes(added, 0.004, color=r.ORANGE))
        if removed.size and name == "opening":
            layers.append(r.Boxes(removed, 0.004, color=r.RED, opacity=0.25, show_edges=False))
        panels.append(r.Panel(layers, title=name))
    return r.Figure(panels, camera=data.camera("bunny"), panel_size=(440, 440))
