from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="I1",
    title="Validation masks: boundaries, non-manifold elements, orientation",
    summary="""
    Three small meshes, each broken in a different way. The Möbius strip has one boundary loop and
    no consistent orientation, Boy's surface is closed but non-orientable, and the broken box has
    a fin on one edge (an edge with three faces) and two cubes meeting at one corner (a vertex
    whose faces form two fans). The per-element masks locate each defect:
    [`face_watertight_mask`][ordito.validation.face_watertight_mask],
    [`edge_manifold_mask`][ordito.validation.edge_manifold_mask],
    [`vertex_manifold_mask`][ordito.validation.vertex_manifold_mask] and
    [`edge_winding_consistent_mask`][ordito.validation.edge_winding_consistent_mask];
    [`boundary_edges`][ordito.boundary.boundary_edges] lists the rims and
    [`is_orientable`][ordito.validation.is_orientable] gives the global verdict.
    """,
    notes="""
    Magenta faces touch an edge used by three faces, magenta dots are non-manifold vertices, red
    edges are boundary edges and orange edges are traversed in the same direction by both of their
    faces (where the orientation flips). Both ends of the fin's edge are reported as non-manifold
    vertices along with the shared corner.
    """,
    credits=(
        (
            "Open3D: mesh properties",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
        ("PyVista: mesh quality", "https://docs.pyvista.org/examples/01-filter/mesh_quality"),
    ),
)

NAMES = ("mobius", "boy", "broken_box")


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    results = {}
    for name in ("mobius", "boy", "broken_box"):
        vertices, faces = data.load(name, device)
        results[name] = {
            "open_faces": ~od.validation.face_watertight_mask(faces).numpy(),
            "non_manifold_faces": ~od.validation.edge_manifold_mask(faces).numpy(),
            "non_manifold_vertices": ~od.validation.vertex_manifold_mask(vertices, faces).numpy(),
            "boundary_edges": od.boundary.boundary_edges(vertices, faces).numpy(),
            "orientable": od.validation.is_orientable(faces),
            "inconsistent_edges": int(
                (~od.validation.edge_winding_consistent_mask(faces).numpy()).sum()
            ),
        }
        m = results[name]
        print(
            f"{name}: {len(m['boundary_edges'])} boundary edges, "
            f"{m['non_manifold_faces'].sum()} faces on non-manifold edges, "
            f"{m['non_manifold_vertices'].sum()} non-manifold vertices, "
            f"{m['inconsistent_edges']} edges wound the same way by both faces, "
            f"orientable: {m['orientable']}"
        )
    # --8<-- [end:code]
    return results


def _same_direction_edges(faces: np.ndarray) -> np.ndarray:
    """Edges whose two faces traverse them in the same direction (the orientation seam)."""
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    rows, counts = np.unique(directed, axis=0, return_counts=True)
    return rows[counts > 1]


def figure(result: dict[str, Any]) -> r.Figure:
    titles = {"mobius": "Möbius strip", "boy": "Boy's surface", "broken_box": "Broken box"}
    panels: list[r.Panel | r.Table] = []
    for name in NAMES:
        vertices, faces = data.arrays(name)
        m = result[name]
        colors = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (faces.shape[0], 1))
        colors[m["non_manifold_faces"]] = (0xC2, 0x18, 0x9B)
        layers: list[r.Layer] = [r.Mesh(vertices, faces, face_colors=colors, smooth=False)]
        seam = _same_direction_edges(faces)
        if len(seam):
            layers.append(r.Segments(vertices[seam], color=r.ORANGE, width=4.0))
        if len(m["boundary_edges"]):
            layers.append(r.Segments(vertices[m["boundary_edges"]], color=r.RED, width=6.0))
        bad = np.flatnonzero(m["non_manifold_vertices"])
        if bad.size:
            layers.append(r.Points(vertices[bad], color=r.MAGENTA, size=26))
        camera = data.camera(name)
        if name == "broken_box":  # side on, so the shared corner and the fin are both in view
            camera = r.Camera(direction=(0.7, -1.0, 0.35), up=(0.0, 0.0, 1.0), zoom=1.0)
        else:
            camera = dataclasses.replace(camera, zoom=max(camera.zoom, 1.25))
        panels.append(r.Panel(layers, title=titles[name], camera=camera))
    rows = [["", "boundary", "non-manifold", "orientable"]]
    for name in NAMES:
        m = result[name]
        rows.append(
            [
                titles[name],
                str(len(m["boundary_edges"])),
                f"{m['non_manifold_faces'].sum()} f / {m['non_manifold_vertices'].sum()} v",
                "yes" if m["orientable"] else "no",
            ]
        )
    panels.append(r.Table(rows, title="Summary"))
    return r.Figure(panels, link_bounds=False, panel_size=(600, 560))
