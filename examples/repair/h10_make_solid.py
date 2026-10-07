from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="H10",
    title="One-call watertight solid",
    summary="""
    The wrecked bunny has everything at once: punched holes, 5 % of its faces flipped, an
    unwelded band of duplicated vertices and forty floating blobs of debris. After welding the
    duplicates with [`remove_duplicated_vertices`][ordito.repair.remove_duplicated_vertices],
    [`make_solid`][ordito.repair.make_solid] runs the whole repair loop: connectivity repair and
    consistent winding, debris removal, hole filling, then degeneracy and self-intersection
    passes until nothing changes. [`make_volume`][ordito.repair.make_volume] finally turns the
    closed result outward. The table compares the two with the
    [`ordito.validation`][ordito.validation] predicates.
    """,
    credits=(
        ("pymeshfix", "https://pymeshfix.pyvista.org/examples/index.html"),
        ("MeshLib examples", "https://meshlib.io/documentation/Examples.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("wrecked_bunny", device)
    welded, _, _, welded_faces = od.repair.remove_duplicated_vertices(vertices, faces, epsilon=1e-6)
    solid, solid_faces = od.repair.make_solid(welded, welded_faces)
    solid_faces = od.repair.make_volume(solid, solid_faces)

    def report(v: wp.array[wp.vec3], f: wp.array[wp.int32]) -> dict[str, object]:
        labels = od.adjacency.face_connected_component_labels(f).numpy()
        return {
            "faces": f.shape[0] // 3,
            "components": np.unique(labels).size,
            "holes": len(od.boundary.boundary_loops(v, f)),
            "winding consistent": od.validation.is_winding_consistent(f),
            "watertight": od.validation.is_watertight(v, f),
            "volume": round(od.measures.volume(v, f), 6),
        }

    before, after = report(vertices, faces), report(solid, solid_faces)
    for key in before:
        print(f"{key}: {before[key]} -> {after[key]}")
    # --8<-- [end:code]
    return {
        "solid": (solid.numpy(), solid_faces.numpy().reshape(-1, 3)),
        "before": before,
        "after": after,
    }


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3g}"
    return f"{value:,}"


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("wrecked_bunny")
    sv, sf = result["solid"]
    rows = [["", "input", "make_solid"]] + [
        [key, _cell(result["before"][key]), _cell(result["after"][key])] for key in result["before"]
    ]
    view = data.camera("wrecked_bunny")
    return r.Figure(
        [
            r.Panel([r.Mesh(vertices, faces, color=r.LIGHT_GREEN)], title="Wrecked bunny"),
            r.Panel([r.Mesh(sv, sf)], title="make_solid"),
            r.Panel(
                [r.Mesh(sv, sf)],
                title="make_solid (from below)",
                camera=r.Camera(direction=(0.3, -1.0, 0.4), up=(0.0, 0.0, -1.0), zoom=view.zoom),
            ),
            r.Table(rows, title="Validation"),
        ],
        camera=view,
        panel_size=(560, 520),
    )
