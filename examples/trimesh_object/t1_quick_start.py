from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="T1",
    title="Quick start: build a mesh and read its properties",
    summary="""
    A [`Trimesh`][ordito.mesh.Trimesh] wraps a vertex buffer and a face buffer (or an existing
    `warp.Mesh`, through [`from_warp_mesh`][ordito.mesh.Trimesh.from_warp_mesh]). Every property
    is computed on the mesh's device the first time it is read and cached after that. Four inputs
    with different defects are compared here: the bunny as scanned, with five holes in its base;
    the same bunny closed by [`make_solid`][ordito.repair.make_solid]; a Möbius strip; and two
    cubes that touch at one corner, with a fin glued to one edge. The validity checks tell them
    apart. Note that [`is_watertight`][ordito.mesh.Trimesh.is_watertight] follows Open3D's
    definition, so it also requires a manifold surface with no self-intersections.
    """,
    credits=(
        ("trimesh: quick start", "https://trimesh.org/quick_start.html"),
        (
            "Open3D: mesh properties",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
        ("libigl 701", "https://libigl.github.io/tutorial/#mesh-statistics"),
        ("PyMeshLab: measures", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
    notes="""
    Volume is left blank for the three meshes that do not enclose a volume
    ([`is_volume`][ordito.mesh.Trimesh.is_volume] is false): its integral is only meaningful on a
    closed, consistently wound surface. For the same reason the orange dot is the centre of mass
    of the solid bunny and the area-weighted surface centroid elsewhere. Bodies are counted across
    edges shared by exactly two faces, so the broken box's fin is a body of its own.
    """,
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    scan = od.Trimesh(*data.load("bunny", device))
    meshes = {
        "bunny scan": scan,
        "solid bunny": od.Trimesh(*od.repair.make_solid(scan.vertices, scan.faces)),
        "Möbius strip": od.Trimesh(*data.load("mobius", device)),
        "broken box": od.Trimesh(*data.load("broken_box", device)),
    }
    for name, mesh in meshes.items():
        print(
            f"{name:>12}: {mesh.n_faces:6d} faces, {mesh.body_count} bodies, "
            f"Euler {mesh.euler_characteristic:2d}, {len(mesh.boundary_loops)} boundary loops, "
            f"watertight {mesh.is_watertight}"
        )
    # --8<-- [end:code]
    out: dict[str, Any] = {}
    for name, mesh in meshes.items():
        lower, upper = mesh.bounds
        out[name] = {
            "vertices": mesh.vertices.numpy(),
            "faces": mesh.faces.numpy(),
            "labels": mesh.face_connected_component_labels.numpy(),
            "loops": [loop.numpy() for loop in mesh.boundary_loops],
            "lower": np.array(lower),
            "upper": np.array(upper),
            "centroid": np.array(mesh.centroid),
            "center_mass": np.array(mesh.center_mass) if mesh.is_volume else None,
            "rows": {
                "vertices": f"{mesh.n_vertices}",
                "faces": f"{mesh.n_faces}",
                "bodies": f"{mesh.body_count}",
                "Euler characteristic": f"{mesh.euler_characteristic}",
                "area": f"{mesh.area:.4g}",
                "volume": f"{mesh.volume:.4g}" if mesh.is_volume else "-",
                "box diagonal": f"{mesh.enclosing_diagonal:.4g}",
                "watertight": _yes(mesh.is_watertight),
                "is a volume": _yes(mesh.is_volume),
                "orientable": _yes(mesh.is_orientable),
                "winding consistent": _yes(mesh.is_winding_consistent),
                "edge manifold": _yes(mesh.is_edge_manifold),
                "vertex manifold": _yes(mesh.is_vertex_manifold),
                "self-intersecting": _yes(mesh.is_self_intersecting),
            },
        }
    return out


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


_VIEWS = {
    "bunny scan": r.Camera(direction=(0.35, -1.0, 0.45), up=(0.0, 0.0, -1.0), zoom=1.1),
    "solid bunny": data.camera("bunny"),
    "Möbius strip": data.camera("mobius"),
    "broken box": r.Camera(direction=(1.0, -0.55, -0.25), up=(0.0, 1.0, 0.0), zoom=0.95),
}
_BODY_COLORS = np.array([[0x76, 0xB9, 0x00], [0x2C, 0x7B, 0xE8], [0xF3, 0x9C, 0x12]])


def _render(name: str, mesh: dict[str, Any]) -> r.Panel:
    corners = np.array(list(itertools.product(*zip(mesh["lower"], mesh["upper"], strict=True))))
    rims = [mesh["vertices"][loop] for loop in mesh["loops"]]
    marker = mesh["center_mass"] if mesh["center_mass"] is not None else mesh["centroid"]
    _, body = np.unique(mesh["labels"], return_inverse=True)
    layers: list[r.Layer] = [
        r.Mesh(
            mesh["vertices"],
            mesh["faces"],
            face_colors=_BODY_COLORS[body % len(_BODY_COLORS)].astype(np.uint8),
        ),
        r.WireBox(corners, color=r.GREY, width=2.0),
        r.Points(marker[None], color=r.ORANGE, size=22),
    ]
    if rims:
        layers.append(r.Lines(rims, closed=True, width=5.0))
    title = f"{name} (from below)" if name == "bunny scan" else name
    return r.Panel(layers, title=title, camera=_VIEWS[name])


def _table(result: dict[str, Any], keys: list[str], title: str) -> r.Plot:
    header = ["", "bunny scan", "solid bunny", "Möbius", "broken box"]
    rows = [[key, *(result[n]["rows"][key] for n in result)] for key in keys]

    def draw(ax: Any) -> None:
        ax.axis("off")
        widget = ax.table(
            cellText=rows,
            colLabels=header,
            colWidths=[0.29, 0.1775, 0.1775, 0.1775, 0.1775],
            loc="center",
            cellLoc="center",
        )
        widget.auto_set_font_size(False)
        widget.set_fontsize(12)
        widget.scale(1.0, 2.0)
        for (_row, col), cell in widget.get_celld().items():
            cell.set_edgecolor(r.TEXT)
            cell.set_facecolor("none")
            cell.get_text().set_color(r.TEXT)
            if col == 0:
                cell.get_text().set_horizontalalignment("left")
                cell.PAD = 0.04

    return r.Plot(draw, title=title, aspect="")


def figure(result: dict[str, Any]) -> r.Figure:
    names = list(result)
    measures = ["vertices", "faces", "bodies", "Euler characteristic", "area", "volume"]
    measures.append("box diagonal")
    checks = [
        "watertight",
        "is a volume",
        "orientable",
        "winding consistent",
        "edge manifold",
        "vertex manifold",
        "self-intersecting",
    ]
    return r.Figure(
        [
            _render(names[0], result[names[0]]),
            _render(names[1], result[names[1]]),
            _table(result, measures, "Measures"),
            _render(names[2], result[names[2]]),
            _render(names[3], result[names[3]]),
            _table(result, checks, "Validity checks"),
        ],
        ncols=3,
        link_bounds=False,
        panel_size=(640, 560),
    )
