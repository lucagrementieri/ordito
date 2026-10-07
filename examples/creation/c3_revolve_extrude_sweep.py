from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="C3",
    title="Revolve, extrude and sweep",
    summary="""
    Three ways to turn a 2-D outline into a solid.
    [`revolve`][ordito.creation.revolve] spins a profile about the Z axis: an open profile that
    starts and ends on the axis gives a closed vase, and a closed profile, whose last point repeats
    its first, revolved through part of a turn gets capped ends.
    [`extrude_polygon`][ordito.creation.extrude_polygon] triangulates a star and raises it into a
    prism. [`sweep_polygon`][ordito.creation.sweep_polygon] carries the
    same star along a trefoil knot, rolling it twice around the path's tangent on the way.
    """,
    credits=(
        ("trimesh: creation", "https://trimesh.org/trimesh.creation.html"),
        ("PyVista: extrude rotate", "https://docs.pyvista.org/examples/01-filter/extrude_rotate"),
        ("MeshLib: extrude", "https://meshlib.io/documentation/ExampleMeshExtrude.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od

    def vec2(points: np.ndarray) -> wp.array[wp.vec2]:
        return wp.array(points, dtype=wp.vec2, device=device)

    # A vase: radius against height, from the axis at the bottom to the axis at the top.
    height = np.linspace(0.0, 2.0, 40)
    radius = 0.55 + 0.25 * np.sin(2.6 * height + 0.6) - 0.1 * height
    profile = np.concatenate([[[0.0, 0.0]], np.stack([radius, height], 1), [[0.0, 2.0]]])
    vase = od.creation.revolve(vec2(profile), sections=64)

    # A closed section (its last point repeats the first) revolved through three quarters
    # of a turn, with capped ends.
    t = np.linspace(0.0, 2.0 * np.pi, 49)
    section = np.stack([1.0 + 0.3 * np.cos(t), 0.5 * np.sin(t)], 1)
    ring = od.creation.revolve(vec2(section), angle=1.5 * np.pi, cap=True, sections=48)

    angle = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    radii = np.where(np.arange(10) % 2, 0.45, 1.0)
    star = np.stack([radii * np.cos(angle), radii * np.sin(angle)], 1)
    prism = od.creation.extrude_polygon(vec2(star), 0.5)

    s = np.linspace(0.0, 2.0 * np.pi, 301)
    knot = np.stack(
        [np.sin(s) + 2 * np.sin(2 * s), np.cos(s) - 2 * np.cos(2 * s), -np.sin(3 * s)], 1
    )
    swept = od.creation.sweep_polygon(
        vec2(0.45 * star),
        wp.array(knot, dtype=wp.vec3, device=device),
        angles=wp.array(np.linspace(0.0, 4.0 * np.pi, 301), dtype=wp.float32, device=device),
    )
    for name, (v, f) in [("vase", vase), ("ring", ring), ("prism", prism), ("knot", swept)]:
        print(
            f"{name:>5}: {f.shape[0] // 3:5d} faces, watertight {od.validation.is_watertight(v, f)}"
        )
    # --8<-- [end:code]
    return {
        name: (v.numpy(), f.numpy(), *extra)
        for name, (v, f), *extra in [
            ("vase", vase, profile),
            ("ring", ring, section),
            ("prism", prism, star),
            ("knot", swept, knot),
        ]
    }


def figure(result: dict[str, Any]) -> r.Figure:
    view = r.Camera(direction=(1.0, -1.4, 0.8), up=(0.0, 0.0, 1.0))
    panels = []
    for name, title in [
        ("vase", "revolve: open profile"),
        ("ring", "revolve: 270°, capped"),
        ("prism", "extrude_polygon"),
        ("knot", "sweep_polygon, rolled"),
    ]:
        v, f, curve = result[name]
        if name == "knot":
            guide = [curve]
        elif name == "prism":
            guide = [np.c_[curve, np.full(len(curve), 0.5)]]
        else:
            guide = [np.c_[curve[:, 0], np.zeros(len(curve)), curve[:, 1]]]
        panels.append(
            r.Panel(
                [
                    r.Mesh(v, f, show_edges=name == "prism", line_width=0.8),
                    r.Lines(guide, color=r.ORANGE, width=3.0, closed=name in ("ring", "prism")),
                ],
                title=title,
                camera=r.Camera(direction=(0.0, 0.0, 1.0)) if name == "knot" else None,
            )
        )
    return r.Figure(panels, camera=view, link_bounds=False, panel_size=(520, 520))
