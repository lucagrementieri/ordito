from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="H1",
    title="Filling holes: fan, minimum weight and smooth",
    summary="""
    The bunny has holes punched through its side and the scan's own openings in its base.
    [`boundary_loops`][ordito.boundary.boundary_loops] finds every rim, and three fills close them:

    - [`fill_fan`][ordito.holes.fill_fan] fans each hole from one rim vertex;
    - [`fill_min_weight`][ordito.holes.fill_min_weight] picks the minimum-weight triangulation of
      each rim;
    - [`fill_smooth`][ordito.holes.fill_smooth] refines that patch and fairs it into the
      surrounding surface.
    """,
    credits=(
        ("MeshLib: fill holes", "https://meshlib.io/documentation/ExampleMeshFillHole.html"),
        ("PyVista: fill_holes", "https://docs.pyvista.org/examples/01-filter/fill_holes"),
        ("pymeshfix: bunny", "https://pymeshfix.pyvista.org/examples/bunny.html"),
        ("PyMeshLab: close holes", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("holey_bunny", device)
    loops = od.boundary.boundary_loops(vertices, faces)
    print(f"{len(loops)} holes, rims of {sorted(loop.shape[0] for loop in loops)} vertices")

    fan_faces = od.holes.fill_fan(vertices, faces)
    min_weight_faces = od.holes.fill_min_weight(vertices, faces)
    smooth_vertices, smooth_faces, patch = od.holes.fill_smooth(vertices, faces, return_patch=True)
    print(
        "watertight after the smooth fill:",
        od.validation.is_watertight(smooth_vertices, smooth_faces),
    )
    # --8<-- [end:code]
    return {
        "loops": [loop.numpy() for loop in loops],
        "fan": fan_faces.numpy(),
        "min_weight": min_weight_faces.numpy(),
        "smooth_vertices": smooth_vertices.numpy(),
        "smooth_faces": smooth_faces.numpy(),
        "patch": patch.numpy(),
    }


def _tinted(n_faces: int, n_original: int) -> np.ndarray:
    colors = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (n_faces, 1))
    colors[n_original:] = (0xF3, 0x9C, 0x12)
    return colors


def _around(rim: np.ndarray) -> np.ndarray:
    """Bounds framing one hole's rim with some margin."""
    center, half = rim.mean(axis=0), 1.1 * np.ptp(rim, axis=0).max()
    return np.stack([center - half, center + half])


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("holey_bunny")
    n = faces.shape[0]
    rims = [vertices[loop] for loop in result["loops"]]
    fan = result["fan"].reshape(-1, 3)
    weight = result["min_weight"].reshape(-1, 3)
    smooth = result["smooth_faces"].reshape(-1, 3)
    smooth_colors = _tinted(smooth.shape[0], smooth.shape[0])
    smooth_colors[result["patch"]] = (0xF3, 0x9C, 0x12)
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(vertices, faces, color=r.LIGHT_GREEN), r.Lines(rims, closed=True)],
                title="Holes",
            ),
            r.Panel([r.Mesh(vertices, fan, face_colors=_tinted(fan.shape[0], n))], title="Fan"),
            r.Panel(
                [r.Mesh(vertices, weight, face_colors=_tinted(weight.shape[0], n))],
                title="Minimum weight",
            ),
            r.Panel(
                [r.Mesh(result["smooth_vertices"], smooth, face_colors=smooth_colors)],
                title="Smooth",
            ),
            r.Panel(
                [
                    r.Mesh(
                        result["smooth_vertices"],
                        smooth,
                        face_colors=smooth_colors,
                        show_edges=True,
                        line_width=0.5,
                    )
                ],
                title="Smooth (close-up)",
                bounds=_around(max(rims, key=len)),
            ),
        ],
        camera=data.camera("holey_bunny"),
        ncols=5,
        panel_size=(560, 560),
    )
