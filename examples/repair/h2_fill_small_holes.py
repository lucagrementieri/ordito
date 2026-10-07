from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="H2",
    title="Filling only the small holes",
    summary="""
    A scan's large openings are often intended while its small holes are dropouts.
    [`fill_small`][ordito.holes.fill_small] closes only the rims up to a size (here 45 boundary
    edges) with minimum-weight patches and leaves the rest open.
    [`fillable_loop_mask`][ordito.holes.fillable_loop_mask] answers a different question: which
    rims a fill over the rim's own vertices can close without creating an invalid mesh. The holes
    punched into the bunny have ragged rims with *chords* (an existing edge between two rim
    vertices), which a fill could duplicate, so they are flagged; the scan's own holes in the
    base are not.
    """,
    credits=(
        ("pymeshfix", "https://pymeshfix.pyvista.org/examples/index.html"),
        ("PyMeshLab: close holes", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("holey_bunny", device)
    loops = od.boundary.boundary_loops(vertices, faces)
    fillable = od.holes.fillable_loop_mask(vertices, faces, loops).numpy()
    print("rim sizes:", [loop.shape[0] for loop in loops])
    print("chord-free and simple:", fillable.tolist())

    filled = od.holes.fill_small(vertices, faces, max_edges=45)
    left = od.boundary.boundary_loops(vertices, filled)
    print(f"after fill_small: {len(left)} holes left, of {[loop.shape[0] for loop in left]} edges")
    # --8<-- [end:code]
    return {
        "loops": [loop.numpy() for loop in loops],
        "fillable": fillable,
        "filled": filled.numpy().reshape(-1, 3),
        "left": [loop.numpy() for loop in left],
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("holey_bunny")
    loops = result["loops"]
    small = [vertices[loop] for loop in loops if loop.size <= 45]
    large = [vertices[loop] for loop in loops if loop.size > 45]
    filled = result["filled"]
    colors = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (filled.shape[0], 1))
    colors[faces.shape[0] :] = (0xF3, 0x9C, 0x12)
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN),
                    r.Lines(small, color=r.ORANGE, closed=True),
                    r.Lines(large, color=r.RED, closed=True),
                ],
                title="Small (orange) and large (red) rims",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, filled, face_colors=colors),
                    r.Lines([vertices[loop] for loop in result["left"]], closed=True),
                ],
                title="fill_small(max_edges=45)",
            ),
        ],
        camera=data.camera("holey_bunny"),
        panel_size=(680, 620),
    )
