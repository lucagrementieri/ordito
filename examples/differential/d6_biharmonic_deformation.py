from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="D6",
    title="Handle-based biharmonic deformation",
    summary="""
    The armadillo's feet are held in place and the top of the scan (its head, ears and raised
    claws) is moved as one handle; every other vertex moves by the
    displacement that minimises the biharmonic energy, the operator
    [`k_harmonic`][ordito.energies.k_harmonic] builds at `k = 2` from
    [`cotmatrix`][ordito.laplacian.cotmatrix] and
    [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries].
    [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] solves the three coordinates of the
    displacement as three right-hand sides of one system. The displacement is smooth, so the body
    bends as a whole and its surface detail rides along unchanged; the multigrid preconditioner
    keeps the fourth-order solve over the full 173 k-vertex scan short.
    """,
    credits=(
        ("libigl 401", "https://libigl.github.io/tutorial/#biharmonic-deformation"),
        (
            "MeshLib: Laplacian deformation",
            "https://meshlib.io/documentation/ExampleLaplacian.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("armadillo", device)
    rest = vertices.numpy()
    height = rest[:, 1]
    span = height.max() - height.min()
    feet = height < height.min() + 0.1 * span
    head = height > height.max() - 0.12 * span  # head, ears and claw tips

    displacement = np.zeros((3, len(rest)))  # one row per coordinate
    displacement[:, head] = np.array([[50.0], [-15.0], [-20.0]])

    laplacian = od.laplacian.cotmatrix(vertices, faces, dtype=wp.float64)
    mass = od.laplacian.mass_matrix_entries(vertices, faces, dtype=wp.float64)
    q = od.energies.k_harmonic(laplacian, mass, k=2)
    pinned = feet | head
    solution, free_map, n_free = od.linalg.min_quad_with_fixed(
        q,
        wp.array(pinned, dtype=wp.bool, device=device),
        wp.array(displacement, dtype=wp.float64, device=device),
        preconditioner="multigrid",
    )
    displacement[:, ~pinned] = solution.numpy()[:, free_map.numpy()[~pinned]]
    deformed = rest + displacement.T
    print(f"{n_free} free vertices, {feet.sum()} fixed, {head.sum()} moved")
    # --8<-- [end:code]
    return {"deformed": deformed, "feet": feet, "head": head}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("armadillo")
    colors = np.tile(np.array([0xC8, 0xC8, 0xC8], dtype=np.uint8), (vertices.shape[0], 1))
    colors[result["feet"]] = (0x2C, 0x7B, 0xE8)
    colors[result["head"]] = (0xF3, 0x9C, 0x12)
    colors = colors / 255.0
    both = np.concatenate([vertices, result["deformed"]])
    bounds = np.stack([both.min(axis=0), both.max(axis=0)])
    camera = r.Camera(direction=(0.25, 0.1, -1.0), up=(0.0, 1.0, 0.0), zoom=1.45)
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(vertices, faces, scalars=colors, rgb=True)],
                title="Rest pose: fixed (blue), handle (orange)",
                bounds=bounds,
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY, opacity=0.25),
                    r.Mesh(result["deformed"], faces, scalars=colors, rgb=True),
                ],
                title="Deformed (rest pose ghosted)",
                bounds=bounds,
            ),
        ],
        camera=camera,
    )
