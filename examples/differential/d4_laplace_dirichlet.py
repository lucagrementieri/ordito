from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="D4",
    title="Laplace equation with Dirichlet conditions",
    summary="""
    The harmonic function on the bunny that is 0 on its base and 1 at the tips of its ears.
    [`cotmatrix`][ordito.laplacian.cotmatrix] builds the cotangent Laplacian `L`, and
    [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] minimises the Dirichlet energy `x^T
    (-L) x / 2` with the pinned vertices held at their values, which solves `Lx = 0` on the free
    ones. [`marching_triangles`][ordito.intersection.marching_triangles] traces level sets of the
    solution; they crowd where the heat would flow fastest, around the neck and the ears.
    """,
    credits=(
        ("libigl 303", "https://libigl.github.io/tutorial/#laplace-equation"),
        ("PyMeshLab: filters", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    height = vertices.numpy()[:, 1]
    base = height < height.min() + 0.005
    ears = height > height.max() - 0.008

    laplacian = od.laplacian.cotmatrix(vertices, faces, dtype=wp.float64)
    q = od.energies.k_harmonic(laplacian, k=1)  # the Dirichlet energy -L
    fixed = wp.array(base | ears, dtype=wp.bool, device=device)
    fixed_values = od.typing.as_array2d(
        wp.array(ears[None].astype(np.float64), dtype=wp.float64, device=device), wp.float64
    )
    solution, free_map, n_free = od.linalg.min_quad_with_fixed(q, fixed, fixed_values)

    field = ears.astype(np.float64)  # pinned values, then the solved free ones
    free = ~(base | ears)
    field[free] = solution.numpy()[0][free_map.numpy()[free]]
    print(f"{n_free} free vertices, {base.sum()} pinned to 0, {ears.sum()} pinned to 1")

    values = wp.array(field, dtype=wp.float64, device=device)
    isolines = [
        od.intersection.marching_triangles(vertices, faces, values, level)[0]
        for level in np.linspace(0.05, 0.95, 10)
    ]
    # --8<-- [end:code]
    return {
        "field": field,
        "pinned": np.flatnonzero(base | ears),
        "isolines": [line.numpy() for lines in isolines for line in lines],
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    pinned = result["pinned"]
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY),
                    r.Points(
                        vertices[pinned],
                        scalars=result["field"][pinned],
                        cmap="viridis",
                        clim=(0.0, 1.0),
                        size=6,
                    ),
                ],
                title="Pinned vertices",
            ),
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=result["field"],
                        cmap="viridis",
                        clim=(0.0, 1.0),
                        scalar_bar="u",
                    ),
                    r.Lines(result["isolines"], color="#303030", width=1.8),
                ],
                title="Harmonic field and level sets",
            ),
        ],
        camera=data.camera("bunny"),
    )
