from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="D2",
    title="Principal curvatures and directions",
    summary="""
    [`principal_curvature`][ordito.curvature.principal_curvature] fits a quadric to the
    neighbourhood of every vertex (a ball of a few mean edge lengths) and returns the two principal
    curvatures with their directions: the directions in which the surface bends most and least.
    The close-up draws both direction fields as short line segments, maximal curvature in red and
    minimal in blue; they follow the ridges and valleys of the bunny's face and ears.
    """,
    credits=(
        ("libigl 203", "https://libigl.github.io/tutorial/#curvature-directions"),
        ("PyMeshLab: filters", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    direction_max, direction_min, k_max, k_min = od.curvature.principal_curvature(
        vertices, faces, radius=5
    )
    k_max, k_min = k_max.numpy(), k_min.numpy()
    print(f"k_max >= k_min at every vertex: {bool(np.all(k_max >= k_min))}")
    print(f"median curvatures: k_max {np.median(k_max):.1f}, k_min {np.median(k_min):.1f}")
    # --8<-- [end:code]
    return {
        "k_max": k_max,
        "k_min": k_min,
        "d_max": direction_max.numpy(),
        "d_min": direction_min.numpy(),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    k_max, k_min = result["k_max"], result["k_min"]
    clim = r.symmetric_clim(np.concatenate([k_max, k_min]), 95.0)
    # Close-up of the head: a random subset of its vertices carries the direction segments.
    center = np.array([-0.06, 0.135, 0.03])
    half = 0.035
    near = np.flatnonzero(np.all(np.abs(vertices - center) < 1.6 * half, axis=1))
    picked = np.random.default_rng(0).choice(near, size=min(1600, near.size), replace=False)
    length = 0.0011
    lift = 0.0003 * data.vertex_normals(vertices, faces)[picked]

    def segments(direction: np.ndarray) -> np.ndarray:
        start = vertices[picked] + lift - length * direction[picked]
        return np.stack([start, start + 2 * length * direction[picked]], axis=1)

    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(
                        vertices, faces, scalars=k_max, cmap="RdBu_r", clim=clim, scalar_bar="k_max"
                    )
                ],
                title="Maximal curvature",
            ),
            r.Panel(
                [
                    r.Mesh(
                        vertices, faces, scalars=k_min, cmap="RdBu_r", clim=clim, scalar_bar="k_min"
                    )
                ],
                title="Minimal curvature",
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN),
                    r.Segments(segments(result["d_max"]), color=r.RED, width=1.6),
                    r.Segments(segments(result["d_min"]), color=r.BLUE, width=1.6),
                ],
                title="Principal directions (head)",
                bounds=np.stack([center - half, center + half]),
            ),
        ],
        camera=data.camera("bunny"),
    )
