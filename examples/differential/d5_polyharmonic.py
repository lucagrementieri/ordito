from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="D5",
    title="Polyharmonic surfaces (k = 1, 2, 3)",
    summary="""
    A flat square grid from [`grid`][ordito.creation.grid] is pinned at height 0 outside a circle
    and at height 1 on a small disk in its middle; the heights in between minimise a k-harmonic
    energy. [`k_harmonic`][ordito.energies.k_harmonic] composes the operator
    `(-L) (M^-1 (-L))^(k-1)` from the cotangent Laplacian of
    [`cotmatrix`][ordito.laplacian.cotmatrix] and the lumped mass of
    [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries], and
    [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] solves for the free heights. Each
    step up in k makes the surface smoother where it meets the pinned regions: k = 1 leaves a kink
    at both rims, k = 2 meets them with a continuous slope, k = 3 also with a continuous
    curvature, which widens the shoulders.
    """,
    credits=(
        ("libigl 401", "https://libigl.github.io/tutorial/#biharmonic-deformation"),
        ("libigl 402", "https://libigl.github.io/tutorial/#polyharmonic-deformation"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(101, 101), extents=(2.0, 2.0), device=device)
    radius = np.linalg.norm(vertices.numpy()[:, :2], axis=1)
    outside, top = radius > 0.9, radius < 0.15
    pinned = outside | top

    laplacian = od.laplacian.cotmatrix(vertices, faces, dtype=wp.float64)
    mass = od.laplacian.mass_matrix_entries(vertices, faces, dtype=wp.float64)
    fixed = wp.array(pinned, dtype=wp.bool, device=device)
    heights = od.typing.as_array2d(
        wp.array(top[None].astype(np.float64), dtype=wp.float64, device=device), wp.float64
    )

    surfaces = {}
    for k in (1, 2, 3):
        q = od.energies.k_harmonic(laplacian, mass, k=k)
        solution, free_map, _ = od.linalg.min_quad_with_fixed(q, fixed, heights)
        z = top.astype(np.float64)
        z[~pinned] = solution.numpy()[0][free_map.numpy()[~pinned]]
        surfaces[k] = z
        print(f"k = {k}: height at radius 0.5 is {z[np.abs(radius - 0.5) < 0.02].mean():.3f}")
    # --8<-- [end:code]
    return {
        "vertices": vertices.numpy(),
        "faces": faces.numpy().reshape(-1, 3),
        "surfaces": surfaces,
        "pinned": pinned,
    }


def figure(result: dict[str, Any]) -> r.Figure:
    base, faces = result["vertices"], result["faces"]
    panels = []
    for k, z in result["surfaces"].items():
        vertices = base.copy()
        vertices[:, 2] = 0.8 * z
        panels.append(
            r.Panel(
                [r.Mesh(vertices, faces, scalars=z, cmap="viridis", clim=(0.0, 1.0))],
                title=f"k = {k}",
            )
        )
    return r.Figure(
        panels,
        camera=r.Camera(direction=(0.6, -1.0, 0.55), up=(0.0, 0.0, 1.0), zoom=1.4),
        panel_size=(620, 480),
    )
