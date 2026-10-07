from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="X6",
    title="Polyline processing",
    summary="""
    The longest contour of the dragon's section at y = 0.128 (example X1)
    ([`marching_triangles`][ordito.intersection.marching_triangles] over the height field) is a
    closed polyline of over a thousand points.
    [`polyline_simplify`][ordito.polyline.polyline_simplify] keeps the Ramer-Douglas-Peucker
    subset within a tolerance,
    [`polyline_resample`][ordito.polyline.polyline_resample] redistributes a fixed number of
    points evenly along the arc length, and
    [`polyline_smooth_upsample`][ordito.polyline.polyline_smooth_upsample] refines the coarse
    simplified loop back along circular arcs fitted to its tangents.
    """,
    credits=(
        ("PyVista: decimate a polyline", "https://docs.pyvista.org/examples/01-filter/decimate"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("dragon", device)
    height = wp.array(vertices.numpy()[:, 1] - 0.128, dtype=wp.float32, device=device)
    curves, closed = od.intersection.marching_triangles(vertices, faces, height)
    loop = max(
        (c for c, is_closed in zip(curves, closed, strict=True) if is_closed),
        key=lambda c: c.shape[0],
    )

    simplified, _ = od.polyline.polyline_simplify(loop, 0.002, closed=True)
    resampled = od.polyline.polyline_resample(loop, 60, closed=True)
    smooth = od.polyline.polyline_smooth_upsample(simplified, 0.002, closed=True)
    print(f"section loop: {loop.shape[0]} points")
    print(f"simplified: {simplified.shape[0]}, resampled: {resampled.shape[0]}")
    print(f"smoothly upsampled from the simplified loop: {smooth.shape[0]}")
    # --8<-- [end:code]
    return {
        "loop": loop.numpy(),
        "simplified": simplified.numpy(),
        "resampled": resampled.numpy(),
        "smooth": smooth.numpy(),
    }


def _xz(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    closed = np.vstack([points, points[:1]])
    return closed[:, 0], closed[:, 2]


def figure(result: dict[str, Any]) -> r.Figure:
    loop = result["loop"]

    def panel(key: str | None, color: str, markers: bool) -> Any:
        def draw(ax: Any) -> None:
            x, z = _xz(loop)
            ax.plot(x, z, color=r.GREY if key else r.GREEN, lw=2.5 if key else 1.5)
            if key is not None:
                x, z = _xz(result[key])
                ax.plot(x, z, color=color, lw=1.4, marker="o" if markers else None, ms=3.5)
            ax.set_xticks([])
            ax.set_yticks([])

        return draw

    return r.Figure(
        [
            r.Plot(panel(None, r.GREEN, False), title=f"Section loop ({loop.shape[0]} points)"),
            r.Plot(
                panel("simplified", r.RED, True),
                title=f"Simplified ({result['simplified'].shape[0]} points)",
            ),
            r.Plot(
                panel("resampled", r.BLUE, True),
                title=f"Resampled ({result['resampled'].shape[0]} points)",
            ),
            r.Plot(panel("smooth", r.ORANGE, False), title="Simplified, then smoothly upsampled"),
        ],
        ncols=2,
        panel_size=(760, 330),
    )
