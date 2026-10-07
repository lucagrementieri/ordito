from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import to_rgb

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="P1",
    title="Flattening a disk: Tutte, harmonic, LSCM and ARAP",
    summary="""
    Four ways to map a disk-shaped patch of the bunny into the plane, each drawn as a checkerboard
    texture and as its UV layout:

    - [`tutte`][ordito.parametrization.tutte] and [`harmonic`][ordito.parametrization.harmonic]
      pin the boundary ([`longest_boundary_loop`][ordito.boundary.longest_boundary_loop]) to a
      circle with [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle] and
      solve for the interior, with uniform and cotangent weights;
    - [`lscm`][ordito.parametrization.lscm] pins only two vertices and finds the most conformal
      (angle-preserving) map, with a free boundary;
    - [`arap`][ordito.parametrization.arap] starts from the LSCM map and makes every triangle as
      close to a rigid copy of itself as it can, so lengths are preserved too.

    [`face_flipped_mask`][ordito.parametrization.face_flipped_mask] checks that no triangle is
    turned over. The UV layouts are coloured by how much each triangle's area is scaled (red:
    enlarged, blue: shrunk, on a log scale from 1/4 to 4): forcing the boundary onto a circle
    stretches part of the patch, LSCM keeps angles but not areas, and ARAP keeps both close to the
    surface's.
    """,
    credits=(
        ("libigl 501: harmonic", "https://libigl.github.io/tutorial/#harmonic-parametrization"),
        ("libigl 502: LSCM", "https://libigl.github.io/tutorial/#least-squares-conformal-maps"),
        ("libigl 503: ARAP", "https://libigl.github.io/tutorial/#as-rigid-as-possible"),
        (
            "PyMeshLab: parametrization",
            "https://pymeshlab.readthedocs.io/en/latest/filter_list.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("face_patch", device)
    boundary = od.boundary.longest_boundary_loop(vertices, faces)
    circle = od.parametrization.map_vertices_to_circle(vertices, boundary)

    uv = {
        "Tutte": od.parametrization.tutte(vertices, faces, boundary, circle),
        "Harmonic": od.parametrization.harmonic(vertices, faces, boundary, circle),
    }
    # LSCM: pin two opposite boundary vertices at their true distance apart.
    b = boundary.numpy()
    pins = wp.array([b[0], b[b.size // 2]], dtype=wp.int32, device=device)
    span = np.linalg.norm(vertices.numpy()[b[0]] - vertices.numpy()[b[b.size // 2]])
    pinned_uv = wp.array([(0.0, 0.0), (span, 0.0)], dtype=wp.vec2, device=device)
    uv["LSCM"] = od.parametrization.lscm(vertices, faces, pins, pinned_uv)
    uv["ARAP"] = od.parametrization.arap(
        vertices,
        faces,
        od.typing.as_dense(pins[:1]),
        od.typing.as_dense(pinned_uv[:1]),
        uv["LSCM"],
        max_iterations=50,
    )

    for name, coordinates in uv.items():
        flipped = od.parametrization.face_flipped_mask(coordinates, faces).numpy().sum()
        print(f"{name:>8}: {flipped} flipped triangles")
    # --8<-- [end:code]
    return {name: coordinates.numpy() for name, coordinates in uv.items()}


LIGHT, DARK = "#f39c12", "#404040"


def _areas(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Per-face log2 area ratio (UV over surface), the UV scaled to the surface's total area."""
    tri = vertices[faces]
    area3 = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    t = uv[faces]
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    area2 = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    area2 *= area3.sum() / area2.sum()
    return np.log2(np.maximum(area2, 1e-30) / np.maximum(area3, 1e-30))


def _checker(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Per-face checkerboard parity, ten cells across the surface's own size."""
    tri = vertices[faces]
    area3 = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    t = uv[faces]
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    area2 = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    center = (uv * np.sqrt(area3.sum() / area2.sum()))[faces].mean(axis=1)
    cell = np.sqrt(area3.sum()) / 10.0
    return ((np.floor(center[:, 0] / cell) + np.floor(center[:, 1] / cell)) % 2).astype(int)


def _layout(uv: np.ndarray, faces: np.ndarray, distortion: np.ndarray):
    def draw(ax: Any) -> None:
        ax.tripcolor(
            uv[:, 0],
            uv[:, 1],
            faces,
            facecolors=np.clip(distortion, -2.0, 2.0),
            cmap="RdBu_r",
            vmin=-2.0,
            vmax=2.0,
            edgecolors="none",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    return draw


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("face_patch")
    rgb = (np.array([to_rgb(DARK), to_rgb(LIGHT)]) * 255).astype(np.uint8)
    renders, layouts = [], []
    for name, uv in result.items():
        parity = _checker(vertices, faces, uv)
        renders.append(
            r.Panel([r.Mesh(vertices, faces, face_colors=rgb[parity], smooth=False)], title=name)
        )
        distortion = _areas(vertices, faces, uv)
        layouts.append(r.Plot(_layout(uv, faces, distortion), title=f"{name}: area scale in UV"))
    return r.Figure(
        [*renders, *layouts], ncols=4, camera=data.camera("face_patch"), panel_size=(520, 480)
    )
