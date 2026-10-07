from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="R4",
    title="Isotropic remeshing",
    summary="""
    A scan's triangles come in every shape. [`isotropic_remesh`][ordito.remesh.isotropic_remesh]
    drives every edge toward one target length by repeated splits, collapses, valence-improving
    flips and tangential smoothing, reprojecting onto the input after each pass, so the result is
    the same surface sampled by near-equilateral triangles. The colours are
    [`face_quality`][ordito.triangles.face_quality]'s radius ratio (1 for an equilateral
    triangle, 0 for a degenerate one). Edges sharper than `feature_angle` are kept as creases;
    on a noisy scan the default treats too many of them as features, so it is raised here.
    """,
    credits=(("PyMeshLab", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    # A scan has no real creases; the default 30-degree feature angle would freeze its noise.
    new_vertices, new_faces = od.remesh.isotropic_remesh(vertices, faces, feature_angle=90.0)

    before = od.triangles.face_quality(vertices, faces, metric="radius_ratio").numpy()
    after = od.triangles.face_quality(new_vertices, new_faces, metric="radius_ratio").numpy()
    print(f"faces: {before.size} -> {after.size}")
    print(f"mean radius ratio: {before.mean():.3f} -> {after.mean():.3f}")
    print(f"faces below 0.5: {(before < 0.5).sum()} -> {(after < 0.5).sum()}")
    # --8<-- [end:code]
    return {
        "vertices": new_vertices.numpy(),
        "faces": new_faces.numpy().reshape(-1, 3),
        "before": before,
        "after": after,
    }


FACE = np.array([[-0.095, 0.1, 0.0], [-0.045, 0.15, 0.05]])
"""Bounds of the bunny's face, for the close-ups."""


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")

    def quality(v: np.ndarray, f: np.ndarray, q: np.ndarray, **kw: Any) -> r.Mesh:
        return r.Mesh(v, f, scalars=q, clim=(0.4, 1.0), smooth=False, line_width=0.5, **kw)

    def histogram(ax: Any) -> None:
        bins = np.linspace(0.0, 1.0, 41)
        ax.hist(result["before"], bins=bins, color=r.GREY, label="input", density=True)
        ax.hist(
            result["after"], bins=bins, color=r.GREEN, alpha=0.8, label="remeshed", density=True
        )
        ax.set_xlabel("radius ratio")
        ax.set_yticks([])
        ax.legend(frameon=False, labelcolor=r.TEXT)

    new_v, new_f = result["vertices"], result["faces"]
    return r.Figure(
        [
            r.Panel(
                [quality(new_v, new_f, result["after"], scalar_bar="radius ratio")],
                title="Remeshed",
            ),
            r.Panel(
                [quality(vertices, faces, result["before"], show_edges=True)],
                title="Input (close-up)",
                bounds=FACE,
            ),
            r.Panel(
                [quality(new_v, new_f, result["after"], show_edges=True)],
                title="Remeshed (close-up)",
                bounds=FACE,
            ),
            r.Plot(histogram, title="Triangle quality", aspect=""),
        ],
        camera=data.camera("bunny"),
        ncols=4,
        panel_size=(620, 600),
    )
