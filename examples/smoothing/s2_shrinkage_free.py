from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.collections import LineCollection

from examples import _render as r
from examples import data
from examples._meta import Meta

SLICE_Z = -0.005

META = Meta(
    id="S2",
    title="Smoothing without shrinking",
    summary="""
    Many passes of plain Laplacian smoothing pull every vertex towards its neighbours' centroid,
    and the whole surface shrinks: thin parts such as the ears go first. Three filters smooth
    as hard without losing volume:

    - [`filter_humphrey`][ordito.smoothing.filter_humphrey] follows each Laplacian step by
      pushing the vertices part of the way back towards their original positions (HC
      filtering);
    - [`relax_keep_volume`][ordito.smoothing.relax_keep_volume] subtracts from each vertex's
      move the average move of its neighbourhood, so a local drift inwards cancels while the
      noise is still removed;
    - [`filter_mut_dif_laplacian`][ordito.smoothing.filter_mut_dif_laplacian] adapts the
      diffusion speed per vertex and inflates along the normals to restore the volume.

    The lower row shows a slice through the head and body of each result (colour) over the noisy
    input (grey). Volumes are from [`volume`][ordito.measures.volume].
    """,
    credits=(
        (
            "Open3D: Taubin vs Laplacian",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
        ("PyMeshLab: smoothing", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("noisy_bunny", device)
    passes = 200
    smoothed = {
        "Laplacian": od.smoothing.filter_laplacian(
            vertices, faces, iterations=passes, volume_constraint=False
        ),
        "Humphrey": od.smoothing.filter_humphrey(vertices, faces, iterations=passes),
        "Relax, keep volume": od.smoothing.relax_keep_volume(vertices, faces, iterations=passes),
        "Mut. dif. Laplacian": od.smoothing.filter_mut_dif_laplacian(
            vertices, faces, iterations=passes
        ),
    }
    before = od.measures.volume(vertices, faces)
    for name, result in smoothed.items():
        print(f"{name:>19}: volume x {od.measures.volume(result, faces) / before:.3f}")
    # --8<-- [end:code]
    return {"input": vertices.numpy(), **{k: v.numpy() for k, v in smoothed.items()}}


def _slice(vertices: np.ndarray, faces: np.ndarray, z: float) -> np.ndarray:
    """Segments ``(m, 2, 2)`` (x, y) where the plane ``Z = z`` crosses the triangles."""
    tri = vertices[faces]
    d = tri[:, :, 2] - z
    segments = []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        # Triangles where vertex ``a`` is alone on its side: the two crossing edges are a-b, a-c.
        alone = (np.sign(d[:, a]) != np.sign(d[:, b])) & (np.sign(d[:, a]) != np.sign(d[:, c]))
        t, da = tri[alone], d[alone]
        p = t[:, a] + (t[:, b] - t[:, a]) * (da[:, a] / (da[:, a] - da[:, b]))[:, None]
        q = t[:, a] + (t[:, c] - t[:, a]) * (da[:, a] / (da[:, a] - da[:, c]))[:, None]
        segments.append(np.stack([p[:, :2], q[:, :2]], axis=1))
    return np.concatenate(segments)


def figure(result: dict[str, Any]) -> r.Figure:
    _, faces = data.arrays("noisy_bunny")
    reference = _slice(result["input"], faces, SLICE_Z)
    names = [k for k in result if k != "input"]
    renders = [
        r.Panel(
            [
                r.Mesh(result["input"], faces, color=r.GREY, opacity=0.25, smooth=False),
                r.Mesh(result[name], faces, color=r.GREEN),
            ],
            title=name,
        )
        for name in names
    ]

    def outline(name: str):
        segments = _slice(result[name], faces, SLICE_Z)

        def draw(ax: Any) -> None:
            ax.add_collection(LineCollection(list(reference), colors=r.GREY, linewidths=2.5))
            color = r.RED if name == "Laplacian" else r.GREEN
            ax.add_collection(LineCollection(list(segments), colors=color, linewidths=1.2))
            ax.autoscale()
            ax.set_xticks([])
            ax.set_yticks([])

        return draw

    plots = [r.Plot(outline(name), title=f"{name}: slice") for name in names]
    return r.Figure([*renders, *plots], ncols=4, camera=data.camera("bunny"), panel_size=(500, 460))
