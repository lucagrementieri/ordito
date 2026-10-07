from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="S1",
    title="Denoising a scan: four smoothing filters",
    summary="""
    The bunny with Gaussian noise along its normals, cleaned four ways:

    - [`filter_laplacian`][ordito.smoothing.filter_laplacian] moves every vertex towards the
      mean of its neighbours (here rescaled after each pass to keep the volume); a few passes
      remove noise, more start to erase detail;
    - [`filter_taubin`][ordito.smoothing.filter_taubin] alternates a shrinking and an inflating
      step, which removes noise with far less shrinkage;
    - [`filter_implicit_fairing`][ordito.smoothing.filter_implicit_fairing] takes one implicit
      step of cotangent curvature flow;
    - [`filter_two_step`][ordito.smoothing.filter_two_step] first smooths the face normals with
      [`filter_normals`][ordito.smoothing.filter_normals] and then moves the vertices to fit
      them, which keeps sharp features.

    The printed number is the mean distance of each result's vertices from the clean scan.
    """,
    credits=(
        (
            "Open3D: mesh filtering",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
        ("PyVista: smoothing", "https://docs.pyvista.org/examples/01-filter/surface_smoothing"),
        ("MeshLib: denoise", "https://meshlib.io/documentation/ExampleNoiseDenoise.html"),
        ("PyMeshLab: smoothing", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
        ("pytorch3d: taubin_smoothing", "https://pytorch3d.org/tutorials"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("noisy_bunny", device)
    smoothed = {
        "Laplacian": od.smoothing.filter_laplacian(vertices, faces, iterations=5),
        "Taubin": od.smoothing.filter_taubin(vertices, faces, iterations=20),
        "Implicit fairing": od.smoothing.filter_implicit_fairing(
            vertices, faces, lamb=1e-6, iterations=1
        ),
        "Two-step (normals)": od.smoothing.filter_two_step(vertices, faces),
    }

    clean = data.load("bunny", device)[0].numpy()
    print(f"{'noisy':>18}: {np.linalg.norm(vertices.numpy() - clean, axis=1).mean():.2e}")
    for name, result in smoothed.items():
        print(f"{name:>18}: {np.linalg.norm(result.numpy() - clean, axis=1).mean():.2e}")
    # --8<-- [end:code]
    return {"Noisy": vertices.numpy(), **{k: v.numpy() for k, v in smoothed.items()}}


def figure(result: dict[str, Any]) -> r.Figure:
    _, faces = data.arrays("noisy_bunny")
    vertices = result["Noisy"]
    head = vertices[11842] + np.array([0.035, 0.015, 0.0])
    close = np.stack([head - 0.035, head + 0.035])
    panels = []
    for name, positions in result.items():
        color = r.LIGHT_GREEN if name == "Noisy" else r.GREEN
        panels.append(r.Panel([r.Mesh(positions, faces, color=color)], title=name))
    for name, positions in result.items():
        color = r.LIGHT_GREEN if name == "Noisy" else r.GREEN
        panels.append(r.Panel([r.Mesh(positions, faces, color=color)], title="", bounds=close))
    return r.Figure(panels, ncols=5, camera=data.camera("noisy_bunny"), panel_size=(480, 440))
