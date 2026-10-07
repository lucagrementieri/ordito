from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="Q7",
    title="Chamfer and Hausdorff distances",
    summary="""
    How far is a simplified mesh from the original? The bunny is decimated twice with
    [`quadric_decimate`][ordito.remesh.quadric_decimate], and
    [`chamfer_points_to_mesh`][ordito.metrics.chamfer_points_to_mesh] with
    `point_reduction=None` gives the squared distance from every original vertex to the
    simplified surface: an error map. The two summary numbers are the symmetric
    [`chamfer_mesh_to_mesh`][ordito.metrics.chamfer_mesh_to_mesh] (a mean of squared distances,
    the pytorch3d convention) and [`hausdorff_mesh_to_mesh`][ordito.metrics.hausdorff_mesh_to_mesh]
    (the worst case).
    """,
    credits=(
        (
            "PyMeshLab: Hausdorff distance",
            "https://pymeshlab.readthedocs.io/en/latest/filter_list.html",
        ),
        (
            "pytorch3d: chamfer loss",
            "https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh",
        ),
    ),
)


def run(device: str) -> dict[float, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    results = {}
    for ratio in (0.05, 0.01):
        # A scan is creased everywhere at a coarse scale: raise feature_angle to reach the target.
        coarse_vertices, coarse_faces = od.remesh.quadric_decimate(
            vertices, faces, target_ratio=ratio, feature_angle=90.0
        )
        error = od.metrics.chamfer_points_to_mesh(
            vertices, coarse_vertices, coarse_faces, point_reduction=None, single_directional=True
        )
        chamfer = od.metrics.chamfer_mesh_to_mesh(vertices, faces, coarse_vertices, coarse_faces)
        hausdorff = od.metrics.hausdorff_mesh_to_mesh(
            vertices, faces, coarse_vertices, coarse_faces
        )
        print(
            f"{coarse_faces.shape[0] // 3} faces: chamfer {chamfer:.2e}, hausdorff {hausdorff:.4f}"
        )
        results[ratio] = (coarse_vertices, coarse_faces, np.sqrt(error.numpy()))
    # --8<-- [end:code]
    return {
        ratio: (v.numpy(), f.numpy().reshape(-1, 3), error)
        for ratio, (v, f, error) in results.items()
    }


def figure(result: dict[float, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    clim = (0.0, float(np.percentile(result[0.01][2], 99)))
    panels: list[r.Panel] = []
    for coarse_v, coarse_f, error in result.values():
        panels.append(
            r.Panel(
                [r.Mesh(coarse_v, coarse_f, color=r.LIGHT_GREEN, show_edges=True, smooth=False)],
                title=f"{coarse_f.shape[0]} faces",
            )
        )
        panels.append(
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=error,
                        cmap="magma_r",
                        clim=clim,
                        scalar_bar="distance to the decimated surface",
                    )
                ],
                title=f"Error of the {coarse_f.shape[0]}-face mesh",
            )
        )
    return r.Figure(panels, camera=data.camera("bunny"), ncols=4, panel_size=(560, 520))
