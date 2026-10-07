from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="RR3",
    title="Local triangulation and uniform resampling",
    summary="""
    [`triangulate_point_cloud`][ordito.reconstruction.triangulate_point_cloud] lets every point
    build a small fan over its nearest neighbours in its tangent plane and keeps the triangles
    that neighbouring fans agree on: a mesh whose vertices are exactly the input points, with
    uneven triangles where the sampling is uneven.
    [`resample_uniform`][ordito.reconstruction.resample_uniform] then rebuilds any mesh from its
    signed distance field on a uniform grid, so the result has evenly sized triangles and no
    holes, at the cost of detail thinner than a voxel.
    """,
    credits=(
        ("MeshLib: points to mesh", "https://meshlib.io/documentation/ExamplePointsToMesh.html"),
        (
            "PyVista: reconstruct surface",
            "https://docs.pyvista.org/examples/01-filter/surface_reconstruction",
        ),
        (
            "PyMeshLab: uniform mesh resampling",
            "https://pymeshlab.readthedocs.io/en/latest/filter_list.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    points = data.load_points("bunny_cloud", device)
    normals = wp.array(data.cloud_normals("bunny_cloud"), dtype=wp.vec3, device=device)
    local_vertices, local_faces = od.reconstruction.triangulate_point_cloud(points, normals)
    holes = od.boundary.boundary_loops(local_vertices, local_faces)
    print(f"local triangulation: {local_faces.shape[0] // 3} faces, {len(holes)} holes")

    vertices, faces = od.reconstruction.resample_uniform(
        local_vertices, local_faces, voxel_size=0.0015
    )
    watertight = od.validation.is_watertight(vertices, faces)
    print(f"resampled: {faces.shape[0] // 3} faces, watertight {watertight}")
    # --8<-- [end:code]
    return {
        "local_vertices": local_vertices.numpy(),
        "local_faces": local_faces.numpy().reshape(-1, 3),
        "holes": [hole.numpy() for hole in holes],
        "vertices": vertices.numpy(),
        "faces": faces.numpy().reshape(-1, 3),
    }


def figure(result: dict[str, Any]) -> r.Figure:
    local_v, local_f = result["local_vertices"], result["local_faces"]
    rims = [local_v[hole] for hole in result["holes"]]
    center = np.array([-0.06, 0.13, 0.03])
    close = np.stack([center - 0.012, center + 0.012])
    edges: r.MeshStyle = {"show_edges": True, "line_width": 0.5, "smooth": False}
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(local_v, local_f), r.Lines(rims, closed=True, width=2.5)],
                title="Local triangulation (holes red)",
            ),
            r.Panel([r.Mesh(local_v, local_f, **edges)], title="Close-up", bounds=close),
            r.Panel([r.Mesh(result["vertices"], result["faces"])], title="Uniformly resampled"),
            r.Panel(
                [r.Mesh(result["vertices"], result["faces"], **edges)],
                title="Close-up",
                bounds=close,
            ),
        ],
        camera=data.camera("bunny_cloud"),
        ncols=4,
        panel_size=(560, 520),
    )
