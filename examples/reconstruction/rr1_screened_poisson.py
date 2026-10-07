from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="RR1",
    title="Screened Poisson reconstruction",
    summary="""
    [`screened_poisson`][ordito.reconstruction.screened_poisson] turns an oriented point cloud
    into a closed surface: it solves for an indicator function whose gradient matches the
    normals, screened towards zero at the points, and extracts its level set. `depth` sets the
    grid resolution (`2 ** depth` cells across): depth 6 gives a smooth blob, depth 8 the
    bunny's fur. `method="adaptive"` solves on an octree refined only near the points instead
    of a dense grid.
    """,
    credits=(
        (
            "Open3D: Poisson surface reconstruction",
            "https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html",
        ),
        (
            "PyMeshLab: screened Poisson",
            "https://pymeshlab.readthedocs.io/en/latest/filter_list.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    from typing import Literal

    import warp as wp

    import ordito as od
    from examples import data

    points = data.load_points("bunny_cloud", device)
    normals = wp.array(data.cloud_normals("bunny_cloud"), dtype=wp.vec3, device=device)

    surfaces = {}
    runs: list[tuple[str, int, Literal["dense", "adaptive"]]] = [
        ("dense6", 6, "dense"),
        ("dense8", 8, "dense"),
        ("adaptive8", 8, "adaptive"),
    ]
    for name, depth, method in runs:
        vertices, faces = od.reconstruction.screened_poisson(
            points, normals, depth=depth, method=method
        )
        print(f"{method} depth {depth}: {faces.shape[0] // 3} faces")
        surfaces[name] = (vertices.numpy(), faces.numpy().reshape(-1, 3))
    # --8<-- [end:code]
    return {"points": points.numpy(), **surfaces}


def figure(result: dict[str, Any]) -> r.Figure:
    return r.Figure(
        [
            r.Panel([r.Points(result["points"], color=r.GREY, size=3)], title="Oriented cloud"),
            r.Panel([r.Mesh(*result["dense6"])], title="Depth 6"),
            r.Panel([r.Mesh(*result["dense8"])], title="Depth 8"),
            r.Panel([r.Mesh(*result["adaptive8"])], title="Depth 8, adaptive octree"),
        ],
        camera=data.camera("bunny_cloud"),
        ncols=4,
        panel_size=(560, 520),
    )
