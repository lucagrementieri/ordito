from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="H8",
    title="Fixing self-intersections",
    summary="""
    A spindle torus, whose tube radius exceeds its ring radius, passes through itself around its
    axis. [`face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask] finds
    the triangles that cross a non-adjacent one (red).
    [`fix_self_intersections`][ordito.repair.fix_self_intersections] removes them in one of two
    ways: `"local"` cuts out the intersecting faces plus a ring around them and refills the
    holes, keeping the rest of the mesh as it was; `"voxel"` rebuilds the whole surface as the
    zero level set of its signed distance field. The cross-sections through the axis show what
    changed: the lens-shaped core where the tube overlaps itself has winding number 0, so it is
    a cavity, and both repairs keep it as one, bounded by a surface that no longer crosses the
    outer shell (the local repair leaves it as a separate inner shell, the voxel rebuild lets
    the two touch). The 3-D view is cut in half to show the core.
    """,
    credits=(
        (
            "MeshLib: self-intersections",
            "https://meshlib.io/documentation/ExampleDetectSelfIntersections.html",
        ),
        ("pymeshfix", "https://pymeshfix.pyvista.org/examples/index.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("self_intersecting_torus", device)
    crossing = od.validation.face_self_intersecting_mask(vertices, faces).numpy()
    print(f"self-intersecting faces: {crossing.sum()} of {crossing.size}")

    results = {}
    for method in ("local", "voxel"):
        results[method] = od.repair.fix_self_intersections(vertices, faces, method=method)
        v, f = results[method]
        left = od.validation.face_self_intersecting_mask(v, f).numpy().sum()
        print(f"{method}: {f.shape[0] // 3} faces, {left} still intersecting")
    # --8<-- [end:code]
    return {
        "crossing": crossing,
        "results": {m: (v.numpy(), f.numpy().reshape(-1, 3)) for m, (v, f) in results.items()},
    }


def _section(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Segments ``(m, 2, 2)`` of the cross-section with the plane ``y = 0``, in ``(x, z)``."""
    import trimesh as tm

    lines = tm.intersections.mesh_plane(
        tm.Trimesh(vertices, faces, process=False), (0, 1, 0), (0, 0, 0)
    )
    return np.asarray(lines)[:, :, [0, 2]]


def figure(result: dict[str, Any]) -> r.Figure:
    from matplotlib.collections import LineCollection

    vertices, faces = data.arrays("self_intersecting_torus")
    meshes = {"input": (vertices, faces), **result["results"]}

    def section(key: str, color: str) -> r.Plot:
        segments = _section(*meshes[key])

        def draw(ax: Any) -> None:
            ax.add_collection(LineCollection(list(segments), colors=color, linewidths=2.0))
            ax.set_xlim(-1.35, 1.35)
            ax.set_ylim(-0.85, 0.85)
            ax.set_xticks([])
            ax.set_yticks([])

        return r.Plot(draw, title=f"{key}: section y = 0")

    colors = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (faces.shape[0], 1))
    colors[result["crossing"]] = (0xE8, 0x41, 0x2C)
    back = vertices[faces][:, :, 1].mean(axis=1) > 0.0
    panels: list[r.Panel | r.Plot] = [
        r.Panel(
            [r.Mesh(vertices, faces[back], face_colors=colors[back])],
            title="Back half: intersecting faces (red)",
        )
    ]
    panels += [section("input", r.RED), section("local", r.GREEN), section("voxel", r.GREEN)]
    return r.Figure(
        panels,
        camera=r.Camera(direction=(0.0, -1.0, 0.8), up=(0.0, 0.0, 1.0), zoom=1.15),
        panel_size=(560, 420),
    )
