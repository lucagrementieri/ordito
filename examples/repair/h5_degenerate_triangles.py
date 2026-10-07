from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="H5",
    title="Degenerate triangles, duplicate vertices and T-vertices",
    summary="""
    Two small patches carry the defects that break normals, cotangent weights and curvature.

    - The first has needles, caps and exactly degenerate triangles (marked).
      [`face_nondegenerate_mask`][ordito.triangles.face_nondegenerate_mask] finds the zero-area
      ones and [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces] would drop
      them; [`collapse_small_triangles`][ordito.repair.collapse_small_triangles] instead
      collapses the shortest edge of every triangle below an area threshold, which removes the
      needles and caps as well without opening a hole.
    - The second is two grids stitched at different resolutions and never welded.
      [`remove_duplicated_vertices`][ordito.repair.remove_duplicated_vertices] welds the seam;
      the fine side's extra seam vertices are then T-vertices, each with a sliver triangle
      (marked) across the coarse edge, and [`flip_t_vertices`][ordito.repair.flip_t_vertices] flips
      those slivers away.
    """,
    credits=(
        (
            "MeshLib: fix degeneracies",
            "https://meshlib.io/documentation/ExampleMeshFixDegeneracies.html",
        ),
        ("PyMeshLab: cleaning", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("sliver_patch", device)
    zero_area = ~od.triangles.face_nondegenerate_mask(vertices, faces).numpy()
    _, dropped = od.repair.remove_degenerate_faces(vertices, faces)
    collapsed_vertices, collapsed = od.repair.collapse_small_triangles(vertices, faces)
    print(f"zero-area faces: {zero_area.sum()}; faces after dropping them: {dropped.shape[0] // 3}")
    print(f"faces after collapsing small ones: {faces.shape[0] // 3} -> {collapsed.shape[0] // 3}")

    patch_vertices, patch_faces = data.load("t_vertex_patch", device)
    welded_vertices, _, _, welded = od.repair.remove_duplicated_vertices(
        patch_vertices, patch_faces, epsilon=1e-6
    )
    flipped = od.repair.flip_t_vertices(welded_vertices, welded)
    for name, f in (("welded", welded), ("flipped", flipped)):
        quality = od.triangles.face_quality(welded_vertices, f).numpy()
        print(f"{name}: {(quality > 40).sum()} slivers, worst aspect ratio {quality.max():.0f}")
    print(f"seam vertices welded: {patch_vertices.shape[0] - welded_vertices.shape[0]}")
    # --8<-- [end:code]
    return {
        "zero_area": zero_area,
        "collapsed": (collapsed_vertices.numpy(), collapsed.numpy().reshape(-1, 3)),
        "welded": (welded_vertices.numpy(), welded.numpy().reshape(-1, 3)),
        "flipped": flipped.numpy().reshape(-1, 3),
    }


def _slivers(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Centroids of the triangles whose aspect ratio exceeds 40 (the T-vertex slivers)."""
    tri = vertices[faces].astype(np.float64)
    a, b, c = (np.linalg.norm(tri[:, i] - tri[:, j], axis=1) for i, j in ((1, 2), (2, 0), (0, 1)))
    s = (a + b + c) / 2
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))
    aspect = a * b * c / np.maximum(8 * area * area / s, 1e-30)  # circumradius / (2 inradius)
    return tri[aspect > 40].mean(axis=1)


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("sliver_patch")
    tri = vertices[faces].astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    small = (area < 1e-6) & ~result["zero_area"]
    cv, cf = result["collapsed"]
    patch_v, _ = data.arrays("t_vertex_patch")
    wv, wf = result["welded"]
    edges: r.MeshStyle = {"show_edges": True, "smooth": False, "line_width": 0.6}
    t_camera = data.camera("t_vertex_patch")
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.LIGHT_GREEN, **edges),
                    r.Points(tri[result["zero_area"]].mean(axis=1), color=r.RED, size=16),
                    r.Points(tri[small].mean(axis=1), color=r.ORANGE, size=16),
                ],
                title="Zero-area (red), tiny (orange)",
            ),
            r.Panel([r.Mesh(cv, cf, **edges)], title="collapse_small_triangles"),
            r.Panel(
                [
                    r.Mesh(wv, wf, color=r.LIGHT_GREEN, **edges),
                    r.Points(_slivers(wv, wf), color=r.RED, size=16),
                ],
                title="Welded: T-vertex slivers",
                camera=t_camera,
                bounds=np.stack([patch_v.min(axis=0), patch_v.max(axis=0)]),
            ),
            r.Panel(
                [r.Mesh(wv, result["flipped"], **edges)],
                title="flip_t_vertices",
                camera=t_camera,
                bounds=np.stack([patch_v.min(axis=0), patch_v.max(axis=0)]),
            ),
        ],
        camera=data.camera("sliver_patch"),
        ncols=4,
        panel_size=(560, 520),
    )
