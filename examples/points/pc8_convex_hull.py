from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC8",
    title="Exact convex hull",
    summary="""
    [`convex_hull`][ordito.points.convex_hull] returns the convex hull of a cloud as triangles
    that index the input points, the layout of [`scipy.spatial.ConvexHull`][]'s `simplices`. Each
    triangle is wound counter-clockwise seen from outside, and a hull of `V` vertices has `2V - 4`
    of them. Every decision the algorithm takes is an exact orientation sign, with the same answer
    on the CPU and the CUDA device and no tolerance to tune, so the hull is the true hull of the
    `float32` points: its vertex set equals qhull's on both clouds below, the bunny sampled at
    30 000 points and the dragon scan's 437 645 vertices.
    [`remove_unreferenced_vertices`][ordito.repair.remove_unreferenced_vertices] turns the
    triangles into a compact mesh, outward-facing and ready for a [`Trimesh`][ordito.mesh.Trimesh].
    """,
    credits=(
        (
            "Open3D: convex hull",
            "https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html",
        ),
        ("trimesh: convex", "https://trimesh.org/trimesh.convex.html"),
    ),
    notes="""
    **How it works.** The algorithm is ffHull (Gao, Cao, Tan and Huang, *Flip-Flop: Convex Hull
    Construction via Star-Shaped Polyhedron in 3D*, I3D 2013). Incremental algorithms such as
    quickhull add one point at a time. ffHull keeps the surface *star-shaped* instead of convex,
    every face seeing one fixed kernel point inside, and that weaker invariant can be kept by
    thousands of local edits at once. The surface starts as a tetrahedron. Each round, every face
    with points beyond it splits into three at the farthest of them, all at once, and the points
    the new surface encloses leave the computation for good, so the cost follows the points near
    the hull rather than the size of the cloud. The figure runs the same growth in the plane (the
    cross is the kernel point; blue points are still outside, grey ones dropped):

    ![ffHull's growth in the plane](../assets/examples/pc8-growth.svg)

    In the plane the growth ends convex. In space it ends with *reflex* edges, dents where two faces
    meet inward. Flip-Flop removes them with edge flips, again in parallel: a 2-2 flip swaps the
    shared diagonal of two triangles (red) for the other one (blue) and only grows the volume;
    where a dent cannot be flipped away, one of its vertices is provably not extreme and a 3-1 flip
    deletes it. ordito also flips during growth, which drops the points under the dents early (2.4x
    fewer flip rounds on the dragon, 3x on the bunny).

    ![A reflex edge before and after a 2-2 flip](../assets/examples/pc8-flip.webp)

    Timings against qhull, the hull behind SciPy, trimesh, Open3D and MeshLab, are on the
    [points benchmarks](../benchmarks/points.md#convex_hull) page.
    """,
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import scipy.spatial

    import ordito as od
    from examples import data

    clouds = {"bunny cloud": data.load_points("bunny_cloud", device)}
    clouds["dragon scan"], _ = data.load("dragon", device)
    hulls = {}
    for name, points in clouds.items():
        faces = od.points.convex_hull(points)
        hull_vertices = np.unique(faces.numpy())
        qhull = scipy.spatial.ConvexHull(points.numpy())
        print(
            f"{name}: {points.shape[0]} points, hull of {hull_vertices.size} vertices "
            f"and {faces.shape[0] // 3} triangles, same vertices as qhull: "
            f"{np.array_equal(hull_vertices, np.sort(qhull.vertices))}"
        )
        hulls[name] = od.repair.remove_unreferenced_vertices(points, faces)[:2]
    # --8<-- [end:code]
    return {
        name: (clouds[name].numpy(), hull[0].numpy(), hull[1].numpy().reshape(-1, 3))
        for name, hull in hulls.items()
    }


def figure(result: dict[str, Any]) -> r.Figure:
    points, hull_vertices, hull_faces = result["bunny cloud"]
    dragon_points, dragon_hull, dragon_faces = result["dragon scan"]
    dragon, faces = data.arrays("dragon")
    return r.Figure(
        [
            r.Panel([r.Points(points, color=r.GREY, size=3)], title="Bunny cloud, 30 000 points"),
            r.Panel(
                [
                    r.Points(points, color=r.GREY, size=2),
                    r.Mesh(
                        hull_vertices,
                        hull_faces,
                        color=r.GREEN,
                        opacity=0.35,
                        show_edges=True,
                        line_width=0.5,
                        smooth=False,
                    ),
                    r.Points(hull_vertices, color=r.RED, size=6),
                ],
                title=f"Hull: {hull_vertices.shape[0]:,} vertices",
            ),
            r.Panel(
                [
                    r.Mesh(
                        hull_vertices,
                        hull_faces,
                        color=r.GREEN,
                        show_edges=True,
                        line_width=0.5,
                        smooth=False,
                    )
                ],
                title=f"Hull mesh: {hull_faces.shape[0]:,} triangles",
            ),
            r.Panel(
                [
                    r.Mesh(dragon, faces, color=r.LIGHT_GREEN),
                    r.Mesh(
                        dragon_hull,
                        dragon_faces,
                        color=r.BLUE,
                        opacity=0.25,
                        show_edges=True,
                        line_width=0.5,
                        smooth=False,
                    ),
                    r.Points(dragon_hull, color=r.RED, size=4),
                ],
                title=f"Dragon: {dragon_points.shape[0]:,} points",
                camera=data.camera("dragon"),
            ),
        ],
        camera=data.camera("bunny_cloud"),
        link_bounds=False,
        ncols=4,
    )
