from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="T6",
    title="Caching and functional updates",
    summary="""
    A [`Trimesh`][ordito.mesh.Trimesh] is frozen: an update returns a new mesh and decides which
    cached values are still valid. [`with_vertices`][ordito.mesh.Trimesh.with_vertices] keeps the
    faces, so it carries forward every cache that is computed from the faces alone. These are the
    edge tables, face adjacency, half-edges, boundaries, manifold and orientation checks, body
    count and the uniform [`laplacian_operator`][ordito.mesh.Trimesh.laplacian_operator]. Anything
    that reads positions is recomputed on first access: normals, areas, angles, edge lengths,
    [`vertex_defects`][ordito.mesh.Trimesh.vertex_defects], the BVH, the cotangent operators and
    the heat bundles. Here the noisy bunny is smoothed with
    [`filter_taubin`][ordito.smoothing.filter_taubin], reusing the cached uniform Laplacian, and the
    angle defect (discrete Gaussian curvature) is read again on the smoothed mesh.
    """,
    credits=(("trimesh: caching", "https://trimesh.org/trimesh.caching.html"),),
    notes="""
    [`with_faces`][ordito.mesh.Trimesh.with_faces] carries nothing, since every cached value
    depends on the faces. [`copy`][ordito.mesh.Trimesh.copy] makes independent buffers, and
    [`invalidate`][ordito.mesh.Trimesh.invalidate] clears the cache after a buffer has been
    edited in place by a kernel.
    """,
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    mesh = od.Trimesh(*data.load("noisy_bunny", device))
    names = [
        "edges_unique",
        "face_adjacency",
        "halfedge_twins",
        "boundary_loops",
        "laplacian_operator",
        "face_normals",
        "edges_unique_length",
        "vertex_defects",
        "cotmatrix",
        "heat_operators",
    ]
    before = {name: getattr(mesh, name) for name in names}  # compute and cache

    smoothed_vertices = od.smoothing.filter_taubin(
        mesh.vertices, mesh.faces, iterations=20, laplacian_operator=mesh.laplacian_operator
    )
    smoothed = mesh.with_vertices(smoothed_vertices)
    kept = [name for name in names if getattr(smoothed, name) is before[name]]
    print("carried forward:", ", ".join(kept))
    print("recomputed:     ", ", ".join(name for name in names if name not in kept))
    print(f"mean edge length {mesh.mean_edge_length:.6f} -> {smoothed.mean_edge_length:.6f}")
    # --8<-- [end:code]
    return {
        "noisy": mesh.vertex_defects.numpy(),
        "smooth": smoothed.vertex_defects.numpy(),
        "smooth_vertices": smoothed.vertices.numpy(),
        "kept": kept,
        "names": names,
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("noisy_bunny")
    clim = r.symmetric_clim(result["smooth"], 98.0)
    rows = [["cached property", "after with_vertices"]] + [
        [name, "carried" if name in result["kept"] else "recomputed"] for name in result["names"]
    ]
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(vertices, faces, scalars=result["noisy"], cmap="RdBu_r", clim=clim)],
                title="Noisy bunny: angle defect",
            ),
            r.Panel(
                [
                    r.Mesh(
                        result["smooth_vertices"],
                        faces,
                        scalars=result["smooth"],
                        cmap="RdBu_r",
                        clim=clim,
                        scalar_bar="angle defect",
                    )
                ],
                title="Smoothed: angle defect recomputed",
            ),
            r.Table(rows, title="What survives with_vertices"),
        ],
        camera=data.camera("noisy_bunny"),
        panel_size=(620, 580),
    )
