from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="I8",
    title="Self-intersections and mesh-mesh collisions",
    summary="""
    A trefoil tube too thick for its path passes through itself where the strands cross;
    [`face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask] flags every face
    that crosses another face of the same mesh. For two separate meshes,
    [`collision_masks`][ordito.intersection.collision_masks] flags the faces of each that touch the
    other, and [`mesh_with_mesh`][ordito.intersection.mesh_with_mesh] returns the intersection
    curve as segments. The two overlapping spheres come out of one input buffer through
    [`split`][ordito.combine.split].
    """,
    credits=(
        ("MeshLib: mesh collision", "https://meshlib.io/documentation/index.html"),
        ("libigl 903", "https://libigl.github.io/tutorial/#self-intersections"),
        ("PyVista: collision", "https://docs.pyvista.org/examples/01-filter/collision"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("fat_knot", device)
    crossing = od.validation.face_self_intersecting_mask(vertices, faces)
    print(f"knot: {crossing.numpy().sum()} of {faces.shape[0] // 3} faces cross another face")

    (vertices_a, faces_a), (vertices_b, faces_b) = od.combine.split(
        *data.load("two_spheres", device)
    )
    hit_a, hit_b = od.intersection.collision_masks(vertices_a, faces_a, vertices_b, faces_b)
    curve = od.intersection.mesh_with_mesh(vertices_a, faces_a, vertices_b, faces_b)
    print(f"spheres: {hit_a.numpy().sum()} + {hit_b.numpy().sum()} colliding faces")
    print(f"intersection curve: {curve.shape[0]} segments")
    # --8<-- [end:code]
    return {
        "crossing": crossing.numpy(),
        "a": (vertices_a.numpy(), faces_a.numpy().reshape(-1, 3), hit_a.numpy()),
        "b": (vertices_b.numpy(), faces_b.numpy().reshape(-1, 3), hit_b.numpy()),
        "curve": curve.numpy().reshape(-1, 2, 3),
    }


def _colors(mask: np.ndarray, base: tuple[int, int, int]) -> np.ndarray:
    colors = np.tile(np.array(base, dtype=np.uint8), (mask.shape[0], 1))
    colors[mask] = (0xE8, 0x41, 0x2C)
    return colors


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("fat_knot")
    va, fa, ha = result["a"]
    vb, fb, hb = result["b"]
    both = np.concatenate([va, vb])
    pair = np.stack([both.min(axis=0), both.max(axis=0)])
    return r.Figure(
        [
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREEN, opacity=0.3),
                    r.Mesh(vertices, faces[result["crossing"]], color=r.RED),
                ],
                title="Self-intersecting faces",
                camera=data.camera("fat_knot"),
            ),
            r.Panel(
                [
                    r.Mesh(va, fa, face_colors=_colors(ha, (0x76, 0xB9, 0x00))),
                    r.Mesh(vb, fb, face_colors=_colors(hb, (0xB8, 0xDC, 0x80))),
                ],
                title="Colliding faces",
                camera=data.camera("two_spheres"),
                bounds=pair,
            ),
            r.Panel(
                [
                    r.Mesh(va, fa, color=r.LIGHT_GREEN, opacity=0.35),
                    r.Mesh(vb, fb, color=r.GREY, opacity=0.35),
                    r.Segments(result["curve"], color=r.RED, width=5.0),
                ],
                title="Intersection curve",
                camera=data.camera("two_spheres"),
                bounds=pair,
            ),
        ],
        link_bounds=False,
        panel_size=(560, 520),
    )
