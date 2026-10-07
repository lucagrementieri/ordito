from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="T2",
    title="Bodies: split, explode and recombine",
    summary="""
    Six primitives packed into one buffer form six bodies.
    [`face_connected_component_labels`][ordito.mesh.Trimesh.face_connected_component_labels] names
    each face's body and [`body_count`][ordito.mesh.Trimesh.body_count] counts them.
    [`split`][ordito.mesh.Trimesh.split] turns them into one `Trimesh` each. Every body is then
    moved away from the assembly's centre of mass with
    [`transform`][ordito.mesh.Trimesh.transform], and `+` concatenates the moved bodies back into
    one mesh: an exploded view. [`submesh`][ordito.mesh.Trimesh.submesh] cuts out a selection of
    faces as a mesh of its own.
    """,
    credits=(
        ("trimesh: quick start", "https://trimesh.org/quick_start.html"),
        ("libigl 809", "https://libigl.github.io/tutorial/#exploded-view"),
        (
            "Open3D: connected components",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import functools
    import operator

    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    mesh = od.Trimesh(*data.load("parts", device))
    labels = mesh.face_connected_component_labels
    print(f"{mesh.body_count} bodies, volume {mesh.volume:.3f}")

    center = np.array(mesh.center_mass)
    exploded = [
        body.transform(
            od.transform.translation_matrix((0.8 * (np.array(body.center_mass) - center)).tolist())
        )
        for body in mesh.split()
    ]
    recombined = functools.reduce(operator.add, exploded)
    print(f"recombined: {recombined.n_faces} faces in {recombined.body_count} bodies")

    upper_half = mesh.submesh(
        wp.array(mesh.triangles_center.numpy()[:, 1] > 0.0, dtype=wp.bool, device=device)
    )
    print(f"upper half: {upper_half.n_faces} faces in {upper_half.body_count} bodies")
    # --8<-- [end:code]
    return {
        "labels": labels.numpy(),
        "centers": np.array([np.array(b.center_mass) for b in mesh.split()]),
        "center": center,
        "recombined": (recombined.vertices.numpy(), recombined.faces.numpy()),
        "recombined_labels": recombined.face_connected_component_labels.numpy(),
        "upper": (upper_half.vertices.numpy(), upper_half.faces.numpy()),
        "upper_labels": upper_half.face_connected_component_labels.numpy(),
    }


def _colors(labels: np.ndarray) -> np.ndarray:
    _, body = np.unique(labels, return_inverse=True)
    palette = (np.array(cast("ListedColormap", plt.get_cmap("tab10")).colors) * 255).astype(
        np.uint8
    )
    return palette[body % len(palette)]


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("parts")
    recombined_v, recombined_f = result["recombined"]
    upper_v, upper_f = result["upper"]
    centers, center = result["centers"], result["center"]
    moved = centers + 0.8 * (centers - center)
    frame = np.stack([recombined_v.min(axis=0), recombined_v.max(axis=0)])
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(vertices, faces, face_colors=_colors(result["labels"]))],
                title="Six bodies",
                bounds=frame,
            ),
            r.Panel(
                [
                    r.Mesh(
                        recombined_v, recombined_f, face_colors=_colors(result["recombined_labels"])
                    ),
                    r.Segments(
                        np.stack([np.broadcast_to(center, moved.shape), moved], axis=1),
                        color=r.GREY,
                        width=2.0,
                    ),
                    r.Points(center[None], color=r.ORANGE, size=20),
                ],
                title="Exploded and recombined",
                bounds=frame,
            ),
            r.Panel(
                [
                    r.Mesh(vertices, faces, color=r.GREY, opacity=0.25),
                    r.Mesh(upper_v, upper_f, face_colors=_colors(result["upper_labels"])),
                ],
                title="Submesh: faces above y = 0",
            ),
        ],
        camera=data.camera("parts"),
        link_bounds=False,
    )
