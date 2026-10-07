from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="T3",
    title="Rigid and mirror transforms",
    summary="""
    [`transform`][ordito.mesh.Trimesh.transform] returns a new mesh under a `4 x 4` matrix built
    by [`rotation_matrix`][ordito.transform.rotation_matrix] or
    [`reflection_matrix`][ordito.transform.reflection_matrix]. It classifies the matrix first and
    carries every cached value the transform preserves: after a rotation the cotangent Laplacian
    [`cotmatrix`][ordito.mesh.Trimesh.cotmatrix] is the same object, not a new assembly. A mirror
    also reverses the winding of every face, so the mirrored bunny keeps outward normals and a
    positive volume. [`invert`][ordito.mesh.Trimesh.invert] reverses the winding without moving
    anything, which turns the solid inside out.
    """,
    credits=(
        ("trimesh: quick start", "https://trimesh.org/quick_start.html"),
        (
            "Open3D: transformation",
            "https://www.open3d.org/docs/release/tutorial/geometry/transformation.html",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import math

    import ordito as od
    from examples import data

    mesh = od.Trimesh(*od.repair.make_solid(*data.load("bunny", device)))
    laplacian = mesh.cotmatrix  # assembled once here

    spun = mesh.transform(od.transform.rotation_matrix((0.0, 1.0, 0.0), math.pi / 2, mesh.centroid))
    mirrored = mesh.transform(od.transform.reflection_matrix((1.0, 0.0, 0.0), mesh.centroid))
    inverted = mesh.invert()

    print(f"cotmatrix carried through the rotation: {spun.cotmatrix is laplacian}")
    for name, m in [("original", mesh), ("rotated", spun), ("mirrored", mirrored)]:
        print(f"{name:>9}: volume {m.volume:.3e}")
    print(f" inverted: volume {inverted.volume:.3e}")
    # --8<-- [end:code]
    out: dict[str, Any] = {}
    for name, m in [
        ("original", mesh),
        ("rotated", spun),
        ("mirrored", mirrored),
        ("inverted", inverted),
    ]:
        out[name] = {
            "vertices": m.vertices.numpy(),
            "faces": m.faces.numpy(),
            "centers": m.triangles_center.numpy(),
            "normals": m.face_normals.numpy(),
        }
    return out


def _arrows(mesh: dict[str, Any], color: str) -> r.Arrows:
    pick = np.random.default_rng(0).choice(mesh["centers"].shape[0], 300, replace=False)
    return r.Arrows(mesh["centers"][pick], mesh["normals"][pick], color=color, scale=0.012)


def figure(result: dict[str, Any]) -> r.Figure:
    original, rotated = result["original"], result["rotated"]
    mirrored, inverted = result["mirrored"], result["inverted"]
    return r.Figure(
        [
            r.Panel([r.Mesh(original["vertices"], original["faces"])], title="Original"),
            r.Panel([r.Mesh(rotated["vertices"], rotated["faces"])], title="Rotated 90° about y"),
            r.Panel(
                [r.Mesh(mirrored["vertices"], mirrored["faces"]), _arrows(mirrored, r.BLUE)],
                title="Mirrored in x: normals outward",
            ),
            r.Panel(
                [
                    r.Mesh(inverted["vertices"], inverted["faces"], opacity=0.3),
                    _arrows(inverted, r.RED),
                ],
                title="Inverted: normals inward",
            ),
        ],
        camera=data.camera("bunny"),
        panel_size=(560, 560),
    )
