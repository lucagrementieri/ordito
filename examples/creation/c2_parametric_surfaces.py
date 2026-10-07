from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="C2",
    title="Parametric surfaces",
    summary="""
    [`parametric_surface`][ordito.creation.parametric_surface] samples sixteen classical surfaces:
    minimal surfaces, surfaces of revolution and immersions of non-orientable ones. Seams and poles
    are glued in the index buffer, so the topology is exact at any resolution: a
    [`Trimesh`][ordito.mesh.Trimesh] reports the Euler characteristic, the number of boundary loops
    and whether the surface is orientable. Each face is coloured by the side that faces the
    camera, green for the front of its winding and blue for the back. On a non-orientable surface
    the two colours meet along a seam that no consistent winding can remove.
    """,
    credits=(
        (
            "PyVista: parametric objects",
            "https://docs.pyvista.org/examples/00-load/create_parametric_geometric_objects",
        ),
    ),
)

_VIEW = r.Camera(direction=(1.0, -1.3, 0.9), up=(0.0, 0.0, 1.0), zoom=1.0)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od

    kinds: list[od.creation.ParametricSurfaceKind] = [
        "bohemian_dome", "bour", "boy", "catalan_minimal", "conic_spiral", "cross_cap",
        "dini", "enneper", "figure8_klein", "henneberg", "klein", "kuen",
        "mobius", "plucker_conoid", "pseudosphere", "roman",
    ]  # fmt: skip
    surfaces = {}
    for kind in kinds:
        surfaces[kind] = od.Trimesh(*od.creation.parametric_surface(kind, 80, 80, device=device))
    closed = [k for k, m in surfaces.items() if not m.boundary_loops]
    print("closed:", ", ".join(closed))
    print("non-orientable:", ", ".join(k for k, m in surfaces.items() if not m.is_orientable))
    # --8<-- [end:code]
    return {
        kind: {
            "vertices": m.vertices.numpy(),
            "faces": m.faces.numpy().reshape(-1, 3),
            "chi": m.euler_characteristic,
            "orientable": m.is_orientable,
            "loops": len(m.boundary_loops),
        }
        for kind, m in surfaces.items()
    }


def _two_sided(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Green where a face's winding faces the camera, blue where its back does."""
    tri = vertices[faces].astype(np.float64)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    direction = np.asarray(_VIEW.direction) / np.linalg.norm(_VIEW.direction)
    eye = (lo + hi) / 2 + direction * np.linalg.norm(hi - lo) * 2.0
    front = np.einsum("ij,ij->i", normals, eye - tri.mean(axis=1)) >= 0.0
    return np.where(front[:, None], [0x76, 0xB9, 0x00], [0x2C, 0x7B, 0xE8]).astype(np.uint8)


def figure(result: dict[str, Any]) -> r.Figure:
    panels = []
    for kind, s in result.items():
        tags = [f"χ = {s['chi']}"]
        tags.append(
            f"{s['loops']} rim" + ("s" if s["loops"] != 1 else "") if s["loops"] else "closed"
        )
        if not s["orientable"]:
            tags.append("non-orientable")
        panels.append(
            r.Panel(
                [
                    r.Mesh(
                        s["vertices"], s["faces"], face_colors=_two_sided(s["vertices"], s["faces"])
                    )
                ],
                title=f"{kind}\n{', '.join(tags)}",
            )
        )
    return r.Figure(panels, ncols=4, camera=_VIEW, link_bounds=False, panel_size=(440, 400))
