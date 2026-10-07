from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="H6",
    title="Consistent and outward orientation",
    summary="""
    A fifth of the bunny's triangles have their winding reversed (red: facing inward).
    [`face_flip_mask`][ordito.validation.face_flip_mask] propagates an orientation across shared
    edges and flags every face that disagrees with its component's first face;
    [`make_winding_consistent`][ordito.repair.make_winding_consistent] applies those flips. That
    makes the winding coherent but not necessarily outward: here the seed face happened to face
    inward, so everything did. [`make_normals_outward`][ordito.repair.make_normals_outward] also
    fixes the global sign from the enclosed volume. The scan is open at its base, so it is
    called with `multibody=True`: the default only flips a watertight mesh (trimesh's rule).
    """,
    credits=(
        ("libigl 706", "https://libigl.github.io/tutorial/#facet-orientation"),
        ("PyMeshLab", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
        ("MeshLib examples", "https://meshlib.io/documentation/Examples.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("flipped_bunny", device)
    print("winding consistent:", od.validation.is_winding_consistent(faces))
    flips = od.validation.face_flip_mask(faces).numpy()
    print(f"faces to flip: {flips.sum()} of {flips.size}")

    consistent = od.repair.make_winding_consistent(faces)
    # The scan has holes in its base: the whole-mesh rule only flips a watertight mesh, the
    # per-body rule trusts each component's own signed volume.
    outward = od.repair.make_normals_outward(vertices, faces, multibody=True)
    for name, f in (("consistent", consistent), ("outward", outward)):
        print(f"{name}: signed volume {od.measures.volume(vertices, f):+.6f}")
    # --8<-- [end:code]
    return {
        "flips": flips,
        "consistent": consistent.numpy().reshape(-1, 3),
        "outward": outward.numpy().reshape(-1, 3),
    }


def _inward_colors(faces: np.ndarray) -> np.ndarray:
    """Red where a face's winding is opposite the original (outward) scan's, green elsewhere."""
    _, reference = data.arrays("bunny")
    same = np.zeros(faces.shape[0], dtype=bool)
    for shift in range(3):
        same |= np.all(np.roll(faces, shift, axis=1) == reference, axis=1)
    colors = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (faces.shape[0], 1))
    colors[~same] = (0xE8, 0x41, 0x2C)
    return colors


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("flipped_bunny")
    flips = np.tile(np.array([0xB8, 0xDC, 0x80], dtype=np.uint8), (faces.shape[0], 1))
    flips[result["flips"]] = (0x2C, 0x7B, 0xE8)
    return r.Figure(
        [
            r.Panel(
                [r.Mesh(vertices, faces, face_colors=_inward_colors(faces))],
                title="Input (red faces inward)",
            ),
            r.Panel([r.Mesh(vertices, faces, face_colors=flips)], title="face_flip_mask (blue)"),
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        result["consistent"],
                        face_colors=_inward_colors(result["consistent"]),
                    )
                ],
                title="make_winding_consistent",
            ),
            r.Panel(
                [
                    r.Mesh(
                        vertices, result["outward"], face_colors=_inward_colors(result["outward"])
                    )
                ],
                title="make_normals_outward",
            ),
        ],
        camera=data.camera("flipped_bunny"),
        panel_size=(520, 500),
    )
