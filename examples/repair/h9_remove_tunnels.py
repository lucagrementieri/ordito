from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="H9",
    title="Removing tunnels",
    summary="""
    A ring torus carries three thin handles fused onto its tube, the kind of spurious topology a
    scan picks up where two sheets touch: genus 4, of which only the ring's own tunnel is meant.
    [`remove_tunnels`][ordito.repair.remove_tunnels] computes a homology basis, shortens each
    loop within its class, and cuts the surface along every loop no longer than `max_length`,
    sealing both rims of each cut (red); nothing moves, so the cut is only visible as new
    topology. The threshold sits far below the ring's own loops, so the
    intended tunnel survives. One call removes at most one tunnel per family of dependent loops,
    hence the loop until nothing is cut.
    [`euler_characteristic`][ordito.measures.euler_characteristic] tracks the genus.

    The test is on *shortened* loop length, and shortening is a local descent: here the basis
    loops through one of the handles stay several times longer than its girth, so that handle is
    not found at this threshold.
    """,
    credits=(("MeshLib examples", "https://meshlib.io/documentation/Examples.html"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("thin_handles", device)
    print("genus before:", (2 - od.measures.euler_characteristic(faces)) // 2)

    n_input = faces.shape[0] // 3
    removed = 1
    while removed:
        vertices, faces, removed = od.repair.remove_tunnels(vertices, faces, max_length=1.0)
        print(f"cut {removed} tunnel(s)")
    print("genus after:", (2 - od.measures.euler_characteristic(faces)) // 2)
    # --8<-- [end:code]
    return {"vertices": vertices.numpy(), "faces": faces.numpy().reshape(-1, 3), "n_input": n_input}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("thin_handles")
    v, f = result["vertices"], result["faces"]
    colors = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (f.shape[0], 1))
    colors[result["n_input"] :] = (0xF3, 0x9C, 0x12)
    sealed = f[result["n_input"] :]
    patch = v[sealed].mean(axis=1)
    # The rims of the cut (red): the boundary edges of the sealing patches.
    edges = np.sort(sealed[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    cut = v[unique[count == 1]]
    panels = [
        r.Panel([r.Mesh(vertices, faces, color=r.LIGHT_GREEN)], title="Genus 4"),
        r.Panel(
            [r.Mesh(v, f, color=r.GREEN), r.Segments(cut, color=r.RED, width=6.0)],
            title="After remove_tunnels (cuts red)",
        ),
    ]
    patch_angle = np.arctan2(patch[:, 1], patch[:, 0]) % (2 * np.pi)
    for angle in (0.3, 2.4, 4.2):
        near = np.abs(patch_angle - angle) < 0.5
        if near.any():
            center, half, title = patch[near].mean(axis=0), 0.12, "Cut and sealed"
        else:
            center, half, title = np.array([np.cos(angle), np.sin(angle), 0.45]), 0.3, "Not found"
        panels.append(
            r.Panel(
                [
                    r.Mesh(v, f, face_colors=colors, show_edges=True, line_width=0.4),
                    r.Segments(cut, color=r.RED, width=4.0),
                ],
                title=f"{title} ({np.degrees(angle):.0f}°)",
                bounds=np.stack([center - half, center + half]),
            )
        )
    return r.Figure(panels, camera=data.camera("thin_handles"), ncols=5, panel_size=(480, 440))
