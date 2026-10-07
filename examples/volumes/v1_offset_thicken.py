from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="V1",
    title="Offset, thicken and shell",
    summary="""
    [`offset_mesh`][ordito.levelset.offset_mesh] moves the surface by a signed distance through
    the level set of its distance field, so concave regions and thin parts are handled exactly:
    growing rounds over the creases, shrinking thins the ears down to their last voxels.
    [`thicken_mesh`][ordito.levelset.thicken_mesh] instead keeps the input's own triangulation:
    it displaces a copy of every vertex inward along its normal, reverses it and joins the two
    layers along the open rims, giving a shell (shown cut open, inner layer orange).
    """,
    credits=(("MeshLib: offset", "https://meshlib.io/documentation/ExampleMeshOffset.html"),),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    grown = od.levelset.offset_mesh(vertices, faces, 0.006)
    shrunk = od.levelset.offset_mesh(vertices, faces, -0.003)
    shell = od.levelset.thicken_mesh(vertices, faces, 0.006)
    for name, (v, f) in {"grown": grown, "shrunk": shrunk, "shell": shell}.items():
        print(f"{name}: {f.shape[0] // 3} faces, volume {od.measures.volume(v, f):.3e}")
    # --8<-- [end:code]
    as_np = {}
    for name, (v, f) in {"grown": grown, "shrunk": shrunk, "shell": shell}.items():
        as_np[name] = (v.numpy(), f.numpy().reshape(-1, 3))
    return as_np


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    sv, sf = result["shell"]
    cut = sv[sf][:, :, 2].mean(axis=1) < 0.0  # drop the front half to show the wall
    layer = np.tile(np.array([0x76, 0xB9, 0x00], dtype=np.uint8), (sf.shape[0], 1))
    layer[faces.shape[0] :] = (0xF3, 0x9C, 0x12)  # the inner layer and the rim bands
    gv, gf = result["grown"]
    iv, if_ = result["shrunk"]
    return r.Figure(
        [
            r.Panel([r.Mesh(gv, gf)], title="offset_mesh, +0.006"),
            r.Panel([r.Mesh(iv, if_)], title="offset_mesh, -0.003"),
            r.Panel(
                [r.Mesh(sv, sf[cut], face_colors=layer[cut], smooth=False)],
                title="thicken_mesh, cut open",
                bounds=np.stack([vertices.min(axis=0), vertices.max(axis=0)]),
            ),
        ],
        camera=data.camera("bunny"),
        panel_size=(520, 500),
    )
