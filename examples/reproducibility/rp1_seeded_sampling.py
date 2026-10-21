from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="RP1",
    title="A seeded sampler, called twice",
    summary="""
    [`sample_surface`][ordito.sample.sample_surface] picks each sample's face by searching a CDF
    of the face areas, which a `float32` scan builds. On CUDA that scan combines its blocks in
    arrival order, so with the setting off two calls with the same seed can round a few CDF
    entries differently, and a sample drawn next to one of those entries lands on the adjacent
    face *in index order*, which on a scan can be across the model. Inside
    [`deterministic`][ordito.config.deterministic] the scan takes a fixed order and the second call
    returns the first call's points bit for bit. The red segments join a sample's two positions.
    """,
    notes="""
    The count of moved points changes from run to run with the setting off; it is zero on the CPU
    device, which runs its blocks in turn, whatever the setting.
    """,
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("dragon", device)
    pairs = {}
    for enabled in (False, True):
        with od.config.deterministic(enabled):
            first, _ = od.sample.sample_surface(vertices, faces, 100_000, seed=7)
            second, _ = od.sample.sample_surface(vertices, faces, 100_000, seed=7)
        moved = np.any(first.numpy() != second.numpy(), axis=1)
        print(f"deterministic={enabled}: {moved.sum()} of 100000 samples moved")
        pairs[enabled] = (first.numpy()[moved], second.numpy()[moved])
    # --8<-- [end:code]
    return {"pairs": pairs}


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("dragon")
    panels = []
    for enabled, label in ((False, "off"), (True, "on")):
        first, second = result["pairs"][enabled]
        layers: list[r.Layer] = [r.Mesh(vertices, faces, color=r.GREY)]
        if first.shape[0]:
            layers += [
                r.Segments(np.stack([first, second], axis=1), color=r.RED, width=2.0),
                r.Points(first, color=r.RED, size=7),
            ]
        panels.append(r.Panel(layers, title=f"Setting {label}: {first.shape[0]:,} samples moved"))
    return r.Figure(panels, camera=data.camera("dragon"), link_bounds=False)
