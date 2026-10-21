from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="RP2",
    title="Sums that repeat bit for bit",
    summary="""
    A floating-point sum depends on the order its terms are added in, and on CUDA the blocks of a
    reduction commit in the order they finish. [`volume`][ordito.measures.volume] of the dragon,
    called 64 times with the setting off, returns several values a few `float32` units in the last
    place apart; with it on, one value. Per-group sums go further:
    [`scatter_sum`][ordito.reduce.scatter_sum] sums each group exactly in fixed point and rounds
    once, so with the setting on its result is also the same on the CPU and the CUDA device.
    """,
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("dragon", device)
    volumes = {}
    for enabled in (False, True):
        with od.config.deterministic(enabled):
            volumes[enabled] = np.array([od.measures.volume(vertices, faces) for _ in range(64)])
        print(f"deterministic={enabled}: {np.unique(volumes[enabled]).size} distinct volumes")

    rng = np.random.default_rng(0)
    values = rng.normal(size=200_000).astype(np.float32)
    groups = rng.integers(0, 16, size=values.size).astype(np.int32)
    with od.config.deterministic(True):
        sums = {
            name: od.reduce.scatter_sum(
                wp.array(values, device=name), wp.array(groups, device=name), 16
            ).numpy()
            for name in ("cpu", device)
        }
    print("CPU and CUDA group sums identical:", np.array_equal(sums["cpu"], sums[device]))
    # --8<-- [end:code]
    return {"volumes": volumes}


def figure(result: dict[str, Any]) -> r.Figure:
    volumes = result["volumes"]
    reference = float(volumes[True][0])
    ulp = float(np.spacing(np.float32(reference)))

    def draw(ax: Any) -> None:
        for row, (enabled, color, label) in enumerate(
            ((False, r.RED, "setting off"), (True, r.GREEN, "setting on"))
        ):
            offsets = (volumes[enabled] - reference) / ulp
            jitter = np.random.default_rng(row).uniform(-0.18, 0.18, offsets.size)
            ax.scatter(offsets, row + jitter, s=28, color=color, alpha=0.75, linewidths=0)
            ax.text(
                max(float(offsets.max()), 0.0) + 0.6,
                row,
                f"{label}: {np.unique(volumes[enabled]).size} distinct",
                color=r.TEXT,
                va="center",
                fontsize=11,
            )
        ax.set_yticks([])
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel("volume - setting-on value, in float32 ulps", color=r.TEXT)
        ax.tick_params(colors=r.TEXT)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(r.TEXT)
        ax.axvline(0.0, color=r.TEXT, linewidth=0.8, alpha=0.5)

    return r.Figure(
        [r.Plot(draw, title="Dragon volume, 64 calls per setting", aspect="auto")],
        panel_size=(1100, 420),
    )
