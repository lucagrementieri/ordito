"""
Extra figures for the gallery notes, written by ``python -m examples._diagrams``.

``pc8-growth.svg`` draws the convex-hull example's picture of ffHull's growth in the plane. It is a
real run of the two-dimensional analogue, not a drawing: points are assigned to the edge whose cone
from the kernel point holds them, every edge with points beyond it is split at the farthest one in
the same round, and the points the new polygon encloses are dropped. In the plane this growth ends
convex; in space it ends star-shaped, which ``pc8-flip.webp`` (a rendered 2-2 flip of a reflex edge)
illustrates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from examples import _render as r

mpl.use("Agg")

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "examples"


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (
        b[..., 0] - o[..., 0]
    )


def _owner(
    points: np.ndarray, polygon: list[int], cloud: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    """Index of the polygon edge whose cone from ``kernel`` holds each point."""
    angles = np.arctan2(*(cloud[polygon] - kernel).T[::-1])
    query = np.arctan2(*(points - kernel).T[::-1])
    order = np.argsort(angles)
    slot = np.searchsorted(angles[order], query, side="right") - 1
    return order[slot % len(polygon)]


def grow(cloud: np.ndarray) -> tuple[np.ndarray, list[list[int]], list[np.ndarray]]:
    """Return the kernel point, the polygon after each round, and the live points before it."""
    lo, hi = int(np.argmin(cloud[:, 0])), int(np.argmax(cloud[:, 0]))
    third = int(np.argmax(np.abs(_cross(cloud[lo], cloud[hi], cloud))))
    polygon = [lo, hi, third] if _cross(cloud[lo], cloud[hi], cloud[third]) > 0 else [lo, third, hi]
    kernel = cloud[polygon].mean(axis=0)
    live = np.setdiff1d(np.arange(cloud.shape[0]), polygon)
    rounds, alive = [list(polygon)], [live]
    while live.size:
        owner = _owner(cloud[live], polygon, cloud, kernel)
        a = cloud[np.array(polygon)[owner]]
        b = cloud[np.roll(np.array(polygon), -1)[owner]]
        height = -_cross(a, b, cloud[live])
        outside = height > 0
        live, owner, height = live[outside], owner[outside], height[outside]
        if not live.size:
            break
        new_polygon = []
        for edge, vertex in enumerate(polygon):
            new_polygon.append(vertex)
            mine = owner == edge
            if mine.any():
                new_polygon.append(int(live[mine][np.argmax(height[mine])]))
        polygon = new_polygon
        live = np.setdiff1d(live, polygon)
        rounds.append(list(polygon))
        alive.append(live)
    return kernel, rounds, alive


def _panel(ax: Any, cloud: np.ndarray, title: str) -> None:
    ax.set_title(title, color=r.TEXT, fontsize=11, pad=6)
    ax.set_aspect("equal")
    ax.axis("off")
    pad = 0.08 * np.ptp(cloud, axis=0).max()
    ax.set_xlim(cloud[:, 0].min() - pad, cloud[:, 0].max() + pad)
    ax.set_ylim(cloud[:, 1].min() - pad, cloud[:, 1].max() + pad)


def _polygon(ax: Any, cloud: np.ndarray, polygon: list[int], color: str) -> None:
    p = cloud[np.array([*polygon, polygon[0]])]
    ax.fill(p[:, 0], p[:, 1], color=color, alpha=0.18, linewidth=0)
    ax.plot(p[:, 0], p[:, 1], color=color, linewidth=1.8, solid_joinstyle="round")


def ffhull_2d(seed: int = 3) -> Figure:
    """Return the four-panel figure: the start, two growth rounds, and the finished polygon."""
    rng = np.random.default_rng(seed)
    radius = rng.uniform(0.0, 1.0, 140) ** 0.6
    angle = rng.uniform(0.0, 2.0 * np.pi, 140)
    cloud = np.column_stack([1.25 * radius * np.cos(angle), radius * np.sin(angle)])
    kernel, rounds, alive = grow(cloud)
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.6))
    stages = [
        ("1. A simplex and its kernel point", rounds[0], alive[0]),
        ("2. Every edge split at once", rounds[1], alive[1]),
        ("3. Enclosed points dropped", rounds[2], alive[2]),
        (f"4. Nothing outside: {len(rounds) - 1} rounds", rounds[-1], alive[-1]),
    ]
    for ax, (title, polygon, live) in zip(axes, stages, strict=True):
        _panel(ax, cloud, title)
        dropped = np.setdiff1d(np.arange(cloud.shape[0]), np.union1d(live, polygon))
        ax.scatter(*cloud[dropped].T, s=6, color=r.GREY, linewidths=0)
        ax.scatter(*cloud[live].T, s=12, color=r.BLUE, linewidths=0)
        _polygon(ax, cloud, polygon, r.GREEN)
        ax.scatter(*cloud[polygon].T, s=22, color=r.GREEN, zorder=3, linewidths=0)
        ax.scatter(*kernel, s=40, marker="x", color=r.ORANGE, linewidths=2.0, zorder=4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.05)
    return fig


def flip_figure() -> r.Figure:
    """Return a reflex edge between two triangles, and the same four points after a 2-2 flip."""
    a, c = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    b, d = np.array([0.0, -1.0, 0.8]), np.array([0.0, 1.0, 0.8])
    kernel = np.array([[0.0, 0.0, -1.0]])
    vertices = np.stack([a, b, c, d])
    outline = [np.stack([p, q]) for p, q in ((a, b), (b, c), (c, d), (d, a), (a, c), (b, d))]
    style: r.MeshStyle = {"show_edges": True, "line_width": 1.5, "smooth": False}
    camera = r.Camera(direction=(0.55, -1.3, 0.75), up=(0.0, 0.0, 1.0), zoom=1.0)

    def panel(
        first: np.ndarray, second: np.ndarray, edge: np.ndarray, color: str, title: str
    ) -> r.Panel:
        return r.Panel(
            [
                r.Lines(outline, color=r.GREY, width=1.5),
                r.Mesh(vertices, first[None], color=r.LIGHT_GREEN, **style),
                r.Mesh(vertices, second[None], color=r.GREEN, **style),
                r.Lines([edge], color=color, width=9.0),
                r.Lines([np.stack([kernel[0], v]) for v in vertices], color=r.ORANGE, width=1.0),
                r.Points(kernel, color=r.ORANGE, size=16),
            ],
            title=title,
        )

    return r.Figure(
        [
            panel(
                np.array([0, 1, 2]),
                np.array([0, 2, 3]),
                np.stack([a, c]),
                r.RED,
                "Reflex edge: a dent",
            ),
            panel(
                np.array([0, 1, 3]),
                np.array([1, 2, 3]),
                np.stack([b, d]),
                r.BLUE,
                "After a 2-2 flip: convex",
            ),
        ],
        camera=camera,
        panel_size=(560, 460),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = ffhull_2d()
    fig.savefig(OUT / "pc8-growth.svg", transparent=True)
    plt.close(fig)
    r.render_figure(flip_figure(), OUT / "pc8-flip.webp")


if __name__ == "__main__":
    main()
