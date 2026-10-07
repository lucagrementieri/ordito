from __future__ import annotations

from typing import Any

import numpy as np

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="PC2",
    title="Surface sampling: uniform, Poisson disk and blue noise",
    summary="""
    [`sample_surface`][ordito.sample.sample_surface] draws points uniformly by area, which leaves
    clumps and gaps; [`sample_surface_poisson_disk`][ordito.sample.sample_surface_poisson_disk]
    draws five times as many and eliminates the most crowded ones until the requested count
    remains, so the same number of points covers the surface more evenly;
    [`sample_surface_blue_noise`][ordito.sample.sample_surface_blue_noise] instead takes a
    radius and guarantees that no two samples are closer than it. The histogram of each sample's
    distance to its nearest neighbour
    ([`nearest_neighbor_distance`][ordito.neighbors.nearest_neighbor_distance]) shows the
    difference: the uniform draw spreads down to zero, the elimination pulls most gaps towards
    the typical spacing, and blue noise has a hard floor at its radius.
    """,
    credits=(
        (
            "Open3D: mesh sampling",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
        ("libigl 810", "https://libigl.github.io/tutorial/#blue-noise-sampling"),
        (
            "PyMeshLab: Poisson-disk sampling",
            "https://pymeshlab.readthedocs.io/en/latest/filter_list.html",
        ),
    ),
    notes="""
    The Poisson-disk histogram keeps a small tail of near-coincident pairs, about 2 % of the
    samples closer than a tenth of the typical spacing. Weighted sample elimination should remove
    the most crowded points first, so that tail is unexpected; use the blue-noise sampler when a
    hard minimum spacing matters.
    """,
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    uniform, _ = od.sample.sample_surface(vertices, faces, 4000, seed=1)
    poisson, _ = od.sample.sample_surface_poisson_disk(vertices, faces, 4000, seed=1)
    blue, _ = od.sample.sample_surface_blue_noise(vertices, faces, radius=0.003, seed=1)

    gap_uniform = od.neighbors.nearest_neighbor_distance(uniform).numpy()
    gap_poisson = od.neighbors.nearest_neighbor_distance(poisson).numpy()
    gap_blue = od.neighbors.nearest_neighbor_distance(blue).numpy()
    for name, gap in (("uniform", gap_uniform), ("Poisson", gap_poisson), ("blue", gap_blue)):
        median, smallest = np.median(gap), gap.min()
        print(f"{name:>8}: {len(gap)} samples, gap median {median:.5f}, min {smallest:.5f}")
    # --8<-- [end:code]
    return {
        "uniform": uniform.numpy(),
        "poisson": poisson.numpy(),
        "gap_uniform": gap_uniform,
        "gap_poisson": gap_poisson,
        "blue": blue.numpy(),
        "gap_blue": gap_blue,
    }


def figure(result: dict[str, Any]) -> r.Figure:
    vertices, faces = data.arrays("bunny")
    mesh = r.Mesh(vertices, faces, color=r.GREY)

    def draw_histogram(ax: Any) -> None:
        bins = np.linspace(0.0, 0.006, 49)
        for key, color, label in (
            ("gap_uniform", r.BLUE, "uniform"),
            ("gap_poisson", r.GREEN, "Poisson disk"),
            ("gap_blue", r.ORANGE, "blue noise"),
        ):
            ax.hist(result[key], bins=bins, color=color, histtype="step", lw=2, label=label)
        ax.set_xlabel("distance to the nearest other sample")
        legend = ax.legend(frameon=False)
        for text in legend.get_texts():
            text.set_color(r.TEXT)

    return r.Figure(
        [
            r.Panel([mesh, r.Points(result["uniform"], color=r.BLUE, size=6)], title="Uniform"),
            r.Panel(
                [mesh, r.Points(result["poisson"], color=r.GREEN, size=6)], title="Poisson disk"
            ),
            r.Panel([mesh, r.Points(result["blue"], color=r.ORANGE, size=6)], title="Blue noise"),
            r.Plot(draw_histogram, title="Nearest-neighbour distances", aspect=""),
        ],
        camera=data.camera("bunny"),
        ncols=4,
        panel_size=(560, 520),
    )
