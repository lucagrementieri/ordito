from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="RR7",
    title="From a bare scan: normals, then Poisson or ball pivoting",
    summary="""
    A scanner gives points without normals, so the pipeline starts with
    [`estimate_normals`][ordito.points.estimate_normals]. The two reconstructions then answer
    different questions. [`screened_poisson`][ordito.reconstruction.screened_poisson] is closed
    whatever the sampling looks like, but its vertices are new points.
    [`ball_pivoting`][ordito.reconstruction.ball_pivoting] keeps the input points as vertices,
    and is only as good as the sampling: a random sample clumps in places and leaves gaps the
    ball falls through, while the same number of points spread evenly by
    [`sample_surface_blue_noise`][ordito.sample.sample_surface_blue_noise] leaves far fewer holes
    (rims in red).
    """,
    credits=(
        (
            "Open3D: surface reconstruction",
            "https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html",
        ),
    ),
    notes="""
    **Orienting normals on a real scan.** A plane fit fixes each normal's line, not its sign. The
    default sign only agrees between neighbours, so a reconstruction asks for
    `orientation="centroid"`: every normal points away from the cloud's centroid, which is right
    for a closed, roughly convex object and wrong in deep concavities ([PC3](points.md#pc3) shows
    where).
    `camera_location=wp.vec3(...)` points every normal toward the scanner, the right choice for a
    single-view scan; `orient_reference=wp.vec3(...)` points them along one direction (`+z` for
    terrain). `k=16` neighbours is a common default: raise it for noisy data, lower it where the
    surface has fine detail.

    **Choosing between the two.** Use `screened_poisson` for noisy or unevenly sampled clouds and
    whenever the surface must be closed (a volume, a print, inside/outside tests); each extra
    `depth` level gives about four times the faces at about eight times the cost
    ([RR1](#rr1)). Use `ball_pivoting` when output vertices must be input points (to carry
    per-point colours or labels) and the cloud is dense and evenly spaced. If the sampling is
    uneven, thin it out evenly first rather than hoping a larger ball bridges the gaps.
    """,
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import warp as wp

    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    random_points, _ = od.sample.sample_surface(vertices, faces, 20_000, seed=0)
    even_points, _ = od.sample.sample_surface_blue_noise(vertices, faces, radius=0.00135, seed=0)

    def normals_of(points: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
        neighbours, _ = od.neighbors.query_nearest(points, points, k=16)
        return od.points.estimate_normals(points, neighbours, orientation="centroid")

    poisson = od.reconstruction.screened_poisson(random_points, normals_of(random_points), depth=7)
    print(f"screened Poisson: {poisson[1].shape[0] // 3} faces")
    print(f"watertight: {od.validation.is_watertight(*poisson)}")

    pivoted = {}
    for name, points in (("random", random_points), ("even", even_points)):
        pivoted[name] = od.reconstruction.ball_pivoting(points, normals_of(points))
        holes = od.boundary.boundary_loops(*pivoted[name])
        print(f"ball pivoting, {name} sample of {points.shape[0]}: {len(holes)} holes")
    # --8<-- [end:code]
    out: dict[str, Any] = {
        "random_points": random_points.numpy(),
        "poisson": (poisson[0].numpy(), poisson[1].numpy().reshape(-1, 3)),
    }
    for name, (v, f) in pivoted.items():
        loops = od.boundary.boundary_loops(v, f)
        out[name] = (v.numpy(), f.numpy().reshape(-1, 3), [loop.numpy() for loop in loops])
    return out


def figure(result: dict[str, Any]) -> r.Figure:
    def pivot_panel(key: str, title: str) -> r.Panel:
        v, f, loops = result[key]
        rims = [v[loop] for loop in loops]
        return r.Panel([r.Mesh(v, f), r.Lines(rims, closed=True, width=2.5)], title=title)

    pv_, pf = result["poisson"]
    return r.Figure(
        [
            r.Panel([r.Points(result["random_points"], color=r.GREY, size=3)], title="Random scan"),
            r.Panel([r.Mesh(pv_, pf)], title="Screened Poisson"),
            pivot_panel("random", "Ball pivoting, random sample"),
            pivot_panel("even", "Ball pivoting, even sample"),
        ],
        camera=data.camera("bunny"),
        ncols=4,
        panel_size=(560, 520),
    )
