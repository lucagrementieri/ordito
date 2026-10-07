from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="R3",
    title="Subdivision: midpoint and Loop",
    summary="""
    Both schemes split every triangle into four. [`subdivide`][ordito.remesh.subdivide] puts the
    new vertices at the edge midpoints, so the surface keeps its facets however often it is
    applied; [`subdivide_loop`][ordito.remesh.subdivide_loop] moves old and new vertices by
    Loop's stencils, and repeated passes converge to a smooth limit surface. The input is the
    bunny decimated to a few hundred faces by
    [`quadric_decimate`][ordito.remesh.quadric_decimate].
    """,
    credits=(
        ("libigl 711", "https://libigl.github.io/tutorial/#subdivision-surfaces"),
        (
            "Open3D: mesh subdivision",
            "https://www.open3d.org/docs/release/tutorial/geometry/mesh.html",
        ),
        ("PyVista: subdivide", "https://docs.pyvista.org/examples/01-filter/subdivide"),
        ("PyMeshLab", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    coarse = od.remesh.quadric_decimate(vertices, faces, target_faces=500, feature_angle=180.0)

    midpoint, loop = [coarse], [coarse]
    for _ in range(3):
        midpoint.append(od.remesh.subdivide(*midpoint[-1]))
        loop.append(od.remesh.subdivide_loop(*loop[-1]))
    print("faces per level:", [f.shape[0] // 3 for _, f in loop])
    # --8<-- [end:code]
    as_np = lambda mesh: (mesh[0].numpy(), mesh[1].numpy().reshape(-1, 3))  # noqa: E731
    return {"midpoint": [as_np(m) for m in midpoint], "loop": [as_np(m) for m in loop]}


def figure(result: dict[str, Any]) -> r.Figure:
    v0, f0 = result["loop"][0]
    panels = [r.Panel([r.Mesh(v0, f0, show_edges=True, smooth=False)], title="Coarse input")]
    for key, label in (("midpoint", "Midpoint"), ("loop", "Loop")):
        for level in (1, 3):
            v, f = result[key][level]
            panels.append(
                r.Panel(
                    [r.Mesh(v, f, show_edges=level == 1, smooth=key == "loop", line_width=0.5)],
                    title=f"{label}, {level} level{'s' if level > 1 else ''}",
                )
            )
    return r.Figure(panels, camera=data.camera("bunny"), ncols=5, panel_size=(520, 500))
