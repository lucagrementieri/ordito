from __future__ import annotations

from typing import Any

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="C1",
    title="Primitive gallery",
    summary="""
    Sixteen builders from [`ordito.creation`][ordito.creation], each returning a vertex buffer and
    a flat face buffer on the requested device:
    [`box`][ordito.creation.box], [`icosphere`][ordito.creation.icosphere],
    [`uv_sphere`][ordito.creation.uv_sphere], [`capsule`][ordito.creation.capsule],
    [`cone`][ordito.creation.cone], [`cylinder`][ordito.creation.cylinder],
    [`annulus`][ordito.creation.annulus], [`torus`][ordito.creation.torus], the Platonic solids
    ([`tetrahedron`][ordito.creation.tetrahedron], [`octahedron`][ordito.creation.octahedron],
    [`dodecahedron`][ordito.creation.dodecahedron], [`icosahedron`][ordito.creation.icosahedron]),
    the superquadrics [`super_ellipsoid`][ordito.creation.super_ellipsoid] and
    [`super_toroid`][ordito.creation.super_toroid], and the two open patches
    [`sphere_cap`][ordito.creation.sphere_cap] and [`grid`][ordito.creation.grid].
    [`is_watertight`][ordito.validation.is_watertight] confirms which of them are closed.
    """,
    credits=(
        ("trimesh: creation", "https://trimesh.org/trimesh.creation.html"),
        (
            "Open3D: mesh primitives",
            "https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html",
        ),
        (
            "PyVista: geometric objects",
            "https://docs.pyvista.org/examples/00-load/create_geometric_objects",
        ),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import math

    import ordito as od

    c = od.creation
    shapes = {
        "box": c.box(extents=(1.6, 1.2, 1.0), device=device),
        "icosphere": c.icosphere(2, device=device),
        "uv_sphere": c.uv_sphere(count=(12, 20), device=device),
        "capsule": c.capsule(height=1.2, radius=0.5, count=(16, 16), device=device),
        "cone": c.cone(0.8, 1.6, sections=24, device=device),
        "cylinder": c.cylinder(0.7, 1.6, sections=24, device=device),
        "annulus": c.annulus(0.5, 1.0, height=0.6, sections=32, device=device),
        "torus": c.torus(1.0, 0.35, 32, 16, device=device),
        "tetrahedron": c.tetrahedron(device=device),
        "octahedron": c.octahedron(device=device),
        "dodecahedron": c.dodecahedron(device=device),
        "icosahedron": c.icosahedron(device=device),
        "super_ellipsoid": c.super_ellipsoid(0.3, 0.6, device=device),
        "super_toroid": c.super_toroid(0.3, 1.0, device=device),
        "sphere_cap": c.sphere_cap(math.pi / 3, subdivisions=3, device=device),
        "grid": c.grid(count=(9, 9), extents=(2.0, 2.0), device=device),
    }
    closed = [name for name, (v, f) in shapes.items() if od.validation.is_watertight(v, f)]
    print(f"{len(closed)} of {len(shapes)} are watertight")
    print("open:", ", ".join(name for name in shapes if name not in closed))
    # --8<-- [end:code]
    return {name: (v.numpy(), f.numpy()) for name, (v, f) in shapes.items()}


_TETRA_VIEW = r.Camera(direction=(1.0, 0.3, 0.25), up=(0.0, 0.0, 1.0), zoom=0.95)


def figure(result: dict[str, Any]) -> r.Figure:
    camera = r.Camera(direction=(1.0, -1.3, 0.9), up=(0.0, 0.0, 1.0), zoom=0.95)
    panels = [
        r.Panel(
            [r.Mesh(v, f, show_edges=True, smooth=False, line_width=0.8)],
            title=name,
            camera=_TETRA_VIEW if name == "tetrahedron" else None,
        )
        for name, (v, f) in result.items()
    ]
    return r.Figure(panels, ncols=4, camera=camera, link_bounds=False, panel_size=(420, 380))
