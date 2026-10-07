from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="S3",
    title="Mean-curvature flow",
    summary="""
    Each step of [`filter_implicit_fairing`][ordito.smoothing.filter_implicit_fairing] moves every
    vertex along its mean-curvature normal by solving one implicit (backward Euler) system with
    the cotangent Laplacian and the mass matrix, rebuilt from the current shape. Repeating it is
    mean-curvature flow: the fur-like bumps vanish in the first step, then the thin ears shrink
    fastest and melt into the head while the body rounds off. The rims of the scan's holes in the
    base are held fixed, so the base stays where it is. Surface areas come from
    [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas].
    """,
    notes="""
    The flow is continued only until the ears are about to vanish. Beyond that point their
    triangles collapse to zero area, the cotangent system rebuilt from them is badly conditioned,
    and further steps become slow and can throw single vertices far off the surface. Pinching
    off thin parts is a known property of mean-curvature flow; the conformalized variant (which
    keeps the Laplacian of the input) avoids it but is not what this filter computes.
    """,
    credits=(("libigl 205", "https://libigl.github.io/tutorial/#laplacian"),),
)


def run(device: str) -> dict[int, Any]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)
    frames = [vertices]
    for _ in range(3):
        frames.append(
            od.smoothing.filter_implicit_fairing(frames[-1], faces, lamb=1e-4, iterations=1)
        )
    for step, positions in enumerate(frames):
        _, areas = od.triangles.face_normals_and_areas(positions, faces)
        print(f"step {step}: surface area {areas.numpy().sum():.4f}")
    # --8<-- [end:code]
    return {step: v.numpy() for step, v in enumerate(frames)}


def figure(result: dict[int, Any]) -> r.Figure:
    _, faces = data.arrays("bunny")
    panels = [
        r.Panel([r.Mesh(v, faces, color=r.GREEN)], title=f"Step {step}" if step else "Input")
        for step, v in result.items()
    ]
    return r.Figure(panels, camera=data.camera("bunny"), panel_size=(520, 500))
