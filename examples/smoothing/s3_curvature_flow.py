from __future__ import annotations

from typing import Any

from examples import _render as r
from examples import data
from examples._meta import Meta

META = Meta(
    id="S3",
    title="Mean-curvature flow, plain and conformalized",
    summary="""
    Each pass of [`filter_implicit_fairing`][ordito.smoothing.filter_implicit_fairing] moves every
    vertex along its mean-curvature normal by one implicit (backward Euler) step of
    ``(M - lamb L) V' = M V``. Rebuilding the cotangent Laplacian ``L`` from the current shape
    every pass is mean-curvature flow: the fur-like bumps vanish first, then the thin ears shrink
    fastest and pinch. With ``conformalized=True`` the stiffness stays the input's and only the
    mass follows the surface (Kazhdan, Solomon and Ben-Chen 2012): each pass solves a system as
    well conditioned as the first, so a long flow stays cheap. The rims of the scan's holes in the
    base are held fixed. Surface areas come from
    [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas].
    """,
    notes="""
    Neither variant keeps the triangles well shaped: on the bunny the ears still shrink to slivers
    under the conformalized flow, and both outputs end with triangles a tiny fraction of the
    input's. What differs is the linear system. The plain flow rebuilds its cotangent weights from
    those slivers, so each further pass needs far more solver iterations; the conformalized flow
    keeps the input's weights, so it does not. Pinching off thin parts is a property of
    mean-curvature flow itself. The conformalized flow keeps one stiffness for the whole call, so
    it is run as one call with several passes rather than as repeated calls.
    """,
    credits=(("libigl 205", "https://libigl.github.io/tutorial/#laplacian"),),
)


def run(device: str) -> dict[str, list[Any]]:
    # --8<-- [start:code]
    import ordito as od
    from examples import data

    vertices, faces = data.load("bunny", device)

    # Plain flow: each call rebuilds the Laplacian from the shape it is given.
    plain = [vertices]
    for _ in range(3):
        plain.append(
            od.smoothing.filter_implicit_fairing(plain[-1], faces, lamb=1e-4, iterations=1)
        )

    # Conformalized flow: one call, the input's Laplacian kept for every pass.
    conformal = [vertices] + [
        od.smoothing.filter_implicit_fairing(
            vertices, faces, lamb=1e-4, iterations=passes, conformalized=True
        )
        for passes in (1, 3, 6)
    ]
    for name, frames in (("plain", plain), ("conformalized", conformal)):
        areas = [od.triangles.face_normals_and_areas(v, faces)[1].numpy() for v in frames]
        print(
            f"{name:>13}: area {' -> '.join(f'{a.sum():.4f}' for a in areas)}, "
            f"smallest triangle {areas[-1].min() / areas[0].min():.2g} of the input's"
        )
    # --8<-- [end:code]
    return {"plain": [v.numpy() for v in plain], "conformal": [v.numpy() for v in conformal]}


def figure(result: dict[str, list[Any]]) -> r.Figure:
    _, faces = data.arrays("bunny")
    panels = [
        r.Panel([r.Mesh(v, faces, color=r.GREEN)], title=f"Plain, step {k}" if k else "Input")
        for k, v in enumerate(result["plain"])
    ]
    panels += [
        r.Panel(
            [r.Mesh(v, faces, color=r.GREEN)],
            title=f"Conformalized, {passes} pass{'es' if passes > 1 else ''}"
            if passes
            else "Input",
        )
        for passes, v in zip((0, 1, 3, 6), result["conformal"], strict=True)
    ]
    return r.Figure(panels, ncols=4, camera=data.camera("bunny"), panel_size=(520, 500))
