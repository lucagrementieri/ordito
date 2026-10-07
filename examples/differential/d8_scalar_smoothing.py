from __future__ import annotations

from typing import Any

from examples import _render as r
from examples._meta import Meta

META = Meta(
    id="D8",
    title="Smoothing a noisy scalar field",
    summary="""
    A smooth function on a square (a ramp plus a wave, drawn as a height field), corrupted by noise,
    and three ways to recover it.
    [`filter_scalar_laplacian`][ordito.smoothing.filter_scalar_laplacian] repeatedly averages each
    value with its one-ring. The other two solve `(M + alpha Q) u = M f`: stay close to the noisy
    field `f` in the mass-weighted norm while keeping a smoothness energy `u^T Q u` small. With the
    biharmonic energy from [`k_harmonic`][ordito.energies.k_harmonic], `Q` assumes the field is flat
    across the boundary and bends the ramp there; the Hessian energy from
    [`hessian_energy`][ordito.energies.hessian_energy] has natural boundary conditions and no
    penalty on linear functions, so the ramp keeps its slope up to the rim.
    [`curved_hessian_energy`][ordito.energies.curved_hessian_energy] is its counterpart for a curved
    surface.
    """,
    notes="""
    The spike at one corner of the Hessian-smoothed square is the noise of that corner vertex
    surviving: it belongs to a single triangle, so the energy constrains it weakly.
    """,
    credits=(
        ("libigl 712", "https://libigl.github.io/tutorial/#data-smoothing"),
        ("PyMeshLab: filters", "https://pymeshlab.readthedocs.io/en/latest/filter_list.html"),
    ),
)


def run(device: str) -> dict[str, Any]:
    # --8<-- [start:code]
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(81, 81), extents=(2.0, 2.0), device=device)
    x, y, _ = vertices.numpy().T.astype(np.float64)
    clean = 1.5 * x + np.cos(2.5 * y)
    noisy = clean + np.random.default_rng(0).normal(0.0, 0.25, clean.shape)

    averaged = od.smoothing.filter_scalar_laplacian(
        wp.array(noisy, dtype=wp.float32, device=device), vertices, faces, iterations=30
    )

    # Minimise |u - f|^2_M + alpha u^T Q u, i.e. solve (M + alpha Q) u = M f, for two energies Q.
    # M is the diagonal lumped mass matrix, so M f is a per-vertex product.
    lumped = od.laplacian.mass_matrix_entries(vertices, faces, dtype=wp.float64)
    mass = od.typing.bsr_diag(lumped)
    mass_f = wp.array(lumped.numpy() * noisy, dtype=wp.float64, device=device)
    laplacian = od.laplacian.cotmatrix(vertices, faces, dtype=wp.float64)
    energies = {
        "biharmonic": od.energies.k_harmonic(laplacian, lumped, k=2),
        "hessian": od.energies.hessian_energy(vertices, faces),
    }
    smoothed = {}
    for name, q in energies.items():
        system = od.typing.bsr_axpy(od.typing.bsr_copy(q), od.typing.bsr_copy(mass), alpha=1e-3)
        u = wp.zeros_like(mass_f)
        od.linalg.solve_spd(system, mass_f, u)
        smoothed[name] = u.numpy()

    rim = (np.abs(x) > 0.95) | (np.abs(y) > 0.95)
    for name, u in {"1-ring average": averaged.numpy(), **smoothed}.items():
        error = u - clean
        print(
            f"{name}: RMS error {np.sqrt(np.mean(error**2)):.3f} overall, "
            f"{np.sqrt(np.mean(error[rim] ** 2)):.3f} along the boundary"
        )
    # --8<-- [end:code]
    return {
        "vertices": vertices.numpy(),
        "faces": faces.numpy().reshape(-1, 3),
        "clean": clean,
        "noisy": noisy,
        "averaged": averaged.numpy(),
        **smoothed,
    }


def figure(result: dict[str, Any]) -> r.Figure:
    base, faces = result["vertices"], result["faces"]
    clim = (float(result["clean"].min()), float(result["clean"].max()))
    panels = []
    for key, title in (
        ("clean", "Clean"),
        ("noisy", "Noisy"),
        ("averaged", "1-ring averaging"),
        ("biharmonic", "Biharmonic energy"),
        ("hessian", "Hessian energy"),
    ):
        vertices = base.copy()
        vertices[:, 2] = 0.25 * result[key]
        panels.append(
            r.Panel(
                [
                    r.Mesh(
                        vertices,
                        faces,
                        scalars=result[key],
                        cmap=r.striped("viridis", 16),
                        clim=clim,
                    )
                ],
                title=title,
            )
        )
    return r.Figure(
        panels,
        camera=r.Camera(direction=(0.8, -1.2, 1.0), up=(0.0, 0.0, 1.0), zoom=1.05),
        panel_size=(480, 440),
    )
