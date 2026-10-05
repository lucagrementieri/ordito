"""
warp.fem integrands and helpers for the adaptive screened-Poisson backend.

These live in ``kernels/`` because they are Warp kernel-DSL objects (``@fem.integrand`` /
``@wp.func``): the weak-form assembly and the refinement oracle used by
[`screened_poisson`][ordito.reconstruction.screened_poisson]'s ``method="adaptive"`` path. The
Python-scope orchestration (grid build, spaces, integrate, conjugate gradient) lives in the private
``_screened_poisson_adaptive`` / ``_extract_poisson_surface_fem`` helpers in
``ordito/reconstruction.py``.

The whole solve runs in **index space**: the finest grid spacing is ``1`` and the domain spans
``[0, 2**depth]`` per axis, so the screening-vs-gradient balance matches the dense backend's
index-space calibration (see the dense kernels in ``ordito/kernels/reconstruction.py``).
"""

import warp as wp
import warp.fem as fem

wp.set_module_options({"enable_backward": False})


@wp.func
def world_to_index(p: wp.vec3, lower: wp.vec3, scale: wp.float32) -> wp.vec3:
    # Map a world point into the index-space grid frame ``[0, 2**depth]``.
    return (p - lower) * scale


@wp.func
def refinement_oracle(
    xyz: wp.vec3, grid: wp.uint64, pts: wp.array[wp.vec3], r: wp.float32, falloff: wp.float32
):
    # NB: no return annotation — fem.ImplicitField builds its argument struct from this function's
    # annotations and rejects a "return" entry (matching warp's own refinement-field examples).
    # Refinement value for a query point: 0 (finest) within ``r`` of a sample, ramping to 1
    # (coarsest) by ``r + falloff``. Never negative, so ``adaptive_nanogrid_from_field`` carves
    # no voxels and the whole cube stays covered (required by the extraction lattice).
    max_r = r + falloff
    query = wp.hash_grid_query(grid, xyz, max_r)
    j = wp.int32(0)
    best = max_r
    while wp.hash_grid_query_next(query, j):
        d = wp.length(pts[j] - xyz)
        if d < best:
            best = d
    return wp.clamp((best - r) / falloff, 0.0, 1.0)


@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    # Stiffness / Poisson operator: the weak Laplacian ``integral(grad(u) . grad(v))``.
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def screening_form(s: fem.Sample, u: fem.Field, v: fem.Field, screen: float):
    # Point-measure screening term, assembled over a PicQuadrature: ``screen * sum_p w_p u(p)v(p)``.
    return screen * u(s) * v(s)


@fem.integrand
def source_form(s: fem.Sample, v: fem.Field, normals: wp.array[wp.vec3]):
    # Right-hand side ``integral(V . grad(v))`` as point sources over a PicQuadrature; the
    # quadrature measures carry the per-sample weight, so ``normals`` are unit directions.
    return wp.dot(normals[s.qp_index], fem.grad(v, s))


@fem.integrand
def sample_field(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    positions: wp.array[wp.vec3],
    out_values: wp.array[wp.float32],
):
    # Evaluate the solved field at arbitrary index-space positions via point location.
    i = s.qp_index
    lookup = fem.lookup(domain, positions[i])
    if lookup.element_index != fem.NULL_ELEMENT_INDEX:
        out_values[i] = u(lookup)


@wp.kernel(enable_backward=False)
def lattice_positions(
    step: wp.float32, res: wp.int32, hi: wp.float32, out_positions: wp.array[wp.vec3]
) -> None:
    # Index-space positions of the dense ``res**3`` extraction lattice, row-major ``(i, j, k)``.
    # Coordinates are clamped a hair inside ``[0, hi]`` so the outermost lattice planes still land
    # in a cell instead of returning a NULL lookup.
    i, j, k = wp.tid()
    x = wp.clamp(wp.float32(i) * step, 1e-3, hi)
    y = wp.clamp(wp.float32(j) * step, 1e-3, hi)
    z = wp.clamp(wp.float32(k) * step, 1e-3, hi)
    out_positions[(i * res + j) * res + k] = wp.vec3(x, y, z)
