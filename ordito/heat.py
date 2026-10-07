"""
Heat-diffusion methods on triangle meshes.

Three solvers that share one idea: diffuse a quantity over the surface for a short time ``t``, then
recover the answer from the *direction* of the resulting field rather than its magnitude. Short-time
heat flow approximates the geodesic kernel, so a single sparse solve carries information a
combinatorial shortest-path search would have to walk edge by edge. Each solver differs only in what
it diffuses and how it reads the result back:

- **Geodesic distance.** [`heat_geodesic`][ordito.heat.heat_geodesic] diffuses a scalar indicator
  from source vertices, normalizes its gradient, and integrates that unit field back with a Poisson
  solve. The heat method of Crane et al. (``igl::heat_geodesics``,
  ``potpourri3d.MeshHeatMethodDistanceSolver``).
- **Signed distance to curves.** [`heat_signed_distance`][ordito.heat.heat_signed_distance]
  diffuses the normals of a set of oriented curves, then solves a Poisson problem against that field
  to get a *signed* distance whose zero set is the curves (``potpourri3d.MeshSignedHeatSolver``).
- **Vector-valued transport.**
  [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors],
  [`extend_scalar`][ordito.heat.extend_scalar] and [`log_map`][ordito.heat.log_map] diffuse
  *tangent vectors* through the connection Laplacian, which transports each vector into its
  neighbour's frame before differencing (``potpourri3d.MeshVectorHeatSolver``).

The three are one module because two of them cannot be separated: the vector solvers import
[`heat_operators`][ordito.heat.heat_operators] and [`heat_geodesic`][ordito.heat.heat_geodesic],
[`VectorHeatOperators`][ordito.heat.VectorHeatOperators] embeds the scalar method's operator tuple,
and [`log_map`][ordito.heat.log_map]'s radius *is* ``heat_geodesic``'s answer.

All three run in ``float64``, because the diffused field decays exponentially and underflows
``float32``. They run on either device.

Every solver takes the mesh as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh] or as its
``vertices`` and ``faces``. The `Trimesh` is where the method's state lives: its
[`HeatSolver`][ordito.heat.HeatSolver] per diffusion time holds the operators and the sparse
Cholesky factorizations that make later solves on the same mesh cheap, for as long as the mesh
lives (or until [`Trimesh.release_factorizations`][ordito.mesh.Trimesh.release_factorizations]).
Nothing is kept between calls of the ``vertices, faces`` form.

The operators these solvers assemble are **not** here -- the cotangent and connection Laplacians
live in [`ordito.laplacian`][ordito.laplacian], tangent frames in
[`ordito.tangent_space`][ordito.tangent_space], and the batched conjugate-gradient machinery in
[`ordito.linalg`][ordito.linalg]. This module is the three algorithms only.
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from typing import cast, overload

import warp as wp
import warp.optim.linear as wpl

import ordito as od
import ordito.linalg as twl
import ordito.typing as odt
from ordito import _launch
from ordito._device import require_same_device
from ordito.constants import TILE_1D
from ordito.kernels import heat as kernel_heat
from ordito.kernels import linalg as kernel_linalg
from ordito.kernels import reduce as kernel_reduce
from ordito.laplacian import (
    MeshOperatorPattern,
    connection_laplacian,
    cotmatrix,
    cotmatrix_entries,
    cotmatrix_entries_intrinsic,
    mass_matrix_entries,
    mesh_operator_pattern,
    mollify_intrinsic,
)
from ordito.mesh import Trimesh, mesh_arguments
from ordito.tangent_space import vertex_tangent_frames
from ordito.triangles import face_normals_and_areas

_LOGGER = logging.getLogger(__name__)

# Every solve here is a conjugate-gradient one and they all converge at the same tolerance; it was
# spelled three times when these were three modules.
_CG_TOLERANCE = 1e-8

# The heat solves' settle rule. Every diffusion here (``heat_geodesic``'s heat solve,
# ``extend_scalar``, the stacked solves of ``transport_tangent_vectors`` and ``log_map``, and
# ``diffuse_tangent_field``) calls ``linalg.solve_spd_settled`` with these three constants: a check
# every ``_HEAT_CHECK_ROUNDS`` rounds, the per-vertex relative change below which a check counts as
# settled, and the rounds past the last newly reached vertex after which the solve stops anyway.
#
# Why a settle test and not a residual tolerance: the diffused sources fall by a near-constant
# factor per ring of vertices, so the far field is hundreds of orders of magnitude below the peak.
# ``float64`` holds it, but a conjugate-gradient iterate after ``k`` rounds is a degree-``k``
# polynomial in the operator applied to the sources -- exactly zero more than ``k`` rings away --
# and the residual converges long before ``k`` reaches the far side of the mesh. So each solve runs
# at a zero tolerance until every entry has settled relative to itself: one continuous iteration
# checked on the device, stopping once no entry is newly reached and none moved by
# ``_HEAT_CHANGE_TOLERANCE``, or ``_HEAT_SETTLE_ROUNDS`` rounds after the last vertex was reached
# for an entry that never settles because it cancels toward zero (a transported vector on the cut
# locus). No readback.
#
# **One iteration, not warm-restarted chunks**: restarting conjugate gradient breaks conjugacy,
# and the vector solves then never pass the settle test. **Jacobi, never the Jacobi-Chebyshev
# polynomial**, although a polynomial round reaches a dozen rings where a Jacobi round reaches one:
# obtuse triangles make the heat system's off-diagonal entries positive, and the polynomial's
# interval does not cover the far field's decay (wrong on ``bunny``: the distance 0.9 of its range
# off igl's, the scalar extension divergent).
#
# Probed on continuous Jacobi-CG against the converged field (1 024 rounds), on the scalar and
# vector systems of spheres from 2.5 k to 164 k vertices, both bunnies and the uniform and graded
# saddles: a 16-round check fires at or after the first round within ``1e-6`` of the converged
# field on every one, where a 64-round check overshoots by up to 160 rounds. The change collapses
# by six or more orders of magnitude once the field settles, so the tolerance is not a tuning knob.
# The rounds a field needs past full reach do not grow with the mesh -- that is the heat system's
# own conditioning, fixed by ``t = h ** 2`` -- and measured at 80-150, so 192 covers every one; the
# graded saddle never settles (its cut-locus entries oscillate) and stops on that bound.
_HEAT_CHECK_ROUNDS = 16
_HEAT_CHANGE_TOLERANCE = 1e-6
_HEAT_SETTLE_ROUNDS = 192


HeatOperators = tuple[
    odt.BsrMatrix[wp.float64],
    wpl.LinearOperator,
    odt.BsrMatrix[wp.float64],
    odt.BsrMatrix[wp.float64],
    wpl.LinearOperator,
    odt.Array2dFloat32,
    wp.array[wp.vec3],
    wp.array[wp.float32],
]
"""What [`heat_operators`][ordito.heat.heat_operators] returns for the heat method's
solves."""


def heat_operators(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    t: float | None = None,
    *,
    use_robust: bool = False,
    cot_entries: odt.Array2dFloat | None = None,
) -> HeatOperators:
    """
    Assemble the source-independent operators the heat method solves against.

    Every quantity here depends on the mesh alone, not on the source set, so a caller computing
    distance from many different sources on one mesh builds these once: a
    [`Trimesh`][ordito.mesh.Trimesh] passed to [`heat_geodesic`][ordito.heat.heat_geodesic] keeps
    them (and the factorizations of their systems) in its [`HeatSolver`][ordito.heat.HeatSolver].
    That is the split ``potpourri3d.MeshHeatMethodDistanceSolver`` and ``igl::heat_geodesics``
    expose as a stateful solver object.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    t
        Diffusion time. When ``None``, defaults to the squared mean edge length (the
        ``igl::heat_geodesics`` default).
    use_robust
        Build the Laplacian from *mollified* edge lengths
        ([`mollify_intrinsic`][ordito.laplacian.mollify_intrinsic]) instead of straight from vertex
        positions. Costs one extra pass and two host readbacks, and is what lets the method run on a
        mesh with degenerate triangles at all. It leaves a clean mesh's operator unchanged.

        This is mollification **only**, not the intrinsic Delaunay retriangulation that
        [`robust_laplacian`][ordito.laplacian.robust_laplacian] also does by default (and that
        ``potpourri3d``'s identically-named flag includes). The reason is structural rather than a
        shortcut: flipping changes which faces exist, and the gradient and divergence stages below
        integrate over faces. Swapping in an operator built on a different triangulation while those
        stages still use the original one is not a cheap approximation, it is inconsistent — so a
        fully intrinsic heat method needs intrinsic *mass*, *gradient* and *divergence* as well. Use
        [`robust_laplacian`][ordito.laplacian.robust_laplacian] directly where only the operator
        matters (smoothing, parametrization, spectral work).
    cot_entries
        ``(n_faces, 3)`` optional precomputed
        [`cotmatrix_entries`][ordito.laplacian.cotmatrix_entries]. Depends on the mesh alone, so a
        caller assembling these operators at several diffusion times reuses one table -- and
        [`Trimesh.cotmatrix_entries`][ordito.mesh.Trimesh.cotmatrix_entries] has it cached. Both
        precisions are accepted: the assembly casts to the matrix dtype in a single build.

    Returns
    -------
    heat_system : warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` ``M - t * L`` in ``float64``, the heat-diffusion system.
    heat_preconditioner : ``warp.optim.linear.LinearOperator``
        Jacobi preconditioner for ``heat_system``.
    laplacian : warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` ``float64`` cotangent stiffness matrix ``L`` (igl sign
        convention, so ``-L`` is positive semi-definite).
    poisson_system : warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` ``-L``, the positive-semi-definite Poisson operator.
    poisson_preconditioner : ``warp.optim.linear.LinearOperator``
        Jacobi-Chebyshev polynomial preconditioner for ``poisson_system``
        ([`chebyshev_preconditioner`][ordito.linalg.chebyshev_preconditioner]), built on its first
        apply.
    cot_entries : odt.Array2dFloat32
        ``(n_faces, 3)`` per-face half-cotangent weights, reused by the divergence. Always
        ``float32`` regardless of the ``cot_entries`` precision passed in or built internally.
    face_normals : wp.array[wp.vec3]
        ``(n_faces,)`` unit normals, one per face.
    face_areas : wp.array[wp.float32]
        ``(n_faces,)`` areas, one per face.

    Notes
    -----
    The Poisson operator and both preconditioners are here because they satisfy this
    function's own contract — they depend on the mesh alone — so
    [`heat_geodesic`][ordito.heat.heat_geodesic] need not rebuild all three on every call, which
    is unnecessary work whenever the mesh is unchanged across several solves. The solver
    **state** is not in the tuple: [`solve_spd`][ordito.linalg.solve_spd] keeps one per operator
    itself, owning its own buffers, so solving against these operators again (as a
    [`HeatSolver`][ordito.heat.HeatSolver] does) also replays the solves' recorded loops rather
    than recording them again -- and, as with the polynomial's vectors below, two solves against
    one operator must run on one stream. A `HeatSolver` also factors a system it solves a second
    time, which no preconditioner competes with. The Poisson solve is the long one -- ``-L`` is the
    ill-conditioned operator here, where the heat system's mass term keeps it close to diagonal --
    so it takes the polynomial preconditioner and the heat system keeps Jacobi, which a
    well-conditioned solve of a few tens of iterations cannot beat. The polynomial holds working
    vectors of its own, so two solves against one ``poisson_preconditioner`` must run on one
    stream, as every caller here does.

    !!! note
        ``solve_spd`` **warm-starts from whatever the solution buffer already holds**, so reusing
        one buffer across unrelated right-hand sides carries over the previous solution as the
        initial guess. And this tuple's *third* field is the raw (singular) Laplacian, not the
        Poisson system — handing it a right-hand side runs CG to its iteration cap.

    Raises
    ------
    ValueError
        If both ``cot_entries`` and ``use_robust`` are given: ``use_robust`` exists to build that
        very table from mollified edge lengths, so the two ask for different weights.
    RuntimeError
        If ``vertices``, ``faces`` and ``cot_entries`` are not all on one device.

    See Also
    --------
    [`heat_geodesic`][ordito.heat.heat_geodesic]
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries]
    [`Trimesh.heat_operators`][ordito.mesh.Trimesh.heat_operators]
    """
    require_same_device(vertices=vertices, faces=faces, cot_entries=cot_entries)
    if cot_entries is not None and use_robust:
        raise ValueError(
            "cot_entries and use_robust are mutually exclusive: use_robust rebuilds the "
            "half-cotangent table from mollified edge lengths."
        )
    return _heat_operators(vertices, faces, t, use_robust=use_robust, cot_entries=cot_entries)[0]


@overload
def heat_geodesic(
    mesh: Trimesh,
    sources: wp.array[wp.int32],
    /,
    *,
    t: float | None = None,
    use_robust: bool = False,
) -> wp.array[wp.float64]: ...
@overload
def heat_geodesic(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    /,
    t: float | None = None,
    *,
    use_robust: bool = False,
) -> wp.array[wp.float64]: ...
def heat_geodesic(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | None = None,
    sources: wp.array[wp.int32] | None = None,
    t: float | None = None,
    *,
    vertices: wp.array[wp.vec3] | None = None,
    use_robust: bool = False,
) -> wp.array[wp.float64]:
    """
    Approximate geodesic distance to the nearest source vertex (Crane et al. heat method).

    Diffuses heat from the source vertices for a short time ``t``, normalizes the resulting
    gradient into a unit vector field pointing away from the sources, and integrates it back into a
    distance field by solving a Poisson problem. Both solves are sparse, symmetric positive
    (semi-)definite systems handled on-device by conjugate gradient. The result is an
    *approximation* of the true geodesic distance (typically a few percent error), matching
    ``igl::heat_geodesics``.

    The computation runs in ``float64``: the diffused heat decays exponentially away from the
    source and would underflow ``float32``, collapsing the far field.

    Takes the mesh as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh] or as its ``vertices`` and
    ``faces``. The `Trimesh` form keeps what the method builds for the mesh -- the
    [`heat_operators`][ordito.heat.heat_operators] and, once a system is solved a second time or
    the heat solve needs one, its sparse Cholesky factorization (see
    [`HeatSolver`][ordito.heat.HeatSolver]) -- so distance from many source sets on one mesh pays
    for them once. The ``vertices, faces`` form builds them for the call and drops them.

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``vertices`` and ``faces``.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    sources
        ``(n_sources,)`` source vertex indices. The returned distance is measured to
        the nearest source and is zero at the source set.
    t
        Diffusion time. When ``None``, defaults to the squared mean edge length (the
        ``igl::heat_geodesics`` default), which balances accuracy and smoothing.
    use_robust
        Forwarded to [`heat_operators`][ordito.heat.heat_operators]: build the Laplacian
        from mollified edge lengths, which is what makes the solves survive degenerate triangles.
        ``potpourri3d.MeshHeatMethodDistanceSolver`` has the same flag and defaults it to ``True``;
        this defaults to ``False`` so the plain call stays exactly ``igl::heat_geodesics``.

    Returns
    -------
    wp.array[wp.float64]
        ``(n_vertices,)`` geodesic distance field on the mesh's device.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``sources`` are not all on one device.

    Notes
    -----
    The ``vertices, faces`` form logs at ``INFO`` on the ``ordito.heat`` logger when it built a
    factorization it is about to drop (the heat solve's verification failed on an ill-conditioned
    mesh), since the `Trimesh` form would have kept it for the next call.

    See Also
    --------
    [`HeatSolver`][ordito.heat.HeatSolver]
    [`heat_operators`][ordito.heat.heat_operators]
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`mean_unique_edge_length`][ordito.edges.mean_unique_edge_length]
    [`marching_triangles`][ordito.intersection.marching_triangles]
    """
    bound, owned, arguments = mesh_arguments(
        "heat_geodesic", mesh, vertices, faces, (sources, t), 1
    )
    return _heat_geodesic(
        bound,
        owned,
        cast("wp.array[wp.int32]", arguments[0]),
        cast("float | None", arguments[1]),
        use_robust,
    )


def _heat_geodesic(
    mesh: Trimesh, owned: bool, sources: wp.array[wp.int32], t: float | None, use_robust: bool
) -> wp.array[wp.float64]:
    """``heat_geodesic`` on a `Trimesh`; ``owned`` is whether the caller passed it."""
    vertices, faces = mesh.vertices, mesh.faces
    require_same_device(vertices=vertices, faces=faces, sources=sources)
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3

    if n_vertices == 0 or n_faces == 0 or sources.size == 0:
        return _launch.zeros(n_vertices, dtype=wp.float64, device=device)

    solver = mesh.heat_solver(t, use_robust=use_robust)

    # Heat solve: (M - t L) u = u0, with u0 the source indicator, run until every vertex's heat has
    # converged relative to its own size (the settle rule at ``_HEAT_CHECK_ROUNDS``) -- not to a
    # residual tolerance, which the far field sits hundreds of orders of magnitude below. Neumann on
    # a boundary, as geometry-central (``potpourri3d``) and MeshLab take it.
    # ``igl::heat_geodesics_solve`` averages it with the solution pinned to zero on the boundary
    # instead, and against exact polyhedral geodesics (``igl.exact_geodesic``) that average is the
    # less accurate of the two: equal on a hemisphere, and 1.15 % against 0.93 % mean error (4.9 %
    # against 3.1 % worst) of the distance range on a half torus.
    u0 = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_heat.seed_source_indicator, dim=sources.size, inputs=[sources, u0], device=device
    )

    heat = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    solver.diffuse(u0, heat)
    distance = _distance_from_heat(mesh, solver, sources, heat)
    _log_discarded("heat_geodesic", owned, solver)
    return distance


# --------------------------------------------------------------------------------------
# The signed heat method: signed distance to a set of oriented curves
# --------------------------------------------------------------------------------------


@overload
def heat_signed_distance(
    mesh: Trimesh,
    curve_vertices: wp.array[wp.int32],
    /,
    *,
    curve_offsets: wp.array[wp.int32] | None = None,
    t: float | None = None,
    closed: bool = True,
    level_set_constraint: str = "zero_set",
) -> wp.array[wp.float64]: ...
@overload
def heat_signed_distance(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    curve_vertices: wp.array[wp.int32],
    /,
    curve_offsets: wp.array[wp.int32] | None = None,
    t: float | None = None,
    *,
    closed: bool = True,
    level_set_constraint: str = "zero_set",
) -> wp.array[wp.float64]: ...
def heat_signed_distance(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | None = None,
    curve_vertices: wp.array[wp.int32] | None = None,
    curve_offsets: wp.array[wp.int32] | None = None,
    t: float | None = None,
    *,
    vertices: wp.array[wp.vec3] | None = None,
    closed: bool = True,
    level_set_constraint: str = "zero_set",
) -> wp.array[wp.float64]:
    """
    Signed distance from every vertex to a set of oriented curves.

    The result is positive inside the region a counter-clockwise curve encloses and negative outside
    it, with the curve itself at (or near) zero — geometry-central's convention. Orientation is what
    fixes the sign: reversing a curve's vertex order negates the whole field.

    Three stages, mirroring ``potpourri3d.MeshSignedHeatSolver.compute_distance``:

    1. each curve segment splats its normal onto its two endpoints, weighted by half its length;
    2. the connection Laplacian diffuses that tangent field for a short time ``t`` and it is
       normalized, giving a unit field that approximates the signed distance's gradient;
    3. a Poisson solve integrates the field back into a scalar.

    Takes the mesh as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh], which keeps the
    [`vector_heat_operators`][ordito.heat.vector_heat_operators] and the factorizations later calls
    reuse (see [`heat_geodesic`][ordito.heat.heat_geodesic]), or as its ``vertices`` and
    ``faces``.

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``vertices`` and ``faces``.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    curve_vertices
        ``(n_curve_vertices,)`` vertex indices along the curves, packed one curve after another.
        Consecutive entries should be adjacent on the mesh; nothing breaks if they are not, but the
        source is then splatted along a chord rather than along the surface.
    curve_offsets
        ``(n_curves + 1,)`` CSR bounds into ``curve_vertices``. When ``None`` the whole buffer
        is treated as a single curve.
    t
        Diffusion time; defaults to the squared mean edge length. Larger values smooth the field.
    closed
        Whether each curve closes back on its first vertex (adding one more segment). Signed
        distance is most meaningful for closed curves; an open curve gives a field whose sign flips
        across it but which has no consistent far-field meaning.
    level_set_constraint
        ``"zero_set"`` pins the curve vertices to exactly zero and solves the Poisson problem on the
        rest, through the machinery behind
        [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed].
        ``"none"`` solves unconstrained and then shifts the field so the curve's mean is zero,
        which leaves the level set slightly off the curve but is cheaper and pins nothing. Both are
        modes ``potpourri3d`` offers under the same names.

    Returns
    -------
    wp.array[wp.float64]
        ``(n_vertices,)`` signed distance field on the mesh's device.

    Raises
    ------
    ValueError
        If ``level_set_constraint`` is not ``"zero_set"`` or ``"none"``.
    RuntimeError
        If ``vertices``, ``faces``, ``curve_vertices`` and ``curve_offsets`` are not all on one
        device.

    Notes
    -----
    The ``vertices, faces`` form logs at ``INFO`` on the ``ordito.heat`` logger when it built a
    factorization it is about to drop, as [`heat_geodesic`][ordito.heat.heat_geodesic] does.

    See Also
    --------
    [`heat_geodesic`][ordito.heat.heat_geodesic]
    [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors]
    [`homology_generators`][ordito.homology.homology_generators]
    [`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh]
    """
    bound, owned, arguments = mesh_arguments(
        "heat_signed_distance", mesh, vertices, faces, (curve_vertices, curve_offsets, t), 1
    )
    return _heat_signed_distance(
        bound,
        owned,
        cast("wp.array[wp.int32]", arguments[0]),
        cast("wp.array[wp.int32] | None", arguments[1]),
        cast("float | None", arguments[2]),
        closed=closed,
        level_set_constraint=level_set_constraint,
    )


def _heat_signed_distance(
    mesh: Trimesh,
    owned: bool,
    curve_vertices: wp.array[wp.int32],
    curve_offsets: wp.array[wp.int32] | None,
    t: float | None,
    *,
    closed: bool,
    level_set_constraint: str,
) -> wp.array[wp.float64]:
    """``heat_signed_distance`` on a `Trimesh`; ``owned`` is whether the caller passed it."""
    vertices, faces = mesh.vertices, mesh.faces
    require_same_device(
        vertices=vertices, faces=faces, curve_vertices=curve_vertices, curve_offsets=curve_offsets
    )
    if level_set_constraint not in ("zero_set", "none"):
        raise ValueError(
            f'level_set_constraint must be "zero_set" or "none", got {level_set_constraint!r}'
        )
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    if n_vertices == 0 or n_faces == 0 or curve_vertices.size == 0:
        return _launch.zeros(n_vertices, dtype=wp.float64, device=device)

    solver = mesh.heat_solver(t)
    _, scalar, frames, _ = solver.vector_operators
    basis_x, basis_y, vertex_normals = frames
    poisson_system = scalar[3]
    cot_entries, face_normals = scalar[5], scalar[6]

    # Stage 1: splat each segment's normal onto its endpoints, one thread per curve entry, each
    # finding its own curve in the offsets on the device.
    source = _launch.zeros(n_vertices, dtype=wp.vec2d, device=device)
    _launch.launch(
        kernel_heat.splat_curve_normals,
        dim=curve_vertices.size,
        inputs=[
            vertices,
            curve_vertices,
            curve_offsets,
            wp.int32(1 if curve_offsets is None else 0),
            wp.int32(1 if closed else 0),
            vertex_normals,
            basis_x,
            basis_y,
            source,
        ],
        device=device,
    )

    # Stage 2: diffuse the tangent field; only its direction is kept. No absolute floor may decide
    # which vectors vanished -- this field carries the mesh's scale, and one zeroes most of it on a
    # mesh not near unit scale (confirmed: 111 of 162 vertices at a 1e-6 scale).
    diffused = _launch.zeros(n_vertices, dtype=wp.vec2d, device=device)
    solver.diffuse_tangent(source, diffused)
    # The divergence kernel normalizes each corner without underflow, zero only where the field is
    # exactly zero: it is converged per vertex (the settle rule at ``_HEAT_CHECK_ROUNDS``), so its
    # far field is a direction however small.

    # Stage 3: integrate the unit field back into a scalar with a Poisson solve. The cotangent
    # weights and face normals come from the same bundle, so the Poisson stage and the diffusion
    # cannot drift apart.
    divergence = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_heat.vertex_field_divergence,
        dim=n_faces,
        inputs=[vertices, faces, face_normals, diffused, basis_x, basis_y, cot_entries, divergence],
        device=device,
    )

    if level_set_constraint == "zero_set":
        field = _solve_poisson_zero_set(
            poisson_system, divergence, curve_vertices, n_vertices, device
        )
    else:
        field = _solve_poisson_shifted(solver, divergence, curve_vertices, vertices)
    _log_discarded("heat_signed_distance", owned, solver)
    return field


def _solve_poisson_zero_set(
    operator: odt.BsrMatrix[wp.float64],
    divergence: wp.array[wp.float64],
    curve_vertices: wp.array[wp.int32],
    n_vertices: int,
    device: wp.DeviceLike,
) -> wp.array[wp.float64]:
    """
    Solve the Poisson problem with the curve pinned to zero.

    ``linalg.min_quad_with_fixed`` minimizes ``0.5 x' Q x`` and has no place for a linear term, so
    the pieces underneath it are used directly: the free-free block comes from
    [`assemble_interior_system`][ordito.linalg.assemble_interior_system] (whose own right-hand side
    is zero here, the pinned values being zero) and the divergence is compacted into it.
    """
    fixed_mask = od.array.indices_to_mask(curve_vertices, n_vertices)
    free_map, n_free = twl.free_partition(fixed_mask)
    if n_free == 0:
        return _launch.zeros(n_vertices, dtype=wp.float64, device=device)

    # No pinned-value right-hand side to assemble: the curve is pinned to zero, so ``-Q_ub bc``
    # vanishes and the extraction is asked for none. The system's one right-hand side is the
    # divergence, compacted to the free rows by the extraction's own row pass (``load``).
    # ``divergence`` is already the -div right-hand side the -L operator takes.
    no_values = odt.empty_2d((0, n_vertices), wp.float64, device=device)
    operator_uu, rhs = twl.assemble_interior_system(
        operator,
        fixed_mask,
        free_map,
        no_values,
        n_free,
        load=odt.as_array2d(divergence.reshape((1, n_vertices)), wp.float64),
    )

    solution = _launch.zeros((1, n_free), dtype=wp.float64, device=device)
    # The same Poisson operator as the unpinned solve, less the curve's rows, so the same
    # polynomial preconditioner; see ``heat_operators``' Notes.
    twl.solve_spd_columns(
        operator_uu,
        rhs,
        odt.as_array2d(solution, wp.float64),
        tol=_CG_TOLERANCE,
        preconditioner="chebyshev",
    )
    field = _launch.empty(n_vertices, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_heat.gather_free_solution,
        dim=n_vertices,
        inputs=[fixed_mask, free_map, solution, field],
        device=device,
    )
    return field


def _solve_poisson_shifted(
    solver: HeatSolver,
    divergence: wp.array[wp.float64],
    curve_vertices: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
) -> wp.array[wp.float64]:
    """
    Solve the Poisson problem unconstrained, then shift so the curve's mean value is zero.

    The unconstrained system is singular up to a constant (a pure Neumann problem), which conjugate
    gradient handles while the right-hand side is consistent; the shift afterwards picks that
    constant, and putting the curve at zero is the choice that makes the result a distance.
    """
    device = vertices.device
    n_vertices = vertices.size
    # ``divergence`` is already the -div right-hand side the -L operator takes.
    field = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    solver.solve_poisson(divergence, field)
    # ``heat_geodesic``'s device-side shift onto the sources, without its orientation: one launch
    # sized to the curve for the mean, one to apply it, and nothing read back.
    n_sources = curve_vertices.size
    sums = _launch.zeros(2, dtype=wp.float64, device=device)
    _launch.launch_tiled(
        kernel_heat.source_and_global_sums,
        dim=[kernel_reduce.blocks_1d(n_sources)],
        inputs=[field, curve_vertices],
        outputs=[sums],
        block_dim=TILE_1D,
        device=device,
    )
    _launch.launch(
        kernel_heat.shift_and_orient,
        dim=n_vertices,
        inputs=[sums, wp.int32(n_sources), wp.int32(0), field],
        device=device,
    )
    return field


# --------------------------------------------------------------------------------------
# The vector heat method: transport, scalar extension and the logarithmic map
# --------------------------------------------------------------------------------------

# Below this fraction of the diffused *magnitudes at the same vertex*, a transported direction
# cannot be told from the round-off left where the transported copies cancel (the cut locus). The
# ratio is 1 where the copies agree -- a vector's length never exceeds the heat of its magnitude --
# so the test is local and scale-free, and a far vertex is resolved however small its field.
# Round-off at an exact cancellation measured 1.4e-7 of the local magnitude (``cave_cube``'s
# antipodal corner: the transport angles are ``float32``), genuine directions 0.36 and more, so the
# threshold sits
# three decades above the round-off. The *only* thing it drives is the mask
# ``transport_tangent_vectors`` returns alongside its vectors -- no value is zeroed by it, so
# flagging a marginal vertex costs the caller nothing.
_RESOLVED_FRACTION = 1e-4


VectorHeatOperators = tuple[
    odt.BsrMatrix[wp.mat22d],
    HeatOperators,
    tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]],
    wpl.LinearOperator,
]
"""What [`vector_heat_operators`][ordito.heat.vector_heat_operators] returns: the vector
heat system, the scalar [`heat_operators`][ordito.heat.heat_operators], the frames, and the
vector system's own Jacobi preconditioner."""


def vector_heat_operators(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    t: float | None = None,
    *,
    scalar_operators: HeatOperators | None = None,
    frames: tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]] | None = None,
) -> VectorHeatOperators:
    """
    Assemble everything the vector-valued solvers need before their solves.

    Four pieces, none of which depends on a source:

    1. the **vector heat system** ``M + t * L_connection``, whose ``2 x 2`` blocks act on tangent
       vectors ([`connection_laplacian`][ordito.laplacian.connection_laplacian]);
    2. the scalar [`heat_operators`][ordito.heat.heat_operators], for the magnitude
       extension and the distance field the log map needs;
    3. the [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames] every 2D component
       is measured in;
    4. the vector system's own Jacobi preconditioner.

    A [`Trimesh`][ordito.mesh.Trimesh] keeps the result
    ([`Trimesh.vector_heat_operators`][ordito.mesh.Trimesh.vector_heat_operators], or its
    [`HeatSolver`][ordito.heat.HeatSolver] at another ``t``), so a caller running many transports
    or log maps against one mesh pays for the assembly once. That is the split
    ``potpourri3d.MeshVectorHeatSolver`` gets from being an object.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    t
        Diffusion time for both the vector and the scalar systems. When ``None``, defaults to the
        squared mean edge length.
    scalar_operators
        Optional prebuilt [`heat_operators`][ordito.heat.heat_operators] bundle to place
        in the second field instead of assembling one. **Must have been built at the same ``t``**,
        which nothing here can check: it is the caller's half of the shared-timestep contract the
        Notes below describe. [`Trimesh.heat_operators`][ordito.mesh.Trimesh.heat_operators]
        caches one at this function's own default.
    frames
        ``(n_vertices,)`` each, optional prebuilt
        [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames] as
        ``(basis_x, basis_y, normal)`` -- the gauge, which depends on the mesh alone and not on
        ``t``. [`Trimesh.vertex_tangent_frames`][ordito.mesh.Trimesh.vertex_tangent_frames] caches
        it.

    Returns
    -------
    vector_system : warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` ``M + t * L_connection`` in ``float64`` with ``wp.mat22d``
        blocks.
    scalar : tuple
        The [`heat_operators`][ordito.heat.heat_operators] bundle for the same ``t``.
    frames : tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]]
        ``(n_vertices,)`` each, ``(basis_x, basis_y, normal)`` per vertex.
    preconditioner : ``warp.optim.linear.LinearOperator``
        Jacobi preconditioner for ``vector_system``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``frames`` are not all on one device.

    Notes
    -----
    The vector and scalar systems must share ``t``: [`log_map`][ordito.heat.log_map]'s
    radius is asserted to *be* the [`heat_geodesic`][ordito.heat.heat_geodesic] distance,
    so two diffusion times would split a quantity that is supposed to be one number. That is why
    ``scalar_operators`` is the one argument here that cannot be validated.

    See Also
    --------
    [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors]
    [`log_map`][ordito.heat.log_map]
    [`heat_signed_distance`][ordito.heat.heat_signed_distance]
    [`Trimesh.vector_heat_operators`][ordito.mesh.Trimesh.vector_heat_operators]
    """
    require_same_device(vertices=vertices, faces=faces, frames=frames)
    device = vertices.device
    n_vertices = vertices.size
    # One sparsity for both operators: the connection Laplacian's is the cotangent Laplacian's.
    pattern = mesh_operator_pattern(faces, n_vertices) if faces.size > 0 else None
    connection = connection_laplacian(vertices, faces, pattern=pattern)
    edge_sums = None
    if t is None:
        # Shares the scalar solver's timestep convention -- the unique-edge mean, matching
        # ``igl::heat_geodesics``. The two solvers must agree: ``log_map``'s radius is asserted to
        # *be* the ``heat_geodesic`` distance, so giving them different diffusion times would split
        # a quantity that is supposed to be one number. They do by construction: the mean is read
        # off the connection Laplacian's sparsity, which is the cotangent Laplacian's, twelve
        # triplets per face and nothing pruned -- and the scalar bundle built here takes these
        # very sums. Both systems square it on the device, so nothing is read back.
        edge_sums = _edge_length_sums(vertices, connection)

    if scalar_operators is None:
        scalar_operators, mass = _heat_operators(
            vertices, faces, t, edge_sums=edge_sums, pattern=pattern
        )
    else:
        mass = mass_matrix_entries(vertices, faces, dtype=wp.float64)
    # ``M + t L_connection`` over the connection Laplacian's own pattern, as the scalar system is
    # built over the cotangent one's (``heat_operators``); the kernel forms each vertex's mass block
    # from the scalar mass as it reads it.
    vector_values = _launch.empty_like(connection.values)
    _launch.launch(
        kernel_linalg.SHIFTED_SYSTEM_VALUES[wp.mat22d],
        dim=int(connection.nrow),
        inputs=[
            connection.offsets,
            connection.columns,
            connection.values,
            mass,
            wp.float64(1.0 if t is None else t),
            edge_sums,
            wp.int32(1 if t is None else 0),
            wp.int32(0),
        ],
        outputs=[vector_values, None],
        device=device,
    )
    vector_system = twl.bsr_with_values(connection, vector_values)
    if frames is None:
        frames = vertex_tangent_frames(vertices, faces)
    preconditioner = twl.jacobi_preconditioner(vector_system)
    return vector_system, scalar_operators, frames, preconditioner


def _heat_operators(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    t: float | None,
    *,
    use_robust: bool = False,
    cot_entries: odt.Array2dFloat | None = None,
    edge_sums: wp.array[wp.float64] | None = None,
    pattern: MeshOperatorPattern | None = None,
) -> tuple[HeatOperators, wp.array[wp.float64]]:
    """
    ``heat_operators``' assembly, returning the ``float64`` lumped mass it built alongside.

    ``vector_heat_operators`` reuses the mass for its own system, and hands over the edge-length
    sums it already reduced for the default timestep as ``edge_sums``: the connection Laplacian's
    sparsity is the cotangent Laplacian's, so the sums are the ones this would reduce -- and the
    ``pattern`` it built them over, so the sparsity is built once for both.
    """
    # Per-face half-cotangent weights (float32, O(1) and safe) reused for both the Laplacian and
    # the divergence. The cotangent stiffness follows the igl convention (negative diagonal, so
    # ``-L`` is positive semi-definite) but is assembled here in float64.
    if use_robust:
        # Mollified lengths: one global constant added to every edge so no triangle is degenerate.
        # The gradient and divergence stages below still use the extrinsic positions, so this makes
        # the *solves* robust rather than turning the whole method intrinsic.
        lengths, _ = mollify_intrinsic(vertices, faces)
        cot_entries = cotmatrix_entries_intrinsic(lengths)
    elif cot_entries is None:
        cot_entries = cotmatrix_entries(vertices, faces)
    # ``cotmatrix`` casts the shared float32 half-cotangent weights to float64 and assembles the
    # operator natively in a single build, avoiding a recast rebuild (see cotmatrix's kernel note).
    laplacian = cotmatrix(
        vertices, faces, cot_entries=cot_entries, dtype=wp.float64, pattern=pattern
    )
    if t is None and edge_sums is None:
        # The unique-edge average, which is what ``igl::heat_geodesics`` uses for its timestep --
        # read off the Laplacian's own sparsity, which already holds the unique edges, and left on
        # the device for the system's assembly to square.
        edge_sums = _edge_length_sums(vertices, laplacian)

    # The divergence kernel this bundle feeds (``kernels/heat.py::unit_gradient_divergence``) is
    # hardcoded ``wp.array2d[wp.float32]`` -- ``cotmatrix`` above accepts either precision because
    # it casts internally, but this tuple's own ``cot_entries`` field is documented and used
    # downstream as float32 only, so a caller-supplied float64 table must be narrowed before it is
    # returned rather than passed through at whatever precision it arrived in.
    if cot_entries.dtype is not wp.float32:
        cot_entries = odt.as_array2d(od.array.astype(cot_entries, wp.float32), wp.float32)

    # Face normals / areas (float32) for the gradient; the lumped mass is built natively in float64
    # by ``mass_matrix_entries``.
    normals, areas = face_normals_and_areas(vertices, faces)
    mass = mass_matrix_entries(vertices, faces, dtype=wp.float64)

    # Heat system (M - t L) and Poisson operator ``-L``, both over the Laplacian's own pattern
    # (which stores every referenced vertex's diagonal) and written in one pass
    # (``kernels/linalg.shifted_system_values``), so all three operators share one pattern. The two
    # preconditioners are mesh-only, so they belong here rather than in every solve. See Notes.
    heat_values = _launch.empty_like(laplacian.values)
    poisson_values = _launch.empty_like(laplacian.values)
    _launch.launch(
        kernel_linalg.SHIFTED_SYSTEM_VALUES[wp.float64],
        dim=int(laplacian.nrow),
        inputs=[
            laplacian.offsets,
            laplacian.columns,
            laplacian.values,
            mass,
            wp.float64(-1.0 if t is None else -t),
            edge_sums if t is None else None,
            wp.int32(1 if t is None else 0),
            wp.int32(1),
        ],
        outputs=[heat_values, poisson_values],
        device=vertices.device,
    )
    heat_system = twl.bsr_with_values(laplacian, heat_values)
    poisson_system = twl.bsr_with_values(laplacian, poisson_values)
    operators = (
        heat_system,
        twl.jacobi_preconditioner(heat_system),
        laplacian,
        poisson_system,
        twl.chebyshev_preconditioner(poisson_system),
        # Narrowed to float32 by the block above, whichever precision it arrived in.
        cast(odt.Array2dFloat32, cot_entries),
        normals,
        areas,
    )
    return operators, mass


def _edge_length_sums(
    vertices: wp.array[wp.vec3], operator: odt.SparseMatrix
) -> wp.array[wp.float64]:
    """
    Sum and count of the unique-edge lengths, read off an operator with one entry per edge.

    The strict upper triangle of the heat method's Laplacians is the mesh's unique edge set, so the
    default timestep costs one launch over an operator that already exists, where
    [`mean_unique_edge_length`][ordito.edges.mean_unique_edge_length] would re-sort every edge of
    the mesh to recover the same set. The two sums stay on the device, where
    ``kernels/linalg.shifted_system_values`` squares their mean -- ``0`` for a mesh with no edges,
    as that function returns.
    """
    device = vertices.device
    n_rows = int(operator.nrow)
    sum_and_count = _launch.zeros(2, dtype=wp.float64, device=device)
    if n_rows > 0:
        _launch.launch_tiled(
            kernel_heat.upper_edge_length_sum_and_count,
            dim=[kernel_reduce.blocks_1d(n_rows)],
            inputs=[operator.offsets, operator.columns, vertices, sum_and_count],
            block_dim=TILE_1D,
            device=device,
        )
    return sum_and_count


@overload
def extend_scalar(
    mesh: Trimesh,
    sources: wp.array[wp.int32],
    values: wp.array[wp.float64],
    /,
    *,
    t: float | None = None,
) -> wp.array[wp.float64]: ...
@overload
def extend_scalar(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    values: wp.array[wp.float64],
    /,
    t: float | None = None,
) -> wp.array[wp.float64]: ...
def extend_scalar(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | None = None,
    sources: wp.array[wp.int32] | wp.array[wp.float64] | None = None,
    values: wp.array[wp.float64] | None = None,
    t: float | None = None,
    *,
    vertices: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.float64]:
    """
    Extend values from a few source vertices over the whole surface by nearest-source interpolation.

    Diffuses the values and an indicator of where they came from for the same short time, then
    divides one by the other. The ratio is what makes the result interpolate rather than decay: both
    numerator and denominator fall off away from the sources at the same rate, so their quotient
    stays close to the value of the nearest source, and blends smoothly where two sources compete.
    Matches ``potpourri3d.MeshVectorHeatSolver.extend_scalar``.

    Takes the mesh as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh], which keeps the operators
    and factorizations later calls reuse (see [`heat_geodesic`][ordito.heat.heat_geodesic]), or as
    its ``vertices`` and ``faces``.

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``vertices`` and ``faces``.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    sources
        ``(n_sources,)`` source vertex indices.
    values
        ``(n_sources,)`` value carried by each source.
    t
        Diffusion time; defaults to the squared mean edge length.

    Returns
    -------
    wp.array[wp.float64]
        ``(n_vertices,)`` extended field on the mesh's device.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``sources`` and ``values`` are not all on one device.

    Notes
    -----
    The ``vertices, faces`` form logs at ``INFO`` on the ``ordito.heat`` logger when it built a
    factorization it is about to drop, as [`heat_geodesic`][ordito.heat.heat_geodesic] does.

    See Also
    --------
    [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors]
    [`heat_geodesic`][ordito.heat.heat_geodesic]
    """
    bound, owned, arguments = mesh_arguments(
        "extend_scalar", mesh, vertices, faces, (sources, values, t), 2
    )
    return _extend_scalar(
        bound,
        owned,
        cast("wp.array[wp.int32]", arguments[0]),
        cast("wp.array[wp.float64]", arguments[1]),
        cast("float | None", arguments[2]),
    )


def _extend_scalar(
    mesh: Trimesh,
    owned: bool,
    sources: wp.array[wp.int32],
    values: wp.array[wp.float64],
    t: float | None,
) -> wp.array[wp.float64]:
    """``extend_scalar`` on a `Trimesh`; ``owned`` is whether the caller passed it."""
    vertices, faces = mesh.vertices, mesh.faces
    require_same_device(vertices=vertices, faces=faces, sources=sources, values=values)
    device = vertices.device
    n_vertices = vertices.size
    n_sources = sources.size
    if n_vertices == 0 or faces.size == 0 or n_sources == 0:
        return _launch.zeros(n_vertices, dtype=wp.float64, device=device)

    solver = mesh.heat_solver(t)
    # The indicator and the weighted values diffuse through the same operator, so they are one
    # batched two-column solve rather than two independent ones, which shares the launches and
    # converges on the worse-behaved of the two columns.
    rhs = odt.as_array2d(
        _launch.zeros((2, n_vertices), dtype=wp.float64, device=device), wp.float64
    )
    _launch.launch(
        kernel_heat.seed_source_scalars,
        dim=n_sources,
        inputs=[sources, values, rhs[0], rhs[1]],
        device=device,
    )
    diffused = odt.as_array2d(
        _launch.zeros((2, n_vertices), dtype=wp.float64, device=device), wp.float64
    )
    solver.diffuse(rhs, diffused)
    # Converged per vertex, so only an exactly zero indicator -- a component no source reaches --
    # has no value to extend (``divide_nonzero``).
    extended = _launch.empty(n_vertices, dtype=wp.float64, device=device)
    _launch.map(
        kernel_heat.divide_nonzero,
        odt.as_dense(diffused[1]),
        odt.as_dense(diffused[0]),
        out=extended,
    )
    _log_discarded("extend_scalar", owned, solver)
    return extended


@overload
def transport_tangent_vectors(
    mesh: Trimesh,
    sources: wp.array[wp.int32],
    vectors: wp.array[wp.vec2],
    /,
    *,
    t: float | None = None,
) -> tuple[wp.array[wp.vec2], wp.array[wp.bool]]: ...
@overload
def transport_tangent_vectors(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    vectors: wp.array[wp.vec2],
    /,
    t: float | None = None,
) -> tuple[wp.array[wp.vec2], wp.array[wp.bool]]: ...
def transport_tangent_vectors(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | None = None,
    sources: wp.array[wp.int32] | wp.array[wp.vec2] | None = None,
    vectors: wp.array[wp.vec2] | None = None,
    t: float | None = None,
    *,
    vertices: wp.array[wp.vec3] | None = None,
) -> tuple[wp.array[wp.vec2], wp.array[wp.bool]]:
    """
    Parallel-transport tangent vectors from a few source vertices to every vertex.

    Three solves, following the vector heat method: the connection Laplacian diffuses the source
    vectors (which preserves their *directions* well but smears their magnitudes), while a scalar
    extension of the source magnitudes supplies the length. The result at each vertex is the source
    vector carried along the shortest path to it. Matches
    ``potpourri3d.MeshVectorHeatSolver.transport_tangent_vectors``, which returns the vectors alone.
    Takes the mesh as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh], which keeps the operators
    and factorizations later calls reuse (see [`heat_geodesic`][ordito.heat.heat_geodesic]), or as
    its ``vertices`` and ``faces``.

    The second return exists because the vectors are not self-describing: a zero is ambiguous and a
    *non*-zero one is not always meaningful. A vertex is unresolved when the diffused direction that
    reached it is shorter than ``1e-07`` of the field's maximum -- about a decade above the
    round-off left where the transported copies cancel. Below that line a direction cannot be told
    from noise, and the mask says so rather than the value being altered: no vector here is changed
    by it. Three situations it separates:

    * **Nothing reached the vertex** -- another connected component, or short-time diffusion
      underflowing. Unresolved, and the vector is zero.
    * **The cut locus** -- several shortest paths arrive and their copies cancel. Unresolved, but
      the
      vector may still have *full length*, pointing wherever the round-off landed. This is the case
      that differs between CPU and CUDA, and the one a caller cannot otherwise detect.
    * **An ordinary vertex.** Resolved.

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``vertices`` and ``faces``.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    sources
        ``(n_sources,)`` source vertex indices.
    vectors
        ``(n_sources,)`` tangent vectors, each in *its own source vertex's* frame.
    t
        Diffusion time; defaults to the squared mean edge length.

    Returns
    -------
    transported : wp.array[wp.vec2]
        ``(n_vertices,)`` transported vectors, each in that vertex's own frame. No frames need to be
        passed in: the components come out in the canonical frames of
        [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames] by construction (see
        [`connection_laplacian`][ordito.laplacian.connection_laplacian]). A **zero** vector means
        the field vanished there: another connected component, or short-time diffusion underflowing
        before it arrived. The second is the common case on a fine mesh, because the default ``t``
        shrinks with the edge length — 39 461 of 40 962 vertices on a subdivision-6 icosphere, which
        is the method behaving as designed rather than a failure. Pass a larger ``t`` to reach
        further.
    resolved : wp.array[wp.bool]
        ``(n_vertices,)`` — ``True`` where the transported direction carries information.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``sources`` and ``vectors`` are not all on one device.

    See Also
    --------
    [`log_map`][ordito.heat.log_map]
    [`connection_laplacian`][ordito.laplacian.connection_laplacian]
    [`tangent_to_world`][ordito.heat.tangent_to_world]

    Notes
    -----
    !!! note "The direction is undefined on the cut locus"

        Where several shortest paths of equal length arrive, the copies they carry cancel, and what
        is left is round-off rather than a direction. This is not a pathological case: the corner of
        a cube shell diagonally opposite the source receives three copies 120 degrees apart whose
        sum is *exactly* zero, for every source vector and every diffusion time. What comes back is
        then decided by the arithmetic -- on CUDA enough round-off survives to be scaled up to full
        length in an arbitrary direction, while on CPU the same point can cancel to exactly zero and
        read as unreached. ``resolved`` is ``False`` on both, and is the only way to tell.

    The ``vertices, faces`` form logs at ``INFO`` on the ``ordito.heat`` logger when it built a
    factorization it is about to drop, as [`heat_geodesic`][ordito.heat.heat_geodesic] does.
    """
    bound, owned, arguments = mesh_arguments(
        "transport_tangent_vectors", mesh, vertices, faces, (sources, vectors, t), 2
    )
    return _transport_tangent_vectors(
        bound,
        owned,
        cast("wp.array[wp.int32]", arguments[0]),
        cast("wp.array[wp.vec2]", arguments[1]),
        cast("float | None", arguments[2]),
    )


def _transport_tangent_vectors(
    mesh: Trimesh,
    owned: bool,
    sources: wp.array[wp.int32],
    vectors: wp.array[wp.vec2],
    t: float | None,
) -> tuple[wp.array[wp.vec2], wp.array[wp.bool]]:
    """``transport_tangent_vectors`` on a `Trimesh`; ``owned`` is whether the caller passed it."""
    vertices, faces = mesh.vertices, mesh.faces
    require_same_device(vertices=vertices, faces=faces, sources=sources, vectors=vectors)
    device = vertices.device
    n_vertices = vertices.size
    n_sources = sources.size
    if n_vertices == 0 or faces.size == 0 or n_sources == 0:
        return (
            _launch.zeros(n_vertices, dtype=wp.vec2, device=device),
            _launch.zeros(n_vertices, dtype=wp.bool, device=device),
        )

    solver = mesh.heat_solver(t)

    # The vector field and the magnitudes' two-column extension do not interact and settle at the
    # same round, so they diffuse as one solve over the block-diagonal stack
    # ``[vector system; heat system; heat system]`` (``HeatSolver.diffuse_stacked``): one settle
    # loop in the launches of one, where two back to back paid for two. Its Jacobi diagonal and
    # narrowed values are per row, so they are the two systems' own, and its settle test reads the
    # union, which is the later of the two stops.
    rhs_field, rhs = _stacked_fields(n_vertices, 2, device)
    _launch.launch(
        kernel_heat.seed_transport_sources,
        dim=n_sources,
        inputs=[sources, vectors],
        outputs=[rhs_field, rhs[2 * n_vertices : 3 * n_vertices], rhs[3 * n_vertices :]],
        device=device,
    )
    direction, diffused = _stacked_fields(n_vertices, 2, device)
    solver.diffuse_stacked(rhs, diffused, 2)
    diffused_indicator = diffused[2 * n_vertices : 3 * n_vertices]
    diffused_magnitudes = diffused[3 * n_vertices :]

    # One map for the whole tail: the magnitude's extension, the rescale, the narrowing to the
    # field's storage precision and the resolution test all read one vertex's own data, so running
    # them apart costs extra launches and full round trips of float64 fields. Resolution is asked
    # locally, against the diffused magnitudes at the same vertex -- see the kernel func.
    transported = _launch.empty(n_vertices, dtype=wp.vec2, device=device)
    resolved = _launch.empty(n_vertices, dtype=wp.bool, device=device)
    _launch.map(
        kernel_heat.transported_and_resolved,
        direction,
        diffused_magnitudes,
        diffused_indicator,
        wp.float64(_RESOLVED_FRACTION),
        out=[transported, resolved],
    )
    _log_discarded("transport_tangent_vectors", owned, solver)
    return transported, resolved


@overload
def log_map(mesh: Trimesh, source: int, /, *, t: float | None = None) -> wp.array[wp.vec2]: ...
@overload
def log_map(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], source: int, /, t: float | None = None
) -> wp.array[wp.vec2]: ...
def log_map(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | int | None = None,
    source: int | None = None,
    t: float | None = None,
    *,
    vertices: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.vec2]:
    """
    Logarithmic map: every vertex's position in the source vertex's tangent plane.

    ``log_map(...)[v]`` is the 2D point in the *source's* frame whose length is the geodesic
    distance to ``v``, and whose direction is the initial direction of the geodesic that reaches
    ``v``. It is the inverse of the exponential map
    [`trace_from_vertex`][ordito.geodesic_walk.trace_from_vertex]
    computes, and the standard way to lay out a local coordinate patch around a point.

    Assembled from two fields that are each cheap: the distance to the source
    ([`heat_geodesic`][ordito.heat.heat_geodesic]) gives the radius, and the source's
    reference direction parallel-transported outwards gives the angle — at any vertex the angle
    between that transported direction and the outward radial direction is exactly the angle at
    which the connecting geodesic left the source, because transport along that geodesic preserves
    it. This is
    the ``VectorHeat`` strategy in ``potpourri3d.MeshVectorHeatSolver.compute_log_map``; its
    ``AffineLocal`` and ``AffineAdaptive`` strategies solve a small dense problem per vertex and are
    deliberately not ported. Takes the mesh as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh],
    which keeps the operators and factorizations later calls reuse (see
    [`heat_geodesic`][ordito.heat.heat_geodesic]), or as its ``vertices`` and ``faces``.

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``vertices`` and ``faces``.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    source
        Index of the vertex the map is centred on.
    t
        Diffusion time; defaults to the squared mean edge length. The *angles* this function
        returns are measured from the source's ``basis_x`` in
        [`Trimesh.vertex_tangent_frames`][ordito.mesh.Trimesh.vertex_tangent_frames].

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` log-map coordinates in the source vertex's frame; ``(0, 0)`` at the
        source.
        On the cut locus — the antipode of a closed surface, where geodesics from the source arrive
        from every side — there is no direction to report, and the entry keeps the correct magnitude
        with an arbitrary angle.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    The ``vertices, faces`` form logs at ``INFO`` on the ``ordito.heat`` logger when it built a
    factorization it is about to drop, as [`heat_geodesic`][ordito.heat.heat_geodesic] does.

    See Also
    --------
    [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors]
    [`trace_from_vertex`][ordito.geodesic_walk.trace_from_vertex]
    """
    bound, owned, arguments = mesh_arguments("log_map", mesh, vertices, faces, (source, t), 1)
    return _log_map(bound, owned, cast("int", arguments[0]), cast("float | None", arguments[1]))


def _log_map(mesh: Trimesh, owned: bool, source: int, t: float | None) -> wp.array[wp.vec2]:
    """``log_map`` on a `Trimesh`; ``owned`` is whether the caller passed it."""
    vertices, faces = mesh.vertices, mesh.faces
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_vertices = vertices.size
    if n_vertices == 0 or faces.size == 0:
        return _launch.zeros(n_vertices, dtype=wp.vec2, device=device)

    solver = mesh.heat_solver(t)
    _, scalar, frames, _ = solver.vector_operators
    basis_x, basis_y, _ = frames

    # The source's own reference direction transported outwards -- the "which way was x?" field
    # the angle is measured against, raw (unnormalized) like ``transport_tangent_vectors``' own
    # ``direction`` -- and ``heat_geodesic``'s heat, diffused together as one solve over
    # ``[vector system; heat system]`` for the reason ``transport_tangent_vectors`` gives.
    sources = _launch.full(1, source, dtype=wp.int32, device=device)
    rhs_field, rhs = _stacked_fields(n_vertices, 1, device)
    _launch.launch(
        kernel_heat.seed_log_map_source,
        dim=1,
        inputs=[source],
        outputs=[rhs_field, rhs[2 * n_vertices :]],
        device=device,
    )
    transported_raw, diffused = _stacked_fields(n_vertices, 1, device)
    solver.diffuse_stacked(rhs, diffused, 1)

    # Radial direction: the unit gradient of the distance field, averaged onto vertices and
    # expressed in each vertex's frame.
    distance = _distance_from_heat(mesh, solver, sources, odt.as_dense(diffused[2 * n_vertices :]))
    normals, areas = scalar[6], scalar[7]
    n_faces = faces.size // 3
    vertex_gradient = _launch.zeros(n_vertices, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_heat.scatter_unit_gradient_to_vertices,
        dim=n_faces,
        inputs=[vertices, faces, normals, areas, distance, vertex_gradient],
        device=device,
    )
    # The kernel expresses the gradient in each vertex's frame and normalizes the transported
    # reference in ``float64`` before narrowing it, which the far field needs to survive.
    logarithm = _launch.empty(n_vertices, dtype=wp.vec2, device=device)
    _launch.launch(
        kernel_heat.log_map_from_angles,
        dim=n_vertices,
        inputs=[vertex_gradient, basis_x, basis_y, transported_raw, distance, logarithm],
        device=device,
    )
    _log_discarded("log_map", owned, solver)
    return logarithm


def _distance_from_heat(
    mesh: Trimesh, solver: HeatSolver, sources: wp.array[wp.int32], heat: wp.array[wp.float64]
) -> wp.array[wp.float64]:
    """
    ``heat_geodesic`` from the diffused heat on: the divergence, the Poisson solve and the shift.

    Split out so ``log_map`` can diffuse the heat inside a stacked solve of its own.
    """
    vertices, faces = mesh.vertices, mesh.faces
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    _, _, _, _, _, cot_entries, normals, areas = solver.operators

    # Integrated divergence b = div(X) of the unit field X = -grad(u)/|grad(u)|, then a Poisson
    # solve L phi = b, i.e. (-L) phi = -b with the positive semi-definite operator. One launch: the
    # field is formed and integrated per face, so it never occupies an (n_faces,) buffer. The kernel
    # integrates ``-X`` and so accumulates ``-b`` directly: the divergence is linear in the field
    # and negation is exact, so this is the negated sum without a pass to negate it.
    neg_divergence = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_heat.unit_gradient_divergence,
        dim=n_faces,
        inputs=[vertices, faces, normals, areas, heat, cot_entries, neg_divergence],
        device=device,
    )

    phi = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    solver.solve_poisson(neg_divergence, phi)

    # Shift so the field's mean over the sources is zero, and orient it positive -- the
    # ``igl::heat_geodesics_solve`` convention, which makes a single source's distance exactly
    # zero. Both means from one reduction launch, applied by a second that reads them on the
    # device, so nothing is read back.
    n_sources = sources.size
    sums = _launch.zeros(2, dtype=wp.float64, device=device)
    _launch.launch_tiled(
        kernel_heat.source_and_global_sums,
        dim=[kernel_reduce.blocks_1d(max(n_vertices, n_sources))],
        inputs=[phi, sources],
        outputs=[sums],
        block_dim=TILE_1D,
        device=device,
    )
    _launch.launch(
        kernel_heat.shift_and_orient,
        dim=n_vertices,
        inputs=[sums, wp.int32(n_sources), wp.int32(1), phi],
        device=device,
    )
    return phi


def _stacked_fields(
    n_vertices: int, n_scalars: int, device: wp.DeviceLike
) -> tuple[wp.array[wp.vec2d], wp.array[wp.float64]]:
    """
    Allocate a zeroed vector laid out as ``block_diag``'s stack of a vector and scalar systems.

    The ``(n_vertices,)`` ``wp.vec2d`` field comes first, as its interleaved ``float64`` rows, then
    ``n_scalars`` scalar fields one after another. Returns the field and the whole stack as one
    flat ``float64`` vector over the same memory.
    """
    rows = (2 + n_scalars) * n_vertices
    storage = _launch.zeros((rows + 1) // 2, dtype=wp.vec2d, device=device)
    flat = storage.view(wp.float64).flatten()
    return odt.as_dense(storage[:n_vertices]), odt.as_dense(flat[:rows])


def tangent_to_world(
    tangent: wp.array[wp.vec2], basis_x: wp.array[wp.vec3], basis_y: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    """
    Expand per-vertex tangent vectors into 3D using their frames.

    The only way to compare a tangent field against another library's: each library measures 2D
    components from its own reference direction, but ``a * basis_x + b * basis_y`` is the same 3D
    vector either way.

    Parameters
    ----------
    tangent
        ``(n_vertices,)`` tangent vectors in each vertex's frame.
    basis_x, basis_y
        ``(n_vertices,)`` frames those components refer to, from
        [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames].

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_vertices,)`` world-space vectors on ``tangent.device``.

    Raises
    ------
    RuntimeError
        If ``tangent``, ``basis_x`` and ``basis_y`` are not all on one device.

    See Also
    --------
    [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames]
    """
    require_same_device(tangent=tangent, basis_x=basis_x, basis_y=basis_y)
    world = _launch.empty(tangent.size, dtype=wp.vec3, device=tangent.device)
    _launch.map(kernel_heat.tangent_to_world, tangent, basis_x, basis_y, out=world)
    return world


@overload
def diffuse_tangent_field(
    mesh: Trimesh, source: wp.array[wp.vec2d], /, *, t: float | None = None
) -> wp.array[wp.vec2d]: ...
@overload
def diffuse_tangent_field(
    system: odt.SparseMatrix,
    source: wp.array[wp.vec2d],
    /,
    *,
    preconditioner: wpl.LinearOperator | None = None,
) -> wp.array[wp.vec2d]: ...
def diffuse_tangent_field(
    mesh: Trimesh | odt.SparseMatrix | None = None,
    source: wp.array[wp.vec2d] | None = None,
    *,
    t: float | None = None,
    system: odt.SparseMatrix | None = None,
    preconditioner: wpl.LinearOperator | None = None,
) -> wp.array[wp.vec2d]:
    """
    Short-time diffusion of a tangent-vector field: solve ``(M + t L_connection) X = source``.

    Public because the source term is where the vector-valued methods differ from one another — a
    handful of vertices for parallel transport, a whole splatted curve for
    [`heat_signed_distance`][ordito.heat.heat_signed_distance] — while the solve is the same
    for all of them.

    Only the *directions* of the result carry meaning: magnitudes decay away from the source, and
    every caller replaces them, either with a scalar extension or by normalizing outright.

    Takes an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh], whose
    [`HeatSolver`][ordito.heat.HeatSolver] at ``t`` holds the system and keeps its factorization
    for later calls, or the vector heat ``system`` itself.

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``system``.
    system
        ``(n_vertices, n_vertices)`` vector heat system from
        [`vector_heat_operators`][ordito.heat.vector_heat_operators].
    source
        ``(n_vertices,)`` right-hand side, in each vertex's own tangent frame.
    t
        With a `Trimesh`: the diffusion time; defaults to the squared mean edge length.
    preconditioner
        With ``system``: ``None``, or the Jacobi preconditioner for ``system`` -- the fourth field
        of [`vector_heat_operators`][ordito.heat.vector_heat_operators]'s return. The solve is
        Jacobi-preconditioned either way (see
        [`solve_spd_settled`][ordito.linalg.solve_spd_settled]).

    Returns
    -------
    wp.array[wp.vec2d]
        ``(n_vertices,)`` diffused field on ``source.device``.

    Raises
    ------
    ValueError
        If ``preconditioner`` is neither ``None`` nor ``system``'s own Jacobi preconditioner.

    Notes
    -----
    The ``system`` form logs at ``INFO`` on the ``ordito.heat`` logger when the solve's
    verification failed and it built a factorization it is about to drop, which the `Trimesh` form
    would have kept.

    See Also
    --------
    [`vector_heat_operators`][ordito.heat.vector_heat_operators]
    [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors]
    """
    if source is None:
        raise TypeError("diffuse_tangent_field: source is required")
    if isinstance(mesh, Trimesh):
        if system is not None or preconditioner is not None:
            raise TypeError("diffuse_tangent_field: a Trimesh takes no system or preconditioner")
        return _diffuse_tangent_field_on(mesh, source, t)
    if mesh is not None:
        if system is not None:
            raise TypeError("diffuse_tangent_field: system given by position and by keyword")
        system = mesh
    if system is None or t is not None:
        raise TypeError("diffuse_tangent_field: pass a Trimesh (and t), or a system")
    return _diffuse_tangent_field(system, source, preconditioner)


def _diffuse_tangent_field_on(
    mesh: Trimesh, source: wp.array[wp.vec2d], t: float | None
) -> wp.array[wp.vec2d]:
    """``diffuse_tangent_field`` on a `Trimesh`'s vector heat system at ``t``."""
    n_vertices = source.size
    diffused = _launch.zeros(n_vertices, dtype=wp.vec2d, device=source.device)
    if n_vertices > 0:
        mesh.heat_solver(t).diffuse_tangent(source, diffused)
    return diffused


def _diffuse_tangent_field(
    system: odt.SparseMatrix, source: wp.array[wp.vec2d], preconditioner: wpl.LinearOperator | None
) -> wp.array[wp.vec2d]:
    """``diffuse_tangent_field`` against a given ``system``, keeping no factorization."""
    n_vertices = source.size
    diffused = _launch.zeros(n_vertices, dtype=wp.vec2d, device=source.device)
    if n_vertices == 0:
        return diffused
    factorization = twl.OperatorFactorization(system)
    twl.solve_spd_settled(
        system,
        source,
        diffused,
        check_rounds=_HEAT_CHECK_ROUNDS,
        change_tolerance=_HEAT_CHANGE_TOLERANCE,
        settle_rounds=_HEAT_SETTLE_ROUNDS,
        preconditioner=preconditioner,
        factorization=factorization,
    )
    if factorization.factorization is not None:
        _LOGGER.info(
            "diffuse_tangent_field: built a sparse Cholesky factorization of the system (%d "
            "bytes) and is discarding it; pass an ordito.mesh.Trimesh to keep it for later calls",
            factorization.nbytes,
        )
    return diffused


class HeatSolver:
    """
    The heat method's operators and solves on one mesh at one diffusion time, kept by the mesh.

    What [`Trimesh.heat_solver`][ordito.mesh.Trimesh.heat_solver] returns, and what every function
    of this module runs on when it is given a `Trimesh`: the
    [`heat_operators`][ordito.heat.heat_operators] (and
    [`vector_heat_operators`][ordito.heat.vector_heat_operators]) at ``t``, assembled on first use,
    and a sparse Cholesky factorization of each system it solves, built when it pays. The split
    ``potpourri3d.MeshHeatMethodDistanceSolver`` and ``MeshVectorHeatSolver`` expose as solver
    objects.

    A system is factored (an [`OperatorFactorization`][ordito.linalg.OperatorFactorization]
    kept here) in two cases. On its **second** solve, because the first one does not repay a
    factorization's analysis and every later one does, whatever the conditioning -- the
    rent-or-buy choice, made online. And at once when a diffusion's settled conjugate-gradient
    iterate fails its backward-error check
    ([`solve_spd_settled`][ordito.linalg.solve_spd_settled]), which only a factorization answers
    correctly. A one-shot solve on a well-conditioned mesh therefore costs what the iteration costs.
    Each decision is logged at ``DEBUG`` on the ``ordito.heat`` logger.

    Attributes
    ----------
    t : float | None
        The diffusion time; ``None`` for the default, the squared mean unique-edge length.
    use_robust : bool
        Whether the scalar operators are built from mollified edge lengths (see
        [`heat_operators`][ordito.heat.heat_operators]).

    Notes
    -----
    A kept factorization holds device memory of the order of its system's fill-in for as long as
    this solver lives -- the mesh's lifetime -- reported by
    [`nbytes`][ordito.heat.HeatSolver.nbytes] and returned by
    [`release`][ordito.heat.HeatSolver.release] or
    [`Trimesh.release_factorizations`][ordito.mesh.Trimesh.release_factorizations]. The solver
    refers to its mesh weakly and is unusable once the mesh is gone.

    See Also
    --------
    [`Trimesh.heat_solver`][ordito.mesh.Trimesh.heat_solver]
    [`heat_geodesic`][ordito.heat.heat_geodesic]
    """

    def __init__(self, mesh: Trimesh, t: float | None = None, *, use_robust: bool = False) -> None:
        """Bind a solver to ``mesh`` at ``t``; nothing is assembled until first use."""
        self.t = None if t is None else float(t)
        self.use_robust = bool(use_robust)
        self._mesh = weakref.ref(mesh)
        self._operators: HeatOperators | None = None
        self._vector_operators: VectorHeatOperators | None = None
        self._systems: dict[str, _KeptSystem] = {}

    @property
    def operators(self) -> HeatOperators:
        """[`heat_operators`][ordito.heat.heat_operators] for the mesh at ``t``."""
        if self._operators is None:
            mesh = self._bound_mesh()
            if self.t is None and not self.use_robust:
                # The mesh's own cached bundle, so the property and this solver share one assembly.
                self._operators = mesh.heat_operators
            else:
                self._operators = heat_operators(
                    mesh.vertices,
                    mesh.faces,
                    self.t,
                    use_robust=self.use_robust,
                    cot_entries=None if self.use_robust else mesh.cotmatrix_entries,
                )
        return self._operators

    @property
    def vector_operators(self) -> VectorHeatOperators:
        """
        [`vector_heat_operators`][ordito.heat.vector_heat_operators] for the mesh at ``t``.

        Its scalar bundle is [`operators`][ordito.heat.HeatSolver.operators], one assembly.

        Raises
        ------
        ValueError
            If this solver was built with ``use_robust``: the connection Laplacian has no mollified
            counterpart.
        """
        if self.use_robust:
            raise ValueError("HeatSolver: the vector heat operators have no use_robust variant")
        if self._vector_operators is None:
            mesh = self._bound_mesh()
            if self.t is None:
                self._vector_operators = mesh.vector_heat_operators
            else:
                self._vector_operators = vector_heat_operators(
                    mesh.vertices,
                    mesh.faces,
                    self.t,
                    scalar_operators=self._operators,
                    frames=mesh.vertex_tangent_frames,
                )
            self._operators = self._vector_operators[1]
        return self._vector_operators

    @property
    def nbytes(self) -> int:
        """Device memory this solver's kept factorizations hold, in bytes."""
        return sum(kept.factorization.nbytes for kept in self._systems.values())

    def release(self) -> None:
        """Drop every kept factorization and forget the solve counts; the operators stay."""
        for kept in self._systems.values():
            kept.factorization.release()
        self._systems.clear()

    def factor(self) -> bool:
        """
        Factor the heat and Poisson systems now, before any solve.

        What [`heat_geodesic`][ordito.heat.heat_geodesic] solves; once both are factored a call on
        this solver's mesh runs no iteration. Without this, each is factored on its second solve
        (or at once when its iterate fails the check).

        Returns
        -------
        bool
            Whether both systems hold a factorization: ``False`` when one is over
            [`CHOLESKY_MEMORY_BUDGET`][ordito.cholesky.CHOLESKY_MEMORY_BUDGET], and its solves
            keep iterating.
        """
        operators = self.operators
        factored = True
        for role, matrix in (("heat", operators[0]), ("poisson", operators[3])):
            kept = self._systems.get(role)
            if kept is None:
                kept = self._systems[role] = _KeptSystem(
                    twl.OperatorFactorization(matrix, self._bound_mesh().vertices)
                )
            if kept.factorization.factorization is None and not kept.factorization.factor():
                factored = False
        return factored

    def diffuse(self, rhs: odt.ArrayNd, solution: odt.ArrayNd) -> None:
        """
        Diffuse scalar fields: solve the heat system ``(M - t L) u = rhs`` to settled entries.

        Parameters
        ----------
        rhs
            ``(n_vertices,)`` right-hand side, or ``(n_columns, n_vertices)`` columns diffused
            together, ``float64``.
        solution
            Same shape: the initial guess, overwritten with the diffused field.
        """
        operators = self.operators
        # Columns are solved together under the Jacobi preconditioner the settle solve builds.
        preconditioner = operators[1] if rhs.ndim == 1 else None
        self._settle("heat", operators[0], rhs, solution, preconditioner=preconditioner)

    def diffuse_tangent(self, rhs: wp.array[wp.vec2d], solution: wp.array[wp.vec2d]) -> None:
        """
        Diffuse a tangent field: solve the vector heat system ``(M + t L_connection) X = rhs``.

        Parameters
        ----------
        rhs
            ``(n_vertices,)`` right-hand side, each vector in its vertex's tangent frame.
        solution
            ``(n_vertices,)`` initial guess, overwritten with the diffused field.

        Raises
        ------
        ValueError
            If this solver was built with ``use_robust``.
        """
        vector_system, _, _, preconditioner = self.vector_operators
        self._settle("vector", vector_system, rhs, solution, preconditioner=preconditioner)

    def diffuse_stacked(
        self, rhs: wp.array[wp.float64], solution: wp.array[wp.float64], n_scalars: int
    ) -> None:
        """
        Diffuse a tangent field and ``n_scalars`` scalar fields as one solve.

        The two systems do not interact and settle at the same round, so they run as one solve
        over the block-diagonal stack ``[vector system; heat system x n_scalars]``
        ([`block_diag`][ordito.linalg.block_diag]), in the launches of one.

        Parameters
        ----------
        rhs
            ``((2 + n_scalars) * n_vertices,)`` right-hand side: the tangent field as interleaved
            ``float64`` pairs, then each scalar field.
        solution
            Same layout: the initial guess, overwritten with the diffused fields.
        n_scalars
            Scalar fields after the tangent field.

        Raises
        ------
        ValueError
            If this solver was built with ``use_robust``.
        """
        role = f"stack{n_scalars}"
        kept = self._systems.get(role)
        if kept is None:
            vector_system, scalar, _, _ = self.vector_operators
            stack = twl.block_diag((vector_system, *(scalar[0],) * n_scalars))
            positions = _stacked_positions(self._bound_mesh().vertices, n_scalars)
            kept = self._systems[role] = _KeptSystem(twl.OperatorFactorization(stack, positions))
        self._settle(role, kept.factorization.matrix, rhs, solution)

    def solve_poisson(self, rhs: wp.array[wp.float64], solution: wp.array[wp.float64]) -> None:
        """
        Solve the Poisson system ``-L phi = rhs`` to the heat method's tolerance.

        Parameters
        ----------
        rhs
            ``(n_vertices,)`` right-hand side; consistent (summing to zero over each component),
            as a divergence is.
        solution
            ``(n_vertices,)`` initial guess, overwritten with the answer, which is defined up to a
            constant per component.
        """
        operators = self.operators
        kept = self._kept("poisson", operators[3])
        if kept.factorization.factorization is not None:
            kept.factorization.solve(rhs, solution, tol=_CG_TOLERANCE)
            return
        twl.solve_spd(operators[3], rhs, solution, tol=_CG_TOLERANCE, preconditioner=operators[4])

    def _settle(
        self,
        role: str,
        matrix: odt.SparseMatrix,
        rhs: odt.ArrayNd,
        solution: odt.ArrayNd,
        *,
        preconditioner: wpl.LinearOperator | None = None,
    ) -> None:
        """Run one settled diffusion of ``role``'s system, factored when that pays or is needed."""
        kept = self._kept(role, matrix)
        held = kept.factorization.factorization is not None
        twl.solve_spd_settled(
            matrix,
            rhs,
            solution,
            check_rounds=_HEAT_CHECK_ROUNDS,
            change_tolerance=_HEAT_CHANGE_TOLERANCE,
            settle_rounds=_HEAT_SETTLE_ROUNDS,
            preconditioner=preconditioner,
            factorization=kept.factorization,
        )
        if not held and kept.factorization.factorization is not None:
            _LOGGER.debug(
                "HeatSolver: the %s system's settled iterate failed its backward-error check; "
                "factored it",
                role,
            )

    def _kept(self, role: str, matrix: odt.SparseMatrix) -> _KeptSystem:
        """Count a solve of ``role``'s system, factoring it on its second (rent or buy)."""
        kept = self._systems.get(role)
        if kept is None:
            kept = self._systems[role] = _KeptSystem(
                twl.OperatorFactorization(matrix, self._bound_mesh().vertices)
            )
        kept.solves += 1
        if (
            kept.solves >= 2
            and kept.factorization.factorization is None
            and kept.factorization.factor()
        ):
            _LOGGER.debug("HeatSolver: factored the %s system on its second solve", role)
        return kept

    def _bound_mesh(self) -> Trimesh:
        """Return the mesh this solver belongs to."""
        mesh = self._mesh()
        if mesh is None:
            raise RuntimeError("HeatSolver: its Trimesh no longer exists")
        return mesh


@dataclass
class _KeptSystem:
    """One system a ``HeatSolver`` solves: its factorization slot and how often it was solved."""

    factorization: twl.OperatorFactorization
    solves: int = 0


def _stacked_positions(vertices: wp.array[wp.vec3], n_scalars: int) -> wp.array[wp.vec3]:
    """
    Return the vertex of every row of ``_stacked_fields``' stack.

    A vector system's two rows, then each scalar system's one: the ordering of the factorization a
    failed settle falls back to.
    """
    n_vertices = vertices.size
    positions = _launch.empty((2 + n_scalars) * n_vertices, dtype=wp.vec3, device=vertices.device)
    _launch.launch(
        kernel_heat.stacked_positions,
        dim=n_vertices,
        inputs=[vertices, wp.int32(n_scalars)],
        outputs=[positions],
        device=vertices.device,
    )
    return positions


def _log_discarded(name: str, owned: bool, solver: HeatSolver) -> None:
    """Log that a call on a temporary `Trimesh` is dropping the factorizations it built."""
    if owned or solver.nbytes == 0:
        return
    _LOGGER.info(
        "%s: built a sparse Cholesky factorization (%d bytes) and is discarding it; pass an "
        "ordito.mesh.Trimesh to keep it for later calls",
        name,
        solver.nbytes,
    )
