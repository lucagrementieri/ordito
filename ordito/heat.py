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
import math
import warnings
import weakref
from dataclasses import dataclass
from typing import Any, cast, overload

import warp as wp
import warp.optim.linear as wpl

import ordito as od
import ordito.linalg as twl
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device
from ordito.constants import TILE_1D
from ordito.kernels import heat as kernel_heat
from ordito.kernels import linalg as kernel_linalg
from ordito.kernels import reduce as kernel_reduce
from ordito.laplacian import (
    MeshOperatorPattern,
    cotmatrix_entries,
    cotmatrix_entries_intrinsic,
    mass_matrix_entries,
    mesh_operator_pattern,
    mollify_intrinsic,
)
from ordito.mesh import Trimesh, mesh_arguments
from ordito.tangent_space import halfedge_transport_angles, vertex_tangent_frames

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

# ``extend_scalar`` warns once its field leaves the source values' range by more than this fraction
# of the range. Where every cotangent weight is non-negative the overshoot is exactly 0 (spheres
# and tori, regular or irregularly sampled, 2 to 40 sources); where obtuse triangles make some
# negative it is 2e-3 to 14x the range (``sphere_irregular``, ``torus_irregular``).
_EXTEND_RANGE_TOLERANCE = 1e-6


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
    pattern: MeshOperatorPattern | None = None,
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
    pattern
        Optional sparsity from [`mesh_operator_pattern`][ordito.laplacian.mesh_operator_pattern]
        (the default ``"cotmatrix"`` operator) on these ``faces``. Depends on the faces alone, so a
        caller assembling these operators again reuses it --
        [`Trimesh.mesh_operator_pattern`][ordito.mesh.Trimesh.mesh_operator_pattern] has it cached.
        The returned matrices share its ``offsets`` and ``columns``.

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
        very table from mollified edge lengths, so the two ask for different weights. Or if
        ``pattern`` was built for another operator or vertex count.
    RuntimeError
        If ``vertices``, ``faces``, ``cot_entries`` and ``pattern`` are not all on one device.

    See Also
    --------
    [`heat_geodesic`][ordito.heat.heat_geodesic]
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries]
    [`Trimesh.heat_operators`][ordito.mesh.Trimesh.heat_operators]

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    operators = od.heat.heat_operators(vertices, faces)
    heat_system, _, laplacian, poisson_system, *_ = operators
    print(len(operators))  # 8
    print(heat_system.nrow == vertices.shape[0])  # True
    ```
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        cot_entries=cot_entries,
        pattern=None if pattern is None else pattern.offsets,
    )
    if cot_entries is not None and use_robust:
        raise ValueError(
            "cot_entries and use_robust are mutually exclusive: use_robust rebuilds the "
            "half-cotangent table from mollified edge lengths."
        )
    _require_cotmatrix_pattern(pattern, vertices.size)
    return _heat_operators(
        vertices, faces, t, use_robust=use_robust, cot_entries=cot_entries, pattern=pattern
    )[0]


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
    (semi-)definite systems, solved on the mesh's device by conjugate gradient -- or by a sparse
    Cholesky factorization where one is held or needed, as [`HeatSolver`][ordito.heat.HeatSolver]
    decides: once a system is solved a second time on a `Trimesh`, and at once for both when the
    heat solve's iterate fails its accuracy check. The result is an *approximation* of the true
    geodesic distance (typically a few percent error), matching ``igl::heat_geodesics``.

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
        On a mesh with none it changes nothing. ``potpourri3d.MeshHeatMethodDistanceSolver``'s
        flag of the same name does more -- it also solves on an intrinsic Delaunay triangulation
        -- and defaults to ``True``; this defaults to ``False`` so the plain call stays exactly
        ``igl::heat_geodesics``.

    Returns
    -------
    wp.array[wp.float64]
        ``(n_vertices,)`` geodesic distance field on the mesh's device.

    Raises
    ------
    ValueError
        If an entry of ``sources`` is not a vertex index in ``[0, n_vertices)``.
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

    Examples
    --------
    Pole to pole on a unit sphere the true geodesic distance is ``pi``; the heat method is within a
    few per cent of it:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3)
    heights = vertices.numpy()[:, 2]
    north, south = int(np.argmax(heights)), int(np.argmin(heights))
    distance = od.heat.heat_geodesic(vertices, faces, wp.array([north], dtype=wp.int32))
    print(distance.numpy()[north])  # 0.0
    print(f"{distance.numpy()[south]:.2f}")  # 3.09
    ```
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
    invalid = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_heat.seed_source_indicator,
        dim=sources.size,
        inputs=[sources],
        outputs=[u0, invalid],
        device=device,
    )
    _require_indices_in_range(invalid, n_vertices, "heat_geodesic", "sources")

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
        If ``level_set_constraint`` is not ``"zero_set"`` or ``"none"``, or if an entry of
        ``curve_vertices`` is not a vertex index in ``[0, n_vertices)``.
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

    Examples
    --------
    A counter-clockwise square of side 0.6 on a flat unit grid: the centre lies 0.3 inside it, and
    the grid's corner lies outside:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(11, 11), extents=(1.0, 1.0))
    side = range(2, 8)  # vertex (i, j) sits at x = -0.5 + 0.1 * i, y = -0.5 + 0.1 * j
    square = (
        [(i, 2) for i in side]
        + [(8, j) for j in side]
        + [(i, 8) for i in range(8, 2, -1)]
        + [(2, j) for j in range(8, 2, -1)]
    )
    curve = wp.array([11 * i + j for i, j in square], dtype=wp.int32)
    distance = od.heat.heat_signed_distance(vertices, faces, curve).numpy()
    print(f"{distance[60]:.2f}")  # 0.30
    print(distance[0] < 0.0)  # True
    ```
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
    invalid = _launch.zeros(1, dtype=wp.int32, device=device)
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
        ],
        outputs=[source, invalid],
        device=device,
    )
    _require_indices_in_range(invalid, n_vertices, "heat_signed_distance", "curve_vertices")

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
            poisson_system, divergence, curve_vertices, vertices, solver.ill_conditioned
        )
    else:
        field = _solve_poisson_shifted(solver, divergence, curve_vertices, vertices)
    _log_discarded("heat_signed_distance", owned, solver)
    return field


def _solve_poisson_zero_set(
    operator: odt.BsrMatrix[wp.float64],
    divergence: wp.array[wp.float64],
    curve_vertices: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    factor: bool,
) -> wp.array[wp.float64]:
    """
    Solve the Poisson problem with the curve pinned to zero; by a factorization when ``factor``.

    ``linalg.min_quad_with_fixed`` minimizes ``0.5 x' Q x`` and has no place for a linear term, so
    the pieces underneath it are used directly: the free-free block comes from
    [`assemble_interior_system`][ordito.linalg.assemble_interior_system] (whose own right-hand side
    is zero here, the pinned values being zero) and the divergence is compacted into it.
    """
    device = vertices.device
    n_vertices = vertices.size
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
    direct = None
    if factor:
        # The diffusion before it failed its check on this mesh: so would this iteration's speed.
        direct = twl.OperatorFactorization(
            operator_uu, twl.free_coordinates(vertices, fixed_mask, free_map, n_free)
        )
    if direct is not None and direct.factor():
        direct.solve(rhs, solution, tol=_CG_TOLERANCE)
    else:
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
    blocks = kernel_reduce.blocks_1d(n_sources)
    sums, stride = _launch.block_partials(blocks, 2, wp.float64, device)
    _launch.launch_tiled(
        kernel_heat.source_and_global_sums,
        dim=[blocks],
        inputs=[field, curve_vertices, stride],
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
    pattern: MeshOperatorPattern | None = None,
    transport_angles: wp.array[wp.float32] | None = None,
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
    pattern
        Optional sparsity from [`mesh_operator_pattern`][ordito.laplacian.mesh_operator_pattern]
        (the default ``"cotmatrix"`` operator) on these ``faces``, shared by the vector and scalar
        systems. [`Trimesh.mesh_operator_pattern`][ordito.mesh.Trimesh.mesh_operator_pattern]
        caches it.
    transport_angles
        ``(3 * n_faces,)`` optional prebuilt
        [`halfedge_transport_angles`][ordito.tangent_space.halfedge_transport_angles] on these
        ``faces``, the connection the vector system is assembled from. A caller holding the mesh's
        twins and one-rings already builds it without walking the rings again;
        [`Trimesh.vector_heat_operators`][ordito.mesh.Trimesh.vector_heat_operators] does.

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
    ValueError
        If ``pattern`` was built for another operator or vertex count.
    RuntimeError
        If ``vertices``, ``faces``, ``frames``, ``pattern`` and ``transport_angles`` are not all on
        one device.

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

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    vector_system, scalar, frames, preconditioner = od.heat.vector_heat_operators(vertices, faces)
    print(vector_system.nrow, vector_system.values.dtype.__name__)  # 162 mat22d
    print(len(scalar), len(frames))  # 8 3
    ```
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        frames=frames,
        pattern=None if pattern is None else pattern.offsets,
        transport_angles=transport_angles,
    )
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    _require_cotmatrix_pattern(pattern, n_vertices)
    # One sparsity for both operators: the connection Laplacian's is the cotangent Laplacian's.
    if n_faces == 0:
        pattern = None
    elif pattern is None:
        pattern = mesh_operator_pattern(faces, n_vertices)
    # A bundle ``_heat_operators`` built carries the mass and timestep sums it assembled from.
    built = None if scalar_operators is None else _SCALAR_BUILDS.get(scalar_operators[0])
    edge_sums = None
    if t is None:
        # Shares the scalar solver's timestep convention -- the unique-edge mean, matching
        # ``igl::heat_geodesics``. The two solvers must agree: ``log_map``'s radius is asserted to
        # *be* the ``heat_geodesic`` distance, so giving them different diffusion times would split
        # a quantity that is supposed to be one number. They do by construction: the mean is read
        # off the shared sparsity, twelve triplets per face and nothing pruned, and the scalar
        # bundle built here takes these very sums (a prebuilt one already holds its own). Both
        # systems square it on the device, so nothing is read back.
        if built is not None and built.edge_sums is not None:
            edge_sums = built.edge_sums
        else:
            edge_sums = _edge_length_sums(vertices, pattern)

    if scalar_operators is None:
        scalar_operators, mass = _heat_operators(
            vertices, faces, t, edge_sums=edge_sums, pattern=pattern
        )
    elif built is not None:
        mass = built.mass
    else:
        # The bundle's own face areas, which ``_heat_operators`` builds the mass from too.
        mass = mass_matrix_entries(
            vertices, faces, dtype=wp.float64, face_areas=scalar_operators[7]
        )
    if pattern is None:
        vector_system = od.array.empty_square_bsr(n_vertices, wp.mat22d, device)
    else:
        # ``M + t L_connection`` over the cotangent Laplacian's pattern, as the scalar system is
        # built over it (``heat_operators``): ``connection_laplacian``'s entries, shifted in the
        # thread that forms them, so the connection Laplacian itself is never stored.
        cot_entries = cotmatrix_entries(vertices, faces, dtype=wp.float64)
        if transport_angles is None:
            transport_angles = halfedge_transport_angles(vertices, faces)
        _, _, offsets, columns, run_start, keys, count, order = pattern
        vector_values = _launch.empty(columns.size, dtype=wp.mat22d, device=device)
        _launch.launch(
            kernel_heat.VECTOR_HEAT_SYSTEM_ROWS[wp.float64],
            dim=n_vertices,
            inputs=[
                offsets,
                columns,
                run_start,
                keys,
                wp.int32(count),
                order,
                faces,
                cot_entries,
                transport_angles,
                mass,
                wp.float64(1.0 if t is None else t),
                edge_sums,
                wp.int32(1 if t is None else 0),
            ],
            outputs=[vector_values],
            device=device,
        )
        vector_system = od.array.bsr_from_csr(
            n_vertices, n_vertices, offsets, columns, vector_values
        )
    if frames is None:
        frames = vertex_tangent_frames(vertices, faces)
    preconditioner = twl.jacobi_preconditioner(vector_system)
    return vector_system, scalar_operators, frames, preconditioner


def _require_cotmatrix_pattern(pattern: MeshOperatorPattern | None, n_vertices: int) -> None:
    """Raise unless a caller's ``pattern`` serves the cotangent Laplacian over ``n_vertices``."""
    if pattern is not None and (
        pattern.operator != "cotmatrix" or pattern.n_vertices != n_vertices
    ):
        raise ValueError(
            f"pattern was built for operator={pattern.operator!r} over {pattern.n_vertices} "
            f"vertices; the heat operators need operator='cotmatrix' over {n_vertices}"
        )


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
    # Face normals and areas (float32) for the gradient, the half-cotangent table when the caller
    # has none, and the lumped mass's exact float64 area sums, in one pass over the faces.
    device = vertices.device
    n_faces = faces.size // 3
    write_cot = cot_entries is None
    if cot_entries is None:
        cot_entries = odt.empty_2d((n_faces, 3), wp.float32, device=device)
    normals = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    areas = _launch.empty(n_faces, dtype=wp.float32, device=device)
    mass = _launch.zeros(vertices.size, dtype=wp.float64, device=device)
    if n_faces > 0:
        _launch.launch(
            kernel_heat.heat_face_terms,
            dim=n_faces,
            inputs=[vertices, faces, wp.int32(write_cot)],
            outputs=[
                cot_entries if write_cot else odt.empty_2d((0, 3), wp.float32, device=device),
                normals,
                areas,
                mass,
            ],
            device=device,
        )
    # The cotangent Laplacian, the heat system ``M - t L`` and the Poisson operator ``-L``, all
    # over the one pattern (which stores every referenced vertex's diagonal) and written by one row
    # kernel (``kernels/heat.heat_system_rows``), which casts the half-cotangent weights to
    # ``float64`` as it reads them and forms the lumped mass from the area sums in the same pass.
    # The two preconditioners are mesh-only, so they belong here rather than in every solve. See
    # Notes.
    n_vertices = vertices.size
    if n_faces == 0:
        laplacian = od.array.empty_square_bsr(n_vertices, wp.float64, device)
        heat_values = _launch.empty_like(laplacian.values)
        poisson_values = _launch.empty_like(laplacian.values)
        if t is None and edge_sums is None:
            edge_sums = _edge_length_sums(vertices, None)
    else:
        if int(cot_entries.shape[0]) != n_faces:
            raise ValueError(
                f"cot_entries must have shape (n_faces, 3) = ({n_faces}, 3), "
                f"got {(int(cot_entries.shape[0]), int(cot_entries.shape[1]))}"
            )
        if pattern is None:
            pattern = mesh_operator_pattern(faces, n_vertices)
        if t is None and edge_sums is None:
            # The unique-edge average, which is what ``igl::heat_geodesics`` uses for its timestep
            # -- read off the Laplacian's own sparsity, which already holds the unique edges, and
            # left on the device for the system's assembly to square.
            edge_sums = _edge_length_sums(vertices, pattern)
        _, _, offsets, columns, run_start, keys, count, order = pattern
        laplacian_values = _launch.empty(columns.size, dtype=wp.float64, device=device)
        heat_values = _launch.empty(columns.size, dtype=wp.float64, device=device)
        poisson_values = _launch.empty(columns.size, dtype=wp.float64, device=device)
        _launch.launch(
            kernel_heat.HEAT_SYSTEM_ROWS[cot_entries.dtype],
            dim=n_vertices,
            inputs=[
                offsets,
                columns,
                run_start,
                keys,
                wp.int32(count),
                order,
                cot_entries,
                wp.float64(-1.0 if t is None else -t),
                edge_sums if t is None else None,
                wp.int32(1 if t is None else 0),
                mass,
            ],
            # The lumped mass over its own area sums, in place.
            outputs=[laplacian_values, heat_values, poisson_values, mass],
            device=device,
        )
        laplacian = od.array.bsr_from_csr(
            n_vertices, n_vertices, offsets, columns, laplacian_values
        )

    # The divergence kernel this bundle feeds (``kernels/heat.py::unit_gradient_divergence``) is
    # hardcoded ``wp.array2d[wp.float32]``, but this tuple's own ``cot_entries`` field is documented
    # and used downstream as float32 only, so a caller-supplied float64 table must be narrowed
    # before it is returned rather than passed through at whatever precision it arrived in.
    if cot_entries.dtype is not wp.float32:
        cot_entries = odt.as_array2d(od.array.astype(cot_entries, wp.float32), wp.float32)

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
    _SCALAR_BUILDS[heat_system] = _ScalarBuild(mass, edge_sums)
    return operators, mass


@dataclass(frozen=True)
class _ScalarBuild:
    """What ``_heat_operators`` assembled a bundle from that ``vector_heat_operators`` reuses."""

    mass: wp.array[wp.float64]
    edge_sums: wp.array[wp.float64] | None


# Each bundle's ``_ScalarBuild``, keyed weakly by its heat system, so it lives exactly as long as
# the bundle does: a prebuilt bundle handed to ``vector_heat_operators`` (a `Trimesh`'s cached
# ``heat_operators``) gives up its ``float64`` mass and timestep sums rather than having them
# rebuilt. ``edge_sums`` is ``None`` for a bundle built at an explicit ``t``.
_SCALAR_BUILDS: weakref.WeakKeyDictionary[odt.SparseMatrix, _ScalarBuild] = (
    weakref.WeakKeyDictionary()
)


def _edge_length_sums(
    vertices: wp.array[wp.vec3], pattern: MeshOperatorPattern | None
) -> wp.array[wp.float64]:
    """
    Sum and count of the unique-edge lengths, read off a mesh operator pattern.

    The strict upper triangle of the heat method's operators' pattern is the mesh's unique edge
    set, so the default timestep costs one launch over a pattern that already exists, where
    [`mean_unique_edge_length`][ordito.edges.mean_unique_edge_length] would re-sort every edge of
    the mesh to recover the same set. The two sums stay on the device, where
    ``kernels/linalg.shift_factor`` squares their mean -- ``0`` for a mesh with no edges (``None``
    for the pattern of a mesh with no faces), as that function returns.
    """
    device = vertices.device
    if pattern is None or pattern.n_vertices == 0:
        return _launch.zeros(2, dtype=wp.float64, device=device)
    n_rows = pattern.n_vertices
    # Block rows added in a fixed order, so the timestep repeats bit for bit on CUDA.
    blocks = kernel_reduce.blocks_1d(n_rows)
    sum_and_count, stride = _launch.block_partials(blocks, 2, wp.float64, device)
    _launch.launch_tiled(
        kernel_heat.upper_edge_length_sum_and_count,
        dim=[blocks],
        inputs=[pattern.offsets, pattern.columns, vertices, stride, sum_and_count],
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
    ValueError
        If ``sources`` is not empty and ``values`` is not as long, or if an entry of ``sources``
        is not a vertex index in ``[0, n_vertices)``.
    RuntimeError
        If ``vertices``, ``faces``, ``sources`` and ``values`` are not all on one device.

    Warns
    -----
    UserWarning
        When the extended field leaves the range of the source values (see Notes).

    Notes
    -----
    The extension stays inside the source values' range exactly when every cotangent weight is
    non-negative: the diffusion is then a convex average. On a mesh with obtuse triangles some
    weights are negative, and the quotient can overshoot -- far, where the indicator is small. The
    call warns when that happens, measuring every vertex a source reaches; geometry-central's
    extension does the same arithmetic and returns the same values.

    The ``vertices, faces`` form logs at ``INFO`` on the ``ordito.heat`` logger when it built a
    factorization it is about to drop, as [`heat_geodesic`][ordito.heat.heat_geodesic] does.

    See Also
    --------
    [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors]
    [`heat_geodesic`][ordito.heat.heat_geodesic]

    Examples
    --------
    Value ``1`` at the north pole and ``-1`` at the south pole: each hemisphere takes its nearest
    pole's value, away from the equator where the two blend:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3)
    heights = vertices.numpy()[:, 2]
    poles = wp.array([np.argmax(heights), np.argmin(heights)], dtype=wp.int32)
    values = wp.array([1.0, -1.0], dtype=wp.float64)
    extended = od.heat.extend_scalar(vertices, faces, poles, values).numpy()
    print(f"{extended[heights > 0.5].min():.2f}")  # 1.00
    print(f"{extended[heights < -0.5].max():.2f}")  # -1.00
    ```
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
    _require_same_length(values, n_sources, "extend_scalar", "values")

    solver = mesh.heat_solver(t)
    # The indicator and the weighted values diffuse through the same operator, so they are one
    # batched two-column solve rather than two independent ones, which shares the launches and
    # converges on the worse-behaved of the two columns.
    rhs = odt.as_array2d(
        _launch.zeros((2, n_vertices), dtype=wp.float64, device=device), wp.float64
    )
    # ``(min, -max, -overshoot)`` of the extension against the source values; see the kernels.
    value_range = wp.full(3, math.inf, dtype=wp.float64, device=device)
    invalid = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_heat.seed_source_scalars,
        dim=n_sources,
        inputs=[sources, values],
        outputs=[rhs[0], rhs[1], value_range, invalid],
        device=device,
    )
    _require_indices_in_range(invalid, n_vertices, "extend_scalar", "sources")
    diffused = odt.as_array2d(
        _launch.zeros((2, n_vertices), dtype=wp.float64, device=device), wp.float64
    )
    solver.diffuse(rhs, diffused)
    # Converged per vertex, so only an exactly zero indicator -- a component no source reaches --
    # has no value to extend.
    extended = _launch.empty(n_vertices, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_heat.extend_from_ratio,
        dim=n_vertices,
        inputs=[odt.as_dense(diffused[0]), odt.as_dense(diffused[1])],
        outputs=[extended, value_range],
        device=device,
    )
    _log_discarded("extend_scalar", owned, solver)
    # Host readback: the overshoot decides the warning; the three values are one copy.
    low, negative_high, negative_excess = read_values(value_range, 0, 3)
    _warn_if_out_of_range(float(low), -float(negative_high), -float(negative_excess))
    return extended


def _warn_if_out_of_range(low: float, high: float, excess: float) -> None:
    """Warn when ``extend_scalar``'s field leaves its source values' range by more than rounding."""
    if math.isinf(excess):  # no vertex had a non-zero indicator to measure
        return
    scale = max(high - low, abs(low), abs(high))
    if excess > _EXTEND_RANGE_TOLERANCE * scale:
        warnings.warn(
            f"extend_scalar: the extended field leaves the source values' range "
            f"[{low:.6g}, {high:.6g}] by {excess:.3g}. The cotangent weights of obtuse triangles "
            "are negative here, so the diffusion is not a convex average; see Notes.",
            stacklevel=4,
        )


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
    ValueError
        If ``sources`` is not empty and ``vectors`` is not as long, or if an entry of ``sources`` is
        not a vertex index in ``[0, n_vertices)``.
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

    Examples
    --------
    On a flat grid parallel transport keeps a vector's direction in space, so the vector carried
    from the centre to an interior vertex is the source vector again:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(11, 11), extents=(1.0, 1.0))
    sources = wp.array([60], dtype=wp.int32)  # the vertex at the origin
    vectors = wp.array([wp.vec2(1.0, 0.0)], dtype=wp.vec2)
    transported, resolved = od.heat.transport_tangent_vectors(vertices, faces, sources, vectors)
    basis_x, basis_y, _ = od.tangent_space.vertex_tangent_frames(vertices, faces)
    world = od.heat.tangent_to_world(transported, basis_x, basis_y).numpy()
    print(resolved.numpy().all())  # True
    print(np.allclose(world[12], world[60], atol=1e-2))  # True
    ```
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
    _require_same_length(vectors, n_sources, "transport_tangent_vectors", "vectors")

    solver = mesh.heat_solver(t)

    # The vector field and the magnitudes' two-column extension do not interact and settle at the
    # same round, so they diffuse as one solve over the block-diagonal stack
    # ``[vector system; heat system; heat system]`` (``HeatSolver.diffuse_stacked``): one settle
    # loop in the launches of one, where two back to back paid for two. Its Jacobi diagonal and
    # narrowed values are per row, so they are the two systems' own, and its settle test reads the
    # union, which is the later of the two stops.
    rhs_field, rhs = _stacked_fields(n_vertices, 2, device)
    invalid = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_heat.seed_transport_sources,
        dim=n_sources,
        inputs=[sources, vectors],
        outputs=[rhs_field, rhs[2 * n_vertices : 3 * n_vertices], rhs[3 * n_vertices :], invalid],
        device=device,
    )
    _require_indices_in_range(invalid, n_vertices, "transport_tangent_vectors", "sources")
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
    ValueError
        If the mesh has faces and ``source`` is not a vertex index in ``[0, n_vertices)``.
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

    Examples
    --------
    On a flat grid the log map centred on the origin returns each vertex's own position, expressed
    in the centre's tangent frame:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.grid(count=(11, 11), extents=(1.0, 1.0))
    coordinates = od.heat.log_map(vertices, faces, 60)  # centred on the vertex at the origin
    basis_x, basis_y, _ = od.tangent_space.vertex_tangent_frames(vertices, faces)
    x, y = coordinates.numpy()[108]  # the vertex at (0.4, 0.4)
    world = x * basis_x.numpy()[60] + y * basis_y.numpy()[60]
    print(np.round(world[:2], 2))  # [0.41 0.41]
    ```
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
    if not 0 <= source < n_vertices:
        raise ValueError(
            f"log_map: source must be a vertex index in [0, {n_vertices}), got {source}"
        )

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
    blocks = kernel_reduce.blocks_1d(max(n_vertices, n_sources))
    sums, stride = _launch.block_partials(blocks, 2, wp.float64, device)
    _launch.launch_tiled(
        kernel_heat.source_and_global_sums,
        dim=[blocks],
        inputs=[phi, sources, stride],
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

    Examples
    --------
    A tangent vector of length 5 at every vertex of a flat grid stays length 5 and in the plane:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(5, 5))
    basis_x, basis_y, _ = od.tangent_space.vertex_tangent_frames(vertices, faces)
    tangent = wp.full(vertices.shape[0], wp.vec2(3.0, 4.0), dtype=wp.vec2)
    world = od.heat.tangent_to_world(tangent, basis_x, basis_y).numpy()
    print(f"{np.linalg.norm(world, axis=1).max():.3f}")  # 5.000
    print(np.abs(world[:, 2]).max())  # 0.0
    ```
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
    TypeError
        If ``source`` is missing; if a `Trimesh` is given with a ``system`` or ``preconditioner``;
        if ``system`` is given both by position and by keyword; or if neither a `Trimesh` nor a
        ``system`` is given, or ``t`` is given with a ``system``.
    ValueError
        If ``source`` is not empty and does not hold one vector per vertex (per block row of
        ``system``), or if ``preconditioner`` is neither ``None`` nor ``system``'s own Jacobi
        preconditioner.
    RuntimeError
        If the mesh (or ``system`` and ``preconditioner``) and ``source`` are not all on one
        device.

    Notes
    -----
    The ``system`` form logs at ``INFO`` on the ``ordito.heat`` logger when the solve's
    verification failed and it built a factorization it is about to drop, which the `Trimesh` form
    would have kept.

    See Also
    --------
    [`vector_heat_operators`][ordito.heat.vector_heat_operators]
    [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors]

    Examples
    --------
    A single vector at the centre of a grid spreads out, staying largest where it started:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(11, 11), extents=(1.0, 1.0))
    mesh = od.mesh.Trimesh(vertices, faces)
    source = np.zeros((vertices.shape[0], 2))
    source[60] = (1.0, 0.0)
    diffused = od.heat.diffuse_tangent_field(mesh, wp.array(source, dtype=wp.vec2d))
    print(diffused.shape)  # (121,)
    print(np.argmax(np.linalg.norm(diffused.numpy(), axis=1)))  # 60
    ```
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
    require_same_device(vertices=mesh.vertices, faces=mesh.faces, source=source)
    if source.size == 0:
        return _launch.zeros(0, dtype=wp.vec2d, device=source.device)
    # Before the solver is reached: a rejected call must not count as one of its solves.
    n_vertices = mesh.n_vertices
    _require_same_length(source, n_vertices, "diffuse_tangent_field", "source")
    diffused = _launch.zeros(n_vertices, dtype=wp.vec2d, device=source.device)
    mesh.heat_solver(t).diffuse_tangent(source, diffused)
    return diffused


def _diffuse_tangent_field(
    system: odt.SparseMatrix, source: wp.array[wp.vec2d], preconditioner: wpl.LinearOperator | None
) -> wp.array[wp.vec2d]:
    """``diffuse_tangent_field`` against a given ``system``, keeping no factorization."""
    require_same_device(system=system, source=source, preconditioner=preconditioner)
    if source.size == 0:
        return _launch.zeros(0, dtype=wp.vec2d, device=source.device)
    n_vertices = int(system.nrow)
    _require_same_length(source, n_vertices, "diffuse_tangent_field", "source")
    diffused = _launch.zeros(n_vertices, dtype=wp.vec2d, device=source.device)
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

    Examples
    --------
    A mesh's solver factors a system on its second solve and keeps the factorization until released:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3)
    mesh = od.mesh.Trimesh(vertices, faces)
    solver = mesh.heat_solver()
    print(solver.nbytes)  # 0
    for source in (0, 1):
        od.heat.heat_geodesic(mesh, sources=wp.array([source], dtype=wp.int32))
    print(solver.nbytes > 0)  # True
    solver.release()
    print(solver.nbytes)  # 0
    ```
    """

    def __init__(self, mesh: Trimesh, t: float | None = None, *, use_robust: bool = False) -> None:
        """Bind a solver to ``mesh`` at ``t``; nothing is assembled until first use."""
        self.t = None if t is None else float(t)
        self.use_robust = bool(use_robust)
        self._mesh = weakref.ref(mesh)
        self._operators: HeatOperators | None = None
        self._vector_operators: VectorHeatOperators | None = None
        self._systems: dict[str, _KeptSystem] = {}
        # Set once a settled iterate of this mesh's systems fails its backward-error check: the
        # operators are ill-conditioned in places, so the Poisson solve is factored at once too.
        self._ill_conditioned = False

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
                    pattern=mesh.mesh_operator_pattern,
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
                    pattern=mesh.mesh_operator_pattern,
                    transport_angles=mesh.halfedge_transport_angles,
                )
            self._operators = self._vector_operators[1]
        return self._vector_operators

    @property
    def ill_conditioned(self) -> bool:
        """Whether a settled diffusion on this mesh failed its backward-error check."""
        return self._ill_conditioned

    @property
    def nbytes(self) -> int:
        """Device memory this solver's kept factorizations hold, in bytes."""
        return sum(kept.factorization.nbytes for kept in self._systems.values())

    def release(self) -> None:
        """
        Drop every kept factorization and forget the solve counts.

        The operators stay, and so does [`ill_conditioned`][ordito.heat.HeatSolver.ill_conditioned]:
        it describes the mesh's operators, not a factorization, so after a release a Poisson solve
        on an ill-conditioned mesh is still factored at once rather than iterated first.
        """
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
            if not kept.factorization.factor(self._sharing(role)):
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

        Raises
        ------
        ValueError
            If ``rhs`` does not hold ``n_vertices`` entries per column, or ``solution`` is not
            shaped as ``rhs``.
        RuntimeError
            If ``rhs``, ``solution`` and the mesh are not all on one device.
        """
        self._require_fields("HeatSolver.diffuse", rhs, solution, rhs.shape[-1], 1)
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
            If this solver was built with ``use_robust``, or if ``rhs`` and ``solution`` do not
            hold one vector per vertex.
        RuntimeError
            If ``rhs``, ``solution`` and the mesh are not all on one device.
        """
        self._require_fields("HeatSolver.diffuse_tangent", rhs, solution, rhs.size, 1)
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
            If this solver was built with ``use_robust``, or if ``rhs`` and ``solution`` do not
            hold ``(2 + n_scalars) * n_vertices`` entries.
        RuntimeError
            If ``rhs``, ``solution`` and the mesh are not all on one device.
        """
        self._require_fields("HeatSolver.diffuse_stacked", rhs, solution, rhs.size, 2 + n_scalars)
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

        Raises
        ------
        ValueError
            If ``rhs`` and ``solution`` do not hold one entry per vertex.
        RuntimeError
            If ``rhs``, ``solution`` and the mesh are not all on one device.
        """
        self._require_fields("HeatSolver.solve_poisson", rhs, solution, rhs.size, 1)
        operators = self.operators
        kept = self._kept("poisson", operators[3])
        heat = self._systems.get("heat")
        if heat is not None and heat.factorization.factorization is not None:
            # The heat system is factored (its settled iterate failed, or it is solved again), and
            # the Poisson system is stored over its pattern: the numeric factorization alone costs
            # less than one Poisson iteration on such an operator.
            kept.factorization.factor(heat.factorization)
        elif self._ill_conditioned:
            # A diffusion on this mesh failed its check: its Poisson iteration is as slow as the
            # diffusion's was, and an analysis and factorization cost less.
            kept.factorization.factor()
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
            self._ill_conditioned = True
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
            and kept.factorization.factor(self._sharing(role))
        ):
            _LOGGER.debug("HeatSolver: factored the %s system on its second solve", role)
        return kept

    def _sharing(self, role: str) -> twl.OperatorFactorization | None:
        """Return the heat system's factorization slot, whose analysis the Poisson one shares."""
        heat = self._systems.get("heat")
        return heat.factorization if role == "poisson" and heat is not None else None

    def _require_fields(
        self, caller: str, rhs: odt.ArrayNd, solution: odt.ArrayNd, rows: int, per_vertex: int
    ) -> None:
        """
        Raise unless ``rhs`` and ``solution`` match each other and the mesh, before any solve.

        ``rows`` is the extent of ``rhs`` that must be ``per_vertex * n_vertices``. Checked before
        the solve is counted, so a rejected call does not move the rent-or-buy tally.
        """
        vertices = self._bound_mesh().vertices
        require_same_device(vertices=vertices, rhs=rhs, solution=solution)
        n_rows = per_vertex * vertices.size
        if int(rows) != n_rows or tuple(solution.shape) != tuple(rhs.shape):
            raise ValueError(
                f"{caller}: rhs must hold {n_rows} rows for this mesh and solution its shape, got "
                f"{tuple(rhs.shape)} and {tuple(solution.shape)}"
            )

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
        kernel_linalg.stacked_positions,
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


def _require_same_length(array: wp.array[Any], length: int, caller: str, name: str) -> None:
    """Raise ``ValueError`` unless ``array`` holds ``length`` entries, before a kernel reads it."""
    if array.size != length:
        raise ValueError(f"{caller}: {name} must hold {length} entries, got {array.size}")


def _require_indices_in_range(
    invalid: wp.array[wp.int32], n_vertices: int, caller: str, name: str
) -> None:
    """
    Raise ``ValueError`` when a seeding kernel flagged a vertex index outside ``[0, n_vertices)``.

    The seeding kernels skip such an index (``kernels/heat.flag_out_of_range``), so nothing was
    written out of bounds; the flag says whether to refuse the call.
    """
    # Host readback: whether the caller's indices were valid decides whether the call raises. Read
    # right after the seeding launch, before any solve, so a rejected call neither pays for a
    # diffusion nor counts as one of the ``HeatSolver``'s solves (its rent-or-buy tally).
    if read_scalar(invalid, 0) != 0:
        raise ValueError(
            f"{caller}: every entry of {name} must be a vertex index in [0, {n_vertices})"
        )
