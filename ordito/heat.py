"""
Heat-diffusion methods on triangle meshes.

Three solvers that share one idea: diffuse a quantity over the surface for a short time ``t``, then
recover the answer from the *direction* of the resulting field rather than its magnitude. Short-time
heat flow approximates the geodesic kernel, so a single sparse solve carries information a
combinatorial shortest-path search would have to walk edge by edge. Each solver differs only in what
it diffuses and how it reads the result back:

- **Geodesic distance.** [`heat_geodesic`][ordito.heat.heat_geodesic] diffuses a scalar indicator
  from source vertices, normalizes its gradient, and integrates that unit field back with a Poisson
  solve. The heat method of Crane et al., as MeshLab or ``igl::heat_geodesics`` discretize it.
- **Signed distance to curves.** [`heat_signed_distance`][ordito.heat.heat_signed_distance]
  diffuses the normals of a set of oriented curves, then solves a Poisson problem against that field
  to get a *signed* distance whose zero set is the curves (``potpourri3d.MeshSignedHeatSolver``).
- **Vector-valued transport.**
  [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors],
  [`extend_scalar`][ordito.heat.extend_scalar] and [`log_map`][ordito.heat.log_map] diffuse
  *tangent vectors* through the connection Laplacian, which transports each vector into its
  neighbour's frame before differencing (``potpourri3d.MeshVectorHeatSolver``).

Beside them, [`fast_marching_distance`][ordito.heat.fast_marching_distance] answers the first
task without a diffusion: a front marched across the faces, the serial alternative to the heat
method, solved here in parallel rounds.

The three are one module because two of them cannot be separated: the vector solvers import
[`heat_operators`][ordito.heat.heat_operators] and [`heat_geodesic`][ordito.heat.heat_geodesic],
[`VectorHeatOperators`][ordito.heat.VectorHeatOperators] embeds the scalar method's operator tuple,
and [`log_map`][ordito.heat.log_map]'s radius *is* ``heat_geodesic``'s answer at the vector
method's diffusion time.

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
from typing import Any, Literal, TypeVar, cast, overload

import warp as wp
import warp.optim.linear as wpl

import ordito as od
import ordito.linalg as twl
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device, run_device_loop
from ordito.cholesky import SparseCholesky, held_nbytes
from ordito.config import is_deterministic
from ordito.constants import INT64_MAX, TILE_1D
from ordito.halfedge import vertex_one_rings
from ordito.kernels import array as kernel_array
from ordito.kernels import heat as kernel_heat
from ordito.kernels import linalg as kernel_linalg
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import scatter as kernel_scatter
from ordito.kernels import tangent_space as kernel_tangent_space
from ordito.laplacian import (
    MeshOperatorPattern,
    cotmatrix_entries,
    cotmatrix_entries_intrinsic,
    mass_matrix_entries,
    mesh_operator_pattern,
    mollify_intrinsic,
)
from ordito.mesh import Trimesh, mesh_arguments
from ordito.tangent_space import halfedge_tangent_angles, vertex_tangent_frames

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
# The kept systems whose operators carry the connection's rotations (a tangent field), besides the
# ``stack{n}`` systems that solve one beside scalar fields.
_ROTATION_ROLES = frozenset({"vector", "vector_pair", "crouzeix_raviart"})
_HEAT_CHANGE_TOLERANCE = 1e-6
_HEAT_SETTLE_ROUNDS = 192

# The decay, in nepers, a diffusion's field may cover across the mesh before the default time of
# a ``"potpourri3d"``-discretization solver grows. A diffused field falls off like
# ``exp(-d / sqrt(t))`` at distance ``d`` from its source, so on a fine enough mesh its far side
# drops below the ``float64`` range and every direction there is lost (the edge system at
# geometry-central's ``(h / 2) ** 2`` from a few hundred thousand faces, the vertex systems at
# ``h ** 2`` from about four times as many; potpourri3d's direct solves too). The default time is
# therefore at least ``(diagonal / _DIFFUSION_DECAY) ** 2``, the bounding-box diagonal standing in
# for the farthest distance from a source. Meshes up to about 400 diffusion lengths across keep
# the method's own time. A longer time needs a longer settle window (``_settle_window``).
_DIFFUSION_DECAY = 400.0

# ``extend_scalar`` warns once its field leaves the source values' range by more than this fraction
# of the range. Where every cotangent weight is non-negative the overshoot is the settled solve's
# own accuracy: 0 on small meshes (spheres and tori, regular or irregularly sampled, 2 to 40
# sources), 2.3e-6 on a sphere whose far field the lengthened default time keeps in range (the
# quotient of two fields each settled to ``_HEAT_CHANGE_TOLERANCE``); where obtuse triangles make
# some negative it is 2e-3 to 14x the range (``sphere_irregular``, ``torus_irregular``).
_EXTEND_RANGE_TOLERANCE = 1e-4


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


HeatMethod = Literal["potpourri3d", "meshlab", "igl"]
"""The heat method discretizations ordito assembles, named for the library each reproduces."""


def heat_operators(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    t: float | None = None,
    *,
    use_robust: bool = False,
    cot_entries: odt.Array2dFloat | None = None,
    pattern: MeshOperatorPattern | None = None,
    method: HeatMethod = "potpourri3d",
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
    method
        Whose discretization of the heat system to assemble -- the lumped mass ``M`` and the mean
        edge length ``h`` of the default ``t = h ** 2``:

        - ``"potpourri3d"`` (default): each vertex a third of its faces' areas, ``h`` over the
          unique edges. What ``potpourri3d``'s solvers assemble, and what the vector heat method
          ([`vector_heat_operators`][ordito.heat.vector_heat_operators]) builds on.
        - ``"meshlab"``: the same mass, ``h`` over every face's three edges (an interior edge
          twice, a rim edge once; the same on a closed mesh).
        - ``"igl"``: Meyer et al.'s mixed Voronoi area per vertex (an obtuse face's obtuse corner
          takes half the face), ``h`` over the unique edges, ``igl::heat_geodesics``' choices.

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
        ``pattern`` was built for another operator or vertex count, or ``method`` is not one of
        the three names above.
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
    mass, timestep = _heat_method_choices(method)
    return _heat_operators(
        vertices,
        faces,
        t,
        use_robust=use_robust,
        cot_entries=cot_entries,
        pattern=pattern,
        mass=mass,
        timestep=timestep,
    )[0]


@overload
def heat_geodesic(
    mesh: Trimesh,
    sources: wp.array[wp.int32],
    /,
    *,
    t: float | None = None,
    use_robust: bool = False,
    method: Literal["meshlab", "igl"] = "meshlab",
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
    method: Literal["meshlab", "igl"] = "meshlab",
) -> wp.array[wp.float64]: ...
def heat_geodesic(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | None = None,
    sources: wp.array[wp.int32] | None = None,
    t: float | None = None,
    *,
    vertices: wp.array[wp.vec3] | None = None,
    use_robust: bool = False,
    method: Literal["meshlab", "igl"] = "meshlab",
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
    geodesic distance (typically a few percent error).

    The method leaves three discretization choices open, and ``method`` makes them as one of two
    libraries does, so the field is that library's up to its direct solves' rounding:

    - ``"meshlab"`` (default), MeshLab's ``compute_scalar_by_heat_geodesic_distance_from_selection
      _per_vertex``: the barycentric lumped mass, Neumann boundary conditions, and
      ``t = h ** 2`` with ``h`` the mean over every face's three edges.
    - ``"igl"``, ``igl::heat_geodesics``: the Voronoi lumped mass, the mean of the Neumann heat and
      the heat pinned to zero on the rim (a second heat solve on a mesh with a rim), and ``h`` the
      mean over the unique edges.

    Against exact polyhedral geodesics MeshLab's Neumann boundary is the more accurate of the two:
    equal on a hemisphere, and 0.93 % against 1.15 % mean error (3.1 % against 4.9 % worst) of the
    distance range on a half torus. geometry-central (``potpourri3d``) makes MeshLab's mass and
    boundary choices but regularizes its Poisson solve with ``1e-6`` times the identity, which
    stops the method converging on fine meshes; ordito leaves that out.

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
        Diffusion time. When ``None``, the squared mean edge length ``method`` names, which
        balances accuracy and smoothing.
    use_robust
        Forwarded to [`heat_operators`][ordito.heat.heat_operators]: build the Laplacian
        from mollified edge lengths, which is what makes the solves survive degenerate triangles.
        On a mesh with none it changes nothing. ``potpourri3d.MeshHeatMethodDistanceSolver``'s
        flag of the same name does more -- it also solves on an intrinsic Delaunay triangulation
        -- and defaults to ``True``; this defaults to ``False``.
    method
        ``"meshlab"`` (default) or ``"igl"``: whose discretization to reproduce (above). The two
        part most on obtuse triangles (the mass) and from a source inside an open mesh (the
        boundary).

    Returns
    -------
    wp.array[wp.float64]
        ``(n_vertices,)`` geodesic distance field on the mesh's device.

    Raises
    ------
    ValueError
        If an entry of ``sources`` is not a vertex index in ``[0, n_vertices)``, or ``method`` is
        not ``"meshlab"`` or ``"igl"``.
    RuntimeError
        If ``vertices``, ``faces`` and ``sources`` are not all on one device.

    Notes
    -----
    The ``vertices, faces`` form logs at ``INFO`` on the ``ordito.heat`` logger when it built a
    factorization it is about to drop (the heat solve's verification failed on an ill-conditioned
    mesh), since the `Trimesh` form would have kept it for the next call. That form runs on
    ``mesh.heat_solver(t, use_robust=use_robust, method=method)``, the solver to
    [`factor`][ordito.heat.HeatSolver.factor] ahead of the calls.

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
    if method not in ("meshlab", "igl"):
        raise ValueError(f"method must be 'meshlab' or 'igl', got {method!r}")
    bound, owned, arguments = mesh_arguments(
        "heat_geodesic", mesh, vertices, faces, (sources, t), 1
    )
    return _heat_geodesic(
        bound,
        owned,
        cast("wp.array[wp.int32]", arguments[0]),
        cast("float | None", arguments[1]),
        use_robust,
        method,
    )


def _heat_geodesic(
    mesh: Trimesh,
    owned: bool,
    sources: wp.array[wp.int32],
    t: float | None,
    use_robust: bool,
    method: Literal["meshlab", "igl"],
) -> wp.array[wp.float64]:
    """``heat_geodesic`` on a `Trimesh`; ``owned`` is whether the caller passed it."""
    vertices, faces = mesh.vertices, mesh.faces
    require_same_device(vertices=vertices, faces=faces, sources=sources)
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3

    if n_vertices == 0 or n_faces == 0 or sources.size == 0:
        return _launch.zeros(n_vertices, dtype=wp.float64, device=device)

    solver = mesh.heat_solver(t, use_robust=use_robust, method=method)

    # Heat solve: (M - t L) u = u0, with u0 the source indicator, run until every vertex's heat has
    # converged relative to its own size (the settle rule at ``_HEAT_CHECK_ROUNDS``) -- not to a
    # residual tolerance, which the far field sits hundreds of orders of magnitude below. Neumann on
    # a boundary, as MeshLab takes it, unless the caller asks for ``igl::heat_geodesics_solve``'s
    # average with the solution pinned to zero on the rim.
    # The indicator, the heat, the divergence's sums, the shift's means and (``"igl"``) the pinned
    # heat: scratch dropped together, one buffer.
    sizes = [n_vertices, n_vertices, n_vertices, _source_means_size(n_vertices, sources.size)]
    if method == "igl":
        sizes.append(n_vertices)
    fields, invalid = _zeroed_fields_and_flag(device, *sizes)
    u0, heat, scratch, means = fields[:4]
    _launch.launch(
        kernel_heat.seed_source_indicator,
        dim=sources.size,
        inputs=[sources],
        outputs=[u0, invalid],
        device=device,
    )
    _require_indices_in_range(invalid, n_vertices, "heat_geodesic", "sources")

    solver.diffuse(u0, heat)
    if method == "igl":
        pinned = fields[4]
        if solver.diffuse_pinned(u0, pinned):
            _launch.map(kernel_heat.average_heat, heat, pinned, out=heat)
    distance = _distance_from_heat(
        mesh, solver, sources, heat, _corner_order(mesh, solver), scratch, means
    )
    _log_discarded("heat_geodesic", owned, solver)
    return distance


FastMarchingMethod = Literal["meshlib", "meshlab"]
"""The fronts [`fast_marching_distance`][ordito.heat.fast_marching_distance] reproduces:
the edge-and-plane-wave front (``"meshlib"``) or the face-unfolding one (``"meshlab"``)."""


def fast_marching_distance(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    *,
    method: FastMarchingMethod = "meshlib",
) -> wp.array[wp.float32]:
    """
    Geodesic distance to the nearest source vertex, by a front marched across the faces.

    Each vertex's distance is the smallest a front from the sources offers it: along an edge from a
    reached neighbour, ``d(a) + |a c|``, or across a face whose other two corners ``a`` and ``b``
    are reached, as the planar wave through them (the first-order Eikonal update of fast marching)
    where that wave enters the face between ``a c`` and ``b c``. The answer is the fixpoint of the
    two rules, which a serial front reaches by popping vertices nearest first and the call reaches
    by relaxing every face against the previous round's distances until nothing moves.

    Unlike [`heat_geodesic`][ordito.heat.heat_geodesic] there is no linear solve and no diffusion
    time: the field is a first-order approximation of the geodesic distance, within about a per
    cent on well-shaped meshes and coarser across needles and obtuse faces, where the face update
    is rejected and the front falls back to edges. The default front runs in ``float32``.

    ``method`` names whose rules the front reproduces:

    - ``"meshlib"``: that library's surface-distance front, with its edge offer, its face update
      and the update's rejection tests, and its rule that an offer always lands strictly past its
      origin. Allowed to re-open a vertex 255 times, its serial front reaches this fixpoint and the
      two agree to rounding. At its default of three re-openings a vertex is left where its queue
      happened to stop, up to a few per cent of the range on irregular meshes.
    - ``"meshlab"``: that library's geodesic front (its per-vertex geodesic distance filters). An
      edge offer as above, and across a face the straight line from the virtual source that the
      two corners' distances place in the face's unfolded plane, used only where the corners were
      reached from the same source and their distances form a triangle with the edge between them;
      where the line would leave the face past a corner, the offer is that corner's edge. Computed
      in ``float64``, as the library does, and narrowed on return. From one source its serial front
      reaches this fixpoint and the two agree to rounding. From several, the library labels each
      vertex with the source whose front reached it first, which follows its queue's order; the
      call instead settles on labels each vertex's smallest offer confirms, which differ from the
      library's only next to where two sources' regions meet, by up to a few per cent of the range
      there.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    sources
        ``(n_sources,)`` source vertex indices, at distance zero.
    method
        ``"meshlib"`` or ``"meshlab"``: whose front to reproduce (above).

    Returns
    -------
    wp.array[wp.float32]
        ``(n_vertices,)`` distance field on the mesh's device; ``inf`` on a vertex no face or edge
        path connects to a source.

    Raises
    ------
    ValueError
        If an entry of ``sources`` is not a vertex index in ``[0, n_vertices)``, or ``method`` is
        not ``"meshlib"`` or ``"meshlab"``.
    RuntimeError
        If ``vertices``, ``faces`` and ``sources`` are not all on one device.

    Notes
    -----
    One face pass a round, each a pure function of the previous round's distances, so the answer
    is the same on both devices and under any thread order. A round carries the front about one ring
    further, so the number of rounds is about the mesh's diameter in edges, plus the rounds a face
    update takes to undercut an edge path already laid. On CUDA the rounds run as one conditional
    graph, with no readback between them. ``"meshlab"`` from several sources settles the rounds
    again for each relabelling, a few times, with one readback each.

    See Also
    --------
    [`heat_geodesic`][ordito.heat.heat_geodesic]
    [`shortest_path_envelope`][ordito.graph.shortest_path_envelope]

    Examples
    --------
    Pole to pole on a unit sphere the true geodesic distance is ``pi``; the front is within a
    per cent of it:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3)
    heights = vertices.numpy()[:, 2]
    north, south = int(np.argmax(heights)), int(np.argmin(heights))
    distance = od.heat.fast_marching_distance(vertices, faces, wp.array([north], dtype=wp.int32))
    print(distance.numpy()[north])  # 0.0
    print(f"{distance.numpy()[south]:.2f}")  # 3.11
    ```
    """
    require_same_device(vertices=vertices, faces=faces, sources=sources)
    if method not in ("meshlib", "meshlab"):
        raise ValueError(f"method must be 'meshlib' or 'meshlab', got {method!r}")
    if method == "meshlab":
        return _unfolding_front(vertices, faces, sources)
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    labels = _launch.full(n_vertices, math.inf, dtype=wp.float32, device=device)
    if sources.size == 0:
        return labels
    invalid = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_heat.seed_front_sources,
        dim=sources.size,
        inputs=[sources],
        outputs=[labels, invalid],
        device=device,
    )
    _require_indices_in_range(invalid, n_vertices, "fast_marching_distance", "sources")
    if n_faces == 0:
        return labels

    relaxed = _launch.clone(labels)
    # The shared round-loop word (``kernels/array.py``), its condition seeded non-zero because
    # ``wp.capture_while`` reads it before the first round. Each pair of rounds lowers at least one
    # distance or ends the loop; the cap of ``n_vertices`` pairs is insurance for a captured loop,
    # never met on a mesh (rounds track the diameter in edges).
    state = _launch.array([0, 1, 0], dtype=wp.int32, device=device)
    max_rounds = wp.int32(n_vertices)
    if wp.get_device(device).is_cpu:
        # The CPU runs a launch's threads one after another, so the whole loop runs as one
        # thread's rounds in that same order (``kernels/heat.fast_marching_rounds_serial``).
        _launch.launch(
            kernel_heat.fast_marching_rounds_serial,
            dim=1,
            inputs=[vertices, faces, max_rounds, labels, relaxed, state],
            device=device,
        )
        return labels

    def front_rounds() -> None:
        # Two rounds, ``labels`` into ``relaxed`` and back: a captured body replays the pointers it
        # recorded, so the buffers cannot swap names between bodies, and the second round leaves
        # the answer in ``labels``. A round past the fixpoint changes nothing.
        for source, target in ((labels, relaxed), (relaxed, labels)):
            _launch.launch(
                kernel_heat.fast_marching_pass,
                dim=n_faces,
                inputs=[vertices, faces, source],
                outputs=[target, state],
                device=device,
            )
        _launch.launch(kernel_array.loop_advance, dim=1, inputs=[max_rounds, state], device=device)

    run_device_loop(device, odt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW]), front_rounds)
    return labels


def _unfolding_front(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], sources: wp.array[wp.int32]
) -> wp.array[wp.float32]:
    # ``fast_marching_distance(method="meshlab")``: the round loop over the unfolding offer
    # (``kernels/heat.unfolding_pass``), in ``float64`` and narrowed at the end. One source needs
    # no labels, and its fixpoint is the serial front's answer.
    #
    # With several sources an offer unfolds only between corners of one source, and the serial
    # front's labels follow its pop order, which rounds cannot replay. The labelled rounds
    # (``unfolding_sources_pass`` after each pass) give a first labelling; then the distances are
    # re-settled from the seeds with the labels frozen, each vertex is relabelled by the origin of
    # its smallest offer, and the two repeat until no label moves: a fixpoint of both. The fixpoint
    # depends on the start (several exist), and this start lands nearest the serial front's.
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    labels = _launch.full(n_vertices, math.inf, dtype=wp.float64, device=device)
    if sources.size == 0:
        return od.array.astype(labels, wp.float32)
    several = sources.size > 1
    keys = _launch.full(n_vertices if several else 0, INT64_MAX, dtype=wp.int64, device=device)
    invalid = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_heat.seed_unfolding_sources,
        dim=sources.size,
        inputs=[sources],
        outputs=[labels, keys, invalid],
        device=device,
    )
    _require_indices_in_range(invalid, n_vertices, "fast_marching_distance", "sources")
    if n_faces == 0:
        return od.array.astype(labels, wp.float32)
    lengths = _launch.empty(n_faces, dtype=wp.vec3d, device=device)
    cosines = _launch.empty(n_faces, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_heat.unfolding_face_geometry,
        dim=n_faces,
        inputs=[vertices, faces],
        outputs=[lengths, cosines],
        device=device,
    )
    schedule = _unfolding_schedule(device, n_faces)
    vertex_faces, vertex_face_offsets = (
        od.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
        if schedule.lists
        else (faces, faces)
    )
    geometry = _UnfoldingMesh(faces, lengths, cosines, vertex_faces, vertex_face_offsets, schedule)
    lanes = schedule.lanes
    if not several:
        _unfolding_rounds(geometry, sources, labels, keys, labelled=False)
        return od.array.astype(labels, wp.float32)

    _unfolding_rounds(geometry, sources, labels, keys, labelled=True)
    seed_keys = _launch.full(n_vertices, INT64_MAX, dtype=wp.int64, device=device)
    seeded = _launch.full(n_vertices, math.inf, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_heat.seed_unfolding_sources,
        dim=sources.size,
        inputs=[sources],
        outputs=[seeded, seed_keys, invalid],
        device=device,
    )
    relabelled = _launch.empty(n_vertices, dtype=wp.int64, device=device)
    changed = _launch.empty(1, dtype=wp.int32, device=device)
    for _ in range(_UNFOLDING_RELABEL_CAP):
        _launch.copy(labels, seeded)
        _unfolding_rounds(geometry, sources, labels, keys, labelled=False)
        _launch.copy(relabelled, seed_keys)
        _launch.launch(
            kernel_heat.unfolding_argmin_sources,
            dim=lanes * n_faces,
            inputs=[faces, lengths, cosines, labels, keys, lanes],
            outputs=[relabelled],
            device=device,
        )
        changed.zero_()
        _launch.launch(
            kernel_heat.keep_or_flag_relabelled,
            dim=n_vertices,
            inputs=[keys],
            outputs=[relabelled, changed],
            device=device,
        )
        # The relabelling's stop test: one word, read once per settle, which runs a whole round
        # loop of its own.
        if not read_scalar(changed):
            break
        _launch.copy(keys, relabelled)
    return od.array.astype(labels, wp.float32)


# Settle-and-relabel passes a several-source unfolding front may take; measured stable within 3 on
# the fixtures, so the cap only bounds a labelling that cycles (the distances stay a settled
# fixpoint of the last labels either way).
_UNFOLDING_RELABEL_CAP = 16


@dataclass(frozen=True)
class _UnfoldingSchedule:
    # How a round of the unfolding front is laid out on its device. ``lists``: drive the rounds by
    # front lists (``_unfolding_front_rounds``: work follows the front, at the cost of queueing
    # it) rather than by scanning every face for an active corner (``_unfolding_rounds``).
    # ``lanes``: threads a face's six offers are spread over. ``threads``: the fixed width of a
    # list-driven launch, which strides over its list.
    lists: bool
    lanes: int
    threads: int


# Measured on one and three sources, icospheres 4-8 (2.5 k to 655 k vertices), whole call, one
# captured graph. CUDA: each offer is a long ``float64`` chain, so six lanes a face keep a round
# short. Below about 2**17 faces the scan wins: queueing a lowered vertex's faces adds atomic
# latency to every round (``icosphere(4)``: 1.37 ms against 0.97), and from there the lists win,
# since the scan's launch grows with the mesh and the front does not (one source,
# ``icosphere(7)`` / ``(8)``: 8.2 / 21.0 ms against 9.9 / 30.8; three sources level, 154 / 154 ms
# at ``(8)``). The list width barely matters (``2**17`` to ``2**21`` threads within 3 %), lanes do
# (two lanes: 12.9 / 25.1 ms). CPU: threads run one after another, so one lane and a single
# strided thread: every extra thread is a call per round whether or not it finds work
# (``2**17`` threads: 327 ms at ``icosphere(6)``; one: 73, against the scan's 163).
_UNFOLDING_LISTS_FROM_FACES_CUDA = 1 << 17
_UNFOLDING_LANES_CUDA = 6
_UNFOLDING_THREADS_CUDA = 1 << 17
_UNFOLDING_CPU = _UnfoldingSchedule(lists=True, lanes=1, threads=1)


def _unfolding_schedule(device: wp.DeviceLike, n_faces: int) -> _UnfoldingSchedule:
    # The round layout for ``n_faces`` faces on ``device`` (``_UNFOLDING_LISTS_FROM_FACES_CUDA``).
    if not wp.get_device(device).is_cuda:
        return _UNFOLDING_CPU
    lists = n_faces >= _UNFOLDING_LISTS_FROM_FACES_CUDA
    threads = min(_UNFOLDING_LANES_CUDA * n_faces, _UNFOLDING_THREADS_CUDA)
    return _UnfoldingSchedule(lists=lists, lanes=_UNFOLDING_LANES_CUDA, threads=threads)


@dataclass(frozen=True)
class _UnfoldingMesh:
    # What every round of one call reads: the faces with their ``float64`` geometry
    # (``kernels/heat.unfolding_face_geometry``), the vertex-face incidence the front lists queue
    # through (``faces`` itself as a placeholder when ``schedule.lists`` is off), and the layout.
    faces: wp.array[wp.int32]
    lengths: wp.array[wp.vec3d]
    cosines: wp.array[wp.vec3d]
    vertex_faces: wp.array[wp.int32]
    vertex_face_offsets: wp.array[wp.int32]
    schedule: _UnfoldingSchedule


def _unfolding_rounds(
    geometry: _UnfoldingMesh,
    sources: wp.array[wp.int32],
    labels: wp.array[wp.float64],
    keys: wp.array[wp.int64],
    *,
    labelled: bool,
) -> None:
    # Relax ``labels`` to the unfolding fixpoint in place, as ``fast_marching_distance``'s loop
    # does, with ``keys`` frozen or (``labelled``) re-keyed each round by the source that lowered
    # each vertex, double-buffered like the distances. ``labels`` holds the seeded distances of
    # ``sources`` on entry: each vertex's round stamp starts from them, so the first round runs
    # only the faces around the sources and each later one only the faces at the moving front.
    if geometry.schedule.lists:
        _unfolding_front_rounds(geometry, sources, labels, keys, labelled=labelled)
        return
    faces, lengths, cosines = geometry.faces, geometry.lengths, geometry.cosines
    device = labels.device
    n_faces = faces.size // 3
    lanes = geometry.schedule.lanes
    lowered_round = _launch.full(
        labels.size, kernel_heat.UNFOLDING_NEVER_LOWERED, dtype=wp.int32, device=device
    )
    _launch.launch(
        kernel_heat.mark_seed_rounds,
        dim=sources.size,
        inputs=[sources],
        outputs=[lowered_round],
        device=device,
    )
    relaxed = _launch.clone(labels)
    relabelled = _launch.clone(keys) if labelled else keys
    state = _launch.array([0, 1, 0], dtype=wp.int32, device=device)
    max_rounds = wp.int32(labels.size)

    def front_round() -> None:
        _launch.launch(
            kernel_heat.unfolding_pass,
            dim=lanes * n_faces,
            inputs=[faces, lengths, cosines, labels, keys, lanes],
            outputs=[relaxed, lowered_round, state],
            device=device,
        )
        if labelled:
            _launch.launch(
                kernel_heat.unfolding_sources_pass,
                dim=lanes * n_faces,
                inputs=[
                    faces,
                    lengths,
                    cosines,
                    labels,
                    keys,
                    relaxed,
                    lowered_round,
                    state,
                    lanes,
                ],
                outputs=[relabelled],
                device=device,
            )
            _launch.copy(keys, relabelled)
        _launch.copy(labels, relaxed)
        _launch.launch(kernel_array.loop_advance, dim=1, inputs=[max_rounds, state], device=device)

    run_device_loop(device, odt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW]), front_round)


def _unfolding_front_rounds(
    geometry: _UnfoldingMesh,
    sources: wp.array[wp.int32],
    labels: wp.array[wp.float64],
    keys: wp.array[wp.int64],
    *,
    labelled: bool,
) -> None:
    # ``_unfolding_rounds`` driven by front lists: each round's passes stride over the faces
    # around the vertices the previous round lowered (``kernels/heat.unfolding_front_pass``, which
    # also queues them), and ``copy_front_vertices`` copies back only the vertices it lowered.
    faces, lengths, cosines = geometry.faces, geometry.lengths, geometry.cosines
    vertex_faces, vertex_face_offsets = geometry.vertex_faces, geometry.vertex_face_offsets
    device = labels.device
    n_vertices = labels.size
    n_faces = faces.size // 3
    lanes = geometry.schedule.lanes
    threads = geometry.schedule.threads
    vertex_threads = min(n_vertices, threads)
    never = kernel_heat.UNFOLDING_NEVER_LOWERED
    lowered_round = _launch.full(n_vertices, never, dtype=wp.int32, device=device)
    face_round = _launch.full(n_faces, never, dtype=wp.int32, device=device)
    front_vertices = _launch.empty(2 * n_vertices, dtype=wp.int32, device=device)
    front_faces = _launch.empty(2 * n_faces, dtype=wp.int32, device=device)
    front_counts = _launch.zeros(4, dtype=wp.int32, device=device)
    relaxed = _launch.clone(labels)
    relabelled = _launch.clone(keys) if labelled else keys
    # The sources, as round ``-1``'s lowered vertices, queue round 0's faces.
    _launch.launch(
        kernel_heat.queue_source_faces,
        dim=sources.size,
        inputs=[sources, vertex_faces, vertex_face_offsets],
        outputs=[lowered_round, face_round, front_faces, front_counts],
        device=device,
    )
    state = _launch.array([0, 1, 0], dtype=wp.int32, device=device)
    max_rounds = wp.int32(n_vertices)
    if wp.get_device(device).is_cpu:
        # The CPU runs a launch's threads one after another, so the whole loop runs as one
        # thread's rounds in that same order (``kernels/heat.unfolding_front_rounds_serial``).
        _launch.launch(
            kernel_heat.unfolding_front_rounds_serial,
            dim=1,
            inputs=[
                faces,
                lengths,
                cosines,
                vertex_faces,
                vertex_face_offsets,
                lanes,
                threads,
                vertex_threads,
                wp.int32(labelled),
                max_rounds,
                labels,
                keys,
                relaxed,
                relabelled,
                lowered_round,
                face_round,
                front_vertices,
                front_faces,
                front_counts,
                state,
            ],
            device=device,
        )
        return

    def front_round() -> None:
        _launch.launch(
            kernel_heat.unfolding_front_pass,
            dim=threads,
            inputs=[
                faces,
                lengths,
                cosines,
                labels,
                keys,
                vertex_faces,
                vertex_face_offsets,
                lanes,
                threads,
            ],
            outputs=[
                relaxed,
                lowered_round,
                face_round,
                front_vertices,
                front_faces,
                front_counts,
                state,
            ],
            device=device,
        )
        if labelled:
            _launch.launch(
                kernel_heat.unfolding_front_sources_pass,
                dim=threads,
                inputs=[
                    faces,
                    lengths,
                    cosines,
                    labels,
                    keys,
                    relaxed,
                    front_faces,
                    front_counts,
                    state,
                    lanes,
                    threads,
                ],
                outputs=[relabelled],
                device=device,
            )
        _launch.launch(
            kernel_heat.copy_front_vertices,
            dim=vertex_threads,
            inputs=[front_vertices, relaxed, relabelled, state, vertex_threads],
            outputs=[labels, keys, front_counts],
            device=device,
        )
        _launch.launch(kernel_array.loop_advance, dim=1, inputs=[max_rounds, state], device=device)

    run_device_loop(device, odt.as_dense(state[kernel_array.LOOP_CONDITION_VIEW]), front_round)


# --------------------------------------------------------------------------------------
# The signed heat method: signed distance to a set of oriented curves
# --------------------------------------------------------------------------------------


SignedHeatMethod = Literal["vertex_connection", "potpourri3d"]
"""The signed heat method discretizations [`heat_signed_distance`][ordito.heat.heat_signed_distance]
offers: the field on vertices, or geometry-central's on edges."""


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
    method: SignedHeatMethod = "vertex_connection",
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
    method: SignedHeatMethod = "vertex_connection",
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
    method: SignedHeatMethod = "vertex_connection",
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

    That is the default ``method="vertex_connection"``, the field on the vertices.
    ``method="potpourri3d"`` is geometry-central's own discretization (Feng and Crane), which
    ``potpourri3d`` runs and which this reproduces vertex for vertex: the field lives on the edges,
    one value per edge in that edge's frame (Crouzeix-Raviart elements), a segment's normal is
    placed on the edge it lies on, and each face samples its three edges before the Poisson solve.

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
        Diffusion time; defaults to the squared mean unique-edge length, or a quarter of it (the
        squared half-length, geometry-central's default) at ``method="potpourri3d"``, either
        lengthened on a mesh too fine for it (see Notes and
        [`HeatSolver`][ordito.heat.HeatSolver]). Larger values smooth the field.
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
        modes ``potpourri3d`` offers under the same names. At ``method="potpourri3d"`` the
        mean is the curve's, its segments weighted by length, as geometry-central takes it.
    method
        Whose discretization of the signed heat method to run:

        - ``"vertex_connection"`` (default): the tangent field on the vertices, diffused by the
          vertex connection Laplacian.
        - ``"potpourri3d"``: the field on the edges, diffused by the Crouzeix-Raviart
          connection Laplacian, ``potpourri3d.MeshSignedHeatSolver``'s answer. Only segments that
          are mesh edges enter its source; one joining two vertices with no edge between them adds
          nothing. On a `Trimesh` the edge system and its factorization are kept with the mesh's
          other heat operators. Its default ``t`` fails on fine meshes, as potpourri3d's does
          (see Notes).

    Returns
    -------
    wp.array[wp.float64]
        ``(n_vertices,)`` signed distance field on the mesh's device.

    Raises
    ------
    ValueError
        If ``level_set_constraint`` is not ``"zero_set"`` or ``"none"``, ``method`` is not
        ``"vertex_connection"`` or ``"potpourri3d"``, or an entry of ``curve_vertices`` is not
        a vertex index in ``[0, n_vertices)``.
    RuntimeError
        If ``vertices``, ``faces``, ``curve_vertices`` and ``curve_offsets`` are not all on one
        device.

    Warns
    -----
    UserWarning
        At ``method="potpourri3d"``, when the diffused field fell below the ``float64`` range on
        part of the mesh (see Notes).

    Notes
    -----
    !!! warning "potpourri3d's signed heat method is wrong far from the curve on fine meshes"
        geometry-central diffuses for a much shorter time, ``(h / 2) ** 2``, and over twice as
        many steps of the mesh (edge midpoints rather than vertices), so its field decays
        steeply with distance from the curve. On a fine enough mesh -- a few hundred thousand
        faces for a curve on one side of a sphere -- it falls below the smallest ``float64``
        before it has crossed the surface, and the distance past that point is wrong by a large
        fraction of its range. ``potpourri3d.MeshSignedHeatSolver`` is wrong on such meshes, and
        on smaller ones (its direct solve loses the far field to round-off first): no solver can
        carry values the format cannot hold. ``method="potpourri3d"`` therefore departs from it
        there: when ``t`` is not given and the mesh's bounding-box diagonal spans more than 400
        of the method's diffusion lengths ``sqrt(t)``, the default time grows until it spans
        400, which keeps the field in range and the answer right; below that the time is
        geometry-central's and the answer potpourri3d's. The default method's time grows the
        same way on meshes about four times finer (see
        [`HeatSolver`][ordito.heat.HeatSolver]). Where the box understates the distance from the
        curve (a surface folded on itself), or for a ``t`` passed explicitly, the field can still
        fall out of range, and the call warns. Prefer the default
        ``method="vertex_connection"``, which is more accurate at every resolution measured.

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
    on_edges = od.heat.heat_signed_distance(vertices, faces, curve, method="potpourri3d")
    print(f"{on_edges.numpy()[60]:.3f}")  # 0.303
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
        method=method,
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
    method: str,
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
    if method not in ("vertex_connection", "potpourri3d"):
        raise ValueError(f'method must be "vertex_connection" or "potpourri3d", got {method!r}')
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    if n_vertices == 0 or n_faces == 0 or curve_vertices.size == 0:
        return _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    if method == "potpourri3d":
        field = _edge_signed_distance(
            mesh,
            curve_vertices,
            curve_offsets,
            t,
            closed=closed,
            zero_set=level_set_constraint == "zero_set",
        )
        _log_discarded("heat_signed_distance", owned, mesh.heat_solver(t))
        return field

    solver = mesh.heat_solver(t)
    _, scalar, frames, _ = solver.vector_operators
    basis_x, basis_y, vertex_normals = frames
    poisson_system = scalar[3]
    cot_entries, face_normals = scalar[5], scalar[6]

    # Stage 1: splat each segment's normal onto its endpoints, one thread per curve entry, each
    # finding its own curve in the offsets on the device.
    # The field, its diffusion and the divergence's sums: scratch dropped together, one buffer.
    (source_rows, diffused_rows, scratch), invalid = _zeroed_fields_and_flag(
        device, 2 * n_vertices, 2 * n_vertices, n_vertices
    )
    source = _pairs(source_rows, n_vertices)
    # The zero set's pinned vertices, marked by the splat's own threads.
    zero_set = level_set_constraint == "zero_set"
    fixed_mask = _launch.zeros(n_vertices if zero_set else 0, dtype=wp.bool, device=device)
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
        outputs=[source, fixed_mask, invalid],
        device=device,
    )
    _require_indices_in_range(invalid, n_vertices, "heat_signed_distance", "curve_vertices")

    # Stage 2: diffuse the tangent field; only its direction is kept. No absolute floor may decide
    # which vectors vanished -- this field carries the mesh's scale, and one zeroes most of it on a
    # mesh not near unit scale (confirmed: 111 of 162 vertices at a 1e-6 scale).
    diffused = _pairs(diffused_rows, n_vertices)
    solver.diffuse_tangent(source, diffused)
    # The divergence kernel normalizes each corner without underflow, zero only where the field is
    # exactly zero: it is converged per vertex (the settle rule at ``_HEAT_CHECK_ROUNDS``), so its
    # far field is a direction however small.

    # Stage 3: integrate the unit field back into a scalar with a Poisson solve. The cotangent
    # weights and face normals come from the same bundle, so the Poisson stage and the diffusion
    # cannot drift apart.
    order = _corner_order(mesh, solver)
    divergence, corner_divergence = _corner_outputs(
        order, n_vertices, n_faces, wp.float64, device, sums=scratch
    )
    _launch.launch(
        kernel_heat.vertex_field_divergence,
        dim=n_faces,
        inputs=[vertices, faces, face_normals, diffused, basis_x, basis_y, cot_entries],
        outputs=[divergence, corner_divergence],
        device=device,
    )
    _sum_corner_outputs(order, corner_divergence, divergence)

    if zero_set:
        field = _solve_poisson_zero_set(
            poisson_system, divergence, fixed_mask, vertices, solver.ill_conditioned
        )
    else:
        field = _solve_poisson_shifted(solver, divergence, curve_vertices, vertices)
    _log_discarded("heat_signed_distance", owned, solver)
    return field


def _solve_poisson_zero_set(
    operator: odt.BsrMatrix[wp.float64],
    divergence: wp.array[wp.float64],
    fixed_mask: wp.array[wp.bool],
    vertices: wp.array[wp.vec3],
    factor: bool,
) -> wp.array[wp.float64]:
    """
    Solve the Poisson problem with ``fixed_mask``'s vertices (the curve) pinned to zero.

    By a factorization when ``factor``.

    ``linalg.min_quad_with_fixed`` minimizes ``0.5 x' Q x`` and has no place for a linear term, so
    the pieces underneath it are used directly: the free partition and the free-free block come
    from [`partitioned_interior_system`][ordito.linalg.partitioned_interior_system] (whose own
    right-hand side is zero here, the pinned values being zero) and the divergence is compacted
    into it.
    """
    device = vertices.device
    n_vertices = vertices.size
    # No pinned-value right-hand side to assemble: the curve is pinned to zero, so ``-Q_ub bc``
    # vanishes and the extraction is asked for none. The system's one right-hand side is the
    # divergence, compacted to the free rows by the extraction's own row pass (``load``).
    # ``divergence`` is already the -div right-hand side the -L operator takes.
    no_values = odt.empty_2d((0, n_vertices), wp.float64, device=device)
    operator_uu, rhs, free_map, n_free = twl.partitioned_interior_system(
        operator,
        fixed_mask,
        no_values,
        load=odt.as_array2d(divergence.reshape((1, n_vertices)), wp.float64),
    )
    if n_free == 0:
        return _launch.zeros(n_vertices, dtype=wp.float64, device=device)

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


def _edge_signed_distance(
    mesh: Trimesh,
    curve_vertices: wp.array[wp.int32],
    curve_offsets: wp.array[wp.int32] | None,
    t: float | None,
    *,
    closed: bool,
    zero_set: bool,
) -> wp.array[wp.float64]:
    """
    ``heat_signed_distance(method="potpourri3d")``: the field diffused on the mesh's edges.

    The source is each curve segment's normal on the edge it lies on, diffused by the
    Crouzeix-Raviart vector heat system
    ([`HeatSolver.diffuse_edge_field`][ordito.heat.HeatSolver.diffuse_edge_field]); each face
    samples its three edges and normalizes, and the Poisson stage is the vertex method's, with the
    unconstrained field shifted by its mean along the curve's segments (the midpoint rule,
    length-weighted) rather than over its vertices.
    """
    vertices, faces = mesh.vertices, mesh.faces
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    n_entries = curve_vertices.size
    solver = mesh.heat_solver(t)
    operators = solver.operators
    edges = mesh.edges_unique
    face_edges = mesh.faces_unique_edges

    single_curve = wp.int32(1 if curve_offsets is None else 0)
    closed_flag = wp.int32(1 if closed else 0)
    # Word 0 flags an out-of-range curve index, word 1 a diffused field that ran into underflow
    # (``kernels/heat.EDGE_FIELD_UNDERFLOW``).
    invalid = _launch.zeros(2, dtype=wp.int32, device=device)
    fixed_mask = _launch.zeros(n_vertices if zero_set else 0, dtype=wp.bool, device=device)
    lengths = _launch.zeros(0 if zero_set else n_entries, dtype=wp.float64, device=device)
    source = _launch.group_sums(edges.shape[0], wp.vec2d, n_entries, device)
    _launch.launch_group_sums(
        kernel_heat.splat_curve_edges,
        dim=n_entries,
        inputs=[vertices, curve_vertices, curve_offsets, single_curve, closed_flag, edges],
        sums=source,
        outputs=[fixed_mask, lengths, invalid],
        device=device,
    )
    _require_indices_in_range(invalid, n_vertices, "heat_signed_distance", "curve_vertices")
    rhs = cast("wp.array[wp.vec2d]", _launch.finish_group_sums(source))

    diffused = _launch.zeros(edges.shape[0], dtype=wp.vec2d, device=device)
    solver.diffuse_edge_field(rhs, diffused)

    order = _corner_order(mesh, solver)
    divergence, corner_divergence = _corner_outputs(order, n_vertices, n_faces, wp.float64, device)
    _launch.launch(
        kernel_heat.edge_field_divergence,
        dim=n_faces,
        inputs=[vertices, faces, face_edges, edges, diffused, operators[5]],
        outputs=[divergence, corner_divergence, _launch.view(invalid, 1, 2)],
        device=device,
    )
    _sum_corner_outputs(order, corner_divergence, divergence)

    if zero_set:
        field = _solve_poisson_zero_set(
            operators[3], divergence, fixed_mask, vertices, solver.ill_conditioned
        )
    else:
        field = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
        solver.solve_poisson(divergence, field)
        blocks = kernel_reduce.blocks_1d(n_entries)
        sums, stride = _launch.block_partials(blocks, 2, wp.float64, device)
        _launch.launch_tiled(
            kernel_heat.curve_midpoint_sums,
            dim=[blocks],
            inputs=[
                field,
                curve_vertices,
                curve_offsets,
                single_curve,
                closed_flag,
                lengths,
                stride,
            ],
            outputs=[sums],
            block_dim=TILE_1D,
            device=device,
        )
        _launch.launch(
            kernel_heat.shift_by_mean, dim=n_vertices, inputs=[sums, field], device=device
        )
    # Host readback: whether to warn that the field underflowed. Read once everything is queued, so
    # it drains the queue once rather than stalling the Poisson stage behind it.
    if read_scalar(invalid, 1) != 0:
        warnings.warn(
            'heat_signed_distance(method="potpourri3d"): the diffused field fell below the '
            "float64 range before it reached the whole mesh, so the distance is wrong far from "
            "the curve. The method inherits this from potpourri3d, whose short diffusion time "
            "(h / 2) ** 2 makes the heat decay too fast for a fine mesh "
            "(potpourri3d.MeshSignedHeatSolver is wrong there too); the default time lengthens on "
            "such meshes, but not enough on this one, or t was passed. Pass a larger t, or use "
            'the default method="vertex_connection", which is more accurate.',
            stacklevel=4,
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
    transport_angles: wp.array[wp.float64] | None = None,
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
    radius is asserted to *be* the [`heat_geodesic`][ordito.heat.heat_geodesic] distance at that
    ``t``, so two diffusion times would split a quantity that is supposed to be one number. That
    is why ``scalar_operators`` is the one argument here that cannot be validated.

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
    # The one-rings the connection is formed over, kept for the frames when it is formed here.
    rings = None
    if n_faces == 0:
        pattern = None
    elif pattern is None:
        pattern = mesh_operator_pattern(faces, n_vertices)
    # A bundle ``_heat_operators`` built carries the mass and timestep sums it assembled from.
    built = None if scalar_operators is None else _SCALAR_BUILDS.get(scalar_operators[0])
    edge_sums = None
    if t is None:
        # Shares the scalar solver's timestep convention -- the unique-edge mean,
        # geometry-central's. The two solvers must agree: ``log_map``'s radius is asserted to *be*
        # the ``heat_geodesic`` distance at that timestep, so giving them different diffusion
        # times would split a quantity that is supposed to be one number. They do by construction:
        # the mean is read off the shared sparsity, twelve triplets per face and nothing pruned,
        # and the scalar bundle built here takes these very sums (a prebuilt one already holds its
        # own). Both systems square it on the device, so nothing is read back.
        # A bundle assembled below reduces them itself, into the allocation its mass shares.
        if built is not None and built.edge_sums is not None:
            edge_sums = built.edge_sums
        elif scalar_operators is not None:
            edge_sums = _edge_length_sums(vertices, pattern)

    vector_values = None
    if scalar_operators is None:
        # The scalar bundle is assembled here, so its row kernel writes the vector system's rows
        # too, from the transport angles formed first.
        if pattern is not None and transport_angles is None:
            transport_angles, rings = _transport_angles_and_rings(vertices, faces)
        scalar_operators, mass, vector_values = _heat_operators(
            vertices,
            faces,
            t,
            edge_sums=edge_sums,
            pattern=pattern,
            transport_angles=transport_angles,
        )
        built = _SCALAR_BUILDS[scalar_operators[0]]
        edge_sums = built.edge_sums
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
        # thread that forms them, so the connection Laplacian itself is never stored. Its weights
        # are ``cotmatrix_entries``': the ``float32`` table a bundle assembled from positions
        # already holds (the ``float64`` one is that table widened on write, and the row kernel
        # widens on read), formed here for any other bundle.
        _, _, offsets, columns, run_start, keys, count, order = pattern
        if vector_values is None:
            cot_entries = (
                scalar_operators[5]
                if built is not None and built.extrinsic_cot
                else cotmatrix_entries(vertices, faces, dtype=wp.float64)
            )
            if transport_angles is None:
                transport_angles, rings = _transport_angles_and_rings(vertices, faces)
            vector_values = _launch.empty(columns.size, dtype=wp.mat22d, device=device)
            _launch.launch(
                kernel_heat.VECTOR_HEAT_SYSTEM_ROWS[cot_entries.dtype],
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
        frames = vertex_tangent_frames(vertices, faces, rings=rings)
    preconditioner = twl.jacobi_preconditioner(vector_system)
    return vector_system, scalar_operators, frames, preconditioner


def _transport_angles_and_rings(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.float64], tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]]]:
    """
    ``halfedge_transport_angles(vertices, faces)``, and the one-rings it walked for the frames.

    The rings and the twins come from one ``vertex_one_rings`` build, which is the table
    ``halfedge_transport_angles`` would form itself; the transport kernel is launched here because
    the public function, handed those twins back, verifies them with a launch and a readback that a
    table this build produced cannot fail.
    """
    ring_halfedges, ring_offsets, is_boundary, twins = vertex_one_rings(
        faces, n_vertices=vertices.size, return_twins=True
    )
    rings = (ring_halfedges, ring_offsets, is_boundary)
    tangent_angles = halfedge_tangent_angles(vertices, faces, rings=rings)
    transport_angles = _launch.empty(twins.size, dtype=wp.float64, device=vertices.device)
    _launch.launch(
        kernel_tangent_space.halfedge_transport_angles,
        dim=twins.size,
        inputs=[twins, tangent_angles, transport_angles],
        device=vertices.device,
    )
    return transport_angles, rings


def _require_cotmatrix_pattern(pattern: MeshOperatorPattern | None, n_vertices: int) -> None:
    """Raise unless a caller's ``pattern`` serves the cotangent Laplacian over ``n_vertices``."""
    if pattern is not None and (
        pattern.operator != "cotmatrix" or pattern.n_vertices != n_vertices
    ):
        raise ValueError(
            f"pattern was built for operator={pattern.operator!r} over {pattern.n_vertices} "
            f"vertices; the heat operators need operator='cotmatrix' over {n_vertices}"
        )


def _heat_method_choices(
    method: str,
) -> tuple[Literal["barycentric", "voronoi"], Literal["unique_edges", "face_edges"]]:
    """Return ``method``'s ``(mass, timestep)``, raising unless it names a heat discretization."""
    if method == "potpourri3d":
        return "barycentric", "unique_edges"
    if method == "meshlab":
        return "barycentric", "face_edges"
    if method == "igl":
        return "voronoi", "unique_edges"
    raise ValueError(f"method must be 'potpourri3d', 'meshlab' or 'igl', got {method!r}")


def _heat_operators(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    t: float | None,
    *,
    use_robust: bool = False,
    cot_entries: odt.Array2dFloat | None = None,
    edge_sums: wp.array[wp.float64] | None = None,
    pattern: MeshOperatorPattern | None = None,
    mass: Literal["barycentric", "voronoi"] = "barycentric",
    timestep: Literal["unique_edges", "face_edges"] = "unique_edges",
    transport_angles: wp.array[wp.float64] | None = None,
) -> tuple[HeatOperators, wp.array[wp.float64], wp.array[wp.mat22d] | None]:
    """
    ``heat_operators``' assembly, returning the ``float64`` lumped mass it built alongside.

    ``vector_heat_operators`` reuses the mass for its own system, and hands over the edge-length
    sums it already reduced for the default timestep as ``edge_sums``: the connection Laplacian's
    sparsity is the cotangent Laplacian's, so the sums are the ones this would reduce -- and the
    ``pattern`` it built them over, so the sparsity is built once for both. Given its
    ``transport_angles``, the row kernel also writes the vector system's values over the pattern,
    returned third (``None`` otherwise, or for a mesh with no faces).
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
    device = vertices.device
    n_faces = faces.size // 3
    n_vertices = vertices.size
    write_cot = cot_entries is None
    if n_faces > 0:
        if cot_entries is not None and int(cot_entries.shape[0]) != n_faces:
            raise ValueError(
                f"cot_entries must have shape (n_faces, 3) = ({n_faces}, 3), "
                f"got {(int(cot_entries.shape[0]), int(cot_entries.shape[1]))}"
            )
        if pattern is None:
            pattern = mesh_operator_pattern(faces, n_vertices)
    else:
        pattern = None
    # The lumped mass's area sums and, for the default timestep, the unique-edge sums reduced
    # into them below: both zeroed, both held by the bundle, so one allocation. The sums are left
    # on the device for the system's assembly to square.
    sum_blocks = 0
    if t is None and edge_sums is None and pattern is not None and pattern.n_vertices > 0:
        sum_blocks = kernel_reduce.blocks_1d(pattern.n_vertices)
    lumped, sum_and_count = _launch.carve(
        wp.float64,
        (n_vertices, _launch.block_partials_size(sum_blocks, 2) if sum_blocks else 0),
        device,
        zeroed=True,
    )
    # Face normals and areas (float32) for the gradient, the half-cotangent table when the caller
    # has none, and the lumped mass's exact float64 area sums, in one pass over the faces. The
    # per-face tables, the three operators' values over the pattern and the vector system's
    # values when formed here are one uninitialized allocation: the bundle holds all of them.
    n_slots = 0 if pattern is None else pattern.columns.size
    parts: list[tuple[Any, int]] = [(wp.float32, (7 if write_cot else 4) * n_faces)]
    if n_faces > 0:
        parts.append((wp.float64, 3 * n_slots))
        if transport_angles is not None:
            parts.append((wp.mat22d, n_slots))
    tables = _launch.carve_typed(parts, device)
    face_terms = tables[0]
    areas = _face_table(face_terms, 0, n_faces, 1)
    normals = _face_table(face_terms, n_faces, n_faces, 3).reshape((n_faces, 3)).view(wp.vec3)
    if cot_entries is None:
        cot_entries = odt.as_array2d(
            _face_table(face_terms, 4 * n_faces, n_faces, 3).reshape((n_faces, 3)), wp.float32
        )
    if n_faces > 0:
        _launch.launch(
            kernel_heat.heat_face_terms,
            dim=n_faces,
            inputs=[vertices, faces, wp.int32(write_cot), wp.int32(mass == "voronoi")],
            outputs=[
                cot_entries if write_cot else odt.empty_2d((0, 3), wp.float32, device=device),
                normals,
                areas,
                lumped,
            ],
            device=device,
        )
    # The cotangent Laplacian, the heat system ``M - t L`` and the Poisson operator ``-L``, all
    # over the one pattern (which stores every referenced vertex's diagonal) and written by one row
    # kernel (``kernels/heat.heat_system_rows``), which casts the half-cotangent weights to
    # ``float64`` as it reads them and forms the lumped mass from the area sums in the same pass.
    # The two preconditioners are mesh-only, so they belong here rather than in every solve. See
    # Notes.
    vector_values = None
    if pattern is None:
        laplacian = od.array.empty_square_bsr(n_vertices, wp.float64, device)
        heat_values = _launch.empty_like(laplacian.values)
        poisson_values = _launch.empty_like(laplacian.values)
        if t is None and edge_sums is None:
            edge_sums = _edge_length_sums(vertices, None)
    else:
        if t is None and edge_sums is None:
            # The unique-edge average, which is what ``igl::heat_geodesics`` uses for its timestep
            # -- read off the Laplacian's own sparsity, which already holds the unique edges.
            edge_sums = _edge_length_sums(vertices, pattern, timestep, sum_and_count)
        _, _, offsets, columns, run_start, keys, count, order = pattern
        # The three operators' values: the bundle holds all three for as long as any of them.
        values = tables[1]
        laplacian_values = _launch.view(values, 0, n_slots)
        heat_values = _launch.view(values, n_slots, 2 * n_slots)
        poisson_values = _launch.view(values, 2 * n_slots, 3 * n_slots)
        if transport_angles is not None:
            vector_values = tables[2]
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
                lumped,
                wp.float64(1.0 if mass == "voronoi" else 3.0),
                faces,
                transport_angles,
            ],
            # The lumped mass over its own area sums, in place.
            outputs=[laplacian_values, heat_values, poisson_values, lumped, vector_values],
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

    heat_system = od.array.bsr_with_values(laplacian, heat_values)
    poisson_system = od.array.bsr_with_values(laplacian, poisson_values)
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
    _SCALAR_BUILDS[heat_system] = _ScalarBuild(lumped, edge_sums, write_cot and not use_robust)
    return operators, lumped, vector_values


def _face_table(
    buffer: wp.array[wp.float32], start: int, n_faces: int, width: int
) -> wp.array[wp.float32]:
    """``width * n_faces`` entries of ``_heat_operators``' face buffer from entry ``start``."""
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.float32, device=buffer.device)
    return _launch.view(buffer, start, start + width * n_faces)


@dataclass(frozen=True)
class _ScalarBuild:
    """What ``_heat_operators`` assembled a bundle from that ``vector_heat_operators`` reuses."""

    mass: wp.array[wp.float64]
    edge_sums: wp.array[wp.float64] | None
    # Whether the bundle's half-cotangent table is ``cotmatrix_entries``' of the positions (formed
    # by ``heat_face_terms``), not a caller's or a mollified one.
    extrinsic_cot: bool


# The face buffer's stable sort ``(sorted_vertices, corner_order)``, or ``None`` for the per-face
# scatter (``_corner_order``).
_CornerOrder = tuple[wp.array[wp.int32], wp.array[wp.int32]] | None
# Each solver's face sort, keyed weakly so it lives exactly as long as the solver -- the mesh's
# lifetime -- as its operators do.
_CORNER_ORDERS: weakref.WeakKeyDictionary[
    HeatSolver, tuple[wp.array[wp.int32], wp.array[wp.int32]]
] = weakref.WeakKeyDictionary()
_T = TypeVar("_T")

# Each bundle's ``_ScalarBuild``, keyed weakly by its heat system, so it lives exactly as long as
# the bundle does: a prebuilt bundle handed to ``vector_heat_operators`` (a `Trimesh`'s cached
# ``heat_operators``) gives up its ``float64`` mass and timestep sums rather than having them
# rebuilt. ``edge_sums`` is ``None`` for a bundle built at an explicit ``t``.
_SCALAR_BUILDS: weakref.WeakKeyDictionary[odt.SparseMatrix, _ScalarBuild] = (
    weakref.WeakKeyDictionary()
)


def _edge_length_sums(
    vertices: wp.array[wp.vec3],
    pattern: MeshOperatorPattern | None,
    timestep: Literal["unique_edges", "face_edges"] = "unique_edges",
    sum_and_count: wp.array[wp.float64] | None = None,
) -> wp.array[wp.float64]:
    """
    Sum and count of the unique-edge lengths, read off a mesh operator pattern.

    At ``timestep="face_edges"`` each edge counts once per face holding it instead, the sum and
    count over every face's three edges.

    The strict upper triangle of the heat method's operators' pattern is the mesh's unique edge
    set, so the default timestep costs one launch over a pattern that already exists, where
    [`mean_unique_edge_length`][ordito.edges.mean_unique_edge_length] would re-sort every edge of
    the mesh to recover the same set. The two sums stay on the device, where
    ``kernels/linalg.shift_factor`` squares their mean -- ``0`` for a mesh with no edges (``None``
    for the pattern of a mesh with no faces), as that function returns. ``sum_and_count``, when
    given, is the zeroed ``block_partials_size(blocks_1d(pattern.n_vertices), 2)`` buffer the
    sums go into (allocated here otherwise).
    """
    device = vertices.device
    if pattern is None or pattern.n_vertices == 0:
        return _launch.zeros(2, dtype=wp.float64, device=device)
    n_rows = pattern.n_vertices
    _, _, offsets, columns, run_start, keys, count, _ = pattern
    # Block rows added in a fixed order, so the timestep repeats bit for bit on CUDA.
    blocks = kernel_reduce.blocks_1d(n_rows)
    out: wp.array[wp.float64] = (
        _launch.block_partials(blocks, 2, wp.float64, device)[0]
        if sum_and_count is None
        else sum_and_count
    )
    _launch.launch_tiled(
        kernel_heat.upper_edge_length_sum_and_count,
        dim=[blocks],
        inputs=[
            offsets,
            columns,
            run_start,
            keys,
            wp.int32(count),
            vertices,
            wp.int32(timestep == "face_edges"),
            2,
            out,
        ],
        block_dim=TILE_1D,
        device=device,
    )
    return out


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
        Diffusion time; defaults to the squared mean edge length, lengthened on a mesh too fine
        for it (see [`HeatSolver`][ordito.heat.HeatSolver]).

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
    (rhs_rows, diffused_rows), invalid = _zeroed_fields_and_flag(
        device, 2 * n_vertices, 2 * n_vertices
    )
    rhs = odt.as_array2d(rhs_rows.reshape((2, n_vertices)), wp.float64)
    # ``(min, -max, -overshoot)`` of the extension against the source values; see the kernels.
    value_range = _launch.full(3, math.inf, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_heat.seed_source_scalars,
        dim=n_sources,
        inputs=[sources, values],
        outputs=[
            _launch.view(rhs_rows, 0, n_vertices),
            _launch.view(rhs_rows, n_vertices, 2 * n_vertices),
            value_range,
            invalid,
        ],
        device=device,
    )
    _require_indices_in_range(invalid, n_vertices, "extend_scalar", "sources")
    diffused = odt.as_array2d(diffused_rows.reshape((2, n_vertices)), wp.float64)
    solver.diffuse(rhs, diffused)
    # Converged per vertex, so only an exactly zero indicator -- a component no source reaches --
    # has no value to extend.
    extended = _launch.empty(n_vertices, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_heat.extend_from_ratio,
        dim=n_vertices,
        inputs=[
            _launch.view(diffused_rows, 0, n_vertices),
            _launch.view(diffused_rows, n_vertices, 2 * n_vertices),
        ],
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
        Diffusion time; defaults to the squared mean edge length, lengthened on a mesh too fine
        for it (see [`HeatSolver`][ordito.heat.HeatSolver]).

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
    rows = 4 * n_vertices
    (rhs, diffused), invalid = _zeroed_fields_and_flag(device, rows, rows)
    _launch.launch(
        kernel_heat.seed_transport_sources,
        dim=n_sources,
        inputs=[sources, vectors],
        outputs=[
            _pairs(rhs, n_vertices),
            _launch.view(rhs, 2 * n_vertices, 3 * n_vertices),
            _launch.view(rhs, 3 * n_vertices, rows),
            invalid,
        ],
        device=device,
    )
    _require_indices_in_range(invalid, n_vertices, "transport_tangent_vectors", "sources")
    direction = _pairs(diffused, n_vertices)
    solver.diffuse_stacked(rhs, diffused, 2)
    diffused_indicator = _launch.view(diffused, 2 * n_vertices, 3 * n_vertices)
    diffused_magnitudes = _launch.view(diffused, 3 * n_vertices, rows)

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


LogMapMethod = Literal["heat_geodesic", "potpourri3d"]
"""How [`log_map`][ordito.heat.log_map] forms its radius and radial direction."""


@overload
def log_map(
    mesh: Trimesh, source: int, /, *, t: float | None = None, method: LogMapMethod = "heat_geodesic"
) -> wp.array[wp.vec2]: ...
@overload
def log_map(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    source: int,
    /,
    t: float | None = None,
    *,
    method: LogMapMethod = "heat_geodesic",
) -> wp.array[wp.vec2]: ...
def log_map(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | int | None = None,
    source: int | None = None,
    t: float | None = None,
    *,
    vertices: wp.array[wp.vec3] | None = None,
    method: LogMapMethod = "heat_geodesic",
) -> wp.array[wp.vec2]:
    """
    Logarithmic map: every vertex's position in the source vertex's tangent plane.

    ``log_map(...)[v]`` is the 2D point in the *source's* frame whose length is the geodesic
    distance to ``v``, and whose direction is the initial direction of the geodesic that reaches
    ``v``. It is the inverse of the exponential map
    [`trace_from_vertex`][ordito.geodesic_walk.trace_from_vertex]
    computes, and the standard way to lay out a local coordinate patch around a point.

    The angle comes from the source's reference direction parallel-transported outwards: at any
    vertex the angle between that transported direction and the outward radial direction is
    exactly the angle at which the connecting geodesic left the source, because transport along
    that geodesic preserves it. ``method`` chooses where the radial direction and the radius come
    from:

    * ``"heat_geodesic"`` (the default): both from the distance to the source,
      [`heat_geodesic`][ordito.heat.heat_geodesic] with ``t`` the squared unique-edge mean (the
      vector heat method's own), whose gradient is the radial direction and whose value is the
      radius.
    * ``"potpourri3d"``: the vector heat method's own construction, the ``VectorHeat``
      strategy of ``potpourri3d.MeshVectorHeatSolver.compute_log_map`` (geometry-central's
      ``computeLogMap_VectorHeat``). A second vector diffusion, of a small outward-pointing disk
      at the source, gives the radial direction, and the radius is that unit radial field
      integrated by one Poisson solve. It reproduces potpourri3d's map; its radius is not
      ``heat_geodesic``'s distance, and is the less accurate of the two on graded meshes.

    potpourri3d's ``AffineLocal`` and ``AffineAdaptive`` strategies solve a small dense problem per
    vertex and are deliberately not ported. Takes the mesh as an
    [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh], which keeps the operators and factorizations
    later calls reuse (see [`heat_geodesic`][ordito.heat.heat_geodesic]), or as its ``vertices``
    and ``faces``.

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
        Diffusion time; defaults to the squared mean edge length, lengthened on a mesh too fine
        for it (see [`HeatSolver`][ordito.heat.HeatSolver]). The *angles* this function returns
        are measured from the source's ``basis_x`` in
        [`Trimesh.vertex_tangent_frames`][ordito.mesh.Trimesh.vertex_tangent_frames].
    method
        Where the radial direction and the radius come from: ``"heat_geodesic"`` or
        ``"potpourri3d"``, as described above.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` log-map coordinates in the source vertex's frame; ``(0, 0)`` at the
        source.
        On the cut locus — the antipode of a closed surface, where geodesics from the source arrive
        from every side — there is no direction to report, and the entry keeps the correct magnitude
        with an arbitrary angle. With ``method="potpourri3d"`` a vertex no diffusion reaches
        (another connected component) is ``(0, 0)``.

    Raises
    ------
    ValueError
        If the mesh has faces and ``source`` is not a vertex index in ``[0, n_vertices)``, or if
        ``method`` is not ``"heat_geodesic"`` or ``"potpourri3d"``.
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
    in the centre's tangent frame; the vector heat method's own construction finds the same
    direction with its shorter radius:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.grid(count=(11, 11), extents=(1.0, 1.0))
    coordinates = od.heat.log_map(vertices, faces, 60)  # centred on the vertex at the origin
    basis_x, basis_y, _ = od.tangent_space.vertex_tangent_frames(vertices, faces)
    x, y = coordinates.numpy()[108]  # the vertex at (0.4, 0.4)
    world = x * basis_x.numpy()[60] + y * basis_y.numpy()[60]
    print(np.round(world[:2], 2))  # [0.41 0.41]
    vector_heat = od.heat.log_map(vertices, faces, 60, method="potpourri3d")
    x, y = vector_heat.numpy()[108]
    world = x * basis_x.numpy()[60] + y * basis_y.numpy()[60]
    print(np.round(world[:2], 2))  # [0.35 0.35]
    ```
    """
    if method not in ("heat_geodesic", "potpourri3d"):
        raise ValueError(
            f"log_map: method must be 'heat_geodesic' or 'potpourri3d', got {method!r}"
        )
    bound, owned, arguments = mesh_arguments("log_map", mesh, vertices, faces, (source, t), 1)
    return _log_map(
        bound, owned, cast("int", arguments[0]), cast("float | None", arguments[1]), method
    )


def _log_map(
    mesh: Trimesh, owned: bool, source: int, t: float | None, method: LogMapMethod
) -> wp.array[wp.vec2]:
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
    if method == "potpourri3d":
        logarithm = _log_map_vector_heat(mesh, solver, source)
        _log_discarded("log_map", owned, solver)
        return logarithm
    _, scalar, frames, _ = solver.vector_operators
    basis_x, basis_y, _ = frames

    # The source's own reference direction transported outwards -- the "which way was x?" field
    # the angle is measured against, raw (unnormalized) like ``transport_tangent_vectors``' own
    # ``direction`` -- and ``heat_geodesic``'s heat, diffused together as one solve over
    # ``[vector system; heat system]`` for the reason ``transport_tangent_vectors`` gives.
    rows = 3 * n_vertices
    # The stack's right-hand side and solution, the divergence's sums, the distance, the shift's
    # means and the source's index (an ``int32`` word the seeding kernel writes): one buffer.
    rhs, diffused, scratch, phi, means, source_word = _zeroed_fields(
        device, rows, rows, n_vertices, n_vertices, _source_means_size(n_vertices, 1), 1
    )
    sources = _launch.alias(source_word, wp.int32, 0, 1)
    _launch.launch(
        kernel_heat.seed_log_map_source,
        dim=1,
        inputs=[source],
        outputs=[sources, _pairs(rhs, n_vertices), _launch.view(rhs, 2 * n_vertices, rows)],
        device=device,
    )
    transported_raw = _pairs(diffused, n_vertices)
    solver.diffuse_stacked(rhs, diffused, 1)

    # Radial direction: the unit gradient of the distance field, averaged onto vertices and
    # expressed in each vertex's frame.
    order = _corner_order(mesh, solver)
    distance = _distance_from_heat(
        mesh,
        solver,
        sources,
        _launch.view(diffused, 2 * n_vertices, rows),
        order,
        scratch,
        means,
        phi,
    )
    normals, areas = scalar[6], scalar[7]
    n_faces = faces.size // 3
    vertex_gradient, corner_gradient = _corner_outputs(order, n_vertices, n_faces, wp.vec3, device)
    _launch.launch(
        kernel_heat.scatter_unit_gradient_to_vertices,
        dim=n_faces,
        inputs=[vertices, faces, normals, areas, distance],
        outputs=[vertex_gradient, corner_gradient],
        device=device,
    )
    _sum_corner_outputs(order, corner_gradient, vertex_gradient)
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


def _log_map_vector_heat(mesh: Trimesh, solver: HeatSolver, source: int) -> wp.array[wp.vec2]:
    """
    ``log_map(method="potpourri3d")``: the vector heat method's radial field and its integral.

    Two vector diffusions on the one system, run as one solve over ``[vector system; vector
    system]`` ([`HeatSolver.diffuse_tangent_pair`][ordito.heat.HeatSolver.diffuse_tangent_pair]):
    the outward ball at the source, whose direction is radial, and the source's reference
    direction, the angle's zero. The radial field's divergence then integrates into the distance by
    the Poisson solve ``heat_geodesic`` ends with.
    """
    vertices, faces = mesh.vertices, mesh.faces
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    ring_halfedges, ring_offsets, _ = mesh.vertex_one_rings
    tangent_angles = _halfedge_tangent_angles(mesh, solver)
    transport_angles = mesh.halfedge_transport_angles
    cot_entries = solver.vector_operators[1][5]

    # The stack's right-hand side and solution (radial pairs, then horizontal pairs), the
    # divergence's sums and the distance: one buffer.
    rows = 4 * n_vertices
    rhs, diffused, scratch, distance = _zeroed_fields(device, rows, rows, n_vertices, n_vertices)
    _launch.launch(
        kernel_heat.seed_log_map_outward_ball,
        dim=1,
        inputs=[
            vertices,
            faces,
            ring_offsets,
            ring_halfedges,
            tangent_angles,
            transport_angles,
            wp.int32(source),
        ],
        outputs=[
            _pairs(rhs, n_vertices),
            _pairs(_launch.view(rhs, 2 * n_vertices, rows), n_vertices),
        ],
        device=device,
    )
    solver.diffuse_tangent_pair(rhs, diffused)
    radial_raw = _pairs(diffused, n_vertices)
    horizontal_raw = _pairs(_launch.view(diffused, 2 * n_vertices, rows), n_vertices)

    order = _corner_order(mesh, solver)
    divergence, corner_divergence = _corner_outputs(
        order, n_vertices, n_faces, wp.float64, device, sums=scratch
    )
    _launch.launch(
        kernel_heat.radial_field_divergence,
        dim=n_faces,
        inputs=[
            vertices,
            faces,
            tangent_angles,
            transport_angles,
            radial_raw,
            wp.int32(source),
            cot_entries,
        ],
        outputs=[divergence, corner_divergence],
        device=device,
    )
    _sum_corner_outputs(order, corner_divergence, divergence)
    solver.solve_poisson(divergence, distance)

    logarithm = _launch.empty(n_vertices, dtype=wp.vec2, device=device)
    _launch.launch(
        kernel_heat.log_map_from_vector_fields,
        dim=n_vertices,
        inputs=[radial_raw, horizontal_raw, distance, wp.int32(source)],
        outputs=[logarithm],
        device=device,
    )
    return logarithm


def _halfedge_tangent_angles(mesh: Trimesh, solver: HeatSolver) -> wp.array[wp.float64]:
    """
    Every halfedge's polar angle in its origin's frame, kept for ``solver``'s lifetime.

    [`halfedge_tangent_angles`][ordito.tangent_space.halfedge_tangent_angles] over the mesh's
    cached rings, the gauge its frames and connection are in, as ``_corner_order`` keeps its sort.
    """
    angles = _TANGENT_ANGLES.get(solver)
    if angles is None:
        angles = _TANGENT_ANGLES[solver] = halfedge_tangent_angles(
            mesh.vertices, mesh.faces, rings=mesh.vertex_one_rings
        )
    return angles


# Each solver's halfedge polar angles (``_halfedge_tangent_angles``), keyed weakly as
# ``_CORNER_ORDERS`` is.
_TANGENT_ANGLES: weakref.WeakKeyDictionary[HeatSolver, wp.array[wp.float64]] = (
    weakref.WeakKeyDictionary()
)


def _distance_from_heat(
    mesh: Trimesh,
    solver: HeatSolver,
    sources: wp.array[wp.int32],
    heat: wp.array[wp.float64],
    order: _CornerOrder,
    scratch: wp.array[wp.float64],
    means: wp.array[wp.float64],
    phi: wp.array[wp.float64] | None = None,
) -> wp.array[wp.float64]:
    """
    ``heat_geodesic`` from the diffused heat on: the divergence, the Poisson solve and the shift.

    Split out so ``log_map`` can diffuse the heat inside a stacked solve of its own. ``order`` is
    `_corner_order`'s choice of how per-face terms reach the vertices; ``scratch`` is a zeroed
    ``(n_vertices,)`` buffer the divergence is summed into, ``means`` a zeroed one of
    `_source_means_size` the shift's two means are reduced into, and ``phi``, when given, a zeroed
    one the distance is solved into (allocated here otherwise).
    """
    vertices, faces = mesh.vertices, mesh.faces
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    _, _, _, _, _, cot_entries, normals, areas = solver.operators

    # Integrated divergence b = div(X) of the unit field X = -grad(u)/|grad(u)|, then a Poisson
    # solve L phi = b, i.e. (-L) phi = -b with the positive semi-definite operator. The field is
    # formed and integrated per face, so it never occupies an (n_faces,) buffer; the terms are
    # added atomically, or (deterministic mode) stored per corner and summed in a second launch.
    # The kernel integrates ``-X`` and so accumulates ``-b`` directly: the divergence is linear in
    # the field and negation is exact, so this is the negated sum without a pass to negate it.
    neg_divergence, corner_divergence = _corner_outputs(
        order, n_vertices, n_faces, wp.float64, device, sums=scratch
    )
    _launch.launch(
        kernel_heat.unit_gradient_divergence,
        dim=n_faces,
        inputs=[vertices, faces, normals, areas, heat, cot_entries],
        outputs=[neg_divergence, corner_divergence],
        device=device,
    )
    _sum_corner_outputs(order, corner_divergence, neg_divergence)

    if phi is None:
        phi = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    solver.solve_poisson(neg_divergence, phi)

    # Shift so the field's mean over the sources is zero, and orient it positive -- the
    # ``igl::heat_geodesics_solve`` convention, which makes a single source's distance exactly
    # zero. Both means from one reduction launch, applied by a second that reads them on the
    # device, so nothing is read back.
    n_sources = sources.size
    blocks = kernel_reduce.blocks_1d(max(n_vertices, n_sources))
    _launch.launch_tiled(
        kernel_heat.source_and_global_sums,
        dim=[blocks],
        inputs=[phi, sources, 2],
        outputs=[means],
        block_dim=TILE_1D,
        device=device,
    )
    _launch.launch(
        kernel_heat.shift_and_orient,
        dim=n_vertices,
        inputs=[means, wp.int32(n_sources), wp.int32(1), phi],
        device=device,
    )
    return phi


def _source_means_size(n_vertices: int, n_sources: int) -> int:
    """Length of the zeroed buffer `_distance_from_heat`'s two means are reduced into."""
    return _launch.block_partials_size(kernel_reduce.blocks_1d(max(n_vertices, n_sources)), 2)


def _corner_order(mesh: Trimesh, solver: HeatSolver) -> _CornerOrder:
    """
    How the heat solvers' per-face terms reach the vertices: the face buffer's stable sort.

    Under ``ordito.config``'s deterministic mode ``(sorted_vertices, corner_order)``, the sort
    ``kernels/scatter.sum_sorted_corner_values`` adds each vertex's corner terms over in ascending
    corner order, so a call returns the same bits on every run; otherwise ``None``, which selects
    the kernels' atomic per-face scatter. Kept for ``solver``'s lifetime once built, like its
    operators: it depends on the faces alone.
    """
    if not is_deterministic():
        return None
    order = _CORNER_ORDERS.get(solver)
    if order is None:
        faces = mesh.faces
        order = _CORNER_ORDERS[solver] = od.array.sort_and_argsort(
            faces, end_bit=max(1, (mesh.vertices.size - 1).bit_length())
        )
    return order


def _corner_outputs(
    order: _CornerOrder,
    n_vertices: int,
    n_faces: int,
    dtype: type[_T],
    device: wp.DeviceLike,
    *,
    sums: wp.array[_T] | None = None,
) -> tuple[wp.array[_T], odt.ArrayNd | None]:
    """
    Allocate the per-vertex sums a face kernel fills and its corner table, as `_corner_order` chose.

    Atomic: a zeroed ``(n_vertices,)`` sum and no table (``None`` launches as an empty array).
    Ordered: an ``(n_faces, 3)`` table for the face kernel to store into and the sum for
    `_sum_corner_outputs` to write. ``sums``, when given, is a zeroed ``(n_vertices,)`` buffer the
    caller holds, used as the sum either way.
    """
    if order is None:
        if sums is None:
            sums = _launch.zeros(n_vertices, dtype=dtype, device=device)
        return sums, None
    if sums is None:
        sums = _launch.empty(n_vertices, dtype=dtype, device=device)
    return sums, odt.empty_2d((n_faces, 3), dtype, device=device)


def _sum_corner_outputs(
    order: _CornerOrder, corners: odt.ArrayNd | None, out_sums: wp.array[Any]
) -> None:
    """Add each vertex's corner values in ascending corner order, when `_corner_order` chose to."""
    if order is None or corners is None:
        return
    sorted_vertices, corner_order = order
    empty = _launch.empty(0, dtype=wp.int32, device=out_sums.device)
    _launch.launch(
        kernel_scatter.SUM_SORTED_CORNER_VALUES[out_sums.dtype],
        dim=out_sums.size,
        inputs=[sorted_vertices, corner_order, corners],
        outputs=[out_sums, empty, empty],
        device=out_sums.device,
    )


def _zeroed_fields(device: wp.DeviceLike, *sizes: int) -> list[wp.array[wp.float64]]:
    """
    One zeroed allocation cut into consecutive ``float64`` fields of ``sizes`` entries, each > 0.

    Each field starts on an even entry, so its leading pairs read as ``wp.vec2d`` (`_pairs`): a
    field of ``(2 + n_scalars) * n_vertices`` entries is laid out as ``block_diag``'s stack of a
    vector system and ``n_scalars`` scalar systems, the tangent field's interleaved rows first.
    Every field holds the whole allocation, so only scratch a call drops together shares one.
    """
    total = sum(size + size % 2 for size in sizes)
    flat = _launch.zeros(total // 2, dtype=wp.vec2d, device=device).view(wp.float64).flatten()
    fields: list[wp.array[wp.float64]] = []
    start = 0
    for size in sizes:
        fields.append(_launch.view(flat, start, start + size))
        start += size + size % 2
    return fields


def _zeroed_fields_and_flag(
    device: wp.DeviceLike, *sizes: int
) -> tuple[list[wp.array[wp.float64]], wp.array[wp.int32]]:
    """
    `_zeroed_fields` of ``sizes``, and a zeroed ``int32`` word in the same allocation.

    The word is the out-of-range flag a seeding kernel raises (`_require_indices_in_range`),
    scratch the call drops with the fields.
    """
    *fields, word = _zeroed_fields(device, *sizes, 1)
    return fields, _launch.alias(word, wp.int32, 0, 1)


def _pairs(field: wp.array[wp.float64], n: int) -> wp.array[wp.vec2d]:
    """Return the leading ``2 * n`` entries of a `_zeroed_fields` field as ``n`` ``wp.vec2d``."""
    return _launch.view(field, 0, 2 * n).reshape((n, 2)).view(wp.vec2d)


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
        With a `Trimesh`: the diffusion time; defaults to the squared mean edge length,
        lengthened on a mesh too fine for it (see [`HeatSolver`][ordito.heat.HeatSolver]).
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
        The diffusion time; ``None`` for the default, the squared mean unique-edge length
        (lengthened on a mesh too fine for it, see Notes).
    use_robust : bool
        Whether the scalar operators are built from mollified edge lengths (see
        [`heat_operators`][ordito.heat.heat_operators]).
    method : str
        Whose discretization the operators follow, ``"potpourri3d"``, ``"meshlab"`` or
        ``"igl"`` (see [`heat_operators`][ordito.heat.heat_operators]).
    float64_rotations : bool
        Whether the iterative tangent-field diffusions read the vector system's rotations in
        ``float64``; ``False`` (the default) reads them in ``float32``, which halves those bytes
        a round. Set it to reproduce geometry-central (potpourri3d) on meshes with needles: there
        the rounding moves [`log_map`][ordito.heat.log_map] by about ``1e-4``, and with it set a
        first call agrees with a factored one. The tangent-field diffusions on a large mesh then
        take up to 1.7 times as long. A factored system is solved in ``float64`` either way.

    Notes
    -----
    **The default time grows on fine meshes.** A diffused field falls off like
    ``exp(-d / sqrt(t))`` at distance ``d`` from its source, so at the squared mean edge length a
    mesh fine enough -- about a million faces for a source on one side of a sphere -- puts its far
    side below the smallest ``float64``, and every direction and ratio computed there is lost.
    With ``t`` left ``None`` and the ``"potpourri3d"`` discretization (the vector heat method's,
    without ``use_robust``), the time is therefore at least ``(diagonal / 400) ** 2``, the
    bounding-box diagonal standing in for the farthest distance from a source: meshes up to about
    400 mean edges across keep the squared mean edge length, and finer ones get the longest time
    the field can span in range. [`heat_signed_distance`][ordito.heat.heat_signed_distance]'s
    edge discretization does the same from its own default. The operators are then this
    solver's own rather than the mesh's cached
    [`heat_operators`][ordito.mesh.Trimesh.heat_operators] and
    [`vector_heat_operators`][ordito.mesh.Trimesh.vector_heat_operators], which keep the squared
    mean edge length. Deciding it reads back the mean edge length and the mesh's
    [`bounds`][ordito.mesh.Trimesh.bounds] once per solver.

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
    solver = mesh.heat_solver(method="meshlab")  # the one heat_geodesic's default runs on
    print(solver.nbytes)  # 0
    for source in (0, 1):
        od.heat.heat_geodesic(mesh, sources=wp.array([source], dtype=wp.int32))
    print(solver.nbytes > 0)  # True
    solver.release()
    print(solver.nbytes)  # 0
    ```
    """

    def __init__(
        self,
        mesh: Trimesh,
        t: float | None = None,
        *,
        use_robust: bool = False,
        method: HeatMethod = "potpourri3d",
    ) -> None:
        """
        Bind a solver to ``mesh`` at ``t``; nothing is assembled until first use.

        Raises
        ------
        ValueError
            If ``method`` is not ``"potpourri3d"``, ``"meshlab"`` or ``"igl"``.
        """
        _heat_method_choices(method)
        self.t = None if t is None else float(t)
        self.use_robust = bool(use_robust)
        self.method: HeatMethod = method
        self.float64_rotations = False
        self._mesh = weakref.ref(mesh)
        self._operators: HeatOperators | None = None
        self._vector_operators: VectorHeatOperators | None = None
        self._systems: dict[str, _KeptSystem] = {}
        # Set once a settled iterate of this mesh's systems fails its backward-error check: the
        # operators are ill-conditioned in places, so the Poisson solve is factored at once too.
        self._ill_conditioned = False
        # The heat system pinned to zero on the rim, its preconditioner and rim mask, once built;
        # ``()`` for a mesh with no rim, which has no pinned solve to run.
        self._pinned: (
            tuple[odt.SparseMatrix, wpl.LinearOperator, wp.array[wp.int32]] | tuple[()] | None
        ) = None
        # The edge system, its preconditioner and the edge midpoints its factorization orders rows
        # by, once built; like the operators, it outlives ``release``.
        self._edge_system: (
            tuple[odt.BsrMatrix[wp.mat22d], wpl.LinearOperator, wp.array[wp.vec3], int] | None
        ) = None
        # The mean unique-edge length, read back once (``_mean_edge_length``), and the time the
        # vertex systems are built at with their settle window, once resolved (``_vertex_time``).
        self._mean_length: float | None = None
        self._resolved: tuple[float | None, int] | None = None

    @property
    def operators(self) -> HeatOperators:
        """[`heat_operators`][ordito.heat.heat_operators] for the mesh at ``t``."""
        if self._operators is None:
            mesh = self._bound_mesh()
            if self._is_default():
                # The mesh's own cached bundle, so the property and this solver share one assembly.
                self._operators = mesh.heat_operators
            else:
                self._operators = heat_operators(
                    mesh.vertices,
                    mesh.faces,
                    self._vertex_time()[0],
                    use_robust=self.use_robust,
                    cot_entries=None if self.use_robust else mesh.cotmatrix_entries,
                    pattern=mesh.mesh_operator_pattern,
                    method=self.method,
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
            If this solver was built with ``use_robust`` or a ``method`` other than
            ``"potpourri3d"``, the discretization the vector heat method is defined on.
        """
        self._require_vector_discretization("the vector heat operators")
        if self._vector_operators is None:
            mesh = self._bound_mesh()
            t = self._vertex_time()[0]
            if t is None:
                self._vector_operators = mesh.vector_heat_operators
            else:
                self._vector_operators = vector_heat_operators(
                    mesh.vertices,
                    mesh.faces,
                    t,
                    scalar_operators=self._operators,
                    frames=mesh.vertex_tangent_frames,
                    pattern=mesh.mesh_operator_pattern,
                    transport_angles=mesh.halfedge_transport_angles,
                )
            self._operators = self._vector_operators[1]
        return self._vector_operators

    def _is_default(self) -> bool:
        """Whether this solver's scalar operators are the mesh's own cached bundle."""
        return (
            not self.use_robust and self.method == "potpourri3d" and self._vertex_time()[0] is None
        )

    def _vertex_time(self) -> tuple[float | None, int]:
        """
        Return the time the vertex systems are built at and the settle window their solves take.

        The time is ``t``, or ``None`` for the default the device forms, ``h ** 2``; a
        ``"potpourri3d"`` solver without ``use_robust`` raises an unset ``t`` to the
        ``_DIFFUSION_DECAY`` floor where that is longer and scales its window to its time.
        """
        if self._resolved is None:
            t, rounds = self.t, _HEAT_SETTLE_ROUNDS
            if self.method == "potpourri3d" and not self.use_robust:
                default = self._mean_edge_length() ** 2
                if t is None:
                    floor = _diffusion_floor(self._bound_mesh())
                    t = floor if floor > default else None
                rounds = _settle_window(t, default)
            self._resolved = (t, rounds)
        return self._resolved

    def _mean_edge_length(self) -> float:
        """Return the mesh's mean unique-edge length ``h``, read back once."""
        if self._mean_length is None:
            mesh = self._bound_mesh()
            sums = _edge_length_sums(mesh.vertices, mesh.mesh_operator_pattern)
            # Host readback, once per solver: whether the default time must grow, and how long a
            # settle a longer time needs, are decided on the host.
            total, count = read_values(sums, 0, 2)
            self._mean_length = float(total) / float(count) if count > 0 else 0.0
        return self._mean_length

    @property
    def ill_conditioned(self) -> bool:
        """Whether a settled diffusion on this mesh failed its backward-error check."""
        return self._ill_conditioned

    @property
    def nbytes(self) -> int:
        """Device memory this solver's kept factorizations hold, in bytes (shared analyses once)."""
        # A stacked system's blocks are the vector and heat systems' own factorizations: each
        # distinct one is counted once.
        held: dict[int, SparseCholesky] = {}
        for kept in self._systems.values():
            for factorization in _held_factorizations(kept.factorization):
                held[id(factorization)] = factorization
        return held_nbytes(held.values())

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
        ([`block_diag`][ordito.array.block_diag]), in the launches of one.

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
            # Iterated as one system; factored, when the settle fails or on the second solve, as
            # its distinct blocks -- the vector and heat systems' own factorization slots, so a
            # factorization either already holds is reused and one built here serves their solves.
            vector_system, scalar, _, _ = self.vector_operators
            stack = od.array.block_diag((vector_system, *(scalar[0],) * n_scalars))
            kept = self._systems[role] = _KeptSystem(
                _StackedFactorization(
                    stack,
                    self._bound_mesh().vertices,
                    self._slot("vector", vector_system).factorization,
                    self._slot("heat", scalar[0]).factorization,
                    n_scalars,
                )
            )
        self._settle(role, kept.factorization.matrix, rhs, solution)

    def diffuse_tangent_pair(
        self, rhs: wp.array[wp.float64], solution: wp.array[wp.float64]
    ) -> None:
        """
        Diffuse two tangent fields as one solve over ``[vector system; vector system]``.

        Two [`diffuse_tangent`][ordito.heat.HeatSolver.diffuse_tangent] calls in the launches of
        one, as [`diffuse_stacked`][ordito.heat.HeatSolver.diffuse_stacked] runs a tangent field
        beside scalar ones. Factored, when that pays, as the vector system's own factorization
        solving both fields as two columns.

        Parameters
        ----------
        rhs
            ``(4 * n_vertices,)`` right-hand side: each tangent field as interleaved ``float64``
            pairs, the first field's ``2 * n_vertices`` entries then the second's.
        solution
            Same layout: the initial guess, overwritten with the diffused fields.

        Raises
        ------
        ValueError
            If this solver was built with ``use_robust`` or a ``method`` other than
            ``"potpourri3d"``, or if ``rhs`` and ``solution`` do not hold ``4 * n_vertices``
            entries.
        RuntimeError
            If ``rhs``, ``solution`` and the mesh are not all on one device.
        """
        self._require_fields("HeatSolver.diffuse_tangent_pair", rhs, solution, rhs.size, 4)
        role = "vector_pair"
        kept = self._systems.get(role)
        if kept is None:
            vector_system, scalar, _, _ = self.vector_operators
            kept = self._systems[role] = _KeptSystem(
                _PairedFactorization(
                    od.array.block_diag((vector_system, vector_system)),
                    self._slot("vector", vector_system).factorization,
                    self._slot("heat", scalar[0]).factorization,
                )
            )
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

    def diffuse_pinned(self, rhs: wp.array[wp.float64], solution: wp.array[wp.float64]) -> bool:
        """
        Diffuse a scalar field with the heat pinned to zero on the mesh's rim.

        The Dirichlet counterpart of [`diffuse`][ordito.heat.HeatSolver.diffuse]: ``(M - t L) u =
        rhs`` on the interior vertices, ``u = 0`` on every vertex with an edge on a single face,
        settled the same way. ``igl::heat_geodesics`` averages this with the Neumann solution
        ([`heat_geodesic`][ordito.heat.heat_geodesic] at ``boundary="average"``). The pinned
        system and the rim are built on first use and kept; a mesh without a rim has nothing to
        pin, and ``solution`` is then left as it was.

        Parameters
        ----------
        rhs
            ``(n_vertices,)`` right-hand side, ``float64``; its rim entries are ignored.
        solution
            ``(n_vertices,)`` initial guess, overwritten with the pinned field.

        Returns
        -------
        bool
            Whether the mesh has a rim, i.e. whether ``solution`` was solved for.

        Raises
        ------
        ValueError
            If ``rhs`` does not hold ``n_vertices`` entries, or ``solution`` is not shaped as
            ``rhs``.
        RuntimeError
            If ``rhs``, ``solution`` and the mesh are not all on one device.
        """
        self._require_fields("HeatSolver.diffuse_pinned", rhs, solution, rhs.shape[-1], 1)
        if self._pinned is None:
            self._pinned = self._pinned_system()
        if not self._pinned:
            return False
        system, preconditioner, rim = self._pinned
        pinned_rhs = _launch.empty(rhs.size, dtype=wp.float64, device=rhs.device)
        _launch.map(kernel_heat.off_rim, rhs, rim, out=pinned_rhs)
        self._settle("pinned", system, pinned_rhs, solution, preconditioner=preconditioner)
        return True

    def _pinned_system(
        self,
    ) -> tuple[odt.SparseMatrix, wpl.LinearOperator, wp.array[wp.int32]] | tuple[()]:
        """Build ``diffuse_pinned``'s system, preconditioner and rim mask; ``()`` with no rim."""
        mesh = self._bound_mesh()
        heat_system = self.operators[0]
        _, _, offsets, columns, run_start, keys, count, _ = mesh.mesh_operator_pattern
        n_vertices = mesh.vertices.size
        device = mesh.vertices.device
        rim = _launch.empty(n_vertices, dtype=wp.int32, device=device)
        n_rim = _launch.zeros(1, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_heat.rim_vertices,
            dim=n_vertices,
            inputs=[offsets, columns, run_start, keys, wp.int32(count)],
            outputs=[rim, n_rim],
            device=device,
        )
        # Readback: a closed mesh has no rim and so no second solve, which the count decides here
        # once per solver rather than every call paying a solve that returns the Neumann heat.
        if read_scalar(n_rim, 0) == 0:
            return ()
        values = _launch.empty_like(heat_system.values)
        _launch.launch(
            kernel_heat.pinned_heat_rows,
            dim=n_vertices,
            inputs=[heat_system.offsets, heat_system.columns, rim, heat_system.values],
            outputs=[values],
            device=device,
        )
        system = od.array.bsr_with_values(heat_system, values)
        return system, twl.jacobi_preconditioner(system), rim

    def diffuse_edge_field(self, rhs: wp.array[wp.vec2d], solution: wp.array[wp.vec2d]) -> None:
        """
        Diffuse a tangent field held on the edges: the Crouzeix-Raviart vector heat system.

        Solves ``(M + t_e L) X = rhs`` over the rows of
        [`Trimesh.edges_unique`][ordito.mesh.Trimesh.edges_unique], each vector in the frame of
        its edge: the edge-based discretization
        [`heat_signed_distance`][ordito.heat.heat_signed_distance] diffuses at
        ``method="potpourri3d"``. ``t_e`` is ``t`` when the solver has one, else
        ``(h / 2) ** 2`` with ``h`` the mean unique-edge length, geometry-central's, raised to
        ``(diagonal / 400) ** 2`` on a mesh whose bounding-box diagonal is longer, where the
        shorter time would let the field fall out of the ``float64`` range (see
        [`heat_signed_distance`][ordito.heat.heat_signed_distance]). The system is built on first
        use and kept; it is settled and factored as the solver's other systems are.

        Parameters
        ----------
        rhs
            ``(n_edges,)`` right-hand side, ``n_edges`` the rows of ``edges_unique``.
        solution
            ``(n_edges,)`` initial guess, overwritten with the diffused field.

        Raises
        ------
        ValueError
            If this solver was built with ``use_robust`` or a ``method`` other than
            ``"potpourri3d"``, or if ``rhs`` and ``solution`` do not hold one vector per unique
            edge.
        RuntimeError
            If ``rhs``, ``solution`` and the mesh are not all on one device.
        """
        self._require_vector_discretization("the edge vector heat system")
        mesh = self._bound_mesh()
        require_same_device(vertices=mesh.vertices, rhs=rhs, solution=solution)
        n_edges = mesh.edges_unique.shape[0]
        if rhs.size != n_edges or tuple(solution.shape) != tuple(rhs.shape):
            raise ValueError(
                f"HeatSolver.diffuse_edge_field: rhs must hold {n_edges} rows for this mesh and "
                f"solution its shape, got {tuple(rhs.shape)} and {tuple(solution.shape)}"
            )
        if self._edge_system is None:
            self._edge_system = self._crouzeix_raviart_system()
        system, preconditioner, midpoints, settle_rounds = self._edge_system
        role = "crouzeix_raviart"
        if role not in self._systems:
            # Placed here, not by ``_slot``: its factorization orders the rows by the edge
            # midpoints, where the vertex systems' slots take the vertices.
            self._systems[role] = _KeptSystem(twl.OperatorFactorization(system, midpoints))
        self._settle(
            role, system, rhs, solution, preconditioner=preconditioner, settle_rounds=settle_rounds
        )

    def _crouzeix_raviart_system(
        self,
    ) -> tuple[odt.BsrMatrix[wp.mat22d], wpl.LinearOperator, wp.array[wp.vec3], int]:
        """
        Build ``diffuse_edge_field``'s system, preconditioner, edge midpoints and settle window.

        ``wp.mat22d`` blocks over the unique edges, assembled from keys
        (``kernels/heat.crouzeix_raviart_keys``) and one row kernel. The default time is
        geometry-central's ``(h / 2) ** 2``, formed on the device from the scalar bundle's own
        edge-length sums, unless ``_DIFFUSION_DECAY`` sets a longer one.
        """
        mesh = self._bound_mesh()
        t = self.t
        vertices, faces = mesh.vertices, mesh.faces
        device = vertices.device
        edges = mesh.edges_unique
        face_edges = mesh.faces_unique_edges
        n_edges = edges.shape[0]
        n_faces = faces.size // 3
        count = 12 * n_faces
        keys, order = od.array.csr_key_buffers(count, device)
        _launch.launch(
            kernel_heat.crouzeix_raviart_keys,
            dim=n_faces,
            inputs=[face_edges, wp.int32(n_edges)],
            outputs=[keys, order],
            device=device,
        )
        offsets, columns, starts = od.array.csr_from_keys(keys, order, count, n_edges, n_edges)
        built = _SCALAR_BUILDS.get(self.operators[0])
        edge_sums = built.edge_sums if built is not None else None
        if edge_sums is None:
            edge_sums = _edge_length_sums(vertices, mesh.mesh_operator_pattern)
        geometry_central = 0.25 * self._mean_edge_length() ** 2
        if t is None:
            floor = _diffusion_floor(mesh)
            t = floor if floor > geometry_central else None
        settle_rounds = _settle_window(t, geometry_central)
        values = _launch.empty(columns.size, dtype=wp.mat22d, device=device)
        midpoints = _launch.empty(n_edges, dtype=wp.vec3, device=device)
        _launch.launch(
            kernel_heat.crouzeix_raviart_system_rows,
            dim=n_edges,
            inputs=[
                offsets,
                starts,
                order,
                vertices,
                faces,
                face_edges,
                edges,
                wp.float64(0.25 if t is None else t),
                edge_sums,
                wp.int32(1 if t is None else 0),
            ],
            outputs=[values, midpoints],
            device=device,
        )
        system = od.array.bsr_from_csr(n_edges, n_edges, offsets, columns, values)
        return system, twl.jacobi_preconditioner(system), midpoints, settle_rounds

    def _settle(
        self,
        role: str,
        matrix: odt.SparseMatrix,
        rhs: odt.ArrayNd,
        solution: odt.ArrayNd,
        *,
        preconditioner: wpl.LinearOperator | None = None,
        settle_rounds: int | None = None,
    ) -> None:
        """
        Run one settled diffusion of ``role``'s system, factored when that pays or is needed.

        ``settle_rounds`` defaults to the vertex systems' window (``_vertex_time``).
        """
        kept = self._kept(role, matrix)
        held = kept.factorization.factorization is not None
        twl.solve_spd_settled(
            matrix,
            rhs,
            solution,
            check_rounds=_HEAT_CHECK_ROUNDS,
            change_tolerance=_HEAT_CHANGE_TOLERANCE,
            settle_rounds=self._vertex_time()[1] if settle_rounds is None else settle_rounds,
            preconditioner=preconditioner,
            factorization=kept.factorization,
            narrow_values=not (
                self.float64_rotations and (role in _ROTATION_ROLES or role.startswith("stack"))
            ),
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
        kept = self._slot(role, matrix)
        kept.solves += 1
        if (
            kept.solves >= 2
            and kept.factorization.factorization is None
            and kept.factorization.factor(self._sharing(role))
        ):
            _LOGGER.debug("HeatSolver: factored the %s system on its second solve", role)
        return kept

    def _slot(self, role: str, matrix: odt.SparseMatrix) -> _KeptSystem:
        """Return ``role``'s kept system, made (unfactored, unsolved) for ``matrix`` if new."""
        kept = self._systems.get(role)
        if kept is None:
            kept = self._systems[role] = _KeptSystem(
                twl.OperatorFactorization(matrix, self._bound_mesh().vertices)
            )
        return kept

    def _sharing(self, role: str) -> twl.OperatorFactorization | None:
        """
        Return the heat system's factorization slot, whose analysis the Poisson one shares.

        The vector system, stored over its pattern by blocks, derives its analysis from it.
        """
        heat = self._systems.get("heat")
        return heat.factorization if role in ("poisson", "vector") and heat is not None else None

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

    def _require_vector_discretization(self, what: str) -> None:
        """Raise unless this solver's discretization is the one the vector heat method uses."""
        if self.use_robust:
            raise ValueError(f"HeatSolver: no use_robust variant of {what}")
        if self.method != "potpourri3d":
            raise ValueError(f"HeatSolver: no method={self.method!r} variant of {what}")

    def _bound_mesh(self) -> Trimesh:
        """Return the mesh this solver belongs to."""
        mesh = self._mesh()
        if mesh is None:
            raise RuntimeError("HeatSolver: its Trimesh no longer exists")
        return mesh


def _diffusion_floor(mesh: Trimesh) -> float:
    """Return the shortest default time a ``"potpourri3d"`` solver takes on ``mesh``."""
    low, high = mesh.bounds
    return (math.dist(low, high) / _DIFFUSION_DECAY) ** 2


def _settle_window(t: float | None, default: float) -> int:
    """
    Return the settle window for a diffusion at ``t`` whose method's own time is ``default``.

    The rounds a field needs past full reach grow like ``sqrt(t)``, the system's conditioning;
    ``_HEAT_SETTLE_ROUNDS`` covers the default time.
    """
    if t is None or not t > default > 0.0:
        return _HEAT_SETTLE_ROUNDS
    return math.ceil(_HEAT_SETTLE_ROUNDS * math.sqrt(t / default))


@dataclass
class _KeptSystem:
    """One system a ``HeatSolver`` solves: its factorization slot and how often it was solved."""

    factorization: twl.OperatorFactorization
    solves: int = 0


class _StackedFactorization(twl.OperatorFactorization):
    """
    The factorization of ``block_diag((vector system, heat system x n_scalars))``, by blocks.

    ``HeatSolver.diffuse_stacked`` iterates the stack as one system. Factored as one operator, the
    stack holds the heat system once per scalar field and shares nothing with the solver's own
    factorizations of the two systems; on ``bunny`` the vector and two heat systems' stack
    factored in 81 ms, its two distinct blocks in 14 + 5.5 ms. So from ``_STACK_BLOCKS_FROM``
    stack rows, or whenever a block is already factored, this factors the blocks -- the solver's
    slots for the vector and heat systems, so either may already hold its factorization and later
    solves of either reuse it -- and solves the stack as the vector field's solve and one solve
    of the scalar fields as columns. Below that the stack is factored whole, as one operator:
    there one analysis costs less than two (a 4 624-vertex stack of the vector and two heat
    systems 6.1 ms whole, 7.5 ms by blocks).
    """

    def __init__(
        self,
        stack: odt.SparseMatrix,
        vertices: wp.array[wp.vec3],
        vector: twl.OperatorFactorization,
        heat: twl.OperatorFactorization,
        n_scalars: int,
    ) -> None:
        """Hold ``stack`` over the mesh's ``vertices`` and its two distinct blocks' slots."""
        # The stack's row positions order only a whole factorization, which most stacks never
        # take (they settle, or factor by blocks): ``factor`` forms them when it does.
        super().__init__(stack)
        self._vertices: wp.array[wp.vec3] | None = vertices
        self._vector = vector
        self._heat = heat
        self._n_scalars = n_scalars

    @property
    def blocks(self) -> tuple[twl.OperatorFactorization, twl.OperatorFactorization]:
        """The vector and heat systems' factorization slots."""
        return self._vector, self._heat

    @property
    def whole(self) -> SparseCholesky | None:
        """The stack's own factorization, when it was factored whole."""
        return super().factorization

    @property
    def factorization(self) -> SparseCholesky | None:
        """A held factorization: the whole stack's, or the vector block's once both blocks are."""
        whole = self.whole
        if whole is not None:
            return whole
        vector, heat = self._vector.factorization, self._heat.factorization
        return vector if vector is not None and heat is not None else None

    @property
    def nbytes(self) -> int:
        """Device memory the factorizations held for the stack hold, in bytes."""
        return held_nbytes(_held_factorizations(self))

    def factor(
        self, sharing: twl.OperatorFactorization | None = None, *, expandable: bool = False
    ) -> bool:
        """Factor the stack, whole or by blocks; ``sharing`` and ``expandable`` are not read."""
        del sharing, expandable
        if self.factorization is not None:
            return True
        blocks = (
            self.matrix.nrow * self.matrix.block_shape[0] >= _STACK_BLOCKS_FROM
            or self._vector.factorization is not None
            or self._heat.factorization is not None
        )
        if not blocks:
            if self._vertices is not None:
                self._coordinates = _stacked_positions(self._vertices, self._n_scalars)
                self._vertices = None
            return super().factor()
        # The heat system first, built for the vector system (stored over its pattern by blocks)
        # to derive its analysis from.
        heat = self._heat.factor(expandable=True)
        vector = self._vector.factor(self._heat)
        return vector and heat

    def solve(
        self, rhs: odt.ArrayNd, solution: odt.ArrayNd, *, tol: float, componentwise: bool = False
    ) -> None:
        """Solve by the whole stack's factorization, or as its blocks."""
        if self.whole is not None:
            super().solve(rhs, solution, tol=tol, componentwise=componentwise)
            return
        if self.factorization is None:
            raise ValueError("_StackedFactorization.solve: nothing is factored; call factor()")
        n_vertices = rhs.size // (2 + self._n_scalars)
        split = 2 * n_vertices
        shape = (self._n_scalars, n_vertices)
        self._vector.solve(
            odt.as_dense(rhs[:split]),
            odt.as_dense(solution[:split]),
            tol=tol,
            componentwise=componentwise,
        )
        scalar_rhs = odt.as_dense(rhs[split:])
        scalar_solution = odt.as_dense(solution[split:])
        if self._n_scalars > 1:
            scalar_rhs, scalar_solution = scalar_rhs.reshape(shape), scalar_solution.reshape(shape)
        self._heat.solve(scalar_rhs, scalar_solution, tol=tol, componentwise=componentwise)

    def release(self) -> None:
        """Drop the stack's factorization and both blocks'."""
        super().release()
        self._vector.release()
        self._heat.release()


class _PairedFactorization(twl.OperatorFactorization):
    """
    The factorization of ``block_diag((vector system, vector system))``: the vector system's.

    ``HeatSolver.diffuse_tangent_pair`` iterates the stack as one system; factored, it is the one
    distinct block, the solver's vector slot (its analysis derived from the heat system's when that
    is factored over the shared pattern), solving the two fields as two columns.
    """

    def __init__(
        self,
        stack: odt.SparseMatrix,
        vector: twl.OperatorFactorization,
        heat: twl.OperatorFactorization,
    ) -> None:
        """Hold ``stack`` and the vector and heat systems' slots."""
        super().__init__(stack)
        self._vector = vector
        self._heat = heat

    @property
    def factorization(self) -> SparseCholesky | None:
        """The vector system's factorization, once held."""
        return self._vector.factorization

    @property
    def nbytes(self) -> int:
        """Device memory the vector system's factorization holds, in bytes."""
        return self._vector.nbytes

    def factor(
        self, sharing: twl.OperatorFactorization | None = None, *, expandable: bool = False
    ) -> bool:
        """Factor the vector system; ``sharing`` and ``expandable`` are not read."""
        del sharing, expandable
        return self._vector.factor(self._heat)

    def solve(
        self, rhs: odt.ArrayNd, solution: odt.ArrayNd, *, tol: float, componentwise: bool = False
    ) -> None:
        """Solve the two fields as columns of the vector system's factorization."""
        shape = (2, rhs.size // 2)
        self._vector.solve(
            odt.as_dense(rhs).reshape(shape),
            odt.as_dense(solution).reshape(shape),
            tol=tol,
            componentwise=componentwise,
        )

    def release(self) -> None:
        """Drop the vector system's factorization."""
        super().release()
        self._vector.release()


# Stack rows from which ``_StackedFactorization`` factors by blocks rather than whole: the vector
# and two heat systems from 8 192 vertices, the vector and one from 10 923. Measured (factor time,
# whole vs blocks): two heat systems 6.1 vs 7.5 ms at 4 624 vertices, 12.8 vs 9.1 at 10 242, 81 vs
# 20 at 35 947; one 7.8 vs 9.0 at 10 242. End to end on the 17 689-vertex graded saddle, where the
# settle fails and the stack is factored on the first call: ``log_map`` 1.12x, transport 1.45x.
_STACK_BLOCKS_FROM = 1 << 15


def _held_factorizations(holder: twl.OperatorFactorization) -> list[SparseCholesky]:
    """Return the sparse factorizations ``holder`` holds: a stacked system's own and its blocks'."""
    if not isinstance(holder, _StackedFactorization):
        return [] if holder.factorization is None else [holder.factorization]
    held = [] if holder.whole is None else [holder.whole]
    return held + [h.factorization for h in holder.blocks if h.factorization is not None]


def _stacked_positions(vertices: wp.array[wp.vec3], n_scalars: int) -> wp.array[wp.vec3]:
    """
    Return the vertex of every row of a stacked field (`_zeroed_fields`).

    A vector system's two rows, then each scalar system's one: the ordering of the stack's
    factorization when it is factored whole.
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
