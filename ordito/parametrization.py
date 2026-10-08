"""
Flatten a mesh into the plane: harmonic, Tutte, ARAP and LSCM parametrizations.

Four maps, differing in what they hold fixed and what they minimize. The first three pin the
boundary and solve for the interior:
[`harmonic`][ordito.parametrization.harmonic] minimizes Dirichlet energy with cotangent weights,
[`tutte`][ordito.parametrization.tutte] is the same solve with uniform weights (guaranteeing an
injective map for a convex boundary), and [`arap`][ordito.parametrization.arap] alternates local
rotation fits with a global solve to trade conformality for low area distortion.
[`lscm`][ordito.parametrization.lscm] instead pins only two vertices and lets the boundary find its
own shape, minimizing conformal rather than Dirichlet energy.

[`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle] supplies the boundary
condition the fixed-boundary three need, and
[`face_flipped_indices`][ordito.parametrization.face_flipped_indices] is the diagnostic that
says whether a result is actually injective. Ports of the corresponding ``igl::`` routines.

The fixed-vertex maps (``harmonic``, ``tutte``, ``lscm``) take the mesh as an
[`ordito.mesh.Trimesh`][ordito.mesh.Trimesh] or as its ``vertices`` and ``faces``. Their solves
iterate, verified, and fall back to a sparse Cholesky factorization where the iteration fails;
given a `Trimesh`, a factorization is kept on it
([`Trimesh.fixed_vertex_solver`][ordito.mesh.Trimesh.fixed_vertex_solver]), and one built ahead
with [`FixedVertexSolver.factor`][ordito.parametrization.FixedVertexSolver.factor] -- or by a
call with ``solver="direct"`` -- turns every later call fixing the same vertices into a factored
solve with no iteration.
"""

from __future__ import annotations

import logging
import weakref
from typing import Literal, cast, overload

import warp as wp

import ordito as od
import ordito.linalg as twl
import ordito.typing as odt
from ordito import _launch
from ordito._device import require_same_device
from ordito.kernels import parametrization as kernel_parametrization
from ordito.laplacian import cotmatrix, cotmatrix_entries, graph_laplacian, mass_matrix_entries
from ordito.mesh import Trimesh, mesh_arguments

_CG_TOLERANCE = 1e-8

_LOGGER = logging.getLogger(__name__)


def face_flipped_mask(vertices: wp.array[wp.vec2], faces: wp.array[wp.int32]) -> wp.array[wp.bool]:
    """
    Per-face flag: whether a triangle is inverted (negative 2D signed area) in the parametrization.

    For each triangle the 2D signed area of its three UV vertices is computed; a face is flagged
    ``True`` when that area is strictly negative, i.e. the triangle has folded over (flipped
    orientation) in the 2D domain. Mirrors libigl's ``flipped_triangles`` per-triangle test
    (determinant of the homogeneous ``3 x 3`` vertex matrix ``< 0``). Degenerate (zero-area)
    triangles are **not** flagged, matching the strict ``< 0`` comparison.
    [`face_flipped_indices`][ordito.parametrization.face_flipped_indices] is the index form of
    this mask.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` 2D vertex positions (the parametrization / UV coordinates).
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_faces,)`` flipped-triangle mask on ``vertices.device``. Empty for an empty mesh.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`face_flipped_indices`][ordito.parametrization.face_flipped_indices]
    [`face_flip_mask`][ordito.validation.face_flip_mask]

    Notes
    -----
    Equivalent to the per-triangle predicate behind libigl ``flipped_triangles``: the 2D cross
    product ``(v1 - v0) x (v2 - v0)`` equals ``det([[x0, x1, x2], [y0, y1, y2], [1, 1, 1]])``, so a
    ``True`` entry corresponds exactly to a triangle libigl would list as flipped.

    This is a different flip from
    [`face_flip_mask`][ordito.validation.face_flip_mask], which is about the *3D winding*: this
    mask flags a triangle whose UV image has folded over, a property of the parametrization that
    says nothing about the surface, while that one flags a triangle whose index order must be
    reversed to agree with its patch. A mesh can be consistently wound and still have flipped UVs,
    and vice versa.
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.bool, device=device)

    out_mask = _launch.empty(n_faces, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_parametrization.face_flipped_mask,
        dim=n_faces,
        inputs=[vertices, faces, out_mask],
        device=device,
    )
    return out_mask


def face_flipped_indices(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32]
) -> wp.array[wp.int32]:
    """
    Return the indices of triangles inverted (negative 2D signed area) in the parametrization.

    Convenience wrapper returning ``flatnonzero`` of
    [`face_flipped_mask`][ordito.parametrization.face_flipped_mask]: the indices into ``faces``
    of triangles whose 2D signed area is strictly negative (folded over in the UV domain). Matches
    libigl's ``flipped_triangles``, which returns the same list of flipped-triangle indices.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` 2D vertex positions (the parametrization / UV coordinates).
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.int32]
        ``(m,)`` ascending face indices of the ``m`` flipped triangles on ``vertices.device``.
        Empty when no
        triangle is flipped.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`face_flipped_mask`][ordito.parametrization.face_flipped_mask]
    [`flatnonzero`][ordito.array.flatnonzero]
    """
    require_same_device(vertices=vertices, faces=faces)
    return od.array.flatnonzero(face_flipped_mask(vertices, faces))


def map_vertices_to_circle(
    vertices: wp.array[wp.vec3], boundary: wp.array[wp.int32]
) -> wp.array[wp.vec2]:
    """
    Map an ordered boundary loop onto the unit circle by arc length.

    Places boundary vertex ``i`` at angle ``2*pi * len[i] / total``, where ``len[i]`` is the
    cumulative edge length from ``boundary[0]`` to ``boundary[i]`` along the loop and ``total`` is
    the full perimeter (closing over the wrap edge). The result is in loop order: row ``i`` is the
    position of ``boundary[i]``. Feed it as ``boundary_uv`` to
    [`tutte`][ordito.parametrization.tutte] or [`harmonic`][ordito.parametrization.harmonic] for a
    disk parametrization.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    boundary
        ``(n_boundary,)`` ordered boundary-loop vertex indices, e.g. from
        [`longest_boundary_loop`][ordito.boundary.longest_boundary_loop].

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_boundary,)`` unit-circle positions on ``vertices.device``, aligned with ``boundary``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``boundary`` are not all on one device.

    Notes
    -----
    The name and the arc-length placement follow ``igl::map_vertices_to_circle``.

    See Also
    --------
    [`longest_boundary_loop`][ordito.boundary.longest_boundary_loop]
    [`tutte`][ordito.parametrization.tutte]
    [`harmonic`][ordito.parametrization.harmonic]
    """
    require_same_device(vertices=vertices, boundary=boundary)
    device = vertices.device
    n_boundary = boundary.size
    out_uv = _launch.empty(n_boundary, dtype=wp.vec2, device=device)
    if n_boundary == 0:
        return out_uv

    segment_lengths = _launch.empty(n_boundary, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_parametrization.boundary_edge_lengths,
        dim=n_boundary,
        inputs=[boundary, vertices, segment_lengths],
        device=device,
    )
    cumulative = _launch.empty(n_boundary, dtype=wp.float32, device=device)
    _launch.array_scan(segment_lengths, out_array=cumulative, inclusive=True)
    _launch.launch(
        kernel_parametrization.circle_positions,
        dim=n_boundary,
        inputs=[boundary, vertices, cumulative, out_uv],
        device=device,
    )
    return out_uv


@overload
def harmonic(
    mesh: Trimesh,
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    /,
    *,
    k: int = 1,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]: ...
@overload
def harmonic(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    /,
    k: int = 1,
    *,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]: ...
def harmonic(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | None = None,
    boundary_indices: wp.array[wp.int32] | wp.array[wp.vec2] | None = None,
    boundary_uv: wp.array[wp.vec2] | None = None,
    k: int = 1,
    *,
    vertices: wp.array[wp.vec3] | None = None,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]:
    """
    Harmonic parametrization with fixed boundary.

    Minimizes the ``k``-harmonic energy built from the cotangent Laplacian
    [`cotmatrix`][ordito.laplacian.cotmatrix] subject to the boundary vertices being pinned to
    ``boundary_uv``. For ``k == 1`` this is the harmonic map (each interior UV is the
    cotangent-weighted average of its neighbors); ``k == 2`` is the biharmonic map, and so on. The
    interior system is solved with conjugate gradient at ``k <= 2`` and by a sparse Cholesky
    factorization above, the iterative result verified and replaced by the factorization's when it
    is not the system's answer (see Notes).

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``vertices`` and ``faces``.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    boundary_indices
        ``(n_boundary,)`` indices of the fixed (constrained) vertices.
    boundary_uv
        ``(n_boundary,)`` target UV positions for ``boundary_indices``, in the same order (e.g. from
        [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]).
    k
        Harmonic power (``>= 1``). ``k > 1`` additionally uses the barycentric lumped mass matrix
        [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries]. The operator is assembled in
        float64 (built native, never recast) so the ill-conditioned ``k > 1`` solve is accurate.
    solver
        ``"auto"`` (default) solves by conjugate gradient, verified, and factors the system only
        when the iteration's result fails the check (always at ``k >= 3``, where the iteration does
        not converge). ``"direct"`` factors it at once (a sparse Cholesky); given a `Trimesh`, the
        factorization is kept on it, so every later call fixing the same vertices -- whatever their
        values -- is a factored solve with no iteration. Pays from about the second such call; a
        one-shot call is usually faster under ``"auto"``.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``k < 1``, if there are interior vertices but ``boundary_indices`` is empty (the
        Dirichlet system would be singular), if ``boundary_indices`` and ``boundary_uv`` have
        different lengths, or if ``solver`` is not ``"auto"`` or ``"direct"``.
    RuntimeError
        If ``vertices``, ``faces``, ``boundary_indices`` and ``boundary_uv`` are not all on one
        device.

    Warns
    -----
    UserWarning
        When no verified solution is reached (see
        [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed]): the interior system cannot be
        factored -- too large, or not definite in ``float64``, which a high ``k`` on a strongly
        graded mesh reaches -- and conjugate gradient does not converge.

    See Also
    --------
    [`tutte`][ordito.parametrization.tutte]
    [`k_harmonic`][ordito.energies.k_harmonic]
        The *operator* this minimizes, as opposed to this *map* -- the two are the only two
        "harmonic" names in the package and they are not interchangeable.
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]

    Notes
    -----
    Matches ``igl::harmonic`` at ``k == 1``. Above, igl's default mass matrix is the Voronoi one
    where this uses the barycentric lumped mass; ``igl::harmonic`` given a barycentric
    ``massmatrix`` is the same map.

    Each ``k`` is solved its own way, since raising the operator to the ``k``-th power raises its
    condition number to it. ``k == 1`` runs conjugate gradient under the Jacobi-Chebyshev polynomial
    ([`chebyshev_preconditioner`][ordito.linalg.chebyshev_preconditioner]), ``k == 2`` under the
    square of that Laplacian's polynomial
    ([`squared_laplacian_preconditioner`][ordito.linalg.squared_laplacian_preconditioner]). Both
    results are verified -- converged within
    [`CG_FACTOR_AFTER_ROUNDS`][ordito.linalg.CG_FACTOR_AFTER_ROUNDS] rounds, with a componentwise
    backward error under
    [`VERIFIED_BACKWARD_ERROR`][ordito.linalg.VERIFIED_BACKWARD_ERROR] -- and a result that is not
    is replaced by a [`sparse_cholesky`][ordito.cholesky.sparse_cholesky] solve: on a strongly
    graded mesh the ``k == 2`` system's condition number nears the reciprocal of ``float64``'s
    precision, where no iteration converges. ``k >= 3`` is factored from the start, the iteration
    (under the multigrid preconditioner
    [`multigrid_preconditioner`][ordito.linalg.multigrid_preconditioner]) taken only when the
    system cannot be factored. Given a `Trimesh`, a factorization built here is kept on it, and
    one prepared with [`FixedVertexSolver.factor`][ordito.parametrization.FixedVertexSolver.factor]
    serves every later call fixing the same vertices -- whatever their values -- with no
    iteration; the ``vertices, faces`` form drops it, and says so at ``INFO``.

    On such an ill-conditioned system the answer itself is determined only to the system's own
    precision: direct solvers under different orderings agree to about a percent of the UV range
    there, as this one does with them.
    """
    bound, owned, arguments = mesh_arguments(
        "harmonic", mesh, vertices, faces, (boundary_indices, boundary_uv, k), 2
    )
    return _fixed_boundary_map(
        bound,
        owned,
        "harmonic",
        cast("wp.array[wp.int32]", arguments[0]),
        cast("wp.array[wp.vec2]", arguments[1]),
        cast("int", arguments[2]),
        solver,
    )


@overload
def tutte(
    mesh: Trimesh,
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    /,
    *,
    k: int = 1,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]: ...
@overload
def tutte(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    /,
    k: int = 1,
    *,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]: ...
def tutte(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | None = None,
    boundary_indices: wp.array[wp.int32] | wp.array[wp.vec2] | None = None,
    boundary_uv: wp.array[wp.vec2] | None = None,
    k: int = 1,
    *,
    vertices: wp.array[wp.vec3] | None = None,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]:
    """
    Tutte embedding with fixed boundary (uniform-Laplacian parametrization).

    Identical to [`harmonic`][ordito.parametrization.harmonic] except the operator is the
    combinatorial [`graph_laplacian`][ordito.laplacian.graph_laplacian] instead of the
    cotangent one — for ``k == 1`` that Laplacian is the *only* difference. Because the uniform
    Laplacian's free-free block is a diagonally dominant M-matrix, the Tutte embedding of a mesh
    with a convex boundary is guaranteed bijective (fold-free), unlike the harmonic/conformal maps.
    For ``k > 1`` the mass matrix is the identity (matching libigl's ``speye`` graph-Laplacian
    variant). Solved, and its factorizations kept, as [`harmonic`][ordito.parametrization.harmonic]
    is (see its Notes).

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``vertices`` and ``faces``.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    boundary_indices
        ``(n_boundary,)`` indices of the fixed (constrained) vertices.
    boundary_uv
        ``(n_boundary,)`` target UV positions for ``boundary_indices``, in the same order (e.g. a
        convex loop from
        [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]).
    k
        Laplacian power (``>= 1``). ``k == 1`` is the classic Tutte embedding.
    solver
        ``"auto"`` (default) solves by conjugate gradient, verified, and factors the system only
        when the iteration's result fails the check (always at ``k >= 3``). ``"direct"`` factors it
        at once (a sparse Cholesky); given a `Trimesh`, the factorization is kept on it, so every
        later call fixing the same vertices -- whatever their values -- is a factored solve with no
        iteration. Pays from about the second such call; a one-shot call is usually faster under
        ``"auto"``.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``k < 1``, if there are interior vertices but ``boundary_indices`` is empty, if
        ``boundary_indices`` and ``boundary_uv`` have different lengths, or if ``solver`` is not
        ``"auto"`` or ``"direct"``.
    RuntimeError
        If ``vertices``, ``faces``, ``boundary_indices`` and ``boundary_uv`` are not all on one
        device.

    See Also
    --------
    [`harmonic`][ordito.parametrization.harmonic]
    [`graph_laplacian`][ordito.laplacian.graph_laplacian]
    [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]
    """
    bound, owned, arguments = mesh_arguments(
        "tutte", mesh, vertices, faces, (boundary_indices, boundary_uv, k), 2
    )
    return _fixed_boundary_map(
        bound,
        owned,
        "tutte",
        cast("wp.array[wp.int32]", arguments[0]),
        cast("wp.array[wp.vec2]", arguments[1]),
        cast("int", arguments[2]),
        solver,
    )


def _fixed_boundary_map(
    mesh: Trimesh,
    owned: bool,
    method: str,
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    k: int,
    solver: str,
) -> wp.array[wp.vec2]:
    """``harmonic`` / ``tutte`` on a resolved mesh: validate, pin the boundary, solve, scatter."""
    vertices = mesh.vertices
    require_same_device(
        vertices=vertices,
        faces=mesh.faces,
        boundary_indices=boundary_indices,
        boundary_uv=boundary_uv,
    )
    if k < 1:
        raise ValueError(f"{method} power k must be >= 1, got {k}.")
    _require_solver_name(solver, method)
    device = vertices.device
    n_vertices = vertices.size
    if n_vertices == 0:
        return _launch.empty(0, dtype=wp.vec2, device=device)
    # A mesh with interior vertices and no fixed boundary is a singular Dirichlet system; checked
    # before any operator is assembled.
    _require_fixed_vertices(
        boundary_indices.size,
        n_vertices,
        1,
        "harmonic / tutte require at least one fixed boundary vertex; the Dirichlet system is "
        "otherwise singular.",
    )
    fixed_mask, fixed_values = _scatter_constraints(
        n_vertices, boundary_indices, boundary_uv, device
    )
    fixed_values_2d = odt.as_array2d(fixed_values, wp.float64)
    sol, free_map = _solve_fixed_vertices(
        mesh, owned, method, k, fixed_mask, fixed_values_2d, solver
    )
    out_uv = _launch.empty(n_vertices, dtype=wp.vec2, device=device)
    _launch.launch(
        kernel_parametrization.scatter_solution,
        dim=n_vertices,
        inputs=[fixed_mask, free_map, sol, fixed_values_2d, out_uv],
        device=device,
    )
    return out_uv


def arap(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    fixed_indices: wp.array[wp.int32],
    fixed_uv: wp.array[wp.vec2],
    uv_init: wp.array[wp.vec2],
    max_iterations: int = 10,
    tolerance: float = 1e-7,
) -> wp.array[wp.vec2]:
    """

    As-rigid-as-possible (ARAP) parametrization with fixed vertices.

    Minimizes the ARAP energy of the 2D parametrization by local/global alternation, starting from
    ``uv_init`` and keeping ``fixed_indices`` pinned to ``fixed_uv`` at every iteration. The local
    step fits, per triangle, the closest rotation between the isometrically flattened rest triangle
    and its current UV image (closed-form 2D polar decomposition, reflections forbidden); the global
    step solves the cotangent-Laplacian Poisson system ``(-L)_uu U_u = (K R)_u - (-L)_ub bc`` for
    each UV column with conjugate gradient.

    A good ``uv_init`` matters: ARAP is non-convex, so feed a fold-free initial map such as
    [`harmonic`][ordito.parametrization.harmonic] or [`tutte`][ordito.parametrization.tutte], with
    the boundary placed by
    [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]. Inspect the result
    for inverted triangles with
    [`face_flipped_indices`][ordito.parametrization.face_flipped_indices].


    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    fixed_indices
        ``(n_fixed,)`` indices of the pinned (constrained) vertices — libigl's ``b``. At least one
        is required whenever the mesh has interior vertices (the ARAP global system is otherwise a
        singular, translation-invariant Poisson problem). Pinning the whole boundary loop reproduces
        the classic fixed-boundary ARAP disk parametrization.
    fixed_uv
        ``(n_fixed,)`` target UV positions for ``fixed_indices``, in the same order — libigl's
        ``bc``.
    uv_init
        ``(n_vertices,)`` initial UV coordinates (the warm start). Never mutated; the interior
        values seed the first local step and the conjugate-gradient warm start, and the pinned rows
        are
        overwritten with ``fixed_uv`` before the first iteration.
    max_iterations
        Number of local/global iterations (``>= 1``). libigl defaults to ``10``.
    tolerance
        Relative residual tolerance of the inner conjugate-gradient solve (``> 0``). See Notes for
        why the default is looser than the ``1e-8`` the other solvers in this module use.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``. Empty for an empty mesh. Rows at
        ``fixed_indices`` equal ``fixed_uv`` exactly.

    Raises
    ------
    ValueError
        If ``max_iterations < 1``, if ``tolerance <= 0``, if there are interior vertices but
        ``fixed_indices`` is empty, or if ``fixed_indices`` and ``fixed_uv`` have different lengths.
    RuntimeError
        If ``vertices``, ``faces``, ``fixed_indices``, ``fixed_uv`` and ``uv_init`` are not all on
        one device.

    See Also
    --------
    [`harmonic`][ordito.parametrization.harmonic]
    [`tutte`][ordito.parametrization.tutte]
    [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]
    [`face_flipped_indices`][ordito.parametrization.face_flipped_indices]
    [`cotmatrix_entries`][ordito.laplacian.cotmatrix_entries]

    Notes
    -----
    Uses the *elements* ARAP energy (one rotation per triangle), libigl's default for the flat ``dim
    = 2`` parametrization case; the covariance scatter is built per corner of the flattened mesh,
    which is why the ``SPOKES`` / ``SPOKES_AND_RIMS`` (per-vertex) energies do not apply here. The
    half-cotangent weights ``c_e`` come from
    [`cotmatrix_entries`][ordito.laplacian.cotmatrix_entries] with no clamping (matching libigl),
    and the closest 2D rotation is the closed form ``theta = atan2(S10 - S01, S00 + S11)`` of
    ``igl::fit_rotations_planar``. The cotangent operator and the two conjugate-gradient solves run
    in float64 for determinism while the UV field is stored ``float32`` between iterations (module
    convention); the resulting per-iteration drift is well under the pinned-boundary tolerance for
    the default iteration count.

    **Why ``tolerance`` defaults to ``1e-7`` and not ``1e-8``.** Unlike
    [`harmonic`][ordito.parametrization.harmonic] or [`lscm`][ordito.parametrization.lscm], whose
    single solve *is* the answer, ARAP's global solves are inner steps of a truncated outer
    iteration, so solving more accurately than the outer iteration's own truncation error wastes
    work without changing the result. Pass ``tolerance=1e-8`` for stricter per-iteration solves;
    going looser risks the inner error exceeding the outer truncation error on large meshes.
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        fixed_indices=fixed_indices,
        fixed_uv=fixed_uv,
        uv_init=uv_init,
    )
    if max_iterations < 1:
        raise ValueError(f"arap max_iterations must be >= 1, got {max_iterations}.")
    if tolerance <= 0.0:
        raise ValueError(f"arap tolerance must be > 0, got {tolerance}.")
    device = vertices.device
    n_vertices = vertices.size
    if n_vertices == 0:
        return _launch.empty(0, dtype=wp.vec2, device=device)

    # Interior vertices with nothing pinned leave the ARAP global system translation-invariant
    # (singular) -- and since n_vertices > 0, an empty ``fixed_indices`` always means at least one
    # interior vertex exists. Checked before the cotangent/Laplacian build below (mirroring
    # harmonic / tutte), so a rejected call doesn't pay for the assembly first.
    _require_fixed_vertices(
        fixed_indices.size,
        n_vertices,
        1,
        "arap requires at least one fixed vertex when the mesh has interior vertices; the ARAP "
        "global system is otherwise singular (translation invariant).",
    )
    n_faces = faces.size // 3

    # Cotangents computed once and reused by both the Laplacian build and the rest-edge flattening;
    # single native-float64 operator build, so nothing is recast or rebuilt (see cotmatrix docs).
    cot_entries = cotmatrix_entries(vertices, faces, dtype=wp.float64)
    laplacian = cotmatrix(vertices, faces, cot_entries=cot_entries, dtype=wp.float64)

    # Partition vertices into pinned (fixed) and interior (free). ``fixed_values`` is the
    # (2, n_vertices) prescribed-UV buffer (row 0 = u, row 1 = v) shared with the assembly / scatter
    # kernels; ``interior_map`` compacts free vertices into the reduced system.
    fixed_mask, fixed_values = _scatter_constraints(n_vertices, fixed_indices, fixed_uv, device)
    fixed_values_2d = odt.as_array2d(fixed_values, wp.float64)
    interior_map, n_interior = twl.free_partition(fixed_mask)

    out_uv = _launch.empty(n_vertices, dtype=wp.vec2, device=device)
    if n_interior == 0:
        # Every vertex pinned: the prescribed positions are the whole answer, no solve. This is the
        # same shape `min_quad_with_fixed`'s own `n_free == 0` branch returns (an empty `(n_rhs,
        # n_free)` solution `scatter_solution` never indexes), but composing through it here would
        # cost more than it shares: `min_quad_with_fixed` re-derives `free_partition` internally
        # (arap already has `interior_map`/`n_interior` from its own call, needed by the iterative
        # loop below regardless) and its single-shot solve has no hook for the warm-started,
        # loop-reused solver state arap's general path builds -- so reaching it would mean throwing
        # away work already done, not reusing it. The real waste this early return avoids is
        # upstream of any solve: `q_uu`/`rhs_const`/`rest_edges`/the CG solver state below are built
        # only for a loop that would immediately do nothing on an empty system, so skipping them
        # here is the point, not a shortcut around composition.
        empty_sol = _launch.zeros((2, 0), dtype=wp.float64, device=device)
        _launch.launch(
            kernel_parametrization.scatter_solution,
            dim=n_vertices,
            inputs=[fixed_mask, interior_map, empty_sol, fixed_values_2d, out_uv],
            device=device,
        )
        return out_uv

    # Global-step operator: interior block of Q = -L and the constant boundary term -(-L)_ub bc.
    # ``future work``: libigl also supports rotation groups ``G`` (shared rotations across grouped
    # faces, replacing the per-face fit with a group-summed covariance) and ``with_dynamics`` (a
    # mass-matrix + timestep term added to Q and the right-hand side); both are out of scope here.
    q_uu, rhs_const = twl.assemble_interior_system(
        laplacian, fixed_mask, interior_map, fixed_values_2d, n_interior, scale=-1.0
    )

    # Weight-folded rest edges of the isometrically flattened triangles (internal buffer, plain
    # wp.empty; kernels index it as wp.array2d).
    rest_edges = _launch.empty((n_faces, 3), dtype=wp.vec2d, device=device)
    # Pre-loop buffers (no allocation inside the loop). ``sol`` (2, n_interior) holds the
    # warm-started CG solution per column; the same launch seeds it from ``uv_init``'s interior
    # values and writes the working ``out_uv`` with the constraints enforced for iteration 1. Every
    # row of ``sol`` is seeded, so it is allocated uninitialised.
    sol = _launch.empty((2, n_interior), dtype=wp.float64, device=device)
    _launch.launch(
        kernel_parametrization.arap_setup,
        dim=max(n_faces, n_vertices),
        inputs=[vertices, faces, cot_entries, fixed_mask, interior_map, uv_init, fixed_values_2d],
        outputs=[rest_edges, sol, out_uv],
        device=device,
    )
    rhs_rot_x = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    rhs_rot_y = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    b = _launch.empty((2, n_interior), dtype=wp.float64, device=device)

    # One batched CG state for both UV columns, built once outside the loop: its temporaries and
    # batch layout are reused across iterations, and it reads ``b`` / writes ``sol`` in place, so
    # the warm start is simply whatever ``sol`` already holds.
    solver = twl.spd_column_solver(
        q_uu,
        odt.as_array2d(b, wp.float64),
        odt.as_array2d(sol, wp.float64),
        tol=tolerance,
        maxiter=10 * n_interior,
        preconditioner="chebyshev",
        # The global steps are warm-started, short Jacobi-Chebyshev solves of an operator rebuilt
        # per call, which a factorization on the second step does not beat: ARAP keeps iterating.
        factorize=False,
    )

    for _ in range(max_iterations):
        _launch.zero_(rhs_rot_x)
        _launch.zero_(rhs_rot_y)
        # Local step: fit per-face rotations from ``out_uv`` and scatter the rotation RHS.
        _launch.launch(
            kernel_parametrization.arap_local_step,
            dim=n_faces,
            inputs=[faces, out_uv, rest_edges, rhs_rot_x, rhs_rot_y],
            device=device,
        )
        # Global step RHS: constant boundary term + rotation term, restricted to interior rows.
        _launch.launch(
            kernel_parametrization.arap_interior_rhs,
            dim=n_vertices,
            inputs=[fixed_mask, interior_map, rhs_const, rhs_rot_x, rhs_rot_y, b],
            device=device,
        )
        # Both UV columns in one batched symmetric-PD solve, warm-started from the previous ``sol``.
        solver()
        # Reconstruct the full UV field, re-enforcing the pinned constraints for the next iteration.
        _launch.launch(
            kernel_parametrization.scatter_solution,
            dim=n_vertices,
            inputs=[fixed_mask, interior_map, sol, fixed_values_2d, out_uv],
            device=device,
        )
    return out_uv


def _scatter_constraints(
    n_vertices: int, indices: wp.array[wp.int32], uv: wp.array[wp.vec2], device: wp.DeviceLike
) -> tuple[wp.array[wp.bool], wp.array[wp.float64]]:
    """
    Expand a list of pinned vertices into the dense mask and prescribed-UV buffers the solvers take.

    ``fixed_mask`` marks the constrained vertices; ``fixed_values`` is the ``(2, n_vertices)``
    prescribed-UV buffer (row 0 = u, row 1 = v) the assembly and scatter kernels read. Shared by
    ``_fixed_boundary_map`` and
    [`arap`][ordito.parametrization.arap]; an empty ``indices`` yields an all-``False`` mask and
    an all-zero value buffer, which each caller rejects on its own terms.

    Raises
    ------
    ValueError
        If ``indices`` and ``uv`` have different lengths -- the scatter kernel below indexes ``uv``
        at every position up to ``indices.shape[0]``, so a shorter ``uv`` is an out-of-bounds read.
    """
    n_fixed = indices.size
    if uv.size != n_fixed:
        raise ValueError(f"indices and uv must have the same length, got {n_fixed} and {uv.size}.")
    # One scatter marks the mask and writes the values, skipping an out-of-range index in both.
    fixed_mask = _launch.zeros(n_vertices, dtype=wp.bool, device=device)
    fixed_values = _launch.zeros((2, n_vertices), dtype=wp.float64, device=device)
    if n_fixed > 0:
        _launch.launch(
            kernel_parametrization.scatter_fixed_uv,
            dim=n_fixed,
            inputs=[indices, uv, fixed_mask, fixed_values],
            device=device,
        )
    return fixed_mask, fixed_values


@overload
def lscm(
    mesh: Trimesh,
    pinned_indices: wp.array[wp.int32],
    pinned_uv: wp.array[wp.vec2],
    /,
    *,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]: ...
@overload
def lscm(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    pinned_indices: wp.array[wp.int32],
    pinned_uv: wp.array[wp.vec2],
    /,
    *,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]: ...
def lscm(
    mesh: Trimesh | wp.array[wp.vec3] | None = None,
    faces: wp.array[wp.int32] | None = None,
    pinned_indices: wp.array[wp.int32] | wp.array[wp.vec2] | None = None,
    pinned_uv: wp.array[wp.vec2] | None = None,
    *,
    vertices: wp.array[wp.vec3] | None = None,
    solver: Literal["auto", "direct"] = "auto",
) -> wp.array[wp.vec2]:
    """
    Constrained least-squares conformal map.

    Computes the conformal (angle-preserving) parametrization that minimizes the LSCM (Levy)
    conformal energy subject to a set of pinned vertices, by solving a single quadratic program
    over the stacked ``[u; v]`` vector of ``2 * n_vertices`` unknowns with the LSCM Hessian
    [`lscm_hessian`][ordito.energies.lscm_hessian] as the operator. Unlike
    [`harmonic`][ordito.parametrization.harmonic] / [`tutte`][ordito.parametrization.tutte] (which
    pin the whole boundary and solve two independent columns), LSCM couples ``u`` and ``v`` through
    the boundary vector-area term, so it needs only a few pins — typically **two** — to fix the
    remaining similarity-transform (rotation + scale + translation) degree of freedom.

    The interior system is solved with conjugate gradient, verified as
    [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] verifies it, with a sparse Cholesky
    factorization taking over a solve that does not converge within its budget. Closed meshes are
    valid input: the boundary vector-area matrix is then zero and the Hessian reduces to
    ``-repdiag(L, 2)``. Given a `Trimesh`,
    a factorization is kept on it; one prepared with
    [`FixedVertexSolver.factor`][ordito.parametrization.FixedVertexSolver.factor] (``"lscm"``, the
    pinned indices) answers every later call pinning the same vertices with no iteration.

    Parameters
    ----------
    mesh
        The mesh, as an [`ordito.mesh.Trimesh`][ordito.mesh.Trimesh]; or, in its place,
        ``vertices`` and ``faces``.
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    pinned_indices
        ``(n_pinned,)`` indices of the pinned (constrained) vertices — igl's ``b``. At least two are
        required (unless the mesh has fewer than two vertices) to remove the conformal map's
        similarity-transform null space.
    pinned_uv
        ``(n_pinned,)`` target UV positions for ``pinned_indices``, in the same order (igl's
        ``bc``).
    solver
        ``"auto"`` (default) solves by conjugate gradient, verified, and factors the system only
        when the iteration's result fails the check. ``"direct"`` factors it at once (a sparse
        Cholesky); given a `Trimesh`, the factorization is kept on it, so every later call fixing
        the same vertices -- whatever their values -- is a factored solve with no iteration. Pays
        from about the second such call; a one-shot call is usually faster under ``"auto"``.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``. Empty for an empty mesh.

    Raises
    ------
    ValueError
        If fewer than two vertices are pinned (and the mesh has at least two vertices), if
        ``pinned_indices`` and ``pinned_uv`` have different lengths, or if ``solver`` is not
        ``"auto"`` or ``"direct"``.
    RuntimeError
        If ``vertices``, ``faces``, ``pinned_indices`` and ``pinned_uv`` are not all on one device.

    Warns
    -----
    UserWarning
        When no verified solution is reached (see
        [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed]).

    See Also
    --------
    [`lscm_hessian`][ordito.energies.lscm_hessian]
    [`vector_area_matrix`][ordito.energies.vector_area_matrix]
    [`harmonic`][ordito.parametrization.harmonic]
    [`face_flipped_indices`][ordito.parametrization.face_flipped_indices]

    Notes
    -----
    The unknowns are stacked ``[u; v]`` (all ``u`` DOFs then all ``v`` DOFs), matching igl's
    ``lscm``: pin ``i`` fixes DOF ``i`` (``u``) and DOF ``i + n_vertices`` (``v``). The returned
    ``Q`` of ``igl.lscm`` equals ``-repdiag(L, 2) - 2 A`` exactly (see
    [`lscm_hessian`][ordito.energies.lscm_hessian]).
    """
    bound, owned, arguments = mesh_arguments(
        "lscm", mesh, vertices, faces, (pinned_indices, pinned_uv), 2
    )
    pinned_indices = cast("wp.array[wp.int32]", arguments[0])
    pinned_uv = cast("wp.array[wp.vec2]", arguments[1])
    _require_solver_name(solver, "lscm")
    require_same_device(
        vertices=bound.vertices,
        faces=bound.faces,
        pinned_indices=pinned_indices,
        pinned_uv=pinned_uv,
    )
    device = bound.device
    n = bound.n_vertices
    if n == 0:
        return _launch.empty(0, dtype=wp.vec2, device=device)

    n_pinned = pinned_indices.size
    _require_fixed_vertices(
        n_pinned,
        n,
        2,
        "lscm requires at least two pinned vertices to remove the conformal map's "
        f"similarity-transform null space; got {n_pinned}.",
    )
    fixed_mask, fixed_values = _pin_stacked(n, pinned_indices, pinned_uv, device)
    sol, free_map = _solve_fixed_vertices(
        bound, owned, "lscm", 1, fixed_mask, odt.as_array2d(fixed_values, wp.float64), solver
    )
    out_uv = _launch.empty(n, dtype=wp.vec2, device=device)
    _launch.launch(
        kernel_parametrization.scatter_solution_stacked,
        dim=n,
        inputs=[fixed_mask, free_map, sol[0], fixed_values[0], out_uv],
        device=device,
    )
    return out_uv


_FIXED_VERTEX_METHODS = ("harmonic", "tutte", "lscm")


class FixedVertexSolver:
    """
    Factorizations of the fixed-vertex maps' systems on one mesh, kept by the mesh.

    What [`Trimesh.fixed_vertex_solver`][ordito.mesh.Trimesh.fixed_vertex_solver] returns, and what
    [`harmonic`][ordito.parametrization.harmonic], [`tutte`][ordito.parametrization.tutte] and
    [`lscm`][ordito.parametrization.lscm] consult when given a `Trimesh`. Each map solves the
    system its operator reduces to once the fixed vertices are eliminated, and that system depends
    on *which* vertices are fixed -- not on the values they are fixed to -- so one factorization
    per ``(method, k)`` serves every later call that fixes the same vertices: such a call runs no
    iteration, only the factorization's solve.

    A factorization is kept here when [`factor`][ordito.parametrization.FixedVertexSolver.factor]
    builds it ahead of the calls, when a call asks for one (``solver="direct"``), or when a call
    built one anyway: the iteration's result failed
    its verification (see [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed]), or the
    system is factored from the start (``k >= 3``). A call fixing other vertices than the kept
    factorization's solves as if none were kept, and replaces it with its own when it builds one.

    Notes
    -----
    A kept factorization holds device memory of the order of its system's fill-in for as long as
    this solver lives -- the mesh's lifetime -- reported by
    [`nbytes`][ordito.parametrization.FixedVertexSolver.nbytes] and returned by
    [`release`][ordito.parametrization.FixedVertexSolver.release] or
    [`Trimesh.release_factorizations`][ordito.mesh.Trimesh.release_factorizations]. The solver
    refers to its mesh weakly and is unusable once the mesh is gone.

    See Also
    --------
    [`Trimesh.fixed_vertex_solver`][ordito.mesh.Trimesh.fixed_vertex_solver]
    [`MinQuadWithFixedData`][ordito.linalg.MinQuadWithFixedData]
    """

    def __init__(self, mesh: Trimesh) -> None:
        """Bind a solver to ``mesh``; nothing is assembled until a factorization is asked for."""
        self._mesh = weakref.ref(mesh)
        self._systems: dict[tuple[str, int], twl.MinQuadWithFixedData] = {}

    @property
    def nbytes(self) -> int:
        """Device memory this solver's kept factorizations hold, in bytes."""
        return sum(system.nbytes for system in self._systems.values())

    def release(self) -> None:
        """Drop every kept factorization."""
        for system in self._systems.values():
            system.release()
        self._systems.clear()

    def factor(self, method: str, fixed_indices: wp.array[wp.int32], *, k: int = 1) -> bool:
        """
        Factor ``method``'s system with ``fixed_indices`` fixed, for every later call fixing them.

        Parameters
        ----------
        method
            ``"harmonic"``, ``"tutte"`` or ``"lscm"``: the map the factorization serves.
        fixed_indices
            ``(n_fixed,)`` the vertices the calls fix: ``boundary_indices`` of
            [`harmonic`][ordito.parametrization.harmonic] / [`tutte`][ordito.parametrization.tutte],
            ``pinned_indices`` of [`lscm`][ordito.parametrization.lscm]. Only the set matters,
            not the order.
        k
            The harmonic power the calls pass (``1`` for ``"lscm"``).

        Returns
        -------
        bool
            Whether the calls will run no iteration: ``False`` when the system cannot be factored
            (over [`CHOLESKY_MEMORY_BUDGET`][ordito.cholesky.CHOLESKY_MEMORY_BUDGET], or not
            definite in ``float64``), and they keep iterating.

        Raises
        ------
        ValueError
            If ``method`` is not one of the three, if ``k < 1``, or if ``k != 1`` for ``"lscm"``.
        RuntimeError
            If ``fixed_indices`` is not on the mesh's device.
        """
        _require_fixed_vertex_method(method, k, "FixedVertexSolver.factor")
        mesh = self._bound_mesh()
        require_same_device(vertices=mesh.vertices, fixed_indices=fixed_indices)
        n_vertices = mesh.n_vertices
        if n_vertices == 0:
            return True
        no_uv = _launch.zeros(fixed_indices.size, dtype=wp.vec2, device=mesh.device)
        if method == "lscm":
            fixed_mask, _ = _pin_stacked(n_vertices, fixed_indices, no_uv, mesh.device)
        else:
            fixed_mask, _ = _scatter_constraints(n_vertices, fixed_indices, no_uv, mesh.device)
        system, _ = _fixed_vertex_system(mesh, method, k, fixed_mask, factor=True)
        if system.factored:
            self._keep(method, k, system)
        return system.factored or system.n_free == 0

    def solve(
        self,
        method: str,
        fixed_mask: wp.array[wp.bool],
        fixed_values: odt.Array2dFloat,
        *,
        k: int = 1,
        solver: Literal["auto", "direct"] = "auto",
    ) -> tuple[odt.Array2dFloat, wp.array[wp.int32]]:
        """
        Solve ``method``'s reduced system: by the kept factorization if it fixes ``fixed_mask``.

        Otherwise the system is assembled and solved verified, as
        [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] solves it, and a factorization
        that solve builds is kept here, replacing the one ``(method, k)`` held.

        Parameters
        ----------
        method
            ``"harmonic"``, ``"tutte"`` or ``"lscm"``.
        fixed_mask
            ``(n_dofs,)`` the fixed degrees of freedom: ``(n_vertices,)`` for ``harmonic`` /
            ``tutte``, ``(2 * n_vertices,)`` (``u`` then ``v``) for ``lscm``.
        fixed_values
            ``(n_columns, n_dofs)`` prescribed values; only the fixed entries are read.
        k
            The harmonic power (``1`` for ``"lscm"``).
        solver
            ``"auto"`` iterates, verified, and factors only when the check fails (or at
            ``k >= 3``); ``"direct"`` factors at once. A factorization either builds is kept.

        Returns
        -------
        solution : odt.Array2dFloat
            ``(n_columns, n_free)`` values of the free degrees of freedom.
        free_map : wp.array[wp.int32]
            ``(n_dofs,)`` compact index of each free degree of freedom.

        Raises
        ------
        ValueError
            If ``method`` is not one of the three, if ``k < 1``, if ``k != 1`` for ``"lscm"``, or
            if ``solver`` is not ``"auto"`` or ``"direct"``.
        RuntimeError
            If ``fixed_mask`` and ``fixed_values`` are not on the mesh's device.

        Warns
        -----
        UserWarning
            When no verified solution is reached (see
            [`min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed]).
        """
        _require_fixed_vertex_method(method, k, "FixedVertexSolver.solve")
        _require_solver_name(solver, "FixedVertexSolver.solve")
        mesh = self._bound_mesh()
        require_same_device(
            vertices=mesh.vertices, fixed_mask=fixed_mask, fixed_values=fixed_values
        )
        system = self._matching(method, k, fixed_mask)
        preconditioner: str | twl.SquaredLaplacianPreconditioner = "adaptive"
        built = system is None
        if system is None:
            system, preconditioner = _fixed_vertex_system(
                mesh,
                method,
                k,
                fixed_mask,
                factor=solver == "direct" or (method != "lscm" and k >= 3),
            )
        solution, free_map, _ = system.solve(
            fixed_values, tol=_CG_TOLERANCE, preconditioner=preconditioner
        )
        if built and system.factored:
            self._keep(method, k, system)
            _LOGGER.debug("%s: kept the factorization of its k=%d system", method, k)
        return solution, free_map

    def _matching(
        self, method: str, k: int, fixed_mask: wp.array[wp.bool]
    ) -> twl.MinQuadWithFixedData | None:
        """Return the kept system of ``(method, k)`` when it fixes exactly ``fixed_mask``."""
        system = self._systems.get((method, k))
        if system is not None and system.matches(fixed_mask):
            return system
        return None

    def _keep(self, method: str, k: int, system: twl.MinQuadWithFixedData) -> None:
        """Keep ``system``'s factorization for ``(method, k)``, releasing the one it replaces."""
        previous = self._systems.get((method, k))
        if previous is not None and previous is not system:
            previous.release()
        self._systems[(method, k)] = system

    def _bound_mesh(self) -> Trimesh:
        """Return the mesh this solver belongs to."""
        mesh = self._mesh()
        if mesh is None:
            raise RuntimeError("FixedVertexSolver: its Trimesh no longer exists")
        return mesh


def _solve_fixed_vertices(
    mesh: Trimesh,
    owned: bool,
    method: str,
    k: int,
    fixed_mask: wp.array[wp.bool],
    fixed_values: odt.Array2dFloat,
    solver: str,
) -> tuple[odt.Array2dFloat, wp.array[wp.int32]]:
    """
    Solve a fixed-vertex map's reduced system on the mesh's solver, or on one made for the call.

    A caller-owned mesh keeps whatever factorization the solve builds; a ``vertices, faces`` call
    drops it, saying so at ``INFO``.
    """
    held = mesh.fixed_vertex_solver() if owned else FixedVertexSolver(mesh)
    solution, free_map = held.solve(
        method, fixed_mask, fixed_values, k=k, solver=cast("Literal['auto', 'direct']", solver)
    )
    if not owned and held.nbytes > 0:
        _LOGGER.info(
            "%s: built a sparse Cholesky factorization (%d bytes) and is discarding it; pass an "
            "ordito.mesh.Trimesh to keep it for later calls fixing the same vertices",
            method,
            held.nbytes,
        )
        held.release()
    return solution, free_map


def _require_solver_name(solver: str, caller: str) -> None:
    """Validate the fixed-vertex maps' ``solver`` menu argument."""
    if solver not in ("auto", "direct"):
        raise ValueError(f'{caller}: solver must be "auto" or "direct", got {solver!r}.')


def _require_fixed_vertex_method(method: str, k: int, caller: str) -> None:
    """Validate a ``FixedVertexSolver`` menu argument and its harmonic power."""
    if method not in _FIXED_VERTEX_METHODS:
        raise ValueError(
            f"{caller}: method must be one of {list(_FIXED_VERTEX_METHODS)}, got {method!r}."
        )
    if k < 1 or (method == "lscm" and k != 1):
        raise ValueError(f"{caller}: k={k} is not a power {method} takes.")


def _fixed_vertex_system(
    mesh: Trimesh, method: str, k: int, fixed_mask: wp.array[wp.bool], *, factor: bool
) -> tuple[twl.MinQuadWithFixedData, str | twl.SquaredLaplacianPreconditioner]:
    """
    Assemble ``method``'s reduced system over ``fixed_mask``'s free set, and its preconditioner.

    ``harmonic`` / ``tutte`` minimize ``Q = -L`` at ``k == 1`` and ``Q = (-L) (M^-1 (-L))^(k-1)``
    above ([`k_harmonic`][ordito.energies.k_harmonic]; ``M`` the barycentric lumped mass, the
    identity for ``tutte``), ``lscm`` its Hessian; the operator is ``float64``, since ``k > 1``
    raises the Laplacian's condition number to the ``k``-th power. The preconditioner is the one
    the iteration runs under when the system is not factored: the Jacobi-Chebyshev polynomial for a
    Laplacian (``k == 1``) or ``lscm`` -- with only the boundary or two vertices fixed their solves
    are long -- and for ``k == 2`` the square of the Laplacian's polynomial
    ([`squared_laplacian_preconditioner`][ordito.linalg.squared_laplacian_preconditioner]): the
    free block of ``L M^-1 L`` is spectrally close to ``L_ff D^-2 L_ff`` with ``D = sqrt(M_f)``,
    the terms it drops being the one ring of the fixed boundary. ``k >= 3`` is factored from the
    start: conjugate gradient needs thousands of rounds there and still leaves an error of 1e-5 to
    1e-2 of the range, so the multigrid ``"auto"`` route remains only for a system the
    factorization refuses.
    """
    vertices, faces = mesh.vertices, mesh.faces
    if method == "lscm":
        q = od.energies.lscm_hessian(vertices, faces)
        return twl.MinQuadWithFixedData(q, fixed_mask, vertices, factor=factor), "chebyshev"
    if method == "harmonic":
        laplacian = cotmatrix(vertices, faces, dtype=wp.float64)
        mass_diag = mass_matrix_entries(vertices, faces, dtype=wp.float64) if k > 1 else None
    else:
        laplacian = graph_laplacian(vertices, faces, dtype=wp.float64)
        mass_diag = None
    q = od.energies.k_harmonic(laplacian, mass_diag, k=k)
    system = twl.MinQuadWithFixedData(q, fixed_mask, vertices, factor=factor)
    if k == 1:
        return system, "chebyshev"
    if k >= 3 or system.factored or system.n_free == 0:
        return system, "auto"
    device = fixed_mask.device
    n_vertices = vertices.size
    no_values = odt.as_array2d(
        _launch.empty((0, n_vertices), dtype=wp.float64, device=device), wp.float64
    )
    # ``-L``'s free block, negated as it is extracted.
    l_ff, _ = twl.assemble_interior_system(
        laplacian, fixed_mask, system.free_map, no_values, system.n_free, scale=-1.0
    )
    if mass_diag is None:
        roots = _launch.full(system.n_free, 1.0, dtype=wp.float64, device=device)
    else:
        roots = _launch.empty(system.n_free, dtype=wp.float64, device=device)
        _launch.launch(
            kernel_parametrization.free_mass_roots,
            dim=n_vertices,
            inputs=[fixed_mask, system.free_map, mass_diag, roots],
            device=device,
        )
    return system, twl.squared_laplacian_preconditioner(l_ff, roots)


def _pin_stacked(
    n_vertices: int, indices: wp.array[wp.int32], uv: wp.array[wp.vec2], device: wp.DeviceLike
) -> tuple[wp.array[wp.bool], wp.array[wp.float64]]:
    """
    ``lscm``'s pins over the stacked ``[u; v]`` unknowns: pin ``i`` fixes DOF ``i`` and ``i + n``.

    Raises
    ------
    ValueError
        If ``indices`` and ``uv`` have different lengths -- the scatter kernel indexes ``uv`` at
        every position up to ``indices.shape[0]``, so a shorter ``uv`` is an out-of-bounds read.
    """
    n_pinned = indices.size
    if uv.size != n_pinned:
        raise ValueError(
            f"pinned_indices and pinned_uv must have the same length, got {n_pinned} and {uv.size}."
        )
    fixed_mask = _launch.zeros(2 * n_vertices, dtype=wp.bool, device=device)
    fixed_values = _launch.zeros((1, 2 * n_vertices), dtype=wp.float64, device=device)
    if n_pinned > 0:
        _launch.launch(
            kernel_parametrization.scatter_pinned_stacked,
            dim=n_pinned,
            inputs=[indices, uv, wp.int32(n_vertices), fixed_mask, fixed_values],
            device=device,
        )
    return fixed_mask, fixed_values


def _require_fixed_vertices(n_fixed: int, n_vertices: int, min_required: int, message: str) -> None:
    """
    Raise ``ValueError(message)`` unless at least ``min_required`` vertices are fixed.

    Waived when the mesh has fewer than ``min_required`` vertices, since it is then impossible to
    pin that many distinct ones. The shared arithmetic behind three otherwise-differently-worded
    guards: harmonic / tutte's "at least one fixed boundary vertex" (``min_required=1``, via
    ``_fixed_boundary_map``), arap's "at least one
    fixed vertex" (``min_required=1``) and lscm's "at least two pinned vertices"
    (``min_required=2``). Each caller keeps its own message, since the *reason* the count is
    required differs (a singular Dirichlet system, translation invariance, a similarity-transform
    null space) even though the arithmetic is identical.
    """
    if n_vertices >= min_required and n_fixed < min_required:
        raise ValueError(message)
