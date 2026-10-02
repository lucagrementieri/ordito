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
says whether a result is actually injective. Ports of the corresponding ``igl::`` routines; every
solve is conjugate-gradient.
"""

from __future__ import annotations

import warp as wp

import ordito as od
import ordito.linalg as twl
import ordito.typing as odt
from ordito import _launch
from ordito._device import require_same_device
from ordito.kernels import parametrization as kernel_parametrization
from ordito.laplacian import cotmatrix, cotmatrix_entries, graph_laplacian, mass_matrix_entries

_CG_TOLERANCE = 1e-8


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


def harmonic(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    k: int = 1,
) -> wp.array[wp.vec2]:
    """
    Harmonic parametrization with fixed boundary.

    Minimizes the ``k``-harmonic energy built from the cotangent Laplacian
    [`cotmatrix`][ordito.laplacian.cotmatrix] subject to the boundary vertices being pinned to
    ``boundary_uv``. For ``k == 1`` this is the harmonic map (each interior UV is the
    cotangent-weighted average of its neighbors); ``k == 2`` is the biharmonic map, and so on. The
    interior system is solved with conjugate gradient.

    Parameters
    ----------
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

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``k < 1``, if there are interior vertices but ``boundary_indices`` is empty (the
        Dirichlet system would be singular), or if ``boundary_indices`` and ``boundary_uv`` have
        different lengths.
    RuntimeError
        If ``vertices``, ``faces``, ``boundary_indices`` and ``boundary_uv`` are not all on one
        device.

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
    Matches ``igl::harmonic``, ``k`` included.

    The interior system is solved iteratively (conjugate gradient) rather than by direct
    factorization. Squaring the operator squares its condition number, so each ``k`` gets its own
    preconditioner: the Jacobi-Chebyshev polynomial
    ([`chebyshev_preconditioner`][ordito.linalg.chebyshev_preconditioner]) at ``k == 1``, the
    square of that Laplacian's polynomial
    ([`squared_laplacian_preconditioner`][ordito.linalg.squared_laplacian_preconditioner]) at
    ``k == 2``, and an automatic multigrid preconditioner
    ([`multigrid_preconditioner`][ordito.linalg.multigrid_preconditioner]) above.
    """
    device, n_vertices = _validate_fixed_boundary_call(
        vertices, faces, boundary_indices, boundary_uv, k, "harmonic"
    )
    if n_vertices == 0:
        return _launch.empty(0, dtype=wp.vec2, device=device)
    laplacian = cotmatrix(vertices, faces, dtype=wp.float64)
    mass_diag = mass_matrix_entries(vertices, faces, dtype=wp.float64) if k > 1 else None
    return _solve_fixed_boundary(
        laplacian, mass_diag, k, n_vertices, boundary_indices, boundary_uv, device
    )


def tutte(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    k: int = 1,
) -> wp.array[wp.vec2]:
    """
    Tutte embedding with fixed boundary (uniform-Laplacian parametrization).

    Identical to [`harmonic`][ordito.parametrization.harmonic] except the operator is the
    combinatorial [`graph_laplacian`][ordito.laplacian.graph_laplacian] instead of the
    cotangent one — for ``k == 1`` that Laplacian is the *only* difference. Because the uniform
    Laplacian's free-free block is a diagonally dominant M-matrix, the Tutte embedding of a mesh
    with a convex boundary is guaranteed bijective (fold-free), unlike the harmonic/conformal maps.
    For
    ``k > 1`` the mass matrix is the identity (matching libigl's ``speye`` graph-Laplacian variant).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions (used for the count/device; the uniform weights
        ignore geometry).
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

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``k < 1``, if there are interior vertices but ``boundary_indices`` is empty, or if
        ``boundary_indices`` and ``boundary_uv`` have different lengths.
    RuntimeError
        If ``vertices``, ``faces``, ``boundary_indices`` and ``boundary_uv`` are not all on one
        device.

    See Also
    --------
    [`harmonic`][ordito.parametrization.harmonic]
    [`graph_laplacian`][ordito.laplacian.graph_laplacian]
    [`map_vertices_to_circle`][ordito.parametrization.map_vertices_to_circle]
    """
    device, n_vertices = _validate_fixed_boundary_call(
        vertices, faces, boundary_indices, boundary_uv, k, "tutte"
    )
    if n_vertices == 0:
        return _launch.empty(0, dtype=wp.vec2, device=device)
    laplacian = graph_laplacian(vertices, faces, dtype=wp.float64)
    return _solve_fixed_boundary(
        laplacian, None, k, n_vertices, boundary_indices, boundary_uv, device
    )


def _validate_fixed_boundary_call(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    k: int,
    name: str,
) -> tuple[wp.DeviceLike, int]:
    """Shared ``harmonic`` / ``tutte`` preamble: device check, ``k`` validation, vertex count."""
    require_same_device(
        vertices=vertices, faces=faces, boundary_indices=boundary_indices, boundary_uv=boundary_uv
    )
    if k < 1:
        raise ValueError(f"{name} power k must be >= 1, got {k}.")
    return vertices.device, vertices.size


def _solve_fixed_boundary(
    laplacian: odt.BsrMatrix[wp.float64],
    mass_diag: wp.array[wp.float64] | None,
    k: int,
    n_vertices: int,
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    device: wp.DeviceLike,
) -> wp.array[wp.vec2]:
    """
    Solve the fixed-boundary quadratic minimization shared by ``harmonic`` and ``tutte``.

    Forms the positive-semi-definite operator ``Q = -L`` for ``k == 1`` and
    ``Q = (-L) (M^-1 (-L))^(k-1)`` for ``k > 1`` (``M`` the diagonal mass, identity when
    ``mass_diag is None``) via [`k_harmonic`][ordito.energies.k_harmonic],
    then solves the interior Dirichlet system ``Q_uu x_u = -Q_ub bc`` per UV column with conjugate
    gradient, keeping the fixed vertices at ``boundary_uv``. ``laplacian`` must be float64:
    ``k > 1`` squares its condition number.
    """
    # A mesh with interior vertices and no fixed boundary is a singular Dirichlet system. Checked
    # before ``k_harmonic`` assembles the (for k > 1, sparse-matrix-product) operator, since once
    # every vertex is fixed (n_vertices > 0, n_boundary == 0 is impossible here because n_vertices
    # > 0 implies interior vertices exist) this cannot be satisfied and the assembly would be
    # wasted.
    n_boundary = boundary_indices.size
    _require_fixed_vertices(
        n_boundary,
        n_vertices,
        1,
        "harmonic / tutte require at least one fixed boundary vertex; the Dirichlet system is "
        "otherwise singular.",
    )

    q = od.energies.k_harmonic(laplacian, mass_diag, k=k)
    fixed_mask, fixed_values = _scatter_constraints(
        n_vertices, boundary_indices, boundary_uv, device
    )
    if k == 2:
        return _solve_biharmonic(laplacian, mass_diag, q, fixed_mask, fixed_values, n_vertices)

    # ``k >= 3`` raises the Laplacian's condition number to the ``k``-th power, which is what
    # makes a multigrid hierarchy worth building (``k == 2`` is the square of a Laplacian, handled
    # above). Routed through ``"auto"`` rather than forced, so the gate's own size floor can still
    # decline a system too small to repay the setup cost; the switch is on ``k`` rather than on the
    # gate alone because a plain (``k == 1``) Laplacian's rows nearly sum to zero and wants
    # iterations, not levels -- and with only the boundary pinned they are many, which is the long
    # solve the polynomial preconditioner is for.
    sol, free_map, _ = twl.min_quad_with_fixed(
        q,
        fixed_mask,
        odt.as_array2d(fixed_values, wp.float64),
        tol=_CG_TOLERANCE,
        preconditioner="auto" if k >= 2 else "chebyshev",
    )

    out_uv = _launch.empty(n_vertices, dtype=wp.vec2, device=device)
    _launch.launch(
        kernel_parametrization.scatter_solution,
        dim=n_vertices,
        inputs=[fixed_mask, free_map, sol, fixed_values, out_uv],
        device=device,
    )
    return out_uv


def _solve_biharmonic(
    laplacian: odt.BsrMatrix[wp.float64],
    mass_diag: wp.array[wp.float64] | None,
    q: odt.BsrMatrix[wp.float64],
    fixed_mask: wp.array[wp.bool],
    fixed_values: wp.array[wp.float64],
    n_vertices: int,
) -> wp.array[wp.vec2]:
    """
    ``_solve_fixed_boundary`` at ``k == 2``, preconditioned as the square of a Laplacian.

    ``Q = L M^-1 L`` is fourth order, and its free block is spectrally close to
    ``L_ff D^-2 L_ff`` with ``D = sqrt(M_f)`` -- the terms it drops are the one ring of the pinned
    boundary -- which is exactly the system
    [`squared_laplacian_preconditioner`][ordito.linalg.squared_laplacian_preconditioner] inverts
    with a fixed polynomial: no hierarchy to set up, where the smoothed-aggregation V-cycle
    ``"auto"`` would build costs a setup of its own and still runs a few hundred rounds.
    """
    device = fixed_mask.device
    fixed_values_2d = odt.as_array2d(fixed_values, wp.float64)
    free_map, n_free = twl.free_partition(fixed_mask)
    sol = odt.as_array2d(_launch.zeros((2, n_free), dtype=wp.float64, device=device), wp.float64)
    if n_free > 0:
        q_uu, rhs = twl.assemble_interior_system(q, fixed_mask, free_map, fixed_values_2d, n_free)
        no_values = odt.as_array2d(
            _launch.empty((0, n_vertices), dtype=wp.float64, device=device), wp.float64
        )
        # ``-L``'s free block, negated as it is extracted.
        l_ff, _ = twl.assemble_interior_system(
            laplacian, fixed_mask, free_map, no_values, n_free, scale=-1.0
        )
        if mass_diag is None:
            roots = _launch.full(n_free, 1.0, dtype=wp.float64, device=device)
        else:
            roots = _launch.empty(n_free, dtype=wp.float64, device=device)
            _launch.launch(
                kernel_parametrization.free_mass_roots,
                dim=n_vertices,
                inputs=[fixed_mask, free_map, mass_diag, roots],
                device=device,
            )
        twl.solve_spd_columns(
            q_uu,
            rhs,
            sol,
            tol=_CG_TOLERANCE,
            preconditioner=twl.squared_laplacian_preconditioner(l_ff, roots),
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
    # wp.empty; kernels index it as wp.array2d per CLAUDE.md).
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
    [`_solve_fixed_boundary`][ordito.parametrization._solve_fixed_boundary] and
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


def lscm(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    pinned_indices: wp.array[wp.int32],
    pinned_uv: wp.array[wp.vec2],
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

    The interior system is solved with conjugate gradient. Closed meshes are valid input: the
    boundary vector-area matrix is then zero and the
    Hessian reduces to ``-repdiag(L, 2)``.

    Parameters
    ----------
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

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``. Empty for an empty mesh.

    Raises
    ------
    ValueError
        If fewer than two vertices are pinned (and the mesh has at least two vertices), or if
        ``pinned_indices`` and ``pinned_uv`` have different lengths.
    RuntimeError
        If ``vertices``, ``faces``, ``pinned_indices`` and ``pinned_uv`` are not all on one device.

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
    require_same_device(
        vertices=vertices, faces=faces, pinned_indices=pinned_indices, pinned_uv=pinned_uv
    )
    device = vertices.device
    n = vertices.size
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
    if pinned_uv.size != n_pinned:
        # The scatter kernel below indexes ``pinned_uv`` at every position up to
        # ``pinned_indices.shape[0]``, so a shorter ``pinned_uv`` is an out-of-bounds read.
        raise ValueError(
            "pinned_indices and pinned_uv must have the same length, got "
            f"{n_pinned} and {pinned_uv.size}."
        )

    q = od.energies.lscm_hessian(vertices, faces)
    fixed_mask = _launch.zeros(2 * n, dtype=wp.bool, device=device)
    fixed_values = _launch.zeros((1, 2 * n), dtype=wp.float64, device=device)
    if n_pinned > 0:
        _launch.launch(
            kernel_parametrization.scatter_pinned_stacked,
            dim=n_pinned,
            inputs=[pinned_indices, pinned_uv, wp.int32(n), fixed_mask, fixed_values],
            device=device,
        )

    # Two pinned vertices leave a system as long to solve as it is large: the polynomial
    # preconditioner's case.
    sol, free_map, _ = twl.min_quad_with_fixed(
        q,
        fixed_mask,
        odt.as_array2d(fixed_values, wp.float64),
        tol=_CG_TOLERANCE,
        preconditioner="chebyshev",
    )

    out_uv = _launch.empty(n, dtype=wp.vec2, device=device)
    _launch.launch(
        kernel_parametrization.scatter_solution_stacked,
        dim=n,
        inputs=[fixed_mask, free_map, sol[0], fixed_values[0], out_uv],
        device=device,
    )
    return out_uv


def _require_fixed_vertices(n_fixed: int, n_vertices: int, min_required: int, message: str) -> None:
    """
    Raise ``ValueError(message)`` unless at least ``min_required`` vertices are fixed.

    Waived when the mesh has fewer than ``min_required`` vertices, since it is then impossible to
    pin that many distinct ones. The shared arithmetic behind three otherwise-differently-worded
    guards: harmonic / tutte's "at least one fixed boundary vertex" (``min_required=1``, via
    [`_solve_fixed_boundary`][ordito.parametrization._solve_fixed_boundary]), arap's "at least one
    fixed vertex" (``min_required=1``) and lscm's "at least two pinned vertices"
    (``min_required=2``). Each caller keeps its own message, since the *reason* the count is
    required differs (a singular Dirichlet system, translation invariance, a similarity-transform
    null space) even though the arithmetic is identical.
    """
    if n_vertices >= min_required and n_fixed < min_required:
        raise ValueError(message)
