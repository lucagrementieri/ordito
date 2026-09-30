"""
Energies over a mesh: the scalar regularizers, and the operators the solvers minimize.

The line between this module and [`triwarp.laplacian`][triwarp.laplacian] is *order*: laplacian
builds the first-order operators -- the cotangent stiffness matrix, its mass matrix and the
intrinsic repairs that keep them finite -- and everything here is assembled out of those.

Four families, and the first is the only one that returns a number rather than a matrix:

- **Scalar mesh regularizers.** [`edge_length_loss`][triwarp.energies.edge_length_loss],
  [`normal_consistency_loss`][triwarp.energies.normal_consistency_loss] and
  [`laplacian_smoothing_loss`][triwarp.energies.laplacian_smoothing_loss] are the priors a mesh
  *optimization* adds to a data term -- one per edge length, one per dihedral angle and one per
  vertex Laplacian residual, each reduced to a single ``float``. They are here rather than in
  [`triwarp.metrics`][triwarp.metrics], which hosts the data terms they pair with, because their
  machinery is this module's: ``laplacian_smoothing_loss``'s two cotangent variants consume
  [`cotmatrix`][triwarp.laplacian.cotmatrix] and
  [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries].
- **Smoothness energies over vertices.** [`k_harmonic`][triwarp.energies.k_harmonic] is the
  integrated ``k``-harmonic form -- ``k = 1`` is Dirichlet, ``k = 2`` the biharmonic operator behind
  smooth interpolation -- and it distorts a field near the boundary, because clamping a biharmonic
  solve there is not a natural condition. [`hessian_energy`][triwarp.energies.hessian_energy] and
  [`curved_hessian_energy`][triwarp.energies.curved_hessian_energy] are the alternatives that do
  not: both integrate a squared Hessian instead, so linear (respectively, locally linear) functions
  sit exactly in the null space, boundary or not.
- **The edge-based Crouzeix-Raviart pair.**
  [`crouzeix_raviart_cotmatrix`][triwarp.energies.crouzeix_raviart_cotmatrix] and
  [`crouzeix_raviart_massmatrix`][triwarp.energies.crouzeix_raviart_massmatrix] put the degrees of
  freedom on edge midpoints rather than vertices, which is the nonconforming-FEM discretization
  ``curved_hessian_energy`` is built on. They are the siblings of
  [`cotmatrix`][triwarp.laplacian.cotmatrix] and [`mass_matrix`][triwarp.laplacian.mass_matrix].
- **The LSCM operator.** [`lscm_hessian`][triwarp.energies.lscm_hessian] is the ``(2n, 2n)`` form
  behind the least-squares conformal map, and
  [`vector_area_matrix`][triwarp.energies.vector_area_matrix] is the boundary term that couples its
  two coordinate blocks. [`lscm`][triwarp.parametrization.lscm] is the solve; these are what it
  minimizes.

Every *operator* is assembled in ``float64`` in a single sparse build, because the
conjugate-gradient solves they feed run in ``float64`` for determinism and a ``float32``
intermediate would be the accuracy floor. The three scalar regularizers are ``float32`` throughout:
they are read by a human or by an optimizer's stopping rule, not solved with.
"""

from __future__ import annotations

from typing import Literal, cast

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar, require_same_device
from triwarp.edges import edges_unique, edges_unique_length
from triwarp.kernels import energies as kernel_energies
from triwarp.laplacian import cotmatrix, cotmatrix_entries, mass_matrix_entries


def edge_length_loss(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], target_length: float = 0.0
) -> float:
    """
    Mean squared deviation of the undirected edge lengths from a resting length.

    The edge regularizer of the deformation losses: ``mean((||e|| - L0)^2)`` over the **unique
    undirected** edges, so an interior edge counts once rather than twice. At the default
    ``target_length = 0.0`` it is the mean squared edge length, which is what a shrinking prior
    wants; a positive value pulls the mesh toward uniform edges of that size instead.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    target_length
        Resting edge length ``L0``.

    Returns
    -------
    float
        The mean, as a host scalar. ``0.0`` for a mesh with no edges.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`normal_consistency_loss`][triwarp.energies.normal_consistency_loss]
    [`laplacian_smoothing_loss`][triwarp.energies.laplacian_smoothing_loss]
    [`edges_unique_length`][triwarp.edges.edges_unique_length]
        The per-edge lengths this reduces, if the distribution rather than the mean is wanted.
    [`triwarp.metrics`][triwarp.metrics]
        The data terms these regularizers are added to, and the differentiable Chamfer family.

    Notes
    -----
    Matches ``pytorch3d.loss.mesh_edge_loss``. Its per-mesh ``1 / E`` weighting collapses to a
    plain mean for one mesh, which is triwarp's only case, so there is no batch weighting to port.
    """
    require_same_device(vertices=vertices, faces=faces)
    # The bound is supplied and the range check skipped: both are host readbacks that
    # serialise the pipeline, and this wrapper trusts its connectivity the same way its
    # own per-edge kernels below do.
    lengths = edges_unique_length(
        vertices, faces, n_vertices=int(vertices.shape[0]), validate=False
    )
    if int(lengths.shape[0]) == 0:
        return 0.0
    deviations = wp.empty(int(lengths.shape[0]), dtype=wp.float32, device=lengths.device)
    wp.map(kernel_energies.squared_deviation, lengths, wp.float32(target_length), out=deviations)
    return float(tw.reduce.mean(deviations))


def normal_consistency_loss(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> float:
    """
    Mean ``1 - cos(theta)`` over the pairs of faces sharing an edge.

    The dihedral regularizer of the deformation losses, and the one that penalizes a fold: it is
    ``0`` for a flat pair, ``1`` at a right angle and ``2`` for a face doubled back on itself. Read
    it against [`edge_length_loss`][triwarp.energies.edge_length_loss], which constrains the
    *sizes*; this one constrains the *orientations* and says nothing about scale.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    float
        The mean, as a host scalar. ``0.0`` for a mesh with no adjacent face pair.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`edge_length_loss`][triwarp.energies.edge_length_loss]
    [`laplacian_smoothing_loss`][triwarp.energies.laplacian_smoothing_loss]
    [`face_adjacency_angles`][triwarp.adjacency.face_adjacency_angles]
        The per-pair dihedral angles this reduces.

    Notes
    -----
    Matches ``pytorch3d.loss.mesh_normal_consistency`` on **edge-manifold** input. The restriction
    is real and is not a tolerance -- the reference enumerates *every* pair of faces sharing an
    edge, so an edge with ``k`` incident faces contributes ``C(k, 2)`` terms where
    [`face_adjacency_angles`][triwarp.adjacency.face_adjacency_angles] reports one pair per
    adjacency. The two coincide exactly wherever every edge has at most two faces.
    """
    require_same_device(vertices=vertices, faces=faces)
    angles = tw.adjacency.face_adjacency_angles(vertices, faces)
    if int(angles.shape[0]) == 0:
        return 0.0
    terms = wp.empty(int(angles.shape[0]), dtype=wp.float32, device=angles.device)
    wp.map(kernel_energies.one_minus_cosine, angles, out=terms)
    return float(tw.reduce.mean(terms))


def laplacian_smoothing_loss(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    method: Literal["uniform", "cot", "cotcurv"] = "uniform",
) -> float:
    """
    Mean magnitude of a Laplacian residual per vertex, under one of three normalizations.

    The smoothness regularizer of the deformation losses. The three methods are **three different
    quantities**, not one with a tuning knob, and they differ by roughly an order of magnitude:

    - ``"uniform"``: ``|| (A v)_i - v_i ||`` with ``A`` the row-normalized 1-ring average
      ([`laplacian`][triwarp.laplacian.laplacian] with ``equal_weight=True``) -- the umbrella
      residual, which is a *length* and therefore scales with the mesh.
    - ``"cot"``: the same residual against the **cotangent-weighted** neighbour average,
      ``|| (L v)_i / s_i ||`` with ``L`` the cotangent stiffness matrix and ``s_i`` its
      off-diagonal row sum. Also a length, and the geometry-aware version of the above.
    - ``"cotcurv"``: ``|| (L v)_i / (6 M_ii) ||`` with ``M`` the barycentric lumped mass -- the mean
      curvature magnitude, so it carries units of one over length and is the largest of the three
      on a unit-scale mesh.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    method
        Which normalization, as above.

    Returns
    -------
    float
        The mean, as a host scalar. ``0.0`` for an empty mesh.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``"uniform"``, ``"cot"`` or ``"cotcurv"``.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`edge_length_loss`][triwarp.energies.edge_length_loss]
    [`normal_consistency_loss`][triwarp.energies.normal_consistency_loss]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
        The operator the two cotangent variants are built from.
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
        The smoother that *minimizes* this, rather than measuring it.

    Notes
    -----
    Matches ``pytorch3d.loss.mesh_laplacian_smoothing`` on all three methods. Two conventions are
    inherited from it rather than chosen here, both because they are what makes the numbers
    comparable at all: the
    reference's ``cot`` and ``cotcurv`` read a cotangent Laplacian whose off-diagonal is **twice**
    triwarp's half-cotangent table and whose diagonal is identically zero, which cancels out of
    both ratios above -- and where a vertex's row sum is not positive its averaging is undefined,
    so ``"cot"`` falls back to ``|| v_i ||`` there, matching the reference's ``norm_w = 0`` branch.
    """
    require_same_device(vertices=vertices, faces=faces)
    if method not in ("uniform", "cot", "cotcurv"):
        raise ValueError(f'method must be "uniform", "cot" or "cotcurv", got {method!r}')
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0:
        return 0.0
    device = vertices.device
    # The three methods initialize the two scale buffers three different ways, so each branch
    # allocates them holding what it needs -- ``wp.full`` where the value is a constant,
    # ``wp.empty`` where a launch writes every element, ``wp.zeros`` where the value is zero.
    if method == "uniform":
        operator = tw.laplacian.laplacian(vertices, faces, equal_weight=True)
        row_scale = wp.full(n_vertices, 1.0, dtype=wp.float32, device=device)
        self_scale = wp.full(n_vertices, -1.0, dtype=wp.float32, device=device)
    else:
        operator = cotmatrix(vertices, faces)
        row_scale = wp.empty(n_vertices, dtype=wp.float32, device=device)
        if method == "cot":
            self_scale = wp.empty(n_vertices, dtype=wp.float32, device=device)
            wp.launch(
                kernel_energies.cot_row_scales,
                dim=n_vertices,
                inputs=[operator.offsets, operator.columns, operator.values, row_scale, self_scale],
                device=device,
            )
        else:
            wp.map(
                kernel_energies.reciprocal_scaled_or_zero,
                mass_matrix_entries(vertices, faces),
                wp.float32(1.0 / 6.0),
                out=row_scale,
            )
            self_scale = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    norms = wp.empty(n_vertices, dtype=wp.float32, device=device)
    wp.launch(
        kernel_energies.laplacian_residual_norms,
        dim=n_vertices,
        inputs=[
            operator.offsets,
            operator.columns,
            operator.values,
            vertices,
            row_scale,
            self_scale,
            norms,
        ],
        device=device,
    )
    return float(tw.reduce.mean(norms))


def k_harmonic(
    laplacian: wps.BsrMatrix[wp.float32], mass: twt.Array1dFloat | None = None, k: int = 2
) -> wps.BsrMatrix[wp.float32]:
    """
    Integrated k-harmonic operator ``Q = (-L) (M^-1 (-L))^(k-1)`` from a Laplacian and a mass.

    The quadratic form whose minimizers are k-harmonic functions: ``k == 1`` gives the Dirichlet
    energy ``-L`` (positive semi-definite for a [`cotmatrix`][triwarp.laplacian.cotmatrix]-sign
    Laplacian), ``k == 2`` the biharmonic operator ``L M^-1 L`` behind
    [`harmonic`][triwarp.parametrization.harmonic]'s smooth interpolation, and so on
    (``igl::harmonic_integrated_from_laplacian_and_mass``). Like igl's, the composition is not
    numerically robust for ``k > 2`` — the entries grow as the k-th power of the inverse mesh
    size — so high powers want a float64 ``laplacian``.

    **Named for the operator, not the map.** The unrelated
    [`harmonic`][triwarp.parametrization.harmonic] is a *map* into the plane rather than an
    operator; the two are related (``harmonic`` minimizes this form with the boundary pinned) but
    they are not interchangeable.

    Each power is assembled by one triplet pass over matching CSR rows —
    ``(A M^-1 B)_ij = sum_t A_ti M_t^-1 B_tj`` with both operands symmetric — followed by a single
    [`csr_from_triplets`][triwarp.array.csr_from_triplets], so the product is built without
    ``warp.sparse.bsr_mm``.

    Parameters
    ----------
    laplacian
        Square 1x1-block BSR Laplacian in igl's sign convention (negative diagonal, each row
        summing to zero), e.g. from [`cotmatrix`][triwarp.laplacian.cotmatrix] or
        [`graph_laplacian`][triwarp.laplacian.graph_laplacian].
    mass
        Length-``n_vertices`` lumped mass diagonal, e.g. from
        [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries]; cast to the Laplacian's
        scalar type if it differs. ``None`` (identity mass) composes plain powers of ``-L``, the
        [`tutte`][triwarp.parametrization.tutte] convention. Zero entries are treated as killed
        degrees of freedom (their rows contribute nothing), matching ``igl::invert_diag``.
    k
        Harmonic power (``>= 1``): 1 harmonic, 2 biharmonic, 3 triharmonic, ...

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_vertices, n_vertices)`` positive semi-definite operator in the Laplacian's
        scalar type on its device.

    Raises
    ------
    ValueError
        If ``k < 1``.
    RuntimeError
        If ``laplacian`` and ``mass`` are not on the same device.

    See Also
    --------
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries]
    [`hessian_energy`][triwarp.energies.hessian_energy]
    [`harmonic`][triwarp.parametrization.harmonic]
    """
    require_same_device(laplacian=laplacian, mass=mass)
    if k < 1:
        raise ValueError(f"harmonic power k must be >= 1, got {k}.")
    negated = cast("wps.BsrMatrix[wp.float32]", wps.bsr_axpy(x=laplacian, alpha=-1.0))
    if k == 1:
        return negated

    dtype = laplacian.values.dtype
    n_rows = int(laplacian.nrow)
    device = laplacian.values.device
    if mass is None:
        inverse_mass = wp.ones(n_rows, dtype=dtype, device=device)
    else:
        if mass.dtype != dtype:
            mass = tw.array.astype(mass, dtype)
        inverse_mass = wp.empty(n_rows, dtype=dtype, device=device)
        wp.map(kernel_energies.reciprocal_or_zero, mass, out=inverse_mass)

    operator = negated
    for _ in range(k - 1):
        operator = _diagonal_sandwich(operator, inverse_mass, negated)
    return operator


def _diagonal_sandwich(
    a: wps.BsrMatrix[wp.float32], inverse_mass: wp.array[wp.Float], b: wps.BsrMatrix[wp.float32]
) -> wps.BsrMatrix[wp.float32]:
    """
    Assemble ``A diag(inverse_mass) B`` for symmetric ``A``, ``B`` by one triplet pass.

    Row ``t`` of the product is the outer product of ``A``'s and ``B``'s rows ``t`` scaled by the
    diagonal weight, so the whole product is one count kernel, one scan, one emission kernel and a
    single ``array.csr_from_triplets``, built without ``warp.sparse.bsr_mm``. ``bsr_mm(bsr_mm(a,
    bsr_diag(inverse_mass)), b)`` is an equivalent construction, but this path avoids materializing
    an intermediate diagonal matrix and its scratch.
    """
    n_rows = int(a.nrow)
    device = inverse_mass.device
    counts = wp.empty(n_rows, dtype=wp.int32, device=device)
    wp.launch(
        kernel_energies.SANDWICH_ROW_COUNTS[inverse_mass.dtype],
        dim=n_rows,
        inputs=[a.offsets, b.offsets, inverse_mass, counts],
        device=device,
    )
    # Host readback: only the device knows the scan total, and it sizes the triplet buffers.
    segment_offsets, n_triplets = tw.array.counts_to_offsets(counts)

    dtype = a.values.dtype
    rows, cols, vals = tw.array.triplet_buffers(n_triplets, dtype, device)
    if n_triplets > 0:
        wp.launch(
            kernel_energies.SANDWICH_ROW_TRIPLETS[dtype],
            dim=n_rows,
            inputs=[
                a.offsets,
                a.columns,
                a.values,
                b.offsets,
                b.columns,
                b.values,
                inverse_mass,
                segment_offsets,
                rows,
                cols,
                vals,
            ],
            device=device,
        )
    return tw.array.csr_from_triplets(n_rows, n_rows, rows, cols, vals)


def hessian_energy(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    dtype: type = wp.float64,
    *,
    vertex_faces: tuple[wp.array[wp.int32], wp.array[wp.int32]] | None = None,
) -> wps.BsrMatrix[wp.float64]:
    """
    Hessian smoothness energy with natural boundary conditions.

    The mixed-FEM quadratic form ``Q = H^T M^-1 H`` of Stein et al. 2018, *Natural Boundary
    Conditions for Smoothing in Geometry Processing*: ``x' Q x`` integrates the squared Hessian of
    the piecewise-linear field ``x``, so minimizing it smooths **without** the boundary distortion
    the clamped biharmonic operator
    ([`k_harmonic`][triwarp.energies.k_harmonic] at ``k == 2``) produces —
    linear functions are exactly in its null space, boundary or not. ``M`` is the Voronoi lumped
    mass with boundary degrees of freedom killed, per the reference.

    Rather than materializing the sparse ``(9 n_faces, n_vertices)`` stacked Hessian ``H``, the
    product is contracted analytically over its nine component pairs and assembled in one triplet
    pass per vertex: ``Q_ij = sum_k M_k^-1 sum_{f,g ni k} A_f A_g (g_fk . g_gk)(g_fi . g_gj)``
    with ``g_fc`` corner ``c``'s hat-function gradient in face ``f``. The per-vertex triplet count
    is ``9 * valence^2``, so cost is quadratic in valence. A degenerate face contributes nothing
    (igl emits NaN there).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    dtype
        Scalar block type of the assembled matrix: ``wp.float64`` (default) or ``wp.float32``.
        Float64 is the default, unlike the first-order operators in this module, because the
        entries scale as the inverse fourth power of the mesh size and the operator exists to be
        solved against.
    vertex_faces
        Optional precomputed [`vertex_face_adjacency`][triwarp.adjacency.vertex_face_adjacency] as
        ``(vertex_faces, offsets)``. Depends on the connectivity alone, so one CSR serves every
        incidence walk over the same mesh --
        [`Trimesh.vertex_face_adjacency`][triwarp.mesh.Trimesh.vertex_face_adjacency] has it cached.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_vertices, n_vertices)`` positive semi-definite energy matrix on
        ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``vertex_faces`` are not all on one device.

    See Also
    --------
    [`curved_hessian_energy`][triwarp.energies.curved_hessian_energy]
    [`k_harmonic`][triwarp.energies.k_harmonic]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]

    Notes
    -----
    Matches ``igl::hessian_energy`` except on degenerate faces, which contribute nothing here and
    ``NaN`` there.
    """
    require_same_device(vertices=vertices, faces=faces, vertex_faces=vertex_faces)
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    device = vertices.device
    if n_faces == 0:
        return tw.array.empty_square_bsr(n_vertices, dtype, device)

    gradients = wp.empty(3 * n_faces, dtype=wp.vec3d, device=device)
    areas = wp.empty(n_faces, dtype=wp.float64, device=device)
    mass = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_energies.hessian_face_terms,
        dim=n_faces,
        inputs=[vertices, faces, gradients, areas, mass],
        device=device,
    )
    # The mass diagonal is defined only at interior vertices; the count kernel below inverts it.
    _zero_at_boundary(vertices, faces, mass)

    vf_indices, vf_offsets = (
        vertex_faces
        if vertex_faces is not None
        else tw.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
    )
    inverse_mass = wp.empty(n_vertices, dtype=wp.float64, device=device)
    counts = wp.empty(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_energies.hessian_energy_counts,
        dim=n_vertices,
        inputs=[vf_offsets, mass, inverse_mass, counts],
        device=device,
    )
    # Host readback: only the device knows the scan total, and it sizes the triplet buffers.
    segment_offsets, n_triplets = tw.array.counts_to_offsets(counts)

    rows, cols, vals = tw.array.triplet_buffers(n_triplets, dtype, device)
    if n_triplets > 0:
        wp.launch(
            kernel_energies.HESSIAN_ENERGY_TRIPLETS[dtype],
            dim=n_vertices,
            inputs=[
                faces,
                vf_offsets,
                vf_indices,
                gradients,
                areas,
                inverse_mass,
                segment_offsets,
                rows,
                cols,
                vals,
            ],
            device=device,
        )
    return tw.array.csr_from_triplets(n_vertices, n_vertices, rows, cols, vals)


def curved_hessian_energy(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], dtype: type = wp.float64
) -> wps.BsrMatrix[wp.float64]:
    """
    Curved Hessian smoothness energy on the Crouzeix-Raviart discretization.

    ``igl::curved_hessian_energy``, from Stein et al. 2020, *A Smoothness Energy without Boundary
    Distortion for Curved Surfaces*: where [`hessian_energy`][triwarp.energies.hessian_energy]
    treats the surface as locally flat, this one carries the Gaussian curvature into the operator
    through a per-vertex angle-defect correction, so the energy is intrinsic to the curved surface
    rather than to its triangles' planes. Constant functions are exactly in its null space.

    Assembled as ``Q = D^T M^-1 (L + K) M^-1 D`` over edge-based Crouzeix-Raviart vector elements
    — ``D`` the scalar-to-CR-vector gradient, ``M`` the CR vector mass, ``L`` the CR vector
    Laplacian and ``K`` the curvature correction — but contracted per face in one pass: ``L + K``
    couples edges within a face only, so each face emits its own 6x6 block sandwiched between its
    edges' gradient rows (a fixed 144 triplets per face), and no intermediate ``(2 n_edges, ...)``
    matrix or sparse product exists. Requires an edge-manifold mesh, like the igl original (which
    asserts it) — raises ``ValueError`` otherwise; a degenerate face contributes nothing.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    dtype
        Scalar block type of the assembled matrix: ``wp.float64`` (default) or ``wp.float32``,
        with float64 the default for the same conditioning reason as
        [`hessian_energy`][triwarp.energies.hessian_energy].

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_vertices, n_vertices)`` positive semi-definite energy matrix on
        ``vertices.device``.

    Raises
    ------
    ValueError
        If the mesh is not edge-manifold (some edge is shared by more than two faces).
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`hessian_energy`][triwarp.energies.hessian_energy]
    [`crouzeix_raviart_cotmatrix`][triwarp.energies.crouzeix_raviart_cotmatrix]
    [`vertex_defects`][triwarp.vertices.vertex_defects]
    """
    require_same_device(vertices=vertices, faces=faces)
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    device = vertices.device
    if n_faces == 0:
        return tw.array.empty_square_bsr(n_vertices, dtype, device)

    if not tw.validation.is_edge_manifold(faces, n_vertices=n_vertices, validate=False):
        raise ValueError(
            "mesh must be edge-manifold (every edge shared by at most two faces); the "
            "Crouzeix-Raviart discretization curved_hessian_energy is built on is undefined "
            "otherwise, like igl::curved_hessian_energy, which asserts it"
        )

    edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
    unique_edges, inverse = edges_unique(faces, edges_sorted, n_vertices=n_vertices, validate=False)
    n_edges = int(unique_edges.shape[0])

    angles = wp.empty((n_faces, 3), dtype=wp.float64, device=device)
    angle_sums = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_energies.internal_angles_and_sums,
        dim=n_faces,
        inputs=[vertices, faces, angles, angle_sums],
        device=device,
    )
    # ``angle_defect / angle_sum``, zeroed on the boundary (curvature is only corrected at interior
    # vertices) and for a non-positive angle sum -- igl::cr_vector_curvature_correction's kappa
    # scaling.
    scaled_kappa = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.map(kernel_energies.scaled_angle_defect, angle_sums, out=scaled_kappa)
    _zero_at_boundary(vertices, faces, scaled_kappa)

    mass = _cr_mass_diagonal(vertices, faces, inverse, n_edges, wp.float64)
    inverse_mass = wp.empty(n_edges, dtype=wp.float64, device=device)
    wp.map(kernel_energies.reciprocal_or_zero, mass, out=inverse_mass)

    edge_halfedges = wp.full((n_edges, 2), -1, dtype=wp.int32, device=device)
    cursor = wp.zeros(n_edges, dtype=wp.int32, device=device)
    wp.launch(
        kernel_energies.scatter_edge_halfedges,
        dim=3 * n_faces,
        inputs=[inverse, cursor, edge_halfedges],
        device=device,
    )
    vertex_slots = wp.full((n_edges, 4), -1, dtype=wp.int32, device=device)
    par = wp.zeros((n_edges, 4), dtype=wp.float64, device=device)
    perp = wp.zeros((n_edges, 4), dtype=wp.float64, device=device)
    wp.launch(
        kernel_energies.cr_gradient_rows,
        dim=n_edges,
        inputs=[vertices, faces, unique_edges, edge_halfedges, vertex_slots, par, perp],
        device=device,
    )

    n_triplets = 144 * n_faces
    rows, cols, vals = tw.array.triplet_buffers(n_triplets, dtype, device)
    wp.launch(
        kernel_energies.CURVED_HESSIAN_TRIPLETS[dtype],
        dim=n_faces,
        inputs=[
            vertices,
            faces,
            inverse,
            angles,
            scaled_kappa,
            inverse_mass,
            vertex_slots,
            par,
            perp,
            rows,
            cols,
            vals,
        ],
        device=device,
    )
    return tw.array.csr_from_triplets(n_vertices, n_vertices, rows, cols, vals)


def crouzeix_raviart_cotmatrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: twt.Array2dFloat | None = None,
    dtype: type = wp.float32,
    *,
    unique_edges: twt.Array2dInt32 | None = None,
    edge_map: wp.array[wp.int32] | None = None,
) -> wps.BsrMatrix[wp.float32]:
    """
    Edge-based Crouzeix-Raviart cotangent stiffness matrix.

    The nonconforming-FEM sibling of [`cotmatrix`][triwarp.laplacian.cotmatrix]: degrees of
    freedom live on edge midpoints, so the matrix is ``(n_edges, n_edges)`` and each face couples
    its three edges pairwise with minus four times the half-cotangent at their shared corner
    (positive diagonal — the igl sign convention for this operator, opposite to ``cotmatrix``'s).
    Rows follow [`edges_unique`][triwarp.edges.edges_unique]'s edge numbering. Requires an
    edge-manifold mesh, like the igl original (which asserts it) — raises ``ValueError``
    otherwise.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    cot_entries
        Optional precomputed ``(n_faces, 3)`` weights from
        [`cotmatrix_entries`][triwarp.laplacian.cotmatrix_entries]; computed in ``dtype`` when
        ``None``.
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``.
    unique_edges, edge_map
        Optional precomputed edge numbering from
        [`edges_unique`][triwarp.edges.edges_unique] — pass both or neither. Sharing it with
        [`crouzeix_raviart_massmatrix`][triwarp.energies.crouzeix_raviart_massmatrix] keeps the
        two operators on identical rows without recomputing the sort.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_edges, n_edges)`` stiffness matrix in 1x1-block BSR form on
        ``vertices.device``.

    Raises
    ------
    ValueError
        If exactly one of ``unique_edges`` / ``edge_map`` is provided, or if the mesh is not
        edge-manifold (some edge is shared by more than two faces).
    RuntimeError
        If ``vertices``, ``faces``, ``cot_entries``, ``unique_edges`` and ``edge_map`` are not all
        on one device.

    See Also
    --------
    [`crouzeix_raviart_massmatrix`][triwarp.energies.crouzeix_raviart_massmatrix]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`edges_unique`][triwarp.edges.edges_unique]

    Notes
    -----
    Matches ``igl::crouzeix_raviart_cotmatrix`` up to the edge numbering, which follows
    [`edges_unique`][triwarp.edges.edges_unique] rather than ``igl::unique_edge_map``.
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        cot_entries=cot_entries,
        unique_edges=unique_edges,
        edge_map=edge_map,
    )
    unique_edges, edge_map = _edge_numbering(vertices, faces, unique_edges, edge_map)
    n_edges = int(unique_edges.shape[0])
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return tw.array.empty_square_bsr(n_edges, dtype, device)

    if not tw.validation.is_edge_manifold(faces, n_vertices=int(vertices.shape[0]), validate=False):
        raise ValueError(
            "mesh must be edge-manifold (every edge shared by at most two faces); the "
            "Crouzeix-Raviart discretization is undefined otherwise, like "
            "igl::crouzeix_raviart_cotmatrix, which asserts it"
        )

    if cot_entries is None:
        cot_entries = cotmatrix_entries(vertices, faces, dtype=dtype)

    n_triplets = 12 * n_faces
    rows, cols, vals = tw.array.triplet_buffers(n_triplets, dtype, device)
    wp.launch(
        kernel_energies.CROUZEIX_RAVIART_COTMATRIX_TRIPLETS[cot_entries.dtype, dtype],
        dim=n_faces,
        inputs=[edge_map, cot_entries, rows, cols, vals],
        device=device,
    )
    return tw.array.csr_from_triplets(n_edges, n_edges, rows, cols, vals)


def crouzeix_raviart_massmatrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    dtype: type = wp.float32,
    *,
    unique_edges: twt.Array2dInt32 | None = None,
    edge_map: wp.array[wp.int32] | None = None,
) -> wps.BsrMatrix[wp.float32]:
    """
    Edge-based Crouzeix-Raviart mass matrix.

    Diagonal ``(n_edges, n_edges)``: each face donates a third of its area to each of its three
    edges, so an interior edge's entry is a third of its two incident faces' summed area. Rows
    follow [`edges_unique`][triwarp.edges.edges_unique]'s edge numbering, the same numbering
    [`crouzeix_raviart_cotmatrix`][triwarp.energies.crouzeix_raviart_cotmatrix] uses.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``.
    unique_edges, edge_map
        Optional precomputed edge numbering from
        [`edges_unique`][triwarp.edges.edges_unique] — pass both or neither.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_edges, n_edges)`` diagonal mass matrix in 1x1-block BSR form on
        ``vertices.device``.

    Raises
    ------
    ValueError
        If exactly one of ``unique_edges`` / ``edge_map`` is provided.
    RuntimeError
        If ``vertices``, ``faces``, ``unique_edges`` and ``edge_map`` are not all on one device.

    See Also
    --------
    [`crouzeix_raviart_cotmatrix`][triwarp.energies.crouzeix_raviart_cotmatrix]
    [`mass_matrix`][triwarp.laplacian.mass_matrix]
    [`edges_unique`][triwarp.edges.edges_unique]

    Notes
    -----
    Matches ``igl::crouzeix_raviart_massmatrix`` up to the edge numbering, as above.
    """
    require_same_device(
        vertices=vertices, faces=faces, unique_edges=unique_edges, edge_map=edge_map
    )
    unique_edges, edge_map = _edge_numbering(vertices, faces, unique_edges, edge_map)
    n_edges = int(unique_edges.shape[0])

    return cast(
        "wps.BsrMatrix[wp.float32]",
        wps.bsr_diag(diag=_cr_mass_diagonal(vertices, faces, edge_map, n_edges, dtype)),
    )


def _cr_mass_diagonal(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edge_map: wp.array[wp.int32],
    n_edges: int,
    dtype: type,
) -> wp.array[wp.Float]:
    """
    Lumped Crouzeix-Raviart mass, one entry per edge, as a dense diagonal.

    Each face gives a third of its area to each of its three edges. Shared so that
    [`crouzeix_raviart_massmatrix`][triwarp.energies.crouzeix_raviart_massmatrix] and
    [`curved_hessian_energy`][triwarp.energies.curved_hessian_energy] cannot drift onto different
    masses -- the latter's derivation assumes they are the same one.
    """
    n_faces = int(faces.shape[0]) // 3
    mass = wp.zeros(n_edges, dtype=dtype, device=faces.device)
    if n_faces > 0:
        wp.launch(
            kernel_energies.CROUZEIX_RAVIART_MASS_DIAG[dtype],
            dim=n_faces,
            inputs=[vertices, faces, edge_map, mass],
            device=faces.device,
        )
    return mass


def lscm_hessian(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wps.BsrMatrix[wp.float64]:
    """
    LSCM Hessian ``Q = -repdiag(L, 2) - 2 A``.

    Assembles the ``(2n, 2n)`` symmetric operator behind the least-squares conformal map, where
    ``L`` is the cotangent Laplacian [`cotmatrix`][triwarp.laplacian.cotmatrix] (negative-diagonal
    convention), ``repdiag(L, 2)`` is the block-diagonal ``[[L, 0], [0, L]]``, and ``A`` is the
    boundary [`vector_area_matrix`][triwarp.energies.vector_area_matrix]. Built natively in
    float64 in a single [`csr_from_triplets`][triwarp.array.csr_from_triplets] (the
    within-quadrant repdiag triplets and the
    cross-quadrant ``-2 A`` triplets never collide), so it feeds the float64 conjugate-gradient
    solve directly. Matches the ``Q`` returned by ``igl.lscm`` exactly.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(2n, 2n)`` float64 matrix in 1x1-block BSR form on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`lscm`][triwarp.parametrization.lscm]
    [`vector_area_matrix`][triwarp.energies.vector_area_matrix]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]

    Notes
    -----
    Matches ``igl::lscm_hessian``.
    """
    require_same_device(vertices=vertices, faces=faces)
    n = int(vertices.shape[0])
    device = vertices.device
    laplacian = cotmatrix(vertices, faces, dtype=wp.float64)
    # The real compressed-CSR entry count is offsets[-1], not laplacian.nnz: bsr_from_triplets
    # reports nnz as the (over-allocated) triplet capacity, so sizing by nnz would leave an
    # uninitialized gap in the wp.empty buffers that bsr_from_triplets reads back as garbage.
    n_entries = int(read_scalar(laplacian.offsets, n))
    boundary = tw.boundary.oriented_boundary_edges(vertices, faces)
    n_be = int(boundary.shape[0])

    # Combined triplet buffers: 2 per Laplacian entry (the two diagonal blocks) plus 4 per oriented
    # boundary edge (the vector-area cross-quadrant terms). Every slot is written, so wp.empty.
    total = 2 * n_entries + 4 * n_be
    rows, cols, vals = tw.array.triplet_buffers(total, wp.float64, device)
    wp.launch(
        kernel_energies.neg_repdiag2_triplets,
        dim=n,
        inputs=[
            laplacian.offsets,
            laplacian.columns,
            laplacian.values,
            wp.int32(n),
            rows,
            cols,
            vals,
        ],
        device=device,
    )
    if n_be > 0:
        # The ``-2 A`` term shares its triplet kernel with
        # [`vector_area_matrix`][triwarp.energies.vector_area_matrix] but writes into a slice
        # of the combined buffer: assembling ``A`` as its own matrix and adding it would need a
        # second build plus a ``bsr_axpy``, where this one pass over exact-size buffers does.
        _vector_area_triplets(
            boundary, n, -2.0, rows[2 * n_entries :], cols[2 * n_entries :], vals[2 * n_entries :]
        )
    return tw.array.csr_from_triplets(2 * n, 2 * n, rows, cols, vals)


def vector_area_matrix(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wps.BsrMatrix[wp.float64]:
    """
    Boundary vector-area matrix ``A``: the signed area enclosed by the UV boundary curve.

    Assembles the ``(2n, 2n)`` matrix that turns the ``[u; v]`` quadratic form into the signed area
    enclosed by the boundary UV curve: for each **oriented** boundary edge ``(i, j)`` (from the face
    winding, via [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges]) it adds the
    cross-quadrant entries ``(i+n, j, -1/4)``, ``(j, i+n, -1/4)``, ``(i, j+n, +1/4)``,
    ``(j+n, i, +1/4)``. On a closed mesh (no boundary) ``A`` is the zero matrix. Built natively in
    float64 in a single [`csr_from_triplets`][triwarp.array.csr_from_triplets].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions; only the count and device are used.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(2n, 2n)`` float64 matrix in 1x1-block BSR form on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`lscm_hessian`][triwarp.energies.lscm_hessian]
    [`lscm`][triwarp.parametrization.lscm]
    [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges]

    Notes
    -----
    Matches ``igl::vector_area_matrix``.
    """
    require_same_device(vertices=vertices, faces=faces)
    n = int(vertices.shape[0])
    device = vertices.device
    boundary = tw.boundary.oriented_boundary_edges(vertices, faces)
    n_be = int(boundary.shape[0])
    if n_be == 0:
        return tw.array.empty_square_bsr(2 * n, wp.float64, device)

    rows, cols, vals = tw.array.triplet_buffers(4 * n_be, wp.float64, device)
    _vector_area_triplets(boundary, n, 1.0, rows, cols, vals)
    return tw.array.csr_from_triplets(2 * n, 2 * n, rows, cols, vals)


def _vector_area_triplets(
    boundary_edges: twt.Array2dInt32,
    n_vertices: int,
    scale: float,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
) -> None:
    """
    Emit the four cross-quadrant vector-area triplets per oriented boundary edge.

    Writes ``4 * n_boundary_edges`` triplets from slot zero of the given buffers, which may be
    slices of a larger triplet array. ``scale = 1`` builds ``A`` itself
    ([`vector_area_matrix`][triwarp.energies.vector_area_matrix]); ``scale = -2`` builds the
    ``-2 A`` term of the LSCM Hessian ([`lscm_hessian`][triwarp.energies.lscm_hessian]).
    """
    wp.launch(
        kernel_energies.vector_area_triplets,
        dim=int(boundary_edges.shape[0]),
        inputs=[
            boundary_edges,
            wp.int32(n_vertices),
            wp.float64(scale),
            out_rows,
            out_cols,
            out_vals,
        ],
        device=out_rows.device,
    )


def _edge_numbering(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: twt.Array2dInt32 | None,
    edge_map: wp.array[wp.int32] | None,
) -> tuple[twt.Array2dInt32, wp.array[wp.int32]]:
    """Validate or build the shared ``edges_unique`` numbering the edge-based operators index."""
    if (unique_edges is None) != (edge_map is None):
        raise ValueError("pass unique_edges and edge_map together, or neither.")
    if unique_edges is None or edge_map is None:
        unique_edges, edge_map = edges_unique(
            faces, n_vertices=int(vertices.shape[0]), validate=False
        )
    return unique_edges, edge_map


def _zero_at_boundary(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], values: wp.array[wp.float64]
) -> None:
    """
    Zero ``values`` in place at every boundary vertex.

    Both quantities these energies build on -- a mass diagonal and an angle defect -- are defined
    only at interior vertices, so each is zeroed on the boundary before being inverted or scaled.
    The guard is required rather than defensive: a closed mesh has no boundary vertices, and
    launching over an empty index buffer is what the check avoids.
    """
    boundary = tw.boundary.boundary_vertex_indices(vertices, faces)
    n_boundary = int(boundary.shape[0])
    if n_boundary > 0:
        wp.launch(
            kernel_energies.ZERO_AT_INDICES[values.dtype],
            dim=n_boundary,
            inputs=[boundary, values],
            device=values.device,
        )
