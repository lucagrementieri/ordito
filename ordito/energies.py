"""
Energies over a mesh: the scalar regularizers, and the operators the solvers minimize.

The line between this module and [`ordito.laplacian`][ordito.laplacian] is *order*: laplacian
builds the first-order operators -- the cotangent stiffness matrix, its mass matrix and the
intrinsic repairs that keep them finite -- and everything here is assembled out of those.

Four families, and the first is the only one that returns a number rather than a matrix:

- **Scalar mesh regularizers.** [`edge_length_loss`][ordito.energies.edge_length_loss],
  [`normal_consistency_loss`][ordito.energies.normal_consistency_loss] and
  [`laplacian_smoothing_loss`][ordito.energies.laplacian_smoothing_loss] are the priors a mesh
  *optimization* adds to a data term -- one per edge length, one per dihedral angle and one per
  vertex Laplacian residual, each reduced to a single ``float``. They are here rather than in
  [`ordito.metrics`][ordito.metrics], which hosts the data terms they pair with, because their
  machinery is this module's: ``laplacian_smoothing_loss``'s two cotangent variants consume
  [`cotmatrix`][ordito.laplacian.cotmatrix] and
  [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries].
- **Smoothness energies over vertices.** [`k_harmonic`][ordito.energies.k_harmonic] is the
  integrated ``k``-harmonic form -- ``k = 1`` is Dirichlet, ``k = 2`` the biharmonic operator behind
  smooth interpolation -- and it distorts a field near the boundary, because clamping a biharmonic
  solve there is not a natural condition. [`hessian_energy`][ordito.energies.hessian_energy] and
  [`curved_hessian_energy`][ordito.energies.curved_hessian_energy] are the alternatives that do
  not: both integrate a squared Hessian instead, so linear (respectively, locally linear) functions
  sit exactly in the null space, boundary or not.
- **The edge-based Crouzeix-Raviart pair.**
  [`crouzeix_raviart_cotmatrix`][ordito.energies.crouzeix_raviart_cotmatrix] and
  [`crouzeix_raviart_massmatrix`][ordito.energies.crouzeix_raviart_massmatrix] put the degrees of
  freedom on edge midpoints rather than vertices, which is the nonconforming-FEM discretization
  ``curved_hessian_energy`` is built on. They are the siblings of
  [`cotmatrix`][ordito.laplacian.cotmatrix] and [`mass_matrix`][ordito.laplacian.mass_matrix].
- **The LSCM operator.** [`lscm_hessian`][ordito.energies.lscm_hessian] is the ``(2n, 2n)`` form
  behind the least-squares conformal map, and
  [`vector_area_matrix`][ordito.energies.vector_area_matrix] is the boundary term that couples its
  two coordinate blocks. [`lscm`][ordito.parametrization.lscm] is the solve; these are what it
  minimizes.

Every *operator* is assembled in ``float64`` in a single sparse build, because the
conjugate-gradient solves they feed run in ``float64`` for determinism and a ``float32``
intermediate would be the accuracy floor. The three scalar regularizers are ``float32`` throughout:
they are read by a human or by an optimizer's stopping rule, not solved with.
"""

from __future__ import annotations

from typing import Literal, cast

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, require_same_device
from ordito.config import is_deterministic
from ordito.edges import edges_unique, edges_unique_length
from ordito.kernels import energies as kernel_energies
from ordito.kernels import laplacian as kernel_laplacian
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import scatter as kernel_scatter
from ordito.laplacian import MeshOperatorPattern, cotmatrix_entries, mesh_operator_pattern


def edge_length_loss(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    target_length: float = 0.0,
    *,
    unique_edges: odt.Array2dInt32 | None = None,
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
        ``(3 * n_faces,)`` triangle index buffer.
    target_length
        Resting edge length ``L0``.
    unique_edges
        ``(n_edges, 2)`` [`edges_unique`][ordito.edges.edges_unique] of ``faces``, which depend on
        the connectivity alone: a loop moving the vertices over fixed faces builds them once
        ([`Trimesh.edges_unique`][ordito.mesh.Trimesh.edges_unique]) and each call then skips the
        edge sort. Trusted, not checked.

    Returns
    -------
    float
        The mean, as a host scalar. ``0.0`` for a mesh with no edges.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``unique_edges`` are not all on one device.

    See Also
    --------
    [`normal_consistency_loss`][ordito.energies.normal_consistency_loss]
    [`laplacian_smoothing_loss`][ordito.energies.laplacian_smoothing_loss]
    [`edges_unique_length`][ordito.edges.edges_unique_length]
        The per-edge lengths this reduces, if the distribution rather than the mean is wanted.
    [`ordito.metrics`][ordito.metrics]
        The data terms these regularizers are added to, and the differentiable Chamfer family.

    Notes
    -----
    Matches ``pytorch3d.loss.mesh_edge_loss``. Its per-mesh ``1 / E`` weighting collapses to a
    plain mean for one mesh, which is ordito's only case, so there is no batch weighting to port.

    Examples
    --------
    A unit cube has 12 unit edges and 6 face diagonals of length ``sqrt(2)``, so its mean squared
    edge length is ``(12 + 6 * 2) / 18``:

    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    print(f"{od.energies.edge_length_loss(vertices, faces):.4f}")  # 1.3333
    print(f"{od.energies.edge_length_loss(vertices, faces, target_length=1.0):.4f}")  # 0.0572
    ```
    """
    require_same_device(vertices=vertices, faces=faces, unique_edges=unique_edges)
    # The bound is supplied and the range check skipped: both are host readbacks that
    # serialise the pipeline, and this wrapper trusts its connectivity the same way its
    # own per-edge kernels below do.
    lengths = edges_unique_length(
        vertices, faces, unique_edges, n_vertices=vertices.size, validate=False
    )
    if lengths.size == 0:
        return 0.0
    terms = kernel_energies.EdgeDeviationTerms()
    terms.n = lengths.size
    terms.lengths = lengths
    terms.target = float(target_length)
    return _mean_of_terms(kernel_energies.EDGE_DEVIATION_SUM, terms, lengths.size, lengths.device)


def normal_consistency_loss(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    sorted_edge_keys: tuple[wp.array[wp.uint64], wp.array[wp.int32]] | None = None,
) -> float:
    """
    Mean ``1 - cos(theta)`` over the pairs of faces sharing an edge.

    The dihedral regularizer of the deformation losses, and the one that penalizes a fold: it is
    ``0`` for a flat pair, ``1`` at a right angle and ``2`` for a face doubled back on itself. Read
    it against [`edge_length_loss`][ordito.energies.edge_length_loss], which constrains the
    *sizes*; this one constrains the *orientations* and says nothing about scale.

    Every pair of faces on an edge counts, ``k * (k - 1) / 2`` pairs at an edge with ``k`` faces,
    and each pair is oriented by the edge rather than by the faces' windings: both normals are
    formed as ``(v1 - v0) x (apex - v0)`` against the one edge ``(v0, v1)``, and ``theta`` is the
    angle between one and the negation of the other. A pair wound against each other therefore
    reads as smooth as a consistently wound one, so the loss is the same on an inconsistently
    oriented or non-orientable mesh as on its oriented counterpart, and a non-manifold fan
    contributes every pair it holds.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    sorted_edge_keys
        [`sorted_face_edge_keys`][ordito.adjacency.sorted_face_edge_keys] of ``faces``, which
        depend on the connectivity alone: a loop moving the vertices over fixed faces builds them
        once and each call then skips the sort that groups the faces by edge. Trusted, not
        checked.

    Returns
    -------
    float
        The mean, as a host scalar. ``0.0`` for a mesh with no pair of faces sharing an edge.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``sorted_edge_keys`` are not all on one device.

    See Also
    --------
    [`edge_length_loss`][ordito.energies.edge_length_loss]
    [`laplacian_smoothing_loss`][ordito.energies.laplacian_smoothing_loss]
    [`face_adjacency_angles`][ordito.adjacency.face_adjacency_angles]
        Per-pair dihedral angles of the manifold edges, oriented by the faces' windings.

    Notes
    -----
    Matches ``pytorch3d.loss.mesh_normal_consistency`` on every input, non-manifold fans and
    inconsistent windings included. Its per-mesh weighting collapses to a plain mean over the pairs
    for one mesh, which is ordito's only case.

    Examples
    --------
    A cube's 12 edges meet at right angles (``1`` each) and its 6 face diagonals are flat (``0``);
    a flat grid has no fold at all:

    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    print(f"{od.energies.normal_consistency_loss(vertices, faces):.4f}")  # 0.6667
    vertices, faces = od.creation.grid(count=(4, 4))
    print(f"{od.energies.normal_consistency_loss(vertices, faces):.4f}")  # 0.0000
    ```
    """
    require_same_device(vertices=vertices, faces=faces, sorted_edge_keys=sorted_edge_keys)
    n = faces.size // 3 * 3
    if n == 0:
        return 0.0
    device = faces.device
    if sorted_edge_keys is None:
        sorted_edge_keys = od.adjacency.sorted_face_edge_keys(
            faces, n_vertices=max(vertices.size, 1)
        )
    terms = kernel_energies.NormalConsistencyTerms()
    terms.n = n
    terms.keys, terms.order = sorted_edge_keys
    terms.faces = faces
    terms.vertices = vertices
    # One pass sums the pair terms and counts the pairs, as a ``wp.vec2``, read back because the
    # mean is a host scalar.
    total, pairs = read_scalar(
        od.reduce.sum_of_terms(kernel_energies.NORMAL_CONSISTENCY_SUM, [terms], n, device=device), 0
    )
    if pairs == 0.0:
        return 0.0
    return float(total) / float(pairs)


def laplacian_smoothing_loss(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    method: Literal["uniform", "cot", "cotcurv"] = "uniform",
    *,
    pattern: MeshOperatorPattern | None = None,
) -> float:
    """
    Mean magnitude of a Laplacian residual per vertex, under one of three normalizations.

    The smoothness regularizer of the deformation losses. The three methods are **three different
    quantities**, not one with a tuning knob, and they differ by roughly an order of magnitude:

    - ``"uniform"``: ``|| (A v)_i - v_i ||`` with ``A`` the row-normalized 1-ring average
      ([`laplacian`][ordito.laplacian.laplacian] with ``equal_weight=True``) -- the umbrella
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
        ``(3 * n_faces,)`` triangle index buffer.
    method
        Which normalization, as above.
    pattern
        A [`mesh_operator_pattern`][ordito.laplacian.mesh_operator_pattern] of ``faces``, which
        depends on the connectivity alone: a loop moving the vertices over fixed faces builds it
        once and each call then skips the operator's sort. Built for ``"laplacian_directed"`` with
        ``method="uniform"`` and for ``"cotmatrix"``
        ([`Trimesh.mesh_operator_pattern`][ordito.mesh.Trimesh.mesh_operator_pattern]) otherwise.

    Returns
    -------
    float
        The mean, as a host scalar. ``0.0`` for an empty mesh.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``"uniform"``, ``"cot"`` or ``"cotcurv"``, or if ``pattern`` was
        built for another operator or vertex count.
    RuntimeError
        If ``vertices``, ``faces`` and ``pattern`` are not all on one device.

    See Also
    --------
    [`edge_length_loss`][ordito.energies.edge_length_loss]
    [`normal_consistency_loss`][ordito.energies.normal_consistency_loss]
    [`cotmatrix`][ordito.laplacian.cotmatrix]
        The operator the two cotangent variants are built from.
    [`filter_laplacian`][ordito.smoothing.filter_laplacian]
        The smoother that *minimizes* this, rather than measuring it.

    Notes
    -----
    Matches ``pytorch3d.loss.mesh_laplacian_smoothing`` on all three methods. Two conventions are
    inherited from it rather than chosen here, both because they are what makes the numbers
    comparable at all: the
    reference's ``cot`` and ``cotcurv`` read a cotangent Laplacian whose off-diagonal is **twice**
    ordito's half-cotangent table and whose diagonal is identically zero, which cancels out of
    both ratios above -- and where a vertex's row sum is not positive its averaging is undefined,
    so ``"cot"`` falls back to ``|| v_i ||`` there, matching the reference's ``norm_w = 0`` branch.

    Examples
    --------
    ``"cotcurv"`` is a curvature, so doubling a sphere's radius halves it, where the ``"uniform"``
    residual, a length, doubles:

    ```python
    import ordito as od

    loss = od.energies.laplacian_smoothing_loss
    unit_vertices, unit_faces = od.creation.icosphere(subdivisions=3)
    big_vertices, big_faces = od.creation.icosphere(subdivisions=3, radius=2.0)
    unit, big = loss(unit_vertices, unit_faces, "cotcurv"), loss(big_vertices, big_faces, "cotcurv")
    print(f"{unit:.4f} {big:.4f}")  # 0.3335 0.1668
    unit, big = loss(unit_vertices, unit_faces), loss(big_vertices, big_faces)
    print(f"{unit:.4f} {big:.4f}")  # 0.0139 0.0279
    ```
    """
    require_same_device(
        vertices=vertices, faces=faces, pattern=None if pattern is None else pattern.offsets
    )
    if method not in ("uniform", "cot", "cotcurv"):
        raise ValueError(f'method must be "uniform", "cot" or "cotcurv", got {method!r}')
    n_vertices = vertices.size
    operator = "laplacian_directed" if method == "uniform" else "cotmatrix"
    if pattern is not None and (pattern.operator != operator or pattern.n_vertices != n_vertices):
        raise ValueError(
            f"pattern was built for operator={pattern.operator!r} over {pattern.n_vertices} "
            f"vertices; method={method!r} needs operator={operator!r} over {n_vertices}"
        )
    if n_vertices == 0 or faces.size == 0:
        return 0.0
    device = vertices.device
    if method == "uniform":
        laplacian = od.laplacian.laplacian(vertices, faces, equal_weight=True, pattern=pattern)
        # Each method's per-row scales are formed in the row's own thread, so no scale buffer
        # exists, and the umbrella norms are summed where they are formed, as ``reduce.mean`` sums
        # them, so no norm buffer exists either.
        terms = kernel_energies.SmoothingNormTerms()
        terms.n = n_vertices
        terms.offsets = laplacian.offsets
        terms.columns = laplacian.columns
        terms.values = laplacian.values
        terms.vertices = vertices
        terms.method = kernel_energies.SMOOTHING_UNIFORM
        return _mean_of_terms(kernel_energies.SMOOTHING_NORM_SUM, terms, n_vertices, device)
    # The cotangent Laplacian's row is written by the thread that takes its residual, over its
    # pattern; each method's per-row scales are formed in the same thread. The norms, the row
    # values and (``"cotcurv"``) the half-cotangents are one ``float32`` allocation.
    if pattern is None:
        pattern = mesh_operator_pattern(faces, n_vertices)
    _, _, offsets, columns, run_start, keys, count, order = pattern
    n_faces = faces.size // 3
    cotcurv = method == "cotcurv"
    norms, values, *cot_flat = _launch.carve(
        wp.float32, (n_vertices, columns.size, *((3 * n_faces,) if cotcurv else ())), device
    )
    area_sums = None
    if method == "cot":
        cot_entries = cotmatrix_entries(vertices, faces)
        selector = kernel_energies.SMOOTHING_COT
    else:
        # ``mass_matrix_entries``' exact area sums, formed in the face pass that writes the
        # half-cotangents; the row kernel divides them by three.
        cot_entries = odt.as_array2d(cot_flat[0].reshape((n_faces, 3)), wp.float32)
        area_sums = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
        _launch.launch(
            kernel_laplacian.cotmatrix_entries_and_area_sums,
            dim=n_faces,
            inputs=[vertices, faces, wp.int32(1)],
            outputs=[cot_entries, area_sums],
            device=device,
        )
        selector = kernel_energies.SMOOTHING_COTCURV
    _launch.launch(
        kernel_energies.laplacian_smoothing_cot_norms,
        dim=n_vertices,
        inputs=[
            offsets,
            columns,
            run_start,
            keys,
            wp.int32(count),
            order,
            cot_entries,
            vertices,
            area_sums,
            selector,
        ],
        outputs=[values, norms],
        device=device,
    )
    return float(od.reduce.mean(norms))


def _mean_of_terms(
    kernels: kernel_reduce.TermSum, terms: object, n: int, device: wp.DeviceLike
) -> float:
    """
    Mean of the ``n`` per-item terms a ``kernels/reduce.term_sum_1d`` pair forms over ``terms``.

    [`sum_of_terms`][ordito.reduce.sum_of_terms] sums them as [`reduce.mean`][ordito.reduce.mean]
    sums an array of them, in either determinism setting, so the mean is the one the regularizers
    would get from materializing their terms and reducing them, with no term buffer.
    """
    total = od.reduce.sum_of_terms(kernels, [terms], n, device=device)
    return float(read_scalar(total, 0)) / n


def k_harmonic(
    laplacian: odt.BsrMatrix[wp.Float], mass: odt.ArrayNdFloat | None = None, k: int = 2
) -> odt.BsrMatrix[wp.Float]:
    """
    Integrated k-harmonic operator ``Q = (-L) (M^-1 (-L))^(k-1)`` from a Laplacian and a mass.

    The quadratic form whose minimizers are k-harmonic functions: ``k == 1`` gives the Dirichlet
    energy ``-L`` (positive semi-definite for a [`cotmatrix`][ordito.laplacian.cotmatrix]-sign
    Laplacian), ``k == 2`` the biharmonic operator ``L M^-1 L`` behind
    [`harmonic`][ordito.parametrization.harmonic]'s smooth interpolation, and so on
    (``igl::harmonic_integrated_from_laplacian_and_mass``). Like igl's, the composition is not
    numerically robust for ``k > 2`` — the entries grow as the k-th power of the inverse mesh
    size — so high powers want a float64 ``laplacian``.

    **Named for the operator, not the map.** The unrelated
    [`harmonic`][ordito.parametrization.harmonic] is a *map* into the plane rather than an
    operator; the two are related (``harmonic`` minimizes this form with the boundary pinned) but
    they are not interchangeable.

    Each power is assembled by one triplet pass over matching CSR rows —
    ``(A M^-1 B)_ij = sum_t A_ti M_t^-1 B_tj`` with both operands symmetric — followed by a single
    [`csr_from_coo_keys`][ordito.array.csr_from_coo_keys], so the product is built without
    ``warp.sparse.bsr_mm``.

    Parameters
    ----------
    laplacian
        ``(n_vertices, n_vertices)`` 1x1-block BSR Laplacian in igl's sign convention (negative
        diagonal, each row summing to zero), e.g. from [`cotmatrix`][ordito.laplacian.cotmatrix] or
        [`graph_laplacian`][ordito.laplacian.graph_laplacian].
    mass
        ``(n_vertices,)`` lumped mass diagonal, e.g. from
        [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries]; cast to the Laplacian's
        scalar type if it differs. ``None`` (identity mass) composes plain powers of ``-L``, the
        [`tutte`][ordito.parametrization.tutte] convention. Zero entries are treated as killed
        degrees of freedom (their rows contribute nothing), matching ``igl::invert_diag``.
    k
        Harmonic power (``>= 1``): 1 harmonic, 2 biharmonic, 3 triharmonic, ...

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` positive semi-definite operator in the Laplacian's
        scalar type on its device.

    Raises
    ------
    ValueError
        If ``k < 1``.
    RuntimeError
        If ``laplacian`` and ``mass`` are not on the same device.

    See Also
    --------
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries]
    [`hessian_energy`][ordito.energies.hessian_energy]
    [`harmonic`][ordito.parametrization.harmonic]

    Examples
    --------
    ``k == 1`` is ``-L`` itself; the biharmonic ``k == 2`` keeps constants in its kernel:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    L = od.laplacian.cotmatrix(vertices, faces, dtype=wp.float64)
    mass = od.laplacian.mass_matrix_entries(vertices, faces, dtype=wp.float64)
    x = wp.array(vertices.numpy()[:, 0], dtype=wp.float64)
    dirichlet = od.energies.k_harmonic(L, k=1)
    print(np.allclose((dirichlet @ x).numpy(), -(L @ x).numpy()))  # True
    biharmonic = od.energies.k_harmonic(L, mass, k=2)
    print(biharmonic.shape)  # (162, 162)
    ones = wp.full(biharmonic.shape[0], 1.0, dtype=wp.float64)
    print(np.abs((biharmonic @ ones).numpy()).max() < 1e-8)  # True
    ```
    """
    require_same_device(laplacian=laplacian, mass=mass)
    if k < 1:
        raise ValueError(f"harmonic power k must be >= 1, got {k}.")
    if k == 1:
        # ``-L`` on ``L``'s own topology: copied offsets and columns and one negating map over the
        # values, rather than ``bsr_axpy``'s general sum, which also reads the stale ``nnz`` field.
        negated_values = _launch.empty_like(laplacian.values)
        _launch.map(wp.neg, laplacian.values, out=negated_values)
        return od.array.bsr_from_csr(
            int(laplacian.nrow),
            int(laplacian.ncol),
            _launch.clone(laplacian.offsets),
            _launch.clone(laplacian.columns),
            negated_values,
        )

    dtype = laplacian.values.dtype
    n_rows = int(laplacian.nrow)
    device = laplacian.values.device
    if mass is None:
        inverse_mass = _launch.ones(n_rows, dtype=dtype, device=device)
    else:
        if mass.dtype != dtype:
            mass = od.array.astype(mass, dtype)
        # Formed from ``mass`` by the first sandwich's count pass.
        inverse_mass = _launch.empty(n_rows, dtype=dtype, device=device)

    # Every factor of ``-L`` is read as ``L`` with its sign carried into the product: the first
    # sandwich ``(-L) M^-1 (-L)`` is ``L M^-1 L`` (``sign = 1``), and each later one multiplies an
    # already signed product by ``-L`` (``sign = -1``). Negation is exact, so the entries are the
    # ones a negated copy of ``L`` would give, without that copy.
    operator = _diagonal_sandwich(laplacian, inverse_mass, laplacian, 1.0, mass=mass)
    for _ in range(k - 2):
        operator = _diagonal_sandwich(operator, inverse_mass, laplacian, -1.0)
    return operator


def _diagonal_sandwich(
    a: odt.BsrMatrix[wp.Float],
    inverse_mass: wp.array[wp.Float],
    b: odt.BsrMatrix[wp.Float],
    sign: float,
    *,
    mass: odt.ArrayNdFloat | None = None,
) -> odt.BsrMatrix[wp.Float]:
    """
    Assemble ``sign * A diag(inverse_mass) B`` for symmetric ``A``, ``B`` by one triplet pass.

    Given ``mass``, ``inverse_mass`` is written here, as ``mass``'s ``reciprocal_or_zero``, by
    the count pass that first reads it.

    Row ``t`` of the product is the outer product of ``A``'s and ``B``'s rows ``t`` scaled by the
    diagonal weight, so the whole product is one count kernel, one scan, one emission kernel and a
    single ``array.csr_from_coo_keys``, built without ``warp.sparse.bsr_mm``. ``bsr_mm(bsr_mm(a,
    bsr_diag(inverse_mass)), b)`` is an equivalent construction, but this path avoids materializing
    an intermediate diagonal matrix and its scratch.
    """
    n_rows = int(a.nrow)
    device = inverse_mass.device
    # Each row's count lands in the offsets' buffer past its leading zero and is scanned in place.
    segment_offsets = _launch.zeros(n_rows + 1, dtype=wp.int32, device=device)
    n_triplets = 0
    if n_rows > 0:
        counts = _launch.view(segment_offsets, 1, n_rows + 1)
        _launch.launch(
            kernel_energies.SANDWICH_ROW_COUNTS[inverse_mass.dtype],
            dim=n_rows,
            inputs=[
                a.offsets,
                b.offsets,
                inverse_mass if mass is None else mass,
                wp.int32(mass is not None),
            ],
            outputs=[counts, None if mass is None else inverse_mass],
            device=device,
        )
        _launch.array_scan(counts, out_array=counts, inclusive=True)
        # Host readback: only the device knows the scan total, and it sizes the triplet buffers.
        n_triplets = int(read_scalar(segment_offsets))

    dtype = a.values.dtype
    keys, order = od.array.csr_key_buffers(n_triplets, device)
    vals = _launch.empty(n_triplets, dtype=dtype, device=device)
    if n_triplets > 0:
        _launch.launch(
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
                dtype(sign),
                segment_offsets,
                wp.int32(n_rows),
            ],
            outputs=[keys, order, vals],
            device=device,
        )
    return od.array.csr_from_coo_keys(n_rows, n_rows, keys, order, vals)


def hessian_energy(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    dtype: type[odt.Block] = wp.float64,
    *,
    vertex_faces: tuple[wp.array[wp.int32], wp.array[wp.int32]] | None = None,
) -> odt.BsrMatrix[odt.Block]:
    """
    Hessian smoothness energy with natural boundary conditions.

    The mixed-FEM quadratic form ``Q = H^T M^-1 H`` of Stein et al. 2018, *Natural Boundary
    Conditions for Smoothing in Geometry Processing*: ``x' Q x`` integrates the squared Hessian of
    the piecewise-linear field ``x``, so minimizing it smooths **without** the boundary distortion
    the clamped biharmonic operator
    ([`k_harmonic`][ordito.energies.k_harmonic] at ``k == 2``) produces —
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
        ``(3 * n_faces,)`` triangle index buffer.
    dtype
        Scalar block type of the assembled matrix: ``wp.float64`` (default) or ``wp.float32``.
        Float64 is the default, unlike the first-order operators in this module, because the
        entries scale as the inverse fourth power of the mesh size and the operator exists to be
        solved against.
    vertex_faces
        ``(3 * n_faces,)`` and ``(n_vertices + 1,)`` optional precomputed
        [`vertex_face_adjacency`][ordito.adjacency.vertex_face_adjacency] as
        ``(vertex_faces, offsets)``. Depends on the connectivity alone, so one CSR serves every
        incidence walk over the same mesh --
        [`Trimesh.vertex_face_adjacency`][ordito.mesh.Trimesh.vertex_face_adjacency] has it cached.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` positive semi-definite energy matrix on
        ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``vertex_faces`` are not all on one device.

    See Also
    --------
    [`curved_hessian_energy`][ordito.energies.curved_hessian_energy]
    [`k_harmonic`][ordito.energies.k_harmonic]
    [`cotmatrix`][ordito.laplacian.cotmatrix]

    Notes
    -----
    Matches ``igl::hessian_energy`` except on degenerate faces, which contribute nothing here and
    ``NaN`` there.

    Examples
    --------
    A linear function costs nothing under the Hessian energy, boundary included, where the clamped
    biharmonic operator penalizes it at the boundary:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(6, 6))
    x = wp.array(vertices.numpy()[:, 0], dtype=wp.float64)
    hessian = od.energies.hessian_energy(vertices, faces)
    print(hessian.shape)  # (36, 36)
    print(np.abs((hessian @ x).numpy()).max() < 1e-8)  # True

    L = od.laplacian.cotmatrix(vertices, faces, dtype=wp.float64)
    mass = od.laplacian.mass_matrix_entries(vertices, faces, dtype=wp.float64)
    biharmonic = od.energies.k_harmonic(L, mass, k=2)
    print(np.abs((biharmonic @ x).numpy()).max() < 1e-8)  # False
    ```
    """
    require_same_device(vertices=vertices, faces=faces, vertex_faces=vertex_faces)
    n_vertices = vertices.size
    n_faces = faces.size // 3
    device = vertices.device
    if n_faces == 0:
        return od.array.empty_square_bsr(n_vertices, dtype, device)

    # The per-face gradients and areas and the per-vertex inverse mass, each written whole before
    # it is read: one allocation.
    gradient_entries, areas, inverse_mass = _launch.carve(
        wp.float64, (9 * n_faces, n_faces, n_vertices), device, zeroed=False
    )
    gradients = gradient_entries.reshape((3 * n_faces, 3)).view(wp.vec3d)
    # The triplets below are emitted in the vertex-face rows' order and summed in that order, so
    # a reproducible result needs ordered rows as well as an ordered mass sum: under
    # ``ordito.config``'s deterministic mode the kernel writes each corner's share and one sort of
    # the corners gives both; otherwise it adds the shares atomically.
    deterministic = is_deterministic()
    mass = _launch.zeros(0 if deterministic else n_vertices, dtype=wp.float64, device=device)
    corner_mass = _launch.empty(
        (n_faces if deterministic else 0, 3), dtype=wp.float64, device=device
    )
    _launch.launch(
        kernel_energies.hessian_face_terms,
        dim=n_faces,
        inputs=[vertices, faces, gradients, areas, mass, corner_mass],
        device=device,
    )
    sorted_rows = None
    if deterministic:
        mass, sorted_rows = _sorted_vertex_corner_sums(
            faces, corner_mass, n_vertices, with_rows=vertex_faces is None
        )

    if vertex_faces is not None:
        vf_indices, vf_offsets = vertex_faces
    elif sorted_rows is not None:
        vf_indices, vf_offsets = sorted_rows
    else:
        vf_indices, vf_offsets = od.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
    # Each vertex's count lands in the offsets' buffer past its leading zero, scanned in place.
    segment_offsets = _launch.zeros(n_vertices + 1, dtype=wp.int32, device=device)
    counts = _launch.view(segment_offsets, 1, n_vertices + 1)
    _launch.launch(
        kernel_energies.hessian_energy_counts,
        dim=n_vertices,
        inputs=[faces, vf_offsets, vf_indices, mass, inverse_mass, counts],
        device=device,
    )
    _launch.array_scan(counts, out_array=counts, inclusive=True)
    # Host readback: only the device knows the scan total, and it sizes the triplet buffers.
    n_triplets = int(read_scalar(segment_offsets))

    keys, order = od.array.csr_key_buffers(n_triplets, device)
    vals = _launch.empty(n_triplets, dtype=dtype, device=device)
    if n_triplets > 0:
        _launch.launch(
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
                wp.int32(n_vertices),
            ],
            outputs=[keys, order, vals],
            device=device,
        )
    return od.array.csr_from_coo_keys(n_vertices, n_vertices, keys, order, vals)


def curved_hessian_energy(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    dtype: type[odt.Block] = wp.float64,
    *,
    unique_edges: odt.Array2dInt32 | None = None,
    edge_map: wp.array[wp.int32] | None = None,
) -> odt.BsrMatrix[odt.Block]:
    """
    Curved Hessian smoothness energy on the Crouzeix-Raviart discretization.

    ``igl::curved_hessian_energy``, from Stein et al. 2020, *A Smoothness Energy without Boundary
    Distortion for Curved Surfaces*: where [`hessian_energy`][ordito.energies.hessian_energy]
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
        ``(3 * n_faces,)`` triangle index buffer.
    dtype
        Scalar block type of the assembled matrix: ``wp.float64`` (default) or ``wp.float32``,
        with float64 the default for the same conditioning reason as
        [`hessian_energy`][ordito.energies.hessian_energy].
    unique_edges, edge_map
        ``(n_edges, 2)`` and ``(3 * n_faces,)`` optional precomputed edge numbering from
        [`edges_unique`][ordito.edges.edges_unique] — pass both or neither — to spare the sort
        that otherwise builds it. [`Trimesh.edges_unique`][ordito.mesh.Trimesh.edges_unique] and
        [`Trimesh.edges_unique_inverse`][ordito.mesh.Trimesh.edges_unique_inverse] are the cached
        ones to pass.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` positive semi-definite energy matrix on
        ``vertices.device``.

    Raises
    ------
    ValueError
        If the mesh is not edge-manifold (some edge is shared by more than two faces), or if only
        one of ``unique_edges`` and ``edge_map`` is passed.
    RuntimeError
        If ``vertices``, ``faces``, ``unique_edges`` and ``edge_map`` are not all on one
        device.

    See Also
    --------
    [`hessian_energy`][ordito.energies.hessian_energy]
    [`crouzeix_raviart_cotmatrix`][ordito.energies.crouzeix_raviart_cotmatrix]
    [`vertex_defects`][ordito.vertices.vertex_defects]

    Examples
    --------
    Constants are in the kernel on a curved surface:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    energy = od.energies.curved_hessian_energy(vertices, faces)
    print(energy.shape)  # (162, 162)
    ones = wp.full(energy.shape[0], 1.0, dtype=wp.float64)
    print(np.abs((energy @ ones).numpy()).max() < 1e-8)  # True
    ```
    """
    require_same_device(
        vertices=vertices, faces=faces, unique_edges=unique_edges, edge_map=edge_map
    )
    n_vertices = vertices.size
    n_faces = faces.size // 3
    device = vertices.device
    if n_faces == 0:
        return od.array.empty_square_bsr(n_vertices, dtype, device)

    unique_edges, inverse = _edge_numbering(vertices, faces, unique_edges, edge_map)
    n_edges = int(unique_edges.shape[0])

    # The buffers below are cut from one allocation per dtype and initial value: the zeroed sums
    # and counters, and the tables their passes write whole.
    angle_entries, par_entries, perp_entries, slot_entries = _launch.carve_typed(
        (
            (wp.float64, 3 * n_faces),
            (wp.float64, 4 * n_edges),
            (wp.float64, 4 * n_edges),
            (wp.int32, 4 * n_edges),
        ),
        device,
    )
    angles = odt.as_array2d(angle_entries.reshape((n_faces, 3)), wp.float64)
    # Summed atomically by the angle pass, or afterwards in a fixed order (see
    # ``_sorted_vertex_corner_sums``) under ``ordito.config``'s deterministic mode; and
    # ``_cr_mass_diagonal``'s edge mass, summed by the same face pass.
    deterministic = is_deterministic()
    angle_sums, mass = _launch.carve(
        wp.float64, (0 if deterministic else n_vertices, n_edges), device, zeroed=True
    )
    # Each unique edge's two halfedges are recorded by the same face pass.
    edge_halfedges = _launch.full((n_edges, 2), -1, dtype=wp.int32, device=device)
    # Each unique edge's halfedge count, and the edge-manifold flag of the launch after.
    cursor, nonmanifold = _launch.carve(wp.int32, (n_edges, 1), device, zeroed=True)
    _launch.launch(
        kernel_energies.internal_angles_and_sums,
        dim=n_faces,
        inputs=[vertices, faces, inverse, cursor],
        outputs=[angles, angle_sums, edge_halfedges, mass],
        device=device,
    )
    if deterministic:
        angle_sums, _ = _sorted_vertex_corner_sums(faces, angles, n_vertices)
    # ``cursor`` now counts each unique edge's halfedges, which settles the topology the rest
    # needs: an edge on three or more faces fails the edge-manifold precondition (flagged by the
    # launch below), and an edge on one is a boundary edge.
    # igl::cr_vector_curvature_correction's kappa scaling, ``angle_defect / angle_sum``, is zero
    # on the boundary (curvature is only corrected at interior vertices) and for a non-positive
    # angle sum: zeroing the boundary's *sums* gives that zero where the triplet kernel forms the
    # kappa as it reads it, and it forms each edge's inverse mass likewise.
    _launch.launch(
        kernel_energies.zero_at_boundary_edges,
        dim=n_edges,
        inputs=[unique_edges, cursor],
        outputs=[angle_sums, nonmanifold],
        device=device,
    )
    # The validation raises on the host, so its flag is read back.
    if int(read_scalar(nonmanifold, 0)) != 0:
        raise ValueError(
            "mesh must be edge-manifold (every edge shared by at most two faces); the "
            "Crouzeix-Raviart discretization curved_hessian_energy is built on is undefined "
            "otherwise, like igl::curved_hessian_energy, which asserts it"
        )

    vertex_slots = slot_entries.reshape((n_edges, 4))
    par = par_entries.reshape((n_edges, 4))
    perp = perp_entries.reshape((n_edges, 4))
    _launch.launch(
        kernel_energies.cr_gradient_rows,
        dim=n_edges,
        inputs=[vertices, faces, unique_edges, edge_halfedges, vertex_slots, par, perp],
        device=device,
    )

    n_triplets = 144 * n_faces
    keys, order = od.array.csr_key_buffers(n_triplets, device)
    vals = _launch.empty(n_triplets, dtype=dtype, device=device)
    _launch.launch(
        kernel_energies.CURVED_HESSIAN_TRIPLETS[dtype],
        dim=n_faces,
        inputs=[
            vertices,
            faces,
            inverse,
            angles,
            angle_sums,
            mass,
            vertex_slots,
            par,
            perp,
            wp.int32(n_vertices),
        ],
        outputs=[keys, order, vals],
        device=device,
    )
    return od.array.csr_from_coo_keys(n_vertices, n_vertices, keys, order, vals)


def crouzeix_raviart_cotmatrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: odt.Array2dFloat | None = None,
    dtype: type = wp.float32,
    *,
    unique_edges: odt.Array2dInt32 | None = None,
    edge_map: wp.array[wp.int32] | None = None,
) -> odt.BsrMatrix[wp.float32]:
    """
    Edge-based Crouzeix-Raviart cotangent stiffness matrix.

    The nonconforming-FEM sibling of [`cotmatrix`][ordito.laplacian.cotmatrix]: degrees of
    freedom live on edge midpoints, so the matrix is ``(n_edges, n_edges)`` and each face couples
    its three edges pairwise with minus four times the half-cotangent at their shared corner
    (positive diagonal — the igl sign convention for this operator, opposite to ``cotmatrix``'s).
    Rows follow [`edges_unique`][ordito.edges.edges_unique]'s edge numbering. Requires an
    edge-manifold mesh, like the igl original (which asserts it) — raises ``ValueError``
    otherwise.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    cot_entries
        ``(n_faces, 3)`` optional precomputed weights from
        [`cotmatrix_entries`][ordito.laplacian.cotmatrix_entries]; computed when ``None``.
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``.
    unique_edges, edge_map
        ``(n_edges, 2)`` and ``(3 * n_faces,)`` optional precomputed edge numbering from
        [`edges_unique`][ordito.edges.edges_unique] — pass both or neither. Sharing it with
        [`crouzeix_raviart_massmatrix`][ordito.energies.crouzeix_raviart_massmatrix] keeps the
        two operators on identical rows without recomputing the sort.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_edges, n_edges)`` stiffness matrix in 1x1-block BSR form on
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
    [`crouzeix_raviart_massmatrix`][ordito.energies.crouzeix_raviart_massmatrix]
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`edges_unique`][ordito.edges.edges_unique]

    Notes
    -----
    Matches ``igl::crouzeix_raviart_cotmatrix`` up to the edge numbering, which follows
    [`edges_unique`][ordito.edges.edges_unique] rather than ``igl::unique_edge_map``.

    Examples
    --------
    One row per edge (an icosphere at two subdivisions has 480) and, like the vertex operator, rows
    that sum to zero:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    L = od.energies.crouzeix_raviart_cotmatrix(vertices, faces)
    print(L.shape)  # (480, 480)
    ones = wp.full(L.shape[0], 1.0, dtype=wp.float32)
    print(np.abs((L @ ones).numpy()).max() < 1e-5)  # True
    ```
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
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return od.array.empty_square_bsr(n_edges, dtype, device)

    # One pass over the faces records each edge's incident faces and flags a third one, and writes
    # the half-cotangents when the caller has none.
    face_counts, nonmanifold = _launch.carve(wp.int32, (n_edges, 1), device, zeroed=True)
    edge_faces = _launch.full(2 * n_edges, -1, dtype=wp.int32, device=device)
    computed_cot = odt.empty_2d(
        (n_faces if cot_entries is None else 0, 3), wp.float32, device=device
    )
    _launch.launch(
        kernel_energies.crouzeix_raviart_face_terms,
        dim=n_faces,
        inputs=[vertices, faces, edge_map, wp.int32(cot_entries is None)],
        outputs=[face_counts, edge_faces, nonmanifold, computed_cot],
        device=device,
    )
    # The validation raises on the host, so its flag is read back.
    if int(read_scalar(nonmanifold, 0)) != 0:
        raise ValueError(
            "mesh must be edge-manifold (every edge shared by at most two faces); the "
            "Crouzeix-Raviart discretization is undefined otherwise, like "
            "igl::crouzeix_raviart_cotmatrix, which asserts it"
        )

    if cot_entries is None:
        cot_entries = computed_cot

    # The pattern follows from the incidence: a row holds its diagonal and the other edges of its
    # (at most two) faces, so at most five entries, and the count pass and a scan give the offsets.
    # The matrix's three arrays in one allocation; the count pass writes the leading zero.
    offsets, columns, values = _launch.carve_typed(
        ((wp.int32, n_edges + 1), (wp.int32, 5 * n_edges), (dtype, 5 * n_edges)), device
    )
    _launch.launch(
        kernel_energies.crouzeix_raviart_row_counts,
        dim=n_edges,
        inputs=[edge_map, edge_faces],
        outputs=[offsets],
        device=device,
    )
    _launch.array_scan(offsets, out_array=offsets, inclusive=True)
    _launch.launch(
        kernel_energies.CROUZEIX_RAVIART_CSR[cot_entries.dtype, dtype],
        dim=n_edges,
        inputs=[edge_map, cot_entries, edge_faces, offsets],
        outputs=[columns, values],
        device=device,
    )
    return od.array.bsr_from_csr(n_edges, n_edges, offsets, columns, values)


def crouzeix_raviart_massmatrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    dtype: type = wp.float32,
    *,
    unique_edges: odt.Array2dInt32 | None = None,
    edge_map: wp.array[wp.int32] | None = None,
) -> odt.BsrMatrix[wp.float32]:
    """
    Edge-based Crouzeix-Raviart mass matrix.

    Diagonal ``(n_edges, n_edges)``: each face donates a third of its area to each of its three
    edges, so an interior edge's entry is a third of its two incident faces' summed area. Rows
    follow [`edges_unique`][ordito.edges.edges_unique]'s edge numbering, the same numbering
    [`crouzeix_raviart_cotmatrix`][ordito.energies.crouzeix_raviart_cotmatrix] uses.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``.
    unique_edges, edge_map
        ``(n_edges, 2)`` and ``(3 * n_faces,)`` optional precomputed edge numbering from
        [`edges_unique`][ordito.edges.edges_unique] — pass both or neither.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_edges, n_edges)`` diagonal mass matrix in 1x1-block BSR form on
        ``vertices.device``.

    Raises
    ------
    ValueError
        If exactly one of ``unique_edges`` / ``edge_map`` is provided.
    RuntimeError
        If ``vertices``, ``faces``, ``unique_edges`` and ``edge_map`` are not all on one device.

    See Also
    --------
    [`crouzeix_raviart_cotmatrix`][ordito.energies.crouzeix_raviart_cotmatrix]
    [`mass_matrix`][ordito.laplacian.mass_matrix]
    [`edges_unique`][ordito.edges.edges_unique]

    Notes
    -----
    Matches ``igl::crouzeix_raviart_massmatrix`` up to the edge numbering, as above.

    Examples
    --------
    One entry per edge of the cube, summing to its surface area:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    M = od.energies.crouzeix_raviart_massmatrix(vertices, faces)
    ones = wp.full(M.shape[0], 1.0, dtype=wp.float32)
    print(M.shape, f"{(M @ ones).numpy().sum():.4f}")  # (18, 18) 6.0000
    ```
    """
    require_same_device(
        vertices=vertices, faces=faces, unique_edges=unique_edges, edge_map=edge_map
    )
    unique_edges, edge_map = _edge_numbering(vertices, faces, unique_edges, edge_map)
    n_edges = int(unique_edges.shape[0])
    device = faces.device
    if n_edges == 0:
        return od.array.empty_square_bsr(0, dtype, device)
    # The diagonal is the matrix's value array as it stands, under a diagonal pattern.
    values = _cr_mass_diagonal(vertices, faces, edge_map, n_edges, dtype)
    offsets, columns = _launch.carve(wp.int32, (n_edges + 1, n_edges), device)
    _launch.launch(
        kernel_laplacian.diagonal_pattern,
        dim=n_edges,
        inputs=[],
        outputs=[offsets, columns],
        device=device,
    )
    return cast(
        "odt.BsrMatrix[wp.float32]",
        od.array.bsr_from_csr(n_edges, n_edges, offsets, columns, values, nnz=n_edges),
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
    [`crouzeix_raviart_massmatrix`][ordito.energies.crouzeix_raviart_massmatrix] and
    [`curved_hessian_energy`][ordito.energies.curved_hessian_energy] cannot drift onto different
    masses -- the latter's derivation assumes they are the same one.
    """
    n_faces = faces.size // 3
    mass = _launch.zeros(n_edges, dtype=dtype, device=faces.device)
    if n_faces > 0:
        _launch.launch(
            kernel_energies.CROUZEIX_RAVIART_MASS_DIAG[dtype],
            dim=n_faces,
            inputs=[vertices, faces, edge_map, mass],
            device=faces.device,
        )
    return mass


def lscm_hessian(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    cot_entries: odt.Array2dFloat | None = None,
    pattern: MeshOperatorPattern | None = None,
) -> odt.BsrMatrix[wp.float64]:
    """
    LSCM Hessian ``Q = -repdiag(L, 2) - 2 A``.

    Assembles the ``(2n, 2n)`` symmetric operator behind the least-squares conformal map, where
    ``L`` is the cotangent Laplacian [`cotmatrix`][ordito.laplacian.cotmatrix] (negative-diagonal
    convention), ``repdiag(L, 2)`` is the block-diagonal ``[[L, 0], [0, L]]``, and ``A`` is the
    boundary [`vector_area_matrix`][ordito.energies.vector_area_matrix]. Built natively in
    float64 by merging the rows of ``L`` and ``A`` (the within-quadrant repdiag entries and the
    cross-quadrant ``-2 A`` entries never collide), so it feeds the float64 conjugate-gradient
    solve directly. Matches the ``Q`` returned by ``igl.lscm`` exactly.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    cot_entries, pattern
        As [`cotmatrix`][ordito.laplacian.cotmatrix] takes them for ``L``: precomputed
        half-cotangent weights and sparsity, to spare their build when the Hessian is assembled
        repeatedly over one mesh.
        [`Trimesh.cotmatrix_entries`][ordito.mesh.Trimesh.cotmatrix_entries] and
        [`Trimesh.mesh_operator_pattern`][ordito.mesh.Trimesh.mesh_operator_pattern] are the cached
        ones to pass.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(2 * n_vertices, 2 * n_vertices)`` matrix in 1x1-block BSR form on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``cot_entries`` and ``pattern`` are not all on one device.
    ValueError
        If ``cot_entries`` is given and its row count does not match ``faces``' triangle count, or
        ``pattern`` was built for another operator or vertex count.

    See Also
    --------
    [`lscm`][ordito.parametrization.lscm]
    [`vector_area_matrix`][ordito.energies.vector_area_matrix]
    [`cotmatrix`][ordito.laplacian.cotmatrix]

    Notes
    -----
    Matches ``igl::lscm_hessian``.

    Examples
    --------
    A flat patch's identity map is conformal, so its LSCM energy is zero:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(4, 4))
    Q = od.energies.lscm_hessian(vertices, faces)
    print(Q.shape)  # (32, 32)
    uv = np.concatenate([vertices.numpy()[:, 0], vertices.numpy()[:, 1]])
    energy = np.dot(uv, (Q @ wp.array(uv, dtype=wp.float64)).numpy())
    print(abs(energy) < 1e-6)  # True
    ```
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        cot_entries=cot_entries,
        pattern=None if pattern is None else pattern.offsets,
    )
    n = vertices.size
    device = vertices.device
    if pattern is not None and (pattern.operator != "cotmatrix" or pattern.n_vertices != n):
        raise ValueError(
            f"pattern was built for operator={pattern.operator!r} over {pattern.n_vertices} "
            f"vertices; lscm_hessian needs operator='cotmatrix' over {n}"
        )
    if n == 0:
        return od.array.empty_square_bsr(0, wp.float64, device)
    if faces.size == 0:
        return od.array.empty_square_bsr(2 * n, wp.float64, device)
    n_faces = faces.size // 3
    if cot_entries is None:
        cot_entries = cotmatrix_entries(vertices, faces, dtype=wp.float64)
    elif int(cot_entries.shape[0]) != n_faces:
        raise ValueError(
            f"cot_entries must have shape (n_faces, 3) = ({n_faces}, 3), "
            f"got {(int(cot_entries.shape[0]), int(cot_entries.shape[1]))}"
        )
    # ``L``'s pattern also carries every edge's halfedges, so the boundary term is read off it
    # rather than paired and sorted again (``kernels/energies.lscm_boundary_side``), in the launch
    # that writes ``L``'s values.
    if pattern is None:
        pattern = mesh_operator_pattern(faces, n)
    _, _, lap_offsets, lap_columns, run_start, keys, n_keys, order = pattern
    laplacian_values, area_offsets = _launch.carve_typed(
        ((wp.float64, lap_columns.size), (wp.int32, n + 1)), device
    )
    _launch.launch(
        kernel_energies.LSCM_LAPLACIAN_ROWS_AND_COUNTS[cot_entries.dtype],
        dim=n,
        inputs=[
            faces,
            lap_offsets,
            lap_columns,
            run_start,
            keys,
            wp.int32(n_keys),
            order,
            cot_entries,
        ],
        outputs=[laplacian_values, area_offsets],
        device=device,
    )
    _launch.array_scan(area_offsets, out_array=area_offsets, inclusive=True)
    # Sizes the entry buffers: the one readback, of the boundary entry count, which the triplet
    # build of ``A`` read as its boundary edge count. Capacities, not counts: ``L``'s columns may
    # be longer than its stored entries.
    capacity = 2 * lap_columns.size + 2 * int(read_scalar(area_offsets))
    offsets, columns, values = _launch.carve_typed(
        ((wp.int32, 2 * n + 1), (wp.int32, capacity), (wp.float64, capacity)), device
    )
    _launch.launch(
        kernel_energies.lscm_hessian_csr,
        dim=2 * n,
        inputs=[
            faces,
            lap_offsets,
            lap_columns,
            laplacian_values,
            run_start,
            keys,
            wp.int32(n_keys),
            order,
            area_offsets,
            wp.int32(n),
        ],
        outputs=[offsets, columns, values],
        device=device,
    )
    return od.array.bsr_from_csr(2 * n, 2 * n, offsets, columns, values)


def vector_area_matrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    mates: wp.array[wp.int32] | None = None,
) -> odt.BsrMatrix[wp.float64]:
    """
    Boundary vector-area matrix ``A``: the signed area enclosed by the UV boundary curve.

    Assembles the ``(2n, 2n)`` matrix that turns the ``[u; v]`` quadratic form into the signed area
    enclosed by the boundary UV curve: for each **oriented** boundary edge ``(i, j)`` (from the face
    winding, via [`oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges]) it adds the
    cross-quadrant entries ``(i+n, j, -1/4)``, ``(j, i+n, -1/4)``, ``(i, j+n, +1/4)``,
    ``(j+n, i, +1/4)``. On a closed mesh (no boundary) ``A`` is the zero matrix. Built natively in
    float64 in a single [`csr_from_coo_keys`][ordito.array.csr_from_coo_keys].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions; only the count and device are used.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    mates
        ``(3 * n_faces,)`` [`halfedge_mates`][ordito.halfedge.halfedge_mates] of ``faces`` when the
        caller holds them ([`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates]), which
        the boundary search reads rather than pairing the halfedges again. Trusted, not checked.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(2 * n_vertices, 2 * n_vertices)`` matrix in 1x1-block BSR form on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``mates`` are not all on one device.

    See Also
    --------
    [`lscm_hessian`][ordito.energies.lscm_hessian]
    [`lscm`][ordito.parametrization.lscm]
    [`oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges]

    Notes
    -----
    Matches ``igl::vector_area_matrix``.

    Examples
    --------
    The quadratic form of the identity map is the area of a ``2 x 3`` rectangle; a closed cube has
    no boundary and an empty matrix:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(4, 4), extents=(2.0, 3.0))
    A = od.energies.vector_area_matrix(vertices, faces)
    print(A.shape)  # (32, 32)
    uv = np.concatenate([vertices.numpy()[:, 0], vertices.numpy()[:, 1]])
    print(f"{np.dot(uv, (A @ wp.array(uv, dtype=wp.float64)).numpy()):.4f}")  # 6.0000

    vertices, faces = od.creation.box()
    print(od.energies.vector_area_matrix(vertices, faces).nnz_sync())  # 0
    ```
    """
    require_same_device(vertices=vertices, faces=faces, mates=mates)
    device = faces.device
    n_vertices = vertices.size
    boundary = od.boundary.oriented_boundary_edges(vertices, faces, mates=mates)
    n_be = int(boundary.shape[0])
    if n_be == 0:
        return od.array.empty_square_bsr(2 * n_vertices, wp.float64, device)

    keys, order = od.array.csr_key_buffers(4 * n_be, device)
    vals = _launch.empty(4 * n_be, dtype=wp.float64, device=device)
    # The four cross-quadrant entries of every oriented boundary edge, written as their keys.
    _launch.launch(
        kernel_energies.vector_area_triplets,
        dim=n_be,
        inputs=[boundary, wp.int32(n_vertices)],
        outputs=[keys, order, vals],
        device=device,
    )
    return od.array.csr_from_coo_keys(2 * n_vertices, 2 * n_vertices, keys, order, vals)


def _edge_numbering(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: odt.Array2dInt32 | None,
    edge_map: wp.array[wp.int32] | None,
) -> tuple[odt.Array2dInt32, wp.array[wp.int32]]:
    """Validate or build the shared ``edges_unique`` numbering the edge-based operators index."""
    if (unique_edges is None) != (edge_map is None):
        raise ValueError("pass unique_edges and edge_map together, or neither.")
    if unique_edges is None or edge_map is None:
        unique_edges, edge_map = edges_unique(faces, n_vertices=vertices.size, validate=False)
    return unique_edges, edge_map


def _sorted_vertex_corner_sums(
    faces: wp.array[wp.int32],
    corner_values: odt.Array2dFloat64,
    n_vertices: int,
    *,
    with_rows: bool = False,
) -> tuple[wp.array[wp.float64], tuple[wp.array[wp.int32], wp.array[wp.int32]] | None]:
    """
    Each vertex's sum of its corners' ``(n_faces, 3)`` values, in ascending corner order.

    The reproducible form of an atomic per-vertex scatter, which ``float64`` values have no exact
    form of, for one sort of the face buffer. With ``with_rows`` the same walk also returns the
    vertex-face incidence ``(vertex_faces, offsets)`` of
    [`vertex_face_adjacency`][ordito.adjacency.vertex_face_adjacency], every row in ascending face
    order.
    """
    device = faces.device
    sorted_vertices, corner_order = od.array.sort_and_argsort(
        faces, end_bit=max(1, (n_vertices - 1).bit_length())
    )
    sums = _launch.empty(n_vertices, dtype=wp.float64, device=device)
    offsets = _launch.empty(n_vertices + 1 if with_rows else 0, dtype=wp.int32, device=device)
    vertex_faces = _launch.empty(faces.size if with_rows else 0, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_scatter.SUM_SORTED_CORNER_VALUES[wp.float64],
        dim=n_vertices,
        inputs=[sorted_vertices, corner_order, corner_values],
        outputs=[sums, offsets, vertex_faces],
        device=device,
    )
    return sums, ((vertex_faces, offsets) if with_rows else None)
