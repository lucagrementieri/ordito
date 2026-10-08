"""
Discrete Laplacians on a triangle mesh, and the intrinsic repairs that keep them well-behaved.

Every Laplacian this package builds lives here: the cotangent (stiffness) operator and its
vector-valued sibling, the combinatorial and row-normalized 1-ring operators, and the lumped mass
matrix that pairs with them.

The cotangent operator needs only *edge lengths*, not vertex positions — which is what makes it
repairable without moving anything. A sliver triangle produces a huge cotangent weight and solves
that are ill-conditioned; a *degenerate* one, whose three edge lengths fail the triangle inequality
outright (common after ``float32`` rounding, decimation, or a boolean), has no finite weight at all.
The assembly refuses to divide by such a face's zero area, so it contributes nothing — the operator
stays finite, but that face's edge couplings are simply **missing** from it, which is a wrong
operator rather than an unusable one.

[`mollify_intrinsic`][ordito.laplacian.mollify_intrinsic] (Sharp & Crane 2020) fixes both by adding
one global constant to every edge length — the smallest that restores the triangle inequality with a
margin. The perturbation is slight and uniform, which beats the alternatives: the operator stays
symmetric, no vertex moves, no connectivity changes, and a mesh that is already fine gets
``delta = 0`` and is untouched. Mollification makes the weights *finite*;
[`intrinsic_delaunay`][ordito.remesh.intrinsic_delaunay] (in
[`ordito.remesh`][ordito.remesh], beside the extrinsic flipper) makes them *non-negative*.
[`robust_laplacian`][ordito.laplacian.robust_laplacian] combines them.

The **higher-order** operators assembled *from* these -- the integrated k-harmonic form, the two
Hessian smoothness energies, the edge-based Crouzeix-Raviart pair and the LSCM Hessian -- live one
module along in [`ordito.energies`][ordito.energies]. The line is first order against second: this
module builds the operators, that one builds the quadratic forms a solver minimizes. A reader
arriving from libigl, where ``cotmatrix`` and ``crouzeix_raviart_cotmatrix`` sit together, wants
[`crouzeix_raviart_cotmatrix`][ordito.energies.crouzeix_raviart_cotmatrix].
"""

from __future__ import annotations

import warnings
from typing import Literal, NamedTuple, overload

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, require_same_device
from ordito.constants import TOLERANCE_MOLLIFY
from ordito.edges import edges_unique, face_edge_lengths, faces_to_edges
from ordito.kernels import laplacian as kernel_laplacian
from ordito.kernels import scatter as kernel_scatter
from ordito.kernels import triangles as kernel_triangles
from ordito.reduce import max as reduce_max
from ordito.reduce import mean as reduce_mean
from ordito.tangent_space import halfedge_transport_angles
from ordito.triangles import face_normals_and_areas

# Face count from which a mesh operator's pattern is built from undirected keys plus a 32-bit
# second sort rather than one sort of directed keys (``_mesh_operator_pattern``). Both give the
# identical matrix; the undirected build sorts less and holds less memory, for three more launches.
# Measured on ``cotmatrix``: directed wins below ~0.5 M faces (undirected 0.55-0.59x at 16-82 k,
# 0.78x at 328 k), undirected above (1.35x at 871 k and 28 M, 1.45x at 1.3 M), where it also holds
# a fifth less memory.
_UNDIRECTED_PATTERN_FROM_FACES = 500_000


def face_gradients(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float64],
    *,
    face_normals: wp.array[wp.vec3] | None = None,
    face_areas: wp.array[wp.float32] | None = None,
) -> wp.array[wp.vec3d]:
    """
    Gradient of a per-vertex scalar field inside each face, as a vector in that face's plane.

    The piecewise-linear gradient is constant per triangle:
    ``grad = 1/(2A) * sum_k values[k] * (n x e_k)``, with ``e_k`` the counter-clockwise edge
    opposite corner ``k``. It satisfies ``dot(grad, e) == values[end] - values[start]`` for every
    edge of the face, which is the property that makes it the discrete gradient rather than a finite
    difference.

    Accumulated in ``float64`` and returned as ``wp.vec3d``: the fields this serves decay
    exponentially (diffused heat, geodesic distance), and a ``float32`` sum of the three cross
    products loses the far field. A degenerate face gets the zero vector rather than a division by
    its zero area.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    values
        ``(n_vertices,)`` scalar field.
    face_normals, face_areas
        ``(n_faces,)`` precomputed per-face unit normals and areas, as returned by
        [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]. Recomputed when
        either is ``None``.

    Returns
    -------
    wp.array[wp.vec3d]
        ``(n_faces,)`` gradient vectors on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``values``, ``face_normals`` and ``face_areas`` are not all on
        one device.

    Notes
    -----
    ``igl.grad(V, F)`` is the same operator in *matrix* form, a sparse ``(3 * n_faces, n_vertices)``
    map whose product with the field stacks the gradients as ``[all x; all y; all z]``. ordito
    returns the applied result instead of the matrix, because that is what every in-repo consumer
    wants -- the heat method takes the normalized gradient face by face and never needs the operator
    itself. ``tests/test_laplacian.py`` compares the two through that product.

    See Also
    --------
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]
    ``igl.grad``
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        values=values,
        face_normals=face_normals,
        face_areas=face_areas,
    )
    device = vertices.device
    n_faces = faces.size // 3
    gradients = _launch.empty(n_faces, dtype=wp.vec3d, device=device)
    if n_faces == 0:
        return gradients

    if face_normals is None or face_areas is None:
        face_normals, face_areas = face_normals_and_areas(vertices, faces)
    _launch.launch(
        kernel_triangles.face_gradients,
        dim=n_faces,
        inputs=[vertices, faces, face_normals, face_areas, values, gradients],
        device=device,
    )
    return gradients


@overload
def cotmatrix_entries(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], dtype: type[wp.float32] = wp.float32
) -> odt.Array2dFloat32: ...
@overload
def cotmatrix_entries(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], dtype: type[wp.float64]
) -> odt.Array2dFloat64: ...
def cotmatrix_entries(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], dtype: type = wp.float32
) -> odt.Array2dFloat:
    """
    Per-triangle half-cotangent weights.

    For each triangle face, column ``e`` stores ``1/2 * cot(angle at vertex e)`` for the
    edge opposite that vertex. Columns follow igl edge order: opposite vertices 0, 1, 2.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    dtype
        Scalar type of the returned weights: ``wp.float32`` (default) or ``wp.float64``. The weights
        are computed in float32 (the vertex precision) and cast to ``dtype`` on write; request
        ``wp.float64`` to feed a native float64 [`cotmatrix`][ordito.laplacian.cotmatrix] build.

    Returns
    -------
    odt.Array2dFloat
        ``(n_faces, 3)`` weights on ``faces.device``. Empty ``(0, 3)`` when ``n_faces == 0``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`cotmatrix_entries_intrinsic`][ordito.laplacian.cotmatrix_entries_intrinsic]
    [`cotmatrix`][ordito.laplacian.cotmatrix]

    Notes
    -----
    Matches ``igl::cotmatrix_entries``, column order included.
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return odt.empty_2d((0, 3), dtype, device=device)

    out_cot = odt.empty_2d((n_faces, 3), dtype, device=device)
    _launch.launch(
        kernel_laplacian.COTMATRIX_ENTRIES[dtype],
        dim=n_faces,
        inputs=[vertices, faces, out_cot],
        device=device,
    )
    return odt.as_array2d(out_cot, dtype)


@overload
def cotmatrix_entries_intrinsic(
    edge_lengths: odt.Array2dFloat32, dtype: type[wp.float32] = wp.float32
) -> odt.Array2dFloat32: ...
@overload
def cotmatrix_entries_intrinsic(
    edge_lengths: odt.Array2dFloat32, dtype: type[wp.float64]
) -> odt.Array2dFloat64: ...
def cotmatrix_entries_intrinsic(
    edge_lengths: odt.Array2dFloat32, dtype: type = wp.float32
) -> odt.Array2dFloat:
    """
    Per-triangle half-cotangent weights from edge lengths.

    (``igl::cotmatrix_entries`` intrinsic overload).

    Each row gives the three edge lengths opposite vertices 0, 1, and 2 of the corresponding
    triangle.

    Parameters
    ----------
    edge_lengths
        ``(n_faces, 3)`` edge lengths on the target device.
    dtype
        Scalar type of the returned weights: ``wp.float32`` (default) or ``wp.float64``.

    Returns
    -------
    odt.Array2dFloat
        ``(n_faces, 3)`` weights on ``edge_lengths.device``. Empty ``(0, 3)`` when ``n_faces == 0``.

    Raises
    ------
    TypeError
        If ``edge_lengths`` is not rank 2 or not ``wp.float32``.

    See Also
    --------
    [`cotmatrix_entries`][ordito.laplacian.cotmatrix_entries]
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    """
    odt.ensure_ndim(edge_lengths, 2, dtype=wp.float32)
    n_faces = int(edge_lengths.shape[0])
    device = edge_lengths.device
    if n_faces == 0:
        return odt.empty_2d((0, 3), dtype, device=device)

    out_cot = odt.empty_2d((n_faces, 3), dtype, device=device)
    _launch.launch(
        kernel_laplacian.COTMATRIX_ENTRIES_INTRINSIC[dtype],
        dim=n_faces,
        inputs=[edge_lengths, out_cot],
        device=device,
    )
    return odt.as_array2d(out_cot, dtype)


@overload
def cotmatrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: odt.Array2dFloat | None = None,
    dtype: type[wp.float32] = wp.float32,
    *,
    pattern: MeshOperatorPattern | None = None,
) -> odt.BsrMatrix[wp.float32]: ...
@overload
def cotmatrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: odt.Array2dFloat | None,
    dtype: type[wp.float64],
    *,
    pattern: MeshOperatorPattern | None = None,
) -> odt.BsrMatrix[wp.float64]: ...
@overload
def cotmatrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: odt.Array2dFloat | None = None,
    *,
    dtype: type[wp.float64],
    pattern: MeshOperatorPattern | None = None,
) -> odt.BsrMatrix[wp.float64]: ...
def cotmatrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: odt.Array2dFloat | None = None,
    dtype: type = wp.float32,
    *,
    pattern: MeshOperatorPattern | None = None,
) -> odt.BsrMatrix[wp.float32] | odt.BsrMatrix[wp.float64]:
    """
    Cotangent stiffness matrix of the mesh: the discrete Laplace-Beltrami operator.

    Builds the sparse ``(n_vertices, n_vertices)`` matrix from triangle geometry. Diagonal
    entries are **negative** (each row sums to zero); ``-L`` is positive semi-definite on
    closed meshes.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    cot_entries
        ``(n_faces, 3)`` precomputed weights from
        [`cotmatrix_entries`][ordito.laplacian.cotmatrix_entries]. When ``None``, entries are
        computed from ``vertices`` and ``faces`` in ``dtype``. May be ``float32`` or ``float64``
        regardless of ``dtype``: the assembly kernel casts them to the matrix precision. Must have
        ``shape[0] == n_faces``.
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``. Use
        ``wp.float64`` when the matrix feeds an ill-conditioned solve (e.g. the biharmonic operator
        in [`harmonic`][ordito.parametrization.harmonic]); the entries are always assembled in
        the requested precision, in one build.
    pattern
        Optional sparsity from [`mesh_operator_pattern`][ordito.laplacian.mesh_operator_pattern]
        (``operator="cotmatrix"``) built on these same ``faces``. Only the values are then
        computed; the returned matrix shares the pattern's ``offsets`` and ``columns``.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` square cotangent matrix in 1x1-block BSR form on
        ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``cot_entries`` and ``pattern`` are not all on one device.
    ValueError
        If ``cot_entries`` is given and its row count does not match ``faces``' triangle count, or
        ``pattern`` was built for another operator or vertex count.

    See Also
    --------
    [`cotmatrix_entries`][ordito.laplacian.cotmatrix_entries]
    [`crouzeix_raviart_cotmatrix`][ordito.energies.crouzeix_raviart_cotmatrix]
        The nonconforming-FEM sibling, with the degrees of freedom on edge midpoints instead of
        vertices. libigl keeps the two together; here it lives in
        [`ordito.energies`][ordito.energies] with the operators it is assembled for.
    [`mass_matrix`][ordito.laplacian.mass_matrix]
    [`edges_to_csr`][ordito.graph.edges_to_csr]

    Notes
    -----
    Matches ``igl::cotmatrix``, sign convention included; asserted in ``tests/test_laplacian.py``.
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        cot_entries=cot_entries,
        pattern=None if pattern is None else pattern.offsets,
    )
    n_vertices = vertices.size
    n_faces = faces.size // 3
    device = vertices.device
    _check_pattern(pattern, "cotmatrix", n_vertices)

    if n_faces == 0:
        return od.array.empty_square_bsr(n_vertices, dtype, device)

    if cot_entries is None:
        cot_entries = cotmatrix_entries(vertices, faces, dtype=dtype)
    elif int(cot_entries.shape[0]) != n_faces:
        raise ValueError(
            f"cot_entries must have shape (n_faces, 3) = ({n_faces}, 3), "
            f"got {(int(cot_entries.shape[0]), int(cot_entries.shape[1]))}"
        )

    # The pattern comes from the faces (``_mesh_operator_pattern``) with no triplets in between;
    # one row kernel then forms each off-diagonal from its contributing half-cotangents and the
    # diagonal from the row sum, casting the (float32 or float64) weights to the matrix dtype.
    if pattern is None:
        pattern = mesh_operator_pattern(faces, n_vertices)
    _, _, offsets, columns, run_start, keys, count, order = pattern
    values = _launch.empty(columns.size, dtype=dtype, device=device)
    _launch.launch(
        kernel_laplacian.COTMATRIX_ROWS[cot_entries.dtype, dtype],
        dim=n_vertices,
        inputs=[offsets, columns, run_start, keys, wp.int32(count), order, cot_entries, values],
        device=device,
    )
    return od.array.bsr_from_csr(n_vertices, n_vertices, offsets, columns, values)


# ``robust_laplacian`` warns when rounding its ``float32`` lengths can move an operator row by more
# than this fraction of the row's scale (``kernels/laplacian.intrinsic_rounding_rows``, a bound the
# measured row error sits about 7x under). Measured 2026-10-08: <= 1e-6 on every needle-free fixture
# and on ``saddle_graded`` (whose needles' rows it puts at 5.4e-7), 2e-4 to 3.9e-4 on
# ``sphere_irregular`` / the irregular tori without the Delaunay flips (measured row error 2.4e-5 to
# 5.6e-5), 9e-6 to 2.6e-5 with them (the flips remove the needles). 1e-4 is ~1.4e-5 of real row
# error, 100x the ``float32`` floor.
_INTRINSIC_ROUNDING_LIMIT = 1e-4


def robust_laplacian(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    epsilon: float = TOLERANCE_MOLLIFY,
    dtype: type = wp.float32,
    *,
    use_intrinsic_delaunay: bool = True,
) -> odt.BsrMatrix[wp.float32]:
    """
    Cotangent Laplacian that a bad triangulation cannot poison, via mollification and flips.

    Two independent repairs, both intrinsic — no vertex moves, so the surface is unchanged:

    * **mollification** adds one constant to every edge length so that no triangle is degenerate,
      which is what keeps the weights finite at all
      ([`mollify_intrinsic`][ordito.laplacian.mollify_intrinsic]);
    * **intrinsic Delaunay flips** retriangulate toward non-negative cotangent weights, which is
      what makes the operator satisfy a maximum principle
      ([`intrinsic_delaunay`][ordito.remesh.intrinsic_delaunay]). Not *until* — that function
      keeps a simplicial output, so it declines a flip whose new edge already joins the same two
      vertices, and a strongly graded surface can leave a handful of negative weights behind that
      no number of rounds removes. See its Notes; check the result rather than assuming it.

    With both on this is ``igl::intrinsic_delaunay_cotmatrix``, and the operator
    ``potpourri3d``'s ``use_robust=True`` solvers build. Turn the flips off for a drop-in
    [`cotmatrix`][ordito.laplacian.cotmatrix] that keeps every edge coupling: the plain operator
    drops the ones belonging to a degenerate face, because a zero-area triangle has no finite
    cotangent to contribute.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    epsilon
        Triangle-inequality margin, relative to the mean edge length. The default ``1e-5`` is
        Sharp & Crane's.
    dtype
        Scalar type of the matrix: ``wp.float32`` (default) or ``wp.float64``.
    use_intrinsic_delaunay
        Flip to the intrinsic Delaunay triangulation first (default), the name and the default
        ``potpourri3d``'s solvers use. The vertex set — and so the matrix's shape and meaning — is
        the same either way; only the edges it sums over change.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` cotangent stiffness matrix, in ``cotmatrix``'s sign convention
        (negative diagonal, so ``-L`` is positive semi-definite).

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Warns
    -----
    UserWarning
        When the ``float32`` edge lengths do not determine the weights (see Notes).

    Notes
    -----
    The operator is built from ``float32`` edge lengths, and a needle triangle's cotangents are
    sensitive to its lengths as ``eps * aspect^2``: on a mesh with aspect ratios in the hundreds the
    rounding of the lengths alone moves operator rows by around 1e-5 to 1e-4 of their scale, where
    [`cotmatrix`][ordito.laplacian.cotmatrix], built from positions, stays near ``float32``'s
    1e-7. The call bounds that movement row by row and warns when it exceeds 1e-4. The Delaunay
    flips usually remove such needles; the warning is mostly a hazard of
    ``use_intrinsic_delaunay=False``.

    See Also
    --------
    [`intrinsic_delaunay`][ordito.remesh.intrinsic_delaunay]
    [`mollify_intrinsic`][ordito.laplacian.mollify_intrinsic]
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`heat_geodesic`][ordito.heat.heat_geodesic]
    """
    require_same_device(vertices=vertices, faces=faces)
    if use_intrinsic_delaunay:
        intrinsic_faces, lengths, _ = od.remesh.intrinsic_delaunay(vertices, faces, epsilon=epsilon)
    else:
        intrinsic_faces = faces
        lengths, _ = mollify_intrinsic(vertices, faces, epsilon=epsilon)
    entries = cotmatrix_entries_intrinsic(lengths, dtype=dtype)
    _warn_if_lengths_undetermined(vertices.size, intrinsic_faces, lengths)
    return cotmatrix(vertices, intrinsic_faces, cot_entries=entries, dtype=dtype)


def _warn_if_lengths_undetermined(
    n_vertices: int, faces: wp.array[wp.int32], lengths: odt.Array2dFloat32
) -> None:
    """Warn when rounding the ``float32`` lengths alone can move an operator row noticeably."""
    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return
    error = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    scale = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_laplacian.intrinsic_rounding_rows,
        dim=n_faces,
        inputs=[faces, lengths],
        outputs=[error, scale],
        device=device,
    )
    worst = _launch.zeros(1, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_laplacian.max_row_ratio,
        dim=n_vertices,
        inputs=[error, scale],
        outputs=[worst],
        device=device,
    )
    # Host readback: the one number the warning decides on.
    ratio = float(read_scalar(worst))
    if ratio > _INTRINSIC_ROUNDING_LIMIT:
        warnings.warn(
            f"robust_laplacian: rounding the float32 edge lengths alone can move an operator row "
            f"by {ratio:.2g} of its scale: needle triangles are not determined by their lengths "
            "to float32 precision. cotmatrix, built from positions, does not have this limit.",
            stacklevel=3,
        )


def mollify_intrinsic(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], epsilon: float = TOLERANCE_MOLLIFY
) -> tuple[odt.Array2dFloat32, float]:
    """
    Add the smallest constant to every edge length that makes every triangle non-degenerate.

    Returns the mollified ``(n_faces, 3)`` length table and the constant used. The constant is a
    single global number, which is the point: it keeps the perturbation uniform, so the operators
    built from these lengths stay symmetric and no triangle is treated as a special case.

    ``delta`` is zero, and the lengths unchanged, whenever every triangle already satisfies the
    triangle inequality with margin ``epsilon * mean_edge_length``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    epsilon
        Required margin, relative to the mean edge length.

    Returns
    -------
    lengths : odt.Array2dFloat32
        ``(n_faces, 3)`` mollified edge lengths, column ``e`` opposite corner ``e``.
    delta : float
        The constant added to every length. Reading it costs one host readback, and it is returned
        because it is the honest measure of how much the geometry had to be changed.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`robust_laplacian`][ordito.laplacian.robust_laplacian]
    [`face_edge_lengths`][ordito.edges.face_edge_lengths]
    [`cotmatrix_entries_intrinsic`][ordito.laplacian.cotmatrix_entries_intrinsic]
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return odt.empty_2d((0, 3), wp.float32, device=device), 0.0

    edge_lengths = face_edge_lengths(vertices, faces)

    # The margin scales with the *mean* edge length (matching intrinsic_delaunay's docstring and
    # this function's own), not the max: a single long edge on a graded mesh would otherwise inflate
    # delta far past what any degenerate face on the rest of the mesh actually needs.
    scale = float(reduce_mean(edge_lengths))
    slack = odt.empty_1d(n_faces, wp.float32, device=device)
    _launch.launch(
        kernel_laplacian.triangle_inequality_slack,
        dim=n_faces,
        inputs=[edge_lengths, wp.float32(epsilon * scale), slack],
        device=device,
    )
    delta = float(reduce_max(slack))
    if delta <= 0.0:
        return odt.as_array2d(edge_lengths, wp.float32), 0.0

    mollified = odt.empty_2d((n_faces, 3), wp.float32, device=device)
    _launch.map(kernel_laplacian.add_constant, edge_lengths, wp.float32(delta), out=mollified)
    return odt.as_array2d(mollified, wp.float32), delta


def connection_laplacian(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: odt.Array2dFloat | None = None,
    transport_angles: wp.array[wp.float32] | None = None,
    *,
    pattern: MeshOperatorPattern | None = None,
) -> odt.BsrMatrix[wp.mat22d]:
    """
    Vector (connection) Laplacian: the cotangent Laplacian for *tangent vector* fields.

    Same cotangent weights and same sparsity as [`cotmatrix`][ordito.laplacian.cotmatrix], but each
    scalar becomes a ``2 x 2`` block and each off-diagonal weight is multiplied by the rotation that
    re-expresses a tangent vector in the neighbouring vertex's frame
    ([`halfedge_transport_angles`][ordito.tangent_space.halfedge_transport_angles]). Without those
    rotations a difference between vectors at two vertices would subtract components measured from
    two unrelated reference directions.

    Assembled ``float64`` and **positive semi-definite** (positive diagonal) — the opposite sign to
    ``cotmatrix``'s igl convention — because its consumers feed it to a conjugate-gradient solve. It
    is symmetric, since transporting from ``i`` to ``j`` and back are inverse rotations.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    cot_entries
        ``(n_faces, 3)`` precomputed half-cotangent weights from
        [`cotmatrix_entries`][ordito.laplacian.cotmatrix_entries]. Must have
        ``shape[0] == n_faces``.
    transport_angles
        ``(3 * n_faces,)`` precomputed per-halfedge
        [`halfedge_transport_angles`][ordito.tangent_space.halfedge_transport_angles]. Must have
        length ``3 * n_faces``.

        There is deliberately no ``frames`` argument. The gauge is fixed by the one-ring
        flattening — angles are measured from each vertex's first outgoing halfedge, the same
        convention [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames] uses to
        pick ``basis_x`` — so a caller-supplied frame cannot change these angles, and solutions are
        already consistent with the frames that convention produces.
    pattern
        Optional sparsity from [`mesh_operator_pattern`][ordito.laplacian.mesh_operator_pattern]
        (``operator="cotmatrix"``, the same pattern ``cotmatrix`` takes) built on these same
        ``faces``. Only the values are then computed.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` matrix of ``wp.mat22d`` blocks on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``cot_entries``, ``transport_angles`` and ``pattern`` are not
        all on one device.
    ValueError
        If ``cot_entries`` or ``transport_angles`` is given and does not match ``faces``' triangle
        count, or ``pattern`` was built for another operator or vertex count.

    See Also
    --------
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`halfedge_transport_angles`][ordito.tangent_space.halfedge_transport_angles]
    [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors]
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        cot_entries=cot_entries,
        transport_angles=transport_angles,
        pattern=None if pattern is None else pattern.offsets,
    )
    n_vertices = vertices.size
    n_faces = faces.size // 3
    device = vertices.device
    _check_pattern(pattern, "cotmatrix", n_vertices)
    if n_faces == 0:
        return od.array.empty_square_bsr(n_vertices, wp.mat22d, device)

    if cot_entries is None:
        cot_entries = cotmatrix_entries(vertices, faces, dtype=wp.float64)
    elif int(cot_entries.shape[0]) != n_faces:
        raise ValueError(
            f"cot_entries must have shape (n_faces, 3) = ({n_faces}, 3), "
            f"got {(int(cot_entries.shape[0]), int(cot_entries.shape[1]))}"
        )
    if transport_angles is None:
        transport_angles = halfedge_transport_angles(vertices, faces)
    elif transport_angles.size != 3 * n_faces:
        raise ValueError(
            f"transport_angles must have length 3 * n_faces = {3 * n_faces}, "
            f"got {transport_angles.size}"
        )

    if pattern is None:
        pattern = mesh_operator_pattern(faces, n_vertices)
    _, _, offsets, columns, run_start, keys, count, order = pattern
    values = _launch.empty(columns.size, dtype=wp.mat22d, device=device)
    _launch.launch(
        kernel_laplacian.CONNECTION_LAPLACIAN_ROWS[cot_entries.dtype],
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
            values,
        ],
        device=device,
    )
    return od.array.bsr_from_csr(n_vertices, n_vertices, offsets, columns, values)


class MeshOperatorPattern(NamedTuple):
    """
    The sparsity of a vertex operator on one face buffer, with each entry's contributing corners.

    Built by [`mesh_operator_pattern`][ordito.laplacian.mesh_operator_pattern] and taken as
    ``pattern=`` by [`cotmatrix`][ordito.laplacian.cotmatrix],
    [`connection_laplacian`][ordito.laplacian.connection_laplacian] and
    [`laplacian`][ordito.laplacian.laplacian], so a caller that assembles an operator repeatedly
    over fixed connectivity -- only the geometry moving -- builds the pattern once. Every matrix
    built from one pattern shares its ``offsets`` and ``columns`` arrays.
    """

    operator: str
    """The operator family the pattern serves: ``"cotmatrix"``, ``"laplacian_symmetric"`` or
    ``"laplacian_directed"``."""
    n_vertices: int
    """Row and column count."""
    offsets: wp.array[wp.int32]
    """``n_vertices + 1`` row bounds, total-terminated."""
    columns: wp.array[wp.int32]
    """Sorted column of each entry, per row; a capacity, of which ``offsets[-1]`` are written."""
    run_start: wp.array[wp.int32]
    """Each entry's first position in the sorted ``keys``."""
    keys: wp.array[wp.uint64]
    """The sorted entry keys; an entry's contributors are its run of equal keys."""
    n_keys: int
    """The number of sorted keys."""
    order: wp.array[wp.int32]
    """Each sorted key's corner slot ``3 * f + e`` (the corner opposite the edge), or halfedge."""


# ``mesh_operator_pattern``'s menu, mapped to ``_mesh_operator_pattern``'s build switches.
_PATTERN_BUILDS: dict[str, tuple[bool, Literal["self", "referenced", "all"]]] = {
    "cotmatrix": (False, "referenced"),
    "laplacian_symmetric": (False, "self"),
    "laplacian_directed": (True, "self"),
}


def mesh_operator_pattern(
    faces: wp.array[wp.int32],
    n_vertices: int,
    *,
    operator: Literal["cotmatrix", "laplacian_symmetric", "laplacian_directed"] = "cotmatrix",
) -> MeshOperatorPattern:
    """
    Sparsity of a vertex operator on ``faces``, reusable while the geometry changes.

    Every entry of the operators [`cotmatrix`][ordito.laplacian.cotmatrix],
    [`connection_laplacian`][ordito.laplacian.connection_laplacian] and
    [`laplacian`][ordito.laplacian.laplacian] build is determined by the connectivity alone, so
    a caller rebuilding one of them over fixed ``faces`` -- a smoothing flow re-linearising on the
    moving surface, or two operators over one mesh -- passes this as their ``pattern=`` and pays
    only for the values.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    n_vertices
        Vertex count; the operator is ``(n_vertices, n_vertices)``.
    operator
        Which operator the pattern serves. ``"cotmatrix"`` (default) serves ``cotmatrix`` and
        ``connection_laplacian`` -- every edge both ways plus every referenced vertex's diagonal.
        ``"laplacian_symmetric"`` and ``"laplacian_directed"`` serve ``laplacian`` with
        ``symmetric=True`` and ``False``: every edge both ways, or one entry per halfedge.

    Returns
    -------
    MeshOperatorPattern
        The pattern, on ``faces.device``.

    Raises
    ------
    ValueError
        If ``operator`` is not one of the three names above.

    See Also
    --------
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`laplacian`][ordito.laplacian.laplacian]
    """
    if operator not in _PATTERN_BUILDS:
        raise ValueError(f"operator must be one of {list(_PATTERN_BUILDS)}, got {operator!r}")
    halfedges, diagonal = _PATTERN_BUILDS[operator]
    offsets, columns, run_start, keys, count, order = _mesh_operator_pattern(
        faces, n_vertices, halfedges=halfedges, diagonal=diagonal
    )
    return MeshOperatorPattern(
        operator, n_vertices, offsets, columns, run_start, keys, count, order
    )


def _mesh_operator_pattern(
    faces: wp.array[wp.int32],
    n_vertices: int,
    *,
    halfedges: bool = False,
    diagonal: Literal["self", "referenced", "all"] = "referenced",
) -> tuple[
    wp.array[wp.int32],
    wp.array[wp.int32],
    wp.array[wp.int32],
    wp.array[wp.uint64],
    int,
    wp.array[wp.int32],
]:
    """
    Build a vertex operator's sparsity on ``faces``, with each entry's contributing corners.

    The pattern is every edge both ways plus every referenced vertex's diagonal, built straight
    from the faces with no triplets. Returns ``(offsets, columns, run_start, keys, count, order)``:
    an off-diagonal entry ``k``'s contributors are the corner slots ``order[p]`` (``3 * f + e``,
    the corner opposite the edge) over the run of equal sorted ``keys[:count]`` that starts at
    ``run_start[k]``, in face order; a diagonal's ``run_start`` is not meaningful. ``columns`` is a
    capacity of ``6 * n_faces + n_vertices``, of which ``offsets[-1]`` are entries.

    ``halfedges`` builds one entry per halfedge instead (``i -> j`` only, self-edges included; no
    diagonal). ``diagonal`` chooses which rows carry a diagonal: only those of a degenerate face's
    self-edge, those of vertices a face references (an unreferenced vertex's row stays empty), or
    all. A self-edge always lands on its vertex's diagonal entry, whose value is the operator's.

    Two builds give the identical pattern. Below ``_UNDIRECTED_PATTERN_FROM_FACES`` one sort of
    ``6 * n_faces`` directed keys plus the diagonal slots is cheapest; above it one sort of the
    ``3 * n_faces`` undirected keys plus a 32-bit sort of the unique edges does less sorting in less
    memory, for three more launches.
    """
    device = faces.device
    n_faces = faces.size // 3
    if halfedges:
        count = 3 * n_faces
        keys, order = od.array.csr_key_buffers(count, device)
        _launch.launch(
            kernel_laplacian.mesh_halfedge_keys,
            dim=count,
            inputs=[faces, wp.int32(n_vertices), keys, order],
            device=device,
        )
        offsets, columns, starts = od.array.csr_from_keys(
            keys, order, count, n_vertices, n_vertices
        )
        return offsets, columns, starts, keys, count, order
    if n_faces < _UNDIRECTED_PATTERN_FROM_FACES or n_vertices == 0:
        # A self-edge's own keys are its diagonal entry, so ``"self"`` needs no diagonal slots.
        tail = 0 if diagonal == "self" else n_vertices
        count = 6 * n_faces + tail
        keys, order = od.array.csr_key_buffers(count, device)
        mode = {"self": 0, "referenced": 1, "all": 2}[diagonal]
        if mode == 1 and n_vertices > 0:
            # The diagonal slots start as the sentinel: a vertex no face references stays out of
            # the pattern, so its row is empty, as ``cotmatrix``'s callers rely on.
            _launch.fill_(keys[6 * n_faces : count], n_vertices * n_vertices)
        _launch.launch(
            kernel_laplacian.mesh_operator_keys,
            dim=max(n_faces, n_vertices) if mode == 2 else n_faces,
            inputs=[faces, wp.int32(n_vertices), wp.int32(mode), keys, order],
            device=device,
        )
        offsets, columns, starts = od.array.csr_from_keys(
            keys, order, count, n_vertices, n_vertices
        )
        return offsets, columns, starts, keys, count, order

    count = 3 * n_faces
    sentinel = n_vertices * n_vertices
    keys, order = od.array.csr_key_buffers(count, device)
    # Rows: per-vertex upper-half count, lower-half count, referenced flag; then their scan.
    tallies = _launch.zeros((3, n_vertices), dtype=wp.int32, device=device)
    upper, lower, referenced = tallies[0], tallies[1], tallies[2]
    mark = {"self": 2, "referenced": 1, "all": 3}[diagonal]
    _launch.launch(
        kernel_laplacian.mesh_edge_keys,
        dim=max(n_faces, n_vertices) if mark == 3 else n_faces,
        inputs=[faces, wp.int32(n_vertices), wp.int32(mark), keys, order, referenced],
        device=device,
    )
    _launch.radix_sort_pairs(keys, order, count=count, end_bit=max(1, sentinel.bit_length()))
    # Scratch in the first sort's upper halves, free once it has run: the edge flags in the
    # payload's, and the second sort's 32-bit keys in the key buffer's (``count`` 8-byte slots are
    # exactly ``2 * count`` 4-byte ones). ``keys`` holds the storage, so the alias cannot outlive
    # it.
    flags = order[count:]
    assert keys.ptr is not None  # a non-empty allocation
    second_keys = wp.array(
        ptr=keys.ptr + count * 8, dtype=wp.int32, shape=(2 * count,), device=device
    )
    second_order = _launch.empty(2 * count, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_laplacian.mesh_edge_runs,
        dim=count,
        inputs=[
            keys,
            wp.uint64(sentinel),
            wp.int32(n_vertices),
            flags,
            upper,
            lower,
            second_keys,
            second_order,
        ],
        device=device,
    )
    _launch.array_scan(flags, out_array=flags, inclusive=True)
    # One exclusive scan of the three tally rows taken as one array places every part of every row
    # (``kernels/laplacian.mesh_row_start``); ``mesh_place_columns`` writes the offsets from it.
    before = _launch.empty((3, n_vertices), dtype=wp.int32, device=device)
    _launch.array_scan(tallies.flatten(), out_array=before.flatten(), inclusive=False)
    _launch.radix_sort_pairs(
        second_keys, second_order, count=count, end_bit=max(1, n_vertices.bit_length())
    )
    capacity = 2 * count + n_vertices
    offsets = _launch.empty(n_vertices + 1, dtype=wp.int32, device=device)
    columns = _launch.empty(capacity, dtype=wp.int32, device=device)
    run_start = _launch.empty(capacity, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_laplacian.mesh_place_columns,
        dim=max(count, n_vertices),
        inputs=[
            keys,
            flags,
            wp.uint64(sentinel),
            second_keys,
            second_order,
            wp.int32(count),
            wp.int32(n_vertices),
            tallies,
            before,
        ],
        outputs=[offsets, columns, run_start],
        device=device,
    )
    return offsets, columns, run_start, keys, count, order


def _check_pattern(pattern: MeshOperatorPattern | None, operator: str, n_vertices: int) -> None:
    """Raise if a caller's ``pattern`` was not built for ``operator`` over ``n_vertices``."""
    if pattern is None:
        return
    if pattern.operator != operator or pattern.n_vertices != n_vertices:
        raise ValueError(
            f"pattern was built for operator={pattern.operator!r} over {pattern.n_vertices} "
            f"vertices; this operator needs operator={operator!r} over {n_vertices}"
        )


def laplacian_entries(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    equal_weight: bool = True,
    symmetric: bool | None = None,
    dtype: type[odt.Block] = wp.float32,
    edges: odt.Array2dInt32 | None = None,
    *,
    validate: bool = True,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[odt.Block]]:
    """
    Per-edge weight triplets for the 1-ring Laplacian, before assembly.

    The analogue of [`cotmatrix_entries`][ordito.laplacian.cotmatrix_entries] for the umbrella
    operator, assembled and row-normalized by [`laplacian`][ordito.laplacian.laplacian].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    equal_weight
        If ``True`` every edge weight is ``1`` (uniform umbrella weights). If ``False`` the
        weight is inverse edge length ``1 / (‖vi - vj‖ + 1e-12)``.
    symmetric
        Adjacency shape. If ``False`` one triplet per directed triangle edge is emitted (trimesh's
        ``mesh.edges``); on meshes with a boundary this is asymmetric. If ``True`` each unique
        undirected edge emits both directed pairs (trimesh's ``vertex_neighbors``), giving a
        symmetric adjacency. When ``None`` (default) the trimesh convention is used:
        ``symmetric = not equal_weight``.
    dtype
        Scalar type of the returned ``vals``: ``wp.float32`` (default) or ``wp.float64``. The
        assembly kernel casts the float32 edge weights to ``dtype`` so the matrix built from these
        triplets is native float32/float64.
    edges
        ``(m, 2)`` precomputed unique undirected edges from
        [`edges_unique`][ordito.edges.edges_unique], used only by the ``symmetric`` branch --
        which is the ``equal_weight=False`` default, so this is the keyword that matters for the
        geometry-weighted operator. When ``None`` the edge set is derived here.

        Worth passing whenever the caller already holds it, because deriving the edge set can
        dominate the whole call on a large mesh. Ignored by the directed branch, which reads
        ``faces`` alone.

        Trusted to already be **unique and undirected**, the way ``edges_unique`` guarantees --
        that invariant is not, and cannot cheaply be, checked (verifying it would mean redoing the
        sort-and-deduplicate pass this argument exists to let a caller skip). A duplicated or
        non-deduplicated array silently doubles or miscounts the affected edges' weights rather
        than raising. ``validate`` below only guards the cheaper, memory-unsafe half of this
        precondition (an out-of-range vertex index), not this one.
    validate
        When ``True`` (default) and a caller-supplied ``edges`` is used (the ``symmetric`` branch
        with ``edges`` not ``None``), check that every index in it falls inside
        ``[0, n_vertices)`` before launching -- a stale ``edges`` array (e.g. from before a
        decimation pass changed the vertex count) otherwise drives the triplet kernel to read
        ``vertices`` out of bounds with no exception (§12.1's memory-safety class). Pass ``False``
        only when ``edges`` is known correct by construction, as this module's own
        [`laplacian`][ordito.laplacian.laplacian] does when it just derived ``edges`` itself.

    Returns
    -------
    tuple of wp.array
        ``(m,)`` triplets ``(rows, cols, vals)`` on ``faces.device``, ``m == 3 * n_faces`` for the
        directed case and ``m == 2 * n_unique_edges`` for the symmetric case.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``edges`` are not all on one device.
    ValueError
        If ``validate`` and a caller-supplied ``edges`` references a vertex index outside
        ``[0, n_vertices)``.

    See Also
    --------
    [`laplacian`][ordito.laplacian.laplacian]
    [`cotmatrix_entries`][ordito.laplacian.cotmatrix_entries]
    [`edges_unique`][ordito.edges.edges_unique]
    """
    require_same_device(vertices=vertices, faces=faces, edges=edges)
    if symmetric is None:
        symmetric = not equal_weight
    device = faces.device
    equal_weight_flag = wp.int32(1 if equal_weight else 0)
    if not symmetric:
        # One directed triplet per triangle edge, matching trimesh's ``edges_to_coo(mesh.edges)``.
        edges = faces_to_edges(faces)
        m = int(edges.shape[0])
        rows, cols, vals = od.array.triplet_buffers(m, dtype, device)
        if m > 0:
            _launch.launch(
                kernel_laplacian.LAPLACIAN_TRIPLETS_DIRECTED[dtype],
                dim=m,
                inputs=[edges, vertices, equal_weight_flag, rows, cols, vals],
                device=device,
            )
        return rows, cols, vals
    # Both directed pairs of each unique undirected edge, matching trimesh's ``vertex_neighbors``
    # (every neighbor counted once).
    if edges is None:
        edges, _ = edges_unique(faces, n_vertices=vertices.size, validate=False)
    elif validate and int(edges.shape[0]) > 0:
        n_vertices = vertices.size
        lowest, highest = od.reduce.minmax(edges)
        if lowest < 0 or highest >= n_vertices:
            raise ValueError(
                f"laplacian_entries: edges must reference vertex indices in [0, {n_vertices}), "
                f"got a range of [{lowest}, {highest}]"
            )
    m_unique = int(edges.shape[0])
    rows, cols, vals = od.array.triplet_buffers(2 * m_unique, dtype, device)
    if m_unique > 0:
        _launch.launch(
            kernel_laplacian.LAPLACIAN_TRIPLETS_SYMMETRIC[dtype],
            dim=m_unique,
            inputs=[edges, vertices, equal_weight_flag, rows, cols, vals],
            device=device,
        )
    return rows, cols, vals


def laplacian(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    equal_weight: bool = True,
    symmetric: bool | None = None,
    dtype: type[odt.Block] = wp.float32,
    edges: odt.Array2dInt32 | None = None,
    *,
    validate: bool = True,
    pattern: MeshOperatorPattern | None = None,
) -> odt.BsrMatrix[odt.Block]:
    """
    Row-normalized 1-ring averaging operator (uniform / umbrella Laplacian).

    Builds the sparse ``(n_vertices, n_vertices)`` matrix whose row ``i`` holds the weights of
    the neighbors of vertex ``i``, normalized so each row sums to ``1``. Applying it to vertex
    positions replaces each vertex by the weighted mean of its 1-ring, matching
    [`trimesh.smoothing.laplacian_calculation`][] with ``equal_weight``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    equal_weight
        If ``True`` all neighbors are weighted equally (``1 / degree``). If ``False`` neighbors
        are weighted by inverse edge length before normalization.
    symmetric
        Adjacency shape (see [`laplacian_entries`][ordito.laplacian.laplacian_entries]). If
        ``False`` the directed ``mesh.edges`` adjacency is used (asymmetric on boundaries); if
        ``True`` the symmetric ``vertex_neighbors`` adjacency is used. When ``None`` (default)
        the trimesh convention ``symmetric = not equal_weight`` is used, so the default matches
        [`trimesh.smoothing.laplacian_calculation`][] for both weightings. The two choices differ
        only on meshes with an open boundary.
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``. Use
        ``wp.float64`` when the operator feeds a linear-system solve; the matrix is built and
        row-normalized natively in the requested precision, in one build.
    edges
        ``(m, 2)`` precomputed unique undirected edges, forwarded to
        [`laplacian_entries`][ordito.laplacian.laplacian_entries], which assembles the operator
        from them as triplets. Ignored when the adjacency is directed. Without ``edges`` the
        operator's pattern is built straight from ``faces``, which is usually cheaper.
    validate
        Forwarded to [`laplacian_entries`][ordito.laplacian.laplacian_entries]: when ``True``
        (default) and ``edges`` is given, check its indices fall in ``[0, n_vertices)`` before
        launching. Pass ``False`` only when ``edges`` is known correct by construction.
    pattern
        Optional sparsity from [`mesh_operator_pattern`][ordito.laplacian.mesh_operator_pattern]
        built on these same ``faces`` -- ``operator="laplacian_symmetric"`` or
        ``"laplacian_directed"``, matching ``symmetric`` -- for a caller that assembles the
        operator repeatedly over fixed connectivity, as
        [`filter_taubin`][ordito.smoothing.filter_taubin] does at ``recompute=True``. Only the
        weights are then computed. Not combined with ``edges``.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` square row-stochastic matrix in 1x1-block BSR form on
        ``vertices.device``. Isolated vertices (empty rows) map to themselves under
        [`filter_laplacian`][ordito.smoothing.filter_laplacian] et al.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``edges`` and ``pattern`` are not all on one device.
    ValueError
        If ``validate`` and ``edges`` references a vertex index outside ``[0, n_vertices)``; if
        both ``edges`` and ``pattern`` are given; or if ``pattern`` was built for another adjacency
        or vertex count.

    See Also
    --------
    [`laplacian_entries`][ordito.laplacian.laplacian_entries]
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`edges_unique`][ordito.edges.edges_unique]
    [`trimesh.smoothing.laplacian_calculation`][]
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        edges=edges,
        pattern=None if pattern is None else pattern.offsets,
    )
    n_vertices = vertices.size
    device = vertices.device
    if symmetric is None:
        symmetric = not equal_weight
    if pattern is not None and edges is not None:
        raise ValueError("laplacian: pass edges or pattern, not both")
    operator = "laplacian_symmetric" if symmetric else "laplacian_directed"
    _check_pattern(pattern, operator, n_vertices)
    if edges is None:
        # The adjacency follows from the faces: its pattern is built straight from them and one row
        # kernel writes the weights and normalizes. Directed is one entry per halfedge; symmetric
        # both directions of every edge, plus a degenerate face's self-edge as a diagonal entry.
        if faces.size == 0:
            return od.array.empty_square_bsr(n_vertices, dtype, device)
        if pattern is None:
            pattern = mesh_operator_pattern(faces, n_vertices, operator=operator)
        _, _, offsets, columns, run_start, keys, count, _ = pattern
        values = _launch.empty(columns.size, dtype=dtype, device=device)
        _launch.launch(
            kernel_laplacian.LAPLACIAN_ROWS[dtype],
            dim=n_vertices,
            inputs=[
                offsets,
                columns,
                run_start,
                keys,
                wp.int32(count),
                vertices,
                wp.int32(equal_weight),
                wp.int32(symmetric),
                values,
            ],
            device=device,
        )
        return od.array.bsr_from_csr(n_vertices, n_vertices, offsets, columns, values)
    rows, cols, vals = laplacian_entries(
        vertices,
        faces,
        equal_weight=equal_weight,
        symmetric=symmetric,
        dtype=dtype,
        edges=edges,
        validate=validate,
    )
    matrix = od.array.csr_from_triplets(n_vertices, n_vertices, rows, cols, vals)
    if n_vertices > 0 and matrix.nnz > 0:
        _launch.launch(
            kernel_laplacian.ROW_NORMALIZE[matrix.values.dtype],
            dim=n_vertices,
            inputs=[matrix.offsets, matrix.values],
            device=device,
        )
    return matrix


def graph_laplacian(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], dtype: type[odt.Block] = wp.float32
) -> odt.BsrMatrix[odt.Block]:
    """
    Combinatorial (graph) Laplacian ``L = A - diag(deg)`` from mesh connectivity.

    Builds the sparse ``(n_vertices, n_vertices)`` matrix with unit off-diagonal weights on every
    undirected edge and the negated vertex degree on the diagonal, ignoring geometry. Mirrors
    libigl's uniform-weight ``igl::harmonic`` variant (``L = A - diag(rowsum(A))`` from
    ``igl::adjacency_matrix``). Diagonal entries are **negative** (each row sums to zero), so ``-L``
    is positive semi-definite — the same sign convention as
    [`cotmatrix`][ordito.laplacian.cotmatrix]. This is the operator behind the Tutte embedding
    [`tutte`][ordito.parametrization.tutte], whose interior block ``-L_uu = diag(deg) - A`` is a
    diagonally dominant M-matrix (hence positive definite for a well-posed Dirichlet problem).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Only the count and device are used; positions do
        not affect the uniform weights.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``. Use
        ``wp.float64`` for the higher-power Tutte operator in
        [`tutte`][ordito.parametrization.tutte].

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` square combinatorial Laplacian in 1x1-block BSR form on
        ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`laplacian`][ordito.laplacian.laplacian]
    [`tutte`][ordito.parametrization.tutte]
    """
    require_same_device(vertices=vertices, faces=faces)
    n_vertices = vertices.size
    n_faces = faces.size // 3
    device = vertices.device

    if n_faces == 0:
        return od.array.empty_square_bsr(n_vertices, dtype, device)

    # ``A - diag(deg)`` with ``A`` the unit-weight adjacency of ``igl::adjacency_matrix``: the mesh
    # pattern with every vertex's diagonal (an unreferenced vertex keeps a zero one), and one row
    # kernel writing the ones and minus their count.
    offsets, columns, _, _, _, _ = _mesh_operator_pattern(faces, n_vertices, diagonal="all")
    values = _launch.empty(columns.size, dtype=dtype, device=device)
    _launch.launch(
        kernel_laplacian.GRAPH_LAPLACIAN_ROWS[dtype],
        dim=n_vertices,
        inputs=[offsets, columns, values],
        device=device,
    )
    return od.array.bsr_from_csr(n_vertices, n_vertices, offsets, columns, values)


@overload
def mass_matrix_entries(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    dtype: type[wp.float32] = wp.float32,
    *,
    face_areas: wp.array[wp.float32] | None = None,
) -> wp.array[wp.float32]: ...
@overload
def mass_matrix_entries(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    dtype: type[wp.float64],
    *,
    face_areas: wp.array[wp.float32] | None = None,
) -> wp.array[wp.float64]: ...
def mass_matrix_entries(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    dtype: type = wp.float32,
    *,
    face_areas: wp.array[wp.float32] | None = None,
) -> wp.array[wp.float32] | wp.array[wp.float64]:
    """
    Per-vertex barycentric lumped mass: a third of each incident triangle's area.

    Each triangle donates a third of its area to each of its three vertices, so entry ``i`` is
    the summed one-third incident-face area at vertex ``i``. This is the
    ``MASSMATRIX_TYPE_BARYCENTRIC`` lumping; it is the diagonal of
    [`mass_matrix`][ordito.laplacian.mass_matrix].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    dtype
        Scalar type of the returned diagonal: ``wp.float32`` (default) or ``wp.float64``. Request
        ``wp.float64`` to feed a float64 solve (e.g. geodesic heat method, implicit fairing)
        without a downstream recast.
    face_areas
        ``(n_faces,)`` triangle areas
        ([`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]), or ``None``. The
        lumping reads nothing else off ``vertices``, so passing them is the whole geometry pass this
        function would otherwise repeat --
        [`Trimesh.face_areas`][ordito.mesh.Trimesh.face_areas] has them cached.
        ``wp.float32`` or ``wp.float64``, converted to ``dtype`` as it is read.

    Returns
    -------
    wp.array[wp.float32] | wp.array[wp.float64]
        ``(n_vertices,)`` diagonal on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``face_areas`` are not all on one device.
    ValueError
        If ``face_areas`` is given and its length does not match ``faces``' triangle count.

    See Also
    --------
    [`mass_matrix`][ordito.laplacian.mass_matrix]
    [`Trimesh.mass_matrix_entries`][ordito.mesh.Trimesh.mass_matrix_entries]

    Notes
    -----
    This is the diagonal of ``igl::massmatrix`` under ``MASSMATRIX_TYPE_BARYCENTRIC``.

    Reproducible bit for bit on both devices when the areas are computed here or given in
    ``float32``: each vertex sums its faces' ``float32`` areas exactly in ``float64`` -- whatever
    order the device commits them in -- and is divided by three once. Given ``float64`` areas, no
    sum is exact and the result is reproducible only to rounding on CUDA.
    """
    require_same_device(vertices=vertices, faces=faces, face_areas=face_areas)
    n_vertices = vertices.size
    device = vertices.device
    n_faces = faces.size // 3
    if face_areas is not None and n_faces > 0 and face_areas.size != n_faces:
        raise ValueError(f"face_areas must have length n_faces={n_faces}, got {face_areas.size}")
    if n_faces == 0:
        return _launch.zeros(n_vertices, dtype=dtype, device=device)
    if face_areas is not None and face_areas.dtype != wp.float32:
        # ``float64`` areas have no exact ``float64`` sum: rounded thirds, committed atomically.
        mass = _launch.zeros(n_vertices, dtype=dtype, device=device)
        _launch.launch(
            kernel_scatter.SCATTER_FACE_THIRDS[face_areas.dtype, dtype],
            dim=n_faces,
            inputs=[faces, face_areas, dtype(3.0), mass],
            device=device,
        )
        return mass
    # ``float32`` areas summed exactly in ``float64`` -- the same bits whatever order the atomics
    # commit in -- then divided by three once, in the requested precision.
    sums = _launch.zeros(n_vertices, dtype=wp.float64, device=device)
    _launch.launch(
        kernel_scatter.scatter_face_areas_exact,
        dim=n_faces,
        inputs=[vertices, faces, face_areas, sums],
        device=device,
    )
    mass = sums if dtype == wp.float64 else _launch.empty(n_vertices, dtype=dtype, device=device)
    _launch.launch(
        kernel_scatter.SCALE_FACE_SUMS[dtype],
        dim=n_vertices,
        inputs=[sums, wp.float64(3.0), mass],
        device=device,
    )
    return mass


def mass_matrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    dtype: type = wp.float32,
    *,
    face_areas: wp.array[wp.float32] | None = None,
) -> odt.BsrMatrix[wp.float32]:
    """
    Diagonal barycentric lumped mass matrix of the mesh.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``. Use
        ``wp.float64`` when the mass matrix feeds a float64 linear-system solve.
    face_areas
        ``(n_faces,)`` precomputed triangle areas, or ``None``, forwarded to
        [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries].

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` square diagonal matrix in 1x1-block BSR form on
        ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``face_areas`` are not all on one device.

    See Also
    --------
    [`mass_matrix_entries`][ordito.laplacian.mass_matrix_entries]
    [`cotmatrix`][ordito.laplacian.cotmatrix]
    [`crouzeix_raviart_massmatrix`][ordito.energies.crouzeix_raviart_massmatrix]
        The edge-based sibling, in [`ordito.energies`][ordito.energies].

    Notes
    -----
    Matches ``igl::massmatrix`` under ``MASSMATRIX_TYPE_BARYCENTRIC``.
    """
    require_same_device(vertices=vertices, faces=faces, face_areas=face_areas)
    return odt.bsr_diag(
        diag=mass_matrix_entries(vertices, faces, dtype=dtype, face_areas=face_areas)
    )
