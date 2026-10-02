"""
Per-vertex quantities: normals under three weighting rules, and the angle defect.

A vertex has no normal of its own -- only the faces around it do -- so a vertex normal is always a
weighted average of incident face normals, and the weight is a modelling choice rather than a
detail. [`vertex_normals`][ordito.vertices.vertex_normals] is the entry point, and its
``weighting=`` keyword *is* that choice: by face area, by incident angle (the one invariant to how a
neighbouring polygon happened to be triangulated), or by Nelson Max's sine-and-edge-length
reciprocals.

Two **accumulators** sit underneath it and are public in their own right:
[`weighted_vertex_normals`][ordito.vertices.weighted_vertex_normals] takes a caller-supplied
per-corner weight table and [`mean_vertex_normals`][ordito.vertices.mean_vertex_normals] is the
same thing with unit weights. They know no geometry -- they take no ``vertices`` at all, which is
why they keep an explicit ``n_vertices`` argument where ``vertex_normals`` derives it. That
asymmetry is the visible marker of which of the two layers you are in.

[`vertex_defects`][ordito.vertices.vertex_defects] is the odd one out and not a normal at all:
``2 * pi`` minus the incident angle sum, which is the intrinsic curvature concentrated at that
vertex and the quantity
[`discrete_gaussian_curvature`][ordito.curvature.discrete_gaussian_curvature] reports.
"""

from __future__ import annotations

from typing import Literal

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import require_same_device
from ordito.kernels import array as kernel_array
from ordito.kernels import predicates as kernel_predicates
from ordito.kernels import scatter as kernel_scatter
from ordito.kernels import vertices as kernel_vertices

# The precision the vertex-normal accumulator runs at, and the key both of
# ``_accumulate_and_normalize``'s launches look their kernel up by. ``float64`` so that the order
# the scatter's atomics happen to pick cannot reach the float32 answer -- the measurement is at
# ``kernels.scatter.atomic_add_vec3``, and both kernels are generic over this, so moving it is a
# one-line change here plus a registration row in each kernel module.
_ACCUMULATOR_DTYPE = wp.float64


def mean_vertex_normals(
    n_vertices: int, faces: wp.array[wp.int32], face_normals: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    """
    Vertex normals as the (unnormalized) sum of incident face normals, then unit-length.

    For each vertex, face normals sharing that corner are accumulated in ``float32`` on
    ``faces.device``, cast to ``wp.vec3``, and L2-normalized. Vertices not referenced by
    any face remain zero.

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        ``(3 * n_faces,)`` triangle indices, interpreted as ``(n_faces, 3)`` via
        ``reshape((-1, 3))`` (row-major ``[i0, i1, i2, …]`` flat layout is fine).
    face_normals
        ``(n_faces,)`` unit (or unnormalized) normals, one per triangle, aligned with the rows of
        ``faces``.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_vertices,)`` device array of unit normals where the accumulated vector was
        non-zero; otherwise the corresponding entry is zero.

    Raises
    ------
    ValueError
        If ``face_normals`` does not have one entry per triangle in ``faces``.
    RuntimeError
        If ``faces`` and ``face_normals`` are not all on one device.

    See Also
    --------
    [`weighted_vertex_normals`][ordito.vertices.weighted_vertex_normals]
        The same accumulator with a per-corner weight table; this is that with unit weights.
    [`vertex_normals`][ordito.vertices.vertex_normals]
        The geometry-aware entry point, which derives a weight table and calls one of these two.
    """
    require_same_device(faces=faces, face_normals=face_normals)
    _require_face_rows(faces.size // 3, face_normals=face_normals)
    return _accumulate_and_normalize(
        n_vertices, faces, kernel_scatter.SCATTER_SUM_VEC, [face_normals, faces.reshape((-1, 3))]
    )


def weighted_vertex_normals(
    n_vertices: int,
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    face_weights: odt.Array2dFloat32,
) -> wp.array[wp.vec3]:
    """
    Vertex normals from a weighted sum of incident face normals, then unit-length.

    Each face contributes its normal scaled by the per-corner weight in ``face_weights``;
    contributions are summed per vertex in ``float32`` on ``faces.device``, cast to
    ``wp.vec3``, and L2-normalized.

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        ``(3 * n_faces,)`` triangle indices, interpreted as ``(n_faces, 3)`` via
        ``reshape((-1, 3))``.
    face_normals
        ``(n_faces,)`` normals, one per triangle, aligned with ``faces``.
    face_weights
        ``(n_faces, 3)`` per-corner weights, with rows matching ``faces`` / ``face_normals``.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_vertices,)`` device array of unit normals where the accumulated vector was
        non-zero; otherwise the corresponding entry is zero.

    Raises
    ------
    ValueError
        If ``face_normals`` or ``face_weights`` does not have one row per triangle in ``faces``.
    RuntimeError
        If ``faces``, ``face_normals`` and ``face_weights`` are not all on one device.

    See Also
    --------
    [`mean_vertex_normals`][ordito.vertices.mean_vertex_normals]
        This with unit weights.
    [`vertex_normals`][ordito.vertices.vertex_normals]
        The geometry-aware entry point, which derives a weight table and calls this.
    """
    require_same_device(faces=faces, face_normals=face_normals, face_weights=face_weights)
    _require_face_rows(faces.size // 3, face_normals=face_normals, face_weights=face_weights)
    return _accumulate_and_normalize(
        n_vertices,
        faces,
        kernel_scatter.SCATTER_WEIGHTED_SUM_VEC,
        [face_normals, faces.reshape((-1, 3)), face_weights],
    )


def _accumulate_and_normalize(
    n_vertices: int,
    faces: wp.array[wp.int32],
    scatter_table: kernel_array.OverloadTable,
    inputs: list[odt.ArrayNd],
) -> wp.array[wp.vec3]:
    """
    Scatter per-face vectors onto their corners and unit-normalize the sums.

    The shared body of every normal in the module: the scatter kernel decides what each face
    contributes to each of its corners -- a supplied vector once or scaled by a per-corner weight,
    or, for ``vertex_normals``' derived paths, a weighted normal it computes from the geometry
    itself so that no per-face table is written just to be read back once.

    Accumulation is ``float64`` in an ``(n_vertices, 3)`` buffer even though both the input and the
    answer are ``float32``, which is what makes the result reproducible run to run on a CUDA device
    -- a float atomic's summation order is the scheduler's, and a ``float32`` accumulator turns
    that into about one ULP of movement per run. The reasoning and the measurements are at
    ``kernels.scatter.atomic_add_vec3``. The narrowing is a real kernel rather than the zero-copy
    ``wp.utils.array_cast`` reinterpretation a ``float32`` accumulator allowed, and it absorbs the
    normalization, so this costs one launch fewer than the pair it replaced.

    Parameters
    ----------
    n_vertices
        Output length.
    faces
        Flat ``wp.int32`` triangle index buffer; reshaped to ``(f, 3)`` for the scatter.
    scatter_table
        Overload table of a kernel launched one thread per face that takes ``(*inputs,
        out_sums)`` -- a ``kernels.scatter`` accumulator or one of ``kernels.vertices``' fused
        ones. All of them are generic over the accumulator's precision, so the table is keyed by
        it.
    inputs
        The kernel's arguments before the accumulator, in its own order.

    Returns
    -------
    wp.array[wp.vec3]
        Length-``n_vertices`` unit normals; zero where the accumulated vector was zero.

    Notes
    -----
    The launch ``dim`` is taken from ``faces``, not from any per-face input: every array here is
    indexed by the face, and ``faces`` is the only one of them whose length also bounds the
    *gather* the scatter kernel does. Callers guarantee the rest agree by calling
    ``_require_face_rows`` first.
    """
    device = faces.device
    n_faces = faces.size // 3
    sums = _launch.zeros((n_vertices, 3), dtype=_ACCUMULATOR_DTYPE, device=device)
    _launch.launch(
        scatter_table[_ACCUMULATOR_DTYPE], dim=n_faces, inputs=[*inputs, sums], device=device
    )
    vec_normals = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_vertices.NORMALIZE_ACCUMULATED_ROWS[_ACCUMULATOR_DTYPE],
        dim=n_vertices,
        inputs=[sums, vec_normals],
        device=device,
    )
    return vec_normals


def vertex_normals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    weighting: Literal["area", "angle", "mwselr"] = "area",
    face_normals: wp.array[wp.vec3] | None = None,
    face_weights: wp.array[wp.float32] | odt.Array2dFloat32 | None = None,
) -> wp.array[wp.vec3]:
    """
    Per-vertex unit normals, as a weighted average of the incident face normals.

    A vertex has no normal of its own -- only the faces around it do -- so the weight is the whole
    content of the choice, and ``weighting`` selects it. Contributions are summed per vertex and
    L2-normalized, so any constant factor in the weights cancels.

    Which weighting to want is a measurable question, not a preference: one reference library
    ships both conventions as separate entry points, and its plain per-vertex normals reproduce
    ``"area"`` to **1.19e-07** while its pseudo-normals reproduce ``"angle"`` to **1.19e-07** --
    each sitting **6.8e-03** from the other's partner, so the two are distinguishable well above
    float32 noise. Reach for ``"angle"`` when the answer must not depend on
    how a neighbouring polygon happened to be triangulated -- it is the only one of the three that
    is invariant to that -- and for ``"area"`` otherwise, since it is what libigl and most of the
    field default to.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. The output length is ``vertices.shape[0]``.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    weighting
        Which average to take.

        - ``"area"`` (default) weights each face by its area, matching libigl's default
          ``per_vertex_normals`` (up to the constant ``2`` from ``doublearea``, which normalization
          cancels).
        - ``"angle"`` weights each face by its interior angle at the corner vertex -- the
          Thuerrner/Wuethrich recipe (*Computing Vertex Normals from Polygonal Facets*, Journal of
          Graphics Tools 3:1, 43-46, 1998) and
          ``igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE``. The triangulation-invariant one.
        - ``"mwselr"`` weights each corner by ``sin(angle) / (||E_i|| ||E_{i+1}||)``, which reduces
          to ``(e1 x e2) / (||e1||^2 ||e2||^2)`` for the two outgoing edges -- Nelson Max's Mean
          Weighted by Sine and Edge Length Reciprocals, optimized for smooth surface reconstruction
          (Jin et al., 2005).
    face_normals
        ``(n_faces,)`` precomputed face normals, to skip deriving them. Under ``"mwselr"``, passing
        them asserts they are *unit* length; omitting them lets the unnormalized cross product be
        used instead, which is cheaper and gives the same answer after normalization.
    face_weights
        ``(n_faces,)`` or ``(n_faces, 3)`` precomputed weights, to skip deriving them. The
        expected shape follows ``weighting``: ``(n_faces,)`` per-face areas for ``"area"``, and
        ``(n_faces, 3)`` per-corner angles for ``"angle"``. Not accepted under ``"mwselr"``, whose
        weights are a joint function of the corner positions rather than a table a caller would
        hold.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_vertices,)`` unit normals on ``vertices.device``, zero wherever the accumulated
        vector was zero (an unreferenced vertex, or a cancelling fan).

    Raises
    ------
    ValueError
        If ``weighting`` is not one of the three names, if ``face_weights`` is passed with
        ``weighting="mwselr"``, or if a supplied ``face_normals`` / ``face_weights`` does not have
        one row per triangle in ``faces``.
    RuntimeError
        If ``vertices``, ``faces``, ``face_normals`` and ``face_weights`` are not all on one
        device.

    See Also
    --------
    [`weighted_vertex_normals`][ordito.vertices.weighted_vertex_normals]
        The accumulator underneath: a caller-supplied per-corner weight table, no geometry.
    [`mean_vertex_normals`][ordito.vertices.mean_vertex_normals]
        The same accumulator with unit weights.
    [`ordito.triangles.face_normals_and_areas`][ordito.triangles.face_normals_and_areas]
        Produces the ``face_normals`` / ``face_weights`` the ``"area"`` path derives.
    [`ordito.triangles.face_angles`][ordito.triangles.face_angles]
        Produces the ``"angle"`` path's weight table.
    """
    require_same_device(
        vertices=vertices, faces=faces, face_normals=face_normals, face_weights=face_weights
    )
    if weighting not in ("area", "angle", "mwselr"):
        raise ValueError(f'weighting must be "area", "angle" or "mwselr", got {weighting!r}')
    if weighting == "mwselr" and face_weights is not None:
        raise ValueError('face_weights is not accepted with weighting="mwselr"')

    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3

    if weighting == "mwselr":
        if n_faces == 0:
            return _launch.zeros(n_vertices, dtype=wp.vec3, device=device)
        # A supplied ``face_normals`` is unit length; a derived one is the raw cross product, and
        # the weight depends on which, because the sine it divides out is already in the cross
        # product's magnitude. Either way one kernel forms the weights in registers and scatters
        # their products, so no normal or weight table is written.
        if face_normals is None:
            return _accumulate_and_normalize(
                n_vertices, faces, kernel_vertices.SCATTER_MAX_WEIGHTED_NORMALS, [vertices, faces]
            )
        _require_face_rows(n_faces, face_normals=face_normals)
        return _accumulate_and_normalize(
            n_vertices,
            faces,
            kernel_vertices.SCATTER_MAX_WEIGHTED_UNIT_NORMALS,
            [vertices, faces, face_normals],
        )

    if weighting == "angle":
        if face_normals is None and face_weights is None:
            # Nothing supplied: one kernel derives each face's normal and corner angles in
            # registers and scatters their products, so neither per-face table is written.
            return _accumulate_and_normalize(
                n_vertices, faces, kernel_vertices.SCATTER_ANGLE_WEIGHTED_NORMALS, [vertices, faces]
            )
        if face_normals is None:
            face_normals, _areas = od.triangles.face_normals_and_areas(vertices, faces)
        angles = od.triangles.face_angles(vertices, faces) if face_weights is None else face_weights
        return weighted_vertex_normals(
            n_vertices, faces, face_normals, odt.as_array2d(angles, wp.float32)
        )

    # "area": each face contributes its normal scaled by its area to all three corners -- the
    # weight is per *face* here rather than per corner, so this is the cheaper of the two paths.
    # With nothing supplied one kernel derives and scatters the product in registers; otherwise
    # the missing table is derived and the product is still formed inside the scatter.
    if face_normals is None and face_weights is None:
        return _accumulate_and_normalize(
            n_vertices, faces, kernel_vertices.SCATTER_AREA_WEIGHTED_NORMALS, [vertices, faces]
        )
    if face_normals is None or face_weights is None:
        computed_normals, computed_areas = od.triangles.face_normals_and_areas(vertices, faces)
        if face_normals is None:
            face_normals = computed_normals
        if face_weights is None:
            face_weights = computed_areas
    _require_face_rows(n_faces, face_normals=face_normals, face_weights=face_weights)
    return _accumulate_and_normalize(
        n_vertices,
        faces,
        kernel_vertices.SCATTER_SCALED_NORMALS,
        [face_normals, face_weights, faces],
    )


def vertex_defects(
    n_vertices: int, faces: wp.array[wp.int32], face_angles: odt.Array2dFloat32
) -> wp.array[wp.float32]:
    """
    Discrete angle defect per vertex: ``2π`` minus the sum of incident corner angles.

    For each vertex, interior angles from every triangle corner that references that vertex
    are accumulated in ``float32`` on ``faces.device``, then subtracted from a full turn.
    This is the standard piecewise-linear angle defect (related to discrete Gaussian
    curvature via the Gauss—Bonnet viewpoint on triangle meshes).

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        ``(3 * n_faces,)`` triangle indices, interpreted as ``(n_faces, 3)`` via
        ``reshape((-1, 3))`` (row-major flat layout is fine).
    face_angles
        ``(n_faces, 3)`` interior angles at the three corners of each triangle, with rows aligned
        with ``faces``.

    Returns
    -------
    wp.array[wp.float32]
        ``(n_vertices,)`` device array ``2π - Σ angles`` at each vertex. Vertices not
        referenced by any face have defect ``2π`` (empty angle sum).

    Raises
    ------
    ValueError
        If ``face_angles`` does not have one row per triangle in ``faces``.
    RuntimeError
        If ``faces`` and ``face_angles`` are not all on one device.

    See Also
    --------
    [`discrete_gaussian_curvature`][ordito.curvature.discrete_gaussian_curvature]
        The same quantity at a *scale*: the Cohen-Steiner/Morvan ball measure sums these defects
        over a ball of given radius, where this is the pointwise value at one vertex. The ball
        measure is what converges under refinement; the pointwise defect does not.
    [`face_angles`][ordito.triangles.face_angles]
        The angles this sums.
    """
    require_same_device(faces=faces, face_angles=face_angles)
    n_faces = faces.size // 3
    _require_face_rows(n_faces, face_angles=face_angles)
    angle_sum = _launch.zeros(n_vertices, dtype=wp.float32, device=faces.device)
    faces2d = faces.reshape((-1, 3))
    _launch.launch(
        kernel_scatter.SCATTER_SUM_SCALAR[face_angles.dtype],
        dim=n_faces,
        inputs=[face_angles, faces2d, angle_sum],
        device=faces.device,
    )
    # In place over the accumulator: ``angle_sum`` is scratch, and the operator spelling
    # ``TWO_PI - angle_sum`` would run the same wp.map into a second allocation.
    _launch.map(kernel_predicates.angle_defect, angle_sum, out=angle_sum)
    return angle_sum


def _require_face_rows(n_faces: int, **named: odt.ArrayNd) -> None:
    """
    Check that every named per-face table has one row per triangle.

    The scatter kernels underneath this module index ``faces``, ``face_normals`` and any weight
    table by the same face id, so a table of the wrong length is read out of range -- which on the
    CUDA device is an illegal access that kills the context and on the host is a heap read, never
    an exception. One comparison of a handful of ``.shape[0]`` values at the public boundary turns
    that into an ordinary Python error, for a cost unmeasurable next to the launch it precedes.

    Parameters
    ----------
    n_faces
        Triangle count the tables must match, derived from the caller's ``faces`` buffer.
    named
        Per-face arrays, keyed by the public argument name they arrived under.

    Raises
    ------
    ValueError
        If any named array's leading dimension is not ``n_faces``.
    """
    for name, array in named.items():
        rows = int(array.shape[0])
        if rows != n_faces:
            raise ValueError(
                f"{name} must have one row per triangle: got {rows} for {n_faces} faces"
            )
