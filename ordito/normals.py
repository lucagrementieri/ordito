"""
Vertex normals under three weighting rules, and the accumulators beneath them.

A vertex has no normal of its own -- only the faces around it do -- so a vertex normal is always a
weighted average of incident face normals, and the weight is a modelling choice rather than a
detail. [`vertex_normals`][ordito.normals.vertex_normals] is the entry point, and its
``weighting=`` keyword *is* that choice: by face area, by incident angle (the one invariant to how a
neighbouring polygon happened to be triangulated), or by Nelson Max's sine-and-edge-length
reciprocals.

Two **accumulators** sit underneath it and are public in their own right:
[`weighted_vertex_normals`][ordito.normals.weighted_vertex_normals] takes a caller-supplied
per-corner weight table and [`mean_vertex_normals`][ordito.normals.mean_vertex_normals] is the
same thing with unit weights. They know no geometry -- they take no ``vertices`` at all, which is
why they keep an explicit ``n_vertices`` argument where ``vertex_normals`` derives it. That
asymmetry is the visible marker of which of the two layers you are in.
"""

from __future__ import annotations

from typing import Literal

import warp as wp

import ordito.typing as odt
from ordito import _launch
from ordito._common import accumulate_and_normalize, require_same_device
from ordito.kernels import normals as kernel_normals
from ordito.kernels import scatter as kernel_scatter

# The precision the vertex-normal accumulator runs at (setting off), and the key every scatter
# kernel is looked up by for ``_common.accumulate_and_normalize``. ``float64`` so that the order
# the scatter's atomics happen to pick cannot reach the float32 answer -- the measurement is at
# ``kernels.scatter.atomic_add_vec3``, and both kernels are generic over this, so moving it is a
# one-line change here plus a registration row in each kernel module.
_ACCUMULATOR_DTYPE = wp.float64


def mean_vertex_normals(
    n_vertices: int, faces: wp.array[wp.int32], face_normals: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    """
    Vertex normals as the (unnormalized) sum of incident face normals, then unit-length.

    For each vertex, face normals sharing that corner are accumulated in ``float64`` on
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
    [`weighted_vertex_normals`][ordito.normals.weighted_vertex_normals]
        The same accumulator with a per-corner weight table; this is that with unit weights.
    [`vertex_normals`][ordito.normals.vertex_normals]
        The geometry-aware entry point, which derives a weight table and calls one of these two.

    Examples
    --------
    Each corner of a cube averages its incident face normals with unit weight; a corner with two
    triangles in one side leans towards that side:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.box()
    face_normals, _areas = od.triangles.face_normals_and_areas(vertices, faces)
    normals = od.normals.mean_vertex_normals(vertices.shape[0], faces, face_normals)
    print(normals.shape)  # (8,)
    print(np.round(normals.numpy()[0], 3))  # [-0.816 -0.408 -0.408]
    ```
    """
    require_same_device(faces=faces, face_normals=face_normals)
    _require_face_rows(faces.size // 3, face_normals=face_normals)
    return accumulate_and_normalize(
        n_vertices,
        faces,
        kernel_scatter.SCATTER_SUM_VEC[_ACCUMULATOR_DTYPE],
        [face_normals, faces],
        exact_kernel=kernel_scatter.SCATTER_SUM_VEC_EXACT[_ACCUMULATOR_DTYPE],
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
    contributions are summed per vertex in ``float64`` on ``faces.device``, cast to
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
    [`mean_vertex_normals`][ordito.normals.mean_vertex_normals]
        This with unit weights.
    [`vertex_normals`][ordito.normals.vertex_normals]
        The geometry-aware entry point, which derives a weight table and calls this.

    Examples
    --------
    Weighting each corner by its interior angle makes every cube corner normal the diagonal,
    whatever the triangulation of the sides:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.box()
    face_normals, _areas = od.triangles.face_normals_and_areas(vertices, faces)
    angles = od.triangles.face_angles(vertices, faces)
    normals = od.normals.weighted_vertex_normals(vertices.shape[0], faces, face_normals, angles)
    print(np.round(normals.numpy()[0], 3))  # [-0.577 -0.577 -0.577]
    ```
    """
    require_same_device(faces=faces, face_normals=face_normals, face_weights=face_weights)
    _require_face_rows(faces.size // 3, face_normals=face_normals, face_weights=face_weights)
    return accumulate_and_normalize(
        n_vertices,
        faces,
        kernel_scatter.SCATTER_WEIGHTED_SUM_VEC[_ACCUMULATOR_DTYPE],
        [face_normals, faces, face_weights],
        exact_kernel=kernel_scatter.SCATTER_WEIGHTED_SUM_VEC_EXACT[_ACCUMULATOR_DTYPE],
    )


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
    [`weighted_vertex_normals`][ordito.normals.weighted_vertex_normals]
        The accumulator underneath: a caller-supplied per-corner weight table, no geometry.
    [`mean_vertex_normals`][ordito.normals.mean_vertex_normals]
        The same accumulator with unit weights.
    [`ordito.triangles.face_normals_and_areas`][ordito.triangles.face_normals_and_areas]
        Produces the ``face_normals`` / ``face_weights`` the ``"area"`` path derives.
    [`ordito.triangles.face_angles`][ordito.triangles.face_angles]
        Produces the ``"angle"`` path's weight table.

    Examples
    --------
    The weighting matters at a cube corner, whose three sides carry different numbers of triangles
    there; only ``"angle"`` returns the symmetric diagonal:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.box()
    by_area = od.normals.vertex_normals(vertices, faces)
    by_angle = od.normals.vertex_normals(vertices, faces, weighting="angle")
    print(by_area.shape)  # (8,)
    print(np.round(by_area.numpy()[0], 3))  # [-0.816 -0.408 -0.408]
    print(np.round(by_angle.numpy()[0], 3))  # [-0.577 -0.577 -0.577]
    ```
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
            return accumulate_and_normalize(
                n_vertices,
                faces,
                kernel_normals.SCATTER_MAX_WEIGHTED_NORMALS[_ACCUMULATOR_DTYPE],
                [vertices, faces],
            )
        _require_face_rows(n_faces, face_normals=face_normals)
        return accumulate_and_normalize(
            n_vertices,
            faces,
            kernel_normals.SCATTER_MAX_WEIGHTED_UNIT_NORMALS[_ACCUMULATOR_DTYPE],
            [vertices, faces, face_normals],
        )

    # With nothing supplied one kernel per weighting derives each face's normal and weights in
    # registers and scatters their products, so no per-face table is written. With one table or
    # both supplied, the kernel built for that combination reads what was passed and forms the
    # rest in the thread (a missing table launches as ``None``), bit for bit the table it would
    # otherwise have written first. "area" weights per *face* rather than per corner, so it is
    # the cheaper of the two.
    if face_normals is None and face_weights is None:
        derived = (
            kernel_normals.SCATTER_ANGLE_WEIGHTED_NORMALS
            if weighting == "angle"
            else kernel_normals.SCATTER_AREA_WEIGHTED_NORMALS
        )
        return accumulate_and_normalize(
            n_vertices, faces, derived[_ACCUMULATOR_DTYPE], [vertices, faces]
        )
    key = (face_normals is not None, face_weights is not None)
    if weighting == "angle":
        supplied = kernel_normals.SCATTER_SUPPLIED_ANGLE_WEIGHTED_NORMALS[key]
        if face_weights is not None:
            face_weights = odt.as_array2d(face_weights, wp.float32)
    else:
        supplied = kernel_normals.SCATTER_SUPPLIED_AREA_WEIGHTED_NORMALS[key]
    _require_face_rows(n_faces, face_normals=face_normals, face_weights=face_weights)
    return accumulate_and_normalize(
        n_vertices,
        faces,
        supplied[_ACCUMULATOR_DTYPE],
        [vertices, faces, face_normals, face_weights],
    )


def _require_face_rows(n_faces: int, **named: odt.ArrayNd | None) -> None:
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
        Per-face arrays, keyed by the public argument name they arrived under; ``None`` (a table
        not supplied) is skipped.

    Raises
    ------
    ValueError
        If any named array's leading dimension is not ``n_faces``.
    """
    for name, array in named.items():
        if array is None:
            continue
        rows = int(array.shape[0])
        if rows != n_faces:
            raise ValueError(
                f"{name} must have one row per triangle: got {rows} for {n_faces} faces"
            )
