"""
Per-vertex tangent spaces: orthonormal frames, halfedge polar angles and transport rotations.

Vector-valued surface algorithms (parallel transport, the vector heat method, log maps) need a 2D
coordinate system per vertex plus a rule for re-expressing a vector from one vertex's coordinates in
its neighbour's. Both come from *intrinsic flattening*: laying a vertex's incident triangles out in
the plane and rescaling the total angle to ``2 * pi`` (``pi`` at a boundary vertex), which turns the
counter-clockwise halfedge ring from [`ordito.halfedge`][ordito.halfedge] into polar coordinates
around the vertex.

Three quantities, in the order they build on each other:

1. [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames] — an extrinsic
   ``(basis_x, basis_y, normal)`` triple per vertex, for moving between 2D tangent coordinates and
   3D. This is what ``potpourri3d.MeshVectorHeatSolver.get_tangent_frames`` returns.
2. [`halfedge_tangent_angles`][ordito.tangent_space.halfedge_tangent_angles] — the polar angle of
   each outgoing halfedge, measured from that vertex's ``basis_x``.
3. [`halfedge_transport_angles`][ordito.tangent_space.halfedge_transport_angles] — the rotation
   that carries a tangent vector across an edge, the off-diagonal phase of the connection Laplacian.

Frames are only defined up to a rotation within the tangent plane, so they are *not* comparable
across libraries element-wise; the transport angles inherit that gauge freedom and are comparable
only through gauge-invariant combinations such as the holonomy around a face.
"""

from __future__ import annotations

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import require_same_device
from ordito.halfedge import halfedge_twins, require_matching_twins, vertex_one_rings
from ordito.kernels import tangent_space as kernel_tangent_space


def vertex_tangent_frames(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3] | None = None,
    rings: tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]]:
    """
    Orthonormal tangent frame at every vertex as ``(basis_x, basis_y, normal)``.

    ``normal`` is the angle-weighted vertex normal
    ([`vertex_normals`][ordito.vertices.vertex_normals] at ``weighting="angle"``);
    ``basis_x`` is the first halfedge of the vertex's counter-clockwise ring, projected into the
    tangent plane and normalized; ``basis_y = normal x basis_x`` completes a right-handed frame. A
    tangent vector ``(a, b)`` in these coordinates is ``a * basis_x + b * basis_y`` in world space.

    Because ``basis_x`` follows the ring's first halfedge, it is the direction that
    [`halfedge_tangent_angles`][ordito.tangent_space.halfedge_tangent_angles] assigns polar angle
    ``0``: the two functions describe one coordinate system, not two. Which halfedge that is depends
    on the face ordering, so a frame agrees with another library's only up to a rotation about the
    normal. A vertex with no incident faces has no tangent plane; it gets the fixed frame ``((1, 0,
    0), (0, 1, 0))`` so that downstream normalizations see unit vectors rather than NaN.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    normals
        ``(n_vertices,)`` precomputed unit vertex normals. When ``None``, the
        angle-weighted normals are computed here.
    rings
        ``(3 * n_faces,)``, ``(n_vertices + 1,)`` and ``(n_vertices,)`` precomputed
        ``(ring_halfedges, offsets, is_boundary)`` from
        [`vertex_one_rings`][ordito.halfedge.vertex_one_rings]. When ``None``, they are computed
        here.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]]
        ``(n_vertices,)`` arrays ``(basis_x, basis_y, normal)`` on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``normals`` and ``rings`` are not all on one device.

    See Also
    --------
    [`halfedge_tangent_angles`][ordito.tangent_space.halfedge_tangent_angles]
    [`vertex_normals`][ordito.vertices.vertex_normals] at ``weighting="angle"``
    [`vertex_one_rings`][ordito.halfedge.vertex_one_rings]
    """
    require_same_device(vertices=vertices, faces=faces, normals=normals, rings=rings)
    device = vertices.device
    n = vertices.size
    basis_x = _launch.empty(n, dtype=wp.vec3, device=device)
    basis_y = _launch.empty(n, dtype=wp.vec3, device=device)
    if normals is None:
        normals = od.vertices.vertex_normals(vertices, faces, weighting="angle")
    if n == 0:
        return basis_x, basis_y, normals

    if rings is None:
        rings = vertex_one_rings(faces, n_vertices=n)
    ring_halfedges, ring_offsets, _ = rings

    _launch.launch(
        kernel_tangent_space.vertex_tangent_frames,
        dim=n,
        inputs=[vertices, faces, normals, ring_offsets, ring_halfedges, basis_x, basis_y],
        device=device,
    )
    return basis_x, basis_y, normals


def face_tangent_frames(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], normals: wp.array[wp.vec3] | None = None
) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]]:
    """
    Orthonormal tangent frame on every face as ``(basis_x, basis_y, normal)``.

    ``basis_x`` is the face's first edge ``v1 - v0`` normalized, ``normal`` its unit face normal and
    ``basis_y = normal x basis_x``. A tangent vector ``(a, b)`` in these coordinates is
    ``a * basis_x + b * basis_y`` in world space.

    Unlike [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames], **this frame is
    not gauge-dependent**: the first edge of a face is a property of the face table, so any library
    using the same rule produces the identical frame rather than one rotated about the normal.
    ``igl.local_basis`` uses exactly this rule, which is why ``tests/test_tangent_space.py`` can
    compare all three vectors element-wise instead of through an invariant.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    normals
        ``(n_faces,)`` precomputed unit face normals, as returned by
        [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]. Computed here when
        ``None``.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]]
        ``(n_faces,)`` arrays ``(basis_x, basis_y, normal)`` on ``vertices.device``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``normals`` are not all on one device.

    See Also
    --------
    [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames]
    [`face_normals_and_areas`][ordito.triangles.face_normals_and_areas]
    ``igl.local_basis``
    """
    require_same_device(vertices=vertices, faces=faces, normals=normals)
    device = vertices.device
    n_faces = faces.size // 3
    basis_x = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    basis_y = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    if n_faces == 0:
        if normals is None:
            normals = _launch.empty(0, dtype=wp.vec3, device=device)
        return basis_x, basis_y, normals

    if normals is None:
        # Derived here rather than through ``triangles.face_normals_and_areas``: the frame needs one
        # cross product the producer already forms, and routing through it costs a launch, a round
        # trip of the normals through global memory and an areas buffer nothing below reads.
        normals = _launch.empty(n_faces, dtype=wp.vec3, device=device)
        _launch.launch(
            kernel_tangent_space.face_tangent_frames_and_normals,
            dim=n_faces,
            inputs=[vertices, faces, basis_x, basis_y, normals],
            device=device,
        )
        return basis_x, basis_y, normals

    _launch.launch(
        kernel_tangent_space.face_tangent_frames,
        dim=n_faces,
        inputs=[vertices, faces, normals, basis_x, basis_y],
        device=device,
    )
    return basis_x, basis_y, normals


def halfedge_tangent_angles(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_angles: odt.Array2dFloat32 | None = None,
    rings: tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]] | None = None,
) -> wp.array[wp.float32]:
    """
    Polar angle of every outgoing halfedge in its origin vertex's tangent plane.

    Walking a vertex's counter-clockwise ring, halfedge ``k`` sits at
    ``theta_k = s * sum(alpha_j for j < k)`` where ``alpha_j`` are the incident corner angles and
    ``s = 2 * pi / Theta`` scales the total angle ``Theta`` to a full turn — ``s = pi / Theta`` at a
    boundary vertex, whose fan covers a half-disk. The first halfedge of each ring therefore has
    angle ``0``, aligning this coordinate system with ``basis_x`` from
    [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    face_angles
        ``(n_faces, 3)`` precomputed corner angles from
        [`face_angles`][ordito.triangles.face_angles]. When ``None``, computed here.
    rings
        ``(3 * n_faces,)``, ``(n_vertices + 1,)`` and ``(n_vertices,)`` precomputed
        ``(ring_halfedges, offsets, is_boundary)`` from
        [`vertex_one_rings`][ordito.halfedge.vertex_one_rings]. When ``None``, computed here.

    Returns
    -------
    wp.array[wp.float32]
        ``(3 * n_faces,)`` angles in radians on ``vertices.device``, indexed by halfedge.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``face_angles`` and ``rings`` are not all on one device.

    See Also
    --------
    [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames]
    [`halfedge_transport_angles`][ordito.tangent_space.halfedge_transport_angles]
    [`vertex_one_rings`][ordito.halfedge.vertex_one_rings]
    """
    require_same_device(vertices=vertices, faces=faces, face_angles=face_angles, rings=rings)
    device = vertices.device
    n = vertices.size
    n_halfedges = faces.size // 3 * 3
    angles = _launch.zeros(n_halfedges, dtype=wp.float32, device=device)
    if n_halfedges == 0 or n == 0:
        return angles

    if face_angles is None:
        face_angles = od.triangles.face_angles(vertices, faces)
    if rings is None:
        rings = vertex_one_rings(faces, n_vertices=n)
    ring_halfedges, ring_offsets, is_boundary = rings

    _launch.launch(
        kernel_tangent_space.halfedge_tangent_angles,
        dim=n,
        inputs=[face_angles, ring_offsets, ring_halfedges, is_boundary, angles],
        device=device,
    )
    return angles


def halfedge_transport_angles(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32] | None = None,
    tangent_angles: wp.array[wp.float32] | None = None,
) -> wp.array[wp.float32]:
    """
    Rotation that carries a tangent vector across each halfedge, in ``(-pi, pi]``.

    For halfedge ``h`` from ``i`` to ``j``, a vector expressed in ``i``'s tangent basis is expressed
    in ``j``'s by rotating it through ``rho = theta_ji + pi - theta_ij``, where ``theta`` are the
    [`halfedge_tangent_angles`][ordito.tangent_space.halfedge_tangent_angles] of the halfedge and
    its twin: the two vertices see the shared edge along directions that differ by ``pi``, and the
    rest is the mismatch between their reference directions. These are the off-diagonal phases of
    the connection Laplacian, ``-w_ij * exp(i * rho_ij)``.

    A boundary edge exists as a halfedge in one direction only. At its destination vertex it is the
    fan-closing edge, whose rescaled polar angle is ``pi``, and that value stands in for the missing
    twin's angle.

    Because each ``theta`` is measured from its own vertex's reference direction, ``rho`` is
    gauge-dependent: rotating one vertex's frame shifts every incident ``rho``. Sums around a closed
    loop — the holonomy — are gauge-invariant and are what cross-library comparisons must use.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    twins
        ``(3 * n_faces,)`` precomputed [`halfedge_twins`][ordito.halfedge.halfedge_twins]. When
        ``None``, computed here.
    tangent_angles
        ``(3 * n_faces,)`` precomputed
        [`halfedge_tangent_angles`][ordito.tangent_space.halfedge_tangent_angles]. When ``None``,
        computed here.

    Returns
    -------
    wp.array[wp.float32]
        ``(3 * n_faces,)`` rotations in radians on ``vertices.device``, indexed by halfedge.

    Raises
    ------
    ValueError
        If ``twins`` is given and is not a twin table for ``faces``
        ([`require_matching_twins`][ordito.halfedge.require_matching_twins] states what that
        means, and checking it costs one launch and one readback).
    RuntimeError
        If ``vertices``, ``faces``, ``twins`` and ``tangent_angles`` are not all on one device.

    See Also
    --------
    [`halfedge_tangent_angles`][ordito.tangent_space.halfedge_tangent_angles]
    [`vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames]
    [`require_matching_twins`][ordito.halfedge.require_matching_twins]
    """
    require_same_device(vertices=vertices, faces=faces, twins=twins, tangent_angles=tangent_angles)
    require_matching_twins(faces, twins)
    device = vertices.device
    n_halfedges = faces.size // 3 * 3
    rho = _launch.empty(n_halfedges, dtype=wp.float32, device=device)
    if n_halfedges == 0:
        return rho

    n = vertices.size
    if twins is None:
        twins = halfedge_twins(faces, n_vertices=n)
    if tangent_angles is None:
        rings = vertex_one_rings(faces, twins=twins, n_vertices=n)
        tangent_angles = halfedge_tangent_angles(vertices, faces, rings=rings)

    _launch.launch(
        kernel_tangent_space.halfedge_transport_angles,
        dim=n_halfedges,
        inputs=[twins, tangent_angles, rho],
        device=device,
    )
    return rho
