"""
Parametric mesh generators: boxes, Platonic solids, surfaces of revolution and extrusions.

Every function returns a bare ``(vertices, faces)`` pair using triwarp's flat ``(3 * n_faces,)``
face layout, so results feed straight into the rest of the package or into
[`Trimesh`][triwarp.mesh.Trimesh] via ``tw.Trimesh(*tw.creation.box())``.

[`revolve`][triwarp.creation.revolve] is the engine behind most of the module:
[`uv_sphere`][triwarp.creation.uv_sphere], [`capsule`][triwarp.creation.capsule],
[`cylinder`][triwarp.creation.cylinder], [`cone`][triwarp.creation.cone],
[`annulus`][triwarp.creation.annulus] and [`torus`][triwarp.creation.torus] each build a small 2D
profile and sweep it around the Z axis. [`box`][triwarp.creation.box] and the four Platonic solids
([`tetrahedron`][triwarp.creation.tetrahedron], [`octahedron`][triwarp.creation.octahedron],
[`icosahedron`][triwarp.creation.icosahedron], [`dodecahedron`][triwarp.creation.dodecahedron]) are
constant tables, and [`icosphere`][triwarp.creation.icosphere] refines one of them in closed form.
[`grid`][triwarp.creation.grid] and
[`sphere_cap`][triwarp.creation.sphere_cap] are the two *open* primitives — a flat patch and a
curved one, each with exactly one boundary loop.

[`parametric_surface`][triwarp.creation.parametric_surface] and its three named siblings
([`super_ellipsoid`][triwarp.creation.super_ellipsoid],
[`super_toroid`][triwarp.creation.super_toroid],
[`random_hills`][triwarp.creation.random_hills]) sample an analytic map on an identified lattice,
and are the only builders here that produce a **non-orientable** surface, an odd Euler
characteristic, or a mesh whose scale is far from 1.

Each function carries the fixed host-side cost of a Warp wrapper call, so build a primitive once
and reuse it rather than calling these repeatedly in a loop.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp import _launch
from triwarp._device import read_scalar, require_same_device
from triwarp.constants import INT32_MAX, TILE_1D, TOLERANCE_MERGE
from triwarp.kernels import array as kernel_array
from triwarp.kernels import creation as kernel_creation
from triwarp.kernels import reduce as kernel_reduce

# Default number of pie wedges per full revolution, matching trimesh.
DEFAULT_SECTIONS = 32

# Absolute tolerance trimesh uses to decide that a revolution angle closes the loop.
_CLOSED_ANGLE_ATOL = 1e-10

# One full revolution, as the ``wp.float32`` the revolution kernels take. Named so the closed-form
# solids and ``revolve`` pass bit-identical spans: the angle divides into the slice fraction, so a
# difference here would move every ring vertex in the last bits.
_FULL_TURN = wp.float32(2.0 * math.pi)

_UNIT_X = np.array([1.0, 0.0, 0.0])
_UNIT_Y = np.array([0.0, 1.0, 0.0])
_UNIT_Z = np.array([0.0, 0.0, 1.0])

# Unit cube in [0, 1]^3 with outward-facing winding, from trimesh's baked
# ``resources/creation.json``. Kept verbatim so `box` is vertex-for-vertex identical to
# ``trimesh.creation.box``.
_BOX_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
_BOX_FACES = np.array(
    [
        [1, 3, 0],
        [4, 1, 0],
        [0, 3, 2],
        [2, 4, 0],
        [1, 7, 3],
        [5, 1, 4],
        [5, 7, 1],
        [3, 7, 2],
        [6, 4, 2],
        [2, 7, 6],
        [6, 5, 4],
        [7, 5, 6],
    ],
    dtype=np.int32,
).reshape(-1)

# Regular icosahedron on the unit sphere: the cyclic permutations of (+-1, +-phi, 0) normalized
# by sqrt(1 + phi^2). Also from trimesh's ``resources/creation.json``.
_ICOSAHEDRON_VERTICES = np.array(
    [
        [-0.5257311121191336, 0.85065080835204, 0.0],
        [0.5257311121191336, 0.85065080835204, 0.0],
        [-0.5257311121191336, -0.85065080835204, 0.0],
        [0.5257311121191336, -0.85065080835204, 0.0],
        [0.0, -0.5257311121191336, 0.85065080835204],
        [0.0, 0.5257311121191336, 0.85065080835204],
        [0.0, -0.5257311121191336, -0.85065080835204],
        [0.0, 0.5257311121191336, -0.85065080835204],
        [0.85065080835204, 0.0, -0.5257311121191336],
        [0.85065080835204, 0.0, 0.5257311121191336],
        [-0.85065080835204, 0.0, -0.5257311121191336],
        [-0.85065080835204, 0.0, 0.5257311121191336],
    ],
    dtype=np.float64,
)
_ICOSAHEDRON_FACES = np.array(
    [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ],
    dtype=np.int32,
).reshape(-1)


def _icosphere_face_table() -> np.ndarray:
    """
    Per-base-face topology of the icosahedron, as the ``(20, 9)`` int table the icosphere needs.

    Columns are ``(a, b, c)`` — the face's corner vertices, which are also their own global
    indices — then the base-edge id of each of the three sides ``a-b``, ``b-c``, ``c-a``, then a
    flag per side saying whether that side runs from the edge's higher-numbered endpoint to its
    lower one. Base edges are numbered by first appearance in ``_ICOSAHEDRON_FACES``.

    The two flags together are what makes the numbering *shared*: the ``n - 1`` interior points of
    a base edge are stored once, in the direction of its lower-numbered endpoint, and each of the
    two faces holding that edge walks them in whichever direction its own winding needs.
    """
    faces_np = _ICOSAHEDRON_FACES.reshape(-1, 3)
    edge_ids: dict[tuple[int, int], int] = {}
    table_np = np.empty((faces_np.shape[0], 9), dtype=np.int32)
    for f, (a, b, c) in enumerate(faces_np.tolist()):
        table_np[f, :3] = (a, b, c)
        for side, (u, v) in enumerate(((a, b), (b, c), (c, a))):
            key = (min(u, v), max(u, v))
            table_np[f, 3 + side] = edge_ids.setdefault(key, len(edge_ids))
            table_np[f, 6 + side] = u > v
    return table_np


# Built once: the icosahedron's topology is a constant, so the only per-call cost is the upload.
_ICOSPHERE_FACE_TABLE = _icosphere_face_table()

# Deliberately NOT cached per device. ``box``, ``tetrahedron``, ``octahedron``, ``icosahedron``
# and ``dodecahedron`` *return* the buffer they upload (``_apply_transform`` may rewrite its
# winding in place), so a cached table would have to be ``wp.clone``-d on read, which gives the
# upload straight back -- a bad trade for a module-level dict pinning device memory for the life
# of the process.


# The remaining three Platonic solids, as MeshLab's ``create_tetrahedron`` /
# ``create_octahedron`` / ``create_dodecahedron`` tables normalized onto the unit sphere (MeshLab
# emits them at whatever circumradius the integer coordinates give). Kept verbatim, exactly as
# ``_BOX_VERTICES`` is kept verbatim from trimesh, so both the vertex order and the pentagon-fan
# triangulation of the dodecahedron match the reference they came from.
_TETRAHEDRON_VERTICES = np.array(
    [
        [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
        [-0.5773502691896258, 0.5773502691896258, -0.5773502691896258],
        [-0.5773502691896258, -0.5773502691896258, 0.5773502691896258],
        [0.5773502691896258, -0.5773502691896258, -0.5773502691896258],
    ],
    dtype=np.float64,
)
_TETRAHEDRON_FACES = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]], dtype=np.int32).reshape(
    -1
)

_OCTAHEDRON_VERTICES = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_OCTAHEDRON_FACES = np.array(
    [[0, 1, 2], [0, 2, 4], [0, 4, 5], [0, 5, 1], [3, 1, 5], [3, 5, 4], [3, 4, 2], [3, 2, 1]],
    dtype=np.int32,
).reshape(-1)

# 20 vertices: the 8 cube corners at 1/sqrt(3) plus the 12 golden rectangle points. The 12
# pentagonal faces are each fanned from one cube corner, giving 36 triangles.
_DODECAHEDRON_VERTICES = np.array(
    [
        [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
        [0.5773502691896258, 0.5773502691896258, -0.5773502691896258],
        [0.5773502691896258, -0.5773502691896258, 0.5773502691896258],
        [0.5773502691896258, -0.5773502691896258, -0.5773502691896258],
        [-0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
        [-0.5773502691896258, 0.5773502691896258, -0.5773502691896258],
        [-0.5773502691896258, -0.5773502691896258, 0.5773502691896258],
        [-0.5773502691896258, -0.5773502691896258, -0.5773502691896258],
        [0.0, 0.35682208977309, 0.9341723589627156],
        [0.0, 0.35682208977309, -0.9341723589627156],
        [0.0, -0.35682208977309, 0.9341723589627156],
        [0.0, -0.35682208977309, -0.9341723589627156],
        [0.35682208977309, 0.9341723589627156, 0.0],
        [0.35682208977309, -0.9341723589627156, 0.0],
        [-0.35682208977309, 0.9341723589627156, 0.0],
        [-0.35682208977309, -0.9341723589627156, 0.0],
        [0.9341723589627156, 0.0, 0.35682208977309],
        [0.9341723589627156, 0.0, -0.35682208977309],
        [-0.9341723589627156, 0.0, 0.35682208977309],
        [-0.9341723589627156, 0.0, -0.35682208977309],
    ],
    dtype=np.float64,
)
_DODECAHEDRON_FACES = np.array(
    [
        [0, 8, 10],
        [0, 10, 2],
        [0, 2, 16],
        [0, 16, 17],
        [0, 17, 1],
        [0, 1, 12],
        [0, 12, 14],
        [0, 14, 4],
        [0, 4, 8],
        [5, 14, 12],
        [5, 12, 1],
        [5, 1, 9],
        [5, 19, 18],
        [5, 18, 4],
        [5, 4, 14],
        [5, 9, 11],
        [5, 11, 7],
        [5, 7, 19],
        [3, 11, 9],
        [3, 9, 1],
        [3, 1, 17],
        [3, 13, 15],
        [3, 15, 7],
        [3, 7, 11],
        [3, 17, 16],
        [3, 16, 2],
        [3, 2, 13],
        [6, 18, 19],
        [6, 19, 7],
        [6, 7, 15],
        [6, 15, 13],
        [6, 13, 2],
        [6, 2, 10],
        [6, 10, 8],
        [6, 8, 4],
        [6, 4, 18],
    ],
    dtype=np.int32,
).reshape(-1)


def box(
    extents: tuple[float, float, float] | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    bounds: Sequence[Sequence[float]] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a cuboid as 8 vertices and 12 triangles.

    Parameters
    ----------
    extents
        ``(3,)`` edge lengths. The box is centered on the origin. Defaults to a unit cube.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``. Face winding is reversed when the transform has negative determinant, so
        normals keep pointing outward.
    bounds
        ``(2, 3)`` axis-aligned corners. Overrides ``extents`` and ``transform``, and yields a
        box spanning exactly those corners rather than one centered on the origin.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with 8 vertices and 36 flat face indices.

    Raises
    ------
    ValueError
        If ``bounds`` is combined with ``extents`` or ``transform``, or if either has the wrong
        shape.

    See Also
    --------
    [`icosahedron`][triwarp.creation.icosahedron]
    [`cylinder`][triwarp.creation.cylinder]
    [`trimesh.creation.box`][]
    """
    vertices_np = _BOX_VERTICES.copy()

    if bounds is not None:
        if transform is not None or extents is not None:
            raise ValueError("bounds overrides extents/transform: pass only one")
        bounds_np = np.asanyarray(bounds, dtype=np.float64)
        if bounds_np.shape != (2, 3):
            raise ValueError(f"bounds must be (2, 3) float, got {bounds_np.shape}")
        vertices_np *= np.ptp(bounds_np, axis=0)
        vertices_np += bounds_np[0]
    elif extents is not None:
        extents_np = np.asanyarray(extents, dtype=np.float64)
        if extents_np.shape != (3,):
            raise ValueError(f"extents must be (3,) float, got {extents_np.shape}")
        vertices_np -= 0.5
        vertices_np *= extents_np
    else:
        vertices_np -= 0.5

    vertices = _upload_points(vertices_np, wp.vec3, device)
    faces = _launch.array(_BOX_FACES, dtype=wp.int32, device=device)
    return _apply_transform(vertices, faces, transform)


def grid(
    count: tuple[int, int] = (10, 10),
    extents: tuple[float, float] = (1.0, 1.0),
    center: bool = True,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a flat, regularly triangulated rectangle in the ``z = 0`` plane.

    The cheapest mesh with interior vertices and a boundary loop, which is what makes it the
    default fixture for anything that needs one: parametrization, boundary conditions, texture
    baking. Every quad cell is split by the same diagonal, so both triangles of a cell are
    right-angled — be aware that this makes every diagonal's cotangent weight exactly zero, which
    some solvers are sensitive to.

    Parameters
    ----------
    count
        ``(nx, ny)`` **vertex** counts along X and Y; each must be at least 2. The result has
        ``nx * ny`` vertices and ``2 * (nx - 1) * (ny - 1)`` triangles.
    extents
        ``(width, height)`` total span along X and Y.
    center
        When ``True`` (the default) the patch spans ``[-extents / 2, extents / 2]``; when
        ``False`` its lower corner sits at the origin, matching MeshLab's ``create_grid``.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``. Face winding is reversed when the transform has negative determinant.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``, wound so the normals point along ``+Z``.

    Raises
    ------
    ValueError
        If either entry of ``count`` is less than 2, or either entry of ``extents`` is negative.

    See Also
    --------
    [`box`][triwarp.creation.box]
    [`extrude_triangulation`][triwarp.creation.extrude_triangulation]

    Notes
    -----
    Both buffers are written **closed-form on the device**, in one launch of one thread per vertex
    that also writes the quad cell it is the lower corner of, rather than assembled on the host.
    Positions come out **bit-identical** to a NumPy build: the kernel does the same arithmetic in
    ``float64`` before the ``float32`` store, and phrases each sample as ``extent * (k / (n - 1))``
    so the far edge lands on the extent exactly, matching what ``numpy.linspace`` needs its
    endpoint special case for.
    """
    nx, ny = int(count[0]), int(count[1])
    if nx < 2 or ny < 2:
        raise ValueError(f"count must be at least 2 along each axis, got {count}")
    width, height = float(extents[0]), float(extents[1])
    if width < 0.0 or height < 0.0:
        raise ValueError(f"extents must be non-negative, got {extents}")

    vertices = _launch.empty(nx * ny, dtype=wp.vec3, device=device)
    faces = _launch.empty(6 * (nx - 1) * (ny - 1), dtype=wp.int32, device=device)
    _launch.launch(
        kernel_creation.grid_mesh,
        dim=(nx, ny),
        inputs=[
            wp.int32(nx),
            wp.int32(ny),
            wp.float64(width),
            wp.float64(height),
            wp.float64(-0.5 * width if center else 0.0),
            wp.float64(-0.5 * height if center else 0.0),
        ],
        outputs=[vertices, faces],
        device=device,
    )
    return _apply_transform(vertices, faces, transform)


def icosahedron(device: wp.DeviceLike = None) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a regular icosahedron on the unit sphere, centered on the origin.

    Parameters
    ----------
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with 12 vertices and 20 triangles.

    See Also
    --------
    [`icosphere`][triwarp.creation.icosphere]
    [`trimesh.creation.icosahedron`][]
    """
    vertices = _upload_points(_ICOSAHEDRON_VERTICES, wp.vec3, device)
    faces = _launch.array(_ICOSAHEDRON_FACES, dtype=wp.int32, device=device)
    return vertices, faces


def tetrahedron(device: wp.DeviceLike = None) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a regular tetrahedron on the unit sphere, centered on the origin.

    The four vertices are the even-parity cube corners ``(+-1, +-1, +-1) / sqrt(3)``. This is the
    coarsest closed triangle mesh there is, which makes it the natural stress case for anything
    that assumes a vertex has a large ring: every vertex has valence 3 and every face borders every
    other.

    Parameters
    ----------
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with 4 vertices and 4 triangles.

    See Also
    --------
    [`octahedron`][triwarp.creation.octahedron]
    [`icosahedron`][triwarp.creation.icosahedron]
    """
    return (
        _upload_points(_TETRAHEDRON_VERTICES, wp.vec3, device),
        _launch.array(_TETRAHEDRON_FACES, dtype=wp.int32, device=device),
    )


def octahedron(device: wp.DeviceLike = None) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a regular octahedron on the unit sphere, centered on the origin.

    The six vertices are the ``+-`` unit axis directions, so every face is an octant of the
    coordinate frame — the shape to reach for when a test wants a closed mesh whose faces are
    exactly axis-aligned.

    Parameters
    ----------
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with 6 vertices and 8 triangles.

    See Also
    --------
    [`tetrahedron`][triwarp.creation.tetrahedron]
    [`icosahedron`][triwarp.creation.icosahedron]
    """
    return (
        _upload_points(_OCTAHEDRON_VERTICES, wp.vec3, device),
        _launch.array(_OCTAHEDRON_FACES, dtype=wp.int32, device=device),
    )


def dodecahedron(device: wp.DeviceLike = None) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a regular dodecahedron on the unit sphere, centered on the origin.

    triwarp is triangle-only, so the twelve pentagonal faces arrive triangulated: each pentagon is
    fanned from one of the eight cube-corner vertices, giving 36 triangles over 20 vertices with no
    added centroid. The fan is MeshLab's, so this is vertex- and face-for-face identical to
    ``create_dodecahedron`` up to the uniform scaling onto the unit sphere.

    Because the triangulation is a fan rather than a symmetric split, the triangles are *not*
    congruent and the mesh is not a good uniform-sampling proxy — use
    [`icosphere`][triwarp.creation.icosphere] for that.

    Parameters
    ----------
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with 20 vertices and 36 triangles.

    See Also
    --------
    [`icosahedron`][triwarp.creation.icosahedron]
    [`octahedron`][triwarp.creation.octahedron]
    """
    return (
        _upload_points(_DODECAHEDRON_VERTICES, wp.vec3, device),
        _launch.array(_DODECAHEDRON_FACES, dtype=wp.int32, device=device),
    )


def icosphere(
    subdivisions: int = 3, radius: float = 1.0, device: wp.DeviceLike = None
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a geodesic sphere by recursively subdividing an icosahedron.

    Each refinement level splits every triangle into four and projects the new vertices back onto
    the sphere, so the result has ``20 * 4 ** subdivisions`` faces and ``10 * 4 ** subdivisions +
    2`` vertices — the same mesh [`trimesh.creation.icosphere`][] builds. Triangles stay far more
    uniform than [`uv_sphere`][triwarp.creation.uv_sphere]'s, at roughly an order of magnitude
    more work.

    Parameters
    ----------
    subdivisions
        Number of refinement levels. Face count grows as ``4 ** subdivisions``, so values above
        ~7 get expensive. Values ``<= 0`` return a plain icosahedron scaled to ``radius``.
    radius
        Sphere radius.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``.

    Notes
    -----
    The connectivity is generated in closed form rather than by iterating
    [`subdivide`][triwarp.remesh.subdivide]: every vertex of the refined mesh is addressed directly
    by its barycentric coordinates within one of the 20 base faces (see
    ``kernels.creation.icosphere_vertex_index``), so the whole face buffer is written by **one**
    kernel and the vertices by one launch per refinement level, with no host synchronization.

    The *geometry* is still the recursive one, level by level, because that is what the reference
    produces: a vertex is the projected midpoint of two vertices of the previous level, which is
    not the same point as the projection of the corresponding barycentric point of the base face
    (the two differ by a few percent of the edge length). Vertex *order* differs from trimesh —
    vertices come out as the 12 base corners, then the interior points of each base edge, then the
    interior points of each base face.

    See Also
    --------
    [`icosahedron`][triwarp.creation.icosahedron]
    [`uv_sphere`][triwarp.creation.uv_sphere]
    [`subdivide`][triwarp.remesh.subdivide]
    [`trimesh.creation.icosphere`][]
    """
    levels = max(0, int(subdivisions))
    n = 1 << levels
    radius_f = float(radius)

    table, corners = _icosphere_tables(wp.get_device(device).alias)
    vertices = _launch.empty(10 * 4**levels + 2, dtype=wp.vec3, device=corners.device)
    faces = _launch.empty(20 * n * n * 3, dtype=wp.int32, device=vertices.device)
    # The face buffer and level 0 -- the 12 base corners, which occupy the first block of the
    # numbering, scaled to `radius` -- in one launch; each level after that refines the last.
    _launch.launch(
        kernel_creation.icosphere_base,
        dim=(20, n, n),
        inputs=[table, corners, wp.int32(n), radius_f, vertices, faces],
        device=vertices.device,
    )
    for level in range(1, levels + 1):
        _launch.launch(
            kernel_creation.icosphere_generation,
            dim=(20, (1 << level) + 1, (1 << level) + 1),
            inputs=[table, wp.int32(n), wp.int32(1 << level), radius_f, vertices],
            device=vertices.device,
        )
    return vertices, faces


@functools.cache
def _icosphere_tables(device: str) -> tuple[wp.array[wp.int32], wp.array[wp.vec3]]:
    """
    Return the base face table and the 12 corners on the device ``device`` names, uploaded once.

    Unlike the Platonic-solid tables, which are *returned* and so must be fresh per call, these are
    read-only kernel inputs, so one copy per device serves every call.
    """
    table = _launch.array(_ICOSPHERE_FACE_TABLE, dtype=wp.int32, device=device)
    return table, _upload_points(_ICOSAHEDRON_VERTICES, wp.vec3, device)


def uv_sphere(
    radius: float = 1.0,
    count: tuple[int, int] | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a latitude/longitude sphere centered on the origin.

    Much cheaper than [`icosphere`][triwarp.creation.icosphere], at the cost of triangles that
    shrink toward the poles.

    Parameters
    ----------
    radius
        Sphere radius.
    count
        ``(2,)`` number of latitude and longitude lines. Defaults to ``(32, 64)``.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``.

    Raises
    ------
    ValueError
        If ``count`` is given and is not a ``(2,)`` integer pair.

    Notes
    -----
    The longitude count is doubled when ``count`` is given explicitly but not when it is left at
    the default, so ``count=(32, 64)`` yields 128 longitude sections while ``count=None`` yields
    64. This asymmetry is inherited from [`trimesh.creation.uv_sphere`][] and kept for parity.

    The two pole points are snapped to exactly ``(0, -radius)`` and ``(0, radius)`` in the 2D
    profile. ``sin(pi)`` is ``1.2e-16`` rather than zero, and without the snap the pole vertices
    of adjacent slices differ by ``1.2e-16 * radius`` — enough that the weld and the
    degenerate-triangle filter in [`revolve`][triwarp.creation.revolve] both start to fail at
    large radii. The correction is far below ``float32`` resolution.

    See Also
    --------
    [`icosphere`][triwarp.creation.icosphere]
    [`capsule`][triwarp.creation.capsule]
    [`revolve`][triwarp.creation.revolve]
    [`trimesh.creation.uv_sphere`][]
    """
    if count is None:
        latitude, longitude = 32, 64
    else:
        counts = np.asanyarray(count, dtype=np.int64)
        if counts.shape != (2,):
            raise ValueError(f"count must be (2,) int, got {counts.shape}")
        counts = counts + counts % 2
        latitude, longitude = int(counts[0]), int(counts[1]) * 2

    radius_f = abs(float(radius))
    theta = np.linspace(0.0, math.pi, num=latitude)
    profile = np.column_stack((np.sin(theta), -np.cos(theta))) * radius_f
    # Snap the poles: sin(0) is exact but sin(pi) is not (see Notes).
    profile[0] = (0.0, -radius_f)
    profile[-1] = (0.0, radius_f)

    n_sections = _resolve_sections(longitude)
    fast = _revolve_regular(profile, n_sections, transform, device)
    if fast is not None:
        return fast
    return revolve(
        _upload_points(profile, wp.vec2, device), sections=longitude, transform=transform
    )


def sphere_cap(
    angle: float = math.pi / 6.0,
    subdivisions: int = 3,
    radius: float = 1.0,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a spherical cap around ``+Z``: an open disc of a sphere, with one boundary loop.

    A triangular lattice of ``2 ** subdivisions`` concentric rings is projected onto the sphere, so
    ring ``r`` carries ``6 * r`` vertices evenly spaced in azimuth at polar angle
    ``angle * r / n_rings``. Unlike a sliced [`icosphere`][triwarp.creation.icosphere] the rim is a
    clean circle of exactly ``6 * n_rings`` vertices, which is what makes this the fixture of choice
    for boundary-condition work on a curved surface.

    Parameters
    ----------
    angle
        Polar half-angle of the cap in **radians**, measured from ``+Z``. Must be in ``(0, pi)``
        exclusive: at ``pi`` the whole rim collapses onto the south pole and every rim triangle
        would be degenerate. MeshLab's ``create_sphere_cap`` takes the full aperture in degrees
        instead, so its ``angle=60`` is ``math.radians(30)`` here.
    subdivisions
        Number of refinement passes; the lattice has ``2 ** subdivisions`` rings, giving
        ``1 + 3 * n * (n + 1)`` vertices and ``6 * n ** 2`` triangles for ``n = 2 **
        subdivisions``. Must be ``>= 0``.
    radius
        Sphere radius.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``, wound so the normals point away from the sphere
        center. Vertex 0 is the apex at ``(0, 0, radius)`` and the last ``6 * n`` vertices are the
        rim, in azimuthal order.

    Raises
    ------
    ValueError
        If ``angle`` is outside ``(0, pi)`` or ``subdivisions`` is negative.

    See Also
    --------
    [`icosphere`][triwarp.creation.icosphere]
    [`uv_sphere`][triwarp.creation.uv_sphere]
    """
    if not 0.0 < angle < math.pi:
        raise ValueError(f"angle must be in (0, pi) radians, got {angle}")
    if subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {subdivisions}")

    n_rings = 2 ** int(subdivisions)
    n_vertices = 1 + 3 * n_rings * (n_rings + 1)
    n_faces = 6 * n_rings * n_rings

    # Both buffers are written entirely on the device from the two counts and three scalars. The
    # lattice is a closed form in the vertex and triangle index -- no ring depends on the one
    # before it -- which is the case CLAUDE.md section 3.8 sanctions for a template whose output
    # scales with a resolution parameter, and the same conversion ``grid``, ``icosphere`` and
    # ``parametric_surface`` already took. The host build it replaces looped over rings in Python
    # and was quadratic in the ring count, against a flat device cost -- so the device path wins at
    # every resolution and needs no size gate.
    vertices_wp = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    faces_wp = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_creation.sphere_cap_mesh,
        dim=max(n_vertices, n_faces),
        inputs=[wp.int32(n_rings), wp.float64(angle), wp.float64(radius), vertices_wp, faces_wp],
        device=device,
    )
    return vertices_wp, faces_wp


def capsule(
    height: float = 1.0,
    radius: float = 1.0,
    count: tuple[int, int] | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a capsule: a cylinder along Z closed by a hemisphere at each end.

    The result is centered on the origin and spans ``z`` in
    ``[-height / 2 - radius, height / 2 + radius]``; ``height`` is the center-to-center distance
    between the two hemispheres.

    Parameters
    ----------
    height
        Center-to-center distance of the two hemispheres.
    radius
        Radius of the cylinder and of both hemispheres.
    count
        ``(2,)`` number of sections along latitude and longitude. Defaults to ``(32, 64)``. Both
        entries are rounded up to even, which is what gives the two quarter-circle profiles below
        the equator the same point count.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``.

    Raises
    ------
    ValueError
        If ``count`` is given and is not a ``(2,)`` integer pair.

    Notes
    -----
    Unlike [`uv_sphere`][triwarp.creation.uv_sphere], the longitude count is *not* doubled here.
    [`trimesh.creation.capsule`][]'s docstring describes the geometry as having one hemisphere at
    the origin and the other at ``height``; the implementation (and this port) centers it
    instead.

    The profile is two quarter-circles, each swept from the equator to a pole, rather than one
    continuous sweep from pole to pole: sharing the equator between two *exactly*
    ``cos(0) = 1``-radius points (one shifted to each hemisphere) is what gives the cylindrical
    wall between them an exact, constant radius, matching
    [`trimesh.creation.capsule`][]'s own construction (which does this for the same reason,
    per its own comment, "two quarter circles sharing an equator vertex"). A single pole-to-pole
    sweep only *approximates* the equator at the two points nearest ``latitude / 2``, which are a
    fraction of a section short of it, and reads as a nearly-vertical wall rather than an exactly
    vertical one; more concretely, it counts two fewer profile points than the exact
    construction, which shows up as a vertex-count mismatch against
    [`trimesh.creation.capsule`][] at every longitude count.

    See Also
    --------
    [`uv_sphere`][triwarp.creation.uv_sphere]
    [`cylinder`][triwarp.creation.cylinder]
    [`trimesh.creation.capsule`][]
    """
    counts = np.array([32, 64], dtype=np.int64) if count is None else np.asanyarray(count, np.int64)
    if counts.shape != (2,):
        raise ValueError(f"count must be (2,) int, got {counts.shape}")
    counts = counts + counts % 2
    latitude, longitude = int(counts[0]), int(counts[1])

    height_f = abs(float(height))
    radius_f = abs(float(radius))
    # Two quarter-circles sharing the equator, not one pole-to-pole sweep -- see the Notes above
    # for why the naive single sweep is both the wrong shape and the wrong vertex count.
    quarter_points = latitude // 2 + 1
    theta = np.concatenate(
        (
            np.linspace(-math.pi / 2.0, 0.0, quarter_points),
            np.linspace(0.0, math.pi / 2.0, quarter_points),
        )
    )
    profile = np.column_stack((np.cos(theta), np.sin(theta))) * radius_f
    half = len(profile) // 2
    profile[:half, 1] -= height_f / 2.0
    profile[half:, 1] += height_f / 2.0
    # Snap the poles the way uv_sphere does: cos(+-pi/2) is 6.1e-17, not zero.
    profile[0] = (0.0, -height_f / 2.0 - radius_f)
    profile[-1] = (0.0, height_f / 2.0 + radius_f)

    n_sections = _resolve_sections(longitude)
    fast = _revolve_regular(profile, n_sections, transform, device)
    if fast is not None:
        return fast
    return revolve(
        _upload_points(profile, wp.vec2, device), sections=longitude, transform=transform
    )


def cylinder(
    radius: float,
    height: float | None = None,
    sections: int | None = None,
    segment: Sequence[Sequence[float]] | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a closed cylinder along Z, centered on the origin.

    Parameters
    ----------
    radius
        Cylinder radius.
    height
        Cylinder height. Required unless ``segment`` is passed.
    sections
        Number of pie wedges around the revolution. Defaults to 32.
    segment
        ``(2, 3)`` axis endpoints. Overrides both ``height`` and ``transform``, placing the
        cylinder along the segment.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``.

    Raises
    ------
    ValueError
        If neither ``height`` nor ``segment`` is given, or ``segment`` is not ``(2, 3)``.

    See Also
    --------
    [`annulus`][triwarp.creation.annulus]
    [`cone`][triwarp.creation.cone]
    [`capsule`][triwarp.creation.capsule]
    [`trimesh.creation.cylinder`][]
    """
    transform, half = _resolve_cylinder_axis(height, segment, transform)
    n_sections = _resolve_sections(sections)
    radius_f = float(radius)
    profile = np.array(
        [[0.0, -half], [radius_f, -half], [radius_f, half], [0.0, half]], dtype=np.float64
    )
    keep_caps, keep_sides = _closed_form_template(profile, n_sections)
    vertices = _launch.empty(2 * n_sections + 2, dtype=wp.vec3, device=device)
    n_face_slots = (2 * keep_caps + 2 * keep_sides) * n_sections * 3
    faces = _launch.empty(n_face_slots, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_creation.cylinder_mesh,
        dim=n_sections,
        inputs=[
            wp.float32(radius_f),
            wp.float32(half),
            wp.int32(n_sections),
            _FULL_TURN,
            wp.int32(keep_caps),
            wp.int32(keep_sides),
        ],
        outputs=[vertices, faces],
        device=device,
    )
    return _apply_transform(vertices, faces, transform)


def cone(
    radius: float,
    height: float,
    sections: int | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a closed cone along Z with its base on the ``z = 0`` plane.

    Parameters
    ----------
    radius
        Radius of the cone at its widest (the base).
    height
        Height of the cone; the apex sits at ``(0, 0, height)``.
    sections
        Number of pie wedges around the revolution. Defaults to 32.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``.

    See Also
    --------
    [`cylinder`][triwarp.creation.cylinder]
    [`trimesh.creation.cone`][]
    """
    n_sections = _resolve_sections(sections)
    profile = np.array([[0.0, 0.0], [float(radius), 0.0], [0.0, float(height)]], dtype=np.float64)
    keep_caps, keep_sides = _closed_form_template(profile, n_sections)
    vertices = _launch.empty(n_sections + 2, dtype=wp.vec3, device=device)
    faces = _launch.empty((keep_caps + keep_sides) * n_sections * 3, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_creation.cone_mesh,
        dim=n_sections,
        inputs=[
            wp.float32(radius),
            wp.float32(height),
            wp.int32(n_sections),
            _FULL_TURN,
            wp.int32(keep_caps),
            wp.int32(keep_sides),
        ],
        outputs=[vertices, faces],
        device=device,
    )
    return _apply_transform(vertices, faces, transform)


def annulus(
    r_min: float,
    r_max: float,
    height: float | None = None,
    sections: int | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    segment: Sequence[Sequence[float]] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create an annular cylinder (a tube with flat ends) along Z, centered on the origin.

    Parameters
    ----------
    r_min
        Inner radius. When smaller than ``1e-8`` this delegates to
        [`cylinder`][triwarp.creation.cylinder], which has a different topology (no inner wall).
    r_max
        Outer radius.
    height
        Height of the annular cylinder. Required unless ``segment`` is passed.
    sections
        Number of pie wedges around the revolution. Defaults to 32.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    segment
        ``(2, 3)`` axis endpoints. Overrides both ``height`` and ``transform``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``.

    Raises
    ------
    ValueError
        If neither ``height`` nor ``segment`` is given, or ``segment`` is not ``(2, 3)``.

    See Also
    --------
    [`cylinder`][triwarp.creation.cylinder]
    [`torus`][triwarp.creation.torus]
    [`trimesh.creation.annulus`][]
    """
    transform, half = _resolve_cylinder_axis(height, segment, transform)
    r_min_f = abs(float(r_min))
    if r_min_f < TOLERANCE_MERGE:
        return cylinder(
            radius=r_max, height=2.0 * half, sections=sections, transform=transform, device=device
        )

    r_max_f = abs(float(r_max))
    # Counter-clockwise rectangle with the first point repeated: the duplicate closes the profile
    # (and therefore the inner-wall/cap seam) once revolve's weld runs.
    profile = np.array(
        [[r_min_f, -half], [r_max_f, -half], [r_max_f, half], [r_min_f, half], [r_min_f, -half]],
        dtype=np.float64,
    )
    n_sections = _resolve_sections(sections)
    fast = _revolve_regular(profile, n_sections, transform, device)
    if fast is not None:
        return fast
    return revolve(_upload_points(profile, wp.vec2, device), sections=sections, transform=transform)


def torus(
    major_radius: float,
    minor_radius: float,
    major_sections: int = 32,
    minor_sections: int = 32,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a torus around Z, centered on the origin.

    Parameters
    ----------
    major_radius
        Distance from the center of the torus to the center of the tube.
    minor_radius
        Radius of the tube.
    major_sections
        Number of sections around the major radius.
    minor_sections
        Number of sections around the minor radius.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with ``2 * major_sections * minor_sections`` triangles.

    Notes
    -----
    The closing point of the tube profile is snapped to equal its first point exactly. ``sin(2 *
    pi)`` evaluates to ``-2.4e-16``, so without the snap the minor seam would only close through
    [`revolve`][triwarp.creation.revolve]'s absolute weld tolerance and would come apart at large
    radii.

    See Also
    --------
    [`annulus`][triwarp.creation.annulus]
    [`revolve`][triwarp.creation.revolve]
    [`trimesh.creation.torus`][]
    """
    minor_f = float(minor_radius)
    phi = np.linspace(0.0, 2.0 * math.pi, int(minor_sections) + 1, endpoint=True)
    profile = np.column_stack((minor_f * np.cos(phi), minor_f * np.sin(phi)))
    profile += (float(major_radius), 0.0)
    profile[-1] = profile[0]

    n_sections = _resolve_sections(major_sections)
    fast = _revolve_regular(profile, n_sections, transform, device)
    if fast is not None:
        return fast
    return revolve(
        _upload_points(profile, wp.vec2, device), sections=major_sections, transform=transform
    )


def revolve(
    linestring: wp.array[wp.vec2],
    angle: float | None = None,
    cap: bool = False,
    sections: int | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Revolve a 2D profile around the 2D Y axis, which becomes the 3D Z axis.

    This is the shared engine behind every radially symmetric primitive in this module. The 2D X
    component of ``linestring`` is used as the revolution radius and the 2D Y component as the
    height along Z.

    Parameters
    ----------
    linestring
        ``(n,)`` ordered 2D profile points, ``n >= 2``. A closed profile should be wound
        counter-clockwise for outward-facing normals.
    angle
        Angle in radians to revolve through. ``None`` (the default) is a full revolution.
    cap
        For a partial revolution, triangulate the two end faces so the result is a closed volume.
        Ignored for a full revolution, which needs no caps. Requires a simple (non
        self-intersecting) profile — see
        [`triangulate_polygon`][triwarp.polyline.triangulate_polygon].
    sections
        Number of pie wedges around the revolution. Defaults to 32 per full revolution, scaled by
        ``angle``.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``. Face winding is reversed when its determinant is negative.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``linestring.device``.

    Raises
    ------
    ValueError
        If ``linestring`` is not a rank-1 ``wp.vec2`` array with at least 2 points, or
        ``sections`` resolves to less than 1.
    RuntimeError
        If ``linestring`` and ``transform`` are not all on one device.

    Notes
    -----
    Two clean-up steps are part of the algorithm, not optional polish:

    - **Degenerate triangles are dropped.** Any template triangle whose area is at most ``1e-8``
      is discarded, which is how the polar triangles of a sphere, the apex and base fans of a
      cone, and the axis triangles of a cylinder disappear without being special-cased. The
      threshold is absolute, exactly as trimesh's ``tol.merge`` is, so a sphere's polar triangles
      stop registering as degenerate somewhere above ``radius = 1e4``. Build near unit scale and
      pass a scaling ``transform`` if you need the result large.
    - **Coincident vertices are collapsed**, which is what makes the result watertight: a cone
      apex, a sphere pole, and the closing point of a closed profile each appear once *per slice*
      beforehand. `trimesh` discovers these by position hashing (the ``merge_vertices`` pass of
      ``Trimesh(process=True)``); this port derives them from the profile instead — a profile
      point within ``1e-8`` of the axis becomes a single shared vertex, and a profile whose last
      point repeats its first shares that column. That is exact and scale independent, where a
      position hash both misses ``+0.0``/``-0.0`` pole pairs and starts merging genuinely distinct
      vertices at high section counts.

    A consequence of deriving the collapse rather than measuring it: a profile that repeats an
    *interior* point, or whose ends are close but not equal, keeps its duplicate vertices (the
    zero-area faces between them are still dropped, so they are simply unreferenced).

    See Also
    --------
    [`uv_sphere`][triwarp.creation.uv_sphere]
    [`cylinder`][triwarp.creation.cylinder]
    [`torus`][triwarp.creation.torus]
    [`extrude_triangulation`][triwarp.creation.extrude_triangulation]
    [`trimesh.creation.revolve`][]
    """
    require_same_device(linestring=linestring, transform=transform)
    twt.ensure_ndim(linestring, 1, dtype=wp.vec2)
    device = linestring.device
    per = linestring.size
    if per < 2:
        raise ValueError(f"linestring must have at least 2 points, got {per}")

    full = 2.0 * math.pi
    closed = angle is None or abs(float(angle) - full) <= _CLOSED_ANGLE_ATOL
    span = full if angle is None else float(angle)
    if sections is None:
        sections = int(span / full * DEFAULT_SECTIONS)
    # trimesh converts the wedge count to a point count here.
    n_points = int(sections) + 1
    n_slices = n_points - 1
    if n_slices < 1:
        raise ValueError(f"sections must be at least 1, got {sections}")

    # Everything that depends only on the profile is decided on the host, from at most a few dozen
    # points: which template triangles survive, and where each (slice, profile point) pair lands in
    # the merged output. The two kernels below then write their final layout directly, with no
    # compaction pass and no device-to-host synchronization.
    profile_np = linestring.numpy().astype(np.float64)
    # A closed revolution folds its last slice back onto the first; an open one keeps them all.
    # The vertex layout and the launch must agree on this exactly -- the layout reserves the block
    # the kernel then writes into -- so it is computed once and passed to both.
    n_kept_slices = n_slices if closed else n_points
    keep_np = _revolve_kept_template(profile_np, span / float(n_slices))
    column_np, offsets_np, on_axis_np, n_vertices = _revolve_vertex_layout(
        profile_np, n_kept_slices
    )
    n_keep = keep_np.size
    # One upload: the three per-point tables and the kept template, in the order ``revolve_mesh``
    # reads them.
    layout = _launch.array(
        np.concatenate((column_np, offsets_np, on_axis_np, keep_np)).astype(np.int32),
        dtype=wp.int32,
        device=device,
    )

    cap_faces = None
    n_cap = 0
    if cap and not closed:
        profile_3d = _launch.empty(per, dtype=wp.vec3, device=device)
        _launch.map(kernel_array.lift_vec2, linestring, wp.float32(0.0), out=profile_3d)
        # Ear clipping introduces no new vertices, so its indices address profile points directly --
        # the guarantee trimesh gets from ``triangulate_polygon(force_vertices=True)``.
        cap_faces = tw.polyline.polyline_triangulate(profile_3d).reshape((-1,))
        n_cap = cap_faces.size // 3

    vertices = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    faces = _launch.empty((n_slices * n_keep + 2 * n_cap) * 3, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_creation.revolve_mesh,
        dim=n_kept_slices * per + n_slices * n_keep + 2 * n_cap,
        inputs=[
            linestring,
            wp.float32(span),
            wp.int32(n_points),
            wp.int32(n_kept_slices),
            wp.int32(n_slices),
            layout,
            cap_faces,
            vertices,
            faces,
        ],
        device=device,
    )

    return _apply_transform(vertices, faces, transform)


def _revolve_kept_template(profile_np: np.ndarray, step: float) -> np.ndarray:
    """
    Template triangles of one revolution slice that are not degenerate.

    trimesh drops any triangle of the slice-0 template whose area is at most ``1e-8``, which is how
    the polar triangles of a sphere, the apex and base fans of a cone, and the axis triangles of a
    cylinder disappear without being special-cased. Every slice is a rotation of slice 0, so one
    mask covers them all, and the whole test is ``2 * (per - 1)`` triangles — cheap enough to keep
    on the host in ``float64``, which also matches the reference's precision.
    """
    per = profile_np.shape[0]
    if per <= _KEPT_TEMPLATE_LOOP_MAX:
        return _revolve_kept_template_scalar(profile_np, step)
    radius_np, height_np = profile_np[:, 0], profile_np[:, 1]
    grid_np = np.vstack(
        (
            np.column_stack((radius_np, np.zeros(per), height_np)),
            np.column_stack((np.cos(step) * radius_np, np.sin(step) * radius_np, height_np)),
        )
    )
    segment_np = np.arange(per - 1)
    triangles_np = np.empty((2 * (per - 1), 3), dtype=np.int64)
    triangles_np[0::2] = np.column_stack((segment_np, segment_np + per, segment_np + 1))
    triangles_np[1::2] = np.column_stack((segment_np + 1, segment_np + per, segment_np + per + 1))
    corner_np = grid_np[triangles_np]
    areas_np = 0.5 * np.linalg.norm(
        np.cross(corner_np[:, 1] - corner_np[:, 0], corner_np[:, 2] - corner_np[:, 0]), axis=1
    )
    return np.flatnonzero(areas_np > TOLERANCE_MERGE).astype(np.int32)


def _revolve_kept_template_scalar(profile_np: np.ndarray, step: float) -> np.ndarray:
    """
    [`_revolve_kept_template`][triwarp.creation._revolve_kept_template] in Python floats.

    The same IEEE operations in the same order -- the template's corner differences, the cross
    product's ``a1 * b2 - a2 * b1`` components, and the norm's left-to-right sum of squares -- so
    the verdict is identical bit for bit. On a profile of a handful of points the array form's cost
    is its per-call NumPy overhead, which this avoids.
    """
    cos_step = float(np.cos(step))
    sin_step = float(np.sin(step))
    rows = profile_np.tolist()
    kept: list[int] = []
    for s in range(len(rows) - 1):
        r0, h0 = rows[s]
        r1, h1 = rows[s + 1]
        # Corners: slice 0 at ``(r, 0, h)`` and slice 1 at ``(cos * r, sin * r, h)``.
        qx, qy = cos_step * r0, sin_step * r0
        # Triangle ``(s, s + per, s + 1)``, then ``(s + 1, s + per, s + per + 1)``.
        for t, (ox, oz, bx, by, bz) in enumerate(
            (
                (r0, h0, r1 - r0, 0.0 - 0.0, h1 - h0),
                (r1, h1, cos_step * r1 - r1, sin_step * r1 - 0.0, h1 - h1),
            )
        ):
            ax, ay, az = qx - ox, qy - 0.0, h0 - oz
            cx = ay * bz - az * by
            cy = az * bx - ax * bz
            cz = ax * by - ay * bx
            if 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz) > TOLERANCE_MERGE:
                kept.append(2 * s + t)
    return np.array(kept, dtype=np.int32)


def _revolve_vertex_layout(
    profile_np: np.ndarray, n_kept_slices: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Per-profile-point tables placing each revolved vertex at its final, already-merged index.

    Returns ``(column, offsets, on_axis, n_vertices)``. ``column[i]`` is the profile point ``i``
    shares its position with (itself, except that a profile repeating its first point as its last
    folds that column onto ``0``), ``on_axis[j]`` marks a point within ``1e-8`` of the revolution
    axis — one vertex for the whole revolution rather than one per slice — and ``offsets[j]`` is
    where the owning column's block starts.

    This replaces the position-hash vertex merge trimesh gets from ``Trimesh(process=True)``. It is
    exact and scale independent, where a position hash both misses ``+0.0``/``-0.0`` pole pairs and
    starts merging genuinely distinct vertices at high section counts.
    """
    per = profile_np.shape[0]
    on_axis_np = np.abs(profile_np[:, 0]) <= TOLERANCE_MERGE
    column_np = np.arange(per, dtype=np.int32)
    if per > 2 and bool(np.all(np.abs(profile_np[0] - profile_np[-1]) <= TOLERANCE_MERGE)):
        column_np[per - 1] = 0
    owns_np = column_np == np.arange(per)
    counts_np = np.where(on_axis_np, 1, n_kept_slices) * owns_np
    offsets_np = np.concatenate(([0], np.cumsum(counts_np)[:-1])).astype(np.int32)
    return column_np, offsets_np, on_axis_np, int(counts_np.sum())


def extrude_triangulation(
    vertices: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    height: float,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extrude a 2D triangulation along Z into a watertight mesh.

    The input triangulation becomes the two caps; the walls are raised from its boundary edges,
    which are recovered from the triangulation itself rather than assumed, so a subdivided outline
    works.

    Parameters
    ----------
    vertices
        ``(n,)`` 2D vertex positions.
    faces
        Length-``3 * n_faces`` flat triangle index buffer into ``vertices``.
    height
        Distance to extrude along Z. May be negative; the triangulation is re-wound to agree with
        its sign so the result always has positive volume.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``. Face winding is reversed when its determinant is negative.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with ``2 * n`` vertices, on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``vertices`` is not a rank-1 ``wp.vec2`` array, ``faces`` is not a flat multiple of 3,
        or ``abs(height)`` is at most ``1e-8``.
    RuntimeError
        If ``vertices``, ``faces`` and ``transform`` are not all on one device.

    Notes
    -----
    Unlike [`trimesh.creation.extrude_triangulation`][], which samples the first ten triangles to
    decide the input winding, this averages the signed area of *every* triangle. The answer is the
    same for any consistently wound triangulation and does not depend on which triangles happen to
    come first.

    The walls index the cap vertices directly instead of carrying their own four-vertex quad per
    boundary edge, so the output is watertight with no vertex-merge pass (trimesh relies on
    ``Trimesh(process=True)`` to fuse the three blocks) and has exactly ``2 * n`` vertices.

    See Also
    --------
    [`extrude_polygon`][triwarp.creation.extrude_polygon]
    [`sweep_polygon`][triwarp.creation.sweep_polygon]
    [`revolve`][triwarp.creation.revolve]
    [`trimesh.creation.extrude_triangulation`][]
    """
    require_same_device(vertices=vertices, faces=faces, transform=transform)
    return _extrude(vertices, faces, height, transform, ring_walls=False)


def extrude_polygon(
    polygon: wp.array[wp.vec2],
    height: float,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    mid_plane: bool = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extrude a simple 2D polygon along Z into a watertight prism.

    Triangulates the ring with [`triangulate_polygon`][triwarp.polyline.triangulate_polygon] and
    hands the result to [`extrude_triangulation`][triwarp.creation.extrude_triangulation].

    Parameters
    ----------
    polygon
        ``(n,)`` 2D ring vertices, in order. A repeated closing point is dropped. Interior rings
        (holes) are not supported.
    height
        Distance to extrude along Z.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    mid_plane
        Center the extrusion on ``z = 0`` instead of starting it there.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``polygon.device``.

    Raises
    ------
    RuntimeError
        If ``polygon`` and ``transform`` are not all on one device.

    See Also
    --------
    [`triangulate_polygon`][triwarp.polyline.triangulate_polygon]
    [`extrude_triangulation`][triwarp.creation.extrude_triangulation]
    [`sweep_polygon`][triwarp.creation.sweep_polygon]
    [`trimesh.creation.extrude_polygon`][]
    """
    require_same_device(polygon=polygon, transform=transform)
    ring, faces = tw.polyline.triangulate_polygon(polygon)
    if mid_plane:
        translation = np.eye(4)
        translation[2, 3] = abs(float(height)) / -2.0
        if transform is None:
            transform = wp.mat44(*translation.flatten())
        else:
            composed = tw.transform.matrix_to_numpy(transform).dot(translation)
            transform = wp.mat44(*composed.flatten())
    # A full triangulation of the ring is bounded by the ring edges themselves; a partial one (a
    # degenerate ring) has its boundary derived, as ``extrude_triangulation`` does.
    full = faces.size // 3 == ring.size - 2
    return _extrude(ring, faces, height, transform, ring_walls=full)


def _extrude(
    vertices: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    height: float,
    transform: wp.mat44 | wp.array[wp.mat44] | None,
    *,
    ring_walls: bool,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Shared body of ``extrude_triangulation`` and ``extrude_polygon``.

    ``ring_walls`` says ``faces`` is a full ``n - 2`` triangulation of the ring ``vertices``, whose
    boundary is then the ring edges and is not derived. Whether the triangulation is re-wound to
    agree with the sign of ``height`` is decided on the device, from its total signed area.
    """
    twt.ensure_ndim(vertices, 1, dtype=wp.vec2)
    twt.ensure_ndim(faces, 1, dtype=wp.int32)
    device = vertices.device
    n = vertices.size
    n_faces = faces.size // 3
    if faces.size % 3 != 0:
        raise ValueError(f"faces size must be a multiple of 3, got {faces.size}")
    height_f = float(height)
    if abs(height_f) < TOLERANCE_MERGE:
        raise ValueError(f"height must be nonzero, got {height_f}")

    if n_faces == 0:
        return _launch.empty(0, dtype=wp.vec3, device=device), _launch.empty(
            0, dtype=wp.int32, device=device
        )

    # Both layers, ``z = 0`` then ``z = height``, and the triangulation's signed area in one pass.
    layers = _launch.empty(2 * n, dtype=wp.vec3, device=device)
    area = _launch.zeros(1, dtype=wp.float32, device=device)
    _launch.launch_tiled(
        kernel_creation.lift_layers_and_signed_area,
        dim=kernel_reduce.chunks_1d(max(n, n_faces)),
        inputs=[vertices, faces, wp.float32(height_f), layers, area],
        block_dim=TILE_1D,
        device=device,
    )

    boundary = None
    n_boundary = n
    if not ring_walls:
        # The lower layer is a contiguous prefix, so the boundary walk reads it as a view. The
        # boundary of the triangulation as given; a re-wound one reverses each edge on the device.
        boundary = tw.boundary.oriented_boundary_edges(twt.as_dense(layers[:n]), faces)
        n_boundary = int(boundary.shape[0])

    # Bottom cap winding is reversed; the top cap keeps it and is offset by one vertex block.
    out_faces = _launch.empty((2 * n_faces + 2 * n_boundary) * 3, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_creation.extrude_faces,
        dim=2 * n_faces + n_boundary,
        inputs=[faces, boundary, wp.int32(n), height_f < 0.0, area, out_faces],
        device=device,
    )
    return _apply_transform(layers, out_faces, transform)


def sweep_polygon(
    polygon: wp.array[wp.vec2],
    path: wp.array[wp.vec3],
    angles: wp.array[wp.float32] | None = None,
    cap: bool = True,
    connect: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Sweep a simple 2D polygon along a 3D path.

    One copy of the polygon is placed at every path vertex, on the plane bisecting the two
    adjacent path segments, and consecutive copies are bridged into walls.

    Parameters
    ----------
    polygon
        ``(n,)`` 2D ring vertices, in order. A repeated closing point is dropped. Interior rings
        (holes) are not supported.
    path
        ``(m,)`` path vertices, ``m >= 2``. The path counts as closed when its first and last
        vertex coincide.
    angles
        ``(m,)`` roll of the polygon about the path tangent at each path vertex, in radians.
        Defaults to no roll.
    cap
        Triangulate the two ends. Ignored when the path is closed and ``connect`` is set, which
        needs no caps.
    connect
        For a closed path, fuse the last slice back onto the first so the sweep is a single
        watertight body.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``polygon.device``.

    Raises
    ------
    ValueError
        If ``path`` has fewer than 2 vertices, ``angles`` does not match ``path`` in length, or the
        polygon's triangulation is not bounded by exactly one edge per ring vertex (which means the
        ring is not a simple polygon).
    RuntimeError
        If ``polygon``, ``path`` and ``angles`` are not all on one device.

    See Also
    --------
    [`extrude_polygon`][triwarp.creation.extrude_polygon]
    [`revolve`][triwarp.creation.revolve]
    [`trimesh.creation.sweep_polygon`][]
    """
    require_same_device(polygon=polygon, path=path, angles=angles)
    twt.ensure_ndim(path, 1, dtype=wp.vec3)
    device = polygon.device
    n_path = path.size
    if n_path < 2:
        raise ValueError(f"path must have at least 2 points, got {n_path}")
    if angles is not None and angles.size != n_path:
        raise ValueError(
            f"angles must have one entry per path point ({n_path}), got {angles.shape}"
        )

    ring, cap_faces = tw.polyline.triangulate_polygon(polygon)
    stride = ring.size
    boundary = None
    n_boundary = stride
    if cap_faces.size // 3 != stride - 2:
        # Only a partial triangulation (a ring that is not simple) can be bounded by anything but
        # the ring edges. ``oriented_boundary_edges`` only uses the vertex count, as its row-hash
        # base, so a zero buffer of the right length is enough.
        boundary = tw.boundary.oriented_boundary_edges(
            _launch.zeros(stride, dtype=wp.vec3, device=device), cap_faces
        )
        n_boundary = int(boundary.shape[0])
        if n_boundary != stride:
            raise ValueError(
                f"polygon must be a simple ring: its triangulation has {n_boundary} boundary "
                f"edges for {stride} vertices"
            )

    # Two 12-byte endpoint reads decide whether the path closes, and only ``connect`` asks; the rest
    # of it never leaves the device. ``read_scalar`` avoids allocating a fresh host array per call
    # the way ``path[k : k + 1].numpy()[0]`` would.
    connect_closed = (
        connect and math.dist(read_scalar(path, 0), read_scalar(path, n_path - 1)) < TOLERANCE_MERGE
    )

    # A connected closed path drops its duplicate final slice and wraps onto the first instead.
    n_slices = n_path - 1
    n_vertices = (n_slices if connect_closed else n_path) * stride
    n_cap = 0 if connect_closed or not cap else cap_faces.size // 3
    vertices = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    faces = _launch.empty(
        (2 * n_slices * n_boundary + 2 * n_cap) * 3, dtype=wp.int32, device=device
    )
    _launch.launch(
        kernel_creation.sweep_mesh,
        dim=n_vertices + n_slices * n_boundary + 2 * n_cap,
        inputs=[
            ring,
            path,
            angles,
            connect_closed,
            boundary,
            cap_faces,
            wp.int32(n_slices),
            wp.int32(n_boundary),
            wp.int32(n_vertices),
            vertices,
            faces,
        ],
        device=device,
    )
    return vertices, faces


def truncated_prisms(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    origin: wp.vec3 | None = None,
    normal: wp.vec3 | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extrude every triangle of a mesh down onto a plane, as one watertight prism each.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    origin
        Point on the truncation plane. ``None`` truncates against the ``z = 0`` plane.
    normal
        Unit normal of the truncation plane. Required when ``origin`` is given.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with ``6 * n_faces`` vertices and ``8 * n_faces`` triangles, on
        ``vertices.device``.

    Raises
    ------
    ValueError
        If ``faces`` is not a flat multiple of 3, or ``origin`` is given without ``normal``.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    Takes an indexed mesh rather than [`trimesh.creation.truncated_prisms`][]'s ``(n, 3, 3)``
    triangle soup, matching the ``(vertices, faces)`` convention used throughout triwarp. Vertices
    are *not* merged, so each prism stays a separate body — the same choice trimesh makes with
    ``process=False``.

    See Also
    --------
    [`extrude_triangulation`][triwarp.creation.extrude_triangulation]
    [`trimesh.creation.truncated_prisms`][]
    """
    require_same_device(vertices=vertices, faces=faces)
    twt.ensure_ndim(faces, 1, dtype=wp.int32)
    device = vertices.device
    if faces.size % 3 != 0:
        raise ValueError(f"faces size must be a multiple of 3, got {faces.size}")
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.vec3, device=device), _launch.empty(
            0, dtype=wp.int32, device=device
        )

    if origin is None:
        transform_np = np.eye(4)
    else:
        if normal is None:
            raise ValueError("normal is required when origin is given")
        transform_np = _plane_transform(np.array(origin), np.array(normal))

    out_vertices = _launch.empty(6 * n_faces, dtype=wp.vec3, device=device)
    out_faces = _launch.empty(24 * n_faces, dtype=wp.int32, device=device)
    # Inverted in NumPy, on the matrix that is already in hand, rather than with ``wp.inverse``:
    # a Warp builtin at Python scope routes through builtin dispatch and is several times dearer,
    # and the two agree to float32 rounding (``wp.inverse`` works in float32). Section 13.1.
    to_plane = wp.mat44(*transform_np.flatten())
    from_plane = wp.mat44(*np.linalg.inv(transform_np).flatten())
    _launch.launch(
        kernel_creation.truncated_prism_geometry,
        dim=n_faces,
        inputs=[vertices, faces, to_plane, from_plane, out_vertices, out_faces],
        device=device,
    )
    return out_vertices, out_faces


def _plane_transform(origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Transform moving the plane through ``origin`` with ``normal`` onto the XY plane."""
    transform = _align_vectors(np.asanyarray(normal, dtype=np.float64).reshape(3), _UNIT_Z)
    transform[:3, 3] = -transform.dot(
        np.append(np.asanyarray(origin, dtype=np.float64).reshape(3), 1.0)
    )[:3]
    return transform


def axis(
    origin_size: float = 0.04,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    axis_radius: float | None = None,
    axis_length: float | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create an XYZ axis marker: a ball at the origin and one cylinder along each axis.

    Parameters
    ----------
    origin_size
        Radius of the ball marking the origin. The other defaults are derived from it.
    transform
        Transform applied to the whole marker, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    axis_radius
        Radius of the three axis cylinders. Defaults to ``origin_size / 5``.
    axis_length
        Length of the three axis cylinders. Defaults to ``origin_size * 10``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``, with the ball and the three cylinders concatenated in
        X, Y, Z order after the ball.

    Notes
    -----
    [`trimesh.creation.axis`][] colors the ball white and the cylinders red/green/blue for X/Y/Z.
    triwarp carries no visual attributes, so this returns geometry only and takes no
    ``origin_color``; the pieces are concatenated in a documented order so a caller can assign
    per-face colors itself.

    See Also
    --------
    [`icosphere`][triwarp.creation.icosphere]
    [`cylinder`][triwarp.creation.cylinder]
    [`concatenate`][triwarp.combine.concatenate]
    [`trimesh.creation.axis`][]
    """
    origin_size_f = float(origin_size)
    if axis_radius is None:
        axis_radius = origin_size_f / 5.0
    if axis_length is None:
        axis_length = origin_size_f * 10.0

    # Each cylinder is built centered on the origin, so it has to be pushed out along its own axis
    # by half its length. The marker transform is applied once, to the assembled result.
    shift = np.eye(4)
    shift[2, 3] = float(axis_length) / 2.0

    parts = [icosphere(radius=origin_size_f, device=device)]
    for direction in (_UNIT_X, _UNIT_Y, _UNIT_Z):
        placement = _align_vectors(_UNIT_Z, direction).dot(shift)
        parts.append(
            cylinder(
                radius=float(axis_radius),
                height=float(axis_length),
                transform=wp.mat44(*placement.flatten()),
                device=device,
            )
        )
    return _apply_transform(*tw.combine.concatenate(parts), transform)


ParametricSurfaceKind = Literal[
    "bohemian_dome",
    "bour",
    "boy",
    "catalan_minimal",
    "conic_spiral",
    "cross_cap",
    "dini",
    "enneper",
    "figure8_klein",
    "henneberg",
    "klein",
    "kuen",
    "mobius",
    "plucker_conoid",
    "pseudosphere",
    "roman",
]
"""Analytic surface selected by [`parametric_surface`][triwarp.creation.parametric_surface]."""


def parametric_surface(
    kind: ParametricSurfaceKind,
    u_resolution: int = 40,
    v_resolution: int = 40,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create one of sixteen classical analytic surfaces by sampling its parameterization.

    These are the meshes with topology the rest of the module cannot produce: four are closed and
    **non-orientable**, two are non-orientable with a boundary, one is closed of genus 1, and the
    other nine are open patches, several with strongly graded triangles. Their Euler characteristics
    are 0 and 1 and their bounding-box diagonals span 1.7 to 28.3, so they are the package's inputs
    for
    [`is_orientable`][triwarp.validation.is_orientable],
    [`homology_generators`][triwarp.homology.homology_generators] and anything whose behaviour
    should not depend on the scale of its input.

    Parameters
    ----------
    kind
        Which surface to build:

        - non-orientable and closed — ``"boy"`` and ``"cross_cap"`` (the two immersions of the real
          projective plane, both with Euler characteristic 1), ``"figure8_klein"`` (the Klein
          bottle) and ``"roman"`` (Steiner's surface, with three double lines).
        - non-orientable with a boundary — ``"mobius"`` and ``"henneberg"``.
        - closed of genus 1 — ``"bohemian_dome"``, and see
          [`super_toroid`][triwarp.creation.super_toroid].
        - open, with two boundary loops — ``"klein"`` (which is *not* a Klein bottle as VTK
          parameterizes it: it is orientable and has a boundary), ``"plucker_conoid"`` and
          ``"pseudosphere"``.
        - open, with one boundary loop — ``"bour"``, ``"catalan_minimal"``, ``"conic_spiral"``,
          ``"dini"``, ``"enneper"`` and ``"kuen"``. ``"dini"`` and ``"enneper"`` are the strongly
          graded ones.
    u_resolution, v_resolution
        Number of samples along each parameter direction, including both ends of the domain. At
        least 2.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``. The vertex count is below
        ``u_resolution * v_resolution`` wherever the surface glues, and the face count is
        ``2 * (u_resolution - 1) * (v_resolution - 1)`` less one triangle per pole-adjacent cell.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the listed names, or either resolution is below 2 — or below
        3 along an axis the surface wraps without a twist, where a resolution of 2 would identify
        every cell's two rows and leave no faces at all.

    Notes
    -----
    Ported from VTK's ``vtkParametric*`` classes, which pyvista exposes as ``pv.Parametric*``; the
    maps are evaluated in VTK's own frame, so the two agree pointwise to float32.

    The identification along a seam or at a pole is **combinatorial** — it is a fact about the map,
    applied to the index buffer — where VTK welds its raw lattice by distance afterwards. Two
    consequences worth knowing. The topology here is exact and identical at every resolution and on
    every device, and it does not depend on
    [`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices], which several of
    these surfaces exist to test. And where an immersed surface merely *crosses* itself, the sheets
    stay separate: on Catalan's minimal surface VTK merges 40 lattice points that the
    parameterization does not identify, and drops the 2 triangles that thereby became degenerate.

    Examples
    --------
    ```python
    vertices, faces = tw.creation.parametric_surface("boy")
    print(tw.measures.euler_characteristic(faces), tw.validation.is_orientable(faces))
    ```

    See Also
    --------
    [`super_ellipsoid`][triwarp.creation.super_ellipsoid]
    [`super_toroid`][triwarp.creation.super_toroid]
    [`random_hills`][triwarp.creation.random_hills]
    [`grid`][triwarp.creation.grid]
    """
    if kind not in _PARAMETRIC_SPECS:
        raise ValueError(f"unknown kind {kind!r}, expected one of {sorted(_PARAMETRIC_SPECS)}")
    return _parametric_surface(_PARAMETRIC_SPECS[kind], u_resolution, v_resolution, device)


def super_ellipsoid(
    n1: float = 1.0,
    n2: float = 1.0,
    radii: tuple[float, float, float] = (1.0, 1.0, 1.0),
    u_resolution: int = 40,
    v_resolution: int = 40,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a superquadric ellipsoid: a closed genus-0 surface with a squareness axis.

    ``n1 = n2 = 1`` is the ellipsoid — the unit sphere at the default ``radii`` — and lowering
    either exponent flattens the surface towards a box, so a sweep of ``n1`` takes one connectivity
    from smooth through creased to nearly sharp-edged. That axis is why this is a builder in its own
    right rather than one entry of
    [`parametric_surface`][triwarp.creation.parametric_surface]'s enum.

    Parameters
    ----------
    n1
        Squareness exponent along the v (latitude) direction. ``1`` is the ellipsoid; below ``1``
        the surface creases at the equator, above ``1`` it pinches towards the poles.
    n2
        Squareness exponent along the u (longitude) direction, creasing the horizontal section the
        same way.
    radii
        ``(3,)`` semi-axes, applied after the shape exponents.
    u_resolution, v_resolution
        Number of samples along each parameter direction, including both ends of the domain. At
        least 2.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``, closed and watertight: the u direction wraps and both
        v extremes are poles.

    Raises
    ------
    ValueError
        If ``radii`` does not have shape ``(3,)``, if ``v_resolution`` is below 2, or if
        ``u_resolution`` is below 3 — the u direction wraps, and a resolution of 2 there would
        identify every cell's two rows and leave no faces at all.

    Notes
    -----
    VTK's ``vtkParametricSuperEllipsoid``, which pyvista exposes as ``pv.ParametricSuperEllipsoid``
    and ``pv.Superquadric``. The exponents apply through a signed power ``sign(x) |x| ** n``, so the
    surface stays symmetric about every coordinate plane.

    See Also
    --------
    [`uv_sphere`][triwarp.creation.uv_sphere]
    [`super_toroid`][triwarp.creation.super_toroid]
    [`parametric_surface`][triwarp.creation.parametric_surface]
    """
    radii_np = np.asanyarray(radii, dtype=np.float64)
    if radii_np.shape != (3,):
        raise ValueError(f"radii must be (3,) float, got {radii_np.shape}")
    vertices, faces = _parametric_surface(
        _SUPER_ELLIPSOID_SPEC, u_resolution, v_resolution, device, n1=n1, n2=n2
    )
    if not np.array_equal(radii_np, np.ones(3)):
        return _apply_transform(vertices, faces, wp.mat44(*np.diag([*radii_np, 1.0]).ravel()))
    return vertices, faces


def super_toroid(
    n1: float = 1.0,
    n2: float = 1.0,
    u_resolution: int = 40,
    v_resolution: int = 40,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a superquadric torus: a closed genus-1 surface with a squareness axis.

    The genus-1 counterpart of [`super_ellipsoid`][triwarp.creation.super_ellipsoid]. ``n1 = n2 =
    1`` is the ordinary torus of major radius 1 and minor radius 0.5, and the exponents square off
    the tube's cross-section (``n1``) and the ring (``n2``) independently.

    Parameters
    ----------
    n1
        Squareness exponent of the tube cross-section.
    n2
        Squareness exponent of the ring.
    u_resolution, v_resolution
        Number of samples along each parameter direction, including both ends of the domain. At
        least 2.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``, closed and watertight with Euler characteristic 0:
        both parameter directions wrap.

    Raises
    ------
    ValueError
        If either resolution is below 3. Both parameter directions wrap, and a resolution of 2
        would identify every cell's two rows and leave no faces at all.

    Notes
    -----
    VTK's ``vtkParametricSuperToroid``, which pyvista exposes as ``pv.ParametricSuperToroid``. The
    radii are VTK's and are not exposed: [`torus`][triwarp.creation.torus] is the builder for that
    axis, and this one exists for the exponents.

    See Also
    --------
    [`torus`][triwarp.creation.torus]
    [`super_ellipsoid`][triwarp.creation.super_ellipsoid]
    [`parametric_surface`][triwarp.creation.parametric_surface]
    """
    return _parametric_surface(_SUPER_TOROID_SPEC, u_resolution, v_resolution, device, n1=n1, n2=n2)


def random_hills(
    n_hills: int = 30,
    amplitude: float = 2.0,
    x_variance: float = 2.5,
    y_variance: float = 2.5,
    seed: int | None = None,
    u_resolution: int = 40,
    v_resolution: int = 40,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a smooth random height field over the square ``[-10, 10] ** 2``.

    A sum of ``n_hills`` Gaussian bumps at seeded random centres, sampled on a regular lattice: an
    open patch with one boundary loop, everywhere smooth, and with curvature that varies across it.
    That makes it the module's input for smoothing, curvature and remeshing, where the Platonic
    solids and the surfaces of revolution are too uniform to distinguish two implementations.

    Parameters
    ----------
    n_hills
        Number of Gaussian bumps summed into the height.
    amplitude
        Height of a single isolated bump. Overlapping bumps add.
    x_variance, y_variance
        Variances of each bump along X and Y, in the units of the ``[-10, 10]`` domain.
    seed
        RNG seed for the bump centres. A random one is drawn when omitted.
    u_resolution, v_resolution
        Number of samples along X and Y, including both ends of the domain. At least 2.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device`` with ``u_resolution * v_resolution`` vertices — nothing
        is identified — and ``2 * (u_resolution - 1) * (v_resolution - 1)`` triangles.

    Raises
    ------
    ValueError
        If either resolution is below 2, or either variance is not positive.

    Notes
    -----
    Named for VTK's ``vtkParametricRandomHills``, and it is the same idea over the same domain, but
    the heights are not comparable: VTK draws each hill's amplitude and variance from its own
    generator, where this takes them as parameters and draws only the centres. The seed goes
    through [`resolve_seed`][triwarp.sample.resolve_seed], so a given seed reproduces a given mesh.

    See Also
    --------
    [`grid`][triwarp.creation.grid]
    [`random_soup`][triwarp.creation.random_soup]
    [`parametric_surface`][triwarp.creation.parametric_surface]
    """
    if float(x_variance) <= 0.0 or float(y_variance) <= 0.0:
        raise ValueError(f"variances must be positive, got {(x_variance, y_variance)}")
    first, tables, faces = _parametric_samples(
        _RANDOM_HILLS_SPEC, u_resolution, v_resolution, device
    )
    generator = np.random.default_rng(tw.sample.resolve_seed(seed))
    centers = generator.uniform(
        low=(_RANDOM_HILLS_SPEC.u_range[0], _RANDOM_HILLS_SPEC.v_range[0]),
        high=(_RANDOM_HILLS_SPEC.u_range[1], _RANDOM_HILLS_SPEC.v_range[1]),
        size=(max(int(n_hills), 0), 2),
    )

    vertices = _launch.empty(first.size, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_creation.random_hills_vertices,
        dim=vertices.size,
        inputs=[
            wp.float32(amplitude),
            wp.float32(x_variance),
            wp.float32(y_variance),
            _launch.array(centers, dtype=wp.vec2, device=device),
            first,
            tables,
            wp.int32(int(v_resolution)),
            vertices,
        ],
        device=device,
    )
    return vertices, faces


def random_soup(
    face_count: int = 100, seed: int | None = None, device: wp.DeviceLike = None
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a soup of random, unconnected triangles inside the unit cube around the origin.

    Parameters
    ----------
    face_count
        Number of triangles. Each gets its own three vertices, so nothing is shared.
    seed
        RNG seed. A random one is drawn when omitted.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with ``3 * face_count`` vertices in ``[-0.5, 0.5] ** 3`` and
        ``faces == arange(3 * face_count)``.

    Notes
    -----
    Takes an explicit ``seed`` where [`trimesh.creation.random_soup`][] draws from NumPy's global
    RNG, so results are reproducible. The point values come from Warp's per-thread generator and do
    not match NumPy's stream.

    See Also
    --------
    [`trimesh.creation.random_soup`][]
    """
    n = 3 * max(int(face_count), 0)
    vertices = _launch.empty(n, dtype=wp.vec3, device=device)
    if n > 0:
        resolved_seed = tw.sample.resolve_seed(seed)
        _launch.launch(
            kernel_creation.random_soup_vertices,
            dim=n,
            inputs=[wp.int32(resolved_seed), vertices],
            device=device,
        )
    return vertices, tw.array.arange(n, device=vertices.device)


# --- private helpers ---------------------------------------------------------------------


class _ParametricSpec(NamedTuple):
    """
    One analytic surface: its kernel id, its parameter rectangle and how the rectangle glues.

    The gluing is what makes these surfaces interesting and it is *combinatorial* -- a fact about
    the map, not about how close two evaluated points happen to land. Every flag was derived from
    the map itself and checked against VTK's welded output; see
    [`parametric_surface`][triwarp.creation.parametric_surface].
    """

    kind: wp.int32
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    u_wrap: bool = False
    u_twist: bool = False
    v_wrap: bool = False
    v_twist: bool = False
    pole_v_min: bool = False
    pole_v_max: bool = False
    pole_u_min: bool = False
    pole_u_max: bool = False


_PARAMETRIC_SPECS: dict[str, _ParametricSpec] = {
    "bohemian_dome": _ParametricSpec(
        kernel_creation.SURFACE_BOHEMIAN_DOME,
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        u_wrap=True,
        v_wrap=True,
    ),
    "bour": _ParametricSpec(
        kernel_creation.SURFACE_BOUR, (0.0, 1.0), (0.0, 4.0 * math.pi), v_wrap=True, pole_u_min=True
    ),
    "boy": _ParametricSpec(
        kernel_creation.SURFACE_BOY,
        (0.0, math.pi),
        (0.0, math.pi),
        u_wrap=True,
        u_twist=True,
        v_wrap=True,
        pole_v_min=True,
        pole_v_max=True,
    ),
    "catalan_minimal": _ParametricSpec(
        kernel_creation.SURFACE_CATALAN_MINIMAL, (-4.0 * math.pi, 4.0 * math.pi), (-1.5, 1.5)
    ),
    "conic_spiral": _ParametricSpec(
        kernel_creation.SURFACE_CONIC_SPIRAL,
        (0.0, 2.0 * math.pi),
        (0.0, 2.0 * math.pi),
        u_wrap=True,
        pole_v_max=True,
    ),
    "cross_cap": _ParametricSpec(
        kernel_creation.SURFACE_CROSS_CAP,
        (0.0, math.pi),
        (0.0, math.pi),
        u_wrap=True,
        u_twist=True,
        v_wrap=True,
        pole_v_min=True,
        pole_v_max=True,
    ),
    "dini": _ParametricSpec(kernel_creation.SURFACE_DINI, (0.0, 4.0 * math.pi), (0.001, 2.0)),
    "enneper": _ParametricSpec(kernel_creation.SURFACE_ENNEPER, (-2.0, 2.0), (-2.0, 2.0)),
    "figure8_klein": _ParametricSpec(
        kernel_creation.SURFACE_FIGURE8_KLEIN,
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        u_wrap=True,
        u_twist=True,
        v_wrap=True,
    ),
    "henneberg": _ParametricSpec(
        kernel_creation.SURFACE_HENNEBERG,
        (-1.0, 1.0),
        (-0.5 * math.pi, 0.5 * math.pi),
        v_wrap=True,
        v_twist=True,
    ),
    "klein": _ParametricSpec(
        kernel_creation.SURFACE_KLEIN, (0.0, math.pi), (0.0, 2.0 * math.pi), v_wrap=True
    ),
    "kuen": _ParametricSpec(
        kernel_creation.SURFACE_KUEN, (-4.5, 4.5), (0.0, math.pi), pole_v_max=True
    ),
    "mobius": _ParametricSpec(
        kernel_creation.SURFACE_MOBIUS, (0.0, 2.0 * math.pi), (-1.0, 1.0), u_wrap=True, u_twist=True
    ),
    "plucker_conoid": _ParametricSpec(
        kernel_creation.SURFACE_PLUCKER_CONOID, (0.0, 3.0), (0.0, 2.0 * math.pi), v_wrap=True
    ),
    "pseudosphere": _ParametricSpec(
        kernel_creation.SURFACE_PSEUDOSPHERE, (-5.0, 5.0), (-math.pi, math.pi), v_wrap=True
    ),
    "roman": _ParametricSpec(
        kernel_creation.SURFACE_ROMAN,
        (0.0, math.pi),
        (0.0, math.pi),
        u_wrap=True,
        u_twist=True,
        v_wrap=True,
    ),
}

# The two superquadrics and the height field, whose shape parameters earn them their own builders.
_SUPER_ELLIPSOID_SPEC = _ParametricSpec(
    kernel_creation.SURFACE_SUPER_ELLIPSOID,
    (-math.pi, math.pi),
    (-0.5 * math.pi, 0.5 * math.pi),
    u_wrap=True,
    pole_v_min=True,
    pole_v_max=True,
)
_SUPER_TOROID_SPEC = _ParametricSpec(
    kernel_creation.SURFACE_SUPER_TOROID,
    (0.0, 2.0 * math.pi),
    (0.0, 2.0 * math.pi),
    u_wrap=True,
    v_wrap=True,
)
# The height field has its own kernel -- it sums a table of bumps rather than evaluating a closed
# form -- so its ``kind`` selects nothing and the lattice is a plain grid.
_RANDOM_HILLS_SPEC = _ParametricSpec(wp.int32(-1), (-10.0, 10.0), (-10.0, 10.0))

# Lattice samples at or above which the device lattice beats the numpy one. The device path costs a
# flat floor of launches, allocations and two readbacks whatever the resolution; the host path is
# quadratic in it, and the two cross here.
_PARAMETRIC_LATTICE_DEVICE_FROM = 625

# Longest profile ``_revolve_regular`` will screen. The screen is ``O(P)`` in the profile length
# while the fast path's win is flat in it -- the general engine's layout tables, three uploads and
# second launch -- so the bet is symmetric at a short profile and lopsided at a long one. A decline
# is not a rare path either: a fine profile revolved into many sections makes the polar triangles
# smaller than ``revolve``'s absolute area tolerance, which drops more of them than the regular
# layout expects, and that is exactly the large-``P`` regime. At this cap the fast path still wins
# and the downside is bounded; at twice it every shape probed declined. Above the cap the general
# engine runs unscreened.
_REVOLVE_REGULAR_MAX_PROFILE = 2048

# Longest profile whose degenerate-triangle screen runs in Python floats rather than NumPy: two
# template triangles per segment, so the scalar loop grows with the profile while the array form's
# cost is flat per-call overhead until the arrays get long.
_KEPT_TEMPLATE_LOOP_MAX = 64


def _parametric_surface(
    spec: _ParametricSpec,
    u_resolution: int,
    v_resolution: int,
    device: wp.DeviceLike,
    n1: float = 1.0,
    n2: float = 1.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Sample one analytic surface on its identified lattice, for the four public builders."""
    # One output per *vertex*, carrying the one lattice sample chosen to represent it, never one per
    # lattice sample: where the surface glues, several lattice samples map to the same vertex, and
    # evaluating only the canonical one keeps the result bit-exact run to run -- mapping over the
    # lattice and letting the identified samples race for the slot would not (a twisted seam and a
    # collapsed pole row reach the same point through different expressions, so they agree only to
    # rounding).
    first, tables, faces = _parametric_samples(spec, u_resolution, v_resolution, device)
    vertices = _launch.empty(first.size, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_creation.parametric_vertices,
        dim=first.size,
        inputs=[
            spec.kind,
            first,
            tables,
            wp.int32(int(v_resolution)),
            wp.float32(n1),
            wp.float32(n2),
            vertices,
        ],
        device=device,
    )
    return vertices, faces


def _parametric_samples(
    spec: _ParametricSpec, u_resolution: int, v_resolution: int, device: wp.DeviceLike
) -> tuple[wp.array[wp.int32], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    ``(first, tables, faces)`` for one surface's lattice.

    ``first`` is the flat lattice index of the sample representing each output vertex, ``tables``
    the ``u`` parameter table followed by the ``v`` one, and ``faces`` the flat triangle buffer.
    The parameter *tables* are built here rather than in the kernel so that both ends of the domain
    are hit exactly: several of these maps are singular one ulp outside their rectangle. They are
    length ``n_u`` and ``n_v``, not one entry per sample, so the gather that spreads them over the
    lattice happens on the device (``kernels/creation.parametric_sample``).
    """
    n_u, n_v = int(u_resolution), int(v_resolution)
    if n_u < 2 or n_v < 2:
        raise ValueError(f"resolutions must be at least 2, got {(n_u, n_v)}")
    # A wrapped, untwisted axis identifies its last row with its first, so at resolution 2 every
    # cell along it has two equal corners and the whole face buffer is filtered away as degenerate
    # -- a vertex-only mesh, which is silently wrong rather than merely coarse (and reaches
    # `wp.Mesh` as a zero-triangle build). A twist glues the seam with a flip, which keeps the two
    # rows distinct, so the floor of 3 applies only to the untwisted case.
    if spec.u_wrap and not spec.u_twist and n_u < 3:
        raise ValueError(f"u_resolution must be at least 3 on a wrapped axis, got {n_u}")
    if spec.v_wrap and not spec.v_twist and n_v < 3:
        raise ValueError(f"v_resolution must be at least 3 on a wrapped axis, got {n_v}")
    # The lattice is a closed-form parallel map, so the device wins it outright once there is
    # enough of it -- and loses below that to its own launch and readback floor, which is flat where
    # the host cost is quadratic in the resolution. Both paths produce byte-identical vertices and
    # faces, so the gate is purely a cost choice; see ``_PARAMETRIC_LATTICE_DEVICE_FROM``.
    tables = _launch.array(
        np.concatenate((np.linspace(*spec.u_range, n_u), np.linspace(*spec.v_range, n_v))),
        dtype=wp.float32,
        device=device,
    )
    if n_u * n_v >= _PARAMETRIC_LATTICE_DEVICE_FROM:
        first, faces = _parametric_lattice_device(spec, n_u, n_v, device)
        return first, tables, faces

    first_np, faces_np = _parametric_lattice_host(spec, n_u, n_v)
    return (
        _launch.array(first_np, dtype=wp.int32, device=device),
        tables,
        _launch.array(faces_np, dtype=wp.int32, device=device),
    )


def _parametric_lattice_host(
    spec: _ParametricSpec, n_u: int, n_v: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Choose one lattice sample per output vertex, and build the face buffer, for one surface.

    Returns ``(first, faces)`` where ``first`` is the flat lattice index of the sample
    representing each output vertex -- several samples land on one vertex wherever the
    surface glues, and evaluating only the representative keeps the result bit-exact -- and
    ``faces`` is triwarp's flat triangle buffer.

    The small-lattice half of the dispatch in
    [`_parametric_samples`][triwarp.creation._parametric_samples]; its device twin is
    ``_parametric_lattice_device``, which produces the identical answer and wins above
    ``_PARAMETRIC_LATTICE_DEVICE_FROM`` samples. Pure index arithmetic, in the same spirit as
    [`grid`][triwarp.creation.grid]: no position is consulted and no tolerance appears anywhere, so
    the topology is exact and independent of resolution, dtype and device. Welding by *distance*
    instead would (a) make every one of these meshes depend on
    [`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices], which several of them
    exist to test, and (b) glue the accidental self-intersections of an immersed surface -- which is
    what VTK does, merging 40 lattice points of Catalan's minimal surface that the map does not
    identify, because two sheets happen to cross there.
    """
    if spec.u_twist and spec.v_twist:
        raise ValueError("a surface twisted in both directions is not supported")
    i_lattice, j_lattice = np.meshgrid(np.arange(n_u), np.arange(n_v), indexing="ij")
    i_canonical, j_canonical = i_lattice.copy(), j_lattice.copy()

    # Wrapped boundary: the last row *is* the first row. A twist reverses the other index on the way
    # across, which is precisely what makes Moebius, Klein, Boy, Roman and the cross-cap
    # non-orientable -- the seam glues the strip to itself with a flip.
    if spec.u_wrap:
        seam = i_canonical == n_u - 1
        if spec.u_twist:
            j_canonical = np.where(seam, n_v - 1 - j_canonical, j_canonical)
        i_canonical = np.where(seam, 0, i_canonical)
    if spec.v_wrap:
        seam = j_canonical == n_v - 1
        if spec.v_twist:
            i_canonical = np.where(seam, n_u - 1 - i_canonical, i_canonical)
        j_canonical = np.where(seam, 0, j_canonical)
        # A v-twist can send an index back onto the u seam, so re-canonicalise it.
        if spec.u_wrap:
            i_canonical = np.where(i_canonical == n_u - 1, 0, i_canonical)

    # A pole is a boundary row the map collapses to a single point, so every sample on it is one
    # vertex. ``v_min`` / ``v_max`` are rows spanning u, ``u_min`` / ``u_max`` columns spanning v.
    # A *wrapped* boundary has already identified its two extreme rows, so a pole on either of them
    # is one pole on the surviving row -- which is why Boy and the cross-cap keep 1 483 vertices
    # rather than losing a second row's worth.
    poles = []
    if spec.v_wrap:
        if spec.pole_v_min or spec.pole_v_max:
            poles.append(j_canonical == 0)
    else:
        if spec.pole_v_min:
            poles.append(j_canonical == 0)
        if spec.pole_v_max:
            poles.append(j_canonical == n_v - 1)
    if spec.u_wrap:
        if spec.pole_u_min or spec.pole_u_max:
            poles.append(i_canonical == 0)
    else:
        if spec.pole_u_min:
            poles.append(i_canonical == 0)
        if spec.pole_u_max:
            poles.append(i_canonical == n_u - 1)
    for on_row in poles:
        anchor_i, anchor_j = int(i_canonical[on_row][0]), int(j_canonical[on_row][0])
        i_canonical = np.where(on_row, anchor_i, i_canonical)
        j_canonical = np.where(on_row, anchor_j, j_canonical)

    _unique, first, inverse = np.unique(
        i_canonical * n_v + j_canonical, return_index=True, return_inverse=True
    )
    vertex_index = inverse.reshape((n_u, n_v)).astype(np.int32)

    # One cell per lattice square -- wrapping reuses vertices rather than adding cells, so the count
    # is ``(n_u - 1) * (n_v - 1)`` however the boundary glues. A cell touching a pole has two
    # identical corners, so it contributes one triangle instead of two.
    corner_a = vertex_index[:-1, :-1].ravel()
    corner_b = vertex_index[1:, :-1].ravel()
    corner_c = vertex_index[1:, 1:].ravel()
    corner_d = vertex_index[:-1, 1:].ravel()
    # Wound against the (u, v) frame, matching VTK's convention, which puts the normals of the
    # closed surfaces outward.
    triangles = np.concatenate(
        (
            np.column_stack((corner_a, corner_c, corner_b)),
            np.column_stack((corner_a, corner_d, corner_c)),
        )
    )
    nondegenerate = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 2] != triangles[:, 0])
    )
    faces_np = np.ascontiguousarray(triangles[nondegenerate].reshape(-1), dtype=np.int32)
    return first.astype(np.int32), faces_np


def _parametric_lattice_device(
    spec: _ParametricSpec, n_u: int, n_v: int, device: wp.DeviceLike
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Choose one lattice sample per output vertex, and build the face buffer, for one surface.

    Returns ``(first, faces)`` where ``first`` is the flat lattice index of the sample representing
    each output vertex -- several samples land on one vertex wherever the surface glues, and
    evaluating only the representative keeps the result bit-exact -- and ``faces`` is triwarp's flat
    triangle buffer.

    The large-lattice half of the dispatch in
    [`_parametric_samples`][triwarp.creation._parametric_samples]; ``_parametric_lattice_host``
    is its numpy twin and returns the identical answer. Pure index arithmetic, in the same spirit
    as [`grid`][triwarp.creation.grid]: no position is consulted and no tolerance appears anywhere,
    so the topology is exact and independent of resolution, dtype and device. Welding by *distance*
    instead would (a) make every one of these meshes depend on
    [`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices], which several of them
    exist to test, and (b) glue the accidental self-intersections of an immersed surface -- which is
    what VTK does, merging 40 lattice points of Catalan's minimal surface that the map does not
    identify, because two sheets happen to cross there.
    """
    if spec.u_twist and spec.v_twist:
        raise ValueError("a surface twisted in both directions is not supported")

    # A pole is a boundary row the map collapses to a single point. A *wrapped* boundary has
    # already identified its two extreme rows, so a pole on either of them is one pole on the
    # surviving row -- which is why Boy and the cross-cap keep 1 483 vertices rather than losing a
    # second row's worth. The kernel takes the four resolved masks, not the eight spec flags.
    pole_j_lo = (spec.pole_v_min or spec.pole_v_max) if spec.v_wrap else spec.pole_v_min
    pole_j_hi = False if spec.v_wrap else spec.pole_v_max
    pole_i_lo = (spec.pole_u_min or spec.pole_u_max) if spec.u_wrap else spec.pole_u_min
    pole_i_hi = False if spec.u_wrap else spec.pole_u_max
    gluing = 0
    for on, flag in (
        (spec.u_wrap, kernel_creation.GLUE_U_WRAP),
        (spec.u_twist, kernel_creation.GLUE_U_TWIST),
        (spec.v_wrap, kernel_creation.GLUE_V_WRAP),
        (spec.v_twist, kernel_creation.GLUE_V_TWIST),
        (pole_j_lo, kernel_creation.GLUE_POLE_J_LO),
        (pole_j_hi, kernel_creation.GLUE_POLE_J_HI),
        (pole_i_lo, kernel_creation.GLUE_POLE_I_LO),
        (pole_i_hi, kernel_creation.GLUE_POLE_I_HI),
    ):
        if on:
            gluing |= int(flag)

    # One cell per lattice square -- wrapping reuses vertices rather than adding cells, so the count
    # is ``(n_u - 1) * (n_v - 1)`` however the boundary glues. A cell touching a pole has two
    # identical corners, so it contributes one triangle instead of two.
    n_lattice = n_u * n_v
    n_rows = n_lattice + 2 * (n_u - 1) * (n_v - 1)
    # The lattice's canonical keys marked, then each triangle's survival, in one buffer and one
    # inclusive scan: the lattice prefix numbers the vertices and the triangle tail ranks the
    # survivors. The two counts are the only readbacks, and they size the outputs.
    scan = _launch.zeros(n_rows, dtype=wp.int32, device=device)
    inputs = [wp.int32(n_u), wp.int32(n_v), wp.int32(gluing)]
    _launch.launch(
        kernel_creation.parametric_lattice_flags, dim=n_rows, inputs=[*inputs, scan], device=device
    )
    _launch.array_scan(scan, out_array=scan, inclusive=True)
    n_vertices = int(read_scalar(scan, n_lattice - 1))
    n_faces = int(read_scalar(scan, n_rows - 1)) - n_vertices

    first = _launch.full(n_vertices, INT32_MAX, dtype=wp.int32, device=device)
    faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_creation.parametric_lattice_emit,
        dim=n_rows,
        inputs=[*inputs, scan, first, faces],
        device=device,
    )
    return first, faces


def _resolve_cylinder_axis(
    height: float | None,
    segment: Sequence[Sequence[float]] | None,
    transform: wp.mat44 | wp.array[wp.mat44] | None,
) -> tuple[wp.mat44 | wp.array[wp.mat44] | None, float]:
    """
    Resolve a cylinder-like body's axis to ``(transform, half_height)``.

    ``segment`` and ``height`` are two spellings of the same axis, shared by
    [`cylinder`][triwarp.creation.cylinder] and [`annulus`][triwarp.creation.annulus]. A segment
    fixes the placement as well as the length, so it supersedes ``transform``.

    Raises
    ------
    ValueError
        If neither ``height`` nor ``segment`` is given, or ``segment`` is not ``(2, 3)``.
    """
    if segment is not None:
        transform, height = _segment_to_cylinder(segment)
    if height is None:
        raise ValueError("either height or segment must be passed")
    return transform, abs(float(height)) / 2.0


def _segment_to_cylinder(segment: Sequence[Sequence[float]]) -> tuple[wp.mat44, float]:
    """Convert a 3D line segment to the transform and height of a Z-extruded origin cylinder."""
    segment_np = np.asanyarray(segment, dtype=np.float64)
    if segment_np.shape != (2, 3):
        raise ValueError(f"segment must be (2, 3) float, got {segment_np.shape}")
    vector = segment_np[1] - segment_np[0]
    height = float(np.linalg.norm(vector))
    matrix = _align_vectors(np.array([0.0, 0.0, 1.0]), vector)
    # Compose translation-to-midpoint with the rotation.
    matrix[:3, 3] = segment_np[0] + vector * 0.5
    return wp.mat44(*matrix.flatten()), height


def _resolve_sections(sections: int | None) -> int:
    """Wedge count of a *closed* revolution, defaulted and validated as ``revolve`` does."""
    n_sections = DEFAULT_SECTIONS if sections is None else int(sections)
    if n_sections < 1:
        raise ValueError(f"sections must be at least 1, got {sections}")
    return n_sections


def _revolve_regular(
    profile_np: np.ndarray,
    sections: int,
    transform: wp.mat44 | wp.array[wp.mat44] | None,
    device: wp.DeviceLike,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]] | None:
    """
    Build a solid of revolution in one launch, or return ``None`` when the layout is not regular.

    ``revolve`` is general over an arbitrary profile and decides its layout on the host: it reads
    the profile back off the device, builds a per-column slot table and a surviving-triangle list
    in NumPy, and uploads three more arrays. A profile whose columns are all full rings except an
    on-axis point at one or both ends does not need any of that — the slots and the face blocks are
    arithmetic — which covers every closed solid this module builds by revolution.

    **Regularity is checked against ``revolve``'s own filter, not asserted.** The surviving template
    triangles must be exactly "both, except the one touching an axis end", which is what makes the
    two engines agree; anything else (a profile with an interior on-axis point, a duplicated
    interior column, geometry small enough for the absolute area tolerance to bite in the middle)
    returns ``None`` and the caller falls back. See ``kernels/creation.revolve_uniform``.

    ``profile_np`` is the caller's own host profile, closed profiles included: the repeated last
    point is detected here and dropped, so the kernel addresses deduplicated columns.
    """
    if sections < 1 or not 2 <= profile_np.shape[0] <= _REVOLVE_REGULAR_MAX_PROFILE:
        return None
    wrap = bool(np.array_equal(profile_np[0], profile_np[-1]))
    columns_np = profile_np[:-1] if wrap else profile_np
    n_columns = int(columns_np.shape[0])
    if n_columns < 2:
        return None
    on_axis = columns_np[:, 0] == 0.0
    # An interior on-axis column would collapse a whole ring into one slot and break the
    # arithmetic; only the ends are handled.
    if wrap and bool(on_axis.any()):
        return None
    if bool(on_axis[1:-1].any()):
        return None
    axis_first, axis_last = bool(on_axis[0]), bool(on_axis[-1])

    n_segments = n_columns - 1 + int(wrap)
    if n_segments < 1:
        return None
    # ``revolve``'s verdict, on ``revolve``'s template, for exactly this profile and step. The
    # template holds two triangles per segment in segment order, so the regular layout is a mask
    # over ``2 * n_segments`` and the comparison is an array equality rather than a set one --
    # both sides come out of ``np.flatnonzero``, so both are sorted and the orders agree. Built
    # with NumPy rather than a Python loop because this runs on every call and the profile length
    # is the axis that grows -- on a fine profile the loop and its two sets dominated the screen.
    kept_np = _revolve_kept_template(profile_np, 2.0 * math.pi / float(sections))
    segment_np = np.arange(n_segments)
    next_np = (segment_np + 1) % n_columns
    expected_np = np.empty(2 * n_segments, dtype=bool)
    expected_np[0::2] = ~((segment_np == 0) & axis_first) & ~(
        (segment_np == n_columns - 1) & axis_last
    )
    expected_np[1::2] = ~((next_np == 0) & axis_first) & ~((next_np == n_columns - 1) & axis_last)
    if not np.array_equal(kept_np, np.flatnonzero(expected_np).astype(np.int32)):
        return None

    # Only the two end segments can be short, so the kernel finds any later segment's place from
    # the first one's count alone.
    first_segment_faces = int(expected_np[0]) + int(expected_np[1])
    faces_per_slice = int(expected_np.sum())
    n_vertices = (
        int(axis_first) + (n_columns - int(axis_first) - int(axis_last)) * sections + int(axis_last)
    )
    vertices = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    faces = _launch.empty(faces_per_slice * sections * 3, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_creation.revolve_uniform,
        dim=(sections, n_columns),
        inputs=[
            _upload_points(columns_np, wp.vec2, device),
            wp.int32(sections),
            _FULL_TURN,
            wp.int32(axis_first),
            wp.int32(axis_last),
            wp.int32(wrap),
            wp.int32(first_segment_faces),
            wp.int32(faces_per_slice),
        ],
        outputs=[vertices, faces],
        device=device,
    )
    return _apply_transform(vertices, faces, transform)


def _closed_form_template(profile_np: np.ndarray, n_sections: int) -> tuple[int, int]:
    """
    ``(keep_caps, keep_sides)``: which of a solid-of-revolution's template triangles survive.

    The closed-form kernels behind [`cone`][triwarp.creation.cone] and
    [`cylinder`][triwarp.creation.cylinder] write a fixed layout, so they need
    [`revolve`][triwarp.creation.revolve]'s degenerate-area verdict as two flags rather than as an
    index list. It is taken from ``revolve``'s own filter on ``revolve``'s own template — not
    re-derived — so the two engines cannot disagree about which triangles collapse: at one or two
    sections the triangles touching the axis have zero area, and on geometry small enough for the
    absolute tolerance to bite they do too.

    The template is two triangles per profile segment, in segment order. Index 1 is the first
    segment's surviving triangle (the cap) and index 2 the second's (the side); a cylinder's second
    cap and second side are the same shapes at the same radius and step, so they share the verdict.
    """
    kept = set(_revolve_kept_template(profile_np, 2.0 * math.pi / float(n_sections)).tolist())
    return int(1 in kept), int(2 in kept)


def _align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Homogeneous rotation taking unit vector ``a`` onto ``b`` (``trimesh.geometry`` port)."""
    au = np.linalg.svd(a.reshape((-1, 1)))[0]
    bu = np.linalg.svd(b.reshape((-1, 1)))[0]
    if np.linalg.det(au) < 0:
        au[:, -1] *= -1.0
    if np.linalg.det(bu) < 0:
        bu[:, -1] *= -1.0
    matrix = np.eye(4)
    matrix[:3, :3] = bu.dot(au.T)
    return matrix


def _apply_transform(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    transform: wp.mat44 | wp.array[wp.mat44] | None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Transform ``vertices`` in place, reversing face winding when the transform is a mirror.

    In place is right *here* and nowhere else in the package: every caller has just built the
    buffer it hands over, so there is no second holder to corrupt and no cache to invalidate --
    the opposite of [`Trimesh.transform`][triwarp.mesh.Trimesh.transform], which is functional for
    exactly that reason.
    """
    if transform is None:
        return vertices, faces
    return tw.transform.transform_mesh(
        vertices, faces, transform, out_vertices=vertices, out_faces=faces
    )


def _upload_points(points_np: np.ndarray, dtype: type, device: wp.DeviceLike) -> twt.ArrayNd:
    """
    Upload an ``(n, 2)`` or ``(n, 3)`` host point table as a ``wp.vec2`` / ``wp.vec3`` buffer.

    Vertex tables and revolution profiles are both built on the host in ``float64`` and downcast
    here. Evaluating a profile on the device in ``float32`` instead would put the closing point of
    a closed profile roughly ``1e-7`` away from its first point — outside
    [`revolve`][triwarp.creation.revolve]'s weld tolerance — while ``float64`` keeps the gap near
    ``1e-16``. These tables are tiny; the per-slice work they drive is what runs on the device.
    """
    return _launch.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=dtype, device=device
    )
