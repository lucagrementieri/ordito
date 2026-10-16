"""
Parametric mesh generators: boxes, Platonic solids, surfaces of revolution and extrusions.

Every function returns a bare ``(vertices, faces)`` pair using ordito's flat ``(3 * n_faces,)``
face layout, so results feed straight into the rest of the package or into
[`Trimesh`][ordito.mesh.Trimesh] via ``od.Trimesh(*od.creation.box())``.

[`revolve`][ordito.creation.revolve] is the engine behind most of the module:
[`uv_sphere`][ordito.creation.uv_sphere], [`capsule`][ordito.creation.capsule],
[`cylinder`][ordito.creation.cylinder], [`cone`][ordito.creation.cone],
[`annulus`][ordito.creation.annulus] and [`torus`][ordito.creation.torus] each build a small 2D
profile and sweep it around the Z axis. [`box`][ordito.creation.box] and the four Platonic solids
([`tetrahedron`][ordito.creation.tetrahedron], [`octahedron`][ordito.creation.octahedron],
[`icosahedron`][ordito.creation.icosahedron], [`dodecahedron`][ordito.creation.dodecahedron]) are
constant tables, and [`icosphere`][ordito.creation.icosphere] refines one of them in closed form.
[`grid`][ordito.creation.grid] and
[`sphere_cap`][ordito.creation.sphere_cap] are the two *open* primitives — a flat patch and a
curved one, each with exactly one boundary loop.

[`parametric_surface`][ordito.creation.parametric_surface] and its three named siblings
([`super_ellipsoid`][ordito.creation.super_ellipsoid],
[`super_toroid`][ordito.creation.super_toroid],
[`random_hills`][ordito.creation.random_hills]) sample an analytic map on an identified lattice,
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

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, require_same_device
from ordito.constants import TILE_1D, TOLERANCE_MERGE
from ordito.kernels import array as kernel_array
from ordito.kernels import creation as kernel_creation
from ordito.kernels import reduce as kernel_reduce

# Default number of pie wedges per full revolution, matching trimesh.
DEFAULT_SECTIONS = 32

# Absolute tolerance trimesh uses to decide that a revolution angle closes the loop.
_CLOSED_ANGLE_ATOL = 1e-10

# One full revolution, as the ``wp.float32`` the revolution kernels take. Named so the closed-form
# solids and ``revolve`` pass bit-identical spans: the angle divides into the slice fraction, so a
# difference here would move every ring vertex in the last bits.
_FULL_TURN = wp.float32(2.0 * math.pi)
# The matrix a generator kernel receives when the caller passed no transform (it is not applied).
_IDENTITY_MATRIX = wp.mat44(*np.eye(4).ravel())

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
    # Sides ``a-b``, ``b-c``, ``c-a`` of every face, row-major, so the flat order is the order the
    # sides are first met in.
    starts_np = faces_np
    ends_np = np.roll(faces_np, -1, axis=1)
    radix = _ICOSAHEDRON_VERTICES.shape[0]
    keys_np = (np.minimum(starts_np, ends_np) * radix + np.maximum(starts_np, ends_np)).ravel()
    _, first_np, inverse_np = np.unique(keys_np, return_index=True, return_inverse=True)
    # ``np.unique`` numbers the keys in sorted order; renumber them by first appearance.
    rank_np = np.empty_like(first_np)
    rank_np[np.argsort(first_np)] = np.arange(first_np.size)
    edge_ids_np = rank_np[inverse_np].reshape(faces_np.shape)
    return np.hstack((faces_np, edge_ids_np, starts_np > ends_np)).astype(np.int32)


# Built once: the icosahedron's topology is a constant, so the only per-call cost is the upload.
_ICOSPHERE_FACE_TABLE = _icosphere_face_table()

# Refinement levels the first icosphere launch descends at once. A thread's work grows with the
# depth it descends while a launch costs a fixed host overhead, so the coarse levels, whose
# lattices are small, share one launch and every finer level gets its own, one level deep.
_ICOSPHERE_LEVELS_PER_LAUNCH = 6
# Refinement levels of ``axis``'s origin ball: ``icosphere``'s default, and a single launch, so
# the marker transform can be applied as the ball is written.
_AXIS_BALL_LEVELS = 3

# Deliberately NOT cached per device. ``box``, ``tetrahedron``, ``octahedron``, ``icosahedron``
# and ``dodecahedron`` *return* the buffer they upload, so a cached table would have to be
# ``wp.clone``-d on read, which gives the upload straight back -- a bad trade for a module-level
# dict pinning device memory for the life of the process.


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
    transform: wp.mat44 | None = None,
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
        Transform applied after construction, as a ``wp.mat44``. Face winding is reversed when the
        transform has negative determinant, so normals keep pointing outward.
    bounds
        ``(2, 3)`` axis-aligned corners. Overrides ``extents`` and ``transform``, and yields a
        box spanning exactly those corners rather than one centered on the origin.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(8,)`` and ``(36,)`` arrays ``(vertices, faces)``: 8 vertices and 36 flat face indices.

    Raises
    ------
    ValueError
        If ``bounds`` is combined with ``extents`` or ``transform``, or if either has the wrong
        shape.

    See Also
    --------
    [`icosahedron`][ordito.creation.icosahedron]
    [`cylinder`][ordito.creation.cylinder]
    [`trimesh.creation.box`][]

    Examples
    --------
    A unit cube centered on the origin, and a box given by its corners:

    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    print(vertices.shape, faces.shape)  # (8,) (36,)
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 1.0000
    vertices, faces = od.creation.box(bounds=[[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 24.0000
    ```
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

    faces_np = _BOX_FACES
    if transform is not None:
        # Eight corners: placed on the host, in ``float64``, before the one upload.
        matrix_np = od.transform.matrix_to_numpy(transform).astype(np.float64)
        vertices_np = vertices_np @ matrix_np[:3, :3].T + matrix_np[:3, 3]
        if np.linalg.det(matrix_np[:3, :3]) < 0.0:
            faces_np = np.ascontiguousarray(faces_np.reshape(-1, 3)[:, ::-1].reshape(-1))
    vertices = _upload_points(vertices_np, wp.vec3, device)
    faces = _launch.array(faces_np, dtype=wp.int32, device=device)
    return vertices, faces


def grid(
    count: tuple[int, int] = (10, 10),
    extents: tuple[float, float] = (1.0, 1.0),
    center: bool = True,
    transform: wp.mat44 | None = None,
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
        Transform applied after construction, as a ``wp.mat44``. Face winding is reversed when the
        transform has negative determinant.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(nx * ny,)`` and ``(6 * (nx - 1) * (ny - 1),)`` arrays ``(vertices, faces)`` on
        ``device``, wound so the normals point along ``+Z``.

    Raises
    ------
    ValueError
        If either entry of ``count`` is less than 2, or either entry of ``extents`` is negative.

    See Also
    --------
    [`box`][ordito.creation.box]
    [`extrude_triangulation`][ordito.creation.extrude_triangulation]

    Notes
    -----
    Both buffers are written **closed-form on the device**, in one launch of one thread per vertex
    that also writes the quad cell it is the lower corner of, rather than assembled on the host.
    Positions come out **bit-identical** to a NumPy build: the kernel does the same arithmetic in
    ``float64`` before the ``float32`` store, and phrases each sample as ``extent * (k / (n - 1))``
    so the far edge lands on the extent exactly, matching what ``numpy.linspace`` needs its
    endpoint special case for.

    Examples
    --------
    A 4 x 3 vertex patch: 6 quad cells, 12 triangles, one boundary loop through all 10 rim vertices:

    ```python
    import ordito as od

    vertices, faces = od.creation.grid(count=(4, 3), extents=(3.0, 2.0))
    print(vertices.shape, faces.size // 3)  # (12,) 12
    print(od.boundary.boundary_vertex_indices(vertices, faces).size)  # 10
    print(vertices.numpy()[0])  # [-1.5 -1.   0. ]
    ```
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
            *_placement(transform),
        ],
        outputs=[vertices, faces],
        device=device,
    )
    return vertices, faces


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
        ``(12,)`` and ``(60,)`` arrays ``(vertices, faces)``: 12 vertices and 20 triangles.

    See Also
    --------
    [`icosphere`][ordito.creation.icosphere]
    [`trimesh.creation.icosahedron`][]

    Examples
    --------
    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.icosahedron()
    print(vertices.shape, faces.size // 3)  # (12,) 20
    print(np.allclose(np.linalg.norm(vertices.numpy(), axis=1), 1.0))  # True
    ```
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
        ``(4,)`` and ``(12,)`` arrays ``(vertices, faces)``: 4 vertices and 4 triangles.

    See Also
    --------
    [`octahedron`][ordito.creation.octahedron]
    [`icosahedron`][ordito.creation.icosahedron]

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.tetrahedron()
    print(vertices.shape, faces.size // 3)  # (4,) 4
    print(od.measures.euler_characteristic(faces))  # 2
    ```
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
        ``(6,)`` and ``(24,)`` arrays ``(vertices, faces)``: 6 vertices and 8 triangles.

    See Also
    --------
    [`tetrahedron`][ordito.creation.tetrahedron]
    [`icosahedron`][ordito.creation.icosahedron]

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.octahedron()
    print(vertices.shape, faces.size // 3)  # (6,) 8
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 1.3333
    ```
    """
    return (
        _upload_points(_OCTAHEDRON_VERTICES, wp.vec3, device),
        _launch.array(_OCTAHEDRON_FACES, dtype=wp.int32, device=device),
    )


def dodecahedron(device: wp.DeviceLike = None) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a regular dodecahedron on the unit sphere, centered on the origin.

    ordito is triangle-only, so the twelve pentagonal faces arrive triangulated: each pentagon is
    fanned from one of the eight cube-corner vertices, giving 36 triangles over 20 vertices with no
    added centroid. The fan is MeshLab's, so this is vertex- and face-for-face identical to
    ``create_dodecahedron`` up to the uniform scaling onto the unit sphere.

    Because the triangulation is a fan rather than a symmetric split, the triangles are *not*
    congruent and the mesh is not a good uniform-sampling proxy — use
    [`icosphere`][ordito.creation.icosphere] for that.

    Parameters
    ----------
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(20,)`` and ``(108,)`` arrays ``(vertices, faces)``: 20 vertices and 36 triangles.

    See Also
    --------
    [`icosahedron`][ordito.creation.icosahedron]
    [`octahedron`][ordito.creation.octahedron]

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.dodecahedron()
    print(vertices.shape, faces.size // 3)  # (20,) 36
    print(od.measures.euler_characteristic(faces))  # 2
    ```
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
    uniform than [`uv_sphere`][ordito.creation.uv_sphere]'s, at roughly an order of magnitude
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
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on ``device``.

    Notes
    -----
    The connectivity is generated in closed form rather than by iterating
    [`subdivide`][ordito.remesh.subdivide]: every vertex of the refined mesh is addressed directly
    by its barycentric coordinates within one of the 20 base faces (see
    ``kernels.creation.icosphere_vertex_index``), so the face buffer is written in closed form and
    the whole mesh by one kernel, with no host synchronization: one launch up to six levels, then
    one more per level.

    The *geometry* is still the recursive one, because that is what the reference produces: a
    vertex is the projected midpoint of two vertices of the previous level, which is not the same
    point as the projection of the corresponding barycentric point of the base face (the two
    differ by a few percent of the edge length). Each vertex is computed on its own by descending
    the refinement from its base face, keeping at every level the child triangle that contains it,
    which reproduces the recursive build bit for bit. Vertex *order* differs from trimesh —
    vertices come out as the 12 base corners, then the interior points of each base edge, then the
    interior points of each base face.

    See Also
    --------
    [`icosahedron`][ordito.creation.icosahedron]
    [`uv_sphere`][ordito.creation.uv_sphere]
    [`subdivide`][ordito.remesh.subdivide]
    [`trimesh.creation.icosphere`][]

    Examples
    --------
    Two levels of refinement give ``20 * 4 ** 2`` faces, with an area approaching ``4 * pi``:

    ```python
    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    print(vertices.shape, faces.size // 3)  # (162,) 320
    _, areas = od.triangles.face_normals_and_areas(vertices, faces)
    print(f"{areas.numpy().sum():.3f}")  # 12.330
    ```
    """
    levels = max(0, int(subdivisions))
    n_vertices, n_faces = _icosphere_sizes(levels)
    device = wp.get_device(device)
    vertices = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    _icosphere_into(levels, float(radius), None, vertices, faces)
    return vertices, faces


def _icosphere_sizes(levels: int) -> tuple[int, int]:
    """``(n_vertices, n_faces)`` of an icosphere refined ``levels`` times."""
    return 10 * 4**levels + 2, 20 * 4**levels


def _icosphere_into(
    levels: int,
    radius: float,
    transform: wp.mat44 | None,
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
) -> None:
    """
    Write an icosphere into ``vertices`` and ``faces``, placed by ``transform``.

    A placement is applied at write time, which only a single-launch build allows (a later launch
    reads the coarser vertices back), so ``transform`` requires at most
    ``_ICOSPHERE_LEVELS_PER_LAUNCH`` levels.
    """
    n = 1 << levels
    table, corners = _icosphere_tables(wp.get_device(vertices.device).alias)
    matrix, transform_mode = _placement(transform)
    assert transform is None or levels <= _ICOSPHERE_LEVELS_PER_LAUNCH
    # One launch for the first ``_ICOSPHERE_LEVELS_PER_LAUNCH`` levels, then one per level; the
    # last one writes the faces too.
    coarse = n
    stride = n >> min(levels, _ICOSPHERE_LEVELS_PER_LAUNCH)
    while True:
        _launch.launch(
            kernel_creation.icosphere_mesh,
            dim=(20, n // stride + 1, n // stride + 1),
            inputs=[
                table,
                corners,
                n,
                coarse,
                stride,
                radius,
                matrix,
                transform_mode,
                vertices,
                faces,
            ],
            device=vertices.device,
        )
        if stride == 1:
            break
        coarse = stride
        stride = coarse >> 1


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
    transform: wp.mat44 | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a latitude/longitude sphere centered on the origin.

    Much cheaper than [`icosphere`][ordito.creation.icosphere], at the cost of triangles that
    shrink toward the poles.

    Parameters
    ----------
    radius
        Sphere radius.
    count
        ``(2,)`` number of latitude lines (profile points, poles included) and longitude sections.
        Defaults to ``(32, 64)``. Both entries are rounded up to even.
    transform
        Transform applied after construction, as a ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on ``device``.

    Raises
    ------
    ValueError
        If ``count`` is given and is not a ``(2,)`` integer pair.

    Notes
    -----
    ``count[1]`` is the longitude section count itself, whether given or defaulted.
    [`trimesh.creation.uv_sphere`][] doubles an explicit ``count[1]`` (and only an explicit one),
    so for an even ``b``, ``trimesh.creation.uv_sphere(count=[a, b])`` is
    ``uv_sphere(count=(a, 2 * b))`` here.

    The two pole points are snapped to exactly ``(0, -radius)`` and ``(0, radius)`` in the 2D
    profile. ``sin(pi)`` is ``1.2e-16`` rather than zero, and without the snap the pole vertices
    of adjacent slices differ by ``1.2e-16 * radius`` — enough that the weld and the
    degenerate-triangle filter in [`revolve`][ordito.creation.revolve] both start to fail at
    large radii. The correction is far below ``float32`` resolution.

    See Also
    --------
    [`icosphere`][ordito.creation.icosphere]
    [`capsule`][ordito.creation.capsule]
    [`revolve`][ordito.creation.revolve]
    [`trimesh.creation.uv_sphere`][]

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.uv_sphere(radius=2.0, count=(8, 16))
    print(vertices.shape, faces.size // 3)  # (98,) 192
    print(od.validation.is_watertight(vertices, faces))  # True
    print(f"{od.measures.volume(vertices, faces):.3f}")  # 31.039
    ```
    """
    counts = np.array([32, 64], dtype=np.int64) if count is None else np.asanyarray(count, np.int64)
    if counts.shape != (2,):
        raise ValueError(f"count must be (2,) int, got {counts.shape}")
    counts = counts + counts % 2
    latitude, longitude = int(counts[0]), int(counts[1])

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
    ``angle * r / n_rings``. Unlike a sliced [`icosphere`][ordito.creation.icosphere] the rim is a
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
        ``(1 + 3 * n * (n + 1),)`` and ``(18 * n ** 2,)`` arrays ``(vertices, faces)`` on
        ``device``, wound so the normals point away from the sphere center. Vertex 0 is the apex at
        ``(0, 0, radius)`` and the last ``6 * n`` vertices are the rim, in azimuthal order.

    Raises
    ------
    ValueError
        If ``angle`` is outside ``(0, pi)`` or ``subdivisions`` is negative.

    See Also
    --------
    [`icosphere`][ordito.creation.icosphere]
    [`uv_sphere`][ordito.creation.uv_sphere]

    Examples
    --------
    With ``subdivisions=1`` the cap has two rings, and its rim is the last 12 vertices:

    ```python
    import math

    import ordito as od

    vertices, faces = od.creation.sphere_cap(angle=math.pi / 4, subdivisions=1)
    print(vertices.shape, faces.size // 3)  # (19,) 24
    rim = sorted(od.boundary.boundary_vertex_indices(vertices, faces).numpy().tolist())
    print(len(rim), rim[0], rim[-1])  # 12 7 18
    ```
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
    # before it -- which is the case that justifies a device build for a template whose output
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
    transform: wp.mat44 | None = None,
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
        Transform applied after construction, as a ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on ``device``.

    Raises
    ------
    ValueError
        If ``count`` is given and is not a ``(2,)`` integer pair.

    Notes
    -----
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
    [`uv_sphere`][ordito.creation.uv_sphere]
    [`cylinder`][ordito.creation.cylinder]
    [`trimesh.creation.capsule`][]

    Examples
    --------
    The capsule spans ``height + 2 * radius`` along Z:

    ```python
    import ordito as od

    vertices, faces = od.creation.capsule(height=2.0, radius=0.5, count=(8, 16))
    z = vertices.numpy()[:, 2]
    print(f"{z.min():.3f} {z.max():.3f}")  # -1.500 1.500
    print(od.validation.is_watertight(vertices, faces))  # True
    ```
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
    transform: wp.mat44 | None = None,
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
        Transform applied after construction, as a ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on ``device``.

    Raises
    ------
    ValueError
        If neither ``height`` nor ``segment`` is given, or ``segment`` is not ``(2, 3)``.

    See Also
    --------
    [`annulus`][ordito.creation.annulus]
    [`cone`][ordito.creation.cone]
    [`capsule`][ordito.creation.capsule]
    [`trimesh.creation.cylinder`][]

    Examples
    --------
    Along Z by default, or between two points with ``segment``:

    ```python
    import ordito as od

    vertices, faces = od.creation.cylinder(radius=1.0, height=2.0, sections=8)
    print(vertices.shape, faces.size // 3)  # (18,) 32
    segment = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    vertices, faces = od.creation.cylinder(radius=0.5, segment=segment, sections=8)
    x = vertices.numpy()[:, 0]
    print(f"{x.min():.3f} {x.max():.3f}")  # 0.000 3.000
    ```
    """
    transform, half = _resolve_cylinder_axis(height, segment, transform)
    n_sections = _resolve_sections(sections)
    template = _cylinder_template(float(radius), half, n_sections)
    vertices = _launch.empty(2 * n_sections + 2, dtype=wp.vec3, device=device)
    faces = _launch.empty(3 * _cylinder_face_count(template), dtype=wp.int32, device=device)
    _cylinder_into(template, transform, 0, vertices, faces)
    return vertices, faces


def _cylinder_template(
    radius: float, half: float, n_sections: int
) -> tuple[float, float, int, int, int]:
    """``(radius, half_height, n_sections, keep_caps, keep_sides)`` of a capped cylinder."""
    profile = np.array(
        [[0.0, -half], [radius, -half], [radius, half], [0.0, half]], dtype=np.float64
    )
    keep_caps, keep_sides = _closed_form_template(profile, n_sections)
    return radius, half, n_sections, keep_caps, keep_sides


def _cylinder_face_count(template: tuple[float, float, int, int, int]) -> int:
    """Face count of the cylinder ``template`` describes."""
    _, _, n_sections, keep_caps, keep_sides = template
    return (2 * keep_caps + 2 * keep_sides) * n_sections


def _cylinder_into(
    template: tuple[float, float, int, int, int],
    transform: wp.mat44 | None,
    vertex_base: int,
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
) -> None:
    """Write a cylinder into ``vertices`` and ``faces``, face indices offset by ``vertex_base``."""
    radius, half, n_sections, keep_caps, keep_sides = template
    _launch.launch(
        kernel_creation.cylinder_mesh,
        dim=n_sections,
        inputs=[
            wp.float32(radius),
            wp.float32(half),
            wp.int32(n_sections),
            _FULL_TURN,
            wp.int32(keep_caps),
            wp.int32(keep_sides),
            *_placement(transform),
            wp.int32(vertex_base),
        ],
        outputs=[vertices, faces],
        device=vertices.device,
    )


def cone(
    radius: float,
    height: float,
    sections: int | None = None,
    transform: wp.mat44 | None = None,
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
        Transform applied after construction, as a ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on ``device``.

    See Also
    --------
    [`cylinder`][ordito.creation.cylinder]
    [`trimesh.creation.cone`][]

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.cone(radius=1.0, height=2.0, sections=16)
    print(vertices.shape, faces.size // 3)  # (18,) 32
    z = vertices.numpy()[:, 2]
    print(f"{z.min():.3f} {z.max():.3f}")  # 0.000 2.000
    ```
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
            *_placement(transform),
        ],
        outputs=[vertices, faces],
        device=device,
    )
    return vertices, faces


def annulus(
    r_min: float,
    r_max: float,
    height: float | None = None,
    sections: int | None = None,
    transform: wp.mat44 | None = None,
    segment: Sequence[Sequence[float]] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create an annular cylinder (a tube with flat ends) along Z, centered on the origin.

    Parameters
    ----------
    r_min
        Inner radius. When smaller than ``1e-8`` this delegates to
        [`cylinder`][ordito.creation.cylinder], which has a different topology (no inner wall).
    r_max
        Outer radius.
    height
        Height of the annular cylinder. Required unless ``segment`` is passed.
    sections
        Number of pie wedges around the revolution. Defaults to 32.
    transform
        Transform applied after construction, as a ``wp.mat44``.
    segment
        ``(2, 3)`` axis endpoints. Overrides both ``height`` and ``transform``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on ``device``.

    Raises
    ------
    ValueError
        If neither ``height`` nor ``segment`` is given, or ``segment`` is not ``(2, 3)``.

    See Also
    --------
    [`cylinder`][ordito.creation.cylinder]
    [`torus`][ordito.creation.torus]
    [`trimesh.creation.annulus`][]

    Examples
    --------
    A tube is closed but has genus 1, so its Euler characteristic is 0:

    ```python
    import ordito as od

    vertices, faces = od.creation.annulus(r_min=0.5, r_max=1.0, height=1.0, sections=16)
    print(vertices.shape, faces.size // 3)  # (64,) 128
    print(od.measures.euler_characteristic(faces))  # 0
    print(od.validation.is_watertight(vertices, faces))  # True
    ```
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
    transform: wp.mat44 | None = None,
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
        Transform applied after construction, as a ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(6 * major_sections * minor_sections,)`` arrays
        ``(vertices, faces)`` with ``2 * major_sections * minor_sections`` triangles.

    Notes
    -----
    The closing point of the tube profile is snapped to equal its first point exactly. ``sin(2 *
    pi)`` evaluates to ``-2.4e-16``, so without the snap the minor seam would only close through
    [`revolve`][ordito.creation.revolve]'s absolute weld tolerance and would come apart at large
    radii.

    See Also
    --------
    [`annulus`][ordito.creation.annulus]
    [`revolve`][ordito.creation.revolve]
    [`trimesh.creation.torus`][]

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.torus(2.0, 0.5, major_sections=16, minor_sections=8)
    print(vertices.shape, faces.size // 3)  # (128,) 256
    print(od.measures.euler_characteristic(faces))  # 0
    ```
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
    transform: wp.mat44 | None = None,
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
        The profile is then the boundary of the cap polygon, so it is closed whether or not its
        last point repeats its first: the segment from its last point back to its first is
        revolved into a wall like every other (a closing segment on the axis sweeps no area and
        adds no wall). Ignored for a full revolution, which needs no caps. Requires a simple (non
        self-intersecting) profile — see
        [`triangulate_polygon`][ordito.polyline.triangulate_polygon].
    sections
        Number of pie wedges around the revolution. Defaults to 32 per full revolution, scaled by
        ``angle``.
    transform
        Transform applied after construction, as a ``wp.mat44``. Face winding is reversed when its
        determinant is negative.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on
        ``linestring.device``.

    Raises
    ------
    ValueError
        If ``linestring`` is not a rank-1 ``wp.vec2`` array with at least 2 points, or
        ``sections`` resolves to less than 1.

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
    [`uv_sphere`][ordito.creation.uv_sphere]
    [`cylinder`][ordito.creation.cylinder]
    [`torus`][ordito.creation.torus]
    [`extrude_triangulation`][ordito.creation.extrude_triangulation]
    [`trimesh.creation.revolve`][]

    Examples
    --------
    Revolving a segment parallel to the axis gives an open tube; a closed square profile gives a
    solid ring:

    ```python
    import warp as wp

    import ordito as od

    segment = wp.array([[1.0, 0.0], [1.0, 1.0]], dtype=wp.vec2)
    vertices, faces = od.creation.revolve(segment, sections=8)
    print(vertices.shape, faces.size // 3)  # (16,) 16
    square = wp.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=wp.vec2)
    vertices, faces = od.creation.revolve(square, sections=16)
    print(od.validation.is_watertight(vertices, faces))  # True
    ```
    """
    odt.ensure_ndim(linestring, 1, dtype=wp.vec2)
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
    capped = cap and not closed
    if capped and per > 2 and not _repeats_first_point(profile_np):
        # A capped profile is the boundary of its cap polygon, so its closing segment is a wall
        # too. Repeating the first point lets the column fold below weld that wall to the first
        # column, exactly as for a profile that closes itself.
        profile_np = np.vstack((profile_np, profile_np[:1]))
        linestring = _upload_points(profile_np, wp.vec2, device)
        per += 1
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
    if capped:
        profile_3d = _launch.empty(per, dtype=wp.vec3, device=device)
        _launch.map(kernel_array.lift_vec2, linestring, wp.float32(0.0), out=profile_3d)
        # Ear clipping introduces no new vertices, so its indices address profile points directly --
        # the guarantee trimesh gets from ``triangulate_polygon(force_vertices=True)``.
        cap_faces = od.polyline.polyline_triangulate(profile_3d).reshape((-1,))
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
            *_placement(transform),
            vertices,
            faces,
        ],
        device=device,
    )
    return vertices, faces


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
    [`_revolve_kept_template`][ordito.creation._revolve_kept_template] in Python floats.

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
    if per > 2 and _repeats_first_point(profile_np):
        column_np[per - 1] = 0
    owns_np = column_np == np.arange(per)
    counts_np = np.where(on_axis_np, 1, n_kept_slices) * owns_np
    offsets_np = np.concatenate(([0], np.cumsum(counts_np)[:-1])).astype(np.int32)
    return column_np, offsets_np, on_axis_np, int(counts_np.sum())


def _repeats_first_point(profile_np: np.ndarray) -> bool:
    """Whether a profile's last point repeats its first, within ``1e-8`` on each component."""
    return bool(np.all(np.abs(profile_np[0] - profile_np[-1]) <= TOLERANCE_MERGE))


def extrude_triangulation(
    vertices: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    height: float,
    transform: wp.mat44 | None = None,
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
        ``(3 * n_faces,)`` flat triangle index buffer into ``vertices``.
    height
        Distance to extrude along Z. May be negative; the triangulation is re-wound to agree with
        its sign so the result always has positive volume.
    transform
        Transform applied after construction, as a ``wp.mat44``. Face winding is reversed when its
        determinant is negative.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(2 * n,)`` and ``(3 * n_out_faces,)`` arrays ``(vertices, faces)``, ``n_out_faces`` the
        output triangle count: ``2 * n`` vertices, on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``vertices`` is not a rank-1 ``wp.vec2`` array, ``faces`` is not a flat multiple of 3,
        or ``abs(height)`` is at most ``1e-8``.
    RuntimeError
        If ``vertices`` and ``faces`` are not on one device.

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
    [`extrude_polygon`][ordito.creation.extrude_polygon]
    [`sweep_polygon`][ordito.creation.sweep_polygon]
    [`revolve`][ordito.creation.revolve]
    [`trimesh.creation.extrude_triangulation`][]

    Examples
    --------
    A unit square in two triangles becomes a closed unit cube:

    ```python
    import warp as wp

    import ordito as od

    square = wp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=wp.vec2)
    triangles = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32)
    vertices, faces = od.creation.extrude_triangulation(square, triangles, height=1.0)
    print(vertices.shape, faces.size // 3)  # (8,) 12
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 1.0000
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    return _extrude(vertices, faces, height, transform, ring_walls=False)


def extrude_polygon(
    polygon: wp.array[wp.vec2],
    height: float,
    transform: wp.mat44 | None = None,
    mid_plane: bool = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extrude a simple 2D polygon along Z into a watertight prism.

    Triangulates the ring with [`triangulate_polygon`][ordito.polyline.triangulate_polygon] and
    hands the result to [`extrude_triangulation`][ordito.creation.extrude_triangulation].

    Parameters
    ----------
    polygon
        ``(n,)`` 2D ring vertices, in order. A repeated closing point is dropped. Interior rings
        (holes) are not supported.
    height
        Distance to extrude along Z.
    transform
        Transform applied after construction, as a ``wp.mat44``.
    mid_plane
        Center the extrusion on ``z = 0`` instead of starting it there.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on
        ``polygon.device``.

    See Also
    --------
    [`triangulate_polygon`][ordito.polyline.triangulate_polygon]
    [`extrude_triangulation`][ordito.creation.extrude_triangulation]
    [`sweep_polygon`][ordito.creation.sweep_polygon]
    [`trimesh.creation.extrude_polygon`][]

    Examples
    --------
    An L-shaped outline of area 3 extruded by 2:

    ```python
    import warp as wp

    import ordito as od

    outline = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]
    polygon = wp.array(outline, dtype=wp.vec2)
    vertices, faces = od.creation.extrude_polygon(polygon, height=2.0)
    print(vertices.shape, faces.size // 3)  # (12,) 20
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 6.0000
    ```
    """
    ring, faces = od.polyline.triangulate_polygon(polygon)
    if mid_plane:
        translation = np.eye(4)
        translation[2, 3] = abs(float(height)) / -2.0
        if transform is None:
            transform = wp.mat44(*translation.flatten())
        else:
            composed = od.transform.matrix_to_numpy(transform).dot(translation)
            transform = wp.mat44(*composed.flatten())
    # A full triangulation of the ring is bounded by the ring edges themselves; a partial one (a
    # degenerate ring) has its boundary derived, as ``extrude_triangulation`` does.
    full = faces.size // 3 == ring.size - 2
    return _extrude(ring, faces, height, transform, ring_walls=full)


def _extrude(
    vertices: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    height: float,
    transform: wp.mat44 | None,
    *,
    ring_walls: bool,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Shared body of ``extrude_triangulation`` and ``extrude_polygon``.

    ``ring_walls`` says ``faces`` is a full ``n - 2`` triangulation of the ring ``vertices``, whose
    boundary is then the ring edges and is not derived. Whether the triangulation is re-wound to
    agree with the sign of ``height`` is decided on the device, from its total signed area.
    """
    odt.ensure_ndim(vertices, 1, dtype=wp.vec2)
    odt.ensure_ndim(faces, 1, dtype=wp.int32)
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
    matrix, transform_mode = _placement(transform)
    layers = _launch.empty(2 * n, dtype=wp.vec3, device=device)
    area = _launch.zeros(1, dtype=wp.float32, device=device)
    _launch.launch_tiled(
        kernel_creation.lift_layers_and_signed_area,
        dim=kernel_reduce.chunks_1d(max(n, n_faces)),
        inputs=[vertices, faces, wp.float32(height_f), matrix, transform_mode, layers, area],
        block_dim=TILE_1D,
        device=device,
    )

    boundary = None
    n_boundary = n
    if not ring_walls:
        # The lower layer is a contiguous prefix, so the boundary walk reads it as a view. The
        # boundary of the triangulation as given; a re-wound one reverses each edge on the device.
        boundary = od.boundary.oriented_boundary_edges(odt.as_dense(layers[:n]), faces)
        n_boundary = int(boundary.shape[0])

    # Bottom cap winding is reversed; the top cap keeps it and is offset by one vertex block.
    out_faces = _launch.empty((2 * n_faces + 2 * n_boundary) * 3, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_creation.extrude_faces,
        dim=2 * n_faces + n_boundary,
        inputs=[faces, boundary, wp.int32(n), height_f < 0.0, area, transform_mode, out_faces],
        device=device,
    )
    return layers, out_faces


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
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on
        ``polygon.device``.

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
    [`extrude_polygon`][ordito.creation.extrude_polygon]
    [`revolve`][ordito.creation.revolve]
    [`trimesh.creation.sweep_polygon`][]

    Examples
    --------
    A unit square swept along a straight path of length 3 is a capped prism:

    ```python
    import warp as wp

    import ordito as od

    square = wp.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=wp.vec2)
    path = wp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=wp.vec3)
    vertices, faces = od.creation.sweep_polygon(square, path)
    print(vertices.shape, faces.size // 3)  # (12,) 20
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 3.0000
    ```
    """
    require_same_device(polygon=polygon, path=path, angles=angles)
    odt.ensure_ndim(path, 1, dtype=wp.vec3)
    device = polygon.device
    n_path = path.size
    if n_path < 2:
        raise ValueError(f"path must have at least 2 points, got {n_path}")
    if angles is not None and angles.size != n_path:
        raise ValueError(
            f"angles must have one entry per path point ({n_path}), got {angles.shape}"
        )

    ring, cap_faces = od.polyline.triangulate_polygon(polygon)
    stride = ring.size
    boundary = None
    n_boundary = stride
    if cap_faces.size // 3 != stride - 2:
        # Only a partial triangulation (a ring that is not simple) can be bounded by anything but
        # the ring edges. ``oriented_boundary_edges`` only uses the vertex count, as its row-hash
        # base, so a zero buffer of the right length is enough.
        boundary = od.boundary.oriented_boundary_edges(
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
        ``(3 * n_faces,)`` flat triangle index buffer.
    origin
        Point on the truncation plane. ``None`` truncates against the ``z = 0`` plane.
    normal
        Unit normal of the truncation plane. Required when ``origin`` is given.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(6 * n_faces,)`` and ``(24 * n_faces,)`` arrays ``(vertices, faces)``: ``6 * n_faces``
        vertices and ``8 * n_faces`` triangles, on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``faces`` is not a flat multiple of 3, or ``origin`` is given without ``normal``.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    Takes an indexed mesh rather than [`trimesh.creation.truncated_prisms`][]'s ``(n, 3, 3)``
    triangle soup, matching the ``(vertices, faces)`` convention used throughout ordito. Vertices
    are *not* merged, so each prism stays a separate body — the same choice trimesh makes with
    ``process=False``.

    See Also
    --------
    [`extrude_triangulation`][ordito.creation.extrude_triangulation]
    [`trimesh.creation.truncated_prisms`][]

    Examples
    --------
    A unit right triangle at height 2 becomes one prism of volume ``0.5 * 2``:

    ```python
    import warp as wp

    import ordito as od

    vertices = wp.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]], dtype=wp.vec3)
    faces = wp.array([0, 1, 2], dtype=wp.int32)
    prism_vertices, prism_faces = od.creation.truncated_prisms(vertices, faces)
    print(prism_vertices.shape, prism_faces.size // 3)  # (6,) 8
    print(f"{od.measures.volume(prism_vertices, prism_faces):.4f}")  # 1.0000
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    odt.ensure_ndim(faces, 1, dtype=wp.int32)
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
    # and the two agree to float32 rounding (``wp.inverse`` works in float32).
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
    transform: wp.mat44 | None = None,
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
        Transform applied to the whole marker, as a ``wp.mat44``.
    axis_radius
        Radius of the three axis cylinders. Defaults to ``origin_size / 5``.
    axis_length
        Length of the three axis cylinders. Defaults to ``origin_size * 10``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on ``device``, with
        the ball and the three cylinders concatenated in X, Y, Z order after the ball.

    Notes
    -----
    [`trimesh.creation.axis`][] colors the ball white and the cylinders red/green/blue for X/Y/Z.
    ordito carries no visual attributes, so this returns geometry only and takes no
    ``origin_color``; the pieces are concatenated in a documented order so a caller can assign
    per-face colors itself.

    See Also
    --------
    [`icosphere`][ordito.creation.icosphere]
    [`cylinder`][ordito.creation.cylinder]
    [`concatenate`][ordito.combine.concatenate]
    [`trimesh.creation.axis`][]

    Examples
    --------
    A ball and three cylinders: four connected components:

    ```python
    import ordito as od

    vertices, faces = od.creation.axis(origin_size=0.05)
    print(len(od.combine.split(vertices, faces)))  # 4
    ```
    """
    origin_size_f = float(origin_size)
    if axis_radius is None:
        axis_radius = origin_size_f / 5.0
    if axis_length is None:
        axis_length = origin_size_f * 10.0

    # Each cylinder is built centered on the origin, so it has to be pushed out along its own axis
    # by half its length, then carried by the marker transform: the two compose on the host into
    # the one matrix its kernel places it with. The four parts are written straight into views of
    # the final buffers, the ball first, so there is no concatenation pass.
    shift = np.eye(4)
    shift[2, 3] = float(axis_length) / 2.0
    marker = np.eye(4) if transform is None else od.transform.matrix_to_numpy(transform)

    template = _cylinder_template(float(axis_radius), float(axis_length) / 2.0, DEFAULT_SECTIONS)
    ball_vertices, ball_faces = _icosphere_sizes(_AXIS_BALL_LEVELS)
    cylinder_vertices = 2 * DEFAULT_SECTIONS + 2
    cylinder_faces = _cylinder_face_count(template)
    device = wp.get_device(device)
    vertices = _launch.empty(ball_vertices + 3 * cylinder_vertices, dtype=wp.vec3, device=device)
    faces = _launch.empty(3 * (ball_faces + 3 * cylinder_faces), dtype=wp.int32, device=device)
    _icosphere_into(
        _AXIS_BALL_LEVELS,
        origin_size_f,
        transform,
        odt.as_dense(vertices[:ball_vertices]),
        odt.as_dense(faces[: 3 * ball_faces]),
    )
    for k, direction in enumerate((_UNIT_X, _UNIT_Y, _UNIT_Z)):
        placement = marker @ _align_vectors(_UNIT_Z, direction).dot(shift)
        vertex_start = ball_vertices + k * cylinder_vertices
        face_start = 3 * (ball_faces + k * cylinder_faces)
        _cylinder_into(
            template,
            wp.mat44(*placement.flatten()),
            vertex_start,
            odt.as_dense(vertices[vertex_start : vertex_start + cylinder_vertices]),
            odt.as_dense(faces[face_start : face_start + 3 * cylinder_faces]),
        )
    return vertices, faces


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
"""Analytic surface selected by [`parametric_surface`][ordito.creation.parametric_surface]."""


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
    [`is_orientable`][ordito.validation.is_orientable],
    [`homology_generators`][ordito.homology.homology_generators] and anything whose behaviour
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
          [`super_toroid`][ordito.creation.super_toroid].
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
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on ``device``. The
        vertex count is below
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
    [`remove_duplicated_vertices`][ordito.repair.remove_duplicated_vertices], which several of
    these surfaces exist to test. And where an immersed surface merely *crosses* itself, the sheets
    stay separate: on Catalan's minimal surface VTK merges 40 lattice points that the
    parameterization does not identify, and drops the 2 triangles that thereby became degenerate.

    See Also
    --------
    [`super_ellipsoid`][ordito.creation.super_ellipsoid]
    [`super_toroid`][ordito.creation.super_toroid]
    [`random_hills`][ordito.creation.random_hills]
    [`grid`][ordito.creation.grid]

    Examples
    --------
    Boy's surface is closed and non-orientable, with Euler characteristic 1:

    ```python
    import ordito as od

    vertices, faces = od.creation.parametric_surface("boy", u_resolution=20, v_resolution=20)
    print(od.measures.euler_characteristic(faces), od.validation.is_orientable(faces))  # 1 False
    ```
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
    [`parametric_surface`][ordito.creation.parametric_surface]'s enum.

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
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on ``device``, closed
        and watertight: the u direction wraps and both v extremes are poles.

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
    [`uv_sphere`][ordito.creation.uv_sphere]
    [`super_toroid`][ordito.creation.super_toroid]
    [`parametric_surface`][ordito.creation.parametric_surface]

    Examples
    --------
    Exponents near 0 square the sphere off towards the cube ``[-1, 1] ** 3``:

    ```python
    import ordito as od

    vertices, faces = od.creation.super_ellipsoid(n1=0.1, n2=0.1, u_resolution=20, v_resolution=20)
    print(od.measures.euler_characteristic(faces))  # 2
    print(f"{od.measures.volume(vertices, faces):.2f}")  # 7.87
    ```
    """
    radii_np = np.asanyarray(radii, dtype=np.float64)
    if radii_np.shape != (3,):
        raise ValueError(f"radii must be (3,) float, got {radii_np.shape}")
    # Radii other than one scale the unit superquadric as it is written.
    scale = None
    if not np.array_equal(radii_np, np.ones(3)):
        scale = wp.mat44(*np.diag([*radii_np, 1.0]).ravel())
    return _parametric_surface(
        _SUPER_ELLIPSOID_SPEC, u_resolution, v_resolution, device, n1=n1, n2=n2, transform=scale
    )


def super_toroid(
    n1: float = 1.0,
    n2: float = 1.0,
    u_resolution: int = 40,
    v_resolution: int = 40,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a superquadric torus: a closed genus-1 surface with a squareness axis.

    The genus-1 counterpart of [`super_ellipsoid`][ordito.creation.super_ellipsoid]. ``n1 = n2 =
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
        ``(n_vertices,)`` and ``(3 * n_faces,)`` arrays ``(vertices, faces)`` on ``device``, closed
        and watertight with Euler characteristic 0: both parameter directions wrap.

    Raises
    ------
    ValueError
        If either resolution is below 3. Both parameter directions wrap, and a resolution of 2
        would identify every cell's two rows and leave no faces at all.

    Notes
    -----
    VTK's ``vtkParametricSuperToroid``, which pyvista exposes as ``pv.ParametricSuperToroid``. The
    radii are VTK's and are not exposed: [`torus`][ordito.creation.torus] is the builder for that
    axis, and this one exists for the exponents.

    See Also
    --------
    [`torus`][ordito.creation.torus]
    [`super_ellipsoid`][ordito.creation.super_ellipsoid]
    [`parametric_surface`][ordito.creation.parametric_surface]

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.super_toroid(n1=0.5, n2=1.0, u_resolution=24, v_resolution=12)
    print(vertices.shape, faces.size // 3)  # (253,) 506
    print(od.measures.euler_characteristic(faces))  # 0
    ```
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
        ``(u_resolution * v_resolution,)`` and ``(6 * (u_resolution - 1) * (v_resolution - 1),)``
        arrays ``(vertices, faces)`` on ``device`` with ``u_resolution * v_resolution`` vertices —
        nothing is identified — and ``2 * (u_resolution - 1) * (v_resolution - 1)`` triangles.

    Raises
    ------
    ValueError
        If either resolution is below 2, or either variance is not positive.

    Notes
    -----
    Named for VTK's ``vtkParametricRandomHills``, and it is the same idea over the same domain, but
    the heights are not comparable: VTK draws each hill's amplitude and variance from its own
    generator, where this takes them as parameters and draws only the centres. The seed goes
    through [`resolve_seed`][ordito.sample.resolve_seed], so a given seed reproduces a given mesh.

    See Also
    --------
    [`grid`][ordito.creation.grid]
    [`random_soup`][ordito.creation.random_soup]
    [`parametric_surface`][ordito.creation.parametric_surface]

    Examples
    --------
    The same seed rebuilds the same height field:

    ```python
    import numpy as np

    import ordito as od

    vertices, faces = od.creation.random_hills(seed=7, u_resolution=16, v_resolution=16)
    print(vertices.shape, faces.size // 3)  # (256,) 450
    again, _ = od.creation.random_hills(seed=7, u_resolution=16, v_resolution=16)
    print(np.array_equal(vertices.numpy(), again.numpy()))  # True
    ```
    """
    if float(x_variance) <= 0.0 or float(y_variance) <= 0.0:
        raise ValueError(f"variances must be positive, got {(x_variance, y_variance)}")
    generator = np.random.default_rng(od.sample.resolve_seed(seed))
    centers = generator.uniform(
        low=(_RANDOM_HILLS_SPEC.u_range[0], _RANDOM_HILLS_SPEC.v_range[0]),
        high=(_RANDOM_HILLS_SPEC.u_range[1], _RANDOM_HILLS_SPEC.v_range[1]),
        size=(max(int(n_hills), 0), 2),
    )
    return _parametric_surface(
        _RANDOM_HILLS_SPEC,
        u_resolution,
        v_resolution,
        device,
        hills=(float(amplitude), float(x_variance), float(y_variance), centers),
    )


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
        ``(3 * face_count,)`` and ``(3 * face_count,)`` arrays ``(vertices, faces)`` with
        ``3 * face_count`` vertices in ``[-0.5, 0.5] ** 3`` and
        ``faces == arange(3 * face_count)``.

    Notes
    -----
    Takes an explicit ``seed`` where [`trimesh.creation.random_soup`][] draws from NumPy's global
    RNG, so results are reproducible. The point values come from Warp's per-thread generator and do
    not match NumPy's stream.

    See Also
    --------
    [`trimesh.creation.random_soup`][]

    Examples
    --------
    Every triangle owns its three vertices, so a soup of 50 faces has 50 components:

    ```python
    import ordito as od

    vertices, faces = od.creation.random_soup(face_count=50, seed=0)
    print(vertices.shape, faces.shape)  # (150,) (150,)
    print(len(od.combine.split(vertices, faces)))  # 50
    ```
    """
    n = 3 * max(int(face_count), 0)
    vertices = _launch.empty(n, dtype=wp.vec3, device=device)
    if n > 0:
        resolved_seed = od.sample.resolve_seed(seed)
        _launch.launch(
            kernel_creation.random_soup_vertices,
            dim=n,
            inputs=[wp.int32(resolved_seed), vertices],
            device=device,
        )
    return vertices, od.array.arange(n, device=vertices.device)


# --- private helpers ---------------------------------------------------------------------


class _ParametricSpec(NamedTuple):
    """
    One analytic surface: its kernel id, its parameter rectangle and how the rectangle glues.

    The gluing is what makes these surfaces interesting and it is *combinatorial* -- a fact about
    the map, not about how close two evaluated points happen to land. Every flag was derived from
    the map itself and checked against VTK's welded output; see
    [`parametric_surface`][ordito.creation.parametric_surface].
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

# The gluing flags as plain ints, for the host's lattice arithmetic.
_GLUE_U_WRAP = int(kernel_creation.GLUE_U_WRAP)
_GLUE_U_TWIST = int(kernel_creation.GLUE_U_TWIST)
_GLUE_V_WRAP = int(kernel_creation.GLUE_V_WRAP)
_GLUE_V_TWIST = int(kernel_creation.GLUE_V_TWIST)
_GLUE_POLE_J_LO = int(kernel_creation.GLUE_POLE_J_LO)
_GLUE_POLE_J_HI = int(kernel_creation.GLUE_POLE_J_HI)
_GLUE_POLE_I_LO = int(kernel_creation.GLUE_POLE_I_LO)
_GLUE_POLE_I_HI = int(kernel_creation.GLUE_POLE_I_HI)

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
# The height field sums a table of bumps rather than evaluating a closed form, and its lattice is a
# plain grid.
_RANDOM_HILLS_SPEC = _ParametricSpec(
    kernel_creation.SURFACE_RANDOM_HILLS, (-10.0, 10.0), (-10.0, 10.0)
)

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
    *,
    n1: float = 1.0,
    n2: float = 1.0,
    transform: wp.mat44 | None = None,
    hills: tuple[float, float, float, np.ndarray] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Sample one analytic surface on its identified lattice, for the four public builders.

    One output per *vertex*, carrying the one lattice sample chosen to represent it, never one per
    lattice sample: where the surface glues, several lattice samples map to the same vertex, and
    evaluating only the canonical one keeps the result bit-exact run to run -- mapping over the
    lattice and letting the identified samples race for the slot would not (a twisted seam and a
    collapsed pole row reach the same point through different expressions, so they agree only to
    rounding).

    The topology is pure index arithmetic, in the same spirit as [`grid`][ordito.creation.grid]: no
    position is consulted and no tolerance appears anywhere, so it is exact and independent of
    resolution, dtype and device. Welding by *distance* instead would (a) make every one of these
    meshes depend on [`remove_duplicated_vertices`][ordito.repair.remove_duplicated_vertices],
    which several of them exist to test, and (b) glue the accidental self-intersections of an
    immersed surface -- which is what VTK does, merging 40 lattice points of Catalan's minimal
    surface that the map does not identify, because two sheets happen to cross there.

    All the gluing happens on the lattice's four edges and acts uniformly along each side, so the
    host resolves a fixed handful of samples (``_parametric_layout``), sizes both outputs in closed
    form, and one launch writes the whole mesh (``kernels/creation.parametric_mesh``). ``hills`` is
    ``random_hills``'s ``(amplitude, x_variance, y_variance, centers)``.
    """
    if spec.u_twist and spec.v_twist:
        raise ValueError("a surface twisted in both directions is not supported")
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

    gluing = _parametric_gluing(spec)
    meta, dead_np, n_vertices, n_dead = _parametric_layout(gluing, n_u, n_v)
    amplitude, x_variance, y_variance, centers_np = (
        (0.0, 1.0, 1.0, np.empty((0, 2))) if hills is None else hills
    )
    # The parameter tables are built here rather than in the kernel so that both ends of the domain
    # are hit exactly: several of these maps are singular one ulp outside their rectangle.
    floats_np = np.concatenate(
        (
            np.linspace(*spec.u_range, n_u),
            np.linspace(*spec.v_range, n_v),
            np.asarray(centers_np, dtype=np.float64).reshape(-1),
        )
    ).astype(np.float32)
    layout = _launch.array(
        np.concatenate((np.array(meta, dtype=np.int32), floats_np.view(np.int32), dead_np)),
        dtype=wp.int32,
        device=device,
    )
    n_triangles = 2 * (n_u - 1) * (n_v - 1)
    vertices = _launch.empty(n_vertices, dtype=wp.vec3, device=layout.device)
    faces = _launch.empty(3 * (n_triangles - n_dead), dtype=wp.int32, device=layout.device)
    _launch.launch(
        kernel_creation.parametric_mesh,
        dim=n_u * n_v + n_triangles,
        inputs=[
            spec.kind,
            wp.int32(n_u),
            wp.int32(n_v),
            wp.int32(gluing),
            wp.int32(centers_np.shape[0]),
            wp.float32(n1),
            wp.float32(n2),
            wp.float32(amplitude),
            wp.float32(x_variance),
            wp.float32(y_variance),
            layout,
            *_placement(transform),
        ],
        outputs=[vertices, faces],
        device=layout.device,
    )
    return vertices, faces


def _parametric_gluing(spec: _ParametricSpec) -> int:
    """
    Return the ``kernels/creation.GLUE_*`` bitmask of one surface's gluing rule.

    A pole is a boundary row the map collapses to a single point. A *wrapped* boundary has already
    identified its two extreme rows, so a pole on either of them is one pole on the surviving row
    -- which is why Boy and the cross-cap keep 1 483 vertices rather than losing a second row's
    worth. The mask carries the four resolved pole rows, not the four spec flags.
    """
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
    return gluing


def _parametric_layout(gluing: int, n_u: int, n_v: int) -> tuple[list[int], np.ndarray, int, int]:
    """
    ``(meta, dead, n_vertices, n_dead)``: the host half of one surface's single launch.

    The canonical keys are the reduced rectangle ``[0, extent_i) x [0, extent_j)`` (a wrapped axis
    loses its last row) less every point on an active pole line, plus the anchors a pole collapse
    reaches, so ``n_vertices`` is closed form. An interior lattice sample is never glued, and along
    each side of the lattice every inner sample goes through the same branch of the gluing rule,
    so a few samples decide everything else: which anchors are reached and the lowest lattice index
    reaching each (the sample its vertex is evaluated from -- a lattice corner, or the first inner
    sample of a side that collapses onto it), and which sides lose a triangle in every non-corner
    cell. ``dead`` lists the remaining dead triangles explicitly: those of the four corner cells, or
    of every cell when the lattice is two samples wide and a cell touches both long sides.

    ``meta`` is the head of ``kernels/creation.parametric_mesh``'s layout (its
    ``PARAMETRIC_META_*`` slots) and ``n_dead`` counts every dead triangle, listed or not.
    """
    u_wrap = gluing & _GLUE_U_WRAP != 0
    v_wrap = gluing & _GLUE_V_WRAP != 0
    j_lo, j_hi = gluing & _GLUE_POLE_J_LO != 0, gluing & _GLUE_POLE_J_HI != 0
    i_lo, i_hi = gluing & _GLUE_POLE_I_LO != 0, gluing & _GLUE_POLE_I_HI != 0
    extent_i, extent_j = n_u - int(u_wrap), n_v - int(v_wrap)
    pole_rows, pole_columns = int(j_lo) + int(j_hi), int(i_lo) + int(i_hi)
    n_pole = pole_rows * extent_i + pole_columns * extent_j - pole_rows * pole_columns
    # The anchors: ``(0, 0)``, ``(0, n_v - 1)``, ``(n_u - 1, 0)``. The second takes the points of
    # its row that no column pole re-anchors afterwards.
    anchor_keys = (0, n_v - 1, (n_u - 1) * n_v)
    hit = (j_lo or i_lo, j_hi and extent_i - int(i_lo) - int(i_hi) > 0, i_hi)
    n_vertices = extent_i * extent_j - n_pole + sum(hit)

    # One key pass over every sample consulted: the four lattice corners, the first two inner
    # samples of each side (top, bottom, left, right; clamped where a side has fewer), and the
    # corners of the four corner cells.
    last_i, last_j = n_u - 1, n_v - 1
    one_i, two_i, one_j, two_j = min(1, last_i), min(2, last_i), min(1, last_j), min(2, last_j)
    sample_i = [0, 0, last_i, last_i, 0, last_i, one_i, one_i, 0, last_i, two_i, two_i]
    sample_j = [0, last_j, 0, last_j, one_j, one_j, 0, last_j, two_j, two_j, 0, last_j]
    n_cells_u, n_cells_v = n_u - 1, n_v - 1
    n_cells = n_cells_u * n_cells_v
    if min(n_u, n_v) >= 3:
        cells = np.array([0, n_cells_v - 1, (n_cells_u - 1) * n_cells_v, n_cells - 1])
    else:
        cells = np.arange(n_cells)
    cell_i, cell_j = cells // n_cells_v, cells % n_cells_v
    keys = _parametric_keys(
        gluing,
        np.concatenate((sample_i, cell_i, cell_i + 1, cell_i + 1, cell_i)),
        np.concatenate((sample_j, cell_j, cell_j, cell_j + 1, cell_j + 1)),
        n_u,
        n_v,
    ).tolist()
    first_dead, second_dead = _corner_dead(*np.split(np.asarray(keys[12:]), 4))
    dead = np.concatenate((np.sort(cells[first_dead]), n_cells + np.sort(cells[second_dead])))

    # Lowest lattice index reaching each anchor: a corner, or a side's first inner sample.
    candidates = [(sample_i[k] * n_v + sample_j[k], keys[k]) for k in range(4)]
    if n_v >= 3:
        candidates += [(one_j, keys[4]), (last_i * n_v + one_j, keys[5])]
    if n_u >= 3:
        candidates += [(n_v, keys[6]), (n_v + last_j, keys[7])]
    first = [
        min((flat for flat, key in candidates if key == anchor), default=0) if reached else 0
        for anchor, reached in zip(anchor_keys, hit, strict=True)
    ]

    # A side whose inner samples collapse loses one triangle in each of its non-corner cells.
    sides = 0
    n_side_dead = 0
    if min(n_u, n_v) >= 3:
        for flag, one, two, count in (
            (kernel_creation.SIDE_TOP, keys[4], keys[8], n_cells_v - 2),
            (kernel_creation.SIDE_BOTTOM, keys[5], keys[9], n_cells_v - 2),
            (kernel_creation.SIDE_LEFT, keys[6], keys[10], n_cells_u - 2),
            (kernel_creation.SIDE_RIGHT, keys[7], keys[11], n_cells_u - 2),
        ):
            if count > 0 and one == two:
                sides |= int(flag)
                n_side_dead += count
    meta = [int(sum(1 << k for k, reached in enumerate(hit) if reached)), *first, sides, dead.size]
    return meta, dead.astype(np.int32), n_vertices, n_side_dead + int(dead.size)


def _corner_dead(
    corner_a: np.ndarray, corner_b: np.ndarray, corner_c: np.ndarray, corner_d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Whether each cell's ``(a, c, b)`` and ``(a, d, c)`` triangles repeat a corner key."""
    first_dead = (corner_a == corner_c) | (corner_c == corner_b) | (corner_b == corner_a)
    second_dead = (corner_a == corner_d) | (corner_d == corner_c) | (corner_c == corner_a)
    return first_dead, second_dead


def _parametric_keys(gluing: int, i: np.ndarray, j: np.ndarray, n_u: int, n_v: int) -> np.ndarray:
    """
    Canonical key ``i_canonical * n_v + j_canonical`` of each lattice sample ``(i, j)``.

    The host form of ``kernels/creation.parametric_canonical_key``, with the same two orderings:
    the ``u``-seam re-canonicalisation after a ``v``-twist is unmasked, and the four pole masks are
    snapshots of the post-wrap state taken before any of them collapses anything. A wrapped
    boundary's last row *is* its first; a twist reverses the other index on the way across, which
    is precisely what makes Moebius, Klein, Boy, Roman and the cross-cap non-orientable.
    """
    u_wrap = gluing & _GLUE_U_WRAP != 0
    if u_wrap:
        seam = i == n_u - 1
        if gluing & _GLUE_U_TWIST:
            j = np.where(seam, n_v - 1 - j, j)
        i = np.where(seam, 0, i)
    if gluing & _GLUE_V_WRAP:
        seam = j == n_v - 1
        if gluing & _GLUE_V_TWIST:
            i = np.where(seam, n_u - 1 - i, i)
        j = np.where(seam, 0, j)
        if u_wrap:
            i = np.where(i == n_u - 1, 0, i)
    # Every mask first, then the collapses in order, a later pole overriding an earlier one.
    poles = []
    if gluing & _GLUE_POLE_J_LO:
        poles.append((j == 0, 0, 0))
    if gluing & _GLUE_POLE_J_HI:
        poles.append((j == n_v - 1, 0, n_v - 1))
    if gluing & _GLUE_POLE_I_LO:
        poles.append((i == 0, 0, 0))
    if gluing & _GLUE_POLE_I_HI:
        poles.append((i == n_u - 1, n_u - 1, 0))
    for on_pole, anchor_i, anchor_j in poles:
        i = np.where(on_pole, anchor_i, i)
        j = np.where(on_pole, anchor_j, j)
    return i * n_v + j


def _resolve_cylinder_axis(
    height: float | None, segment: Sequence[Sequence[float]] | None, transform: wp.mat44 | None
) -> tuple[wp.mat44 | None, float]:
    """
    Resolve a cylinder-like body's axis to ``(transform, half_height)``.

    ``segment`` and ``height`` are two spellings of the same axis, shared by
    [`cylinder`][ordito.creation.cylinder] and [`annulus`][ordito.creation.annulus]. A segment
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
    profile_np: np.ndarray, sections: int, transform: wp.mat44 | None, device: wp.DeviceLike
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
            *_placement(transform),
        ],
        outputs=[vertices, faces],
        device=device,
    )
    return vertices, faces


def _closed_form_template(profile_np: np.ndarray, n_sections: int) -> tuple[int, int]:
    """
    ``(keep_caps, keep_sides)``: which of a solid-of-revolution's template triangles survive.

    The closed-form kernels behind [`cone`][ordito.creation.cone] and
    [`cylinder`][ordito.creation.cylinder] write a fixed layout, so they need
    [`revolve`][ordito.creation.revolve]'s degenerate-area verdict as two flags rather than as an
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


def _placement(transform: wp.mat44 | None) -> tuple[wp.mat44, wp.int32]:
    """
    Return the ``(matrix, transform_mode)`` pair a generator kernel places its output with.

    The kernel applies ``matrix`` to every vertex and reverses every triangle when the matrix
    mirrors, at write time (``kernels/creation.place_point``), so a transformed primitive is
    still one launch.
    """
    if transform is None:
        return _IDENTITY_MATRIX, kernel_creation.TRANSFORM_NONE
    if od.transform.reverses_orientation(transform):
        return transform, kernel_creation.TRANSFORM_MIRROR
    return transform, kernel_creation.TRANSFORM_KEEP


def _upload_points(points_np: np.ndarray, dtype: type, device: wp.DeviceLike) -> odt.ArrayNd:
    """
    Upload an ``(n, 2)`` or ``(n, 3)`` host point table as a ``wp.vec2`` / ``wp.vec3`` buffer.

    Vertex tables and revolution profiles are both built on the host in ``float64`` and downcast
    here. Evaluating a profile on the device in ``float32`` instead would put the closing point of
    a closed profile roughly ``1e-7`` away from its first point — outside
    [`revolve`][ordito.creation.revolve]'s weld tolerance — while ``float64`` keeps the gap near
    ``1e-16``. These tables are tiny; the per-slice work they drive is what runs on the device.
    """
    return _launch.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=dtype, device=device
    )
