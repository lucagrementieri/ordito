"""
Input data shared by every example: scanned meshes, derived defects, point clouds, camera presets.

Every example on the gallery runs on one of the inputs below, so related examples show the *same*
shape from the *same* viewpoint. Each input is a NumPy ``(vertices, faces)`` pair built once and
cached; [`load`][] uploads it to a Warp device in ordito's layout (``wp.vec3`` vertices and a flat
``wp.int32`` face buffer). Everything random is seeded, so the gallery images are reproducible.

The scans (bunny, dragon, happy Buddha, armadillo) are from the Stanford 3D Scanning Repository
(https://graphics.stanford.edu/data/3Dscanrep/). They are not committed: ``python -m
examples.fetch_data`` downloads them into ``benchmarks/data/``.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import warp as wp

from examples._render import Camera

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [
    Path(os.environ["ORDITO_EXAMPLE_DATA"]) if "ORDITO_EXAMPLE_DATA" in os.environ else None,
    ROOT / "benchmarks" / "data",
]

MeshArrays = tuple[np.ndarray, np.ndarray]
"""``(n, 3)`` float32 vertices and ``(m, 3)`` int32 faces."""

_BUILDERS: dict[str, Callable[[], MeshArrays]] = {}
CAMERAS: dict[str, Camera] = {}


def _register(
    name: str, camera: Camera | None = None
) -> Callable[[Callable[[], MeshArrays]], Callable[[], MeshArrays]]:
    def wrap(builder: Callable[[], MeshArrays]) -> Callable[[], MeshArrays]:
        _BUILDERS[name] = builder
        if camera is not None:
            CAMERAS[name] = camera
        return builder

    return wrap


@functools.cache
def arrays(name: str) -> MeshArrays:
    """Return the NumPy ``(vertices, faces)`` of input ``name``."""
    vertices, faces = _BUILDERS[name]()
    return (
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(np.asarray(faces).reshape(-1, 3), dtype=np.int32),
    )


def load(name: str, device: wp.DeviceLike = None) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Upload input ``name`` as ordito's ``(wp.vec3 vertices, flat wp.int32 faces)``."""
    vertices, faces = arrays(name)
    return (
        wp.array(vertices, dtype=wp.vec3, device=device),
        wp.array(faces.ravel(), dtype=wp.int32, device=device),
    )


def camera(name: str) -> Camera:
    """Return the camera preset that frames input ``name``."""
    return CAMERAS.get(name, Camera(direction=(0.6, 0.45, 1.0), up=(0.0, 1.0, 0.0)))


def find_file(filename: str) -> Path:
    """Return the path of a downloaded scan, raising a hint to run the fetch script if missing."""
    for directory in DATA_DIRS:
        if directory is not None and (directory / filename).exists():
            return directory / filename
    raise FileNotFoundError(f"{filename} not found; run `python -m examples.fetch_data` first")


def _read_ply(filename: str) -> MeshArrays:
    import trimesh as tm

    # trimesh rather than meshio: meshio rejects the armadillo's per-face ``intensity`` property.
    mesh = cast("tm.Trimesh", tm.load(find_file(filename), process=False, force="mesh"))
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _compact(vertices: np.ndarray, faces: np.ndarray) -> MeshArrays:
    """Drop unreferenced vertices and renumber ``faces``."""
    used, inverse = np.unique(faces.ravel(), return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals (host-side, for building inputs only)."""
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length > 0, length, 1.0)


def mean_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Mean edge length over face corners (host-side)."""
    tri = vertices[faces]
    edges = np.concatenate([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]])
    return float(np.linalg.norm(edges, axis=1).mean())


def rotation(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    """Return the ``3 x 3`` rotation about ``axis`` by ``degrees``."""
    a = np.asarray(axis, dtype=np.float64)
    a /= np.linalg.norm(a)
    t = np.radians(degrees)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)


# --------------------------------------------------------------------------------------------
# Scanned meshes
# --------------------------------------------------------------------------------------------

_BUNNY_VIEW = Camera(direction=(0.25, 0.25, 1.0), up=(0.0, 1.0, 0.0), zoom=1.15)


@_register("bunny", _BUNNY_VIEW)
def _bunny() -> MeshArrays:
    """
    Return the Stanford bunny (69 451 faces), as scanned: five small holes in its base.

    ``bunny_decimated.ply`` from the benchmark set is not used: its decimation left non-manifold
    edges, which would make every repair example start from a defect it does not show.
    """
    return _compact(*_read_ply("bunny.ply"))


@_register("dragon", Camera(direction=(0.15, 0.2, 1.0), up=(0.0, 1.0, 0.0), zoom=1.2))
def _dragon() -> MeshArrays:
    """Return the Stanford dragon (871 414 faces)."""
    return _compact(*_read_ply("dragon.ply"))


@_register("buddha", Camera(direction=(0.0, 0.1, 1.0), up=(0.0, 1.0, 0.0), zoom=1.1))
def _buddha() -> MeshArrays:
    """Return the Stanford happy Buddha (1 087 716 faces)."""
    return _compact(*_read_ply("happy_buddha.ply"))


@_register("armadillo", Camera(direction=(0.0, 0.1, -1.0), up=(0.0, 1.0, 0.0), zoom=1.1))
def _armadillo() -> MeshArrays:
    """Return the Stanford armadillo (345 944 faces)."""
    return _compact(*_read_ply("Armadillo.ply"))


# --------------------------------------------------------------------------------------------
# Derived defects on the bunny
# --------------------------------------------------------------------------------------------


@_register("noisy_bunny", _BUNNY_VIEW)
def _noisy_bunny() -> MeshArrays:
    """Return the bunny displaced along its normals by Gaussian noise (0.3 mean edge lengths)."""
    vertices, faces = arrays("bunny")
    rng = np.random.default_rng(7)
    sigma = 0.3 * mean_edge_length(vertices, faces)
    normals = vertex_normals(vertices, faces)
    return vertices + normals * rng.normal(0.0, sigma, (vertices.shape[0], 1)), faces


def _ball_faces(
    vertices: np.ndarray, faces: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
    centroids = vertices[faces].mean(axis=1)
    return np.linalg.norm(centroids - center, axis=1) < radius


_HOLE_CENTERS = [  # (vertex index on bunny_small, radius as a fraction of the bbox diagonal)
    (0.45, 0.05),
    (0.15, 0.035),
    (0.7, 0.025),
    (0.85, 0.06),
    (0.3, 0.02),
    (0.6, 0.015),
]


@_register("holey_bunny", _BUNNY_VIEW)
def _holey_bunny() -> MeshArrays:
    """Return the bunny with six holes of mixed sizes punched through its visible side."""
    vertices, faces = arrays("bunny")
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    normals = vertex_normals(vertices, faces)
    # Candidate centres: vertices facing the default camera, ordered by height.
    facing = np.flatnonzero(normals @ np.array([0.25, 0.25, 1.0]) > 0.6)
    facing = facing[np.argsort(vertices[facing, 1])]
    remove = np.zeros(faces.shape[0], dtype=bool)
    for fraction, radius in _HOLE_CENTERS:
        center = vertices[facing[int(fraction * (facing.size - 1))]]
        remove |= _ball_faces(vertices, faces, center, radius * diagonal)
    return _compact(vertices, faces[~remove])


@_register("bunny_debris", _BUNNY_VIEW)
def _bunny_debris() -> MeshArrays:
    """Return the bunny surrounded by 40 small floating blobs (scan debris)."""
    vertices, faces = arrays("bunny")
    rng = np.random.default_rng(3)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    diagonal = float(np.linalg.norm(hi - lo))
    blob_v, blob_f = _icosphere(1)
    parts_v, parts_f, offset = [vertices], [faces], vertices.shape[0]
    for _ in range(40):
        center = rng.uniform(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        scale = diagonal * rng.uniform(0.004, 0.015)
        squash = rng.uniform(0.5, 1.5, 3)
        parts_v.append(blob_v * scale * squash + center)
        parts_f.append(blob_f + offset)
        offset += blob_v.shape[0]
    return np.concatenate(parts_v), np.concatenate(parts_f)


@_register("flipped_bunny", _BUNNY_VIEW)
def _flipped_bunny() -> MeshArrays:
    """Return the bunny with 20 % of its faces' winding reversed."""
    vertices, faces = arrays("bunny")
    flip = np.random.default_rng(11).random(faces.shape[0]) < 0.2
    faces = faces.copy()
    faces[flip] = faces[flip][:, ::-1]
    return vertices, faces


@_register("wrecked_bunny", _BUNNY_VIEW)
def _wrecked_bunny() -> MeshArrays:
    """Return the bunny with holes, flipped faces, debris and an unwelded band of faces."""
    vertices, faces = arrays("holey_bunny")
    rng = np.random.default_rng(5)
    faces = faces.copy()
    flip = rng.random(faces.shape[0]) < 0.05
    faces[flip] = faces[flip][:, ::-1]
    # Unweld one band of faces: give it its own copy of its vertices.
    band = np.abs(vertices[faces].mean(axis=1)[:, 1] - np.median(vertices[:, 1])) < 0.004
    band_vertices = np.unique(faces[band])
    remap = np.full(vertices.shape[0], -1)
    remap[band_vertices] = vertices.shape[0] + np.arange(band_vertices.size)
    faces[band] = remap[faces[band]]
    vertices = np.concatenate([vertices, vertices[band_vertices]])
    debris_v, debris_f = arrays("bunny_debris")
    n_base = arrays("bunny")[0].shape[0]
    extra_f = debris_f[np.all(debris_f >= n_base, axis=1)] - n_base + vertices.shape[0]
    return np.concatenate([vertices, debris_v[n_base:]]), np.concatenate([faces, extra_f])


# --------------------------------------------------------------------------------------------
# Point clouds (faces are empty)
# --------------------------------------------------------------------------------------------


def sample_points(name: str, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``count`` area-uniform surface samples of input ``name`` and their face normals."""
    vertices, faces = arrays(name)
    tri = vertices[faces].astype(np.float64)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    rng = np.random.default_rng(seed)
    face = rng.choice(faces.shape[0], size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[face]
    points = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    normals = cross[face] / np.linalg.norm(cross[face], axis=1, keepdims=True)
    return points.astype(np.float32), normals.astype(np.float32)


_NO_FACES = np.zeros((0, 3), dtype=np.int32)


@_register("bunny_cloud", _BUNNY_VIEW)
def _bunny_cloud() -> MeshArrays:
    """Return 30 000 surface samples of the bunny (see [`cloud_normals`][] for their normals)."""
    return sample_points("bunny", 30_000, seed=1)[0], _NO_FACES


@_register("noisy_cloud", _BUNNY_VIEW)
def _noisy_cloud() -> MeshArrays:
    """Return the bunny cloud with Gaussian jitter and 3 % uniform outliers in its bounding box."""
    points, _ = sample_points("bunny", 30_000, seed=1)
    rng = np.random.default_rng(2)
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    points = points + rng.normal(0.0, 0.002 * diagonal, points.shape)
    lo, hi = points.min(axis=0), points.max(axis=0)
    outliers = rng.uniform(lo, hi, (900, 3))
    return np.concatenate([points, outliers]), _NO_FACES


def cloud_normals(name: str) -> np.ndarray:
    """Return the exact normals of a sampled cloud input (``bunny_cloud``)."""
    if name != "bunny_cloud":
        raise KeyError(name)
    return sample_points("bunny", 30_000, seed=1)[1]


def load_points(name: str, device: wp.DeviceLike = None) -> wp.array[wp.vec3]:
    """Upload the points of a point-cloud input as a ``wp.vec3`` array."""
    return wp.array(arrays(name)[0], dtype=wp.vec3, device=device)


# --------------------------------------------------------------------------------------------
# Synthetic shapes
# --------------------------------------------------------------------------------------------


def _icosphere(subdivisions: int) -> MeshArrays:
    import trimesh as tm

    mesh = tm.creation.icosphere(subdivisions=subdivisions)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _parametric(kind: str, resolution: int = 80) -> MeshArrays:
    import ordito as od

    vertices, faces = od.creation.parametric_surface(kind, resolution, resolution, device="cpu")  # type: ignore[arg-type]
    return vertices.numpy(), faces.numpy().reshape(-1, 3)


@_register("mobius", Camera(direction=(0.4, -0.8, 0.6), up=(0.0, 0.0, 1.0)))
def _mobius() -> MeshArrays:
    """Return a Möbius strip: non-orientable, one boundary loop."""
    return _parametric("mobius")


@_register("boy", Camera(direction=(0.3, -0.6, 0.8), up=(0.0, 0.0, 1.0)))
def _boy() -> MeshArrays:
    """Return Boy's surface: closed and non-orientable, Euler characteristic 1."""
    return _parametric("boy")


@_register("broken_box", Camera(direction=(0.8, 0.6, 1.0), up=(0.0, 1.0, 0.0)))
def _broken_box() -> MeshArrays:
    """
    Return two cubes touching at one corner, plus a fin glued to one cube's edge.

    The shared corner is a non-manifold vertex; the fin makes an edge with three faces.
    """
    import trimesh as tm

    box = tm.creation.box(extents=(1.0, 1.0, 1.0))
    a_v, a_f = np.asarray(box.vertices), np.asarray(box.faces)
    b_v = a_v + 1.0  # shares corner (0.5, 0.5, 0.5)
    vertices = np.concatenate([a_v, b_v])
    faces = np.concatenate([a_f, a_f + 8])
    # Weld the shared corner.
    corner = np.flatnonzero(np.all(np.isclose(a_v, 0.5), axis=1))[0]
    other = 8 + np.flatnonzero(np.all(np.isclose(a_v, -0.5), axis=1))[0]
    faces[faces == other] = corner
    # A fin on the edge between (-0.5, -0.5, -0.5) and (-0.5, -0.5, 0.5) of cube A.
    e0 = np.flatnonzero(np.all(np.isclose(a_v, [-0.5, -0.5, -0.5]), axis=1))[0]
    e1 = np.flatnonzero(np.all(np.isclose(a_v, [-0.5, -0.5, 0.5]), axis=1))[0]
    tip = vertices.shape[0]
    vertices = np.concatenate([vertices, [[-1.2, -1.2, 0.0]]])
    faces = np.concatenate([faces, [[e0, e1, tip]]])
    return _compact(vertices, faces)


@_register("parts", Camera(direction=(0.5, 0.7, 1.0), up=(0.0, 1.0, 0.0)))
def _parts() -> MeshArrays:
    """Return six primitives packed into one mesh: six bodies."""
    import trimesh as tm

    meshes = [
        tm.creation.box(extents=(1.0, 0.6, 0.6)),
        tm.creation.icosphere(subdivisions=3, radius=0.45).apply_translation((1.05, 0.0, 0.0)),
        tm.creation.cylinder(radius=0.3, height=0.9, sections=48).apply_translation(
            (0.0, 0.8, 0.0)
        ),
        tm.creation.cone(radius=0.35, height=0.8, sections=48).apply_translation((1.0, 0.75, -0.3)),
        tm.creation.torus(0.45, 0.13, 48, 24).apply_translation((0.0, -0.75, 0.0)),
        tm.creation.capsule(height=0.6, radius=0.2, count=(24, 24)).apply_translation(
            (1.1, -0.8, 0.0)
        ),
    ]
    vertices, faces, offset = [], [], 0
    for mesh in meshes:
        vertices.append(np.asarray(mesh.vertices))
        faces.append(np.asarray(mesh.faces) + offset)
        offset += len(mesh.vertices)
    return np.concatenate(vertices), np.concatenate(faces)


@_register("cad_part", Camera(direction=(0.9, 0.8, 1.0), up=(0.0, 0.0, 1.0)))
def _cad_part() -> MeshArrays:
    """Return a machined-looking part: a slotted block with a round boss (sharp creases)."""
    import trimesh as tm

    block = tm.creation.box(extents=(2.0, 1.2, 0.5))
    boss = tm.creation.cylinder(radius=0.35, height=0.6, sections=64).apply_translation(
        (0.5, 0.0, 0.5)
    )
    slot = tm.creation.box(extents=(0.8, 0.3, 0.6)).apply_translation((-0.5, 0.0, 0.2))
    hole = tm.creation.cylinder(radius=0.15, height=2.0, sections=48).apply_translation(
        (0.5, 0.0, 0.0)
    )
    part = tm.boolean.difference([tm.boolean.union([block, boss]), slot, hole])
    return np.asarray(part.vertices), np.asarray(part.faces)


@_register("torus", Camera(direction=(0.0, -0.7, 1.0), up=(0.0, 0.0, 1.0)))
def _torus() -> MeshArrays:
    """Return a ring torus (major radius 1, minor radius 0.35)."""
    import trimesh as tm

    mesh = tm.creation.torus(1.0, 0.35, 128, 48)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


@_register("handles", Camera(direction=(0.3, -0.6, 1.0), up=(0.0, 0.0, 1.0)))
def _handles() -> MeshArrays:
    """Return a slab with 3 x 3 tunnels punched through it: a closed surface of genus 9."""
    from benchmarks.meshes import _handles as build  # pyright: ignore[reportPrivateUsage]

    return build(3, max_edge=0.3)


@_register("self_intersecting_torus", Camera(direction=(0.0, -0.7, 1.0), up=(0.0, 0.0, 1.0)))
def _self_intersecting_torus() -> MeshArrays:
    """Return a torus whose tube radius exceeds its ring radius (the tube crosses itself)."""
    import trimesh as tm

    mesh = tm.creation.torus(0.5, 0.7, 96, 48)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


@_register("two_spheres", Camera(direction=(0.3, 0.4, 1.0), up=(0.0, 1.0, 0.0)))
def _two_spheres() -> MeshArrays:
    """Return two overlapping unit icospheres, centres 1.2 apart."""
    v, f = _icosphere(4)
    return np.concatenate([v, v + np.array([1.2, 0.0, 0.0])]), np.concatenate([f, f + v.shape[0]])


@_register("hemisphere", Camera(direction=(0.3, -0.7, 0.8), up=(0.0, 0.0, 1.0)))
def _hemisphere() -> MeshArrays:
    """Return the upper half of an icosphere: a curved disk with one rim."""
    from benchmarks.meshes import _hemisphere as build  # pyright: ignore[reportPrivateUsage]

    return build(4)


@_register("open_parts", Camera(direction=(0.3, 0.4, 1.0), up=(0.0, 1.0, 0.0)))
def _open_parts() -> MeshArrays:
    """Return four open spheres on a lattice, each missing one face."""
    from benchmarks.meshes import _open_spheres as build  # pyright: ignore[reportPrivateUsage]

    return build(3, 4)


@_register("knot", Camera(direction=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0)))
def _knot() -> MeshArrays:
    """Return a trefoil knot tube."""
    return _trefoil_tube(0.4)


def _trefoil_tube(radius: float) -> MeshArrays:
    """Sweep a circle of ``radius`` along a closed trefoil path."""
    import ordito as od

    t = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    path = np.stack(
        [np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), -np.sin(3 * t)], 1
    )
    angle = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    circle = radius * np.stack([np.cos(angle), np.sin(angle)], 1)
    vertices, faces = od.creation.sweep_polygon(
        wp.array(circle, dtype=wp.vec2, device="cpu"),
        wp.array(np.vstack([path, path[:1]]), dtype=wp.vec3, device="cpu"),
    )
    return vertices.numpy(), faces.numpy().reshape(-1, 3)


@_register("scene", Camera(direction=(0.0, 0.5, 1.0), up=(0.0, 1.0, 0.0)))
def _scene() -> MeshArrays:
    """Return the bunny (scaled to unit height), a torus and a box standing on a ground plate."""
    import trimesh as tm

    bunny_v, bunny_f = arrays("bunny")
    lo, hi = bunny_v.min(axis=0), bunny_v.max(axis=0)
    bunny_v = (bunny_v - [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]) / (hi[1] - lo[1])
    torus = tm.creation.torus(0.35, 0.12, 96, 32)
    torus.apply_transform(tm.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    torus.apply_translation((1.1, 0.12, 0.3))
    box = tm.creation.box(extents=(0.5, 0.5, 0.5)).apply_translation((-1.0, 0.25, -0.2))
    box.apply_transform(tm.transformations.rotation_matrix(0.5, [0, 1, 0], point=(-1.0, 0, -0.2)))
    ground = tm.creation.box(extents=(3.6, 0.04, 2.4)).apply_translation((0.0, -0.02, 0.0))
    parts = [(bunny_v, bunny_f)] + [
        (np.asarray(m.vertices), np.asarray(m.faces)) for m in (torus, box, ground)
    ]
    vertices, faces, offset = [], [], 0
    for v, f in parts:
        vertices.append(v)
        faces.append(f + offset)
        offset += v.shape[0]
    return np.concatenate(vertices), np.concatenate(faces)


@_register("fat_knot", Camera(direction=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0)))
def _fat_knot() -> MeshArrays:
    """Return a trefoil tube too thick for its path: the tube passes through itself at crossings."""
    return _trefoil_tube(0.75)


@_register("tilted_bunny_cloud", _BUNNY_VIEW)
def _tilted_bunny_cloud() -> MeshArrays:
    """Return the bunny cloud rotated 40 degrees about a skew axis (not aligned with any axis)."""
    points, _ = sample_points("bunny", 30_000, seed=1)
    center = points.mean(axis=0)
    return (points - center) @ rotation((0.3, 0.2, 1.0), 40.0).T + center, _NO_FACES


@_register("sliver_patch", Camera(direction=(0.0, -0.35, 1.0), up=(0.0, 1.0, 0.0), zoom=1.45))
def _sliver_patch() -> MeshArrays:
    """
    Return a bumpy grid triangulated badly: sheared rows cut along their long diagonals.

    Every cell's two triangles are obtuse, so most interior edges fail the Delaunay test and carry
    a negative cotangent weight. A few faces are split further into needles (a vertex next to a
    corner), caps (a vertex just off an edge) and exactly degenerate triangles (a vertex on an
    edge).
    """
    n = 14
    j, i = np.mgrid[0 : n + 1, 0 : n + 1]
    x = (i + np.where(j % 2 == 1, 0.42, -0.42)) / n
    y = j / n
    z = 0.12 * np.sin(np.pi * x) * np.sin(np.pi * y)
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    faces = []
    for jj in range(n):
        for ii in range(n):
            a, b = jj * (n + 1) + ii, jj * (n + 1) + ii + 1
            c, d = b + n + 1, a + n + 1
            long_ac = np.linalg.norm(vertices[a] - vertices[c])
            if long_ac > np.linalg.norm(vertices[b] - vertices[d]):
                faces += [[a, b, c], [a, c, d]]
            else:
                faces += [[a, b, d], [b, c, d]]
    faces = np.array(faces)
    rng = np.random.default_rng(4)
    picked = rng.choice(faces.shape[0], 12, replace=False)
    extra, keep = [], np.ones(faces.shape[0], dtype=bool)
    new_faces = []
    for k, face in enumerate(picked):
        a, b, c = faces[face]
        pa, pb, pc = vertices[[a, b, c]]
        if k % 3 == 0:  # needle: a vertex right next to corner a
            point = pa + 1e-4 * (pb + pc - 2 * pa)
        elif k % 3 == 1:  # cap: a vertex just off the middle of edge ab
            point = 0.5 * (pa + pb) + 1e-4 * (pc - 0.5 * (pa + pb))
        else:  # exactly degenerate: a vertex on edge ab, splitting only this face
            point = 0.5 * (pa + pb)
        m = vertices.shape[0] + len(extra)
        extra.append(point)
        keep[face] = False
        new_faces += [[a, b, m], [b, c, m], [c, a, m]]
    vertices = np.concatenate([vertices, np.array(extra)])
    return vertices, np.concatenate([faces[keep], np.array(new_faces)])


@_register("t_vertex_patch", Camera(direction=(0.0, -0.35, 1.0), up=(0.0, 1.0, 0.0), zoom=1.3))
def _t_vertex_patch() -> MeshArrays:
    """
    Return two grids stitched at different resolutions: a coarse left half and a fine right half.

    The halves are not welded (each carries its own copy of the seam vertices), and every fine
    seam vertex between two coarse ones is a T-vertex whose crack is closed by a sliver triangle.
    """

    def grid(x0: float, cells: int, offset: int) -> tuple[np.ndarray, list[list[int]]]:
        j, i = np.mgrid[0 : cells + 1, 0 : cells + 1]
        x, y = x0 + i / cells, j / cells
        z = 0.08 * np.sin(np.pi * y) * np.cos(0.5 * np.pi * (x - 1.0))
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        tris = []
        for jj in range(cells):
            for ii in range(cells):
                a = offset + jj * (cells + 1) + ii
                b, d = a + 1, a + cells + 1
                tris += [[a, b, d + 1], [a, d + 1, d]]
        return points, tris

    coarse, coarse_faces = grid(0.0, 4, 0)
    fine, fine_faces = grid(1.0, 8, coarse.shape[0])
    slivers = []
    for k in range(4):  # coarse seam nodes (column 4) at rows k, k + 1; fine middle node row 2k + 1
        a, b = k * 5 + 4, (k + 1) * 5 + 4
        m = coarse.shape[0] + (2 * k + 1) * 9
        slivers.append([b, a, m])  # the coarse face runs a -> b, so the sliver runs b -> a
    return np.concatenate([coarse, fine]), np.array(coarse_faces + fine_faces + slivers)


def _cup(radius: float, sections: int, rings: int, height: float) -> MeshArrays:
    """Return a tube closed at ``z = 0`` by a fan and open at ``z = height``."""
    angle = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    z = np.linspace(0.0, height, rings + 1)
    ring = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    tube = np.concatenate([np.column_stack([ring, np.full(sections, h)]) for h in z])
    vertices = np.concatenate([tube, [[0.0, 0.0, 0.0]]])
    faces = []
    for k in range(rings):
        for i in range(sections):
            a, b = k * sections + i, k * sections + (i + 1) % sections
            faces += [[a, b, b + sections], [a, b + sections, a + sections]]
    center = vertices.shape[0] - 1
    faces += [[center, (i + 1) % sections, i] for i in range(sections)]
    return vertices, np.array(faces)


_CUPS_VIEW = Camera(direction=(1.0, -0.7, 0.55), up=(0.0, 0.0, 1.0), zoom=1.5)


@_register("cup_bottom", _CUPS_VIEW)
def _cup_bottom() -> MeshArrays:
    """Return an upright cup (radius 0.5, 64 sections), open at the top: one rim."""
    return _cup(0.5, 64, 8, 1.0)


@_register("cup_top", _CUPS_VIEW)
def _cup_top() -> MeshArrays:
    """Return a narrower, coarser cup upside down and tilted above ``cup_bottom``: one rim."""
    vertices, faces = _cup(0.35, 40, 6, 0.8)
    vertices = vertices * [1.0, 1.0, -1.0] + [0.0, 0.0, 0.8]  # mouth down
    faces = faces[:, ::-1]  # the mirror reversed the winding
    vertices = vertices @ rotation((0.0, 1.0, 0.0), 15.0).T + [0.15, 0.0, 1.55]
    return vertices, faces


@_register("face_patch", Camera(direction=(0.1, 0.1, 1.0), up=(0.0, 1.0, 0.0), zoom=1.1))
def _face_patch() -> MeshArrays:
    """
    Return a disk cut from the bunny's flank: the faces within 0.07 of a flank vertex.

    The patch is one connected piece with a single boundary loop and Euler characteristic 1,
    which is what a parametrization into the plane needs.
    """
    vertices, faces = arrays("bunny")
    center = vertices[5088]
    keep = np.linalg.norm(vertices[faces].mean(axis=1) - center, axis=1) < 0.07
    vertices, faces = _compact(vertices, faces[keep])
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    n_edges = np.unique(edges, axis=0).shape[0]
    assert vertices.shape[0] - n_edges + faces.shape[0] == 1, "the patch is not a disk"
    return vertices, faces


@_register("torn_hemisphere", Camera(direction=(0.1, -0.25, 1.0), up=(0.0, 1.0, 0.0), zoom=1.2))
def _torn_hemisphere() -> MeshArrays:
    """Return the hemisphere torn into four pieces by two cracks that narrow toward the pole."""
    vertices, faces = arrays("hemisphere")
    centroids = vertices[faces].mean(axis=1)
    width = 0.012 + 0.06 * np.linalg.norm(centroids[:, :2], axis=1)  # cracks widen outwards
    strip = (np.abs(centroids[:, 0] - 0.1) < width) | (np.abs(centroids[:, 1] + 0.05) < width)
    return _compact(vertices, faces[~strip])


@_register("terrain_points", Camera(direction=(0.6, -1.0, 0.9), up=(0.0, 0.0, 1.0), zoom=1.1))
def _terrain_points() -> MeshArrays:
    """Return 4 000 scattered survey points of a random height field (``random_hills``)."""
    import ordito as od

    vertices, _ = od.creation.random_hills(seed=4, u_resolution=100, v_resolution=100, device="cpu")
    points = vertices.numpy()
    rng = np.random.default_rng(4)
    points[:, :2] += rng.uniform(-0.06, 0.06, (len(points), 2))  # a third of a lattice cell
    return points[rng.permutation(len(points))[:4000]], _NO_FACES


@_register("spiky_torus", Camera(direction=(0.0, -0.7, 1.0), up=(0.0, 0.0, 1.0)))
def _spiky_torus() -> MeshArrays:
    """Return the ``torus`` with 40 single vertices pulled out along their normals into needles."""
    vertices, faces = arrays("torus")
    rng = np.random.default_rng(9)
    spikes = rng.choice(vertices.shape[0], 40, replace=False)
    vertices = vertices.astype(np.float64).copy()
    normals = vertex_normals(vertices, faces)
    vertices[spikes] += normals[spikes] * rng.uniform(0.15, 0.4, (40, 1))
    return vertices, faces


@_register("bunny_cloud_displaced", _BUNNY_VIEW)
def _bunny_cloud_displaced() -> MeshArrays:
    """Return the bunny cloud pushed along its normals by a smooth bump field (up to ~3 mm)."""
    points, normals = sample_points("bunny", 30_000, seed=1)
    bump = np.exp(-np.sum((points - np.array([0.02, 0.1, 0.05])) ** 2, axis=1) / (2 * 0.02**2))
    wave = 0.3 * np.sin(60.0 * points[:, 1])
    return points + normals * (0.003 * bump + 0.001 * wave)[:, None], _NO_FACES


_SCAN_ROTATION = (0.3, 1.0, 0.1), 15.0
_SCAN_SHIFT = np.array([0.012, -0.006, 0.008])


@_register("scan_a", _BUNNY_VIEW)
def _scan_a() -> MeshArrays:
    """Return a partial scan of the bunny: its surface samples with ``x < 0.03``."""
    points, _ = sample_points("bunny", 30_000, seed=21)
    return points[points[:, 0] < 0.03], _NO_FACES


def _scan_b_points() -> np.ndarray:
    points, _ = sample_points("bunny", 30_000, seed=22)
    points = points[points[:, 0] > -0.05].astype(np.float64)
    points += np.random.default_rng(23).normal(0.0, 0.0003, points.shape)
    center = points.mean(axis=0)
    return (points - center) @ rotation(*_SCAN_ROTATION).T + center + _SCAN_SHIFT


@_register("scan_b", _BUNNY_VIEW)
def _scan_b() -> MeshArrays:
    """
    Return a second partial scan of the bunny, overlapping ``scan_a``.

    Its samples with ``x > -0.05``, jittered, then turned 15 degrees and shifted, as a scanner
    moved between two captures would leave it.
    """
    return _scan_b_points(), _NO_FACES


@_register("scan_b_outliers", _BUNNY_VIEW)
def _scan_b_outliers() -> MeshArrays:
    """Return ``scan_b`` plus 15 % stray points in a box around it (a cluttered capture)."""
    points = _scan_b_points()
    rng = np.random.default_rng(24)
    lo, hi = points.min(axis=0), points.max(axis=0)
    margin = 0.2 * (hi - lo)
    strays = rng.uniform(lo - margin, hi + margin, (int(0.15 * points.shape[0]), 3))
    return np.concatenate([points, strays]), _NO_FACES


@_register("thin_handles", Camera(direction=(0.0, -0.8, 1.0), up=(0.0, 0.0, 1.0), zoom=1.2))
def _thin_handles() -> MeshArrays:
    """
    Return a ring torus (one intended tunnel) with three thin handles fused onto its outside.

    The handles imitate a scanning artifact where two nearby sheets were bridged: genus 4 in all,
    but only the ring's own tunnel is meant to be there.
    """
    import trimesh as tm

    ring = tm.creation.torus(1.0, 0.4, 96, 48)
    handles = []
    for angle in (0.3, 2.4, 4.2):
        handle = tm.creation.torus(0.22, 0.045, 48, 12)
        handle.apply_transform(tm.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        handle.apply_translation((1.0, 0.0, 0.4))  # astride the top of the tube
        handle.apply_transform(tm.transformations.rotation_matrix(angle, [0, 0, 1]))
        handles.append(handle)
    mesh = tm.boolean.union([ring, *handles])
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def names() -> list[str]:
    """Return every registered input name."""
    return sorted(_BUILDERS)
