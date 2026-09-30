"""
Benchmark mesh registries: the real scan meshes and the synthetic feature meshes.

Two registries, for two different questions.

``MESHES`` holds the scan meshes read from ``benchmarks/data/``. They are indexed by triangle
count and answer "how does this scale with ``N``", which is the whole story only for the
throughput functions — per-face passes, reductions, edge sorts.

``FEATURE_MESHES`` answers every other question. Cost in this package is far more often driven by
something that a face count cannot express: how many connected components there are, how many
boundary loops and how long, the diameter of the adjacency graph, triangle aspect ratio, vertex
valence. So the feature registry is built around a single **control** mesh, ``sphere_med``
(``icosphere(6)``: 40 962 vertices, 81 920 faces, one watertight component, uniform valence 6,
graph diameter ~130, no boundary, aspect ratio ~1.4), and a set of meshes that each perturb
*exactly one* of those properties while pinning ``V`` and/or ``F`` to the control. A group's
timing spread then attributes to a named cause rather than to "a different mesh".

``AXES`` names the resulting comparisons and is what ``@pytest.mark.benchaxis`` selects. The
control comes first in each tuple, so a results table reads left-to-right as "baseline, then the
perturbation".

Meshes are built with ``trimesh.creation``, never with ``triwarp.creation``: a benchmark input
must not depend on the code under test, and ``conftest._load_numpy`` exists so that trimesh, igl,
open3d and warp all receive one shared NumPy source. ``triwarp.creation`` is the *subject* of
``test_creation``, never the supplier of another module's input.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import trimesh as tm

DATA_DIR = Path(__file__).parent / "data"

# Ordered from smallest to largest -- used for size comparisons.
SIZE_ORDER = ["small", "medium", "large", "extralarge", "huge"]


def size_category(n_faces: int) -> str:
    """Classify a mesh by triangle count: <10k, <100k, <1M, <10M, else huge."""
    if n_faces < 10_000:
        return "small"
    if n_faces < 100_000:
        return "medium"
    if n_faces < 1_000_000:
        return "large"
    if n_faces < 10_000_000:
        return "extralarge"
    return "huge"


class MeshSpec(TypedDict):
    """
    A benchmark mesh, described without building it.

    ``n_vertices`` / ``n_faces`` are recorded rather than measured so a mesh can be size-filtered
    or asserted against before it is read off disk or generated; ``test_meshes.py`` checks every
    recorded number against the real thing. ``filename`` is empty for synthetic meshes and
    ``axis`` is the property the mesh perturbs (``None`` for the scan meshes, which vary only in
    size).
    """

    name: str
    filename: str
    n_vertices: int
    n_faces: int
    size: str
    axis: str | None


def _mesh(
    name: str, filename: str, n_vertices: int, n_faces: int, axis: str | None = None
) -> MeshSpec:
    return {
        "name": name,
        "filename": filename,
        "n_vertices": n_vertices,
        "n_faces": n_faces,
        "size": size_category(n_faces),
        "axis": axis,
    }


# ---------------------------------------------------------------------------
# scan meshes: the pure-N axis
# ---------------------------------------------------------------------------

# ``heptoroid.ply`` is intentionally absent: it is a tristrip PLY that meshio cannot decode as
# triangle faces. Counts are read from the PLY headers so a mesh can be skipped by size before it
# is ever loaded.
MESHES: list[MeshSpec] = [
    _mesh("bunny_decimated", "bunny_decimated.ply", 8_171, 16_301),
    _mesh("bunny", "bunny.ply", 35_947, 69_451),
    _mesh("dragon", "dragon.ply", 437_645, 871_414),
    _mesh("happy_buddha", "happy_buddha.ply", 543_652, 1_087_716),
    _mesh("lucy", "lucy.ply", 14_027_872, 28_055_742),
]
MESHES_BY_NAME = {mesh["name"]: mesh for mesh in MESHES}
MESH_ORDER = [mesh["name"] for mesh in MESHES]


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

# Lattice pitch for ``_spheres(layout="lattice")``: unit spheres, so anything above 2 keeps the
# copies disjoint. Offset for ``layout="overlap"``: half a radius, deep enough that the two shells
# intersect in a broad band rather than grazing.
_LATTICE_PITCH = 3.0
_OVERLAP_OFFSET = 0.5

# Innermost radius and radial step for ``_spheres(layout="concentric")``.
_SHELL_INNER_RADIUS, _SHELL_STEP = 0.3, 0.1

# Vertices per boundary rim of ``rim_long``. Its two rims are the asymptotic case for boundary
# loop ranking: O(L log L) pointer jumping against a per-vertex successor walk.
RIM_LONG = 1 << 16
RIM_SHORT = 512

# Isolated triangles punched out of ``holes_many``: as many short loops as ``rim_long`` has long
# ones, at a comparable total boundary length, so loop *count* is the only difference.
N_PUNCHED_HOLES = 512
_PUNCH_SEED = 0

# Holes in ``holes_dense``: the same sphere as ``holes_many`` with 16x the loops, for the costs
# whose axis is the loop count itself (per-loop host readbacks). It is near the ceiling the
# vertex-disjoint greedy punch can reach on this sphere, and it is the scale at which the per-loop
# form of the rim measurement ran to hundreds of milliseconds.
N_DENSE_HOLES = 8_192

# Grid resolutions for the saddle patches: ``k x k`` vertices give ``2 * (k - 1) ** 2`` faces.
SADDLE_SMALL, SADDLE_MEDIUM = 68, 133

# Vertices along the long axis of ``ribbon_long``. Two rows, so the vertex count matches
# ``sphere_med`` exactly and the graph diameter is the only thing that changed.
RIBBON_LENGTH = 20_481

_Arrays = tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]


def _tangled_torus(major_sections: int, minor_sections: int) -> _Arrays:
    """
    Build a torus whose tube is wider than its hole, so the surface passes through itself.

    The registry's other self-intersecting input, ``tangle_2``, is two overlapping spheres and is
    therefore **two components** -- which several repair references decline outright rather than
    handle (MeshLib's ``localFixSelfIntersections`` returns such a mesh unchanged). This one is a
    single closed component that genuinely crosses itself, which is what a repair comparison needs
    if both sides are to do work. ``minor_radius > major_radius`` is the whole trick: the tube
    sweeps through the axis and the inner wall crosses the outer one in a band.

    Self-intersection scales with the resolution rather than with the radii -- a few percent of the
    surface at the small end and a fraction of a percent at the large one.
    """
    mesh = tm.creation.torus(1.0, 1.5, major_sections=major_sections, minor_sections=minor_sections)
    return (
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int64),
    )


def _spheres(subdivisions: int, count: int = 1, layout: str = "lattice") -> _Arrays:
    """
    ``count`` icospheres arranged as disjoint copies, concentric shells or an overlapping pair.

    One builder covers four axes because the arrangement is the only thing that changes: a cubic
    lattice gives disjoint components, concentric radii give ray-crossing depth, and a half-radius
    offset gives a dense self-intersection band. ``count=1`` is the plain icosphere.
    """
    if count == 1:
        mesh = tm.creation.icosphere(subdivisions=subdivisions)
        return mesh.vertices, mesh.faces

    parts: list[tm.Trimesh] = []
    if layout == "concentric":
        parts = [
            tm.creation.icosphere(
                subdivisions=subdivisions, radius=_SHELL_INNER_RADIUS + _SHELL_STEP * i
            )
            for i in range(count)
        ]
    else:
        side = int(np.ceil(count ** (1.0 / 3.0)))
        for i in range(count):
            part = tm.creation.icosphere(subdivisions=subdivisions)
            if layout == "overlap":
                part.apply_translation((_OVERLAP_OFFSET * i, 0.0, 0.0))
            else:
                cell = np.array([i % side, (i // side) % side, i // (side * side)], dtype=float)
                part.apply_translation(_LATTICE_PITCH * cell)
            parts.append(part)

    combined = tm.util.concatenate(parts)
    assert isinstance(combined, tm.Trimesh)
    return combined.vertices, combined.faces


def _open_spheres(subdivisions: int, count: int) -> _Arrays:
    """
    ``count`` disjoint icospheres on a lattice, each **opened** by dropping one face.

    The closed ``_spheres`` lattice cannot serve the component-joining functions: a shell with no
    boundary has nothing to bridge to, so ``join_closest_components`` returns ``parts_64``
    unchanged -- 0 bridges, a timed no-op. One face removed per shell is the smallest edit that
    gives each component a rim, and it leaves the vertex buffer alone (all three of a dropped
    face's vertices are still referenced by its neighbours), so the face count moves by exactly
    ``count`` and nothing else about the lattice changes.
    """
    vertices, faces = _spheres(subdivisions, count, "lattice")
    # ``concatenate`` keeps each part's faces contiguous, so face ``i * per`` belongs to shell i.
    per = len(faces) // count
    keep = np.ones(len(faces), dtype=bool)
    keep[np.arange(count) * per] = False
    return vertices, faces[keep]


def _open_cylinder(sections: int) -> _Arrays:
    """
    Uncapped tube: two boundary rims of ``sections`` vertices, ``2 * sections`` triangles.

    Built by dropping the cap fans from ``trimesh.creation.cylinder``. A cap face is exactly a
    face whose three vertices share a ``z``, which every side triangle fails by construction --
    a topological test rather than a valence heuristic, so it does not depend on how trimesh
    happens to order the cap vertices.
    """
    mesh = tm.creation.cylinder(radius=1.0, height=1.0, sections=sections)
    face_z = mesh.vertices[mesh.faces, 2]
    mesh.update_faces(face_z.max(axis=1) > face_z.min(axis=1))
    mesh.remove_unreferenced_vertices()
    return mesh.vertices, mesh.faces


def _punched_sphere(subdivisions: int, n_holes: int, seed: int = _PUNCH_SEED) -> _Arrays:
    """
    Icosphere with ``n_holes`` isolated triangles removed, giving that many 3-vertex loops.

    Faces are taken greedily from a seeded permutation, skipping any that touches a vertex already
    used, so no two holes share a vertex and every hole is a separate loop of length exactly 3.
    """
    mesh = tm.creation.icosphere(subdivisions=subdivisions)
    used = np.zeros(len(mesh.vertices), dtype=bool)
    keep = np.ones(len(mesh.faces), dtype=bool)
    punched = 0
    for face in np.random.default_rng(seed).permutation(len(mesh.faces)):
        triangle = mesh.faces[face]
        if used[triangle].any():
            continue
        used[triangle] = True
        keep[face] = False
        punched += 1
        if punched == n_holes:
            break
    if punched != n_holes:
        raise ValueError(f"only {punched} vertex-disjoint faces available, wanted {n_holes}")
    mesh.update_faces(keep)
    return mesh.vertices, mesh.faces


def _grid_patch(
    n_slow: int,
    n_fast: int,
    *,
    slow: npt.NDArray[np.floating],
    fast: npt.NDArray[np.floating],
    lift: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]]
    | None = None,
) -> _Arrays:
    """
    Disk-topology quad grid split into ``2 * (n_slow - 1) * (n_fast - 1)`` triangles.

    ``slow`` and ``fast`` are the coordinates along each axis, given explicitly so the same
    builder produces the uniform saddle, the graded saddle (identical connectivity, wildly
    different aspect ratios) and the two-row ribbon (identical vertex count to the control,
    wildly different graph diameter). Every vertex has valence at most 6 and one boundary loop
    runs around the rim.
    """
    u, v = np.meshgrid(slow, fast, indexing="ij")
    w = np.zeros_like(u) if lift is None else lift(u, v)
    vertices = np.column_stack((u.ravel(), v.ravel(), w.ravel()))
    i, j = np.meshgrid(np.arange(n_slow - 1), np.arange(n_fast - 1), indexing="ij")
    corner = (i * n_fast + j).ravel()
    faces = np.vstack(
        (
            np.column_stack((corner, corner + n_fast, corner + n_fast + 1)),
            np.column_stack((corner, corner + n_fast + 1, corner + 1)),
        )
    )
    return vertices, faces


def _saddle(k: int, *, graded: bool = False) -> _Arrays:
    """
    ``k x k`` grid lifted onto a saddle: a disk patch with non-degenerate cotangent weights.

    This is what the parametrization and geodesic solvers are actually for, and it is generated
    from NumPy alone so the face count is exact and independent of any mesh library's version.
    ``graded=True`` cubes the spacing along one axis, which leaves the vertex count, face count
    and connectivity untouched while pushing the worst triangle aspect ratio from ~1.6 to ~4 700 --
    the cleanest available handle on solver conditioning.
    """
    step = np.linspace(-1.0, 1.0, k)
    slow = np.sign(step) * np.abs(step) ** 3 if graded else step
    return _grid_patch(k, k, slow=slow, fast=step, lift=lambda x, y: 0.35 * (x * x - 0.6 * y * y))


def _ribbon(length: int) -> _Arrays:
    """
    Two-row strip whose graph diameter is its length, with square cells so quality is unchanged.

    Cell spacing is 1 along both axes on purpose: the ribbon must differ from ``sphere_med`` in
    diameter *only*. Stretching it into a long thin rectangle would confound the measurement with
    the aspect-ratio axis that ``saddle_graded`` already owns.
    """
    return _grid_patch(length, 2, slow=np.arange(float(length)), fast=np.arange(2.0), lift=None)


def _cone_fan(sections: int) -> _Arrays:
    """
    Cone with a ``sections``-valence apex and an equally high-valence base centre.

    Two hubs against the control's uniform valence 6, at the same vertex *and* face count, which
    isolates atomic-scatter contention in the per-vertex accumulations.
    """
    mesh = tm.creation.cone(radius=1.0, height=1.0, sections=sections)
    return mesh.vertices, mesh.faces


def _handles(holes_per_side: int, max_edge: float = 0.195, span: float = 16.0) -> _Arrays:
    """
    Punch ``holes_per_side ** 2`` tunnels through a closed slab: a surface of that genus.

    The only mesh family here whose perturbed property is **topological genus**, which nothing else
    in either registry has -- every scan mesh and every other feature mesh is genus 0, and
    ``triwarp.homology`` needs a closed surface with handles or it has nothing to find.

    Built in the order that makes the face count a smooth knob: subdivide the slab *first*, then cut
    the holes. Cutting first and subdividing after quantizes the count in powers of four, because
    the slab's six big faces subdivide together, which makes matching the two ends of the axis
    impossible. This way both ends take the identical ``max_edge`` and land within a few percent of
    each other and of the control.

    The hole radius scales with the spacing, so the two ends differ in genus and in nothing else a
    homology basis can see: same footprint, same tessellation scale, same wall-to-slab proportion.
    """
    slab = tm.creation.box(extents=[span, span, 1.0])
    vertices, faces = tm.remesh.subdivide_to_size(slab.vertices, slab.faces, max_edge=max_edge)[:2]
    step = span / holes_per_side
    tunnels = []
    for i in range(holes_per_side):
        for j in range(holes_per_side):
            tunnel = tm.creation.cylinder(radius=0.3 * step, height=3.0, sections=16)
            tunnel.apply_translation(
                [-0.5 * span + step * (i + 0.5), -0.5 * span + step * (j + 0.5), 0.0]
            )
            tunnels.append(tunnel)
    mesh = tm.boolean.difference([tm.Trimesh(vertices, faces, process=False), *tunnels])
    return mesh.vertices, mesh.faces


def _hemisphere(subdivisions: int) -> _Arrays:
    """Curved disk patch with a single rim, cut from an icosphere by an uncapped plane slice."""
    mesh = tm.creation.icosphere(subdivisions=subdivisions).slice_plane(
        plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]), cap=False
    )
    mesh.merge_vertices()
    return mesh.vertices, mesh.faces


# ---------------------------------------------------------------------------
# feature registry
# ---------------------------------------------------------------------------

FEATURE_MESHES: list[MeshSpec] = [
    _mesh("sphere_small", "", 2_562, 5_120, "scale"),
    _mesh("sphere_med", "", 40_962, 81_920, "control"),
    _mesh("sphere_large", "", 163_842, 327_680, "scale"),
    _mesh("parts_64", "", 41_088, 81_920, "components"),
    _mesh("parts_1024", "", 43_008, 81_920, "components"),
    _mesh("open_parts_4", "", 40_968, 81_916, "open_components"),
    _mesh("open_parts_16", "", 40_992, 81_904, "open_components"),
    _mesh("open_parts_64", "", 41_088, 81_856, "open_components"),
    _mesh("ribbon_long", "", 40_962, 40_960, "diameter"),
    _mesh("fan_hub", "", 40_962, 81_920, "valence"),
    _mesh("rim_long", "", 131_072, 131_072, "loops"),
    _mesh("holes_many", "", 40_962, 81_408, "loops"),
    _mesh("holes_dense", "", 40_962, 73_728, "loops_dense"),
    _mesh("rim_short", "", 1_024, 1_024, "loops_dp"),
    _mesh("saddle_small", "", 4_624, 8_978, "patch"),
    _mesh("saddle", "", 17_689, 34_848, "patch"),
    _mesh("saddle_graded", "", 17_689, 34_848, "quality"),
    _mesh("hemisphere", "", 20_737, 41_088, "patch"),
    _mesh("shells_8", "", 20_496, 40_960, "depth"),
    _mesh("tangle_2", "", 20_484, 40_960, "overlap"),
    _mesh("tangle_torus_small", "", 4_096, 8_192, "tangle"),
    _mesh("tangle_torus", "", 81_920, 163_840, "tangle"),
    _mesh("handles_1", "", 44_580, 89_160, "genus"),
    _mesh("handles_64", "", 46_862, 93_976, "genus"),
]
FEATURE_MESHES_BY_NAME = {mesh["name"]: mesh for mesh in FEATURE_MESHES}
ALL_MESHES_BY_NAME = {**MESHES_BY_NAME, **FEATURE_MESHES_BY_NAME}

BUILDERS: dict[str, Callable[[], _Arrays]] = {
    "sphere_small": lambda: _spheres(4),
    "sphere_med": lambda: _spheres(6),
    "sphere_large": lambda: _spheres(7),
    "parts_64": lambda: _spheres(3, 64, "lattice"),
    "parts_1024": lambda: _spheres(1, 1024, "lattice"),
    "open_parts_4": lambda: _open_spheres(5, 4),
    "open_parts_16": lambda: _open_spheres(4, 16),
    "open_parts_64": lambda: _open_spheres(3, 64),
    "ribbon_long": lambda: _ribbon(RIBBON_LENGTH),
    "fan_hub": lambda: _cone_fan(40_960),
    "rim_long": lambda: _open_cylinder(RIM_LONG),
    "holes_many": lambda: _punched_sphere(6, N_PUNCHED_HOLES),
    "holes_dense": lambda: _punched_sphere(6, N_DENSE_HOLES),
    "rim_short": lambda: _open_cylinder(RIM_SHORT),
    "saddle_small": lambda: _saddle(SADDLE_SMALL),
    "saddle": lambda: _saddle(SADDLE_MEDIUM),
    "saddle_graded": lambda: _saddle(SADDLE_MEDIUM, graded=True),
    "hemisphere": lambda: _hemisphere(6),
    "shells_8": lambda: _spheres(4, 8, "concentric"),
    "tangle_2": lambda: _spheres(5, 2, "overlap"),
    "tangle_torus_small": lambda: _tangled_torus(64, 64),
    "tangle_torus": lambda: _tangled_torus(320, 256),
    "handles_1": lambda: _handles(1),
    "handles_64": lambda: _handles(8),
}


# ---------------------------------------------------------------------------
# axes
# ---------------------------------------------------------------------------

# Each axis is one comparison: the control first, then meshes that perturb a single property.
# What is held fixed is as much the point as what varies, so it is spelled out per entry.
AXES: dict[str, tuple[str, ...]] = {
    # Clean manifold N sweep. The igl-safe counterpart of the scan registry: libigl segfaults or
    # fails to factor on the scan meshes, which have non-manifold vertices.
    "scale": ("sphere_small", "sphere_med", "sphere_large"),
    # Components 1 -> 64 -> 1024 at F = 81 920 throughout.
    "components": ("sphere_med", "parts_64", "parts_1024"),
    # *Open* components 4 -> 16 -> 64, each shell holding one 3-edge rim, at F = 81 920 minus one
    # face per shell. The closed ``components`` axis cannot serve the joining functions -- a shell
    # with no boundary has nothing to bridge, so they return it untouched -- and the variable here
    # is the number of *joins* a driver has to make, which is one fewer than the component count.
    "open_components": ("open_parts_4", "open_parts_16", "open_parts_64"),
    # Graph diameter ~130 -> 20 481 at V = 40 962, same triangle quality.
    "diameter": ("sphere_med", "ribbon_long"),
    # Max vertex valence 6 -> 40 960 at both V = 40 962 and F = 81 920.
    "valence": ("sphere_med", "fan_hub"),
    # No boundary -> 2 loops of 65 536 -> 512 loops of 3: loop length against loop count.
    "loops": ("sphere_med", "rim_long", "holes_many"),
    # The same contrast at a scale the O(B^3) hole-filling DP can actually run.
    "loops_dp": ("rim_short", "holes_many"),
    # Loop count 512 -> 8 192 on the *same* 40 962-vertex sphere: the pure per-loop-cost axis,
    # for work that scales with how many rims there are rather than with their length or the mesh.
    "loops_dense": ("holes_many", "holes_dense"),
    # Worst aspect ratio 1.6 -> 4 719 at identical V, F and connectivity.
    "quality": ("saddle", "saddle_graded"),
    # Disk-topology solver inputs: flat and curved, small to medium.
    "patch": ("saddle_small", "saddle", "hemisphere"),
    # Longest boundary loop 268 -> 528 -> 65 536, for the functions whose input *is* a loop.
    "polyline": ("saddle_small", "saddle", "rim_long"),
    # Surface crossings along a ray, 2 -> 16.
    "depth": ("sphere_med", "shells_8"),
    # Disjoint -> deeply interpenetrating, for collision density.
    "overlap": ("sphere_med", "tangle_2"),
    # Self-intersecting *single* component, 8 192 -> 163 840 faces. A size axis rather than a
    # feature contrast, because the repair comparison it serves is a crossover: a serial C++ fixer
    # leads at the small end and loses the lead as the mesh grows, so a one-size row reports
    # whichever side of it the mesh landed on.
    "tangle": ("tangle_torus_small", "tangle_torus"),
    # Genus 0 -> 1 -> 64 at ~90 000 faces: the number of handles, which is the only thing a
    # homology basis is looking for and the only property no other mesh here varies.
    "genus": ("sphere_med", "handles_1", "handles_64"),
}
