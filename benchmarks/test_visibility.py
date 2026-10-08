"""
Benchmarks for ``ordito.visibility``: how far the surface is from a point.

Five functions, three shapes of work, and the split is what the rows are for.

- **Ray bundles.** ``ambient_occlusion`` traces a hemisphere lattice outward per point and
  ``shape_diameter`` fires the same lattice inward and then trims the distances. Both are
  embarrassingly parallel and both are among the slowest per-vertex filters MeshLab ships, so they
  are the module's clearest GPU-versus-one-core rows. ``rays`` is the axis rather than the mesh:
  cost is exactly linear in it on both sides, and the sweep pins that. Read the two groups against
  each other at equal ray counts -- the gap is what ``shape_diameter``'s two extra passes over its
  ``(n_vertices, n_rays)`` distance scratch cost, and it should be small, because the rays dominate.
- **Iterated closest-point.** ``max_tangent_sphere`` shrinks a sphere until nothing but the surface
  touches it, so it pays a full BVH closest-point pass *per iteration* (up to 100) plus an 8-byte
  convergence readback each time -- deliberately, since an extra iteration costs far more than the
  readback. The ``_reach`` row runs it outward, which is the branch where the first ray escapes and
  the packed support-argmax passes over the vertex cloud run instead.
- **One ray, or one sphere.** ``thickness`` is a dispatcher, and its two methods are two orders of
  magnitude apart by construction. It sits on the **depth** axis because both pay for surface
  crossings: a ray through ``shells_8`` meets 16 of them against ``sphere_med``'s 2, and the sphere
  method's convergence depends on local thickness, which nested shells make small.

- **Every vertex, one ray or one sphere.** ``thickness_at_vertices`` is the same dispatcher as
  ``thickness`` run over the mesh's *own* vertices rather than a 10 000-point subsample, and it
  exists for one reason: it is the only shape MeshLib can be timed in. Its
  ``computeRayThicknessAtVertices`` takes no query set -- it answers at every vertex, in parallel
  over all cores -- so a row against the subsampled group would price a different number of
  queries, the reason to bar ``findNClosestPointsPerPoint`` from
  ``query_nearest_bvh_k7``. Read it against ``thickness_interior`` for the per-query cost and
  against MeshLib for the one fair CPU-versus-GPU comparison this module has.

``volumetric_obscurance`` has **no group**. It shares ``ambient_occlusion``'s bundle and block
layout and differs only in tracing to the nearest hit (``ambient_occlusion`` stops at any hit, 1.06-
1.14x faster) and a per-hit ``exp(-tau * t)`` factor, so a group over it would re-measure the axis;
MeshLab's
``compute_scalar_by_volumetric_obscurance`` would nonetheless be a real second reference, and that
is a benchmark gap rather than an API one. Recorded in ``benchmarks/README.md``.

Two caveats for reading the pymeshlab rows. ``compute_scalar_by_shape_diameter_function_per_vertex``
is the most expensive per-vertex filter MeshLab ships and its ``cone_amplitude`` parameter is a
**no-op** in the 2025.07 build -- byte-identical output at 90 and
120 degrees -- so its cone is whatever it is. Both filters write only the vertex scalar attribute,
so the MeshSet is shared rather than rebuilt per call. Open3D has no equivalent for anything here.

**trimesh is the oracle for two of these five functions and is now timed for both.**
``trimesh.proximity.thickness`` and ``max_tangent_sphere`` have been the correctness references in
``tests/test_visibility.py`` since those functions landed, which this docstring did not say -- it
recorded only that Open3D has nothing, leaving ``thickness_interior`` reading as unreferenced and
``max_tangent_sphere_reach``'s exemption reading as "no library has an exterior form". Both take a
**query set** and their own normals, so unlike MeshLib they fit the subsampled groups directly.
They are single-threaded Python over embree and the ``max_sphere`` branch iterates in Python, so
their rows run at ``_N_QUERIES_TM`` queries rather than 10 000 and must be read per query.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import trimesh as tm
import trimesh.proximity as tm_proximity
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
from conftest import BenchCase, skip_larger_than

_QUERY_SEED = 42
_N_QUERIES = 10_000

_surface_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}
_mesh_cache: dict[tuple[str, str], wp.Mesh] = {}
_normals_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}


def _mesh_wp(bench_case: BenchCase) -> wp.Mesh:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _mesh_cache:
        _mesh_cache[key] = wp.Mesh(points=bench_case.vertices_wp, indices=bench_case.faces_wp)
    return _mesh_cache[key]


def _surface_points_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """10k on-surface points (subsampled mesh vertices, no jitter) for tangent-sphere queries."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _surface_cache:
        rng = np.random.default_rng(_QUERY_SEED)
        vertices = bench_case.vertices_np
        idx = rng.choice(vertices.shape[0], size=min(_N_QUERIES, vertices.shape[0]), replace=False)
        _surface_cache[key] = wp.array(
            np.ascontiguousarray(vertices[idx], dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _surface_cache[key]


# Rays per point for the bundle groups. MeshLab's default is 64; 256 shows the cost is exactly
# linear in it on both sides, which is the whole shape of these two groups.
_N_RAYS_SWEEP = [64, 256]

# Queries for the trimesh rows. ``trimesh.proximity`` takes a query set and its own normals, and
# both are single-threaded Python-plus-embree, so the subsample is cut hard: the rows below run at
# ``_N_QUERIES_TM`` where ordito runs at 10 000, and the per-query cost is what to compare, not
# the row totals. 256 keeps every trimesh row inside a couple of seconds on ``bunny``.
_N_QUERIES_TM = 256

_surface_np_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _surface_points_np(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """
    ``(points, normals)`` for the trimesh rows: a vertex subsample with angle-weighted normals.

    Vertices rather than ``sample_surface`` points so the query set is the same *kind* of input
    ordito's rows use, and angle-weighted normals because that is the convention
    recorded as the one the ray methods pair on.
    Cached per mesh, since building it is not what is timed.
    """
    if bench_case.mesh_name not in _surface_np_cache:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
        rng = np.random.default_rng(_QUERY_SEED)
        count = min(_N_QUERIES_TM, vertices_np.shape[0])
        chosen = rng.choice(vertices_np.shape[0], size=count, replace=False)
        _surface_np_cache[bench_case.mesh_name] = (
            np.ascontiguousarray(vertices_np[chosen]),
            np.ascontiguousarray(mesh_tm.vertex_normals[chosen]),
        )
    return _surface_np_cache[bench_case.mesh_name]


def _vertex_normals_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """Smooth outward normals over the mesh's own vertices -- an *input* of the bundle queries."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _normals_cache:
        _normals_cache[key] = od.vertices.vertex_normals(
            bench_case.vertices_wp, bench_case.faces_wp
        )
    return _normals_cache[key]


@pytest.mark.benchmark(group="ambient_occlusion")
@pytest.mark.benchlibs("ordito", "pymeshlab")
@pytest.mark.parametrize("n_rays", _N_RAYS_SWEEP)
def test_ambient_occlusion(bench_case: BenchCase, n_rays: int) -> None:
    """A hemisphere ray bundle per vertex: the embarrassingly parallel case, against one core."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":
        skip_larger_than(bench_case, "bunny", "MeshLab traces the whole bundle on one core")
        meshset_pml = bench_case.meshset_pml  # writes only the vertex scalar
        bench_case.run(lambda: meshset_pml.compute_scalar_ambient_occlusion(rays=n_rays), rounds=3)
        assert meshset_pml.current_mesh().vertex_scalar_array().shape == (n_vertices,)
        return
    mesh, points = _mesh_wp(bench_case), bench_case.vertices_wp
    normals = _vertex_normals_wp(bench_case)
    occlusion = bench_case.run(
        lambda: od.visibility.ambient_occlusion(mesh, points, normals=normals, n_rays=n_rays)
    )
    assert occlusion.shape == (n_vertices,)


@pytest.mark.benchmark(group="shape_diameter")
@pytest.mark.benchlibs("ordito", "pymeshlab")
@pytest.mark.parametrize("n_rays", _N_RAYS_SWEEP)
def test_shape_diameter(bench_case: BenchCase, n_rays: int) -> None:
    """
    The same bundle fired inward, plus the trimming passes over a ``(n_vertices, n_rays)`` scratch.

    Read against ``ambient_occlusion``: identical ray count, and the gap between the two rows is
    what the two extra passes over the distance scratch cost. It should be small -- the rays
    dominate -- and if it ever is not, the scratch is the thing to attack.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":
        skip_larger_than(bench_case, "bunny", "674 ms a call on bunny at the default ray count")
        meshset_pml = bench_case.meshset_pml
        bench_case.run(
            lambda: meshset_pml.compute_scalar_by_shape_diameter_function_per_vertex(rays=n_rays),
            rounds=3,
        )
        assert meshset_pml.current_mesh().vertex_scalar_array().shape == (n_vertices,)
        return
    mesh, points = _mesh_wp(bench_case), bench_case.vertices_wp
    normals = _vertex_normals_wp(bench_case)
    diameter = bench_case.run(
        lambda: od.visibility.shape_diameter(mesh, points, normals=normals, n_rays=n_rays)
    )
    assert diameter.shape == (n_vertices,)


@pytest.mark.benchmark(group="thickness_interior")
@pytest.mark.benchaxis("depth")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("method", ["ray", "max_sphere"])
def test_thickness_interior(bench_case: BenchCase, method: Literal["ray", "max_sphere"]) -> None:
    """
    Interior thickness by one ray cast against up to 100 shrinking-sphere iterations.

    On the **depth** axis because both methods pay for surface crossings: a ray through
    ``shells_8`` meets 16 of them against ``sphere_med``'s 2, and the sphere method's convergence
    depends on local thickness, which nested shells make small. The two rows are roughly two
    orders of magnitude apart by construction -- the point is whether that ratio holds when the
    geometry stops being convex.

    ``trimesh.proximity.thickness`` is the reference, and it is the one that fits *this* group
    rather than ``thickness_at_vertices``: it takes a **query set** and a ``method=`` switch with
    the same two values, which is exactly what MeshLib's whole-vertex-buffer form cannot do and why
    that second group had to exist. ``tests/test_visibility.py`` carries the Class-A comparison for
    both branches; trimesh is the oracle for two of the five functions here.

    **Read the per-query cost, not the row.** trimesh runs at ``_N_QUERIES_TM`` queries against
    ordito's 10 000, because its ``max_sphere`` branch is a Python loop over closest-point queries;
    dividing each row by its own query count is the only fair reading.
    """
    if bench_case.kind == "trimesh":
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        points_np, normals_np = _surface_points_np(bench_case)
        thickness_tm = bench_case.run(
            lambda: tm_proximity.thickness(mesh_tm, points_np, normals=normals_np, method=method),
            rounds=3,
        )
        assert thickness_tm.shape == (points_np.shape[0],)
        return
    mesh = _mesh_wp(bench_case)
    points = _surface_points_wp(bench_case)
    result = bench_case.run(lambda: od.visibility.thickness(mesh, points, method=method))
    assert result.shape == points.shape


_mesh_ml_cache: dict[str, mm.Mesh] = {}


def _mesh_ml(bench_case: BenchCase) -> mm.Mesh:
    """
    One ``meshlib.Mesh`` per mesh, pre-warmed: ``computeRayThicknessAtVertices`` does not mutate it.

    Built and warmed *outside* the timed callable, as ``new_mesh_ml``'s docstring requires -- the
    AABB tree is lazily built on first query and cached on the mesh, so a row that builds per round
    would time the tree rather than the rays.
    """
    if bench_case.mesh_name not in _mesh_ml_cache:
        mesh_ml = bench_case.new_mesh_ml()
        mm.computeRayThicknessAtVertices(mesh_ml)  # pre-warm the AABB tree
        _mesh_ml_cache[bench_case.mesh_name] = mesh_ml
    return _mesh_ml_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="thickness_at_vertices")
@pytest.mark.benchlibs("ordito", "meshlib", "trimesh")
def test_thickness_at_vertices(bench_case: BenchCase) -> None:
    """
    Interior thickness at **every** vertex by one inward ray: the module's fair MeshLib row.

    The same dispatcher as ``thickness_interior``'s ``method="ray"`` row, over the whole vertex
    buffer instead of a 10 000-point subsample, because MeshLib's ``computeRayThicknessAtVertices``
    takes no query set and would otherwise be timed on a different amount of work. Both sides cast
    one ray per vertex along minus the vertex normal and return the distance to the first surface
    they meet; ``tests/test_visibility.py::test_thickness_at_vertices_matches_meshlib`` pins that
    they agree (5.96e-07) and that the normal convention is the angle-weighted one.

    MeshLib is the only multi-threaded CPU reference in the suite, so this is a fair
    fight rather than a GPU against one core -- and the ``ordito-cpu`` row will lose to it for that
    reason regardless of algorithm, which is the "decide on the CUDA number".

    **trimesh is the third row, and unlike the other two it is timed on both groups.** It takes a
    query set, so it can be asked at every vertex here *and* at the subsample in
    ``thickness_interior`` -- which makes the pair of groups readable as one query-count axis with
    the same reference on both ends, the thing MeshLib's no-query-set form cannot provide. Its
    normals are its own ``vertex_normals``, the angle-weighted convention both other rows use. It is
    single-threaded over embree and cleanly linear in the vertex count, so it is capped at
    ``bunny``.
    """
    if bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "bunny", "trimesh casts one ray per vertex on one core")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        vertices_np = np.ascontiguousarray(mesh_tm.vertices)
        normals_np = np.ascontiguousarray(mesh_tm.vertex_normals)
        thickness_tm = bench_case.run(
            lambda: tm_proximity.thickness(mesh_tm, vertices_np, normals=normals_np, method="ray"),
            rounds=3,
        )
        assert thickness_tm.shape == (bench_case.n_vertices,)
        return
    if bench_case.kind == "meshlib":
        mesh_ml = _mesh_ml(bench_case)
        thickness_ml = bench_case.run(lambda: mm.computeRayThicknessAtVertices(mesh_ml))
        assert thickness_ml is not None
        return
    mesh, points = _mesh_wp(bench_case), bench_case.vertices_wp
    normals = od.vertices.vertex_normals(
        bench_case.vertices_wp, bench_case.faces_wp, weighting="angle"
    )
    thickness = bench_case.run(
        lambda: od.visibility.thickness(mesh, points, method="ray", normals=normals)
    )
    assert thickness.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="max_tangent_sphere_reach")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_max_tangent_sphere_reach(bench_case: BenchCase) -> None:
    """
    Exterior tangent spheres: exercises the ``init_sphere_radii`` inf-distance branch.

    ``trimesh.proximity.max_tangent_sphere(inwards=False)`` is the exterior branch, and it is the
    only one in any installed library -- MeshLib's ``insideAndOutside`` returns the *smaller* of the
    two spheres with a sign rather than the outer one, which is why that reference is declared
    untimed here rather than timed. So this group's ``noparity`` reason should not be read as "no
    library has an exterior form"; trimesh does.

    The comparison needs a **non-convex** input to say anything: the exterior tangent sphere of a
    convex body is unbounded, so on a sphere both libraries correctly return ``inf`` everywhere
    (measured 0 of 24 finite). The scan meshes are non-convex, and
    ``tests/test_visibility.py::test_max_tangent_sphere_reach_matches_trimesh`` draws the assertion
    on ``cave_cube`` for the same reason.

    Per-query, like the ``thickness_interior`` trimesh row: trimesh runs at ``_N_QUERIES_TM``
    queries against ordito's 10 000, since its iteration is a Python loop.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "trimesh":
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        points_np, normals_np = _surface_points_np(bench_case)
        _centers_tm, radii_tm = bench_case.run(
            lambda: tm_proximity.max_tangent_sphere(
                mesh_tm, points_np, normals=normals_np, inwards=False
            ),
            rounds=3,
        )
        assert radii_tm.shape == (points_np.shape[0],)
        return
    mesh = _mesh_wp(bench_case)
    points = _surface_points_wp(bench_case)
    _, radii = bench_case.run(lambda: od.visibility.max_tangent_sphere(mesh, points, inwards=False))
    assert radii.shape == points.shape
