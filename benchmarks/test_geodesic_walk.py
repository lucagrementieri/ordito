"""
Benchmarks for ``ordito.geodesic_walk``: batched straightest-geodesic geodesic_walk.

Two axes, and the parameter is the interesting one:

* **ray count** -- one thread traces one ray, so this is the axis the design is *for*. potpourri3d's
  ``GeodesicTracer`` traces one ray per call, which makes its row linear in the count by
  construction; the point of the comparison is where the crossover sits, not who wins at one ray.
* **diameter** -- ``sphere_med`` against ``ribbon_long`` at pinned ``V``. Each ray walks a fixed
  arc length, so meshes with similar triangle sizes give the same per-ray step count; what changes
  is locality, and this axis says whether that matters.

Both entry points do the same walk. ``trace_from_face`` skips the wedge search that
``trace_from_vertex`` needs to pick a starting face, so timing them apart separates the
walk from the setup.

References
----------
**potpourri3d** is the only reference with an equivalent, and it covers **all three** walk groups:
``GeodesicTracer`` binds ``trace_geodesic_from_vertex`` *and* ``trace_geodesic_from_face``, the two
entry points this module splits, so every axis here has a second implementation rather than just the
ray-count one. Two things to keep in mind when reading its rows: it traces a single ray per call
(the API takes one start point), and its construction -- building geometry-central's halfedge mesh
-- is inside the timed callable, matching this suite's convention for reference setup. ordito's row
likewise includes its own ``halfedge_twins`` and one-ring prologue.

That construction is the *dominant* term once the mesh is large -- an order of magnitude or more
over the tracing itself -- so on the ``scale`` axis the reference row is essentially a build
benchmark, and the per-ray comparison lives in the ``trace_rays`` ray-count sweep where the build is
amortized across up to 4 096 rays.

**trimesh**, **libigl**, **open3d** and **scipy** have nothing comparable: tracing a straightest
geodesic needs an unfolding walk across edges, and none of them exposes one. ``igl`` does join
the ``geodesic_path`` group through ``exact_geodesic``, whose *distance* bounds any path length.

``shorten_loop`` runs on a third axis, **genus**, against potpourri3d's
``EdgeFlipGeodesicSolver.find_geodesic_loop``. It is the only axis in either registry with a
genus, and it is shared with ``benchmarks/test_homology.py``, which produces this group's input.
"""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import warp as wp

import ordito as od
from conftest import BenchCase, skip_larger_than

# The reference traces one ray per Python call, so the largest ray count is slow on its side.
_GENERATORS = {"sphere_med": 0, "handles_1": 2, "handles_64": 128}
_ROUNDS = 3
_RAY_COUNTS = [1, 64, 4096]

_rays_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def _rays(bench_case: BenchCase, n_rays: int) -> tuple[np.ndarray, np.ndarray]:
    """Random start vertices and directions of a few edge lengths, cached per (mesh, count)."""
    key = (bench_case.mesh_name, n_rays)
    if key not in _rays_cache:
        rng = np.random.default_rng(0)
        start = rng.integers(0, bench_case.n_vertices, n_rays).astype(np.int32)
        directions = rng.normal(size=(n_rays, 3))
        directions *= (5.0 * bench_case.mean_edge) / np.linalg.norm(
            directions, axis=1, keepdims=True
        )
        _rays_cache[key] = (start, directions)
    return _rays_cache[key]


def _run_case(bench_case: BenchCase, n_rays: int) -> None:
    """Trace ``n_rays`` geodesics, in ordito or potpourri3d."""
    start_np, directions_np = _rays(bench_case, n_rays)
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        start = wp.array(start_np, dtype=wp.int32, device=bench_case.device)
        directions = wp.array(
            directions_np.astype(np.float32), dtype=wp.vec3, device=bench_case.device
        )
        _, offsets = bench_case.run(
            lambda: od.geodesic_walk.trace_from_vertex(vertices, faces, start, directions),
            rounds=_ROUNDS,
        )
        assert offsets.shape == (n_rays + 1,)
    else:
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)

        def trace_pp() -> int:
            tracer = pp3d.GeodesicTracer(vertices_np, faces_np)
            return sum(
                len(tracer.trace_geodesic_from_vertex(int(vertex), direction))
                for vertex, direction in zip(start_np, directions_np, strict=True)
            )

        assert bench_case.run(trace_pp, rounds=_ROUNDS) > 0


@pytest.mark.benchmark(group="trace_rays")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito", "potpourri3d")
@pytest.mark.parametrize("n_rays", _RAY_COUNTS)
def test_trace_ray_count(bench_case: BenchCase, n_rays: int) -> None:
    """One mesh, 1 to 4 096 rays: the axis batching exists for."""
    _run_case(bench_case, n_rays)


@pytest.mark.benchmark(group="trace_locality")
@pytest.mark.benchaxis("diameter")
@pytest.mark.benchlibs("ordito", "potpourri3d")
def test_trace_locality(bench_case: BenchCase) -> None:
    """
    The same 1 024 rays on meshes of equal size but very different shape.

    Identical call to ``trace_rays``, second axis -- so it takes the same potpourri3d row, through
    the same ``_run_case``. Both fixtures on this axis are manifold with every vertex referenced,
    which is what geometry-central requires; probed before the row landed.

    The reference is per-ray and its halfedge build is inside the timed callable, so its row barely
    moves across this axis while ordito's is the one carrying the locality signal. That asymmetry
    is the row's content: it says the diameter axis is a *GPU* locality question, not a property of
    the algorithm.
    """
    _run_case(bench_case, 1024)


@pytest.mark.benchmark(group="trace_from_face")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "potpourri3d")
def test_trace_from_face(bench_case: BenchCase) -> None:
    """
    The walk without the wedge search, from face-interior start points.

    ``GeodesicTracer.trace_geodesic_from_face`` is the matching entry point to the
    ``trace_geodesic_from_vertex`` the ``trace_rays`` group already times -- same tracer object,
    same barycentric start convention -- so this group takes the same reference rather than none.

    **On this axis the reference row is its own construction**: the halfedge build dwarfs the
    tracing, so the potpourri3d row is almost entirely build at the top of the scale sweep. That is
    the suite's convention for reference setup (see the module docstring) and it is the honest
    number for a caller who has no tracer in hand -- but read it as "what geometry-central charges
    to be ready", and read the ``trace_rays`` ray-count sweep for the per-ray cost.
    """
    n_rays = 1024
    rng = np.random.default_rng(1)
    faces_np = rng.integers(0, bench_case.n_faces, n_rays).astype(np.int32)
    directions_np = rng.normal(size=(n_rays, 3))
    directions_np *= (5.0 * bench_case.mean_edge) / np.linalg.norm(
        directions_np, axis=1, keepdims=True
    )

    if bench_case.kind == "potpourri3d":
        vertices_np = bench_case.vertices_np
        faces_pp = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)
        barycentric_np = np.full(3, 1.0 / 3.0)

        def trace_from_face_pp() -> int:
            tracer = pp3d.GeodesicTracer(vertices_np, faces_pp)
            return sum(
                len(tracer.trace_geodesic_from_face(int(face), barycentric_np, direction))
                for face, direction in zip(faces_np, directions_np, strict=True)
            )

        assert bench_case.run(trace_from_face_pp, rounds=_ROUNDS) > 0
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    start_faces = wp.array(faces_np, dtype=wp.int32, device=bench_case.device)
    barycentric = wp.array(
        np.full((n_rays, 3), 1.0 / 3.0, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
    )
    directions = wp.array(directions_np.astype(np.float32), dtype=wp.vec3, device=bench_case.device)
    _, offsets = bench_case.run(
        lambda: od.geodesic_walk.trace_from_face(
            vertices, faces, start_faces, barycentric, directions
        ),
        rounds=_ROUNDS,
    )
    assert offsets.shape == (n_rays + 1,)


_PATH_COUNTS = [1, 64, 4096]


@pytest.mark.benchmark(group="geodesic_path")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito", "potpourri3d", "igl")
@pytest.mark.parametrize("n_paths", _PATH_COUNTS)
def test_geodesic_path(bench_case: BenchCase, n_paths: int) -> None:
    """
    Point-to-point geodesics: one heat solve, then one descent per target.

    The **path count** is the axis, and it is the axis this design exists for in a stronger sense
    than ``trace_rays``: the heat solve is shared by every target, so its cost is amortized away
    while the walks are independent. A single path pays the whole factorization and 4 096 pay it
    once, which is what the three points are there to show -- and it is also why the references,
    both of which do per-pair work, are expected to cross over.

    ``potpourri3d``'s ``EdgeFlipGeodesicSolver`` gives the *exact* geodesic (edge flips to a locally
    shortest path) and ``igl.exact_geodesic`` the exact geodesic **distance** by MMP window
    propagation, so neither is doing the same amount of work as an approximate descent. They are
    here for scale and because they bound the answer: ``tests/test_geodesic_walk.py`` asserts
    ordito's path is never shorter than igl's exact distance and measures how much longer it is.
    Both take one query per call, so their rows include a Python loop -- read them as the cost of
    *that* API shape.

    The amortization is the whole row: **4 096 paths cost about twice what one costs**, because the
    heat solve is shared and only the walks scale. Against that, potpourri3d is linear in the path
    count and igl is flat but far off, since MMP propagates windows over the whole surface whatever
    is asked of it.

    ordito's cost at *one* path is almost entirely the factorization, not the walk -- which is the
    honest caveat on the left column and the reason the wrapper takes a ``Trimesh``, which keeps
    it for the next source.
    """
    device = bench_case.device
    rng = np.random.default_rng(4)
    targets_np = rng.integers(1, bench_case.n_vertices, n_paths).astype(np.int32)

    if bench_case.kind == "potpourri3d":
        skip_larger_than(bench_case, "sphere_med", "one exact path per call")
        if n_paths > 64:
            pytest.skip("EdgeFlipGeodesicSolver answers one pair per call: capped at 64 paths")
        vertices_np = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64)
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)

        def paths_pp() -> int:
            solver = pp3d.EdgeFlipGeodesicSolver(vertices_np, faces_np)
            return sum(
                len(solver.find_geodesic_path(v_start=0, v_end=int(target)))
                for target in targets_np
            )

        assert bench_case.run(paths_pp, rounds=_ROUNDS) > 0
        return

    if bench_case.kind == "igl":
        if n_paths > 64:
            pytest.skip(
                "MMP window propagation: 839 ms at 10k vertices, and it grows 15-25x per 4x"
            )
        vertices_np = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64)
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        empty_np = np.array([], dtype=np.int64)
        source_np = np.array([0], dtype=np.int64)
        target_np = targets_np.astype(np.int64)
        distance_igl = bench_case.run(
            lambda: igl.exact_geodesic(
                vertices_np, faces_np, source_np, empty_np, target_np, empty_np
            ),
            rounds=_ROUNDS,
        )
        assert distance_igl.shape[0] == n_paths
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    source = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    targets = wp.array(targets_np, dtype=wp.int32, device=device)
    _points, offsets = bench_case.run(
        lambda: od.geodesic_walk.geodesic_path(vertices, faces, source, targets), rounds=_ROUNDS
    )
    assert offsets.shape == (n_paths + 1,)


@pytest.mark.benchmark(group="shorten_loop")
@pytest.mark.benchaxis("genus")
@pytest.mark.benchlibs("ordito", "potpourri3d")
def test_shorten_loop(bench_case: BenchCase) -> None:
    """
    Shortening a homology basis, on the one axis in either registry that has a genus.

    The loops come from [`test_homology_generators`]'s own call and are built **outside** the timed
    callable on both sides, so this group times the shortening and not the basis, which would
    otherwise swamp it (see ``benchmarks/test_homology.py``).

    potpourri3d's ``find_geodesic_loop`` is the reference and answers **one loop per call**, so its
    row is linear in the genus by construction; the solver build is inside its callable, matching
    this suite's convention for reference setup, and ordito's row likewise carries its own
    ``halfedge_twins`` and one-ring prologue. The two do not compute the same curve -- this one
    stays on mesh edges, that one flips its way into face interiors -- so read the row against the
    length gap ``tests/test_geodesic_walk.py`` pins, not as a like-for-like.

    Both columns are dominated by their fixed cost at genus 1 and by the loops at genus 64, so the
    *marginal* per-loop cost is the number worth keeping; ordito is well ahead on it. This is the
    first row in this family where ordito is ahead -- ``homology_generators``, which produces the
    input, is behind meshlib -- so the two groups together say the basis is the module's bottleneck
    and shortening it is not.
    """
    if _GENERATORS[bench_case.mesh_name] == 0:
        pytest.skip(f"{bench_case.mesh_name} is genus 0: there is no loop to shorten")

    if bench_case.kind == "potpourri3d":
        vertices_np = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64)
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)
        # The basis is ordito's either way, and a reference case has no device -- so build it on
        # the host, outside the timed callable, and hand both sides the same loops.
        loops_np = [
            loop.numpy().astype(np.int64)
            for loop in od.homology.homology_generators(
                wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device="cpu"),
                wp.array(faces_np.ravel().astype(np.int32), dtype=wp.int32, device="cpu"),
            )
        ]

        def shorten_pp() -> int:
            solver = pp3d.EdgeFlipGeodesicSolver(vertices_np, faces_np)
            return sum(len(solver.find_geodesic_loop(loop_np)) for loop_np in loops_np)

        assert bench_case.run(shorten_pp, rounds=_ROUNDS) > 0
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    loops = od.homology.homology_generators(vertices, faces)
    shortened, sweeps = bench_case.run(
        lambda: od.geodesic_walk.shorten_loop(vertices, faces, loops), rounds=_ROUNDS
    )
    assert len(shortened) == len(loops)
    assert sweeps > 0


@pytest.mark.benchmark(group="shorten_loop_with_offsets")
@pytest.mark.benchaxis("genus")
@pytest.mark.benchlibs("ordito")
def test_shorten_loop_with_offsets(bench_case: BenchCase) -> None:
    """
    ``shorten_loop``'s row on the packed basis: no pack of the input, no split of the output.

    Read against that group's ordito row, from the same basis built outside the timed callable.
    """
    if _GENERATORS[bench_case.mesh_name] == 0:
        pytest.skip(f"{bench_case.mesh_name} is genus 0: there is no loop to shorten")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    loops, offsets = od.homology.homology_generators_with_offsets(vertices, faces)
    _shortened, shortened_offsets, sweeps = bench_case.run(
        lambda: od.geodesic_walk.shorten_loop_with_offsets(vertices, faces, loops, offsets),
        rounds=_ROUNDS,
    )
    assert shortened_offsets.size == offsets.size
    assert sweeps > 0
