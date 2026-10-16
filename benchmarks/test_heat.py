"""
Benchmarks for the three heat-diffusion solvers of ``ordito.heat``.

One module because they are one family and share the property this suite really measures: each is
two or three conjugate-gradient solves against a cotangent or connection Laplacian, so each is
sensitive to the *conditioning* of that operator in a way every factorizing reference is not.
**That is the module's whole subject**, and the ``quality`` axis is how it is measured — ``saddle``
against ``saddle_graded`` holds vertices, faces and connectivity fixed and only worsens the aspect
ratio, which no face-count registry can express. Bad aspect ratios and obtuse angles (which send
cotangent weights negative) inflate the condition number and so the iteration count directly, and
the finding recurs at ``heat_geodesic_conditioning``, at ``heat_signed_distance`` and at
``log_map``.

Geodesic distance
=================
Axes: **scale**, the clean size sweep across two orders of magnitude of faces, and **quality**.

The method (Crane et al.) is three stages, two of which are sparse solves and dominate: diffuse heat
from the sources for a short time ``t``, normalize the gradient into a unit field pointing away from
them, then integrate that field back with a Poisson solve. It runs in ``float64`` — the diffused
heat decays exponentially and underflows ``float32``, collapsing the far field — which on a consumer
GPU means the solves run at the device's much lower double-precision rate. Inherent to the method,
not a tuning choice.

``geodesic_ball`` is **not** here: it lives in ``ordito.neighbors`` and is timed as
``query_geodesic_ball`` in [`test_proximity.py`](test_proximity.py).

Four references, three of them the same method a different way. **libigl**'s ``heat_geodesics``
runs on every mesh in both axes, making the quality axis a direct iterative-versus-direct
comparison; on a scan mesh ``heat_geodesics_precompute`` raises ``Precomputation failed`` because it
factors the cotangent and Poisson systems directly. **potpourri3d** (geometry-central) is
constructed with ``use_robust=False`` so all three sides discretize the same triangulation — its
default mollifies and flips to an intrinsic Delaunay triangulation first, which is more work and a
different operator — and it also ships **fast marching**, timed as its own group, which ordito has
no equivalent of by design. **pymeshlab**'s
``compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex`` states the two properties
this module is built around in its own documentation ("very sensitive to triangulation", "first run
takes longer as factorization has to be built"), so it is the only reference whose amortized row
needs no API gymnastics; sources go in as a vertex selection. It is the third independent
confirmation of the quality axis's point — dead flat across the pair, like the other two direct
solvers — and it ships a **second, unrelated** algorithm for the same task, a Dijkstra-style front
over the edge graph, timed in ``fast_marching_distance``. **trimesh** and **open3d** have no
geodesic distance of any kind.

All three libraries split this into mesh-dependent setup (assembly, and for the references a
factorization) and a per-source solve, and the ``heat_geodesic`` group reports both points:
``setup=full`` puts the setup **inside** the timed callable, which is what a caller computing one
field pays — timing a back-substitution against a full iterative solve would compare nothing — and
``setup=amortized`` hoists it out. On ordito's side that is a real API path (a ``Trimesh`` passed to
``heat_geodesic``, which keeps the operators and, from its second call, their factorizations; the
row warms it with two calls before timing), not a benchmark-only shortcut. The other groups
report ``full`` only. A single source vertex throughout: the method's cost is essentially
independent of the number of sources, which change only the right-hand side.

Signed distance to curves
=========================
The axis is the **source curve**, not the mesh: the mesh sets the solve size (fixed at
``sphere_med``) while the curve sets how much of the surface the source touches. ``ring`` is one
vertex's one-ring — the smallest curve that is closed, edge-connected and separating — and ``band``
a full latitude band of them, hundreds of segments. That contrast asks whether the cost is in the
*source*, which scales with the curve, or in the three solves, which do not. **The solves** — the
longer curve is if anything *cheaper*, because a source spread over the surface converges in fewer
iterations than a point-like one, and the splat does not register.

The stages are a vector diffusion on the connection Laplacian (a ``2 x 2``-block CG), a
normalization, and a Poisson solve; ``zero_set`` makes that last one a *constrained* solve through
the machinery behind ``linalg.min_quad_with_fixed``, which is why both level-set modes are timed —
extracting the free-free block costs several times one unconstrained solve.

**potpourri3d** is the only reference (``MeshSignedHeatSolver``), constructed inside the timed
callable per this suite's convention, and it requires every curve segment to lie within one face,
which is why the curves here are edge paths. trimesh, libigl, open3d and scipy have no signed
distance *on a surface* at all — trimesh's ``proximity.signed_distance`` signs against a closed
volume, a different question already benchmarked in [`test_proximity.py`](test_proximity.py).

Vector transport, scalar extension and the log map
=================================================
Everything here is conjugate gradient, so **quality** is again the axis that matters; ``scale`` is
reported alongside for the assembly cost, and ``transport_tangent_vectors`` carries a
``setup=full`` / ``setup=amortized`` layer with the operators hoisted out on both sides.

The three cost different numbers of solves, which is most of what separates them: ``extend_scalar``
two scalar solves, ``transport_tangent_vectors`` one ``2 x 2``-block vector solve plus an
``extend_scalar``, and ``log_map`` a vector solve plus a full ``heat_geodesic`` plus a gradient
pass.

**The comparison against the reference flips sign between the two amortized rows** — ordito far
ahead at ``full`` and behind at ``amortized`` — for the reason the ``heat_geodesic`` rows show: a
factorization is expensive once and cheap thereafter, conjugate gradient is neither. Transport
barely feels the quality axis where ``log_map`` feels it several times over, the difference being
the distance field the log map also solves; **``log_map`` is the module's real result**.

Two implementation notes this level cannot see. ``vector_heat_operators`` caches the vector system's
own Jacobi preconditioner, which is a real saving on the vector solve alone and diluted below noise
once the rest of the call is included — landed for consistency with ``heat_operators``, not for a
win visible here. And ``extend_scalar``'s two right-hand sides diffuse through the identical
operator, so they are one batched two-column solve rather than two independent CG calls, which
``transport_tangent_vectors`` inherits and ``log_map`` does not (it calls ``heat_geodesic``).

Assembly alone is measured as ``connection_laplacian`` in
[`test_laplacian.py`](test_laplacian.py), since the operator lives in ``ordito.laplacian``.

**potpourri3d** is again the only reference, constructed inside the timed callable — which for
``MeshVectorHeatSolver`` means a halfedge mesh and factoring both Laplacians — with
``use_intrinsic_delaunay=False`` so the discretization matches. It cannot run on ``cave_cube`` at
all: its right-angle diagonals give zero cotangent weights and geometry-central's factorization
fails there, so no group uses that mesh. trimesh, libigl and open3d have no tangent-space machinery.
"""

from __future__ import annotations

import itertools

import igl
import numpy as np
import potpourri3d as pp3d
import pymeshlab as ml
import pytest
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
from conftest import BenchCase, skip_larger_than

# --------------------------------------------------------------------------
# heat_operators / heat_geodesic
# --------------------------------------------------------------------------

_SOURCES = np.array([0], dtype=np.int32)

# Two float64 CG solves plus a direct Cholesky on the reference side; both run into hundreds of
# milliseconds at the top of the scale axis.
_GEODESIC_ROUNDS = 3

_sources_cache: dict[str, wp.array[wp.int32]] = {}


def _sources_wp(bench_case: BenchCase) -> wp.array[wp.int32]:
    """Return the single-source index buffer on this case's device."""
    key = str(bench_case.device)
    if key not in _sources_cache:
        _sources_cache[key] = wp.array(_SOURCES, dtype=wp.int32, device=bench_case.device)
    return _sources_cache[key]


def _seeded_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """Build a fresh MeshSet with vertex 0 selected -- the single source the other libraries get."""
    meshset_pml = bench_case.new_meshset_pml()
    meshset_pml.compute_selection_by_condition_per_vertex(condselect="(vi == 0)")
    return meshset_pml


def _run_case_pml(bench_case: BenchCase, *, amortized: bool) -> None:
    """
    Time MeshLab's heat method, with its factorization cache either cold or warm.

    The filter caches its factorization on the mesh, so ``amortized=True`` is a *warm* MeshSet --
    the cache primed by one untimed call outside -- and ``amortized=False`` rebuilds the MeshSet per
    round so every round pays the factorization. That is the same distinction the ordito and
    reference rows draw, expressed in the one API where it needs no special path.
    """
    if amortized:
        warm_pml = _seeded_meshset_pml(bench_case)
        warm_pml.compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex()
        bench_case.run(
            warm_pml.compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex,
            rounds=_GEODESIC_ROUNDS,
        )
        assert warm_pml.current_mesh().vertex_scalar_array().shape == (bench_case.n_vertices,)
        return

    def solve_pml() -> None:
        _seeded_meshset_pml(
            bench_case
        ).compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex()

    bench_case.run(solve_pml, rounds=_GEODESIC_ROUNDS)


def _run_case(bench_case: BenchCase, *, amortized: bool = False) -> None:
    """
    Time one heat-geodesic field from vertex 0, in ordito, libigl or potpourri3d.

    With ``amortized=False`` the mesh-dependent setup is inside the timed callable for all three
    libraries; with ``amortized=True`` it is hoisted out and only the solve is timed. See the module
    docstring for why both are reported.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pyvista":
        # VTK's geodesic is Dijkstra over mesh *edges*, so it answers a different (and exactly
        # solvable) question -- an upper bound on the geodesic rather than the heat method's
        # smoothed approximation of it, which tests/test_heat_distance.py asserts. It has nothing
        # to amortize: the path search is the whole call.
        if amortized:
            pytest.skip("Dijkstra has no reusable factorization to hoist out of the timed region")
        # ``skip_larger_than`` is a no-op on the synthetic feature meshes, so the cap is by name,
        # as igl's is in the exact-geodesic row below.
        if bench_case.mesh_name == "sphere_large":
            pytest.skip("vtkDijkstra is a serial priority-queue walk: capped at sphere_med")
        mesh_pv = bench_case.mesh_pv
        target = n_vertices - 1
        distance_pv = bench_case.run(
            lambda: float(mesh_pv.geodesic_distance(0, target)), rounds=_GEODESIC_ROUNDS
        )
        assert np.isfinite(distance_pv)
        return
    if bench_case.kind == "pymeshlab":
        _run_case_pml(bench_case, amortized=amortized)
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        sources = _sources_wp(bench_case)
        if amortized:
            mesh = od.Trimesh(vertices, faces)
            for _ in range(2):
                od.heat.heat_geodesic(mesh, sources)
            solve = lambda: od.heat.heat_geodesic(mesh, sources)  # noqa: E731
        else:
            solve = lambda: od.heat.heat_geodesic(vertices, faces, sources)  # noqa: E731
        distance = bench_case.run(solve, rounds=_GEODESIC_ROUNDS)
        assert distance.shape == (n_vertices,)
        return

    vertices_np = bench_case.vertices_np
    faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)

    if bench_case.kind == "potpourri3d":
        # ``use_robust=False`` matches ordito's discretization: potpourri3d otherwise mollifies and
        # flips to an intrinsic Delaunay triangulation first (that path arrives with the plan's P5).
        if amortized:
            solver = pp3d.MeshHeatMethodDistanceSolver(vertices_np, faces_np, use_robust=False)
            solve_pp = lambda: solver.compute_distance(0)  # noqa: E731
        else:

            def solve_pp() -> np.ndarray:
                solver = pp3d.MeshHeatMethodDistanceSolver(vertices_np, faces_np, use_robust=False)
                return np.asarray(solver.compute_distance(0))

        assert np.asarray(bench_case.run(solve_pp, rounds=_GEODESIC_ROUNDS)).shape == (n_vertices,)
        return

    if amortized:
        data = igl.HeatGeodesicsData()
        igl.heat_geodesics_precompute(vertices_np, faces_np, data)
        solve_igl = lambda: np.asarray(igl.heat_geodesics_solve(data, _SOURCES))  # noqa: E731
    else:

        def solve_igl() -> np.ndarray:
            data = igl.HeatGeodesicsData()
            igl.heat_geodesics_precompute(vertices_np, faces_np, data)
            return np.asarray(igl.heat_geodesics_solve(data, _SOURCES))

    assert bench_case.run(solve_igl, rounds=_GEODESIC_ROUNDS).shape == (n_vertices,)


@pytest.mark.benchmark(group="heat_geodesic")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "igl", "potpourri3d", "pymeshlab", "pyvista")
@pytest.mark.parametrize("setup", ["full", "amortized"])
def test_heat_geodesic(bench_case: BenchCase, setup: str) -> None:
    """
    Two float64 CG solves plus the gradient normalization, over the clean size sweep.

    pyvista's row is the odd one out and is here as the bound rather than as a race: VTK computes
    the exact shortest path along *edges* to **one** target where every other row returns the whole
    field, so it does far less work and answers a different question. It is the class-C oracle for
    this group (``tests/test_heat_distance.py``), which is why it is timed at all.
    """
    _run_case(bench_case, amortized=setup == "amortized")


@pytest.mark.benchmark(group="heat_geodesic_conditioning")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "igl", "potpourri3d", "pymeshlab")
def test_heat_geodesic_conditioning(bench_case: BenchCase) -> None:
    """The same field and connectivity, well- and ill-conditioned: several times the cost."""
    _run_case(bench_case)


_mesh_ml_cache: dict[str, mm.Mesh] = {}


def _mesh_ml(bench_case: BenchCase) -> mm.Mesh:
    """Cache one ``meshlib.Mesh`` per mesh: the distance call reads it and returns a field."""
    if bench_case.mesh_name not in _mesh_ml_cache:
        _mesh_ml_cache[bench_case.mesh_name] = bench_case.new_mesh_ml()
    return _mesh_ml_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="fast_marching_distance")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("potpourri3d", "pymeshlab", "igl", "meshlib")
def test_fast_marching_distance(bench_case: BenchCase) -> None:
    """
    The serial single-source geodesics, for scale against the heat solvers on the same meshes.

    ordito deliberately has no equivalent -- fast marching advances a priority queue one vertex at
    a time and has no parallel formulation -- so it has no row here. It is here to price that
    decision: the non-PDE alternatives for the same task, on the same axis and the same meshes, so
    the numbers can be read next to the ``heat_geodesic`` table. This group therefore contributes no
    ``parity`` pairs by construction (see ``tests/parity.py``): "ordito agrees" is not a statement
    about a row ordito does not have.

    Four rows, four serial fronts: potpourri3d's fast marching solves the local Eikonal update per
    triangle, MeshLab's ``compute_scalar_by_geodesic_distance_from_given_point_per_vertex`` advances
    a Dijkstra-style front over the edge graph, **libigl's ``exact_geodesic``** propagates the MMP
    exact windows -- the only one that is exact rather than first-order -- and **meshlib's
    ``computeSurfaceDistances``** is a fourth Eikonal front, the only one of the four that is
    multi-threaded. Its accuracy is measured against ordito's heat method and against the exact
    great-circle field in ``tests/test_heat_distance.py``, where it matches ordito's own worst
    deviation, so this row is a like-for-like cost for a like-for-like answer.

    **igl is capped at ``sphere_small`` with ``rounds=1``, and the slope says why.** At one source
    with every vertex as a target it rises by well over an order of magnitude per 4x step, so the
    next mesh up is already impractical per round. Window propagation is the price of exactness, and
    that slope is the most useful thing this row records.

    Its call is also the one signature trap in the module: ``exact_geodesic(V, F, VS, FS, VT, FT)``
    needs **all six** arguments. A four-argument call binds ``vt`` to ``FS`` and returns an *empty
    array* rather than raising, so the two face arrays are passed explicitly empty.
    """
    if bench_case.kind == "meshlib":
        # ``startVertices`` is a VertBitSet over the vertex domain -- there is no index-list
        # overload -- and it is the input, so it is built outside the timed callable. The mesh is
        # read-only here and cached; ``maxVertUpdates`` stays at its default of 3, the accuracy
        # setting the correctness comparison uses.
        mesh_ml = _mesh_ml(bench_case)
        starts_ml = mm.VertBitSet()
        starts_ml.resize(mesh_ml.points.size(), False)
        starts_ml.set(mm.VertId(int(_SOURCES[0])), True)
        distances_ml = bench_case.run(
            lambda: mm.computeSurfaceDistances(mesh_ml, starts_ml), rounds=_GEODESIC_ROUNDS
        )
        assert distances_ml.size() == bench_case.n_vertices
        return
    if bench_case.kind == "igl":
        # ``skip_larger_than`` is a no-op on the synthetic feature meshes (they are not on the size
        # ladder), so the cap is by name.
        if bench_case.mesh_name != "sphere_small":
            pytest.skip("MMP window propagation: 839 ms at 10k vertices, 20.9 s at 41k")
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        sources_np = np.array([int(_SOURCES[0])], dtype=np.int64)
        targets_np = np.arange(bench_case.n_vertices, dtype=np.int64)
        no_faces_np = np.array([], dtype=np.int64)
        distance_igl = bench_case.run(
            lambda: igl.exact_geodesic(
                vertices_np, faces_np, sources_np, no_faces_np, targets_np, no_faces_np
            ),
            rounds=1,
        )
        assert distance_igl.shape == (bench_case.n_vertices,)
        return
    if bench_case.kind == "pymeshlab":
        # ``maxdistance=PureValue(0)`` disables the cut-off, so the front covers the whole mesh --
        # the default 50% of the bbox diagonal would stop early and measure less work.
        start_np = np.ascontiguousarray(bench_case.vertices_np[int(_SOURCES[0])], dtype=np.float64)
        bench_case.run(
            lambda: (
                bench_case.new_meshset_pml()
            ).compute_scalar_by_geodesic_distance_from_given_point_per_vertex(
                startpoint=start_np, maxdistance=ml.PureValue(0.0)
            ),
            rounds=_GEODESIC_ROUNDS,
        )
        return
    vertices_np = bench_case.vertices_np
    faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)
    # One source vertex, as a single-point "curve" of barycentric points.
    sources_pp = [[(int(_SOURCES[0]), [])]]

    def solve_pp() -> np.ndarray:
        solver = pp3d.MeshFastMarchingDistanceSolver(vertices_np, faces_np)
        return np.asarray(solver.compute_distance(sources_pp))

    assert bench_case.run(solve_pp, rounds=_GEODESIC_ROUNDS).shape == (bench_case.n_vertices,)


# --------------------------------------------------------------------------
# heat_signed_distance
# --------------------------------------------------------------------------

# Three float64 solves per case on ordito's side; two factorizations on the reference's.
_SIGNED_DISTANCE_ROUNDS = 3

_curve_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def _curve(bench_case: BenchCase, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Source curves as a packed vertex buffer plus CSR bounds, cached per (mesh, kind).

    The bounds are not implicit: a one-ring has five or six vertices depending on the centre's
    valence, so slicing a packed buffer at a fixed stride would splice two rings together and give
    segments that are not mesh edges at all -- which the reference rejects outright, and which this
    method would silently splat along a chord.
    """
    key = (bench_case.mesh_name, kind)
    if key not in _curve_cache:
        ring, offsets, is_boundary = (
            array.numpy()
            for array in od.halfedge.vertex_one_rings(
                bench_case.faces_wp, n_vertices=bench_case.n_vertices
            )
        )
        faces_np = bench_case.faces_np
        interior = np.flatnonzero(~is_boundary)

        def cycle(center: int) -> np.ndarray:
            halfedges = ring[offsets[center] : offsets[center + 1]]
            return np.array([faces_np[h // 3][(h % 3 + 1) % 3] for h in halfedges], dtype=np.int32)

        if kind == "ring":
            cycles = [cycle(int(interior[len(interior) // 2]))]
        else:
            # 64 rings spread over the surface: still edge paths, two orders of magnitude more
            # segments than one ring.
            cycles = [
                cycle(int(center)) for center in interior[:: max(len(interior) // 64, 1)][:64]
            ]
        bounds = np.cumsum([0] + [len(c) for c in cycles]).astype(np.int32)
        _curve_cache[key] = (np.concatenate(cycles), bounds)
    return _curve_cache[key]


@pytest.mark.benchmark(group="heat_signed_distance")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito", "potpourri3d")
@pytest.mark.parametrize("curve_kind", ["ring", "band"])
def test_heat_signed_distance(bench_case: BenchCase, curve_kind: str) -> None:
    """One mesh, a 6-segment curve against a ~370-segment one, to price the source splat."""
    curve_np, bounds_np = _curve(bench_case, curve_kind)
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        curve = wp.array(curve_np, dtype=wp.int32, device=bench_case.device)
        offsets = wp.array(bounds_np, dtype=wp.int32, device=bench_case.device)
        distance = bench_case.run(
            lambda: od.heat.heat_signed_distance(vertices, faces, curve, offsets),
            rounds=_SIGNED_DISTANCE_ROUNDS,
        )
        assert distance.shape == (bench_case.n_vertices,)
    else:
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)
        curves_pp = [
            [(int(vertex), []) for vertex in curve_np[begin:end]]
            for begin, end in itertools.pairwise(bounds_np)
        ]

        def solve_pp() -> np.ndarray:
            solver = pp3d.MeshSignedHeatSolver(vertices_np, faces_np)
            return np.asarray(solver.compute_distance(curves_pp, level_set_constraint="ZeroSet"))

        assert bench_case.run(solve_pp, rounds=_SIGNED_DISTANCE_ROUNDS).shape == (
            bench_case.n_vertices,
        )


@pytest.mark.benchmark(group="heat_signed_distance_constraint")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("level_set_constraint", ["zero_set", "none"])
def test_heat_signed_distance_constraint(bench_case: BenchCase, level_set_constraint: str) -> None:
    """Pinning the curve to zero against solving unconstrained and shifting afterwards."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    curve = wp.array(_curve(bench_case, "ring")[0], dtype=wp.int32, device=bench_case.device)
    distance = bench_case.run(
        lambda: od.heat.heat_signed_distance(
            vertices, faces, curve, level_set_constraint=level_set_constraint
        ),
        rounds=_SIGNED_DISTANCE_ROUNDS,
    )
    assert distance.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="heat_signed_distance_conditioning")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "potpourri3d")
def test_heat_signed_distance_conditioning(bench_case: BenchCase) -> None:
    """The same curve on well- and ill-conditioned connectivity: three CG solves feel it thrice."""
    test_heat_signed_distance(bench_case, "ring")


# --------------------------------------------------------------------------
# vector_heat_operators / extend_scalar / transport_tangent_vectors / log_map
# --------------------------------------------------------------------------

# Every case is at least two float64 CG solves; the reference also factors two sparse systems.
_VECTOR_HEAT_ROUNDS = 3


def _solver_pp(bench_case: BenchCase) -> pp3d.MeshVectorHeatSolver:
    return pp3d.MeshVectorHeatSolver(
        bench_case.vertices_np,
        np.ascontiguousarray(bench_case.faces_np, dtype=np.int32),
        use_intrinsic_delaunay=False,
    )


@pytest.mark.benchmark(group="extend_scalar")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "potpourri3d")
def test_extend_scalar(bench_case: BenchCase) -> None:
    """Two scalar diffusions and a division, on the conditioning axis."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        sources = _sources_wp(bench_case)
        values = wp.array(np.array([1.0]), dtype=wp.float64, device=bench_case.device)
        extended = bench_case.run(
            lambda: od.heat.extend_scalar(vertices, faces, sources, values),
            rounds=_VECTOR_HEAT_ROUNDS,
        )
        assert extended.shape == (n_vertices,)
    else:
        assert np.asarray(
            bench_case.run(
                lambda: _solver_pp(bench_case).extend_scalar([0], [1.0]), rounds=_VECTOR_HEAT_ROUNDS
            )
        ).shape == (n_vertices,)


def _run_transport(bench_case: BenchCase, *, amortized: bool) -> None:
    """Time one parallel transport, with the operators either rebuilt or reused."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        sources = _sources_wp(bench_case)
        vectors = wp.array(
            np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=bench_case.device
        )
        if amortized:
            mesh = od.Trimesh(vertices, faces)
            for _ in range(2):
                od.heat.transport_tangent_vectors(mesh, sources, vectors)
            transport = lambda: od.heat.transport_tangent_vectors(  # noqa: E731
                mesh, sources, vectors
            )
        else:
            transport = lambda: od.heat.transport_tangent_vectors(  # noqa: E731
                vertices, faces, sources, vectors
            )
        transported, resolved = bench_case.run(transport, rounds=_VECTOR_HEAT_ROUNDS)
        assert transported.shape == (n_vertices,)
        assert resolved.shape == (n_vertices,)
    elif amortized:
        # The reference's own split: construct the solver once, then time only its solve.
        solver = _solver_pp(bench_case)
        assert np.asarray(
            bench_case.run(
                lambda: solver.transport_tangent_vectors([0], [[1.0, 0.0]]),
                rounds=_VECTOR_HEAT_ROUNDS,
            )
        ).shape == (n_vertices, 2)
    else:
        assert np.asarray(
            bench_case.run(
                lambda: _solver_pp(bench_case).transport_tangent_vectors([0], [[1.0, 0.0]]),
                rounds=_VECTOR_HEAT_ROUNDS,
            )
        ).shape == (n_vertices, 2)


@pytest.mark.benchmark(group="transport_tangent_vectors")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "potpourri3d")
@pytest.mark.parametrize("setup", ["full", "amortized"])
def test_transport_tangent_vectors(bench_case: BenchCase, setup: str) -> None:
    """A 2x2-block vector solve plus a scalar extension for the magnitude."""
    _run_transport(bench_case, amortized=setup == "amortized")


@pytest.mark.benchmark(group="log_map")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "potpourri3d")
def test_log_map(bench_case: BenchCase) -> None:
    """The most expensive of the three: a vector solve, a distance field and a gradient pass."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        logarithm = bench_case.run(
            lambda: od.heat.log_map(vertices, faces, 0), rounds=_VECTOR_HEAT_ROUNDS
        )
        assert logarithm.shape == (n_vertices,)
    else:
        # geometry-central's own strategy name for the same construction.
        assert np.asarray(
            bench_case.run(
                lambda: _solver_pp(bench_case).compute_log_map(0, "VectorHeat"),
                rounds=_VECTOR_HEAT_ROUNDS,
            )
        ).shape == (n_vertices, 2)


@pytest.mark.benchmark(group="vector_heat_scale")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "potpourri3d")
def test_transport_tangent_vectors_scale(bench_case: BenchCase) -> None:
    """The same transport over the size sweep, for the assembly-versus-solve split."""
    skip_larger_than(bench_case, "sphere_large")
    _run_transport(bench_case, amortized=False)
