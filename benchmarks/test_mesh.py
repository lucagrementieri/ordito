"""
Benchmarks for ``ordito.mesh.Trimesh``: what a cold cached property costs.

Axis: **cache state**, which is not a mesh property at all. ``Trimesh`` is a lazily-computed
container -- first access to ``warp_mesh`` or ``face_adjacency`` or ``boundary_loops`` runs the real
work, every access after that is a dict lookup. So the honest question about this class is not "how
does it scale" (each property scales the way its own module's benchmark says it does) but "which
properties are expensive enough that a caller needs to know they are being computed", and the
warm/cold gap is the only way to see that.

Each group therefore times two rows on the same mesh:

* **cold** -- a freshly constructed ``Trimesh``, built *inside* the timed callable, so every round
  pays the full computation. Construction itself is two array assignments and is negligible.
* **warm** -- one shared instance whose property was already forced, so the row measures the dict
  lookup and nothing else. It exists as the floor: a warm row that is not ~0 means the property is
  not actually being cached.

The ratio between them is what a caller saves by holding onto the mesh, and it is the number to
quote when deciding whether an API should take a ``Trimesh`` or raw buffers.

Cache invalidation is a third row where it applies. ``with_vertices`` keeps every topology cache and
only drops the geometric ones, ``with_faces`` drops everything; that distinction is the reason
``with_vertices`` exists, so it is measured rather than asserted.

Properties chosen
-----------------
The six that are expensive for genuinely different reasons: ``warp_mesh`` (a BVH build),
``vertex_normals`` (an atomic scatter), ``face_adjacency`` (an edge sort plus manifold-pair
grouping), ``boundary_loops`` (pointer-jumping list ranking plus per-loop host work),
``is_watertight`` (which composes an edge test with a self-intersection pass over a fresh BVH) and
``vector_heat_operators`` (three sparse assemblies, and the priciest of the lot). Timing all of them
would just re-run the rest of the suite through a different door.

Ranked by cold cost, the properties fall into two groups an order of magnitude apart: the operator
bundles (``vector_heat_operators``, ``heat_operators``, ``vertex_tangent_frames``) plus
``is_watertight`` at the top, and the connectivity and bounds properties
(``halfedge_twins``, ``warp_mesh``, ``mass_matrix_entries``, ``vertex_face_adjacency``, ``bounds``)
at the bottom. The operator bundles are where the cache pays.

References
----------
**trimesh**'s ``Trimesh`` is the model this class mirrors, and it caches the same way, so the cold
rows are directly comparable -- the trimesh side is likewise rebuilt inside the timed callable. It
has no ``with_vertices`` / ``with_faces`` equivalent (assigning to ``mesh.vertices`` invalidates
everything), so the invalidation group is ordito-only. open3d and libigl have no caching container
at all: open3d recomputes on request and libigl is free functions over raw arrays.
"""

from __future__ import annotations

import pytest
import trimesh as tm

import ordito as od
from conftest import BenchCase, skip_larger_than

# ``is_watertight`` composes a self-intersection pass over a fresh BVH; a few rounds is enough.
_ROUNDS = 3

_warm_cache: dict[tuple[str, str], od.Trimesh] = {}


def _cold(bench_case: BenchCase) -> od.Trimesh:
    """Construct a fresh ``Trimesh`` with an empty cache -- two array assignments, no work."""
    return od.Trimesh(bench_case.vertices_wp, bench_case.faces_wp)


def _warm(bench_case: BenchCase, name: str) -> od.Trimesh:
    """Return a shared ``Trimesh`` with ``name`` already forced into its cache."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _warm_cache:
        _warm_cache[key] = _cold(bench_case)
    mesh = _warm_cache[key]
    getattr(mesh, name)
    return mesh


def _time_property(bench_case: BenchCase, name: str, *, warm: bool, rounds: int = 10) -> None:
    """Time one cached property, either from a fresh mesh or from a pre-forced shared one."""
    if warm:
        mesh = _warm(bench_case, name)
        assert bench_case.run(lambda: getattr(mesh, name), rounds=rounds) is not None
    else:
        assert bench_case.run(lambda: getattr(_cold(bench_case), name), rounds=rounds) is not None


@pytest.mark.benchmark(group="mesh_warp_mesh")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_warp_mesh(bench_case: BenchCase, warm: bool) -> None:
    """
    ``warp_mesh``: a BVH build over all faces, or a dict lookup.

    ordito-only, and the cold row is why: what it builds is a ``wp.Mesh``, a Warp construct with no
    counterpart to time. The reference libraries do keep lazily-built accelerators of their own
    (trimesh's ``ray`` / ``nearest`` adaptors, meshlib's ``AABBTree``), but each wraps a different
    structure with a different fanout, so a build-time ratio would be a comparison of data
    structures rather than of this property. What *is* comparable is the query cost those structures
    exist for, and the ``proximity`` and ``ray`` groups own that against all of them.
    """
    skip_larger_than(bench_case, "happy_buddha")
    _time_property(bench_case, "warp_mesh", warm=warm)


@pytest.mark.benchmark(group="mesh_vertex_normals")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_vertex_normals(bench_case: BenchCase, warm: bool) -> None:
    """``vertex_normals``: face normals and areas, then an atomic scatter into the vertices."""
    if bench_case.kind == "ordito":
        _time_property(bench_case, "vertex_normals", warm=warm)
        return
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    if warm:
        mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
        assert mesh_tm.vertex_normals is not None
        assert bench_case.run(lambda: mesh_tm.vertex_normals) is not None
    else:
        assert (
            bench_case.run(lambda: tm.Trimesh(vertices_np, faces_np, process=False).vertex_normals)
            is not None
        )


@pytest.mark.benchmark(group="mesh_face_adjacency")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_face_adjacency(bench_case: BenchCase, warm: bool) -> None:
    """``face_adjacency``: a radix sort over ``3F`` edge keys plus manifold-pair grouping."""
    if bench_case.kind == "ordito":
        _time_property(bench_case, "face_adjacency", warm=warm)
        return
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    if warm:
        mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
        assert mesh_tm.face_adjacency is not None
        assert bench_case.run(lambda: mesh_tm.face_adjacency) is not None
    else:
        assert (
            bench_case.run(lambda: tm.Trimesh(vertices_np, faces_np, process=False).face_adjacency)
            is not None
        )


def _time_trimesh_property(
    bench_case: BenchCase, name: str, *, warm: bool, rounds: int = 10
) -> None:
    """
    Time the matching ``tm.Trimesh`` cached property or method, cold or warm.

    The trimesh side caches exactly the way this container does, which is what makes the cold rows
    comparable and the warm rows meaningful on both sides: cold rebuilds the ``Trimesh`` inside the
    timed callable so every round pays the computation, warm forces the attribute once outside it
    and then measures the memoized access.

    ``getattr`` covers both shapes because ``outline`` is a method and ``is_watertight`` a property
    -- calling the result when it is callable is what lets one helper serve both, and trimesh caches
    the method's internals either way (see ``benchmarks/test_boundary.py``).
    """
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

    def force(mesh_tm: tm.Trimesh) -> object:
        attribute = getattr(mesh_tm, name)
        return attribute() if callable(attribute) else attribute

    if warm:
        mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
        assert force(mesh_tm) is not None
        assert bench_case.run(lambda: force(mesh_tm), rounds=rounds) is not None
    else:
        assert (
            bench_case.run(
                lambda: force(tm.Trimesh(vertices_np, faces_np, process=False)), rounds=rounds
            )
            is not None
        )


@pytest.mark.benchmark(group="mesh_boundary_loops")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_boundary_loops(bench_case: BenchCase, warm: bool) -> None:
    """
    ``boundary_loops`` through the container, on the loop axis.

    The one property whose cold cost depends on something other than size, so it gets the ``loops``
    axis rather than the scan sweep -- six rows that show the caching gap and the loop-count gap at
    once. [`test_boundary.py`](test_boundary.py) measures the underlying function.

    ``Trimesh.outline()`` is the reference, the same call that module's ``boundary_loops`` group
    times -- and it belongs here for the reason the whole module exists: it is a *cached* entry
    point on the class this one mirrors, so the warm/cold pair is a like-for-like comparison of two
    containers rather than of two loop extractors. Its answer is a ``Path3D`` of the same loops,
    which ``tests/test_boundary.py`` compares as ordered vertex lists.
    """
    if bench_case.kind == "trimesh":
        _time_trimesh_property(bench_case, "outline", warm=warm)
        return
    _time_property(bench_case, "boundary_loops", warm=warm)


@pytest.mark.benchmark(group="mesh_is_watertight")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_is_watertight(bench_case: BenchCase, warm: bool) -> None:
    """
    ``is_watertight``: the priciest property, since it builds a BVH for self-intersection.

    ``Trimesh.is_watertight`` is the reference and carries the caveat its own group in
    ``benchmarks/test_validation.py`` records: trimesh answers the *edge-manifold* clause alone,
    where ordito's property composes that with a self-intersection pass over a fresh BVH. So the
    trimesh row is a **lower bound** on this group's work rather than the same computation, and the
    cold ratio should be read as "what the self-intersection half costs" -- which is the number the
    property's docstring is really about. The two agree on edge-manifold input, which is what
    ``tests/test_validation.py`` pins.
    """
    skip_larger_than(bench_case, "dragon", "the self-intersection pass dominates beyond dragon")
    if bench_case.kind == "trimesh":
        _time_trimesh_property(bench_case, "is_watertight", warm=warm, rounds=_ROUNDS)
        return
    _time_property(bench_case, "is_watertight", warm=warm, rounds=_ROUNDS)


@pytest.mark.benchmark(group="mesh_vector_heat_operators")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_vector_heat_operators(bench_case: BenchCase, warm: bool) -> None:
    """
    ``vector_heat_operators``: the priciest property, and the one three others are shared with.

    Three sparse assemblies -- the connection Laplacian, the scalar heat bundle and the tangent
    frames -- of which the last two are the class's own ``heat_operators`` and
    ``vertex_tangent_frames``, so this cold row is also the cold cost of all three together. That
    sharing is what the group is really about: a caller solving on one mesh through a ``Trimesh``
    pays this once where a caller passing raw buffers to
    [`transport_tangent_vectors`](../ordito/heat/vector.py) et al. pays it per call.

    On the **scale** axis rather than the scan sweep, for a correctness reason and not a cost one:
    the tangent frames are a rotation about each vertex, so this whole property raises on an
    edge-non-manifold mesh -- ``bunny_decimated``'s 150 three-faced edges, measured -- and the
    scan registry is exactly the set of meshes that have them. That is the same restriction
    ``benchmarks/test_tangent_space.py`` and ``vector_heat_scale`` already run under.

    ordito-only. ``potpourri3d.MeshVectorHeatSolver`` is the stateful analogue and would make a
    fair cold row, but it is a *different* set of operators (it retriangulates to an intrinsic
    Delaunay mesh by default), so pairing it here would need a parity entry the value comparison in
    ``tests/test_heat_vector.py`` already owns under its own group name.
    """
    _time_property(bench_case, "vector_heat_operators", warm=warm, rounds=_ROUNDS)


@pytest.mark.benchmark(group="mesh_invalidation")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("mode", ["with_vertices", "with_faces"], ids=["keepstopo", "dropall"])
def test_invalidation(bench_case: BenchCase, mode: str) -> None:
    """
    Rebuilding ``face_adjacency`` after a geometry change against after a topology change.

    ``with_vertices`` keeps the topology caches, so the adjacency should come back free;
    ``with_faces`` drops everything and pays the edge sort again. That gap is the entire reason
    ``with_vertices`` exists as a separate method, so it is worth a measurement rather than a
    comment.

    ordito-only, and not for want of a reference: the quantity is a *cache policy*, so there is
    nothing to agree with. Which derived properties survive a geometry change is a decision this
    class makes, and no other library partitions its caches the same way -- trimesh invalidates by
    a hash over the whole buffer, so its equivalent of ``with_vertices`` drops the topology caches
    too and the two modes collapse into one. A pair here would assert that two libraries made the
    same design choice, which is not a correctness claim.
    """
    skip_larger_than(bench_case, "happy_buddha")
    mesh = _warm(bench_case, "face_adjacency")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp

    def run() -> object:
        derived = (
            mesh.with_vertices(vertices) if mode == "with_vertices" else mesh.with_faces(faces)
        )
        return derived.face_adjacency

    assert bench_case.run(run) is not None
