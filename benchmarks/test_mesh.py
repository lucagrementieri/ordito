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

Every other public member has a group too, after those six: the remaining cached properties as the
same cold/warm pair, the solver accessors (``heat_solver``, ``fixed_vertex_solver``) and
``from_warp_mesh`` on one small mesh, since they do no device work, and the methods that query
(``contains``, ``mesh_for_rays``) at the operating point of the function they wrap.

References
----------
**trimesh**'s ``Trimesh`` is the model this class mirrors, and it caches the same way, so the cold
rows are directly comparable -- the trimesh side is likewise rebuilt inside the timed callable. It
has no ``with_vertices`` / ``with_faces`` equivalent (assigning to ``mesh.vertices`` invalidates
everything), so the invalidation group is ordito-only. open3d and libigl have no caching container
at all: open3d recomputes on request and libigl is free functions over raw arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp

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


@pytest.mark.benchmark(group="mesh_carry_directions")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("mode", ["invert", "rotate"])
def test_carry_directions(bench_case: BenchCase, mode: str) -> None:
    """
    ``invert`` and a rigid ``transform`` with both normal buffers cached: the carry, not a rebuild.

    ``invert`` negates the cached face and vertex normals, a rotation maps them by the normal
    matrix; either way in one launch over both buffers, beside the face-buffer work the new mesh
    needs anyway. ordito-only for the reason ``mesh_invalidation`` is: the quantity is a cache
    policy, with nothing to agree with.
    """
    skip_larger_than(bench_case, "happy_buddha")
    mesh = _warm(bench_case, "face_normals")
    _ = mesh.vertex_normals
    rotation = od.transform.rotation_matrix((0.0, 0.0, 1.0), 0.3)

    def run() -> od.Trimesh:
        return mesh.invert() if mode == "invert" else mesh.transform(rotation)

    assert bench_case.run(run) is not None


@pytest.mark.benchmark(group="mesh_from_warp_mesh")
@pytest.mark.benchmeshes("sphere_small")
@pytest.mark.benchlibs("ordito")
def test_from_warp_mesh(bench_case: BenchCase) -> None:
    """
    ``from_warp_mesh``: wrapping a built ``wp.Mesh``, its BVH seeded as ``warp_mesh``.

    The container's host cost and nothing else, so one small mesh: read against
    ``mesh_warp_mesh``'s cold row, it is what a caller who already holds a ``wp.Mesh`` saves.
    ordito-only, as ``mesh_warp_mesh`` is.
    """
    warp_mesh = wp.Mesh(points=bench_case.vertices_wp, indices=bench_case.faces_wp)
    mesh = bench_case.run(lambda: od.Trimesh.from_warp_mesh(warp_mesh))
    assert mesh.n_faces == bench_case.n_faces


@pytest.mark.benchmark(group="mesh_traced_mesh")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_traced_mesh(bench_case: BenchCase, warm: bool) -> None:
    """
    ``traced_mesh``: the cuBQL-built BVH on CUDA, ``warp_mesh`` itself on the CPU.

    Read the cold row against ``mesh_warp_mesh``'s: the gap is the costlier build the faster tree
    pays, which ``mesh_for_rays`` weighs against the rays it saves. ordito-only for the reason
    ``mesh_warp_mesh`` is.
    """
    skip_larger_than(bench_case, "happy_buddha")
    _time_property(bench_case, "traced_mesh", warm=warm, rounds=_ROUNDS)


@pytest.mark.benchmark(group="mesh_mesh_for_rays")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("n_rays", [1, 1 << 40], ids=["rent", "buy"])
def test_mesh_for_rays(bench_case: BenchCase, n_rays: int) -> None:
    """
    ``mesh_for_rays`` on a fresh mesh: one ray builds ``warp_mesh``, a huge count ``traced_mesh``.

    Both rows start from an empty cache, so they are the two builds the rent-or-buy rule chooses
    between, reached through the method that chooses. On the CPU device both return the same mesh.
    """
    skip_larger_than(bench_case, "happy_buddha")
    traced = bench_case.run(lambda: _cold(bench_case).mesh_for_rays(n_rays), rounds=_ROUNDS)
    assert traced is not None


@pytest.mark.benchmark(group="mesh_triangles_center")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_triangles_center(bench_case: BenchCase, warm: bool) -> None:
    """``triangles_center``: one per-face map, against ``trimesh.Trimesh.triangles_center``."""
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "trimesh":
        _time_trimesh_property(bench_case, "triangles_center", warm=warm)
        return
    _time_property(bench_case, "triangles_center", warm=warm)


@pytest.mark.benchmark(group="mesh_centroid")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_centroid(bench_case: BenchCase, warm: bool) -> None:
    """``centroid``: the area-weighted face-centre mean, one reduction and one readback."""
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "trimesh":
        _time_trimesh_property(bench_case, "centroid", warm=warm)
        return
    _time_property(bench_case, "centroid", warm=warm)


@pytest.mark.benchmark(group="mesh_center_mass")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_center_mass(bench_case: BenchCase, warm: bool) -> None:
    """
    ``center_mass``: the volume-weighted centroid of the origin tetrahedra, against trimesh's.

    Only meaningful on a closed surface (``tests/test_mesh.py`` compares on closed fixtures), but
    the integration costs the same on an open one, so the scan sweep times it as it stands.
    """
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "trimesh":
        _time_trimesh_property(bench_case, "center_mass", warm=warm)
        return
    _time_property(bench_case, "center_mass", warm=warm)


@pytest.mark.benchmark(group="mesh_moment_inertia")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_moment_inertia(bench_case: BenchCase, warm: bool) -> None:
    """``moment_inertia``: the second moments of the same tetrahedra, about ``center_mass``."""
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "trimesh":
        _time_trimesh_property(bench_case, "moment_inertia", warm=warm)
        return
    _time_property(bench_case, "moment_inertia", warm=warm)


@pytest.mark.benchmark(group="mesh_extents")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_extents(bench_case: BenchCase, warm: bool) -> None:
    """``extents``: the box's side lengths, one vertex reduction on a cold mesh."""
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "trimesh":
        _time_trimesh_property(bench_case, "extents", warm=warm)
        return
    _time_property(bench_case, "extents", warm=warm)


@pytest.mark.benchmark(group="mesh_edges_sorted")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_edges_sorted(bench_case: BenchCase, warm: bool) -> None:
    """``edges_sorted``: every half-edge with its endpoints ordered, against trimesh's."""
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "trimesh":
        _time_trimesh_property(bench_case, "edges_sorted", warm=warm)
        return
    _time_property(bench_case, "edges_sorted", warm=warm)


@pytest.mark.benchmark(group="mesh_faces_unique_edges")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_faces_unique_edges(bench_case: BenchCase, warm: bool) -> None:
    """
    ``faces_unique_edges``: each face's three unique-edge indices, a view of the sort's inverse.

    So the cold row is the ``edges_unique`` sort, and trimesh's is its own unique-row grouping.
    """
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "trimesh":
        _time_trimesh_property(bench_case, "faces_unique_edges", warm=warm)
        return
    _time_property(bench_case, "faces_unique_edges", warm=warm)


@pytest.mark.benchmark(group="mesh_halfedge_mates")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_halfedge_mates(bench_case: BenchCase, warm: bool) -> None:
    """
    ``halfedge_mates``: each half-edge's partner, the pairing ``halfedge_twins`` validates.

    ordito-only: trimesh has no half-edge table, and ``tests/test_mesh.py`` pins the property to
    its free function.
    """
    skip_larger_than(bench_case, "happy_buddha")
    _time_property(bench_case, "halfedge_mates", warm=warm)


@pytest.mark.benchmark(group="mesh_oriented_boundary_edges")
@pytest.mark.benchlibs("ordito")
@pytest.mark.censusmesh("saddle_small")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_oriented_boundary_edges(bench_case: BenchCase, warm: bool) -> None:
    """
    ``oriented_boundary_edges``: the open edges, wound with their face.

    ordito-only: ``tests/test_mesh.py`` pins it to its free function, whose own group in
    ``benchmarks/test_boundary.py`` carries the reference.
    """
    skip_larger_than(bench_case, "happy_buddha")
    _time_property(bench_case, "oriented_boundary_edges", warm=warm)


@pytest.mark.benchmark(group="mesh_boundary_vertex_indices")
@pytest.mark.benchlibs("ordito")
@pytest.mark.censusmesh("saddle_small")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_boundary_vertex_indices(bench_case: BenchCase, warm: bool) -> None:
    """
    ``boundary_vertex_indices``: the vertices on an open edge, ascending.

    ordito-only, for the reason ``mesh_oriented_boundary_edges`` is.
    """
    skip_larger_than(bench_case, "happy_buddha")
    _time_property(bench_case, "boundary_vertex_indices", warm=warm)


@pytest.mark.benchmark(group="mesh_is_winding_consistent")
@pytest.mark.benchlibs("ordito", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_is_winding_consistent(bench_case: BenchCase, warm: bool) -> None:
    """
    ``is_winding_consistent``: every shared edge traversed once each way, against trimesh's.

    A cached ``bool``, so the warm row's ``is not None`` check passes on ``False`` too.
    """
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "trimesh":
        _time_trimesh_property(bench_case, "is_winding_consistent", warm=warm)
        return
    _time_property(bench_case, "is_winding_consistent", warm=warm)


@pytest.mark.benchmark(group="mesh_is_orientable")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_is_orientable(bench_case: BenchCase, warm: bool) -> None:
    """
    ``is_orientable``: whether some winding of every component is consistent.

    ordito-only: trimesh has no orientability predicate (its ``is_winding_consistent`` answers the
    stronger question ``mesh_is_winding_consistent`` times).
    """
    skip_larger_than(bench_case, "happy_buddha")
    _time_property(bench_case, "is_orientable", warm=warm)


@pytest.mark.benchmark(group="mesh_is_self_intersecting")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_is_self_intersecting(bench_case: BenchCase, warm: bool) -> None:
    """
    ``is_self_intersecting``: the self-intersection half of ``is_watertight``, over a fresh BVH.

    Read against ``mesh_is_watertight``: the gap is its edge-manifold clause. ordito-only: the
    property delegates to ``validation.is_self_intersecting``, whose group carries the references.
    """
    skip_larger_than(bench_case, "dragon", "the self-intersection pass dominates beyond dragon")
    _time_property(bench_case, "is_self_intersecting", warm=warm, rounds=_ROUNDS)


@pytest.mark.benchmark(group="mesh_mesh_operator_pattern")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_mesh_operator_pattern(bench_case: BenchCase, warm: bool) -> None:
    """
    ``mesh_operator_pattern``: the key-sorted CSR pattern every mesh operator here shares.

    The structural sort ``cotmatrix``, ``laplacian_operator`` and the heat bundles reuse, so its
    cold row is the part of each of those a held ``Trimesh`` pays once. ordito-only: a sparsity
    pattern is this package's assembly scheme, with nothing to agree with.
    """
    skip_larger_than(bench_case, "happy_buddha")
    _time_property(bench_case, "mesh_operator_pattern", warm=warm)


@pytest.mark.benchmark(group="mesh_laplacian_operator")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_laplacian_operator(bench_case: BenchCase, warm: bool) -> None:
    """
    ``laplacian_operator``: the uniform graph Laplacian, assembled on ``mesh_operator_pattern``.

    ordito-only here: ``tests/test_mesh.py`` pins it to ``laplacian.laplacian``, whose own group in
    ``benchmarks/test_laplacian.py`` carries the reference.
    """
    skip_larger_than(bench_case, "happy_buddha")
    _time_property(bench_case, "laplacian_operator", warm=warm)


@pytest.mark.benchmark(group="mesh_heat_solver")
@pytest.mark.benchmeshes("sphere_small")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_heat_solver(bench_case: BenchCase, warm: bool) -> None:
    """
    ``heat_solver()``: the kept ``HeatSolver``, built on first call and looked up after.

    The solver assembles and factors nothing until its first solve (``benchmarks/test_heat.py``
    times those), so both rows are the accessor's host cost and one small mesh says it all.
    """
    if warm:
        mesh = _cold(bench_case)
        mesh.heat_solver()
        assert bench_case.run(mesh.heat_solver) is not None
    else:
        assert bench_case.run(lambda: _cold(bench_case).heat_solver()) is not None


@pytest.mark.benchmark(group="mesh_fixed_vertex_solver")
@pytest.mark.benchmeshes("sphere_small")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_fixed_vertex_solver(bench_case: BenchCase, warm: bool) -> None:
    """
    ``fixed_vertex_solver()``: the kept ``FixedVertexSolver``, built on first call.

    Like ``mesh_heat_solver``, the solver does no work until it factors or solves
    (``benchmarks/test_parametrization.py``), so both rows are the accessor's host cost.
    """
    if warm:
        mesh = _cold(bench_case)
        mesh.fixed_vertex_solver()
        assert bench_case.run(mesh.fixed_vertex_solver) is not None
    else:
        assert bench_case.run(lambda: _cold(bench_case).fixed_vertex_solver()) is not None


# Query count for ``contains``, matching ``benchmarks/test_ray.py``'s ``contains_points``.
_N_CONTAINS = 10_000


def _contains_points_np(bench_case: BenchCase) -> np.ndarray:
    """Uniform queries over the bounding box grown 40% about its centre, so both answers occur."""
    vertices_np = bench_case.vertices_np
    low_np, high_np = vertices_np.min(axis=0), vertices_np.max(axis=0)
    extent_np = 1.4 * (high_np - low_np)
    rng = np.random.default_rng(17)
    return 0.5 * (low_np + high_np) - 0.5 * extent_np + rng.random((_N_CONTAINS, 3)) * extent_np


@pytest.mark.benchmark(group="mesh_contains")
@pytest.mark.benchaxis("depth")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_contains(bench_case: BenchCase) -> None:
    """
    ``contains``: ray parity through the cached ``warp_mesh``, ``contains_points`` as a method.

    On the depth axis for the reason ``contains_points`` is: every crossing along each ray is
    counted. The mesh's BVH is forced outside the timed callable, which is the point of the method
    over the free function, so the row reads against ``contains_points`` at equal BVH state.

    ``trimesh.Trimesh.contains`` is the oracle in ``tests/test_mesh.py``; it casts through the
    embree intersector, whose scene is built outside the timed callable on the same rule. Three
    rounds.
    """
    points_np = _contains_points_np(bench_case)
    if bench_case.kind == "trimesh":
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        _ = mesh_tm.ray  # the embree scene is built lazily
        inside_tm = bench_case.run(lambda: mesh_tm.contains(points_np), rounds=3)
        assert inside_tm.shape == (_N_CONTAINS,)
        return
    mesh = _warm(bench_case, "warp_mesh")
    points = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
    )
    inside = bench_case.run(lambda: mesh.contains(points))
    assert inside.shape == (_N_CONTAINS,)
