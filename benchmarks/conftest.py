"""
Benchmark harness: mesh loading, CLI flags, parametrization and a GPU-safe timing fixture.

Run with ``uv run pytest benchmarks/`` (the default ``pytest`` run only collects ``tests/``).

The mesh registries themselves live in [`meshes.py`](meshes.py). There are two, because there are
two questions: the scan meshes sweep triangle count, and the synthetic *feature* meshes each
perturb one cost driver -- component count, boundary loop count or length, graph diameter,
triangle quality, vertex valence, ray depth, collision density -- while pinning the vertex and/or
face count to a shared control. A benchmark picks whichever is relevant:

- ``@pytest.mark.benchaxis("components")`` runs the named ``meshes.AXES`` comparison,
- ``@pytest.mark.benchmeshes("sphere_med", ...)`` names meshes directly for a one-off,
- neither runs the scan sweep, filtered by ``--size`` / ``--cpu-max-size``.

The rule the suite follows is **one group measures one axis, at 2-3 points**. A group that has no
mesh axis at all (``ordito.creation``) takes ``bench_lib`` instead of ``bench_case`` and sizes its
work with a plain ``pytest.mark.parametrize``.

Flags
-----
``--device={auto,cpu,cuda,both}``
    Which ``ordito`` targets to time. ``auto`` (default) uses cuda when CUDA is available,
    else falls back to cpu — ``ordito-cpu`` is not timed alongside cuda by default. Pass
    ``cpu`` for cpu only or ``both`` to time both ordito targets. The ``trimesh`` / ``igl`` /
    ``open3d`` / ``scipy`` / ``numpy`` / ``potpourri3d`` / ``pymeshlab`` / ``pyvista`` /
    ``meshlib`` / ``pymeshfix`` CPU baselines are always included, and
    ``pytorch3d-cuda`` whenever the installed pytorch3d carries a working CUDA extension --
    ``pytorch3d`` is the one reference with GPU kernels of its own, so it is *not* selected by
    ``--device``, which chooses among ordito's targets.
``--size=<comma list | all>``
    Restrict meshes to these size categories (``small,medium,large,extralarge,huge``).
``--cpu-max-size=<category>``
    CPU-bound libraries (``ordito-cpu``, ``trimesh``, ``igl``, ``open3d``, ``scipy``,
    ``numpy``, ``potpourri3d``, ``pymeshlab``, ``pyvista``, ``meshlib``, ``pymeshfix``) skip
    meshes larger than this unless the size was named explicitly in ``--size``. Default ``large``
    — so
    ``happy_buddha`` and ``lucy`` run GPU-only by default while
    ``bunny_decimated``/``bunny``/``dragon`` run on CPU.
``--bench-all-libs``
    Run every reference library regardless of ``_known_slow_libraries.json``. Off by default: a
    library already measured losing to ordito by more than 2x, and ranking 3rd-or-worse overall
    for that exact cell, is skipped rather than re-timed (``skip_known_slow``, below). Reference
    rows dominate the suite's timed regions and a library already far slower does not become more
    informative on the next mesh size. Pass this flag before regenerating the table with
    ``benchmarks/regenerate_known_slow_libraries.py``, so the new table is cut from a run that saw
    every library.
"""

from __future__ import annotations

import functools
import json
import operator
import os
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import pytest
import warp as wp
from meshes import (
    ALL_MESHES_BY_NAME,
    AXES,
    BUILDERS,
    DATA_DIR,
    MESH_ORDER,
    MESHES,
    MESHES_BY_NAME,
    SIZE_ORDER,
    MeshSpec,
)

if TYPE_CHECKING:
    import open3d as o3d
    import pymeshlab as ml
    import pytorch3d.structures as p3d_structures
    import pyvista as pv
    from meshlib import mrmeshpy as mm
    from pymeshfix._meshfix import PyTMesh
    from pytest_benchmark.fixture import BenchmarkFixture
    from typing_extensions import Buffer

# Number of timed rounds and untimed warm-up rounds. The warm-up covers Warp kernel JIT
# compilation (first launch of each kernel) and CPU cache priming. Groups whose work runs into
# hundreds of milliseconds pass a smaller ``rounds`` to ``run`` instead.
_ROUNDS = 10
_WARMUP_ROUNDS = 1
# Rounds for a ordito row of a *reference-free* cell -- one no reference library measures, such as
# the visibility family at ``lucy`` -- read from ``_reference_free_cells.json``. Ten rounds of a
# multi-second call timed against nothing were most of ``test_visibility``'s wall clock.
REFERENCE_FREE_ROUNDS = 3


class LibrarySpec(TypedDict):
    """A benchmarked library variant. ``device`` is ``None`` for the CPU-only references."""

    id: str
    kind: str
    device: str | None
    cpu_bound: bool


def skip_larger_than(bench_case: BenchCase, largest: str, reason: str = "") -> None:
    """
    Skip the current case when its mesh is larger than ``largest``.

    On the scan registry the comparison is registry order, which is triangle count. Off it -- the
    synthetic feature meshes -- it is a **vertex-count** comparison against the cap's own count,
    because a feature mesh is not on the size ladder and has no order to index.

    A cap is honoured wherever it is written, including off the size ladder: a feature mesh is
    sized to run everywhere that "everywhere" is linear or tree-accelerated, which says nothing
    about a reference with a quadratic cost curve.
    """
    spec = ALL_MESHES_BY_NAME.get(bench_case.mesh_name)
    cap_spec = ALL_MESHES_BY_NAME.get(largest)
    if spec is None or cap_spec is None:
        return
    if bench_case.mesh_name in MESHES_BY_NAME and largest in MESHES_BY_NAME:
        larger = MESH_ORDER.index(bench_case.mesh_name) > MESH_ORDER.index(largest)
    else:
        larger = spec["n_vertices"] > cap_spec["n_vertices"]
    if larger:
        pytest.skip(reason or f"{bench_case.mesh_name} is larger than the {largest} cap")


_KNOWN_SLOW_LIBRARIES_PATH = os.path.join(os.path.dirname(__file__), "_known_slow_libraries.json")


@functools.cache
def _known_slow_libraries() -> dict[
    tuple[str | None, str | None, tuple[tuple[str, str], ...], str], float
]:
    """
    Read the ``(group, mesh_name, rest, library)`` -> ratio table it names.

    Generated by ``benchmarks/regenerate_known_slow_libraries.py`` from a full round's own
    ``--benchmark-json`` output -- see that script's docstring for the policy. Missing or
    unreadable is treated as "skip nothing", not an error: the
    table is an optimization, and its absence should degrade to the pre-existing (slower)
    behaviour rather than fail collection.
    """
    try:
        with open(_KNOWN_SLOW_LIBRARIES_PATH) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    table = {}
    for entry in data.get("entries", []):
        key = (
            entry["group"],
            entry["mesh_name"],
            tuple(tuple(pair) for pair in entry["rest"]),
            entry["library"],
        )
        table[key] = entry["ratio"]
    return table


def skip_known_slow(
    request: pytest.FixtureRequest, group: str | None, mesh_name: str | None, library: str
) -> None:
    """
    Skip a reference-library row already measured losing badly enough to be uninformative.

    ``library`` beginning with ``ordito`` is never skipped — it is the subject, not a reference.
    Matching is exact on ``(group, mesh_name, rest, library)``, ``rest`` being every other
    parametrize value for this exact test case (read off ``request.node.callspec.params``, the
    same dict ``pytest-benchmark`` itself writes into ``--benchmark-json``'s ``params`` field) —
    so a cell the table does not name runs exactly as before, including every library the source
    round never saw (a fresh mesh, a fresh ``rest`` value, a fresh library). ``--bench-all-libs``
    bypasses this entirely, for regenerating the table or for a one-off run that wants every
    comparison back.
    """
    if library.startswith("ordito") or request.config.getoption("--bench-all-libs"):
        return
    table = _known_slow_libraries()
    if not table:
        return
    params = dict(request.node.callspec.params)
    params.pop("mesh_name", None)
    params.pop("library", None)
    rest = tuple(sorted((k, str(v)) for k, v in params.items()))
    ratio = table.get((group, mesh_name, rest, library))
    if ratio is not None:
        pytest.skip(
            f"known slow: {library} measured {ratio}x ordito-cuda's time and ranked "
            f"3rd-or-worse for this cell in the source round (see "
            f"benchmarks/_known_slow_libraries.json); pass --bench-all-libs to run it anyway"
        )


_REFERENCE_FREE_CELLS_PATH = os.path.join(os.path.dirname(__file__), "_reference_free_cells.json")


@functools.cache
def _reference_free_cells() -> frozenset[tuple[str, str | None, tuple[tuple[str, str], ...]]]:
    """
    Read the ``(group, mesh_name, rest)`` cells no reference library measured.

    Generated by ``benchmarks/regenerate_reference_free_cells.py`` from a full round's own
    ``--benchmark-json`` output -- see that script's docstring for why this is a table and not a
    run-time decision. Missing or unreadable is treated as "no such cell", which leaves every row at
    its default round count.
    """
    try:
        with open(_REFERENCE_FREE_CELLS_PATH) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return frozenset()
    return frozenset(
        (entry["group"], entry["mesh_name"], tuple(tuple(pair) for pair in entry["rest"]))
        for entry in data.get("entries", [])
    )


def reference_free_rounds(
    request: pytest.FixtureRequest, group: str | None, mesh_name: str | None, library: str
) -> int | None:
    """
    Return ``REFERENCE_FREE_ROUNDS`` for a ordito row of a reference-free cell, else ``None``.

    Keyed exactly as ``skip_known_slow`` keys its table, so the two agree on what a cell is. Only a
    ``ordito`` row is capped: a reference row of a listed cell does not exist in the source round.
    """
    if not library.startswith("ordito"):
        return None
    cells = _reference_free_cells()
    if not cells:
        return None
    params = dict(request.node.callspec.params)
    params.pop("mesh_name", None)
    params.pop("library", None)
    rest = tuple(sorted((k, str(v)) for k, v in params.items()))
    return REFERENCE_FREE_ROUNDS if (group, mesh_name, rest) in cells else None


# ``cpu_bound`` marks references that only ever run on the CPU (the baselines), so the harness can
# skip them on the largest meshes by default. ``open3d`` is marked CPU-bound even though the
# installed wheel is a CUDA build: the legacy ``open3d.pipelines`` / ``open3d.geometry`` APIs the
# baselines use are CPU-only (only the newer ``open3d.t`` tensor API has GPU kernels).
#
# ``scipy`` is a narrow baseline -- only a *geometry* reference for the k-NN queries
# (``spatial.KDTree``). Every benchmark carries an explicit ``benchlibs`` marker, so it generates
# cases only where a branch exists.
#
# ``potpourri3d`` (geometry-central) is CPU-only and the only reference for the heat-method family.
# Its solver objects cache their factorizations, so a benchmark must construct the solver *inside*
# the timed callable to measure the work ordito does per call.
#
# ``pymeshlab`` (MeshLab / VCGlib) is CPU-only and the broadest reference here. Two things shape
# every row: almost every filter *mutates* ``current_mesh()`` in place, and building the ``MeshSet``
# is a real per-vertex cost. So the MeshSet is built inside the timed callable via
# ``BenchCase.new_meshset_pml`` unless the filter is verified pure, and any row cheaper than the
# build cost is reporting the build. See ``BenchCase.new_meshset_pml`` for the full rule.
#
# ``pyvista`` (VTK) is CPU-only and single-threaded, and it wraps the **same VTK as vedo** -- so a
# group carries one of the two, never both, or the ratio compares a library against itself. Nothing
# is cached (repeat calls recompute in full, so one ``PolyData`` may be shared across the pure
# family), almost everything returns a *new* object and leaves its input alone (``inplace=False`` is
# the default; ``edge_mask`` is the one exception, writing ``point_ind`` into the input), and the
# build is cheap enough that ``mesh_pv`` is a shared cached property.
#
# ``meshlib`` (MeshLib's C++ core) is CPU-bound but, uniquely here, **multi-threaded**, where
# trimesh / igl / pymeshlab / pyvista are all effectively single-threaded. So a ``ordito-cuda`` vs
# ``meshlib`` ratio is a fair fight in a way the other CPU ratios are not, and the other edge of the
# same knife is that a ``ordito-cpu`` row loses to it on any parallel op regardless of algorithm.
# ``meshlib.mrcudapy`` is deliberately **not** used -- the plain ``mrmeshpy`` free functions are the
# reference, and a CUDA module would make the row incomparable with the others. Almost all of those
# mutate their ``Mesh`` in place, so the accessor is ``BenchCase.new_mesh_ml()`` and there is no
# cached property.
#
# ``pymeshfix`` (MeshFix / the TMesh kernel) is CPU-only, single-threaded and the *narrowest deep*
# reference here: nine bound algorithms, all repair. One fact decides every row: **the load is a
# large and often dominant share of it.** ``load_array`` is not a load but a connectivity repair,
# and a ``PyTMesh`` accepts exactly one load and one mutating call (a second raises), so the build
# goes inside the timed callable via ``BenchCase.new_tmesh_pmf()``. So a row exists **only where the
# operation is a substantial share of the round** -- the intersection family and
# ``clean_from_arrays`` qualify; the hole-fill and component-removal rows carry
# ``pytest.mark.parity(..., benchmarked=False)`` with the share in the reason instead. Capped at
# ``bunny``.
#
# For the algorithms *not* timed, **a share is necessary and not sufficient.** ``fix_connectivity``
# clears the bar on cost and is the counter-example: most of a round doing provably nothing, since
# ``load_array`` already ran it, so a row would price a no-op. ``clean`` and
# ``strong_intersection_removal`` are real work whose *result* is not comparable with any single
# ordito function. ``strong_degeneracy_removal`` is under the bar outright.
#
# ``numpy`` is the narrowest baseline of all: only a reference for the *array primitives*
# (``ordito.reduce``), where a host reduction over an already-resident buffer is the honest CPU
# floor. Deliberately not a geometry reference -- every other CPU baseline is built on NumPy.
#
# ``pytorch3d`` is the **only** reference with CUDA kernels of its own, so it takes *two* rows the
# way ordito does. The ``-cuda`` row is the point: the one GPU-against-GPU comparison in the suite,
# and not a foregone conclusion. pytorch3d's ``_C`` carries **no spatial structure on either
# device** -- no tree, no grid, just the pairwise loop -- so the ratio is a crossover rather than a
# constant: brute force with perfect coalescing beats a BVH descent while the problem still fits the
# device's bandwidth, and loses by orders of magnitude once the cloud is large. So the neighbour and
# chamfer groups need the point count as an *axis*; a one-size row reports whichever side of the
# crossover it landed on.
#
# **Half of that crossover is ordito's, and the row must not be read as a statement about brute
# force.** pytorch3d is a clean quadratic; ordito's ``query_nearest`` is **non-monotonic** over the
# same sweep at ``k >= 8``, so the honest reading of the small-cloud row is "ordito is well off its
# own large-cloud cost here", not "pytorch3d is faster". Two qualifications: the effect **does not
# exist at k = 1**, so it must not be read onto the ``*_k1`` rows, and it is **hash-grid specific**
# -- the ``bvh`` backend is monotonic over the identical sweep. Mechanism: the grid's cell width
# *is* ``initial_radius``, and a row whose true k-th distance runs past
# ``_knn_widest_grid_radius(cell, n)`` abandons the walk for an exact linear scan of the cloud.
#
# **There is deliberately no ``pytorch3d-cpu`` row.** pytorch3d is registered for its CUDA kernels,
# and its host path is a reference implementation rather than a tuned one, so a ``-cpu`` row
# measures the same missing spatial index at Theta(N^2) and adds nothing the other CPU baselines do
# not say better. Its neighbour and chamfer rows scale as a clean quadratic, which extrapolates to
# unbounded rounds on a scan mesh -- and two such rows once cost two whole modules' JSON, because
# ``--benchmark-json`` is written at session end. Where the CUDA extension is missing the reference
# is simply absent: ``_pytorch3d_cuda_available`` gates it, and nothing falls back to the host.
#
# Two things every pytorch3d row has to do. **Synchronize torch's stream** -- ``BenchCase.run``
# does it on the ``pytorch3d`` kind, because ``wp.synchronize_device`` synchronizes *Warp's* stream
# and torch launches are asynchronous, so without it a ``-cuda`` row times the launch and not the
# kernel. And **state its measured share**, on the pymeshfix precedent: the per-round overhead is
# the batched ``[None]`` wrap plus the ``Meshes`` construction and its cached ``*_packed()``
# derivations, and on a ``-cuda`` row also the host-to-device copy.
LIBRARIES: list[LibrarySpec] = [
    {"id": "ordito-cpu", "kind": "ordito", "device": "cpu", "cpu_bound": True},
    {"id": "ordito-cuda", "kind": "ordito", "device": "cuda:0", "cpu_bound": False},
    {"id": "trimesh", "kind": "trimesh", "device": None, "cpu_bound": True},
    {"id": "igl", "kind": "igl", "device": None, "cpu_bound": True},
    {"id": "open3d", "kind": "open3d", "device": None, "cpu_bound": True},
    {"id": "scipy", "kind": "scipy", "device": None, "cpu_bound": True},
    {"id": "numpy", "kind": "numpy", "device": None, "cpu_bound": True},
    {"id": "potpourri3d", "kind": "potpourri3d", "device": None, "cpu_bound": True},
    {"id": "pymeshlab", "kind": "pymeshlab", "device": None, "cpu_bound": True},
    {"id": "pyvista", "kind": "pyvista", "device": None, "cpu_bound": True},
    {"id": "meshlib", "kind": "meshlib", "device": None, "cpu_bound": True},
    {"id": "pymeshfix", "kind": "pymeshfix", "device": None, "cpu_bound": True},
    {"id": "pytorch3d-cuda", "kind": "pytorch3d", "device": "cuda:0", "cpu_bound": False},
]
LIBRARIES_BY_ID = {lib["id"]: lib for lib in LIBRARIES}


# ---------------------------------------------------------------------------
# mesh loading (shared numpy source; per-device warp buffers built from it)
# ---------------------------------------------------------------------------

_numpy_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_faces_wp_cache: dict[tuple[str, str, str], wp.array[wp.int32]] = {}
_vertices_wp_cache: dict[tuple[str, str, str], wp.array[wp.vec3]] = {}
_mean_edge_cache: dict[str, float] = {}
_o3d_cache: dict[str, o3d.geometry.TriangleMesh] = {}
_pml_cache: dict[str, ml.MeshSet] = {}
_pv_cache: dict[str, pv.PolyData] = {}
_p3d_cache: dict[tuple[str, str], p3d_structures.Meshes] = {}


def _load_numpy(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read ``(vertices_f64, faces_i64)`` once with meshio, cached across the session.

    Uses the same ``meshio.read`` path as ``ordito.io.load_mesh_data``; kept at NumPy level
    (float64 vertices / int64 faces) so igl and trimesh get their arrays directly and the warp
    buffers are built from the same source. Feature meshes are generated by ``meshes.BUILDERS``
    instead of read.
    """
    if name not in _numpy_cache:
        builder = BUILDERS.get(name)
        if builder is not None:
            vertices, faces = builder()
            _numpy_cache[name] = (
                np.ascontiguousarray(vertices, dtype=np.float64),
                np.ascontiguousarray(faces, dtype=np.int64),
            )
            return _numpy_cache[name]

        import meshio

        mesh = meshio.read(DATA_DIR / MESHES_BY_NAME[name]["filename"])
        faces = mesh.cells_dict.get("triangle")
        if faces is None or len(faces) == 0:
            raise ValueError(f"{name!r} has no triangle faces to benchmark.")
        vertices = np.ascontiguousarray(mesh.points, dtype=np.float64)
        _numpy_cache[name] = (vertices, np.ascontiguousarray(faces, dtype=np.int64))
    return _numpy_cache[name]


def _faces_wp(name: str, device: str) -> wp.array[wp.int32]:
    """Flat ``wp.int32`` face buffer on ``device``, cached per ``(name, device)``."""
    key = ("faces", name, device)
    if key not in _faces_wp_cache:
        _, faces = _load_numpy(name)
        _faces_wp_cache[key] = wp.array(
            np.ascontiguousarray(faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
        )
    return _faces_wp_cache[key]


def _mean_edge(name: str) -> float:
    """
    Mean undirected edge length of a mesh, from the shared NumPy source.

    Computed on the host so every library variant of a case gets the *same* value: benchmarks that
    size their work by edge length (remeshing targets, ball-pivoting radii) must not hand ordito
    and its reference slightly different parameters.
    """
    if name not in _mean_edge_cache:
        vertices, faces = _load_numpy(name)
        triangles = vertices[faces]
        edges = np.concatenate(
            (
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 1],
                triangles[:, 0] - triangles[:, 2],
            )
        )
        _mean_edge_cache[name] = float(np.linalg.norm(edges, axis=1).mean())
    return _mean_edge_cache[name]


def _new_mesh_o3d(name: str) -> o3d.geometry.TriangleMesh:
    """
    Build a fresh legacy ``open3d.geometry.TriangleMesh`` from the shared NumPy source.

    Imported lazily (like ``meshio`` above) so the ~1 s open3d import is only paid by runs that
    actually include an open3d case.
    """
    import open3d as o3d

    vertices, faces = _load_numpy(name)
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.ascontiguousarray(faces, dtype=np.int32)),
    )


def _new_meshset_pml(name: str) -> ml.MeshSet:
    """
    Build a fresh single-mesh ``pymeshlab.MeshSet`` from the shared NumPy source.

    Imported lazily (like ``meshio`` and ``open3d`` above) so the pymeshlab import and its Qt
    dependencies are only paid by runs that include a pymeshlab case. MeshLab wants float64
    positions and takes the ``(n_faces, 3)`` index array as-is.
    """
    import pymeshlab as ml

    vertices, faces = _load_numpy(name)
    meshset = ml.MeshSet()
    meshset.add_mesh(ml.Mesh(vertices, faces))
    return meshset


def _new_mesh_pv(name: str) -> pv.PolyData:
    """
    Build a ``pyvista.PolyData`` from the shared NumPy source, float64 positions preserved.

    Imported lazily (like ``meshio``, ``open3d`` and ``pymeshlab`` above) so the pyvista/VTK import
    is only paid by runs that include a pyvista case. Goes through ``from_regular_faces`` rather
    than the padded ``[3, i, j, k]`` cell array, which is an order of magnitude cheaper.
    """
    import pyvista as pv

    vertices, faces = _load_numpy(name)
    return pv.PolyData.from_regular_faces(vertices, np.ascontiguousarray(faces, dtype=np.int32))


def _new_mesh_p3d(name: str, device: str) -> p3d_structures.Meshes:
    """
    Build a ``pytorch3d.structures.Meshes`` on ``device`` from the shared NumPy source.

    Positions land as **float32** and faces as int64, which is what ordito's ``wp.vec3`` /
    ``wp.int32`` buffers hold and what pytorch3d's own kernels want; a float64 ``Meshes`` keeps
    float64 through ``verts_packed()``, so matching the storage keeps a ratio comparing the same
    arithmetic on both sides.

    Imported lazily (like ``meshio``, ``open3d``, ``pymeshlab``, ``pyvista``, ``meshlib`` and
    ``pymeshfix`` above) so the ~4 s torch + pytorch3d import is only paid by runs that include a
    pytorch3d case.
    """
    import pytorch3d.structures as p3d_structures
    import torch

    vertices, faces = _load_numpy(name)
    return p3d_structures.Meshes(
        verts=[torch.as_tensor(np.ascontiguousarray(vertices, dtype=np.float32), device=device)],
        faces=[torch.as_tensor(np.ascontiguousarray(faces, dtype=np.int64), device=device)],
    )


def mesh_ml_from_numpy(vertices: np.ndarray, faces: np.ndarray) -> mm.Mesh:
    """
    Build a ``meshlib.mrmeshpy.Mesh`` from a NumPy pair, **vertices first**.

    The benchmark-side twin of ``tests.conversions.numpy_to_meshlib`` (the two suites do not import
    each other), and it exists for the same reason: ``mn.meshFromFacesVerts`` takes *faces* first,
    the reverse of every other builder here, and a swapped call raises nothing because the two
    arrays differ in shape only when the counts differ. Keeping that swap in one place per suite is
    the point. Use this for a row whose mesh is constructed rather than loaded; a row on a registry
    mesh wants ``BenchCase.new_mesh_ml()`` instead.

    Imported lazily (like ``meshio``, ``open3d``, ``pymeshlab`` and ``pyvista`` above) so the
    0.256 s meshlib import is only paid by runs that include a meshlib case. Both dtypes are
    permissive -- float64 positions and int64 indices go through as the loader holds them.
    """
    from meshlib import mrmeshnumpy as mn

    # MeshLib's stubs type these as ``Buffer``, which numpy's stubs implement only from Python 3.12.
    return mn.meshFromFacesVerts(
        cast("Buffer", np.ascontiguousarray(np.asarray(faces).reshape(-1, 3))),
        cast("Buffer", np.ascontiguousarray(vertices)),
    )


def face_bitset_ml(mask_np: np.ndarray) -> mm.FaceBitSet:
    """
    Load a dense face mask into a ``FaceBitSet`` through the packed blocks, not a per-face loop.

    The benchmark-side twin of ``tests.conversions.numpy_to_meshlib_bitset`` (the two suites do not
    import each other). Three details that are each a silent wrong answer on their own:
    ``BitSet.fromBlocks`` takes ``uint64`` blocks and rejects a NumPy array (pass ``.tolist()``),
    ``bitorder="little"`` is not NumPy's default and is not optional, and ``fromBlocks`` rounds the
    size up to whole 64-bit blocks so the domain has to be restored afterwards.

    Orders of magnitude faster than the per-face ``set()`` loop it replaces.
    """
    from meshlib import mrmeshpy as mm

    packed_np = np.packbits(np.ascontiguousarray(mask_np, dtype=bool), bitorder="little")
    packed_np = np.pad(packed_np, (0, (-packed_np.size) % 8)).view(np.uint64)
    bitset_ml = mm.BitSet.fromBlocks(mm.std_vector_unsigned_long(packed_np.tolist()))
    bitset_ml.resize(int(mask_np.size))
    return mm.FaceBitSet(bitset_ml)


def _new_mesh_ml(name: str) -> mm.Mesh:
    """Build a fresh ``meshlib.mrmeshpy.Mesh`` from the shared NumPy source."""
    return mesh_ml_from_numpy(*_load_numpy(name))


def tmesh_pmf_from_numpy(vertices: np.ndarray, faces: np.ndarray) -> PyTMesh:
    """
    Build a ``pymeshfix._meshfix.PyTMesh`` from a NumPy pair, **vertices first** and quiet.

    The benchmark-side twin of ``tests.conversions.numpy_to_pymeshfix`` (the two suites do not
    import each other). ``set_quiet(True)`` is not cosmetic: the kernel narrates to stderr and one
    of its messages is reported inverted, so a row that left it on would print
    "MeshFix could not fix everything" once per round on meshes it repaired successfully.

    ``load_array`` is itself a connectivity repair -- it renumbers and may change both counts -- so
    a row asserting anything about the result reads the counts off the ``PyTMesh``, never off the
    input arrays.

    Imported lazily (like ``meshio``, ``open3d``, ``pymeshlab``, ``pyvista`` and ``meshlib`` above)
    so the pymeshfix import is only paid by runs that include a pymeshfix case.
    """
    from pymeshfix import _meshfix

    tin_pmf = _meshfix.PyTMesh()
    tin_pmf.set_quiet(True)
    tin_pmf.load_array(
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(np.asarray(faces).reshape(-1, 3), dtype=np.int32),
    )
    return tin_pmf


def _new_tmesh_pmf(name: str) -> PyTMesh:
    """Build a fresh ``PyTMesh`` from the shared NumPy source."""
    return tmesh_pmf_from_numpy(*_load_numpy(name))


def points_torch_from_numpy(points_np: np.ndarray, device: str) -> Any:
    """
    Upload an ``(n, 3)`` cloud as the ``(1, n, 3)`` float32 tensor ``pytorch3d.ops`` wants.

    The benchmark-side twin of ``tests.conversions.points_to_torch`` (the two suites do not import
    each other). **Every** ``pytorch3d.ops`` entry point is batched with a leading minibatch axis,
    so a single cloud goes in as ``x[None]`` and its answer comes out as ``result[0]``; a bare
    ``(P, 3)`` handed to ``knn_points`` is silently read as ``(N=P, P1=3, D)`` and compares three
    points at full speed, which would read as pytorch3d being 1 000x faster than it is. Keeping the
    wrap in one place per suite is the point.

    The upload is **not free and it is inside every row**: a real share of a small row and
    negligible on a large one. State the share in the group docstring, on the pymeshfix precedent.

    Imported lazily (like ``meshio``, ``open3d``, ``pymeshlab``, ``pyvista``, ``meshlib`` and
    ``pymeshfix`` above) so the ~4 s torch import is only paid by runs that include a pytorch3d
    case.
    """
    import torch

    return torch.as_tensor(
        np.ascontiguousarray(points_np, dtype=np.float32), device=device
    ).unsqueeze(0)


def _vertices_wp(name: str, device: str) -> wp.array[wp.vec3]:
    """``wp.vec3`` (float32) vertex buffer on ``device``, cached per ``(name, device)``."""
    key = ("verts", name, device)
    if key not in _vertices_wp_cache:
        vertices, _ = _load_numpy(name)
        _vertices_wp_cache[key] = wp.array(
            np.ascontiguousarray(vertices, dtype=np.float32), dtype=wp.vec3, device=device
        )
    return _vertices_wp_cache[key]


# ---------------------------------------------------------------------------
# flags & (mesh, library) parametrization
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    # --device is registered by the repo-root conftest.py: argparse rejects a duplicate option
    # string, and tests/conftest.py reads the same flag. See that file for why both suites' ``auto``
    # selects one device. Registering it here too made ``pytest tests benchmarks`` die with
    # ``conflicting option string: --device`` before collecting anything.
    group = parser.getgroup("ordito-bench")
    group.addoption(
        "--size",
        action="store",
        default="all",
        help="comma-separated mesh size categories to include, or 'all' (default).",
    )
    group.addoption(
        "--cpu-max-size",
        action="store",
        default="large",
        choices=SIZE_ORDER,
        help="largest mesh size CPU-bound libraries run on by default (default: large).",
    )
    group.addoption(
        "--bench-all-libs",
        action="store_true",
        default=False,
        help=(
            "run every reference library even where _known_slow_libraries.json says one already "
            "loses badly enough to skip by default (see skip_known_slow)."
        ),
    )


def _selected_sizes(config: pytest.Config) -> tuple[set[str], bool]:
    """Return ``(sizes, explicit)`` — the requested size set and whether it was user-specified."""
    raw = str(config.getoption("--size")).strip().lower()
    if raw == "all":
        return set(SIZE_ORDER), False
    sizes = {s.strip() for s in raw.split(",") if s.strip()}
    unknown = sizes - set(SIZE_ORDER)
    if unknown:
        raise pytest.UsageError(f"unknown --size categories: {sorted(unknown)}")
    return sizes, True


@functools.cache
def _pytorch3d_cuda_available() -> bool:
    """
    Whether the installed pytorch3d has a **working** CUDA extension, by running one kernel.

    ``torch.cuda.is_available()`` is the wrong probe and this is not a hypothetical: a wheel whose
    arch list stops short of the installed device returns ``True`` and then fails every kernel with
    *"no kernel image is available"*. The other failure
    mode is a build that compiled the CPU extension only -- selected silently whenever
    ``setup.py`` finds no ``CUDA_HOME``, which is the normal state of a stock CI runner -- and it
    raises ``RuntimeError: Not compiled with GPU support.`` on a CUDA tensor. Both would otherwise
    turn a whole benchmark file's ``pytorch3d-cuda`` rows into errors, so the row is gated on the
    only probe that distinguishes them: launch a two-point ``knn_points`` and see.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    import pytorch3d.ops as p3d_ops

    try:
        points_t = torch.zeros(1, 2, 3, device="cuda:0")
        p3d_ops.knn_points(points_t, points_t, K=1)
    except RuntimeError:
        return False
    return True


def _selected_libraries(config: pytest.Config) -> list[LibrarySpec]:
    device = str(config.getoption("--device"))
    cuda_available = wp.is_cuda_available()
    if device == "auto":
        # ordito-cpu is disabled by default: run it only as a fallback when there is no CUDA
        # device. Pass --device=cpu or --device=both to time it explicitly.
        include_cpu, include_cuda = not cuda_available, cuda_available
    elif device == "cpu":
        include_cpu, include_cuda = True, False
    elif device == "cuda":
        include_cpu, include_cuda = False, True
    else:  # both
        include_cpu, include_cuda = True, True

    if include_cpu and cuda_available:
        # A ``ordito-cpu`` row timed in a process where CUDA has been initialised is *inflated*, so
        # it is not merely a slow row -- it is a wrong one, and it reads as ordito losing to the
        # CPU references. The cost behaves like a per-launch charge, so the factor scales with
        # launch count rather than with work: a launch-light row barely notices, an iterative solver
        # pays an order of magnitude. Unchanged by ``warp.config.launch_array_access_mode``, so it
        # is CUDA presence and not the launch guard. This is why the
        # default ``auto`` picks ``ordito-cpu`` only when there is no CUDA device. A *warning*
        # rather than a refusal: a deliberate same-process CPU-vs-CUDA comparison is still a
        # legitimate thing to ask for, as long as the asker knows the CPU side is not publishable.
        warnings.warn(
            "ordito-cpu rows are being timed in a CUDA-initialised process, which inflates them "
            "(measured 1.0-1.4x on launch-light ops, ~20x on an iterative solver) and makes them "
            "incomparable with the CPU references. Run 'uv run python benchmarks/devices.py' "
            "instead, which times the CPU target in a separate process with "
            'CUDA_VISIBLE_DEVICES="".',
            UserWarning,
            stacklevel=2,
        )

    selected = []
    for lib in LIBRARIES:
        if lib["id"] == "ordito-cpu" and not include_cpu:
            continue
        if lib["id"] == "ordito-cuda" and not (include_cuda and cuda_available):
            continue
        if lib["id"] == "pytorch3d-cuda" and not _pytorch3d_cuda_available():
            continue
        # Every other baseline is always included: --device selects among ordito's targets, and
        # pytorch3d-cuda is gated on its own extension rather than on that flag.
        selected.append(lib)
    return selected


def _selected_meshes(config: pytest.Config, lib: LibrarySpec) -> list[MeshSpec]:
    sizes, explicit = _selected_sizes(config)
    cpu_max_rank = SIZE_ORDER.index(str(config.getoption("--cpu-max-size")))
    meshes = []
    for mesh in MESHES:
        if mesh["size"] not in sizes:
            continue
        # Cap CPU-bound libraries at --cpu-max-size unless the size was explicitly requested.
        if lib["cpu_bound"] and not explicit and SIZE_ORDER.index(mesh["size"]) > cpu_max_rank:
            continue
        meshes.append(mesh)
    return meshes


def _supported_kinds(metafunc: pytest.Metafunc) -> set[str]:
    marker = metafunc.definition.get_closest_marker("benchlibs")
    if marker is None:
        return {lib["kind"] for lib in LIBRARIES}
    return set(marker.args)


def _mesh_names(metafunc: pytest.Metafunc) -> tuple[str, ...] | None:
    """Explicit mesh names for a test from ``benchaxis`` / ``benchmeshes``, or ``None``."""
    axis = metafunc.definition.get_closest_marker("benchaxis")
    explicit = metafunc.definition.get_closest_marker("benchmeshes")
    if axis is not None and explicit is not None:
        raise pytest.UsageError(
            f"{metafunc.definition.nodeid}: benchaxis and benchmeshes are mutually exclusive."
        )
    if axis is not None:
        if len(axis.args) != 1 or axis.args[0] not in AXES:
            raise pytest.UsageError(
                f"benchaxis takes one name from {sorted(AXES)}, got {list(axis.args)}"
            )
        return AXES[axis.args[0]]
    return tuple(explicit.args) if explicit is not None else None


def _test_meshes(metafunc: pytest.Metafunc, lib: LibrarySpec) -> list[MeshSpec]:
    """Meshes for one test: the axis / explicit names, else the registry filtered by CLI flags."""
    names = _mesh_names(metafunc)
    if names is None:
        return _selected_meshes(metafunc.config, lib)
    meshes = []
    for name in names:
        spec = ALL_MESHES_BY_NAME.get(name)
        if spec is None:
            raise pytest.UsageError(f"unknown mesh name: {name!r}")
        meshes.append(spec)
    return meshes


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "library" not in metafunc.fixturenames:
        return
    supported = _supported_kinds(metafunc)
    libraries = [lib for lib in _selected_libraries(metafunc.config) if lib["kind"] in supported]

    if "mesh_name" not in metafunc.fixturenames:
        # Mesh-free benchmark (``bench_lib``): parametrize over libraries alone. The mesh registry
        # and its size filters have nothing to select here -- so a mesh marker on such a test would
        # be silently dropped, which is worth failing over rather than ignoring.
        if _mesh_names(metafunc) is not None:
            raise pytest.UsageError(
                f"{metafunc.definition.nodeid}: benchaxis / benchmeshes need the bench_case "
                "fixture; a bench_lib benchmark has no mesh to select."
            )
        metafunc.parametrize("library", [lib["id"] for lib in libraries], ids=None)
        return

    cases: list[tuple[str, str]] = []
    ids: list[str] = []
    for lib in libraries:
        for mesh in _test_meshes(metafunc, lib):
            cases.append((mesh["name"], lib["id"]))
            ids.append(f"{mesh['name']}-{lib['id']}")
    metafunc.parametrize(("mesh_name", "library"), cases, ids=ids)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "benchlibs(*kinds): library kinds "
        "(ordito/trimesh/igl/open3d/scipy/numpy/potpourri3d/pymeshlab/pyvista/meshlib/"
        "pymeshfix/pytorch3d) a "
        "benchmark supports.",
    )
    config.addinivalue_line(
        "markers",
        "benchaxis(name): feature-mesh axis from meshes.AXES for a benchmark; bypasses --size.",
    )
    config.addinivalue_line(
        "markers",
        "benchmeshes(*names): mesh names for a benchmark (registry or feature); bypasses --size.",
    )
    # Declared here but enforced from tests/: the parity gate is a static scan so that it runs in
    # the default ``pytest`` invocation, which does not collect this directory at all. Validating
    # here as well would put the same rule in two places, and the weaker of the two would be the
    # one that only fires when somebody remembers to run the benchmarks.
    config.addinivalue_line(
        "markers",
        "noparity(kind, reason=..., oracle=...): a reference this benchmark times whose *result* "
        "is not comparable with ordito's. reason= is required prose; oracle= names the library "
        "that is the correctness oracle instead, and must itself be covered. See tests/parity.py.",
    )
    # Default to one comparison table per (function, mesh): the ``group`` marker is the function
    # name and ``param:mesh_name`` splits by mesh, so each table lists the libraries side by side.
    # Only applied when the user did not pass their own ``--benchmark-group-by``.
    if getattr(config.option, "benchmark_group_by", "group") == "group":
        config.option.benchmark_group_by = "group,param:mesh_name"


@pytest.hookimpl(tryfirst=True)
def pytest_benchmark_group_stats(
    config: pytest.Config, benchmarks: list[Any], group_by: str
) -> list[tuple[str | None, list[Any]]]:
    """
    Group results like pytest-benchmark does, but tolerate a ``param:`` axis a case does not have.

    The plugin's own implementation indexes ``bench["params"][name]`` directly, so the default
    ``group,param:mesh_name`` grouping raises ``KeyError`` as soon as one mesh-free benchmark
    (``bench_lib``) is collected alongside the mesh-driven ones. Missing axes are skipped instead,
    which puts each mesh-free group under its ``group`` name alone.
    """
    del config
    groups: dict[str | None, list[Any]] = defaultdict(list)
    for bench in benchmarks:
        key: tuple[object, ...] = ()
        for grouping in group_by.split(","):
            if grouping == "group":
                key += (bench["group"],)
            elif grouping == "name":
                key += (bench["name"],)
            elif grouping == "func":
                key += (bench["name"].split("[")[0],)
            elif grouping == "fullname":
                key += (bench["fullname"],)
            elif grouping == "fullfunc":
                key += (bench["fullname"].split("[")[0],)
            elif grouping == "param":
                key += (bench["param"],)
            elif grouping.startswith("param:"):
                name = grouping[len("param:") :]
                params = bench["params"] or {}
                if name in params:
                    key += (f"{name}={params[name]}",)
            else:
                raise NotImplementedError(f"Unsupported grouping {group_by!r}.")
        groups[" ".join(str(part) for part in key if part) or None].append(bench)

    for grouped in groups.values():
        grouped.sort(key=operator.itemgetter("fullname" if "full" in group_by else "name"))
    return sorted(groups.items(), key=lambda pair: pair[0] or "")


# ---------------------------------------------------------------------------
# per-case fixture: bundles inputs + a GPU-safe timing call
# ---------------------------------------------------------------------------


class BenchLibrary:
    """One library variant plus a timed ``run``, with no mesh attached."""

    def __init__(
        self, library: str, benchmark: BenchmarkFixture, max_rounds: int | None = None
    ) -> None:
        """
        Bind a library variant and the pytest-benchmark fixture.

        ``max_rounds`` caps every ``run`` of this case -- ``REFERENCE_FREE_ROUNDS`` for a ordito
        row of a reference-free cell, ``None`` otherwise.
        """
        self.library: LibrarySpec = LIBRARIES_BY_ID[library]
        self._benchmark = benchmark
        self._max_rounds = max_rounds

    @property
    def kind(self) -> str:
        """Library family: one of the ``kind`` values in ``LIBRARIES``."""
        return self.library["kind"]

    @property
    def device(self) -> str | None:
        """Warp device for ordito targets; ``None`` for CPU-only references."""
        return self.library["device"]

    @property
    def torch_device(self) -> str:
        """
        Torch device string for the ``pytorch3d`` row.

        Always ``"cuda:0"`` in practice: pytorch3d is registered for its CUDA kernels alone (see
        the ``LIBRARIES`` block) and ``_pytorch3d_cuda_available`` drops the reference entirely
        where the extension is missing, so there is no host row to fall back to. Warp's
        ``"cuda:0"`` spelling is a valid torch device string, so the entry is forwarded verbatim.
        The ``or "cpu"`` is the harmless floor for a non-pytorch3d row that reads this.
        """
        return self.device or "cpu"

    def run(
        self,
        fn: Callable[..., Any],
        *,
        rounds: int = _ROUNDS,
        setup: Callable[[], Any] | None = None,
    ) -> Any:
        """
        Time ``fn`` with pytest-benchmark, synchronising inside the timed region on CUDA.

        Warp launches are asynchronous, so ``wp.synchronize_device`` must sit inside the timed
        callable to capture real GPU compute; CPU targets/references skip it.

        ``rounds`` lowers the repeat count for the handful of groups whose single call runs into
        hundreds of milliseconds (the hole-filling DP, ``split`` on a thousand components, an
        isotropic remesh, a Poisson reconstruction). Ten rounds of those alone would dominate the
        suite's wall clock, and their spread is wide enough that the extra samples buy nothing.

        ``setup`` builds a **fresh input per round, outside the timed region**, and its return value
        is passed to ``fn`` as the single positional argument. Most references that mutate their
        input rebuild it *inside* the timed callable instead (``new_mesh_ml``, ``new_meshset_pml``),
        which is right when the build is a small fraction of the call -- and honest, since a caller
        pays it too. Use ``setup`` where it is not: the MeshLib dilation row's voxel-bitset load
        costs more than the dilation itself, so folding it in would report the load. ``pedantic``
        runs ``setup`` before every round including the warmup, and forbids it above
        ``iterations=1`` -- which is what every row here already uses.

        **The sync is per-library, not per-device.** ``wp.synchronize_device`` synchronizes *Warp's*
        stream and says nothing about torch's, so a ``pytorch3d-cuda`` row synchronized the Warp way
        would time the launch and not the kernel -- the same class of error as section 8's
        launch-device hazard, in that it does not raise and simply reports a number that is far too
        good. That row therefore calls ``torch.cuda.synchronize()`` instead.

        **And a ``pytorch3d-cuda`` row releases torch's cached blocks when it finishes**, outside
        the timed region. torch's ``CUDACachingAllocator`` reserves device memory and returns it
        only on ``torch.cuda.empty_cache()``, so a large row can leave nothing for the Warp
        allocator that runs next and every ``ordito-cuda`` row after it in the module dies with it
        -- and because the failures are all ordito's, the surviving cells *understate* the loss
        table. The release is per-row rather than per-module because the two allocators interleave
        at row granularity. It is also the *whole* fix: the largest pytorch3d rows run uncapped with
        it in place, so the size cap that was going to accompany it was dropped rather than shipped.

        **A ``pytorch3d`` row also opts out of torch's sparse-tensor invariant checks**, before the
        timed region. torch validates nothing by default and warns once per process that it is doing
        so, which is the only warning a benchmark run raises; opting out explicitly silences it
        while leaving the row timing exactly what it timed before. Out rather than in because the
        checks cost a few percent of a sparse construction and ``ops.laplacian`` / ``cot_laplacian``
        / ``norm_laplacian`` / ``mesh_laplacian_smoothing`` all build a COO tensor inside the call
        being timed -- so leaving them on would charge the reference for validation ordito's row
        does not perform, in the one module (``test_laplacian``) whose whole point is a
        like-for-like race between two sparse assemblies. ``tests/conftest.py`` opts *in*, where
        there is no clock to bias and a malformed tensor should raise rather than segfault.
        """
        if self._max_rounds is not None:
            rounds = min(rounds, self._max_rounds)
        device = self.device
        needs_cuda_sync = device is not None and device.startswith("cuda")
        needs_torch_sync = needs_cuda_sync and self.kind == "pytorch3d"
        if self.kind == "pytorch3d":
            # Guarded on the kind so this stays inside the lazy-import discipline the rest of this
            # file keeps: a run with no pytorch3d row never reaches it and never pays the ~4 s
            # torch import. Idempotent, so once per row is fine.
            import torch

            torch.sparse.check_sparse_tensor_invariants.disable()

        def target(*args: Any) -> Any:
            result = fn(*args)
            if needs_torch_sync:
                import torch

                torch.cuda.synchronize()
            elif needs_cuda_sync:
                wp.synchronize_device(device)
            return result

        if setup is None:
            result = self._benchmark.pedantic(
                target, rounds=rounds, warmup_rounds=_WARMUP_ROUNDS, iterations=1
            )
        else:

            def make_arguments() -> tuple[tuple[Any, ...], dict[str, Any]]:
                """Hand ``pedantic`` this round's freshly-built input as ``fn``'s only argument."""
                return (setup(),), {}

            result = self._benchmark.pedantic(
                target,
                setup=make_arguments,
                rounds=rounds,
                warmup_rounds=_WARMUP_ROUNDS,
                iterations=1,
            )

        if needs_torch_sync:
            import torch

            torch.cuda.empty_cache()
        return result


class BenchCase(BenchLibrary):
    """One ``(mesh, library)`` benchmark case: lazily-built inputs plus a timed ``run``."""

    def __init__(
        self,
        mesh_name: str,
        library: str,
        benchmark: BenchmarkFixture,
        max_rounds: int | None = None,
    ) -> None:
        """Bind a mesh name, its library variant and the pytest-benchmark fixture."""
        super().__init__(library, benchmark, max_rounds)
        self.mesh_name = mesh_name
        self.spec: MeshSpec = ALL_MESHES_BY_NAME[mesh_name]

    @property
    def faces_wp(self) -> wp.array[wp.int32]:
        """Flat ``wp.int32`` face buffer on this case's device (ordito only)."""
        assert self.device is not None
        return _faces_wp(self.mesh_name, self.device)

    @property
    def vertices_wp(self) -> wp.array[wp.vec3]:
        """``wp.vec3`` vertex buffer on this case's device (ordito only)."""
        assert self.device is not None
        return _vertices_wp(self.mesh_name, self.device)

    @property
    def vertices_np(self) -> np.ndarray:
        """Float64 ``(n_vertices, 3)`` vertices for the trimesh / igl / scipy references."""
        return _load_numpy(self.mesh_name)[0]

    @property
    def faces_np(self) -> np.ndarray:
        """Int64 ``(n_faces, 3)`` faces for the trimesh / igl references."""
        return _load_numpy(self.mesh_name)[1]

    @property
    def n_vertices(self) -> int:
        """Vertex count (the ``edges_unique`` hash base), from the spec rather than the buffer."""
        return self.spec["n_vertices"]

    @property
    def n_faces(self) -> int:
        """Triangle count, from the spec so it is available without building the mesh."""
        return self.spec["n_faces"]

    @property
    def mean_edge(self) -> float:
        """Mean undirected edge length; the natural scale for remeshing / reconstruction sizing."""
        return _mean_edge(self.mesh_name)

    @property
    def mesh_o3d(self) -> o3d.geometry.TriangleMesh:
        """
        Shared legacy ``open3d`` mesh, built once per mesh.

        Safe for references that either do not touch their input or recompute unconditionally
        (``subdivide_midpoint`` and ``filter_smooth_laplacian`` return new meshes;
        ``compute_vertex_normals`` overwrites but never caches). For a reference that mutates
        *idempotently* — ``remove_duplicated_triangles``, where rounds 2..n would find nothing left
        to do — construct the mesh inside the timed callable instead, the way the trimesh references
        rebuild their ``tm.Trimesh`` (see ``test_repair.py``).
        """
        if self.mesh_name not in _o3d_cache:
            _o3d_cache[self.mesh_name] = _new_mesh_o3d(self.mesh_name)
        return _o3d_cache[self.mesh_name]

    @property
    def mesh_pv(self) -> pv.PolyData:
        """
        Shared ``pyvista.PolyData``, built once per mesh.

        Safe to share, unlike ``meshset_pml``: pyvista caches nothing (a repeat ``cell_quality`` or
        ``decimate`` call recomputes in full) and almost every filter returns a *new* object rather
        than mutating -- ``inplace=False`` is the default
        throughout, so never pass ``inplace=True`` in a row. Two exceptions to build inside the
        timed callable instead: ``edge_mask``, which writes ``point_ind`` into its input, and any
        filter a row calls with ``inplace=True``.
        """
        if self.mesh_name not in _pv_cache:
            _pv_cache[self.mesh_name] = _new_mesh_pv(self.mesh_name)
        return _pv_cache[self.mesh_name]

    @property
    def mesh_p3d(self) -> p3d_structures.Meshes:
        """
        Shared ``pytorch3d.structures.Meshes``, built once per ``(mesh, torch device)``.

        Safe to share, and for a stronger reason than ``mesh_pv``: a ``Meshes`` is **immutable**
        and every ``pytorch3d.ops`` / ``pytorch3d.loss`` entry point is pure, so there is no
        freshness rule here at all -- the opposite end of the scale from ``new_meshset_pml``. What
        it *does* have is a cache of its own derived quantities (``verts_packed``, ``edges_packed``,
        ``faces_packed_to_edges_packed``, ``verts_normals_packed``), memoized on first request. So
        a row must decide which side of that it wants to time: reading an accessor for the first
        time inside the timed callable prices the derivation, and reading it again prices nothing.
        Warm the accessor outside the callable where the row names an operation rather than a
        derivation, and say which in the docstring.
        """
        key = (self.mesh_name, self.torch_device)
        if key not in _p3d_cache:
            _p3d_cache[key] = _new_mesh_p3d(*key)
        return _p3d_cache[key]

    @property
    def cloud_p3d(self) -> Any:
        """
        The mesh's own vertices as a ``pytorch3d.structures.Pointclouds`` on this row's device.

        The container the ``loss`` entry points take, where the ``ops`` ones take the bare batched
        tensor ``points_torch_from_numpy`` builds. Not cached: it is one tensor wrap over an array
        already on the device, and a row that wants it out of the timed region can hoist it itself.
        """
        import pytorch3d.structures as p3d_structures

        return p3d_structures.Pointclouds(
            points=[points_torch_from_numpy(self.vertices_np, self.torch_device)[0]]
        )

    def new_meshset_pml(self) -> ml.MeshSet:
        """
        Build a **fresh** single-mesh ``pymeshlab.MeshSet``; call this *inside* the timed callable.

        Almost every MeshLab filter mutates ``current_mesh()`` in place -- ``apply_coord_*`` moves
        vertices, ``meshing_*`` rewrites the topology, ``compute_*_per_vertex`` writes an attribute,
        and ``generate_*`` pushes a new mesh onto the set -- so a shared MeshSet would have rounds
        2..n measure a filter applied to its own output. That is the trimesh situation rather than
        the open3d one, so the build goes inside the timed region, exactly as ``test_repair.py``'s
        trimesh rows rebuild their ``tm.Trimesh`` and the potpourri3d rows construct their solver.

        Two filters were checked and are *not* idempotent even in geometry, because they default to
        ``autoclean=True`` and delete unreferenced vertices under you:
        ``compute_curvature_principal_directions_per_vertex`` and
        ``meshing_decimation_quadric_edge_collapse``.

        The build is **not free** and it is the floor under every pymeshlab row: it is linear in the
        vertex count and reaches tens of milliseconds on a scan mesh. Any pymeshlab row cheaper than
        that floor is reporting the build and nothing else --
        read it that way rather than as a filter cost. Use ``meshset_pml`` for the narrow set of
        filters verified to leave the geometry alone.
        """
        return _new_meshset_pml(self.mesh_name)

    @property
    def meshset_pml(self) -> ml.MeshSet:
        """
        Return a shared ``pymeshlab.MeshSet``, built once per mesh -- geometry-preserving only.

        Two families qualify, both of which leave positions and topology untouched so that rounds
        2..n do the identical work:

        - the ``compute_scalar_*`` / ``compute_normal_*`` filters, which write one vertex or face
          attribute and read only positions (``compute_scalar_ambient_occlusion``,
          ``compute_scalar_by_volumetric_obscurance``,
          ``compute_scalar_by_shape_diameter_function_per_vertex``,
          ``compute_scalar_by_aspect_ratio_per_face``, ``compute_normal_per_vertex``);
        - the selection filters, which write only the per-element *selected* bit. These are not
          idempotent -- ``apply_selection_dilatation`` grows the set every call -- but their cost is
          independent of how much is selected, because each one is a full pass over the face set
          (measured flat from under a percent of the faces to most of them). So a group may seed
          the selection once, outside the timed callable, and still read a clean per-call cost.

        Because the state persists, a group that *depends* on the selection (the heat-geodesic
        source set, say) must establish it itself rather than inherit what a previous group left.

        Everything else -- including every ``meshing_*``, ``apply_coord_*`` and ``generate_*``
        filter -- must call ``new_meshset_pml`` instead.
        """
        if self.mesh_name not in _pml_cache:
            _pml_cache[self.mesh_name] = _new_meshset_pml(self.mesh_name)
        return _pml_cache[self.mesh_name]

    def new_mesh_ml(self) -> mm.Mesh:
        """
        Build a **fresh** ``meshlib.mrmeshpy.Mesh``; call this *inside* the timed callable.

        A method rather than a cached property, and for the ``new_meshset_pml`` reason: almost every
        MeshLib free function mutates its ``Mesh`` in place and returns something else -- ``relax``,
        ``fillHole``, ``decimateMesh``, ``remesh``, ``subdivideMesh``, ``fixMeshDegeneracies``,
        ``denoiseNormals``, ``expand`` and ``shrink`` all return a status, a count, an ``EdgeId`` or
        a bitset of *new* elements, never the mesh -- so a shared mesh would have rounds 2..n
        measure a filter applied to its own output.

        Two things a row using this must decide explicitly and state in its docstring:

        - **The AABB tree is lazily built, cached on the ``Mesh``, and worth two orders of
          magnitude** between a mesh's first ``findProjection`` and its second.
          A query row should therefore build the mesh *outside* the timed callable and pre-warm it
          with one throwaway query, so the row times the query rather than the tree -- which is
          what ordito's ``wp.Mesh``-in-hand rows already do with their BVH. A row that rebuilds per
          round is timing the build.
        - A mutating call **invalidates** that tree, so the two decisions are not independent.
        """
        return _new_mesh_ml(self.mesh_name)

    def new_tmesh_pmf(self) -> PyTMesh:
        """
        Build a **fresh** ``pymeshfix._meshfix.PyTMesh``; call this *inside* the timed callable.

        A method rather than a cached property, and unlike ``new_mesh_ml`` there is no alternative:
        a ``PyTMesh`` accepts exactly **one** ``load_array`` (a second raises ``RuntimeError``) and
        every one of its nine algorithms mutates in place and returns a status, a count or an array
        rather than the mesh. So a shared object could not even be reloaded, let alone reused.

        The consequence is that **every pymeshfix row prices the load**, and the load is not small:
        on a scan mesh it is comparable with the most expensive operation and far larger than the
        cheap ones. A row whose operation is a small share of that total reports the load and reads
        as pymeshfix being an order of magnitude slower at the operation than it is -- so rows exist
        only above ~30 %, and each states its share. See
        the ``pymeshfix`` paragraph in the LIBRARIES comment block for the full rule.
        """
        return _new_tmesh_pmf(self.mesh_name)


@pytest.fixture
def bench_case(
    benchmark: BenchmarkFixture, mesh_name: str, library: str, request: pytest.FixtureRequest
) -> BenchCase:
    skip_known_slow(request, benchmark.group, mesh_name, library)
    max_rounds = reference_free_rounds(request, benchmark.group, mesh_name, library)
    return BenchCase(mesh_name, library, benchmark, max_rounds)


@pytest.fixture
def bench_lib(
    benchmark: BenchmarkFixture, library: str, request: pytest.FixtureRequest
) -> BenchLibrary:
    """
    Time a benchmark that has no input mesh, e.g. the ``ordito.creation`` generators.

    Requesting this instead of ``bench_case`` parametrizes over libraries alone: ``--device`` still
    selects the ordito targets, but ``--size`` / ``--cpu-max-size`` and the ``benchmeshes`` marker
    have nothing to act on. Size the work with a plain ``pytest.mark.parametrize`` instead.
    """
    skip_known_slow(request, benchmark.group, None, library)
    max_rounds = reference_free_rounds(request, benchmark.group, None, library)
    return BenchLibrary(library, benchmark, max_rounds)
