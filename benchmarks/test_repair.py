"""
Benchmarks for ``ordito.repair``.

Axis: **defect count** for three of the four groups, **diameter** for the fourth. Repair functions
are the clearest case in the package of cost a face count cannot predict — they all "find the broken
things and fix them", so the number of broken things is the driver and the mesh they sit in is
nearly irrelevant. Every defect group runs on ``sphere_med`` and sweeps the *injected defect count*:

* ``resolve_duplicated_faces`` — 0 % against 10 % cancelling flipped pairs.
* ``remove_non_manifold_faces`` — 0 against 1 024 extra faces, each making its three edges
  3-incident. This one loops: removing faces can create *new* non-manifold edges, so a pathological
  input uses all ``max_iter`` rounds where a clean one exits after the first test.
* ``remove_duplicated_vertices`` — on an *unwelded* soup (every face owning its own three vertices,
  the state a freshly loaded STL is in), sweeping ``epsilon``. The two values are two code paths
  rather than two thresholds: positive ``epsilon`` snaps to ``round(v / epsilon)``, while ``0``
  buckets by the high bits of the float32 representation — a *relative* cell about 2.4e-4 wide, not
  an equality test.

``make_winding_consistent`` gets the **diameter** axis instead: its cost is a property of the
face-adjacency graph, not the defect count. A formulation that *propagates* the orientation bits one
level per round is two orders of magnitude slower on the high-diameter mesh than on the compact one
at identical vertex count, where trimesh's equivalent goes the other way; the shipped union-find
form is flat across the axis and faster than trimesh at both ends. Same finding and same fix as
``face_orientation_bits`` in [`test_validation.py`](test_validation.py), the machinery underneath.
pymeshlab settles what that axis was really measuring: its serial face-to-face visit is *faster* on
the high-diameter mesh too. So graph diameter is not intrinsically expensive for this problem; it
was expensive only for the level-propagating formulation.

The two *geometric* defect groups sit on **quality** instead of a defect sweep, because their
defects are not injectable: ``flip_t_vertices`` looks for thin and folded triangles and
``saddle_graded`` already has them by construction, where ``saddle`` gives it nothing to do.

References
----------
**open3d**'s ``remove_duplicated_triangles`` solves the same "deduplicate a face array" problem with
a hash set over index triples against ordito's sort-based grouping. It compares dedup *machinery*,
not results — the semantics differ twice over: open3d keeps one representative of each duplicate
group where ordito applies a signed-count rule that drops cancelling ``(+1, -1)`` pairs outright,
and open3d's hash is **orientation-sensitive**, so on this deliberately-flipped input it removes
nothing and returns the face count unchanged. The hash pass over all ``n`` triples still runs, which
is the cost being compared; the assertion below only checks the count did not grow. Open3D mutates
in place and the operation is idempotent, so its mesh is rebuilt inside the timed callable.
**trimesh**'s ``repair.fix_winding`` is the orientation reference; it has no non-manifold face
removal.

**pymeshlab** is the only library covering *all four* groups, and the first reference of any kind
for ``remove_non_manifold_faces`` — neither trimesh nor open3d nor libigl removes non-manifold faces
at all:

* ``meshing_repair_non_manifold_edges(method='Remove Faces')`` is the same idea, greedier: for each
  non-manifold edge MeshLab iteratively deletes the *smallest-area* incident face until the edge is
  2-manifold, where ordito drops every face on an over-incident edge and re-tests.
* ``meshing_remove_duplicate_faces`` is orientation-*insensitive*, so unlike open3d's hash it does
  see the flipped copies — but it keeps one representative rather than cancelling pairs.
* ``meshing_remove_duplicate_vertices`` and ``meshing_merge_close_vertices(threshold=...)`` are
  exactly ordito's two code paths, so this is the one group where the ``epsilon`` sweep maps across
  libraries one-for-one.
* ``meshing_re_orient_faces_coherently`` is a serial face-to-face visit against ordito's parity
  union-find, precisely the contrast the ``diameter`` axis exposes.

Every one rewrites the topology, so the MeshSet is built inside the timed callable from the
*defect-injected* arrays rather than the registry mesh, and each row carries that build — a large
share at this size, most of a clean ``remove_duplicate_faces`` row and a third of the unwelded
soup's dedup. Subtract it before quoting a ratio.
"""

from __future__ import annotations

from typing import Literal

import igl
import numpy as np
import pymeshlab as ml
import pytest
import pyvista as pv
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
from conftest import BenchCase, mesh_ml_from_numpy, skip_larger_than

_DUP_SEED = 7

# Fraction of faces re-appended as flipped copies: nothing to do, against a tenth of the mesh.
_DUPLICATE_FRACTIONS = [0.0, 0.1]

# Extra same-orientation face copies, each making its three edges 3-incident.
_NON_MANIFOLD_COUNTS = [0, 1_024]

# Vertex merge tolerances: two different code paths, not two thresholds (see the module docstring).
_MERGE_EPSILONS = [0.0, 1e-6]

_defect_np_cache: dict[tuple[str, float], np.ndarray] = {}
_defect_wp_cache: dict[tuple[str, str, float], wp.array[wp.int32]] = {}
_nonmanifold_cache: dict[tuple[str, str, int], wp.array[wp.int32]] = {}
_soup_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], wp.array[wp.int32]]] = {}


def _new_meshset_pml(vertices_np: np.ndarray, faces_np: np.ndarray) -> ml.MeshSet:
    """
    Build a fresh MeshSet over *defect-injected* arrays, inside the timed callable.

    ``BenchCase.new_meshset_pml`` builds the registry mesh, which is exactly the mesh these groups
    are not measuring: every one of them needs the duplicated / non-manifold / unwelded variant.
    """
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(vertices_np, dtype=np.float64),
            np.ascontiguousarray(faces_np, dtype=np.int32),
        )
    )
    return meshset_pml


def _clean_faces_np(bench_case: BenchCase) -> np.ndarray:
    """Return the mesh faces with any pre-existing duplicate or degenerate triangle removed."""
    faces = bench_case.faces_np
    sorted_rows = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(sorted_rows, axis=0, return_inverse=True, return_counts=True)
    base = faces[counts[inverse] == 1]
    nondegenerate = (
        (base[:, 0] != base[:, 1]) & (base[:, 1] != base[:, 2]) & (base[:, 0] != base[:, 2])
    )
    return np.ascontiguousarray(base[nondegenerate], dtype=np.int32)


def _faces_with_duplicates_np(bench_case: BenchCase, fraction: float) -> np.ndarray:
    """Append a fixed-seed subset of the faces back as *flipped* copies: cancelling pairs."""
    key = (bench_case.mesh_name, fraction)
    if key not in _defect_np_cache:
        base = _clean_faces_np(bench_case)
        if fraction <= 0.0:
            _defect_np_cache[key] = base
        else:
            rng = np.random.default_rng(_DUP_SEED)
            n_dup = max(1, int(base.shape[0] * fraction))
            idx = rng.choice(base.shape[0], size=n_dup, replace=False)
            _defect_np_cache[key] = np.ascontiguousarray(
                np.vstack((base, base[idx][:, ::-1])), dtype=np.int32
            )
    return _defect_np_cache[key]


def _faces_with_duplicates_wp(bench_case: BenchCase, fraction: float) -> wp.array[wp.int32]:
    key = (bench_case.mesh_name, str(bench_case.device), fraction)
    if key not in _defect_wp_cache:
        combined = _faces_with_duplicates_np(bench_case, fraction).reshape(-1)
        _defect_wp_cache[key] = wp.array(
            np.ascontiguousarray(combined), dtype=wp.int32, device=bench_case.device
        )
    return _defect_wp_cache[key]


@pytest.mark.noparity(
    "open3d",
    reason="D2 three different dedup rules: ordito applies a signed-count rule that cancels "
    "(+1, -1) pairs outright, while open3d hashes orientation-*sensitively* and keeps one "
    "representative of each duplicate group -- on this deliberately-flipped input it therefore "
    "removes nothing and returns the face count unchanged. The rows compare dedup machinery, not "
    "results.",
)
@pytest.mark.noparity(
    "pymeshlab",
    reason="D2 as above but the other way round: MeshLab's hash is orientation-*insensitive*, so "
    "it does see the flipped copies, yet it still keeps one representative per group where "
    "ordito cancels a (+1, -1) pair to nothing. Same input, three different survivor sets by "
    "design; the numpy oracle in tests/test_repair.py covers the signed-count rule itself.",
)
@pytest.mark.benchmark(group="resolve_duplicated_faces")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito", "open3d", "pymeshlab")
@pytest.mark.parametrize("fraction", _DUPLICATE_FRACTIONS, ids=["clean", "dup10pct"])
def test_resolve_duplicated_faces(bench_case: BenchCase, fraction: float) -> None:
    """Sort-based grouping with a signed-count rule: nothing to do, against a tenth of the mesh."""
    if bench_case.kind == "pymeshlab":
        # Orientation-insensitive, so unlike open3d's hash it does see the flipped copies.
        vertices_np = bench_case.vertices_np
        faces_dup_np = _faces_with_duplicates_np(bench_case, fraction)
        bench_case.run(
            lambda: _new_meshset_pml(vertices_np, faces_dup_np).meshing_remove_duplicate_faces()
        )
        return
    if bench_case.kind == "ordito":
        faces_dup = _faces_with_duplicates_wp(bench_case, fraction)
        resolved, kept = bench_case.run(lambda: od.repair.resolve_duplicated_faces(faces_dup))
        assert resolved.size == kept.size * 3
        assert kept.size > 0
    else:
        # open3d dedups in place and is idempotent: build the mesh inside the timed callable.
        import open3d as o3d

        vertices = o3d.utility.Vector3dVector(bench_case.vertices_np)
        faces_dup_np = _faces_with_duplicates_np(bench_case, fraction)

        def run() -> o3d.geometry.TriangleMesh:
            mesh = o3d.geometry.TriangleMesh(vertices, o3d.utility.Vector3iVector(faces_dup_np))
            return mesh.remove_duplicated_triangles()

        deduped = bench_case.run(run)
        assert 0 < len(deduped.triangles) <= faces_dup_np.shape[0]


def _faces_with_non_manifold_np(bench_case: BenchCase, extra: int) -> np.ndarray:
    """Return ``(n, 3)`` faces with ``extra`` same-orientation copies appended: 3-incident edges."""
    base = _clean_faces_np(bench_case)
    if extra > 0:
        rng = np.random.default_rng(_DUP_SEED)
        idx = rng.choice(base.shape[0], size=min(extra, base.shape[0]), replace=False)
        base = np.vstack((base, base[idx]))
    return np.ascontiguousarray(base, dtype=np.int32)


def _faces_with_non_manifold_wp(bench_case: BenchCase, extra: int) -> wp.array[wp.int32]:
    """Append ``extra`` same-orientation face copies, so their edges become 3-incident."""
    key = (bench_case.mesh_name, str(bench_case.device), extra)
    if key not in _nonmanifold_cache:
        _nonmanifold_cache[key] = wp.array(
            np.ascontiguousarray(
                _faces_with_non_manifold_np(bench_case, extra).reshape(-1), dtype=np.int32
            ),
            dtype=wp.int32,
            device=bench_case.device,
        )
    return _nonmanifold_cache[key]


@pytest.mark.benchmark(group="make_solid")
@pytest.mark.benchmeshes("bunny_decimated", "bunny")
@pytest.mark.benchlibs("ordito", "pymeshfix")
def test_make_solid(bench_case: BenchCase) -> None:
    """
    The whole pipeline: broken scan in, single watertight solid out.

    The one group in this file whose input is a real scan rather than an injected defect, because
    the whole point of the composite is that the *distribution* of defects drives it -- 86 boundary
    loops and 1 084 self-intersecting faces on ``bunny_decimated``, 5 loops and none on ``bunny``,
    so the two meshes exercise different stages of the same call and the pair is the measurement.
    Capped at ``bunny``: ``dragon`` is seconds of pymeshfix load plus seconds of query per round,
    and the composite runs the self-intersection loop over all of it.

    ``clean_from_arrays`` is pymeshfix's headline and this is the one group where its row is timed
    rather than declared, because the operation clears the load by a wide margin -- around 70 % of
    the round on both meshes. The load still cannot leave the timed callable -- a ``PyTMesh`` takes
    exactly one ``load_array`` -- so read the row as pipeline-plus-load and subtract accordingly.

    ordito wins the group by an order of magnitude and the gap widens with the mesh, because the
    composite's per-stage cost is a fixed chain of wrapper calls plus device passes where the
    reference is sequential C++ throughout. Read it knowing what
    dominates ordito's side, which is **not** kernel time: a dozen wrapper chains inside a
    convergence loop, each a handful of launches. Anything spent optimizing this belongs in the
    refill chain and the self-intersection loop, exactly as the ``fix_self_intersections`` group's
    own note says; a faster kernel would not move it.

    Both rows assert a watertight one-component answer rather than a count, since the two libraries
    sacrifice different amounts of surface around a defect. On these two meshes they agree
    closely -- ``bunny_decimated`` comes back at **8 188 v / 16 372 f from both**, watertight with
    chi = 2 -- and ``tests/test_repair.py`` measures where they agree exactly (3.11e-08 on
    interpenetrating shells) and where they do not (a self-intersecting torus, 0.400).
    """
    if bench_case.kind == "pymeshfix":
        from pymeshfix import _meshfix

        vertices_np = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64)
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)
        vertices_pmf, faces_pmf = bench_case.run(
            lambda: _meshfix.clean_from_arrays(vertices_np, faces_np), rounds=3
        )
        assert tm.Trimesh(vertices_pmf, faces_pmf, process=False).is_watertight
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    solid_vertices, solid_faces = bench_case.run(
        lambda: od.repair.make_solid(vertices, faces), rounds=3
    )
    assert od.validation.is_watertight(solid_vertices, solid_faces)


@pytest.mark.benchmark(group="remove_small_components")
@pytest.mark.benchaxis("components")
@pytest.mark.benchlibs("ordito", "pymeshlab", "open3d")
def test_remove_small_components(bench_case: BenchCase) -> None:
    """
    Label the components, threshold them, re-extract: the first step of every repair pipeline.

    The axis is the component count -- 1, 64 then 1 024 at a fixed 81 920 faces -- which is this
    function's only cost driver: the labelling is a connected-components pass over the face
    adjacency and the extraction is one compaction, and neither cares how the faces are distributed
    between components. The mesh size is pinned across the three points so the row reads as the
    component count alone.

    **The threshold keeps everything**, deliberately: ``min_faces=2`` passes every component of
    every one of these meshes. A threshold that dropped components would make the extraction
    cheaper on exactly the meshes that have more of them, so the axis would be measuring the
    selection rather than the labelling, and the three points would stop being comparable. All the
    cost is in the labelling and the compaction, and this is what prices them.

    pymeshlab's ``meshing_remove_connected_component_by_face_number`` is the same operation with
    the same inclusive bound (``tests/test_repair.py`` compares the answers at 80 and 81 on a
    fixture built to straddle it); it mutates ``current_mesh()``, so the MeshSet is built inside the
    timed callable and the row carries the per-vertex build. open3d has no filter -- its
    ``cluster_connected_triangles`` returns the per-triangle cluster id plus each cluster's triangle
    count and area, and the mask and the removal are the caller's, which is what the row times; it
    also mutates, so its mesh is rebuilt per round too.

    pymeshfix is **not** a row here: ``remove_smallest_components`` is a tenth or so of a round
    behind a load that cannot be hoisted out of it, so the number would be the load. Its rule is
    nonetheless what
    ``keep_largest`` defaults to, and ``tests/test_repair.py`` pins that.
    """
    min_faces = 2
    if bench_case.kind == "pymeshlab":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def run_pml() -> ml.MeshSet:
            meshset_pml = _new_meshset_pml(vertices_np, faces_np)
            meshset_pml.meshing_remove_connected_component_by_face_number(
                mincomponentsize=min_faces, removeunref=True
            )
            return meshset_pml

        kept_pml = bench_case.run(run_pml)
        assert kept_pml.current_mesh().face_number() == bench_case.n_faces
        return
    if bench_case.kind == "open3d":
        import open3d as o3d

        vertices_o3d = o3d.utility.Vector3dVector(bench_case.vertices_np)
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)

        def run_o3d() -> o3d.geometry.TriangleMesh:
            mesh_o3d = o3d.geometry.TriangleMesh(vertices_o3d, o3d.utility.Vector3iVector(faces_np))
            clusters_o3d, counts_o3d, _areas_o3d = mesh_o3d.cluster_connected_triangles()
            small_np = np.asarray(counts_o3d)[np.asarray(clusters_o3d)] < min_faces
            # pybind11 takes the bool array as the sequence its stub spells.
            mesh_o3d.remove_triangles_by_mask(small_np)  # pyright: ignore[reportArgumentType]
            mesh_o3d.remove_unreferenced_vertices()
            return mesh_o3d

        kept_o3d = bench_case.run(run_o3d)
        assert len(kept_o3d.triangles) == bench_case.n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _kept_vertices, kept_faces = bench_case.run(
        lambda: od.repair.remove_small_components(vertices, faces, min_faces=min_faces)
    )
    assert kept_faces.size // 3 == bench_case.n_faces


@pytest.mark.noparity(
    "pymeshlab",
    reason="D2 the same idea, greedier: for each non-manifold edge MeshLab iteratively deletes the "
    "smallest-area incident face until that edge is 2-manifold, where ordito drops every face on "
    "an over-incident edge and re-tests. Both leave an edge-manifold mesh but they delete "
    "different faces and different numbers of them, so only the post-condition is shared -- and "
    "tests/test_repair.py now asserts that post-condition through igl and open3d rather than "
    "through ordito's own detector, which is the half that was missing.",
)
@pytest.mark.benchmark(group="remove_non_manifold_faces")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito", "pymeshlab")
@pytest.mark.parametrize("extra", _NON_MANIFOLD_COUNTS, ids=["clean", "nm1024"])
def test_remove_non_manifold_faces(bench_case: BenchCase, extra: int) -> None:
    """
    Iterated edge-sort and manifold test: a clean mesh exits in one round, a broken one loops.

    **open3d's ``remove_non_manifold_edges`` is deliberately not a second row.** It uses the same
    greedier rule MeshLab does -- on a mesh carrying one extra face on an existing edge it deletes
    that **one** face where ordito drops all three on that edge. Both land edge-manifold, so only
    the post-condition is shared, and a second
    incomparable timing row would say nothing the exemption above does not. What open3d *does*
    supply is that post-condition: it and igl both flip False -> True with ordito on every input in
    ``tests/test_repair.py``, which is where this group's real coverage now sits -- before that, the
    contract was asserted with ordito's own ``is_edge_manifold``.
    """
    if bench_case.kind == "pymeshlab":
        # MeshLab deletes the smallest-area incident face per non-manifold edge until the edge is
        # 2-manifold; ordito drops every face on an over-incident edge and re-tests. Same goal,
        # a greedier rule, and the first reference this group has had.
        vertices_np = bench_case.vertices_np
        faces_nm_np = _faces_with_non_manifold_np(bench_case, extra)
        bench_case.run(
            lambda: _new_meshset_pml(vertices_np, faces_nm_np).meshing_repair_non_manifold_edges(
                method="Remove Faces"
            )
        )
        return
    vertices = bench_case.vertices_wp
    faces = _faces_with_non_manifold_wp(bench_case, extra)
    _kept_vertices, kept_faces = bench_case.run(
        lambda: od.repair.remove_non_manifold_faces(vertices, faces)
    )
    assert kept_faces.size > 0


@pytest.mark.benchmark(group="split_non_manifold_vertices")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito", "igl", "meshlib")
@pytest.mark.parametrize("extra", _NON_MANIFOLD_COUNTS, ids=["clean", "nm1024"])
def test_split_nonmanifold(bench_case: BenchCase, extra: int) -> None:
    """
    Vertex duplication instead of face deletion: the non-lossy repair for the same defect.

    Read against ``remove_non_manifold_faces`` above, which is timed on the identical input: the
    two reach an edge-manifold mesh from opposite directions, one by dropping every face on an
    over-incident edge and the other by splitting vertices apart and keeping all of them. The
    ``clean`` case is the floor -- both must detect that there is nothing to do -- and the
    ``nm1024`` case is the work.

    ordito's cost is an edge sort plus a connected-components pass over ``3 * n_faces`` corner
    nodes, so it barely moves between the two cases. ``igl.split_nonmanifold`` is sequential by
    construction -- it explodes the mesh to ``3 * n_faces`` singleton vertices and greedily re-
    merges pairs, re-testing manifoldness after each candidate, with the source calling its own
    inner check "Omega(m) and probably O(m log m) or worse" -- and it does move, measurably, between
    the clean and the duplicated input. It takes ``rounds=3``.

    The two libraries agree on the split exactly for a bowtie vertex, a same-wound fan of three
    faces on one edge, a flipped face and a boundary, but **not on this group's defect**: for a
    duplicated face igl keeps one arbitrarily chosen pair joined where ordito splits all copies
    (18 vertices against 15 on an icosahedron with one face duplicated). Both outputs are
    manifold with every face kept; the rows are a cost comparison, and ``tests/test_repair.py``
    carries both the agreement and the divergence.
    """
    if bench_case.kind == "meshlib":
        # The build is inside the timed callable and that is not a converter tax: MeshLib's
        # half-edge topology cannot represent a non-edge-manifold mesh at all, so
        # ``meshFromFacesVerts`` does this group's split while *constructing* -- on the injected
        # defect, which is edge-incidence, ``duplicateMultiHoleVertices`` then reports 0 and the
        # only honest row is both calls together. The ``tests/test_repair.py`` pair shows the two
        # routes reaching the same vertex count.
        vertices_np = bench_case.vertices_np
        faces_nm_np = _faces_with_non_manifold_np(bench_case, extra)

        def split_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_nm_np)
            mm.duplicateMultiHoleVertices(mesh_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(split_ml, rounds=3) >= bench_case.n_vertices
        return
    if bench_case.kind == "igl":
        faces_nm_np = np.ascontiguousarray(
            _faces_with_non_manifold_np(bench_case, extra), dtype=np.int64
        )
        faces_igl, source_igl = bench_case.run(lambda: igl.split_nonmanifold(faces_nm_np), rounds=3)
        assert faces_igl.shape[0] == faces_nm_np.shape[0]
        assert source_igl.shape[0] >= bench_case.n_vertices
        return
    vertices = bench_case.vertices_wp
    faces = _faces_with_non_manifold_wp(bench_case, extra)
    split_vertices, split_faces, _source = bench_case.run(
        lambda: od.repair.split_non_manifold_vertices(vertices, faces)
    )
    assert split_faces.size == faces.size
    assert split_vertices.size >= bench_case.n_vertices


@pytest.mark.benchmark(group="remove_unreferenced_vertices")
@pytest.mark.benchlibs("ordito", "igl", "open3d", "meshlib")
@pytest.mark.parametrize("unreferenced", [0, 1], ids=["clean", "padded"])
def test_remove_unreferenced_vertices(bench_case: BenchCase, unreferenced: int) -> None:
    """
    Compact away vertices no face indexes: a mask, a scan and a gather.

    The ``padded`` id appends a *copy* of the vertex buffer that no face references, so half the
    vertices are dead and the compaction has real work to do; ``clean`` is the identity case, where
    the whole cost is the mask-and-scan that proves there is nothing to remove. Both are on the scan
    sweep because the operation has no other axis: it is one pass over ``V`` plus one over ``3F``.

    ``igl.remove_unreferenced`` returns the same four things in the same order (vertices, faces,
    forward map, inverse map) and is already the oracle in ``tests/test_repair.py``. Note it takes
    ``int32`` faces here while most of the package wants ``int64`` -- passing ``faces_np`` directly
    works because nanobind casts, but the cast is a copy of the face buffer on every call and it is
    inside the timing.

    ``bunny`` itself carries **1 113 unreferenced vertices** of 35 947, so the ``clean`` id is only
    clean in the sense of "nothing added": all the libraries really do drop those.

    open3d's ``remove_unreferenced_vertices`` mutates its mesh in place, so the legacy container is
    rebuilt inside the timed callable (the ``test_repair.py`` rule) -- its row prices the pybind
    ``Vector3dVector`` copy along with the compaction, which is what a caller holding NumPy buffers
    pays. It agrees with ordito element-wise on both compacted buffers
    (``tests/test_repair.py::test_remove_unreferenced_matches_open3d``).
    """
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    if unreferenced:
        vertices_np = np.ascontiguousarray(np.vstack([vertices_np, vertices_np]))
    if bench_case.kind == "meshlib":
        # ``pack()`` renumbers in place, so the mesh is built inside the timed callable and this
        # row prices the build with it. The ``padded`` id costs meshlib nothing extra either way:
        # the duplicated block is *trailing*, and ``meshFromFacesVerts`` sizes its point buffer by
        # ``F.max() + 1``, so those vertices never reach the mesh. What it does price is the case
        # the other three rows cannot show -- bunny's 1 113 *interior* unreferenced vertices, which
        # arrive as invalid entries and are what pack() compacts away.
        def pack_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            mesh_ml.pack()
            return mesh_ml.topology.numValidVerts()

        # Not bounded above by the input count: on a non-edge-manifold mesh the builder splits
        # vertices apart before pack() ever runs (8 360 from bunny_decimated's 8 171).
        assert bench_case.run(pack_ml) > 0
        return
    if bench_case.kind == "open3d":
        skip_larger_than(bench_case, "bunny", "the container rebuild dominates past bunny")
        import open3d as o3d

        faces_i32 = np.ascontiguousarray(faces_np, dtype=np.int32)

        def remove_unreferenced_o3d() -> int:
            mesh_o3d = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices_np), o3d.utility.Vector3iVector(faces_i32)
            )
            mesh_o3d.remove_unreferenced_vertices()
            return len(mesh_o3d.vertices)

        assert bench_case.run(remove_unreferenced_o3d) <= vertices_np.shape[0]
        return
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "the reference is a single-threaded scan and gather")
        faces_igl = np.ascontiguousarray(faces_np, dtype=np.int32)
        kept_vertices_igl, kept_faces_igl, _remap_igl, _inverse_igl = bench_case.run(
            lambda: igl.remove_unreferenced(vertices_np, faces_igl)
        )
        assert kept_faces_igl.shape == (bench_case.n_faces, 3)
        assert kept_vertices_igl.shape[0] <= vertices_np.shape[0]
        return
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
    )
    faces_wp = bench_case.faces_wp
    kept_vertices, kept_faces, _remap = bench_case.run(
        lambda: od.repair.remove_unreferenced_vertices(vertices_wp, faces_wp)
    )
    assert int(kept_faces.shape[0]) == faces_wp.size
    assert int(kept_vertices.shape[0]) <= vertices_np.shape[0]


def _soup(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Unweld the mesh so every face owns its three vertices: maximal duplication to collapse."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _soup_cache:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        soup_np = vertices_np[faces_np].reshape(-1, 3)
        _soup_cache[key] = (
            wp.array(
                np.ascontiguousarray(soup_np, dtype=np.float32),
                dtype=wp.vec3,
                device=bench_case.device,
            ),
            wp.array(
                np.arange(soup_np.shape[0], dtype=np.int32),
                dtype=wp.int32,
                device=bench_case.device,
            ),
        )
    return _soup_cache[key]


@pytest.mark.benchmark(group="remove_duplicated_vertices")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito", "open3d", "pymeshlab", "pyvista", "meshlib")
@pytest.mark.parametrize("epsilon", _MERGE_EPSILONS, ids=["exact", "eps1e-6"])
def test_remove_duplicated_vertices(bench_case: BenchCase, epsilon: float) -> None:
    """
    Weld an unwelded soup: 245 760 positions down to 40 962, by two different bucketings.

    VTK's ``clean`` takes both paths through one call -- ``tolerance`` with ``absolute=True`` is the
    epsilon form and ``0.0`` the exact one -- and it additionally drops degenerate cells and unused
    points, so its row does a little more than the weld. The soup is rebuilt inside the timed
    callable because ``clean`` returns a new mesh from an input this benchmark does not otherwise
    hold as a ``PolyData``.
    """
    if bench_case.kind == "meshlib":
        # ``uniteCloseVertices`` welds in place and returns the merge count, so the soup mesh is
        # rebuilt inside the timed callable. ``uniteOnlyBd=False`` is the setting that matches
        # ordito; MeshLib's default of ``True`` would weld only boundary vertices. Its single
        # parameter is a distance, so the exact id is that distance at zero rather than a second
        # code path -- unlike pymeshlab and VTK, whose two ids are two different calls.
        soup_np = np.ascontiguousarray(bench_case.vertices_np[bench_case.faces_np].reshape(-1, 3))
        faces_np = np.ascontiguousarray(np.arange(soup_np.shape[0], dtype=np.int32).reshape(-1, 3))

        def weld_ml() -> int:
            return mm.uniteCloseVertices(mesh_ml_from_numpy(soup_np, faces_np), epsilon, False)

        assert 0 <= bench_case.run(weld_ml) < soup_np.shape[0]
        return
    if bench_case.kind == "pyvista":
        soup_np = np.ascontiguousarray(bench_case.vertices_np[bench_case.faces_np].reshape(-1, 3))
        faces_np = np.ascontiguousarray(np.arange(soup_np.shape[0], dtype=np.int32).reshape(-1, 3))
        soup_pv = pv.PolyData.from_regular_faces(soup_np, faces_np)
        welded_pv = bench_case.run(
            lambda: soup_pv.clean(point_merging=True, tolerance=epsilon, absolute=True)
        )
        assert 0 < welded_pv.n_points <= soup_np.shape[0]
        return
    if bench_case.kind == "pymeshlab":
        # The one group whose epsilon sweep maps one-for-one: MeshLab has a filter per path.
        soup_np = np.ascontiguousarray(bench_case.vertices_np[bench_case.faces_np].reshape(-1, 3))
        faces_np = np.ascontiguousarray(np.arange(soup_np.shape[0], dtype=np.int32).reshape(-1, 3))

        def weld_pml() -> None:
            meshset_pml = _new_meshset_pml(soup_np, faces_np)
            if epsilon > 0.0:
                meshset_pml.meshing_merge_close_vertices(threshold=ml.PureValue(epsilon))
            else:
                meshset_pml.meshing_remove_duplicate_vertices()

        bench_case.run(weld_pml)
        return
    if bench_case.kind == "ordito":
        vertices, faces = _soup(bench_case)
        unique_vertices, _unique_indices, _inverse, _faces = bench_case.run(
            lambda: od.repair.remove_duplicated_vertices(vertices, faces, epsilon)
        )
        assert unique_vertices.size <= vertices.size
    else:
        import open3d as o3d

        soup_np = np.ascontiguousarray(bench_case.vertices_np[bench_case.faces_np].reshape(-1, 3))
        faces_np = np.ascontiguousarray(np.arange(soup_np.shape[0], dtype=np.int32).reshape(-1, 3))

        def run() -> o3d.geometry.TriangleMesh:
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(soup_np), o3d.utility.Vector3iVector(faces_np)
            )
            return mesh.remove_duplicated_vertices()

        welded = bench_case.run(run)
        assert 0 < len(welded.vertices) <= soup_np.shape[0]


@pytest.mark.benchmark(group="reverse_winding")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_reverse_winding(bench_case: BenchCase) -> None:
    """
    One unconditional corner swap per face: the floor every orientation row is measured against.

    No axis beyond the scan sweep -- there is no traversal, no predicate and no data-dependent
    branch, so this is the cost of touching the index buffer once and nothing else. Read
    ``make_winding_consistent`` and ``make_normals_outward`` against it: whatever they cost above
    this row is the traversal, since all three write the same buffer.

    trimesh's ``invert`` is an ``np.fliplr`` **plus** a cache invalidation and a normal flip on
    whatever it had cached, so it is an upper bound; it also mutates, hence a fresh mesh per round.
    """
    if bench_case.kind == "trimesh":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def invert_tm() -> tm.Trimesh:
            mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
            mesh_tm.invert()
            return mesh_tm

        inverted_tm = bench_case.run(invert_tm)
        assert len(inverted_tm.faces) == bench_case.n_faces
        return
    faces_wp = bench_case.faces_wp
    reversed_wp = bench_case.run(lambda: od.repair.reverse_winding(faces_wp))
    assert reversed_wp.shape == faces_wp.shape


@pytest.mark.benchmark(group="make_winding_consistent")
@pytest.mark.benchaxis("diameter")
@pytest.mark.benchlibs("ordito", "trimesh", "igl", "pymeshlab", "meshlib")
def test_make_winding_consistent(bench_case: BenchCase) -> None:
    """
    Flip mask from the parity union-find, then one relabel pass: flat across the axis.

    Three references, three traversals, and this is the group where the ``diameter`` axis earns its
    keep: ``igl.bfs_orient`` is a breadth-first walk, MeshLab's
    ``meshing_re_orient_faces_coherently`` a serial face-to-face visit and ``trimesh.repair
    .fix_winding`` a flood fill, so all three grow with graph depth where the union-find does not.

    ``igl.bfs_orient`` returns ``(FF, C)`` -- the reoriented faces and the per-face *component id*,
    not the flip mask; the mask is recovered by comparing ``FF`` to ``F`` (see
    ``tests/test_validation.py``). Unlike the other two it does not mutate an input, so nothing is
    rebuilt inside the callable.
    """
    if bench_case.kind == "meshlib":
        # A fourth traversal, and the only one that also decides the *global* sign: it ray-casts to
        # find which side is outside, so it does strictly more than the flip mask and is expected
        # to cost more. It returns a bitset without touching the mesh, so one mesh serves every
        # round -- but the ray casting builds an AABB tree lazily, so it is pre-warmed outside the
        # timed callable and this row prices the traversal rather than the tree.
        mesh_ml = bench_case.new_mesh_ml()
        mm.findDisorientedFaces(mesh_ml)
        disoriented_ml = bench_case.run(lambda: mm.findDisorientedFaces(mesh_ml))
        assert disoriented_ml.size() <= bench_case.n_faces
        return
    if bench_case.kind == "igl":
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        oriented_igl, _components_igl = bench_case.run(lambda: igl.bfs_orient(faces_np))
        assert oriented_igl.shape == (bench_case.n_faces, 3)
        return
    if bench_case.kind == "pymeshlab":  # a serial face-to-face visit, against the union-find
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        bench_case.run(
            lambda: _new_meshset_pml(vertices_np, faces_np).meshing_re_orient_faces_coherently()
        )
        return
    if bench_case.kind == "ordito":
        faces, n_vertices = bench_case.faces_wp, bench_case.vertices_wp.size
        # The vertex count every reference is handed with the faces; it narrows the key sort.
        oriented = bench_case.run(
            lambda: od.repair.make_winding_consistent(faces, n_vertices=n_vertices)
        )
        assert oriented.size == faces.size
    else:  # trimesh mutates in place: rebuild inside the timed callable
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def fix_winding_tm() -> tm.Trimesh:
            mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
            tm.repair.fix_winding(mesh_tm)
            return mesh_tm

        assert len(bench_case.run(fix_winding_tm).faces) == bench_case.n_faces


@pytest.mark.benchmark(group="make_volume")
@pytest.mark.benchlibs("ordito", "trimesh", "pyvista")
def test_make_volume(bench_case: BenchCase) -> None:
    """
    Orient the whole surface outward: one watertightness predicate, one reduction, one flip pass.

    ``trimesh.repair.fix_inversion`` is the same operation and the same decision -- is the mesh
    closed, and is its signed volume negative -- so the two are directly comparable. It mutates the
    mesh in place and caches the volume on it, so the ``tm.Trimesh`` is rebuilt inside the timed
    callable as the other trimesh rows here do.

    pyvista's ``compute_normals(consistent_normals=True, auto_orient_normals=True)`` is the one
    other library that performs *this* operation, and the distinction is worth stating precisely
    because two libraries have an obviously-named filter that does something else. Probed on two
    inputs -- a *locally* inconsistent mesh and a consistently **inward** one -- only pyvista
    recovers a positive signed volume from both. ``orient_triangles`` and
    ``meshing_re_orient_faces_coherently`` make the winding *coherent*; on a locally inconsistent
    mesh that recovers the majority orientation and looks like this operation, and on a
    consistently **inward** mesh -- the state ``make_volume`` exists for -- open3d leaves it inward
    and pymeshlab always does. So both belong to ``make_winding_consistent`` and only pyvista is a
    row here -- note that a comparison probed on the inconsistent input alone reads all three as
    agreeing.

    pyvista is not identical either, on an input class the scan meshes do not contain: on a
    **multi-shell** mesh it turns each shell outward *from itself*, so a cavity's contribution adds
    where ordito's subtracts. Every registry mesh here is a single open shell, so the row is
    unaffected; the
    divergence is pinned in ``tests/test_repair.py``.

    The scan meshes are open, so every side takes the "not watertight, return unchanged" path and
    what this row prices is the *test*, which is the point: the watertightness check dominates a
    call that would otherwise be one reduction and one relabel.
    """
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        oriented_pv = bench_case.run(
            lambda: mesh_pv.compute_normals(consistent_normals=True, auto_orient_normals=True)
        )
        assert oriented_pv.n_faces == bench_case.n_faces
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        oriented = bench_case.run(lambda: od.repair.make_volume(vertices, faces))
        assert int(oriented.shape[0]) == faces.size
    else:  # trimesh mutates in place and caches the volume: rebuild inside the timed callable
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def fix_inversion_tm() -> tm.Trimesh:
            mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
            tm.repair.fix_inversion(mesh_tm)
            return mesh_tm

        assert len(bench_case.run(fix_inversion_tm).faces) == bench_case.n_faces


@pytest.mark.benchmark(group="flip_t_vertices")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "pymeshlab")
def test_remove_t_vertices(bench_case: BenchCase) -> None:
    """A flip loop over the slivers: no work on ``saddle``, real work on ``saddle_graded``."""
    if bench_case.kind == "pymeshlab":
        # Rewrites the topology, so the MeshSet is rebuilt inside the timed callable.
        # ``repeat=True`` is MeshLab's own iterate-to-convergence, which is what ordito's
        # ``max_iter`` passes are.
        new_meshset_pml = bench_case.new_meshset_pml

        def repair_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.meshing_remove_t_vertices(method="Edge Flip", threshold=40.0, repeat=True)
            return meshset_pml.current_mesh().face_number()

        assert bench_case.run(repair_pml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    flipped = bench_case.run(lambda: od.repair.flip_t_vertices(vertices, faces), rounds=3)
    assert flipped.size == faces.size


@pytest.mark.benchmark(group="remove_degenerate_faces")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "meshlib", "trimesh", "pymeshlab")
def test_remove_degenerate_faces(bench_case: BenchCase) -> None:
    """
    Find the zero-area triangles and compact them away: an altitude test, a scan and a gather.

    On the ``quality`` axis rather than a defect sweep for the reason ``face_defective_mask`` is: a
    degenerate face cannot be injected without changing what the other rows measure, and
    ``saddle_graded``'s worst aspect ratio of 4 719 is the closest a registry mesh comes to one.
    Both sides should be flat across the axis -- neither's cost depends on how many it finds -- and
    a row that is not is the finding.

    Three references, and the split between them is *detect* against *rebuild* -- which is what the
    rows have to be read through, because ordito does both:

    * **meshlib** ``findDegenerateFaces`` and **trimesh** ``Trimesh.nondegenerate_faces`` stop at
      the face set, so they do strictly less. MeshLib's ``criticalAspectRatio`` is left at its
      ``FLT_MAX`` default, the setting under which its criterion is ordito's; trimesh's ``height``
      is left at its ``1e-08`` default, the same order as ordito's own ``TOLERANCE_ZERO``, which
      ordito does not expose as a parameter.
    * **pymeshlab** ``meshing_remove_null_faces`` rebuilds, which is ordito's whole operation. It
      mutates, so its MeshSet is rebuilt per round.

    All three find the same faces (``tests/test_repair.py``).

    **open3d is deliberately absent, and the reason is a criterion difference rather than a cost.**
    ``remove_degenerate_triangles`` removes triangles that *reference a vertex twice*, not
    triangles of zero area: on an exactly collinear face it removes **nothing** where trimesh flags
    it and pymeshlab
    drops it. On a repeated-index face all three agree, which is why an injected-degeneracy probe
    using ``[0, 0, 1]`` reads as agreement and hides this. Timing it here would price a
    strictly narrower predicate under this group's name.
    """
    if bench_case.kind == "trimesh":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
        kept_tm = bench_case.run(lambda: mesh_tm.nondegenerate_faces(height=1e-8))
        assert np.asarray(kept_tm).shape[0] == bench_case.n_faces
        return
    if bench_case.kind == "pymeshlab":
        meshset_pml = bench_case.new_meshset_pml
        bench_case.run(lambda: meshset_pml().meshing_remove_null_faces())
        return
    if bench_case.kind == "meshlib":
        mesh_part_ml = mm.MeshPart(bench_case.new_mesh_ml())
        degenerate_ml = bench_case.run(lambda: mm.findDegenerateFaces(mesh_part_ml))
        assert degenerate_ml.size() <= bench_case.n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    kept_vertices, kept_faces = bench_case.run(
        lambda: od.repair.remove_degenerate_faces(vertices, faces)
    )
    assert kept_faces.size <= faces.size
    assert kept_vertices.size <= bench_case.n_vertices


@pytest.mark.benchmark(group="fix_self_intersections")
@pytest.mark.benchaxis("tangle")
@pytest.mark.benchlibs("ordito", "meshlib")
@pytest.mark.parametrize("method", ["local", "voxel"])
def test_fix_self_intersections(bench_case: BenchCase, method: Literal["local", "voxel"]) -> None:
    """
    Repair a genuine self-intersection, by cutting-and-refilling or by rebuilding.

    The two methods are different costs of different kinds and the parametrize keeps them
    attributable. ``local`` is a detect-dilate-delete-refill loop whose cost is the *damage*: the
    detector runs on the whole mesh but the DP runs on the rims, so it tracks the intersecting band
    rather than the face count. ``voxel`` is a signed distance field plus a marching pass, so its
    cost is the *lattice* and it does not care what was wrong -- visible in its output, which is the
    same face count from a 20x-larger input.

    Why the ``tangle`` axis and not two welded spheres
    --------------------------------------------------
    This group ran for four rounds on ``sphere_med`` concatenated with a shifted copy of itself, and
    **the meshlib ``local`` cell was timing a no-op**: that construction is two components, and
    ``mm.localFixSelfIntersections`` returns a multi-component mesh unchanged -- byte-identical
    buffers, every colliding face intact, at every configuration probed.
    The row read as this suite's largest single loss against a call that
    returned its argument.

    ``tangle_torus`` is a self-intersecting **single** component, so both libraries do real work and
    the comparison is like-for-like for the first time. It is also a *size* axis rather than one
    point, because this is a crossover and a single row would report whichever side of it the mesh
    landed on: the serial C++ fixer leads by several-fold at the small end and is level at the large
    one, while the voxel path wins throughout.

    **Read the ``local`` parity at the large end with its quality caveat, which runs the other
    way.** On this axis' own inputs, with ordito's detector applied to both outputs: at the small
    end both clear every intersection, but at the large one ordito leaves a residue where MeshLib
    reaches zero. So the large cell is a tie for a slightly *less* complete repair -- and the two
    are still different algorithms (MeshLib subdivides the affected band and relaxes it, growing the
    face count; this cuts the band out and refills the rim, shrinking it). ``max_iter`` is what
    closes ordito's residue, and the function's Notes carry that table.

    The post-condition is what
    ``tests/test_repair.py::test_fix_self_intersections_local_clears_them`` claims, and it uses
    MeshLib as a **detector** rather than as a fixer -- the sound way to consult it here, since on
    that test's own fixture the MeshLib *fixer* doubles the intersecting face count.
    """
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
    voxel = diagonal / 128.0
    n_faces_in = faces_np.shape[0]

    if bench_case.kind == "meshlib":

        def fix_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            if method == "voxel":
                mm.fixSelfIntersections(mesh_ml, voxel)
            else:
                mm.localFixSelfIntersections(mesh_ml, mm.SelfIntersections.Settings())
            mesh_ml.pack()
            return mesh_ml.topology.numValidFaces()

        # Assert the mutator actually mutated, which is the standing rule for
        # MeshLib and the guard whose absence let this group time a no-op for four rounds:
        # ``localFixSelfIntersections`` returns normally on an input it declines, and
        # ``numValidFaces() > 0`` passes that. Both methods change the face count here -- the local
        # one subdivides the band, the voxel one remeshes the surface -- so inequality is the
        # assertion, and a wheel that starts declining this input fails rather than drifting.
        n_faces_ml = bench_case.run(fix_ml, rounds=3)
        assert n_faces_ml > 0
        assert n_faces_ml != n_faces_in
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    fixed_vertices, fixed_faces = bench_case.run(
        lambda: od.repair.fix_self_intersections(vertices, faces, method=method), rounds=3
    )
    assert fixed_faces.size > 0
    assert fixed_vertices.size > 0
    assert fixed_faces.size // 3 != n_faces_in


@pytest.mark.benchmark(group="collapse_small_triangles")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_collapse_small_triangles(bench_case: BenchCase) -> None:
    """
    Collapse the sub-threshold triangles to a fixpoint: the one repair group that iterates.

    The ``quality`` axis is the axis this group is *about*: ``saddle`` has nothing under the
    threshold and exits after one pass, while ``saddle_graded``'s fine end gives the loop rounds of
    real work, so the pair separates the detection cost from the collapsing cost. Both rows use the
    same threshold, expressed in each library's own parameter -- ordito's ``epsilon`` is a relative
    *area* against the squared bounding-box diagonal and MeshLib's ``tinyEdgeLength`` an absolute
    *length* -- so this is a cost comparison at a matched scale rather than at a matched knob.

    meshlib is the only bound reference for this function at all: libigl does not export
    ``collapse_small_triangles`` despite shipping the header. ``resolveMeshDegenerations`` mutates
    and invalidates the mesh's tree, so its mesh is rebuilt inside the timed callable and this row
    carries the build. Both loop, so both take ``rounds=3``.
    """
    diagonal = float(
        np.linalg.norm(bench_case.vertices_np.max(axis=0) - bench_case.vertices_np.min(axis=0))
    )
    epsilon = 1e-5
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        settings_ml = mm.ResolveMeshDegenSettings()
        settings_ml.tinyEdgeLength = 1e-2 * diagonal
        settings_ml.maxDeviation = 1e-3 * diagonal

        def resolve_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            mm.resolveMeshDegenerations(mesh_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert 0 < bench_case.run(resolve_ml, rounds=3) <= bench_case.n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    kept_vertices, kept_faces = bench_case.run(
        lambda: od.repair.collapse_small_triangles(vertices, faces, epsilon), rounds=3
    )
    assert kept_faces.size <= faces.size
    assert kept_vertices.size <= bench_case.n_vertices


@pytest.mark.benchmark(group="remove_tunnels")
@pytest.mark.benchaxis("genus")
@pytest.mark.benchlibs("ordito")
def test_remove_tunnels(bench_case: BenchCase) -> None:
    """
    Removing every thin handle: homology basis, shorten, cut, fill -- the whole chain in one call.

    On the ``genus`` axis, the only one in either registry with a handle to remove, so the row is
    read against ``benchmarks/test_homology.py`` and ``test_geodesic_walk.py::test_shorten_loop``:
    those two time the first and second stages of this call.

    Against the two upstream groups the **cut and fill dominate**, and increasingly: about half the
    call at genus 1 and more than three quarters at genus 64, where the
    basis and the shortening together are a small minority. That is worth stating because the
    obvious reading of the chain is the reverse -- the basis is the expensive thing in its own group
    and the shortening is the new code -- and neither is where the time goes.
    Note both rows eliminate **one** tunnel: the loops kept per call are vertex-disjoint, so
    ``handles_64``'s 128 overlapping generators still yield one, and its extra cost is the larger
    basis and the longer loops rather than more cutting.

    ordito-only, and not by omission. MeshLib is the only library that binds the operation and its
    ``eliminateTunnels`` is a **no-op** on every input probed -- unchanged face count and Euler
    characteristic at ``maxTunnelLength`` 4.0 and 1e9, ``maxIters`` 1 to 100, all three
    ``TunnelLoopType`` values, ``buildCoLoops`` off, and through the ``FillHoleNicelySettings``
    overload -- while its own ``detectTunnelFaces`` fires on the same mesh. A row that changes
    nothing is not a comparison, so it is left out rather than timed; see
    ``tests/test_repair.py::test_remove_tunnels_drops_the_genus_by_the_count_it_reports``.
    """
    if bench_case.mesh_name == "sphere_med":
        pytest.skip("sphere_med is genus 0: there is no tunnel to eliminate")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    # Every handle in these fixtures is a real one, so an unbounded length eliminates a maximal
    # disjoint family and times the whole chain rather than the length test.
    cut_vertices, cut_faces, removed = bench_case.run(
        lambda: od.repair.remove_tunnels(vertices, faces, 1e9), rounds=3
    )
    assert removed >= 1
    assert cut_vertices.size >= bench_case.n_vertices
    assert od.measures.euler_characteristic(cut_faces) == (
        od.measures.euler_characteristic(faces) + 2 * removed
    )


@pytest.mark.benchmark(group="remove_degree3_vertices")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_remove_degree3_vertices(bench_case: BenchCase) -> None:
    """
    One pass over every vertex, plus a face compaction -- and usually nothing to remove.

    The scan meshes carry few valence-3 interior vertices, so this group mostly times the *search*:
    one pass over the halfedges recording each vertex's face count and first three faces, one
    selection-and-emit pass, and one readback. That is the honest thing to measure, because the
    search is what a caller pays unconditionally in a repair pipeline while the removal is
    proportional to a defect that may not be there.

    meshlib's row is ``findInnerVertsOfDegree(topology, 3)`` -- the candidate mask, not the removal,
    since ``eliminateDegree3Vertices`` mutates in place and would need a fresh mesh per round while
    finding nothing after the first. So it is the same *search* on both sides; where it is not a
    like-for-like is that meshlib is handed a ``MeshTopology`` built outside its row and ordito
    validates edge-manifoldness inside its own.

    **The pass needs no halfedge structure.** An interior degree-3 vertex's three opposite edges
    close into one directed 3-cycle, which *is* the replacement triangle, so the per-pass
    ``vertex_one_rings`` build (42-70 % of the call) became a face-count table, and the selection,
    the emit and the next-pass signal one launch reading it. Byte-identical on the CPU device on the
    three scan meshes and on a two-level cascade; 1.43x on ``bunny``, 1.59x on ``dragon``, 2.15x on
    ``happy_buddha`` (one pass, two, three; interleaved, min, shared box). What is left is the one
    validating ``halfedge_twins`` on the input -- the documented non-manifold ``ValueError``,
    which is most of the call on the scan meshes -- and the final vertex compaction.

    Removing a fan changes only its rim vertices' face counts, so the emit kernel counts the rim
    vertices it may turn into candidates and a zero skips the confirming pass outright; that signal
    is conservative (it cannot see a count that falls past 3) and ran no extra pass on a scan mesh.

    **The passes append into one fixed buffer and compact once.** A replaced fan keeps its rows with
    a cleared flag and the replacements go after the last used row, so no pass scans, compacts or
    allocates, and one final launch writes the kept faces and the referenced vertices together:
    6 / 9 / 12 launches and 12 / 13 / 14 allocations at ``bunny`` / ``dragon`` / ``happy_buddha``
    became 5 / 7 / 9 and 7, scans 2-4 to 1, byte-identical on the CPU device; 1.13x / 1.16x /
    1.20x (interleaved processes, min, shared box). Recording the passes after the first as one
    device loop was built and declined -- see ``kernels/repair.degree3_fan_tables``.

    **The independent-set rewrite is declined, measured.** Every pass-0 candidate on all three scan
    meshes is selected -- no two candidates are adjacent (9, 1 285 and 1 839 candidates, zero
    candidate-candidate edges) -- so the selection is already the whole set and the later passes
    remove vertices that the earlier ones *created*, which no pass-0 selection can see.

    Moving the vertex compaction out of the loop was tried and is **flat** at the pass counts a scan
    mesh reaches; it is kept because it is strictly less work, not because it showed up.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        bits_ml = bench_case.run(lambda: mm.findInnerVertsOfDegree(mesh_ml.topology, 3))
        assert bits_ml.size() >= 0
        return
    if bench_case.mesh_name in {"bunny_decimated", "lucy"}:
        pytest.skip(f"{bench_case.mesh_name} is not edge-manifold, so it has no vertex fans")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    out_vertices, out_faces, removed = bench_case.run(
        lambda: od.repair.remove_degree3_vertices(vertices, faces, return_count=True), rounds=3
    )
    assert removed >= 0
    assert int(out_faces.shape[0]) <= faces.size
    assert int(out_vertices.shape[0]) <= bench_case.n_vertices


@pytest.mark.benchmark(group="straighten_boundary")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_straighten_boundary(bench_case: BenchCase) -> None:
    """
    Closing rim notches: per pass, a halfedge build, a rim walk and one candidate test per vertex.

    The cost is dominated by the halfedge build and the per-*vertex* candidate launch, so it tracks
    the mesh rather than the rim -- which is the honest thing to say about it, since a rim is
    ``O(sqrt(n_vertices))`` and the work is not. A rim-indexed candidate pass would fix that and is
    the obvious follow-up; it is not done because the pass count is one by default.

    meshlib's ``straightenBoundary`` is per **rim** and in place, so its row loops over
    ``findHoleRepresentiveEdges`` on a fresh mesh per round, and its topology build is inside the
    timed callable the way ours is. Both take the same two gates by the same definitions and agree
    exactly on a ragged planar rim (``tests/test_repair.py``).

    ordito wins the group by more than an order of magnitude and by more the larger the mesh. Both
    rows carry their own structure build, so the ratio is the parallel candidate test against a
    serial rim walk, and it widens with the mesh exactly as that predicts.

    The rim walk is keyed on **halfedges** rather than on vertices, because edge-manifoldness does
    not make the rim a set of simple loops and the per-vertex tables raced at a bowtie vertex
    (``kernels/repair.py::collect_rim_links_and_candidates``). It is a correctness fix and it
    also came out
    slightly *cheaper*: two per-halfedge tables replaced three per-vertex ones plus a face table,
    and the candidate and emit kernels lost three arguments between them; the fan walk that finds
    each boundary halfedge's successor is paid only on the rim.
    """
    if bench_case.kind == "meshlib":

        def straighten_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            holes_ml = mesh_ml.topology.findHoleRepresentiveEdges()
            for edge in holes_ml:
                mm.straightenBoundary(mesh_ml, edge, 0.9, 10.0)
            return len(holes_ml)

        assert bench_case.run(straighten_ml, rounds=3) >= 0
        return
    if bench_case.mesh_name in {"bunny_decimated", "lucy"}:
        pytest.skip(f"{bench_case.mesh_name} is not edge-manifold, so the rim cannot be walked")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    straightened, added = bench_case.run(
        lambda: od.repair.straighten_boundary(vertices, faces, return_count=True), rounds=3
    )
    assert added >= 0
    assert int(straightened.shape[0]) >= faces.size


@pytest.mark.benchmark(group="flatten_degree3_vertices")
@pytest.mark.benchlibs("ordito", "meshlib")
def test_flatten_degree3_vertices(bench_case: BenchCase) -> None:
    """
    The geometric answer to the same defect ``remove_degree3_vertices`` removes topologically.

    Read against that group directly: both build the same ``vertex_one_rings`` and both then run
    the same independent-set pass, and everything after *that* differs -- no face rewrite, no
    compaction and no loop here, so the gap between the two rows is the whole cost of removing
    rather than moving, and this row should sit at roughly the halfedge build alone -- which is
    about two thirds of the other group's single pass.

    meshlib's ``hardSmoothTetrahedrons`` is the same move on the same set, vertex for vertex --
    it sweeps sequentially, reading neighbours it has already moved, and one maximal independent
    set per pass with lowest index winning reproduces that order exactly (``tests/test_repair.py``,
    Class A on a tetrahedron where all four vertices are candidates). It mutates in place, so its
    mesh is rebuilt per round; the other group's meshlib row times only the *mask*, which is why
    this one is the like-for-like pair.

    The independent-set pass and its loop are not free: they cost roughly a third of the call, most
    of it the one host readback that terminates the loop (routing that through ``reduce.sum``
    instead of a bool readback is dearer still). It is a correctness fix rather than a tuning
    choice -- see ``kernels/repair.py``'s ``select_and_flatten_degree3`` for what moving two
    neighbours at once does to a tetrahedron -- so the cost is recorded rather than weighed.

    ordito wins the group by more than an order of magnitude at every size. Against
    ``remove_degree3_vertices`` on the same meshes this is several times cheaper -- the ratio the
    docstring predicts, since that group repeats the shared halfedge build once per pass and this
    one runs it once. The first mesh in a selection reads high for its size, because it carries the
    module's compile.
    """
    if bench_case.kind == "meshlib":

        def flatten_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            mm.hardSmoothTetrahedrons(mesh_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(flatten_ml, rounds=3) > 0
        return
    if bench_case.mesh_name in {"bunny_decimated", "lucy"}:
        pytest.skip(f"{bench_case.mesh_name} is not edge-manifold, so it has no vertex fans")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    flattened = bench_case.run(
        lambda: od.repair.flatten_degree3_vertices(vertices, faces), rounds=3
    )
    assert flattened.shape == (bench_case.n_vertices,)
