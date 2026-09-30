"""
Benchmarks for ``triwarp.selection``: submesh extraction and vertex-selection morphology.

Two axes, and neither is the face count:

* **hops**, for the morphology pair. ``expand_vertex_mask`` / ``shrink_vertex_mask`` run a host loop
  of one full face pass per hop, so their cost is ``hops x |F|`` -- completely independent of how
  many vertices are actually selected. That makes them the cleanest pure-iteration-count group in
  the suite: the slope between 1 and 8 hops should be exactly 8, and anything else means the pass
  is not doing constant work per hop.
* **components**, for ``exclude_fully_selected_components``. It runs a connected-components pass and
  then an all-selected test per component, so it inherits the component-count sensitivity that
  [`test_combine.py`](test_combine.py) documents for ``split``.

References
----------
**trimesh**'s ``Trimesh.submesh`` is the reference for the extraction group; it takes a sequence of
face-index groups and returns a list of meshes, so it is given a single group to match triwarp's
single submesh.

Neither trimesh nor open3d nor libigl has selection *morphology* -- growing or shrinking a mask
across the mesh graph. **pymeshlab** does, and is the only reference the morphology pair has:
``apply_selection_dilatation`` / ``apply_selection_erosion``, MeshLab's Dilate / Erode Selection.

Two differences to read the rows against, neither of them correctable:

- **It is face morphology, not vertex morphology.** MeshLab dilates the selected *face* set (via
  VCGlib's loose vertex-from-face / face-from-vertex pair), so seeding it needs
  ``compute_selection_by_condition_per_face`` and a vertex selection handed to it is simply cleared.
  triwarp grows a vertex mask through the faces' corners. Same operation class, same asymptotic work
  -- one full pass over the elements per hop -- on a different element type.
- **One filter call is one hop**, so the reference is a host loop of ``hops`` calls, which is
  structurally what ``expand_vertex_mask``'s own per-hop launch loop does.

The MeshSet is *shared* here rather than rebuilt per round, which is the exception to the rule in
``BenchCase.new_meshset_pml``: these filters touch only the selected bit, and their cost does not
depend on how much is selected (measured flat over 120 consecutive dilatations carrying
``sphere_med`` from a fraction of a percent to most of the faces selected). Rebuilding instead would
put a MeshSet build an order of magnitude dearer than the filter on top of it and flatten the slope
this group exists to measure. Both reference slopes are linear in the hop count, confirming
independently that a hop is constant work; triwarp is an order of magnitude ahead throughout.

trimesh's ``graph.connected_component_labels`` could reproduce
``exclude_fully_selected_components`` in several steps, but not as one call, so that group remains a
before/after self-comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pymeshlab as ml
import pytest
import pyvista as pv
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import triwarp as tw
import triwarp.typing as twt
from conftest import BenchCase

if TYPE_CHECKING:
    from typing_extensions import Buffer

_SEED = 5

# Hop counts for the morphology pair: cost is hops x edges, so the slope should be exactly 8.
_HOPS = [1, 8]

# Fraction of vertices seeded into the mask before growing it.
_SEED_FRACTION = 0.01

# Fraction of faces extracted into the submesh.
_SUBMESH_FRACTION = 0.5

_mask_cache: dict[tuple[str, str], wp.array[wp.bool]] = {}
_edges_cache: dict[tuple[str, str], twt.Array2dInt32] = {}
_indices_cache: dict[tuple[str, str], tuple[wp.array[wp.int32], np.ndarray]] = {}


def _seed_mask(bench_case: BenchCase) -> wp.array[wp.bool]:
    """Sparse boolean vertex mask -- 1% of the vertices, at a fixed seed."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _mask_cache:
        rng = np.random.default_rng(_SEED)
        n = bench_case.n_vertices
        mask_np = np.zeros(n, dtype=bool)
        mask_np[rng.choice(n, size=max(1, int(n * _SEED_FRACTION)), replace=False)] = True
        _mask_cache[key] = wp.array(mask_np, dtype=wp.bool, device=bench_case.device)
    return _mask_cache[key]


def _seed_mask_np(bench_case: BenchCase) -> np.ndarray:
    """Build the same 1% seed as [`_seed_mask`], as a host bool array for the meshlib rows."""
    rng = np.random.default_rng(_SEED)
    n = bench_case.n_vertices
    mask_np = np.zeros(n, dtype=bool)
    mask_np[rng.choice(n, size=max(1, int(n * _SEED_FRACTION)), replace=False)] = True
    return mask_np


def _unique_edges(bench_case: BenchCase) -> twt.Array2dInt32:
    """Build the unique edge table once: an *input*, so morphology never re-times the sort."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _edges_cache:
        _edges_cache[key] = tw.edges.edges_unique(
            bench_case.faces_wp, n_vertices=bench_case.n_vertices
        )[0]
    return _edges_cache[key]


def _seeded_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """
    Return the shared MeshSet with a small face selection already established.

    Seeded *outside* the timed callable, which the module docstring justifies: MeshLab's dilate and
    erode cost the same whatever fraction is selected, so what the rounds start from does not change
    what they measure. The condition picks a polar cap rather than the random 1% ``_seed_mask``
    uses -- MeshLab has no way to set a selection array directly, only to derive one.
    """
    meshset_pml = bench_case.meshset_pml
    meshset_pml.compute_selection_by_condition_per_face(condselect="(z0 > 0.98)")
    return meshset_pml


@pytest.mark.benchmark(group="expand_vertex_mask")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "pymeshlab", "meshlib")
@pytest.mark.parametrize("hops", _HOPS)
def test_expand_vertex_mask(bench_case: BenchCase, hops: int) -> None:
    """
    One full face pass per hop, regardless of selection size: the slope should be exactly 8.

    meshlib's ``expand`` takes the hop count itself and dilates a ``VertBitSet`` over the *same*
    vertex neighbourhood, agreeing element for element (``tests/test_selection.py``), so unlike the
    pymeshlab row it needs no face round trip. It mutates the bitset in place, so a fresh one is
    built inside the timed callable from the shared seed -- the bitset build is a bit-pack of the
    seed array and is the row's floor.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        seed_np = np.ascontiguousarray(_seed_mask_np(bench_case))

        def expand_ml() -> int:
            region_ml = mn.vertBitSetFromBools(cast("Buffer", seed_np))
            mm.expand(mesh_ml.topology, region_ml, hops)
            return region_ml.count()

        assert bench_case.run(expand_ml) > 0
        return
    if bench_case.kind == "triwarp":
        faces, mask = bench_case.faces_wp, _seed_mask(bench_case)
        grown = bench_case.run(lambda: tw.selection.expand_vertex_mask(faces, mask, hops))
        assert grown.shape == mask.shape
    else:  # one Dilate Selection call per hop, on the face set
        meshset_pml = _seeded_meshset_pml(bench_case)

        def dilate_pml() -> int:
            for _ in range(hops):
                meshset_pml.apply_selection_dilatation()
            return meshset_pml.current_mesh().selected_face_number()

        assert bench_case.run(dilate_pml) > 0


@pytest.mark.noparity(
    "pymeshlab",
    reason="D4: MeshLab's Erode Selection removes a face when *any* of its vertices is on the "
    "selection boundary, so reading the vertex selection back gives the vertices of the "
    "surviving faces, not an eroded vertex set. Measured on a 9x9 grid: 51 / 39 / 25 vertices "
    "after 1 / 2 / 3 erosions where shrink_vertex_mask gives 19 / 7 / 1. Dilation does map and is "
    "covered; erosion is a different operation, and meshlib's shrink -- which erodes a VertBitSet "
    "by one-ring layers and agrees element for element -- is the oracle for this function, in "
    "tests/test_selection.py::test_expand_and_shrink_vertex_mask_match_meshlib.",
)
@pytest.mark.benchmark(group="shrink_vertex_mask")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "pymeshlab", "meshlib")
@pytest.mark.parametrize("hops", _HOPS)
def test_shrink_vertex_mask(bench_case: BenchCase, hops: int) -> None:
    """
    The erosion counterpart, on the same input: should match ``expand`` row for row.

    meshlib is the reference the exemption above says this group lacked: ``shrink`` erodes a
    ``VertBitSet`` by one-ring layers, which is triwarp's operation and not MeshLab's face-based
    one, and the two agree element for element at every hop count. The mask is pre-grown outside
    the timed callable on both sides, as the triwarp row does, so the rounds erode the same set.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        seed_np = np.ascontiguousarray(_seed_mask_np(bench_case))
        grown_ml = mn.vertBitSetFromBools(cast("Buffer", seed_np))
        mm.expand(mesh_ml.topology, grown_ml, max(_HOPS))
        grown_np = np.ascontiguousarray(mn.getNumpyBitSet(grown_ml))

        def shrink_ml() -> int:
            region_ml = mn.vertBitSetFromBools(cast("Buffer", grown_np))
            mm.shrink(mesh_ml.topology, region_ml, hops)
            return region_ml.count()

        assert bench_case.run(shrink_ml) >= 0
        return
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        # Grow first so there is something left to erode after 8 hops.
        mask = tw.selection.expand_vertex_mask(faces, _seed_mask(bench_case), max(_HOPS))
        shrunk = bench_case.run(lambda: tw.selection.shrink_vertex_mask(faces, mask, hops))
        assert shrunk.shape == mask.shape
    else:  # dilate well past the erosion depth first, so there is something left to erode
        meshset_pml = _seeded_meshset_pml(bench_case)
        for _ in range(2 * max(_HOPS)):
            meshset_pml.apply_selection_dilatation()

        def erode_pml() -> None:
            for _ in range(hops):
                meshset_pml.apply_selection_erosion()

        bench_case.run(erode_pml)


def _face_indices(bench_case: BenchCase) -> tuple[wp.array[wp.int32], np.ndarray]:
    """Half the faces as a ``(indices_wp, indices_np)`` pair, already unique and sorted."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _indices_cache:
        count = max(1, int(bench_case.n_faces * _SUBMESH_FRACTION))
        indices_np = np.arange(count, dtype=np.int32)
        _indices_cache[key] = (
            wp.array(indices_np, dtype=wp.int32, device=bench_case.device),
            indices_np,
        )
    return _indices_cache[key]


@pytest.mark.benchmark(group="submesh_from_face_indices")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pyvista")
def test_submesh_from_face_indices(bench_case: BenchCase) -> None:
    """
    Face gather plus a vertex remap.

    Three references, and all three do the same two things triwarp does -- gather the faces and
    **compact** the vertex buffer. So the rows are like-for-like on the work; what differs is
    the interface, in two ways that both cost something. open3d takes a *mask* rather than an index
    list, so building one is part of its row -- an ``O(n_faces)`` scatter against triwarp's
    ``O(len(indices))`` gather. And open3d returns **no vertex map**, where triwarp's
    ``return_index`` and pyvista's ``vtkOriginalPointIds`` both do, which is why the correctness
    comparison has to match its positions instead (``tests/test_selection.py``).

    **pymeshlab is absent for an interface reason rather than a cost one.** It has no array-valued
    face-selection setter: a selection has to be produced by a
    ``compute_selection_by_condition_per_face`` *expression* over face attributes, so it can express
    a contiguous range (``fi<80``) and not an arbitrary index list. This group's input is a random
    index set, which that filter cannot be handed at all.
    """
    if bench_case.kind == "open3d":
        import open3d as o3d

        _indices_wp, indices_np = _face_indices(bench_case)
        mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(bench_case.mesh_o3d)
        n_faces = bench_case.n_faces

        def select_faces_o3d() -> int:
            mask_np = np.zeros(n_faces, dtype=bool)
            mask_np[indices_np] = True
            return int(
                mesh_o3d.select_faces_by_mask(
                    # The stub omits ``Tensor``'s device default.
                    o3d.core.Tensor(mask_np, dtype=o3d.core.Dtype.Bool)  # pyright: ignore[reportCallIssue]
                ).triangle.indices.shape[0]
            )

        assert bench_case.run(select_faces_o3d) <= indices_np.size
        return
    if bench_case.kind == "pyvista":
        _indices_wp, indices_np = _face_indices(bench_case)
        mesh_pv = bench_case.mesh_pv
        extracted_pv = bench_case.run(lambda: mesh_pv.extract_cells(indices_np))
        assert extracted_pv.n_cells == indices_np.shape[0]
        return
    if bench_case.kind == "triwarp":
        indices_wp, _indices_np = _face_indices(bench_case)
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _sub_vertices, sub_faces = bench_case.run(
            lambda: tw.selection.submesh_from_face_indices(vertices, faces, indices_wp)
        )
        assert sub_faces.size == 3 * indices_wp.size
    else:
        _indices_wp, indices_np = _face_indices(bench_case)
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        parts = bench_case.run(lambda: mesh_tm.submesh([indices_np], append=False))
        assert len(parts[0].faces) == indices_np.shape[0]


_region_cache: dict[tuple[str, str], tuple[wp.array[wp.bool], np.ndarray]] = {}


def _cap_region(bench_case: BenchCase) -> tuple[wp.array[wp.bool], np.ndarray]:
    """
    Select a **contiguous** face region: the cap above the mesh's 80th height percentile.

    Contiguity is the point: a scattered mask opens one rim per face and turns a region deletion
    into a hole-filling benchmark. One cap opens one rim, the shape a real region edit has.
    """
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _region_cache:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        height_np = vertices_np[faces_np].mean(axis=1)[:, 2]
        mask_np = height_np > np.quantile(height_np, 0.8)
        _region_cache[key] = (wp.array(mask_np, dtype=wp.bool, device=bench_case.device), mask_np)
    return _region_cache[key]


def _face_mask_ml(mask_np: np.ndarray) -> mm.FaceBitSet:
    """
    Load a dense face mask into a MeshLib ``FaceBitSet``, through the packed blocks.

    The benchmark-side twin of ``tests.conversions.numpy_to_meshlib_bitset`` (the two suites do not
    import each other), and the same reason it is one ``np.packbits`` rather than a per-face
    ``set()`` loop: ``BitSet.fromBlocks`` takes ``uint64`` blocks, ``bitorder="little"`` is not
    NumPy's default and is not optional, and ``fromBlocks`` rounds up to whole blocks so the size is
    trimmed back after. It is ``setup`` work either way -- the region is an input.
    """
    packed_np = np.packbits(mask_np, bitorder="little")
    packed_np = np.pad(packed_np, (0, (-packed_np.size) % 8)).view(np.uint64)
    bitset_ml = mm.BitSet.fromBlocks(mm.std_vector_unsigned_long(packed_np.tolist()))
    bitset_ml.resize(int(mask_np.size))
    return mm.FaceBitSet(bitset_ml)


@pytest.mark.benchmark(group="delete_region_keep_boundary")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_delete_region_keep_boundary(bench_case: BenchCase) -> None:
    """
    Delete a contiguous face region and trace the rims it opened.

    Read against ``submesh_from_face_indices`` above: the extraction is the same gather-and-remap,
    and the difference between the two rows is the rim work -- one boundary-loop trace plus the
    host-side classification that decides which loops are *new*. That classification is the reason
    this is not simply the submesh call, and the reason it is worth its own row.

    meshlib's ``delRegionKeepBd`` is the same operation and returns the rims as edge lists;
    ``tests/test_selection.py`` pins the two to the same survivor count and the same loop lengths.
    It mutates its mesh, so the row gets a fresh one per round.

    An order of magnitude behind meshlib, and attributed rather than left open: ``boundary_loops``
    on the survivor is the majority of the call, the submesh extraction and the input's rim pass
    make up most of the rest, and the host-side loop classification is negligible. So the
    composition is not the problem and neither is the host code: the loop trace is, and it is a
    function of its own with its own group. Read this row's ratio as a statement about
    ``boundary_loops``. Note also a substantial run-to-run spread on unchanged code here, so read
    medians across sessions with care.
    """
    if bench_case.kind == "meshlib":
        _mask_wp, mask_np = _cap_region(bench_case)
        region_ml = _face_mask_ml(mask_np)

        loops_ml = bench_case.run(
            lambda mesh: mm.delRegionKeepBd(mesh, region_ml, False), setup=bench_case.new_mesh_ml
        )
        assert len(loops_ml) >= 1
        return

    mask_wp, _mask_np = _cap_region(bench_case)
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _kept_vertices, kept_faces, loops = bench_case.run(
        lambda: tw.selection.delete_region_keep_boundary(vertices, faces, mask_wp)
    )
    assert kept_faces.size > 0
    assert len(loops) >= 1


@pytest.mark.benchmark(group="delete_region_keep_boundary_with_offsets")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp")
def test_delete_region_keep_boundary_with_offsets(bench_case: BenchCase) -> None:
    """
    The packed form of ``delete_region_keep_boundary``, over the same cap region.

    Read against that group's triwarp row: the difference is the per-loop views, which the packed
    form replaces with a device compaction of the kept loops.
    """
    mask_wp, _mask_np = _cap_region(bench_case)
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _kept_vertices, kept_faces, loops, offsets = bench_case.run(
        lambda: tw.selection.delete_region_keep_boundary_with_offsets(vertices, faces, mask_wp)
    )
    assert kept_faces.size > 0
    assert offsets.size >= 2
    assert loops.size > 0


@pytest.mark.benchmark(group="exclude_fully_selected_components")
@pytest.mark.benchaxis("components")
@pytest.mark.benchlibs("triwarp")
def test_exclude_fully_selected_components(bench_case: BenchCase) -> None:
    """A components pass plus an all-selected reduce per component: inherits ``split``'s axis."""
    faces, n_vertices = bench_case.faces_wp, bench_case.n_vertices
    edges = _unique_edges(bench_case)
    mask = _seed_mask(bench_case)
    kept = bench_case.run(
        lambda: tw.selection.exclude_fully_selected_components(
            faces, mask, n_vertices, unique_edges=edges
        )
    )
    assert kept.shape == mask.shape


_FACE_REGION_FRACTION = 0.25


def _seed_face_region(bench_case: BenchCase) -> np.ndarray:
    """Mask a contiguous quarter of the face buffer as the region -- one seam, not a scatter."""
    mask_np = np.zeros(bench_case.n_faces, dtype=bool)
    mask_np[: int(bench_case.n_faces * _FACE_REGION_FRACTION)] = True
    return mask_np


@pytest.mark.benchmark(group="region_boundary_edges")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "meshlib", "pyvista")
def test_region_boundary_edges(bench_case: BenchCase) -> None:
    """
    The interior seam around a face region: one edge pass, independent of the region's size.

    meshlib's ``findRegionBoundaryUndirectedEdgesInsideMesh`` is the function triwarp's is named
    after and the only reference that has it -- pinned edge-for-edge in
    tests/test_selection.py::test_region_boundary_edges. Its answer is an ``UndirectedEdgeBitSet``
    rather than an ``(k, 2)`` array, so this row times the seam pass on both sides but not the
    decoding, which is test-side. Pure, so the mesh is built once outside the timed callable.

    pyvista has no seam filter and reaches the same edges by **construction**: extract the region as
    a sub-surface, then take that surface's boundary edges. So its row is doing strictly more -- a
    cell extraction and a surface pass before the edge walk -- and its answer is a
    superset, since a region touching the mesh's own rim contributes those edges too. The transform
    is subtracting them, and at that it is exact (``tests/test_selection.py``).
    """
    region_np = _seed_face_region(bench_case)
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        region_pv = np.flatnonzero(region_np)

        def region_edges_pv() -> pv.PolyData:
            surface_pv = mesh_pv.extract_cells(region_pv).extract_surface(
                algorithm="dataset_surface"
            )
            return surface_pv.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                non_manifold_edges=False,
                manifold_edges=False,
            )

        assert bench_case.run(region_edges_pv).n_cells > 0
        return
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        region_ml = mn.faceBitSetFromBools(cast("Buffer", region_np))
        bits_ml = bench_case.run(
            lambda: mm.findRegionBoundaryUndirectedEdgesInsideMesh(mesh_ml.topology, region_ml)
        )
        assert bits_ml.count() > 0
        return
    faces = bench_case.faces_wp
    region_wp = wp.array(region_np, dtype=wp.bool, device=bench_case.device)
    n_vertices = bench_case.n_vertices
    edges = bench_case.run(
        lambda: tw.selection.region_boundary_edges(faces, region_wp, n_vertices=n_vertices)
    )
    assert int(edges.shape[0]) > 0


@pytest.mark.benchmark(group="faces_left_of_contour")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_faces_left_of_contour(bench_case: BenchCase) -> None:
    """
    The reverse of the group above: a seam back into the region it bounds.

    A flood fill on the dual graph blocked by the contour, so the cost is a face-adjacency pass plus
    a connected-component labelling -- both independent of the contour's length, which is why the
    contour is built once outside the timed callable on both sides.

    meshlib's ``fillContourLeft`` is the reference and agrees with this **exactly**, mask for mask
    and with the same convention for which side is "left"
    (tests/test_selection.py::test_faces_left_of_contour_matches_meshlib). Its input is a vector of
    directed ``EdgeId`` rather than an ``(k, 2)`` array; that vector is assembled outside the timed
    callable too, since it is a Python loop over ``findEdge`` and would otherwise be the row.

    An order of magnitude behind meshlib on a mesh with a short contour. Two things about that.

    It is **not** a like-for-like: meshlib is handed a ``MeshTopology`` built outside its row and
    floods from the seeds with a serial BFS, while triwarp builds the halfedge structure inside its
    row and labels *every* component before gathering. ``connected_component_labels_from_edges`` is
    most of the call once ``twins`` is supplied: the labelling is the algorithm, and the way to beat
    it would be a device-side frontier BFS, which this package has measured as a loss.

    The dual graph, the blocking test and the seeding are one pass over ``twins`` rather than a
    ``face_adjacency`` call plus a key sort over every halfedge plus a mask-compact over every dual
    edge -- a real win, and less than the stage timings projected, which is the usual direction for
    a projection built by subtraction.
    """
    region_np = _seed_face_region(bench_case)
    n_faces = bench_case.n_faces
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        contour_ml = mm.std_vector_Id_EdgeTag()
        for start, end in _oriented_seam_np(bench_case, region_np).tolist():
            contour_ml.append(mesh_ml.topology.findEdge(mm.VertId(int(start)), mm.VertId(int(end))))
        assert len(contour_ml) > 0
        bits_ml = bench_case.run(lambda: mm.fillContourLeft(mesh_ml.topology, contour_ml))
        assert 0 < bits_ml.count() < n_faces
        return
    faces = bench_case.faces_wp
    n_vertices = bench_case.n_vertices
    contour = tw.selection.region_boundary_edges(
        faces,
        wp.array(region_np, dtype=wp.bool, device=bench_case.device),
        n_vertices=n_vertices,
        oriented=True,
    )
    left = bench_case.run(
        lambda: tw.selection.faces_left_of_contour(faces, contour, n_vertices=n_vertices)
    )
    assert 0 < int(left.numpy().sum()) < n_faces


def _oriented_seam_np(bench_case: BenchCase, region_np: np.ndarray) -> np.ndarray:
    """Return the region's oriented seam as host rows, for a reference wanting vertex pairs."""
    faces = _faces_for_reference(bench_case)
    return tw.selection.region_boundary_edges(
        faces,
        wp.array(region_np, dtype=wp.bool, device=faces.device),
        n_vertices=bench_case.n_vertices,
        oriented=True,
    ).numpy()


def _faces_for_reference(bench_case: BenchCase) -> wp.array[wp.int32]:
    """Build a host face buffer: a reference case has no device, so ``faces_wp`` is absent."""
    return wp.array(
        np.ascontiguousarray(bench_case.faces_np.ravel(), dtype=np.int32),
        dtype=wp.int32,
        device="cpu",
    )
