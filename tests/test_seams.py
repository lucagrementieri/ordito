"""Regression tests for ``ordito.seams`` against pymeshlab (CPU reference)."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
import ordito.typing as odt
from tests.comparisons import lexsort_rows, same_partition
from tests.conversions import (
    meshlib_bitset_to_numpy,
    meshlib_to_trimesh,
    numpy_to_warp,
    points_to_warp_uv,
    pyvista_edges_to_indices,
    trimesh_to_meshlib,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
    warp_empty,
    warp_to_trimesh,
    wedge_uv_to_pymeshlab,
)


def _sorted_edge_set(edges_np: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(row) for row in np.sort(edges_np, axis=1).tolist()}


def _face_component_count(
    vertices_wp: wp.array[wp.vec3] | wp.array[wp.vec3d], faces_wp: wp.array[wp.int32]
) -> int:
    mesh_tm = warp_to_trimesh(cast("wp.array[wp.vec3]", vertices_wp), faces_wp)
    return len(
        tm.graph.connected_components(
            mesh_tm.face_adjacency, nodes=np.arange(mesh_tm.faces.shape[0])
        )
    )


@pytest.mark.parametrize(
    ("angle", "n_creases"), [(0.0, 12), (30.0, 12), (89.0, 12), (91.0, 0), (180.0, 0)]
)
def test_crease_edges_thresholds(
    unit_box: tuple[tm.Trimesh, wp.Mesh], angle: float, n_creases: int
) -> None:
    """
    A unit cube has exactly 12 crease edges at 90 degrees, and its 6 face diagonals are flat.

    The comparison is strict, so ``0`` selects every *non-coplanar* interior edge: on a cube that
    is the 12 cube edges and not the 6 face diagonals, whose dihedral is exactly zero -- which is
    the useful reading of "all of them", and the reason the test pins ``0`` rather than treating it
    as a synonym for the whole 18-edge set. Every selected edge joins two cube corners one unit
    apart; a face diagonal would be ``sqrt(2)``.
    """
    box_tm, box_wp = unit_box
    creases_np = od.seams.crease_edges(box_wp.points, box_wp.indices, angle=angle).numpy()
    assert creases_np.shape == (n_creases, 2)
    lengths_np = np.linalg.norm(
        box_tm.vertices[creases_np[:, 0]] - box_tm.vertices[creases_np[:, 1]], axis=1
    )
    assert np.allclose(lengths_np, 1.0, rtol=1e-5)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "saddle_graded"])
@pytest.mark.parity("crease_edges", "pymeshlab")
def test_crease_edges_matches_pymeshlab(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (selection to edge rows): the same dihedral threshold, reported as a selection.

    MeshLab selects *vertices* of crease edges rather than the edges themselves, so the comparison
    is on the vertex set the two edge lists span — which is the quantity a caller of either one
    actually uses (it is what gets duplicated by a cut, or pinned by a solver).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    angle = 40.0
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_selection_crease_per_edge(angledegneg=-angle, angledegpos=angle)
    selected_pml = np.flatnonzero(meshset_pml.current_mesh().vertex_selection_array())

    creases_np = od.seams.crease_edges(mesh_wp.points, mesh_wp.indices, angle=angle).numpy()
    assert set(np.unique(creases_np).tolist()) == set(selected_pml.tolist())


@pytest.mark.parametrize("angle", [30.0, 60.0])
@pytest.mark.parity("crease_edges", "meshlib")
def test_crease_edges_matches_meshlib(unit_box: tuple[tm.Trimesh, wp.Mesh], angle: float) -> None:
    """
    Class B (undirected-edge decode): the same dihedral threshold, reported as a bitset.

    ``findCreaseEdges`` returns an ``UndirectedEdgeBitSet``, so the two named transforms are its
    **units** and its **indexing**. The angle is in *radians* -- there is no degree overload -- and
    each set bit is an undirected edge id, which becomes a vertex pair through
    ``EdgeId(2 * ue)`` and the topology's ``org`` / ``dest``. Both sides are then sorted within a
    row and across rows, since neither orients an undirected edge.

    Exact on both thresholds: 12 edges each on a unit box, the same 12 vertex pairs. The box is the
    fixture for the reason the pyvista pairing above uses it -- a smooth sphere yields ``[] == []``
    at 30 degrees and would test nothing -- and the count is asserted before the sets are compared.

    Note the *sign* of the threshold is not shared vocabulary: MeshLab takes a signed pair
    (``angledegneg`` / ``angledegpos``) and this takes ``angleFromPlanar``, one number measured from
    flat, which is ordito's ``angle``.
    """
    mesh_tm, mesh_wp = unit_box
    creases_wp = od.seams.crease_edges(mesh_wp.points, mesh_wp.indices, angle=angle)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    crease_bits_ml = mm.findCreaseEdges(mesh_ml.topology, mesh_ml.points, float(np.radians(angle)))
    edges_ml = np.array(
        sorted(
            sorted(
                (
                    int(mesh_ml.topology.org(mm.EdgeId(int(edge) * 2))),
                    int(mesh_ml.topology.dest(mm.EdgeId(int(edge) * 2))),
                )
            )
            for edge in np.flatnonzero(
                meshlib_bitset_to_numpy(crease_bits_ml, mesh_ml.topology.undirectedEdgeSize())
            )
        )
    )
    edges_wp = np.array(sorted(sorted(row) for row in creases_wp.numpy().tolist()))

    assert edges_ml.shape[0] == 12  # the box's creases, so neither side is empty
    assert np.array_equal(edges_wp, edges_ml)


@pytest.mark.parity("crease_edges", "pyvista")
def test_crease_edges_matches_pyvista(unit_box: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B: ``extract_feature_edges`` is the same dihedral threshold, as line cells.

    Two named transforms, both exact. VTK returns a **new** ``PolyData`` whose points are its own
    renumbered subset, so its lines go through
    [`tests.conversions.pyvista_edges_to_indices`][] first; and the three other edge classes
    (``boundary_edges``, ``non_manifold_edges``, ``manifold_edges``) have to be switched **off**,
    because VTK's default emits all four and ordito's ``include_boundary=False`` emits only the
    crease.

    The fixture is a **box**, not a curved mesh: on a smooth sphere the reference finds nothing at
    ``30`` degrees, so the comparison would be ``[] == []`` -- the ``test_ears`` failure. A box has
    exactly 12 feature edges, which is asserted before the sets are compared.
    """
    mesh_tm, mesh_wp = unit_box
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices

    edges_pv = pyvista_edges_to_indices(
        trimesh_to_pyvista(mesh_tm).extract_feature_edges(
            feature_angle=30.0,
            feature_edges=True,
            boundary_edges=False,
            non_manifold_edges=False,
            manifold_edges=False,
        ),
        mesh_tm.vertices,
    )
    assert len(edges_pv) == 12  # non-vacuous, and the count a cube's creases must have

    creases_wp = od.seams.crease_edges(vertices_wp, faces_wp, angle=30.0)
    assert np.array_equal(lexsort_rows(np.sort(creases_wp.numpy(), axis=1)), lexsort_rows(edges_pv))


def test_crease_edges_include_boundary(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Ordito against ordito: the flag adds exactly the boundary edges and nothing else.

    Stated as a set identity against ``boundary.boundary_edges``, which carries its own
    reference comparison. MeshLab's crease filter has no equivalent flag, so this branch has no
    external oracle -- the boundary count is asserted non-zero first so the union cannot hold
    trivially.
    """
    mesh_tm, mesh_wp = hemisphere
    interior_np = od.seams.crease_edges(mesh_wp.points, mesh_wp.indices, angle=40.0).numpy()
    with_boundary_np = od.seams.crease_edges(
        mesh_wp.points, mesh_wp.indices, angle=40.0, include_boundary=True
    ).numpy()
    boundary_np = od.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices).numpy()
    assert len(boundary_np) > 0
    assert _sorted_edge_set(with_boundary_np) == _sorted_edge_set(interior_np) | _sorted_edge_set(
        boundary_np
    )
    assert mesh_tm.vertices.shape[0] > 0


def test_crease_edges_bucketed_match_the_key_sort(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ordito against ordito: the bucketed mates give the key sort's creases and boundary, in order.

    The sort path carries the oracles (pymeshlab, MeshLib, pyvista); on the bucket path only the
    flagged halfedges are sorted, both classes in one sort, so each block -- creases, then the
    boundary edges -- is pinned to strictly ascending ``(max, min)`` keys and the whole to the sort
    path's rows.
    """
    _mesh_tm, mesh_wp = hemisphere
    vertices, faces = mesh_wp.points, mesh_wp.indices
    sorted_np = od.seams.crease_edges(vertices, faces, 10.0, include_boundary=True).numpy()
    n_creases = od.seams.crease_edges(vertices, faces, 10.0).shape[0]
    monkeypatch.setattr(od.halfedge, "_BUCKETED_PAIRING_ON_CPU", True)
    monkeypatch.setattr(od.halfedge, "_BUCKETED_MATES_FROM_HALFEDGES", 0)
    bucketed_np = od.seams.crease_edges(vertices, faces, 10.0, include_boundary=True).numpy()
    assert 0 < n_creases < len(sorted_np)
    _assert_edge_key_order(bucketed_np[:n_creases])
    _assert_edge_key_order(bucketed_np[n_creases:])
    assert np.array_equal(sorted_np, bucketed_np)


def test_cut_along_edges_is_geometrically_a_noop(device: str) -> None:
    """Every output vertex sits exactly where its input did, so the surface is unchanged."""
    box_tm = tm.creation.box(extents=[1.0, 2.0, 3.0])
    vertices_wp, faces_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    creases_wp = od.seams.crease_edges(vertices_wp, faces_wp, angle=30.0)
    cut_vertices_wp, cut_faces_wp = od.seams.cut_along_edges(vertices_wp, faces_wp, creases_wp)

    before_np = box_tm.vertices[box_tm.faces].astype(np.float32)
    after_np = cut_vertices_wp.numpy()[cut_faces_wp.numpy().reshape(-1, 3)]
    assert np.allclose(after_np, before_np, rtol=1e-6, atol=1e-6)


def test_cut_along_edges_with_no_edges_is_the_identity(
    unit_box: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    _box_tm, box_wp = unit_box
    vertices_wp, faces_wp = box_wp.points, box_wp.indices
    cut_vertices_wp, cut_faces_wp = od.seams.cut_along_edges(
        vertices_wp, faces_wp, odt.empty_2d((0, 2), wp.int32, device=device)
    )
    assert cut_vertices_wp.size == vertices_wp.size
    assert np.allclose(
        np.sort(cut_vertices_wp.numpy(), axis=0), np.sort(vertices_wp.numpy(), axis=0)
    )
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == 1


def test_cut_along_edges_all_interior_edges_gives_a_triangle_soup(
    device: str, icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """Cutting everything leaves one vertex per corner: the definition of a soup."""
    sphere_tm, _sphere_tm_wp = icosphere_coarse
    vertices_wp, faces_wp = numpy_to_warp(sphere_tm.vertices, sphere_tm.faces, device)
    all_edges_wp = od.seams.crease_edges(vertices_wp, faces_wp, angle=0.0)
    cut_vertices_wp, cut_faces_wp = od.seams.cut_along_edges(vertices_wp, faces_wp, all_edges_wp)
    assert cut_vertices_wp.size == faces_wp.size
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == faces_wp.size // 3


def test_cut_along_edges_ignores_the_winding(
    device: str, icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Ordito against ordito: reversing some faces' winding does not change which corners merge.

    The cut is defined on undirected edges, so an edge-manifold but inconsistently wound mesh must
    give the same corner partition as the consistently wound one -- which carries the igl and
    MeshLib comparisons below. The flipped faces put same-direction twins on every edge they share
    with an unflipped face, the branch no consistently wound fixture reaches (and which only a
    caller's own ``twins`` table can reach); a cut that joins the wrong corners there splits or
    merges vertices the consistent cut does not.
    """
    sphere_tm, _sphere_tm_wp = icosphere_coarse
    faces_np = np.asarray(sphere_tm.faces, dtype=np.int32)
    flipped = np.arange(faces_np.shape[0]) % 7 == 0
    flipped_faces_np = faces_np.copy()
    flipped_faces_np[flipped] = faces_np[flipped][:, [0, 2, 1]]
    vertices_wp, faces_wp = numpy_to_warp(sphere_tm.vertices, faces_np, device)
    edges_wp = odt.as_array2d(
        wp.array(
            od.seams.crease_edges(vertices_wp, faces_wp, angle=0.0).numpy()[::3],
            dtype=wp.int32,
            device=device,
        ),
        wp.int32,
    )
    _, cut_faces_wp = od.seams.cut_along_edges(vertices_wp, faces_wp, edges_wp)
    _, flipped_faces_wp = numpy_to_warp(sphere_tm.vertices, flipped_faces_np, device)
    # ``halfedge_twins`` pairs only opposite halfedges and leaves a same-direction edge at ``-1``,
    # so the table that reaches the same-direction branch is a caller's own, pairing the two
    # halfedges of each interior edge by their undirected endpoints.
    corners = np.stack([flipped_faces_np, np.roll(flipped_faces_np, -1, axis=1)], axis=-1)
    keys_np = np.sort(corners.reshape(-1, 2), axis=1)
    order = np.lexsort((keys_np[:, 1], keys_np[:, 0]))
    twins_np = np.full(keys_np.shape[0], -1, dtype=np.int32)
    first, second = order[0::2], order[1::2]
    assert np.array_equal(keys_np[first], keys_np[second])  # closed and edge-manifold
    twins_np[first], twins_np[second] = second, first
    flipped_twins_wp = wp.array(twins_np, dtype=wp.int32, device=device)
    _, cut_flipped_wp = od.seams.cut_along_edges(
        vertices_wp, flipped_faces_wp, edges_wp, twins=flipped_twins_wp
    )
    cut_np = cut_faces_wp.numpy().reshape(-1, 3)
    # Undo the flip on the output so corner ``(f, k)`` names the same corner in both cuts.
    cut_flipped_np = cut_flipped_wp.numpy().reshape(-1, 3)
    cut_flipped_np[flipped] = cut_flipped_np[flipped][:, [0, 2, 1]]
    n_out = np.unique(cut_np).size
    assert sphere_tm.vertices.shape[0] < n_out < cut_np.size
    assert same_partition(cut_np.ravel(), cut_flipped_np.ravel())


def test_cut_along_edges_opens_a_boundary(
    unit_box: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    A cut edge set becomes boundary, which is the point: a closed cube gains two boundary loops.

    The cut set has to be a **closed curve in the edge graph**, not merely a band of edges. Cutting
    a single edge of a closed surface separates nothing — both endpoints stay connected the long way
    round their fans — and so does any set that leaves some vertex with only one cut edge. The four
    edges bounding one cube face are the smallest set that does close, which is why they are the
    fixture: each of its four corners then has *two* cut edges and splits in two.
    """
    box_tm, box_wp = unit_box
    vertices_wp, faces_wp = box_wp.points, box_wp.indices
    assert od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False)

    creases_np = od.seams.crease_edges(vertices_wp, faces_wp, angle=30.0).numpy()
    on_face_np = (box_tm.vertices[creases_np[:, 0], 0] > 0.0) & (
        box_tm.vertices[creases_np[:, 1], 0] > 0.0
    )
    assert on_face_np.sum() == 4
    ring_wp = wp.array(
        np.ascontiguousarray(creases_np[on_face_np], dtype=np.int32), dtype=wp.int32, device=device
    )

    cut_vertices_wp, cut_faces_wp = od.seams.cut_along_edges(
        vertices_wp, faces_wp, odt.as_array2d(ring_wp, wp.int32)
    )
    cut_vertices_wp = cast("wp.array[wp.vec3]", cut_vertices_wp)  # a vec3 input cuts to vec3
    # The four corners of that face each split in two; the other four are untouched.
    assert cut_vertices_wp.size == vertices_wp.size + 4
    assert not od.validation.is_edge_manifold(cut_faces_wp, allow_boundary_edges=False)
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == 2
    assert len(od.boundary.boundary_loops(cut_vertices_wp, cut_faces_wp)) == 2


def test_cut_along_edges_round_trips_through_a_weld(unit_box: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Welding coincident positions undoes the cut exactly — the documented inverse."""
    _box_tm, box_wp = unit_box
    vertices_wp, faces_wp = box_wp.points, box_wp.indices
    creases_wp = od.seams.crease_edges(vertices_wp, faces_wp, angle=30.0)
    cut_vertices_wp, cut_faces_wp = od.seams.cut_along_edges(vertices_wp, faces_wp, creases_wp)
    cut_vertices_wp = cast("wp.array[wp.vec3]", cut_vertices_wp)  # a vec3 input cuts to vec3

    welded_vertices_wp, _unique, _inverse, welded_faces_wp = od.repair.remove_duplicated_vertices(
        cut_vertices_wp, cut_faces_wp, epsilon=1e-6
    )
    assert welded_vertices_wp.size == vertices_wp.size
    assert _face_component_count(welded_vertices_wp, welded_faces_wp) == 1


def _ordered_loop_np(edges_np: np.ndarray) -> list[int]:
    """
    Walk an unordered set of undirected edge rows into one ordered vertex cycle.

    MeshLib's ``cutAlongEdgeLoop`` takes an *ordered* ``EdgeId`` path where ordito's
    ``cut_along_edges`` takes unordered ``(v0, v1)`` rows, so this is the named transform the
    comparison needs -- and it is what any caller of the two would have to write.
    """
    neighbours: dict[int, list[int]] = {}
    for first, second in edges_np.tolist():
        neighbours.setdefault(int(first), []).append(int(second))
        neighbours.setdefault(int(second), []).append(int(first))
    start = int(edges_np[0, 0])
    walk = [start]
    previous = None
    while True:
        candidates = [vertex for vertex in neighbours[walk[-1]] if vertex != previous]
        if not candidates or candidates[0] == start:
            break
        previous, walk = walk[-1], [*walk, candidates[0]]
    return walk


@pytest.mark.parity(
    "cut_along_edges",
    "meshlib",
    benchmarked=False,
    reason="cutAlongEdgeLoop takes an ordered *closed loop* of EdgeIds where cut_along_edges takes "
    "an arbitrary undirected edge set, and the benchmark group's input is exactly that -- 25 % or "
    "100 % of the mesh's edges, which is not a loop and cannot be handed to it at all. Its sibling "
    "cutMesh takes surface contours rather than an edge set, so neither form fits the timed input. "
    "igl.cut_mesh does take the same edge set and carries the timed row; the comparison here is on "
    "the one input shape both accept.",
)
def test_cut_along_edges_matches_meshlib(icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B on the split: the same vertex duplication, from an ordered loop instead of edge rows.

    ``cutAlongEdgeLoop`` walks a closed ``EdgeId`` path and separates the two sides, which is
    ``cut_along_edges``' operation on a loop -- the difference is entirely in how the loop is
    *named*. Two transforms: the rows are ordered into a cycle by [`_ordered_loop_np`], and each
    consecutive pair becomes an ``EdgeId`` through ``MeshTopology.findEdge``, which is the lookup
    the rest of this suite's EdgeId plumbing rests on.

    Measured on the equator of ``icosphere(2)``: a 22-edge loop takes both libraries from 162
    vertices to **184** with the face count unchanged at 320 -- the same 22 duplications, since a
    cut adds one copy per loop vertex and creates no geometry.

    The face count is asserted precisely because it must *not* move: a cut that retriangulated, or
    one that dropped the seam faces, would still plausibly raise the vertex count.
    """
    mesh_tm, mesh_wp = icosphere_coarse
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices

    region_np = mesh_tm.vertices[:, 2] > 0.0
    region_wp = wp.array(
        np.ascontiguousarray(region_np[mesh_tm.faces].all(axis=1)),
        dtype=wp.bool,
        device=vertices_wp.device,
    )
    edges_wp = od.selection.region_boundary_edges(faces_wp, region_wp)
    edges_np = edges_wp.numpy()
    assert edges_wp.shape[0] > 5  # non-vacuity: there is a real seam to cut along

    cut_vertices_wp, cut_faces_wp = od.seams.cut_along_edges(vertices_wp, faces_wp, edges_wp)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    loop_np = _ordered_loop_np(edges_np)
    assert len(loop_np) == edges_wp.shape[0]  # the rows really do form one cycle
    loop_ml = mm.std_vector_Id_EdgeTag()
    for tail, head in zip(loop_np, [*loop_np[1:], loop_np[0]], strict=True):
        loop_ml.append(mesh_ml.topology.findEdge(mm.VertId(tail), mm.VertId(head)))
    mm.cutAlongEdgeLoop(mesh_ml, loop_ml)
    cut_tm = meshlib_to_trimesh(mesh_ml)

    assert cut_tm.vertices.shape[0] == mesh_tm.vertices.shape[0] + edges_wp.shape[0]
    assert cut_vertices_wp.size == cut_tm.vertices.shape[0]
    assert cut_faces_wp.size // 3 == cut_tm.faces.shape[0] == mesh_tm.faces.shape[0]


@pytest.mark.parity("cut_along_edges", "pymeshlab", "igl")
def test_cut_along_edges_matches_pymeshlab_and_igl(
    unit_box: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Cutting every crease of a cube leaves six disconnected quads; against MeshLab and igl.

    Each of the 8 corners is incident to 3 of the 6 quads and every crease through it is cut, so it
    becomes 3 vertices -- **24 in all**, which is minimal -- while the 12 faces and their winding
    are untouched.

    **Class B (topology only) against MeshLab**, which opens the same seams but re-welds some
    vertices: both sides produce 6 face-connected components, 12 faces and the same surface area --
    the whole content of the operation -- but MeshLab emits **32** vertices, carrying 8 redundant
    duplicates, apparently from splitting per face corner and re-welding only some of them. So the
    count is asserted as an inequality in ordito's favour rather than as a match; the topology is
    asserted exactly.

    **Class B (an edge set becomes a per-corner mask) against igl**, which **settles** that
    disagreement. ``igl.cut_mesh(V, F, C)`` takes ``C`` as a ``(n_faces, 3)`` **bool** per-corner
    mask rather than an edge list, so the named transform is to mark ``C[f, i]`` for every
    face-corner whose edge is in ordito's cut set. igl emits **24 vertices, exactly ordito's
    answer**, so MeshLab's extra 8 are its own.

    **The corner numbering is ``(i, i + 1)`` here, unlike ``igl.ears``.** ``cut_mesh``'s edge ``i``
    of face ``f`` is ``(F[f, i], F[f, (i + 1) % 3])`` -- the same convention ordito uses -- where
    ``igl.ears`` inherits ``igl::on_boundary``'s *opposite-vertex* numbering. The two conventions
    coexist inside one library, so the mask is built with the ``(i, i + 1)`` rule and the
    alternative is checked to be wrong rather than assumed: it yields 28 vertices, so a
    convention slip would fail this test rather than pass it.
    """
    box_tm, _box_wp = unit_box
    angle = 30.0
    vertices_np = np.ascontiguousarray(box_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(box_tm.faces, dtype=np.int64)

    vertices_wp, faces_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    creases_wp = od.seams.crease_edges(vertices_wp, faces_wp, angle=angle)
    cut_vertices_wp, cut_faces_wp = od.seams.cut_along_edges(vertices_wp, faces_wp, creases_wp)
    cut_tm = warp_to_trimesh(cut_vertices_wp, cut_faces_wp)
    assert cut_vertices_wp.size == 24
    assert cut_faces_wp.size == faces_wp.size
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == 6
    assert od.validation.is_winding_consistent(cut_faces_wp)
    assert np.isclose(cut_tm.area, box_tm.area, rtol=1e-5)

    meshset_pml = trimesh_to_pymeshlab(box_tm)
    meshset_pml.meshing_cut_along_crease_edges(angledeg=angle)
    mesh_pml = meshset_pml.current_mesh()
    faces_pml = mesh_pml.face_matrix()
    mesh_cut_pml = tm.Trimesh(mesh_pml.vertex_matrix(), faces_pml, process=False)
    components_pml = len(
        tm.graph.connected_components(
            mesh_cut_pml.face_adjacency, nodes=np.arange(faces_pml.shape[0])
        )
    )
    assert cut_faces_wp.size // 3 == faces_pml.shape[0]
    assert components_pml == 6
    assert np.isclose(mesh_cut_pml.area, box_tm.area, rtol=1e-5)
    assert cut_vertices_wp.size < mesh_pml.vertex_number()

    cut_set = {tuple(sorted(pair)) for pair in creases_wp.numpy().tolist()}
    corner_mask_igl = np.array(
        [
            [
                tuple(sorted((int(faces_np[f, i]), int(faces_np[f, (i + 1) % 3])))) in cut_set
                for i in range(3)
            ]
            for f in range(faces_np.shape[0])
        ],
        dtype=bool,
    )
    assert int(corner_mask_igl.sum()) == 2 * len(cut_set), (
        "every cut edge is marked from both sides"
    )
    vertices_cut_igl, faces_cut_igl = map(
        np.asarray, igl.cut_mesh(vertices_np, faces_np, corner_mask_igl)[:2]
    )
    assert vertices_cut_igl.shape[0] == cut_vertices_wp.size
    assert faces_cut_igl.shape[0] == cut_faces_wp.size // 3
    mesh_cut_igl = tm.Trimesh(vertices_cut_igl, faces_cut_igl, process=False)
    assert np.isclose(mesh_cut_igl.area, cut_tm.area, rtol=1e-5)
    # The opposite-vertex convention is genuinely a different answer, so the mask above is a choice.
    opposite_mask_igl = np.array(
        [
            [
                tuple(sorted((int(faces_np[f, (i + 1) % 3]), int(faces_np[f, (i + 2) % 3]))))
                in cut_set
                for i in range(3)
            ]
            for f in range(faces_np.shape[0])
        ],
        dtype=bool,
    )
    assert np.asarray(igl.cut_mesh(vertices_np, faces_np, opposite_mask_igl)[0]).shape[0] == 28


def _spherical_wedge_atlas(vertices_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    """
    Build a ``(3 * n_faces, 2)`` per-corner atlas with one genuine seam: the ``+-pi`` wrap.

    Each face is unwrapped independently -- corners are pulled to within half a period of corner 0
    -- so faces straddling the antimeridian disagree with their neighbours about ``u`` by exactly
    one period. That disagreement is the seam, and it is a *ring*, not the whole mesh.
    """
    centered_np = vertices_np - vertices_np.mean(axis=0)
    corners_np = centered_np[faces_np]
    u_np = np.arctan2(corners_np[..., 1], corners_np[..., 0]) / (2.0 * np.pi) + 0.5
    u_np = u_np - np.round(u_np - u_np[:, :1])
    radius_np = np.linalg.norm(corners_np, axis=-1)
    v_np = np.arccos(np.clip(corners_np[..., 2] / np.maximum(radius_np, 1e-12), -1.0, 1.0)) / np.pi
    return np.stack([u_np, v_np], axis=-1).reshape(-1, 2).astype(np.float32)


def _seam_edges_np(
    texcoords_np: np.ndarray,
    faces_np: np.ndarray,
    face_texcoords_np: np.ndarray,
    match: str = "index",
    tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU port of ``igl::seam_edges``."""
    directed = {
        (int(faces_np[f, i]), int(faces_np[f, (i + 1) % 3])): (f, i)
        for f in range(faces_np.shape[0])
        for i in range(3)
    }
    undirected = {(min(a, b), max(a, b)) for a, b in directed}

    def orientation(a_np: np.ndarray, b_np: np.ndarray, c_np: np.ndarray) -> float:
        return float((a_np - c_np)[0] * (b_np - c_np)[1] - (b_np - c_np)[0] * (a_np - c_np)[1])

    seams, boundaries, foldovers = [], [], []
    for edge in undirected:
        reverse = (edge[1], edge[0])
        if edge not in directed or reverse not in directed:
            boundaries.append(list(directed[edge if edge in directed else reverse]))
            continue
        forwards, backwards = directed[edge], directed[reverse]
        tail = (
            face_texcoords_np[forwards[0], forwards[1]],
            face_texcoords_np[backwards[0], (backwards[1] + 1) % 3],
        )
        head = (
            face_texcoords_np[forwards[0], (forwards[1] + 1) % 3],
            face_texcoords_np[backwards[0], backwards[1]],
        )
        if match == "index":
            matched = tail[0] == tail[1] and head[0] == head[1]
        else:
            matched = all(
                np.linalg.norm(texcoords_np[pair[0]] - texcoords_np[pair[1]]) <= tolerance
                for pair in (tail, head)
            )
        row = [forwards[0], forwards[1], backwards[0], backwards[1]]
        if not matched:
            seams.append(row)
            continue
        a_np, b_np = texcoords_np[tail[0]], texcoords_np[head[0]]
        opposite_forwards = texcoords_np[face_texcoords_np[forwards[0], (forwards[1] + 2) % 3]]
        opposite_backwards = texcoords_np[face_texcoords_np[backwards[0], (backwards[1] + 2) % 3]]
        forward_side = orientation(a_np, b_np, opposite_forwards)
        backward_side = orientation(a_np, b_np, opposite_backwards)
        if (forward_side > 0.0 and backward_side > 0.0) or (
            forward_side < 0.0 and backward_side < 0.0
        ):
            foldovers.append(row)

    def rows(collected: list[list[int]], width: int) -> np.ndarray:
        return np.array(collected, dtype=np.int32).reshape(-1, width)

    return rows(seams, 4), rows(boundaries, 2), rows(foldovers, 4)


def _quad_mesh(device: str) -> tuple[wp.array[wp.int32], np.ndarray]:
    """Two triangles sharing the diagonal ``1-2``: the smallest mesh with an interior edge."""
    faces_np = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return wp.array(faces_np.reshape(-1).copy(), dtype=wp.int32, device=device), faces_np


@pytest.mark.parity("uv_seam_edges", "pymeshlab")
@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "hemisphere", "saddle_graded"])
@pytest.mark.parametrize("include_boundary", [True, False])
def test_uv_seam_vertex_mask_matches_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str, include_boundary: bool
) -> None:
    """
    Class A: ``uv_seam_vertex_mask`` equals MeshLab's texture-seam selection element-wise.

    MeshLab stores per-wedge UVs and so compares coordinates rather than texcoord indices, and it
    folds boundary edges into the seam set -- which is exactly ``match="uv"`` plus
    ``include_boundary=True``. Both closed fixtures (where the two settings agree) and the open one
    (where they must not) are covered, so neither answer of the boolean passes by default.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    wedge_uv_np = _spherical_wedge_atlas(np.asarray(mesh_tm.vertices, dtype=np.float64), faces_np)

    mask_wp = od.seams.uv_seam_vertex_mask(
        mesh_wp.indices,
        points_to_warp_uv(wedge_uv_np, str(mesh_wp.device)),
        include_boundary=include_boundary,
        n_vertices=int(mesh_tm.vertices.shape[0]),
    )
    meshset_pml = wedge_uv_to_pymeshlab(mesh_tm.vertices, faces_np, wedge_uv_np)
    meshset_pml.compute_selection_by_texture_seams_per_vertex()
    selected_pml = meshset_pml.current_mesh().vertex_selection_array()

    # The atlas has a real seam ring, so this is not the trivially-empty comparison.
    assert 0 < int(selected_pml.sum()) < selected_pml.size
    if include_boundary:
        assert np.array_equal(mask_wp.numpy(), selected_pml)
    else:
        # Dropping the boundary can only ever remove marks, and on an open mesh it must remove some.
        assert not (mask_wp.numpy() & ~selected_pml).any()
        assert bool((mask_wp.numpy() != selected_pml).any()) == (not mesh_tm.is_watertight)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "saddle_graded"])
def test_uv_seam_edges_matches_igl_port(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (row-set canonicalization): all three blocks equal the ``igl::seam_edges`` CPU port.

    igl's output order is an ``unordered_set`` walk and ordito's is halfedge order, so both sides
    are lexsorted first -- exact on integer rows. Run with an explicit ``FTC`` so the *index*
    predicate, the one MeshLab cannot express, is the thing under test.

    Each block is pinned to a per-fixture count first, because a block that is empty on both sides
    compares equal and asserts nothing. Measured: 5 seams and 2 foldovers on either fixture, and 0
    boundaries on ``icosahedron`` against 24 on ``hemisphere`` -- a closed mesh has no UV boundary
    to report, so there the emptiness *is* the claim rather than a gap in the comparison.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    wedge_uv_np = _spherical_wedge_atlas(np.asarray(mesh_tm.vertices, dtype=np.float64), faces_np)
    # Pool the wedge UVs into a shared texcoord array, so equal coordinates share an index and the
    # index predicate has something to disagree about.
    texcoords_np, face_texcoords_np = np.unique(wedge_uv_np, axis=0, return_inverse=True)
    face_texcoords_np = face_texcoords_np.reshape(-1, 3).astype(np.int32)

    seams_wp, boundaries_wp, foldovers_wp = od.seams.uv_seam_edges(
        mesh_wp.indices,
        points_to_warp_uv(texcoords_np, str(mesh_wp.device)),
        wp.array(face_texcoords_np.reshape(-1).copy(), dtype=wp.int32, device=str(mesh_wp.device)),
        n_vertices=int(mesh_tm.vertices.shape[0]),
    )
    seams_igl, boundaries_igl, foldovers_igl = _seam_edges_np(
        texcoords_np.astype(np.float32), faces_np, face_texcoords_np
    )

    assert seams_igl.shape[0] > 0
    assert foldovers_igl.shape[0] > 0
    assert (boundaries_igl.shape[0] == 0) == mesh_tm.is_watertight
    assert np.array_equal(lexsort_rows(seams_wp.numpy()), lexsort_rows(seams_igl))
    assert np.array_equal(lexsort_rows(boundaries_wp.numpy()), lexsort_rows(boundaries_igl))
    assert np.array_equal(lexsort_rows(foldovers_wp.numpy()), lexsort_rows(foldovers_igl))


def test_uv_seam_edges_match_uv_ignores_duplicate_indices(device: str) -> None:
    """
    The one case where the two references genuinely disagree: a duplicated *identical* texcoord.

    An atlas that stores the same coordinate twice makes igl's index predicate report a seam that
    is not one; MeshLab, holding no indices at all, cannot. Perturbing one copy past the tolerance
    brings the seam back under both.
    """
    faces_wp, _faces_np = _quad_mesh(device)
    # Corners of the shared edge 1-2 point at duplicate texcoord entries 4 and 5.
    texcoords_np = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    face_texcoords_wp = wp.array(
        np.array([0, 1, 2, 4, 3, 5], dtype=np.int32), dtype=wp.int32, device=device
    )

    seams_index_wp, _b, _f = od.seams.uv_seam_edges(
        faces_wp, points_to_warp_uv(texcoords_np, device), face_texcoords_wp, n_vertices=4
    )
    assert int(seams_index_wp.shape[0]) == 1

    seams_uv_wp, _b, _f = od.seams.uv_seam_edges(
        faces_wp,
        points_to_warp_uv(texcoords_np, device),
        face_texcoords_wp,
        match="uv",
        n_vertices=4,
    )
    assert int(seams_uv_wp.shape[0]) == 0

    moved_np = texcoords_np.copy()
    moved_np[4] += 0.25
    seams_moved_wp, _b, _f = od.seams.uv_seam_edges(
        faces_wp, points_to_warp_uv(moved_np, device), face_texcoords_wp, match="uv", n_vertices=4
    )
    assert int(seams_moved_wp.shape[0]) == 1
    # ...and a tolerance wide enough to swallow the move hides it again.
    seams_tolerant_wp, _b, _f = od.seams.uv_seam_edges(
        faces_wp,
        points_to_warp_uv(moved_np, device),
        face_texcoords_wp,
        match="uv",
        tolerance=1.0,
        n_vertices=4,
    )
    assert int(seams_tolerant_wp.shape[0]) == 0


@pytest.mark.parametrize(
    ("opposite_uv", "n_foldovers"),
    [((0.5, 1.0), 0), ((0.5, -0.5), 1), ((0.5, 0.5), 0)],
    ids=["unfolded", "folded", "collinear"],
)
def test_uv_seam_edges_foldover(
    device: str, opposite_uv: tuple[float, float], n_foldovers: int
) -> None:
    """
    A foldover is matched texcoords whose two opposite corners land on the *same* UV side.

    The three cases sweep the second triangle's free corner across the shared diagonal: clear of it,
    folded back over it, and exactly on it. Collinear is a degenerate UV triangle rather than a
    fold, and igl's strict comparison does not flag it -- so this pins the boundary of the test,
    which a ``>=`` would silently move.
    """
    faces_wp, _faces_np = _quad_mesh(device)
    corner_uv_np = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],  # face 0: vertices 0, 1, 2
            [1.0, 0.0],
            list(opposite_uv),
            [0.5, 0.5],  # face 1: vertices 1, 3, 2
        ],
        dtype=np.float32,
    )
    seams_wp, boundaries_wp, foldovers_wp = od.seams.uv_seam_edges(
        faces_wp, points_to_warp_uv(corner_uv_np, device), n_vertices=4
    )
    assert int(seams_wp.shape[0]) == 0
    assert int(boundaries_wp.shape[0]) == 4
    assert int(foldovers_wp.shape[0]) == n_foldovers


def test_uv_seam_edges_corner_uv_matches_explicit_indices(device: str) -> None:
    """Wedge input is exactly ``match="uv"`` with an identity ``FTC``, and returns the same rows."""
    faces_wp, _faces_np = _quad_mesh(device)
    corner_uv_np = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.6, 1.0]], dtype=np.float32
    )
    identity_wp = wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device=device)

    wedge = od.seams.uv_seam_edges(faces_wp, points_to_warp_uv(corner_uv_np, device), n_vertices=4)
    explicit = od.seams.uv_seam_edges(
        faces_wp, points_to_warp_uv(corner_uv_np, device), identity_wp, match="uv", n_vertices=4
    )
    assert int(wedge[0].shape[0]) == 1  # corner 2 and corner 5 disagree, so the diagonal tears
    for block_wedge, block_explicit in zip(wedge, explicit, strict=True):
        assert np.array_equal(block_wedge.numpy(), block_explicit.numpy())


def test_seam_edge_vertices_boundaries_match_oriented_boundary(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Boundary rows name the same directed edges ``oriented_boundary_edges`` reports."""
    mesh_tm, mesh_wp = hemisphere
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    wedge_uv_np = _spherical_wedge_atlas(np.asarray(mesh_tm.vertices, dtype=np.float64), faces_np)
    _seams_wp, boundaries_wp, _foldovers_wp = od.seams.uv_seam_edges(
        mesh_wp.indices,
        points_to_warp_uv(wedge_uv_np, str(mesh_wp.device)),
        n_vertices=int(mesh_tm.vertices.shape[0]),
    )
    pairs_np = od.seams.seam_edge_vertices(mesh_wp.indices, boundaries_wp).numpy()
    oriented_np = od.boundary.oriented_boundary_edges(mesh_wp.points, mesh_wp.indices).numpy()

    assert len(pairs_np) > 0
    assert np.array_equal(lexsort_rows(pairs_np), lexsort_rows(oriented_np))


def test_seam_edge_vertices_feeds_cut_along_edges(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: the detect-convert-cut loop on an atlas' own seams.

    Seam pairs come out smaller-index-first (the forward halfedge is by definition the one running
    that way), and feeding them to the cut duplicates vertices along the ring while leaving the
    face count, the surface area and the component count untouched.
    """
    mesh_tm, mesh_wp = icosahedron
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    wedge_uv_np = _spherical_wedge_atlas(np.asarray(mesh_tm.vertices, dtype=np.float64), faces_np)
    seams_wp, _boundaries_wp, _foldovers_wp = od.seams.uv_seam_edges(
        mesh_wp.indices,
        points_to_warp_uv(wedge_uv_np, str(mesh_wp.device)),
        n_vertices=int(mesh_tm.vertices.shape[0]),
    )
    pairs_np = od.seams.seam_edge_vertices(mesh_wp.indices, seams_wp).numpy()
    assert len(pairs_np) > 0
    assert (pairs_np[:, 0] < pairs_np[:, 1]).all()

    cut_vertices_wp, cut_faces_wp = od.seams.cut_along_edges(
        mesh_wp.points, mesh_wp.indices, od.seams.seam_edge_vertices(mesh_wp.indices, seams_wp)
    )
    cut_vertices_wp = cast("wp.array[wp.vec3]", cut_vertices_wp)  # a vec3 input cuts to vec3
    cut_tm = warp_to_trimesh(cut_vertices_wp, cut_faces_wp)
    assert cut_faces_wp.size == mesh_wp.indices.size
    assert cut_vertices_wp.size > int(mesh_tm.vertices.shape[0])
    assert np.isclose(cut_tm.area, mesh_tm.area, rtol=1e-5)
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == 1
    # The cut turned the seam ring into a real boundary.
    assert int(od.boundary.boundary_edges(cut_vertices_wp, cut_faces_wp).shape[0]) > 0


def test_uv_seam_edges_single_triangle(device: str) -> None:
    """Every edge is a boundary, and neither seam nor foldover can exist without a second face."""
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    corner_uv_np = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    seams_wp, boundaries_wp, foldovers_wp = od.seams.uv_seam_edges(
        faces_wp, points_to_warp_uv(corner_uv_np, device), n_vertices=3
    )
    assert seams_wp.shape == (0, 4)
    assert foldovers_wp.shape == (0, 4)
    assert np.array_equal(boundaries_wp.numpy(), np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int32))
    assert _sorted_edge_set(od.seams.seam_edge_vertices(faces_wp, boundaries_wp).numpy()) == {
        (0, 1),
        (1, 2),
        (0, 2),
    }


def test_uv_seam_match_rejects_an_off_menu_mode(device: str) -> None:
    """
    Not a library comparison: the ``match`` menu's own guard, on both entry points.

    An unrecognised mode used to reach ``_UV_MATCH_MODES[match]`` and surface as a bare
    ``KeyError('bogus')`` naming neither the argument nor the two modes -- the one place in the
    package where an off-menu value did not raise the ``ValueError`` every other menu argument
    raises. ``uv_seam_vertex_mask`` delegates, so it is checked too: the guard lives in the
    function the delegation lands in, and only a call through the wrapper shows that.
    """
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    corner_uv_wp = points_to_warp_uv(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32), device
    )
    for match in ("bogus", "Index", ""):
        with pytest.raises(ValueError, match="match must be one of"):
            od.seams.uv_seam_edges(faces_wp, corner_uv_wp, n_vertices=3, match=match)  # pyright: ignore[reportArgumentType]  # deliberately off-menu
        with pytest.raises(ValueError, match="match must be one of"):
            od.seams.uv_seam_vertex_mask(faces_wp, corner_uv_wp, n_vertices=3, match=match)  # pyright: ignore[reportArgumentType]  # deliberately off-menu
    # Both documented modes still run; ``index`` needs ``face_texcoords`` and is covered above.
    assert od.seams.uv_seam_edges(faces_wp, corner_uv_wp, n_vertices=3, match="uv")[0].shape == (
        0,
        4,
    )


def test_cut_along_edges_accepts_a_row_in_either_order(
    device: str, unit_box: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: this pins the documented row-order contract against itself.

    ``edges`` is documented as taking a row "in either order", and the halfedge side sorts its own
    key, so a reversed row must cut the same edge. It did not: the wrapper packed the row as given,
    producing a key no halfedge can build, and the edge was **silently skipped** -- the call
    returned the input mesh unchanged rather than raising. Reversing every row and comparing the two
    results is what makes that failure visible.
    """
    _mesh_tm, mesh_wp = unit_box
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    creases_wp = od.seams.crease_edges(vertices_wp, faces_wp, angle=0.5)
    reversed_np = np.ascontiguousarray(creases_wp.numpy()[:, ::-1])
    reversed_wp = odt.as_array2d(wp.array(reversed_np, dtype=wp.int32, device=device), wp.int32)

    ascending_vertices_wp, ascending_faces_wp = od.seams.cut_along_edges(
        vertices_wp, faces_wp, creases_wp
    )
    descending_vertices_wp, descending_faces_wp = od.seams.cut_along_edges(
        vertices_wp, faces_wp, reversed_wp
    )
    # Non-vacuity: the cut has to do something, or both sides agree on having done nothing.
    assert ascending_vertices_wp.size > vertices_wp.size
    assert np.array_equal(ascending_faces_wp.numpy(), descending_faces_wp.numpy())
    assert np.allclose(ascending_vertices_wp.numpy(), descending_vertices_wp.numpy())


def _quad_vertices(device: str) -> wp.array[wp.vec3]:
    """Return the four corners of [`_quad_mesh`], a unit square in the ``z = 0`` plane."""
    corners_np = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    return wp.array(corners_np, dtype=wp.vec3, device=device)


_INVALID_CALLS: dict[str, tuple[Callable[[str], object], str]] = {
    "crease_angle_below_zero": (
        lambda device: od.seams.crease_edges(
            _quad_vertices(device), _quad_mesh(device)[0], angle=-1.0
        ),
        r"angle must be in \[0, 180\]",
    ),
    "cut_edges_not_pairs": (
        lambda device: od.seams.cut_along_edges(
            _quad_vertices(device),
            _quad_mesh(device)[0],
            odt.empty_2d((3, 3), wp.int32, device=device),
        ),
        r"edges must have shape \(k, 2\)",
    ),
    "uv_index_match_without_face_texcoords": (
        lambda device: od.seams.uv_seam_edges(
            _quad_mesh(device)[0],
            points_to_warp_uv(np.zeros((6, 2), dtype=np.float32), device),
            match="index",
        ),
        "match='index' needs face_texcoords",
    ),
    "uv_face_texcoords_not_per_corner": (
        lambda device: od.seams.uv_seam_edges(
            _quad_mesh(device)[0],
            points_to_warp_uv(np.zeros((6, 2), dtype=np.float32), device),
            wp.array(np.arange(3, dtype=np.int32), dtype=wp.int32, device=device),
        ),
        "one entry per face corner",
    ),
    "uv_texcoords_not_per_corner": (
        lambda device: od.seams.uv_seam_edges(
            _quad_mesh(device)[0], points_to_warp_uv(np.zeros((4, 2), dtype=np.float32), device)
        ),
        "texcoords must be per-corner",
    ),
    # Three faces on one edge: "the other side" is undefined, as it is for the cut.
    "uv_nonmanifold": (
        lambda device: od.seams.uv_seam_edges(
            wp.array(
                np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype=np.int32), dtype=wp.int32, device=device
            ),
            points_to_warp_uv(np.zeros((9, 2), dtype=np.float32), device),
            n_vertices=5,
        ),
        "edge-manifold",
    ),
}


@pytest.mark.parametrize("case", list(_INVALID_CALLS))
def test_invalid_arguments_raise(device: str, case: str) -> None:
    """Not a library comparison: each documented ``ValueError``, with its message."""
    call, message = _INVALID_CALLS[case]
    with pytest.raises(ValueError, match=message):
        call(device)


def test_empty_mesh(device: str) -> None:
    """Not a library comparison: every entry point answers a face-less mesh with empty buffers."""
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = warp_empty(0, wp.int32, device)
    assert od.seams.crease_edges(vertices_wp, faces_wp).shape == (0, 2)

    out_vertices_wp, out_faces_wp = od.seams.cut_along_edges(
        vertices_wp, faces_wp, odt.empty_2d((0, 2), wp.int32, device=device)
    )
    assert out_vertices_wp.size == 0
    assert out_faces_wp.size == 0

    seams_wp, boundaries_wp, foldovers_wp = od.seams.uv_seam_edges(
        faces_wp, warp_empty(0, wp.vec2, device), n_vertices=0
    )
    assert seams_wp.shape == (0, 4)
    assert boundaries_wp.shape == (0, 2)
    assert foldovers_wp.shape == (0, 4)
    assert (
        od.seams.uv_seam_vertex_mask(faces_wp, warp_empty(0, wp.vec2, device), n_vertices=0).size
        == 0
    )


def _assert_edge_key_order(rows_np: np.ndarray) -> None:
    """Assert undirected rows strictly ascend by ``(max, min)``, ``edges_unique``'s key order."""
    high = rows_np.max(axis=1).astype(np.int64)
    low = rows_np.min(axis=1).astype(np.int64)
    assert np.all(np.diff(high * (int(high.max(initial=0)) + 1) + low) > 0)
