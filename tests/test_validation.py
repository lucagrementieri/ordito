from __future__ import annotations

import igl
import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import trimesh.repair as tm_repair
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
from tests.comparisons import canonical_winding, undirected_edges
from tests.conftest import CLOSED_MESHES, MESHES, OPEN_MESHES
from tests.conversions import (
    meshlib_bitset_to_numpy,
    numpy_to_meshlib,
    numpy_to_warp,
    points_to_warp,
    pymeshfix_face_remap,
    pymeshfix_intersecting_faces,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshfix,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
    warp_empty,
)


def _faces_igl(mesh_tm: tm.Trimesh) -> np.ndarray:
    """Faces as an ``(n_faces, 3)`` int64 array for the libigl reference functions."""
    return mesh_tm.faces.astype(np.int64)


def _edge_manifold_np(faces_np: np.ndarray, allow_boundary_edges: bool) -> bool:
    """NumPy reference: manifold edge-count check over undirected edges."""
    edges = undirected_edges(faces_np)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if allow_boundary_edges:
        return bool((counts <= 2).all())
    return bool((counts == 2).all())


def _edge_manifold_mask_np(faces_np: np.ndarray, allow_boundary_edges: bool) -> np.ndarray:
    """NumPy reference: per-face flag that all three edges are manifold."""
    edges = undirected_edges(faces_np)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    edge_ok = counts <= 2 if allow_boundary_edges else counts == 2
    return np.asarray(edge_ok[inverse.reshape(-1)].reshape(-1, 3).all(axis=1))


def _orientable_np(faces_np: np.ndarray) -> bool:
    """NumPy reference: Z2 parity over the face-adjacency graph (BFS with a flip bit)."""
    n_faces = faces_np.shape[0]
    edge_to_faces: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for f, tri in enumerate(faces_np):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            forward = 1 if a < b else 0
            key = (min(a, b), max(a, b))
            edge_to_faces.setdefault(key, []).append((f, forward))

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_faces)]
    for shared in edge_to_faces.values():
        if len(shared) != 2:
            continue
        (f0, d0), (f1, d1) = shared
        flip = 1 if d0 == d1 else 0
        adjacency[f0].append((f1, flip))
        adjacency[f1].append((f0, flip))

    orient = [-1] * n_faces
    for seed in range(n_faces):
        if orient[seed] != -1:
            continue
        orient[seed] = 0
        stack = [seed]
        while stack:
            f = stack.pop()
            for nb, flip in adjacency[f]:
                expected = orient[f] ^ flip
                if orient[nb] == -1:
                    orient[nb] = expected
                    stack.append(nb)
                elif orient[nb] != expected:
                    return False
    return True


def _tangled_boxes() -> tm.Trimesh:
    """
    Two unit boxes, one offset by (0.5, 0.5, 0.5), welded at the crossing.

    The module's one transversal self-intersection fixture -- every closed conftest fixture either
    self-intersects tangentially or not at all -- so the five tests that need a clean crossing all
    built this identically before it was factored out here.
    """
    first_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    second_tm = first_tm.copy()
    second_tm.apply_translation([0.5, 0.5, 0.5])
    tangled_tm = tm.util.concatenate([first_tm, second_tm])
    assert isinstance(tangled_tm, tm.Trimesh)
    tangled_tm.merge_vertices()
    return tangled_tm


def _mobius_strip(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangulated Möbius strip (edge- and vertex-manifold, non-orientable)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    vertices = []
    for t in theta:
        for s in (-0.3, 0.3):
            vertices.append(
                [
                    (1.0 + s * np.cos(t / 2.0)) * np.cos(t),
                    (1.0 + s * np.cos(t / 2.0)) * np.sin(t),
                    s * np.sin(t / 2.0),
                ]
            )
    faces = []
    for i in range(n):
        a0, a1 = 2 * i, 2 * i + 1
        if i < n - 1:
            b0, b1 = 2 * (i + 1), 2 * (i + 1) + 1
        else:
            b0, b1 = 1, 0  # swap rails at the seam -> half twist
        faces.append([a0, a1, b1])
        faces.append([a0, b1, b0])
    return np.array(vertices), np.array(faces)


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("is_vertex_manifold", "pymeshlab")
@pytest.mark.parity("is_watertight", "pymeshlab")
@pytest.mark.parity("is_volume", "pymeshlab")
def test_topological_measures_match_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B (one report, four predicates): MeshLab reports the topology in a single dict.

    ``get_topological_measures`` returns edge count, boundary-edge count, component count, genus and
    edge-manifoldness together, so a single call checks the whole family at once and -- unlike the
    per-function trimesh and libigl oracles -- checks them *for mutual consistency* as well. That is
    the value here: an inconsistent set (a genus that does not follow from the Euler characteristic,
    say) is a class of bug no single-quantity comparison can see.

    MeshLab reports a genus for open meshes too, so the genus check is written as the full ``chi =
    2C - 2g - L`` identity (components, genus, boundary loops) rather than the closed-surface ``chi
    = 2 - 2g``. That form is what makes it a *joint* constraint tying four independently computed
    quantities together, and it is the one that catches ``cave_cube``: a hollow shell is two
    components, so its Euler characteristic is 4 and the single-component form would be wrong.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    measures_pml = trimesh_to_pymeshlab(mesh_tm).get_topological_measures()

    assert od.validation.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=True) == bool(
        measures_pml["is_mesh_two_manifold"]
    )

    unique_edges_wp, _ = od.edges.edges_unique(mesh_wp.indices)
    assert int(unique_edges_wp.shape[0]) == int(measures_pml["edges_number"])

    boundary_edges_wp = od.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices)
    assert int(boundary_edges_wp.shape[0]) == int(measures_pml["boundary_edges"])

    n_vertices = mesh_wp.points.size
    labels_np = od.graph.connected_component_labels_from_edges(unique_edges_wp, n_vertices).numpy()
    assert len(np.unique(labels_np)) == int(measures_pml["connected_components_number"])

    assert od.validation.is_vertex_manifold(mesh_wp.indices) == (
        int(measures_pml["non_two_manifold_vertices"]) == 0
    )

    # ``is_watertight`` follows Open3D, which is two-manifold + no boundary + no self-intersection.
    # MeshLab's topology report covers the first two; the third needs a second filter, so the
    # composition below is what ``benchmarks/test_validation.py`` times as its pymeshlab row.
    selfx_meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    selfx_meshset_pml.compute_selection_by_self_intersections_per_face()
    n_self_intersecting = int(selfx_meshset_pml.current_mesh().face_selection_array().sum())
    watertight_pml = (
        bool(measures_pml["is_mesh_two_manifold"])
        and int(measures_pml["boundary_edges"]) == 0
        and int(measures_pml["non_two_manifold_vertices"]) == 0
        and n_self_intersecting == 0
    )
    assert od.validation.is_watertight(mesh_wp.points, mesh_wp.indices) == watertight_pml
    # ``is_volume`` adds consistent winding on top, which is true of every fixture here.
    assert od.validation.is_volume(mesh_wp.points, mesh_wp.indices) == (
        watertight_pml and od.validation.is_winding_consistent(mesh_wp.indices)
    )

    n_loops = len(od.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices))
    assert (mesh_name in CLOSED_MESHES) == (n_loops == 0)
    n_components = int(measures_pml["connected_components_number"])
    euler_wp = od.measures.euler_characteristic(mesh_wp.indices)
    assert euler_wp == 2 * n_components - 2 * int(measures_pml["genus"]) - n_loops


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("is_edge_manifold", "igl")
def test_is_edge_manifold_allow_boundary(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: the reduced verdict against ``igl.is_edge_manifold``'s first return, plus NumPy.

    igl always allows boundary edges, so this is the comparison at ordito's default
    ``allow_boundary_edges=True``; the ``False`` setting has no igl counterpart and is pinned
    against the NumPy reference in the test below.

    The fixture list spans closed and open meshes, so the assert is not a constant: every fixture
    here is edge-manifold, and the *non*-manifold direction -- where a constant ``True`` would fail
    -- is covered by the deliberately non-manifold inputs in
    [`test_edge_manifold_mask`][tests.test_validation.test_edge_manifold_mask] and the two
    single-edge fan cases further down.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    manifold_wp = od.validation.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=True)
    manifold_igl = bool(igl.is_edge_manifold(_faces_igl(mesh_tm))[0])
    manifold_np = _edge_manifold_np(mesh_tm.faces, allow_boundary_edges=True)
    assert manifold_wp == manifold_igl
    assert manifold_wp == manifold_np


@pytest.mark.parametrize("mesh_name", MESHES)
def test_is_edge_manifold_no_boundary(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    manifold_wp = od.validation.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=False)
    manifold_np = _edge_manifold_np(mesh_tm.faces, allow_boundary_edges=False)
    assert manifold_wp == manifold_np
    # Closed meshes have no boundary edges; open surfaces do.
    assert manifold_wp == (mesh_name in CLOSED_MESHES)


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parametrize("allow_boundary_edges", [True, False])
@pytest.mark.parity("is_edge_manifold", "open3d")
def test_is_edge_manifold_matches_open3d(
    request: pytest.FixtureRequest, mesh_name: str, allow_boundary_edges: bool
) -> None:
    """
    Class A: Open3D's ``is_edge_manifold`` shares ordito's switch with identical semantics.

    Unlike igl (which always allows boundary edges), Open3D exposes ``allow_boundary_edges`` with
    the same two meanings as ordito's, so both settings are compared. The fixture list spans
    closed and open meshes, so at ``allow_boundary_edges=False`` both answers appear; the
    non-manifold direction at ``True`` is pinned by the three-faces-one-edge case below.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    manifold_wp = od.validation.is_edge_manifold(
        mesh_wp.indices, allow_boundary_edges=allow_boundary_edges
    )
    manifold_o3d = trimesh_to_open3d(mesh_tm).is_edge_manifold(
        allow_boundary_edges=allow_boundary_edges
    )
    assert manifold_wp == manifold_o3d
    if not allow_boundary_edges:
        assert manifold_o3d == (mesh_name in CLOSED_MESHES)


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("is_edge_manifold", "pyvista")
def test_is_edge_manifold_matches_pyvista(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A at ``allow_boundary_edges=False``, which is the only setting VTK has.

    ``PolyData.is_manifold`` is ``n_open_edges == 0``, and ``n_open_edges`` is ``vtkFeatureEdges``
    with boundary **and** non-manifold edges on -- so it rejects a boundary edge exactly as
    ordito's ``False`` setting does and there is nothing to compare at ``True``. The fixtures span
    closed and open meshes, so both answers appear rather than the assert riding on a constant.

    The *count* does not map even though the predicate does: on three faces sharing one edge
    ``n_open_edges`` reads 7 where ordito counts 6 boundary edges plus 1 non-manifold edge, which
    is why ``tests/test_boundary.py`` compares against ``extract_feature_edges`` instead. The fan
    case below pins the predicate on that same input.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh_pv = trimesh_to_pyvista(mesh_tm)

    manifold_wp = od.validation.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=False)
    assert manifold_wp == bool(mesh_pv.is_manifold)
    assert manifold_wp == (mesh_name in CLOSED_MESHES)
    assert (mesh_pv.n_open_edges == 0) == manifold_wp


def test_is_edge_manifold_nonmanifold_fan_matches_pyvista(device: str) -> None:
    """
    Class B (count convention): three faces on one edge, where pyvista's count differs.

    The second assert is the one worth keeping -- it pins ``n_open_edges == 7`` against ordito's 6
    boundary edges on the same mesh, so the "do not map the count" note in
    ``test_is_edge_manifold_matches_pyvista`` is asserted rather than merely written down.
    """
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces_np = np.array([[0, 1, 2], [0, 3, 1], [0, 1, 4]])
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    mesh_pv = trimesh_to_pyvista(tm.Trimesh(vertices_np, faces_np, process=False))

    assert od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False) is False
    assert bool(mesh_pv.is_manifold) is False
    assert int(mesh_pv.n_open_edges) == 7
    assert int(od.boundary.boundary_edges(vertices_wp, faces_wp).shape[0]) == 6


def test_is_edge_manifold_nonmanifold_fan_matches_open3d(device: str) -> None:
    """Class A: three faces on one edge is non-manifold under both switches, in both libraries."""
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces_np = np.array([[0, 1, 2], [0, 3, 1], [0, 1, 4]])
    _, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    mesh_o3d = trimesh_to_open3d(tm.Trimesh(vertices_np, faces_np, process=False))
    for allow_boundary_edges in (True, False):
        manifold_wp = od.validation.is_edge_manifold(
            faces_wp, allow_boundary_edges=allow_boundary_edges
        )
        assert manifold_wp == mesh_o3d.is_edge_manifold(allow_boundary_edges=allow_boundary_edges)
        assert manifold_wp is False


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parametrize("allow_boundary_edges", [True, False])
def test_edge_manifold_mask(
    request: pytest.FixtureRequest, mesh_name: str, allow_boundary_edges: bool
) -> None:
    """
    Class A: the per-face mask against a numpy edge-multiplicity oracle, over both switches.

    No library exposes a per-face edge-manifold mask, so the reference is written here from the
    edge multiplicity table; the *predicate* it reduces to has real library oracles above.
    Length is asserted too, since a short mask would compare equal on its prefix.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = od.validation.edge_manifold_mask(
        mesh_wp.indices, allow_boundary_edges=allow_boundary_edges
    )
    mask_np = _edge_manifold_mask_np(mesh_tm.faces, allow_boundary_edges)
    assert mask_wp.size == mesh_tm.faces.shape[0]
    assert np.array_equal(mask_wp.numpy(), mask_np)
    # is_edge_manifold is the reduction of the per-face mask.
    assert bool(mask_wp.numpy().all()) == od.validation.is_edge_manifold(
        mesh_wp.indices, allow_boundary_edges=allow_boundary_edges
    )
    if allow_boundary_edges:
        # libigl BF is per-corner; a face is manifold iff all three corners are.
        edge_manifold_bf = np.asarray(igl.is_edge_manifold(_faces_igl(mesh_tm))[1])
        assert np.array_equal(mask_wp.numpy(), edge_manifold_bf.all(axis=1))


def test_is_edge_manifold_precomputed_shortcut(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    n_vertices = od.array.index_bound(mesh_wp.indices)
    assert od.validation.is_edge_manifold(
        mesh_wp.indices, n_vertices=n_vertices
    ) == od.validation.is_edge_manifold(mesh_wp.indices)


@pytest.mark.parametrize("allow_boundary_edges", [True, False])
def test_is_edge_manifold_radix_is_invariant_to_an_oversized_base(
    request: pytest.FixtureRequest, allow_boundary_edges: bool
) -> None:
    """
    Class A: the verdict is unchanged by any hash base above ``max(faces)``.

    Parametrized over both meshes and both switch positions so each answer appears: the closed
    icosahedron is edge-manifold either way, the open hemisphere only with boundary edges allowed.
    Callers holding ``vertices`` pass ``vertices.shape[0]``, which exceeds ``max(faces) + 1``
    whenever the mesh carries unreferenced vertices.
    """
    answers = set()
    for mesh_name in ("icosahedron", "hemisphere"):
        _, mesh_wp = request.getfixturevalue(mesh_name)
        tight = od.array.index_bound(mesh_wp.indices)
        baseline = od.validation.is_edge_manifold(
            mesh_wp.indices, allow_boundary_edges, n_vertices=tight
        )
        answers.add(baseline)
        for base in (tight + 1, tight + 1000):
            assert (
                od.validation.is_edge_manifold(
                    mesh_wp.indices, allow_boundary_edges, n_vertices=base
                )
                == baseline
            )
    assert answers == ({True} if allow_boundary_edges else {True, False})


@pytest.mark.parametrize(("offset", "n_vertices"), [(-100, None), (0, "tight")])
def test_is_edge_manifold_rejects_out_of_range_indices(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], offset: int, n_vertices: str | None
) -> None:
    """
    Not a library comparison: the documented ``ValueError`` from the folded range check.

    The check rides on the key pass: a negative index with no bound given, and an index reaching a
    given bound, both raise, while ``validate=False`` skips it and a valid buffer passes it.
    """
    mesh_tm, mesh_wp = icosahedron
    faces_np = mesh_tm.faces.astype(np.int32).ravel()
    bad_np = faces_np.copy()
    bad_np[4] = offset if offset < 0 else len(mesh_tm.vertices)
    bound = None if n_vertices is None else len(mesh_tm.vertices)
    bad_wp = wp.array(bad_np, dtype=wp.int32, device=mesh_wp.device)
    with pytest.raises(ValueError, match="faces must be non-negative indices"):
        od.validation.is_edge_manifold(bad_wp, n_vertices=bound)
    if offset < 0:
        # Unvalidated, the negative index is packed like any other and the call returns.
        assert isinstance(od.validation.is_edge_manifold(bad_wp, validate=False), bool)
    assert od.validation.is_edge_manifold(mesh_wp.indices, n_vertices=bound)


def test_is_vertex_manifold_precomputed_shortcut(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    adjacency, adjacency_edges = od.adjacency.face_adjacency(mesh_wp.indices, return_edges=True)
    assert od.validation.is_vertex_manifold(
        mesh_wp.indices, face_adjacency=adjacency, face_adjacency_edges=adjacency_edges
    ) == od.validation.is_vertex_manifold(mesh_wp.indices)
    # The half-pair raise now comes from the shared ``adjacency.require_paired_adjacency``, so the
    # message is the one every caller of that resolver reports rather than this module's own.
    with pytest.raises(ValueError, match="both be provided or both omitted"):
        od.validation.is_vertex_manifold(mesh_wp.indices, face_adjacency=adjacency)


def test_is_volume_precomputed_edges(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    edges = od.edges.faces_to_edges(mesh_wp.indices)
    assert od.validation.is_volume(
        mesh_wp.points, mesh_wp.indices, edges=edges
    ) == od.validation.is_volume(mesh_wp.points, mesh_wp.indices)


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("is_vertex_manifold", "igl")
def test_is_vertex_manifold(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (reduce igl's per-vertex mask): the predicate agrees, and reads ``True`` here.

    The second assert is the one that makes this non-vacuous -- ``manifold_wp is True`` --
    because a predicate that always returned ``False`` would still match a reference reduced
    the same way if both were wrong. The ``False`` branch is
    [`test_is_vertex_manifold_bowtie`].
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    manifold_wp = od.validation.is_vertex_manifold(mesh_wp.indices)
    manifold_igl = bool(np.asarray(igl.is_vertex_manifold(_faces_igl(mesh_tm))).all())
    assert manifold_wp == manifold_igl
    assert manifold_wp is True


def test_is_vertex_manifold_bowtie(device: str) -> None:
    """
    Class B: the ``False`` branch, on two triangles meeting at a single apex vertex.

    A bowtie is edge-manifold but not vertex-manifold, so it separates the two predicates
    rather than failing both -- and igl's fan definition agrees here where Open3D's
    connectivity one does not (section 6, and
    [`test_is_vertex_manifold_open3d_agreement_and_divergence`]).
    """
    # Two triangles sharing only the apex vertex 0 -> non-manifold vertex.
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    faces_np = np.array([[0, 1, 2], [0, 3, 4]])
    _, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    manifold_wp = od.validation.is_vertex_manifold(faces_wp)
    manifold_igl = bool(np.asarray(igl.is_vertex_manifold(faces_np.astype(np.int64))).all())
    assert manifold_wp == manifold_igl
    assert manifold_wp is False
    # The bow-tie is still edge-manifold (each edge used once).
    assert od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=True) is True


def test_vertex_manifold_compressed_forest(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ordito against ordito: compressing the corner forest changes no verdict.

    Both corner union-finds (from the sorted keys, and from a precomputed adjacency) compress their
    pre-hooked forest only from ``connected_components.ECL_COMPRESS_FROM`` corners, which no
    fixture reaches, so the threshold is lowered to force it. A bowtie next to a closed
    icosahedron, so the mask holds both answers; the oracles are igl's, in the tests above.
    """
    from ordito.kernels.algorithms import connected_components as kernel_cc

    ico = tm.creation.icosahedron()
    bowtie_np = np.array([[0, 1, 2], [0, 3, 4]]) + len(ico.vertices)
    vertices_np = np.concatenate([ico.vertices, np.zeros((5, 3))])
    faces_np = np.concatenate([ico.faces, bowtie_np])
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    adjacency, adjacency_edges = od.adjacency.face_adjacency(faces_wp, return_edges=True)

    def answers() -> tuple[np.ndarray, bool]:
        mask = od.validation.vertex_manifold_mask(vertices_wp, faces_wp).numpy()
        tabled = od.validation.is_vertex_manifold(
            faces_wp, face_adjacency=adjacency, face_adjacency_edges=adjacency_edges
        )
        return mask, tabled

    plain_mask, plain_tabled = answers()
    monkeypatch.setattr(kernel_cc, "ECL_COMPRESS_FROM", 0)
    compressed_mask, compressed_tabled = answers()
    assert plain_mask.any()
    assert not plain_mask.all()
    assert plain_tabled is False
    assert np.array_equal(compressed_mask, plain_mask)
    assert compressed_tabled == plain_tabled


@pytest.mark.parametrize(("spare", "expected"), [(None, True), ("interior", False)])
@pytest.mark.parity("is_vertex_manifold", "igl")
def test_is_vertex_manifold_unreferenced_vertices(
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh], spare: str | None, expected: bool
) -> None:
    """
    Class B (reduce igl's per-vertex mask): an unreferenced vertex counts only below ``max(faces)``.

    igl sizes its answer by ``F.max() + 1``, so a spare vertex renumbered into the middle of the
    buffer is non-manifold and one past the end is invisible. The predicate never reads that bound
    on the host -- it tests each unreferenced vertex against its successor -- so both arms are asked
    again with ``n_vertices`` at the exact bound and past it, where the extra vertices are trailing
    spares and must not move the answer.
    """
    mesh_tm, mesh_wp = icosphere_coarse
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)
    n = int(mesh_tm.vertices.shape[0])
    if spare == "interior":
        faces_np = np.where(faces_np >= n // 2, faces_np + 1, faces_np)
    n_bound = int(faces_np.max()) + 1
    faces_wp = wp.array(faces_np.astype(np.int32).ravel(), dtype=wp.int32, device=mesh_wp.device)
    manifold_igl = bool(np.asarray(igl.is_vertex_manifold(faces_np)).all())
    assert manifold_igl is expected
    answers = {
        od.validation.is_vertex_manifold(faces_wp, n_vertices=n_vertices)
        for n_vertices in (None, n_bound, n_bound + 7)
    }
    assert answers == {manifold_igl}


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("is_vertex_manifold", "open3d")
def test_is_vertex_manifold_matches_open3d(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: equal on edge-manifold input, which every fixture here is.

    Open3D's ``IsVertexManifold`` tests whether the faces incident to a vertex are *edge-connected
    at all*; ordito and igl require a manifold fan. The two definitions coincide exactly when the
    mesh is edge-manifold, and diverge on vertices sitting on a non-manifold edge -- three faces
    sharing one edge are mutually connected (Open3D: manifold) but not a fan (ordito: not). The
    divergent input class is asserted below so the restriction stays measured, and the bow-tie
    case supplies the ``False`` answer that keeps this comparison non-vacuous.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    assert od.validation.is_edge_manifold(mesh_wp.indices) is True  # the class-B precondition
    manifold_o3d = trimesh_to_open3d(mesh_tm).is_vertex_manifold()
    assert od.validation.is_vertex_manifold(mesh_wp.indices) == manifold_o3d


def test_is_vertex_manifold_open3d_agreement_and_divergence(device: str) -> None:
    """
    Class B, restriction pinned from both sides: Open3D tests connectivity, not a fan.

    On the edge-manifold bow-tie the libraries agree (both ``False``); on the edge-non-manifold
    three-face fan they deliberately diverge (Open3D ``True``, ordito ``False``) because Open3D
    checks connectivity where ordito checks for a fan. If Open3D ever changes its answer here,
    the class-B precondition in the test above stops being the right restriction.
    """
    bow_v = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    bow_f = np.array([[0, 1, 2], [0, 3, 4]])
    _, bow_faces_wp = numpy_to_warp(bow_v, bow_f, device)
    bow_o3d = trimesh_to_open3d(tm.Trimesh(bow_v, bow_f, process=False)).is_vertex_manifold()
    assert bow_o3d is False
    assert od.validation.is_vertex_manifold(bow_faces_wp) == bow_o3d

    fan_v = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    fan_f = np.array([[0, 1, 2], [0, 3, 1], [0, 1, 4]])
    _, fan_faces_wp = numpy_to_warp(fan_v, fan_f, device)
    assert trimesh_to_open3d(tm.Trimesh(fan_v, fan_f, process=False)).is_vertex_manifold() is True
    assert od.validation.is_vertex_manifold(fan_faces_wp) is False


@pytest.mark.parametrize("mesh_name", MESHES)
def test_vertex_manifold_mask(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: the per-vertex mask against ``igl.is_vertex_manifold``, which has the same shape.

    Unlike the edge mask this needs no transform -- igl's answer *is* per-vertex. The final
    assert pins the predicate as the mask's reduction, so the two cannot drift apart.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = od.validation.vertex_manifold_mask(mesh_wp.points, mesh_wp.indices)
    mask_igl = igl.is_vertex_manifold(_faces_igl(mesh_tm))
    assert mask_wp.size == mesh_tm.vertices.shape[0]
    assert np.array_equal(mask_wp.numpy(), mask_igl)
    # is_vertex_manifold is the reduction of the per-vertex mask.
    assert bool(mask_wp.numpy().all()) == od.validation.is_vertex_manifold(mesh_wp.indices)


def test_vertex_manifold_mask_unreferenced(device: str) -> None:
    # Bow-tie (vertex 0 non-manifold) plus a trailing unreferenced vertex 5.
    vertices_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [9.0, 9.0, 9.0],
        ]
    )
    faces_np = np.array([[0, 1, 2], [0, 3, 4]])
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    mask_wp = od.validation.vertex_manifold_mask(vertices_wp, faces_wp)
    expected = np.array([False, True, True, True, True, False])
    assert mask_wp.size == vertices_np.shape[0]
    assert np.array_equal(mask_wp.numpy(), expected)


def test_vertex_manifold_mask_faces_without_vertices(device: str) -> None:
    # Vertices present but no faces: every vertex is unreferenced -> all False.
    vertices_wp = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = warp_empty(0, wp.int32, device)
    mask_wp = od.validation.vertex_manifold_mask(vertices_wp, faces_wp)
    assert np.array_equal(mask_wp.numpy(), np.zeros(4, dtype=bool))


@pytest.mark.parametrize("mesh_name", CLOSED_MESHES)
def test_is_self_intersecting_clean(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    assert od.validation.is_self_intersecting(mesh_wp) is False


def test_is_self_intersecting_crossing(device: str) -> None:
    vertices_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 0.0],
        ]
    )
    faces_np = np.array([[0, 1, 2], [3, 4, 5]])
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)
    assert od.validation.is_self_intersecting(mesh_wp) is True


def test_is_self_intersecting_separated(device: str) -> None:
    vertices_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
            [5.0, 6.0, 5.0],
        ]
    )
    faces_np = np.array([[0, 1, 2], [3, 4, 5]])
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)
    assert od.validation.is_self_intersecting(mesh_wp) is False


@pytest.mark.parametrize("mesh_name", MESHES)
def test_face_self_intersecting_mask_matches_predicate(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: the mask's reduction must equal the predicate.

    The predicate carries the reference comparison (trimesh and MeshLab); what only this can
    check is that the per-face mask and the whole-mesh answer come from the same test.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = od.validation.face_self_intersecting_mask(mesh_wp.points, mesh_wp.indices)
    assert mask_wp.size == mesh_tm.faces.shape[0]
    predicate = od.validation.is_self_intersecting(mesh_wp)
    assert bool(od.reduce.any(mask_wp)) == predicate


@pytest.mark.parametrize("mesh_name", ["boy_surface", "icosahedron", "cave_cube"])
@pytest.mark.parity("face_self_intersecting_mask", "meshlib")
def test_face_self_intersecting_mask_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B, face for face: MeshLib's ``findSelfCollidingTrianglesBS`` with the bitset padded.

    The transform is the padding and nothing else. A MeshLib bitset built by *insertion* is only as
    long as its highest set bit needs -- measured **608** entries for a 640-face mesh whose last
    colliding face is 607, and an **empty** array on a clean mesh -- so it goes through
    [`meshlib_bitset_to_numpy`][tests.conversions.meshlib_bitset_to_numpy], which states the face
    domain. Read raw, the comparison fails by shape on exactly the meshes where it should pass.

    ``touchIsIntersection=False`` is the setting that matches ordito, which flags a pair only when
    Moller's interval test finds a genuine crossing; at ``True`` MeshLib additionally flags coplanar
    contact and reads **253** faces against ordito's 177 on ``boy_surface``. That is the convention
    knob, so it is passed explicitly rather than left at its default.

    Non-vacuous in both directions: ``boy_surface`` is a closed surface that passes through itself,
    where both sides flag 177 of 2 964 faces, and the two closed fixtures flag none. A comparison
    run only on the clean meshes would be ``[] == []``.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_faces = mesh_tm.faces.shape[0]
    mask_wp = od.validation.face_self_intersecting_mask(mesh_wp.points, mesh_wp.indices)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    colliding_ml = mm.findSelfCollidingTrianglesBS(mm.MeshPart(mesh_ml), touchIsIntersection=False)
    mask_ml = meshlib_bitset_to_numpy(colliding_ml, n_faces)

    assert mask_ml.sum() > 0 if mesh_name == "boy_surface" else mask_ml.sum() == 0
    assert np.array_equal(mask_wp.numpy(), mask_ml)
    assert od.validation.is_self_intersecting(mesh_wp) is bool(mask_ml.any())


def test_face_self_intersecting_mask_two_boxes_matches_meshlib(device: str) -> None:
    """
    Class B on the transversal case the fixtures cannot supply: two interpenetrating boxes.

    Every closed fixture either self-intersects along a *tangency* curve (``bohemian_dome``, where
    the two sides disagree -- see
    [`test_face_self_intersecting_mask_tangential_contact_divergence`]) or not at all, so this is
    the only input in the module where two triangles cross cleanly through each other's interior
    and both libraries have to say so. Both flag the same 12 of 24 faces.
    """
    tangled_tm = _tangled_boxes()

    vertices_wp, faces_wp = numpy_to_warp(tangled_tm.vertices, tangled_tm.faces, device)
    mask_wp = od.validation.face_self_intersecting_mask(vertices_wp, faces_wp)

    mesh_ml = trimesh_to_meshlib(tangled_tm)
    colliding_ml = mm.findSelfCollidingTrianglesBS(mm.MeshPart(mesh_ml), touchIsIntersection=False)
    mask_ml = meshlib_bitset_to_numpy(colliding_ml, tangled_tm.faces.shape[0])

    assert int(mask_ml.sum()) == 12  # non-vacuity: the reference found the crossing
    assert np.array_equal(mask_wp.numpy(), mask_ml)


@pytest.mark.parity("face_self_intersecting_mask", "open3d", "pymeshlab")
@pytest.mark.parametrize("kind", ["boxes", "spheres", "clean"])
def test_face_self_intersecting_mask_matches_open3d_and_pymeshlab(device: str, kind: str) -> None:
    """
    Class A for pymeshlab (a per-face bool array), Class B for open3d (pairs reduced to a set).

    These are the fourth and fifth implementations of the predicate, and five is worth having on
    this one because it is the clause ``is_watertight`` reduces and the post-condition
    ``fix_self_intersections`` is verified by -- so a shared bug here would pass three tests in two
    files.

    ``compute_selection_by_self_intersections_per_face`` writes the selection onto
    ``current_mesh()`` and ``face_selection_array()`` reads it back as a bool array, which needs no
    transform at all and matches ordito's mask **exactly on all three inputs**.
    ``get_self_intersecting_triangles`` returns colliding **pairs**, so ``np.unique`` over them is
    the named transform.

    Measured, faces flagged of the input's total:

    | input | ordito | meshlib | pymeshlab | open3d |
    |---|---|---|---|---|
    | two boxes offset (0.5, 0.5, 0.5), 24 faces | 12 | 12 | 12 | **11** |
    | two icosphere(2) offset 0.7, 640 faces | 84 | 84 | 84 | 84 |
    | one icosphere(2), 320 faces | 0 | 0 | 0 | 0 |

    **open3d is one short on the boxes, and that is pinned rather than tolerated.** It is the same
    configuration ``tests/test_intersection.py``'s
    ``test_mesh_collision_pairs_beats_meshlib_on_axis_aligned_boxes`` records: every crossing there
    is between axis-aligned triangles with parallel edges, which a separating-axis narrow phase gets
    wrong and an interval test does not. So open3d's answer is asserted as a strict **subset** on
    that input and as an equality on the other two -- and a future open3d that found all 12 would
    fail here, which is the point of pinning it.

    Worth knowing before generalising: **MeshLib gets these boxes right** (12) although its
    *two-mesh* entry point gets the equivalent pair wrong (5 and 4 of 6 and 6). The single-mesh and
    two-mesh paths are different code, so a divergence measured on one says nothing about the other.

    Non-vacuous in both directions by construction, and the counts are asserted rather than merely
    compared: two of the three inputs intersect and the third does not, so a reference that silently
    returned nothing would fail rather than agree.
    """
    if kind == "boxes":
        mesh_tm = _tangled_boxes()
        n_expected = 12
    elif kind == "spheres":
        first_tm = tm.creation.icosphere(subdivisions=2)
        second_tm = tm.creation.icosphere(subdivisions=2)
        second_tm.apply_translation([0.7, 0.0, 0.0])
        mesh_tm = tm.util.concatenate([first_tm, second_tm])
        assert isinstance(mesh_tm, tm.Trimesh)
        n_expected = 84
    else:
        mesh_tm = tm.creation.icosphere(subdivisions=2)
        n_expected = 0

    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    mask_np = od.validation.face_self_intersecting_mask(vertices_wp, faces_wp).numpy()

    pairs_o3d = np.asarray(trimesh_to_open3d(mesh_tm).get_self_intersecting_triangles())
    faces_o3d = np.unique(pairs_o3d) if pairs_o3d.size else np.empty(0, dtype=np.int64)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_selection_by_self_intersections_per_face()
    mask_pml = np.asarray(meshset_pml.current_mesh().face_selection_array())

    assert int(mask_np.sum()) == n_expected  # non-vacuity, and it pins the fixture
    assert int(mask_pml.sum()) == n_expected
    assert np.array_equal(mask_np, mask_pml)

    ordito_faces = np.flatnonzero(mask_np).astype(np.int64)
    if kind == "boxes":
        # The parallel-edge configuration open3d's narrow phase misses one of.
        assert faces_o3d.size == n_expected - 1
        assert set(faces_o3d.tolist()) < set(ordito_faces.tolist())
    else:
        assert faces_o3d.size == n_expected
        assert np.array_equal(faces_o3d.astype(np.int64), ordito_faces)


@pytest.mark.parity("face_self_intersecting_mask", "pymeshfix")
@pytest.mark.parity(
    "is_self_intersecting",
    "pymeshfix",
    benchmarked=False,
    reason="is_self_intersecting has no benchmark group of its own -- it is the "
    "any() of this mask and is timed as part of is_watertight -- so the "
    "predicate is asserted here alongside the mask it reduces.",
)
@pytest.mark.parametrize("kind", ["boxes", "spheres", "clean"])
def test_face_self_intersecting_mask_matches_pymeshfix(device: str, kind: str) -> None:
    """
    Class B, face for face: ``select_intersecting_triangles`` remapped to the input face order.

    The transform is the remap and nothing else, and it has to be there twice over. The call writes
    its ``n`` face indices into the **flat** prefix of an ``(n, 3)`` buffer and leaves ``2n``
    entries of heap garbage behind them -- measured ``arr.max()`` of 30 751 against 640 faces, a
    value that varies between processes -- so the read goes through
    [`pymeshfix_intersecting_faces`][tests.conversions.pymeshfix_intersecting_faces]. And those
    indices address the buffer ``return_arrays`` gives back, which is a reordering of the input's
    even when nothing was repaired, so they go through
    [`pymeshfix_face_remap`][tests.conversions.pymeshfix_face_remap], which refuses outright if the
    load changed the mesh.

    Where both libraries analyse the mesh they were handed, they agree **exactly**: 12 of 24 faces
    on two interpenetrating boxes, 72 of 640 on two icospheres translated 1.2 apart, and 0 of 320
    on a clean one. That makes this the strongest pymeshfix pair in the suite and the reason it is
    the anchor -- it exercises the converter, the tail drop, the remap and the mask in one assert,
    so a plumbing error here is unambiguous where a derived-scalar comparison would absorb it.

    Non-vacuous in both directions by construction: two of the three inputs intersect and the third
    does not, and the intersecting counts are asserted rather than merely compared, so a reference
    that silently returned nothing would fail.

    ``tris_per_cell`` and ``justproper`` are passed explicitly at values measured to be no-ops
    (10 / 50 / 200 crossed with False / True all return 72 on the sphere pair) rather than left to
    default, so a future wheel that starts honouring either one fails here instead of drifting.
    """
    if kind == "boxes":
        mesh_tm = _tangled_boxes()
        n_expected = 12
    elif kind == "spheres":
        first_tm = tm.creation.icosphere(subdivisions=2)
        second_tm = tm.creation.icosphere(subdivisions=2)
        second_tm.apply_translation([1.2, 0.0, 0.0])
        mesh_tm = tm.util.concatenate([first_tm, second_tm])
        assert isinstance(mesh_tm, tm.Trimesh)
        n_expected = 72
    else:
        mesh_tm = tm.creation.icosphere(subdivisions=2)
        n_expected = 0

    tin_pmf = trimesh_to_pymeshfix(mesh_tm)
    assert tin_pmf.n_faces == mesh_tm.faces.shape[0]  # the remap below needs an untouched load
    faces_pmf = pymeshfix_face_remap(tin_pmf, mesh_tm.faces)[
        pymeshfix_intersecting_faces(tin_pmf, tris_per_cell=50, justproper=False)
    ]

    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    mask_wp = od.validation.face_self_intersecting_mask(vertices_wp, faces_wp)

    assert faces_pmf.size == n_expected  # non-vacuity: the reference answered this input
    assert np.array_equal(np.flatnonzero(mask_wp.numpy()), np.sort(faces_pmf))
    mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)
    assert od.validation.is_self_intersecting(mesh_wp) is (n_expected > 0)


@pytest.mark.parity("face_self_intersecting_mask", "meshlib")
def test_face_self_intersecting_mask_on_an_interpenetrating_torus(
    torus_self_intersecting: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class B, and the regression test for a narrow phase that used to over-report by 2x here.

    A 16x16 torus whose tube is wider than its hole, so its inner wall passes through itself in a
    band -- and, being a regular grid, it is full of **parallel edges**. That is what makes it the
    input this belongs on, and it has caught two separate defects.

    The first: an 11-axis separating-axis narrow phase projects onto edge-edge cross products that
    *vanish* for parallel edges, read the collapsed interval as an overlap, and flagged **128**
    faces where an exact ``float64`` Moller test and MeshLib both find **64** -- the same 64, face
    for face. The second: Moller's interval test in ``float32`` then flagged **62**, missing 4 and
    adding 2 on the tangency where the two walls meet. Running the same test in
    ``float64`` -- the vertices are float32 either way, so only the *decisions* change -- lands on
    **64** exactly.

    So the assert is an equality now, in both directions, which is the strongest form available and
    the one a return to either defect would fail.
    """
    mesh_tm, mesh_wp = torus_self_intersecting
    n_faces = mesh_tm.faces.shape[0]
    mask_wp = od.validation.face_self_intersecting_mask(mesh_wp.points, mesh_wp.indices).numpy()

    colliding_ml = mm.findSelfCollidingTrianglesBS(
        mm.MeshPart(trimesh_to_meshlib(mesh_tm)), touchIsIntersection=False
    )
    mask_ml = meshlib_bitset_to_numpy(colliding_ml, n_faces)

    assert int(mask_ml.sum()) == 64  # non-vacuity, and the number the old code doubled
    assert np.array_equal(mask_wp, mask_ml)
    assert od.validation.is_self_intersecting(mesh_wp) is True


def test_face_self_intersecting_mask_tangential_contact_divergence(
    bohemian_dome: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a parity assert: the input class where ordito and MeshLib disagree, pinned with numbers.

    The Bohemian dome's two sheets meet along a curve they are *tangent* to rather than crossing
    transversally, so which triangles count is decided at the tolerance. Both sides now find
    **161** of 3 042 faces and disagree about **2** of them -- and against an exact ``float64``
    arbiter it is *MeshLib* that has one false positive and one false negative there, while ordito
    matches exactly.

    Two earlier readings of this test were wrong, which is why the history is kept. It first said
    205 against 161 and called it two separating-axis implementations classifying a band
    differently; arbitration showed 42 of those 45 extra faces were **false positives** from a
    degenerate SAT axis. It then recorded 160 against 161 and called the remaining gap genuine
    float32 tangency; running the same interval test in ``float64`` closed it. What is left is a
    2-face disagreement that no longer favours the reference.

    The bound stays at **0.5 %** of the faces, which is what makes this a regression test: either
    defect would take it back over 1 %.
    """
    mesh_tm, mesh_wp = bohemian_dome
    n_faces = mesh_tm.faces.shape[0]
    mask_wp = od.validation.face_self_intersecting_mask(mesh_wp.points, mesh_wp.indices).numpy()

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    colliding_ml = mm.findSelfCollidingTrianglesBS(mm.MeshPart(mesh_ml), touchIsIntersection=False)
    mask_ml = meshlib_bitset_to_numpy(colliding_ml, n_faces)

    assert mask_wp.sum() > 0
    assert mask_ml.sum() > 0
    assert int((mask_wp != mask_ml).sum()) < 0.005 * n_faces


@pytest.mark.parametrize("mesh_name", MESHES)
def test_is_winding_consistent(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on a boolean: agrees with ``Trimesh.is_winding_consistent``, and reads ``True`` here.

    The literal ``is True`` is what keeps this from passing on a constant-``False``
    implementation; [`test_is_winding_consistent_flipped`] supplies the other branch.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    winding_wp = od.validation.is_winding_consistent(mesh_wp.indices)
    assert winding_wp == bool(mesh_tm.is_winding_consistent)
    assert winding_wp is True


def test_is_winding_consistent_flipped(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: the ``False`` branch, half the faces reversed so the winding genuinely conflicts.

    Reversing *every* face would leave the mesh consistent (just inward), which is the trap
    this avoids by flipping alternate faces -- and is what [`test_is_volume_inward_normals`]
    tests instead.
    """
    mesh_tm, mesh_wp = icosahedron
    faces_flipped = mesh_tm.faces.copy()
    faces_flipped[::2] = faces_flipped[::2][:, ::-1]  # reverse winding of half the faces
    _, faces_wp = numpy_to_warp(mesh_tm.vertices, faces_flipped, mesh_wp.device)
    mesh_flipped_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=faces_flipped, process=False)
    winding_wp = od.validation.is_winding_consistent(faces_wp)
    assert winding_wp == bool(mesh_flipped_tm.is_winding_consistent)
    assert winding_wp is False


@pytest.mark.parametrize("mesh_name", MESHES)
def test_edge_winding_consistent_mask_matches_predicate(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = od.validation.edge_winding_consistent_mask(mesh_wp.indices)
    aggregated = bool(od.reduce.all(mask_wp)) if mask_wp.size > 0 else True
    assert aggregated == od.validation.is_winding_consistent(mesh_wp.indices)
    assert aggregated is True


def test_edge_winding_consistent_mask_flags_flipped(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    mesh_tm, mesh_wp = icosahedron
    faces_flipped = mesh_tm.faces.copy()
    faces_flipped[::2] = faces_flipped[::2][:, ::-1]  # reverse winding of half the faces
    _, faces_wp = numpy_to_warp(mesh_tm.vertices, faces_flipped, mesh_wp.device)
    mask_wp = od.validation.edge_winding_consistent_mask(faces_wp)
    assert mask_wp.size > 0
    assert bool(od.reduce.all(mask_wp)) is False


@pytest.mark.parametrize("mesh_name", MESHES)
def test_is_orientable_fixtures(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    orientable_wp = od.validation.is_orientable(mesh_wp.indices)
    orientable_np = _orientable_np(mesh_tm.faces)
    assert orientable_wp == orientable_np
    assert orientable_wp is True


def test_is_orientable_flip_invariance(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, _ = icosahedron
    faces_flipped = mesh_tm.faces.copy()
    faces_flipped[::2] = faces_flipped[::2][:, ::-1]  # reverse winding of half the faces
    faces_wp = wp.array(faces_flipped.reshape(-1).astype(np.int32), dtype=wp.int32)
    # Orientability is flip-invariant even though the winding is now inconsistent.
    assert od.validation.is_orientable(faces_wp) is True
    assert _orientable_np(faces_flipped) is True


def test_is_orientable_mobius(device: str) -> None:
    vertices_np, faces_np = _mobius_strip(12)
    _, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    assert od.validation.is_orientable(faces_wp) is False
    assert _orientable_np(faces_np) is False
    # The Möbius strip is still edge- and vertex-manifold.
    assert od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=True) is True
    assert od.validation.is_vertex_manifold(faces_wp) is True


def test_is_orientable_closed_non_orientable(boy_surface: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A on the ``False`` branch: Boy's surface is closed *and* non-orientable.

    Every other ``False`` input in this file has a boundary, so the flood-fill always had an edge to
    stop at; here it wraps all the way round and must still find the contradiction. That the mesh is
    simultaneously watertight and non-orientable is the whole point of the fixture, so it is
    asserted rather than assumed.
    """
    mesh_tm, mesh_wp = boy_surface
    assert mesh_tm.is_watertight
    assert od.validation.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=False) is True
    assert od.validation.is_orientable(mesh_wp.indices) is False
    assert _orientable_np(mesh_tm.faces) is False
    assert od.validation.is_winding_consistent(mesh_wp.indices) is False


@pytest.mark.parametrize("mesh_name", ["icosahedron", "mobius", "boy_surface"])
@pytest.mark.parametrize("flip_half", [False, True])
def test_orientation_predicates_with_vertex_bound_match_without(
    request: pytest.FixtureRequest, mesh_name: str, flip_half: bool
) -> None:
    """
    Ordito against ordito: ``n_vertices=`` only narrows the key sort; no answer moves.

    The oracle-carrying path is the one without the bound (the tests above). Flipping half the
    faces makes ``is_winding_consistent`` ``False`` and the flip mask non-trivial, and the two
    non-orientable fixtures carry the ``False`` branch of ``is_orientable``, so neither boolean is
    compared on one answer only.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = mesh_tm.faces.copy()
    if flip_half:
        faces_np[::2] = faces_np[::2][:, ::-1]
    faces_wp = wp.array(
        faces_np.reshape(-1).astype(np.int32), dtype=wp.int32, device=mesh_wp.device
    )
    n_vertices = mesh_tm.vertices.shape[0]

    winding = od.validation.is_winding_consistent(faces_wp)
    assert od.validation.is_winding_consistent(faces_wp, n_vertices=n_vertices) == winding
    assert winding is (not flip_half and mesh_name == "icosahedron")
    orientable = od.validation.is_orientable(faces_wp)
    assert od.validation.is_orientable(faces_wp, n_vertices=n_vertices) == orientable
    assert orientable is (mesh_name == "icosahedron")
    if not orientable:
        # On a non-orientable mesh the parity hooks race on CUDA, so the best-effort flip mask
        # differs between two runs of the unbounded path alone; only the verdicts compare there.
        return
    flips_np = od.validation.face_flip_mask(faces_wp).numpy()
    assert flips_np.any() == flip_half
    assert np.array_equal(
        od.validation.face_flip_mask(faces_wp, n_vertices=n_vertices).numpy(), flips_np
    )
    assert np.array_equal(
        od.repair.make_winding_consistent(faces_wp, n_vertices=n_vertices).numpy(),
        od.repair.make_winding_consistent(faces_wp).numpy(),
    )


@pytest.mark.parametrize("mesh_name", MESHES)
def test_face_flip_mask_all_false_on_consistent(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Not a library comparison: on a consistently-wound mesh no face needs flipping.

    An all-``False`` answer, which is only a claim because the fixtures are known consistent;
    the non-trivial branch is exercised by the ``face_orientation_bits`` tests on the non-
    orientable fixtures.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = od.validation.face_flip_mask(mesh_wp.indices)
    assert mask_wp.size == mesh_tm.faces.shape[0]
    # Fixtures are consistently wound, so no face needs flipping.
    assert bool(od.reduce.any(mask_wp)) is False


def _triangle_ribbon(n_quads: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat triangle strip: the face-adjacency graph is a path of ``2 * n_quads`` nodes."""
    x = np.arange(n_quads + 1, dtype=np.float64)
    vertices = np.empty((2 * (n_quads + 1), 3))
    vertices[0::2] = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    vertices[1::2] = np.column_stack((x, np.ones_like(x), np.zeros_like(x)))
    i = np.arange(n_quads)
    faces = np.empty((2 * n_quads, 3), dtype=np.int64)
    faces[0::2] = np.column_stack((2 * i, 2 * i + 1, 2 * i + 2))
    faces[1::2] = np.column_stack((2 * i + 1, 2 * i + 3, 2 * i + 2))
    return vertices, faces


@pytest.mark.parity("face_orientation_bits", "igl")
def test_face_flip_mask_matches_igl(device: str) -> None:
    """
    Class B: igl's flip mask is *derived* from its reoriented face table, not returned.

    ``igl.bfs_orient(F)`` returns ``(FF, C)`` and ``C`` is the per-face **component id** -- all
    zeros on a connected mesh -- which is the trap this test exists to pin: comparing ordito's bits
    against ``C`` would compare them against a constant and pass for the wrong reason. The named
    transform is therefore *recover the mask*: a face was flipped iff its row in ``FF`` is the
    reversal of its row in ``F``, which the first assert checks is the only possibility (every row
    is either identical or reversed, never a different triangle).

    With that mask recovered the two agree **exactly, with no global sign fix**: both libraries
    anchor each component on its lowest-numbered face, so the answer is unique rather than
    determined up to a per-component flip. The second half of the test pins the same statement one
    level up, on the repaired face table, through
    [`tests.comparisons.canonical_winding`][] -- a flip is emitted as a rotation of the reversed
    triangle, so the rows are compared cyclically.
    """
    vertices_np, faces_np = _triangle_ribbon(1024)
    rng = np.random.default_rng(11)
    scrambled_np = rng.random(faces_np.shape[0]) < 0.5
    flipped_np = faces_np.copy()
    flipped_np[scrambled_np] = flipped_np[scrambled_np][:, ::-1]
    _, faces_wp = numpy_to_warp(vertices_np, flipped_np, device)

    oriented_igl, components_igl = igl.bfs_orient(np.ascontiguousarray(flipped_np, dtype=np.int64))
    oriented_igl = np.asarray(oriented_igl)
    unchanged_igl = (oriented_igl == flipped_np).all(axis=1)
    reversed_igl = (oriented_igl == flipped_np[:, ::-1]).all(axis=1)
    assert bool((unchanged_igl | reversed_igl).all()), "a row is neither kept nor reversed"
    assert np.unique(components_igl).size == 1, "the ribbon is one component"

    mask_wp = od.validation.face_flip_mask(faces_wp).numpy()
    repaired_np = od.repair.make_winding_consistent(faces_wp).numpy().reshape(-1, 3)

    assert np.array_equal(mask_wp, ~unchanged_igl)
    assert np.array_equal(canonical_winding(repaired_np), canonical_winding(oriented_igl))


@pytest.mark.parity("face_orientation_bits", "trimesh")
def test_face_flip_mask_long_path(device: str) -> None:
    """
    A ribbon whose face-adjacency graph is a path of 8 192 nodes.

    This is the deep-propagation case: an orientation flood fill needs one round per node here,
    while the parity union-find behind
    [`face_orientation_bits`][ordito.validation.face_orientation_bits] is depth-independent. The
    answer is unique because the component representative is the smallest face id, so face 0 keeps
    its winding and every other face is determined relative to it.

    Class B against ``trimesh.repair.fix_winding``, the flood fill the benchmark times. Two named
    transforms. The reference *rewrites the mesh* rather than returning bits, so ordito's bits are
    applied via [`make_winding_consistent`][ordito.repair.make_winding_consistent] to compare the
    two rewritten face arrays; and a flip is emitted as a rotation of the reversed triangle, so both
    go through [`tests.comparisons.canonical_winding`][] -- which is insensitive to the starting
    corner and still sensitive to the orientation under test. No global sign fix is needed: both
    libraries anchor on the lowest-numbered face of each component, so the answer is unique.
    """
    vertices_np, faces_np = _triangle_ribbon(4096)
    rng = np.random.default_rng(7)
    scrambled_np = rng.random(faces_np.shape[0]) < 0.5
    flipped_np = faces_np.copy()
    flipped_np[scrambled_np] = flipped_np[scrambled_np][:, ::-1]
    _, faces_wp = numpy_to_warp(vertices_np, flipped_np, device)

    assert od.validation.is_orientable(faces_wp) is True
    bits_wp, _signed_edges_wp, _signs_wp, _m = od.validation.face_orientation_bits(faces_wp)
    mask_np = od.validation.face_flip_mask(faces_wp).numpy()
    assert np.array_equal(mask_np, bits_wp.numpy().astype(bool))
    assert bool(mask_np[0]) is False
    assert np.array_equal(mask_np, scrambled_np != scrambled_np[0])

    # Applying the mask must reproduce the original strip, up to the global flip face 0 anchors.
    # Compared as cyclic windings: a flip is emitted as a rotation of the reversed triangle.
    repaired_np = od.repair.make_winding_consistent(faces_wp).numpy().reshape(-1, 3)
    expected_np = faces_np[:, ::-1] if scrambled_np[0] else faces_np
    assert np.array_equal(canonical_winding(repaired_np), canonical_winding(expected_np))
    assert od.validation.is_winding_consistent(faces_wp) is False

    # trimesh reference: its fix_winding is a flood fill over the same face-adjacency graph, and
    # agrees face for face (both anchor on the first face of the component).
    mesh_tm = tm.Trimesh(vertices_np, flipped_np, process=False)
    tm.repair.fix_winding(mesh_tm)
    assert np.array_equal(canonical_winding(mesh_tm.faces), canonical_winding(repaired_np))
    # ... and so the bits themselves are the faces trimesh chose to flip.
    assert np.array_equal(
        mask_np, np.any(canonical_winding(mesh_tm.faces) != canonical_winding(flipped_np), axis=1)
    )


@pytest.mark.parametrize(
    ("mesh_name", "orientable"),
    [("icosahedron", True), ("hemisphere", True), ("mobius", False), ("boy_surface", False)],
)
def test_face_orientation_bits_leave_edges_unsatisfied_only_when_non_orientable(
    request: pytest.FixtureRequest, mesh_name: str, orientable: bool
) -> None:
    """
    The Z2 potential is exact iff the mesh is orientable, which is the branch nothing else reached.

    ``face_orientation_bits`` returns ``orient`` together with the signed face-adjacency graph
    ``(signed_edges, signs)`` it was solved from, and the contract between them is
    ``orient[u] ^ orient[v] == sign`` on every adjacency edge. On an orientable mesh that system is
    consistent and the potential satisfies all of it; on a non-orientable one no potential exists,
    so the solver necessarily leaves some edges **frustrated** -- and the count of those is what
    ``is_orientable`` is really reading.

    Both of its existing tests build a ``_triangle_ribbon`` and open by asserting
    ``is_orientable is True``, so the whole ``False`` side of this primitive -- the engine under
    ``is_orientable`` *and* under ``make_winding_consistent`` -- had never been executed. The two
    non-orientable fixtures are the only inputs in the suite that can execute it.

    Measured, and the separation is not marginal: 0 frustrated edges of 480 on ``icosphere`` and 30
    of 30 on ``icosahedron``, against **41** of 4 524 on ``mobius`` and **42** of 4 446 on
    ``boy_surface``. Asserting the direction rather than the count, since the count is a property of
    the particular triangulation.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    orient_wp, signed_edges_wp, signs_wp, m = od.validation.face_orientation_bits(mesh_wp.indices)

    assert m > 0, "a mesh with no face adjacency would make this vacuous"
    orient_np = orient_wp.numpy()
    edges_np = signed_edges_wp.numpy()
    frustrated_np = (orient_np[edges_np[:, 0]] ^ orient_np[edges_np[:, 1]]) != signs_wp.numpy()

    assert bool(od.validation.is_orientable(mesh_wp.indices)) is orientable
    assert (int(frustrated_np.sum()) == 0) is orientable


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("is_watertight", "open3d")
def test_is_watertight_matches_open3d(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A, and a definitional claim: this follows Open3D's ``IsWatertight``, not trimesh's.

    Worth a test of its own because the docstring makes that claim explicitly and nothing checked
    it. The two definitions genuinely differ -- trimesh's ``is_watertight`` is edge-manifoldness
    alone, while Open3D additionally requires vertex-manifoldness *and* the absence of
    self-intersections -- so the trimesh row in ``benchmarks/test_validation.py`` is timing context
    rather than an oracle, and open3d is the one that pins the semantics.

    Parametrized over closed and open fixtures so both answers appear; a boolean asserted only over
    watertight meshes would pass on a function that returned ``True`` unconditionally.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    watertight_o3d = trimesh_to_open3d(mesh_tm).is_watertight()

    assert od.validation.is_watertight(mesh_wp.points, mesh_wp.indices) == watertight_o3d
    assert watertight_o3d == (mesh_name in CLOSED_MESHES)


def test_is_watertight_rejects_self_intersection_like_open3d(device: str) -> None:
    """
    Class A: the self-intersection clause, the half of Open3D's definition trimesh lacks.

    Two interpenetrating unit boxes are closed, edge-manifold and vertex-manifold, so every test
    above passes them and ``trimesh.is_watertight`` calls them watertight. Open3D does not, and
    neither may ordito -- this is the only case in the module that separates the two definitions,
    and without it the parametrized test above would be satisfied by an edge-manifold check alone.
    """
    tangled_tm = _tangled_boxes()

    vertices_wp = points_to_warp(tangled_tm.vertices, device)
    faces_wp = wp.array(
        np.ascontiguousarray(tangled_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )

    assert trimesh_to_open3d(tangled_tm).is_watertight() is False
    assert od.validation.is_watertight(vertices_wp, faces_wp) is False
    # The clause that does the work: it *is* edge-manifold and closed, which is all trimesh checks.
    assert od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False)
    assert bool(tangled_tm.is_watertight) is True


@pytest.mark.parametrize("mesh_name", ["bohemian_dome", "boy_surface"])
@pytest.mark.parity("is_watertight", "open3d")
def test_is_watertight_rejects_a_connected_surface_that_intersects_itself(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A against open3d, on the self-intersection case the two-box test cannot reach.

    The test above builds its counterexample from **two** interpenetrating boxes, so a hypothetical
    implementation that rejected any multi-component input would pass it for the wrong reason. These
    two fixtures are single connected surfaces that pass through themselves -- a Bohemian dome and
    Boy's surface -- and they are the only inputs in the suite of that class, which is what their
    docstrings in ``tests/conftest.py`` say they are for.

    Non-vacuous, and deliberately so: every precondition is asserted first, so the ``False`` can
    only be coming from the self-intersection clause. Both are edge- and vertex-manifold with
    **zero** boundary edges, so the manifold and closedness clauses are satisfied and only the third
    can fail.

    Sharpest as a disagreement: ``trimesh.is_watertight`` returns **True** for both, because its
    definition is "every edge has exactly two faces" and stops there. ordito follows Open3D, and
    the assert pins the two of them together against trimesh rather than merely restating a
    convention.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    assert od.validation.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=False) is True
    assert od.validation.is_vertex_manifold(mesh_wp.indices) is True
    assert int(od.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices).shape[0]) == 0
    assert od.validation.is_self_intersecting(mesh_wp) is True

    assert od.validation.is_watertight(mesh_wp.points, mesh_wp.indices) is False
    assert trimesh_to_open3d(mesh_tm).is_watertight() is False
    assert (
        mesh_tm.is_watertight is True
    )  # trimesh's weaker definition, and why open3d is the oracle


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity(
    "is_watertight",
    "meshlib",
    benchmarked=False,
    reason="MeshTopology.isClosed reads closedness off half-edge state the mesh already holds, "
    "so a timed row would price a cached lookup against ordito's edge census plus "
    "self-intersection pass; it is the oracle for the closedness clause only.",
)
def test_is_watertight_closedness_clause_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: ``MeshTopology.isClosed`` is the *closedness* clause of ordito's definition.

    The named transform is which clause is being compared. ordito follows Open3D -- edge-manifold
    without boundary, vertex-manifold, and no self-intersection -- where ``isClosed`` answers only
    "every edge has two faces", trimesh's weaker definition. On these fixtures, none of which
    self-intersects, the three definitions coincide and the comparison is exact in both directions.

    Where they part is asserted here rather than left to the docstring, on the same two
    interpenetrating boxes [`test_is_watertight_rejects_self_intersection_like_open3d`] uses:
    ``isClosed`` reads **True** and ordito reads ``False``. So MeshLib is the oracle for the
    closedness clause and open3d stays the oracle for the composite predicate.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh_ml = trimesh_to_meshlib(mesh_tm)

    closed_ml = mesh_ml.topology.isClosed()
    assert closed_ml == (mesh_name in CLOSED_MESHES)
    assert od.validation.is_watertight(mesh_wp.points, mesh_wp.indices) == closed_ml
    assert od.validation.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=False) == closed_ml

    tangled_tm = _tangled_boxes()
    vertices_wp, faces_wp = numpy_to_warp(tangled_tm.vertices, tangled_tm.faces, mesh_wp.device)
    # The clause MeshLib does not carry: closed, and still not watertight under Open3D's definition.
    assert trimesh_to_meshlib(tangled_tm).topology.isClosed() is True
    assert od.validation.is_watertight(vertices_wp, faces_wp) is False


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("is_watertight", "trimesh")
def test_is_watertight(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on a boolean, against *both* the fixture class and trimesh, which is the point.

    Open3D's definition adds a self-intersection clause trimesh lacks, so the two agree only on
    non-self-intersecting input -- these fixtures. Where they diverge is pinned separately by
    [`test_is_watertight_rejects_self_intersection_like_open3d`].
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    watertight_wp = od.validation.is_watertight(mesh_wp.points, mesh_wp.indices)
    assert watertight_wp == (mesh_name in CLOSED_MESHES)
    # These fixtures are not self-intersecting, so Open3D's composite definition agrees with
    # trimesh's "every edge shared by exactly two faces" check.
    assert watertight_wp == bool(mesh_tm.is_watertight)


@pytest.mark.parametrize("mesh_name", [*MESHES, "mobius", "boy_surface", "bohemian_dome"])
def test_is_watertight_calls_a_mesh_factory_only_past_the_manifold_tests(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Ordito against ordito: a ``mesh=`` callable gives the answer a ``wp.Mesh`` does, lazily.

    ``test_is_watertight`` carries the oracle. The factory must be called exactly when both
    manifold tests pass -- the edge test stops ``mobius`` and the open fixtures before it, and the
    closed ``boy_surface`` / ``bohemian_dome`` reach the broad phase -- and at most once.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    calls: list[int] = []

    def factory() -> wp.Mesh:
        calls.append(1)
        return mesh_wp

    answer = od.validation.is_watertight(mesh_wp.points, mesh_wp.indices, mesh=factory)
    assert answer == od.validation.is_watertight(mesh_wp.points, mesh_wp.indices, mesh=mesh_wp)
    reaches_broad_phase = od.validation.is_edge_manifold(
        mesh_wp.indices, allow_boundary_edges=False
    ) and od.validation.is_vertex_manifold(mesh_wp.indices)
    assert len(calls) == int(reaches_broad_phase)


@pytest.mark.parametrize("mesh_name", MESHES)
def test_face_watertight_mask_matches_reference(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = od.validation.face_watertight_mask(mesh_wp.indices)
    mask_np = _edge_manifold_mask_np(mesh_tm.faces, allow_boundary_edges=False)
    assert np.array_equal(mask_wp.numpy(), mask_np)
    # Equivalent to edge_manifold_mask with boundary edges disallowed.
    edge_mask_wp = od.validation.edge_manifold_mask(mesh_wp.indices, allow_boundary_edges=False)
    assert np.array_equal(mask_wp.numpy(), edge_mask_wp.numpy())


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_face_watertight_mask_broken_faces_reference(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B (complement, then sort): trimesh reports the *broken* faces, this the good ones.

    Two named transforms -- invert the mask to indices, and sort both sides, since neither
    library defines the order.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = od.validation.face_watertight_mask(mesh_wp.indices)
    # Faces breaking watertightness are the complement of the mask (trimesh's broken_faces).
    broken_ours = np.flatnonzero(~mask_wp.numpy())
    broken_tm = np.asarray(tm_repair.broken_faces(mesh_tm), dtype=np.int64)
    assert np.array_equal(np.sort(broken_ours), np.sort(broken_tm))


@pytest.mark.parametrize("mesh_name", MESHES)
@pytest.mark.parity("is_volume", "trimesh")
def test_is_volume(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on a boolean: trimesh's conjunction, cross-checked against the fixture's own class.

    Asserting both means a predicate agreeing with trimesh for the wrong reason still has to
    agree with the fixture table, which is maintained independently.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    volume_wp = od.validation.is_volume(mesh_wp.points, mesh_wp.indices)
    assert volume_wp == bool(mesh_tm.is_volume)
    assert volume_wp == (mesh_name in CLOSED_MESHES)


def test_is_volume_inward_normals(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: reversing every face keeps it watertight and consistent, but flips the volume's sign.

    This is the clause that distinguishes ``is_volume`` from ``is_watertight``, and the only
    fixture state that isolates it -- the mesh passes every other check in the module.
    """
    mesh_tm, mesh_wp = icosahedron
    faces_inward = mesh_tm.faces[:, ::-1].copy()  # reverse every face -> inward-facing normals
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, faces_inward, mesh_wp.device)
    mesh_inward_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=faces_inward, process=False)
    # Still watertight and winding-consistent, but the enclosed signed volume is negative.
    volume_wp = od.validation.is_volume(vertices_wp, faces_wp)
    assert volume_wp == bool(mesh_inward_tm.is_volume)
    assert volume_wp is False


def test_empty_mesh(device: str) -> None:
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = warp_empty(0, wp.int32, device)
    assert od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=True) is True
    assert od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False) is True
    assert od.validation.is_vertex_manifold(faces_wp) is True
    assert od.validation.is_orientable(faces_wp) is True
    assert od.validation.is_winding_consistent(faces_wp) is True
    assert od.validation.is_volume(vertices_wp, faces_wp) is False
    assert od.measures.euler_characteristic(faces_wp) == 0
    assert od.validation.edge_manifold_mask(faces_wp).size == 0
    assert od.validation.vertex_manifold_mask(vertices_wp, faces_wp).size == 0


def test_is_self_intersecting_fewer_than_two_faces(device: str) -> None:
    # A `warp.Mesh` with zero triangles corrupts CUDA state when its BVH is built (a Warp 1.18
    # bug independent of ordito), so this exercises the n_faces < 2 short-circuit with a
    # single-triangle mesh instead of a fully empty one.
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces_np = np.array([[0, 1, 2]])
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)
    assert od.validation.is_self_intersecting(mesh_wp) is False


def test_new_masks_empty_mesh(device: str) -> None:
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = warp_empty(0, wp.int32, device)
    assert od.validation.edge_winding_consistent_mask(faces_wp).size == 0
    assert od.validation.face_self_intersecting_mask(vertices_wp, faces_wp).size == 0
    assert od.validation.face_watertight_mask(faces_wp).size == 0
    assert od.validation.face_flip_mask(faces_wp).size == 0


@pytest.mark.parametrize("mesh_name", MESHES)
def test_supplied_mesh_gives_the_same_answer(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    ``mesh=`` spares the broad phase a BVH build and must change nothing; asserted exactly.

    Both entry points that build one: the per-face mask and the whole-mesh predicate it feeds.
    Non-vacuous over the fixture set, which spans watertight (icosahedron, cave_cube) and open
    (hemisphere, half_torus) meshes, so ``is_watertight`` returns both answers across the
    parametrization rather than one constant.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    prebuilt_wp = wp.Mesh(points=mesh_wp.points, indices=mesh_wp.indices)

    assert np.array_equal(
        od.validation.face_self_intersecting_mask(mesh_wp.points, mesh_wp.indices).numpy(),
        od.validation.face_self_intersecting_mask(
            mesh_wp.points, mesh_wp.indices, mesh=prebuilt_wp
        ).numpy(),
    )
    assert od.validation.is_watertight(
        mesh_wp.points, mesh_wp.indices
    ) == od.validation.is_watertight(mesh_wp.points, mesh_wp.indices, mesh=prebuilt_wp)


def test_face_defective_mask_flags_the_thin_face(
    device: str, t_vertex_patch: tuple[np.ndarray, np.ndarray]
) -> None:
    vertices_np, faces_np = t_vertex_patch
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    bad_np = od.validation.face_defective_mask(vertices_wp, faces_wp, min_quality=0.2).numpy()
    # The sliver (1, 4, 2) is the last face, and it is the only thin one.
    assert bad_np[-1]
    assert bad_np.sum() == 1


def test_face_defective_mask_flags_the_fold(
    device: str, folded_patch: tuple[np.ndarray, np.ndarray]
) -> None:
    """Only the *culprit* of a fold is flagged, not the good face on the other side of the edge."""
    vertices_np, faces_np = folded_patch
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    folded_np = od.validation.face_defective_mask(
        vertices_wp, faces_wp, min_quality=None, max_fold_angle=160.0
    ).numpy()
    assert np.array_equal(folded_np.astype(bool), np.array([False, False, True]))


def test_face_defective_mask_flags_the_misoriented_face(device: str) -> None:
    """One face wound the wrong way in a consistent patch reads 180 degrees off the consensus."""
    n = 5
    i_grid, j_grid = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    vertices_np = np.column_stack(
        [i_grid.ravel().astype(np.float64), j_grid.ravel().astype(np.float64), np.zeros(n * n)]
    )
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    faces_np = np.ascontiguousarray(faces, dtype=np.int32)
    # A grid rather than a three-triangle strip: the criterion compares a face against the *sum* of
    # its neighbours' normals, and on a strip the flipped face's own neighbours have only it to
    # agree with, so they would be flagged too.
    target = faces_np.shape[0] // 2
    faces_np[target] = faces_np[target][::-1]

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    bad_np = od.validation.face_defective_mask(
        vertices_wp, faces_wp, min_quality=None, max_normal_angle=60.0
    ).numpy()
    assert np.array_equal(np.flatnonzero(bad_np), np.array([target]))


def test_face_defective_mask_all_criteria_disabled(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    bad_np = od.validation.face_defective_mask(
        mesh_wp.points, mesh_wp.indices, min_quality=None, max_normal_angle=None
    ).numpy()
    assert not bad_np.any()


def test_face_defective_mask_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="max_fold_angle must be in"):
        od.validation.face_defective_mask(mesh_wp.points, mesh_wp.indices, max_fold_angle=200.0)
    with pytest.raises(ValueError, match="max_normal_angle must be in"):
        od.validation.face_defective_mask(mesh_wp.points, mesh_wp.indices, max_normal_angle=0.0)


def test_face_defective_mask_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = warp_empty(0, wp.int32, device)
    assert od.validation.face_defective_mask(vertices_wp, faces_wp).shape == (0,)


@pytest.mark.parity("face_defective_mask", "pymeshlab")
def test_face_defective_mask_matches_pymeshlab_on_folds(
    device: str, folded_patch: tuple[np.ndarray, np.ndarray]
) -> None:
    """
    Class B (compare detection): MeshLab flips the fold where this deletes it.

    ``compute_selection_bad_faces(select_folded_faces=True)`` is the same dihedral criterion at the
    same threshold, and it reports a selection rather than editing the mesh — which makes it the
    oracle for ``face_defective_mask``'s fold gate even though ``meshing_remove_folded_faces`` and
    ``remove_folded_faces`` then do different things with the answer.
    """
    vertices_np, faces_np = folded_patch
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(ml.Mesh(vertices_np, np.ascontiguousarray(faces_np, dtype=np.int32)))
    meshset_pml.compute_selection_bad_faces(
        usear=False, usenf=False, select_folded_faces=True, folded_faces_angle_threshold=160.0
    )
    selected_pml = meshset_pml.current_mesh().face_selection_array()

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    folded_np = od.validation.face_defective_mask(
        vertices_wp, faces_wp, min_quality=None, max_fold_angle=160.0
    ).numpy()
    assert np.array_equal(folded_np.astype(bool), selected_pml.astype(bool))


def _hinge_fan_np(angles_deg: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one independent hinged triangle pair per requested dihedral angle, spaced 3 units apart.

    Each pair shares one edge and nothing else, so the fold angle is prescribed exactly and no pair
    can be confused with another -- which is what separates a dihedral criterion from a proximity
    one (see [`test_remove_folded_faces_matches_meshlib`]).
    """
    vertices, faces = [], []
    for index, angle in enumerate(angles_deg):
        origin = np.array([3.0 * index, 0.0, 0.0])
        tilt = np.radians(180.0 - angle)
        base = len(vertices)
        vertices += [
            origin,
            origin + np.array([0.0, 1.0, 0.0]),
            origin + np.array([1.0, 0.0, 0.0]),
            origin + np.array([np.cos(tilt), 0.0, np.sin(tilt)]),
        ]
        faces += [[base, base + 2, base + 1], [base, base + 1, base + 3]]
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


@pytest.mark.parametrize("threshold", [160.0, 120.0])
@pytest.mark.parity("face_defective_mask", "meshlib")
def test_face_defective_mask_matches_meshlib(
    device: str, threshold: float, folded_patch: tuple[np.ndarray, np.ndarray]
) -> None:
    """
    Class B (compare detection): ``findOverlappingTris`` under the named angle-to-dot transform.

    MeshLib parameterizes a fold by the **dot product** of the two normals where ordito takes the
    dihedral angle in degrees, so the transform is ``maxNormalDot = cos(radians(angle))``: its own
    default of ``-0.99`` is 171.9 degrees, not ordito's 160. Fed that, the two agree face for face
    on a fan of seven independently hinged pairs spanning 10 to 175 degrees, at both thresholds.

    ``findNotSmoothFaces`` is **not** the pairing, and that was measured: it reports **zero** faces
    on this fan at every ``minAngle`` from 0.1 to 3.0 radians, so a comparison built on it would
    pass vacuously against any implementation.

    Two conventions the fan is shaped around. MeshLib is **inclusive at the threshold** where
    ordito is exclusive -- a pair at exactly 140 degrees is flagged by MeshLib and not by ordito
    at ``angle=140`` -- so no fixture angle sits on a threshold used here. And MeshLib's criterion
    is *proximity plus antiparallel normals*, not adjacency: on the three-face
    ``folded_patch`` it flags all three faces because the folded apex triangle lies over both quad
    halves, where ordito flags only the one face whose dihedral exceeds the threshold. That
    divergence is asserted below rather than avoided, since it is the reason the fan exists.
    """
    fan_angles = (10.0, 60.0, 100.0, 140.0, 150.0, 165.0, 175.0)
    vertices_np, faces_np = _hinge_fan_np(fan_angles)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    folded_wp = od.validation.face_defective_mask(
        vertices_wp, faces_wp, min_quality=None, max_fold_angle=threshold
    ).numpy()

    settings_ml = mm.FindOverlappingSettings()
    settings_ml.maxNormalDot = float(np.cos(np.radians(threshold)))
    mesh_ml = numpy_to_meshlib(vertices_np, faces_np)
    folded_ml = meshlib_bitset_to_numpy(
        mm.findOverlappingTris(mm.MeshPart(mesh_ml), settings_ml), faces_np.shape[0]
    )

    n_folded = 2 * sum(angle > threshold for angle in fan_angles)
    assert int(folded_ml.sum()) == n_folded  # non-vacuity: neither empty nor everything
    assert np.array_equal(folded_wp, folded_ml)

    # The divergence the fan avoids: proximity, not adjacency, so an apex over two faces flags both.
    patch_vertices_np, patch_faces_np = folded_patch
    patch_vertices_wp, patch_faces_wp = numpy_to_warp(patch_vertices_np, patch_faces_np, device)
    patch_wp = od.validation.face_defective_mask(
        patch_vertices_wp, patch_faces_wp, min_quality=None, max_fold_angle=threshold
    ).numpy()
    patch_ml = meshlib_bitset_to_numpy(
        mm.findOverlappingTris(
            mm.MeshPart(numpy_to_meshlib(patch_vertices_np, patch_faces_np)), settings_ml
        ),
        patch_faces_np.shape[0],
    )
    assert int(patch_wp.sum()) == 1
    assert int(patch_ml.sum()) == 3
