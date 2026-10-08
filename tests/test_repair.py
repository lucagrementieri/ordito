"""Regression tests for ``ordito.repair`` against libigl."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import TypedDict, cast

import igl
import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import trimesh.repair as tm_repair
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from pymeshfix import _meshfix

import ordito as od
from benchmarks.meshes import BUILDERS
from tests.comparisons import (
    assert_unordered_rows_equal,
    canonical_winding,
    hausdorff_surface_two_sided,
    hausdorff_two_sided,
    lexsort_rows,
    same_partition,
    undirected_edges,
)
from tests.conversions import (
    meshlib_bitset_to_numpy,
    meshlib_to_trimesh,
    numpy_to_meshlib,
    numpy_to_pymeshfix,
    numpy_to_warp,
    points_to_warp,
    pymeshfix_intersecting_faces,
    pymeshfix_to_numpy,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshfix,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
    trimesh_to_warp,
    warp_empty,
    warp_to_meshlib,
    warp_to_pymeshfix,
    warp_to_trimesh,
)


class _ComponentCriteria(TypedDict, total=False):
    keep_largest: bool
    min_faces: int
    min_area: float
    min_diameter: float


class _StraightenGates(TypedDict):
    min_normal_dot: float
    max_aspect_ratio: float
    iterations: int


# --------------------------------------------------------------------------------------
# make_solid
# --------------------------------------------------------------------------------------


def _overlapping_spheres_tm() -> tm.Trimesh:
    """Two ``icosphere(2)``s translated 1.2 apart: 72 of 640 faces self-intersect."""
    first_tm = tm.creation.icosphere(subdivisions=2)
    second_tm = tm.creation.icosphere(subdivisions=2)
    second_tm.apply_translation([1.2, 0.0, 0.0])
    joined_tm = tm.util.concatenate([first_tm, second_tm])
    assert isinstance(joined_tm, tm.Trimesh)
    return joined_tm


def _spaced_bowls_tm(count: int, gap: float = 3.0) -> tm.Trimesh:
    """
    Build ``count`` open hemispherical bowls in a row, each ``gap`` apart along ``x``.

    Built here rather than from the ``hemisphere`` fixture, and the reason is a measured limit of
    the pipeline rather than convenience: that fixture is rotated 45 degrees about ``(1, 1, 0)``, so
    three spaced copies have three rims in three different planes, the loop they merge into is badly
    non-planar, and the minimum-weight patch across it self-intersects. ``make_solid`` then cuts the
    patch out and undoes the join -- measured, three separate shells back out (chi = 6), and 17 of
    them at ``gap = 4.0`` on the CPU. With the bowls unrotated the answer is byte-identical at
    ``gap`` 2.5 / 3.0 / 3.5 / 5.0 on both devices: 291 v / 578 f, chi = 2, one component.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=2, radius=1.0)
    bowl_tm = sphere_tm.slice_plane(
        plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]), cap=False
    )
    bowl_tm.merge_vertices()
    shells_tm = []
    for index in range(count):
        shell_tm = bowl_tm.copy()
        shell_tm.apply_translation([gap * index, 0.0, 0.0])
        shells_tm.append(shell_tm)
    joined_tm = tm.util.concatenate(shells_tm)
    assert isinstance(joined_tm, tm.Trimesh)
    return joined_tm


def _assert_is_a_solid(vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]) -> tm.Trimesh:
    """Assert the four post-conditions ``make_solid`` exists to establish, together."""
    solid_tm = warp_to_trimesh(vertices_wp, faces_wp)
    assert solid_tm.is_watertight
    assert solid_tm.euler_number == 2
    assert len(solid_tm.split(only_watertight=False)) == 1
    assert int(od.validation.face_self_intersecting_mask(vertices_wp, faces_wp).numpy().sum()) == 0
    return solid_tm


@pytest.mark.parity("make_solid", "pymeshfix")
def test_make_solid_matches_pymeshfix_on_interpenetrating_shells(device: str) -> None:
    """
    Class C, and the strongest pipeline pair available: the two answers are the same surface.

    ``clean_from_arrays`` is pymeshfix's headline and the reason anyone installs it -- broken
    digitised surface in, single watertight solid out -- and this is ordito's composite of the same
    stages. On two ``icosphere(2)``s translated 1.2 apart (324 v / 640 f, 72 of them
    self-intersecting) the two agree on **everything measurable**: 162 v / 320 f, watertight,
    Euler characteristic 2, one component, no self-intersecting face, enclosed volume 4.0470, and a
    two-sided surface distance of **3.11e-08**.

    That is a class-C statement rather than a class-B one because nothing pairs the buffers: both
    sides delete, refill and renumber, so only the surface and the derived scalars are comparable.

    Mutation probe and margin: the *unrepaired* input sits **1.192** from pymeshfix's answer, so the
    1e-6 bound asserted here is seven orders of magnitude clear of "did not repair it", and the
    volume equality is what rules out the two having kept different shells.
    """
    mesh_tm = _overlapping_spheres_tm()
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)

    solid_wp, solid_faces_wp = od.repair.make_solid(vertices_wp, faces_wp)
    solid_tm = _assert_is_a_solid(solid_wp, solid_faces_wp)

    vertices_pmf, faces_pmf = _meshfix.clean_from_arrays(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
    )
    solid_pmf = tm.Trimesh(vertices_pmf, faces_pmf, process=False)

    # Non-vacuity: the reference really repaired the input rather than passing it through.
    assert faces_pmf.shape[0] < mesh_tm.faces.shape[0]
    assert solid_pmf.is_watertight
    assert solid_pmf.euler_number == 2
    assert solid_faces_wp.size // 3 == faces_pmf.shape[0]
    assert solid_wp.size == vertices_pmf.shape[0]
    assert np.isclose(solid_tm.volume, solid_pmf.volume, rtol=1e-5, atol=1e-5)
    assert (
        hausdorff_surface_two_sided(
            np.asarray(solid_tm.vertices), solid_tm.faces, vertices_pmf, faces_pmf
        )
        < 1e-6
    )


@pytest.mark.parametrize("mesh_name", ["torus_self_intersecting", "bohemian_dome"])
def test_make_solid_singular_curve_divergence(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Not a parity assert: the input class where the two pipelines part, pinned with numbers.

    A surface that crosses itself along a *curve* rather than in a band has no canonical repair --
    every implementation has to decide how much surface to sacrifice around the singularity, and the
    two decide differently. Measured, both ending in a watertight one-component solid with no
    self-intersecting face:

    | input | ordito | pymeshfix | surface distance |
    |---|---|---|---|
    | ``torus_self_intersecting`` | 419 v, volume -9.60 | 80 v, volume -6.53 | **0.400** |
    | ``bohemian_dome`` | 855 v, volume 1.42 | 747 v, volume -1.54 | **2.615** |

    pymeshfix removes far more of the torus (80 vertices of 256) where ordito cuts closer to the
    intersection; on the dome both are far from the *input* too (2.61 and 2.22), which is what says
    the gap is the question rather than either answer. So this test asserts the post-conditions on
    both sides and records the divergence instead of bounding it -- a surface bound here would
    either be loose enough to mean nothing or would pin one library's sacrifice as correct.

    [`test_make_solid_matches_pymeshfix_on_interpenetrating_shells`] is where the pair *is*
    comparable, and it carries the parity claim.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_faces = mesh_tm.faces.shape[0]
    assert (
        int(
            od.validation.face_self_intersecting_mask(mesh_wp.points, mesh_wp.indices).numpy().sum()
        )
        > 0
    )

    solid_wp, solid_faces_wp = od.repair.make_solid(mesh_wp.points, mesh_wp.indices)
    _assert_is_a_solid(solid_wp, solid_faces_wp)

    vertices_pmf, faces_pmf = _meshfix.clean_from_arrays(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
    )
    solid_pmf = tm.Trimesh(vertices_pmf, faces_pmf, process=False)

    assert solid_pmf.is_watertight
    assert solid_pmf.euler_number == 2
    # Both sacrificed surface rather than growing it, which is the one thing they do share here.
    assert solid_faces_wp.size // 3 < 2 * n_faces
    assert faces_pmf.shape[0] < n_faces


def test_make_solid_closes_and_connects_open_shells(device: str) -> None:
    """
    Not a library comparison: what ``keep_largest`` and ``join_components`` each mean.

    Three open bowls 3.0 apart are the input that separates them, and both answers are solids --
    which is the point: the flag chooses *which* solid. ``keep_largest`` (the default, and
    ``clean_from_arrays``' own) keeps one shell and caps it, at 97 v / 190 f and volume 2.0235 --
    the same volume pymeshfix reports on this input. Turning it off and welding instead gives one
    connected solid of 291 v / 578 f and volume 6.0706 from all three, a combination no reference
    here can produce: MeshFix has ``joincomp`` but not ``joincomp`` without its component filter.

    The face count separates them independently of the volume, which is what makes the pair an
    assertion about the flags rather than about the geometry: 190 against 578.

    Both numbers are byte-identical across ``gap`` 2.5 / 3.0 / 3.5 / 5.0 and both devices. The input
    is built rather than taken from the ``hemisphere`` fixture for a reason worth reading in
    [`_spaced_bowls_tm`] -- with rotated bowls the welded path does *not* converge, which is a real
    limit of the composite and is documented on ``make_solid`` itself.
    """
    mesh_tm = _spaced_bowls_tm(3)
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)

    largest_wp, largest_faces_wp = od.repair.make_solid(vertices_wp, faces_wp)
    largest_tm = _assert_is_a_solid(largest_wp, largest_faces_wp)

    welded_wp, welded_faces_wp = od.repair.make_solid(
        vertices_wp, faces_wp, keep_largest=False, join_components=True
    )
    welded_tm = _assert_is_a_solid(welded_wp, welded_faces_wp)

    assert largest_faces_wp.size // 3 < mesh_tm.faces.shape[0] // 2
    assert welded_tm.volume > 2.5 * largest_tm.volume
    assert welded_faces_wp.size > largest_faces_wp.size


def test_make_solid_leaves_a_solid_alone(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: a mesh that is already a solid comes back as one, unchanged in size.

    The idempotence case, and the one that would catch a pipeline that "repairs" a clean input --
    a filler that patched a hole that was not there, or a component filter that dropped the mesh.
    The counts are asserted exactly rather than bounded, since nothing here has anything to do.
    """
    mesh_tm, mesh_wp = sphere_irregular

    solid_wp, solid_faces_wp = od.repair.make_solid(mesh_wp.points, mesh_wp.indices)
    solid_tm = _assert_is_a_solid(solid_wp, solid_faces_wp)

    assert solid_faces_wp.size // 3 == mesh_tm.faces.shape[0]
    assert solid_wp.size == mesh_tm.vertices.shape[0]
    assert np.isclose(solid_tm.volume, mesh_tm.volume, rtol=1e-5, atol=1e-5)


def _resolve_duplicated_faces_ref(faces_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CPU reference mirroring ``igl::resolve_duplicated_faces``."""
    faces = np.asarray(faces_np, dtype=np.int32).reshape((-1, 3))
    n_faces = faces.shape[0]
    if n_faces == 0:
        return faces.copy(), np.empty(0, dtype=np.int32)

    sorted_faces = np.sort(faces, axis=1)
    unique_sorted, inverse = np.unique(sorted_faces, axis=0, return_inverse=True)
    num_unique = unique_sorted.shape[0]

    kept: list[int] = []
    for ui in range(num_unique):
        member = np.flatnonzero(inverse == ui)
        urow = unique_sorted[ui]
        signed_ids: list[int] = []
        count = 0
        for fi in member:
            row = faces[fi]
            consistent = (
                (row[0] == urow[0] and row[1] == urow[1] and row[2] == urow[2])
                or (row[0] == urow[1] and row[1] == urow[2] and row[2] == urow[0])
                or (row[0] == urow[2] and row[1] == urow[0] and row[2] == urow[1])
            )
            signed = int(fi + 1) if consistent else -int(fi + 1)
            signed_ids.append(signed)
            count += 1 if consistent else -1

        if member.size == 1:
            kept.append(int(member[0]))
            continue
        if count == 1:
            for fid in signed_ids:
                if fid > 0:
                    kept.append(fid - 1)
                    break
        elif count == -1:
            for fid in signed_ids:
                if fid < 0:
                    kept.append(-fid - 1)
                    break
        elif count == 0:
            continue
        else:
            raise ValueError(f"non-orientable duplicate face group {ui} with count {count}")

    if len(kept) == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.int32)
    kept_np = np.asarray(kept, dtype=np.int32)
    return faces[kept_np], kept_np


def _assert_duplicate_vertices_match(
    vertices_np: np.ndarray,
    sv_wp: np.ndarray,
    svj_wp: np.ndarray,
    sf_wp: np.ndarray | None = None,
    faces_np: np.ndarray | None = None,
    epsilon: float = 0.0,
) -> None:
    sv_igl, _, _svj_igl, sf_igl = (
        igl.remove_duplicate_vertices(vertices_np, faces_np, epsilon)
        if faces_np is not None
        else (*igl.remove_duplicate_vertices(vertices_np, epsilon), None)
    )
    # ordito keeps float32 vertices where igl works in float64, so the surviving position sets are
    # compared with a tolerance rather than exactly (exact equality happens to hold for coordinates
    # that are exactly representable, but not for a mesh with irrational ones).
    assert np.allclose(lexsort_rows(sv_wp), lexsort_rows(np.asarray(sv_igl)), rtol=1e-5, atol=1e-5)
    for i, vertex in enumerate(vertices_np):
        assert np.allclose(sv_wp[svj_wp[i]], vertex, rtol=1e-5, atol=1e-5)
    if sf_wp is not None and faces_np is not None and sf_igl is not None:
        tri_wp = sv_wp[sf_wp.reshape(-1, 3)]
        tri_igl = np.asarray(sv_igl)[np.asarray(sf_igl)]
        assert np.allclose(np.sort(tri_wp, axis=1), np.sort(tri_igl, axis=1), rtol=1e-5, atol=1e-5)


@pytest.mark.parity("remove_unreferenced_vertices", "igl")
@pytest.mark.parametrize("case", ["fully_referenced", "extra_vertices"])
def test_remove_unreferenced_vertices_matches_igl(
    request: pytest.FixtureRequest, device: str, case: str
) -> None:
    """
    Class A: all four returns against ``igl.remove_unreferenced``, with and without spares.

    On the fully referenced ``sphere_irregular`` the call is the identity; with four unreferenced
    vertices appended the compaction is real. The forward map and the inverse map are both compared,
    not just the compacted buffers: a compaction that dropped the right vertices but numbered them
    differently would pass on the first two returns alone, and every caller that keeps per-vertex
    data alongside relies on the maps.
    """
    if case == "fully_referenced":
        mesh_tm, _mesh_wp = request.getfixturevalue("sphere_irregular")
        vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
        faces_np = np.asarray(mesh_tm.faces, dtype=np.int32)
    else:
        rng = np.random.default_rng(0)
        triangle_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        vertices_np = np.vstack([triangle_np, rng.normal(size=(4, 3))])
        faces_np = np.array([[0, 1, 2]], dtype=np.int32)

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    nv_wp, nf_wp, remap_wp, inverse_wp = od.repair.remove_unreferenced_vertices(
        vertices_wp, faces_wp, return_inverse=True
    )

    nv_igl, nf_igl, remap_igl, inverse_igl = igl.remove_unreferenced(vertices_np, faces_np)

    assert np.allclose(nv_wp.numpy(), nv_igl, rtol=1e-5, atol=1e-5)
    assert np.array_equal(nf_wp.numpy().reshape(-1, 3), nf_igl)
    assert np.array_equal(remap_wp.numpy(), np.asarray(remap_igl).ravel())
    assert np.array_equal(inverse_wp.numpy(), np.asarray(inverse_igl).ravel())


@pytest.mark.parity("remove_unreferenced_vertices", "open3d")
def test_remove_unreferenced_matches_open3d(device: str):
    """
    Class A on both compacted buffers -- open3d keeps first-occurrence order like ordito does.

    open3d returns no forward or inverse map (the mesh is mutated in place and read back), so
    unlike the igl tests only the vertices and faces are comparable; the maps stay pinned by igl.
    The four appended vertices make the compaction real, and the interleaved layout (an
    unreferenced vertex *between* referenced ones) is what catches a compaction that preserves
    a prefix instead of an order.
    """
    rng = np.random.default_rng(0)
    referenced_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64
    )
    # Interleave dead vertices among live ones rather than appending them.
    vertices_full_np = np.insert(referenced_np, [1, 3, 4, 4], rng.normal(size=(4, 3)), axis=0)
    faces_np = np.array([[0, 2, 4], [2, 5, 4]], dtype=np.int32)

    vertices_wp, faces_wp = numpy_to_warp(vertices_full_np, faces_np, device)
    nv_wp, nf_wp, _remap_wp = od.repair.remove_unreferenced_vertices(vertices_wp, faces_wp)

    mesh_o3d = trimesh_to_open3d(tm.Trimesh(vertices_full_np, faces_np, process=False))
    mesh_o3d.remove_unreferenced_vertices()
    vertices_o3d = np.asarray(mesh_o3d.vertices)

    assert vertices_o3d.shape[0] == referenced_np.shape[0]  # the reference really compacted
    assert np.allclose(nv_wp.numpy(), vertices_o3d, rtol=1e-5, atol=1e-5)
    assert np.array_equal(nf_wp.numpy().reshape(-1, 3), np.asarray(mesh_o3d.triangles))


def test_remove_unreferenced_sentinel(device: str):
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, -1], [0, 2, 1]], dtype=np.int32)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    nv_wp, nf_wp, remap_wp = od.repair.remove_unreferenced_vertices(vertices_wp, faces_wp)

    assert np.allclose(nv_wp.numpy(), vertices_np[[0, 1, 2]], rtol=1e-5, atol=1e-5)
    assert np.array_equal(nf_wp.numpy().reshape(-1, 3), faces_np)
    expected_remap = np.array([0, 1, 2], dtype=np.int32)
    assert np.array_equal(remap_wp.numpy(), expected_remap)


@pytest.mark.parity("remove_unreferenced_vertices", "meshlib")
def test_remove_unreferenced_vertices_matches_meshlib_pack(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Class A on both compacted buffers, and the clearest demonstration of MeshLib's ``pack()``.

    MeshLib has no ``remove_unreferenced_vertices``: dropping a face leaves its now-orphaned vertex
    in the point buffer as an *invalid* entry, and ``Mesh.pack()`` is what compacts the buffers and
    renumbers. Reading before packing is silent and wrong -- measured here on a 42-vertex sphere
    with one interior vertex orphaned: ``points.size()`` reads **42** and ``getNumpyVerts`` returns
    42 rows while ``numValidVerts`` reads **41**, so a comparison that skipped the pack would report
    a vertex ordito had removed as a disagreement. After the pack both sides read 41 and 75 faces.

    The orphaned vertex is chosen *interior* deliberately: the converter sizes its point buffer by
    ``F.max() + 1``, so a **trailing** unreferenced vertex is dropped by
    [`numpy_to_meshlib`][tests.conversions.numpy_to_meshlib] before MeshLib ever sees it and the
    comparison would be vacuous. Positions are compared after a lexsort because the two compactions
    renumber differently; the face *count* is exact on both sides.
    """
    mesh_tm, _mesh_wp = sphere_irregular
    orphan = 7
    faces_np = mesh_tm.faces[~(mesh_tm.faces == orphan).any(axis=1)]
    n_vertices = mesh_tm.vertices.shape[0]

    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, faces_np, device)
    kept_wp, kept_faces_wp, _remap_wp = od.repair.remove_unreferenced_vertices(
        vertices_wp, faces_wp
    )

    mesh_ml = numpy_to_meshlib(mesh_tm.vertices, faces_np)
    # The state pack() exists for, asserted rather than described: the buffer still holds it.
    assert mesh_ml.points.size() == n_vertices
    assert mn.getNumpyVerts(mesh_ml).shape[0] == n_vertices
    assert mesh_ml.topology.numValidVerts() == n_vertices - 1
    packed_tm = meshlib_to_trimesh(mesh_ml)

    assert packed_tm.vertices.shape[0] == n_vertices - 1  # the reference really compacted
    assert kept_wp.size == packed_tm.vertices.shape[0]
    assert kept_faces_wp.size // 3 == packed_tm.faces.shape[0]
    assert np.allclose(
        lexsort_rows(np.round(kept_wp.numpy().astype(np.float64), 5)),
        lexsort_rows(np.round(packed_tm.vertices, 5)),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parity(
    "remove_unreferenced_vertices",
    "pymeshfix",
    benchmarked=False,
    reason="load_array *is* the operation here -- it drops unreferenced vertices as part of the "
    "connectivity repair it runs before returning -- so there is nothing separable to time and a "
    "row would price the 67.9 ms load on bunny_decimated under this group's name. The answer is "
    "the loaded mesh, which is exactly what this test reads.",
)
@pytest.mark.parametrize("placement", ["trailing", "interior", "spread"])
def test_remove_unreferenced_vertices_matches_pymeshfix(device: str, placement: str) -> None:
    """
    Class A on the compacted buffer, positions equal **in order** -- the strongest form available.

    Unusual among the pymeshfix comparisons in that the reference is the *loader*: ``load_array``
    runs a connectivity fix before it returns anything, and dropping unreferenced vertices is part
    of it. So there is no call to make -- the mesh that comes back is the answer -- and unlike
    MeshLib's ``pack()`` (which renumbers in its own order and forces a lexsort) it preserves the
    survivors' relative order, so the two buffers compare element for element with no
    canonicalization at all.

    Three placements, because the two references that size their vertex buffer by ``F.max() + 1``
    behave *differently* by position and this one does not: a **trailing** spare, an **interior**
    one, and three spread through the buffer all come back as the same 162 positions in the same
    order. That is what makes the class-A claim safe here where
    [`test_remove_unreferenced_vertices_matches_meshlib_pack`] has to lexsort.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=2)
    vertices_np, faces_np = np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces)
    if placement == "trailing":
        padded_np = np.vstack([vertices_np, [[5.0, 5.0, 5.0]]])
        shifted_np = faces_np
    elif placement == "interior":
        padded_np = np.vstack([vertices_np[:5], [[5.0, 5.0, 5.0]], vertices_np[5:]])
        shifted_np = np.where(faces_np >= 5, faces_np + 1, faces_np)
    else:
        padded_np = np.vstack(
            [
                vertices_np[:3],
                [[9.0, 0.0, 0.0]],
                vertices_np[3:20],
                [[9.0, 1.0, 0.0]],
                vertices_np[20:],
                [[9.0, 2.0, 0.0]],
            ]
        )
        shifted_np = np.where(
            faces_np >= 20, faces_np + 2, np.where(faces_np >= 3, faces_np + 1, faces_np)
        )

    vertices_wp, faces_wp = numpy_to_warp(padded_np, shifted_np, device)
    kept_wp, kept_faces_wp, _remap_wp = od.repair.remove_unreferenced_vertices(
        vertices_wp, faces_wp
    )

    kept_pmf, kept_faces_pmf = pymeshfix_to_numpy(numpy_to_pymeshfix(padded_np, shifted_np))

    # Non-vacuity: the reference really dropped the spares rather than passing the buffer through.
    assert kept_pmf.shape[0] == vertices_np.shape[0] < padded_np.shape[0]
    assert kept_faces_pmf.shape[0] == faces_np.shape[0]
    assert kept_wp.size == kept_pmf.shape[0]
    assert kept_faces_wp.size // 3 == kept_faces_pmf.shape[0]
    assert np.allclose(kept_wp.numpy().astype(np.float64), kept_pmf, rtol=1e-5, atol=1e-5)


_DUPLICATE_VERTEX_CASES = {
    # Two exactly coincident positions, no faces.
    "exact": ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], None, 0.0),
    # Two pairs a hair apart, inside the tolerance.
    "epsilon": ([[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-9, 0.0]], None, 1e-8),
    # The exact pair again, now referenced by faces that must be remapped.
    "faces": (
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 3], [2, 1, 3]],
        0.0,
    ),
}


@pytest.mark.parametrize("case", list(_DUPLICATE_VERTEX_CASES))
def test_remove_duplicated_vertices_matches_igl(device: str, case: str) -> None:
    """
    Class B against ``igl.remove_duplicate_vertices``: survivors after a lexsort, plus both maps.

    Both libraries renumber the survivors, so the surviving positions are compared as sorted rows,
    every input vertex must map onto a survivor at its own position, and with faces the remapped
    triangles must be the reference's as sorted corner positions.
    """
    vertices_list, faces_list, epsilon = _DUPLICATE_VERTEX_CASES[case]
    vertices_np = np.array(vertices_list, dtype=np.float64)
    if faces_list is None:
        vertices_wp = points_to_warp(vertices_np, device)
        sv_wp, _, svj_wp, _ = od.repair.remove_duplicated_vertices(
            vertices_wp, warp_empty(0, wp.int32, device), epsilon=epsilon
        )
        _assert_duplicate_vertices_match(
            vertices_np, sv_wp.numpy(), svj_wp.numpy(), epsilon=epsilon
        )
    else:
        faces_np = np.array(faces_list, dtype=np.int32)
        vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
        sv_wp, _, svj_wp, sf_wp = od.repair.remove_duplicated_vertices(
            vertices_wp, faces_wp, epsilon
        )
        _assert_duplicate_vertices_match(
            vertices_np, sv_wp.numpy(), svj_wp.numpy(), sf_wp.numpy(), faces_np, epsilon=epsilon
        )


def test_remove_duplicate_vertices_epsilon_negative_coordinates(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Class B (partition): igl decides which vertices merge when the grid indices go negative.

    The fixture is centred at ``(-1, 0, 2)``, so every duplicated vertex has a negative
    coordinate -- the case the row packing cannot take directly and where a wrong offset
    silently merges the wrong pairs.
    """
    # Any mesh spanning the origin rounds to negative grid indices, which the row packing cannot
    # take directly. The fixture is centred at (-1, 0, 2), so every duplicated vertex below has at
    # least one negative coordinate; igl is the oracle for which ones merge.
    mesh_tm, _ = sphere_irregular
    vertices_np = np.vstack((mesh_tm.vertices, mesh_tm.vertices[:4] + 1e-9)).astype(np.float64)
    assert vertices_np.min() < 0.0
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int32)

    epsilon = 1e-6
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    sv_wp, _, svj_wp, sf_wp = od.repair.remove_duplicated_vertices(
        vertices_wp, faces_wp, epsilon=epsilon
    )
    assert sv_wp.size == mesh_tm.vertices.shape[0]
    _assert_duplicate_vertices_match(
        vertices_np, sv_wp.numpy(), svj_wp.numpy(), sf_wp.numpy(), faces_np, epsilon=epsilon
    )


@pytest.mark.parametrize("epsilon", [1e-3, 1e-2])
def test_duplicate_vertex_inverse_matches_the_documented_quantization(
    device: str, epsilon: float
) -> None:
    """
    Class B (label packing): the equivalence classes are the ``epsilon`` grid cells, as documented.

    ``hash_vector_rows`` snaps each coordinate to a multiple of ``epsilon`` measured from the data's
    own minimum corner, so that is the numpy oracle. Only the *partition* is shared -- ordito names
    a class by its hash-table slot and ``np.unique`` numbers them in lexicographic order -- hence
    [`same_partition`][tests.comparisons.same_partition]. Measured 34 classes on both sides at both
    tolerances, out of 60 input positions, so the comparison is neither all-merged nor all-distinct.
    """
    rng = np.random.default_rng(9)
    positions_np = rng.integers(0, 4, size=(60, 3)).astype(np.float32) * 0.25
    # Jitter well inside one cell, so a correct implementation merges exactly the seeded collisions.
    positions_np += rng.normal(scale=1e-5, size=positions_np.shape).astype(np.float32)
    positions_wp = points_to_warp(positions_np, device)

    inverse_np = od.repair.duplicate_vertex_inverse(positions_wp, epsilon).numpy()

    cells_np = np.rint((positions_np - positions_np.min(axis=0)) / epsilon).astype(np.int64)
    _cells_np, expected_np = np.unique(cells_np, axis=0, return_inverse=True)
    assert 1 < int(expected_np.max()) + 1 < positions_np.shape[0]
    assert int(inverse_np.max()) + 1 == int(expected_np.max()) + 1
    assert same_partition(inverse_np, expected_np)


def test_duplicate_vertex_inverse_at_zero_epsilon_collapses_bitwise_equal_positions(
    device: str,
) -> None:
    """
    The ``epsilon == 0`` branch, on the input its docstring promises it handles reliably.

    A relative bucket is not an equality test, and the Notes say so; what it *does* guarantee is
    collapsing positions that are already bitwise equal, including across ``+0.0`` / ``-0.0``. Three
    classes from five rows here, so neither the merge nor the separation is vacuous.
    """
    positions_np = np.array(
        [[0.0, 0.0, 0.0], [-0.0, -0.0, -0.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [5.0, 5.0, 5.0]],
        dtype=np.float32,
    )
    positions_wp = points_to_warp(positions_np, device)

    inverse_np = od.repair.duplicate_vertex_inverse(positions_wp, 0.0).numpy()

    assert int(inverse_np.max()) + 1 == 3
    assert inverse_np[0] == inverse_np[1]
    assert inverse_np[2] == inverse_np[3]
    assert len({int(inverse_np[0]), int(inverse_np[2]), int(inverse_np[4])}) == 3


def test_duplicate_vertex_inverse_is_what_remove_duplicated_vertices_remaps_by(device: str) -> None:
    """
    The reason it is public: the same map ``remove_duplicated_vertices`` used, for other attributes.

    A caller remapping per-vertex colors or UVs needs the map without the deduplicated buffers, so
    what has to hold is that applying it to the faces reproduces that function's own remapped face
    buffer, and that gathering the deduplicated positions by it returns the input to within
    ``epsilon``.
    """
    epsilon = 1e-3
    base_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    positions_np = np.vstack((base_np, base_np + 1e-6, base_np[[0]])).astype(np.float32)
    faces_np = np.array([[0, 1, 2], [3, 4, 5], [6, 1, 2]], dtype=np.int32)
    positions_wp, faces_wp = numpy_to_warp(positions_np, faces_np, device)

    inverse_np = od.repair.duplicate_vertex_inverse(positions_wp, epsilon).numpy()

    unique_wp, _indices_wp, _inverse_wp, unique_faces_wp = od.repair.remove_duplicated_vertices(
        positions_wp, faces_wp, epsilon
    )
    assert unique_wp.size == 3
    assert np.array_equal(unique_faces_wp.numpy(), inverse_np[faces_np.reshape(-1)])
    assert np.abs(unique_wp.numpy()[inverse_np] - positions_np).max() <= epsilon


@pytest.mark.parametrize(
    "faces_list",
    [
        [[0, 1, 2], [0, 1, 2], [0, 2, 1], [0, 2, 1]],
        [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 2, 1], [0, 2, 1]],
    ],
    ids=["cancelling", "keep_positive"],
)
def test_resolve_duplicated_faces_small_groups(device: str, faces_list: list[list[int]]) -> None:
    """
    Class A against the NumPy transcription of the signed-count rule, on one duplicate group.

    Two copies each way cancel to nothing; three against two leave one face of the majority
    winding. Both the kept faces and the kept indices must match exactly.
    """
    faces_np = np.array(faces_list, dtype=np.int32)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )

    f2_wp, j_wp = od.repair.resolve_duplicated_faces(faces_wp)
    f2_np, j_np = _resolve_duplicated_faces_ref(faces_np)

    assert np.array_equal(f2_wp.numpy().reshape(-1, 3), f2_np)
    assert np.array_equal(j_wp.numpy(), j_np)


@pytest.mark.parity(
    "resolve_duplicated_faces",
    "numpy",
    benchmarked=False,
    reason="the reference is a hand-rolled NumPy transcription of the signed-count rule, so it is "
    "not a library implementation to race -- and the two libraries that *are* timed here (open3d "
    "and pymeshlab) are both exempt because their de-duplication rule differs by design. This is "
    "the only oracle for the rule itself, which is why it is claimed untimed rather than left "
    "as the group's third incomparable row.",
)
def test_resolve_duplicated_faces_random(device: str):
    """
    Class B (set equality): the signed-count rule against a NumPy transcription of it.

    The named transform is the emission order. The reference groups with ``np.unique`` and so emits
    lexicographically, where the production path follows its hash-sorted unique order -- so the
    *kept sets* are what must agree, and the surviving index list is compared after sorting.

    This is the group's only oracle. Both benchmarked references are exempt: open3d and pymeshlab
    both de-duplicate by a different rule than the signed count (a cancelling pair vanishes here and
    survives there), which ``benchmarks/test_repair.py`` records. So the rule itself is checked
    exactly once, here, and the input is built to exercise all three of its branches -- 100 singles,
    80 cancelling pairs, and 60 groups with a positive majority.
    """
    rng = np.random.default_rng(17)
    pool = np.stack(np.meshgrid(np.arange(12), np.arange(12, 24), np.arange(24, 36)), -1).reshape(
        -1, 3
    )
    base_np = rng.permutation(pool, axis=0)[:300].astype(np.int32)
    singles = base_np[:100]
    cancelling = np.vstack([base_np[100:180], base_np[100:180][:, ::-1]])
    majority_pos = np.vstack([np.repeat(base_np[180:240], 2, axis=0), base_np[180:240][:, ::-1]])
    faces_np = np.vstack([singles, cancelling, majority_pos])
    faces_np = rng.permutation(faces_np, axis=0)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )

    f2_wp, j_wp = od.repair.resolve_duplicated_faces(faces_wp)
    f2_np, j_np = _resolve_duplicated_faces_ref(faces_np)

    # The reference emits groups in lexicographic (np.unique) order while the production path
    # follows its hash-sorted unique order; the kept sets must agree exactly.
    assert np.array_equal(np.sort(j_wp.numpy()), np.sort(j_np))
    resolved_rows = {tuple(row) for row in f2_wp.numpy().reshape(-1, 3).tolist()}
    reference_rows = {tuple(row) for row in f2_np.tolist()}
    assert resolved_rows == reference_rows


def _faces_2d(faces_wp: wp.array[wp.int32]) -> np.ndarray:
    return faces_wp.numpy().reshape(-1, 3)


@pytest.mark.parametrize("epsilon", [0.0, 1e-6])
@pytest.mark.parity("remove_duplicated_vertices", "open3d", "pymeshlab", "pyvista", "meshlib")
def test_remove_duplicated_vertices_welds_a_soup_like_the_references(
    device: str, epsilon: float, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Welding an unwelded soup, the one repair group whose ``epsilon`` sweep maps across all three.

    ``benchmarks/test_repair.py`` singles this out: MeshLab has a filter per path
    (``meshing_remove_duplicate_vertices`` for exact, ``meshing_merge_close_vertices`` for a
    tolerance) that line up with ordito's two code paths one-for-one, and Open3D's
    ``remove_duplicated_vertices`` is the exact path. So unlike its siblings in this module this is
    a straight comparison rather than a semantics negotiation. VTK's ``clean`` is a fourth
    implementation of the exact path and lands on the same 162 (measured 172 -> 162 on a mesh with
    ten duplicated vertices, matching ordito exactly).

    **``validate_mesh`` is not the oracle here** and that was measured rather than assumed: its
    ``coincident_points`` field reads *empty* on ten exactly duplicated vertices, so a comparison
    against it would report no duplicates to remove. ``clean`` is the dedup reference.

    Class B on the *positions*: every reference renumbers the survivors differently, so the vertex
    sets are compared after a lexsort. Counting alone would be too weak -- a welder that merged the
    wrong pairs can still land on 500 -- which is why the positions are asserted and the count is
    only the headline. Measured: 2 988 soup positions collapse to exactly 500 on every side at both
    epsilon values, matching ``sphere_irregular``'s true vertex count.

    MeshLib is Class B on the survivors too: ``uniteCloseVertices`` welds in place and reports a
    *count*. The result is not returned -- the mesh is mutated and the return is the number of
    vertices merged (2 488 here, from 2 988 soup positions down to the mesh's 500) -- so it is
    read back through [`meshlib_to_trimesh`][tests.conversions.meshlib_to_trimesh], which packs
    first; without the pack the buffer still holds the 2 988 slots. ``uniteOnlyBd=False`` is the
    setting that matches ordito and is passed explicitly: MeshLib's default is ``True``, which
    welds only vertices on a boundary and would leave an interior soup untouched. On this input
    every position is a boundary vertex of its own triangle, so the default happens to agree --
    which is exactly why it is pinned rather than relied on. ``findCloseVertices`` is *not* the
    pairing: it flags **both** members of a close pair (2 bits for 1 duplicate, measured), where
    ordito's answer is the survivors.
    """
    mesh_tm, _mesh_tm_wp = sphere_irregular
    soup_np = np.ascontiguousarray(mesh_tm.vertices[mesh_tm.faces].reshape(-1, 3))
    faces_np = np.arange(soup_np.shape[0], dtype=np.int32).reshape(-1, 3)

    vertices_wp = points_to_warp(soup_np, device)
    faces_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    unique_wp, _indices_wp, _inverse_wp, _faces_wp = od.repair.remove_duplicated_vertices(
        vertices_wp, faces_wp, epsilon
    )

    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(ml.Mesh(soup_np, faces_np))
    if epsilon > 0.0:
        meshset_pml.meshing_merge_close_vertices(threshold=ml.PureValue(epsilon))
    else:
        meshset_pml.meshing_remove_duplicate_vertices()
    vertices_pml = meshset_pml.current_mesh().vertex_matrix()

    soup_tm = tm.Trimesh(soup_np, faces_np, process=False)
    vertices_o3d = np.asarray(trimesh_to_open3d(soup_tm).remove_duplicated_vertices().vertices)
    cleaned_pv = trimesh_to_pyvista(soup_tm).clean(
        point_merging=True, tolerance=epsilon, absolute=True
    )
    vertices_pv = np.asarray(cleaned_pv.points)

    mesh_ml = numpy_to_meshlib(soup_np, faces_np)
    n_merged_ml = mm.uniteCloseVertices(mesh_ml, epsilon, False)
    vertices_ml = meshlib_to_trimesh(mesh_ml).vertices

    assert n_merged_ml == soup_np.shape[0] - mesh_tm.vertices.shape[0]  # 2 488 of 2 988
    assert unique_wp.size == len(mesh_tm.vertices)
    assert vertices_pml.shape[0] == len(mesh_tm.vertices)
    assert vertices_o3d.shape[0] == len(mesh_tm.vertices)
    assert vertices_pv.shape[0] == len(mesh_tm.vertices)
    assert vertices_ml.shape[0] == len(mesh_tm.vertices)

    survivors_wp = lexsort_rows(np.round(unique_wp.numpy().astype(np.float64), 5))
    assert np.allclose(survivors_wp, lexsort_rows(np.round(vertices_pml, 5)), atol=1e-5)
    assert np.allclose(survivors_wp, lexsort_rows(np.round(vertices_o3d, 5)), atol=1e-5)
    assert np.allclose(survivors_wp, lexsort_rows(np.round(vertices_pv, 5)), atol=1e-5)
    assert np.allclose(survivors_wp, lexsort_rows(np.round(vertices_ml, 5)), rtol=1e-5, atol=1e-5)

    # The trap this row is written against: validate_mesh does not see exact duplicates.
    coincident_pv = trimesh_to_pyvista(soup_tm).validate_mesh().coincident_points
    assert coincident_pv is not None
    assert len(coincident_pv) == 0


def test_reverse_winding_leaves_its_input_alone(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a parity assert: the free function allocates rather than rewriting the caller's buffer.

    Worth pinning because the kernel it launches is safe in place (``creation`` uses it that way),
    so an in-place shortcut here would pass every value test while corrupting a shared buffer.
    """
    _mesh_tm, mesh_wp = sphere_irregular
    before = mesh_wp.indices.numpy().copy()
    reversed_faces = od.repair.reverse_winding(mesh_wp.indices)
    assert reversed_faces.ptr != mesh_wp.indices.ptr
    assert np.array_equal(mesh_wp.indices.numpy(), before)


@pytest.mark.parity("make_winding_consistent", "pymeshlab")
def test_make_winding_consistent_matches_pymeshlab(
    device: str, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Orientation repair against MeshLab's, which reaches the identical winding on every face.

    The two algorithms are not the same -- ``benchmarks/test_repair.py`` notes MeshLab does a
    serial face-to-face visit where ordito solves the Z2 bits with a parity-carrying union-find --
    but on a connected orientable surface the answer is unique up to one global flip, so
    agreement is not
    only possible, it is required.

    Class B twice: ``canonical_winding`` rotates each triangle to start at its lowest index (a
    rotation preserves orientation, so a flipped face still compares unequal) and a lexsort removes
    the face ordering. Measured on ``sphere_irregular`` with every third face reversed, they agree
    face for face **without** needing the global-flip escape -- both anchor on the first face of the
    component -- so the test asserts the strict form and would catch a flip if one appeared.
    """
    mesh_tm, _mesh_tm_wp = sphere_irregular
    flipped_np = mesh_tm.faces.copy()
    flipped_np[::3] = flipped_np[::3][:, ::-1]

    faces_wp = wp.array(
        np.ascontiguousarray(flipped_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    repaired_wp = od.repair.make_winding_consistent(faces_wp)

    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(flipped_np, dtype=np.int32),
        )
    )
    meshset_pml.meshing_re_orient_faces_coherently()
    faces_pml = meshset_pml.current_mesh().face_matrix()

    assert np.array_equal(
        lexsort_rows(canonical_winding(repaired_wp.numpy())),
        lexsort_rows(canonical_winding(faces_pml)),
    )
    assert od.validation.is_winding_consistent(repaired_wp)


@pytest.mark.parametrize("flip_seed_face", [False, True])
@pytest.mark.parity("make_winding_consistent", "meshlib")
def test_make_winding_consistent_matches_meshlib(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], device: str, flip_seed_face: bool
) -> None:
    """
    Class B: the same flip set up to a global flip, which is the gauge the two libraries fix apart.

    ``findDisorientedFaces`` returns the faces MeshLib would reverse, so the comparison is between
    two *sets of faces to flip* -- ordito's is recovered by differencing its output buffer against
    its input. They agree exactly, but only up to complementation, and the parametrization is what
    makes that visible rather than lucky: ordito's flood fill keeps the **seed face** (face 0) as
    it found it, while MeshLib decides globally by ray casting and picks the *outward* orientation.
    So with face 0 left alone the two sets are identical (40 of 320 faces, measured), and with face
    0 among the flipped ones they are exact complements (3 flagged by MeshLib against 317 by
    ordito). Either way both windings are consistent, which is the property the function promises.

    The bitset is padded to the face count through
    [`meshlib_bitset_to_numpy`][tests.conversions.meshlib_bitset_to_numpy]: it comes back at the
    length of its highest set bit, so an unpadded read fails by shape on exactly the meshes where
    few faces are wrong.
    """
    mesh_tm, _mesh_wp = sphere_irregular
    rng = np.random.default_rng(3)
    n_faces = mesh_tm.faces.shape[0]
    flipped = rng.choice(np.arange(1, n_faces), size=39, replace=False)
    flipped = np.concatenate([[0], flipped]) if flip_seed_face else flipped

    faces_np = mesh_tm.faces.copy()
    faces_np[flipped] = faces_np[flipped][:, ::-1]

    _vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, faces_np, device)
    fixed_np = od.repair.make_winding_consistent(faces_wp).numpy().reshape(-1, 3)
    flip_set_wp = ~(fixed_np == faces_np).all(axis=1)

    mesh_ml = numpy_to_meshlib(mesh_tm.vertices, faces_np)
    flip_set_ml = meshlib_bitset_to_numpy(mm.findDisorientedFaces(mesh_ml), n_faces)

    assert flip_set_ml.sum() == len(flipped)  # non-vacuity: it found every seeded flip
    if flip_seed_face:
        assert np.array_equal(flip_set_wp, ~flip_set_ml)
    else:
        assert np.array_equal(flip_set_wp, flip_set_ml)

    # Both repairs land on a consistent winding; they differ only in which global sign they pick.
    faces_ml_np = faces_np.copy()
    faces_ml_np[flip_set_ml] = faces_ml_np[flip_set_ml][:, ::-1]
    _vertices_wp, faces_ml_wp = numpy_to_warp(mesh_tm.vertices, faces_ml_np, device)
    assert od.validation.is_winding_consistent(faces_ml_wp) is True
    assert od.validation.is_winding_consistent(
        wp.array(fixed_np.reshape(-1), dtype=wp.int32, device=device)
    )


@pytest.mark.parity("make_winding_consistent", "igl", "trimesh")
def test_make_winding_consistent_matches_igl_and_trimesh(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class B against igl, exactly, and Class B against trimesh, up to the global flip.

    The input has every second face reversed, asserted inconsistent first so the repair cannot pass
    by doing nothing, and must come back consistent as a pure per-face winding operation: corner 0
    preserved, the same vertex set per face.

    igl (cyclic winding): the repaired face table against ``igl.bfs_orient``'s ``FF``. No global
    sign fix is needed and that is the interesting part: both libraries anchor each component on
    its lowest-numbered face, so on a connected mesh the answer is unique rather than determined up
    to a per-component flip. The only transform is [`tests.comparisons.canonical_winding`][],
    because a flip is emitted as a rotation of the reversed triangle. ``bfs_orient``'s second
    return is the per-face **component id**, not a flip mask -- the trap recorded in
    ``tests/test_validation.py::test_face_flip_mask_matches_igl``. Here it is asserted to be
    single-valued, which is what makes "no global sign fix" a claim about the anchoring rather than
    a coincidence of this fixture.

    trimesh: ``trimesh.repair.fix_winding`` reaches a consistent state too, and on one connected
    component the two windings agree face normal for face normal up to that global flip. The
    impossible case is [`test_make_winding_consistent_on_a_non_orientable_mesh`].
    """
    mesh_tm, mesh_wp = sphere_irregular
    faces_flipped = mesh_tm.faces.copy()
    faces_flipped[::2] = faces_flipped[::2][:, ::-1]
    _, faces_wp = numpy_to_warp(mesh_tm.vertices, faces_flipped, mesh_wp.device)

    oriented_igl, components_igl = igl.bfs_orient(
        np.ascontiguousarray(faces_flipped, dtype=np.int64)
    )
    assert np.unique(components_igl).size == 1

    assert od.validation.is_winding_consistent(faces_wp) is False
    repaired_wp = od.repair.make_winding_consistent(faces_wp)
    assert od.validation.is_winding_consistent(repaired_wp) is True

    after = _faces_2d(repaired_wp)
    assert np.array_equal(canonical_winding(after), canonical_winding(np.asarray(oriented_igl)))
    assert np.array_equal(after[:, 0], faces_flipped[:, 0])
    assert np.array_equal(np.sort(after, axis=1), np.sort(faces_flipped, axis=1))

    reference_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=faces_flipped, process=False)
    tm_repair.fix_winding(reference_tm)
    assert bool(reference_tm.is_winding_consistent) is True
    ours_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=after, process=False)
    assert bool(ours_tm.is_winding_consistent) is True
    same = np.allclose(ours_tm.face_normals, reference_tm.face_normals, atol=1e-5)
    opposite = np.allclose(ours_tm.face_normals, -reference_tm.face_normals, atol=1e-5)
    assert same or opposite


def test_make_winding_consistent_on_a_non_orientable_mesh(
    mobius: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: a Moebius band admits no consistent winding, so the pass cannot win.

    What it does instead is worth pinning, because the docstring's post-condition does not hold
    here and a caller has to know what it gets: the flood-fill orients everything it reaches and the
    contradiction is confined to a seam — measured 41 of 4 524 edges, under 1%, against 39 before
    the pass, so this is *not* a repair that partially helps. It stays a pure per-face winding
    operation either way.
    """
    mesh_tm, mesh_wp = mobius
    assert od.validation.is_orientable(mesh_wp.indices) is False

    repaired_wp = od.repair.make_winding_consistent(mesh_wp.indices)
    assert od.validation.is_winding_consistent(repaired_wp) is False

    inconsistent_np = ~od.validation.edge_winding_consistent_mask(repaired_wp).numpy()
    assert 0 < inconsistent_np.sum() < 0.02 * inconsistent_np.size

    before, after = mesh_wp.indices.numpy().reshape(-1, 3), _faces_2d(repaired_wp)
    assert np.array_equal(after[:, 0], before[:, 0])
    assert np.array_equal(np.sort(after, axis=1), np.sort(before, axis=1))
    assert len(mesh_tm.faces) == after.shape[0]


@pytest.mark.parity(
    "make_winding_consistent",
    "pymeshfix",
    benchmarked=False,
    reason="rewinding happens inside load_array, so as with remove_unreferenced_vertices there is "
    "no separable call to time and a row would price the load. Measured on bunny that load is "
    "439.6 ms; the operation inside it cannot be isolated at all.",
)
@pytest.mark.parametrize("n_flipped", [1, 10, 40])
def test_make_winding_consistent_matches_pymeshfix(device: str, n_flipped: int) -> None:
    """
    Class B: both reach a consistent winding, equal **up to the global sign**, which is free.

    The buffer cannot be compared -- ``return_arrays`` reorders the faces and starts each row at a
    different corner even when nothing was repaired -- so the transform is to compare the *property*
    the operation exists to establish plus the enclosed volume, which is what says the two chose the
    same surface rather than merely each choosing something self-consistent. That is more than a
    tautology: ``is_winding_consistent`` alone would pass for a mesh wound entirely inward, and only
    the magnitude match rules out one side having rewound a subset the other left alone.

    The **sign** is compared with ``abs`` deliberately, because it is a genuine free choice and the
    two do disagree. Neither algorithm prefers outward -- pymeshfix leaves an *all*-backwards
    icosphere at volume -3.6587 rather than fixing it -- and each propagates from its own seed face,
    so which of the two orientation classes wins depends on the input. Measured across 1 / 10 / 40 /
    160 / 300 of 320 faces flipped: they agree at +4.0470 for 1 and 10, at **-4.0470** for 160 and
    300, and **split** at 40, where ordito reads +4.0470 and pymeshfix -4.0470. Turning the surface
    inside out is [`make_normals_outward`][ordito.repair.make_normals_outward]'s job, not this
    one's, and asserting a sign here would pin an arbitrary tie-break.

    Parametrized over 1, 10 and 40 flipped faces, which spans both the agreeing and the diverging
    cases.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=2)
    vertices_np, faces_np = np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces).copy()
    flipped_np = np.random.default_rng(7).choice(faces_np.shape[0], n_flipped, replace=False)
    faces_np[flipped_np] = faces_np[flipped_np][:, ::-1]

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    assert not od.validation.is_winding_consistent(faces_wp)  # non-vacuity: input really is broken
    fixed_wp = od.repair.make_winding_consistent(faces_wp)

    tin_pmf = numpy_to_pymeshfix(vertices_np, faces_np)
    assert tin_pmf.n_faces == faces_np.shape[0]  # the loader rewound rather than cutting
    assert tin_pmf.n_points == vertices_np.shape[0]
    vertices_pmf, faces_pmf = pymeshfix_to_numpy(tin_pmf)

    assert od.validation.is_winding_consistent(fixed_wp)
    assert od.validation.is_winding_consistent(numpy_to_warp(vertices_pmf, faces_pmf, device)[1])
    volume_wp = warp_to_trimesh(vertices_wp, fixed_wp).volume
    volume_pmf = tm.Trimesh(vertices_pmf, faces_pmf, process=False).volume
    assert np.isclose(abs(volume_wp), abs(volume_pmf), rtol=1e-5, atol=1e-5)
    assert np.isclose(abs(volume_wp), mesh_tm.volume, rtol=1e-5, atol=1e-5)


def test_make_winding_consistent_keeps_each_components_lowest_face(device: str) -> None:
    """
    Not a library comparison: no reference documents which face of a component keeps its winding.

    The seed is each connected component's lowest-indexed face, which pins the output to the input
    rather than to thread order. Three spheres with 30 % of their faces reversed: the result is
    consistently wound, each component's first face is untouched, and three runs agree exactly.
    A seed taken anywhere else in a component flips that component wholesale, which the first-face
    assert catches whichever face it was.
    """
    sphere = tm.creation.icosphere(subdivisions=2)
    n_per_sphere = sphere.faces.shape[0]
    n_sphere_vertices = sphere.vertices.shape[0]
    vertices_np = np.concatenate(
        [sphere.vertices + np.array([3.0 * index, 0.0, 0.0]) for index in range(3)]
    )
    faces_np = np.concatenate(
        [sphere.faces + index * n_sphere_vertices for index in range(3)]
    ).astype(np.int32)
    flipped = np.random.default_rng(5).random(faces_np.shape[0]) < 0.3
    faces_np[flipped] = faces_np[flipped][:, ::-1]
    _, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    first_faces = np.arange(3) * n_per_sphere
    # Both seed states are exercised: some first faces arrive flipped, some do not.
    assert flipped[first_faces].any()
    assert not flipped[first_faces].all()

    runs = [_faces_2d(od.repair.make_winding_consistent(faces_wp)) for _ in range(3)]
    assert all(np.array_equal(runs[0], run) for run in runs[1:])
    assert np.array_equal(runs[0][first_faces], faces_np[first_faces])
    repaired_wp = wp.array(runs[0].reshape(-1), dtype=wp.int32, device=device)
    assert od.validation.is_winding_consistent(repaired_wp) is True


@pytest.mark.parity("make_volume", "trimesh", "pyvista")
@pytest.mark.parametrize("mesh_name", ["sphere_irregular", "sphere_irregular_hollow"])
def test_make_volume_repairs_inversion(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on the enclosed volume, and a pinned three-way split over what "orient" means.

    Reversing *every* face leaves the winding consistent, so this is the state
    ``make_winding_consistent`` cannot fix and ``make_volume`` exists for. Asserted false before and
    true after -- and then against the two libraries that make the same decision:
    ``trimesh.repair.fix_inversion`` (compared face normal for face normal) and pyvista's
    ``compute_normals(consistent_normals=True, auto_orient_normals=True)``.

    **Two other libraries have an obviously-named filter that does something else, and this fixture
    is the one that tells them apart.** Measured on ``sphere_irregular``, signed volume, target
    +21.572407:

    | input | pyvista | open3d ``orient_triangles`` | pymeshlab ``re_orient_faces_coherently`` |
    |---|---|---|---|
    | 62 of 996 faces reversed (inconsistent) | +21.572407 | +21.572407 | +21.572407 |
    | every face reversed (consistent, inward) | **+21.572407** | **-21.572407** | -21.572407 |

    On a *locally inconsistent* mesh, making the winding coherent recovers the majority orientation
    and therefore looks like this operation -- which is why a probe on that input alone reads all
    three as agreeing. On a consistently **inward** mesh, open3d leaves it inward and pymeshlab
    always does, so both are ``make_winding_consistent``'s counterparts. This test runs on the
    inward fixture and asserts their answers stay negative, so the distinction is pinned rather than
    described.

    **And pyvista parts company on a multi-shell mesh, which is why its equality runs on
    ``sphere_irregular`` only.** ``sphere_irregular_hollow`` is a shell around an inner void, so its
    enclosed solid is ``outer - cavity``; ordito returns **21.0318** and pyvista **22.1130**, i.e.
    pyvista orients each shell outward *from itself* -- including the cavity's, which then adds
    instead of subtracting -- where ordito orients for a positive total. Both are defensible
    readings of "outward" and only one is this function's contract, so the divergence is asserted
    exactly: the two sum to twice the outer shell's volume (21.5724) and differ by twice the
    cavity's (0.5406).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_inward = mesh_tm.faces[:, ::-1].copy()  # reverse every face -> inward normals
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, faces_inward, mesh_wp.device)

    assert od.validation.is_volume(vertices_wp, faces_wp) is False
    repaired_wp = od.repair.make_volume(vertices_wp, faces_wp)
    assert od.validation.is_volume(vertices_wp, repaired_wp) is True

    # Reference: trimesh.repair.fix_inversion also produces an outward-oriented volume.
    inward_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=faces_inward, process=False)
    reference_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=faces_inward, process=False)
    tm_repair.fix_inversion(reference_tm)
    ours_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=_faces_2d(repaired_wp), process=False)
    assert inward_tm.volume < 0.0  # non-vacuity: the input really is inside out
    assert ours_tm.volume > 0.0
    assert np.allclose(ours_tm.face_normals, reference_tm.face_normals, atol=1e-5)

    mesh_o3d = trimesh_to_open3d(inward_tm)
    mesh_o3d.orient_triangles()
    volume_o3d = float(
        tm.Trimesh(
            np.asarray(mesh_o3d.vertices), np.asarray(mesh_o3d.triangles), process=False
        ).volume
    )
    oriented_pv = trimesh_to_pyvista(inward_tm).compute_normals(
        consistent_normals=True, auto_orient_normals=True
    )
    volume_pv = float(
        tm.Trimesh(
            np.asarray(oriented_pv.points), np.asarray(oriented_pv.regular_faces), process=False
        ).volume
    )
    if mesh_tm.body_count > 1:
        # Two shells: pyvista turns the cavity outward too, so its volume gains what ours loses.
        assert volume_pv > ours_tm.volume
        shells = sorted(abs(shell.volume) for shell in mesh_tm.split(only_watertight=False))
        assert np.isclose(volume_pv + ours_tm.volume, 2.0 * shells[-1], rtol=1e-6, atol=0.0)
        assert np.isclose(volume_pv - ours_tm.volume, 2.0 * shells[0], rtol=1e-6, atol=0.0)
    else:
        assert np.isclose(volume_pv, ours_tm.volume, rtol=1e-5, atol=1e-5)

    # The two coherent-but-not-outward filters, pinned on the input that separates them.
    meshset_pml = trimesh_to_pymeshlab(inward_tm)
    meshset_pml.meshing_re_orient_faces_coherently()
    volume_pml = float(
        tm.Trimesh(
            meshset_pml.current_mesh().vertex_matrix(),
            meshset_pml.current_mesh().face_matrix(),
            process=False,
        ).volume
    )
    assert np.isclose(volume_o3d, -ours_tm.volume, rtol=1e-5, atol=1e-5)
    assert np.isclose(volume_pml, -ours_tm.volume, rtol=1e-5, atol=1e-5)


def test_make_normals_outward(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: both defects at once -- inconsistent winding *and* global inversion.

    The composition is the claim: fixing consistency alone can leave the mesh inverted, and
    fixing orientation alone cannot run on inconsistent input, so only the pair together
    produces a volume.
    """
    mesh_tm, mesh_wp = sphere_irregular
    faces_bad = mesh_tm.faces.copy()
    faces_bad[::2] = faces_bad[::2][:, ::-1]  # inconsistent winding
    faces_bad = faces_bad[:, ::-1]  # then invert everything
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, faces_bad, mesh_wp.device)

    assert od.validation.is_volume(vertices_wp, faces_wp) is False
    repaired_wp = od.repair.make_normals_outward(vertices_wp, faces_wp)
    assert od.validation.is_winding_consistent(repaired_wp) is True
    assert od.validation.is_volume(vertices_wp, repaired_wp) is True

    # Reference: trimesh.repair.fix_normals. On a closed mesh outward orientation is unique,
    # so per-face normals must agree exactly (index representation may differ).
    reference_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=faces_bad, process=False)
    tm_repair.fix_normals(reference_tm)
    ours_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=_faces_2d(repaired_wp), process=False)
    assert np.allclose(ours_tm.face_normals, reference_tm.face_normals, atol=1e-5)


def test_make_volume_multibody(sphere_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = sphere_irregular
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)
    n_vertices = vertices_np.shape[0]
    n_faces = faces_np.shape[0]

    # Body A: as-is (outward). Body B: translated and inverted (inward normals).
    vertices_two = np.vstack([vertices_np, vertices_np + np.array([10.0, 0.0, 0.0])])
    faces_two = np.vstack([faces_np, faces_np[:, ::-1] + n_vertices])
    vertices_wp, faces_wp = numpy_to_warp(vertices_two, faces_two, mesh_wp.device)

    body_a = slice(0, 3 * n_faces)
    body_b = slice(3 * n_faces, 6 * n_faces)

    def _body_is_volume(flat_faces_wp: wp.array[wp.int32], body: slice) -> bool:
        sub = wp.array(flat_faces_wp.numpy()[body], dtype=wp.int32, device=mesh_wp.device)
        return od.validation.is_volume(vertices_wp, sub)

    # Initially only body B has inward-facing normals.
    assert _body_is_volume(faces_wp, body_a) is True
    assert _body_is_volume(faces_wp, body_b) is False

    # multibody corrects each connected component independently -> both outward.
    repaired_multi = od.repair.make_volume(vertices_wp, faces_wp, multibody=True)
    assert _body_is_volume(repaired_multi, body_a) is True
    assert _body_is_volume(repaired_multi, body_b) is True

    # The single-body path flips all-or-nothing and cannot orient two opposing bodies outward.
    repaired_single = od.repair.make_volume(vertices_wp, faces_wp, multibody=False)
    ok_a = _body_is_volume(repaired_single, body_a)
    ok_b = _body_is_volume(repaired_single, body_b)
    assert not (ok_a and ok_b)


# --- degenerate / small triangle removal ----------------------------------------------------


_MERGE_TOL = (
    1e-8  # matches ordito.constants.TOLERANCE_MERGE used by triangles.face_nondegenerate_mask
)


def _nondegenerate_ref(vertices_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    """CPU mirror of ``triangles.face_nondegenerate_mask`` (per-edge altitude vs the tolerance)."""
    if faces_np.shape[0] == 0:
        return np.empty(0, dtype=bool)
    v0 = vertices_np[faces_np[:, 0]]
    v1 = vertices_np[faces_np[:, 1]]
    v2 = vertices_np[faces_np[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v0
    area = 0.5 * np.linalg.norm(np.cross(e0, e1), axis=1)
    length_e0 = np.linalg.norm(e0, axis=1)
    length_e1 = np.linalg.norm(e1, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        height_e0 = 2.0 * area / length_e0
        height_e1 = 2.0 * area / length_e1
    return (
        (height_e0 > _MERGE_TOL)
        & (height_e1 > _MERGE_TOL)
        & (length_e0 > _MERGE_TOL)
        & (length_e1 > _MERGE_TOL)
    )


def _uf_find(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = int(parent[x])
    return int(x)


def _collapse_small_triangles_ref(
    vertices_np: np.ndarray, faces_np: np.ndarray, epsilon: float
) -> np.ndarray:
    """
    Return surviving faces of the reference collapse as sorted position triples ``(m, 3, 3)``.

    Mirrors ``ordito.repair.collapse_small_triangles``: each iteration flags faces with doubled
    area below ``2 * epsilon * bbd**2``, merges the two endpoints of each flagged triangle's
    shortest edge (union-find, lowest original index as representative), drops faces that become
    degenerate, and repeats to a fixpoint. Comparing surviving triangles as sorted position triples
    is invariant to vertex reindexing and to the representative chosen among near-coincident
    endpoints.
    """
    vertices = np.asarray(vertices_np, dtype=np.float64)
    faces = np.asarray(faces_np, dtype=np.int64).reshape(-1, 3)
    if faces.shape[0] == 0 or vertices.shape[0] == 0:
        return np.empty((0, 3, 3), dtype=np.float64)

    bbd = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    min_dbl_area = 2.0 * epsilon * bbd * bbd

    while faces.shape[0] > 0:
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        dbl_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        small = dbl_area < min_dbl_area
        if not small.any():
            break

        n = vertices.shape[0]
        parent = np.arange(n)
        for fi in np.flatnonzero(small):
            tri = faces[fi]
            edge_sq = [
                float(np.sum((vertices[tri[1]] - vertices[tri[0]]) ** 2)),
                float(np.sum((vertices[tri[2]] - vertices[tri[1]]) ** 2)),
                float(np.sum((vertices[tri[0]] - vertices[tri[2]]) ** 2)),
            ]
            e = int(np.argmin(edge_sq))
            ra, rb = _uf_find(parent, int(tri[e])), _uf_find(parent, int(tri[(e + 1) % 3]))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        labels = np.array([_uf_find(parent, i) for i in range(n)], dtype=np.int64)
        _, inverse = np.unique(labels, return_inverse=True)
        first = np.full(int(inverse.max()) + 1, n, dtype=np.int64)
        for i in range(n):
            first[inverse[i]] = min(first[inverse[i]], i)
        vertices = vertices[first]
        faces = inverse[faces]
        faces = faces[_nondegenerate_ref(vertices, faces)]

    tris = vertices[faces]
    return np.sort(tris, axis=1)


def _sorted_triangle_positions(vertices_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    if faces_np.shape[0] == 0:
        return np.empty((0, 3, 3), dtype=np.float64)
    return np.sort(vertices_np[faces_np], axis=1)


def _triangle_set_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-4) -> bool:
    """Order-independent equality of two ``(m, 3, 3)`` sorted-triangle sets."""
    if a.shape != b.shape:
        return False
    if a.shape[0] == 0:
        return True
    a_sorted = lexsort_rows(a.reshape(a.shape[0], -1))
    b_sorted = lexsort_rows(b.reshape(b.shape[0], -1))
    return bool(np.allclose(a_sorted, b_sorted, atol=atol))


# --------------------------------------------------------------------------------------
# non-manifold input builders, shared by remove_non_manifold_faces and split_non_manifold_vertices
# --------------------------------------------------------------------------------------


def _bowtie_np() -> tuple[np.ndarray, np.ndarray]:
    """Two triangles meeting at a single vertex: edge-manifold, vertex-non-manifold."""
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    return vertices_np, np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int32)


def _three_faces_on_one_edge_np() -> tuple[np.ndarray, np.ndarray]:
    """Build a fan of three faces on edge ``(0, 1)``, all wound the same way round it."""
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return vertices_np, np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int32)


# --------------------------------------------------------------------------------------
# remove_non_manifold_faces
# --------------------------------------------------------------------------------------


def _remove_non_manifold_faces_np(
    vertices_np: np.ndarray, faces_np: np.ndarray, max_iter: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Drop faces carrying a >2-incident edge, iterating, then compact the vertices (numpy)."""
    faces_np = faces_np.copy()
    for _ in range(max_iter):
        if faces_np.shape[0] == 0:
            break
        # ``undirected_edges`` already orders each row min-first, face-major, three per face.
        edges_np = undirected_edges(faces_np)
        _unique_np, inverse_np, counts_np = np.unique(
            edges_np, axis=0, return_inverse=True, return_counts=True
        )
        bad_np = counts_np[inverse_np].reshape(-1, 3).max(axis=1) > 2
        if not bad_np.any():
            break
        faces_np = faces_np[~bad_np]
    used_np = np.unique(faces_np) if faces_np.size else np.zeros(0, dtype=np.int64)
    remap_np = np.full(vertices_np.shape[0], -1, dtype=np.int64)
    remap_np[used_np] = np.arange(used_np.size)
    kept_np = remap_np[faces_np] if faces_np.size else faces_np.reshape(0, 3)
    return vertices_np[used_np], kept_np


def _cascading_non_manifold_np() -> tuple[np.ndarray, np.ndarray]:
    """Build a three-face fan plus a face hanging off it, so one removal pass is not enough."""
    vertices_np, faces_np = _three_faces_on_one_edge_np()
    vertices_np = np.vstack((vertices_np, [[2.0, 0.0, 0.0]])).astype(np.float32)
    return vertices_np, np.vstack((faces_np, [[1, 2, 5]])).astype(np.int32)


def _icosahedron_plus_a_face_on_an_existing_edge_np() -> tuple[np.ndarray, np.ndarray]:
    """Glue one extra face to an existing edge of a closed mesh: most of it must survive."""
    mesh_tm = tm.creation.icosahedron()
    vertices_np = np.vstack((mesh_tm.vertices, [[3.0, 3.0, 3.0]])).astype(np.float32)
    edge_np = mesh_tm.faces[0][:2]
    extra_np = [[edge_np[0], edge_np[1], mesh_tm.vertices.shape[0]]]
    return vertices_np, np.vstack((mesh_tm.faces, extra_np)).astype(np.int32)


def _edge_manifold_igl(faces_np: np.ndarray) -> bool:
    """
    ``igl.is_edge_manifold``'s verdict, which is the ``allow_boundary_edges=True`` form.

    It returns a **5-tuple** ``(verdict, BF, E, EMAP, BE)``, so ``bool(igl.is_edge_manifold(F))`` is
    ``bool`` of a non-empty tuple and reads ``True`` for every input ever passed to it. Only ``[0]``
    is the answer.
    """
    return bool(igl.is_edge_manifold(np.ascontiguousarray(faces_np, dtype=np.int64))[0])


def _edge_manifold_o3d(vertices_np: np.ndarray, faces_np: np.ndarray) -> bool:
    """Open3D's verdict, which shares ordito's ``allow_boundary_edges`` switch exactly."""
    mesh_tm = tm.Trimesh(
        np.asarray(vertices_np, dtype=np.float64), np.asarray(faces_np), process=False
    )
    return bool(trimesh_to_open3d(mesh_tm).is_edge_manifold(allow_boundary_edges=True))


@pytest.mark.parity(
    "remove_non_manifold_faces",
    "igl",
    "open3d",
    benchmarked=False,
    reason="both are used here as the post-condition detector rather than as the repair. libigl "
    "binds no non-manifold face remover at all, so there is nothing on its side to time; Open3D's "
    "remove_non_manifold_edges is a comparable repair and would be its own row. What this claims "
    "is is_edge_manifold(allow_boundary_edges=True), which both answer identically to ordito and "
    "which is timed in the is_edge_manifold group.",
)
@pytest.mark.parametrize(
    ("mesh_kind", "n_surviving"),
    [("three_faces_on_one_edge", 0), ("cascading", 1), ("icosahedron_plus_a_face", 18)],
)
def test_remove_non_manifold_faces_matches_a_numpy_oracle(
    device: str, mesh_kind: str, n_surviving: int
) -> None:
    """
    Class A twice: the numpy oracle pins which faces go, two other libraries pin the result.

    The expected count is pinned per input because the interesting property is *which* faces go: the
    three-face fan loses all three (every one of them carries the 3-incident edge), the cascading
    input needs a second pass to reach the single survivor a one-pass implementation would miss, and
    the icosahedron keeps 18 of its 21 faces -- the substantive case, since two empty answers would
    compare equal.

    The numpy oracle pins *which* faces survive; it says nothing about whether the result is
    actually edge-manifold, because it runs the same rule ordito does. So that half was asserted
    with [`is_edge_manifold`][ordito.validation.is_edge_manifold] -- ordito's repair checked by
    ordito's detector, where a shared bug in the detector passes both sides. igl and open3d both
    answer the identical question and both flip with ordito: measured False -> True on all three
    inputs here, and on an ``icosphere(2)`` carrying one extra face (163 V / 321 F -> 162 / 318).

    **pyvista is deliberately absent.** Its ``is_manifold`` is ``n_open_edges == 0``, i.e. the
    ``allow_boundary_edges=False`` form, and removing the faces on an over-incident edge *opens* the
    surface -- measured False both before and after on that same icosphere, so it cannot see this
    post-condition at all.
    """
    builders = {
        "three_faces_on_one_edge": _three_faces_on_one_edge_np,
        "cascading": _cascading_non_manifold_np,
        "icosahedron_plus_a_face": _icosahedron_plus_a_face_on_an_existing_edge_np,
    }
    vertices_np, faces_np = builders[mesh_kind]()
    faces_wp = wp.array(np.ascontiguousarray(faces_np).ravel(), dtype=wp.int32, device=device)
    manifold_igl = _edge_manifold_igl(faces_np)
    manifold_o3d = _edge_manifold_o3d(vertices_np, faces_np)
    assert not od.validation.is_edge_manifold(faces_wp, allow_boundary_edges=True)
    assert not manifold_igl
    assert not manifold_o3d

    new_vertices_wp, new_faces_wp = od.repair.remove_non_manifold_faces(
        points_to_warp(vertices_np, device), faces_wp
    )
    new_faces_np = new_faces_wp.numpy().reshape(-1, 3)
    expected_vertices_np, expected_faces_np = _remove_non_manifold_faces_np(vertices_np, faces_np)

    assert new_faces_np.shape[0] == n_surviving
    assert np.array_equal(new_faces_np, expected_faces_np)
    assert np.allclose(new_vertices_wp.numpy(), expected_vertices_np, rtol=1e-5, atol=1e-5)
    if n_surviving > 0:
        fixed_igl = _edge_manifold_igl(new_faces_np)
        fixed_o3d = _edge_manifold_o3d(new_vertices_wp.numpy(), new_faces_np)
        assert od.validation.is_edge_manifold(new_faces_wp, allow_boundary_edges=True)
        assert fixed_igl
        assert fixed_o3d


# --------------------------------------------------------------------------------------
# remove_degenerate_and_non_manifold_faces
# --------------------------------------------------------------------------------------


def _icosahedron_with_a_degenerate_and_a_non_manifold_face_np() -> tuple[np.ndarray, np.ndarray]:
    """
    Glue a repeated-index face onto one icosahedron edge and a real face onto another.

    The degenerate face ``(a, b, b)`` carries edge ``(a, b)``, so it makes that edge 3-incident
    unless it is dropped *before* the manifold test -- the ordering the combined function must
    keep. The real extra face makes a second edge genuinely non-manifold. A trailing unreferenced
    vertex rides along so the compaction has something to drop.
    """
    mesh_tm = tm.creation.icosahedron()
    n = mesh_tm.vertices.shape[0]
    vertices_np = np.vstack((mesh_tm.vertices, [[3.0, 3.0, 3.0], [9.0, 9.0, 9.0]]))
    a, b = mesh_tm.faces[0][:2]
    c, d = mesh_tm.faces[7][1:]
    extra_np = [[a, b, b], [c, d, n]]
    faces_np = np.vstack((mesh_tm.faces[:5], extra_np[:1], mesh_tm.faces[5:], extra_np[1:]))
    return vertices_np.astype(np.float32), faces_np.astype(np.int32)


@pytest.mark.parametrize("max_iter", [1, 3])
def test_remove_degenerate_and_non_manifold_faces_matches_trimesh_and_a_numpy_oracle(
    device: str, max_iter: int
) -> None:
    """
    Class A: trimesh picks the degenerate faces, the numpy oracle the non-manifold ones.

    The reference is the two-stage rule written out with no intermediate compaction: trimesh's
    ``nondegenerate_faces`` filters the input, and ``_remove_non_manifold_faces_np`` runs the
    manifold passes on what is left and compacts once. The fixture is chosen so both stages bite
    and so their order matters: 22 faces in, one degenerate (on an edge it would otherwise make
    non-manifold) and one genuinely non-manifold, 18 out -- where dropping the degenerate face
    *after* the manifold test would also delete the two real faces on its edge.
    """
    vertices_np, faces_np = _icosahedron_with_a_degenerate_and_a_non_manifold_face_np()
    mesh_tm = tm.Trimesh(vertices_np.astype(np.float64), faces_np, process=False)
    nondegenerate_tm = mesh_tm.nondegenerate_faces(height=_MERGE_TOL)
    assert int((~nondegenerate_tm).sum()) == 1
    expected_vertices_np, expected_faces_np = _remove_non_manifold_faces_np(
        vertices_np, faces_np[nondegenerate_tm], max_iter=max_iter
    )
    assert expected_faces_np.shape[0] == 18
    assert expected_vertices_np.shape[0] < vertices_np.shape[0]

    new_vertices_wp, new_faces_wp = od.repair.remove_degenerate_and_non_manifold_faces(
        *numpy_to_warp(vertices_np, faces_np, device), max_iter=max_iter
    )

    assert np.array_equal(new_faces_wp.numpy().reshape(-1, 3), expected_faces_np)
    assert np.allclose(new_vertices_wp.numpy(), expected_vertices_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("mesh_kind", "max_iter"),
    [
        # Only the cascading input needs a second pass, so only it can tell 2 from 3.
        *(("cascading", max_iter) for max_iter in (1, 2, 3)),
        *(
            (kind, max_iter)
            for kind in ("three_faces_on_one_edge", "icosahedron_plus_a_face", "degenerate_and_nm")
            for max_iter in (1, 3)
        ),
    ],
)
def test_remove_degenerate_and_non_manifold_faces_equals_the_two_calls(
    device: str, mesh_kind: str, max_iter: int
) -> None:
    """
    Ordito against ordito: the one-compaction path is byte-identical to the two public calls.

    The two-call sequence carries the oracle (the numpy and trimesh comparisons above and in the
    ``remove_non_manifold_faces`` / ``remove_degenerate_faces`` tests); this pins the combined
    entry point to it exactly -- vertex bytes and face indices -- on every input shape the
    stopping rule distinguishes: everything removed, a second pass needed, one pass enough, and
    a degenerate face whose removal decides the manifold test.
    """
    builders = {
        "three_faces_on_one_edge": _three_faces_on_one_edge_np,
        "cascading": _cascading_non_manifold_np,
        "icosahedron_plus_a_face": _icosahedron_plus_a_face_on_an_existing_edge_np,
        "degenerate_and_nm": _icosahedron_with_a_degenerate_and_a_non_manifold_face_np,
    }
    vertices_wp, faces_wp = numpy_to_warp(*builders[mesh_kind](), device)

    staged_vertices_wp, staged_faces_wp = od.repair.remove_non_manifold_faces(
        *od.repair.remove_degenerate_faces(vertices_wp, faces_wp), max_iter=max_iter
    )
    new_vertices_wp, new_faces_wp = od.repair.remove_degenerate_and_non_manifold_faces(
        vertices_wp, faces_wp, max_iter=max_iter
    )

    assert new_vertices_wp.numpy().tobytes() == staged_vertices_wp.numpy().tobytes()
    assert np.array_equal(new_faces_wp.numpy(), staged_faces_wp.numpy())


def test_remove_degenerate_and_non_manifold_faces_empty(device: str) -> None:
    """Not a library comparison: a face-less input comes back as copies, vertices kept."""
    vertices_wp = wp.array(np.eye(3, dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = warp_empty(0, wp.int32, device)

    new_vertices_wp, new_faces_wp = od.repair.remove_degenerate_and_non_manifold_faces(
        vertices_wp, faces_wp
    )

    assert new_faces_wp.size == 0
    assert np.array_equal(new_vertices_wp.numpy(), np.eye(3, dtype=np.float32))
    assert new_vertices_wp.ptr != vertices_wp.ptr


# --------------------------------------------------------------------------------------
# remove_small_components
# --------------------------------------------------------------------------------------


_ThreeShells = tuple[tm.Trimesh, wp.array[wp.vec3], wp.array[wp.int32]]


@pytest.fixture
def three_shells(device: str) -> _ThreeShells:
    """
    Three disjoint spheres whose face-count, area and diameter rankings all **disagree**.

    80 faces / area 11.666 / diagonal 3.464, then 320 / 12.330 / 3.464, then 80 / 1166.593 / 34.641
    at radius 10. So the component with the most *faces* is neither the largest by area nor the
    largest by diameter, which is what makes each of the four criteria testable against the others
    -- a fixture whose rankings agreed would pass for any of them.
    """
    small_tm = tm.creation.icosphere(subdivisions=1)
    dense_tm = tm.creation.icosphere(subdivisions=2)
    dense_tm.apply_translation([5.0, 0.0, 0.0])
    wide_tm = tm.creation.icosphere(subdivisions=1, radius=10.0)
    wide_tm.apply_translation([0.0, 40.0, 0.0])
    joined_tm = tm.util.concatenate([small_tm, dense_tm, wide_tm])
    assert isinstance(joined_tm, tm.Trimesh)
    return (joined_tm, *numpy_to_warp(joined_tm.vertices, joined_tm.faces, device))


def _assert_same_surviving_mesh(
    kept_wp: wp.array[wp.vec3],
    kept_faces_wp: wp.array[wp.int32],
    vertices_ref: np.ndarray,
    faces_ref: np.ndarray,
) -> None:
    """
    Assert two survivor meshes are the same surface, both sides having renumbered independently.

    Every reference here compacts its vertex buffer after deleting, in its own order, so the index
    buffers are incomparable by construction. What is comparable exactly is the *set* of surviving
    positions -- a two-sided Hausdorff distance of zero says each side's vertices are the other's --
    together with the counts and the total area, which together pin *which* components survived
    rather than merely how many.

    The distance bound is **scale-relative**, at ``1e-5`` of the scene's bounding-box diagonal,
    because the floor is ordito's ``float32`` vertex buffer rather than any disagreement: measured
    on the three-shell fixture, whose furthest vertex is 43 units out and whose diagonal is 58.3,
    the two sides differ by **1.87e-06** -- exactly float32 quantization there, and a bound of
    ``1e-6`` absolute would fail on an answer that is correct to the last bit.
    """
    kept_tm = warp_to_trimesh(kept_wp, kept_faces_wp)
    reference_tm = tm.Trimesh(vertices_ref, faces_ref, process=False)
    scale = float(np.linalg.norm(vertices_ref.max(axis=0) - vertices_ref.min(axis=0)))
    assert kept_faces_wp.size // 3 == faces_ref.shape[0]
    assert kept_wp.size == vertices_ref.shape[0]
    assert hausdorff_two_sided(kept_wp.numpy().astype(np.float64), vertices_ref) < 1e-5 * scale
    assert np.isclose(kept_tm.area, reference_tm.area, rtol=1e-5, atol=1e-5)


@pytest.mark.parity(
    "remove_small_components",
    "pymeshfix",
    benchmarked=False,
    reason="remove_smallest_components is 9-12 % of a pymeshfix round -- 6.7 ms against 67.9 ms of "
    "load on bunny_decimated, 60.5 against 439.6 on bunny -- and a PyTMesh accepts exactly one "
    "load_array, so the build cannot leave the timed callable and a row would report the load as a "
    "component filter. pymeshlab and open3d carry the timed rows.",
)
def test_remove_small_components_keep_largest_matches_pymeshfix(three_shells: _ThreeShells) -> None:
    """
    Class B (both sides renumber): ``keep_largest`` is exactly ``remove_smallest_components``.

    The rule is measured, not read off a docstring, and it is the reason ``keep_largest`` ranks by
    face count: handed 80 / 320 / 80 faces where the third shell is by far the largest in area
    (1166.593 against 12.330) and in diameter (34.641 against 3.464), pymeshfix removes **2** and
    keeps the **320-face** one. So "largest" there means most triangles, and a fixture whose three
    rankings agreed could not have told the three apart.

    Both sides reduce to exactly one component, which is asserted as an invariant rather than
    inferred: nothing in the surface comparison would notice two survivors that happened to sum to
    the right area.
    """
    mesh_tm, vertices_wp, faces_wp = three_shells

    kept_wp, kept_faces_wp = od.repair.remove_small_components(
        vertices_wp, faces_wp, keep_largest=True
    )

    tin_pmf = trimesh_to_pymeshfix(mesh_tm)
    assert tin_pmf.n_faces == mesh_tm.faces.shape[0]  # the loader left the mesh alone
    assert tin_pmf.remove_smallest_components() == 2  # non-vacuity: it really removed two
    vertices_pmf, faces_pmf = pymeshfix_to_numpy(tin_pmf)

    assert faces_pmf.shape[0] == 320  # face count, not area or diameter
    _assert_same_surviving_mesh(kept_wp, kept_faces_wp, vertices_pmf, faces_pmf)
    assert len(warp_to_trimesh(kept_wp, kept_faces_wp).split(only_watertight=False)) == 1


@pytest.mark.parity("remove_small_components", "pymeshlab")
@pytest.mark.parametrize(("criterion", "threshold"), [("min_faces", 81), ("min_diameter", 3.4651)])
def test_remove_small_components_matches_pymeshlab(
    three_shells: _ThreeShells, criterion: str, threshold: float
) -> None:
    """
    Class B (both sides renumber): the face-count and diameter criteria, thresholds included.

    ``min_faces`` is ``meshing_remove_connected_component_by_face_number(mincomponentsize=...)`` and
    ``min_diameter`` is ``meshing_remove_connected_component_by_diameter(mincomponentdiag=...)``,
    both keeping a component whose measure **reaches** the threshold. That inclusivity is measured
    rather than assumed, at both bounds: with the fixture's 80-face shells,
    ``mincomponentsize=80`` keeps all 480 faces and 81 keeps 320; with its 3.4641016-diagonal
    shells, ``mincomponentdiag`` at exactly that value keeps all 480 and a hair above it keeps 80.
    The thresholds parametrized here sit on the far side of each boundary, so the two criteria
    select *different* components -- 320 faces against 80 -- which is what makes the pair
    non-vacuous.

    ``mincomponentdiag`` is passed as a ``PureValue``: its own default is a ``PercentageValue`` of
    the bounding-box diagonal, and the fixture's shells are 3.464 across in a scene 58.3 across, so
    a percentage would silently be measuring something else.

    ``removeunref=True`` matches ordito, which compacts the vertex buffer of any submesh it
    extracts; at ``False`` the reference would keep the deleted components' vertices and the
    position-set comparison would fail on the vertex count alone.
    """
    mesh_tm, vertices_wp, faces_wp = three_shells

    kept_wp, kept_faces_wp = od.repair.remove_small_components(
        vertices_wp, faces_wp, **cast("_ComponentCriteria", {criterion: threshold})
    )

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    if criterion == "min_faces":
        meshset_pml.meshing_remove_connected_component_by_face_number(
            mincomponentsize=int(threshold), removeunref=True
        )
    else:
        meshset_pml.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=ml.PureValue(float(threshold)), removeunref=True
        )
    vertices_pml = meshset_pml.current_mesh().vertex_matrix()
    faces_pml = meshset_pml.current_mesh().face_matrix()

    # Non-vacuity in both directions: something survived, and something was removed.
    assert 0 < faces_pml.shape[0] < mesh_tm.faces.shape[0]
    _assert_same_surviving_mesh(kept_wp, kept_faces_wp, vertices_pml, faces_pml)


@pytest.mark.parity("remove_small_components", "open3d")
def test_remove_small_components_min_area_matches_open3d(three_shells: _ThreeShells) -> None:
    """
    Class B (both sides renumber): the area criterion, against open3d's per-cluster areas.

    open3d has no component *filter* -- ``cluster_connected_triangles`` returns the per-triangle
    cluster id together with each cluster's triangle count and its **area**, and the caller builds
    the mask -- so this is the reference for ``min_area`` and the transform is that mask
    construction. Its areas match trimesh's per-shell areas exactly (11.6659 / 12.3298 / 1166.5931),
    which is what makes the comparison a comparison rather than two independent thresholdings.

    The threshold sits between the fixture's two small shells, so the answer is two components of
    three: the shell with the most *faces* survives here alongside the one with the largest area,
    and the one with the fewest of both is dropped. A threshold outside that range would agree with
    ``keep_largest`` or with the identity and test nothing new.
    """
    mesh_tm, vertices_wp, faces_wp = three_shells
    min_area = 12.0

    kept_wp, kept_faces_wp = od.repair.remove_small_components(
        vertices_wp, faces_wp, min_area=min_area
    )

    mesh_o3d = trimesh_to_open3d(mesh_tm)
    clusters_o3d, _counts_o3d, areas_o3d = mesh_o3d.cluster_connected_triangles()
    areas_np = np.asarray(areas_o3d)
    assert areas_np.size == 3  # non-vacuity: the reference found the three shells
    mesh_o3d.remove_triangles_by_mask(
        cast("Sequence[bool]", areas_np[np.asarray(clusters_o3d)] < min_area)
    )
    mesh_o3d.remove_unreferenced_vertices()
    vertices_o3d = np.asarray(mesh_o3d.vertices)
    faces_o3d = np.asarray(mesh_o3d.triangles)

    assert faces_o3d.shape[0] == 400  # the two shells above the threshold, 320 + 80
    _assert_same_surviving_mesh(kept_wp, kept_faces_wp, vertices_o3d, faces_o3d)


def test_remove_small_components_invariants(three_shells: _ThreeShells) -> None:
    """
    Not a library comparison: the three properties the criteria share, on the same fixture.

    ``min_faces=1`` is the identity (every component has at least one face), the survivors are
    always a subset of the input triangles as unordered vertex-position rows, and a threshold above
    everything empties the mesh rather than raising. None of the three is visible in a reference
    comparison, which asserts only that two implementations agree on one threshold.
    """
    mesh_tm, vertices_wp, faces_wp = three_shells
    n_faces = mesh_tm.faces.shape[0]

    identity_wp, identity_faces_wp = od.repair.remove_small_components(
        vertices_wp, faces_wp, min_faces=1
    )
    assert identity_faces_wp.size // 3 == n_faces
    assert identity_wp.size == mesh_tm.vertices.shape[0]

    kept_wp, kept_faces_wp = od.repair.remove_small_components(
        vertices_wp, faces_wp, keep_largest=True
    )
    kept_rows = np.round(warp_to_trimesh(kept_wp, kept_faces_wp).triangles, 5).reshape(-1, 9)
    input_rows = np.round(mesh_tm.triangles, 5).reshape(-1, 9)
    assert {tuple(row) for row in kept_rows} <= {tuple(row) for row in input_rows}

    _empty_wp, empty_faces_wp = od.repair.remove_small_components(
        vertices_wp, faces_wp, min_faces=n_faces + 1
    )
    assert empty_faces_wp.size == 0


# --------------------------------------------------------------------------------------
# split_non_manifold_vertices
# --------------------------------------------------------------------------------------


def _split_nonmanifold_wp(
    vertices_np: np.ndarray, faces_np: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``split_non_manifold_vertices`` on NumPy input and bring all three results back."""
    vertices_wp = points_to_warp(vertices_np, device)
    faces_wp = wp.array(np.ascontiguousarray(faces_np).ravel(), dtype=wp.int32, device=device)
    new_vertices_wp, new_faces_wp, source_wp = od.repair.split_non_manifold_vertices(
        vertices_wp, faces_wp
    )
    return new_vertices_wp.numpy(), new_faces_wp.numpy().reshape(-1, 3), source_wp.numpy()


def _assert_split_is_manifold_and_final(
    vertices_np: np.ndarray, faces_np: np.ndarray, device: str
) -> None:
    """Assert the split's post-condition: edge-manifold out, every face kept, a fixpoint."""
    vertices_wp = points_to_warp(vertices_np, device)
    faces_wp = wp.array(np.ascontiguousarray(faces_np).ravel(), dtype=wp.int32, device=device)
    new_vertices_wp, new_faces_wp, _source_wp = od.repair.split_non_manifold_vertices(
        vertices_wp, faces_wp
    )
    assert od.validation.is_edge_manifold(new_faces_wp)
    assert new_faces_wp.size == faces_wp.size
    assert new_vertices_wp.size >= vertices_wp.size
    # Idempotent: a second pass has nothing left to split.
    again_vertices_wp, again_faces_wp, _ = od.repair.split_non_manifold_vertices(
        new_vertices_wp, new_faces_wp
    )
    assert again_vertices_wp.size == new_vertices_wp.size
    assert np.array_equal(again_faces_wp.numpy(), new_faces_wp.numpy())


@pytest.mark.parametrize(
    "mesh_kind", ["manifold", "bowtie", "three_faces_on_one_edge", "flipped_face", "boundary"]
)
@pytest.mark.parity("split_non_manifold_vertices", "igl")
def test_split_nonmanifold_matches_igl(
    mesh_kind: str, device: str, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class B: the same vertex split as ``igl.split_nonmanifold``, up to which copy gets which index.

    Neither library's vertex numbering is meaningful -- both invent copies -- so what is compared is
    the *partition of corners* the two induce, canonicalised by first occurrence with
    [`same_partition`][tests.comparisons.same_partition]. That is exact rather than tolerant, and it
    is the whole answer: given the partition, the face table and the source map follow.

    The two implementations are unrelated. ordito takes connected components of a corner graph in
    one parallel pass; ``igl::split_nonmanifold`` explodes the mesh into ``3 * n_faces`` singleton
    vertices and greedily re-merges pairs, re-testing manifoldness after each candidate. They agree
    on every input class here, including the two that a naive implementation gets wrong: a fan of
    three faces round one edge must split into three (not two-plus-one), and a face wound against
    its neighbours must be cut free rather than silently kept.

    ``manifold`` is the identity case and is *not* vacuous coverage -- it is asserted to return the
    input unchanged, which an implementation that split on every edge would fail.

    Not a parity assert, beside it: the post-condition callers actually depend on, checked
    independently of igl -- edge-manifold out, the same face count, and a second pass with nothing
    left to split -- with each defective input first asserted to violate the precondition it
    targets (a bowtie is edge-manifold with a non-manifold vertex, a flipped face is manifold but
    not orientable). The duplicated-face input, where the two libraries split differently, carries
    the same post-condition in the next test.
    """
    if mesh_kind == "bowtie":
        vertices_np, faces_np = _bowtie_np()
        expected_new = 6
    elif mesh_kind == "three_faces_on_one_edge":
        vertices_np, faces_np = _three_faces_on_one_edge_np()
        expected_new = 9  # every corner its own vertex: no pair of the three is oppositely wound
    elif mesh_kind == "flipped_face":
        mesh_tm, _ = sphere_irregular
        vertices_np = mesh_tm.vertices.astype(np.float32)
        faces_np = mesh_tm.faces.astype(np.int32).copy()
        faces_np[-1] = faces_np[-1][::-1]
        # The flipped face is cut free of all three neighbours: one new copy per corner.
        expected_new = int(vertices_np.shape[0]) + 3
    elif mesh_kind == "boundary":
        sphere_tm, _sphere_wp = sphere_irregular
        hemisphere_tm = sphere_tm.slice_plane(
            plane_origin=sphere_tm.vertices.mean(axis=0),
            plane_normal=np.array([0.0, 0.0, 1.0]),
            cap=False,
        )
        hemisphere_tm.merge_vertices()
        vertices_np = hemisphere_tm.vertices.astype(np.float32)
        faces_np = hemisphere_tm.faces.astype(np.int32)
        expected_new = int(vertices_np.shape[0])  # a boundary is not a reason to split
    else:
        mesh_tm, _ = sphere_irregular
        vertices_np = mesh_tm.vertices.astype(np.float32)
        faces_np = mesh_tm.faces.astype(np.int32)
        expected_new = int(vertices_np.shape[0])

    input_faces_wp = wp.array(np.ascontiguousarray(faces_np).ravel(), dtype=wp.int32, device=device)
    if mesh_kind == "three_faces_on_one_edge":
        assert not od.validation.is_edge_manifold(input_faces_wp)
    elif mesh_kind == "flipped_face":
        assert not od.validation.is_winding_consistent(input_faces_wp)
    elif mesh_kind == "bowtie":
        assert not od.validation.is_vertex_manifold(input_faces_wp)

    new_vertices_np, new_faces_np, source_np = _split_nonmanifold_wp(vertices_np, faces_np, device)

    faces_igl, source_igl = igl.split_nonmanifold(
        np.ascontiguousarray(faces_np, dtype=np.int64).reshape(-1, 3)
    )
    assert np.asarray(source_igl).size == expected_new, "the reference produced the expected split"
    assert new_vertices_np.shape[0] == expected_new
    assert same_partition(new_faces_np.ravel(), np.asarray(faces_igl).ravel())

    # The split moves nothing and drops nothing: every copy sits on the vertex it came from.
    assert new_faces_np.shape == faces_np.reshape(-1, 3).shape
    assert np.allclose(new_vertices_np, vertices_np[source_np], atol=1e-6)
    assert np.array_equal(np.sort(source_igl), np.sort(source_np.astype(np.int64)))
    _assert_split_is_manifold_and_final(vertices_np, faces_np, device)


@pytest.mark.parametrize("mesh_kind", ["bowtie", "three_faces_on_one_edge"])
@pytest.mark.parity("split_non_manifold_vertices", "meshlib")
def test_split_nonmanifold_matches_meshlib(mesh_kind: str, device: str) -> None:
    """
    Class B on the final vertex count: ``duplicateMultiHoleVertices`` mutates and returns a count.

    The named transform is *where the split happens*, and it is not the same place on both sides.
    MeshLib's half-edge builder cannot represent a non-edge-manifold topology at all, so
    ``meshFromFacesVerts`` splits the offending vertices while **constructing** the mesh: the
    three-faces-on-one-edge input arrives with 9 vertices from a 5-vertex array, and
    ``duplicateMultiHoleVertices`` then reports **0**. On the bowtie, where the defect is a vertex
    rather than an edge, the build is faithful and the call reports **1**. So the returned count is
    not the comparable quantity -- the vertex total after both steps is, and it agrees exactly (6
    and 9), as does the face count, which neither side may change.

    That makes this the pair that pins ordito's promise to keep every face: MeshLib reaches the
    same manifold vertex set by two different routes and never drops a triangle either.
    """
    builders = {"bowtie": _bowtie_np, "three_faces_on_one_edge": _three_faces_on_one_edge_np}
    vertices_np, faces_np = builders[mesh_kind]()

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    assert not od.validation.is_vertex_manifold(faces_wp)  # non-vacuity: there is a defect to fix
    split_wp, split_faces_wp, _source_wp = od.repair.split_non_manifold_vertices(
        vertices_wp, faces_wp
    )

    mesh_ml = numpy_to_meshlib(vertices_np, faces_np)
    n_duplicated_ml = mm.duplicateMultiHoleVertices(mesh_ml)
    mesh_ml.pack()

    expected_count = {"bowtie": 1, "three_faces_on_one_edge": 0}[mesh_kind]
    assert n_duplicated_ml == expected_count
    assert split_wp.size == mesh_ml.topology.numValidVerts()
    assert split_faces_wp.size // 3 == mesh_ml.topology.numValidFaces()
    assert split_faces_wp.size // 3 == faces_np.shape[0]
    assert od.validation.is_vertex_manifold(split_faces_wp)


def test_split_nonmanifold_splits_a_duplicated_face_further_than_igl(
    device: str, sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class D exemption, pinned: the one documented divergence from igl, on a duplicated face.

    Where an edge carries one half-edge in one direction and several in the other -- what a
    duplicated face produces -- igl keeps one arbitrarily chosen pair joined while ordito
    splits every copy. Both answers are manifold with the input's face count; ordito's is
    order-independent and makes no arbitrary choice.

    Asserted as exact counts in both directions, so a change to either library's rule fails here
    rather than quietly altering the output of a repair function. The split's post-condition --
    edge-manifold out, every face kept, a fixpoint -- holds here as on every other input.
    """
    mesh_tm, _ = sphere_irregular
    vertices_np = mesh_tm.vertices.astype(np.float32)
    faces_np = np.vstack([mesh_tm.faces.astype(np.int32), mesh_tm.faces.astype(np.int32)[:1]])

    new_vertices_np, _new_faces_np, _source_np = _split_nonmanifold_wp(
        vertices_np, faces_np, device
    )
    _faces_igl, source_igl = igl.split_nonmanifold(
        np.ascontiguousarray(faces_np, dtype=np.int64).reshape(-1, 3)
    )

    n_vertices = vertices_np.shape[0]
    # The duplicated face's three corners each split three ways (+6); igl joins one pair (+3).
    assert new_vertices_np.shape[0] == n_vertices + 6, (
        "all three copies of each shared vertex split"
    )
    assert np.asarray(source_igl).size == n_vertices + 3, "igl keeps one pair of the three joined"
    # Both answers are legal repairs of the same input: manifold, with every face kept.
    assert _new_faces_np.shape[0] == faces_np.shape[0] == np.asarray(_faces_igl).shape[0]
    faces_wp = wp.array(np.ascontiguousarray(faces_np).ravel(), dtype=wp.int32, device=device)
    assert not od.validation.is_edge_manifold(faces_wp)
    _assert_split_is_manifold_and_final(vertices_np, faces_np, device)
    # And `resolve_duplicated_faces` is *not* the escape hatch here: libigl's cancellation rules
    # cover a +1/-1 imbalance, so a face duplicated in the *same* orientation makes it raise.
    with pytest.raises(ValueError, match="non-orientable duplicate face group"):
        od.repair.resolve_duplicated_faces(faces_wp)


@pytest.mark.parity(
    "split_non_manifold_vertices",
    "pymeshfix",
    benchmarked=False,
    reason="the cut happens inside load_array, which also drops unreferenced vertices and rewinds, "
    "so a row would price the whole loader (67.9 ms on bunny_decimated) under this group's name "
    "and would be timing three operations at once. Nor does the standalone fix_connectivity help: "
    "measured at 57.3 % of a round on bunny_decimated and provably a no-op, leaving 8 372 V / "
    "16 220 F / 86 boundaries unchanged because load_array already ran it. The cut itself is what "
    "this test compares.",
)
def test_split_non_manifold_vertices_matches_pymeshfix(device: str) -> None:
    """
    Class B on the one input where the minimal cut is **unique**: the bowtie.

    Two triangles meeting at a single vertex have exactly one way to be pulled apart -- duplicate
    that vertex -- so both sides go from 5 vertices to 6 with all 2 faces kept, and the counts
    compare directly. The transform is that pymeshfix's cut arrives through ``load_array`` rather
    than through a call of its own.

    The three-faces-on-one-edge input is deliberately **not** compared, and the numbers say why:
    pymeshfix cuts 5 vertices to **7** and ordito to **9**, both edge-manifold afterwards. Neither
    is wrong -- an edge with three faces can be separated into three boundary edges (ordito, which
    keeps two corners together only across a manifold consistently-oriented edge) or into a
    manifold pair plus one loose sheet (pymeshfix) -- and there is no canonical answer to compare
    against, so a count assert there would be pinning an arbitrary choice.
    """
    vertices_np, faces_np = _bowtie_np()
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    assert not od.validation.is_vertex_manifold(faces_wp)  # non-vacuity: there is a cut to make

    split_wp, split_faces_wp, _source_wp = od.repair.split_non_manifold_vertices(
        vertices_wp, faces_wp
    )
    vertices_pmf, faces_pmf = pymeshfix_to_numpy(numpy_to_pymeshfix(vertices_np, faces_np))

    assert vertices_pmf.shape[0] == vertices_np.shape[0] + 1  # the reference really cut
    assert faces_pmf.shape[0] == faces_np.shape[0]
    assert split_wp.size == vertices_pmf.shape[0]
    assert split_faces_wp.size // 3 == faces_pmf.shape[0]
    assert od.validation.is_vertex_manifold(split_faces_wp)
    assert od.validation.is_vertex_manifold(numpy_to_warp(vertices_pmf, faces_pmf, device)[1])


@pytest.mark.parity(
    "remove_degenerate_faces",
    "pymeshfix",
    benchmarked=False,
    reason="strong_degeneracy_removal is 7-13 % of a pymeshfix round -- 5.4 ms against 77.4 ms of "
    "load on bunny_decimated, 66.8 against 432.2 on bunny -- and a PyTMesh takes exactly one "
    "load_array, so the build cannot leave the timed callable and the row would be the load. "
    "meshlib carries the timed row for this group.",
)
@pytest.mark.parametrize("n_degenerate", [1, 3])
def test_remove_degenerate_faces_matches_pymeshfix(device: str, n_degenerate: int) -> None:
    """
    Class B: on **exactly** degenerate input the two recover the identical clean mesh.

    ``strong_degeneracy_removal`` deletes zero-area faces, refills what that opens and iterates, so
    the comparison is on the recovered mesh rather than on which faces went: appending ``k``
    zero-area triangles to an ``icosphere(2)`` (each a duplicated vertex, so the area is exactly
    zero in ``float64`` and in ``float32``), both sides come back at **162 v / 320 f, watertight,
    Euler characteristic 2, volume 4.0470** -- the clean sphere, for ``k`` = 1 and 3. The transform
    is that pymeshfix's loader cuts before the removal runs (163 v in, 165 v loaded), so only the
    recovered counts are comparable, not the intermediate.

    The near-degenerate class is where they part, and it is a *precision* difference rather than a
    tolerance one: TMesh's coordinates are ``double``, so a flat 12-column strip offset by ``1e-9``
    is **not** degenerate to it and comes back unchanged at 24 v / 22 f, where ordito's ``float32``
    altitude test removes the whole strip. On the *exactly* collinear version of the same strip both
    reduce it to 0 v / 0 f. So this comparison runs on exact degeneracy on purpose -- a
    near-degenerate fixture would be pinning float32 against float64 and calling it a disagreement.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=2)
    n_vertices = mesh_tm.vertices.shape[0]
    vertices_np = np.vstack([mesh_tm.vertices, mesh_tm.vertices[:n_degenerate]])
    zero_area_np = np.array(
        [[i, (i + 1) % n_vertices, n_vertices + i] for i in range(n_degenerate)]
    )
    faces_np = np.vstack([mesh_tm.faces, zero_area_np])

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    kept_wp, kept_faces_wp = od.repair.remove_degenerate_faces(vertices_wp, faces_wp)
    kept_tm = warp_to_trimesh(kept_wp, kept_faces_wp)

    tin_pmf = numpy_to_pymeshfix(vertices_np, faces_np)
    assert tin_pmf.n_faces == faces_np.shape[0]  # the loader kept the faces, cutting vertices only
    assert tin_pmf.strong_degeneracy_removal(3)
    vertices_pmf, faces_pmf = pymeshfix_to_numpy(tin_pmf)
    kept_pmf = tm.Trimesh(vertices_pmf, faces_pmf, process=False)

    # Non-vacuity: the reference removed the degeneracies rather than passing the mesh through.
    assert faces_pmf.shape[0] == mesh_tm.faces.shape[0] < faces_np.shape[0]
    assert vertices_pmf.shape[0] == n_vertices
    assert kept_faces_wp.size // 3 == faces_pmf.shape[0]
    assert kept_wp.size == vertices_pmf.shape[0]
    assert kept_tm.is_watertight
    assert kept_pmf.is_watertight
    assert kept_tm.euler_number == 2
    assert kept_pmf.euler_number == 2
    assert np.isclose(kept_tm.volume, kept_pmf.volume, rtol=1e-5, atol=1e-5)


def test_remove_degenerate_faces_near_degenerate_divergence(device: str) -> None:
    """
    Not a parity assert: the precision boundary, pinned so the class-B pair above stays honest.

    A flat strip 12 columns long, offset by ``1e-9`` in ``y``. TMesh stores coordinates in
    ``double``, where ``1e-9`` is emphatically not zero, so ``strong_degeneracy_removal`` returns
    ``True`` having changed **nothing** (24 v / 22 f in and out); ordito's altitude test runs on
    the ``float32`` vertex buffer and removes the entire strip. Set the offset to exactly zero and
    both reduce it to 0 v / 0 f.

    Neither is wrong -- it is the same predicate at two precisions -- and recording it is what stops
    a future author reading the exact-degeneracy comparison as a general one.
    """
    columns = 12
    x_np = np.repeat(np.arange(columns, dtype=np.float64), 2)
    quads_np = np.array(
        [[2 * i, 2 * i + 2, 2 * i + 1] for i in range(columns - 1)]
        + [[2 * i + 1, 2 * i + 2, 2 * i + 3] for i in range(columns - 1)]
    )
    for offset, pmf_faces, ordito_faces in ((1e-9, 22, 0), (0.0, 0, 0)):
        vertices_np = np.stack(
            [x_np, np.tile([0.0, offset], columns), np.zeros(2 * columns)], axis=1
        )
        tin_pmf = numpy_to_pymeshfix(vertices_np, quads_np)
        tin_pmf.strong_degeneracy_removal(3)
        assert tin_pmf.n_faces == pmf_faces, offset

        vertices_wp, faces_wp = numpy_to_warp(vertices_np, quads_np, device)
        _kept_wp, kept_faces_wp = od.repair.remove_degenerate_faces(vertices_wp, faces_wp)
        assert kept_faces_wp.size // 3 == ordito_faces, offset


def test_remove_degenerate_faces_matches_trimesh(device: str) -> None:
    rng = np.random.default_rng(7)
    # Scale ~1e-2: keeps genuine-face altitudes (~1e-2) far above the absolute merge tolerance
    # (1e-8) while the FMA rounding error of a zero-area face's altitude (~1e-7 * scale) stays far
    # below it, so the geometric degeneracy test agrees on CPU and GPU. This scaling is deliberate:
    # a zero-area face's cross product is exactly 0 on CPU but ~1e-8 on CUDA because FMA fusion
    # (fuse_fp) is left ON for triangle_cross to keep the geometry kernels fast, so degenerate-face
    # tests must stay clear of the tolerance boundary rather than disabling fusion module-wide.
    vertices_np = (0.01 * rng.normal(size=(9, 3))).astype(np.float32)
    # Mix of good faces and injected degeneracies (repeated index and coincident vertices).
    faces_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 1], [2, 2, 2]], dtype=np.int32)
    vertices_np[8] = vertices_np[6]  # face [6, 7, 8] has two coincident vertices -> zero area

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    kept_vertices_wp, kept_faces_wp = od.repair.remove_degenerate_faces(vertices_wp, faces_wp)

    mesh_tm = tm.Trimesh(vertices=vertices_np.astype(np.float64), faces=faces_np, process=False)
    keep_mask_tm = mesh_tm.nondegenerate_faces(height=_MERGE_TOL)

    kept_ref = _sorted_triangle_positions(vertices_np.astype(np.float64), faces_np[keep_mask_tm])
    kept_wp = _sorted_triangle_positions(
        kept_vertices_wp.numpy().astype(np.float64), kept_faces_wp.numpy().reshape(-1, 3)
    )
    assert _triangle_set_close(kept_wp, kept_ref, atol=1e-5)


@pytest.mark.parity("remove_degenerate_faces", "meshlib", "trimesh", "pymeshlab")
def test_remove_degenerate_faces_matches_meshlib(device: str) -> None:
    """
    Class B (compare detection): three references report the faces this function drops.

    MeshLib and trimesh have no remover, so the transform is the same one the pymeshlab fold
    comparison makes -- compare the *mask* rather than the output mesh, then check the removal
    against it. MeshLib's ``criticalAspectRatio`` selects additional near-degenerate slivers above
    the truly degenerate ones; at its ``FLT_MAX`` default only the zero-area faces are reported,
    which is ordito's criterion, so the default is the setting compared here and is asserted to be
    insensitive over three orders of magnitude on this input. trimesh's ``nondegenerate_faces``
    returns the mask directly at a ``height`` of 1e-08, the same order as ordito's own
    ``TOLERANCE_ZERO``; pymeshlab's ``meshing_remove_null_faces`` rebuilds, so its answer is a face
    *count* and the comparison is on that.

    **open3d is deliberately absent, and the divergence is pinned here rather than assumed.**
    ``remove_degenerate_triangles`` removes a triangle that references a vertex **twice**, not one
    of zero area: on this very input it removes **nothing** where the other three all find the
    collinear face. On a repeated-index face it agrees with them, which is asserted too -- so this
    is a narrower predicate rather than a broken one, and a probe built on ``[0, 0, 1]`` would have
    read it as agreement.

    Non-vacuous by construction: one collinear triangle among two good ones, so every side returns a
    mixed answer and neither an empty nor a full mask would pass.
    """
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=np.float64,
    )
    faces_np = np.array([[0, 1, 2], [1, 3, 2], [0, 4, 1]], dtype=np.int32)  # face 2 is collinear

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    keep_wp = od.triangles.face_nondegenerate_mask(vertices_wp, faces_wp)
    kept_vertices_wp, kept_faces_wp = od.repair.remove_degenerate_faces(vertices_wp, faces_wp)

    mesh_ml = numpy_to_meshlib(vertices_np, faces_np)
    degenerate_ml = meshlib_bitset_to_numpy(
        mm.findDegenerateFaces(mm.MeshPart(mesh_ml)), faces_np.shape[0]
    )

    mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
    keep_tm = np.asarray(mesh_tm.nondegenerate_faces(height=1e-8))

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.meshing_remove_null_faces()
    n_kept_pml = int(meshset_pml.current_mesh().face_number())

    assert degenerate_ml.sum() == 1  # non-vacuity: the reference found exactly the collinear face
    assert int((~keep_tm).sum()) == 1
    assert n_kept_pml == 2
    assert np.array_equal(~keep_wp.numpy(), degenerate_ml)
    assert np.array_equal(keep_wp.numpy(), keep_tm)
    assert int(keep_wp.numpy().sum()) == n_kept_pml

    # open3d's criterion is repeated *indices*, not zero area -- it sees nothing here...
    mesh_o3d = trimesh_to_open3d(mesh_tm)
    mesh_o3d.remove_degenerate_triangles()
    assert len(mesh_o3d.triangles) == faces_np.shape[0]
    # ...and agrees with the other three on a face that names a vertex twice.
    repeated_np = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 1]], dtype=np.int32)
    repeated_tm = tm.Trimesh(vertices_np, repeated_np, process=False)
    repeated_o3d = trimesh_to_open3d(repeated_tm)
    repeated_o3d.remove_degenerate_triangles()
    assert len(repeated_o3d.triangles) == 2
    assert int((~np.asarray(repeated_tm.nondegenerate_faces(height=1e-8))).sum()) == 1
    assert kept_faces_wp.size // 3 == int((~degenerate_ml).sum())
    assert kept_vertices_wp.size == 4  # the collinear apex is now unreferenced

    # The knob that is *not* in play: at any aspect ratio these three faces classify the same way.
    for critical_aspect_ratio in (20.0, 1e3, 1e5):
        assert np.array_equal(
            meshlib_bitset_to_numpy(
                mm.findDegenerateFaces(
                    mm.MeshPart(mesh_ml), criticalAspectRatio=critical_aspect_ratio
                ),
                faces_np.shape[0],
            ),
            degenerate_ml,
        )


def test_collapse_small_triangles_removes_sliver(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    mesh_tm, mesh_wp = sphere_irregular
    vertices_np = mesh_tm.vertices.astype(np.float32)
    a, b = vertices_np[0], vertices_np[1]
    near_edge = a + 0.5 * (b - a) + 1e-6 * (b - a)  # almost on segment a->b
    vertices_np = np.vstack([vertices_np, near_edge]).astype(np.float32)
    faces_np = np.vstack([mesh_tm.faces, [0, 1, vertices_np.shape[0] - 1]]).astype(np.int32)

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, mesh_wp.device)
    epsilon = 1e-6
    out_vertices_wp, out_faces_wp = od.repair.collapse_small_triangles(
        vertices_wp, faces_wp, epsilon=epsilon
    )

    # Sliver is gone.
    assert out_faces_wp.size // 3 < faces_np.shape[0]
    # Invariant (libigl fixpoint guarantee): no surviving triangle is below threshold.
    bbd = od.bounds.enclosing_diagonal(out_vertices_wp)
    _, areas_wp = od.triangles.face_normals_and_areas(out_vertices_wp, out_faces_wp)
    assert (2.0 * areas_wp.numpy()).min() >= epsilon * bbd * bbd * (1.0 - 1e-3)
    # Surviving triangle geometry matches the CPU reference.
    ref = _collapse_small_triangles_ref(vertices_np.astype(np.float64), faces_np, epsilon)
    collapsed_np = _sorted_triangle_positions(
        out_vertices_wp.numpy().astype(np.float64), out_faces_wp.numpy().reshape(-1, 3)
    )
    assert _triangle_set_close(collapsed_np, ref, atol=1e-4)


def test_collapse_small_triangles_fan_chain(device: str) -> None:
    # A row of stacked slivers sharing a spine: several become small only after neighbours
    # collapse, exercising the fixpoint loop.
    xs = np.linspace(0.0, 1.0, 6)
    top = np.stack([xs, np.full_like(xs, 1e-5), np.zeros_like(xs)], axis=1)
    bottom = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
    vertices_np = np.vstack([top, bottom]).astype(np.float32)
    n = xs.size
    faces = []
    for i in range(n - 1):
        faces.append([i, i + 1, n + i])  # thin triangles
        faces.append([i + 1, n + i + 1, n + i])
    # Add one well-shaped triangle far away so the mesh is not entirely collapsed.
    big_triangle = [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.5, 1.0, 0.0]]
    vertices_np = np.vstack([vertices_np, big_triangle]).astype(np.float32)
    big = vertices_np.shape[0]
    faces.append([big - 3, big - 2, big - 1])
    faces_np = np.asarray(faces, dtype=np.int32)

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    epsilon = 1e-3
    out_vertices_wp, out_faces_wp = od.repair.collapse_small_triangles(
        vertices_wp, faces_wp, epsilon=epsilon
    )

    bbd = od.bounds.enclosing_diagonal(out_vertices_wp)
    _, areas_wp = od.triangles.face_normals_and_areas(out_vertices_wp, out_faces_wp)
    assert (2.0 * areas_wp.numpy()).min() >= epsilon * bbd * bbd * (1.0 - 1e-3)
    ref = _collapse_small_triangles_ref(vertices_np.astype(np.float64), faces_np, epsilon)
    collapsed_np = _sorted_triangle_positions(
        out_vertices_wp.numpy().astype(np.float64), out_faces_wp.numpy().reshape(-1, 3)
    )
    assert _triangle_set_close(collapsed_np, ref, atol=1e-3)


# ``icosphere_coarse``, not ``sphere_irregular``: the comparison rests on "small triangle" and
# "short edge" picking the same faces, which holds only without needles. ordito collapses by area
# (igl's rule), MeshLib by edge length; on ``sphere_irregular`` with the same 20 planted edges the
# natural needles reach 1.1e-6 of ``2 bbd**2`` while a face on a shrunk edge keeps its full height,
# so the two sets overlap at every ``epsilon`` (at 1e-5 ordito collapses 39 edges where MeshLib,
# below the mesh's shortest natural edge, collapses the 20; at 1e-7 ordito collapses 3). The area
# rule itself is checked on ``sphere_irregular`` against the transcription of igl's algorithm,
# ``test_collapse_small_triangles_removes_sliver``.
@pytest.mark.parity("collapse_small_triangles", "meshlib")
def test_collapse_small_triangles_matches_meshlib(
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Class C: MeshLib is the only bound reference for this function, and it collapses differently.

    libigl does not bind ``collapse_small_triangles`` despite the C++ header existing, so
    ``resolveMeshDegenerations`` -- an edge-collapse pass parameterized by ``tinyEdgeLength``
    rather than by a relative area -- is the only implementation to compare against. There is no
    correspondence between the two outputs: neither the vertex numbering nor which endpoint of a
    collapsed edge survives is shared (ordito keeps the component representative, MeshLib solves
    for a position under ``maxDeviation``), so what is asserted is a statistic.

    Three of them, on a 320-face sphere with 20 edges shrunk to a thousandth of their length:

    - both reach exactly **286 faces and 145 vertices** from 320 and 162;
    - neither output contains a single edge shorter than the critical length, where the input has
      **17** -- MeshLib's own ``findShortEdges`` is the judge on both sides;
    - the two surfaces stay within **0.0027** of each other, 7.8 % of the critical length.

    Mutation probe and margin: running ordito's pass an order of magnitude weaker
    (``epsilon=1e-6``) leaves **316** faces and **15** short edges, so every threshold here is at
    least 15 counts clear of the value a mis-scaled implementation produces, and the face-count
    assert is exact rather than a bound.
    """
    mesh_tm, _mesh_wp = icosphere_coarse
    rng = np.random.default_rng(0)
    vertices_np = mesh_tm.vertices.copy()
    edges_np = mesh_tm.edges_unique
    for edge in edges_np[rng.choice(edges_np.shape[0], size=20, replace=False)]:
        vertices_np[edge[1]] = vertices_np[edge[0]] + 1e-3 * (
            vertices_np[edge[1]] - vertices_np[edge[0]]
        )
    diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
    critical_length = 1e-2 * diagonal

    def short_edges_ml(mesh: tm.Trimesh) -> int:
        part = mm.MeshPart(trimesh_to_meshlib(mesh))
        return mm.findShortEdges(part, critical_length).count()

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, mesh_tm.faces, device)
    collapsed_tm = warp_to_trimesh(*od.repair.collapse_small_triangles(vertices_wp, faces_wp, 1e-5))

    mesh_ml = numpy_to_meshlib(vertices_np, mesh_tm.faces)
    settings_ml = mm.ResolveMeshDegenSettings()
    settings_ml.tinyEdgeLength = critical_length
    settings_ml.maxDeviation = 0.1 * critical_length
    assert mm.resolveMeshDegenerations(mesh_ml, settings_ml) is True
    resolved_tm = meshlib_to_trimesh(mesh_ml)

    assert short_edges_ml(tm.Trimesh(vertices_np, mesh_tm.faces, process=False)) == 17
    assert collapsed_tm.faces.shape[0] == resolved_tm.faces.shape[0] == 286
    assert collapsed_tm.vertices.shape[0] == resolved_tm.vertices.shape[0] == 145
    assert short_edges_ml(collapsed_tm) == 0
    assert short_edges_ml(resolved_tm) == 0
    assert (
        hausdorff_surface_two_sided(
            collapsed_tm.vertices, collapsed_tm.faces, resolved_tm.vertices, resolved_tm.faces
        )
        < 0.1 * critical_length
    )


# ---------------------------------------------------------------------------
# Geometric defects: bad faces, folds and T-vertices (pymeshlab reference)
# ---------------------------------------------------------------------------


def _worst_aspect(vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]) -> float:
    return float(
        od.triangles.face_quality(vertices_wp, faces_wp, metric="aspect_ratio").numpy().max()
    )


def test_remove_folded_faces_drops_the_fold(
    device: str, folded_patch: tuple[np.ndarray, np.ndarray]
) -> None:
    vertices_np, faces_np = folded_patch
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    kept_vertices_wp, kept_faces_wp = od.repair.remove_folded_faces(vertices_wp, faces_wp)
    # Only the fold goes, so the flat quad it folded over survives intact.
    assert kept_faces_wp.size // 3 == 2
    # The folded apex was referenced only by the dropped face, so it is gone too.
    assert kept_vertices_wp.size == vertices_np.shape[0] - 1
    assert (
        not od.validation.face_defective_mask(
            kept_vertices_wp, kept_faces_wp, min_quality=None, max_fold_angle=160.0
        )
        .numpy()
        .any()
    )


def test_remove_folded_faces_drops_exactly_the_folded_pairs(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: exactly the folded faces go, nothing else.

    A face is folded when a neighbour meets it past 160 degrees -- trimesh's
    ``face_adjacency_angles``, the angle between the two normals -- *and* its normal points against
    the sum of its neighbours' normals, so of a fold's two faces only the one turned back over its
    ring goes. The expected set is computed that way from trimesh. ``sphere_irregular`` carries one
    such pair (163.9 degrees at its sharpest spike) among 1 494 edges whose angles reach every value
    below it; the kept faces are compared by their corner positions, since the vertices are
    compacted.
    """
    mesh_tm, mesh_wp = sphere_irregular
    adjacency = mesh_tm.face_adjacency
    folded_pairs = adjacency[np.degrees(mesh_tm.face_adjacency_angles) > 160.0]
    assert folded_pairs.shape[0] > 0  # non-vacuity: the fixture does fold
    normals = np.asarray(mesh_tm.face_normals)
    consensus = np.zeros_like(normals)
    np.add.at(consensus, adjacency[:, 0], normals[adjacency[:, 1]])
    np.add.at(consensus, adjacency[:, 1], normals[adjacency[:, 0]])
    against = np.einsum("ij,ij->i", normals, consensus) < 0.0
    dropped = np.zeros(mesh_tm.faces.shape[0], dtype=bool)
    dropped[folded_pairs.ravel()] = True
    dropped &= against
    assert 0 < dropped.sum() < folded_pairs.size  # one face of the fold, not both
    expected_np = np.asarray(mesh_tm.vertices)[mesh_tm.faces[~dropped]]

    kept_vertices_wp, kept_faces_wp = od.repair.remove_folded_faces(mesh_wp.points, mesh_wp.indices)
    kept_np = kept_vertices_wp.numpy()[kept_faces_wp.numpy().reshape(-1, 3)].astype(np.float64)
    assert_unordered_rows_equal(kept_np.reshape(-1, 9), expected_np.reshape(-1, 9))


def _self_intersecting_count_ml(
    vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]
) -> int:
    """Count self-intersecting faces with MeshLib, on exactly the buffer it is handed."""
    mesh_ml = warp_to_meshlib(vertices_wp, faces_wp)
    colliding_ml = mm.findSelfCollidingTrianglesBS(mm.MeshPart(mesh_ml), touchIsIntersection=False)
    return int(meshlib_bitset_to_numpy(colliding_ml, faces_wp.size // 3).sum())


def _self_intersecting_count_pmf(
    vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]
) -> int:
    """
    Count self-intersecting faces with pymeshfix.

    No remap and no ``n_faces`` guard, deliberately: only the **count** is read, and
    ``load_array``'s connectivity repair neither creates nor removes a crossing -- it duplicates
    vertices along non-manifold edges and rewinds faces, both of which leave the point set of every
    triangle alone. Measured on every mesh this helper is called with, the load is in fact a no-op
    on the face count (512 in and out on the fixture, 26 688 on the voxel rebuild), so a remap would
    succeed; it is skipped because the face *identities* are not what is being compared.
    """
    tin_pmf = warp_to_pymeshfix(vertices_wp, faces_wp)
    return pymeshfix_intersecting_faces(tin_pmf, tris_per_cell=50, justproper=False).size


@pytest.mark.parity("fix_self_intersections", "meshlib")
@pytest.mark.parametrize("max_expand", [1, 2])
def test_fix_self_intersections_local_clears_them(
    torus_self_intersecting: tuple[tm.Trimesh, wp.Mesh], max_expand: int
) -> None:
    """
    Class A on the *post-condition*, through two detectors that are not ordito's.

    No reference does this **repair** the way this does -- MeshLib's ``localFixSelfIntersections``
    subdivides and relaxes rather than cutting and refilling, and on this very input it makes the
    problem *worse*: by this file's own detector it doubles the intersecting faces while
    quintupling the face count, where this leaves **0**. So the *outputs* are not comparable, and
    the claim about them is the contract: the intersecting faces are gone, the result is
    watertight, and the surface did not run away from the input. MeshLib's local fixer is not a
    weaker version of this repair, it is a different operation that does not converge on this
    input -- it *does* clear a self-intersecting torus built the other way (the ``tangle`` axis in
    ``benchmarks/test_repair.py``), so its behaviour is fixture-dependent and neither reading
    generalizes.

    What *is* a library comparison is the contract's own predicate. Asserting it with
    [`face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask] alone would
    check ordito's repair against ordito's detector, so a shared bug in the detector passes both
    sides -- and that detector is exactly the one MeshLib and pymeshfix are already the oracles for
    one group over. Both are therefore counted here, before and after: all three agree on the
    input's intersecting-face count and on zero after the repair, at both dilation budgets.
    Agreement on the *input* is what makes the assertion non-vacuous -- the fixture's count is
    confirmed by two independent implementations rather than asserted against itself.

    Only meshlib is *claimed* here, because it is the pair this group times and a test may name a
    group once; pymeshfix's claim sits on ``test_fix_self_intersections_voxel_rebuilds``, which
    consults it on the other method. Its assert runs on both regardless.

    The two detectors are asked with the conventions they have, which makes them unequally strict
    and is worth knowing if one of them ever fails alone. MeshLib is passed
    ``touchIsIntersection=False``, ordito's convention; pymeshfix has no such switch and counts a
    *touching* pair, so its zero is the **stronger** statement -- and where the two conventions
    diverge they differ by a lot rather than a little (elsewhere in this file, dozens of faces
    against zero). The cut-and-refill leaves no touching pair, so they coincide here.

    The Hausdorff bound is the one that stops a trivial pass: deleting the whole mesh also has no
    self-intersections. It is two-sided against the input and must stay within a fraction of the
    bounding-box diagonal -- the repair cuts a band out and refills it, so it moves the surface
    locally and nowhere else.

    Both dilation budgets are run because they take different amounts of surface with them, and
    both must land clean.
    """
    mesh_tm, mesh_wp = torus_self_intersecting
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    before_np = od.validation.face_self_intersecting_mask(vertices_wp, faces_wp).numpy()
    assert int(before_np.sum()) > 0  # non-vacuity: the fixture really does intersect itself
    # Two independent detectors confirm the input's count, so "0 after" is a repair and not a
    # detector that stopped answering.
    before_ml = _self_intersecting_count_ml(vertices_wp, faces_wp)
    before_pmf = _self_intersecting_count_pmf(vertices_wp, faces_wp)
    assert before_ml == int(before_np.sum())
    assert before_pmf == int(before_np.sum())

    fixed_vertices_wp, fixed_faces_wp = od.repair.fix_self_intersections(
        vertices_wp, faces_wp, max_expand=max_expand
    )
    assert fixed_faces_wp.size > 0
    after_np = od.validation.face_self_intersecting_mask(fixed_vertices_wp, fixed_faces_wp).numpy()
    after_ml = _self_intersecting_count_ml(fixed_vertices_wp, fixed_faces_wp)
    after_pmf = _self_intersecting_count_pmf(fixed_vertices_wp, fixed_faces_wp)
    assert int(after_np.sum()) == 0
    assert after_ml == 0
    assert after_pmf == 0
    assert od.validation.is_watertight(fixed_vertices_wp, fixed_faces_wp)

    diagonal = float(np.linalg.norm(mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)))
    deviation = hausdorff_surface_two_sided(
        np.asarray(mesh_tm.vertices, dtype=np.float64),
        np.asarray(mesh_tm.faces),
        fixed_vertices_wp.numpy().astype(np.float64),
        fixed_faces_wp.numpy().reshape(-1, 3),
    )
    assert deviation < 0.35 * diagonal  # it patched a band, it did not rebuild the object


@pytest.mark.parity(
    "fix_self_intersections",
    "pymeshfix",
    benchmarked=False,
    reason="select_intersecting_triangles verifies the post-condition; a timed row would have "
    "to be "
    "strong_intersection_removal, which is 69.5 % of a round on bunny_decimated and is a different "
    "algorithm -- it ends with one component where this ends with two (chi 2 against 4, volume "
    "-6.53 against -10.42 on this fixture's shape), so the outputs are not comparable. The "
    "detector "
    "is timed in the face_self_intersecting_mask group.",
)
def test_fix_self_intersections_voxel_rebuilds(
    torus_self_intersecting: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class A on the residual count, through two detectors that are not ordito's.

    A level set cannot self-intersect, so this always terminates -- but the *triangulation* of one
    can still carry a touching pair at an ambiguous marching-cubes cell, and that is
    resolution-dependent: measured on this fixture, **0** intersecting faces at a 1 % lattice and
    **2 of 26 688** at 1/128. The assertion is therefore "almost none, and far fewer than the input
    had" rather than zero, which is what the function promises.

    Because the answer is only *promised* to be small rather than zero, both references are asserted
    to **agree with ordito's count** rather than to read zero -- the stronger claim, and the
    resolution-independent one. Measured on this fixture at the default lattice, on both devices: 26
    688 faces out, all three detectors reading **0**, and pymeshfix's ``load_array`` leaving the
    face count untouched at 26 688, so its answer is about this mesh and not about a repaired copy
    of it.

    Also asserts the rebuild is a rebuild: the face count grows by more than an order of magnitude,
    because every part of the surface is resampled and not just the damaged band.
    """
    mesh_tm, mesh_wp = torus_self_intersecting
    n_faces = mesh_tm.faces.shape[0]
    before_np = od.validation.face_self_intersecting_mask(mesh_wp.points, mesh_wp.indices).numpy()
    assert int(before_np.sum()) > 0  # non-vacuity: "fewer than before" has to mean something

    rebuilt_vertices_wp, rebuilt_faces_wp = od.repair.fix_self_intersections(
        mesh_wp.points, mesh_wp.indices, method="voxel"
    )
    assert rebuilt_faces_wp.size // 3 > 10 * n_faces  # everything was resampled

    after_np = od.validation.face_self_intersecting_mask(
        rebuilt_vertices_wp, rebuilt_faces_wp
    ).numpy()
    after_ml = _self_intersecting_count_ml(rebuilt_vertices_wp, rebuilt_faces_wp)
    after_pmf = _self_intersecting_count_pmf(rebuilt_vertices_wp, rebuilt_faces_wp)
    assert int(after_np.sum()) < 0.001 * rebuilt_faces_wp.size // 3
    assert int(after_np.sum()) < int(before_np.sum())
    assert after_ml == int(after_np.sum())
    assert after_pmf == int(after_np.sum())


def test_fix_self_intersections_leaves_a_clean_mesh_alone(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: a clean input is returned unchanged, and the three value guards.

    The identity case matters more than it looks: the local method's loop reads its own detector to
    decide whether to run at all, so a mesh with nothing wrong must come back with the same faces
    rather than through one round of cut-and-refill.
    """
    _, mesh_wp = sphere_irregular
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    assert not od.validation.is_self_intersecting(mesh_wp)

    same_vertices_wp, same_faces_wp = od.repair.fix_self_intersections(vertices_wp, faces_wp)
    assert np.array_equal(same_faces_wp.numpy(), faces_wp.numpy())
    assert np.allclose(same_vertices_wp.numpy(), vertices_wp.numpy())

    with pytest.raises(ValueError, match="method must be"):
        od.repair.fix_self_intersections(
            vertices_wp,
            faces_wp,
            method="nonsense",  # pyright: ignore[reportArgumentType]  # the off-menu value under test
        )
    with pytest.raises(ValueError, match="max_expand"):
        od.repair.fix_self_intersections(vertices_wp, faces_wp, max_expand=-1)
    with pytest.raises(ValueError, match="max_iter"):
        od.repair.fix_self_intersections(vertices_wp, faces_wp, max_iter=0)


@pytest.mark.parametrize("hops", [0, 1, 2, 3])
def test_fix_self_intersections_dilation_matches_expand_vertex_mask(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh], hops: int
) -> None:
    """
    Ordito against ordito: the loop's face-ring dilation is ``expand_vertex_mask``'s.

    ``fix_self_intersections`` grows its cut region by hopping through faces (any-corner lookup,
    then mark the corners) instead of building a unique-edge table every pass. That is the same
    dilation only because two vertices of a triangle mesh are one-ring neighbours exactly when they
    share a face, and each hop must grow from the previous hop's complete mask. The reference path
    is ``expand_vertex_mask`` followed by the same any-corner face rule, which is what the loop
    called before; it carries its own oracle in ``tests/test_selection.py``. Non-vacuity: the
    seed is a few faces, and every hop grows the answer.
    """
    _, mesh_wp = sphere_irregular
    faces_wp = mesh_wp.indices
    n_vertices = mesh_wp.points.size
    n_faces = faces_wp.size // 3
    seed_np = np.zeros(n_faces, dtype=bool)
    seed_np[[0, 101, 222]] = True
    seed_wp = wp.array(seed_np, dtype=wp.bool, device=faces_wp.device)

    grown_wp = od.repair._dilate_face_mask(  # pyright: ignore[reportPrivateUsage]
        faces_wp, seed_wp, hops, n_vertices
    )

    faces_np = faces_wp.numpy().reshape(-1, 3)
    vertex_seed_np = np.zeros(n_vertices, dtype=bool)
    vertex_seed_np[faces_np[seed_np].ravel()] = True
    vertex_seed_wp = wp.array(vertex_seed_np, dtype=wp.bool, device=faces_wp.device)
    expanded_np = od.selection.expand_vertex_mask(faces_wp, vertex_seed_wp, hops).numpy()
    reference_np = expanded_np[faces_np].any(axis=1)
    assert np.array_equal(grown_wp.numpy(), reference_np)
    assert int(reference_np.sum()) > (int(seed_np.sum()) if hops == 0 else 3 * (hops + 1))


def _ragged_grid(device: str) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Build a flat grid with every third rim face removed, plus the intact grid for comparison.

    Planar on purpose. The notches of a *curved* rim are slivers nearly perpendicular to the faces
    they join -- on a sliced ``icosphere(3)`` rim, 21 candidates exist and every one fails both the
    default ``max_aspect_ratio`` and any ``min_normal_dot`` above 0.5 -- so a curved fixture tests
    the gates rejecting rather than the straightening working. On a plane the notch triangle is
    exactly coplanar with its neighbours and well shaped, and the correct answer is the grid itself.
    """
    vertices_wp, faces_wp = od.creation.grid(count=(8, 8), device=device)
    n_faces = faces_wp.size // 3
    rim = set(od.boundary.boundary_loops(vertices_wp, faces_wp)[0].list())
    faces_np = faces_wp.numpy().reshape(-1, 3)
    keep_np = np.ones(n_faces, dtype=bool)
    rim_faces = [
        index
        for index in range(n_faces)
        if sum(1 for corner in faces_np[index] if int(corner) in rim) >= 2
    ]
    for index in rim_faces[::3]:
        keep_np[index] = False
    ragged_vertices_wp, ragged_faces_wp = od.selection.submesh_from_face_mask(
        vertices_wp, faces_wp, wp.array(keep_np, dtype=wp.bool, device=device)
    )
    return ragged_vertices_wp, ragged_faces_wp, faces_wp


@pytest.mark.parity("straighten_boundary", "meshlib")
def test_straighten_boundary_matches_meshlib(device: str) -> None:
    """
    Class A on the triangles added, at matched thresholds -- the plan expected Class C.

    ``straightenBoundary(mesh, bd, minNeiNormalsDot, maxTriAspectRatio)`` takes the same two gates
    by the same definitions, and on a ragged planar rim the two agree exactly: both add **10**
    triangles at ``(0.9, 10.0)``, taking 88 faces to 98. The face *sets* are compared too, as
    unordered rows, so agreeing on the count alone could not hide a different choice of notch.

    The reason the agreement can be exact here and not on a curved rim is the fixture: see
    ``_ragged_grid``. And the answer is independently known -- 98 faces is the intact grid -- which
    is what makes this stronger than two implementations agreeing with each other.
    """
    ragged_vertices_wp, ragged_faces_wp, grid_faces_wp = _ragged_grid(device)
    n_ragged = ragged_faces_wp.size // 3
    assert n_ragged < grid_faces_wp.size // 3  # non-vacuity: faces really were removed

    straightened_wp, added = od.repair.straighten_boundary(
        ragged_vertices_wp,
        ragged_faces_wp,
        min_normal_dot=0.9,
        max_aspect_ratio=10.0,
        iterations=6,
        return_count=True,
    )
    assert added > 0

    mesh_ml = numpy_to_meshlib(
        ragged_vertices_wp.numpy().astype(np.float64), ragged_faces_wp.numpy().reshape(-1, 3)
    )
    holes_ml = mesh_ml.topology.findHoleRepresentiveEdges()
    assert len(holes_ml) == 1  # one rim, so the per-rim call covers the whole boundary
    mm.straightenBoundary(mesh_ml, holes_ml[0], 0.9, 10.0)
    faces_ml = np.asarray(meshlib_to_trimesh(mesh_ml).faces)

    assert added == len(faces_ml) - n_ragged
    assert_unordered_rows_equal(
        canonical_winding(straightened_wp.numpy().reshape(-1, 3)), canonical_winding(faces_ml)
    )


def test_straighten_boundary_restores_the_grid(device: str) -> None:
    """
    Not a library comparison: the answer is known, so the invariant is an equality.

    Straightening a grid whose rim faces were removed must give the **grid back** -- same face
    count, same face set, and a rim perimeter of exactly 4.0 for the unit square, down from
    6.0203. That is stronger than "the perimeter decreased", and it pins the winding of the added
    triangles as well as their choice: a reversed one would leave the mesh non-manifold.

    The gates are also asserted to *bind*. On this fixture the notches are coplanar, so
    ``min_normal_dot`` up to 0.99 accepts all ten; at ``max_aspect_ratio`` below the notch
    triangles' own ratio none is accepted, and the mesh comes back untouched. A test that only ran
    the permissive case would not distinguish the gates from constants.

    The ``return_count`` keyword's two return shapes are checked on the same call: the default is
    the bare face buffer, so this drops into a chain like the other face-returning repairs, and
    both shapes must describe the same answer, which a shape-only check would not see. The face
    sets are compared unordered: ``emit_straighten_faces`` appends through an atomic cursor, so the
    order of the new triangles differs run to run and is not part of the contract.
    """
    ragged_vertices_wp, ragged_faces_wp, grid_faces_wp = _ragged_grid(device)
    loops_before = od.boundary.boundary_loops(ragged_vertices_wp, ragged_faces_wp)
    perimeter_before = float(
        od.boundary.loop_perimeters(ragged_vertices_wp, loops_before).numpy().sum()
    )
    assert perimeter_before > 4.0 + 1e-3  # non-vacuity: the rim really is ragged

    straightened_wp, added = od.repair.straighten_boundary(
        ragged_vertices_wp, ragged_faces_wp, min_normal_dot=0.99, iterations=6, return_count=True
    )
    assert added == grid_faces_wp.size // 3 - ragged_faces_wp.size // 3
    # Without ``return_count`` the same call returns the bare face buffer, with the same face set.
    faces_only_wp = od.repair.straighten_boundary(
        ragged_vertices_wp, ragged_faces_wp, min_normal_dot=0.99, iterations=6
    )
    assert isinstance(faces_only_wp, wp.array)
    assert_unordered_rows_equal(
        canonical_winding(faces_only_wp.numpy().reshape(-1, 3)),
        canonical_winding(straightened_wp.numpy().reshape(-1, 3)),
    )
    assert od.validation.is_edge_manifold(straightened_wp)
    loops_after = od.boundary.boundary_loops(ragged_vertices_wp, straightened_wp)
    assert len(loops_after) == 1
    assert np.isclose(
        float(od.boundary.loop_perimeters(ragged_vertices_wp, loops_after).numpy().sum()),
        4.0,
        rtol=1e-6,
    )
    assert_unordered_rows_equal(
        canonical_winding(straightened_wp.numpy().reshape(-1, 3)),
        canonical_winding(grid_faces_wp.numpy().reshape(-1, 3)),
    )

    # The aspect-ratio gate binds: below the notch triangles' own ratio, nothing is accepted.
    rejected_wp, rejected = od.repair.straighten_boundary(
        ragged_vertices_wp, ragged_faces_wp, max_aspect_ratio=1.0, iterations=6, return_count=True
    )
    assert rejected == 0
    assert np.array_equal(rejected_wp.numpy(), ragged_faces_wp.numpy())

    with pytest.raises(ValueError, match="iterations must be non-negative"):
        od.repair.straighten_boundary(ragged_vertices_wp, ragged_faces_wp, iterations=-1)


def test_straighten_boundary_closes_an_independent_set(device: str) -> None:
    """
    Not a library comparison: the one-face-per-rim-edge rule, on a fixture that reaches it.

    Two adjacent notches share a rim edge, so a pass may close only one of them or that edge ends
    up with three incident faces. **At the documented gates this rule is never exercised**: on
    ``_ragged_grid`` -- the fixture every other test in this group uses -- pass one produces 10
    candidates and *zero* adjacent pairs, so the pass converges in one round and ``iterations=6``
    does nothing. Measured directly off the notch test in
    ``collect_rim_links_and_candidates``. A test at those gates
    therefore says nothing about the rule, and both halves of it can be deleted without failing
    anything.

    Opening both gates is what reaches it: at ``min_normal_dot=-1.0`` every rim halfedge whose two
    neighbours differ qualifies, so *every* consecutive pair is adjacent and the rule has to reject
    half of them. The claim asserted is the invariant it exists for -- no rim edge carries two of
    the new triangles -- read off the emitted faces, since each is ``(following, v, previous)`` and
    so uses exactly the two rim edges ``{v, previous}`` and ``{following, v}``.

    Deleting either half of the rule takes this from 12 triangles sharing no rim edge to 18
    sharing 6. Note the output at these gates is *not* a mesh anyone wants -- the fold-over gate is
    off, so it attaches triangles that duplicate existing edges -- which is why the assertion is on
    the rim-edge count and not on manifoldness.
    """
    ragged_vertices_wp, ragged_faces_wp, _grid_faces_wp = _ragged_grid(device)
    n_faces = ragged_faces_wp.size // 3

    out_wp, added = od.repair.straighten_boundary(
        ragged_vertices_wp,
        ragged_faces_wp,
        min_normal_dot=-1.0,
        max_aspect_ratio=1e9,
        iterations=1,
        return_count=True,
    )
    rim_length = len(od.boundary.boundary_loops(ragged_vertices_wp, ragged_faces_wp)[0].numpy())
    assert 0 < added < rim_length  # non-vacuity: candidates existed and the rule rejected some

    used: Counter[frozenset[int]] = Counter()
    for following, corner, previous in out_wp.numpy().reshape(-1, 3)[n_faces:].tolist():
        used[frozenset((corner, previous))] += 1
        used[frozenset((following, corner))] += 1
    assert used  # non-vacuity: the new faces really do sit on rim edges
    assert max(used.values()) == 1


def test_straighten_boundary_closes_both_loops_at_a_bowtie(device: str) -> None:
    """
    Not a library comparison: the rim contract on a mesh edge-manifoldness does not cover.

    Two copies of ``_ragged_grid``, the second turned 180 degrees in plane and welded to the first
    at a single rim vertex. The result is still edge-manifold -- ``halfedge_twins`` raises nothing,
    which is ``straighten_boundary``'s only validation -- but that vertex now carries **two** rim
    loops, so it has two outgoing and two incoming boundary halfedges and no single
    ``previous -> v -> following``.

    Keyed by vertex, the rim tables have two writers per slot there and the three of them can be
    won by different halfedges, so the fold-over normal gate ends up testing a face that does not
    border the candidate triangle: this fixture's ancestor measured **15** triangles added on cpu
    against **10** on cuda:0, a different answer per device. Keyed by *halfedge* each loop keeps its
    own links, and the claim asserted here is the strong one -- both loops are restored **whole**,
    including the notch at the pinch itself on each side, so the count is exactly twice what one
    grid alone restores and every triangle added is a face of one of the two intact grids.

    A degree guard that merely *skipped* the pinch would pass a device-agreement check and fail
    this: it costs one notch per loop, giving 18.
    """
    ragged_vertices_wp, ragged_faces_wp, grid_faces_wp = _ragged_grid(device)
    vertices_np = ragged_vertices_wp.numpy()
    faces_np = ragged_faces_wp.numpy().reshape(-1, 3)
    n_vertices = len(vertices_np)
    loop_np = od.boundary.boundary_loops(ragged_vertices_wp, ragged_faces_wp)[0].numpy()
    pinch, welded = int(loop_np[0]), int(loop_np[len(loop_np) // 2])

    turned_np = vertices_np.copy()
    turned_np[:, :2] *= -1
    turned_np += vertices_np[pinch] - turned_np[welded]
    kept = [index for index in range(n_vertices) if index != welded]
    remap = {old: n_vertices + new for new, old in enumerate(kept)}
    remap[welded] = pinch
    pinched_vertices_wp, pinched_faces_wp = numpy_to_warp(
        np.vstack([vertices_np, turned_np[kept]]),
        np.vstack([faces_np, np.vectorize(remap.get)(faces_np)]).ravel().astype(np.int32),
        device,
    )
    assert od.validation.is_edge_manifold(pinched_faces_wp)  # non-vacuity: the guard accepts it

    gates: _StraightenGates = {"min_normal_dot": 0.9, "max_aspect_ratio": 10.0, "iterations": 6}
    out_wp, added = od.repair.straighten_boundary(
        pinched_vertices_wp, pinched_faces_wp, return_count=True, **gates
    )
    _clean_wp, clean_added = od.repair.straighten_boundary(
        ragged_vertices_wp, ragged_faces_wp, return_count=True, **gates
    )
    assert clean_added == 10  # non-vacuity: one grid on its own really is ragged
    assert added == 2 * clean_added

    added_rows = canonical_winding(out_wp.numpy().reshape(-1, 3)[2 * len(faces_np) :])
    grid_rows = canonical_winding(grid_faces_wp.numpy().reshape(-1, 3))
    intact = {tuple(row) for row in grid_rows.tolist()}
    intact |= {tuple(row) for row in canonical_winding(np.vectorize(remap.get)(grid_rows)).tolist()}
    assert {tuple(row) for row in added_rows.tolist()} <= intact
    # One triangle per loop touches the pinch, and both are closed -- the point of keying by
    # halfedge rather than by vertex.
    assert sum(1 for row in added_rows.tolist() if pinch in row) == 2


@pytest.mark.parity("flatten_degree3_vertices", "meshlib")
def test_flatten_degree3_vertices_moves_an_independent_set(device: str) -> None:
    """
    Class A against ``hardSmoothTetrahedrons`` where every vertex is a candidate at once.

    A tetrahedron is four interior valence-3 vertices each adjacent to the other three -- the
    densest this case gets, and the counterexample to the "two of them cannot be neighbours" rule
    this function was once written against. Moving all four from the input positions maps it to
    minus a third of itself: mirrored, every normal flipped and the signed volume negated from
    -2.667 to +0.099, a repair returning an inverted mesh.

    Flattening a maximal independent set per pass, lowest index wins, and repeating is not merely
    *a* fix but exactly meshlib's: it sweeps its vertices sequentially, reading neighbours it has
    already moved, and the two rules coincide vertex for vertex, because a vertex is selected in
    the pass after the last of its lower-indexed candidate neighbours -- which is when a sequential
    index-order sweep would reach it. Measured: identical to **1.19e-07** here and to **0.0** on a
    subdivided tetrahedron. Four passes, which is what makes ``max_iter`` load-bearing rather than
    decorative -- at ``max_iter=1`` only one vertex moves.

    Asserted alongside: each moved vertex lands in its three neighbours' plane at the time it
    moves, and the signed volume does not flip. The all-four pass measured +0.099 against
    |before| = 2.667, i.e. 3.7 %, so the volume bar separates the two by ~37x.
    """
    vertices_np = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    faces_np = np.array([0, 2, 1, 0, 3, 2, 0, 1, 3, 1, 2, 3], dtype=np.int32)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    flattened_np = od.repair.flatten_degree3_vertices(vertices_wp, faces_wp).numpy()
    mesh_ml = trimesh_to_meshlib(tm.Trimesh(vertices_np, faces_np.reshape(-1, 3), process=False))
    mm.hardSmoothTetrahedrons(mesh_ml)
    flattened_ml = mn.toNumpyArray(mesh_ml.points)
    moved_ml = np.linalg.norm(flattened_ml - vertices_np, axis=1) > 1e-6
    assert int(moved_ml.sum()) == 4  # non-vacuity: the reference moved every vertex
    assert np.allclose(flattened_np, flattened_ml, rtol=1e-5, atol=1e-5)

    # One pass is one vertex here, which is what the loop exists for.
    once_np = od.repair.flatten_degree3_vertices(vertices_wp, faces_wp, max_iter=1).numpy()
    assert int((np.linalg.norm(once_np - vertices_np, axis=1) > 1e-6).sum()) == 1
    with pytest.raises(ValueError, match="max_iter must be non-negative"):
        od.repair.flatten_degree3_vertices(vertices_wp, faces_wp, max_iter=-1)

    before = tm.Trimesh(vertices_np, faces_np.reshape(-1, 3), process=False).volume
    after = tm.Trimesh(flattened_np, faces_np.reshape(-1, 3), process=False).volume
    assert before < -1.0  # non-vacuity: the input really encloses a signed volume
    assert abs(after) < 1e-3 * abs(before)  # flat is honest here; inverted is not


def test_remove_tunnels_count_is_unconditional(torus_irregular: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: that this count is *not* behind ``return_count``, deliberately.

    Unlike the three diagnostics beside it, ``removed`` is the caller's documented
    loop-termination signal -- one call cuts at most one tunnel per disjoint family, so the usage is
    to loop until it reads zero. A caller who could not see it could not use the function
    correctly, which is why it stays a third return element. Pinned so a future consistency pass
    does not sweep it up with the others.
    """
    _mesh_tm, mesh_wp = torus_irregular
    result = od.repair.remove_tunnels(mesh_wp.points, mesh_wp.indices, 1e-9)
    assert len(result) == 3
    assert isinstance(result[2], int)
    with pytest.raises(TypeError):
        od.repair.remove_tunnels(
            mesh_wp.points,
            mesh_wp.indices,
            1e-9,
            return_count=True,  # pyright: ignore[reportCallIssue]  # the rejected keyword under test
        )


def _mesh_with_a_degree3_vertex(bump: float = 0.0) -> tm.Trimesh:
    """
    Build an icosahedron with one face split at its centroid: one valence-3 vertex.

    ``bump`` pushes that vertex out along the split face's normal, which is what turns the fixture
    from "one valence-3 vertex" into "one valence-3 *pimple*" -- the flattening repair has nothing
    to do at ``0.0``, where the vertex is already at the centroid it would be moved to.
    """
    mesh_tm = tm.creation.icosahedron()
    vertices = [list(map(float, point)) for point in mesh_tm.vertices]
    faces = mesh_tm.faces.tolist()
    split = faces.pop(0)
    corners_np = np.asarray([vertices[index] for index in split])
    centroid_np = corners_np.mean(axis=0)
    if bump != 0.0:
        normal_np = np.cross(corners_np[1] - corners_np[0], corners_np[2] - corners_np[0])
        centroid_np = centroid_np + bump * normal_np / np.linalg.norm(normal_np)
    vertices.append(list(map(float, centroid_np)))
    centre = len(vertices) - 1
    faces += [
        [split[0], split[1], centre],
        [split[1], split[2], centre],
        [split[2], split[0], centre],
    ]
    return tm.Trimesh(np.array(vertices), np.array(faces), process=False)


@pytest.mark.parity("remove_degree3_vertices", "meshlib")
def test_remove_degree3_vertices_mask_matches_meshlib(device: str) -> None:
    """
    Class A on which vertices qualify, against ``findInnerVertsOfDegree(topology, 3)``.

    That predicate is the clean half of MeshLib's answer: a bitset of the interior vertices of a
    given valence, which is exactly the candidate set this removes. The removal itself is compared
    only by its *effect* -- ``eliminateDegree3Vertices`` returns a count and mutates in place -- so
    the count is asserted and the resulting mesh is checked by invariant.

    The fixture is an icosahedron with one face split at its centroid: that centroid is the only
    valence-3 interior vertex, and removing it must return the icosahedron. So the whole answer is
    known in advance, which is what makes this stronger than a comparison on a scan mesh where
    neither side's answer is independently known. Measured: MeshLib marks vertex 12 alone, and the
    result is 12 vertices and 20 faces, watertight and edge-manifold.

    The **control** matters as much: a plain icosahedron has no such vertex, so the mask is empty
    and the function is a no-op returning its input buffers -- which rules out a pass that removes
    something on any input.

    Both inputs also check the ``return_count`` keyword's two return shapes: two elements by
    default, three with the count, describing the same answer (faces as an unordered set, as in
    ``test_straighten_boundary_restores_the_grid``).
    """
    for mesh_tm, expected_removed in (
        (_mesh_with_a_degree3_vertex(), 1),
        (tm.creation.icosahedron(), 0),
    ):
        vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
        mesh_ml = trimesh_to_meshlib(mesh_tm)
        degree3_np = meshlib_bitset_to_numpy(
            mm.findInnerVertsOfDegree(mesh_ml.topology, 3), len(mesh_tm.vertices)
        )
        assert int(degree3_np.sum()) == expected_removed

        out_vertices_wp, out_faces_wp, removed = od.repair.remove_degree3_vertices(
            vertices_wp, faces_wp, return_count=True
        )
        assert removed == expected_removed
        # Without ``return_count`` the same call returns the pair alone, with the same answer.
        pair = od.repair.remove_degree3_vertices(vertices_wp, faces_wp)
        assert len(pair) == 2
        assert np.allclose(pair[0].numpy(), out_vertices_wp.numpy(), rtol=1e-5, atol=1e-5)
        assert_unordered_rows_equal(
            canonical_winding(pair[1].numpy().reshape(-1, 3)),
            canonical_winding(out_faces_wp.numpy().reshape(-1, 3)),
        )
        assert out_vertices_wp.size == len(mesh_tm.vertices) - expected_removed
        assert out_faces_wp.size // 3 == len(mesh_tm.faces) - 2 * expected_removed
        assert od.validation.is_edge_manifold(out_faces_wp)
        assert warp_to_trimesh(out_vertices_wp, out_faces_wp).is_watertight


def test_remove_degree3_vertices_is_idempotent_and_area_preserving(device: str) -> None:
    """
    Not a library comparison: the two properties that say the collapse was the right triangle.

    Collapsing a valence-3 fan keeps the triangle its three neighbours span, so the **area is
    unchanged** -- the three faces tile exactly that triangle. Getting the replacement's winding or
    its vertices wrong changes the area, and nothing else in the output would show it. And a second
    call must remove nothing, since the pass runs to a fixpoint.
    """
    mesh_tm = _mesh_with_a_degree3_vertex()
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    out_vertices_wp, out_faces_wp, removed = od.repair.remove_degree3_vertices(
        vertices_wp, faces_wp, return_count=True
    )
    assert removed == 1  # non-vacuity
    assert np.isclose(warp_to_trimesh(out_vertices_wp, out_faces_wp).area, mesh_tm.area, rtol=1e-6)

    again_vertices_wp, again_faces_wp, again_removed = od.repair.remove_degree3_vertices(
        out_vertices_wp, out_faces_wp, return_count=True
    )
    assert again_removed == 0
    assert np.array_equal(again_faces_wp.numpy(), out_faces_wp.numpy())
    assert np.array_equal(again_vertices_wp.numpy(), out_vertices_wp.numpy())

    with pytest.raises(ValueError, match="max_iter must be non-negative"):
        od.repair.remove_degree3_vertices(vertices_wp, faces_wp, max_iter=-1)


def test_remove_degree3_vertices_reads_only_closed_fans(device: str) -> None:
    """
    Not a library comparison: the candidate test on the inputs where a face count of three lies.

    A candidate is found from its three faces alone -- their opposite edges must close into one
    directed 3-cycle -- so the cases that test has to reject are the ones where a vertex has three
    faces and is *not* an interior valence-3 vertex: a boundary fan of three (four neighbours, an
    open chain of opposite edges), and a pinched vertex joining two closed icosahedra, which is
    accepted as input and must remove nothing. The tetrahedron is the other side: every vertex is a
    candidate and each is adjacent to the rest, so the lowest index wins alone and its fan becomes
    the face opposite it with the winding reversed -- a two-face closed pillow over the other three
    vertices. And a mesh with an edge on three faces, or with an edge both of whose faces
    traverse it the same way, raises from the input validation, with the count of each.
    """
    fan_vertices_np = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], float)
    fan_faces_np = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    vertices_wp, faces_wp = numpy_to_warp(fan_vertices_np, fan_faces_np.ravel(), device)
    assert od.repair.remove_degree3_vertices(vertices_wp, faces_wp, return_count=True)[2] == 0

    icosahedron_tm = tm.creation.icosahedron()
    offset = len(icosahedron_tm.vertices)
    mirrored_np = 2.0 * icosahedron_tm.vertices[0] - icosahedron_tm.vertices
    second_np = icosahedron_tm.faces + offset
    second_np[second_np == offset] = 0  # share vertex 0: a pinch
    vertices_wp, faces_wp = numpy_to_warp(
        np.concatenate([icosahedron_tm.vertices, mirrored_np]),
        np.concatenate([icosahedron_tm.faces, second_np]).ravel(),
        device,
    )
    out_vertices_wp, out_faces_wp, removed = od.repair.remove_degree3_vertices(
        vertices_wp, faces_wp, return_count=True
    )
    assert removed == 0
    assert out_faces_wp.size == 2 * 3 * len(icosahedron_tm.faces)
    assert out_vertices_wp is vertices_wp  # nothing removed: the input buffers come back

    tetrahedron_np = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    tetrahedron_faces_np = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    vertices_wp, faces_wp = numpy_to_warp(tetrahedron_np, tetrahedron_faces_np.ravel(), device)
    out_vertices_wp, out_faces_wp, removed = od.repair.remove_degree3_vertices(
        vertices_wp, faces_wp, return_count=True
    )
    assert removed == 1
    assert np.allclose(out_vertices_wp.numpy(), tetrahedron_np[1:], rtol=1e-5, atol=1e-5)
    assert np.array_equal(out_faces_wp.numpy(), [0, 1, 2, 1, 0, 2])

    fin_faces_np = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    vertices_wp, faces_wp = numpy_to_warp(fan_vertices_np, fin_faces_np.ravel(), device)
    with pytest.raises(ValueError, match="edge-manifold mesh: 1 edge"):
        od.repair.remove_degree3_vertices(vertices_wp, faces_wp)

    # The other rejection: the tetrahedron with one face reversed is edge-manifold, but three of
    # its edges are traversed the same way by both faces. Checked before a pass removes anything,
    # although vertex 0 would otherwise be a candidate.
    flipped_np = tetrahedron_faces_np.copy()
    flipped_np[3] = flipped_np[3, ::-1]
    vertices_wp, faces_wp = numpy_to_warp(tetrahedron_np, flipped_np.ravel(), device)
    with pytest.raises(ValueError, match="consistently wound mesh: 3 edge"):
        od.repair.remove_degree3_vertices(vertices_wp, faces_wp)
    # ``max_iter=0`` runs no pass, so it validates nothing and returns the input.
    assert od.repair.remove_degree3_vertices(vertices_wp, faces_wp, max_iter=0)[1] is faces_wp


def _nested_face_splits(
    mesh_tm: tm.Trimesh, face_indices: list[int], depth: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split each listed face at its centroid, then split one child of that split, ``depth`` deep.

    Only the innermost centroid of each chain is valence 3; removing it drops the one before it to
    valence 3, so the chain is a cascade that needs ``depth`` removal passes and that no pass-0
    selection can see. The splits are laid over the input's own faces, so removing every centroid
    returns ``mesh_tm`` exactly.
    """
    vertices = [np.asarray(point, dtype=np.float64) for point in mesh_tm.vertices]
    faces = mesh_tm.faces.tolist()
    for face in face_indices:
        target = face
        for level in range(depth):
            a, b, c = faces[target]
            vertices.append((vertices[a] + vertices[b] + vertices[c]) / 3.0)
            centre = len(vertices) - 1
            faces[target] = [a, b, centre]
            faces.append([b, c, centre])
            faces.append([c, a, centre])
            target = len(faces) - 1 - level % 2
    return np.asarray(vertices), np.asarray(faces)


@pytest.mark.parametrize("closed", [True, False], ids=["closed", "open"])
def test_remove_degree3_vertices_runs_a_cascade_to_its_fixpoint(device: str, closed: bool) -> None:
    """
    Class A against ``findInnerVertsOfDegree``: a cascade leaves no interior valence-3 vertex.

    The pass loop stops as soon as a pass creates no new candidate, which the emit kernel counts
    from the rim vertices' face counts rather than by building the next pass's rings. A miscount
    toward zero stops a cascade early and leaves a valence-3 vertex MeshLib still finds, so this
    compares MeshLib's interior valence-3 mask of the *output* against the empty answer, and the
    removed count against the number of splits -- the assert that fails first when the kernel's
    count is mutated to undercount. The open arm splits faces touching the rim, so the count runs
    over boundary vertices too.

    Non-vacuity is asserted on the input: MeshLib finds exactly one valence-3 vertex per chain (the
    innermost), and one pass short of the chain depth leaves some behind.
    """
    depth = 4
    mesh_tm = tm.creation.icosphere(subdivisions=1)
    if not closed:
        mesh_tm = tm.Trimesh(
            mesh_tm.vertices, mesh_tm.faces[mesh_tm.triangles_center[:, 2] < 0.5], process=False
        )
        mesh_tm.remove_unreferenced_vertices()
    boundary_np = np.zeros(len(mesh_tm.vertices), dtype=bool)
    edges_np, counts_np = np.unique(np.sort(mesh_tm.edges, axis=1), axis=0, return_counts=True)
    boundary_np[edges_np[counts_np == 1].ravel()] = True
    on_rim = np.flatnonzero(boundary_np[mesh_tm.faces].any(axis=1))
    chains = [int(face) for face in (on_rim[::7] if not closed else range(0, 80, 9))]
    vertices_np, faces_np = _nested_face_splits(mesh_tm, chains, depth)
    n_splits = len(chains) * depth

    input_ml = numpy_to_meshlib(vertices_np, faces_np)
    input_degree3_ml = meshlib_bitset_to_numpy(
        mm.findInnerVertsOfDegree(input_ml.topology, 3), len(vertices_np)
    )
    assert int(input_degree3_ml.sum()) == len(chains)  # non-vacuity: one per chain

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np.ravel().astype(np.int32), device)
    _, _, short_removed = od.repair.remove_degree3_vertices(
        vertices_wp, faces_wp, max_iter=depth - 1, return_count=True
    )
    assert short_removed < n_splits  # non-vacuity: the cascade really needs every pass

    out_vertices_wp, out_faces_wp, removed = od.repair.remove_degree3_vertices(
        vertices_wp, faces_wp, return_count=True
    )
    assert removed == n_splits
    assert out_vertices_wp.size == len(mesh_tm.vertices)
    assert out_faces_wp.size // 3 == len(mesh_tm.faces)
    output_ml = warp_to_meshlib(out_vertices_wp, out_faces_wp)
    output_degree3_ml = meshlib_bitset_to_numpy(
        mm.findInnerVertsOfDegree(output_ml.topology, 3), out_vertices_wp.size
    )
    assert not output_degree3_ml.any()
    _, _, again_removed = od.repair.remove_degree3_vertices(
        out_vertices_wp, out_faces_wp, return_count=True
    )
    assert again_removed == 0


@pytest.mark.parity("flatten_degree3_vertices", "meshlib")
def test_flatten_degree3_vertices_matches_meshlib(device: str) -> None:
    """
    Class A against ``hardSmoothTetrahedrons``: the same vertex moved to the same place.

    One candidate here, and the adjacent-candidate case is
    ``test_flatten_degree3_vertices_moves_an_independent_set``, which is Class A against the same
    filter on a tetrahedron -- so the pair agrees on both, and this fixture is not load-bearing for
    the scope of the claim.

    Both find the interior valence-3 vertices and hard-set each to the centroid of its three
    neighbours, so this is an element-wise position comparison. The fixture is an icosahedron with
    one face split and the new vertex pushed **0.3 out along that face's normal**, which is what
    gives the repair something to do: at ``bump=0.0`` the vertex is already at the centroid and both
    sides would return the input, passing vacuously.

    The invariant asserted alongside says the flattening happened and no comparison implies it: the
    moved vertex ends up **coplanar** with its three neighbours, which is the geometric claim the
    name makes. Its distance from their plane goes from 0.3 to zero.

    The control is the same mesh at ``bump=0.0``: every position comes back **identical**, so a pass
    that nudged everything a little would fail here rather than merely look close.
    """
    for bump, expect_move in ((0.3, True), (0.0, False)):
        mesh_tm = _mesh_with_a_degree3_vertex(bump)
        vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
        flattened_wp = od.repair.flatten_degree3_vertices(vertices_wp, faces_wp)

        mesh_ml = trimesh_to_meshlib(mesh_tm)
        mm.hardSmoothTetrahedrons(mesh_ml)
        flattened_ml = mn.toNumpyArray(mesh_ml.points)
        moved_ml = np.linalg.norm(flattened_ml - np.asarray(mesh_tm.vertices), axis=1) > 1e-6
        assert bool(moved_ml.any()) is expect_move  # non-vacuity for the bumped case
        assert np.allclose(flattened_wp.numpy(), flattened_ml, rtol=1e-5, atol=1e-5)

        if not expect_move:
            assert np.array_equal(flattened_wp.numpy(), vertices_wp.numpy())
            continue
        # The point of the name: the pimple's apex now lies in its neighbours' plane.
        apex = int(np.flatnonzero(moved_ml)[0])
        ring = np.unique(np.asarray(mesh_tm.faces)[(np.asarray(mesh_tm.faces) == apex).any(axis=1)])
        ring = ring[ring != apex]
        assert len(ring) == 3
        corners_np = flattened_wp.numpy()[ring]
        normal_np = np.cross(corners_np[1] - corners_np[0], corners_np[2] - corners_np[0])
        normal_np /= np.linalg.norm(normal_np)
        before = abs(float(np.dot(np.asarray(mesh_tm.vertices)[apex] - corners_np[0], normal_np)))
        after = abs(float(np.dot(flattened_wp.numpy()[apex] - corners_np[0], normal_np)))
        assert before > 0.25
        assert after < 1e-6
        assert flattened_wp.size == vertices_wp.size


def test_flatten_degree3_vertices_respects_its_region(device: str) -> None:
    """
    Not a library comparison: the axis meshlib carries as a params field and this takes directly.

    An empty region has to be the identity and an all-true one has to reproduce the default, so
    neither the mask nor its absence can be silently ignored. The guards are checked here too --
    a wrong-length mask raises rather than reading past its end.
    """
    mesh_tm = _mesh_with_a_degree3_vertex(0.3)
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    n_vertices = vertices_wp.size
    default_np = od.repair.flatten_degree3_vertices(vertices_wp, faces_wp).numpy()
    assert not np.array_equal(default_np, vertices_wp.numpy())  # non-vacuity

    none_wp = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    assert np.array_equal(
        od.repair.flatten_degree3_vertices(vertices_wp, faces_wp, none_wp).numpy(),
        vertices_wp.numpy(),
    )
    all_wp = wp.full(n_vertices, True, dtype=wp.bool, device=device)
    assert np.array_equal(
        od.repair.flatten_degree3_vertices(vertices_wp, faces_wp, all_wp).numpy(), default_np
    )
    with pytest.raises(ValueError, match="region must be a length-"):
        od.repair.flatten_degree3_vertices(
            vertices_wp, faces_wp, wp.zeros(3, dtype=wp.bool, device=device)
        )


def _genus(faces_wp: wp.array[wp.int32]) -> int:
    """Genus of a closed connected surface, from its Euler characteristic."""
    return (2 - od.measures.euler_characteristic(faces_wp)) // 2


@pytest.fixture
def handles_64(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Build the benchmark's genus-64 slab: the one input whose disjoint short loops are *dependent*.

    Local rather than shared because nothing else in ``tests/`` needs a genus this high. It is the
    fixture for the independence half of ``remove_tunnels``: a vertex-disjoint family of 23 of its
    shortened generators together bounds a piece of the slab, so cutting all 23 splits the surface
    in two -- measured, before the component check, chi -126 -> -78 against the -80 that 23 cuts
    claim. None of the smaller ``_handles`` variants probed (3 to 10 holes a side, at three edge
    lengths) produces a dependent family, so the size is load-bearing.
    """
    vertices_np, faces_np = BUILDERS["handles_64"]()
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.mark.parity(
    "remove_tunnels",
    "trimesh",
    "open3d",
    benchmarked=False,
    reason="neither binds a tunnel eliminator, so there is no repair on either side to time -- and "
    "MeshLib, which does bind one, is a no-op on every input probed. What they contribute is the "
    "predicates the claim rests on: Trimesh.euler_number is an independent chi, so the genus "
    "drop is "
    "not measured by the code that performs it, and Open3D's is_edge_manifold / is_vertex_manifold "
    "are the two post-conditions that stop a shattering cut. All three are timed in their own "
    "groups.",
)
@pytest.mark.parametrize("mesh_name", ["torus_irregular", "genus_two", "handles_64"])
def test_remove_tunnels_drops_the_genus_by_the_count_it_reports(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A on the topology, through chi and the manifold predicates as other libraries read them.

    **No library performs this repair.** MeshLib's ``eliminateTunnels`` is the only binding and it
    is a **no-op** on every input probed -- measured on a 2 048-face torus and a genus-2 union, at
    ``maxTunnelLength`` of 4.0 and of 1e9, at ``maxIters`` 1 / 2 / 5 / 100, at all three
    ``TunnelLoopType`` values, with ``buildCoLoops`` off, and through the ``FillHoleNicelySettings``
    overload: identical face count and identical Euler characteristic every time. Its detector
    *does* fire on the same mesh (``detectTunnelFaces`` returns 128 faces, ``detectBasisTunnels``
    two loops), so this is the "a reference's zero is not always off" case, not a wiring mistake.

    So the *output* has nothing to be compared against, and the invariant carries the claim: cutting
    a surface along a non-separating cycle and sealing the two rims drops the genus by **one**, so
    chi must rise by exactly ``2 * removed``. What the invariant must not do is measure itself.
    Read only through [`euler_characteristic`][ordito.measures.euler_characteristic], the assertion
    is ordito's cut checked by ordito's chi, and this group has no other oracle at all -- so chi
    is read a second time off ``trimesh.Trimesh.euler_number``, and the manifold post-conditions off
    Open3D. Both are independent implementations of the predicates, not of the repair.

    Three more properties come with it -- the result stays connected, closed and edge-manifold --
    and together they exclude the failure this function's shape invites: a cut along loops that
    cross, which shatters the surface into pieces while every individual step still looks correct
    (measured, before the disjointness rule: four spheres from a genus-2 union, and chi 8). And
    disjointness alone is not enough: on ``handles_64`` a disjoint family is *dependent*, splitting
    the slab in two while chi still rises, which is what the component assert catches and why the
    parametrization carries a genus-64 input.

    Open3D's ``is_watertight`` is **not** among them, and the reason is a measured convention rather
    than a defect on either side. It is the composition ``is_edge_manifold && is_vertex_manifold &&
    !is_self_intersecting``, and its last clause counts a *touching* pair as an intersection.
    Sealing two rims over their own vertices produces exactly that: on the torus, **97 of 1 054**
    output faces touch without crossing -- reported by ``findSelfCollidingTrianglesBS`` at
    ``touchIsIntersection=True`` and by pymeshfix's ``select_intersecting_triangles``, and reported
    as **0** by the same MeshLib call at ``touchIsIntersection=False`` and by
    [`face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask]. So the two
    manifold clauses are asserted directly, where the two libraries agree exactly, and the third is
    left to the group that owns it.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    genus_before = _genus(faces_wp)
    assert genus_before >= 1  # non-vacuity: there has to be a tunnel to eliminate
    assert warp_to_trimesh(vertices_wp, faces_wp).euler_number == (
        od.measures.euler_characteristic(faces_wp)
    )

    cut_vertices_wp, cut_faces_wp, removed = od.repair.remove_tunnels(vertices_wp, faces_wp, 1e9)
    assert removed >= 1
    assert od.measures.euler_characteristic(cut_faces_wp) == (
        od.measures.euler_characteristic(faces_wp) + 2 * removed
    )
    assert _genus(cut_faces_wp) == genus_before - removed
    assert od.validation.is_edge_manifold(cut_faces_wp)
    assert len(od.boundary.boundary_loops(cut_vertices_wp, cut_faces_wp)) == 0
    labels_np = od.adjacency.face_connected_component_labels(cut_faces_wp).numpy()
    assert np.unique(labels_np).size == 1
    # Every output position is an input position: the rims are filled over their own vertices.
    assert cut_vertices_wp.size >= vertices_wp.size

    # The genus drop, and the two properties that stop a shattering cut, read by other libraries.
    cut_tm = warp_to_trimesh(cut_vertices_wp, cut_faces_wp)
    assert cut_tm.euler_number == od.measures.euler_characteristic(faces_wp) + 2 * removed
    mesh_o3d = trimesh_to_open3d(cut_tm)
    assert mesh_o3d.is_edge_manifold(allow_boundary_edges=False)
    assert mesh_o3d.is_vertex_manifold()


def test_remove_tunnels_iterates_to_a_sphere(genus_two: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: see above. This pins the documented "call it again" contract.

    One call takes at most one loop per vertex-disjoint family, so a basis whose loops all overlap
    needs another round. The docstring tells callers to loop until ``removed`` is ``0``, and this
    is that loop: a genus-2 union reaches genus 0 in **two** rounds and the third reports nothing,
    which is both the termination proof and the reason the count is not just the genus.
    """
    _, mesh_wp = genus_two
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    rounds = 0
    while True:
        vertices_wp, faces_wp, removed = od.repair.remove_tunnels(vertices_wp, faces_wp, 1e9)
        if removed == 0:
            break
        rounds += 1
        assert rounds <= 4  # it must terminate, and two rounds is what this fixture takes
    assert rounds == 2
    assert _genus(faces_wp) == 0
    assert od.validation.is_edge_manifold(faces_wp)


def test_remove_tunnels_leaves_a_long_tunnel_and_a_sphere_alone(
    torus_irregular: tuple[tm.Trimesh, wp.Mesh], sphere_irregular: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: see above. The two do-nothing branches, which are the safety claim.

    ``max_length`` below the tunnel's own girth must leave the mesh **identical**, not merely
    equivalent -- that is what makes the function safe to run on a mesh whose genus is intended. And
    a genus-0 input has no basis at all, so it returns before cutting anything.
    """
    _, torus_wp = torus_irregular
    kept_vertices_wp, kept_faces_wp, removed = od.repair.remove_tunnels(
        torus_wp.points, torus_wp.indices, 0.5
    )
    assert removed == 0
    assert np.array_equal(kept_faces_wp.numpy(), torus_wp.indices.numpy())
    assert np.array_equal(kept_vertices_wp.numpy(), torus_wp.points.numpy())

    _, sphere_wp = sphere_irregular
    assert _genus(sphere_wp.indices) == 0
    _, sphere_faces_wp, sphere_removed = od.repair.remove_tunnels(
        sphere_wp.points, sphere_wp.indices, 1e9
    )
    assert sphere_removed == 0
    assert np.array_equal(sphere_faces_wp.numpy(), sphere_wp.indices.numpy())

    with pytest.raises(ValueError, match="max_length must be non-negative"):
        od.repair.remove_tunnels(torus_wp.points, torus_wp.indices, -1.0)


def test_remove_t_vertices_flips_the_sliver(
    device: str, t_vertex_patch: tuple[np.ndarray, np.ndarray]
) -> None:
    """The sliver goes, the face count and the vertices stay, and the patch stays manifold."""
    vertices_np, faces_np = t_vertex_patch
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    before = _worst_aspect(vertices_wp, faces_wp)
    assert before > 40.0  # the fixture really does carry a T-vertex sliver

    flipped_wp = od.repair.flip_t_vertices(vertices_wp, faces_wp, threshold=40.0)
    assert _worst_aspect(vertices_wp, flipped_wp) < before
    assert flipped_wp.size == faces_wp.size
    assert od.validation.is_winding_consistent(flipped_wp)
    assert od.validation.is_edge_manifold(flipped_wp)


@pytest.mark.parametrize("mesh_kind", ["t_vertex_patch", "clean_icosphere"])
@pytest.mark.parity("flip_t_vertices", "pymeshlab")
def test_remove_t_vertices_matches_pymeshlab(
    device: str, mesh_kind: str, t_vertex_patch: tuple[np.ndarray, np.ndarray]
) -> None:
    """
    Class B: MeshLab's ``Edge Flip`` mode picks the *identical* flips, face for face.

    ``meshing_remove_t_vertices`` rewrites the topology in place rather than returning a face
    buffer, so the transform is reading ``face_matrix()`` back and comparing the two as sorted face
    sets (neither library defines a face or corner order). ``method="Edge Flip"`` selects the
    flip-only mode -- its other modes split edges and would change the vertex count -- and
    ``repeat=True`` is MeshLab's own iterate-to-convergence, which is what ordito's ``max_iter``
    passes are. Both are what the benchmark passes.

    Two-sided by construction, so neither answer can be reached by a constant: on the T-vertex patch
    both flip the sliver and land on the same 6 faces, **different from the input's 6**; on a clean
    icosphere neither touches anything. Vertex counts are unchanged on both sides in both cases,
    which is the check that MeshLab really took the flip path and not a split. On the icosphere no
    triangle is anywhere near the threshold, and ordito's face buffer must come back byte for byte.
    """
    if mesh_kind == "t_vertex_patch":
        vertices_np, faces_np = t_vertex_patch
        expect_change = True
    else:
        sphere_tm = tm.creation.icosphere(subdivisions=3)
        vertices_np, faces_np = np.asarray(sphere_tm.vertices), np.asarray(sphere_tm.faces)
        expect_change = False
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    meshset_pml = trimesh_to_pymeshlab(tm.Trimesh(vertices_np, faces_np, process=False))
    meshset_pml.meshing_remove_t_vertices(method="Edge Flip", threshold=40.0, repeat=True)
    assert meshset_pml.current_mesh().vertex_number() == vertices_np.shape[0]
    faces_pml = np.asarray(meshset_pml.current_mesh().face_matrix())

    flipped_np = od.repair.flip_t_vertices(vertices_wp, faces_wp, threshold=40.0).numpy()

    def face_set(faces: np.ndarray) -> np.ndarray:
        sorted_np = np.sort(np.asarray(faces).reshape(-1, 3), axis=1)
        return lexsort_rows(sorted_np)

    assert np.array_equal(face_set(flipped_np), face_set(faces_pml))
    assert (not np.array_equal(face_set(flipped_np), face_set(faces_np))) is expect_change
    if not expect_change:
        # Not just the same set: a clean mesh comes back as the identical buffer.
        assert np.array_equal(flipped_np, faces_wp.numpy())


# ---------------------------------------------------------------------------
# empty mesh is a no-op, across every repair operator
# ---------------------------------------------------------------------------

# (name, callable) pairs; each callable takes (vertices, faces) and returns the tuple of arrays
# the wrapper produces, normalized to a tuple even where the wrapper returns a single array.
_EMPTY_MESH_REPAIR_CASES = [
    ("make_solid", lambda v, f: od.repair.make_solid(v, f)),
    ("reverse_winding", lambda v, f: (od.repair.reverse_winding(f),)),
    ("make_winding_consistent", lambda v, f: (od.repair.make_winding_consistent(f),)),
    ("make_volume", lambda v, f: (od.repair.make_volume(v, f),)),
    ("make_normals_outward", lambda v, f: (od.repair.make_normals_outward(v, f),)),
    ("split_non_manifold_vertices", lambda v, f: od.repair.split_non_manifold_vertices(v, f)),
    ("collapse_small_triangles", lambda v, f: od.repair.collapse_small_triangles(v, f)),
    ("remove_folded_faces", lambda v, f: od.repair.remove_folded_faces(v, f)),
    ("flip_t_vertices", lambda v, f: (od.repair.flip_t_vertices(v, f),)),
]


@pytest.mark.parametrize(
    "repair_fn",
    [case[1] for case in _EMPTY_MESH_REPAIR_CASES],
    ids=[case[0] for case in _EMPTY_MESH_REPAIR_CASES],
)
def test_repair_empty_mesh_is_a_noop(
    device: str,
    repair_fn: Callable[[wp.array[wp.vec3], wp.array[wp.int32]], tuple[wp.array[object], ...]],
) -> None:
    """
    Not a library comparison: every repair operator returns an all-empty result on an empty mesh.

    No reference is consulted here -- the claim is only that the shape stays ``(0,)`` through
    every array a repair function returns, rather than raising.
    """
    vertices_wp = warp_empty(0, wp.vec3, device)
    faces_wp = warp_empty(0, wp.int32, device)
    for result_wp in repair_fn(vertices_wp, faces_wp):
        assert result_wp.size == 0


# ---------------------------------------------------------------------------
# a clean mesh is a no-op, across the repairs that remove or rewind faces
# ---------------------------------------------------------------------------

# (name, callable) pairs; each callable takes (vertices, faces) and returns (vertices, faces), the
# input vertices standing in where the wrapper returns faces only.
_CLEAN_MESH_REPAIR_CASES = [
    ("make_winding_consistent", lambda v, f: (v, od.repair.make_winding_consistent(f))),
    ("make_volume", lambda v, f: (v, od.repair.make_volume(v, f))),
    ("remove_non_manifold_faces", lambda v, f: od.repair.remove_non_manifold_faces(v, f)),
    ("remove_degenerate_faces", lambda v, f: od.repair.remove_degenerate_faces(v, f)),
    (
        "collapse_small_triangles",
        lambda v, f: od.repair.collapse_small_triangles(v, f, epsilon=1e-9),
    ),
    # Not ``remove_folded_faces``: ``sphere_irregular`` has one genuine fold (a 163.9-degree
    # dihedral at its sharpest spike, past the 160-degree default), so it is not a clean input for
    # that predicate; ``test_remove_folded_faces_drops_exactly_the_folded_pairs`` holds it to the
    # exact set instead.
]


@pytest.mark.parametrize(
    "repair_fn",
    [case[1] for case in _CLEAN_MESH_REPAIR_CASES],
    ids=[case[0] for case in _CLEAN_MESH_REPAIR_CASES],
)
def test_repair_clean_mesh_is_a_noop(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
    repair_fn: Callable[
        [wp.array[wp.vec3], wp.array[wp.int32]], tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    ],
) -> None:
    """
    Not a library comparison: a clean mesh comes back with the same faces and vertices.

    The icosahedron is consistently wound, outward, manifold and well shaped. This is the no-op
    direction, which a threshold that is slightly too aggressive fails while still passing every
    test that feeds it a genuinely defective face. ``collapse_small_triangles`` has no reference at
    all here -- it is unbound in the igl wheel (section 6) -- so the numpy oracle covers its
    collapsing case and this its identity. ``remove_non_manifold_faces`` also hands back the
    caller's own vertex buffer, since nothing was rebuilt.
    """
    mesh_tm, mesh_wp = sphere_irregular
    vertices_wp, faces_wp = repair_fn(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(faces_wp.numpy().reshape(-1, 3), mesh_tm.faces)
    assert np.allclose(vertices_wp.numpy(), mesh_wp.points.numpy(), rtol=1e-6, atol=1e-6)
    if repair_fn is _CLEAN_MESH_REPAIR_CASES[2][1]:
        assert vertices_wp.ptr == mesh_wp.points.ptr


# ---------------------------------------------------------------------------
# the documented argument guards
# ---------------------------------------------------------------------------

_INVALID_ARGUMENT_CASES = [
    (
        "make_solid_max_iter",
        lambda v, f: od.repair.make_solid(v, f, max_iter=-1),
        "must be non-negative",
    ),
    (
        "make_solid_inner_iter",
        lambda v, f: od.repair.make_solid(v, f, inner_iter=-1),
        "must be non-negative",
    ),
    ("components_none", lambda v, f: od.repair.remove_small_components(v, f), "exactly one"),
    (
        "components_two",
        lambda v, f: od.repair.remove_small_components(v, f, keep_largest=True, min_faces=3),
        "exactly one",
    ),
    (
        "components_two_mins",
        lambda v, f: od.repair.remove_small_components(v, f, min_area=1.0, min_diameter=1.0),
        "exactly one",
    ),
    (
        "flip_t_vertices_threshold",
        lambda v, f: od.repair.flip_t_vertices(v, f, threshold=0.0),
        "aspect_threshold must be positive",
    ),
]


@pytest.mark.parametrize(
    ("repair_fn", "match"),
    [(case[1], case[2]) for case in _INVALID_ARGUMENT_CASES],
    ids=[case[0] for case in _INVALID_ARGUMENT_CASES],
)
def test_repair_rejects_invalid_arguments(
    sphere_irregular: tuple[tm.Trimesh, wp.Mesh],
    repair_fn: Callable[[wp.array[wp.vec3], wp.array[wp.int32]], object],
    match: str,
) -> None:
    """Not a library comparison: each documented ``ValueError``, raised before any work."""
    _mesh_tm, mesh_wp = sphere_irregular
    with pytest.raises(ValueError, match=match):
        repair_fn(mesh_wp.points, mesh_wp.indices)
