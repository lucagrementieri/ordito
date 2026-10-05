"""
Regression tests for ``ordito.levelset``.

The level-set offset against pymeshlab's uniform resampler and MeshLib's ``offsetMesh``, both of
which march the same kind of field -- plus the invariant that is stronger than either comparison:
every output vertex must sit at the requested signed distance from the input.
"""

from __future__ import annotations

import math

import igl
import numpy as np
import pymeshlab as ml
import pytest
import pyvista as pv
import torch
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from pytorch3d.ops.marching_cubes import marching_cubes as p3d_marching_cubes
from scipy.spatial import cKDTree

import ordito as od
import ordito.typing as odt
from tests.comparisons import (
    assert_unordered_rows_equal,
    canonical_winding,
    euler_characteristic,
    hausdorff_surface_two_sided,
    hausdorff_two_sided,
    open_edge_count,
)
from tests.conversions import (
    meshlib_to_trimesh,
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_pymeshlab,
    warp_empty,
    warp_to_trimesh,
)

# One spacing for every comparison here, passed to both sides so neither is resampled finer than
# the other. 0.05 on a unit sphere is ~70 samples across, which is what the automatic default lands
# on and coarse enough to keep the reference rows quick.
_VOXEL = 0.05


def _sphere_field(resolution: int, radius: float) -> np.ndarray:
    """Build an analytic SDF of a sphere on a ``[-1, 1]`` lattice: the exact-answer fixture."""
    axis_np = np.linspace(-1.0, 1.0, resolution)
    x_np, y_np, z_np = np.meshgrid(axis_np, axis_np, axis_np, indexing="ij")
    return (np.sqrt(x_np**2 + y_np**2 + z_np**2) - radius).astype(np.float32)


def test_marching_cubes_extracts_an_analytic_sphere(device: str) -> None:
    """
    Not a library comparison: the extracted vertices lie on the sphere the field describes.

    With ``bounds`` every vertex must land on the sphere to grid resolution; without them the
    vertices are lattice indices, the documented convention, so they lie inside the lattice and
    the centred field's surface is centred on the lattice centre.
    """
    radius, resolution = 0.6, 32
    field_wp = odt.as_array3d(
        wp.array(_sphere_field(resolution, radius), dtype=wp.float32, device=device), wp.float32
    )
    vertices_wp, faces_wp = od.levelset.marching_cubes(
        field_wp, bounds=(wp.vec3(-1.0, -1.0, -1.0), wp.vec3(1.0, 1.0, 1.0))
    )
    assert faces_wp.size > 0
    spacing = 2.0 / (resolution - 1)
    radii_np = np.linalg.norm(vertices_wp.numpy(), axis=1)
    assert np.abs(radii_np - radius).max() < spacing

    index_vertices_np = od.levelset.marching_cubes(field_wp)[0].numpy()
    assert index_vertices_np.min() >= 0.0
    assert index_vertices_np.max() <= float(resolution - 1)
    assert np.allclose(index_vertices_np.mean(axis=0), 0.5 * (resolution - 1), atol=0.5)


@pytest.mark.parity("marching_cubes", "igl", "pyvista")
def test_marching_cubes_matches_igl_and_pyvista(device: str) -> None:
    """
    Class A: the same vertices and the same triangles, once each lattice convention is applied.

    Three implementations of one case table, and they agree **exactly** rather than closely -- on a
    24^3 unit-sphere SDF over ``[-1.5, 1.5]^3``, all three return **1 128** vertices, **2 252**
    faces and a mean radius of **0.999226** to six digits. That is what makes this group the
    best-referenced function in the package; the fourth implementation (meshlib) is the test above.

    The whole content of the comparison is the lattice convention, and the two references disagree
    with each other about it, which is the reason both are here:

    * **igl** takes the sample *positions* explicitly as ``GV`` and wants them in **Fortran** order,
      with ``nx, ny, nz`` separately. It returns **three** values -- a 2-tuple unpack raises
      ``ValueError: too many values to unpack``.
    * **pyvista**'s ``ImageData`` addresses samples on grid **nodes**, so its ``origin`` is
      ordito's ``bounds`` lower corner *directly* -- the exact opposite of MeshLib's
      voxel-*centre* origin, which the test above shifts by half a voxel. Its ``point_data`` wants
      Fortran order too.

    Getting either convention wrong shifts the marched surface by up to a voxel rather than raising,
    so the mean-radius assert is the guard: at this resolution a half-voxel shift moves it by 0.065,
    five hundred times the 1e-04 tolerance below.

    **Bug class excluded:** a case table that emits the right count of the wrong triangles. Vertex
    *sets* are matched by nearest neighbour with a bijection check rather than by index, since
    nothing pins the three libraries to one emission order.
    """
    resolution, half, radius = 24, 1.5, 1.0
    axis_np = np.linspace(-half, half, resolution)
    x_np, y_np, z_np = np.meshgrid(axis_np, axis_np, axis_np, indexing="ij")
    field_np = np.sqrt(x_np**2 + y_np**2 + z_np**2) - radius

    field_wp = wp.array(
        np.ascontiguousarray(field_np, dtype=np.float32), dtype=wp.float32, device=device
    )
    bounds = (wp.vec3(-half, -half, -half), wp.vec3(half, half, half))
    vertices_wp, faces_wp = od.levelset.marching_cubes(
        odt.as_array3d(field_wp, wp.float32), 0.0, bounds=bounds
    )
    vertices_np = vertices_wp.numpy().astype(np.float64)

    lattice_igl = np.ascontiguousarray(
        np.stack([x_np.ravel(order="F"), y_np.ravel(order="F"), z_np.ravel(order="F")], axis=1),
        dtype=np.float64,
    )
    vertices_igl, faces_igl, _info_igl = igl.marching_cubes(
        np.ascontiguousarray(field_np.ravel(order="F"), dtype=np.float64),
        lattice_igl,
        resolution,
        resolution,
        resolution,
        0.0,
    )

    spacing = 2.0 * half / (resolution - 1)
    grid_pv = pv.ImageData(
        dimensions=(resolution, resolution, resolution),
        origin=(-half, -half, -half),
        spacing=(spacing, spacing, spacing),
    )
    grid_pv.point_data["field"] = field_np.ravel(order="F")
    contour_pv = grid_pv.contour([0.0], scalars="field")
    vertices_pv = np.asarray(contour_pv.points, dtype=np.float64)

    assert len(vertices_np) > 0  # non-vacuity: there is a surface to compare
    for reference_np, n_faces in (
        (np.asarray(vertices_igl, dtype=np.float64), np.asarray(faces_igl).shape[0]),
        (vertices_pv, contour_pv.n_cells),
    ):
        assert reference_np.shape[0] == len(vertices_np)
        assert n_faces == faces_wp.size // 3
        # A half-voxel origin error moves this by 0.065; the tolerance is 1e-04.
        assert np.isclose(
            np.linalg.norm(reference_np, axis=1).mean(),
            np.linalg.norm(vertices_np, axis=1).mean(),
            rtol=0.0,
            atol=1e-4,
        )
        # The vertex sets match as sets: no library promises an emission order.
        residual_np, matched_np = map(np.asarray, cKDTree(vertices_np).query(reference_np))
        assert residual_np.max() < 1e-5
        assert np.unique(matched_np).size == reference_np.shape[0]


@pytest.mark.parity("marching_cubes", "meshlib")
def test_marching_cubes_matches_meshlib(device: str) -> None:
    """
    Class B, and the transform is half a voxel: ``params.origin`` addresses the voxel **centre**.

    MeshLib's ``marchingCubes`` marches the same lattice with the same case table, so given the
    identical field the two return the *same mesh* -- 3 744 vertices and 7 484 faces on both sides
    here, agreeing to a two-sided Hausdorff of **1.2e-07**, which is the float32 floor
    ``getNumpyVerts`` bottoms out at (CLAUDE.md section 7.6). The transform is the whole content of
    the comparison and it is load-bearing: where ordito's ``bounds`` lower corner is the position
    of sample ``[0, 0, 0]``, ``params.origin`` is that sample's *cell* corner, so passing the same
    number to both leaves the surfaces a rigid half-voxel apart -- measured at 0.0369, exactly the
    half diagonal ``sqrt(3) * spacing / 2``, and 3e5 times the agreement the shift buys.

    ``lessInside=True`` is the other convention, and it is the winding rather than the geometry:
    with it the extracted volume is ``+0.902`` against ordito's ``+0.902`` (8e-08 relative), and
    with ``lessInside=False`` it is exactly the negation. True is the value that matches ordito's
    outside-positive field convention.

    The invariants beside the comparison are what a vertex-cloud match cannot see: both meshes are
    closed (no boundary edge) with Euler characteristic 2, and their triangle centroids match as
    well as their vertices do -- so the two agree on the *triangulation*, not merely on the point
    set.
    """
    resolution, radius = 48, 0.6
    field_np = _sphere_field(resolution, radius)
    spacing = 2.0 / (resolution - 1)
    field_wp = wp.array(field_np, dtype=wp.float32, device=device)
    vertices_wp, faces_wp = od.levelset.marching_cubes(
        odt.as_array3d(field_wp, wp.float32),
        bounds=(wp.vec3(-1.0, -1.0, -1.0), wp.vec3(1.0, 1.0, 1.0)),
    )
    vertices_np, faces_od_np = vertices_wp.numpy(), faces_wp.numpy().reshape(-1, 3)

    def march_ml(origin: float) -> tuple[np.ndarray, np.ndarray]:
        """March the identical field with the lower corner at ``origin`` on every axis."""
        # The stub names the parameter ``3DvoxelsArray``, which is not an identifier.
        volume_ml = mn.simpleVolumeFrom3Darray(field_np)  # pyright: ignore[reportCallIssue]
        volume_ml.voxelSize = mm.Vector3f(spacing, spacing, spacing)
        params_ml = mm.MarchingCubesParams()
        params_ml.iso = 0.0
        params_ml.lessInside = True
        params_ml.origin = mm.Vector3f(origin, origin, origin)
        mesh_ml = mm.marchingCubes(volume_ml, params_ml)
        return np.asarray(mn.getNumpyVerts(mesh_ml)), np.asarray(mn.getNumpyFaces(mesh_ml.topology))

    vertices_ml_np, faces_ml_np = march_ml(-1.0 - spacing / 2)
    assert faces_ml_np.shape[0] > 0
    assert vertices_ml_np.shape[0] == len(vertices_np)
    assert faces_ml_np.shape[0] == faces_od_np.shape[0]
    assert hausdorff_two_sided(vertices_np, vertices_ml_np) < 1e-5
    assert (
        hausdorff_two_sided(
            vertices_np[faces_od_np].mean(axis=1), vertices_ml_np[faces_ml_np].mean(axis=1)
        )
        < 1e-5
    )

    # The transform is the claim, so show the un-shifted call fails by the half diagonal.
    unshifted_np = march_ml(-1.0)[0]
    assert hausdorff_two_sided(vertices_np, unshifted_np) == pytest.approx(
        math.sqrt(3.0) * spacing / 2.0, rel=1e-3
    )

    # Invariants a point-set match cannot see: both are closed spheres, and both wind outward.
    for mesh_faces_np in (faces_od_np, faces_ml_np):
        assert open_edge_count(mesh_faces_np) == 0
        assert euler_characteristic(mesh_faces_np) == 2
    volume_od = tm.Trimesh(vertices_np, faces_od_np, process=False).volume
    volume_ml = tm.Trimesh(vertices_ml_np, faces_ml_np, process=False).volume
    assert volume_od > 0.0
    assert volume_ml == pytest.approx(volume_od, rel=1e-5)


@pytest.mark.parity("marching_cubes", "pytorch3d")
def test_marching_cubes_matches_pytorch3d(device: str) -> None:
    """
    Class A: the fifth implementation of the case table, in ordito's own index space.

    ``return_local_coords=False`` is what makes this a direct comparison: with it ``True``
    pytorch3d rescales its output into a normalized ``[-1, 1]`` cube and the two would differ by an
    affine map. Off, it emits lattice indices, which is exactly what ``marching_cubes`` returns
    without ``bounds`` -- so the comparison needs no convention fix at all, unlike the igl /
    pyvista / meshlib trio above, each of which needed a different one.

    Measured on a 16^3 radial field at ``iso=1.0``: **480** vertices and **956** faces from both
    sides, and the matched coordinates agree at **0.0**. Matched rather than indexed because
    nothing pins two case-table walks to one emission order -- and matched by ``cKDTree`` with a
    bijection check rather than by ``lexsort_rows``, which CLAUDE.md section 7.5 records as
    unusable on float coordinates: the rounding this used to do mitigates the hazard without
    removing it, since two values straddling a rounding boundary still sort differently.

    Note the import: ``marching_cubes`` is **not** re-exported from ``pytorch3d.ops``, only from
    ``pytorch3d.ops.marching_cubes``, so ``p3d_ops.marching_cubes`` is an ``AttributeError``.
    """
    resolution, extent, iso = 16, 1.5, 1.0
    axis_np = np.linspace(-extent, extent, resolution)
    x_np, y_np, z_np = np.meshgrid(axis_np, axis_np, axis_np, indexing="ij")
    field_np = (x_np**2 + y_np**2 + z_np**2).astype(np.float32)

    vertices_p3d, faces_p3d = p3d_marching_cubes(
        torch.as_tensor(field_np, device=device)[None], isolevel=iso, return_local_coords=False
    )
    vertices_p3d = vertices_p3d[0].cpu().numpy()

    field_wp = wp.array(field_np, dtype=wp.float32, device=device)
    vertices_wp, faces_wp = od.levelset.marching_cubes(odt.as_array3d(field_wp, wp.float32), iso)

    assert vertices_p3d.shape[0] > 0
    assert vertices_wp.size == vertices_p3d.shape[0]
    assert faces_wp.size // 3 == faces_p3d[0].shape[0]
    distance_np, match_np = map(np.asarray, cKDTree(vertices_p3d).query(vertices_wp.numpy()))
    assert distance_np.max() == 0.0, f"vertices differ by up to {distance_np.max():.3e}"
    assert len(set(match_np.tolist())) == match_np.size, "the vertex match is not a bijection"


def test_marching_cubes_empty_and_invalid(device: str) -> None:
    """Not a library comparison: a field that never crosses is empty, a 1-wide lattice raises."""
    field_wp = wp.array(np.full((8, 8, 8), 1.0, dtype=np.float32), dtype=wp.float32, device=device)
    _vertices_wp, faces_wp = od.levelset.marching_cubes(odt.as_array3d(field_wp, wp.float32))
    assert faces_wp.size == 0

    thin_wp = wp.array(np.zeros((1, 8, 8), dtype=np.float32), dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="at least 2 wide"):
        od.levelset.marching_cubes(odt.as_array3d(thin_wp, wp.float32))


def _signed_distance_to(
    mesh_wp: tuple[wp.array[wp.vec3], wp.array[wp.int32]], points_np: np.ndarray
) -> np.ndarray:
    """Signed distance from every row of ``points_np`` to the mesh, by winding sign."""
    vertices_wp, faces_wp = mesh_wp
    points_wp = points_to_warp(points_np, vertices_wp.device)
    return od.proximity.signed_distance_on_mesh(
        vertices_wp, faces_wp, points_wp, sign_mode="winding"
    ).numpy()


@pytest.mark.parametrize("distance", [0.2, -0.2])
@pytest.mark.parity("offset_mesh", "meshlib")
def test_offset_mesh_lands_at_the_distance_and_matches_meshlib(
    icosphere: tuple[tm.Trimesh, wp.Mesh], distance: float
) -> None:
    """
    The defining property, then Class C against MeshLib's ``offsetMesh`` at a matched voxel size.

    Not a library comparison, first: every output vertex is at signed distance ``distance``. This
    is the check that actually constrains the function, and it is stronger than any comparison
    against another implementation -- an offset surface *is* a level set of the distance field, so
    measuring the field at the output is measuring the answer. It is measured with
    ``signed_distance_on_mesh``, which is not the code under test's own field sampler applied twice:
    the offset marches a **lattice** and this queries the **vertices** it produced. The tolerance is
    the lattice's: a marching-cubes vertex is linearly interpolated inside a cell, so it lands
    within a fraction of ``_VOXEL`` of the true level set rather than within a whole cell. Measured
    max deviation **0.0019** at a 0.05 spacing, i.e. 3.8 % of one cell. The sign of the volume
    change is asserted too, which no distance check would catch: an outward offset must enclose
    more and an inward one less.

    Class C against MeshLib: no correspondence exists between the two triangulations -- both march
    their own field on their own lattice -- so the comparison is the two-sided Hausdorff distance
    between the *surfaces*, plus the vertex counts as a sanity check on the resolution actually
    used. They agree closely enough that the counts are worth asserting: measured **10 746 against
    10 736** vertices at ``distance = 0.2`` and 4 758 against 4 760 at ``-0.2``, i.e. within 0.1 %,
    because at a matched spacing the two lattices differ only in where their origin falls.
    ``OffsetParameters.voxelSize`` is set explicitly rather than left at its default, which is the
    parameter that would otherwise decide the comparison.

    **Mutation probe**, and it retightened the threshold. The bug class the deviation bar has to
    exclude is an offset applied at the wrong *distance*, so the probe re-runs MeshLib at a wrong
    one: against a measured agreement of **0.00263**, 0.15 and 0.25 (25 % out) give 0.0528 and
    0.0515 and both **fail**, but **0.22 -- 10 % out -- gives 0.0210**, which the original
    ``0.5 * _VOXEL`` bar (0.025, a 9.5x headroom) let through. It is now ``0.25 * _VOXEL``: 0.0125,
    still **4.8x** the agreement, and the same bar
    [`test_offset_mesh_matches_pymeshlab`][] carries, so the two references are held to one
    standard.
    """
    mesh_tm, mesh_wp = icosphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    offset_vertices_wp, offset_faces_wp = od.levelset.offset_mesh(
        vertices_wp, faces_wp, distance, _VOXEL
    )
    assert offset_faces_wp.size > 0

    signed_np = _signed_distance_to((vertices_wp, faces_wp), offset_vertices_wp.numpy())
    assert np.abs(signed_np - distance).max() < 0.1 * _VOXEL

    volume_before = float(od.measures.volume(vertices_wp, faces_wp))
    volume_after = float(od.measures.volume(offset_vertices_wp, offset_faces_wp))
    assert (volume_after > volume_before) is (distance > 0.0)
    assert od.validation.is_watertight(offset_vertices_wp, offset_faces_wp)

    parameters_ml = mm.OffsetParameters()
    parameters_ml.voxelSize = _VOXEL
    offset_ml = meshlib_to_trimesh(
        mm.offsetMesh(mm.MeshPart(trimesh_to_meshlib(mesh_tm)), distance, parameters_ml)
    )
    assert offset_ml.faces.shape[0] > 0  # non-vacuity: the reference produced a surface

    count_wp = offset_vertices_wp.size
    count_ml = offset_ml.vertices.shape[0]
    assert abs(count_wp - count_ml) < 0.05 * count_ml

    offset_tm = warp_to_trimesh(offset_vertices_wp, offset_faces_wp)
    deviation = hausdorff_surface_two_sided(
        np.asarray(offset_tm.vertices, dtype=np.float64),
        np.asarray(offset_tm.faces),
        np.asarray(offset_ml.vertices, dtype=np.float64),
        np.asarray(offset_ml.faces),
    )
    assert deviation < 0.25 * _VOXEL


@pytest.mark.parity("offset_mesh", "pymeshlab")
def test_offset_mesh_matches_pymeshlab(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C: MeshLab's uniform resampler at a matched cell size and the same absolute offset.

    ``generate_resampled_uniform_mesh`` is MeshLab's offset, and its two length parameters are the
    trap: both take a wrapper type, and its ``offset`` as a ``PercentageValue`` runs from *full
    erosion* at 0 % to full dilation at 100 %, so its own 50 % default is the **zero** offset.
    ``PureValue`` is therefore mandatory here, and it is the same number ordito gets.

    Compared on the surfaces (two-sided Hausdorff) and on the same distance invariant the test above
    applies to ordito: MeshLab's own output sits at mean signed distance **+0.1999** with a spread
    of 0.0002 from the input, so the two implementations are measuring the same quantity rather than
    two things that happen to look alike.

    **Mutation probe**, and it retightened the threshold, identically to the MeshLab comparison
    above. Measured agreement **0.00179**, the same on both devices; re-running pymeshlab at a wrong
    offset gives 0.0521 at 0.15 and 0.0516 at 0.25 (25 % out, both **fail**) and **0.0212 at 0.22**
    -- 10 % out, which the original ``0.5 * _VOXEL`` bar (0.025, a 14x headroom) admitted. It is now
    ``0.25 * _VOXEL``: 0.0125, still **7.0x** the measured agreement.
    """
    distance = 0.2
    mesh_tm, mesh_wp = icosphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    offset_vertices_wp, offset_faces_wp = od.levelset.offset_mesh(
        vertices_wp, faces_wp, distance, _VOXEL
    )

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.generate_resampled_uniform_mesh(
        cellsize=ml.PureValue(_VOXEL), offset=ml.PureValue(distance), mergeclosevert=True
    )
    mesh_pml = meshset_pml.current_mesh()
    offset_pml = tm.Trimesh(mesh_pml.vertex_matrix(), mesh_pml.face_matrix(), process=False)
    assert offset_pml.faces.shape[0] > 0

    signed_pml = _signed_distance_to((vertices_wp, faces_wp), np.asarray(offset_pml.vertices))
    assert np.abs(signed_pml.mean() - distance) < 0.1 * _VOXEL  # same quantity, not just a shape

    offset_tm = warp_to_trimesh(offset_vertices_wp, offset_faces_wp)
    deviation = hausdorff_surface_two_sided(
        np.asarray(offset_tm.vertices, dtype=np.float64),
        np.asarray(offset_tm.faces),
        np.asarray(offset_pml.vertices, dtype=np.float64),
        np.asarray(offset_pml.faces),
    )
    assert deviation < 0.25 * _VOXEL


def test_offset_mesh_resolves_what_survives_a_large_inward_offset(
    icosphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: the automatic ``voxel_size``'s resolution **floor**, and its absence.

    Tying the spacing to the offset distance alone -- the obvious rule, and the first one written
    -- resolves the band the level set sits in and not what is left of the object. An inward offset
    of 0.9 on a unit sphere leaves a sphere of radius ~0.1, which at a spacing of ``0.9 / 3`` is
    smaller than a single cell: the call returned **empty** for a level set that plainly exists. A
    floor of 64 samples across the mesh fixes it, and this is the case that would fail without it.

    The genuinely empty case is asserted beside it, since the two must stay distinguishable: at
    ``-1.5`` there is no point at that distance inside a unit sphere and an empty answer is correct.
    """
    _mesh_tm, mesh_wp = icosphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices

    survivor_vertices_wp, survivor_faces_wp = od.levelset.offset_mesh(vertices_wp, faces_wp, -0.9)
    assert survivor_faces_wp.size > 0
    radius_np = np.linalg.norm(survivor_vertices_wp.numpy(), axis=1)
    assert 0.05 < radius_np.max() < 0.15  # the sphere that is left, not a stray cell

    _empty_vertices_wp, empty_faces_wp = od.levelset.offset_mesh(vertices_wp, faces_wp, -1.5)
    assert empty_faces_wp.size == 0


@pytest.mark.parametrize(
    ("mesh_name", "sparse_expected", "distance"),
    [
        *(
            (name, True, distance)
            for name in ("icosphere", "cave_cube")
            for distance in (0.08, -0.04)
        ),
        # The open arms pin only that the gate keeps the dense lattice, which one distance shows.
        *((name, False, 0.08) for name in ("boy_surface", "hemisphere", "half_torus")),
    ],
)
def test_offset_mesh_sparse_extraction_matches_the_dense_lattice(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    mesh_name: str,
    sparse_expected: bool,
    distance: float,
) -> None:
    """
    Ordito against ordito: the sparse extraction is the dense lattice's surface, and only it runs.

    ``test_offset_mesh_lands_at_the_distance_and_matches_meshlib`` and the pymeshlab comparison
    carry the oracle on the dense path; this pins the sparse path to it by lowering the node gate
    to zero (every fixture here is far below the shipped one). On a closed, consistently wound
    input the two must agree vertex for vertex up to the order of the buffers -- matched by nearest
    neighbour with a bijection check -- and face for face with the same winding. On an open or
    non-orientable input the gate must keep the dense lattice, whose winding-signed field jumps
    away from the surface: the sparse extraction measured a fifth of a hemisphere's offset faces
    missing there.

    ``cave_cube`` is turned by a generic rotation first. Axis-aligned, its offset level runs
    exactly through lattice nodes, and a node whose distance rounds onto the level lands on
    different sides in the two extractions (they place the same node one rounding apart), which
    picks different, equally valid triangulations there -- a tie, not the defect under test.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    if mesh_name == "cave_cube":
        rotation_np = np.asarray(tm.transformations.rotation_matrix(0.3, [1.0, 2.0, 3.0]))[:3, :3]
        vertices_wp = points_to_warp(np.asarray(mesh_tm.vertices) @ rotation_np.T, mesh_wp.device)
    # A spacing of its own, coarser than the derived one: the open arms run the dense lattice twice.
    voxel_size = float(mesh_tm.scale) / 48.0
    dense_vertices_wp, dense_faces_wp = od.levelset.offset_mesh(
        vertices_wp, faces_wp, distance, voxel_size
    )

    calls: list[int] = []
    sparse = od.levelset.sparse_marching_cubes

    def counted(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return sparse(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(od.levelset, "_SPARSE_LEVEL_SET_FROM_NODES", 0)
    monkeypatch.setattr(od.levelset, "sparse_marching_cubes", counted)
    sparse_vertices_wp, sparse_faces_wp = od.levelset.offset_mesh(
        vertices_wp, faces_wp, distance, voxel_size
    )
    assert len(calls) == int(sparse_expected)

    assert dense_vertices_wp.size > 100, "the offset has a surface to compare"
    assert sparse_vertices_wp.size == dense_vertices_wp.size
    gap_np, to_dense_np = map(
        np.asarray, cKDTree(dense_vertices_wp.numpy()).query(sparse_vertices_wp.numpy())
    )
    assert gap_np.max() < 1e-5 * float(mesh_tm.scale)
    assert np.unique(to_dense_np).shape[0] == to_dense_np.shape[0]
    assert_unordered_rows_equal(
        canonical_winding(to_dense_np[sparse_faces_wp.numpy().reshape(-1, 3)]),
        canonical_winding(dense_faces_wp.numpy().reshape(-1, 3)),
    )


def test_offset_mesh_guards(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a library comparison: the three documented value guards."""
    _mesh_tm, mesh_wp = icosphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    with pytest.raises(ValueError, match="non-zero"):
        od.levelset.offset_mesh(vertices_wp, faces_wp, 0.0)
    with pytest.raises(ValueError, match="voxel_size must be positive"):
        od.levelset.offset_mesh(vertices_wp, faces_wp, 0.1, -1.0)
    with pytest.raises(ValueError, match="at least one face"):
        od.levelset.offset_mesh(vertices_wp, warp_empty(0, wp.int32, mesh_wp.device), 0.1)


@pytest.mark.parametrize("iso", [0.0, 0.05, -0.05])
def test_signed_distance_level_set_is_the_dense_field_on_an_anisotropic_lattice(
    icosphere: tuple[tm.Trimesh, wp.Mesh], monkeypatch: pytest.MonkeyPatch, iso: float
) -> None:
    """
    Ordito against ordito: both extractions equal ``marching_cubes`` of the sampled field.

    The dense path is checked against composing the public parts by hand (sample the lattice with
    ``grid_points``, sign it with ``signed_distance_on_mesh``, march it), and the sparse path --
    forced by lowering the node gate -- against the dense one, on a lattice whose spacing differs
    per axis, the shape ``reconstruction.resample_uniform`` builds. Vertices are matched by nearest
    neighbour with a bijection check and faces compared with their winding.
    """
    _mesh_tm, mesh_wp = icosphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    shape = (41, 29, 53)
    bounds = (wp.vec3(-1.3, -1.25, -1.35), wp.vec3(1.3, 1.25, 1.35))

    samples_wp = od.voxels.grid_points(shape, bounds=bounds, device=mesh_wp.device)
    field_wp = od.proximity.signed_distance_on_mesh(
        vertices_wp, faces_wp, samples_wp, sign_mode="winding"
    )
    composed_vertices_wp, composed_faces_wp = od.levelset.marching_cubes(
        odt.as_array3d(field_wp.reshape(shape), wp.float32), iso, bounds=bounds
    )
    dense_vertices_wp, dense_faces_wp = od.levelset.signed_distance_level_set(
        vertices_wp, faces_wp, iso, shape, bounds=bounds
    )
    assert np.array_equal(dense_vertices_wp.numpy(), composed_vertices_wp.numpy())
    assert np.array_equal(dense_faces_wp.numpy(), composed_faces_wp.numpy())

    monkeypatch.setattr(od.levelset, "_SPARSE_LEVEL_SET_FROM_NODES", 0)
    sparse_vertices_wp, sparse_faces_wp = od.levelset.signed_distance_level_set(
        vertices_wp, faces_wp, iso, shape, bounds=bounds
    )
    assert dense_vertices_wp.size > 100, "the level set has a surface to compare"
    assert sparse_vertices_wp.size == dense_vertices_wp.size
    gap_np, to_dense_np = map(
        np.asarray, cKDTree(dense_vertices_wp.numpy()).query(sparse_vertices_wp.numpy())
    )
    assert gap_np.max() < 1e-5
    assert np.unique(to_dense_np).shape[0] == to_dense_np.shape[0]
    assert_unordered_rows_equal(
        canonical_winding(to_dense_np[sparse_faces_wp.numpy().reshape(-1, 3)]),
        canonical_winding(dense_faces_wp.numpy().reshape(-1, 3)),
    )


def test_signed_distance_level_set_guards(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a library comparison: the two documented value guards."""
    _mesh_tm, mesh_wp = icosphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    bounds = (wp.vec3(-1.5, -1.5, -1.5), wp.vec3(1.5, 1.5, 1.5))
    with pytest.raises(ValueError, match="at least one face"):
        od.levelset.signed_distance_level_set(
            vertices_wp, warp_empty(0, wp.int32, mesh_wp.device), 0.0, (8, 8, 8), bounds=bounds
        )
    with pytest.raises(ValueError, match="at least 2"):
        od.levelset.signed_distance_level_set(vertices_wp, faces_wp, 0.0, (8, 1, 8), bounds=bounds)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus", "icosphere_coarse", "unit_box"])
def test_thicken_mesh_closes_into_a_solid(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Not a library comparison: the shell is a valid solid, on open and closed inputs alike.

    Four claims, and the first two are what a thickening is *for*: the result is **watertight** and
    **consistently wound**, so it can be measured, printed or booleaned. On an open input that
    depends entirely on the band -- the two layers alone leave two rims -- and the band's winding is
    inherited from ``oriented_boundary_edges`` rather than guessed, which is why it comes out right
    on ``half_torus``'s *two* loops as well as ``hemisphere``'s one.

    The counts are exact and asserted: ``2 * n_vertices`` positions and
    ``2 * n_faces + 2 * n_boundary_edges`` triangles. And the volume is positive and close to
    ``area * thickness`` -- 0.289 against 0.308 on ``hemisphere``, the 6 % being the inward layer's
    smaller area -- which is the check that catches a shell built inside out, where every other
    assertion here still passes.
    """
    thickness = 0.05
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    n_vertices = vertices_wp.size
    n_faces = faces_wp.size // 3
    n_rim = int(od.boundary.oriented_boundary_edges(vertices_wp, faces_wp).shape[0])

    shell_vertices_wp, shell_faces_wp = od.levelset.thicken_mesh(vertices_wp, faces_wp, thickness)

    assert shell_vertices_wp.size == 2 * n_vertices
    assert shell_faces_wp.size // 3 == 2 * n_faces + 2 * n_rim
    assert od.validation.is_watertight(shell_vertices_wp, shell_faces_wp)
    assert od.validation.is_winding_consistent(shell_faces_wp)
    assert od.validation.is_edge_manifold(shell_faces_wp, False)

    volume = float(od.measures.volume(shell_vertices_wp, shell_faces_wp))
    assert volume > 0.0
    assert volume < mesh_tm.area * thickness  # the inward layer has the smaller area
    assert volume > 0.5 * mesh_tm.area * thickness


@pytest.mark.parity("thicken_mesh", "meshlib")
def test_thicken_mesh_matches_meshlib(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B: the same shell as ``makeThickMesh``, under a nearest-neighbour vertex bijection.

    MeshLib's thickening is the same construction rather than a different one -- extrude along the
    vertex normals, reverse a copy, band the rim -- and the outputs agree to a degree worth stating:
    identical counts (194 vertices, 384 faces), volumes equal to five decimals (0.28940), a
    **bijective** nearest-neighbour vertex match at **4.6e-05**, and *identical face sets* once the
    windings are canonicalized and MeshLib's vertex ids are mapped through that bijection.

    The transform is the bijection, which is what makes this Class B rather than A: neither library
    promises a vertex order. The 4.6e-05 residual is 0.09 % of the thickness and is the float32
    normal normalization, not a difference of rule.

    ``ThickenParams`` splits the displacement into ``insideOffset`` and ``outsideOffset``, so the
    single-sided default here is ``(thickness, 0)`` -- passed explicitly, since it is the parameter
    that would otherwise decide the comparison.
    """
    thickness = 0.05
    mesh_tm, mesh_wp = hemisphere
    shell_vertices_wp, shell_faces_wp = od.levelset.thicken_mesh(
        mesh_wp.points, mesh_wp.indices, thickness
    )

    parameters_ml = mm.ThickenParams()
    parameters_ml.insideOffset = thickness
    parameters_ml.outsideOffset = 0.0
    shell_ml = meshlib_to_trimesh(mm.makeThickMesh(trimesh_to_meshlib(mesh_tm), parameters_ml))
    assert shell_ml.faces.shape[0] > 0  # non-vacuity: the reference produced a shell

    assert shell_ml.vertices.shape[0] == shell_vertices_wp.size
    assert shell_ml.faces.shape[0] == shell_faces_wp.size // 3
    assert np.isclose(
        float(od.measures.volume(shell_vertices_wp, shell_faces_wp)),
        shell_ml.volume,
        rtol=1e-4,
        atol=1e-6,
    )

    # The bijection, then the face sets through it.
    shell_np = shell_vertices_wp.numpy().astype(np.float64)
    distance_np, match_np = map(np.asarray, cKDTree(np.asarray(shell_ml.vertices)).query(shell_np))
    assert distance_np.max() < 1e-4
    assert len(set(match_np.tolist())) == match_np.size
    inverse_np = np.empty(shell_ml.vertices.shape[0], dtype=np.int64)
    inverse_np[match_np] = np.arange(len(shell_np))
    assert_unordered_rows_equal(
        canonical_winding(shell_faces_wp.numpy().reshape(-1, 3)),
        canonical_winding(inverse_np[np.asarray(shell_ml.faces)]),
    )


def test_thicken_mesh_self_intersects_past_the_curvature_radius(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: the documented failure mode, asserted rather than left as prose.

    Displacing along vertex normals folds the surface wherever the thickness exceeds the local
    radius of curvature, and this function deliberately does not guard against it: the guard would
    be a whole-mesh intersection test on every call. So the contract is that the condition is
    *detectable*, and that is what is checked: ``half_torus``'s tube has minor radius 0.5 before
    its graded scaling, and thickening it by 0.6 makes the inward layer pass through the tube's own
    axis and out the other side. ``face_self_intersecting_mask`` flags **130** faces there and
    ``is_watertight`` -- which includes a self-intersection test, as open3d's does -- turns
    ``False``, where at 0.05 both are clean. It scales as the geometry says it should: 273 faces at
    a thickness of 1.0 and 467 at 1.5.

    A *closed* input does not fold this way, and that is worth recording because it is the obvious
    thing to test and it does not work: a unit sphere thickened by 1.5 puts its inward layer at
    radius 0.5 with the orientation inverted, which is two nested spheres -- wrong volume, no
    intersection. The tube is the shape whose normals actually converge.

    The alternative for such a thickness is named in the docstring and exercised here: a level-set
    ``offset_mesh`` cannot self-intersect by construction, and does not.
    """
    thickness = 0.6
    _, mesh_wp = half_torus
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices

    thin_vertices_wp, thin_faces_wp = od.levelset.thicken_mesh(vertices_wp, faces_wp, 0.05)
    thin_np = od.validation.face_self_intersecting_mask(thin_vertices_wp, thin_faces_wp).numpy()
    assert int(thin_np.sum()) == 0
    assert od.validation.is_watertight(thin_vertices_wp, thin_faces_wp)

    folded_vertices_wp, folded_faces_wp = od.levelset.thicken_mesh(vertices_wp, faces_wp, thickness)
    folded_np = od.validation.face_self_intersecting_mask(
        folded_vertices_wp, folded_faces_wp
    ).numpy()
    assert int(folded_np.sum()) > 100
    assert not od.validation.is_watertight(folded_vertices_wp, folded_faces_wp)

    # The recommended alternative at the same distance, and it comes out clean.
    inward_vertices_wp, inward_faces_wp = od.levelset.offset_mesh(vertices_wp, faces_wp, -thickness)
    if inward_faces_wp.size > 0:
        assert (
            int(
                od.validation.face_self_intersecting_mask(inward_vertices_wp, inward_faces_wp)
                .numpy()
                .sum()
            )
            == 0
        )


def test_thicken_mesh_guards(icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a library comparison: the three documented value guards."""
    _, mesh_wp = icosphere_coarse
    with pytest.raises(ValueError, match="thickness must be positive"):
        od.levelset.thicken_mesh(mesh_wp.points, mesh_wp.indices, 0.0)
    with pytest.raises(ValueError, match="outside must be non-negative"):
        od.levelset.thicken_mesh(mesh_wp.points, mesh_wp.indices, 0.1, outside=-1.0)
    with pytest.raises(ValueError, match="at least one face"):
        od.levelset.thicken_mesh(mesh_wp.points, warp_empty(0, wp.int32, mesh_wp.device), 0.1)
