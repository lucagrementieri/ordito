"""
Regression tests for ``ordito.levelset``.

The level-set offset against pymeshlab's uniform resampler and MeshLib's ``offsetMesh``, both of
which march the same kind of field -- plus the invariant that is stronger than either comparison:
every output vertex must sit at the requested signed distance from the input.
"""

from __future__ import annotations

import math
from typing import cast

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
from warp.geometry import IsoSurfaceMarchingCubes

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
    numpy_to_warp,
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


@pytest.mark.parametrize("case", ["torus", "anisotropic", "ties", "nan"])
def test_marching_cubes_is_warps_extraction_bit_for_bit(device: str, case: str) -> None:
    """
    Class A: byte-identical to ``warp.geometry.IsoSurfaceMarchingCubes.extract``, buffer order too.

    ``marching_cubes`` ports Warp's dense extraction with its own bookkeeping, and promises the same
    vertex numbering, triangle order and interpolation arithmetic. The arms reach what a port gets
    wrong: an anisotropic lattice with world bounds (the per-axis spacing), values quantized onto
    ``iso`` (the ``>=`` side of every comparison), and NaN and infinite samples, where Warp's case
    code reads NaN as below ``iso`` while its edge test never crosses -- so a face names an edge
    with no vertex, and gets ``-1``.
    """
    rng = np.random.default_rng(7)
    bounds = None
    iso = 0.0
    if case == "torus":
        axis_np = np.linspace(-1.1, 1.1, 48)
        x_np, y_np, z_np = np.meshgrid(axis_np, axis_np, axis_np, indexing="ij")
        field_np = np.sqrt((np.sqrt(x_np**2 + y_np**2) - 0.65) ** 2 + z_np**2) - 0.28
        bounds = (wp.vec3(-1.1, -1.1, -1.1), wp.vec3(1.1, 1.1, 1.1))
    elif case == "anisotropic":
        field_np = rng.standard_normal((17, 23, 31))
        bounds = (wp.vec3(0.3, -2.0, 5.0), wp.vec3(1.7, 4.0, 5.5))
    elif case == "ties":
        field_np = np.round(rng.standard_normal((20, 20, 20)) * 2.0) / 2.0
        iso = 0.5
    else:
        field_np = rng.standard_normal((16, 16, 16))
        field_np[3:5, 4, 7] = np.nan
        field_np[10, 10, 10] = np.inf
    field_wp = odt.as_array3d(
        wp.array(field_np.astype(np.float32), dtype=wp.float32, device=device), wp.float32
    )
    lower, upper = bounds if bounds is not None else (None, None)
    # Warp annotates the field ``wp.array3d``, a static helper no runtime array is typed as.
    vertices_warp, faces_warp = IsoSurfaceMarchingCubes.extract(
        cast("wp.array3d[wp.float32]", field_wp), iso, lower=lower, upper=upper
    )
    vertices_wp, faces_wp = od.levelset.marching_cubes(field_wp, iso, bounds=bounds)
    assert faces_warp.size > 0
    assert np.array_equal(
        vertices_wp.numpy().view(np.uint32), vertices_warp.numpy().view(np.uint32)
    )
    assert np.array_equal(faces_wp.numpy(), faces_warp.numpy())
    if case == "nan":
        assert (faces_wp.numpy() == -1).any(), "the NaN arm never reached an edge with no vertex"


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
    to zero (every fixture here is far below the shipped one) and closing the winding lattice's,
    which would otherwise take every winding-signed lattice this size. On a closed, consistently
    wound input the two must agree vertex for vertex up to the order of the buffers -- matched by
    nearest neighbour with a bijection check -- and face for face with the same winding. On an open
    or non-orientable input the gate must keep the dense lattice, whose winding-signed field jumps
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

    monkeypatch.setattr(od.levelset, "_LATTICE_WINDING_BELOW_NODES", 0)
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
    ``grid_points``, sign it with ``signed_distance_on_mesh``, march it), bit for bit, and the
    sparse path -- forced by lowering its node gate and closing the winding lattice's -- against the
    dense one, on a lattice whose spacing differs
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

    monkeypatch.setattr(od.levelset, "_LATTICE_WINDING_BELOW_NODES", 0)
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


# Every topology the winding lattice treats differently: closed and convex, closed with flat faces
# lying on lattice planes (``unit_box``: the lattice starts ``pad`` cells below its box), a hollow
# shell, open with one rim and with two (``half_torus`` is cut on a lattice plane), non-orientable
# closed and with a boundary, and closed but self-intersecting.
_WINDING_LATTICE_MESHES = [
    "icosahedron",
    "icosphere_coarse",
    "unit_box",
    "cave_cube",
    "hemisphere",
    "half_torus",
    "boy_surface",
    "mobius",
    "bohemian_dome",
]


def _sampled_level_set(
    monkeypatch: pytest.MonkeyPatch,
    vertices_wp: wp.array[wp.vec3],
    faces_wp: wp.array[wp.int32],
    iso: float,
    shape: tuple[int, int, int],
    bounds: tuple[wp.vec3, wp.vec3],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Extract the level set by sampling every lattice node, the winding lattice switched off."""
    with monkeypatch.context() as patch:
        patch.setattr(od.levelset, "_LATTICE_WINDING_BELOW_NODES", 0)
        patch.setattr(od.levelset, "_SPARSE_LEVEL_SET_FROM_NODES", 1 << 62)
        return od.levelset.signed_distance_level_set(
            vertices_wp, faces_wp, iso, shape, bounds=bounds
        )


@pytest.mark.parametrize(
    ("settle", "constant", "value"),
    [
        pytest.param("shipped", None, None, id="shipped"),
        # Every node in doubt goes straight to Warp's own sign, building its solid-angle BVH.
        pytest.param("warp_sign", "_EXACT_WINDING_CAPACITY", 0, id="warp_sign"),
        # A margin of nearly 1/2 puts most of the lattice in doubt, through both settling stages.
        pytest.param("wide_margin", "_WINDING_UNDECIDED_DELTA", 0.45, id="wide_margin"),
        # No room for any cone: an open input falls back to sampling every node.
        pytest.param("no_cones", "_MAX_CONE_EDGES", -1, id="no_cones"),
    ],
)
@pytest.mark.parametrize("mesh_name", _WINDING_LATTICE_MESHES)
def test_signed_distance_level_set_winding_lattice_is_the_sampled_lattice(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    mesh_name: str,
    settle: str,
    constant: str | None,
    value: float | None,
) -> None:
    """
    Ordito against ordito: the winding lattice extracts the sampled lattice's surface, bit for bit.

    The sampled path (``grid_points`` signed by ``signed_distance_on_mesh(sign_mode="winding")``,
    marched; pinned to composing those parts by hand in
    ``test_signed_distance_level_set_is_the_dense_field_on_an_anisotropic_lattice``) carries the
    oracle, ``offset_mesh``'s meshlib and pymeshlab comparisons above. Here the winding lattice --
    a band-capped closest-point search signed by the exact winding number counted along lattice
    columns, Warp's own sign only where the two could differ -- must reproduce it exactly: the
    same vertex and face buffers, at an outward, a zero and an inward level, on every topology the
    method treats differently, and with each settling stage forced in turn.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    if constant is not None:
        monkeypatch.setattr(od.levelset, constant, value)
    spacing = float(mesh_tm.scale) / 40.0
    compared = 0
    for cells in (2.0, 0.0, -1.0):
        iso = cells * spacing
        pad = 2 + max(math.ceil(cells), 0)
        shape, bounds = od.proximity.signed_distance_lattice(
            vertices_wp, spacing, bounds=od.bounds.aabb(vertices_wp), pad=pad
        )
        lattice_vertices_wp, lattice_faces_wp = od.levelset.signed_distance_level_set(
            vertices_wp, faces_wp, iso, shape, bounds=bounds
        )
        sampled_vertices_wp, sampled_faces_wp = _sampled_level_set(
            monkeypatch, vertices_wp, faces_wp, iso, shape, bounds
        )
        assert np.array_equal(lattice_vertices_wp.numpy(), sampled_vertices_wp.numpy()), (
            settle,
            cells,
        )
        assert np.array_equal(lattice_faces_wp.numpy(), sampled_faces_wp.numpy()), (settle, cells)
        compared += int(sampled_faces_wp.size > 0)
    assert compared >= 2, "the comparison has surfaces to compare"


@pytest.mark.parametrize("opening", [0, 1, 2])
def test_signed_distance_level_set_winding_lattice_with_a_rim_on_a_lattice_plane(
    device: str, monkeypatch: pytest.MonkeyPatch, opening: int
) -> None:
    """
    Ordito against ordito: a hole whose rim lies on a lattice plane, opening along each axis.

    The sampled lattice carries the oracle, as in
    ``test_signed_distance_level_set_winding_lattice_is_the_sampled_lattice``. The cone closing the
    hole is then flat and lies on a plane of lattice nodes: opening along ``z`` the columns cross
    it, opening along ``x`` or ``y`` it is edge-on to the columns of at least one direction, and
    opening along ``y`` to both, so no rasterised crossing can mark its nodes and only the
    on-cone test does (removing it fails the ``y`` arm).
    """
    sphere_tm = tm.creation.icosphere(subdivisions=2, radius=1.0)
    hemisphere_tm = sphere_tm.slice_plane(
        plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]), cap=False
    )
    hemisphere_tm.merge_vertices()
    permutation = {0: [2, 0, 1], 1: [0, 2, 1], 2: [0, 1, 2]}[opening]
    vertices_np = np.asarray(hemisphere_tm.vertices)[:, permutation]
    faces_np = np.asarray(hemisphere_tm.faces)
    if np.linalg.det(np.eye(3)[permutation]) < 0.0:
        faces_np = faces_np[:, ::-1]
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np.ravel(), device)
    spacing = 2.0 / 40.0
    for cells in (2.0, 0.0, -1.0):
        iso = cells * spacing
        shape, bounds = od.proximity.signed_distance_lattice(
            vertices_wp, spacing, bounds=od.bounds.aabb(vertices_wp), pad=2 + max(int(cells), 0)
        )
        lattice_vertices_wp, lattice_faces_wp = od.levelset.signed_distance_level_set(
            vertices_wp, faces_wp, iso, shape, bounds=bounds
        )
        sampled_vertices_wp, sampled_faces_wp = _sampled_level_set(
            monkeypatch, vertices_wp, faces_wp, iso, shape, bounds
        )
        assert sampled_faces_wp.size > 0
        assert np.array_equal(lattice_vertices_wp.numpy(), sampled_vertices_wp.numpy()), cells
        assert np.array_equal(lattice_faces_wp.numpy(), sampled_faces_wp.numpy()), cells


def test_signed_distance_level_set_re_queries_capped_crossing_endpoints(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: a capped node that ends a crossing edge gets its exact distance back.

    On an open surface the winding-signed field changes sign away from the surface (across the
    hemisphere's opening), so an edge marching cubes interpolates across can end at a node farther
    than the band the closest-point search is capped at. The sampled field
    (``signed_distance_on_mesh``) carries the oracle: at every endpoint of every crossing edge the
    winding lattice's field must equal it bit for bit, and some of those endpoints must lie past
    the cap, so the re-query is what this exercises.
    """
    mesh_tm, mesh_wp = hemisphere
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    spacing = float(mesh_tm.scale) / 40.0
    shape, bounds = od.proximity.signed_distance_lattice(
        vertices_wp, spacing, bounds=od.bounds.aabb(vertices_wp), pad=2
    )
    iso = 0.0
    field_wp = od.levelset._winding_band_field(vertices_wp, faces_wp, iso, shape, bounds)  # pyright: ignore[reportPrivateUsage]
    assert field_wp is not None
    samples_wp = od.voxels.grid_points(shape, bounds=bounds, device=mesh_wp.device)
    sampled_np = (
        od.proximity.signed_distance_on_mesh(vertices_wp, faces_wp, samples_wp, sign_mode="winding")
        .numpy()
        .reshape(shape)
    )
    lattice_np = field_wp.numpy()
    endpoints_np = np.zeros(shape, dtype=bool)
    above_np = sampled_np >= iso
    for axis in range(3):
        crossing_np = np.diff(above_np.astype(np.int8), axis=axis) != 0
        lower_np = [slice(None)] * 3
        upper_np = [slice(None)] * 3
        lower_np[axis] = slice(0, -1)
        upper_np[axis] = slice(1, None)
        endpoints_np[tuple(lower_np)] |= crossing_np
        endpoints_np[tuple(upper_np)] |= crossing_np
    assert np.array_equal(lattice_np[endpoints_np], sampled_np[endpoints_np])
    diagonal = math.sqrt(3.0) * spacing
    assert (np.abs(sampled_np[endpoints_np]) > 1.5 * diagonal).sum() > 0, "a capped endpoint"


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus", "mobius"])
def test_boundary_chain_is_the_net_boundary(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Not a library comparison: no reference returns a mesh's boundary as a 1-chain with multiplicity.

    The chain is checked against its definition, summed on the host: per undirected edge, the net
    number of its halfedges running from the lower vertex to the higher, emitted that many times in
    that direction. ``mobius`` is the non-orientable case whose seam halfedges run the same way and
    count twice; a flipped face on the closed ``icosahedron`` adds three such edges where there
    were none; a third face on one edge of ``hemisphere`` makes a run of three halfedges.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = mesh_wp.indices.numpy().reshape(-1, 3)
    if mesh_name == "icosahedron":
        faces_np = faces_np.copy()
        faces_np[0] = faces_np[0, ::-1]
    if mesh_name == "hemisphere":
        faces_np = np.vstack([faces_np, [[faces_np[0, 1], faces_np[0, 0], faces_np[1, 2]]]])
    faces_wp = wp.array(faces_np.ravel(), dtype=wp.int32, device=mesh_wp.device)
    n_vertices = int(mesh_wp.points.size)
    chain_np = od.levelset._boundary_chain(faces_wp, n_vertices).numpy()  # pyright: ignore[reportPrivateUsage]

    tails_np = faces_np.ravel()
    heads_np = np.roll(faces_np, -1, axis=1).ravel()
    low_np = np.minimum(tails_np, heads_np)
    high_np = np.maximum(tails_np, heads_np)
    net_np: dict[tuple[int, int], int] = {}
    for low, high, tail in zip(low_np.tolist(), high_np.tolist(), tails_np.tolist(), strict=True):
        if low != high:
            net_np[(low, high)] = net_np.get((low, high), 0) + (1 if tail == low else -1)
    expected_np = np.array(
        [
            (low, high) if net > 0 else (high, low)
            for (low, high), net in net_np.items()
            for _ in range(abs(net))
        ],
        dtype=np.int32,
    ).reshape(-1, 2)
    assert expected_np.shape[0] > 0, "the chain is not empty"
    assert_unordered_rows_equal(chain_np, expected_np)


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
