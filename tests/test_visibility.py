"""Regression tests for ``ordito.visibility`` against pymeshlab (CPU reference)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import trimesh.proximity as tm_proximity
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import ordito as od
from tests.comparisons import assert_nonconstant
from tests.conversions import (
    meshlib_scalars_to_numpy,
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_pymeshlab,
    trimesh_to_warp,
    warp_empty,
)

if TYPE_CHECKING:
    from typing_extensions import Buffer


@pytest.fixture
def ellipsoid(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Build a non-uniformly scaled ``icosphere(3)``: closed, curved, of *varying* thickness.

    The thickness fixtures need a mesh whose answer is not a constant -- on a sphere every inward
    ray reads ``2 * radius`` and a comparison would pass against any implementation that returned
    the diameter. Scaled, the ray thickness spans 1.31 to 2.80.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    mesh_tm.apply_scale([1.0, 1.4, 0.7])
    return mesh_tm, trimesh_to_warp(mesh_tm, device)


def _vertex_normals_wp(mesh_wp: wp.Mesh) -> wp.array[wp.vec3]:
    """Smooth outward normals; face normals would make the field piecewise constant per ring."""
    return od.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices)


@pytest.mark.parametrize("weight", ["cosine", "uniform"])
@pytest.mark.parametrize("mesh_name", ["icosahedron", "torus"])
def test_ambient_occlusion_is_zero_on_a_convex_mesh(
    request: pytest.FixtureRequest, mesh_name: str, weight: od.visibility.RayWeight
) -> None:
    """
    No ray leaving a convex closed surface can come back, so the occlusion is *exactly* zero.

    The one analytic check this quantity has, and the reason ``torus`` is here too — it is not
    convex, so it must *not* read zero, which is what makes the icosahedron result meaningful
    rather than a stuck kernel. ``volumetric_obscurance`` attenuates the same occluders, so it
    reads exactly zero on the convex mesh too, and non-zero on the torus.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    normals_wp = _vertex_normals_wp(mesh_wp)
    occlusion_np = od.visibility.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=64, weight=weight
    ).numpy()
    obscurance_np = od.visibility.volumetric_obscurance(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=64, weight=weight
    ).numpy()
    assert (occlusion_np >= 0.0).all()
    assert (occlusion_np <= 1.0).all()
    if mesh_name == "icosahedron":
        assert occlusion_np.max() == 0.0
        assert obscurance_np.max() == 0.0
    else:
        assert occlusion_np.max() > 0.1
        assert obscurance_np.max() > 0.0


def test_ambient_occlusion_finds_the_cavity(cave_cube: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """The inner shell of a hollow cube must be far more occluded than the outer one."""
    mesh_tm, mesh_wp = cave_cube
    occlusion_np = od.visibility.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=_vertex_normals_wp(mesh_wp), n_rays=128
    ).numpy()
    # The cavity is the inner 0.1-cube; its vertices are the ones near the origin.
    inner_np = np.linalg.norm(np.asarray(mesh_tm.vertices), axis=1) < 0.2
    assert inner_np.any()
    assert (~inner_np).any()
    assert occlusion_np[inner_np].mean() > 0.5
    assert occlusion_np[~inner_np].mean() < 0.1


@pytest.mark.parity("ambient_occlusion", "pymeshlab")
def test_ambient_occlusion_ranks_like_pymeshlab(torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C (rank correlation): MeshLab's scalar is the unnormalized *complement* of this one.

    ``compute_scalar_ambient_occlusion`` sums ``cos`` over the visible directions of its own
    whole-sphere set without dividing, so a fully exposed vertex reads ~``rays / 4`` rather than
    ``1``, and its direction set is not this one. What must hold is that both call the same vertices
    the occluded ones — hence a rank correlation over a torus, whose hole occludes half of its inner
    wall.

    **Mutation probe**, the one section 7.4 requires of a Class C threshold and this test did not
    carry: shuffling the ordito side, 30 permutations on this fixture, gives ``|r|`` at most
    **0.0695** (mean 0.0269) against the measured **-0.8621** -- a separation of **12.4x**, well
    past the 3x floor, so the threshold is testing the correspondence and not the two marginal
    distributions.
    """
    mesh_tm, mesh_wp = torus
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_scalar_ambient_occlusion(rays=256)
    exposure_pml = meshset_pml.current_mesh().vertex_scalar_array()

    occlusion_np = od.visibility.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=_vertex_normals_wp(mesh_wp), n_rays=256
    ).numpy()

    # Spearman: rank both sides, then correlate. Exposure is the complement of occlusion, so the
    # correlation must be strongly *negative*.
    rank_pml = np.argsort(np.argsort(exposure_pml))
    rank_wp = np.argsort(np.argsort(occlusion_np))
    assert np.corrcoef(rank_pml, rank_wp)[0, 1] < -0.8


@pytest.mark.parity(
    "ambient_occlusion",
    "meshlib",
    benchmarked=False,
    reason="computeSkyViewFactor takes one world-space patch set for *all* samples, so the two "
    "quantities coincide only where every sample shares a hemisphere -- a terrain, which is what "
    "this test builds. The benchmark group's input is a mesh's own vertices with a per-vertex "
    "tangent frame, where a single patch set measures a different integral at every vertex but the "
    "first. pymeshlab carries the timed row.",
)
def test_ambient_occlusion_matches_meshlib_sky_view_factor(device: str) -> None:
    """
    Class C (a quadrature statistic): the sky-view factor is one minus the occlusion.

    MeshLib solves the terrain form of this integral -- how much of the sky each sample point can
    see -- and it is the same quantity ``ambient_occlusion`` reports as the blocked share, so the
    named part of the comparison is the complement ``svf = 1 - occlusion``. What keeps it Class C
    rather than Class B is that the two integrate over *different direction sets*: ordito rotates
    its own Fibonacci lattice into each point's frame and MeshLib takes the patch list it is given,
    so the residual is quadrature error and not a correspondence.

    Two things the scene has to arrange, and both are why this is not simply a row on the group. The
    patch set is **global**, so every sample must share one hemisphere: the samples sit just above a
    flat plate, where the normal is exactly ``+z``. And ``sampleHalfSphere()`` is *not* a hemisphere
    -- measured, its 145 directions span ``z`` from -1 to +1 and only 72 have ``z > 0`` -- so
    feeding it directly makes the sky-view factor read half of what it should (0.52 against 0.98 at
    the open corner). The patches are therefore an equal-area hemisphere lattice built here, with
    equal radiation, which is what makes MeshLib's weighted mean an isotropic one.

    Measured at 256 patches against 256 rays: **mean |difference| 0.0035, maximum 0.0195, Pearson
    0.99985** over 441 samples spanning 0 to 0.98. The bound below is 3x the measured maximum. The
    mutation probe: shuffling one side gives mean |difference| **0.235** (67x) and correlation
    -0.02, so the agreement is a correspondence rather than two similar marginal distributions.
    """
    n_patches = 256
    ground = tm.creation.box(extents=(8.0, 8.0, 0.2))
    ground.apply_translation([0.0, 0.0, -0.1])
    dome = tm.creation.icosphere(subdivisions=3, radius=1.2)
    dome.apply_translation([0.0, 0.0, 0.3])
    terrain_tm = tm.util.concatenate([ground, dome])
    assert isinstance(terrain_tm, tm.Trimesh)

    grid = np.stack(np.meshgrid(np.linspace(-3.0, 3.0, 21), np.linspace(-3.0, 3.0, 21)), axis=-1)
    samples_np = np.ascontiguousarray(
        np.column_stack([grid.reshape(-1, 2), np.full(grid.size // 2, 1e-3)]), dtype=np.float32
    )

    # Equal-area hemisphere lattice: z uniform in (0, 1], azimuth by the golden angle.
    index = np.arange(n_patches) + 0.5
    z_np = index / n_patches
    radius_np = np.sqrt(np.maximum(0.0, 1.0 - z_np * z_np))
    azimuth_np = index * np.pi * (3.0 - np.sqrt(5.0))
    patches_np = np.column_stack(
        [radius_np * np.cos(azimuth_np), radius_np * np.sin(azimuth_np), z_np]
    )

    mesh_ml = trimesh_to_meshlib(terrain_tm)
    patches_ml = mm.std_vector_SkyPatch()
    for direction in patches_np:
        patch_ml = mm.SkyPatch()
        patch_ml.dir = mm.Vector3f(*direction.tolist())
        patch_ml.radiation = 1.0  # equal weight, so the weighted mean is an isotropic one
        patches_ml.append(patch_ml)
    valid_ml = mm.VertBitSet()
    valid_ml.resize(len(samples_np), True)
    sky_view_ml = meshlib_scalars_to_numpy(
        mm.computeSkyViewFactor(
            mesh_ml,
            # The stub names ``VertCoords``; pybind converts this vector implicitly.
            mn.fromNumpyArray(cast("Buffer", samples_np)),  # pyright: ignore[reportArgumentType]
            valid_ml,
            patches_ml,
        )
    )

    mesh_wp = trimesh_to_warp(terrain_tm, device)
    samples_wp = points_to_warp(samples_np, device)
    normals_wp = wp.array(
        np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(samples_np), 1)),
        dtype=wp.vec3,
        device=device,
    )
    occlusion_np = od.visibility.ambient_occlusion(
        mesh_wp, samples_wp, normals=normals_wp, n_rays=n_patches, weight="uniform"
    ).numpy()

    # Non-vacuity: the dome must actually occlude some samples and leave others open.
    assert sky_view_ml.min() < 0.2
    assert sky_view_ml.max() > 0.9
    difference = np.abs((1.0 - occlusion_np) - sky_view_ml)
    assert difference.mean() < 0.02  # 5.7x the measured 0.0035
    assert difference.max() < 0.06  # 3x the measured 0.0195
    assert np.corrcoef(1.0 - occlusion_np, sky_view_ml)[0, 1] > 0.99


def test_ambient_occlusion_converges_with_more_rays(torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Successive doublings of the ray count must move the field by less and less."""
    _mesh_tm, mesh_wp = torus
    normals_wp = _vertex_normals_wp(mesh_wp)
    fields = [
        od.visibility.ambient_occlusion(
            mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=n_rays
        ).numpy()
        for n_rays in (64, 256, 1024)
    ]
    assert np.abs(fields[2] - fields[1]).mean() < np.abs(fields[1] - fields[0]).mean()


def test_ambient_occlusion_uniform_weight_differs_from_cosine(
    torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """The two conventions are genuinely different integrals, not a rescaling of each other."""
    _mesh_tm, mesh_wp = torus
    normals_wp = _vertex_normals_wp(mesh_wp)
    cosine_np = od.visibility.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=256, weight="cosine"
    ).numpy()
    uniform_np = od.visibility.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=256, weight="uniform"
    ).numpy()
    assert not np.allclose(cosine_np, uniform_np, atol=1e-3)
    # Grazing directions are the ones the cosine weight discounts, and in a cavity they are the
    # blocked ones, so uniform weighting reports more occlusion on average.
    assert uniform_np.mean() > cosine_np.mean()


@pytest.mark.parametrize("weight", ["cosine", "uniform"])
@pytest.mark.parametrize("n_rays", [7, 64, 300])
def test_ambient_occlusion_point_major_matches_block_per_point(
    torus: tuple[tm.Trimesh, wp.Mesh],
    weight: Literal["cosine", "uniform"],
    n_rays: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ordito against ordito: the point-major layout against one block per point.

    A CUDA cloud past ``_OCCLUSION_POINT_MAJOR_FROM`` points is traced 32 Morton-neighbouring
    points to a block; forcing the threshold to zero must reproduce the default layout's field
    (the same rays and tests, summed in another lane order: equal to float32 rounding). The ray
    counts cover fewer rays than a block's four groups, a partial last group and several rays per
    lane. On the CPU device the threshold has no effect and both arms are the default path. The
    pymeshlab ranking test carries the oracle.
    """
    _mesh_tm, mesh_wp = torus
    normals_wp = _vertex_normals_wp(mesh_wp)
    default_np = od.visibility.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=n_rays, weight=weight
    ).numpy()
    monkeypatch.setattr(od.visibility, "_OCCLUSION_POINT_MAJOR_FROM", 0)
    point_major_np = od.visibility.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=n_rays, weight=weight
    ).numpy()
    assert np.ptp(default_np) > 0.1  # non-vacuity: the torus's hole occludes some points
    assert np.allclose(point_major_np, default_np, rtol=0.0, atol=1e-6)


def test_ambient_occlusion_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="n_rays >= 1"):
        od.visibility.ambient_occlusion(mesh_wp, mesh_wp.points, n_rays=0)
    with pytest.raises(ValueError, match="weight must be"):
        od.visibility.ambient_occlusion(mesh_wp, mesh_wp.points, weight="lambert")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="one entry per point"):
        od.visibility.ambient_occlusion(
            mesh_wp, mesh_wp.points, normals=wp.zeros(2, dtype=wp.vec3, device=mesh_wp.device)
        )


def test_ambient_occlusion_empty(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a parity assert: an empty ``points`` gives an empty field, and still checks ``normals``.

    The empty early return must not silently skip the ``normals`` length check: a caller passing a
    stale, wrong-length ``normals`` array alongside an empty ``points`` gets the ``ValueError`` the
    docstring promises unconditionally, not an empty result.
    """
    _mesh_tm, mesh_wp = icosahedron
    points_wp = wp.zeros(0, dtype=wp.vec3, device=mesh_wp.device)
    assert od.visibility.ambient_occlusion(mesh_wp, points_wp).shape == (0,)
    with pytest.raises(ValueError, match="one entry per point"):
        od.visibility.ambient_occlusion(
            mesh_wp, points_wp, normals=wp.zeros(2, dtype=wp.vec3, device=mesh_wp.device)
        )


def test_volumetric_obscurance_approaches_ambient_occlusion_as_tau_falls(
    torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Ambient occlusion is the ``tau -> 0`` limit, so the gap must shrink monotonically."""
    _mesh_tm, mesh_wp = torus
    normals_wp = _vertex_normals_wp(mesh_wp)
    occlusion_np = od.visibility.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128
    ).numpy()
    gaps = [
        np.abs(
            od.visibility.volumetric_obscurance(
                mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, tau=tau
            ).numpy()
            - occlusion_np
        ).mean()
        for tau in (1.0, 0.1, 0.01, 1e-4)
    ]
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 1e-3


def test_volumetric_obscurance_attenuates_distant_occluders(
    torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """A larger ``tau`` discounts every occluder, so the field can only go down."""
    _mesh_tm, mesh_wp = torus
    normals_wp = _vertex_normals_wp(mesh_wp)
    low_np = od.visibility.volumetric_obscurance(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, tau=0.1
    ).numpy()
    high_np = od.visibility.volumetric_obscurance(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, tau=10.0
    ).numpy()
    assert (high_np <= low_np + 1e-6).all()
    assert high_np.mean() < low_np.mean()


def test_volumetric_obscurance_ranks_like_pymeshlab(torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C (rank correlation), and the sign of the correlation is the claim.

    MeshLab reports *exposure* where ordito reports *obscurance*, so agreement means a correlation
    below **-0.8**, not above it -- an implementation that returned exposure would pass a
    ``|corr| > 0.8`` bar and fails this one. The torus is the fixture because its hole obscures the
    inner wall and leaves the outer wall exposed, giving the ranks something to disagree about.

    **Mutation probe**: shuffling the ordito side, 30 permutations on this fixture, gives ``|r|``
    at most **0.0777** (mean 0.0274) against the measured **-0.8651** -- a separation of **11.1x**,
    past section 7.4's 3x floor.
    """
    mesh_tm, mesh_wp = torus
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_scalar_by_volumetric_obscurance(rays=256, tau=0.1)
    exposure_pml = meshset_pml.current_mesh().vertex_scalar_array()

    obscurance_np = od.visibility.volumetric_obscurance(
        mesh_wp, mesh_wp.points, normals=_vertex_normals_wp(mesh_wp), n_rays=256, tau=0.1
    ).numpy()
    rank_pml = np.argsort(np.argsort(exposure_pml))
    rank_wp = np.argsort(np.argsort(obscurance_np))
    assert np.corrcoef(rank_pml, rank_wp)[0, 1] < -0.8


def test_volumetric_obscurance_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="tau must be positive"):
        od.visibility.volumetric_obscurance(mesh_wp, mesh_wp.points, tau=0.0)


# ---------------------------------------------------------------------------
# Shape diameter function (pymeshlab reference; analytic on a sphere)
# ---------------------------------------------------------------------------


def _sphere_wp(device: str, radius: float, subdivisions: int = 3):
    """
    Build an icosphere at a caller-chosen radius: ``(trimesh, wp.Mesh, vertex normals)``.

    Parametrized over ``radius`` -- which is the axis every occlusion test here needs and no fixture
    can carry -- so it stays a builder over ``trimesh_to_warp`` rather than becoming one.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh_wp = trimesh_to_warp(sphere_tm, device)
    normals_wp = od.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices)
    return sphere_tm, mesh_wp, normals_wp


@pytest.mark.parametrize("radius", [1.0, 3.0])
def test_shape_diameter_is_the_diameter_of_a_sphere(device: str, radius: float) -> None:
    """
    A cone through a sphere is bracketed analytically, at any radius — the exact check.

    A ray leaving the surface at angle ``theta`` from the inward normal crosses a chord of exactly
    ``2 R cos(theta)``, so every ray in a cone of half-angle ``alpha`` lands in
    ``[2 R cos(alpha), 2 R]`` and so does any weighted mean of them. Both ends are tight: widening
    the cone lowers the answer by exactly that factor, which is why this is a bracket rather than an
    ``allclose`` against ``2 R``.
    """
    _sphere_tm, mesh_wp, normals_wp = _sphere_wp(device, radius)
    cone_angle = np.deg2rad(5.0)
    diameter_np = od.visibility.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, cone_angle=cone_angle
    ).numpy()
    assert (diameter_np <= 2.0 * radius * (1.0 + 1e-4)).all()
    assert (diameter_np >= 2.0 * radius * np.cos(cone_angle) * (1.0 - 1e-4)).all()


def test_shape_diameter_reduces_to_thickness(device: str) -> None:
    """One ray down a vanishing cone *is* ``thickness(method="ray")``, to float32."""
    _sphere_tm, mesh_wp, normals_wp = _sphere_wp(device, 1.5)
    diameter_np = od.visibility.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=1, cone_angle=1e-4
    ).numpy()
    thickness_np = od.visibility.thickness(
        mesh_wp, mesh_wp.points, normals=normals_wp, method="ray"
    ).numpy()
    assert np.allclose(diameter_np, thickness_np, rtol=1e-5, atol=1e-5)


def test_shape_diameter_measures_a_slab(device: str) -> None:
    """On a 1 x 1 x 4 box the large faces are 1 apart, and the cone must say so."""
    box_tm = tm.creation.box(extents=[1.0, 1.0, 4.0]).subdivide().subdivide().subdivide()
    mesh_wp = trimesh_to_warp(box_tm, device)
    normals_wp = od.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices)
    diameter_np = od.visibility.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, cone_angle=np.deg2rad(10.0)
    ).numpy()

    # Vertices strictly *inside* one of the two x-walls: away from the y edges (where the smooth
    # normal is diagonal and the ray crosses the 1.41 diagonal instead) and away from the z ends.
    vertices_np = np.asarray(box_tm.vertices)
    on_wall_np = (
        (np.abs(np.abs(vertices_np[:, 0]) - 0.5) < 1e-6)
        & (np.abs(vertices_np[:, 1]) < 0.5 - 1e-6)
        & (np.abs(vertices_np[:, 2]) < 1.5)
    )
    assert on_wall_np.sum() > 10
    assert np.allclose(diameter_np[on_wall_np], 1.0, rtol=1e-2)


def test_shape_diameter_trimming_rejects_the_escaping_rays(device: str) -> None:
    """
    Ordito against ordito: trimming rejects the rays that escape through the cavity.

    On a hollow shell the untrimmed mean is dragged out by the rays that cross the whole cavity.
    This is what the outlier rejection is *for*, so it has to be visible: with ``trim`` wide open
    the inner-shell diameters inflate well past the shell's own thickness. Not a parity assert and
    not a Class C threshold -- there is no reference here, both fields come from
    ``shape_diameter`` at two settings of one keyword, and what is pinned is that the keyword does
    something in the direction it claims. ``pymeshlab`` carries the oracle for this group, in
    ``test_shape_diameter_agrees_with_pymeshlab_on_which_part_is_thinner`` below.
    """
    outer_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    inner_tm = tm.creation.icosphere(subdivisions=3, radius=0.8)
    inner_tm.invert()
    shell_tm = tm.util.concatenate([outer_tm, inner_tm])
    assert isinstance(shell_tm, tm.Trimesh)
    mesh_wp = trimesh_to_warp(shell_tm, device)
    normals_wp = od.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices)
    trimmed_np = od.visibility.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, trim=1.0
    ).numpy()
    untrimmed_np = od.visibility.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, trim=100.0
    ).numpy()
    assert trimmed_np.mean() < untrimmed_np.mean()
    # The shell is 0.2 thick; trimming has to keep the outer wall near that, untrimmed does not.
    outer_np = np.linalg.norm(np.asarray(shell_tm.vertices), axis=1) > 0.9
    assert trimmed_np[outer_np].mean() < 0.5
    assert untrimmed_np[outer_np].mean() > trimmed_np[outer_np].mean()


@pytest.mark.parity("shape_diameter", "pymeshlab")
def test_shape_diameter_agrees_with_pymeshlab_on_which_part_is_thinner(device: str) -> None:
    """
    Class C (rank structure): MeshLab's SDF differs from this one by roughly a constant factor.

    Its ``cone_amplitude`` parameter is a **no-op** in the 2025.07 build (byte-identical output at
    90 and 120 degrees) and its trimming is not the paper's, so neither a value comparison nor a
    per-vertex rank correlation is available. What both must agree on is the thing the field is
    *for*: on a dumbbell — two radius-1 balls joined by a radius-0.2 bar — the bar is thin and the
    balls are thick, which is the segmentation cue SDF exists to provide.

    Note this is *not* a test that either side reports the bar's diameter as 0.4. A cone of rays
    from a point on a slender bar mostly hits the bar's own walls, so SDF measures the local
    cross-section rather than any global extent — which is exactly why a 1 x 1 x 4 box reads ~1 at
    both ends and is useless as a fixture here.
    """
    ball_left_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    ball_right_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    ball_right_tm.apply_translation([4.0, 0.0, 0.0])
    bar_tm = tm.creation.cylinder(radius=0.2, height=5.0, sections=24)
    bar_tm.apply_transform(tm.transformations.rotation_matrix(np.pi / 2.0, [0.0, 1.0, 0.0]))
    bar_tm.apply_translation([2.0, 0.0, 0.0])
    # Subdivided after the union: the raw cylinder carries vertices only at its two end caps, which
    # the union buries inside the balls, leaving the bar's *surface* with nothing to measure on.
    dumbbell_tm = tm.boolean.union([ball_left_tm, ball_right_tm, bar_tm]).subdivide_to_size(0.25)

    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(dumbbell_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(dumbbell_tm.faces, dtype=np.int32),
        )
    )
    meshset_pml.compute_scalar_by_shape_diameter_function_per_vertex(rays=256)
    diameter_pml = meshset_pml.current_mesh().vertex_scalar_array()

    mesh_wp = trimesh_to_warp(dumbbell_tm, device)
    normals_wp = od.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices)
    diameter_np = od.visibility.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=256
    ).numpy()

    vertices_np = np.asarray(dumbbell_tm.vertices)
    bar_np = (np.abs(vertices_np[:, 0] - 2.0) < 0.8) & (
        np.linalg.norm(vertices_np[:, 1:], axis=1) < 0.3
    )
    ball_np = vertices_np[:, 0] < -0.3  # only the left ball reaches there
    assert bar_np.sum() > 10
    assert ball_np.sum() > 10
    # Both must call the bar the thinner part. The margin is loose because MeshLab compresses the
    # contrast: it reads a 0.68 bar-to-ball ratio here where this port reads 0.55.
    for field_np in (diameter_np, diameter_pml):
        assert field_np[bar_np].mean() < 0.8 * field_np[ball_np].mean()


@pytest.mark.parity(
    "shape_diameter",
    "meshlib",
    benchmarked=False,
    reason="computeRayThicknessAtVertices casts *one* ray where this group's rows cast 64 or 256, "
    "so a row here would price two different amounts of work under one name. It is timed in the "
    "thickness_at_vertices group instead, where it is the class-A pair for "
    "thickness(method='ray'); pymeshlab carries the timed row here.",
)
def test_shape_diameter_collapses_onto_meshlibs_single_ray(
    ellipsoid: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class C (a relative-error statistic): the cone, closed down, is MeshLib's one ray.

    MeshLib has no shape-diameter function -- ``computeRayThicknessAtVertices`` is a single inward
    ray per vertex -- so the comparable claim is the limit
    [`test_shape_diameter_reduces_to_thickness`] already checks against ordito itself: as
    ``cone_angle`` goes to zero the bundle collapses onto the inward normal. Running that limit
    against an outside implementation is what makes it a reference comparison, and it pins the ray
    *direction* and the trimming's neutrality at the same time.

    Measured on the ellipsoid at ``cone_angle=0.05``: **6.6e-04** median relative difference,
    Pearson **0.99998**, against values spanning 1.31 to 2.80. At the default 60-degree cone the two
    are **uncorrelated** (Pearson -0.07, median relative difference 0.097) -- the trimmed cone mean
    is a genuinely different measure of thickness, not a noisier one, and that number is the reason
    this claim is stated at the narrow cone and the group keeps pymeshlab as its wide-cone oracle.

    The mutation probe: shuffling ordito's answer takes the median relative difference to **0.157**
    (238x the measured agreement) and the correlation to -0.02.
    """
    mesh_tm, mesh_wp = ellipsoid
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    normals_wp = od.vertices.vertex_normals(vertices_wp, faces_wp, weighting="angle")

    thickness_ml = meshlib_scalars_to_numpy(
        mm.computeRayThicknessAtVertices(trimesh_to_meshlib(mesh_tm))
    )
    finite = thickness_ml < 1e30
    assert finite.all()  # non-vacuity: every vertex found an opposite surface
    assert_nonconstant(thickness_ml, tol=1.0)  # ... and the answer is not a constant

    narrow_np = od.visibility.shape_diameter(
        mesh_wp, vertices_wp, normals=normals_wp, n_rays=64, cone_angle=0.05
    ).numpy()
    relative = np.abs(narrow_np - thickness_ml) / thickness_ml
    assert np.median(relative) < 0.002  # 3x the measured 6.6e-04
    assert np.corrcoef(narrow_np, thickness_ml)[0, 1] > 0.999

    # The wide cone is a different measure, and saying so is half the claim.
    wide_np = od.visibility.shape_diameter(mesh_wp, vertices_wp, normals=normals_wp).numpy()
    assert np.median(np.abs(wide_np - thickness_ml) / thickness_ml) > 0.05


def test_shape_diameter_invalid(device: str) -> None:
    _sphere_tm, mesh_wp, normals_wp = _sphere_wp(device, 1.0, subdivisions=1)
    with pytest.raises(ValueError, match="n_rays >= 1"):
        od.visibility.shape_diameter(mesh_wp, mesh_wp.points, n_rays=0)
    with pytest.raises(ValueError, match=r"cone_angle must be in \(0, pi / 2\]"):
        od.visibility.shape_diameter(mesh_wp, mesh_wp.points, cone_angle=2.0)
    with pytest.raises(ValueError, match="trim must be non-negative"):
        od.visibility.shape_diameter(mesh_wp, mesh_wp.points, trim=-1.0)
    with pytest.raises(ValueError, match="one entry per point"):
        od.visibility.shape_diameter(
            mesh_wp, mesh_wp.points, normals=wp.zeros(2, dtype=wp.vec3, device=device)
        )
    assert normals_wp.size > 0


def test_shape_diameter_empty(device: str) -> None:
    """Not a parity assert: the empty case of ``test_ambient_occlusion_empty``, both halves."""
    _sphere_tm, mesh_wp, _normals_wp = _sphere_wp(device, 1.0, subdivisions=1)
    points_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    assert od.visibility.shape_diameter(mesh_wp, points_wp).shape == (0,)
    with pytest.raises(ValueError, match="one entry per point"):
        od.visibility.shape_diameter(
            mesh_wp, points_wp, normals=wp.zeros(2, dtype=wp.vec3, device=device)
        )


# ---------------------------------------------------------------------------
# thickness (trimesh reference)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "seed", "atol"),
    [pytest.param("max_sphere", 7, 1e-5, id="max_sphere"), pytest.param("ray", 13, 1e-8, id="ray")],
)
@pytest.mark.parity("thickness_interior", "trimesh")
def test_thickness_matches_trimesh(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
    method: Literal["max_sphere", "ray"],
    seed: int,
    atol: float,
) -> None:
    """
    Class A against ``trimesh.proximity.thickness``, including *which* points are infinite.

    The ``isfinite`` mask is compared before the values, so a point where one library finds no
    opposite surface and the other does is a failure rather than a skipped element. Measured 20 of
    20 finite on this fixture for both methods, so the guarded ``allclose`` does run -- the guard
    is there for a fixture where it would not, and is not silently making this test vacuous here.

    ``method="ray"`` is a separate algorithm from ``max_sphere`` rather than a tuning of it, so it
    has its own row, at ``allclose``'s default tolerance; the ``thickness_interior`` benchmark group
    is parametrized over both ``method`` values and trimesh is timed for both, so neither row rests
    on the other branch's comparison.
    """
    mesh_tm, mesh_wp = icosahedron
    points_np, face_ids = tm.sample.sample_surface(mesh_tm, 20, seed=seed)[:2]
    normals_np = mesh_tm.face_normals[face_ids]

    points_wp = points_to_warp(points_np, mesh_wp.device)
    normals_wp = points_to_warp(normals_np, mesh_wp.device)

    thickness_wp = od.visibility.thickness(
        mesh_wp, points_wp, normals=normals_wp, method=method
    ).numpy()
    thickness_tm = tm_proximity.thickness(mesh_tm, points_np, normals=normals_np, method=method)

    finite_tm = np.isfinite(thickness_tm)
    assert np.array_equal(np.isfinite(thickness_wp), finite_tm)
    if finite_tm.any():
        assert np.allclose(thickness_wp[finite_tm], thickness_tm[finite_tm], rtol=1e-5, atol=atol)


@pytest.mark.parity("thickness_at_vertices", "meshlib", "trimesh")
def test_thickness_at_vertices_matches_meshlib(ellipsoid: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A, and it pins the normal convention: **angle-weighted**, not area-weighted.

    ``computeRayThicknessAtVertices`` is the same measure as ``method="ray"`` -- the distance from
    each vertex along minus its normal to the first surface the ray meets -- and it is the only
    reference in the suite that answers it for a whole vertex buffer at once, which is why the
    benchmark group runs at every vertex rather than on a subsample.

    The pairing is exact only with the right normals, and that is the substance of this test rather
    than an incidental detail. MeshLib's ``MeshPoint::set`` takes the direction from the
    *pseudonormal*, which section 6 records as the match for
    [`vertex_normals`][ordito.vertices.vertex_normals] at ``weighting="angle"`` (1.19e-07).
    Measured on the ellipsoid: **5.96e-07** absolute and 3.48e-07 relative with those normals,
    against **0.031** -- five orders worse -- with the area-weighted ones. So the second assert is
    what makes the first one a claim about the ray rather than about the tolerance.

    MeshLib reports ``FLT_MAX`` where no opposite surface is found and ordito reports ``inf``; the
    fixture is closed, so all 642 vertices are finite here and the mask is asserted rather than
    used to skip elements.

    **trimesh is checked here too**, on the same whole-vertex-buffer call, because that group times
    both references. It takes a query set, so unlike MeshLib it can be asked at exactly these
    vertices with exactly these normals -- which makes it the arm that rules out the two libraries
    agreeing on a shared convention mistake, since it derives its ray direction from its own
    ``vertex_normals`` rather than from a pseudonormal.
    """
    mesh_tm, mesh_wp = ellipsoid
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    n_vertices = vertices_wp.size

    thickness_ml = meshlib_scalars_to_numpy(
        mm.computeRayThicknessAtVertices(trimesh_to_meshlib(mesh_tm))
    )
    assert thickness_ml.shape == (n_vertices,)
    assert (thickness_ml < 1e30).all()  # every vertex found an opposite surface
    # non-vacuity: on a sphere every ray would read 2 * radius
    assert_nonconstant(thickness_ml, tol=1.0)

    angle_normals_wp = od.vertices.vertex_normals(vertices_wp, faces_wp, weighting="angle")
    thickness_wp = od.visibility.thickness(
        mesh_wp, vertices_wp, method="ray", normals=angle_normals_wp
    ).numpy()
    assert np.isfinite(thickness_wp).all()
    assert np.allclose(thickness_wp, thickness_ml, rtol=1e-5, atol=1e-5)

    # The other weighting is not the pairing, and the gap is five orders of magnitude.
    area_normals_wp = od.vertices.vertex_normals(vertices_wp, faces_wp)
    thickness_area_np = od.visibility.thickness(
        mesh_wp, vertices_wp, method="ray", normals=area_normals_wp
    ).numpy()
    assert np.abs(thickness_area_np - thickness_ml).max() > 1e-3

    # The third implementation, on the identical query set and normals: trimesh takes both as
    # arguments, so this arm is not sharing a normal convention with either of the other two.
    thickness_tm = tm_proximity.thickness(
        mesh_tm,
        np.ascontiguousarray(vertices_wp.numpy(), dtype=np.float64),
        normals=np.ascontiguousarray(angle_normals_wp.numpy(), dtype=np.float64),
        method="ray",
    )
    assert np.isfinite(thickness_tm).all()
    assert np.allclose(thickness_wp, thickness_tm, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# max_tangent_sphere (trimesh reference)
# ---------------------------------------------------------------------------


def test_max_tangent_sphere(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A on both returns, at ``1e-2`` -- the loosest tolerance in this file, and why.

    The sphere's centre slides along the normal as its radius grows, so a small radius disagreement
    displaces the centre by the same amount; both are compared rather than just the radius, since a
    centre off the normal would be a different failure. The tolerance is set by
    ``mesh_query_point``'s own accuracy (section 6 records it as up to 2.1e-5 absolute) amplified by
    that sliding, not by a disagreement about the definition. 20 of 20 points finite.
    """
    mesh_tm, mesh_wp = icosahedron
    points_np, face_ids = tm.sample.sample_surface(mesh_tm, 20, seed=42)[:2]
    normals_np = mesh_tm.face_normals[face_ids]

    points_wp = points_to_warp(points_np, mesh_wp.device)
    normals_wp = points_to_warp(normals_np, mesh_wp.device)

    centers_wp, radii_wp = od.visibility.max_tangent_sphere(mesh_wp, points_wp, normals=normals_wp)
    centers_tm, radii_tm = tm_proximity.max_tangent_sphere(mesh_tm, points_np, normals=normals_np)

    finite_tm = np.isfinite(radii_tm)
    assert np.array_equal(np.isfinite(radii_wp.numpy()), finite_tm)
    if finite_tm.any():
        assert np.allclose(radii_wp.numpy()[finite_tm], radii_tm[finite_tm], rtol=1e-2, atol=1e-2)
        assert np.allclose(
            centers_wp.numpy()[finite_tm], centers_tm[finite_tm], rtol=1e-2, atol=1e-2
        )


@pytest.mark.parity("max_tangent_sphere_reach", "trimesh")
def test_max_tangent_sphere_reach_matches_trimesh(cave_cube: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A on both returns and on which queries are unbounded, for the **exterior** branch.

    ``trimesh.proximity.max_tangent_sphere(inwards=False)`` is the only exterior tangent sphere in
    any installed library, so this closes the gap the group's meshlib exemption describes: that
    exemption says *MeshLib* has no exterior form, not that nothing does.

    Two things about the setup, both measured, and each of which makes the test either vacuous or
    ill-conditioned if got wrong:

    - **The fixture must be non-convex.** The exterior tangent sphere of a convex body is unbounded,
      so on ``icosahedron`` both libraries correctly return ``inf`` at every query (measured 0 of 24
      finite) and an ``isfinite``-guarded ``allclose`` would compare nothing while reading as
      coverage.
    - **The sample count has to reach the concavity.** ``cave_cube`` is a unit cube minus a
      0.1 interior cube, so the cavity is ~0.6% of the surface area: at 24 samples 0 land on it and
      at 1 024 between 8 and 12 do (measured over four seeds). The bounded spheres are the ones
      spanning that cavity's opposing walls.

    Flat opposing walls are also why this fixture is the right one rather than ``half_torus``, which
    has far more finite queries (147 of 256) but whose smoothly curved, exponentially scaled surface
    is ill-conditioned: a float32 difference in the tangent direction moves the radius by up to 0.12
    relative there, against **4.9e-07** here. That is conditioning, not a disagreement about the
    definition -- radius grows without bound as the surface turns locally convex.

    Measured over seeds 42 / 7 / 13 / 0: the ``isfinite`` masks are equal element for element every
    time, and on the finite subset the radii agree to at most 5.7e-07 relative and the centres to
    4.3e-09 absolute. The ``1e-5`` tolerance is ~18x above that. Both returns are compared, as in
    [`test_max_tangent_sphere`]: the centre slides along the normal with the radius, so a centre off
    the normal is a distinct failure from a wrong radius.
    """
    mesh_tm, mesh_wp = cave_cube
    points_np, face_ids = tm.sample.sample_surface(mesh_tm, 1024, seed=42)[:2]
    normals_np = mesh_tm.face_normals[face_ids]

    points_wp = points_to_warp(points_np, mesh_wp.device)
    normals_wp = points_to_warp(normals_np, mesh_wp.device)

    centers_wp, radii_wp = od.visibility.max_tangent_sphere(
        mesh_wp, points_wp, normals=normals_wp, inwards=False
    )
    centers_tm, radii_tm = tm_proximity.max_tangent_sphere(
        mesh_tm, points_np, normals=normals_np, inwards=False
    )

    finite_tm = np.isfinite(radii_tm)
    # Non-vacuous in both directions: a convex fixture would leave every radius infinite, and an
    # implementation that never escaped the surface would leave none of them.
    assert 5 <= finite_tm.sum() < radii_tm.size
    assert np.array_equal(np.isfinite(radii_wp.numpy()), finite_tm)
    assert np.allclose(radii_wp.numpy()[finite_tm], radii_tm[finite_tm], rtol=1e-5, atol=1e-5)
    assert np.allclose(centers_wp.numpy()[finite_tm], centers_tm[finite_tm], rtol=1e-5, atol=1e-5)


@pytest.mark.parity(
    "max_tangent_sphere_reach",
    "meshlib",
    benchmarked=False,
    reason="that group times the *exterior* branch, and MeshLib has no exterior form: "
    "InSphereSearchSettings.insideAndOutside returns the smaller of the inside and outside spheres "
    "with a sign rather than the outside one. Its interior form also takes no query set -- it "
    "answers at every vertex, where ordito's iteration is ill-conditioned (measured 0.0018 radius "
    "on a unit sphere, see test_max_tangent_sphere_agrees_across_devices) -- so it cannot be asked "
    "about the interior points this test uses. trimesh has an exterior form, carries the oracle "
    "for the values and is now the group's timed reference; this declaration is about MeshLib "
    "alone.",
)
def test_max_tangent_sphere_matches_meshlib(ellipsoid: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C (a median relative difference): the same shrinking-sphere algorithm, from just inside.

    Both implementations are Inui et al.'s shrinking sphere, so this is the closest thing to a
    second implementation ordito's iteration has -- and the reason it is Class C rather than A is a
    query-point difference neither side can remove. MeshLib excludes the faces incident to the
    vertex it measures at (``MeshPoint::notIncidentFaces``); ordito takes no such predicate, so at
    a point exactly *on* the surface its sphere collapses -- 0.0018 on a unit sphere, the degeneracy
    [`test_max_tangent_sphere_agrees_across_devices`] is written around. MeshLib in turn takes no
    query set, so it cannot be asked at the offset points. The comparison therefore pulls ordito's
    queries a short way inside along the normal and compares the two fields.

    Measured on the ellipsoid over 642 vertices spanning 0.52 to 1.40, the median relative
    difference **tracks the offset one for one**: 0.201% at an offset of 0.2% of the smallest
    bounding-box side, 0.503% at 0.5%, 1.006% at 1.0%, 2.008% at 2.0% (Pearson 0.979 down to 0.900
    across that range). So what is left between the two implementations is the offset itself rather
    than a disagreement -- which is the strongest form this claim can take, given that neither side
    can be asked the other's question. Below ~0.2% the collapse takes over instead and the
    difference jumps to 41% at 0.1% and 71% at 0.05%. The test runs at 0.5% and bounds the median at
    3x it. The mutation probe: shuffling one side takes the median relative difference to **0.19**
    (38x) and the correlation to 0.01.

    ``maxRadius`` must be set: it defaults to **1**, which on a mesh of any other scale silently
    caps every answer. Half the smallest bounding-box side is the article's own recommendation.
    """
    mesh_tm, mesh_wp = ellipsoid
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    normals_wp = od.vertices.vertex_normals(vertices_wp, faces_wp, weighting="angle")
    extent_np = mesh_tm.bounds[1] - mesh_tm.bounds[0]

    settings_ml = mm.InSphereSearchSettings()
    settings_ml.maxRadius = float(0.5 * extent_np.min())  # the default is 1, whatever the scale
    settings_ml.maxIters = 100  # ordito's max_iter default, so neither side stops earlier
    diameter_ml = meshlib_scalars_to_numpy(
        mm.computeInSphereThicknessAtVertices(trimesh_to_meshlib(mesh_tm), settings_ml)
    )
    assert (diameter_ml < settings_ml.maxRadius * 2.0).all()  # nothing hit the cap
    assert_nonconstant(diameter_ml, tol=0.5)  # non-vacuity: a sphere would read one constant

    offset = 0.005 * float(extent_np.min())  # 0.5% of the smallest side; see the docstring sweep
    inside_np = np.ascontiguousarray(
        mesh_tm.vertices - offset * normals_wp.numpy(), dtype=np.float32
    )
    inside_wp = points_to_warp(inside_np, mesh_wp.device)
    _centers_wp, radii_wp = od.visibility.max_tangent_sphere(
        mesh_wp, inside_wp, inwards=True, normals=normals_wp
    )
    diameter_wp = 2.0 * radii_wp.numpy()
    assert np.isfinite(diameter_wp).all()

    relative = np.abs(diameter_wp - diameter_ml) / diameter_ml
    assert np.median(relative) < 0.015  # 3x the measured 0.00503, which is the offset itself
    assert np.corrcoef(diameter_wp, diameter_ml)[0, 1] > 0.93


def test_max_tangent_sphere_empty(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a parity assert: the empty case of ``test_ambient_occlusion_empty``, both halves.

    ``max_tangent_sphere``'s ``m == 0`` early return sat *after* its inline normals-length check
    rather than before it, but reached the same bug from the AABB-reduction side: two ``aabb``
    passes ran before the check regardless. Both are pinned here.
    """
    _, mesh_wp = icosahedron
    points_wp = warp_empty(0, wp.vec3, mesh_wp.device)
    centers_wp, radii_wp = od.visibility.max_tangent_sphere(mesh_wp, points_wp)
    assert centers_wp.shape == (0,)
    assert radii_wp.shape == (0,)
    with pytest.raises(ValueError, match="one entry per point"):
        od.visibility.max_tangent_sphere(
            mesh_wp, points_wp, normals=wp.zeros(2, dtype=wp.vec3, device=mesh_wp.device)
        )


def test_max_tangent_sphere_normalizes_a_non_unit_normal(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Ordito against ordito: a caller-supplied non-unit normal must not silently scale the radius.

    Every ray-bundle measure in this module normalizes its normals defensively (``hemisphere_frame``
    does it per-thread); the shrinking-sphere path had no equivalent guard, so a normal scaled by
    2.0 used to move both the sphere's center (``point + normal * radius``) and, through
    ``step_sphere_shrink``'s convergence test, its radius -- silently, since nothing here documents
    or checks unit length beyond the docstring's word.
    """
    _mesh_tm, mesh_wp = icosahedron
    points_wp = mesh_wp.points
    unit_normals_wp = od.vertices.vertex_normals(mesh_wp.points, mesh_wp.indices)
    scaled_normals_wp = warp_empty(unit_normals_wp.size, wp.vec3, mesh_wp.device)
    wp.map(wp.mul, unit_normals_wp, wp.float32(2.0), out=scaled_normals_wp)

    centers_unit_wp, radii_unit_wp = od.visibility.max_tangent_sphere(
        mesh_wp, points_wp, normals=unit_normals_wp
    )
    centers_scaled_wp, radii_scaled_wp = od.visibility.max_tangent_sphere(
        mesh_wp, points_wp, normals=scaled_normals_wp
    )
    assert np.allclose(radii_scaled_wp.numpy(), radii_unit_wp.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(centers_scaled_wp.numpy(), centers_unit_wp.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kernel_device", ["cpu", "cuda:0"])
def test_max_tangent_sphere_agrees_across_devices(kernel_device: str) -> None:
    """
    Pin the packed support-argmax reduction, which had the same one-lane-per-block CPU bug.

    Query points are pulled *inside* the surface on purpose: with them exactly on it the
    shrinking-sphere iteration is ill-conditioned (the sphere collapses to a 0.0018 radius on a unit
    sphere) and the two devices then differ by 23% from float ordering alone, which would make this
    a test of that degeneracy rather than of the reduction.
    """
    if kernel_device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    mesh_tm = tm.creation.icosphere(subdivisions=2)
    mesh_wp = trimesh_to_warp(mesh_tm, kernel_device)
    vertices_np = np.asarray(mesh_tm.vertices)
    points_wp = points_to_warp(vertices_np * 0.95, kernel_device)
    normals_wp = points_to_warp(np.asarray(mesh_tm.vertex_normals), kernel_device)
    _centers_wp, radii_wp = od.visibility.max_tangent_sphere(mesh_wp, points_wp, normals=normals_wp)
    # The inscribed tangent sphere of a unit sphere, from just inside it, is the sphere itself.
    assert np.allclose(radii_wp.numpy(), 0.95, rtol=0.1, atol=0.1)
