"""
Benchmarks for ``triwarp.levelset``.

Two groups, and both are sized by the **lattice** rather than by a mesh, which is what the module
has in common: ``marching_cubes`` takes a field and no mesh at all, and ``offset_mesh`` turns its
input into a field before doing anything.

``marching_cubes`` is the primitive the other rows here end in, so it is timed on its own first
-- against ``meshlib``'s ``marchingCubes``, the same algorithm on the same case table, which makes
it
one of the suite's cleanest comparisons. Its resolution pair is a slope check rather than a size
sweep; read a regression as the slope steepening, not the absolute number moving. ``thicken_mesh``
has no group: it is a per-vertex extrusion plus a rim band, so it belongs to the mesh-edit cost
family and not to this file's axis -- which is the same reason it is the one member of the module
that is not a level-set operation.

The offset group's axis is likewise the **lattice**, not the mesh. A level-set offset samples a
signed distance at every point of a regular grid and marches it, so its cost is
``resolution ** 3`` closest-point queries plus one marching-cubes pass -- the input's face count
enters only through the BVH descent each query pays. That is why the cell width is the parametrized
axis here and the mesh sweep is along for the ride: doubling the resolution is 8x the queries
whatever the mesh, and a row whose slope is not ~8x is telling you the BVH, not the lattice, is the
cost.

References
----------
**meshlib** ``offsetMesh`` is the same algorithm: an OpenVDB level set at a given ``voxelSize``,
marched back to a mesh. It is the fair row -- both sides sample a field on a lattice of the same
pitch, and ``tests/test_levelset.py`` pins them to within half a cell of each other on the surface
and to within 5 % on the vertex count.

**pymeshlab** ``generate_resampled_uniform_mesh`` is MeshLab's offset and is timed at the same cell
size. Its ``offset`` parameter is passed as ``PureValue``, which is mandatory rather than stylistic:
as a ``PercentageValue`` it runs from full erosion at 0 % to full dilation at 100 %, so its own
default is the *zero* offset (CLAUDE.md section 7.6).

open3d has no offset, and neither does trimesh or igl: a level-set offset needs a signed distance
field on a lattice, and of the six CPU references only these two build one.
"""

from __future__ import annotations

import igl
import numpy as np
import pymeshlab as ml
import pytest
import pyvista as pv
import torch
import warp as wp
from meshlib import mrmeshpy as mm
from pytorch3d.ops.marching_cubes import marching_cubes as p3d_marching_cubes

import triwarp as tw
from conftest import BenchCase, BenchLibrary, skip_larger_than

# Cell widths as a fraction of the bounding-box diagonal, and the offset distance as a multiple of
# the cell. The pair is a slope check: 1/64 to 1/128 is 8x the samples, and an offset of four cells
# is wide enough that the padding (``ceil(distance / cell) + 2``) is a real part of the lattice.
_CELL_DIVISORS = [64, 128]
_OFFSET_CELLS = 4.0


def _cell_and_offset(bench_case: BenchCase, divisor: int) -> tuple[float, float]:
    """Absolute cell width from the mesh's own bbox diagonal, and the offset that goes with it."""
    vertices_np = bench_case.vertices_np
    diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
    cell = diagonal / divisor
    return cell, _OFFSET_CELLS * cell


@pytest.mark.benchmark(group="offset_mesh")
@pytest.mark.benchlibs("triwarp", "meshlib", "pymeshlab")
@pytest.mark.parametrize("divisor", _CELL_DIVISORS)
def test_offset_mesh(bench_case: BenchCase, divisor: int) -> None:
    """
    Outward level-set offset at a matched cell width: field sampling plus one marching-cubes pass.

    Every side gets the identical absolute cell and the identical absolute offset, so the row
    compares two field-and-march implementations rather than two parameter conventions. The lattice
    includes the padding the offset needs -- ``ceil(distance / cell) + 2`` cells on every side, or
    the level set is clipped by the boundary -- which at four cells of offset is a noticeable
    fraction of the box and is paid by triwarp's row alone (both references pad internally).

    Read the two divisors as a slope. 2x the resolution is **8x the samples**, and triwarp's row
    grows only **2.8x** -- the samples are independent queries, so a lattice this size does not
    saturate the GPU and the wall clock tracks occupancy rather than work. The reference rows grow
    2.2x (meshlib) and 1.6x (pymeshlab) for the same reason on their own cores, and sit an order of
    magnitude and two orders behind respectively.

    Read meshlib's *median*, not its mean: its OpenVDB band build spreads sevenfold here.
    """
    cell, distance = _cell_and_offset(bench_case, divisor)

    if bench_case.kind == "meshlib":
        skip_larger_than(bench_case, "bunny", "OpenVDB's level set is a single-threaded band build")
        mesh_ml = bench_case.new_mesh_ml()  # held in a name: MeshPart does not own it
        part_ml = mm.MeshPart(mesh_ml)
        parameters_ml = mm.OffsetParameters()
        parameters_ml.voxelSize = cell
        offset_ml = bench_case.run(lambda: mm.offsetMesh(part_ml, distance, parameters_ml))
        assert offset_ml.topology.numValidFaces() > 0
        return

    if bench_case.kind == "pymeshlab":
        skip_larger_than(bench_case, "bunny", "MeshLab's resampler is a serial dense pass")
        cell_pml = ml.PureValue(cell)
        offset_pml = ml.PureValue(distance)

        def resample_pml() -> int:
            meshset_pml = bench_case.new_meshset_pml()
            meshset_pml.generate_resampled_uniform_mesh(
                cellsize=cell_pml, offset=offset_pml, mergeclosevert=True
            )
            return meshset_pml.current_mesh().face_number()

        assert bench_case.run(resample_pml) > 0
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    offset_vertices, offset_faces = bench_case.run(
        lambda: tw.levelset.offset_mesh(vertices, faces, distance, cell)
    )
    assert int(offset_faces.shape[0]) > 0
    assert int(offset_vertices.shape[0]) > 0


@pytest.mark.benchmark(group="thicken_mesh")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_thicken_mesh(bench_case: BenchCase) -> None:
    """
    The topology-preserving shell: two vertex passes and one band, no lattice anywhere.

    Read this against ``offset_mesh`` above -- the two answer the same question and their costs are
    not comparable in kind. This one is linear in the *mesh* (normals, two displacements, a rim
    walk) and produces ``2F + 2E`` triangles; the level-set offset is cubic in the *lattice* and
    produces whatever the marching finds. So the interesting number is the ratio, and it is the
    argument for having both.

    meshlib's ``makeThickMesh`` is the same construction, and ``tests/test_offset.py`` shows just
    how same: identical counts, volumes equal to five decimals, and identical face sets under a
    vertex bijection. That makes this the cleanest row in the module -- two implementations of one
    algorithm, no parameter mapping in between. Its ``ThickenParams`` is built outside the timed
    callable and its mesh is fresh per round, since the call mutates nothing but the params object
    is an input.

    The thickness is 1 % of the bounding-box diagonal, which is small enough that neither side folds
    (a shell that self-intersects past the local curvature radius is not the same amount of work).

    **The reference is capped at ``sphere_med`` because of what the slope shows.** triwarp is flat
    across the ``scale`` axis -- three launches against the launch floor -- where meshlib is
    superlinear, so the gap runs from single-figure to four orders of magnitude over that axis. For
    two implementations of the same construction that means the reference's cost is not the
    construction.
    """
    thickness = 0.01 * float(
        np.linalg.norm(bench_case.vertices_np.max(axis=0) - bench_case.vertices_np.min(axis=0))
    )
    if bench_case.kind == "meshlib":
        # ``skip_larger_than`` is a no-op on the synthetic feature meshes, so the cap is by name,
        # as ``test_heat_distance.py``'s serial rows are.
        if bench_case.mesh_name == "sphere_large":
            pytest.skip("makeThickMesh is 4.2 s at 1 M faces: capped at sphere_med")
        parameters_ml = mm.ThickenParams()
        parameters_ml.insideOffset = thickness
        parameters_ml.outsideOffset = 0.0
        shell_ml = bench_case.run(
            lambda mesh: mm.makeThickMesh(mesh, parameters_ml), setup=bench_case.new_mesh_ml
        )
        assert shell_ml.topology.numValidFaces() > 0
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    shell_vertices, shell_faces = bench_case.run(
        lambda: tw.levelset.thicken_mesh(vertices, faces, thickness)
    )
    assert shell_vertices.size == 2 * bench_case.n_vertices
    assert shell_faces.size >= 6 * bench_case.n_faces


@pytest.mark.benchmark(group="signed_distance_grid")
@pytest.mark.benchlibs("triwarp", "open3d")
@pytest.mark.parametrize("divisor", _CELL_DIVISORS)
def test_signed_distance_grid(bench_case: BenchCase, divisor: int) -> None:
    """
    The field alone, without the marching: ``resolution ** 3`` signed closest-point queries.

    Subtract this from ``offset_mesh`` above to price the extraction, which is the only reason both
    groups exist -- they run the identical lattice, so the difference is Warp's ``MarchingCubes``
    over it.

    open3d's ``RaycastingScene.compute_signed_distance`` is the same quantity, to 1.27e-07
    (``tests/test_proximity.py``), and it is the one reference here that batches: the lattice is
    built once outside the timed callable on both sides, so each row prices the queries. It is taken
    *from the function under test* rather than re-derived, so neither side gets a different lattice.

    The two slopes are the whole row. For 8x the samples open3d grows 6.7x -- a saturated CPU doing
    the work -- where triwarp grows 1.9x, a GPU that was not full at the coarser lattice, so the gap
    widens several-fold across the resolution pair. Subtracting these from ``offset_mesh`` prices
    the extraction, which grows an order of magnitude over the same pair: past a certain resolution
    the offset is dominated by ``MarchingCubes`` rather than by the field.
    """
    cell, _distance = _cell_and_offset(bench_case, divisor)

    if bench_case.kind == "open3d":
        import open3d as o3d

        skip_larger_than(bench_case, "bunny", "the reference is a single-threaded BVH walk")
        # The lattice has to be the *same* lattice, so it is taken from the function under test
        # rather than re-derived here -- run once, untimed, on host copies, since ``vertices_wp`` is
        # a triwarp-only accessor and this row has no device.
        vertices_cpu = wp.array(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
            dtype=wp.vec3,
            device="cpu",
        )
        faces_cpu = wp.array(
            np.ascontiguousarray(bench_case.faces_np, dtype=np.int32).reshape(-1),
            dtype=wp.int32,
            device="cpu",
        )
        field_wp, box = tw.proximity.signed_distance_grid(vertices_cpu, faces_cpu, cell)
        samples_np = tw.voxels.grid_points(field_wp.shape, bounds=box, device="cpu").numpy()
        scene_o3d = o3d.t.geometry.RaycastingScene()
        scene_o3d.add_triangles(
            o3d.core.Tensor(np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32)),
            o3d.core.Tensor(np.ascontiguousarray(bench_case.faces_np, dtype=np.uint32)),
        )
        queries_o3d = o3d.core.Tensor(samples_np.astype(np.float32))
        distance_o3d = bench_case.run(lambda: scene_o3d.compute_signed_distance(queries_o3d))
        assert distance_o3d.shape[0] == samples_np.shape[0]
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    field, box = bench_case.run(lambda: tw.proximity.signed_distance_grid(vertices, faces, cell))
    assert field.shape[0] >= 2
    assert float(box[1][0]) > float(box[0][0])


# Lattice resolutions for the marching-cubes group. The axis is the lattice rather than a mesh, and
# the pair is a slope check: 64 -> 128 is 8x the cells (262 144 -> 2 097 152).
_MARCHING_RESOLUTIONS = [64, 128]

# Major radius, minor radius and the lattice's half-extent for the marched field. A torus rather
# than a sphere because a genus-1 surface crosses roughly twice the cells at the same resolution, so
# the row measures the marching rather than the scan over empty ones.
_MARCHING_TORUS = (0.65, 0.28, 1.1)

_marching_field_cache: dict[int, np.ndarray] = {}
_marching_field_wp_cache: dict[tuple[int, str], wp.array] = {}
_marching_volume_ml_cache: dict[int, mm.SimpleVolume] = {}


def _marching_field_np(resolution: int) -> np.ndarray:
    """Analytic torus SDF on a ``resolution ** 3`` lattice, cached -- the input, not the work."""
    if resolution not in _marching_field_cache:
        major, minor, half = _MARCHING_TORUS
        axis_np = np.linspace(-half, half, resolution)
        x_np, y_np, z_np = np.meshgrid(axis_np, axis_np, axis_np, indexing="ij")
        radial_np = np.sqrt(x_np**2 + y_np**2) - major
        field_np = np.sqrt(radial_np**2 + z_np**2) - minor
        _marching_field_cache[resolution] = np.ascontiguousarray(field_np, dtype=np.float32)
    return _marching_field_cache[resolution]


def _marching_volume_ml(resolution: int) -> mm.SimpleVolume:
    """
    Build the same lattice as a ``meshlib.SimpleVolume``, cached per resolution.

    ``marchingCubes`` reads the volume and returns a new ``Mesh``, so unlike almost every other free
    function in the library this one does *not* mutate its input -- measured, two calls on one
    volume return the identical face count and leave ``dims`` intact -- which is what makes the
    cache legal here where ``new_mesh_ml`` is required elsewhere.

    ``simpleVolumeFrom3Darray`` returns ``voxelSize = (1, 1, 1)`` whatever the array it was handed,
    so the spacing is assigned afterwards; both are outside the timed region, matching triwarp's
    row, whose field is likewise a device array in hand before the clock starts.
    """
    if resolution not in _marching_volume_ml_cache:
        from meshlib import mrmeshnumpy as mn

        half = _MARCHING_TORUS[2]
        spacing = 2.0 * half / (resolution - 1)
        volume_ml = mn.simpleVolumeFrom3Darray(_marching_field_np(resolution))
        volume_ml.voxelSize = mm.Vector3f(spacing, spacing, spacing)
        _marching_volume_ml_cache[resolution] = volume_ml
    return _marching_volume_ml_cache[resolution]


@pytest.mark.benchmark(group="marching_cubes")
@pytest.mark.benchlibs("triwarp", "meshlib", "igl", "pyvista", "pytorch3d")
@pytest.mark.parametrize("resolution", _MARCHING_RESOLUTIONS)
def test_marching_cubes(bench_lib: BenchLibrary, resolution: int) -> None:
    """
    Extract one iso-surface from a dense lattice: the module's second **mesh-free** group.

    It takes ``bench_lib`` rather than ``bench_case`` for the reason ``delaunay_triangulation`` does
    -- the input is a field, not a mesh, so the work is sized by a plain ``parametrize``. All rows
    march the identical analytic torus SDF, built once per resolution outside the timed region.

    **Five implementations of one case table**, which makes this the best-referenced function in the
    package. MeshLib's ``marchingCubes`` is the same algorithm and the parity test pins it hard
    (same vertex and face counts, two-sided Hausdorff 1.2e-07), so this is one of the suite's
    cleanest comparisons -- two implementations of one function, not two algorithms answering one
    question. The named transform the test carries is a convention rather than a cost:
    ``params.origin`` addresses the voxel *centre*, so it is handed ``lower - spacing / 2``, and
    ``lessInside=True`` is what matches triwarp's outside-positive winding.

    It is also the fairest CPU-versus-GPU row this module has, since MeshLib is the suite's only
    **multi-threaded** CPU reference. Both are far sublinear in the lattice, which is the shape to
    watch: read a regression here as the *slope* steepening rather than the absolute number moving.
    ``triwarp-cpu`` loses to meshlib by an order of magnitude, which is the other edge of the same
    knife and not a defect to chase -- a hundred-odd threads of C++ against Warp's CPU backend is
    not a comparison of algorithms, and CLAUDE.md section 9's "decide on the CUDA number" governs.

    Each remaining reference has a lattice convention that has to be got right or the row marches a
    shifted field:

    * ``igl.marching_cubes(S, GV, nx, ny, nz, iso)`` takes the sample *positions* explicitly in
      **Fortran order** and returns **three** values (a 2-tuple unpack raises ``ValueError``).
    * ``pv.ImageData(...).contour([iso])`` addresses samples on grid **nodes**, so its ``origin`` is
      the bounds' lower corner *directly* -- the opposite of MeshLib's voxel-centre convention. It
      needs the field in Fortran order too.
    * **pytorch3d is the only one needing no convention fix at all**: ``return_local_coords=False``
      emits lattice indices, exactly what ``marching_cubes`` returns without ``bounds``. It is also
      the only reference here with GPU kernels. Note the import -- it is **not** re-exported from
      ``pytorch3d.ops``, only from ``pytorch3d.ops.marching_cubes``.

    All of them return the same surface, pinned in ``tests/test_levelset.py``.
    """
    if bench_lib.kind == "pytorch3d":
        field_p3d = torch.as_tensor(_marching_field_np(resolution), device=bench_lib.torch_device)[
            None
        ]
        vertices_p3d, faces_p3d = bench_lib.run(
            lambda: p3d_marching_cubes(field_p3d, isolevel=0.0, return_local_coords=False)
        )
        assert faces_p3d[0].shape[0] > 0
        assert vertices_p3d[0].shape[0] > 0
        return
    if bench_lib.kind == "igl":
        field_np = _marching_field_np(resolution)
        half = _MARCHING_TORUS[2]
        axis_np = np.linspace(-half, half, resolution)
        x_np, y_np, z_np = np.meshgrid(axis_np, axis_np, axis_np, indexing="ij")
        lattice_igl = np.ascontiguousarray(
            np.stack([x_np.ravel(order="F"), y_np.ravel(order="F"), z_np.ravel(order="F")], axis=1),
            dtype=np.float64,
        )
        values_igl = np.ascontiguousarray(field_np.ravel(order="F"), dtype=np.float64)
        vertices_igl, faces_igl, _info_igl = bench_lib.run(
            lambda: igl.marching_cubes(
                values_igl, lattice_igl, resolution, resolution, resolution, 0.0
            )
        )
        assert faces_igl.shape[0] > 0
        assert vertices_igl.shape[0] > 0
        return
    if bench_lib.kind == "pyvista":
        field_np = _marching_field_np(resolution)
        half = _MARCHING_TORUS[2]
        spacing = 2.0 * half / (resolution - 1)
        grid_pv = pv.ImageData(
            dimensions=(resolution, resolution, resolution),
            origin=(-half, -half, -half),
            spacing=(spacing, spacing, spacing),
        )
        grid_pv.point_data["field"] = field_np.ravel(order="F")
        contour_pv = bench_lib.run(lambda: grid_pv.contour([0.0], scalars="field"))
        assert contour_pv.n_cells > 0
        return
    if bench_lib.kind == "meshlib":
        volume_ml = _marching_volume_ml(resolution)
        half = _MARCHING_TORUS[2]
        spacing = 2.0 * half / (resolution - 1)
        corner = -half - spacing / 2.0
        params_ml = mm.MarchingCubesParams()
        params_ml.iso = 0.0
        params_ml.lessInside = True
        params_ml.origin = mm.Vector3f(corner, corner, corner)
        mesh_ml = bench_lib.run(lambda: mm.marchingCubes(volume_ml, params_ml))
        assert mesh_ml.topology.numValidFaces() > 0
        return
    device = bench_lib.device
    key = (resolution, str(device))
    if key not in _marching_field_wp_cache:
        _marching_field_wp_cache[key] = wp.array(
            _marching_field_np(resolution), dtype=wp.float32, device=device
        )
    field_wp = _marching_field_wp_cache[key]
    half = _MARCHING_TORUS[2]
    bounds = (wp.vec3(-half, -half, -half), wp.vec3(half, half, half))
    _vertices_wp, faces_wp = bench_lib.run(
        lambda: tw.levelset.marching_cubes(field_wp, 0.0, bounds=bounds)
    )
    assert faces_wp.size > 0
