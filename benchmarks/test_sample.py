"""
Benchmarks for ``triwarp.sample``: uniform surface sampling and blue-noise selection.

The radius targets ~2,000 samples (same helper formula as ``tests/test_sample.py``). Capped at
``bunny``.

**This group is overwhelmingly device-bound, and reading that correctly is what closed it.** A
Bridson active-list sampler spends nearly all of its time in kernels across ~100 rounds, and four
micro-optimizations of its hot kernel all lost because the cost is structural: an active parent must
enumerate a ``9x9x9`` shell of background cells every round to find a child in its ``[r, 2r]``
annulus, and the round count is set by how the front advances rather than by the work.
Randomized-priority selection over the whole pool — 27 cells, a handful of rounds — is **an order of
magnitude**, with *tighter* coverage than either reference. A fifth micro-optimization is worth
recording as a null: making the shell permutation lazy (a partial Fisher-Yates drawing only the
prefix the loop consumes) is a **wash**, so the shuffle was never the cost either.

**open3d**'s ``sample_points_poisson_disk`` is the reference: the same blue-noise / Poisson-disk
surface sampling problem, parametrized by sample *count* rather than by radius, so it is given
``_TARGET_SAMPLES`` — the count triwarp's radius is derived to produce. Open3D's implementation
starts from a dense uniform sample and eliminates points down to the target (Yuksel's sample
elimination), where triwarp reduces a dense pool by randomized priority; the comparison is of
cost per sample delivered, not of identical work.

**pymeshlab**'s ``generate_sampling_poisson_disk`` and **libigl**'s ``igl.blue_noise`` are the two
references that can be given the *radius* rather than a count (``radius=PureValue(r)`` overrides
MeshLab's ``samplenum`` outright; igl's third positional argument *is* ``r``), so they receive the
identical parameter and the radius sweep this group is built around maps across libraries. MeshLab's
algorithm is Corsini et al.'s *hierarchical* dart throwing and igl's is Bridson active-list dart
throwing -- **four implementations, four schemes, one parametrization**, with triwarp's randomized-
priority selection and open3d's sample elimination as the other two. MeshLab is the closest of the
three in output: same exact minimum distance, coverage within a few percent. It pushes the sample
cloud onto
the MeshSet as a new mesh, so the set is rebuilt per round.

**libigl is where this port came from, and the row inverted when the algorithm changed.** The
``30x`` oversampling factor ``sample_surface_blue_noise`` draws its pool at is
``igl::blue_noise``'s, and while triwarp ran Bridson too, igl was *faster*. With
randomized-priority selection triwarp leads, and the margin **grows as the radius falls** -- from a
few times at a wide radius to more than an order of magnitude at a narrow one -- because igl's
serial cost is per accepted sample where triwarp's is per round. igl also returns a few percent
*fewer* samples at the same radius, so the ratios are a lower bound per sample delivered -- and its
*quality* is the best of the three references (``tests/test_sample.py`` measures the coverage gaps),
so this is not speed bought with quality on either side. It gets ``rounds=3``.

MeshLab's uniform ``generate_sampling_montecarlo`` is deliberately **not** a row in ``blue_noise``:
it is not a blue-noise sampler at all (no minimum-distance guarantee), so it would be a floor
rather than a comparison. It belongs to the same question as the ``sample_surface`` group below,
where it is not a row either -- for a different reason, given there.
``generate_sampling_volumetric`` and ``generate_simplified_point_cloud`` are likewise different
problems.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import igl
import numpy as np
import pymeshlab as ml
import pytest
import pytorch3d.ops as p3d_ops
import trimesh as tm
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase, skip_larger_than

if TYPE_CHECKING:
    from typing_extensions import Buffer

_TARGET_SAMPLES = 2_000
_SEED = 11

# Pool oversampling for the one reference that thins a cloud instead of a surface. 30x is the factor
# ``igl::blue_noise`` uses and the one triwarp's own sampler inherited, so every row selects from a
# pool of the same density -- otherwise the row would be measuring how big a pool it was handed.
_POOL_FACTOR = 30

# Radius multipliers applied to the ~2k-sample baseline. Halving the radius multiplies the
# background grid's cells by 8 and quadruples the samples that fit, so this is the module's dominant
# knob -- the output count is *derived* from the radius, never requested.
_RADIUS_SCALES = [1.0, 0.5]

# Sample counts for the uniform group: a decade apart, since the cost is linear in the count and
# the mesh contributes only the one-off area CDF.
_UNIFORM_COUNTS = [10_000, 100_000]

_radius_cache: dict[str, float] = {}


def _radius_for_mesh(bench_case: BenchCase) -> float:
    """Blue-noise radius targeting ~2k samples (mirrors tests/test_sample.py)."""
    if bench_case.mesh_name not in _radius_cache:
        area = float(tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False).area)
        _radius_cache[bench_case.mesh_name] = math.sqrt(
            (area * 0.5 / (_TARGET_SAMPLES * 0.6162910373)) / math.pi
        )
    return _radius_cache[bench_case.mesh_name]


def _mesh_p3d_fresh(bench_case: BenchCase):
    """
    Return a zero-argument builder for a fresh ``Meshes``, for use *inside* a timed callable.

    ``sample_points_from_meshes`` reads ``faces_areas_packed``, which a ``Meshes`` memoizes, so a
    shared container would have rounds 2..n sample from a cached area table -- the same reason
    trimesh's row rebuilds its ``tm.Trimesh`` for its cached area CDF. The tensors themselves are
    hoisted, so only the container and its derivations are inside.
    """
    import pytorch3d.structures as p3d_structures
    import torch

    vertices_p3d = torch.as_tensor(
        np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
        device=bench_case.torch_device,
    )
    faces_p3d = torch.as_tensor(
        np.ascontiguousarray(bench_case.faces_np, dtype=np.int64), device=bench_case.torch_device
    )
    return lambda: p3d_structures.Meshes(verts=[vertices_p3d], faces=[faces_p3d])


@pytest.mark.benchmark(group="sample_surface")
@pytest.mark.benchlibs("triwarp", "igl", "trimesh", "pytorch3d")
@pytest.mark.parametrize("count", _UNIFORM_COUNTS, ids=["n10k", "n100k"])
def test_sample_surface(bench_case: BenchCase, count: int) -> None:
    """
    Uniform area-weighted surface sampling: the dense pool every blue-noise sampler starts from.

    All three libraries take the same two parameters (a count and a seed) and return the same two
    things (positions and the face index each sample landed on), so this is the module's one group
    where nothing has to be matched up -- ``igl.random_points_on_mesh`` and
    ``tm.sample.sample_surface`` are the same call as triwarp's.

    The axis is the count, and it separates the two sides cleanly: **the references are linear in it
    and triwarp is flat** -- a decade more samples for no more time, because at these counts
    triwarp's row is the two launches and the area CDF rather than the sampling. So read the *slope*
    as the statement: the crossover where triwarp's per-sample cost becomes visible is above a
    hundred thousand samples.

    open3d's ``sample_points_uniformly`` and MeshLab's ``generate_sampling_montecarlo`` are the same
    operation but return a bare point cloud with no face index, so they would need a closest-point
    decode before they could be asserted against the area law -- a transform on the *reference* to
    make it comparable, which is what the two references above avoid.

    trimesh rebuilds its ``tm.Trimesh`` inside the timed callable, as the other trimesh rows do,
    because the area CDF is cached on the mesh object and reusing it would time a lookup.

    **pytorch3d**'s ``sample_points_from_meshes`` is the same area-weighted sampler and the only GPU
    one, so it is where the flatness above is tested against another parallel implementation rather
    than a serial baseline. It returns positions alone -- the face index is internal -- so unlike
    igl and trimesh it needs no decode but also cannot be asserted against the area law directly;
    the parity test compares the two clouds distributionally and pins the area law on triwarp's own
    face indices. Its ``Meshes`` memoizes the per-face areas it samples from, so it is built
    **inside** the timed callable and the row carries that derivation, the same thing trimesh's row
    does with its area CDF. One hard limit: it draws the face index with ``torch.multinomial``,
    which refuses more than 2^24 categories, so ``lucy`` raises rather than sampling and is skipped.
    """
    if bench_case.kind == "pytorch3d":
        # ``torch.multinomial`` refuses more than 2^24 categories, and the face buffer *is* the
        # category set here -- so ``lucy``'s 28 055 742 faces raise ``RuntimeError: number of
        # categories cannot exceed 2^24`` rather than sampling. A hard ceiling on the reference, not
        # a cost cap: ``happy_buddha``'s 1 087 716 faces are fine and this is the only mesh in the
        # registry past it.
        skip_larger_than(
            bench_case, "happy_buddha", "torch.multinomial caps the face count at 2^24 categories"
        )
        mesh_p3d = _mesh_p3d_fresh(bench_case)
        samples_p3d = bench_case.run(lambda: p3d_ops.sample_points_from_meshes(mesh_p3d(), count))
        assert samples_p3d.shape == (1, count, 3)
        return
    skip_larger_than(bench_case, "bunny", "the CPU references are single-threaded per sample")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        points, face_index = bench_case.run(
            lambda: tw.sample.sample_surface(vertices, faces, count, seed=_SEED)
        )
        assert points.shape == (count,)
        assert face_index.shape == (count,)
    elif bench_case.kind == "igl":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        _barycentric, face_index_igl, points_igl = bench_case.run(
            lambda: igl.random_points_on_mesh(count, vertices_np, faces_np, _SEED)
        )
        assert points_igl.shape == (count, 3)
        assert face_index_igl.shape == (count,)
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        points_tm, face_index_tm = bench_case.run(
            lambda: tm.sample.sample_surface(
                tm.Trimesh(vertices_np, faces_np, process=False), count, seed=_SEED
            )
        )
        assert points_tm.shape == (count, 3)
        assert face_index_tm.shape == (count,)


@pytest.mark.benchmark(group="blue_noise")
@pytest.mark.benchlibs("triwarp", "igl", "open3d", "pymeshlab", "meshlib")
@pytest.mark.parametrize("radius_scale", _RADIUS_SCALES, ids=["r1", "rhalf"])
def test_sample_surface_blue_noise(bench_case: BenchCase, radius_scale: float) -> None:
    """
    Maximal Poisson-disk selection from a dense pool, on a background grid sized by the radius.

    Halving the radius is 8x the cells and 4x the output, so the pair should show a large,
    superlinear step -- and on ``bunny_decimated`` it is a *flat* one, because the round count does
    not grow with it there and the per-cell summaries that prune each round's shell sweep prune
    hardest exactly where the cells are most numerous. open3d is parametrized by *count* rather than
    radius, so its two rows are matched to the sample count each radius implies.

    **The flatness is ``bunny_decimated``'s alone, and the variable is the round count.** The pool
    build is a rounding error on every row, so this group times the selection almost alone. What
    sets a row apart is how fast the work list decays, and ``bunny`` at half the radius *stalls*:
    its alive count is still a sixth of the pool after twenty rounds, where ``bunny_decimated`` at
    the same pool size is down to a hundredth. The cost is spread across those mid-size rounds and
    **not** in a tail -- rounds with a handful of alive points are a few percent of the selection
    time on every row -- so a tail-specific engine has nothing to collect.

    Two mechanisms are already refuted here and must not be re-proposed: inverting the covering
    sweep to a scatter (built, byte-gated, a win on a dense pool and a **loss** at the radius the
    row is scored at) and compacting the dead entries out of the per-cell membership lists (built,
    byte-gated, a small loss -- the summaries had already removed the work it targets; see
    ``kernels/algorithms/blue_noise.py`` for both).

    **libigl is the reference this port was written from** -- ``sample_surface_blue_noise`` still
    sizes its pool at the ``30x`` oversampling factor ``igl::blue_noise`` uses -- and it takes the
    radius directly, so it and MeshLab both receive triwarp's own parameter. It is Bridson
    active-list dart throwing, which is what triwarp *was*: four schemes across four libraries on
    one parametrization.

    **meshlib is the fifth, and the only one that thins a point cloud rather than a surface.** Its
    row therefore gets the dense pool that every one of these algorithms builds internally, supplied
    as its input and *not* timed -- which makes it the one row that prices the selection alone,
    where the other four each carry their own pool construction. Read the gap between it and triwarp
    as selection-against-selection, and the gap between triwarp and igl or MeshLab as the whole
    pipeline. ``UniformSamplingSettings.distance`` is the radius, given the same value.
    """
    skip_larger_than(bench_case, "bunny")
    # Halving the radius quadruples the samples that fit (area / radius^2).
    target = int(_TARGET_SAMPLES / (radius_scale * radius_scale))
    if bench_case.kind == "igl":
        radius = radius_scale * _radius_for_mesh(bench_case)
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        _barycentric, _face_index, points_igl = bench_case.run(
            lambda: igl.blue_noise(vertices_np, faces_np, radius), rounds=3
        )
        assert points_igl.shape[0] > 0
        return
    if bench_case.kind == "pymeshlab":
        # MeshLab takes *either* a count or an explicit radius, so this is the one blue-noise
        # reference that can be matched to triwarp's actual parameter: ``radius=PureValue(r)``
        # overrides ``samplenum`` and is fed the identical radius. It pushes a new point-cloud mesh
        # onto the set, so the MeshSet is rebuilt per round.
        radius = radius_scale * _radius_for_mesh(bench_case)
        bench_case.run(
            lambda: bench_case.new_meshset_pml().generate_sampling_poisson_disk(
                radius=ml.PureValue(radius)
            )
        )
        return
    if bench_case.kind == "meshlib":
        # The pool is the input here, so it is built (and uploaded) outside the timed callable; the
        # cloud must also outlive the call, which is MeshLib's rule for anything holding one.
        from meshlib import mrmeshnumpy as mn

        radius = radius_scale * _radius_for_mesh(bench_case)
        pool_np, _face_index_np = tm.sample.sample_surface(
            tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False),
            _POOL_FACTOR * target,
            seed=_SEED,
        )[:2]
        # MeshLib's stubs type this as ``Buffer``, which numpy's stubs implement only from 3.12.
        cloud_ml = mn.pointCloudFromPoints(
            cast("Buffer", np.ascontiguousarray(pool_np, dtype=np.float64))
        )
        settings_ml = mm.UniformSamplingSettings()
        settings_ml.distance = radius
        sampled_ml = bench_case.run(lambda: mm.pointUniformSampling(cloud_ml, settings_ml))
        assert 0 < sampled_ml.count() <= pool_np.shape[0]
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        radius = radius_scale * _radius_for_mesh(bench_case)
        points, face_index = bench_case.run(
            lambda: tw.sample.sample_surface_blue_noise(vertices, faces, radius, seed=_SEED)
        )
        assert points.shape[0] > 0
        assert face_index.shape == points.shape
    else:  # open3d takes a target count instead of a radius; sampling does not mutate the mesh
        mesh_o3d = bench_case.mesh_o3d
        cloud = bench_case.run(lambda: mesh_o3d.sample_points_poisson_disk(number_of_points=target))
        assert len(cloud.points) == target


@pytest.mark.benchmark(group="sample_volume")
@pytest.mark.benchmeshes("sphere_small", "sphere_med", "sphere_large")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("count", _UNIFORM_COUNTS, ids=["n10k", "n100k"])
def test_sample_volume(bench_case: BenchCase, count: int) -> None:
    """
    Uniform sampling *inside* a closed mesh: the tetrahedron-fan CDF and one draw kernel.

    triwarp-only, and deliberately so. ``trimesh.sample.volume_mesh`` is rejection sampling against
    a ray-parity containment test, so it returns a *variable* number of points for a requested
    count and its cost is set by the mesh's fill ratio rather than by the count; timing the two
    against each other would compare an exact method with a stochastic one. The fan decomposition
    here has zero rejection on a star-shaped mesh, which is the precondition the wrapper enforces.

    Read this row as the fixed cost of the prologue against the per-sample cost: the fan is one
    per-face kernel, two reductions and a scan, all flat in ``count``, so the slope across the two
    counts is the sampling kernel alone, and the mesh axis moves only the prologue.

    The meshes are named rather than taken from the scan sweep because the decomposition needs a
    **closed, star-shaped** surface: the wrapper rejects a non-watertight mesh outright and refuses
    any mesh whose fan from the centroid produces a negative tetrahedron, which rules out every
    scan mesh (``bunny`` and ``dragon`` are open scans). The three spheres satisfy both by
    construction and span 5k to 328k faces.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    points = bench_case.run(lambda: tw.sample.sample_volume(vertices, faces, count, seed=_SEED))
    assert points.shape == (count,)
