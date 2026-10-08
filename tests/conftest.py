from __future__ import annotations

import os

# Cap the reference libraries' thread pools **before** NumPy is imported, because OpenBLAS reads
# these at library load and never again. One thread per core is not a mild pessimisation on a
# many-core box -- it is a cliff: a small dense ``np.linalg.solve`` runs two to three orders of
# magnitude slower at one thread per core than at eight, and the eigendecomposition behind
# ``test_connection_laplacian_is_symmetric_psd_and_a_rotation_per_block`` an order of magnitude.
# Not a contention artifact -- the same holds on an idle machine. Eight rather than one because the
# eigendecomposition genuinely parallelises where the solves do not care.
#
# This cap does **not** reach a library that sets its own count. MeshLab's screened-Poisson filter
# takes a ``threads`` parameter defaulting to ``hardware_concurrency`` and overrides the
# environment, which is why
# ``tests/test_reconstruction.py`` pins that one at its call site instead. ``meshlib`` is
# deliberately left alone by name -- it is the suite's one legitimately multi-threaded reference --
# but it uses its own pool rather than OpenMP, so these variables do not touch it either.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import functools  # noqa: E402
from types import CodeType  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import trimesh as tm  # noqa: E402
import warp as wp  # noqa: E402
from meshlib import mrmeshpy as mm  # noqa: E402
from warp._src.sparse import (  # noqa: E402
    _bsr_accumulate_triplet_values,  # pyright: ignore[reportPrivateUsage]
    _bsr_assign_copy_blocks,  # pyright: ignore[reportPrivateUsage]
    _bsr_transpose_values,  # pyright: ignore[reportPrivateUsage]
)

import ordito as od  # noqa: E402
from ordito.kernels.linalg import register_warp_overload  # noqa: E402
from ordito.mesh import _CachedProperty  # noqa: E402  # pyright: ignore[reportPrivateUsage]
from tests.conversions import meshlib_to_trimesh, trimesh_to_warp, warp_to_trimesh  # noqa: E402

# Reject a launch whose array arguments do not live on the launch device. Warp's default is
# RELAXED, which passes the pointers straight through: a launch that forgets ``device=`` lands on
# the default CUDA device, reads the CPU arrays over HMM, returns the *right answer*, and then
# corrupts the host heap when those arrays are freed while the kernel is still running (measured:
# 20/20 aborts with a free and no sync, 0/20 with either). CHECKED does not catch it -- it
# validates addressability, which HMM genuinely provides. STRICT is the only mode that rejects a
# genuine cross-device argument, and no ordito launch is intentionally cross-device. It is only
# half the guard: on a CUDA run an omitted ``device=`` resolves to the arrays' own device, so there
# is no mismatch to reject and only check 15's static scan sees it.
if hasattr(wp.config, "launch_array_access_mode"):  # warp >= 1.14
    wp.config.launch_array_access_mode = wp.config.LaunchArrayAccessMode.STRICT

# Opt **in** to torch's sparse-tensor invariant checks, explicitly. torch validates nothing by
# default and says so once per process -- "Sparse invariant checks are implicitly disabled. Memory
# errors (e.g. SEGFAULT) will occur when operating on a sparse tensor which violates the
# invariants ... explicitly opt in or out" -- which is this suite's only warning on a CPU run. It
# is raised from inside pytorch3d's ``laplacian_matrices.py``, where it builds the COO tensors
# ``test_laplacian`` and ``test_energies`` compare against, so nothing ordito passes it changes
# whether it fires: the notice is about torch's global state, not about our input.
#
# *In* is the right half of the choice here, for the same reason STRICT is above. This suite holds
# nine reference libraries to their published behaviour and several of them are documented to be
# memory-unsafe on ordinary input, while pytorch3d specifically is pinned to upstream ``main`` --
# so a malformed tensor from a future pin should surface as an exception naming the invariant it
# broke, not as a SIGSEGV with no Python frame. Measured cost of the checks, interleaved in one
# process on a 40 962-vertex ``ops.laplacian``: **1.04-1.07x** of the sparse construction, in the
# handful of tests that build one.
#
# ``benchmarks/conftest.py`` opts *out* instead, and that is not an inconsistency: a timed
# pytorch3d row must not be charged for validation ordito's row does not perform.
torch.sparse.check_sparse_tensor_invariants.enable()


# The suite builds its own reference and input matrices through ``warp.sparse`` at several dtypes
# the package never reaches: ``bsr_from_triplets`` at float32 / float64 / int32 (the
# ``csr_from_triplets`` oracle), ``bsr_transposed`` at float32 / float64 (the ``csr_transpose``
# oracle), and ``bsr_copy`` within float64 and down to float32. Those kernels are Warp's generic
# ones, so each second dtype rebuilt its module mid-run on whichever test met it first; registered
# here, before anything launches, each module compiles once with all of them.
for _scalar in (wp.float32, wp.float64, wp.int32):
    register_warp_overload(_bsr_accumulate_triplet_values, _scalar)
for _scalar in (wp.float32, wp.float64):
    register_warp_overload(_bsr_transpose_values, _scalar)
    register_warp_overload(_bsr_assign_copy_blocks, wp.float64, dest_values=_scalar)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--cpu-blocks``, the opt-in pass that runs every lane of a CPU block."""
    parser.getgroup("ordito").addoption(
        "--cpu-blocks",
        action="store_true",
        default=False,
        help="set wp.config.enable_cpu_blocks, so a launch_tiled kernel runs all its lanes on the "
        "CPU device and the tiled reductions take their CUDA code path there. A correctness oracle "
        "for tiled kernels, not a speed setting; run it through `python -m tests.devices "
        "--cpu-blocks`.",
    )
    parser.getgroup("ordito").addoption(
        "--skip-device-agnostic",
        action="store_true",
        default=False,
        help="deselect the device_agnostic tests that do not use the device fixture: the static "
        "scans of the source tree, which give the same answer in every process. "
        "`python -m tests.devices` passes it to its CPU passes so they do not repeat the CUDA "
        "pass's scans.",
    )


def pytest_configure(config: pytest.Config) -> None:
    # Warp reads the flag at each launch, so setting it here -- before any test launches -- is
    # enough. On the CPU device ``wp.launch_tiled`` otherwise runs one lane per block, which hides
    # every kernel whose lanes partition work by anything but ``wp.block_dim()`` and keeps
    # ``_device.prefers_tiled_reduction``'s tiled half off the CPU.
    if config.getoption("--cpu-blocks"):
        wp.config.enable_cpu_blocks = True
    config.addinivalue_line(
        "markers",
        "parity(group, *libraries, benchmarked=..., reason=...): this test asserts ordito agrees "
        "with each named reference library for the benchmark group of that name. The gate in "
        "tests/test_parity.py requires one of these (or a noparity exemption in benchmarks/) for "
        "every benchmarked pair. Pass benchmarked=False with a written reason= where the pair is "
        "compared here but deliberately not timed.",
    )
    # The cut is at 15 s, measured, and it is four tests. On a full CPU-only run the whole
    # ``screened_poisson`` family is ~80 % of the wall clock and these four alone are ~two thirds of
    # it. Each is one depth-6 Poisson solve that costs under a second on CUDA, so skipping their
    # ``cpu`` half cuts the CPU run by more than half and loses no claim: the answers are
    # device-independent, ``test_poisson_cpu_matches_cuda`` pins the two devices to each other at
    # depth 4, and eleven more Poisson tests still run on CPU at depth 5 (``_poisson_depth``).
    #
    # Read the seconds in the marker as an order of magnitude, not a contract: the same four swing
    # widely between a full-suite run and their own ``-k`` selection, with the ranking inverted, so
    # a threshold cannot be re-derived by rerunning. What is stable is the shape: one depth-6
    # solve each.
    config.addinivalue_line(
        "markers",
        "slow_cpu(seconds): this test costs the stated measured seconds on the CPU device, so its "
        "``cpu`` parametrization is skipped unless --device=both. It still runs on CUDA, where the "
        "same test costs under a second. Reserved for the handful of tests that dominate a CPU "
        "run -- see the comment above for the measurements and why the cut sits where it does.",
    )
    config.addinivalue_line(
        "markers",
        "device_agnostic: a static scan of the source tree whose answer does not depend on the "
        "device or the process. --skip-device-agnostic deselects it unless it uses the device "
        "fixture (test_docstring_examples_run does), so a second device pass does not repeat it.",
    )


def _selected_devices(config: pytest.Config) -> list[str]:
    """
    Devices to parametrize the ``device`` fixture over, for **this process**.

    ``auto`` picks one device, matching ``benchmarks/conftest.py``. Both-device coverage is worth
    having -- it is what caught the ``warp.fem`` device leak in ``_screened_poisson_adaptive`` and
    the module-scope ``wp.array`` in ``test_grouping`` -- but it must not be bought *inside one
    process*, because **CPU work is ~36x slower once CUDA has been initialised**. Measured on one
    ``heat_signed_distance`` call, same mesh, same code, only ``CUDA_VISIBLE_DEVICES`` differing,
    and unchanged across all three ``launch_array_access_mode`` settings: so it is CUDA *presence*,
    not the launch-access guard, and the guard is free to stay ``STRICT``. Whole-suite
    consequence is roughly 4x, in-process ``--device=both`` against the two passes run separately;
    ``uv run python -m tests.devices`` is the runner that spawns them, and the CPU one sets
    ``CUDA_VISIBLE_DEVICES=""`` for exactly this reason.

    ``both`` stays meaningful and is not the slow trap it sounds like: it means "every device this
    process can see, and skip nothing". In a CUDA-hidden process that is precisely "all of CPU,
    including the ``slow_cpu`` tests", which is how the runner asks for a full CPU pass.
    """
    mode = str(config.getoption("--device"))
    if mode == "cuda":
        if not wp.is_cuda_available():
            raise pytest.UsageError("--device=cuda was requested but no CUDA device is available")
        return ["cuda:0"]
    if mode == "cpu":
        return ["cpu"]
    if mode == "auto":
        return ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
    return ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]


def _calls_getfixturevalue(code: CodeType) -> bool:
    """
    Whether this code object, or any code object nested in it, names ``getfixturevalue``.

    The recursion is the point. On Python 3.11 a comprehension compiles to its own code object, so
    ``[request.getfixturevalue(n) for n in names]`` puts the name in the *comprehension's*
    ``co_names`` and leaves the enclosing function's clean -- which is how a flat check silently
    missed ``test_combine.py::test_split_with_offsets_matches_split`` and let it run single-device.
    (3.12 inlines comprehensions and would have hidden the bug the other way, on a future upgrade.)
    """
    if "getfixturevalue" in code.co_names:
        return True
    return any(
        _calls_getfixturevalue(const) for const in code.co_consts if isinstance(const, CodeType)
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """
    Parametrize ``device`` over the selected devices, so a test id names the device it ran on.

    ``metafunc.parametrize`` only reaches a fixture in the test's *static* closure, and 206 tests
    reach ``device`` only through ``request.getfixturevalue(mesh_name)`` -- a lookup by string that
    pytest cannot see at collection time, so those tests would silently keep running on one device
    (and, with no ``device`` fixture to fall back on, fail outright with ``fixture 'device' not
    found``). Appending to ``metafunc.fixturenames`` puts it in the closure anyway, which is what
    lets a mesh fixture resolved later pick up the parametrized value. Keyed off the *bytecode*
    rather than off ``request`` being requested, because several tests take ``request`` for other
    reasons and doubling those buys nothing.
    """
    if "device" not in metafunc.fixturenames and _calls_getfixturevalue(metafunc.function.__code__):
        metafunc.fixturenames.append("device")
    if "device" in metafunc.fixturenames:
        metafunc.parametrize(
            "device", _selected_devices(metafunc.config), ids=lambda name: name.replace(":", "")
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Deselect repeated static scans, and skip the ``slow_cpu`` tests' ``cpu`` half.

    The scans go only under --skip-device-agnostic, and only those not reaching the ``device``
    fixture: a docstring example runs on the device and so belongs to every pass. The
    ``slow_cpu`` skip applies unless --device=both.
    """
    if config.getoption("--skip-device-agnostic"):
        dropped = [
            item
            for item in items
            if item.get_closest_marker("device_agnostic") is not None
            and "device" not in getattr(item, "fixturenames", ())
        ]
        if dropped:
            config.hook.pytest_deselected(items=dropped)
            kept = set(map(id, dropped))
            items[:] = [item for item in items if id(item) not in kept]
    if str(config.getoption("--device")) == "both":
        return
    for item in items:
        marker = item.get_closest_marker("slow_cpu")
        if marker is None:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is None or callspec.params.get("device") != "cpu":
            continue
        seconds = marker.args[0] if marker.args else "many"
        item.add_marker(
            pytest.mark.skip(
                reason=f"slow on CPU ({seconds} s measured); pass --device=both to run it"
            )
        )


# The fixture sets tests parametrize over, shared because they are a decision about *coverage* and
# not a local convenience: "the four that span closed/open and convex/non-convex" was restated in
# ten separate test files under three different names (``_MESHES``, ``_MESH_FIXTURES``,
# ``_LOOP_FIXTURES``), plus ``OPEN_MESHES`` in five and ``CLOSED_MESHES`` in four, which meant a
# fixture added to the set reached exactly one file. Import them as
# ``from tests.conftest import MESHES``; keep a local list only where it is genuinely a different
# set, and say in a comment why.
CLOSED_MESHES = ["sphere_irregular"]
"""
The hard closed fixture, which covers what a regular solid would: one component, genus 0.

A test whose subject is the topology -- several components, another Euler characteristic, an
inward-facing shell, flat coplanar faces -- names the fixture that has it (``cave_cube``,
``torus``, ``boy_surface`` ...) instead of widening this set.
"""

OPEN_MESHES = ["saddle_graded"]
"""
The hard open fixture: one rim, a disk, triangles of aspect ratio ~4 900.

A comparison tuned on well-shaped meshes meets an extreme operator here. A test about several
rims takes ``BOUNDARY_MESHES``; one about a specific rim shape (planar, two equal rims) names it.
"""

MESHES = CLOSED_MESHES + OPEN_MESHES
"""The default sweep: the hard closed fixture, then the hard open one."""

BOUNDARY_MESHES = ["saddle_graded", "torus_irregular_holes"]
"""
The open fixtures for boundary-loop tests: one long rim, then several unequal ones.

``saddle_graded``'s 268-edge rim, with two corners on a single face, and ``torus_irregular_holes``'
3-, 5- and 41-edge rims on a genus-1 surface, so loop ranking, pairing and pointer-jumping see one
loop and several, short and long, across several 1 024-item blocks and launches.
"""


@pytest.fixture
def device(request: pytest.FixtureRequest) -> str:
    """
    Fallback only: ``pytest_generate_tests`` parametrizes this name for every test that reaches it.

    Kept so a path that hook does not anticipate degrades to a single device instead of erroring
    with ``fixture 'device' not found``. If this body ever runs, some test is getting one device
    where it should be getting both -- which is a gap, not a failure, so it must not raise.
    """
    return _selected_devices(request.config)[-1]


@pytest.fixture
def icosahedron(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    icosahedron = tm.creation.icosahedron()
    icosahedron.apply_translation(translation=np.array([-1.0, 0.0, 2.0]))
    return icosahedron, trimesh_to_warp(icosahedron, device)


@pytest.fixture
def unit_box(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Build the unit cube: the sharp-featured convex solid, keyed to a right dihedral angle.

    Neither ``icosahedron`` nor ``cave_cube`` substitutes here -- a cube's 12 creases sit at exactly
    90 degrees with 6 flat face diagonals between them, which is what crease and seam tests are
    written against, and ``cave_cube`` is non-convex. Left untranslated, since several callers read
    coordinate signs to pick out one face.
    """
    box = tm.creation.box(extents=[1.0, 1.0, 1.0])
    return box, trimesh_to_warp(box, device)


@pytest.fixture
def icosphere(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Build a unit icosphere at ``subdivisions=3``: 642 vertices, 1 280 faces, closed and regular.

    The workhorse curved fixture, and the one whose absence was structural: do not
    hand-roll a mesh when a fixture will do, but the only closed fixture was ``icosahedron`` at 12
    vertices -- too coarse for anything that needs curvature -- so 47 tests across 21 files built
    this by hand. ``subdivisions=3`` is where 26 of them clustered.

    Untranslated, unlike ``icosahedron``: a test that wants the origin off the centroid should move
    it itself, and several of the callers this replaced depend on the sphere being centred.

    Function-scoped like every fixture here, so mutating ``mesh_tm.vertices`` in a test is safe.
    Session scoping is declined: the fixture is a trimesh build plus a ``wp.Mesh``, so all of the
    suite's fixture construction is a few percent of a run.

    **The 41 remaining inline ``tm.creation.icosphere`` sites are not migration debt**, which was
    checked rather than assumed: a classifier over all of them found **0** a fixture could take
    over. Every one either deforms the mesh (scales an axis, punches a hole, translates a copy),
    builds two or three of them, sits inside a private builder that has no fixture access, or pins
    ``trimesh_to_warp(mesh_tm, "cpu")`` deliberately because its reference has no device axis. So
    a call to ``tm.creation.icosphere`` in a test is not by itself a defect -- check what the test
    does with it before proposing the fixture.

    See Also
    --------
    ``icosphere_coarse``
        The same sphere at ``subdivisions=2``, where another 13 of the call sites sat.
    """
    sphere = tm.creation.icosphere(subdivisions=3, radius=1.0)
    return sphere, trimesh_to_warp(sphere, device)


@pytest.fixture
def icosphere_coarse(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Build the same sphere at ``subdivisions=2``: 162 vertices, 320 faces.

    A quarter the faces of ``icosphere`` and still genuinely curved. Prefer it wherever the test's
    claim does not need the resolution: a serial reference call over the larger sphere is the
    difference between a fast suite and a slow one, and most of these comparisons are about
    correctness rather than about mesh size.
    """
    sphere = tm.creation.icosphere(subdivisions=2, radius=1.0)
    return sphere, trimesh_to_warp(sphere, device)


@pytest.fixture
def half_torus(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    torus = tm.creation.torus(major_radius=1.0, minor_radius=0.5)
    half_torus = torus.slice_plane(plane_origin=np.zeros(3), plane_normal=np.array([1.0, 0.0, 0.0]))
    scale = 1 + np.exp(-half_torus.vertices[:, 1])
    half_torus.vertices *= scale[:, None]
    half_torus.apply_translation(translation=np.array([-1.0, 0.0, 2.0]))
    return half_torus, trimesh_to_warp(half_torus, device)


@pytest.fixture
def torus(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    torus = tm.creation.torus(major_radius=1.0, minor_radius=0.4)
    return torus, trimesh_to_warp(torus, device)


@pytest.fixture
def genus_two(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    left = tm.creation.torus(major_radius=1.0, minor_radius=0.35)
    right = tm.creation.torus(major_radius=1.0, minor_radius=0.35)
    right.apply_translation(translation=np.array([1.8, 0.0, 0.0]))
    mesh = tm.boolean.union([left, right])
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def cave_cube(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    mesh = tm.boolean.difference(
        [tm.creation.box(extents=[1.0, 1.0, 1.0]), tm.creation.box(extents=[0.1, 0.1, 0.1])]
    )
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def hemisphere(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    sphere = tm.creation.icosphere(subdivisions=2, radius=1.0)
    hemisphere = sphere.slice_plane(
        plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]), cap=False
    )
    hemisphere.merge_vertices()
    rotation = tm.transformations.rotation_matrix(
        np.deg2rad(45.0), direction=np.array([1.0, 1.0, 0.0])
    )
    rotation[:3, 3] = np.array([-1.0, 0.0, 2.0])
    hemisphere.apply_transform(rotation)
    return hemisphere, trimesh_to_warp(hemisphere, device)


@functools.cache
def saddle_graded_arrays(
    resolution: int = 34, exponent: float = 3.4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``saddle_graded``'s ``(n_vertices, 3)`` vertices and ``(n_faces, 3)`` faces.

    A ``resolution x resolution`` grid lifted onto the saddle ``z = 0.35 (x^2 - 0.6 y^2)``, its
    spacing graded along ``x`` as ``|t| ** exponent``. An even side puts the two finest columns
    either side of ``x = 0`` (gap ``2 / (resolution - 1) ** exponent``) against a ``y`` spacing of
    ``2 / (resolution - 1)``, so the worst aspect ratio (longest edge over its altitude) grows as
    ``(resolution - 1) ** (exponent - 1)``: the default ``34 x 34`` at 3.4 reaches 4 761 with
    1 156 vertices, 2 178 faces (6 534 halfedges, several 1 024-item blocks) and a 132-edge rim
    (several pointer-jumping launches). ``saddle_graded_large_arrays`` is the ``68 x 68`` at 3
    (aspect 4 858, 4 624 vertices) the size-dependent regression tests need.

    The vertices are rounded to ``float32`` and held as ``float64``, so a reference library and
    ordito's ``float32`` buffers see the same geometry: on operators this ill-conditioned the
    rounding of the input alone is not negligible.
    """
    k = resolution
    step = np.linspace(-1.0, 1.0, k)
    u, v = np.meshgrid(np.sign(step) * np.abs(step) ** exponent, step, indexing="ij")
    vertices = np.column_stack((u.ravel(), v.ravel(), 0.35 * (u * u - 0.6 * v * v).ravel()))
    i, j = np.meshgrid(np.arange(k - 1), np.arange(k - 1), indexing="ij")
    corner = (i * k + j).ravel()
    faces = np.vstack(
        (
            np.column_stack((corner, corner + k, corner + k + 1)),
            np.column_stack((corner, corner + k + 1, corner + 1)),
        )
    )
    return vertices.astype(np.float32).astype(np.float64), faces.astype(np.int64)


def saddle_graded_large_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Return the ``68 x 68`` graded saddle: for the two defects that need its size, and only them.

    The heat method's settle rule stopped on round-off here (before the backward-error fallback the
    distance sat 34x and 7x over the parity test's mean and max bounds, where the default fixture's
    pre-fix error is under them), and ``filter_implicit_fairing``'s first solve is slow enough here
    to take the refactorization path. Everything else runs on the smaller default.
    """
    return saddle_graded_arrays(68, 3.0)


@pytest.fixture
def saddle_graded(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Graded saddle: an open disk whose triangles reach an aspect ratio of ~4 900.

    One rim, every vertex manifold, consistently wound, no degenerate face -- an input every
    open-mesh function accepts -- yet its cotangent operators are ill-conditioned in places. Its
    68 x 68 form exposed two silent wrong answers no other fixture reached: the heat method's
    settled CG iterate (4.4 % mean, 38 % worst of the range off ``igl.exact_geodesic``) and the
    ``k = 2`` harmonic map (29 % of the range off a SciPy direct solve); this 34 x 34 form still
    fails the pre-fix ``k = 2`` solve and the pre-2026-10-07 needle numerics (first-corner normals,
    law-of-cosines cotangents) by the same tests, at a quarter of the cost. Built
    without trimesh's processing, so the face buffer is the grid's own.
    """
    vertices_np, faces_np = saddle_graded_arrays()
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


def _periodic_delaunay(uv: np.ndarray, period: tuple[float, float]) -> np.ndarray:
    """Delaunay of points on a flat torus: tile 3 x 3, keep the faces whose centroid is central."""
    from scipy.spatial import Delaunay

    n = uv.shape[0]
    shifts = [(i * period[0], j * period[1]) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    tiled = np.vstack([uv + np.array(shift) for shift in shifts])
    faces = Delaunay(tiled).simplices
    centroids = tiled[faces].mean(axis=1)
    central = (
        (centroids[:, 0] >= 0.0)
        & (centroids[:, 0] < period[0])
        & (centroids[:, 1] >= 0.0)
        & (centroids[:, 1] < period[1])
    )
    return faces[central] % n


def _scramble(
    vertices: np.ndarray, faces: np.ndarray, rng: np.random.Generator, *, placed: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffle vertex and face order, rotate each face's corners, place the mesh off-axis.

    ``placed=False`` skips the random placement, for a fixture that applies the round frame.
    """
    order = rng.permutation(vertices.shape[0])
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    faces = rank[faces][rng.permutation(faces.shape[0])]
    shift = rng.integers(0, 3, faces.shape[0])
    faces = np.take_along_axis(faces, (np.arange(3)[None, :] + shift[:, None]) % 3, axis=1)
    vertices = vertices[order]
    if placed:
        rotation = tm.transformations.random_rotation_matrix(rng.random(3))[:3, :3]
        vertices = 1.7 * vertices @ rotation.T + np.array([0.3, -2.1, 1.4])
    return vertices.astype(np.float32).astype(np.float64), faces.astype(np.int64)


ROUND_FRAME_SCALE = 1.7
ROUND_FRAME_ROTATION = tm.transformations.random_rotation_matrix(np.array([0.2, 0.6, 0.9]))[:3, :3]
ROUND_FRAME_SHIFT = np.array([0.3, -2.1, 1.4])


def round_frame_coordinates(vertices: np.ndarray) -> np.ndarray:
    """Map ``sphere_round`` / ``torus_round`` positions back to their surface's canonical frame."""
    return (
        (np.asarray(vertices, dtype=np.float64) - ROUND_FRAME_SHIFT)
        @ ROUND_FRAME_ROTATION
        / (ROUND_FRAME_SCALE)
    )


def _round_frame_placed(vertices: np.ndarray) -> np.ndarray:
    """Place canonical positions in the round frame (rotated, scaled 1.7, off-origin) as float32."""
    placed = ROUND_FRAME_SCALE * vertices @ ROUND_FRAME_ROTATION.T + ROUND_FRAME_SHIFT
    return placed.astype(np.float32).astype(np.float64)


def _bumpy_sphere(
    rng: np.random.Generator, n_points: int, *, squeeze: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return an unscrambled bumpy star-shaped sphere: the hull of ``n_points`` random directions.

    Wound outward. Radius ``1 + 0.35 sin 3 theta cos 2 phi + 0.15 cos 5 theta`` (between 0.5 and
    1.5), longitude squeezed by ``phi - squeeze sin phi``.
    """
    from scipy.spatial import ConvexHull

    directions = rng.normal(size=(n_points, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    faces = ConvexHull(directions).simplices
    corners = directions[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inward = np.einsum("ij,ij->i", normals, corners.mean(axis=1)) < 0.0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    phi = phi - squeeze * np.sin(phi)
    radius = 1.0 + 0.35 * np.sin(3.0 * theta) * np.cos(2.0 * phi) + 0.15 * np.cos(5.0 * theta)
    vertices = radius[:, None] * np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    return vertices, faces


@functools.cache
def sphere_irregular_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``sphere_irregular``'s ``(n_vertices, 3)`` vertices and ``(n_faces, 3)`` faces.

    The convex hull of 500 random directions -- a Delaunay triangulation of the sphere, so
    vertex degree runs 3-12 and nothing is a subdivision pattern -- pushed out along each direction
    to a bumpy radius ``1 + 0.35 sin 3 theta cos 2 phi + 0.15 cos 5 theta``, which is star-shaped
    (so still embedded) but non-convex, with negative curvature at ~40 % of the vertices. The
    longitude is squeezed by ``phi - 0.97 sin phi`` first, a 33x compression near ``phi = 0`` that
    makes the triangles there needles (aspect ratio up to ~300); nearly three quarters of all faces
    are obtuse. Order and placement are scrambled as in ``torus_irregular_arrays``.
    """
    rng = np.random.default_rng(5)
    vertices, faces = _bumpy_sphere(rng, 500, squeeze=0.97)
    return _scramble(vertices, faces, rng)


@pytest.fixture
def sphere_irregular(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Irregular sphere: the hard closed fixture -- one component, genus 0, nothing regular about it.

    What a regular solid (``icosahedron``: twelve degree-5 vertices, all faces equilateral) cannot
    reach: varying degree, negative cotangent weights, curvature of both signs, needles, ~3 000
    halfedges (several 1 024-item blocks), and an index order with no spatial meaning.
    """
    vertices_np, faces_np = sphere_irregular_arrays()
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


def _cut_sphere_irregular(
    offsets: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``sphere_irregular`` cut by planes, as ``(n_vertices, 3)`` and ``(n_faces, 3)`` arrays.

    Each ``(offset, sign)`` keeps the side ``sign * (x - c) . n >= offset`` of a fixed oblique
    plane normal ``n`` through the centroid ``c``. The cut adds rim vertices next to existing ones,
    so the rim carries slivers of its own besides the sphere's needles. Rounded to ``float32``.
    """
    vertices, faces = sphere_irregular_arrays()
    mesh = tm.Trimesh(vertices, faces, process=False)
    centre = vertices.mean(axis=0)
    normal = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    for offset, sign in offsets:
        mesh = mesh.slice_plane(
            plane_origin=centre + sign * offset * normal, plane_normal=sign * normal, cap=False
        )
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices).astype(np.float32).astype(np.float64), np.asarray(mesh.faces)


@functools.cache
def sphere_irregular_cap_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return ``sphere_irregular_cap``'s vertices and faces: one planar 91-edge rim."""
    return _cut_sphere_irregular(((0.1, 1.0),))


@functools.cache
def sphere_irregular_band_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return ``sphere_irregular_band``'s vertices and faces: two planar rims, 99 and 78 edges."""
    return _cut_sphere_irregular(((-0.5, 1.0), (-0.6, -1.0)))


@pytest.fixture
def sphere_irregular_cap(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    ``sphere_irregular`` cut by one plane: an open disk with a planar rim, the hard ``hemisphere``.

    For tests whose premise is a planar boundary (a fill that must cover one flat region).
    """
    vertices_np, faces_np = sphere_irregular_cap_arrays()
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def sphere_irregular_band(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Return ``sphere_irregular`` between two parallel planes: an annulus, the hard ``half_torus``.

    Its two planar rims have different lengths, 99 and 78 edges.
    """
    vertices_np, faces_np = sphere_irregular_band_arrays()
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


def _cut_convex_irregular(
    offsets: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a convex irregular ellipsoid cut by planes, as ``(n_vertices, 3)`` / ``(n_faces, 3)``.

    The hull of 500 random directions, longitude-squeezed as ``sphere_irregular`` is (needles to
    aspect ~400) but without its bumps, so the body is convex and every planar section of it is a
    convex polygon. Cut as ``_cut_sphere_irregular`` cuts; rounded to ``float32``.
    """
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(13)
    directions = rng.normal(size=(500, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    phi = phi - 0.97 * np.sin(phi)
    points = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    ) * np.array([1.3, 1.0, 0.8])
    mesh = tm.Trimesh(points, ConvexHull(points).simplices, process=False)
    mesh.fix_normals()
    normal = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    for offset, sign in offsets:
        mesh = mesh.slice_plane(
            plane_origin=sign * offset * normal, plane_normal=sign * normal, cap=False
        )
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices).astype(np.float32).astype(np.float64), np.asarray(mesh.faces)


@pytest.fixture
def convex_irregular_cap(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Return a convex irregular body cut by one plane: one *convex* planar rim, needles intact.

    Only for a premise the bumpy ``sphere_irregular_cap`` breaks: a filler that must not fold a
    planar rim needs it convex (pymeshfix's ear clipping folds a non-convex one).
    """
    vertices_np, faces_np = _cut_convex_irregular(((0.1, 1.0),))
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def convex_irregular_band(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """Return the same body between two planes: two convex planar rims, 62 and 79 edges."""
    vertices_np, faces_np = _cut_convex_irregular(((-0.4, 1.0), (-0.5, -1.0)))
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


@functools.cache
def sphere_well_shaped_arrays(
    bumps: float = 1.0, *, placed: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``sphere_well_shaped``'s ``(n_vertices, 3)`` vertices and ``(n_faces, 3)`` faces.

    ``sphere_irregular``'s bumpy surface over 400 dart-thrown directions (no two closer than 0.7 of
    the mean spacing) instead of 500 random ones, and no squeeze: irregular -- degree 4-8, a third
    of the faces obtuse, curvature of both signs, scrambled -- but with no needle (aspect ratio at
    most ~6), so ``float32`` edge lengths still determine every triangle. ``bumps=0`` keeps the
    sampling and lays it on the exact unit sphere (``sphere_round``, placed in the
    round frame).
    """
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(5)
    spacing = 0.7 * np.sqrt(4.0 * np.pi / 400)
    directions: list[np.ndarray] = []
    while len(directions) < 400:
        candidate = rng.normal(size=3)
        candidate /= np.linalg.norm(candidate)
        if all(np.linalg.norm(candidate - kept) > spacing for kept in directions):
            directions.append(candidate)
    points = np.array(directions)
    faces = ConvexHull(points).simplices
    corners = points[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inward = np.einsum("ij,ij->i", normals, corners.mean(axis=1)) < 0.0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])
    radius = 1.0 + bumps * (
        0.35 * np.sin(3.0 * theta) * np.cos(2.0 * phi) + 0.15 * np.cos(5.0 * theta)
    )
    vertices = radius[:, None] * np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    return _scramble(vertices, faces, rng, placed=placed)


@functools.cache
def sphere_well_shaped_open_arrays() -> tuple[np.ndarray, np.ndarray]:
    """``sphere_well_shaped`` with the faces above an oblique plane deleted: one jagged rim."""
    vertices, faces = sphere_well_shaped_arrays()
    centres = vertices[faces].mean(axis=1)
    normal = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    keep = (centres - vertices.mean(axis=0)) @ normal < 0.2
    mesh = tm.Trimesh(vertices, faces[keep], process=False)
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


@functools.cache
def torus_well_shaped_arrays(
    minor: float = 0.45, bumps: float = 0.25, *, placed: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``torus_well_shaped``'s ``(n_vertices, 3)`` vertices and ``(n_faces, 3)`` faces.

    ``torus_irregular``'s bumpy torus over 400 points dart-thrown on its flat parameter torus (no
    two closer than 0.7 of the mean spacing) and without the squeeze: genus 1, irregular degree,
    scrambled, but needle-free, for the same ``float32``-length premise as ``sphere_well_shaped``.
    ``minor=0.4, bumps=0`` lays the sampling on an exact torus of revolution (``torus_round``,
    placed in the round frame).
    """
    rng = np.random.default_rng(7)
    major = 1.0
    period = np.array((2.0 * np.pi * major, 2.0 * np.pi * minor))
    spacing = 0.7 * np.sqrt(period.prod() / 400)
    points: list[np.ndarray] = []
    while len(points) < 400:
        candidate = rng.random(2) * period
        if all(
            np.linalg.norm((candidate - kept + period / 2) % period - period / 2) > spacing
            for kept in points
        ):
            points.append(candidate)
    uv = np.array(points)
    faces = _periodic_delaunay(uv, (float(period[0]), float(period[1])))
    u = uv[:, 0] / major
    v = uv[:, 1] / minor
    radius = minor * (1.0 + bumps * np.sin(3.0 * u) * np.cos(2.0 * v))
    ring = major + radius * np.cos(v)
    vertices = np.column_stack((ring * np.cos(u), ring * np.sin(u), radius * np.sin(v)))
    return _scramble(vertices, faces, rng, placed=placed)


@pytest.fixture
def torus_well_shaped(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """Irregular, needle-free genus-1 fixture; see ``sphere_well_shaped`` for its premise."""
    vertices_np, faces_np = torus_well_shaped_arrays()
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def sphere_well_shaped(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Irregular but needle-free closed fixture, for a premise ``sphere_irregular``'s needles break.

    Every use names the premise: a path that re-derives triangles from ``float32`` edge lengths
    (the intrinsic Laplacians), which a needle's rounded lengths do not determine.
    """
    vertices_np, faces_np = sphere_well_shaped_arrays()
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def sphere_round(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Return ``sphere_well_shaped``'s irregular sampling on the exact unit sphere.

    For an analytic oracle only: every point is umbilic with curvature 1, which no bumpy fixture
    has. Degree 4-8, scrambled, dart-thrown, so it is not an icosphere's regular lattice.
    """
    vertices_np, faces_np = sphere_well_shaped_arrays(bumps=0.0, placed=False)
    mesh = tm.Trimesh(_round_frame_placed(vertices_np), faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def torus_round(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Return ``torus_well_shaped``'s irregular sampling on the exact torus ``R = 1``, ``r = 0.4``.

    For an analytic oracle only: its lines of curvature are the meridians and parallels and its two
    principal curvatures differ by at least 1.79 everywhere (no umbilic point).
    """
    vertices_np, faces_np = torus_well_shaped_arrays(minor=0.4, bumps=0.0, placed=False)
    mesh = tm.Trimesh(_round_frame_placed(vertices_np), faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def sphere_well_shaped_open(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """Return the open counterpart of ``sphere_well_shaped``: same premise, one boundary loop."""
    vertices_np, faces_np = sphere_well_shaped_open_arrays()
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


@functools.cache
def sphere_irregular_hollow_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``sphere_irregular_hollow``'s ``(n_vertices, 3)`` vertices and ``(n_faces, 3)`` faces.

    ``sphere_irregular``'s outer surface around a second bumpy sphere of 200 directions, scaled by
    0.3 (radius at most 0.45 against the outer surface's at least 0.5) and wound inward: a solid
    shell with a cavity, two components. Both shells are scrambled together.
    """
    rng = np.random.default_rng(5)
    outer_vertices, outer_faces = _bumpy_sphere(rng, 500, squeeze=0.97)
    inner_vertices, inner_faces = _bumpy_sphere(np.random.default_rng(11), 200, squeeze=0.9)
    vertices = np.concatenate((outer_vertices, 0.3 * inner_vertices))
    faces = np.concatenate((outer_faces, inner_faces[:, [0, 2, 1]] + outer_vertices.shape[0]))
    return _scramble(vertices, faces, rng)


@pytest.fixture
def sphere_irregular_hollow(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Irregular hollow sphere: a solid with a cavity, the hard counterpart of ``cave_cube``.

    A query in the cavity is outside the solid with surface all around it, and its nearest face's
    normal points away from the outer shell's; a sign rule or a containment test that assumes one
    outer surface fails there.
    """
    vertices_np, faces_np = sphere_irregular_hollow_arrays()
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


@functools.cache
def torus_irregular_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``torus_irregular``'s ``(n_vertices, 3)`` vertices and ``(n_faces, 3)`` faces.

    500 random points on a flat torus, Delaunay-triangulated in the periodic parameter domain, so
    nothing is a grid: vertex degree runs 3-10. Mapped onto a bumpy torus whose angle ``u`` is
    squeezed by ``u - 0.97 sin u``, a 33x compression near ``u = 0`` that turns the Delaunay
    triangles there into needles (aspect ratio up to ~540) and leaves two thirds of all faces
    obtuse, so cotangent weights go negative. Vertex order, face order and each face's starting
    corner are shuffled, and the mesh is rotated off the axes and translated off the origin.
    """
    rng = np.random.default_rng(7)
    major, minor = 1.0, 0.45
    period = (2.0 * np.pi * major, 2.0 * np.pi * minor)
    uv = rng.random((500, 2)) * np.array(period)
    faces = _periodic_delaunay(uv, period)
    u = uv[:, 0] / major
    v = uv[:, 1] / minor
    u = u - 0.97 * np.sin(u)
    radius = minor * (1.0 + 0.25 * np.sin(3.0 * u) * np.cos(2.0 * v))
    ring = major + radius * np.cos(v)
    vertices = np.column_stack((ring * np.cos(u), ring * np.sin(u), radius * np.sin(v)))
    return _scramble(vertices, faces, rng)


@pytest.fixture
def torus_irregular(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Irregular torus: closed, genus 1, no grid structure, needles and obtuse faces, shuffled order.

    The hard closed fixture: what a regular solid (``icosahedron``: twelve degree-5 vertices, all
    faces equilateral) cannot reach -- varying degree, negative cotangent weights, a curvature that
    changes sign, ~3 000 halfedges (several 1 024-item blocks), and index order with no spatial
    meaning.
    """
    vertices_np, faces_np = torus_irregular_arrays()
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


@functools.cache
def torus_irregular_holes_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``torus_irregular_holes``' vertices and faces: ``torus_irregular`` with three holes.

    The holes are one face (a 3-edge rim), one vertex's star (its vertex removed with it) and a
    wide patch on the outer side, so the rims run 3, 7 and 24 edges; the surface stays edge- and
    vertex-manifold and genus 1.
    """
    vertices, faces = torus_irregular_arrays()
    centroids = vertices[faces].mean(axis=1)
    scale = np.ptp(vertices, axis=0).max()
    anchors = vertices[np.argsort(np.linalg.norm(vertices - vertices.mean(axis=0), axis=1))]
    wide_centre, star_point, single_point = anchors[-1], anchors[0], anchors[vertices.shape[0] // 2]
    wide = np.linalg.norm(centroids - wide_centre, axis=1) < 0.3 * scale
    star_vertex = int(np.argmin(np.linalg.norm(vertices - star_point, axis=1)))
    star = (faces == star_vertex).any(axis=1)
    single = np.zeros(faces.shape[0], dtype=bool)
    single[int(np.argmin(np.linalg.norm(centroids - single_point, axis=1)))] = True
    kept = faces[~(wide | star | single)]
    used = np.unique(kept)
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return vertices[used], remap[kept]


@pytest.fixture
def torus_irregular_holes(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    ``torus_irregular`` with three holes: the hard open fixture with several rims of unequal size.

    Open but not a disk (genus 1, three boundary loops of very different lengths), on the same
    irregular, needle-carrying, shuffled triangulation.
    """
    vertices_np, faces_np = torus_irregular_holes_arrays()
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def boy_surface(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Boy's surface: closed, watertight and **non-orientable**, with Euler characteristic 1.

    The only fixture of its class. Every other closed mesh in this file is orientable with an even
    characteristic, so the ``False`` branch of ``is_orientable`` / ``face_orientation_bits`` and the
    impossible branch of ``make_winding_consistent`` are unreachable without it.
    """
    vertices_wp, faces_wp = od.creation.parametric_surface("boy", device=device)
    mesh = warp_to_trimesh(vertices_wp, faces_wp)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def mobius(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """Moebius band: non-orientable *with* a boundary — one loop of 78 edges, and χ = 0."""
    vertices_wp, faces_wp = od.creation.parametric_surface("mobius", device=device)
    mesh = warp_to_trimesh(vertices_wp, faces_wp)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def bohemian_dome(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """Build a closed genus-1 surface that intersects itself: watertight, orientable, χ = 0."""
    vertices_wp, faces_wp = od.creation.parametric_surface("bohemian_dome", device=device)
    mesh = warp_to_trimesh(vertices_wp, faces_wp)
    return mesh, trimesh_to_warp(mesh, device)


# Two pathological tori, built by ``meshlib``'s own generators rather than by hand: a closed surface
# that passes through itself, and one that is eight disconnected pieces. The only fixtures here that
# are *deliberately broken*, and fixed low resolutions, so each is a constant. The same family also
# generates a spiked torus and an undercut one (``makeTorusWithSpikes`` / ``makeTorusWithUndercut``)
# which are not fixtures yet because nothing consumes them; add them with their first caller.
#
# This is the one place in ``tests/`` where a fixture's geometry comes from MeshLib. That is
# deliberate and it is allowed -- a test dependency is what MeshLib is licensed for, and nothing
# under ``ordito/`` names it -- but it goes through ``meshlib_to_trimesh``, which packs the mesh:
# reading ``getNumpyFaces`` off an unpacked one returns rows of ``[0, 0, 0]``.
_TORUS_PRIMARY_RADIUS = 1.0
_TORUS_RESOLUTION = 16


def _torus_fixture(mesh_ml: mm.Mesh, device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """Pack a generated MeshLib torus and hand it over in both of the suite's usual forms."""
    mesh = meshlib_to_trimesh(mesh_ml)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def torus_self_intersecting(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Build a genus-1 torus whose tube is wider than its hole, so the surface passes through itself.

    512 faces, edge-manifold and consistently wound, and **not** a repairable mesh: the inner wall
    crosses the outer one, which is the input ``face_self_intersecting_mask`` and any
    self-intersection repair need and which no clean fixture can provide.
    [`bohemian_dome`][tests.conftest.bohemian_dome] is the other self-intersecting closed surface
    here; that one intersects along a curve by construction, where this one interpenetrates in a
    band.
    """
    return _torus_fixture(
        mm.makeTorusWithSelfIntersections(
            _TORUS_PRIMARY_RADIUS, 0.5, _TORUS_RESOLUTION, _TORUS_RESOLUTION
        ),
        device,
    )


@pytest.fixture
def torus_spikes(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Build a torus whose tube radius alternates, leaving needle-like vertices on the wide rings.

    512 faces, closed and edge-manifold, with the *only* genuinely spiky vertices in the suite: at a
    1.5-pi angle-sum threshold twelve of its 256 vertices fail, and five at pi. Every clean fixture
    has none, so a spike detector or a spike repair tested on one is asserting an empty answer --
    which is the trap ``test_ears`` fell into.

    The inner and outer tube radii are what make the needles: 0.1 against 0.5 means alternate rings
    sit far apart radially while their neighbours along the tube are close, so the cone at a wide
    ring's vertex closes up.
    """
    return _torus_fixture(
        mm.makeTorusWithSpikes(
            _TORUS_PRIMARY_RADIUS, 0.1, 0.5, _TORUS_RESOLUTION, _TORUS_RESOLUTION
        ),
        device,
    )


@pytest.fixture
def torus_components(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Build a torus broken into **eight** disconnected open pieces, 256 faces in total.

    The multi-component input for ``combine.split``, the component labellings in
    [`ordito.graph`][ordito.graph] and anything that has to survive a mesh which is not one
    surface. Every piece has a boundary, so it is also the fixture where a per-component *and*
    per-loop answer both have several instances to be wrong about.
    """
    return _torus_fixture(
        mm.makeTorusWithComponents(
            _TORUS_PRIMARY_RADIUS, 0.1, _TORUS_RESOLUTION, _TORUS_RESOLUTION
        ),
        device,
    )


@pytest.fixture
def sliver_patch(
    device: str,
) -> tuple[np.ndarray, np.ndarray, wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Build a patch with one zero-area triangle: collinear corners, the triangle inequality tight.

    Mollification and the robust Laplacian are only interesting on a mesh that needs them: the
    plain cotangent Laplacian has no cotangent for the collinear face and drops its couplings, the
    robust one must not. Returned as both NumPy (``float64``, for the CPU references) and Warp
    (``float32``) -- the corners are exact in both, so the two sides see the same mesh.

    The middle corner sat at ``y = 1e-9`` until the cotangents were formed as ``dot / |cross|``:
    that is a sliver of height 1e-9, not a degenerate face, and its true cotangent (-1.25e8) is
    what the plain operator now returns, where the law of cosines on ``float32`` lengths -- whose
    rounding broke the inequality -- had read it as zero.
    """
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=np.float64
    )
    faces_np = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 3]], dtype=np.int32)
    return (
        vertices_np,
        faces_np,
        wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device=device),
        wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device),
    )


@pytest.fixture
def parabolic_lattice(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Build a regular 21x21 lattice on ``z = 0.3 * x^2``: curvature aligned with the grid directions.

    The **axis-aligned** fixture, and the curvature-side counterpart to ``unit_box``. Every other
    curved fixture here is irregular (``icosahedron``, ``icosphere``) or has its parameter lines
    running obliquely to its curvature (``half_torus``, ``torus``), so nothing else in the suite
    reaches the case where a per-vertex tangent frame built from a ring halfedge lands *on* a
    principal direction. That case is not exotic -- it is what a scanned or CAD surface sampled on a
    lattice looks like -- and it is where a shape operator comes out exactly diagonal, so an
    eigen-decomposition that reads a fixed row of ``m - lam*I`` silently returns the wrong axis.
    Measured on this mesh: 41 of 441 vertices on ``cpu`` and 22 on ``cuda:0``, against 0 on all four
    of the fixtures above.

    A parabolic cylinder rather than a plane, because the two principal curvatures must *differ* --
    at an umbilic point any tangent pair is a valid answer and a direction test is vacuous. It is
    developable (``k2 == 0`` everywhere), which is also what makes the expected answer readable by
    hand: ``PD1`` runs along x and ``PD2`` along y.

    !!! warning

        Both triangles of every quad cell are right-angled, so every diagonal's cotangent weight is
        exactly zero -- the same caveat [`creation.grid`][ordito.creation.grid] carries. That makes
        this fixture unusable as input to a ``potpourri3d`` connection-Laplacian comparison, which
        drops such an edge's phase entirely (see ``tests/test_tangent.py``).
    """
    axis_np = np.linspace(-1.0, 1.0, 21)
    grid_x_np, grid_y_np = np.meshgrid(axis_np, axis_np, indexing="ij")
    vertices_np = np.stack(
        [grid_x_np.ravel(), grid_y_np.ravel(), (0.3 * grid_x_np**2).ravel()], axis=1
    )
    # Lower-left corner of each quad cell; both of its triangles are wound counter-clockwise seen
    # from +z, so the surface normals point up and the fit's sign convention is unambiguous.
    corner_np = (np.arange(20)[:, None] * 21 + np.arange(20)[None, :]).ravel()
    faces_np = np.concatenate(
        [
            np.stack([corner_np, corner_np + 21, corner_np + 1], axis=1),
            np.stack([corner_np + 1, corner_np + 21, corner_np + 22], axis=1),
        ]
    )
    lattice = tm.Trimesh(vertices_np, faces_np, process=False)
    return lattice, trimesh_to_warp(lattice, device)


@pytest.fixture
def t_vertex_patch() -> tuple[np.ndarray, np.ndarray]:
    """
    Two quads stitched at different resolutions, so the left one carries a T-vertex.

    Vertex 4 sits on the interior of the edge ``(1, 2)`` of the right quad's triangulation, which is
    exactly a T-junction: the triangle ``(1, 2, 4)`` is a sliver whose apex is on its own long edge.
    Flipping ``(1, 2)`` to ``(3, 4)`` removes it without moving a vertex.
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 2.0, 0.0],  # 2
            [0.0, 2.0, 0.0],  # 3
            [1.0, 1.0, 0.02],  # 4 -- barely off the (1, 2) edge, and off-plane so a flip is legal
            [2.0, 0.0, 0.0],  # 5
            [2.0, 2.0, 0.0],  # 6
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 3], [1, 2, 3], [1, 5, 4], [4, 5, 6], [4, 6, 2], [1, 4, 2]], dtype=np.int32
    )
    return vertices, faces


@pytest.fixture
def folded_patch() -> tuple[np.ndarray, np.ndarray]:
    """
    Build a flat two-triangle quad plus a third triangle folded back on top of it.

    Face 2 shares edge ``(1, 3)`` with face 1 and lies almost in the same plane with the *opposite*
    normal, so the dihedral there is ~179 degrees. Every edge still has at most two faces — a third
    face on the folded edge would make it non-manifold, which ``face_adjacency`` drops entirely and
    which would make this fixture measure nothing.
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.1, 0.9, 0.02],  # the folded apex: back over the quad
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2], [3, 1, 4]], dtype=np.int32)
    return vertices, faces


@pytest.fixture
def force_halfedge_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Route every halfedge pairing given a vertex count through the per-vertex edge buckets.

    The buckets run on CUDA from a halfedge-count gate and never on the CPU device; this opens
    both gates, so a test runs the bucket path on both devices and at fixture size. A call without
    a vertex count still takes the key sort.
    """
    monkeypatch.setattr(od.halfedge, "_BUCKETED_PAIRING_ON_CPU", True)
    monkeypatch.setattr(od.halfedge, "_BUCKETED_MATES_FROM_HALFEDGES", 0)


CACHED_TRIMESH_KEYS = frozenset(
    name for name, value in vars(od.Trimesh).items() if isinstance(value, _CachedProperty)
)


def populate_cache(mesh: od.Trimesh) -> od.Trimesh:
    """
    Force *every* cached property the fixture supports, not only the ones a caller expects.

    Deriving the set from the class rather than from a carry set is what makes the cache-carrying
    gates in ``test_transform.py`` and ``test_mesh.py`` bite on future edits: a key populated but
    deliberately not carried costs nothing today, and the moment someone adds it to a stratum it
    starts being compared against recomputation. Populating only the carried keys would make every
    such addition vacuously correct.
    """
    for key in CACHED_TRIMESH_KEYS:
        try:
            getattr(mesh, key)
        except (ValueError, RuntimeError):
            pass  # `warp_mesh` on an empty mesh, `halfedge_twins` on a non-manifold one
    return mesh
