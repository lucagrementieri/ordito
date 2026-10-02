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

from types import CodeType  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import trimesh as tm  # noqa: E402
import warp as wp  # noqa: E402
from meshlib import mrmeshpy as mm  # noqa: E402

import ordito as od  # noqa: E402
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


def pytest_configure(config: pytest.Config) -> None:
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


def _selected_devices(config: pytest.Config) -> list[str]:
    """
    Devices to parametrize the ``device`` fixture over, for **this process**.

    ``auto`` picks one device, matching ``benchmarks/conftest.py``. Both-device coverage is worth
    having -- it is what caught the ``warp.fem`` device leak in ``_screened_poisson_adaptive`` and
    the module-scope ``wp.array`` in ``test_grouping`` -- but it must not be bought *inside one
    process*, because **CPU work is ~36x slower once CUDA has been initialised**. Measured on one
    ``heat_signed_distance`` call, same mesh, same code, only ``CUDA_VISIBLE_DEVICES`` differing,
    and unchanged across all three ``launch_array_access_mode`` settings: so it is CUDA *presence*,
    not section 8's launch-access guard, and the guard is free to stay ``STRICT``. Whole-suite
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
    """Drop the ``cpu`` parametrization of the ``slow_cpu`` tests unless --device=both."""
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
CLOSED_MESHES = ["icosahedron", "cave_cube"]
"""Watertight, and one convex and one not -- ``cave_cube`` is a hollow non-convex shell."""

OPEN_MESHES = ["hemisphere", "half_torus"]
"""Curved with a boundary: one rim on ``hemisphere``, two on ``half_torus``."""

MESHES = CLOSED_MESHES + OPEN_MESHES
"""The default four-fixture sweep: closed and open, convex and not, in that order."""


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

    The workhorse curved fixture, and the one whose absence was structural: section 6 says not to
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
# reading ``getNumpyFaces`` off an unpacked one returns rows of ``[0, 0, 0]`` (CLAUDE.md
# section 7.6).
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
    which is the trap ``test_ears`` fell into (CLAUDE.md section 7.4).

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
    Build a patch with one near-zero-area triangle, thin enough to break the triangle inequality.

    Mollification and the robust Laplacian are only interesting on a mesh that needs them: the
    plain cotangent Laplacian returns NaN here, the robust one must not. Returned as both NumPy
    (``float64``, for the CPU references) and Warp (``float32``, where the inequality actually
    fails) so the two sides see the same mesh.
    """
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-9, 0.0], [0.5, 1.0, 0.0]], dtype=np.float64
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
