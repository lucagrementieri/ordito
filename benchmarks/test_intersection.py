"""
Benchmarks for ``ordito.intersection``.

Four functions on two cost shapes:

* ``mesh_with_plane`` / ``slice_mesh_with_plane`` / ``clip_mesh_with_field`` — one pass over the
  faces, a per-vertex scalar, then a compaction of the few faces the level set crosses.
  Memory-bound and dominated by the full-mesh sweep rather than the segment count: a plane meets
  ``O(sqrt(n_faces))`` triangles but every face is still classified. The last two run the *same*
  engine (a plane's signed distance is one such scalar), so their ordito rows track each other and
  a divergence means the shared path changed under one of them. ``clip_mesh_with_field``'s
  ``cap=True`` case is a different shape: the ``O(B ** 3)`` min-weight fill of the section loop
  dominates the clip by orders of magnitude, so read it as a ``holes.fill_min_weight`` measurement
  on a long rim.
* ``mesh_with_mesh`` — the only quadratic-ish one. A ``wp.Mesh`` BVH is built over the smaller mesh
  and every triangle of the other supplies an AABB query, so the cost tracks the number of
  *candidate* pairs (every box overlap; ``max_triangle_collisions`` is only the first stride)
  rather than the face count. This is the benchmark that moves when the separating-axis narrow
  phase changes.
* ``segments_with_plane`` — a pure ``wp.map`` over independent segments; the array-primitive
  baseline.

References
----------
**trimesh** covers three of the four (``mesh_plane``, ``slice_faces_plane``, ``plane_lines``). Its
``Trimesh`` is rebuilt inside the timed callable for ``mesh_plane`` because ``triangles`` /
``face_normals`` are cached properties that would make rounds 2..n measure only the plane
arithmetic; the other two take raw arrays.

**``mesh_with_mesh`` has no trimesh or open3d reference.** trimesh's mesh-mesh intersection routes
through the optional ``python-fcl`` collision backend, which reports *whether* pairs collide rather
than returning the curve, and open3d's booleans need the optional tensor backend with a coupled
remesh. The two that do answer it: **meshlib**'s ``findIntersectionContours``, which links the
crossing into ordered contours, and **pyvista**'s ``intersection``
(``vtkIntersectionPolyDataFilter``), which returns the same unordered segment soup ordito does and
therefore pins the value as well as the cost.

**pymeshlab** has the right filter and cannot run it here.
``generate_polyline_from_planar_section`` does what ``mesh_with_plane`` does and more (it *orders*
the segments), and works on the synthetic meshes — but raises ``Failed to apply filter`` on **every
scan mesh**, at any ``planeaxis``, ``planeoffset`` or ``relativeto``. That is the same non-manifold
boundary the libigl and potpourri3d references hit elsewhere, and this module's groups are all on
the scan sweep, so there is nowhere for the row to move. Recorded rather than skipped so it is not
re-derived.

**libigl** has no plane-section or mesh-mesh binding, so it is absent from those groups. It *does*
have the isocontour operation: ``igl.isolines`` is precisely ``marching_triangles``, and it is a row
in both of that function's groups.

**pyvista** answers ``clip_mesh_with_field``'s uncapped case through ``PolyData.clip_scalar`` over
the identical per-vertex field. ``invert=False`` is passed explicitly: its default keeps the side
*below* the value. Its capped counterpart ``clip_closed_surface`` cannot be timed here at all — it
validates the mesh first and raises on any open edge, which every scan mesh has — so the capped
comparison lives in ``tests/test_intersection.py`` on a closed synthetic mesh and the capped
benchmark case is ordito-only.

Caps
----
``mesh_with_mesh`` stops at ``bunny``: the broad phase allocates at least
``max_triangle_collisions`` candidate slots per query triangle, so the pair buffer alone is
``16 * n_faces`` ints (more where a face overflows) before the narrow phase filters it.
``clip_mesh_with_field``'s pyvista row stops there too, VTK's clip being a
single-threaded per-cell sweep.

Geometry
--------
Every plane cuts through the middle of the mesh — origin at the vertex-bounding-box centre, a fixed
off-axis normal — so the section is a full cross-section rather than a near-miss that would exit
early. ``mesh_with_mesh`` intersects the mesh with a copy translated by a fraction of its own
extent, which guarantees a large, genuinely overlapping curve on every mesh.
"""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import pyvista as pv
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import ordito as od
from conftest import BenchCase, mesh_ml_from_numpy, skip_larger_than

# Off-axis so no cut is degenerate w.r.t. the (axis-aligned) scan-mesh geometry.
_PLANE_DIRECTION = np.array([0.3, 0.8, 0.5])
_PLANE_NORMAL = _PLANE_DIRECTION / np.linalg.norm(_PLANE_DIRECTION)

# Self-intersection offsets, as fractions of the mesh bounding-box diagonal. ``mesh_with_mesh``
# costs the number of *actually overlapping* triangle pairs, not the face count, so the offset is
# the axis: a deep overlap intersects a broad band, a grazing one barely touches.
_SELF_OFFSET_FRACTIONS = [0.05, 0.60]

_shifted_cache: dict[tuple[str, str, float], wp.array[wp.vec3]] = {}


def _plane_origin(bench_case: BenchCase) -> np.ndarray:
    """Centre of the vertex bounding box — guarantees the plane cuts the mesh."""
    vertices = bench_case.vertices_np
    return 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))


def _shifted_vertices_wp(bench_case: BenchCase, offset_fraction: float) -> wp.array[wp.vec3]:
    """Translate the mesh's own vertices along the plane normal, to act as the second mesh."""
    key = (bench_case.mesh_name, str(bench_case.device), offset_fraction)
    if key not in _shifted_cache:
        vertices = bench_case.vertices_np
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        shifted = vertices + offset_fraction * diagonal * _PLANE_NORMAL
        _shifted_cache[key] = wp.array(
            np.ascontiguousarray(shifted, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
        )
    return _shifted_cache[key]


_field_ml_cache: dict[tuple[str, str], mm.VertScalars] = {}
_shifted_ml_cache: dict[tuple[str, float], mm.Mesh] = {}


def _field_ml(bench_case: BenchCase, field: str) -> mm.VertScalars:
    """
    Build the same scalar field as a ``VertScalars``, cached.

    There is no array constructor, so filling it is a per-vertex Python loop over ``VertId`` keys.
    Like every other library's copy of this field it is the benchmark's *input* and is built once.
    """
    key = (bench_case.mesh_name, field)
    if key not in _field_ml_cache:
        values_np = _field_np(bench_case, field)
        values_ml = mm.VertScalars()
        values_ml.resize(values_np.size, 0.0)
        for index, value in enumerate(values_np):
            values_ml[mm.VertId(index)] = float(value)
        _field_ml_cache[key] = values_ml
    return _field_ml_cache[key]


def _shifted_mesh_ml(bench_case: BenchCase, offset_fraction: float) -> mm.Mesh:
    """Build the translated self-copy [`_shifted_vertices_wp`] makes, as a cached MeshLib mesh."""
    key = (bench_case.mesh_name, offset_fraction)
    if key not in _shifted_ml_cache:
        vertices_np = bench_case.vertices_np
        diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
        _shifted_ml_cache[key] = mesh_ml_from_numpy(
            vertices_np + offset_fraction * diagonal * _PLANE_NORMAL, bench_case.faces_np
        )
    return _shifted_ml_cache[key]


_shifted_pv_cache: dict[tuple[str, float], pv.PolyData] = {}


def _shifted_mesh_pv(bench_case: BenchCase, offset_fraction: float) -> pv.PolyData:
    """Build the translated self-copy [`_shifted_vertices_wp`] makes, as a cached ``PolyData``."""
    key = (bench_case.mesh_name, offset_fraction)
    if key not in _shifted_pv_cache:
        vertices_np = bench_case.vertices_np
        diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
        _shifted_pv_cache[key] = pv.PolyData.from_regular_faces(
            np.ascontiguousarray(
                vertices_np + offset_fraction * diagonal * _PLANE_NORMAL, dtype=np.float64
            ),
            np.ascontiguousarray(bench_case.faces_np, dtype=np.int32),
        )
    return _shifted_pv_cache[key]


_mesh_ml_cache: dict[str, mm.Mesh] = {}


def _mesh_ml(bench_case: BenchCase) -> mm.Mesh:
    """
    Cache one ``meshlib.Mesh`` per mesh, for the rows whose call does **not** mutate it.

    ``extractPlaneSections``, ``findIntersectionContours`` and ``extractIsolines`` all read the mesh
    and return a new contour, so a shared mesh is safe and keeps the lazily built AABB tree warm.
    The two *trimming* rows build their own inside the timed callable, because they rewrite it.
    """
    if bench_case.mesh_name not in _mesh_ml_cache:
        _mesh_ml_cache[bench_case.mesh_name] = bench_case.new_mesh_ml()
    return _mesh_ml_cache[bench_case.mesh_name]


def _plane_ml(bench_case: BenchCase) -> mm.Plane3f:
    """Express the benchmark's plane in MeshLib's ``n . x == d`` form -- ``d`` is the transform."""
    origin_np = _plane_origin(bench_case)
    return mm.Plane3f(mm.Vector3f(*_PLANE_NORMAL.tolist()), float(np.dot(_PLANE_NORMAL, origin_np)))


@pytest.mark.benchmark(group="mesh_with_plane")
@pytest.mark.benchlibs("ordito", "trimesh", "meshlib")
def test_mesh_with_plane(bench_case: BenchCase) -> None:
    """
    Cross-section segments of a mid-mesh plane: a full face sweep plus a compaction.

    meshlib's ``extractPlaneSections`` does **more** than the other two rows and by a different
    route: it walks the section into ordered closed contours over an AABB tree
    (``UseAABBTree::Yes``, its default) rather than sweeping every face, so its cost tracks the
    section's length where ordito's and trimesh's track the face count. Read the pair across mesh
    sizes rather than at one point. Its output is ``EdgePoint`` contours, which
    ``tests/test_intersection.py`` decodes; the row asserts only that a section came back.
    """
    origin = _plane_origin(bench_case)
    if bench_case.kind == "meshlib":
        mesh_part_ml = mm.MeshPart(_mesh_ml(bench_case))
        plane_ml = _plane_ml(bench_case)
        mm.extractPlaneSections(mesh_part_ml, plane_ml)  # pre-warm the lazily built tree
        sections_ml = bench_case.run(lambda: mm.extractPlaneSections(mesh_part_ml, plane_ml))
        assert len(sections_ml) > 0
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        normal = wp.vec3(*_PLANE_NORMAL.tolist())
        plane_origin = wp.vec3(*origin.tolist())
        lines = bench_case.run(
            lambda: od.intersection.mesh_with_plane(vertices, faces, normal, plane_origin)
        )
        assert lines.shape[1] == 2
    else:  # rebuild inside: triangles / face_normals are cached Trimesh properties
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        lines_tm = bench_case.run(
            lambda: tm.intersections.mesh_plane(
                tm.Trimesh(vertices_np, faces_np, process=False), _PLANE_NORMAL, origin
            )
        )
        assert lines_tm.shape[1] == 2


@pytest.mark.benchmark(group="slice_mesh_with_plane")
@pytest.mark.benchlibs("ordito", "trimesh", "meshlib")
def test_slice_mesh_with_plane(bench_case: BenchCase) -> None:
    """
    Keep the positive-normal half of the mesh: classify every face, then re-triangulate cuts.

    meshlib's ``trimWithPlane`` keeps the same side and reaches the same answer -- 670 faces and an
    identical area on the test fixture (``tests/test_intersection.py``) -- by editing its half-edge
    topology in place, so its mesh is rebuilt inside the timed callable and the row carries that
    build. That is the honest cost for a caller holding NumPy buffers, which is what the trimesh row
    prices too.
    """
    origin = _plane_origin(bench_case)
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        params_ml = mm.TrimWithPlaneParams()
        params_ml.plane = _plane_ml(bench_case)

        def trim_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            mm.trimWithPlane(mesh_ml, params_ml)
            return mesh_ml.topology.numValidFaces()

        assert 0 < bench_case.run(trim_ml) < bench_case.n_faces
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        normal = wp.vec3(*_PLANE_NORMAL.tolist())
        plane_origin = wp.vec3(*origin.tolist())
        new_vertices, new_faces = bench_case.run(
            lambda: od.intersection.slice_mesh_with_plane(vertices, faces, normal, plane_origin)
        )
        assert int(new_faces.shape[0]) % 3 == 0
        assert new_vertices.shape[0] >= 0
    else:  # takes raw arrays, no Trimesh cache to defeat
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        sliced = bench_case.run(
            lambda: tm.intersections.slice_faces_plane(vertices_np, faces_np, _PLANE_NORMAL, origin)
        )
        assert sliced[1].shape[1] == 3


@pytest.mark.benchmark(group="split_mesh_with_plane")
@pytest.mark.benchlibs("ordito", "pyvista", "meshlib")
@pytest.mark.parity("split_mesh_with_plane", "pyvista")
def test_split_mesh_with_plane(bench_case: BenchCase) -> None:
    """
    Keep **both** halves with the section inserted as shared edges, plus a per-face side label.

    A different cost shape from ``slice_mesh_with_plane`` despite the same input, and the comparison
    between the two rows is the point: the slice classifies faces and compacts three kept classes,
    while this builds the *unique edge* table (a radix sort over ``3 * n_faces`` keys) so an edge's
    crossing can be inserted once for both its faces. That sort is the extra cost of being
    crack-free and is what this row measures — expect it above the slice's, growing with the face
    count rather than with the number of crossed triangles.

    pyvista's counterpart is ``clip(return_clipped=True)``, VTK's both-sides plane clip. Note its
    ``kept`` output is the *low* side, i.e. ordito's ``~above``; the values are compared in
    ``tests/test_intersection.py``, this row only times them.
    """
    origin = _plane_origin(bench_case)
    if bench_case.kind == "meshlib":
        # ``subdivideWithPlane`` is the closest counterpart in the suite: it inserts the section as
        # real edges and returns the positive side as a FaceBitSet, which is ordito's
        # ``(vertices, faces, side_mask)`` triple. It mutates, so the mesh is rebuilt per round.
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        plane_ml = _plane_ml(bench_case)

        def subdivide_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            return mm.subdivideWithPlane(mesh_ml, plane_ml).count()

        assert 0 < bench_case.run(subdivide_ml) <= bench_case.n_faces
        return
    if bench_case.kind == "pyvista":
        # Same cap and reason as the ``clip_mesh_with_field`` row: VTK's clip is single-threaded.
        skip_larger_than(bench_case, "bunny", "VTK's clip is a single-threaded per-cell sweep")
        mesh_pv = bench_case.mesh_pv
        kept_pv, clipped_pv = bench_case.run(
            lambda: mesh_pv.clip(
                normal=tuple(_PLANE_NORMAL.tolist()),
                origin=tuple(origin.tolist()),
                return_clipped=True,
            )
        )
        assert kept_pv.n_cells + clipped_pv.n_cells > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    normal = wp.vec3(*_PLANE_NORMAL.tolist())
    plane_origin = wp.vec3(*origin.tolist())
    new_vertices, new_faces, above = bench_case.run(
        lambda: od.intersection.split_mesh_with_plane(vertices, faces, normal, plane_origin)
    )
    assert int(new_faces.shape[0]) % 3 == 0
    assert int(above.shape[0]) == int(new_faces.shape[0]) // 3
    assert new_vertices.shape[0] >= vertices.size


def _plane_field(bench_case: BenchCase) -> tuple[wp.array[wp.float32], np.ndarray]:
    """Build the cutting plane's signed distance as a per-vertex field, for both sides of a row."""
    vertices_np = bench_case.vertices_np
    field_np = (vertices_np - _plane_origin(bench_case)) @ _PLANE_NORMAL
    return (
        wp.array(
            np.ascontiguousarray(field_np, dtype=np.float32),
            dtype=wp.float32,
            device=bench_case.device,
        ),
        field_np,
    )


@pytest.mark.benchmark(group="split_faces_along_field")
@pytest.mark.benchlibs("ordito", "pyvista")
def test_split_faces_along_field(bench_case: BenchCase) -> None:
    """
    The both-sides cut: same classify-and-compact sweep, but every face is kept and relabelled.

    Read against ``clip_mesh_with_field`` below, on the identical field, because the extra work is
    exactly what this group exists to price: four face classes instead of three, an ``edges_unique``
    pass to give each crossed *edge* one shared crossing vertex (which is what makes the result
    watertight where the clip's per-face crossings are not), and two emit kernels instead of two
    that between them write three times as many triangles.

    pyvista's counterpart is the same filter with ``both=True``, which returns a ``(below, above)``
    tuple; it is the same VTK sweep as the clip's row plus the second half's output, and it is
    capped at ``bunny`` for the same reason -- a single-threaded per-cell sweep.

    Costing a few times the clip is understood and is the price of keeping both sides: the clip's
    work scales with the *cut* (a plane meets ``O(sqrt(n_faces))`` triangles) while the split adds
    an ``edges_unique`` pass over the whole mesh to give each crossed edge one shared crossing
    vertex. That pass alone is about half the split at every size.

    **``lucy``'s much larger ratio is not explained by that** and is left as a finding rather than
    a guess: a synthetic sweep four times past ``lucy``'s scale stays near the general ratio, so
    size is not the answer and the cause is something about that mesh. Restricting the edge pass to
    the crossed faces is the obvious fix for the general ratio either way, since the pairing only
    needs the halfedges the level set actually meets.
    """
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny", "VTK's clip is a single-threaded per-cell sweep")
        mesh_pv = bench_case.mesh_pv
        mesh_pv.point_data["field"] = _plane_field(bench_case)[1]
        below_pv, above_pv = bench_case.run(
            lambda: mesh_pv.clip_scalar(scalars="field", value=0.0, both=True)
        )
        assert below_pv.n_faces > 0
        assert above_pv.n_faces > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    field_wp = _plane_field(bench_case)[0]
    _new_vertices, new_faces, positive = bench_case.run(
        lambda: od.intersection.split_faces_along_field(vertices, faces, field_wp)
    )
    assert int(new_faces.shape[0]) % 3 == 0
    assert 0 < int(positive.numpy().sum()) < int(new_faces.shape[0]) // 3


@pytest.mark.benchmark(group="clip_mesh_with_field")
@pytest.mark.benchlibs("ordito", "pyvista")
@pytest.mark.parametrize("cap", [False, True])
def test_clip_mesh_with_field(bench_case: BenchCase, cap: bool) -> None:
    """
    The same classify-and-compact sweep as ``slice_mesh_with_plane``, driven by a per-vertex field.

    Timed against the plane's own signed distance so the work is identical to that group's and the
    two are directly comparable — the field costs one extra buffer read per vertex and saves the
    per-edge dot products. ``cap=True`` adds the crossing merge (an exact-position hash over the
    crossings) plus the ``O(B^3)`` min-weight fill of the section, which is why it is a separate
    case rather than a flag folded into one row: on a scan mesh the section loop is long and the
    fill, not the clip, is what is being measured.

    pyvista's counterpart is ``clip_scalar`` (``invert=False`` — its default keeps the *low* side).
    **The capped case has no reference row**: ``clip_closed_surface`` validates its input first and
    raises ``ValueError: This surface appears to be non-manifold`` on every scan mesh, all of which
    carry open edges, so there is nowhere for the row to move. It *is* compared, on a closed
    synthetic mesh, in ``tests/test_intersection.py``.
    """
    if bench_case.kind == "pyvista":
        if cap:
            pytest.skip("clip_closed_surface rejects a non-manifold input; every scan mesh is one")
        # VTK's clip is a single-threaded per-cell sweep, capped like the suite's other host-bound
        # references.
        skip_larger_than(bench_case, "bunny", "VTK's clip is a single-threaded per-cell sweep")
        mesh_pv = bench_case.mesh_pv
        mesh_pv.point_data["field"] = _plane_field(bench_case)[1]
        clipped_pv = bench_case.run(
            lambda: mesh_pv.clip_scalar(scalars="field", value=0.0, invert=False)
        )
        assert clipped_pv.n_faces > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    field_wp = _plane_field(bench_case)[0]
    _new_vertices, new_faces = bench_case.run(
        lambda: od.intersection.clip_mesh_with_field(vertices, faces, field_wp, cap=cap)
    )
    assert int(new_faces.shape[0]) % 3 == 0


@pytest.mark.benchmark(group="mesh_with_mesh")
@pytest.mark.benchlibs("ordito", "meshlib", "pyvista")
@pytest.mark.parametrize("offset_fraction", _SELF_OFFSET_FRACTIONS, ids=["deep", "grazing"])
def test_mesh_with_mesh(bench_case: BenchCase, offset_fraction: float) -> None:
    """
    BVH broad phase plus the separating-axis narrow phase, against a translated self-copy.

    Cost is the number of overlapping triangle *pairs*, so the translation distance is the axis
    rather than the face count. The ``deep`` row shares most of its volume with the original and
    the ``grazing`` row barely touches it; the gap is the collision density. A face whose box
    overlaps more than ``max_triangle_collisions`` others makes the broad phase walk again at the
    wider stride, so a saturated row pays a second walk rather than losing pairs.

    pyvista's ``intersection`` returns the same *unordered* segment soup ordito does, which is what
    makes it the value reference for this group as well as a cost one. Note what its ``grazing`` row
    measures: at 0.60 of the diagonal the two copies do not touch at all, so VTK does its broad
    phase, logs ``No Intersection between objects`` and returns **0** line cells -- at well over
    half the ``deep`` case's cost, i.e. most of the cost is the traversal rather than the crossing.
    Only the ``deep`` case can assert a non-empty answer, and only it does.
    """
    skip_larger_than(bench_case, "bunny", "broad phase allocates 16 candidate slots per triangle")
    if bench_case.kind == "pyvista":
        mesh_pv, shifted_pv = bench_case.mesh_pv, _shifted_mesh_pv(bench_case, offset_fraction)
        intersection_pv, _first_pv, _second_pv = bench_case.run(
            lambda: mesh_pv.intersection(shifted_pv, split_first=False, split_second=False)
        )
        if offset_fraction == min(_SELF_OFFSET_FRACTIONS):
            assert intersection_pv.n_cells > 0  # the deep case really does cross
        return
    if bench_case.kind == "meshlib":
        # ``findIntersectionContours`` links the crossing into ordered contours where ordito emits
        # an unordered segment soup, so it does strictly more -- and it takes the second mesh's
        # placement as a rigid transform rather than as moved vertices, which is how the same
        # translation is expressed here. Neither mesh is modified, so both are cached.
        mesh_ml = _mesh_ml(bench_case)
        shifted_ml = _shifted_mesh_ml(bench_case, offset_fraction)
        contours_ml = bench_case.run(lambda: mm.findIntersectionContours(mesh_ml, shifted_ml))
        assert len(contours_ml) >= 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    shifted = _shifted_vertices_wp(bench_case, offset_fraction)
    lines = bench_case.run(lambda: od.intersection.mesh_with_mesh(vertices, faces, shifted, faces))
    assert lines.shape[1] == 2


@pytest.mark.noparity(
    "pyvista",
    oracle="meshlib",
    reason="D2 a different quantity with a measured disagreement: PolyData.collision counts "
    "contact pairs from VTK's OBB tree rather than the set of crossing triangles, and it "
    "reports 2 600 hits for a 320-cell mesh against its own copy where 0 triangles cross. "
    "Its row is a cost "
    "comparison; meshlib's findCollidingTriangleBitsets is the oracle, in "
    "tests/test_intersection.py::test_mesh_collision_pairs_matches_meshlib.",
)
@pytest.mark.benchmark(group="mesh_collision_pairs")
@pytest.mark.benchlibs("ordito", "meshlib", "pyvista")
@pytest.mark.parametrize("offset_fraction", _SELF_OFFSET_FRACTIONS, ids=["deep", "grazing"])
@pytest.mark.parametrize("setup", ["full", "amortized"])
def test_mesh_collision_pairs(bench_case: BenchCase, offset_fraction: float, setup: str) -> None:
    """
    The same broad and narrow phase as ``mesh_with_mesh``, stopping before the segments.

    ``setup`` says where the acceleration structure is built. ``full`` builds it inside the timed
    callable: ordito's ``wp.Mesh`` BVH from the raw arrays, and VTK's OBB trees, which
    ``PolyData.collision`` rebuilds on every call. ``amortized`` hoists it out: ordito queries the
    BVH an [`ordito.Trimesh`][ordito.mesh.Trimesh] caches (``mesh_a=`` / ``mesh_b=``), and MeshLib
    its AABB trees, which it builds lazily on first use and caches on each ``Mesh``. Each reference
    has a row only at the setup it actually runs: MeshLib never rebuilds a cached tree and pyvista
    cannot keep one, so a ``full`` MeshLib row would time the mesh converter and an ``amortized``
    pyvista row would not be amortized.

    Read the two groups against each other: they share the broad phase (``_candidate_face_pairs``)
    verbatim and run the same narrow phase over its candidates, each compacting once, so the gap is
    what computing an intersection *segment* per crossing pair costs, plus the degenerate-segment
    filter. That is the reason this exists as its own group rather than being
    assumed cheaper.

    meshlib's ``findCollidingTriangleBitsets`` answers the same question and returns the two masks;
    it is the oracle in ``tests/test_intersection.py``, where the two agree face for face on a
    sphere-versus-box pair. pyvista's ``collision`` is a *contact* count rather than a crossing set
    -- CLAUDE.md section 7.6 records it reporting 2 600 hits for a 320-cell mesh against its own
    copy -- so its row is a cost comparison only, and the noparity entry says so.

    The ``deep`` / ``grazing`` pair is the fixed-width broad phase showing through. ordito
    allocates a fixed number of candidate slots per query triangle whatever the geometry, so its
    cost barely moves between the two offsets; MeshLib descends two trees and exits almost
    immediately when there is nothing to find, then pays for the pairs when there is. So the ratio
    here is a question about *early exit*, not about the narrow phase -- and the ``grazing`` row is
    the one to watch if that ever changes.
    """
    skip_larger_than(bench_case, "bunny", "broad phase allocates 16 candidate slots per triangle")
    if bench_case.kind == "pyvista" and setup == "amortized":
        pytest.skip("PolyData.collision rebuilds its OBB trees on every call: no amortized form")
    if bench_case.kind == "meshlib" and setup == "full":
        pytest.skip("MeshLib caches its AABB trees on the Mesh: timed at setup=amortized")
    if bench_case.kind == "pyvista":
        # VTK's OBB collision does not survive ``bunny`` here: it answers at a few thousand faces
        # and does not return at all on two 70k-face copies overlapping deeply, which is the
        # quadratic blow-up an OBB tree hits when most boxes overlap. Capped where it still answers.
        skip_larger_than(
            bench_case, "bunny_decimated", "VTK's OBB collision does not return on 70k x 70k"
        )
        mesh_pv, shifted_pv = bench_case.mesh_pv, _shifted_mesh_pv(bench_case, offset_fraction)
        collision_pv, n_contacts = bench_case.run(lambda: mesh_pv.collision(shifted_pv))
        assert collision_pv.n_cells >= 0
        assert n_contacts >= 0
        return
    if bench_case.kind == "meshlib":
        mesh_ml = _mesh_ml(bench_case)
        shifted_ml = _shifted_mesh_ml(bench_case, offset_fraction)
        # Pre-warm both lazily built trees, so every timed round prices the query alone.
        mm.findCollidingTriangleBitsets(mm.MeshPart(mesh_ml), mm.MeshPart(shifted_ml))
        masks_ml = bench_case.run(
            lambda: mm.findCollidingTriangleBitsets(mm.MeshPart(mesh_ml), mm.MeshPart(shifted_ml))
        )
        assert len(masks_ml) == 2
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    shifted = _shifted_vertices_wp(bench_case, offset_fraction)
    if setup == "amortized":
        mesh_a = od.Trimesh(vertices, faces).warp_mesh
        mesh_b = od.Trimesh(shifted, faces).warp_mesh
        pairs = bench_case.run(
            lambda: od.intersection.mesh_collision_pairs(
                vertices, faces, shifted, faces, mesh_a=mesh_a, mesh_b=mesh_b
            )
        )
    else:
        pairs = bench_case.run(
            lambda: od.intersection.mesh_collision_pairs(vertices, faces, shifted, faces)
        )
    assert pairs.shape[1] == 2


@pytest.mark.benchmark(group="segments_with_plane")
@pytest.mark.benchlibs("ordito", "trimesh")
def test_segments_with_plane(bench_case: BenchCase) -> None:
    """Batched segment-plane hits over the mesh's directed edges: one ``wp.map``, no adjacency."""
    origin = _plane_origin(bench_case)
    faces_np = bench_case.faces_np
    # One segment per face corner: (v0,v1) of every triangle, so the count scales with the mesh.
    start_np = bench_case.vertices_np[faces_np[:, 0]]
    end_np = bench_case.vertices_np[faces_np[:, 1]]
    if bench_case.kind == "ordito":
        device = bench_case.device
        start = wp.array(
            np.ascontiguousarray(start_np, dtype=np.float32), dtype=wp.vec3, device=device
        )
        end = wp.array(np.ascontiguousarray(end_np, dtype=np.float32), dtype=wp.vec3, device=device)
        normal = wp.vec3(*_PLANE_NORMAL.tolist())
        plane_origin = wp.vec3(*origin.tolist())
        _points, valid = bench_case.run(
            lambda: od.intersection.segments_with_plane(start, end, normal, plane_origin)
        )
        assert valid.shape[0] == start_np.shape[0]
    else:
        endpoints = np.stack((start_np, end_np))
        points_tm, valid_tm = bench_case.run(
            lambda: tm.intersections.plane_lines(origin, _PLANE_NORMAL, endpoints)
        )
        assert points_tm.shape[0] == valid_tm.sum()


# --- marching_triangles ---------------------------------------------------------------
_ISOVALUE = 0.1137
# potpourri3d runs into hundreds of milliseconds on the many-curve fields.
_ROUNDS = 3

# Field frequency -> (name, wavenumber). ``plane`` is the single-contour control.
_FIELDS = {"plane": 0, "wave12": 12, "wave40": 40}

_field_cache: dict[tuple[str, str], np.ndarray] = {}
_field_wp_cache: dict[tuple[str, str, str], wp.array[wp.float64]] = {}


def _field_np(bench_case: BenchCase, field: str) -> np.ndarray:
    """Scalar field on this mesh, cached: it is an input, not part of the measurement."""
    key = (bench_case.mesh_name, field)
    if key not in _field_cache:
        vertices = bench_case.vertices_np
        wavenumber = _FIELDS[field]
        if wavenumber == 0:
            values = vertices[:, 2]
        else:
            values = (
                np.sin(wavenumber * vertices[:, 0])
                * np.cos(wavenumber * vertices[:, 1])
                * np.sin(wavenumber * vertices[:, 2])
            )
        _field_cache[key] = np.ascontiguousarray(values, dtype=np.float64)
    return _field_cache[key]


def _field_wp(bench_case: BenchCase, field: str) -> wp.array[wp.float64]:
    """Return the same field as a device buffer."""
    key = (bench_case.mesh_name, field, str(bench_case.device))
    if key not in _field_wp_cache:
        _field_wp_cache[key] = wp.array(
            _field_np(bench_case, field), dtype=wp.float64, device=bench_case.device
        )
    return _field_wp_cache[key]


def _run_case(bench_case: BenchCase, field: str) -> None:
    """Extract one level set, in ordito, potpourri3d, libigl or meshlib."""
    if bench_case.kind == "meshlib":
        # ``extractIsolines`` returns linked contours like ordito and potpourri3d, not igl's
        # segment soup. Its ``VertScalars`` field has no array constructor -- the fill is a
        # per-vertex Python loop -- so it is built once outside the timed callable, as every other
        # row's field is.
        mesh_ml = _mesh_ml(bench_case)
        values_ml = _field_ml(bench_case, field)
        isolines_ml = bench_case.run(
            lambda: mm.extractIsolines(mesh_ml.topology, values_ml, _ISOVALUE), rounds=_ROUNDS
        )
        assert len(isolines_ml) > 0
        return
    if bench_case.kind == "igl":
        # igl returns a segment *soup* -- (points, segments, segment_values), no curve linkage --
        # so it does strictly less than ordito and potpourri3d, both of which return linked
        # polylines. Read the row as the floor for the extraction without the linking.
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        values_np = _field_np(bench_case, field)
        points_igl, segments_igl, _values_igl = bench_case.run(
            lambda: igl.isolines(vertices_np, faces_np, values_np, np.array([_ISOVALUE])),
            rounds=_ROUNDS,
        )
        assert points_igl.shape[1] == 3
        assert segments_igl.shape[0] > 0
        return
    if bench_case.kind == "ordito":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        values, n_vertices = _field_wp(bench_case, field), bench_case.n_vertices
        curves, _ = bench_case.run(
            lambda: od.intersection.marching_triangles(
                vertices, faces, values, _ISOVALUE, n_vertices=n_vertices
            ),
            rounds=_ROUNDS,
        )
        assert len(curves) > 0
    else:
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)
        values_np = _field_np(bench_case, field)
        curves_pp = bench_case.run(
            lambda: pp3d.marching_triangles(vertices_np, faces_np, values_np, _ISOVALUE),
            rounds=_ROUNDS,
        )
        assert len(curves_pp) > 0


@pytest.mark.benchmark(group="marching_triangles")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("ordito", "potpourri3d", "igl", "meshlib")
def test_marching_triangles(bench_case: BenchCase) -> None:
    """One long closed contour of a coordinate function, over the clean size sweep."""
    _run_case(bench_case, "plane")


@pytest.mark.benchmark(group="marching_triangles_curves")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito", "potpourri3d", "igl", "meshlib")
@pytest.mark.parametrize("field", list(_FIELDS))
def test_marching_triangles_curves(bench_case: BenchCase, field: str) -> None:
    """One mesh, level sets from 1 to ~1 000 curves, to see whether linking cost shows up."""
    _run_case(bench_case, field)


@pytest.mark.benchmark(group="marching_triangles_with_offsets")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("ordito")
@pytest.mark.parametrize("field", list(_FIELDS))
def test_marching_triangles_with_offsets(bench_case: BenchCase, field: str) -> None:
    """
    The packed form of ``marching_triangles_curves``' level sets: no per-curve object.

    Read against that group's ordito row: the difference is the list of per-curve views, which is
    the whole of what the list form adds.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    values, n_vertices = _field_wp(bench_case, field), bench_case.n_vertices
    points, offsets, closed = bench_case.run(
        lambda: od.intersection.marching_triangles_with_offsets(
            vertices, faces, values, _ISOVALUE, n_vertices=n_vertices
        ),
        rounds=_ROUNDS,
    )
    assert offsets.size == closed.size + 1
    assert points.size > 0
