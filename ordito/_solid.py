"""
Closed solids in halfedge form, shared by the Booleans, simplification, subdivision and level sets.

A solid is a closed, edge-manifold mesh with ``float64`` positions in Morton order, each face's
normal and coplanar group, each vertex's pseudo-normal and the tolerances. This module builds
solids from meshes (``solid_from_mesh``, ``solid_from_positions``), lays them out again as
``float32`` meshes (``mesh_from_solid``), and runs the topology passes on a halfedge result
(``simplify_topology``, ``finish_solid``). It ports the Manifold library's ``Impl``, so every
answer built on it is Manifold's bit for bit. Its kernels are ``kernels/algorithms/solid`` and
``kernels/algorithms/solid_topology``; ``ray.segment_hits`` builds its solids here too.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Protocol, cast

import warp as wp

import ordito.typing as odt
from ordito import _launch
from ordito._common import offsets_buffer, read_array, read_scalar, read_values, scan_count
from ordito.array import gather
from ordito.halfedge import halfedge_twins
from ordito.kernels import array as kernel_array
from ordito.kernels import scatter as kernel_scatter
from ordito.kernels.algorithms import solid as kernel_solid
from ordito.kernels.algorithms import solid_topology as kernel_solid_topology

FLOAT32_EPSILON = 1.1920929e-07


# A collapse, dedupe or swap pass on CUDA of at most this many flagged edges, plus one for every
# ``_SERIAL_DEVICE_HALFEDGES`` halfedges of the mesh, runs in one GPU thread; a longer one goes to
# the CPU (``_launch_serial``). One GPU thread takes ~1.25 us a flagged edge, the CPU's round trip
# of the mesh's used prefixes ~0.3 ms plus ~12 ns a halfedge (2026-11-21: passes of 2, 6 and 19
# edges at 4 M halfedges paid 49-57 ms each for the trip, their host kernel 0.1 ms; 55 824
# collapses took 73 ms in a GPU thread against 3.1 ms on the CPU).
_SERIAL_DEVICE_BASE = 256
_SERIAL_DEVICE_HALFEDGES = 128


# Up to this many claiming seeds the coplanar flood runs in one thread, as Manifold's loop does;
# above it in parallel rounds on CUDA (2026-11-21, quad-strip revolves: 64 seeds at par, 256
# 0.39x, 2 048 0.07x, 16 384 0.008x). The CPU always takes the one thread: it runs a launch's
# threads in turn, so the rounds only add work.
_SERIAL_FLOOD_SEEDS = 64


class Solid(NamedTuple):
    """
    A closed mesh as Manifold's ``Impl`` carries it between operations.

    ``float64`` positions and Morton-ordered vertices and faces (``SortGeometry``); each face's
    normal (inherited from the face it was cut from, never recomputed), its mesh's id and its
    coplanar group (``triRef``); each vertex's pseudo-normal; and the tolerances.
    """

    positions: wp.array[wp.vec3d]
    faces: wp.array[wp.int32]
    face_normals: wp.array[wp.vec3d]
    vertex_normals: wp.array[wp.vec3d]
    face_mesh: wp.array[wp.int32]
    face_coplanar: wp.array[wp.int32]
    epsilon: float
    tolerance: float
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    # For a solid built from a caller's mesh: each face's index in that mesh.
    face_source: wp.array[wp.int32] | None = None

    @property
    def n_vertices(self) -> int:
        """The vertex count."""
        return int(self.positions.shape[0])

    @property
    def n_faces(self) -> int:
        """The face count."""
        return self.faces.size // 3

    @property
    def is_empty(self) -> bool:
        """Whether the solid has no faces."""
        return self.faces.size == 0

    @property
    def scale(self) -> float:
        """``Box::Scale``: the largest absolute coordinate of the box."""
        if self.is_empty:
            return 0.0
        return max(abs(x) for x in (*self.lower, *self.upper))


class _HalfedgeMesh(Protocol):
    """
    The fields of a ``kernel_solid_topology.HalfedgeMesh`` struct, for the type checker.

    The result's halfedges, triangles and vertices as the topology passes edit them in place, and
    their used ``sizes``.
    """

    start: wp.array[wp.int32]
    pair: wp.array[wp.int32]
    tri_mesh: wp.array[wp.int32]
    tri_coplanar: wp.array[wp.int32]
    tri_normal: wp.array[wp.vec3d]
    tri_source: wp.array[wp.int32]
    pos: wp.array[wp.vec3d]
    sizes: wp.array[wp.int32]


def halfedge_mesh_from_solid(solid: Solid) -> _HalfedgeMesh:
    """Return a solid's halfedges as the topology passes edit them."""
    device = solid.positions.device
    n_faces = solid.n_faces
    mesh = halfedge_mesh(n_faces, solid.n_vertices, solid.positions, device)
    twins = halfedge_twins(solid.faces, solid.n_vertices, validate=False)
    _launch.copy(mesh.start, solid.faces, count=solid.faces.size)
    _launch.copy(mesh.pair, twins, count=solid.faces.size)
    _launch.copy(mesh.tri_normal, solid.face_normals, count=n_faces)
    _launch.copy(mesh.tri_mesh, solid.face_mesh, count=n_faces)
    _launch.copy(mesh.tri_coplanar, solid.face_coplanar, count=n_faces)
    return mesh


def solid_from_mesh(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], mesh_id: int, name: str
) -> Solid:
    """
    Build a solid from a ``float32`` mesh, as Manifold's ``MeshGL`` constructor does.

    The positions widened to ``float64`` and the tolerances set from their box (``SetEpsilon``);
    a triangle naming one vertex twice dropped; then ``CleanupTopology`` (pinched vertices split),
    ``SetNormalsAndCoplanar``, ``RemoveDegenerates`` (edges shorter than the precision collapsed,
    degenerate triangles' long edges swapped), ``RemoveUnreferencedVerts`` and ``SortGeometry`` in
    the input's box. ``face_source`` names each face's triangle of the caller's mesh: the one it
    was cut from, or whose reference it took in a swap. A mesh with none of those defects, which
    one pass over its halfedges finds, is the one ``solid_from_positions`` builds, and the cleanup
    does not run.

    Raises
    ------
    ValueError
        If the mesh is not closed, edge-manifold and consistently wound.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if faces.size == 0:
        return empty_solid(device)
    widened = _launch.empty(n_vertices, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_solid.widen_positions,
        dim=n_vertices,
        inputs=[vertices],
        outputs=[widened],
        device=device,
    )
    lower, upper = positions_box(widened)
    scale = max(abs(x) for x in (*lower, *upper))
    epsilon = kernel_solid.PRECISION * scale
    # ``SetEpsilon(-1, useSingle=true)``: a single-precision mesh's tolerance.
    tolerance = max(epsilon, FLOAT32_EPSILON * scale)
    twins, first_halfedge, defects = _paired_halfedges(widened, faces)
    source: wp.array[wp.int32] | None = None
    if any(defects):
        # A triangle naming a vertex twice pairs none of its halfedges; Manifold drops it before
        # pairing. Looked for only once the pairing fails, so a clean mesh pays nothing for it.
        distinct = _faces_with_distinct_corners(faces)
        if distinct is not None:
            faces, source = distinct
            if faces.size == 0:
                return empty_solid(device)
            twins, first_halfedge, defects = _paired_halfedges(widened, faces)
        _require_closed(defects, name)
    face_normals, coplanar = _normals_and_coplanar(widened, faces, twins, tolerance)
    paired = (widened, faces, twins, first_halfedge, face_normals, coplanar)
    if source is None and not _needs_cleanup(paired, epsilon, tolerance):
        return _paired_solid(paired, mesh_id, epsilon, tolerance, lower, upper)
    return _cleaned_solid(
        widened, faces, twins, source, mesh_id, epsilon, tolerance, (lower, upper)
    )


def _needs_cleanup(
    paired: tuple[
        wp.array[wp.vec3d],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.vec3d],
        wp.array[wp.int32],
    ],
    epsilon: float,
    tolerance: float,
) -> bool:
    """
    Whether the ``MeshGL`` constructor's cleanup would change the paired mesh at all.

    A pinched vertex, an edge shorter than ``epsilon``, a degenerate triangle's long edge to swap
    (projected on its coplanar group's normal) or a vertex no face uses; none of them, and the
    cleanup leaves the mesh as ``solid_from_positions`` builds it.
    """
    positions, faces, twins, first_halfedge, face_normals, coplanar = paired
    device = positions.device
    nothing = _launch.empty(0, dtype=wp.int32, device=device)
    view = cast(_HalfedgeMesh, kernel_solid_topology.HalfedgeMesh())
    view.start = faces
    view.pair = twins
    view.tri_mesh = nothing
    view.tri_coplanar = coplanar
    view.tri_normal = face_normals
    view.tri_source = nothing
    view.pos = positions
    view.sizes = nothing
    outgoing = _launch.zeros(first_halfedge.size, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_solid_topology.count_outgoing_halfedges,
        dim=faces.size,
        inputs=[view],
        outputs=[outgoing],
        device=device,
    )
    found = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_solid_topology.flag_cleanup_candidates,
        dim=max(faces.size // 3, first_halfedge.size),
        inputs=[view, first_halfedge, outgoing, tolerance, epsilon],
        outputs=[found],
        device=device,
    )
    # Host readback: whether the cleanup's passes run at all.
    return int(read_scalar(found, 0)) != 0


def _cleaned_solid(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    source: wp.array[wp.int32] | None,
    mesh_id: int,
    epsilon: float,
    tolerance: float,
    box: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> Solid:
    """
    ``CleanupTopology`` through ``SortGeometry``, as the ``MeshGL`` constructor runs them.

    ``source`` names each face's triangle of the caller's mesh where faces were dropped first
    (``None``: the identity).
    """
    device = positions.device
    n_faces = faces.size // 3
    n_vertices = int(positions.shape[0])
    mesh = halfedge_mesh(n_faces, n_vertices, positions, device, with_source=True)
    _launch.copy(mesh.start, faces, count=3 * n_faces)
    _launch.copy(mesh.pair, twins, count=3 * n_faces)
    _launch.view(mesh.tri_mesh, 0, n_faces).fill_(mesh_id)
    # ``coplanarID`` starts as the triangle's index; ``SetNormalsAndCoplanar`` replaces it.
    _launch.launch(
        kernel_array.ARANGE[wp.int32],
        dim=n_faces,
        outputs=[_launch.view(mesh.tri_coplanar, 0, n_faces)],
        device=device,
    )
    if source is None:
        _launch.launch(
            kernel_array.ARANGE[wp.int32],
            dim=n_faces,
            outputs=[_launch.view(mesh.tri_source, 0, n_faces)],
            device=device,
        )
    else:
        _launch.copy(mesh.tri_source, source, count=n_faces)
    n_h, n_v = simplify_topology(
        mesh, (3 * n_faces, n_vertices), 0, epsilon, tolerance, device, cleanup_only=True
    )
    # ``SetNormalsAndCoplanar`` on the cleaned triangles, written into the mesh's own rows.
    n_t = n_h // 3
    cleaned_faces = _launch.view(mesh.start, 0, n_h)
    cleaned_positions = _launch.view(mesh.pos, 0, n_v)
    face_normals = _launch.view(mesh.tri_normal, 0, n_t)
    _launch.launch(
        kernel_solid.mesh_face_normals,
        dim=n_t,
        inputs=[cleaned_positions, cleaned_faces],
        outputs=[face_normals],
        device=device,
    )
    coplanar = coplanar_groups(
        cleaned_positions, cleaned_faces, _launch.view(mesh.pair, 0, n_h), face_normals, tolerance
    )
    _launch.copy(mesh.tri_coplanar, coplanar, count=n_t)
    # ``RemoveDegenerates``, whose own ``CleanupTopology`` the call above has just done.
    sizes = simplify_topology(
        mesh, (n_h, n_v), 0, epsilon, tolerance, device, colinear=False, cleanup=False
    )
    return finish_solid(mesh, sizes, epsilon, tolerance, device, box)


def _faces_with_distinct_corners(
    faces: wp.array[wp.int32],
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]] | None:
    """Return the faces whose corners differ, with each one's index; ``None`` if all do."""
    device = faces.device
    n_faces = faces.size // 3
    offsets, flags = offsets_buffer(n_faces, device)
    _launch.launch(
        kernel_solid.distinct_corner_flags,
        dim=n_faces,
        inputs=[faces],
        outputs=[flags],
        device=device,
    )
    # Host readback: the kept count sizes the compacted faces.
    n_kept = scan_count(flags)
    if n_kept == n_faces:
        return None
    kept = _launch.empty(3 * n_kept, dtype=wp.int32, device=device)
    source = _launch.empty(n_kept, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_solid.compact_distinct_faces,
        dim=n_faces,
        inputs=[faces, offsets],
        outputs=[kept, source],
        device=device,
    )
    return kept, source


def solid_from_positions(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    mesh_id: int,
    epsilon: float,
    tolerance: float,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    name: str,
    *,
    sort_first: bool = False,
) -> Solid:
    """
    Build a solid from ``float64`` positions: coplanar groups, normals and Morton order.

    ``SetNormalsAndCoplanar`` (the coplanar groups and their normals, then the vertices'
    pseudo-normals) in the given order, then ``SortGeometry``, as the ``MeshGL`` constructor runs
    them; with ``sort_first`` the other way round, as Manifold's primitive shapes do. The coplanar
    ids are triangle indices in the order the groups were formed in.

    Raises
    ------
    ValueError
        If the mesh is not closed, edge-manifold and consistently wound.
    """
    if sort_first:
        positions, faces, _, _, _, _, _ = _morton_sorted(positions, faces, lower, upper)
    twins, first_halfedge, defects = _paired_halfedges(positions, faces)
    _require_closed(defects, name)
    face_normals, coplanar = _normals_and_coplanar(positions, faces, twins, tolerance)
    paired = (positions, faces, twins, first_halfedge, face_normals, coplanar)
    return _paired_solid(paired, mesh_id, epsilon, tolerance, lower, upper, sorted_=not sort_first)


def _normals_and_coplanar(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    tolerance: float,
) -> tuple[wp.array[wp.vec3d], wp.array[wp.int32]]:
    """``SetNormalsAndCoplanar``'s face half: each face's normal and coplanar group."""
    n_faces = faces.size // 3
    face_normals = _launch.empty(n_faces, dtype=wp.vec3d, device=positions.device)
    _launch.launch(
        kernel_solid.mesh_face_normals,
        dim=n_faces,
        inputs=[positions, faces],
        outputs=[face_normals],
        device=positions.device,
    )
    coplanar = coplanar_groups(positions, faces, twins, face_normals, tolerance)
    return face_normals, coplanar


def _paired_solid(
    paired: tuple[
        wp.array[wp.vec3d],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.vec3d],
        wp.array[wp.int32],
    ],
    mesh_id: int,
    epsilon: float,
    tolerance: float,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    *,
    sorted_: bool = True,
) -> Solid:
    """
    Return the solid of paired faces: vertex pseudo-normals, then the Morton order.

    ``paired`` holds the positions, faces, twins, each vertex's least outgoing halfedge, the face
    normals and the coplanar groups; without ``sorted_`` the order is kept.
    """
    positions, faces, twins, first_halfedge, face_normals, coplanar = paired
    device = positions.device
    n_vertices = int(positions.shape[0])
    vertex_normals = _launch.empty(n_vertices, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_solid.vertex_pseudo_normals,
        dim=n_vertices,
        inputs=[faces, twins, positions, face_normals, first_halfedge],
        outputs=[vertex_normals],
        device=device,
    )
    face_mesh = _launch.full(faces.size // 3, mesh_id, dtype=wp.int32, device=device)
    if not sorted_:
        return Solid(
            positions,
            faces,
            face_normals,
            vertex_normals,
            face_mesh,
            coplanar,
            epsilon,
            tolerance,
            lower,
            upper,
        )
    return _sorted_solid(
        positions,
        faces,
        vertex_normals,
        face_normals,
        face_mesh,
        coplanar,
        epsilon,
        tolerance,
        lower,
        upper,
    )


def _paired_halfedges(
    positions: wp.array[wp.vec3d], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], tuple[int, int, int]]:
    """
    Return the halfedges' twins, each vertex's least outgoing halfedge, and the defects.

    The defects are the counts of edges with three or more faces, of edges whose faces wind the
    same way, and of unpaired halfedges.
    """
    device = positions.device
    n_vertices = int(positions.shape[0])
    n_faces = faces.size // 3
    defects = _launch.zeros(3, dtype=wp.int32, device=device)
    twins = halfedge_twins(faces, n_vertices, defect_counts=defects)
    first_halfedge = _launch.full(n_vertices, 2147483647, dtype=wp.int32, device=device)
    lowers = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    uppers = _launch.empty(n_faces, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_solid.operand_boxes,
        dim=n_faces,
        inputs=[positions, faces, twins, 0.0],
        outputs=[lowers, uppers, first_halfedge, _launch.view(defects, 2, 3)],
        device=device,
    )
    # Host readback: whether the mesh is closed and paired decides whether there is a solid.
    non_manifold, inconsistent, boundary = read_values(defects, 0, 3)
    return twins, first_halfedge, (int(non_manifold), int(inconsistent), int(boundary))


def _require_closed(defects: tuple[int, int, int], name: str) -> None:
    """
    Raise unless the pairing found no defect.

    Raises
    ------
    ValueError
        If the mesh is not closed, edge-manifold and consistently wound.
    """
    non_manifold, inconsistent, boundary = defects
    if non_manifold or inconsistent or boundary:
        raise ValueError(
            f"mesh {name} must be closed, edge-manifold and consistently wound: {boundary} "
            f"boundary halfedges, {non_manifold} edges with three or more faces, {inconsistent} "
            "edges whose faces wind the same way"
        )


def empty_solid(device: wp.DeviceLike) -> Solid:
    """Return the empty solid, whose box is Manifold's default ``Box``, meeting no other box."""
    inf = math.inf
    return Solid(
        _launch.empty(0, dtype=wp.vec3d, device=device),
        _launch.empty(0, dtype=wp.int32, device=device),
        _launch.empty(0, dtype=wp.vec3d, device=device),
        _launch.empty(0, dtype=wp.vec3d, device=device),
        _launch.empty(0, dtype=wp.int32, device=device),
        _launch.empty(0, dtype=wp.int32, device=device),
        -1.0,
        -1.0,
        (inf, inf, inf),
        (-inf, -inf, -inf),
    )


def mesh_from_solid(
    solid: Solid, device: wp.DeviceLike
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Return a solid as a ``float32`` mesh, as Manifold's ``MeshGL`` lays it out.

    Vertices in the solid's Morton order; faces in it too, stably grouped by the input mesh each
    came from (``GetMeshGL``'s runs).
    """
    n_vertices = solid.n_vertices
    n_faces = solid.n_faces
    vertices = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
    if n_faces == 0:
        return vertices, _launch.empty(0, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_solid.narrow_positions,
        dim=n_vertices,
        inputs=[solid.positions],
        outputs=[vertices],
        device=device,
    )
    keys, order = _launch.sort_buffers(n_faces, wp.int32, device)
    _launch.launch(
        kernel_array.KEYS_AND_IDENTITY[wp.int32],
        dim=n_faces,
        inputs=[solid.face_mesh],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(keys, order, n_faces)
    faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_solid.gather_faces,
        dim=n_faces,
        inputs=[solid.faces, order],
        outputs=[faces],
        device=device,
    )
    return vertices, faces


def positions_box(
    positions: wp.array[wp.vec3d],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``CalculateBBox`` over the finite positions, read back."""
    device = positions.device
    inf = float("inf")
    box = _launch.array([inf, inf, inf, -inf, -inf, -inf], dtype=wp.float64, device=device)
    n = int(positions.shape[0])
    items = kernel_solid.POSITIONS_BOX_ITEMS
    _launch.launch_tiled(
        kernel_solid.positions_box,
        dim=max((n + items - 1) // items, 1),
        inputs=[positions, inf],
        outputs=[box],
        block_dim=kernel_solid.POSITIONS_BOX_BLOCK_DIM,
        device=device,
    )
    # Host readback: the box frames the Morton codes and sets the tolerances.
    values = [float(x) for x in read_values(box, 0, 6)]
    return (values[0], values[1], values[2]), (values[3], values[4], values[5])


def coplanar_groups(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3d],
    tolerance: float,
) -> wp.array[wp.int32]:
    """
    ``SetNormalsAndCoplanar``: each face's coplanar group, its normal set to its seed's.

    Faces in decreasing area seed groups that spread across edges to neighbours within
    ``tolerance`` of the seed's plane; ``face_normals`` is updated in place.
    """
    device = positions.device
    n_faces = faces.size // 3
    keys, order = _launch.sort_buffers(n_faces, wp.float64, device)
    _launch.launch(
        kernel_solid.face_area_keys,
        dim=n_faces,
        inputs=[positions, faces],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(keys, order, n_faces)
    rank = _launch.empty(n_faces, dtype=wp.int32, device=device)
    _, seed_flags = offsets_buffer(n_faces, device)
    _launch.launch(
        kernel_solid.coplanar_seed_flags,
        dim=n_faces,
        inputs=[faces, twins, positions, face_normals, order, tolerance],
        outputs=[rank, seed_flags],
        device=device,
    )
    is_seed = _launch.clone(seed_flags)
    n_seeds = scan_count(seed_flags)
    coplanar = _launch.full(n_faces, -1, dtype=wp.int32, device=device)
    if n_seeds > 0:
        seeds = _launch.empty(n_seeds, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_scatter.scatter_index_where_scanned,
            dim=n_faces,
            inputs=[seed_flags],
            outputs=[seeds],
            device=device,
        )
        flood = (faces, twins, positions, order, rank, is_seed, tolerance, face_normals, coplanar)
        if n_seeds <= _SERIAL_FLOOD_SEEDS or wp.get_device(device).is_cpu:
            _serial_flood(flood, seeds, n_seeds)
        else:
            _flood_in_rounds(flood, seeds, n_seeds)
    _launch.launch(kernel_solid.own_coplanar_groups, dim=n_faces, inputs=[coplanar], device=device)
    return coplanar


def _serial_flood(flood: tuple[object, ...], seeds: wp.array[wp.int32], n_seeds: int) -> None:
    """Flood the seeds' coplanar groups one after another in one thread, in area order."""
    faces, twins, positions, order, rank, is_seed, tolerance, face_normals, coplanar = flood
    device = seeds.device
    n_faces = rank.size  # pyright: ignore[reportAttributeAccessIssue]
    inputs = [
        faces,
        twins,
        positions,
        seeds,
        n_seeds,
        order,
        rank,
        is_seed,
        tolerance,
        face_normals,
        _launch.empty(2 * n_faces + 4, dtype=wp.int32, device=device),
    ]
    if wp.get_device(device).is_cpu or n_seeds <= _SERIAL_FLOOD_SEEDS:
        _launch.launch(
            kernel_solid.flood_coplanar, dim=1, inputs=inputs, outputs=[coplanar], device=device
        )
    else:
        # A long flood is a serial walk of dependent loads: one CPU thread runs it far faster.
        launch_on_host(kernel_solid.flood_coplanar, inputs, [coplanar])


def _flood_in_rounds(flood: tuple[object, ...], seeds: wp.array[wp.int32], n_seeds: int) -> None:
    """
    Flood the seeds' coplanar groups in parallel rounds, the same groups as the serial flood.

    Each round every pending seed floods tentatively and bids on its region; the seeds holding
    their region's lowest bid commit together (``kernels/boolean.tentative_flood``). Once a round
    leaves more than half its seeds pending, the serial flood takes every seed left.
    """
    device = seeds.device
    n_faces = flood[4].size  # pyright: ignore[reportAttributeAccessIssue]
    bids = _launch.empty(n_faces, dtype=wp.int32, device=device)
    scratch = _launch.empty(
        n_seeds * 4 * kernel_solid.COPLANAR_REGION, dtype=wp.int32, device=device
    )
    first_overflow = _launch.empty(1, dtype=wp.int32, device=device)
    commit = _launch.empty(n_seeds, dtype=wp.int32, device=device)
    pending = seeds
    m = n_seeds
    while m > 0:
        bids.fill_(2147483647)
        first_overflow.fill_(2147483647)
        left_offsets, left = offsets_buffer(m, device)
        _launch.launch(
            kernel_solid.coplanar_round_bids,
            dim=m,
            inputs=[*flood, pending, bids, scratch],
            outputs=[first_overflow],
            device=device,
        )
        _launch.launch(
            kernel_solid.coplanar_round_checks,
            dim=m,
            inputs=[*flood, pending, bids, scratch, first_overflow],
            outputs=[commit],
            device=device,
        )
        _launch.launch(
            kernel_solid.coplanar_round_commits,
            dim=m,
            inputs=[*flood, pending, bids, scratch, commit],
            outputs=[left],
            device=device,
        )
        # Host readback: the next round's width.
        n_left = scan_count(left)
        if 2 * n_left > m:
            # Large regions overflow or overlap round after round (a round that commits nothing
            # has an overflowing seed as its earliest): the serial flood takes every seed left,
            # in order, from the claims so far (2026-11-21: an icosphere's 82 k faces at a loose
            # tolerance 18 -> 4.2 ms; quad strips commit all but a few in their first round).
            _serial_flood(flood, pending, m)
            return
        if n_left > 0:
            remaining = _launch.empty(n_left, dtype=wp.int32, device=device)
            _launch.launch(
                kernel_solid.compact_pending,
                dim=m,
                inputs=[pending, left_offsets],
                outputs=[remaining],
                device=device,
            )
            pending = remaining
        m = n_left


def _morton_sorted(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    vertex_normals: wp.array[wp.vec3d] | None = None,
    face_normals: wp.array[wp.vec3d] | None = None,
    face_mesh: wp.array[wp.int32] | None = None,
    face_coplanar: wp.array[wp.int32] | None = None,
) -> tuple[
    wp.array[wp.vec3d],
    wp.array[wp.int32],
    wp.array[wp.vec3d],
    wp.array[wp.vec3d],
    wp.array[wp.int32],
    wp.array[wp.int32],
    wp.array[wp.int32],
]:
    """
    Renumber vertices and faces in Morton order within the box, as ``SortGeometry`` does.

    Returns the sorted arrays, then each sorted face's old index.

    The order decides nothing geometric; it is the order Manifold's ties fall in (which halfedge of
    an edge is forward, which edge collapses first), so following it reproduces its answers in
    the degenerate cases too. The attributes, where given, move with their vertex or face.
    """
    device = positions.device
    n_vertices = int(positions.shape[0])
    n_faces = faces.size // 3
    low = wp.vec3d(*lower)
    high = wp.vec3d(*upper)
    keys, order = _launch.sort_buffers(n_vertices, wp.uint32, device)
    _launch.launch(
        kernel_solid.vertex_morton_keys,
        dim=n_vertices,
        inputs=[positions, low, high],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(keys, order, n_vertices)
    with_normals = vertex_normals is not None
    sorted_positions = _launch.empty(n_vertices, dtype=wp.vec3d, device=device)
    sorted_normals = _launch.empty(n_vertices if with_normals else 0, dtype=wp.vec3d, device=device)
    old_to_new = _launch.empty(n_vertices, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_solid.renumber_vertices,
        dim=n_vertices,
        inputs=[
            positions,
            vertex_normals if vertex_normals is not None else positions,
            order,
            1 if with_normals else 0,
        ],
        outputs=[sorted_positions, sorted_normals, old_to_new],
        device=device,
    )
    keys, order = _launch.sort_buffers(n_faces, wp.uint32, device)
    _launch.launch(
        kernel_solid.face_morton_keys,
        dim=n_faces,
        inputs=[positions, faces, low, high],
        outputs=[keys, order],
        device=device,
    )
    _launch.radix_sort_pairs(keys, order, n_faces)
    with_attributes = face_normals is not None
    n_attributes = n_faces if with_attributes else 0
    sorted_faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    sorted_face_normals = _launch.empty(n_attributes, dtype=wp.vec3d, device=device)
    sorted_face_mesh = _launch.empty(n_attributes, dtype=wp.int32, device=device)
    sorted_face_coplanar = _launch.empty(n_attributes, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_solid.renumber_faces,
        dim=n_faces,
        inputs=[
            faces,
            face_normals if face_normals is not None else positions,
            face_mesh if face_mesh is not None else faces,
            face_coplanar if face_coplanar is not None else faces,
            order,
            old_to_new,
            1 if with_attributes else 0,
        ],
        outputs=[sorted_faces, sorted_face_normals, sorted_face_mesh, sorted_face_coplanar],
        device=device,
    )
    return (
        sorted_positions,
        sorted_faces,
        sorted_normals,
        sorted_face_normals,
        sorted_face_mesh,
        sorted_face_coplanar,
        _launch.view(order, 0, n_faces),
    )


def halfedge_mesh(
    n_triangles: int,
    n_vertices: int,
    positions: wp.array[wp.vec3d],
    device: wp.DeviceLike,
    *,
    with_source: bool = False,
) -> _HalfedgeMesh:
    """
    Allocate the result's halfedges with room for the simplification to grow them.

    With ``with_source``, room for each triangle's source face (``tri_source``) too; without, it is
    empty and the passes keep none.
    """
    n_halfedges = 3 * n_triangles
    capacity = 2 * n_halfedges + 64
    start = _launch.empty(capacity, dtype=wp.int32, device=device)
    pair = _launch.empty(capacity, dtype=wp.int32, device=device)
    tri_mesh = _launch.empty(capacity // 3, dtype=wp.int32, device=device)
    tri_coplanar = _launch.empty(capacity // 3, dtype=wp.int32, device=device)
    tri_normal = _launch.empty(capacity // 3, dtype=wp.vec3d, device=device)
    tri_source = _launch.empty(capacity // 3 if with_source else 0, dtype=wp.int32, device=device)
    # The simplification appends vertices too: at most two a collapse or split, fewer than the
    # halfedges.
    pos = _launch.empty(n_vertices + capacity, dtype=wp.vec3d, device=device)
    _launch.copy(pos, positions, count=n_vertices)
    sizes = _launch.array(
        [n_halfedges, n_vertices, capacity // 3 * 3, n_vertices + capacity],
        dtype=wp.int32,
        device=device,
    )
    struct = kernel_solid_topology.HalfedgeMesh()
    struct.start = start
    struct.pair = pair
    struct.tri_mesh = tri_mesh
    struct.tri_coplanar = tri_coplanar
    struct.tri_normal = tri_normal
    struct.tri_source = tri_source
    struct.pos = pos
    struct.sizes = sizes
    return cast(_HalfedgeMesh, struct)


def simplify_topology(
    result: _HalfedgeMesh,
    sizes: tuple[int, int],
    first_new_vert: int,
    epsilon: float,
    tolerance: float,
    device: wp.DeviceLike,
    *,
    colinear: bool = True,
    cleanup_only: bool = False,
    cleanup: bool = True,
) -> tuple[int, int]:
    """
    ``SimplifyTopology(firstNewVert)`` on the result's halfedges, in place.

    ``sizes`` are the halfedge and vertex counts the mesh holds (its ``sizes``' first two
    entries); the counts it holds afterwards are returned. Without ``colinear``,
    ``RemoveDegenerates``: the same less the collinear-edge collapses. With ``cleanup_only``,
    ``CleanupTopology`` alone: pinched vertices split, duplicate edges removed. Without
    ``cleanup``, the passes after it alone.
    """
    mesh = result
    capacity = result.start.size
    flags_offsets, flags = offsets_buffer(capacity, device)
    flagged = _launch.empty(capacity, dtype=wp.int32, device=device)
    collapsed = _launch.zeros(1, dtype=wp.int32, device=device)
    # The serial passes' scratch: device buffers on the CPU device, host ones on CUDA.
    on_cpu = wp.get_device(device).is_cpu
    scratch_arg: object = (
        _launch.empty(capacity, dtype=wp.int32, device=device) if on_cpu else _HostScratch(capacity)
    )
    swap_scratch: list[object] = (
        [
            _launch.full(capacity, -1, dtype=wp.int32, device=device),
            _launch.empty(8 * capacity, dtype=wp.int32, device=device),
        ]
        if on_cpu
        else [_HostScratch(capacity, -1), _HostScratch(8 * capacity)]
    )

    # The counts as they stand: only the serial passes and the pinch split change them, and each
    # hands back its new counts, so the flag passes' grids need no readback.
    known = list(sizes)
    # A CUDA pass of few flagged edges for the mesh's size runs in one GPU thread, on device
    # scratch made once.
    device_scratch: dict[_HostScratch, odt.ArrayNd] = {}

    def serial(kernel: odt.Kernel, inputs: list[object], outputs: list[object]) -> None:
        count = cast(int, inputs[1])
        if on_cpu or count > _SERIAL_DEVICE_BASE + known[0] // _SERIAL_DEVICE_HALFEDGES:
            known[:] = _launch_serial(result, (known[0], known[1]), kernel, inputs, outputs)
            return
        plain: list[object] = []
        for value in inputs:
            if isinstance(value, _HostScratch):
                if value not in device_scratch:
                    device_scratch[value] = (
                        _launch.empty(value.size, dtype=wp.int32, device=device)
                        if value.fill is None
                        else _launch.full(value.size, value.fill, dtype=wp.int32, device=device)
                    )
                plain.append(device_scratch[value])
            else:
                plain.append(_device_value(value))
        _launch.launch(kernel, dim=1, inputs=[result, *plain], outputs=outputs, device=device)
        # Host readback: the pass grew the mesh, and the next pass's grid follows it.
        n_h, n_v = read_values(result.sizes, 0, 2)
        known[:] = [int(n_h), int(n_v)]

    def flag(kernel: odt.Kernel, inputs: list[object]) -> int:
        n_h = known[0]
        # Every flag kernel writes its whole range, so the buffer is reused unzeroed.
        _launch.launch(kernel, dim=n_h, inputs=[mesh, *inputs], outputs=[flags], device=device)
        count = scan_count(_launch.view(flags, 0, n_h))
        if count > 0:
            _launch.launch(
                kernel_scatter.scatter_index_where_scanned,
                dim=n_h,
                inputs=[flags],
                outputs=[flagged],
                device=device,
            )
        return count

    # ``CleanupTopology``: split pinched vertices, then duplicate edges until none is left; skipped
    # (``cleanup=False``) right after a ``cleanup_only`` call, which leaves nothing to find.
    if cleanup:
        n_h, n_v = known
        fan_min = _launch.empty(capacity, dtype=wp.int32, device=device)
        vertex_min = _launch.full(result.pos.shape[0], 2147483647, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_solid_topology.flag_pinched_fans,
            dim=n_h,
            inputs=[mesh],
            outputs=[fan_min, vertex_min],
            device=device,
        )
        _launch.launch(
            kernel_solid_topology.count_pinched_fans,
            dim=n_h,
            inputs=[mesh, fan_min, vertex_min],
            outputs=[flags],
            device=device,
        )
        n_pinched = scan_count(_launch.view(flags, 0, n_h))
        if n_pinched > 0:
            _launch.launch(
                kernel_solid_topology.split_pinched_fans,
                dim=n_h,
                inputs=[mesh, _launch.view(flags_offsets, 0, n_h + 1), n_v],
                device=device,
            )
            _launch.copy(
                result.sizes, _launch.array([n_v + n_pinched], dtype=wp.int32, device=device), 1
            )
            known[1] = n_v + n_pinched
        while True:
            count = flag(kernel_solid_topology.flag_duplicate_edges, [])
            if count == 0:
                break
            serial(kernel_solid_topology.dedupe_flagged, [_HostIn(flagged, count), count], [])
    if cleanup_only:
        return known[0], known[1]
    # ``CollapseShortEdges`` (to ``tolerance`` only past the new vertices: everywhere it is
    # ``epsilon``), ``CollapseColinearEdges``, ``SwapDegenerates``.
    short_tolerance = tolerance if first_new_vert > 0 else epsilon
    count = flag(kernel_solid_topology.flag_short_edges, [first_new_vert, short_tolerance, epsilon])
    if count > 0:
        serial(
            kernel_solid_topology.collapse_flagged,
            [_HostIn(flagged, count), count, short_tolerance, first_new_vert, epsilon, scratch_arg],
            [collapsed],
        )
    while colinear:
        count = flag(kernel_solid_topology.flag_colinear_edges, [first_new_vert])
        if count == 0:
            break
        serial(
            kernel_solid_topology.collapse_flagged,
            [_HostIn(flagged, count), count, epsilon, 0, epsilon, scratch_arg],
            [collapsed],
        )
        # Host readback: ``CollapseColinearEdges`` repeats until a pass collapses nothing.
        if read_values(collapsed, 0, 1)[0] == 0:
            break
    count = flag(kernel_solid_topology.flag_swappable_edges, [first_new_vert, tolerance])
    if count > 0:
        serial(
            kernel_solid_topology.swap_flagged,
            [_HostIn(flagged, count), count, tolerance, epsilon, *swap_scratch, scratch_arg],
            [],
        )
    return known[0], known[1]


def _launch_serial(
    result: _HalfedgeMesh,
    sizes: tuple[int, int],
    kernel: odt.Kernel,
    inputs: list[object],
    outputs: list[object],
) -> tuple[int, int]:
    """
    Launch a one-thread pass over the result's halfedges, on the CPU when they are on CUDA.

    The collapses and swaps are a serial walk of dependent loads, which one CPU thread runs far
    faster than one GPU thread. Only the used prefixes travel: the halfedges, triangles and
    vertices the ``sizes`` count, each ``_HostIn`` its count, each ``_HostScratch`` nothing (a
    fresh host buffer, optionally filled); the outputs go back whole and the mesh's prefixes as
    the pass left them. Returns the halfedge and vertex counts the pass left.
    """
    device = result.start.device
    if wp.get_device(device).is_cpu:
        plain = [_device_value(value) for value in inputs]
        _launch.launch(kernel, dim=1, inputs=[result, *plain], outputs=outputs, device=device)
        # Host readback: the pass grew the mesh, and the next pass's grid follows it.
        n_h, n_v = read_values(result.sizes, 0, 2)
        return int(n_h), int(n_v)
    host = wp.get_device("cpu")
    n_h, n_v = sizes
    struct = kernel_solid_topology.HalfedgeMesh()
    prefixes = {
        "start": n_h,
        "pair": n_h,
        "tri_mesh": n_h // 3,
        "tri_coplanar": n_h // 3,
        "tri_normal": n_h // 3,
        "tri_source": n_h // 3,
        "pos": n_v,
    }
    host_fields: dict[str, odt.ArrayNd] = {}
    for name, count in prefixes.items():
        array = getattr(result, name)
        copy = _launch.empty(array.shape[0], dtype=array.dtype, device=host)
        if count > 0 and array.shape[0] > 0:
            _launch.copy(copy, array, count=count)
        setattr(struct, name, copy)
        host_fields[name] = copy
    host_sizes = _launch.clone(result.sizes, device=host)
    struct.sizes = host_sizes
    host_inputs: list[object] = []
    for value in inputs:
        if isinstance(value, _HostIn):
            copy = _launch.empty(value.array.shape[0], dtype=value.array.dtype, device=host)
            if value.used > 0:
                _launch.copy(copy, value.array, count=value.used)
            host_inputs.append(copy)
        elif isinstance(value, _HostScratch):
            if value.fill is None:
                host_inputs.append(_launch.empty(value.size, dtype=wp.int32, device=host))
            else:
                host_inputs.append(
                    _launch.full(value.size, value.fill, dtype=wp.int32, device=host)
                )
        else:
            host_inputs.append(value)
    host_outputs = [_launch.clone(cast(odt.ArrayNd, value), device=host) for value in outputs]
    _launch.launch(kernel, dim=1, inputs=[struct, *host_inputs], outputs=host_outputs, device=host)
    sizes_np = read_array(host_sizes)
    n_h, n_v = int(sizes_np[0]), int(sizes_np[1])
    prefixes = {
        k: (n_h if k in ("start", "pair") else n_v if k == "pos" else n_h // 3) for k in prefixes
    }
    for name, count in prefixes.items():
        if count > 0 and host_fields[name].shape[0] > 0:
            _launch.copy(getattr(result, name), host_fields[name], count=count)
    _launch.copy(result.sizes, host_sizes)
    for array, copy in zip(outputs, host_outputs, strict=True):
        _launch.copy(cast(odt.ArrayNd, array), copy)
    return n_h, n_v


class _HostIn(NamedTuple):
    """A serial pass's input array, of which the first ``used`` entries are read."""

    array: wp.array[wp.int32]
    used: int


class _HostScratch(NamedTuple):
    """A serial pass's scratch of ``size`` entries, arriving filled with ``fill`` if given."""

    size: int
    fill: int | None = None


def _device_value(value: object) -> object:
    """Return a serial pass's argument as the device launch takes it."""
    if isinstance(value, _HostIn):
        return value.array
    if isinstance(value, _HostScratch):
        raise AssertionError("a device launch is handed its scratch")
    return value


def launch_on_host(
    kernel: odt.Kernel,
    inputs: list[object],
    outputs: list[object],
    struct_fields: dict[str, odt.ArrayNd] | None = None,
) -> None:
    """
    Launch a one-thread kernel on the CPU over device arrays: each goes to the host and back.

    The first input may be a struct whose ``struct_fields`` are filled with the host copies.
    """
    host = wp.get_device("cpu")
    moved: list[tuple[odt.ArrayNd, odt.ArrayNd]] = []

    def to_host(value: object) -> object:
        if isinstance(value, wp.array):
            copy = _launch.clone(value, device=host)
            moved.append((value, copy))
            return copy
        return value

    for name, array in (struct_fields or {}).items():
        setattr(inputs[0], name, to_host(array))
    host_inputs = [to_host(value) for value in inputs]
    host_outputs = [to_host(value) for value in outputs]
    _launch.launch(kernel, dim=1, inputs=host_inputs, outputs=host_outputs, device=host)
    for array, copy in moved:
        _launch.copy(array, copy)


def finish_solid(
    result: _HalfedgeMesh,
    sizes: tuple[int, int],
    epsilon: float,
    tolerance: float,
    device: wp.DeviceLike,
    box: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
) -> Solid:
    """
    Return the simplified result as a solid.

    ``CalculateVertNormals`` (the end of ``SimplifyTopology``) on the halfedges as they stand,
    ``RemoveUnreferencedVerts`` with the collapsed triangles, then ``SortGeometry`` in the kept
    vertices' box (``CalculateBBox``), or in ``box`` where the caller never recalculates it.
    ``sizes`` are the halfedge and vertex counts ``simplify_topology`` returned.
    """
    mesh = result
    n_h, n_v = sizes
    n_t = n_h // 3
    first = _launch.full(n_v, 2147483647, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_solid.least_outgoing_halfedges,
        dim=n_h,
        inputs=[mesh.start],
        outputs=[first],
        device=device,
    )
    vertex_normals = _launch.empty(n_v, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_solid_topology.hm_vertex_normals,
        dim=n_v,
        inputs=[mesh, first],
        outputs=[vertex_normals],
        device=device,
    )
    kept = compacted(result, vertex_normals, n_t, n_v)
    if kept is None:
        return empty_solid(device)
    positions, faces, kept_normals, face_normals, face_mesh, face_coplanar, source = kept
    lower, upper = positions_box(positions) if box is None else box
    solid = _sorted_solid(
        positions,
        faces,
        kept_normals,
        face_normals,
        face_mesh,
        face_coplanar,
        epsilon,
        tolerance,
        lower,
        upper,
    )
    if source is None or solid.face_source is None:
        return solid
    # Each sorted face's source: the kept face it was, read through the kept faces' sources.
    return solid._replace(face_source=gather(source, solid.face_source))


def compacted(
    result: _HalfedgeMesh, vertex_normals: wp.array[wp.vec3d], n_t: int, n_v: int
) -> (
    tuple[
        wp.array[wp.vec3d],
        wp.array[wp.int32],
        wp.array[wp.vec3d],
        wp.array[wp.vec3d],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32] | None,
    ]
    | None
):
    """
    ``RemoveUnreferencedVerts`` with the removed triangles, order kept.

    Returns the kept positions, faces, vertex normals and each face's normal, mesh, coplanar
    group and source face (``None`` where the mesh keeps no sources), or ``None`` when no triangle
    is left.
    """
    mesh = result
    device = result.start.device
    tri_offsets, tri_kept = offsets_buffer(n_t, device)
    vert_offsets, vert_used = offsets_buffer(n_v, device)
    _launch.launch(
        kernel_solid_topology.mark_kept,
        dim=n_t,
        inputs=[mesh],
        outputs=[tri_kept, vert_used],
        device=device,
    )
    n_kept = scan_count(tri_kept)
    n_used = scan_count(vert_used)
    if n_kept == 0:
        return None
    faces = _launch.empty(3 * n_kept, dtype=wp.int32, device=device)
    face_normals = _launch.empty(n_kept, dtype=wp.vec3d, device=device)
    face_mesh = _launch.empty(n_kept, dtype=wp.int32, device=device)
    face_coplanar = _launch.empty(n_kept, dtype=wp.int32, device=device)
    face_source = (
        _launch.empty(n_kept, dtype=wp.int32, device=device)
        if mesh.tri_source.shape[0] > 0
        else None
    )
    _launch.launch(
        kernel_solid_topology.write_kept_faces,
        dim=n_t,
        inputs=[mesh, tri_offsets, vert_offsets],
        outputs=[faces, face_normals, face_mesh, face_coplanar, face_source],
        device=device,
    )
    positions = _launch.empty(n_used, dtype=wp.vec3d, device=device)
    kept_normals = _launch.empty(n_used, dtype=wp.vec3d, device=device)
    _launch.launch(
        kernel_solid_topology.write_kept_vertices,
        dim=n_v,
        inputs=[mesh, vertex_normals, vert_offsets],
        outputs=[positions, kept_normals],
        device=device,
    )
    return positions, faces, kept_normals, face_normals, face_mesh, face_coplanar, face_source


def _sorted_solid(
    positions: wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    vertex_normals: wp.array[wp.vec3d],
    face_normals: wp.array[wp.vec3d],
    face_mesh: wp.array[wp.int32],
    face_coplanar: wp.array[wp.int32],
    epsilon: float,
    tolerance: float,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> Solid:
    """``SortGeometry``: the solid with its vertices and faces in Morton order, attributes along."""
    (
        sorted_positions,
        sorted_faces,
        sorted_vertex_normals,
        sorted_face_normals,
        sorted_face_mesh,
        sorted_face_coplanar,
        face_source,
    ) = _morton_sorted(
        positions, faces, lower, upper, vertex_normals, face_normals, face_mesh, face_coplanar
    )
    return Solid(
        sorted_positions,
        sorted_faces,
        sorted_face_normals,
        sorted_vertex_normals,
        sorted_face_mesh,
        sorted_face_coplanar,
        epsilon,
        tolerance,
        lower,
        upper,
        face_source,
    )
