"""
Closing mesh boundary holes, and stitching two open meshes along a rim.

Every boundary of a triangle mesh is an ordered vertex loop
([`boundary_loops`][ordito.boundary.boundary_loops], the analog of ``trimesh.repair.fill_holes``'s
``nx.cycle_basis`` and ``igl::boundary_loop_all``). Each loop of ``B`` vertices is sealed with a
purely topological triangulation -- **no smoothing or refinement**:

- [`fill_fan`][ordito.holes.fill_fan] fans ``B - 2`` triangles from the loop's first vertex,
  reusing only existing vertices (``trimesh.repair.fill_holes(use_fan=True)``).
- [`fill_cone`][ordito.holes.fill_cone] inserts one centroid vertex per hole and cones ``B``
  triangles onto it (``igl::topological_hole_fill``,
  ``trimesh.repair.stitch(insert_vertices=True)``).
- [`fill_min_weight`][ordito.holes.fill_min_weight] computes the **minimum-weight triangulation**
  of each loop (the Liepa/Klincsek interval DP): the ``B - 2`` triangles over the existing loop
  vertices that minimize a geometric metric (plane-normalized circumcircle by default, with a
  min-area fallback), avoiding chords that would duplicate existing mesh edges. This is the robust,
  general-purpose filler for non-convex and non-planar holes; it adds no vertices.

Because ``boundary_loops`` orders each loop following the face-winding direction of the boundary
half-edges, a fill triangle sharing a rim edge is emitted **reversed** on that edge so its winding
is consistent with the adjacent original face. This assumes the input mesh is consistently wound;
run [`make_winding_consistent`][ordito.repair.make_winding_consistent] first on meshes with mixed
winding.

Joining **two** open meshes across one boundary loop each is the same family and lives here too:
[`stitch`][ordito.holes.stitch] and its variants, over the loop-level engines
[`stitch_loops`][ordito.holes.stitch_loops] and
[`stitch_loops_min_weight`][ordito.holes.stitch_loops_min_weight]. It is the *same* minimum-weight
machinery -- the same interval DP, rim bookkeeping and metric vocabulary -- applied to a band
between two rims rather than a cap over one.

For a smooth, well-graded patch, [`fill_smooth`][ordito.holes.fill_smooth] and
[`stitch_smooth`][ordito.holes.stitch_smooth] run a three-stage pipeline on top of the min-weight
fill/stitch: the patch is refined to a target edge length with Delaunay edge flips
([`subdivide_region_to_size`][ordito.remesh.subdivide_region_to_size]) and its new interior
vertices are smoothed into the surrounding surface -- a sharp-boundary umbrella solve
([`smooth_region_fixed_rim`][ordito.smoothing.smooth_region_fixed_rim]) followed by a
cross-boundary least-squares solve ([`smooth_region`][ordito.smoothing.smooth_region]), with an
optional ``natural_smooth`` collar that blends the patch into the neighbouring surface.

Not every boundary wants closing. [`extend_hole`][ordito.holes.extend_hole] and
[`build_bottom`][ordito.holes.build_bottom] *extrude* a rim -- out to a plane you place, or down to
a base fitted under each rim's own lowest point -- which leaves the mesh open with a planar rim that
a min-weight fill then closes without folding; that two-step is how a scanned shell becomes a
printable solid. [`bridge_edges`][ordito.holes.bridge_edges] and
[`bridge_edges_smooth`][ordito.holes.bridge_edges_smooth] are the *local* form of stitching: they
join one boundary edge to another with a small patch or a curved strip, leaving the rest of both
boundaries open, which is what joins two tubes at a chosen seam or adds a handle where the rim
family would consume the whole loop.
[`join_closest_components`][ordito.holes.join_closest_components] is the driver over the first of
those: it settles *which* edges to bridge, welding several open shells into one connected surface
with one rim, for a filler to close afterwards.

The return shape follows from that: a filler that only triangulates existing rim vertices returns
``faces`` alone, while one that inserts a vertex -- a cone's apex, a refined patch's interior, an
extrusion's projected ring -- returns ``(vertices, faces)``.

Every filler returns a buffer **independent of** ``faces``, including on the no-op path where there
was no hole to fill, so a caller may write into the result without disturbing its input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import warp as wp

# ``@wp.struct`` rebinds the decorated name to a ``Struct`` *value*, so the class itself is not
# annotatable and a helper taking a bundle has to name the instance's base instead. Private on
# Warp 1.18; an upgrade that moves it fails at import rather than silently.
from warp._src.codegen import StructInstance

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device
from ordito.constants import TOLERANCE_ZERO
from ordito.kernels import holes as kernel_holes
from ordito.kernels import scatter as kernel_scatter

if TYPE_CHECKING:
    import numpy.typing as npt


class _PackedLoops:
    """
    Every fillable loop of one mesh in one packed buffer, plus the host metadata to index it.

    ``flat_loops`` concatenates the loop vertex indices; loop ``ell`` occupies
    ``flat_loops[offsets[ell] : offsets[ell + 1]]`` (the package's total-terminated offsets, which
    every kernel here reads a loop's extent from). ``loop_id`` inverts that mapping so a
    ``dim=total`` kernel can find its own loop without a search, and ``dp_offsets`` is the exclusive
    scan of ``sizes ** 2`` — where each loop's ``B x B`` dynamic-programming block begins in the
    ragged tables, and ``triangle_offsets`` where its fill triangles begin. The device tables are
    uploaded once and shared by every stage of the fill.
    """

    def __init__(self, flat_loops: wp.array[wp.int32], sizes_np: np.ndarray) -> None:
        device = flat_loops.device
        self.device = device
        self.flat_loops = flat_loops
        self.sizes_np = sizes_np.astype(np.int64)
        self.starts_np = np.concatenate([[0], np.cumsum(self.sizes_np)[:-1]]).astype(np.int64)
        self.dp_offsets_np = np.concatenate(
            [[0], np.cumsum(self.sizes_np * self.sizes_np)[:-1]]
        ).astype(np.int64)

        self.n_loops = sizes_np.size
        self.total = int(self.sizes_np.sum())
        self.max_size = int(self.sizes_np.max())
        self.dp_total = int((self.sizes_np * self.sizes_np).sum())

        triangle_sizes_np = np.maximum(self.sizes_np - 2, 0)
        triangle_offsets_np = np.concatenate([[0], np.cumsum(triangle_sizes_np)[:-1]])

        # One upload for all four tables, each a view into it: the per-call cost of a
        # ``wp.array`` construction is several times a slice's, and none of these is written.
        tables_np = np.concatenate(
            [
                self.starts_np,
                [self.total],
                self.dp_offsets_np,
                triangle_offsets_np,
                np.repeat(np.arange(self.n_loops, dtype=np.int64), self.sizes_np),
            ]
        ).astype(np.int32)
        tables = _launch.array(tables_np, dtype=wp.int32, device=device)
        n = self.n_loops
        self.offsets = _table_view(tables, 0, n + 1)
        self.dp_offsets = _table_view(tables, n + 1, 2 * n + 1)
        self.triangle_offsets = _table_view(tables, 2 * n + 1, 3 * n + 1)
        self.loop_id = _table_view(tables, 3 * n + 1, 3 * n + 1 + self.total)

    def perimeters(self, vertices: wp.array[wp.vec3]) -> np.ndarray:
        """
        Measure the closed arc length of every packed loop, returning it on the host.

        Delegates to
        [`boundary.loop_perimeters_from_offsets`][ordito.boundary.loop_perimeters_from_offsets]
        rather than launching the segmented kernel again here: the packed layout this class holds
        *is* that function's argument list, and ``loop_id`` is passed rather than rebuilt, so the
        call costs exactly what the private copy this replaced did. ``validate=False``: ``offsets``
        is built from ``sizes_np`` right above in this class's own constructor, not handed in from
        outside, so it already fits ``flat_loops`` by construction.
        """
        return od.boundary.loop_perimeters_from_offsets(
            vertices, self.flat_loops, self.offsets, loop_id=self.loop_id, validate=False
        ).numpy()


def _table_view(tables: wp.array[wp.int32], start: int, stop: int) -> wp.array[wp.int32]:
    """
    Dense view of ``tables[start:stop]``, or an empty buffer when the range is empty.

    Warp raises on a zero-length slice at the end of a buffer, which is where an empty ``loop_id``
    lands when every packed loop is empty.
    """
    if stop > start:
        return odt.as_dense(tables[start:stop])
    return _launch.empty(0, dtype=wp.int32, device=tables.device)


def fill_fan(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], preserve_largest_hole: bool = False
) -> wp.array[wp.int32]:
    """
    Fill every boundary hole with a triangle fan from each loop's first vertex.

    A boundary loop of ``B`` vertices is sealed with ``B - 2`` triangles all sharing the loop's
    first vertex, reusing only existing vertices (``trimesh.repair.fill_holes(use_fan=True)``).
    The fill is topological: a fan is only geometrically ideal for convex holes. The vertex
    buffer is unchanged.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    preserve_largest_hole
        When ``True``, leave the single largest boundary loop (greatest perimeter) open and fill
        only the rest. This turns a mesh with a known disk-like topology into a single-boundary
        disk, which is the input a robust UV parametrization expects.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_out_faces,)`` flat face buffer of the original faces followed by the new fill
        triangles, on ``faces.device``. A watertight or empty mesh — or, with
        ``preserve_largest_hole``, a mesh whose only hole is the largest — is returned unchanged (a
        copy).

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`fill_cone`][ordito.holes.fill_cone]
    [`fill_min_weight`][ordito.holes.fill_min_weight]
        Far more robust on a non-convex or non-planar hole, and it also adds no vertices.
    [`boundary_loops`][ordito.boundary.boundary_loops]
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]

    Notes
    -----
    Fill winding is consistent with the adjacent faces only when the input mesh is consistently
    wound (see [`make_winding_consistent`][ordito.repair.make_winding_consistent]).
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    n_faces = faces.size // 3
    packed = _hole_loops(vertices, faces, preserve_largest_hole) if n_faces > 0 else None
    if packed is None:
        return _launch.clone(faces)

    flat_loops, loop_offsets = packed.flat_loops, packed.offsets
    n_loops, total = packed.n_loops, packed.total
    n_tri = total - 2 * n_loops
    fill_faces = _launch.empty(3 * n_tri, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_holes.fan_faces,
        dim=n_loops,
        inputs=[flat_loops, loop_offsets, fill_faces],
        device=device,
    )
    return od.array.concatenate([faces, fill_faces])


def fill_cone(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], preserve_largest_hole: bool = False
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Fill every boundary hole by coning it onto a new centroid vertex.

    A boundary loop of ``B`` vertices is sealed with ``B`` triangles fanning from one new vertex
    placed at the loop's centroid (``igl::topological_hole_fill``,
    ``trimesh.repair.stitch(insert_vertices=True)``). Unlike
    [`fill_fan`][ordito.holes.fill_fan] this appends one vertex per hole, so
    the vertex buffer grows.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    preserve_largest_hole
        When ``True``, leave the single largest boundary loop (greatest perimeter) open and fill
        only the rest. This turns a mesh with a known disk-like topology into a single-boundary
        disk, which is the input a robust UV parametrization expects.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` original vertices followed by one centroid per filled hole, on
        ``vertices.device``.
    new_faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` original faces followed by the new cone triangles, on
        ``faces.device``. A watertight or
        empty mesh — or, with ``preserve_largest_hole``, a mesh whose only hole is the largest —
        is returned unchanged (copies).

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`fill_fan`][ordito.holes.fill_fan]
    [`fill_min_weight`][ordito.holes.fill_min_weight]
        Far more robust on a non-convex or non-planar hole, and unlike this it adds no vertices.
    [`boundary_loops`][ordito.boundary.boundary_loops]
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]

    Notes
    -----
    Fill winding is consistent with the adjacent faces only when the input mesh is consistently
    wound (see [`make_winding_consistent`][ordito.repair.make_winding_consistent]).
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    n_faces = faces.size // 3
    packed = _hole_loops(vertices, faces, preserve_largest_hole) if n_faces > 0 else None
    if packed is None:
        return _launch.clone(vertices), _launch.clone(faces)

    flat_loops, loop_offsets = packed.flat_loops, packed.offsets
    n_loops, total = packed.n_loops, packed.total
    n_vertices = vertices.size

    centroids = _launch.empty(n_loops, dtype=wp.vec3, device=device)
    fill_faces = _launch.empty(3 * total, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_holes.cone_fill,
        dim=n_loops,
        inputs=[vertices, flat_loops, loop_offsets, wp.int32(n_vertices), centroids, fill_faces],
        device=device,
    )
    return (od.array.concatenate([vertices, centroids]), od.array.concatenate([faces, fill_faces]))


# Fill-metric name -> kernel selector (must match the METRIC_* constants in
# kernels/holes.py).
_METRIC_IDS = {
    "plane_normalized": 0,
    "min_area": 1,
    "circumscribed": 2,
    "plane": 3,
    "min_tri_angle": 4,
    "edge_length": 5,
    "universal": 6,
    "max_dihedral": 7,
    "complex_fill": 8,
}
# Metrics that accumulate with ``max`` instead of ``sum`` (kernel COMBINE_MAX == 1); default sum.
_METRIC_COMBINE = {"max_dihedral": 1}
_BAD_TRIANGULATION_METRIC = 1e10  # kernel ``BAD_METRIC``; a forced-bad triangulation reaches it.


class _EdgeTable:
    """
    Device rim-edge lookups for the pre-DP hole-fill stage.

    ``rim_opposite`` finds, per rim edge, the opposite vertex of its single adjacent face (and
    that it has exactly one); a ``(n_vertices,)`` slot scratch
    supports the forbidden-chord mask of every loop at once. Both answers are rim-sized, so neither
    builds a table of the mesh's edges: the one sort is of the rim's own keys.
    """

    def __init__(self, vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> None:
        device = faces.device
        # Derived from the face indices rather than taken from ``len(vertices)``, deliberately: the
        # ``min`` half of this reduction is the only negative-index guard between a malformed face
        # buffer and the gathers below, which would otherwise read out of bounds instead of raising.
        n_vertices = od.array.index_bound(faces, require_non_negative=True)
        self.vertices = vertices
        self.faces = faces
        self.n_halfedges = faces.size // 3 * 3
        self.base = wp.uint64(n_vertices)
        self.n_vertices = n_vertices
        self.device = device

    def rim_opposite(
        self, loops: _PackedLoops, keys: wp.array[wp.uint64] | None = None
    ) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
        """
        Opposite-vertex position + validity per rim edge, for every loop in one pass.

        ``keys`` are the rim edges' undirected keys under ``self.base``, for a caller whose own
        pass over the rim already wrote them (the fill's ``loop_rim_metrics``); ``None`` keys them
        here.
        """
        if keys is None:
            keys = _launch.empty(loops.total, dtype=wp.uint64, device=self.device)
            _launch.launch(
                kernel_holes.rim_edge_keys,
                dim=loops.total,
                inputs=[loops.flat_loops, loops.loop_id, loops.offsets, self.base, keys],
                device=self.device,
            )
        sorted_keys, slots = od.array.sort_and_argsort(keys)
        third = _launch.full(
            loops.total, kernel_holes.RIM_NO_FACE, dtype=wp.int32, device=self.device
        )
        _launch.launch(
            kernel_holes.probe_rim_edges,
            dim=self.faces.size // 3,
            inputs=[self.faces, sorted_keys, slots, self.base, third],
            device=self.device,
        )
        positions = _launch.empty(loops.total, dtype=wp.vec3, device=self.device)
        valid = _launch.empty(loops.total, dtype=wp.int32, device=self.device)
        _launch.launch(
            kernel_holes.rim_opposite_positions,
            dim=loops.total,
            inputs=[self.vertices, third, positions, valid],
            device=self.device,
        )
        return positions, valid

    def forbidden_chords(self, loops: _PackedLoops) -> wp.array[wp.int32]:
        """
        Ragged mask of the chords that already exist as mesh edges, for every loop at once.

        The ``(n_vertices,)`` scratch holds each vertex's *flat* slot rather than its position
        within one loop, which is what lets every loop's mask be marked by a single pass over the
        mesh edges — the per-loop version ran one full ``dim = 3 * n_faces`` pass per hole, and
        cleared the scratch between them.
        """
        slot = _launch.full(self.n_vertices, -1, dtype=wp.int32, device=self.device)
        _launch.launch(
            kernel_holes.scatter_loop_positions,
            dim=loops.total,
            inputs=[loops.flat_loops, slot],
            device=self.device,
        )
        mask = _launch.zeros(loops.dp_total, dtype=wp.int32, device=self.device)
        _launch.launch(
            kernel_holes.mark_forbidden_chords,
            dim=self.n_halfedges,
            inputs=[self.faces, slot, loops.loop_id, loops.offsets, loops.dp_offsets, mask],
            device=self.device,
        )
        return mask


def fill_min_weight(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    metric: str = "plane_normalized",
    resolve_multiple_edges: bool = True,
    preserve_largest_hole: bool = False,
    smooth_boundary: bool = True,
) -> wp.array[wp.int32]:
    """
    Fill every boundary hole with a minimum-weight triangulation over its existing vertices.

    The classic Liepa/Klincsek interval dynamic program: each boundary loop of ``B`` vertices is
    sealed with the ``B - 2`` triangles that minimize a geometric metric, reusing only existing
    vertices (the vertex buffer is unchanged). This is far more robust
    than [`fill_fan`][ordito.holes.fill_fan] for non-convex or non-planar holes
    and, unlike [`fill_cone`][ordito.holes.fill_cone], adds no vertices.

    The ``O(B^3)`` DP runs on device as one parallel kernel launch per triangulation span, and
    **every hole is solved in the same launches**: the per-loop ``B x B`` tables are packed into one
    ragged buffer, so the launch count is ``max(B) - 1`` for the whole mesh rather than ``B - 1``
    per hole, and the chord test, the plane normals and the min-area fallback decision are likewise
    one pass each. The traceback runs on the device too, one thread per rim, so the predecessor
    table never crosses the bus. A mesh with many small holes therefore costs about what one hole
    costs.

    Each span launch puts a **block** on every interval rather than a thread, with the block's lanes
    striding the apex loop and a two-stage tile reduction picking the winner: the interval grid is
    only ``n_loops * (max(B) - span)`` wide, so a mesh with a couple of long rims would otherwise
    have only a few hundred threads carrying the whole cubic term. The reduction reproduces the DP's
    smallest-apex tie-break exactly, which it has to, because the tie decides the triangles (see
    ``kernels/holes.py::fill_dp_span_tiled``).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    metric
        Which fill metric to minimize:

        - ``"plane_normalized"`` (default) — circumcircle-diameter times aspect ratio, penalizing
          triangles flipped or tilted more than 60 degrees off the hole plane; falls back to
          ``"min_area"`` when the best triangulation is still bad (non-planar/degenerate).
        - ``"min_area"`` — summed triangle area; never rejects a triangulation as bad, which is
          what makes it the fallback the other metrics fall back *to*. That is a statement about
          the metric only: see ``resolve_multiple_edges`` for the one thing that can still leave an
          interval unfilled under it.
        - ``"circumscribed"`` — summed circumcircle diameter.
        - ``"plane"`` — circumcircle diameter with a flipped-normal penalty.
        - ``"min_tri_angle"`` — maximizes the minimal triangle angle.
        - ``"edge_length"`` — summed new-edge length.
        - ``"universal"`` — circumcircle diameter plus a dihedral-smoothing edge term; the smooth,
          general-purpose choice.
        - ``"max_dihedral"`` — minimizes the maximal dihedral angle.
        - ``"complex_fill"`` — area/aspect triangle term plus a strong dihedral edge term.

        The dihedral (edge-based) metrics — ``universal``, ``max_dihedral``, ``complex_fill``,
        ``edge_length`` — blend into the surrounding surface via ``smooth_boundary``.
    resolve_multiple_edges
        When ``True`` (default), forbid the triangulation from creating a chord that duplicates an
        existing mesh edge, avoiding non-manifold results on pinched holes. Such a chord is
        forbidden outright rather than re-routed, and the constraint binds under **every** metric
        including the ``min_area`` fallback. An interval with no admissible apex left contributes
        no triangles and is skipped silently, so a hole whose every triangulation is blocked comes
        back still open rather than raising — pass ``False`` to trade that for a possibly
        non-manifold fill.
    preserve_largest_hole
        When ``True``, leave the single largest boundary loop (greatest perimeter) open and fill
        only the rest — the single-boundary disk a robust UV parametrization expects.
    smooth_boundary
        When ``True`` (default), the dihedral edge metrics also score the hole
        rim edges against the existing adjacent faces, so the patch blends smoothly into the surface
        instead of turning sharply at the boundary. No effect on the triangle-only metrics.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_out_faces,)`` flat face buffer of the original faces followed by the fill
        triangles, on ``faces.device``. A watertight or empty mesh — or, with
        ``preserve_largest_hole``, a mesh whose only hole is the largest — is returned unchanged (a
        copy).

    Raises
    ------
    ValueError
        If ``metric`` is not one of the supported metric names.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`fill_fan`][ordito.holes.fill_fan]
    [`fill_cone`][ordito.holes.fill_cone]
    [`boundary_loops`][ordito.boundary.boundary_loops]
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]

    Notes
    -----
    Fill winding is consistent with the adjacent faces only when the input mesh is consistently
    wound (see [`make_winding_consistent`][ordito.repair.make_winding_consistent]).
    """
    require_same_device(vertices=vertices, faces=faces)
    _check_fill_metric(metric)
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.clone(faces)
    loops = _hole_loops(vertices, faces, preserve_largest_hole)
    if loops is None:
        return _launch.clone(faces)
    return _fill_packed_loops(
        vertices, faces, loops, metric, resolve_multiple_edges, smooth_boundary
    )


def fill_loops_min_weight(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: list[wp.array[wp.int32]],
    metric: str,
    resolve_multiple_edges: bool,
    smooth_boundary: bool = True,
) -> wp.array[wp.int32]:
    """
    Min-weight-triangulate the given boundary ``loops`` and append the fill faces.

    The loop-selecting form of [`fill_min_weight`][ordito.holes.fill_min_weight]: same engine, but
    the caller names which boundary loops to close rather than getting every one of them. Both
    wrappers hand the packed loops to one shared private engine, so the triangulation is identical
    where the loop sets are. Every loop is sealed **together** by the interval DP under ``metric``
    (with a ``min_area`` fallback where the primary metric yields a bad triangulation), reusing
    only existing vertices. Cost is set by the longest loop, not by the loop count; see
    [`fill_min_weight`][ordito.holes.fill_min_weight].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    loops
        Boundary loops to fill, as ordered vertex-index arrays (``>= 3`` vertices each), e.g. from
        [`boundary_loops`][ordito.boundary.boundary_loops].
    metric
        Fill metric name; see [`fill_min_weight`][ordito.holes.fill_min_weight].
    resolve_multiple_edges
        When ``True``, forbid chords that duplicate an existing mesh edge.
    smooth_boundary
        See [`fill_min_weight`][ordito.holes.fill_min_weight].

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_out_faces,)`` flat face buffer of ``faces`` followed by the fill triangles for
        ``loops``, on ``faces.device``. Unchanged (a copy) when ``loops`` is empty.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the names
        [`fill_min_weight`][ordito.holes.fill_min_weight] lists.
    RuntimeError
        If ``vertices``, ``faces`` and ``loops`` are not all on one device.

    See Also
    --------
    [`fill_min_weight`][ordito.holes.fill_min_weight]
    """
    require_same_device(vertices=vertices, faces=faces, loops=loops)
    _check_fill_metric(metric)
    if len(loops) == 0:
        return _launch.clone(faces)
    return _fill_packed_loops(
        vertices, faces, _pack_loops(loops), metric, resolve_multiple_edges, smooth_boundary
    )


def _fill_packed_loops(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: _PackedLoops,
    metric: str,
    resolve_multiple_edges: bool,
    smooth_boundary: bool,
) -> wp.array[wp.int32]:
    """
    Min-weight-triangulate every packed loop **together** and append the fill faces.

    The engine behind [`fill_loops_min_weight`][ordito.holes.fill_loops_min_weight]. Everything
    before the traceback is batched across loops — one Newell-normal and longest-edge pass, one
    chord pass over the mesh, one ragged ``dp`` / ``prev`` pair, one launch per span rather than
    per (loop, span), a device-side min-area retry mask instead of a host branch per loop, and a
    one-thread-per-loop traceback that keeps the predecessor table on the device. What is left on
    the host is two scalar reads of one small state buffer: whether any loop needs the fallback
    metric, and whether the traceback fell short of a full triangulation anywhere.
    """
    device = faces.device
    edge_table = _EdgeTable(vertices, faces)
    primary_id = _METRIC_IDS[metric]
    combine_id = _METRIC_COMBINE.get(metric, 0)
    min_area_id = _METRIC_IDS["min_area"]

    loop_pos = _launch.empty(loops.total, dtype=wp.vec3, device=device)
    rim_keys = _launch.empty(loops.total, dtype=wp.uint64, device=device)
    # Accumulated into, then finished in place: the finalize pass reads each loop's sum and writes
    # that loop's slot, so the plane normals overwrite the Newell sums and the ``char_area`` scales
    # overwrite the longest edges rather than taking two buffers of their own.
    plane_normals = _launch.zeros(loops.n_loops, dtype=wp.vec3, device=device)
    char_areas = _launch.zeros(loops.n_loops, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_holes.loop_rim_metrics,
        dim=loops.total,
        inputs=[
            loops.flat_loops,
            loops.loop_id,
            loops.offsets,
            vertices,
            edge_table.base,
            char_areas,
            plane_normals,
            loop_pos,
            rim_keys,
        ],
        device=device,
    )
    _launch.launch(
        kernel_holes.finalize_rim_metrics,
        dim=loops.n_loops,
        inputs=[char_areas, plane_normals, plane_normals, char_areas],
        device=device,
    )

    forbidden = (
        edge_table.forbidden_chords(loops)
        if resolve_multiple_edges
        else _launch.zeros(loops.dp_total, dtype=wp.int32, device=device)
    )
    rim_opp_pos, rim_opp_valid = edge_table.rim_opposite(loops, rim_keys)

    dp = _launch.empty(loops.dp_total, dtype=wp.float32, device=device)
    prev = _launch.empty(loops.dp_total, dtype=wp.int32, device=device)
    # The retry flag and the traceback's shortfall, read back at two different points; see
    # ``kernels/holes.py::FILL_STATE_SLOTS``.
    state = _launch.zeros(kernel_holes.FILL_STATE_SLOTS, dtype=wp.int32, device=device)
    active = _launch.ones(loops.n_loops, dtype=wp.int32, device=device)
    _run_hole_dp(
        loops,
        loop_pos,
        plane_normals,
        forbidden,
        rim_opp_pos,
        rim_opp_valid,
        char_areas,
        active,
        primary_id,
        combine_id,
        smooth_boundary,
        dp,
        prev,
    )

    if primary_id != min_area_id:
        # *Which* loops the primary metric failed on is decided on device and fed straight back in
        # as the re-run's active mask, so the fallback is one more batched pass rather than a branch
        # per loop. Loops the primary metric handled keep their ``prev`` rows. The mask overwrites
        # the all-loops one in place: the first sweep has read it by the time the flag pass runs,
        # and every slot is rewritten.
        _launch.launch(
            kernel_holes.flag_bad_triangulations,
            dim=loops.n_loops,
            inputs=[loops.offsets, loops.dp_offsets, dp, active, state],
            device=device,
        )
        # ...but *whether* any loop failed is worth one host read, because a pass with an all-zero
        # mask is still a full ``max(B) - 1`` launch sweep in which every thread returns
        # immediately -- that is pure marshalling and buys nothing on its own. ``prev`` is read
        # back a few lines below regardless, so this adds no synchronisation point that was not
        # there. The flag kernel raises the test itself as it writes the mask, so it is one
        # 4-byte read rather than a readback of the ``n_loops`` buffer or a reduction over it.
        if read_scalar(state, kernel_holes.FILL_STATE_RETRY) > 0:
            _run_hole_dp(
                loops,
                loop_pos,
                plane_normals,
                forbidden,
                rim_opp_pos,
                rim_opp_valid,
                char_areas,
                active,
                min_area_id,
                0,
                smooth_boundary,
                dp,
                prev,
            )

    return _traceback_fill_faces(faces, loops, prev, state)


def _traceback_fill_faces(
    faces: wp.array[wp.int32],
    loops: _PackedLoops,
    prev: wp.array[wp.int32],
    state: wp.array[wp.int32],
) -> wp.array[wp.int32]:
    """
    Walk every loop's DP predecessor table into fill faces, returned appended to ``faces``.

    One thread per loop, so the ``O(B)`` walk each rim needs runs where its table already is
    instead of crossing the bus: the host form read the whole ``sum(B^2)`` predecessor table back
    to reach ``sum(B)`` of its entries, then built the triangles a Python tuple at a time.

    Each loop writes into its own ``B - 2`` slots of a padded block and reports how many it
    actually used, which is fewer exactly where an interval had no legal apex. The padded block is
    the tail of the returned buffer itself, with ``faces`` copied in front of it, so when no loop
    falls short -- the usual case -- the walk has written the answer in place. The single readback
    is the total shortfall the walk accumulated into ``state``; only where it is non-zero does a
    scan place the compacted blocks and a second buffer receive them.
    """
    device = loops.device
    triangle_sizes_np = np.maximum(loops.sizes_np - 2, 0)
    n_padded = int(triangle_sizes_np.sum())
    if n_padded == 0:
        return _launch.clone(faces)
    n_face_indices = faces.size
    filled = _launch.empty(n_face_indices + 3 * n_padded, dtype=wp.int32, device=device)
    _launch.copy(filled, faces, count=n_face_indices)
    padded = odt.as_dense(filled[n_face_indices:]).reshape((n_padded, 3))
    triangle_offsets = loops.triangle_offsets
    counts = odt.empty_1d(loops.n_loops, wp.int32, device=device)
    # The walk's pending intervals: one slot per packed rim vertex, which is exactly enough
    # (see ``traceback_fill_triangles``), and caller-allocated because a kernel local cannot be
    # sized by a runtime rim length.
    stack = _launch.empty(loops.total, dtype=wp.vec2i, device=device)
    _launch.launch(
        kernel_holes.traceback_fill_triangles,
        dim=loops.n_loops,
        inputs=[
            loops.flat_loops,
            loops.offsets,
            loops.dp_offsets,
            triangle_offsets,
            prev,
            stack,
            counts,
            padded,
            state,
        ],
        device=device,
    )
    shortfall = int(read_scalar(state, kernel_holes.FILL_STATE_SHORTFALL))
    if shortfall == 0:
        return filled
    n_triangles = n_padded - shortfall
    if n_triangles == 0:
        return _launch.clone(faces)
    packed_ends = odt.empty_1d(loops.n_loops, wp.int32, device=device)
    _launch.array_scan(counts, packed_ends, inclusive=True)
    compacted = _launch.empty(n_face_indices + 3 * n_triangles, dtype=wp.int32, device=device)
    _launch.copy(compacted, faces, count=n_face_indices)
    _launch.launch(
        kernel_holes.compact_fill_triangles,
        dim=(loops.n_loops, int(triangle_sizes_np.max())),
        inputs=[
            counts,
            triangle_offsets,
            packed_ends,
            padded,
            odt.as_dense(compacted[n_face_indices:]).reshape((n_triangles, 3)),
        ],
        device=device,
    )
    return compacted


def _run_hole_dp(
    loops: _PackedLoops,
    loop_pos: wp.array[wp.vec3],
    plane_normals: wp.array[wp.vec3],
    forbidden: wp.array[wp.int32],
    rim_opp_pos: wp.array[wp.vec3],
    rim_opp_valid: wp.array[wp.int32],
    char_areas: wp.array[wp.float32],
    active: wp.array[wp.int32],
    metric_id: int,
    combine_id: int,
    smooth_boundary: bool,
    dp: wp.array[wp.float32],
    prev: wp.array[wp.int32],
    tiled: bool | None = None,
    captured: bool | None = None,
) -> None:
    """
    Fill the ragged ``dp`` / ``prev`` tables for every loop flagged in ``active``, in place.

    One launch per triangulation span **across all loops**, so the sequence is ``max(B) - 2``
    long for the whole mesh rather than ``B - 1`` per hole. That length is the floor: the spans are
    a dependency chain and each one's parallelism is its ``(n_loops, max_B - span)`` grid, which is
    what fills the device, so merging levels into one kernel trades exactly that away. Folding the
    spans into one persistent block per loop was tried and loses for that reason, and a *blocked*
    interval DP loses for it too -- tiles on one tile-diagonal are independent, but there are only
    ``B / tile`` of them, against the ``B - span`` blocks a plain span level already has. Beating
    either needs a grid-wide barrier, which Warp does not expose.

    **What the chain does not have to be is a chain of *issued* launches**, and that is what
    ``captured`` is. Every span launch differs from the next in one integer, so putting that
    integer on the device (``HoleFillTables.span_base``) makes them identical, and one group of
    ``HOLE_DP_GRAPH_SPANS`` is then recorded once and replayed to cover the sweep. Replaying costs
    a fraction of issuing, so the host stops being the critical path and the sweep falls back to
    the device time the kernels were already taking underneath it. The recorded group's grid is
    sized for the *first* group, which over-covers every later one -- correct, because a thread
    whose span has outrun its loop returns at the first guard, and nearly free, because the span
    kernel's cost is flat in the grid's unused width. The same guard absorbs the final group's
    overshoot past the last span, so no remainder group is needed.

    ``None`` asks [`hole_dp_captures`][ordito.kernels.holes.hole_dp_captures], which wants the
    sweep long enough that recording it is cheap beside it *and* light enough per span that the
    host was the critical path at all -- the second bound is why this is not simply a rim-length
    test. The CPU device has no graph to record and always takes the plain loop. ``captured``
    exists so a test can force either and compare them, and the two must agree byte for byte.

    ``dp`` / ``prev`` are filled in place. They are bound into ``HoleFillTables`` rather than
    passed per launch, because their pointers are invariant across the sweep -- see that struct's
    docstring.

    ``tiled`` selects the per-span engine: a block per interval with its lanes striding the apex
    loop, or one thread per interval. Both produce byte-identical ``dp`` / ``prev`` on **both**
    devices, so this is a pure cost knob and ``None`` picks whichever is faster on the device at
    hand -- tiled on CUDA, where a block has lanes to spread the apex loop over; serial on CPU,
    where it has one and the two engines do the same work. ``tiled`` exists so a test can force
    either and compare them.

    That portability rests on ``fill_dp_span_tiled`` striding by ``wp.block_dim()`` rather than by
    the ``HOLE_DP_BLOCK`` it is launched with: the two agree on CUDA, and on CPU the former reads 1,
    so the single lane covers every apex and the tile reductions after it degenerate to one-element
    tiles holding that lane's own answer. With the constant it would silently minimize over every
    32nd apex instead.
    """
    device = loops.device
    resolved = wp.get_device(device)
    if tiled is None:
        tiled = not resolved.is_cpu
    n_spans = loops.max_size - 2
    if captured is None:
        # Only CUDA has a graph to record; the rest of the decision is the kernel module's.
        captured = resolved.is_cuda and kernel_holes.hole_dp_captures(loops.max_size, loops.n_loops)
    # One of two lane counts, picked from the rim count *and* the longest rim -- see
    # ``kernels/holes.hole_dp_block``, whose table shows why both matter. Read once here rather
    # than per span so the whole sweep shares one module hash.
    block = kernel_holes.hole_dp_block(loops.max_size, loops.n_loops)
    _launch.launch(
        kernel_holes.init_dp_base,
        dim=(loops.n_loops, loops.max_size),
        inputs=[loops.offsets, loops.dp_offsets, active, dp, prev],
        device=device,
    )
    if n_spans <= 0:
        # Every rim is a triangle or smaller, so the base table is already the whole answer.
        return
    # Built once, outside the loop: every field is invariant across spans, and a wp.launch argument
    # costs host time whatever it holds. Rebuilding it per span would give the saving straight back.
    tables = kernel_holes.HoleFillTables()
    tables.loop_pos = loop_pos
    tables.loop_offsets = loops.offsets
    tables.dp_offsets = loops.dp_offsets
    tables.active = active
    tables.plane_normals = plane_normals
    tables.forbidden = forbidden
    tables.rim_opp_pos = rim_opp_pos
    tables.rim_opp_valid = rim_opp_valid
    tables.char_areas = char_areas
    # The two in-place DP tables live in the bundle too: their pointers are invariant across the
    # whole span sweep, and at about a microsecond per launch argument a long rim's hundreds of
    # launches pay milliseconds to keep them in the signature. See ``HoleFillTables``.
    tables.dp = dp
    tables.prev = prev
    # The first span of the group a launch belongs to; the launch argument is the offset within it.
    # Allocated holding its value rather than filled afterwards, and the plain loop below is simply
    # one group covering every span, so both paths read the same convention.
    tables.span_base = _launch.full(1, 2, dtype=wp.int32, device=device)
    tables.metric_id = wp.int32(metric_id)
    tables.combine_id = wp.int32(combine_id)
    tables.smooth_bd = wp.int32(1 if smooth_boundary else 0)

    def issue(offset: int, width: int) -> None:
        inputs = [tables, wp.int32(offset)]
        dim = (loops.n_loops, width)
        if tiled:
            _launch.launch_tiled(
                kernel_holes.fill_dp_span_tiled,
                dim=dim,
                inputs=inputs,
                block_dim=block,
                device=device,
            )
        else:
            _launch.launch(kernel_holes.fill_dp_span, dim=dim, inputs=inputs, device=device)

    if not captured:
        for span in range(2, loops.max_size):
            issue(span - 2, loops.max_size - span)
        return
    group = kernel_holes.HOLE_DP_GRAPH_SPANS
    with wp.ScopedCapture(resolved) as capture:
        for offset in range(group):
            # Sized for the first group, which is the widest any later one needs.
            issue(offset, max(n_spans - offset, 1))
        _launch.launch(
            kernel_holes.advance_span_base,
            dim=1,
            inputs=[wp.int32(group), tables.span_base],
            device=device,
        )
    graph = capture.graph
    assert graph is not None
    for _ in range((n_spans + group - 1) // group):
        wp.capture_launch(graph)


def _pack_loops(loops: list[wp.array[wp.int32]]) -> _PackedLoops:
    """Concatenate a caller's per-loop arrays into the packed form the fill engine consumes."""
    # ``copy=False``: every kernel below takes ``flat_loops`` as an input and none writes it.
    flat_loops, _offsets = od.array.pack_1d_arrays(loops, copy=False)
    sizes_np = np.asarray([loop.size for loop in loops], dtype=np.int64)
    return _PackedLoops(flat_loops, sizes_np)


def fill_small(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_perimeter: float | None = None,
    *,
    max_edges: int | None = None,
) -> wp.array[wp.int32]:
    """
    Fill only the *small* boundary loops, sized either by perimeter or by boundary-edge count.

    Intended open boundaries (large loops) are left untouched; spurious small holes are sealed by
    the shared min-weight interval DP
    ([`fill_loops_min_weight`][ordito.holes.fill_loops_min_weight]). Used by
    [`ordito.reconstruction.triangulate_point_cloud`][ordito.reconstruction.triangulate_point_cloud]
    to seal small gaps left by sparse or non-uniform point-cloud sampling.

    Exactly one of the two thresholds is given, and they are genuinely different questions rather
    than two spellings of one:

    - ``max_perimeter`` is a **length**, so it is scale-dependent and sampling-independent -- the
      right threshold when "small" means *small on the object*.
    - ``max_edges`` is a **count**, so it is scale-independent and sampling-dependent -- the right
      threshold when "small" means *few triangles to patch*, which is what bounds the ``O(B^3)``
      interval DP behind the fill.

    Neither converts into the other without knowing the rim's sampling, and the reference
    implementations split the same way: two of the three take a boundary-edge count and only one
    takes a length. On a mesh whose rims are sampled unevenly the two thresholds select **opposite**
    loops -- ``tests/test_holes.py`` builds exactly that case, a 16-vertex rim of perimeter 2.41
    beside a 5-vertex rim of perimeter 3.16.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    max_perimeter
        Only boundary loops with perimeter at most this value are filled. Positional for backward
        compatibility. Mutually exclusive with ``max_edges``.
    max_edges
        Only boundary loops with at most this many boundary edges are filled -- **inclusive**, so a
        24-edge rim is filled at ``max_edges=24`` and left open at 23. A closed loop has as many
        edges as vertices, so this is also its vertex count. Mutually exclusive with
        ``max_perimeter``.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_out_faces,)`` flat face buffer of ``faces`` followed by the fill triangles for the
        small loops, on ``faces.device``. A copy of ``faces`` when no boundary loop meets the
        threshold.

    Raises
    ------
    ValueError
        If neither ``max_perimeter`` nor ``max_edges`` is given, or if both are.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    ``max_edges`` costs nothing to evaluate: the loop sizes are already host-side metadata of the
    packed loops, so that branch skips the segmented perimeter launch and its readback entirely.

    The inclusive bound is stated because it is easy to get wrong from the outside -- one reference
    whose parameter this matches documents itself as "less than" and is measurably "at most".

    See Also
    --------
    [`fill_min_weight`][ordito.holes.fill_min_weight]
    [`fill_loops_min_weight`][ordito.holes.fill_loops_min_weight]
    [`fillable_loop_mask`][ordito.holes.fillable_loop_mask]
        Which loops the DP can fill at all, a separate question from which are small.
    """
    require_same_device(vertices=vertices, faces=faces)
    if (max_perimeter is None) == (max_edges is None):
        raise ValueError("pass exactly one of max_perimeter or max_edges")

    packed = _hole_loops(vertices, faces)
    if packed is None:
        return _launch.clone(faces)

    if max_edges is not None:
        small_np = packed.sizes_np <= max_edges
    else:
        # One segmented-sum launch measures every rim, so the selection costs a single readback
        # rather than a copy of the whole vertex buffer plus one of each loop.
        small_np = packed.perimeters(vertices) <= max_perimeter
    if not small_np.any():
        return _launch.clone(faces)
    if not small_np.all():
        packed = _compact_packed_loops(packed.flat_loops, packed.sizes_np, small_np)
    return _fill_packed_loops(vertices, faces, packed, "plane_normalized", True, True)


@overload
def fill_smooth(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    metric: str = "plane_normalized",
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    max_edge_splits: int = 1000,
    max_angle_change_after_flip: float = math.radians(30.0),
    smooth_curvature: bool = True,
    natural_smooth: bool = False,
    edge_weights: str = "cotan",
    preserve_largest_hole: bool = False,
    resolve_multiple_edges: bool = True,
    smooth_boundary: bool = True,
    refine: Literal["max_edge", "density"] = "max_edge",
    return_patch: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def fill_smooth(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    metric: str = "plane_normalized",
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    max_edge_splits: int = 1000,
    max_angle_change_after_flip: float = math.radians(30.0),
    smooth_curvature: bool = True,
    natural_smooth: bool = False,
    edge_weights: str = "cotan",
    preserve_largest_hole: bool = False,
    resolve_multiple_edges: bool = True,
    smooth_boundary: bool = True,
    refine: Literal["max_edge", "density"] = "max_edge",
    return_patch: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]: ...
def fill_smooth(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    metric: str = "plane_normalized",
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    max_edge_splits: int = 1000,
    max_angle_change_after_flip: float = math.radians(30.0),
    smooth_curvature: bool = True,
    natural_smooth: bool = False,
    edge_weights: str = "cotan",
    preserve_largest_hole: bool = False,
    resolve_multiple_edges: bool = True,
    smooth_boundary: bool = True,
    refine: Literal["max_edge", "density"] = "max_edge",
    return_patch: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]
):
    """
    Fill every boundary hole with a smooth, refined patch.

    Runs the full three-stage pipeline: a minimum-weight triangulation seals each hole over its
    existing rim vertices ([`fill_min_weight`][ordito.holes.fill_min_weight]),
    the patch is then subdivided to ``max_edge`` with Delaunay edge flips
    ([`subdivide_region_to_size`][ordito.remesh.subdivide_region_to_size]), and finally the new
    interior patch vertices are smoothed into the surrounding surface
    ([`smooth_region_fixed_rim`][ordito.smoothing.smooth_region_fixed_rim]
    then [`smooth_region`][ordito.smoothing.smooth_region]). Unlike the purely
    topological fillers, this produces a well-graded, curvature-continuous patch.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    metric
        Minimum-weight fill metric; see
        [`fill_min_weight`][ordito.holes.fill_min_weight].
    triangulate_only
        When ``True``, only fill (no subdivision or smoothing) — equivalent to
        [`fill_min_weight`][ordito.holes.fill_min_weight].
    max_edge
        Target maximum patch edge length. ``None`` (default) derives it from the mean rim edge
        length of the holes being filled; a sequential budget target has no parallel analogue.
    max_edge_splits
        Soft cap on the number of edge splits during subdivision.
    max_angle_change_after_flip
        Dihedral-angle-change gate for the Delaunay flip pass (default 30°).
    smooth_curvature
        When ``True`` (default), smooth the new patch vertices after subdivision.
    natural_smooth
        When ``True``, additionally grow a collar around the patch and smooth it so the patch
        blends into the surrounding surface.
    edge_weights
        Laplacian edge weights for the cross-boundary smooth solve: ``"cotan"`` (default) or
        ``"unit"``.
    preserve_largest_hole
        When ``True``, leave the single largest boundary loop open.
    resolve_multiple_edges
        When ``True`` (default), forbid fill chords that duplicate existing mesh edges.
    smooth_boundary
        When ``True`` (default), also run the cross-boundary smooth solve so the patch is C¹ across
        its rim. Also tunes the fill metric's rim edge terms.
    refine
        Which subdivision criterion the refinement stage uses. ``"max_edge"`` (default) bisects
        patch edges longer than ``max_edge``; ``"density"`` splits patch triangles at their centroid
        while their sampling is coarser than the surrounding mesh's, which is Liepa's criterion and
        what the reference hole fillers do. The two differ only on a **graded** neighbourhood, where
        a single target length cannot be right at both ends of the grading: ``"density"`` keeps the
        patch triangle size closer to the local surrounding scale there than a single ``max_edge``
        target can. ``max_edge`` and ``max_edge_splits`` are ignored under ``"density"``.
    return_patch
        When ``True``, also return a length-``n_out_faces`` ``wp.bool`` mask of the patch faces.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` original vertices followed by the inserted patch vertices, on
        ``vertices.device``.
    new_faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` original faces followed by the patch faces.
    patch_mask : wp.array[wp.bool]
        ``(n_out_faces,)`` mask of the patch faces in ``new_faces``; only when ``return_patch`` is
        ``True``.

    Raises
    ------
    ValueError
        If ``metric``, ``edge_weights`` or ``refine`` is unknown.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`fill_min_weight`][ordito.holes.fill_min_weight]
    [`stitch_smooth`][ordito.holes.stitch_smooth]
    [`subdivide_region_to_size`][ordito.remesh.subdivide_region_to_size]

    Notes
    -----
    The three stages are: minimum-weight fill, refine the patch to the target edge length, then
    smooth its new interior vertices into the surrounding surface.

    Winding is consistent with the surrounding faces only for a consistently wound input.
    """
    require_same_device(vertices=vertices, faces=faces)
    _check_fill_metric(metric)
    if edge_weights not in ("cotan", "unit"):
        raise ValueError(f"edge_weights must be 'cotan' or 'unit', got {edge_weights!r}")
    _check_refine(refine)

    device = faces.device
    n_faces = faces.size // 3
    packed = _hole_loops(vertices, faces, preserve_largest_hole) if n_faces > 0 else None
    if packed is None:
        empty = _patch_mask(faces.size // 3, faces.size // 3, device)
        result = (_launch.clone(vertices), _launch.clone(faces))
        return (*result, empty) if return_patch else result

    n_faces_before = n_faces
    n_vertices_before = vertices.size
    faces_filled = _fill_packed_loops(
        vertices, faces, packed, metric, resolve_multiple_edges, smooth_boundary
    )
    n_faces_after = faces_filled.size // 3
    patch_mask = _patch_mask(n_faces_before, n_faces_after, device)

    if triangulate_only:
        result = (_launch.clone(vertices), faces_filled)
        return (*result, patch_mask) if return_patch else result

    target_edge = max_edge if max_edge is not None else _mean_rim_edge_length(vertices, packed)
    new_vertices, new_faces, out_patch = od.smoothing.refine_and_smooth_region(
        vertices,
        faces_filled,
        n_vertices_before,
        patch_mask,
        target_edge,
        max_edge_splits,
        max_angle_change_after_flip,
        smooth_curvature,
        smooth_boundary,
        natural_smooth,
        edge_weights,
        refine=refine,
    )
    return (new_vertices, new_faces, out_patch) if return_patch else (new_vertices, new_faces)


@overload
def refill_region(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    metric: str = "plane_normalized",
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    smooth_curvature: bool = True,
    refine: Literal["max_edge", "density"] = "max_edge",
    return_patch: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def refill_region(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    metric: str = "plane_normalized",
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    smooth_curvature: bool = True,
    refine: Literal["max_edge", "density"] = "max_edge",
    return_patch: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]: ...
def refill_region(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    metric: str = "plane_normalized",
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    smooth_curvature: bool = True,
    refine: Literal["max_edge", "density"] = "max_edge",
    return_patch: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]
):
    """
    Replace a face region with a fresh patch: delete it, then fill the hole nicely.

    [`fill_smooth`][ordito.holes.fill_smooth] applied to a region rather than to the mesh's own
    holes -- the operation for *rebuilding* a piece of surface that is wrong rather than missing: a
    self-intersecting band, a noisy patch, a set of faces a user painted. The region is removed, the
    rims that opens are triangulated by the same minimum-weight DP, and the patch is refined and
    smoothed to match its surroundings.

    Only the rims the deletion **opened** are filled. An input that already had a boundary keeps it,
    which is what makes this usable on an open mesh at all --
    [`ordito.selection.delete_region_keep_boundary`][ordito.selection.delete_region_keep_boundary]
    is where that distinction is drawn.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    face_mask
        ``(n_faces,)`` mask, ``True`` for each face to **replace**.
    metric
        Minimum-weight fill metric, as in [`fill_min_weight`][ordito.holes.fill_min_weight].
    triangulate_only
        Stop after the minimum-weight triangulation, skipping the refinement and smoothing. The
        patch is then ``n - 2`` triangles per rim of ``n`` vertices and adds no new vertices.
    max_edge
        Target edge length for the refinement. ``None`` uses the mean rim edge length, so the
        patch arrives at roughly the surrounding mesh's resolution.
    smooth_curvature
        Minimize curvature rather than area when smoothing the patch, as in
        [`fill_smooth`][ordito.holes.fill_smooth].
    refine
        Subdivision criterion for the refinement stage; see
        [`fill_smooth`][ordito.holes.fill_smooth].
    return_patch
        Also return the per-face mask of the new patch, for a caller that wants to keep working on
        it.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]] | tuple[..., wp.array[wp.bool]]
        ``(n_out_vertices,)`` vertices and ``(3 * n_out_faces,)`` faces of the rebuilt mesh, plus
        the ``(n_out_faces,)`` patch mask when ``return_patch`` is ``True``. When the mask selects
        nothing, or selects faces whose removal opens no new rim, the surviving mesh is returned
        with an all-``False`` patch.

    Raises
    ------
    ValueError
        If ``metric`` or ``refine`` is not one of the supported names, or ``face_mask`` does not
        have one entry per face.
    RuntimeError
        If ``vertices``, ``faces`` and ``face_mask`` are not all on one device.

    Examples
    --------
    ```python
    bad = od.validation.face_self_intersecting_mask(v, f)
    patched_v, patched_f = od.holes.refill_region(v, f, bad)
    ```

    See Also
    --------
    [`fill_smooth`][ordito.holes.fill_smooth]
        The same pipeline over the mesh's existing holes, when nothing needs deleting first.
    [`ordito.selection.delete_region_keep_boundary`][ordito.selection.delete_region_keep_boundary]
        The first half, when the rims are wanted rather than filled.
    [`ordito.repair.remove_folded_faces`][ordito.repair.remove_folded_faces]
        One of several ways to produce the mask this takes.
    """
    require_same_device(vertices=vertices, faces=faces, face_mask=face_mask)
    _check_fill_metric(metric)
    _check_refine(refine)

    kept_vertices, kept_faces, new_loops = od.selection.delete_region_keep_boundary(
        vertices, faces, face_mask
    )
    device = faces.device
    n_kept_faces = kept_faces.size // 3
    if not new_loops:
        empty = _patch_mask(n_kept_faces, n_kept_faces, device)
        result = (kept_vertices, kept_faces)
        return (*result, empty) if return_patch else result

    packed = _pack_loops(new_loops)
    faces_filled = _fill_packed_loops(kept_vertices, kept_faces, packed, metric, True, True)
    patch_mask = _patch_mask(n_kept_faces, faces_filled.size // 3, device)
    if triangulate_only:
        result = (kept_vertices, faces_filled)
        return (*result, patch_mask) if return_patch else result

    target_edge = max_edge if max_edge is not None else _mean_rim_edge_length(kept_vertices, packed)
    new_vertices, new_faces, out_patch = od.smoothing.refine_and_smooth_region(
        kept_vertices,
        faces_filled,
        kept_vertices.size,
        patch_mask,
        target_edge,
        1000,
        math.radians(30.0),
        smooth_curvature,
        True,
        False,
        "cotan",
        refine=refine,
    )
    return (new_vertices, new_faces, out_patch) if return_patch else (new_vertices, new_faces)


def extend_hole(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    loops: Sequence[wp.array[wp.int32]] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extend every boundary rim out to a plane, adding the ruled surface between them.

    Each rim vertex is projected orthogonally onto the plane, and the rim is bridged to that ring of
    projections with two triangles per rim edge. The result is open again -- the new rim lies
    *in* the plane -- which is what makes this the step before a flat cap rather than a cap itself:
    the extended rim is planar, so
    [`fill_min_weight`][ordito.holes.fill_min_weight] closes it with a triangulation that has no
    reason to fold.

    The typical use is turning a scanned shell into a solid: extend its rims to a base plane, then
    fill. Doing that in one step with a min-weight fill instead gives a curved cap over the original
    rim, which is a different shape and usually not the wanted one.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    plane_normal
        The plane's normal. Need not be unit length in principle, but **is assumed to be** -- the
        projection scales with it otherwise. Normalize it.
    plane_origin
        A point on the target plane.
    loops
        Rims to extend. ``None`` extends every one, via
        [`boundary_loops`][ordito.boundary.boundary_loops]; pass a subset to extend only those.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` input positions, unchanged and in order, with one projected vertex
        appended per rim vertex.
    faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` input faces with the bridge triangles appended -- two per rim edge.

    Raises
    ------
    TypeError
        If any loop is not a rank-1 ``wp.int32`` array.
    RuntimeError
        If ``vertices``, ``faces`` and ``loops`` are not all on one device.

    See Also
    --------
    [`fill_min_weight`][ordito.holes.fill_min_weight]
        Closes the planar rim this leaves.
    [`build_bottom`][ordito.holes.build_bottom]
        The same extension with the plane fitted to each rim, instead of placed by the caller.
    [`stitch_loops`][ordito.holes.stitch_loops]
        Bridges two rims that both already exist, where this generates the second one.
    [`boundary_loops`][ordito.boundary.boundary_loops]

    Notes
    -----
    !!! note "The rim may cross the plane"
        Nothing here checks which side of the plane a rim vertex is on. A rim straddling it gets an
        extension that folds through the plane, which is geometrically what "project each vertex"
        means and is almost never wanted -- place the plane clear of the rim, and check with
        [`bounds.aabb`][ordito.bounds.aabb] if the input is not yours.
    """
    require_same_device(vertices=vertices, faces=faces, loops=loops)
    packed = _packed_rims(vertices, faces, loops)
    if packed is None:
        return _launch.clone(vertices), _launch.clone(faces)
    origins = _launch.full(packed.n_loops, plane_origin, dtype=wp.vec3, device=faces.device)
    return _extend_packed_rims(vertices, faces, packed, plane_normal, origins=origins)


def build_bottom(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    direction: wp.vec3,
    hole_extension: float = 0.0,
    loops: Sequence[wp.array[wp.int32]] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extend every boundary rim down to a flat base placed under its own lowest vertex.

    This is [`extend_hole`][ordito.holes.extend_hole] with the plane chosen for you: for each rim
    separately, the plane with normal ``direction`` through that rim's most extreme vertex in the
    ``-direction`` sense, pushed a further ``hole_extension`` past it. So the base sits flush with
    the lowest point of the rim and nothing folds -- which is what makes this the safe form when the
    rim is not level and a hand-placed plane would cut through it.

    Each rim gets its **own** plane. A mesh with two rims at different heights therefore gets two
    bases, not one shared one; pass a single-element ``loops`` to bottom just one of them.

    The result is still open -- the base is a rim in the plane, not a cap. Follow with
    [`fill_min_weight`][ordito.holes.fill_min_weight] to close it.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    direction
        The "up" direction the base is placed against: the plane's normal, and the rim is extended
        towards ``-direction``. Assumed unit length -- ``hole_extension`` is measured in its units
        and the projection scales with it otherwise.
    hole_extension
        Extra distance past the extreme vertex, along ``-direction``. ``0.0`` puts the base exactly
        through the lowest rim vertex, which leaves that vertex unmoved.
    loops
        Rims to extend. ``None`` extends every one, via
        [`boundary_loops`][ordito.boundary.boundary_loops].

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` input positions, unchanged and in order, with one projected vertex
        appended per rim vertex.
    faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` input faces with the bridge triangles appended -- two per rim edge.

    Raises
    ------
    TypeError
        If any loop is not a rank-1 ``wp.int32`` array.
    RuntimeError
        If ``vertices``, ``faces`` and ``loops`` are not all on one device.

    See Also
    --------
    [`extend_hole`][ordito.holes.extend_hole]
        The general form, where the plane is yours to place.
    [`fill_min_weight`][ordito.holes.fill_min_weight]
        Closes the flat rim this leaves.
    """
    require_same_device(vertices=vertices, faces=faces, loops=loops)
    device = faces.device
    packed = _packed_rims(vertices, faces, loops)
    if packed is None:
        return _launch.clone(vertices), _launch.clone(faces)

    extremes = _launch.full(packed.n_loops, math.inf, dtype=wp.float32, device=device)
    _launch.launch(
        kernel_holes.loop_extreme_projection,
        dim=packed.total,
        inputs=[vertices, packed.flat_loops, packed.loop_id, direction, extremes],
        device=device,
    )
    # Each rim's plane origin is formed from its extreme as the extension reads it.
    return _extend_packed_rims(
        vertices, faces, packed, direction, extremes=extremes, extension=hole_extension
    )


def _packed_rims(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: Sequence[wp.array[wp.int32]] | None,
) -> _PackedLoops | None:
    """
    Pack the rims to extend, or ``None`` when there is nothing to extend.

    The packed form is ``_PackedLoops`` -- the same one the fill engine consumes -- rather than a
    second bookkeeping type of its own. The extension kernels read exactly its ``flat_loops`` /
    ``loop_id`` / ``starts`` / ``sizes``, and the two were building all four the same way. Its DP
    fields go unused here, which costs one host cumsum and one ``n_loops``-long upload.

    A rim set that is present but entirely empty is "nothing to extend" and answers ``None``, which
    is why this does not simply return
    [`_packed_loop_argument`][ordito.holes._packed_loop_argument]'s result: that helper keeps an
    empty loop, since a per-loop answer still owes it an entry.
    """
    rims = _packed_loop_argument(vertices, faces, loops)
    return None if rims is None or rims.total == 0 else rims


def _extend_packed_rims(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    rims: _PackedLoops,
    plane_normal: wp.vec3,
    *,
    origins: wp.array[wp.vec3] | None = None,
    extremes: wp.array[wp.float32] | None = None,
    extension: float = 0.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Project every packed rim vertex onto its loop's plane and bridge the two rings.

    Each loop's plane passes through ``origins[loop]``, or -- given ``extremes`` instead -- through
    ``plane_normal * (extremes[loop] - extension)``.
    """
    device = faces.device
    total = rims.total
    n_vertices = vertices.size
    n_faces = faces.size // 3
    extended_vertices = _launch.empty(n_vertices + total, dtype=wp.vec3, device=device)
    extended_faces = _launch.empty(3 * (n_faces + 2 * total), dtype=wp.int32, device=device)
    _launch.copy(extended_vertices[:n_vertices], vertices)
    _launch.copy(extended_faces[: 3 * n_faces], faces)
    _launch.launch(
        kernel_holes.extend_rim_to_ring,
        dim=total,
        inputs=[
            vertices,
            rims.flat_loops,
            rims.loop_id,
            rims.offsets,
            plane_normal,
            origins,
            extremes,
            wp.float32(extension),
            wp.int32(n_vertices),
            extended_vertices[n_vertices:],
            odt.as_dense(extended_faces[3 * n_faces :]).reshape((2 * total, 3)),
        ],
        device=device,
    )
    return extended_vertices, extended_faces


def fillable_loop_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: Sequence[wp.array[wp.int32]] | None = None,
) -> wp.array[wp.bool]:
    """
    Which boundary loops a min-weight fill can close without producing an invalid mesh.

    [`fill_min_weight`][ordito.holes.fill_min_weight] triangulates a rim over its **own** vertices,
    which fails in two combinatorial ways that have nothing to do with the metric. This names them
    per loop, so the policy is the caller's rather than a flag's:

    * **A repeated vertex.** A rim that visits one vertex twice is pinched there, and any
      triangulation
      of it folds through that pinch. A vertex shared by two *different* rims is the same pinch
      spread across two loops, and disqualifies both.
    * **A chord.** Two loop vertices that are *not* neighbours along the rim but are already joined
      by
      a mesh edge. The fill may propose that pair as a fill edge, and the mesh already has one, so
      that edge ends up with three faces. This is what
      ``fill_min_weight(resolve_multiple_edges=True)`` repairs *after* the fact, and a ``True`` here
      is exactly the case where that repair has nothing to do.

    !!! note "Conservative on the chord"
        The chord test is **sufficient, not necessary**: a chord-free loop cannot produce a
        duplicated edge whatever the dynamic program chooses, but a loop *with* a chord may still
        fill cleanly if the program happens to avoid it. A four-vertex rim with one diagonal already
        present is the smallest example -- there are two triangulations and only one collides. So
        read ``False`` as "check the result", not as "cannot be filled".

        Whether a loop is the "outer" one is deliberately absent: which boundary of an open surface
        is outer is a property of an embedding, not of the mesh. The practical form of that
        question, *which rim is the big one*, is answered by
        ``fill_min_weight(preserve_largest_hole=True)`` and by
        [`loop_perimeters`][ordito.boundary.loop_perimeters].

    !!! warning "A known gap: a seam *and* a pinch at the same vertex"
        The repeated-vertex check catches a pinch because
        [`boundary_loops`][ordito.boundary.boundary_loops]'s underlying walk represents it as a
        loop that visits one vertex twice. But when that vertex also sits on a non-orientable seam,
        the walk has no 2-regular path to fall back on and instead silently *drops* one of the
        vertex's two outgoing boundary edges -- so the loop this function sees is simply **shorter**
        than the true rim, with no repeated vertex to flag. Such a loop can pass this mask as
        fillable while not actually closing along real mesh edges. Closing the gap needs
        [`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets] itself to
        detect and report the dropped edge.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    loops
        The boundary loops to test. When ``None`` they are computed with
        [`boundary_loops`][ordito.boundary.boundary_loops]; pass them when you already have them,
        since the returned mask is indexed by their order and a caller almost always needs both. A
        loop naming a vertex outside ``vertices`` answers ``False``: it cannot be triangulated over
        mesh vertices, and the two vertex-indexed tables below have no entry for it.

    Returns
    -------
    wp.array[wp.bool]
        ``(n_loops,)`` one entry per loop, ``True`` where the loop is simple, shares no vertex with
        another loop, and is chord-free, on ``faces.device``. ``True`` is a guarantee; ``False`` is
        a warning, per the note above.

    Raises
    ------
    TypeError
        If any loop is not a rank-1 ``wp.int32`` array.
    RuntimeError
        If ``vertices``, ``faces`` and ``loops`` are not all on one device.

    See Also
    --------
    [`fill_min_weight`][ordito.holes.fill_min_weight]
        The fill this predicts the safety of.
    [`boundary_loops`][ordito.boundary.boundary_loops]
        Produces the loops, in the order this mask indexes.
    [`loop_perimeters`][ordito.boundary.loop_perimeters]
        Ranks the same loops by size, which is the other question a caller asks about a rim.
    """
    require_same_device(vertices=vertices, faces=faces, loops=loops)
    device = faces.device
    packed = _packed_loop_argument(vertices, faces, loops)
    if packed is None:
        return _launch.empty(0, dtype=wp.bool, device=device)

    # ``vertices`` is the domain, so its length is the bound ``index_bound`` would go to the
    # device to re-derive. An unreferenced vertex only widens the two tables below, which are
    # indexed by vertex id.
    n_vertices = vertices.size
    fillable = _launch.full(packed.n_loops, value=True, dtype=wp.bool, device=device)
    if packed.total > 0:
        # A vertex on two different rims is a pinch *between* loops, and a vertex a single rim
        # visits twice is a pinch *within* one: filling either leaves that vertex non-manifold,
        # and a clean loop sharing a pinched vertex is exactly as unfillable as the pinched one.
        # Both read off the same per-vertex slot count, and the pass that applies it also fills the
        # vertex-indexed tables the chord test reads, with every unpinched vertex.
        counts = _launch.zeros(n_vertices, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_holes.count_loop_vertices,
            dim=packed.total,
            inputs=[packed.flat_loops, packed.loop_id, wp.int32(n_vertices), counts, fillable],
            device=device,
        )
        # Each vertex's ``(loop, position along it)``, uninitialized: an entry is written exactly
        # where the count is 1, and the chord test reads the count first.
        loop_slots = _launch.empty(n_vertices, dtype=wp.vec2i, device=device)
        _launch.launch(
            kernel_holes.scatter_fillable_loop_slots,
            dim=packed.total,
            inputs=[packed.flat_loops, packed.loop_id, packed.offsets, counts],
            outputs=[fillable, loop_slots],
            device=device,
        )
        # Over the faces' corners, not the unique edges: the test is order-free and idempotent, so
        # the sort that deduplicates the edges buys nothing (``clear_loops_with_chords``).
        _launch.launch(
            kernel_holes.clear_loops_with_chords,
            dim=faces.size // 3,
            inputs=[faces, counts, loop_slots, packed.offsets, fillable],
            device=device,
        )
    return fillable


def stitch(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Stitch two single-boundary open meshes into one watertight mesh.

    Extracts the single boundary loop of each mesh
    ([`boundary_loops`][ordito.boundary.boundary_loops]) and joins them with
    [`stitch_loops`][ordito.holes.stitch_loops]. Each mesh must have
    exactly one boundary loop -- a restriction of this entry point, not of the seam construction;
    use ``stitch_loops`` directly to join specific loops of multi-boundary meshes. **No
    smoothing** is applied.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        ``(3 * n_faces,)`` flat triangle index buffers of each mesh.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` concatenated vertices (larger-boundary mesh first), on
        ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` concatenated, reindexed faces followed by the bridge triangles, on
        ``faces_a.device``.

    Raises
    ------
    ValueError
        If either mesh does not have exactly one boundary loop of at least 3 vertices.
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b`` and ``faces_b`` are not all on one device.

    See Also
    --------
    [`stitch_loops`][ordito.holes.stitch_loops]
    [`stitch_min_weight`][ordito.holes.stitch_min_weight]
    [`boundary_loops`][ordito.boundary.boundary_loops]
    """
    require_same_device(
        vertices_a=vertices_a, faces_a=faces_a, vertices_b=vertices_b, faces_b=faces_b
    )
    loop_a, loop_b = _single_boundary_loops(
        vertices_a, faces_a, vertices_b, faces_b, caller="stitch"
    )
    return stitch_loops(vertices_a, faces_a, loop_a, vertices_b, faces_b, loop_b)


def stitch_min_weight(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Stitch two single-boundary open meshes with a minimum-weight band.

    Like [`stitch`][ordito.holes.stitch] but joins the two rims with the metric-minimizing band
    of [`stitch_loops_min_weight`][ordito.holes.stitch_loops_min_weight]
    instead of the greedy correspondence. Each mesh must have exactly one boundary loop.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        ``(3 * n_faces,)`` flat triangle index buffers of each mesh.
    metric
        Stitch metric; see
        [`stitch_loops_min_weight`][ordito.holes.stitch_loops_min_weight].
    up_dir
        Up direction for the ``"vertical"`` metric.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` concatenated vertices, on ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` concatenated, reindexed faces followed by the band triangles.

    Raises
    ------
    ValueError
        If either mesh does not have exactly one boundary loop of at least 3 vertices, or ``metric``
        is unknown.
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b`` and ``faces_b`` are not all on one device.

    See Also
    --------
    [`stitch_loops_min_weight`][ordito.holes.stitch_loops_min_weight]
    [`stitch`][ordito.holes.stitch]

    Notes
    -----
    The band is the minimum-weight two-loop stitch; see
    [`stitch_min_weight`][ordito.holes.stitch_min_weight] for the metrics.
    """
    require_same_device(
        vertices_a=vertices_a, faces_a=faces_a, vertices_b=vertices_b, faces_b=faces_b
    )
    loop_a, loop_b = _single_boundary_loops(
        vertices_a, faces_a, vertices_b, faces_b, caller="stitch_min_weight"
    )
    return stitch_loops_min_weight(
        vertices_a, faces_a, loop_a, vertices_b, faces_b, loop_b, metric, up_dir
    )


@overload
def stitch_smooth(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    max_edge_splits: int = 1000,
    max_angle_change_after_flip: float = math.radians(30.0),
    smooth_curvature: bool = True,
    natural_smooth: bool = False,
    edge_weights: str = "cotan",
    refine: Literal["max_edge", "density"] = "max_edge",
    return_patch: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def stitch_smooth(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    max_edge_splits: int = 1000,
    max_angle_change_after_flip: float = math.radians(30.0),
    smooth_curvature: bool = True,
    natural_smooth: bool = False,
    edge_weights: str = "cotan",
    refine: Literal["max_edge", "density"] = "max_edge",
    return_patch: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]: ...
def stitch_smooth(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    max_edge_splits: int = 1000,
    max_angle_change_after_flip: float = math.radians(30.0),
    smooth_curvature: bool = True,
    natural_smooth: bool = False,
    edge_weights: str = "cotan",
    refine: Literal["max_edge", "density"] = "max_edge",
    return_patch: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]
):
    """
    Stitch two open meshes with a smooth, refined band.

    Like [`stitch_min_weight`][ordito.holes.stitch_min_weight] but the connecting band is then
    subdivided and smoothed by the same finisher as
    [`fill_smooth`][ordito.holes.fill_smooth]. Each mesh must have exactly one
    boundary loop. The cross-boundary smooth solve is always applied.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        ``(3 * n_faces,)`` flat triangle index buffers of each mesh.
    metric
        Stitch metric; see
        [`stitch_loops_min_weight`][ordito.holes.stitch_loops_min_weight].
    up_dir
        Up direction for the ``"vertical"`` stitch metric.
    triangulate_only
        See [`fill_smooth`][ordito.holes.fill_smooth].
    max_edge
        See [`fill_smooth`][ordito.holes.fill_smooth].
    max_edge_splits
        See [`fill_smooth`][ordito.holes.fill_smooth].
    max_angle_change_after_flip
        See [`fill_smooth`][ordito.holes.fill_smooth].
    smooth_curvature
        See [`fill_smooth`][ordito.holes.fill_smooth].
    natural_smooth
        See [`fill_smooth`][ordito.holes.fill_smooth].
    edge_weights
        See [`fill_smooth`][ordito.holes.fill_smooth].
    refine
        See [`fill_smooth`][ordito.holes.fill_smooth].
    return_patch
        See [`fill_smooth`][ordito.holes.fill_smooth].

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` concatenated vertices followed by any inserted band vertices, on
        ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` concatenated, reindexed faces followed by the band faces.
    patch_mask : wp.array[wp.bool]
        ``(n_out_faces,)`` mask of the band faces; only when ``return_patch`` is ``True``.

    Raises
    ------
    ValueError
        If either mesh lacks exactly one boundary loop, or ``metric`` / ``edge_weights`` /
        ``refine`` is unknown.
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b`` and ``faces_b`` are not all on one device.

    See Also
    --------
    [`stitch_min_weight`][ordito.holes.stitch_min_weight]
    [`fill_smooth`][ordito.holes.fill_smooth]

    Notes
    -----
    The three stages are: minimum-weight stitch, refine the band to the target edge length, then
    smooth its new interior vertices into both surrounding surfaces.
    """
    require_same_device(
        vertices_a=vertices_a, faces_a=faces_a, vertices_b=vertices_b, faces_b=faces_b
    )
    _check_stitch_metric(metric)
    if edge_weights not in ("cotan", "unit"):
        raise ValueError(f"edge_weights must be 'cotan' or 'unit', got {edge_weights!r}")
    _check_refine(refine)

    loop_a, loop_b = _single_boundary_loops(
        vertices_a, faces_a, vertices_b, faces_b, caller="stitch_smooth"
    )

    device = faces_a.device
    n_faces_before = (faces_a.size + faces_b.size) // 3
    n_vertices_before = vertices_a.size + vertices_b.size
    combined_vertices, combined_faces = stitch_loops_min_weight(
        vertices_a, faces_a, loop_a, vertices_b, faces_b, loop_b, metric, up_dir
    )
    n_faces_after = combined_faces.size // 3
    patch_mask = _patch_mask(n_faces_before, n_faces_after, device)

    if triangulate_only:
        result = (combined_vertices, combined_faces)
        return (*result, patch_mask) if return_patch else result

    # Independent copies, not views: ``boundary_loops`` returns slices of one shared packed buffer
    # and ``_mean_rim_edge_length`` must not alias it. ``wp.clone`` says exactly that; the round
    # trip through ``.numpy()`` these used to take said nothing and crossed the bus twice. The
    # second loop's shift into the combined numbering is elementwise, so it maps on the device.
    shifted_loop_b = _launch.empty(loop_b.size, dtype=wp.int32, device=device)
    _launch.map(wp.add, loop_b, wp.int32(vertices_a.size), out=shifted_loop_b)
    rim_loops = [_launch.clone(loop_a), shifted_loop_b]
    target_edge = (
        max_edge if max_edge is not None else _mean_rim_edge_length(combined_vertices, rim_loops)
    )
    new_vertices, new_faces, out_patch = od.smoothing.refine_and_smooth_region(
        combined_vertices,
        combined_faces,
        n_vertices_before,
        patch_mask,
        target_edge,
        max_edge_splits,
        max_angle_change_after_flip,
        smooth_curvature,
        True,  # the reference pipeline forces the cross-boundary smooth here
        natural_smooth,
        edge_weights,
        refine=refine,
    )
    return (new_vertices, new_faces, out_patch) if return_patch else (new_vertices, new_faces)


def _check_fill_metric(metric: str) -> None:
    """
    Reject an unknown fill metric, for the four entry points that take one.

    The four reach the engine (``_fill_packed_loops``) by different routes and three of them have
    paths that never reach it at all -- an empty mesh, a mesh with no hole, an empty loop list --
    so the check belongs at each boundary rather than at the table lookup. One spelling in one
    place because the message names the admissible set and has to stay derived from
    ``_METRIC_IDS``; ``fill_loops_min_weight`` went without it and surfaced a bare ``KeyError``.

    Raises
    ------
    ValueError
        If ``metric`` is not a key of ``_METRIC_IDS``.
    """
    if metric not in _METRIC_IDS:
        raise ValueError(f"metric must be one of {sorted(_METRIC_IDS)}, got {metric!r}")


def _check_stitch_metric(metric: str) -> None:
    """
    Reject an unknown stitch metric, for the two entry points that take one.

    The band metrics are a different menu from the fill metrics, so this is a separate table and a
    separate check; see ``_check_fill_metric`` for why the check sits at the boundary.

    Raises
    ------
    ValueError
        If ``metric`` is not a key of ``_STITCH_METRIC_IDS``.
    """
    if metric not in _STITCH_METRIC_IDS:
        raise ValueError(f"metric must be one of {sorted(_STITCH_METRIC_IDS)}, got {metric!r}")


def _check_refine(refine: str) -> None:
    """
    Reject an unknown subdivision criterion, for the three ``*_smooth`` / ``refill`` entry points.

    [`smoothing.refine_and_smooth_region`][ordito.smoothing.refine_and_smooth_region] raises on
    the same names, but each of these functions has paths that never reach it -- an empty mesh,
    ``triangulate_only``, a face mask selecting nothing -- so without this the documented
    ``Raises`` would hold on one path and not another, and a typo'd criterion would be silently
    accepted on the cheap ones.

    Raises
    ------
    ValueError
        If ``refine`` is neither ``"max_edge"`` nor ``"density"``.
    """
    if refine not in ("max_edge", "density"):
        raise ValueError(f"unknown refine {refine!r}, expected 'max_edge' or 'density'")


def _single_boundary_loops(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    caller: str,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Return the one boundary loop of each mesh, or raise naming ``caller``.

    The shared precondition of the whole ``stitch*`` family: each mesh must have exactly one hole
    to zip to the other's. Loops shorter than three vertices are degenerate rather than holes and
    are dropped before counting, so a mesh carrying one is rejected on its real loop count.
    """
    loops_a = [loop for loop in od.boundary.boundary_loops(vertices_a, faces_a) if loop.size >= 3]
    loops_b = [loop for loop in od.boundary.boundary_loops(vertices_b, faces_b) if loop.size >= 3]
    if len(loops_a) != 1 or len(loops_b) != 1:
        raise ValueError(
            f"{caller} requires each mesh to have exactly one boundary loop (>= 3 vertices); "
            f"got {len(loops_a)} and {len(loops_b)}"
        )
    return loops_a[0], loops_b[0]


def stitch_loops(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    loop_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    loop_b: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Join two meshes by triangulating the band between one boundary loop on each.

    The two rims are zippered into a watertight seam by a greedy, order-preserving correspondence
    -- the gap-bridging problem Barequet and Sharir (1995) pose, solved here by the standard
    minimal-perimeter heuristic: every A-edge is matched to the B vertex minimizing the
    added-triangle perimeter, the association is rotated so its shortest pair comes first, and a
    longest-increasing-subsequence correction forces the matching to be monotone
    (non-self-intersecting). **No smoothing or refinement** is applied — the seam reuses
    only the two loops' existing vertices, adding ``len(loop_a) + len(loop_b)`` bridge triangles.

    The larger loop is treated as A (the meshes are swapped internally when
    ``len(loop_a) < len(loop_b)``), so the output is invariant to argument order **except when the
    two loops are the same length**: the swap is on a strict inequality, so equal-length rims keep
    the caller's order, and A and B are not interchangeable — A's edges are matched to B's
    vertices, not the reverse — so two equal-length rims can produce a different seam either way
    round. Both loops must
    wind following the face orientation, as returned by
    [`boundary_loops`][ordito.boundary.boundary_loops]; the two boundaries are assumed to wind in
    opposite directions (as two open meshes facing each other do), so loop A is reversed to align
    them.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        ``(3 * n_faces,)`` flat triangle index buffers of each mesh.
    loop_a, loop_b
        ``(k_a,)`` and ``(k_b,)`` ordered vertex-index loops (``>= 3`` vertices each) around the
        boundary to join on each mesh, indexing ``vertices_a`` / ``vertices_b`` respectively.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` concatenation of ``vertices_a`` then ``vertices_b`` (larger loop
        first), on the device of the **larger** loop's mesh — which is the caller's ``faces_a``
        unless the swap above fired.
    new_faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` original faces (B reindexed by ``len(vertices_a)``) followed by the
        bridge triangles, on that same device.

    Raises
    ------
    ValueError
        If either loop has fewer than 3 vertices.
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``loop_a``, ``vertices_b``, ``faces_b`` and ``loop_b`` are
        not all on one device.

    See Also
    --------
    [`stitch_loops_min_weight`][ordito.holes.stitch_loops_min_weight]
        The more robust choice, and the one to reach for when seam quality matters: it minimizes a
        triangulation metric over the whole rim pair instead of correcting a greedy correspondence.
    [`stitch`][ordito.holes.stitch]
    [`boundary_loops`][ordito.boundary.boundary_loops]
    [`concatenate`][ordito.combine.concatenate]

    Notes
    -----
    The correspondence and its monotonicity correction are inherently sequential and run on the
    host over the length-``len(loop_a)`` association array; the O(N·M) perimeter matrix, the
    reductions, and the triangle emission run in Warp kernels, and the perimeter matrix itself is
    never copied off the device.

    **This is the fast one, and that is the reason to keep it.** It is metric-free and makes one
    O(N·M) pass, where
    [`stitch_loops_min_weight`][ordito.holes.stitch_loops_min_weight] has to fill the same size
    table under a dependency that makes it ``N + M`` sequential steps. The DP is never the cheaper
    option -- prefer it for the seam it produces, not for speed.
    """
    require_same_device(
        vertices_a=vertices_a,
        faces_a=faces_a,
        loop_a=loop_a,
        vertices_b=vertices_b,
        faces_b=faces_b,
        loop_b=loop_b,
    )
    n = loop_a.size
    m = loop_b.size
    if n < m:
        vertices_a, vertices_b = vertices_b, vertices_a
        faces_a, faces_b = faces_b, faces_a
        loop_a, loop_b = loop_b, loop_a
        n, m = m, n
    # Read *after* the swap: every allocation and launch below is on the A mesh's device, and the
    # swap is what decides which of the caller's two meshes that is.
    device = faces_a.device
    if m < 3:
        raise ValueError(f"each boundary loop must have at least 3 vertices, got {n} and {m}")
    n_vertices_a = vertices_a.size

    # Rim positions, loop A reversed so both rims wind the same way. Everything below reads loop A
    # reversed (``kernels/holes.rim_vertex``): neither the reversed nor the rolled loops are
    # written.
    a_pos = _launch.empty(n, dtype=wp.vec3, device=device)
    b_pos = _launch.empty(m, dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_holes.stitch_rim_positions,
        dim=n + m,
        inputs=[vertices_a, loop_a, vertices_b, loop_b],
        outputs=[a_pos, b_pos],
        device=device,
    )

    # Each A edge's best B vertex by the perimeter |a_i - b_j| + |a_{i+1} - b_j|.
    col_min = _launch.empty(n, dtype=wp.int32, device=device)
    val_min = _launch.empty(n, dtype=wp.float32, device=device)
    _launch.launch_tiled(
        kernel_holes.row_argmin,
        dim=[n],
        inputs=[a_pos, b_pos, wp.int32(n), wp.int32(m), kernel_holes.LOOP_PAIR_PERIMETER],
        outputs=[col_min, val_min],
        block_dim=kernel_holes.ROW_ARGMIN_BLOCK,
        device=device,
    )

    # ``[edge_0 .. edge_{n-1}, shift_a, shift_b]`` in one buffer: the edge map reads the two
    # shifts on the device, so the host reads the whole answer back once rather than the shifts
    # first and the edge map after.
    edge_and_shift = _launch.empty(n + 2, dtype=wp.int32, device=device)
    shift = edge_and_shift[n:]
    _launch.launch(
        kernel_holes.global_argmin,
        dim=1,
        inputs=[col_min, val_min, wp.int32(n), shift],
        device=device,
    )
    _launch.launch(
        kernel_holes.rolled_edge_map,
        dim=n,
        inputs=[col_min, shift, wp.int32(n), wp.int32(m), edge_and_shift[:n]],
        device=device,
    )
    edge_and_shift_np = edge_and_shift.numpy()
    edge = edge_and_shift_np[:n]
    shift_a = int(edge_and_shift_np[n])
    shift_b = int(edge_and_shift_np[n + 1])

    row_roll = shift_a
    col_roll = shift_b

    # Wraparound-group fix: if the lowest-index group is split across the ends of ``edge``, roll
    # A so the group is contiguous. This has to run *before* the monotonicity pass below, which
    # reads ``edge`` as a linear sequence and would otherwise see one group as two.
    if edge[-1] == edge[0]:
        trailing = int(np.argmin(np.flip(edge) == edge[0]))
        if trailing > 0:
            edge = np.roll(edge, trailing)
            row_roll = (shift_a - trailing) % n

    # Monotonicity correction: re-pick the B vertex of every edge outside the longest
    # non-decreasing subsequence within the bracket of its stable neighbours, so the matching is
    # order-preserving. The sentinel value ``m`` closes the last bracket.
    edge_ext = np.append(edge, np.int32(m))
    out_edge = _launch.array(edge_ext.astype(np.int32), dtype=wp.int32, device=device)
    if not np.all(np.diff(edge) >= 0):
        unsorted_indices = _non_increasing_indices(edge_ext)
        stable_indices = np.delete(np.arange(edge_ext.size), unsorted_indices)
        next_indices = stable_indices[np.searchsorted(stable_indices, unsorted_indices)]
        _launch.launch(
            kernel_holes.resolve_corrections,
            dim=1,
            inputs=[
                a_pos,
                b_pos,
                _launch.array(unsorted_indices.astype(np.int32), dtype=wp.int32, device=device),
                _launch.array(next_indices.astype(np.int32), dtype=wp.int32, device=device),
                wp.int32(unsorted_indices.size),
                wp.int32(row_roll),
                wp.int32(col_roll),
                wp.int32(n),
                wp.int32(m),
                out_edge,
            ],
            device=device,
        )

    # The band, referencing the concatenated vertex buffer (A first, then B offset).
    bridge = _launch.empty(3 * (n + m), dtype=wp.int32, device=device)
    _launch.launch(
        kernel_holes.bridge_faces,
        dim=n + m,
        inputs=[
            loop_a,
            loop_b,
            out_edge,
            wp.int32(row_roll),
            wp.int32(col_roll),
            wp.int32(n_vertices_a),
        ],
        outputs=[bridge],
        device=device,
    )

    combined_vertices, combined_faces = od.combine.concatenate(
        [(vertices_a, faces_a), (vertices_b, faces_b)]
    )
    return combined_vertices, od.array.concatenate([combined_faces, bridge])


def _non_increasing_indices(numbers: np.ndarray) -> np.ndarray:
    """
    Return the entry indices **not** in the longest non-decreasing subsequence of ``numbers``.

    Repeated integers are perturbed by adding ``linspace(0, 1, count, endpoint=False)`` within
    each equal-value group, so a run of equal values is treated as (weakly) increasing and kept.
    The equal-value grouping is done in NumPy rather than through ``trimesh.grouping.group``, so
    that nothing here puts ``trimesh`` on the shipped package's dependency list.
    """
    different = numbers.astype(np.float64)
    order = np.argsort(numbers, kind="stable")
    sorted_values = numbers[order]
    cut = np.flatnonzero(np.diff(sorted_values)) + 1
    for group in np.split(order, cut):
        different[group] += np.linspace(0.0, 1.0, num=len(group), endpoint=False)
    subsequence = _longest_increasing_subsequence(different)
    absence_mask = np.isin(different, subsequence, assume_unique=True, invert=True)
    return np.flatnonzero(absence_mask)


def _longest_increasing_subsequence(numbers: np.ndarray) -> np.ndarray:
    """
    Longest strictly increasing subsequence of ``numbers`` (patience-sorting, O(N log N)).

    Repeated values must be pre-perturbed to distinct values (see
    [`_non_increasing_indices`][ordito.holes._non_increasing_indices]); the algorithm does
    not handle ties.
    """
    p = np.zeros_like(numbers, dtype=np.int64)
    m = -np.ones(numbers.size + 1, dtype=np.int64)
    size = 0
    for i, x in enumerate(numbers):
        subseq_size = int(np.searchsorted(numbers[m[1 : size + 1]], x))
        p[i] = m[subseq_size]
        m[subseq_size + 1] = i
        size = max(subseq_size + 1, size)
    subseq = np.empty(size, numbers.dtype)
    k = m[size]
    for i in range(size - 1, -1, -1):
        subseq[i] = numbers[k]
        k = p[k]
    return subseq


# Stitch-metric name -> kernel selector (must match the METRIC_*_STITCH constants in
# kernels/holes.py).
_STITCH_METRIC_IDS = {"complex_stitch": 0, "edge_length_stitch": 1, "vertical": 2}


def stitch_loops_min_weight(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    loop_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    loop_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Join two meshes with a **minimum-weight** cylindrical band between one boundary loop on each.

    The two rims are aligned at their closest vertex pair and zippered by the band of
    ``len(loop_a) + len(loop_b)`` triangles that minimizes a stitch metric, found by a grid dynamic
    program over the two loops (``dp[i, j]`` = best band consuming ``i`` A-edges and ``j`` B-edges,
    each cell reading only the two before it, so the grid is filled one anti-diagonal at a time).
    Reuses only the loops' existing vertices. The metric-free greedy
    [`stitch_loops`][ordito.holes.stitch_loops] remains available.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        ``(3 * n_faces,)`` flat triangle index buffers of each mesh.
    loop_a, loop_b
        ``(k_a,)`` and ``(k_b,)`` ordered vertex-index loops (``>= 3`` vertices each) around the
        boundary to join on each.
    metric
        Stitch metric to minimize:

        - ``"complex_stitch"`` (default) — triangle aspect ratio plus a dihedral-smoothness edge
          term between adjacent band triangles and the surface.
        - ``"edge_length_stitch"`` — summed connection-edge length.
        - ``"vertical"`` — penalizes band area and normal deviation from ``up_dir``; pass
          ``up_dir``.
    up_dir
        Up direction for the ``"vertical"`` metric (defaults to ``(0, 0, 1)``); ignored otherwise.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` concatenation of ``vertices_a`` then ``vertices_b``, on
        ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` original faces (B reindexed by ``len(vertices_a)``) followed by the
        band triangles.

    Raises
    ------
    ValueError
        If either loop has fewer than 3 vertices, or ``metric`` is unknown.
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``loop_a``, ``vertices_b``, ``faces_b`` and ``loop_b`` are
        not all on one device.

    See Also
    --------
    [`stitch_loops`][ordito.holes.stitch_loops]
    [`stitch_min_weight`][ordito.holes.stitch_min_weight]
    """
    require_same_device(
        vertices_a=vertices_a,
        faces_a=faces_a,
        loop_a=loop_a,
        vertices_b=vertices_b,
        faces_b=faces_b,
        loop_b=loop_b,
    )
    _check_stitch_metric(metric)
    n_a = loop_a.size
    n_b = loop_b.size
    if n_a < 3 or n_b < 3:
        raise ValueError(f"each boundary loop must have at least 3 vertices, got {n_a} and {n_b}")

    device = faces_a.device
    metric_id = _STITCH_METRIC_IDS[metric]
    offset = vertices_a.size

    # Reverse loop A so the two rims wind oppositely (facing), then align both at the closest pair.
    # ``la`` / ``lb`` stay on the host for the sequential band traceback, but the closest-pair
    # search, rim gathers and rim-opposite lookups all run on device. This preamble (two readbacks,
    # the closest-pair gather, the host roll and two more uploads) is a fixed cost that the
    # anti-diagonal DP below dominates and outgrows as the rims lengthen.
    la = loop_a.numpy()[::-1].copy()
    lb = loop_b.numpy().copy()
    a_rim = od.array.gather(vertices_a, _launch.array(la, dtype=wp.int32, device=device))
    b_rim = od.array.gather(vertices_b, _launch.array(lb, dtype=wp.int32, device=device))
    start_a, start_b = _closest_loop_pair(a_rim, b_rim)
    la = np.roll(la, -start_a)
    lb = np.roll(lb, -start_b)

    la_wp = _launch.array(la, dtype=wp.int32, device=device)
    lb_wp = _launch.array(lb, dtype=wp.int32, device=device)
    a_pos = od.array.gather(vertices_a, la_wp)
    b_pos = od.array.gather(vertices_b, lb_wp)
    table_a = _EdgeTable(vertices_a, faces_a)
    table_b = _EdgeTable(vertices_b, faces_b)
    # The stitch DP works on exactly one rim per side, so each rim is its own one-loop batch.
    a_opp, a_opp_valid = table_a.rim_opposite(_PackedLoops(la_wp, np.array([n_a], dtype=np.int64)))
    b_opp, b_opp_valid = table_b.rim_opposite(_PackedLoops(lb_wp, np.array([n_b], dtype=np.int64)))
    up = wp.vec3(*(up_dir if up_dir is not None else (0.0, 0.0, 1.0)))

    dp = odt.as_array2d(
        _launch.full(
            (n_a + 1, n_b + 1), _BAD_TRIANGULATION_METRIC, dtype=wp.float32, device=device
        ),
        wp.float32,
    )
    _launch.launch(kernel_holes.set_dp_origin, dim=1, inputs=[dp], device=device)
    came = odt.as_array2d(
        _launch.full((n_a + 1, n_b + 1), -1, dtype=wp.int32, device=device), wp.int32
    )
    # Built once, outside the loop: the ten values below are the same on every one of the
    # ``n_a + n_b`` launches, and a wp.launch argument costs host time linearly on both devices, so
    # bundling them once here rather than passing all ten on every launch is a real saving.
    tables = kernel_holes.StitchTables()
    tables.a_pos = a_pos
    tables.b_pos = b_pos
    tables.a_opp = a_opp
    tables.a_opp_valid = a_opp_valid
    tables.b_opp = b_opp
    tables.b_opp_valid = b_opp_valid
    tables.up = up
    tables.metric_id = wp.int32(metric_id)
    tables.n_a = wp.int32(n_a)
    tables.n_b = wp.int32(n_b)
    _run_stitch_dp(tables, n_a, n_b, dp, came, device)
    came_np = came.numpy()

    band = _stitch_band_triangles(came_np, la, lb, n_a, n_b, offset)
    combined_vertices, combined_faces = od.combine.concatenate(
        [(vertices_a, faces_a), (vertices_b, faces_b)]
    )
    band_faces = _launch.array(band.reshape(-1), dtype=wp.int32, device=device)
    return combined_vertices, od.array.concatenate([combined_faces, band_faces])


def _run_stitch_dp(
    tables: StructInstance,
    n_a: int,
    n_b: int,
    dp: odt.Array2dFloat32,
    came: odt.Array2dInt32,
    device: wp.DeviceLike,
    tiled: bool | None = None,
) -> None:
    """
    Fill the ``(n_a + 1) x (n_b + 1)`` band tables in place, seeded at cell ``(0, 0)``.

    A cell reads only the cells above and to the left of it, so the grid needs ``n_a + n_b``
    sequential steps whichever way it is cut. ``tiled`` chooses where those steps' barriers come
    from, and it is a pure cost knob -- both schedules compute the same cells with the same
    ``stitch_dp_cell``, so ``dp`` and ``came`` come out byte-identical and ``None`` simply picks
    whichever is faster on the device at hand.

    * ``False`` walks one anti-diagonal per launch: ``n_a + n_b`` of them, each a barrier. This is
      the reference schedule, kept because it is the one the tiled kernel has to agree with.
    * ``True`` gives one **block** a ``tile x tile`` square and launches one tile-diagonal at a
      time, so all but ``2 * ceil(n / tile)`` of the barriers become block-local. That is the whole
      cost of this DP -- the per-cell work is small and the sweep was launch-bound -- and it is the
      default on **both** devices, unlike the fill DP's tiled engine: the CPU device runs one lane
      per block, so its gain is the launch count alone rather than the lanes, and it still takes it.
    """
    if tiled is None:
        tiled = True
    if not tiled:
        for diag in range(1, n_a + n_b + 1):
            _launch.launch(
                kernel_holes.stitch_dp_diag,
                dim=min(diag, n_a) - max(0, diag - n_b) + 1,
                inputs=[tables, wp.int32(diag), dp, came],
                device=device,
            )
        return
    tile = kernel_holes.STITCH_DP_TILE
    rows = -(-(n_a + 1) // tile)
    cols = -(-(n_b + 1) // tile)
    for block_diag in range(rows + cols - 1):
        bi_lo = max(0, block_diag - cols + 1)
        bi_hi = min(block_diag, rows - 1)
        _launch.launch_tiled(
            kernel_holes.stitch_dp_tile,
            dim=bi_hi - bi_lo + 1,
            inputs=[tables, wp.int32(block_diag), wp.int32(bi_lo), wp.int32(tile), dp, came],
            block_dim=tile,
            device=device,
        )


def _closest_loop_pair(a_pos: wp.array[wp.vec3], b_pos: wp.array[wp.vec3]) -> tuple[int, int]:
    """
    Return the closest vertex pair ``(i, j)`` between the two rims: the band's start pair.

    Each rim-A vertex's nearest rim-B vertex and the argmin over those run in Warp kernels (the
    same ``row_argmin`` / ``global_argmin`` reduction the greedy zippering uses, at the squared
    distance rather than the perimeter); only the two winning indices come back to the host. Ties
    resolve to the smallest ``i`` then smallest ``j``, matching ``numpy.argmin`` on the flattened
    squared-distance matrix, which is never built.

    ``kernels/holes.reduce_closest_cross_label_pair`` answers the same question in **one** kernel
    by reducing a ``pack_nearest_key`` atomic over labelled members, but the two are not one
    function wearing two hats: that kernel takes members and labels over a shared vertex buffer,
    this takes two separate position arrays, and the launches here reuse ``row_argmin`` /
    ``global_argmin``, which the caller needs anyway for its *perimeter* objective.
    """
    device = a_pos.device
    n_a = a_pos.size
    n_b = b_pos.size
    col_min = _launch.empty(n_a, dtype=wp.int32, device=device)
    val_min = _launch.empty(n_a, dtype=wp.float32, device=device)
    _launch.launch_tiled(
        kernel_holes.row_argmin,
        dim=[n_a],
        inputs=[a_pos, b_pos, wp.int32(n_a), wp.int32(n_b), kernel_holes.LOOP_PAIR_SQ_DISTANCE],
        outputs=[col_min, val_min],
        block_dim=kernel_holes.ROW_ARGMIN_BLOCK,
        device=device,
    )
    pair = _launch.empty(2, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_holes.global_argmin,
        dim=1,
        inputs=[col_min, val_min, wp.int32(n_a), pair],
        device=device,
    )
    pair_np = pair.numpy()
    return int(pair_np[0]), int(pair_np[1])


def _stitch_band_triangles(
    came_np: np.ndarray, la: np.ndarray, lb: np.ndarray, n_a: int, n_b: int, offset: int
) -> np.ndarray:
    """Trace the grid-DP came-from table from ``(n_a, n_b)`` to ``(0, 0)`` into band triangles."""
    triangles: list[tuple[int, int, int]] = []
    i, j = n_a, n_b
    while i > 0 or j > 0:
        if came_np[i, j] == 0:  # advanced A: triangle (a[i-1], a[i], b[j])
            triangles.append((int(la[(i - 1) % n_a]), int(la[i % n_a]), int(lb[j % n_b]) + offset))
            i -= 1
        elif came_np[i, j] == 1:  # advanced B: triangle (a[i], b[j], b[j-1])
            triangles.append(
                (int(la[i % n_a]), int(lb[j % n_b]) + offset, int(lb[(j - 1) % n_b]) + offset)
            )
            j -= 1
        else:  # unreachable cell (should not happen for valid rims)
            break
    return np.asarray(triangles, dtype=np.int32)


def bridge_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edge_a: tuple[int, int],
    edge_b: tuple[int, int],
    validate: bool = True,
) -> wp.array[wp.int32]:
    """
    Join two boundary edges with a two-triangle patch, leaving the rest of both rims open.

    Where [`stitch_loops`][ordito.holes.stitch_loops] consumes two *complete* rims, this is the
    local operation underneath it: pick one edge on each side and close only that gap. Two open
    rims become one, so a bridge is how a tube is joined to another tube at a chosen seam, how a
    partially torn boundary is tacked back together, and -- when both edges lie on the *same* rim --
    how a handle is added, since bridging one loop to itself splits it into two.

    The patch is the quadrilateral ``(a1, a0, b1, b0)`` split along the ``a0 - b0`` diagonal, so it
    adds **two triangles and no vertices**. When the two edges already share a vertex -- they are
    consecutive along one rim -- that quadrilateral is a triangle and only **one** is added.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Only read to validate the two edges; the patch
        itself is purely topological, and nothing moves.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    edge_a
        A boundary edge as ``(v0, v1)``, **directed the way its face winds it** -- a row of
        [`oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges].
    edge_b
        The second boundary edge, in the same direction convention.
    validate
        Check that both edges are boundary edges of ``faces`` and that the patch would not
        duplicate an existing edge. Costs one pass over the faces and one readback, and builds no
        edge table; pass ``False`` when the edges came from
        [`oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges] and the pairing is
        known good.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces + 3,)`` or ``(3 * n_faces + 6,)`` input faces with the patch appended, on
        ``faces.device``. The appended block is the tail -- ``3`` entries longer than the input for
        the shared-vertex case, ``6`` otherwise -- so its size is what says which case was taken.

    Raises
    ------
    ValueError
        If the two edges are the same edge, if ``validate`` is set and either is not a boundary
        edge of ``faces``, or if ``validate`` is set and the patch would create a second edge
        between a pair of vertices that already share one (which would leave the mesh
        non-manifold).
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Examples
    --------
    ```python
    edges_np = od.boundary.oriented_boundary_edges(open_v, open_f).numpy()
    bridged_f = od.holes.bridge_edges(
        open_v, open_f, tuple(edges_np[0]), tuple(edges_np[len(edges_np) // 2])
    )
    ```

    See Also
    --------
    [`bridge_edges_smooth`][ordito.holes.bridge_edges_smooth]
        The multi-segment form, which curves the patch and adds vertices.
    [`join_closest_components`][ordito.holes.join_closest_components]
        The driver over this, when the pair to bridge is not the caller's to name.
    [`stitch_loops`][ordito.holes.stitch_loops]
        Joins two rims completely, where this joins one edge of each.
    [`oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges]
        Produces the edges this takes, already in the right direction.
    """
    require_same_device(vertices=vertices, faces=faces)
    edge_a = (int(edge_a[0]), int(edge_a[1]))
    edge_b = (int(edge_b[0]), int(edge_b[1]))
    _require_distinct_edges(edge_a, edge_b)
    if validate:
        joined = _bridge_joined_pairs(edge_a, edge_b)
        census = _bridge_census(faces, [edge_a, edge_b, *joined])
        _require_rim_edges(edge_a, edge_b, census)
        _require_new_chords(edge_a, edge_b, joined, census[2:])
    return _append_bridge_patch(faces, edge_a, edge_b)


def _append_bridge_patch(
    faces: wp.array[wp.int32], edge_a: tuple[int, int], edge_b: tuple[int, int]
) -> wp.array[wp.int32]:
    """Append the flat patch's one or two triangles to ``faces``."""
    triangles = _bridge_triangles(edge_a, edge_b)
    patch = _launch.array(
        np.asarray(triangles, dtype=np.int32).reshape(-1), dtype=wp.int32, device=faces.device
    )
    return od.array.concatenate([faces, patch])


def _bridge_joined_pairs(edge_a: tuple[int, int], edge_b: tuple[int, int]) -> list[tuple[int, int]]:
    """List the pairs of *existing* vertices the flat patch joins: its two sides and diagonal."""
    a0, a1 = edge_a
    b0, b1 = edge_b
    incident = {(min(a0, a1), max(a0, a1)), (min(b0, b1), max(b0, b1))}
    pairs = []
    for triangle in _bridge_triangles(edge_a, edge_b):
        for k in range(3):
            u, v = triangle[k], triangle[(k + 1) % 3]
            key = (min(u, v), max(u, v))
            if key not in incident and key not in pairs:
                pairs.append(key)
    return pairs


def bridge_edges_smooth(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edge_a: tuple[int, int],
    edge_b: tuple[int, int],
    sampling_step: float,
    validate: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Join two boundary edges with a curved strip that leaves both surfaces smoothly.

    [`bridge_edges`][ordito.holes.bridge_edges] spans the gap with a flat patch, which creases
    against both surfaces as soon as the two edges are more than a triangle apart. This spans it
    with a strip of quadrilaterals instead, following the cubic through the two edge midpoints
    whose end tangents lie **in** the two incident triangles -- so the strip leaves each surface in
    the direction that surface was already going, and the crease is spread over the whole strip
    rather than concentrated at its two ends.

    The strip is subdivided until its segments are no longer than ``sampling_step``, and its width
    tapers linearly from the length of ``edge_a`` to that of ``edge_b``. Its two boundary chains
    start at the two ends of ``edge_a`` and finish at the two ends of ``edge_b``, so the existing
    four vertices are reused and only the interior ones are new. As in the flat form, two edges
    that already share a vertex give a fan from that vertex rather than a strip.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    edge_a
        A boundary edge as ``(v0, v1)``, directed the way its face winds it.
    edge_b
        The second boundary edge, in the same direction convention.
    sampling_step
        Target segment length along the strip. Smaller means more segments and a smoother patch;
        a step at or above the span gives the single segment
        [`bridge_edges`][ordito.holes.bridge_edges] would have produced, on the curve rather than
        on the chord.
    validate
        Check that both edges are boundary edges of ``faces`` and that the patch would not
        duplicate an existing edge, as in [`bridge_edges`][ordito.holes.bridge_edges].

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_out_vertices,)`` input positions, unchanged and in order, with the strip's interior
        vertices appended.
    faces : wp.array[wp.int32]
        ``(3 * n_out_faces,)`` input faces with the strip's triangles appended.

    Raises
    ------
    ValueError
        If ``sampling_step`` is not positive, if either edge is not a directed edge of ``faces``
        (checked whatever ``validate`` says, since the strip needs both incident faces), or for any
        of the reasons [`bridge_edges`][ordito.holes.bridge_edges] raises.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`bridge_edges`][ordito.holes.bridge_edges]
        The single-quadrilateral form, which adds no vertices.
    [`fill_smooth`][ordito.holes.fill_smooth]
        The same idea for a whole rim: a patch refined and faired rather than merely spanned.

    Notes
    -----
    !!! note "The curve is a cubic, not an optimum"
        The strip follows one cubic Hermite segment fitted to the two edge midpoints and the two
        incident-face tangents. Nothing minimizes its bending energy or checks it for
        self-intersection, so a step far smaller than the span across two nearly opposed edges can
        fold; run [`fix_self_intersections`][ordito.repair.fix_self_intersections] if the input
        pairing is not yours to choose.

    !!! note "Two opposed edges give a half-twisted strip, not an error"
        The strip's width direction interpolates from ``edge_a``'s to ``edge_b``'s. When the two
        run opposite ways the interpolation passes through a half turn whose axis is arbitrary, and
        the strip twists through it at one sample rather than curving smoothly. Reversing one of
        the two edges is what removes the twist.
    """
    require_same_device(vertices=vertices, faces=faces)
    if sampling_step <= 0.0:
        raise ValueError(f"sampling_step must be positive, got {sampling_step}")
    device = faces.device
    a0, a1 = int(edge_a[0]), int(edge_a[1])
    b0, b1 = int(edge_b[0]), int(edge_b[1])
    _require_distinct_edges((a0, a1), (b0, b1))
    # One scan of the faces answers the whole validation and finds each edge's opposite corner: a
    # strip with interior samples joins nothing but its own new vertices, so only the one-segment
    # case -- the flat patch -- reads the chord half, which is asked for up front so that case needs
    # no second scan. Then one gather of the six positions the spline needs. Both are here so the
    # host never reads back a buffer that scales with the mesh.
    joined = _bridge_joined_pairs((a0, a1), (b0, b1)) if validate else []
    census = _bridge_census(faces, [(a0, a1), (b0, b1), *joined])
    if validate:
        _require_rim_edges((a0, a1), (b0, b1), census)
    opposite_a, opposite_b = int(census[0, 2]) - 1, int(census[1, 2]) - 1
    if opposite_a < 0 or opposite_b < 0:
        raise ValueError("both edges must be directed edges of faces, wound as their face winds")
    corners = np.array([a0, a1, b0, b1, opposite_a, opposite_b], dtype=np.int32)
    gathered = _launch.empty(6, dtype=wp.vec3, device=device)
    _launch.copy(gathered, vertices[_launch.array(corners, dtype=wp.int32, device=device)])
    positions_np = gathered.numpy().astype(np.float64)

    n_vertices = vertices.size
    interior_np, strip_np = _bridge_strip(positions_np, corners, n_vertices, sampling_step)
    if interior_np.shape[0] == 0:
        # One segment: the strip *is* the flat patch, which adds edges between existing vertices,
        # so the chord half of the check applies to it as it does in ``bridge_edges``.
        if validate:
            _require_new_chords((a0, a1), (b0, b1), joined, census[2:])
        return _launch.clone(vertices), _append_bridge_patch(faces, (a0, a1), (b0, b1))

    bridged_vertices = _launch.empty(
        n_vertices + interior_np.shape[0], dtype=wp.vec3, device=device
    )
    _launch.copy(bridged_vertices[:n_vertices], vertices)
    _launch.copy(
        bridged_vertices[n_vertices:],
        _launch.array(interior_np.astype(np.float32), dtype=wp.vec3, device=device),
    )
    strip = _launch.array(strip_np.reshape(-1), dtype=wp.int32, device=device)
    return bridged_vertices, od.array.concatenate([faces, strip])


def _bridge_strip(
    positions_np: np.ndarray, corners: np.ndarray, n_vertices: int, sampling_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the bridge cubic and build the strip: interior positions, and the triangle rows.

    Host-side NumPy over six positions and a handful of samples -- the strip never scales with the
    mesh, so a kernel here would buy a launch and no parallelism.

    **Slerping the width direction instead of normalizing a lerp was measured and declined.** The
    two spellings trace the same great-circle arc at different speeds, so adopting slerp would move
    every sample of every call; what it buys, against the angle between the two edges' width
    directions, is nothing where callers actually are and a real improvement only in a band nothing
    reaches:

    | angle apart | ``max|lerp - slerp|``, as a fraction of the width | worst swing, lerp -> slerp |
    |---|---|---|
    | 5 deg | 0.00 % | 0.4 -> 0.4 deg |
    | 30 deg | 0.11 % | 2.6 -> 2.5 deg |
    | 65 deg (the benchmarked hemisphere pair) | 1.12 % | 10.5 -> 9.3 deg |
    | 90 deg | 3.55 % | 9.5 -> 7.5 deg |
    | 170 deg | 40.8 % | 62.3 -> 14.2 deg |
    | 179 deg | 58.8 % | 87.0 -> 14.9 deg |

    So it perturbs the one pair the suite times by about a percent of the width for no gain, and
    only pays past ~150 degrees -- where the docstring of the public entry point already tells the
    caller to reverse one of the two edges instead. **And it does not fix the case that raised the
    question**: at exactly 180 degrees ``sin(omega)`` is zero, slerp is ``0 / 0``, and the twist
    axis is as arbitrary as it is here, so the degenerate branch below is needed either way.
    """
    a0, a1, b0, b1 = (int(corners[k]) for k in range(4))
    pa0, pa1, pb0, pb1, pta, ptb = positions_np

    center_a, center_b = 0.5 * (pa0 + pa1), 0.5 * (pb0 + pb1)
    # The tangent lies in the incident triangle's plane, runs across its edge and points into it, so
    # the spline's end control points sit inside the two surfaces and the curve leaves them
    # tangentially rather than at an angle.
    tangent_a = _unit(np.cross(_unit(np.cross(pa1 - pa0, pta - pa0)), _unit(pa1 - pa0)))
    tangent_b = _unit(np.cross(_unit(np.cross(pb1 - pb0, ptb - pb0)), _unit(pb1 - pb0)))
    # The cubic leaves A along -tangent_a and arrives at B along +tangent_b, so it continues each
    # surface rather than turning off it, and both velocities are scaled by the span -- a cubic
    # whose end velocities do not grow with the gap it crosses is a chord with a kink at each end.
    span = float(np.linalg.norm(center_b - center_a))
    velocity_0 = -span * tangent_a
    velocity_1 = span * tangent_b

    def hermite(u: np.ndarray) -> np.ndarray:
        column = u[:, None]
        squared, cubed = column * column, column * column * column
        return (
            (2.0 * cubed - 3.0 * squared + 1.0) * center_a
            + (cubed - 2.0 * squared + column) * velocity_0
            + (-2.0 * cubed + 3.0 * squared) * center_b
            + (cubed - squared) * velocity_1
        )

    dense = hermite(np.linspace(0.0, 1.0, 64))
    arc_length = float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum())
    n_segments = max(1, math.ceil(arc_length / sampling_step))
    if n_segments == 1:
        rows = [t for t in ((a1, a0, b0), (a0, b1, b0)) if len(set(t)) == 3]
        return np.zeros((0, 3), dtype=np.float64), np.asarray(rows, dtype=np.int32).reshape(-1, 3)

    # The strip's two chains run a1 -> b0 and a0 -> b1, which are the quadrilateral's own two sides.
    # Their width tapers from one edge's length to the other's, so both ends land on the existing
    # four vertices exactly and only the interior samples are new.
    width_dir_a, width_dir_b = _unit(pa0 - pa1), _unit(pb1 - pb0)
    length_a = float(np.linalg.norm(pa1 - pa0))
    length_b = float(np.linalg.norm(pb1 - pb0))
    parameters = np.linspace(0.0, 1.0, n_segments + 1)
    samples = hermite(parameters)
    directions = (1.0 - parameters)[:, None] * width_dir_a + parameters[:, None] * width_dir_b
    # The blend collapses to zero where the two width directions are anti-parallel: the strip has
    # to make a half turn there and every half turn is as good as another, so the normalized lerp
    # has a jump discontinuity rather than a value at that sample. Take the limit from the A side.
    # Without the guard this divides by zero and returns NaN positions -- and a *nearly* opposed
    # pair is worse still, dividing by round-off to give a finite but arbitrary direction, which is
    # why the floor is a tolerance and not an equality test. Both operands are unit vectors, so the
    # norm is in ``[0, 1]`` and an absolute floor is the scale-free test here.
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    fallback = width_dir_a if float(np.linalg.norm(width_dir_a)) > 0.0 else width_dir_b
    collapsed = norms[:, 0] < TOLERANCE_ZERO
    directions[collapsed] = fallback
    norms[collapsed] = max(float(np.linalg.norm(fallback)), TOLERANCE_ZERO)
    directions /= norms
    half_width = (0.5 * ((1.0 - parameters) * length_a + parameters * length_b))[:, None]
    minus_np = samples - half_width * directions
    plus_np = samples + half_width * directions

    # A shared vertex collapses that side of the strip onto the vertex itself, turning the strip
    # into a fan -- the multi-segment form of the flat patch's one-triangle case.
    interior: list[np.ndarray] = []
    minus_ids, plus_ids = [a1], [a0]
    for i in range(1, n_segments):
        for shared, ring, ids in ((a1 == b0, minus_np, minus_ids), (a0 == b1, plus_np, plus_ids)):
            if shared:
                ids.append(ids[0])
            else:
                ids.append(n_vertices + len(interior))
                interior.append(ring[i])
    minus_ids.append(b0)
    plus_ids.append(b1)

    rows = []
    for i in range(n_segments):
        for triangle in (
            (minus_ids[i], plus_ids[i], minus_ids[i + 1]),
            (plus_ids[i], plus_ids[i + 1], minus_ids[i + 1]),
        ):
            if len(set(triangle)) == 3:
                rows.append(triangle)
    return (
        np.asarray(interior, dtype=np.float64).reshape(-1, 3),
        np.asarray(rows, dtype=np.int32).reshape(-1, 3),
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    """Normalize, leaving a zero vector alone."""
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def join_closest_components(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    max_distance: float | None = None,
    max_joins: int | None = None,
) -> wp.array[wp.int32]:
    """
    Bridge the closest pairs of *open* components until the mesh is connected.

    The driver over [`bridge_edges`][ordito.holes.bridge_edges], and the step that turns a repair
    pipeline's output into a **single** solid rather than several. Where ``bridge_edges`` joins two
    edges the caller names and [`stitch_loops`][ordito.holes.stitch_loops] consumes two complete
    rims, this answers *"these are several open shells; make them one"* -- which no other entry
    point here does, because the question it has to settle first is *which* pieces to join.

    The rule is greedy nearest-link agglomeration, i.e. Kruskal over the components with the
    distance between their boundary vertices as the edge weight: take the globally closest pair of
    boundary vertices belonging to different components, bridge it, and repeat while a cross-
    component pair remains. Each join adds exactly **two triangles and no vertices** -- the two rims
    it touches become one open rim, left for a filler to close -- so the result grows by
    ``2 * (k - 1)`` faces for ``k`` open components.

    Ties are broken by the lower boundary-vertex index, so the answer does not depend on thread
    order.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Nothing moves and nothing is added, so this is only
        read.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    max_distance
        Refuse a join whose two boundary vertices are further apart than this. ``None`` (default)
        joins unconditionally, which is what the reference implementations do -- and which on a mesh
        holding genuinely separate objects welds them together, so pass a bound when the input might
        not be one broken surface.
    max_joins
        Stop after this many bridges. ``None`` (default) continues until one component remains or no
        admissible pair is left.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_out_faces,)`` input face buffer with the bridge triangles appended, on
        ``faces.device``. A copy of ``faces`` when there is nothing to join -- fewer than two
        components, or no component with a boundary.

    Raises
    ------
    ValueError
        If ``max_joins`` is negative.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`bridge_edges`][ordito.holes.bridge_edges]
        The primitive this drives, and where the two-triangle patch is defined.
    [`stitch_loops`][ordito.holes.stitch_loops]
        Close two rims *completely* rather than tacking them together at one seam.
    [`fill_min_weight`][ordito.holes.fill_min_weight]
        What to run afterwards: the merged rim is left open on purpose.
    [`repair.remove_small_components`][ordito.repair.remove_small_components]
        The other answer to a multi-component mesh -- throw the extra pieces away instead.

    Notes
    -----
    The loop is host-sequential over the ``k - 1`` joins. The component labelling and the
    boundary edges are computed once; each join then edits them as its patch dictates -- the two
    bridged rim edges close, the patch's two chords open, the two components merge -- which is
    exactly what recomputing them from the grown face buffer would give, so each round costs one
    pass over the boundary vertices rather than over the mesh. The pairing itself is on the device.

    A component with no boundary -- a closed shell -- has nothing to bridge to and is left alone, so
    an input of closed shells comes back unchanged rather than raising.
    """
    require_same_device(vertices=vertices, faces=faces)
    if max_joins is not None and max_joins < 0:
        raise ValueError(f"max_joins must be non-negative, got {max_joins}")
    max_distance_sq = wp.float32(float("inf") if max_distance is None else float(max_distance) ** 2)

    rim = _JoinRim.of(vertices, faces)
    patch: list[tuple[int, int, int]] = []
    joins = 0
    while rim is not None and (max_joins is None or joins < max_joins):
        pair = rim.closest_cross_component_edges(max_distance_sq)
        if pair is None:
            break
        # ``bridge_edges(..., validate=False)``'s patch: both checks ``validate`` runs hold by
        # construction, since the edges are rows of the current oriented boundary and lie in
        # different components, so no chord between them can already be an edge.
        triangles = _bridge_triangles(*pair)
        rim.bridge(triangles)
        patch.extend(triangles)
        joins += 1

    if joins == 0:
        return _launch.clone(faces)
    # Each join appended its patch to the face buffer the next one read; appending them all at
    # once is the same buffer.
    patch_wp = _launch.array(
        np.asarray(patch, dtype=np.int32).reshape(-1), dtype=wp.int32, device=faces.device
    )
    return od.array.concatenate([faces, patch_wp])


# ``pack_nearest_key`` is non-negative for any real candidate, so the largest ``int64`` is a seed no
# pair can reach -- which is what makes "no admissible pair" a value rather than a second flag.
_NEAREST_KEY_SEED = (1 << 63) - 1


class _JoinRim:
    """
    ``join_closest_components``' boundary and component labels, kept current across its joins.

    A join appends a two-triangle patch and changes nothing else, so instead of regrouping the
    whole face buffer each round -- labelling its components, sorting every halfedge for the
    oriented boundary -- this edits the boundary table the way the patch edits the boundary: the
    two bridged rim edges become interior, the patch's diagonal is interior, and its two chords are
    new boundary edges wound as the patch winds them. Every boundary vertex stays one (the chords'
    tails are the bridged edges' tails), and the two components the patch touches merge. The table
    is kept in the order ``boundary.oriented_boundary_edges`` emits it -- ascending undirected key,
    which packs ``(min, max)`` with the larger index as the high digit -- so the pairing kernel sees
    exactly the rows, the row order (its tie-break) and the label equalities a regroup would give.

    That holds while the boundary is edge-manifold around the joins: a chord between two components
    cannot already be an edge, and a bridged edge occurred once, so the patch's edges are all that
    change in the count of each key.
    """

    def __init__(
        self,
        vertices: wp.array[wp.vec3],
        rows: npt.NDArray[np.int64],
        labels: npt.NDArray[np.int64],
        vertex_labels: npt.NDArray[np.int64],
        n_labels: int,
    ) -> None:
        self._vertices = vertices
        self._radix = max(int(vertices.size), 1)
        self._tails = rows[:, 0].copy()
        self._heads = rows[:, 1].copy()
        self._keys = self._key(self._tails, self._heads)
        self._labels = labels
        self._vertex_labels = vertex_labels
        # The current component of each original one: equality is all the pairing reads.
        self._component = np.arange(n_labels, dtype=np.int64)
        # The pairing's device buffers, kept across rounds: a join removes two rim rows and adds
        # two chords, so the table's size does not change (rebuilt only if it ever does), and
        # ``closest_pair_rows`` re-arms the best key as it decodes it.
        self._buffers: (
            tuple[
                wp.array[wp.int32],
                odt.Array2dInt32,
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
            ]
            | None
        ) = None
        self._best: wp.array[wp.int64] | None = None

    @staticmethod
    def of(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> _JoinRim | None:
        """Build the rim of ``faces``, or return ``None`` when it has no boundary."""
        device = faces.device
        boundary = od.boundary.oriented_boundary_edges(vertices, faces)
        if int(boundary.shape[0]) == 0:
            return None
        n_faces = faces.size // 3
        face_labels = od.adjacency.face_connected_component_labels(faces, n_vertices=vertices.size)
        vertex_labels = _launch.full(vertices.size, -1, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_scatter.scatter_face_labels_to_vertices,
            dim=3 * n_faces,
            inputs=[faces, face_labels, vertex_labels],
            device=device,
        )
        # The two host reads of the whole loop: the rim and every vertex's component.
        rows = boundary.numpy().astype(np.int64)
        names, compact = np.unique(vertex_labels.numpy(), return_inverse=True)
        vertex_compact = compact.astype(np.int64)
        return _JoinRim(vertices, rows, vertex_compact[rows[:, 0]], vertex_compact, names.size)

    def _key(
        self, tails: npt.NDArray[np.int64], heads: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        return np.maximum(tails, heads) * self._radix + np.minimum(tails, heads)

    def closest_cross_component_edges(
        self, max_distance_sq: wp.float32
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """
        Pick the two oriented boundary edges to bridge next, or ``None`` when nothing is left.

        ``None`` covers every reason at once: one component, or no cross-component boundary-vertex
        pair within ``max_distance_sq``. The candidates are the rows' tails, so the winning rows
        name their own edges, wound as ``bridge_edges`` takes them. One upload of the table and one
        read of the five-integer answer per call.
        """
        device = self._vertices.device
        n = int(self._keys.size)
        table = np.empty(3 * n, dtype=np.int32)
        table[0 : 2 * n : 2] = self._tails
        table[1 : 2 * n : 2] = self._heads
        table[2 * n :] = self.row_components()
        if self._buffers is None or self._buffers[0].size != 3 * n:
            table_wp = _launch.empty(3 * n, dtype=wp.int32, device=device)
            rows = odt.as_array2d(odt.as_dense(table_wp[: 2 * n]).reshape((n, 2)), wp.int32)
            labels = odt.as_dense(table_wp[2 * n :])
            partner = _launch.empty(n, dtype=wp.int32, device=device)
            answer = _launch.empty(5, dtype=wp.int32, device=device)
            self._buffers = (table_wp, rows, labels, partner, answer)
            self._best = _launch.full(1, _NEAREST_KEY_SEED, dtype=wp.int64, device=device)
        table_wp, rows, labels, partner, answer = self._buffers
        best = self._best
        assert best is not None
        table_wp.assign(table)
        _launch.launch_tiled(
            kernel_holes.reduce_closest_cross_label_pair,
            dim=[n],
            inputs=[self._vertices, rows, labels, max_distance_sq, best, partner],
            block_dim=kernel_holes.CLOSEST_PAIR_BLOCK,
            device=device,
        )
        # The winner decoded on the device into ``[found, a0, a1, b0, b1]``: one readback.
        _launch.launch(
            kernel_holes.closest_pair_rows,
            dim=1,
            inputs=[best, partner, rows, wp.int64(_NEAREST_KEY_SEED), answer],
            device=device,
        )
        found, a0, a1, b0, b1 = (int(value) for value in read_values(answer, 0, 5))
        if not found:
            return None
        return ((a0, a1), (b0, b1))

    def rows(self) -> npt.NDArray[np.int64]:
        """Return the ``(n_boundary, 2)`` oriented boundary rows, in the table's order."""
        return np.stack([self._tails, self._heads], axis=1)

    def row_components(self) -> npt.NDArray[np.int64]:
        """Return each row's current component, as a name only equality may read."""
        return self._component[self._labels]

    def bridge(self, triangles: Sequence[tuple[int, int, int]]) -> None:
        """Apply a bridge patch: edit the rim as its edges dictate, and merge the components."""
        added: dict[int, tuple[int, int]] = {}
        removed: list[int] = []
        touched: list[int] = []
        for triangle in triangles:
            for k in range(3):
                u, v = triangle[k], triangle[(k + 1) % 3]
                key = max(u, v) * self._radix + min(u, v)
                slot = int(np.searchsorted(self._keys, key))
                if slot < self._keys.size and int(self._keys[slot]) == key:
                    removed.append(slot)  # a rim edge the patch closes
                elif key in added:
                    del added[key]  # the patch's own diagonal
                else:
                    added[key] = (u, v)
                touched.append(u)
        merged = sorted({int(c) for c in self._component[self._vertex_labels[touched]]})
        for component in merged[1:]:
            self._component[self._component == component] = merged[0]
        keep = np.ones(self._keys.size, dtype=bool)
        keep[removed] = False
        new_keys = np.fromiter(added.keys(), dtype=np.int64, count=len(added))
        new_rows = np.asarray(list(added.values()), dtype=np.int64).reshape(-1, 2)
        keys = np.concatenate([self._keys[keep], new_keys])
        tails = np.concatenate([self._tails[keep], new_rows[:, 0]])
        heads = np.concatenate([self._heads[keep], new_rows[:, 1]])
        labels = np.concatenate([self._labels[keep], self._vertex_labels[new_rows[:, 0]]])
        order = np.argsort(keys, kind="stable")
        self._keys, self._tails, self._heads, self._labels = (
            keys[order],
            tails[order],
            heads[order],
            labels[order],
        )


def _require_distinct_edges(edge_a: tuple[int, int], edge_b: tuple[int, int]) -> None:
    """Reject a bridge from an edge to itself, the one check ``validate=False`` keeps."""
    if tuple(edge_a) == tuple(edge_b):
        raise ValueError("edge_a and edge_b must be different edges")


def _bridge_census(faces: wp.array[wp.int32], pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """
    ``kernels/holes.bridge_edge_census`` over a handful of vertex pairs, read back as ``(q, 3)``.

    One launch over the faces and one readback of the ``3 q`` counts, where the rim table and the
    mesh's edge table this answers for are each a pass that scales with the mesh.
    """
    device = faces.device
    census = _launch.zeros((len(pairs), 3), dtype=wp.int32, device=device)
    n_faces = faces.size // 3
    if n_faces > 0:
        queries = _launch.array(np.asarray(pairs, dtype=np.int32), dtype=wp.int32, device=device)
        _launch.launch(
            kernel_holes.bridge_edge_census,
            dim=(n_faces, len(pairs)),
            inputs=[faces, queries, census],
            device=device,
        )
    return census.numpy()


def _require_rim_edges(
    edge_a: tuple[int, int], edge_b: tuple[int, int], census: np.ndarray
) -> None:
    """
    Reject a bridge edge that is not on the boundary wound as its face winds it.

    ``census`` is ``_bridge_census``' rows for ``edge_a`` and ``edge_b`` first. A row of
    [`oriented_boundary_edges`][ordito.boundary.oriented_boundary_edges] is a face edge whose
    undirected key occurs once, so on the rim both of its counts are exactly one.
    """
    for row, (name, edge) in enumerate((("edge_a", edge_a), ("edge_b", edge_b))):
        if int(census[row, 0]) != 1 or int(census[row, 1]) != 1:
            raise ValueError(
                f"{name}={edge} is not a boundary edge of faces, wound as its face winds it"
            )


def _require_new_chords(
    edge_a: tuple[int, int],
    edge_b: tuple[int, int],
    joined: Sequence[tuple[int, int]],
    census: np.ndarray,
) -> None:
    """
    Reject a flat patch whose new edges duplicate an existing one.

    ``joined`` is what the patch adds *between vertices that already exist*, and ``census`` its
    ``_bridge_census`` rows. The lookup is over the mesh's own edges rather than the rim's, since a
    chord across a thin neck is usually an interior edge.
    """
    for (u, v), row in zip(joined, census, strict=True):
        if int(row[0]) > 0:
            raise ValueError(
                f"bridging {tuple(edge_a)} to {tuple(edge_b)} would add a second edge between "
                f"{u} and {v}, leaving the mesh non-manifold; pick a different pair"
            )


def _bridge_triangles(
    edge_a: tuple[int, int], edge_b: tuple[int, int]
) -> list[tuple[int, int, int]]:
    """Build the one or two triangles of the flat patch, wound against both edges' own faces."""
    a0, a1 = edge_a
    b0, b1 = edge_b
    # The quadrilateral runs (a1, a0, b1, b0): each edge is traversed backwards from the way its own
    # face winds it, which is what makes the patch's outward side agree with the mesh's. It is split
    # along the a0-b0 diagonal, so a shared endpoint *on that diagonal* is the one case the split
    # cannot express: at a0 == b0 both halves fold onto a line and the patch comes out empty, and at
    # a1 == b1 the two halves are the same triangle wound opposite ways. Both are named here, and
    # the surviving triangle is the collapsed quad's own traversal. (Sharing both endpoints means
    # the two edges are one edge, which the caller rejects before reaching this.)
    if a0 == b0:
        return [(a1, a0, b1)]
    if a1 == b1:
        return [(a1, a0, b0)]
    # A *crossed* shared endpoint (a0 == b1 or a1 == b0) collapses exactly one half, which the
    # degeneracy filter removes correctly -- the survivor is already the right triangle.
    triangles = [(a1, a0, b0), (a0, b1, b0)]
    return [t for t in triangles if len(set(t)) == 3]


# ---------------------------------------------------------------------------
# Smooth-patch pipeline: min-weight fill / stitch -> region subdivision -> region smoothing.
# ---------------------------------------------------------------------------


def _patch_mask(
    n_faces_before: int, n_faces_after: int, device: wp.DeviceLike
) -> wp.array[wp.bool]:
    """
    Boolean face mask marking the trailing ``[n_faces_before, n_faces_after)`` fill faces.

    The eight ``return (*result, mask) if return_patch else result`` lines that consume this are
    **deliberately not** folded into a helper. Each is already one line, so a
    ``_finish_patch(vertices, faces, mask, return_patch)`` call would replace one line with one
    line and add an indirection the reader has to follow to learn it is a conditional tuple.
    Section 2.4 asks for merging on identity of *meaning*; these agree because "return two arrays,
    optionally with a mask" has only one shape, which is the noise case that rule names rather than
    a duplicated decision rule.
    """
    mask = _launch.zeros(n_faces_after, dtype=wp.bool, device=device)
    # Warp rejects a zero-length slice, and the "nothing was filled" caller passes an empty range.
    if n_faces_after > n_faces_before:
        _launch.fill_(mask[n_faces_before:], True)
    return mask


def _mean_rim_edge_length(
    vertices: wp.array[wp.vec3], loops: _PackedLoops | list[wp.array[wp.int32]]
) -> float:
    """
    Mean edge length over the rims of the given loops (the derived subdivision target).

    Every rim is closed, so its edge count is its vertex count and the mean is the total perimeter
    over ``sum(sizes)`` -- which makes this one
    [`_PackedLoops.perimeters`][ordito.holes._PackedLoops.perimeters] launch plus its one
    readback, the same measurement [`fill_small`][ordito.holes.fill_small] already pays. A list
    is accepted because
    [`ordito.holes.stitch_smooth`][ordito.holes.stitch_smooth] holds its two rims
    individually; it is packed here rather than at that call site so the private surface stays one
    name wide.
    """
    packed = loops if isinstance(loops, _PackedLoops) else _pack_loops(loops)
    count = int(packed.sizes_np.sum())
    if count == 0:
        return 0.0
    return float(packed.perimeters(vertices).sum()) / count


def _hole_loops(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], preserve_largest_hole: bool = False
) -> _PackedLoops | None:
    """
    Pack the fillable boundary loops (>= 3 vertices) of a mesh for on-device triangulation.

    Returns ``None`` when there is no fillable boundary loop. Costs **one** host readback (the loop
    offsets, which the ragged indexing needs anyway), plus a second one only under
    ``preserve_largest_hole``.

    That second readback stays on the host on purpose, and not because the argmax could not run on
    the device: what the host needs is the resulting *mask*, to build the ragged gather index that
    compacts the surviving loops. Reducing on the device would still have to bring the winner back,
    so it would add a launch and remove nothing.

    When ``preserve_largest_hole`` is ``True`` the single largest loop — the one with the greatest
    perimeter arc length; the first one on a tie — is excluded, leaving it open. This is the
    standard cut for disk-topology repair and UV parametrization, where exactly one boundary must
    survive. The perimeters come from one segmented-sum launch over the packed loops rather than a
    [`polyline_length`][ordito.polyline.polyline_length] call per loop (two synchronizations
    each).
    """
    packed = _boundary_loops_packed(vertices, faces)
    if packed is None:
        return None

    flat_loops, sizes_np = packed
    keep_np = sizes_np >= 3
    if preserve_largest_hole and bool(keep_np.any()):
        all_loops = _PackedLoops(flat_loops, sizes_np)
        perimeter_np = all_loops.perimeters(vertices)
        # Only a fillable loop can be the one preserved, matching the pre-filter order.
        candidates_np = np.flatnonzero(keep_np)
        keep_np[candidates_np[int(np.argmax(perimeter_np[candidates_np]))]] = False
    if not keep_np.any():
        return None
    if keep_np.all():
        return _PackedLoops(flat_loops, sizes_np)
    return _compact_packed_loops(flat_loops, sizes_np, keep_np)


def _compact_packed_loops(
    flat_loops: wp.array[wp.int32], sizes_np: np.ndarray, keep_np: np.ndarray
) -> _PackedLoops:
    """
    Repack only the loops ``keep_np`` selects, as one gather over their slots.

    Not one copy per surviving loop, and not a split into per-loop views followed by a re-pack of
    them: both of those cost a host call per loop, where this is one upload and one gather whatever
    the loop count.
    """
    keep_index_np = np.flatnonzero(np.repeat(keep_np, sizes_np))
    keep_index_wp = _launch.array(
        keep_index_np.astype(np.int32), dtype=wp.int32, device=flat_loops.device
    )
    return _PackedLoops(od.array.gather(flat_loops, keep_index_wp), sizes_np[keep_np])


def _boundary_loops_packed(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.int32], np.ndarray] | None:
    """
    Every boundary loop of a mesh as one packed buffer plus its host sizes, or ``None`` if none.

    The packed buffer is
    [`boundary_loops_with_offsets`][ordito.boundary.boundary_loops_with_offsets]' own, handed on
    unsplit; the sizes come from the single offsets readback the ragged indexing needs anyway.
    """
    flat_loops, offsets = od.boundary.boundary_loops_with_offsets(vertices, faces)
    if offsets.size == 1:
        return None
    return flat_loops, np.diff(offsets.numpy().astype(np.int64))


def _packed_loop_argument(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: Sequence[wp.array[wp.int32]] | None,
) -> _PackedLoops | None:
    """
    Pack a public ``loops=`` argument, or the mesh's own boundary loops when it is ``None``.

    ``None`` goes through [`_boundary_loops_packed`][ordito.holes._boundary_loops_packed] rather
    than [`boundary_loops`][ordito.boundary.boundary_loops]: the latter splits that same buffer
    into one ``wp.array`` view per loop only for this to concatenate them straight back, and both
    halves cost a host call per loop -- so the round trip scales with the *loop count*, the one
    axis a packed engine exists to make free.

    Returns ``None`` when there is no loop at all. A caller that also rejects all-empty loops
    tests ``total`` itself, since an empty loop is still an entry in a per-loop answer.
    """
    if loops is None:
        packed = _boundary_loops_packed(vertices, faces)
        return None if packed is None else _PackedLoops(*packed)
    loops = list(loops)
    for loop in loops:
        odt.ensure_ndim(loop, 1, dtype=wp.int32)
    return _pack_loops(loops) if loops else None
