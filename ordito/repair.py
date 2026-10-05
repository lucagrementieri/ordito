"""
Removing what should not be in a mesh: redundant elements, bad triangles and inconsistent winding.

[`make_solid`][ordito.repair.make_solid] is the composite most callers want -- a broken digitised
surface in, a single watertight solid out -- and everything below is a stage of it that is also
useful on its own.

Five defects are visible in the index buffer alone, and each has a remover:
[`remove_unreferenced_vertices`][ordito.repair.remove_unreferenced_vertices],
[`remove_duplicated_vertices`][ordito.repair.remove_duplicated_vertices],
[`resolve_duplicated_faces`][ordito.repair.resolve_duplicated_faces],
[`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces], and
[`collapse_small_triangles`][ordito.repair.collapse_small_triangles].

A sixth remover answers a different question -- not *which elements are wrong* but *which parts of
the mesh are not the mesh*: [`remove_small_components`][ordito.repair.remove_small_components]
drops face-connected components that are too small, by face count, area or bounding-box diameter. It
is the first step of every repair pipeline, and the debris it removes is not defective in itself.

Two further defects need geometry rather than topology to detect, so they are found by a threshold
rather than a rule: [`validation.face_defective_mask`][ordito.validation.face_defective_mask] flags
faces that are too thin, misoriented against their neighbourhood, or folded back over it.
[`remove_folded_faces`][ordito.repair.remove_folded_faces] deletes the folded ones and
[`flip_t_vertices`][ordito.repair.flip_t_vertices] flips away the slivers a T-junction leaves
behind.

The verb predicts the return shape, and that is a rule rather than a coincidence:

- **``remove_*`` / ``collapse_*``** change the element count, so they return ``(vertices, faces)``
  or more -- there is a new position buffer because vertices went away or moved.
- **``make_*``** preserve positions and counts and rewrite only the index buffer, so they return
  ``faces`` alone -- *unless* the name is a whole-mesh **outcome** rather than a property of the
  index buffer, which is [`make_solid`][ordito.repair.make_solid] alone. It is the composite of
  most of this module and returns ``(vertices, faces)`` like the removers it runs; the three
  property-fixers beside it ([`make_winding_consistent`][ordito.repair.make_winding_consistent],
  [`make_volume`][ordito.repair.make_volume],
  [`make_normals_outward`][ordito.repair.make_normals_outward]) return ``faces``.
- **``reverse_winding``** is the one verb outside that scheme and obeys the same shape rule for the
  same reason: it rewrites only the index buffer, so it returns ``faces``. It is not a ``make_*``
  because it establishes no property -- it flips orientation unconditionally, where the three
  ``make_*`` fixers decide face by face.
- **A verb that only *moves* vertices** returns the positions alone, since neither buffer of indices
  changes: [`flatten_degree3_vertices`][ordito.repair.flatten_degree3_vertices] is the only one,
  and it is here rather than in [`ordito.smoothing`][ordito.smoothing] -- whose every member has
  that same signature -- because it is the gentler half of a pair with
  [`remove_degree3_vertices`][ordito.repair.remove_degree3_vertices]: same defect, same test, one
  answering it by deleting the vertex and one by flattening the bump.
- **``*_mask``** are detectors: they return a ``wp.array[wp.bool]`` and mutate nothing; they live in
  [`ordito.validation`][ordito.validation].

[`flip_t_vertices`][ordito.repair.flip_t_vertices] is named for the ``make_*`` pattern rather than
the ``remove_*`` one because it removes nothing: it flips the long edge of each sliver a T-junction
leaves, so the face count is unchanged and only ``faces`` comes back.
[`straighten_boundary`][ordito.repair.straighten_boundary] is the same shape under a geometric
verb.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, cast, overload

import numpy as np
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device, require_valid_faces
from ordito.constants import INDEX_RADIX_PAIR, TILE_1D
from ordito.grouping import hash_vector_rows, unique_1d, unique_faces
from ordito.kernels import array as kernel_array
from ordito.kernels import polyline as kernel_polyline
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import repair as kernel_repair
from ordito.kernels import scatter as kernel_scatter
from ordito.kernels import selection as kernel_selection

# Lattice resolution for ``fix_self_intersections(method="voxel")``, in samples across the mesh's
# bounding-box diagonal. The same order as ``offset.offset_mesh``'s automatic floor, and the field
# it allocates is cubic in it; a caller who needs the surface resolved finer passes ``voxel_size``.
_VOXEL_REBUILD_RESOLUTION = 128


def make_solid(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    keep_largest: bool = True,
    join_components: bool = False,
    max_iter: int = 10,
    inner_iter: int = 3,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Turn a broken digitised surface into a single watertight solid.

    The composite this module's pieces exist to make possible: debris removed, holes closed,
    degeneracies collapsed and self-intersections cut out and refilled, alternating until nothing is
    left to fix. Every stage is a public function here or in [`ordito.holes`][ordito.holes]; what
    this adds is the order and the loop, which is where the difficulty actually is -- closing a hole
    can create a self-intersection, and cutting one out reopens a hole.

    The stages, in order:

    0. The connectivity repair that does not look like a stage at all until it is missing:
       [`remove_unreferenced_vertices`][ordito.repair.remove_unreferenced_vertices],
       [`make_winding_consistent`][ordito.repair.make_winding_consistent] and
       [`split_non_manifold_vertices`][ordito.repair.split_non_manifold_vertices]. Without it a
       mesh whose defect is a non-manifold *edge* comes back with the right Euler characteristic and
       one component and is still not watertight, because no later stage looks at edge manifoldness.
    1. ``keep_largest`` -> [`remove_small_components`][ordito.repair.remove_small_components], so
      the
       scan debris goes before anything expensive runs on it.
    2. ``join_components`` ->
      [`holes.join_closest_components`][ordito.holes.join_closest_components],
       for an input whose pieces are meant to be one surface rather than a largest piece plus
       rubbish. Mutually useful with ``keep_largest`` rather than exclusive: keep the big piece
       *and* weld what is left.
    3. [`holes.fill_min_weight`][ordito.holes.fill_min_weight] if any boundary remains.
    4. Up to ``max_iter`` rounds of
       [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces] and
       [`collapse_small_triangles`][ordito.repair.collapse_small_triangles], then
       [`fix_self_intersections`][ordito.repair.fix_self_intersections] at ``max_iter=inner_iter``.
       The loop exits as soon as a round changes nothing.
    5. A final fill if stage 4 reopened a boundary -- which it routinely does, since cutting an
       intersecting region out is what opens one. Nothing geometric runs after it, and that is not
       an omission: filling a 3-vertex rim produces one sliver, a degeneracy pass deletes the sliver
       and reopens the rim, and the two trade the same faces indefinitely.

    Under ``keep_largest`` the component filter runs **inside** stage 4 as well as at the top, and
    that is not belt and braces: cutting an intersecting band out can disconnect the surface, so the
    extra piece does not exist yet when stage 1 looks.

    !!! warning "It returns the best it managed, not a guarantee"
        There is no success flag, deliberately. Convergence is not guaranteed for any input -- a
        patch can intersect something, and its repair can open another hole -- so on a stubborn mesh
        this returns a *partly* repaired surface rather than looping harder or raising. Ask
        [`validation.is_watertight`][ordito.validation.is_watertight] if the answer matters.

        ``join_components`` is where that bites in practice: welding several shells leaves **one**
        rim spanning all of them, and if that rim is badly non-planar the minimum-weight patch
        across it self-intersects, so stage 4 cuts the patch out and undoes the join. Hemispherical
        bowls joined at coplanar rims converge to one solid; the same bowls tilted relative to each
        other come back as separate shells. Rims far from coplanar want
        [`holes.stitch_loops`][ordito.holes.stitch_loops] or a per-pair
        [`holes.bridge_edges`][ordito.holes.bridge_edges] followed by a targeted fill, not this.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    keep_largest
        Drop every connected component but the one with the most faces, first.
    join_components
        Bridge the remaining open components together instead of leaving them separate.
    max_iter
        Cap on the alternating degeneracy / self-intersection rounds.
    inner_iter
        Cap on the cut-and-refill passes inside each self-intersection repair.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_new_vertices,)`` positions of the repaired mesh, on ``vertices.device``.
    new_faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat face buffer of the repaired mesh.

    Raises
    ------
    ValueError
        If ``max_iter`` or ``inner_iter`` is negative, or if a face references a vertex index
        ``vertices`` does not cover.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`validation.is_watertight`][ordito.validation.is_watertight]
        The question this does not answer for you.
    [`remove_small_components`][ordito.repair.remove_small_components]
    [`fix_self_intersections`][ordito.repair.fix_self_intersections]
    [`holes.fill_min_weight`][ordito.holes.fill_min_weight]
    [`make_normals_outward`][ordito.repair.make_normals_outward]
        What to run afterwards if the winding has to face outward as well as be consistent.
    """
    require_same_device(vertices=vertices, faces=faces)
    if max_iter < 0 or inner_iter < 0:
        raise ValueError(
            f"max_iter and inner_iter must be non-negative, got {max_iter}, {inner_iter}"
        )

    if faces.size == 0:
        return vertices, faces

    # This is the trust boundary for "a broken digitised surface": every stage below indexes
    # ``vertices[faces]`` without a bound check, the same way every other per-face kernel wrapper in
    # this package does, so an out-of-range index arriving here would otherwise reach the first one
    # silently (§12.1's memory-safety class) rather than raising a Python exception.
    require_valid_faces(faces, vertices.size, "make_solid")

    # Stage 0. The reference does this inside its *loader*, which is why it is easy to leave out and
    # why leaving it out is dangerous: skipping it can leave a mesh at the right Euler
    # characteristic and one component while still **not watertight**, because nothing downstream
    # addresses a non-manifold edge.
    vertices, faces, _remap = remove_unreferenced_vertices(vertices, faces)
    faces = make_winding_consistent(faces, n_vertices=vertices.size)
    vertices, faces, _source = split_non_manifold_vertices(vertices, faces)

    if keep_largest:
        vertices, faces = remove_small_components(vertices, faces, keep_largest=True)
    if join_components:
        faces = od.holes.join_closest_components(vertices, faces)

    faces = _fill_any_boundary(vertices, faces)
    for _ in range(max_iter):
        n_faces_before = faces.size
        vertices, faces = remove_degenerate_faces(vertices, faces)
        vertices, faces = collapse_small_triangles(vertices, faces)
        vertices, faces = fix_self_intersections(vertices, faces, max_iter=inner_iter)
        # Cutting an intersecting band out can *disconnect* the surface, so the component filter has
        # to run again here and not only at the top: a single pass at the start cannot see a
        # component that did not exist yet.
        if keep_largest:
            vertices, faces = remove_small_components(vertices, faces, keep_largest=True)
        if faces.size == n_faces_before:
            break

    # Stage 4 opens a rim whenever it cuts an intersecting region out, so the last fill is not a
    # belt-and-braces repeat of stage 3 -- it is what makes the common case come back closed. And
    # nothing geometric may run *after* it: filling a small rim produces one sliver, degeneracy
    # removal deletes that sliver and reopens the rim, and the two then trade the same faces
    # forever -- so the fill goes last.
    faces = _fill_any_boundary(vertices, faces)
    if keep_largest:
        vertices, faces = remove_small_components(vertices, faces, keep_largest=True)
    return vertices, faces


def _fill_any_boundary(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.int32]:
    """Close every boundary loop, or hand the buffer back untouched when there is none."""
    if int(od.boundary.boundary_edges(vertices, faces).shape[0]) == 0:
        return faces
    return od.holes.fill_min_weight(vertices, faces)


@overload
def remove_unreferenced_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    return_inverse: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
@overload
def remove_unreferenced_vertices(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, return_inverse: Literal[True]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]: ...
def remove_unreferenced_vertices(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, return_inverse: bool = False
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Remove vertices not referenced by any face and remap face indices.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    return_inverse
        If ``True``, also return ``inverse`` with ``new_vertices[inverse]`` sourcing
        ``vertices``.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_referenced,)`` referenced vertices only, compacted from index zero.
    new_faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` face buffer with indices remapped into ``new_vertices``.
    remap : wp.array[wp.int32]
        ``(n_vertices,)`` old-to-new map (``-1`` when unreferenced).
    inverse : wp.array[wp.int32], optional
        ``(n_referenced,)`` source index of each new vertex. Present when
        ``return_inverse=True``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    n_vertices = vertices.size

    # The referenced flags are marked as ``int32`` and scanned in place, and one pass reads the scan
    # to write all three maps and the remapped faces: the tail of the scan is the referenced count,
    # which sizes the outputs and is the one readback.
    remap = _launch.empty(n_vertices, dtype=wp.int32, device=device)
    if n_vertices == 0:
        new_vertices = _launch.empty(0, dtype=wp.vec3, device=device)
        inverse = _launch.empty(0, dtype=wp.int32, device=device)
        new_faces = od.array.remap_indices(faces, remap)
    else:
        inclusive = _launch.zeros(n_vertices, dtype=wp.int32, device=device)
        n_indices = faces.size
        if n_indices > 0:
            _launch.launch(
                kernel_repair.mark_referenced,
                dim=n_indices,
                inputs=[faces, inclusive],
                device=device,
            )
        _launch.array_scan(inclusive, out_array=inclusive, inclusive=True)
        n_referenced = int(read_scalar(inclusive))
        new_vertices = _launch.empty(n_referenced, dtype=wp.vec3, device=device)
        inverse = _launch.empty(n_referenced, dtype=wp.int32, device=device)
        new_faces = _launch.empty(n_indices, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_repair.compact_referenced,
            dim=max(n_vertices, n_indices),
            inputs=[vertices, faces, inclusive, remap, new_vertices, inverse, new_faces],
            device=device,
        )

    if return_inverse:
        return new_vertices, new_faces, remap, inverse
    return new_vertices, new_faces, remap


def remove_duplicated_vertices(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], epsilon: float = 0.0
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Merge duplicate vertex positions up to a coordinate tolerance and remap face indices.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    epsilon
        Uniqueness tolerance. Positive values snap coordinates to ``round(v / epsilon)``, so the
        tolerance is absolute and is the one you chose. ``0`` instead groups by a *relative*
        bucket about ``2.4e-4`` wide (see Notes) — pass an explicit ``epsilon`` unless that is
        what you want.

    Returns
    -------
    unique_vertices : wp.array[wp.vec3]
        ``(n_unique,)`` deduplicated vertex positions (first occurrence per equivalence class).
    unique_indices : wp.array[wp.int32]
        ``(n_unique,)`` original indices into ``vertices`` for each output row.
    inverse : wp.array[wp.int32]
        ``(n_vertices,)`` map from each input vertex to its slot in ``unique_vertices``.
    unique_faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` face buffer with indices remapped into ``unique_vertices``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    Both tolerance modes quantize positions and group by cell, so both separate a pair straddling a
    cell boundary however close it is. That is worth knowing about ``epsilon=0`` in particular,
    which is **not** an exact-equality test despite requiring no tolerance: it buckets by the high
    bits of each coordinate's ``float32`` representation, giving a *relative* cell about ``2.4e-4``
    wide. So ``1.0`` and ``1.000244`` merge, while two adjacent ``float32`` values on either side of
    a bucket edge do not. Prefer an explicit ``epsilon`` whenever the tolerance matters; use ``0``
    only to collapse positions that are already bitwise equal, which it does reliably (including
    across ``+0.0`` / ``-0.0``).

    Duplicates that are known from construction rather than measured are better collapsed directly:
    see [`revolve`][ordito.creation.revolve], which derives them from its profile instead of
    hashing positions.

    See Also
    --------
    [`duplicate_vertex_inverse`][ordito.repair.duplicate_vertex_inverse]
    [`hash_vector_rows`][ordito.grouping.hash_vector_rows]
    [`ordito.seams.cut_along_edges`][ordito.seams.cut_along_edges]
        The inverse operation: welding coincident positions back together closes every cut it makes.
    """
    require_same_device(vertices=vertices, faces=faces)
    # The class count comes with the inverse, so the representatives need no reduction to size them.
    n_unique, inverse = _vertex_classes(vertices, epsilon)
    unique_indices = od.grouping.first_occurrence_indices(inverse, n_unique)
    unique_vertices = od.array.gather(vertices, unique_indices)
    unique_faces = od.array.remap_indices(faces, inverse)
    return unique_vertices, unique_indices, inverse, unique_faces


def duplicate_vertex_inverse(vertices: wp.array[wp.vec3], epsilon: float) -> wp.array[wp.int32]:
    """
    Map each vertex to the slot of its coincident-vertex equivalence class.

    The inverse map produced by welding vertices at ``epsilon`` tolerance, without also
    computing the deduplicated vertex/face buffers — useful for remapping per-vertex
    attributes (colors, UVs, ...) to match a [`remove_duplicated_vertices`]
    [ordito.repair.remove_duplicated_vertices] call made with the same ``epsilon``.

    This is the shared equivalence map the dedup remaps by, not a remover -- which is why it lives
    here rather than in [`ordito.grouping`][ordito.grouping] beside
    [`first_occurrence_indices`][ordito.grouping.first_occurrence_indices], whose shape it has. Its
    key is a *tolerance-quantized position*, and choosing that tolerance is mesh-repair policy;
    ``grouping`` is deliberately dtype-generic and geometry-free, so an ``epsilon`` there would be
    the first crack in that contract.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    epsilon
        Uniqueness tolerance, with the same meaning as in
        [`remove_duplicated_vertices`][ordito.repair.remove_duplicated_vertices]: positive values
        snap coordinates to ``round(v / epsilon)``, while ``0`` groups by a *relative* bucket about
        ``2.4e-4`` wide rather than testing for equality.

    Returns
    -------
    wp.array[wp.int32]
        ``(n_vertices,)`` map from each input vertex to its slot in the deduplicated set.

    See Also
    --------
    [`remove_duplicated_vertices`][ordito.repair.remove_duplicated_vertices]
        The remover this map is the shared half of.
    [`hash_vector_rows`][ordito.grouping.hash_vector_rows]
    [`grouping.first_occurrence_indices`][ordito.grouping.first_occurrence_indices]
        The geometry-free counterpart: the same class-to-representative reduction over any key.
    """
    return _vertex_classes(vertices, epsilon)[1]


def _vertex_classes(vertices: wp.array[wp.vec3], epsilon: float) -> tuple[int, wp.array[wp.int32]]:
    """Return the coincident-vertex class count and each vertex's class, as ``unique_1d`` does."""
    device = vertices.device
    if vertices.size == 0:
        return 0, _launch.empty(0, dtype=wp.int32, device=device)
    # One key per vertex and one ``unique_1d`` over it, at either tolerance: ``hash_vector_rows``
    # quantizes at ``epsilon > 0`` and packs the relative float buckets at ``0`` -- the same key
    # ``unique_rows`` would hash the vertices to, without the representative gather it would then
    # compute.
    unique, inverse = unique_1d(hash_vector_rows(vertices, epsilon=epsilon), return_inverse=True)
    return unique.size, inverse


def resolve_duplicated_faces(
    faces: wp.array[wp.int32],
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Resolve duplicated triangles by orientation-aware cancellation rules.

    For each geometric duplicate:
    - equal positive and negative counts: remove all copies;
    - one extra positive copy: keep one positively oriented face;
    - one extra negative copy: keep one negatively oriented face;
    - otherwise raise ``ValueError`` when counts are not orientable.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    resolved_faces : wp.array[wp.int32]
        ``(3 * n_kept,)`` flat buffer of kept faces.
    kept_indices : wp.array[wp.int32]
        ``(n_kept,)`` original face indices into the input ``faces`` buffer.

    Raises
    ------
    ValueError
        If a duplicate group's signed count is not orientable, i.e. its positive and negative
        copies differ by more than one.
    """
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        empty = _launch.empty(0, dtype=wp.int32, device=device)
        return empty, empty

    faces2d = faces.reshape((-1, 3))
    unique_faces_wp, inverse = unique_faces(faces, return_inverse=True)
    num_unique = unique_faces_wp.size // 3

    # Per-group orientation stats scattered on device: member/signed counts plus the smallest
    # member index of each sign class (seeded with the ``n_faces`` sentinel).
    member_count = _launch.zeros(num_unique, dtype=wp.int32, device=device)
    signed_count = _launch.zeros(num_unique, dtype=wp.int32, device=device)
    first_member = _launch.full(num_unique, n_faces, dtype=wp.int32, device=device)
    first_positive = _launch.full(num_unique, n_faces, dtype=wp.int32, device=device)
    first_negative = _launch.full(num_unique, n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.scatter_duplicate_face_stats,
        dim=n_faces,
        inputs=[
            faces,
            unique_faces_wp,
            inverse,
            member_count,
            signed_count,
            first_member,
            first_positive,
            first_negative,
        ],
        device=device,
    )

    keep = _launch.empty(num_unique, dtype=wp.int32, device=device)
    keep_mask = _launch.empty(num_unique, dtype=wp.bool, device=device)
    error_group = _launch.full(1, num_unique, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.resolve_duplicate_groups,
        dim=num_unique,
        inputs=[
            member_count,
            signed_count,
            first_member,
            first_positive,
            first_negative,
            keep,
            keep_mask,
            error_group,
        ],
        device=device,
    )
    first_error = int(read_scalar(error_group, 0))
    if first_error < num_unique:
        count = int(read_scalar(signed_count, first_error))
        raise ValueError(
            f"resolve_duplicated_faces: non-orientable duplicate face group {first_error} "
            f"with signed count {count}"
        )

    # Compact kept decisions in ascending group order (matches the reference emission order).
    kept_slots = od.array.flatnonzero(keep_mask)
    if kept_slots.size == 0:
        empty = _launch.empty(0, dtype=wp.int32, device=device)
        return empty, empty

    kept_wp = od.array.gather(keep, kept_slots)
    resolved = od.array.gather(faces2d, kept_wp).reshape((-1,))
    return resolved, kept_wp


def remove_degenerate_faces(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Drop degenerate (zero-area) triangles and reindex, keeping vertex positions unchanged.

    Mirrors ``trimesh.Trimesh.nondegenerate_faces`` + ``update_faces``: a face is degenerate when
    two of its vertices coincide or its three vertices are collinear, detected by
    [`face_nondegenerate_mask`][ordito.triangles.face_nondegenerate_mask] (both triangle altitudes
    exceed the merge tolerance). Surviving faces are unchanged; vertices left unreferenced after the
    drop are removed by the reindexing in
    [`submesh_from_face_mask`][ordito.selection.submesh_from_face_mask].

    Unlike [`collapse_small_triangles`][ordito.repair.collapse_small_triangles], no vertices are
    merged and no edges are collapsed: this only removes faces already degenerate in the input.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_new_vertices,)`` vertices still referenced by a non-degenerate face, compacted from
        index zero.
    new_faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat buffer of the non-degenerate faces, remapped into
        ``new_vertices``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`collapse_small_triangles`][ordito.repair.collapse_small_triangles]
    [`face_nondegenerate_mask`][ordito.triangles.face_nondegenerate_mask]
    [`trimesh.triangles.nondegenerate`][]
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.clone(vertices), _launch.clone(faces)

    keep_mask = od.triangles.face_nondegenerate_mask(vertices, faces)
    return od.selection.submesh_from_face_mask(vertices, faces, keep_mask)


def remove_non_manifold_faces(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], max_iter: int = 3
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Remove faces touching a non-manifold (>2-incident) edge, iterating until edge-manifold.

    Each pass keeps only faces whose three edges are each used by at most two faces
    ([`edge_manifold_mask`][ordito.validation.edge_manifold_mask]); dropping a face can make a
    neighbour manifold, so it repeats up to ``max_iter`` times.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    max_iter
        Maximum number of removal passes.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_new_vertices,)`` vertices still referenced after non-manifold faces are dropped,
        compacted from index zero.
    new_faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat buffer of the surviving (edge-manifold, up to ``max_iter``
        passes) faces.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]
    [`remove_degenerate_and_non_manifold_faces`][ordito.repair.remove_degenerate_and_non_manifold_faces]
    [`edge_manifold_mask`][ordito.validation.edge_manifold_mask]
    """
    require_same_device(vertices=vertices, faces=faces)
    # The radix of the edge keys; any bound above every index groups the edges identically, so the
    # one taken from the input serves every pass over a subset of its faces.
    n_vertices = od.array.index_bound(faces, require_non_negative=True)
    survivors, keep = _edge_manifold_survivors(faces, None, n_vertices, max_iter)
    if keep is None:
        return vertices, faces
    return od.selection.submesh_from_face_mask(vertices, survivors, keep)


def remove_degenerate_and_non_manifold_faces(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], max_iter: int = 3
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Drop degenerate faces, then faces on a non-manifold edge, compacting the vertices once.

    The same answer as [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]
    followed by [`remove_non_manifold_faces`][ordito.repair.remove_non_manifold_faces], byte for
    byte. The manifold test reads only face indices and every compaction preserves the order of
    both the faces and the vertices, so the passes run over the input's own vertex numbering and
    the referenced vertices are compacted once, at the end, instead of after every pass.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer, every index in
        ``[0, n_vertices)``.
    max_iter
        Maximum number of non-manifold removal passes.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_new_vertices,)`` vertices still referenced by a surviving face, compacted from index
        zero.
    new_faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat buffer of the surviving faces, remapped into
        ``new_vertices``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]
    [`remove_non_manifold_faces`][ordito.repair.remove_non_manifold_faces]
    """
    require_same_device(vertices=vertices, faces=faces)
    if faces.size == 0:
        return _launch.clone(vertices), _launch.clone(faces)
    keep = od.triangles.face_nondegenerate_mask(vertices, faces)
    survivors, keep = _edge_manifold_survivors(faces, keep, vertices.size, max_iter)
    assert keep is not None
    return od.selection.submesh_from_face_mask(vertices, survivors, keep)


def _edge_manifold_survivors(
    faces: wp.array[wp.int32], keep: wp.array[wp.bool] | None, n_vertices: int, max_iter: int
) -> tuple[wp.array[wp.int32], wp.array[wp.bool] | None]:
    """
    Run up to ``max_iter`` edge-manifold removal passes over ``faces[keep]``, in index space.

    The surviving faces are always ``faces[keep]`` for the pair this returns (all of ``faces``
    when ``keep`` is ``None``), with ``keep`` ``None`` only when it came in ``None`` and no pass
    removed a face. Vertex indices are never remapped, so a caller compacts once with
    [`submesh_from_face_mask`][ordito.selection.submesh_from_face_mask]. A pass stops the loop
    when it keeps every face or when no face is left. The pair returned is the last removing pass's
    input and mask, so the final removal is never gathered out and the compaction reads the mask
    that pass already wrote.

    ``n_vertices`` is the edge-key radix: any bound above every index of ``faces``.
    """
    tested = faces
    if keep is not None:
        kept = od.array.flatnonzero(keep)
        # Readback: the kept count decides whether the prefilter dropped anything to gather out.
        if kept.size != faces.size // 3:
            tested = od.array.gather(faces.reshape((-1, 3)), kept).reshape((-1,))
    pending = None
    for _ in range(max_iter):
        # The previous pass's removal is applied only once another pass is going to test it.
        if pending is not None:
            tested = od.array.gather(tested.reshape((-1, 3)), pending).reshape((-1,))
        n_faces = tested.size // 3
        if n_faces == 0:
            break
        manifold = od.validation.edge_manifold_mask(
            tested, allow_boundary_edges=True, n_vertices=n_vertices, validate=False
        )
        pending = od.array.flatnonzero(manifold)
        # Readback: the kept count is the stopping test.
        if pending.size == n_faces:
            break  # already edge-manifold
        faces, keep = tested, manifold
    return faces, keep


def remove_small_components(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    keep_largest: bool = False,
    min_faces: int | None = None,
    min_area: float | None = None,
    min_diameter: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Drop face-connected components that are too small, by one of four measures of "small".

    Debris -- a stray shell from a scan, a sliver left by a boolean, a component a decimation
    stranded -- is what every repair pipeline removes first, and until now
    [`combine.split`][ordito.combine.split] handed back *every* component and left the caller to
    write the argmax. This is that step, and it stays on the device: the per-component statistic is
    accumulated by one scatter and thresholded by one kernel, so nothing is read back and no
    component is ever materialized as its own mesh.

    Exactly one criterion may be given, because they are four different questions rather than four
    spellings of one and combining them would hide which one rejected a component:

    - ``keep_largest`` keeps the single component with the **most faces** and drops every other, so
      the result always has exactly one component. Face count, not area or diameter: a small dense
      shell outranks a large coarse one. A tie goes to the component whose lowest face index is
      smallest, which makes the choice deterministic rather than dependent on thread order.
    - ``min_faces`` keeps components with at least that many faces -- scale-free, and the measure
      that tracks *how much data* a component carries.
    - ``min_area`` keeps components whose summed triangle area reaches it.
    - ``min_diameter`` keeps components whose axis-aligned bounding-box diagonal reaches it, which
      is the measure that survives a component being a thin sheet of many tiny triangles.

    All three ``min_*`` bounds are **inclusive**, matching the reference implementations: a
    component of exactly the threshold face count, or of exactly the threshold diagonal, survives.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    keep_largest
        Keep only the component with the most faces.
    min_faces
        Minimum face count for a component to be kept.
    min_area
        Minimum summed triangle area for a component to be kept.
    min_diameter
        Minimum bounding-box diagonal for a component to be kept.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_new_vertices,)`` vertices of the surviving components, compacted from index zero.
    new_faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat buffer of the surviving faces, in their input order.

    Raises
    ------
    ValueError
        If no criterion is given, or if more than one is.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`combine.split`][ordito.combine.split]
        The whole decomposition, when every component is wanted rather than a subset.
    [`holes.join_closest_components`][ordito.holes.join_closest_components]
        The other answer to a multi-component mesh -- weld the pieces together instead of
        discarding them.
    [`adjacency.face_connected_component_labels`][ordito.adjacency.face_connected_component_labels]
        The labelling this thresholds.
    [`remove_non_manifold_faces`][ordito.repair.remove_non_manifold_faces]

    Notes
    -----
    A component's label is a representative *face* index rather than a dense ``0..k-1`` id, so the
    per-component statistic is an ``n_faces``-long array of which only ``k`` slots are ever written.
    That is deliberate: densifying the labels first would cost a sort and a search per face to save
    an allocation, and it is the allocation that is cheap.
    """
    require_same_device(vertices=vertices, faces=faces)
    given = (keep_largest, min_faces is not None, min_area is not None, min_diameter is not None)
    if sum(given) != 1:
        raise ValueError(
            "pass exactly one of keep_largest, min_faces, min_area or min_diameter, "
            f"got {sum(given)}"
        )

    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return vertices, faces

    labels = od.adjacency.face_connected_component_labels(faces, n_vertices=vertices.size)
    keep = _launch.empty(n_faces, dtype=wp.bool, device=device)

    if min_area is not None:
        # Each face's area is summed into its component where it is computed, rather than written
        # to a per-face buffer for a scatter pass to read back.
        statistic = _launch.zeros(n_faces, dtype=wp.float32, device=device)
        _launch.launch(
            kernel_repair.scatter_face_area_by_group,
            dim=n_faces,
            inputs=[vertices, faces, labels, statistic],
            device=device,
        )
        # Gather-then-compare at Python scope rather than a kernel: ``statistic[labels]`` is a
        # per-component table read through the per-face label, and ``wp.map`` over that
        # ``indexedarray`` is the standard elementwise-op idiom (and ``make_volume`` below already
        # uses it). ``labels`` is a dense array, so the strided-index hazard -- which applies to a
        # *column* of a rank-2 buffer -- does not arise. The bound is inclusive at every criterion,
        # matching both references.
        _launch.map(kernel_array.greater_equal, statistic[labels], wp.float32(min_area), out=keep)
    elif min_diameter is not None:
        # ``+inf`` in all six slots seeds both ends at once: the packing stores the upper corner
        # negated so every update is a ``wp.atomic_min``. Each face then decodes its own
        # component's box as it reads it.
        corners = _launch.full(6 * n_faces, value=math.inf, dtype=wp.float32, device=device)
        _launch.launch(
            kernel_scatter.scatter_group_bounds,
            dim=n_faces,
            inputs=[vertices, faces, labels, corners],
            device=device,
        )
        _launch.launch(
            kernel_repair.group_diameter_flags,
            dim=n_faces,
            inputs=[corners, labels, wp.float32(min_diameter), keep],
            device=device,
        )
    else:
        counts = _launch.zeros(n_faces, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_scatter.count_occurrences, dim=n_faces, inputs=[labels, counts], device=device
        )
        if min_faces is not None:
            _launch.map(kernel_array.greater_equal, counts[labels], wp.int32(min_faces), out=keep)
        else:
            # ``-1`` is below every packed key, so the reduction needs no separate seeding pass and
            # the winning label never reaches the host -- the mask kernel recomputes its key.
            best = _launch.full(1, -1, dtype=wp.int64, device=device)
            _launch.launch(
                kernel_repair.reduce_largest_group,
                dim=n_faces,
                inputs=[counts, best],
                device=device,
            )
            _launch.launch(
                kernel_repair.mark_largest_group_mask,
                dim=n_faces,
                inputs=[labels, counts, best, keep],
                device=device,
            )

    return od.selection.submesh_from_face_mask(vertices, faces, keep)


def split_non_manifold_vertices(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """

    Make a mesh manifold and orientable by duplicating vertices, keeping every face.

    The non-lossy counterpart of
    [`remove_non_manifold_faces`][ordito.repair.remove_non_manifold_faces], which deletes geometry
    to reach the same property: this changes no position and drops no triangle, it only splits
    vertices apart, so the surface is unchanged and the face count is exactly preserved. Use it to
    feed a mesh to code that requires manifold input -- several reference libraries (e.g.
    potpourri3d) reject non-manifold meshes outright, and so do ordito's own halfedge consumers.

    Two corners are kept together only across an edge that is **manifold and consistently
    oriented**: exactly one half-edge each way. Every other edge -- a boundary edge, one shared by
    three or more faces, or one whose two faces traverse it the same way -- separates the copies. So
    a bowtie vertex splits in two, an edge with three faces splits into three boundary edges, and a
    flipped face is cut free of its neighbours rather than reoriented (that is
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]'s job, and running it first
    leaves less to split).


    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_new,)`` positions with ``n_new >= n_referenced``; each is a copy of the original vertex
        it came from, so the point set is unchanged as a *set*.
    new_faces : wp.array[wp.int32]
        ``(3 * n_faces,)`` flat buffer of the same ``n_faces`` triangles in input order, indexing
        ``new_vertices``.
    source : wp.array[wp.int32]
        ``(n_new,)`` map from each new vertex to the original it duplicates, so
        ``new_vertices == vertices[source]`` and a per-vertex attribute transfers with
        [`gather`][ordito.array.gather]. This is ``igl.split_nonmanifold``'s ``SVI``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    Vertices unreferenced by any face are **dropped**, since a new vertex only exists as some face's
    corner -- the same convention as
    [`remove_unreferenced_vertices`][ordito.repair.remove_unreferenced_vertices], and the reason
    this function cannot simply return an ``n_vertices``-length remap.

    Remove degenerate faces first, with
    [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]: a triangle with a repeated
    index has two corners at one vertex that no edge can join, so it survives as two copies of that
    vertex and the face stays degenerate.

    The implementation is a connected-components pass over a graph of ``3 * n_faces`` corner nodes
    rather than a per-vertex star walk: one kernel counts each edge's half-edges by direction, a
    second emits two links per mergeable edge, and
    [`connected_component_labels_from_edges`][ordito.graph.connected_component_labels_from_edges]
    does the merging. ``igl::split_nonmanifold`` instead explodes the mesh to ``3 * n_faces``
    singleton vertices and greedily re-merges pairs, re-testing manifoldness after each candidate --
    order-dependent and sequential by construction. The two agree exactly on a manifold mesh, a
    bowtie vertex, a consistently-wound fan of three faces on one edge, a flipped face and an open
    boundary.

    **They differ on one input class, by design.** Where an edge carries one half-edge in one
    direction and *several* in the other -- which is what a duplicated face produces -- igl keeps
    one arbitrarily chosen pair joined, while this splits every copy. Both results are edge- and
    vertex-manifold with the input's face count; this one is order-independent and makes no
    arbitrary choice. Note that
    [`resolve_duplicated_faces`][ordito.repair.resolve_duplicated_faces] is not a way around the
    difference: the cancellation rules it implements cover a ``+1``/``-1`` imbalance, so a face
    duplicated in the *same* orientation makes it raise rather than dropping the copy.

    See Also
    --------
    [`remove_non_manifold_faces`][ordito.repair.remove_non_manifold_faces]
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]
    [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]
    [`is_edge_manifold`][ordito.validation.is_edge_manifold]
    ``igl.split_nonmanifold``
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0:
        empty_index = _launch.empty(0, dtype=wp.int32, device=device)
        return _launch.empty(0, dtype=wp.vec3, device=vertices.device), faces, empty_index

    n_corners = 3 * n_faces
    _unique_edges, edge_of_corner = od.edges.edges_unique(
        faces, n_vertices=vertices.size, validate=False
    )
    n_unique = int(_unique_edges.shape[0])

    forward_count = _launch.zeros(n_unique, dtype=wp.int32, device=device)
    backward_count = _launch.zeros(n_unique, dtype=wp.int32, device=device)
    forward_corner = _launch.full(n_unique, -1, dtype=wp.int32, device=device)
    backward_corner = _launch.full(n_unique, -1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.halfedge_orientation_slots,
        dim=n_corners,
        inputs=[
            faces,
            edge_of_corner,
            forward_count,
            backward_count,
            forward_corner,
            backward_corner,
        ],
        device=device,
    )

    links = odt.empty_2d((2 * n_unique, 2), wp.int32, device=device)
    _launch.launch(
        kernel_repair.corner_merge_links,
        dim=n_unique,
        inputs=[forward_count, backward_count, forward_corner, backward_corner, links],
        device=device,
    )

    # ``validate=False``: the links are corner ids this function just built, so the range check
    # would buy nothing but a full readback of the link buffer.
    labels = od.graph.connected_component_labels_from_edges(
        links, node_count=n_corners, validate=False
    )
    # ECL-CC labels each component by its smallest node id, and a node id *is* a corner, so the
    # representative's vertex is the original this copy came from.
    representatives, new_faces = od.grouping.unique_1d(
        labels, return_inverse=True, max_value=n_corners - 1
    )
    source = od.array.gather(faces, representatives)
    return od.array.gather(vertices, source), new_faces, source


def collapse_small_triangles(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], epsilon: float = 1e-6
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Collapse triangles smaller than a bounding-box-relative area threshold.

    Mirrors ``igl::collapse_small_triangles``. A triangle is *small* when its doubled area is below
    ``2 * epsilon * bbd ** 2``, where ``bbd`` is the diagonal of the axis-aligned bounding box of
    ``vertices``. Each small triangle has its **shortest edge** collapsed by merging that edge's two
    endpoints; the merged face (now carrying a repeated vertex) is discarded. The process repeats to
    a fixpoint, so triangles that only become small after a neighbouring collapse are also removed.

    This subsumes degenerate-triangle removal: an exactly degenerate face (zero area) is always
    below the threshold, so passing a small ``epsilon`` removes it. To drop only degenerate faces
    without any bounding-box-relative collapsing, use
    [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    epsilon
        Relative area tolerance. The doubled-area threshold is ``2 * epsilon * bbd ** 2``; larger
        values collapse more (and larger) triangles.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_new_vertices,)`` vertices surviving the collapse, compacted from index zero.
    new_faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat buffer of the surviving faces, remapped into
        ``new_vertices``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]
    [`face_nondegenerate_mask`][ordito.triangles.face_nondegenerate_mask]

    Notes
    -----
    Where ``igl::collapse_small_triangles`` merges vertices by a sequentially updated index map and
    recurses until no edge collapses, this resolves all shortest-edge merges of one pass at once via
    the connected-components closure of
    [`connected_component_labels_from_edges`][ordito.graph.connected_component_labels_from_edges],
    then loops over the shrinking mesh. Both converge to a mesh with no sub-threshold triangle. The
    surviving vertex of a collapsed edge keeps the position of the component representative (the
    lowest original index in its class) rather than libigl's longest-edge-preserving endpoint; for
    sub-threshold triangles the two endpoints are close enough that the difference is negligible.
    The bounding-box diagonal is measured once on the input ``vertices`` so the threshold is fixed
    across iterations.
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0 or vertices.size == 0:
        return _launch.clone(vertices), _launch.clone(faces)

    bbd = od.bounds.enclosing_diagonal(vertices)
    min_dbl_area = wp.float32(2.0 * epsilon * bbd * bbd)

    current_vertices = vertices
    current_faces = faces
    max_iterations = faces.size  # bounded: each collapsing pass drops at least one face
    # One small-face counter for the whole loop, zeroed per pass: the kernel that finds the small
    # faces counts them, so the stopping test is one 4-byte read rather than a per-face flag buffer
    # reduced on the device first.
    n_small = _launch.zeros(1, dtype=wp.int32, device=device)
    for _ in range(max_iterations):
        n_current = current_faces.size // 3
        if n_current == 0:
            break

        _launch.zero_(n_small)
        pairs = odt.empty_2d((n_current, 2), wp.int32, device=device)
        _launch.launch(
            kernel_repair.small_triangle_collapse_edges,
            dim=n_current,
            inputs=[current_vertices, current_faces, min_dbl_area, pairs, n_small],
            device=device,
        )

        if int(read_scalar(n_small, 0)) == 0:
            break

        # Non-flagged faces emit a self-pair (i0, i0); these are self-loops that leave the
        # connected-components closure unchanged, so all rows can be passed without filtering.
        n_vertices = current_vertices.size
        # ``validate=False``: the pairs are vertex indices read straight out of the face buffer
        # this function is repairing, which every kernel around it already indexes unchecked.
        labels = od.graph.connected_component_labels_from_edges(
            pairs, node_count=n_vertices, validate=False
        )

        unique_labels, inverse = unique_1d(labels, return_inverse=True, max_value=n_vertices - 1)
        # Each label is its class's smallest vertex, so the sorted labels are already every class's
        # first occurrence in ``inverse``: the representatives, with no scatter to find them.
        class_vertices = od.array.gather(current_vertices, unique_labels)
        remapped_faces = od.array.remap_indices(current_faces, inverse)

        keep_mask = od.triangles.face_nondegenerate_mask(class_vertices, remapped_faces)
        current_vertices, current_faces = od.selection.submesh_from_face_mask(
            class_vertices, remapped_faces, keep_mask
        )

    return current_vertices, current_faces


@overload
def straighten_boundary(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    min_normal_dot: float = 0.9,
    max_aspect_ratio: float = 10.0,
    iterations: int = 1,
    return_count: Literal[False] = False,
) -> wp.array[wp.int32]: ...
@overload
def straighten_boundary(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    min_normal_dot: float = 0.9,
    max_aspect_ratio: float = 10.0,
    iterations: int = 1,
    return_count: Literal[True],
) -> tuple[wp.array[wp.int32], int]: ...
def straighten_boundary(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    min_normal_dot: float = 0.9,
    max_aspect_ratio: float = 10.0,
    iterations: int = 1,
    return_count: bool = False,
) -> wp.array[wp.int32] | tuple[wp.array[wp.int32], int]:
    """
    Close concave notches in the mesh's rim, one triangle at a time.

    A rim that came out of a clip, a decimation or a scan is *ragged*: it zig-zags by one triangle
    even where the surface is smooth. Each pass adds the single triangle that spans a rim vertex's
    two neighbours, wherever that triangle faces the same way as the surface it joins and is not a
    sliver. No vertex moves and none is added -- the mesh only gains faces, so the interior is
    untouched and every existing index stays valid.

    Two gates, and both matter. ``min_normal_dot`` is what distinguishes a notch from a corner: a
    *convex* rim corner would be closed by a triangle facing away from the surface, and filling it
    folds the mesh over. ``max_aspect_ratio`` stops the pass trading a ragged rim for a fan of
    slivers. Neither has a natural default, which is why both are keywords rather than constants.

    Adjacent notches share a rim edge, so one pass closes a maximal independent set of them --
    lowest index wins, deterministically -- and ``iterations`` passes go round again on what is
    left. A pass that closes nothing ends the loop early.

    Edge-manifoldness is not enough to make the rim a set of simple loops: at a *bowtie* rim
    vertex two loops are pinched at one point, so that vertex has two outgoing boundary halfedges
    and no single successor. The rim is therefore followed by halfedge rather than by vertex, which
    gives each loop its own links: both notches at such a vertex are closed, neither is attached
    across the pinch, and the answer is the same on every device.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Read only; nothing moves.
    faces
        ``(3 * n_faces,)`` triangle index buffer. Must be edge-manifold, since the
        rim is found through the halfedge twins.
    min_normal_dot
        Minimum cosine between the new triangle's normal and each of the two rim faces it will
        border. ``1.0`` accepts only a perfectly flat notch; ``0.0`` accepts any notch that is not
        folded back on the surface.
    max_aspect_ratio
        Largest circum-radius over twice the in-radius the new triangle may have, as
        [`face_quality`][ordito.triangles.face_quality] measures it.
    iterations
        Number of independent-set passes.
    return_count
        If ``True``, also return ``added``.

    Returns
    -------
    faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat triangle index buffer with the new triangles appended. The
        vertex buffer is unchanged and is not returned.
    added : int, optional
        Present when ``return_count=True``. How many triangles were added, summed over the passes.
        Zero means no notch passed both gates, and the buffer is the input's. A diagnostic: the
        pass loop already stops itself when a pass closes nothing, so a caller needs this only to
        report what happened.

    Raises
    ------
    ValueError
        If ``iterations`` is negative, or propagated from
        [`halfedge_twins`][ordito.halfedge.halfedge_twins] when the mesh is not edge-manifold.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`fill_min_weight`][ordito.holes.fill_min_weight]
        Closes a rim *completely*; this only tidies its shape and leaves it open.
    [`ears`][ordito.boundary.ears]
        Finds the faces with two rim edges, which is the dual situation -- an ear sticks out where a
        notch cuts in.
    [`boundary_loops`][ordito.boundary.boundary_loops]
    """
    require_same_device(vertices=vertices, faces=faces)
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")
    device = faces.device
    n_vertices = vertices.size
    if n_vertices == 0 or faces.size == 0 or iterations == 0:
        return (faces, 0) if return_count else faces

    added = 0
    # One emit cursor for the whole loop, zeroed per pass rather than reallocated: a four-byte
    # buffer per iteration is an allocation where a memset does.
    cursor = _launch.zeros(1, dtype=wp.int32, device=device)
    for _ in range(iterations):
        _launch.zero_(cursor)
        n_faces = faces.size // 3
        n_halfedges = 3 * n_faces
        twins = od.halfedge.halfedge_twins(faces, n_vertices=n_vertices)
        # Keyed by halfedge, not by vertex: a bowtie rim vertex carries two rim loops and has no
        # single successor, and edge-manifoldness -- all `halfedge_twins` checks -- does not
        # exclude one. One slot per halfedge gives each loop its own links and one writer each.
        rim_next = _launch.full(n_halfedges, -1, dtype=wp.int32, device=device)
        rim_prev = _launch.full(n_halfedges, -1, dtype=wp.int32, device=device)
        candidate = _launch.zeros(n_halfedges, dtype=wp.bool, device=device)
        # One launch: a halfedge's notch test needs only the successor this same thread computes,
        # so linking the rim and classifying it are one pass rather than two with the link table
        # written out and read back between them.
        _launch.launch(
            kernel_repair.collect_rim_links_and_candidates,
            dim=n_halfedges,
            inputs=[
                vertices,
                faces,
                twins,
                od.triangles.face_normals_and_areas(vertices, faces)[0],
                wp.float32(min_normal_dot),
                wp.float32(max_aspect_ratio),
                rim_next,
                rim_prev,
                candidate,
            ],
            device=device,
        )
        # One notch per boundary halfedge at most, and a pinched rim can carry more of those than
        # the mesh has vertices, so the bound is the halfedge count. Trimmed to the real count
        # below, so the slack never leaves this loop.
        new_faces = odt.empty_2d((n_halfedges, 3), wp.int32, device=device)
        _launch.launch(
            kernel_repair.emit_straighten_faces,
            dim=n_halfedges,
            inputs=[faces, rim_next, rim_prev, candidate, cursor, new_faces],
            device=device,
        )
        # One readback per pass, and it is the loop's own stopping test: how many notches the pass
        # actually closed is a device-side fact and a Python loop cannot branch on it otherwise.
        n_added, (accepted,) = od.array.trim_to_count(cursor, new_faces)
        if n_added == 0:
            break
        faces = od.array.concatenate([faces, accepted.reshape(3 * n_added)])
        added += n_added
    return (faces, added) if return_count else faces


@overload
def remove_degree3_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    max_iter: int = 8,
    return_count: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def remove_degree3_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    max_iter: int = 8,
    return_count: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], int]: ...
def remove_degree3_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    max_iter: int = 8,
    return_count: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]] | tuple[wp.array[wp.vec3], wp.array[wp.int32], int]
):
    """
    Remove interior vertices with exactly three incident faces, collapsing each fan to one triangle.

    A valence-3 interior vertex carries no information the surface needs: its three faces tile the
    triangle formed by its three neighbours, so deleting it and keeping that triangle changes the
    connectivity and nothing else. They are a standard residue of subdivision, of decimation and of
    hole filling, and they make every downstream valence statistic worse.

    Adjacent candidates share faces, so a pass removes a maximal **independent** set -- the
    lowest-indexed of any two neighbouring candidates wins, deterministically -- and the loop
    repeats until none is left or ``max_iter`` passes have run. Removing one vertex can create
    another, which is why ``return_count`` reports what happened rather than promising it.

    !!! note "The other half of a double-face pass already exists"
        A pair of triangles on the same three vertices is
        [`resolve_duplicated_faces`][ordito.repair.resolve_duplicated_faces]' job, and this
        function deliberately does not repeat it. The two together are what a "remove double faces"
        pass means elsewhere; they are kept apart because they have different orientation rules and
        different answers on a non-orientable mesh.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer. Must be edge-manifold, since the
        fan around a vertex is what this reasons about. A pinched (vertex-non-manifold) vertex is
        accepted and is never a candidate: its faces do not close into one fan of three.
    max_iter
        Cap on the number of passes. Each pass removes an independent set, so a chain of adjacent
        candidates needs one pass per link; the default covers any chain length likely in practice.
    return_count
        If ``True``, also return ``removed``.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_new_vertices,)`` positions of the result, with the removed vertices compacted away. No
        position moves.
    faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat triangle index buffer, three faces shorter per removed vertex
        plus one longer.
    removed : int, optional
        Present when ``return_count=True``. How many vertices were removed. Zero means the input
        had none and the buffers are it. A diagnostic: the pass loop stops itself when a pass finds
        no candidate, so nothing about calling this correctly depends on reading the count.

    Raises
    ------
    ValueError
        If ``max_iter`` is negative, or (when a pass runs) if an edge is shared by three or more
        faces or both of an edge's halfedges traverse it the same way -- the two meshes
        [`halfedge_twins`][ordito.halfedge.halfedge_twins] rejects, tested without building its
        table.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`flatten_degree3_vertices`][ordito.repair.flatten_degree3_vertices]
        The geometric answer to the same defect: move the vertex instead of deleting it, which
        leaves the connectivity and every per-vertex attribute intact.
    [`resolve_duplicated_faces`][ordito.repair.resolve_duplicated_faces]
        The other half: triangles repeated on the same three vertices.
    [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]
        Removes faces by *geometry*; this one removes a vertex by its connectivity alone.
    [`vertex_one_rings`][ordito.halfedge.vertex_one_rings]
        The whole fan of every vertex; this reads only the three faces of a candidate.
    """
    require_same_device(vertices=vertices, faces=faces)
    if max_iter < 0:
        raise ValueError(f"max_iter must be non-negative, got {max_iter}")
    device = faces.device
    n_vertices = vertices.size
    n_input = faces.size // 3
    if n_input == 0 or max_iter == 0:
        return (vertices, faces, 0) if return_count else (vertices, faces)
    # The documented edge-manifold check, on the input only: the run-length test on the sorted
    # halfedge keys, counted inside pass 0's first launch and read back with that pass's counters,
    # rather than a twin table the pass never reads. Replacing an interior degree-3 fan by the one
    # triangle over its rim keeps every rim edge at two faces with the same orientation, so a pass
    # cannot make a valid mesh invalid and later passes do not re-prove it.
    input_keys, input_order = od.adjacency.sorted_face_edge_keys(faces, n_vertices=n_vertices)
    # The passes run over one fixed-capacity face buffer rather than compacting after each: a
    # replaced fan's three faces keep their rows with their kept flag cleared, and the replacements
    # are appended after the last used row. Each removal takes three kept faces and adds one, so at
    # most ``n_input // 2`` rows are ever appended; kept rows in buffer order are exactly the order
    # a compaction after every pass gives, so the output is compacted once, after the loop, with no
    # per-pass scan, compaction or allocation.
    #
    # The loop stays on the host, with one small read per pass: recording the passes costs more
    # than those reads at the pass counts real meshes take (``kernels/repair.degree3_fan_tables``).
    n_slots = n_input + max(n_input // 2, 1)
    # One zeroed buffer: the kept flags; the referenced flags the final compaction scans with them
    # (one in-place scan over both); the append cursor (the removed count) and pass 0's two
    # twin-defect counts and referenced vertex count; then, zeroed per pass, the next-pass
    # candidate count and each vertex's corner count, link sum and decrement tally
    # (``degree3_fan_tables`` and ``emit_degree3_replacement`` say what each means). The cursor
    # through the candidate count is what each pass reads, in one copy. The views are stamped by
    # one ``array.split`` rather than sliced one by one.
    tail = n_slots + n_vertices
    per_pass = tail + 4
    state = _launch.zeros(per_pass + 1 + 3 * n_vertices, dtype=wp.int32, device=device)
    bounds = [0, n_slots, tail, tail + 1, per_pass, per_pass + 1]
    bounds += [per_pass + 1 + n_vertices, per_pass + 1 + 2 * n_vertices, state.size]
    kept, referenced, cursor, defects, next_candidates, counts, link_sums, lost = od.array.split(
        state, bounds
    )
    tables = state[per_pass:]
    fans = odt.empty_2d((n_vertices, 3), wp.int32, device=device)
    face_slots = _launch.empty(3 * n_slots, dtype=wp.int32, device=device)
    new_faces = odt.as_dense(face_slots[3 * n_input :]).reshape((n_slots - n_input, 3))
    emit_inputs = [
        face_slots,
        fans,
        counts,
        link_sums,
        n_input,
        cursor,
        lost,
        kept,
        new_faces,
        next_candidates,
    ]
    _launch.launch(
        kernel_repair.degree3_fan_tables_input,
        dim=3 * n_input,
        inputs=[faces, input_keys, input_order, counts, link_sums, fans, face_slots, kept, defects],
        device=device,
    )
    _launch.launch(
        kernel_repair.emit_degree3_replacement, dim=n_vertices, inputs=emit_inputs, device=device
    )
    # One readback per pass, carrying both of the loop's host decisions: the removed count, which
    # ends the call when pass 0 found nothing and sizes the next pass, and the next-pass candidate
    # count, which skips the pass that would only learn it finds nothing. Pass 0's also carries the
    # input validation, whose failure discards that pass.
    removed, n_nonmanifold, n_misoriented, n_referenced, n_next = read_values(state, tail, 5)
    _raise_degree3_defects(n_nonmanifold, n_misoriented)
    if removed == 0:
        return (vertices, faces, 0) if return_count else (vertices, faces)
    for _pass_index in range(1, max_iter):
        if n_next == 0:
            break
        _launch.zero_(tables)
        _launch.launch(
            kernel_repair.degree3_fan_tables,
            dim=3 * (n_input + removed),
            inputs=[face_slots, kept, counts, link_sums, fans],
            device=device,
        )
        _launch.launch(
            kernel_repair.emit_degree3_replacement,
            dim=n_vertices,
            inputs=emit_inputs,
            device=device,
        )
        removed, _nonmanifold, _misoriented, _referenced, n_next = read_values(state, tail, 5)
    n_used = n_input + removed
    # Compacted **once**, after the loop rather than inside it -- the kept faces and, as
    # ``remove_unreferenced_vertices`` would, the vertices they reference. A dead vertex has an
    # empty ring and so is never a candidate, which is what makes deferring safe.
    _launch.launch(
        kernel_repair.mark_kept_face_vertices,
        dim=3 * n_used,
        inputs=[face_slots, kept, referenced],
        device=device,
    )
    scanned = state[:tail]
    _launch.array_scan(scanned, out_array=scanned, inclusive=True)
    # Both output sizes are known: each removal drops two faces net and exactly one vertex, the
    # fan's centre (the replacement face references the whole rim).
    n_kept = n_input - 2 * removed
    n_referenced -= removed
    out_vertices = _launch.empty(n_referenced, dtype=wp.vec3, device=device)
    out_faces = _launch.empty(3 * n_kept, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.compact_kept_faces_and_vertices,
        dim=max(n_vertices, n_used),
        inputs=[vertices, face_slots, kept, referenced, n_kept, out_vertices, out_faces],
        device=device,
    )
    return (out_vertices, out_faces, removed) if return_count else (out_vertices, out_faces)


def _raise_degree3_defects(n_nonmanifold: int, n_misoriented: int) -> None:
    """Raise ``remove_degree3_vertices``' two input rejections from pass 0's defect counts."""
    if n_nonmanifold > 0:
        raise ValueError(
            f"remove_degree3_vertices requires an edge-manifold mesh: {n_nonmanifold} edge(s) are "
            f"shared by three or more faces."
        )
    if n_misoriented > 0:
        raise ValueError(
            f"remove_degree3_vertices requires a consistently wound mesh: {n_misoriented} edge(s) "
            f"are traversed in the same direction by both of their halfedges. Run "
            f"make_winding_consistent first."
        )


def flatten_degree3_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool] | None = None,
    *,
    max_iter: int = 8,
    rings: tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]] | None = None,
) -> wp.array[wp.vec3]:
    """
    Flatten each interior valence-3 vertex into the plane of its three neighbours.

    A valence-3 interior vertex sits on a little tetrahedral bump: three triangles meeting at a
    point over the triangle its neighbours form. Moving it to their centroid puts it **in** that
    triangle's plane, so the bump disappears and the three faces become coplanar -- which is what
    makes a subdivision, a decimation or a hole fill stop leaving visible pimples.

    The gentler half of a pair.
    [`remove_degree3_vertices`][ordito.repair.remove_degree3_vertices]
    answers the same defect by deleting the vertex and keeping one triangle, which changes the
    connectivity; this keeps every vertex and every face and only moves positions, so a caller
    holding per-vertex attributes or a face selection can use it and the other one would invalidate
    both.

    Two interior valence-3 vertices **can** be neighbours -- a tetrahedron is four of them, each
    adjacent to the other three -- and moving both at once puts neither in the other's new plane.
    So a pass flattens a maximal **independent** set of them, lowest index wins, exactly as
    [`remove_degree3_vertices`][ordito.repair.remove_degree3_vertices] does, and ``max_iter``
    passes go round again on the candidates left over. Every vertex lands in its neighbours' plane
    as it is moved, which is the guarantee the name makes; where two candidates are adjacent the
    later one sees the earlier one's new position, so the order the passes impose is part of the
    answer rather than an implementation detail. The loop stops as soon as no candidate is left, so
    the usual mesh -- where no two candidates touch -- takes one pass and one check.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer. Must be edge-manifold, since the
        fan around a vertex is what this reasons about.
    region
        ``(n_vertices,)`` mask restricting which vertices may be flattened. ``None``
        flattens every one that qualifies. Masked-out vertices are also excluded from the
        independence rule, since a vertex that cannot move cannot spoil a neighbour's plane.
    max_iter
        Cap on the number of passes. Each pass flattens an independent set, so a chain of adjacent
        candidates needs one pass per link; the default covers any chain length likely in practice
        -- a tetrahedron, the densest case there is, needs four.
    rings
        Optional precomputed [`vertex_one_rings`][ordito.halfedge.vertex_one_rings] as
        ``(ring_halfedges, offsets, is_boundary)``. Depends on the connectivity alone, so one CSR
        serves every fan walk over the same mesh --
        [`Trimesh.vertex_one_rings`][ordito.mesh.Trimesh.vertex_one_rings] has it cached, and
        passing it skips the vertex-manifold check and the host readback that check costs.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_vertices,)`` positions on ``vertices.device``, with the selected vertices at their
        neighbours' centroid. Connectivity is untouched, so ``faces`` stays valid.

    Raises
    ------
    ValueError
        If ``max_iter`` is negative, if ``region`` is not a length-``n_vertices`` ``wp.bool``
        array, or propagated from
        [`halfedge_twins`][ordito.halfedge.halfedge_twins] when the mesh is not edge-manifold.
    RuntimeError
        If ``vertices``, ``faces``, ``region`` and ``rings`` are not all on one device.

    See Also
    --------
    [`remove_degree3_vertices`][ordito.repair.remove_degree3_vertices]
        The topological answer to the same defect: delete the vertex instead of moving it.
    [`equalize_triangle_areas`][ordito.smoothing.equalize_triangle_areas]
        Relaxes every vertex toward an area objective, where this hard-sets only the valence-3 ones.
    """
    require_same_device(vertices=vertices, faces=faces, region=region, rings=rings)
    if max_iter < 0:
        raise ValueError(f"max_iter must be non-negative, got {max_iter}")
    device = faces.device
    n_vertices = vertices.size
    if n_vertices == 0 or faces.size == 0 or max_iter == 0:
        return _launch.clone(vertices)
    if region is not None and (
        len(region.shape) != 1 or region.size != n_vertices or region.dtype is not wp.bool
    ):
        raise ValueError(
            f"region must be a length-{n_vertices} wp.bool array, got shape {tuple(region.shape)} "
            f"of {region.dtype}"
        )

    ring_halfedges, ring_offsets, is_boundary = (
        rings if rings is not None else od.halfedge.vertex_one_rings(faces, n_vertices=n_vertices)
    )
    candidate = _launch.empty(n_vertices, dtype=wp.bool, device=device)
    if region is None:
        _launch.map(
            kernel_repair.is_interior_degree3,
            ring_offsets[:-1],
            ring_offsets[1:],
            is_boundary,
            out=candidate,
        )
    else:
        _launch.map(
            kernel_repair.is_interior_degree3_in_region,
            ring_offsets[:-1],
            ring_offsets[1:],
            is_boundary,
            region,
            out=candidate,
        )

    positions = vertices
    # Scratch hoisted out of the loop: the connectivity and the vertex count are both invariant
    # here, so a pass reuses these rather than allocating. The two position buffers alternate so
    # the caller's own ``vertices`` is never written -- pass 0 reads it and writes ``buffers[0]``,
    # pass 1 reads that and writes ``buffers[1]``, and so on -- and the two candidate masks
    # alternate the same way, each pass writing the next one's.
    remaining = _launch.empty(n_vertices, dtype=wp.bool, device=device)
    # Allocated on first use, not upfront: the common mesh has no two candidates adjacent, so the
    # loop runs a single pass and only ever needs one of the two.
    buffers: dict[int, wp.array[wp.vec3]] = {}
    for iteration in range(max_iter):
        # Adjacent candidates have to be separated before anything moves -- see the kernel comment
        # for what flattening both ends of an edge at once does to a tetrahedron. The lowest
        # remaining index always wins its own conflict, so every pass retires at least one
        # candidate and the loop cannot spin.
        slot = iteration % 2
        if slot not in buffers:
            buffers[slot] = _launch.empty(n_vertices, dtype=wp.vec3, device=device)
        flattened = buffers[slot]
        # One launch: choosing the independent set, moving the vertices it chose and retiring them
        # from the candidates are the same thread's decision about the same vertex.
        _launch.launch(
            kernel_repair.select_and_flatten_degree3,
            dim=n_vertices,
            inputs=[positions, faces, ring_offsets, ring_halfedges, candidate],
            outputs=[remaining, flattened],
            device=device,
        )
        positions = flattened
        candidate, remaining = remaining, candidate
        if iteration + 1 == max_iter:
            break
        # One readback per pass beyond the first, and it is the loop's own termination test: how
        # many candidates are still unflattened is a device-side fact and a Python loop cannot
        # branch on it otherwise. It is taken *after* the pass rather than before so that a mesh
        # with no two candidates adjacent -- the usual one -- pays exactly one, and ``max_iter=1``
        # pays none. The connectivity never changes here, so nothing above the loop is rebuilt.
        #
        # Read back and reduced on the host rather than through ``reduce``: a bool mask is one byte
        # per vertex, so below about half a million elements the copy is cheaper than the launch a
        # device reduction costs (CLAUDE.md section 14.5). This call is over the *vertex* count, so
        # it stays on the readback at every registry mesh.
        if not bool(candidate.numpy().any()):
            break
    return positions


def reverse_winding(faces: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Reverse every face's winding, flipping the surface's orientation.

    Rewrites each triangle ``(a, b, c)`` as ``(c, b, a)``, which negates every face normal and the
    enclosed signed volume. Unconditional: unlike
    [`make_winding_consistent`][ordito.repair.make_winding_consistent] and
    [`make_normals_outward`][ordito.repair.make_normals_outward], which decide per face, this
    flips all of them, so a consistently wound mesh stays consistent and an inconsistent one stays
    inconsistent.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` triangle index buffer.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` new face buffer. ``faces`` is not modified.

    Notes
    -----
    An involution: applying it twice returns the original buffer exactly.

    Each face's signed volume negates *exactly* -- reversing a triangle's corners swaps two
    arguments of a scalar triple product, which negates the same floating-point products rather
    than recomputing them -- so the total negates bit-for-bit wherever the reduction visits faces
    in a fixed order. The example below asserts only the sign, since that is what holds
    independently of the reduction.

    The corner that stays first is a convention, and this one matches ``np.fliplr`` -- which is
    what ``trimesh.Trimesh.invert`` applies, so the two agree elementwise rather than only up to a
    rotation of each row. ``kernels.repair.flip_faces_masked``, which the per-face flippers use,
    keeps corner 0 instead; both reverse orientation and they differ by a cyclic rotation.

    Examples
    --------
    ```python
    flipped = od.repair.reverse_winding(f)
    assert od.measures.volume(v, flipped) < 0.0 < od.measures.volume(v, f)
    ```

    See Also
    --------
    [`ordito.mesh.Trimesh.invert`][]
        The cached-mesh form, which carries what survives a flip.
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]
    [`make_normals_outward`][ordito.repair.make_normals_outward]
    [`ordito.validation.is_winding_consistent`][]
    """
    reversed_faces = _launch.empty(faces.size, dtype=wp.int32, device=faces.device)
    n_faces = faces.size // 3
    if n_faces > 0:
        _launch.launch(
            kernel_repair.reverse_face_winding,
            dim=n_faces,
            inputs=[faces, reversed_faces],
            device=faces.device,
        )
    return reversed_faces


def make_winding_consistent(
    faces: wp.array[wp.int32], *, n_vertices: int | None = None
) -> wp.array[wp.int32]:
    """
    Flip faces so every shared edge is traversed in opposite directions by its two faces.

    Reuses the orientation flood-fill of
    [`face_flip_mask`][ordito.validation.face_flip_mask] (each connected component seeded at
    its lowest-indexed face) and reverses the winding of every face whose orientation bit is
    set. The result satisfies
    [`is_winding_consistent`][ordito.validation.is_winding_consistent] **whenever one exists**,
    which is to say whenever the mesh is
    [`is_orientable`][ordito.validation.is_orientable]; an already-consistent mesh is returned
    unchanged. Mirrors ``trimesh.repair.fix_winding``.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    n_vertices
        Optional exclusive bound on the vertex indices, forwarded to
        [`face_flip_mask`][ordito.validation.face_flip_mask] for its key sort. It does not
        change the answer, and it is trusted, not checked.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` new flat face buffer with corrected winding, on ``faces.device``.
        Vertices are unchanged.

    See Also
    --------
    [`is_winding_consistent`][ordito.validation.is_winding_consistent]
    [`face_flip_mask`][ordito.validation.face_flip_mask]
    [`make_normals_outward`][ordito.repair.make_normals_outward]

    Notes
    -----
    Each connected component keeps the winding of its lowest-indexed face, so on an orientable
    mesh the result is a deterministic function of ``faces`` on either device;
    ``trimesh.repair.fix_winding``'s BFS likewise keeps its seed face's winding, though its seed
    choice is its own. Use [`make_volume`][ordito.repair.make_volume] afterwards to also orient
    normals outward.

    On a **non-orientable** mesh no consistent winding exists, so this cannot succeed and does not
    fail either: the flood-fill orients everything it reaches and the contradiction is left on a
    seam. The seam can even end up with *more* inconsistent edges than before the pass ran, not
    merely the same ones — so treat the result as unrepaired rather than partly repaired, and test
    with [`is_orientable`][ordito.validation.is_orientable] first if that matters.
    """
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.int32, device=device)

    # The flip mask solves the bits straight off the sorted halfedge keys, with no adjacency table
    # and no host read of its length -- the bits ``face_orientation_bits`` gives.
    flip = od.validation.face_flip_mask(faces, n_vertices=n_vertices)
    out_faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.flip_faces_masked, dim=n_faces, inputs=[faces, flip, out_faces], device=device
    )
    return out_faces


def make_volume(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, multibody: bool = False
) -> wp.array[wp.int32]:
    """
    Orient faces so the mesh encloses a positive signed volume (normals point outward).

    Mirrors ``trimesh.repair.fix_inversion``. With ``multibody=False`` (default) the mesh is only
    corrected when it is watertight (every undirected edge shared by exactly two faces) and its
    total signed volume is negative, in which case every face is reversed. With ``multibody=True``
    each connected component is corrected independently by the sign of its own signed volume.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    multibody
        When ``True`` correct each connected component independently rather than the mesh as a
        whole.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` new flat face buffer with outward-oriented normals, on
        ``faces.device``. Vertices are unchanged.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`is_volume`][ordito.validation.is_volume]
        Test whether this succeeded.
    [`volume`][ordito.measures.volume]
        Measure the result.
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]
    [`make_normals_outward`][ordito.repair.make_normals_outward]

    Notes
    -----
    The signed volume is ``sum(dot(v0, cross(v1, v2)) / 6)`` measured from the origin, as in
    [`is_volume`][ordito.validation.is_volume]. Unlike ``trimesh.repair.fix_inversion``'s
    multibody path, this does not skip components that are not watertight/consistently wound: an
    open component's signed volume is ill-defined and may be flipped spuriously. Run
    [`make_winding_consistent`][ordito.repair.make_winding_consistent] first (see
    [`make_normals_outward`][ordito.repair.make_normals_outward]) and reserve ``multibody``
    for meshes whose bodies are individually closed.
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.int32, device=device)

    if multibody:
        out_faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
        labels = od.adjacency.face_connected_component_labels(faces, n_vertices=vertices.size)
        accum = _launch.zeros(n_faces, dtype=wp.float32, device=device)
        _launch.launch(
            kernel_repair.scatter_face_volume_by_group,
            dim=n_faces,
            inputs=[vertices, faces, labels, accum],
            device=device,
        )
        # One launch: each face reads its own component's signed volume through its label, where
        # gathering that into a per-face flag buffer first cost a map, a launch and the buffer.
        _launch.launch(
            kernel_repair.flip_faces_by_component_volume,
            dim=n_faces,
            inputs=[faces, labels, accum, out_faces],
            device=device,
        )
        return out_faces

    # The predicate, not ``all(face_watertight_mask(faces))``: the mask additionally builds
    # ``unique_1d``'s inverse and runs a per-face gather pass, only to be reduced to one bool.
    # Both answer "is every undirected edge shared by exactly two faces", because every unique edge
    # in the table comes from a face.
    if not od.validation.is_edge_manifold(
        faces, allow_boundary_edges=False, n_vertices=vertices.size
    ):
        return _launch.clone(faces)

    if od.measures.volume(vertices, faces) >= 0.0:
        return _launch.clone(faces)

    out_faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.flip_all_faces, dim=n_faces, inputs=[faces, out_faces], device=device
    )
    return out_faces


def make_normals_outward(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, multibody: bool = False
) -> wp.array[wp.int32]:
    """
    Make winding consistent and orient normals outward (winding fix followed by inversion fix).

    Equivalent to ``trimesh.repair.fix_normals``: first
    [`make_winding_consistent`][ordito.repair.make_winding_consistent] gives every connected
    component a coherent winding, then [`make_volume`][ordito.repair.make_volume] flips it (or each
    body, with ``multibody=True``) so normals point outward. On a watertight, orientable mesh the
    result satisfies [`is_volume`][ordito.validation.is_volume].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    multibody
        Forwarded to [`make_volume`][ordito.repair.make_volume]: correct each connected component
        independently.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` new flat face buffer with consistent winding and outward normals, on
        ``faces.device``. Vertices are unchanged.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]
    [`make_volume`][ordito.repair.make_volume]
    [`is_volume`][ordito.validation.is_volume]
    """
    require_same_device(vertices=vertices, faces=faces)
    wound = make_winding_consistent(faces, n_vertices=vertices.size)
    return make_volume(vertices, wound, multibody=multibody)


def remove_folded_faces(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, angle: float = 160.0
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Drop faces that fold back over their own ring, and reindex.

    A folded face is one whose dihedral angle to a neighbour is near ``pi``: the two triangles lie
    almost on top of each other with opposite normals, which is what a badly reconstructed or
    self-intersecting patch looks like locally. Such a face contributes no surface and breaks every
    normal-based computation downstream, so removing it is a repair rather than a simplification.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    angle
        Dihedral threshold in **degrees**; a face with a neighbour above it is dropped. MeshLab's
        ``folded_faces_angle_threshold``, whose default of ``160`` is this one. Must be in
        ``(0, 180]``.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_new_vertices,)`` vertices still referenced by a kept face, compacted from index zero.
    new_faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat buffer of the kept faces, remapped into ``new_vertices``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`validation.face_defective_mask`][ordito.validation.face_defective_mask]
    [`flip_t_vertices`][ordito.repair.flip_t_vertices]
    [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]

    Notes
    -----
    MeshLab's ``meshing_remove_folded_faces`` *flips* the offending edge instead of deleting the
    face, which preserves the face count but can only help when the fold is a triangulation mistake
    rather than genuinely folded geometry. Deletion is the choice the rest of this module makes (see
    [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces] and
    [`remove_non_manifold_faces`][ordito.repair.remove_non_manifold_faces]), and it leaves a hole
    that [`ordito.holes`][ordito.holes] can retriangulate properly.
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.clone(vertices), _launch.clone(faces)
    folded = od.validation.face_defective_mask(
        vertices, faces, min_quality=None, max_fold_angle=angle
    )
    keep = _launch.empty(n_faces, dtype=wp.bool, device=faces.device)
    _launch.map(kernel_array.mask_not, folded, out=keep)
    return od.selection.submesh_from_face_mask(vertices, faces, keep)


def fix_self_intersections(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    method: Literal["local", "voxel"] = "local",
    max_expand: int = 1,
    max_iter: int = 3,
    voxel_size: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """

    Remove a mesh's self-intersections, either locally or by rebuilding it.

    ordito could *detect* a self-intersection
    ([`ordito.validation.face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask])
    and not repair one. This is the repair, in the two forms that exist:

    - ``"local"`` cuts the trouble out and rebuilds it. The intersecting faces are dilated by
      ``max_expand`` rings, that region is deleted, and the rims it opens are refilled by the
      minimum-weight patch -- so the surface away from the intersection is **untouched**. Iterated,
      because a patch can intersect something itself.
    - ``"voxel"`` rebuilds the whole surface as the zero level set of its own signed distance field.
      A level set cannot self-intersect, so this always terminates and resamples everything --
      including the parts that were fine. Read the qualification in the Notes: the level set is
      clean, its *triangulation* can still carry an artifact at an ambiguous marching-cubes cell.


    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    method
        ``"local"`` (default) to cut and refill, ``"voxel"`` to rebuild through a distance field.
    max_expand
        Rings of faces added around each intersecting face before deleting, in the ``"local"``
        method. Larger takes more surface with it and is likelier to succeed in one pass.
    max_iter
        Cap on cut-and-refill passes. The loop also stops as soon as nothing intersects. Raising it
        does **not** help a deep interpenetration and inflates the mesh; see the Notes.
    voxel_size
        Lattice spacing for the ``"voxel"`` method. ``None`` uses 1/128 of the bounding-box
        diagonal.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_new_vertices,)`` vertices and ``(3 * n_new_faces,)`` faces, as ``(vertices, faces)``
        on ``vertices.device``. A clean input is returned as a copy.

    Raises
    ------
    ValueError
        If ``method`` is not ``"local"`` or ``"voxel"``, ``max_expand`` is negative, or ``max_iter``
        is less than 1.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Examples
    --------
    ```python
    clean_v, clean_f = od.repair.fix_self_intersections(v, f)
    ```

    Notes
    -----
    **Success is not guaranteed by either method, and neither is asserted.** For ``"local"``: a
    region whose rim cannot be triangulated without crossing something, or one that grows to swallow
    the mesh, leaves intersections behind; the loop stops at ``max_iter`` and returns what it has.
    For ``"voxel"``: the level set is clean, but Warp's ``IsoSurfaceMarchingCubes`` can emit a
    touching or non-manifold pair at an ambiguous cell, and that is resolution-dependent -- a finer
    lattice can introduce a handful of such faces where a coarser one has none. Check with
    [`ordito.validation.is_self_intersecting`][ordito.validation.is_self_intersecting] when it
    matters.

    **What the ``"local"`` method is for.** It clears a *shallow* self-intersection outright -- a
    torus whose tube passes through itself can be fully cleared at either dilation budget -- and
    only reduces a *deep* one, such as two icospheres overlapping by a third of their diameter. That
    is the method's shape rather than a tuning failure: cutting out a lens-shaped overlap leaves a
    rim whose minimum-weight patch runs back through the other shell, so the pass converges only
    where the damage is a band. Reach for ``"voxel"`` when two closed pieces genuinely
    interpenetrate -- a level set has no notion of two shells.

    **On that input class the result is nondeterministic and ``max_iter`` is not a quality knob.**
    Past a certain point, raising it does not reduce the residual intersections further while the
    face count climbs past the input's -- each pass refills a rim the next one cuts out again.
    Repeated runs on one input can also differ, from the refill chain's own atomic-ordering
    nondeterminism, so that variation is not a regression. Do not raise ``max_iter`` hoping for
    convergence on a deep interpenetration; the answer is ``"voxel"``.

    The two methods differ in what they preserve, not in quality. ``"local"`` keeps the input's
    triangulation everywhere it did not cut, so a per-vertex attribute survives outside the patch;
    ``"voxel"`` keeps nothing but the shape, and its accuracy is the lattice's.

    See Also
    --------
    [`ordito.validation.face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask]
        The detector, and what to check the result with.
    [`ordito.holes.refill_region`][ordito.holes.refill_region]
        The cut-and-refill step each ``"local"`` pass runs.
    [`ordito.levelset.offset_mesh`][ordito.levelset.offset_mesh]
        The same level-set machinery at a non-zero distance.
    """
    require_same_device(vertices=vertices, faces=faces)
    if method not in ("local", "voxel"):
        raise ValueError(f"method must be 'local' or 'voxel', got {method!r}")
    if max_expand < 0:
        raise ValueError("max_expand must be non-negative")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    if faces.size == 0:
        return _launch.clone(vertices), _launch.clone(faces)

    if method == "voxel":
        spacing = voxel_size
        if spacing is None:
            spacing = float(od.bounds.enclosing_diagonal(vertices)) / _VOXEL_REBUILD_RESOLUTION
        shape, box = od.proximity.signed_distance_lattice(vertices, spacing, pad=2)
        return od.levelset.signed_distance_level_set(
            vertices, faces, 0.0, shape, bounds=box, sign_mode="winding"
        )

    # The input buffers are only read until the first refill replaces them, so the copy a clean
    # input is owed is taken at the end rather than paid on every call.
    current_vertices, current_faces = vertices, faces
    for _ in range(max_iter):
        bad_mask = od.validation.face_self_intersecting_mask(current_vertices, current_faces)
        # Two readbacks per pass, and each decides the loop. Deliberately *not* ``od.reduce.any`` /
        # ``od.reduce.all``: a device reduction has a roughly fixed cost, while copying a ``bool``
        # array is one byte per element, so below the crossover -- about half a million elements --
        # the copy wins. Every mesh this runs on is well under it, so the readback stays; revisit at
        # a mesh past half a million faces.
        if not bool(bad_mask.numpy().any()):
            break
        region = _dilate_face_mask(current_faces, bad_mask, max_expand, current_vertices.size)
        if bool(region.numpy().all()):
            break  # the region swallowed the mesh: refilling it would delete everything
        current_vertices, current_faces = od.holes.refill_region(
            current_vertices, current_faces, region
        )
        if current_faces.size == 0:
            break
    if current_faces is faces:
        return _launch.clone(vertices), _launch.clone(faces)
    return current_vertices, current_faces


def _dilate_face_mask(
    faces: wp.array[wp.int32], face_mask: wp.array[wp.bool], hops: int, n_vertices: int
) -> wp.array[wp.bool]:
    """
    Grow a face selection by ``hops`` rings, through the vertices it touches.

    No adjacency is needed for this and none is built: a face ring is the faces incident on the
    selection's vertex ring, and on a triangle mesh two vertices are one-ring neighbours exactly
    when they share a face. So one hop is the any-corner face mask of the vertex mask, whose
    corners are then marked -- the same face-hop dilation
    [`ordito.selection.expand_vertex_mask`][ordito.selection.expand_vertex_mask] runs, kept in
    face form because the loop here wants the face mask of each ring, not the vertex mask.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    face_mask
        Length-``n_faces`` selection to grow.
    hops
        Rings to add. Zero returns the selection's own faces, which is *not* the input mask: it is
        every face sharing a vertex with it, since a cut has to leave a rim rather than a slit.
    n_vertices
        Length of the vertex buffer ``faces`` indexes, supplied by the caller. Not inferred from
        ``faces.max()``: that is a whole-buffer readback, and ``fix_self_intersections`` calls this
        once per pass, to recover a number the caller is already holding -- which is what
        ``face_adjacency(n_vertices=...)`` exists to avoid.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n_faces`` grown selection on ``faces.device``.
    """
    device = faces.device
    n_faces = faces.size // 3

    # Mask to mask throughout, so nothing is compacted and nothing is read back: the selection's
    # corners are marked by one pass over the faces, and each hop is an any-corner lookup into the
    # vertex mask followed by marking the corners it selected. A hop's two launches read and write
    # different buffers, so every hop grows from the previous hop's complete mask. ``grown`` is the
    # hops' face scratch as well as the answer: the last lookup writes every face.
    vertex_mask = _launch.zeros(n_vertices, dtype=wp.bool, device=device)
    grown = _launch.empty(n_faces, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_selection.mark_incident_vertices,
        dim=n_faces,
        inputs=[faces, face_mask, vertex_mask],
        device=device,
    )
    for _ in range(hops):
        _launch.launch(
            kernel_selection.face_mask_from_vertex_mask,
            dim=n_faces,
            inputs=[faces, vertex_mask, wp.bool(False), grown],
            device=device,
        )
        _launch.launch(
            kernel_selection.mark_incident_vertices,
            dim=n_faces,
            inputs=[faces, grown, vertex_mask],
            device=device,
        )
    _launch.launch(
        kernel_selection.face_mask_from_vertex_mask,
        dim=n_faces,
        inputs=[faces, vertex_mask, wp.bool(False), grown],
        device=device,
    )
    return grown


def remove_tunnels(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_length: float,
    *,
    metric: str = "plane_normalized",
    max_iter: int = 100,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], int]:
    """
    Remove thin handles by cutting along their short tunnel loops and sealing the two rims.

    A handle is a genus the surface did not need: a scanning artifact where two sheets fused, or a
    reconstruction that bridged across a gap. Its signature is a **short non-contractible loop** --
    short being the whole test, since every genus of the intended shape has loops the size of the
    shape. So: take a homology basis, shorten each loop within its class
    ([`shorten_loop`][ordito.geodesic_walk.shorten_loop]), keep the ones that come in under
    ``max_length``, cut along those and fill the boundary loops the cut opens. Cutting a surface
    along a non-separating cycle and sealing the two rims it creates drops the genus by exactly one,
    so ``2 * removed`` is the rise in
    [`euler_characteristic`][ordito.measures.euler_characteristic] -- verified rather than
    reported, and the invariant to assert if you extend this.

    Nothing is removed when no loop is short enough, and the input is returned unchanged -- so this
    is safe to run on a mesh whose genus is intended, provided ``max_length`` is below the scale of
    its real handles.

    !!! note "One disjoint pass per call"
        The loops kept are pairwise **vertex-disjoint**, shortest first. Cutting along two loops
        that cross is not the same operation as cutting along each in turn -- the shared vertex is
        split by both cuts at once -- and without the restriction the genus can stop dropping one
        per loop, or the surface can shatter into extra pieces. Disjoint loops can still be
        **dependent**, together bounding a piece of the surface that cutting along all of them
        would split off, so the longest loop of any such family is left uncut and the result stays
        one connected surface. The cost is that one call removes at most one tunnel per disjoint
        family, so a mesh whose basis loops all overlap needs to be run again. Call it in a loop
        until ``removed`` is ``0``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer. Must be a closed, connected,
        edge-manifold surface, which is what the homology basis needs.
    max_length
        Loops at or under this length are removed. It is an absolute length in the mesh's own
        units, so scale it off something intrinsic -- the mean edge length times the number of
        triangles a real handle would take to go round.
    metric
        Triangulation metric for the two rims, as
        [`fill_min_weight`][ordito.holes.fill_min_weight] takes it.
    max_iter
        Sweep cap handed to [`shorten_loop`][ordito.geodesic_walk.shorten_loop]. Shortening is what
        makes the length test meaningful: a tree-cotree loop around a thin handle can be many times
        the handle's own girth, so an unshortened basis under-reports every tunnel.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_new_vertices,)`` positions of the result. Longer than the input's wherever the cut
        split a vertex; the existing positions are unchanged and no new position is invented, since
        the rims are filled over their own vertices.
    faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat triangle index buffer, the cut mesh plus the fill triangles.
    removed : int
        How many loops were cut. Zero means nothing was short enough, and the buffers are the
        input's.

        Returned **unconditionally**, unlike the diagnostic counts on
        [`straighten_boundary`][ordito.repair.straighten_boundary] and
        [`remove_degree3_vertices`][ordito.repair.remove_degree3_vertices], which sit behind
        a ``return_count`` keyword. This one is part of the answer rather than a report on it: one
        call removes at most one tunnel per disjoint family, so the documented usage is to loop
        until it reads zero, and a caller who cannot see it cannot use the function correctly.
        [`remesh.intrinsic_delaunay`][ordito.remesh.intrinsic_delaunay]'s iteration count is
        unconditional for the same reason.

    Raises
    ------
    ValueError
        If ``max_length`` is negative, or the mesh has a boundary (a surface with boundary has a
        different homology basis, so "tunnel" is not defined by this test).
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`shorten_loop`][ordito.geodesic_walk.shorten_loop]
        Makes the length test meaningful, and is where the loops come from.
    [`homology_generators`][ordito.homology.homology_generators]
    [`fix_self_intersections`][ordito.repair.fix_self_intersections]
        The other topological repair here: that one removes crossings, this one removes genus.
    """
    require_same_device(vertices=vertices, faces=faces)
    if max_length < 0.0:
        raise ValueError(f"max_length must be non-negative, got {max_length}")
    # The basis stays packed from the generators through the shortening to the measuring, so no
    # per-loop array is built and split again on the way.
    loops, loop_offsets = od.homology.homology_generators_with_offsets(vertices, faces)
    if loop_offsets.size == 1:
        return vertices, faces, 0

    shortened, shortened_offsets, _sweeps = od.geodesic_walk.shorten_loop_with_offsets(
        vertices, faces, loops, loop_offsets, max_iter=max_iter
    )
    # A basis has 2 * genus loops of a handful of indices each, so everything after the measuring
    # runs on the host, over one readback of them all.
    lengths, loops_np = _measure_loops(vertices, shortened, shortened_offsets)
    short = sorted(zip(lengths, loops_np, strict=True), key=lambda pair: pair[0])
    selected = _disjoint_loops([loop for length, loop in short if length <= max_length])
    if not selected:
        return vertices, faces, 0

    # Disjoint is not independent: a family of disjoint non-trivial loops can still bound a piece
    # of the surface between them, and cutting along all of it splits that piece off instead of
    # dropping the genus by one per loop. The face labels of the cut mesh say which -- every
    # component after the first is one loop too many -- and they are read before cutting, so the
    # mesh is cut once, along the family that survives.
    labels, sides = _cut_face_labels(faces, selected)
    labels_np = labels.numpy()
    if np.unique(labels_np).size > 1:
        selected = _independent_loops(selected, labels_np, sides())
    cut_vertices, cut_faces = _cut_along_loops(vertices, faces, selected)
    cut_vertices = cast("wp.array[wp.vec3]", cut_vertices)
    return (
        cut_vertices,
        od.holes.fill_min_weight(cut_vertices, cut_faces, metric=metric),
        len(selected),
    )


def _measure_loops(
    vertices: wp.array[wp.vec3], packed: wp.array[wp.int32], offsets: wp.array[wp.int32]
) -> tuple[list[float], list[np.ndarray]]:
    """
    Measure every loop's closed length, and read every loop's indices back, in one pass each.

    The lengths are ``polyline_length(closed=True)`` of each loop's gathered positions, bit for
    bit: ``kernels/polyline.packed_closed_loop_lengths`` folds each loop the way that function folds
    a polyline of one block. The sort that follows keeps
    equal lengths in basis order, and a regular fixture has many, so any other summation order
    would change which loops are cut. A loop longer than one block is measured on its own.
    """
    device = vertices.device
    n_loops = offsets.size - 1
    lengths = _launch.empty(n_loops, dtype=wp.float32, device=device)
    _launch.launch_tiled(
        kernel_polyline.packed_closed_loop_lengths,
        dim=n_loops,
        inputs=[vertices, packed, offsets],
        outputs=[lengths],
        block_dim=TILE_1D,
        device=device,
    )
    lengths_list = [float(length) for length in lengths.numpy()]
    bounds_np = offsets.numpy().astype(np.int64)
    sizes_np = np.diff(bounds_np)
    for index in np.flatnonzero(sizes_np > int(kernel_reduce.ITEMS_PER_BLOCK_1D)):
        start, stop = int(bounds_np[index]), int(bounds_np[index + 1])
        lengths_list[index] = _cycle_length(vertices, odt.as_dense(packed[start:stop]))
    packed_np = packed.numpy()
    return lengths_list, [
        packed_np[start:stop]
        for start, stop in zip(bounds_np[:-1].tolist(), bounds_np[1:].tolist(), strict=True)
    ]


def _disjoint_loops(loops: list[np.ndarray]) -> list[np.ndarray]:
    """
    Greedily keep the loops that share no vertex, taking them shortest first.

    Cutting along two loops that *cross* is not the same operation as cutting along each in turn:
    the shared vertex is split by both cuts at once, and the genus can stop dropping by one per
    loop, or cutting the whole basis can shatter the surface into several pieces. Keeping the
    selection pairwise disjoint is what makes ``removed`` mean what it says.
    """
    claimed: set[int] = set()
    kept: list[np.ndarray] = []
    for loop in loops:
        loop_indices = {int(index) for index in loop}
        if loop_indices & claimed:
            continue
        claimed |= loop_indices
        kept.append(loop)
    return kept


def _cut_face_labels(
    faces: wp.array[wp.int32], loops: list[np.ndarray]
) -> tuple[wp.array[wp.int32], Callable[[], dict[tuple[int, int], tuple[int, int]]]]:
    """
    Label the faces as cutting along ``loops`` would leave them, without cutting.

    Cutting along an edge separates exactly the two faces across it and leaves every other face
    pair of an edge-manifold surface adjacent -- a pair across a non-loop edge at a loop vertex lies
    on one side of the loop there -- so the cut mesh's face adjacency is the input's with the loop
    edges' pairs removed. Severing those pairs in place of removing them keeps the face numbering,
    which the cut preserves too, so these labels are the cut mesh's own.

    Also returns a function reading back the severed pairs themselves, keyed by their ascending
    edge: the two faces across each loop edge, which telling a loop's two sides apart needs. It is
    a function because only a dependent family needs it, and it costs three readbacks.
    """
    device = faces.device
    adjacency, shared = od.adjacency.face_adjacency(faces, return_edges=True)
    # Each cycle's edges as ascending rows, which is what a cut keys on.
    cut_edges = np.concatenate(
        [np.sort(np.stack([loop, np.roll(loop, -1)], axis=1), axis=1) for loop in loops]
    ).astype(np.uint64)
    barrier_keys = _launch.array(
        np.sort(cut_edges[:, 0] + cut_edges[:, 1] * np.uint64(INDEX_RADIX_PAIR)),
        dtype=wp.uint64,
        device=device,
    )
    n_pairs = int(adjacency.shape[0])
    severed = odt.empty_2d((n_pairs, 2), wp.int32, device=device)
    barrier = _launch.empty(n_pairs, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_repair.sever_barrier_pairs,
        dim=n_pairs,
        inputs=[shared, barrier_keys, adjacency, severed, barrier],
        device=device,
    )
    labels = od.graph.connected_component_labels_from_edges(
        severed, node_count=faces.size // 3, validate=False
    )

    def sides() -> dict[tuple[int, int], tuple[int, int]]:
        # A loop edge is one pair of a few hundred; the readbacks are of the mesh-sized buffers
        # the table is picked out of.
        rows = np.flatnonzero(barrier.numpy())
        return {
            (int(a), int(b)): (int(f0), int(f1))
            for (a, b), (f0, f1) in zip(shared.numpy()[rows], adjacency.numpy()[rows], strict=True)
        }

    return labels, sides


def _independent_loops(
    loops: list[np.ndarray], labels_np: np.ndarray, sides: dict[tuple[int, int], tuple[int, int]]
) -> list[np.ndarray]:
    """
    Drop the fewest loops from a disjoint family so that cutting along the rest stays connected.

    ``labels_np`` are the face component labels after cutting along every loop in ``loops``. Each
    loop joins the component on its left to the one on its right, so the loops are the edges of a
    graph over those components, and leaving a loop uncut merges its two sides back together. The
    cut surface is connected exactly when the uncut loops span that graph, so a spanning forest
    built longest-first (Kruskal) leaves the longest loops uncut and keeps cutting the shortest --
    the ones most likely to be tunnels. A loop whose two sides already share a component is always
    kept. ``loops`` arrive shortest first, and the kept ones keep that order. ``sides`` maps each
    loop edge to the two faces across it, as ``_cut_face_labels`` returns them.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    uncut: set[int] = set()
    for index in range(len(loops) - 1, -1, -1):
        a, b = (int(v) for v in loops[index][:2])
        # The two faces across the loop's first edge: one on each side of the cut.
        face_a, face_b = sides[(min(a, b), max(a, b))]
        left, right = find(int(labels_np[face_a])), find(int(labels_np[face_b]))
        if left != right:
            parent[left] = right
            uncut.add(index)
    return [loop for index, loop in enumerate(loops) if index not in uncut]


def _cut_along_loops(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], loops: list[np.ndarray]
) -> tuple[wp.array[wp.vec3] | wp.array[wp.vec3d], wp.array[wp.int32]]:
    """Cut the mesh along the edges of a family of closed vertex-index cycles."""
    # Each cycle's edges as ascending rows, which is what a cut keys on.
    cut_edges = _launch.array(
        np.concatenate(
            [np.sort(np.stack([loop, np.roll(loop, -1)], axis=1), axis=1) for loop in loops]
        ),
        dtype=wp.int32,
        device=faces.device,
    )
    return od.seams.cut_along_edges(vertices, faces, odt.as_array2d(cut_edges, wp.int32))


def _cycle_length(vertices: wp.array[wp.vec3], loop: wp.array[wp.int32]) -> float:
    """Length of a closed vertex-index cycle, gathered onto its positions."""
    points = _launch.empty(loop.size, dtype=wp.vec3, device=vertices.device)
    _launch.copy(points, vertices[loop])
    return od.polyline.polyline_length(points, closed=True)


def flip_t_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    threshold: float = 40.0,
    max_iter: int = 10,
) -> wp.array[wp.int32]:
    """
    Repair T-vertices by flipping the long edge of each sliver they create.

    A **T-vertex** is a vertex that sits in the interior of a neighbouring triangle's edge rather
    than at one of its corners — the classic symptom of two patches stitched at different
    resolutions. The vertex is topologically fine, but the triangle opposite it is a sliver: its
    apex lies (nearly) on the far edge, which sends its circumradius-to-inradius ratio to infinity
    and makes every cotangent weight, normal and curvature estimate around it unusable.

    The repair is a flip, not a deletion: flipping the sliver's long edge moves the diagonal off the
    T and leaves two well-shaped triangles, with the same vertices and the same face count.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Never modified.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    threshold
        Aspect ratio above which a triangle counts as a T-vertex sliver, in the
        ``aspect_ratio`` sense of [`face_quality`][ordito.triangles.face_quality] (``1`` is
        equilateral, unbounded above). MeshLab's ``meshing_remove_t_vertices`` threshold, whose
        default of ``40`` is this one. Must be positive.
    max_iter
        Maximum number of parallel flip passes. Each pass commits a conflict-free independent set of
        flips; MeshLab's ``repeat=True`` is the same idea serially.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` flat face buffer with the slivers re-triangulated, on ``faces.device``
        (a copy; the input is not modified).

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`ordito.remesh.flip_by_objective`][ordito.remesh.flip_by_objective]
    [`remove_folded_faces`][ordito.repair.remove_folded_faces]
    [`collapse_small_triangles`][ordito.repair.collapse_small_triangles]

    Notes
    -----
    A flip cannot fix a T-vertex on the mesh **boundary** or on a non-manifold edge, because there
    is no second triangle to flip against. MeshLab offers an edge *collapse* method for that case;
    here the equivalent is [`collapse_small_triangles`][ordito.repair.collapse_small_triangles],
    which removes the sliver by merging its short edge instead.
    """
    require_same_device(vertices=vertices, faces=faces)
    return od.remesh.flip_by_objective(
        vertices, faces, objective="t_vertex", aspect_threshold=threshold, max_iter=max_iter
    )
