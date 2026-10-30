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
from typing import Literal, NamedTuple, cast, overload

import numpy as np
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, read_values, require_same_device, require_valid_faces
from ordito.adjacency import _face_component_forest  # pyright: ignore[reportPrivateUsage]
from ordito.constants import INDEX_RADIX_PAIR, TILE_1D
from ordito.grouping import (
    _unique_keys_core,  # pyright: ignore[reportPrivateUsage]
    hash_vector_rows,
)
from ordito.kernels import array as kernel_array
from ordito.kernels import polyline as kernel_polyline
from ordito.kernels import reduce as kernel_reduce
from ordito.kernels import repair as kernel_repair
from ordito.kernels import scatter as kernel_scatter
from ordito.kernels import selection as kernel_selection
from ordito.kernels.algorithms import connected_components as kernel_connected_components
from ordito.remesh import (
    _collapse_pass,  # pyright: ignore[reportPrivateUsage]
    _flip_region_faces,  # pyright: ignore[reportPrivateUsage]
)
from ordito.selection import _compact_marked_submesh  # pyright: ignore[reportPrivateUsage]
from ordito.validation import (
    _closed_signed_volume,  # pyright: ignore[reportPrivateUsage]
    _corner_parents_from_mates,  # pyright: ignore[reportPrivateUsage]
    _face_flip_words,  # pyright: ignore[reportPrivateUsage]
    _face_self_intersecting_mask,  # pyright: ignore[reportPrivateUsage]
)

# Lattice resolution for ``fix_self_intersections(method="voxel")``, in samples across the mesh's
# bounding-box diagonal. The same order as ``offset.offset_mesh``'s automatic floor, and the field
# it allocates is cubic in it; a caller who needs the surface resolved finer passes ``voxel_size``.
_VOXEL_REBUILD_RESOLUTION = 128

# Collapse passes per ``fix_degeneracies`` round: they stop early on one that commits nothing, so
# this only bounds a round.
_DEGENERACY_COLLAPSE_PASSES = 10
_DEGENERACY_COLLAPSE_ROUNDS = 2
# Delone flip passes per ``fix_degeneracies`` round, ``remesh.flip_to_delaunay``'s default.
_DEGENERACY_FLIP_ITERATIONS = 100

# ``remove_tunnels``' search through the basis vertices: the ball rows a root starts with, how much
# wider a root whose ball outgrew them is searched again, the scratch entries one launch may
# allocate across its roots (about 40 bytes each), and the relative slack on the kernel's own
# float32 length test, which the measured length then decides.
_TUNNEL_BALL_CAPACITY = 1024
_TUNNEL_BALL_GROWTH = 8
_TUNNEL_SCRATCH_ENTRIES = 1 << 22
_TUNNEL_LENGTH_SLACK = 1e-4


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
       the scan debris goes before anything expensive runs on it.
    2. ``join_components`` ->
       [`holes.join_closest_components`][ordito.holes.join_closest_components], for an input
       whose pieces are meant to be one surface rather than a largest piece plus rubbish. Mutually
       useful with ``keep_largest`` rather than exclusive: keep the big piece *and* weld what is
       left.
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

    Examples
    --------
    An icosphere with three faces missing and a speck of debris beside it comes back as one
    closed sphere:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    debris_vertices, debris_faces = od.creation.tetrahedron()
    broken_vertices = wp.array(
        np.concatenate([vertices.numpy(), 0.1 * debris_vertices.numpy() + 3.0]), dtype=wp.vec3
    )
    broken_faces = wp.array(
        np.concatenate([faces.numpy()[9:], debris_faces.numpy() + vertices.shape[0]]),
        dtype=wp.int32,
    )
    print(od.validation.is_watertight(broken_vertices, broken_faces))  # False
    print(len(od.boundary.boundary_loops(broken_vertices, broken_faces)))  # 3
    solid_vertices, solid_faces = od.repair.make_solid(broken_vertices, broken_faces)
    print(od.validation.is_watertight(solid_vertices, solid_faces))  # True
    print(od.measures.euler_characteristic(solid_faces))  # 2
    ```
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
    # silently rather than raising a Python exception.
    require_valid_faces(faces, vertices.size, "make_solid")

    # Stage 0. The reference does this inside its *loader*, which is why it is easy to leave out and
    # why leaving it out is dangerous: skipping it can leave a mesh at the right Euler
    # characteristic and one component while still **not watertight**, because nothing downstream
    # addresses a non-manifold edge.
    # ``make_winding_consistent`` then ``split_non_manifold_vertices`` over one pairing of the
    # halfedges: the flip carries the mates over to the wound faces, which the split reads. The
    # split keeps only the vertices a face names, so it is the unreferenced-vertex removal too: the
    # pairing and the winding read vertex indices only for equality, and the split numbers its
    # vertices by corner, so a compaction first would change nothing it returns. It also turns
    # the mates into twins, which pair the split faces into their components without a pairing of
    # their own for the component filter.
    wound, twins = _wound_with_mates(faces, vertices.size)
    vertices, faces, _source = _split_corners(vertices, wound, twins, twins)

    # Every stage below is a deterministic function of the mesh it is handed, and ``version`` counts
    # the changes made to the mesh, so a stage is skipped while the mesh is still one it is known
    # to leave alone: the output of the last run of an idempotent stage (degeneracy removal, the
    # collapse to a fixpoint, the largest-component filter), or a mesh the cut-and-refill found
    # nothing to cut in. Each stage reports a change exactly: the removers only drop faces and
    # compact the vertices those keep, so equal counts are an identical mesh; the fill only adds
    # faces; the cut-and-refill returns its own input when it cuts nothing.
    version = 0
    clean_after = {"degenerate": -1, "collapse": -1, "cut": -1, "components": -1}

    def counts() -> tuple[int, int]:
        return vertices.size, faces.size

    if keep_largest:
        vertices, faces = _remove_small_components(vertices, faces, mates=twins)
        clean_after["components"] = version
    if join_components:
        faces = od.holes.join_closest_components(vertices, faces)
        version += 1

    # The fill finds the rims itself and returns the faces unchanged (a copy) when there is none, so
    # no separate boundary test runs first.
    before = counts()
    # A component filter has compacted the faces, so their indices are below the vertex count by
    # construction and the fill needs no reduction to bound them; the caller's own faces keep it,
    # as the guard against a negative index.
    faces = od.holes.fill_min_weight(
        vertices, faces, n_vertices=vertices.size if keep_largest else None
    )
    version += int(counts() != before)
    for _ in range(max_iter):
        n_faces_before = faces.size
        for name, remover in (
            ("degenerate", remove_degenerate_faces),
            ("collapse", collapse_small_triangles),
        ):
            if clean_after[name] != version:
                before = counts()
                vertices, faces = remover(vertices, faces)
                version += int(counts() != before)
                clean_after[name] = version
        # The collapse runs on the degeneracy pass's output and keeps only faces passing the same
        # test (its remap applies the predicate on the positions it keeps), so what it returns is
        # also a mesh that pass leaves alone.
        if clean_after["collapse"] == version:
            clean_after["degenerate"] = version
        if clean_after["cut"] != version:
            uncut = faces
            vertices, faces = _cut_and_refill(vertices, faces, 1, inner_iter)
            if faces is uncut:
                clean_after["cut"] = version
            else:
                version += 1
        # Cutting an intersecting band out can *disconnect* the surface, so the component filter has
        # to run again here and not only at the top: a single pass at the start cannot see a
        # component that did not exist yet.
        if keep_largest and clean_after["components"] != version:
            before = counts()
            vertices, faces = remove_small_components(vertices, faces, keep_largest=True)
            if counts() != before:
                # A subset of the faces on the same positions: what the three stages above found
                # nothing to do in still holds nothing for them (no degenerate face, none below the
                # collapse's threshold, which a smaller box only lowers, no crossing pair).
                for name in ("degenerate", "collapse", "cut"):
                    if clean_after[name] == version:
                        clean_after[name] = version + 1
                version += 1
            clean_after["components"] = version
        if faces.size == n_faces_before:
            break

    # Stage 4 opens a rim whenever it cuts an intersecting region out, so the last fill is not a
    # belt-and-braces repeat of stage 3 -- it is what makes the common case come back closed. And
    # nothing geometric may run *after* it: filling a small rim produces one sliver, degeneracy
    # removal deletes that sliver and reopens the rim, and the two then trade the same faces
    # forever -- so the fill goes last.
    before = counts()
    # The faces here are a compacting stage's output, or else the ones the first fill left: it
    # range-checked them whenever it filled a rim, and with none filled there is none here to fill.
    # Either way the bound is the vertex count.
    faces = od.holes.fill_min_weight(vertices, faces, n_vertices=vertices.size)
    version += int(counts() != before)
    if keep_largest and clean_after["components"] != version:
        vertices, faces = remove_small_components(vertices, faces, keep_largest=True)
    return vertices, faces


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

    Examples
    --------
    A stray vertex no face uses is dropped and maps to ``-1``:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    padded = wp.array(np.vstack([vertices.numpy(), [5.0, 5.0, 5.0]]), dtype=wp.vec3)
    new_vertices, new_faces, remap = od.repair.remove_unreferenced_vertices(padded, faces)
    print(padded.shape, new_vertices.shape)  # (9,) (8,)
    print(remap.numpy().tolist())  # [0, 1, 2, 3, 4, 5, 6, 7, -1]
    ```
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
        ``(n_unique,)`` deduplicated vertex positions (first occurrence per equivalence class), in
        order of first occurrence -- the order of Open3D's ``remove_duplicated_vertices``, VTK's
        ``vtkCleanPolyData`` and MeshLab's -- so a mesh with no duplicates comes back unchanged.
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

    Examples
    --------
    A cube stored as a triangle soup, three private vertices per face, is not watertight until its
    coincident corners are welded:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    soup_vertices = wp.array(vertices.numpy()[faces.numpy()], dtype=wp.vec3)
    soup_faces = wp.array(np.arange(36), dtype=wp.int32)
    print(od.validation.is_watertight(soup_vertices, soup_faces))  # False
    welded, _, inverse, welded_faces = od.repair.remove_duplicated_vertices(
        soup_vertices, soup_faces, epsilon=1e-6
    )
    print(soup_vertices.shape, welded.shape)  # (36,) (8,)
    print(od.validation.is_watertight(welded, welded_faces))  # True
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    # Each class's first occurrence comes with the inverse, read off the grouping's stable sort.
    inverse, unique_indices = _vertex_classes(vertices, epsilon)
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

    Examples
    --------
    The 36 corners of a cube's triangle soup fall into 8 classes:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    soup_vertices = wp.array(vertices.numpy()[faces.numpy()], dtype=wp.vec3)
    inverse = od.repair.duplicate_vertex_inverse(soup_vertices, 1e-6)
    print(inverse.shape, int(inverse.numpy().max()) + 1)  # (36,) 8
    ```
    """
    return _vertex_classes(vertices, epsilon)[0]


def _vertex_classes(
    vertices: wp.array[wp.vec3], epsilon: float
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Return each vertex's coincident-vertex class and each class's first vertex.

    Classes are numbered by first occurrence, so the first vertices ascend: the inverse
    ``unique_1d`` gives for the vertex keys, renumbered, and its
    [`first_occurrence_indices`][ordito.grouping.first_occurrence_indices] in that order.
    """
    device = vertices.device
    n = vertices.size
    if n == 0:
        empty = _launch.empty(0, dtype=wp.int32, device=device)
        return empty, empty
    # One key per vertex, at either tolerance: ``hash_vector_rows`` quantizes at ``epsilon > 0`` and
    # packs the relative float buckets at ``0`` -- the same key ``unique_rows`` would hash the
    # vertices to. The keys and the identity payload go into the leading halves of the sort's
    # double-width buffers, and the grouping reads each class's first occurrence off the stable
    # sort in the launch that writes the inverse, as ``unique_rows`` does.
    # The relative buckets (``epsilon`` not positive) need no bounds, so their keys are packed in
    # the launch that writes the identity rather than to a buffer of their own first.
    keys = _launch.empty(2 * n, dtype=wp.uint64, device=device)
    order = _launch.empty(2 * n, dtype=wp.int32, device=device)
    if epsilon > 0.0:
        _launch.launch(
            kernel_array.KEYS_AND_IDENTITY[wp.uint64],
            dim=n,
            inputs=[hash_vector_rows(vertices, epsilon=epsilon), keys, order],
            device=device,
        )
    else:
        _launch.launch(
            kernel_repair.bucket_vertex_keys_and_identity,
            dim=n,
            inputs=[vertices, keys, order],
            device=device,
        )
    inverse, first, _counts = _unique_keys_core(keys, order, None, False)
    # The classes come out in key order; renumber them by first occurrence. The sort's payload is
    # spent, so it takes the ranks, and each vertex's class is renumbered in place.
    ranks = _launch.view(order, 0, n)
    _launch.launch(
        kernel_repair.flag_class_first_occurrences,
        dim=n,
        inputs=[inverse, first, ranks],
        device=device,
    )
    _launch.array_scan(ranks, ranks, inclusive=True)
    first_ascending = _launch.empty(first.size, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.renumber_classes_by_first_occurrence,
        dim=n,
        inputs=[inverse, first, ranks],
        outputs=[inverse, first_ascending],
        device=device,
    )
    return inverse, first_ascending


def resolve_duplicated_faces(
    faces: wp.array[wp.int32],
    *,
    n_vertices: int | None = None,
    method: Literal["cancel", "igl", "open3d", "meshlab"] = "cancel",
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Resolve duplicated triangles, by default cancelling oppositely wound copies.

    A duplicate group is the faces sharing one vertex set. Under ``method="cancel"`` (the default)
    and ``"igl"`` a face counts ``+1`` when it is wound as the group's first face and ``-1`` when
    reversed, and the group's signed count decides:

    - equal positive and negative counts: remove all copies;
    - one extra positive copy: keep the first positively oriented face;
    - one extra negative copy: keep the first negatively oriented face;
    - otherwise (two or more surplus copies of one winding) the group is not orientable:
      ``"cancel"`` raises ``ValueError``, ``"igl"`` drops the whole group, as
      ``igl::resolve_duplicated_faces`` does.

    ``"open3d"`` and ``"meshlab"`` cancel nothing: each keeps one face per group, the first
    occurrence. Open3D's ``remove_duplicated_triangles`` groups by winding too, so a face and its
    reversed copy are two groups and both stay; MeshLab's ``meshing_remove_duplicate_faces`` groups
    by vertex set alone, so one of them stays (MeshLab keeps whichever its unstable sort puts last;
    this keeps the first).

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    n_vertices
        Optional exclusive bound on the vertex indices, typically the vertex count. Trusted, not
        checked: it spares the reduction that otherwise range-checks ``faces`` and infers the
        bound, and its host readback.
    method
        ``"cancel"`` (default), ``"igl"``, ``"open3d"`` or ``"meshlab"``: the rule above.

    Returns
    -------
    resolved_faces : wp.array[wp.int32]
        ``(3 * n_kept,)`` flat buffer of kept faces. Under ``"cancel"`` and ``"igl"`` in libigl's
        order: one per surviving group, the groups ascending lexicographically by their sorted
        corner indices. Under ``"open3d"`` and ``"meshlab"`` in input order, as both libraries
        return them.
    kept_indices : wp.array[wp.int32]
        ``(n_kept,)`` original face indices into the input ``faces`` buffer, in the same order.

    Raises
    ------
    ValueError
        If ``method`` is not one of the four names; under ``"cancel"``, if a duplicate group's
        signed count is not orientable, i.e. its positive and negative copies differ by more than
        one; or if ``n_vertices`` is ``None`` and a face index is negative.

    Examples
    --------
    A face and its reversed copy cancel, opening a hole; a third copy settles the pair in its own
    orientation:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    rows = faces.numpy().reshape(-1, 3)
    cancelled = wp.array(np.concatenate([rows, rows[:1, ::-1]]).ravel(), dtype=wp.int32)
    resolved, kept = od.repair.resolve_duplicated_faces(cancelled)
    print(resolved.shape[0] // 3, sorted(kept.numpy().tolist())[:3])  # 11 [1, 2, 3]
    tripled = wp.array(np.concatenate([rows, rows[:1, ::-1], rows[:1]]).ravel(), dtype=wp.int32)
    resolved, kept = od.repair.resolve_duplicated_faces(tripled)
    print(resolved.shape[0] // 3)  # 12
    ```

    Open3D's rule keeps both windings, MeshLab's one face of the pair:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    rows = faces.numpy().reshape(-1, 3)
    doubled = wp.array(np.concatenate([rows, rows[:1, ::-1]]).ravel(), dtype=wp.int32)
    print(od.repair.resolve_duplicated_faces(doubled, method="open3d")[1].shape[0])  # 13
    print(od.repair.resolve_duplicated_faces(doubled, method="meshlab")[1].shape[0])  # 12
    ```
    """
    if method not in ("cancel", "igl", "open3d", "meshlab"):
        raise ValueError(f"method must be 'cancel', 'igl', 'open3d' or 'meshlab', got {method!r}")
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        empty = _launch.empty(0, dtype=wp.int32, device=device)
        return empty, empty

    if n_vertices is None:
        # The range check and the key radix ``grouping.unique_faces`` takes without a bound.
        min_index, max_value = od.reduce.minmax(faces)
        if min_index < 0:
            raise ValueError(f"faces must be non-negative, got a minimum of {min_index}")
        n_vertices = max_value + 1
    if method in ("open3d", "meshlab"):
        return _first_occurrence_faces(faces, n_vertices, oriented=method == "open3d")
    # Orientation-free keys ordered as the sorted rows are lexicographically, with the identity
    # payload, sorted, and then each run of equal keys -- one duplicate group, its faces in
    # ascending index -- decided by the thread at its start: no unique table, inverse or per-group
    # statistics are written between the sort and the decision. Three sorted indices below
    # ``n_vertices`` pack below ``n_vertices ** 3``.
    keys = _launch.empty(2 * n_faces, dtype=wp.uint64, device=device)
    order = _launch.empty(2 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.pack_lexicographic_face_keys_and_order,
        dim=n_faces,
        inputs=[faces, wp.uint64(n_vertices), keys, order],
        device=device,
    )
    end_bit = max(1, (min(n_vertices**3, 1 << 64) - 1).bit_length())
    _launch.radix_sort_pairs(keys, order, count=n_faces, end_bit=end_bit)
    # The keep flags at the sorted positions with the error word right behind them, so one readback
    # takes the kept count and the error together; the decided face of each run start beside them,
    # in the payload's upper half, the sort's scratch until now.
    flags = _launch.zeros(n_faces + 1, dtype=wp.int32, device=device)
    keep = _launch.view(order, n_faces, 2 * n_faces)
    _launch.launch(
        kernel_repair.resolve_duplicate_runs,
        dim=n_faces,
        inputs=[faces, keys, order, n_faces, int(method == "igl")],
        outputs=[keep, flags],
        device=device,
    )
    # Scanned in place, a kept group's flag becomes its output row plus one, in ascending key order
    # (the reference's emission order); the last entry is the kept total.
    ranks = odt.as_dense(flags[:n_faces])
    _launch.array_scan(ranks, out_array=ranks, inclusive=True)
    n_kept, error_word = read_values(flags, n_faces - 1, 2)
    if error_word != 0:
        _raise_duplicate_group(faces, keys, order, n_faces - error_word)

    resolved = _launch.empty(3 * n_kept, dtype=wp.int32, device=device)
    kept_wp = _launch.empty(n_kept, dtype=wp.int32, device=device)
    if n_kept > 0:
        _launch.launch(
            kernel_repair.compact_kept_duplicate_runs,
            dim=n_faces,
            inputs=[faces, keep, ranks],
            outputs=[resolved, kept_wp],
            device=device,
        )
    return resolved, kept_wp


def _first_occurrence_faces(
    faces: wp.array[wp.int32], n_vertices: int, *, oriented: bool
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Keep the first face of each duplicate group, in input order: Open3D's and MeshLab's rule.

    ``oriented`` groups by winding (the corners rotated to start at the smallest), else by vertex
    set. The stable sort of the identity puts each group's first occurrence at its run's start; the
    flag there, scanned in input order, is the kept face's output row.
    """
    n_faces = faces.size // 3
    device = faces.device
    keys = _launch.empty(2 * n_faces, dtype=wp.uint64, device=device)
    order = _launch.empty(2 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.pack_face_keys_and_order,
        dim=n_faces,
        inputs=[faces, wp.uint64(n_vertices), int(oriented), keys, order],
        device=device,
    )
    end_bit = max(1, (min(n_vertices**3, 1 << 64) - 1).bit_length())
    _launch.radix_sort_pairs(keys, order, count=n_faces, end_bit=end_bit)
    ranks = _launch.zeros(n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.flag_first_sorted_members,
        dim=n_faces,
        inputs=[keys, order],
        outputs=[ranks],
        device=device,
    )
    _launch.array_scan(ranks, out_array=ranks, inclusive=True)
    # The kept count sizes both outputs.
    n_kept = int(read_scalar(ranks, index=-1))
    resolved = _launch.empty(3 * n_kept, dtype=wp.int32, device=device)
    kept = _launch.empty(n_kept, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.compact_scanned_faces_and_indices,
        dim=n_faces,
        inputs=[ranks, faces],
        outputs=[resolved, kept],
        device=device,
    )
    return resolved, kept


def _raise_duplicate_group(
    faces: wp.array[wp.int32], keys: wp.array[wp.uint64], order: wp.array[wp.int32], start: int
) -> None:
    """
    Raise ``resolve_duplicated_faces``' error for the group whose run starts at sorted ``start``.

    The group is numbered by its rank in key order, the lexicographic order of its sorted corners
    (the index ``np.unique`` of the sorted rows gives it), and its signed count is the run's, both
    recovered on the host: readbacks paid only on the way to the raise.
    """
    n_faces = faces.size // 3
    # Readbacks on the error path only: the group's rank and its members.
    keys_np = keys.numpy()[:n_faces]
    order_np = order.numpy()[:n_faces]
    rows_np = faces.numpy().reshape(-1, 3)
    group = int(np.count_nonzero(keys_np[1 : start + 1] != keys_np[:start]))
    stop = start + int(np.count_nonzero(keys_np[start:] == keys_np[start]))
    representative = rows_np[order_np[start]].tolist()
    rotations = [representative[k:] + representative[:k] for k in range(3)]
    signs = [1 if rows_np[f].tolist() in rotations else -1 for f in order_np[start:stop]]
    raise ValueError(
        f"resolve_duplicated_faces: non-orientable duplicate face group {group} "
        f"with signed count {sum(signs)}"
    )


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

    Examples
    --------
    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    with_sliver = wp.array(np.concatenate([faces.numpy(), [0, 1, 1]]), dtype=wp.int32)
    new_vertices, new_faces = od.repair.remove_degenerate_faces(vertices, with_sliver)
    print(with_sliver.shape[0] // 3, new_faces.shape[0] // 3)  # 13 12
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.clone(vertices), _launch.clone(faces)

    # The degeneracy test marks the survivors for the compaction in the pass that decides them.
    keep_mask = _launch.empty(n_faces, dtype=wp.bool, device=faces.device)
    ranks = _launch.zeros(n_faces + vertices.size, dtype=wp.int32, device=faces.device)
    _launch.launch(
        kernel_repair.mark_nondegenerate_faces,
        dim=n_faces,
        inputs=[vertices, faces, keep_mask, ranks],
        device=faces.device,
    )
    new_vertices, new_faces, _vertex_index, _ranks = _compact_marked_submesh(
        vertices, faces, keep_mask, ranks, keep_masked=True, with_index=False
    )
    return new_vertices, new_faces


def remove_non_manifold_faces(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_iter: int = 3,
    *,
    method: Literal["all", "meshlab"] = "all",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Remove faces touching a non-manifold (>2-incident) edge, leaving the mesh edge-manifold.

    Keeps only faces whose three edges are each used by at most two faces
    ([`edge_manifold_mask`][ordito.validation.edge_manifold_mask]). Dropping a face only lowers the
    share count of its own edges, and every edge used three or more times loses all its faces, so
    one pass leaves every edge used at most twice: a further pass would keep every face.

    ``method="meshlab"`` removes fewer faces: visiting the faces on a non-manifold edge from the
    smallest area up (ties by face index), it deletes a face while one of its edges still has more
    than two faces, so each such edge keeps its two largest faces where they do not conflict. A
    duplicated face loses one copy rather than both and its neighbour, and a fin on an edge goes
    alone. This is MeshLab's ``meshing_repair_non_manifold_edges(method="Remove Faces")``; the
    visit runs as rounds in which every face whose smaller rivals on its edges have been decided
    decides at once, so the number of rounds is the longest chain of faces of increasing area
    linked through shared non-manifold edges, one host readback each.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    max_iter
        Maximum number of removal passes. One pass already reaches the fixpoint, so every value
        from ``1`` gives the same answer; ``0`` removes nothing. Unread under
        ``method="meshlab"``, which runs until every face on a non-manifold edge is decided.
    method
        Which faces go.

        - ``"all"`` (default) — every face on a non-manifold edge.
        - ``"meshlab"`` — the smallest first, until each edge is manifold (see above).

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_new_vertices,)`` vertices still referenced after non-manifold faces are dropped,
        compacted from index zero; ``vertices`` itself, not a copy, when no face is dropped.
    new_faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat buffer of the surviving (edge-manifold) faces; ``faces``
        itself, not a copy, when no face is dropped.

    Raises
    ------
    ValueError
        If ``method`` is not ``"all"`` or ``"meshlab"``.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]
    [`remove_degenerate_and_non_manifold_faces`][ordito.repair.remove_degenerate_and_non_manifold_faces]
    [`edge_manifold_mask`][ordito.validation.edge_manifold_mask]

    Examples
    --------
    A fin glued to one edge of a cube gives that edge three faces; all three go:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    a, b = faces.numpy()[:2]
    fin_vertices = wp.array(np.vstack([vertices.numpy(), [0.0, 0.0, 2.0]]), dtype=wp.vec3)
    fin_faces = wp.array(np.concatenate([faces.numpy(), [a, b, 8]]), dtype=wp.int32)
    print(od.validation.is_edge_manifold(fin_faces))  # False
    new_vertices, new_faces = od.repair.remove_non_manifold_faces(fin_vertices, fin_faces)
    print(new_faces.shape[0] // 3, od.validation.is_edge_manifold(new_faces))  # 10 True
    ```

    ``method="meshlab"`` deletes the fin alone, the smallest face on that edge:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    a, b = faces.numpy()[:2]
    fin_vertices = wp.array(np.vstack([vertices.numpy(), [0.0, 0.0, 0.2]]), dtype=wp.vec3)
    fin_faces = wp.array(np.concatenate([faces.numpy(), [a, b, 8]]), dtype=wp.int32)
    _, new_faces = od.repair.remove_non_manifold_faces(fin_vertices, fin_faces, method="meshlab")
    print(new_faces.shape[0] // 3, od.validation.is_edge_manifold(new_faces))  # 12 True
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    if method not in ("all", "meshlab"):
        raise ValueError(f"method must be 'all' or 'meshlab', got {method!r}")
    if method == "meshlab":
        keep = _smallest_first_survivors(vertices, faces)
        if keep is None:
            return vertices, faces
        return od.selection.submesh_from_face_mask(vertices, faces, keep)
    # The radix of the edge keys; any bound above every index groups the edges identically, so the
    # vertex count serves every pass over a subset of the faces, as it does for
    # ``remove_degenerate_and_non_manifold_faces``, with no reduction and readback to infer one.
    survivors, keep = _edge_manifold_survivors(faces, None, vertices.size, max_iter)
    if keep is None:
        return vertices, faces
    return od.selection.submesh_from_face_mask(vertices, survivors, keep)


def _smallest_first_survivors(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.bool] | None:
    """
    Return the faces ``method="meshlab"`` keeps, or ``None`` when no edge is non-manifold.

    The halfedges of each non-manifold edge are a run of the sorted key order beginning at the
    edge's lowest halfedge, which the mates name; each round decides, then commits.
    """
    n_faces = faces.size // 3
    if n_faces == 0:
        return None
    device = faces.device
    mates, key_order = od.halfedge.halfedge_mates(faces, return_key_order=True)
    assert key_order is not None  # without a vertex bound the pairing sorts
    positions = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    run_lengths = _launch.zeros(3 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.non_manifold_edge_runs,
        dim=3 * n_faces,
        inputs=[mates, key_order],
        outputs=[positions, run_lengths],
        device=device,
    )
    states = _launch.empty(n_faces, dtype=wp.int32, device=device)
    areas = _launch.empty(n_faces, dtype=wp.float32, device=device)
    keep = _launch.empty(n_faces, dtype=wp.bool, device=device)
    undecided = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.mark_non_manifold_candidates,
        dim=n_faces,
        inputs=[vertices, faces, mates],
        outputs=[states, areas, keep, undecided],
        device=device,
    )
    # Readback: a mesh with no face on a non-manifold edge is returned as it came.
    if read_scalar(undecided) == 0:
        return None
    live = _launch.clone(run_lengths)
    decisions = _launch.empty(n_faces, dtype=wp.int32, device=device)
    # Each round decides at least the smallest undecided face, so ``n_faces`` rounds bound it.
    for _ in range(n_faces):
        _launch.launch(
            kernel_repair.decide_non_manifold_faces,
            dim=n_faces,
            inputs=[mates, key_order, positions, run_lengths, live, states, areas],
            outputs=[decisions, undecided],
            device=device,
        )
        _launch.launch(
            kernel_repair.commit_non_manifold_faces,
            dim=n_faces,
            inputs=[mates, decisions, states, live, keep],
            outputs=[undecided],
            device=device,
        )
        # Readback: the faces still waiting decide whether another round runs.
        if read_scalar(undecided) == 0:
            break
    return keep


def remove_degenerate_and_non_manifold_faces(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], max_iter: int = 3
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Drop degenerate faces, then faces on a non-manifold edge, compacting the vertices once.

    The same answer as [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces]
    followed by [`remove_non_manifold_faces`][ordito.repair.remove_non_manifold_faces], byte for
    byte. The manifold test reads only face indices and every compaction preserves the order of
    both the faces and the vertices, so the manifold pass runs over the input's own vertex
    numbering and the referenced vertices are compacted once, at the end.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer, every index in
        ``[0, n_vertices)``.
    max_iter
        Maximum number of non-manifold removal passes, as in
        [`remove_non_manifold_faces`][ordito.repair.remove_non_manifold_faces]: every value from
        ``1`` gives the same answer.

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

    Examples
    --------
    A cube with a fin on one edge and a zero-area face:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    a, b = faces.numpy()[:2]
    broken_vertices = wp.array(np.vstack([vertices.numpy(), [0.0, 0.0, 2.0]]), dtype=wp.vec3)
    broken_faces = wp.array(np.concatenate([faces.numpy(), [a, b, 8, 0, 1, 1]]), dtype=wp.int32)
    new_vertices, new_faces = od.repair.remove_degenerate_and_non_manifold_faces(
        broken_vertices, broken_faces
    )
    print(broken_faces.shape[0] // 3, new_faces.shape[0] // 3)  # 14 10
    print(od.validation.is_edge_manifold(new_faces))  # True
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    if faces.size == 0:
        return _launch.clone(vertices), _launch.clone(faces)
    # ``triangles.face_nondegenerate_mask`` with the flags its survivors are compacted from, in the
    # one pass.
    n_faces = faces.size // 3
    keep = _launch.empty(n_faces, dtype=wp.bool, device=faces.device)
    keep_flags = _launch.empty(n_faces, dtype=wp.int32, device=faces.device)
    _launch.launch(
        kernel_repair.nondegenerate_face_flags,
        dim=n_faces,
        inputs=[vertices, faces],
        outputs=[keep, keep_flags],
        device=faces.device,
    )
    survivors, keep = _edge_manifold_survivors(faces, (keep, keep_flags), vertices.size, max_iter)
    assert keep is not None
    return od.selection.submesh_from_face_mask(vertices, survivors, keep)


def _edge_manifold_survivors(
    faces: wp.array[wp.int32],
    prefilter: tuple[wp.array[wp.bool], wp.array[wp.int32]] | None,
    n_vertices: int,
    max_iter: int,
) -> tuple[wp.array[wp.int32], wp.array[wp.bool] | None]:
    """
    Run the edge-manifold removal over ``faces[keep]``, in index space.

    ``prefilter`` is ``keep`` with its ``0`` / ``1`` ``int32`` flags (consumed by the scan), or
    ``None`` for all of ``faces``.

    The surviving faces are always ``faces[keep]`` for the pair this returns (all of ``faces``
    when ``keep`` is ``None``), with ``keep`` ``None`` only when ``prefilter`` is and the pass
    removed no face. Vertex indices are never remapped, so a caller compacts once with
    [`submesh_from_face_mask`][ordito.selection.submesh_from_face_mask].

    One pass is the fixpoint whatever ``max_iter`` (at least one) allows: removing a face only
    lowers the share count of its own edges, and every edge shared by three or more faces loses
    all of them, so the survivors' edges are each shared at most twice and a second pass would
    keep every face. ``max_iter`` below one runs none.

    ``n_vertices`` is the edge-key radix: any bound above every index of ``faces``.
    """
    tested = faces
    keep = None
    if prefilter is not None:
        keep, keep_flags = prefilter
        # Readback: the kept count decides whether the prefilter dropped anything to compact out.
        n_kept = _scan_kept_count(keep_flags)
        if n_kept != faces.size // 3:
            tested = _compact_scanned_faces(faces, keep_flags, n_kept)
    n_faces = tested.size // 3
    if max_iter < 1 or n_faces == 0:
        return faces, keep
    mates = od.halfedge.halfedge_mates(tested, n_vertices)
    if keep is not None:
        # The caller compacts ``tested[manifold]`` whatever the pass kept, so no kept count is
        # needed: an all-``True`` mask over the prefilter's survivors is the same submesh as the
        # prefilter's own mask over ``faces``.
        manifold = od.validation.edge_manifold_mask(
            tested, n_vertices=n_vertices, validate=False, mates=mates
        )
        return tested, manifold
    # ``validation.edge_manifold_mask`` with the flags its survivors would be counted from, written
    # by the same per-face pass.
    manifold = _launch.empty(n_faces, dtype=wp.bool, device=tested.device)
    flags = _launch.empty(n_faces, dtype=wp.int32, device=tested.device)
    _launch.launch(
        kernel_repair.face_mate_manifold_flags,
        dim=n_faces,
        inputs=[mates, wp.bool(True)],
        outputs=[manifold, flags],
        device=tested.device,
    )
    # Readback: an edge-manifold input is returned as it came, uncompacted.
    if _scan_kept_count(flags) == n_faces:
        return faces, None
    return tested, manifold


def _scan_kept_count(flags: wp.array[wp.int32]) -> int:
    """Scan ``0`` / ``1`` face flags in place and return how many are set (one tail readback)."""
    _launch.array_scan(flags, out_array=flags, inclusive=True)
    return int(read_scalar(flags))


def _compact_scanned_faces(
    faces: wp.array[wp.int32], inclusive: wp.array[wp.int32], n_kept: int
) -> wp.array[wp.int32]:
    """Return the ``n_kept`` faces whose flags ``_scan_kept_count`` scanned, in face order."""
    kept = _launch.empty(3 * n_kept, dtype=wp.int32, device=faces.device)
    if n_kept > 0:
        _launch.launch(
            kernel_repair.compact_scanned_faces,
            dim=inclusive.size,
            inputs=[inclusive, faces, kept],
            device=faces.device,
        )
    return kept


def remove_small_components(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    keep_largest: bool = False,
    min_faces: int | None = None,
    min_area: float | None = None,
    min_diameter: float | None = None,
    labels: wp.array[wp.int32] | None = None,
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
        Minimum summed triangle area for a component to be kept. Under
        [`is_deterministic`][ordito.config.is_deterministic] each component's area is summed
        exactly, so a component at the threshold is kept or dropped the same way on every call.
    min_diameter
        Minimum bounding-box diagonal for a component to be kept.
    labels
        ``(n_faces,)`` face component labels of ``faces``, each a face id naming its component, as
        [`face_connected_component_labels`][ordito.adjacency.face_connected_component_labels]
        returns them
        ([`Trimesh.face_connected_component_labels`][ordito.mesh.Trimesh.face_connected_component_labels]),
        to spare the labelling. Trusted, not checked.

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
        If ``vertices``, ``faces`` and ``labels`` are not all on one device.

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

    Examples
    --------
    A small tetrahedron floating beside a sphere is dropped, whether by keeping the largest
    component or by a face-count threshold:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    debris_vertices, debris_faces = od.creation.tetrahedron()
    noisy_vertices = wp.array(
        np.concatenate([vertices.numpy(), 0.1 * debris_vertices.numpy() + 3.0]), dtype=wp.vec3
    )
    noisy_faces = wp.array(
        np.concatenate([faces.numpy(), debris_faces.numpy() + vertices.shape[0]]), dtype=wp.int32
    )
    print(noisy_vertices.shape, noisy_faces.shape[0] // 3)  # (166,) 324
    kept_vertices, kept_faces = od.repair.remove_small_components(
        noisy_vertices, noisy_faces, keep_largest=True
    )
    print(kept_vertices.shape, kept_faces.shape[0] // 3)  # (162,) 320
    kept_vertices, kept_faces = od.repair.remove_small_components(
        noisy_vertices, noisy_faces, min_faces=10
    )
    print(kept_vertices.shape, kept_faces.shape[0] // 3)  # (162,) 320
    ```
    """
    require_same_device(vertices=vertices, faces=faces, labels=labels)
    given = (keep_largest, min_faces is not None, min_area is not None, min_diameter is not None)
    if sum(given) != 1:
        raise ValueError(
            "pass exactly one of keep_largest, min_faces, min_area or min_diameter, "
            f"got {sum(given)}"
        )
    return _remove_small_components(
        vertices,
        faces,
        min_faces=min_faces,
        min_area=min_area,
        min_diameter=min_diameter,
        labels=labels,
    )


def _remove_small_components(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    min_faces: int | None = None,
    min_area: float | None = None,
    min_diameter: float | None = None,
    labels: wp.array[wp.int32] | None = None,
    mates: wp.array[wp.int32] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    [`remove_small_components`][ordito.repair.remove_small_components] past its checks.

    At most one ``min_*`` bound; with none it keeps the largest component. ``mates`` are halfedge
    mates of ``faces`` -- or any table whose pairs connect the same faces -- for the labelling
    when ``labels`` is not given, trusted.
    """
    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return vertices, faces

    counts_faces = min_area is None and min_diameter is None
    counts = None
    if labels is None and counts_faces:
        # The face-count criteria histogram the labels in the flatten that writes them.
        parents = _face_component_forest(faces, vertices.size, mates)
        labels = _launch.empty(n_faces, dtype=wp.int32, device=device)
        counts = _launch.zeros(n_faces, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_repair.flatten_and_count_labels,
            dim=n_faces,
            inputs=[parents, labels, counts],
            device=device,
        )
    elif labels is None:
        labels = od.adjacency.face_connected_component_labels(
            faces, n_vertices=vertices.size, mates=mates
        )

    # Each criterion accumulates one per-component table; the others stay ``None`` and are never
    # read. The table is then thresholded inside the submesh extraction's own marking pass, so no
    # separate keep-mask pass runs.
    statistic = corners = best = None
    statistic_sums = _launch.empty_group_sum_pass()
    if min_area is not None:
        criterion = kernel_repair.COMPONENT_MIN_AREA
        statistic, statistic_sums = _component_measures(vertices, faces, labels, "area")
    elif min_diameter is not None:
        criterion = kernel_repair.COMPONENT_MIN_DIAMETER
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
    else:
        if counts is None:
            counts = _launch.zeros(n_faces, dtype=wp.int32, device=device)
            _launch.launch(
                kernel_scatter.count_occurrences,
                dim=n_faces,
                inputs=[labels, counts],
                device=device,
            )
        if min_faces is not None:
            criterion = kernel_repair.COMPONENT_MIN_FACES
        else:
            criterion = kernel_repair.COMPONENT_LARGEST
            # ``-1`` is below every packed key, so the reduction needs no separate seeding pass and
            # the winning label never reaches the host -- the marking pass recomputes its key.
            best = _launch.full(1, -1, dtype=wp.int64, device=device)
            _launch.launch(
                kernel_repair.reduce_largest_group,
                dim=n_faces,
                inputs=[counts, best],
                device=device,
            )

    # ``wp.empty``: the marking pass writes every face's entry.
    keep = _launch.empty(n_faces, dtype=wp.bool, device=device)
    ranks = _launch.zeros(n_faces + vertices.size, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.mark_kept_components,
        dim=n_faces,
        inputs=[
            faces,
            labels,
            criterion,
            counts,
            statistic,
            statistic_sums,
            corners,
            best,
            int(min_faces) if min_faces is not None else 0,
            float(min_area if min_area is not None else (min_diameter or 0.0)),
            keep,
            ranks,
        ],
        device=device,
    )
    new_vertices, new_faces, _vertex_index, _ranks = _compact_marked_submesh(
        vertices, faces, keep, ranks, keep_masked=True, with_index=False
    )
    return new_vertices, new_faces


def split_non_manifold_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    mates: wp.array[wp.int32] | None = None,
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
    mates
        ``(3 * n_faces,)`` [`halfedge_mates`][ordito.halfedge.halfedge_mates] of ``faces`` when the
        caller holds them ([`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates]), to spare
        pairing the halfedges again. Trusted, not checked.

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
        If ``vertices``, ``faces`` and ``mates`` are not all on one device.

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
    rather than a per-vertex star walk: the half-edges are paired by their undirected edge, and a
    union-find joins the two corner pairs of every edge whose pair runs it both ways, as
    [`connected_component_labels_from_edges`][ordito.graph.connected_component_labels_from_edges]
    does for an edge list. ``igl::split_nonmanifold`` instead explodes the mesh to ``3 * n_faces``
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

    Examples
    --------
    Two cubes touching at one corner share a bowtie vertex; splitting it adds one vertex and keeps
    all 24 faces:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    lower_vertices, lower_faces = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    upper_vertices, upper_faces = od.creation.box(bounds=[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    vertices = wp.array(np.vstack([lower_vertices.numpy(), upper_vertices.numpy()]), dtype=wp.vec3)
    faces = wp.array(np.concatenate([lower_faces.numpy(), upper_faces.numpy() + 8]), dtype=wp.int32)
    vertices, _, _, faces = od.repair.remove_duplicated_vertices(vertices, faces, epsilon=1e-6)
    print(vertices.shape, od.validation.is_vertex_manifold(faces))  # (15,) False
    new_vertices, new_faces, source = od.repair.split_non_manifold_vertices(vertices, faces)
    print(new_vertices.shape, new_faces.shape[0] // 3)  # (16,) 24
    print(od.validation.is_vertex_manifold(new_faces))  # True
    ```
    """
    require_same_device(vertices=vertices, faces=faces, mates=mates)
    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0:
        empty_index = _launch.empty(0, dtype=wp.int32, device=device)
        return _launch.empty(0, dtype=wp.vec3, device=vertices.device), faces, empty_index

    if mates is None:
        mates = od.halfedge.halfedge_mates(faces, vertices.size)
    return _split_corners(vertices, faces, mates)


def _split_corners(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    mates: wp.array[wp.int32],
    out_twins: wp.array[wp.int32] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    [`split_non_manifold_vertices`][ordito.repair.split_non_manifold_vertices] over given mates.

    Nonempty ``faces`` and their [`halfedge_mates`][ordito.halfedge.halfedge_mates], which a
    caller that already paired the halfedges passes rather than sorting them again. ``out_twins``,
    ``(3 * n_faces,)`` and possibly ``mates`` itself, receives each halfedge's twin (``-1`` where
    none): mates of the returned faces as far as their face components go
    (``kernels/repair.scatter_split_corners``), for a caller labelling those next.
    """
    device = faces.device
    n_corners = faces.size // 3 * 3
    # Components over the corners, by union-find straight over the halfedge mates: a pair wound
    # both ways (the halfedge twins) names two joins of the corner graph, formed in the thread, and
    # every other halfedge a self-loop, so neither an edge table nor a link list is materialised.
    # Every root is its component's smallest corner, and a corner *is* a node, so a root's vertex
    # is the original its copy came from.
    parents = _corner_parents_from_mates(faces, mates, False, None, oriented=True)
    roots, root_ranks = _launch.carve(wp.int32, [n_corners, n_corners], device)
    _launch.launch(
        kernel_connected_components.ecl_flatten_root_flags,
        dim=n_corners,
        inputs=[parents, roots, root_ranks],
        device=device,
    )
    # Scanned in place, a root's flag becomes its rank plus one among the roots in ascending order,
    # which numbers the new vertices by their smallest corner; the total is the last entry, read
    # back because it sizes the outputs.
    _launch.array_scan(root_ranks, out_array=root_ranks, inclusive=True)
    n_new = int(read_scalar(root_ranks))
    source = _launch.empty(n_new, dtype=wp.int32, device=device)
    new_vertices = _launch.empty(n_new, dtype=wp.vec3, device=device)
    # The union-find is done with ``parents``, so it takes the new face buffer: corner ``h`` of the
    # flat layout is entry ``h``.
    new_faces = parents
    _launch.launch(
        kernel_repair.scatter_split_corners,
        dim=n_corners,
        inputs=[faces, vertices, roots, root_ranks, mates, new_faces, source, new_vertices],
        outputs=[out_twins],
        device=device,
    )
    return new_vertices, new_faces, source


def collapse_small_triangles(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    epsilon: float = 1e-6,
    *,
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
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
    bounds
        The box enclosing ``vertices`` when the caller holds it, typically
        [`Trimesh.bounds`][ordito.mesh.Trimesh.bounds]. Its diagonal scales ``epsilon``, so passing
        it spares the reduction and readback that measure the box again. Trusted, not checked.

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
    [`fix_degeneracies`][ordito.repair.fix_degeneracies]

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

    Examples
    --------
    Moving a grid vertex onto its neighbour crushes the two triangles between them; collapsing
    merges the pair:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(4, 4))
    positions = vertices.numpy()
    positions[6] = positions[5] + [1e-5, 0.0, 0.0]
    crushed = wp.array(positions, dtype=wp.vec3)
    new_vertices, new_faces = od.repair.collapse_small_triangles(crushed, faces)
    print(vertices.shape, faces.shape[0] // 3)  # (16,) 18
    print(new_vertices.shape, new_faces.shape[0] // 3)  # (15,) 16
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0 or vertices.size == 0:
        return _launch.clone(vertices), _launch.clone(faces)

    bbd = (
        od.bounds.enclosing_diagonal(vertices)
        if bounds is None
        else od.bounds.box_diagonal(*bounds)
    )
    min_dbl_area = wp.float32(2.0 * epsilon * bbd * bbd)

    current_vertices = vertices
    current_faces = faces
    max_iterations = faces.size  # bounded: each collapsing pass drops at least one face
    # Two small-face counters for the whole loop, never reset: slot 0 counts the input's small faces
    # as the first pass finds them, slot 1 the small faces each pass leaves behind, counted by the
    # remap that decides them. So the stopping test is one 4-byte read rather than a per-face flag
    # buffer reduced on the device first, a pass left none exactly when slot 1 did not move, and
    # the pass after the last collapse is never launched.
    n_small = _launch.zeros(2, dtype=wp.int32, device=device)
    n_left = odt.as_dense(n_small[1:])
    small_so_far = 0
    for iteration in range(max_iterations):
        n_current = current_faces.size // 3
        if n_current == 0:
            break

        pairs = odt.empty_2d((n_current, 2), wp.int32, device=device)
        _launch.launch(
            kernel_repair.small_triangle_collapse_edges,
            dim=n_current,
            inputs=[current_vertices, current_faces, min_dbl_area, pairs, n_small],
            device=device,
        )
        if iteration == 0 and int(read_scalar(n_small, 0)) == 0:
            break

        # Non-flagged faces emit a self-pair (i0, i0); these are self-loops that leave the
        # connected-components closure unchanged, so all rows can be passed without filtering.
        n_vertices = current_vertices.size
        # ``validate=False``: the pairs are vertex indices read straight out of the face buffer
        # this function is repairing, which every kernel around it already indexes unchecked.
        labels = od.graph.connected_component_labels_from_edges(
            pairs, node_count=n_vertices, validate=False
        )

        # Each label is its class's smallest vertex, so remapping every corner to its label points
        # it at the representative's own position, with no dense class numbering built first. One
        # pass remaps the corners, keeps the faces ``triangles.face_nondegenerate_mask`` would
        # keep, and marks them for the submesh extraction, which then keeps exactly the vertices the
        # surviving faces name -- only representatives -- in ascending index order, which is
        # ascending label order. ``wp.empty``: the pass writes every corner and every mask entry.
        remapped_faces = _launch.empty(current_faces.size, dtype=wp.int32, device=device)
        keep_mask = _launch.empty(n_current, dtype=wp.bool, device=device)
        ranks = _launch.zeros(n_current + n_vertices, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_repair.remap_mark_nondegenerate_faces,
            dim=n_current,
            inputs=[current_vertices, current_faces, labels, min_dbl_area],
            outputs=[remapped_faces, keep_mask, ranks, n_left],
            device=device,
        )
        current_vertices, current_faces, _vertex_index, _ranks = _compact_marked_submesh(
            current_vertices, remapped_faces, keep_mask, ranks, keep_masked=True, with_index=False
        )
        small_total = int(read_scalar(n_small, 1))
        if small_total == small_so_far:
            break
        small_so_far = small_total

    return current_vertices, current_faces


def fix_degeneracies(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_deviation: float,
    *,
    tiny_edge_length: float = 0.0,
    critical_aspect_ratio: float = 1e4,
    max_angle_change: float = 60.0,
    region_rings: int = 3,
    max_iter: int = 10,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Remove needle and cap triangles and tiny edges by local flips and collapses around them.

    A triangle is *degenerate* when its aspect ratio (circumradius over twice the inradius, the
    ``aspect_ratio`` of [`face_quality`][ordito.triangles.face_quality]) reaches
    ``critical_aspect_ratio``. The repair works in a **region**: the degenerate faces and those on
    an edge no longer than ``tiny_edge_length``, grown by ``region_rings`` rings of neighbouring
    faces. Inside it, two local moves alternate until no collapse remains:

    - **Flip** every edge that fails the Delaunay test, as
      [`flip_to_delaunay`][ordito.remesh.flip_to_delaunay] does, when the flip moves the surface
      by at most ``max_deviation`` and changes the dihedral angle across it by at most
      ``max_angle_change`` (a quad holding a degenerate face may exceed the angle). This is what
      fixes a *cap*, whose apex sits on its long edge.
    - **Collapse** every edge no longer than ``max_deviation``, shortest first, into the endpoint
      whose removal moves the surface least. This is what fixes a *needle*. A collapse that would
      leave a face of aspect ratio above 20 is skipped, unless a face it touches was already worse
      or its edge is no longer than ``tiny_edge_length``.

    The surface keeps its topology: a collapse that would pinch a handle or a sheet shut (the link
    condition) is skipped, as is one that would fold a face over unless a degenerate face is
    involved, and boundary vertices stay on the boundary.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Never modified.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    max_deviation
        Longest edge collapsed and largest surface move of a flip, in mesh units. A collapse keeps
        one endpoint in place, so it too moves the surface by at most this length.
    tiny_edge_length
        Edges no longer than this collapse whatever the shape they leave, and seed the region.
    critical_aspect_ratio
        Aspect ratio from which a triangle counts as degenerate (``1`` is equilateral).
    max_angle_change
        Largest change of dihedral angle, in **degrees**, a flip may make.
    region_rings
        Rings of faces the region extends past the faces that seed it.
    max_iter
        Cap on the alternating flip-and-collapse rounds.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        ``(n_new_vertices,)`` vertex positions, compacted from index zero when a collapse ran.
    faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat triangle index buffer.

    Raises
    ------
    ValueError
        If ``max_deviation``, ``tiny_edge_length`` or ``region_rings`` is negative,
        ``critical_aspect_ratio`` is below ``1``, or ``max_angle_change`` is outside ``[0, 180]``.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`collapse_small_triangles`][ordito.repair.collapse_small_triangles]
    [`flip_t_vertices`][ordito.repair.flip_t_vertices]
    [`ordito.remesh.flip_to_delaunay`][ordito.remesh.flip_to_delaunay]

    Notes
    -----
    The collapses of one pass are a conflict-free independent set, each neighbourhood's winner
    chosen by a hashed priority rather than by length, so they are applied together but never
    overlap; the sequence differs from a serial shortest-edge-first queue, and so does the
    triangulation it ends on.
    The region is fixed when the call starts and follows the surviving vertices, so the moves never
    spread past it. A degenerate triangle no move can fix (a cap on the boundary, a needle whose
    collapse would leave worse faces) is left in place.

    Examples
    --------
    Sliding a grid vertex up to its neighbour makes two needles; the fix collapses their short
    edge:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(4, 4))
    positions = vertices.numpy()
    positions[6] = positions[5] + [0.0, 1e-5, 0.0]
    needled = wp.array(positions, dtype=wp.vec3)
    new_vertices, new_faces = od.repair.fix_degeneracies(needled, faces, 1e-3)
    print(new_vertices.shape, new_faces.shape[0] // 3)  # (15,) 16
    print(f"{od.triangles.face_quality(new_vertices, new_faces).numpy().max():.1f}")  # 1.2
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    if max_deviation < 0.0:
        raise ValueError(f"max_deviation must be non-negative, got {max_deviation}")
    if tiny_edge_length < 0.0:
        raise ValueError(f"tiny_edge_length must be non-negative, got {tiny_edge_length}")
    if critical_aspect_ratio < 1.0:
        raise ValueError(f"critical_aspect_ratio must be at least 1, got {critical_aspect_ratio}")
    if not 0.0 <= max_angle_change <= 180.0:
        raise ValueError(f"max_angle_change must be in [0, 180] degrees, got {max_angle_change}")
    if region_rings < 0:
        raise ValueError(f"region_rings must be non-negative, got {region_rings}")

    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.clone(vertices), _launch.clone(faces)
    seeds = _launch.zeros(vertices.size, dtype=wp.bool, device=device)
    n_seeds = _launch.zeros(1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.mark_degeneracy_seeds,
        dim=n_faces,
        inputs=[
            vertices,
            faces,
            wp.float32(critical_aspect_ratio),
            wp.float32(tiny_edge_length),
            seeds,
            n_seeds,
        ],
        device=device,
    )
    # The one readback before any work: a mesh with nothing to repair costs one pass.
    if int(read_scalar(n_seeds, 0)) == 0:
        return _launch.clone(vertices), _launch.clone(faces)
    # The region is held per vertex, a face being inside when its three corners are: the collapse
    # passes compact it with the vertices, which a face mask would not survive.
    grown = od.selection.expand_vertex_mask(faces, seeds, region_rings)
    region = _launch.empty(vertices.size, dtype=wp.float32, device=device)
    _launch.map(kernel_repair.region_weight, grown, out=region)

    degeneracy = wp.vec3(tiny_edge_length, max_deviation, critical_aspect_ratio)
    # Only the boundary is a feature: a degenerate face's normal is noise, so a dihedral test would
    # freeze exactly the vertices that must move.
    feature = wp.float32(math.pi)
    current_vertices = vertices
    current_faces = faces
    # Collapses only ever remove faces, so one flag buffer sized for the input serves every round
    # through its leading entries.
    region_flags = _launch.empty(n_faces, dtype=wp.int32, device=device)
    for _ in range(max_iter):
        n_current = current_faces.size
        # The caller's buffer is copied before the first flips; every later one is this call's own
        # (a flip or collapse output) and is flipped in place.
        flipped = (
            _launch.empty(n_current, dtype=wp.int32, device=device)
            if current_faces is faces
            else current_faces
        )
        face_region = _launch.view(region_flags, 0, n_current // 3)
        _launch.launch(
            kernel_repair.faces_in_vertex_region,
            dim=n_current // 3,
            inputs=[current_faces, region],
            outputs=[flipped, face_region],
            device=device,
        )
        current_faces = flipped
        _flip_region_faces(
            current_vertices,
            current_faces,
            face_region,
            math.radians(max_angle_change),
            max_deviation,
            _DEGENERACY_FLIP_ITERATIONS,
            critical_aspect_ratio=critical_aspect_ratio,
        )
        current_vertices, current_faces, _incidence, kept_region = _collapse_pass(
            current_vertices,
            current_faces,
            region,
            0.0,
            feature,
            max_passes=_DEGENERACY_COLLAPSE_PASSES,
            degeneracy=degeneracy,
            rounds=_DEGENERACY_COLLAPSE_ROUNDS,
        )
        assert kept_region is not None
        region = kept_region
        # Each committed collapse removes a face, and flips never undo a flip fixpoint on unchanged
        # positions, so a round without one ends the repair.
        if current_faces.size == n_current:
            break
    if current_vertices is vertices:
        current_vertices = _launch.clone(vertices)
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

    Examples
    --------
    Deleting a rim triangle of a grid cuts a notch; one pass closes it again:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(5, 5))
    notched = wp.array(np.delete(faces.numpy().reshape(-1, 3), 1, axis=0).ravel(), dtype=wp.int32)
    straightened, added = od.repair.straighten_boundary(vertices, notched, return_count=True)
    print(notched.shape[0] // 3, straightened.shape[0] // 3, added)  # 31 32 1
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")
    device = faces.device
    n_vertices = vertices.size
    if n_vertices == 0 or faces.size == 0 or iterations == 0:
        return (faces, 0) if return_count else faces

    added = 0
    # One emit cursor for the whole loop, allocated zeroed and re-zeroed before each later pass
    # rather than reallocated: a four-byte buffer per pass is an allocation where a memset does.
    cursor = _launch.zeros(1, dtype=wp.int32, device=device)
    for iteration in range(iterations):
        if iteration > 0:
            _launch.zero_(cursor)
        n_faces = faces.size // 3
        n_halfedges = 3 * n_faces
        twins = od.halfedge.halfedge_twins(faces, n_vertices=n_vertices)
        # Keyed by halfedge, not by vertex: a bowtie rim vertex carries two rim loops and has no
        # single successor, and edge-manifoldness -- all `halfedge_twins` checks -- does not
        # exclude one. One slot per halfedge gives each loop its own links and one writer each.
        # Both link tables are halves of one buffer: one allocation and one fill.
        rim_links = _launch.full(2 * n_halfedges, -1, dtype=wp.int32, device=device)
        rim_next = odt.as_dense(rim_links[:n_halfedges])
        rim_prev = odt.as_dense(rim_links[n_halfedges:])
        # ``wp.empty``: the pass writes every halfedge's candidate flag.
        candidate = _launch.empty(n_halfedges, dtype=wp.bool, device=device)
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
                wp.float32(min_normal_dot),
                wp.float32(max_aspect_ratio),
                rim_next,
                rim_prev,
                candidate,
            ],
            device=device,
        )
        # One notch per boundary halfedge at most, and a pinched rim can carry more of those than
        # the mesh has vertices, so the bound is the halfedge count.
        new_faces = _launch.empty(3 * n_halfedges, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_repair.emit_straighten_faces,
            dim=n_halfedges,
            inputs=[
                faces,
                rim_next,
                rim_prev,
                candidate,
                cursor,
                new_faces.reshape((n_halfedges, 3)),
            ],
            device=device,
        )
        # One readback per pass, and it is the loop's own stopping test: how many notches the pass
        # actually closed is a device-side fact and a Python loop cannot branch on it otherwise.
        n_added = int(read_scalar(cursor))
        if n_added == 0:
            break
        # The emitted prefix goes straight behind the current faces in one exact-size buffer: one
        # allocation and two copies, where trimming the emit and then concatenating cost two of
        # each and a third copy.
        grown = _launch.empty(faces.size + 3 * n_added, dtype=wp.int32, device=device)
        _launch.copy(grown, faces, count=faces.size)
        _launch.copy(grown, new_faces, dest_offset=faces.size, count=3 * n_added)
        faces = grown
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

    Examples
    --------
    A cube face split into a three-triangle fan around a raised apex loses the apex and gets its
    triangle back:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    a, b, c = faces.numpy()[:3]
    apex = vertices.numpy()[[a, b, c]].mean(axis=0) + [-0.2, 0.0, 0.0]
    bumped_vertices = wp.array(np.vstack([vertices.numpy(), apex]), dtype=wp.vec3)
    fan = [a, b, 8, b, c, 8, c, a, 8]
    bumped_faces = wp.array(np.concatenate([fan, faces.numpy()[3:]]), dtype=wp.int32)
    print(bumped_vertices.shape, bumped_faces.shape[0] // 3)  # (9,) 14
    new_vertices, new_faces, removed = od.repair.remove_degree3_vertices(
        bumped_vertices, bumped_faces, return_count=True
    )
    print(new_vertices.shape, new_faces.shape[0] // 3, removed)  # (8,) 12 1
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    if max_iter < 0:
        raise ValueError(f"max_iter must be non-negative, got {max_iter}")
    device = faces.device
    n_vertices = vertices.size
    n_input = faces.size // 3
    if n_input == 0 or max_iter == 0:
        return (vertices, faces, 0) if return_count else (vertices, faces)
    # The documented edge-manifold check, on the input only: the twin defects of the halfedge
    # mates, counted inside pass 0's first launch and read back with that pass's counters,
    # rather than a twin table the pass never reads. Replacing an interior degree-3 fan by the one
    # triangle over its rim keeps every rim edge at two faces with the same orientation, so a pass
    # cannot make a valid mesh invalid and later passes do not re-prove it.
    input_mates = od.halfedge.halfedge_mates(faces, n_vertices)
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
        inputs=[faces, input_mates, counts, link_sums, fans, face_slots, kept, defects],
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

    Examples
    --------
    A cube face split into a three-triangle fan around a raised apex: the apex drops back into the
    face, and the cube's volume returns to 1:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    a, b, c = faces.numpy()[:3]
    apex = vertices.numpy()[[a, b, c]].mean(axis=0) + [-0.2, 0.0, 0.0]
    bumped_vertices = wp.array(np.vstack([vertices.numpy(), apex]), dtype=wp.vec3)
    fan = [a, b, 8, b, c, 8, c, a, 8]
    bumped_faces = wp.array(np.concatenate([fan, faces.numpy()[3:]]), dtype=wp.int32)
    flattened = od.repair.flatten_degree3_vertices(bumped_vertices, bumped_faces)
    print(np.round(flattened.numpy()[8], 4))  # [-0.5    -0.1667  0.1667]
    print(f"{od.measures.volume(bumped_vertices, bumped_faces):.4f}")  # 1.0333
    print(f"{od.measures.volume(flattened, bumped_faces):.4f}")  # 1.0000
    ```
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
        # device reduction costs. This call is over the *vertex* count, so
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
    in a fixed order. The example below prints the volumes rounded, since only that holds
    independently of the reduction.

    The corner that stays first is a convention, and this one matches ``np.fliplr`` -- which is
    what ``trimesh.Trimesh.invert`` applies, so the two agree elementwise rather than only up to a
    rotation of each row. The per-face flippers write the same reversal.

    See Also
    --------
    [`ordito.mesh.Trimesh.invert`][]
        The cached-mesh form, which carries what survives a flip.
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]
    [`make_normals_outward`][ordito.repair.make_normals_outward]
    [`ordito.validation.is_winding_consistent`][]

    Examples
    --------
    Reversing a cube turns it inside out, and its signed volume negates:

    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    flipped = od.repair.reverse_winding(faces)
    print(faces.numpy()[:3], flipped.numpy()[:3])  # [1 3 0] [0 3 1]
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 1.0000
    print(f"{od.measures.volume(vertices, flipped):.4f}")  # -1.0000
    ```
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
        [`face_flip_mask`][ordito.validation.face_flip_mask], which pairs faster with it. It does
        not change the answer, and it is trusted, not checked: it must exceed every index.

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
    with [`is_orientable`][ordito.validation.is_orientable] first if that matters. Under
    [`is_deterministic`][ordito.config.is_deterministic] the seam is a function of ``faces`` too
    ([`face_flip_mask`][ordito.validation.face_flip_mask] says which).

    Examples
    --------
    Every seventh face of an icosphere flipped, face 0 among them: the whole mesh follows face 0,
    so it comes out consistent but inside out:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    rows = faces.numpy().reshape(-1, 3)
    rows[::7] = rows[::7, ::-1]
    mixed = wp.array(rows.ravel(), dtype=wp.int32)
    print(od.validation.is_winding_consistent(mixed))  # False
    consistent = od.repair.make_winding_consistent(mixed)
    print(od.validation.is_winding_consistent(consistent))  # True
    print(f"{od.measures.volume(vertices, consistent):.3f}")  # -4.047
    ```
    """
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.int32, device=device)

    # The flip mask's parity words, solved straight off the halfedge mates with no adjacency table
    # and no host read of its length -- the bits ``face_orientation_bits`` gives -- and read by the
    # flip itself, so the mask is never stored.
    words = _face_flip_words(faces, n_vertices, None)
    out_faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.flip_faces_by_parity,
        dim=n_faces,
        inputs=[faces, words, out_faces],
        device=device,
    )
    return out_faces


def make_volume(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    multibody: bool = False,
    mates: wp.array[wp.int32] | None = None,
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
        whole. Under [`is_deterministic`][ordito.config.is_deterministic] each component's volume
        is summed exactly, so a component of near-zero volume is flipped or kept the same way on
        every call.
    mates
        ``(3 * n_faces,)`` [`halfedge_mates`][ordito.halfedge.halfedge_mates] of ``faces`` when the
        caller holds them ([`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates]), to spare
        pairing the halfedges again. Trusted, not checked.

    Returns
    -------
    wp.array[wp.int32]
        ``(3 * n_faces,)`` new flat face buffer with outward-oriented normals, on
        ``faces.device``. Vertices are unchanged.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``mates`` are not all on one device.

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

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    inside_out = od.repair.reverse_winding(faces)
    print(od.validation.is_volume(vertices, inside_out))  # False
    fixed = od.repair.make_volume(vertices, inside_out)
    print(od.validation.is_volume(vertices, fixed))  # True
    print(f"{od.measures.volume(vertices, fixed):.3f}")  # 4.047
    ```
    """
    require_same_device(vertices=vertices, faces=faces, mates=mates)
    return _outward_faces(vertices, faces, multibody=multibody, owned=False, mates=mates)


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

    Examples
    --------
    Every seventh face of an icosphere flipped comes back consistent and facing out:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    rows = faces.numpy().reshape(-1, 3)
    rows[::7] = rows[::7, ::-1]
    mixed = wp.array(rows.ravel(), dtype=wp.int32)
    print(od.validation.is_winding_consistent(mixed))  # False
    outward = od.repair.make_normals_outward(vertices, mixed)
    print(od.validation.is_winding_consistent(outward))  # True
    print(f"{od.measures.volume(vertices, outward):.3f}")  # 4.047
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    if faces.size // 3 == 0:
        return _launch.empty(0, dtype=wp.int32, device=faces.device)
    # ``make_winding_consistent`` carrying the halfedge mates over to the wound faces, which the
    # orientation test reads rather than pairing the halfedges again. ``wound`` is this call's own
    # buffer, so a mesh already facing outward returns it as it is.
    wound, wound_mates = _wound_with_mates(faces, vertices.size)
    return _outward_faces(vertices, wound, multibody=multibody, owned=True, mates=wound_mates)


def _wound_with_mates(
    faces: wp.array[wp.int32], n_vertices: int
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    [`make_winding_consistent`][ordito.repair.make_winding_consistent]'s faces, with their mates.

    The [`halfedge_mates`][ordito.halfedge.halfedge_mates] of the wound faces, moved with each
    flipped face (``kernels/repair.flip_faces_and_mates_by_parity``) from the one pairing the
    winding fix sorts anyway, for a caller that pairs the wound halfedges next. A ``-2 - lowest``
    code may name another halfedge of its edge than the lowest. Nonempty ``faces``.
    """
    n_faces = faces.size // 3
    mates = od.halfedge.halfedge_mates(faces, n_vertices)
    wound = _launch.empty(3 * n_faces, dtype=wp.int32, device=faces.device)
    wound_mates = _launch.empty(3 * n_faces, dtype=wp.int32, device=faces.device)
    _launch.launch(
        kernel_repair.flip_faces_and_mates_by_parity,
        dim=n_faces,
        inputs=[faces, mates, _face_flip_words(faces, n_vertices, mates)],
        outputs=[wound, wound_mates],
        device=faces.device,
    )
    return wound, wound_mates


def _outward_faces(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    multibody: bool,
    owned: bool,
    mates: wp.array[wp.int32] | None = None,
) -> wp.array[wp.int32]:
    """
    [`make_volume`][ordito.repair.make_volume]'s answer, without its device check.

    ``owned`` says ``faces`` is a buffer the caller allocated and hands over: where the faces need
    no flip it is returned itself rather than cloned. ``mates`` are ``faces``' halfedge mates when
    the caller holds them (``-2`` codes need not name the lowest halfedge).
    """
    n_faces = faces.size // 3
    device = faces.device
    if n_faces == 0:
        return _launch.empty(0, dtype=wp.int32, device=device)

    if multibody:
        out_faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
        labels = od.adjacency.face_connected_component_labels(
            faces, n_vertices=vertices.size, mates=mates
        )
        volumes, volume_sums = _component_measures(vertices, faces, labels, "volume")
        # One launch: each face reads its own component's signed volume through its label, where
        # gathering that into a per-face flag buffer first cost a map, a launch and the buffer.
        _launch.launch(
            kernel_repair.flip_faces_by_component_sign,
            dim=n_faces,
            inputs=[faces, labels, volumes, volume_sums, out_faces],
            device=device,
        )
        return out_faces

    # "Is every undirected edge shared by exactly two faces" -- ``is_edge_manifold`` with
    # ``allow_boundary_edges=False`` -- is "does every halfedge have a mate", counted over the mates
    # in the launch that folds the volume (``measures.volume`` bit for bit) and read back with it.
    if mates is None:
        mates = od.halfedge.halfedge_mates(faces, vertices.size)
    volume, closed = _closed_signed_volume(vertices, faces, mates, None, False)
    if not closed or volume >= 0.0:
        return faces if owned else _launch.clone(faces)

    out_faces = _launch.empty(3 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.flip_all_faces, dim=n_faces, inputs=[faces, out_faces], device=device
    )
    return out_faces


def _component_measures(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    measure: Literal["area", "volume"],
) -> tuple[wp.array[wp.float32] | None, object]:
    """
    Each component's summed face measure (``kernels/repair.face_group_measure``), slot per label.

    The accumulator and the ``kernels/reduce.GroupSumPass`` a consumer kernel reads each sum
    through (``kernels/reduce.read_group_sum``): ``(n_faces,)`` slots, read only at the slots
    ``labels`` names (a label is a representative face index). Each face's measure is summed into
    its component where it is computed, rather than written to a per-face buffer for a scatter
    pass to read back. With [`is_deterministic`][ordito.config.is_deterministic] off this is one
    ``float32`` atomic scatter in arrival order; on, each component is summed exactly in fixed
    point (``_launch.group_sums``), so the sums and the sign and threshold decisions read from
    them repeat bit for bit on CUDA, where a component's last bits otherwise move from call to
    call (a component of near-zero volume then flips its orientation). There the consumer reads
    the words, so no finish launch runs and the accumulator is ``None``.
    """
    n_faces = faces.size // 3
    device = faces.device
    # A component can hold every face, so ``n_faces`` terms bound any one sum.
    sums = _launch.group_sums(n_faces, wp.float32, n_faces, device, fused=True)
    _launch.launch_group_sums(
        kernel_repair.scatter_face_measure_by_group,
        dim=n_faces,
        inputs=[vertices, faces, labels, _GROUP_MEASURES[measure]],
        sums=sums,
        device=device,
    )
    return sums.out, sums.launch_pass


# Each measure's selector of ``kernels/repair.face_group_measure``.
_GROUP_MEASURES = {
    "area": kernel_repair.GROUP_MEASURE_AREA,
    "volume": kernel_repair.GROUP_MEASURE_VOLUME,
}


def remove_folded_faces(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    angle: float = 160.0,
    mates: wp.array[wp.int32] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Drop faces that fold back over their own ring, and reindex.

    A fold is a pair of neighbouring faces whose dihedral angle is near ``pi``: the two triangles
    lie almost on top of each other with opposite normals, which is what a badly reconstructed or
    self-intersecting patch looks like locally. Of the two, the face dropped is the one whose
    normal points against its neighbours' summed normals -- the one turned back over its ring; the
    other agrees with its surroundings and is kept. Such a face contributes no surface and breaks
    every normal-based computation downstream, so removing it is a repair rather than a
    simplification.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    angle
        Dihedral threshold in **degrees**; a face with a neighbour above it is dropped when it also
        faces against its neighbours (see above). MeshLab's
        ``folded_faces_angle_threshold``, whose default of ``160`` is this one. Must be in
        ``(0, 180]``.
    mates
        ``(3 * n_faces,)`` [`halfedge_mates`][ordito.halfedge.halfedge_mates] of ``faces`` when the
        caller holds them ([`Trimesh.halfedge_mates`][ordito.mesh.Trimesh.halfedge_mates]), to spare
        pairing the halfedges again. Trusted, not checked.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_new_vertices,)`` vertices still referenced by a kept face, compacted from index zero.
    new_faces : wp.array[wp.int32]
        ``(3 * n_new_faces,)`` flat buffer of the kept faces, remapped into ``new_vertices``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``mates`` are not all on one device.
    ValueError
        If ``angle`` is not in ``(0, 180]``.

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

    Examples
    --------
    Dragging a grid vertex past its neighbour turns two triangles upside down; both are dropped:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(5, 5))
    positions = vertices.numpy()
    positions[6] += [0.3, 0.0, 0.0]
    folded = wp.array(positions, dtype=wp.vec3)
    normals, _ = od.triangles.face_normals_and_areas(folded, faces)
    print(int((normals.numpy()[:, 2] < 0.0).sum()))  # 2
    new_vertices, new_faces = od.repair.remove_folded_faces(folded, faces)
    print(faces.shape[0] // 3, new_faces.shape[0] // 3)  # 32 30
    ```
    """
    require_same_device(vertices=vertices, faces=faces, mates=mates)
    if not 0.0 < angle <= 180.0:
        raise ValueError(f"angle must be in (0, 180] degrees, got {angle}")
    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return _launch.clone(vertices), _launch.clone(faces)
    if mates is None:
        mates = od.halfedge.halfedge_mates(faces, vertices.size)
    # The fold decision and the submesh extraction's marking pass in one launch; the keep mask it
    # writes is what the compaction reads.
    keep = _launch.empty(n_faces, dtype=wp.bool, device=device)
    ranks = _launch.zeros(n_faces + vertices.size, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.mark_unfolded_faces,
        dim=n_faces,
        inputs=[vertices, faces, None, mates, wp.float32(math.radians(angle)), keep, ranks],
        device=device,
    )
    new_vertices, new_faces, _vertex_index, _ranks = _compact_marked_submesh(
        vertices, faces, keep, ranks, keep_masked=True, with_index=False
    )
    return new_vertices, new_faces


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

    [`ordito.validation.face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask]
    detects a self-intersection; this is the repair, in the two forms that exist:

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

    Notes
    -----
    **Success is not guaranteed by either method, and neither is asserted.** For ``"local"``: a
    region whose rim cannot be triangulated without crossing something, or one that grows to swallow
    the mesh, leaves intersections behind; the loop stops at ``max_iter`` and returns what it has.
    For ``"voxel"``: the level set is clean, but marching cubes can emit a touching or non-manifold
    pair at an ambiguous cell, and that is resolution-dependent -- a finer lattice can introduce a
    handful of such faces where a coarser one has none. Check with
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

    Examples
    --------
    An icosphere with one vertex pulled through to the far side is cut and refilled locally:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    positions = vertices.numpy()
    positions[0] *= -1.25
    spiked = wp.array(positions, dtype=wp.vec3)
    print(od.validation.is_self_intersecting(wp.Mesh(points=spiked, indices=faces)))  # True
    clean_vertices, clean_faces = od.repair.fix_self_intersections(spiked, faces)
    clean_mesh = wp.Mesh(points=clean_vertices, indices=clean_faces)
    print(od.validation.is_self_intersecting(clean_mesh))  # False
    print(od.validation.is_watertight(clean_vertices, clean_faces))  # True
    ```

    Two interpenetrating spheres need the ``"voxel"`` rebuild, which merges them into one shell:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    shifted = vertices.numpy() + [1.0, 0.0, 0.0]
    pair_vertices = wp.array(np.vstack([vertices.numpy(), shifted]), dtype=wp.vec3)
    pair_faces = wp.array(
        np.concatenate([faces.numpy(), faces.numpy() + vertices.shape[0]]), dtype=wp.int32
    )
    pair_mesh = wp.Mesh(points=pair_vertices, indices=pair_faces)
    print(od.validation.is_self_intersecting(pair_mesh))  # True
    merged_vertices, merged_faces = od.repair.fix_self_intersections(
        pair_vertices, pair_faces, method="voxel", voxel_size=0.1
    )
    print(od.validation.is_watertight(merged_vertices, merged_faces))  # True
    ```
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
    # input is owed is taken here rather than paid on every pass.
    new_vertices, new_faces = _cut_and_refill(vertices, faces, max_expand, max_iter)
    if new_faces is faces:
        return _launch.clone(vertices), _launch.clone(faces)
    return new_vertices, new_faces


def _cut_and_refill(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], max_expand: int, max_iter: int
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    [`fix_self_intersections`][ordito.repair.fix_self_intersections]' ``"local"`` passes.

    Returns ``vertices`` and ``faces`` themselves, not copies, when nothing intersects, so a caller
    tells "unchanged" by identity: [`make_solid`][ordito.repair.make_solid] then knows the mesh is
    one this pass leaves alone, and skips it while nothing else has changed the mesh.
    """
    current_vertices, current_faces = vertices, faces
    # Two decision words per pass, each raised on the device by the launch that produced its mask
    # (the detector's last launch, the dilation's last lookup): a four-byte read each rather than a
    # copy of the whole face mask. A word is raised to the pass's stamp, so neither is reset
    # between passes.
    state = _launch.zeros(2, dtype=wp.int32, device=vertices.device)
    for pass_index in range(max_iter):
        stamp = pass_index + 1
        bad_mask = _face_self_intersecting_mask(
            current_vertices, current_faces, any_marked=(stamp, state)
        )
        if int(read_scalar(state, 0)) != stamp:
            break
        region = _dilate_face_mask(
            current_faces, bad_mask, max_expand, current_vertices.size, state, stamp
        )
        if int(read_scalar(state, 1)) != stamp:
            break  # the region swallowed the mesh: refilling it would delete everything
        current_vertices, current_faces = od.holes.refill_region(
            current_vertices, current_faces, region
        )
        if current_faces.size == 0:
            break
    return current_vertices, current_faces


def _dilate_face_mask(
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    hops: int,
    n_vertices: int,
    state: wp.array[wp.int32] | None = None,
    stamp: int = 1,
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
    state
        ``(2,)`` decision words; slot 1 is raised to ``stamp`` when some face stays outside the
        grown selection. ``None`` raises nothing.
    stamp
        Value slot 1 of ``state`` is raised to.

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
    if state is None:
        _launch.launch(
            kernel_selection.face_mask_from_vertex_mask,
            dim=n_faces,
            inputs=[faces, vertex_mask, wp.bool(False), grown],
            device=device,
        )
    else:
        _launch.launch(
            kernel_repair.face_mask_from_vertex_mask_flag_unselected,
            dim=n_faces,
            inputs=[faces, vertex_mask, stamp, grown, state],
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

    The basis is the greedy system of loops through a root vertex
    ([`homology_generators`][ordito.homology.homology_generators]), so a handle near the root comes
    out as its own short loop. One far from the root can come out only inside a combination with
    other classes, which no shortening within the class can separate again; so when no loop of that
    basis is short enough, every vertex of the shortened basis is searched for the shortest
    non-separating loop through it, along mesh edges and within ``max_length / 2`` of it. Every
    non-separating loop crosses the basis, so this search finds the shortest one on the surface:
    a call returns ``0`` only when **no** non-separating edge loop is within ``max_length``, and
    every handle whose girth along the edges is under ``max_length`` is removed by calling it until
    then. The loops this search finds are shortest-path loops already and are cut as found. Its
    cost grows with the number of vertices within ``max_length / 2`` of the basis, and the call
    that finds nothing, the last one of the documented loop, always runs it.

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

    Examples
    --------
    A thin torus has a short loop around its tube (length about 1.26) and a long one around its
    hole; cutting the short one leaves a sphere:

    ```python
    import ordito as od

    vertices, faces = od.creation.torus(1.0, 0.2, major_sections=24, minor_sections=8)
    print(od.measures.euler_characteristic(faces))  # 0
    new_vertices, new_faces, removed = od.repair.remove_tunnels(vertices, faces, max_length=2.0)
    print(removed, od.measures.euler_characteristic(new_faces))  # 1 2
    print(od.validation.is_watertight(new_vertices, new_faces))  # True
    _, _, removed = od.repair.remove_tunnels(vertices, faces, max_length=0.5)
    print(removed)  # 0
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    if max_length < 0.0:
        raise ValueError(f"max_length must be non-negative, got {max_length}")
    basis = _shortened_basis(vertices, faces, max_iter)
    if basis is None:
        return vertices, faces, 0
    selected = _disjoint_loops([loop for length, loop in basis.short if length <= max_length])
    if not selected:
        # The basis is the greedy one through its root, so a handle far from the root can appear
        # only inside a longer combination of classes. Before answering "nothing is short", search
        # every vertex of the basis for the shortest non-separating loop through it: every
        # non-separating loop crosses some basis loop, so a short one passes through one of them.
        selected = _disjoint_loops(_short_loops_through_basis(vertices, faces, basis, max_length))
    if not selected:
        return vertices, faces, 0

    # Disjoint is not independent: a family of disjoint non-trivial loops can still bound a piece
    # of the surface between them, and cutting along all of it splits that piece off instead of
    # dropping the genus by one per loop. The face labels of the cut mesh say which -- every
    # component after the first is one loop too many -- and they are read before cutting, so the
    # mesh is cut once, along the family that survives.
    labels, sides = _cut_face_labels(faces, selected, basis.twins)
    labels_np = labels.numpy()
    if np.unique(labels_np).size > 1:
        selected = _independent_loops(selected, labels_np, sides())
    cut_vertices, cut_faces = _cut_along_loops(vertices, faces, selected, basis.twins)
    cut_vertices = cast("wp.array[wp.vec3]", cut_vertices)
    # The cut writes every face index below its own vertex count, so the fill needs no reduction
    # to bound them.
    return (
        cut_vertices,
        od.holes.fill_min_weight(
            cut_vertices, cut_faces, metric=metric, n_vertices=cut_vertices.size
        ),
        len(selected),
    )


class _TunnelBasis(NamedTuple):
    """A shortened homology basis, with what the search through its vertices reuses."""

    short: list[tuple[float, np.ndarray]]
    """``(length, loop)`` pairs, shortest first; the sort is stable, so ties keep basis order."""
    loops: wp.array[wp.int32]
    """The generators before shortening, packed: simple cycles, as the cocycle needs them."""
    loop_offsets: wp.array[wp.int32]
    ring_halfedges: wp.array[wp.int32]
    ring_offsets: wp.array[wp.int32]
    twins: wp.array[wp.int32]


def _shortened_basis(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], max_iter: int
) -> _TunnelBasis | None:
    """
    Shorten and measure the homology basis, or ``None`` for a sphere.

    The basis stays packed from the generators through the shortening to the measuring, so no
    per-loop array is built and split again on the way; a basis has ``2 * genus`` loops of a
    handful of indices each, so everything after the measuring runs on the host, over one readback
    of them all. The one-rings the shortening walks are built here, once, so the search through
    the basis vertices reuses them.
    """
    loops, loop_offsets = od.homology.homology_generators_with_offsets(vertices, faces)
    if loop_offsets.size == 1:
        return None
    ring_halfedges, ring_offsets, is_boundary, twins = od.halfedge.vertex_one_rings(
        faces, n_vertices=vertices.size, return_twins=True
    )
    shortened, shortened_offsets, _sweeps = od.geodesic_walk.shorten_loop_with_offsets(
        vertices,
        faces,
        loops,
        loop_offsets,
        max_iter=max_iter,
        rings=(ring_halfedges, ring_offsets, is_boundary),
    )
    lengths, loops_np = _measure_loops(vertices, shortened, shortened_offsets)
    short = sorted(zip(lengths, loops_np, strict=True), key=lambda pair: pair[0])
    return _TunnelBasis(short, loops, loop_offsets, ring_halfedges, ring_offsets, twins)


def _short_loops_through_basis(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], basis: _TunnelBasis, max_length: float
) -> list[np.ndarray]:
    """
    Find the shortest non-separating loop through each basis vertex; keep those within the bound.

    Sorted shortest first, stably, each distinct cycle once. A non-separating loop crosses some
    loop of any homology basis -- the intersection form of a closed orientable surface is
    non-degenerate -- and two edge loops that cross share a vertex, so every non-separating loop
    passes through a vertex of the shortened basis; and Erickson and Whittlesey's argument
    (``kernels/repair.shortest_nontrivial_loop_per_root``) makes the shortest one through a root a
    shortest-path-tree loop of that root. So the shortest non-separating loop of the surface is
    among these whenever it is within ``max_length``.
    """
    device = faces.device
    n_vertices = vertices.size
    # Every basis loop's cocycle, on every halfedge: zero means "crosses an even number of times".
    cocycle = _launch.zeros(faces.size, dtype=wp.uint64, device=device)
    _launch.launch(
        kernel_repair.tunnel_loop_cocycle,
        dim=basis.loops.size,
        inputs=[
            faces,
            basis.ring_offsets,
            basis.ring_halfedges,
            basis.twins,
            basis.loops,
            basis.loop_offsets,
        ],
        outputs=[cocycle],
        device=device,
    )
    roots_np = np.unique(np.concatenate([loop for _, loop in basis.short])).astype(np.int32)
    # The kernel compares in float32 sums of its own; the slack only widens the candidates, and
    # the measured length below is the test that decides.
    bound = max_length * (1.0 + _TUNNEL_LENGTH_SLACK)
    found: list[np.ndarray] = []
    capacity = min(_TUNNEL_BALL_CAPACITY, n_vertices)
    while roots_np.size > 0:
        chunk = max(1, _TUNNEL_SCRATCH_ENTRIES // capacity)
        overflowed: list[np.ndarray] = []
        for start in range(0, roots_np.size, chunk):
            part = roots_np[start : start + chunk]
            loops_np, missed_np = _root_loop_search(
                vertices, faces, basis, cocycle, part, bound, capacity
            )
            found.extend(loops_np)
            overflowed.append(missed_np)
        roots_np = np.concatenate(overflowed)
        # A ball that outgrew its rows is searched again in wider ones; at ``n_vertices`` none can.
        capacity = min(capacity * _TUNNEL_BALL_GROWTH, n_vertices)
    if not found:
        return []

    # One cycle reached from several of its own vertices is one candidate.
    unique: dict[bytes, np.ndarray] = {}
    for loop in found:
        unique.setdefault(np.sort(loop).tobytes(), loop)
    candidates = list(unique.values())
    sizes = np.array([0, *(loop.size for loop in candidates)], dtype=np.int32)
    packed = _launch.array(np.concatenate(candidates), dtype=wp.int32, device=device)
    offsets = _launch.array(np.cumsum(sizes, dtype=np.int32), dtype=wp.int32, device=device)
    lengths, loops_np = _measure_loops(vertices, packed, offsets)
    return [
        loop
        for length, loop in sorted(zip(lengths, loops_np, strict=True), key=lambda pair: pair[0])
        if length <= max_length
    ]


def _root_loop_search(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    basis: _TunnelBasis,
    cocycle: wp.array[wp.uint64],
    roots_np: np.ndarray,
    bound: float,
    capacity: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Search ``roots_np`` in balls of up to ``capacity`` vertices: loops found, and roots missed.

    A missed root is one whose ball outgrew ``capacity``; a root with no loop within ``bound`` is
    neither.
    """
    device = faces.device
    n_roots = int(roots_np.size)
    roots = _launch.array(roots_np, dtype=wp.int32, device=device)
    # Open addressing over a power of two at least twice the rows, so the table stays half empty.
    table_size = 1 << (2 * capacity - 1).bit_length()
    table = odt.as_array2d(
        _launch.full((n_roots, table_size), -1, dtype=wp.int32, device=device), wp.int32
    )
    heap = odt.empty_2d((n_roots, capacity), wp.int32, device=device)
    counts = _launch.empty(n_roots, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.shortest_nontrivial_loop_per_root,
        dim=n_roots,
        inputs=[
            vertices,
            faces,
            basis.ring_offsets,
            basis.ring_halfedges,
            cocycle,
            roots,
            wp.float32(0.5 * bound),
            wp.float32(bound),
            table,
            odt.empty_2d((n_roots, capacity), wp.int32, device=device),
            odt.empty_2d((n_roots, capacity), wp.float32, device=device),
            odt.empty_2d((n_roots, capacity), wp.int32, device=device),
            odt.empty_2d((n_roots, capacity), wp.uint64, device=device),
            heap,
            odt.empty_2d((n_roots, capacity), wp.int32, device=device),
        ],
        outputs=[counts],
        device=device,
    )
    # Which roots found a loop, and how long it is, sizes the packed buffer; a few bytes a root.
    counts_np = counts.numpy()
    missed_np = roots_np[counts_np < 0]
    sizes_np = np.maximum(counts_np, 0)
    total = int(sizes_np.sum())
    if total == 0:
        return [], missed_np
    bounds_np = np.concatenate([[0], np.cumsum(sizes_np)]).astype(np.int32)
    packed = _launch.empty(total, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_repair.pack_root_loops,
        dim=n_roots,
        inputs=[heap, _launch.array(bounds_np, dtype=wp.int32, device=device)],
        outputs=[packed],
        device=device,
    )
    # The cycles go to the host to be told apart; a few dozen indices a root.
    packed_np = packed.numpy()
    return [
        packed_np[start:stop]
        for start, stop in zip(bounds_np[:-1].tolist(), bounds_np[1:].tolist(), strict=True)
        if stop > start
    ], missed_np


def _measure_loops(
    vertices: wp.array[wp.vec3], packed: wp.array[wp.int32], offsets: wp.array[wp.int32]
) -> tuple[list[float], list[np.ndarray]]:
    """
    Measure every loop's closed length, and read every loop's indices back, in one pass each.

    The lengths are ``polyline_length(closed=True)`` of each loop's gathered positions, bit for
    bit: ``kernels/polyline.packed_closed_loop_lengths`` folds each loop the way that function folds
    a polyline of one block. The sort that follows keeps
    equal lengths in basis order, and a regular fixture has many, so any other summation order
    would change which loops are cut. A loop longer than one block is measured on its own, its
    block sums committed in a fixed order.
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
    if not loops:
        return []
    # A vertex mask rather than a set of Python ints: the loops are NumPy arrays already.
    claimed = np.zeros(int(max(int(loop.max(initial=-1)) for loop in loops)) + 1, dtype=bool)
    kept: list[np.ndarray] = []
    for loop in loops:
        if claimed[loop].any():
            continue
        claimed[loop] = True
        kept.append(loop)
    return kept


def _cut_face_labels(
    faces: wp.array[wp.int32], loops: list[np.ndarray], twins: wp.array[wp.int32]
) -> tuple[wp.array[wp.int32], Callable[[], dict[tuple[int, int], tuple[int, int]]]]:
    """
    Label the faces as cutting along ``loops`` would leave them, without cutting.

    Cutting along an edge separates exactly the two faces across it and leaves every other face
    pair of an edge-manifold surface adjacent -- a pair across a non-loop edge at a loop vertex lies
    on one side of the loop there -- so the cut mesh's face adjacency is the input's with the loop
    edges' pairs removed. Severing those pairs in place of removing them keeps the face numbering,
    which the cut preserves too, so these labels are the cut mesh's own. The pairs are read off
    ``twins``, ``faces``' verified halfedge twins, so no adjacency table is sorted for them.

    Also returns a function reading back the severed pairs themselves, keyed by their ascending
    edge: the two faces across each loop edge, which telling a loop's two sides apart needs. It is
    a function because only a dependent family needs it, and it costs three readbacks.
    """
    device = faces.device
    # Each cycle's edges as ascending rows, which is what a cut keys on.
    cut_edges = np.concatenate(
        [np.sort(np.stack([loop, np.roll(loop, -1)], axis=1), axis=1) for loop in loops]
    ).astype(np.uint64)
    barrier_keys = _launch.array(
        np.sort(cut_edges[:, 0] + cut_edges[:, 1] * np.uint64(INDEX_RADIX_PAIR)),
        dtype=wp.uint64,
        device=device,
    )
    n_halfedges = twins.size
    severed = _launch.empty(n_halfedges, dtype=wp.int32, device=device)
    barrier = _launch.empty(n_halfedges, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_repair.sever_barrier_twins,
        dim=n_halfedges,
        inputs=[faces, twins, barrier_keys],
        outputs=[severed, barrier],
        device=device,
    )
    labels = od.adjacency.face_connected_component_labels(faces, mates=severed)

    def sides() -> dict[tuple[int, int], tuple[int, int]]:
        # A loop edge is one pair of a few hundred; the readbacks are of the mesh-sized buffers
        # the table is picked out of.
        rows = np.flatnonzero(barrier.numpy())
        faces_np = faces.numpy()
        twins_np = twins.numpy()
        tips = rows - rows % 3 + (rows + 1) % 3
        return {
            (int(min(a, b)), int(max(a, b))): (int(h // 3), int(t // 3))
            for a, b, h, t in zip(faces_np[rows], faces_np[tips], rows, twins_np[rows], strict=True)
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
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: list[np.ndarray],
    twins: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3] | wp.array[wp.vec3d], wp.array[wp.int32]]:
    """
    Cut the mesh along the edges of a family of closed vertex-index cycles.

    ``twins`` are ``faces``' verified halfedge twins, which the cut reads rather than pairing the
    halfedges and checking them again.
    """
    # Each cycle's edges as ascending rows, which is what a cut keys on.
    cut_edges = _launch.array(
        np.concatenate(
            [np.sort(np.stack([loop, np.roll(loop, -1)], axis=1), axis=1) for loop in loops]
        ),
        dtype=wp.int32,
        device=faces.device,
    )
    return od.seams.cut_along_edges(
        vertices, faces, odt.as_array2d(cut_edges, wp.int32), twins=twins
    )


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
        Longest edge over the height onto it above which a triangle counts as a T-vertex
        sliver; that longest edge is the one flipped. The reciprocal of the ``"area_max_side"``
        measure of [`face_quality`][ordito.triangles.face_quality] (about ``2 / sqrt(3)`` for an
        equilateral triangle, unbounded above). MeshLab's ``meshing_remove_t_vertices``
        threshold, whose default of ``40`` is this one. Must be positive.
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

    Examples
    --------
    Vertex 2 sits a hair above the edge ``(0, 1)`` of the triangle below it, making triangle
    ``(0, 1, 2)`` a sliver; one flip replaces the edge by ``(2, 3)``:

    ```python
    import warp as wp

    import ordito as od

    vertices = wp.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.01, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=wp.vec3,
    )
    faces = wp.array([0, 3, 1, 0, 1, 2, 0, 2, 4, 2, 1, 4], dtype=wp.int32)
    print(f"{od.triangles.face_quality(vertices, faces).numpy().max():.0f}")  # 5001
    flipped = od.repair.flip_t_vertices(vertices, faces)
    print(flipped.numpy().reshape(-1, 3).tolist())  # [[1, 2, 3], [0, 3, 2], [0, 2, 4], [2, 1, 4]]
    print(f"{od.triangles.face_quality(vertices, flipped).numpy().max():.2f}")  # 1.22
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    return od.remesh.flip_by_objective(
        vertices, faces, objective="t_vertex", aspect_threshold=threshold, max_iter=max_iter
    )
