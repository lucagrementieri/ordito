from typing import Any

import warp as wp

from ordito.kernels.array import (
    OverloadTable,
    atomic_min_packed_box,
    mark_at,
    scanned_count,
    trilinear_cell,
    trilinear_corner,
    trilinear_weight,
    unpack_edge_key,
)
from ordito.kernels.triangles import face_normals_and_area


@wp.func
def atomic_add_vec3(out_sum: wp.array2d[wp.Float], row: wp.int32, v: wp.vec3) -> None:
    # Component-wise atomic accumulation of a wp.vec3 into row ``row`` of a (n, 3) buffer.
    #
    # ``out_sum.dtype(...)`` is the conversion, and it is required rather than cosmetic: Warp does
    # **not** promote a float32 value into a float64 accumulator, and ``wp.atomic_add`` with the
    # two mismatched fails at kernel-parse time (probed on Warp 1.17). Reading the array's dtype
    # in kernel scope costs nothing -- unlike ``type(out_sum[row, 0])(...)``, the other spelling
    # that works, which loads the element it is about to update just to name its type.
    #
    # **Every caller today accumulates at float64 while its values are float32, and that is what
    # makes the sum reproducible.** A float atomic's summation order is whatever the scheduler
    # hands it, and float addition is not associative, so a float32 accumulator moves by about a
    # ULP between runs of the identical launch. That is harmless in itself and not harmless
    # downstream: ``curvature.principal_curvature`` fits an ill-conditioned quadric to these normals
    # and turns it into swings of most of the returned curvature at near-flat vertices. Widening the
    # accumulator does not fix the ordering -- nothing here can -- but it drops the disagreement
    # between two orderings far below what the float32 result can represent, so the narrowed answer
    # is reproducible in practice. It is not a *guarantee*: a sum landing within an eps of a float32
    # rounding boundary could still round both ways.
    #
    # Cost: the scatter kernel itself is within noise of the float32 one, and
    # ``vertices.vertex_normals`` end to end wins on a small mesh -- the fused narrow-and-normalize
    # tail removes a launch, since a float32 accumulator could reach ``wp.vec3`` through a zero-copy
    # ``array_cast`` and a float64 one cannot -- and loses slightly on a large one, where the wider
    # atomics start to show. The (n, 3) buffer doubles.
    wp.atomic_add(out_sum, row, 0, out_sum.dtype(v[0]))
    wp.atomic_add(out_sum, row, 1, out_sum.dtype(v[1]))
    wp.atomic_add(out_sum, row, 2, out_sum.dtype(v[2]))


@wp.kernel
def scatter_add(values: wp.array[Any], indices: wp.array[wp.int32], out_sum: wp.array[Any]) -> None:
    # 1D indexed accumulation: out_sum[indices[tid]] += values[tid]. ``Any`` rather than
    # ``wp.Scalar`` because the latter does not instantiate for vectors, and the vector heat
    # method seeds a ``wp.vec2d`` field through exactly this kernel. The widening gives up a
    # compile-time dtype guard: Warp still rejects a mismatched ``values`` / ``out_sum`` pair, but
    # the error arrives from codegen rather than from overload resolution and reads worse.
    tid = wp.int32(wp.tid())
    wp.atomic_add(out_sum, indices[tid], values[tid])


@wp.kernel
def count_occurrences(indices: wp.array[wp.int32], out_counts: wp.array[wp.int32]) -> None:
    # Histogram of ``indices``: one atomic increment per entry. Launch over ``indices.shape[0]``.
    # Sizing a CSR is what every caller wants it for -- incident faces per vertex (a flat face
    # buffer *is* the corner -> vertex map), face-corners per unique edge, outgoing halfedges per
    # vertex -- so the counts feed a scan and the caller allocates ``out_counts`` zeroed.
    tid = wp.int32(wp.tid())
    wp.atomic_add(out_counts, indices[tid], 1)


@wp.func
def add_edge_valence(
    sorted_keys: wp.array[wp.uint64], i: wp.int32, base: wp.uint64, out_valence: wp.array[wp.int32]
) -> None:
    # Vertex degree straight off a sorted ``pack_edge_key`` buffer: each run of equal keys is one
    # undirected edge, so the caller calls this at its first position -- and only its first -- and
    # it increments both endpoints.
    #
    # Anything holding materialized unique-edge rows has already run a grouping pass, and that pass
    # sorted these very keys -- so re-deriving the rows to reach a number the sorted buffer already
    # carries is the waste this avoids. The distinction is the *keys*, not the rows: a caller that
    # holds only the rows has nothing to run this on, and reaches ``count_occurrences`` over the
    # flattened pair buffer instead, which is what ``graph.edges_to_neighbor_lists`` does.
    # ``remesh``'s flip loop is the case that made it visible: its topology rebuild radix-sorts the
    # keys every pass, and recovering valence through ``edges.edges_unique`` grouped the identical
    # corner rows a *third* time. It runs inside that rebuild's ``remesh.mark_edge_pair_starts``.
    #
    # Run length is not tested by the caller here, unlike that kernel's pair flag, which wants the
    # manifold-interior edges alone: an edge is one edge whether one, two or five face corners
    # claim it, which is what makes this match ``edges_unique``'s row set exactly -- a pair-only
    # marker would silently drop every boundary edge.
    lo, hi = unpack_edge_key(sorted_keys[i], base)
    wp.atomic_add(out_valence, lo, 1)
    wp.atomic_add(out_valence, hi, 1)


@wp.kernel
def scatter_sum_scalar(
    values: wp.array2d[wp.Scalar], indices: wp.array2d[wp.int32], out_sum: wp.array[wp.Scalar]
) -> None:
    tid = wp.int32(wp.tid())
    index = indices[tid]
    for j in range(indices.shape[1]):
        wp.atomic_add(out_sum, index[j], values[tid, j])


@wp.kernel
def scatter_sum_vec(
    values: wp.array[wp.vec3], indices: wp.array2d[wp.int32], out_sum: wp.array2d[wp.Float]
) -> None:
    tid = wp.int32(wp.tid())
    index = indices[tid]
    value = values[tid]
    for j in range(indices.shape[1]):
        atomic_add_vec3(out_sum, index[j], value)


@wp.kernel
def scatter_weighted_sum_vec(
    values: wp.array[wp.vec3],
    indices: wp.array2d[wp.int32],
    weights: wp.array2d[wp.float32],
    out_sum: wp.array2d[wp.Float],
) -> None:
    tid = wp.int32(wp.tid())
    index = indices[tid]
    value = values[tid]
    for j in range(indices.shape[1]):
        atomic_add_vec3(out_sum, index[j], value * weights[tid, j])


@wp.func
def add_corner_triple(
    out_sum: wp.array[Any], indices: wp.array[wp.int32], row: wp.int32, a: Any, b: Any, c: Any
) -> None:
    wp.atomic_add(out_sum, indices[row * 3 + 0], a)
    wp.atomic_add(out_sum, indices[row * 3 + 1], b)
    wp.atomic_add(out_sum, indices[row * 3 + 2], c)


@wp.kernel
def scatter_face_thirds(
    faces: wp.array[wp.int32], areas: wp.array[Any], count: wp.Float, out_mass: wp.array[wp.Float]
) -> None:
    # Barycentric (lumped) mass: each face donates ``areas[f] / count`` to each incident vertex,
    # the area converted to the accumulator's dtype as it is read (exact for float32 into float64),
    # so a caller's areas of the other precision need no cast pass first.
    f = wp.int32(wp.tid())
    third = type(count)(areas[f]) / count
    add_corner_triple(out_mass, faces, f, third, third, third)


@wp.kernel
def scatter_face_area_thirds(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    count: wp.Float,
    out_mass: wp.array[wp.Float],
) -> None:
    # ``scatter_face_thirds`` with each face's area formed in the thread rather than read from a
    # table: ``face_normals_and_area``, the body of ``triangles.face_normals_and_areas``, so the
    # area is that table's entry bit for bit, and no area pass or buffer precedes the scatter.
    f = wp.int32(wp.tid())
    _normal, area = face_normals_and_area(vertices, faces, f)
    third = type(count)(area) / count
    add_corner_triple(out_mass, faces, f, third, third, third)


@wp.func
def add_value_and_count(
    vertex: wp.int32,
    value: wp.float32,
    out_sum: wp.array[wp.float32],
    out_valence: wp.array[wp.float32],
) -> None:
    # One contribution to a per-vertex mean: the value into the sum, one into the valence. Shared
    # by the face-valued and edge-valued averaging scatters below.
    wp.atomic_add(out_sum, vertex, value)
    wp.atomic_add(out_valence, vertex, wp.float32(1.0))


@wp.kernel
def scatter_face_values_sum_and_valence(
    faces: wp.array[wp.int32],
    face_values: wp.array[wp.float32],
    out_sum: wp.array[wp.float32],
    out_valence: wp.array[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    value = face_values[f]
    for j in range(3):
        add_value_and_count(faces[f * 3 + j], value, out_sum, out_valence)


@wp.func
def accumulate_endpoint_value(
    vi: wp.int32,
    vj: wp.int32,
    value: wp.float32,
    out_sum: wp.array[wp.float32],
    out_valence: wp.array[wp.float32],
) -> None:
    # Add one edge's value to both of its endpoints and count it at each -- the shared body of the
    # two "average an edge field onto the vertices" scatters below. They compute the same quantity
    # and differ only in where they find the endpoints and the value: one walks the *half*-edges of
    # each face (the ``igl::orient_halfedges`` convention, skipping the negatively oriented copy so
    # an interior edge is counted once), the other walks the unique-edge list directly.
    add_value_and_count(vi, value, out_sum, out_valence)
    add_value_and_count(vj, value, out_sum, out_valence)


@wp.kernel
def scatter_edges_sum_and_valence(
    faces: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    edges_orientation: wp.array2d[wp.int32],
    edge_values: wp.array[wp.float32],
    out_sum: wp.array[wp.float32],
    out_valence: wp.array[wp.float32],
) -> None:
    # Half-edge form: launch over faces, skip the negatively oriented copy of each interior edge.
    f = wp.int32(wp.tid())
    for j in range(3):
        if edges_orientation[f, j] < 0:
            continue
        accumulate_endpoint_value(
            faces[f * 3 + (j + 1) % 3],
            faces[f * 3 + (j + 2) % 3],
            edge_values[edges[f, j]],
            out_sum,
            out_valence,
        )


@wp.func
def mark_corners(
    out_marks: wp.array[Any], offset: wp.int32, a: wp.int32, b: wp.int32, c: wp.int32, value: Any
) -> None:
    # Store ``value`` at ``offset`` plus each of a face's three corner vertices: the "vertex named
    # by a kept face" marks every submesh compaction and mask dilation writes. Plain stores, so
    # only for a value every racing writer agrees on; ``offset`` places the vertex half of a joint
    # ``[faces | vertices]`` flag buffer.
    out_marks[offset + a] = value
    out_marks[offset + b] = value
    out_marks[offset + c] = value


@wp.func
def lock_two_rings(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    s: wp.int32,
    r: wp.int32,
    key: wp.int64,
    out_claim: wp.array[wp.int64],
) -> None:
    # Atomic-min ``key`` into every vertex of the two closed 1-rings of ``s`` and ``r``: the lock
    # half of a parallel independent set over edge candidates, so that whichever candidate wins
    # everywhere it touched has a neighbourhood disjoint from every other winner's.
    #
    # Both of ``kernels/remesh.py``'s collapse paths run this once per pass, with the key
    # ``remesh.scramble_index`` builds. Two properties of that key are load-bearing and both are
    # recorded there: it must not be spatially monotone (a monotone key commits one collapse per
    # pass), and it must be *injective*, or two candidates can tie and both believe they won --
    # which is why it is 64 bits wide here rather than the natural int32 of a vertex index.
    wp.atomic_min(out_claim, s, key)
    wp.atomic_min(out_claim, r, key)
    for i in range(offsets[s], offsets[s + 1]):
        wp.atomic_min(out_claim, columns[i], key)
    for i in range(offsets[r], offsets[r + 1]):
        wp.atomic_min(out_claim, columns[i], key)


@wp.func
def two_rings_hold(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    table: wp.array[Any],
    s: wp.int32,
    r: wp.int32,
    value: Any,
) -> wp.bool:
    # Does every vertex of the two closed 1-rings of ``s`` and ``r`` hold ``value`` in ``table``?
    # The read half of ``lock_two_rings`` (a candidate won its claim where the minimum is its own
    # key) and of ``stamp_two_rings`` (a neighbourhood is untouched where no mark is set).
    if table[s] != value or table[r] != value:
        return False
    for i in range(offsets[s], offsets[s + 1]):
        if table[columns[i]] != value:
            return False
    for i in range(offsets[r], offsets[r + 1]):
        if table[columns[i]] != value:
            return False
    return True


@wp.func
def stamp_two_rings(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    s: wp.int32,
    r: wp.int32,
    value: wp.int32,
    out_marks: wp.array[wp.int32],
) -> None:
    # Store ``value`` into every vertex of the two closed 1-rings of ``s`` and ``r``: plain stores
    # rather than ``lock_two_rings``' atomic minimum, so it is only for a value every racing writer
    # agrees on.
    out_marks[s] = value
    out_marks[r] = value
    for i in range(offsets[s], offsets[s + 1]):
        out_marks[columns[i]] = value
    for i in range(offsets[r], offsets[r + 1]):
        out_marks[columns[i]] = value


@wp.kernel
def scatter_index(index: wp.array[wp.int32], out_scattered: wp.array[wp.int32]) -> None:
    tid = wp.int32(wp.tid())
    out_scattered[index[tid]] = tid


@wp.kernel
def scatter_index_where_scanned(
    inclusive: wp.array[wp.int32], out_scattered: wp.array[wp.int32]
) -> None:
    # Scatter the index of every selected position to its compact slot, for a caller that scanned
    # its 0/1 flags *in place*, so only the inclusive scan survives: position ``i`` was selected
    # exactly when the scan steps there (``array.scanned_count``), and it lands at the exclusive
    # value. One flag buffer fewer than keeping the flags beside their scan -- ``array.flatnonzero``
    # allocates one ``(n,)`` ``int32`` instead of two -- for one neighbouring, coalesced read per
    # thread.
    i = wp.int32(wp.tid())
    start, count = scanned_count(inclusive, i)
    if count != 0:
        out_scattered[start] = i


@wp.func
def record_edge_incidence(
    e: wp.int32,
    corner: wp.int32,
    out_edge_face_count: wp.array[wp.int32],
    out_edge_faces: wp.array2d[wp.int32],
) -> None:
    # Count corner ``corner``'s face onto unique edge ``e`` and record it in the edge's first two
    # face slots; the count doubles as the write cursor (``scatter_edge_incidence`` below).
    slot = wp.atomic_add(out_edge_face_count, e, 1)
    if slot < 2:
        out_edge_faces[e, slot] = corner // 3


@wp.kernel
def scatter_edge_incidence(
    inverse: wp.array[wp.int32],
    out_edge_face_count: wp.array[wp.int32],
    out_edge_faces: wp.array2d[wp.int32],
) -> None:
    # Face-corners per unique edge *and* the faces themselves, in one pass over the corner ->
    # unique-edge map ``inverse``: corner ``c`` belongs to face ``c // 3``, so no face table is
    # needed. The count is 1 on a boundary edge and 2 on an interior one; a non-manifold edge
    # counts higher and its faces past the second are dropped, which is what the exactly-2 row
    # grouping behind ``adjacency.face_adjacency`` does with them too.
    #
    # Launch over ``inverse.shape[0]`` with ``out_edge_face_count`` zeroed: the count doubles as the
    # write cursor, which is why this replaces ``count_occurrences`` rather than following it.
    # ``out_edge_faces`` needs no zeroing and none of the three callers zeroes it -- but that means
    # a row whose count came back below 2 has an **unwritten** second column (a boundary edge) or
    # both columns unwritten (an edge no corner named), so read the count first. It is not a zero.
    #
    # It lives here rather than in ``kernels/remesh.py``, where the decimation passes first needed
    # it, because ``homology.homology_generators`` groups the same rows for the same reason -- one
    # grouping answering "how many faces meet along this edge, and which" for the whole package.
    c = wp.int32(wp.tid())
    record_edge_incidence(inverse[c], c, out_edge_face_count, out_edge_faces)


@wp.kernel
def scatter_group_bounds(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    groups: wp.array[wp.int32],
    out_corners: wp.array[wp.float32],
) -> None:
    # Axis-aligned bounds of every face group, six ``float32`` slots per group packed
    # ``[min_x, min_y, min_z, -max_x, -max_y, -max_z]`` -- the ``kernels/bounds.py`` convention, so
    # one ``wp.full(inf)`` seeds both ends and every update is a single ``wp.atomic_min``. Launch
    # over ``n_faces`` with ``out_corners`` sized ``6 * n_groups``; ``groups[f]`` is the group index
    # of face ``f``, which for a connected-component label is a representative face index and so
    # needs ``n_groups == n_faces``. A group no face names keeps its ``inf`` seed, which
    # [`packed_box_diagonal`][ordito.kernels.bounds.packed_box_diagonal] reads as empty.
    f = wp.int32(wp.tid())
    group = groups[f]
    for c in range(3):
        # One point is its own degenerate box, so it reduces into both halves of the packing.
        position = vertices[faces[f * 3 + c]]
        atomic_min_packed_box(out_corners, group, position, position)


@wp.kernel
def scatter_face_labels_to_vertices(
    faces: wp.array[wp.int32], labels: wp.array[wp.int32], out_vertex_label: wp.array[wp.int32]
) -> None:
    # Carry a per-*face* label onto the vertices its faces reference. Launch over ``3 * n_faces``
    # corners, with ``out_vertex_label`` seeded below every real label (``-1``); an unreferenced
    # vertex keeps the seed.
    #
    # ``wp.atomic_max`` rather than a plain store, so that a vertex shared by two differently
    # labelled faces -- a bowtie between two face-connected components -- takes the **larger** label
    # deterministically instead of whichever thread landed last.
    c = wp.int32(wp.tid())
    wp.atomic_max(out_vertex_label, faces[c], labels[c // 3])


@wp.kernel
def mark_membership_mask(indices: wp.array[wp.int32], out_mask: wp.array[wp.bool]) -> None:
    # Mark ``out_mask[indices[tid]] = True``, skipping out-of-range indices (``array.mark_at``).
    mark_at(out_mask, indices[wp.int32(wp.tid())])


@wp.kernel(enable_backward=False)
def splat_grid_trilinear(
    points: wp.array[wp.vec3],
    values: wp.array[Any],
    lower: wp.vec3,
    inverse_spacing: wp.vec3,
    out_field: wp.array3d[Any],
    out_density: wp.array3d[wp.float32],
) -> None:
    # Accumulate one value into the eight lattice corners around its position, weighted
    # trilinearly, and the same eight weights into a density lattice.
    #
    # The density is not an optional extra: the eight weights sum to 1 per point, so ``density``
    # holds the number of points each corner "saw" and dividing the field by it is what turns an
    # accumulation into an average. The wrapper does that division, because the floor it needs
    # (a corner no point reached) is a policy rather than arithmetic.
    #
    # ``kernels/interpolation.sample_grid_trilinear`` is the transpose of this kernel, and
    # ``kernels/reconstruction.splat_normals`` is the same stencil specialized to a flat
    # ``res**3`` float32 buffer with a confidence weight; all three share ``trilinear_cell`` and
    # ``trilinear_weight``.
    s = wp.int32(wp.tid())
    shape = wp.vec3i(out_field.shape[0], out_field.shape[1], out_field.shape[2])
    base, next_corner, fractions = trilinear_cell(
        wp.cw_mul(points[s] - lower, inverse_spacing), shape
    )
    value = values[s]
    for offset_x in range(2):
        for offset_y in range(2):
            for offset_z in range(2):
                weight = trilinear_weight(fractions, offset_x, offset_y, offset_z)
                corner = trilinear_corner(base, next_corner, wp.vec3i(offset_x, offset_y, offset_z))
                wp.atomic_add(out_field, corner[0], corner[1], corner[2], weight * value)
                wp.atomic_add(out_density, corner[0], corner[1], corner[2], weight)


@wp.kernel(enable_backward=False)
def divide_by_density(
    density: wp.array3d[wp.float32], min_weight: wp.float32, out_field: wp.array3d[Any]
) -> None:
    # Turn ``splat_grid_trilinear``'s accumulation into a weighted mean, in place.
    #
    # The denominator is ``max(density, min_weight)``, not a branch on it: that is what keeps a
    # lattice larger than its point cloud finite instead of full of amplified noise, and it is the
    # convention the reference uses (``volume_densities.clamp(min_weight)``). A corner nothing
    # reached still comes out zero either way, since its numerator is zero too -- the two spellings
    # differ only for a corner whose density is *between* zero and the floor, where the clamp
    # scales the answer down smoothly rather than discarding it. An in-place output, so the
    # argument keeps the ``out_`` prefix and the CLAUDE.md section 2.1 allowlist carries it.
    i, j, k = wp.tid()
    out_field[i, j, k] = out_field[i, j, k] / wp.max(density[i, j, k], min_weight)


# Concrete overloads, registered at import -- rationale in ``ordito/kernels/reduce.py``, rule in
# CLAUDE.md section 2.5. This module is reached from 11 wrappers at scattered moments, so it forked
# nearly twice per overload before registration.
#
# Each set is the dtypes its call sites actually build, not a menu: ``scatter_add`` accumulates a
# ``wp.float32`` per-component volume in ``repair`` and a ``wp.vec2d`` tangent field in
# ``heat.vector``, and the two mass/curvature scatters follow their wrapper's precision keyword.
_VALUE_DTYPES = (wp.float32, wp.float64)
# ``scatter_add`` gets its own, narrower set: it has exactly three call sites -- ``repair.py``'s two
# per-component ``wp.float32`` volume/area accumulators and ``heat.py``'s ``wp.vec2d`` tangent seed
# -- so the ``wp.float64`` and ``wp.vec3`` rows registered beside them were compile time paid on
# every rebuild for an overload nothing can reach.
_SCATTER_ADD_DTYPES = (wp.float32, wp.vec2d)
# The two vector scatters get a set of **one**, for the same reason: their only caller is
# ``vertices._accumulate_and_normalize``, which accumulates at ``wp.float64`` so that the summation
# order the atomics pick cannot reach the answer (see ``atomic_add_vec3``). The kernels stay generic
# over the accumulator anyway, so a float32 caller is a row here rather than a second kernel -- but
# registering that row today would be an overload nothing launches, paid for on every rebuild.
_VECTOR_ACCUMULATOR_DTYPES = (wp.float64,)


# ``splat_grid_trilinear``'s dtype set is the pair its wrapper
# [`voxels.splat_onto_grid`][ordito.voxels.splat_onto_grid] documents and its transpose
# ``kernels/interpolation.sample_grid_trilinear`` registers (the public reader of that kernel is
# [`voxels.sample_grid_trilinear`][ordito.voxels.sample_grid_trilinear], not an ``interpolation``
# one): ``wp.float32`` for a scalar field and ``wp.vec3`` for a vector one. Not ``wp.float64`` --
# the weights and the density lattice are float32, so a float64 field would carry a float32
# accuracy floor and the wider dtype would be a promise the kernel cannot keep.
_GRID_DTYPES = (wp.float32, wp.vec3)


# The concrete handles keyed by the caller's value dtype -- see
# [`OverloadTable`][ordito.kernels.array.OverloadTable]. This module's kernels are one launch of a
# short wrapper each, which is where a generic launch's host-side resolution is worth the most.
DIVIDE_BY_DENSITY: OverloadTable
SPLAT_GRID_TRILINEAR: OverloadTable
SCATTER_ADD: OverloadTable
SCATTER_FACE_THIRDS: OverloadTable
SCATTER_FACE_AREA_THIRDS: OverloadTable
SCATTER_SUM_SCALAR: OverloadTable
SCATTER_SUM_VEC: OverloadTable
SCATTER_WEIGHTED_SUM_VEC: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global DIVIDE_BY_DENSITY, SPLAT_GRID_TRILINEAR, SCATTER_ADD
    global SCATTER_FACE_AREA_THIRDS, SCATTER_FACE_THIRDS, SCATTER_SUM_SCALAR
    global SCATTER_SUM_VEC, SCATTER_WEIGHTED_SUM_VEC
    DIVIDE_BY_DENSITY = OverloadTable(
        divide_by_density,
        {d: [wp.array3d[wp.float32], wp.float32, wp.array3d[d]] for d in _GRID_DTYPES},
    )
    SPLAT_GRID_TRILINEAR = OverloadTable(
        splat_grid_trilinear,
        {
            d: [
                wp.array[wp.vec3],
                wp.array[d],
                wp.vec3,
                wp.vec3,
                wp.array3d[d],
                wp.array3d[wp.float32],
            ]
            for d in _GRID_DTYPES
        },
    )
    SCATTER_ADD = OverloadTable(
        scatter_add,
        {d: [wp.array[d], wp.array[wp.int32], wp.array[d]] for d in _SCATTER_ADD_DTYPES},
    )
    # Keyed ``(areas dtype, mass dtype)``, the cross product: a caller's areas may be either.
    SCATTER_FACE_THIRDS = OverloadTable(
        scatter_face_thirds,
        {
            (a, d): [wp.array[wp.int32], wp.array[a], d, wp.array[d]]
            for a in _VALUE_DTYPES
            for d in _VALUE_DTYPES
        },
    )
    SCATTER_FACE_AREA_THIRDS = OverloadTable(
        scatter_face_area_thirds,
        {d: [wp.array[wp.vec3], wp.array[wp.int32], d, wp.array[d]] for d in _VALUE_DTYPES},
    )
    SCATTER_SUM_SCALAR = OverloadTable(
        scatter_sum_scalar,
        {d: [wp.array2d[d], wp.array2d[wp.int32], wp.array[d]] for d in _VALUE_DTYPES},
    )
    SCATTER_SUM_VEC = OverloadTable(
        scatter_sum_vec,
        {
            d: [wp.array[wp.vec3], wp.array2d[wp.int32], wp.array2d[d]]
            for d in _VECTOR_ACCUMULATOR_DTYPES
        },
    )
    SCATTER_WEIGHTED_SUM_VEC = OverloadTable(
        scatter_weighted_sum_vec,
        {
            d: [wp.array[wp.vec3], wp.array2d[wp.int32], wp.array2d[wp.float32], wp.array2d[d]]
            for d in _VECTOR_ACCUMULATOR_DTYPES
        },
    )


_register_overloads()
