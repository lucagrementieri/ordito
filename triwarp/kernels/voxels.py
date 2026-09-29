"""
Kernels for ``triwarp.voxels``: cell indexing, tri-box voxelization, morphology, dense conversion.

Every kernel here that reads a grid takes the volume's ``uint64`` id and probes it with
``wp.volume_lookup_index``, which is ``O(1)`` and returns the voxel's linear index (``-1`` when the
cell is empty). That index is the row index of ``Volume.get_voxels()``, so a per-voxel payload is a
plain ``wp.array`` and no side table is ever built.

The half-voxel convention is the wrapper's: a volume whose translation is ``origin + 0.5 * s`` has
NanoVDB voxel ``i`` covering world ``[origin + i * s, origin + (i + 1) * s)``, so
``floor(world_to_index(p) + 0.5)`` is the cell containing ``p``.

**Two halves of this module convert between cells and world positions differently, and the split is
deliberate.** Every kernel *downstream* of a built grid -- ``point_cell``,
``cell_center_positions``, ``corner_positions`` -- goes through ``wp.volume_world_to_index`` /
``wp.volume_index_to_world``, so the convention above is read off the object that defines it
instead of re-implemented. The *voxelization* kernels -- ``voxel_cell``, ``voxel_cell_indices``,
``triangle_voxel_window`` -- run before any volume exists: they produce the cells
``Volume.allocate_by_voxels`` then builds a grid from, so there is no volume id to pass and the
``(origin, voxel_size)`` scalar form is required rather than preferred. Measured perf-neutral either
way, so this is a single-source-of-truth split and not a speed one.
"""

import warp as wp

from triwarp.constants import INT32_MAX_CONSTANT
from triwarp.kernels.algorithms.connected_components import ecl_hook_edge, find_representative
from triwarp.kernels.array import (
    binary_search_index,
    binary_search_index_left,
    lattice_position,
    ravel_index,
)
from triwarp.kernels.grouping import HASH_MULT_U64, hash_slot, next_slot
from triwarp.kernels.predicates import triangle_aabb, triangle_aabb_overlap
from triwarp.kernels.triangles import face_vertices, row_triple, write_row_triple

# ---------------------------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------------------------


@wp.func
def voxel_cell(position: wp.vec3, origin: wp.vec3, inverse_size: wp.float32) -> wp.vec3i:
    # Integer voxel a position falls in, for the grid anchored at ``origin`` with cell width
    # ``1 / inverse_size``. ``wp.floor`` rather than a cast, so negative coordinates round the same
    # way positive ones do (a C-style truncation would fold the two cells either side of the origin
    # into one).
    local = (position - origin) * inverse_size
    return wp.vec3i(
        wp.int32(wp.floor(local[0])), wp.int32(wp.floor(local[1])), wp.int32(wp.floor(local[2]))
    )


@wp.func
def voxel_cell_center(cell: wp.vec3i, origin: wp.vec3, voxel_size: wp.float32) -> wp.vec3:
    # World position of a cell's centre: the inverse of ``voxel_cell`` above, up to the half-voxel
    # that names the centre rather than the lower corner.
    #
    # The hand-rolled form rather than ``wp.volume_index_to_world`` (which
    # ``cell_center_positions`` below uses, and which the module docstring names as the convention)
    # because every caller here is *pre-volume*: it holds an origin and a cell width and no
    # ``wp.Volume`` handle to ask. The two agree to 3.58e-07 over 3 929 voxels -- float32 rounding,
    # not a convention difference -- and the measurement is recorded on ``cell_center_positions``.
    return wp.vec3(
        origin[0] + (wp.float32(cell[0]) + 0.5) * voxel_size,
        origin[1] + (wp.float32(cell[1]) + 0.5) * voxel_size,
        origin[2] + (wp.float32(cell[2]) + 0.5) * voxel_size,
    )


@wp.func
def squared_distance_to_own_cell_center(
    position: wp.vec3, origin: wp.vec3, voxel_size: wp.float32
) -> wp.float32:
    # How far a point sits from the centre of the voxel it falls in, squared. The quantity a
    # "closest to the cell centre" cluster representative is chosen by.
    #
    # Named because that choice is a *two-pass* argmin -- one kernel reduces the winning distance
    # per cluster and a second re-tests it to break the tie by lowest index -- and the two passes
    # agree only while both compute this expression identically. Two copies of it is a silent
    # correctness hazard rather than a duplication nit: a change to one that rounds differently
    # leaves clusters with no representative at all.
    cell = voxel_cell(position, origin, 1.0 / voxel_size)
    return wp.length_sq(position - voxel_cell_center(cell, origin, voxel_size))


@wp.kernel
def voxel_cell_indices(
    points: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.float32,
    out_cells: wp.array2d[wp.int32],
) -> None:
    v = wp.int32(wp.tid())
    cell = voxel_cell(points[v], origin, inverse_size)
    write_row_triple(out_cells, v, cell[0], cell[1], cell[2])


@wp.func
def triangle_voxel_window_from_vertices(
    v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, origin: wp.vec3, inverse_size: wp.float32
) -> tuple[wp.vec3i, wp.vec3i]:
    # Inclusive lower/upper cell of the exact AABB window of a triangle already in hand. Shared by
    # the count and test passes below, which MUST enumerate the same window: the second writes into
    # the slots the first reserved, so a window that disagreed by one cell would write out of range.
    #
    # Open3D walks ``round((max - min) / vs) + 2`` cells, a strict superset of this one; a cell
    # outside the triangle's own AABB cannot overlap the triangle, so the *accepted* sets are
    # identical and only the number of rejected candidates differs.
    lower, upper = triangle_aabb(v0, v1, v2)
    return voxel_cell(lower, origin, inverse_size), voxel_cell(upper, origin, inverse_size)


@wp.func
def triangle_voxel_window(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_index: wp.int32,
    origin: wp.vec3,
    inverse_size: wp.float32,
) -> tuple[wp.vec3i, wp.vec3i]:
    # Gathers the face's own vertices and defers to the vertex-taking form above. Kept separate from
    # it (rather than folded into one signature) because ``test_triangle_candidates`` below needs
    # the gathered v0/v1/v2 for its own overlap test right after computing the window, and calling
    # through this wrapper would gather them a second time to get at them.
    v0, v1, v2 = face_vertices(vertices, faces, face_index)
    return triangle_voxel_window_from_vertices(v0, v1, v2, origin, inverse_size)


@wp.kernel
def count_triangle_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    origin: wp.vec3,
    inverse_size: wp.float32,
    out_counts: wp.array[wp.int32],
    out_counts_f32: wp.array[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    lo, hi = triangle_voxel_window(vertices, faces, f, origin, inverse_size)
    # int64 so a wildly under-sized voxel does not wrap the product into a plausible small count --
    # each axis span widens to int64 *before* the subtraction, not after, so the subtraction itself
    # cannot already overflow int32 for a triangle whose window is that wide (the convention
    # ``kernels/graph.py``'s packed labels key uses: widen the operands, not their difference).
    span_x = wp.int64(hi[0]) - wp.int64(lo[0]) + wp.int64(1)
    span_y = wp.int64(hi[1]) - wp.int64(lo[1]) + wp.int64(1)
    span_z = wp.int64(hi[2]) - wp.int64(lo[2]) + wp.int64(1)
    span = span_x * span_y * span_z
    out_counts[f] = wp.int32(wp.min(span, wp.int64(INT32_MAX_CONSTANT)))
    out_counts_f32[f] = wp.float32(span)


@wp.kernel
def test_triangle_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    origin: wp.vec3,
    voxel_size: wp.float32,
    inverse_size: wp.float32,
    out_cells: wp.array2d[wp.int32],
    out_mask: wp.array[wp.int32],
) -> None:
    # One thread per (triangle, candidate cell) pair: per-triangle window sizes span orders of
    # magnitude on any real mesh, so a thread-per-triangle launch is load-imbalanced by that same
    # factor. ``binary_search_index`` recovers the owning triangle from the flat work-item id
    # (``wp.lower_bound`` clamps to ``n - 1`` and would misattribute the last window).
    #
    # The per-item decode below (``span_y``/``span_z``/``plane``/``i``/``j``/``k``) stays int32,
    # unlike ``count_triangle_candidates``'s widened axis spans above -- a single axis wide enough
    # to overflow int32 on its own would still misdecode here, but reaching that needs
    # ``max_candidates`` raised past its ``2**31`` default (already impractical: the candidate
    # buffers alone would be tens of GB at that width). Left as a documented residual rather than
    # widening this hot loop's arithmetic to int64 for a practically unreachable input; revisit only
    # if a caller legitimately needs a wider ``max_candidates``.
    item = wp.int32(wp.tid())
    f = binary_search_index(offsets, item) - 1
    # Gather the face's own vertices once and feed them to both the window (for this item's cell)
    # and the overlap test below, rather than calling ``triangle_voxel_window`` (which would gather
    # the same three rows again internally) -- this is the dominant work unit in the kernel, one
    # gather per (triangle, candidate-cell) work item rather than two.
    v0, v1, v2 = face_vertices(vertices, faces, f)
    lo, hi = triangle_voxel_window_from_vertices(v0, v1, v2, origin, inverse_size)
    span_y = hi[1] - lo[1] + 1
    span_z = hi[2] - lo[2] + 1

    local = item - offsets[f]
    plane = span_y * span_z
    i = local // plane
    rest = local % plane
    j = rest // span_z
    k = rest % span_z
    cell = wp.vec3i(lo[0] + i, lo[1] + j, lo[2] + k)

    half = wp.vec3(0.5 * voxel_size, 0.5 * voxel_size, 0.5 * voxel_size)
    center = voxel_cell_center(cell, origin, voxel_size)
    write_row_triple(out_cells, item, cell[0], cell[1], cell[2])
    out_mask[item] = wp.where(triangle_aabb_overlap(center, half, v0, v1, v2), 1, 0)


# ---------------------------------------------------------------------------------------------
# Grid <-> world
# ---------------------------------------------------------------------------------------------


@wp.kernel
def cell_center_positions(
    volume: wp.uint64, voxels: wp.array2d[wp.int32], out_centers: wp.array[wp.vec3]
) -> None:
    # NanoVDB centres voxel ``i`` *on* index-space coordinate ``i``, so the integer cell coordinate
    # maps straight to the cell centre and the volume's own transform supplies the half-voxel shift
    # the module docstring describes. It agrees with the hand-rolled
    # ``origin + (cell + 0.5) * voxel_size`` to float32 rounding, i.e. no convention disagreement.
    v = wp.int32(wp.tid())
    out_centers[v] = wp.volume_index_to_world(
        volume,
        wp.vec3(wp.float32(voxels[v, 0]), wp.float32(voxels[v, 1]), wp.float32(voxels[v, 2])),
    )


@wp.func
def cell_slot(volume: wp.uint64, cells: wp.array2d[wp.int32], v: wp.int32) -> wp.int32:
    # The grid and ``get_voxels()`` share one numbering, so a cell's slot is also its payload row;
    # ``-1`` means the cell is not in the grid. Shared by the slot kernel and the occupancy one
    # below, which differ only in whether the caller wants the row or just its existence.
    return wp.volume_lookup_index(volume, cells[v, 0], cells[v, 1], cells[v, 2])


@wp.kernel
def lookup_cell_slots(
    volume: wp.uint64, cells: wp.array2d[wp.int32], out_slots: wp.array[wp.int32]
) -> None:
    v = wp.int32(wp.tid())
    out_slots[v] = cell_slot(volume, cells, v)


@wp.kernel
def cell_occupancy(
    volume: wp.uint64, cells: wp.array2d[wp.int32], present: wp.bool, out_mask: wp.array[wp.bool]
) -> None:
    # The probe and the comparison in one pass: the slot is a register here, where a separate
    # lookup kernel would write every one of them to global memory for a second launch to read
    # back and test. ``present`` is warp-uniform and selects membership or its complement, which
    # is what lets ``intersection`` and ``difference`` share this kernel instead of the second
    # paying a third launch to invert the first's answer.
    #
    # Measured against the lookup-kernel-plus-map form it replaced, output byte-identical: 2.0x on
    # ``occupancy_at_cells``, 2.1x on ``occupancy_at_points`` and 1.25x on ``difference``, which
    # carried the extra inversion launch.
    v = wp.int32(wp.tid())
    out_mask[v] = (cell_slot(volume, cells, v) >= 0) == present


@wp.kernel
def cell_occupancy_flags(
    volume: wp.uint64, cells: wp.array2d[wp.int32], present: wp.bool, out_flags: wp.array[wp.int32]
) -> None:
    # ``cell_occupancy``'s test written as the ``int32`` 0/1 flags ``Volume.allocate_by_voxels``
    # takes as its ``point_mask``, which is the whole difference between the two: a set operation
    # that rebuilds a grid from the kept cells hands the flags to the builder with the unfiltered
    # cell array, so no compaction pass, readback or gathered copy stands between them.
    v = wp.int32(wp.tid())
    out_flags[v] = wp.where((cell_slot(volume, cells, v) >= 0) == present, 1, 0)


@wp.func
def index_space_cell(uvw: wp.vec3) -> wp.vec3i:
    # NanoVDB centres voxel ``i`` on index-space coordinate ``i``, so the cell containing index
    # position ``uvw`` is ``floor(uvw + 0.5)`` -- not ``round``, which sends ``-0.5`` to ``-1``
    # instead of ``0``. Shared by ``point_cell`` and ``table_point_slot``.
    return wp.vec3i(
        wp.int32(wp.floor(uvw[0] + 0.5)),
        wp.int32(wp.floor(uvw[1] + 0.5)),
        wp.int32(wp.floor(uvw[2] + 0.5)),
    )


@wp.func
def map_world_to_index(
    position: wp.vec3, translation: wp.vec3, inverse_size: wp.float32, zero: wp.float32
) -> wp.vec3:
    # ``wp.volume_world_to_index`` for an isotropic grid, bit for bit, without the volume:
    # PNanoVDB's ``map_apply_inverse`` is ``(p - vecf) . invmatf`` row by row, and for a diagonal
    # map each row is one product plus two exact zeros. ``zero`` is a *runtime* 0 so the compiler
    # keeps that shape -- a literal would let ``product + 0.5`` in ``index_space_cell`` contract to
    # an FMA, which rounds once where the volume's path rounds twice. ``translation`` is the float32
    # of the volume's, ``inverse_size`` the float32 of ``1 / float64(float32(voxel_size))``.
    s = position - translation
    return wp.vec3(
        s[0] * inverse_size + s[1] * zero + s[2] * zero,
        s[0] * zero + s[1] * inverse_size + s[2] * zero,
        s[0] * zero + s[1] * zero + s[2] * inverse_size,
    )


@wp.func
def point_cell(volume: wp.uint64, position: wp.vec3) -> wp.vec3i:
    # The cell containing a world position, read through the volume's own transform.
    return index_space_cell(wp.volume_world_to_index(volume, position))


@wp.func
def point_slot(volume: wp.uint64, position: wp.vec3) -> wp.int32:
    # The point twin of ``cell_slot``, and shared for the same reason: a query's voxel row, or
    # ``-1`` outside the grid.
    cell = point_cell(volume, position)
    return wp.volume_lookup_index(volume, cell[0], cell[1], cell[2])


@wp.kernel
def point_occupancy(
    volume: wp.uint64, points: wp.array[wp.vec3], present: wp.bool, out_mask: wp.array[wp.bool]
) -> None:
    # The point form of ``cell_occupancy``; see it for why the probe and the test share a kernel.
    p = wp.int32(wp.tid())
    out_mask[p] = (point_slot(volume, points[p]) >= 0) == present


@wp.kernel
def pack_cell_keys(
    cells: wp.array2d[wp.int32],
    lower: wp.array[wp.int32],
    upper: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
) -> None:
    # Column 0 is the least significant digit, matching ``kernels/grouping.pack_indices``: sorting
    # these keys reproduces ``grouping.unique_rows``'s row order exactly. The shift is
    # ``min(lower[c], 0)`` per axis -- order-preserving because it is a per-axis constant, and only
    # non-zero where a column actually goes negative, so a non-negative cell set keeps exactly the
    # keys ``grouping.hash_indices_rows`` would produce.
    #
    # ``lower`` / ``upper`` are the two three-element buffers ``reduce.minmax(cells, axis=0)``
    # returns, read here rather than passed in as ``wp.vec3i`` / ``wp.uint64`` scalars: every lane
    # wants the same six values, so they are broadcast loads out of L2, and taking them by value
    # would instead cost the wrapper two host readbacks. Byte-identical values, and worth about a
    # quarter of ``voxels.cells`` at every size -- the call is host-bound throughout, so removing
    # host work is the whole win.
    v = wp.int32(wp.tid())
    lo = wp.int32(0)
    hi = upper[0]
    for c in range(3):
        lo = wp.min(lo, lower[c])
        hi = wp.max(hi, upper[c])
    radix = wp.uint64(hi - lo + 1)
    key = wp.uint64(0)
    power = wp.uint64(1)
    for c in range(3):
        key = key + wp.uint64(wp.uint32(cells[v, c] - wp.min(lower[c], 0))) * power
        power = power * radix
    out_keys[v] = key


@wp.kernel
def lattice_points(lower: wp.vec3, step: wp.vec3, out_points: wp.array3d[wp.vec3]) -> None:
    # A dense node lattice, rank-2 destination. The position is ``array.lattice_position``, shared
    # with ``reconstruction.lattice_points``, which writes the same quantity into a flat row-major
    # buffer instead. This is one of the two sites this module's docstring names as running before
    # a ``wp.Volume`` exists, so ``wp.volume_index_to_world`` is not available to it.
    i, j, k = wp.tid()
    out_points[i, j, k] = lattice_position(lower, step, i, j, k)


# ---------------------------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------------------------


@wp.func
def record_point_bucket(
    slot: wp.int32,
    p: wp.int32,
    n_voxels: wp.int32,
    out_slots: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> wp.int32:
    # The histogram the min/max pooling opens with once it holds point ``p``'s voxel row --
    # probed from a volume (``point_slot``) or from ``voxel_down_sample``'s cell table
    # (``table_point_slot``): the row, written to ``out_slots`` unless the caller passed a
    # length-zero one, counted into its bucket. A point outside the grid goes into a sentinel
    # bucket past the last voxel and keeps the slot ``-1``. Returns the bucket.
    if out_slots.shape[0] > 0:
        out_slots[p] = slot
    bucket = wp.where(slot < 0, n_voxels, slot)
    wp.atomic_add(out_counts, bucket, 1)
    return bucket


@wp.func
def write_point_bucket(
    slot: wp.int32,
    p: wp.int32,
    n_voxels: wp.int32,
    out_slots: wp.array[wp.int32],
    out_buckets: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
) -> None:
    # The mean/sum pooling's first launch after the probe: each point's bucket, with
    # ``out_buckets`` / ``out_order`` the leading halves of ``radix_sort_pairs``' two double
    # buffers, keys and identity payload, so the sort needs no key copy and no separate payload
    # seed (the upper halves are scratch the sort fills before reading). No histogram: the
    # segments are read off the sorted buckets (``segment_reduce_vec3``), so the per-point atomic
    # into a few hot counters -- most of this launch at a coarse pitch -- is not paid. Shared by
    # ``bucket_point_slots`` and ``bucket_table_slots``, which differ only in the probe.
    if out_slots.shape[0] > 0:
        out_slots[p] = slot
    out_buckets[p] = wp.where(slot < 0, n_voxels, slot)
    out_order[p] = p


@wp.func
def pool_point_extremum(
    slot: wp.int32,
    p: wp.int32,
    values: wp.array[wp.vec3],
    n_voxels: wp.int32,
    largest: wp.bool,
    out_slots: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_values: wp.array[wp.vec3],
) -> None:
    # The min/max pooling in the probe launch itself: component-wise atomic min / max,
    # order-independent for floats, so no sort is needed. ``out_values`` arrives filled with the
    # +-inf the atomics reduce from; ``zero_empty_voxels`` then resets the voxels no point reached,
    # which only ``counts`` -- final once the launch ends -- can name. Shared by
    # ``pool_extremum_points`` and ``pool_extremum_table``, which differ only in the probe.
    record_point_bucket(slot, p, n_voxels, out_slots, out_counts)
    if slot < 0:
        return
    if largest:
        wp.atomic_max(out_values, slot, values[p])
    else:
        wp.atomic_min(out_values, slot, values[p])


@wp.func
def cell_hash_slot(cell: wp.vec3i, mask: wp.int32) -> wp.int32:
    # Home slot of a cell in ``voxel_down_sample``'s open-addressing table: x and y side by side in
    # one 64-bit word, xor'ed with z spread by the Fibonacci multiplier, then ``hash_slot``'s own
    # fold -- ``points.position_hash_slot``'s mixing over the cell's integers rather than a
    # position's bits. A collision is resolved by comparing cells, never trusted.
    x = wp.uint64(wp.uint32(cell[0]))
    y = wp.uint64(wp.uint32(cell[1]))
    z = wp.uint64(wp.uint32(cell[2]))
    return hash_slot(wp.int64(((x << wp.uint64(32)) | y) ^ (z * HASH_MULT_U64)), mask)


@wp.func
def same_cell(a: wp.vec3i, b: wp.vec3i) -> wp.bool:
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@wp.func
def grid_order_key(cell: wp.vec3i) -> wp.uint64:
    # ``Volume.get_voxels()``'s order *within one root tile* (4096 cells a side), so sorting a
    # tile's cells by it reproduces the rows ``allocate_by_voxels`` would number them in: NanoVDB
    # lays out upper-node children (bits 11..7 of each coordinate), then lower-node children (bits
    # 6..3), then leaf voxels (bits 2..0), each x-major. 36 bits. Verified against the builder on
    # both devices, negative and multi-tile sets included (``tests/test_voxels.py``).
    x = wp.uint64(wp.uint32(cell[0]))
    y = wp.uint64(wp.uint32(cell[1]))
    z = wp.uint64(wp.uint32(cell[2]))
    five = wp.uint64(31)
    four = wp.uint64(15)
    three = wp.uint64(7)
    key = (((x >> wp.uint64(7)) & five) << wp.uint64(31)) | (
        ((y >> wp.uint64(7)) & five) << wp.uint64(26)
    )
    key = key | (((z >> wp.uint64(7)) & five) << wp.uint64(21))
    key = key | (((x >> wp.uint64(3)) & four) << wp.uint64(17))
    key = key | (((y >> wp.uint64(3)) & four) << wp.uint64(13))
    key = key | (((z >> wp.uint64(3)) & four) << wp.uint64(9))
    key = key | ((x & three) << wp.uint64(6)) | ((y & three) << wp.uint64(3)) | (z & three)
    return key


@wp.func
def grid_root_key(cell: wp.vec3i) -> wp.uint64:
    # The root tile a cell lies in, in the builder's tile order: the *signed* arithmetic shift of
    # each coordinate by 12, lexicographic x-major (an unsigned ``uint32 >> 12`` key would put
    # negative tiles last, which is not what the builder does). Biased into 20 bits an axis.
    bias = wp.int32(1 << 19)
    x = wp.uint64((cell[0] >> 12) + bias)
    y = wp.uint64((cell[1] >> 12) + bias)
    z = wp.uint64((cell[2] >> 12) + bias)
    return (x << wp.uint64(40)) | (y << wp.uint64(20)) | z


@wp.kernel
def insert_point_cells(
    points: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.float32,
    mask: wp.int32,
    table: wp.array[wp.int32],
    out_unique: wp.array[wp.int32],
) -> None:
    # ``voxel_down_sample``'s voxel set without a ``wp.Volume``: one open-addressing table of point
    # indices, ``-1`` empty, keyed on the cell ``voxel_cell_indices`` would hand the builder. The
    # point that claims an empty slot appends the slot to ``out_unique`` through the cursor in
    # ``table``'s last element -- one past the ``mask + 1`` hashed slots, seeded ``-1`` by the same
    # fill, so it ends at the count minus one. The append order is arbitrary and is sorted away
    # afterwards (``unique_cell_keys``). A slot is read before it is claimed: a held slot never
    # changes again, so a non-empty read is final and the atomic is paid only on an apparently
    # empty slot -- most points repeat a cell a neighbour already claimed. ``table`` is scratch
    # state carried into ``assign_voxel_rows``.
    i = wp.int32(wp.tid())
    cell = voxel_cell(points[i], origin, inverse_size)
    h = cell_hash_slot(cell, mask)
    while True:
        held = table[h]
        if held == -1:
            held = wp.atomic_cas(table, h, wp.int32(-1), i)
            if held == -1:
                out_unique[wp.atomic_add(table, mask + 1, 1) + 1] = h
                return
        if same_cell(voxel_cell(points[held], origin, inverse_size), cell):
            return
        h = next_slot(h, mask)


@wp.kernel
def unique_cell_keys(
    points: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.float32,
    table: wp.array[wp.int32],
    unique: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # The in-tile grid-order key of each distinct cell, keyed to its append position: the leading
    # halves of a ``radix_sort_pairs`` double buffer. A cell is recovered from the point holding
    # its slot, the same ``voxel_cell`` arithmetic that inserted it.
    j = wp.int32(wp.tid())
    cell = voxel_cell(points[table[unique[j]]], origin, inverse_size)
    out_keys[j] = grid_order_key(cell)
    out_order[j] = j


@wp.kernel
def unique_cell_root_keys(
    points: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.float32,
    table: wp.array[wp.int32],
    unique: wp.array[wp.int32],
    order: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
) -> None:
    # The second, stable pass of the two-key grid order, for a cell set that spans more than one
    # root tile: the root key of the cell at each position of the in-tile sort.
    r = wp.int32(wp.tid())
    cell = voxel_cell(points[table[unique[order[r]]]], origin, inverse_size)
    out_keys[r] = grid_root_key(cell)


@wp.kernel
def assign_voxel_rows(
    points: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.float32,
    unique: wp.array[wp.int32],
    order: wp.array[wp.int32],
    table: wp.array[wp.int32],
    out_row_cells: wp.array[wp.vec3i],
) -> None:
    # Output row ``r`` is the ``r``-th distinct cell in grid order. Its table slot, which held the
    # claiming point, is rewritten to hold ``r`` -- rows are non-negative, so the empty test the
    # probe below walks by is unchanged -- and the row's cell is kept for that probe to compare.
    r = wp.int32(wp.tid())
    h = unique[order[r]]
    out_row_cells[r] = voxel_cell(points[table[h]], origin, inverse_size)
    table[h] = r


@wp.func
def table_point_slot(
    position: wp.vec3,
    translation: wp.vec3,
    inverse_size: wp.float32,
    zero: wp.float32,
    mask: wp.int32,
    table: wp.array[wp.int32],
    row_cells: wp.array[wp.vec3i],
) -> wp.int32:
    # ``point_slot`` against the cell table instead of a volume: the row of the voxel a point
    # *probes* into, ``-1`` when that cell is not in the set. The probe cell is ``point_cell``'s --
    # the volume's index transform, not ``voxel_cell``'s -- because a point on a cell boundary can
    # insert into one cell and probe into its neighbour, and ``pool_by_voxel`` has always pooled by
    # the probe. ``map_world_to_index`` is that transform bit for bit.
    cell = index_space_cell(map_world_to_index(position, translation, inverse_size, zero))
    h = cell_hash_slot(cell, mask)
    while True:
        row = table[h]
        if row == -1:
            return -1
        if same_cell(row_cells[row], cell):
            return row
        h = next_slot(h, mask)
    return -1


@wp.kernel
def bucket_point_slots(
    volume: wp.uint64,
    points: wp.array[wp.vec3],
    n_voxels: wp.int32,
    out_slots: wp.array[wp.int32],
    out_buckets: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
) -> None:
    # The mean/sum pooling's first launch over a volume; ``bucket_table_slots`` is the same over
    # ``voxel_down_sample``'s cell table, ``pool_extremum_points`` the min/max twin.
    p = wp.int32(wp.tid())
    slot = point_slot(volume, points[p])
    write_point_bucket(slot, p, n_voxels, out_slots, out_buckets, out_order)


@wp.kernel
def bucket_table_slots(
    points: wp.array[wp.vec3],
    translation: wp.vec3,
    inverse_size: wp.float32,
    zero: wp.float32,
    mask: wp.int32,
    table: wp.array[wp.int32],
    row_cells: wp.array[wp.vec3i],
    n_voxels: wp.int32,
    out_slots: wp.array[wp.int32],
    out_buckets: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
) -> None:
    p = wp.int32(wp.tid())
    slot = table_point_slot(points[p], translation, inverse_size, zero, mask, table, row_cells)
    write_point_bucket(slot, p, n_voxels, out_slots, out_buckets, out_order)


@wp.kernel
def segment_reduce_vec3(
    order: wp.array[wp.int32],
    values: wp.array[wp.vec3],
    buckets: wp.array[wp.int32],
    average: wp.bool,
    out_values: wp.array[wp.vec3],
) -> None:
    # One thread per voxel walking its segment in index order: the sum is bitwise reproducible,
    # which a float ``wp.atomic_add`` over the points would not be. ``buckets`` is the sorted
    # bucket prefix, so voxel ``v``'s segment is ``[lower_bound(v), lower_bound(v + 1))`` -- the
    # same bounds the old histogram's inclusive scan carried, two binary searches per voxel
    # instead of an atomic per point, a zeroed buffer and a scan launch.
    #
    # Serial per voxel on purpose, and starved at a coarse pitch (a few thousand voxels over
    # millions of points): a lane-strided block sum would fill the device but sum in a different
    # order, and the mean's bits are this function's contract.
    v = wp.int32(wp.tid())
    start = binary_search_index_left(buckets, v)
    end = binary_search_index_left(buckets, v + 1)
    total = wp.vec3(0.0, 0.0, 0.0)
    for j in range(start, end):
        total = total + values[order[j]]
    if average and end > start:
        total = total / wp.float32(end - start)
    out_values[v] = total


@wp.kernel
def pool_extremum_points(
    volume: wp.uint64,
    points: wp.array[wp.vec3],
    values: wp.array[wp.vec3],
    n_voxels: wp.int32,
    largest: wp.bool,
    out_slots: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_values: wp.array[wp.vec3],
) -> None:
    p = wp.int32(wp.tid())
    slot = point_slot(volume, points[p])
    pool_point_extremum(slot, p, values, n_voxels, largest, out_slots, out_counts, out_values)


@wp.kernel
def pool_extremum_table(
    points: wp.array[wp.vec3],
    translation: wp.vec3,
    inverse_size: wp.float32,
    zero: wp.float32,
    mask: wp.int32,
    table: wp.array[wp.int32],
    row_cells: wp.array[wp.vec3i],
    largest: wp.bool,
    out_slots: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_values: wp.array[wp.vec3],
) -> None:
    # ``voxel_down_sample`` pools its own points, so the payload is ``points`` itself.
    p = wp.int32(wp.tid())
    slot = table_point_slot(points[p], translation, inverse_size, zero, mask, table, row_cells)
    n_voxels = wp.int32(row_cells.shape[0])
    pool_point_extremum(slot, p, points, n_voxels, largest, out_slots, out_counts, out_values)


@wp.kernel
def zero_empty_voxels(counts: wp.array[wp.int32], out_values: wp.array[wp.vec3]) -> None:
    # A voxel no point landed in pools to zero, for every pooling; the min/max atomics left it at
    # the +-inf they reduce from.
    v = wp.int32(wp.tid())
    if counts[v] == 0:
        out_values[v] = wp.vec3(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------------------------


@wp.func
def neighbor_cell(
    cells: wp.array2d[wp.int32], v: wp.int32, offsets: wp.array2d[wp.int32], m: wp.int32
) -> wp.vec3i:
    # Cell ``v`` displaced by stencil row ``m``. The four kernels in the two sections below -- the
    # morphology candidates and completeness passes, and the box-face count and emit passes --
    # all walk a stencil this way and wrote the three component sums out longhand; one statement
    # of it rather than four; all four compile to byte-identical SASS, so it is free.
    i, j, k = row_triple(cells, v)
    di, dj, dk = row_triple(offsets, m)
    return wp.vec3i(i + di, j + dj, k + dk)


@wp.func
def neighbor_slot(
    volume: wp.uint64,
    cells: wp.array2d[wp.int32],
    v: wp.int32,
    offsets: wp.array2d[wp.int32],
    m: wp.int32,
) -> wp.int32:
    # Grid row of cell ``v``'s stencil neighbour ``m``, or ``-1`` when that neighbour is empty --
    # the probe the completeness, count and emit passes below each run once per stencil row.
    #
    # It returns the *slot* rather than a boolean, which is ``cell_slot``'s convention above and is
    # load-bearing here: the three callers want opposite polarities, and Warp lowers a ``not`` on a
    # bool as a select rather than by flipping the comparison, so a boolean helper makes one of
    # them pay an extra select per unrolled stencil row (measured: 48 SASS instructions on
    # ``count_box_faces`` at sm_120). Handing back the slot lets each caller keep the comparison it
    # already had, and the shared arithmetic still has one home.
    cell = neighbor_cell(cells, v, offsets, m)
    return wp.volume_lookup_index(volume, cell[0], cell[1], cell[2])


@wp.kernel
def neighborhood_candidates(
    voxels: wp.array2d[wp.int32], neighbors: wp.array2d[wp.int32], out_cells: wp.array2d[wp.int32]
) -> None:
    v, m = wp.tid()
    cell = neighbor_cell(voxels, v, neighbors, m)
    write_row_triple(out_cells, v * neighbors.shape[0] + m, cell[0], cell[1], cell[2])


@wp.kernel
def neighborhood_complete(
    volume: wp.uint64,
    voxels: wp.array2d[wp.int32],
    neighbors: wp.array2d[wp.int32],
    interior: wp.bool,
    out_flags: wp.array[wp.int32],
) -> None:
    # 1 when every neighbour of the voxel is occupied (an interior voxel), 0 otherwise. Neighbours
    # of a voxel live in the same 8-cubed leaf most of the time, so the probes are cache-local.
    #
    # ``interior`` is warp-uniform and selects which of the two complementary answers to write:
    # ``erode`` wants the interior set and ``surface_voxels`` its complement. One selector rather
    # than a second kernel, because the complement is this kernel's own result negated -- writing
    # it here costs nothing, where a separate pass costs a launch and a full round trip of the
    # flags through global memory.
    #
    # Measured 1.2-1.3x on ``surface_voxels``, byte-identical.
    v = wp.int32(wp.tid())
    complete = wp.int32(1)
    for m in range(neighbors.shape[0]):
        if neighbor_slot(volume, voxels, v, neighbors, m) < 0:
            complete = wp.int32(0)
    out_flags[v] = wp.where(interior, complete, 1 - complete)


@wp.func
def span_cell(axis: wp.int32, a: wp.int32, b: wp.int32, t: wp.int32) -> wp.vec3i:
    # ``(a, b)`` index the two axes other than ``axis``, ``t`` runs along ``axis``.
    if axis == 0:
        return wp.vec3i(t, a, b)
    if axis == 1:
        return wp.vec3i(a, t, b)
    return wp.vec3i(a, b, t)


@wp.kernel
def fill_axis_span(
    occupancy: wp.array3d[wp.bool],
    axis: wp.int32,
    length: wp.int32,
    accumulate: wp.bool,
    out_filled: wp.array3d[wp.bool],
) -> None:
    # One thread per line along ``axis``: mark every cell between the first and the last occupied
    # one. trimesh's ``ops.fill_orthographic`` intersects the three axes' results.
    #
    # ``accumulate`` (warp-uniform) intersects into ``out_filled`` rather than overwriting it, so
    # the three axis passes write one lattice and need no separate intersection pass between them.
    # That read is race-free: every cell lies on exactly one line per axis, so the thread that
    # reads a cell here is the only one writing it in this launch.
    a, b = wp.tid()
    first = wp.int32(-1)
    last = wp.int32(-1)
    for t in range(length):
        cell = span_cell(axis, a, b, t)
        if occupancy[cell[0], cell[1], cell[2]]:
            if first < 0:
                first = t
            last = t
    for t in range(length):
        cell = span_cell(axis, a, b, t)
        inside = first >= 0 and t >= first and t <= last
        if accumulate:
            inside = inside and out_filled[cell[0], cell[1], cell[2]]
        out_filled[cell[0], cell[1], cell[2]] = inside


@wp.func
def flat_cell_index(i: wp.int32, j: wp.int32, k: wp.int32, ny: wp.int32, nz: wp.int32) -> wp.int32:
    return ravel_index(i, j, k, ny, nz)


@wp.kernel
def flood_init_parent(occupancy: wp.array3d[wp.bool], out_parents: wp.array[wp.int32]) -> None:
    # ECL-CC initialisation over the *empty* complement, with the 6-neighbour stencil implicit:
    # three backward probes, no edge list. An occupied cell is its own singleton and never unions.
    i, j, k = wp.tid()
    ny = occupancy.shape[1]
    nz = occupancy.shape[2]
    v = flat_cell_index(i, j, k, ny, nz)
    out_parents[v] = v
    if occupancy[i, j, k]:
        return
    if i > 0 and not occupancy[i - 1, j, k]:
        out_parents[v] = flat_cell_index(i - 1, j, k, ny, nz)
        return
    if j > 0 and not occupancy[i, j - 1, k]:
        out_parents[v] = flat_cell_index(i, j - 1, k, ny, nz)
        return
    if k > 0 and not occupancy[i, j, k - 1]:
        out_parents[v] = flat_cell_index(i, j, k - 1, ny, nz)


@wp.kernel
def flood_hook(occupancy: wp.array3d[wp.bool], parents: wp.array[wp.int32]) -> None:
    # The three backward neighbours own each undirected edge exactly once, so every 6-connection
    # between two empty cells is hooked once. ``rep_v`` is carried across the three, ECL-CC's
    # ``vstat``.
    i, j, k = wp.tid()
    if occupancy[i, j, k]:
        return
    ny = occupancy.shape[1]
    nz = occupancy.shape[2]
    v = flat_cell_index(i, j, k, ny, nz)
    rep_v = find_representative(parents, v)
    if i > 0 and not occupancy[i - 1, j, k]:
        rep_v = ecl_hook_edge(parents, rep_v, flat_cell_index(i - 1, j, k, ny, nz))
    if j > 0 and not occupancy[i, j - 1, k]:
        rep_v = ecl_hook_edge(parents, rep_v, flat_cell_index(i, j - 1, k, ny, nz))
    if k > 0 and not occupancy[i, j, k - 1]:
        rep_v = ecl_hook_edge(parents, rep_v, flat_cell_index(i, j, k - 1, ny, nz))


@wp.kernel
def mark_outside_roots(
    occupancy: wp.array3d[wp.bool], labels: wp.array[wp.int32], out_outside: wp.array[wp.bool]
) -> None:
    # The padded shell is empty by construction, so its components are exactly the "outside".
    i, j, k = wp.tid()
    nx = occupancy.shape[0]
    ny = occupancy.shape[1]
    nz = occupancy.shape[2]
    on_shell = i == 0 or j == 0 or k == 0 or i == nx - 1 or j == ny - 1 or k == nz - 1
    if not on_shell or occupancy[i, j, k]:
        return
    out_outside[labels[flat_cell_index(i, j, k, ny, nz)]] = True


@wp.func
def write_dense_candidate(
    row: wp.int32,
    base: wp.vec3i,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    occupied: wp.bool,
    out_cells: wp.array2d[wp.int32],
    out_mask: wp.array[wp.int32],
) -> None:
    # One node of a dense lattice as a candidate for ``Volume.allocate_by_voxels``: its world cell
    # ``base + (i, j, k)`` and the 0/1 ``point_mask`` flag saying whether to keep it. The three
    # kernels that end a dense pass in a grid build write it this way, so a lattice's occupancy
    # never has to be stored as a lattice of its own just to be read back by ``occupied_cells``.
    write_row_triple(out_cells, row, base[0] + i, base[1] + j, base[2] + k)
    out_mask[row] = wp.where(occupied, 1, 0)


@wp.kernel
def enclosed_cell_candidates(
    occupancy: wp.array3d[wp.bool],
    labels: wp.array[wp.int32],
    outside: wp.array[wp.bool],
    base: wp.vec3i,
    out_cells: wp.array2d[wp.int32],
    out_mask: wp.array[wp.int32],
) -> None:
    # The flood fill's answer -- an occupied cell, or an empty one whose component does not reach
    # the padded shell -- written as grid-build candidates directly: ``occupied_cells``' output
    # with the filled value computed in a register rather than read from a filled lattice.
    i, j, k = wp.tid()
    row = flat_cell_index(i, j, k, occupancy.shape[1], occupancy.shape[2])
    filled = occupancy[i, j, k]
    if not filled:
        filled = not outside[labels[row]]
    write_dense_candidate(row, base, i, j, k, filled, out_cells, out_mask)


# ---------------------------------------------------------------------------------------------
# Dense conversion and meshing
# ---------------------------------------------------------------------------------------------


@wp.func
def cell_is_occupied(
    volume: wp.uint64, base: wp.vec3i, i: wp.int32, j: wp.int32, k: wp.int32
) -> wp.bool:
    # The dense-conversion counterpart of ``neighbor_slot``: these kernels index by their own
    # ``wp.tid()`` triple rather than through a stencil table, and want the boolean rather than the
    # row, so the offset arithmetic is a lattice one and does not go through ``neighbor_cell``.
    return wp.volume_lookup_index(volume, base[0] + i, base[1] + j, base[2] + k) >= 0


@wp.kernel
def dense_occupancy(volume: wp.uint64, base: wp.vec3i, out_occupancy: wp.array3d[wp.bool]) -> None:
    i, j, k = wp.tid()
    out_occupancy[i, j, k] = cell_is_occupied(volume, base, i, j, k)


@wp.kernel
def dense_field(volume: wp.uint64, base: wp.vec3i, out_field: wp.array3d[wp.float32]) -> None:
    i, j, k = wp.tid()
    out_field[i, j, k] = wp.where(cell_is_occupied(volume, base, i, j, k), 1.0, 0.0)


@wp.kernel
def occupied_cells(
    occupancy: wp.array3d[wp.bool],
    base: wp.vec3i,
    out_cells: wp.array2d[wp.int32],
    out_mask: wp.array[wp.int32],
) -> None:
    i, j, k = wp.tid()
    row = flat_cell_index(i, j, k, occupancy.shape[1], occupancy.shape[2])
    write_dense_candidate(row, base, i, j, k, occupancy[i, j, k], out_cells, out_mask)


@wp.kernel
def resampled_cell_candidates(
    volume: wp.uint64,
    lower: wp.vec3,
    step: wp.vec3,
    shape: wp.vec3i,
    base: wp.vec3i,
    out_cells: wp.array2d[wp.int32],
    out_mask: wp.array[wp.int32],
) -> None:
    # One cell of a new lattice, kept when the old grid holds its centre: ``lattice_points``,
    # ``point_occupancy`` and ``occupied_cells`` over the same ``(i, j, k)`` in one pass, so neither
    # the centre lattice nor its occupancy is ever stored. The centre is ``lattice_position`` of the
    # node lattice whose node 0 is the first new cell's centre -- the same call ``lattice_points``
    # makes -- and the probe is ``point_slot``, the one ``point_occupancy`` makes.
    i, j, k = wp.tid()
    row = flat_cell_index(i, j, k, shape[1], shape[2])
    occupied = point_slot(volume, lattice_position(lower, step, i, j, k)) >= 0
    write_dense_candidate(row, base, i, j, k, occupied, out_cells, out_mask)


@wp.kernel
def cell_corner_indices(
    corner_volume: wp.uint64, voxels: wp.array2d[wp.int32], out_corners: wp.array2d[wp.int32]
) -> None:
    # Corner ``c`` of a cell is ``cell + (c >> 2, (c >> 1) & 1, c & 1)`` -- x-major binary counting.
    v = wp.int32(wp.tid())
    for c in range(8):
        i = voxels[v, 0] + (c >> 2)
        j = voxels[v, 1] + ((c >> 1) & 1)
        k = voxels[v, 2] + (c & 1)
        out_corners[v, c] = wp.volume_lookup_index(corner_volume, i, j, k)


@wp.kernel
def corner_positions(
    volume: wp.uint64, corners: wp.array2d[wp.int32], out_positions: wp.array[wp.vec3]
) -> None:
    # Corner ``(i, j, k)`` is the *lower* corner of cell ``(i, j, k)``. A cell is centred on its
    # integer index-space coordinate, so its lower corner sits half a voxel below on every axis --
    # which is a shift in *index* space, where it is the NanoVDB convention itself rather than a
    # constant re-derived from the grid's translation.
    c = wp.int32(wp.tid())
    out_positions[c] = wp.volume_index_to_world(
        volume,
        wp.vec3(
            wp.float32(corners[c, 0]) - 0.5,
            wp.float32(corners[c, 1]) - 0.5,
            wp.float32(corners[c, 2]) - 0.5,
        ),
    )


@wp.kernel
def count_box_faces(
    volume: wp.uint64,
    voxels: wp.array2d[wp.int32],
    neighbors: wp.array2d[wp.int32],
    cull_internal: wp.bool,
    out_counts: wp.array[wp.int32],
) -> None:
    v = wp.int32(wp.tid())
    if not cull_internal:
        out_counts[v] = 6
        return
    exposed = wp.int32(0)
    for d in range(6):
        if neighbor_slot(volume, voxels, v, neighbors, d) < 0:
            exposed += 1
    out_counts[v] = exposed


@wp.kernel
def emit_box_faces(
    volume: wp.uint64,
    voxels: wp.array2d[wp.int32],
    corners: wp.array2d[wp.int32],
    neighbors: wp.array2d[wp.int32],
    face_corners: wp.array2d[wp.int32],
    offsets: wp.array[wp.int32],
    cull_internal: wp.bool,
    out_faces: wp.array[wp.int32],
) -> None:
    # Two triangles per exposed cube face, wound so the normal points away from the voxel.
    v = wp.int32(wp.tid())
    quad = offsets[v]
    for d in range(6):
        if cull_internal and neighbor_slot(volume, voxels, v, neighbors, d) >= 0:
            continue
        a = corners[v, face_corners[d, 0]]
        b = corners[v, face_corners[d, 1]]
        c = corners[v, face_corners[d, 2]]
        e = corners[v, face_corners[d, 3]]
        base = quad * 6
        out_faces[base + 0] = a
        out_faces[base + 1] = b
        out_faces[base + 2] = c
        out_faces[base + 3] = a
        out_faces[base + 4] = c
        out_faces[base + 5] = e
        quad += 1
