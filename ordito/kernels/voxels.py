"""
Kernels for ``ordito.voxels``: cell indexing, tri-box voxelization, morphology, dense conversion.

Every kernel here that reads a grid takes the volume's ``uint64`` id and probes it with
``wp.volume_lookup_index``, which is ``O(1)`` and returns the voxel's linear index (``-1`` when the
cell is empty). That index is the row index of ``Volume.get_voxels()``, so a per-voxel payload is a
plain ``wp.array`` and no side table is ever built.

The half-voxel convention is the wrapper's: a volume whose translation is ``origin + 0.5 * s`` has
NanoVDB voxel ``i`` covering world ``[origin + i * s, origin + (i + 1) * s)``, so
``floor(world_to_index(p) + 0.5)`` is the cell containing ``p``.

**Two halves of this module convert between cells and world positions differently, and the split is
deliberate.** Every kernel *downstream* of a built grid -- ``point_cell``,
``cell_center_positions``, ``corner_position`` -- goes through ``wp.volume_world_to_index`` /
``wp.volume_index_to_world``, so the convention above is read off the object that defines it
instead of re-implemented. The *voxelization* kernels -- ``voxel_cell``, ``voxel_cell_indices``,
``triangle_lattice_window`` -- run before any volume exists: they produce the cells
``Volume.allocate_by_voxels`` then builds a grid from, so there is no volume id to pass and the
``(origin, voxel_size)`` scalar form is required rather than preferred. Measured perf-neutral either
way, so this is a single-source-of-truth split and not a speed one.
"""

import warp as wp

from ordito.kernels.algorithms.connected_components import ecl_hook_edge, find_representative
from ordito.kernels.array import (
    append_slot,
    append_slot_if,
    binary_search_index,
    cube_corner_offset,
    curve_cell,
    curve_index,
    lattice_position,
    ravel_index,
)
from ordito.kernels.grouping import hash_slot_words3, key_set_insert, next_slot
from ordito.kernels.predicates import triangle_aabb, triangle_aabb_overlap
from ordito.kernels.reduce import GroupSumPass, peer_counted_add
from ordito.kernels.scatter import (
    ARRIVAL_POOL_PEERS,
    finish_run_sum,
    peer_extremum_vec3,
    scatter_add_vector,
)
from ordito.kernels.triangles import face_vertices, row_triple, write_row_triple

wp.set_module_options({"enable_backward": False})

# ---------------------------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------------------------


@wp.func
def voxel_lattice_position(position: wp.vec3, origin: wp.vec3, inverse_size: wp.float32) -> wp.vec3:
    # A world position in lattice units: cell ``(i, j, k)`` is the box ``[i, i + 1] x [j, j + 1] x
    # [k, k + 1]`` here, with integer corners that are exact in float32. The one place the
    # world-to-lattice map is spelled, so a cell index and a tri-box test in these units round
    # identically.
    return (position - origin) * inverse_size


@wp.func
def voxel_cell(position: wp.vec3, origin: wp.vec3, inverse_size: wp.float32) -> wp.vec3i:
    # Integer voxel a position falls in, for the grid anchored at ``origin`` with cell width
    # ``1 / inverse_size``. ``wp.floor`` rather than a cast, so negative coordinates round the same
    # way positive ones do (a C-style truncation would fold the two cells either side of the origin
    # into one).
    local = voxel_lattice_position(position, origin, inverse_size)
    return wp.vec3i(
        wp.int32(wp.floor(local[0])), wp.int32(wp.floor(local[1])), wp.int32(wp.floor(local[2]))
    )


@wp.func
def voxel_cell_axes(position: wp.vec3, origin: wp.vec3, inverse_size: wp.vec3) -> wp.vec3i:
    # ``voxel_cell`` on a grid whose cells have their own width along each axis (a grid stretched
    # to fit a box, MeshLab's clustering grid). With three equal widths it is ``voxel_cell`` bit for
    # bit: the same product per component, floored the same way.
    local = wp.cw_mul(position - origin, inverse_size)
    return wp.vec3i(
        wp.int32(wp.floor(local[0])), wp.int32(wp.floor(local[1])), wp.int32(wp.floor(local[2]))
    )


@wp.func
def voxel_cell_center(cell: wp.vec3i, origin: wp.vec3, voxel_size: wp.vec3) -> wp.vec3:
    # World position of a cell's centre: the inverse of ``voxel_cell_axes`` above, up to the
    # half-voxel that names the centre rather than the lower corner.
    #
    # The hand-rolled form rather than ``wp.volume_index_to_world`` (which
    # ``cell_center_positions`` below uses, and which the module docstring names as the convention)
    # because every caller here is *pre-volume*: it holds an origin and a cell width and no
    # ``wp.Volume`` handle to ask. The two agree to 3.58e-07 over 3 929 voxels -- float32 rounding,
    # not a convention difference -- and the measurement is recorded on ``cell_center_positions``.
    return origin + wp.cw_mul(
        wp.vec3(wp.float32(cell[0]), wp.float32(cell[1]), wp.float32(cell[2])) + wp.vec3(0.5),
        voxel_size,
    )


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
def triangle_lattice_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_index: wp.int32,
    origin: wp.vec3,
    inverse_size: wp.float32,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    # A face's corners in lattice units (``voxel_lattice_position``). The count and test passes
    # below both start here, so the window one reserves and the window the other walks come from
    # the same rounded corners.
    v0, v1, v2 = face_vertices(vertices, faces, face_index)
    return (
        voxel_lattice_position(v0, origin, inverse_size),
        voxel_lattice_position(v1, origin, inverse_size),
        voxel_lattice_position(v2, origin, inverse_size),
    )


@wp.func
def triangle_lattice_window(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3) -> tuple[wp.vec3i, wp.vec3i]:
    # Inclusive lower/upper cell of every *closed* lattice cell the triangle's closed AABB meets,
    # for corners already in lattice units. Shared by the count and test passes below, which MUST
    # enumerate the same window: the second writes into the slots the first reserved, so a window
    # that disagreed by one cell would write out of range.
    #
    # The lower end is ``ceil(min) - 1``, not ``floor(min)``: the two agree unless the bound sits
    # exactly on a lattice plane, and there the cell *below* the plane touches the triangle too (a
    # face of a box lying on a cell boundary meets both cells sharing it). ``floor`` would omit
    # that cell from the window, so the closed test could never accept it.
    #
    # Open3D walks ``round((max - min) / vs) + 2`` cells from ``floor(min)``: wider above, but it
    # shares the omission below, so on a triangle lying on a cell plane it misses the cell under the
    # plane. Off the planes a cell disjoint from the triangle's closed AABB cannot overlap the
    # triangle, and the *accepted* sets agree.
    lower, upper = triangle_aabb(p0, p1, p2)
    return (
        wp.vec3i(
            wp.int32(wp.ceil(lower[0])) - 1,
            wp.int32(wp.ceil(lower[1])) - 1,
            wp.int32(wp.ceil(lower[2])) - 1,
        ),
        wp.vec3i(
            wp.int32(wp.floor(upper[0])), wp.int32(wp.floor(upper[1])), wp.int32(wp.floor(upper[2]))
        ),
    )


@wp.kernel
def count_triangle_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    origin: wp.vec3,
    inverse_size: wp.float32,
    over_budget: wp.int64,
    out_counts: wp.array[wp.int64],
) -> None:
    # Each triangle's window size, exact up to ``over_budget`` (the caller's budget plus one) and
    # saturated there: any saturated count already puts the scanned total over the budget, and a
    # total of saturated counts stays far inside int64, so the budget is decided on the exact scan.
    f = wp.int32(wp.tid())
    p0, p1, p2 = triangle_lattice_vertices(vertices, faces, f, origin, inverse_size)
    lo, hi = triangle_lattice_window(p0, p1, p2)
    # Each axis span widens to int64 *before* the subtraction, not after, so the subtraction itself
    # cannot already overflow int32 for a triangle whose window is that wide (the convention
    # ``kernels/graph.py``'s packed labels key uses: widen the operands, not their difference).
    span_x = wp.int64(hi[0]) - wp.int64(lo[0]) + wp.int64(1)
    span_y = wp.int64(hi[1]) - wp.int64(lo[1]) + wp.int64(1)
    span_z = wp.int64(hi[2]) - wp.int64(lo[2]) + wp.int64(1)
    # The product of three int32-wide spans can overflow int64 itself, so the saturation is decided
    # on the float64 product, which is exact far past any budget; the wrapped int64 product is
    # then discarded.
    approximate = wp.float64(span_x) * wp.float64(span_y) * wp.float64(span_z)
    out_counts[f] = wp.where(
        approximate < wp.float64(over_budget), span_x * span_y * span_z, over_budget
    )


@wp.func
def triangle_candidate(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int64],
    origin: wp.vec3,
    inverse_size: wp.float32,
    item: wp.int32,
) -> tuple[wp.vec3i, wp.bool]:
    # The cell of one (triangle, candidate cell) work item and whether the triangle meets it. One
    # item a thread: per-triangle window sizes span orders of magnitude on any real mesh, so a
    # thread-per-triangle launch is load-imbalanced by that same factor. ``binary_search_index``
    # recovers the owning triangle from the flat work-item id (``wp.lower_bound`` clamps to
    # ``n - 1`` and would misattribute the last window). ``test_triangle_candidates`` writes every
    # item and its verdict; ``append_unique_triangle_cells`` keeps the accepted cells once each.
    #
    # The per-item decode below (``span_y``/``span_z``/``plane``/``i``/``j``/``k``) stays int32,
    # unlike ``count_triangle_candidates``'s widened axis spans above -- a single axis wide enough
    # to overflow int32 on its own would still misdecode here, but reaching that needs
    # ``max_candidates`` raised past its ``2**31`` default (already impractical: the candidate
    # buffers alone would be tens of GB at that width). Left as a documented residual rather than
    # widening this hot loop's arithmetic to int64 for a practically unreachable input; revisit only
    # if a caller legitimately needs a wider ``max_candidates``.
    f = binary_search_index(offsets, wp.int64(item)) - 1
    p0, p1, p2 = triangle_lattice_vertices(vertices, faces, f, origin, inverse_size)
    lo, hi = triangle_lattice_window(p0, p1, p2)
    span_y = hi[1] - lo[1] + 1
    span_z = hi[2] - lo[2] + 1

    local = wp.int32(wp.int64(item) - offsets[f])
    plane = span_y * span_z
    i = local // plane
    rest = local % plane
    j = rest // span_z
    k = rest % span_z
    cell = wp.vec3i(lo[0] + i, lo[1] + j, lo[2] + k)

    # The overlap test runs in lattice units, where the cell is exactly ``[i, i + 1]`` per axis
    # (centre and half-width are exact in float32). In world units the cell's faces would be
    # ``origin + i * voxel_size`` rounded, which need not agree with the rounding that placed the
    # triangle in the window, so a triangle on a cell plane could fall outside every cell it is
    # tested against. Window and test read the same rounded corners instead: a corner exactly on a
    # plane is on it for both, and the closed test accepts the cells on either side of it.
    half = wp.vec3(0.5, 0.5, 0.5)
    center = wp.vec3(
        wp.float32(cell[0]) + 0.5, wp.float32(cell[1]) + 0.5, wp.float32(cell[2]) + 0.5
    )
    return cell, triangle_aabb_overlap(center, half, p0, p1, p2)


@wp.kernel
def test_triangle_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int64],
    origin: wp.vec3,
    inverse_size: wp.float32,
    out_cells: wp.array2d[wp.int32],
    out_mask: wp.array[wp.int32],
) -> None:
    # Every candidate and its verdict, for a ``point_mask`` build: the form a small candidate set
    # takes, where the builder's cost is its floor and a dedup's table and count read would not pay
    # (``append_unique_triangle_cells`` is the large form).
    item = wp.int32(wp.tid())
    cell, accepted = triangle_candidate(vertices, faces, offsets, origin, inverse_size, item)
    write_row_triple(out_cells, item, cell[0], cell[1], cell[2])
    out_mask[item] = wp.where(accepted, 1, 0)


@wp.func
def box_cell_key(cell: wp.vec3i, box_lower: wp.vec3i, box_shape: wp.vec3i) -> wp.int64:
    # A cell's C-order rank in the box ``[box_lower, box_lower + box_shape)``: a key that is
    # distinct for distinct cells of the box and never ``-1`` (``encode_key``'s empty sentinel).
    return (
        wp.int64(cell[0] - box_lower[0]) * wp.int64(box_shape[1]) + wp.int64(cell[1] - box_lower[1])
    ) * wp.int64(box_shape[2]) + wp.int64(cell[2] - box_lower[2])


@wp.func
def append_unique_cell(
    cell: wp.vec3i,
    kept: wp.bool,
    box_lower: wp.vec3i,
    box_shape: wp.vec3i,
    slot_key: wp.array[wp.int64],
    mask: wp.int32,
    counter: wp.array[wp.int32],
    out_cells: wp.array2d[wp.int32],
):
    # A kept cell appended to ``out_cells`` by the first thread to insert it into the key set, so a
    # producer that meets each cell many times (a cloud's points, a mesh's tri-box candidates) hands
    # ``Volume.allocate_by_voxels`` its distinct cells alone: the builder keys and sorts every row
    # it is given, duplicates and masked rows included, and that sort was most of a large call
    # (lucy's 28 M faces at 64 cells across: ~14 of 17 ms). Called by every lane, the append is one
    # ballot a warp (``append_slot_if``). The builder sorts its keys, so arrival order decides
    # nothing.
    inserted = wp.int32(0)
    if kept:
        inserted = key_set_insert(box_cell_key(cell, box_lower, box_shape), slot_key, mask)
    slot = append_slot_if(counter, 0, inserted)
    if slot >= 0:
        write_row_triple(out_cells, slot, cell[0], cell[1], cell[2])


@wp.kernel
def append_unique_triangle_cells(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int64],
    origin: wp.vec3,
    inverse_size: wp.float32,
    box_lower: wp.vec3i,
    box_shape: wp.vec3i,
    slot_key: wp.array[wp.int64],
    mask: wp.int32,
    counter: wp.array[wp.int32],
    out_cells: wp.array2d[wp.int32],
) -> None:
    # ``test_triangle_candidates``' items, each accepted cell kept once (``append_unique_cell``).
    # ``slot_key`` and ``counter`` are scratch carried out to the build.
    item = wp.int32(wp.tid())
    cell, accepted = triangle_candidate(vertices, faces, offsets, origin, inverse_size, item)
    append_unique_cell(cell, accepted, box_lower, box_shape, slot_key, mask, counter, out_cells)


@wp.kernel
def append_unique_point_cells(
    points: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.float32,
    box_lower: wp.vec3i,
    box_shape: wp.vec3i,
    slot_key: wp.array[wp.int64],
    mask: wp.int32,
    counter: wp.array[wp.int32],
    out_cells: wp.array2d[wp.int32],
) -> None:
    # ``voxel_cell_indices``' cells, each kept once (``append_unique_cell``).
    v = wp.int32(wp.tid())
    cell = voxel_cell(points[v], origin, inverse_size)
    append_unique_cell(cell, True, box_lower, box_shape, slot_key, mask, counter, out_cells)


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


@wp.kernel
def cell_curve_codes(
    cells: wp.array2d[wp.int32],
    bits: wp.int32,
    curve: wp.int32,
    out_codes: wp.array[wp.int64],
    out_outside: wp.array[wp.int32],
) -> None:
    # ``voxels.space_filling_codes``: each cell's index along ``curve`` (``kernels/array.
    # curve_index``). A coordinate outside ``[0, 2**bits)`` raises ``out_outside``, which the
    # wrapper reads instead of a separate range reduction.
    v = wp.int32(wp.tid())
    a, b, c = row_triple(cells, v)
    cell = wp.vec3i(a, b, c)
    side = 1 << bits
    if wp.min(cell) < 0 or wp.max(cell) >= side:
        out_outside[0] = 1
    out_codes[v] = curve_index(cell, bits, curve)


@wp.kernel
def curve_code_cells(
    codes: wp.array[wp.int64],
    bits: wp.int32,
    curve: wp.int32,
    out_cells: wp.array2d[wp.int32],
    out_outside: wp.array[wp.int32],
) -> None:
    # ``voxels.cells_from_space_filling_codes``: ``cell_curve_codes``' inverse (``kernels/array.
    # curve_cell``), a code outside ``[0, 2**(3 * bits))`` raising ``out_outside``.
    v = wp.int32(wp.tid())
    code = codes[v]
    # The bits above the code, not a comparison with ``1 << 3 * bits``, which wraps at 21 bits.
    if code < wp.int64(0) or (code >> wp.int64(3 * bits)) != wp.int64(0):
        out_outside[0] = 1
    cell = curve_cell(code, bits, curve)
    write_row_triple(out_cells, v, cell[0], cell[1], cell[2])


@wp.func
def cell_slot(volume: wp.uint64, cells: wp.array2d[wp.int32], v: wp.int32) -> wp.int32:
    # The grid and ``get_voxels()`` share one numbering, so a cell's slot is also its payload row;
    # ``-1`` means the cell is not in the grid. Shared by the occupancy kernels below, which
    # differ only in whether the caller wants a mask or the builder's ``int32`` flags.
    return wp.volume_lookup_index(volume, cells[v, 0], cells[v, 1], cells[v, 2])


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
def append_cells_by_occupancy(
    volume: wp.uint64,
    cells: wp.array2d[wp.int32],
    present: wp.bool,
    counter: wp.array[wp.int32],
    out_cells: wp.array2d[wp.int32],
) -> None:
    # ``cell_occupancy``'s test, each kept cell appended at the next slot of ``counter[0]`` for a
    # set operation to rebuild its grid from: the builder handed the whole cell array with a
    # ``point_mask`` keys and sorts the rejected cells too (``append_dense_candidate``).
    v = wp.int32(wp.tid())
    if (cell_slot(volume, cells, v) >= 0) == present:
        i, j, k = row_triple(cells, v)
        write_row_triple(out_cells, append_slot(counter, 0), i, j, k)


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
    out_order: wp.array[wp.int32],
) -> None:
    # Each cell's key and the identity payload, into the leading halves of a radix sort's
    # double-width buffers, so the sort reads them where they are written.
    #
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
    out_order[v] = v


@wp.kernel
def lattice_points(lower: wp.vec3, step: wp.vec3, out_points: wp.array3d[wp.vec3]) -> None:
    # A dense node lattice, rank-3 destination; the position is ``array.lattice_position``. This
    # is one of the two sites this module's docstring names as running before
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
    # A coarse grid crowds a scan's consecutive points into a few hot bins, so a warp's peers count
    # once (``peer_counted_add``).
    if out_slots.shape[0] > 0:
        out_slots[p] = slot
    bucket = wp.where(slot < 0, n_voxels, slot)
    peer_counted_add(out_counts, bucket)
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
    # segments are read off the sorted buckets (``scatter.sorted_run_sums``), so the per-point
    # atomic into a few hot counters -- most of this launch at a coarse pitch -- is not paid.
    # Shared by ``bucket_point_slots`` and ``bucket_table_slots``, which differ only in the probe.
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
    if slot >= 0:
        peer_extremum_vec3(out_values, slot, values[p], largest)


@wp.func
def pool_point_sum(
    slot: wp.int32,
    p: wp.int32,
    values: wp.array[wp.vec3],
    n_voxels: wp.int32,
    sums: GroupSumPass,
    out_sums: wp.array[wp.vec3],
    out_slots: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # The mean/sum pooling in the probe launch itself, in arrival order: the form the deterministic
    # setting's off mode takes, where the fixed-order form sorts every point by voxel first
    # (``write_point_bucket``). A voxel's points arrive together from a scan, so a warp's peers add
    # once (``ARRIVAL_POOL_PEERS``). ``finish_pooled_means`` divides by ``out_counts`` afterwards.
    # Shared by ``pool_sum_points`` and ``pool_sum_table``, which differ only in the probe.
    record_point_bucket(slot, p, n_voxels, out_slots, out_counts)
    if slot >= 0:
        scatter_add_vector(out_sums, sums, slot, values[p], ARRIVAL_POOL_PEERS)


@wp.func
def cell_hash_slot(cell: wp.vec3i, mask: wp.int32) -> wp.int32:
    # Home slot of a cell in ``voxel_down_sample``'s open-addressing table. A collision is
    # resolved by comparing cells, never trusted.
    return hash_slot_words3(cell[0], cell[1], cell[2], mask)


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
    # ``table``'s last element (``append_slot``) -- one past the ``mask + 1`` hashed slots, seeded
    # ``-1`` by the same fill, so it ends at the count minus one. The append order is arbitrary and
    # is sorted away afterwards (``unique_cell_keys``). A slot is read before it is claimed: a held
    # slot never changes again, so a non-empty read is final and the atomic is paid only on an
    # apparently empty slot -- most points repeat a cell a neighbour already claimed. ``table`` is
    # scratch state carried into ``assign_voxel_rows``.
    i = wp.int32(wp.tid())
    cell = voxel_cell(points[i], origin, inverse_size)
    h = cell_hash_slot(cell, mask)
    while True:
        held = table[h]
        if held == -1:
            held = wp.atomic_cas(table, h, wp.int32(-1), i)
            if held == -1:
                out_unique[append_slot(table, mask + 1) + 1] = h
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
def pool_sum_points(
    volume: wp.uint64,
    points: wp.array[wp.vec3],
    values: wp.array[wp.vec3],
    n_voxels: wp.int32,
    sums: GroupSumPass,
    out_sums: wp.array[wp.vec3],
    out_slots: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    p = wp.int32(wp.tid())
    slot = point_slot(volume, points[p])
    pool_point_sum(slot, p, values, n_voxels, sums, out_sums, out_slots, out_counts)


@wp.kernel
def pool_sum_table(
    points: wp.array[wp.vec3],
    translation: wp.vec3,
    inverse_size: wp.float32,
    zero: wp.float32,
    mask: wp.int32,
    table: wp.array[wp.int32],
    row_cells: wp.array[wp.vec3i],
    sums: GroupSumPass,
    out_sums: wp.array[wp.vec3],
    out_slots: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # ``voxel_down_sample`` pools its own points, so the payload is ``points`` itself.
    p = wp.int32(wp.tid())
    slot = table_point_slot(points[p], translation, inverse_size, zero, mask, table, row_cells)
    n_voxels = wp.int32(row_cells.shape[0])
    pool_point_sum(slot, p, points, n_voxels, sums, out_sums, out_slots, out_counts)


@wp.kernel
def finish_pooled_means(counts: wp.array[wp.int32], out_values: wp.array[wp.vec3]) -> None:
    # The arrival-order mean pooling's divide, the fixed-order form's ``finish_run_sum``: an empty
    # voxel stays at the zero its sum started from.
    v = wp.int32(wp.tid())
    out_values[v] = finish_run_sum(out_values[v], counts[v], True)


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
def dilation_candidates(
    volume: wp.uint64,
    voxels: wp.array2d[wp.int32],
    neighbors: wp.array2d[wp.int32],
    counter: wp.array[wp.int32],
    out_cells: wp.array2d[wp.int32],
) -> None:
    # A voxel (its stencil's zero row) and each of its stencil neighbours the grid does not hold,
    # appended at the next slot of ``counter[0]``: the dilation's cells, with an empty cell once
    # for every voxel reaching it, and no occupied neighbour, which the voxel's own row already
    # gives. The builder sorts and merges the rows, so neither their order nor the repeats matter.
    # Appending an empty cell only from the first stencil row that reaches it from an occupied
    # voxel (up to 26 more probes a candidate) measured 0.80-0.85x on a thin shell at
    # connectivity 26 but 1.04-1.08x on a solid at 6, ``dragon`` at 256 a side (2026-11-02):
    # declined.
    v, m = wp.tid()
    di, dj, dk = row_triple(neighbors, m)
    cell = neighbor_cell(voxels, v, neighbors, m)
    if (di == 0 and dj == 0 and dk == 0) or wp.volume_lookup_index(
        volume, cell[0], cell[1], cell[2]
    ) < 0:
        write_row_triple(out_cells, append_slot(counter, 0), cell[0], cell[1], cell[2])


@wp.kernel
def neighborhood_complete_cells(
    volume: wp.uint64,
    voxels: wp.array2d[wp.int32],
    neighbors: wp.array2d[wp.int32],
    interior: wp.bool,
    counter: wp.array[wp.int32],
    out_cells: wp.array2d[wp.int32],
) -> None:
    # A voxel appended at the next slot of ``counter[0]`` when every neighbour is occupied (an
    # interior voxel), or with ``interior`` false when one is not (``append_dense_candidate`` has
    # why the builder takes the kept rows rather than a mask). Neighbours of a voxel live in the
    # same 8-cubed leaf most of the time, so the probes are cache-local.
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
    if (complete == 1) == interior:
        i, j, k = row_triple(voxels, v)
        write_row_triple(out_cells, append_slot(counter, 0), i, j, k)


@wp.func
def span_cell(axis: wp.int32, a: wp.int32, b: wp.int32, t: wp.int32) -> wp.vec3i:
    # ``(a, b)`` index the two axes other than ``axis``, ``t`` runs along ``axis``.
    if axis == 0:
        return wp.vec3i(t, a, b)
    if axis == 1:
        return wp.vec3i(a, t, b)
    return wp.vec3i(a, b, t)


@wp.func
def span_line_offsets(shape: wp.vec3i) -> wp.vec3i:
    # Where each axis's lines start in ``axis_span_bounds``' table: every line along ``axis`` is
    # indexed ``a * shape[other_b] + b`` over the two other axes, axis 0's first.
    return wp.vec3i(0, shape[1] * shape[2], shape[1] * shape[2] + shape[0] * shape[2])


@wp.kernel
def axis_span_bounds(occupancy: wp.array3d[wp.bool], out_spans: wp.array[wp.vec2i]) -> None:
    # One thread per line of the lattice along each of its three axes: the first and last occupied
    # cell along it (``-1, -1`` for an empty line), for ``occupied_cells_in_spans``. trimesh's
    # ``ops.fill_orthographic`` intersects the three axes' ``[first, last]`` intervals.
    #
    # The lines only read; the cell pass after them intersects the three intervals at each cell and
    # writes the grid-build candidates coalesced, so no filled lattice is written. Declined,
    # measured (device time, 50**3 to 400**3 cells), for the passes this replaced: the first pass
    # probing the grid instead of reading ``dense_occupancy``'s lattice, 0.39-0.69x, and a line
    # pass writing the grid-build candidates instead of a cell pass reading its lattice,
    # 0.44-0.85x. A thread per *line* walks its cells in sequence and writes them strided, where a
    # pass that writes is best a thread per cell, coalesced.
    line = wp.int32(wp.tid())
    shape = wp.vec3i(occupancy.shape[0], occupancy.shape[1], occupancy.shape[2])
    offsets = span_line_offsets(shape)
    axis = wp.int32(0)
    if line >= offsets[2]:
        axis = 2
    elif line >= offsets[1]:
        axis = 1
    local = line - offsets[axis]
    # The second of the two other axes: ``k`` for axes 0 and 1, ``j`` for axis 2.
    width = wp.where(axis == 2, shape[1], shape[2])
    a = local // width
    b = local - a * width
    first = wp.int32(-1)
    last = wp.int32(-1)
    for t in range(shape[axis]):
        cell = span_cell(axis, a, b, t)
        if occupancy[cell[0], cell[1], cell[2]]:
            if first < 0:
                first = t
            last = t
    out_spans[line] = wp.vec2i(first, last)


@wp.func
def in_span(span: wp.vec2i, t: wp.int32) -> wp.bool:
    # Whether ``t`` lies in a line's ``[first, last]`` occupied interval; never for an empty line.
    return span[0] >= 0 and t >= span[0] and t <= span[1]


@wp.func
def flat_cell_index(i: wp.int32, j: wp.int32, k: wp.int32, ny: wp.int32, nz: wp.int32) -> wp.int32:
    return ravel_index(i, j, k, ny, nz)


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
def flatten_and_mark_outside(
    occupancy: wp.array3d[wp.bool],
    parents: wp.array[wp.int32],
    out_labels: wp.array[wp.int32],
    out_outside: wp.array[wp.bool],
) -> None:
    # ``connected_components.ecl_flatten`` over the flood forest, plus the outside marking it
    # feeds: the padded shell is empty by construction, so its components are exactly the
    # "outside", and an empty shell cell marks its own component's root as the label is found
    # rather than reading the labels back in a second pass. ``parents`` is compressed in place by
    # the finds, the benign race ``find_representative`` documents.
    i, j, k = wp.tid()
    nx = occupancy.shape[0]
    ny = occupancy.shape[1]
    nz = occupancy.shape[2]
    label = find_representative(parents, flat_cell_index(i, j, k, ny, nz))
    out_labels[flat_cell_index(i, j, k, ny, nz)] = label
    on_shell = i == 0 or j == 0 or k == 0 or i == nx - 1 or j == ny - 1 or k == nz - 1
    if on_shell and not occupancy[i, j, k]:
        out_outside[label] = True


@wp.func
def append_dense_candidate(
    counter: wp.array[wp.int32],
    base: wp.vec3i,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    occupied: wp.bool,
    out_cells: wp.array2d[wp.int32],
):
    # One node of a dense lattice as a candidate for ``Volume.allocate_by_voxels``, appended when it
    # is kept: its world cell ``base + (i, j, k)`` at the next slot of ``counter[0]``. The kernels
    # that end a dense pass in a grid build write it this way, so a lattice's occupancy is never
    # stored as a lattice of its own, and the builder is handed the kept cells alone: given every
    # cell with a ``point_mask`` it keys and sorts the rejected ones too (2.2-4.0x its time at 27 %
    # kept). The builder sorts its keys, so the slots' arrival order decides nothing.
    if occupied:
        write_row_triple(out_cells, append_slot(counter, 0), base[0] + i, base[1] + j, base[2] + k)


@wp.kernel
def enclosed_cell_candidates(
    occupancy: wp.array3d[wp.bool],
    labels: wp.array[wp.int32],
    outside: wp.array[wp.bool],
    base: wp.vec3i,
    counter: wp.array[wp.int32],
    out_cells: wp.array2d[wp.int32],
) -> None:
    # The flood fill's answer -- an occupied cell, or an empty one whose component does not reach
    # the padded shell -- written as grid-build candidates directly: ``occupied_cells``' output
    # with the filled value computed in a register rather than read from a filled lattice.
    i, j, k = wp.tid()
    row = flat_cell_index(i, j, k, occupancy.shape[1], occupancy.shape[2])
    filled = occupancy[i, j, k]
    if not filled:
        filled = not outside[labels[row]]
    append_dense_candidate(counter, base, i, j, k, filled, out_cells)


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
def flood_init_parent(
    volume: wp.uint64,
    base: wp.vec3i,
    out_occupancy: wp.array3d[wp.bool],
    out_parents: wp.array[wp.int32],
) -> None:
    # The flood fill's dense occupancy (``dense_occupancy``'s probe) and its ECL-CC initialisation
    # over the *empty* complement in one pass, with the 6-neighbour stencil implicit: three
    # backward probes, no edge list. The backward neighbours' occupancy is probed from the grid
    # rather than read from ``out_occupancy``, which other threads of this launch are writing. An
    # occupied cell is its own singleton and never unions.
    i, j, k = wp.tid()
    ny = out_occupancy.shape[1]
    nz = out_occupancy.shape[2]
    v = flat_cell_index(i, j, k, ny, nz)
    occupied = cell_is_occupied(volume, base, i, j, k)
    out_occupancy[i, j, k] = occupied
    out_parents[v] = v
    if occupied:
        return
    if i > 0 and not cell_is_occupied(volume, base, i - 1, j, k):
        out_parents[v] = flat_cell_index(i - 1, j, k, ny, nz)
        return
    if j > 0 and not cell_is_occupied(volume, base, i, j - 1, k):
        out_parents[v] = flat_cell_index(i, j - 1, k, ny, nz)
        return
    if k > 0 and not cell_is_occupied(volume, base, i, j, k - 1):
        out_parents[v] = flat_cell_index(i, j, k - 1, ny, nz)


@wp.kernel
def dense_field(volume: wp.uint64, base: wp.vec3i, out_field: wp.array3d[wp.float32]) -> None:
    i, j, k = wp.tid()
    out_field[i, j, k] = wp.where(cell_is_occupied(volume, base, i, j, k), 1.0, 0.0)


@wp.kernel
def occupied_cells(
    occupancy: wp.array3d[wp.bool],
    base: wp.vec3i,
    counter: wp.array[wp.int32],
    out_cells: wp.array2d[wp.int32],
) -> None:
    i, j, k = wp.tid()
    append_dense_candidate(counter, base, i, j, k, occupancy[i, j, k], out_cells)


@wp.kernel
def occupied_cells_in_spans(
    spans: wp.array[wp.vec2i],
    shape: wp.vec3i,
    base: wp.vec3i,
    counter: wp.array[wp.int32],
    out_cells: wp.array2d[wp.int32],
) -> None:
    # ``occupied_cells`` of the orthographic fill: a cell is kept when it lies inside the occupied
    # interval of its line along every axis (``axis_span_bounds``), the intersection trimesh's
    # ``ops.fill_orthographic`` forms, read off the three spans with no filled lattice between.
    i, j, k = wp.tid()
    offsets = span_line_offsets(shape)
    inside = (
        in_span(spans[offsets[0] + j * shape[2] + k], i)
        and in_span(spans[offsets[1] + i * shape[2] + k], j)
        and in_span(spans[offsets[2] + i * shape[1] + j], k)
    )
    append_dense_candidate(counter, base, i, j, k, inside, out_cells)


@wp.kernel
def resampled_cell_candidates(
    volume: wp.uint64,
    lower: wp.vec3,
    step: wp.vec3,
    shape: wp.vec3i,
    base: wp.vec3i,
    counter: wp.array[wp.int32],
    out_cells: wp.array2d[wp.int32],
) -> None:
    # One cell of a new lattice, kept when the old grid holds its centre: ``lattice_points``,
    # ``point_occupancy`` and ``occupied_cells`` over the same ``(i, j, k)`` in one pass, so neither
    # the centre lattice nor its occupancy is ever stored. The centre is ``lattice_position`` of the
    # node lattice whose node 0 is the first new cell's centre -- the same call ``lattice_points``
    # makes -- and the probe is ``point_slot``, the one ``point_occupancy`` makes.
    i, j, k = wp.tid()
    occupied = point_slot(volume, lattice_position(lower, step, i, j, k)) >= 0
    append_dense_candidate(counter, base, i, j, k, occupied, out_cells)


@wp.kernel
def append_owned_corners(
    volume: wp.uint64,
    voxels: wp.array2d[wp.int32],
    counter: wp.array[wp.int32],
    out_corners: wp.array2d[wp.int32],
) -> None:
    # Every corner of the grid's cells once, for the corner grid's build: corner ``c`` of voxel
    # ``v`` is appended by ``v`` unless an active voxel reaches the same corner through a lower
    # corner index (the voxel ``corner - cube_corner_offset(j)``, ``j < c``), so each corner has
    # one owner and the builder sorts the distinct corners alone, not all eight a voxel. Arrival
    # order decides nothing (the builder sorts). ``counter`` is scratch carried out to the build.
    v, c = wp.tid()
    o = cube_corner_offset(c)
    corner = wp.vec3i(voxels[v, 0] + o[0], voxels[v, 1] + o[1], voxels[v, 2] + o[2])
    owned = wp.int32(1)
    for j in range(c):
        p = corner - cube_corner_offset(j)
        if wp.volume_lookup_index(volume, p[0], p[1], p[2]) >= 0:
            owned = wp.int32(0)
            break
    slot = append_slot_if(counter, 0, owned)
    if slot >= 0:
        write_row_triple(out_corners, slot, corner[0], corner[1], corner[2])


@wp.kernel
def cell_corner_indices(
    corner_volume: wp.uint64, voxels: wp.array2d[wp.int32], out_corners: wp.array2d[wp.int32]
) -> None:
    # Corner ``c`` of a cell is ``cell + cube_corner_offset(c)`` -- x-major binary counting.
    v = wp.int32(wp.tid())
    for c in range(8):
        o = cube_corner_offset(c)
        out_corners[v, c] = wp.volume_lookup_index(
            corner_volume, voxels[v, 0] + o[0], voxels[v, 1] + o[1], voxels[v, 2] + o[2]
        )


@wp.func
def corner_position(volume: wp.uint64, corners: wp.array2d[wp.int32], c: wp.int32) -> wp.vec3:
    # Corner ``(i, j, k)`` is the *lower* corner of cell ``(i, j, k)``. A cell is centred on its
    # integer index-space coordinate, so its lower corner sits half a voxel below on every axis --
    # which is a shift in *index* space, where it is the NanoVDB convention itself rather than a
    # constant re-derived from the grid's translation.
    return wp.volume_index_to_world(
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
    corners: wp.array2d[wp.int32],
    out_counts: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
) -> None:
    # Launched over ``max(n_corners, n_voxels)``: each corner's world position
    # (``corner_position``) and each voxel's exposed-face count, independent writes in one launch.
    v = wp.int32(wp.tid())
    if v < corners.shape[0]:
        out_positions[v] = corner_position(volume, corners, v)
    if v >= voxels.shape[0]:
        return
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
