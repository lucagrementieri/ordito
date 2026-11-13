"""Global reductions: tiled ``wp.tile_load`` reductions generated from a shared template."""

import math
from typing import Any, NamedTuple

import warp as wp

# Tile-reduction builtins are resolved only inside kernel source text and are not
# exposed as Python-scope attributes (``wp.tile_max`` etc. raise ``AttributeError``),
# unlike the atomic and scalar builtins (``wp.atomic_max``, ``wp.max`` …). To let the
# kernel factories below *capture* them, pull the concrete ``Function`` objects out of
# Warp's builtin registry. A captured builtin is emitted inline at codegen (same as a
# literal ``wp.tile_sum`` call), so it templates on the tile dtype correctly — wrapping
# it in a ``@wp.func`` instead would erase the dtype and produce ambiguous C++ overloads.
from warp._src.context import builtin_functions as _warp_builtins

from ordito.constants import (
    INT32_MAX_CONSTANT,
    TILE_1D,
    TILE_2D,
    TILES_PER_BLOCK_1D,
    UINT64_MAX_CONSTANT,
)
from ordito.kernels.array import (
    KernelTable,
    atomic_min_packed_box,
    csr_row_long,
    device_is_cpu,
    is_close_scalar,
    is_close_vec3,
    ordered_double_bits,
    ordered_float_bits,
    popcount32,
)

wp.set_module_options({"enable_backward": False})

_tile_min = _warp_builtins["tile_min"]
_tile_max = _warp_builtins["tile_max"]
_tile_sum = _warp_builtins["tile_sum"]


@wp.func
def tile_chunk(n: wp.int32, chunk: wp.int32, width: wp.int32) -> tuple[wp.int32, wp.int32]:
    # The ``(offset, remaining)`` of one chunk of a 1-D array: where chunk ``chunk`` of ``width``
    # elements starts, and how many elements are left from there. ``remaining <= 0`` means the
    # chunk is past the end and the block has nothing to do; a positive ``remaining`` below
    # ``width`` is the ragged last chunk, which the caller clamps with ``wp.min(remaining, width)``.
    #
    # Named because the *contract* is what goes wrong here, not the arithmetic: ``width`` is
    # ``TILE_1D`` for a one-tile-per-block kernel and ``TILES_PER_BLOCK_1D * TILE_1D`` for the
    # folding ones in this module, and a launch that assumes the wrong one silently folds every
    # element 16 times -- see ``blocks_1d``, which is the same hazard from the launch side.
    offset = chunk * width
    return offset, n - offset


# Elements one block of a 1-D global reduction owns. It is one shared name because both halves
# of the contract read it and they must read the *same* number: [`blocks_1d`] derives the launch
# width from it at Python scope, and a kernel-scope ``tile_chunk(n, chunk, ITEMS_PER_BLOCK_1D)``
# derives each block's slice from it. A launch and a kernel that disagree here do not fail -- they
# fold some elements twice and drop others, which for a sum is a wrong answer and for an extremum
# looks right. See [`blocks_1d`] for the same hazard stated from the launch side.
ITEMS_PER_BLOCK_1D = TILE_1D * TILES_PER_BLOCK_1D


@wp.func
def block_chunk(n: wp.int32, block: wp.int32, width: wp.int32) -> tuple[wp.int32, wp.int32]:
    # ``tile_chunk`` **clamped to the block's own share**: ``(offset, count)`` of the ``width``
    # elements block ``block`` owns, ``count <= 0`` past the end. The clamp is the part a loop
    # bounded by the count cannot omit -- ``tile_chunk``'s ``remaining`` runs to the end of the
    # array, so without it block 0 walks everything, which a sum gets wrong loudly and a
    # min/max/any/all silently. The one spelling of that clamp for every block-owned chunk, whatever
    # its width (a tile, the fold width, a kernel's own rows per block).
    offset, remaining = tile_chunk(n, block, width)
    return offset, wp.min(remaining, width)


@wp.func
def block_chunk_1d(n: wp.int32, block: wp.int32) -> tuple[wp.int32, wp.int32]:
    # ``block_chunk`` at the ``ITEMS_PER_BLOCK_1D`` fold width.
    return block_chunk(n, block, ITEMS_PER_BLOCK_1D)


# The three block-wide reductions of one value per lane, correct on **every** lane and a full
# block barrier either side, so a caller runs them outside its ``lane == 0`` commit guard. On the
# CPU device a block is one lane and each returns that lane's own value.
#
# ``block_sum`` passes ``preserve_type=True``: a plain ``wp.tile(v)`` of a vector decomposes it and
# sums its components together, where this reduces a ``wp.vec3`` / ``wp.mat33`` /
# ``wp.types.vector(length=N)`` componentwise in **one** tile reduction, bit-identical to one
# ``wp.tile_sum`` per component (same tree, same order; verified on both devices). So several
# same-dtype quantities a block folds are packed into one vector and pay one barrier, not one each.
# Generic wrappers over a *value* are fine; the ambiguity the note above describes is the tile-
# load factories', whose argument is a tile.


@wp.func
def block_sum(value: Any):
    """Sum of ``value`` over the block's lanes, componentwise for a vector or matrix."""
    return wp.tile_sum(wp.tile(value, preserve_type=True))[0]


@wp.func
def block_barrier() -> None:
    # A full block barrier: a block-wide reduction synchronizes before and after, and Warp spells
    # no bare ``__syncthreads``, so this is the one name for "every lane sees the writes before
    # it". The sum itself is discarded.
    _ = block_sum(wp.int32(0))


@wp.func
def block_min(value: Any):
    """Minimum of a scalar ``value`` over the block's lanes."""
    return wp.tile_min(wp.tile(value))[0]


@wp.func
def block_max(value: Any):
    """Maximum of a scalar ``value`` over the block's lanes."""
    return wp.tile_max(wp.tile(value))[0]


@wp.func
def block_running_sum(value: Any, carry: Any) -> tuple[Any, Any]:
    # One block-wide chunk of a running sum carried across chunks: the lane's inclusive prefix of
    # ``value`` plus ``carry``, and the carry past the chunk (``carry`` plus the last lane's
    # prefix), block-uniform. A block walking an array a ``wp.block_dim()``-wide chunk at a time
    # scans it in one launch, in a fixed order; on the CPU device a chunk is one lane, so the sum
    # runs in index order. Every lane must call it (the scan is a block barrier).
    scanned = wp.tile_scan_inclusive(wp.tile(value))
    return carry + wp.untile(scanned), carry + wp.tile_extract(scanned, wp.block_dim() - 1)


@wp.func
def block_argmin(value: Any, index: wp.int32) -> tuple[Any, wp.int32]:
    # The block-cooperative counterpart of ``update_argmin``: given one candidate per lane, the
    # smallest value over the block and the **lowest index among the lanes attaining it**. Every
    # lane holds the same pair afterwards, so any lane may store it.
    #
    # The two-stage form is what makes the winner independent of *which* lane saw it, and that is
    # the whole reason this is one function rather than three: a block reduction hands back a value
    # and not the lane that held it, so recovering the index is a second reduction with a tie-break
    # -- a *decision rule*, which diverges silently where duplicated arithmetic only reads badly.
    # Four sites need it: the hole-filling DP's apex choice, ball pivoting's pivot search,
    # ``proximity``'s straggler faces and ``bounds``' oriented-box candidate sweep.
    #
    # No ``wp.ref``, so unlike ``update_argmin`` this imposes no ``enable_backward=False`` on its
    # callers. Correct on the CPU device too, where ``wp.launch_tiled`` runs one lane per block and
    # both tiles hold that lane's own pair.
    block_value = block_min(value)
    attained = wp.where(value == block_value, index, INT32_MAX_CONSTANT)
    return block_value, block_min(attained)


@wp.func
def commit_block_sum(lane: wp.int32, local: Any, out: wp.array[Any], base: wp.int32):
    # The commit of every packed block fold: each lane's register vector, folded by one
    # ``block_sum`` (a barrier, so every lane runs it), then added component by component by lane 0
    # into ``out[base:base + len(local)]`` -- one atomic per slot per block, the ``reduce`` fold's
    # commit. ``wp.atomic_add`` reads trailing indices as array dimensions, not vector components,
    # which is why the components are committed one at a time.
    block = block_sum(local)
    if lane == 0:
        for slot in range(len(block)):
            wp.atomic_add(out, base + slot, block[slot])


@wp.func
def commit_block_total(lane: wp.int32, local: Any, out: wp.array[Any], slot: wp.int32):
    # The commit of a single-value block fold: each lane's register value (scalar, vector or
    # matrix, matching ``out``'s dtype), folded by one ``block_sum`` (a barrier, so every lane runs
    # it), then added whole by lane 0 into ``out[slot]`` -- one atomic per block.
    # ``commit_block_sum`` is the packed sibling that spreads a vector's components over slots.
    # Forward only: the taped ``metrics`` chamfer kernels keep the commit inline, because routing
    # it through this ``@wp.func`` broke their ``wp.Tape`` gradient
    # (``test_chamfer_points_to_points_loss_grad_on_either_backward_search``).
    block = block_sum(local)
    if lane == 0:
        wp.atomic_add(out, slot, block)


@wp.kernel
def index_extrema(
    indices: wp.array[wp.int32],
    n_indices: wp.int32,
    slot: wp.int32,
    out_extrema: wp.array[wp.int32],
) -> None:
    # ``_common.index_extrema``: the minimum and maximum of the first ``n_indices`` entries of an
    # index buffer, committed by ``atomic_min`` / ``atomic_max`` into ``out_extrema[slot]`` and
    # ``out_extrema[slot + 1]``, two zeroed slots -- of a buffer the caller allocates anyway where
    # it has one (``edges.edges_unique``'s sort payload). The zero seeds clamp the pair to
    # ``(min(0, minimum), max(0, maximum))``, which is exact wherever an index check acts: a
    # negative minimum, a non-negative maximum. That is what lets it skip ``reduce.minmax``'s
    # seeded result buffer, its fill and its dispatch. A block commits its minimum only when it is
    # negative. Lane-strided by ``wp.block_dim()``; integer, so exact in any order.
    # Measured through ``edges_unique`` with no ``n_vertices`` (2026-11-07, CUDA new/old, with
    # ``edges.face_edge_keys_inferred`` issued before the read): 0.78-0.81x on 1 k-82 k faces,
    # 0.87x at 330 k, 0.93x at 1.3 M; the device-read radix alone was worth 0.97x of that. CPU
    # 0.78-0.84x at 1 k faces, 0.97-1.02x from 20 k. Declined there: packing against
    # ``INDEX_RADIX_PAIR`` with no bound read at all, since the 64-bit sort costs the extra passes
    # the read saves (0.94-0.97x the old ``minmax`` path on 1 k-80 k faces, 1.08-1.15x from 1.3 M)
    # and loses to this one.
    block, lane = wp.tid()
    offset, count = block_chunk_1d(n_indices, block)
    low = wp.int32(0)
    high = wp.int32(0)
    for k in range(lane, count, wp.block_dim()):
        v = indices[offset + k]
        low = wp.min(low, v)
        high = wp.max(high, v)
    block_low = block_min(low)
    block_high = block_max(high)
    if lane == 0:
        if block_low < 0:
            wp.atomic_min(out_extrema, slot, block_low)
        wp.atomic_max(out_extrema, slot + 1, block_high)


# The run-to-run reproducible block commit, in the launch that forms the block sums.
#
# One ``atomic_add`` per block adds the blocks in arrival order, so on CUDA a float total's last
# bits move from call to call; integer sums are exact in any order and keep ``commit_block_sum``.
# Here each block instead stores its fold in its own row of a zeroed buffer and takes a ticket; the
# block that draws the last ticket adds every row in one fixed order (lanes stride the rows by
# ``wp.block_dim()``, ``block_sum`` combines them) and writes the totals. The buffer holds
# ``stride`` result slots, the ticket counter at ``stride`` and then one ``stride``-slot row per
# block, ``(n_blocks + 1) * stride + 1`` elements (``_launch.block_partials``), so a consumer reads
# the total in ``out[slot]`` exactly where the atomic commit left it. The last block resets the
# counter, so the next launch over the same buffer (a loop's next round, another column of it)
# needs no clearing. A fence between a block's row and its ticket is what makes the rows visible
# to the folding block (the CUDA ``threadFenceReduction`` pattern); on the CPU device the blocks
# run in turn and the fence compiles to nothing. One launch and one zeroed allocation, like the
# atomic commit; a separate fold launch cost more on a short call.
#
# A buffer of exactly ``stride`` elements (``_launch.block_partials`` with determinism off in
# ``ordito.config``) takes the atomic commit instead, trading reproducibility for the few
# microseconds the ticket, the fence and the last block's fold add to a short call.
_THREAD_FENCE = """
#if defined(__CUDA_ARCH__)
__threadfence();
#endif
"""


@wp.func_native(_THREAD_FENCE)
def thread_fence(): ...


@wp.func
def take_ticket(out: wp.array[Any], counter: wp.int32, unit: Any):
    # Lane 0's ticket, once it has stored its row: the count of rows stored before it, kept in
    # ``out[counter]`` in ``out``'s own dtype (whole counts are exact; ``unit`` is any value of it,
    # no longer read). The fence publishes the row before the ticket can be seen. A vector or
    # matrix counter is one atomic per component, so only a component's count is a ticket:
    # ``last_ticket_vector`` and ``last_ticket_matrix`` read the first. The one is
    # ``out.dtype(1.0)``, not a ``float32`` fill, which a ``float64`` vector or matrix rejects.
    thread_fence()
    return wp.atomic_add(out, counter, out.dtype(1.0))


@wp.func
def last_ticket(ticket: Any, n_blocks: wp.int32) -> wp.int32:
    # 1 when a scalar ticket is the last of ``n_blocks``.
    return wp.where(ticket == type(ticket)(wp.float32(n_blocks - 1)), 1, 0)


@wp.func
def last_ticket_vector(ticket: Any, n_blocks: wp.int32) -> wp.int32:
    # ``last_ticket`` on a vector counter's first component.
    return last_ticket(ticket[0], n_blocks)


@wp.func
def last_ticket_matrix(ticket: Any, n_blocks: wp.int32) -> wp.int32:
    # ``last_ticket`` on a matrix counter's first component.
    return last_ticket(ticket[0, 0], n_blocks)


@wp.func
def sum_strided_rows(
    lane: wp.int32,
    out: wp.array[Any],
    first: wp.int32,
    row_stride: wp.int32,
    n_rows: wp.int32,
    zero: Any,
):
    # This lane's share of ``sum(out[first + r * row_stride] for r in range(n_rows))``: rows
    # ``lane``, ``lane + block_dim``, ... in four interleaved running sums, so four loads are in
    # flight at a time where one dependent sum waited on each in turn (the last block's fold is
    # latency-bound: a 15k-block launch left it a few hundred loads a lane). Combined in a fixed
    # order, so the result depends only on the block size.
    step = wp.block_dim()
    t0 = zero
    t1 = zero
    t2 = zero
    t3 = zero
    r = lane
    while r + 3 * step < n_rows:
        t0 += out[first + r * row_stride]
        t1 += out[first + (r + step) * row_stride]
        t2 += out[first + (r + 2 * step) * row_stride]
        t3 += out[first + (r + 3 * step) * row_stride]
        r += 4 * step
    while r < n_rows:
        t0 += out[first + r * row_stride]
        r += step
    return (t0 + t1) + (t2 + t3)


@wp.func
def block_last_ticket(
    lane: wp.int32,
    out: wp.array[Any],
    counter: wp.int32,
    unit: Any,
    n_rows: wp.int32,
    is_last: wp.Function,
) -> wp.int32:
    # Lane 0's ticket on ``out[counter]`` once it has stored its row, read by ``is_last`` against
    # ``n_rows``: non-zero, in every lane, for the block that drew the last ticket. Every
    # reproducible commit draws its ticket here, so the fence-then-ticket order is written once.
    last = wp.int32(0)
    if lane == 0:
        last = is_last(take_ticket(out, counter, unit), n_rows)
    return block_max(last)


@wp.func
def last_atomic_block(
    lane: wp.int32, out_ticket: wp.array[wp.int32], n_blocks: wp.int32
) -> wp.int32:
    # The last block of a launch whose blocks committed by ``atomic_add`` (``ordito.config``'s
    # deterministic setting off): non-zero, in every lane, for the block drawing the last of
    # ``n_blocks`` integer tickets on ``out_ticket[0]``, whose reads then see every block's adds
    # (each block fenced before its ticket). That block resets the counter, so a loop's next round
    # needs no clearing. The reproducible commit has its last block already, the one that folds.
    # Shared by ``registration``'s round-closing blocks and ``points``' fused fit.
    last = block_last_ticket(lane, out_ticket, 0, wp.int32(0), n_blocks, last_ticket)
    if last != 0:
        thread_fence()
        if lane == 0:
            out_ticket[0] = wp.int32(0)
    return last


@wp.func
def stage_block_value(
    lane: wp.int32,
    value: Any,
    out: wp.array[Any],
    row: wp.int32,
    counter: wp.int32,
    n_rows: wp.int32,
    is_last: wp.Function,
) -> wp.int32:
    # A block's half of a reproducible single-value commit, at any layout: ``value`` stored at
    # ``out[row]`` and a ticket drawn on ``out[counter]`` (``block_last_ticket``).
    if lane == 0:
        out[row] = value
    return block_last_ticket(lane, out, counter, value, n_rows, is_last)


@wp.func
def fold_block_values(
    lane: wp.int32,
    out: wp.array[Any],
    first: wp.int32,
    row_stride: wp.int32,
    n_rows: wp.int32,
    zero: Any,
):
    # The last block's fold at any layout: ``out[first + r * row_stride]`` for every row ``r``,
    # added in one fixed order, after the fence that makes the other blocks' rows visible.
    thread_fence()
    return block_sum(sum_strided_rows(lane, out, first, row_stride, n_rows, zero))


@wp.func
def write_block_value(
    lane: wp.int32, out: wp.array[Any], result: wp.int32, counter: wp.int32, total: Any
) -> None:
    # The last block's write: ``total`` into ``out[result]`` and the ticket counter reset, so the
    # next launch over the same buffer (a loop's next round) needs no clearing.
    if lane == 0:
        out[result] = total
        out[counter] = out.dtype(0.0)


@wp.func
def stage_block_total(
    block: wp.int32,
    lane: wp.int32,
    folded: Any,
    out: wp.array[Any],
    slot: wp.int32,
    stride: wp.int32,
    n_blocks: wp.int32,
    is_last: wp.Function,
) -> wp.int32:
    # ``stage_block_value`` in the ``_launch.block_partials`` layout: the fold stored in slot
    # ``slot`` of the block's row, the ticket drawn on the counter at ``stride``.
    return stage_block_value(
        lane, folded, out, stride + 1 + block * stride + slot, stride, n_blocks, is_last
    )


@wp.func
def fold_block_totals(
    lane: wp.int32,
    out: wp.array[Any],
    slot: wp.int32,
    stride: wp.int32,
    n_blocks: wp.int32,
    zero: Any,
):
    # ``fold_block_values`` in the ``_launch.block_partials`` layout: every block's row of slot
    # ``slot``.
    return fold_block_values(lane, out, stride + 1 + slot, stride, n_blocks, zero)


@wp.func
def finish_block_total(
    lane: wp.int32,
    out: wp.array[Any],
    slot: wp.int32,
    stride: wp.int32,
    n_blocks: wp.int32,
    zero: Any,
):
    # The last block's half of a reproducible commit: every block's row of slot ``slot`` added in
    # one fixed order into the result slot, and the ticket counter reset for the next launch.
    write_block_value(
        lane, out, slot, stride, fold_block_totals(lane, out, slot, stride, n_blocks, zero)
    )


@wp.func
def stage_block_fold(
    block: wp.int32,
    lane: wp.int32,
    folded: Any,
    out: wp.array[Any],
    slot: wp.int32,
    stride: wp.int32,
    is_last: wp.Function,
) -> wp.int32:
    # A single-value block commit by its buffer's form: into a ``stride``-slot buffer (determinism
    # off) one ``atomic_add`` and 0; otherwise ``stage_block_total``, non-zero for the last block.
    if out.shape[0] == stride:
        if lane == 0:
            wp.atomic_add(out, slot, folded)
        return wp.int32(0)
    return stage_block_total(
        block, lane, folded, out, slot, stride, (out.shape[0] - stride - 1) // stride, is_last
    )


@wp.func
def store_folded_sum(
    block: wp.int32,
    lane: wp.int32,
    folded: Any,
    out: wp.array[Any],
    base: wp.int32,
    stride: wp.int32,
) -> wp.int32:
    # ``store_block_sum``'s commit of a block's row already folded (the same vector in every lane):
    # for a caller that scales the fold once a block rather than each lane's terms. Same return.
    if out.shape[0] == stride:
        if lane == 0:
            for slot in range(len(folded)):
                wp.atomic_add(out, base + slot, folded[slot])
        return wp.int32(0)
    n_blocks = (out.shape[0] - stride - 1) // stride
    if lane == 0:
        row = stride + 1 + block * stride + base
        for slot in range(len(folded)):
            out[row + slot] = folded[slot]
    last = block_last_ticket(lane, out, stride, folded[0], n_blocks, last_ticket)
    if last != 0:
        thread_fence()
        # Row by row: every slot of a row is an independent load, so ``len(folded)`` are in flight
        # where a slot-by-slot walk waited on each row in turn. A narrow row is walked two at a
        # time for twice that; a wide one (``registration``'s 25 and 44 slots) one at a time, which
        # a second accumulator would push out of registers.
        zero = type(folded[0])(wp.float32(0.0))
        even = folded * zero
        odd = folded * zero
        step = wp.block_dim()
        r = lane
        if len(folded) <= 16:
            while r + step < n_blocks:
                first = stride + 1 + r * stride + base
                second = first + step * stride
                for slot in range(len(even)):
                    even[slot] = even[slot] + out[first + slot]
                    odd[slot] = odd[slot] + out[second + slot]
                r += 2 * step
        while r < n_blocks:
            first = stride + 1 + r * stride + base
            for slot in range(len(even)):
                even[slot] = even[slot] + out[first + slot]
            r += step
        grand = block_sum(even + odd)
        if lane == 0:
            for slot in range(len(grand)):
                out[base + slot] = grand[slot]
            out[stride] = zero
    return last


@wp.func
def store_block_sum(
    block: wp.int32,
    lane: wp.int32,
    local: Any,
    out: wp.array[Any],
    base: wp.int32,
    stride: wp.int32,
) -> wp.int32:
    # The reproducible ``commit_block_sum``: the block's fold of each lane's register vector into
    # slots ``base:base + len(local)`` of its row, and of the result row once the last block has
    # added every row (see the note above). Every block of the launch must reach this, and every
    # lane of it (it is block-collective). The vector-row form of ``stage_block_value`` and
    # ``fold_block_values``: a row of ``len(local)`` slots stored and added at once. Returns
    # non-zero, in every lane, for the block that wrote the result row (lane 0 wrote it, so that
    # lane reads it back without a fence); 0 for every block of the atomic commit.
    return store_folded_sum(block, lane, block_sum(local), out, base, stride)


@wp.func
def store_folded_total(
    block: wp.int32,
    lane: wp.int32,
    folded: Any,
    out: wp.array[Any],
    slot: wp.int32,
    stride: wp.int32,
    is_last: wp.Function,
):
    # ``store_block_total``'s commit of a block sum already folded (the same value in every lane):
    # for a caller that folds in one type and commits in another, such as ``float32`` block sums
    # widened to ``float64`` before the blocks are added.
    if stage_block_fold(block, lane, folded, out, slot, stride, is_last) != 0:
        n_blocks = (out.shape[0] - stride - 1) // stride
        finish_block_total(lane, out, slot, stride, n_blocks, out.dtype(0.0))


@wp.func
def store_block_total(
    block: wp.int32,
    lane: wp.int32,
    local: Any,
    out: wp.array[Any],
    slot: wp.int32,
    stride: wp.int32,
    is_last: wp.Function,
):
    # ``store_block_sum`` for a single value (scalar, vector or matrix, matching ``out``'s dtype),
    # stored whole in slot ``slot`` of the block's row and of the result row. ``is_last`` reads the
    # ticket of ``out``'s kind: ``last_ticket``, ``last_ticket_vector`` or ``last_ticket_matrix``.
    store_folded_total(block, lane, block_sum(local), out, slot, stride, is_last)


@wp.func
def store_block_total_divided(
    block: wp.int32,
    lane: wp.int32,
    local: Any,
    out: wp.array[Any],
    slot: wp.int32,
    stride: wp.int32,
    is_last: wp.Function,
    divisor: wp.float32,
):
    # ``store_block_total`` whose result slot holds the total divided by ``divisor``, so a mean
    # needs no launch of its own. The reproducible form divides once, in the last block, the
    # division ``total / divisor`` per component a ``wp.map(wp.div, ...)`` makes; the
    # ``atomic_add`` form has no last block, so each block adds its own fold divided (arrival
    # order already leaves the last bits free there).
    folded = block_sum(local)
    if out.shape[0] == stride:
        if lane == 0:
            wp.atomic_add(out, slot, folded / divisor)
        return
    if stage_block_fold(block, lane, folded, out, slot, stride, is_last) != 0:
        n_blocks = (out.shape[0] - stride - 1) // stride
        zero = type(folded)(wp.float32(0.0))
        total = fold_block_totals(lane, out, slot, stride, n_blocks, zero)
        write_block_value(lane, out, slot, stride, total / divisor)


@wp.func
def store_sum_and_count(
    block: wp.int32,
    lane: wp.int32,
    total: wp.float64,
    count: wp.float64,
    out_sum_and_count: wp.array[wp.float64],
    stride: wp.int32,
):
    # The commit of a fused "mean over the entries that qualify" block fold: each lane's register
    # sum and count, as one ``store_block_sum`` pair into slots 0 and 1. Shared by
    # ``heat.upper_edge_length_sum_and_count``, ``heat.source_and_global_sums``,
    # ``points.neighbor_distance_moments``, ``reconstruction.positive_finite_sum_and_count`` and
    # ``registration.residual_moment``, which differ only in what qualifies.
    store_block_sum(block, lane, wp.vec2d(total, count), out_sum_and_count, 0, stride)


# ---------------------------------------------------------------------------
# Exact group sums: fixed point per group, committed by integer atomics.
#
# The reproducible form of a per-group float sum (a scatter into vertices, components, cells,
# lattice nodes). Integer additions are exact, so integer atomics give the same total in any
# commit order, on either device. A group has one slot per component (a scalar one, a vector or
# matrix ``n``, slot ``group * n + c``). Each term is converted to ``float64`` (exact for a
# ``float32``) and then to a count of a power-of-two unit fixed per group, in two passes over the
# terms (``GroupSumPass.phase``):
#
# - pass 0 raises the group's exponent field to the largest ``float64`` biased exponent among its
#   terms' components (``raise_group_sum_field``; 0 at allocation, ``GROUP_SUM_NONFINITE`` for a
#   non-finite component, which makes the group NaN);
# - pass 1 adds every component to its slot's words at the scale that field sets
#   (``add_group_sum_term``).
#
# The finish (``group_sum_value``) reads a slot's words back as one ``float64``; a consumer kernel
# may call it instead of the finish launch (``read_group_sum``, a scalar sum in either mode).
# ``kernels/scatter.scatter_add`` and its vector and matrix siblings are the entry points, and
# ``_launch.group_sums`` allocates the buffer.
#
# **The scale.** With ``h = headroom`` the caller's bound ``ceil(log2(terms per slot))``
# (``h <= 31``), a term of a group whose field is ``F`` is scaled to ``x = v * 2^k`` with
# ``k = 1084 - h - max(F, 1)``, so ``|x| < 2^(62 - h)`` and ``2^h`` terms sum below ``2^62``. The
# scale is applied in integer arithmetic on the term's mantissa (``group_sum_words``), never as a
# ``float64`` product: no ``float64`` arithmetic runs per term, and on this hardware ``float64``
# throughput was the cost (the ``float64``-multiply form ran 2x the integer one on coherent
# scatters). Scales are built from exponent bits (``group_sum_power``), never ``log2`` / ``pow``,
# whose rounding is device-dependent.
#
# **Width.** One ``int64`` word holds ``rint(x)``: a unit of ``2^-(62 - h) * P``, ``P`` the power
# of two above the group's largest component magnitude ``M`` (``P <= 2M``) -- the width for
# ``float32`` results, below ``float32`` precision of ``M`` for ``h <= 30``. Two words hold
# ``rint(x * 2^(62 - h))`` split at ``2^(62 - h)``, a unit of ``2^-(124 - 2h) * P``: the width for
# ``float64`` results. A word a term leaves zero is not committed, so a ``float32`` term costs one
# atomic at either width unless it lies more than ``38 - h`` binades below ``P``.
#
# **Bound.** For ``n <= 2^h`` terms of exact sum ``S``, the words hold ``S`` within
# ``n * unit / 2``; exact when every term is a multiple of the unit, which a ``float32`` term
# within ``38 - h`` (one word) or ``100 - 2h`` (two words) binades of ``P`` is. The finish reads
# two words with an absolute error below ``2^-(114 - h) * P`` and a relative one below ``2^-52``
# of the result, and one word with the relative one alone; a ``float32`` result is rounded once
# more. Identical on CPU and CUDA (integer words, then the same IEEE roundings).
#
# **The field.** One field per group, or one shared by every group (``n_fields == 1``): the shared
# form needs no gathered field read in either pass and sets every group's unit from the largest
# term overall, so a group far below it loses the accuracy above. ``base_field`` floors every
# field: a caller that knows a bound on every term's magnitude sets it and skips pass 0
# (``_launch.group_sums(bound=)``), and a loop whose terms cannot outgrow its first round's fields
# keeps them (``GroupSums.add_only``). A term whose exponent exceeds its group's field makes the
# group NaN (``GROUP_SUM_NONFINITE``) rather than wrapping its words, so a broken bound is loud.
# Fields are per group rather than per component: a ``wp.vec4`` splat lattice kept four ``int64``
# fields a node and ran 1.9x the device time of one.
#
# **Hot slots.** Where every active lane of a warp adds to one slot (a single group, a long sorted
# run), the words are summed across the warp first and one lane commits (``warp_lanes_sum``):
# 10 M terms into one slot ran 0.011x the time of the per-term ``float32`` atomic. Declined:
# matching any subset of lanes (``__match_any_sync``), 1.5x the plain commit on coherent runs of
# six and no gain on few randomly ordered groups (512 cells, 2 M terms), whose collisions are
# across warps, not within one. A per-vertex float scatter does take it
# (``kernels/scatter.peer_summed_add``): its commit is one float atomic a component.
#
# **Cost against the forms it may replace** (CUDA, deterministic setting on; small = 20 k faces or
# 100 k points, large = 5.2 M faces or 2 M points). It wins against ``repair``'s single-word
# fixed point (0.1-0.8x device time large, at par small) and against the sort-based forms on small
# inputs (Poisson splat 0.5x, cluster means 0.5x wall), and loses where one pass of float
# atomics or a sort with an amortized run walk was already reproducible enough: the exact
# ``float32``-in-``float64`` vertex scatters (1.2-1.9x with a caller bound, 2.5-3x with pass 0),
# the ARAP run walk over a sort made once per call (2x on a scrambled large mesh, 0.8x on an
# ordered one), and the large-lattice splat and fine cluster means (1.5-1.8x: twice the bytes of
# ``float32`` accumulators to clear, scatter into and read back). Those sites keep their forms.
#
# With ``words`` empty the pass is the arrival-order float form: ``kernels/scatter.scatter_add``
# then does the ``atomic_add`` into the caller's float accumulator, pre-summed the same way where
# a warp shares the group.
GROUP_SUM_NONFINITE = wp.int32(2047)


@wp.struct
class GroupSumPass:
    """One launch's view of a group sum: its buffer's layout, the headroom and the pass."""

    # ``n_fields`` exponent fields (1, shared by every group, or one per group), then ``n_slots``
    # high words, then ``n_slots`` low words for a two-word sum; empty with the setting off.
    words: wp.array[wp.int64]
    n_fields: wp.int32
    n_slots: wp.int32
    # A floor under every field: a caller's bound on every term (0 when there is none).
    base_field: wp.int32
    headroom: wp.int32
    phase: wp.int32


@wp.func
def group_sum_field(value: wp.float64) -> wp.int32:
    # The biased exponent field of ``value``: ``|value| < 2^(max(field, 1) - 1022)``, subnormals
    # and zero (field 0) included; ``GROUP_SUM_NONFINITE`` for an infinity or NaN.
    return wp.int32((wp.cast(value, wp.int64) >> wp.int64(52)) & wp.int64(2047))


@wp.func
def group_sum_power(exponent: wp.int32) -> wp.float64:
    # ``2^exponent`` from its bits, for ``-1022 <= exponent <= 1023``.
    return wp.cast(wp.int64(exponent + 1023) << wp.int64(52), wp.float64)


@wp.func
def group_sum_field_index(sums: GroupSumPass, group: wp.int32) -> wp.int32:
    # Where ``group``'s field lives: its own word, or the one shared by every group.
    return wp.where(sums.n_fields == 1, 0, group)


@wp.func
def group_sum_group_field(sums: GroupSumPass, group: wp.int32) -> wp.int32:
    # ``group``'s field: the larger of its stored field and the caller's floor.
    return wp.max(sums.base_field, wp.int32(sums.words[group_sum_field_index(sums, group)]))


@wp.func
def raise_group_sum_field(sums: GroupSumPass, group: wp.int32, field: wp.int32) -> None:
    # Pass 0: raise ``group``'s field to ``field`` (``group_sum_field`` of a term, or the largest
    # over a vector's components). The read first skips the atomic for most terms of a large
    # group, whose field settles early; a stale read only costs an atomic.
    s = group_sum_field_index(sums, group)
    if wp.int64(field) > sums.words[s]:
        wp.atomic_max(sums.words, s, wp.int64(field))


@wp.func
def round_shift_right(mantissa: wp.uint64, shift: wp.int32) -> wp.uint64:
    # ``rint(mantissa / 2^shift)``, ties to even, for ``shift >= 1`` and ``mantissa < 2^53``; a
    # shift past 63 is clamped (the quotient is 0 and the remainder below half either way).
    s = wp.min(shift, 63)
    quotient = mantissa >> wp.uint64(s)
    remainder = mantissa & ((wp.uint64(1) << wp.uint64(s)) - wp.uint64(1))
    half = wp.uint64(1) << wp.uint64(s - 1)
    if remainder > half or (remainder == half and (quotient & wp.uint64(1)) != wp.uint64(0)):
        quotient += wp.uint64(1)
    return quotient


@wp.func
def group_sum_words(
    value: wp.float64, field: wp.int32, headroom: wp.int32, two_words: wp.int32
) -> wp.vec2l:
    # ``value``'s fixed-point words at the scale ``field`` sets (the note above), in integer
    # arithmetic on its mantissa: ``(rint(x), 0)`` for one word, the split of
    # ``rint(x * 2^(62 - h))`` at ``2^(62 - h)`` for two, both words carrying ``value``'s sign.
    # Requires ``group_sum_field(value) <= field < GROUP_SUM_NONFINITE``.
    bits = wp.cast(value, wp.int64)
    exponent = wp.int32((bits >> wp.int64(52)) & wp.int64(2047))
    mantissa = wp.uint64(bits) & wp.uint64(0xFFFFFFFFFFFFF)
    if exponent != 0:
        mantissa = mantissa | wp.uint64(0x10000000000000)
    width = wp.where(two_words != 0, 62 - headroom, 0)
    # ``rint(x * 2^width) = rint(mantissa * 2^shift)``.
    shift = wp.max(exponent, 1) - wp.max(field, 1) + 9 - headroom + width
    high = wp.uint64(0)
    low = wp.uint64(0)
    mask = (wp.uint64(1) << wp.uint64(width)) - wp.uint64(1)
    if shift >= width:
        high = mantissa << wp.uint64(shift - width)
    elif shift >= 0:
        high = mantissa >> wp.uint64(width - shift)
        low = (mantissa << wp.uint64(shift)) & mask
    else:
        rounded = round_shift_right(mantissa, -shift)
        high = rounded >> wp.uint64(width)
        low = rounded & mask
    signed_high = wp.int64(high)
    signed_low = wp.int64(low)
    if bits < wp.int64(0):
        signed_high = -signed_high
        signed_low = -signed_low
    return wp.vec2l(signed_high, signed_low)


# Warp-uniform pre-sums: where every active lane of a warp adds to one slot, the lanes' terms are
# summed across the warp and the lowest lane commits once. ``warp_uniform_lanes`` decides (one
# shuffle and a vote): the active mask when the lanes share ``key`` and are more than one, else 0;
# the CPU device runs one lane at a time and always reads 0, so its commits are the plain ones.
# ``warp_lanes_sum`` then sums a value over those lanes into the lowest (``warp_lane_leads``);
# the other lanes hold partial sums. A full warp folds by halves; a partial one by rank over the
# set lanes (``__fns``), since a lane outside the mask cannot be read.
_WARP_UNIFORM_LANES = """
#if defined(__CUDA_ARCH__)
const unsigned active = __activemask();
if (__popc(active) > 1 && __all_sync(active, __shfl_sync(active, key, __ffs(active) - 1) == key)) {
    return (int)active;
}
#endif
return 0;
"""

# ``warp_uniform_lanes`` for a warp whose lanes add to several slots at once: the active lanes
# sharing this lane's ``key`` (``__match_any_sync``), when they are more than one, else 0. Each
# group then sums over its own mask with ``warp_lanes_sum`` and its lowest lane commits, as a
# labelled partition does, so a slot that half a warp hits (a fan's hub in every other face) takes
# one atomic a warp, not sixteen. 0 on the CPU device and before ``sm_70``.
_WARP_PEER_LANES = """
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
const unsigned peers = __match_any_sync(__activemask(), key);
if (__popc(peers) > 1) {
    return (int)peers;
}
#endif
return 0;
"""

_WARP_LANES_SUM = """
#if defined(__CUDA_ARCH__)
const unsigned active = (unsigned)lanes;
auto total = value;
if (active == 0xffffffffu) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(active, total, offset);
    }
    return total;
}
const unsigned lane = threadIdx.x & 31u;
const unsigned rank = __popc(active & ((1u << lane) - 1u));
const unsigned count = __popc(active);
for (unsigned offset = 1u; offset < count; offset <<= 1) {
    const bool has = rank + offset < count;
    const int source = has ? (int)__fns(active, 0u, (int)(rank + offset + 1u)) : (int)lane;
    const auto other = __shfl_sync(active, total, source);
    if (has) {
        total += other;
    }
}
return total;
#else
return value;
#endif
"""

_WARP_LANE_LEADS = """
#if defined(__CUDA_ARCH__)
return (int)((threadIdx.x & 31u) == (unsigned)(__ffs((unsigned)lanes) - 1));
#else
return 1;
#endif
"""


# The smallest ``value`` among the warp's active lanes, returned to its lowest active lane only;
# every other lane gets ``INT_MAX``, so one ``atomic_min`` a warp commits it. Each thread is its own
# warp on the CPU device.
_WARP_MIN_AT_LEAD = """
#if defined(__CUDA_ARCH__)
const unsigned active = __activemask();
const int smallest = __reduce_min_sync(active, value);
return (threadIdx.x & 31u) == (unsigned)(__ffs(active) - 1) ? smallest : 2147483647;
#else
return value;
#endif
"""


@wp.func_native(_WARP_UNIFORM_LANES)
def warp_uniform_lanes(key: wp.int32) -> wp.int32: ...


@wp.func_native(_WARP_PEER_LANES)
def warp_peer_lanes(key: wp.int32) -> wp.int32: ...


@wp.func_native(_WARP_MIN_AT_LEAD)
def warp_min_at_lead(value: wp.int32) -> wp.int32: ...


@wp.func_native(_WARP_LANES_SUM)
def warp_lanes_sum(lanes: wp.int32, value: wp.int64) -> wp.int64: ...


@wp.func_native(_WARP_LANES_SUM)
def warp_lanes_sum_float(lanes: wp.int32, value: wp.float64) -> wp.float64: ...


@wp.func_native(_WARP_LANES_SUM)
def warp_lanes_sum_float32(lanes: wp.int32, value: wp.float32) -> wp.float32: ...


@wp.func_native(_WARP_LANE_LEADS)
def warp_lane_leads(lanes: wp.int32) -> wp.int32: ...


@wp.func
def peer_counted_add(out_counts: wp.array[wp.int32], index: wp.int32) -> None:
    # ``wp.atomic_add(out_counts, index, 1)`` with the lanes of a warp counting into one bin
    # (``warp_peer_lanes``) committed once, by the lowest, as their number; a lane with no peer, and
    # every lane on the CPU device, takes the plain atomic. A count is exact in any order. For a
    # histogram whose keys crowd a few bins: the radix select's 2 048-bin digits over a 1 M pool
    # cost the Poisson-disk round more than the sort they replaced unpooled; component sizes
    # (``repair.flatten_and_count_labels`` 0.06x on dragon, ``points.component_sizes`` 0.05x,
    # ``cholesky.component_sizes`` 0.12x on bunny) and cluster counts (0.5x on dragon) the same.
    lanes = warp_peer_lanes(index)
    if lanes == 0:
        wp.atomic_add(out_counts, index, 1)
    elif warp_lane_leads(lanes) != 0:
        wp.atomic_add(out_counts, index, popcount32(wp.uint32(lanes)))


# A warp sharing one long walk (a CSR row ``array.csr_row_long`` names) among its active lanes,
# in a thread-a-row launch with no block barrier: ``warp_active_lanes`` is the converged lanes'
# mask, ``warp_ballot`` the lanes of a mask whose ``flag`` is set, ``warp_lane_rank`` a
# lane's rank in a mask and ``array.popcount32`` its size (the walk's start and stride),
# ``warp_lowest_lane`` the lowest lane in a mask and ``warp_broadcast`` a value from one lane to
# the mask. Each thread is its own warp on the CPU device, where the active mask and every ballot
# read 0, so a caller's ordinary path takes every row there.
_WARP_ACTIVE_LANES = """
#if defined(__CUDA_ARCH__)
return (int)__activemask();
#else
return 0;
#endif
"""

_WARP_BALLOT = """
#if defined(__CUDA_ARCH__)
return (int)__ballot_sync((unsigned)lanes, flag != 0);
#else
return 0;
#endif
"""

_WARP_LANE_RANK = """
#if defined(__CUDA_ARCH__)
return __popc((unsigned)lanes & ((1u << (threadIdx.x & 31u)) - 1u));
#else
return 0;
#endif
"""

_WARP_LOWEST_LANE = """
#if defined(__CUDA_ARCH__)
return __ffs((unsigned)lanes) - 1;
#else
return 0;
#endif
"""

_WARP_BROADCAST = """
#if defined(__CUDA_ARCH__)
return __shfl_sync((unsigned)lanes, value, source);
#else
return value;
#endif
"""


@wp.func_native(_WARP_ACTIVE_LANES)
def warp_active_lanes() -> wp.int32: ...


@wp.func_native(_WARP_BALLOT)
def warp_ballot(lanes: wp.int32, flag: wp.int32) -> wp.int32: ...


@wp.func_native(_WARP_LANE_RANK)
def warp_lane_rank(lanes: wp.int32) -> wp.int32: ...


@wp.func_native(_WARP_LOWEST_LANE)
def warp_lowest_lane(lanes: wp.int32) -> wp.int32: ...


@wp.func_native(_WARP_BROADCAST)
def warp_broadcast(lanes: wp.int32, value: wp.int32, source: wp.int32) -> wp.int32: ...


@wp.func_native(_WARP_BROADCAST)
def warp_broadcast_float(lanes: wp.int32, value: wp.float64, source: wp.int32) -> wp.float64: ...


# The inclusive sum of ``value`` over a whole warp's lanes up to each lane (a shuffle scan), the
# value itself on the CPU device. Every lane of the warp must call it.
_WARP_INCLUSIVE_SUM = """
#if defined(__CUDA_ARCH__)
const unsigned lane = threadIdx.x & 31u;
int sum = value;
for (unsigned offset = 1; offset < 32u; offset <<= 1) {
    const int below = __shfl_up_sync(0xffffffffu, sum, offset);
    if (lane >= offset)
        sum += below;
}
return sum;
#else
return value;
#endif
"""


@wp.func_native(_WARP_INCLUSIVE_SUM)
def warp_inclusive_sum(value: wp.int32) -> wp.int32: ...


# A whole warp's compaction into one register a lane: the lanes of ``passing`` (a full-warp
# ``warp_ballot``) hand their ``value``, in lane order, to the lanes from ``filled`` on, and every
# other lane keeps ``held``. A list of up to 32 items lives an item a lane, appended a ballot at
# a time, the caller keeping ``filled + popcount32(passing) <= 32``; a second register takes the
# items past the first's 32 at ``filled - 32``. Every lane of the warp must call it. The CPU
# device's one-lane block keeps ``value`` when it passes.
_WARP_DEPOSIT = """
#if defined(__CUDA_ARCH__)
const int k = (int)(threadIdx.x & 31u) - filled;
const bool takes = k >= 0 && k < __popc((unsigned)passing);
const int source = takes ? (int)__fns((unsigned)passing, 0u, k + 1) : 0;
const int moved = __shfl_sync(0xffffffffu, value, source);
return takes ? moved : held;
#else
return passing != 0 ? value : held;
#endif
"""


@wp.func_native(_WARP_DEPOSIT)
def warp_deposit(
    passing: wp.int32, filled: wp.int32, value: wp.int32, held: wp.int32
) -> wp.int32: ...


@wp.func
def warp_walk(lanes: wp.int32) -> tuple[wp.int32, wp.int32]:
    # A walk's first entry and stride: a lane's rank among the lanes of ``lanes`` and their count
    # (``warp_lane_rank``, ``array.popcount32``), every one of which must walk, or ``(0, 1)`` for
    # one thread walking alone (``lanes == 0``, as on the CPU device).
    if lanes == 0:
        return 0, 1
    return warp_lane_rank(lanes), popcount32(wp.uint32(lanes))


@wp.func
def warp_walk_total(lanes: wp.int32, value: wp.float64) -> wp.float64:
    # The sum of ``value`` over the lanes of a ``warp_walk`` (``warp_lanes_sum_float``'s fixed
    # tree), to every one of them: a row a warp walks that is then rescaled or closed by its sum.
    # ``value`` itself for one thread (``lanes == 0``).
    if lanes == 0:
        return value
    return warp_broadcast_float(lanes, warp_lanes_sum_float(lanes, value), warp_lowest_lane(lanes))


@wp.func
def warp_long_rows(offsets: wp.array[wp.int32], i: wp.int32, lanes: wp.int32) -> wp.int32:
    # The lanes of ``lanes`` whose CSR row ``i`` is long (``array.csr_row_long``): 0 on the CPU
    # device and wherever none is. A thread-a-row kernel writes its short rows as before and walks
    # each of these with the warp in turn, lowest first (``warp_lowest_row``), clearing it with
    # ``long_rows & (long_rows - 1)``.
    return warp_ballot(lanes, wp.where(csr_row_long(offsets, i), 1, 0))


@wp.func
def warp_lowest_row(lanes: wp.int32, long_rows: wp.int32, i: wp.int32) -> wp.int32:
    # The row of the lowest lane of ``long_rows``, to every lane of ``lanes``.
    return warp_broadcast(lanes, i, warp_lowest_lane(long_rows))


# The largest non-negative ``value`` over the lanes in the ``lanes`` mask (from
# ``warp_uniform_lanes``), to every one of them: a non-negative float orders like its bits, so the
# integer warp reduction is exact. The CPU device returns ``value``.
_WARP_LANES_MAX_NONNEGATIVE = """
#if defined(__CUDA_ARCH__)
return __int_as_float(__reduce_max_sync((unsigned)lanes, __float_as_int(value)));
#else
return value;
#endif
"""


@wp.func_native(_WARP_LANES_MAX_NONNEGATIVE)
def warp_lanes_max_nonnegative(lanes: wp.int32, value: wp.float32) -> wp.float32: ...


# The largest (``largest``) or smallest ``value`` over the lanes in the ``lanes`` mask, returned to
# its lowest lane (the one ``warp_lane_leads`` names), for a float of either sign: flipping a
# negative float's magnitude bits makes its bits order like the float as a signed integer, so the
# reduction is exact, and the flip is its own inverse. ``-0.0`` orders below ``+0.0``, as the float
# atomics do, and a NaN lane counts as the reduction's identity (``-inf`` for a maximum), since the
# atomics drop a NaN too; lanes that are all NaN return that infinity, which no atomic applies. A
# ``float32`` is exact in ``float64``, and with ``single`` it reduces as 32-bit keys in one
# ``__reduce_*_sync``, where the ``float64`` keys take a shuffle tree (voxel max pooling 1.05-1.11x
# through the tree). The CPU device returns ``value``.
_WARP_LANES_EXTREMUM = """
#if defined(__CUDA_ARCH__)
if (single) {
    float narrow = (float)value;
    if (isnan(narrow)) {
        narrow = largest ? -INFINITY : INFINITY;
    }
    const int narrow_bits = __float_as_int(narrow);
    int narrow_key = narrow_bits ^ ((narrow_bits >> 31) & 0x7fffffff);
    narrow_key = largest ? __reduce_max_sync((unsigned)lanes, narrow_key)
                         : __reduce_min_sync((unsigned)lanes, narrow_key);
    return (double)__int_as_float(narrow_key ^ ((narrow_key >> 31) & 0x7fffffff));
}
const double finite = isnan(value) ? (largest ? -INFINITY : INFINITY) : value;
const long long bits = __double_as_longlong(finite);
long long key = bits ^ ((bits >> 63) & 0x7fffffffffffffffLL);
const unsigned active = (unsigned)lanes;
const unsigned lane = threadIdx.x & 31u;
const unsigned rank = __popc(active & ((1u << lane) - 1u));
const unsigned count = __popc(active);
for (unsigned offset = 1u; offset < count; offset <<= 1) {
    const bool has = rank + offset < count;
    const int source = has ? (int)__fns(active, 0u, (int)(rank + offset + 1u)) : (int)lane;
    const long long other = __shfl_sync(active, key, source);
    if (has) {
        key = largest ? (other > key ? other : key) : (other < key ? other : key);
    }
}
return __longlong_as_double(key ^ ((key >> 63) & 0x7fffffffffffffffLL));
#else
return value;
#endif
"""


@wp.func_native(_WARP_LANES_EXTREMUM)
def warp_lanes_extremum(
    lanes: wp.int32, value: wp.float64, largest: wp.bool, single: wp.bool
) -> wp.float64: ...


# The ``value`` of the lane below, for a whole converged warp: a wavefront whose lane ``t`` reads
# what lane ``t - 1`` wrote a step earlier passes it in a register instead of through memory and a
# barrier. Lane 0 gets its own ``value``. One native a dtype (a native cannot return its argument's
# type). The CPU device returns ``value``: its blocks run one lane.
_WARP_SHUFFLE_UP = """
#if defined(__CUDA_ARCH__)
return __shfl_up_sync(0xffffffffu, value, 1);
#else
return value;
#endif
"""


@wp.func_native(_WARP_SHUFFLE_UP)
def warp_shuffle_up_float(value: wp.float32) -> wp.float32: ...


@wp.func_native(_WARP_SHUFFLE_UP)
def warp_shuffle_up_int(value: wp.int32) -> wp.int32: ...


# An element another block of the same launch wrote, read from L2 (``ld.cg``) rather than through
# this SM's L1, which may hold the line from before the write. Pair it with the writer's
# ``thread_fence`` before whatever it publishes, and read only after seeing that. The CPU device
# runs its blocks in turn and reads plainly.
_LOAD_COHERENT_2D = """
#if defined(__CUDA_ARCH__)
return __ldcg(wp::data_at_byte_offset(values, wp::byte_offset(values, i, j)));
#else
return *wp::data_at_byte_offset(values, wp::byte_offset(values, i, j));
#endif
"""

# A flag another block raises, read anew every call through a ``volatile`` pointer: a spin on it
# must not be hoisted out of its loop. ``__ldcv`` is not ``volatile`` to the compiler, which
# assumed a loop over it alone terminates and dropped it (2026-11-09). Fence once the flag is seen,
# before reading what it published.
_LOAD_VOLATILE = """
return *(volatile int*)wp::data_at_byte_offset(values, wp::byte_offset(values, index));
"""


@wp.func_native(_LOAD_COHERENT_2D)
def load_coherent_float_2d(
    values: wp.array2d[wp.float32], i: wp.int32, j: wp.int32
) -> wp.float32: ...


@wp.func_native(_LOAD_COHERENT_2D)
def load_coherent_int_2d(values: wp.array2d[wp.int32], i: wp.int32, j: wp.int32) -> wp.int32: ...


@wp.func_native(_LOAD_VOLATILE)
def load_volatile_int(values: wp.array[wp.int32], index: wp.int32) -> wp.int32: ...


def _load_coherent_1d(value_type: str, length: int) -> str:
    # ``load_coherent_*_2d``'s read of one element of a rank-1 array: an ``int`` / ``float`` scalar
    # (``length`` 0) or a vector of ``length`` components of ``value_type``, each read from L2.
    cuda = "return __ldcg(p);"
    if length > 0:
        cuda = (
            f"wp::vec_t<{length}, {value_type}> result;\n"
            f"for (int c = 0; c < {length}; ++c) result.c[c] = __ldcg(&p->c[c]);\n"
            "return result;"
        )
    return f"""
const auto* p = wp::data_at_byte_offset(values, wp::byte_offset(values, index));
#if defined(__CUDA_ARCH__)
{cuda}
#else
return *p;
#endif
"""


@wp.func_native(_load_coherent_1d("int", 0))
def load_coherent_int(values: wp.array[wp.int32], index: wp.int32) -> wp.int32: ...


@wp.func_native(_load_coherent_1d("float", 3))
def load_coherent_vec3(values: wp.array[wp.vec3], index: wp.int32) -> wp.vec3: ...


@wp.func_native(_load_coherent_1d("float", 4))
def load_coherent_vec4(values: wp.array[wp.vec4], index: wp.int32) -> wp.vec4: ...


@wp.func_native(_load_coherent_1d("double", 3))
def load_coherent_vec3d(values: wp.array[wp.vec3d], index: wp.int32) -> wp.vec3d: ...


@wp.func
def commit_group_sum_word(sums: GroupSumPass, index: wp.int32, word: wp.int64):
    # One word of a term into ``sums.words[index]`` (a slot's high or low word): summed across the
    # warp first where the warp shares the word (keyed on ``index`` so lanes converged on a high
    # and a low word of one slot never pool), committed only where non-zero.
    total = word
    lanes = warp_uniform_lanes(index)
    if lanes != 0:
        total = warp_lanes_sum(lanes, word)
        if warp_lane_leads(lanes) == 0:
            total = wp.int64(0)
    if total != wp.int64(0):
        wp.atomic_add(sums.words, index, total)


@wp.func
def group_sum_two_words(sums: GroupSumPass) -> wp.int32:
    # 1 when the buffer holds low words.
    return wp.where(sums.words.shape[0] > sums.n_fields + sums.n_slots, 1, 0)


@wp.func
def add_group_sum_term(
    sums: GroupSumPass, group: wp.int32, slot: wp.int32, field: wp.int32, value: wp.float64
) -> None:
    # Pass 1: ``value``'s words added into slot ``slot``'s (zeroed) words at the scale of its
    # group's ``field`` (``group_sum_group_field``, read once for all of a vector's components). A
    # term above the field -- a caller's bound broken -- makes the group read NaN.
    if field == GROUP_SUM_NONFINITE:
        return
    if group_sum_field(value) > field:
        sums.words[group_sum_field_index(sums, group)] = wp.int64(GROUP_SUM_NONFINITE)
        return
    two_words = group_sum_two_words(sums)
    words = group_sum_words(value, field, sums.headroom, two_words)
    commit_group_sum_word(sums, sums.n_fields + slot, words[0])
    if two_words != 0:
        commit_group_sum_word(sums, sums.n_fields + sums.n_slots + slot, words[1])


@wp.func
def group_sum_term(sums: GroupSumPass, slot: wp.int32, value: wp.float64) -> None:
    # A scalar term's part in the pass ``sums.phase`` names: its bound (0) or its words (1).
    if sums.phase == 0:
        raise_group_sum_field(sums, slot, group_sum_field(value))
    else:
        add_group_sum_term(sums, slot, slot, group_sum_group_field(sums, slot), value)


@wp.func
def group_sum_from_words(
    high: wp.int64, low: wp.int64, field: wp.int32, headroom: wp.int32, two_words: wp.int32
) -> wp.float64:
    # The ``float64`` sum a slot's finished words hold at the scale ``field`` and ``headroom`` set,
    # from the words as values (a buffer's, or a block's sum of them in registers): the carry
    # normalized (``low`` to ``[0, 2^(62 - h))``, flooring for a negative total), the words read as
    # one ``float64``, and the scale undone in two power-of-two factors (``k`` spans
    # ``[-993, 1083]``). ``low`` is ignored for a one-word sum. Requires
    # ``field < GROUP_SUM_NONFINITE``.
    total = wp.float64(0.0)
    if two_words != 0:
        width = 62 - headroom
        carried = high + (low >> wp.int64(width))
        rest = low & ((wp.int64(1) << wp.int64(width)) - wp.int64(1))
        total = wp.float64(carried) + wp.float64(rest) * group_sum_power(-width)
    else:
        total = wp.float64(high)
    undo = wp.max(field, 1) + headroom - 1084
    first = undo >> 1
    return total * group_sum_power(first) * group_sum_power(undo - first)


@wp.func
def group_sum_value(sums: GroupSumPass, group: wp.int32, slot: wp.int32) -> wp.float64:
    # Slot ``slot``'s sum (of group ``group``) once pass 1 has added every term
    # (``group_sum_from_words`` of its words); NaN for a non-finite group. A consumer kernel may
    # call this itself instead of launching the finish.
    field = group_sum_group_field(sums, group)
    if field == GROUP_SUM_NONFINITE:
        return wp.float64(wp.nan)
    two_words = group_sum_two_words(sums)
    low = wp.int64(0)
    if two_words != 0:
        low = sums.words[sums.n_fields + sums.n_slots + slot]
    return group_sum_from_words(
        sums.words[sums.n_fields + slot], low, field, sums.headroom, two_words
    )


@wp.func
def read_group_sum(out_sums: wp.array[Any], sums: GroupSumPass, group: wp.int32):
    # Group ``group``'s scalar sum in ``out_sums``' dtype, whatever the deterministic setting: the
    # accumulator ``out_sums[group]`` the arrival-order scatter added into when ``sums`` holds no
    # words, else ``group_sum_value`` rounded as the finish rounds it (bit for bit the finished
    # value). The consumer-fused finish: a kernel that reads each group's sum takes ``out_sums``
    # and ``sums`` from ``_launch.group_sums(fused=True)`` and calls this, so the setting on skips
    # the finish launch, its pass over the words and the result buffer (``out_sums`` is then
    # ``None`` and never read). The branch is uniform over a launch.
    if sums.words.shape[0] == 0:
        return out_sums[group]
    return out_sums.dtype(group_sum_value(sums, group, group))


@wp.func
def read_group_sum_vector(out_sums: wp.array[Any], sums: GroupSumPass, group: wp.int32):
    # ``read_group_sum`` for a vector group sum: group ``group``'s components are its slots
    # ``group * len + c``, each rounded as the finish rounds it, so the vector is bit for bit the
    # finished one.
    if sums.words.shape[0] == 0:
        return out_sums[group]
    value = out_sums.dtype()
    for c in range(len(value)):
        value[c] = type(value[0])(group_sum_value(sums, group, group * len(value) + c))
    return value


@wp.func
def take_group_sum(
    sums: GroupSumPass, group: wp.int32, slot: wp.int32, clear: wp.int32
) -> wp.float64:
    # ``group_sum_value``, leaving the slot's words zeroed when ``clear`` is set, so the same
    # buffer serves a loop's next round with no clearing launch (its fields stay: a valid bound
    # while no term outgrows them).
    value = group_sum_value(sums, group, slot)
    if clear != 0:
        sums.words[sums.n_fields + slot] = wp.int64(0)
        if group_sum_two_words(sums) != 0:
            sums.words[sums.n_fields + sums.n_slots + slot] = wp.int64(0)
    return value


def _finish_group_sums(name, dtype, scalar, shape):
    """
    Build the finish kernel for one result dtype: each group's sums written as ``dtype``.

    ``shape`` is ``()`` for a scalar, ``(n,)`` for a vector and ``(rows, cols)`` for a matrix; a
    group's components are its slots ``group * size + c`` in row-major order, ``scalar`` the
    component dtype. ``clear`` zeroes the words read (``take_group_sum``).
    """
    size = 1
    for extent in shape:
        size *= extent
    rank = len(shape)
    rows = shape[0] if rank == 2 else 1
    cols = shape[1] if rank == 2 else 1

    def _k(sums: GroupSumPass, clear: wp.int32, out_sums: Any) -> None:
        g = wp.int32(wp.tid())
        if wp.static(rank == 0):
            out_sums[g] = scalar(take_group_sum(sums, g, g, clear))
        elif wp.static(rank == 1):
            v = dtype()
            for c in range(wp.static(size)):
                v[c] = scalar(take_group_sum(sums, g, g * size + c, clear))
            out_sums[g] = v
        else:
            m = dtype()
            for r in range(wp.static(rows)):
                for c in range(wp.static(cols)):
                    m[r, c] = scalar(take_group_sum(sums, g, g * size + r * cols + c, clear))
            out_sums[g] = m

    _k.__annotations__["out_sums"] = wp.array[dtype]
    return wp.kernel(_k, name=name)


# The dtypes a group sum can produce, with their component dtype and shape; a ``float64``
# component takes two words.
GROUP_SUM_DTYPES = {
    wp.float32: (wp.float32, ()),
    wp.float64: (wp.float64, ()),
    wp.vec2: (wp.float32, (2,)),
    wp.vec3: (wp.float32, (3,)),
    wp.vec4: (wp.float32, (4,)),
    wp.vec2d: (wp.float64, (2,)),
    wp.vec3d: (wp.float64, (3,)),
    wp.vec4d: (wp.float64, (4,)),
    wp.mat33: (wp.float32, (3, 3)),
    wp.mat33d: (wp.float64, (3, 3)),
}

# Each dtype's word count and slots per group, for ``_launch.group_sums``.
GROUP_SUM_LAYOUTS = {
    d: (scalar is wp.float64, math.prod(shape)) for d, (scalar, shape) in GROUP_SUM_DTYPES.items()
}

FINISH_GROUP_SUMS = KernelTable(
    "finish_group_sums",
    {
        d: _finish_group_sums(f"finish_group_sums_{d.__name__}", d, scalar, shape)
        for d, (scalar, shape) in GROUP_SUM_DTYPES.items()
    },
)


def blocks_1d(n: int) -> int:
    """
    Launch width for the 1-D global reduction kernels in this module.

    Every ``*1d_tiled`` kernel here folds ``TILES_PER_BLOCK_1D`` tiles per block, so its grid is
    ``n / (TILE_1D * TILES_PER_BLOCK_1D)`` and **not** ``n / TILE_1D``. Call this rather than
    open-coding the division: passing the tile count would give each block the same 16 tiles the
    next 15 blocks also claim, folding every element 16 times over -- and, the fold being
    idempotent for the extrema, that would leave ``min`` / ``max`` / ``any`` / ``all`` looking
    correct while ``sum`` silently returned 16x its answer.

    It is also the launch width for the *lane-strided* reductions outside this module --
    ``registration.fit_transform_and_accumulate_cost`` / ``accumulate_procrustes_moments`` /
    ``accumulate_point_to_plane``, ``points.centered_covariance``,
    ``polyline.accumulate_turning_angle`` /
    ``accumulate_radius_frame`` / ``accumulate_loop_frame`` -- which own the same
    [`ITEMS_PER_BLOCK_1D`][ordito.kernels.reduce.ITEMS_PER_BLOCK_1D] chunk per block but partition
    it across lanes with ``wp.block_dim()`` rather than loading tiles from it.

    Parameters
    ----------
    n
        Number of elements in the 1-D array being reduced.

    Returns
    -------
    int
        Block count to pass as ``dim`` to ``wp.launch_tiled`` with ``block_dim=TILE_1D``.
    """
    items = ITEMS_PER_BLOCK_1D
    return (n + items - 1) // items


# The block count a spanned fold (``spanned_blocks_1d``) narrows toward on CUDA: enough to fill
# the device. Past it each extra block costs the last block's fold of the partials more than its
# lanes' shorter walks save (2026-11-08, ``heat.upper_edge_length_sum_and_count``: a row a lane
# 0.20x on ``sphere_small`` to ``bunny``, but 1.31x on ``lucy``'s 219 k blocks).
SPANNED_TARGET_BLOCKS_1D = 2048


def spanned_blocks_1d(n: int, device: wp.DeviceLike) -> tuple[int, int]:
    """
    ``(items a block, blocks)`` for a lane-strided fold whose items are each a dependent walk.

    A lane folding [`ITEMS_PER_BLOCK_1D`][ordito.kernels.reduce.ITEMS_PER_BLOCK_1D] items walks
    ``TILES_PER_BLOCK_1D`` of them in turn, which costs nothing for a load a term and is the whole
    launch for a term that chases indices (a CSR row's entries, then their vertices). This is the
    fewest whole powers of two of ``TILE_1D`` items a block, from one item a lane up to the
    standard fold width, that bring the block count to at most ``SPANNED_TARGET_BLOCKS_1D``. The
    kernel takes the width as its block chunk (``block_chunk(n, block, width)``) and launches
    ``blocks`` blocks of ``TILE_1D`` lanes; its partials hold ``blocks`` rows. The CPU runs a
    block as one lane, which walks every width alike, so on the CPU this is the standard fold,
    [`blocks_1d`][ordito.kernels.reduce.blocks_1d] blocks of ``ITEMS_PER_BLOCK_1D`` items.

    Parameters
    ----------
    n
        Number of items in the fold.
    device
        The device the fold runs on.

    Returns
    -------
    tuple[int, int]
        The items each block owns and the block count.
    """
    if wp.get_device(device).is_cpu:
        return ITEMS_PER_BLOCK_1D, blocks_1d(n)
    tiles = max((n + TILE_1D - 1) // TILE_1D, 1)
    per_block = 1
    while per_block < TILES_PER_BLOCK_1D and tiles > per_block * SPANNED_TARGET_BLOCKS_1D:
        per_block *= 2
    width = per_block * TILE_1D
    return width, (n + width - 1) // width


def chunks_1d(n: int) -> int:
    """
    Launch width for [`minmax_vec3_chunked`][ordito.kernels.reduce.minmax_vec3_chunked].

    That kernel is the *unfolded* kind: one plain thread per ``TILE_1D`` points, walked serially,
    with no tile load and no ``TILES_PER_BLOCK_1D`` fold. So its grid is ``n / TILE_1D`` and
    [`blocks_1d`][ordito.kernels.reduce.blocks_1d] -- which divides by
    ``TILE_1D * TILES_PER_BLOCK_1D`` -- is **wrong** for it by a factor of
    ``TILES_PER_BLOCK_1D``, dropping fifteen sixteenths of the cloud from the box. The two helpers
    exist so that choosing between them is a visible decision rather than an open-coded division
    that has to be re-derived against the kernel's body.

    Parameters
    ----------
    n
        Number of elements in the 1-D array being reduced.

    Returns
    -------
    int
        Chunk count to pass as ``dim`` to ``wp.launch``.
    """
    width = TILE_1D
    return (n + width - 1) // width


# ---------------------------------------------------------------------------
# Kernel factories.
#
# Each builds one ``@wp.kernel`` capturing a ``(tile_reduce, atomic, scalar)``
# triple. The full-tile branch loads a tile and reduces it with ``tile_reduce``;
# the partial-tile remainder folds elements with ``scalar``; the result is
# committed with ``atomic``. A unique ``name`` per instantiation is REQUIRED:
# Warp keys generated kernels by name, so reusing one name would alias distinct
# reductions to the same compiled kernel.
# ---------------------------------------------------------------------------


@wp.func
def array_term(local: wp.int32, offset: wp.int32, values: Any):
    # Element ``offset + local`` of ``values``: the loaded ``_reduce_1d_tiled`` form's term, so
    # the ragged-tail fold reads the same way in both forms.
    return values[offset + local]


def _reduce_1d_tiled(
    tile_reduce,
    atomic,
    scalar,
    name,
    dtype,
    partials=False,
    term=None,
    term_dtype=wp.float32,
    tiles_per_block=TILES_PER_BLOCK_1D,
    folded=False,
):
    """
    axis=None on a 1-D array: ``TILES_PER_BLOCK_1D`` tiles per block, one atomic per block.

    A sum on CUDA folds each block's chunk lane-strided instead and reduces it once (the
    ``device_is_cpu`` arm): the tiles' barriers held 84 M ``float32`` at 1.05 TB/s against 1.65,
    and ``reduce.sum`` took 0.70x on small arrays, 0.66x on ``lucy``'s corners (2026-11-13). The
    CPU keeps the tiles, which vectorize there, and min / max keep them everywhere (CUDA takes
    ``extrema_1d``'s kernels).

    With ``partials=True`` the block's result is *stored* at ``out_result[block]`` instead of
    committed with ``atomic`` (which is then unused): the deterministic float sum's first stage,
    and, launched over its own output, the fold of those partials. A float ``atomic_add`` commits
    blocks in arrival order, so its total moves in the last bits from run to run on CUDA; stored
    partials folded by the same fixed tree give one answer on every run.

    With ``folded=True`` the block's result goes through ``store_folded_total`` at stride 1: into a
    one-slot ``out_result`` (``_launch.block_partials``' form with the deterministic setting off)
    one ``atomic_add``, into ``blocks + 2`` slots a stored row whose last block adds every row into
    slot 0. One launch either way, where the stored partials take a fold launch (and a buffer) per
    factor of ``ITEMS_PER_BLOCK_1D``: the form ``reduce.sum_of_terms`` launches.

    The accumulator is seeded from the block's *first* chunk rather than from an identity, which
    keeps the kernel generic over ``wp.Scalar`` without the wrapper having to pass a per-dtype
    identity value in. Every subsequent chunk folds in with ``scalar``.

    ``dtype`` widens the template past ``wp.Scalar`` -- ``wp.vec3`` is the one in use, for the vec3
    sum, whose ``wp.add`` fold and ``wp.atomic_add`` commit work component-wise. It is a *codegen*
    parameter, so each instantiation is a concrete kernel: annotating ``wp.array[Any]`` and letting
    one kernel serve both dtypes also works and is declined, because a generic kernel pays Warp's
    host-side overload resolution on every launch.

    **The mapped form** (``term`` given; built by
    [`term_sum_1d`][ordito.kernels.reduce.term_sum_1d]). The elements are formed on the fly
    instead of loaded: ``source`` is a ``@wp.struct`` of type ``dtype`` carrying an
    ``n: wp.int32`` item count and the arrays ``term`` reads, element ``offset + local`` is
    ``term(local, offset, source)``, and a full chunk is ``wp.tile_map(term, local, offset,
    source)`` over the lane indices ``local``. A struct, because ``wp.tile_map`` passes one through
    where it does not pass an array; ``local`` beside ``offset``, because ``wp.tile_arange``'s
    bounds must be constants. Only the element count and the chunk's tile differ between the forms
    (the ``wp.static`` branches below); the chunk walk, the seed, the fold order and the commit are
    one body, so a mapped sum is bit for bit the loaded sum over the materialized terms, and the
    terms are never stored. A mapped kernel is a ``float32`` reduction and goes into ``term``'s own
    module, so a caller's term family never changes this module's hash. ``tiles_per_block`` is the
    fold width: ``TILES_PER_BLOCK_1D`` but for a mapped sum built narrower (see ``term_sum_1d``).
    """
    element = array_term if term is None else term
    value_dtype = dtype if term is None else term_dtype
    is_sum = scalar is wp.add
    accumulator_type = value_dtype
    is_last = last_ticket_vector if wp.types.type_is_vector(value_dtype) else last_ticket

    def _k(source: Any, out_result: Any) -> None:
        i, t = wp.tid()
        if wp.static(term is None):
            n = source.shape[0]
        else:
            n = source.n
        base, remaining = tile_chunk(n, i, tiles_per_block * TILE_1D)
        if remaining <= 0:
            return
        if wp.static(term is not None):
            local = wp.tile_arange(TILE_1D, dtype=wp.int32, storage="register")

        if wp.static(is_sum) and device_is_cpu() == 0:
            # A sum on CUDA: each lane folds its stride of the chunk, then one block reduction,
            # where the tiled walk below ends each of its tiles in a barrier (84 M ``float32``:
            # 1.05 against 1.65 TB/s). A fixed order, so the total still repeats bit for bit.
            # The typed constructor makes the constant zero a mutable accumulator (a bare
            # ``wp.static`` value is a constant in the loop).
            lane_sum = accumulator_type(wp.static(value_dtype(0)))
            # ``remaining`` runs to the end of the array; the block owns its chunk's width.
            count = wp.min(remaining, tiles_per_block * TILE_1D)
            for k in range(t, count, wp.block_dim()):
                lane_sum = scalar(lane_sum, element(k, base, source))
            result = block_sum(lane_sum)
        else:
            # First chunk seeds the accumulator (both branches assign it -- Warp's conditional
            # scoping).
            if remaining >= TILE_1D:
                if wp.static(term is None):
                    tile = wp.tile_load(source, shape=TILE_1D, offset=base, storage="register")
                else:
                    tile = wp.tile_map(term, local, base, source)
                result = tile_reduce(tile)[0]
            else:
                result = element(0, base, source)
                for k in range(1, remaining):
                    result = scalar(result, element(k, base, source))

            for s in range(1, tiles_per_block):
                offset = base + s * TILE_1D
                rest = n - offset
                if rest >= TILE_1D:
                    if wp.static(term is None):
                        chunk = wp.tile_load(
                            source, shape=TILE_1D, offset=offset, storage="register"
                        )
                    else:
                        chunk = wp.tile_map(term, local, offset, source)
                    result = scalar(result, tile_reduce(chunk)[0])
                elif rest > 0:
                    for k in range(rest):
                        result = scalar(result, element(k, offset, source))

        if wp.static(folded):
            store_folded_total(i, t, result, out_result, 0, 1, is_last)
        elif t == 0:
            if wp.static(partials):
                out_result[i] = result
            else:
                atomic(out_result, 0, result)

    if term is None:
        _k.__annotations__["source"] = wp.array[dtype]
        _k.__annotations__["out_result"] = wp.array[dtype]
        return wp.kernel(_k, name=name)
    _k.__annotations__["source"] = dtype
    _k.__annotations__["out_result"] = wp.array[term_dtype]
    return wp.kernel(_k, name=name, module=term.module)


def _weighted_sum_1d_tiled(name, dtype, partials=False, folded=False):
    """
    Weighted 1-D sum ``sum_k values[k] * weights[k]``, in ``_reduce_1d_tiled``'s block shape.

    Its own factory rather than an instantiation of the one above because the second array makes
    the tile path a *pair* of loads and an element-wise product, which the single-array template
    cannot express; everything else -- the ``TILES_PER_BLOCK_1D`` fold, the seed-from-first-chunk
    accumulator, the one atomic per block -- is the same skeleton and the same launch contract
    ([`blocks_1d`][ordito.kernels.reduce.blocks_1d], never the tile count).

    ``dtype`` is the *value* dtype (``wp.float32`` or ``wp.vec3``); weights are always
    ``wp.float32``, and a ``wp.vec3`` tile times a ``wp.float32`` tile scales component-wise. The
    accumulator is seeded from the first weighted element for the same reason as above: neither a
    ``wp.float32(0.0)`` nor a ``wp.vec3(0.0)`` literal spells both instantiations.

    ``partials=True`` stores each block's sum at ``out_sum[block]``, and ``folded=True`` commits
    it through ``store_folded_total``, as in ``_reduce_1d_tiled``.
    """
    is_last = last_ticket_vector if wp.types.type_is_vector(dtype) else last_ticket
    accumulator_type = dtype

    def _k(
        values: wp.array[wp.Scalar], weights: wp.array[wp.float32], out_sum: wp.array[wp.Scalar]
    ) -> None:
        i, t = wp.tid()
        n = values.shape[0]
        base, remaining = tile_chunk(n, i, TILES_PER_BLOCK_1D * TILE_1D)
        if remaining <= 0:
            return

        if device_is_cpu() == 0:
            # CUDA: ``_reduce_1d_tiled``'s lane-strided fold and one block reduction.
            lane_sum = accumulator_type(wp.static(dtype(0)))
            for k in range(t, wp.min(remaining, ITEMS_PER_BLOCK_1D), wp.block_dim()):
                lane_sum = lane_sum + values[base + k] * weights[base + k]
            result = block_sum(lane_sum)
        else:
            if remaining >= TILE_1D:
                v_tile = wp.tile_load(values, shape=TILE_1D, offset=base, storage="register")
                w_tile = wp.tile_load(weights, shape=TILE_1D, offset=base, storage="register")
                result = wp.tile_sum(v_tile * w_tile)[0]
            else:
                result = values[base] * weights[base]
                for k in range(1, remaining):
                    result = result + values[base + k] * weights[base + k]

            for s in range(1, TILES_PER_BLOCK_1D):
                offset = base + s * TILE_1D
                rest = n - offset
                if rest >= TILE_1D:
                    v_chunk = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
                    w_chunk = wp.tile_load(
                        weights, shape=TILE_1D, offset=offset, storage="register"
                    )
                    result = result + wp.tile_sum(v_chunk * w_chunk)[0]
                elif rest > 0:
                    for k in range(rest):
                        result = result + values[offset + k] * weights[offset + k]

        if wp.static(folded):
            store_folded_total(i, t, result, out_sum, 0, 1, is_last)
        elif t == 0:
            if wp.static(partials):
                out_sum[i] = result
            else:
                wp.atomic_add(out_sum, 0, result)

    _k.__annotations__["values"] = wp.array[dtype]
    _k.__annotations__["out_sum"] = wp.array[dtype]
    return wp.kernel(_k, name=name)


def _reduce_2d_tiled(tile_reduce, atomic, scalar, name, dtype):
    """
    axis=None on a 2-D array: one 2-D tile per block, atomically fold into slot 0.

    Reached only for a **non-contiguous** rank-2 array: ``reduce._flattened_for_global`` flattens
    every contiguous one onto the 1-D kernel instead, which carries the anti-contention
    ``TILES_PER_BLOCK_1D`` fold this kernel does not — flattening a
    contiguous buffer is a free reshape, so there is no reason to fold this kernel's atomics too. A
    non-contiguous view (a column slice, a transpose) is the rare remaining caller.
    """

    def _k(values: wp.array2d[wp.Scalar], out_result: wp.array[wp.Scalar]) -> None:
        i, j, t = wp.tid()
        n_rows = values.shape[0]
        n_cols = values.shape[1]
        row_offset = i * TILE_2D
        col_offset = j * TILE_2D
        if row_offset >= n_rows or col_offset >= n_cols:
            return

        remaining_rows = n_rows - row_offset
        remaining_cols = n_cols - col_offset
        tile_rows = wp.where(remaining_rows >= TILE_2D, TILE_2D, remaining_rows)
        tile_cols = wp.where(remaining_cols >= TILE_2D, TILE_2D, remaining_cols)
        if remaining_rows >= TILE_2D and remaining_cols >= TILE_2D:
            tile = wp.tile_load(
                values,
                shape=(TILE_2D, TILE_2D),
                offset=(row_offset, col_offset),
                storage="register",
            )
            result = tile_reduce(tile)[0]
            if t == 0:
                atomic(out_result, 0, result)
        elif t == 0:
            # A boundary block short in either dimension has no tile shape to load -- ``tile_rows``
            # / ``tile_cols`` are runtime values and a tile's shape must be a compile-time constant
            # -- so this folds serially. Only lane 0 runs it (every other lane's ``result`` would be
            # thrown away at the atomic below anyway), rather than every one of the block's
            # ``TILE_2D**2`` lanes redundantly recomputing the identical sum.
            result = values[row_offset, col_offset]
            for r in range(tile_rows):
                for c in range(tile_cols):
                    if r == 0 and c == 0:
                        continue
                    result = scalar(result, values[row_offset + r, col_offset + c])
            atomic(out_result, 0, result)

    _k.__annotations__["values"] = wp.array2d[dtype]
    _k.__annotations__["out_result"] = wp.array[dtype]
    return wp.kernel(_k, name=name)


# Element access for a per-axis reduction, as two ``@wp.func``s a factory *captures* rather than as
# a branch every factory repeats: ``i`` is always the output slot and ``k`` always the position
# along the reduced extent, so each loop below is written once and reads the same for both axes.
# Two things still name the axis directly -- which ``shape`` the extent comes from, and the tile's
# shape -- and both are ``wp.static(rows)``, so nothing is decided at runtime.
#
# The axis is a **codegen** parameter, never a kernel argument: Warp indexes ``array2d`` row-major,
# so the two axes have genuinely different access patterns and a runtime ``axis`` selector would
# compile one of them into a strided walk regardless. Each instantiation is therefore exactly the
# specialised kernel it was hand-written as -- one source of truth, two compiled kernels. Verified
# as such: against the four hand-written rows/cols kernels this replaced, output arrays are equal
# element-for-element and the timings are flat on both axes, tiled and serial.


@wp.func
def _element_along_row(values: wp.array2d[wp.Scalar], i: wp.int32, k: wp.int32):
    # axis=1: slot ``i`` is a row, ``k`` walks its columns.
    return values[i, k]


@wp.func
def _element_along_col(values: wp.array2d[wp.Scalar], i: wp.int32, k: wp.int32):
    # axis=0: slot ``i`` is a column, ``k`` walks its rows.
    return values[k, i]


def _reduce_2d_axis_tiled(tile_reduce, atomic, scalar, name, rows, dtype):
    """
    axis=1 (``rows=True``) or axis=0: one output slot per grid index, tiled along its extent.

    Each block folds one ``TILE_1D`` tile of slot ``i``'s extent and commits it with ``atomic``, so
    the wrapper must pre-fill the output with the reduction's identity. With ``atomic=None`` it
    is the reproducible commit of ``store_block_sum`` per output slot instead, for the float sums,
    whose atomic commit adds the tiles in arrival order and moves the last bits from call to call
    on CUDA: tile ``j`` stores its value in row ``j`` of a zeroed buffer of ``n_out`` results,
    ``n_out`` ticket counters and ``n_tiles`` rows of ``n_out``, and the last tile of slot ``i`` to
    finish adds its column in a fixed order into ``out_result[i]``.
    """
    element = _element_along_row if rows else _element_along_col
    store = atomic is None

    def _k(values: wp.array2d[wp.Scalar], out_result: wp.array[wp.Scalar]) -> None:
        i, j, t = wp.tid()
        if wp.static(rows):
            extent = values.shape[1]
        else:
            extent = values.shape[0]
        offset = j * TILE_1D
        if offset >= extent:
            return
        remaining = extent - offset
        if remaining >= TILE_1D:
            # A tile is a shape, so this cannot go through ``element``: the reduced extent has to
            # be the tile's long side for the load to be contiguous.
            if wp.static(rows):
                tile = wp.tile_load(
                    values, shape=(1, TILE_1D), offset=(i, offset), storage="register"
                )
            else:
                tile = wp.tile_load(
                    values, shape=(TILE_1D, 1), offset=(offset, i), storage="register"
                )
            result = tile_reduce(tile)[0]
        else:
            result = element(values, i, offset)
            for k in range(1, remaining):
                result = scalar(result, element(values, i, offset + k))
        if wp.static(store):
            if wp.static(rows):
                n_out = values.shape[0]
            else:
                n_out = values.shape[1]
            n_tiles = (out_result.shape[0] - 2 * n_out) // n_out
            row = 2 * n_out + j * n_out + i
            if stage_block_value(t, result, out_result, row, n_out + i, n_tiles, last_ticket) != 0:
                zero = type(result)(wp.float32(0.0))
                total = fold_block_values(t, out_result, 2 * n_out + i, n_out, n_tiles, zero)
                write_block_value(t, out_result, i, n_out + i, total)
        elif t == 0:
            atomic(out_result, i, result)

    _k.__annotations__["values"] = wp.array2d[dtype]
    _k.__annotations__["out_result"] = wp.array[dtype]
    return wp.kernel(_k, name=name)


# When the reduced extent is narrower than ``TILE_1D`` the tiled kernels above never take their
# ``tile_load`` branch -- all ``TILE_1D`` lanes of every block redundantly run the serial remainder
# loop, a ``TILE_1D``-fold read amplification and a large measured loss. These serial variants
# launch one plain thread per *output* element and write directly: no tiles, no atomics, and no
# init fill needed on the output buffer. The wrapper picks them whenever ``reduced extent <
# TILE_1D``; past that the tiled kernels stay, since a tall ``(n, 3)`` table reduced along axis=0
# has only 3 outputs and 3 serial threads would be far slower.


def _reduce_2d_axis_serial(scalar, name, rows, dtype):
    """
    axis=1 (``rows=True``) or axis=0 with a reduced extent under ``TILE_1D``: one thread per output.

    Writes ``out_result`` directly, so unlike the tiled form it needs no identity pre-fill. The
    ``rows=False`` instantiation walks a column, whose consecutive threads read consecutive
    addresses -- coalesced, where the ``rows=True`` one is strided by the row length.
    """
    element = _element_along_row if rows else _element_along_col

    def _k(values: wp.array2d[wp.Scalar], out_result: wp.array[wp.Scalar]) -> None:
        i = wp.int32(wp.tid())
        if wp.static(rows):
            extent = values.shape[1]
        else:
            extent = values.shape[0]
        result = element(values, i, 0)
        for k in range(1, extent):
            result = scalar(result, element(values, i, k))
        out_result[i] = result

    _k.__annotations__["values"] = wp.array2d[dtype]
    _k.__annotations__["out_result"] = wp.array[dtype]
    return wp.kernel(_k, name=name)


# A **global** reduction takes whatever scalar dtype the caller's buffer carries, and the package
# itself hands it two families: geometry and solver values (``wp.float32``, ``wp.float64`` in the
# heat and smoothing solvers) and *keys* --
# [`isin`][ordito.array.isin] and [`hash_indices_rows`][ordito.grouping.hash_indices_rows] bound
# an index range with ``reduce.minmax`` over the caller's key dtype, whose public surface is
# ``wp.int32`` / ``wp.int64`` / ``wp.uint32`` / ``wp.uint64`` (``isin`` widens anything narrower
# with ``sortable_dtype`` before reducing, so sub-32-bit dtypes never reach a kernel here).
_GLOBAL_DTYPES = (wp.int32, wp.int64, wp.uint32, wp.uint64, wp.float32, wp.float64)

# A **per-axis** reduction is only ever reached with an index or a geometry dtype: nothing in the
# package reduces a table of 64-bit keys along an axis, and the six-dtype cross product over these
# 24 kernels would be 72 more kernels to compile on every rebuild for no call site.
# A caller who does reduce a ``wp.uint64`` table along an axis pays one fork, once.
_AXIS_DTYPES = (wp.int32, wp.float32, wp.float64)


# ---------------------------------------------------------------------------
# Scalar reductions (min / max / sum) over ``wp.Scalar`` arrays.
# ---------------------------------------------------------------------------

# One concrete kernel per dtype, and the tables are what the wrapper launches. The factories have
# always been able to bake the dtype in -- ``sum_vec3_1d_partials`` and the weighted sums below
# always did -- and leaving the rest at the ``wp.Scalar`` template made every launch pay Warp's
# host-side ``infer_argument_types`` over the whole argument list, which roughly doubles a small
# reduction's host cost and every scalar-returning reduction in the package issues one. Nothing
# extra is compiled: these are the same instantiations ``_register_overloads`` was already creating.
MIN1D_TILED = KernelTable(
    "min1d_tiled",
    {
        d: _reduce_1d_tiled(_tile_min, wp.atomic_min, wp.min, f"min1d_tiled_{d.__name__}", d)
        for d in _GLOBAL_DTYPES
    },
)
MIN2D_TILED = KernelTable(
    "min2d_tiled",
    {
        d: _reduce_2d_tiled(_tile_min, wp.atomic_min, wp.min, f"min2d_tiled_{d.__name__}", d)
        for d in _GLOBAL_DTYPES
    },
)
MIN_2D_ROWS_TILED = KernelTable(
    "min_2d_rows_tiled",
    {
        d: _reduce_2d_axis_tiled(
            _tile_min, wp.atomic_min, wp.min, f"min_2d_rows_tiled_{d.__name__}", True, d
        )
        for d in _AXIS_DTYPES
    },
)
MIN_2D_COLS_TILED = KernelTable(
    "min_2d_cols_tiled",
    {
        d: _reduce_2d_axis_tiled(
            _tile_min, wp.atomic_min, wp.min, f"min_2d_cols_tiled_{d.__name__}", False, d
        )
        for d in _AXIS_DTYPES
    },
)
MIN_2D_ROWS_SERIAL = KernelTable(
    "min_2d_rows_serial",
    {
        d: _reduce_2d_axis_serial(wp.min, f"min_2d_rows_serial_{d.__name__}", True, d)
        for d in _AXIS_DTYPES
    },
)
MIN_2D_COLS_SERIAL = KernelTable(
    "min_2d_cols_serial",
    {
        d: _reduce_2d_axis_serial(wp.min, f"min_2d_cols_serial_{d.__name__}", False, d)
        for d in _AXIS_DTYPES
    },
)

MAX1D_TILED = KernelTable(
    "max1d_tiled",
    {
        d: _reduce_1d_tiled(_tile_max, wp.atomic_max, wp.max, f"max1d_tiled_{d.__name__}", d)
        for d in _GLOBAL_DTYPES
    },
)
MAX2D_TILED = KernelTable(
    "max2d_tiled",
    {
        d: _reduce_2d_tiled(_tile_max, wp.atomic_max, wp.max, f"max2d_tiled_{d.__name__}", d)
        for d in _GLOBAL_DTYPES
    },
)
MAX_2D_ROWS_TILED = KernelTable(
    "max_2d_rows_tiled",
    {
        d: _reduce_2d_axis_tiled(
            _tile_max, wp.atomic_max, wp.max, f"max_2d_rows_tiled_{d.__name__}", True, d
        )
        for d in _AXIS_DTYPES
    },
)
MAX_2D_COLS_TILED = KernelTable(
    "max_2d_cols_tiled",
    {
        d: _reduce_2d_axis_tiled(
            _tile_max, wp.atomic_max, wp.max, f"max_2d_cols_tiled_{d.__name__}", False, d
        )
        for d in _AXIS_DTYPES
    },
)
MAX_2D_ROWS_SERIAL = KernelTable(
    "max_2d_rows_serial",
    {
        d: _reduce_2d_axis_serial(wp.max, f"max_2d_rows_serial_{d.__name__}", True, d)
        for d in _AXIS_DTYPES
    },
)
MAX_2D_COLS_SERIAL = KernelTable(
    "max_2d_cols_serial",
    {
        d: _reduce_2d_axis_serial(wp.max, f"max_2d_cols_serial_{d.__name__}", False, d)
        for d in _AXIS_DTYPES
    },
)

SUM1D_TILED = KernelTable(
    "sum1d_tiled",
    {
        d: _reduce_1d_tiled(_tile_sum, wp.atomic_add, wp.add, f"sum1d_tiled_{d.__name__}", d)
        for d in _GLOBAL_DTYPES
    },
)
# Stored block partials (see ``_reduce_1d_tiled``), launched over its own output as a fold stage
# by a caller whose first stage stores them (``linalg``'s multigrid power step). The library's own
# float sums take ``SUM1D_FOLDED`` below; integer sums are exact in any order and keep the atomic.
SUM1D_PARTIALS = KernelTable(
    "sum1d_partials",
    {
        d: _reduce_1d_tiled(_tile_sum, None, wp.add, f"sum1d_partials_{d.__name__}", d, True)
        for d in (wp.float32, wp.float64)
    },
)


# The one-launch float sums ``reduce.sum`` takes under the deterministic setting and
# ``reduce.sum_of_terms`` always: the ``folded`` commit, whose buffer picks the mode. Against the
# stored partials and their fold launch (2026-11-11, CUDA, new/old): setting off, the vector and
# weighted sums 0.76-0.92x from 5 k to 14 M items (they were fixed-order in both modes); setting
# on, 0.87-0.98x from 100 k to 4 M, but the scalar ``reduce.sum`` 1.17-1.21x at 5 k-100 k, where
# the zeroed buffer and the ticket chain sit on the device path just before the readback (5 us
# against 29). CPU 0.62-0.98x, par past 1 M.
SUM1D_FOLDED = KernelTable(
    "sum1d_folded",
    {
        d: _reduce_1d_tiled(_tile_sum, None, wp.add, f"sum1d_folded_{d.__name__}", d, folded=True)
        for d in (wp.float32, wp.float64)
    },
)


class TermSum(NamedTuple):
    """
    The ``dtype`` block sum ``reduce.sum_of_terms`` launches.

    ``kernel`` commits each block's sum through ``store_folded_total``: into a one-slot buffer by
    ``atomic_add`` (``ordito.config``'s determinism off), into a ``_launch.block_partials`` buffer
    as a stored row the last block folds in a fixed tree. ``partials`` stores each block's sum in
    its own slot instead, for a launch of more blocks than one last block should fold
    (``FOLDED_ROWS_UP_TO``), whose rows a ``FOLD_ROWS`` kernel then adds. Each takes its inputs
    followed by its output. ``dtype`` is ``float32`` but for a ``term_sum_1d`` built with another.
    """

    kernel: wp.Kernel
    partials: wp.Kernel
    dtype: type = wp.float32
    tiles_per_block: int = TILES_PER_BLOCK_1D
    """Tiles a block folds; ``blocks_1d`` assumes ``TILES_PER_BLOCK_1D``."""

    def blocks(self, n: int) -> int:
        """Return the block count over ``n`` terms."""
        return -(-n // (self.tiles_per_block * TILE_1D))


def term_sum_1d(
    term: wp.Function,
    terms_type: type,
    name: str,
    dtype: type = wp.float32,
    tiles_per_block: int = TILES_PER_BLOCK_1D,
) -> TermSum:
    """
    Build the [`TermSum`][ordito.kernels.reduce.TermSum] of ``term(local, offset, terms)``.

    The fused map and reduction: ``reduce.sum``'s ``float32`` kernel in its mapped form (see
    [`_reduce_1d_tiled`][ordito.kernels.reduce._reduce_1d_tiled]), so the total is bit for bit
    ``reduce.sum_of_terms`` of the materialized terms in either determinism setting, with no term
    buffer. ``terms_type`` is the ``@wp.struct`` the kernels take, holding the item count ``n`` and
    the arrays ``term`` reads; ``name`` must be unique, and the two kernels are named ``name`` and
    ``name + "_partials"``. ``dtype`` is what ``term`` returns: ``float32``, or ``wp.vec2`` for
    two sums that share one pass over their items (each component folds as its own ``float32``
    sum). ``tiles_per_block`` narrows the fold for a term whose own
    latency, not the items' count, bounds a lane's walk: the total is then that width's block
    tree, no longer the standard width's bits.
    """
    return TermSum(
        kernel=_reduce_1d_tiled(
            _tile_sum,
            None,
            wp.add,
            name,
            terms_type,
            term=term,
            term_dtype=dtype,
            tiles_per_block=tiles_per_block,
            folded=True,
        ),
        partials=_reduce_1d_tiled(
            _tile_sum,
            None,
            wp.add,
            f"{name}_partials",
            terms_type,
            True,
            term,
            dtype,
            tiles_per_block,
        ),
        dtype=dtype,
        tiles_per_block=tiles_per_block,
    )


SUM2D_TILED = KernelTable(
    "sum2d_tiled",
    {
        d: _reduce_2d_tiled(_tile_sum, wp.atomic_add, wp.add, f"sum2d_tiled_{d.__name__}", d)
        for d in _GLOBAL_DTYPES
    },
)
SUM_2D_ROWS_TILED = KernelTable(
    "sum_2d_rows_tiled",
    {
        d: _reduce_2d_axis_tiled(
            _tile_sum, wp.atomic_add, wp.add, f"sum_2d_rows_tiled_{d.__name__}", True, d
        )
        for d in _AXIS_DTYPES
    },
)
SUM_2D_COLS_TILED = KernelTable(
    "sum_2d_cols_tiled",
    {
        d: _reduce_2d_axis_tiled(
            _tile_sum, wp.atomic_add, wp.add, f"sum_2d_cols_tiled_{d.__name__}", False, d
        )
        for d in _AXIS_DTYPES
    },
)
# The reproducible float axis sums: tile partials, added in a fixed order by each slot's last tile.
SUM_2D_ROWS_PARTIALS = KernelTable(
    "sum_2d_rows_partials",
    {
        d: _reduce_2d_axis_tiled(
            _tile_sum, None, wp.add, f"sum_2d_rows_partials_{d.__name__}", True, d
        )
        for d in (wp.float32, wp.float64)
    },
)
SUM_2D_COLS_PARTIALS = KernelTable(
    "sum_2d_cols_partials",
    {
        d: _reduce_2d_axis_tiled(
            _tile_sum, None, wp.add, f"sum_2d_cols_partials_{d.__name__}", False, d
        )
        for d in (wp.float32, wp.float64)
    },
)
SUM_2D_ROWS_SERIAL = KernelTable(
    "sum_2d_rows_serial",
    {
        d: _reduce_2d_axis_serial(wp.add, f"sum_2d_rows_serial_{d.__name__}", True, d)
        for d in _AXIS_DTYPES
    },
)
SUM_2D_COLS_SERIAL = KernelTable(
    "sum_2d_cols_serial",
    {
        d: _reduce_2d_axis_serial(wp.add, f"sum_2d_cols_serial_{d.__name__}", False, d)
        for d in _AXIS_DTYPES
    },
)

# ---------------------------------------------------------------------------
# Boolean reductions over int32 0/1 masks. ``any`` == OR == max; ``all`` == AND
# == min (a 0/1 mask minimises to 1 iff every element is 1). The 2-D global case
# is handled by the wrapper flattening the mask and reusing the 1-D kernel, so no
# ``*2d_tiled`` bool kernel is needed.
#
# These reach a kernel only through ``reduce._reduce_bool``, which casts the mask to a 0/1
# ``wp.int32`` first -- and ``wp.bool`` is not a ``wp.Scalar`` anyway -- so each is a single
# concrete kernel rather than a dtype table.
# ---------------------------------------------------------------------------


def _reduce_bool_1d_tiled(block_reduce, atomic, scalar, identity, name):
    """
    axis=None over a ``wp.bool`` mask, read as bytes instead of through an ``int32`` copy.

    ``wp.Scalar`` does not instantiate for ``wp.bool``, so the factories
    above cannot serve a mask and the wrapper used to widen one with ``array.astype`` first: an
    allocation of ``4n`` bytes, a launch, a full read of ``n`` and a full write of ``4n``, after
    which the reduction read ``4n`` rather than ``n`` -- nine bytes of traffic per mask byte, plus
    a launch and an allocation, to answer one boolean. This reads the mask directly.

    **The shape differs from ``_reduce_1d_tiled``'s and the difference is forced.** There is no
    ``wp.tile_load`` of a ``bool`` array, so the block's chunk is walked by the lanes rather than
    loaded as tiles: each lane strides by ``wp.block_dim()`` -- never by ``TILE_1D``, which is what
    keeps it correct on the CPU device, where ``wp.launch_tiled`` runs one lane per block and
    ``wp.block_dim()`` reads 1 -- accumulates in a register, and the block folds the
    per-lane values with a single block reduction (``block_reduce``, one of ``block_sum`` /
    ``block_max`` / ``block_min``). That is one tile reduction per block where
    the tile-load form runs ``TILES_PER_BLOCK_1D`` of them, and the strided reads are coalesced
    (consecutive lanes, consecutive bytes).

    ``identity`` seeds a lane that draws no element: ``0`` for ``sum`` and ``any``, ``1`` for
    ``all``. The tile-load factory seeds from the block's first chunk instead and so needs none,
    which it cannot do here because a lane may legitimately have nothing.
    """

    def _k(values: wp.array[wp.bool], out_result: wp.array[wp.int32]) -> None:
        i, t = wp.tid()
        n = values.shape[0]
        base, remaining = block_chunk_1d(n, i)
        if remaining <= 0:
            return
        acc = wp.int32(identity)
        for k in range(t, remaining, wp.block_dim()):
            acc = scalar(acc, wp.where(values[base + k], wp.int32(1), wp.int32(0)))
        block_result = block_reduce(acc)
        if t == 0:
            atomic(out_result, 0, block_result)

    return wp.kernel(_k, name=name)


sum_bool_1d_tiled = _reduce_bool_1d_tiled(block_sum, wp.atomic_add, wp.add, 0, "sum_bool_1d_tiled")
any_bool_1d_tiled = _reduce_bool_1d_tiled(block_max, wp.atomic_max, wp.max, 0, "any_bool_1d_tiled")
all_bool_1d_tiled = _reduce_bool_1d_tiled(block_min, wp.atomic_min, wp.min, 1, "all_bool_1d_tiled")

any_1d_tiled = _reduce_1d_tiled(_tile_max, wp.atomic_max, wp.max, "any_1d_tiled", wp.int32)
any_2d_rows_tiled = _reduce_2d_axis_tiled(
    _tile_max, wp.atomic_max, wp.max, "any_2d_rows_tiled", True, wp.int32
)
any_2d_cols_tiled = _reduce_2d_axis_tiled(
    _tile_max, wp.atomic_max, wp.max, "any_2d_cols_tiled", False, wp.int32
)
any_2d_rows_serial = _reduce_2d_axis_serial(wp.max, "any_2d_rows_serial", True, wp.int32)
any_2d_cols_serial = _reduce_2d_axis_serial(wp.max, "any_2d_cols_serial", False, wp.int32)

all_1d_tiled = _reduce_1d_tiled(_tile_min, wp.atomic_min, wp.min, "all_1d_tiled", wp.int32)
all_2d_rows_tiled = _reduce_2d_axis_tiled(
    _tile_min, wp.atomic_min, wp.min, "all_2d_rows_tiled", True, wp.int32
)
all_2d_cols_tiled = _reduce_2d_axis_tiled(
    _tile_min, wp.atomic_min, wp.min, "all_2d_cols_tiled", False, wp.int32
)
all_2d_rows_serial = _reduce_2d_axis_serial(wp.min, "all_2d_rows_serial", True, wp.int32)
all_2d_cols_serial = _reduce_2d_axis_serial(wp.min, "all_2d_cols_serial", False, wp.int32)


# ---------------------------------------------------------------------------
# Min/max together. The dual tile-reduction, dual atomic and two output slots do
# not fit the single-primitive factory template, so these get their own — the
# axis-parameterized pair below — and the two axis=None forms stay hand-written,
# there being one instantiation of each to generate.
# ---------------------------------------------------------------------------


def _minmax_1d_tiled(name, dtype):
    """
    axis=None on a 1-D array, both extrema in one pass.

    A factory for the reason in [`_reduce_1d_tiled`][ordito.kernels.reduce._reduce_1d_tiled]: a
    ``wp.Scalar`` template would make every launch pay Warp's host-side overload resolution,
    roughly doubling the host cost of a call this small.
    """

    def _k(values: wp.array[wp.Scalar], out_minmax: wp.array[wp.Scalar]) -> None:
        # Same TILES_PER_BLOCK_1D fold as the factory kernels above, with two accumulators seeded
        # from the block's first chunk; two atomics per block instead of two per tile.
        i, t = wp.tid()
        n = values.shape[0]
        base, remaining = tile_chunk(n, i, TILES_PER_BLOCK_1D * TILE_1D)
        if remaining <= 0:
            return

        if remaining >= TILE_1D:
            tile = wp.tile_load(values, shape=TILE_1D, offset=base, storage="register")
            tile_min = wp.tile_min(tile)[0]
            tile_max = wp.tile_max(tile)[0]
        else:
            tile_min = values[base]
            tile_max = values[base]
            for k in range(1, remaining):
                v = values[base + k]
                tile_min = wp.min(tile_min, v)
                tile_max = wp.max(tile_max, v)

        for s in range(1, TILES_PER_BLOCK_1D):
            offset = base + s * TILE_1D
            rest = n - offset
            if rest >= TILE_1D:
                chunk = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
                tile_min = wp.min(tile_min, wp.tile_min(chunk)[0])
                tile_max = wp.max(tile_max, wp.tile_max(chunk)[0])
            elif rest > 0:
                for k in range(rest):
                    v = values[offset + k]
                    tile_min = wp.min(tile_min, v)
                    tile_max = wp.max(tile_max, v)

        if t == 0:
            wp.atomic_min(out_minmax, 0, tile_min)
            wp.atomic_max(out_minmax, 1, tile_max)

    _k.__annotations__["values"] = wp.array[dtype]
    _k.__annotations__["out_minmax"] = wp.array[dtype]
    return wp.kernel(_k, name=name)


def _minmax_2d_tiled(name, dtype):
    """
    axis=None on a 2-D array, both extrema in one pass; concrete for the same reason.

    Reached only for a non-contiguous rank-2 array, for the same reason as
    [`_reduce_2d_tiled`][ordito.kernels.reduce._reduce_2d_tiled] — see its docstring.
    """

    def _k(values: wp.array2d[wp.Scalar], out_minmax: wp.array[wp.Scalar]) -> None:
        i, j, t = wp.tid()
        n_rows = values.shape[0]
        n_cols = values.shape[1]
        row_offset = i * TILE_2D
        col_offset = j * TILE_2D
        if row_offset >= n_rows or col_offset >= n_cols:
            return

        remaining_rows = n_rows - row_offset
        remaining_cols = n_cols - col_offset
        tile_rows = wp.where(remaining_rows >= TILE_2D, TILE_2D, remaining_rows)
        tile_cols = wp.where(remaining_cols >= TILE_2D, TILE_2D, remaining_cols)
        if remaining_rows >= TILE_2D and remaining_cols >= TILE_2D:
            tile = wp.tile_load(
                values,
                shape=(TILE_2D, TILE_2D),
                offset=(row_offset, col_offset),
                storage="register",
            )
            tile_min = wp.tile_min(tile)[0]
            tile_max = wp.tile_max(tile)[0]
            if t == 0:
                wp.atomic_min(out_minmax, 0, tile_min)
                wp.atomic_max(out_minmax, 1, tile_max)
        elif t == 0:
            # Only lane 0 folds the boundary serially -- see _reduce_2d_tiled's comment at the same
            # branch for why every other lane running this loop too would be pure redundant work.
            tile_min = values[row_offset, col_offset]
            tile_max = values[row_offset, col_offset]
            for r in range(tile_rows):
                for c in range(tile_cols):
                    if r == 0 and c == 0:
                        continue
                    v = values[row_offset + r, col_offset + c]
                    tile_min = wp.min(tile_min, v)
                    tile_max = wp.max(tile_max, v)
            wp.atomic_min(out_minmax, 0, tile_min)
            wp.atomic_max(out_minmax, 1, tile_max)

    _k.__annotations__["values"] = wp.array2d[dtype]
    _k.__annotations__["out_minmax"] = wp.array[dtype]
    return wp.kernel(_k, name=name)


def _minmax_2d_axis_tiled(name, rows, dtype):
    """axis=1 (``rows=True``) or axis=0: ``_reduce_2d_axis_tiled`` with both extrema at once."""
    element = _element_along_row if rows else _element_along_col

    def _k(
        values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar], out_max: wp.array[wp.Scalar]
    ) -> None:
        i, j, t = wp.tid()
        if wp.static(rows):
            extent = values.shape[1]
        else:
            extent = values.shape[0]
        offset = j * TILE_1D
        if offset >= extent:
            return
        remaining = extent - offset
        if remaining >= TILE_1D:
            if wp.static(rows):
                tile = wp.tile_load(
                    values, shape=(1, TILE_1D), offset=(i, offset), storage="register"
                )
            else:
                tile = wp.tile_load(
                    values, shape=(TILE_1D, 1), offset=(offset, i), storage="register"
                )
            tile_min = wp.tile_min(tile)[0]
            tile_max = wp.tile_max(tile)[0]
        else:
            tile_min = element(values, i, offset)
            tile_max = tile_min
            for k in range(1, remaining):
                v = element(values, i, offset + k)
                tile_min = wp.min(tile_min, v)
                tile_max = wp.max(tile_max, v)
        if t == 0:
            wp.atomic_min(out_min, i, tile_min)
            wp.atomic_max(out_max, i, tile_max)

    _k.__annotations__["values"] = wp.array2d[dtype]
    _k.__annotations__["out_min"] = wp.array[dtype]
    _k.__annotations__["out_max"] = wp.array[dtype]
    return wp.kernel(_k, name=name)


def _minmax_2d_axis_serial(name, rows, dtype):
    """
    axis=1 (``rows=True``) or axis=0 with a reduced extent under ``TILE_1D``: one thread per output.

    The tiled form above loses ``TILE_1D``-fold there for the reason given on
    [`_reduce_2d_axis_serial`][ordito.kernels.reduce._reduce_2d_axis_serial].
    """
    element = _element_along_row if rows else _element_along_col

    def _k(
        values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar], out_max: wp.array[wp.Scalar]
    ) -> None:
        i = wp.int32(wp.tid())
        if wp.static(rows):
            extent = values.shape[1]
        else:
            extent = values.shape[0]
        slot_min = element(values, i, 0)
        slot_max = slot_min
        for k in range(1, extent):
            v = element(values, i, k)
            slot_min = wp.min(slot_min, v)
            slot_max = wp.max(slot_max, v)
        out_min[i] = slot_min
        out_max[i] = slot_max

    _k.__annotations__["values"] = wp.array2d[dtype]
    _k.__annotations__["out_min"] = wp.array[dtype]
    _k.__annotations__["out_max"] = wp.array[dtype]
    return wp.kernel(_k, name=name)


MINMAX1D_TILED = KernelTable(
    "minmax1d_tiled",
    {d: _minmax_1d_tiled(f"minmax1d_tiled_{d.__name__}", d) for d in _GLOBAL_DTYPES},
)


@wp.func
def strided_extrema(
    values: Any, begin: wp.int32, end: wp.int32, lane: wp.int32, low: Any, high: Any
) -> tuple[Any, Any]:
    # Fold ``values[begin:end]`` into ``(low, high)``, the block's lanes strided by
    # ``wp.block_dim()``.
    for k in range(begin + lane, end, wp.block_dim()):
        v = values[k]
        low = wp.min(low, v)
        high = wp.max(high, v)
    return low, high


@wp.func
def chunk_extrema(
    first: Any, second: Any, n_first: wp.int32, offset: wp.int32, count: wp.int32, lane: wp.int32
) -> tuple[Any, Any]:
    # A lane's ``(low, high)`` over elements ``[offset, offset + count)`` of ``first`` followed by
    # ``second``: the chunk splits at ``n_first`` into a run of each array, walked with no branch
    # an element (one cost the CPU's one-lane blocks 1.4-1.6x). Seeded from the chunk's first
    # element, so no per-dtype identity is needed.
    split = wp.min(wp.max(n_first - offset, 0), count)
    if split > 0:
        low = first[offset]
    else:
        low = second[offset - n_first]
    high = low
    low, high = strided_extrema(first, offset, offset + split, lane, low, high)
    return strided_extrema(
        second, offset + split - n_first, offset + count - n_first, lane, low, high
    )


@wp.func
def commit_block_extrema(
    out_result: Any, low_slot: wp.int32, high_slot: wp.int32, lane: wp.int32, low: Any, high: Any
):
    # The block's minimum into ``out_result[low_slot]`` and maximum into ``[high_slot]`` by one
    # atomic each, a slot ``-1`` committing nothing. Kernels pass literal slots, so the unused
    # branch and its block reduction fold away; the branches are block-uniform either way.
    if low_slot >= 0:
        block_low = block_min(low)
        if lane == 0:
            wp.atomic_min(out_result, low_slot, block_low)
    if high_slot >= 0:
        block_high = block_max(high)
        if lane == 0:
            wp.atomic_max(out_result, high_slot, block_high)


def _extrema_1d_strided(name, dtype, low_slot, high_slot, pair):
    """
    axis=None ``min`` / ``max`` / ``minmax`` on a 1-D array, lane-strided: the CUDA form.

    Each block folds ``ITEMS_PER_BLOCK_1D`` elements (``chunk_extrema``), then one block reduction
    and one atomic a slot into ``out_result``'s seeds (``commit_block_extrema``; ``low_slot`` /
    ``high_slot`` ``-1`` where the kernel commits none). ``pair`` takes a second array and folds
    the two as one concatenation (``array.isin``'s joint span). Against the tiled ``*1D_TILED``
    forms, whose sixteen tile loads a block each end in a barrier, CUDA 2026-11-13: 5.7 / 6.6 /
    204 us a launch at 15 k / 2.6 M / 84 M ``int32`` against 15 / 18 / 547 us (1.6 against
    0.6 TB/s). The CPU keeps the tiled forms (``extrema_1d``): its one-lane blocks run this loop
    unvectorized, 1.27-1.44x the tiled pair from 35 k elements.
    """
    if pair:

        def _k(first: Any, second: Any, out_result: Any) -> None:
            block, lane = wp.tid()
            n_first = first.shape[0]
            offset, count = block_chunk_1d(n_first + second.shape[0], block)
            if count <= 0:
                return
            low, high = chunk_extrema(first, second, n_first, offset, count, lane)
            commit_block_extrema(out_result, low_slot, high_slot, lane, low, high)

        _k.__annotations__["first"] = wp.array[dtype]
        _k.__annotations__["second"] = wp.array[dtype]
    else:

        def _k(values: Any, out_result: Any) -> None:
            block, lane = wp.tid()
            n = values.shape[0]
            offset, count = block_chunk_1d(n, block)
            if count <= 0:
                return
            low, high = chunk_extrema(values, values, n, offset, count, lane)
            commit_block_extrema(out_result, low_slot, high_slot, lane, low, high)

        _k.__annotations__["values"] = wp.array[dtype]
    _k.__annotations__["out_result"] = wp.array[dtype]
    return wp.kernel(_k, name=name)


MIN1D_STRIDED = KernelTable(
    "min1d_strided",
    {
        d: _extrema_1d_strided(f"min1d_strided_{d.__name__}", d, 0, -1, False)
        for d in _GLOBAL_DTYPES
    },
)
MAX1D_STRIDED = KernelTable(
    "max1d_strided",
    {
        d: _extrema_1d_strided(f"max1d_strided_{d.__name__}", d, -1, 0, False)
        for d in _GLOBAL_DTYPES
    },
)
MINMAX1D_STRIDED = KernelTable(
    "minmax1d_strided",
    {
        d: _extrema_1d_strided(f"minmax1d_strided_{d.__name__}", d, 0, 1, False)
        for d in _GLOBAL_DTYPES
    },
)
MINMAX_PAIR_1D = KernelTable(
    "minmax_pair_1d",
    {
        d: _extrema_1d_strided(f"minmax_pair_1d_{d.__name__}", d, 0, 1, True)
        for d in (wp.int32, wp.int64, wp.uint32, wp.uint64)
    },
)


def extrema_1d(kind: str, dtype: Any, device: Any) -> Any:
    """
    Return the axis=None ``"min"`` / ``"max"`` / ``"minmax"`` kernel for a 1-D ``dtype`` array.

    Lane-strided on CUDA, tiled on the CPU (``_extrema_1d_strided``); both launch
    ``dim=[blocks_1d(n)]`` at ``block_dim=TILE_1D`` with ``(values, out_result)``.
    """
    return (_EXTREMA_1D_CUDA if device.is_cuda else _EXTREMA_1D_CPU)[kind][dtype]


_EXTREMA_1D_CUDA = {"min": MIN1D_STRIDED, "max": MAX1D_STRIDED, "minmax": MINMAX1D_STRIDED}
_EXTREMA_1D_CPU = {"min": MIN1D_TILED, "max": MAX1D_TILED, "minmax": MINMAX1D_TILED}


MINMAX2D_TILED = KernelTable(
    "minmax2d_tiled",
    {d: _minmax_2d_tiled(f"minmax2d_tiled_{d.__name__}", d) for d in _GLOBAL_DTYPES},
)
MINMAX_2D_ROWS_TILED = KernelTable(
    "minmax_2d_rows_tiled",
    {d: _minmax_2d_axis_tiled(f"minmax_2d_rows_tiled_{d.__name__}", True, d) for d in _AXIS_DTYPES},
)
MINMAX_2D_COLS_TILED = KernelTable(
    "minmax_2d_cols_tiled",
    {
        d: _minmax_2d_axis_tiled(f"minmax_2d_cols_tiled_{d.__name__}", False, d)
        for d in _AXIS_DTYPES
    },
)
MINMAX_2D_ROWS_SERIAL = KernelTable(
    "minmax_2d_rows_serial",
    {
        d: _minmax_2d_axis_serial(f"minmax_2d_rows_serial_{d.__name__}", True, d)
        for d in _AXIS_DTYPES
    },
)
MINMAX_2D_COLS_SERIAL = KernelTable(
    "minmax_2d_cols_serial",
    {
        d: _minmax_2d_axis_serial(f"minmax_2d_cols_serial_{d.__name__}", False, d)
        for d in _AXIS_DTYPES
    },
)


# ---------------------------------------------------------------------------
# Sums whose dtype the ``wp.Scalar`` template above does not admit: ``wp.vec3``
# values, and the weighted forms whose second array the single-array template
# cannot express. All three come from the same two factories as the scalar
# reductions, so the file has one way of generating a 1-D reduction.
#
# Folding ``TILES_PER_BLOCK_1D`` tiles per block is what these three gained by
# moving onto the template: several-fold at a million elements and more above
# that, because it is the block count reaching the accumulator that is the
# contended quantity. Below a few hundred thousand the fold has too few blocks to
# fill the device and costs a few microseconds -- the trade the scalar reductions
# have made unconditionally since they were written. On CPU there is no such
# crossover: one lane per block means the fold is strictly fewer blocks for the
# same work, and it wins at every size.
# ---------------------------------------------------------------------------

sum_vec2_1d_folded = _reduce_1d_tiled(
    _tile_sum, None, wp.add, "sum_vec2_1d_folded", dtype=wp.vec2, folded=True
)
sum_vec2_1d_partials = _reduce_1d_tiled(
    _tile_sum, None, wp.add, "sum_vec2_1d_partials", dtype=wp.vec2, partials=True
)

sum_vec3_1d_folded = _reduce_1d_tiled(
    _tile_sum, None, wp.add, "sum_vec3_1d_folded", dtype=wp.vec3, folded=True
)
sum_vec3_1d_partials = _reduce_1d_tiled(
    _tile_sum, None, wp.add, "sum_vec3_1d_partials", dtype=wp.vec3, partials=True
)

sum_vec4_1d_folded = _reduce_1d_tiled(
    _tile_sum, None, wp.add, "sum_vec4_1d_folded", dtype=wp.vec4, folded=True
)
sum_vec4_1d_partials = _reduce_1d_tiled(
    _tile_sum, None, wp.add, "sum_vec4_1d_partials", dtype=wp.vec4, partials=True
)

weighted_sum1d_folded = _weighted_sum_1d_tiled("weighted_sum1d_folded", wp.float32, folded=True)
weighted_sum1d_partials = _weighted_sum_1d_tiled("weighted_sum1d_partials", wp.float32, True)
weighted_sum_vec3_1d_folded = _weighted_sum_1d_tiled(
    "weighted_sum_vec3_1d_folded", wp.vec3, folded=True
)
weighted_sum_vec3_1d_partials = _weighted_sum_1d_tiled(
    "weighted_sum_vec3_1d_partials", wp.vec3, True
)

# The most blocks whose rows one last block folds (``reduce.sum_of_terms`` with the deterministic
# setting on). Past it the single ticket counter and the last block's walk over the rows cost more
# than a second launch: stored partials, then their rows summed by ``FOLD_ROWS[dtype]``
# (2026-11-11, CUDA, new/old against the fold launches: 2 M items, 1 954 blocks, 0.83-0.92x; 8 M
# 0.97-1.03x; 14 M 1.11-1.14x for the scalar and ``vec3`` sums, a ``vec3`` ticket being three
# float atomics a block on one address).
FOLDED_ROWS_UP_TO = 4096

# ``reduce.sum``'s own sums and ``reduce.weighted_sum``'s, as ``reduce.sum_of_terms`` takes them:
# the input is the array (twice for the weighted sum).
ARRAY_SUMS = {
    d: TermSum(kernel=SUM1D_FOLDED[d], partials=SUM1D_PARTIALS[d], dtype=d)
    for d in (wp.float32, wp.float64)
}
ARRAY_SUM = ARRAY_SUMS[wp.float32]
VECTOR_SUMS = {
    wp.vec3: TermSum(kernel=sum_vec3_1d_folded, partials=sum_vec3_1d_partials, dtype=wp.vec3),
    wp.vec4: TermSum(kernel=sum_vec4_1d_folded, partials=sum_vec4_1d_partials, dtype=wp.vec4),
}
WEIGHTED_SUM = TermSum(kernel=weighted_sum1d_folded, partials=weighted_sum1d_partials)
WEIGHTED_SUM_VEC3 = TermSum(
    kernel=weighted_sum_vec3_1d_folded, partials=weighted_sum_vec3_1d_partials, dtype=wp.vec3
)
# The sum of a sum's stored partials, by its dtype: the array sum over the rows.
FOLD_ROWS = {
    **ARRAY_SUMS,
    **VECTOR_SUMS,
    wp.vec2: TermSum(kernel=sum_vec2_1d_folded, partials=sum_vec2_1d_partials, dtype=wp.vec2),
}


@wp.func
def outer_sum_chunk(
    points: wp.array[wp.vec3],
    center: wp.vec3,
    offset: wp.int32,
    remaining: wp.int32,
    lane: wp.int32,
    stride: wp.int32,
) -> wp.mat33:
    # ``_chunk``, not ``_tile``: this walks its share of the block's chunk with a plain loop and
    # uses no tile primitive, so it returns one lane's *partial* matrix and the caller is
    # responsible for reducing across lanes. The name says so rather than promising a cooperative
    # reduction that is not here.
    #
    # ``lane`` / ``stride`` are passed rather than read from ``wp.block_dim()`` because this is a
    # ``@wp.func``: the caller is the kernel that knows its own launch shape.
    count = wp.min(remaining, ITEMS_PER_BLOCK_1D)
    # M = sum_k outer(x_k, x_k) where x_k = points[k] - center
    m = wp.mat33(0.0)
    for k in range(lane, count, stride):
        x = points[offset + k] - center
        m += wp.outer(x, x)
    return m


# The chunked-fold skeleton -- walk ``tile_chunk``/``outer_sum_chunk``'s share, then commit the
# packed block sum -- recurs at several call sites (``points.centered_covariance``,
# ``points.neighbor_distance_moments``, ``measures.moment_integrals``,
# ``registration.accumulate_procrustes_moments`` / ``accumulate_point_to_plane``, ``homology``'s
# two counters). The commit tail is shared (``commit_block_sum``, one packed reduction per block);
# the loops are deliberately not. Each folds a different fixed component count, and the bodies
# differ in more than the count: ``moment_integrals`` takes its chunk width as a launch argument
# because its per-element arithmetic has a real occupancy crossover, and
# ``homology.dual_candidate_mask`` and ``points.neighbor_distance_moments`` fold narrower than
# ``ITEMS_PER_BLOCK_1D`` because they also write per element. What *is* shared is already
# factored: ``tile_chunk`` and the clamp rule it documents, which is the part that goes wrong.


@wp.func
def minmax_vec3_chunk(
    points: wp.array[wp.vec3], chunk: wp.int32, out_corners: wp.array[wp.float32], box: wp.int32
) -> None:
    # One ``TILE_1D`` chunk of ``points`` folded into packed box ``box`` of ``out_corners`` -- a box
    # *index*, so its six slots are ``6 * box`` onwards (``array.atomic_min_packed_box``,
    # ``minmax_vec3_chunked``'s layout). Shared by the one-cloud and the two-cloud kernels.
    offset, count = block_chunk(points.shape[0], chunk, TILE_1D)
    if count <= 0:
        return

    lower = points[offset]
    upper = points[offset]
    for k in range(1, count):
        p = points[offset + k]
        lower = wp.min(lower, p)  # wp.min / wp.max on a vector are component-wise
        upper = wp.max(upper, p)

    atomic_min_packed_box(out_corners, box, lower, upper)


@wp.kernel
def minmax_vec3_chunked(points: wp.array[wp.vec3], out_corners: wp.array[wp.float32]) -> None:
    # Component-wise min and max of a ``wp.vec3`` array, in one launch into one buffer.
    #
    # ``out_corners`` is ``[min_x, min_y, min_z, -max_x, -max_y, -max_z]``, *negating* the upper
    # half so a single ``wp.full(6, inf)`` initializes both and every update is an
    # ``atomic_min``. The alternative -- separate min and max buffers -- needs two allocations, two
    # fills and two readbacks, and at this size the reduction is entirely host-latency-bound.
    #
    # One thread per ``TILE_1D`` points, so the atomics see a few hundred contenders per address
    # rather than one per point. Launch it with
    # [`chunks_1d`][ordito.kernels.reduce.chunks_1d] and not ``blocks_1d``: this kernel does not
    # fold ``TILES_PER_BLOCK_1D`` tiles, so the two differ by that factor.
    minmax_vec3_chunk(points, wp.int32(wp.tid()), out_corners, wp.int32(0))


@wp.kernel
def minmax_vec3_gathered_chunked(
    points: wp.array[wp.vec3], indices: wp.array[wp.int32], out_corners: wp.array[wp.float32]
) -> None:
    # ``minmax_vec3_chunked`` over ``points[indices[i]]``: the bounding box of the indexed points
    # (a mesh's referenced vertices, through its face buffer) without the gathered copy, chunk for
    # chunk the same fold as over that copy. Launch with ``chunks_1d(indices.shape[0])``.
    offset, count = block_chunk(indices.shape[0], wp.int32(wp.tid()), TILE_1D)
    if count <= 0:
        return
    lower = points[indices[offset]]
    upper = lower
    for k in range(1, count):
        p = points[indices[offset + k]]
        lower = wp.min(lower, p)
        upper = wp.max(upper, p)
    atomic_min_packed_box(out_corners, wp.int32(0), lower, upper)


@wp.kernel
def minmax_vec3_pair_chunked(
    points_a: wp.array[wp.vec3],
    points_b: wp.array[wp.vec3],
    b_box: wp.int32,
    out_corners: wp.array[wp.float32],
) -> None:
    # ``minmax_vec3_chunked`` over two clouds in one launch, ``chunks_1d(n_a) + chunks_1d(n_b)``
    # wide: the first cloud's chunks fold into box 0, the second's into box ``b_box`` -- 0 for the
    # union of the two, 1 for two boxes side by side (a box *index*: twelve slots in all). No
    # concatenation of the clouds (an allocation and a copy of both, which loses to two launches).
    t = wp.int32(wp.tid())
    chunks_a = (points_a.shape[0] + TILE_1D - 1) // TILE_1D
    if t < chunks_a:
        minmax_vec3_chunk(points_a, t, out_corners, wp.int32(0))
    else:
        minmax_vec3_chunk(points_b, t - chunks_a, out_corners, b_box)


# ---------------------------------------------------------------------------
# Concrete overloads, registered at import.
#
# Every kernel above annotated ``wp.Scalar`` would be *generic*, and Warp instantiates an overload
# lazily, on the first launch at each new dtype — which changes the module hash and recompiles
# every kernel in this file. Registering every (kernel, dtype) pair the wrapper's dispatch can
# reach gives the module one hash for its whole lifetime. Registration is not compilation, so it
# costs milliseconds of import and nothing on a process that never reduces anything.
#
# The trade is honest about one thing: the *single* compile is bigger, since the module holds ~110
# concrete kernels rather than the ~40 a lazy fork built. That is the cost of editing this file,
# not of using it, and it is cached. If it ever becomes the bottleneck, the escape is
# ``@wp.kernel(module="unique")`` per generated kernel, the way ``warp.sparse`` does, so a rebuild
# touches one kernel instead of all of them.
#
# Two rules for keeping it that way:
#
# - **A new generic kernel here must be added to a group below, and a new dtype to the right
#   tuple.** ``test_generic_kernels_register_their_overloads`` catches the first; nothing catches
#   the second, because a missing dtype does not fail — it re-forks the chain on its first launch.
#   The symptom is a test or a script that suddenly takes tens of seconds; read it as a rebuild.
# - **The dtype set is the one ``ordito.reduce`` dispatches over**, not every dtype ``wp.Scalar``
#   admits: an unused overload is compile time paid on every rebuild. The boolean reductions are
#   ``wp.int32`` only because ``_reduce_bool`` converts the mask first, and ``wp.bool`` is not a
#   ``wp.Scalar`` in any case.
#
# ``block_dim`` forks the hash independently and is deliberately left forked. It is not the same
# pathology: the values in use are fixed by *this package's* launch code — ``TILE_1D`` for the
# ``wp.launch_tiled`` reductions, Warp's 256 default for the plain ones, 1 on CPU — so they are a
# bounded set, not a chain whose length grows with what a caller reduces first. Collapsing them by
# passing ``block_dim=TILE_1D`` at the plain-launch sites was measured and declined: a substantial
# loss on the per-axis reductions and nothing at scale.
# ---------------------------------------------------------------------------

# Nothing in this module is generic any more, so there is no ``_register_overloads`` here: every
# kernel above is a concrete factory instantiation, which is what a registered overload *is*. The
# registration rule is unchanged and still binds every other kernel module -- what
# changed is that the tables now hand the wrapper the concrete kernel instead of making
# ``wp.launch`` re-derive it (see the note above ``MIN1D_TILED``).


# ---------------------------------------------------------------------------
# Fused all-close reduction.
#
# ``array.allclose`` ran a ``wp.map`` of the element predicate into an ``(n,)`` mask and then a
# whole ``reduce.all`` over it -- two launches, two allocations, the map's own host-side resolution
# and a reduction that reads back what the predicate pass already knew. The answer is one boolean,
# so the predicate folds into its own reduction: one launch, one four-byte accumulator, one
# readback.
#
# **A table of concrete kernels rather than one generic one**, for ``_reduce_1d_tiled``'s reason
#: a generic kernel pays host-side overload resolution on every launch,
# which on a call this small is most of what the fusion just saved. Registered over exactly the
# dtypes ``allclose``'s own signature admits -- ``float16`` / ``float32`` / ``float64`` and
# ``wp.vec3`` -- which is the registration rule, not every dtype the template would accept.
#
# The fold is ``wp.min`` over a per-lane 0/1, so "all close" is "the block minimum is 1"; lanes with
# no element seed 1 (the identity), and the commit is one ``wp.atomic_min`` per block.
# ---------------------------------------------------------------------------


def _allclose_1d_tiled(name, dtype, predicate, tolerance_dtype):
    """One concrete ``allclose`` kernel: the element predicate folded into its own reduction."""

    def _k(
        a: wp.array[wp.Scalar],
        b: wp.array[wp.Scalar],
        rtol: wp.Scalar,
        atol: wp.Scalar,
        out_flag: wp.array[wp.int32],
    ) -> None:
        i, lane = wp.tid()
        offset, remaining = block_chunk_1d(a.shape[0], i)
        if remaining <= 0:
            return
        close = wp.int32(1)
        for k in range(lane, remaining, wp.block_dim()):
            slot = offset + k
            if not predicate(a[slot], b[slot], rtol, atol):
                close = wp.int32(0)
        block_close = block_min(close)
        if lane == 0:
            wp.atomic_min(out_flag, 0, block_close)

    _k.__annotations__["a"] = wp.array[dtype]
    _k.__annotations__["b"] = wp.array[dtype]
    _k.__annotations__["rtol"] = tolerance_dtype
    _k.__annotations__["atol"] = tolerance_dtype
    return wp.kernel(_k, name=name)


ALLCLOSE_1D_TILED = KernelTable(
    "allclose_1d_tiled",
    {
        d: _allclose_1d_tiled(f"allclose_1d_tiled_{d.__name__}", d, is_close_scalar, d)
        for d in (wp.float16, wp.float32, wp.float64)
    }
    | {wp.vec3: _allclose_1d_tiled("allclose_1d_tiled_vec3", wp.vec3, is_close_vec3, wp.float32)},
)


# ---------------------------------------------------------------------------
# Radix select: an order statistic without a sort.
# ---------------------------------------------------------------------------

# The ``rank``-th smallest (0-based) of an array's order-preserving ``uint64`` keys, found a digit
# a pass from the top: every key carrying the digits selected so far counts into its digit's bin
# of a ``RADIX_SELECT_BINS`` histogram, and the pass's last block finds the bin holding the rank
# still sought. A 32-bit key takes three passes (``RADIX_SELECT_PASSES_32``), a 64-bit one six.
# On CUDA it replaces a full radix sort read at one rank: three passes of ~5 us against the
# ~50 us sort of a 10 k array (four onesweep passes), at par from 1 M keys. Two 16-bit passes
# lose: the last block reading 65 536 bins is one SM's bandwidth (~33 us coalesced). The CPU's
# radix sort beats the passes. Used by ``reduce.quantile`` / ``median`` and the Poisson-disk
# elimination's weight floor (``kernels/sample.floor_digit``, with its own keys).
RADIX_SELECT_BINS = 1 << 11
RADIX_SELECT_PASSES_32 = ((21, 11), (10, 11), (0, 10))
RADIX_SELECT_PASSES_64 = ((53, 11), (42, 11), (31, 11), (20, 11), (9, 11), (0, 9))
# A block's share of the array, its lanes striding over it.
RADIX_SELECT_ITEMS_PER_BLOCK = 256
# The ``uint64`` state of one select, zeroed: the digits selected so far (after the last pass the
# whole key), the rank still sought below them (then the rank within the key's run of equals;
# the first pass takes it as ``seed_rank``), the length of that run, and the smallest key above it
# complemented (``UINT64_MAX - key``, ``radix_select_successor``), so zero is "none". Nothing in
# it needs a nonzero seed, so it shares one zeroed allocation with the histogram and the ticket.
SELECT_PREFIX = 0
SELECT_RANK = 1
SELECT_RUN = 2
SELECT_SUCCESSOR = 3
SELECT_STATE_SIZE = 4


@wp.func
def radix_select_digit(
    lane: wp.int32, rank: wp.int32, hist: wp.array[wp.int32]
) -> tuple[wp.int32, wp.int32, wp.int32]:
    # In a pass's last block, every lane: the digit whose bin of ``hist`` holds the ``rank``-th
    # (0-based) counted key, the rank within that bin and the bin's count, block-uniform. Each lane
    # sums its contiguous share of the bins, one scan of the shares finds the owning lane, and that
    # lane walks its share (eight bins at 256 lanes, all of them on the CPU device's one lane).
    # Every lane then re-zeroes its share, so the next pass needs no fill.
    span = RADIX_SELECT_BINS // wp.block_dim()
    first = lane * span
    share = wp.int32(0)
    for m in range(first, first + span):
        share += hist[m]
    through, _total = block_running_sum(share, wp.int32(0))
    digit = wp.int32(-1)
    within = wp.int32(0)
    run = wp.int32(0)
    if through - share <= rank and rank < through:
        before = through - share
        for m in range(first, first + span):
            count_m = hist[m]
            if digit < 0 and rank < before + count_m:
                digit = m
                within = rank - before
                run = count_m
            before += count_m
    for m in range(first, first + span):
        hist[m] = 0
    return block_max(digit), block_max(within), block_max(run)


@wp.func
def radix_select_bin(
    key: wp.uint64, prefix: wp.uint64, shift: wp.int32, width: wp.int32
) -> wp.int32:
    # The bin ``key`` counts into in the pass taking ``width`` bits at ``shift``, or ``-1`` when it
    # lacks the digits selected so far. A 64-bit key's first pass has no prefix to match (a shift
    # by the whole width is undefined).
    if shift + width < 64 and (key >> wp.uint64(shift + width)) != prefix:
        return -1
    return wp.int32((key >> wp.uint64(shift)) & ((wp.uint64(1) << wp.uint64(width)) - wp.uint64(1)))


@wp.func
def radix_select_close(
    lane: wp.int32,
    width: wp.int32,
    seed_rank: wp.int32,
    hist: wp.array[wp.int32],
    state: wp.array[wp.uint64],
) -> None:
    # A pass's last block: append the digit holding the rank still sought (``seed_rank`` on the
    # first pass, a negative one on every later pass, which reads the state's) and keep the rank
    # within it and its bin's count.
    rank = wp.where(seed_rank >= 0, seed_rank, wp.int32(state[SELECT_RANK]))
    digit, within, run = radix_select_digit(lane, rank, hist)
    if lane == 0:
        state[SELECT_PREFIX] = (state[SELECT_PREFIX] << wp.uint64(width)) | wp.uint64(digit)
        state[SELECT_RANK] = wp.uint64(within)
        state[SELECT_RUN] = wp.uint64(run)


@wp.func
def ordered_int32_key(x: wp.int32) -> wp.uint64:
    return wp.uint64(wp.cast(x, wp.uint32) ^ wp.uint32(0x80000000))


@wp.func
def ordered_uint32_key(x: wp.uint32) -> wp.uint64:
    return wp.uint64(x)


@wp.func
def ordered_float32_key(x: wp.float32) -> wp.uint64:
    return wp.uint64(ordered_float_bits(x))


@wp.func
def ordered_int64_key(x: wp.int64) -> wp.uint64:
    return wp.cast(x, wp.uint64) ^ (wp.uint64(1) << wp.uint64(63))


@wp.func
def ordered_uint64_key(x: wp.uint64) -> wp.uint64:
    return x


def _radix_select_pass(name: str, dtype: Any, key: wp.Function) -> wp.Kernel:
    def radix_select_pass(
        values: wp.array[Any],
        shift: wp.int32,
        width: wp.int32,
        seed_rank: wp.int32,
        hist: wp.array[wp.int32],
        ticket: wp.array[wp.int32],
        state: wp.array[wp.uint64],
    ) -> None:
        # One digit pass of ``values``' select (launched tiled, ``RADIX_SELECT_ITEMS_PER_BLOCK``
        # items a block): every key carrying the selected digits counts into its bin, peers pooled
        # (the top digits crowd a few bins), and the last block, whose reads see every block's
        # adds, appends the digit.
        block, lane = wp.tid()
        prefix = state[SELECT_PREFIX]
        n = values.shape[0]
        offset, items = block_chunk(n, block, RADIX_SELECT_ITEMS_PER_BLOCK)
        for t in range(offset + lane, offset + items, wp.block_dim()):
            b = radix_select_bin(key(values[t]), prefix, shift, width)
            if b >= 0:
                peer_counted_add(hist, b)
        n_blocks = (n + RADIX_SELECT_ITEMS_PER_BLOCK - 1) // RADIX_SELECT_ITEMS_PER_BLOCK
        if last_atomic_block(lane, ticket, n_blocks) != 0:
            radix_select_close(lane, width, seed_rank, hist, state)

    radix_select_pass.__annotations__["values"] = wp.array[dtype]
    return wp.kernel(radix_select_pass, name=name)


def _radix_select_successor(name: str, dtype: Any, key: wp.Function) -> wp.Kernel:
    def radix_select_successor(values: wp.array[Any], state: wp.array[wp.uint64]) -> None:
        # The smallest key above the selected one, complemented into the zeroed
        # ``state[SELECT_SUCCESSOR]``: a block minimum, then one atomic maximum a block.
        block, lane = wp.tid()
        selected = state[SELECT_PREFIX]
        n = values.shape[0]
        offset, items = block_chunk(n, block, RADIX_SELECT_ITEMS_PER_BLOCK)
        best = UINT64_MAX_CONSTANT
        for t in range(offset + lane, offset + items, wp.block_dim()):
            k = key(values[t])
            if k > selected:
                best = wp.min(best, k)
        lowest = block_min(best)
        if lane == 0 and lowest != UINT64_MAX_CONSTANT:
            wp.atomic_max(state, SELECT_SUCCESSOR, UINT64_MAX_CONSTANT - lowest)

    radix_select_successor.__annotations__["values"] = wp.array[dtype]
    return wp.kernel(radix_select_successor, name=name)


_RADIX_SELECT_KEYS = {
    wp.int32: ordered_int32_key,
    wp.uint32: ordered_uint32_key,
    wp.float32: ordered_float32_key,
    wp.int64: ordered_int64_key,
    wp.uint64: ordered_uint64_key,
    wp.float64: ordered_double_bits,
}
RADIX_SELECT_PASS = KernelTable(
    "radix_select_pass",
    {
        d: _radix_select_pass(f"radix_select_pass_{d.__name__}", d, key)
        for d, key in _RADIX_SELECT_KEYS.items()
    },
)
RADIX_SELECT_SUCCESSOR = KernelTable(
    "radix_select_successor",
    {
        d: _radix_select_successor(f"radix_select_successor_{d.__name__}", d, key)
        for d, key in _RADIX_SELECT_KEYS.items()
    },
)
