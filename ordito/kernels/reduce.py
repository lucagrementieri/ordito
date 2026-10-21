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

from ordito.constants import INT32_MAX_CONSTANT, TILE_1D, TILE_2D, TILES_PER_BLOCK_1D
from ordito.kernels.array import KernelTable, atomic_min_packed_box, is_close_scalar, is_close_vec3

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


# Elements one block of a 1-D global reduction owns. It is a ``wp.constant`` because both halves
# of the contract read it and they must read the *same* number: [`blocks_1d`] derives the launch
# width from it at Python scope, and a kernel-scope ``tile_chunk(n, chunk, ITEMS_PER_BLOCK_1D)``
# derives each block's slice from it. A launch and a kernel that disagree here do not fail -- they
# fold some elements twice and drop others, which for a sum is a wrong answer and for an extremum
# looks right. See [`blocks_1d`] for the same hazard stated from the launch side.
ITEMS_PER_BLOCK_1D = wp.constant(TILE_1D * TILES_PER_BLOCK_1D)


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
    # ``out[counter]`` in ``out``'s own dtype (``unit`` is any value of it; whole counts are exact).
    # The fence publishes the row before the ticket can be seen. A vector or matrix counter is one
    # atomic per component, so only a component's count is a ticket: ``last_ticket_vector`` and
    # ``last_ticket_matrix`` read the first.
    thread_fence()
    return wp.atomic_add(out, counter, type(unit)(wp.float32(1.0)))


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
        out[counter] = type(total)(wp.float32(0.0))


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
def store_block_sum(
    block: wp.int32,
    lane: wp.int32,
    local: Any,
    out: wp.array[Any],
    base: wp.int32,
    stride: wp.int32,
):
    # The reproducible ``commit_block_sum``: the block's fold of each lane's register vector into
    # slots ``base:base + len(local)`` of its row, and of the result row once the last block has
    # added every row (see the note above). Every block of the launch must reach this, and every
    # lane of it (it is block-collective). The vector-row form of ``stage_block_value`` and
    # ``fold_block_values``: a row of ``len(local)`` slots stored and added at once.
    folded = block_sum(local)
    if out.shape[0] == stride:
        if lane == 0:
            for slot in range(len(folded)):
                wp.atomic_add(out, base + slot, folded[slot])
        return
    n_blocks = (out.shape[0] - stride - 1) // stride
    if lane == 0:
        row = stride + 1 + block * stride + base
        for slot in range(len(folded)):
            out[row + slot] = folded[slot]
    if block_last_ticket(lane, out, stride, folded[0], n_blocks, last_ticket) != 0:
        thread_fence()
        # Row by row: every slot of a row is an independent load, so ``len(local)`` are in flight
        # where a slot-by-slot walk waited on each row in turn. A narrow row is walked two at a
        # time for twice that; a wide one (``registration``'s 25 and 44 slots) one at a time, which
        # a second accumulator would push out of registers.
        zero = type(local[0])(wp.float32(0.0))
        even = local * zero
        odd = local * zero
        step = wp.block_dim()
        r = lane
        if len(local) <= 16:
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
    folded = block_sum(local)
    if stage_block_fold(block, lane, folded, out, slot, stride, is_last) != 0:
        n_blocks = (out.shape[0] - stride - 1) // stride
        finish_block_total(lane, out, slot, stride, n_blocks, type(folded)(wp.float32(0.0)))


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
    # ``store_block_total`` whose reproducible result slot holds the total divided by ``divisor``,
    # the division done once by the last block, so a mean needs no launch of its own. The
    # ``atomic_add`` form has no last block: its slot holds the undivided total, and the caller
    # divides (``ordito.config.is_deterministic``, which ``_launch.block_partials`` reads to pick
    # the form, tells it which it got). The
    # division is ``total / divisor`` per component, the one a ``wp.map(wp.div, ...)`` makes.
    folded = block_sum(local)
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
# may call it instead of the finish launch. ``kernels/scatter.scatter_add`` and its vector and
# matrix siblings are the entry points, and ``_launch.group_sums`` allocates the buffer.
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
# run), the words are summed across the warp first and one lane commits (``warp_uniform_sum``):
# 10 M terms into one slot ran 0.011x the time of the per-term ``float32`` atomic. Declined:
# matching any subset of lanes (``__match_any_sync``), 1.5x the plain commit on coherent runs of
# six and no gain on few randomly ordered groups (512 cells, 2 M terms), whose collisions are
# across warps, not within one.
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
# then does the plain ``atomic_add`` into the caller's float accumulator.
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


# A warp's words summed before the atomic when every active lane holds the same ``key`` (one
# shuffle and a vote decide): the total to the lowest active lane, 0 to the others; otherwise each
# lane's own ``value``. The CPU device runs one lane at a time and returns ``value``.
_WARP_UNIFORM_SUM = """
#if defined(__CUDA_ARCH__)
const unsigned active = __activemask();
if (!__all_sync(active, __shfl_sync(active, key, __ffs(active) - 1) == key)) {
    return value;
}
const unsigned lane = threadIdx.x & 31u;
const unsigned rank = __popc(active & ((1u << lane) - 1u));
const unsigned count = __popc(active);
long long total = value;
for (unsigned offset = 1u; offset < count; offset <<= 1) {
    const bool has = rank + offset < count;
    const int source = has ? (int)__fns(active, 0u, (int)(rank + offset + 1u)) : (int)lane;
    const long long other = __shfl_sync(active, total, source);
    if (has) {
        total += other;
    }
}
return rank == 0u ? total : 0ll;
#else
return value;
#endif
"""


@wp.func_native(_WARP_UNIFORM_SUM)
def warp_uniform_sum(key: wp.int32, value: wp.int64) -> wp.int64: ...


@wp.func
def commit_group_sum_word(sums: GroupSumPass, index: wp.int32, word: wp.int64):
    # One word of a term into ``sums.words[index]`` (a slot's high or low word): summed across the
    # warp first where the warp shares the word (``warp_uniform_sum``, keyed on ``index`` so lanes
    # converged on a high and a low word of one slot never pool), committed only where non-zero.
    total = warp_uniform_sum(index, word)
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
def group_sum_value(sums: GroupSumPass, group: wp.int32, slot: wp.int32) -> wp.float64:
    # Slot ``slot``'s sum (of group ``group``) once pass 1 has added every term: the carry
    # normalized (``lo`` to ``[0, 2^(62 - h))``, flooring for a negative total), the words read as
    # one ``float64``, and the scale undone in two power-of-two factors (``k`` spans
    # ``[-993, 1083]``). NaN for a non-finite group. A consumer kernel may call this itself instead
    # of launching the finish.
    field = group_sum_group_field(sums, group)
    if field == GROUP_SUM_NONFINITE:
        return wp.float64(wp.nan)
    headroom = sums.headroom
    high = sums.words[sums.n_fields + slot]
    total = wp.float64(0.0)
    if group_sum_two_words(sums) != 0:
        width = 62 - headroom
        low = sums.words[sums.n_fields + sums.n_slots + slot]
        high += low >> wp.int64(width)
        low = low & ((wp.int64(1) << wp.int64(width)) - wp.int64(1))
        total = wp.float64(high) + wp.float64(low) * group_sum_power(-width)
    else:
        total = wp.float64(high)
    undo = wp.max(field, 1) + headroom - 1084
    first = undo >> 1
    return total * group_sum_power(first) * group_sum_power(undo - first)


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


def _reduce_1d_tiled(tile_reduce, atomic, scalar, name, dtype, partials=False, term=None):
    """
    axis=None on a 1-D array: ``TILES_PER_BLOCK_1D`` tiles per block, one atomic per block.

    With ``partials=True`` the block's result is *stored* at ``out_result[block]`` instead of
    committed with ``atomic`` (which is then unused): the deterministic float sum's first stage,
    and, launched over its own output, the fold of those partials. A float ``atomic_add`` commits
    blocks in arrival order, so its total moves in the last bits from run to run on CUDA; stored
    partials folded by the same fixed tree give one answer on every run.

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
    module, so a caller's term family never changes this module's hash.
    """
    element = array_term if term is None else term

    def _k(source: Any, out_result: Any) -> None:
        i, t = wp.tid()
        if wp.static(term is None):
            n = source.shape[0]
        else:
            n = source.n
        base, remaining = tile_chunk(n, i, TILES_PER_BLOCK_1D * TILE_1D)
        if remaining <= 0:
            return
        if wp.static(term is not None):
            local = wp.tile_arange(TILE_1D, dtype=wp.int32, storage="register")

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

        for s in range(1, TILES_PER_BLOCK_1D):
            offset = base + s * TILE_1D
            rest = n - offset
            if rest >= TILE_1D:
                if wp.static(term is None):
                    chunk = wp.tile_load(source, shape=TILE_1D, offset=offset, storage="register")
                else:
                    chunk = wp.tile_map(term, local, offset, source)
                result = scalar(result, tile_reduce(chunk)[0])
            elif rest > 0:
                for k in range(rest):
                    result = scalar(result, element(k, offset, source))

        if t == 0:
            if wp.static(partials):
                out_result[i] = result
            else:
                atomic(out_result, 0, result)

    if term is None:
        _k.__annotations__["source"] = wp.array[dtype]
        _k.__annotations__["out_result"] = wp.array[dtype]
        return wp.kernel(_k, name=name)
    _k.__annotations__["source"] = dtype
    _k.__annotations__["out_result"] = wp.array[wp.float32]
    return wp.kernel(_k, name=name, module=term.module)


def _weighted_sum_1d_tiled(name, dtype, partials=False):
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

    ``partials=True`` stores each block's sum at ``out_sum[block]``, as in ``_reduce_1d_tiled``.
    """

    def _k(
        values: wp.array[wp.Scalar], weights: wp.array[wp.float32], out_sum: wp.array[wp.Scalar]
    ) -> None:
        i, t = wp.tid()
        n = values.shape[0]
        base, remaining = tile_chunk(n, i, TILES_PER_BLOCK_1D * TILE_1D)
        if remaining <= 0:
            return

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
                w_chunk = wp.tile_load(weights, shape=TILE_1D, offset=offset, storage="register")
                result = result + wp.tile_sum(v_chunk * w_chunk)[0]
            elif rest > 0:
                for k in range(rest):
                    result = result + values[offset + k] * weights[offset + k]

        if t == 0:
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
# The deterministic float sums' stages (see ``_reduce_1d_tiled``). Integer sums are exact in any
# order and keep the one-launch atomic form.
SUM1D_PARTIALS = KernelTable(
    "sum1d_partials",
    {
        d: _reduce_1d_tiled(_tile_sum, None, wp.add, f"sum1d_partials_{d.__name__}", d, True)
        for d in (wp.float32, wp.float64)
    },
)


class TermSum(NamedTuple):
    """
    The two first stages of a ``float32`` block sum, the pair ``reduce.sum_of_terms`` launches.

    ``partials`` stores one sum per block, folded after by ``SUM1D_PARTIALS`` (the fixed-order
    tree); ``atomic`` adds each block's sum into slot 0, the form taken when ``ordito.config``
    turns determinism off, or ``None`` where the sum takes the fixed order either way (the
    weighted sum, as ``reduce.weighted_sum`` does). Each kernel takes its inputs followed by its
    output.
    """

    partials: wp.Kernel
    atomic: wp.Kernel | None


def term_sum_1d(term: wp.Function, terms_type: type, name: str) -> TermSum:
    """
    Build the [`TermSum`][ordito.kernels.reduce.TermSum] of ``term(local, offset, terms)``.

    The fused map and reduction: ``reduce.sum``'s two ``float32`` kernels in their mapped form (see
    [`_reduce_1d_tiled`][ordito.kernels.reduce._reduce_1d_tiled]), so the total is bit for bit
    ``reduce.sum`` of the materialized terms in either determinism setting, with no term buffer.
    ``terms_type`` is the ``@wp.struct`` the kernels take, holding the item count ``n`` and the
    arrays ``term`` reads; ``name`` must be unique, and the two kernels are named ``name`` and
    ``name + "_partials"``.
    """
    return TermSum(
        partials=_reduce_1d_tiled(
            _tile_sum, None, wp.add, f"{name}_partials", terms_type, True, term=term
        ),
        atomic=_reduce_1d_tiled(_tile_sum, wp.atomic_add, wp.add, name, terms_type, term=term),
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

sum_vec3_1d_partials = _reduce_1d_tiled(
    _tile_sum, None, wp.add, "sum_vec3_1d_partials", dtype=wp.vec3, partials=True
)

sum_vec4_1d_partials = _reduce_1d_tiled(
    _tile_sum, None, wp.add, "sum_vec4_1d_partials", dtype=wp.vec4, partials=True
)

weighted_sum1d_partials = _weighted_sum_1d_tiled("weighted_sum1d_partials", wp.float32, True)
weighted_sum_vec3_1d_partials = _weighted_sum_1d_tiled(
    "weighted_sum_vec3_1d_partials", wp.vec3, True
)


# ``reduce.sum``'s own ``float32`` pair, and ``reduce.weighted_sum``'s fixed-order stage, as
# ``reduce.sum_of_terms`` takes them: the input is the array (twice for the weighted sum).
ARRAY_SUM = TermSum(partials=SUM1D_PARTIALS[wp.float32], atomic=SUM1D_TILED[wp.float32])
WEIGHTED_SUM = TermSum(partials=weighted_sum1d_partials, atomic=None)


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


@wp.kernel
def minmax_rows3_int32_chunked(rows: wp.array2d[wp.int32], out_corners: wp.array[wp.int32]) -> None:
    # ``minmax_vec3_chunked`` for ``(n, 3)`` ``int32`` rows (integer cell coordinates): one thread
    # per ``TILE_1D`` rows, six atomic minima into ``[min_x, min_y, min_z, ~max_x, ~max_y,
    # ~max_z]`` seeded ``INT32_MAX``. The upper half is bit-complemented rather than negated --
    # ``~v`` is ``-v - 1``, which cannot overflow at ``INT32_MIN`` -- so one fill initializes both
    # halves and the caller reads the six back in one copy. Launch with ``chunks_1d``.
    offset, count = block_chunk(rows.shape[0], wp.int32(wp.tid()), TILE_1D)
    if count <= 0:
        return
    lower = wp.vec3i(rows[offset, 0], rows[offset, 1], rows[offset, 2])
    upper = lower
    for k in range(1, count):
        row = wp.vec3i(rows[offset + k, 0], rows[offset + k, 1], rows[offset + k, 2])
        lower = wp.min(lower, row)
        upper = wp.max(upper, row)
    for axis in range(3):
        wp.atomic_min(out_corners, axis, lower[axis])
        wp.atomic_min(out_corners, 3 + axis, ~upper[axis])


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
