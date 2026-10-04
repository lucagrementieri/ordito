"""Global reductions: tiled ``wp.tile_load`` reductions generated from a shared template."""

from typing import Any

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
# CPU device a block is one lane and each returns that lane's own value (CLAUDE.md section 2.2).
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


@wp.func
def commit_sum_and_count(
    lane: wp.int32, total: wp.float64, count: wp.float64, out_sum_and_count: wp.array[wp.float64]
):
    # The commit of a fused "mean over the entries that qualify" block fold: each lane's register
    # sum and count, as one ``commit_block_sum`` pair into the two-slot buffer the caller reads
    # once. Shared by ``heat.upper_edge_length_sum_and_count``, ``heat.source_and_global_sums``,
    # ``points.neighbor_distance_moments`` and ``reconstruction.positive_finite_sum_and_count``,
    # which differ only in what qualifies.
    commit_block_sum(lane, wp.vec2d(total, count), out_sum_and_count, 0)


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


def _reduce_1d_tiled(tile_reduce, atomic, scalar, name, dtype, partials=False):
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
    """

    def _k(values: wp.array[wp.Scalar], out_result: wp.array[wp.Scalar]) -> None:
        i, t = wp.tid()
        n = values.shape[0]
        base, remaining = tile_chunk(n, i, TILES_PER_BLOCK_1D * TILE_1D)
        if remaining <= 0:
            return

        # First chunk seeds the accumulator (both branches assign it -- see CLAUDE.md section 1.4 on
        # Warp's conditional scoping).
        if remaining >= TILE_1D:
            tile = wp.tile_load(values, shape=TILE_1D, offset=base, storage="register")
            result = tile_reduce(tile)[0]
        else:
            result = values[base]
            for k in range(1, remaining):
                result = scalar(result, values[base + k])

        for s in range(1, TILES_PER_BLOCK_1D):
            offset = base + s * TILE_1D
            rest = n - offset
            if rest >= TILE_1D:
                chunk = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
                result = scalar(result, tile_reduce(chunk)[0])
            elif rest > 0:
                for k in range(rest):
                    result = scalar(result, values[offset + k])

        if t == 0:
            if wp.static(partials):
                out_result[i] = result
            else:
                atomic(out_result, 0, result)

    _k.__annotations__["values"] = wp.array[dtype]
    _k.__annotations__["out_result"] = wp.array[dtype]
    return wp.kernel(_k, name=name)


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
    ``TILES_PER_BLOCK_1D`` fold this kernel does not (CLAUDE.md section 13.2) — flattening a
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
    the wrapper must pre-fill the output with the reduction's identity.
    """
    element = _element_along_row if rows else _element_along_col

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
        if t == 0:
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
# 24 kernels would be 72 more kernels to compile on every rebuild for no call site (CLAUDE.md
# section 4.2). A caller who does reduce a ``wp.uint64`` table along an axis pays one fork, once.
_AXIS_DTYPES = (wp.int32, wp.float32, wp.float64)


# ---------------------------------------------------------------------------
# Scalar reductions (min / max / sum) over ``wp.Scalar`` arrays.
# ---------------------------------------------------------------------------

# One concrete kernel per dtype, and the tables are what the wrapper launches. The factories have
# always been able to bake the dtype in -- ``sum_vec3_1d_tiled`` and the weighted sums below always
# did -- and leaving the rest at the ``wp.Scalar`` template made every launch pay Warp's host-side
# ``infer_argument_types`` over the whole argument list, which roughly doubles a small reduction's
# host cost and every scalar-returning reduction in the package issues one. Nothing extra is
# compiled: these are the same instantiations ``_register_overloads`` was already creating.
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

    ``wp.Scalar`` does not instantiate for ``wp.bool`` (CLAUDE.md section 12.4), so the factories
    above cannot serve a mask and the wrapper used to widen one with ``array.astype`` first: an
    allocation of ``4n`` bytes, a launch, a full read of ``n`` and a full write of ``4n``, after
    which the reduction read ``4n`` rather than ``n`` -- nine bytes of traffic per mask byte, plus
    a launch and an allocation, to answer one boolean. This reads the mask directly.

    **The shape differs from ``_reduce_1d_tiled``'s and the difference is forced.** There is no
    ``wp.tile_load`` of a ``bool`` array, so the block's chunk is walked by the lanes rather than
    loaded as tiles: each lane strides by ``wp.block_dim()`` -- never by ``TILE_1D``, which is what
    keeps it correct on the CPU device, where ``wp.launch_tiled`` runs one lane per block and
    ``wp.block_dim()`` reads 1 (section 2.2) -- accumulates in a register, and the block folds the
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

sum_vec3_1d_tiled = _reduce_1d_tiled(
    _tile_sum, wp.atomic_add, wp.add, "sum_vec3_1d_tiled", dtype=wp.vec3
)

sum_vec3_1d_partials = _reduce_1d_tiled(
    _tile_sum, None, wp.add, "sum_vec3_1d_partials", dtype=wp.vec3, partials=True
)

weighted_sum1d_partials = _weighted_sum_1d_tiled("weighted_sum1d_partials", wp.float32, True)
weighted_sum_vec3_1d_partials = _weighted_sum_1d_tiled(
    "weighted_sum_vec3_1d_partials", wp.vec3, True
)


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


# ---------------------------------------------------------------------------
# Concrete overloads, registered at import.
#
# Every kernel above annotated ``wp.Scalar`` would be *generic*, and Warp instantiates an overload
# lazily, on the first launch at each new dtype — which changes the module hash and recompiles
# every kernel in this file. Registering every (kernel, dtype) pair the wrapper's dispatch can
# reach gives the module one hash for its whole lifetime. Registration is not compilation, so it
# costs milliseconds of import and nothing on a process that never reduces anything (CLAUDE.md
# section 2.5).
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
#   The symptom is a test or a script that suddenly takes tens of seconds; read it as a rebuild
#   (CLAUDE.md section 15.1).
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
# rule in CLAUDE.md section 2.5 is unchanged and still binds every other kernel module -- what
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
# (CLAUDE.md section 2.7): a generic kernel pays host-side overload resolution on every launch,
# which on a call this small is most of what the fusion just saved. Registered over exactly the
# dtypes ``allclose``'s own signature admits -- ``float16`` / ``float32`` / ``float64`` and
# ``wp.vec3`` -- which is section 2.5's rule, not every dtype the template would accept.
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
