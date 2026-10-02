"""
Blue-noise surface sampling: maximal Poisson-disk selection from a dense pool.

**Randomized-priority parallel dart throwing.** Every pool point draws a priority from the seed; a
point is accepted when no *smaller-priority* point still in play sits within ``r`` of it; everything
within ``r`` of an accepted point is then discarded. Iterate until the pool is empty.

Three properties fall out of that, and they are why this replaced the Bridson active-list walk this
module used to hold:

- **The minimum distance is exact, not approximate.** Two accepted points cannot be within ``r`` of
  each other: within one round the larger-priority one of the pair would have seen the smaller and
  declined, and across rounds the later one would already have been discarded by the earlier one's
  ``r``-ball.
- **The result is maximal**, so the uncovered gap is bounded: a survivor is by definition a point no
  accepted point covers, so the loop cannot stop while an ``r``-gap remains. The measured worst gap
  is *tighter* than MeshLab's hierarchical dart throwing and Open3D's sample elimination at the same
  parameter.
- **It is the serial algorithm's distribution.** Accepting local priority minima and discarding
  their balls, repeatedly, visits the pool in exactly priority order, so the output is what
  sequential dart throwing over a uniformly random order gives -- parallel, not approximate.
  And because each pass writes only its own thread's slot and reads only states it cannot itself
  change, a round is a deterministic function of its input state: same seed, same samples.

The round count is what makes it fast. The smallest-priority point still in play is always a local
minimum, so every round accepts something, and each acceptance discards a whole ``r``-ball -- which
empties the pool in a handful of rounds where an active-list walk needed a hundred or so, for a
several-fold end-to-end win at a slightly higher sample count.

Background grid cells are ``r`` on a side, so a ``3x3x3`` neighbourhood contains every point within
``r`` and both sweeps are 27 cells. Bridson's, by contrast, had to enumerate a ``9x9x9`` shell per
active parent per round to find its ``[r, 2r]`` annulus -- 729 cells against 27, which together with
the round count is the whole difference.

**Most of those 27 cells cannot affect the answer, and one word per cell says which.** Each sweep
vetoes a candidate on a single property of the points in a neighbouring cell -- an acceptance for
the covering sweep, a smaller priority for the selection sweep -- so a per-cell summary of that
property decides the whole cell without loading a point from it. Two summaries are rebuilt per
round, at a cost of two fills and one atomic pass over the work list (after the first round that
pass rides on the previous round's compaction); they remove work whose outcome
was already determined, so the accepted set is identical by construction rather than by tolerance,
and together they are worth roughly 2x.

**And those summaries are why the obvious next step is not one.** The membership lists are built
once over the whole pool and never compacted, so as rounds retire points the sweeps walk lists that
are mostly dead -- entries scanned exceed live ones severalfold. Compacting the covered entries out
is provably answer-preserving (the selection sweep skips a ``DART_COVERED`` vetoer explicitly and
the covering sweep only ever matches ``DART_ACCEPTED``, so the dropped entries are exactly the ones
both already walked past) and it was built, byte-gated and **refuted** as a small loss.

The reason is the summaries above, and it is worth stating because the dead-entry ratio looks
compelling on its own. A per-cell summary skips a cell *without loading a point from it*, so the
inner loop only ever runs on cells that genuinely hold a smaller-priority alive point or a fresh
acceptance -- and those cells are mostly live, so there is little dead weight left to remove. The
device time barely moves and the compaction's own launches and readbacks are the whole loss.
Occupancy stopped being the cost when the summaries landed; do not re-derive this from the alive
counts.
"""

import warp as wp

from triwarp.kernels import array as kernel_array
from triwarp.kernels.array import element_priority, sorted_run_start

INVALID = wp.constant(wp.int32(-1))

# Cells are ``r`` wide, so the ball of radius ``r`` around any point in a cell is contained in that
# cell's 3x3x3 neighbourhood. Unlike the Bridson table, the centre slot points at the cell *itself*:
# both sweeps below have to see the candidate's own bucket.
DART_SHELL_W = 3
DART_SHELL_CELLS = DART_SHELL_W * DART_SHELL_W * DART_SHELL_W
_DART_SHELL_W = wp.constant(wp.int32(DART_SHELL_W))

# Pool-point states. ``ALIVE`` is still in play, ``ACCEPTED`` is in the output, ``COVERED`` is
# within ``r`` of an accepted point and can never become either.
DART_ALIVE = wp.constant(wp.int32(0))
DART_ACCEPTED = wp.constant(wp.int32(1))
DART_COVERED = wp.constant(wp.int32(2))

# Empty-cell sentinel for the per-cell priority summary below. A drawn priority can legitimately
# *equal* it, which costs nothing: a thread only skips a cell whose summary is *strictly* greater
# than its own key, so a cell holding a real ``0xffffffff`` is never wrongly skipped -- the only
# thread that could skip it holds a smaller key, and a larger-priority point can never veto it.
DART_NO_PRIORITY = wp.constant(wp.uint32(0xFFFFFFFF))


@wp.func
def cell_key(w: wp.int64, x: wp.int64, y: wp.int64, z: wp.int64) -> wp.int64:
    return x + w * (y + w * z)


@wp.func
def lookup_cell(sorted_unique_keys: wp.array[wp.int64], nk: wp.int64) -> wp.int32:
    if sorted_unique_keys.shape[0] == 0:
        return INVALID
    idx = kernel_array.binary_search_index(sorted_unique_keys, nk)
    if idx > wp.int32(0) and sorted_unique_keys[idx - wp.int32(1)] == nk:
        return idx - wp.int32(1)
    return INVALID


@wp.func
def grid_coord(point: wp.vec3, bbox_min: wp.vec3, inv_cell_size: wp.float32) -> wp.vec3i:
    p = point - bbox_min
    gx = wp.int32(p.x * inv_cell_size)
    gy = wp.int32(p.y * inv_cell_size)
    gz = wp.int32(p.z * inv_cell_size)
    return wp.vec3i(gx, gy, gz)


@wp.func
def point_cell_key(
    point: wp.vec3, bbox_min: wp.vec3, inv_cell_size: wp.float32, grid_w: wp.int32
) -> wp.int64:
    # The background grid's cell key of ``point``: its ``grid_coord``, packed at ``grid_w`` cells a
    # side. One map from the points, because ``grid_w`` is known before any coordinate is: every
    # step of ``grid_coord`` is monotone in the point, so the largest coordinate is the one at the
    # bounding box's upper corner, which the caller evaluates in the same float32 arithmetic.
    coord = grid_coord(point, bbox_min, inv_cell_size)
    w64 = wp.int64(grid_w)
    return cell_key(w64, wp.int64(coord.x), wp.int64(coord.y), wp.int64(coord.z))


@wp.kernel
def cell_run_starts(sorted_keys: wp.array[wp.int64], out_is_start: wp.array[wp.bool]) -> None:
    # Where each occupied cell's run begins in the cell-sorted pool. The keys arrive sorted, so the
    # distinct cells and their bounds are the run starts -- no hash table and no second sort, which
    # is what ``unique_1d(return_counts=True)`` followed by a counts scan cost for the same answer.
    s = wp.int32(wp.tid())
    out_is_start[s] = sorted_run_start(sorted_keys, s)


@wp.kernel
def cell_table(
    sorted_keys: wp.array[wp.int64],
    run_starts: wp.array[wp.int32],
    n_pool: wp.int32,
    out_unique_keys: wp.array[wp.int64],
    out_cell_offsets: wp.array[wp.int32],
) -> None:
    # The distinct cell keys and the sentinel-terminated cell bounds, in one pass over the
    # ``n_cells + 1`` run starts: slot ``n_cells`` is the terminator and owns no key.
    c = wp.int32(wp.tid())
    n_cells = run_starts.shape[0]
    if c < n_cells:
        start = run_starts[c]
        out_unique_keys[c] = sorted_keys[start]
        out_cell_offsets[c] = start
    else:
        out_cell_offsets[c] = n_pool


@wp.kernel
def dart_cell_neighbors(
    unique_keys: wp.array[wp.int64], grid_w: wp.int32, out_cell_neighbors: wp.array2d[wp.int32]
) -> None:
    # One-time table: ``out_cell_neighbors[c, s]`` is the compacted index of cell ``c``'s neighbour
    # at shell slot ``s``, or -1 when that neighbour is out of bounds or unoccupied. Built once (the
    # grid never changes), so the round kernels replace a binary search per cell with a table load.
    c, s = wp.tid()
    w64 = wp.int64(grid_w)
    key = unique_keys[c]
    x = wp.int32(key % w64)
    y = wp.int32((key // w64) % w64)
    z = wp.int32(key // (w64 * w64))
    dx = s % _DART_SHELL_W - wp.int32(1)
    dy = (s // _DART_SHELL_W) % _DART_SHELL_W - wp.int32(1)
    dz = s // (_DART_SHELL_W * _DART_SHELL_W) - wp.int32(1)
    out_cell_neighbors[c, s] = INVALID
    cx = x + dx
    cy = y + dy
    cz = z + dz
    if cx < wp.int32(0) or cx >= grid_w:
        return
    if cy < wp.int32(0) or cy >= grid_w:
        return
    if cz < wp.int32(0) or cz >= grid_w:
        return
    out_cell_neighbors[c, s] = lookup_cell(
        unique_keys, cell_key(w64, wp.int64(cx), wp.int64(cy), wp.int64(cz))
    )


@wp.func
def summarize_min_priority(
    priority: wp.array[wp.uint32],
    point_cell: wp.array[wp.int32],
    i: wp.int32,
    out_cell_min_priority: wp.array[wp.uint32],
) -> None:
    # Fold one alive point into its cell's minimum-priority summary: ``dart_compact_alive`` builds
    # every later round's summary over the survivors it is compacting, as ``dart_point_setup``
    # builds the first round's; ``wp.atomic_min`` is order-independent, so the two give the same
    # summary for the same alive set.
    wp.atomic_min(out_cell_min_priority, point_cell[i], priority[i])


@wp.kernel
def dart_point_setup(
    seed: wp.int32,
    sorted_keys: wp.array[wp.int64],
    unique_keys: wp.array[wp.int64],
    bucket: wp.array[wp.int32],
    out_point_cell: wp.array[wp.int32],
    out_priority: wp.array[wp.uint32],
    out_cell_min_priority: wp.array[wp.uint32],
) -> None:
    # Everything the dart loop keeps per pool point, written straight into the cell-sorted index
    # space it runs in, plus the first round's per-cell summary -- one launch over the pool:
    #
    # - the compacted cell index of every point (its cell is occupied by construction). Reading the
    #   *sorted* key rather than recomputing the point's key from its grid coordinate is what makes
    #   the per-point table and the permutation through ``bucket`` one pass: ``sorted_keys[s]``
    #   already is the key of pool point ``bucket[s]``;
    # - ``array.random_priorities`` drawn into sorted space: the draw is keyed on the point's
    #   *pool* index ``bucket[s]``, so each point holds exactly the priority the unsorted draw gave
    #   it;
    # - the selection sweep's per-cell summary, the smallest priority any *alive* point in the
    #   cell holds (``out_cell_min_priority``, arriving at ``DART_NO_PRIORITY``). The first round's
    #   alive list is the whole pool. ``dart_select_minima`` vetoes a candidate only from a strictly
    #   smaller priority, so a cell whose minimum already loses to the candidate's key cannot
    #   contribute and is skipped whole -- which is most of the 27, most rounds. Summarising the
    #   alive list rather than every not-COVERED point leaves the points ACCEPTED in an *earlier*
    #   round out, and that is safe: such a point covered its own ``r``-ball in the round it was
    #   accepted, so no point still alive now is within ``r`` of it and none of them could have
    #   been vetoed by it anyway. Every later round's summary is folded into the previous round's
    #   compaction, which already visits exactly the survivors.
    s = wp.int32(wp.tid())
    cell = lookup_cell(unique_keys, sorted_keys[s])
    out_point_cell[s] = cell
    priority = element_priority(seed, bucket[s])
    out_priority[s] = priority
    wp.atomic_min(out_cell_min_priority, cell, priority)


@wp.kernel
def dart_select_minima(
    pool_points: wp.array[wp.vec3],
    priority: wp.array[wp.uint32],
    point_cell: wp.array[wp.int32],
    cell_neighbors: wp.array2d[wp.int32],
    bucket: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    alive: wp.array[wp.int32],
    cell_min_priority: wp.array[wp.uint32],
    rr: wp.float32,
    out_state: wp.array[wp.int32],
    out_cell_accepted: wp.array[wp.bool],
) -> None:
    # Accept every alive point that no *smaller-priority* point still in play sits within ``r`` of.
    #
    # Ties in the drawn priority fall back to the pool index, so the order is total and the "smaller
    # of the two" argument that keeps two acceptances apart has no exception. The pass writes only
    # its own slot, and the only state it reads of others is "not yet covered" — which this pass
    # never sets — so it is race-free and its output depends only on the round's input state.
    #
    # **Every index in this kernel is a cell-sorted one.** ``sample._dart_throw_blue_noise``
    # permutes the pool through the very sort that built the cell list, so a cell's members are the
    # contiguous run ``[cell_offsets[c], cell_offsets[c + 1])`` and the loop variable ``k`` *is*
    # the point. That makes the three payload reads below stride 1 where they were a scatter into
    # the unsorted pool, and it takes ``bucket`` out of the hot path -- it is read only on the
    # priority-tie branch, which is what preserves the *original* pool index as the tie-break so
    # the accepted set is unchanged by the renumbering.
    t = wp.int32(wp.tid())
    i = alive[t]
    my_key = priority[i]
    p = pool_points[i]
    row = point_cell[i]
    for s in range(DART_SHELL_CELLS):
        c = cell_neighbors[row, s]
        if c < wp.int32(0):
            continue
        # Nothing in this cell can veto the candidate. The candidate's own cell never trips this,
        # since its own key is one of the minimands.
        if cell_min_priority[c] > my_key:
            continue
        for k in range(cell_offsets[c], cell_offsets[c + 1]):
            if k == i or out_state[k] == DART_COVERED:
                continue
            other = priority[k]
            if other > my_key or (other == my_key and bucket[k] > bucket[i]):
                continue
            if wp.length_sq(pool_points[k] - p) < rr:
                return
    out_state[i] = DART_ACCEPTED
    # Summary for the covering sweep that follows: this cell now holds a point accepted *this*
    # round. Every writer stores the same value, so the unsynchronized store is benign.
    out_cell_accepted[row] = True


@wp.kernel
def dart_cover_neighbors(
    pool_points: wp.array[wp.vec3],
    point_cell: wp.array[wp.int32],
    cell_neighbors: wp.array2d[wp.int32],
    cell_offsets: wp.array[wp.int32],
    alive: wp.array[wp.int32],
    cell_accepted: wp.array[wp.bool],
    rr: wp.float32,
    out_state: wp.array[wp.int32],
    out_flag: wp.array[wp.int32],
) -> None:
    # Retire every still-alive point within ``r`` of a point this round accepted. This is what makes
    # the result maximal — a survivor is a point no accepted point covers, so the loop cannot stop
    # while an uncovered gap of radius ``r`` remains.
    #
    # As above, only the thread's own slot is written, and a slot going ``ALIVE -> COVERED`` cannot
    # change another thread's ``== ACCEPTED`` test.
    #
    # ``out_flag`` is the round's 0/1 survivor flag over the *work list*, in the dtype
    # ``wp.utils.array_scan`` wants: this thread is the last writer of its point's state in the
    # round, so the flag can be written at each exit rather than by a separate pass re-reading the
    # state afterwards. The caller scans it **inclusively**, so the last entry is the survivor count
    # and no second read is needed to recover it -- the same one-tail-read shape
    # ``array.flatnonzero`` uses, and for the same reason: a host readback costs about as much as
    # the whole rest of a round.
    t = wp.int32(wp.tid())
    i = alive[t]
    if out_state[i] != DART_ALIVE:
        out_flag[t] = 0
        return
    p = pool_points[i]
    row = point_cell[i]
    for s in range(DART_SHELL_CELLS):
        c = cell_neighbors[row, s]
        if c < wp.int32(0):
            continue
        # Cells with nothing accepted this round are skipped whole. Restricting the sweep to *this*
        # round's acceptances loses nothing: a point accepted earlier covered its ``r``-ball in that
        # same round, and this thread was already alive then, so it would not still be alive now.
        if not cell_accepted[c]:
            continue
        # Cell-sorted indices throughout, as in ``dart_select_minima``: ``k`` is the point, so
        # both reads are stride 1 and ``bucket`` is not needed here at all (no tie-break).
        for k in range(cell_offsets[c], cell_offsets[c + 1]):
            if out_state[k] != DART_ACCEPTED:
                continue
            if wp.length_sq(pool_points[k] - p) < rr:
                out_state[i] = DART_COVERED
                out_flag[t] = 0
                return
    out_flag[t] = 1


@wp.kernel
def dart_compact_alive(
    alive: wp.array[wp.int32],
    state: wp.array[wp.int32],
    positions: wp.array[wp.int32],
    priority: wp.array[wp.uint32],
    point_cell: wp.array[wp.int32],
    out_next: wp.array[wp.int32],
    out_cell_min_priority: wp.array[wp.uint32],
) -> None:
    # Next round's work list, from this round's: survivors keep their relative order, so the loop
    # walks a shrinking prefix instead of the whole pool. ``positions`` is the **inclusive** scan of
    # the survivor flags, so a survivor's slot is ``positions[t] - 1``; the caller reads the same
    # array's last entry as the round's survivor count, which is what makes one readback do the
    # work of two.
    #
    # The survivors are exactly the next round's alive set, so this is also where that round's
    # per-cell priority summary is built (the caller refills it to ``DART_NO_PRIORITY`` first):
    # one pass over this round's list instead of a second launch over the next one's.
    t = wp.int32(wp.tid())
    i = alive[t]
    if state[i] == DART_ALIVE:
        out_next[positions[t] - 1] = i
        summarize_min_priority(priority, point_cell, i, out_cell_min_priority)


@wp.kernel
def dart_accepted_pool_mask(
    state: wp.array[wp.int32], bucket: wp.array[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    # The accepted set as a pool-order mask, from the cell-sorted ``state``: ``bucket`` maps a
    # sorted position back to its pool index, so the scatter *is* the inverse permutation and no
    # inverse table or gather is built. The caller allocates ``out_mask`` zeroed.
    s = wp.int32(wp.tid())
    if state[s] == DART_ACCEPTED:
        out_mask[bucket[s]] = True
