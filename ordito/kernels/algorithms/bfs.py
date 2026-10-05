"""
Per-source breadth-first collection over a CSR adjacency graph.

``per_source_bfs_collect`` -- one thread per source over caller-provided global-memory scratch rows
(a FIFO queue whose emitted prefix is the discovery order, an open-addressing visited hash set, and
a small nearest-fallback pool). With a finite ``radius`` it enqueues only neighbors within
``radius`` of the center (and backfills the nearest out-of-ball vertices up to ``min_count``) -- the
geodesic-ball query behind [`ordito.neighbors.geodesic_ball`][ordito.neighbors.geodesic_ball].
Its nearest-fallback pool scans with the shared ``wp.ref`` argmin/argmax helpers, so any kernel
calling it must be decorated ``@wp.kernel(enable_backward=False)`` (see ``ordito.kernels.array``).

**This module used to carry two whole-graph engines besides**: a level-synchronous frontier loop
that reproduced ``scipy.sparse.csgraph.breadth_first_order``'s FIFO order without a sort, and a
one-thread serial drain it handed off to once the frontier went narrow. Both existed for a public
``graph.bfs``, whose only in-tree consumer was the homology decomposition -- which reads a spanning
tree's ``parents`` and depths and never a discovery order. Reproducing scipy's order cost a tiled
block scan and a serial advance per level, more than a third of the level, so the decomposition now
carries its own two-kernel level loop in ``ordito.kernels.homology`` and the order-exact engines
are gone. Anything needing scipy's exact visit order should call scipy.
"""

import warp as wp

from ordito.kernels import grouping as kernel_grouping
from ordito.kernels.array import update_argmax, update_argmin

wp.set_module_options({"enable_backward": False})

# Per-source scratch capacities (rows of the wrapper-allocated global-memory pools).
# ``PER_SOURCE_MAX_NEIGHBORS`` caps the queue — and therefore the collected set — as before;
# the visited hash row is power-of-two sized with a 3/4 load-factor fill bound; the extras pool
# only needs to hold the nearest out-of-ball frontier for the ``min_count`` backfill.
PER_SOURCE_MAX_NEIGHBORS = 512
VISITED_HASH_CAPACITY = 1024
_VISITED_MAX_FILL = 768
EXTRAS_CAPACITY = 64


@wp.func
def bfs_visited_insert(
    visited: wp.array[wp.int32], mask: wp.int32, value: wp.int32, count: wp.int32
) -> tuple[wp.bool, wp.int32]:
    """
    Insert ``value`` into the open-addressing ``visited`` row (empty slots hold ``-1``).

    Returns ``(is_new, new_count)``: ``is_new`` is ``False`` when the value was already present.
    Beyond the load-factor fill bound the insert is dropped and the value reads as new, so dropped
    nodes can be revisited; the drop returns ``new_count == _VISITED_MAX_FILL + 1``, which the bound
    test still reads as full, so a caller learns whether anything was dropped from the final count
    alone rather than comparing counts around every insert.
    """
    slot = kernel_grouping.hash_slot(value, mask)
    while True:
        stored = visited[slot]
        if stored == value:
            return False, count
        if stored == wp.int32(-1):
            if count >= _VISITED_MAX_FILL:
                return True, _VISITED_MAX_FILL + wp.int32(1)
            visited[slot] = value
            return True, count + 1
        slot = kernel_grouping.next_slot(slot, mask)


@wp.func
def bfs_extras_push_nearest(
    ext_dist: wp.array[wp.float32],
    ext_idx: wp.array[wp.int32],
    distance: wp.float32,
    neighbor: wp.int32,
    count: wp.int32,
) -> wp.int32:
    """Keep the ``cap`` nearest candidates: append, or replace the farthest kept one."""
    cap = ext_dist.shape[0]
    if count < cap:
        ext_dist[count] = distance
        ext_idx[count] = neighbor
        return count + 1
    farthest = wp.int32(0)
    farthest_distance = ext_dist[0]
    for k in range(1, cap):
        update_argmax(farthest_distance, farthest, ext_dist[k], k)
    if distance < farthest_distance:
        ext_dist[farthest] = distance
        ext_idx[farthest] = neighbor
    return count


@wp.func
def bfs_extras_pop_nearest(
    ext_dist: wp.array[wp.float32], ext_idx: wp.array[wp.int32], count: wp.int32
) -> tuple[wp.int32, wp.int32]:
    """Remove and return the nearest candidate (swap-remove); caller ensures ``count > 0``."""
    best = wp.int32(0)
    best_distance = ext_dist[0]
    for k in range(1, count):
        update_argmin(best_distance, best, ext_dist[k], k)
    nearest = ext_idx[best]
    last = count - 1
    ext_dist[best] = ext_dist[last]
    ext_idx[best] = ext_idx[last]
    return nearest, last


@wp.func
def bfs_center_distance(
    vertices: wp.array[wp.vec3], use_geometry: wp.bool, center: wp.vec3, neighbor: wp.int32
) -> wp.float32:
    # A newly visited neighbour's Euclidean distance to the source, the geodesic ball's predicate
    # and the nearest fallback's key; ``0`` when the traversal is purely topological
    # (``use_geometry`` false), so ``vertices`` is never read. One definition for the BFS phase and
    # the fallback drain of ``per_source_bfs_collect``.
    distance = wp.float32(0.0)
    if use_geometry:
        distance = wp.length(vertices[neighbor] - center)
    return distance


@wp.func
def per_source_bfs_collect(
    i: wp.int32,
    vertices: wp.array[wp.vec3],
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    radius: wp.float32,
    min_count: wp.int32,
    queue: wp.array[wp.int32],
    visited: wp.array[wp.int32],
    ext_dist: wp.array[wp.float32],
    ext_idx: wp.array[wp.int32],
    out_overflow: wp.array[wp.int32],
) -> wp.int32:
    """
    BFS over the CSR edge graph from source ``i``; returns the collected count.

    Traverses ``adj_offsets``/``adj_columns`` with a FIFO ``queue`` and an O(1) open-addressing
    ``visited`` hash row — power-of-two length, pre-filled with ``-1`` by the caller before the
    launch. When ``radius`` is finite the traversal is *geodesic* (libigl ``getSphere``): a
    neighbor is enqueued only when within Euclidean ``radius`` of the center, and out-of-ball
    neighbors feed a nearest fallback (``ext_dist``/``ext_idx``) drained to ``min_count``. When
    ``radius`` is ``+inf`` the geometric predicate is disabled (``vertices`` is never read, so a
    length-1 placeholder is fine); pass ``min_count = 0`` so the fallback never engages.

    On return the collected set *is* ``queue[:count]`` in BFS-then-backfill order (drained
    fallback candidates are appended to the queue), so ``count == q_tail <= queue capacity``
    always and the caller gathers results straight from its queue row — no second traversal.
    Exceeding the queue or visited capacity drops the surplus, and a source that dropped anything
    adds exactly one to ``out_overflow``, so the counter is the number of clipped sources.
    """
    use_geometry = not wp.isinf(radius)

    queue_cap = queue.shape[0]
    mask = visited.shape[0] - 1

    center = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_geometry:
        center = vertices[i]

    visited_n = wp.int32(0)
    is_new = wp.bool(True)
    is_new, visited_n = bfs_visited_insert(visited, mask, i, visited_n)
    queue[0] = i
    q_head = wp.int32(0)
    q_tail = wp.int32(1)
    ext_n = wp.int32(0)
    collected = wp.int32(0)
    # Once an in-ball neighbour finds the queue full, nothing more can be enqueued: the remaining
    # dequeues would only count further drops, and -- with ``min_count`` within the capacity -- the
    # nearest fallback can no longer engage, since every queued vertex counts as collected. So the
    # traversal stops there with the identical queue. That is why the overflow counter counts
    # clipped *sources* (``clipped``, committed once at the end): the number of dropped neighbours
    # is only known by walking the tail this skips, which on a radius several times the cap's reach
    # was nearly all of the walk.
    full = wp.bool(False)
    clipped = wp.bool(False)

    while q_head < q_tail and not full:
        current = queue[q_head]
        q_head += wp.int32(1)
        collected += wp.int32(1)

        start = adj_offsets[current]
        end = adj_offsets[current + 1]
        for k in range(start, end):
            neighbor = adj_columns[k]
            is_new, visited_n = bfs_visited_insert(visited, mask, neighbor, visited_n)
            if not is_new:
                continue
            distance = bfs_center_distance(vertices, use_geometry, center, neighbor)
            if distance < radius:
                if q_tail < queue_cap:
                    queue[q_tail] = neighbor
                    q_tail += wp.int32(1)
                else:
                    clipped = wp.bool(True)
                    if min_count <= queue_cap:
                        full = wp.bool(True)
                        break
            elif collected < min_count:
                ext_n = bfs_extras_push_nearest(ext_dist, ext_idx, distance, neighbor, ext_n)
    if full:
        # What the skipped dequeues would have counted.
        collected = q_tail

    # Drained candidates are appended to the queue (not re-expanded from it) so the queue prefix
    # stays the complete collected set. The main loop exits with collected == q_tail, so with the
    # default min_count << queue capacity the drain never sees a full queue; only a pathological
    # min_count > capacity can overflow here, dropping the surplus like the main phase does.
    while ext_n > wp.int32(0) and collected < min_count:
        cand = wp.int32(0)
        cand, ext_n = bfs_extras_pop_nearest(ext_dist, ext_idx, ext_n)

        if q_tail < queue_cap:
            queue[q_tail] = cand
            q_tail += wp.int32(1)
            collected += wp.int32(1)
        else:
            clipped = wp.bool(True)
            continue

        start = adj_offsets[cand]
        end = adj_offsets[cand + 1]
        for k in range(start, end):
            neighbor = adj_columns[k]
            is_new, visited_n = bfs_visited_insert(visited, mask, neighbor, visited_n)
            if not is_new:
                continue
            distance = bfs_center_distance(vertices, use_geometry, center, neighbor)
            ext_n = bfs_extras_push_nearest(ext_dist, ext_idx, distance, neighbor, ext_n)

    # A visited-table drop leaves ``visited_n`` one past the fill bound (``bfs_visited_insert``).
    if clipped or visited_n > _VISITED_MAX_FILL:
        wp.atomic_add(out_overflow, 0, 1)
    return collected
