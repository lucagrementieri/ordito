from typing import Any

import warp as wp

from ordito.kernels.array import (
    OverloadTable,
    append_slot,
    atomic_min_packed_box,
    binary_search_index_left,
    declare_map_signatures,
    device_is_cpu,
    map_probe,
    mark_at,
    popcount32,
    scanned_count,
    trilinear_cell,
    trilinear_corner,
    trilinear_weight,
    unpack_edge_key,
)
from ordito.kernels.reduce import (
    GroupSumPass,
    add_group_sum_term,
    block_sum,
    group_sum_field,
    group_sum_group_field,
    group_sum_term,
    raise_group_sum_field,
    warp_lane_leads,
    warp_lanes_sum_float,
    warp_peer_lanes,
    warp_uniform_lanes,
)
from ordito.kernels.triangles import corner_triple, face_normals_and_area

wp.set_module_options({"enable_backward": False})


# The one per-group scatter, whatever the deterministic setting: ``out_sums[group] += value``. The
# mode is the buffer ``sums`` holds (``_launch.group_sums``), as ``reduce.store_block_sum`` reads
# its mode off its buffer's size: with no words it is the float ``atomic_add`` into the caller's
# accumulator ``out_sums``, in arrival order; with them it is pass ``sums.phase`` of the exact
# group sum (``kernels/reduce.py``), and ``out_sums`` is not touched until the finish. So a kernel
# has one body calling it, launched once with the setting off and once per pass with it on
# (``_launch.launch_group_sums``), and the finish writes ``out_sums``
# (``_launch.finish_group_sums``). A kernel that calls it takes the ``GroupSumPass`` and then
# ``out_sums``, after its inputs.
#
# ``scatter_add_vector`` and ``scatter_add_matrix`` are the componentwise siblings: component ``c``
# of group ``g`` is slot ``g * n_components + c`` (row-major for a matrix), and the group's
# components share one field, raised by their largest.
#
# With the setting off, the lanes of a warp adding to one group sum their values first
# (``kernels/reduce.warp_lanes_sum``, in ``float64`` for either precision, per component) and the
# lowest commits once; any other lane takes the plain atomic, and the CPU device always does.
# ``pooling`` names the lanes that pool, a constant at each site or a launch argument:
# ``ARRIVAL_POOL_UNIFORM`` only a warp whose active lanes all share the group
# (``reduce.warp_uniform_lanes``), ``ARRIVAL_POOL_PEERS`` every set of lanes sharing one
# (``reduce.warp_peer_lanes``, a ``__match_any_sync``). One group of 10 M terms ran 0.04x the plain
# atomic under either, ``remove_small_components`` and ``make_volume(multibody=True)`` 0.22-0.27x
# on 1 and 64 components (5 M and 21 M faces), at par with every face its own component and on
# 1 M random groups. Declined: a ``float32`` sum for ``float32`` values (0.94-1.00x; a generic
# func_native cannot return its argument's type, so it would be a second, same-named native).
#
# Pooling peers pays where the lanes sharing a group in a warp share an address hot across the
# launch, and costs where they share a cold one (device time peers/uniform, 10 M terms): 8 random
# groups 0.20x, 64 random groups 0.85-0.88x, every other term into one group 0.08-0.12x,
# ``fan_hub`` corners into vertices 0.23x and its rounding rows 0.15x, the component and region
# areas of dragon 0.86-0.87x; but sorted runs of 4 and 16 terms 2.1x and 1.3x (``float32``),
# ``segment_sum`` 1.09-1.39x, dragon's corners into vertices 1.38x and its rounding rows 1.11x
# (lucy 1.07x), the Poisson octree splats 1.10-1.74x; random keys over many groups at par. So
# repair's component sums and selection's region areas take peers and the other sites keep the
# uniform test, each by its keys' structure; ``reduce.scatter_sum``, whose keys are the caller's,
# chooses by ``scatter_sum_pooling``.
ARRIVAL_POOL_UNIFORM = wp.int32(0)
ARRIVAL_POOL_PEERS = wp.int32(1)

# The mean terms a group from which ``scatter_sum_pooling`` pools peers. Mesh size does not predict
# the gain (1 M and 10 M terms agree; dragon and lucy part by face order), the keys' structure
# does, and the mean is the one free signal of it: few groups are hot whatever their order. Peers
# against uniform, device time at 1 M and 10 M terms: random keys 0.98-1.02x up to 1 024 terms a
# group, 0.95-0.99x at 4 096, 0.84-0.98x at 16 384; interleaved runs of 4 and 16 terms (the cold
# peers that cost) 0.36-1.27x at 1 024, 0.19-1.03x at 2 048 and 0.12-0.75x at 4 096. It misses a
# hub among many groups (every other term into one group, two terms a group on average), which
# only a histogram pass could see and which would cost more than it saves.
SCATTER_POOL_PEERS_TERMS = 2048


def scatter_sum_pooling(n_terms: int, n_groups: int) -> int:
    """Return the ``pooling`` for ``n_terms`` arrival-order terms over ``n_groups`` caller keys."""
    if n_terms >= SCATTER_POOL_PEERS_TERMS * max(n_groups, 1):
        return int(ARRIVAL_POOL_PEERS)
    return int(ARRIVAL_POOL_UNIFORM)


# The off-mode commit, ``out_sums[index] += value`` without a negative index's wrap, which costs
# the CPU device a load of the array's length per term (every caller's index lies in
# ``[0, n_groups)``, ``reduce.scatter_sum`` validates its own).
_ATOMIC_ADD_AT = """
wp::atomic_add(wp::data_at_byte_offset(out, wp::byte_offset(out, index)), value);
"""


@wp.func_native(_ATOMIC_ADD_AT)
def atomic_add_at(out: wp.array[Any], index: wp.int32, value: Any): ...


@wp.func
def arrival_order_lanes(group: wp.int32, pooling: wp.int32) -> wp.int32:
    # The lanes that pool their values into ``group`` under ``pooling``, 0 for none.
    if pooling == ARRIVAL_POOL_PEERS:
        return warp_peer_lanes(group)
    return warp_uniform_lanes(group)


@wp.func
def arrival_order_add(out_sums: wp.array[Any], group: wp.int32, value: wp.Float, pooling: wp.int32):
    # The off-mode commit of a scalar: pre-summed across the lanes sharing ``group``, else per lane.
    lanes = arrival_order_lanes(group, pooling)
    if lanes == 0:
        atomic_add_at(out_sums, group, value)
    else:
        total = type(value)(warp_lanes_sum_float(lanes, wp.float64(value)))
        if warp_lane_leads(lanes) != 0:
            atomic_add_at(out_sums, group, total)


@wp.func
def arrival_order_add_vector(
    out_sums: wp.array[Any], group: wp.int32, value: Any, pooling: wp.int32
):
    # ``arrival_order_add`` for a vector, componentwise.
    lanes = arrival_order_lanes(group, pooling)
    if lanes == 0:
        atomic_add_at(out_sums, group, value)
    else:
        total = type(value)()
        for c in range(len(value)):
            total[c] = type(value[c])(warp_lanes_sum_float(lanes, wp.float64(value[c])))
        if warp_lane_leads(lanes) != 0:
            atomic_add_at(out_sums, group, total)


@wp.func
def group_sum_vector_term(sums: GroupSumPass, group: wp.int32, value: Any):
    # ``reduce.group_sum_term`` for a vector: its components' bound (pass 0) or words (pass 1).
    n = len(value)
    if sums.phase == 0:
        field = wp.int32(0)
        for c in range(n):
            field = wp.max(field, group_sum_field(wp.float64(value[c])))
        raise_group_sum_field(sums, group, field)
    else:
        field = group_sum_group_field(sums, group)
        for c in range(n):
            add_group_sum_term(sums, group, group * n + c, field, wp.float64(value[c]))


@wp.func
def arrival_order_add_matrix(
    out_sums: wp.array[Any], group: wp.int32, value: Any, pooling: wp.int32
):
    # ``arrival_order_add`` for a matrix, componentwise (``arrival_order_add_vector``'s twin).
    lanes = arrival_order_lanes(group, pooling)
    if lanes == 0:
        atomic_add_at(out_sums, group, value)
    else:
        total = type(value)()
        for r in range(len(value)):
            for c in range(len(value[0])):
                total[r, c] = type(value[r, c])(
                    warp_lanes_sum_float(lanes, wp.float64(value[r, c]))
                )
        if warp_lane_leads(lanes) != 0:
            atomic_add_at(out_sums, group, total)


@wp.func
def group_sum_matrix_term(sums: GroupSumPass, group: wp.int32, value: Any):
    # ``reduce.group_sum_term`` for a matrix: its components' bound (pass 0) or words (pass 1), in
    # row-major slots (``group_sum_vector_term``'s twin).
    rows = len(value)
    cols = len(value[0])
    if sums.phase == 0:
        field = wp.int32(0)
        for r in range(rows):
            for c in range(cols):
                field = wp.max(field, group_sum_field(wp.float64(value[r, c])))
        raise_group_sum_field(sums, group, field)
    else:
        field = group_sum_group_field(sums, group)
        for r in range(rows):
            for c in range(cols):
                slot = (group * rows + r) * cols + c
                add_group_sum_term(sums, group, slot, field, wp.float64(value[r, c]))


@wp.func
def scatter_add(
    out_sums: wp.array[Any], sums: GroupSumPass, group: wp.int32, value: wp.Float, pooling: wp.int32
):
    # ``out_sums[group] += value`` for a scalar.
    if sums.words.shape[0] == 0:
        arrival_order_add(out_sums, group, value, pooling)
    else:
        group_sum_term(sums, group, wp.float64(value))


@wp.func
def scatter_add_vector(
    out_sums: wp.array[Any], sums: GroupSumPass, group: wp.int32, value: Any, pooling: wp.int32
):
    # ``out_sums[group] += value`` for a vector, componentwise.
    if sums.words.shape[0] == 0:
        arrival_order_add_vector(out_sums, group, value, pooling)
    else:
        group_sum_vector_term(sums, group, value)


@wp.func
def scatter_add_matrix(
    out_sums: wp.array[Any], sums: GroupSumPass, group: wp.int32, value: Any, pooling: wp.int32
):
    # ``out_sums[group] += value`` for a matrix, componentwise.
    if sums.words.shape[0] == 0:
        arrival_order_add_matrix(out_sums, group, value, pooling)
    else:
        group_sum_matrix_term(sums, group, value)


# ``scatter_add`` and ``scatter_add_vector`` for a kernel whose body is little more than the scatter
# (``scatter_sum_*``, ``segment_sum_*``): on the CPU device the on-mode terms run out of line
# (``device_is_cpu`` folds, so CUDA inlines them as the plain forms do). Inlined, a term makes so
# small a kernel too large for the CPU compiler to inline into its per-thread loop, and the off mode
# then pays a call per thread: 1.3-1.6x a plain atomic per term, against 1.00-1.06x out of line.
# Not the plain forms: a kernel with work of its own is never inlined into that loop, and there the
# out-of-line call (its spilled registers) cost the on mode 1.26-1.40x (``robust_laplacian``'s
# rounding rows, six calls a face).
@wp.func(inline=False)
def group_sum_term_out_of_line(sums: GroupSumPass, group: wp.int32, value: wp.float64):
    group_sum_term(sums, group, value)


@wp.func(inline=False)
def group_sum_vector_term_out_of_line(sums: GroupSumPass, group: wp.int32, value: Any):
    group_sum_vector_term(sums, group, value)


@wp.func
def sole_scatter_add(
    out_sums: wp.array[Any], sums: GroupSumPass, group: wp.int32, value: wp.Float, pooling: wp.int32
):
    # ``scatter_add`` as a kernel's whole work.
    if sums.words.shape[0] == 0:
        arrival_order_add(out_sums, group, value, pooling)
    elif device_is_cpu() != 0:
        group_sum_term_out_of_line(sums, group, wp.float64(value))
    else:
        group_sum_term(sums, group, wp.float64(value))


@wp.func
def sole_scatter_add_vector(
    out_sums: wp.array[Any], sums: GroupSumPass, group: wp.int32, value: Any, pooling: wp.int32
):
    # ``scatter_add_vector`` as a kernel's whole work.
    if sums.words.shape[0] == 0:
        arrival_order_add_vector(out_sums, group, value, pooling)
    elif device_is_cpu() != 0:
        group_sum_vector_term_out_of_line(sums, group, value)
    else:
        group_sum_vector_term(sums, group, value)


@wp.kernel
def scatter_sum_scalars(
    values: wp.array[wp.Float],
    indices: wp.array[wp.int32],
    pooling: wp.int32,
    sums: GroupSumPass,
    out_sums: wp.array[wp.Float],
) -> None:
    # ``reduce.scatter_sum`` of scalars: each value into its group, its lanes pooled as
    # ``pooling`` names (``arrival_order_add``). Also the second pass of a producer that stores its
    # values in the bound pass rather than computing them twice.
    i = wp.int32(wp.tid())
    sole_scatter_add(out_sums, sums, indices[i], values[i], pooling)


@wp.kernel
def scatter_sum_vectors(
    values: wp.array[Any],
    indices: wp.array[wp.int32],
    pooling: wp.int32,
    sums: GroupSumPass,
    out_sums: wp.array[Any],
) -> None:
    # ``scatter_sum_scalars`` for vectors.
    i = wp.int32(wp.tid())
    sole_scatter_add_vector(out_sums, sums, indices[i], values[i], pooling)


@wp.func
def segment_of(offsets: wp.array[wp.int32], i: wp.int32) -> wp.int32:
    # The segment holding item ``i`` of a CSR packing: the last ``s`` with ``offsets[s] <= i``,
    # which skips empty segments (``n + 1`` total-terminated offsets, ``i < offsets[n]``).
    lo = wp.int32(0)
    hi = offsets.shape[0] - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if offsets[mid] <= i:
            lo = mid
        else:
            hi = mid
    return lo


@wp.kernel
def segment_sum_scalars(
    values: wp.array[wp.Float],
    offsets: wp.array[wp.int32],
    sums: GroupSumPass,
    out_sums: wp.array[wp.Float],
) -> None:
    # ``reduce.segment_sum`` of scalars: one thread per item, into its segment. Consecutive items
    # share a segment, so a long segment's warps commit once each (``reduce.warp_lanes_sum``); a
    # short segment's lanes are peers on a cold address, which pooling peers would cost.
    i = wp.int32(wp.tid())
    sole_scatter_add(out_sums, sums, segment_of(offsets, i), values[i], ARRIVAL_POOL_UNIFORM)


@wp.kernel
def segment_sum_vectors(
    values: wp.array[Any], offsets: wp.array[wp.int32], sums: GroupSumPass, out_sums: wp.array[Any]
) -> None:
    # ``segment_sum_scalars`` for vectors.
    i = wp.int32(wp.tid())
    sole_scatter_add_vector(out_sums, sums, segment_of(offsets, i), values[i], ARRIVAL_POOL_UNIFORM)


# ``reduce.scatter_min`` / ``scatter_max`` and their segment forms: ``maximum`` (warp-uniform)
# selects the extremum. The two kernels differ only in how an item finds its group. A minimum is
# exact in any order, so unlike a sum the arrival order moves no bits: these need no deterministic
# form. A vector commits componentwise (one atomic a component).
EXTREMUM_MIN = wp.int32(0)
EXTREMUM_MAX = wp.int32(1)


@wp.func
def commit_extremum(out_extrema: wp.array[Any], group: wp.int32, value: Any, maximum: wp.int32):
    # ``out_extrema[group]`` lowered (``EXTREMUM_MIN``) or raised (``EXTREMUM_MAX``) to ``value``.
    if maximum == EXTREMUM_MAX:
        wp.atomic_max(out_extrema, group, value)
    else:
        wp.atomic_min(out_extrema, group, value)


@wp.kernel
def scatter_extrema(
    values: wp.array[Any],
    indices: wp.array[wp.int32],
    maximum: wp.int32,
    out_extrema: wp.array[Any],
) -> None:
    # Each value into its group's extremum.
    i = wp.int32(wp.tid())
    commit_extremum(out_extrema, indices[i], values[i], maximum)


@wp.kernel
def segment_extrema(
    values: wp.array[Any],
    offsets: wp.array[wp.int32],
    maximum: wp.int32,
    out_extrema: wp.array[Any],
) -> None:
    # ``scatter_extrema`` with each item's group its segment (``segment_of``).
    i = wp.int32(wp.tid())
    commit_extremum(out_extrema, segment_of(offsets, i), values[i], maximum)


@wp.func
def scalar_mean_of_sum(total: wp.Float, count: wp.int32):
    # ``reduce.scatter_mean`` / ``segment_mean``'s division, mapped over the groups: ``0 / 0``, a
    # NaN, for an empty group.
    return total / type(total)(count)


@wp.func
def vector_mean_of_sum(total: Any, count: wp.int32):
    # ``scalar_mean_of_sum`` for a vector, componentwise.
    return total / type(total[0])(count)


@wp.func
def atomic_add_vec3(out_sum: wp.array2d[wp.Float], row: wp.int32, v: wp.vec3) -> None:
    # Component-wise atomic accumulation of a wp.vec3 into row ``row`` of a (n, 3) buffer.
    #
    # ``out_sum.dtype(...)`` is the conversion, and it is required rather than cosmetic: Warp does
    # **not** promote a float32 value into a float64 accumulator, and ``wp.atomic_add`` with the
    # two mismatched fails at kernel-parse time (probed on Warp 1.19). Reading the array's dtype
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
    # ``normals.vertex_normals`` end to end wins on a small mesh -- the fused narrow-and-normalize
    # tail removes a launch, since a float32 accumulator could reach ``wp.vec3`` through a zero-copy
    # ``array_cast`` and a float64 one cannot -- and loses slightly on a large one, where the wider
    # atomics start to show. The (n, 3) buffer doubles.
    wp.atomic_add(out_sum, row, 0, out_sum.dtype(v[0]))
    wp.atomic_add(out_sum, row, 1, out_sum.dtype(v[1]))
    wp.atomic_add(out_sum, row, 2, out_sum.dtype(v[2]))


# ``wp.atomic_add`` and ``atomic_add_vec3`` for a per-vertex scatter: the lanes of a warp adding to
# one slot (``reduce.warp_peer_lanes``) sum their values in ``float64`` across the warp and the
# lowest commits once; a lane with no peer, and every lane on the CPU device, takes the plain
# atomic, so CPU bits are unchanged. A sum of ``float32`` values in ``float64`` stays exact, so
# ``scatter_sum_exact`` keeps its order-free result. The hub of a fan sits in every other face
# (trimesh's cone), where ``arrival_order_add``'s warp-*uniform* test never fires.
#
# Device time against the plain atomics (``fan_hub``: two hubs of valence 40 960): the ``float64``
# vector scatter 0.09x on the hub, 0.79-0.85x on dragon, happy_buddha and lucy (consecutive faces
# share a slot's vertex in pairs there), 0.84-1.03x on sphere and bunny; the exact ``float64``
# scalar 0.10x on the hub and 0.85-0.93x large. The ``float32`` scalar takes 0.10x on the hub and
# 0.87x on lucy but 1.2x on dragon and happy_buddha, where the match costs more than the
# cheap atomics it saves; pooling only four or more lanes lost lucy's gain and still cost 1.1x
# there, and ``float32`` shuffles changed nothing, so one rule serves all four. The match is not
# ``kernels/reduce``'s declined one: that pooled the exact group sum's two integer words.
@wp.func
def peer_summed_add(out_sum: wp.array[Any], index: wp.int32, value: wp.Float):
    lanes = warp_peer_lanes(index)
    if lanes == 0:
        wp.atomic_add(out_sum, index, value)
    else:
        total = type(value)(warp_lanes_sum_float(lanes, wp.float64(value)))
        if warp_lane_leads(lanes) != 0:
            wp.atomic_add(out_sum, index, total)


@wp.func
def peer_counted_add(out_counts: wp.array[wp.int32], index: wp.int32) -> None:
    # ``wp.atomic_add(out_counts, index, 1)`` with the lanes of a warp counting into one bin
    # (``reduce.warp_peer_lanes``) committed once, by the lowest, as their number; a lane with no
    # peer, and every lane on the CPU device, takes the plain atomic. A count is exact in any
    # order. For a histogram whose keys crowd a few bins: the Poisson-disk weight floor's
    # 2 048-bin digits over a 1 M pool cost the round more than the sort they replaced unpooled.
    lanes = warp_peer_lanes(index)
    if lanes == 0:
        wp.atomic_add(out_counts, index, 1)
    elif warp_lane_leads(lanes) != 0:
        wp.atomic_add(out_counts, index, popcount32(wp.uint32(lanes)))


@wp.func
def peer_summed_add_vec3(out_sum: wp.array2d[wp.Float], row: wp.int32, v: wp.vec3) -> None:
    # ``peer_summed_add`` into row ``row`` of an ``(n, 3)`` accumulator, as ``atomic_add_vec3``.
    lanes = warp_peer_lanes(row)
    if lanes == 0:
        atomic_add_vec3(out_sum, row, v)
    else:
        x = warp_lanes_sum_float(lanes, wp.float64(v[0]))
        y = warp_lanes_sum_float(lanes, wp.float64(v[1]))
        z = warp_lanes_sum_float(lanes, wp.float64(v[2]))
        if warp_lane_leads(lanes) != 0:
            wp.atomic_add(out_sum, row, 0, out_sum.dtype(x))
            wp.atomic_add(out_sum, row, 1, out_sum.dtype(y))
            wp.atomic_add(out_sum, row, 2, out_sum.dtype(z))


@wp.kernel
def count_occurrences(indices: wp.array[wp.int32], out_counts: wp.array[wp.int32]) -> None:
    # Histogram of ``indices``: one atomic increment per entry. Launch over ``indices.shape[0]``.
    # Sizing a CSR is what every caller wants it for -- incident faces per vertex (a flat face
    # buffer *is* the corner -> vertex map), face-corners per unique edge, outgoing halfedges per
    # vertex -- so the counts feed a scan and the caller allocates ``out_counts`` zeroed.
    tid = wp.int32(wp.tid())
    wp.atomic_add(out_counts, indices[tid], 1)


@wp.func
def claim_row_slot(out_offsets: wp.array[wp.int32], row: wp.int32) -> wp.int32:
    # The next free slot of CSR row ``row``, in a counting-sort fill with no cursor buffer:
    # ``out_offsets[row + 1]`` holds the row's start (the caller counted the rows into
    # ``out_offsets[2:]`` of an ``(n_rows + 2,)`` buffer whose first two entries are zero and
    # scanned them there in place, or scanned counts it holds exclusively into ``out_offsets[1:]``),
    # and each claim advances it by one; a full row has advanced it to its end, row ``row + 1``'s
    # start. So ``out_offsets[: n_rows + 1]`` ends as the CSR offsets, its leading entry never
    # claimed, and the slots go to the claims in arrival order, from the front.
    return wp.atomic_add(out_offsets, row + 1, 1)


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


# A face's three corner values added to the slots ``indices`` names for its corners (vertices, or
# edges through a halfedge map). ``add_corner_triple`` commits a plain atomic per corner;
# ``peer_summed_add_corner_triple`` pools a warp's lanes that share a slot (``peer_summed_add``).
# The pooled form serves the kernels whose work is the scatter (``scatter_face_areas_exact``,
# ``scatter_face_thirds``, ``laplacian.add_face_area_exact``, ``curvature.scatter_corner_angles``):
# device time new/old 0.09-0.10x on ``fan_hub``, 0.83-0.85x on lucy, 0.91-0.95x on dragon but
# 1.11-1.23x for its ``float32`` thirds and angles, and +0.5-1 us on 80 k-face meshes. Declined in
# the kernels with work of their own, where the match's registers cost more than the atomics it
# saves on scan meshes (dragon new/old): ``heat`` divergences and face terms 1.04-1.18x, the
# cover's 1.10-1.17x, ``energies.hessian_face_terms`` 1.03x and ``internal_angles_and_sums``
# 1.05x, ``remesh``'s area sum 1.02x (1.18-1.27x small), ``heat``'s ``vec3`` unit-gradient scatter
# 1.29x; and on edge slots (``crouzeix_raviart_mass_diag`` 1.04-1.16x, ``edge_cotan_add``
# 0.97-1.26x), which two consecutive faces rarely share inside a warp. Lucy took 0.90x on the
# heat face terms; the hub's gain there is moot, since its solves run for a minute.
@wp.func
def add_corner_triple(
    out_sum: wp.array[Any], indices: wp.array[wp.int32], row: wp.int32, a: Any, b: Any, c: Any
) -> None:
    i0, i1, i2 = corner_triple(indices, row)
    wp.atomic_add(out_sum, i0, a)
    wp.atomic_add(out_sum, i1, b)
    wp.atomic_add(out_sum, i2, c)


@wp.func
def peer_summed_add_corner_triple(
    out_sum: wp.array[Any], indices: wp.array[wp.int32], row: wp.int32, a: Any, b: Any, c: Any
) -> None:
    i0, i1, i2 = corner_triple(indices, row)
    peer_summed_add(out_sum, i0, a)
    peer_summed_add(out_sum, i1, b)
    peer_summed_add(out_sum, i2, c)


# A per-corner ``(n_faces, 3)`` table, or one value a face, added to its three vertices. The three
# corners are read off the flat ``faces`` buffer, never by a loop over an ``(n, k)`` index view:
# that loop's dynamic trip count cost the CPU 2.1x (scalar), 1.6x (vector) and 1.45x (weighted)
# at every size, where a view read entry by entry did not; CUDA at par (lucy 1.01-1.04x, inside
# the 1.01-1.07x two identical bodies read there). The corners are added in the order the loop
# added them, so CPU sums are bit-identical.
@wp.kernel
def scatter_sum_scalar(
    values: wp.array2d[wp.Scalar], faces: wp.array[wp.int32], out_sum: wp.array[wp.Scalar]
) -> None:
    f = wp.int32(wp.tid())
    peer_summed_add_corner_triple(out_sum, faces, f, values[f, 0], values[f, 1], values[f, 2])


@wp.kernel
def scatter_sum_vec(
    values: wp.array[wp.vec3], faces: wp.array[wp.int32], out_sum: wp.array2d[wp.Float]
) -> None:
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    value = values[f]
    peer_summed_add_vec3(out_sum, i0, value)
    peer_summed_add_vec3(out_sum, i1, value)
    peer_summed_add_vec3(out_sum, i2, value)


@wp.kernel
def scatter_weighted_sum_vec(
    values: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    weights: wp.array2d[wp.float32],
    out_sum: wp.array2d[wp.Float],
) -> None:
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    value = values[f]
    peer_summed_add_vec3(out_sum, i0, value * weights[f, 0])
    peer_summed_add_vec3(out_sum, i1, value * weights[f, 1])
    peer_summed_add_vec3(out_sum, i2, value * weights[f, 2])


@wp.func
def add_or_store_corner_triple(
    out_sum: wp.array[Any],
    out_corners: wp.array2d[Any],
    indices: wp.array[wp.int32],
    row: wp.int32,
    a: Any,
    b: Any,
    c: Any,
) -> None:
    # A face's three corner values, to whichever output the caller sized: stored per corner into
    # a non-empty ``out_corners`` (``(n_faces, 3)``), for ``sum_sorted_corner_values`` to add in
    # ascending corner order under ``ordito.config``'s deterministic mode, or added atomically
    # into ``out_sum`` (``add_corner_triple``) in arrival order otherwise.
    if out_corners.shape[0] > 0:
        out_corners[row, 0] = a
        out_corners[row, 1] = b
        out_corners[row, 2] = c
    else:
        add_corner_triple(out_sum, indices, row, a, b, c)


@wp.kernel
def scatter_face_thirds(
    faces: wp.array[wp.int32], areas: wp.array[Any], count: wp.Float, out_mass: wp.array[wp.Float]
) -> None:
    # Barycentric (lumped) mass: each face donates ``areas[f] / count`` to each incident vertex,
    # the area converted to the accumulator's dtype as it is read (exact for float32 into float64),
    # so a caller's areas of the other precision need no cast pass first.
    f = wp.int32(wp.tid())
    third = type(count)(areas[f]) / count
    peer_summed_add_corner_triple(out_mass, faces, f, third, third, third)


@wp.kernel
def sum_sorted_face_thirds(
    sorted_vertices: wp.array[wp.int32],
    corner_order: wp.array[wp.int32],
    areas: wp.array[Any],
    count: wp.Float,
    out_mass: wp.array[wp.Float],
) -> None:
    # ``scatter_face_thirds`` in ascending corner order: each vertex walks its run of the face
    # buffer's stable sort and adds the same rounded ``areas[f] / count`` per corner, so the sum
    # repeats bit for bit where the atomics commit in arrival order. The deterministic form for
    # ``float64`` areas, which have no exact ``float64`` sum (``sum_sorted_corner_values`` walks
    # the same runs over a per-corner table).
    v = wp.int32(wp.tid())
    n = sorted_vertices.shape[0]
    i = binary_search_index_left(sorted_vertices, v)
    total = type(count)(0.0)
    while i < n:
        if sorted_vertices[i] != v:
            break
        total += type(count)(areas[corner_order[i] // 3]) / count
        i += 1
    out_mass[v] = total


@wp.kernel
def scatter_face_areas_exact(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    areas: wp.array[wp.float32],
    out_sum: wp.array[wp.float64],
) -> None:
    # Each face's ``float32`` area into each of its corners' ``float64`` sum, unscaled. A
    # ``float32`` value is exact in ``float64``, and a sum of them stays exact while one vertex's
    # areas span less than ``2^(29 - log2(faces at it))``, so the atomic commit order changes no
    # bit: the result is reproducible on CUDA, where a sum of rounded thirds was not. The area is
    # the caller's ``areas[f]``, or, given ``None``, ``face_normals_and_area``'s, bit for bit;
    # ``scale_face_sums`` divides by three afterwards. The exact group sum (``kernels/reduce.py``)
    # drops the span condition but was declined here and in ``scatter_sum_exact``: one pass of
    # ``float64`` atomics beat it at every size priced, 1.2-1.9x with a caller's bound skipping its
    # bound pass and 2.5-3x without (20 k to 5.2 M faces, ordered and scrambled).
    f = wp.int32(wp.tid())
    area = wp.float32(0.0)
    if areas.shape[0] > 0:
        area = areas[f]
    else:
        _normal, area = face_normals_and_area(vertices, faces, f)
    value = wp.float64(area)
    peer_summed_add_corner_triple(out_sum, faces, f, value, value, value)


@wp.kernel
def scale_face_sums(
    sums: wp.array[wp.float64], count: wp.float64, out_mass: wp.array[wp.Float]
) -> None:
    # The exact per-vertex sums over ``count`` (three, for the barycentric mass), one rounding, in
    # the requested precision; ``out_mass`` may be ``sums`` itself.
    i = wp.int32(wp.tid())
    out_mass[i] = type(out_mass[i])(sums[i] / count)


@wp.kernel
def scatter_sum_exact(
    values: wp.array2d[wp.float32], faces: wp.array[wp.int32], out_sum: wp.array[wp.float64]
) -> None:
    # A per-corner ``(n_faces, 3)`` ``float32`` table summed into its vertices in ``float64``: a
    # ``float32`` value is exact in ``float64`` and so is a sum of them while one vertex's values
    # span less than ``2^(29 - log2(values at it))``, so the atomic commit order changes no bit, as
    # in ``scatter_face_areas_exact``. The angle sums of ``curvature.vertex_defects`` and of
    # ``curvature.discrete_gaussian_curvature``. Three corners off the flat ``faces`` buffer rather
    # than a loop over an ``(n, k)`` index view: the view's dynamic loop cost the CPU 1.6-2.0x
    # (bit-identical sums); and pooled (``peer_summed_add_corner_triple``) where the plain
    # ``float64`` atomics it replaced in ``discrete_gaussian_curvature`` were not, device time
    # new/old 0.10x on ``fan_hub``, 0.86-0.90x on dragon to lucy, at par below 100 k faces.
    f = wp.int32(wp.tid())
    peer_summed_add_corner_triple(
        out_sum,
        faces,
        f,
        wp.float64(values[f, 0]),
        wp.float64(values[f, 1]),
        wp.float64(values[f, 2]),
    )


@wp.kernel
def sum_sorted_corner_values(
    sorted_vertices: wp.array[wp.int32],
    corner_order: wp.array[wp.int32],
    corner_values: wp.array2d[Any],
    out_sums: wp.array[Any],
    out_offsets: wp.array[wp.int32],
    out_vertex_faces: wp.array[wp.int32],
) -> None:
    # Each vertex's sum of its corners' ``corner_values``, accumulated in ascending corner order:
    # ``sorted_vertices`` / ``corner_order`` are the face buffer's stable sort and its permutation,
    # so a vertex's run lists its corners in ascending order. The same bits on every run, where
    # ``add_corner_triple``'s atomics commit in arrival order -- the deterministic form of that
    # scatter for ``float64`` values, which have no exact float atomic sum, and for vectors
    # (``SUM_SORTED_CORNER_VALUES``' dtypes). The exact group sum (``kernels/reduce.py``,
    # ``scatter_add_vector``) covers both and was priced against it as ``parametrization.arap``
    # uses it, one sort a call and a walk a round: with its bound pass skipped (``|R w| = |w|``
    # bounds every term) it ran 2x the walk's device time on a scrambled 5 M-face mesh and 0.8x on
    # an ordered one, so the walk stays.
    #
    # A "corner" is a slot of the flat buffer ``sorted_vertices`` was sorted from, and
    # ``corner_values`` has one row per row of that buffer: three slots a face here, two an edge
    # for ``remesh.subdivide_loop``'s ring sums, which sorts its unique edges' endpoints, six a
    # face (two per edge) for ``parametrization.arap``'s rotation right-hand side.
    #
    # The runs are also the vertex-face incidence CSR ``adjacency.vertex_face_adjacency`` returns
    # in deterministic mode, each row in ascending face order, so a caller that wants those rows
    # passes non-empty ``out_offsets`` (``(n_vertices + 1,)``) and ``out_vertex_faces``
    # (``(3 * n_faces,)``) and gets them from the same walk instead of a second sort.
    v = wp.int32(wp.tid())
    n = sorted_vertices.shape[0]
    columns = corner_values.shape[1]
    rows = out_offsets.shape[0] > 0
    i = binary_search_index_left(sorted_vertices, v)
    if rows:
        out_offsets[v] = i
        if v == out_sums.shape[0] - 1:
            out_offsets[v + 1] = n
    total = type(out_sums[v])(0.0)
    while i < n:
        if sorted_vertices[i] != v:
            break
        corner = corner_order[i]
        total += corner_values[corner // columns, corner % columns]
        if rows:
            out_vertex_faces[i] = corner // columns
        i += 1
    out_sums[v] = total


@wp.func
def add_value_and_count(
    vertex: wp.int32,
    value: wp.float32,
    count: wp.float32,
    out_sum: wp.array[Any],
    out_valence: wp.array[wp.float32],
    pool_peers: wp.int32,
) -> None:
    # One contribution to a per-vertex mean: the value into the sum, ``count`` into the valence.
    # Shared by the face-valued and edge-valued averaging scatters below and remesh's scale
    # attribute. Under ``pool_peers`` (a constant at each call site, so the other arm folds away)
    # it pools as ``peer_summed_add``: the lanes sharing a vertex sum their values in ``float64``
    # and the lowest commits the sum and the lanes' number off one match. Only the CUDA device
    # pools, and its callers count 1 a call; a CPU caller may count 0 (an exact zero added in
    # place of a branch). A ``float64`` sum of ``float32`` values is exact, so it is order-free
    # (the deterministic setting); a ``float32`` sum is the arrival-order form.
    #
    # Pooled against the plain pair of atomics, device time with the setting off: the averaging
    # scatters ``fan_hub`` 0.11x (faces) and 0.19x (edges), lucy 0.80x / 0.94x, dragon 0.94x /
    # 1.03x, below 100 k faces +2-3 us hidden by host time; CPU bits unchanged. Declined for
    # remesh's scale attribute (``remesh.scatter_scale_attribute``): ``fan_hub`` 0.24x, but dragon
    # 1.41x, lucy 1.11x and +2 us small, its launch over every halfedge with half the lanes gone
    # by the commit, where the match costs more than the atomics it saves.
    lanes = wp.int32(0)
    if pool_peers != 0:
        lanes = warp_peer_lanes(vertex)
    if lanes == 0:
        wp.atomic_add(out_sum, vertex, out_sum.dtype(value))
        wp.atomic_add(out_valence, vertex, count)
    else:
        total = warp_lanes_sum_float(lanes, wp.float64(value))
        if warp_lane_leads(lanes) != 0:
            wp.atomic_add(out_sum, vertex, out_sum.dtype(total))
            wp.atomic_add(out_valence, vertex, wp.float32(popcount32(wp.uint32(lanes))))


@wp.kernel
def scatter_face_values_sum_and_valence(
    faces: wp.array[wp.int32],
    face_values: wp.array[wp.float32],
    out_sum: wp.array[Any],
    out_valence: wp.array[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    value = face_values[f]
    for j in range(3):
        add_value_and_count(faces[f * 3 + j], value, wp.float32(1.0), out_sum, out_valence, 1)


@wp.func
def accumulate_endpoint_value(
    vi: wp.int32,
    vj: wp.int32,
    value: wp.float32,
    count: wp.float32,
    out_sum: wp.array[Any],
    out_valence: wp.array[wp.float32],
    pool_peers: wp.int32,
) -> None:
    # Add one edge's value to both of its endpoints and count it at each -- the shared body of the
    # two "average an edge field onto the vertices" scatters below. They compute the same quantity
    # and differ only in where they find the endpoints and the value: one walks the *half*-edges of
    # each face (the ``igl::orient_halfedges`` convention, skipping the negatively oriented copy so
    # an interior edge is counted once), the other walks the unique-edge list directly.
    add_value_and_count(vi, value, count, out_sum, out_valence, pool_peers)
    add_value_and_count(vj, value, count, out_sum, out_valence, pool_peers)


@wp.func
def accumulate_oriented_edge(
    edges: wp.array2d[wp.int32],
    edges_orientation: wp.array2d[wp.int32],
    edge_values: wp.array[wp.float32],
    f: wp.int32,
    j: wp.int32,
    u: wp.int32,
    v: wp.int32,
    out_sum: wp.array[Any],
    out_valence: wp.array[wp.float32],
) -> None:
    # Face ``f``'s edge ``j`` from ``u`` to ``v``, unless it is the negatively oriented copy. Half
    # of a closed mesh's halfedges are, in no pattern a branch predictor learns, so the CPU adds an
    # exact zero for them instead of branching (the same bits): the kernel 0.51x on dragon, the
    # call 0.42x, with a three-call form over the corners in place of a ``continue`` loop 0.83x of
    # that. The CUDA arm keeps the branch, where a doubled atomic would cost more than divergence.
    if device_is_cpu() != 0:
        kept = edges_orientation[f, j] >= 0
        value = wp.where(kept, edge_values[edges[f, j]], wp.float32(0.0))
        count = wp.where(kept, wp.float32(1.0), wp.float32(0.0))
        accumulate_endpoint_value(u, v, value, count, out_sum, out_valence, 0)
    elif edges_orientation[f, j] >= 0:
        accumulate_endpoint_value(
            u, v, edge_values[edges[f, j]], wp.float32(1.0), out_sum, out_valence, 1
        )


@wp.kernel
def scatter_edges_sum_and_valence(
    faces: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    edges_orientation: wp.array2d[wp.int32],
    edge_values: wp.array[wp.float32],
    out_sum: wp.array[Any],
    out_valence: wp.array[wp.float32],
) -> None:
    # Half-edge form: launch over faces, skip the negatively oriented copy of each interior edge.
    # Edge ``j`` is halfedge ``halfedge_next(3 f + j)``, running from corner ``j + 1`` to ``j + 2``.
    f = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, f)
    accumulate_oriented_edge(
        edges, edges_orientation, edge_values, f, 0, b, c, out_sum, out_valence
    )
    accumulate_oriented_edge(
        edges, edges_orientation, edge_values, f, 1, c, a, out_sum, out_valence
    )
    accumulate_oriented_edge(
        edges, edges_orientation, edge_values, f, 2, a, b, out_sum, out_valence
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
    # half of a parallel independent set over edge collapses. A candidate that still holds its own
    # key at both endpoints conflicts with no claimant of a smaller key (``endpoints_hold``).
    #
    # Both of ``kernels/remesh.py``'s collapse paths run it, with the keys ``remesh.scramble_index``
    # and ``remesh.bucketed_lock_key`` build. Two properties of
    # those keys are load-bearing and both are recorded there: the key field must not be spatially
    # monotone (a monotone key commits one collapse per pass), and the key must be *injective*, or
    # two candidates can tie and both believe they won -- which is why it is 64 bits wide here
    # rather than the natural int32 of a vertex index.
    wp.atomic_min(out_claim, s, key)
    wp.atomic_min(out_claim, r, key)
    for i in range(offsets[s], offsets[s + 1]):
        wp.atomic_min(out_claim, columns[i], key)
    for i in range(offsets[r], offsets[r + 1]):
        wp.atomic_min(out_claim, columns[i], key)


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
    # agrees on. A collapse whose endpoints are both unmarked afterwards (``endpoints_hold``)
    # conflicts with none of the stamped ones.
    out_marks[s] = value
    out_marks[r] = value
    for i in range(offsets[s], offsets[s + 1]):
        out_marks[columns[i]] = value
    for i in range(offsets[r], offsets[r + 1]):
        out_marks[columns[i]] = value


@wp.func
def endpoints_hold(table: wp.array[Any], s: wp.int32, r: wp.int32, value: Any) -> wp.bool:
    # Do both endpoints of the collapse ``(s, r)`` hold ``value`` in ``table``? The read half of
    # both collapse paths' independent sets: after ``lock_two_rings`` it is the win test (``value``
    # the collapse's own key), and after ``stamp_two_rings`` the quadric pass's "no committed
    # collapse conflicts" test (``value`` zero).
    #
    # Two collapses conflict when an endpoint of either lies in the other's closed 1-rings, and
    # only then. A collapse ``(s, r)`` changes the faces incident to ``s`` or ``r`` and nothing
    # else: it moves ``s`` and deletes ``r``. When neither endpoint of ``B`` lies in ``A``'s closed
    # 1-rings, no face of ``A``'s star holds a vertex ``B`` moves or deletes, and ``B`` adds or
    # removes no edge at ``A``'s endpoints, so ``A``'s link condition, fold veto, quadrics and
    # placement are exactly the ones its scoring saw: the pair is equivalent to applying the two
    # one after the other. The rings themselves may overlap.
    #
    # Adjacency is symmetric, so "an endpoint of ``B`` is in ``A``'s closed rings" is the same
    # statement as "an endpoint of ``A`` is in ``B``'s", and reading the claim at the two endpoints
    # alone decides the conflict both ways. Reading every ring vertex instead is the stricter
    # disjoint-rings rule the decimators used before: it admits about half as many collapses a
    # pass, and a collapse stage capped at a few passes then stops well short of its target.
    return table[s] == value and table[r] == value


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


@wp.kernel
def scatter_value_where_scanned(
    inclusive: wp.array[wp.int32], values: wp.array[wp.vec3], out_scattered: wp.array[wp.vec3]
) -> None:
    # ``scatter_index_where_scanned`` with the gather folded in: every selected position's
    # *value* lands at its compact slot, for a caller that wants ``values[flatnonzero(flags)]``
    # and not the indices, so no index buffer is written and gathered back.
    i = wp.int32(wp.tid())
    start, count = scanned_count(inclusive, i)
    if count != 0:
        out_scattered[start] = values[i]


@wp.func
def append_to_pair(
    row: wp.int32, value: wp.int32, out_count: wp.array[wp.int32], out_pairs: wp.array2d[wp.int32]
) -> None:
    # Count ``value`` onto ``row`` and record it in the row's first two slots, in arbitrary order:
    # the count doubles as the write cursor, and a third or later arrival is counted but not
    # stored -- a defensive bound on the ``(n, 2)`` row width, whatever the caller's manifoldness
    # guarantee. The rule behind every per-unique-edge "its two faces / halfedges" table.
    slot = wp.atomic_add(out_count, row, 1)
    if slot < 2:
        out_pairs[row, slot] = value


@wp.func
def append_to_pair_slot(
    row: wp.int32, value: wp.int32, out_count: wp.array[wp.int32], out_pairs: wp.array2d[wp.int32]
) -> wp.int32:
    # ``append_to_pair`` returning the arrival slot it drew, for a caller that acts once per row
    # (slot 0 is drawn by exactly one arrival): ``remesh.count_edge_topology`` counts each unique
    # edge onto its endpoints' degrees there.
    slot = wp.atomic_add(out_count, row, 1)
    if slot < 2:
        out_pairs[row, slot] = value
    return slot


@wp.func
def record_edge_incidence(
    e: wp.int32,
    corner: wp.int32,
    out_edge_face_count: wp.array[wp.int32],
    out_edge_faces: wp.array2d[wp.int32],
) -> None:
    # Count corner ``corner``'s face onto unique edge ``e`` and record it in the edge's first two
    # face slots (``append_to_pair``; ``scatter_edge_incidence`` below).
    append_to_pair(e, corner // 3, out_edge_face_count, out_edge_faces)


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
    # argument keeps the ``out_`` prefix and the output-naming allowlist carries it.
    i, j, k = wp.tid()
    out_field[i, j, k] = out_field[i, j, k] / wp.max(density[i, j, k], min_weight)


# Concrete overloads, registered at import -- rationale in ``ordito/kernels/reduce.py``.
# This module is reached from 11 wrappers at scattered moments, so it forked
# nearly twice per overload before registration.
#
# Each set is the dtypes its call sites actually build, not a menu: the two mass/curvature
# scatters follow their wrapper's precision keyword.
_VALUE_DTYPES = (wp.float32, wp.float64)
# The two vector scatters get a set of **one**, for the same reason: their only caller is
# ``normals._accumulate_and_normalize``, which accumulates at ``wp.float64`` so that the summation
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
# ``reduce.scatter_sum`` / ``segment_sum``'s value dtypes: the scalars and the vectors it documents.
_GROUP_SUM_SCALARS = (wp.float32, wp.float64)
_GROUP_SUM_VECTORS = (wp.vec2, wp.vec3, wp.vec4, wp.vec2d, wp.vec3d, wp.vec4d)
# ``reduce.scatter_min`` / ``scatter_max`` and their segment forms' value dtypes.
EXTREMUM_DTYPES = (wp.int32, wp.int64, wp.float32, wp.float64, *_GROUP_SUM_VECTORS)

SCATTER_SUM_SCALARS: OverloadTable
SCATTER_SUM_VECTORS: OverloadTable
SEGMENT_SUM_SCALARS: OverloadTable
SEGMENT_SUM_VECTORS: OverloadTable
SCATTER_EXTREMA: OverloadTable
SEGMENT_EXTREMA: OverloadTable
DIVIDE_BY_DENSITY: OverloadTable
SPLAT_GRID_TRILINEAR: OverloadTable
SCATTER_FACE_THIRDS: OverloadTable
SUM_SORTED_FACE_THIRDS: OverloadTable
SCALE_FACE_SUMS: OverloadTable
SCATTER_SUM_SCALAR: OverloadTable
SCATTER_SUM_VEC: OverloadTable
SCATTER_WEIGHTED_SUM_VEC: OverloadTable
SCATTER_FACE_VALUES_SUM_AND_VALENCE: OverloadTable
SCATTER_EDGES_SUM_AND_VALENCE: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global DIVIDE_BY_DENSITY, SPLAT_GRID_TRILINEAR
    global SCATTER_FACE_VALUES_SUM_AND_VALENCE, SCATTER_EDGES_SUM_AND_VALENCE
    global SCATTER_SUM_SCALARS, SCATTER_SUM_VECTORS, SEGMENT_SUM_SCALARS, SEGMENT_SUM_VECTORS
    SCATTER_SUM_SCALARS = OverloadTable(
        scatter_sum_scalars,
        {
            d: [wp.array[d], wp.array[wp.int32], wp.int32, GroupSumPass, wp.array[d]]
            for d in _GROUP_SUM_SCALARS
        },
    )
    SCATTER_SUM_VECTORS = OverloadTable(
        scatter_sum_vectors,
        {
            d: [wp.array[d], wp.array[wp.int32], wp.int32, GroupSumPass, wp.array[d]]
            for d in _GROUP_SUM_VECTORS
        },
    )
    SEGMENT_SUM_SCALARS = OverloadTable(
        segment_sum_scalars,
        {
            d: [wp.array[d], wp.array[wp.int32], GroupSumPass, wp.array[d]]
            for d in _GROUP_SUM_SCALARS
        },
    )
    SEGMENT_SUM_VECTORS = OverloadTable(
        segment_sum_vectors,
        {
            d: [wp.array[d], wp.array[wp.int32], GroupSumPass, wp.array[d]]
            for d in _GROUP_SUM_VECTORS
        },
    )
    global SCATTER_EXTREMA, SEGMENT_EXTREMA
    SCATTER_EXTREMA = OverloadTable(
        scatter_extrema,
        {d: [wp.array[d], wp.array[wp.int32], wp.int32, wp.array[d]] for d in EXTREMUM_DTYPES},
    )
    SEGMENT_EXTREMA = OverloadTable(
        segment_extrema,
        {d: [wp.array[d], wp.array[wp.int32], wp.int32, wp.array[d]] for d in EXTREMUM_DTYPES},
    )
    global SCALE_FACE_SUMS, SCATTER_FACE_THIRDS, SCATTER_SUM_SCALAR, SUM_SORTED_FACE_THIRDS
    global SCATTER_SUM_VEC, SCATTER_WEIGHTED_SUM_VEC, SUM_SORTED_CORNER_VALUES
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
    # Keyed ``(areas dtype, mass dtype)``, the cross product: a caller's areas may be either.
    SCATTER_FACE_THIRDS = OverloadTable(
        scatter_face_thirds,
        {
            (a, d): [wp.array[wp.int32], wp.array[a], d, wp.array[d]]
            for a in _VALUE_DTYPES
            for d in _VALUE_DTYPES
        },
    )
    SUM_SORTED_FACE_THIRDS = OverloadTable(
        sum_sorted_face_thirds,
        {
            (a, d): [wp.array[wp.int32], wp.array[wp.int32], wp.array[a], d, wp.array[d]]
            for a in _VALUE_DTYPES
            for d in _VALUE_DTYPES
        },
    )
    SCALE_FACE_SUMS = OverloadTable(
        scale_face_sums, {d: [wp.array[wp.float64], wp.float64, wp.array[d]] for d in _VALUE_DTYPES}
    )
    SCATTER_SUM_SCALAR = OverloadTable(
        scatter_sum_scalar,
        {d: [wp.array2d[d], wp.array[wp.int32], wp.array[d]] for d in _VALUE_DTYPES},
    )
    SCATTER_SUM_VEC = OverloadTable(
        scatter_sum_vec,
        {
            d: [wp.array[wp.vec3], wp.array[wp.int32], wp.array2d[d]]
            for d in _VECTOR_ACCUMULATOR_DTYPES
        },
    )
    SCATTER_WEIGHTED_SUM_VEC = OverloadTable(
        scatter_weighted_sum_vec,
        {
            d: [wp.array[wp.vec3], wp.array[wp.int32], wp.array2d[wp.float32], wp.array2d[d]]
            for d in _VECTOR_ACCUMULATOR_DTYPES
        },
    )

    # The averaging scatters' sums: ``float32`` in arrival order, ``float64`` exact (the
    # deterministic setting).
    SCATTER_FACE_VALUES_SUM_AND_VALENCE = OverloadTable(
        scatter_face_values_sum_and_valence,
        {
            d: [wp.array[wp.int32], wp.array[wp.float32], wp.array[d], wp.array[wp.float32]]
            for d in (wp.float32, wp.float64)
        },
    )
    SCATTER_EDGES_SUM_AND_VALENCE = OverloadTable(
        scatter_edges_sum_and_valence,
        {
            d: [
                wp.array[wp.int32],
                wp.array2d[wp.int32],
                wp.array2d[wp.int32],
                wp.array[wp.float32],
                wp.array[d],
                wp.array[wp.float32],
            ]
            for d in (wp.float32, wp.float64)
        },
    )

    # ``float64`` sums (the energies' angle sums and mass, the heat method's divergence), the
    # ``wp.vec3`` ones of the log map's radial direction and Loop subdivision's ring sums, and the
    # ``wp.vec2d`` ones of the ARAP local step's rotation right-hand side.
    SUM_SORTED_CORNER_VALUES = OverloadTable(
        sum_sorted_corner_values,
        {
            d: [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array2d[d],
                wp.array[d],
                wp.array[wp.int32],
                wp.array[wp.int32],
            ]
            for d in (wp.float64, wp.vec3, wp.vec2d)
        },
    )


_register_overloads()


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's ``wp.map`` signatures so each builds one module.

    See ``kernels/array.py::declare_map_signatures`` for why this exists.
    """
    dense = map_probe
    declare_map_signatures(
        [
            *((scalar_mean_of_sum, (dense(d), dense(wp.int32)), d) for d in _GROUP_SUM_SCALARS),
            *((vector_mean_of_sum, (dense(d), dense(wp.int32)), d) for d in _GROUP_SUM_VECTORS),
        ]
    )


_declare_map_kernels()


# --------------------------------------------------------------------------------------
# Sorted runs added in a fixed order
#
# A stably sorted key list groups each output's items in ascending order, a run per output; the run
# sums below add each run in one fixed order, so the result repeats bit for bit run to run without
# a float atomic. A run of at most ``SORTED_RUN_SERIAL_MAX`` items is added by one thread
# (``sorted_run_sums``), a longer one by a block of ``HEAVY_RUN_BLOCK_DIM`` lanes
# (``heavy_run_sums``), at most ``HEAVY_RUN_BLOCKS`` blocks striding the list of long runs. Swept on
# CUDA over cells of 2 and 6 mean edges and of 10 % and 30 % of the diagonal on ``dragon`` and
# ``lucy`` (``remesh.cluster_decimate``): a block per run at every size cost 1.6-3.3x the atomic
# scatter on fine cells (32-256 lanes, mostly idle over a few items), a thread per run up to 40x on
# coarse ones (one serial walk over most of the mesh); split at 128 the ordered mean costs
# 1.05-1.17x the scatter on fine cells and 0.1-0.7x on coarse ones, where the scatter's atomics
# pile onto a few addresses. Serial bounds of 16-32 handed too many short runs to blocks (up to
# 2.3x at ``lucy`` / 6 edges); 64-256 measured alike. The split decides each run's summation
# order, so it is a fixed constant, not a device query. The CPU device runs one lane a block,
# which adds a long run in ascending order like a short one.
# --------------------------------------------------------------------------------------
SORTED_RUN_SERIAL_MAX = 128
HEAVY_RUN_BLOCK_DIM = 256
HEAVY_RUN_BLOCKS = 4096


@wp.func
def sorted_run(sorted_keys: wp.array[wp.int32], r: wp.int32) -> tuple[wp.int32, wp.int32]:
    # Output ``r``'s run ``[start, stop)`` in a stably sorted key list, empty when no item has it.
    return binary_search_index_left(sorted_keys, r), binary_search_index_left(sorted_keys, r + 1)


@wp.func
def strided_run_sum(
    values: wp.array[wp.vec3],
    order: wp.array[wp.int32],
    first: wp.int32,
    stop: wp.int32,
    stride: wp.int32,
) -> wp.vec3:
    # Every ``stride``-th item of a sorted run from ``first``, added in ascending order from zero:
    # the whole run at stride one, one lane's share of it at the block's width.
    total = wp.vec3(0.0)
    for i in range(first, stop, stride):
        total += values[order[i]]
    return total


@wp.func
def mean_from_sum(total: wp.vec3, count: wp.int32) -> wp.vec3:
    # A group's mean from its sum and count, dividing by the count converted in place rather than
    # by a separately materialised ``float32`` copy of the counts.
    return total / wp.float32(count)


@wp.func
def finish_run_sum(total: wp.vec3, count: wp.int32, average: wp.bool) -> wp.vec3:
    # A run's sum, or with ``average`` its mean; an empty run stays zero.
    result = total
    if average and count > 0:
        result = mean_from_sum(total, count)
    return result


@wp.kernel
def sorted_run_sums(
    values: wp.array[wp.vec3],
    sorted_keys: wp.array[wp.int32],
    order: wp.array[wp.int32],
    serial_max: wp.int32,
    average: wp.bool,
    out_values: wp.array[wp.vec3],
    out_heavy_count: wp.array[wp.int32],
    out_heavy: wp.array[wp.int32],
) -> None:
    # A thread per output: a run of at most ``serial_max`` items is added here in ascending order,
    # and a longer one is appended to ``out_heavy`` (``out_heavy_count`` caller-zeroed) for
    # ``heavy_run_sums``. The list's arrival order reaches no sum: each long run is added by one
    # block in its own fixed order, whichever block takes it.
    r = wp.int32(wp.tid())
    start, stop = sorted_run(sorted_keys, r)
    if stop - start > serial_max:
        out_heavy[append_slot(out_heavy_count, 0)] = r
        return
    total = strided_run_sum(values, order, start, stop, wp.int32(1))
    out_values[r] = finish_run_sum(total, stop - start, average)


@wp.kernel
def heavy_run_sums(
    values: wp.array[wp.vec3],
    sorted_keys: wp.array[wp.int32],
    order: wp.array[wp.int32],
    heavy_count: wp.array[wp.int32],
    heavy: wp.array[wp.int32],
    n_blocks: wp.int32,
    average: wp.bool,
    out_values: wp.array[wp.vec3],
) -> None:
    # The runs ``sorted_run_sums`` left, a block per run over ``n_blocks`` blocks striding the list:
    # each lane adds every ``wp.block_dim()``-th item of the run in ascending order, and the block
    # folds the lanes' sums in its fixed tree. The list's length is uniform over a block, so every
    # lane reaches the fold.
    b, lane = wp.tid()
    for k in range(b, heavy_count[0], n_blocks):
        r = heavy[k]
        start, stop = sorted_run(sorted_keys, r)
        total = block_sum(strided_run_sum(values, order, start + lane, stop, wp.block_dim()))
        if lane == 0:
            out_values[r] = finish_run_sum(total, stop - start, average)
