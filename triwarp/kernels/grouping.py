import warp as wp

from triwarp.kernels import array as kernel_array
from triwarp.kernels.array import OverloadTable, pack_triangle_key, sorted_run_start
from triwarp.kernels.triangles import corner_triple


@wp.kernel
def scatter_first_occurrence(inverse: wp.array[wp.int32], out_first: wp.array[wp.int32]) -> None:
    # Smallest index mapping to each class. ``out_first`` must be pre-filled with a sentinel at
    # least ``inverse.shape[0]``, so a class with no member keeps it.
    i = wp.int32(wp.tid())
    wp.atomic_min(out_first, inverse[i], i)


@wp.func
def sorted_run_of_length(
    sorted_values: wp.array[wp.Int], n: wp.int32, i: wp.int32, length: wp.int32
) -> wp.bool:
    # Does position ``i`` begin a run of *exactly* ``length`` equal values among the first ``n``?
    # ``mark_group_starts`` materializes it as flags; ``remesh.mark_edge_pair_starts`` asks it at
    # ``length == 2``; ``boundary``'s halfedge masks and ``selection``'s region seam read the
    # verdict at a sorted position directly. Each ``and`` short-circuits in kernel scope, so the
    # last test reads ``i + length`` only when it is inside the data.
    if i + length > n:
        return False  # run would extend past the data
    if not sorted_run_start(sorted_values, i):
        return False  # not the start of a run
    if sorted_values[i] != sorted_values[i + length - 1]:
        return False  # run shorter than ``length``
    if i + length < n and sorted_values[i] == sorted_values[i + length]:
        return False  # run longer than ``length``
    return True


@wp.kernel
def mark_group_starts(
    sorted_values: wp.array[wp.Int], n: wp.int32, length: wp.int32, out_flags: wp.array[wp.int32]
) -> None:
    # Flag (``1``) positions that start a run of exactly ``length`` equal values in the sorted
    # buffer (which may be over-allocated radix-sort scratch; only the first ``n`` entries are
    # data). ``int32`` rather than ``wp.bool`` because the flags go straight into
    # ``wp.utils.array_scan``, which has no bool overload -- the same reason
    # ``remesh.mark_edge_pair_starts``, this rule specialised to ``length == 2``, emits ``int32``.
    tid = wp.int32(wp.tid())
    is_start = sorted_run_of_length(sorted_values, n, tid, length)
    out_flags[tid] = wp.where(is_start, wp.int32(1), wp.int32(0))


@wp.kernel
def emit_groups(
    offsets: wp.array[wp.int32], indices: wp.array[wp.int32], out_groups: wp.array2d[wp.int32]
) -> None:
    # Launched over the ``n`` sorted positions with ``offsets`` the total-terminated exclusive scan
    # of ``mark_group_starts``' flags: a position starts a group exactly where the scan steps, and
    # the step's value is the group's row. Reading the flag back off the scan is what lets one
    # launch do the compaction and the emit together, instead of a ``flatnonzero`` of the starts
    # followed by a launch over them.
    i = wp.int32(wp.tid())
    g, starts = kernel_array.scanned_slot(offsets, i)
    if not starts:
        return
    for j in range(out_groups.shape[1]):
        out_groups[g, j] = indices[i + j]


HASH_MULT_U64 = wp.constant(wp.uint64(11400714819323198485))  # 0x9e3779b97f4a7c15

# Folds the multiplicative hash's high half onto its low half; see ``hash_slot``.
HASH_FOLD_SHIFT = wp.constant(wp.uint64(32))

VEC3_PACK_PRECISION = wp.constant(wp.uint64(64 // 3))
VEC3_PACK_SHIFT = wp.constant(wp.uint32(11))


# ---------------------------------------------------------------------------
# Shared hash helpers
# All hash kernels receive `mask: wp.int32` = capacity - 1 rather than capacity
# itself.  For capacity <= 2^31, mask = capacity - 1 <= INT32_MAX, which is always
# a valid non-negative int32.  Slot indices h = hash(...) & mask satisfy
# h <= mask <= INT32_MAX, so h is also safely non-negative as int32.
# ---------------------------------------------------------------------------


def hash_table_mask(n: int) -> int:
    """
    Return the slot mask of an open-addressing table for ``n`` keys, at least 8 slots.

    At least twice ``n`` slots, a power of two, so a table of distinct keys is at most half full.
    Host-side sizing shared by every table the hash functions below address.
    """
    return (1 << max(3, (n - 1).bit_length() + 1)) - 1


@wp.func
def hash_slot(key: wp.Int, mask: wp.int32) -> wp.int32:
    """Fibonacci hash; mask = capacity-1, capacity must be power-of-2."""
    # wp.cast is same-size only; use constructors for cross-size conversion.
    h = wp.uint64(wp.int64(key)) * HASH_MULT_U64
    # Fold the product's high half down before masking, or this is not the Fibonacci hash named
    # above. ``mask`` keeps the *low* bits, and the low bits of a multiplicative hash depend on
    # nothing but the key's low bits -- the multiplier is odd, so ``key * M mod 2^p`` is a
    # bijection of ``key mod 2^p`` and every higher bit of the key is discarded. Any key family
    # holding its entropy above bit ``p`` then collapses onto a handful of probe chains: on an
    # axis-aligned grid, whose ``pack_vec3`` keys carry the x bucket in the low bits and y/z above
    # them, that is thousands of probes per insert against the fold's ~1.5, and an order of
    # magnitude end to end on ``unique_rows``. A *random* cloud is flat either way, which is the
    # point: the fold costs nothing and only the structured key families were paying.
    #
    # Probe count only -- no caller depends on the mapping: ``_unique_hash`` sorts the compacted
    # keys afterwards, ``bfs_visited_insert`` uses the table as a set, and ``remesh``'s claim locks
    # re-read the slot they wrote. A hash that mixes the high bits cannot also be a bijection on
    # the low ones, so this gives up the old form's accidental collision-free behaviour on *dense
    # consecutive* keys to stop degrading on every other family -- and that advantage was already
    # gone once the ids were merely strided.
    h = h ^ (h >> HASH_FOLD_SHIFT)
    return wp.int32(h & wp.uint64(mask))


@wp.func
def hash_slot_words3(high: wp.int32, low: wp.int32, spread: wp.int32, mask: wp.int32) -> wp.int32:
    """Home slot of a three-word key: ``high`` and ``low`` side by side, xor ``spread`` spread."""
    # The mixing every open-addressing table of *indices* keyed on three 32-bit words shares
    # (``points.position_hash_slot`` over a position's bits, ``voxels.cell_hash_slot`` over a
    # cell, ``remesh.cluster_mark_faces`` over a sorted cluster triple): two words side by side in
    # one 64-bit word, xor'ed with the third spread by the Fibonacci multiplier, then
    # ``hash_slot``'s fold. Any mixing works -- every caller resolves a collision by comparing the
    # stored entry's key, never trusting the slot -- so this only has to spread the probes,
    # including over an axis-aligned lattice.
    x = wp.uint64(wp.uint32(high))
    y = wp.uint64(wp.uint32(low))
    z = wp.uint64(wp.uint32(spread))
    return hash_slot(wp.int64(((x << wp.uint64(32)) | y) ^ (z * HASH_MULT_U64)), mask)


@wp.func
def next_slot(h: wp.int32, mask: wp.int32) -> wp.int32:
    """Linear-probe to the next slot, wrapping with the power-of-2 mask."""
    return wp.cast((wp.cast(h, wp.uint32) + wp.uint32(1)) & wp.cast(mask, wp.uint32), wp.int32)


@wp.func
def empty_key(key: wp.Int) -> wp.Int:
    """Typed zero matching ``key`` (for empty-slot sentinel and comparisons)."""
    return key - key


@wp.func
def encode_key(key: wp.Int) -> wp.Int:
    """Map keys to table slots; 0 is reserved as the empty sentinel."""
    return -~key


@wp.func
def decode_key(stored: wp.Int) -> wp.Int:
    zero = stored - stored
    return stored + (~zero)


# ---------------------------------------------------------------------------
# Generic hash table kernels
# Slot layout: slot_key (Int, 0 = empty) + slot_counts (int32).
# Insertion uses atomic_cas on slot_key so publication is a single atomic
# operation — no separate lock/ready state and no cross-thread visibility races.
# ---------------------------------------------------------------------------


@wp.func
def hash_find_or_insert(key: wp.Int, slot_key: wp.array[wp.Int], mask: wp.int32) -> wp.int32:
    """
    Slot holding ``key``, inserting it first if it is not there yet.

    Publication is the single ``atomic_cas`` that claims an empty slot, so there is no separate
    lock or ready state and no cross-thread visibility race. The probe cannot terminate on a
    **full** table, so callers must size the table above the number of distinct keys.
    """
    empty = empty_key(key)
    encoded = encode_key(key)
    h = hash_slot(key, mask)
    while True:
        prev = wp.atomic_cas(slot_key, h, empty, encoded)
        if prev == empty or prev == encoded:
            break
        h = next_slot(h, mask)
    return h


@wp.func
def hash_find(key: wp.Int, slot_key: wp.array[wp.Int], mask: wp.int32) -> wp.int32:
    """
    Slot holding ``key``, or ``-1`` when it is absent.

    Read-only, so it terminates at the first empty slot in the probe chain even on a full table.
    """
    empty = empty_key(key)
    encoded = encode_key(key)
    h = hash_slot(key, mask)
    # No boolean accumulator: a bare ``True`` / ``False`` is a *constant* in kernel scope and Warp
    # refuses to let a dynamic loop mutate one. The probe state itself carries the answer.
    stored = slot_key[h]
    while stored != empty and stored != encoded:
        h = next_slot(h, mask)
        stored = slot_key[h]
    if stored != encoded:
        h = wp.int32(-1)
    return h


# A deleted slot of a ``hash_find_or_insert`` table used as a *set* with removal: never ``0`` (the
# empty sentinel) and never an encoded key, since ``encode_key`` of every non-negative ``int64`` key
# stays below it. ``hash_find`` probes past it and ``hash_find_or_insert`` claims only empty slots,
# so an insert racing a removal in one launch is safe; the cost is that tombstones only accumulate,
# and the owner rebuilds the table before they fill it.
KEY_SET_TOMBSTONE = wp.constant(wp.uint64(0xFFFFFFFFFFFFFFFF))


@wp.func
def key_set_remove(key: wp.uint64, slot_key: wp.array[wp.uint64], mask: wp.int32) -> None:
    """Delete ``key`` from a ``hash_find_or_insert`` set, leaving a tombstone; absent is a no-op."""
    h = hash_find(key, slot_key, mask)
    if h >= 0:
        slot_key[h] = KEY_SET_TOMBSTONE


@wp.kernel
def hash_insert(
    data: wp.array[wp.Int],
    slot_key: wp.array[wp.Int],
    slot_counts: wp.array[wp.int32],
    mask: wp.int32,
    out_occupied: wp.array[wp.int32],
) -> None:
    # `encode_key` reserves 0 as the empty-slot sentinel, so exactly one key -- the one that encodes
    # to 0, i.e. -1 -- can never be published in the table: its CAS would look like an untouched
    # slot. Since no bijection on the full integer range can avoid mapping *something* onto the
    # sentinel, that key gets a dedicated slot one past the end of the table instead. Callers must
    # therefore allocate `mask + 2` slots, and occupancy is stamped here rather than read back off
    # `slot_key`, whose reserved slot keeps an untouched 0 and would otherwise look empty --
    # `decode_key` turns that 0 straight back into -1, so compaction needs no special case either.
    #
    # `out_occupied` is a zero-filled 0/1 array, the dtype `wp.utils.array_scan` wants, so the scan
    # of it gives the compaction's write positions directly. Every thread landing in a slot stores
    # the same 1, which is why the unsynchronized store is benign; writing it here rather than in a
    # second pass over the whole `mask + 2` table saves that pass, which is a real share of
    # ``unique_1d`` since the table is several times the input.
    i = wp.int32(wp.tid())
    key = data[i]
    slot = mask + 1
    if encode_key(key) != empty_key(key):
        slot = hash_find_or_insert(key, slot_key, mask)
    wp.atomic_add(slot_counts, slot, wp.int32(1))
    out_occupied[slot] = wp.int32(1)


@wp.kernel
def compact_from_table(
    slot_key: wp.array[wp.Int],
    slot_counts: wp.array[wp.int32],
    occupied_scan: wp.array[wp.int32],
    out_keys: wp.array[wp.Int],
    out_counts: wp.array[wp.int32],
    out_perm: wp.array[wp.int32],
) -> None:
    # ``occupied_scan`` is ``hash_insert``'s 0/1 occupancy, scanned *in place* and inclusively, so
    # a slot is occupied exactly where the scan steps and its compact position is the exclusive
    # value (``array.scanned_count``) -- no second table-sized buffer for the scan, and the scan's
    # last element reads ``n_unique`` outright.
    #
    # ``out_perm`` is the identity permutation the radix sort pairs with the keys. Writing it here
    # replaces an ``arange`` launch of its own, which cost about as much as the sort it feeds. Only
    # the leading ``n_unique`` entries are written; the rest of the buffer is the sort's
    # double-buffer scratch, which it fills before reading. ``out_counts`` may be a null
    # descriptor (``None`` at the launch) for a caller that does not want the counts.
    h = wp.int32(wp.tid())
    pos, occupied = kernel_array.scanned_count(occupied_scan, h)
    if occupied != 0:
        out_keys[pos] = decode_key(slot_key[h])
        if out_counts.shape[0] > 0:
            out_counts[pos] = slot_counts[h]
        out_perm[pos] = pos


@wp.func
def bucket_float32(value: wp.float32) -> wp.uint32:
    # Bit-cast float32 to uint32 and drop the low 11 mantissa bits, so each key names a bucket
    # roughly 2^-12 wide *relative* to the value's magnitude.
    #
    # The sign bit is the most significant bit and survives the shift, so opposite signs never
    # share a bucket. That is what you want everywhere except at zero, where IEEE-754 has two
    # representations that compare equal: -0.0 has to fold onto +0.0 or a point sitting exactly on
    # an axis will not match itself. A revolved sphere's pole is the standard way to hit this,
    # since ``cos(theta) * 0.0`` is -0.0 for half the slices.
    if value == 0.0:
        return wp.uint32(0)
    return wp.cast(value, wp.uint32) >> VEC3_PACK_SHIFT


@wp.func
def pack_vec3(vector: wp.vec3) -> wp.uint64:
    ix = bucket_float32(vector[0])
    iy = bucket_float32(vector[1])
    iz = bucket_float32(vector[2])

    # Promote each component to uint64 before shifting, to avoid 32-bit overflow in the large
    # left-shifts (<< 21 and << 42).
    return wp.uint64(ix) | (
        (wp.uint64(iy) << VEC3_PACK_PRECISION)
        | (wp.uint64(iz) << (VEC3_PACK_PRECISION + VEC3_PACK_PRECISION))
    )


@wp.kernel
def pack_directed_index_keys(
    indices: wp.array2d[wp.int32], base: wp.uint64, out_keys: wp.array[wp.uint64]
) -> None:
    # ``pack_indices`` for exactly two columns, taking the radix directly instead of inferring it.
    # Row order is preserved, so ``(a, b)`` and ``(b, a)`` get different keys -- which is the point
    # wherever a directed edge has to be told from its twin.
    i = wp.int32(wp.tid())
    out_keys[i] = kernel_array.pack_directed_key(indices[i, 0], indices[i, 1], base)


@wp.kernel
def pack_undirected_edge_keys(
    edges: wp.array2d[wp.int32], base: wp.uint64, out_keys: wp.array[wp.uint64]
) -> None:
    # Deliberately not ``pack_indices``: that packs a row in the order it is given, and these keys
    # are compared against ``array.pack_edge_key``, which sorts. A caller's reversed row would
    # hash to something no halfedge can produce, so the edge would be silently missed.
    i = wp.int32(wp.tid())
    out_keys[i] = kernel_array.pack_edge_key(edges[i, 0], edges[i, 1], base)


@wp.func
def pack_index_digit(
    packed: wp.uint64, power: wp.uint64, value: wp.int32, max_index: wp.uint64
) -> tuple[wp.uint64, wp.uint64]:
    # One digit of the mixed-radix row key: ``value`` (reinterpreted as ``uint32``) times the
    # running ``power`` of the radix, and the power for the next digit. Every row packing in this
    # module accumulates through it, so the keys agree bit for bit whatever the caller's layout.
    return packed + wp.uint64(wp.uint32(value)) * power, power * max_index


@wp.kernel
def pack_indices(
    indices: wp.array2d[wp.int32], max_index: wp.uint64, out_packed: wp.array[wp.uint64]
) -> None:
    tid = wp.int32(wp.tid())
    indices_row = indices[tid]
    packed_value = wp.uint64(0)
    power = wp.uint64(1)
    for i in range(indices_row.shape[0]):
        packed_value, power = pack_index_digit(packed_value, power, indices_row[i], max_index)

    out_packed[tid] = packed_value


@wp.kernel
def pack_sorted_face_keys(
    faces: wp.array[wp.int32], max_index: wp.uint64, out_packed: wp.array[wp.uint64]
) -> None:
    # A face's orientation-free key, ``array.pack_triangle_key``: its three corners in ascending
    # order, packed as ``pack_indices`` packs a three-column row. Sorting in registers and packing
    # in the same thread drops the ``(n, 3)`` table of sorted rows a separate sort launch would
    # write for this one to read.
    f = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, f)
    out_packed[f] = pack_triangle_key(a, b, c, max_index)


@wp.kernel
def round_vec3_scaled(
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    inv_epsilon: wp.float32,
    out_rounded: wp.array2d[wp.int32],
) -> None:
    # Snapping relative to `origin` -- the data's own minimum corner -- rather than to the
    # coordinate origin does two things. The cell indices come out non-negative, which the row
    # packing needs since it treats a row as digits in a positive radix. And the product stays the
    # size of the data's *extent* instead of its distance from zero: float32 carries about 7 digits,
    # so scaling a coordinate near 100 by 1e6 has already quantised away the low bits.
    tid = wp.int32(wp.tid())
    v = (vertices[tid] - origin) * inv_epsilon
    out_rounded[tid, 0] = wp.int32(wp.round(v[0]))
    out_rounded[tid, 1] = wp.int32(wp.round(v[1]))
    out_rounded[tid, 2] = wp.int32(wp.round(v[2]))


# Concrete overloads, registered at import -- rationale in ``triwarp/kernels/reduce.py``, rule in
# CLAUDE.md section 2.5.
#
# These take the caller's *key* dtype, and the two sets differ because the two call paths do.
# ``mark_group_starts`` is reached from ``grouping.group``, which widens through
# ``twt.sortable_dtype`` and is handed ``wp.uint64`` row hashes by ``hash_indices_rows`` -- so it
# needs the full unsigned surface ``triwarp.array``'s search kernels see. The open-addressing pair
# does not: their only caller is ``grouping._unique_hash``, whose ``data_int`` parameter is typed
# ``wp.array[wp.int32] | wp.array[wp.int64]`` because ``array.bitcast_to_int`` reinterprets every
# key into one *signed* space before the table sees it. The unsigned rows were unreachable, and
# section 2.5's rule is to register what the wrapper's dispatch can reach.
_KEY_DTYPES = (wp.int32, wp.int64, wp.uint32, wp.uint64)
_TABLE_DTYPES = (wp.int32, wp.int64)


# The concrete handles keyed by the caller's key dtype -- see
# [`OverloadTable`][triwarp.kernels.array.OverloadTable]. Worth a few percent to fifteen percent on
# this module's own consumers, since each of them issues two or three generic launches.
MARK_GROUP_STARTS: OverloadTable
HASH_INSERT: OverloadTable
COMPACT_FROM_TABLE: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global MARK_GROUP_STARTS, HASH_INSERT, COMPACT_FROM_TABLE
    MARK_GROUP_STARTS = OverloadTable(
        mark_group_starts,
        {d: [wp.array[d], wp.int32, wp.int32, wp.array[wp.int32]] for d in _KEY_DTYPES},
    )
    HASH_INSERT = OverloadTable(
        hash_insert,
        {
            d: [wp.array[d], wp.array[d], wp.array[wp.int32], wp.int32, wp.array[wp.int32]]
            for d in _TABLE_DTYPES
        },
    )
    # ``slot_key`` and ``out_keys`` carry the key dtype; every count/offset buffer is int32.
    COMPACT_FROM_TABLE = OverloadTable(
        compact_from_table,
        {
            d: [
                wp.array[d],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[d],
                wp.array[wp.int32],
                wp.array[wp.int32],
            ]
            for d in _TABLE_DTYPES
        },
    )


_register_overloads()
