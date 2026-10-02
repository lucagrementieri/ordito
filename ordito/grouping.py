"""Index grouping by equal value/row, plus value and row deduplication for Warp arrays."""

from __future__ import annotations

from typing import Any, Literal, TypeVar, cast, overload

import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar
from ordito.array import bitcast_from_int, bitcast_to_int, gather, sort_pair_indices
from ordito.constants import INDEX_RADIX_PAIR
from ordito.kernels import array as kernel_array
from ordito.kernels import grouping as kernel_grouping

# The element dtype of ``unique_1d``'s input, so the checker gives its unique values the same
# one: ``wp.array[wp.float32]`` in, ``wp.array[wp.float32]`` out. A static name only; which
# dtypes the call accepts at runtime is in its docstring.
DType = TypeVar("DType")


def group(values: wp.array[wp.Int], length: int) -> odt.Array2dInt32:
    """
    Return index groups of exactly ``length`` entries that share the same value.

    ``values`` are radix-sorted with their original indices; each output row lists
    ``length`` indices whose corresponding entries are equal and form a run of
    precisely that size (runs shorter or longer than ``length`` are omitted). This
    matches [`trimesh.grouping.group`][] with ``min_len == max_len == length`` and
    [`trimesh.grouping.group_rows`][] with ``require_count=length``.

    Parameters
    ----------
    values
        ``(n,)`` device array of integer keys (any ``wp.Int`` dtype).
    length
        Required run length; each returned group contains exactly this many indices.

    Returns
    -------
    odt.Array2dInt32
        ``(g, length)`` array on ``values.device`` where ``g`` is the number of
        groups found. Empty when no run has exactly ``length`` equal neighbors.

    See Also
    --------
    [`group_int_rows`][ordito.grouping.group_int_rows]
    """
    n = values.size
    device = values.device
    if n < length or length <= 0:
        return odt.as_array2d(odt.empty_2d((0, max(length, 0)), wp.int32, device=device), wp.int32)

    sort_dtype = odt.sortable_dtype(values.dtype)
    values_buffer = _launch.empty(2 * n, dtype=sort_dtype, device=device)
    if sort_dtype == values.dtype:
        _launch.copy(values_buffer, values, count=n)
    else:
        wp.utils.array_cast(values, values_buffer, count=n)
    indices_buffer = sort_pair_indices(n, -1, device)
    _launch.radix_sort_pairs(values_buffer, indices_buffer, count=n)

    # Scan compaction: flag run starts, scan the flags, then emit one right-sized row per group
    # (deterministic ascending-value order, no atomic counter and no (n, length) over-allocation).
    # The emit reads each flag back as a step in the scan, so it compacts and writes in one launch.
    # The flags live in the sort payload's upper half, which is free scratch once the sort has run
    # (``adjacency``'s pair grouping does the same), and are scanned there in place.
    inclusive = odt.as_dense(indices_buffer[n:])
    _launch.launch(
        kernel_grouping.MARK_GROUP_STARTS[values_buffer.dtype],
        dim=n,
        inputs=[values_buffer, wp.int32(n), wp.int32(length), inclusive],
        device=device,
    )
    _launch.array_scan(inclusive, inclusive, inclusive=True)
    # The group count sizes the output, so it has to come back to the host.
    n_groups = int(read_scalar(inclusive))
    groups = odt.empty_2d((n_groups, length), wp.int32, device=device)
    if n_groups > 0:
        _launch.launch(
            kernel_grouping.emit_groups,
            dim=n,
            inputs=[inclusive, indices_buffer, groups],
            device=device,
        )
    return odt.as_array2d(groups, wp.int32)


def group_int_rows(
    data: odt.Array2dInt, length: int, max_value: int | None = None, *, validate: bool = True
) -> odt.Array2dInt32:
    """
    Return index groups of exactly ``length`` rows that are identical.

    Each row is hashed with [`hash_indices_rows`][ordito.grouping.hash_indices_rows], then
    [`group`][ordito.grouping.group] finds runs of ``length`` equal keys in sorted order. For
    example, ``[[1, 2], [3, 4], [1, 2]]`` with ``length=2`` yields one group ``[[0, 2]]`` (same as
    [`trimesh.grouping.group_rows`][] with ``require_count=2``).

    Parameters
    ----------
    data
        ``(n, w)`` device array of non-negative ``int32`` row values.
    length
        Required number of duplicate rows per group.
    max_value
        Optional exclusive upper bound on entries and radix for row hashing; passed
        through to [`hash_indices_rows`][ordito.grouping.hash_indices_rows] as ``max_index``. If
        ``None``, inferred from ``max(data) + 1`` -- or, with ``validate=False`` on a row of at most
        two columns, taken as
        [`constants.INDEX_RADIX_PAIR`][ordito.constants.INDEX_RADIX_PAIR] with no reduction.
    validate
        Forwarded to [`hash_indices_rows`][ordito.grouping.hash_indices_rows]. ``False`` skips the
        range-check reduction (and its host readback); it requires ``max_value`` for a row of more
        than two columns. See the warning there before using it.

    Returns
    -------
    odt.Array2dInt32
        ``(g, length)`` array on ``data.device`` with original row indices per group.

    See Also
    --------
    [`group`][ordito.grouping.group]
    """
    odt.ensure_ndim(data, 2)
    hashed_rows = hash_indices_rows(data, max_value, validate=validate)
    return group(hashed_rows, length)


@overload
def unique_1d(
    data: wp.array[DType],
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
    max_value: int | None = None,
) -> wp.array[DType]: ...
@overload
def unique_1d(
    data: wp.array[DType],
    *,
    return_inverse: Literal[True],
    return_counts: Literal[False] = False,
    max_value: int | None = None,
) -> tuple[wp.array[DType], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[DType],
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[True],
    max_value: int | None = None,
) -> tuple[wp.array[DType], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[DType],
    *,
    return_inverse: Literal[True],
    return_counts: Literal[True],
    max_value: int | None = None,
) -> tuple[wp.array[DType], wp.array[wp.int32], wp.array[wp.int32]]: ...
def unique_1d(
    data: wp.array[DType],
    *,
    return_inverse: bool = False,
    return_counts: bool = False,
    max_value: int | None = None,
) -> (
    wp.array[DType]
    | tuple[wp.array[DType], wp.array[wp.int32]]
    | tuple[wp.array[DType], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Find sorted unique elements of a 1D Warp array (``numpy.unique`` subset).

    Uses an open-addressing hash table: O(n) inserts plus an O(n_unique log n_unique) sort of only
    the unique values.

    ``return_inverse`` maps each input position to the index of its value in the sorted unique
    output.
    ``return_index`` is not supported.

    Parameters
    ----------
    data
        ``(n,)`` array with any scalar ``dtype``.
    return_inverse
        If ``True``, include the inverse mapping in the return tuple.
    return_counts
        If ``True``, include per-unique occurrence counts in the return tuple.
    max_value
        Optional inclusive upper bound on the values of an integer ``data`` whose values are all
        non-negative -- a packed key's known range. The final sort then orders only the low bits
        that bound needs, which is cheaper and leaves the result unchanged. It is trusted, not
        checked: a value above it, or a negative one, sorts out of order.

    Returns
    -------
    unique : wp.array
        ``(n_unique,)`` sorted unique values (same ``dtype`` and ``device`` as ``data``).
    inverse : wp.array[wp.int32], optional
        ``(n,)``, present if ``return_inverse=True``. ``inverse[i]`` indexes ``unique`` such
        that conceptually ``unique[inverse[i]] == data[i]`` (float ``NaN`` matches
        NumPy: all ``NaN`` values share one slot).
    counts : wp.array[wp.int32], optional
        ``(n_unique,)``, present if ``return_counts=True``. Number of occurrences per ``unique``
        row.

    Raises
    ------
    ValueError
        If ``data`` is not rank-1 or length is ``> 2**30``, or if ``max_value`` is negative or
        given for a floating-point ``data``.
    """
    if int(data.ndim) != 1:
        raise ValueError(f"unique_1d expects a rank-1 array, got ndim={data.ndim}")
    end_bit = None
    if max_value is not None:
        if max_value < 0:
            raise ValueError(f"max_value must be non-negative, got {max_value}")
        if data.dtype in (wp.float16, wp.float32, wp.float64):
            raise ValueError("max_value applies to integer data only")
        end_bit = max(1, int(max_value).bit_length())

    device = data.device
    n = data.size

    if n == 0:
        empty_unique = _launch.empty(0, dtype=data.dtype, device=device)
        empty_i32 = _launch.empty(0, dtype=wp.int32, device=device)
        return _pack_unique_result(
            empty_unique,
            inverse=empty_i32 if return_inverse else None,
            counts=empty_i32 if return_counts else None,
        )
    if n > (1 << 30):
        raise ValueError(f"unique_1d requires length <= 2**30, got length {n}")

    if data.dtype in _SORTED_UNIQUE_DTYPES:
        unique, inverse, counts = _unique_sorted(data, n, return_inverse, return_counts, end_bit)
        return _pack_unique_result(unique, inverse=inverse, counts=counts)
    mask = wp.int32(kernel_grouping.hash_table_mask(n))

    # ``_unique_hash`` only ever *reads* the integer key array, so when the input already is one of
    # the two key dtypes the bit reinterpretation is the identity and the buffer can be shared --
    # skipping a full-length copy and its allocation. A four- or eight-byte dtype (the ``uint64``
    # row keys every ``edges_unique`` call packs, ``float32``, ...) is the same bits under another
    # name, so it is shared too, through a zero-copy ``view``: ``bitcast_to_int`` would produce
    # byte-identical keys in a fresh buffer. Only a narrower dtype needs the real conversion.
    key_bytes = wp.types.type_size_in_bytes(data.dtype)
    if data.dtype in (wp.int32, wp.int64):
        data_int = data
    elif key_bytes in (4, 8):
        data_int = data.view(wp.int32 if key_bytes == 4 else wp.int64)
    else:
        data_int = bitcast_to_int(cast("wp.array[Any]", data), n)
    return _unique_hash(data, data_int, data.dtype, n, mask, return_inverse, return_counts, end_bit)


# The dtypes ``unique_1d`` deduplicates by one radix sort of the values: every integer type Warp's
# sort orders directly. Equal values are one run of the sort, so marking each run's first position,
# scanning the marks and one emit give the sorted unique values, the inverse and the counts. The
# hash table the other dtypes take has ``~2 n`` slots updated by random atomics where the sort
# streams, and its probes serialize on a hot slot when few values repeat many times.
_SORTED_UNIQUE_DTYPES = (wp.int32, wp.int64, wp.uint32, wp.uint64)


def _unique_sorted(
    data: wp.array[Any], n: int, return_inverse: bool, return_counts: bool, end_bit: int | None
) -> tuple[wp.array[Any], wp.array[wp.int32] | None, wp.array[wp.int32] | None]:
    """
    [`unique_1d`][ordito.grouping.unique_1d] by one radix sort of an integer ``data``.

    Returns ``(unique, inverse, counts)``, each optional one ``None`` unless asked for. The keys and
    the identity payload are written into the leading halves of the sort's double-width buffers,
    and ``_unique_sorted_runs`` does the rest. ``n > 0``.
    """
    device = data.device
    dtype = data.dtype
    keys = _launch.empty(2 * n, dtype=dtype, device=device)
    order = _launch.empty(2 * n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_grouping.KEYS_AND_IDENTITY[dtype], dim=n, inputs=[data, keys, order], device=device
    )
    unique, inverse, counts, _first = _unique_sorted_runs(
        keys, order, n, True, return_inverse, return_counts, False, end_bit
    )
    assert unique is not None
    return unique, inverse, counts


def _unique_hash(
    data: wp.array[Any],
    data_int: wp.array[Any],
    original_dtype: type[Any],
    n: int,
    mask: wp.int32,
    return_inverse: bool,
    return_counts: bool,
    end_bit: int | None,
) -> (
    wp.array[Any]
    | tuple[wp.array[Any], wp.array[wp.int32]]
    | tuple[wp.array[Any], wp.array[wp.int32], wp.array[wp.int32]]
):
    key_dtype = data_int.dtype
    device = data_int.device

    # Phase 1: parallel insert into the open-addressing hash table.
    slot_key, slot_counts, occupied = _hash_insert(data_int, n, mask)
    cap = slot_key.size

    # Phase 2: prefix-scan the occupancy in place to get compact positions -- nothing reads the
    # flags afterwards except the compaction, which recovers each one as a step of the scan.
    _launch.array_scan(occupied, occupied, inclusive=True)
    # An inclusive scan of 0/1 flags ends at the number set, so one 4-byte tail read sizes the
    # output where ``reduce.max`` would scan all ``cap`` (~2n) slots. The same idiom as
    # ``array.flatnonzero`` and ``array.counts_to_offsets``.
    n_unique = int(read_scalar(occupied))

    # Phase 3: compact unique keys, their occurrence counts, and the identity permutation the sort
    # below pairs with them -- all three in one pass over the table.
    #
    # ``keys_compact`` is allocated at the *sort's* double width and the kernel writes its leading
    # half, so the radix sort below can ping-pong in this same buffer. Sizing it to ``n_unique``
    # and widening afterwards means ``bitcast_from_int`` allocates the double-width buffer and
    # copies the keys into it -- an allocation and a full copy, to move bytes that could have been
    # written here in the first place.
    sort_dtype = cast("type[Any]", odt.sortable_dtype(original_dtype))
    keys_compact = _launch.empty(2 * n_unique, dtype=key_dtype, device=device)
    counts_compact = (
        _launch.empty(n_unique, dtype=wp.int32, device=device) if return_counts else None
    )
    perm_buf = _launch.empty(2 * n_unique, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_grouping.COMPACT_FROM_TABLE[key_dtype],
        dim=cap,
        inputs=[slot_key, slot_counts, occupied, keys_compact, counts_compact, perm_buf],
        device=device,
    )

    # Phase 4: sort only the n_unique keys (typically n_unique << n), in a dtype that orders them
    # the way the caller's dtype does rather than by their reinterpreted bit pattern. Reading the
    # keys under that dtype is a reinterpreting *view* whenever it is the same width as the integer
    # they were compacted as -- which every key dtype this package packs takes -- and a real
    # widening copy only for a caller whose dtype is too narrow for Warp to sort at all.
    keys_buf = (
        keys_compact.view(sort_dtype)
        if wp.types.type_size_in_bytes(sort_dtype) == wp.types.type_size_in_bytes(key_dtype)
        else bitcast_from_int(keys_compact, sort_dtype, count=2 * n_unique)
    )
    _launch.radix_sort_pairs(keys_buf, perm_buf, count=n_unique, end_bit=end_bit)

    if sort_dtype == original_dtype:
        # The sorted prefix of the sort's own scratch, handed back as a view rather than copied out:
        # nothing else holds that buffer, so the view is the answer's sole owner, and a copy would
        # be an allocation and a full pass to move bytes already where the caller wants them.
        unique_values = odt.as_dense(keys_buf[:n_unique])
    else:
        # A dtype narrower than 32 bits was widened to be sortable; narrow it back.
        unique_values = bitcast_from_int(
            bitcast_to_int(keys_buf, n_unique), original_dtype, count=n_unique
        )

    unique_counts = None
    if counts_compact is not None:
        # A contiguous prefix slice, not a gather-unsafe strided view (CLAUDE.md §3.4) -- no copy
        # needed before handing it to ``gather`` as the index array.
        unique_counts = gather(counts_compact, odt.as_dense(perm_buf[:n_unique]))

    unique_inverse = None
    if return_inverse:
        if sort_dtype == original_dtype:
            # ``unique_values`` is already this exact buffer -- the same ``keys_buf`` prefix, under
            # the caller's dtype, which the sort dtype *is* on this branch. The common key dtypes
            # here (the ``uint64`` edge and row keys every ``edges_unique`` call packs) all take it.
            sorted_dense = unique_values
        else:
            sorted_dense = _launch.empty(n_unique, dtype=sort_dtype, device=device)
            _launch.copy(sorted_dense, keys_buf, count=n_unique)
        # The binary search has to probe in the same space the keys were sorted in.
        data_sorted_space = (
            data if data.dtype == sort_dtype else bitcast_from_int(data_int, sort_dtype, count=n)
        )
        unique_inverse = _launch.empty(n, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_array.MAP_SORTED_INVERSE[data_sorted_space.dtype],
            dim=n,
            inputs=[data_sorted_space, sorted_dense, unique_inverse],
            device=device,
        )

    return _pack_unique_result(unique_values, inverse=unique_inverse, counts=unique_counts)


def hashed_occurrence_counts(keys: wp.array[wp.uint64] | wp.array[wp.int64]) -> wp.array[wp.int32]:
    """
    Occurrence count of every distinct 64-bit key, in no particular order.

    The counting half of [`unique_1d`][ordito.grouping.unique_1d]'s hash table, handed back as
    the table itself: one slot per table entry, holding how many keys landed in it, and zero for
    an empty slot. That answers any question about the *multiset* of counts -- is every key
    present exactly twice -- without the compaction, the sort and the host read of the distinct
    count that sizing them needs.

    Parameters
    ----------
    keys
        ``(n,)`` ``wp.uint64`` or ``wp.int64`` keys, ``n <= 2**30``.

    Returns
    -------
    wp.array[wp.int32]
        ``(m,)`` per-slot counts, ``m`` somewhat over ``2 * n``, on ``keys.device``. Their
        nonzero entries are the counts of the distinct keys; which slot holds which key is
        unspecified.

    See Also
    --------
    [`unique_1d`][ordito.grouping.unique_1d]
    """
    n = keys.size
    return _hash_insert(keys.view(wp.int64), n, wp.int32(kernel_grouping.hash_table_mask(n)))[1]


def _hash_insert(
    keys: wp.array[wp.int32] | wp.array[wp.int64], n: int, mask: wp.int32
) -> tuple[wp.array[Any], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Insert ``n`` keys into a fresh table; return its ``(slot_key, slot_counts, occupied)``.

    One slot past the ``mask + 1`` table is reserved for the single key that collides with the
    empty-slot sentinel; see the comment on ``kernel_grouping.hash_insert``. ``occupied`` is stamped
    by the insert itself rather than derived from ``slot_counts`` in a second pass, so it is
    zero-filled rather than ``wp.empty``; it shares one zero-filled allocation with
    ``slot_counts``, the two ``int32`` tables being adjacent halves of it.
    """
    cap = int(mask) + 2
    device = keys.device
    slot_key = _launch.zeros(cap, dtype=keys.dtype, device=device)
    counts_and_occupied = _launch.zeros(2 * cap, dtype=wp.int32, device=device)
    slot_counts = odt.as_dense(counts_and_occupied[:cap])
    occupied = odt.as_dense(counts_and_occupied[cap:])
    _launch.launch(
        kernel_grouping.HASH_INSERT[keys.dtype],
        dim=n,
        inputs=[keys, slot_key, slot_counts, mask, occupied],
        device=device,
    )
    return slot_key, slot_counts, occupied


@overload
def unique_rows(
    data: wp.array[wp.vec3],
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
) -> wp.array[wp.vec3]: ...
@overload
def unique_rows(
    data: wp.array[wp.vec3], *, return_inverse: Literal[True], return_counts: Literal[False] = False
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: wp.array[wp.vec3], *, return_inverse: Literal[False] = False, return_counts: Literal[True]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: wp.array[wp.vec3], *, return_inverse: Literal[True], return_counts: Literal[True]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: odt.Array2dInt32,
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
) -> odt.Array2dInt32: ...
@overload
def unique_rows(
    data: odt.Array2dInt32, *, return_inverse: Literal[True], return_counts: Literal[False] = False
) -> tuple[odt.Array2dInt32, wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: odt.Array2dInt32, *, return_inverse: Literal[False] = False, return_counts: Literal[True]
) -> tuple[odt.Array2dInt32, wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: odt.Array2dInt32, *, return_inverse: Literal[True], return_counts: Literal[True]
) -> tuple[odt.Array2dInt32, wp.array[wp.int32], wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: odt.Array2dFloat32,
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
) -> odt.Array2dFloat32: ...
@overload
def unique_rows(
    data: odt.Array2dFloat32,
    *,
    return_inverse: Literal[True],
    return_counts: Literal[False] = False,
) -> tuple[odt.Array2dFloat32, wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: odt.Array2dFloat32,
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[True],
) -> tuple[odt.Array2dFloat32, wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: odt.Array2dFloat32, *, return_inverse: Literal[True], return_counts: Literal[True]
) -> tuple[odt.Array2dFloat32, wp.array[wp.int32], wp.array[wp.int32]]: ...
def unique_rows(
    data: odt.ArrayNd, *, return_inverse: bool = False, return_counts: bool = False
) -> (
    odt.ArrayNd
    | tuple[odt.ArrayNd, wp.array[wp.int32]]
    | tuple[odt.ArrayNd, wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Find unique rows of a 2D Warp array (``numpy.unique`` along axis 0).

    Each row is hashed with [`hash_rows`][ordito.grouping.hash_rows], then deduplicated via the
    same open-addressing hash table as [`unique_1d`][ordito.grouping.unique_1d]. Unique rows are
    returned in
    sorted hash-key order; the representative row for each key is the first input
    row in that equivalence class.

    ``return_inverse`` maps each input row index to its slot in the sorted unique
    output. ``return_index`` is not supported.

    Parameters
    ----------
    data
        ``(n, w)`` ``int32`` or ``float32`` array, or ``(n,)`` ``wp.vec3`` array.
    return_inverse
        If ``True``, include the inverse mapping in the return tuple.
    return_counts
        If ``True``, include per-unique occurrence counts in the return tuple.

    Returns
    -------
    unique
        ``(n_unique, w)`` or ``(n_unique,)`` ``wp.vec3`` array on ``data.device``.
    inverse : wp.array[wp.int32], optional
        ``(n,)``, present if ``return_inverse=True``. ``inverse[i]`` indexes ``unique``.
    counts : wp.array[wp.int32], optional
        ``(n_unique,)``, present if ``return_counts=True``. Occurrences per ``unique`` row.

    Raises
    ------
    ValueError
        If ``data`` is not rank-2 ``int32``/``float32`` or rank-1 ``wp.vec3``.
    """
    device = data.device
    n = int(data.shape[0])
    is_vec3 = data.dtype == wp.vec3

    if n == 0:
        if is_vec3:
            empty_unique = _launch.empty(0, dtype=wp.vec3, device=device)
        else:
            # Same dtype check ``hash_rows`` performs on the non-empty path below, so a caller who
            # only ever exercises this function on empty input still gets the documented raise
            # rather than a silently wrong output dtype.
            odt.ensure_ndim(data, 2)
            if data.dtype == wp.int32:
                empty_unique = odt.empty_2d((0, int(data.shape[1])), wp.int32, device=device)
            elif data.dtype == wp.float32:
                empty_unique = odt.empty_2d((0, int(data.shape[1])), wp.float32, device=device)
            else:
                raise ValueError(f"unique_rows unsupported dtype {data.dtype}")
        empty_i32 = _launch.empty(0, dtype=wp.int32, device=device)
        return _pack_unique_result(
            empty_unique,
            inverse=empty_i32 if return_inverse else None,
            counts=empty_i32 if return_counts else None,
        )

    inverse, first_idx, counts = _unique_keys_core(hash_rows(data), return_counts=return_counts)

    if not is_vec3:
        odt.ensure_ndim(data, 2)
    # ``gather`` performs both rank-1 (vec3) and rank-2 (int32/float32 row) gather.
    unique_rows_out = gather(data, first_idx)

    if not return_inverse:
        inverse = None
    return _pack_unique_result(unique_rows_out, inverse=inverse, counts=counts)


@overload
def unique_faces(
    faces: wp.array[wp.int32],
    *,
    return_inverse: Literal[False] = False,
    max_index: int | None = None,
) -> wp.array[wp.int32]: ...
@overload
def unique_faces(
    faces: wp.array[wp.int32], *, return_inverse: Literal[True], max_index: int | None = None
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]: ...
def unique_faces(
    faces: wp.array[wp.int32], *, return_inverse: bool = False, max_index: int | None = None
) -> wp.array[wp.int32] | tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Find unique triangular faces up to vertex permutation (orientation-agnostic).

    Two faces are equal when they share the same three vertices regardless of order. Each
    face's indices are sorted before deduplication; the representative returned for a class is
    the first occurring face, with its original vertex order preserved.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer (common 1D format).
    return_inverse
        If ``True``, also return the inverse mapping from each input face to its slot in the
        unique output.
    max_index
        Exclusive upper bound on every vertex index, typically the vertex count. When given it is
        trusted rather than checked, which skips the range-checking reduction over ``faces`` and
        its host readback -- only pass it where the bound holds by construction, since a bound
        below the true maximum makes distinct faces collide silently. When ``None`` (default) the
        indices are validated and the bound inferred from them.

    Returns
    -------
    unique_faces : wp.array[wp.int32]
        ``(3 * n_unique,)`` flat buffer of the ``n_unique`` unique faces.
    inverse : wp.array[wp.int32], optional
        ``(n_faces,)``, present if ``return_inverse=True``; ``inverse[i]`` is the unique
        slot of input face ``i``.

    Raises
    ------
    ValueError
        If ``max_index`` is not positive, or -- when it is ``None`` -- if ``faces`` holds a negative
        index.
    """
    if max_index is not None and max_index <= 0:
        raise ValueError(f"max_index must be positive, got {max_index}")
    device = faces.device
    n_faces = faces.size // 3
    if n_faces == 0:
        empty_faces = _launch.empty(0, dtype=wp.int32, device=device)
        if return_inverse:
            return empty_faces, _launch.empty(0, dtype=wp.int32, device=device)
        return empty_faces

    if max_index is None:
        # Sorting a face's corners does not change the buffer's extremes, so the range check and
        # the inferred radix read the input directly. The reduction ends in a host readback.
        min_index, max_value = od.reduce.minmax(faces)
        if min_index < 0:
            raise ValueError(f"faces must be non-negative, got a minimum of {min_index}")
        max_index = max_value + 1
    # The orientation-free face keys and the identity payload, written straight into the leading
    # halves of the radix sort's double-width buffers.
    keys = _launch.empty(2 * n_faces, dtype=wp.uint64, device=device)
    order = _launch.empty(2 * n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_grouping.pack_sorted_face_keys_and_order,
        dim=n_faces,
        inputs=[faces, wp.uint64(max_index), keys, order],
        device=device,
    )
    # Not ``unique_rows`` on the sorted rows: that gathers the sorted rows by the first-occurrence
    # indices to build its own return, an answer this function has no use for -- it gathers the
    # unsorted faces by the same indices instead, to keep each representative's original winding.
    # Three sorted indices below ``max_index`` pack below ``max_index ** 3``.
    end_bit = max(1, (min(max_index**3, 1 << 64) - 1).bit_length())
    _unique, inverse, _counts, first = _unique_sorted_runs(
        keys, order, n_faces, False, return_inverse, False, True, end_bit
    )
    assert first is not None
    unique_faces_out = gather(faces.reshape((-1, 3)), first).reshape((-1,))
    if return_inverse:
        assert inverse is not None
        return unique_faces_out, inverse
    return unique_faces_out


def _unique_keys_core(
    row_keys: wp.array[wp.uint64], *, return_counts: bool
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32] | None]:
    """
    Shared core of [`unique_rows`][ordito.grouping.unique_rows]: dedup row keys, find first index.

    Returns ``(inverse, first_idx, counts)``: ``unique_1d``'s sorted path, which reads each class's
    first occurrence off its stable sort in the launch that writes the inverse -- no
    ``first_occurrence_indices`` scatter after it -- and writes no unique key buffer, since
    [`unique_rows`][ordito.grouping.unique_rows] gathers ``data`` itself by ``first_idx``.
    Requires ``row_keys.shape[0] > 0``.
    """
    n = row_keys.size
    keys = _launch.empty(2 * n, dtype=wp.uint64, device=row_keys.device)
    order = _launch.empty(2 * n, dtype=wp.int32, device=row_keys.device)
    _launch.launch(
        kernel_grouping.KEYS_AND_IDENTITY[wp.uint64],
        dim=n,
        inputs=[row_keys, keys, order],
        device=row_keys.device,
    )
    _unique, inverse, counts, first_idx = _unique_sorted_runs(
        keys, order, n, False, True, return_counts, True, None
    )
    assert inverse is not None
    assert first_idx is not None
    return inverse, first_idx, counts


def _unique_sorted_runs(
    keys: wp.array[Any],
    order: wp.array[wp.int32],
    n: int,
    return_values: bool,
    return_inverse: bool,
    return_counts: bool,
    return_first: bool,
    end_bit: int | None,
) -> tuple[
    wp.array[Any] | None,
    wp.array[wp.int32] | None,
    wp.array[wp.int32] | None,
    wp.array[wp.int32] | None,
]:
    """
    Sort the double-width ``keys`` / ``order`` buffers and emit their runs of equal keys.

    The leading ``n`` entries of each hold the keys and the identity permutation. Returns
    ``(unique, inverse, counts, first)``, each ``None`` unless asked for; ``first`` is
    [`first_occurrence_indices`][ordito.grouping.first_occurrence_indices] of the inverse, read off
    the stable sort. The run marks and their scan live in the payload's upper half, which is free
    scratch once the sort has run. A producer that can write its keys straight into the sort's
    buffers (``unique_faces``) skips ``_unique_sorted``'s copy. ``n > 0``.
    """
    device = keys.device
    dtype = keys.dtype
    _launch.radix_sort_pairs(keys, order, count=n, end_bit=end_bit)
    ranks = odt.as_dense(order[n:])
    _launch.launch(
        kernel_grouping.MARK_SORTED_RUN_STARTS[dtype], dim=n, inputs=[keys, ranks], device=device
    )
    _launch.array_scan(ranks, ranks, inclusive=True)
    # The distinct count sizes the outputs.
    n_unique = int(read_scalar(order, 2 * n - 1))
    unique_values = _launch.empty(n_unique, dtype=dtype, device=device) if return_values else None
    inverse = _launch.empty(n, dtype=wp.int32, device=device) if return_inverse else None
    starts = _launch.empty(n_unique + 1, dtype=wp.int32, device=device) if return_counts else None
    first = _launch.empty(n_unique, dtype=wp.int32, device=device) if return_first else None
    _launch.launch(
        kernel_grouping.EMIT_SORTED_UNIQUE[dtype],
        dim=n,
        inputs=[keys, order, ranks, unique_values, inverse, starts, first],
        device=device,
    )
    counts = None
    if starts is not None:
        counts = _launch.empty(n_unique, dtype=wp.int32, device=device)
        _launch.launch(
            kernel_grouping.run_lengths, dim=n_unique, inputs=[starts, counts], device=device
        )
    return unique_values, inverse, counts, first


def first_occurrence_indices(
    inverse: wp.array[wp.int32], n_unique: int | None = None
) -> wp.array[wp.int32]:
    """
    Index of the first element of each equivalence class in an ``inverse`` map.

    The representative-picking half of every deduplication in this package: given the
    ``inverse`` that ``unique_1d`` / ``unique_rows`` return, produce for each class the smallest
    input index belonging to it. Gathering any per-element payload by the result yields one
    representative per class, taken from its **first** occurrence -- which is what makes
    ``unique_faces`` keep the original winding and ``edges_unique`` keep the first-seen edge.

    Parameters
    ----------
    inverse
        ``(n,)`` map from each element to its class slot, as returned by
        [`unique_1d`][ordito.grouping.unique_1d] or
        [`unique_rows`][ordito.grouping.unique_rows].
    n_unique
        Number of classes (the output length). Defaults to ``int(reduce.max(inverse)) + 1``,
        which costs a device reduction **and** a host synchronization; pass it whenever the
        caller already knows it. It is almost always a ``.shape[0]`` the caller is holding --
        the length of the unique array that came back beside ``inverse``.

    Returns
    -------
    wp.array[wp.int32]
        ``(n_unique,)`` array whose entry ``c`` is ``min{i : inverse[i] == c}``. A class with
        no member (only reachable when ``n_unique`` is passed too large) holds the sentinel ``n``.

    See Also
    --------
    [`unique_rows`][ordito.grouping.unique_rows]
    [`unique_1d`][ordito.grouping.unique_1d]
    [`gather`][ordito.array.gather]
        The companion step: gather the payload by these indices to get the representatives.
    """
    device = inverse.device
    n = inverse.size
    if n_unique is None:
        n_unique = int(od.reduce.max(inverse)) + 1 if n > 0 else 0
    first = _launch.full(n_unique, n, dtype=wp.int32, device=device)
    if n > 0 and n_unique > 0:
        _launch.launch(
            kernel_grouping.scatter_first_occurrence, dim=n, inputs=[inverse, first], device=device
        )
    return first


@overload
def hash_rows(data: wp.array[wp.vec3]) -> wp.array[wp.uint64]: ...
@overload
def hash_rows(data: odt.Array2dInt32) -> wp.array[wp.uint64]: ...
@overload
def hash_rows(data: odt.Array2dFloat32) -> wp.array[wp.uint64]: ...
def hash_rows(
    data: wp.array[wp.vec3] | odt.Array2dInt32 | odt.Array2dFloat32,
) -> wp.array[wp.uint64]:
    """
    Pack each row of a 2D or ``wp.vec3`` array into a single ``uint64`` key.

    Dispatches to [`hash_vector_rows`][ordito.grouping.hash_vector_rows] for ``wp.vec3``
    (including ``float32`` arrays with width 3) or
    [`hash_indices_rows`][ordito.grouping.hash_indices_rows] for ``int32`` rows.

    Parameters
    ----------
    data
        ``(n, w)`` ``int32`` or ``float32`` array, or ``(n,)`` ``wp.vec3`` array.

    Returns
    -------
    wp.array[wp.uint64]
        ``(n,)`` array on ``data.device`` with one packed key per row.

    Raises
    ------
    ValueError
        If ``data`` has an unsupported dtype or shape.
    """
    if data.dtype == wp.vec3:
        return hash_vector_rows(cast("wp.array[wp.vec3]", data))
    odt.ensure_ndim(data, 2)
    if data.dtype == wp.int32:
        return hash_indices_rows(cast("odt.Array2dInt32", data))
    if data.dtype == wp.float32:
        n = int(data.shape[0])
        if int(data.shape[1]) != 3:
            raise ValueError("float32 hash_rows currently requires width 3")
        if data.is_contiguous:
            # The ``(n, 3)`` rows *are* ``wp.vec3`` values in memory, so a zero-copy view reads
            # them as such; the copy below is only for a strided table, which a view cannot retype.
            return hash_vector_rows(data.view(wp.vec3))
        vec = _launch.empty(n, dtype=wp.vec3, device=data.device)
        wp.utils.array_cast(cast("wp.array[wp.float32]", data), vec)
        return hash_vector_rows(vec)
    raise ValueError(f"hash_rows unsupported dtype {data.dtype}")


def hash_vector_rows(data: wp.array[wp.vec3], epsilon: float = 0.0) -> wp.array[wp.uint64]:
    """
    Pack each ``wp.vec3`` row into a single ``uint64`` key.

    Two schemes, selected by ``epsilon``:

    - ``epsilon > 0.0`` — an **absolute** tolerance. Each coordinate is snapped to a multiple of
      ``epsilon`` measured from the data's own minimum corner, and the resulting integer row is
      packed via [`hash_indices_rows`][ordito.grouping.hash_indices_rows]. Equal keys mean the rows
      landed in the same grid cell. Snapping relative to the minimum corner rather than to the
      coordinate origin is what lets negative coordinates work at all, and it keeps the scaled
      values at the size of the data's extent, where ``float32`` still resolves ``epsilon``.
    - ``epsilon == 0.0`` — a **relative** bucket. Each coordinate's ``float32`` bits are
      right-shifted by 11, so a key names an interval roughly ``2 ** -12`` (about ``2.4e-4``) wide
      relative to the coordinate's own magnitude. This is *not* an equality test in either
      direction, and callers that need one should pass an explicit ``epsilon``: see Notes.

    Parameters
    ----------
    data
        ``(n,)`` device array of values.
    epsilon
        Uniqueness tolerance. ``0`` uses the relative bit-truncation bucket; positive values snap
        coordinates to ``round(v / epsilon)`` before packing.

    Returns
    -------
    wp.array[wp.uint64]
        ``(n,)`` array on ``data.device`` with one packed key per row.

    Raises
    ------
    ValueError
        If ``data`` is not a ``wp.array[wp.vec3]``.

    Notes
    -----
    Both schemes quantize, so both split a pair that straddles a cell boundary no matter how close
    the two values are. For ``epsilon > 0`` the packing is additionally injective only while
    ``radix ** 3`` fits a ``uint64``, where ``radix`` is the widest per-axis extent in cells; beyond
    that the row keys wrap and behave as a hash with a small collision probability rather than an
    exact cell identity. The relative scheme additionally:

    - **collides distinct coordinates** that share a bucket — ``1.0`` and ``1.000244`` produce the
      same key, so a mesh whose vertex spacing is below ``2.4e-4`` relative is merged too
      aggressively;
    - has *unbounded* resolution approaching zero, so ``+1e-6`` and ``-1e-6`` are thousands of
      buckets apart (correct — they are distinct points), but so are ``+1e-40`` and ``-1e-40``,
      which for most purposes are the same point. Only ``+0.0`` and ``-0.0`` are folded together,
      because IEEE-754 defines them as equal.

    Neither is a defect of the packing so much as the nature of a fixed-width key; pass an explicit
    ``epsilon`` when the tolerance has to be one you chose.

    See Also
    --------
    [`hash_indices_rows`][ordito.grouping.hash_indices_rows]
    [`hash_rows`][ordito.grouping.hash_rows]
    """
    if data.dtype != wp.vec3:
        raise ValueError(f"data must be a wp.array[wp.vec3], got wp.array[{data.dtype}]")
    n = data.size
    if epsilon > 0.0:
        if n == 0:
            return _launch.empty(0, dtype=wp.uint64, device=data.device)
        # `hash_indices_rows` packs each row as digits in a positive radix, so negative cell indices
        # -- which every mesh spanning the origin produces -- cannot be packed. Snapping relative to
        # the data's own minimum corner makes them non-negative *by construction*: subtracting the
        # true minimum cannot give a negative result, so no validation pass or shift is needed. The
        # bounds also supply the radix, and doing it per component keeps that radix as small as the
        # widest single extent rather than the whole diagonal, which matters because the row packing
        # is only injective while ``radix ** 3`` fits a ``uint64``.
        min_bound, max_bound = od.bounds.aabb(data)
        rounded = odt.empty_2d((n, 3), wp.int32, device=data.device)
        _launch.launch(
            kernel_grouping.round_vec3_scaled,
            dim=n,
            inputs=[data, min_bound, wp.float32(1.0 / epsilon), rounded],
            device=data.device,
        )
        extent = max(float(max_bound[c]) - float(min_bound[c]) for c in range(3)) / epsilon
        # The device rounds a float32 product where this divides in float64; the relative slack plus
        # the half-cell of rounding covers the difference, and an over-wide radix is harmless.
        radix = int(extent * (1.0 + 1e-6)) + 3
        return hash_indices_rows(rounded, max_index=radix, validate=False)
    hashes = _launch.empty(data.size, dtype=wp.uint64, device=data.device)
    _launch.map(kernel_grouping.pack_vec3, data, out=hashes)
    return hashes


def hash_indices_rows(
    data: odt.Array2dInt32, max_index: int | None = None, *, validate: bool = True
) -> wp.array[wp.uint64]:
    """
    Pack each row of non-negative ``int32`` values into a single ``uint64`` key.

    Each row is treated as digits in base ``max_index`` (or ``max(data)+1`` when
    ``max_index`` is omitted) and combined into one integer key per row. A global
    min/max over ``data`` (via ``ordito.reduce.minmax``) validates the range and
    supplies the radix when it is inferred.

    Parameters
    ----------
    data
        ``(n, w)`` device array; every entry must be non-negative. If
        ``max_index`` is given, every entry must satisfy ``entry < max_index``.
    max_index
        Optional exclusive upper bound on entries and radix for packing; must be
        positive when provided. If ``None``, set to ``max(data) + 1`` after validation -- or, when
        ``validate`` is ``False`` and ``data`` has at most two columns, to
        [`constants.INDEX_RADIX_PAIR`][ordito.constants.INDEX_RADIX_PAIR], which bounds every
        ``int32`` and so needs no reduction at all.
    validate
        When ``True`` (default), a ``ordito.reduce.minmax`` over ``data`` checks that entries are
        non-negative and below ``max_index``. That reduction ends in a host readback, which
        serialises the device pipeline — measurably so when this is called once per pass inside a
        remeshing loop. Pass ``False``, together with an explicit ``max_index``, to skip it when the
        bound is already guaranteed by construction (mesh edge rows are built from face indices, so
        they are non-negative and below the vertex count by definition).

        For a row of **two** columns ``max_index`` may then be left ``None`` as well, which packs
        against [`constants.INDEX_RADIX_PAIR`][ordito.constants.INDEX_RADIX_PAIR] and removes the
        bound-inferring reduction too -- the right spelling wherever the count exists only to be
        this radix. It is not available for a wider row, whose radix has to keep ``radix ** w``
        inside a ``uint64``.

        **A caller that derived its ``max_index`` from
        [`index_bound`][ordito.array.index_bound] has *not* thereby made this
        redundant.** ``index_bound`` is a ``max`` reduction; the validation is a ``minmax``, and
        it is the ``min`` half -- the negative-index guard -- that has no counterpart above it, so
        skipping it would turn a malformed face buffer from a raise into a silently wrong
        grouping. So ``edges.edges_unique`` validates by default, and the callers that pass
        ``False`` are the ones whose bound *and* non-negativity are structural.

    Returns
    -------
    wp.array[wp.uint64]
        ``(n,)`` array on ``data.device`` with one packed key per row.

    Raises
    ------
    ValueError
        If ``max_index`` is not positive, if ``validate=False`` is passed without a ``max_index``
        for a row of more than two columns, or -- when validating -- if ``data`` is negative or
        reaches ``max_index``.

    Warnings
    --------
    ``validate=False`` with a ``max_index`` smaller than the true maximum silently produces
    colliding keys, and therefore wrong groupings, rather than raising. Only use it where the bound
    is structurally guaranteed.

    The mixed-radix key is injective only while ``max_index ** w`` fits a ``uint64``; past that the
    positional sum wraps and equal keys no longer imply equal rows. Two 32-bit columns always fit,
    but three columns need ``max_index <= 2 ** (64 / 3)``, about ``2.6e6``. Beyond that the keys
    degrade from an exact row identity into a hash whose collision probability grows with the square
    of the row count -- around ``2e-5`` for 28M rows at a radix of ``1.4e7``. This is deliberately
    not validated: raising would reject meshes that group correctly in practice.

    See Also
    --------
    [`hash_vector_rows`][ordito.grouping.hash_vector_rows]
    [`hash_rows`][ordito.grouping.hash_rows]
    """
    odt.ensure_ndim(data, 2, dtype=wp.int32)
    if max_index is not None and max_index <= 0:
        raise ValueError(f"max_index must be positive, got {max_index}")
    n = int(data.shape[0])
    width = int(data.shape[1])
    if not validate:
        if max_index is None:
            if width > 2:
                raise ValueError(
                    "validate=False requires an explicit max_index (the radix to use) for a row "
                    f"wider than two columns, got width {width}."
                )
            # Two columns need no bound at all: every ``int32`` reinterpreted as ``uint32`` is
            # below ``INDEX_RADIX_PAIR``, so packing against it is injective without reducing the
            # data, and it orders rows exactly as a tighter radix would.
            max_index = INDEX_RADIX_PAIR
    elif n > 0:
        # The min/max is a device reduction with a host readback, so it serialises the pipeline.
        # It is unavoidable when the radix has to be inferred, and skippable via validate=False
        # when the caller already knows the bound. Skipped for an empty ``data`` -- there is
        # nothing to validate, and ``reduce.minmax`` itself raises on an empty array.
        min_data, max_data = od.reduce.minmax(data)
        if min_data < 0:
            raise ValueError(f"data must be non-negative, got a minimum of {min_data}")
        if max_index is not None and max_data >= max_index:
            raise ValueError(
                f"data must be less than max_index {max_index}, got a maximum of {max_data}"
            )
        if max_index is None:
            max_index = max_data + 1
    hashes = _launch.empty(n, dtype=wp.uint64, device=data.device)
    if n > 0:
        assert max_index is not None  # set by the bound above whenever ``n > 0``
        _launch.launch(
            kernel_grouping.pack_indices,
            dim=n,
            inputs=[data, wp.uint64(max_index), hashes],
            device=data.device,
        )
    return hashes


def sorted_undirected_edge_keys(edges: odt.Array2dInt32, n_vertices: int) -> wp.array[wp.uint64]:
    """
    Sorted packed keys of an undirected edge set, for binary-search membership testing.

    Each row packs to the same key regardless of which endpoint comes first, so a halfedge's own
    ``(origin, destination)`` key -- built the same way from either order -- can be tested for
    membership against the result with a binary search, without caring which order a caller's row
    was given in.

    Parameters
    ----------
    edges
        ``(k, 2)`` vertex-index pairs, in either order per row.
    n_vertices
        Total vertex count, used as the packing radix.

    Returns
    -------
    wp.array[wp.uint64]
        ``(k,)`` sorted keys on ``edges.device``. Empty when ``edges`` is empty.
    """
    device = edges.device
    n_edges = int(edges.shape[0])
    if n_edges == 0:
        return _launch.empty(0, dtype=wp.uint64, device=device)
    # The keys are packed straight into the leading half of the radix sort's double-width buffer,
    # and the payload the sort insists on is left uninitialized: nothing reads the permutation, so
    # seeding it with an identity (``sort_and_argsort``) and staging the keys in a buffer of their
    # own would be an allocation, a fill and a copy for an answer that is thrown away.
    keys = _launch.empty(2 * n_edges, dtype=wp.uint64, device=device)
    _launch.launch(
        kernel_grouping.pack_undirected_edge_keys,
        dim=n_edges,
        inputs=[edges, wp.uint64(n_vertices), keys],
        device=device,
    )
    # Every key is below ``n_vertices ** 2``, so the sort orders only that many low bits.
    _launch.radix_sort_pairs(
        keys,
        _launch.empty(2 * n_edges, dtype=wp.int32, device=device),
        count=n_edges,
        end_bit=max(1, (n_vertices * n_vertices - 1).bit_length()),
    )
    return odt.as_dense(keys[:n_edges])


def _pack_unique_result(
    unique: odt.ArrayNd,
    *,
    inverse: wp.array[wp.int32] | None = None,
    counts: wp.array[wp.int32] | None = None,
) -> (
    odt.ArrayNd
    | tuple[odt.ArrayNd, wp.array[wp.int32]]
    | tuple[odt.ArrayNd, wp.array[wp.int32], wp.array[wp.int32]]
):
    """Assemble the ``(unique[, inverse][, counts])`` return tuple shared by the unique_* family."""
    if inverse is not None and counts is not None:
        return unique, inverse, counts
    if inverse is not None:
        return unique, inverse
    if counts is not None:
        return unique, counts
    return unique
