"""Kernels for assembling meshes from parts (``ordito.combine``)."""

import warp as wp

from ordito.kernels import array as kernel_array
from ordito.kernels.algorithms.connected_components import find_representative

wp.set_module_options({"enable_backward": False})


@wp.struct
class MeshPieceWords:
    """One buffer of one ``concatenate`` piece, its source and target addressed as 4-byte words."""

    source: wp.array[wp.int32]
    target: wp.array[wp.int32]
    offset: wp.int32
    count: wp.int32
    shift: wp.int32


@wp.kernel
def pack_mesh_pieces(pieces: wp.array[MeshPieceWords], width: wp.int32) -> None:
    # Pack every piece's vertices and faces into the two concatenated buffers and renumber the
    # faces, one launch for all of it: each piece contributes two records, its vertex words (shift
    # 0) and its face indices (shift = the piece's vertex offset). ``array.pack_segment_words`` with
    # a destination and a shift per record, so one table serves both outputs; same grid and stride.
    s, t = wp.tid()
    piece = pieces[s]
    kernel_array.copy_shifted_words(
        piece.source, piece.offset, piece.count, piece.shift, t, width, piece.target
    )


@wp.kernel
def offset_packed_faces(
    piece_starts: wp.array[wp.int32], vertex_offsets: wp.array[wp.int32], faces: wp.array[wp.int32]
) -> None:
    # Renumber a packed face buffer in place: every index of piece ``p`` shifts by that piece's
    # cumulative vertex offset. One launch over all indices, so the *number of pieces* costs
    # nothing here — the owning piece is found by an upper-bound search over its start offsets
    # (``piece_starts[0]`` is 0, so the search never returns 0). Pieces contributing no faces share
    # a start with their successor; the upper bound skips them, which is the right answer.
    i = wp.int32(wp.tid())
    piece = kernel_array.binary_search_index(piece_starts, i) - 1
    faces[i] = faces[i] + vertex_offsets[piece]


@wp.func
def store_label_root_and_identity(
    label: wp.int32, f: wp.int32, out_keys: wp.array[wp.int32], out_order: wp.array[wp.int32]
):
    # One face's entry of ``split_with_offsets``' component sort, written into the radix sort's
    # double-width buffers: its label as the key, its id as the payload, and in the keys' upper
    # half (the sort's scratch, free until the sort runs) whether it is its component's root. A
    # label is its component's smallest face id, so the scanned root flags number the components
    # in label order.
    out_keys[f] = label
    out_order[f] = f
    out_keys[out_keys.shape[0] // 2 + f] = wp.where(label == f, wp.int32(1), wp.int32(0))


@wp.kernel
def flatten_labels_and_identity(
    parents: wp.array[wp.int32], out_keys: wp.array[wp.int32], out_order: wp.array[wp.int32]
) -> None:
    # ``connected_components.ecl_flatten`` storing each face's component label as
    # ``store_label_root_and_identity`` lays it out: no label buffer is filled to be copied there.
    # ``labels_and_identity`` is the same entry from a caller's labels.
    f = wp.int32(wp.tid())
    store_label_root_and_identity(find_representative(parents, f), f, out_keys, out_order)


@wp.kernel
def labels_and_identity(
    labels: wp.array[wp.int32], out_keys: wp.array[wp.int32], out_order: wp.array[wp.int32]
) -> None:
    # ``flatten_labels_and_identity`` for labels the caller passed in.
    f = wp.int32(wp.tid())
    store_label_root_and_identity(labels[f], f, out_keys, out_order)


@wp.kernel
def component_rank_keys(keys: wp.array[wp.int32]) -> None:
    # Replace each face's label key by its component's number: the inclusive scan of the root
    # flags in the keys' upper half, read at the root (the label), less one. The number is
    # monotone in the label, so the sort's order is the label sort's, over ``key_bits(k)`` bits
    # instead of ``key_bits(n_faces)``. The reads stay in the upper half and the writes in the
    # lower one.
    f = wp.int32(wp.tid())
    keys[f] = keys[keys.shape[0] // 2 + keys[f]] - 1


@wp.kernel
def component_face_offsets(
    sorted_components: wp.array[wp.int32], out_offsets: wp.array[wp.int32]
) -> None:
    # The total-terminated face offsets of the component-sorted faces: component ``c`` starts at
    # the first entry not below ``c``, and ``c == k`` reads ``n_faces``.
    c = wp.int32(wp.tid())
    out_offsets[c] = kernel_array.binary_search_index_left(sorted_components, c)
