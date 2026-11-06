"""Kernels for assembling meshes from parts (``ordito.combine``)."""

import warp as wp

from ordito.kernels import array as kernel_array
from ordito.kernels.algorithms.connected_components import find_representative
from ordito.kernels.array import sorted_run_start

wp.set_module_options({"enable_backward": False})


@wp.struct
class MeshPieceWords:
    """One buffer of one ``concatenate`` piece, its source and target addressed as 4-byte words."""

    source: wp.array[wp.int32]
    target: wp.array[wp.int32]
    offset: wp.int32
    count: wp.int32
    shift: wp.int32


@wp.kernel(module="unique")
def pack_mesh_pieces(pieces: wp.array[MeshPieceWords], width: wp.int32) -> None:
    # Pack every piece's vertices and faces into the two concatenated buffers and renumber the
    # faces, one launch for all of it: each piece contributes two records, its vertex words (shift
    # 0) and its face indices (shift = the piece's vertex offset). ``array.pack_segment_words`` with
    # a destination and a shift per record, so one table serves both outputs; same grid and stride.
    # A module of its own: beside ``offset_packed_faces`` it changed that kernel's CPU build, and
    # the CPU path below 32 pieces, which still launches it, ran 1.07-1.12x.
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


@wp.kernel
def flatten_labels_and_identity(
    parents: wp.array[wp.int32], out_keys: wp.array[wp.int32], out_order: wp.array[wp.int32]
) -> None:
    # ``connected_components.ecl_flatten`` writing each face's component label straight into the
    # leading half of the radix sort's double-width key buffer, beside the identity payload, for
    # ``split_with_offsets``' label sort: no label buffer is filled to be copied there by
    # ``array.keys_and_identity``.
    f = wp.int32(wp.tid())
    out_keys[f] = find_representative(parents, f)
    out_order[f] = f


@wp.kernel
def label_run_starts(sorted_labels: wp.array[wp.int32], out_flags: wp.array[wp.int32]) -> None:
    # Segment boundaries of a label-sorted array as ``0`` / ``1`` flags: position 0, plus every
    # label change, plus the end -- ``out_flags`` is one longer than the labels, so the indices of
    # its ones are the total-terminated offsets directly, and the flags are written as ``int32``
    # straight into the buffer ``array.flatnonzero_in_place`` scans, with no mask between them.
    # ``sorted_run_start``'s own ``i == 0`` test covers the single-element buffer.
    i = wp.int32(wp.tid())
    out_flags[i] = wp.where(sorted_run_start(sorted_labels, i), wp.int32(1), wp.int32(0))
    if i == sorted_labels.shape[0] - 1:
        out_flags[i + 1] = wp.int32(1)
