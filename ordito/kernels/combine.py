"""Kernels for assembling meshes from parts (``ordito.combine``)."""

import warp as wp

from ordito.kernels import array as kernel_array
from ordito.kernels.array import sorted_run_start

wp.set_module_options({"enable_backward": False})


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
