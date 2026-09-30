"""Kernels for assembling meshes from parts (``triwarp.combine``)."""

import warp as wp

from triwarp.kernels import array as kernel_array
from triwarp.kernels.array import sorted_run_start


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
def label_run_starts(sorted_labels: wp.array[wp.int32], out_is_start: wp.array[wp.bool]) -> None:
    # Segment boundaries of a label-sorted array: position 0, plus every label change, plus the
    # end -- ``out_is_start`` is one longer than the labels, so its ``flatnonzero`` is the
    # total-terminated offsets directly. One launch over the whole buffer, where the adjacent-
    # element map it replaces needed two shifted views, an output view and a separate write for
    # position 0 -- and a guard for the single-element buffer, which ``sorted_run_start``'s own
    # ``i == 0`` test makes unnecessary.
    i = wp.int32(wp.tid())
    out_is_start[i] = sorted_run_start(sorted_labels, i)
    if i == sorted_labels.shape[0] - 1:
        out_is_start[i + 1] = True
