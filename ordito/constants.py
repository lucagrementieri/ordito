"""
Shared numeric tolerances, sentinel values and tuning constants.

Everything here is imported from this module rather than defined next to its first user, so that
a value two modules share does not make one of them import the other. Each name is spelled twice
where both scopes need it: a plain Python number for host-scope arithmetic and a ``wp.<dtype>(...)``
twin (the ``_CONSTANT`` / ``_F64`` suffixes) for the kernel scope, which fixes the precision the
value enters kernel arithmetic at.
"""

import warp as wp

TOLERANCE_MERGE = 1e-8
TOLERANCE_PLANAR = 1e-5
TOLERANCE_ZERO = 1e-12

# Intrinsic mollification margin, as a fraction of the **mean** edge length (not the longest --
# one long edge on a graded mesh would otherwise inflate the perturbation past anything a
# degenerate face needs): the triangle inequality is satisfied with this much slack rather than
# exactly, so a barely-valid triangle still yields a finite cotangent weight. Shared by
# `laplacian.mollify_intrinsic` and `remesh.intrinsic_delaunay`.
#
# The value is bracketed on both sides and neither bound is slack. Too *small* and the single
# global constant added to every length falls under a float32 length's own resolution, so no
# stored length changes, the degenerate face stays degenerate and its couplings are still dropped
# from the operator -- a silent no-op, which conditioning cannot reveal because a dropped face
# contributes zero and so looks perfectly conditioned. Too *large* and that same global constant
# perturbs the clean part of the mesh away from the true cotangent operator. ``1e-5`` sits more
# than a decade from each arm; the bracket is measured in
# `kernels/laplacian.triangle_inequality_slack`.
TOLERANCE_MOLLIFY = 1e-5

# Default tolerances of [`allclose`][ordito.array.allclose], shared with the per-thread closure
# test of ``polyline``'s kernels ("the last point repeats the first"): "close" has to mean one thing
# across the package. The kernel twins are read in kernel scope, so a closure test costs no launch
# arguments.
ALLCLOSE_RTOL = 1e-05
ALLCLOSE_ATOL = 1e-08

TOLERANCE_MERGE_CONSTANT = wp.float32(TOLERANCE_MERGE)
TOLERANCE_PLANAR_CONSTANT = wp.float32(TOLERANCE_PLANAR)
TOLERANCE_ZERO_CONSTANT = wp.float32(TOLERANCE_ZERO)
ALLCLOSE_RTOL_CONSTANT = wp.float32(ALLCLOSE_RTOL)
ALLCLOSE_ATOL_CONSTANT = wp.float32(ALLCLOSE_ATOL)
TOLERANCE_ZERO_F64 = wp.float64(TOLERANCE_ZERO)
"""``TOLERANCE_ZERO`` at ``float64``, for the predicates that widen before deciding."""

# Largest representable values, usable as "sorts last" sentinels inside kernels (computing them
# here in Python scope avoids the literals Warp cannot evaluate at kernel scope).
INT32_MAX = 2**31 - 1
INT32_MAX_CONSTANT = wp.int32(INT32_MAX)
INT64_MAX = 2**63 - 1
UINT64_MAX = 2**64 - 1
UINT64_MAX_CONSTANT = wp.uint64(UINT64_MAX)
FLOAT32_INF_CONSTANT = wp.INF
# The float64 twin, for a kernel that widens float32 geometry to make its *decisions* in double --
# ``kernels/intersection.triangles_intersect`` is the one that does. ``wp.INF`` is a float32
# constant, and mixing it into float64 arithmetic does not parse.
FLOAT64_INF_CONSTANT = wp.float64(float("inf"))

PI = wp.PI
TWO_PI = 2 * wp.PI

TILE_1D = 64
TILE_2D = 8

# Tiles folded per block by the *global* (``axis=None``) 1-D reductions in ``kernels/reduce.py``.
# One block per tile means one ``atomic_add`` per 64 elements, and at scale that single accumulator
# address is the bottleneck rather than bandwidth. Folding several tiles into a register first
# divides the atomic traffic by this factor before the final commit.
#
# The CPU device pays for this: ``wp.launch_tiled`` runs one lane per block there, so folding 16
# tiles means 16x fewer blocks and correspondingly less parallelism. Accepted anyway because the
# CUDA gain is large and the CPU cost shrinks with input size.
TILES_PER_BLOCK_1D = 16

# Elements reduced per thread by the *lane-free* reductions -- those that partition the **outer**
# work the grid is over, so there is no block-owned sequence for the lanes to share and no
# ``wp.block_dim()`` to stride by. Each thread walks a strided slice of its input and commits one
# atomic, so this trades launch width against atomic traffic.
#
# **The stride's source is the rule, not the tile.** A lane-parallel body is correct on both devices
# exactly when its stride is ``wp.block_dim()`` -- which reads 1 on the CPU device, where
# ``wp.launch_tiled`` runs one lane per block, so that lane covers the whole sequence. Striding by
# a *kernel argument* instead is wrong on **both** devices: the CPU answer comes up short by the
# stride, and CUDA double-counts whenever the argument does not match the block width. See
# ``kernels/visibility.py::obscurance`` for a lane-parallel kernel that is correct on both devices
# because it strides by ``wp.block_dim()``.
#
# The four consumers of this constant do not fit a block-per-item shape instead: that form only
# pays where the *outer* dimension alone starves the device, and the whole point of a slice
# dimension is that it does not.
#
# The optimum splits by device, so there are two values and
# [`items_per_slice`][ordito._device.items_per_slice] picks between them; do not read either
# directly. CUDA wants long slices; CPU wants short ones. A *per-query* reduction wants a much
# longer slice on both devices -- the query dimension already fills the device -- and sets its own
# value locally (see ``proximity.ITEMS_PER_QUERY_SLICE``).
ITEMS_PER_SLICE_CUDA = 128
ITEMS_PER_SLICE_CPU = 32

# Radix for packing a row of **two** ``int32`` index columns into one ``uint64`` key when no
# tighter bound is known. Every ``int32`` reinterpreted as ``uint32`` is below ``2 ** 32``, so
# ``row[0] + row[1] * 2 ** 32`` is injective for any pair without reducing the data to find its
# maximum first -- and it stays inside ``uint64``, which two 32-bit columns always do.
#
# It packs in the same order a tighter radix does: the key is monotone lexicographic in
# ``(row[1], row[0])`` for every radix above the largest entry, so a wider one leaves the sorted
# row order of ``grouping.unique_1d`` unchanged. That is what lets a caller whose bound is only
# ever the packing's radix (and not, say, the length of an output buffer) skip the reduction
# entirely -- measured on ``grouping.hash_indices_rows``, packing against this radix is several
# times cheaper than inferring the tight bound first.
#
# Three or more columns cannot use it: ``radix ** w`` has to fit ``uint64``, which caps a
# three-column radix near ``2.6e6``.
INDEX_RADIX_PAIR = 1 << 32
