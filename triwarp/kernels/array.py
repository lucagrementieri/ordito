from collections.abc import Mapping, Sequence
from typing import Any

import warp as wp

from triwarp.constants import INT32_MAX_CONSTANT

# Slot table for the **device-side round loop**: one zero-initialized ``wp.array[wp.int32]`` that a
# ``dim=1`` kernel updates at the end of each round, so ``wp.capture_while`` can drive the rounds
# with no host readback. Slot 0 counts rounds -- read against a cap, which is what bounds a loop
# whose progress test could otherwise stall -- and slot 1 is the condition; the wrapper hands
# ``wp.capture_while`` the view ``state[LOOP_CONDITION_VIEW]``.
#
# The condition is written by a **plain store**, never an atomic: it is one address taking one
# value, so there is nothing to serialize even where every thread of a wide launch may write it
# (``polyline.rdp_split_spans``, ``homology.forest_link``). **It must be seeded non-zero before the
# loop starts**: ``wp.capture_while`` evaluates it *before* the first round, so a plain
# ``wp.zeros`` state runs zero rounds. Seed with ``wp.array([0, 1])`` / ``assign([0, 1])``, or from
# the kernel that resets the rest of the pass (``remesh.begin_decimation_pass``).
#
# Six loops share this, in two shapes. **Arm-at-the-front**: a round's first kernel clears the
# condition and a later one raises it, so two slots are enough -- the level-synchronous
# Ramer-Douglas-Peucker and the ear-clipping rounds in ``kernels/polyline.py``, the conjugate
# gradient's iteration test, and ``kernels/remesh.py``'s collapse rounds, which needs a third slot
# for the previous round's commit count and **appends** it so the shared two keep their numbers.
# **Publish-at-the-back**: the round raises ``LOOP_PROGRESS`` and a closing ``loop_advance`` turns
# it into the condition, which is what a loop with a *round cap* needs, since the cap is only known
# once the round index has been stepped -- ``graph.shortest_path_envelope``'s relaxation passes and
# both of ``kernels/homology.py``'s (the breadth-first level loop and the Boruvka forest rounds).
# ``kernels/algorithms/bfs.py`` deliberately shares neither: its seven slots are a frontier window
# (``start``, ``tail``) rather than a round counter, so slot 0 does not mean the same thing.
LOOP_ROUND = wp.constant(wp.int32(0))
LOOP_CONDITION = wp.constant(wp.int32(1))
LOOP_STATE_SIZE = 2
# Appended by the publish-at-the-back shape, so the shared two keep their numbers.
LOOP_PROGRESS = wp.constant(wp.int32(2))
LOOP_ADVANCE_STATE_SIZE = 3

# The length-1 slot views the wrappers take, as **plain** slices, which is a measured cost rather
# than tidiness: slicing a ``wp.array`` with the ``wp.int32`` constants above routes the bound
# arithmetic through Warp's Python-scope builtin dispatch, an order of magnitude dearer than the
# identical view taken with plain ints, and every round loop in the package took one or two per
# call. Both bounds have to be unwrapped -- ``__getitem__`` forms ``stop - start`` and
# ``strides * start`` itself, so one Warp-typed bound is three dispatches. Derived from the
# constants rather than written out, so the two cannot drift. See ``.claude/CLAUDE.md`` section
# 13.1.
LOOP_CONDITION_VIEW = slice(int(LOOP_CONDITION), int(LOOP_CONDITION) + 1)
LOOP_PROGRESS_VIEW = slice(int(LOOP_PROGRESS), int(LOOP_PROGRESS) + 1)


@wp.kernel
def loop_advance(max_rounds: wp.int32, out_state: wp.array[wp.int32]) -> None:
    # dim=1. Close a device-side round: step the round index, publish whether another round should
    # run, and re-arm the progress flag for the next one. Host-side bookkeeping moved onto the
    # device so the whole loop is one ``wp.capture_while`` graph with no per-round readback.
    #
    # The progress flag has to be *cleared here* rather than by the round's own first kernel,
    # because ``wp.capture_while`` reads the condition after the body -- so a round's claim and the
    # test of that claim must sit in the same body, which is what makes this the closing kernel and
    # not the opening one. Seed ``LOOP_CONDITION`` non-zero, since the condition is read before the
    # first round as well.
    #
    # ``max_rounds`` bounds a loop whose progress test is already a sound termination argument on
    # its own; it is cheap insurance, because a captured loop that fails that argument hangs the
    # device rather than returning a wrong answer.
    #
    # **This is not free.** It replaced three bespoke ``dim=1`` advance kernels and runs inside the
    # *replayed* body of every captured round loop, so its one extra compare and select over the
    # three-statement form each of those had costs a fraction of a percent of a whole call. The
    # merge was taken with that known -- three near-identical bookkeeping kernels drifting apart is
    # the more expensive failure -- but a caller adding a *hot* round loop should price this against
    # a bespoke advance first. Carrying the stepped round index in a local rather than re-reading
    # ``out_state[LOOP_ROUND]`` for the cap test was tried and measured flat, so the plainer
    # spelling stays; nvcc forwards the store.
    out_state[LOOP_ROUND] = out_state[LOOP_ROUND] + 1
    keep_going = out_state[LOOP_PROGRESS] != 0 and out_state[LOOP_ROUND] < max_rounds
    out_state[LOOP_CONDITION] = wp.where(keep_going, wp.int32(1), wp.int32(0))
    out_state[LOOP_PROGRESS] = 0


@wp.func
def sort3(a: wp.Scalar, b: wp.Scalar, c: wp.Scalar) -> tuple[wp.Scalar, wp.Scalar, wp.Scalar]:
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return a, b, c


@wp.func
def sign_with_tolerance(value: wp.Float, tolerance: wp.Float) -> wp.int32:
    # Sign of ``value`` with a dead zone of half-width ``tolerance`` around zero, which reads 0.
    #
    # **The dead zone is an argument and not a constant on purpose, and every pipeline that uses
    # this twice must use the same one both times.** A plane cut asks the question in two places --
    # which side is each *vertex* on, and does the plane cross each *edge*'s interior -- and the two
    # answers have to agree or the classifier flags a face the edge pass does not split. Splitting
    # it into a fixed-tolerance variant for the classifiers and this one for the edge mask puts that
    # coupling beyond the reach of a reader of either.
    #
    # The classifiers pass ``TOLERANCE_MERGE_CONSTANT`` because their public entry points
    # (``slice_mesh_with_plane``, ``clip_mesh_with_field``, ``split_faces_along_field``) expose no
    # tolerance and, per CLAUDE.md section 4.2, should not grow one until a caller needs it;
    # ``split_mesh_with_plane`` documents a ``tolerance=`` and passes it through. Both spellings are
    # visible at the call site, which is the whole point.
    if value < -tolerance:
        return wp.int32(-1)
    if value > tolerance:
        return wp.int32(1)
    return wp.int32(0)


@wp.func
def wrap_index(i: wp.int32, n: wp.int32) -> wp.int32:
    # Positive modulo: ``%`` follows C++11 semantics (sign of the dividend).
    return ((i % n) + n) % n


@wp.func
def loop_point(k: wp.int32, n: wp.int32) -> wp.int32:
    # Point index of entry ``k`` in ``[0, n]`` of a closed loop's ``n + 1`` entries, the last being
    # the first point again -- ``wrap_index`` for an index known to be at most ``n``, as a select
    # rather than two modulos (an integer division each). ``k + 1`` for ``k`` in ``[0, n)`` is the
    # usual caller.
    return wp.where(k < n, k, wp.int32(0))


@wp.func
def ravel_index(i: wp.int32, j: wp.int32, k: wp.int32, ny: wp.int32, nz: wp.int32) -> wp.int32:
    # Row-major flat index into an (nx, ny, nz) box, ``nx`` implicit in ``i``'s own range. Shared by
    # ``voxels.flat_cell_index`` (a box) and ``reconstruction.poisson_grid_index`` (the cube
    # specialization ``ny == nz == res``) -- one formula, two domain names for readability at each
    # call site.
    return (i * ny + j) * nz + k


@wp.func
def atomic_min_packed_box(
    out_corners: wp.array[wp.float32], box: wp.int32, lower: wp.vec3, upper: wp.vec3
) -> None:
    # Reduce one axis-aligned box into the six ``float32`` slots at ``box``, packed
    # ``[min_x, min_y, min_z, -max_x, -max_y, -max_z]``: *negating* the upper half is what lets a
    # single ``wp.full(inf)`` seed both ends and makes every update one ``wp.atomic_min``.
    #
    # A layout convention rather than an arithmetic run, which is why it is named here in the leaf
    # library and not at any one of its writers -- ``bounds.oriented_box_extents`` (one box per
    # candidate frame), ``reduce.minmax_vec3_chunked`` (the whole cloud into box 0) and
    # ``scatter.scatter_group_bounds`` (one box per face group, which passes a single point as both
    # corners). Three writers of one packing that several readers then decode is exactly the shape
    # that goes silently wrong when one copy drifts: a transposed or un-negated half still runs and
    # still returns a plausible box. Same reason ``ravel_index`` above is shared.
    #
    # The reader is [`packed_box_sides`][triwarp.kernels.bounds.packed_box_sides], which stays with
    # the module that scores boxes; a slot nothing reduced into keeps its ``+inf`` seed in both
    # halves and so decodes to a negative extent.
    base = box * wp.int32(6)
    for c in range(3):
        wp.atomic_min(out_corners, base + c, lower[c])
        wp.atomic_min(out_corners, base + 3 + c, -upper[c])


@wp.func
def scanned_count(inclusive: wp.array[wp.int32], i: wp.int32) -> tuple[wp.int32, wp.int32]:
    # Exclusive offset and own count of entry ``i``, recovered from the in-place inclusive scan of
    # the counts that overwrote them. A branch, not ``wp.where``: that evaluates both arms, and
    # ``inclusive[-1]`` is out of bounds.
    start = wp.int32(0)
    if i > 0:
        start = inclusive[i - 1]
    return start, inclusive[i] - start


@wp.func
def scanned_slot(offsets: wp.array[wp.int32], i: wp.int32) -> tuple[wp.int32, wp.bool]:
    # ``scanned_count``'s sibling for the *exclusive* scan that ends in its total -- the
    # total-terminated offsets of 0/1 marks: element ``i``'s compacted slot, and whether it is kept,
    # which is exactly where the scan steps. The slot is taken relative to ``offsets[0]`` because
    # the scan may be the tail of a longer one (``remesh``'s vertex marks scanned after its face
    # flags), whose leading entry is then the count of everything before it.
    slot = offsets[i]
    return slot - offsets[0], offsets[i + 1] != slot


@wp.func
def loop_next_slot(
    loop_id: wp.array[wp.int32], offsets: wp.array[wp.int32], slot: wp.int32
) -> wp.int32:
    # Given a flat slot in a packed array of *closed* loops, the slot of the next element around
    # **its own** loop -- so the last element of a loop wraps to that loop's first and not into the
    # next loop's. ``offsets`` is the total-terminated ``n + 1`` form.
    #
    # Named because five kernels across ``boundary`` and ``holes`` had written this arithmetic out,
    # in three spellings a duplicate scan keying on statement text cannot connect. Every one of them
    # is one edge of a rim, and getting the wrap wrong silently joins two different holes.
    ell = loop_id[slot]
    begin = offsets[ell]
    return begin + loop_point(slot - begin + 1, offsets[ell + 1] - begin)


@wp.func
def loop_rim_edge_vertices(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    slot: wp.int32,
) -> tuple[wp.int32, wp.int32]:
    # The two vertex ids of the rim edge leaving packed slot ``slot`` -- the far one found through
    # ``loop_next_slot`` above, so it wraps inside its own loop rather than into the next one's.
    # ``holes.loop_rim_metrics`` and ``holes.rim_edge_keys`` read the ids to key the edge;
    # ``loop_rim_edge`` below reads per-vertex values at them.
    return (flat_loops[slot], flat_loops[loop_next_slot(loop_id, offsets, slot)])


@wp.func
def loop_rim_edge(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    values: wp.array[Any],
    slot: wp.int32,
) -> tuple[wp.int32, Any, Any]:
    # The rim edge leaving packed slot ``slot``: which loop owns it, and the two per-vertex values
    # at its ends (``loop_rim_edge_vertices``).
    #
    # The per-segment prologue of the segmented loop reductions ``boundary.loop_perimeters`` and
    # ``boundary.loop_directed_areas``, which differ only in what they fold the edge into. Generic
    # over the value dtype the way ``triangles.face_vertices`` is, so a float64 rim reduction needs
    # no second spelling.
    u, v = loop_rim_edge_vertices(flat_loops, loop_id, offsets, slot)
    return loop_id[slot], values[u], values[v]


@wp.func
def update_argmin(
    best_value: wp.ref[wp.float32], best_index: wp.ref[wp.int32], value: wp.float32, index: wp.int32
):
    # Running min-with-index update in place. Callers must be compiled with
    # ``enable_backward=False`` (``wp.ref`` helpers have no adjoint). Concrete ``float32``:
    # ``wp.ref[wp.Scalar]`` generics do not instantiate through Warp 1.17 (a ``WarpCodegenError`` at
    # kernel parse), so float64 sites keep a hand-written loop; the index/tag stays ``int32``.
    if value < best_value:
        best_value = value
        best_index = index  # noqa: F841 — writes through the wp.ref parameter


@wp.func
def update_argmax(
    best_value: wp.ref[wp.float32], best_index: wp.ref[wp.int32], value: wp.float32, index: wp.int32
):
    # Running max-with-index mirror of ``update_argmin`` (same ``enable_backward=False`` rule).
    if value > best_value:
        best_value = value
        best_index = index  # noqa: F841 — writes through the wp.ref parameter


@wp.func
def update_argmax_lowest_index(
    best_value: wp.ref[wp.float32], best_index: wp.ref[wp.int32], value: wp.float32, index: wp.int32
):
    # argmax with a lowest-index tie-break (matches the label-vote rule in ``rasterize_labels``).
    if value > best_value or (value == best_value and index < best_index):
        best_value = value
        best_index = index


@wp.func
def update_argmin_pair(
    best_value: wp.ref[wp.float32],
    first: wp.ref[wp.int32],
    second: wp.ref[wp.int32],
    value: wp.float32,
    a: wp.int32,
    b: wp.int32,
):
    # Running argmin carrying a pair of associated indices (shortest-edge endpoints).
    if value < best_value:
        best_value = value
        first = a  # noqa: F841 — writes through the wp.ref parameter
        second = b  # noqa: F841 — writes through the wp.ref parameter


@wp.func
def cross2(a: Any, b: Any) -> wp.Float:
    # 2D cross product (signed parallelogram area). Generic so float32 and float64 call sites share
    # one definition.
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def mat33_column(m: wp.mat33, col: wp.int32) -> wp.vec3:
    # A `wp.mat33`'s column as a `wp.vec3`, so extracting an SVD basis vector (`wp.svd3` returns
    # its bases as matrix columns) doesn't need three `m[row, col]` reads spelled out at every call
    # site.
    return wp.vec3(m[0, col], m[1, col], m[2, col])


@wp.func
def to_vec3d(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def to_vec3(v: wp.vec3d) -> wp.vec3:
    return wp.vec3(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))


@wp.func
def to_vec2d(v: wp.vec2) -> wp.vec2d:
    return wp.vec2d(wp.float64(v[0]), wp.float64(v[1]))


@wp.func
def to_vec2(v: wp.vec2d) -> wp.vec2:
    return wp.vec2(wp.float32(v[0]), wp.float32(v[1]))


@wp.func
def lift_vec2(p: wp.vec2, z: wp.float32) -> wp.vec3:
    # 2D point -> 3D at a fixed height: trimesh's ``util.stack_3D`` plus a z offset.
    #
    # The fifth of the vec-conversion family above, and one definition rather than two on purpose:
    # ``wp.map``'s cache is keyed by the *unqualified* function name plus the input dtypes
    # (``.claude/CLAUDE.md`` section 3.5), so two same-named ops in two kernel modules fork one
    # generated module and load it at two hashes.
    #
    # ``z`` stays a parameter because ``creation.extrude_triangulation`` genuinely lifts to a
    # height; the zero-lifting call sites pass ``wp.float32(0.0)`` explicitly.
    return wp.vec3(p[0], p[1], z)


@wp.func
def square_scalar(value: wp.Scalar) -> wp.Scalar:
    return value * value


@wp.func
def sqrt_abs(value: wp.Float) -> wp.Float:
    # ``sqrt(|x|)``, the per-row scaling the algebraic-multigrid strength test takes its geometric
    # mean from: ``|A_ij| >= theta sqrt(A_ii) sqrt(A_jj)`` is then a product and no square root runs
    # per edge. The magnitude is taken because a Laplacian written negative-semi-definite has a
    # negative diagonal.
    return wp.sqrt(wp.abs(value))


@wp.func
def inverse_or_one(value: wp.Float) -> wp.Float:
    # The Jacobi preconditioner's reciprocal, with a **zero diagonal mapped to 1 rather than to
    # infinity** -- what ``warp.optim.linear.preconditioner(m, "diag")`` does, and necessary because
    # such a row contributes nothing and must not poison the whole vector with a NaN.
    return wp.where(value != type(value)(0.0), type(value)(1.0) / value, type(value)(1.0))


@wp.func
def divide_if_positive(value: wp.Float, divisor: wp.Float) -> wp.Float:
    # Guarded division: leave ``value`` unchanged when ``divisor <= 0``.
    if divisor > type(divisor)(0.0):
        return value / divisor
    return value


@wp.kernel
def arange(out_indices: wp.array[wp.Int]) -> None:
    # The ``start == 0, step == 1`` fast path of ``array.arange``, kept beside the general affine
    # form because it is the only one any in-repo caller reaches and it marshals two fewer
    # arguments.
    #
    # ``wp.tile_arange`` was measured here and **declined**, on two counts. It cannot express this
    # kernel at all: its bounds are read at *codegen*, so a runtime ``block * TILE`` start is a
    # parse error and the only expressible form is a constant tile shifted by a ``tile_map`` over a
    # broadcast scalar. Measured that way, values identical, it is a loss below the bandwidth limit
    # and a wash at it, because writing ``out[i] = i`` is a pure streaming store with no reuse for a
    # tile to exploit.
    i = wp.int32(wp.tid())
    out_indices[i] = i


@wp.kernel
def arange_affine(start: wp.Int, step: wp.Int, out_indices: wp.array[wp.Int]) -> None:
    i = wp.int32(wp.tid())
    out_indices[i] = start + type(start)(i) * step


@wp.kernel
def sort_pair_indices(n: wp.Int, fill_value: wp.Int, out_indices: wp.array[wp.Int]) -> None:
    i = wp.int32(wp.tid())
    if i < n:
        out_indices[i] = i
    else:
        out_indices[i] = fill_value


@wp.kernel
def arange_repeat(repeats: wp.Int, out_indices: wp.array[wp.Int]) -> None:
    i = wp.int32(wp.tid())
    out_indices[i] = i // repeats


@wp.kernel
def segment_owner_labels(offsets: wp.array[wp.int32], out_owner: wp.array[wp.int32]) -> None:
    # For every element of a packed ragged array, which segment it belongs to -- the ragged
    # counterpart of ``arange_repeat``, whose segments are all one width. ``offsets`` is the
    # total-terminated ``n + 1`` form, so segment ``s`` ends at ``offsets[s + 1]``. Launched over
    # the *segment* count; each thread writes its own label across its own span.
    #
    # One thread per segment rather than one per element (a binary search into ``offsets``) is the
    # right shape for the callers here: ``geodesic_walk`` labels each packed loop position with its
    # loop and ``boundary`` each rim slot with its rim -- the segments are few and short, so this
    # beats a search *and* needs no readback of the offsets. Where the segments are few but
    # enormous, the per-element form would win instead -- nothing in the tree is in that regime.
    segment = wp.int32(wp.tid())
    for slot in range(offsets[segment], offsets[segment + 1]):
        out_owner[slot] = segment


@wp.func
def element_priority(seed: wp.int32, index: wp.int32) -> wp.uint32:
    # Element ``index``'s draw of the priority order ``random_priorities`` below writes. Named so a
    # kernel drawing into another index space (``blue_noise.sorted_random_priorities``) gives each
    # element exactly the priority the plain draw would.
    return wp.randu(wp.rand_init(seed, index))


@wp.kernel
def random_priorities(seed: wp.int32, out_priority: wp.array[wp.uint32]) -> None:
    # A total order on the elements, drawn once for the whole run rather than per round. Every
    # multi-round selection that breaks ties by priority -- blue-noise dart throwing, the
    # maximal-independent-set aggregation -- reads the same order in every round, which is what
    # makes the loop a deterministic function of ``seed`` alone. Drawing per round would make the
    # answer depend on how many rounds the input happened to need.
    i = wp.int32(wp.tid())
    out_priority[i] = element_priority(seed, i)


@wp.kernel
def gather_1d_skip_negative(
    indices: wp.array[wp.int32], table: wp.array[wp.int32], out_gathered: wp.array[wp.int32]
) -> None:
    tid = wp.int32(wp.tid())
    index = indices[tid]
    if index < wp.int32(0):
        out_gathered[tid] = index
    else:
        out_gathered[tid] = table[index]


@wp.kernel
def sort_rows_insertion(data: wp.array2d[wp.Scalar]) -> None:
    # One thread per row, in-place insertion sort across the row. For the narrow rows this library
    # actually sorts (vertex pairs, triangle corners) that is 1-3 register comparisons, versus a
    # segmented radix sort whose fixed per-segment cost dominates completely at these widths.
    row = wp.int32(wp.tid())
    width = data.shape[1]
    for i in range(1, width):
        value = data[row, i]
        j = i - 1
        while j >= 0 and data[row, j] > value:
            data[row, j + 1] = data[row, j]
            j = j - 1
        data[row, j + 1] = value


@wp.kernel
def sort_segments(offsets: wp.array[wp.int32], data: wp.array[wp.int32]) -> None:
    # One thread per segment, in-place ascending sort of ``data[offsets[s] : offsets[s + 1]]``.
    # The ragged sibling of ``sort_rows_insertion``, and the two differ in exactly one thing: a
    # rank-2 row's width is a *shape*, so that kernel can be a plain insertion sort and the wide
    # case escapes to ``segmented_sort_pairs`` on a host-side branch (``array.sort_rows``). A
    # segment's width is *data*, known only per thread, so there is no host branch to make -- which
    # is why this one is a shell sort rather than the insertion sort it degenerates into. For the
    # vertex valences its caller sorts the gap loop runs twice and costs a couple of comparisons;
    # what it buys is that a single high-degree segment cannot take the whole launch quadratic,
    # since the launch waits for its slowest thread. It beats ``edges_to_csr`` up to a few hundred
    # neighbours in one row and loses badly above that, which is why the caller chooses.
    segment = wp.int32(wp.tid())
    start = offsets[segment]
    width = offsets[segment + 1] - start
    gap = wp.int32(1)
    while gap < width // 3:
        gap = 3 * gap + 1
    while gap >= 1:
        for i in range(gap, width):
            value = data[start + i]
            j = i
            while j >= gap and data[start + j - gap] > value:
                data[start + j] = data[start + j - gap]
                j = j - gap
            data[start + j] = value
        gap = gap // 3


@wp.func
def masked_at(mask: wp.array[wp.bool], index: wp.int32) -> wp.bool:
    # The mask's value at ``index``, reading ``False`` for an index outside it rather than off the
    # end of the buffer. The guard is what lets a *membership* mask stand in for an ``isin`` over
    # the equivalent index list: ``isin`` answers ``False`` for a value it has never seen, so a
    # malformed buffer carrying an out-of-range index keeps the answer it had instead of turning
    # into an out-of-bounds read. Two compares on a memory-bound lookup.
    if index < 0 or index >= mask.shape[0]:
        return False
    return mask[index]


@wp.func
def mark_at(out_mask: wp.array[wp.bool], index: wp.int32) -> None:
    # The write side of ``masked_at``: set the mask at ``index``, dropping an index outside it
    # rather than writing off the end of the buffer -- which on the CPU device is host-heap
    # corruption (CLAUDE.md section 12.1). Every writer stores ``True``, so racing threads agree.
    if index >= 0 and index < out_mask.shape[0]:
        out_mask[index] = True


@wp.func
def shifted_index(value: wp.Scalar, offset: wp.Scalar, last: wp.Scalar) -> wp.int32:
    # Position of ``value`` in a table anchored at ``offset`` and holding values up to ``last``
    # inclusive, or ``-1`` for a value outside ``[offset, last]``.
    #
    # The range test runs in the value's **own** dtype and the narrowing to ``int32`` runs after
    # it, and that order is the whole reason ``last`` is an argument. Narrowing first is exact only
    # while the difference is known to fit, and the caller that establishes that -- ``array.isin``
    # inferring the span with two ``reduce.minmax`` reductions -- is precisely the caller a supplied
    # ``max_index`` exists to skip; on that path a 64-bit value far above the table wraps into a
    # valid slot and reads as *present*. Testing ``value`` rather than the difference cannot
    # overflow, and it leaves ``value - offset`` bounded by the table's own length, so the cast
    # below is exact by construction rather than by precondition.
    #
    # ``last`` rather than a span: ``offset + span`` is not always representable at the top of a
    # dtype, where ``offset + span - 1`` is the largest value the table holds and therefore always
    # is.
    if value < offset or value > last:
        return wp.int32(-1)
    return wp.int32(value - offset)


@wp.kernel
def isin_mark_table(
    test_elements: wp.array[wp.Scalar],
    anchor: wp.Scalar,
    last: wp.Scalar,
    out_membership: wp.array[wp.bool],
) -> None:
    # The test side of ``array.isin``'s direct-index strategy: shift into the table anchored at
    # ``anchor`` and mark the slot, in one launch. It replaces ``wp.map(shifted_index, ...)`` into
    # a ``(k,)`` int32 slot buffer followed by ``scatter.mark_membership_mask`` over it -- one
    # launch and one allocation fewer, the same slots written. ``isin_lookup_mask`` below is the
    # element side and reads the table this writes; both shift through ``shifted_index``.
    mark_at(out_membership, shifted_index(test_elements[wp.int32(wp.tid())], anchor, last))


@wp.kernel
def isin_lookup_mask(
    elements: wp.array[wp.Scalar],
    anchor: wp.Scalar,
    last: wp.Scalar,
    membership: wp.array[wp.bool],
    out_mask: wp.array[wp.bool],
) -> None:
    # The element side of ``array.isin``'s direct-index strategy: shift into the table anchored at
    # ``anchor`` and read the flag there, in one launch. Written as a kernel rather than as
    # ``wp.map(shifted_index, ...)`` followed by a Python-scope gather for two reasons. It halves
    # the element-side work -- one launch instead of a launch, an ``(n,)`` int32 slot buffer and
    # the gather's own ``wp.copy``. And the range guard is what makes ``isin``'s ``max_index``
    # escape hatch *safe*: a caller-supplied bound the values exceed reads ``False`` here, where an
    # unguarded gather would read past the end of ``membership`` -- silent memory unsafety rather
    # than a wrong answer (CLAUDE.md section 12.1).
    #
    # The guard is ``masked_at``, which reads the *buffer's* own length. ``shifted_index`` already
    # rejects an out-of-range value, so this is a second, independent bound rather than the only
    # one -- and reading ``membership.shape[0]`` means the two cannot disagree, where a separate
    # ``span`` argument beside ``last`` would be one launch argument encoding the same limit twice.
    i = wp.int32(wp.tid())
    out_mask[i] = masked_at(membership, shifted_index(elements[i], anchor, last))


@wp.kernel
def isin_lookup_sorted(
    elements: wp.array[wp.Scalar], sorted_test: wp.array[wp.Scalar], out_mask: wp.array[wp.bool]
) -> None:
    tid = wp.int32(wp.tid())
    out_mask[tid] = binary_search_sorted_contains(sorted_test, elements[tid])


@wp.func
def mask_not(a: wp.bool) -> wp.bool:
    # Mask complement -- for a caller holding the region to *delete* and needing the one to keep,
    # among others. Warp exposes no ``logical_not`` builtin (``wp.invert`` is the bitwise
    # complement, which is wrong for a ``wp.bool``), so this one-liner is what ``wp.map`` needs, and
    # it is the tree's only spelling of it.
    return not a


@wp.func
def mask_and(a: wp.bool, b: wp.bool) -> wp.bool:
    return a and b


@wp.func
def mask_and_not(a: wp.bool, b: wp.bool) -> wp.bool:
    return a and not b


@wp.func
def complement_flag(a: wp.bool) -> wp.int32:
    # ``1`` where the mask is False, ``0`` where it is True: the scan input for an inverted
    # ``mask_to_compact_ranks`` (free/interior DOFs of a fixed-vertex mask, for instance).
    return wp.where(a, wp.int32(0), wp.int32(1))


@wp.kernel
def bool_flags(mask: wp.array[wp.bool], out_flags: wp.array[wp.int32]) -> None:
    # ``1`` / ``0`` per mask entry: the ``int32`` a ``wp.utils.array_scan`` needs from a ``wp.bool``
    # buffer, which it cannot read directly.
    #
    # A plain kernel rather than ``wp.utils.array_cast``, which produces the identical bytes but
    # resolves a generic cast kernel per call and measures about twice as slow, flat in the element
    # count -- so the difference is its host-side resolution and not the copy. ``wp.map`` is not the
    # spelling either: ``wp.Scalar`` does not instantiate for ``wp.bool`` (CLAUDE.md section 12.4),
    # which is why ``nonzero_flag`` below covers every dtype except this one.
    i = wp.int32(wp.tid())
    out_flags[i] = wp.where(mask[i], 1, 0)


def _astype_kernel(name: str, source: type, target: type) -> wp.Kernel:
    """Build one concrete ``out[i] = target(values[i])`` kernel: ``wp.utils.array_cast``'s body."""

    # ``wp.utils.array_cast`` launches its own ``Any``-generic ``_array_cast_kernel``, so every
    # ``array.astype`` paid Warp's host-side overload resolution on top of the launch -- about
    # twice a concrete launch, flat in the element count (the reason ``bool_flags`` above exists).
    # The body is that kernel's, so the bytes are identical; only the dispatch changes.
    def _k(values: wp.array[wp.Scalar], out_values: wp.array[wp.Scalar]) -> None:
        i = wp.int32(wp.tid())
        out_values[i] = out_values.dtype(values[i])

    _k.__annotations__["values"] = wp.array[source]
    _k.__annotations__["out_values"] = wp.array[target]
    return wp.kernel(_k, name=name)


@wp.func
def nonzero_flag(value: wp.Scalar) -> wp.int32:
    # ``1`` for any non-zero value, ``0`` otherwise: the scan input that lets ``flatnonzero``
    # accept integer and float arrays as well as masks. A bool array does not need this -- Warp's
    # ``wp.Scalar`` does not instantiate for ``wp.bool``, and ``wp.utils.array_cast`` already
    # produces exactly 0/1 for one, which is why the wrapper keeps that path separate.
    return wp.where(value != type(value)(0), wp.int32(1), wp.int32(0))


# The comparison family below is the tree's spelling for a thresholding ``wp.map``, and it is worth
# saying so here because it kept being re-spelled under domain names. What such a private predicate
# carries that is worth keeping is never the comparison -- it is *which quantity* and *which
# threshold* its caller chose, and that argument belongs in the wrapper that chooses it, where a
# user of the public function reads it. Reach for a named predicate when it computes something
# (``is_positive_finite``, ``is_close_scalar``); reach for these when it is a comparison.
@wp.func
def greater(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a > b


@wp.func
def greater_equal(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a >= b


@wp.func
def less(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a < b


@wp.func
def less_equal(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a <= b


@wp.func
def is_positive_finite(value: wp.Float) -> wp.bool:
    return value > type(value)(0) and wp.isfinite(value)


@wp.func
def is_close_scalar(a: wp.Float, b: wp.Float, rtol: wp.Float, atol: wp.Float) -> wp.bool:
    # Generic over the caller's float precision, so ``allclose`` works on float16/32/64 from one
    # definition. The tolerances must arrive at that same precision -- see the wrapper.
    return wp.abs(a - b) <= atol + rtol * wp.abs(b)


@wp.func
def is_close_vec3(a: wp.vec3, b: wp.vec3, rtol: wp.float32, atol: wp.float32) -> wp.bool:
    close = True
    for k in range(3):
        if wp.abs(a[k] - b[k]) > atol + rtol * wp.abs(b[k]):
            close = False
    return close


@wp.func
def array_shift_insert(array: wp.array[wp.Scalar], value: wp.Scalar, index: wp.int32) -> None:
    for i in range(array.shape[0] - 1, index, -1):
        array[i] = array[i - 1]
    array[index] = value


@wp.func
def binary_search_index(values: wp.array[wp.Scalar], value: wp.Scalar) -> wp.int32:
    """First index i with values[i] > value, or len(values) (numpy searchsorted side='right')."""
    n = values.shape[0]
    left = wp.int32(0)
    right = n - 1
    result = n
    while left <= right:
        mid = (left + right) // 2
        if values[mid] > value:
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return result


@wp.func
def binary_search_index_left(values: wp.array[wp.Scalar], value: wp.Scalar) -> wp.int32:
    """First index i with values[i] >= value, or len(values) (numpy searchsorted side='left')."""
    # ``wp.lower_bound`` is the same search, but it clamps its result to ``n - 1``
    # (``warp/native/array.h``), so a value past the last element reads back as the last index
    # instead of ``n``. Its caller (``holes.probe_rim_edges``, which probes edges absent
    # from the table) does query past the end, so the fix-up is mandatory, not defensive.
    n = values.shape[0]
    index = wp.lower_bound(values, value)
    if index == n - 1 and values[n - 1] < value:
        index = n
    return index


@wp.func
def binary_search_sorted_contains(values: wp.array[wp.Scalar], value: wp.Scalar) -> wp.bool:
    # ``wp.lower_bound``'s clamp to ``n - 1`` is harmless here: a value past the end lands on the
    # last element, which then compares unequal. ``and`` short-circuits in kernel scope, so the
    # element read is skipped on an empty array.
    n = values.shape[0]
    index = wp.lower_bound(values, value)
    return index < n and values[index] == value


@wp.kernel
def map_sorted_inverse(
    data: wp.array[wp.Scalar], sorted_unique: wp.array[wp.Scalar], out_inverse: wp.array[wp.int32]
) -> None:
    # ``numpy.unique(return_inverse=True)``: each element's position in the sorted unique output,
    # which is ``searchsorted(sorted_unique, x, side="left")``. Every key searched for came out of
    # that same unique set, so on an ordered dtype the left search lands on the exact slot and the
    # equality below always holds.
    #
    # **The two lines that are not cosmetic are the ``side="left"`` search and the equality, and
    # both are the ``NaN`` case.** Warp's float radix sort puts ``NaN`` last and every comparison
    # against it is false, so a binary search whose midpoint lands in that tail is steered by a
    # predicate that never fires. ``side="right"`` is steered *into* the tail and reports the last
    # slot for every finite value -- a silent wrong answer on ``grouping.unique_1d``'s documented
    # float path. ``side="left"`` is steered *away* from it and is correct for every finite value;
    # what it then gets wrong is ``NaN`` itself, which lands at slot 0 -- and that is exactly what
    # the equality catches, ``NaN`` comparing unequal to everything including itself, since the only
    # slot a ``NaN`` can belong to is the last one.
    i = wp.int32(wp.tid())
    value = data[i]
    n = wp.int32(sorted_unique.shape[0])
    index = binary_search_index_left(sorted_unique, value)
    if index >= n or sorted_unique[index] != value:
        index = n - 1
    out_inverse[i] = index


# --- Sparse assembly: sorted keys to CSR ---------------------------------------------------
#
# ``warp.sparse.bsr_from_triplets`` sorts every triplet on a full-width key and allocates scratch
# several times the matrix, which on a large mesh made the build most of an operator's cost. The
# tree builds its CSR matrices from *keys* instead: a producer writes one ``uint64`` per entry
# (``csr_key``: ``row * n_cols + col``, or the sentinel ``n_rows * n_cols`` for a slot that holds
# nothing), ``radix_sort_pairs`` orders only the bits ``n_rows * n_cols`` needs, and the three
# kernels below turn the sorted runs into ``offsets`` / ``columns`` plus each entry's first sorted
# position, from which a caller-specific kernel forms the entry's value out of its contributors in
# sorted -- that is, stable, producer -- order. A mesh operator writes its keys straight from the
# faces, so no ``(row, col, value)`` arrays exist at all.


@wp.func
def csr_key(row: wp.int32, col: wp.int32, n_rows: wp.int32, n_cols: wp.int32) -> wp.uint64:
    """Row-major sort key of entry ``(row, col)``; the sentinel ``n_rows * n_cols`` if outside."""
    if row < 0 or row >= n_rows or col < 0 or col >= n_cols:
        return wp.uint64(n_rows) * wp.uint64(n_cols)
    return wp.uint64(row) * wp.uint64(n_cols) + wp.uint64(col)


@wp.kernel
def coo_keys(
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    n_rows: wp.int32,
    n_cols: wp.int32,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # One key and the identity payload per triplet, into the first half of the sort's double-width
    # buffers. An index outside the matrix gets the sentinel and is dropped, as
    # ``bsr_from_triplets`` drops it.
    i = wp.int32(wp.tid())
    out_keys[i] = csr_key(rows[i], cols[i], n_rows, n_cols)
    out_order[i] = i


@wp.kernel
def coo_keys_nonzero(
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[Any],
    n_rows: wp.int32,
    n_cols: wp.int32,
    zero: Any,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # ``coo_keys`` for a pruning build: a triplet whose value is exactly ``zero`` gets the sentinel,
    # so it is never sorted. That cannot change any entry's sum (``x + 0 == x``), so it is purely
    # a saving; ``csr_nonzero_run_flags`` then drops the entries whose remaining triplets cancel.
    i = wp.int32(wp.tid())
    key = csr_key(rows[i], cols[i], n_rows, n_cols)
    if values[i] == zero:
        key = wp.uint64(n_rows) * wp.uint64(n_cols)
    out_keys[i] = key
    out_order[i] = i


@wp.func
def sorted_run_start(sorted_values: wp.array[wp.Int], i: wp.int32) -> wp.bool:
    # Does position ``i`` begin a run of equal values? The one test every run-length grouping in
    # the package shares, and its callers add their own conditions on top: ``mark_group_starts``
    # requires the run to be exactly ``length`` long, ``remesh``'s ``mark_edge_pair_starts``
    # specialises that to two, and the plain run-start markers want every run whatever its length.
    # Bounding the *data* is the caller's job too -- the buffer is usually over-allocated
    # radix-sort scratch.
    if i == 0:
        return True
    return sorted_values[i] != sorted_values[i - 1]


@wp.func
def csr_run_start(keys: wp.array[wp.Int], i: wp.int32, sentinel: wp.Int) -> wp.bool:
    """Whether sorted position ``i`` opens a run of equal keys below ``sentinel``."""
    # ``sorted_run_start`` over a sort whose tail holds out-of-range keys at or above
    # ``sentinel`` (a hole, not a value): the CSR entries, and ``intersection``'s crossed edges.
    return keys[i] < sentinel and sorted_run_start(keys, i)


@wp.func
def sorted_run_end(keys: wp.array[wp.uint64], i: wp.int32, count: wp.int32) -> wp.int32:
    """One past the last sorted position of the run of equal keys that starts at ``i``."""
    key = keys[i]
    p = i + 1
    while p < count and keys[p] == key:
        p += 1
    return p


@wp.func
def csr_run_sum(
    keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    values: wp.array[Any],
    i: wp.int32,
    count: wp.int32,
) -> Any:
    # The summed value of the entry whose run starts at sorted position ``i``: its contributors in
    # sorted -- that is, triplet -- order. The one summation both of ``csr_from_triplets``' passes
    # form, so the pruning test and the placed value are the same number bit for bit.
    acc = values[order[i]]
    for p in range(i + 1, sorted_run_end(keys, i, count)):
        acc += values[order[p]]
    return acc


@wp.func
def csr_row_offsets(
    keys: wp.array[wp.uint64],
    inclusive: wp.array[wp.int32],
    i: wp.int32,
    before: wp.int32,
    sentinel: wp.uint64,
    n_rows: wp.int32,
    n_cols: wp.uint64,
    out_offsets: wp.array[wp.int32],
) -> None:
    # The row bounds sorted in-range position ``i`` owns, ``before`` being the number of entries
    # ranked ahead of it (``scanned_count``'s start): a run start that opens a new row writes the
    # bound of every row from the previous run's row up to its own -- so an empty row gets the right
    # value -- and the last in-range position writes the rest. The rule both CSR placements share;
    # ``out_offsets`` arrives zero-filled, which is the whole answer when nothing is in range.
    key = keys[i]
    row = wp.int32(key // n_cols)
    if i == 0 or keys[i - 1] != key:
        previous = wp.int32(-1)
        if i > 0:
            previous = wp.int32(keys[i - 1] // n_cols)
        for r in range(previous + 1, row + 1):
            out_offsets[r] = before
    if i + 1 == inclusive.shape[0] or keys[i + 1] >= sentinel:
        for r in range(row + 1, n_rows + 1):
            out_offsets[r] = inclusive[i]


@wp.kernel
def csr_run_flags(
    keys: wp.array[wp.uint64], sentinel: wp.uint64, out_flags: wp.array[wp.int32]
) -> None:
    i = wp.int32(wp.tid())
    out_flags[i] = wp.where(csr_run_start(keys, i, sentinel), 1, 0)


@wp.kernel
def csr_from_runs(
    keys: wp.array[wp.uint64],
    inclusive: wp.array[wp.int32],
    sentinel: wp.uint64,
    n_rows: wp.int32,
    n_cols: wp.uint64,
    out_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_starts: wp.array[wp.int32],
) -> None:
    # Launched over the sorted positions; ``inclusive`` is the inclusive scan of
    # ``csr_run_flags``, so an entry's index is its run start's exclusive scan value. A run start
    # writes its column and first sorted position, the last in-range position closes
    # ``out_starts``, and ``csr_row_offsets`` fills the row bounds.
    i = wp.int32(wp.tid())
    key = keys[i]
    if key >= sentinel:
        return
    entry, opens = scanned_count(inclusive, i)
    if opens != 0:
        out_columns[entry] = wp.int32(key % n_cols)
        out_starts[entry] = i
    if i + 1 == inclusive.shape[0] or keys[i + 1] >= sentinel:
        out_starts[inclusive[i]] = i + 1
    csr_row_offsets(keys, inclusive, i, entry, sentinel, n_rows, n_cols, out_offsets)


@wp.kernel
def csr_nonzero_run_flags(
    keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    values: wp.array[Any],
    sentinel: wp.uint64,
    zero: Any,
    out_flags: wp.array[wp.int32],
) -> None:
    # ``csr_run_flags`` for a pruning build: a run start is an entry only if its summed value is not
    # exactly ``zero``. Pruning therefore acts on the *assembled* entry -- one whose triplets cancel
    # is dropped, which is where the value a caller sees is zero. The flags' inclusive scan numbers
    # the kept entries; ``csr_from_flagged_runs`` forms the same sum again rather than reading a
    # table-sized buffer of them.
    i = wp.int32(wp.tid())
    flag = 0
    if csr_run_start(keys, i, sentinel):
        flag = 1
        if csr_run_sum(keys, order, values, i, out_flags.shape[0]) == zero:
            flag = 0
    out_flags[i] = flag


@wp.kernel
def csr_from_flagged_runs(
    keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    values: wp.array[Any],
    inclusive: wp.array[wp.int32],
    sentinel: wp.uint64,
    n_rows: wp.int32,
    n_cols: wp.uint64,
    out_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[Any],
) -> None:
    # ``csr_from_runs`` for a triplet build: ``inclusive`` is the scan of ``csr_run_flags`` or,
    # pruning, of ``csr_nonzero_run_flags``. A kept run start sums its run (``csr_run_sum``) and
    # writes its column and value at its rank; a row's offset is the number of kept entries before
    # its first run start, pruned or not, so a row left empty by pruning still gets its bound.
    i = wp.int32(wp.tid())
    key = keys[i]
    if key >= sentinel:
        return
    before, kept = scanned_count(inclusive, i)
    if kept != 0:
        out_columns[before] = wp.int32(key % n_cols)
        out_values[before] = csr_run_sum(keys, order, values, i, inclusive.shape[0])
    csr_row_offsets(keys, inclusive, i, before, sentinel, n_rows, n_cols, out_offsets)


# Concrete overloads, registered at import -- rationale in ``triwarp/kernels/reduce.py``, rule in
# CLAUDE.md section 2.5. This module is imported by 25 kernel modules and 15 wrappers, so its
# rebuilds are felt widely.
#
# The index-buffer kernels fill a buffer every caller in the package allocates ``wp.int32``;
# ``wp.Int`` in their annotation is the template, not a menu. Their three wrappers
# (``array.arange``, ``arange_repeat``, ``sort_pair_indices``) expose no ``dtype=`` keyword, so this
# tuple is an exact statement of the public surface rather than a restriction of it: a second width
# belongs here only alongside a wrapper that offers one (sections 2.5 and 4.2).
#
# The two search kernels take the caller's *key* dtype, whose surface is the one
# ``sortable_dtype`` maps onto.
_INDEX_DTYPES = (wp.int32,)

# Every block type a CSR build sums over.
_CSR_VALUE_DTYPES = (wp.float32, wp.float64, wp.int32, wp.mat22d)
# ``isin_lookup_sorted``'s only caller is ``array.isin``, which raises on a non-integer dtype, so
# its key surface stops at the integers.
_KEY_DTYPES = (wp.int32, wp.int64, wp.uint32, wp.uint64)
# ``map_sorted_inverse`` reaches further: its sole caller
# ``grouping.unique_1d(return_inverse=True)`` launches it in ``twt.sortable_dtype(data.dtype)``, and
# ``unique_1d`` accepts *any* scalar dtype -- its own docstring documents the float ``NaN`` slot
# semantics for exactly this path. So the float rows are reachable public API, and omitting them
# rebuilds this widely imported module on the first float call.
_SORT_KEY_DTYPES = (*_KEY_DTYPES, wp.float32, wp.float64)


class KernelTable(dict[Any, wp.Kernel]):
    """
    Concrete kernels by dtype key, so a wrapper hands ``wp.launch`` the kernel it already resolved.

    The base of [`OverloadTable`][triwarp.kernels.array.OverloadTable], and used directly by
    ``kernels/reduce.py``, whose kernels are factory instantiations rather than ``wp.overload``
    results. The whole content of the class is what a *missing* key means: a dtype the module's
    dispatch can reach but never registered, which is a registration gap and not a slow path -- see
    [`OverloadTable`][triwarp.kernels.array.OverloadTable] for what that silently costs.
    """

    def __init__(self, owner: str, entries: Mapping[Any, wp.Kernel]) -> None:
        """Key ``entries`` by dtype, naming ``owner`` in the error a missing key raises."""
        super().__init__(entries)
        self._owner = owner

    def __missing__(self, key: Any) -> wp.Kernel:
        raise KeyError(
            f"{self._owner} has no kernel registered for {key!r}; add the dtype to the owning "
            "kernel module's registration rather than launching a generic kernel, which would "
            "rebuild the whole module on first use (CLAUDE.md section 2.5)."
        )


class OverloadTable(KernelTable):
    """
    Concrete kernel handles by dtype key, so a launch never re-infers the overload.

    ``wp.launch`` on a generic kernel runs ``infer_argument_types`` over the whole argument list
    and *then* looks the overload up, on **every** call -- and that inference is the cost, not the
    lookup. A generic launch is roughly *twice* the host cost of a concrete one, and more again
    when several parameters are generic.

    Nothing new is compiled. ``_register_overloads`` was already creating these overloads at import
    for the reason in ``.claude/CLAUDE.md`` section 2.5 (a lazily instantiated overload rebuilds the
    whole module); this only keeps the ``wp.Kernel`` that ``wp.overload`` returns instead of
    discarding it. The kernel source stays dtype-generic, so section 1.2's preference for generic
    ``@wp.func``/kernel bodies is untouched -- what changes is only which object the wrapper hands
    ``wp.launch``.

    A missing key raises rather than falling back to the generic kernel. Falling back would work
    and would be *slow in the way section 2.5 exists to prevent* -- the first launch at an
    unregistered dtype rebuilds the whole module, which can take a minute -- so an unregistered
    dtype is a registration gap to fix, and this turns it from a clock reading into an error naming
    the kernel.
    """

    def __init__(self, kernel: wp.Kernel, signatures: Mapping[Any, Sequence[Any]]) -> None:
        """Instantiate ``kernel``'s overloads, keyed by whatever ``signatures`` keys them by."""
        super().__init__(
            kernel.key, {key: wp.overload(kernel, list(types)) for key, types in signatures.items()}
        )


# The conversions ``array.astype`` launches as concrete kernels, keyed ``(source, target)``. The set
# is a census rather than a menu: instrumenting ``wp.utils.array_cast`` over the full test suite,
# these four pairs are 97 % of the ``astype`` calls (the masks every compaction widens, the
# ``float32`` areas an energy or mass assembly promotes, and the narrowing back to a mask). A pair
# outside the table falls through to ``wp.utils.array_cast``, which is correct and merely pays the
# generic dispatch; that is Warp's own module, so the fall-through forks nothing of this one.
# ``(wp.bool, wp.int32)`` is ``bool_flags`` itself -- identical bytes, one kernel.
ASTYPE = KernelTable(
    "astype",
    {
        (wp.bool, wp.int32): bool_flags,
        (wp.int32, wp.bool): _astype_kernel("astype_int32_bool", wp.int32, wp.bool),
        (wp.int32, wp.float32): _astype_kernel("astype_int32_float32", wp.int32, wp.float32),
        (wp.float32, wp.float64): _astype_kernel("astype_float32_float64", wp.float32, wp.float64),
    },
)


# The concrete handles ``wp.overload`` hands back, keyed by the caller's dtype -- see
# [`OverloadTable`][triwarp.kernels.array.OverloadTable] for why a wrapper launches through these
# rather than through the generic kernel above it. Declared here so a type checker sees them at
# module scope; ``_register_overloads`` fills them in at import.
ARANGE: OverloadTable
ARANGE_AFFINE: OverloadTable
ARANGE_REPEAT: OverloadTable
SORT_PAIR_INDICES: OverloadTable
ISIN_MARK_TABLE: OverloadTable
ISIN_LOOKUP_MASK: OverloadTable
ISIN_LOOKUP_SORTED: OverloadTable
MAP_SORTED_INVERSE: OverloadTable
SORT_ROWS_INSERTION: OverloadTable
COO_KEYS_NONZERO: OverloadTable
CSR_NONZERO_RUN_FLAGS: OverloadTable
CSR_FROM_FLAGGED_RUNS: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global ARANGE, ARANGE_AFFINE, ARANGE_REPEAT, SORT_PAIR_INDICES
    global \
        ISIN_MARK_TABLE, \
        ISIN_LOOKUP_MASK, \
        ISIN_LOOKUP_SORTED, \
        MAP_SORTED_INVERSE, \
        SORT_ROWS_INSERTION
    global COO_KEYS_NONZERO, CSR_NONZERO_RUN_FLAGS, CSR_FROM_FLAGGED_RUNS
    ARANGE = OverloadTable(arange, {d: [wp.array[d]] for d in _INDEX_DTYPES})
    ARANGE_AFFINE = OverloadTable(arange_affine, {d: [d, d, wp.array[d]] for d in _INDEX_DTYPES})
    ARANGE_REPEAT = OverloadTable(arange_repeat, {d: [d, wp.array[d]] for d in _INDEX_DTYPES})
    SORT_PAIR_INDICES = OverloadTable(
        sort_pair_indices, {d: [d, d, wp.array[d]] for d in _INDEX_DTYPES}
    )
    ISIN_MARK_TABLE = OverloadTable(
        isin_mark_table, {d: [wp.array[d], d, d, wp.array[wp.bool]] for d in _KEY_DTYPES}
    )
    ISIN_LOOKUP_MASK = OverloadTable(
        isin_lookup_mask,
        {d: [wp.array[d], d, d, wp.array[wp.bool], wp.array[wp.bool]] for d in _KEY_DTYPES},
    )
    ISIN_LOOKUP_SORTED = OverloadTable(
        isin_lookup_sorted, {d: [wp.array[d], wp.array[d], wp.array[wp.bool]] for d in _KEY_DTYPES}
    )
    MAP_SORTED_INVERSE = OverloadTable(
        map_sorted_inverse,
        {d: [wp.array[d], wp.array[d], wp.array[wp.int32]] for d in _SORT_KEY_DTYPES},
    )
    # ``sort_rows_insertion`` sorts a rank-2 table in place. Its only caller is ``array.sort_rows``,
    # so the set is that wrapper's annotation, ``Array2dInt32 | Array2dFloat32``. A ``wp.float64``
    # entry here would make the accepted dtypes depend on the row width: this kernel is generic, so
    # a float64 table would sort silently while the wide-row fallback (``segmented_sort_pairs``,
    # int32/float32 keys only) raised from inside Warp. ``sort_rows`` rejects the dtype up front and
    # this set matches it.
    SORT_ROWS_INSERTION = OverloadTable(
        sort_rows_insertion, {d: [wp.array2d[d]] for d in (wp.int32, wp.float32)}
    )
    # The value types a CSR assembly reaches: the scalar operators in either precision, the
    # connection Laplacian's ``mat22d`` blocks, and ``index_sparse``'s documented integer data.
    COO_KEYS_NONZERO = OverloadTable(
        coo_keys_nonzero,
        {
            d: [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[d],
                wp.int32,
                wp.int32,
                d,
                wp.array[wp.uint64],
                wp.array[wp.int32],
            ]
            for d in _CSR_VALUE_DTYPES
        },
    )
    CSR_NONZERO_RUN_FLAGS = OverloadTable(
        csr_nonzero_run_flags,
        {
            d: [
                wp.array[wp.uint64],
                wp.array[wp.int32],
                wp.array[d],
                wp.uint64,
                d,
                wp.array[wp.int32],
            ]
            for d in _CSR_VALUE_DTYPES
        },
    )
    CSR_FROM_FLAGGED_RUNS = OverloadTable(
        csr_from_flagged_runs,
        {
            d: [
                wp.array[wp.uint64],
                wp.array[wp.int32],
                wp.array[d],
                wp.array[wp.int32],
                wp.uint64,
                wp.int32,
                wp.uint64,
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[d],
            ]
            for d in _CSR_VALUE_DTYPES
        },
    )


_register_overloads()


# ``wp.map`` has the same lazy-instantiation property ``_register_overloads`` exists for, and the
# same fix: a ``wp.map(op, ...)`` call generates a module named ``map_<unqualified op name>``, each
# distinct *call signature* forks its hash, and a module's hash covers the kernels instantiated in
# it. Declaring the signatures up front with ``return_kernel=True`` reaches the final module
# directly for milliseconds of import. The chain is order-dependent, so this is a developer-loop
# tax rather than a one-time install cost (CLAUDE.md section 3.5).
#
# **The fork axis is not only the dtype, and that is the part that is not guessable.**
# ``warp._src.utils.map`` keys its cache on ``(is_array, type(input).__name__, dtype, ndim,
# broadcast_mask)`` per input, where ``broadcast_mask`` is ``tuple(d == 1 for d in shape)``. Three
# things fork a module that a dtype census cannot see: a **length-1** array (the commonest axis in
# the tree -- it is how every reduction-into-a-scalar wrapper calls ``wp.map``); an
# **``indexedarray``**, i.e. a Python-scope gather view; and the **rank**.
#
# **The table is what the wrappers' dispatch reaches, not what the ops admit** (no speculative
# generality). Generate it by monkeypatching ``wp.map`` over a full ``tests/`` run and recording
# Warp's own cache key, so a new signature is a measurement rather than a guess; only ops that
# actually fork are listed. **The gate is the load count**: distinct
# ``(map_* module, hash, device, block_dim)`` loads against distinct ``(module, device, block_dim)``
# pairs, whose floor is one module per op per device. A non-zero excess names the op that reopened
# a chain.
#
# **Placement is decided by the shape of the imports.** Several of the longest chains are Warp
# *builtins*, so one generated module is shared across several wrapper modules and every
# declaration for it must run before the *first* launch from any of them; those live here, because
# this module is reached first. Ops belonging to a single kernel module are declared at that
# module's own bottom — they *cannot* be declared here, since those modules import this one.
#
# One cost that looks like a regression and is not: the first ``wp.zeros`` below forces
# ``wp.init()``, so ``import triwarp.kernels.array`` alone gets much slower. A process that imports
# a wrapper *and does one call* is unchanged, and ``import triwarp`` is untouched because PEP 562
# keeps it from importing any of this.


def map_probe(dtype: type) -> wp.array:
    """Build a zero-length host array: the cheapest stand-in for a dense ``wp.map`` input."""
    # The generated module is keyed by dtype, rank and broadcast mask, not by device or by length,
    # so nothing needs allocating on an accelerator to declare a signature.
    return wp.zeros(0, dtype=dtype, device="cpu")


def map_probe_single(dtype: type) -> wp.array:
    """Build a length-1 host array, whose broadcast mask is its own ``wp.map`` cache key."""
    return wp.zeros(1, dtype=dtype, device="cpu")


def map_probe_gathered(dtype: type) -> wp.indexedarray:
    """Build a Python-scope gather view, whose array *kind* is part of the ``wp.map`` key."""
    return map_probe(dtype)[wp.zeros(0, dtype=wp.int32, device="cpu")]


def declare_map_signatures(
    signatures: Sequence[tuple[Any, tuple[Any, ...], Any | Sequence[Any]]],
) -> None:
    """
    Pre-declare ``wp.map`` call signatures so their generated module is built once, not per fork.

    Each entry is ``(op, inputs, out_dtype)``, where ``out_dtype`` is one dtype or a sequence of
    them for a multi-output ``@wp.func``. See the comment above for why a kernel module needs this
    at all, how the table is derived, and what forks a ``map_*`` module; call it from a
    ``_declare_map_kernels()`` at the bottom of the module that owns the ops.

    The output *dtype* is declared rather than an output array because ``wp.map`` validates the
    destination's shape against the broadcast result of the inputs, so a row built from
    [`map_probe_single`][triwarp.kernels.array.map_probe_single] inputs needs a length-1
    destination and one built from [`map_probe`][triwarp.kernels.array.map_probe] needs a
    zero-length one -- a distinction with no bearing on the signature being declared, and one that
    raises ``TypeError`` at import if a table row gets it wrong. Deriving it here removes the trap.
    """
    for op, inputs, out_dtype in signatures:
        # ``wp.indexedarray`` (a Python-scope gather view, from ``map_probe_gathered``) is not a
        # ``wp.array`` subclass -- checked directly, both derive from a common ``Array`` base but
        # neither from the other -- so a length check against ``wp.array`` alone silently drops a
        # gathered operand. Every row declared here today pairs a gathered operand with a bare
        # scalar, so the omission has never mattered (an all-gathered or gathered-plus-single row
        # would infer length 0 regardless of the gathered view's real length); include both kinds so
        # the next such row is not the one that finds it.
        arrays = [value for value in inputs if isinstance(value, (wp.array, wp.indexedarray))]
        length = 1 if arrays and all(value.shape[0] == 1 for value in arrays) else 0
        dtypes = out_dtype if isinstance(out_dtype, (list, tuple)) else [out_dtype]
        outputs = [wp.zeros(length, dtype=dtype, device="cpu") for dtype in dtypes]
        wp.map(op, *inputs, out=outputs if len(outputs) > 1 else outputs[0], return_kernel=True)


def _declare_map_kernels() -> None:
    """Declare the Warp builtins the tree forks, plus this module's own mapped ``@wp.func``s."""
    dense, single, gathered = map_probe, map_probe_single, map_probe_gathered
    declare_map_signatures(
        [
            # --- Warp builtins: one generated module each, shared across every wrapper that
            # maps them, which is why they are declared here and not per wrapper.
            (wp.add, (dense(wp.int32), dense(wp.int32)), wp.int32),
            (wp.add, (dense(wp.int32), wp.int32(1)), wp.int32),
            (wp.add, (single(wp.int32), single(wp.int32)), wp.int32),
            (wp.div, (dense(wp.float32), dense(wp.float32)), wp.float32),
            (wp.div, (dense(wp.float32), wp.float32(1)), wp.float32),
            (wp.div, (dense(wp.mat33), wp.float32(1)), wp.mat33),
            (wp.div, (single(wp.mat33), wp.float32(1)), wp.mat33),
            (wp.div, (dense(wp.vec3), dense(wp.float32)), wp.vec3),
            (wp.div, (dense(wp.vec3), wp.float32(1)), wp.vec3),
            (wp.div, (single(wp.vec3), wp.float32(1)), wp.vec3),
            (wp.length, (dense(wp.vec2d),), wp.float64),
            (wp.length, (single(wp.vec2d),), wp.float64),
            (wp.length, (dense(wp.vec3),), wp.float32),
            (wp.mul, (dense(wp.float32), wp.float32(1)), wp.float32),
            (wp.neg, (dense(wp.float32),), wp.float32),
            (wp.neg, (dense(wp.float64),), wp.float64),
            (wp.neg, (dense(wp.vec3),), wp.vec3),
            (wp.normalize, (dense(wp.vec3),), wp.vec3),
            (wp.normalize, (single(wp.vec3),), wp.vec3),
            (wp.sub, (dense(wp.float32), wp.float32(1)), wp.float32),
            (wp.sub, (dense(wp.float64), wp.float64(1)), wp.float64),
            # --- This module's own mapped ``@wp.func``s. Every scalar argument is spelled
            # with an explicit constructor because a bare Python int at a ``wp.map`` scalar
            # infers ``wp.int32`` whatever the mapped function's own dtype is.
            (greater, (dense(wp.float32), wp.float32(1)), wp.bool),
            (greater, (dense(wp.float64), wp.float64(1)), wp.bool),
            (greater, (dense(wp.int32), wp.int32(1)), wp.bool),
            (greater_equal, (dense(wp.float32), wp.float32(1)), wp.bool),
            (greater_equal, (dense(wp.int32), wp.int32(1)), wp.bool),
            (greater_equal, (single(wp.int32), wp.int32(1)), wp.bool),
            (greater_equal, (gathered(wp.float32), wp.float32(1)), wp.bool),
            (greater_equal, (gathered(wp.int32), wp.int32(1)), wp.bool),
            (inverse_or_one, (dense(wp.float64),), wp.float64),
            (inverse_or_one, (single(wp.float64),), wp.float64),
            # ``linalg._BatchedCg``'s Jacobi diagonal of a ``float32`` system (``reconstruction``).
            (inverse_or_one, (dense(wp.float32),), wp.float32),
            (
                is_close_scalar,
                (dense(wp.float32), dense(wp.float32), wp.float32(1), wp.float32(1)),
                wp.bool,
            ),
            (
                is_close_scalar,
                (dense(wp.float64), dense(wp.float64), wp.float64(1), wp.float64(1)),
                wp.bool,
            ),
            (
                is_close_vec3,
                (dense(wp.vec3), dense(wp.vec3), wp.float32(1), wp.float32(1)),
                wp.bool,
            ),
            (
                is_close_vec3,
                (single(wp.vec3), single(wp.vec3), wp.float32(1), wp.float32(1)),
                wp.bool,
            ),
            (less, (dense(wp.float32), wp.float32(1)), wp.bool),
            (less, (dense(wp.int32), dense(wp.int32)), wp.bool),
            (nonzero_flag, (dense(wp.float32),), wp.int32),
            (nonzero_flag, (dense(wp.int32),), wp.int32),
            (nonzero_flag, (dense(wp.int8),), wp.int32),
        ]
    )


_declare_map_kernels()


@wp.func
def lowbias32(x: wp.uint32) -> wp.uint32:
    # The ``lowbias32`` finalizer: a **bijection** on uint32 whose output is decorrelated from its
    # input. Two properties, and the tree needs both.
    #
    # *Bijective*, so distinct inputs never collide: a priority drawn as ``lowbias32(index)`` is a
    # strict total order on the indices with no tie to break, which is what
    # ``polyline.ear_outranks`` relies on (its index tiebreak is dead code kept only in case the
    # mixer is ever swapped).
    #
    # *Decorrelating*, and this is load-bearing for any parallel independent-set pass.
    # ``edges_unique`` orders edges lexicographically by endpoint index, which on a structured mesh
    # is spatially *monotone*, and a monotone key field has essentially one local minimum -- so a
    # min-key lock commits a single winner per pass however many candidates there are. Hashing the
    # index restores the expected ~candidates/valence. Locking by a smoothly graded *cost* field
    # instead has the same defect, for the same reason.
    #
    # For a random order that a caller can *vary*, use ``random_priorities`` instead: it takes a
    # seed. This one is a pure function of the index, so it needs no state and is reproducible
    # across runs and across launches -- which is why the collapse pass can recompute a lock key in
    # a later kernel instead of storing it.
    x = (x ^ (x >> wp.uint32(16))) * wp.uint32(0x7FEB352D)
    x = (x ^ (x >> wp.uint32(15))) * wp.uint32(0x846CA68B)
    return x ^ (x >> wp.uint32(16))


@wp.func
def pack_directed_key(a: wp.int32, b: wp.int32, base: wp.uint64) -> wp.uint64:
    """Key of directed pair ``(a, b)``; ``(a, b)`` and ``(b, a)`` get different keys."""
    return wp.uint64(wp.uint32(a)) + wp.uint64(wp.uint32(b)) * base


@wp.func
def pack_edge_key(u: wp.int32, v: wp.int32, base: wp.uint64) -> wp.uint64:
    """Key of undirected edge (u, v); matches ``pack_indices`` for a sorted 2-index row."""
    return pack_directed_key(wp.min(u, v), wp.max(u, v), base)


@wp.func
def pack_index_triple(a: wp.int32, b: wp.int32, c: wp.int32, base: wp.uint64) -> wp.uint64:
    """Key ``grouping.pack_indices`` gives the three-column row ``(a, b, c)``, from registers."""
    # For a caller holding the row in registers -- ``pack_triangle_key`` below, and
    # ``kernels/remesh.py``'s cluster cell and face keys -- which would otherwise write an
    # ``(n, 3)`` table only for the packing to read. The digits are those of
    # ``grouping.pack_index_digit``; modular arithmetic makes the two groupings bit-identical.
    return pack_directed_key(a, b, base) + wp.uint64(wp.uint32(c)) * base * base


@wp.func
def pack_triangle_key(a: wp.int32, b: wp.int32, c: wp.int32, base: wp.uint64) -> wp.uint64:
    """Key of unoriented triangle ``(a, b, c)``: ``pack_indices`` of its sorted 3-index row."""
    s0, s1, s2 = sort3(a, b, c)
    return pack_index_triple(s0, s1, s2, base)


@wp.func
def unpack_edge_key(key: wp.uint64, base: wp.uint64) -> tuple[wp.int32, wp.int32]:
    """Endpoints ``(lo, hi)`` of a ``pack_edge_key`` key, min first as it was packed."""
    # Beside its inverse rather than at the one call site, because the packing is a *convention*
    # shared by ``face_edge_keys``, ``begin_decimation_pass`` and ``grouping.hash_indices_rows``: a
    # caller that recovers the endpoints by open-coding the divmod is one that can drift from it
    # silently.
    return wp.int32(key % base), wp.int32(key // base)


@wp.func
def complement_rank_index(index: wp.int32) -> wp.uint32:
    # ``index`` complemented within int32 so that a *smaller* index compares *larger* once packed
    # into the low half of a rank key. Shared by ``pack_farthest_key`` and ``pack_ranked_key``'s low
    # half, and by ``unpack_ranked_index``'s inverse -- the complement is its own inverse (comparing
    # ``INT32_MAX_CONSTANT`` on both sides), so one function serves both directions rather than the
    # same expression appearing three times.
    return wp.uint32(INT32_MAX_CONSTANT - index)


@wp.func
def float_order_bits(value: wp.float32) -> wp.uint64:
    # A **non-negative** float reinterpreted as the high half of a sortable integer key. The
    # IEEE-754 bits of a non-negative float increase monotonically with the value, so comparing
    # these bits as an unsigned integer is comparing the floats -- which is the whole trick behind
    # ``pack_farthest_key`` and ``pack_nearest_key`` below, and it is *only* valid because both
    # take a distance. A negative input orders backwards and silently.
    #
    # Named for the same reason ``complement_rank_index`` is: the convention is what a second
    # packer has to get right, and an expression repeated once per packer is one that can drift
    # from its precondition.
    return wp.uint64(wp.uint32(wp.cast(value, wp.int32)))


@wp.func
def pack_farthest_key(distance_sq: wp.float32, index: wp.int32) -> wp.int64:
    # One int64 whose ``wp.atomic_max`` is "largest distance, lowest index on a tie". The IEEE-754
    # bits of a non-negative float increase monotonically with the value, so the high half orders
    # by distance; the low half stores ``index`` complemented within int32 so that a *smaller*
    # index compares *larger*. Both halves are non-negative, so the whole key is, which is what
    # makes ``-1`` a sentinel below every real candidate. ``unpack_ranked_index`` inverts the
    # low half.
    distance_bits = float_order_bits(distance_sq)
    rank = wp.uint64(complement_rank_index(index))
    return wp.int64((distance_bits << wp.uint64(32)) | rank)


@wp.func
def pack_ranked_key(value: wp.int32, index: wp.int32) -> wp.int64:
    # The integer sibling of ``pack_farthest_key``: one int64 whose ``wp.atomic_max`` is "largest
    # value, lowest index on a tie". ``value`` must be non-negative (a count, a size, a degree), so
    # the high half orders by it directly with no bit trick; the low half stores ``index``
    # complemented within int32 so that a *smaller* index compares *larger*. Both halves are
    # non-negative, so ``-1`` is a sentinel below every real candidate.
    #
    # Comparing this key is also how a caller applies the result without a readback: recomputing
    # ``pack_ranked_key(value, index)`` in a second kernel and testing it against the reduced
    # maximum identifies the winner on the device. Where the host does need the index,
    # ``unpack_ranked_index`` inverts the low half.
    return wp.int64(
        (wp.uint64(wp.uint32(value)) << wp.uint64(32)) | wp.uint64(complement_rank_index(index))
    )


@wp.func
def unpack_ranked_index(key: wp.int64) -> wp.int32:
    # The index out of a ``pack_farthest_key`` / ``pack_ranked_key`` key -- the inverse of the
    # complement both use in their low half, so one decoder serves both. It lives here, beside its
    # packers, because a pack/unpack pair in two modules is a pair that cannot be read.
    #
    # Note the alternative a caller may prefer: ``repair.mark_largest_group_mask`` never unpacks at
    # all, it *recomputes* the key and tests it against the reduced maximum, keeping the winner on
    # the device instead of pulling it back to pick a row. Unpack when the host needs the index (a
    # greedy loop's next seed); recompute when only the device does.
    low = wp.int32(wp.uint32(wp.uint64(key) & wp.uint64(4294967295)))
    return wp.int32(complement_rank_index(low))


@wp.func
def pack_nearest_key(distance: wp.float32, index: wp.int32) -> wp.int64:
    # The ``min``-ordered twin of ``pack_farthest_key``: one int64 whose minimum is "smallest
    # distance, lowest index on a tie". Same ``float_order_bits`` high half (valid because a
    # distance is non-negative), but the index is stored plainly rather than complemented, because a
    # *smaller* index must now compare *smaller*. The key stays non-negative, so it needs no
    # sentinel.
    distance_bits = float_order_bits(distance)
    return wp.int64((distance_bits << wp.uint64(32)) | wp.uint64(wp.uint32(index)))


@wp.func
def lattice_position(
    lower: wp.vec3, step: wp.vec3, i: wp.int32, j: wp.int32, k: wp.int32
) -> wp.vec3:
    # World position of node ``(i, j, k)`` of a dense lattice whose node 0 sits at ``lower``.
    #
    # A **node** lattice, not a cell-centre one: there is no half-step shift, so a lattice of
    # ``dims`` nodes with ``step = extent / (dims - 1)`` spans its box inclusively at both ends.
    # Both callers want that -- ``reconstruction``'s signed-distance sample grid, in the row-major
    # order ``wp.MarchingCubes`` expects, and ``voxels.lattice`` -- and neither is a candidate for
    # ``wp.volume_index_to_world``, the spelling ``kernels/voxels.py``'s module docstring records
    # as preferred: both run *before* any ``wp.Volume`` exists, so there is no volume id to pass.
    # That is the same un-convertible half of the split that docstring names. Getting this wrong is
    # a rigid half-diagonal offset, which is exactly the failure mode a half-step convention has.
    #
    # It differs from the two kernels that share it only in the *destination's rank* -- a flat
    # row-major buffer against a ``wp.array3d`` -- so each keeps its own indexing and shares the
    # arithmetic, per section 3's "factor the family, not the pair".
    return lower + wp.cw_mul(step, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))


@wp.func
def trilinear_cell(coordinate: wp.vec3, shape: wp.vec3i) -> tuple[wp.vec3i, wp.vec3i, wp.vec3]:
    # Decompose a continuous lattice coordinate into the two corners of its cell and the three
    # fractional offsets. Exact integer and subtraction work, which is why the three callers share
    # it safely: extracting it reorders nothing and a float32 result cannot drift.
    #
    # **It returns the far corner rather than leaving the caller to write ``base + 1``, and that is
    # the whole point of the second return value.** Clamping the base into ``[0, shape - 2]`` bounds
    # the base and says nothing about the stencil, so on a degenerate axis (``shape[k] == 1``) the
    # base is 0 and ``base + 1`` is one slice past the end -- an out-of-bounds access at every one
    # of the four stencil corners on that axis. The fraction there is 0, so the *weight* is 0 and no
    # number is ever wrong; the address is computed and dereferenced regardless, which on the CPU
    # device is host-heap corruption (CLAUDE.md section 12.1) and in release mode is silent on both.
    # Handing back ``next_corner`` is what makes that unwriteable rather than merely documented.
    #
    # For any axis with two or more samples ``base <= shape - 2``, so ``next_corner`` is exactly
    # ``base + 1`` and every currently-legal lattice indexes bit-identically to before.
    #
    # ``shape[k] == 0`` is a distinct precondition this function does not guard: with a zero-size
    # axis there is no valid index at all, ``base`` clamps to 0 and ``next_corner`` clamps to -1, so
    # every one of the four callers below must keep rejecting it before this is ever reached (as
    # ``voxels.splat_onto_grid`` / ``sample_grid_trilinear`` already do with a `shape` validation,
    # and as every ``res`` this module's own callers derive is bounded well above zero). Unlike
    # the ``shape[k] == 1`` case above, there is no in-range corner to hand back for a zero-size
    # axis, so the fix there does not generalize here.
    limit = wp.vec3i(wp.max(shape[0] - 2, 0), wp.max(shape[1] - 2, 0), wp.max(shape[2] - 2, 0))
    base = wp.vec3i(
        wp.clamp(wp.int32(wp.floor(coordinate[0])), 0, limit[0]),
        wp.clamp(wp.int32(wp.floor(coordinate[1])), 0, limit[1]),
        wp.clamp(wp.int32(wp.floor(coordinate[2])), 0, limit[2]),
    )
    next_corner = wp.vec3i(
        wp.min(base[0] + 1, shape[0] - 1),
        wp.min(base[1] + 1, shape[1] - 1),
        wp.min(base[2] + 1, shape[2] - 1),
    )
    return (
        base,
        next_corner,
        wp.vec3(
            wp.clamp(coordinate[0] - wp.float32(base[0]), 0.0, 1.0),
            wp.clamp(coordinate[1] - wp.float32(base[1]), 0.0, 1.0),
            wp.clamp(coordinate[2] - wp.float32(base[2]), 0.0, 1.0),
        ),
    )


@wp.func
def trilinear_corner(base: wp.vec3i, next_corner: wp.vec3i, offset: wp.vec3i) -> wp.vec3i:
    # Pick one of the eight corners a ``trilinear_cell`` decomposition addresses, given the 0/1
    # offset per axis. Paired with ``trilinear_weight``, which takes the same three offsets and
    # returns that corner's weight, so the two cannot disagree about which corner is being named.
    return wp.vec3i(
        wp.where(offset[0] == 0, base[0], next_corner[0]),
        wp.where(offset[1] == 0, base[1], next_corner[1]),
        wp.where(offset[2] == 0, base[2], next_corner[2]),
    )


@wp.func
def trilinear_weight(
    fractions: wp.vec3, offset_x: wp.int32, offset_y: wp.int32, offset_z: wp.int32
) -> wp.float32:
    # Weight of one of the eight corners a ``trilinear_cell`` decomposition addresses. The eight
    # sum to exactly 1, which is what makes ``scatter.splat_grid_trilinear`` and
    # ``interpolation.sample_grid_trilinear`` transposes of each other -- and, measured, what makes
    # a splatted density sum to the point count exactly.
    return (
        wp.where(offset_x == 0, 1.0 - fractions[0], fractions[0])
        * wp.where(offset_y == 0, 1.0 - fractions[1], fractions[1])
        * wp.where(offset_z == 0, 1.0 - fractions[2], fractions[2])
    )


# ---------------------------------------------------------------------------
# Segmented copy: many separate arrays into one buffer, in a single launch.
#
# ``array._pack_segments`` issues one ``wp.copy`` per segment on the small-segment path, a host
# cost linear in the segment count and independent of the data volume. **Warp does have an
# array-of-arrays**, contrary to a long-standing comment there: a ``@wp.struct`` may carry a
# ``wp.array`` field, and a ``wp.array`` of that struct is exactly a descriptor table, which
# ``segments[s].data`` indexes from kernel scope.
#
# The kernel form is **flat in both the segment count and the data volume**, where the copy loop is
# linear in the segment count, so the two cross at a couple of dozen segments and the kernel wins by
# orders of magnitude above that. The whole choice is therefore a segment-count threshold; see
# ``array.PACK_SEGMENTS_KERNEL_FROM``.
#
# **One kernel serves every dtype**, rather than a table of them. The descriptor's array field is
# declared ``wp.array[wp.int32]`` and pointed at the segment's storage with a length in 4-byte
# *words*, and the destination is aliased the same way, so this is a byte copy that never names the
# caller's dtype -- verified byte-identical for ``int32``, ``float32``, ``float64``, ``vec3`` and
# ``uint64``. A dtype whose itemsize is not a multiple of 4 (``bool``, ``int8``, ``int16``) cannot
# be addressed this way and keeps the copy loop.
# ---------------------------------------------------------------------------


@wp.struct
class WordSegment:
    """One source segment of a packed copy, addressed as 4-byte words."""

    data: wp.array[wp.int32]
    offset: wp.int32
    count: wp.int32


@wp.kernel
def pack_segment_words(
    segments: wp.array[WordSegment], width: wp.int32, out_flat: wp.array[wp.int32]
) -> None:
    # Copy every segment into its slot of ``out_flat``, one launch for all of them.
    #
    # Launched ``dim=(n_segments, width)`` with ``width`` the host's ``min(longest, cap)``, and each
    # thread strides its own segment by ``width``. **The stride is what bounds the grid**: a plain
    # ``dim=(n_segments, longest)`` is mostly threads that exit immediately whenever the split is
    # uneven -- 670 M of them for a 256-way split whose first piece holds nearly all of a 2.6 M
    # buffer. This is an ordinary grid stride and not section 2.2's ``wp.block_dim()`` case: these
    # lanes do not cooperate, there is no tile here, and every thread of a plain ``wp.launch``
    # grid runs on both devices, so taking the stride from the grid's own second dimension is
    # correct on CPU as well.
    s, t = wp.tid()
    segment = segments[s]
    for k in range(t, segment.count, width):
        out_flat[segment.offset + k] = segment.data[k]


@wp.kernel
def unpack_segment_words(
    flat: wp.array[wp.int32], width: wp.int32, out_segments: wp.array[WordSegment]
) -> None:
    # The inverse of ``pack_segment_words``: copy each slot of ``flat`` out into its own segment,
    # which is how ``array.split(copy=True)`` fills its independent allocations in one launch.
    # Same grid and stride as the packing direction; ``out_segments`` is the descriptor table of the
    # destinations, so the writes land in the storage its ``data`` fields point at.
    s, t = wp.tid()
    segment = out_segments[s]
    for k in range(t, segment.count, width):
        segment.data[k] = flat[segment.offset + k]
