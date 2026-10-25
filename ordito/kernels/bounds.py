import math

import warp as wp

from ordito.constants import FLOAT32_INF_CONSTANT
from ordito.kernels.array import RegisterBlockedTable, atomic_min_packed_box, scanned_count
from ordito.kernels.predicates import TWO_PI_F64, is_in_aabb, is_in_obb
from ordito.kernels.reduce import block_argmin
from ordito.kernels.selection import mark_submesh_face
from ordito.kernels.triangles import corner_triple

wp.set_module_options({"enable_backward": False})

# Super-Fibonacci spiral constants [Alexa 2022]: the two irrational strides whose phase pair
# equidistributes over SO(3). Held as reciprocals, and multiplied rather than divided by, so the
# candidate set is bit-comparable with ``igl::super_fibonacci``'s. The full turn they scale is
# ``kernels/predicates.TWO_PI_F64`` -- one definition, because a second spelling of 2*pi is a
# second thing to keep in step for no gain.
SUPER_FIBONACCI_RSQRT2 = wp.constant(wp.float64(1.0 / math.sqrt(2.0)))
SUPER_FIBONACCI_RPSI = wp.constant(wp.float64(1.0 / 1.533751168755204288118041))


@wp.func
def super_fibonacci_quat(i: wp.int32, n: wp.int32) -> wp.vec4d:
    """
    Draw the ``i``-th of ``n`` Super-Fibonacci quaternions [Alexa 2022], as ``(x, y, z, w)``.

    The low-discrepancy sample of SO(3) both axis kernels below draw from, which each had spelled
    out: the sampled candidate set, and the deltas a refinement round shrinks toward the identity.
    One definition because the sequence is a *contract* between them -- the refinement's
    monotonicity argument rests on its last element being the identity rotation -- and a drifted
    stride or a permuted component order would still return plausible frames.

    Returned in float64 and narrowed by the caller: the phases reach ``2 * pi * n`` (~6e4 radians at
    igl's default), where float32 argument reduction has already lost four digits of the angle and
    the low-discrepancy property with it. The constants are reciprocals multiplied rather than
    divided by, so the candidate set stays bit-comparable with ``igl::super_fibonacci``'s.
    """
    s = wp.float64(i) + wp.float64(0.5)
    phase = TWO_PI_F64 * s
    alpha = phase * SUPER_FIBONACCI_RSQRT2
    beta = phase * SUPER_FIBONACCI_RPSI
    height = s / wp.float64(n)
    radius = wp.sqrt(height)
    radius_conjugate = wp.sqrt(wp.float64(1.0) - height)
    return wp.vec4d(
        radius * wp.sin(alpha),
        radius * wp.cos(alpha),
        radius_conjugate * wp.sin(beta),
        radius_conjugate * wp.cos(beta),
    )


@wp.func
def seed_packed_box(out_corners: wp.array[wp.float32], box: wp.int32) -> None:
    """
    Reset box ``box``'s six ``[min, -max]`` slots to ``+inf``, the empty box every update narrows.

    Called by the two kernels that write a round's candidate frames, one thread per candidate, so
    each candidate's accumulator is reset by the thread that writes its frame -- rather than by a
    whole-buffer ``fill_`` before every
    [`oriented_box_extents`][ordito.kernels.bounds.oriented_box_extents] launch.
    """
    base = box * 6
    for c in range(6):
        out_corners[base + c] = FLOAT32_INF_CONSTANT


@wp.kernel
def oriented_box_candidate_axes(
    n_rotations: wp.int32, out_axes: wp.array[wp.mat33], out_corners: wp.array[wp.float32]
) -> None:
    # Candidate box orientations, as world -> box frames whose *rows* are the box axes, each with
    # its extent accumulator seeded (``seed_packed_box``).
    #
    # The set is the Super-Fibonacci spiral [Alexa 2022] — the same low-discrepancy sampling of
    # SO(3) ``igl::oriented_bounding_box`` searches — with the identity as the **last** candidate,
    # so a returned box can never be worse than the axis-aligned one and ``n_rotations = 1`` reduces
    # to exactly the axis-aligned reduction.
    i = wp.int32(wp.tid())
    seed_packed_box(out_corners, i)
    n_spiral = n_rotations - 1
    if i >= n_spiral:
        out_axes[i] = wp.identity(n=3, dtype=wp.float32)
        return

    q = super_fibonacci_quat(i, n_spiral)
    rotation = wp.quat_to_matrix(
        wp.quat(wp.float32(q[0]), wp.float32(q[1]), wp.float32(q[2]), wp.float32(q[3]))
    )
    # The quaternion names a box -> world rotation; the extent reduction wants world -> box.
    out_axes[i] = wp.transpose(rotation)


@wp.func
def refine_candidate(
    base: wp.mat33, p: wp.int32, count: wp.int32, angle_scale: wp.float64
) -> wp.mat33:
    """
    Return candidate ``p`` of the ``count``-frame trust-region ball around ``base``.

    The Super-Fibonacci sample of SO(3), geodesically shrunk toward the identity (each rotation
    angle scaled by ``sigma / pi``) and composed onto the chain's base frame. Draws the same
    [`super_fibonacci_quat`][ordito.kernels.bounds.super_fibonacci_quat] sequence
    ``oriented_box_candidate_axes`` samples, so the last delta of every ball is the identity --
    which re-scores the base and keeps each chain monotone. Shared by the two kernels that write a
    round's frames: the seeding walk (the first round) and each round's selection (the next one).
    """
    if p == count - 1:
        return base

    delta_quat = super_fibonacci_quat(p, count - 1)
    qx = delta_quat[0]
    qy = delta_quat[1]
    qz = delta_quat[2]
    qw = delta_quat[3]
    if qw < wp.float64(0.0):  # same rotation, angle in [0, pi]
        qx = -qx
        qy = -qy
        qz = -qz
        qw = -qw
    angle = wp.float64(2.0) * wp.acos(qw)
    axis_norm = wp.sqrt(qx * qx + qy * qy + qz * qz)
    ax = wp.float64(1.0)
    ay = wp.float64(0.0)
    az = wp.float64(0.0)
    if axis_norm > wp.float64(1e-12):
        ax = qx / axis_norm
        ay = qy / axis_norm
        az = qz / axis_norm
    shrunk_half = wp.float64(0.5) * angle * angle_scale
    sin_half = wp.sin(shrunk_half)
    delta = wp.quat_to_matrix(
        wp.quat(
            wp.float32(ax * sin_half),
            wp.float32(ay * sin_half),
            wp.float32(az * sin_half),
            wp.float32(wp.cos(shrunk_half)),
        )
    )
    return delta * base


# Candidate frames one ``oriented_box_extents`` thread may project into together; a launch takes
# the widest ``RegisterBlockedTable.launch_shape`` allows, as
# ``kernels/points.hull_support_extremes`` does.
BOX_EXTENT_WIDTHS = (4, 2, 1)


def _oriented_box_extents_kernel(width: int) -> wp.Kernel:
    """Build ``oriented_box_extents`` over ``width`` candidate frames per thread."""
    corners_t = wp.types.matrix(shape=(width, 3), dtype=wp.float32)

    def oriented_box_extents(
        points: wp.array[wp.vec3],
        axes: wp.array[wp.mat33],
        n_slices: wp.int32,
        out_corners: wp.array[wp.float32],
    ) -> None:
        # Extent of the cloud in every candidate frame: six slots per candidate, packed ``[min_x,
        # min_y, min_z, -max_x, -max_y, -max_z]`` exactly as ``kernels/reduce.py``'s
        # ``minmax_vec3_chunked`` packs its one box, so a single ``+inf`` seed covers both ends and
        # every update is an ``atomic_min``. The seed is written by the kernel that produced
        # ``axes`` (``seed_packed_box``).
        #
        # Strided slice rather than a contiguous chunk, and lane-free, for the same two reasons as
        # ``kernels/points.py::hull_support_extremes``: consecutive threads read consecutive points
        # so the loads coalesce, and the threads partition the **outer** work -- the cloud -- rather
        # than a sequence one block owns, so there is no ``wp.block_dim()`` to stride by and a
        # ``wp.tile(...)`` reduction cannot be reached without changing the launch. See
        # ``kernels/visibility.py::obscurance`` for a lane-parallel kernel on the other side of it.
        # Converting this one is declined on the measurement ``hull_support_extremes`` carries: the
        # slice dimension is what fills the device, so one block per candidate frame loses badly on
        # a large cloud.
        #
        # Thread ``(b, j)`` takes frames ``b * width ..`` over slice ``j``: each point it loads is
        # projected into every one of them from registers, so the cloud is streamed once per
        # ``width`` frames, which is what bounds the kernel. Component-wise min and max are exact,
        # so the grouping changes no corner.
        b, j = wp.tid()
        n_points = points.shape[0]
        n_frames = axes.shape[0]
        k0 = b * width
        lower = corners_t()
        upper = corners_t()
        for d in range(width):
            lower[d] = wp.vec3(FLOAT32_INF_CONSTANT, FLOAT32_INF_CONSTANT, FLOAT32_INF_CONSTANT)
            upper[d] = -lower[d]
        for i in range(j, n_points, n_slices):
            point = points[i]
            for d in range(width):
                # A slot past the last frame re-projects into it; only the commit skips it.
                local = axes[wp.min(k0 + d, n_frames - 1)] * point
                lower[d] = wp.min(lower[d], local)  # component-wise on a vector
                upper[d] = wp.max(upper[d], local)

        for d in range(width):
            # A slice past the end of the cloud contributes nothing.
            if k0 + d < n_frames and upper[d, 0] > -FLOAT32_INF_CONSTANT:
                atomic_min_packed_box(out_corners, k0 + d, lower[d], upper[d])

    return wp.kernel(oriented_box_extents, name=f"oriented_box_extents_{width}")


OBB_EXTENTS = RegisterBlockedTable(
    "oriented_box_extents", _oriented_box_extents_kernel, BOX_EXTENT_WIDTHS
)


BOX_OBJECTIVE_VOLUME = wp.constant(wp.int32(0))
BOX_OBJECTIVE_SURFACE_AREA = wp.constant(wp.int32(1))
BOX_OBJECTIVE_DIAGONAL = wp.constant(wp.int32(2))


@wp.func
def packed_box_sides(corners: wp.array[wp.float32], box: wp.int32) -> wp.vec3:
    """Side lengths of the box in the six ``[min, -max]`` slots at ``box``."""
    base = box * 6
    return wp.vec3(
        -corners[base + 3] - corners[base],
        -corners[base + 4] - corners[base + 1],
        -corners[base + 5] - corners[base + 2],
    )


@wp.func
def box_objective_loss(sides: wp.vec3, objective: wp.int32) -> wp.float32:
    """
    Score a box's sides under one of the three objectives ``oriented_bounding_box`` exposes.

    The branch is warp-uniform -- every thread in the launch is handed the same ``objective`` -- so
    this is the int-selector form rather than three kernels.
    """
    if objective == BOX_OBJECTIVE_VOLUME:
        return sides[0] * sides[1] * sides[2]
    if objective == BOX_OBJECTIVE_SURFACE_AREA:
        return wp.float32(2.0) * (sides[0] * sides[2] + sides[1] * sides[0] + sides[2] * sides[1])
    return wp.dot(sides, sides)


# Width of a chain-state row: the running loss, then the winning box's ``lower`` and ``upper``
# corners, then its frame. One row is everything a finished chain has to say, so the whole search
# reads back once at the end rather than once per round.
BOX_STATE_COLUMNS = 16

# Two frames belong to the same basin when the rotation taking one to the other is under 0.2 rad.
# Tested as a trace rather than an angle: ``wp.ddot`` of two rotations is ``1 + 2 cos(theta)`` and
# ``acos`` is decreasing, so the comparison needs no inverse trig and no clamp.
BOX_SEED_SAME_BASIN_TRACE = wp.constant(wp.float32(1.0 + 2.0 * math.cos(0.2)))


@wp.func
def write_chain_state(
    out_state: wp.array2d[wp.float32],
    chain: wp.int32,
    loss: wp.float32,
    corners: wp.array[wp.float32],
    box: wp.int32,
    frame: wp.mat33,
) -> None:
    # One chain's whole answer, into one ``BOX_STATE_COLUMNS``-wide row. Shared by the two kernels
    # that write it -- the seeding pass and each refinement round -- because the column layout is a
    # contract between them and the wrapper that unpacks it, and three copies of an offset table is
    # how one of them ends up reading the frame out of the corners' slots.
    out_state[chain, 0] = loss
    base = box * 6
    for c in range(3):
        out_state[chain, 1 + c] = corners[base + c]
        out_state[chain, 4 + c] = -corners[base + 3 + c]  # slots 3..5 hold the negated upper corner
    for r in range(3):
        for c in range(3):
            out_state[chain, 7 + 3 * r + c] = frame[r, c]


@wp.kernel
def oriented_box_losses(
    corners: wp.array[wp.float32], objective: wp.int32, out_loss: wp.array[wp.float32]
) -> None:
    # Score every candidate's box under the requested objective, one thread per candidate. Keeps
    # the ``(n_candidates, 6)`` extent table on the device, where the only consumers -- the seeding
    # walk below and each refinement round -- already live.
    #
    # Scoring the table in ``oriented_box_seed_chains``' prologue instead (each lane its own stride,
    # so no barrier) was measured and declined: 0.90-1.01x on ``oriented_bounding_box`` from
    # ``bunny_decimated`` to ``dragon``, byte-identical. The one block's lanes score 4 096 boxes
    # where this grid spreads them over the device, which costs what the launch saved.
    box = wp.int32(wp.tid())
    out_loss[box] = box_objective_loss(packed_box_sides(corners, box), objective)


@wp.kernel
def oriented_box_seed_chains(
    loss: wp.array[wp.float32],
    axes: wp.array[wp.mat33],
    corners: wp.array[wp.float32],
    window: wp.int32,
    n_chains: wp.int32,
    angle_scale: wp.float64,
    count_per_chain: wp.int32,
    out_state: wp.array2d[wp.float32],
    out_axes: wp.array[wp.mat33],
    out_corners: wp.array[wp.float32],
) -> None:
    """
    Choose the refinement's starting frames: the loss table's head, greedily spread across basins.

    Adjacent spiral candidates score adjacently, so the best few are usually one basin sampled
    several times; walking the leading ``window`` of them and keeping only frames at least 0.2 rad
    apart gives the chains *different* basins to descend. The first pick is the global argmin, which
    is what keeps the refined answer from ever losing to the sampled one.

    Launched ``wp.launch_tiled(dim=1)``: one block, whose lanes stride the loss table by
    ``wp.block_dim()`` and fold with [`block_argmin`][ordito.kernels.reduce.block_argmin]. On the
    CPU device that stride is 1, so the single lane walks the whole table and each fold is a
    one-element tile holding its own answer.

    A single walk covers both halves of the rule. The shortfall fill -- take the leading candidates
    the spread rejected, in order -- can only run when the walk reached the end of the window with
    fewer than ``n_chains`` picks, so everything it rejected is still a legal fallback and can be
    banked on the way past rather than re-enumerated.

    With ``count_per_chain > 0`` the lanes also write the first refinement round's
    ``count_per_chain`` frames per chain into ``out_axes`` and seed their extent slots in
    ``out_corners`` ([`refine_candidate`][ordito.kernels.bounds.refine_candidate]): every lane holds
    the same picks, so the frames need no barrier, and the round needs no axes launch of its own.
    """
    _block, lane = wp.tid()
    n_boxes = loss.shape[0]

    picked = wp.vec4i(-1, -1, -1, -1)
    reserve = wp.vec3i(-1, -1, -1)  # rejects, oldest first: at most ``n_chains - 1`` are ever used
    n_picked = wp.int32(0)
    n_reserve = wp.int32(0)
    # The walk's cursor, as the (loss, index) pair already taken. Ordering by the index as well as
    # the loss is what makes a run of equal losses advance instead of returning its first member
    # for ever, and it settles ties by candidate rather than by lane.
    taken_loss = wp.float32(-FLOAT32_INF_CONSTANT)
    taken_box = wp.int32(-1)

    for _round in range(window):
        # The next candidate in ascending (loss, index) order. ``<`` rather than ``<=`` inside the
        # lane so the lowest index wins there too; ``block_argmin`` applies the same rule across
        # lanes, and returns the ``-1`` seed when no lane found anything left to take.
        best_loss = wp.float32(FLOAT32_INF_CONSTANT)
        best_box = wp.int32(-1)
        for box in range(lane, n_boxes, wp.block_dim()):
            value = loss[box]
            if (
                value > taken_loss or (value == taken_loss and box > taken_box)
            ) and value < best_loss:
                best_loss = value
                best_box = box
        taken_loss, taken_box = block_argmin(best_loss, best_box)
        if taken_box < 0:
            break

        candidate = axes[taken_box]
        same_basin = wp.int32(0)
        for q in range(n_picked):
            if wp.ddot(candidate, axes[picked[q]]) > BOX_SEED_SAME_BASIN_TRACE:
                same_basin = wp.int32(1)
        if same_basin == 0:
            picked[n_picked] = taken_box
            n_picked += 1
        elif n_reserve < 3:
            reserve[n_reserve] = taken_box
            n_reserve += 1
        if n_picked >= n_chains:
            break

    for q in range(n_reserve):
        if n_picked < n_chains:
            picked[n_picked] = reserve[q]
            n_picked += 1
    # Fewer distinct candidates than chains -- only reachable below ``rotations = n_chains``.
    # Repeat the last pick rather than leave a chain holding an unwritten frame; the duplicates
    # converge to the same place and the final argmin keeps one of them.
    for c in range(1, n_chains):
        if picked[c] < 0:
            picked[c] = picked[c - 1]

    if lane == 0:
        for c in range(n_chains):
            # The clamp is unreachable once the fill and the padding above have run -- ``picked[0]``
            # is always set for a non-empty table, and the pad propagates it. It is here because
            # the alternative to a range check on an index that reached a *global* read through two
            # conditional fills is an out-of-bounds load, and nothing downstream would report one:
            # a chain seeded from garbage simply loses the final argmin, so the answer still comes
            # out right and no test would fail.
            box = wp.max(picked[c], 0)
            # ``+inf`` rather than the seed's own loss, so the first refinement round always
            # improves on it and each chain stays monotone. The corners and frame are real: they
            # are what a caller asking for no refinement gets back.
            write_chain_state(out_state, c, FLOAT32_INF_CONSTANT, corners, box, axes[box])

    for i in range(lane, n_chains * count_per_chain, wp.block_dim()):
        seed_packed_box(out_corners, i)
        base = axes[wp.max(picked[i // count_per_chain], 0)]
        out_axes[i] = refine_candidate(base, i % count_per_chain, count_per_chain, angle_scale)


@wp.func
def chain_frame(state: wp.array2d[wp.float32], chain: wp.int32) -> wp.mat33:
    """Return the frame a ``write_chain_state`` row holds: the chain's base."""
    frame = wp.mat33()
    for r in range(3):
        for c in range(3):
            frame[r, c] = state[chain, 7 + 3 * r + c]
    return frame


@wp.kernel
def oriented_box_select_chains(
    corners: wp.array[wp.float32],
    axes: wp.array[wp.mat33],
    chain_state: wp.array2d[wp.float32],
    count_per_chain: wp.int32,
    objective: wp.int32,
    angle_scale: wp.float64,
    out_state: wp.array2d[wp.float32],
    out_axes: wp.array[wp.mat33],
    out_corners: wp.array[wp.float32],
) -> None:
    """
    Keep each chain's best candidate of this round, and write the next round's frames around it.

    One thread per next-round candidate. Each walks its chain's ``count_per_chain`` block of
    scored candidates and takes the argmin of the objective, which replaces the chain only when the
    round improved on it -- what makes the refinement monotone per chain. Every thread of a chain
    computes the same winner, so each writes its own frame of the next ball
    ([`refine_candidate`][ordito.kernels.bounds.refine_candidate]) and seeds its extent slots with
    no barrier, and the chain's first thread writes its row of ``out_state``. The inputs and
    outputs are two halves of one ping-ponged buffer each, so no thread overwrites what another of
    this launch still reads.

    ``chain_state`` is one [`write_chain_state`][ordito.kernels.bounds.write_chain_state] row per
    chain, whose frame is the chain's base, so the whole refinement reads back once at the end
    instead of once per round.
    """
    i = wp.int32(wp.tid())
    chain = i // count_per_chain
    p = i % count_per_chain
    base = chain * count_per_chain
    best_row = wp.int32(-1)
    best_loss = chain_state[chain, 0]
    for k in range(count_per_chain):
        row = base + k
        loss = box_objective_loss(packed_box_sides(corners, row), objective)
        if loss < best_loss:
            best_loss = loss
            best_row = row
    frame = chain_frame(chain_state, chain)
    if best_row >= 0:
        frame = axes[best_row]
    if p == 0:
        if best_row >= 0:
            write_chain_state(out_state, chain, best_loss, corners, best_row, frame)
        else:
            for c in range(BOX_STATE_COLUMNS):
                out_state[chain, c] = chain_state[chain, c]
    seed_packed_box(out_corners, i)
    out_axes[i] = refine_candidate(frame, p, count_per_chain, angle_scale)


@wp.func
def packed_box_diagonal(corners: wp.array[wp.float32], box: wp.int32) -> wp.float32:
    """Diagonal length of box ``box`` of a six-slot ``[min, -max]`` packing; zero when empty."""
    # The shared reader for the packing
    # [`oriented_box_extents`][ordito.kernels.bounds.oriented_box_extents] and
    # [`scatter_group_bounds`][ordito.kernels.scatter.scatter_group_bounds] both write.
    #
    # A box nothing accumulated into still holds the ``+inf`` seed in both halves, so its extent
    # comes out negative. That reads as **zero** rather than as ``nan``, which is what lets a caller
    # threshold every box uniformly instead of masking the empty slots first.
    extent = packed_box_sides(corners, box)
    if extent[0] < wp.float32(0.0):
        return wp.float32(0.0)
    return wp.length(extent)


@wp.func
def point_in_box(
    point: wp.vec3, rotation: wp.mat33, min_bound: wp.vec3, max_bound: wp.vec3, oriented: wp.int32
) -> wp.bool:
    # "Inside the box" for the box kernels below. ``oriented`` is a warp-uniform selector, not a
    # rotation by the identity: ``0 * inf`` is ``nan``, so the identity product would move a point
    # at an infinity into the box's ``nan`` convention. The two arms are the predicates
    # ``points_in_aabb_mask`` / ``points_in_obb_mask`` map, so the index, crop and mask forms
    # cannot disagree.
    if oriented != 0:
        return is_in_obb(point, rotation, min_bound, max_bound)
    return is_in_aabb(point, min_bound, max_bound)


@wp.kernel
def box_flags(
    points: wp.array[wp.vec3],
    rotation: wp.mat33,
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    oriented: wp.int32,
    out_flags: wp.array[wp.int32],
) -> None:
    # ``1`` / ``0`` per point for "inside the box", written as the ``int32`` an in-place
    # ``wp.utils.array_scan`` reads -- the index and crop forms of ``bounds.points_in_*`` scan this
    # directly instead of mapping a ``wp.bool`` mask and widening it.
    i = wp.int32(wp.tid())
    inside = point_in_box(points[i], rotation, min_bound, max_bound, oriented)
    out_flags[i] = wp.where(inside, 1, 0)


@wp.func
def corner_in_box(
    vertices: wp.array[wp.vec3],
    v: wp.int32,
    rotation: wp.mat33,
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    oriented: wp.int32,
) -> wp.bool:
    # ``point_in_box`` at vertex ``v``, ``False`` for an index outside ``vertices`` -- the answer
    # ``selection.masked_at`` gives a vertex mask, so a malformed face buffer reads no further.
    if v < 0 or v >= vertices.shape[0]:
        return False
    return point_in_box(vertices[v], rotation, min_bound, max_bound, oriented)


@wp.kernel
def mark_submesh_in_box(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    rotation: wp.mat33,
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    oriented: wp.int32,
    out_face_mask: wp.array[wp.bool],
    out_flags: wp.array[wp.int32],
) -> None:
    # ``crop_mesh``'s marking pass: ``kernels/selection.mark_submesh_by_vertex_mask`` with
    # ``require_all`` and the box test in place of the vertex mask, so no mask is written. A
    # corner shared by several faces is tested once per face, which costs less than the mask's
    # launch and buffer.
    f = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, f)
    selected = (
        corner_in_box(vertices, a, rotation, min_bound, max_bound, oriented)
        and corner_in_box(vertices, b, rotation, min_bound, max_bound, oriented)
        and corner_in_box(vertices, c, rotation, min_bound, max_bound, oriented)
    )
    out_face_mask[f] = selected
    if selected:
        mark_submesh_face(faces, out_face_mask.shape[0], f, out_flags)


@wp.kernel
def compact_scanned_points(
    inclusive: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    out_kept: wp.array[wp.vec3],
    out_indices: wp.array[wp.int32],
) -> None:
    # ``kernels/scatter.scatter_index_where_scanned`` plus the gather it would feed: every selected
    # point lands at its exclusive rank together with its index, so a crop is one compaction over
    # the in-place scan of its flags rather than an index scatter and a gathered copy.
    # ``out_indices`` may be a null descriptor (``None`` at the launch) when only the points are
    # wanted.
    i = wp.int32(wp.tid())
    start, count = scanned_count(inclusive, i)
    if count != 0:
        out_kept[start] = points[i]
        if out_indices.shape[0] > 0:
            out_indices[start] = i
