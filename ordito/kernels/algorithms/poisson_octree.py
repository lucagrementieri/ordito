"""
Screened-Poisson kernels that solve PoissonRecon's own octree system, on per-depth lattices.

Backs [`screened_poisson`][ordito.reconstruction.screened_poisson]'s ``method="octree"``. The
Python-scope orchestration lives in ``reconstruction._screened_poisson_octree``. Everything is in
the unit cube the cloud is mapped into (``scale`` pads it); depth ``d`` splits it into ``2**d``
cells per axis.

**Two bases per depth.** The indicator is a sum over depths of degree-1 (trilinear hat) functions,
one per lattice node ``0..2**d`` with free (Neumann) walls; the normal field is a sum of
cell-centred quadratic B-splines, one per cell, reflected oddly at the walls (they vanish there).
Depth ``d``'s lattice has ``(2**d + 1)**3`` nodes, but the right-hand sides and the fields are
stored only for the ``4**3``-node bricks a read can reach (the stored nodes section). Cells are
dense below the finest depth (``8**d``, every such depth packed in one buffer at
``cell_offsets[d]``); the finest depth's cell tables (the sample slots and its normal
coefficients) are stored by ``4**3`` brick, for the bricks a sample or a finest-depth normal splat
reaches (``octree_finest_bricks``), and a brick table numbers them. A cell outside them holds no
sample and a zero coefficient, which every reader already skips, so the layout changes no value.

**Samples.** Points are merged per finest cell (summed position, unit normal and count). The
density of a depth ``d`` cell is the splat of its children's ``count / samples_per_node`` at their
centroids into the 27 quadratic B-splines around it, at every depth up to ``depth - 2``; a sample's
depth estimate reads it there (``octree_sample_depth``) and sets its normal's splat depth and its
area weight ``4**-depth``.

**An upstream quirk is reproduced, not fixed.** While the splats run, PoissonRecon's octree root
and the root's parent brood both sit at depth 0, so the neighbour key computes the root's window in
place over its parent's: every window slot past a high face of the cube resolves to the *real*
depth ``d + 1`` node with the same integer offset -- a node on the ``x``, ``y`` or ``z = 0.5``
plane (``splat_target``). Slots past a low face do not exist. Both the density and the normal splat
land there, and every later read of those nodes sees them; without it the sample weights of points
within a cell of a high face, and through them the surface near the cube's midplanes, move by up to
a few tenths of a percent.

**Which nodes are unknowns.** Depths up to ``full_depth`` are complete. Deeper, a depth ``d`` brood
(the eight children of a depth ``d - 1`` cell) exists when that cell's subtree holds normal data,
and then, finest depth first, every node with active children activates the broods of its depth's
nodes within two of it (``octree_clip_flags``, ``octree_dilate_flags``). ``flags`` holds that
"children are active" bit per cell, each depth padded by one cell on both sides.

**The system.** Depth ``d``'s operator is the exact hat-function stiffness plus a screening term
``alpha * 2**d * W * phi phi^T`` at the centroid of every active occupied depth ``d`` cell (``W``
its point count); the right-hand side is ``-int grad(phi) . V`` over the whole multi-depth normal
field plus ``alpha / 2 * 2**d * W * phi``. Own-depth and finer data reach a depth exactly (the
finer by restriction, the hats being nested); coarser data reaches it only through its parent
depth, after being carried up by the hat prolongation applied to the B-spline coefficients -- the
upstream solver's own shortcut, reproduced (``octree_upsample_coarse_data``).

**The solve is one cascadic pass, not a converged one.** Depth 0 (eight unknowns) is solved on the
host; every finer depth starts its correction at zero against the residual of the coarser solution
and runs ``iterations`` Gauss-Seidel sweeps, slices from the top ``z`` down, four colours a slice
in ascending ``x % 2 + 2 (y % 2)`` order -- the upstream order, sweep for sweep. The sweeps run as a
wavefront (``octree_gauss_seidel_wave``): sweep ``s`` on slice ``z`` only waits for sweep ``s`` on
``z + 1`` and sweep ``s - 1`` on ``z - 1``, so one launch relaxes every ``(s, z)`` on a diagonal
and a level costs ``4 * (res + 2 * (iterations - 1))`` launches instead of ``4 * res *
iterations``, with the same values. The CPU runs a launch's threads one after another anyway, so
there one launch a level runs every wave in that order (``octree_gauss_seidel_depth``).

**Extraction** is the upstream's, cell by cell (the extraction section's note): corner values
copied up from the deepest leaf at each point, edge vertices placed by the upstream non-linear fit
and shared with finer neighbours (taken over or paired), each leaf's faces by marching squares or
borrowed from the finer leaves across them in the upstream's single-threaded order, loops walked as
the upstream walks them and triangulated by its minimal-area rule. Its triangles are the upstream's
up to the splits its own rounding decides (two splits of a loop whose areas agree to a few
``1e-6``), which a field computed in another order can decide the other way.
"""

import warp as wp

from ordito.kernels.array import (
    binary_search_index_left,
    bit_is_set,
    bit_rank,
    declare_map_signatures,
    map_probe,
    map_probe_single,
    trilinear_cell,
)
from ordito.kernels.reconstruction import (
    poisson_grid_index,
    poisson_lerp_corners,
    poisson_restrict_weight,
)
from ordito.kernels.reduce import GroupSumPass
from ordito.kernels.scatter import scatter_add, scatter_add_vector

wp.set_module_options({"enable_backward": False})

# PoissonRecon's density kernel: a unit splat's peak, ``1 / sum(beta_2(1/2)**2)`` over the three
# component values at a cell centre, ``1 / (1/64 + 9/16 + 1/64) = 32 / 19``.
DENSITY_SCALE = wp.float32(32.0 / 19.0)
# ``samples_per_node``: the density at which a sample's depth estimate reaches its splat depth.
SAMPLES_PER_NODE = wp.float32(1.5)
# A normal weight below this is the upstream "no coarser contribution" test for the second splat.
SPLIT_EPSILON = wp.float32(1e-6)


# ---------------------------------------------------------------------------------------------
# Cells and quadratic B-splines
# ---------------------------------------------------------------------------------------------


@wp.func
def octree_cell_of(u: wp.float32, depth: wp.int32) -> wp.int32:
    # The depth ``depth`` cell holding unit coordinate ``u``. The upstream descent sends a point to
    # the upper child only when it lies past the centre, so a point on a face is the lower cell's.
    n = 1 << depth
    return wp.clamp(wp.int32(wp.ceil(u * wp.float32(n))) - 1, 0, n - 1)


@wp.func
def octree_cell_of_point(p: wp.vec3, depth: wp.int32) -> wp.vec3i:
    return wp.vec3i(
        octree_cell_of(p[0], depth), octree_cell_of(p[1], depth), octree_cell_of(p[2], depth)
    )


@wp.func
def cell_flat(n: wp.int32, c: wp.vec3i) -> wp.int32:
    # Row-major index of cell ``c`` in an ``n**3`` cell lattice.
    return (c[0] * n + c[1]) * n + c[2]


@wp.func
def lattice_index(d: wp.int32, node: wp.vec3i) -> wp.int32:
    return poisson_grid_index(node[0], node[1], node[2], (1 << d) + 1)


@wp.func
def quadratic_bspline_values(t: wp.float32) -> wp.vec3:
    # The three cell-centred quadratic B-splines nonzero in a cell, at local coordinate ``t`` in
    # ``[0, 1]``: those of the cells below, at and above it.
    return wp.vec3(0.5 * (1.0 - t) * (1.0 - t), 0.5 * (1.0 + 2.0 * t - 2.0 * t * t), 0.5 * t * t)


@wp.func
def splat_target(d: wp.int32, o: wp.vec3i) -> wp.vec2i:
    # ``(depth, cell)`` holding depth ``d``'s window slot ``o`` while the splats run, or depth -1
    # when no node holds it: the slot itself when inside the cube, nothing past a low face, and the
    # real depth ``d + 1`` cell with the same offset past a high face (the module note's quirk).
    n = 1 << d
    lowest = wp.min(wp.min(o[0], o[1]), o[2])
    highest = wp.max(wp.max(o[0], o[1]), o[2])
    target = wp.vec2i(-1, 0)
    if lowest >= 0:
        if highest < n:
            target = wp.vec2i(d, cell_flat(n, o))
        elif highest == n:
            target = wp.vec2i(d + 1, cell_flat(2 * n, o))
    return target


# The finest depth's per-cell tables are stored by 4**3-cell brick, and only for the bricks a
# sample or a finest-depth normal splat reaches (``octree_finest_bricks``); a brick table holds each
# brick's number among the stored ones, -1 for none. Every other depth stays dense.
BRICK_SHIFT = 2
BRICK_CELLS = 64


@wp.func
def brick_flat(depth: wp.int32, c: wp.vec3i) -> wp.int32:
    # The brick table's entry for finest cell ``c``.
    return cell_flat(
        1 << wp.max(depth - BRICK_SHIFT, 0),
        wp.vec3i(c[0] >> BRICK_SHIFT, c[1] >> BRICK_SHIFT, c[2] >> BRICK_SHIFT),
    )


@wp.func
def brick_entry(bricks: wp.array[wp.int32], depth: wp.int32, c: wp.vec3i) -> wp.int32:
    # Finest cell ``c``'s entry in a brick-packed table, -1 when its brick is not stored.
    brick = bricks[brick_flat(depth, c)]
    if brick < 0:
        return -1
    mask = (1 << BRICK_SHIFT) - 1
    return brick * BRICK_CELLS + cell_flat(
        1 << BRICK_SHIFT, wp.vec3i(c[0] & mask, c[1] & mask, c[2] & mask)
    )


@wp.func
def sample_slot(
    slots: wp.array[wp.int32], bricks: wp.array[wp.int32], depth: wp.int32, c: wp.vec3i
) -> wp.int32:
    # The sample merged into finest cell ``c``, -1 for none.
    entry = brick_entry(bricks, depth, c)
    if entry < 0:
        return -1
    return slots[entry]


@wp.func
def normal_slot(
    cell_offsets: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    depth: wp.int32,
    d: wp.int32,
    c: wp.vec3i,
) -> wp.int32:
    # Depth ``d`` cell ``c``'s normal coefficient in the field: dense at the coarser depths, by
    # brick after them at the finest (-1 when the brick is not stored, a zero coefficient).
    if d < depth:
        return cell_offsets[d] + cell_flat(1 << d, c)
    entry = brick_entry(bricks, depth, c)
    if entry < 0:
        return -1
    return cell_offsets[depth] + entry


@wp.func
def normal_at(
    normals: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    depth: wp.int32,
    d: wp.int32,
    c: wp.vec3i,
) -> wp.vec3:
    slot = normal_slot(cell_offsets, bricks, depth, d, c)
    if slot < 0:
        return wp.vec3(0.0, 0.0, 0.0)
    return normals[slot]


@wp.func
def mark_finest_targets(
    depth: wp.int32, d: wp.int32, node: wp.vec3i, out_marks: wp.array[wp.int32]
):
    # Mark the bricks of the finest cells among depth ``d``'s 27 window slots around ``node``
    # (``splat_target``: the slots themselves at the finest depth, or slots past a high face of
    # depth ``depth - 1``).
    for a in range(3):
        for b in range(3):
            for c in range(3):
                o = node + wp.vec3i(a - 1, b - 1, c - 1)
                if splat_target(d, o)[0] == depth:
                    out_marks[brick_flat(depth, o)] = 1


@wp.kernel
def octree_finest_bricks(
    sample_keys: wp.array[wp.int32], depth: wp.int32, out_marks: wp.array[wp.int32]
) -> None:
    # Mark the bricks a sample's finest cell or a finest-depth normal splat can reach: a splat
    # lands there only at the finest depth itself (around the sample's cell) or aliased from the
    # depth above it (around its parent), whatever the sample's depth estimate.
    s = wp.int32(wp.tid())
    n = 1 << depth
    k = sample_keys[s]
    cell = wp.vec3i(k // (n * n), (k // n) % n, k % n)
    out_marks[brick_flat(depth, cell)] = 1
    mark_finest_targets(depth, depth, cell, out_marks)
    if depth > 0:
        mark_finest_targets(
            depth, depth - 1, wp.vec3i(cell[0] >> 1, cell[1] >> 1, cell[2] >> 1), out_marks
        )


@wp.func
def brick_number(mark: wp.int32, rank: wp.int32) -> wp.int32:
    # A brick's number from its mark and the marks' inclusive scan, -1 when unmarked.
    return wp.where(mark != 0, rank - 1, -1)


@wp.func
def samples_per_node(
    density: wp.array[wp.float32], cell_offsets: wp.array[wp.int32], d: wp.int32, p: wp.vec3
) -> wp.float32:
    # The depth ``d`` density at ``p``: the 27 B-splines around ``p``'s cell, read through
    # ``splat_target`` as the splat wrote them.
    cell = octree_cell_of_point(p, d)
    scale = wp.float32(1 << d)
    vx = quadratic_bspline_values(p[0] * scale - wp.float32(cell[0]))
    vy = quadratic_bspline_values(p[1] * scale - wp.float32(cell[1]))
    vz = quadratic_bspline_values(p[2] * scale - wp.float32(cell[2]))
    total = wp.float32(0.0)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                target = splat_target(d, cell + wp.vec3i(a - 1, b - 1, c - 1))
                if target[0] >= 0:
                    total += vx[a] * vy[b] * vz[c] * density[cell_offsets[target[0]] + target[1]]
    return total


@wp.func
def octree_sample_depth(
    density: wp.array[wp.float32],
    cell_offsets: wp.array[wp.int32],
    kernel_depth: wp.int32,
    p: wp.vec3,
) -> wp.float32:
    # The upstream sample depth at ``p``: ``kernel_depth + log4(rho)`` where the density reaches 1
    # there, else the log-linear interpolation to 1 between the first depth (climbing) where it
    # does and the one below it, or depth 0's extrapolation.
    rho = samples_per_node(density, cell_offsets, kernel_depth, p)
    if rho >= 1.0:
        return wp.float32(kernel_depth) + wp.log(rho) / wp.log(wp.float32(4.0))
    level = kernel_depth
    previous = rho
    current = rho
    while current < 1.0 and level > 0:
        level -= 1
        previous = current
        current = samples_per_node(density, cell_offsets, level, p)
    return wp.float32(level) + wp.log(current) / wp.log(current / previous)


# ---------------------------------------------------------------------------------------------
# Samples and density
# ---------------------------------------------------------------------------------------------


@wp.kernel
def octree_point_keys(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    inverse: wp.float32,
    translation: wp.vec3,
    depth: wp.int32,
    out_keys: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
    out_normals: wp.array[wp.vec3],
) -> None:
    # Each point's finest cell (the sentinel ``8**depth`` for a zero or non-finite normal, which
    # the upstream reader drops), its unit-cube position and its unit normal. The position is the
    # upstream transform's arithmetic: a product, then the translation added.
    i = wp.int32(wp.tid())
    u = points[i] * inverse + translation
    n = normals[i]
    length = wp.length(n)
    key = wp.int32(1 << (3 * depth))
    unit = wp.vec3(0.0, 0.0, 0.0)
    if length > 0.0 and wp.isfinite(length):
        key = cell_flat(1 << depth, octree_cell_of_point(u, depth))
        unit = n / length
    out_keys[i] = key
    out_positions[i] = u
    out_normals[i] = unit


@wp.kernel
def octree_sample_slots(
    keys: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    depth: wp.int32,
    out_slots: wp.array[wp.int32],
) -> None:
    # The finest lattice's cell-to-sample map, by brick (cells without a sample keep -1).
    s = wp.int32(wp.tid())
    n = 1 << depth
    k = keys[s]
    out_slots[brick_entry(bricks, depth, wp.vec3i(k // (n * n), (k // n) % n, k % n))] = s


@wp.kernel
def octree_child_aggregates(
    d: wp.int32,
    depth: wp.int32,
    slots: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    sample_counts: wp.array[wp.int32],
    sample_positions: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    counts: wp.array[wp.float32],
    position_sums: wp.array[wp.vec3],
) -> None:
    # Depth ``d`` cell's point count and summed position, gathered from its eight children in a
    # fixed order: the samples at the finest depth, the depth ``d + 1`` aggregates above it.
    cell = wp.int32(wp.tid())
    n = 1 << d
    c = wp.vec3i(cell // (n * n), (cell // n) % n, cell % n)
    total = wp.float32(0.0)
    position = wp.vec3(0.0, 0.0, 0.0)
    for child in range(8):
        cc = 2 * c + wp.vec3i(child >> 2, (child >> 1) & 1, child & 1)
        index = cell_flat(2 * n, cc)
        if d + 1 == depth:
            s = sample_slot(slots, bricks, depth, cc)
            if s >= 0:
                total += wp.float32(sample_counts[s])
                position += sample_positions[s]
        else:
            total += counts[cell_offsets[d + 1] + index]
            position += position_sums[cell_offsets[d + 1] + index]
    counts[cell_offsets[d] + cell] = total
    position_sums[cell_offsets[d] + cell] = position


@wp.func
def level_of(
    offsets: wp.array[wp.int32], first: wp.int32, last: wp.int32, flat: wp.int32
) -> wp.int32:
    # The depth in ``[first, last]`` whose packed range holds ``flat``.
    level = first
    for d in range(first, last + 1):
        if flat >= offsets[d]:
            level = d
    return level


@wp.kernel
def octree_density_splat(
    counts: wp.array[wp.float32],
    position_sums: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    kernel_depth: wp.int32,
    sums: GroupSumPass,
    out_density: wp.array[wp.float32],
) -> None:
    # One child cell of depths ``1 ..= kernel_depth + 1`` (threads index the packed range from
    # depth 1): its ``count / samples_per_node`` splatted at its centroid into the 27 B-splines
    # around its parent cell.
    flat = wp.int32(wp.tid()) + cell_offsets[1]
    child_depth = level_of(cell_offsets, 1, kernel_depth + 1, flat)
    count = counts[flat]
    if count <= 0.0:
        return
    d = child_depth - 1
    n = 1 << child_depth
    local = flat - cell_offsets[child_depth]
    child = wp.vec3i(local // (n * n), (local // n) % n, local % n)
    parent = wp.vec3i(child[0] >> 1, child[1] >> 1, child[2] >> 1)
    p = position_sums[flat] / count
    scale = wp.float32(1 << d)
    vx = quadratic_bspline_values(p[0] * scale - wp.float32(parent[0]))
    vy = quadratic_bspline_values(p[1] * scale - wp.float32(parent[1]))
    vz = quadratic_bspline_values(p[2] * scale - wp.float32(parent[2]))
    mass = DENSITY_SCALE * count / SAMPLES_PER_NODE
    for a in range(3):
        for b in range(3):
            for c in range(3):
                target = splat_target(d, parent + wp.vec3i(a - 1, b - 1, c - 1))
                if target[0] >= 0:
                    scatter_add(
                        out_density,
                        sums,
                        cell_offsets[target[0]] + target[1],
                        mass * vx[a] * vy[b] * vz[c],
                    )


@wp.kernel
def octree_sample_weights(
    sample_counts: wp.array[wp.int32],
    sample_positions: wp.array[wp.vec3],
    sample_normals: wp.array[wp.vec3],
    density: wp.array[wp.float32],
    cell_offsets: wp.array[wp.int32],
    kernel_depth: wp.int32,
    out_depths: wp.array[wp.float32],
    out_area: wp.array[wp.float32],
    out_count: wp.array[wp.float32],
) -> None:
    # Each sample's depth estimate, and its terms of the mean area weight ``pointWeightSum``
    # (``count * 4**-depth`` and ``count``, zero for a sample whose normals cancel).
    s = wp.int32(wp.tid())
    count = wp.float32(sample_counts[s])
    p = sample_positions[s] / count
    depth = octree_sample_depth(density, cell_offsets, kernel_depth, p)
    out_depths[s] = depth
    has_normal = wp.length(sample_normals[s]) > 0.0
    out_area[s] = wp.where(has_normal, count * wp.pow(wp.float32(4.0), -depth), wp.float32(0.0))
    out_count[s] = wp.where(has_normal, count, wp.float32(0.0))


@wp.func
def splat_normal_at(
    d: wp.int32,
    depth: wp.int32,
    cell: wp.vec3i,
    p: wp.vec3,
    amount: wp.vec3,
    cell_offsets: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    sums: GroupSumPass,
    out_normals: wp.array[wp.vec3],
):
    # ``amount`` splatted into the 27 depth ``d`` B-splines around the depth ``d`` ancestor of the
    # finest cell ``cell``. A slot aliased past the finest depth would need a node the solve does
    # not have; a cloud padded by ``scale >= 1`` never reaches one there. A slot holds the same
    # integer offset at whichever depth ``splat_target`` puts it, and every finest one lies in a
    # brick ``octree_finest_bricks`` marked.
    node = wp.vec3i(cell[0] >> (depth - d), cell[1] >> (depth - d), cell[2] >> (depth - d))
    scale = wp.float32(1 << d)
    vx = quadratic_bspline_values(p[0] * scale - wp.float32(node[0]))
    vy = quadratic_bspline_values(p[1] * scale - wp.float32(node[1]))
    vz = quadratic_bspline_values(p[2] * scale - wp.float32(node[2]))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                o = node + wp.vec3i(a - 1, b - 1, c - 1)
                target = splat_target(d, o)
                if target[0] >= 0 and target[0] <= depth:
                    scatter_add_vector(
                        out_normals,
                        sums,
                        normal_slot(cell_offsets, bricks, depth, target[0], o),
                        amount * (vx[a] * vy[b] * vz[c]),
                    )


@wp.kernel
def octree_normal_splat(
    sample_keys: wp.array[wp.int32],
    sample_counts: wp.array[wp.int32],
    sample_positions: wp.array[wp.vec3],
    sample_normals: wp.array[wp.vec3],
    depths: wp.array[wp.float32],
    cell_offsets: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    depth: wp.int32,
    sums: GroupSumPass,
    out_normals: wp.array[wp.vec3],
) -> None:
    # A sample's normal, ``count * 4**-depth`` times its unit mean, splatted at its depth estimate:
    # split between the two depths around it (all at the finest when it lies beyond), scaled by
    # ``8**d`` so the field integrates to the sample's area.
    s = wp.int32(wp.tid())
    normal_sum = sample_normals[s]
    length = wp.length(normal_sum)
    if length <= 0.0:
        return
    count = wp.float32(sample_counts[s])
    p = sample_positions[s] / count
    estimate = depths[s]
    weight = wp.pow(wp.float32(4.0), -estimate)
    clamped = wp.clamp(estimate, 0.0, wp.float32(depth))
    top = wp.int32(wp.ceil(clamped))
    fraction = 1.0 - (wp.float32(top) - clamped)
    if top <= 0:
        top = 0
        fraction = wp.float32(1.0)
    n = 1 << depth
    k = sample_keys[s]
    cell = wp.vec3i(k // (n * n), (k // n) % n, k % n)
    base = (count * weight / length) * normal_sum
    splat_normal_at(
        top,
        depth,
        cell,
        p,
        base * (wp.pow(wp.float32(8.0), wp.float32(top)) * fraction),
        cell_offsets,
        bricks,
        sums,
        out_normals,
    )
    if wp.abs(1.0 - fraction) > SPLIT_EPSILON:
        splat_normal_at(
            top - 1,
            depth,
            cell,
            p,
            base * (wp.pow(wp.float32(8.0), wp.float32(top - 1)) * (1.0 - fraction)),
            cell_offsets,
            bricks,
            sums,
            out_normals,
        )


# ---------------------------------------------------------------------------------------------
# Active nodes
# ---------------------------------------------------------------------------------------------


@wp.func
def padded_flat(n: wp.int32, c: wp.vec3i) -> wp.int32:
    # Index of cell ``c`` (each coordinate in ``[-1, n]``) in a depth's padded flag lattice.
    m = n + 2
    return ((c[0] + 1) * m + c[1] + 1) * m + c[2] + 1


@wp.func
def children_active(
    flags: wp.array[wp.int32], flag_offsets: wp.array[wp.int32], d: wp.int32, c: wp.vec3i
) -> wp.bool:
    # Whether depth ``d`` cell ``c``'s children exist and are active (``c`` may be a padding cell).
    return flags[flag_offsets[d] + padded_flat(1 << d, c)] != 0


@wp.func
def node_active(
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    full_depth: wp.int32,
    d: wp.int32,
    o: wp.vec3i,
) -> wp.bool:
    # Whether the depth ``d`` node (or cell) ``o`` is active: always up to ``full_depth``, else
    # when its parent's children are.
    if d <= full_depth:
        return True
    return children_active(flags, flag_offsets, d - 1, wp.vec3i(o[0] >> 1, o[1] >> 1, o[2] >> 1))


@wp.kernel
def octree_clip_flags(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    normals: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    flags: wp.array[wp.int32],
) -> None:
    # Depth ``d`` cell's "children active" flag before the dilation: always below ``full_depth``,
    # else whether its subtree below it holds a nonzero normal coefficient. Run finest first.
    flat = wp.int32(wp.tid())
    n = 1 << d
    m = n + 2
    c = wp.vec3i(flat // (m * m) - 1, (flat // m) % m - 1, flat % m - 1)
    active = wp.int32(0)
    if d < full_depth:
        active = 1
    elif c[0] >= 0 and c[1] >= 0 and c[2] >= 0 and c[0] < n and c[1] < n and c[2] < n:
        for child in range(8):
            cc = 2 * c + wp.vec3i(child >> 2, (child >> 1) & 1, child & 1)
            v = normal_at(normals, cell_offsets, bricks, depth, d + 1, cc)
            if v[0] != 0.0 or v[1] != 0.0 or v[2] != 0.0:
                active = 1
            if d + 1 < depth:
                if children_active(flags, flag_offsets, d + 1, cc):
                    active = 1
    flags[flag_offsets[d] + flat] = active


@wp.kernel
def octree_dilate_flags(
    d: wp.int32, flag_offsets: wp.array[wp.int32], flags: wp.array[wp.int32]
) -> None:
    # Depth ``d - 1`` cell's children become active when one of them lies within two of a depth
    # ``d`` node whose own children are active. Run finest first, after the clip.
    flat = wp.int32(wp.tid())
    n = 1 << (d - 1)
    m = n + 2
    p = wp.vec3i(flat // (m * m) - 1, (flat // m) % m - 1, flat % m - 1)
    if flags[flag_offsets[d - 1] + flat] != 0:
        return
    fine = 1 << d
    for a in range(6):
        for b in range(6):
            for c in range(6):
                q = 2 * p + wp.vec3i(a - 2, b - 2, c - 2)
                if q[0] >= -1 and q[1] >= -1 and q[2] >= -1:
                    if q[0] <= fine and q[1] <= fine and q[2] <= fine:
                        if children_active(flags, flag_offsets, d, q):
                            flags[flag_offsets[d - 1] + flat] = 1
                            return


# ---------------------------------------------------------------------------------------------
# Stored nodes
# ---------------------------------------------------------------------------------------------
#
# The right-hand sides and the fields are stored by ``4**3``-node brick, per depth, for the bricks
# a read can reach (``octree_node_bricks``): every node within three of an active node, the finest
# corners of every sample's cell (the iso-value's reads), and every coarser node a stored node's
# prolongation reads. Within three covers every nonzero own-or-restricted right-hand side (a data
# cell's divergence reaches two nodes past its active brood, a restricted one three) and every
# field read: the sweeps' stencils (one), the extraction's slopes (two, and their coarser nodes).
# ``node_bricks`` holds the depths' brick tables after ``depth + 2`` offsets into itself, each
# entry a brick's number among the stored ones (numbered depth by depth) or -1. A node outside
# them reads ``+0.0``, which is what its dense entry held in a right-hand side; no field read
# reaches one.


@wp.func
def node_bricks_per_axis(d: wp.int32) -> wp.int32:
    # Depth ``d``'s node bricks along an axis: its ``2**d + 1`` nodes in fours.
    return ((1 << d) >> BRICK_SHIFT) + 1


@wp.func
def node_brick_flat(node_bricks: wp.array[wp.int32], d: wp.int32, b: wp.vec3i) -> wp.int32:
    # Depth ``d``'s node brick ``b`` in ``node_bricks``.
    return node_bricks[d] + cell_flat(node_bricks_per_axis(d), b)


@wp.func
def node_entry(node_bricks: wp.array[wp.int32], d: wp.int32, node: wp.vec3i) -> wp.int32:
    # Depth ``d``'s lattice node ``node`` in the brick-packed node values, -1 when not stored.
    brick = node_bricks[
        node_brick_flat(
            node_bricks,
            d,
            wp.vec3i(node[0] >> BRICK_SHIFT, node[1] >> BRICK_SHIFT, node[2] >> BRICK_SHIFT),
        )
    ]
    if brick < 0:
        return -1
    mask = (1 << BRICK_SHIFT) - 1
    return brick * BRICK_CELLS + cell_flat(
        1 << BRICK_SHIFT, wp.vec3i(node[0] & mask, node[1] & mask, node[2] & mask)
    )


@wp.func
def node_value(
    values: wp.array[wp.float32], node_bricks: wp.array[wp.int32], d: wp.int32, node: wp.vec3i
) -> wp.float32:
    # Depth ``d``'s value at lattice node ``node``, ``+0.0`` when it is not stored.
    entry = node_entry(node_bricks, d, node)
    if entry < 0:
        return wp.float32(0.0)
    return values[entry]


@wp.func
def stored_node(
    node_bricks: wp.array[wp.int32],
    brick_slots: wp.array[wp.int32],
    depth: wp.int32,
    entry: wp.int32,
) -> wp.vec4i:
    # ``(d, i, j, k)`` of a brick-packed node entry, ``d = -1`` past its depth's lattice (the last
    # brick of an axis holds the wall node and three slots beyond it). ``brick_slots`` maps a
    # brick's number to its entry of ``node_bricks``.
    slot = brick_slots[entry // BRICK_CELLS]
    d = level_of(node_bricks, 0, depth, slot)
    nb = node_bricks_per_axis(d)
    local = slot - node_bricks[d]
    side = 1 << BRICK_SHIFT
    within = entry % BRICK_CELLS
    node = wp.vec3i(
        (local // (nb * nb)) * side + within // (side * side),
        ((local // nb) % nb) * side + (within // side) % side,
        (local % nb) * side + within % side,
    )
    n = 1 << d
    if node[0] > n or node[1] > n or node[2] > n:
        d = -1
    return wp.vec4i(d, node[0], node[1], node[2])


@wp.kernel
def octree_sample_node_bricks(
    sample_keys: wp.array[wp.int32], depth: wp.int32, out_marks: wp.array[wp.int32]
) -> None:
    # Mark the finest node bricks holding a sample cell's eight corners, which the iso-value
    # interpolates.
    s = wp.int32(wp.tid())
    n = 1 << depth
    k = sample_keys[s]
    cell = wp.vec3i(k // (n * n), (k // n) % n, k % n)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                corner = cell + wp.vec3i(a, b, c)
                out_marks[
                    node_brick_flat(
                        out_marks,
                        depth,
                        wp.vec3i(
                            corner[0] >> BRICK_SHIFT,
                            corner[1] >> BRICK_SHIFT,
                            corner[2] >> BRICK_SHIFT,
                        ),
                    )
                ] = 1


@wp.kernel
def octree_node_bricks(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    marks: wp.array[wp.int32],
) -> None:
    # Whether depth ``d``'s node brick is stored (the section note): already marked by a sample,
    # within three of an active node (a padded depth ``d - 1`` cell with active children has nodes
    # ``2 p`` and ``2 p + 1``), or holding a coarser node of a stored depth ``d + 1`` brick (those
    # of fine brick ``f`` span ``2 f .. 2 f + 2``). Run finest first.
    flat = wp.int32(wp.tid())
    nb = node_bricks_per_axis(d)
    b = wp.vec3i(flat // (nb * nb), (flat // nb) % nb, flat % nb)
    slot = marks[d] + flat
    keep = marks[slot] != 0 or d <= full_depth
    if not keep:
        half = 1 << (d - 1)
        for x in range(6):
            px = 2 * b[0] - 2 + x
            if px >= -1 and px <= half and not keep:
                for y in range(6):
                    py = 2 * b[1] - 2 + y
                    if py >= -1 and py <= half and not keep:
                        for z in range(6):
                            pz = 2 * b[2] - 2 + z
                            if pz >= -1 and pz <= half and not keep:
                                if children_active(
                                    flags, flag_offsets, d - 1, wp.vec3i(px, py, pz)
                                ):
                                    keep = True
    if not keep and d < depth:
        nf = node_bricks_per_axis(d + 1)
        for x in range(3):
            fx = 2 * b[0] - 1 + x
            if fx >= 0 and fx < nf and not keep:
                for y in range(3):
                    fy = 2 * b[1] - 1 + y
                    if fy >= 0 and fy < nf and not keep:
                        for z in range(3):
                            fz = 2 * b[2] - 1 + z
                            if fz >= 0 and fz < nf:
                                if marks[node_brick_flat(marks, d + 1, wp.vec3i(fx, fy, fz))] != 0:
                                    keep = True
    marks[slot] = wp.where(keep, 1, 0)


@wp.kernel
def octree_node_brick_ends(
    node_bricks: wp.array[wp.int32], ranks: wp.array[wp.int32], out_ends: wp.array[wp.int32]
) -> None:
    # The stored bricks up to each depth's table's end: the marks' inclusive scan (``ranks``,
    # over the tables after the ``node_bricks`` header) at its last entry.
    d = wp.int32(wp.tid())
    head = out_ends.shape[0] + 1
    out_ends[d] = ranks[node_bricks[d + 1] - head - 1]


@wp.kernel
def octree_node_brick_slots(
    node_bricks: wp.array[wp.int32], first: wp.int32, out_slots: wp.array[wp.int32]
) -> None:
    # Each stored brick's entry of ``node_bricks``, by its number (``stored_node``'s inverse map).
    slot = wp.int32(wp.tid()) + first
    brick = node_bricks[slot]
    if brick >= 0:
        out_slots[brick] = slot


# ---------------------------------------------------------------------------------------------
# Right-hand side
# ---------------------------------------------------------------------------------------------


@wp.func
def quadratic_bspline(t: wp.float64) -> wp.float64:
    a = wp.abs(t)
    value = wp.float64(0.0)
    if a <= wp.float64(0.5):
        value = wp.float64(0.75) - a * a
    elif a <= wp.float64(1.5):
        value = wp.float64(0.5) * (a - wp.float64(1.5)) * (a - wp.float64(1.5))
    return value


@wp.func
def normal_bspline(d: wp.int32, k: wp.int32, x: wp.float64) -> wp.float64:
    # Depth ``d``'s cell-centred B-spline ``k`` at ``x``, reflected oddly at the walls (the first
    # and last lose their mirrored neighbour), so every one vanishes on the cube's faces.
    n = 1 << d
    t = x * wp.float64(n)
    value = quadratic_bspline(t - (wp.float64(k) + wp.float64(0.5)))
    if k == 0:
        value -= quadratic_bspline(t + wp.float64(0.5))
    if k == n - 1:
        value -= quadratic_bspline(t - (wp.float64(n) + wp.float64(0.5)))
    return value


@wp.func
def hat_bspline_integrals(dt: wp.int32, i: wp.int32, dd: wp.int32, k: wp.int32) -> wp.vec2d:
    # ``(int hat_i B_k, int hat_i' B_k)`` over ``[0, 1]`` for depth ``dt``'s hat ``i`` and depth
    # ``dd <= dt``'s normal B-spline ``k``: three Gauss points on each of the hat's two cells, which
    # is exact since the B-spline is one quadratic piece on each.
    n = 1 << dt
    h = wp.float64(1.0) / wp.float64(n)
    gx = wp.vec3d(wp.float64(-0.7745966692414834), wp.float64(0.0), wp.float64(0.7745966692414834))
    gw = wp.vec3d(wp.float64(5.0 / 9.0), wp.float64(8.0 / 9.0), wp.float64(5.0 / 9.0))
    value = wp.float64(0.0)
    slope = wp.float64(0.0)
    for side in range(2):
        cell = i - 1 + side
        if cell >= 0 and cell < n:
            dhat = wp.where(side == 0, wp.float64(n), -wp.float64(n))
            for q in range(3):
                x = (wp.float64(cell) + wp.float64(0.5) * (gx[q] + wp.float64(1.0))) * h
                hat = wp.float64(1.0) - wp.abs(x * wp.float64(n) - wp.float64(i))
                b = normal_bspline(dd, k, x) * gw[q] * wp.float64(0.5) * h
                value += hat * b
                slope += dhat * b
    return wp.vec2d(value, slope)


@wp.func
def axis_integral_slot(d: wp.int32, shift: wp.int32, i: wp.int32, q: wp.int32) -> wp.int32:
    # Entry of ``octree_axis_integrals``' table for depth ``d``'s hat ``i`` and the ``q``-th of the
    # four depth ``d - shift`` B-splines overlapping it: per depth, both shifts, ``res`` hats each.
    return 8 * ((1 << d) - 1 + d) + (shift * ((1 << d) + 1) + i) * 4 + q


@wp.kernel
def octree_axis_integrals(depth: wp.int32, out_integrals: wp.array[wp.vec2d]) -> None:
    # Every ``hat_bspline_integrals`` a divergence reads, once: they depend only on the depth, the
    # hat's coordinate along one axis and the B-spline's, the last fixed by the first and ``q``.
    flat = wp.int32(wp.tid())
    d = wp.int32(0)
    for e in range(depth + 1):
        if flat >= 8 * ((1 << e) - 1 + e):
            d = e
    res = (1 << d) + 1
    local = flat - 8 * ((1 << d) - 1 + d)
    shift = local // (4 * res)
    i = (local % (4 * res)) // 4
    q = local % 4
    if d - shift < 0:
        out_integrals[flat] = wp.vec2d()
        return
    out_integrals[flat] = hat_bspline_integrals(d, i, d - shift, (i >> shift) - 2 + q)


@wp.func
def window_has_brick(
    bricks: wp.array[wp.int32], depth: wp.int32, first: wp.vec3i, m: wp.int32
) -> wp.bool:
    # Whether a stored brick meets the finest cells ``first + [0, 4)**3`` inside the cube.
    lo = wp.vec3i(
        wp.max(first[0], 0) >> BRICK_SHIFT,
        wp.max(first[1], 0) >> BRICK_SHIFT,
        wp.max(first[2], 0) >> BRICK_SHIFT,
    )
    hi = wp.vec3i(
        wp.min(first[0] + 3, m - 1) >> BRICK_SHIFT,
        wp.min(first[1] + 3, m - 1) >> BRICK_SHIFT,
        wp.min(first[2] + 3, m - 1) >> BRICK_SHIFT,
    )
    nb = 1 << wp.max(depth - BRICK_SHIFT, 0)
    for a in range(lo[0], hi[0] + 1):
        for b in range(lo[1], hi[1] + 1):
            for c in range(lo[2], hi[2] + 1):
                if bricks[cell_flat(nb, wp.vec3i(a, b, c))] >= 0:
                    return True
    return False


@wp.func
def window_coefficient(
    data: wp.array[wp.vec3],
    base: wp.int32,
    finest: wp.bool,
    bricks: wp.array[wp.int32],
    depth: wp.int32,
    m: wp.int32,
    c: wp.vec3i,
) -> wp.vec3:
    # ``normal_at`` with the depth's offset (``base``) and layout read once per window: the
    # ``m**3`` dense cells there, or at the finest depth the stored bricks.
    if not finest:
        return data[base + cell_flat(m, c)]
    entry = brick_entry(bricks, depth, c)
    if entry < 0:
        return wp.vec3(0.0, 0.0, 0.0)
    return data[base + entry]


@wp.func
def divergence_term(
    d: wp.int32,
    data_depth: wp.int32,
    data: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    depth: wp.int32,
    node: wp.vec3i,
    integrals: wp.array[wp.vec2d],
) -> wp.float32:
    # Depth ``d`` node's ``int grad(hat) . V`` for the depth ``data_depth`` (``d`` or ``d - 1``)
    # normal coefficients ``data``, laid out as the normal field (``normal_at``): four B-splines
    # per axis overlap a hat at either depth.
    m = 1 << data_depth
    shift = d - data_depth
    first = wp.vec3i((node[0] >> shift) - 2, (node[1] >> shift) - 2, (node[2] >> shift) - 2)
    # The data is a band around the surface, so most of a window's cells hold zero, and the whole
    # window does for most nodes. A zero cell's term is a signed zero and the running sum is never
    # ``-0.0``, so skipping those cells, and the axis integrals only they would read, leaves the
    # ``float64`` sum's bits as they were; an empty window returns the ``+0.0`` it would sum to.
    # At the finest depth a window outside every stored brick is such a window: its at most eight
    # bricks are tested first. This pass records which axis offsets some nonzero cell uses.
    if data_depth == depth and not window_has_brick(bricks, depth, first, m):
        return wp.float32(0.0)
    base = cell_offsets[data_depth]
    finest = data_depth == depth
    used = wp.vec3i(0, 0, 0)
    for a in range(4):
        ka = first[0] + a
        if ka >= 0 and ka < m:
            for b in range(4):
                kb = first[1] + b
                if kb >= 0 and kb < m:
                    for c in range(4):
                        kc = first[2] + c
                        if kc >= 0 and kc < m:
                            v = window_coefficient(
                                data, base, finest, bricks, depth, m, wp.vec3i(ka, kb, kc)
                            )
                            if v[0] != 0.0 or v[1] != 0.0 or v[2] != 0.0:
                                used = wp.vec3i(
                                    used[0] | (1 << a), used[1] | (1 << b), used[2] | (1 << c)
                                )
    if used[0] == 0:
        return wp.float32(0.0)
    value_x = wp.vec4d()
    slope_x = wp.vec4d()
    value_y = wp.vec4d()
    slope_y = wp.vec4d()
    value_z = wp.vec4d()
    slope_z = wp.vec4d()
    for q in range(4):
        if (used[0] >> q) & 1 != 0:
            ix = integrals[axis_integral_slot(d, shift, node[0], q)]
            value_x[q] = ix[0]
            slope_x[q] = ix[1]
        if (used[1] >> q) & 1 != 0:
            iy = integrals[axis_integral_slot(d, shift, node[1], q)]
            value_y[q] = iy[0]
            slope_y[q] = iy[1]
        if (used[2] >> q) & 1 != 0:
            iz = integrals[axis_integral_slot(d, shift, node[2], q)]
            value_z[q] = iz[0]
            slope_z[q] = iz[1]
    total = wp.float64(0.0)
    for a in range(4):
        ka = first[0] + a
        if ka >= 0 and ka < m:
            for b in range(4):
                kb = first[1] + b
                if kb >= 0 and kb < m:
                    for c in range(4):
                        kc = first[2] + c
                        if kc >= 0 and kc < m:
                            v = window_coefficient(
                                data, base, finest, bricks, depth, m, wp.vec3i(ka, kb, kc)
                            )
                            if v[0] != 0.0 or v[1] != 0.0 or v[2] != 0.0:
                                total += (
                                    wp.float64(v[0]) * slope_x[a] * value_y[b] * value_z[c]
                                    + wp.float64(v[1]) * value_x[a] * slope_y[b] * value_z[c]
                                    + wp.float64(v[2]) * value_x[a] * value_y[b] * slope_z[c]
                                )
    return wp.float32(total)


@wp.kernel
def octree_own_constraints(
    depth: wp.int32,
    normals: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    node_bricks: wp.array[wp.int32],
    brick_slots: wp.array[wp.int32],
    integrals: wp.array[wp.vec2d],
    out_b: wp.array[wp.float32],
) -> None:
    # Every stored node's ``-int grad(hat) . V`` for its own depth's normal coefficients, all
    # depths in one launch (``octree_coarse_screening_constraints`` adds the coarser data's). A
    # slot past its lattice takes zero.
    e = wp.int32(wp.tid())
    at = stored_node(node_bricks, brick_slots, depth, e)
    d = at[0]
    value = wp.float32(0.0)
    if d >= 0:
        node = wp.vec3i(at[1], at[2], at[3])
        value = wp.float32(0.0) - divergence_term(
            d, d, normals, cell_offsets, bricks, depth, node, integrals
        )
    out_b[e] = value


@wp.kernel
def octree_restrict_add(
    depth: wp.int32,
    first: wp.int32,
    node_bricks: wp.array[wp.int32],
    brick_slots: wp.array[wp.int32],
    b: wp.array[wp.float32],
) -> None:
    # A coarse hat is a sum of fine ones, so finer constraints restrict to it exactly: one depth's
    # stored nodes (from entry ``first``) gather the next depth's, in ``poisson_restrict_node``'s
    # order and weights. A fine node not stored holds a zero right-hand side (the section note).
    e = wp.int32(wp.tid()) + first
    at = stored_node(node_bricks, brick_slots, depth, e)
    d = at[0]
    if d < 0:
        return
    res_f = (1 << (d + 1)) + 1
    acc = wp.float32(0.0)
    for di in range(-1, 2):
        fi = 2 * at[1] + di
        if fi >= 0 and fi < res_f:
            for dj in range(-1, 2):
                fj = 2 * at[2] + dj
                if fj >= 0 and fj < res_f:
                    for dk in range(-1, 2):
                        fk = 2 * at[3] + dk
                        if fk >= 0 and fk < res_f:
                            acc += poisson_restrict_weight(di, dj, dk) * node_value(
                                b, node_bricks, d + 1, wp.vec3i(fi, fj, fk)
                            )
    b[e] = b[e] + acc


@wp.kernel
def octree_upsample_coarse_data(
    d: wp.int32,
    normals: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    accumulated: wp.array[wp.vec3],
) -> None:
    # Depth ``d``'s own normal coefficients plus every coarser depth's, carried up by the hat
    # prolongation's weights (1 at ``2p``, 1/2 at ``2p +- 1``) applied to B-spline coefficients:
    # the upstream solver's shortcut for coarser data, reproduced.
    flat = wp.int32(wp.tid())
    n = 1 << d
    c = wp.vec3i(flat // (n * n), (flat // n) % n, flat % n)
    total = normals[cell_offsets[d] + flat]
    half = n >> 1
    for a in range(2):
        pa = (c[0] + a) >> 1
        wa = wp.where(c[0] % 2 == 0, wp.float32(1.0 - wp.float32(a)), wp.float32(0.5))
        if pa < half and wa != 0.0:
            for b in range(2):
                pb = (c[1] + b) >> 1
                wb = wp.where(c[1] % 2 == 0, wp.float32(1.0 - wp.float32(b)), wp.float32(0.5))
                if pb < half and wb != 0.0:
                    for e in range(2):
                        pe = (c[2] + e) >> 1
                        we = wp.where(
                            c[2] % 2 == 0, wp.float32(1.0 - wp.float32(e)), wp.float32(0.5)
                        )
                        if pe < half and we != 0.0:
                            total += (wa * wb * we) * accumulated[
                                cell_offsets[d - 1] + cell_flat(half, wp.vec3i(pa, pb, pe))
                            ]
    accumulated[cell_offsets[d] + flat] = total


# ---------------------------------------------------------------------------------------------
# Screening and the operator
# ---------------------------------------------------------------------------------------------


@wp.func
def cell_screen(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    c: wp.vec3i,
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    counts: wp.array[wp.float32],
    position_sums: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    sample_counts: wp.array[wp.int32],
    sample_positions: wp.array[wp.vec3],
) -> wp.vec4:
    # Depth ``d`` cell ``c``'s screening sample: ``(centroid, point count)``, a zero count when the
    # cell is outside the lattice, inactive or empty.
    n = 1 << d
    result = wp.vec4(0.0, 0.0, 0.0, 0.0)
    if c[0] < 0 or c[1] < 0 or c[2] < 0 or c[0] >= n or c[1] >= n or c[2] >= n:
        return result
    if not node_active(flags, flag_offsets, full_depth, d, c):
        return result
    count = wp.float32(0.0)
    position = wp.vec3(0.0, 0.0, 0.0)
    if d == depth:
        s = sample_slot(slots, bricks, depth, c)
        if s >= 0:
            count = wp.float32(sample_counts[s])
            position = sample_positions[s]
    else:
        count = counts[cell_offsets[d] + cell_flat(n, c)]
        position = position_sums[cell_offsets[d] + cell_flat(n, c)]
    if count > 0.0:
        p = position / count
        result = wp.vec4(p[0], p[1], p[2], count)
    return result


@wp.func
def hat_value(d: wp.int32, node: wp.vec3i, p: wp.vec3) -> wp.float32:
    scale = wp.float32(1 << d)
    return (
        (1.0 - wp.abs(p[0] * scale - wp.float32(node[0])))
        * (1.0 - wp.abs(p[1] * scale - wp.float32(node[1])))
        * (1.0 - wp.abs(p[2] * scale - wp.float32(node[2])))
    )


@wp.func
def hat_mass_stiffness(i: wp.int32, offset: wp.int32, res: wp.int32, h: wp.float32) -> wp.vec2:
    # 1-D ``(int hat_i hat_{i+offset}, int hat_i' hat_{i+offset}')`` on ``[0, 1]``: a wall node's
    # hat is cut in half.
    wall = i == 0 or i == res - 1
    mass = wp.float32(0.0)
    stiffness = wp.float32(0.0)
    if offset == 0:
        mass = wp.where(wall, h / 3.0, 2.0 * h / 3.0)
        stiffness = wp.where(wall, 1.0 / h, 2.0 / h)
    else:
        mass = h / 6.0
        stiffness = -1.0 / h
    return wp.vec2(mass, stiffness)


# A screened cell's record in ``octree_screen_records``' table: its weight
# ``alpha * 2**d * count`` and its eight corners' hat values at its centroid, corner ``q`` at offset
# ``(q >> 2, (q >> 1) & 1, q & 1)``. The Gauss-Seidel rows read these instead of forming them from
# the cell's samples on every update.
SCREEN_RECORD = 9
_Corners8i = wp.types.vector(length=8, dtype=wp.int32)
_Corners8f = wp.types.vector(length=8, dtype=wp.float32)
_Window27f = wp.types.vector(length=27, dtype=wp.float32)


@wp.func
def screen_record(
    d: wp.int32,
    depth: wp.int32,
    c: wp.vec3i,
    cell_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    screen_index: wp.array[wp.int32],
) -> wp.int32:
    # Depth ``d`` cell ``c``'s record in the screening table, -1 for none: the finest depth's are
    # its samples' (``slots``), a coarser depth's are indexed per cell (``screen_index``).
    n = 1 << d
    if c[0] < 0 or c[1] < 0 or c[2] < 0 or c[0] >= n or c[1] >= n or c[2] >= n:
        return -1
    if d == depth:
        return sample_slot(slots, bricks, depth, c)
    return screen_index[cell_offsets[d] + cell_flat(n, c)]


@wp.kernel
def octree_screen_records(
    depth: wp.int32,
    full_depth: wp.int32,
    alpha: wp.float32,
    n_samples: wp.int32,
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    counts: wp.array[wp.float32],
    position_sums: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    sample_counts: wp.array[wp.int32],
    sample_positions: wp.array[wp.vec3],
    out_index: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
    out_records: wp.array[wp.float32],
) -> None:
    # Every depth's screened cells' records, one thread a cell of every depth's lattice. A finest
    # cell's record sits at its sample's slot (weight 0 when the cell is inactive, which
    # ``cell_screen`` reports as a zero count); a coarser cell with samples takes the next record
    # after the samples', and ``out_index`` points its cell at it (-1 for none). The arrival order
    # only numbers the records, so it changes no value a row reads.
    flat = wp.int32(wp.tid())
    d = level_of(cell_offsets, 0, depth, flat)
    n = 1 << d
    local = flat - cell_offsets[d]
    c = wp.vec3i(local // (n * n), (local // n) % n, local % n)
    record = wp.int32(-1)
    if d == depth:
        record = sample_slot(slots, bricks, depth, c)
    sample = cell_screen(
        d,
        depth,
        full_depth,
        c,
        flags,
        flag_offsets,
        counts,
        position_sums,
        cell_offsets,
        slots,
        bricks,
        sample_counts,
        sample_positions,
    )
    if d < depth:
        if sample[3] > 0.0:
            record = n_samples + wp.atomic_add(out_count, 0, 1)
        out_index[flat] = record
    if record < 0:
        return
    base = record * SCREEN_RECORD
    scale = alpha * wp.float32(1 << d)
    out_records[base] = scale * sample[3]
    p = wp.vec3(sample[0], sample[1], sample[2])
    for q in range(8):
        corner = c + wp.vec3i(q >> 2, (q >> 1) & 1, q & 1)
        out_records[base + 1 + q] = hat_value(d, corner, p)


@wp.func
def octree_row(
    d: wp.int32,
    depth: wp.int32,
    node: wp.vec3i,
    field: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    screen_index: wp.array[wp.int32],
    screen: wp.array[wp.float32],
) -> wp.vec2:
    # ``(row . field, diagonal)`` of depth ``d``'s operator at ``node``: the hat stiffness (27
    # neighbours) plus the screening of the eight cells around the node, read off their
    # ``octree_screen_records`` (the node is corner ``corner`` of the cell at ``node - corner``).
    # Every field value either term reads is one of the 27 around the node: they are looked up
    # once (``window``, offset ``(a, b, c)`` at ``9 a + 3 b + c``), not once per read.
    res = (1 << d) + 1
    h = 1.0 / wp.float32(1 << d)
    window = _Window27f()
    for a in range(3):
        ia = node[0] + a - 1
        if ia >= 0 and ia < res:
            for b in range(3):
                ib = node[1] + b - 1
                if ib >= 0 and ib < res:
                    for c in range(3):
                        ic = node[2] + c - 1
                        if ic >= 0 and ic < res:
                            window[9 * a + 3 * b + c] = node_value(
                                field, node_bricks, d, wp.vec3i(ia, ib, ic)
                            )
    total = wp.float32(0.0)
    diagonal = wp.float32(0.0)
    for a in range(3):
        ia = node[0] + a - 1
        if ia >= 0 and ia < res:
            mx = hat_mass_stiffness(node[0], a - 1, res, h)
            for b in range(3):
                ib = node[1] + b - 1
                if ib >= 0 and ib < res:
                    my = hat_mass_stiffness(node[1], b - 1, res, h)
                    for c in range(3):
                        ic = node[2] + c - 1
                        if ic >= 0 and ic < res:
                            mz = hat_mass_stiffness(node[2], c - 1, res, h)
                            entry = (
                                mx[1] * my[0] * mz[0]
                                + mx[0] * my[1] * mz[0]
                                + mx[0] * my[0] * mz[1]
                            )
                            total += entry * window[9 * a + 3 * b + c]
                            if a == 1 and b == 1 and c == 1:
                                diagonal = entry
    # The eight cells' records and weights are each loaded in one batch before any is used, so
    # the loads of one cell do not wait on another's.
    records = _Corners8i()
    for corner in range(8):
        cell = node - wp.vec3i(corner >> 2, (corner >> 1) & 1, corner & 1)
        records[corner] = screen_record(d, depth, cell, cell_offsets, slots, bricks, screen_index)
    weights = _Corners8f()
    for corner in range(8):
        weights[corner] = wp.where(
            records[corner] >= 0, screen[wp.max(records[corner], 0) * SCREEN_RECORD], 0.0
        )
    for corner in range(8):
        weight = weights[corner]
        if weight > 0.0:
            base = records[corner] * SCREEN_RECORD
            phi = screen[base + 1 + corner]
            interpolated = wp.float32(0.0)
            for q in range(8):
                # Corner ``q`` of the cell at ``node - corner``, as an offset from the node.
                a = 1 + (q >> 2) - (corner >> 2)
                b = 1 + ((q >> 1) & 1) - ((corner >> 1) & 1)
                c = 1 + (q & 1) - (corner & 1)
                interpolated += screen[base + 1 + q] * window[9 * a + 3 * b + c]
            total += weight * phi * interpolated
            diagonal += weight * phi * phi
    return wp.vec2(total, diagonal)


@wp.func
def screening_term(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    alpha: wp.float32,
    node: wp.vec3i,
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    counts: wp.array[wp.float32],
    position_sums: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    sample_counts: wp.array[wp.int32],
    sample_positions: wp.array[wp.vec3],
) -> wp.float32:
    # The screening's right-hand side at depth ``d`` node ``node``: target value 1/2 at every
    # screening sample.
    scale = 0.5 * alpha * wp.float32(1 << d)
    total = wp.float32(0.0)
    for corner in range(8):
        cell = node - wp.vec3i(corner >> 2, (corner >> 1) & 1, corner & 1)
        sample = cell_screen(
            d,
            depth,
            full_depth,
            cell,
            flags,
            flag_offsets,
            counts,
            position_sums,
            cell_offsets,
            slots,
            bricks,
            sample_counts,
            sample_positions,
        )
        if sample[3] > 0.0:
            total += (
                scale * sample[3] * hat_value(d, node, wp.vec3(sample[0], sample[1], sample[2]))
            )
    return total


@wp.kernel
def octree_coarse_screening_constraints(
    depth: wp.int32,
    full_depth: wp.int32,
    alpha: wp.float32,
    accumulated: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    counts: wp.array[wp.float32],
    position_sums: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    sample_counts: wp.array[wp.int32],
    sample_positions: wp.array[wp.vec3],
    node_bricks: wp.array[wp.int32],
    brick_slots: wp.array[wp.int32],
    integrals: wp.array[wp.vec2d],
    out_b: wp.array[wp.float32],
    out_base: wp.array[wp.float32],
) -> None:
    # Every stored node's right-hand side finished, after its own and finer data (restricted): the
    # coarser data's divergence carried up from the parent depth, then the screening. Depth 0's
    # eight entries (in lattice order), its one cell's point count and position sum also go to
    # ``out_base``, the host's base solve in one read.
    e = wp.int32(wp.tid())
    at = stored_node(node_bricks, brick_slots, depth, e)
    d = at[0]
    if d < 0:
        return
    node = wp.vec3i(at[1], at[2], at[3])
    b = out_b[e]
    if d > 0:
        b = b - divergence_term(d, d - 1, accumulated, cell_offsets, bricks, depth, node, integrals)
    b = b + screening_term(
        d,
        depth,
        full_depth,
        alpha,
        node,
        flags,
        flag_offsets,
        counts,
        position_sums,
        cell_offsets,
        slots,
        bricks,
        sample_counts,
        sample_positions,
    )
    out_b[e] = b
    if d == 0:
        flat = lattice_index(0, node)
        out_base[flat] = b
        if flat == 0:
            out_base[8] = counts[0]
            p = position_sums[0]
            out_base[9] = p[0]
            out_base[10] = p[1]
            out_base[11] = p[2]


@wp.func
def stored_sample(
    values: wp.array[wp.float32], node_bricks: wp.array[wp.int32], d: wp.int32, g: wp.vec3
) -> wp.float32:
    # ``poisson_sample_grid`` of depth ``d``'s stored nodes at lattice coordinate ``g``: the same
    # cell, corners and arithmetic (``poisson_lerp_corners``).
    res = (1 << d) + 1
    base, far, fractions = trilinear_cell(g, wp.vec3i(res, res, res))
    low = wp.vec4(
        node_value(values, node_bricks, d, wp.vec3i(base[0], base[1], base[2])),
        node_value(values, node_bricks, d, wp.vec3i(far[0], base[1], base[2])),
        node_value(values, node_bricks, d, wp.vec3i(base[0], far[1], base[2])),
        node_value(values, node_bricks, d, wp.vec3i(far[0], far[1], base[2])),
    )
    high = wp.vec4(
        node_value(values, node_bricks, d, wp.vec3i(base[0], base[1], far[2])),
        node_value(values, node_bricks, d, wp.vec3i(far[0], base[1], far[2])),
        node_value(values, node_bricks, d, wp.vec3i(base[0], far[1], far[2])),
        node_value(values, node_bricks, d, wp.vec3i(far[0], far[1], far[2])),
    )
    return poisson_lerp_corners(low, high, fractions)


@wp.func
def prolonged_at(
    fields: wp.array[wp.float32], node_bricks: wp.array[wp.int32], d: wp.int32, node: wp.vec3i
) -> wp.float32:
    # Depth ``d``'s (``d >= 1``) coarser field prolonged at ``node``: ``poisson_prolong_node`` over
    # the stored depth ``d - 1`` nodes. Its weights are 0, 1/2 and 1, so every product is exact and
    # the value takes the same bits in any kernel: the sweeps' start (``octree_prolong``) and the
    # extraction's slopes form it alike, and no prolonged copy is kept.
    return stored_sample(
        fields,
        node_bricks,
        d - 1,
        wp.vec3(wp.float32(node[0]) * 0.5, wp.float32(node[1]) * 0.5, wp.float32(node[2]) * 0.5),
    )


@wp.kernel
def octree_prolong(
    depth: wp.int32,
    first: wp.int32,
    node_bricks: wp.array[wp.int32],
    brick_slots: wp.array[wp.int32],
    fields: wp.array[wp.float32],
) -> None:
    # One depth's stored nodes (from entry ``first``) start at the coarser field prolonged, the
    # start of its Gauss-Seidel sweeps.
    e = wp.int32(wp.tid()) + first
    at = stored_node(node_bricks, brick_slots, depth, e)
    if at[0] >= 0:
        fields[e] = prolonged_at(fields, node_bricks, at[0], wp.vec3i(at[1], at[2], at[3]))


@wp.func
def gauss_seidel_node(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    wave: wp.int32,
    sweep: wp.int32,
    colour: wp.int32,
    a: wp.int32,
    c: wp.int32,
    b: wp.array[wp.float32],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    screen_index: wp.array[wp.int32],
    screen: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    field: wp.array[wp.float32],
) -> None:
    # The Gauss-Seidel update of node ``(2 a + colour % 2, 2 c + colour / 2)`` on the slice sweep
    # ``sweep`` relaxes on wavefront ``wave = 2 sweep + (res - 1 - z)``, when the node is active;
    # the caller passes only sweeps whose slice exists. ``field`` starts at the coarser field
    # prolonged, so its updates are the upstream correction's (which starts at zero against
    # ``b - A prolonged``) with the coarser field added. Shared by the two schedules of the same
    # sweeps: ``octree_gauss_seidel_wave`` (one launch a wave and colour, CUDA's) and
    # ``octree_gauss_seidel_depth`` (every wave of a depth in one thread, the CPU's). The updates
    # of one wave and colour touch nodes two apart in ``x``, ``y`` and (across sweeps) ``z``,
    # beyond the stencil's reach, so their order is free and the two give the same bits. ``b``
    # and ``field`` hold every depth's stored nodes (``node_bricks``).
    res = (1 << d) + 1
    z = res - 1 - (wave - 2 * sweep)
    x = 2 * a + (colour & 1)
    y = 2 * c + (colour >> 1)
    if x >= res or y >= res:
        return
    node = wp.vec3i(x, y, z)
    if not node_active(flags, flag_offsets, full_depth, d, node):
        return
    row = octree_row(
        d, depth, node, field, node_bricks, cell_offsets, slots, bricks, screen_index, screen
    )
    index = node_entry(node_bricks, d, node)
    field[index] = field[index] + (b[index] - row[0]) / row[1]


@wp.kernel
def octree_gauss_seidel_wave(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    wave: wp.int32,
    first_sweep: wp.int32,
    colour: wp.int32,
    b: wp.array[wp.float32],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    screen_index: wp.array[wp.int32],
    screen: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    field: wp.array[wp.float32],
) -> None:
    # One colour of every ``(sweep, slice)`` on wavefront ``wave``, the sweeps from
    # ``first_sweep`` on: a launch a wave and colour, sized to the wave's sweeps.
    #
    # **Recording the waves as a CUDA graph is declined** (measured 2026-10-26): with the wave read
    # off a device counter, a group of 8 waves recorded once per depth and replayed was
    # bit-identical but 0.93-0.96x at depths 6-7 and 0.97-1.00x at depth 8. The sweeps are
    # device-bound, not launch-bound: the call's kernel time equals its wall time (depth 7: 37.6
    # against 36.1 ms), each wave launch costs ~14 us of device time rebuilding its rows
    # (``octree_row``), and the host issues launches faster than the device drains them. The lever
    # is a launch's own cost, not the launch count.
    s, a, c = wp.tid()
    gauss_seidel_node(
        d,
        depth,
        full_depth,
        wave,
        first_sweep + s,
        colour,
        a,
        c,
        b,
        flags,
        flag_offsets,
        cell_offsets,
        slots,
        bricks,
        screen_index,
        screen,
        node_bricks,
        field,
    )


@wp.kernel
def octree_gauss_seidel_depth(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    iterations: wp.int32,
    b: wp.array[wp.float32],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    bricks: wp.array[wp.int32],
    screen_index: wp.array[wp.int32],
    screen: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    field: wp.array[wp.float32],
) -> None:
    # Every wave and colour of depth ``d`` in one thread, in ``octree_gauss_seidel_wave``'s launch
    # order: the CPU device runs a launch's threads one after another anyway, so this is those
    # launches' arithmetic in one launch.
    res = (1 << d) + 1
    half = (res + 1) // 2
    for wave in range(res + 2 * (iterations - 1)):
        first = wp.max(0, (wave - res + 2) // 2)
        last = wp.min(iterations - 1, wave // 2)
        for colour in range(4):
            for sweep in range(first, last + 1):
                for a in range(half):
                    for c in range(half):
                        gauss_seidel_node(
                            d,
                            depth,
                            full_depth,
                            wave,
                            sweep,
                            colour,
                            a,
                            c,
                            b,
                            flags,
                            flag_offsets,
                            cell_offsets,
                            slots,
                            bricks,
                            screen_index,
                            screen,
                            node_bricks,
                            field,
                        )


@wp.kernel
def octree_iso_terms(
    sample_counts: wp.array[wp.int32],
    sample_positions: wp.array[wp.vec3],
    fields: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    depth: wp.int32,
    out_terms: wp.array[wp.float32],
) -> None:
    # A sample's count times the finest field at its centroid; their sum over the point count is
    # the iso-value.
    s = wp.int32(wp.tid())
    count = wp.float32(sample_counts[s])
    g = sample_positions[s] / count * wp.float32(1 << depth)
    out_terms[s] = count * stored_sample(fields, node_bricks, depth, g)


# ---------------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------------
#
# PoissonRecon's extraction, cell by cell (``IsoSurfaceExtractor`` in its
# ``FEMTree.IsoSurface.specialized.inl``), run as independent per-depth and per-cell passes.
#
# - **Corner values.** A leaf reads its corners through its depth's lattice, and every finer leaf
#   copies its corner values (and slopes) up to the coarser lattices at the same point, finest
#   first: a corner takes the value of the deepest leaf having it as a corner (``corner_sources``).
# - **Edge vertices.** Finest depth first, an edge of a refined cell takes the vertex of the one
#   finer half that has one, or pairs the two halves' vertices when both do; an edge of a leaf whose
#   corners straddle the iso-value and that holds no vertex yet gets its own (``classify_edges``).
# - **Faces.** A leaf's face is marching squares on its corners (the below-iso corners joined
#   through the centre on an ambiguous face), unless the cell across is refined: then the face is
#   the finer leaves' faces on it, in the order the upstream single-threaded pass pushes them
#   (``borrowed_face_key``).
# - **Polygons.** The faces' segments, oriented by each face's flag, are walked into loops exactly
#   as the upstream does (last segment first, swap-removal, a pair crossed where a segment ends on
#   one), and each loop is triangulated by minimal area (``polygon_triangles``).
#
# The tables come from PoissonRecon's ``HyperCube`` (corner ``c`` is ``(c & 1, c >> 1 & 1,
# c >> 2 & 1)``; faces ``z0 y0 x0 x1 y1 z1``; a face's square edges ``x0 y0 y1 x1`` in its own
# axes), dumped by compiling its headers.

# Packed table: cube edges' lower-corner offset bits and axis (12), each face's four square edges
# as cube edges (24), each face's four square corners as cube corners (24), each face's orientation
# flag (6), and marching squares' segments per square case (16 x 4, -1 padded).
EXTRACTION_EDGES = 0
EXTRACTION_FACE_EDGES = 12
EXTRACTION_FACE_CORNERS = 36
EXTRACTION_FACE_ORIENTED = 60
EXTRACTION_SQUARE_SEGMENTS = 66
_EDGE_CORNERS = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)
_SQUARE_SEGMENTS = (
    (),
    (1, 0),
    (0, 2),
    (1, 2),
    (3, 1),
    (3, 0),
    (0, 1, 3, 2),
    (3, 2),
    (2, 3),
    (1, 3, 2, 0),
    (0, 3),
    (1, 3),
    (2, 1),
    (2, 0),
    (0, 1),
    (),
)
EXTRACTION_TABLE = (
    *(a | ((b ^ a).bit_length() - 1) << 3 for a, b in _EDGE_CORNERS),
    *(0, 1, 2, 3, 0, 4, 5, 8, 1, 4, 6, 9, 2, 5, 7, 10, 3, 6, 7, 11, 8, 9, 10, 11),
    *(0, 1, 2, 3, 0, 1, 4, 5, 0, 2, 4, 6, 1, 3, 5, 7, 2, 3, 6, 7, 4, 5, 6, 7),
    *(0, 1, 0, 1, 0, 1),
    *(x for seg in _SQUARE_SEGMENTS for x in (*seg, -1, -1, -1, -1)[:4]),
)


@wp.func
def face_axis(f: wp.int32) -> wp.int32:
    # The axis a face of the cube is normal to (faces ``z0 y0 x0 x1 y1 z1``).
    axis = wp.int32(2)
    if f == 1 or f == 4:
        axis = 1
    elif f == 2 or f == 3:
        axis = 0
    return axis


@wp.func
def face_across(c: wp.vec3i, f: wp.int32) -> wp.vec3i:
    # The cell sharing face ``f`` of cell ``c``.
    across = wp.vec3i(c[0], c[1], c[2])
    axis = face_axis(f)
    across[axis] = across[axis] + wp.where(f >= 3, 1, -1)
    return across


@wp.func
def octree_inside(d: wp.int32, c: wp.vec3i) -> wp.bool:
    n = 1 << d
    return c[0] >= 0 and c[1] >= 0 and c[2] >= 0 and c[0] < n and c[1] < n and c[2] < n


@wp.func
def leaf_cell(
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    depth: wp.int32,
    full_depth: wp.int32,
    d: wp.int32,
    c: wp.vec3i,
) -> wp.bool:
    # Whether depth ``d`` cell ``c`` is a leaf: inside the cube, active, its children not.
    if not octree_inside(d, c):
        return False
    if not node_active(flags, flag_offsets, full_depth, d, c):
        return False
    if d == depth:
        return True
    return not children_active(flags, flag_offsets, d, c)


@wp.func
def refined_cell(
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    depth: wp.int32,
    full_depth: wp.int32,
    d: wp.int32,
    c: wp.vec3i,
) -> wp.bool:
    # Whether depth ``d`` cell ``c`` is inside the cube, active and has active children.
    if d >= depth or not octree_inside(d, c):
        return False
    return node_active(flags, flag_offsets, full_depth, d, c) and children_active(
        flags, flag_offsets, d, c
    )


@wp.func
def split_slope(
    fields: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    d: wp.int32,
    node: wp.vec3i,
    axis: wp.int32,
) -> wp.float32:
    # The field's slope along ``axis`` at depth ``d``'s lattice node, times the depth's cell width,
    # as the upstream evaluator forms it: the depth's own correction's (``field - prolonged``)
    # averaged one-sided slopes plus the coarser field's slope on its own lattice (averaged at its
    # nodes, exact between them), interpolated linearly across the other two axes.
    res = (1 << d) + 1
    res_c = (1 << (d - 1)) + 1
    up = wp.vec3i(node[0], node[1], node[2])
    down = wp.vec3i(node[0], node[1], node[2])
    up[axis] = wp.min(node[axis] + 1, res - 1)
    down[axis] = wp.max(node[axis] - 1, 0)
    slope = 0.5 * (
        (node_value(fields, node_bricks, d, up) - prolonged_at(fields, node_bricks, d, up))
        - (node_value(fields, node_bricks, d, down) - prolonged_at(fields, node_bricks, d, down))
    )
    u = (axis + 1) % 3
    v = (axis + 2) % 3
    coarse_slope = wp.float32(0.0)
    for q in range(4):
        cu = node[u] >> 1
        cv = node[v] >> 1
        weight = wp.float32(1.0)
        if node[u] % 2 == 1:
            cu += q & 1
            weight *= 0.5
        elif (q & 1) == 1:
            weight = wp.float32(0.0)
        if node[v] % 2 == 1:
            cv += q >> 1
            weight *= 0.5
        elif (q >> 1) == 1:
            weight = wp.float32(0.0)
        if weight > 0.0:
            m = node[axis] >> 1
            hi = wp.vec3i(0, 0, 0)
            lo = wp.vec3i(0, 0, 0)
            hi[u] = cu
            hi[v] = cv
            lo[u] = cu
            lo[v] = cv
            factor = wp.float32(0.25)
            if node[axis] % 2 == 0:
                hi[axis] = wp.min(m + 1, res_c - 1)
                lo[axis] = wp.max(m - 1, 0)
            else:
                hi[axis] = m + 1
                lo[axis] = m
                factor = wp.float32(0.5)
            coarse_slope += (
                weight
                * factor
                * (
                    node_value(fields, node_bricks, d - 1, hi)
                    - node_value(fields, node_bricks, d - 1, lo)
                )
            )
    return slope + coarse_slope


@wp.func
def touched_by_leaf(
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    depth: wp.int32,
    full_depth: wp.int32,
    d: wp.int32,
    node: wp.vec3i,
) -> wp.bool:
    # Whether a depth ``d`` leaf has lattice node ``node`` as a corner.
    for k in range(8):
        cell = node - wp.vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1)
        if leaf_cell(flags, flag_offsets, depth, full_depth, d, cell):
            return True
    return False


@wp.kernel
def corner_sources(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    node_offsets: wp.array[wp.int32],
    sources: wp.array[wp.int8],
) -> None:
    # The depth whose value a depth ``d`` leaf reads at this lattice node: the deepest leaf having
    # the point as a corner (copied up the lattices finest first), ``d`` itself when only depth
    # ``d`` leaves have it, -1 when none. Run finest first.
    i, j, k = wp.tid()
    node = wp.vec3i(i, j, k)
    source = wp.int32(-1)
    if d < depth:
        source = wp.int32(sources[node_offsets[d + 1] + lattice_index(d + 1, 2 * node)])
    if source < 0 and touched_by_leaf(flags, flag_offsets, depth, full_depth, d, node):
        source = d
    sources[node_offsets[d] + lattice_index(d, node)] = wp.int8(source)


@wp.func
def corner_sample(
    fields: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    sources: wp.array[wp.int8],
    node_offsets: wp.array[wp.int32],
    d: wp.int32,
    node: wp.vec3i,
    axis: wp.int32,
) -> wp.vec2:
    # ``(value, slope along axis times depth d's cell width)`` a depth ``d`` leaf reads at
    # ``node``: the source depth's lattice value and slope (``corner_sources``).
    source = wp.int32(sources[node_offsets[d] + lattice_index(d, node)])
    if source < 0:
        source = d
    shift = source - d
    at = wp.vec3i(node[0] << shift, node[1] << shift, node[2] << shift)
    value = node_value(fields, node_bricks, source, at)
    slope = wp.float32(0.0)
    if axis >= 0:
        slope = split_slope(fields, node_bricks, source, at, axis) * wp.float32(1 << shift)
    return wp.vec2(value, slope)


@wp.func
def corner_below(
    fields: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    sources: wp.array[wp.int8],
    node_offsets: wp.array[wp.int32],
    iso: wp.float32,
    d: wp.int32,
    node: wp.vec3i,
) -> wp.bool:
    # The upstream marching-cubes bit: the corner's value is below the iso-value.
    sample = corner_sample(fields, node_bricks, sources, node_offsets, d, node, -1)
    return sample[0] < iso


@wp.func
def cell_case(
    fields: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    sources: wp.array[wp.int8],
    node_offsets: wp.array[wp.int32],
    iso: wp.float32,
    d: wp.int32,
    c: wp.vec3i,
) -> wp.int32:
    # A leaf's 8-bit case, bit ``k`` set when corner ``k`` is below the iso-value.
    code = wp.int32(0)
    for k in range(8):
        corner = c + wp.vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1)
        if corner_below(fields, node_bricks, sources, node_offsets, iso, d, corner):
            code = code | (1 << k)
    return code


@wp.func
def edge_slot(
    edge_offsets: wp.array[wp.int32], d: wp.int32, axis: wp.int32, node: wp.vec3i
) -> wp.int32:
    # Depth ``d``'s edge from ``node`` along ``axis`` in the packed edge-vertex table.
    res = (1 << d) + 1
    return edge_offsets[d] + axis * res * res * res + lattice_index(d, node)


@wp.func
def fitted_edge_root(
    here: wp.float32,
    there: wp.float32,
    slope_here: wp.float32,
    slope_there: wp.float32,
    iso: wp.float32,
) -> wp.float64:
    # The upstream non-linear fit, in its precisions: the end slopes rescaled so the Hermite cubic
    # loses its cubic term (``float32`` value differences, ``float64`` slopes), the mean of the
    # quadratic's real roots in ``[0, 1]`` as ``Factor`` forms them, else the ``float32`` linear
    # root.
    step = there - here
    d0 = wp.float64(slope_here)
    d1 = wp.float64(slope_there)
    scale = wp.float64(step) / ((d1 + d0) / wp.float64(2.0))
    d0 = d0 * scale
    d1 = d1 * scale
    a2 = wp.float64(3.0 * step) - d1 - wp.float64(2.0) * d0
    a1 = d0
    a0 = wp.float64(here) - wp.float64(iso)
    roots = wp.float64(0.0)
    found = wp.int32(0)
    if wp.abs(a2) <= wp.float64(0.0):
        if wp.abs(a1) > wp.float64(0.0):
            t = -a0 / a1
            if t >= wp.float64(0.0) and t <= wp.float64(1.0):
                roots = t
                found = 1
    else:
        disc = a1 * a1 - wp.float64(4.0) * a0 * a2
        if disc >= wp.float64(0.0):
            half = a1 / (wp.float64(2.0) * a2)
            root = wp.sqrt(disc) / (wp.float64(2.0) * a2)
            for sign in range(2):
                t = -half + wp.where(sign == 0, -root, root)
                if t >= wp.float64(0.0) and t <= wp.float64(1.0):
                    roots += t
                    found += 1
    if found > 0:
        return roots / wp.float64(found)
    return wp.float64((iso - here) / step)


@wp.func
def edge_cells(
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    depth: wp.int32,
    full_depth: wp.int32,
    d: wp.int32,
    axis: wp.int32,
    node: wp.vec3i,
) -> wp.vec2i:
    # Whether any of the four depth ``d`` cells around the edge from ``node`` along ``axis`` is
    # refined, and whether any is a leaf (1 or 0 each).
    u = (axis + 1) % 3
    v = (axis + 2) % 3
    refined = wp.int32(0)
    leaf = wp.int32(0)
    for q in range(4):
        cell = wp.vec3i(node[0], node[1], node[2])
        cell[u] = cell[u] - (q & 1)
        cell[v] = cell[v] - (q >> 1)
        if refined_cell(flags, flag_offsets, depth, full_depth, d, cell):
            refined = 1
        if leaf_cell(flags, flag_offsets, depth, full_depth, d, cell):
            leaf = 1
    return wp.vec2i(refined, leaf)


@wp.func
def edge_owner(
    holds: wp.array[wp.uint32],
    created: wp.array[wp.uint32],
    edge_offsets: wp.array[wp.int32],
    depth: wp.int32,
    d: wp.int32,
    axis: wp.int32,
    node: wp.vec3i,
) -> wp.int32:
    # The vertex depth ``d``'s edge from ``node`` along ``axis`` holds, named by the edge that
    # created it, -1 for none. An edge holding a vertex it did not create took it over from the
    # one finer half holding one (``classify_edges``), so the walk follows those halves down to
    # the creating edge: one bit an edge (``holds``) instead of a name an edge.
    slot = edge_slot(edge_offsets, d, axis, node)
    if not bit_is_set(holds, slot):
        return -1
    step = wp.vec3i(0, 0, 0)
    step[axis] = 1
    at = wp.vec3i(node[0], node[1], node[2])
    level = d
    while level < depth and not bit_is_set(created, slot):
        level += 1
        at = 2 * at
        slot = edge_slot(edge_offsets, level, axis, at)
        if not bit_is_set(holds, slot):
            at = at + step
            slot = edge_slot(edge_offsets, level, axis, at)
    return slot


@wp.func
def finer_half_owners(
    holds: wp.array[wp.uint32],
    created: wp.array[wp.uint32],
    edge_offsets: wp.array[wp.int32],
    depth: wp.int32,
    d: wp.int32,
    axis: wp.int32,
    node: wp.vec3i,
) -> wp.vec2i:
    # The vertices the two depth ``d + 1`` halves of an edge hold (``edge_owner``, -1 for none).
    step = wp.vec3i(0, 0, 0)
    step[axis] = 1
    return wp.vec2i(
        edge_owner(holds, created, edge_offsets, depth, d + 1, axis, 2 * node),
        edge_owner(holds, created, edge_offsets, depth, d + 1, axis, 2 * node + step),
    )


@wp.kernel
def classify_edges(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    iso: wp.float32,
    fields: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    sources: wp.array[wp.int8],
    node_offsets: wp.array[wp.int32],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    edge_offsets: wp.array[wp.int32],
    holds: wp.array[wp.uint32],
    created: wp.array[wp.uint32],
) -> None:
    # Depth ``d``'s edge from node ``(i, j, k)`` along ``axis``: whether it holds a vertex (bit
    # ``slot`` of ``holds``; ``edge_owner`` names it). An edge of a refined cell takes over the
    # vertex of the one finer half that has one (two halves with one each are paired by
    # ``pair_edge_vertices``); an edge still without one gets its own (bit ``slot`` of
    # ``created``) when a leaf's corners straddle the iso-value along it. Run finest first.
    axis, i, j, k = wp.tid()
    res = (1 << d) + 1
    node = wp.vec3i(i, j, k)
    end = wp.vec3i(i, j, k)
    end[axis] = end[axis] + 1
    if end[axis] >= res:
        return
    slot = edge_slot(edge_offsets, d, axis, node)
    around = edge_cells(flags, flag_offsets, depth, full_depth, d, axis, node)
    held = wp.int32(0)
    if around[0] != 0:
        step = wp.vec3i(0, 0, 0)
        step[axis] = 1
        lower = bit_is_set(holds, edge_slot(edge_offsets, d + 1, axis, 2 * node))
        upper = bit_is_set(holds, edge_slot(edge_offsets, d + 1, axis, 2 * node + step))
        if lower != upper:
            held = 1
    if held == 0 and around[1] != 0:
        here = corner_below(fields, node_bricks, sources, node_offsets, iso, d, node)
        there = corner_below(fields, node_bricks, sources, node_offsets, iso, d, end)
        if here != there:
            held = 1
            wp.atomic_or(created, slot >> 5, wp.uint32(1) << wp.uint32(slot & 31))
    if held != 0:
        wp.atomic_or(holds, slot >> 5, wp.uint32(1) << wp.uint32(slot & 31))


@wp.kernel
def pair_edge_vertices(
    first_depth: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    edge_offsets: wp.array[wp.int32],
    holds: wp.array[wp.uint32],
    created: wp.array[wp.uint32],
    ranks: wp.array[wp.int32],
    out_partners: wp.array[wp.int32],
) -> None:
    # The edges ``classify_edges`` leaves without a vertex because both finer halves hold one:
    # each half's vertex names the other (``out_partners``, by vertex number), the crossing the loop
    # walk takes when no segment leaves a vertex. One thread an edge of every depth but the finest.
    slot = wp.int32(wp.tid()) + edge_offsets[first_depth]
    d = level_of(edge_offsets, first_depth, depth - 1, slot)
    res = (1 << d) + 1
    cube = res * res * res
    local = slot - edge_offsets[d]
    axis = local // cube
    r = local % cube
    node = wp.vec3i(r // (res * res), (r // res) % res, r % res)
    if node[axis] + 1 >= res:
        return
    if edge_cells(flags, flag_offsets, depth, full_depth, d, axis, node)[0] == 0:
        return
    halves = finer_half_owners(holds, created, edge_offsets, depth, d, axis, node)
    if halves[0] >= 0 and halves[1] >= 0:
        out_partners[bit_rank(created, ranks, halves[0])] = halves[1]
        out_partners[bit_rank(created, ranks, halves[1])] = halves[0]


@wp.kernel
def place_edge_vertices(
    d: wp.int32,
    iso: wp.float32,
    margin: wp.float32,
    fields: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    sources: wp.array[wp.int8],
    node_offsets: wp.array[wp.int32],
    edge_offsets: wp.array[wp.int32],
    created: wp.array[wp.uint32],
    ranks: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
) -> None:
    # A created vertex's unit-cube position at its number (its bit's rank in ``created``, ``ranks``
    # the scan of the words' bit counts), by the non-linear fit on its edge's corner values and
    # slopes, kept ``margin`` of the edge off its ends.
    axis, i, j, k = wp.tid()
    node = wp.vec3i(i, j, k)
    slot = edge_slot(edge_offsets, d, axis, node)
    if not bit_is_set(created, slot):
        return
    end = wp.vec3i(i, j, k)
    end[axis] = end[axis] + 1
    here = corner_sample(fields, node_bricks, sources, node_offsets, d, node, axis)
    there = corner_sample(fields, node_bricks, sources, node_offsets, d, end, axis)
    t = fitted_edge_root(here[0], there[0], here[1], there[1], iso)
    t = wp.clamp(t, wp.float64(margin), wp.float64(1.0 - margin))
    # The upstream's position: the cell's corner in ``float32`` and ``start + width * t`` along the
    # edge in ``float64``, rounded once.
    width = 1.0 / wp.float32(1 << d)
    position = wp.vec3(wp.float32(i) * width, wp.float32(j) * width, wp.float32(k) * width)
    position[axis] = wp.float32(wp.float64(position[axis]) + wp.float64(width) * t)
    out_positions[bit_rank(created, ranks, slot)] = position


@wp.func
def face_segments(
    table: wp.array[wp.int32],
    holds: wp.array[wp.uint32],
    created: wp.array[wp.uint32],
    edge_offsets: wp.array[wp.int32],
    depth: wp.int32,
    code: wp.int32,
    d: wp.int32,
    c: wp.vec3i,
    f: wp.int32,
) -> wp.vec4i:
    # Marching squares on face ``f`` of depth ``d`` cell ``c`` (case ``code``): up to two segments
    # as vertex pairs (``edge_owner``'s names) in the face's own orientation, ``-1`` past the last.
    square = wp.int32(0)
    for s in range(4):
        if (code >> table[EXTRACTION_FACE_CORNERS + 4 * f + s]) & 1 != 0:
            square = square | (1 << s)
    out = wp.vec4i(-1, -1, -1, -1)
    for s in range(4):
        local = table[EXTRACTION_SQUARE_SEGMENTS + 4 * square + s]
        if local >= 0:
            packed = table[EXTRACTION_EDGES + table[EXTRACTION_FACE_EDGES + 4 * f + local]]
            corner = wp.vec3i(packed & 1, (packed >> 1) & 1, (packed >> 2) & 1)
            out[s] = edge_owner(holds, created, edge_offsets, depth, d, packed >> 3, c + corner)
    return out


@wp.func
def unpack_cell(
    cell_offsets: wp.array[wp.int32], first: wp.int32, last: wp.int32, flat: wp.int32
) -> wp.vec4i:
    # ``(x, y, z, depth)`` of the cell at ``flat`` in the packed per-depth cell range.
    d = level_of(cell_offsets, first, last, flat)
    n = 1 << d
    local = flat - cell_offsets[d]
    return wp.vec4i(local // (n * n), (local // n) % n, local % n, d)


@wp.func
def depth_first_rank(c: wp.vec3i, d: wp.int32) -> wp.int32:
    # Rank of a depth ``d`` cell among those of its slice in the upstream's depth-first order:
    # per level the ``y`` bit above the ``x`` bit.
    rank = wp.int32(0)
    for b in range(d - 1, -1, -1):
        rank = (rank << 2) | (((c[1] >> b) & 1) << 1) | ((c[0] >> b) & 1)
    return rank


@wp.kernel
def extraction_band(
    first_depth: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    iso: wp.float32,
    fields: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    sources: wp.array[wp.int8],
    node_offsets: wp.array[wp.int32],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    out_band: wp.array[wp.uint32],
) -> None:
    # Bit ``t`` set for the leaves that can have segments (``cell_segments``' test): corners
    # straddling the iso-value, or a refined cell across a face, whose finer faces it takes over.
    # A borrowed face is a straddling leaf's, so it is in the band too. The extraction's per-cell
    # passes run over these alone, in the same order.
    t = wp.int32(wp.tid())
    flat = t + cell_offsets[first_depth]
    cell = unpack_cell(cell_offsets, first_depth, depth, flat)
    d = cell[3]
    c = wp.vec3i(cell[0], cell[1], cell[2])
    if not leaf_cell(flags, flag_offsets, depth, full_depth, d, c):
        return
    code = cell_case(fields, node_bricks, sources, node_offsets, iso, d, c)
    keep = code != 0 and code != 255
    for f in range(6):
        if not keep and refined_cell(flags, flag_offsets, depth, full_depth, d, face_across(c, f)):
            keep = True
    if keep:
        wp.atomic_or(out_band, t >> 5, wp.uint32(1) << wp.uint32(t & 31))


@wp.kernel
def band_cells(
    band: wp.array[wp.uint32], ranks: wp.array[wp.int32], out_cells: wp.array[wp.int32]
) -> None:
    # The band's cells (``extraction_band``) in ascending order: each word's set bits at their rank.
    w = wp.int32(wp.tid())
    word = band[w]
    if word == wp.uint32(0):
        return
    rank = bit_rank(band, ranks, 32 * w)
    for b in range(32):
        if (word >> wp.uint32(b)) & wp.uint32(1) != wp.uint32(0):
            out_cells[rank] = 32 * w + b
            rank += 1


@wp.func
def borrowed_face_key(
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    depth: wp.int32,
    full_depth: wp.int32,
    d: wp.int32,
    c: wp.vec3i,
    f: wp.int32,
) -> wp.uint64:
    # When face ``f`` of leaf ``c`` lies on a coarser leaf ``C`` whose cell across is refined, the
    # sort key placing its segments in ``C``'s face list in the upstream push order: ``C``, ``C``'s
    # face, then for a ``z`` face the finer depth first and the slice's depth-first rank, for an
    # ``x`` or ``y`` face the slab's top first, then the finer depth and the rank. 0 otherwise.
    across = face_across(c, f)
    if not octree_inside(d, across) or node_active(flags, flag_offsets, full_depth, d, across):
        return wp.uint64(0)
    key = wp.uint64(0)
    for step in range(1, d - full_depth + 1):
        j = d - step
        coarse = wp.vec3i(across[0] >> step, across[1] >> step, across[2] >> step)
        if node_active(flags, flag_offsets, full_depth, j, coarse):
            mine = wp.vec3i(c[0] >> step, c[1] >> step, c[2] >> step)
            if leaf_cell(flags, flag_offsets, depth, full_depth, j, coarse) and refined_cell(
                flags, flag_offsets, depth, full_depth, j, mine
            ):
                order = (wp.uint64(15 - d) << wp.uint64(18)) | wp.uint64(depth_first_rank(c, d))
                if face_axis(f) != 2:
                    slab_end = wp.uint64((c[2] + 1) << (depth - d))
                    order = order | (slab_end << wp.uint64(22))
                flat = cell_offsets[j] + cell_flat(1 << j, coarse)
                key = (
                    (wp.uint64(flat) << wp.uint64(35)) | (wp.uint64(5 - f) << wp.uint64(32)) | order
                )
            return key
    return key


@wp.kernel
def borrowed_faces(
    first_depth: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    iso: wp.float32,
    fields: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    sources: wp.array[wp.int8],
    node_offsets: wp.array[wp.int32],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    table: wp.array[wp.int32],
    holds: wp.array[wp.uint32],
    created: wp.array[wp.uint32],
    edge_offsets: wp.array[wp.int32],
    band: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
    out_faces: wp.array[wp.int32],
) -> None:
    # One band leaf's face with segments lying on a coarser leaf's face (``borrowed_face_key``):
    # counted into ``offsets`` while ``out_keys`` is empty, written at its scanned slot otherwise.
    t = wp.int32(wp.tid())
    flat = band[t // 6] + cell_offsets[first_depth]
    f = t % 6
    cell = unpack_cell(cell_offsets, first_depth, depth, flat)
    d = cell[3]
    c = wp.vec3i(cell[0], cell[1], cell[2])
    key = wp.uint64(0)
    if leaf_cell(flags, flag_offsets, depth, full_depth, d, c):
        key = borrowed_face_key(flags, flag_offsets, cell_offsets, depth, full_depth, d, c, f)
        if key != wp.uint64(0):
            code = cell_case(fields, node_bricks, sources, node_offsets, iso, d, c)
            if face_segments(table, holds, created, edge_offsets, depth, code, d, c, f)[0] < 0:
                key = wp.uint64(0)
    if out_keys.shape[0] == 0:
        offsets[t] = wp.where(key != wp.uint64(0), 1, 0)
    elif key != wp.uint64(0):
        slot = offsets[t] - 1
        out_keys[slot] = key
        out_faces[slot] = flat * 6 + f


@wp.func
def push_face(
    segments: wp.vec4i,
    flip: wp.int32,
    count: wp.int32,
    base: wp.int32,
    out_segments: wp.array[wp.vec2i],
) -> wp.int32:
    # Append a face's segments, reversed when the cell's face is not oriented; return the count. A
    # segment with an end on an edge holding no vertex (the upstream stops there) is skipped.
    for s in range(2):
        if segments[2 * s] >= 0 and segments[2 * s + 1] >= 0:
            if out_segments.shape[0] > 0:
                a = segments[2 * s + flip]
                b = segments[2 * s + 1 - flip]
                out_segments[base + count] = wp.vec2i(a, b)
            count += 1
    return count


@wp.kernel
def cell_segments(
    first_depth: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    iso: wp.float32,
    fields: wp.array[wp.float32],
    node_bricks: wp.array[wp.int32],
    sources: wp.array[wp.int8],
    node_offsets: wp.array[wp.int32],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    table: wp.array[wp.int32],
    holds: wp.array[wp.uint32],
    created: wp.array[wp.uint32],
    edge_offsets: wp.array[wp.int32],
    borrowed_keys: wp.array[wp.uint64],
    borrowed: wp.array[wp.int32],
    band: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_segments: wp.array[wp.vec2i],
) -> None:
    # A band leaf's directed segments, face by face (``z0 y0 x0 x1 y1 z1``): its own face's
    # marching squares, or the finer leaves' faces across it in their sorted order. Counted into
    # ``offsets`` while ``out_segments`` is empty, written after the scanned count otherwise.
    t = wp.int32(wp.tid())
    flat = band[t] + cell_offsets[first_depth]
    cell = unpack_cell(cell_offsets, first_depth, depth, flat)
    d = cell[3]
    c = wp.vec3i(cell[0], cell[1], cell[2])
    count = wp.int32(0)
    base = wp.int32(0)
    if out_segments.shape[0] > 0:
        base = wp.where(t > 0, offsets[t - 1], 0)
    if leaf_cell(flags, flag_offsets, depth, full_depth, d, c):
        code = cell_case(fields, node_bricks, sources, node_offsets, iso, d, c)
        for side in range(6):
            f = wp.int32(side)
            flip = 1 - table[EXTRACTION_FACE_ORIENTED + f]
            if refined_cell(flags, flag_offsets, depth, full_depth, d, face_across(c, f)):
                prefix = (wp.uint64(flat) << wp.uint64(35)) | (wp.uint64(side) << wp.uint64(32))
                r = binary_search_index_left(borrowed_keys, prefix)
                while r < borrowed_keys.shape[0] and (borrowed_keys[r] >> wp.uint64(32)) == (
                    prefix >> wp.uint64(32)
                ):
                    fine = unpack_cell(cell_offsets, first_depth, depth, borrowed[r] // 6)
                    fc = wp.vec3i(fine[0], fine[1], fine[2])
                    fine_code = cell_case(
                        fields, node_bricks, sources, node_offsets, iso, fine[3], fc
                    )
                    segments = face_segments(
                        table,
                        holds,
                        created,
                        edge_offsets,
                        depth,
                        fine_code,
                        fine[3],
                        fc,
                        borrowed[r] % 6,
                    )
                    count = push_face(segments, flip, count, base, out_segments)
                    r += 1
            else:
                segments = face_segments(table, holds, created, edge_offsets, depth, code, d, c, f)
                count = push_face(segments, flip, count, base, out_segments)
    if out_segments.shape[0] == 0:
        offsets[t] = count


@wp.kernel
def cell_loops(
    offsets: wp.array[wp.int32],
    created: wp.array[wp.uint32],
    ranks: wp.array[wp.int32],
    partners: wp.array[wp.int32],
    segments: wp.array[wp.vec2i],
    out_loops: wp.array[wp.int32],
    out_sizes: wp.array[wp.int32],
    out_counts: wp.array[wp.vec2i],
) -> None:
    # A leaf's segments walked into closed loops as the upstream walks them: start from the last
    # segment, follow the first segment leaving the current vertex (swap-removing it), cross to a
    # vertex's pair when no segment leaves it. Loop ``k``'s vertices land after the previous ones in
    # the cell's ``2 n``-slot range of ``out_loops`` (``n`` its segments), its size in
    # ``out_sizes``; ``out_counts`` gets the cell's triangles and largest loop.
    t = wp.int32(wp.tid())
    begin = wp.where(t > 0, offsets[t - 1], 0)
    remaining = offsets[t] - begin
    written = wp.int32(0)
    loops = wp.int32(0)
    triangles = wp.int32(0)
    largest = wp.int32(0)
    while remaining > 0:
        last = segments[begin + remaining - 1]
        remaining -= 1
        start = last[0]
        current = last[1]
        size = wp.int32(0)
        closed = wp.bool(True)
        while current != start:
            if size >= 2 * (offsets[t] - begin):
                closed = False
                break
            out_loops[2 * begin + written + size] = current
            size += 1
            found = wp.int32(-1)
            for j in range(remaining):
                if found < 0 and segments[begin + j][0] == current:
                    found = j
            if found >= 0:
                current = segments[begin + found][1]
                segments[begin + found] = segments[begin + remaining - 1]
                remaining -= 1
            else:
                current = partners[bit_rank(created, ranks, current)]
                if current < 0:
                    closed = False
                    break
        if closed:
            out_loops[2 * begin + written + size] = start
            size += 1
            out_sizes[begin + loops] = size
            written += size
            loops += 1
            triangles += size - 2
            largest = wp.max(largest, size)
    out_counts[t] = wp.vec2i(triangles, wp.where(largest >= 5, largest * largest, 0))


# PoissonRecon's ``SquareArea``, ``(|v1|^2 |v2|^2 - <v1, v2>^2) / 4``, operation by operation and
# never fused: it cancels on thin triangles, which is why the minimal-area search clamps it at zero,
# and its rounding decides the upstream's near-ties (an x86 build fuses none of it).
_SQUARE_AREA = """
const float x1 = b.c[0] - a.c[0], y1 = b.c[1] - a.c[1], z1 = b.c[2] - a.c[2];
const float x2 = c.c[0] - a.c[0], y2 = c.c[1] - a.c[1], z2 = c.c[2] - a.c[2];
#if defined(__CUDA_ARCH__)
const float dot = __fadd_rn(__fadd_rn(__fmul_rn(x1, x2), __fmul_rn(y1, y2)), __fmul_rn(z1, z2));
const float l1 = __fadd_rn(__fadd_rn(__fmul_rn(x1, x1), __fmul_rn(y1, y1)), __fmul_rn(z1, z1));
const float l2 = __fadd_rn(__fadd_rn(__fmul_rn(x2, x2), __fmul_rn(y2, y2)), __fmul_rn(z2, z2));
return __fdiv_rn(__fsub_rn(__fmul_rn(l1, l2), __fmul_rn(dot, dot)), 4.0f);
#else
float terms[9] = {x1 * x2, y1 * y2, z1 * z2, x1 * x1, y1 * y1, z1 * z1, x2 * x2, y2 * y2, z2 * z2};
const float dot = (terms[0] + terms[1]) + terms[2];
const float l1 = (terms[3] + terms[4]) + terms[5];
const float l2 = (terms[6] + terms[7]) + terms[8];
const float products[2] = {l1 * l2, dot * dot};
return (products[0] - products[1]) / 4.0f;
#endif
"""


@wp.func_native(_SQUARE_AREA)
def square_area(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32: ...


@wp.func
def vertex_of(created: wp.array[wp.uint32], ranks: wp.array[wp.int32], owner: wp.int32) -> wp.int32:
    # The output number of the vertex created on edge slot ``owner``.
    return bit_rank(created, ranks, owner)


@wp.kernel
def triangulate_loops(
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.vec2i],
    created: wp.array[wp.uint32],
    ranks: wp.array[wp.int32],
    positions: wp.array[wp.vec3],
    loops: wp.array[wp.int32],
    sizes: wp.array[wp.int32],
    scratch: wp.array[wp.float32],
    choices: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # Each loop of a leaf triangulated as ``MinimalAreaTriangulation`` does, every triangle written
    # reversed (the upstream winding): a triangle as is, a quad split by comparing the second
    # triangle of each split (the upstream's ``area[i] =`` assigns rather than accumulates), a
    # longer loop by the minimal-area interval search with the first best split kept.
    t = wp.int32(wp.tid())
    begin = wp.where(t > 0, offsets[t - 1], 0)
    n_segments = offsets[t] - begin
    face = wp.where(t > 0, counts[t - 1][0], 0)
    work = wp.where(t > 0, counts[t - 1][1], 0)
    written = wp.int32(0)
    k = wp.int32(0)
    while written < 2 * n_segments and k < n_segments and sizes[begin + k] > 0:
        n = sizes[begin + k]
        first = 2 * begin + written
        if n == 3:
            out_faces[3 * face] = vertex_of(created, ranks, loops[first + 2])
            out_faces[3 * face + 1] = vertex_of(created, ranks, loops[first + 1])
            out_faces[3 * face + 2] = vertex_of(created, ranks, loops[first])
            face += 1
        elif n == 4:
            p0 = positions[vertex_of(created, ranks, loops[first])]
            p1 = positions[vertex_of(created, ranks, loops[first + 1])]
            p2 = positions[vertex_of(created, ranks, loops[first + 2])]
            p3 = positions[vertex_of(created, ranks, loops[first + 3])]
            if square_area(p2, p3, p0) > square_area(p3, p1, p2):
                out_faces[3 * face] = vertex_of(created, ranks, loops[first + 3])
                out_faces[3 * face + 1] = vertex_of(created, ranks, loops[first + 1])
                out_faces[3 * face + 2] = vertex_of(created, ranks, loops[first])
                out_faces[3 * face + 3] = vertex_of(created, ranks, loops[first + 2])
                out_faces[3 * face + 4] = vertex_of(created, ranks, loops[first + 1])
                out_faces[3 * face + 5] = vertex_of(created, ranks, loops[first + 3])
            else:
                out_faces[3 * face] = vertex_of(created, ranks, loops[first + 2])
                out_faces[3 * face + 1] = vertex_of(created, ranks, loops[first + 1])
                out_faces[3 * face + 2] = vertex_of(created, ranks, loops[first])
                out_faces[3 * face + 3] = vertex_of(created, ranks, loops[first])
                out_faces[3 * face + 4] = vertex_of(created, ranks, loops[first + 3])
                out_faces[3 * face + 5] = vertex_of(created, ranks, loops[first + 2])
            face += 2
        elif n >= 5:
            # Interval (i, j) of the loop's vertices 1..n (n standing for vertex 0) at
            # ``work + (i - 1) n + j - 1``: its best area and split.
            for span in range(2, n):
                for i in range(1, n - span + 1):
                    j = i + span
                    best = wp.float32(3.402823e38)
                    split = wp.int32(-1)
                    for r in range(i + 1, j):
                        area = square_area(
                            positions[vertex_of(created, ranks, loops[first + i % n])],
                            positions[vertex_of(created, ranks, loops[first + r % n])],
                            positions[vertex_of(created, ranks, loops[first + j % n])],
                        )
                        total = wp.sqrt(wp.max(area, 0.0))
                        if r - i >= 2:
                            total += scratch[work + (i - 1) * n + r - 1]
                        if j - r >= 2:
                            total += scratch[work + (r - 1) * n + j - 1]
                        if total < best:
                            best = total
                            split = r
                    scratch[work + (i - 1) * n + j - 1] = best
                    choices[work + (i - 1) * n + j - 1] = split
            # Emit the chosen splits, outermost first, with an explicit stack in ``choices``'
            # diagonal (unused by the search).
            stack = wp.int32(0)
            choices[work + stack * n + stack] = (1 << 16) | n
            stack = wp.int32(1)
            while stack > 0:
                stack -= 1
                packed = choices[work + stack * n + stack]
                i = packed >> 16
                j = packed & 0xFFFF
                if j - i >= 2:
                    r = choices[work + (i - 1) * n + j - 1]
                    out_faces[3 * face] = vertex_of(created, ranks, loops[first + j % n])
                    out_faces[3 * face + 1] = vertex_of(created, ranks, loops[first + r % n])
                    out_faces[3 * face + 2] = vertex_of(created, ranks, loops[first + i % n])
                    face += 1
                    choices[work + stack * n + stack] = (i << 16) | r
                    stack += 1
                    choices[work + stack * n + stack] = (r << 16) | j
                    stack += 1
        written += n
        k += 1


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module.

    See ``kernels/array.py::declare_map_signatures``. ``brick_number`` numbers the finest depth's
    stored bricks; the brick table is a single entry up to depth 2.
    """
    dense, single = map_probe, map_probe_single
    declare_map_signatures(
        [
            (brick_number, (dense(wp.int32), dense(wp.int32)), wp.int32),
            (brick_number, (single(wp.int32), single(wp.int32)), wp.int32),
        ]
    )


_declare_map_kernels()
