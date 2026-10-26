"""
Screened-Poisson kernels that solve PoissonRecon's own octree system, on dense per-depth lattices.

Backs [`screened_poisson`][ordito.reconstruction.screened_poisson]'s ``method="octree"``. The
Python-scope orchestration lives in ``reconstruction._screened_poisson_octree``. Everything is in
the unit cube the cloud is mapped into (``scale`` pads it); depth ``d`` splits it into ``2**d``
cells per axis.

**Two bases per depth.** The indicator is a sum over depths of degree-1 (trilinear hat) functions,
one per lattice node ``0..2**d`` with free (Neumann) walls; the normal field is a sum of
cell-centred quadratic B-splines, one per cell, reflected oddly at the walls (they vanish there).
Lattices are dense: depth ``d``'s nodes are ``(2**d + 1)**3`` and its cells ``8**d``, every depth's
cells packed in one buffer at ``cell_offsets[d]``.

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
iterations``, with the same values.

**Extraction** runs marching cubes on each depth's leaf cells (active cells with no active
children) from that depth's field, placing every vertex with the upstream non-linear fit: the
Hermite cubic through the edge's ends whose end slopes are rescaled to remove its cubic term, the
slope at a corner split into the depth's own correction (a central difference) and the coarser
field's (its own lattice's slope). The upstream extraction also stitches a coarse leaf's faces to
its finer neighbours' vertices and triangulates its polygons by minimal area; this one does
neither, which moves a triangle, never a vertex, and can leave a crack along a depth change.
"""

import warp as wp

from ordito.kernels.levelset import (
    MC_CORNERS,
    mc_case_corner,
    mc_case_final,
    mc_triangle_edge,
    mc_values_cross,
)
from ordito.kernels.reconstruction import (
    poisson_grid_index,
    poisson_restrict_node,
    poisson_sample_grid,
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
def octree_sample_slots(keys: wp.array[wp.int32], out_slots: wp.array[wp.int32]) -> None:
    # The finest lattice's cell-to-sample map (cells without a sample keep -1).
    s = wp.int32(wp.tid())
    out_slots[keys[s]] = s


@wp.kernel
def octree_child_aggregates(
    d: wp.int32,
    depth: wp.int32,
    slots: wp.array[wp.int32],
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
            s = slots[index]
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
    sums: GroupSumPass,
    out_normals: wp.array[wp.vec3],
):
    # ``amount`` splatted into the 27 depth ``d`` B-splines around the depth ``d`` ancestor of the
    # finest cell ``cell``. A slot aliased past the finest depth would need a node the solve does
    # not have; a cloud padded by ``scale >= 1`` never reaches one there.
    node = wp.vec3i(cell[0] >> (depth - d), cell[1] >> (depth - d), cell[2] >> (depth - d))
    scale = wp.float32(1 << d)
    vx = quadratic_bspline_values(p[0] * scale - wp.float32(node[0]))
    vy = quadratic_bspline_values(p[1] * scale - wp.float32(node[1]))
    vz = quadratic_bspline_values(p[2] * scale - wp.float32(node[2]))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                target = splat_target(d, node + wp.vec3i(a - 1, b - 1, c - 1))
                if target[0] >= 0 and target[0] <= depth:
                    scatter_add_vector(
                        out_normals,
                        sums,
                        cell_offsets[target[0]] + target[1],
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
            v = normals[cell_offsets[d + 1] + cell_flat(2 * n, cc)]
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


@wp.kernel
def octree_divergence_constraints(
    d: wp.int32,
    data_depth: wp.int32,
    data: wp.array[wp.vec3],
    data_offset: wp.int32,
    out_b: wp.array[wp.float32],
) -> None:
    # Depth ``d`` node's ``-int grad(hat) . V`` for the depth ``data_depth`` (``d`` or ``d - 1``)
    # normal coefficients ``data``: four B-splines per axis overlap a hat at either depth.
    i, j, k = wp.tid()
    res = (1 << d) + 1
    m = 1 << data_depth
    shift = d - data_depth
    node = wp.vec3i(i, j, k)
    first = wp.vec3i((i >> shift) - 2, (j >> shift) - 2, (k >> shift) - 2)
    value_x = wp.vec4d()
    slope_x = wp.vec4d()
    value_y = wp.vec4d()
    slope_y = wp.vec4d()
    value_z = wp.vec4d()
    slope_z = wp.vec4d()
    for q in range(4):
        ix = hat_bspline_integrals(d, node[0], data_depth, first[0] + q)
        iy = hat_bspline_integrals(d, node[1], data_depth, first[1] + q)
        iz = hat_bspline_integrals(d, node[2], data_depth, first[2] + q)
        value_x[q] = ix[0]
        slope_x[q] = ix[1]
        value_y[q] = iy[0]
        slope_y[q] = iy[1]
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
                            v = data[data_offset + cell_flat(m, wp.vec3i(ka, kb, kc))]
                            total += (
                                wp.float64(v[0]) * slope_x[a] * value_y[b] * value_z[c]
                                + wp.float64(v[1]) * value_x[a] * slope_y[b] * value_z[c]
                                + wp.float64(v[2]) * value_x[a] * value_y[b] * slope_z[c]
                            )
    index = poisson_grid_index(i, j, k, res)
    out_b[index] = out_b[index] - wp.float32(total)


@wp.kernel
def octree_restrict_add(
    fine: wp.array[wp.float32], res_f: wp.int32, res_c: wp.int32, out_coarse: wp.array[wp.float32]
) -> None:
    # A coarse hat is a sum of fine ones, so finer constraints restrict to it exactly.
    i, j, k = wp.tid()
    index = poisson_grid_index(i, j, k, res_c)
    out_coarse[index] = out_coarse[index] + poisson_restrict_node(fine, res_f, i, j, k)


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
        s = slots[cell_flat(n, c)]
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


@wp.func
def octree_row(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    alpha: wp.float32,
    node: wp.vec3i,
    field: wp.array[wp.float32],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    counts: wp.array[wp.float32],
    position_sums: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    sample_counts: wp.array[wp.int32],
    sample_positions: wp.array[wp.vec3],
) -> wp.vec2:
    # ``(row . field, diagonal)`` of depth ``d``'s operator at ``node``: the hat stiffness (27
    # neighbours) plus the screening of the eight cells around the node.
    res = (1 << d) + 1
    h = 1.0 / wp.float32(1 << d)
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
                            total += entry * field[poisson_grid_index(ia, ib, ic, res)]
                            if a == 1 and b == 1 and c == 1:
                                diagonal = entry
    scale = alpha * wp.float32(1 << d)
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
            sample_counts,
            sample_positions,
        )
        if sample[3] > 0.0:
            p = wp.vec3(sample[0], sample[1], sample[2])
            phi = hat_value(d, node, p)
            interpolated = wp.float32(0.0)
            for q in range(8):
                other = cell + wp.vec3i(q >> 2, (q >> 1) & 1, q & 1)
                interpolated += (
                    hat_value(d, other, p)
                    * field[poisson_grid_index(other[0], other[1], other[2], res)]
                )
            weight = scale * sample[3]
            total += weight * phi * interpolated
            diagonal += weight * phi * phi
    return wp.vec2(total, diagonal)


@wp.kernel
def octree_screening_constraints(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    alpha: wp.float32,
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    counts: wp.array[wp.float32],
    position_sums: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    sample_counts: wp.array[wp.int32],
    sample_positions: wp.array[wp.vec3],
    out_b: wp.array[wp.float32],
) -> None:
    # The screening's right-hand side: target value 1/2 at every screening sample.
    i, j, k = wp.tid()
    node = wp.vec3i(i, j, k)
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
            sample_counts,
            sample_positions,
        )
        if sample[3] > 0.0:
            total += (
                scale * sample[3] * hat_value(d, node, wp.vec3(sample[0], sample[1], sample[2]))
            )
    index = poisson_grid_index(i, j, k, (1 << d) + 1)
    out_b[index] = out_b[index] + total


@wp.kernel
def octree_gauss_seidel_wave(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    alpha: wp.float32,
    wave: wp.int32,
    first_sweep: wp.int32,
    colour: wp.int32,
    b: wp.array[wp.float32],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    counts: wp.array[wp.float32],
    position_sums: wp.array[wp.vec3],
    cell_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    sample_counts: wp.array[wp.int32],
    sample_positions: wp.array[wp.vec3],
    field: wp.array[wp.float32],
) -> None:
    # One colour of every ``(sweep, slice)`` on wavefront ``wave = 2 sweep + (res - 1 - z)``: a
    # Gauss-Seidel update of each active node of that colour on that slice. ``field`` starts at the
    # coarser field prolonged, so its updates are the upstream correction's (which starts at zero
    # against ``b - A prolonged``) with the coarser field added.
    s, a, c = wp.tid()
    res = (1 << d) + 1
    sweep = first_sweep + s
    z = res - 1 - (wave - 2 * sweep)
    x = 2 * a + (colour & 1)
    y = 2 * c + (colour >> 1)
    if x >= res or y >= res:
        return
    node = wp.vec3i(x, y, z)
    if not node_active(flags, flag_offsets, full_depth, d, node):
        return
    row = octree_row(
        d,
        depth,
        full_depth,
        alpha,
        node,
        field,
        flags,
        flag_offsets,
        counts,
        position_sums,
        cell_offsets,
        slots,
        sample_counts,
        sample_positions,
    )
    index = poisson_grid_index(x, y, z, res)
    field[index] = field[index] + (b[index] - row[0]) / row[1]


@wp.kernel
def octree_iso_terms(
    sample_counts: wp.array[wp.int32],
    sample_positions: wp.array[wp.vec3],
    field: wp.array[wp.float32],
    res: wp.int32,
    out_terms: wp.array[wp.float32],
) -> None:
    # A sample's count times the field at its centroid; their sum over the point count is the
    # iso-value.
    s = wp.int32(wp.tid())
    count = wp.float32(sample_counts[s])
    g = sample_positions[s] / count * wp.float32(res - 1)
    out_terms[s] = count * poisson_sample_grid(field, res, g[0], g[1], g[2])


# ---------------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------------


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
    n = 1 << d
    if c[0] < 0 or c[1] < 0 or c[2] < 0 or c[0] >= n or c[1] >= n or c[2] >= n:
        return False
    if not node_active(flags, flag_offsets, full_depth, d, c):
        return False
    if d == depth:
        return True
    return not children_active(flags, flag_offsets, d, c)


@wp.func
def leaf_edge_crosses(
    field: wp.array[wp.float32],
    iso: wp.float32,
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    depth: wp.int32,
    full_depth: wp.int32,
    d: wp.int32,
    node: wp.vec3i,
    axis: wp.int32,
) -> wp.int32:
    # 1 when the edge from ``node`` along ``axis`` crosses ``iso`` and bounds a leaf cell.
    res = (1 << d) + 1
    end = node + wp.vec3i(
        wp.where(axis == 0, 1, 0), wp.where(axis == 1, 1, 0), wp.where(axis == 2, 1, 0)
    )
    if end[0] >= res or end[1] >= res or end[2] >= res:
        return 0
    crossing = mc_values_cross(
        field[poisson_grid_index(node[0], node[1], node[2], res)],
        field[poisson_grid_index(end[0], end[1], end[2], res)],
        iso,
    )
    if crossing == 0:
        return 0
    u = (axis + 1) % 3
    v = (axis + 2) % 3
    for q in range(4):
        cell = wp.vec3i(node[0], node[1], node[2])
        cell[u] = cell[u] - (q & 1)
        cell[v] = cell[v] - (q >> 1)
        if leaf_cell(flags, flag_offsets, depth, full_depth, d, cell):
            return 1
    return 0


@wp.func
def leaf_cell_triangles(
    field: wp.array[wp.float32],
    iso: wp.float32,
    table: wp.array[wp.int32],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    depth: wp.int32,
    full_depth: wp.int32,
    d: wp.int32,
    node: wp.vec3i,
) -> wp.vec2i:
    # ``(first table slot, triangle count)`` of the leaf cell at ``node``; nothing for a non-leaf.
    res = (1 << d) + 1
    if not leaf_cell(flags, flag_offsets, depth, full_depth, d, node):
        return wp.vec2i(0, 0)
    code = wp.int32(0)
    for c in range(8):
        code = mc_case_corner(
            code,
            field[
                poisson_grid_index(
                    node[0] + wp.static(MC_CORNERS[c][0]),
                    node[1] + wp.static(MC_CORNERS[c][1]),
                    node[2] + wp.static(MC_CORNERS[c][2]),
                    res,
                )
            ],
            iso,
            wp.static(1 << c),
        )
    code = mc_case_final(code)
    start = table[code]
    return wp.vec2i(start, (table[code + 1] - start) // 3)


@wp.kernel
def octree_leaf_counts(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    field: wp.array[wp.float32],
    iso: wp.float32,
    table: wp.array[wp.int32],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    out_counts: wp.array[wp.vec2i],
) -> None:
    # Per node: its leaf-bounding crossing edges (vertices) and its leaf cell's triangles.
    i, j, k = wp.tid()
    node = wp.vec3i(i, j, k)
    vertices = wp.int32(0)
    for axis in range(3):
        vertices += leaf_edge_crosses(
            field, iso, flags, flag_offsets, depth, full_depth, d, node, axis
        )
    cell = leaf_cell_triangles(field, iso, table, flags, flag_offsets, depth, full_depth, d, node)
    out_counts[poisson_grid_index(i, j, k, (1 << d) + 1)] = wp.vec2i(vertices, cell[1])


@wp.func
def split_slope(
    field: wp.array[wp.float32],
    prolonged: wp.array[wp.float32],
    coarse: wp.array[wp.float32],
    d: wp.int32,
    node: wp.vec3i,
    axis: wp.int32,
) -> wp.float32:
    # The field's slope along ``axis`` at ``node``, times the cell width, as the upstream evaluator
    # forms it: the depth's own correction's (``field - prolonged``) averaged one-sided slopes plus
    # the coarser field's slope on its own lattice (averaged at its nodes, exact between them),
    # interpolated linearly across the other two axes.
    res = (1 << d) + 1
    res_c = (1 << (d - 1)) + 1
    up = wp.vec3i(node[0], node[1], node[2])
    down = wp.vec3i(node[0], node[1], node[2])
    up[axis] = wp.min(node[axis] + 1, res - 1)
    down[axis] = wp.max(node[axis] - 1, 0)
    upper = poisson_grid_index(up[0], up[1], up[2], res)
    lower = poisson_grid_index(down[0], down[1], down[2], res)
    slope = 0.5 * ((field[upper] - prolonged[upper]) - (field[lower] - prolonged[lower]))
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
                    coarse[poisson_grid_index(hi[0], hi[1], hi[2], res_c)]
                    - coarse[poisson_grid_index(lo[0], lo[1], lo[2], res_c)]
                )
            )
    return slope + coarse_slope


@wp.func
def fitted_edge_root(
    here: wp.float32,
    there: wp.float32,
    slope_here: wp.float32,
    slope_there: wp.float32,
    iso: wp.float32,
) -> wp.float32:
    # The upstream non-linear fit: the end slopes rescaled so the Hermite cubic loses its cubic
    # term, the mean of the quadratic's roots in ``[0, 1]``, else the linear root.
    x0 = wp.float64(here)
    x1 = wp.float64(there)
    scale = (x1 - x0) / ((wp.float64(slope_here) + wp.float64(slope_there)) * wp.float64(0.5))
    d0 = wp.float64(slope_here) * scale
    d1 = wp.float64(slope_there) * scale
    a = wp.float64(3.0) * (x1 - x0) - d1 - wp.float64(2.0) * d0
    c = x0 - wp.float64(iso)
    roots = wp.float64(0.0)
    found = wp.int32(0)
    if a == wp.float64(0.0):
        if d0 != wp.float64(0.0):
            t = -c / d0
            if t >= wp.float64(0.0) and t <= wp.float64(1.0):
                roots = t
                found = 1
    else:
        disc = d0 * d0 - wp.float64(4.0) * a * c
        if disc >= wp.float64(0.0):
            root = wp.sqrt(disc)
            for sign in range(2):
                t = (-d0 + wp.where(sign == 0, root, -root)) / (wp.float64(2.0) * a)
                if t >= wp.float64(0.0) and t <= wp.float64(1.0):
                    roots += t
                    found += 1
    if found > 0:
        return wp.float32(roots / wp.float64(found))
    return (iso - here) / (there - here)


@wp.kernel
def octree_leaf_emit(
    d: wp.int32,
    depth: wp.int32,
    full_depth: wp.int32,
    field: wp.array[wp.float32],
    prolonged: wp.array[wp.float32],
    coarse: wp.array[wp.float32],
    iso: wp.float32,
    lower: wp.vec3,
    size: wp.float32,
    margin: wp.float32,
    table: wp.array[wp.int32],
    flags: wp.array[wp.int32],
    flag_offsets: wp.array[wp.int32],
    offsets: wp.array[wp.vec2i],
    base: wp.vec2i,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # ``marching_cubes_emit`` restricted to leaf cells, writing after ``base`` (this depth's first
    # vertex and face), each vertex placed by ``fitted_edge_root`` and kept ``margin`` of the edge
    # off its ends.
    i, j, k = wp.tid()
    res = (1 << d) + 1
    node = wp.vec3i(i, j, k)
    ends = offsets[poisson_grid_index(i, j, k, res)]
    crossing = wp.vec3i(
        leaf_edge_crosses(field, iso, flags, flag_offsets, depth, full_depth, d, node, 0),
        leaf_edge_crosses(field, iso, flags, flag_offsets, depth, full_depth, d, node, 1),
        leaf_edge_crosses(field, iso, flags, flag_offsets, depth, full_depth, d, node, 2),
    )
    slot = ends[0] - (crossing[0] + crossing[1] + crossing[2])
    width = size / wp.float32(1 << d)
    for axis in range(3):
        if crossing[axis] != 0:
            end = wp.vec3i(i, j, k)
            end[axis] = end[axis] + 1
            here = field[poisson_grid_index(i, j, k, res)]
            there = field[poisson_grid_index(end[0], end[1], end[2], res)]
            t = fitted_edge_root(
                here,
                there,
                split_slope(field, prolonged, coarse, d, node, axis),
                split_slope(field, prolonged, coarse, d, end, axis),
                iso,
            )
            t = wp.clamp(t, margin, 1.0 - margin)
            position = wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k))
            position[axis] = position[axis] + t
            out_vertices[base[0] + slot] = lower + position * width
            slot += 1

    cell = leaf_cell_triangles(field, iso, table, flags, flag_offsets, depth, full_depth, d, node)
    first_face = ends[1] - cell[1]
    for tri in range(cell[1]):
        for s in range(3):
            edge = mc_triangle_edge(table, cell[0] + 3 * tri + s)
            owner = node + wp.vec3i(edge[0], edge[1], edge[2])
            axis = edge[3]
            vertex = offsets[poisson_grid_index(owner[0], owner[1], owner[2], res)][0] - 1
            for higher in range(axis + 1, 3):
                vertex -= leaf_edge_crosses(
                    field, iso, flags, flag_offsets, depth, full_depth, d, owner, higher
                )
            out_faces[3 * (base[1] + first_face + tri) + s] = base[0] + vertex
