"""
Narrow-band screened-Poisson kernels: the dense node lattice stored only near the surface.

Backs [`screened_poisson`][ordito.reconstruction.screened_poisson]'s ``method="adaptive"``. The
Python-scope orchestration lives in ``reconstruction._screened_poisson_band``.

**Layout.** A level's lattice is the dense ``res**3`` node grid of the dense backend, cut into
``8**3``-node *bricks*; only the bricks near the surface are stored. A dense ``nbk**3`` brick map
(``nbk = ceil(res / 8)``) holds each brick's slot or ``-1``, and a band node's storage index is
``slot * 512 + local``, ``local`` row-major in the brick with ``k`` fastest. Nodes of the last
brick row past ``res - 1`` exist in storage only: they are inert (zero, never a neighbour).

**Hierarchy.** The finest level's splat is the only data. Every coarser level's right-hand side and
screening weights are its ``P^T`` restriction, and its operator is the dense multigrid's own coarse
operator ``2 ** (L - l) * L_l + screen * W_l`` (``kernels/reconstruction``'s V-cycle note), so a
coarse solution approximates the *fine* solution and its prolongation is a valid Dirichlet value at
a finer band's boundary. A coarse level's band contains its finer level's bricks and one more brick
around them, so every ghost and every restriction target exists; misses are counted, not ignored.

**What is solved per band level** is the band's own equations with every out-of-band neighbour fixed
at the prolonged coarse solution: ``A_band x = b - A_ghost g``, started from ``g`` itself.
"""

import warp as wp

from ordito.kernels.algorithms.conjugate_gradient import (
    cg_publish_round_dots,
    cg_round_terms,
    cg_widen,
)
from ordito.kernels.array import trilinear_cell, trilinear_corner, trilinear_weight
from ordito.kernels.levelset import MC_CORNERS, mc_edge_vertex, mc_triangle_edge, mc_values_cross
from ordito.kernels.reconstruction import (
    normalized_field_component,
    poisson_grid_index,
    poisson_prolong_node,
    screened_laplacian_scaled_inverse_diagonal,
)
from ordito.kernels.reduce import block_sum

wp.set_module_options({"enable_backward": False})

# Nodes along a brick's edge, as a shift, and nodes in a brick.
BRICK_SHIFT = 3
BRICK_NODES = 512


# ---------------------------------------------------------------------------------------------
# Brick topology
# ---------------------------------------------------------------------------------------------


@wp.func
def brick_index(i: wp.int32, j: wp.int32, k: wp.int32, nbk: wp.int32) -> wp.int32:
    # Flat index into the ``nbk**3`` brick map of the brick holding node ``(i, j, k)``.
    return ((i >> BRICK_SHIFT) * nbk + (j >> BRICK_SHIFT)) * nbk + (k >> BRICK_SHIFT)


@wp.func
def band_node(
    brick_map: wp.array[wp.int32], nbk: wp.int32, i: wp.int32, j: wp.int32, k: wp.int32
) -> wp.int32:
    # Band storage index of node ``(i, j, k)`` (all three non-negative), or ``-1`` when its brick
    # is not in the band.
    slot = brick_map[brick_index(i, j, k, nbk)]
    if slot < 0:
        return wp.int32(-1)
    return slot * BRICK_NODES + ((((i & 7) << 3) + (j & 7)) << 3) + (k & 7)


@wp.func
def band_coordinates(brick_coords: wp.array[wp.vec3i], n: wp.int32) -> wp.vec3i:
    # Lattice node of band storage index ``n``: the inverse of ``band_node``.
    brick = brick_coords[n >> 9]
    local = n & 511
    return wp.vec3i(
        (brick[0] << BRICK_SHIFT) + (local >> 6),
        (brick[1] << BRICK_SHIFT) + ((local >> 3) & 7),
        (brick[2] << BRICK_SHIFT) + (local & 7),
    )


@wp.func
def node_live(g: wp.vec3i, res: wp.int32) -> wp.bool:
    # Whether a band slot's node is on the lattice (the last brick row overhangs ``res - 1``).
    return g[0] < res and g[1] < res and g[2] < res


@wp.kernel
def mark_sample_bricks(
    points: wp.array[wp.vec3],
    lower: wp.vec3,
    inv_cell: wp.float32,
    res: wp.int32,
    nbk: wp.int32,
    out_flags: wp.array[wp.int32],
) -> None:
    # Every brick a sample's splat writes (its cell's eight nodes) or its right-hand side reads (one
    # node further, the divergence's central difference), so the band holds the whole data.
    s = wp.int32(wp.tid())
    g = (points[s] - lower) * inv_cell
    lo = wp.vec3i(
        wp.clamp(wp.int32(wp.floor(g[0])) - 1, 0, res - 1),
        wp.clamp(wp.int32(wp.floor(g[1])) - 1, 0, res - 1),
        wp.clamp(wp.int32(wp.floor(g[2])) - 1, 0, res - 1),
    )
    hi = wp.vec3i(
        wp.clamp(wp.int32(wp.floor(g[0])) + 2, 0, res - 1),
        wp.clamp(wp.int32(wp.floor(g[1])) + 2, 0, res - 1),
        wp.clamp(wp.int32(wp.floor(g[2])) + 2, 0, res - 1),
    )
    for bi in range(lo[0] >> BRICK_SHIFT, (hi[0] >> BRICK_SHIFT) + 1):
        for bj in range(lo[1] >> BRICK_SHIFT, (hi[1] >> BRICK_SHIFT) + 1):
            for bk in range(lo[2] >> BRICK_SHIFT, (hi[2] >> BRICK_SHIFT) + 1):
                out_flags[(bi * nbk + bj) * nbk + bk] = 1


@wp.kernel
def mark_crossing_bricks(
    coarse: wp.array[wp.float32],
    res_c: wp.int32,
    iso: wp.float32,
    scale: wp.int32,
    res: wp.int32,
    nbk: wp.int32,
    out_flags: wp.array[wp.int32],
) -> None:
    # A dense coarse cell whose corners straddle ``iso`` marks every fine brick its footprint
    # covers (``scale`` fine cells per coarse cell): where the coarse surface runs, including where
    # it closes a hole the samples leave open, far from any sample brick.
    ci, cj, ck = wp.tid()
    lo = wp.float32(wp.inf)
    hi = wp.float32(-wp.inf)
    for c in range(8):
        value = coarse[
            poisson_grid_index(
                ci + wp.static(MC_CORNERS[c][0]),
                cj + wp.static(MC_CORNERS[c][1]),
                ck + wp.static(MC_CORNERS[c][2]),
                res_c,
            )
        ]
        lo = wp.min(lo, value)
        hi = wp.max(hi, value)
    if lo < iso and hi >= iso:
        last = res - 1
        for bi in range(
            (ci * scale) >> BRICK_SHIFT, (wp.min((ci + 1) * scale, last) >> BRICK_SHIFT) + 1
        ):
            for bj in range(
                (cj * scale) >> BRICK_SHIFT, (wp.min((cj + 1) * scale, last) >> BRICK_SHIFT) + 1
            ):
                for bk in range(
                    (ck * scale) >> BRICK_SHIFT, (wp.min((ck + 1) * scale, last) >> BRICK_SHIFT) + 1
                ):
                    out_flags[(bi * nbk + bj) * nbk + bk] = 1


@wp.kernel
def dilate_bricks(flags: wp.array[wp.int32], nbk: wp.int32, out_flags: wp.array[wp.int32]) -> None:
    # One 26-neighbourhood dilation of a brick set.
    i, j, k = wp.tid()
    hit = wp.int32(0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                a = i + di
                b = j + dj
                c = k + dk
                if a >= 0 and a < nbk and b >= 0 and b < nbk and c >= 0 and c < nbk:
                    if flags[(a * nbk + b) * nbk + c] != 0:
                        hit = 1
    out_flags[(i * nbk + j) * nbk + k] = hit


@wp.kernel
def project_bricks(
    flags: wp.array[wp.int32], nbk: wp.int32, nbk_c: wp.int32, out_flags: wp.array[wp.int32]
) -> None:
    # A fine brick marks the coarse brick covering it (a coarse brick covers two fine ones a side);
    # ``out_flags`` is zeroed by the caller.
    i, j, k = wp.tid()
    if flags[(i * nbk + j) * nbk + k] != 0:
        out_flags[((i >> 1) * nbk_c + (j >> 1)) * nbk_c + (k >> 1)] = 1


@wp.kernel
def compact_bricks(
    ranks: wp.array[wp.int32],
    nbk: wp.int32,
    out_brick_map: wp.array[wp.int32],
    out_brick_coords: wp.array[wp.vec3i],
) -> None:
    # ``ranks`` is the 0/1 brick flags scanned inclusively in place, so a brick is set where its
    # rank rises: its slot is that rank less one, and every unset brick maps to ``-1``.
    b = wp.int32(wp.tid())
    rank = ranks[b]
    previous = wp.int32(0)
    if b > 0:
        previous = ranks[b - 1]
    if rank == previous:
        out_brick_map[b] = -1
        return
    out_brick_map[b] = rank - 1
    out_brick_coords[rank - 1] = wp.vec3i(b // (nbk * nbk), (b // nbk) % nbk, b % nbk)


# ---------------------------------------------------------------------------------------------
# Data: the finest splat and right-hand side, and the restriction down the levels
# ---------------------------------------------------------------------------------------------


@wp.kernel
def band_splat(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    lower: wp.vec3,
    inv_cell: wp.float32,
    res: wp.int32,
    confidence: wp.int32,
    brick_map: wp.array[wp.int32],
    nbk: wp.int32,
    out_vx: wp.array[wp.float32],
    out_vy: wp.array[wp.float32],
    out_vz: wp.array[wp.float32],
    out_w: wp.array[wp.float32],
) -> None:
    # ``kernels/reconstruction.splat_normals`` into the band's storage: the same unit direction,
    # confidence weight and trilinear corners. Every corner is in the band (``mark_sample_bricks``).
    s = wp.int32(wp.tid())
    n = normals[s]
    length = wp.length(n)
    weight = wp.float32(1.0)
    if confidence != 0:
        weight = length
    if length > 0.0:
        n = n / length
    g = (points[s] - lower) * inv_cell
    base, next_corner, fractions = trilinear_cell(g, wp.vec3i(res, res, res))
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                w = trilinear_weight(fractions, di, dj, dk) * weight
                corner = trilinear_corner(base, next_corner, wp.vec3i(di, dj, dk))
                idx = band_node(brick_map, nbk, corner[0], corner[1], corner[2])
                wp.atomic_add(out_vx, idx, w * n[0])
                wp.atomic_add(out_vy, idx, w * n[1])
                wp.atomic_add(out_vz, idx, w * n[2])
                wp.atomic_add(out_w, idx, w)


@wp.func
def band_field_component(
    field: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    brick_map: wp.array[wp.int32],
    nbk: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
) -> wp.float32:
    # ``normalized_field_component`` at a band node, zero off the band (no sample reaches there).
    idx = band_node(brick_map, nbk, i, j, k)
    if idx < 0:
        return wp.float32(0.0)
    return normalized_field_component(field, weights, idx)


@wp.kernel
def band_rhs(
    brick_coords: wp.array[wp.vec3i],
    brick_map: wp.array[wp.int32],
    nbk: wp.int32,
    res: wp.int32,
    vx: wp.array[wp.float32],
    vy: wp.array[wp.float32],
    vz: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    out_b: wp.array[wp.float32],
) -> None:
    # ``poisson_level_setup``'s right-hand side ``-div V`` on the band: the same normalized central
    # differences, one-sided on the lattice boundary.
    n = wp.int32(wp.tid())
    g = band_coordinates(brick_coords, n)
    if not node_live(g, res):
        out_b[n] = 0.0
        return
    i = g[0]
    j = g[1]
    k = g[2]
    ip = wp.min(i + 1, res - 1)
    im = wp.max(i - 1, 0)
    jp = wp.min(j + 1, res - 1)
    jm = wp.max(j - 1, 0)
    kp = wp.min(k + 1, res - 1)
    km = wp.max(k - 1, 0)
    dx = (
        band_field_component(vx, weights, brick_map, nbk, ip, j, k)
        - band_field_component(vx, weights, brick_map, nbk, im, j, k)
    ) / wp.float32(ip - im)
    dy = (
        band_field_component(vy, weights, brick_map, nbk, i, jp, k)
        - band_field_component(vy, weights, brick_map, nbk, i, jm, k)
    ) / wp.float32(jp - jm)
    dz = (
        band_field_component(vz, weights, brick_map, nbk, i, j, kp)
        - band_field_component(vz, weights, brick_map, nbk, i, j, km)
    ) / wp.float32(kp - km)
    out_b[n] = -(dx + dy + dz)


@wp.func
def restrict_axis(i: wp.int32, t: wp.int32) -> wp.vec2i:
    # The ``t``-th coarse node (``t`` 0 or 1) fine node ``i`` restricts into and its weight in
    # halves: an even node sits on coarse node ``i / 2`` at weight 1 (``t = 1`` then has weight 0),
    # an odd one between two at a half each -- the transpose of ``poisson_prolong_node``.
    if (i & 1) == 0:
        return wp.vec2i(i >> 1, wp.where(t == 0, 2, 0))
    return wp.vec2i((i >> 1) + t, 1)


@wp.kernel
def restrict_band(
    brick_coords: wp.array[wp.vec3i],
    res: wp.int32,
    b: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    coarse_map: wp.array[wp.int32],
    nbk_c: wp.int32,
    res_c: wp.int32,
    out_b: wp.array[wp.float32],
    out_weights: wp.array[wp.float32],
    out_missing: wp.array[wp.int32],
) -> None:
    # ``P^T`` of a band level's ``b`` and ``W`` into the next coarser level, scattered: into its
    # band when ``coarse_map`` is given, into a dense ``res_c**3`` grid when it is ``None``. Both
    # outputs are zeroed by the caller. A target missing from the coarse band is counted in
    # ``out_missing[0]`` and dropped -- the nesting makes it impossible, and a test pins the count.
    n = wp.int32(wp.tid())
    g = band_coordinates(brick_coords, n)
    if not node_live(g, res):
        return
    bv = b[n]
    wv = weights[n]
    if bv == 0.0 and wv == 0.0:
        return
    for ti in range(2):
        a = restrict_axis(g[0], ti)
        for tj in range(2):
            c = restrict_axis(g[1], tj)
            for tk in range(2):
                e = restrict_axis(g[2], tk)
                halves = a[1] * c[1] * e[1]
                if halves > 0:
                    idx = wp.int32(-1)
                    if coarse_map.shape[0] > 0:
                        idx = band_node(coarse_map, nbk_c, a[0], c[0], e[0])
                    else:
                        idx = poisson_grid_index(a[0], c[0], e[0], res_c)
                    if idx < 0:
                        wp.atomic_add(out_missing, 0, 1)
                    else:
                        weight = wp.float32(halves) * 0.125
                        wp.atomic_add(out_b, idx, weight * bv)
                        wp.atomic_add(out_weights, idx, weight * wv)


@wp.kernel
def dense_base_setup(
    b: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    res: wp.int32,
    inv_lap: wp.float32,
    screen: wp.float32,
    omega: wp.float32,
    out_b: wp.array[wp.float32],
    out_smoother: wp.array[wp.float32],
) -> None:
    # The dense base level's system in the form the dense solver takes: its operator
    # ``lap * L + s W`` divided by ``lap`` (so ``screen`` here is ``s / lap`` and the right-hand
    # side is scaled by ``inv_lap``), and that operator's damped-Jacobi scaling.
    i, j, k = wp.tid()
    idx = poisson_grid_index(i, j, k, res)
    out_b[idx] = b[idx] * inv_lap
    out_smoother[idx] = screened_laplacian_scaled_inverse_diagonal(
        wp.float32(1.0), weights[idx], screen, omega, res, i, j, k
    )


# ---------------------------------------------------------------------------------------------
# A band level's system: Dirichlet ghosts from the coarser level, and its CG
# ---------------------------------------------------------------------------------------------


@wp.func
def band_prolong(
    coarse_map: wp.array[wp.int32],
    coarse: wp.array[wp.float32],
    nbk_c: wp.int32,
    res_c: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
) -> wp.vec2:
    # ``(P x_coarse)`` at fine node ``(i, j, k)`` and whether every coarse node it reads exists:
    # ``poisson_prolong_node`` on a dense coarse level (``coarse_map`` empty), the same trilinear
    # factor-2 average on a band one.
    if coarse_map.shape[0] == 0:
        return wp.vec2(poisson_prolong_node(coarse, res_c, i, j, k), 1.0)
    i0 = i >> 1
    j0 = j >> 1
    k0 = k >> 1
    i1 = (i + 1) >> 1
    j1 = (j + 1) >> 1
    k1 = (k + 1) >> 1
    total = wp.float32(0.0)
    found = wp.float32(1.0)
    for c in range(8):
        idx = band_node(
            coarse_map,
            nbk_c,
            wp.where((c & 4) == 0, i0, i1),
            wp.where((c & 2) == 0, j0, j1),
            wp.where((c & 1) == 0, k0, k1),
        )
        if idx < 0:
            found = 0.0
        else:
            total += coarse[idx]
    return wp.vec2(total * 0.125, found)


@wp.func
def band_ghost(
    brick_map: wp.array[wp.int32],
    nbk: wp.int32,
    coarse_map: wp.array[wp.int32],
    coarse: wp.array[wp.float32],
    nbk_c: wp.int32,
    res_c: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
) -> wp.vec2:
    # A neighbour's Dirichlet value and whether it was found: zero for a band node (an unknown, not
    # a ghost), the prolonged coarse solution otherwise.
    if band_node(brick_map, nbk, i, j, k) >= 0:
        return wp.vec2(0.0, 1.0)
    return band_prolong(coarse_map, coarse, nbk_c, res_c, i, j, k)


@wp.func
def band_neighbor(
    brick_map: wp.array[wp.int32],
    nbk: wp.int32,
    n: wp.int32,
    step: wp.int32,
    on_face: wp.bool,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
) -> wp.int32:
    # Storage index of a stencil neighbour: ``n + step`` inside the brick, a brick-map lookup when
    # the step leaves it (``on_face``); ``-1`` off the band.
    if on_face:
        return band_node(brick_map, nbk, i, j, k)
    return n + step


@wp.func
def band_row(
    brick_coords: wp.array[wp.vec3i],
    brick_map: wp.array[wp.int32],
    nbk: wp.int32,
    res: wp.int32,
    lap: wp.float32,
    weights: wp.array[wp.float32],
    screen: wp.float32,
    x: wp.array[wp.float32],
    n: wp.int32,
) -> wp.float32:
    # Row ``n`` of ``A_band x`` for ``A = lap * L_N + screen * diag(W)`` with every out-of-band
    # neighbour at zero -- ``screened_laplacian_row`` on the band, its ghosts having been moved to
    # the right-hand side by ``band_level_setup``. Zero on an overhanging slot.
    g = band_coordinates(brick_coords, n)
    if not node_live(g, res):
        return wp.float32(0.0)
    i = g[0]
    j = g[1]
    k = g[2]
    li = (n >> 6) & 7
    lj = (n >> 3) & 7
    lk = n & 7
    deg = wp.float32(0.0)
    acc = wp.float32(0.0)
    if i + 1 < res:
        deg += 1.0
        idx = band_neighbor(brick_map, nbk, n, 64, li == 7, i + 1, j, k)
        if idx >= 0:
            acc += x[idx]
    if i > 0:
        deg += 1.0
        idx = band_neighbor(brick_map, nbk, n, -64, li == 0, i - 1, j, k)
        if idx >= 0:
            acc += x[idx]
    if j + 1 < res:
        deg += 1.0
        idx = band_neighbor(brick_map, nbk, n, 8, lj == 7, i, j + 1, k)
        if idx >= 0:
            acc += x[idx]
    if j > 0:
        deg += 1.0
        idx = band_neighbor(brick_map, nbk, n, -8, lj == 0, i, j - 1, k)
        if idx >= 0:
            acc += x[idx]
    if k + 1 < res:
        deg += 1.0
        idx = band_neighbor(brick_map, nbk, n, 1, lk == 7, i, j, k + 1)
        if idx >= 0:
            acc += x[idx]
    if k > 0:
        deg += 1.0
        idx = band_neighbor(brick_map, nbk, n, -1, lk == 0, i, j, k - 1)
        if idx >= 0:
            acc += x[idx]
    xc = x[n]
    return lap * (deg * xc - acc) + screen * weights[n] * xc


@wp.kernel
def band_level_setup(
    brick_coords: wp.array[wp.vec3i],
    brick_map: wp.array[wp.int32],
    nbk: wp.int32,
    res: wp.int32,
    lap: wp.float32,
    weights: wp.array[wp.float32],
    screen: wp.float32,
    coarse_map: wp.array[wp.int32],
    coarse: wp.array[wp.float32],
    nbk_c: wp.int32,
    res_c: wp.int32,
    b: wp.array[wp.float32],
    out_x: wp.array[wp.float32],
    out_inv_diag: wp.array[wp.float32],
    out_missing: wp.array[wp.int32],
) -> None:
    # Moves every out-of-band neighbour's Dirichlet value onto the right-hand side in place
    # (``b += lap * sum ghosts``), starts the solution at the prolonged coarse one, and writes the
    # Jacobi inverse diagonal ``1 / (lap * deg + screen * W)``. Overhanging slots are all zero. A
    # coarse node a ghost or the start reads but the coarse band lacks is counted.
    n = wp.int32(wp.tid())
    g = band_coordinates(brick_coords, n)
    if not node_live(g, res):
        b[n] = 0.0
        out_x[n] = 0.0
        out_inv_diag[n] = 0.0
        return
    i = g[0]
    j = g[1]
    k = g[2]
    deg = wp.float32(0.0)
    ghosts = wp.vec2(0.0, 1.0)
    if i + 1 < res:
        deg += 1.0
        ghost = band_ghost(brick_map, nbk, coarse_map, coarse, nbk_c, res_c, i + 1, j, k)
        ghosts = wp.vec2(ghosts[0] + ghost[0], wp.min(ghosts[1], ghost[1]))
    if i > 0:
        deg += 1.0
        ghost = band_ghost(brick_map, nbk, coarse_map, coarse, nbk_c, res_c, i - 1, j, k)
        ghosts = wp.vec2(ghosts[0] + ghost[0], wp.min(ghosts[1], ghost[1]))
    if j + 1 < res:
        deg += 1.0
        ghost = band_ghost(brick_map, nbk, coarse_map, coarse, nbk_c, res_c, i, j + 1, k)
        ghosts = wp.vec2(ghosts[0] + ghost[0], wp.min(ghosts[1], ghost[1]))
    if j > 0:
        deg += 1.0
        ghost = band_ghost(brick_map, nbk, coarse_map, coarse, nbk_c, res_c, i, j - 1, k)
        ghosts = wp.vec2(ghosts[0] + ghost[0], wp.min(ghosts[1], ghost[1]))
    if k + 1 < res:
        deg += 1.0
        ghost = band_ghost(brick_map, nbk, coarse_map, coarse, nbk_c, res_c, i, j, k + 1)
        ghosts = wp.vec2(ghosts[0] + ghost[0], wp.min(ghosts[1], ghost[1]))
    if k > 0:
        deg += 1.0
        ghost = band_ghost(brick_map, nbk, coarse_map, coarse, nbk_c, res_c, i, j, k - 1)
        ghosts = wp.vec2(ghosts[0] + ghost[0], wp.min(ghosts[1], ghost[1]))
    start = band_prolong(coarse_map, coarse, nbk_c, res_c, i, j, k)
    if ghosts[1] == 0.0 or start[1] == 0.0:
        wp.atomic_add(out_missing, 0, 1)
    b[n] = b[n] + lap * ghosts[0]
    out_x[n] = start[0]
    out_inv_diag[n] = 1.0 / (lap * deg + screen * weights[n])


@wp.kernel
def band_cg_initial(
    n: wp.int32,
    span: wp.int32,
    brick_coords: wp.array[wp.vec3i],
    brick_map: wp.array[wp.int32],
    nbk: wp.int32,
    res: wp.int32,
    lap: wp.float32,
    weights: wp.array[wp.float32],
    screen: wp.float32,
    rhs: wp.array[wp.float32],
    x: wp.array[wp.float32],
    inv_diag: wp.array[wp.float32],
    out_r: wp.array[wp.float32],
    out_u: wp.array[wp.float32],
    out_p: wp.array[wp.float32],
    out_s: wp.array[wp.float32],
    out_partials: wp.array3d[wp.float64],
) -> None:
    # ``kernels/reconstruction.poisson_cg_initial`` on a band level, with Jacobi in place of the
    # V-cycle: ``r = b - A x``, ``u = D^-1 r``, ``p = s = 0`` and the first stage of ``||b||^2``,
    # over one column padded to whole blocks of ``span``; the pad is zero. Lanes stride by
    # ``wp.block_dim()`` (section 2.2).
    blk, t = wp.tid()
    acc = wp.float64(0.0)
    for kk in range(t, span, wp.block_dim()):
        local = blk * span + kk
        bv = wp.float64(0.0)
        r = wp.float32(0.0)
        u = wp.float32(0.0)
        if local < n:
            bv = wp.float64(rhs[local])
            r = rhs[local] - band_row(
                brick_coords, brick_map, nbk, res, lap, weights, screen, x, local
            )
            u = inv_diag[local] * r
        out_r[local] = r
        out_u[local] = u
        out_p[local] = wp.float32(0.0)
        out_s[local] = wp.float32(0.0)
        acc += bv * bv
    total = block_sum(acc)
    if t == 0:
        out_partials[0, 0, blk] = total


@wp.kernel
def band_cg_matvec_dots(
    n: wp.int32,
    span: wp.int32,
    brick_coords: wp.array[wp.vec3i],
    brick_map: wp.array[wp.int32],
    nbk: wp.int32,
    res: wp.int32,
    lap: wp.float32,
    weights: wp.array[wp.float32],
    screen: wp.float32,
    r: wp.array[wp.float32],
    u: wp.array[wp.float32],
    gamma_new: wp.array[wp.float64],
    alpha_new: wp.array[wp.float64],
    out_w: wp.array[wp.float32],
    out_partials: wp.array3d[wp.float64],
    out_gamma_old: wp.array[wp.float64],
    out_alpha_old: wp.array[wp.float64],
    out_state: wp.array[wp.int32],
) -> None:
    # ``poisson_cg_matvec_dots`` on a band level: ``w = A u`` with the band row, the round's three
    # dots (``cg_round_terms``) and the shared tail (``cg_publish_round_dots``).
    blk, t = wp.tid()
    acc = wp.vec3(0.0, 0.0, 0.0)
    for kk in range(t, span, wp.block_dim()):
        local = blk * span + kk
        w = wp.float32(0.0)
        if local < n:
            w = band_row(brick_coords, brick_map, nbk, res, lap, weights, screen, u, local)
        out_w[local] = w
        acc += cg_round_terms(r[local], u[local], w)
    total = cg_widen(block_sum(acc))
    cg_publish_round_dots(
        total,
        0,
        blk,
        t,
        gamma_new,
        alpha_new,
        out_partials,
        out_gamma_old,
        out_alpha_old,
        out_state,
    )


# ---------------------------------------------------------------------------------------------
# Iso-value and extraction on the composite field
# ---------------------------------------------------------------------------------------------


@wp.kernel
def band_sample(
    points: wp.array[wp.vec3],
    lower: wp.vec3,
    inv_cell: wp.float32,
    res: wp.int32,
    brick_map: wp.array[wp.int32],
    nbk: wp.int32,
    x: wp.array[wp.float32],
    out_values: wp.array[wp.float32],
) -> None:
    # The finest band's solution at every sample, trilinearly (``sample_field_trilinear``'s cell
    # and weights); every corner is in the band (``mark_sample_bricks``).
    s = wp.int32(wp.tid())
    g = (points[s] - lower) * inv_cell
    base, next_corner, fractions = trilinear_cell(g, wp.vec3i(res, res, res))
    acc = wp.float32(0.0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                corner = trilinear_corner(base, next_corner, wp.vec3i(di, dj, dk))
                acc += (
                    trilinear_weight(fractions, di, dj, dk)
                    * x[band_node(brick_map, nbk, corner[0], corner[1], corner[2])]
                )
    out_values[s] = acc


@wp.func
def composite_value(
    brick_map: wp.array[wp.int32],
    nbk: wp.int32,
    x: wp.array[wp.float32],
    coarse_map: wp.array[wp.int32],
    coarse: wp.array[wp.float32],
    nbk_c: wp.int32,
    res_c: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
) -> wp.float32:
    # The finest level's field at node ``(i, j, k)``: its band value, or the prolonged coarse
    # solution off the band -- the same value the band's Dirichlet ghosts held, so the field is
    # continuous across the band boundary.
    idx = band_node(brick_map, nbk, i, j, k)
    if idx >= 0:
        return x[idx]
    return band_prolong(coarse_map, coarse, nbk_c, res_c, i, j, k)[0]


@wp.func
def composite_edge_crosses(
    brick_map: wp.array[wp.int32],
    nbk: wp.int32,
    x: wp.array[wp.float32],
    coarse_map: wp.array[wp.int32],
    coarse: wp.array[wp.float32],
    nbk_c: wp.int32,
    res_c: wp.int32,
    res: wp.int32,
    iso: wp.float32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    axis: wp.int32,
) -> wp.int32:
    # ``kernels/levelset.mc_edge_crosses`` on the composite field.
    io = i + wp.where(axis == 0, 1, 0)
    jo = j + wp.where(axis == 1, 1, 0)
    ko = k + wp.where(axis == 2, 1, 0)
    if io >= res or jo >= res or ko >= res:
        return 0
    return mc_values_cross(
        composite_value(brick_map, nbk, x, coarse_map, coarse, nbk_c, res_c, i, j, k),
        composite_value(brick_map, nbk, x, coarse_map, coarse, nbk_c, res_c, io, jo, ko),
        iso,
    )


@wp.func
def composite_cell(
    extract_map: wp.array[wp.int32],
    nbk: wp.int32,
    brick_map: wp.array[wp.int32],
    x: wp.array[wp.float32],
    coarse_map: wp.array[wp.int32],
    coarse: wp.array[wp.float32],
    nbk_c: wp.int32,
    res_c: wp.int32,
    res: wp.int32,
    iso: wp.float32,
    table: wp.array[wp.int32],
    g: wp.vec3i,
) -> wp.vec3i:
    # ``(first table slot, triangle count, skipped)`` of the cell whose lower node is ``g``. A cell
    # owns triangles only when every node owning one of its edges -- each corner but the far one --
    # is in the extraction band, since an edge's vertex slot belongs to its owner; a crossing cell
    # that fails that is reported as ``skipped`` (the caller counts it: a hole).
    if g[0] + 1 >= res or g[1] + 1 >= res or g[2] + 1 >= res:
        return wp.vec3i(0, 0, 0)
    code = wp.int32(0)
    for c in range(8):
        value = composite_value(
            brick_map,
            nbk,
            x,
            coarse_map,
            coarse,
            nbk_c,
            res_c,
            g[0] + wp.static(MC_CORNERS[c][0]),
            g[1] + wp.static(MC_CORNERS[c][1]),
            g[2] + wp.static(MC_CORNERS[c][2]),
        )
        if value >= iso:
            code += wp.static(1 << c)
    start = table[code]
    count = (table[code + 1] - start) // 3
    if count == 0:
        return wp.vec3i(0, 0, 0)
    for c in range(8):
        # Corner 6 is ``(1, 1, 1)``, the one corner that owns none of the cell's edges.
        if c != 6 and (
            band_node(
                extract_map,
                nbk,
                g[0] + wp.static(MC_CORNERS[c][0]),
                g[1] + wp.static(MC_CORNERS[c][1]),
                g[2] + wp.static(MC_CORNERS[c][2]),
            )
            < 0
        ):
            return wp.vec3i(0, 0, 1)
    return wp.vec3i(start, count, 0)


@wp.kernel
def band_marching_cubes_counts(
    extract_coords: wp.array[wp.vec3i],
    extract_map: wp.array[wp.int32],
    nbk: wp.int32,
    brick_map: wp.array[wp.int32],
    x: wp.array[wp.float32],
    coarse_map: wp.array[wp.int32],
    coarse: wp.array[wp.float32],
    nbk_c: wp.int32,
    res_c: wp.int32,
    res: wp.int32,
    iso: wp.float32,
    table: wp.array[wp.int32],
    out_counts: wp.array[wp.vec2i],
    out_skipped: wp.array[wp.int32],
) -> None:
    # ``kernels/levelset.marching_cubes_counts`` over the extraction band's nodes on the composite
    # field: each node's crossing edges and its cell's triangles, into one ``vec2i``.
    n = wp.int32(wp.tid())
    g = band_coordinates(extract_coords, n)
    if not node_live(g, res):
        out_counts[n] = wp.vec2i(0, 0)
        return
    vertices = wp.int32(0)
    for axis in range(3):
        vertices += composite_edge_crosses(
            brick_map, nbk, x, coarse_map, coarse, nbk_c, res_c, res, iso, g[0], g[1], g[2], axis
        )
    cell = composite_cell(
        extract_map, nbk, brick_map, x, coarse_map, coarse, nbk_c, res_c, res, iso, table, g
    )
    if cell[2] != 0:
        wp.atomic_add(out_skipped, 0, 1)
    out_counts[n] = wp.vec2i(vertices, cell[1])


@wp.kernel
def band_marching_cubes_emit(
    extract_coords: wp.array[wp.vec3i],
    extract_map: wp.array[wp.int32],
    nbk: wp.int32,
    brick_map: wp.array[wp.int32],
    x: wp.array[wp.float32],
    coarse_map: wp.array[wp.int32],
    coarse: wp.array[wp.float32],
    nbk_c: wp.int32,
    res_c: wp.int32,
    res: wp.int32,
    iso: wp.float32,
    lower: wp.vec3,
    delta: wp.vec3,
    margin: wp.float32,
    table: wp.array[wp.int32],
    offsets: wp.array[wp.vec2i],
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # ``kernels/levelset.marching_cubes_emit`` over the extraction band: the same vertex arithmetic
    # (``mc_edge_vertex``) and face numbering, with an edge's vertex found through its owner node's
    # extraction-band slot.
    n = wp.int32(wp.tid())
    g = band_coordinates(extract_coords, n)
    if not node_live(g, res):
        return
    ends = offsets[n]
    here = composite_value(brick_map, nbk, x, coarse_map, coarse, nbk_c, res_c, g[0], g[1], g[2])
    crossing = wp.vec3i(0, 0, 0)
    for axis in range(3):
        crossing[axis] = composite_edge_crosses(
            brick_map, nbk, x, coarse_map, coarse, nbk_c, res_c, res, iso, g[0], g[1], g[2], axis
        )
    slot = ends[0] - (crossing[0] + crossing[1] + crossing[2])
    for axis in range(3):
        if crossing[axis] != 0:
            there = composite_value(
                brick_map,
                nbk,
                x,
                coarse_map,
                coarse,
                nbk_c,
                res_c,
                g[0] + wp.where(axis == 0, 1, 0),
                g[1] + wp.where(axis == 1, 1, 0),
                g[2] + wp.where(axis == 2, 1, 0),
            )
            out_vertices[slot] = mc_edge_vertex(
                lower, delta, g[0], g[1], g[2], axis, here, there, iso, margin
            )
            slot += 1
    cell = composite_cell(
        extract_map, nbk, brick_map, x, coarse_map, coarse, nbk_c, res_c, res, iso, table, g
    )
    first_face = ends[1] - cell[1]
    for tri in range(cell[1]):
        for s in range(3):
            edge = mc_triangle_edge(table, cell[0] + 3 * tri + s)
            oi = g[0] + edge[0]
            oj = g[1] + edge[1]
            ok = g[2] + edge[2]
            axis = edge[3]
            vertex = wp.int32(-1)
            if (
                composite_edge_crosses(
                    brick_map, nbk, x, coarse_map, coarse, nbk_c, res_c, res, iso, oi, oj, ok, axis
                )
                != 0
            ):
                vertex = offsets[band_node(extract_map, nbk, oi, oj, ok)][0] - 1
                for higher in range(axis + 1, 3):
                    vertex -= composite_edge_crosses(
                        brick_map,
                        nbk,
                        x,
                        coarse_map,
                        coarse,
                        nbk_c,
                        res_c,
                        res,
                        iso,
                        oi,
                        oj,
                        ok,
                        higher,
                    )
            out_faces[3 * (first_face + tri) + s] = vertex


@wp.kernel
def embed_band_data(
    brick_coords: wp.array[wp.vec3i],
    res: wp.int32,
    b: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    target_map: wp.array[wp.int32],
    nbk: wp.int32,
    out_b: wp.array[wp.float32],
    out_weights: wp.array[wp.float32],
) -> None:
    # A level's data from the band it was first built on into a band containing it (the final
    # band adds the bricks the coarse surface crosses, where the data is zero): ``out_*`` are zeroed
    # by the caller and every source node lands on its own target slot.
    n = wp.int32(wp.tid())
    g = band_coordinates(brick_coords, n)
    if not node_live(g, res):
        return
    idx = band_node(target_map, nbk, g[0], g[1], g[2])
    out_b[idx] = b[n]
    out_weights[idx] = weights[n]
