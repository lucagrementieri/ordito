import math

import warp as wp

from ordito.constants import TILE_1D, TOLERANCE_ZERO_CONSTANT
from ordito.kernels.array import lift_vec2
from ordito.kernels.polyline import segment_displacement
from ordito.kernels.predicates import orient2d
from ordito.kernels.reduce import block_sum, tile_chunk
from ordito.kernels.triangles import corner_triple, face_vertices, write_corner_triple_reversible

SQRT3 = wp.constant(wp.float32(math.sqrt(3.0)))
PI_F = wp.constant(wp.float32(math.pi))

# Below this magnitude a normal component is snapped to zero before the spherical conversion in
# ``sweep_transforms``, matching trimesh's ``vector_to_spherical``. Without it a Z+ normal whose
# x/y are float noise gets a meaningless azimuth.
SPHERICAL_SNAP = wp.constant(wp.float32(1e-8))


@wp.func
def project_to_radius(v: wp.vec3, radius: wp.float32) -> wp.vec3:
    # trimesh's icosphere projection, kept in its additive form (v + unit * (r - |v|)) rather
    # than the algebraically equal unit * r, so the float32 result matches the reference bit
    # for bit.
    return v + wp.normalize(v) * (radius - wp.length(v))


@wp.kernel
def grid_mesh(
    nx: wp.int32,
    ny: wp.int32,
    width: wp.float64,
    height: wp.float64,
    origin_x: wp.float64,
    origin_y: wp.float64,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # A whole flat patch in the z = 0 plane from six scalars, one thread per vertex: the (nx, ny)
    # lattice row-major with X the slow axis, and -- from every thread that is also the lower corner
    # of a quad cell, i.e. all but the last row and column -- that cell's two triangles. Both
    # buffers are closed form in the thread index, so one launch writes them, as ``cone_mesh`` and
    # ``cylinder_mesh`` do for theirs.
    #
    # Written as ``extent * (k / (n - 1))`` rather than ``k * (extent / (n - 1))``: the fraction is
    # exactly 1 at the last sample, so the far edge lands on the extent without the endpoint
    # special case ``numpy.linspace`` needs. The arithmetic is float64 for the same reason the host
    # build was -- the float32 store then rounds off an exact value rather than an accumulated one.
    i, j = wp.tid()
    x = origin_x + width * (wp.float64(i) / wp.float64(nx - 1))
    y = origin_y + height * (wp.float64(j) / wp.float64(ny - 1))
    # ``wp.float32(...)``, not ``wp.cast``: the latter is a same-size bit reinterpretation and
    # fails to compile on a float64 source ("source and destination must have the same size").
    out_vertices[i * ny + j] = wp.vec3(wp.float32(x), wp.float32(y), wp.float32(0.0))
    if i >= nx - 1 or j >= ny - 1:
        return
    # The cell's two triangles, wound counter-clockwise seen from +Z. Its corners are ``corner``,
    # ``corner + ny`` (next X) and ``+ 1`` (next Y), and cell ``(i, j)`` owns face slots
    # ``2 * (i * (ny - 1) + j)`` and the next.
    corner = i * ny + j
    slot = (i * (ny - 1) + j) * 6
    out_faces[slot + 0] = corner
    out_faces[slot + 1] = corner + ny
    out_faces[slot + 2] = corner + ny + 1
    out_faces[slot + 3] = corner
    out_faces[slot + 4] = corner + ny + 1
    out_faces[slot + 5] = corner + 1


# Vertex and edge counts of the base icosahedron, which set the two block boundaries of the
# icosphere's closed-form vertex numbering: the 12 corners, then 30 blocks of ``n - 1`` base-edge
# points, then 20 blocks of face-interior points.
ICOSAHEDRON_VERTICES = wp.constant(wp.int32(12))
ICOSAHEDRON_EDGES = wp.constant(wp.int32(30))


@wp.func
def icosphere_vertex_index(
    table: wp.array2d[wp.int32], n: wp.int32, f: wp.int32, i: wp.int32, j: wp.int32
) -> wp.int32:
    # Global index of the icosphere vertex at barycentric ``(i, j, n - i - j)`` of base face ``f``,
    # under the numbering described in the ``table`` docstring of ``_icosphere_face_table``: the 12
    # base corners, then the interior points of each base edge from its lower-numbered endpoint,
    # then the interior points of each base face in row-major barycentric order.
    #
    # The map is many-to-one on purpose: a point on a shared base edge is addressed by both of the
    # faces holding it and lands on the same index from either side, which is what makes the result
    # watertight with no merge pass.
    k = n - i - j
    index = 0
    if i == n:
        index = table[f, 0]
    elif j == n:
        index = table[f, 1]
    elif k == n:
        index = table[f, 2]
    elif k == 0 or i == 0 or j == 0:
        # On a base edge. ``side`` is 0 for a->b (parameter j), 1 for b->c (k), 2 for c->a (i),
        # and the stored flag says whether that side runs against the edge's own direction.
        side = 0
        t = j
        if i == 0:
            side = 1
            t = k
        elif j == 0:
            side = 2
            t = i
        if table[f, 6 + side] != 0:
            t = n - t
        index = ICOSAHEDRON_VERTICES + table[f, 3 + side] * (n - 1) + t - 1
    else:
        interior = (n - 1) * (n - 2) // 2
        local = (i - 1) * (n - 1) - (i - 1) * i // 2 + (j - 1)
        index = ICOSAHEDRON_VERTICES + ICOSAHEDRON_EDGES * (n - 1) + f * interior + local
    return index


@wp.func
def icosphere_edge_vertex(
    table: wp.array2d[wp.int32],
    n: wp.int32,
    f: wp.int32,
    corner_a: wp.vec2i,
    index_a: wp.int32,
    position_a: wp.vec3,
    corner_b: wp.vec2i,
    index_b: wp.int32,
    position_b: wp.vec3,
    radius: wp.float32,
) -> tuple[wp.vec2i, wp.int32, wp.vec3]:
    # The vertex the recursive build places on the edge between lattice points ``corner_a`` and
    # ``corner_b`` (barycentric ``(i, j)`` of base face ``f``, global indices ``index_a`` and
    # ``index_b``), as its lattice point, global index and position. The midpoint is exact: both
    # coordinates of a corner are multiples of its triangle's even size. The position is the
    # projected lerp from the endpoint with the lower global index, matching
    # ``remesh.subdivide``'s sorted unique edges, so the float32 result is the one the
    # iterated-subdivide implementation produced.
    corner = wp.vec2i((corner_a[0] + corner_b[0]) // 2, (corner_a[1] + corner_b[1]) // 2)
    midpoint = wp.lerp(position_b, position_a, wp.float32(0.5))
    if index_a < index_b:
        midpoint = wp.lerp(position_a, position_b, wp.float32(0.5))
    index = icosphere_vertex_index(table, n, f, corner[0], corner[1])
    return corner, index, project_to_radius(midpoint, radius)


@wp.kernel
def icosphere_mesh(
    table: wp.array2d[wp.int32],
    corners: wp.array[wp.vec3],
    n: wp.int32,
    coarse: wp.int32,
    stride: wp.int32,
    radius: wp.float32,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # Several refinement levels of the icosphere at once: every vertex on the barycentric lattice
    # of spacing ``stride`` (``(i, j) = stride * (ti, tj)`` of base face ``f``, ``i + j <= n``),
    # given every vertex on the coarser lattice of spacing ``coarse`` -- already written, or, at
    # ``coarse == n``, the 12 base corners, which this launch writes. The launch with
    # ``stride == 1`` also writes the whole face buffer.
    #
    # The vertex *geometry* is the recursive one: every vertex is the projected midpoint of an
    # edge of the previous level, which is not the projection of the barycentric point. So the
    # thread descends the refinement from the coarse triangle that contains it, keeping the one
    # child triangle that contains it, until it is a corner: it holds the triangle's three lattice
    # corners, their global indices and positions -- each one a vertex of the recursive build,
    # computed from the same two parents -- and its own integer weights ``(w0, w1, w2)`` over the
    # triangle's size ``s``. A corner child halves the size and keeps the weights
    # ``(w0 - s / 2, w1, w2)`` (with its corner in the same slot), the centre child takes
    # ``(s / 2 - w0, s / 2 - w1, s / 2 - w2)``. A point on the edge between two triangles may
    # descend into either: both reach the same vertices from the same parents. A vertex on a shared
    # base edge is computed by both of its faces, identically, and the many-to-one
    # ``icosphere_vertex_index`` lands both on one slot.
    #
    # Each launch descends ``log2(coarse / stride)`` levels, so a thread's work grows with that
    # span where one launch per level does O(1) work per vertex; the wrapper picks the span. Against
    # one launch per level, a single launch for every level was 1.8-3.0x up to six levels and
    # 0.30x at nine (the full lattice descending nine levels); six levels in the first launch and
    # one launch per level after it is 1.5-3.0x through eight levels and 0.97x at nine, where the
    # call is device-bound. Five or seven levels in the first launch lose at seven to nine.
    f, ti, tj = wp.tid()
    i = ti * stride
    j = tj * stride
    if stride == 1 and i < n and j < n:
        # The ``n ** 2`` sub-triangles of one base face, one per thread of the inner ``n x n``
        # block. The ``n (n + 1) / 2`` upward triangles are the threads with ``i + j < n``; the
        # rest of the block is remapped by ``(i, j) -> (n - 1 - i, n - 1 - j)`` onto the
        # ``n (n - 1) / 2`` downward ones, which is a bijection -- so every thread writes exactly
        # one triangle and no prefix-sum over rows is needed.
        slot = (f * n * n + i * n + j) * 3
        if i + j < n:
            out_faces[slot + 0] = icosphere_vertex_index(table, n, f, i + 1, j)
            out_faces[slot + 1] = icosphere_vertex_index(table, n, f, i, j + 1)
            out_faces[slot + 2] = icosphere_vertex_index(table, n, f, i, j)
        else:
            di = n - 1 - i
            dj = n - 1 - j
            out_faces[slot + 0] = icosphere_vertex_index(table, n, f, di, dj + 1)
            out_faces[slot + 1] = icosphere_vertex_index(table, n, f, di + 1, dj)
            out_faces[slot + 2] = icosphere_vertex_index(table, n, f, di + 1, dj + 1)
    if i + j > n:
        return
    # Offsets inside the coarse lattice cell; at ``coarse == n`` the cell is the base face itself
    # (``i == n`` is its corner ``a``, not the next cell's origin).
    ri = i
    rj = j
    if coarse < n:
        ri = i % coarse
        rj = j % coarse
        if ri == 0 and rj == 0:
            # Already on the coarse lattice, so already written.
            return

    # The coarse triangle holding the vertex, corner ``k`` weighted by ``wk``: the upward one with
    # corners ``(bi + 1, bj)``, ``(bi, bj + 1)``, ``(bi, bj)`` (in units of ``coarse``) when
    # ``ri + rj <= coarse``, else the downward one ``(bi, bj + 1)``, ``(bi + 1, bj)``,
    # ``(bi + 1, bj + 1)``. At ``coarse == n`` that is the base face, corners ``a``, ``b``, ``c``.
    bi = i - ri
    bj = j - rj
    corner0 = wp.vec2i(bi + coarse, bj)
    corner1 = wp.vec2i(bi, bj + coarse)
    corner2 = wp.vec2i(bi, bj)
    w0 = ri
    w1 = rj
    if ri + rj > coarse:
        corner0 = wp.vec2i(bi, bj + coarse)
        corner1 = wp.vec2i(bi + coarse, bj)
        corner2 = wp.vec2i(bi + coarse, bj + coarse)
        w0 = coarse - ri
        w1 = coarse - rj
    w2 = coarse - w0 - w1
    s = wp.int32(coarse)
    index0 = icosphere_vertex_index(table, n, f, corner0[0], corner0[1])
    index1 = icosphere_vertex_index(table, n, f, corner1[0], corner1[1])
    index2 = icosphere_vertex_index(table, n, f, corner2[0], corner2[1])
    position0 = wp.vec3()
    position1 = wp.vec3()
    position2 = wp.vec3()
    if coarse == n:
        position0 = project_to_radius(corners[index0], radius)
        position1 = project_to_radius(corners[index1], radius)
        position2 = project_to_radius(corners[index2], radius)
    else:
        position0 = out_vertices[index0]
        position1 = out_vertices[index1]
        position2 = out_vertices[index2]
    while w0 != s and w1 != s and w2 != s:
        h = s // 2
        # A vertex that is the midpoint of one of the triangle's sides is that edge's vertex, with
        # no need for the child's other corners: always the case at the last level.
        if w2 == 0 and w0 == h:
            c01, k01, p01 = icosphere_edge_vertex(
                table, n, f, corner0, index0, position0, corner1, index1, position1, radius
            )
            out_vertices[k01] = p01
            return
        if w0 == 0 and w1 == h:
            c12, k12, p12 = icosphere_edge_vertex(
                table, n, f, corner1, index1, position1, corner2, index2, position2, radius
            )
            out_vertices[k12] = p12
            return
        if w1 == 0 and w2 == h:
            c20, k20, p20 = icosphere_edge_vertex(
                table, n, f, corner2, index2, position2, corner0, index0, position0, radius
            )
            out_vertices[k20] = p20
            return
        if w0 >= h:
            c01, k01, p01 = icosphere_edge_vertex(
                table, n, f, corner0, index0, position0, corner1, index1, position1, radius
            )
            c20, k20, p20 = icosphere_edge_vertex(
                table, n, f, corner2, index2, position2, corner0, index0, position0, radius
            )
            corner1 = c01
            index1 = k01
            position1 = p01
            corner2 = c20
            index2 = k20
            position2 = p20
            w0 = w0 - h
        elif w1 >= h:
            c01, k01, p01 = icosphere_edge_vertex(
                table, n, f, corner0, index0, position0, corner1, index1, position1, radius
            )
            c12, k12, p12 = icosphere_edge_vertex(
                table, n, f, corner1, index1, position1, corner2, index2, position2, radius
            )
            corner0 = c01
            index0 = k01
            position0 = p01
            corner2 = c12
            index2 = k12
            position2 = p12
            w1 = w1 - h
        elif w2 >= h:
            c20, k20, p20 = icosphere_edge_vertex(
                table, n, f, corner2, index2, position2, corner0, index0, position0, radius
            )
            c12, k12, p12 = icosphere_edge_vertex(
                table, n, f, corner1, index1, position1, corner2, index2, position2, radius
            )
            corner0 = c20
            index0 = k20
            position0 = p20
            corner1 = c12
            index1 = k12
            position1 = p12
            w2 = w2 - h
        else:
            # The centre child: corner ``k`` becomes the midpoint of the side opposite it.
            c01, k01, p01 = icosphere_edge_vertex(
                table, n, f, corner0, index0, position0, corner1, index1, position1, radius
            )
            c12, k12, p12 = icosphere_edge_vertex(
                table, n, f, corner1, index1, position1, corner2, index2, position2, radius
            )
            c20, k20, p20 = icosphere_edge_vertex(
                table, n, f, corner2, index2, position2, corner0, index0, position0, radius
            )
            corner0 = c12
            index0 = k12
            position0 = p12
            corner1 = c20
            index1 = k20
            position1 = p20
            corner2 = c01
            index2 = k01
            position2 = p01
            w0 = h - w0
            w1 = h - w1
            w2 = h - w2
        s = h
    # The corner the descent ended on is the vertex itself, so its index is the thread's slot.
    if w0 == s:
        out_vertices[index0] = position0
    elif w1 == s:
        out_vertices[index1] = position1
    else:
        out_vertices[index2] = position2


@wp.func
def revolve_template_triangle(t: wp.int32, per: wp.int32) -> wp.vec3i:
    # trimesh's quad template [0, per, 1, 1, per, per + 1] tiled over profile segment i = t // 2
    # and offset by i: two triangles per segment, `per` being the vertex stride between slices.
    i = t // 2
    if t % 2 == 0:
        return wp.vec3i(i, i + per, i + 1)
    return wp.vec3i(i + 1, i + per, i + per + 1)


@wp.func
def revolve_vertex_slot(
    s: wp.int32, i: wp.int32, n_slices: wp.int32, layout: wp.array[wp.int32], per: wp.int32
) -> wp.int32:
    # Final index of the vertex at slice `s`, profile point `i`. ``layout`` holds three
    # per-profile-point tables, ``column | offsets | on_axis`` at stride ``per``, built on the host;
    # they encode every coincidence a revolution produces, so the buffer is written in its final,
    # already-merged layout: a profile point on the revolution axis owns one vertex for the whole
    # revolution, and a profile whose last point repeats its first shares that column. The modulus
    # closes a full revolution by folding the last slice onto slice 0.
    j = layout[i]
    if layout[2 * per + j] != 0:
        return layout[per + j]
    return layout[per + j] + s % n_slices


@wp.func
def revolution_point(
    radius: wp.float32,
    height: wp.float32,
    slice_index: wp.int32,
    n_slices: wp.int32,
    angle: wp.float32,
) -> wp.vec3:
    # One profile point (its 2D x is the revolution radius, its y the height along Z) carried to
    # slice ``slice_index`` of ``n_slices``. Shared by the general ``revolve`` engine and by the
    # closed-form solids below, so the three of them cannot drift apart in the last bit.
    #
    # theta = np.linspace(0, angle, n_slices + 1)[slice_index] -- written as a fraction of the span
    # so the final slice lands exactly on ``angle`` instead of accumulating a step.
    theta = angle * wp.float32(slice_index) / wp.float32(n_slices)
    return wp.vec3(wp.cos(theta) * radius, wp.sin(theta) * radius, height)


@wp.kernel
def cone_mesh(
    radius: wp.float32,
    height: wp.float32,
    sections: wp.int32,
    angle: wp.float32,
    keep_caps: wp.int32,
    keep_sides: wp.int32,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # A whole closed cone from six scalars: ``sections + 2`` vertices and ``2 * sections`` faces,
    # one thread per wedge, no host-side layout and nothing uploaded.
    #
    # ``creation.cone`` used to reach this shape through ``revolve``, which is general over an
    # arbitrary profile and pays for that generality on the host: it builds a three-point profile in
    # NumPy, uploads it, **reads it straight back** to decide which template triangles survive and
    # where each (slice, profile point) pair lands, then uploads four more layout tables. The layout
    # a cone needs is closed form -- the base fan around the axis point, the side fan to the apex --
    # so the whole prologue collapses to this launch. See ``revolution_point`` for the one shared
    # piece, which is what keeps the two engines bit-identical.
    k = wp.int32(wp.tid())
    # Vertex 0 is the base's axis point, 1..sections the base ring, sections + 1 the apex. The two
    # axis points are written by one thread rather than by every one: they are a single slot each,
    # and the profile points they come from lie *on* the axis, where ``revolution_point`` returns
    # the same value for every slice.
    out_vertices[1 + k] = revolution_point(radius, 0.0, k, sections, angle)
    if k == 0:
        out_vertices[0] = revolution_point(0.0, 0.0, 0, sections, angle)
        out_vertices[sections + 1] = revolution_point(0.0, height, 0, sections, angle)

    # ``keep_caps`` / ``keep_sides`` are ``revolve``'s own degenerate-area verdict on this
    # profile's template, decided by the caller through the very function ``revolve`` uses -- so a
    # one- or two-section cone drops the triangles that collapse onto the axis exactly as the
    # general engine does, and so does a cone small enough for its caps to fall under the area
    # tolerance. Emission stays in template order: cap, then side.
    next_ring = 1 + (k + 1) % sections
    slot = (keep_caps + keep_sides) * 3 * k
    if keep_caps != 0:
        out_faces[slot + 0] = 1 + k
        out_faces[slot + 1] = 0
        out_faces[slot + 2] = next_ring
        slot = slot + 3
    if keep_sides != 0:
        out_faces[slot + 0] = 1 + k
        out_faces[slot + 1] = next_ring
        out_faces[slot + 2] = sections + 1


@wp.kernel
def cylinder_mesh(
    radius: wp.float32,
    half_height: wp.float32,
    sections: wp.int32,
    angle: wp.float32,
    keep_caps: wp.int32,
    keep_sides: wp.int32,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # The closed-form counterpart of ``cone_mesh`` for a capped cylinder: ``2 * sections + 2``
    # vertices and ``4 * sections`` faces, one thread per wedge. Same reasoning -- see
    # ``cone_mesh``.
    k = wp.int32(wp.tid())
    # Vertex 0 is the bottom axis point, 1..sections the bottom ring, sections + 1..2 * sections
    # the top ring, 2 * sections + 1 the top axis point: the profile's four points in order, each
    # contributing a whole ring or a single shared slot.
    out_vertices[1 + k] = revolution_point(radius, -half_height, k, sections, angle)
    out_vertices[1 + sections + k] = revolution_point(radius, half_height, k, sections, angle)
    if k == 0:
        out_vertices[0] = revolution_point(0.0, -half_height, 0, sections, angle)
        out_vertices[2 * sections + 1] = revolution_point(0.0, half_height, 0, sections, angle)

    bottom = 1 + k
    bottom_next = 1 + (k + 1) % sections
    top = 1 + sections + k
    top_next = 1 + sections + (k + 1) % sections
    # Bottom cap, the wedge's two side triangles, then the top cap -- the order ``revolve`` emits
    # them in, which is per slice rather than per band. The two flags are its degenerate-area
    # verdict; see ``cone_mesh``. Both caps share a radius and a step and both sides split one
    # rectangle, so each pair stands or falls together and two flags cover the four.
    slot = (2 * keep_caps + 2 * keep_sides) * 3 * k
    if keep_caps != 0:
        out_faces[slot + 0] = bottom
        out_faces[slot + 1] = 0
        out_faces[slot + 2] = bottom_next
        slot = slot + 3
    if keep_sides != 0:
        out_faces[slot + 0] = bottom
        out_faces[slot + 1] = bottom_next
        out_faces[slot + 2] = top
        out_faces[slot + 3] = top
        out_faces[slot + 4] = bottom_next
        out_faces[slot + 5] = top_next
        slot = slot + 6
    if keep_caps != 0:
        out_faces[slot + 0] = top
        out_faces[slot + 1] = top_next
        out_faces[slot + 2] = 2 * sections + 1


@wp.kernel
def revolve_mesh(
    linestring: wp.array[wp.vec2],
    angle: wp.float32,
    n_points: wp.int32,
    n_slices: wp.int32,
    n_face_slices: wp.int32,
    layout: wp.array[wp.int32],
    cap_faces: wp.array[wp.int32],
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # A whole revolution in one launch, over ``n_slices * per + n_face_slices * n_keep + 2 * n_cap``
    # rows; ``layout`` is ``revolve_vertex_slot``'s three tables followed by the ``n_keep``
    # template triangles that survived the host-side degenerate-area filter.
    #
    # Vertex rows: the 2D profile X becomes the revolution radius and the 2D profile Y the height
    # along Z. Only the thread that *owns* a slot writes it, so shared slots have a single
    # deterministic writer (slice 0 for an axis point, the representative column for a closed
    # profile) -- matching trimesh's merge, which also keeps the first occurrence.
    #
    # Face rows: each kept template index addresses slice 0 or slice 1 of the profile grid, so it
    # splits into a slice offset and a profile point before being mapped to its final slot.
    #
    # Cap rows: a profile triangulation (null unless the revolution is partial and capped) on both
    # end slices, the near cap on slice 0 then the far cap on the last slice, whose winding is
    # reversed (trimesh's np.fliplr) so it faces outward too.
    r = wp.int32(wp.tid())
    per = linestring.shape[0]
    if r < n_slices * per:
        s = r // per
        i = r % per
        if layout[i] == i and (s == 0 or layout[2 * per + i] == 0):
            p = linestring[i]
            slot = revolve_vertex_slot(s, i, n_slices, layout, per)
            out_vertices[slot] = revolution_point(p[0], p[1], s, n_points - 1, angle)
        return
    n_keep = layout.shape[0] - 3 * per
    w = r - n_slices * per
    if w < n_face_slices * n_keep:
        s = w // n_keep
        tri = revolve_template_triangle(layout[3 * per + w % n_keep], per)
        for k in range(3):
            g = tri[k]
            out_faces[w * 3 + k] = revolve_vertex_slot(s + g // per, g % per, n_slices, layout, per)
        return
    c = w - n_face_slices * n_keep
    n_cap = cap_faces.shape[0] // 3
    end = c // n_cap
    t = c % n_cap
    slice_index = wp.where(end == 0, wp.int32(0), n_slices - 1)
    a, b, v = corner_triple(cap_faces, t)
    write_corner_triple_reversible(
        out_faces,
        n_face_slices * n_keep + end * n_cap + t,
        revolve_vertex_slot(slice_index, a, n_slices, layout, per),
        revolve_vertex_slot(slice_index, b, n_slices, layout, per),
        revolve_vertex_slot(slice_index, v, n_slices, layout, per),
        end == 1,
    )


@wp.func
def write_cap_face(
    cap_faces: wp.array[wp.int32],
    t: wp.int32,
    end: wp.int32,
    far_offset: wp.int32,
    flip: wp.bool,
    row_base: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    # One face of either cap of an extruded or swept solid, into row
    # ``row_base + end * n_cap + t``. End 0 is the near cap -- no vertex offset, winding reversed so
    # its normals point outward -- and end 1 the far cap, shifted by ``far_offset`` and keeping the
    # input winding. ``flip`` reverses the input triangulation first (trimesh's ``np.fliplr``), so
    # it swaps which of the two ends is reversed. Shared by ``extrude_faces`` and ``sweep_mesh``,
    # whose callers write both caps into adjacent blocks of one buffer.
    n_cap = cap_faces.shape[0] // 3
    offset = wp.where(end == 0, wp.int32(0), far_offset)
    a, b, c = corner_triple(cap_faces, t)
    write_corner_triple_reversible(
        out_faces,
        row_base + end * n_cap + t,
        a + offset,
        b + offset,
        c + offset,
        (end == 0) != flip,
    )


@wp.func
def ring_boundary_edge(e: wp.int32, n: wp.int32) -> tuple[wp.int32, wp.int32]:
    # Boundary edge ``e`` of a full ``n - 2`` triangulation of an ``n``-vertex ring, as a directed
    # pair wound like the triangulation. Every triangle an ear clip or a fan emits is ``(left, i,
    # right)`` in ring order, so the boundary is exactly the ring edges ``(i, i + 1 mod n)``. They
    # are listed in the order ``boundary.oriented_boundary_edges`` returns them -- ascending by
    # ``(max, min)`` of the unordered pair -- so a caller that knows its triangulation is full
    # emits the same wall rows the derived boundary would.
    a = e
    b = e + 1
    if e == n - 2:
        a = n - 1
        b = wp.int32(0)
    elif e == n - 1:
        a = n - 2
        b = n - 1
    return a, b


@wp.func
def wall_edge(
    boundary: wp.array2d[wp.int32], e: wp.int32, n: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # Wall edge ``e``: row ``e`` of a derived boundary table, or -- when the caller passed none (a
    # null array reads shape 0) because its triangulation is a full triangulation of an ``n``-ring
    # -- ``ring_boundary_edge``. A caller with a derived table of zero rows launches no wall rows,
    # so the two cases never meet.
    if boundary.shape[0] == 0:
        return ring_boundary_edge(e, n)
    return boundary[e, 0], boundary[e, 1]


@wp.kernel
def lift_layers_and_signed_area(
    vertices: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    height: wp.float32,
    out_vertices: wp.array[wp.vec3],
    out_area: wp.array[wp.float32],
) -> None:
    # Both vertex layers of an extrusion -- ``z = 0`` into the first ``n`` slots, ``z = height``
    # into the next ``n`` -- and, in the same pass, twice the triangulation's total signed area
    # into ``out_area[0]``, whose sign decides on the device whether ``extrude_faces`` re-winds the
    # triangulation to agree with the sign of the extrusion. One element per lane over
    # ``max(n, n_faces)``: the kernel writes per vertex, so it keeps one tile per block (the fold
    # would collapse the grid) and commits one atomic per block. Lane-strided by ``wp.block_dim()``
    # so the CPU device's single lane covers the chunk.
    chunk, lane = wp.tid()
    n = vertices.shape[0]
    n_faces = faces.shape[0] // 3
    offset, remaining = tile_chunk(wp.max(n, n_faces), chunk, TILE_1D)
    if remaining <= 0:
        return
    remaining = wp.min(remaining, TILE_1D)
    area = wp.float32(0.0)
    for k in range(lane, remaining, wp.block_dim()):
        i = offset + k
        if i < n:
            out_vertices[i] = lift_vec2(vertices[i], wp.float32(0.0))
            out_vertices[n + i] = lift_vec2(vertices[i], height)
        if i < n_faces:
            p0, p1, p2 = face_vertices(vertices, faces, i)
            area += orient2d(p0, p1, p2)
    total = block_sum(area)
    if lane == 0:
        wp.atomic_add(out_area, 0, total)


@wp.kernel
def extrude_faces(
    faces: wp.array[wp.int32],
    boundary: wp.array2d[wp.int32],
    stride: wp.int32,
    height_negative: wp.bool,
    area: wp.array[wp.float32],
    out_faces: wp.array[wp.int32],
) -> None:
    # Every face of an extrusion in one launch: rows ``[0, 2 n_faces)`` are the two caps
    # (``write_cap_face``) and each row past them two wall triangles bridging boundary edge
    # ``(a, b)`` between the bottom layer (``a, b``) and the top (``a + stride, b + stride``).
    # trimesh builds the walls from a 4-vertex soup per edge and relies on its vertex merge to fuse
    # them onto the caps; indexing the caps directly makes the result watertight with no merge.
    #
    # The triangulation is re-wound when its signed area (``lift_layers_and_signed_area``) and the
    # height disagree in sign, so both caps and the walls face outward. A re-wound triangulation's
    # boundary is the same edges reversed, which is the swap below.
    r = wp.int32(wp.tid())
    n_faces = faces.shape[0] // 3
    flip = (area[0] < wp.float32(0.0)) != height_negative
    if r < 2 * n_faces:
        write_cap_face(faces, r % n_faces, r // n_faces, stride, flip, 0, out_faces)
        return
    e = r - 2 * n_faces
    a, b = wall_edge(boundary, e, stride)
    if flip:
        a, b = b, a
    base = 2 * n_faces * 3 + e * 6
    out_faces[base + 0] = b + stride
    out_faces[base + 1] = a + stride
    out_faces[base + 2] = b
    out_faces[base + 3] = b
    out_faces[base + 4] = a + stride
    out_faces[base + 5] = a


@wp.func
def path_tangent(path: wp.array[wp.vec3], i: wp.int32) -> wp.vec3:
    # Unit vector of path segment i -> i + 1.
    return wp.normalize(segment_displacement(path, i))


@wp.func
def snap_spherical(value: wp.float32) -> wp.float32:
    if wp.abs(value) < SPHERICAL_SNAP:
        return 0.0
    return value


@wp.func
def sweep_transform(
    path: wp.array[wp.vec3], angles: wp.array[wp.float32], connect_closed: wp.bool, i: wp.int32
) -> wp.mat44:
    # The rotation taking Z+ onto normals[i], pre-rolled by angles[i], with path[i] as origin.
    # Unrolled by trimesh from inv(Rz(roll) @ Rx(phi) @ Rz(pi/2 - theta)), so it is the identity
    # for a Z+ normal and needs no matrix inverse at runtime.
    # One plane normal per path vertex, formed here rather than in a buffer of its own: the end
    # planes lie along their single adjacent segment, interior planes bisect the two. trimesh
    # unitizes the sum rather than halving it because opposing segments can cancel.
    last = path.shape[0] - 1
    normal = wp.vec3(0.0, 0.0, 0.0)
    if i == 0:
        normal = path_tangent(path, 0)
        if connect_closed:
            # A closed path averages the first and last planes so the seam has one frame.
            normal = wp.normalize(normal + path_tangent(path, last - 1))
    elif i == last:
        normal = path_tangent(path, last - 1)
    else:
        normal = wp.normalize(path_tangent(path, i) + path_tangent(path, i - 1))
    # A degenerate (near-zero) normal has no direction to convert -- two consecutive path tangents
    # cancelling at a sharp path reversal, which the branch above can produce -- so leave both
    # angles at the Z+-identity zero rather than computing one, matching trimesh's
    # ``vector_to_spherical`` (``unitize(..., check_valid=True)`` marks such a row invalid and its
    # spherical angles stay at their zeroed default). Without this, ``wp.acos(0.0) == pi/2`` here
    # would instead rotate local +Z onto +X -- an arbitrary ~90 degree twist, not the identity a
    # degenerate plane should fall back to. Initialized before the branch, per the kernel-scope
    # conditional-scoping rule.
    theta = wp.float32(0.0)
    phi = wp.float32(0.0)
    if wp.length(normal) > TOLERANCE_ZERO_CONSTANT:
        theta = wp.atan2(snap_spherical(normal[1]), snap_spherical(normal[0]))
        phi = wp.acos(snap_spherical(normal[2]))
    cos_theta, sin_theta = wp.cos(theta), wp.sin(theta)
    cos_phi, sin_phi = wp.cos(phi), wp.sin(phi)
    # A null ``angles`` (shape 0) is no roll.
    roll = wp.float32(0.0)
    if angles.shape[0] > 0:
        roll = angles[i]
    cos_roll, sin_roll = wp.cos(roll), wp.sin(roll)
    origin = path[i]
    return wp.mat44(
        -sin_roll * cos_phi * cos_theta + sin_theta * cos_roll,
        sin_roll * sin_theta + cos_phi * cos_roll * cos_theta,
        sin_phi * cos_theta,
        origin[0],
        -sin_roll * sin_theta * cos_phi - cos_roll * cos_theta,
        -sin_roll * cos_theta + sin_theta * cos_phi * cos_roll,
        sin_phi * sin_theta,
        origin[1],
        sin_phi * sin_roll,
        -sin_phi * cos_roll,
        cos_phi,
        origin[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


@wp.kernel
def sweep_mesh(
    ring: wp.array[wp.vec2],
    path: wp.array[wp.vec3],
    angles: wp.array[wp.float32],
    connect_closed: wp.bool,
    boundary: wp.array2d[wp.int32],
    cap_faces: wp.array[wp.int32],
    n_slices: wp.int32,
    n_boundary: wp.int32,
    n_vertices: wp.int32,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # A whole swept solid in one launch, over ``n_vertices + n_slices * n_boundary + 2 * n_cap``
    # rows. The first ``n_vertices`` place one ring vertex each at its slice's frame
    # (``sweep_transform``, evaluated per vertex rather than read from a per-slice table: a few
    # trigonometric calls against a launch and a buffer). The next are two wall triangles per
    # boundary edge per slice, bridging slice ``s`` to ``s + 1`` -- the modulus wraps the final
    # slice onto slice 0 when the path is closed and connected and is a no-op otherwise -- and the
    # last are the two caps (``write_cap_face``, far cap at the last slice).
    r = wp.int32(wp.tid())
    stride = ring.shape[0]
    if r < n_vertices:
        s = r // stride
        out_vertices[r] = wp.transform_point(
            sweep_transform(path, angles, connect_closed, s),
            lift_vec2(ring[r % stride], wp.float32(0.0)),
        )
        return
    w = r - n_vertices
    if w < n_slices * n_boundary:
        s = w // n_boundary
        e = w % n_boundary
        a, b = wall_edge(boundary, e, stride)
        offset = s * stride
        a = a + offset
        b = b + offset
        base = w * 6
        out_faces[base + 0] = a % n_vertices
        out_faces[base + 1] = b % n_vertices
        out_faces[base + 2] = (a + stride) % n_vertices
        out_faces[base + 3] = (b + stride) % n_vertices
        out_faces[base + 4] = (a + stride) % n_vertices
        out_faces[base + 5] = b % n_vertices
        return
    c = w - n_slices * n_boundary
    n_cap = cap_faces.shape[0] // 3
    far = stride * n_slices
    write_cap_face(
        cap_faces, c % n_cap, c // n_cap, far, False, 2 * n_slices * n_boundary, out_faces
    )


@wp.func
def write_prism_face(
    slot: wp.int32,
    offset: wp.int32,
    a: wp.int32,
    b: wp.int32,
    c: wp.int32,
    flip: wp.bool,
    out_faces: wp.array[wp.int32],
) -> None:
    # One triangle of a prism's 8-face template, offset into its own 6-vertex block. `flip`
    # reverses the winding for prisms whose source triangle faces the plane (trimesh's
    # ``f_seq[cross > 0] = np.fliplr(f)``).
    write_corner_triple_reversible(out_faces, slot, offset + a, offset + b, offset + c, flip)


@wp.kernel
def truncated_prism_geometry(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    transform: wp.mat44,
    inverse: wp.mat44,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # One independent watertight prism per input triangle: the triangle itself, its projection onto
    # the truncation plane, and the six side triangles bridging them. Vertices 0-2 are the source
    # triangle and 3-5 its projection.
    f = wp.int32(wp.tid())
    v0, v1, v2 = face_vertices(vertices, faces, f)
    t0 = wp.transform_point(transform, v0)
    t1 = wp.transform_point(transform, v1)
    t2 = wp.transform_point(transform, v2)

    base = f * 6
    out_vertices[base + 0] = v0
    out_vertices[base + 1] = v1
    out_vertices[base + 2] = v2
    out_vertices[base + 3] = wp.transform_point(inverse, wp.vec3(t0[0], t0[1], 0.0))
    out_vertices[base + 4] = wp.transform_point(inverse, wp.vec3(t1[0], t1[1], 0.0))
    out_vertices[base + 5] = wp.transform_point(inverse, wp.vec3(t2[0], t2[1], 0.0))

    flip = wp.cross(t1 - t0, t2 - t0)[2] > 0.0
    slot = f * 8
    write_prism_face(slot + 0, base, 2, 1, 0, flip, out_faces)
    write_prism_face(slot + 1, base, 3, 4, 5, flip, out_faces)
    write_prism_face(slot + 2, base, 0, 1, 4, flip, out_faces)
    write_prism_face(slot + 3, base, 1, 2, 5, flip, out_faces)
    write_prism_face(slot + 4, base, 2, 0, 3, flip, out_faces)
    write_prism_face(slot + 5, base, 4, 3, 0, flip, out_faces)
    write_prism_face(slot + 6, base, 5, 4, 1, flip, out_faces)
    write_prism_face(slot + 7, base, 3, 5, 2, flip, out_faces)


@wp.kernel
def random_soup_vertices(seed: wp.int32, out_vertices: wp.array[wp.vec3]) -> None:
    i = wp.int32(wp.tid())
    state = wp.rand_init(seed, i)
    out_vertices[i] = wp.vec3(wp.randf(state) - 0.5, wp.randf(state) - 0.5, wp.randf(state) - 0.5)


# --- parametric surfaces -----------------------------------------------------------------
#
# One @wp.func per analytic surface, each the map VTK's ``vtkParametric*::Evaluate`` computes and in
# VTK's own frame (several are an x/y swap or a z flip away from the textbook form, which is folded
# into the expressions here). Verified against ``Evaluate`` over the whole sampled lattice: the
# largest disagreement is 2.5e-06 on a surface of scale 13.5, i.e. float32 rounding. The dispatch
# below is a warp-uniform branch over an int kind, so the whole family compiles into one module --
# ``wp.launch`` cannot take a ``wp.Function`` argument, and a kernel factory per surface would build
# eighteen kernels for one ``Literal``.

SURFACE_BOHEMIAN_DOME = wp.constant(wp.int32(0))
SURFACE_BOUR = wp.constant(wp.int32(1))
SURFACE_BOY = wp.constant(wp.int32(2))
SURFACE_CATALAN_MINIMAL = wp.constant(wp.int32(3))
SURFACE_CONIC_SPIRAL = wp.constant(wp.int32(4))
SURFACE_CROSS_CAP = wp.constant(wp.int32(5))
SURFACE_DINI = wp.constant(wp.int32(6))
SURFACE_ENNEPER = wp.constant(wp.int32(7))
SURFACE_FIGURE8_KLEIN = wp.constant(wp.int32(8))
SURFACE_HENNEBERG = wp.constant(wp.int32(9))
SURFACE_KLEIN = wp.constant(wp.int32(10))
SURFACE_KUEN = wp.constant(wp.int32(11))
SURFACE_MOBIUS = wp.constant(wp.int32(12))
SURFACE_PLUCKER_CONOID = wp.constant(wp.int32(13))
SURFACE_PSEUDOSPHERE = wp.constant(wp.int32(14))
SURFACE_ROMAN = wp.constant(wp.int32(15))
SURFACE_SUPER_ELLIPSOID = wp.constant(wp.int32(16))
SURFACE_SUPER_TOROID = wp.constant(wp.int32(17))

# ``vtkParametricKuen::DeltaV0``: the v it substitutes for the v = 0 row, where ``log(tan(v / 2))``
# is -inf. Its default, and the value the surface "has the best appearance with".
KUEN_DELTA_V0 = wp.constant(wp.float32(0.05))


@wp.func
def signed_power(value: wp.float32, exponent: wp.float32) -> wp.float32:
    """``sign(value) * |value| ** exponent``, the superquadric shape function."""
    return wp.sign(value) * wp.pow(wp.abs(value), exponent)


@wp.func
def surface_bohemian_dome(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(0.5 * wp.cos(u), 1.5 * wp.cos(v) + 0.5 * wp.sin(u), wp.sin(v))


@wp.func
def surface_bour(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(
        u * wp.cos(v) - 0.5 * u * u * wp.cos(2.0 * v),
        -u * wp.sin(v) - 0.5 * u * u * wp.sin(2.0 * v),
        4.0 / 3.0 * wp.pow(u, 1.5) * wp.cos(1.5 * v),
    )


@wp.func
def surface_boy(u: wp.float32, v: wp.float32) -> wp.vec3:
    # VTK evaluates a *polynomial* Steiner-type immersion of the unit-sphere point, not the rational
    # Apery form -- there is no denominator, and ``ZScale`` (0.125) scales only the third component.
    a = wp.cos(u) * wp.sin(v)
    b = wp.sin(u) * wp.sin(v)
    c = wp.cos(v)
    s = a + b + c
    return wp.vec3(
        0.5
        * (
            2.0 * a * a
            - b * b
            - c * c
            + 2.0 * b * c * (b * b - c * c)
            + c * a * (a * a - c * c)
            + a * b * (b * b - a * a)
        ),
        0.5 * SQRT3 * (b * b - c * c + c * a * (c * c - a * a) + a * b * (b * b - a * a)),
        0.125 * s * (s * s * s + 4.0 * (b - a) * (c - b) * (a - c)),
    )


@wp.func
def surface_catalan_minimal(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(
        u - wp.sin(u) * wp.cosh(v),
        1.0 - wp.cos(u) * wp.cosh(v),
        4.0 * wp.sin(0.5 * u) * wp.sinh(0.5 * v),
    )


@wp.func
def surface_conic_spiral(u: wp.float32, v: wp.float32) -> wp.vec3:
    taper = 1.0 - v / (2.0 * PI_F)
    return wp.vec3(
        0.2 * taper * wp.cos(2.0 * v) * (1.0 + wp.cos(u)) + 0.1 * wp.cos(2.0 * v),
        0.2 * taper * wp.sin(2.0 * v) * (1.0 + wp.cos(u)) + 0.1 * wp.sin(2.0 * v),
        v / (2.0 * PI_F) + 0.2 * taper * wp.sin(u),
    )


@wp.func
def surface_cross_cap(u: wp.float32, v: wp.float32) -> wp.vec3:
    cu, su = wp.cos(u), wp.sin(u)
    cv, sv = wp.cos(v), wp.sin(v)
    return wp.vec3(cu * wp.sin(2.0 * v), su * wp.sin(2.0 * v), cv * cv - cu * cu * sv * sv)


@wp.func
def surface_dini(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(
        wp.cos(u) * wp.sin(v), wp.sin(u) * wp.sin(v), wp.cos(v) + wp.log(wp.tan(0.5 * v)) + 0.2 * u
    )


@wp.func
def surface_enneper(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(u - u * u * u / 3.0 + u * v * v, v - v * v * v / 3.0 + v * u * u, u * u - v * v)


@wp.func
def surface_figure8_klein(u: wp.float32, v: wp.float32) -> wp.vec3:
    half_u = 0.5 * u
    radial = 1.0 + wp.cos(half_u) * wp.sin(v) - 0.5 * wp.sin(half_u) * wp.sin(2.0 * v)
    return wp.vec3(
        radial * wp.cos(u),
        radial * wp.sin(u),
        wp.sin(half_u) * wp.sin(v) + 0.5 * wp.cos(half_u) * wp.sin(2.0 * v),
    )


@wp.func
def surface_henneberg(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(
        2.0 * wp.sinh(u) * wp.cos(v) - 2.0 / 3.0 * wp.sinh(3.0 * u) * wp.cos(3.0 * v),
        2.0 * wp.sinh(u) * wp.sin(v) + 2.0 / 3.0 * wp.sinh(3.0 * u) * wp.sin(3.0 * v),
        2.0 * wp.cosh(2.0 * u) * wp.cos(2.0 * v),
    )


@wp.func
def surface_klein(u: wp.float32, v: wp.float32) -> wp.vec3:
    cu, su = wp.cos(u), wp.sin(u)
    cv, sv = wp.cos(v), wp.sin(v)
    cu2 = cu * cu
    cu3 = cu2 * cu
    cu4 = cu2 * cu2
    cu5 = cu4 * cu
    cu6 = cu4 * cu2
    cu7 = cu6 * cu
    return wp.vec3(
        -2.0
        / 15.0
        * cu
        * (3.0 * cv - 30.0 * su + 90.0 * cu4 * su - 60.0 * cu6 * su + 5.0 * cu * cv * su),
        -1.0
        / 15.0
        * su
        * (
            3.0 * cv
            - 3.0 * cu2 * cv
            - 48.0 * cu4 * cv
            + 48.0 * cu6 * cv
            - 60.0 * su
            + 5.0 * cu * cv * su
            - 5.0 * cu3 * cv * su
            - 80.0 * cu5 * cv * su
            + 80.0 * cu7 * cv * su
        ),
        2.0 / 15.0 * (3.0 + 5.0 * cu * su) * sv,
    )


@wp.func
def surface_kuen(u: wp.float32, v: wp.float32) -> wp.vec3:
    # Both ends of the v domain are singular and VTK names a value at each: it substitutes
    # ``DeltaV0`` for v = 0, and reports v = pi as the pole (0, 0, -1) rather than the +inf the
    # limit of ``log(tan(v / 2))`` actually goes to. Both rows are sampled, so both are replicated
    # here -- and the second guard doubles as the float32 one, since ``tan(v / 2)`` turns negative
    # a single ulp past pi and ``log`` of it is NaN.
    v_safe = v
    if v_safe <= 0.0:
        v_safe = KUEN_DELTA_V0
    sv = wp.sin(v_safe)
    position = wp.vec3(0.0, 0.0, -1.0)
    if sv > 0.0:
        denominator = 1.0 + u * u * sv * sv
        # VTK's frame swaps the first two components relative to the textbook form.
        position = wp.vec3(
            2.0 * (wp.sin(u) - u * wp.cos(u)) * sv / denominator,
            2.0 * (wp.cos(u) + u * wp.sin(u)) * sv / denominator,
            wp.log(wp.tan(0.5 * v_safe)) + 2.0 * wp.cos(v_safe) / denominator,
        )
    return position


@wp.func
def surface_mobius(u: wp.float32, v: wp.float32) -> wp.vec3:
    # VTK does not halve v, and the half-angle roles are the opposite of the usual writing:
    # the radius carries sin(u/2) and the height cos(u/2). First two components swapped.
    radial = 1.0 - v * wp.sin(0.5 * u)
    return wp.vec3(radial * wp.sin(u), radial * wp.cos(u), v * wp.cos(0.5 * u))


@wp.func
def surface_plucker_conoid(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(u * wp.sin(v), u * wp.cos(v), wp.sin(2.0 * v))


@wp.func
def surface_pseudosphere(u: wp.float32, v: wp.float32) -> wp.vec3:
    sech = 1.0 / wp.cosh(u)
    return wp.vec3(sech * wp.cos(v), sech * wp.sin(v), u - wp.tanh(u))


@wp.func
def surface_roman(u: wp.float32, v: wp.float32) -> wp.vec3:
    cv = wp.cos(v)
    s2v = wp.sin(2.0 * v)
    return wp.vec3(0.5 * cv * cv * wp.sin(2.0 * u), 0.5 * wp.sin(u) * s2v, 0.5 * wp.cos(u) * s2v)


@wp.func
def surface_super_ellipsoid(
    u: wp.float32, v: wp.float32, n1: wp.float32, n2: wp.float32
) -> wp.vec3:
    cv = signed_power(wp.cos(v), n1)
    sv = signed_power(wp.sin(v), n1)
    cu = signed_power(wp.cos(u), n2)
    su = signed_power(wp.sin(u), n2)
    # First two components swapped, matching VTK's frame.
    return wp.vec3(cv * su, cv * cu, sv)


@wp.func
def surface_super_toroid(u: wp.float32, v: wp.float32, n1: wp.float32, n2: wp.float32) -> wp.vec3:
    cu = signed_power(wp.cos(u), n2)
    su = signed_power(wp.sin(u), n2)
    cv = signed_power(wp.cos(v), n1)
    sv = signed_power(wp.sin(v), n1)
    return wp.vec3(su * (1.0 + 0.5 * cv), cu * (1.0 + 0.5 * cv), 0.5 * sv)


# The gluing rule of one parametric surface as a warp-uniform bitmask, so one kernel serves all
# sixteen surfaces with one launch argument rather than eight.
GLUE_U_WRAP = wp.constant(wp.int32(1))
GLUE_U_TWIST = wp.constant(wp.int32(2))
GLUE_V_WRAP = wp.constant(wp.int32(4))
GLUE_V_TWIST = wp.constant(wp.int32(8))
GLUE_POLE_J_LO = wp.constant(wp.int32(16))
GLUE_POLE_J_HI = wp.constant(wp.int32(32))
GLUE_POLE_I_LO = wp.constant(wp.int32(64))
GLUE_POLE_I_HI = wp.constant(wp.int32(128))


@wp.func
def glued(gluing: wp.int32, flag: wp.int32) -> wp.bool:
    return (gluing & flag) != 0


@wp.func
def parametric_canonical_key(
    flat: wp.int32, n_u: wp.int32, n_v: wp.int32, gluing: wp.int32
) -> wp.int32:
    """
    Identify lattice sample ``flat`` with the sample that represents its output vertex.

    Returns ``i_canonical * n_v + j_canonical`` for the sample at C-order index ``flat``, so that
    two samples the surface glues together get the same key. This is the gluing rule of
    ``creation._parametric_lattice_host`` moved to the device verbatim, including the two
    orderings that are easy to get wrong: the ``u``-seam re-canonicalisation after a ``v``-twist
    is *unmasked* (it applies to every sample, not only the seam), and all four pole masks are
    snapshots of the post-wrap state, taken before any of them collapses anything.

    The pole anchors need no arguments because they are fixed by the rule itself -- a pole on
    ``j == 0`` or ``i == 0`` collapses to ``(0, 0)``, one on ``j == n_v - 1`` to ``(0, n_v - 1)``,
    one on ``i == n_u - 1`` to ``(n_u - 1, 0)``. A key need not be its own key (a sample on two
    pole rows collapses twice), so the vertex set is the keys' *image*, not their fixed points.
    """
    i_c = flat // n_v
    j_c = flat % n_v
    u_wrap = glued(gluing, GLUE_U_WRAP)
    if u_wrap and i_c == n_u - 1:
        if glued(gluing, GLUE_U_TWIST):
            j_c = n_v - 1 - j_c
        i_c = 0
    if glued(gluing, GLUE_V_WRAP):
        if j_c == n_v - 1:
            if glued(gluing, GLUE_V_TWIST):
                i_c = n_u - 1 - i_c
            j_c = 0
        # Unmasked, as in the host form: a v-twist can send any sample back onto the u seam.
        if u_wrap and i_c == n_u - 1:
            i_c = 0

    # Masks first, collapses after -- the host builds the whole pole list before applying any of it.
    on_j_lo = glued(gluing, GLUE_POLE_J_LO) and j_c == 0
    on_j_hi = glued(gluing, GLUE_POLE_J_HI) and j_c == n_v - 1
    on_i_lo = glued(gluing, GLUE_POLE_I_LO) and i_c == 0
    on_i_hi = glued(gluing, GLUE_POLE_I_HI) and i_c == n_u - 1
    if on_j_lo:
        i_c = 0
        j_c = 0
    if on_j_hi:
        i_c = 0
        j_c = n_v - 1
    if on_i_lo:
        i_c = 0
        j_c = 0
    if on_i_hi:
        i_c = n_u - 1
        j_c = 0
    return i_c * n_v + j_c


@wp.func
def parametric_triangle_keys(
    t: wp.int32, n_u: wp.int32, n_v: wp.int32, gluing: wp.int32
) -> wp.vec3i:
    # The canonical keys of lattice triangle ``t``'s corners. Two triangles per lattice cell,
    # wound against the ``(u, v)`` frame: triangle ``t`` below the cell count is that cell's
    # ``(a, c, b)`` and the rest are ``(a, d, c)``, the order the host's two stacked
    # ``column_stack`` blocks produce. Keys and vertex indices are in bijection, so a triangle with
    # two equal keys is the degenerate one a pole cell carries.
    n_cells = (n_u - 1) * (n_v - 1)
    cell = t
    if t >= n_cells:
        cell = t - n_cells
    i = cell // (n_v - 1)
    j = cell % (n_v - 1)
    key_a = parametric_canonical_key(i * n_v + j, n_u, n_v, gluing)
    key_c = parametric_canonical_key((i + 1) * n_v + j + 1, n_u, n_v, gluing)
    if t >= n_cells:
        return wp.vec3i(key_a, parametric_canonical_key(i * n_v + j + 1, n_u, n_v, gluing), key_c)
    return wp.vec3i(key_a, key_c, parametric_canonical_key((i + 1) * n_v + j, n_u, n_v, gluing))


@wp.kernel
def parametric_lattice_flags(
    n_u: wp.int32, n_v: wp.int32, gluing: wp.int32, out_flags: wp.array[wp.int32]
) -> None:
    # The first half of the device lattice, over ``n_lattice + n_triangles`` rows into one
    # zero-filled buffer the caller then scans once. A lattice row marks its canonical key, so the
    # scan's prefix over the lattice numbers the vertices in ascending key order -- the order
    # ``numpy.unique`` gives the host path -- and a triangle row flags whether it survives.
    r = wp.int32(wp.tid())
    n_lattice = n_u * n_v
    if r < n_lattice:
        out_flags[parametric_canonical_key(r, n_u, n_v, gluing)] = 1
        return
    keys = parametric_triangle_keys(r - n_lattice, n_u, n_v, gluing)
    out_flags[r] = wp.where(keys[0] != keys[1] and keys[1] != keys[2] and keys[2] != keys[0], 1, 0)


@wp.kernel
def parametric_lattice_emit(
    n_u: wp.int32,
    n_v: wp.int32,
    gluing: wp.int32,
    scan: wp.array[wp.int32],
    out_first: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # The second half, over the same rows and the inclusive scan of ``parametric_lattice_flags``.
    # A lattice row folds itself into its vertex's ``out_first`` (seeded with ``INT32_MAX``): the
    # lowest lattice index in each group, the ``return_index`` of the host's ``numpy.unique``. A
    # surviving triangle row writes its corners' vertex numbers at its rank among the survivors,
    # which keeps the host's row order.
    r = wp.int32(wp.tid())
    n_lattice = n_u * n_v
    if r < n_lattice:
        vertex = scan[parametric_canonical_key(r, n_u, n_v, gluing)] - 1
        wp.atomic_min(out_first, vertex, r)
        return
    if scan[r] == scan[r - 1]:
        return
    keys = parametric_triangle_keys(r - n_lattice, n_u, n_v, gluing)
    row = scan[r] - scan[n_lattice - 1] - 1
    for k in range(3):
        out_faces[row * 3 + k] = scan[keys[k]] - 1


@wp.func
def parametric_position(
    kind: wp.int32, u: wp.float32, v: wp.float32, n1: wp.float32, n2: wp.float32
) -> wp.vec3:
    """Evaluate surface ``kind`` at ``(u, v)``; ``n1`` / ``n2`` are for the superquadrics only."""
    position = wp.vec3()
    if kind == SURFACE_BOHEMIAN_DOME:
        position = surface_bohemian_dome(u, v)
    elif kind == SURFACE_BOUR:
        position = surface_bour(u, v)
    elif kind == SURFACE_BOY:
        position = surface_boy(u, v)
    elif kind == SURFACE_CATALAN_MINIMAL:
        position = surface_catalan_minimal(u, v)
    elif kind == SURFACE_CONIC_SPIRAL:
        position = surface_conic_spiral(u, v)
    elif kind == SURFACE_CROSS_CAP:
        position = surface_cross_cap(u, v)
    elif kind == SURFACE_DINI:
        position = surface_dini(u, v)
    elif kind == SURFACE_ENNEPER:
        position = surface_enneper(u, v)
    elif kind == SURFACE_FIGURE8_KLEIN:
        position = surface_figure8_klein(u, v)
    elif kind == SURFACE_HENNEBERG:
        position = surface_henneberg(u, v)
    elif kind == SURFACE_KLEIN:
        position = surface_klein(u, v)
    elif kind == SURFACE_KUEN:
        position = surface_kuen(u, v)
    elif kind == SURFACE_MOBIUS:
        position = surface_mobius(u, v)
    elif kind == SURFACE_PLUCKER_CONOID:
        position = surface_plucker_conoid(u, v)
    elif kind == SURFACE_PSEUDOSPHERE:
        position = surface_pseudosphere(u, v)
    elif kind == SURFACE_ROMAN:
        position = surface_roman(u, v)
    elif kind == SURFACE_SUPER_ELLIPSOID:
        position = surface_super_ellipsoid(u, v, n1, n2)
    elif kind == SURFACE_SUPER_TOROID:
        position = surface_super_toroid(u, v, n1, n2)
    return position


@wp.func
def parametric_sample(
    first: wp.array[wp.int32], tables: wp.array[wp.float32], n_v: wp.int32, k: wp.int32
) -> wp.vec2:
    # The ``(u, v)`` parameters of output vertex ``k``. ``first`` is the flat lattice index of the
    # sample chosen to represent it, so its lattice position is one divmod and the parameters are
    # two gathers from ``tables`` -- the host's ``linspace`` over ``u`` then over ``v``, unchanged:
    # both ends of the domain have to be hit exactly, because several of these maps are singular
    # one ulp outside their rectangle.
    flat = first[k]
    n_u = tables.shape[0] - n_v
    return wp.vec2(tables[flat // n_v], tables[n_u + flat % n_v])


@wp.kernel
def parametric_vertices(
    kind: wp.int32,
    first: wp.array[wp.int32],
    tables: wp.array[wp.float32],
    n_v: wp.int32,
    n1: wp.float32,
    n2: wp.float32,
    out_vertices: wp.array[wp.vec3],
) -> None:
    k = wp.int32(wp.tid())
    uv = parametric_sample(first, tables, n_v, k)
    out_vertices[k] = parametric_position(kind, uv[0], uv[1], n1, n2)


@wp.kernel
def random_hills_vertices(
    amplitude: wp.float32,
    x_variance: wp.float32,
    y_variance: wp.float32,
    hill_centers: wp.array[wp.vec2],
    first: wp.array[wp.int32],
    tables: wp.array[wp.float32],
    n_v: wp.int32,
    out_vertices: wp.array[wp.vec3],
) -> None:
    t = wp.int32(wp.tid())
    uv = parametric_sample(first, tables, n_v, t)
    x = uv[0]
    y = uv[1]
    height = wp.float32(0.0)
    for h in range(hill_centers.shape[0]):
        offset = uv - hill_centers[h]
        height += wp.exp(
            -0.5 * (offset[0] * offset[0] / x_variance + offset[1] * offset[1] / y_variance)
        )
    out_vertices[t] = wp.vec3(x, y, amplitude * height)


@wp.func
def revolve_uniform_slot(
    column: wp.int32,
    slice_index: wp.int32,
    n_columns: wp.int32,
    n_slices: wp.int32,
    ring_base: wp.int32,
    axis_first: wp.int32,
    axis_last: wp.int32,
) -> wp.int32:
    # Output slot of one (column, slice) pair under ``revolve_uniform``'s regular layout: an
    # on-axis end column is a single shared slot, every other column a full ring of ``n_slices``.
    if column == 0 and axis_first != 0:
        return 0
    if column == n_columns - 1 and axis_last != 0:
        return ring_base + (n_columns - 1 - ring_base) * n_slices
    return ring_base + (column - ring_base) * n_slices + slice_index


@wp.kernel
def revolve_uniform(
    profile: wp.array[wp.vec2],
    n_slices: wp.int32,
    angle: wp.float32,
    axis_first: wp.int32,
    axis_last: wp.int32,
    wrap: wp.int32,
    first_segment_faces: wp.int32,
    faces_per_slice: wp.int32,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # A whole solid of revolution in one launch, for the profiles whose layout is *regular*: every
    # column is a full ring except an on-axis point at one or both ends, and every profile segment
    # contributes two triangles per slice except the ones touching the axis.
    #
    # ``creation.revolve`` is general over an arbitrary profile and pays for that on the host --
    # it reads the profile back off the device, computes a per-column slot table and a surviving
    # triangle list in NumPy, and uploads three more arrays. Every shape that meets the regularity
    # test can address the same layout arithmetically instead, which is what this kernel does. The
    # caller checks regularity against ``revolve``'s own filter and falls back to it when the test
    # fails, so this is a fast path rather than a second definition of the geometry.
    #
    # Launched ``dim=(n_slices, n_columns)`` over the *deduplicated* columns: a closed profile
    # repeats its first point last, and ``wrap`` says so rather than the profile carrying the
    # duplicate.
    #
    # ``first_segment_faces`` and ``faces_per_slice`` are the caller's own count of what survives,
    # so a thread can find its slice's block and its segment's place in it without a prefix scan:
    # only the two end segments can be short, so every interior segment contributes exactly two.
    s, c = wp.tid()
    n_columns = profile.shape[0]
    ring_base = axis_first
    on_axis_c = (c == 0 and axis_first != 0) or (c == n_columns - 1 and axis_last != 0)

    # --- the vertex this thread owns ---------------------------------------------------------
    # An on-axis column is one shared slot, written by slice 0 alone: ``revolution_point`` returns
    # the same value for every slice there, and a single writer keeps it deterministic.
    here = revolve_uniform_slot(c, s, n_columns, n_slices, ring_base, axis_first, axis_last)
    if not on_axis_c or s == 0:
        point = profile[c]
        out_vertices[here] = revolution_point(point[0], point[1], s, n_slices, angle)

    # --- the faces of the segment leaving this column ------------------------------------------
    n_segments = n_columns - 1 + wrap
    if c >= n_segments:
        return
    nxt = (c + 1) % n_columns
    on_axis_next = (nxt == 0 and axis_first != 0) or (nxt == n_columns - 1 and axis_last != 0)

    slice_next = (s + 1) % n_slices
    here_next = revolve_uniform_slot(
        c, slice_next, n_columns, n_slices, ring_base, axis_first, axis_last
    )
    there = revolve_uniform_slot(nxt, s, n_columns, n_slices, ring_base, axis_first, axis_last)
    there_next = revolve_uniform_slot(
        nxt, slice_next, n_columns, n_slices, ring_base, axis_first, axis_last
    )

    # Only segment 0 can be short at the front, so every later segment starts two faces on from
    # where the one before it did.
    within = wp.int32(0)
    if c > 0:
        within = first_segment_faces + 2 * (c - 1)
    base = (s * faces_per_slice + within) * 3
    if not on_axis_c:
        out_faces[base + 0] = here
        out_faces[base + 1] = here_next
        out_faces[base + 2] = there
        base = base + 3
    if not on_axis_next:
        out_faces[base + 0] = there
        out_faces[base + 1] = here_next
        out_faces[base + 2] = there_next


@wp.func
def cap_ring_start(ring: wp.int32) -> wp.int32:
    # First vertex slot of a spherical cap's concentric ring ``ring``, which carries ``6 * ring``
    # vertices. Ring 0 is the single apex and sits at slot 0; every later ring starts at the
    # running total ``1 + 3 * ring * (ring - 1)``, which is why ring 0 is not that formula's value
    # at zero -- it would read 1, the *first* rim vertex, and quietly wind every apex triangle
    # around its neighbour instead.
    return wp.where(ring == 0, 0, 1 + 3 * ring * (ring - 1))


@wp.func
def cap_ring_of(vertex: wp.int32) -> wp.int32:
    # Inverse of ``cap_ring_start`` for ``vertex >= 1``: the largest ``ring`` whose block starts at
    # or before ``vertex``. Solving ``3 r^2 - 3 r + 1 <= v`` gives
    # ``r <= (3 + sqrt(12 v - 3)) / 6``, and the two corrections below turn that float64 estimate
    # into the exact integer -- each runs
    # at most once, and they are what makes this safe at a ring count where the square root's last
    # bit could land either side of a block boundary.
    # Every literal carries its precision: a bare float literal is ``wp.float32`` in kernel scope
    # and mixing one into a ``float64`` expression is a parse error (CLAUDE.md section 1.2).
    estimate = (wp.float64(3.0) + wp.sqrt(wp.float64(12 * vertex - 3))) / wp.float64(6.0)
    ring = wp.int32(estimate)
    if ring < 1:
        ring = 1
    while cap_ring_start(ring + 1) <= vertex:
        ring = ring + 1
    while cap_ring_start(ring) > vertex:
        ring = ring - 1
    return ring


@wp.func
def cap_strip_of(face: wp.int32) -> wp.int32:
    # Which ring-to-ring strip owns triangle ``face``. Strip ``r`` joins ring ``r - 1`` to ring
    # ``r`` and contributes ``6 * r`` outward triangles followed by ``6 * (r - 1)`` inward ones, so
    # the strips end at ``6 * r^2`` and strip ``r`` starts at ``6 * (r - 1)^2``. Same shape as
    # ``cap_ring_of``: a float64 estimate and two at-most-once corrections.
    strip = wp.int32(wp.sqrt(wp.float64(face) / wp.float64(6.0))) + 1
    if strip < 1:
        strip = 1
    while 6 * strip * strip <= face:
        strip = strip + 1
    while 6 * (strip - 1) * (strip - 1) > face:
        strip = strip - 1
    return strip


@wp.func
def sphere_cap_vertex(
    v: wp.int32, n_rings: wp.int32, angle: wp.float64, radius: wp.float64
) -> wp.vec3:
    # Vertex ``v`` of the concentric-ring cap lattice. Ring ``r`` sits at polar angle
    # ``angle * r / n_rings`` and carries ``6 * r`` vertices evenly spaced in azimuth, so a thread
    # recovers its own ring from its slot rather than being told: the whole lattice is a closed
    # form in the vertex index, which is what lets it replace a host loop over rings whose cost was
    # quadratic in the ring count.
    #
    # float64 throughout and stored to ``wp.vec3``, matching the host build this replaces: the
    # rounding happens once, in the same place, at the store.
    if v == 0:
        return wp.vec3(0.0, 0.0, wp.float32(radius))
    ring = cap_ring_of(v)
    step = v - cap_ring_start(ring)

    theta = angle * wp.float64(ring) / wp.float64(n_rings)
    phi = wp.float64(2.0) * wp.float64(wp.PI) * wp.float64(step) / wp.float64(6 * ring)
    ring_radius = radius * wp.sin(theta)
    return wp.vec3(
        wp.float32(ring_radius * wp.cos(phi)),
        wp.float32(ring_radius * wp.sin(phi)),
        wp.float32(radius * wp.cos(theta)),
    )


@wp.func
def write_sphere_cap_face(f: wp.int32, out_faces: wp.array[wp.int32]) -> None:
    # Triangle ``f`` of the cap. Strip ``r`` stitches ring ``r - 1`` to ring ``r``, and because the
    # outer ring carries exactly one more vertex per sector than the inner one the strip is
    # ``6 * r`` outward-pointing triangles (one per outer edge) followed by ``6 * (r - 1)``
    # inward-pointing ones (one per inner edge), rather than an even fan. That is why the total
    # lands on exactly ``6 * n_rings ** 2``, and it is the order the buffer is written in, so a
    # thread's triangle is a closed form in its own index.
    strip = cap_strip_of(f)
    local = f - 6 * (strip - 1) * (strip - 1)

    outer_base = cap_ring_start(strip)
    inner_base = cap_ring_start(strip - 1)
    outer_count = 6 * strip
    # Ring 0 is the lone apex, not a ring of ``6 * 0`` vertices, so the inner ring of strip 1 has
    # one member and every sector's inner index wraps onto it.
    inner_count = 6 * (strip - 1)
    if strip == 1:
        inner_count = 1

    base = 3 * f
    if local < outer_count:
        sector = local // strip
        step = local % strip
        out_faces[base + 0] = outer_base + local
        out_faces[base + 1] = outer_base + (local + 1) % outer_count
        out_faces[base + 2] = inner_base + (sector * (strip - 1) + step) % inner_count
    else:
        inner = local - outer_count
        sector = inner // (strip - 1)
        step = inner % (strip - 1)
        out_faces[base + 0] = inner_base + inner
        out_faces[base + 1] = outer_base + (sector * strip + step + 1) % outer_count
        out_faces[base + 2] = inner_base + (inner + 1) % inner_count


@wp.kernel
def sphere_cap_mesh(
    n_rings: wp.int32,
    angle: wp.float64,
    radius: wp.float64,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # The whole cap in one launch over ``max(n_vertices, n_faces)`` threads: thread ``t`` writes
    # vertex ``t`` and triangle ``t`` where each exists. Neither buffer reads the other, and both
    # are closed form in the index, so the two were separate launches only by habit. The faces
    # outnumber the vertices from two rings on (``6 r^2`` against ``3 r^2 + 3 r + 1``), and a
    # single ring is the one size where the vertices are the longer buffer.
    t = wp.int32(wp.tid())
    if t < out_vertices.shape[0]:
        out_vertices[t] = sphere_cap_vertex(t, n_rings, angle, radius)
    if t < out_faces.shape[0] // 3:
        write_sphere_cap_face(t, out_faces)
