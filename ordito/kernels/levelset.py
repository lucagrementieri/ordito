import warp as wp
from warp.geometry import IsoSurfaceMarchingCubes

from ordito.kernels.array import ravel_index

wp.set_module_options({"enable_backward": False})


# ---------------------------------------------------------------------------------------------
# marching_cubes: the dense-lattice extraction, in two passes over the nodes
# ---------------------------------------------------------------------------------------------
#
# A port of ``warp.geometry.IsoSurfaceMarchingCubes.extract``'s algorithm (Apache-2.0): the same
# case tables, the same vertex numbering -- one vertex per crossing lattice edge, edges ordered by
# their lower node row-major and then by axis -- the same per-cell triangle order and the same
# interpolation arithmetic, so its output is Warp's bit for bit, vertex and face order included.
# What changed is the bookkeeping. Warp counts and emits vertices over a ``(nx, ny, nz, 3)`` edge
# grid and faces over the cell grid, with two zeroed ``3 * n_nodes`` int32 buffers, a scan of
# each count buffer and a ``(nx, ny, nz, 3)`` edge-to-vertex table the face pass reads back -- 44
# bytes of scratch a node, eleven times the field, and two scans and two host reads. Here one
# thread per node counts both its crossing edges and its cell's triangles into one ``vec2i``, one
# scan turns the counts into both offsets, and the emit pass finds an edge's vertex from its
# owner node's scanned count, recomputing which of that node's edges cross: 8 bytes a node, one
# scan, one read. The tables are Warp's public ``IsoSurfaceMarchingCubes`` attributes, packed into
# one array by ``levelset._marching_cubes_table``; the edge-owner layout is derived from them.

# Offsets into that packed table: the case-to-triangle ranges first (257 entries), then the
# triangles' local edge indices, then one packed entry per cube edge -- its owner corner's offset
# in bits 0-2 and its axis from bit 3 (``MARCHING_CUBES_EDGES``).
MC_TRI_BASE = len(IsoSurfaceMarchingCubes.CASE_TO_TRI_RANGE)
MC_EDGE_BASE = MC_TRI_BASE + len(IsoSurfaceMarchingCubes.TRI_LOCAL_INDICES)

_MC_CORNERS = IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS


def _packed_edge(first: int, second: int) -> int:
    """One cube edge packed as ``owner offset | axis << 3``: its lower corner and its axis."""
    a, b = _MC_CORNERS[first], _MC_CORNERS[second]
    owner = [min(a[c], b[c]) for c in range(3)]
    axis = next(c for c in range(3) if a[c] != b[c])
    return owner[0] | (owner[1] << 1) | (owner[2] << 2) | (axis << 3)


MARCHING_CUBES_EDGES = tuple(
    _packed_edge(*pair) for pair in IsoSurfaceMarchingCubes.EDGE_TO_CORNERS
)

MARCHING_CUBES_TABLE = (
    tuple(IsoSurfaceMarchingCubes.CASE_TO_TRI_RANGE)
    + tuple(IsoSurfaceMarchingCubes.TRI_LOCAL_INDICES)
    + MARCHING_CUBES_EDGES
)


@wp.func
def mc_edge_crosses(
    field: wp.array3d[wp.float32],
    iso: wp.float32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    axis: wp.int32,
) -> wp.int32:
    # 1 when the lattice edge from node ``(i, j, k)`` along ``axis`` exists and its ends straddle
    # ``iso`` (``>=`` on one end, ``<`` on the other: Warp's test, so a NaN end never crosses).
    io = i + wp.where(axis == 0, 1, 0)
    jo = j + wp.where(axis == 1, 1, 0)
    ko = k + wp.where(axis == 2, 1, 0)
    if io >= field.shape[0] or jo >= field.shape[1] or ko >= field.shape[2]:
        return 0
    here = field[i, j, k]
    there = field[io, jo, ko]
    return wp.where((here >= iso and there < iso) or (here < iso and there >= iso), 1, 0)


@wp.func
def mc_case_code(
    field: wp.array3d[wp.float32], iso: wp.float32, i: wp.int32, j: wp.int32, k: wp.int32
) -> wp.int32:
    # The cell at node ``(i, j, k)``'s 8-bit case: bit ``c`` set when corner ``c`` is at or above
    # ``iso``, corners in ``CUBE_CORNER_OFFSETS`` order.
    code = wp.int32(0)
    for c in range(8):
        value = field[
            i + wp.static(_MC_CORNERS[c][0]),
            j + wp.static(_MC_CORNERS[c][1]),
            k + wp.static(_MC_CORNERS[c][2]),
        ]
        if value >= iso:
            code += wp.static(1 << c)
    return code


@wp.func
def mc_cell_triangles(
    field: wp.array3d[wp.float32],
    iso: wp.float32,
    table: wp.array[wp.int32],
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
) -> wp.vec2i:
    # ``(first table slot, triangle count)`` of the cell whose lower corner is node ``(i, j, k)``;
    # a node on the lattice's upper face along any axis owns no cell and returns no triangles.
    if i + 1 >= field.shape[0] or j + 1 >= field.shape[1] or k + 1 >= field.shape[2]:
        return wp.vec2i(0, 0)
    code = mc_case_code(field, iso, i, j, k)
    start = table[code]
    return wp.vec2i(start, (table[code + 1] - start) // 3)


@wp.kernel
def marching_cubes_counts(
    field: wp.array3d[wp.float32],
    iso: wp.float32,
    table: wp.array[wp.int32],
    out_counts: wp.array[wp.vec2i],
) -> None:
    # Per node: how many of its three positive-axis edges cross ``iso`` (its vertices) and how many
    # triangles its cell emits. Scanned inclusively in place, the pair is both output offsets.
    i, j, k = wp.tid()
    vertices = (
        mc_edge_crosses(field, iso, i, j, k, 0)
        + mc_edge_crosses(field, iso, i, j, k, 1)
        + mc_edge_crosses(field, iso, i, j, k, 2)
    )
    cell = mc_cell_triangles(field, iso, table, i, j, k)
    out_counts[ravel_index(i, j, k, field.shape[1], field.shape[2])] = wp.vec2i(vertices, cell[1])


@wp.kernel
def marching_cubes_emit(
    field: wp.array3d[wp.float32],
    iso: wp.float32,
    lower: wp.vec3,
    delta: wp.vec3,
    table: wp.array[wp.int32],
    offsets: wp.array[wp.vec2i],
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # Writes node ``(i, j, k)``'s crossing-edge vertices and its cell's triangles at the slots the
    # inclusive scan of ``marching_cubes_counts`` gives (each count is subtracted back off its
    # inclusive total). The vertex arithmetic is Warp's ``extract_vertices_kernel``'s, statement for
    # statement, which is what keeps the positions bit-identical to it.
    i, j, k = wp.tid()
    ny = field.shape[1]
    nz = field.shape[2]
    ends = offsets[ravel_index(i, j, k, ny, nz)]
    crossing = wp.vec3i(
        mc_edge_crosses(field, iso, i, j, k, 0),
        mc_edge_crosses(field, iso, i, j, k, 1),
        mc_edge_crosses(field, iso, i, j, k, 2),
    )
    slot = ends[0] - (crossing[0] + crossing[1] + crossing[2])
    for axis in range(3):
        if crossing[axis] != 0:
            io = i + wp.where(axis == 0, 1, 0)
            jo = j + wp.where(axis == 1, 1, 0)
            ko = k + wp.where(axis == 2, 1, 0)
            here = field[i, j, k]
            there = field[io, jo, ko]
            t = (iso - here) / (there - here)
            t = wp.clamp(t, 0.0, 1.0)
            here_pos = lower + wp.vec3(
                wp.float32(i) * delta.x, wp.float32(j) * delta.y, wp.float32(k) * delta.z
            )
            there_pos = lower + wp.vec3(
                wp.float32(io) * delta.x, wp.float32(jo) * delta.y, wp.float32(ko) * delta.z
            )
            out_vertices[slot] = wp.lerp(here_pos, there_pos, t)
            slot += 1

    cell = mc_cell_triangles(field, iso, table, i, j, k)
    first_face = ends[1] - cell[1]
    for tri in range(cell[1]):
        for s in range(3):
            packed = table[MC_EDGE_BASE + table[MC_TRI_BASE + cell[0] + 3 * tri + s]]
            oi = i + (packed & 1)
            oj = j + ((packed >> 1) & 1)
            ok = k + ((packed >> 2) & 1)
            axis = packed >> 3
            # The owner's vertices sit at the end of its scanned count, in axis order, so this
            # edge's is its total less one less each crossing edge on a higher axis. An edge the
            # case table names but the crossing test rejects -- only possible with a NaN corner,
            # which the case code reads as below ``iso`` -- has no vertex, and gets Warp's ``-1``.
            vertex = wp.int32(-1)
            if mc_edge_crosses(field, iso, oi, oj, ok, axis) != 0:
                vertex = offsets[ravel_index(oi, oj, ok, ny, nz)][0] - 1
                for higher in range(axis + 1, 3):
                    vertex -= mc_edge_crosses(field, iso, oi, oj, ok, higher)
            out_faces[3 * (first_face + tri) + s] = vertex


@wp.func
def shell_vertex(
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    outside: wp.float32,
    inside: wp.float32,
    v: wp.int32,
    out_vertices: wp.array[wp.vec3],
) -> None:
    # Both layers of a thickened shell at vertex ``v``: the outward-displaced copy in the first
    # ``n_vertices`` slots and the inward-displaced one after it, so the second layer's vertex ``v``
    # is at ``v + n_vertices`` and the face emitters below can shift by a constant.
    n_vertices = vertices.shape[0]
    position = vertices[v]
    normal = normals[v]
    out_vertices[v] = position + outside * normal
    out_vertices[n_vertices + v] = position - inside * normal


@wp.func
def shell_face(
    faces: wp.array[wp.int32], n_vertices: wp.int32, f: wp.int32, out_faces: wp.array[wp.int32]
) -> None:
    # The two layers' triangles of face ``f``: the outer copy verbatim, the inner copy shifted by
    # ``n_vertices`` and **wound backwards**, because it faces into the shell rather than out of it.
    # Corners 1 and 2 are swapped, which is ``repair.flip_faces_masked``'s reversal without the
    # mask.
    n_faces = faces.shape[0] // 3
    corner0 = faces[3 * f]
    corner1 = faces[3 * f + 1]
    corner2 = faces[3 * f + 2]
    out_faces[3 * f] = corner0
    out_faces[3 * f + 1] = corner1
    out_faces[3 * f + 2] = corner2
    inner = 3 * (n_faces + f)
    out_faces[inner] = n_vertices + corner0
    out_faces[inner + 1] = n_vertices + corner2
    out_faces[inner + 2] = n_vertices + corner1


@wp.func
def shell_band(
    boundary_edges: wp.array2d[wp.int32],
    n_vertices: wp.int32,
    base: wp.int32,
    e: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    # The band closing the shell along boundary edge ``e``: two triangles spanning the outer edge
    # ``(a, b)`` and its inner copy. The winding follows the *directed* boundary edge, which
    # ``boundary.oriented_boundary_edges`` returns in the outer layer's own face winding -- so the
    # band inherits that orientation instead of guessing one, and the whole shell comes out
    # consistently wound. Verified on ``hemisphere`` and ``half_torus``: watertight, consistent, and
    # positive volume.
    outer_a = boundary_edges[e, 0]
    outer_b = boundary_edges[e, 1]
    inner_a = n_vertices + outer_a
    inner_b = n_vertices + outer_b
    slot = base + 6 * e
    out_faces[slot] = outer_a
    out_faces[slot + 1] = inner_b
    out_faces[slot + 2] = outer_b
    out_faces[slot + 3] = outer_a
    out_faces[slot + 4] = inner_a
    out_faces[slot + 5] = inner_b


@wp.kernel
def shell_mesh(
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    outside: wp.float32,
    inside: wp.float32,
    faces: wp.array[wp.int32],
    boundary_edges: wp.array2d[wp.int32],
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # The whole thickened shell in one launch over ``n_vertices + n_faces + n_rim`` threads: the
    # three emitters write disjoint slots and read only the input, so each thread takes one vertex,
    # one face or one boundary edge by its range (``creation.revolve_mesh``'s layout).
    t = wp.int32(wp.tid())
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0] // 3
    if t < n_vertices:
        shell_vertex(vertices, normals, outside, inside, t, out_vertices)
    elif t < n_vertices + n_faces:
        shell_face(faces, n_vertices, t - n_vertices, out_faces)
    else:
        shell_band(boundary_edges, n_vertices, 6 * n_faces, t - n_vertices - n_faces, out_faces)
