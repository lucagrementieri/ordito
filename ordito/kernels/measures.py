from typing import Any

import warp as wp

from ordito.constants import TILE_1D
from ordito.kernels.adjacency import write_face_edge_keys
from ordito.kernels.array import OverloadTable, mark_at, sorted_run_start
from ordito.kernels.reduce import (
    block_chunk,
    block_chunk_1d,
    block_sum,
    commit_block_sum,
    last_ticket,
    last_ticket_vector,
    store_block_total,
    store_folded_sum,
    store_folded_total,
)
from ordito.kernels.triangles import (
    block_area_weighted_centroid_sum,
    face_signed_volume,
    face_vertices_vec3d,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel
def mesh_signed_volume(
    vertices: wp.array[Any],
    faces: wp.array[wp.int32],
    stride: wp.int32,
    out_volume: wp.array[wp.Float],
) -> None:
    # The whole mesh's signed volume from the origin, folded as it is formed: each block owns the
    # ``ITEMS_PER_BLOCK_1D`` faces ``block_chunk_1d`` gives it, its lanes stride them by
    # ``wp.block_dim()`` (so the one CPU lane walks them all), and lane 0
    # stores the block's sum in its own row, which the last block adds in a fixed order
    # (``store_block_total``). ``triangles.face_signed_volume`` per face, the volume
    # ``triangles.face_signed_volumes`` would have stored for a ``reduce.sum`` pass to read back.
    block, lane = wp.tid()
    offset, count = block_chunk_1d(faces.shape[0] // 3, block)
    total = out_volume.dtype(0.0)
    origin = vertices.dtype()
    for k in range(lane, count, wp.block_dim()):
        total += face_signed_volume(vertices, faces, offset + k, origin)
    store_block_total(block, lane, total, out_volume, 0, stride, last_ticket)


@wp.kernel
def surface_centroid_totals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    stride: wp.int32,
    out_totals: wp.array[wp.vec4d],
) -> None:
    # The whole surface's area-weighted barycentre sum (components 0-2) and area (component 3), in
    # ``_launch.block_partials``' slot 0 for ``measures.surface_centroid`` to read and divide. Each
    # block's ``float32`` sum (``triangles.block_area_weighted_centroid_sum``) is widened to
    # ``float64`` before it is committed, so the order the blocks add in moves only digits far
    # below the ``float32`` centroid the wrapper returns, whichever commit the setting picks.
    block, lane = wp.tid()
    folded = block_area_weighted_centroid_sum(vertices, faces, block, lane)
    store_folded_total(block, lane, wp.vec4d(folded), out_totals, 0, stride, last_ticket_vector)


@wp.func
def tetrahedron_first_integrals(
    a: wp.vec3d, b: wp.vec3d, c: wp.vec3d
) -> tuple[wp.float64, wp.vec3d, wp.vec3d]:
    # The tetrahedron ``(0, a, b, c)``'s ``det = dot(a, cross(b, c))`` (six times its volume), its
    # corner sum ``s = a + b + c`` and ``det * s`` (24 times its first moment): the leading terms
    # ``moment_integrals`` and ``centroid_integrals`` both sum, in one spelling so the two agree
    # bit for bit. The constant factors wait for ``scaled_first_integrals``, once a block.
    s = a + b + c
    det = wp.dot(a, wp.cross(b, c))
    return det, s, det * s


@wp.func
def scaled_first_integrals(det_sum: wp.float64, first_sum: wp.vec3d) -> wp.vec4d:
    # A block's volume and first moment from its sums of ``tetrahedron_first_integrals``' terms.
    return wp.vec4d(
        det_sum / wp.float64(6.0),
        first_sum[0] / wp.float64(24.0),
        first_sum[1] / wp.float64(24.0),
        first_sum[2] / wp.float64(24.0),
    )


# Blocks to aim for in ``moment_integrals``' grid. Below it the chunk stays one tile wide, so the
# device fills; above it the chunk doubles instead, so fewer blocks pay the ten-slot ``float64``
# block fold and commit. Swept with the regrouped integrand (2026-11-09, CUDA, calls): 640 against
# 1280 took bunny 61 -> 53 us; 160-640 within noise from dragon up.
MOMENT_TARGET_BLOCKS = 640
# Widest chunk worth using: 32 tiles took lucy 2.91 -> 2.44 ms against 8; 64 and 128 gained
# nothing more there and cost dragon up to 1.25x at a target of 320 (too few blocks to fill the
# device).
MOMENT_MAX_CHUNK_FACES = 32 * TILE_1D


def moment_chunk_faces(n_faces: int) -> int:
    """
    Faces one block of ``moment_integrals`` reduces, from the mesh size.

    Doubles from one tile until the grid is no wider than ``MOMENT_TARGET_BLOCKS``, then stops at
    ``MOMENT_MAX_CHUNK_FACES``. A host helper beside its kernel because both ``measures.moments``
    and ``smoothing.inflate`` launch it.
    """
    chunk = TILE_1D
    while chunk < MOMENT_MAX_CHUNK_FACES and -(-n_faces // chunk) > MOMENT_TARGET_BLOCKS:
        chunk *= 2
    return chunk


@wp.kernel
def moment_integrals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    chunk_faces: wp.int32,
    stride: wp.int32,
    out_totals: wp.array[wp.float64],
) -> None:
    # The ten mass integrals of the solid bounded by the mesh, summed over faces into one ``(10,)``
    # accumulator: volume, the three first moments, the three second moments and the three
    # products, in the order ``measures.moments`` reads them back.
    #
    # The reduction is *in* this kernel rather than four ``wp.utils.array_sum`` calls over four
    # per-face buffers, which is four host readbacks over 80 bytes a face written once and read
    # once. Launched ``wp.launch_tiled(dim=[ceil(n_faces / chunk_faces)], block_dim=TILE_1D)``,
    # lanes striding their own block's chunk by ``wp.block_dim()`` so it is correct on the CPU
    # device too -- the same form as ``points.centered_covariance``.
    #
    # ``chunk_faces`` is a launch argument rather than the ``ITEMS_PER_BLOCK_1D`` constant the rest
    # of the family bakes in, because this body has a real crossover the family's single value sits
    # on the wrong side of: the integrand is ``float64``, so a wide chunk starves the device on a
    # small mesh while a narrow one makes too many blocks pay the ten-slot fold and commit on a
    # large one. A runtime width measures identical to a module-constant one; see
    # ``moment_chunk_faces``.
    #
    # Everything accumulates in float64: the second moments scale as length^5, so a float32 sum
    # over a large mesh loses the answer's low digits before the reduction finishes.
    #
    # For the tetrahedron (0, a, b, c) with det = dot(a, cross(b, c)):
    #   int dV     = det / 6
    #   int x dV   = det * s.x / 24                       (s = a + b + c)
    #   int x^2 dV = det * (s.x^2 + a.x^2 + b.x^2 + c.x^2) / 120
    #   int xy dV  = det * (s.x s.y + a.x a.y + b.x b.y + c.x c.y) / 120
    # the second-moment forms being the textbook ones (``(a.x^2 + ... + b.x c.x) / 60`` and its
    # product analogue) regrouped around ``s``. Each lane sums the bracketed terms and the block
    # divides its folds once: the integrand is ``float64`` at 1/64 of the ``float32`` rate on this
    # device, so the kernel is bound by its ``float64`` operations, and the textbook form's ten
    # divisions and 45 products a face set its time.
    chunk, lane = wp.tid()
    offset, count = block_chunk(faces.shape[0] // 3, chunk, chunk_faces)
    if count <= 0:
        return

    det_sum = wp.float64(0.0)
    first = wp.vec3d()
    squares = wp.vec3d()
    products = wp.vec3d()
    for k in range(lane, count, wp.block_dim()):
        a, b, c = face_vertices_vec3d(vertices, faces, offset + k)
        det, s, first_term = tetrahedron_first_integrals(a, b, c)
        det_sum = det_sum + det
        first = first + first_term
        squares = squares + det * (
            wp.cw_mul(s, s) + wp.cw_mul(a, a) + wp.cw_mul(b, b) + wp.cw_mul(c, c)
        )
        # (xy, xz, yz), in the same order the wrapper reads them back.
        products = products + det * (
            wp.vec3d(s[0] * s[1], s[0] * s[2], s[1] * s[2])
            + wp.vec3d(a[0] * a[1], a[0] * a[2], a[1] * a[2])
            + wp.vec3d(b[0] * b[1], b[0] * b[2], b[1] * b[2])
            + wp.vec3d(c[0] * c[1], c[0] * c[2], c[1] * c[2])
        )

    # All ten sums in one block reduction, which is block-collective and so runs outside the
    # ``lane == 0`` guard, then the block's constant factors.
    local = wp.vector(length=10, dtype=wp.float64)
    local[0] = det_sum
    for j in range(3):
        local[1 + j] = first[j]
        local[4 + j] = squares[j]
        local[7 + j] = products[j]
    folded = block_sum(local)
    leading = scaled_first_integrals(folded[0], wp.vec3d(folded[1], folded[2], folded[3]))
    for j in range(4):
        folded[j] = leading[j]
    for j in range(6):
        folded[4 + j] = folded[4 + j] / wp.float64(120.0)
    store_folded_sum(chunk, lane, folded, out_totals, 0, stride)


@wp.kernel
def centroid_integrals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    chunk_faces: wp.int32,
    stride: wp.int32,
    out_totals: wp.array[wp.float64],
) -> None:
    # The first four of ``moment_integrals``' ten totals -- volume and first moment, all a centre of
    # mass needs -- with that kernel's chunking, per-face arithmetic and lane order, so on a
    # deterministic device they are its totals bit for bit. Four contended ``float64`` slots per
    # block instead of ten, and a third of the integrand: ``smoothing.inflate`` reads the centre of
    # every pass's input from here.
    chunk, lane = wp.tid()
    offset, count = block_chunk(faces.shape[0] // 3, chunk, chunk_faces)
    if count <= 0:
        return
    det_sum = wp.float64(0.0)
    first = wp.vec3d()
    for k in range(lane, count, wp.block_dim()):
        a, b, c = face_vertices_vec3d(vertices, faces, offset + k)
        det, _s, first_term = tetrahedron_first_integrals(a, b, c)
        det_sum = det_sum + det
        first = first + first_term
    folded = block_sum(wp.vec4d(det_sum, first[0], first[1], first[2]))
    leading = scaled_first_integrals(folded[0], wp.vec3d(folded[1], folded[2], folded[3]))
    store_folded_sum(chunk, lane, leading, out_totals, 0, stride)


# Keyed by the vertex dtype; the volume is accumulated in its scalar type, as
# ``triangles.face_signed_volumes`` stores it.
MESH_SIGNED_VOLUME: OverloadTable


@wp.kernel
def face_edge_keys_and_referenced(
    faces: wp.array[wp.int32],
    base: wp.uint64,
    out_keys: wp.array[wp.uint64],
    out_referenced: wp.array[wp.bool],
) -> None:
    # ``measures.euler_characteristic``'s one pass over the faces: the three undirected edge keys
    # of face ``f`` (``adjacency.write_face_edge_keys``, into the leading half of the radix sort's
    # buffer) and the referenced-vertex mask its corners mark (``array.indices_to_mask``'s
    # scatter). No identity payload: the count reads only the sorted keys.
    f = wp.int32(wp.tid())
    write_face_edge_keys(faces, f, 3 * f, base, out_keys)
    for k in range(3):
        mark_at(out_referenced, faces[3 * f + k])


@wp.kernel
def referenced_vertex_and_unique_edge_counts(
    referenced: wp.array[wp.bool],
    sorted_keys: wp.array[wp.uint64],
    n_items: wp.int32,
    out_counts: wp.array[wp.int32],
) -> None:
    # ``measures.euler_characteristic``'s two counts in one fold, committed into a zeroed two-slot
    # buffer the wrapper reads once: slot 0 the ``True`` entries of the referenced-vertex mask,
    # slot 1 the run starts of the sorted edge keys (one per unique edge, which is what
    # ``edges.edges_unique``'s scan tail counts). ``n_items`` is the longer of the two arrays;
    # each test is bounded by its own array's length. Lane-strided by ``wp.block_dim()``.
    block, lane = wp.tid()
    offset, count = block_chunk_1d(n_items, block)
    n_vertices = referenced.shape[0]
    n_keys = sorted_keys.shape[0]
    local = wp.vec2i(0, 0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        if i < n_vertices:
            if referenced[i]:
                local += wp.vec2i(1, 0)
        if i < n_keys:
            if sorted_run_start(sorted_keys, i):
                local += wp.vec2i(0, 1)
    commit_block_sum(lane, local, out_counts, 0)


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global MESH_SIGNED_VOLUME
    MESH_SIGNED_VOLUME = OverloadTable(
        mesh_signed_volume,
        {
            vector: [wp.array[vector], wp.array[wp.int32], wp.int32, wp.array[scalar]]
            for vector, scalar in ((wp.vec3, wp.float32), (wp.vec3d, wp.float64))
        },
    )


_register_overloads()
