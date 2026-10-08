from typing import Any

import warp as wp

from ordito.constants import TILE_1D
from ordito.kernels.array import OverloadTable
from ordito.kernels.reduce import (
    block_chunk,
    block_chunk_1d,
    block_sum,
    commit_block_sum,
    commit_block_total,
)
from ordito.kernels.triangles import (
    face_area_weighted_centroid,
    face_signed_volume,
    face_vertices_vec3d,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel
def mesh_signed_volume(
    vertices: wp.array[Any], faces: wp.array[wp.int32], out_volume: wp.array[wp.Float]
) -> None:
    # The whole mesh's signed volume from the origin, folded as it is formed: each block owns the
    # ``ITEMS_PER_BLOCK_1D`` faces ``block_chunk_1d`` gives it, its lanes stride them by
    # ``wp.block_dim()`` (so the one CPU lane walks them all), and lane 0
    # commits one atomic per block. ``triangles.face_signed_volume`` per face, the volume
    # ``triangles.face_signed_volumes`` would have stored for a ``reduce.sum`` pass to read back.
    block, lane = wp.tid()
    offset, count = block_chunk_1d(faces.shape[0] // 3, block)
    total = out_volume.dtype(0.0)
    origin = vertices.dtype()
    for k in range(lane, count, wp.block_dim()):
        total += face_signed_volume(vertices, faces, offset + k, origin)
    commit_block_total(lane, total, out_volume, 0)


@wp.kernel
def centroid_partials(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    out_partials: wp.array[wp.vec4],
) -> None:
    # Each block's area-weighted centroid sum (components 0-2) and area (component 3), stored in the
    # block's own slot for ``reduce.sum`` to fold in a fixed order. A block owns the
    # ``ITEMS_PER_BLOCK_1D`` faces ``block_chunk_1d`` gives it and its lanes stride them by
    # ``wp.block_dim()`` -- so the one CPU lane walks them all) and one
    # kernel serves both devices -- and the per-face contribution is
    # ``triangles.face_area_weighted_centroid``.
    #
    # Partials rather than an atomic commit: one ``atomic_add`` per block summed the blocks in
    # arrival order, so on CUDA the centroid's last bits moved from run to run, and with them every
    # point ``sample.sample_volume`` fans from it -- a seeded draw was not reproducible (10
    # distinct centroids in 10 calls on every scan mesh). It is also the faster form: against the
    # old one-tile-per-block atomic kernel 5.4x at 28 M faces and 1.1x at 0.87 M on CUDA, level
    # below; against the old lane-free ``_sliced`` CPU kernel 1.1-1.4x small and 2.1x at 28 M.
    block, lane = wp.tid()
    offset, count = block_chunk_1d(n_faces, block)
    total = wp.vec4(0.0, 0.0, 0.0, 0.0)
    for k in range(lane, count, wp.block_dim()):
        contrib, area = face_area_weighted_centroid(vertices, faces, offset + k)
        total += wp.vec4(contrib[0], contrib[1], contrib[2], area)
    block_total = block_sum(total)
    if lane == 0:
        out_partials[block] = block_total


@wp.func
def tetrahedron_first_integrals(
    a: wp.vec3d, b: wp.vec3d, c: wp.vec3d
) -> tuple[wp.float64, wp.float64, wp.vec3d]:
    # The tetrahedron ``(0, a, b, c)``'s ``det = dot(a, cross(b, c))``, volume ``det / 6`` and first
    # moment ``det * (a + b + c) / 24``: the leading integrals ``moment_integrals`` and
    # ``centroid_integrals`` both sum, in one spelling so the two agree bit for bit.
    det = wp.dot(a, wp.cross(b, c))
    return det, det / wp.float64(6.0), det * (a + b + c) / wp.float64(24.0)


# Blocks to aim for in ``moment_integrals``' grid. Below it the chunk stays one tile wide, so the
# device fills; above it the chunk doubles instead, so the ten contended ``float64`` accumulator
# slots do not collect an atomic from every one of tens of thousands of blocks. This is where the
# measured table at the kernel crosses over -- the block count at which a wider chunk first stops
# losing.
MOMENT_TARGET_BLOCKS = 1280
# Widest chunk worth using: past this the grid stops filling the device before contention is the
# problem, and the measured table is flat from here on.
MOMENT_MAX_CHUNK_FACES = 8 * TILE_1D


def moment_chunk_faces(n_faces: int) -> int:
    """
    Faces one block of ``moment_integrals`` reduces, from the mesh size.

    Doubles from one tile until the grid is no wider than ``MOMENT_TARGET_BLOCKS``, then stops.
    The measured device times this is derived from are tabulated at the kernel. A host helper
    beside its kernel because both ``measures.moments`` and ``smoothing.inflate`` launch it.
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
    # device too -- the same form as ``points.centered_covariance``, and
    # unlike ``centroid_tiled`` above, whose lanes partition the outer work and which therefore
    # needs a device pair.
    #
    # ``chunk_faces`` is a launch argument rather than the ``ITEMS_PER_BLOCK_1D`` constant the rest
    # of the family bakes in, because this body has a real crossover the family's single value sits
    # on the wrong side of: the integrand is ~80 ``float64`` flops per face, so a wide chunk starves
    # the device on a small mesh while a narrow one puts too many blocks on ten contended addresses
    # on a large one. A runtime width measures identical to a ``wp.constant`` one; see
    # ``measures._moment_chunk_faces``.
    #
    # Everything accumulates in float64: the second moments scale as length^5, so a float32 sum
    # over a large mesh loses the answer's low digits before the reduction finishes.
    #
    # For the tetrahedron (0, a, b, c) with det = dot(a, cross(b, c)):
    #   int dV     = det / 6
    #   int x dV   = det * (a.x + b.x + c.x) / 24
    #   int x^2 dV = det * (a.x^2 + b.x^2 + c.x^2 + a.x b.x + a.x c.x + b.x c.x) / 60
    #   int xy dV  = det * (2(a.x a.y + b.x b.y + c.x c.y)
    #                       + a.x b.y + b.x a.y + a.x c.y + c.x a.y + b.x c.y + c.x b.y) / 120
    chunk, lane = wp.tid()
    offset, count = block_chunk(faces.shape[0] // 3, chunk, chunk_faces)
    if count <= 0:
        return

    volume = wp.float64(0.0)
    first = wp.vec3d()
    squares = wp.vec3d()
    products = wp.vec3d()
    for k in range(lane, count, wp.block_dim()):
        a, b, c = face_vertices_vec3d(vertices, faces, offset + k)
        det, volume_term, first_term = tetrahedron_first_integrals(a, b, c)
        volume = volume + volume_term
        first = first + first_term
        squares = squares + det * wp.vec3d(
            a[0] * a[0] + b[0] * b[0] + c[0] * c[0] + a[0] * b[0] + a[0] * c[0] + b[0] * c[0],
            a[1] * a[1] + b[1] * b[1] + c[1] * c[1] + a[1] * b[1] + a[1] * c[1] + b[1] * c[1],
            a[2] * a[2] + b[2] * b[2] + c[2] * c[2] + a[2] * b[2] + a[2] * c[2] + b[2] * c[2],
        ) / wp.float64(60.0)
        # (xy, xz, yz), in the same order the wrapper reads them back.
        products = products + det * wp.vec3d(
            wp.float64(2.0) * (a[0] * a[1] + b[0] * b[1] + c[0] * c[1])
            + a[0] * b[1]
            + b[0] * a[1]
            + a[0] * c[1]
            + c[0] * a[1]
            + b[0] * c[1]
            + c[0] * b[1],
            wp.float64(2.0) * (a[0] * a[2] + b[0] * b[2] + c[0] * c[2])
            + a[0] * b[2]
            + b[0] * a[2]
            + a[0] * c[2]
            + c[0] * a[2]
            + b[0] * c[2]
            + c[0] * b[2],
            wp.float64(2.0) * (a[1] * a[2] + b[1] * b[2] + c[1] * c[2])
            + a[1] * b[2]
            + b[1] * a[2]
            + a[1] * c[2]
            + c[1] * a[2]
            + b[1] * c[2]
            + c[1] * b[2],
        ) / wp.float64(120.0)

    # All ten integrals in one block reduction, which is block-collective and so runs outside the
    # ``lane == 0`` guard.
    local = wp.vector(length=10, dtype=wp.float64)
    local[0] = volume
    for j in range(3):
        local[1 + j] = first[j]
        local[4 + j] = squares[j]
        local[7 + j] = products[j]
    commit_block_sum(lane, local, out_totals, 0)


@wp.kernel
def centroid_integrals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    chunk_faces: wp.int32,
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
    volume = wp.float64(0.0)
    first = wp.vec3d()
    for k in range(lane, count, wp.block_dim()):
        a, b, c = face_vertices_vec3d(vertices, faces, offset + k)
        _det, volume_term, first_term = tetrahedron_first_integrals(a, b, c)
        volume = volume + volume_term
        first = first + first_term
    commit_block_sum(lane, wp.vec4d(volume, first[0], first[1], first[2]), out_totals, 0)


# Keyed by the vertex dtype; the volume is accumulated in its scalar type, as
# ``triangles.face_signed_volumes`` stores it.
MESH_SIGNED_VOLUME: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global MESH_SIGNED_VOLUME
    MESH_SIGNED_VOLUME = OverloadTable(
        mesh_signed_volume,
        {
            vector: [wp.array[vector], wp.array[wp.int32], wp.array[scalar]]
            for vector, scalar in ((wp.vec3, wp.float32), (wp.vec3d, wp.float64))
        },
    )


_register_overloads()
