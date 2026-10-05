from typing import Any

import warp as wp

from ordito.constants import TILE_1D
from ordito.kernels.array import OverloadTable
from ordito.kernels.reduce import block_chunk, block_chunk_1d, commit_block_sum, commit_block_total
from ordito.kernels.triangles import (
    face_area_weighted_centroid,
    face_signed_volume,
    face_vertices_vec3d,
)


@wp.kernel
def mesh_signed_volume(
    vertices: wp.array[Any], faces: wp.array[wp.int32], out_volume: wp.array[wp.Float]
) -> None:
    # The whole mesh's signed volume from the origin, folded as it is formed: each block owns the
    # ``ITEMS_PER_BLOCK_1D`` faces ``block_chunk_1d`` gives it, its lanes stride them by
    # ``wp.block_dim()`` (so the one CPU lane walks them all, CLAUDE.md section 2.2), and lane 0
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
def centroid_tiled(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    out_totals: wp.array[wp.float32],
) -> None:
    # CUDA path, launched via wp.launch_tiled with block_dim=TILE_1D: each lane computes one face's
    # area-weighted centroid contribution, the block reduces each component cooperatively with
    # wp.tile_sum, and lane 0 commits four atomics per block (out-of-range lanes contribute zero).
    # CPU MUST NOT use this: the lanes partition the **outer** work (the face list), so the stride
    # is the constant `TILE_1D` and not `wp.block_dim()` -- and `wp.launch_tiled` runs one lane per
    # block on the CPU device through Warp 1.18 (unless the experimental
    # `wp.config.enable_cpu_blocks` is set), where that lane would see one face per tile. This
    # is the constant-stride case of the rule in `.claude/CLAUDE.md` section 2.2, and it is why this
    # reduction keeps a device pair where `kernels/visibility.py::obscurance` needs only one
    # kernel. See `centroid_sliced` and `_device.prefers_tiled_reduction`.
    #
    # **Rewriting it into the lane-strided `ITEMS_PER_BLOCK_1D` fold was measured and declined**,
    # and the reason is worth keeping, because the same rewrite is worth several-fold on the
    # `registration` and `points` accumulators (`.claude/CLAUDE.md` section 13.2). Built as
    # `tile_chunk(n_faces, chunk, ITEMS_PER_BLOCK_1D)` plus a `wp.block_dim()` stride -- which would
    # also make it portable and retire `centroid_sliced` -- it measures flat at every mesh size. The
    # fold pays on *atomic contention*, and contention here is `blocks x slots`: four slots is an
    # order of magnitude short of the accumulators that won, which carry twenty-five and
    # forty-three. Below that the launch floor hides it. With no CUDA win to pay for it the
    # portability is not free either -- `blocks_1d(n)` would give the CPU path far fewer,
    # single-lane blocks than `slice_count`'s threads -- so the device pair stays.
    #
    # One ``(4,)`` accumulator rather than a ``(3,)`` and a ``(1,)``: slots 0-2 are the
    # area-weighted centroid sum and slot 3 the area sum. The wrapper's return is a host-side
    # ``wp.vec3``, so both have to cross the bus -- and a readback's cost is almost all fixed, so
    # one read of four floats beats two reads of three and one. ``centroid_sliced`` writes the
    # same four slots. Measured 1.75x on ``surface_centroid``, which also loses an allocation and
    # a launch argument.
    #
    # The per-face contribution is ``triangles.face_area_weighted_centroid``, shared with
    # ``centroid_sliced`` below so the two paths cannot drift; see it for why the area is taken
    # from the corners this kernel already holds.
    i, t = wp.tid()
    f = i * TILE_1D + t
    contrib = wp.vec3(0.0, 0.0, 0.0)
    area = wp.float32(0.0)
    if f < n_faces:
        contrib, area = face_area_weighted_centroid(vertices, faces, f)
    commit_block_sum(t, wp.vec4(contrib[0], contrib[1], contrib[2], area), out_totals, 0)


@wp.kernel
def centroid_sliced(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    n_slices: wp.int32,
    out_totals: wp.array[wp.float32],
) -> None:
    # Portable path: one thread per face slice walks a strided slice, accumulates locally and
    # commits four atomics. Lane-free, so it is correct on the CPU device where `centroid_tiled`
    # is not; it gives up the block shuffle-reduce and is measurably slower on CUDA at a large face
    # count, which is why both exist.
    #
    # The bug this sibling exists to prevent: running `centroid_tiled` on the CPU device silently
    # averaged **every TILE_1D-th face** rather than every face, because `wp.launch_tiled` gives
    # that device one lane per block, so `t` is always 0 and `f = i * TILE_1D + t` skips the rest of
    # each tile. It was invisible on a symmetric mesh, whose every-Nth-face centroid is still the
    # true centroid.
    #
    # The per-face contribution is the same shared ``triangles.face_area_weighted_centroid``
    # `centroid_tiled` folds, which is what makes the two paths the same reduction over the same
    # summands and their disagreement purely one of summation order.
    j = wp.int32(wp.tid())
    total = wp.vec3(0.0, 0.0, 0.0)
    area_total = wp.float32(0.0)
    for f in range(j, n_faces, n_slices):
        contrib, area = face_area_weighted_centroid(vertices, faces, f)
        total = total + contrib
        area_total = area_total + area
    wp.atomic_add(out_totals, 0, total[0])
    wp.atomic_add(out_totals, 1, total[1])
    wp.atomic_add(out_totals, 2, total[2])
    wp.atomic_add(out_totals, 3, area_total)


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
    # device too (CLAUDE.md section 2.2) -- the same form as ``points.centered_covariance``, and
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
