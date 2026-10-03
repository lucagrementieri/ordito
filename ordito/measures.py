"""
What the whole mesh adds up to: the volume it encloses, its moments and its Euler characteristic.

Every function here reduces the entire surface to a single host-side number (or a small tuple of
them), which is why they are gathered rather than left beside the per-element fields they sum over.
Each therefore costs at least one host readback and forces a device synchronisation -- call them
once and keep the answer, and reach for [`ordito.reduce`][ordito.reduce] when a reduction should
stay on the device.

Three of the four are **mass properties of the enclosed solid**, and all three are only meaningful
on a closed, consistently wound surface -- [`is_volume`][ordito.validation.is_volume] is the check
for that, and [`make_volume`][ordito.repair.make_volume] the repair.
[`volume`][ordito.measures.volume] is the signed volume,
[`moments`][ordito.measures.moments] adds the centre of mass and the inertia tensor, and
[`surface_centroid`][ordito.measures.surface_centroid] is the one member that is a *normalized*
total rather than a sum: the area-weighted mean of the per-face centroids. It is a property of the
**shell**, so it differs from ``moments``' centre of mass on any solid whose mass is not distributed
like its surface -- and it needs no watertightness, since a surface has a centroid whether or not it
bounds anything.

[`euler_characteristic`][ordito.measures.euler_characteristic] is the odd one out: a topological
invariant rather than a measurement, counted off the index buffer with no geometry involved. It sits
here because it is a whole-mesh integer, and it is what
[`homology_generators`][ordito.homology.homology_generators] derives the genus from.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import ordito as od
from ordito import _launch
from ordito._device import prefers_tiled_reduction, read_scalar, require_same_device, slice_count
from ordito.constants import TILE_1D
from ordito.kernels import measures as kernel_measures
from ordito.kernels import reduce as kernel_reduce


def volume(vertices: wp.array[wp.vec3] | wp.array[wp.vec3d], faces: wp.array[wp.int32]) -> float:
    """
    Signed volume enclosed by the mesh.

    Sum of per-face signed tetrahedron volumes measured from the origin
    (``dot(v0, cross(v1, v2)) / 6``); for a closed, consistently wound surface this is
    independent of the reference point, and its sign follows the orientation of the face
    normals (positive for outward-facing normals). For an open or inconsistently wound mesh
    the result is not a meaningful volume.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions, ``wp.vec3`` or ``wp.vec3d``. The accumulation
        follows: pass ``vec3d`` where the sum's low digits matter, as
        [`filter_laplacian`][ordito.smoothing.filter_laplacian]'s volume constraint does.
    faces
        ``(3 * n_faces,)`` triangle index buffer.

    Returns
    -------
    float
        Signed volume, accumulated in ``vertices``' scalar type. ``0.0`` for an empty mesh.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`face_signed_volumes`][ordito.triangles.face_signed_volumes]
        The per-face primitive this sums.
    [`is_volume`][ordito.validation.is_volume]
        Whether the sum means anything on this mesh.
    [`make_volume`][ordito.repair.make_volume]
        Make it mean something.
    [`moments`][ordito.measures.moments]
        The same volume in float64, plus the centre of mass and inertia.
    [`filter_laplacian`][ordito.smoothing.filter_laplacian]
        The ``float64`` consumer: its volume constraint rescales the mesh to hold this fixed.
    [`trimesh.Trimesh.volume`][]
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    if n_faces == 0:
        return 0.0
    device = vertices.device
    # Each face's tetrahedron is formed where it is summed: no per-face volume table.
    total = (
        _launch.zeros(1, dtype=wp.float64, device=device)
        if vertices.dtype is wp.vec3d
        else _launch.zeros(1, dtype=wp.float32, device=device)
    )
    _launch.launch_tiled(
        kernel_measures.MESH_SIGNED_VOLUME[vertices.dtype],
        dim=[kernel_reduce.blocks_1d(n_faces)],
        inputs=[vertices, faces, total],
        block_dim=TILE_1D,
        device=device,
    )
    return float(read_scalar(total))


def surface_centroid(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.vec3:
    """
    Area-weighted centroid of the mesh surface: the mean of the per-face centroids by area.

    A *normalized* total rather than a sum, and a property of the shell alone -- unlike
    [`moments`][ordito.measures.moments]' centre of mass it needs no watertightness and no
    consistent winding, because a surface has a centroid whether or not it bounds a solid. The two
    differ on any solid whose mass is not distributed like its surface.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.

    Returns
    -------
    wp.vec3
        Area-weighted mean of per-face centroids. All-``NaN`` when the weights sum to zero --
        an empty ``faces``, or a mesh whose every face is degenerate -- since no weighted mean of
        the face centroids exists in either case.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`moments`][ordito.measures.moments]
        The centre of mass of the enclosed *solid*, which this is not.
    [`face_centroids`][ordito.triangles.face_centroids]
        The per-face barycentres this averages.
    [`centroid`][ordito.points.centroid]
        The unweighted mean of a point cloud, and the only other ``centroid`` in the package.
    [`trimesh.Trimesh.centroid`][]
    """
    require_same_device(vertices=vertices, faces=faces)
    f = faces.size // 3
    if f == 0:
        return wp.vec3(float("nan"), float("nan"), float("nan"))
    device = vertices.device
    # One accumulator, read once: slots 0-2 hold the area-weighted centroid sum and slot 3 the
    # area sum. See the kernel.
    totals = _launch.zeros(4, dtype=wp.float32, device=device)
    if prefers_tiled_reduction(device):
        _launch.launch_tiled(
            kernel_measures.centroid_tiled,
            dim=[(f + TILE_1D - 1) // TILE_1D],
            inputs=[vertices, faces, wp.int32(f), totals],
            block_dim=TILE_1D,
            device=device,
        )
    else:
        n_slices = slice_count(f, device)
        _launch.launch(
            kernel_measures.centroid_sliced,
            dim=n_slices,
            inputs=[vertices, faces, wp.int32(f), wp.int32(n_slices), totals],
            device=device,
        )
    # One unavoidable readback: the return type is a host-side wp.vec3, so the sums have to cross
    # to the host to be divided, and the kernel put all four of them in one buffer to do it in a
    # single sync.
    weighted = totals.numpy()
    total_area = float(weighted[3])
    if total_area == 0.0:
        # Every face degenerate: the weights are all zero, so there is no weighted mean. Same
        # answer as the empty mesh above, and the same shape as ``moments``' zero-volume guard --
        # a ``ZeroDivisionError`` out of a mesh that is merely degenerate would be a surprise.
        return wp.vec3(float("nan"), float("nan"), float("nan"))
    return wp.vec3(
        float(weighted[0]) / total_area,
        float(weighted[1]) / total_area,
        float(weighted[2]) / total_area,
    )


def moments(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[float, wp.vec3, wp.mat33d]:
    """
    Mass properties of the solid bounded by the mesh: volume, centre of mass and inertia tensor.

    Integrates over the enclosed solid at unit density by summing the contribution of every
    (origin, face) tetrahedron, so the result is only meaningful for a **closed, consistently
    wound** surface -- the same precondition [`volume`][ordito.measures.volume] carries, and
    [`is_volume`][ordito.validation.is_volume] is the check for it.

    The integrals accumulate in ``float64`` even though the positions are ``float32``: the second
    moments scale as ``length ** 5``, so a ``float32`` sum loses their low digits on any sizeable
    mesh.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.

    Returns
    -------
    volume : float
        Signed volume, identical to [`volume`][ordito.measures.volume] up to its ``float32``
        accumulation.
    center_of_mass : wp.vec3
        Volume centroid, i.e. the first moment divided by the volume. This is **not**
        [`surface_centroid`][ordito.measures.surface_centroid], the area-weighted centre of the
        *surface*; the two differ on any solid whose mass is not distributed like its shell.
    inertia : wp.mat33d
        ``(3, 3)`` inertia tensor about the centre of mass, at unit density. ``float64``, like the
        integrals it is assembled from -- a ``wp.mat33`` would discard exactly the low digits this
        function accumulates in double precision to keep.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    ``igl.moments`` returns ``(m0, m1, m2)`` where ``m1`` is the first moment -- the centre of mass
    times the mass -- rather than the centre of mass itself, and ``m2`` is already referred to the
    centre of mass (verified against a translated mesh, not assumed). This function returns the
    decoded forms, so the transform between the two is ``m1 / m0``.

    One host readback is unavoidable here: all three returns are host-side values, so the ten
    accumulated sums have to cross the device boundary to be combined. They cross together.

    See Also
    --------
    [`volume`][ordito.measures.volume]
    [`surface_centroid`][ordito.measures.surface_centroid]
    [`is_volume`][ordito.validation.is_volume]
    [`trimesh.Trimesh.moment_inertia`][]
    ``igl.moments``
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return 0.0, wp.vec3(float("nan"), float("nan"), float("nan")), wp.mat33d()

    # One launch and one readback for all ten integrals: the kernel reduces over faces itself and
    # commits into a single ``(10,)`` accumulator, so there are no per-face buffers to allocate and
    # no per-group reduction to launch. ``wp.zeros`` rather than ``wp.empty`` because the kernel
    # accumulates into it.
    totals = _launch.zeros(10, dtype=wp.float64, device=device)
    chunk_faces = kernel_measures.moment_chunk_faces(n_faces)
    _launch.launch_tiled(
        kernel_measures.moment_integrals,
        dim=[(n_faces + chunk_faces - 1) // chunk_faces],
        inputs=[vertices, faces, chunk_faces, totals],
        block_dim=TILE_1D,
        device=device,
    )

    # Every return here is a host-side value, so the ten sums have to cross the device boundary --
    # but only the ten, and in one readback rather than one per group.
    totals_np = totals.numpy()
    total_volume = float(totals_np[0])
    first_moment = totals_np[1:4]
    integral_squares = totals_np[4:7]
    integral_products = totals_np[7:10]

    center = first_moment / total_volume if total_volume != 0.0 else np.full(3, np.nan)

    # Inertia about the origin from the raw integrals, then shifted to the centre of mass. Nine
    # scalars of host float64 arithmetic, packed into a wp.mat33d so no caller needs numpy to read
    # the answer.
    x2, y2, z2 = integral_squares
    xy, xz, yz = integral_products
    inertia = np.array(
        [[y2 + z2, -xy, -xz], [-xy, x2 + z2, -yz], [-xz, -yz, x2 + y2]], dtype=np.float64
    )
    if total_volume != 0.0:
        shift = total_volume * (float(center @ center) * np.eye(3) - np.outer(center, center))
        inertia = inertia - shift

    return total_volume, wp.vec3(*center), wp.mat33d(*inertia.ravel())


def euler_characteristic(faces: wp.array[wp.int32]) -> int:
    """
    Euler characteristic ``V - E + F`` of the mesh (topological invariant).

    Counts distinct referenced vertices, unique undirected edges, and faces, matching
    [`trimesh.Trimesh.euler_number`][] (which uses referenced vertices, ``edges_unique``, and
    faces). For a closed genus-``g`` surface this equals ``2 - 2 * g``.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.

    Returns
    -------
    int
        ``(#distinct referenced vertices) - (#unique edges) + (#faces)``. ``0`` for an empty mesh.

    See Also
    --------
    [`edges_unique`][ordito.edges.edges_unique]
    [`trimesh.Trimesh.euler_number`][]
    """
    n_faces = faces.size // 3
    if n_faces == 0:
        return 0

    # The bound is taken once, here, so ``edges_unique`` re-checking it would reduce the same
    # indices twice; ``require_non_negative`` keeps the half of that check a derived bound cannot
    # give, and costs nothing -- both ends come out of the one reduction.
    n_vertices = od.array.index_bound(faces, require_non_negative=True)
    # The referenced-vertex count is a membership mask's popcount rather than the length of
    # ``unique_1d(faces)``: the indices are bounded by ``n_vertices``, so the mask is the same set
    # without the hash table, compaction and sort, and the count is one read instead of two.
    n_referenced = od.reduce.sum(od.array.indices_to_mask(faces, n_vertices))
    unique_edges, _ = od.edges.edges_unique(faces, n_vertices=n_vertices, validate=False)
    n_edges = int(unique_edges.shape[0])
    return n_referenced - n_edges + n_faces
