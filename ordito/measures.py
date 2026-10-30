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
from ordito._device import read_scalar, read_values, require_same_device
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

    Examples
    --------
    A unit cube encloses volume 1; an icosphere approaches the unit ball's ``4 * pi / 3`` from
    below:

    ```python
    import ordito as od

    vertices, faces = od.creation.box()
    print(f"{od.measures.volume(vertices, faces):.4f}")  # 1.0000

    vertices, faces = od.creation.icosphere(subdivisions=3)
    print(f"{od.measures.volume(vertices, faces):.3f}")  # 4.153
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = faces.size // 3
    if n_faces == 0:
        return 0.0
    device = vertices.device
    # Each face's tetrahedron is formed where it is summed: no per-face volume table. The block
    # sums are added in a fixed order, so the volume repeats bit for bit on CUDA.
    blocks = kernel_reduce.blocks_1d(n_faces)
    total, stride = _launch.block_partials(
        blocks, 1, wp.float64 if vertices.dtype is wp.vec3d else wp.float32, device
    )
    _launch.launch_tiled(
        kernel_measures.MESH_SIGNED_VOLUME[vertices.dtype],
        dim=[blocks],
        inputs=[vertices, faces, stride, total],
        block_dim=TILE_1D,
        device=device,
    )
    return float(read_scalar(total, 0))


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

    Examples
    --------
    ```python
    import ordito as od

    vertices, faces = od.creation.box(bounds=[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    print(od.measures.surface_centroid(vertices, faces))  # [1.0, 2.0, 3.0]
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    f = faces.size // 3
    if f == 0:
        return wp.vec3(float("nan"), float("nan"), float("nan"))
    device = vertices.device
    # Per-block sums (slots 0-2 the area-weighted centroid sum, 3 the area), folded by
    # ``reduce.sum`` in a fixed order so the centroid is the same bits on every run. One readback:
    # the return type is a host-side wp.vec3, so the sums cross to the host to be divided.
    blocks = kernel_reduce.blocks_1d(f)
    partials = _launch.empty(blocks, dtype=wp.vec4, device=device)
    _launch.launch_tiled(
        kernel_measures.centroid_partials,
        dim=[blocks],
        inputs=[vertices, faces, wp.int32(f), partials],
        block_dim=TILE_1D,
        device=device,
    )
    if blocks <= kernel_measures.CENTROID_HOST_FOLD_BLOCKS:
        # Read back: the few block partials themselves, folded on the host in ``float64`` -- a
        # fixed order still, without the device reduction's launches.
        weighted = partials.numpy().astype(np.float64).sum(axis=0)
    else:
        weighted = od.reduce.sum(partials)
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

    Examples
    --------
    A ``2 x 4 x 6`` box of unit density has mass 48 and principal moments
    ``48 * (b**2 + c**2) / 12`` about its centre:

    ```python
    import ordito as od

    vertices, faces = od.creation.box(bounds=[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    volume, center, inertia = od.measures.moments(vertices, faces)
    print(f"{volume:.3f}")  # 48.000
    print(center)  # [1.0, 2.0, 3.0]
    print([round(float(inertia[i, i]), 3) for i in range(3)])  # [208.0, 160.0, 80.0]
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return 0.0, wp.vec3(float("nan"), float("nan"), float("nan")), wp.mat33d()

    # One readback for all ten integrals: the kernel reduces over faces itself into one row of ten
    # per block, so there are no per-face buffers to allocate and no per-group reduction to launch,
    # and the rows are added in a fixed order, so the moments repeat bit for bit on CUDA.
    chunk_faces = kernel_measures.moment_chunk_faces(n_faces)
    blocks = (n_faces + chunk_faces - 1) // chunk_faces
    totals, stride = _launch.block_partials(blocks, 10, wp.float64, device)
    _launch.launch_tiled(
        kernel_measures.moment_integrals,
        dim=[blocks],
        inputs=[vertices, faces, chunk_faces, stride, totals],
        block_dim=TILE_1D,
        device=device,
    )

    # Every return here is a host-side value, so the ten sums have to cross the device boundary --
    # but only the ten, and in one readback rather than one per group.
    totals_np = np.asarray(read_values(totals, 0, 10), dtype=np.float64)
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


def euler_characteristic(faces: wp.array[wp.int32], *, n_vertices: int | None = None) -> int:
    """
    Euler characteristic ``V - E + F`` of the mesh (topological invariant).

    Counts distinct referenced vertices, unique undirected edges, and faces, matching
    [`trimesh.Trimesh.euler_number`][] (which uses referenced vertices, ``edges_unique``, and
    faces). For a closed genus-``g`` surface this equals ``2 - 2 * g``.

    Parameters
    ----------
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    n_vertices
        A bound on the indices in ``faces`` (the vertex count), to spare the reduction and
        readback that otherwise derive it. Trusted, not checked: it must exceed every index in
        ``faces``.

    Returns
    -------
    int
        ``(#distinct referenced vertices) - (#unique edges) + (#faces)``. ``0`` for an empty mesh.

    Raises
    ------
    ValueError
        If ``n_vertices`` is given and not positive, or, when it is inferred, if ``faces`` holds a
        negative index.

    See Also
    --------
    [`unique_edge_count`][ordito.edges.unique_edge_count]
    [`trimesh.Trimesh.euler_number`][]

    Examples
    --------
    A sphere has ``chi = 2`` and a torus ``chi = 0``:

    ```python
    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    print(od.measures.euler_characteristic(faces))  # 2

    vertices, faces = od.creation.torus(1.0, 0.3, major_sections=16, minor_sections=8)
    print(od.measures.euler_characteristic(faces))  # 0
    ```
    """
    n_faces = faces.size // 3
    if n_faces == 0:
        return 0

    # The bound is taken once, here, so ``edges_unique`` re-checking it would reduce the same
    # indices twice; ``require_non_negative`` keeps the half of that check a derived bound cannot
    # give, and costs nothing -- both ends come out of the one reduction.
    if n_vertices is None:
        n_vertices = od.array.index_bound(faces, require_non_negative=True)
    elif n_vertices <= 0:
        raise ValueError(f"n_vertices must be positive, got {n_vertices}")
    device = faces.device
    n = 3 * n_faces
    # The referenced-vertex count is a membership mask's popcount rather than the length of
    # ``unique_1d(faces)``: the indices are bounded by ``n_vertices``, so the mask is the same set
    # without the hash table, compaction and sort, and the count is one read instead of two. The
    # mask is marked in the pass that packs the halfedge keys into the sort's buffer, over the
    # bits a key below ``n_vertices ** 2`` can set; the sort's payload is never read.
    referenced = _launch.zeros(n_vertices, dtype=wp.bool, device=device)
    keys, payload = _launch.sort_buffers(n, wp.uint64, device)
    _launch.launch(
        kernel_measures.face_edge_keys_and_referenced,
        dim=n_faces,
        inputs=[faces, wp.uint64(n_vertices), keys, referenced],
        device=device,
    )
    _launch.radix_sort_pairs(
        keys, payload, count=n, end_bit=min(64, max(1, (n_vertices * n_vertices - 1).bit_length()))
    )
    sorted_keys = _launch.view(keys, 0, n)
    # The unique edges are the sorted halfedge keys' runs, so only their count is formed: no run
    # scan and no emitted edge table. Both counts fold in one launch and come back in one read.
    counts = _launch.zeros(2, dtype=wp.int32, device=device)
    _launch.launch_tiled(
        kernel_measures.referenced_vertex_and_unique_edge_counts,
        dim=[kernel_reduce.blocks_1d(max(n_vertices, n))],
        inputs=[referenced, sorted_keys, max(n_vertices, n), counts],
        block_dim=TILE_1D,
        device=device,
    )
    # The two counts are the answer.
    n_referenced, n_edges = read_values(counts, 0, 2)
    return n_referenced - n_edges + n_faces
