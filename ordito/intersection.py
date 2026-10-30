"""
Cross-sections and level sets: where a mesh meets a plane, another mesh, or an isovalue.

A plane section and an isocontour are the same operation seen twice: intersecting a mesh with a
plane is exactly extracting the zero level set of that plane's signed distance. So
[`mesh_with_plane`][ordito.intersection.mesh_with_plane] and
[`marching_triangles`][ordito.intersection.marching_triangles] sit side by side here — the first
takes the field implicitly as a plane, the second takes any per-vertex scalar field — and both emit
the [`ordito.polyline`][ordito.polyline] convention: one array of points per curve, closed curves
not repeating their first point, and a parallel list of closed flags.

[`mesh_with_mesh`][ordito.intersection.mesh_with_mesh] is the genuine intersection in the set
sense, and [`slice_mesh_with_plane`][ordito.intersection.slice_mesh_with_plane] keeps the cut
geometry rather than the curve.

Every entry point here that takes a plane takes it as ``(plane_normal, plane_origin)``, in that
order, and so does every plane argument elsewhere in the package. Both are ``wp.vec3``, so a
transposed call type-checks and sections the wrong plane -- pass them by keyword where the call site
is not obvious. [`points.fit_plane`][ordito.points.fit_plane] returns the pair in this order, so
its result splats straight into any of them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeVar, cast, overload

import numpy as np
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._device import read_scalar, require_nonempty_mesh, require_same_device
from ordito.constants import TILE_1D, TOLERANCE_MERGE
from ordito.kernels import graph as kernel_graph
from ordito.kernels import grouping as kernel_grouping
from ordito.kernels import intersection as kernel_intersections
from ordito.kernels import points as kernel_points
from ordito.kernels import triangles as kernel_triangles

if TYPE_CHECKING:
    import numpy.typing as npt

DType = TypeVar("DType")

# Segment count from which ``marching_triangles`` links its curves on the device.
_LINK_ON_DEVICE_FROM = 1024

# Largest distance from its best-fit plane, relative to the loop's extent, at which a capped clip's
# section loop counts as planar and is ear-clipped. Crossing points of a plane field sit on the
# plane to ``float32`` interpolation error, orders of magnitude below this.
_CAP_PLANAR_TOLERANCE = 1e-5

# Most broad-phase candidate slots: slot indices, launch dimensions and the narrow phase's flag
# scan are all ``int32``.
_MAX_CANDIDATE_SLOTS = 2**31 - 1


def segments_with_plane(
    start_points: wp.array[wp.vec3],
    end_points: wp.array[wp.vec3],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    *,
    line_segments: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.bool]]:
    """
    Calculate plane-line intersections for batched segment endpoints.

    Each row pair ``(start_points[i], end_points[i])`` defines one line to test.
    Matches [`trimesh.intersections.plane_lines`][] with Trimesh's ``(2, n, 3)``
    layout expressed as two length-``n`` ``wp.vec3`` arrays.

    Parameters
    ----------
    plane_normal
        Plane normal vector.
    plane_origin
        Point on the plane.
    start_points
        ``(n,)`` first endpoint of each segment.
    end_points
        ``(n,)`` second endpoint of each segment.
    line_segments
        When ``True``, only mark intersections valid if endpoints lie on
        different sides of the plane.

    Returns
    -------
    intersections
        ``(n,)`` intersection points (undefined where ``valid`` is ``False``).
    valid
        ``(n,)`` mask indicating a valid intersection per segment.

    Raises
    ------
    ValueError
        If ``start_points`` and ``end_points`` do not have the same shape.
    RuntimeError
        If ``start_points`` and ``end_points`` are not all on one device.

    Examples
    --------
    The plane ``z = 0`` against a segment crossing it and one wholly above it, whose line still
    meets the plane once ``line_segments=False``:

    ```python
    import warp as wp

    import ordito as od

    starts = wp.array([[0.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=wp.vec3)
    ends = wp.array([[0.0, 0.0, 3.0], [1.0, 0.0, 2.0]], dtype=wp.vec3)
    normal, origin = wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0)
    points, valid = od.intersection.segments_with_plane(starts, ends, normal, origin)
    print(valid.numpy().tolist())  # [True, False]
    print(points.numpy()[0])  # [0. 0. 0.]
    _, valid = od.intersection.segments_with_plane(
        starts, ends, normal, origin, line_segments=False
    )
    print(valid.numpy().tolist())  # [True, True]
    ```
    """
    require_same_device(start_points=start_points, end_points=end_points)
    if start_points.shape != end_points.shape:
        raise ValueError("start_points and end_points must have the same shape")
    n = start_points.size
    device = start_points.device
    intersections = _launch.empty(n, dtype=wp.vec3, device=device)
    valid = _launch.empty(n, dtype=wp.bool, device=device)
    if n == 0:
        return intersections, valid

    _launch.map(
        kernel_intersections.plane_with_line,
        plane_normal,
        plane_origin,
        start_points,
        end_points,
        wp.bool(line_segments),
        out=[intersections, valid],
    )
    return intersections, valid


@overload
def mesh_with_plane(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    *,
    return_faces: Literal[False] = False,
) -> odt.Array2dVec3: ...
@overload
def mesh_with_plane(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    *,
    return_faces: Literal[True],
) -> tuple[odt.Array2dVec3, wp.array[wp.int32]]: ...
def mesh_with_plane(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    *,
    return_faces: bool = False,
) -> odt.Array2dVec3 | tuple[odt.Array2dVec3, wp.array[wp.int32]]:
    """
    Intersect a mesh with a plane, returning line segments on the plane.

    Matches [`trimesh.intersections.mesh_plane`][] for indexed triangle meshes.
    To section a face subset, extract a submesh first (e.g.
    [`submesh_from_face_indices`][ordito.selection.submesh_from_face_indices]).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    plane_normal
        Normal vector of the plane.
    plane_origin
        Point on the plane.
    return_faces
        If ``True``, also return the source face index for each segment.

    Returns
    -------
    lines
        ``(m, 2)`` segment endpoints, one row per segment (logical shape ``(m, 2, 3)``).
    face_index
        ``(m,)`` source face indices into the mesh. Returned only when ``return_faces=True``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Examples
    --------
    A horizontal plane through a unit cube cuts each of its four sides, two triangles each, and the
    section is the unit square of perimeter 4:

    ```python
    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    lines, face_index = od.intersection.mesh_with_plane(
        vertices, faces, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.1), return_faces=True
    )
    print(lines.shape, face_index.shape)  # (8, 2) (8,)
    segments = lines.numpy()
    print(f"{np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).sum():.4f}")  # 4.0000
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        empty_segments = odt.empty_2d((0, 2), wp.vec3, device=device)
        if return_faces:
            return empty_segments, _launch.empty(0, dtype=wp.int32, device=device)
        return empty_segments

    cut = _launch.empty(n_faces, dtype=wp.int32, device=device)
    segments = odt.empty_2d((n_faces, 2), wp.vec3, device=device)
    _launch.launch(
        kernel_intersections.mesh_with_plane_segments,
        dim=n_faces,
        inputs=[vertices, faces, plane_normal, plane_origin, cut, segments],
        device=device,
    )

    lines, hit_faces = _compact_cut_segments(cut, segments, return_rows=return_faces)
    if not return_faces:
        return lines
    assert hit_faces is not None
    return lines, hit_faces


def marching_triangles(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float32] | wp.array[wp.float64],
    isovalue: float = 0.0,
    n_vertices: int | None = None,
) -> tuple[list[wp.array[wp.vec3]], list[bool]]:
    """
    Extract the ``values == isovalue`` level set as a list of polylines.

    Each returned curve is a connected component of the level set, oriented so that the region where
    ``values > isovalue`` lies to its left (with the vertex normals as up). Curves close up unless
    they run into a mesh boundary, so an open curve begins and ends on a boundary edge.

    A value equal to the isovalue counts as positive, which keeps every cut face at exactly one
    segment; a contour running exactly through a vertex therefore yields zero-length segments rather
    than an ambiguous junction.

    Crossings are matched by the vertex pair of the edge they lie on, and the segments are linked
    into curves by pointer-jumping list ranking over that successor graph (the shape
    [`boundary_loops`][ordito.boundary.boundary_loops] solves too): on the device once the level
    set is large, on the host over one readback of the crossing keys while it is small enough for a
    handful of launches to cost more than the walk.

    !!! note "The returned arrays are views"
        Every curve slices one packed buffer, so holding a single curve keeps them all alive and
        writing into one writes into the shared allocation. ``wp.clone`` a curve for an independent
        buffer.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    values
        ``(n_vertices,)`` scalar field, ``wp.float32`` or ``wp.float64``. A ``float64`` field (what
        [`heat_geodesic`][ordito.heat.heat_geodesic] returns) is interpolated in
        ``float64``.
    isovalue
        Level to extract.
    n_vertices
        Total vertex count, used as the base that packs a crossed edge's two vertex indices into
        the key matching it across the two faces that share it. When ``None`` the length of
        ``vertices`` is used, which bounds every index the faces may reference.

    Returns
    -------
    curves : list[wp.array[wp.vec3]]
        One array of points per level-set component, in order along the curve, on
        ``vertices.device``. Closed curves do not repeat their first point.
    closed : list[bool]
        Whether each curve is a closed loop.

    Raises
    ------
    ValueError
        If two segments start on the same mesh edge, which means the faces are not consistently
        oriented (the level set cannot then be linked into oriented curves). Repair the winding with
        [`make_winding_consistent`][ordito.repair.make_winding_consistent] first.
    RuntimeError
        If ``vertices``, ``faces`` and ``values`` are not all on one device.

    See Also
    --------
    [`marching_triangles_with_offsets`][ordito.intersection.marching_triangles_with_offsets]
    [`clip_mesh_with_field`][ordito.intersection.clip_mesh_with_field]
    [`mesh_with_plane`][ordito.intersection.mesh_with_plane]
    [`heat_geodesic`][ordito.heat.heat_geodesic]
    [`polyline_length`][ordito.polyline.polyline_length]
    ``potpourri3d.MarchingTrianglesSolver``

    Examples
    --------
    The height field ``z`` on a unit sphere: its zero level set is the equator, one closed curve of
    length close to ``2 * pi``:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3)
    heights = wp.array(vertices.numpy()[:, 2], dtype=wp.float32)
    curves, closed = od.intersection.marching_triangles(vertices, faces, heights, 0.0)
    print(len(curves), closed)  # 1 [True]
    print(f"{od.polyline.polyline_length(curves[0], closed=True):.2f}")  # 6.27
    ```
    """
    linked = _marching_curves(vertices, faces, values, isovalue, n_vertices)
    if linked is None:
        return [], []
    points, bounds, closed = linked
    if isinstance(points, np.ndarray):
        points = _launch.array(points, dtype=wp.vec3, device=vertices.device)
    return od.array.split(points, bounds), closed


def marching_triangles_with_offsets(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float32] | wp.array[wp.float64],
    isovalue: float = 0.0,
    n_vertices: int | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Extract the ``values == isovalue`` level set as packed polylines: points, offsets, closed.

    Each returned curve is a connected component of the level set, oriented so that the region where
    ``values > isovalue`` lies to its left (with the vertex normals as up). Curves close up unless
    they run into a mesh boundary, so an open curve begins and ends on a boundary edge.

    A value equal to the isovalue counts as positive, which keeps every cut face at exactly one
    segment; a contour running exactly through a vertex therefore yields zero-length segments rather
    than an ambiguous junction.

    Crossings are matched by the vertex pair of the edge they lie on, and the segments are linked
    into curves by pointer-jumping list ranking over that successor graph (the shape
    [`boundary_loops`][ordito.boundary.boundary_loops] solves too): on the device once the level
    set is large, on the host over one readback of the crossing keys while it is small enough for a
    handful of launches to cost more than the walk.

    The packed form of [`marching_triangles`][ordito.intersection.marching_triangles]: the same
    curves in the same order, concatenated, with no per-curve Python object.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` triangle index buffer.
    values
        ``(n_vertices,)`` scalar field, ``wp.float32`` or ``wp.float64``. A ``float64`` field (what
        [`heat_geodesic`][ordito.heat.heat_geodesic] returns) is interpolated in
        ``float64``.
    isovalue
        Level to extract.
    n_vertices
        Total vertex count, used as the base that packs a crossed edge's two vertex indices into
        the key matching it across the two faces that share it. When ``None`` the length of
        ``vertices`` is used, which bounds every index the faces may reference.

    Returns
    -------
    points : wp.array[wp.vec3]
        ``(n_curve_points,)`` every curve's points in order along the curve, curve after curve, on
        ``vertices.device``.
        Closed curves do not repeat their first point.
    offsets : wp.array[wp.int32]
        ``(n_curves + 1,)`` total-terminated offsets: curve ``c`` is
        ``points[offsets[c] : offsets[c + 1]]``, and ``[0]`` when the level set is empty.
    closed : wp.array[wp.bool]
        ``(n_curves,)`` whether each curve is a closed loop.

    Raises
    ------
    ValueError
        If two segments start on the same mesh edge, which means the faces are not consistently
        oriented (the level set cannot then be linked into oriented curves). Repair the winding with
        [`make_winding_consistent`][ordito.repair.make_winding_consistent] first.
    RuntimeError
        If ``vertices``, ``faces`` and ``values`` are not all on one device.

    See Also
    --------
    [`marching_triangles`][ordito.intersection.marching_triangles]
    [`clip_mesh_with_field`][ordito.intersection.clip_mesh_with_field]
    [`mesh_with_plane`][ordito.intersection.mesh_with_plane]
    [`heat_geodesic`][ordito.heat.heat_geodesic]
    [`polyline_length`][ordito.polyline.polyline_length]
    ``potpourri3d.MarchingTrianglesSolver``

    Examples
    --------
    The equator of a unit sphere as one packed closed curve:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3)
    heights = wp.array(vertices.numpy()[:, 2], dtype=wp.float32)
    points, offsets, closed = od.intersection.marching_triangles_with_offsets(
        vertices, faces, heights
    )
    print(offsets.numpy().tolist(), closed.numpy().tolist())  # [0, 94] [True]
    print(points.shape)  # (94,)
    ```
    """
    device = vertices.device
    linked = _marching_curves(vertices, faces, values, isovalue, n_vertices)
    if linked is None:
        return (
            *_launch.empty_packed(wp.vec3, device),
            _launch.empty(0, dtype=wp.bool, device=device),
        )
    return _upload_curves(*linked, device)


def _upload_curves(
    points: wp.array[wp.vec3] | np.ndarray,
    bounds: list[int],
    closed: list[bool],
    device: wp.DeviceLike,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Upload a link's host-side results -- offsets, closed flags, host-link points -- in one copy.

    Both links finish with the curve bounds on the host (they size the output), so the offsets and
    flags -- and the host link's points -- are packed into one byte buffer, uploaded once and
    addressed through typed views that keep it alive: offsets first (4-byte aligned), the points'
    ``float32`` words after them, the one-byte flags last.
    """
    parts = [np.asarray(bounds, dtype=np.int32).view(np.uint8)]
    n_points = 0
    if isinstance(points, np.ndarray):
        n_points = int(points.shape[0])
        parts.append(np.ascontiguousarray(points, dtype=np.float32).reshape(-1).view(np.uint8))
    parts.append(np.asarray(closed, dtype=np.bool_).view(np.uint8))
    buffer = _launch.array(np.concatenate(parts), dtype=wp.uint8, device=device)
    offsets_bytes = 4 * len(bounds)
    offsets = _byte_view(buffer, 0, wp.int32, len(bounds))
    if isinstance(points, np.ndarray):
        points = _byte_view(buffer, offsets_bytes, wp.vec3, n_points)
    flags = _byte_view(buffer, offsets_bytes + 12 * n_points, wp.bool, len(closed))
    return points, offsets, flags


def _byte_view(
    buffer: wp.array[wp.uint8], start: int, dtype: type[DType], count: int
) -> wp.array[DType]:
    """Return ``count`` ``dtype`` elements of ``buffer`` from byte ``start`` as a view of it."""
    assert buffer.ptr is not None
    view = wp.array(ptr=int(buffer.ptr) + start, dtype=dtype, shape=(count,), device=buffer.device)
    # Warp's own back-reference from a view to the array it keeps alive, as ``flatten`` sets it.
    view._ref = buffer  # pyright: ignore[reportPrivateUsage]
    return view


def _marching_curves(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float32] | wp.array[wp.float64],
    isovalue: float,
    n_vertices: int | None,
) -> tuple[wp.array[wp.vec3] | np.ndarray, list[int], list[bool]] | None:
    """
    Cut and link the level set into ``(points, bounds, closed)``, ``None`` when it is empty.

    ``points`` is a device array from the device link and the ``(n_points, 3)`` ``float32`` host
    array from the host link, which leaves the upload to the caller so that everything it returns
    can travel in one copy (``_upload_curves``).
    """
    require_same_device(vertices=vertices, faces=faces, values=values)
    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return None

    # The key base only has to exceed every vertex index the faces reference; ``vertices`` is the
    # buffer they index, so its length is the bound whenever the caller did not give one.
    key_base = int(n_vertices) if n_vertices is not None else vertices.size
    cut = _launch.empty(n_faces, dtype=wp.int32, device=device)
    segments = odt.empty_2d((n_faces, 2), wp.vec3, device=device)
    segment_edges = odt.empty_2d((n_faces, 2), wp.int64, device=device)
    _launch.launch(
        kernel_intersections.MARCHING_TRIANGLES_SEGMENTS[values.dtype],
        dim=n_faces,
        inputs=[
            vertices,
            faces,
            values,
            values.dtype(isovalue),
            wp.int64(key_base),
            cut,
            segments,
            segment_edges,
        ],
        device=device,
    )

    n_segments = _scan_cut_flags(cut)
    if n_segments == 0:
        return None
    return _link_curves(cut, segments, segment_edges, n_segments, key_base)


def _link_curves(
    inclusive: wp.array[wp.int32],
    segments: odt.Array2dVec3,
    segment_edges: wp.array[wp.int64, Literal[2]],
    n_segments: int,
    key_base: int,
) -> tuple[wp.array[wp.vec3] | np.ndarray, list[int], list[bool]]:
    """
    Compact the cut segments and chain them into curves: ``(points, bounds, closed)``.

    ``points`` packs every curve in order, curve ``c`` occupying ``points[bounds[c] :
    bounds[c + 1]]``. Above ``_LINK_ON_DEVICE_FROM`` segments on a CUDA device the chaining runs
    on the device (``_link_on_device``), below it -- where a handful of launches outweigh the host
    walk -- and on the CPU device, whose serial launches never beat NumPy's walk, through
    ``_link_segments``; both produce the identical packing.
    """
    device = cast("wp.Device", inclusive.device)
    if _links_on_device(device, n_segments):
        return _link_on_device(inclusive, segments, segment_edges, n_segments, key_base)
    n = n_segments
    # The host link needs the kept segments' keys *and* their endpoints, so both are compacted into
    # one buffer and read back together: ``2 n`` keys, then the ``2 n`` endpoint positions in the
    # remaining ``24 n`` bytes, which the ``wp.vec3`` view below addresses. The host then gathers
    # the curves' points itself and the caller uploads them, where a device gather would need the
    # slots uploaded, an output allocated and an indexed copy.
    record = _launch.empty(5 * n, dtype=wp.int64, device=device)
    assert record.ptr is not None
    record_points = wp.array(
        ptr=int(record.ptr) + 16 * n, dtype=wp.vec3, shape=(n, 2), device=device
    )
    _launch.launch(
        kernel_intersections.compact_cut_segment_keys,
        dim=inclusive.size,
        inputs=[inclusive, segments, segment_edges, record_points, record, None],
        device=device,
    )
    # The one readback of the host link.
    table = record.numpy()
    points_np = table[2 * n :].view(np.float32).reshape(2 * n, 3)
    return _link_on_host(points_np, table[: 2 * n].reshape(n, 2))


def _links_on_device(device: wp.Device, n_segments: int) -> bool:
    """
    Whether ``_link_curves`` chains ``n_segments`` segments on the device, not the host.

    A function rather than an inline test so the tests can force either link on either device.
    """
    return device.is_cuda and n_segments >= _LINK_ON_DEVICE_FROM


def _link_on_device(
    inclusive: wp.array[wp.int32],
    segments: odt.Array2dVec3,
    segment_edges: wp.array[wp.int64, Literal[2]],
    n_segments: int,
    key_base: int,
) -> tuple[wp.array[wp.vec3] | np.ndarray, list[int], list[bool]]:
    """
    Device counterpart of ``_link_on_host``, with the same result.

    The compaction writes the ``2 n`` endpoint keys straight into a radix sort, whose equal
    adjacent keys pair each segment's incoming endpoint with its successor's outgoing one; ranking
    along the curves is multi-hop pointer jumping (``kernels/intersection.link_rank_round``), and
    one inclusive scan of per-curve weights lays the curves out in the host link's order. One
    readback of that scan sizes the output and gives the bounds; a level set the device cannot
    pair cleanly is handed to the host link, which judges it exactly as it always did.
    """
    device = inclusive.device
    n = n_segments
    m = 2 * n
    hit_segments = odt.empty_2d((n, 2), wp.vec3, device=device)
    keys = _launch.empty(2 * m, dtype=wp.int64, device=device)
    endpoints = _launch.empty(2 * m, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_intersections.compact_cut_segment_keys,
        dim=inclusive.size,
        inputs=[inclusive, segments, segment_edges, hit_segments, keys, endpoints],
        device=device,
    )
    # Every key packs two vertex indices below ``key_base``, so only its low bits need sorting.
    end_bit = max(1, (key_base * key_base - 1).bit_length())
    _launch.radix_sort_pairs(keys, endpoints, m, end_bit=end_bit)

    link = _launch.empty(m, dtype=wp.int32, device=device)
    state = _launch.empty(n, dtype=wp.vec4i, device=device)
    # The per-curve weights, then their scan, with the malformed-input flag in the last slot.
    weights = _launch.zeros(m + 1, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_intersections.link_sorted_endpoints,
        dim=m,
        inputs=[keys, endpoints, m, m, link, state, weights],
        device=device,
    )
    # Rank until the windows cover the longest possible curve, all ``n`` segments, in the fewest
    # launches and, for that count, the fewest chased pointers (``graph.pointer_jump_schedule``).
    scratch = _launch.empty(n, dtype=wp.vec4i, device=device)
    hops, rounds = kernel_graph.pointer_jump_schedule(n)
    width = 1
    for _round in range(rounds):
        _launch.launch(
            kernel_intersections.link_rank_round,
            dim=n,
            inputs=[state, hops, width, scratch],
            device=device,
        )
        state, scratch = scratch, state
        width *= hops

    head_of_tail = _launch.empty(n, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_intersections.link_curve_weights,
        dim=n,
        inputs=[link, state, n, weights, head_of_tail],
        device=device,
    )
    bounds_view = weights[:m]
    _launch.array_scan(bounds_view, out_array=bounds_view, inclusive=True)
    # The one readback of the device link: the scanned weights size the output and bound every
    # curve, and the last slot says whether the pairing was clean.
    table = weights.numpy()
    if table[m] != 0:
        # Undo the sort to recover each endpoint's key, and let the host link decide.
        edges_np = np.empty(m, dtype=np.int64)
        np.put(
            edges_np,
            cast("npt.NDArray[np.int32]", odt.as_dense(endpoints[:m]).numpy()),
            odt.as_dense(keys[:m]).numpy(),
        )
        points_np = hit_segments.numpy().reshape(m, 3)
        return _link_on_host(points_np, edges_np.reshape(n, 2))
    scan = table[:m]
    names = np.flatnonzero(np.diff(scan, prepend=0))
    points = _launch.empty(int(scan[-1]), dtype=wp.vec3, device=device)
    _launch.launch(
        kernel_intersections.emit_linked_curves,
        dim=n,
        inputs=[hit_segments, link, state, weights, head_of_tail, n, points],
        device=device,
    )
    return points, [0, *scan[names].tolist()], (names >= n).tolist()


def _link_on_host(
    points_np: np.ndarray, edges_np: np.ndarray
) -> tuple[wp.array[wp.vec3] | np.ndarray, list[int], list[bool]]:
    """
    Link ``_link_segments``' curves on the host and gather their points there.

    ``points_np`` is the ``(2 n, 3)`` ``float32`` endpoint buffer the slots index: endpoint ``e``
    of kept segment ``i`` at row ``2 i + e``. The packed points stay on the host for the caller's
    one upload.
    """
    slots_np, starts_np, closed = _link_segments(edges_np)
    return points_np[slots_np], [*starts_np.tolist(), slots_np.size], closed


def _link_segments(segment_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """
    Chain oriented segments into curves, returning endpoint slots, curve starts and closed flags.

    ``segment_edges[i]`` holds the unique-edge ids the two endpoints of segment ``i`` lie on. Since
    the segments are consistently oriented, an interior crossing edge appears once as some segment's
    outgoing endpoint and once as another's incoming endpoint, so the successor relation is a
    permutation on all but the boundary-terminated chains -- the same successor-graph structure
    [`boundary_loops`][ordito.boundary.boundary_loops] walks.

    Slots index the flattened endpoint buffer: endpoint ``e`` of segment ``i`` is ``2 * i + e``.
    Curves come out in the same order the serial walk produced: the open ones first, each from its
    unique predecessor-free segment, then the closed ones, each entered at its lowest-indexed
    segment so the result does not depend on face order.

    **Nothing here iterates per segment.** The successor comes from a lookup table rather than a
    sort -- an endpoint's densified edge id indexes "which segment starts here", three ``O(n)``
    passes -- and the ordering is two passes of Wyllie pointer doubling, vectorized in NumPy rather
    than one Python iteration per segment.
    """
    n = int(segment_edges.shape[0])
    # The kernel hands back a *sparse* edge key per endpoint -- the crossed edge's packed vertex
    # pair -- not a dense id, so the lookup table below is indexed through one ``np.unique`` over
    # the crossing endpoints alone. Densifying on the *device* instead, over every edge of the mesh
    # through ``edges.edges_unique_inverse``, is a large share of ``marching_triangles`` and is
    # sized by the mesh where a contour crosses a small fraction of its edges. Here it grows with
    # the level set, which is a clear win until a contour touches enough edges for a host sort of
    # them to cost what a device pass over all of them would.
    #
    # **A sorted join was tried here instead and is slower at every size** -- ``argsort`` the start
    # keys, ``searchsorted`` the end keys into them, no densification and no ``owner`` table at all.
    # The reason is not the extra gathers: the join needs **two** sort-class passes where this one
    # needs one. ``np.searchsorted`` of n keys into n costs about what ``np.unique`` over ``2n``
    # does, because a binary search is ``n log n`` *dependent, cache-missing* probes rather than a
    # streaming sort, and ``np.argsort`` is several times ``np.sort`` because it permutes indices
    # through indirect comparisons -- which the join needs specifically, to recover *which* segment
    # won.
    #
    # So densifying is not overhead paid to enable the table: the dense labels make the join itself
    # a single O(1)-per-element lookup, which is worth more than the search it avoids. Do not
    # re-propose the join.
    dense = np.unique(segment_edges.reshape(-1), return_inverse=True)[1].reshape(-1, 2)
    start_edge = np.ascontiguousarray(dense[:, 0])
    end_edge = np.ascontiguousarray(dense[:, 1])
    index = np.arange(n, dtype=np.int64)

    # ``owner[e]`` is the segment whose *outgoing* endpoint lies on edge ``e``, so the successor of
    # segment ``i`` is whoever owns ``i``'s incoming edge. A second segment claiming an edge
    # overwrites the first, and the loser then fails to find itself -- which is exactly the
    # inconsistent-winding case, detected without a duplicate scan of its own.
    owner = np.full(int(max(start_edge.max(), end_edge.max())) + 1, -1, dtype=np.int64)
    owner[start_edge] = index
    if not np.array_equal(owner[start_edge], index):
        raise ValueError(
            "marching_triangles cannot link the level set: two segments start on the same edge, "
            "which means the faces are not consistently oriented."
        )
    successor = owner[end_edge]
    rounds = max(1, math.ceil(math.log2(max(n, 2))))

    # Pass 1: pointer doubling with a fixed point at every open curve's last segment, carrying the
    # smallest index seen along the way. A segment on an open curve lands on that curve's end; one
    # on a closed curve never does, and its window wraps, so its minimum becomes the whole loop's.
    ahead = np.where(successor >= 0, successor, index)
    lowest = np.minimum(index, ahead)
    for _ in range(rounds):
        lowest = np.minimum(lowest, lowest[ahead])
        ahead = ahead[ahead]
    is_closed = successor[ahead] >= 0

    # Cut every loop at its lowest-indexed segment, which turns it into a chain headed there. After
    # this every curve is a chain, so one ranking pass covers both kinds.
    entry = is_closed & (lowest == index)
    cut = successor.copy()
    cut[(successor >= 0) & entry[np.maximum(successor, 0)]] = -1

    has_predecessor = np.zeros(n, dtype=bool)
    has_predecessor[cut[cut >= 0]] = True
    is_head = ~has_predecessor

    # Pass 2: the same doubling on the cut graph gives each segment its hop count to its curve's
    # last segment. It stops as soon as every pointer has reached one, which is ``log2`` of the
    # *longest curve* rather than of the segment count -- 6 rounds against 15 on a 26k-segment
    # level set whose curves are 34 segments at their longest.
    tail = np.where(cut >= 0, cut, index)
    steps = (cut >= 0).astype(np.int64)
    for _ in range(rounds):
        if not (cut[tail] >= 0).any():
            break
        steps = steps + steps[tail]
        tail = tail[tail]

    # A curve is named by its head, reached from any of its segments through its shared last one.
    head_by_tail = np.empty(n, dtype=np.int64)
    head_by_tail[tail[is_head]] = index[is_head]
    head_of = head_by_tail[tail]
    sizes = np.bincount(head_of, minlength=n)
    position = sizes[head_of] - 1 - steps

    heads = np.flatnonzero(is_head)
    head_closed = is_closed[heads]
    curve_heads = np.concatenate([heads[~head_closed], heads[head_closed]])
    closed = is_closed[curve_heads]
    curve_of_head = np.empty(n, dtype=np.int64)
    curve_of_head[curve_heads] = np.arange(curve_heads.size)
    curve = curve_of_head[head_of]

    # An open curve carries one extra slot: its last segment contributes both endpoints.
    slot_counts = sizes[curve_heads] + ~closed
    starts = np.concatenate([[0], np.cumsum(slot_counts)[:-1]]).astype(np.int32)
    slots = np.empty(int(slot_counts.sum()), dtype=np.int32)
    slots[starts[curve] + position] = 2 * index
    last = successor < 0
    slots[starts[curve[last]] + position[last] + 1] = 2 * index[last] + 1
    return slots, starts, closed.tolist()


def mesh_with_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    max_triangle_collisions: int = 16,
    mesh_a: wp.Mesh | None = None,
    mesh_b: wp.Mesh | None = None,
) -> odt.Array2dVec3:
    """
    Intersect two meshes, returning line segments along the intersection curve(s).

    Broad phase builds a ``wp.Mesh`` over the mesh with fewer faces and queries
    triangle AABBs via ``wp.mesh_query_aabb``; each triangle of the other mesh
    supplies the query box. Narrow phase runs Moller's interval test and clips the intersection
    line to both triangles. Coplanar overlapping faces produce no segments.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` and ``(3 * n_faces_a,)`` first indexed triangle mesh.
    vertices_b, faces_b
        ``(n_vertices_b,)`` and ``(3 * n_faces_b,)`` second indexed triangle mesh.
    max_triangle_collisions
        Broad-phase slots reserved per query triangle. Not a cap: the candidates of a triangle with
        more go to an overflow list, so the answer does not depend on it.
    mesh_a, mesh_b
        A ``wp.Mesh`` already built over ``vertices_a`` / ``faces_a`` (respectively ``vertices_b``
        / ``faces_b``), to spare the BVH build, as
        [`mesh_collision_pairs`][ordito.intersection.mesh_collision_pairs] takes them: only the
        mesh with fewer faces (``a`` on a tie) is read, and neither is checked against the arrays.
        [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] is what to pass.

    Returns
    -------
    lines
        ``(m, 2)`` segment endpoints, one row per segment (logical shape ``(m, 2, 3)``).

    Raises
    ------
    ValueError
        If ``max_triangle_collisions`` is less than 1.
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b``, ``faces_b``, ``mesh_a`` and ``mesh_b`` are
        not all on one device.

    Examples
    --------
    Two overlapping unit cubes; the intersection curve has total length 4:

    ```python
    import numpy as np

    import ordito as od

    vertices_a, faces_a = od.creation.box()
    vertices_b, faces_b = od.creation.box(bounds=[[0.0, -0.2, -0.3], [1.0, 0.8, 0.7]])
    lines = od.intersection.mesh_with_mesh(vertices_a, faces_a, vertices_b, faces_b)
    print(lines.shape)  # (14, 2)
    segments = lines.numpy()
    print(f"{np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).sum():.4f}")  # 4.0000
    ```
    """
    require_same_device(
        vertices_a=vertices_a,
        faces_a=faces_a,
        vertices_b=vertices_b,
        faces_b=faces_b,
        mesh_a=mesh_a,
        mesh_b=mesh_b,
    )
    device = vertices_a.device
    candidates = _candidate_face_pairs(
        vertices_a,
        faces_a,
        vertices_b,
        faces_b,
        max_triangle_collisions,
        "mesh_with_mesh",
        mesh_a=mesh_a,
        mesh_b=mesh_b,
    )
    if candidates is None:
        return odt.empty_2d((0, 2), wp.vec3, device=device)
    segments = _compact_crossing_candidates(candidates, segments=True)
    if segments is None:
        return odt.empty_2d((0, 2), wp.vec3, device=device)
    return odt.as_array2d(segments, wp.vec3)


def mesh_collision_pairs(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    max_triangle_collisions: int = 16,
    mesh_a: wp.Mesh | None = None,
    mesh_b: wp.Mesh | None = None,
) -> odt.Array2dInt32:
    """
    Which faces of two meshes cross each other, as index pairs.

    The **collision** question, as against
    [`mesh_with_mesh`][ordito.intersection.mesh_with_mesh]'s intersection *curve*: this answers
    "do these two touch, and where" without computing the geometry of the contact, which is what a
    contact resolver, a fit check or an assembly validator wants.
    [`ordito.validation.face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask]
    is the same question asked of one mesh against itself.

    Parameters
    ----------
    vertices_a, faces_a
        ``(n_vertices_a,)`` positions and ``(3 * n_faces_a,)`` index buffer of the first mesh.
    vertices_b, faces_b
        ``(n_vertices_b,)`` and ``(3 * n_faces_b,)`` second mesh, in the same form.
    max_triangle_collisions
        Broad-phase slots reserved per query triangle. Not a cap: the candidates of a triangle with
        more go to an overflow list, so the answer does not depend on it.
    mesh_a, mesh_b
        A ``wp.Mesh`` already built over ``vertices_a`` / ``faces_a`` (respectively ``vertices_b``
        / ``faces_b``), to spare the BVH build this otherwise pays on every call. Only the mesh with
        fewer faces (``a`` on a tie) supplies the BVH, so only that one is read; the other may be
        passed and is ignored. Purely an optimization: the answer is identical either way, and it
        is not checked against the arrays -- a mesh over *different* geometry silently answers for
        that geometry. [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] is a cached property
        and is what to pass.

    Returns
    -------
    odt.Array2dInt32
        ``(n_pairs, 2)`` face-index pairs, column 0 into ``faces_a`` and column 1 into ``faces_b``.
        Empty ``(0, 2)`` when the meshes do not cross. Pairs that merely touch or are coplanar are
        **not** collisions -- the same convention
        [`ordito.validation.is_self_intersecting`][ordito.validation.is_self_intersecting] uses.
        A pair whose triangles share an exact vertex position is one case of "touch" and is
        excluded unconditionally, even where the two triangles otherwise cross through each other's
        interior away from that vertex -- this is what keeps two meshes glued along a common seam
        (or one watertight mesh split into pieces) from reporting every shared-edge face pair as a
        collision.

    Raises
    ------
    ValueError
        If ``max_triangle_collisions`` is less than 1.
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b``, ``faces_b``, ``mesh_a`` and ``mesh_b`` are
        not all on one device.

    See Also
    --------
    [`collision_masks`][ordito.intersection.collision_masks]
        The same answer as one boolean mask per mesh, which is the form a repair pass wants.
    [`mesh_with_mesh`][ordito.intersection.mesh_with_mesh]
        The intersection curve, when the contact geometry is wanted and not just its existence.
    [`ordito.proximity.closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh]
        What to reach for when the meshes do *not* touch and the clearance is the question.

    Examples
    --------
    Two overlapping unit cubes cross; a third, apart from the first, does not:

    ```python
    import ordito as od

    vertices_a, faces_a = od.creation.box()
    vertices_b, faces_b = od.creation.box(bounds=[[0.0, -0.2, -0.3], [1.0, 0.8, 0.7]])
    pairs = od.intersection.mesh_collision_pairs(vertices_a, faces_a, vertices_b, faces_b)
    print(pairs.shape)  # (15, 2)
    vertices_c, faces_c = od.creation.box(bounds=[[2.0, 0.0, 0.0], [3.0, 1.0, 1.0]])
    apart = od.intersection.mesh_collision_pairs(vertices_a, faces_a, vertices_c, faces_c)
    print(apart.shape)  # (0, 2)
    ```

    A mesh queried repeatedly keeps its BVH in an [`ordito.Trimesh`][ordito.mesh.Trimesh]:

    ```python
    import ordito as od

    mesh_a = od.Trimesh(*od.creation.box())
    vertices_b, faces_b = od.creation.box(bounds=[[0.0, -0.2, -0.3], [1.0, 0.8, 0.7]])
    pairs = od.intersection.mesh_collision_pairs(
        mesh_a.vertices, mesh_a.faces, vertices_b, faces_b, mesh_a=mesh_a.warp_mesh
    )
    print(pairs.shape)  # (15, 2)
    ```
    """
    require_same_device(
        vertices_a=vertices_a,
        faces_a=faces_a,
        vertices_b=vertices_b,
        faces_b=faces_b,
        mesh_a=mesh_a,
        mesh_b=mesh_b,
    )
    device = vertices_a.device
    crossing = _colliding_face_pairs(
        vertices_a,
        faces_a,
        vertices_b,
        faces_b,
        max_triangle_collisions,
        "mesh_collision_pairs",
        mesh_a=mesh_a,
        mesh_b=mesh_b,
    )
    if crossing is None:
        return odt.empty_2d((0, 2), wp.int32, device=device)
    return crossing


def _colliding_face_pairs(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    max_triangle_collisions: int,
    caller: str,
    *,
    mesh_a: wp.Mesh | None = None,
    mesh_b: wp.Mesh | None = None,
) -> odt.Array2dInt32 | None:
    """
    Broad phase plus narrow phase for two meshes: the crossing face pairs, or ``None`` if none.

    The pairs are ``(n_hit, 2)`` rows in the caller's ``(a, b)`` column order, in broad-phase
    candidate order. The candidates are ``(query, target)``, and which input is the query depends
    on the **face counts**; the compaction puts the columns back, so a caller never sees that order.

    Raises
    ------
    ValueError
        If ``max_triangle_collisions`` is less than 1.
    """
    candidates = _candidate_face_pairs(
        vertices_a,
        faces_a,
        vertices_b,
        faces_b,
        max_triangle_collisions,
        caller,
        mesh_a=mesh_a,
        mesh_b=mesh_b,
    )
    if candidates is None:
        return None
    pairs = _compact_crossing_candidates(candidates, segments=False)
    return None if pairs is None else odt.as_array2d(pairs, wp.int32)


def _compact_crossing_candidates(candidates: _Candidates, *, segments: bool) -> odt.ArrayNd | None:
    """
    Narrow phase plus compaction of the broad-phase slots: the crossing pairs, or their segments.

    With ``segments`` the kept rows are the ``(n_hit, 2)`` ``wp.vec3`` intersection segments, a
    pair whose segment is degenerate rejected; otherwise the ``(n_hit, 2)`` ``wp.int32`` face pairs
    in the caller's ``(a, b)`` column order. Either way in candidate order, or ``None`` when no
    pair is kept. The verdicts are written as ``0`` / ``1`` flags and scanned in place, so one
    buffer carries the verdict, its scan and -- at its tail, the one readback -- the kept count.
    """
    device = candidates.targets.device
    n_slots = candidates.targets.size
    flags = _launch.empty(n_slots, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_intersections.crossing_slot_flags,
        dim=n_slots,
        inputs=[*candidates.narrow_phase_inputs(), candidates.tail_queries, segments, flags],
        device=device,
    )
    _launch.array_scan(flags, out_array=flags, inclusive=True)
    n_hit = int(read_scalar(flags))
    if n_hit == 0:
        return None
    out_pairs = None if segments else odt.empty_2d((n_hit, 2), wp.int32, device=device)
    out_segments = odt.empty_2d((n_hit, 2), wp.vec3, device=device) if segments else None
    _launch.launch(
        kernel_intersections.compact_crossing_slots,
        dim=n_hit,
        inputs=[
            candidates.query_vertices,
            candidates.query_faces,
            candidates.target_vertices,
            candidates.target_faces,
            candidates.targets,
            candidates.counts,
            candidates.max_hits,
            candidates.tail_queries,
            flags,
            candidates.swapped,
            out_pairs,
            out_segments,
        ],
        device=device,
    )
    return out_segments if segments else out_pairs


def collision_masks(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    max_triangle_collisions: int = 16,
    mesh_a: wp.Mesh | None = None,
    mesh_b: wp.Mesh | None = None,
) -> tuple[wp.array[wp.bool], wp.array[wp.bool]]:
    """
    Which faces of each mesh are involved in a collision with the other, as one mask per mesh.

    [`mesh_collision_pairs`][ordito.intersection.mesh_collision_pairs] reduced to the two questions
    a repair or a selection actually asks -- *which of my faces are in trouble* -- and the form a
    per-mesh collision bitset takes. It exists as its own entry point because deriving it from the
    pairs means scattering a **column** of a rank-2 array, and a column is a strided view that
    Warp's Python-scope gather silently misreads.

    Parameters
    ----------
    vertices_a, faces_a, vertices_b, faces_b
        The two meshes, as in [`mesh_collision_pairs`][ordito.intersection.mesh_collision_pairs].
    max_triangle_collisions
        Broad-phase slots reserved per query triangle. Not a cap: a triangle with more candidates
        walks the rest again on the device, so the answer does not depend on it.
    mesh_a, mesh_b
        A ``wp.Mesh`` already built over ``vertices_a`` / ``faces_a`` (respectively ``vertices_b``
        / ``faces_b``), to spare the BVH build, as
        [`mesh_collision_pairs`][ordito.intersection.mesh_collision_pairs] takes them: only the
        mesh with fewer faces (``a`` on a tie) is read, and neither is checked against the arrays.
        [`Trimesh.warp_mesh`][ordito.mesh.Trimesh.warp_mesh] is what to pass.

    Returns
    -------
    tuple[wp.array[wp.bool], wp.array[wp.bool]]
        ``(n_faces_a,)`` and ``(n_faces_b,)`` masks on ``vertices_a.device``, ``True`` for a face
        that crosses some face of the other mesh.

    Raises
    ------
    ValueError
        If ``max_triangle_collisions`` is less than 1.
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b``, ``faces_b``, ``mesh_a`` and ``mesh_b`` are
        not all on one device.

    See Also
    --------
    [`mesh_collision_pairs`][ordito.intersection.mesh_collision_pairs]
        The pair list this reduces, when *which* faces meet matters.

    Examples
    --------
    Two overlapping unit cubes: six of the twelve faces of each cross the other:

    ```python
    import ordito as od

    vertices_a, faces_a = od.creation.box()
    vertices_b, faces_b = od.creation.box(bounds=[[0.0, -0.2, -0.3], [1.0, 0.8, 0.7]])
    mask_a, mask_b = od.intersection.collision_masks(vertices_a, faces_a, vertices_b, faces_b)
    print(mask_a.shape)  # (12,)
    print(int(mask_a.numpy().sum()), int(mask_b.numpy().sum()))  # 6 6
    ```
    """
    require_same_device(
        vertices_a=vertices_a,
        faces_a=faces_a,
        vertices_b=vertices_b,
        faces_b=faces_b,
        mesh_a=mesh_a,
        mesh_b=mesh_b,
    )
    device = vertices_a.device
    n_faces_a = faces_a.size // 3
    n_faces_b = faces_b.size // 3
    mask_a = _launch.zeros(n_faces_a, dtype=wp.bool, device=device)
    mask_b = _launch.zeros(n_faces_b, dtype=wp.bool, device=device)

    # The narrow phase runs inside the marking kernel, over the broad-phase slots: the masks need
    # no compacted pair list, so the verdict flags, their scan and readback and the compaction
    # ``mesh_collision_pairs`` pays for are all skipped; an overflowing face is walked again on the
    # device and tested inline rather than collected into an overflow tail, so nothing is read
    # back. The candidates
    # are ``(query, target)``, so ``swapped`` only decides which mask is which.
    candidates = _candidate_face_pairs(
        vertices_a,
        faces_a,
        vertices_b,
        faces_b,
        max_triangle_collisions,
        "mesh_collision_pairs",
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        grow=False,
    )
    if candidates is None:
        return mask_a, mask_b
    mask_query, mask_target = (mask_b, mask_a) if candidates.swapped else (mask_a, mask_b)
    _mark_crossing_faces(candidates, mask_query, mask_target)
    return mask_a, mask_b


def _mark_crossing_faces(
    candidates: _Candidates, out_mask_query: wp.array[wp.bool], out_mask_target: wp.array[wp.bool]
) -> None:
    """
    Mark both faces of every crossing pair, from a ``_candidate_face_pairs(grow=False)`` table.

    One launch tests every live slot and one more re-walks only the faces whose candidates the
    stride could not hold, so the masks are complete with no host read.
    """
    device = candidates.query_vertices.device
    marks = [False, out_mask_query, out_mask_target]
    _launch.launch(
        kernel_intersections.mark_crossing_slot_masks,
        dim=candidates.targets.size,
        inputs=[*candidates.narrow_phase_inputs(), *marks],
        device=device,
    )
    _launch.launch(
        kernel_intersections.mark_overflow_crossings,
        dim=candidates.counts.size,
        inputs=[candidates.target_mesh.id, *candidates.narrow_phase_inputs(skip_targets=True),
                *marks],
        device=device,
    )  # fmt: skip


class _Candidates(NamedTuple):
    """``_candidate_face_pairs``' broad phase: candidate slots, their overflow and the meshes."""

    target_mesh: wp.Mesh
    targets: wp.array[wp.int32]
    counts: wp.array[wp.int32]
    max_hits: int
    tail_queries: wp.array[wp.int32] | None
    query_vertices: wp.array[wp.vec3]
    query_faces: wp.array[wp.int32]
    target_vertices: wp.array[wp.vec3]
    target_faces: wp.array[wp.int32]
    swapped: bool

    def narrow_phase_inputs(self, *, skip_targets: bool = False) -> list[object]:
        """
        Return the leading launch arguments every narrow-phase kernel over the slots shares.

        ``skip_targets`` leaves out the slot table, for the overflow walk that reads only counts.
        """
        meshes = [self.query_vertices, self.query_faces, self.target_vertices, self.target_faces]
        table = [self.counts] if skip_targets else [self.targets, self.counts]
        return [*meshes, *table, self.max_hits]


def _candidate_face_pairs(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    max_triangle_collisions: int,
    caller: str,
    *,
    mesh_a: wp.Mesh | None = None,
    mesh_b: wp.Mesh | None = None,
    grow: bool = True,
) -> _Candidates | None:
    """
    Broad phase alone for two meshes: every candidate ``(query, target)`` pair, or ``None``.

    The candidates are a fixed-stride table: query face ``q`` owns slots ``[q * max_hits, (q + 1) *
    max_hits)`` of ``targets``, the first ``counts[q]`` of them live, in BVH traversal order. The
    fixed stride is what spares a count pass, its scan, the host read of the total and a second
    traversal to fill a packed list; a dead slot is skipped by every consumer. With ``grow`` the
    candidates past a face's block follow the table in ``targets``, tail slot ``j`` owned by
    ``tail_queries[j]``; without it ``tail_queries`` is ``None`` and the caller walks an overflowing
    face again itself. ``swapped`` says whether the query is mesh **b**: the smaller mesh supplies
    the BVH, so which input is the query depends on the face counts. Every caller runs its narrow
    phase in the kernel that consumes the slots. ``None`` only when a mesh has no faces. A prebuilt
    ``mesh_a`` / ``mesh_b`` is used when it is the side that supplies the BVH, and built otherwise.

    Raises
    ------
    ValueError
        If ``max_triangle_collisions`` is less than 1.
    """
    # Before the empty-mesh early return, not after: the three public callers all document this
    # unconditionally ("Raises: ValueError if max_triangle_collisions is less than 1"), and an empty
    # mesh plus an invalid cap must not silently return `None` instead.
    if max_triangle_collisions < 1:
        raise ValueError("max_triangle_collisions must be >= 1")
    n_faces_a = faces_a.size // 3
    n_faces_b = faces_b.size // 3
    if n_faces_a == 0 or n_faces_b == 0:
        return None

    # The smaller mesh supplies the BVH, so the larger one's faces are the queries.
    swapped = n_faces_a <= n_faces_b
    if swapped:
        target_vertices, target_faces, target_mesh = vertices_a, faces_a, mesh_a
        query_vertices, query_faces = vertices_b, faces_b
    else:
        target_vertices, target_faces, target_mesh = vertices_b, faces_b, mesh_b
        query_vertices, query_faces = vertices_a, faces_a

    if target_mesh is None:
        require_nonempty_mesh(target_faces, caller)
        target_mesh = wp.Mesh(points=target_vertices, indices=target_faces)

    targets, counts, stride, tail_queries = _face_box_candidates(
        query_vertices, query_faces, target_mesh, max_triangle_collisions, grow
    )
    return _Candidates(
        target_mesh,
        targets,
        counts,
        stride,
        tail_queries,
        query_vertices,
        query_faces,
        target_vertices,
        target_faces,
        swapped,
    )


def face_box_candidates(
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    target_mesh: wp.Mesh,
    *,
    capacity: int = 16,
    grow: bool = True,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], int]:
    """
    Every target face whose bounding box overlaps each query face's, as a fixed-stride table.

    The broad phase of every triangle-pair test in the package (two-mesh intersection and
    collision, self-intersection): one walk of ``target_mesh``'s BVH per query face over that
    face's bounding box. Query face ``q`` owns slots ``[q * stride, (q + 1) * stride)`` of
    ``targets``, the first ``min(counts[q], stride)`` of them live, in BVH traversal order. A fixed
    stride spares the scan and second walk a packed list needs.

    The answer is complete whatever ``capacity`` is: a face with more than ``stride`` candidates
    keeps the rest in an overflow tail after the table, packed in query-face order, so the
    table's size does not follow the most crowded face. Costs one host read of the overflow's size
    -- unless ``grow`` is off, for a caller that handles the overflow on the device itself.

    Parameters
    ----------
    query_vertices, query_faces
        ``(n_query_vertices,)`` positions and ``(3 * n_query,)`` index buffer of the query faces.
    target_mesh
        BVH over the target faces.
    capacity
        Slots per query face in the table. Lowered when ``n_query * capacity`` would not fit in
        ``int32`` slot indices; the overflow tail keeps the answer complete either way.
    grow
        Collect the candidates past a face's block into the overflow tail (default). When
        ``False`` nothing is read back and there is no tail: a face whose count exceeds ``stride``
        holds its first ``stride`` candidates, in traversal order, and the rest are the caller's to
        walk again.

    Returns
    -------
    targets
        ``(n_query * stride + n_overflow,)`` candidate target face indices: the table, then (with
        ``grow``) the overflow tail, where face ``q``'s ``max(counts[q] - stride, 0)`` further
        candidates follow those of every lower-numbered face, in traversal order. ``n_overflow``
        is the sum of those excesses, 0 without ``grow``. Table slots past a face's count are
        unwritten.
    counts
        ``(n_query,)`` candidates per query face, table and tail together; may exceed ``stride``.
    stride
        Slots per query face in the table.

    Raises
    ------
    ValueError
        If ``capacity`` is less than 1, or if the table and its overflow together exceed
        ``2**31 - 1`` slots.
    RuntimeError
        If ``query_vertices``, ``query_faces`` and ``target_mesh`` are not all on one device.

    See Also
    --------
    [`mesh_collision_pairs`][ordito.intersection.mesh_collision_pairs]
    [`ordito.validation.face_self_intersecting_mask`][ordito.validation.face_self_intersecting_mask]

    Examples
    --------
    The faces of one cube against a BVH over an overlapping one. No face has more than four
    candidates, so the default stride of 16 holds all 24; at ``capacity=2`` the six faces with four
    each keep two in the table and two in the overflow tail, 24 + 12 slots:

    ```python
    import warp as wp

    import ordito as od

    vertices_a, faces_a = od.creation.box()
    vertices_b, faces_b = od.creation.box(bounds=[[0.0, -0.2, -0.3], [1.0, 0.8, 0.7]])
    target = wp.Mesh(points=vertices_b, indices=faces_b)
    targets, counts, stride = od.intersection.face_box_candidates(vertices_a, faces_a, target)
    print(targets.shape, counts.shape, stride)  # (192,) (12,) 16
    print(int(counts.numpy().sum()))  # 24
    narrow, _, stride = od.intersection.face_box_candidates(vertices_a, faces_a, target, capacity=2)
    print(narrow.shape, stride)  # (36,) 2
    ```
    """
    require_same_device(
        query_vertices=query_vertices, query_faces=query_faces, target_mesh=target_mesh
    )
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")
    targets, counts, stride, _ = _face_box_candidates(
        query_vertices, query_faces, target_mesh, capacity, grow
    )
    return targets, counts, stride


def _face_box_candidates(
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    target_mesh: wp.Mesh,
    capacity: int,
    grow: bool,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], int, wp.array[wp.int32] | None]:
    """
    ``face_box_candidates``, plus the owning query face of each overflow-tail slot.

    The fourth return is ``(n_overflow,)``, or ``None`` when there is no tail (nothing overflowed,
    or ``grow`` is off). The overflow is collected by walking only the overflowing faces again and
    skipping the candidates their block holds, so the table stays ``n_query * capacity`` and the
    tail holds exactly the excess.

    Raises
    ------
    ValueError
        If the table and its overflow together exceed ``2**31 - 1`` slots.
    """
    device = query_vertices.device
    n_query = query_faces.size // 3
    stride = min(capacity, max(1, _MAX_CANDIDATE_SLOTS // max(n_query, 1)))
    n_table = n_query * stride
    # The query boxes are read by the walks alone, so both corners share one allocation.
    lower, upper = _launch.carve(wp.vec3, [n_query, n_query], device)
    _launch.launch(
        kernel_triangles.face_aabb_bounds,
        dim=n_query,
        inputs=[query_vertices, query_faces, lower, upper],
        device=device,
    )
    counts = _launch.empty(n_query, dtype=wp.int32, device=device)
    extras = _launch.empty(n_query, dtype=wp.int32, device=device) if grow else None
    overflow = _launch.zeros(1, dtype=wp.int64, device=device)
    targets = _launch.empty(n_table, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_intersections.collect_face_box_candidates,
        dim=n_query,
        inputs=[target_mesh.id, lower, upper, stride],
        outputs=[targets, counts, extras, overflow],
        device=device,
    )
    if extras is None:
        return targets, counts, stride, None
    # The one read: the candidates the table could not hold, which sizes the tail.
    n_tail = int(read_scalar(overflow))
    if n_tail == 0:
        return targets, counts, stride, None
    if n_table + n_tail > _MAX_CANDIDATE_SLOTS:
        raise ValueError(
            f"{n_table + n_tail} broad-phase candidate slots exceed the int32 limit "
            f"{_MAX_CANDIDATE_SLOTS}"
        )
    _launch.array_scan(extras, out_array=extras, inclusive=False)
    with_tail = _launch.empty(n_table + n_tail, dtype=wp.int32, device=device)
    _launch.copy(with_tail, targets, count=n_table)
    tail_queries = _launch.empty(n_tail, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_intersections.collect_overflow_candidates,
        dim=n_query,
        inputs=[target_mesh.id, lower, upper, counts, stride, extras],
        outputs=[with_tail, tail_queries],
        device=device,
    )
    return with_tail, counts, stride, tail_queries


def slice_mesh_with_plane(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Slice a mesh with a plane, returning the portion on the positive normal side.

    Matches [`trimesh.intersections.slice_faces_plane`][] for indexed triangle meshes
    (without UV handling). To slice a face subset, extract a submesh first (e.g.
    [`submesh_from_face_indices`][ordito.selection.submesh_from_face_indices]).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    plane_normal
        Normal vector of the plane.
    plane_origin
        Point on the plane.

    Returns
    -------
    new_vertices
        ``(n_new_vertices,)`` vertices of the sliced mesh.
    new_faces
        ``(3 * m,)`` flat triangle index buffer for the sliced mesh, ``m`` its face count.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    The plane's signed distance is one per-vertex scalar field, so this is
    [`clip_mesh_with_field`][ordito.intersection.clip_mesh_with_field] over
    ``dot(v - plane_origin, plane_normal)`` — the only thing the plane adds is the tie-break for a
    face lying *in* the plane, whose side is decided from its own normal rather than from its
    vertices.

    Every face falls into exactly one of three kept classes — wholly inside, cut into a quad, cut
    into a triangle — and the classes, the crossing points and the vertices the kept faces
    reference are all ranked by a *single* scan, so the call makes one host readback (the counts
    that size the output) and writes the compacted mesh in one pass, with nothing concatenated or
    compacted afterwards.

    See Also
    --------
    [`clip_mesh_with_field`][ordito.intersection.clip_mesh_with_field]
    [`mesh_with_plane`][ordito.intersection.mesh_with_plane]
    [`trimesh.intersections.slice_faces_plane`][]

    Examples
    --------
    The upper half of a unit cube: its top face and half of each side, area ``1 + 4 * 0.5``, left
    open along the cut:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    kept_vertices, kept_faces = od.intersection.slice_mesh_with_plane(
        vertices, faces, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0)
    )
    _, areas = od.triangles.face_normals_and_areas(kept_vertices, kept_faces)
    print(f"{areas.numpy().sum():.4f}")  # 3.0000
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_vertices = vertices.size
    if n_vertices == 0:
        return vertices, faces
    if faces.size == 0:
        return _launch.empty(0, dtype=wp.vec3, device=device), _launch.empty(
            0, dtype=wp.int32, device=device
        )

    # No field: the kernels form each vertex's plane dot where they read it.
    return _clip_with_vertex_field(
        vertices,
        faces,
        _launch.empty(0, dtype=wp.float32, device=device),
        plane=(plane_normal, plane_origin),
    )


def split_mesh_with_plane(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    *,
    tolerance: float = TOLERANCE_MERGE,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Insert a plane's cross-section into the mesh as real edges, keeping both sides.

    The third thing a plane can do to a mesh, next to
    [`mesh_with_plane`][ordito.intersection.mesh_with_plane], which returns the section curve and
    leaves the mesh alone, and
    [`slice_mesh_with_plane`][ordito.intersection.slice_mesh_with_plane], which keeps one side and
    discards the other: this keeps **everything** and refines the triangles the plane crosses so the
    section becomes a set of mesh edges, then labels every output face by the side it fell on. It is
    what to reach for to select a region by a plane and then remesh, decimate or smooth only that
    region, since the complementary side is still there to be welded back to.

    The result is **crack-free**: the crossing point of an edge is computed once per *edge*, so the
    two triangles sharing it reference one vertex index and a watertight input stays watertight.
    That is the difference from
    [`slice_mesh_with_plane`][ordito.intersection.slice_mesh_with_plane], whose per-face cut leaves
    two coincident copies along the section.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    plane_normal
        Normal vector of the plane. Need not be unit length, but ``tolerance`` is a band on
        ``dot(v - plane_origin, plane_normal)``, so a non-unit normal scales it.
    plane_origin
        Point on the plane.
    tolerance
        Half-width of the band around the plane within which a vertex counts as lying *on* it. Such
        a vertex is used as the crossing itself rather than having a near-duplicate inserted beside
        it, which is what keeps a plane through an existing vertex from producing slivers. Defaults
        to [`TOLERANCE_MERGE`][ordito.constants.TOLERANCE_MERGE]; scale it with the model when the
        coordinates are large.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_vertices + n_crossed,)`` original vertices followed by one crossing point per crossed
        edge.
    new_faces : wp.array[wp.int32]
        ``(3 * m,)`` flat triangle index buffer for the refined mesh, both sides included, ``m``
        its face count.
    above : wp.array[wp.bool]
        ``(m,)`` per-face mask: ``True`` where the face is on the ``plane_normal`` side. This is the
        side [`slice_mesh_with_plane`][ordito.intersection.slice_mesh_with_plane] returns,
        including its tie-break for a face lying *in* the plane (kept when its own normal opposes
        ``plane_normal``), so
        ``submesh_from_face_mask(*split_mesh_with_plane(...))`` and ``slice_mesh_with_plane`` agree
        — with one exception: a face lying in the plane whose own area is degenerate (below
        [`TOLERANCE_MERGE`][ordito.constants.TOLERANCE_MERGE]-scale zero) has no normal to break
        the tie with, so [`slice_mesh_with_plane`][ordito.intersection.slice_mesh_with_plane] drops
        it from the output entirely, on either side, while this function — which keeps every face
        somewhere — assigns it to ``True`` by convention.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    A plane crosses at most **two** of a triangle's three edges — two of three vertices always share
    a side, so their edge is not crossed — which means every crossed face is refined by the 1-split
    or 2-split case of [`subdivide_to_size`][ordito.remesh.subdivide_to_size]'s crack-free
    templates, reused unchanged. Those templates are indexed by *which* corners carry a new vertex
    and never assume the new vertex is a midpoint, so the plane crossing drops straight in. The
    2-split quad is cut along its shorter diagonal, as there; both of its diagonals lie wholly on
    one side of the plane, so the choice cannot make a face straddle.

    Faces are **not** reordered: an output face's position is the compaction order of the split
    templates, not grouped by side. Pass ``above`` to
    [`submesh_from_face_mask`][ordito.selection.submesh_from_face_mask] (or its negation) to
    materialize either half.

    Equivalent to VTK's ``vtkClipPolyData`` with ``GenerateClippedOutput``, which pyvista exposes as
    ``PolyData.clip(..., return_clipped=True)``.

    See Also
    --------
    [`slice_mesh_with_plane`][ordito.intersection.slice_mesh_with_plane]
    [`mesh_with_plane`][ordito.intersection.mesh_with_plane]
    [`ordito.remesh.split_edges`][ordito.remesh.split_edges]
    [`ordito.remesh.subdivide_to_size`][ordito.remesh.subdivide_to_size]
    [`ordito.selection.submesh_from_face_mask`][ordito.selection.submesh_from_face_mask]

    Examples
    --------
    A horizontal plane crosses a unit cube's four vertical edges and the diagonals of its four
    sides, adding eight vertices; the refined cube stays watertight, with both halves in it:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    split_vertices, split_faces, above = od.intersection.split_mesh_with_plane(
        vertices, faces, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.1)
    )
    print(split_vertices.shape)  # (16,)
    print(above.shape, int(above.numpy().sum()))  # (28,) 14
    print(od.validation.is_watertight(split_vertices, split_faces))  # True
    ```
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    if n_vertices == 0 or n_faces == 0:
        return (
            _launch.clone(vertices),
            _launch.clone(faces),
            _launch.empty(n_faces, dtype=wp.bool, device=device),
        )

    # Each kernel forms the plane dots of the corners it reads, so no per-vertex table is written.
    # Each face's child count, scanned, is its first output row; ``row - f`` is then its first slot
    # among the crossed halfedges, which are written densely and sorted alone. That numbers the
    # crossed edges in the order ``edges.edges_unique`` lists them -- what ``remesh.split_edges``
    # would number them by -- without grouping the whole mesh's edges.
    tol = wp.float32(tolerance)
    child_counts = _launch.empty(n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_intersections.plane_split_child_counts,
        dim=n_faces,
        inputs=[vertices, faces, plane_normal, plane_origin, tol, child_counts],
        device=device,
    )
    face_offsets, n_out = od.array.counts_to_offsets(child_counts)
    n_halfedges = n_out - n_faces
    if n_halfedges == 0:
        new_vertices, new_faces = _launch.clone(vertices), _launch.clone(faces)
    else:
        key_base = wp.int64(n_vertices)
        keys = _launch.empty(2 * n_halfedges, dtype=wp.int64, device=device)
        # The sort's payload, the crossed-edge ranks and each halfedge's new vertex: one allocation.
        halfedges, rank, cut_vertices = _launch.carve(
            wp.int32, [2 * n_halfedges, n_halfedges, n_halfedges], device
        )
        _launch.launch(
            kernel_intersections.plane_crossed_halfedge_keys,
            dim=n_faces,
            inputs=[
                vertices,
                faces,
                plane_normal,
                plane_origin,
                tol,
                face_offsets,
                key_base,
                keys,
                halfedges,
            ],
            device=device,
        )
        _launch.radix_sort_pairs(
            keys, halfedges, n_halfedges, end_bit=max(1, (n_vertices * n_vertices).bit_length())
        )
        _launch.launch(
            kernel_intersections.split_crossed_edge_flags,
            dim=n_halfedges,
            inputs=[keys, key_base, rank],
            device=device,
        )
        n_new = _scan_cut_flags(rank)
        new_vertices = _launch.empty(n_vertices + n_new, dtype=wp.vec3, device=device)
        # The same launch copies the input positions to the front of ``new_vertices``.
        _launch.launch(
            kernel_intersections.split_crossed_edge_vertices,
            dim=max(n_halfedges, n_vertices),
            inputs=[
                vertices,
                None,
                plane_normal,
                plane_origin,
                keys,
                halfedges,
                rank,
                key_base,
                wp.int32(n_vertices),
                cut_vertices,
                new_vertices,
            ],
            device=device,
        )
        new_faces = _launch.empty(3 * n_out, dtype=wp.int32, device=device)
        above = _launch.empty(n_out, dtype=wp.bool, device=device)
        # The same launch labels each child as it writes it.
        _launch.launch(
            kernel_intersections.emit_plane_split_faces,
            dim=n_faces,
            inputs=[
                faces,
                tol,
                cut_vertices,
                face_offsets,
                new_vertices,
                plane_normal,
                plane_origin,
                new_faces,
                new_faces.reshape((n_out, 3)),
                None,
                above,
            ],
            device=device,
        )
        return new_vertices, new_faces, above

    above = _launch.empty(n_faces, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_intersections.label_faces_by_plane_side,
        dim=n_faces,
        inputs=[new_vertices, new_faces, plane_normal, plane_origin, tol, above],
        device=device,
    )
    return new_vertices, new_faces, above


def clip_mesh_with_field(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float32] | wp.array[wp.float64],
    isovalue: float = 0.0,
    *,
    cap: bool = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Keep the ``values >= isovalue`` region of the mesh, cutting faces the level set crosses.

    The region counterpart of
    [`marching_triangles`][ordito.intersection.marching_triangles], which returns the level set
    itself: this returns the surface on one side of it, with every crossed triangle re-triangulated
    against the crossing points. The caller supplies the field, so one implementation covers
    clipping by a plane, by another surface's signed distance
    ([`signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh]), by a box, by geodesic
    distance ([`heat_geodesic`][ordito.heat.heat_geodesic]) or by any per-vertex quantity
    the caller can threshold.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    values
        ``(n_vertices,)`` scalar field, ``wp.float32`` or ``wp.float64``. A ``float64`` field is
        shifted in ``float64`` and then compared in ``float32``, which is the vertex buffer's
        precision.
    isovalue
        Level to clip at. A vertex whose value equals it counts as kept, matching
        [`marching_triangles`][ordito.intersection.marching_triangles].
    cap
        When ``True``, seal the section: every closed loop of the level set is triangulated, so a
        closed input gives a closed output. A planar loop (any section of a plane field) is
        ear-clipped in its plane as a polygon; any other loop gets
        [`fill_loops_min_weight`][ordito.holes.fill_loops_min_weight]'s minimum-weight fill. Only
        the section is sealed -- a boundary the input already had stays open, and so does a section
        that runs into one -- and the input vertices are returned exactly as given (coincident ones
        are not merged); only the crossing points the cut appends are shared. Use
        [`fill_min_weight`][ordito.holes.fill_min_weight] on the result to close every hole
        instead.

    Returns
    -------
    new_vertices
        ``(n_new_vertices,)`` vertices of the clipped region, with the crossing points appended.
    new_faces
        ``(3 * m,)`` flat triangle index buffer for the clipped region, ``m`` its face count.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``values`` are not all on one device.

    Notes
    -----
    A face whose three values all equal ``isovalue`` is dropped: it lies *in* the level set, so
    neither side claims it and the field gives no tie-break.
    [`slice_mesh_with_plane`][ordito.intersection.slice_mesh_with_plane] is the one caller that has
    one, and keeps such a face when its normal opposes the plane's.

    The uncapped result is **cracked along the section**, like
    [`trimesh.intersections.slice_faces_plane`][]'s: each cut face writes its own copy of the
    crossing points, so a rim edge carries two coincident vertices. Those copies are bitwise equal
    by construction (a crossing is interpolated from the endpoint that comes first by position), so
    ``cap=True`` merges them by exact equality rather than by a tolerance, and merges nothing else.

    Equivalent to VTK's ``clip_scalar`` (uncapped) and ``clip_closed_surface`` (capped, over a
    plane's signed distance), which pyvista exposes on ``PolyData``; VTK's default keeps
    ``scalar >= value`` as this does, and its capped clip likewise closes only the cut and leaves
    the input's points unmerged.

    See Also
    --------
    [`marching_triangles`][ordito.intersection.marching_triangles]
    [`slice_mesh_with_plane`][ordito.intersection.slice_mesh_with_plane]
    [`fill_min_weight`][ordito.holes.fill_min_weight]

    Examples
    --------
    Keep the part of a unit sphere above ``z = 0.1``; with ``cap=True`` the cut is sealed and the
    result is closed:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3)
    heights = wp.array(vertices.numpy()[:, 2], dtype=wp.float32)
    kept_vertices, kept_faces = od.intersection.clip_mesh_with_field(vertices, faces, heights, 0.1)
    print(od.validation.is_watertight(kept_vertices, kept_faces))  # False
    kept_vertices, kept_faces = od.intersection.clip_mesh_with_field(
        vertices, faces, heights, 0.1, cap=True
    )
    print(od.validation.is_watertight(kept_vertices, kept_faces))  # True
    ```

    Keep the geodesic disk of radius 1 around vertex 0, sealed into a solid. The field is the
    ``float64`` one [`heat_geodesic`][ordito.heat.heat_geodesic] returns, and the region wanted is
    the *near* side, so the field enters negated:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=3)
    distance = od.heat.heat_geodesic(vertices, faces, wp.array([0], dtype=wp.int32))
    negated = wp.empty_like(distance)
    wp.map(wp.neg, distance, out=negated)
    disk_vertices, disk_faces = od.intersection.clip_mesh_with_field(
        vertices, faces, negated, -1.0, cap=True
    )
    print(od.validation.is_edge_manifold(disk_faces, allow_boundary_edges=False))  # True
    ```
    """
    require_same_device(vertices=vertices, faces=faces, values=values)
    device = vertices.device
    n_vertices = vertices.size
    if n_vertices == 0:
        return vertices, faces
    if faces.size == 0:
        return _launch.empty(0, dtype=wp.vec3, device=device), _launch.empty(
            0, dtype=wp.int32, device=device
        )

    shifted = _shifted_field(values, isovalue)
    if not cap:
        return _clip_with_vertex_field(vertices, faces, shifted)
    # The cut writes its crossing points per face, so a rim edge shared by two cut faces arrives as
    # two bitwise-equal vertices; merging those (and only those -- the input vertices are kept as
    # given) turns the section into real boundary loops.
    new_vertices, new_faces, source = _clip_with_sources(
        vertices, faces, shifted, merge_crossings=True
    )
    assert source is not None
    if new_faces.size == 0:
        return new_vertices, new_faces
    # The cap closes the section and nothing else, as VTK caps the cut's own polylines: the boundary
    # edges with both ends on the level set (a crossing point, or an input vertex the classifier
    # reads as on it, within its tolerance), chained into cycles. Any other boundary edge was the
    # input's and stays open, and a section that runs into one is a chain, not a cycle, and stays
    # open too. An edge with both ends on the section lies only in faces with two corners on it, so
    # its boundary status is decided among those faces: the rim is taken over them alone, not the
    # whole mesh, and only its section edges are kept.
    n_faces = new_faces.size // 3
    scanned = _launch.empty(n_faces, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_intersections.section_face_flags,
        dim=n_faces,
        inputs=[new_faces, source, shifted, scanned],
        device=device,
    )
    _launch.array_scan(scanned, scanned, inclusive=True)
    # Readback: the touching-face count sizes their buffer.
    n_touching = int(read_scalar(scanned, n_faces - 1))
    if n_touching == 0:
        return new_vertices, new_faces
    touching = _launch.empty(3 * n_touching, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_intersections.emit_section_faces,
        dim=n_faces,
        inputs=[new_faces, scanned, touching],
        device=device,
    )
    rim = od.boundary.oriented_boundary_edges(new_vertices, touching)
    # Host bookkeeping over the touching faces' rim alone: its edges and their section flags cross
    # in one small copy, and the section edges go back in one.
    n_rim = int(rim.shape[0])
    if n_rim == 0:
        return new_vertices, new_faces
    rim_rows = odt.empty_2d((n_rim, 3), wp.int32, device=device)
    _launch.launch(
        kernel_intersections.section_rim_edges,
        dim=n_rim,
        inputs=[rim, source, shifted, rim_rows],
        device=device,
    )
    # Readback: the section edges are chained on the host.
    rim_rows_np = rim_rows.numpy()
    section_np = rim_rows_np[rim_rows_np[:, 2] != 0, :2]
    if section_np.shape[0] == 0:
        return new_vertices, new_faces
    # The section's vertices are numbered compactly for the chaining, so its tables are sized by
    # the section rather than the mesh; one upload carries the edges and the numbering back.
    section_ids, compact_np = np.unique(section_np, return_inverse=True)
    n_ids = section_ids.size
    upload = _launch.array(
        np.concatenate((section_ids, compact_np.reshape(-1))).astype(np.int32),
        dtype=wp.int32,
        device=device,
    )
    section_edges = odt.as_array2d(odt.as_dense(upload[n_ids:]).reshape((-1, 2)), wp.int32)
    compact_cycles, offsets = od.graph.successor_cycles(section_edges, n_ids, validate=False)
    if compact_cycles.size == 0:
        return new_vertices, new_faces
    cycles = cast(
        "wp.array[wp.int32]", od.array.gather(odt.as_dense(upload[:n_ids]), compact_cycles)
    )
    return new_vertices, _cap_section_loops(new_vertices, new_faces, cycles, offsets)


def _cap_section_loops(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cycles: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
) -> wp.array[wp.int32]:
    """
    Append a cap for every section loop to ``faces``.

    A planar loop is ear-clipped, any other gets the min-weight fill. A loop is planar when every
    point lies within ``_CAP_PLANAR_TOLERANCE`` of the loop's extent from its best-fit plane, which
    a plane field's section always does. Such a loop is a polygon, and is triangulated as one, as
    VTK caps its clipped contours; a loop whose ear clip does not return the full ``B - 2``
    triangles (a projection that is not a simple polygon) falls back to the min-weight fill with
    the non-planar loops.
    """
    device = vertices.device
    n_loops = offsets.size - 1
    n_points = cycles.size
    # The loop offsets, the loops' face offsets and their points share one byte buffer, so the host
    # reads everything it decides on in one copy: offsets first, then face offsets (both 4-byte
    # aligned), then the points' ``float32`` words.
    offsets_bytes = 4 * (n_loops + 1)
    packed = _launch.empty(2 * offsets_bytes + 12 * n_points, dtype=wp.uint8, device=device)
    gathered = _byte_view(packed, 2 * offsets_bytes, wp.vec3, n_points)
    _launch.copy(gathered, vertices[cycles])
    _launch.copy(_byte_view(packed, 0, wp.int32, n_loops + 1), offsets)
    # Every loop is ear-clipped, all of them together, while the host tests which are planar; a
    # non-planar loop's triangulation is then discarded.
    triangulated, face_offsets = od.polyline.polyline_triangulate_from_offsets(gathered, offsets)
    _launch.copy(_byte_view(packed, offsets_bytes, wp.int32, n_loops + 1), face_offsets)
    # Readback: the planarity test and the loop sizes, over the section alone.
    packed_np = packed.numpy()
    offsets_np = packed_np[:offsets_bytes].view(np.int32)
    face_offsets_np = packed_np[offsets_bytes : 2 * offsets_bytes].view(np.int32)
    points_np = packed_np[2 * offsets_bytes :].view(np.float32).reshape(-1, 3).astype(np.float64)

    cap_rows_np = np.full(n_loops, -1, dtype=np.int32)
    n_cap_rows = 0
    fallback: list[wp.array[wp.int32]] = []
    for r, (start, stop) in enumerate(
        zip(offsets_np[:-1].tolist(), offsets_np[1:].tolist(), strict=True)
    ):
        size = stop - start
        n_rows = int(face_offsets_np[r + 1] - face_offsets_np[r])
        if size >= 3 and n_rows == size - 2 and _is_planar_loop(points_np[start:stop]):
            cap_rows_np[r] = n_cap_rows
            n_cap_rows += n_rows
            continue
        fallback.append(odt.as_dense(cycles[start:stop]))
    if fallback:
        # The clip wrote every face index below its own vertex count, so the fill needs no reduction
        # to bound them.
        faces = od.holes.fill_loops_min_weight(
            vertices, faces, fallback, "plane_normalized", True, n_vertices=vertices.size
        )
    if n_cap_rows == 0:
        return faces
    # With every loop capped by its ear clip, its rows land where they sit in the triangulation,
    # so the face offsets already place them; otherwise the kept loops' rows go up packed.
    cap_rows = (
        face_offsets if not fallback else _launch.array(cap_rows_np, dtype=wp.int32, device=device)
    )
    out = _launch.empty(faces.size + 3 * n_cap_rows, dtype=wp.int32, device=device)
    _launch.copy(out, faces, count=faces.size)
    _launch.launch_tiled(
        kernel_intersections.emit_cap_faces,
        dim=(n_loops,),
        inputs=[triangulated, offsets, face_offsets, cycles, cap_rows, wp.int32(faces.size)],
        outputs=[out],
        block_dim=TILE_1D,
        device=device,
    )
    return out


def _is_planar_loop(points: np.ndarray) -> bool:
    """Whether every point lies within ``_CAP_PLANAR_TOLERANCE`` of the extent from the best fit."""
    centred = points - points.mean(axis=0)
    extent = float(np.linalg.norm(np.ptp(points, axis=0)))
    if extent == 0.0:
        return False
    # Ascending eigenvalues: the first vector is the plane normal.
    _, vectors = np.linalg.eigh(centred.T @ centred)
    return float(np.abs(centred @ vectors[:, 0]).max()) <= _CAP_PLANAR_TOLERANCE * extent


def split_faces_along_field(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float32] | wp.array[wp.float64],
    isovalue: float = 0.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Cut every face the level set crosses, keeping **both** sides as one connected mesh.

    The both-sides form of [`clip_mesh_with_field`][ordito.intersection.clip_mesh_with_field]: that
    one keeps the ``values >= isovalue`` region and discards the rest, this one keeps everything and
    makes the level set a set of real mesh edges. Nothing moves and nothing is welded -- every input
    vertex survives at its own position, and the crossing points are appended and **shared** by the
    faces on both sides, so the result is as watertight as the input was.

    That sharing is the whole point. It is what makes the returned mask a genuine partition of the
    surface along the curve rather than two overlapping selections, so
    [`submesh_from_face_mask`][ordito.selection.submesh_from_face_mask] on the mask and on its
    negation gives two pieces that meet exactly along the level set.

    A face with one corner exactly on the level set splits into **two** triangles, not three: the
    cut runs from that corner to the single crossing on the opposite edge. Emitting three would put
    a zero-area sliver in the output.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    values
        ``(n_vertices,)`` scalar field, ``wp.float32`` or ``wp.float64``. A ``float64`` field is
        shifted in ``float64`` and then compared in ``float32``, which is the vertex buffer's
        precision -- the same rule as
        [`marching_triangles`][ordito.intersection.marching_triangles].
    isovalue
        Level to cut at. A vertex whose value equals it counts as positive, so a face lying wholly
        in the level set is kept whole on the positive side rather than cut along itself.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_vertices + n_crossed,)`` input vertices, unchanged and in order, with the crossing
        points appended.
    new_faces : wp.array[wp.int32]
        ``(3 * m,)`` flat triangle index buffer, ``m`` the output face count. Uncut faces keep
        their winding and their vertex indices; cut faces are replaced by the two or three
        triangles they split into.
    positive : wp.array[wp.bool]
        ``(m,)`` mask, ``True`` for the faces on the ``values >= isovalue`` side.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``values`` are not all on one device.

    See Also
    --------
    [`clip_mesh_with_field`][ordito.intersection.clip_mesh_with_field]
        Keeps one side and discards the other, which is cheaper when that is all you need.
    [`marching_triangles`][ordito.intersection.marching_triangles]
        Returns the level set itself, as polylines, instead of cutting along it.
    [`split_mesh_with_plane`][ordito.intersection.split_mesh_with_plane]
        The plane special case of the clip.
    [`faces_left_of_contour`][ordito.selection.faces_left_of_contour]
        The mask this returns for free, for a contour that was *given* rather than just created.

    Examples
    --------
    Cut a unit sphere along ``z = 0.1``: the crossings are appended to the vertices, and the
    result is as watertight as the input:

    ```python
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=2)
    heights = wp.array(vertices.numpy()[:, 2], dtype=wp.float32)
    new_vertices, new_faces, positive = od.intersection.split_faces_along_field(
        vertices, faces, heights, 0.1
    )
    print(vertices.shape, new_vertices.shape)  # (162,) (208,)
    print(positive.shape, int(positive.numpy().sum()))  # (412,) 190
    print(od.validation.is_watertight(new_vertices, new_faces))  # True
    ```
    """
    require_same_device(vertices=vertices, faces=faces, values=values)
    device = vertices.device
    if faces.size == 0 or vertices.size == 0:
        return (
            _launch.clone(vertices),
            _launch.empty(0, dtype=wp.int32, device=device),
            _launch.empty(0, dtype=wp.bool, device=device),
        )
    return _split_with_vertex_field(vertices, faces, _shifted_field(values, isovalue))


def _shifted_field(
    values: wp.array[wp.float32] | wp.array[wp.float64], isovalue: float
) -> wp.array[wp.float32]:
    """
    Re-zero the field at ``isovalue``, in the vertex buffer's precision.

    Shifted in the field's own dtype, then narrowed once, in the same element-wise op: every
    classifier and crossing kernel downstream works in ``float32``, which is what the vertex buffer
    carries, so narrowing earlier would lose the subtraction's precision and narrowing later would
    mean doing it per face.
    """
    if values.dtype is wp.float32 and isovalue == 0.0:
        # ``x - 0.0`` is ``x`` bit for bit, signed zeros and NaN included, and every consumer only
        # reads the field, so the caller's array serves as it is.
        return cast("wp.array[wp.float32]", values)
    device = values.device
    shifted = _launch.empty(values.size, dtype=wp.float32, device=device)
    if values.dtype is wp.float32:
        _launch.map(wp.sub, values, values.dtype(isovalue), out=shifted)
    else:
        # Subtracted in ``float64`` and narrowed in the same op, so the wide intermediate is never
        # stored and never cast in a pass of its own.
        _launch.map(
            kernel_intersections.shift_to_float32, values, wp.float64(isovalue), out=shifted
        )
    return shifted


def _split_with_vertex_field(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], vertex_dots: wp.array[wp.float32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Cut along ``vertex_dots == 0``, keeping both sides; the engine behind the public split.

    Each face's class is a one-hot ``wp.vec4i`` scanned in place, which ranks every face within
    its class and ends in the four class counts, read together; every later launch runs over the
    faces and places each by its class and rank, so no index partition is built.
    """
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3

    ranks = _launch.empty(n_faces, dtype=wp.vec4i, device=device)
    _launch.launch(
        kernel_intersections.classify_faces_for_split,
        dim=n_faces,
        inputs=[faces, vertex_dots, ranks],
        device=device,
    )
    _launch.array_scan(ranks, out_array=ranks, inclusive=True)
    # The class counts size every buffer downstream: one readback.
    n_positive, n_negative, n_edges, n_corner = (int(c) for c in read_scalar(ranks))
    if n_edges + n_corner == 0:
        # Nothing crossed, so the mesh is untouched and only the side labels are new.
        positive = _launch.empty(n_faces, dtype=wp.bool, device=device)
        _launch.launch(
            kernel_intersections.emit_split_faces,
            dim=n_faces,
            inputs=[
                faces,
                vertex_dots,
                ranks,
                None,
                n_positive,
                n_faces,
                0,
                odt.empty_2d((0, 3), wp.int32, device=device),
                positive,
            ],
            device=device,
        )
        return _launch.clone(vertices), _launch.clone(faces), positive

    # One new vertex per crossed *edge*: both faces sharing it address the same index, which is
    # what makes the cut watertight rather than a seam of coincident pairs. Only a cut face has a
    # crossed edge, so the edges are numbered by sorting the cut faces' crossed halfedges alone --
    # in the order ``edges.edges_unique`` would list them.
    n_uncut = n_positive + n_negative
    n_cut = n_edges + n_corner
    n_emitted = n_uncut + 3 * n_edges + 2 * n_corner
    n_halfedges = 3 * n_cut
    key_base = wp.int64(n_vertices)
    keys = _launch.empty(2 * n_halfedges, dtype=wp.int64, device=device)
    # The sort's payload, the crossed-edge ranks and each halfedge's new vertex: one allocation.
    halfedges, rank, cut_vertices = _launch.carve(
        wp.int32, [2 * n_halfedges, n_halfedges, n_halfedges], device
    )
    _launch.launch(
        kernel_intersections.split_crossed_halfedge_keys,
        dim=(n_faces, 3),
        inputs=[faces, vertex_dots, ranks, n_edges, key_base, keys, halfedges],
        device=device,
    )
    # Every key is below ``n_vertices ** 2`` or the not-crossed marker equal to it.
    _launch.radix_sort_pairs(
        keys, halfedges, n_halfedges, end_bit=max(1, (n_vertices * n_vertices).bit_length())
    )
    _launch.launch(
        kernel_intersections.split_crossed_edge_flags,
        dim=n_halfedges,
        inputs=[keys, key_base, rank],
        device=device,
    )
    n_new = _scan_cut_flags(rank)
    all_vertices = _launch.empty(n_vertices + n_new, dtype=wp.vec3, device=device)
    # The same launch copies the input positions to the front of ``all_vertices``.
    _launch.launch(
        kernel_intersections.split_crossed_edge_vertices,
        dim=max(n_halfedges, n_vertices),
        inputs=[
            vertices,
            vertex_dots,
            wp.vec3(),
            wp.vec3(),
            keys,
            halfedges,
            rank,
            key_base,
            wp.int32(n_vertices),
            cut_vertices,
            all_vertices,
        ],
        device=device,
    )

    all_faces = _launch.empty(3 * n_emitted, dtype=wp.int32, device=device)
    positive = _launch.empty(n_emitted, dtype=wp.bool, device=device)
    _launch.launch(
        kernel_intersections.emit_split_faces,
        dim=n_faces,
        inputs=[
            faces,
            vertex_dots,
            ranks,
            cut_vertices,
            wp.int32(n_positive),
            wp.int32(n_uncut),
            wp.int32(n_edges),
            all_faces.reshape((n_emitted, 3)),
            positive,
        ],
        device=device,
    )

    return all_vertices, all_faces, positive


def _clip_with_vertex_field(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    plane: tuple[wp.vec3, wp.vec3] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Keep the ``vertex_dots >= 0`` region, the engine behind both public clippers.

    ``plane`` is the plane clip's ``(normal, origin)``: the field is then its signed distance,
    formed in the kernels where an empty ``vertex_dots`` leaves it to them, and the normal is the
    tie-break for a face lying in the level set -- the *only* other difference between the two: a
    general field has no orientation to compare, so such a face is dropped.
    """
    new_vertices, new_faces, _ = _clip_with_sources(vertices, faces, vertex_dots, plane)
    return new_vertices, new_faces


def _clip_with_sources(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    plane: tuple[wp.vec3, wp.vec3] | None = None,
    *,
    merge_crossings: bool = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32] | None]:
    """
    ``_clip_with_vertex_field``, plus each output vertex's source when ``merge_crossings`` is set.

    With ``merge_crossings`` the cut faces' crossing points that are bitwise equal -- the two copies
    of every rim edge's crossing, evaluated from its lower-numbered endpoint on both sides -- become
    one vertex, while the input vertices are left exactly as given; and the third return maps every
    output vertex to its input vertex, or to an index ``>= vertices.size`` for a crossing point.
    Without it the third return is ``None``.

    One classifying launch writes every flag one inclusive scan needs to rank the kept faces within
    their class, the crossing points and the referenced input vertices
    (``kernels/intersection.classify_clip_faces``); one small readback sizes the output, and one
    launch writes it whole (``emit_clip_faces``). The cap keeps every input vertex and every
    crossing addressable instead, so that its exact merge numbers them as the uncompacted cut did.
    """
    device = vertices.device
    n_vertices = vertices.size
    n_faces = faces.size // 3
    plane_normal, plane_origin = (wp.vec3(), wp.vec3()) if plane is None else plane
    # The vertex flags are marks into zeros; without them every entry is written whole.
    rank_type = kernel_intersections.ClipRanks
    if merge_crossings:
        ranks = _launch.empty(n_faces, dtype=rank_type, device=device)
    else:
        ranks = _launch.zeros(max(n_faces, n_vertices), dtype=rank_type, device=device)
    _launch.launch(
        kernel_intersections.classify_clip_faces,
        dim=n_faces,
        inputs=[
            vertices,
            faces,
            vertex_dots,
            plane_normal,
            plane_origin,
            plane is not None,
            merge_crossings,
            ranks,
        ],
        device=device,
    )
    _launch.array_scan(ranks, out_array=ranks, inclusive=True)
    # The one host synchronization in the clip: the scan's last entry holds every count, and these
    # counts size every buffer.
    n_in, n_quad, n_tri, n_tri_crossings, n_kept = (int(c) for c in read_scalar(ranks))
    n_rows = n_in + 2 * n_quad + n_tri
    if n_rows == 0:
        return (
            _launch.empty(0, dtype=wp.vec3, device=device),
            _launch.empty(0, dtype=wp.int32, device=device),
            _launch.empty(0, dtype=wp.int32, device=device) if merge_crossings else None,
        )
    n_crossings = 2 * n_quad + n_tri_crossings
    if merge_crossings:
        n_kept = n_vertices
    out_faces = _launch.empty(3 * n_rows, dtype=wp.int32, device=device)
    # With nothing crossed the cap compacts the input buffer itself, so none is copied here.
    out_vertices = _launch.empty(
        0 if merge_crossings and n_crossings == 0 else n_kept + n_crossings,
        dtype=wp.vec3,
        device=device,
    )
    _launch.launch(
        kernel_intersections.emit_clip_faces,
        dim=max(n_faces, n_vertices),
        inputs=[
            vertices,
            faces,
            vertex_dots,
            plane_normal,
            plane_origin,
            ranks,
            n_in,
            n_quad,
            n_kept,
            not merge_crossings,
            out_vertices,
            out_faces.reshape((n_rows, 3)),
        ],
        device=device,
    )
    if not merge_crossings:
        return out_vertices, out_faces, None
    if n_crossings == 0:
        new_vertices, new_faces, _, source = od.repair.remove_unreferenced_vertices(
            vertices, out_faces, return_inverse=True
        )
        return new_vertices, new_faces, source
    # Each crossing's class is found among the crossings alone (an exact-equality table, the
    # ``points.point_duplicate_mask`` one) and every face index into the crossings is pointed at
    # its class's first member; the copies are then unreferenced and the compaction drops them.
    crossings = odt.as_dense(out_vertices[n_vertices:])
    slot_mask = kernel_grouping.hash_table_mask(n_crossings)
    first = _launch.full(slot_mask + 1, -1, dtype=wp.int32, device=device)
    slot = _launch.empty(n_crossings, dtype=wp.int32, device=device)
    _launch.launch(
        kernel_points.point_duplicate_first,
        dim=n_crossings,
        inputs=[crossings, slot_mask],
        outputs=[first, slot],
        device=device,
    )
    _launch.launch(
        kernel_intersections.remap_to_first_crossing,
        dim=out_faces.size,
        inputs=[first, slot, wp.int32(n_vertices), out_faces],
        device=device,
    )
    new_vertices, new_faces, _, source = od.repair.remove_unreferenced_vertices(
        out_vertices, out_faces, return_inverse=True
    )
    return new_vertices, new_faces, source


def _compact_cut_segments(
    cut: wp.array[wp.int32], segments: odt.Array2dVec3, *, return_rows: bool = False
) -> tuple[odt.Array2dVec3, wp.array[wp.int32] | None]:
    """
    Keep the rows of ``segments`` whose ``0`` / ``1`` ``cut`` flag is set, in order.

    ``cut`` is scanned in place (``_scan_cut_flags``), and its tail -- the kept count, the one
    readback -- sizes the outputs, which one ``compact_cut_segments`` launch fills; with
    ``return_rows`` it also writes the source row of each kept segment. Returns
    ``(segments, rows)``, the second ``None`` when not asked for.
    """
    device = cut.device
    n = cut.size
    n_kept = _scan_cut_flags(cut)
    out_segments = odt.empty_2d((n_kept, 2), wp.vec3, device=device)
    out_rows = _launch.empty(n_kept, dtype=wp.int32, device=device) if return_rows else None
    if n_kept > 0:
        _launch.launch(
            kernel_intersections.compact_cut_segments,
            dim=n,
            inputs=[cut, segments, out_segments, out_rows],
            device=device,
        )
    return out_segments, out_rows


def _scan_cut_flags(cut: wp.array[wp.int32]) -> int:
    """Scan ``0`` / ``1`` flags in place into an inclusive scan and return their total."""
    if cut.size == 0:
        return 0
    _launch.array_scan(cut, out_array=cut, inclusive=True)
    # The kept count sizes every compacted output.
    return int(read_scalar(cut))
