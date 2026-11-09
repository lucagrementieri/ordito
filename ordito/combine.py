"""
Assembling meshes from parts and splitting them back apart.

[`concatenate`][ordito.combine.concatenate] joins whole meshes into one buffer pair without
touching their geometry, and [`split`][ordito.combine.split] /
[`split_with_offsets`][ordito.combine.split_with_offsets] decompose one back into its connected
components.
Mirrors ``trimesh.util.concatenate`` and ``trimesh.Trimesh.split``.

Nothing here welds anything: ``concatenate`` packs parts side by side and leaves any seam open.
Joining two open meshes *across* one boundary loop each is [`stitch`][ordito.holes.stitch] and its
variants, in [`ordito.holes`][ordito.holes], where the minimum-weight triangulation machinery
they share with the hole fillers lives.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import cast

import numpy as np
import warp as wp

import ordito as od
import ordito.typing as odt
from ordito import _launch
from ordito._common import read_array, require_same_device, scan_count
from ordito.adjacency import _face_component_forest  # pyright: ignore[reportPrivateUsage]
from ordito.kernels import combine as kernel_combine
from ordito.selection import _submeshes_from_face_groups  # pyright: ignore[reportPrivateUsage]


def concatenate(
    meshes_data: Sequence[tuple[wp.array[wp.vec3], wp.array[wp.int32]]],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Concatenate meshes, each given as ``(vertices, faces)`` on the same device.

    Face indices are renumbered with cumulative vertex offsets, matching
    [`trimesh.util.concatenate`][] (with ordito's flat ``(3 * n_faces,)`` face
    layout instead of ``(n_faces, 3)``).

    Parameters
    ----------
    meshes_data
        Sequence of ``(vertices, faces)`` pairs using ordito's flat face layout.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(n_vertices,)`` combined vertices and ``(3 * n_faces,)`` reindexed faces, ``n_vertices``
        and ``n_faces`` summed over the inputs, on the shared device. An empty sequence carries no
        device to share, so the empty result is allocated on Warp's **current** device.

    Raises
    ------
    RuntimeError
        If the given vertex and face buffers are not all on one device.

    See Also
    --------
    [`split`][ordito.combine.split]
        The inverse, by connected component.
    [`concatenate`][ordito.array.concatenate]
        The buffer-level operation of the same name, which joins plain 1-D arrays and does no
        reindexing. Both names are required: this one mirrors
        [`trimesh.util.concatenate`][], that one [`numpy.concatenate`][].
    [`trimesh.util.concatenate`][]

    Examples
    --------
    The second part's face indices are shifted past the first part's 8 vertices:

    ```python
    import ordito as od

    box_vertices, box_faces = od.creation.box()
    tet_vertices, tet_faces = od.creation.tetrahedron()
    vertices, faces = od.combine.concatenate([(box_vertices, box_faces), (tet_vertices, tet_faces)])
    print(vertices.shape, faces.shape)  # (12,) (48,)
    print(faces.numpy()[36:39])  # [ 8  9 10]
    print(len(od.combine.split(vertices, faces)))  # 2
    ```
    """
    if len(meshes_data) == 0:
        return (_launch.empty(0, dtype=wp.vec3), _launch.empty(0, dtype=wp.int32))

    require_same_device(meshes_data=meshes_data)
    device = meshes_data[0][0].device
    if len(meshes_data) == 1:
        vertices, faces = meshes_data[0]
        return _launch.clone(vertices), _launch.clone(faces)
    packed = _pack_mesh_pieces(meshes_data)
    if packed is not None:
        return packed
    # Both packings' offsets are host metadata, read off each piece's length: they reach the device
    # once, as the one table the renumbering reads, rather than once per packing.
    vertex_counts = [vertices.size for vertices, _ in meshes_data]
    face_sizes = [faces.size for _, faces in meshes_data]
    concatenated_vertices = od.array.concatenate([vertices for vertices, _ in meshes_data])
    concatenated_faces = od.array.concatenate([faces for _, faces in meshes_data])
    total_indices = concatenated_faces.size
    if total_indices > 0:
        n_pieces = len(meshes_data)
        offsets = _launch.array(
            [
                *itertools.accumulate(face_sizes, initial=0),
                *itertools.accumulate(vertex_counts, initial=0),
            ],
            dtype=wp.int32,
            device=device,
        )
        _launch.launch(
            kernel_combine.offset_packed_faces,
            dim=total_indices,
            inputs=[
                odt.as_dense(offsets[: n_pieces + 1]),
                odt.as_dense(offsets[n_pieces + 1 :]),
                concatenated_faces,
            ],
            device=device,
        )
    return concatenated_vertices, concatenated_faces


def _pack_mesh_pieces(
    meshes_data: Sequence[tuple[wp.array[wp.vec3], wp.array[wp.int32]]],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]] | None:
    """
    Pack and renumber every piece in one launch, or return ``None`` where that cannot apply.

    One descriptor table holds two records a piece, its vertex words and its face indices with the
    piece's vertex offset as their shift (``kernels.combine.pack_mesh_pieces``), uploaded once.
    ``None`` -- leaving the packing to ``array.concatenate`` and the renumbering launch, which
    also validate -- unless every buffer is a contiguous rank-1 ``wp.array`` of ``vec3`` vertices
    and ``int32`` faces.
    """
    device = cast("wp.Device", meshes_data[0][0].device)
    n_pieces = len(meshes_data)
    if n_pieces < _launch.SEGMENTS_IN_ONE_LAUNCH_FROM and not device.is_cuda:
        # A CPU word loop runs 1.06-1.2x the per-buffer copies it replaces below that many pieces
        # (CUDA takes the launch at any count: it also saves the renumbering launch). Not
        # ``segments_in_one_launch``' mean-size test: copying large pieces one buffer at a time
        # still pays the renumbering launch, 1.11-1.68x this path at 32-128 pieces of 82 k faces.
        return None
    sizes: list[int] = []  # every piece's vertex count, then every piece's index count
    pointers: list[int] = []  # the same order
    face_sizes: list[int] = []
    face_pointers: list[int] = []
    for vertices, faces in meshes_data:
        if (
            type(vertices) is not wp.array
            or type(faces) is not wp.array
            or vertices.dtype is not wp.vec3
            or faces.dtype is not wp.int32
            or vertices.ndim != 1
            or faces.ndim != 1
            or not vertices.is_contiguous
            or not faces.is_contiguous
        ):
            return None
        sizes.append(vertices.size)
        pointers.append(vertices.ptr or 0)
        face_sizes.append(faces.size)
        face_pointers.append(faces.ptr or 0)
    sizes += face_sizes
    pointers += face_pointers
    counts = np.array(sizes, dtype=np.int64)
    # Each half's exclusive scan: a piece's first vertex, then a piece's first index.
    starts = counts.copy()
    starts[:n_pieces] = np.cumsum(counts[:n_pieces]) - counts[:n_pieces]
    starts[n_pieces:] = np.cumsum(counts[n_pieces:]) - counts[n_pieces:]
    total_vertices = int(starts[n_pieces - 1] + counts[n_pieces - 1])
    total_indices = int(starts[-1] + counts[-1])
    concatenated_vertices = _launch.empty(total_vertices, dtype=wp.vec3, device=device)
    concatenated_faces = _launch.empty(total_indices, dtype=wp.int32, device=device)
    # A vertex is three words.
    counts[:n_pieces] *= 3
    longest = int(counts.max())
    if longest == 0:
        return concatenated_vertices, concatenated_faces

    record_np = _launch.struct_records(kernel_combine.MeshPieceWords, 2 * n_pieces)
    record_np["source"]["data"] = pointers
    record_np["source"]["shape"][:, 0] = counts
    record_np["source"]["strides"][:, 0] = 4
    record_np["source"]["ndim"] = 1
    record_np["target"]["data"][:n_pieces] = concatenated_vertices.ptr or 0
    record_np["target"]["data"][n_pieces:] = concatenated_faces.ptr or 0
    record_np["target"]["shape"][:n_pieces, 0] = 3 * total_vertices
    record_np["target"]["shape"][n_pieces:, 0] = total_indices
    record_np["target"]["strides"][:, 0] = 4
    record_np["target"]["ndim"] = 1
    record_np["shift"][n_pieces:] = starts[:n_pieces]
    starts[:n_pieces] *= 3
    record_np["offset"] = starts
    record_np["count"] = counts
    descriptor = _launch.struct_array(record_np, kernel_combine.MeshPieceWords, device)
    width = _launch.segment_copy_width(longest, device)
    _launch.launch(
        kernel_combine.pack_mesh_pieces,
        dim=(2 * n_pieces, width),
        inputs=[descriptor, wp.int32(width)],
        device=device,
    )
    return concatenated_vertices, concatenated_faces


def split(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    copy: bool = False,
    labels: wp.array[wp.int32] | None = None,
) -> list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]]:
    """
    Split a mesh into connected components by face adjacency.

    Each returned pair is a compact ``(vertices, faces)`` submesh with vertices
    reindexed from zero, matching [`trimesh.graph.split`][] with
    ``only_watertight=False``. [`concatenate`][ordito.combine.concatenate] on the
    result recovers the input mesh (up to vertex/face ordering within each body).

    All components are extracted in one batched pass
    ([`split_with_offsets`][ordito.combine.split_with_offsets]); this is
    [`array.split`][ordito.array.split] over its two packed results.

    !!! note "The returned arrays are views"
        Each pair slices the two shared buffers ``split_with_offsets`` produced, which costs no
        device memory and no launches. Two consequences: holding on to a single component keeps
        *both* whole buffers alive, and writing into one component writes into the shared
        allocation. Pass ``copy=True`` for independent buffers.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer (same layout as
        [`face_adjacency`][ordito.adjacency.face_adjacency]).
    copy
        Return independent buffers instead of views into the batched result. One allocation per
        buffer of each component, so the returned list no longer pins the batched buffers.
    labels
        ``(n_faces,)`` face component labels, as
        [`split_with_offsets`][ordito.combine.split_with_offsets] takes them.

    Returns
    -------
    list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]]
        One ``(vertices, faces)`` pair per face-connected component on
        ``vertices.device``. Empty when ``n_faces == 0``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``labels`` are not all on one device.

    See Also
    --------
    [`split_with_offsets`][ordito.combine.split_with_offsets]
        The packed form this splits: every component in two buffers plus offsets.
    [`concatenate`][ordito.combine.concatenate]
    [`face_connected_component_labels`][ordito.adjacency.face_connected_component_labels]
    [`repair.remove_small_components`][ordito.repair.remove_small_components]
        Keep a *subset* of the components in one mesh, rather than taking them all apart.
    [`submesh_from_face_indices`][ordito.selection.submesh_from_face_indices]
    [`trimesh.graph.split`][]

    Examples
    --------
    Two disjoint boxes come back as two 8-vertex meshes:

    ```python
    import ordito as od

    first = od.creation.box(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    second = od.creation.box(bounds=[[2.0, 0.0, 0.0], [3.0, 1.0, 1.0]])
    vertices, faces = od.combine.concatenate([first, second])
    parts = od.combine.split(vertices, faces)
    print(len(parts))  # 2
    part_vertices, part_faces = parts[1]
    print(part_vertices.size, part_faces.size // 3)  # 8 12
    ```
    """
    require_same_device(vertices=vertices, faces=faces, labels=labels)
    vertices_all, faces_all, offsets = _split_packed(vertices, faces, labels)
    n_components = offsets.size // 2 - 1
    if n_components == 0:
        return []
    if n_components == 1:
        # One component is the whole of both buffers, so its offsets need no readback.
        if copy:
            return [(_launch.clone(vertices_all), _launch.clone(faces_all))]
        return [(vertices_all, faces_all)]
    # The one readback: both offsets share a buffer, so a second read is not needed.
    # Host bounds as array slices: a list round trip of the offsets cost more than the read saved.
    bounds = cast("Sequence[int]", read_array(offsets))
    component_vertices = od.array.split(vertices_all, bounds[n_components + 1 :], copy=copy)
    component_faces = od.array.split(faces_all, bounds[: n_components + 1], copy=copy, row_size=3)
    return list(zip(component_vertices, component_faces, strict=True))


def split_with_offsets(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    labels: wp.array[wp.int32] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Split a mesh into connected components, returned as two CSR buffers.

    Same decomposition as [`split`][ordito.combine.split] — same components, same vertex and face
    order — but with no per-component Python at all, which is the only truly ``O(1)``-in-``k``
    form. Prefer it when the component count is large or when the components feed straight back
    into another batched kernel.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        ``(3 * n_faces,)`` flat triangle index buffer.
    labels
        ``(n_faces,)`` face component labels of ``faces``, each component's smallest face id, as
        [`face_connected_component_labels`][ordito.adjacency.face_connected_component_labels]
        returns them
        ([`Trimesh.face_connected_component_labels`][ordito.mesh.Trimesh.face_connected_component_labels]),
        to spare the labelling. Trusted, not checked: the components come out in the order of these
        labels.

    Returns
    -------
    vertices_all : wp.array[wp.vec3]
        ``(m,)`` every component's compacted vertices, concatenated, ``m <= n_vertices``.
    vertex_offsets : wp.array[wp.int32]
        ``(k + 1,)`` total-terminated offsets over the ``k`` components: component ``g`` owns
        ``vertices_all[vertex_offsets[g] : vertex_offsets[g + 1]]``.
    faces_all : wp.array[wp.int32]
        ``(3 * n_faces,)`` every component's reindexed flat faces, concatenated.
    face_offsets : wp.array[wp.int32]
        ``(k + 1,)`` total-terminated offsets **in faces, not indices**: component ``g`` owns
        ``faces_all[3 * face_offsets[g] : 3 * face_offsets[g + 1]]``.

    When ``n_faces == 0`` the buffers are empty, ``k == 0`` and both offsets are ``[0]``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``labels`` are not all on one device.

    See Also
    --------
    [`split`][ordito.combine.split]
        The list form: [`array.split`][ordito.array.split] over these buffers.
    [`submeshes_from_face_groups`][ordito.selection.submeshes_from_face_groups]

    Examples
    --------
    A box and a tetrahedron, packed: component ``g`` owns rows ``offsets[g]`` to ``offsets[g + 1]``:

    ```python
    import ordito as od

    vertices, faces = od.combine.concatenate([od.creation.box(), od.creation.tetrahedron()])
    vertices_all, vertex_offsets, faces_all, face_offsets = od.combine.split_with_offsets(
        vertices, faces
    )
    print(sorted(vertex_offsets.numpy().tolist()))  # [0, 8, 12]
    print(face_offsets.numpy())  # [ 0 12 16]
    ```
    """
    require_same_device(vertices=vertices, faces=faces, labels=labels)
    vertices_all, faces_all, offsets = _split_packed(vertices, faces, labels)
    k = offsets.size // 2 - 1
    return (
        vertices_all,
        _launch.view(offsets, k + 1, 2 * k + 2),
        faces_all,
        _launch.view(offsets, 0, k + 1),
    )


def _split_packed(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], labels: wp.array[wp.int32] | None
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    ``split_with_offsets`` with both offsets in one ``(2 * k + 2,)`` buffer, face offsets first.

    One buffer is one readback for ``split``, which needs both on the host.
    """
    device = vertices.device
    n_faces = faces.size // 3
    if n_faces == 0:
        return (
            _launch.empty(0, dtype=wp.vec3, device=device),
            _launch.empty(0, dtype=wp.int32, device=device),
            _launch.zeros(2, dtype=wp.int32, device=device),
        )

    # The labels and the identity payload go straight into the sort's double-width buffers, with
    # each component's root flagged in the keys' upper half; the flags' scan numbers the components.
    keys_buffer, order_buffer = _launch.sort_buffers(n_faces, wp.int32, device)
    if labels is None:
        parents = _face_component_forest(faces, vertices.size, None)
        _launch.launch(
            kernel_combine.flatten_labels_and_identity,
            dim=n_faces,
            inputs=[parents, keys_buffer, order_buffer],
            device=device,
        )
    else:
        _launch.launch(
            kernel_combine.labels_and_identity,
            dim=n_faces,
            inputs=[labels, keys_buffer, order_buffer],
            device=device,
        )
    n_components = scan_count(_launch.view(keys_buffer, n_faces, 2 * n_faces))

    if n_components == 1:
        # One component is every face in its own order: no sort, and both offsets are known here.
        # The memset under the extraction stays: its vertex mask is scattered from the faces, so
        # an identity order still needs the zeros (about 1 % of the call on CUDA).
        component_vertices, component_faces = od.selection.submesh_from_face_indices(
            vertices, faces, _launch.view(order_buffer, 0, n_faces)
        )
        offsets = _launch.array(
            [0, n_faces, 0, component_vertices.size], dtype=wp.int32, device=device
        )
        return component_vertices, component_faces, offsets

    # Sorting by component number rather than label orders ``key_bits(k)`` bits, not
    # ``key_bits(n_faces)``; the stable sort keeps each component's faces ascending.
    _launch.launch(
        kernel_combine.component_rank_keys, dim=n_faces, inputs=[keys_buffer], device=device
    )
    sorted_components, sorted_face_ids = od.array.sort_and_argsort_in_place(
        keys_buffer, order_buffer, end_bit=_launch.key_bits(n_components)
    )
    offsets = _launch.empty(2 * n_components + 2, dtype=wp.int32, device=device)
    face_offsets = _launch.view(offsets, 0, n_components + 1)
    _launch.launch(
        kernel_combine.component_face_offsets,
        dim=n_components + 1,
        inputs=[sorted_components, face_offsets],
        device=device,
    )
    vertices_all, faces_all = _submeshes_from_face_groups(
        vertices,
        faces,
        sorted_face_ids,
        face_offsets,
        _launch.view(offsets, n_components + 1, 2 * n_components + 2),
    )
    return vertices_all, faces_all, offsets
