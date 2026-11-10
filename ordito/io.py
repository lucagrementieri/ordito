"""Mesh file loading via the optional ``meshio`` dependency."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Required, TypedDict

import numpy as np
import warp as wp

from ordito import _launch
from ordito.mesh import Trimesh

if TYPE_CHECKING:
    # meshio ships neither a ``py.typed`` marker nor stubs, and there is no ``types-meshio``
    # on PyPI. It is an optional dependency reached through ``_import_meshio`` alone, so the
    # untyped surface is confined to this module.
    import meshio  # pyright: ignore[reportMissingTypeStubs]

# Per-vertex / per-face attribute column names as written by common mesh formats.
_NORMAL_COLUMNS = ("nx", "ny", "nz")
_UV_COLUMN_SETS = (("u", "v"), ("s", "t"), ("texture_u", "texture_v"))
_COLOR_COLUMNS = ("red", "green", "blue")
_COLOR_COLUMNS_RGBA = ("red", "green", "blue", "alpha")


class MeshData(TypedDict, total=False):
    """
    The attributes [`load_mesh_data`][ordito.io.load_mesh_data] read from a file, by name.

    A plain ``dict`` at runtime. ``vertices`` is always present; every other key only when the
    file carries that attribute.

    Examples
    --------
    Only the attributes the file carries appear as keys:

    ```python
    import pathlib
    import tempfile

    import meshio
    import numpy as np

    import ordito as od

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "triangle.ply"
        meshio.write(path, meshio.Mesh(points, [("triangle", np.array([[0, 1, 2]]))]))
        data = od.io.load_mesh_data(path)
    print(sorted(data))  # ['faces', 'vertices']
    ```
    """

    vertices: Required[wp.array[wp.vec3]]
    faces: wp.array[wp.int32]
    vertex_normals: wp.array[wp.vec3]
    uv: wp.array[wp.vec2]
    corner_uv: wp.array[wp.vec2]
    colors: wp.array[wp.vec3] | wp.array[wp.vec4]
    face_normals: wp.array[wp.vec3]


def load_mesh_data(path: str | Path, *, device: wp.DeviceLike = None) -> MeshData:
    """
    Load every mesh attribute ``meshio`` can read from a file into Warp arrays.

    Reads ``path`` once with ``meshio`` and returns a dictionary whose keys are present
    only when the corresponding data exists in the file. ``vertices`` is always present;
    everything else is optional.

    Parameters
    ----------
    path
        Path to a mesh file in any format supported by ``meshio`` (PLY, STL, OBJ, OFF,
        VTK, glTF, ...).
    device
        Warp device for the returned arrays. Defaults to the current Warp device.

    Returns
    -------
    MeshData
        Mapping with these possible keys:

        - ``vertices`` : ``wp.array[wp.vec3]`` of ``float32`` vertex positions (always present).
        - ``faces`` : ``(3 * n_faces,)`` flat ``wp.array[wp.int32]`` (triangle cells only).
        - ``vertex_normals`` : ``wp.array[wp.vec3]`` from ``nx, ny, nz`` per-vertex data.
        - ``uv`` : ``wp.array[wp.vec2]`` from ``u, v`` (or ``s, t`` / ``texture_u, texture_v``).
        - ``corner_uv`` : ``(3 * n_faces,)`` ``wp.array[wp.vec2]``, one texcoord per face corner,
          from an OBJ file's ``vt`` indices or a PLY file's ``texcoord`` face list -- the form of
          a UV atlas, whose seams give one vertex several texcoords.
          [`split_vertices_by_corner_values`][ordito.repair.split_vertices_by_corner_values]
          turns it into per-vertex ``uv``.
        - ``colors`` : ``wp.array[wp.vec3]`` or ``wp.array[wp.vec4]`` from ``red, green, blue``
          (and optional ``alpha``); integer channels are normalized to ``[0, 1]``.
        - ``face_normals`` : ``wp.array[wp.vec3]`` — only when the file stored per-face normals.

    Raises
    ------
    ValueError
        If an OBJ file's face names a vertex or a texcoord it does not define, or a PLY file with
        per-corner texcoords is malformed or names a vertex it does not define.

    See Also
    --------
    [`load_mesh`][ordito.io.load_mesh]

    Notes
    -----
    Only triangle cells are consumed; non-triangle cell blocks are ignored. ``meshio`` reads
    attributes verbatim and never computes them, so ``face_normals`` appears only when the
    file itself contained per-face normals.

    An OBJ file is read here rather than by ``meshio``, which refuses one whose ``vt`` count
    differs from its ``v`` count -- every UV-mapped OBJ with a seam. Its positions, triangles and
    ``corner_uv`` are returned; polygons with more than three corners are ignored as above. A PLY
    file whose faces carry a ``texcoord`` (or ``texcoords``) list, which ``meshio`` cannot decode,
    is read here too, ASCII or binary, with every attribute above.

    Examples
    --------
    A box written to PLY with per-vertex normals reads back with them:

    ```python
    import pathlib
    import tempfile

    import meshio

    import ordito as od

    vertices, faces = od.creation.box()
    normals = od.normals.vertex_normals(vertices, faces).numpy()
    mesh = meshio.Mesh(
        vertices.numpy(),
        [("triangle", faces.numpy().reshape(-1, 3))],
        point_data={"nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2]},
    )
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "box.ply"
        meshio.write(path, mesh)
        data = od.io.load_mesh_data(path)
    print(sorted(data))  # ['faces', 'vertex_normals', 'vertices']
    print(data["faces"].shape, data["vertex_normals"].shape)  # (36,) (8,)
    ```
    """
    if Path(path).suffix.lower() == ".obj":
        return _load_obj_data(path, device)
    if _is_ply_with_corner_texcoords(path):
        ply = _read_ply(path)
        if ply.faces.size > 0:
            _require_valid_faces_host(ply.faces, len(ply.points), "load_mesh_data")
        return _mesh_data(
            ply.points, ply.faces, ply.point_data, ply.face_normals, ply.corner_uv, device
        )
    meshio = _import_meshio()
    mesh = meshio.read(path)
    return _mesh_data(
        mesh.points,
        mesh.cells_dict.get("triangle"),
        mesh.point_data,
        _face_normals_from_cell_data(mesh),
        None,
        device,
    )


def _mesh_data(
    points: np.ndarray,
    faces_np: np.ndarray | None,
    point_data: dict[str, np.ndarray],
    face_normals: np.ndarray | None,
    corner_uv: np.ndarray | None,
    device: wp.DeviceLike,
) -> MeshData:
    """Return ``load_mesh_data``'s dictionary from a decoded file's host arrays."""
    points = np.ascontiguousarray(points, dtype=np.float32)
    result: MeshData = {"vertices": _launch.array(points, dtype=wp.vec3, device=device)}

    if faces_np is not None and len(faces_np) > 0:
        faces_flat = np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32)
        result["faces"] = _launch.array(faces_flat, dtype=wp.int32, device=device)
        if corner_uv is not None:
            result["corner_uv"] = _launch.array(
                np.ascontiguousarray(corner_uv, dtype=np.float32), dtype=wp.vec2, device=device
            )

    normals = _stack_columns(point_data, _NORMAL_COLUMNS)
    if normals is not None:
        result["vertex_normals"] = _launch.array(
            np.ascontiguousarray(normals, dtype=np.float32), dtype=wp.vec3, device=device
        )

    for names in _UV_COLUMN_SETS:
        uv = _stack_columns(point_data, names)
        if uv is not None:
            result["uv"] = _launch.array(
                np.ascontiguousarray(uv, dtype=np.float32), dtype=wp.vec2, device=device
            )
            break

    colors_raw = _stack_columns(point_data, _COLOR_COLUMNS_RGBA)
    has_alpha = colors_raw is not None
    if colors_raw is None:
        colors_raw = _stack_columns(point_data, _COLOR_COLUMNS)
    if colors_raw is not None:
        colors = colors_raw.astype(np.float32)
        if np.issubdtype(colors_raw.dtype, np.integer):
            colors /= 255.0
        colors = np.ascontiguousarray(colors, dtype=np.float32)
        result["colors"] = (
            _launch.array(colors, dtype=wp.vec4, device=device)
            if has_alpha
            else _launch.array(colors, dtype=wp.vec3, device=device)
        )

    if face_normals is not None:
        result["face_normals"] = _launch.array(
            np.ascontiguousarray(face_normals, dtype=np.float32), dtype=wp.vec3, device=device
        )

    return result


def _load_obj_data(path: str | Path, device: wp.DeviceLike) -> MeshData:
    """Return ``load_mesh_data`` of an OBJ file: positions, triangles and per-corner texcoords."""
    points, faces, corner_uv = _read_obj(path)
    result: MeshData = {"vertices": _launch.array(points, dtype=wp.vec3, device=device)}
    if faces.size > 0:
        _require_valid_faces_host(faces, len(points), "load_mesh_data")
        result["faces"] = _launch.array(faces.astype(np.int32), dtype=wp.int32, device=device)
        if corner_uv is not None:
            result["corner_uv"] = _launch.array(corner_uv, dtype=wp.vec2, device=device)
    return result


def _read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Parse an OBJ file's ``v``, ``vt`` and triangular ``f`` lines on the host.

    Returns ``(points, faces, corner_uv)``: ``(n, 3)`` ``float32`` positions, ``(3 * n_faces,)``
    ``int64`` zero-based corners (unvalidated) and ``(3 * n_faces, 2)`` ``float32`` texcoords, or
    ``None`` unless every corner names one. Negative (relative) indices count back from the
    elements read so far, as the format defines.
    """
    positions: list[list[float]] = []
    texcoords: list[list[float]] = []
    corners: list[int] = []
    corner_texcoords: list[int] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "v":
                positions.append([float(x) for x in tokens[1:4]])
            elif tokens[0] == "vt":
                texcoords.append([float(x) for x in tokens[1:3]])
            elif tokens[0] == "f" and len(tokens) == 4:
                for token in tokens[1:]:
                    fields = token.split("/")
                    index = int(fields[0])
                    corners.append(index - 1 if index > 0 else len(positions) + index)
                    if len(fields) > 1 and fields[1]:
                        t = int(fields[1])
                        corner_texcoords.append(t - 1 if t > 0 else len(texcoords) + t)
                    else:
                        corner_texcoords.append(-1)
    points = np.ascontiguousarray(np.asarray(positions, dtype=np.float32).reshape(-1, 3))
    faces = np.asarray(corners, dtype=np.int64)
    corner_index = np.asarray(corner_texcoords, dtype=np.int64)
    corner_uv = None
    if texcoords and corners and (corner_index >= 0).all():
        table = np.asarray(texcoords, dtype=np.float32).reshape(-1, 2)
        if int(corner_index.max()) >= len(table):
            raise ValueError(f"OBJ file {path!r} names a texcoord past its {len(table)} vt lines")
        corner_uv = np.ascontiguousarray(table[corner_index])
    return points, faces, corner_uv


class _PlyProperty(NamedTuple):
    """One PLY property: its name, NumPy value type, and NumPy count type (``None``: a scalar)."""

    name: str
    dtype: str
    count_dtype: str | None


class _PlyElement(NamedTuple):
    """One PLY element: its name, row count and properties in file order."""

    name: str
    n_rows: int
    properties: list[_PlyProperty]


class _PlyMesh(NamedTuple):
    """A PLY file's triangles and attributes, as ``_mesh_data`` takes them."""

    points: np.ndarray
    faces: np.ndarray
    point_data: dict[str, np.ndarray]
    face_normals: np.ndarray | None
    corner_uv: np.ndarray | None


# PLY scalar types by every name the format allows, as NumPy type codes (byte order added per file).
_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
# The face list properties that hold one texcoord pair per corner.
_PLY_TEXCOORD_LISTS = ("texcoord", "texcoords")


def _is_ply_with_corner_texcoords(path: str | Path) -> bool:
    """Whether ``path`` is a PLY file whose faces carry a per-corner texcoord list."""
    if Path(path).suffix.lower() != ".ply":
        return False
    try:
        _, elements, _ = _read_ply_header(path)
    except (ValueError, OSError):
        return False  # a header this reader does not understand is left to ``meshio``
    return any(
        prop.name in _PLY_TEXCOORD_LISTS and prop.count_dtype is not None
        for element in elements
        if element.name == "face"
        for prop in element.properties
    )


def _read_ply_header(path: str | Path) -> tuple[str, list[_PlyElement], int]:
    """Return a PLY file's ``(format, elements, body offset)``, reading its header alone."""
    elements: list[_PlyElement] = []
    file_format = ""
    with open(path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"PLY file {path!r} does not start with 'ply'")
        for line in handle:
            tokens = line.decode("ascii", errors="replace").split()
            if not tokens or tokens[0] in ("comment", "obj_info"):
                continue
            if tokens[0] == "end_header":
                return file_format, elements, handle.tell()
            if tokens[0] == "format" and len(tokens) == 3:
                file_format = tokens[1]
            elif tokens[0] == "element" and len(tokens) == 3:
                elements.append(_PlyElement(tokens[1], int(tokens[2]), []))
            elif tokens[0] == "property" and elements and len(tokens) in (3, 5):
                if len(tokens) == 5 and tokens[1] == "list":
                    prop = _PlyProperty(tokens[4], _ply_type(tokens[3]), _ply_type(tokens[2]))
                else:
                    prop = _PlyProperty(tokens[-1], _ply_type(tokens[1]), None)
                elements[-1].properties.append(prop)
            else:
                raise ValueError(f"PLY file {path!r} has a malformed header line {line!r}")
    raise ValueError(f"PLY file {path!r} has no end_header")


def _ply_type(name: str) -> str:
    """Return a PLY scalar type's NumPy type code."""
    if name not in _PLY_TYPES:
        raise ValueError(f"unknown PLY type {name!r}")
    return _PLY_TYPES[name]


def _read_ply(path: str | Path) -> _PlyMesh:
    """
    Read a PLY file on the host: positions, triangles, vertex columns, face normals, texcoords.

    ASCII and binary of either byte order. Only faces of three corners are kept, as for OBJ, and
    ``corner_uv`` is ``None`` unless every kept face carries three texcoord pairs. The face
    indices are returned unvalidated.
    """
    file_format, elements, offset = _read_ply_header(path)
    with open(path, "rb") as handle:
        handle.seek(offset)
        body = handle.read()
    if file_format == "ascii":
        columns = _ply_ascii_columns(path, body, elements)
    elif file_format in ("binary_little_endian", "binary_big_endian"):
        order = "<" if file_format == "binary_little_endian" else ">"
        columns = _ply_binary_columns(path, body, elements, order)
    else:
        raise ValueError(f"PLY file {path!r} has an unknown format {file_format!r}")

    vertex = columns.get("vertex", {})
    if not all(name in vertex for name in "xyz"):
        raise ValueError(f"PLY file {path!r} has no vertex x, y and z")
    points = np.column_stack([vertex[name] for name in "xyz"]).astype(np.float32)
    point_data = {
        name: values
        for name, values in vertex.items()
        if name not in "xyz" and isinstance(values, np.ndarray)
    }
    face = columns.get("face", {})
    corners: list[np.ndarray] = face.get("vertex_indices", face.get("vertex_index", []))
    triangle = np.array([row.size == 3 for row in corners], dtype=bool)
    faces = np.zeros(0, dtype=np.int64)
    if triangle.any():
        faces = np.concatenate([row for row, kept in zip(corners, triangle, strict=True) if kept])
        faces = faces.astype(np.int64)
    face_normals = None
    if all(name in face for name in ("nx", "ny", "nz")):
        face_normals = np.column_stack([face[name][triangle] for name in ("nx", "ny", "nz")])
    corner_uv = None
    texcoords = next((face[name] for name in _PLY_TEXCOORD_LISTS if name in face), None)
    if texcoords is not None and triangle.any():
        kept_texcoords = [row for row, kept in zip(texcoords, triangle, strict=True) if kept]
        if all(row.size == 6 for row in kept_texcoords):
            corner_uv = np.concatenate(kept_texcoords).astype(np.float32).reshape(-1, 2)
    return _PlyMesh(points, faces, point_data, face_normals, corner_uv)


def _ply_ascii_columns(
    path: str | Path, body: bytes, elements: list[_PlyElement]
) -> dict[str, dict[str, Any]]:
    """
    Decode an ASCII PLY body into ``{element: {property: values}}``.

    A scalar property is a ``(count,)`` array, a list property a list of one array a row.
    """
    lines = [line for line in body.decode("ascii", errors="replace").splitlines() if line.strip()]
    columns: dict[str, dict[str, Any]] = {}
    start = 0
    for element in elements:
        rows = lines[start : start + element.n_rows]
        start += element.n_rows
        if len(rows) != element.n_rows:
            raise ValueError(f"PLY file {path!r} ends inside element {element.name!r}")
        values: dict[str, list[Any]] = {prop.name: [] for prop in element.properties}
        for row in rows:
            tokens = row.split()
            column = 0
            for prop in element.properties:
                if prop.count_dtype is None:
                    values[prop.name].append(float(tokens[column]))
                    column += 1
                    continue
                width = int(tokens[column])
                items = np.asarray(tokens[column + 1 : column + 1 + width], dtype=np.float64)
                values[prop.name].append(items.astype(prop.dtype))
                column += 1 + width
            if column != len(tokens):
                raise ValueError(f"PLY file {path!r} has a malformed {element.name!r} row")
        columns[element.name] = {
            prop.name: values[prop.name]
            if prop.count_dtype is not None
            else np.asarray(values[prop.name]).astype(prop.dtype)
            for prop in element.properties
        }
    return columns


def _ply_binary_columns(
    path: str | Path, body: bytes, elements: list[_PlyElement], order: str
) -> dict[str, dict[str, Any]]:
    """
    Decode a binary PLY body into ``{element: {property: values}}``, as ``_ply_ascii_columns``.

    An element whose lists all keep their first row's length is read as one structured array;
    any other is walked row by row.
    """
    columns: dict[str, dict[str, Any]] = {}
    position = 0
    for element in elements:
        fields: list[Any] = []
        cursor = position
        for k, prop in enumerate(element.properties):
            item = np.dtype(order + prop.dtype)
            if prop.count_dtype is None:
                fields.append((f"v{k}", item))
                cursor += item.itemsize
                continue
            count_type = np.dtype(order + prop.count_dtype)
            width = 0
            if element.n_rows > 0 and cursor + count_type.itemsize <= len(body):
                width = int(np.frombuffer(body, count_type, 1, cursor)[0])
            fields.extend([(f"c{k}", count_type), (f"v{k}", item, (width,))])
            cursor += count_type.itemsize + width * item.itemsize
        row_type = np.dtype(fields)
        table = None
        if position + element.n_rows * row_type.itemsize <= len(body):
            table = np.frombuffer(body, row_type, element.n_rows, position)
            uniform = all(
                (table[f"c{k}"] == table[f"c{k}"][:1]).all()
                for k, prop in enumerate(element.properties)
                if prop.count_dtype is not None
            )
            table = table if uniform else None
        if table is not None:
            columns[element.name] = {
                prop.name: table[f"v{k}"].copy()
                if prop.count_dtype is None
                else list(table[f"v{k}"].copy())
                for k, prop in enumerate(element.properties)
            }
            position += element.n_rows * row_type.itemsize
            continue
        values: dict[str, list[Any]] = {prop.name: [] for prop in element.properties}
        for _ in range(element.n_rows):
            for prop in element.properties:
                item = np.dtype(order + prop.dtype)
                width = 1
                if prop.count_dtype is not None:
                    count_type = np.dtype(order + prop.count_dtype)
                    if position + count_type.itemsize > len(body):
                        raise ValueError(f"PLY file {path!r} ends inside {element.name!r}")
                    width = int(np.frombuffer(body, count_type, 1, position)[0])
                    position += count_type.itemsize
                if position + width * item.itemsize > len(body):
                    raise ValueError(f"PLY file {path!r} ends inside {element.name!r}")
                read = np.frombuffer(body, item, width, position)
                values[prop.name].append(read if prop.count_dtype is not None else read[0])
                position += width * item.itemsize
        columns[element.name] = {
            prop.name: values[prop.name]
            if prop.count_dtype is not None
            else np.asarray(values[prop.name])
            for prop in element.properties
        }
    return columns


def _stack_columns(data: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    """Stack scalar ``point_data``/``cell_data`` columns into ``(n, len(names))`` or ``None``."""
    if not all(name in data for name in names):
        return None
    return np.column_stack([np.asarray(data[name]) for name in names])


def _face_normals_from_cell_data(mesh: meshio.Mesh) -> np.ndarray | None:
    """Read ``nx, ny, nz`` cell data aligned with the triangle faces, if present."""
    # ``cell_data_dict[name]["triangle"]`` concatenates every triangle block in the same
    # order ``cells_dict["triangle"]`` does, so the rows line up with ``faces`` even when a
    # file splits its triangles across several blocks (gmsh / VTK physical groups). Reading
    # ``cell_data[name][block_index]`` instead returns only the first block's normals.
    cell_data_dict = mesh.cell_data_dict
    columns = []
    for name in _NORMAL_COLUMNS:
        per_type = cell_data_dict.get(name)
        if per_type is None or "triangle" not in per_type:
            return None
        columns.append(np.asarray(per_type["triangle"]))
    return np.column_stack(columns)


def _import_meshio():
    try:
        import meshio  # pyright: ignore[reportMissingTypeStubs]
    except ImportError as exc:
        raise ImportError(
            "ordito.io requires meshio. Install it with `pip install ordito[io]`."
        ) from exc
    return meshio


def load_mesh(path: str | Path, *, device: wp.DeviceLike = None) -> wp.Mesh:
    """
    Load a triangle mesh file into a ``warp.Mesh``.

    Reads only the vertices and triangle faces (the subset
    [`load_mesh_data`][ordito.io.load_mesh_data] returns) and wraps them in a ``warp.Mesh`` that
    owns its buffers.

    Parameters
    ----------
    path
        Path to a mesh file in any format supported by ``meshio``.
    device
        Warp device for the returned mesh. Defaults to the current Warp device.

    Returns
    -------
    warp.Mesh
        Mesh with ``vec3`` points and flat ``int32`` indices.

    Raises
    ------
    ValueError
        If the file contains no triangle faces (e.g. a point cloud or a triangle-strip
        PLY that ``meshio`` cannot decode), if a face references a vertex index the file's own
        vertex count does not cover, or if a PLY file with per-corner texcoords is malformed.

    See Also
    --------
    [`load_mesh_data`][ordito.io.load_mesh_data]

    Examples
    --------
    Round-trip an icosphere through an OBJ file:

    ```python
    import pathlib
    import tempfile

    import meshio

    import ordito as od

    vertices, faces = od.creation.icosphere(subdivisions=1)
    mesh = meshio.Mesh(vertices.numpy(), [("triangle", faces.numpy().reshape(-1, 3))])
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "sphere.obj"
        meshio.write(path, mesh)
        loaded = od.io.load_mesh(path)
    print(loaded.points.shape, loaded.indices.shape)  # (42,) (240,)
    ```
    """
    if Path(path).suffix.lower() == ".obj":
        points, faces_np, _ = _read_obj(path)
        if faces_np.size == 0:
            raise ValueError(f"Mesh file {path!r} has no triangle faces; cannot build a wp.Mesh.")
        _require_valid_faces_host(faces_np, len(points), "load_mesh")
        return wp.Mesh(
            points=_launch.array(points, dtype=wp.vec3, device=device),
            indices=_launch.array(faces_np.astype(np.int32), dtype=wp.int32, device=device),
        )
    if _is_ply_with_corner_texcoords(path):
        ply = _read_ply(path)
        points_np, faces_np = ply.points, ply.faces
    else:
        meshio = _import_meshio()
        mesh = meshio.read(path)
        points_np, faces_np = mesh.points, mesh.cells_dict.get("triangle")
    if faces_np is None or len(faces_np) == 0:
        raise ValueError(f"Mesh file {path!r} has no triangle faces; cannot build a wp.Mesh.")
    # A malformed file (or a meshio decoder bug) can hand back a face index the vertex buffer
    # doesn't cover; this is the trust boundary where that first enters ordito. The check reads
    # the decoder's own integers before the ``int32`` narrowing, which would wrap an index at or
    # above ``2**31`` into range.
    _require_valid_faces_host(faces_np, len(points_np), "load_mesh")
    vertices_wp = _launch.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = _launch.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    # Both buffers were allocated on the lines above and nothing else holds them, so the mesh
    # can own them directly.
    return wp.Mesh(points=vertices_wp, indices=faces_wp)


def mesh_from_numpy(
    vertices: np.ndarray, faces: np.ndarray, *, device: wp.DeviceLike = None
) -> Trimesh:
    """
    Build a [`Trimesh`][ordito.mesh.Trimesh] from NumPy vertex and face arrays.

    Parameters
    ----------
    vertices
        ``(n_vertices, 3)`` vertex positions, any NumPy float dtype.
    faces
        ``(n_faces, 3)`` (or already-flat ``(3 * n_faces,)``) triangle vertex indices,
        any NumPy integer dtype.
    device
        Warp device for the returned mesh. Defaults to the current Warp device.

    Returns
    -------
    Trimesh
        New mesh with ``vec3`` vertices and a flat ``int32`` face buffer, owning freshly
        allocated Warp arrays (no aliasing with ``vertices`` / ``faces``).

    Raises
    ------
    ValueError
        If a face references a vertex index ``vertices`` does not cover.

    See Also
    --------
    [`load_mesh`][ordito.io.load_mesh]

    Examples
    --------
    ``(n_faces, 3)`` faces are flattened into ordito's ``(3 * n_faces,)`` layout:

    ```python
    import numpy as np

    import ordito as od

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = od.io.mesh_from_numpy(vertices, faces)
    print(mesh.n_vertices, mesh.n_faces, mesh.faces.shape)  # 4 2 (6,)
    print(f"{mesh.area:.4f}")  # 1.0000
    ```
    """
    # This is the trust boundary where an external caller's raw NumPy connectivity first enters
    # ordito. The range is checked on the caller's own integers, before the ``int32`` narrowing
    # (which would wrap an index at or above ``2**31`` into range) and before any upload.
    _require_valid_faces_host(faces, len(vertices), "mesh_from_numpy")
    vertices_wp = _launch.array(
        np.ascontiguousarray(vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = _launch.array(
        np.ascontiguousarray(faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    return Trimesh(vertices_wp, faces_wp)


def _require_valid_faces_host(faces: np.ndarray, n_vertices: int, name: str) -> None:
    """Raise ``ValueError`` if a host face index falls outside ``[0, n_vertices)``."""
    if faces.size == 0:
        return
    if faces.dtype.kind in "iu":
        # One reduction rather than a ``min`` and a ``max``: read as unsigned, a negative index
        # wraps above every vertex count, so the largest value alone decides the range.
        unsigned = faces.view(np.dtype(f"u{faces.dtype.itemsize}"))
        if int(unsigned.max()) < n_vertices:
            return
    lo = int(faces.min())
    hi = int(faces.max())
    if lo < 0 or hi >= n_vertices:
        raise ValueError(
            f"{name}: faces must reference vertex indices in [0, {n_vertices}), "
            f"got a range of [{lo}, {hi}]"
        )
