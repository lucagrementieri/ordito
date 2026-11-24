"""Mesh file reading and writing, with colours, UVs and textures, through ``trimesh``."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Required, TypedDict

import numpy as np
import warp as wp

import ordito.typing as odt
from ordito import _launch
from ordito._common import read_array
from ordito.mesh import Trimesh
from ordito.texture import Material

if TYPE_CHECKING:
    # ``trimesh`` is an optional dependency reached through ``_import_trimesh`` alone; its material
    # is named here for the annotations only.
    from trimesh.visual.material import PBRMaterial

# The visuals each file format holds when trimesh writes it, by suffix: ``"texture"`` is a
# material's image and factor, ``"uv"`` texture coordinates alone. Measured by writing and reading
# back each format: PLY stores the UVs but writes no image; OBJ and glTF turn face colours into
# vertex colours; OFF and STL keep nothing beyond the geometry.
_FORMAT_VISUALS: dict[str, frozenset[str]] = {
    ".obj": frozenset({"vertex", "uv", "texture"}),
    ".ply": frozenset({"vertex", "face", "uv"}),
    ".glb": frozenset({"vertex", "uv", "texture"}),
    ".gltf": frozenset({"vertex", "uv", "texture"}),
}


class MeshData(TypedDict, total=False):
    """
    The arrays [`load_mesh_data`][ordito.io.load_mesh_data] read from a file, by name.

    A plain ``dict`` at runtime. ``vertices`` is always present; every other key only when the
    file carries that attribute.

    Examples
    --------
    Only the attributes the file carries appear as keys:

    ```python
    import pathlib
    import tempfile

    import ordito as od

    mesh = od.Trimesh(*od.creation.box())
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "box.ply"
        od.io.save_mesh(mesh, path)
        data = od.io.load_mesh_data(path)
    print(sorted(data))  # ['faces', 'vertices']
    ```
    """

    vertices: Required[wp.array[wp.vec3]]
    faces: wp.array[wp.int32]
    uv: wp.array[wp.vec2]
    colors: wp.array[wp.vec4]
    face_colors: wp.array[wp.vec4]


def load_trimesh(path: str | Path, *, device: wp.DeviceLike = None) -> Trimesh:
    """
    Load a mesh file into a [`Trimesh`][ordito.mesh.Trimesh], with its colours, UVs and texture.

    Parameters
    ----------
    path
        Path to a mesh file in any format ``trimesh`` reads (OBJ with its MTL and images, PLY,
        glTF / GLB, STL, OFF, ...; VTK, Gmsh and the other volume formats when ``meshio`` is
        installed too).
    device
        Warp device for the returned mesh. Defaults to the current Warp device.

    Returns
    -------
    Trimesh
        The file's triangles, polygons triangulated, with these visuals when the file has them:
        vertex colours or face colours as the ``"color"`` vertex or face attribute, texture
        coordinates as the ``"uv"`` vertex attribute, and the texture as its
        [`material`][ordito.mesh.Trimesh.material]. Colours are RGBA in ``[0, 1]``.

    Raises
    ------
    ImportError
        If ``trimesh`` is not installed.
    ValueError
        If a face references a vertex index the file's own vertex count does not cover.
    IndexError
        If an OBJ file's face does: trimesh's OBJ reader refuses it first.

    See Also
    --------
    [`save_mesh`][ordito.io.save_mesh]
        The inverse.
    [`load_mesh_data`][ordito.io.load_mesh_data]
        The same arrays as a dictionary.
    [`trimesh.load`][]

    Notes
    -----
    UVs are per vertex, as in trimesh: a file whose vertex takes several texcoords (a UV seam)
    reads as one vertex for each, so the vertices need not be numbered as in the file. A file with
    several materials, or several meshes, is joined into one mesh, its textures packed into one
    image. A material is read as its base colour: an MTL ``map_Kd`` image or ``Kd`` colour, or a
    glTF base colour texture and factor (the ``Kd`` beside a ``map_Kd`` is left out, as trimesh
    does when it reads colours off a texture). Normals in the file are not read: trimesh computes
    its own, and so does [`vertex_normals`][ordito.mesh.Trimesh.vertex_normals].

    Examples
    --------
    A coloured box saved and read back:

    ```python
    import pathlib
    import tempfile

    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    red = wp.full(8, wp.vec4(1.0, 0.0, 0.0, 1.0), dtype=wp.vec4, device=vertices.device)
    mesh = od.Trimesh(vertices, faces, vertex_attributes={"color": red})
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "box.ply"
        od.io.save_mesh(mesh, path)
        loaded = od.io.load_trimesh(path)
    print(loaded.n_vertices, loaded.n_faces, loaded.vertex_colors.numpy()[0].tolist())
    # 8 12 [1.0, 0.0, 0.0, 1.0]
    ```
    """
    read = _read(path, "load_trimesh", device)
    return Trimesh(
        read.vertices,
        read.faces,
        vertex_attributes=read.vertex_attributes,
        face_attributes=read.face_attributes,
        material=read.material,
    )


def save_mesh(mesh: Trimesh, path: str | Path) -> None:
    """
    Write a [`Trimesh`][ordito.mesh.Trimesh] to a file, with the visuals its format can hold.

    The format follows the suffix of ``path``. Vertex colours, face colours, UVs and the material
    are written where the format holds them: OBJ writes UVs, vertex colours and the material
    (an MTL file and a PNG image beside it, named after ``path`` and the material); PLY writes
    UVs, vertex colours and face colours; glTF / GLB writes UVs, vertex colours and the material
    inside the file. A visual the format cannot hold is left out with a warning.

    Parameters
    ----------
    mesh
        Mesh to write.
    path
        Destination file. Its suffix picks the format: any format ``trimesh`` writes.

    Raises
    ------
    ImportError
        If ``trimesh`` (or, to write an image, ``Pillow``) is not installed.

    Warns
    -----
    UserWarning
        When a visual of ``mesh`` is not written: one the format cannot hold, the colours of a
        mesh whose texture is written (a file holds one or the other), the face colours of a mesh
        whose vertex colours are written, or a material with no UVs to read it.

    See Also
    --------
    [`load_trimesh`][ordito.io.load_trimesh]
        The inverse.
    [`trimesh.Trimesh.export`][]

    Notes
    -----
    Colours are written as bytes, rounded from ``[0, 1]``; OBJ drops their alpha. A material's
    image is written as 8-bit RGBA. OBJ stores the material's factor as ``Kd``, which
    [`load_trimesh`][ordito.io.load_trimesh] reads only when there is no image.

    Examples
    --------
    A textured square written to OBJ, its MTL and image beside it:

    ```python
    import pathlib
    import tempfile

    import numpy as np
    import warp as wp

    import ordito as od

    vertices, faces = od.creation.grid(count=(2, 2), center=False)
    texels = np.random.default_rng(0).random((4, 4, 4), dtype=np.float32)
    mesh = od.Trimesh(
        vertices,
        faces,
        vertex_attributes={"uv": wp.array(vertices.numpy()[:, :2], dtype=wp.vec2)},
        material=od.texture.Material(image=wp.array(texels, dtype=wp.float32), name="skin"),
    )
    with tempfile.TemporaryDirectory() as folder:
        od.io.save_mesh(mesh, pathlib.Path(folder) / "square.obj")
        print(sorted(path.name for path in pathlib.Path(folder).iterdir()))
    # ['skin.png', 'square.mtl', 'square.obj']
    ```
    """
    tm = _import_trimesh()
    path = Path(path)
    suffix = path.suffix.lower()
    holds = _FORMAT_VISUALS.get(suffix, frozenset())
    # Readbacks: trimesh writes from host arrays, so the whole mesh comes back once.
    vertices_np = read_array(mesh.vertices)
    faces_np = read_array(mesh.faces).reshape(-1, 3)
    visual = _export_visual(mesh, holds, suffix)
    mesh_tm = tm.Trimesh(vertices_np, faces_np, visual=visual, process=False)
    options: dict[str, Any] = {"mtl_name": f"{path.stem}.mtl"} if suffix == ".obj" else {}
    mesh_tm.export(path, **options)


def _export_visual(mesh: Trimesh, holds: frozenset[str], suffix: str) -> object:
    """
    Build the trimesh visual of what ``mesh`` writes to a format holding ``holds``, warning on loss.

    One visual a file: a texture (UVs and the material) when the mesh has UVs the format holds,
    else colours, vertex colours before face colours.
    """
    from trimesh.visual import ColorVisuals, TextureVisuals

    uv, material = mesh.uv, mesh.material
    vertex_colors, face_colors = mesh.vertex_colors, mesh.face_colors
    lost: list[str] = []
    visual: object = None
    if uv is not None and "uv" in holds:
        texture = None
        if material is not None:
            if "texture" in holds:
                texture = _material_to_trimesh(material)
            else:
                lost.append("the material")
        # Readback: trimesh writes from host arrays.
        visual = TextureVisuals(uv=read_array(uv), material=texture)
        if vertex_colors is not None or face_colors is not None:
            lost.append("the colours, beside a texture")
    else:
        if uv is not None:
            lost.append("the UVs")
        if material is not None:
            lost.append("the material" if uv is not None else "the material, which has no UVs")
        colors: dict[str, np.ndarray] = {}
        if vertex_colors is not None:
            if "vertex" in holds:
                colors["vertex_colors"] = _color_bytes(vertex_colors)
            else:
                lost.append("the vertex colours")
        if face_colors is not None:
            if "face" in holds and not colors:
                colors["face_colors"] = _color_bytes(face_colors)
            else:
                lost.append(
                    "the face colours, beside vertex colours"
                    if "face" in holds and colors
                    else "the face colours"
                )
        if colors:
            visual = ColorVisuals(**colors)
    if lost:
        warnings.warn(
            f"not written to the {suffix or 'suffix-less'} file: {'; '.join(lost)}",
            UserWarning,
            stacklevel=3,
        )
    return visual


def _material_to_trimesh(material: Material) -> PBRMaterial:
    """Return ``material`` as trimesh's base-colour material, its image as 8-bit RGBA."""
    from PIL import Image
    from trimesh.visual.material import PBRMaterial

    image = None
    if material.image is not None:
        # Readback: the image is written from the host.
        image = Image.fromarray(_byte_values(read_array(material.image)), mode="RGBA")
    factor = _byte_values(np.array([float(c) for c in material.base_color]))
    return PBRMaterial(name=material.name, baseColorTexture=image, baseColorFactor=factor)


def _color_bytes(colors: wp.array[wp.vec4]) -> np.ndarray:
    """Read float RGBA colours back as bytes. Readback: trimesh writes from host arrays."""
    return _byte_values(read_array(colors))


def _byte_values(values: np.ndarray) -> np.ndarray:
    """Round values in ``[0, 1]`` to bytes, clamping those outside."""
    return np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


def load_mesh_data(path: str | Path, *, device: wp.DeviceLike = None) -> MeshData:
    """
    Load a mesh file's vertices, faces, colours and UVs into Warp arrays.

    Reads what [`load_trimesh`][ordito.io.load_trimesh] reads, as a dictionary whose keys are
    present only when the file carries that attribute; ``vertices`` is always present.

    Parameters
    ----------
    path
        Path to a mesh file in any format ``trimesh`` reads.
    device
        Warp device for the returned arrays. Defaults to the current Warp device.

    Returns
    -------
    MeshData
        Mapping with these possible keys:

        - ``vertices`` : ``(n_vertices,)`` ``wp.vec3`` positions (always present).
        - ``faces`` : ``(3 * n_faces,)`` ``wp.int32`` triangle indices, polygons triangulated.
        - ``uv`` : ``(n_vertices,)`` ``wp.vec2`` texture coordinates, a vertex for each texcoord
          of a seam.
        - ``colors`` : ``(n_vertices,)`` ``wp.vec4`` RGBA vertex colours in ``[0, 1]``.
        - ``face_colors`` : ``(n_faces,)`` ``wp.vec4`` RGBA face colours in ``[0, 1]``.

    Raises
    ------
    ImportError
        If ``trimesh`` is not installed.
    ValueError
        If a face references a vertex index the file's own vertex count does not cover.
    IndexError
        If an OBJ file's face does: trimesh's OBJ reader refuses it first.

    See Also
    --------
    [`load_trimesh`][ordito.io.load_trimesh]
    [`load_mesh`][ordito.io.load_mesh]

    Examples
    --------
    A box written with UVs reads back with them:

    ```python
    import pathlib
    import tempfile

    import warp as wp

    import ordito as od

    vertices, faces = od.creation.box()
    uv = wp.array(vertices.numpy()[:, :2] + 0.5, dtype=wp.vec2)
    mesh = od.Trimesh(vertices, faces, vertex_attributes={"uv": uv})
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "box.ply"
        od.io.save_mesh(mesh, path)
        data = od.io.load_mesh_data(path)
    print(sorted(data))  # ['faces', 'uv', 'vertices']
    print(data["faces"].shape, data["uv"].shape)  # (36,) (8,)
    ```
    """
    read = _read(path, "load_mesh_data", device)
    result: MeshData = {"vertices": read.vertices}
    if read.faces.size > 0:
        result["faces"] = read.faces
    if (uv := read.vertex_attributes.get("uv")) is not None:
        result["uv"] = uv
    if (colors := read.vertex_attributes.get("color")) is not None:
        result["colors"] = colors
    if (face_colors := read.face_attributes.get("color")) is not None:
        result["face_colors"] = face_colors
    return result


def load_mesh(path: str | Path, *, device: wp.DeviceLike = None) -> wp.Mesh:
    """
    Load a triangle mesh file into a ``warp.Mesh``.

    Reads only the vertices and triangles (the subset
    [`load_mesh_data`][ordito.io.load_mesh_data] returns) and wraps them in a ``warp.Mesh`` that
    owns its buffers.

    Parameters
    ----------
    path
        Path to a mesh file in any format ``trimesh`` reads.
    device
        Warp device for the returned mesh. Defaults to the current Warp device.

    Returns
    -------
    warp.Mesh
        Mesh with ``vec3`` points and flat ``int32`` indices.

    Raises
    ------
    ImportError
        If ``trimesh`` is not installed.
    ValueError
        If the file contains no triangle faces (a point cloud, say) or a face references a vertex
        index the file's own vertex count does not cover.
    IndexError
        If an OBJ file's face references a vertex it does not define: trimesh's OBJ reader refuses
        it first.

    See Also
    --------
    [`load_mesh_data`][ordito.io.load_mesh_data]

    Examples
    --------
    Round-trip an icosphere through an OBJ file:

    ```python
    import pathlib
    import tempfile

    import ordito as od

    mesh = od.Trimesh(*od.creation.icosphere(subdivisions=1))
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "sphere.obj"
        od.io.save_mesh(mesh, path)
        loaded = od.io.load_mesh(path)
    print(loaded.points.shape, loaded.indices.shape)  # (42,) (240,)
    ```
    """
    read = _read(path, "load_mesh", device)
    if read.faces.size == 0:
        raise ValueError(f"Mesh file {str(path)!r} has no triangle faces; cannot build a wp.Mesh.")
    # Both buffers were allocated by the read and nothing else holds them, so the mesh can own
    # them directly.
    return wp.Mesh(points=read.vertices, indices=read.faces)


class _Read(NamedTuple):
    """A mesh file as ordito holds it: buffers, attribute tables and material."""

    vertices: wp.array[wp.vec3]
    faces: wp.array[wp.int32]
    vertex_attributes: dict[str, odt.ArrayNd]
    face_attributes: dict[str, odt.ArrayNd]
    material: Material | None


def _read(path: str | Path, name: str, device: wp.DeviceLike) -> _Read:
    """Read ``path`` with trimesh into device buffers, its visuals as attributes and a material."""
    tm = _import_trimesh()
    loaded = tm.load(path, process=False, force="mesh")
    points_np = np.asarray(loaded.vertices)
    faces_np = np.asarray(loaded.faces).reshape(-1, 3)
    # A malformed file can hand back a face index the vertex buffer does not cover; this is the
    # trust boundary where that first enters ordito. The check reads trimesh's own integers before
    # the ``int32`` narrowing, which would wrap an index at or above ``2**31`` into range.
    _require_valid_faces_host(faces_np, len(points_np), name)
    vertices = _launch.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces = _launch.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    vertex_attributes: dict[str, odt.ArrayNd] = {}
    face_attributes: dict[str, odt.ArrayNd] = {}
    material = None
    visual = loaded.visual
    kind = None if visual is None else visual.kind
    if visual is not None and kind == "vertex":
        vertex_attributes["color"] = _float_colors(visual.vertex_colors, device)
    elif visual is not None and kind == "face":
        face_attributes["color"] = _float_colors(visual.face_colors, device)
    elif visual is not None and kind == "texture":
        uv_np = getattr(visual, "uv", None)
        if uv_np is not None:
            vertex_attributes["uv"] = _launch.array(
                np.ascontiguousarray(uv_np, dtype=np.float32), dtype=wp.vec2, device=device
            )
        material = _material_from_trimesh(getattr(visual, "material", None), device)
    return _Read(vertices, faces, vertex_attributes, face_attributes, material)


def _float_colors(colors: np.ndarray, device: wp.DeviceLike) -> wp.array[wp.vec4]:
    """Upload trimesh's ``uint8`` RGBA colours as ``wp.vec4`` in ``[0, 1]``."""
    rgba = np.ascontiguousarray(np.asarray(colors, dtype=np.float32)[:, :4] / 255.0)
    return _launch.array(rgba, dtype=wp.vec4, device=device)


def _material_from_trimesh(material: object, device: wp.DeviceLike) -> Material | None:
    """
    Return trimesh's material as a [`Material`][ordito.texture.Material]: its base colour.

    ``None`` for no material, and for the placeholder trimesh invents for UVs read with no image
    (``trimesh.visual.material.empty_material``: a 2x2 image of one grey), which a file never
    stored. A material of another kind than a simple or PBR one is left out with a warning.
    """
    from trimesh.visual.material import PBRMaterial, SimpleMaterial, color_image

    if material is None:
        return None
    if isinstance(material, PBRMaterial):
        image, factor = material.baseColorTexture, material.baseColorFactor
    elif isinstance(material, SimpleMaterial):
        image = material.image
        if image is not None and np.array_equal(np.asarray(image), np.asarray(color_image())):
            return None
        # The diffuse colour beside an image is not applied, as trimesh's own ``to_color`` does
        # not: trimesh gives every image-less MTL a grey ``Kd`` it would darken the image by.
        factor = None if image is not None else material.diffuse
    else:
        warnings.warn(
            f"a {type(material).__name__} is not read: the mesh has no material",
            UserWarning,
            stacklevel=4,
        )
        return None
    base_color = wp.vec4(1.0, 1.0, 1.0, 1.0)
    if factor is not None:
        factor_np = np.asarray(factor, dtype=np.float32).reshape(-1)
        if np.asarray(factor).dtype.kind in "iu":
            factor_np = factor_np / 255.0
        rgba = np.ones(4, dtype=np.float32)
        rgba[: factor_np.size] = factor_np[:4]
        base_color = wp.vec4(*rgba.tolist())
    texture = None
    if image is not None:
        texels = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
        texture = odt.as_array3d(
            _launch.array(np.ascontiguousarray(texels), dtype=wp.float32, device=device), wp.float32
        )
    name = getattr(material, "name", None) or "material"
    return Material(image=texture, base_color=base_color, name=str(name))


def _import_trimesh() -> Any:
    """Import ``trimesh``, the optional dependency every reader and writer here goes through."""
    try:
        import trimesh as tm
    except ImportError as exc:
        raise ImportError(
            "ordito.io requires trimesh. Install it with `pip install ordito[io]`."
        ) from exc
    return tm


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
