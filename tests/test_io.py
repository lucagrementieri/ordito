from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import open3d as o3d
import pytest
import pytorch3d.io as p3d_io
import pyvista as pv
import warp as wp
from PIL import Image

import ordito as od
import ordito.typing as odt

# A square of two triangles whose shared edge is a UV seam: vertex 2 (``1 0 0``) takes texcoords 2
# and 4, so a reader splitting seams gives it two vertices. ``skin.png`` is written beside it.
_SEAM_OBJ = (
    "mtllib square.mtl\n"
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
    "vt 0 0\nvt 1 0\nvt 0 1\nvt 0.5 0.5\nvt 0.9 0.9\n"
    "usemtl skin\n"
    "f 1/1 2/2 3/3\nf 2/4 4/5 3/3\n"
)
_SEAM_MTL = "newmtl skin\nKd 1 1 1\nmap_Kd skin.png\n"


def _texels_np(rng: np.random.Generator, channels: int) -> npt.NDArray[np.uint8]:
    """Return a random ``(4, 8, channels)`` ``uint8`` image: not square, so a transpose shows."""
    return rng.integers(0, 256, size=(4, 8, channels), dtype=np.uint8)


def _textured_grid(device: str) -> od.Trimesh:
    """Return a 3x3 grid with UVs inside the unit square and a named RGBA material."""
    vertices_wp, faces_wp = od.creation.grid(count=(3, 3), center=False, device=device)
    uv_np = vertices_wp.numpy()[:, :2] * 0.8 + 0.1
    texels_np = _texels_np(np.random.default_rng(0), 4).astype(np.float32) / 255.0
    image_wp = wp.array(texels_np, dtype=wp.float32, device=device)
    material = od.texture.Material(
        image=odt.as_array3d(image_wp, wp.float32),
        base_color=wp.vec4(1.0, 0.5, 0.25, 1.0),
        name="skin",
    )
    return od.Trimesh(
        vertices_wp,
        faces_wp,
        vertex_attributes={"uv": wp.array(uv_np, dtype=wp.vec2, device=device)},
        material=material,
    )


def _colored_grid(device: str, element: str) -> od.Trimesh:
    """Return a 3x3 grid with random RGBA colours on its vertices or its faces."""
    vertices_wp, faces_wp = od.creation.grid(count=(3, 3), center=False, device=device)
    count = vertices_wp.size if element == "vertex" else faces_wp.size // 3
    colors_np = np.random.default_rng(1).random((count, 4)).astype(np.float32)
    colors_wp = wp.array(colors_np, dtype=wp.vec4, device=device)
    if element == "vertex":
        return od.Trimesh(vertices_wp, faces_wp, vertex_attributes={"color": colors_wp})
    return od.Trimesh(vertices_wp, faces_wp, face_attributes={"color": colors_wp})


def _corners_np(mesh: od.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Return each face corner's position and UV, ``(3 * n_faces, 3)`` and ``(3 * n_faces, 2)``."""
    assert mesh.uv is not None
    faces_np = mesh.faces.numpy()
    return mesh.vertices.numpy()[faces_np], mesh.uv.numpy()[faces_np]


def test_load_trimesh_reads_a_textured_obj_like_pytorch3d(tmp_path: Path, device: str) -> None:
    """
    Class A against ``pytorch3d.io.load_obj``: corner positions, corner UVs and the texture.

    pytorch3d keeps the file's vertices and indexes texcoords per corner; ordito, as trimesh, gives
    each texcoord of a seam its own vertex. The two agree corner for corner, which is free of the
    numbering, and on the image (pytorch3d reads RGB in ``[0, 1]``; the alpha the PNG lacks reads
    1). The seam is real: five vertices, not four. ``Kd`` beside the image is left out, so the
    factor is white.
    """
    rng = np.random.default_rng(0)
    texels_np = _texels_np(rng, 3)
    Image.fromarray(texels_np).save(tmp_path / "skin.png")
    (tmp_path / "square.mtl").write_text(_SEAM_MTL)
    path = tmp_path / "square.obj"
    path.write_text(_SEAM_OBJ)

    mesh = od.io.load_trimesh(path, device=device)
    verts_p3d, faces_p3d, aux_p3d = p3d_io.load_obj(str(path), load_textures=True)
    assert aux_p3d.verts_uvs is not None
    assert aux_p3d.texture_images is not None

    assert mesh.n_vertices == 5
    assert mesh.n_faces == 2
    positions_np, uv_np = _corners_np(mesh)
    positions_p3d = verts_p3d.numpy()[faces_p3d.verts_idx.numpy().reshape(-1)]
    uv_p3d = aux_p3d.verts_uvs.numpy()[faces_p3d.textures_idx.numpy().reshape(-1)]
    assert np.allclose(positions_np, positions_p3d, rtol=1e-6, atol=1e-6)
    assert np.allclose(uv_np, uv_p3d, rtol=1e-6, atol=1e-6)

    assert mesh.material is not None
    assert mesh.material.image is not None
    assert mesh.material.name == "skin"
    assert list(mesh.material.base_color) == [1.0, 1.0, 1.0, 1.0]
    image_np = mesh.material.image.numpy()
    assert np.allclose(image_np[..., :3], aux_p3d.texture_images["skin"].numpy(), atol=1e-6)
    assert np.array_equal(image_np[..., 3], np.ones(texels_np.shape[:2], dtype=np.float32))


def test_load_mesh_triangulates_polygons_like_pytorch3d(tmp_path: Path, device: str) -> None:
    """
    Class A against ``pytorch3d.io.load_obj``: a quad is split into two triangles, as there.

    The two pick the same diagonal and differ only in where each triangle starts, which
    [`faces_equal`][ordito.equality.faces_equal] reads past.
    """
    path = tmp_path / "quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 0\nf 1 2 3 4\nf 2 5 3\n")
    mesh_wp = od.io.load_mesh(path, device=device)
    _verts_p3d, faces_p3d, _aux_p3d = p3d_io.load_obj(str(path), load_textures=False)
    faces_p3d_wp = wp.array(
        faces_p3d.verts_idx.numpy().reshape(-1).astype(np.int32), dtype=wp.int32, device=device
    )
    assert faces_p3d_wp.size == 9
    assert od.equality.faces_equal(mesh_wp.indices, faces_p3d_wp)


def test_save_mesh_writes_a_textured_obj_pytorch3d_reads(tmp_path: Path, device: str) -> None:
    """
    Class A against ``pytorch3d.io.load_obj``: what ordito writes, an independent reader reads back.

    Corner positions and UVs exactly; the texture's RGB to the byte the PNG stores, under the
    material's own name. The factor is written as ``Kd`` (read back with the next test's GLB).
    """
    mesh = _textured_grid(device)
    path = tmp_path / "grid.obj"
    od.io.save_mesh(mesh, path)
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [
        "grid.mtl",
        "grid.obj",
        "skin.png",
    ]

    verts_p3d, faces_p3d, aux_p3d = p3d_io.load_obj(str(path), load_textures=True)
    assert aux_p3d.verts_uvs is not None
    assert aux_p3d.texture_images is not None
    positions_np, uv_np = _corners_np(mesh)
    positions_p3d = verts_p3d.numpy()[faces_p3d.verts_idx.numpy().reshape(-1)]
    uv_p3d = aux_p3d.verts_uvs.numpy()[faces_p3d.textures_idx.numpy().reshape(-1)]
    assert np.allclose(positions_np, positions_p3d, rtol=1e-6, atol=1e-6)
    assert np.allclose(uv_np, uv_p3d, rtol=1e-6, atol=1e-6)
    assert mesh.material is not None
    assert mesh.material.image is not None
    image_np = mesh.material.image.numpy()
    assert np.ptp(image_np) > 0.5
    assert np.allclose(aux_p3d.texture_images["skin"].numpy(), image_np[..., :3], atol=0.5 / 255.0)


def test_save_mesh_writes_a_textured_glb_open3d_reads(tmp_path: Path, device: str) -> None:
    """
    Class B against open3d's glTF reader: corner UVs, and the RGBA texture flipped upside down.

    open3d reads a corner UV per face corner and the base-colour texture as an RGBA image, at the
    index its triangles' material ids name. It stores that image bottom row first (OpenGL's order,
    ``row = v * H``) where ordito and the PNG inside the file store the top row first, so the
    transform is a vertical flip; unflipped, the random texels disagree everywhere. The UVs are
    compared as they are.
    """
    mesh = _textured_grid(device)
    path = tmp_path / "grid.glb"
    od.io.save_mesh(mesh, path)

    mesh_o3d = o3d.io.read_triangle_mesh(path, enable_post_processing=False)
    triangles_o3d = np.asarray(mesh_o3d.triangles)
    positions_np, uv_np = _corners_np(mesh)
    positions_o3d = np.asarray(mesh_o3d.vertices)[triangles_o3d.reshape(-1)]
    assert np.allclose(positions_np, positions_o3d, rtol=1e-6, atol=1e-6)
    assert np.allclose(uv_np, np.asarray(mesh_o3d.triangle_uvs), rtol=1e-6, atol=1e-6)
    assert mesh.material is not None
    assert mesh.material.image is not None
    material_ids_o3d = np.unique(np.asarray(mesh_o3d.triangle_material_ids))
    assert material_ids_o3d.size == 1
    texture_o3d = np.asarray(mesh_o3d.textures[int(material_ids_o3d[0])])
    assert texture_o3d.shape == (4, 8, 4)
    assert np.allclose(texture_o3d[::-1] / 255.0, mesh.material.image.numpy(), atol=0.5 / 255.0)


@pytest.mark.parametrize("suffix", [".obj", ".glb"])
def test_textured_mesh_round_trips(tmp_path: Path, device: str, suffix: str) -> None:
    """
    Ordito against ordito: ``load_trimesh`` reads back what ``save_mesh`` wrote, textures included.

    Its readers are compared with independent ones above; this pins the pair to each other: UVs,
    the material's name and image to the byte, and its factor where the format keeps it beside an
    image (glTF; OBJ's ``Kd`` is read only with no image, so it comes back white).
    """
    mesh = _textured_grid(device)
    path = tmp_path / f"grid{suffix}"
    od.io.save_mesh(mesh, path)
    loaded = od.io.load_trimesh(path, device=device)

    assert loaded.n_vertices == mesh.n_vertices
    assert loaded.n_faces == mesh.n_faces
    assert np.array_equal(loaded.faces.numpy(), mesh.faces.numpy())
    assert loaded.uv is not None
    assert mesh.uv is not None
    assert np.allclose(loaded.uv.numpy(), mesh.uv.numpy(), rtol=1e-6, atol=1e-6)
    assert loaded.material is not None
    assert mesh.material is not None
    assert loaded.material.image is not None
    assert mesh.material.image is not None
    assert loaded.material.name == "skin"
    assert np.allclose(loaded.material.image.numpy(), mesh.material.image.numpy(), atol=0.5 / 255.0)
    factor_np = [1.0, 0.5, 0.25, 1.0] if suffix == ".glb" else [1.0, 1.0, 1.0, 1.0]
    assert np.allclose(list(loaded.material.base_color), factor_np, atol=0.5 / 255.0)


def _write_colored_ply(path: Path) -> dict[str, np.ndarray]:
    """
    Write an ASCII square with RGBA vertex colours, and siblings with face colours and with UVs.

    Returns the values written, by the key ``load_mesh_data`` reads them under.
    """
    rng = np.random.default_rng(2)
    vertices_np = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    faces_np = np.array([[0, 1, 2], [1, 3, 2]])
    colors_np = rng.integers(0, 256, size=(4, 4), dtype=np.uint8)
    face_colors_np = rng.integers(0, 256, size=(2, 4), dtype=np.uint8)
    uv_np = rng.random((4, 2)).astype(np.float32)
    position = "property float x\nproperty float y\nproperty float z\n"
    rgba = "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"

    def write(
        name: str, vertex_props: str, vertex_rows: list[str], face_props: str, face_rows: list[str]
    ) -> None:
        header = (
            f"ply\nformat ascii 1.0\nelement vertex 4\n{position}{vertex_props}"
            f"element face 2\nproperty list uchar int vertex_indices\n{face_props}end_header\n"
        )
        (path.parent / name).write_text(header + "\n".join(vertex_rows + face_rows) + "\n")

    def row(*parts: np.ndarray) -> str:
        return " ".join(str(value) for part in parts for value in part.tolist())

    plain_faces = [row(np.array([3]), face) for face in faces_np]
    write(
        "vertex.ply",
        rgba,
        [row(v, c) for v, c in zip(vertices_np, colors_np, strict=True)],
        "",
        plain_faces,
    )
    write(
        "face.ply",
        "",
        [row(v) for v in vertices_np],
        rgba,
        [row(np.array([3]), face, c) for face, c in zip(faces_np, face_colors_np, strict=True)],
    )
    write(
        "uv.ply",
        "property float s\nproperty float t\n",
        [row(v, t) for v, t in zip(vertices_np, uv_np, strict=True)],
        "",
        plain_faces,
    )
    return {
        "vertices": vertices_np,
        "faces": faces_np.reshape(-1),
        "colors": colors_np,
        "face_colors": face_colors_np,
        "uv": uv_np,
    }


def test_load_mesh_data_reads_ply_visuals_like_pyvista(tmp_path: Path, device: str) -> None:
    """
    Class A against VTK's PLY reader: vertex colours, face colours and UVs, as written.

    pyvista reads colours as ``RGBA`` bytes on the points or cells and UVs as ``TCoords``; ordito
    returns them in ``[0, 1]``, so the comparison divides by 255 (exact in ``float32``). Each
    visual's file reads that key alone, and a file with UVs and no image reads no material.
    """
    written = _write_colored_ply(tmp_path / "unused.ply")

    vertex = od.io.load_mesh_data(tmp_path / "vertex.ply", device=device)
    assert sorted(vertex) == ["colors", "faces", "vertices"]
    assert "faces" in vertex
    assert "colors" in vertex
    vertex_pv = pv.read(tmp_path / "vertex.ply")
    assert np.allclose(vertex["vertices"].numpy(), np.asarray(vertex_pv.points))
    assert np.array_equal(vertex["faces"].numpy(), written["faces"])
    assert np.allclose(vertex["colors"].numpy(), np.asarray(vertex_pv.point_data["RGBA"]) / 255.0)

    face = od.io.load_mesh_data(tmp_path / "face.ply", device=device)
    assert sorted(face) == ["face_colors", "faces", "vertices"]
    assert "face_colors" in face
    face_pv = pv.read(tmp_path / "face.ply")
    assert np.allclose(face["face_colors"].numpy(), np.asarray(face_pv.cell_data["RGBA"]) / 255.0)

    uv = od.io.load_mesh_data(tmp_path / "uv.ply", device=device)
    assert sorted(uv) == ["faces", "uv", "vertices"]
    assert "uv" in uv
    uv_pv = pv.read(tmp_path / "uv.ply")
    assert np.allclose(uv["uv"].numpy(), np.asarray(uv_pv.point_data["TCoords"]), atol=1e-7)
    assert od.io.load_trimesh(tmp_path / "uv.ply", device=device).material is None


@pytest.mark.parametrize("element", ["vertex", "face"])
def test_save_mesh_writes_ply_colours_pyvista_reads(
    tmp_path: Path, device: str, element: str
) -> None:
    """
    Class A against VTK's PLY reader: colours written as the rounded bytes of their values.

    And ordito reads the same bytes back (ordito against ordito, to half a byte of the values).
    """
    mesh = _colored_grid(device, element)
    path = tmp_path / "grid.ply"
    od.io.save_mesh(mesh, path)
    mesh_pv = pv.read(path)
    colors = mesh.vertex_colors if element == "vertex" else mesh.face_colors
    assert colors is not None
    colors_np = colors.numpy()
    data_pv = mesh_pv.point_data if element == "vertex" else mesh_pv.cell_data
    assert np.array_equal(np.asarray(data_pv["RGBA"]), np.round(colors_np * 255.0))

    loaded = od.io.load_trimesh(path, device=device)
    loaded_colors = loaded.vertex_colors if element == "vertex" else loaded.face_colors
    assert loaded_colors is not None
    assert np.allclose(loaded_colors.numpy(), colors_np, atol=0.5 / 255.0)


@pytest.mark.parametrize(
    ("suffix", "mesh_kind", "lost"),
    [
        pytest.param(".ply", "texture", "the material", id="ply-image"),
        pytest.param(".off", "texture", "the UVs; the material", id="off-texture"),
        pytest.param(".obj", "face", "the face colours", id="obj-face-colours"),
        pytest.param(".stl", "vertex", "the vertex colours", id="stl-colours"),
        pytest.param(".obj", "both", "the colours, beside a texture", id="colours-beside-texture"),
        pytest.param(".ply", "material-only", "the material, which has no UVs", id="no-uv"),
    ],
)
def test_save_mesh_warns_on_what_the_format_cannot_hold(
    tmp_path: Path, device: str, suffix: str, mesh_kind: str, lost: str
) -> None:
    """
    Not a library comparison: a visual the file cannot hold is named in a warning, not dropped mute.

    Which visual each format keeps was measured by writing and reading each one back with trimesh
    (``io._FORMAT_VISUALS``); the cases above are one loss of each kind.
    """
    if mesh_kind == "texture":
        mesh = _textured_grid(device)
    elif mesh_kind == "both":
        colored = _colored_grid(device, "vertex")
        mesh = _textured_grid(device).with_attributes(
            vertex_attributes={"color": colored.vertex_colors}
        )
    elif mesh_kind == "material-only":
        mesh = _textured_grid(device).with_attributes(vertex_attributes={"uv": None})
    else:
        mesh = _colored_grid(device, mesh_kind)
    with pytest.warns(UserWarning, match=f"not written to the \\{suffix} file: {lost}$"):
        od.io.save_mesh(mesh, tmp_path / f"grid{suffix}")


@pytest.mark.parametrize("suffix", [".obj", ".ply", ".glb"])
def test_save_mesh_is_silent_when_everything_is_written(
    tmp_path: Path, device: str, suffix: str
) -> None:
    """Not a library comparison: a mesh whose visuals the format holds writes with no warning."""
    mesh = _colored_grid(device, "vertex") if suffix == ".ply" else _textured_grid(device)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        od.io.save_mesh(mesh, tmp_path / f"grid{suffix}")


def _write_ply(path: Path, vertices_np: np.ndarray, faces_np: np.ndarray, index_type: str) -> None:
    """Write a plain ASCII PLY whose face indices have ``index_type``."""
    header = (
        f"ply\nformat ascii 1.0\nelement vertex {len(vertices_np)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        f"element face {len(faces_np)}\nproperty list uchar {index_type} vertex_indices\n"
        "end_header\n"
    )
    rows = [" ".join(repr(float(x)) for x in vertex) for vertex in vertices_np]
    rows += [" ".join(str(int(i)) for i in [3, *face]) for face in faces_np]
    path.write_text(header + "\n".join(rows) + "\n")


@pytest.mark.parity(
    "load_mesh",
    "pytorch3d",
    benchmarked=False,
    reason="ordito.io has no benchmark group at all and deliberately so -- it is a trimesh "
    "round-trip, so a row here would time trimesh's PLY parser against pytorch3d's, neither of "
    "which is ordito code. tests/api_conventions.py carries io in "
    "_MODULES_WITHOUT_BENCHMARKS for that reason. The values are still worth comparing, which is "
    "what this test does.",
)
def test_load_mesh_matches_pytorch3d(tmp_path: Path, device: str) -> None:
    """
    Class A: ``pytorch3d.io.load_ply`` reads the same file to the same buffers, bit for bit.

    Both sides parse the *same* PLY, so the only thing between them is storage: pytorch3d returns
    float32 verts and int64 faces where trimesh hands ordito float64 that ``wp.vec3`` narrows, and
    the faces are byte-equal with no transform. A second independent parser is worth having on a
    loader precisely because a format bug is invisible to a round trip through the writer that
    produced the file.
    """
    vertices_np = np.random.default_rng(3).random((5, 3))
    faces_np = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 4]])
    path = tmp_path / "mesh.ply"
    _write_ply(path, vertices_np, faces_np, "int")
    vertices_p3d, faces_p3d = p3d_io.load_ply(str(path))
    mesh_wp = od.io.load_mesh(path, device=device)

    assert isinstance(mesh_wp, wp.Mesh)
    assert np.allclose(mesh_wp.points.numpy(), vertices_np, rtol=1e-5, atol=1e-5)
    assert np.array_equal(mesh_wp.points.numpy(), vertices_p3d.numpy())
    assert np.array_equal(mesh_wp.indices.numpy().reshape(-1, 3), faces_p3d.numpy())


def test_load_mesh_rejects_a_point_cloud(tmp_path: Path, device: str) -> None:
    """
    Not a library comparison: a file of points alone has no triangle to build a ``wp.Mesh`` from.

    ``load_mesh_data`` reads it as its vertices alone.
    """
    path = tmp_path / "cloud.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n0 0 0\n1 0 0\n0 1 0\n"
    )
    with pytest.raises(ValueError, match="has no triangle faces"):
        od.io.load_mesh(path, device=device)
    assert sorted(od.io.load_mesh_data(path, device=device)) == ["vertices"]


@pytest.mark.parametrize(
    ("bad_index", "index_type"),
    [
        pytest.param(3, "int", id="past-the-end"),
        pytest.param(-1, "int", id="negative"),
        pytest.param(2**32 - 1, "uint", id="wraps-to-minus-one-in-int32"),
    ],
)
def test_load_mesh_rejects_an_out_of_range_index(
    tmp_path: Path, device: str, bad_index: int, index_type: str
) -> None:
    """
    Not a library comparison: an input-validation contract no reference library checks.

    ``2**32 - 1`` reads as ``-1`` once narrowed to ``int32``, so a device-side check would see a
    negative index and the plain bounds a valid one; the host check reads trimesh's own integers
    first.
    """
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    path = tmp_path / "bad.ply"
    _write_ply(path, vertices_np, np.array([[0, 1, 2], [0, 1, bad_index]]), index_type)
    with pytest.raises(ValueError, match="faces must reference vertex indices"):
        od.io.load_mesh(path, device=device)


@pytest.mark.parametrize("load", [od.io.load_mesh, od.io.load_mesh_data, od.io.load_trimesh])
def test_obj_reader_rejects_an_out_of_range_index(
    tmp_path: Path, device: str, load: Callable[..., object]
) -> None:
    """Not a library comparison: trimesh's OBJ reader refuses a face naming an undefined vertex."""
    path = tmp_path / "bad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 5\n")
    with pytest.raises(IndexError):
        load(path, device=device)


@pytest.mark.parametrize("bad_index", [2**32, 2**31, 3, -1])
def test_mesh_from_numpy_rejects_out_of_range_indices(bad_index: int, device: str):
    """
    Not a library comparison: an input-validation contract no reference library checks.

    ``2**32`` wraps to ``0`` and ``2**31`` to a negative ``int32``; both must be caught on the host
    before the narrowing, as must the plain ``n_vertices`` and ``-1`` boundaries.
    """
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces_np = np.array([[0, 1, 2], [0, 1, bad_index]], dtype=np.int64)
    with pytest.raises(ValueError, match="faces must reference vertex indices"):
        od.io.mesh_from_numpy(vertices_np, faces_np, device=device)
