from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import meshio
import numpy as np
import pytest
import pytorch3d.io as p3d_io
import trimesh as tm
import warp as wp

import ordito as od

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _write_synthetic_mesh(path: Path) -> dict[str, np.ndarray]:
    """Write a tetrahedron surface with normals, colors and uv; return the source arrays."""
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    faces_np = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    normals_np = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32
    )
    colors_np = np.array([[10, 20, 30], [200, 100, 50], [0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    uv_np = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    point_data_mio: dict[str, ArrayLike] = {
        "nx": normals_np[:, 0],
        "ny": normals_np[:, 1],
        "nz": normals_np[:, 2],
        "red": colors_np[:, 0],
        "green": colors_np[:, 1],
        "blue": colors_np[:, 2],
        "u": uv_np[:, 0],
        "v": uv_np[:, 1],
    }
    mesh_mio = meshio.Mesh(vertices_np, [("triangle", faces_np)], point_data=point_data_mio)
    meshio.write(str(path), mesh_mio)
    return {
        "vertices": vertices_np,
        "faces": faces_np,
        "vertex_normals": normals_np,
        "colors": colors_np,
        "uv": uv_np,
    }


def test_load_mesh_data_roundtrip(tmp_path: Path, device: str):
    path = tmp_path / "mesh.ply"
    source = _write_synthetic_mesh(path)

    data_wp = od.io.load_mesh_data(path, device=device)

    assert "faces" in data_wp
    assert "vertex_normals" in data_wp
    assert "uv" in data_wp
    assert "colors" in data_wp
    assert np.allclose(data_wp["vertices"].numpy(), source["vertices"], rtol=1e-5, atol=1e-5)
    assert np.array_equal(data_wp["faces"].numpy().reshape(-1, 3), source["faces"])
    assert np.allclose(
        data_wp["vertex_normals"].numpy(), source["vertex_normals"], rtol=1e-5, atol=1e-5
    )
    assert np.allclose(data_wp["uv"].numpy(), source["uv"], rtol=1e-5, atol=1e-5)
    assert np.allclose(
        data_wp["colors"].numpy(), source["colors"].astype(np.float32) / 255.0, rtol=1e-5, atol=1e-5
    )
    # meshio never computes face normals and none were written.
    assert "face_normals" not in data_wp


def test_load_mesh_data_reads_obj_corner_texcoords(tmp_path: Path, device: str) -> None:
    """
    Class A against trimesh's OBJ reader: one texcoord per face corner, seams kept.

    A square of two triangles whose shared edge is a UV seam (one vertex has two texcoords), with a
    quad that is ignored. trimesh, loading without merging, splits the seam vertices and reports a
    texcoord per vertex; the corner texcoords here equal its texcoords at its own corners, and the
    positions and faces are the file's.
    """
    path = tmp_path / "square.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\nvt 0.5 0.5\nvt 0.9 0.9\n"
        "f 1/1 2/2 3/3\nf 2/4 4/5 -2/3\nf 1/1 2/2 4/5 3/3\n"
    )
    data = od.io.load_mesh_data(path, device=device)
    assert sorted(data) == ["corner_uv", "faces", "vertices"]
    assert "faces" in data
    assert "corner_uv" in data
    faces_np = data["faces"].numpy()
    assert faces_np.tolist() == [0, 1, 2, 1, 3, 2]
    mesh_tm = tm.load(path, process=False, force="mesh")
    assert isinstance(mesh_tm, tm.Trimesh)
    triangles_tm = mesh_tm.vertices[mesh_tm.faces[:2]].reshape(-1, 3)
    uv_tm = np.asarray(getattr(mesh_tm.visual, "uv"))  # noqa: B009 - TextureVisuals is untyped
    corner_uv_tm = uv_tm[mesh_tm.faces[:2]].reshape(-1, 2)
    assert np.allclose(data["vertices"].numpy()[faces_np], triangles_tm)
    assert np.allclose(data["corner_uv"].numpy(), corner_uv_tm, rtol=1e-6, atol=1e-6)

    bad = tmp_path / "bad.obj"
    bad.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1 2/2 3/1\n")
    with pytest.raises(ValueError, match="texcoord"):
        od.io.load_mesh_data(bad, device=device)


def _write_ply_with_corner_texcoords(path: Path, binary: bool, *, quad: bool = True) -> None:
    """
    Write a square of two triangles, and one quad, whose faces carry a ``texcoord`` list.

    The triangles' shared edge is a UV seam (vertex 1 and vertex 2 take two texcoords each), and
    each vertex carries a ``nx, ny, nz`` normal.
    """
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32
    )
    faces = [[0, 1, 2], [1, 3, 2], [0, 1, 3, 2]]
    texcoords = [
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.9, 0.9, 0.25, 0.75],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    ]
    if not quad:
        faces, texcoords = faces[:2], texcoords[:2]
    header = (
        "ply\n"
        f"format {'binary_little_endian' if binary else 'ascii'} 1.0\n"
        "comment corner texcoords\n"
        "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
        "property list uchar float texcoord\nend_header\n"
    )
    normal = [0.0, 0.0, 1.0]
    if not binary:
        lines = [" ".join(str(x) for x in [*vertex.tolist(), *normal]) for vertex in vertices_np]
        for face, uv in zip(faces, texcoords, strict=True):
            lines.append(" ".join(str(x) for x in [len(face), *face, len(uv), *uv]))
        path.write_text(header + "\n".join(lines) + "\n")
        return
    body = b"".join(
        np.array([*vertex.tolist(), *normal], dtype="<f4").tobytes() for vertex in vertices_np
    )
    for face, uv in zip(faces, texcoords, strict=True):
        body += np.uint8(len(face)).tobytes() + np.array(face, dtype="<i4").tobytes()
        body += np.uint8(len(uv)).tobytes() + np.array(uv, dtype="<f4").tobytes()
    path.write_bytes(header.encode("ascii") + body)


@pytest.mark.parametrize("binary", [False, True], ids=["ascii", "binary"])
def test_load_mesh_data_reads_ply_corner_texcoords(
    tmp_path: Path, device: str, binary: bool
) -> None:
    """
    Class A against trimesh's PLY reader: one texcoord per face corner, seams kept.

    ``meshio`` cannot decode a ``texcoord`` face list, so the file is read by ordito's own PLY
    reader; the quad is ignored as for OBJ. trimesh, loading without merging, splits the seam
    vertices and reports a texcoord per vertex; the corner texcoords here equal its texcoords at
    its own corners, the positions and faces are the file's, and the per-vertex normals survive.
    """
    path = tmp_path / "square.ply"
    _write_ply_with_corner_texcoords(path, binary)
    data = od.io.load_mesh_data(path, device=device)
    assert sorted(data) == ["corner_uv", "faces", "vertex_normals", "vertices"]
    assert "faces" in data
    assert "corner_uv" in data
    faces_np = data["faces"].numpy()
    assert faces_np.tolist() == [0, 1, 2, 1, 3, 2]
    ascii_path = tmp_path / "square_ascii.ply"
    _write_ply_with_corner_texcoords(ascii_path, False, quad=False)  # trimesh mixes no quad
    mesh_tm = tm.load(ascii_path, process=False, force="mesh")
    assert isinstance(mesh_tm, tm.Trimesh)
    triangles_tm = mesh_tm.vertices[mesh_tm.faces].reshape(-1, 3)
    uv_tm = np.asarray(getattr(mesh_tm.visual, "uv"))  # noqa: B009 - TextureVisuals is untyped
    corner_uv_tm = uv_tm[mesh_tm.faces].reshape(-1, 2)
    assert np.unique(corner_uv_tm, axis=0).shape[0] == 6  # four vertices: the seam is real
    assert np.allclose(data["vertices"].numpy()[faces_np], triangles_tm)
    assert np.allclose(data["corner_uv"].numpy(), corner_uv_tm, rtol=1e-6, atol=1e-6)
    assert "vertex_normals" in data
    assert np.allclose(data["vertex_normals"].numpy(), [[0.0, 0.0, 1.0]] * 4)

    mesh_wp = od.io.load_mesh(path, device=device)
    assert mesh_wp.indices.numpy().tolist() == faces_np.tolist()


@pytest.mark.parity(
    "load_mesh",
    "pytorch3d",
    benchmarked=False,
    reason="ordito.io has no benchmark group at all and deliberately so -- it is a meshio "
    "round-trip, so a row here would time meshio's PLY parser against pytorch3d's, neither of "
    "which is ordito code. tests/api_conventions.py carries io in "
    "_MODULES_WITHOUT_BENCHMARKS for that reason. The values are still worth comparing, which is "
    "what this test does.",
)
def test_load_mesh_matches_pytorch3d(tmp_path: Path, device: str):
    """
    Class A: ``pytorch3d.io.load_ply`` reads the same file to the same buffers, bit for bit.

    Both sides parse the *same* PLY, so the only thing between them is storage: pytorch3d returns
    float32 verts and int64 faces where meshio hands ordito float64 that ``wp.vec3`` narrows, and
    the faces are byte-equal with no transform. A second independent parser is worth having on a
    loader precisely because a format bug is invisible to a round trip through the writer that
    produced the file.

    Note ``load_ply`` returns a bare ``(verts, faces)`` tuple, not a ``Meshes`` -- the container
    form is ``IO().load_mesh``, and only the ``.obj`` reader has an ``as_meshes`` helper.

    The returned object is a ``wp.Mesh`` whose points and faces also match the written arrays.
    """
    path = tmp_path / "mesh.ply"
    source = _write_synthetic_mesh(path)
    vertices_p3d, faces_p3d = p3d_io.load_ply(str(path))
    mesh_wp = od.io.load_mesh(path, device=device)

    assert isinstance(mesh_wp, wp.Mesh)
    assert np.allclose(mesh_wp.points.numpy(), source["vertices"], rtol=1e-5, atol=1e-5)
    assert np.array_equal(mesh_wp.indices.numpy().reshape(-1, 3), source["faces"])
    assert vertices_p3d.shape == source["vertices"].shape
    assert np.array_equal(mesh_wp.points.numpy(), vertices_p3d.numpy())
    assert np.array_equal(mesh_wp.indices.numpy().reshape(-1, 3), faces_p3d.numpy())


def test_face_normals_cover_every_triangle_block(monkeypatch: pytest.MonkeyPatch, device: str):
    """
    Ordito against meshio: ``face_normals`` must be row-aligned with ``faces``, block count aside.

    Not a library comparison -- no reference loader exposes ordito's dict -- so the oracle is
    meshio's own ``cell_data_dict[name]["triangle"]``, which concatenates the triangle blocks in the
    same order ``cells_dict["triangle"]`` does. A file that splits its triangles across several
    blocks is ordinary (gmsh and VTK physical groups do it), and reading ``cell_data[name][i]`` for
    a single block index instead returns only the first block's normals: two faces, one normal row,
    no error. The per-block values are made distinct so a short *or* misaligned read fails on value
    and not only on length.

    The mesh is handed to ``load_mesh_data`` through a patched ``meshio.read`` rather than written
    out, because no format round-trips both halves of the fixture: meshio's VTK and VTU writers
    merge the two triangle blocks into one, and its gmsh writer keeps the blocks but drops
    non-standard ``nx``/``ny``/``nz`` cell data entirely (both measured). Patching the reader keeps
    the public entry point under test while still delivering the multi-block mesh a real gmsh file
    would.
    """
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64
    )
    blocks: list[tuple[str, ArrayLike] | meshio.CellBlock] = [
        ("triangle", np.array([[0, 1, 2]])),
        ("triangle", np.array([[1, 3, 2]])),
    ]
    normals_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    cell_data_mio: dict[str, list[ArrayLike]] = {
        name: [normals_np[:1, k], normals_np[1:, k]] for k, name in enumerate(("nx", "ny", "nz"))
    }
    mesh_mio = meshio.Mesh(vertices_np, blocks, cell_data=cell_data_mio)
    assert len(mesh_mio.cells) == 2, "the fixture must split its triangles across two blocks"
    monkeypatch.setattr(meshio, "read", lambda _path: mesh_mio)

    data_wp = od.io.load_mesh_data("unread.msh", device=device)

    faces_mio = mesh_mio.cells_dict["triangle"]
    normals_mio = np.column_stack(
        [mesh_mio.cell_data_dict[name]["triangle"] for name in ("nx", "ny", "nz")]
    )
    assert faces_mio.shape[0] == 2
    assert normals_mio.shape == (2, 3)
    assert "faces" in data_wp
    assert "face_normals" in data_wp
    assert np.array_equal(data_wp["faces"].numpy().reshape(-1, 3), faces_mio)
    assert data_wp["face_normals"].size == faces_mio.shape[0]
    assert np.allclose(data_wp["face_normals"].numpy(), normals_mio, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("filename", "mesh_mio", "match"),
    [
        pytest.param(
            "cloud.ply",
            meshio.Mesh(np.random.default_rng(0).random((10, 3)).astype(np.float64), []),
            "no triangle faces",
            id="no_faces",
        ),
        pytest.param(
            "wrap.obj",
            meshio.Mesh(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                [("triangle", np.array([[0, 1, 2], [0, 1, 2**32]], dtype=np.int64))],
            ),
            "faces must reference vertex indices",
            id="index_wraps_past_int32",
        ),
    ],
)
def test_load_mesh_rejects_an_invalid_file(
    tmp_path: Path, device: str, filename: str, mesh_mio: meshio.Mesh, match: str
):
    """
    Not a library comparison: input-validation contracts no reference library checks.

    A file with no triangle faces is refused, and so is an index of ``2**32``, which wraps to ``0``
    under the ``int32`` narrowing, so a device-side range check would pass it; the host check reads
    the decoder's own integers first.
    """
    path = tmp_path / filename
    meshio.write(str(path), mesh_mio)
    with pytest.raises(ValueError, match=match):
        od.io.load_mesh(path, device=device)


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
