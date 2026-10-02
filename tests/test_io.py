from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import meshio
import numpy as np
import pytest
import pytorch3d.io as p3d_io
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


def test_load_mesh_returns_wp_mesh(tmp_path: Path, device: str):
    path = tmp_path / "mesh.ply"
    source = _write_synthetic_mesh(path)

    mesh_wp = od.io.load_mesh(path, device=device)

    assert isinstance(mesh_wp, wp.Mesh)
    assert np.allclose(mesh_wp.points.numpy(), source["vertices"], rtol=1e-5, atol=1e-5)
    assert np.array_equal(mesh_wp.indices.numpy().reshape(-1, 3), source["faces"])


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
    Class B: ``pytorch3d.io.load_ply`` reads the same file to the same buffers after a float cast.

    Both sides parse the *same* PLY, so the only thing between them is storage: pytorch3d returns
    float32 verts and int64 faces where meshio hands ordito float64 that ``wp.vec3`` narrows, and
    the faces are byte-equal with no transform. A second independent parser is worth having on a
    loader precisely because a format bug is invisible to a round trip through the writer that
    produced the file.

    Note ``load_ply`` returns a bare ``(verts, faces)`` tuple, not a ``Meshes`` -- the container
    form is ``IO().load_mesh``, and only the ``.obj`` reader has an ``as_meshes`` helper.
    """
    path = tmp_path / "mesh.ply"
    source = _write_synthetic_mesh(path)
    vertices_p3d, faces_p3d = p3d_io.load_ply(str(path))
    mesh_wp = od.io.load_mesh(path, device=device)

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


def test_load_mesh_without_faces_raises(tmp_path: Path, device: str):
    path = tmp_path / "cloud.ply"
    points_np = np.random.default_rng(0).random((10, 3)).astype(np.float64)
    meshio.write(str(path), meshio.Mesh(points_np, []))

    with pytest.raises(ValueError, match="no triangle faces"):
        od.io.load_mesh(path, device=device)
