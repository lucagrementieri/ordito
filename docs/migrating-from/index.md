# Migrating from another library

If you already have a mesh-processing pipeline built on one of the libraries below, these pages
map the functions you're calling today onto their ordito equivalent, and call out the handful of
places where the mapping isn't 1:1 — a different return convention, a parameter the reference
library doesn't expose, or an algorithm that's genuinely different rather than merely renamed.

| Coming from | For | Page |
|---|---|---|
| [trimesh](https://trimesh.org) | Mesh bookkeeping — edges, adjacency, boundary, validation, primitives, sampling, proximity | [trimesh →](trimesh.md) |
| [libigl](https://libigl.github.io/) | Discrete differential geometry — cotangent Laplacians, curvature, parametrization, heat geodesics | [libigl →](igl.md) |
| [Open3D](https://www.open3d.org/) | Point clouds, registration, surface reconstruction | [Open3D →](open3d.md) |
| [MeshLab](https://www.meshlab.net/) / [PyMeshLab](https://github.com/cnr-isti-vislab/PyMeshLab) | Editing filters — remeshing, decimation, smoothing, hole filling | [MeshLab →](meshlab.md) |
| [potpourri3d](https://github.com/nmwsharp/potpourri3d) | The heat-method family — vector heat, parallel transport, log maps, signed distance | [potpourri3d →](potpourri3d.md) |
| [PyTorch3D](https://pytorch3d.org/) | Batched neighbour/Chamfer primitives, mesh regularization losses | [PyTorch3D →](pytorch3d.md) |

Three things worth knowing before diving into a specific mapping:

- **Every ordito function takes and returns `wp.array`, never `np.ndarray`.** Converting is
  `wp.array(numpy_array, dtype=..., device=...)` in and `warp_array.numpy()` out; see
  [Concepts](../concepts.md#arrays-in-arrays-out) for why the boundary is drawn there.
- **Faces are a flat `(3 * n_faces,)` `wp.int32` buffer, not `(n_faces, 3)`.** Every mapping table
  below assumes this. Convert once, where your data enters and leaves ordito:

    ```python
    import numpy as np
    import trimesh
    import warp as wp

    import ordito as od

    mesh_tm = trimesh.creation.icosphere()
    vertices = wp.array(mesh_tm.vertices, dtype=wp.vec3)  # float64 rows become float32 vec3
    faces = wp.array(mesh_tm.faces.reshape(-1), dtype=wp.int32)  # (n_faces, 3) -> flat
    mesh = od.Trimesh(vertices, faces)

    faces_rows = mesh.faces.numpy().reshape(-1, 3)  # and back to (n_faces, 3)
    print(np.array_equal(faces_rows, mesh_tm.faces), mesh.area, mesh_tm.area)
    ```
- **There is no scene graph, viewer, or mesh "session" object.** ordito is a library of
  functions (plus one optional, stateless [`Trimesh`][ordito.mesh.Trimesh] convenience wrapper),
  not a mutable-document editor like a MeshLab `MeshSet` or an Open3D `TriangleMesh` with in-place
  filters. A function that would mutate its input in one of those libraries instead returns a new
  array in ordito.

These pages name the closest ordito equivalent for each function; they are not exhaustive.
The full picture is the [API Reference](../api/ordito/creation.md). Most rows are backed by a
test that compares the two functions' outputs directly; where a row says the two differ (a
convention, a sign, a different algorithm), that difference is measured and documented in the
ordito function's docstring.
