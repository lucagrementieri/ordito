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

- **Every ordito function takes and returns `wp.array`, never `np.ndarray`.** A one-time
  `wp.array(numpy_array, dtype=...)` / `warp_array.numpy()` pair is the whole conversion; see
  [Concepts](../concepts.md#arrays-in-arrays-out) for why the boundary is drawn there.
- **Faces are a flat `(3 * n_faces,)` `wp.int32` buffer, not `(n_faces, 3)`.** Every mapping table
  below assumes this; reshape once at the boundary
  (`faces_flat = faces_np.reshape(-1)`, `faces_rows = faces_flat.numpy().reshape(-1, 3)`) rather
  than at every call site.
- **There is no scene graph, viewer, or mesh "session" object.** ordito is a library of
  functions (plus one optional, stateless [`Trimesh`][ordito.mesh.Trimesh] convenience wrapper),
  not a mutable-document editor like a MeshLab `MeshSet` or an Open3D `TriangleMesh` with in-place
  filters. A function that would mutate its input in one of those libraries instead returns a new
  array in ordito.

These pages name the closest ordito equivalent for each function; they are not exhaustive — the
full picture is the generated [API Reference](../api/ordito/creation.md), and every mapping asserted here
is one this project's own test suite checks by comparing outputs directly against the reference
library, not merely by name.
