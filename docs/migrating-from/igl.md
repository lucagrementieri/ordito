# Migrating from libigl

libigl's input convention (`float64` `(n, 3)` vertices, `int64` `(n_faces, 3)` faces) differs
from ordito's in layout and precision: ordito takes `float32` `wp.vec3` vertices and a flat
`(3 * n_faces,)` `wp.int32` face buffer. Precision is not lost where it matters: solvers that
need `float64` (the heat method, the parametrizations, the sparse solves) promote internally and
document their output dtype.

```python
import igl  # before
import ordito as od  # after
```

## Discrete differential geometry

| libigl | ordito |
|---|---|
| `igl.cotmatrix(V, F)` | [`laplacian.cotmatrix(vertices, faces)`][ordito.laplacian.cotmatrix] |
| `igl.massmatrix(V, F)` | [`laplacian.mass_matrix(vertices, faces)`][ordito.laplacian.mass_matrix] |
| `igl.gaussian_curvature(V, F)` | [`vertices.vertex_defects(vertices, faces)`][ordito.vertices.vertex_defects] |
| `igl.principal_curvature(V, F)` | [`curvature.principal_curvature`][ordito.curvature.principal_curvature] |
| `igl.per_vertex_normals(V, F)` | [`vertices.vertex_normals`][ordito.vertices.vertex_normals] / [`mean_vertex_normals`][ordito.vertices.mean_vertex_normals] / [`weighted_vertex_normals`][ordito.vertices.weighted_vertex_normals] |
| `igl.per_face_normals(V, F, Z)` | [`triangles.face_normals_and_areas`][ordito.triangles.face_normals_and_areas] (normals and areas together) |
| `igl.doublearea(V, F) / 2` | [`triangles.face_normals_and_areas`][ordito.triangles.face_normals_and_areas]'s area output |
| `igl.heat_geodesics_precompute` + `igl.heat_geodesics_solve` | [`heat.heat_operators`][ordito.heat.heat_operators] + [`heat.heat_geodesic`][ordito.heat.heat_geodesic] |
| `igl.exact_geodesic` | No exact solver. [`heat.heat_geodesic`][ordito.heat.heat_geodesic] approximates the same distance (within about 1% on a sphere at 2 562 vertices; see the [cookbook](../cookbook/geodesic-distance.md)). |
| `igl.harmonic(V, F, b, bc, k)` | [`parametrization.harmonic`][ordito.parametrization.harmonic] |
| `igl.lscm(V, F, b, bc)` | [`parametrization.lscm`][ordito.parametrization.lscm] |
| `igl.arap_precomputation` + `igl.arap_solve` | [`parametrization.arap`][ordito.parametrization.arap] |
| `igl.min_quad_with_fixed` | [`linalg.min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] |
| `igl.adjacency_matrix(F)` | [`edges.edges_unique`][ordito.edges.edges_unique] then [`graph.edges_to_csr`][ordito.graph.edges_to_csr] (vertex-vertex, as a sparse matrix); [`adjacency.face_adjacency`][ordito.adjacency.face_adjacency] for face-face pairs |
| `igl.boundary_loop(F)` | [`boundary.longest_boundary_loop`][ordito.boundary.longest_boundary_loop] (all loops: [`boundary.boundary_loops`][ordito.boundary.boundary_loops]) |
| `igl.upsample` / `igl.loop` | [`remesh.subdivide`][ordito.remesh.subdivide] / [`remesh.subdivide_loop`][ordito.remesh.subdivide_loop] |
| `igl.decimate` | [`remesh.quadric_decimate`][ordito.remesh.quadric_decimate] |
| `igl.is_vertex_manifold` | [`validation.is_vertex_manifold`][ordito.validation.is_vertex_manifold] |
| `igl.is_edge_manifold` | [`validation.is_edge_manifold`][ordito.validation.is_edge_manifold] |
| `igl.connected_components(igl.adjacency_matrix(F))` | [`Trimesh.face_connected_component_labels`][ordito.mesh.Trimesh.face_connected_component_labels] |

## What's different, not just renamed

- **`igl`'s `(V, F)` positional convention becomes ordito's `(vertices, faces)` keyword-friendly
  one**, with `faces` flat rather than `(n_faces, 3)`.
- **Neither library checks every index on every path.** In libigl an out-of-range face index
  crashes the process. ordito checks indices where a function documents it (a `Raises` entry, as
  in [`repair.make_solid`][ordito.repair.make_solid]); elsewhere an out-of-range index is a wrong
  answer or a memory error, not an exception. Validate meshes from untrusted files once, at load
  time.
- **libigl factorizes, ordito iterates.** `igl.min_quad_with_fixed` and friends solve with a
  sparse direct factorization (Eigen's `SimplicialLLT`); ordito's
  [`linalg.solve_spd`][ordito.linalg.solve_spd] family solves on the device with preconditioned
  conjugate gradient. Both reach the same answer to the solver tolerance; an iterative solve's
  cost grows with how ill-conditioned the system is, a factorization's does not.
