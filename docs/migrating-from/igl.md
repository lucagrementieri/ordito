# Migrating from libigl

libigl's input convention (`float64` `(n, 3)` vertices, `int64` `(n_faces, 3)` faces) is the
closest of any reference library to ordito's own — the difference is entirely the face layout
(flat `(3 * n_faces,)` `wp.int32` in ordito, `(n_faces, 3)` `int64` in libigl) and dtype
(`float32` by default in ordito; pass `float64` arrays through where a solve genuinely needs the
extra precision, as `heat_geodesic` itself does internally).

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
| `igl.exact_geodesic` | [`geodesic_walk`][ordito.geodesic_walk] (a direct combinatorial surface walk rather than the heat-method approximation) |
| `igl.harmonic(V, F, b, bc, k)` | [`parametrization.harmonic`][ordito.parametrization.harmonic] |
| `igl.lscm(V, F, b, bc)` | [`parametrization.lscm`][ordito.parametrization.lscm] |
| `igl.arap_precomputation` + `igl.arap_solve` | [`parametrization.arap`][ordito.parametrization.arap] |
| `igl.min_quad_with_fixed` | [`linalg.min_quad_with_fixed`][ordito.linalg.min_quad_with_fixed] |
| `igl.adjacency_matrix(F)` | [`adjacency.face_adjacency(faces)`][ordito.adjacency.face_adjacency] for face-face, or [`vertices`][ordito.vertices]/[`edges`][ordito.edges] helpers for vertex-vertex |
| `igl.boundary_loop(F)` | [`boundary.longest_boundary_loop`][ordito.boundary.longest_boundary_loop] (all loops: [`boundary.boundary_loops`][ordito.boundary.boundary_loops]) |
| `igl.upsample` / `igl.loop` | [`remesh.subdivide`][ordito.remesh.subdivide] / [`remesh.subdivide_loop`][ordito.remesh.subdivide_loop] |
| `igl.decimate` | [`remesh.quadric_decimate`][ordito.remesh.quadric_decimate] |
| `igl.is_vertex_manifold` | [`validation.is_vertex_manifold`][ordito.validation.is_vertex_manifold] |
| `igl.is_edge_manifold` | [`validation.is_edge_manifold`][ordito.validation.is_edge_manifold] |
| `igl.connected_components(igl.adjacency_matrix(F))` | [`Trimesh.face_connected_component_labels`][ordito.mesh.Trimesh.face_connected_component_labels] |

## What's different, not just renamed

- **`igl`'s `(V, F)` positional convention becomes ordito's `(vertices, faces)` keyword-friendly
  one**, with `faces` flat rather than `(n_faces, 3)`.
- **libigl bounds nothing** — an out-of-range face index is a process crash in libigl (documented
  in this project's own test-oracle notes) and a `ValueError` or a defined result in ordito.
  Don't port an "it happened to work" libigl call site without re-checking its inputs.
- **`igl.min_quad_with_fixed` and friends return NumPy arrays libigl solved with its own
  factorization (`Eigen::SimplicialLLT` under the hood); ordito's
  [`linalg.solve_spd`][ordito.linalg.solve_spd] family solves on-device with conjugate gradient**,
  optionally multigrid-preconditioned for large or ill-conditioned systems. Both converge to the
  same answer; only the method (direct factorization vs. iterative solve) differs.
