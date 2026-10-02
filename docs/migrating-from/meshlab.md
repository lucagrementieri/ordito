# Migrating from MeshLab / PyMeshLab

MeshLab's editing filters map onto `ordito.remesh`, `ordito.smoothing`, `ordito.holes`, and
`ordito.repair`. The structural difference: a PyMeshLab `MeshSet` is a mutable, stateful
container where each filter mutates `current_mesh()` in place; ordito filters take
`(vertices, faces)` and return a new pair, so nothing needs a fresh `MeshSet` per call.

```python
import pymeshlab as ml  # before
import ordito as od  # after
```

## Remeshing and decimation

| PyMeshLab | ordito |
|---|---|
| `meshing_isotropic_explicit_remeshing(targetlen=...)` | [`remesh.isotropic_remesh(target_length=...)`][ordito.remesh.isotropic_remesh] |
| `meshing_decimation_quadric_edge_collapse(targetfacenum=...)` | [`remesh.quadric_decimate`][ordito.remesh.quadric_decimate] |
| `meshing_decimation_clustering(threshold=...)` | [`remesh.cluster_decimate`][ordito.remesh.cluster_decimate] |
| `meshing_surface_subdivision_midpoint` | [`remesh.subdivide`][ordito.remesh.subdivide] |
| `meshing_surface_subdivision_loop` | [`remesh.subdivide_loop`][ordito.remesh.subdivide_loop] |
| `meshing_repair_non_manifold_edges` | [`repair.split_non_manifold_vertices`][ordito.repair.split_non_manifold_vertices] / [`remove_non_manifold_faces`][ordito.repair.remove_non_manifold_faces] |

## Hole filling and cleanup

| PyMeshLab | ordito |
|---|---|
| `meshing_close_holes(maxholesize=...)` | [`holes.fill_small(max_edges=...)`][ordito.holes.fill_small] (a boundary-edge-count threshold, matching pymeshfix — see the function's own docstring for the exact off-by-one convention against MeshLab's `maxholesize`), or [`holes.fill_min_weight`][ordito.holes.fill_min_weight] for every boundary regardless of size |
| `meshing_remove_duplicate_faces` | [`repair.resolve_duplicated_faces`][ordito.repair.resolve_duplicated_faces] |
| `meshing_remove_unreferenced_vertices` | [`repair.remove_unreferenced_vertices`][ordito.repair.remove_unreferenced_vertices] |
| `meshing_remove_connected_component_by_diameter` / `_by_face_number` | [`repair.remove_small_components`][ordito.repair.remove_small_components] |
| `meshing_snap_mismatched_borders` | [`holes.stitch`][ordito.holes.stitch] / [`holes.stitch_min_weight`][ordito.holes.stitch_min_weight] |
| No direct equivalent (a `PyTMesh` / pymeshfix operation) | [`repair.collapse_small_triangles`][ordito.repair.collapse_small_triangles], [`repair.fix_self_intersections`][ordito.repair.fix_self_intersections] |

## Smoothing and curvature

| PyMeshLab | ordito |
|---|---|
| `apply_coord_laplacian_smoothing` | [`smoothing.filter_laplacian`][ordito.smoothing.filter_laplacian] |
| `apply_coord_taubin_smoothing` | [`smoothing.filter_taubin`][ordito.smoothing.filter_taubin] |
| `apply_coord_hc_laplacian_smoothing` | [`smoothing.filter_humphrey`][ordito.smoothing.filter_humphrey] |
| `compute_curvature_principal_directions_per_vertex` | [`curvature.principal_curvature`][ordito.curvature.principal_curvature] |
| `compute_scalar_by_shape_diameter_function_per_vertex` | [`visibility.shape_diameter`][ordito.visibility.shape_diameter] |
| `compute_scalar_ambient_occlusion` | [`visibility.ambient_occlusion`][ordito.visibility.ambient_occlusion] |

## Reconstruction and resampling

| PyMeshLab | ordito |
|---|---|
| `generate_surface_reconstruction_screened_poisson` | [`reconstruction.screened_poisson`][ordito.reconstruction.screened_poisson] |
| `generate_surface_reconstruction_ball_pivoting` | [`reconstruction.ball_pivoting`][ordito.reconstruction.ball_pivoting] |
| `generate_resampled_uniform_mesh` | [`reconstruction.resample_uniform`][ordito.reconstruction.resample_uniform] |

## What's different, not just renamed

- **No `MeshSet`, no `current_mesh()`.** Every ordito filter is a pure function over
  `(vertices, faces)`; there's no session object to build, select a layer in, or read a result
  back off of.
- **Length parameters are always the same unit `float` the rest of the API uses**, never a
  `PercentageValue`/`PureValue` wrapper — pass an absolute length (or a fraction you compute
  yourself against the mesh's own bounding-box diagonal, via
  [`bounds.aabb`][ordito.bounds.aabb] / `Trimesh.extents`, if that's what a MeshLab default was
  doing under the hood).
- **`meshing_close_holes`'s size threshold is a perimeter** (an edge-length sum); ordito's
  [`holes.fill_small`][ordito.holes.fill_small] follows pymeshfix's convention instead (a
  boundary **edge count**) — see that function's docstring for why, and for the exact conversion
  when porting a MeshLab-tuned threshold.
