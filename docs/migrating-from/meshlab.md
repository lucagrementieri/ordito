# Migrating from MeshLab / PyMeshLab

MeshLab's editing filters map onto `triwarp.remesh`, `triwarp.smoothing`, `triwarp.holes`, and
`triwarp.repair`. The structural difference: a PyMeshLab `MeshSet` is a mutable, stateful
container where each filter mutates `current_mesh()` in place; triwarp filters take
`(vertices, faces)` and return a new pair, so nothing needs a fresh `MeshSet` per call.

```python
import pymeshlab as ml  # before
import triwarp as tw  # after
```

## Remeshing and decimation

| PyMeshLab | triwarp |
|---|---|
| `meshing_isotropic_explicit_remeshing(targetlen=...)` | [`remesh.isotropic_remesh(target_length=...)`][triwarp.remesh.isotropic_remesh] |
| `meshing_decimation_quadric_edge_collapse(targetfacenum=...)` | [`remesh.quadric_decimate`][triwarp.remesh.quadric_decimate] |
| `meshing_decimation_clustering(threshold=...)` | [`remesh.cluster_decimate`][triwarp.remesh.cluster_decimate] |
| `meshing_surface_subdivision_midpoint` | [`remesh.subdivide`][triwarp.remesh.subdivide] |
| `meshing_surface_subdivision_loop` | [`remesh.subdivide_loop`][triwarp.remesh.subdivide_loop] |
| `meshing_repair_non_manifold_edges` | [`repair.split_non_manifold_vertices`][triwarp.repair.split_non_manifold_vertices] / [`remove_non_manifold_faces`][triwarp.repair.remove_non_manifold_faces] |

## Hole filling and cleanup

| PyMeshLab | triwarp |
|---|---|
| `meshing_close_holes(maxholesize=...)` | [`holes.fill_small(max_edges=...)`][triwarp.holes.fill_small] (a boundary-edge-count threshold, matching pymeshfix — see the function's own docstring for the exact off-by-one convention against MeshLab's `maxholesize`), or [`holes.fill_min_weight`][triwarp.holes.fill_min_weight] for every boundary regardless of size |
| `meshing_remove_duplicate_faces` | [`repair.resolve_duplicated_faces`][triwarp.repair.resolve_duplicated_faces] |
| `meshing_remove_unreferenced_vertices` | [`repair.remove_unreferenced_vertices`][triwarp.repair.remove_unreferenced_vertices] |
| `meshing_remove_connected_component_by_diameter` / `_by_face_number` | [`repair.remove_small_components`][triwarp.repair.remove_small_components] |
| `meshing_snap_mismatched_borders` | [`holes.stitch`][triwarp.holes.stitch] / [`holes.stitch_min_weight`][triwarp.holes.stitch_min_weight] |
| No direct equivalent (a `PyTMesh` / pymeshfix operation) | [`repair.collapse_small_triangles`][triwarp.repair.collapse_small_triangles], [`repair.fix_self_intersections`][triwarp.repair.fix_self_intersections] |

## Smoothing and curvature

| PyMeshLab | triwarp |
|---|---|
| `apply_coord_laplacian_smoothing` | [`smoothing.filter_laplacian`][triwarp.smoothing.filter_laplacian] |
| `apply_coord_taubin_smoothing` | [`smoothing.filter_taubin`][triwarp.smoothing.filter_taubin] |
| `apply_coord_hc_laplacian_smoothing` | [`smoothing.filter_humphrey`][triwarp.smoothing.filter_humphrey] |
| `compute_curvature_principal_directions_per_vertex` | [`curvature.principal_curvature`][triwarp.curvature.principal_curvature] |
| `compute_scalar_by_shape_diameter_function_per_vertex` | [`visibility.shape_diameter`][triwarp.visibility.shape_diameter] |
| `compute_scalar_ambient_occlusion` | [`visibility.ambient_occlusion`][triwarp.visibility.ambient_occlusion] |

## Reconstruction and resampling

| PyMeshLab | triwarp |
|---|---|
| `generate_surface_reconstruction_screened_poisson` | [`reconstruction.screened_poisson`][triwarp.reconstruction.screened_poisson] |
| `generate_surface_reconstruction_ball_pivoting` | [`reconstruction.ball_pivoting`][triwarp.reconstruction.ball_pivoting] |
| `generate_resampled_uniform_mesh` | [`reconstruction.resample_uniform`][triwarp.reconstruction.resample_uniform] |

## What's different, not just renamed

- **No `MeshSet`, no `current_mesh()`.** Every triwarp filter is a pure function over
  `(vertices, faces)`; there's no session object to build, select a layer in, or read a result
  back off of.
- **Length parameters are always the same unit `float` the rest of the API uses**, never a
  `PercentageValue`/`PureValue` wrapper — pass an absolute length (or a fraction you compute
  yourself against the mesh's own bounding-box diagonal, via
  [`bounds.aabb`][triwarp.bounds.aabb] / `Trimesh.extents`, if that's what a MeshLab default was
  doing under the hood).
- **`meshing_close_holes`'s size threshold is a perimeter** (an edge-length sum); triwarp's
  [`holes.fill_small`][triwarp.holes.fill_small] follows pymeshfix's convention instead (a
  boundary **edge count**) — see that function's docstring for why, and for the exact conversion
  when porting a MeshLab-tuned threshold.
