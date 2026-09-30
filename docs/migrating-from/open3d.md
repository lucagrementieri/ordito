# Migrating from Open3D

Open3D's point-cloud and reconstruction API maps closely onto `triwarp.points`,
`triwarp.registration`, `triwarp.reconstruction`, and `triwarp.voxels` — the main structural
difference is that Open3D's `PointCloud` / `TriangleMesh` are mutable objects with in-place
filters, where triwarp functions take arrays and return new arrays.

```python
import open3d as o3d  # before
import triwarp as tw  # after
```

## Point clouds

| Open3D | triwarp |
|---|---|
| `pcd.estimate_normals()` | [`points.estimate_normals(points, neighbor_idx)`][triwarp.points.estimate_normals] (neighbourhood built explicitly via [`neighbors.query_nearest`][triwarp.neighbors.query_nearest]) |
| `pcd.remove_radius_outlier(nb_points, radius)` | [`points.radius_outlier_mask`][triwarp.points.radius_outlier_mask] |
| `pcd.remove_statistical_outlier(nb_neighbors, std_ratio)` | [`points.statistical_outlier_mask`][triwarp.points.statistical_outlier_mask] |
| `pcd.voxel_down_sample(voxel_size)` | [`voxels.voxel_down_sample`][triwarp.voxels.voxel_down_sample] |
| `pcd.farthest_point_down_sample(n)` | [`points.farthest_point_sample`][triwarp.points.farthest_point_sample] |
| `o3d.geometry.VoxelGrid.create_from_point_cloud` | [`voxels.voxelize_points`][triwarp.voxels.voxelize_points] |
| `o3d.geometry.VoxelGrid.create_from_triangle_mesh` | [`voxels.voxelize_mesh`][triwarp.voxels.voxelize_mesh] |

## Registration

| Open3D | triwarp |
|---|---|
| `o3d.pipelines.registration.registration_icp(..., TransformationEstimationPointToPoint())` | [`registration.icp`][triwarp.registration.icp] |
| `o3d.pipelines.registration.registration_icp(..., TransformationEstimationPointToPlane())` | [`registration.icp_point_to_plane`][triwarp.registration.icp_point_to_plane] |
| A manual Procrustes / Kabsch fit over known correspondences | [`registration.procrustes`][triwarp.registration.procrustes] |

## Surface reconstruction and meshes

| Open3D | triwarp |
|---|---|
| `TriangleMesh.create_from_point_cloud_poisson` | [`reconstruction.screened_poisson`][triwarp.reconstruction.screened_poisson] |
| `TriangleMesh.create_from_point_cloud_ball_pivoting` | [`reconstruction.ball_pivoting`][triwarp.reconstruction.ball_pivoting] |
| `TriangleMesh.get_volume()` | [`measures.volume`][triwarp.measures.volume] / `Trimesh.volume` |
| `TriangleMesh.is_watertight()` | [`validation.is_watertight`][triwarp.validation.is_watertight] / `Trimesh.is_watertight` |
| `TriangleMesh.simplify_quadric_decimation` | [`remesh.quadric_decimate`][triwarp.remesh.quadric_decimate] |
| `TriangleMesh.filter_smooth_laplacian` | [`smoothing.filter_laplacian`][triwarp.smoothing.filter_laplacian] |
| `TriangleMesh.filter_smooth_taubin` | [`smoothing.filter_taubin`][triwarp.smoothing.filter_taubin] |

## Ray casting

| Open3D (`o3d.t.geometry.RaycastingScene`) | triwarp |
|---|---|
| `.compute_closest_points()` | [`proximity.closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh] |
| `.compute_signed_distance()` | [`proximity.signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh] |
| `.cast_rays()` (first hit) | [`ray.intersects_first`][triwarp.ray.intersects_first] / [`ray.intersects_location`][triwarp.ray.intersects_location] |
| `.count_intersections()` (parity test) | [`ray.contains_points`][triwarp.ray.contains_points] |

## What's different, not just renamed

- **No mutable-object filters.** Every Open3D `TriangleMesh`/`PointCloud` method that returns
  `self` after mutating in place (`.remove_duplicated_vertices()`, `.compute_vertex_normals()`,
  ...) is a triwarp function returning a new array, e.g.
  [`repair.remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices].
- **Legacy vs. tensor API distinction doesn't apply.** triwarp has one API; there's no separate
  `o3d.t.geometry` GPU path to choose — every function already runs on whatever device its input
  arrays are on.
