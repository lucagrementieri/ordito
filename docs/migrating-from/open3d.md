# Migrating from Open3D

Open3D's point-cloud and reconstruction API maps closely onto `ordito.points`,
`ordito.registration`, `ordito.reconstruction`, and `ordito.voxels` — the main structural
difference is that Open3D's `PointCloud` / `TriangleMesh` are mutable objects with in-place
filters, where ordito functions take arrays and return new arrays.

```python
import open3d as o3d  # before
import ordito as od  # after
```

## Point clouds

| Open3D | ordito |
|---|---|
| `pcd.estimate_normals()` | [`points.estimate_normals(points, neighbor_idx)`][ordito.points.estimate_normals] (neighbourhood built explicitly via [`neighbors.query_nearest`][ordito.neighbors.query_nearest]; normals oriented away from the cloud's centroid, where Open3D leaves the sign arbitrary until an `orient_normals_*` call) |
| `pcd.remove_radius_outlier(nb_points, radius)` | [`points.radius_outlier_mask`][ordito.points.radius_outlier_mask] (returns a mask of the outliers; keep the rest with [`array.flatnonzero`][ordito.array.flatnonzero] and a gather) |
| `pcd.remove_statistical_outlier(nb_neighbors, std_ratio)` | [`points.statistical_outlier_mask`][ordito.points.statistical_outlier_mask] (also a mask) |
| `pcd.voxel_down_sample(voxel_size)` | [`voxels.voxel_down_sample`][ordito.voxels.voxel_down_sample] |
| `pcd.farthest_point_down_sample(n)` | [`points.farthest_point_sample`][ordito.points.farthest_point_sample] |
| `o3d.geometry.VoxelGrid.create_from_point_cloud` | [`voxels.voxelize_points`][ordito.voxels.voxelize_points] |
| `o3d.geometry.VoxelGrid.create_from_triangle_mesh` | [`voxels.voxelize_mesh`][ordito.voxels.voxelize_mesh] |

## Registration

| Open3D | ordito |
|---|---|
| `o3d.pipelines.registration.registration_icp(..., TransformationEstimationPointToPoint())` | [`registration.icp`][ordito.registration.icp] |
| `o3d.pipelines.registration.registration_icp(..., TransformationEstimationPointToPlane())` | [`registration.icp_point_to_plane`][ordito.registration.icp_point_to_plane] |
| A manual Procrustes / Kabsch fit over known correspondences | [`registration.procrustes`][ordito.registration.procrustes] |

## Surface reconstruction and meshes

| Open3D | ordito |
|---|---|
| `TriangleMesh.create_from_point_cloud_poisson` | [`reconstruction.screened_poisson`][ordito.reconstruction.screened_poisson] |
| `TriangleMesh.create_from_point_cloud_ball_pivoting` | [`reconstruction.ball_pivoting`][ordito.reconstruction.ball_pivoting] |
| `TriangleMesh.get_volume()` | [`measures.volume`][ordito.measures.volume] / `Trimesh.volume` |
| `TriangleMesh.is_watertight()` | [`validation.is_watertight`][ordito.validation.is_watertight] / `Trimesh.is_watertight` |
| `TriangleMesh.simplify_quadric_decimation` | [`remesh.quadric_decimate`][ordito.remesh.quadric_decimate] |
| `TriangleMesh.filter_smooth_laplacian` | [`smoothing.filter_laplacian`][ordito.smoothing.filter_laplacian] |
| `TriangleMesh.filter_smooth_taubin` | [`smoothing.filter_taubin`][ordito.smoothing.filter_taubin] |

## Ray casting

| Open3D (`o3d.t.geometry.RaycastingScene`) | ordito |
|---|---|
| `.compute_closest_points()` | [`proximity.closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh] |
| `.compute_signed_distance()` | [`proximity.signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh] (same sign convention: negative inside) |
| `.cast_rays()` (first hit) | [`ray.intersects_first`][ordito.ray.intersects_first] / [`ray.intersects_location`][ordito.ray.intersects_location] |
| `.count_intersections()` (parity test) | [`ray.contains_points`][ordito.ray.contains_points] |

## What's different, not just renamed

- **No mutable-object filters.** Every Open3D `TriangleMesh`/`PointCloud` method that returns
  `self` after mutating in place (`.remove_duplicated_vertices()`, `.compute_vertex_normals()`,
  ...) is a ordito function returning a new array, e.g.
  [`repair.remove_duplicated_vertices`][ordito.repair.remove_duplicated_vertices].
- **Legacy vs. tensor API distinction doesn't apply.** ordito has one API; there's no separate
  `o3d.t.geometry` GPU path to choose — every function already runs on whatever device its input
  arrays are on.
