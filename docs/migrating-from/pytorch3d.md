# Migrating from PyTorch3D

PyTorch3D is the one reference library with CUDA kernels of its own, and its batched
neighbour/Chamfer primitives and mesh regularization losses map onto `ordito.neighbors`,
`ordito.metrics`, and `ordito.energies`. The structural difference: PyTorch3D operates on
padded, batched `torch.Tensor`s wrapped in a `Meshes`/`Pointclouds` container; ordito has no
batch dimension or container — pass one mesh's arrays per call, and loop (or, for a truly large
batch, concatenate meshes with [`combine.concatenate`][ordito.combine.concatenate] and track
per-mesh index ranges yourself) where PyTorch3D would use its batch axis.

```python
import pytorch3d.ops as p3d_ops  # before
import pytorch3d.loss as p3d_loss
import ordito as od  # after
```

## Neighbours and distances

| PyTorch3D | ordito |
|---|---|
| `p3d_ops.knn_points(p, q, K=k)` | [`neighbors.query_nearest(points, queries, k)`][ordito.neighbors.query_nearest] |
| `p3d_ops.ball_query(p, q, radius=r)` | [`neighbors.query_ball`][ordito.neighbors.query_ball] / [`query_ball_with_offsets`][ordito.neighbors.query_ball_with_offsets] |
| `p3d_loss.chamfer_distance(a, b)` | [`metrics.chamfer_points_to_points_loss`][ordito.metrics.chamfer_points_to_points_loss] (differentiable, via `wp.Tape`) or [`chamfer_points_to_points`][ordito.metrics.chamfer_points_to_points] for the plain (non-differentiable) distance |
| `p3d_ops.point_mesh_face_distance` | [`metrics.chamfer_points_to_mesh`][ordito.metrics.chamfer_points_to_mesh] / [`chamfer_points_to_mesh_loss`][ordito.metrics.chamfer_points_to_mesh_loss] |

Remember to take a square root: every neighbour and Chamfer distance PyTorch3D returns is
**squared**; ordito's are not.

## Mesh regularization losses

| PyTorch3D | ordito |
|---|---|
| `p3d_loss.mesh_edge_loss` | [`energies.edge_length_loss`][ordito.energies.edge_length_loss] |
| `p3d_loss.mesh_laplacian_smoothing` | [`energies.laplacian_smoothing_loss`][ordito.energies.laplacian_smoothing_loss] |
| `p3d_loss.mesh_normal_consistency` | [`energies.normal_consistency_loss`][ordito.energies.normal_consistency_loss] (counts one term per **edge-manifold** adjacency, unlike PyTorch3D's per-*pair*-of-incident-faces count — the two agree exactly on edge-manifold input) |
| `p3d_ops.cot_laplacian` | [`laplacian.cotmatrix`][ordito.laplacian.cotmatrix] (ordito assembles the diagonal via the row sum and de-duplicates; PyTorch3D's is an uncoalesced sparse tensor with an all-zero diagonal — coalesce and compare, don't assume the raw tensors already agree) |
| `p3d_ops.taubin_smoothing` | [`smoothing.filter_taubin`][ordito.smoothing.filter_taubin] (pass `recompute=True` to match PyTorch3D's per-half-pass operator rebuild; the default reuses one fixed operator, which is both cheaper and the more standard formulation of Taubin's filter) |
| `p3d_ops.corresponding_points_alignment` | [`registration.procrustes`][ordito.registration.procrustes] (ordito's rotation is the transpose of PyTorch3D's, since PyTorch3D solves the row-vector form `s·X·R + T = Y`) |
| `p3d_ops.iterative_closest_point` | [`registration.icp`][ordito.registration.icp] |
| `p3d_ops.sample_points_from_meshes` | [`sample.sample_surface`][ordito.sample.sample_surface] |
| `p3d_ops.mesh_face_areas_normals` | [`triangles.face_normals_and_areas`][ordito.triangles.face_normals_and_areas] |

## What's different, not just renamed

- **No batch axis.** Every ordito call is over one mesh/cloud's arrays; there's no `Meshes`
  container tracking per-item padding, and no `packed_to_padded`/`padded_to_packed` bookkeeping to
  do at the boundary.
- **No implicit differentiation.** PyTorch3D's losses are always differentiable because
  everything is a `torch.Tensor`; ordito's `*_loss` functions are the differentiable variants
  (built with `wp.Tape`), and the plain functions with the same base name (e.g.
  [`metrics.chamfer_points_to_points`][ordito.metrics.chamfer_points_to_points]) are not — pick
  the one your use case needs rather than assuming every call carries a gradient.
- **`torch.Tensor` in, `wp.array` out.** There is no `torch` dependency anywhere in ordito;
  converting between a `torch.Tensor` and a `wp.array` for values already on the GPU is a
  `wp.from_torch`/`wp.to_torch` call ([documented directly by Warp](https://nvidia.github.io/warp/modules/interoperability.html)),
  not a host round-trip.
