# Point cloud to watertight surface

Two different reconstruction algorithms answer two different questions, and ordito ships both
GPU-native: [`reconstruction.screened_poisson`][ordito.reconstruction.screened_poisson] fits an
**implicit** surface (smooth, always watertight, doesn't interpolate the input points exactly) and
[`reconstruction.ball_pivoting`][ordito.reconstruction.ball_pivoting] builds an **interpolating**
one (its output vertices are literally a subset of the input points, but it can leave small holes
where the ball can't find three points to rest on). Both need oriented normals.

```python
import warp as wp

import ordito as od

vertices, faces = od.creation.icosphere(subdivisions=3)
mesh = od.Trimesh(vertices, faces)

# Simulate a scanner: sample the surface into a point cloud and keep each sample's true normal
# (a real scan would estimate normals instead -- see the note below).
points, face_index = od.sample.sample_surface(vertices, faces, 20_000, seed=0)
normals = wp.array(
    mesh.face_normals.numpy()[face_index.numpy()], dtype=wp.vec3, device=vertices.device
)

poisson_vertices, poisson_faces = od.reconstruction.screened_poisson(points, normals, depth=7)
print("screened Poisson faces:", poisson_faces.shape[0] // 3)  # 127460
print(
    "screened Poisson watertight:", od.validation.is_watertight(poisson_vertices, poisson_faces)
)  # True

bpa_vertices, bpa_faces = od.reconstruction.ball_pivoting(points, normals)
print(
    "ball pivoting faces:", bpa_faces.shape[0] // 3, "verts:", bpa_vertices.shape[0]
)  # 34804, 19864
```

## Estimating normals for a real scan

A real point cloud rarely comes with normals attached. Build a per-point neighbourhood with
[`neighbors.query_nearest`][ordito.neighbors.query_nearest] and estimate normals by local PCA with
[`points.estimate_normals`][ordito.points.estimate_normals]:

```python
neighbor_idx, _ = od.neighbors.query_nearest(points, points, k=16)
estimated_normals = od.points.estimate_normals(points, neighbor_idx)
```

`estimate_normals` orients each normal consistently with its neighbours, but not globally — for a
star-shaped cloud (like the sphere above), pass `orient_reference` or reconstruct with
`ball_pivoting`'s own PCA fallback (leave `normals=None`); for a general cloud, a global
orientation pass is a separate problem this function deliberately doesn't solve for you.

## Choosing between the two

- **`screened_poisson`** is the right default for a noisy or unevenly-sampled cloud: it's a global
  least-squares fit, always closed, and its `depth` parameter trades resolution for solve cost —
  see the function's own docstring for the `"adaptive"` method, which concentrates resolution
  where the input actually has detail.
- **`ball_pivoting`** is the right choice when the reconstruction needs to interpolate the input
  points exactly (e.g., because downstream code correlates reconstructed vertices back to scan
  samples), or when the cloud is dense and clean enough that its holes will be small. `radius`
  defaults to an automatic guess from the mean nearest-neighbour spacing; tune it down for a
  denser cloud or up to bridge a sparser one.
