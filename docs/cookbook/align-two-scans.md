# Aligning two scans

Registering two overlapping point clouds is a two-stage problem: a **coarse** alignment when
correspondences are already known (or can be guessed), and a **refinement** that alternates
finding nearest-point correspondences with re-solving for the best rigid transform.
[`registration.procrustes`][ordito.registration.procrustes] does the first, and
[`registration.icp`][ordito.registration.icp] does the second, taking the first's result as its
starting point.

```python
import numpy as np
import warp as wp

import ordito as od

vertices, faces = od.creation.icosphere(subdivisions=3)
points_a, _ = od.sample.sample_surface(vertices, faces, 2_000, seed=1)

# Simulate a second, misaligned scan of the same surface: rotate, translate, and jitter it.
rng = np.random.default_rng(0)
theta = 0.4
rotation = np.array(
    [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
    dtype=np.float32,
)
points_b_np = points_a.numpy() @ rotation.T + np.array([0.3, -0.1, 0.05], dtype=np.float32)
points_b_np += rng.normal(scale=0.01, size=points_b_np.shape).astype(np.float32)
points_b = wp.array(points_b_np, dtype=wp.vec3, device=vertices.device)

# Coarse alignment: procrustes needs points_b and points_a in correspondence already (same
# sampling, here), and solves for the similarity transform that best maps one onto the other.
transforms, aligned, cost = od.registration.procrustes(points_b, points_a)
print("procrustes residual:", cost)  # ~3e-4 (the jitter's own variance)

# Refinement against the original surface: icp re-derives correspondences by nearest point on
# the target mesh every iteration, starting from procrustes's transform.
transforms, aligned, rmse = od.registration.icp(points_b, vertices, faces, initial=transforms)
print("icp rmse:", rmse)  # ~1e-4 -- a real, if small, improvement over the coarse fit
```

## When there's no known correspondence

The example above cheats a little: `points_a` and `points_b` are already index-aligned (`points_b`
is a transform of `points_a`), so `procrustes` can solve directly. A real pair of independent scans
has no such correspondence, and `procrustes` alone isn't the right tool — it needs matched pairs.
In that case, skip straight to [`registration.icp`][ordito.registration.icp] with a rough initial
guess (identity, or a coarse alignment from any prior knowledge of the scan setup); each iteration
finds its own nearest-point correspondences and only needs the previous iteration's transform to be
in the right neighbourhood, not exact.

## Point-to-plane for faster convergence on flat regions

[`registration.icp_point_to_plane`][ordito.registration.icp_point_to_plane] minimizes the
point-to-plane distance (using the target's normals) rather than point-to-point, which converges
faster on largely flat surfaces — the same trade Open3D's `TransformationEstimationPointToPlane`
makes over `PointToPoint`. It takes the same arguments as `icp` plus the target's normals.
