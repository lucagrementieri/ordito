# Aligning two scans

Registration finds the rigid motion that places one scan on top of another. ordito has two
tools for it, and which one applies depends on what you know about the data:

- **[`registration.procrustes`][ordito.registration.procrustes]** solves in closed form when the
  correspondences are already known: point `i` of one cloud is the same physical point as point
  `i` of the other (matched landmarks, or two poses of one tracked point set).
- **[`registration.icp`][ordito.registration.icp]** (iterative closest point) finds its own
  correspondences. Each iteration pairs every scan point with the closest point on the target,
  solves for the best rigid transform, moves the scan and repeats. It needs a starting pose close
  enough that "closest" is mostly right, and no correspondences at all.

This recipe builds a scan with a known motion, so you can see how close each method gets to it.

```python
import numpy as np
import warp as wp

import ordito as od

# The target: an ellipsoid with three different semi-axes. Don't test registration on a sphere:
# every rotation of a sphere fits it equally well, so there is no rotation to recover.
sphere_vertices, faces = od.creation.icosphere(subdivisions=4)
vertices = od.transform.transform_points(
    sphere_vertices, od.transform.scale_matrix((1.0, 0.6, 0.35))
)

# The scan: 5 000 points sampled independently from the same surface (no scan point is a target
# vertex), then moved by a known rotation and translation, then jittered with 2 mm of noise.
scan_clean, _ = od.sample.sample_surface(vertices, faces, 5_000, seed=1)
theta = 0.3  # radians, about the z axis
rotation = np.array(
    [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
)
offset = np.array([0.1, -0.05, 0.02])
rng = np.random.default_rng(0)
moved_np = scan_clean.numpy() @ rotation.T + offset + rng.normal(scale=0.002, size=(5_000, 3))
scan = wp.array(moved_np.astype(np.float32), dtype=wp.vec3, device=vertices.device)


def report(name, matrix):
    """How far the recovered transform is from undoing the known motion."""
    m = matrix.numpy()[0]  # a (1,) array of wp.mat44 reads back as a (1, 4, 4) NumPy array
    cos_angle = (np.trace(m[:3, :3] @ rotation) - 1.0) / 2.0
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    shift = np.linalg.norm(m[:3, :3] @ offset + m[:3, 3])
    print(f"{name:<24} rotation error {angle:.1e} rad, translation error {shift:.1e}")


# 1. Known correspondences: scan[i] is scan_clean[i] moved, so procrustes solves directly.
#    scale=False and reflection=False restrict it to a rigid motion; by default it would also
#    fit a uniform scale and allow a mirror image.
matrix, _, cost = od.registration.procrustes(scan, scan_clean, scale=False, reflection=False)
report("procrustes", matrix)  # ~2e-4 rad: only the noise limits it
print("  mean squared residual:", cost)  # ~1.2e-5 = 3 * 0.002**2, the noise variance

# 2. No correspondences: point-to-point ICP against the surface, from the identity.
matrix, _, cost = od.registration.icp(scan, vertices, faces)
report("icp, default stop", matrix)  # ~9e-2 rad (5 degrees): stopped too early, see below
matrix, _, cost = od.registration.icp(scan, vertices, faces, max_iterations=200, threshold=1e-9)
report("icp, tighter stop", matrix)  # ~5e-4 rad
print("  rms distance to surface:", np.sqrt(cost))  # ~0.002: down to the noise level

# 3. Point-to-plane ICP: same inputs, converges in far fewer iterations on a smooth surface.
matrix, _, cost = od.registration.icp_point_to_plane(scan, vertices, faces)
report("icp_point_to_plane", matrix)  # under 1e-3 rad, at the default stopping rule
```

## Reading the result

`matrix` is a `(1,)` array holding one `wp.mat44`: the transform that maps the *first* argument
onto the second. Apply it to more points with
[`transform.transform_points`][ordito.transform.transform_points]; the second return value is
already the first argument moved by it.

The `cost` is how well the moved scan fits, and it is the quantity to check:

- `procrustes` and `icp` report the **mean** squared distance, so `sqrt(cost)` is an RMS
  distance in the scan's own units. A converged fit sits at the noise level of the data; here
  that is about 0.002.
- `icp_point_to_plane` reports the **sum** of squared point-to-plane residuals (Open3D's
  convention). Divide by the number of points before taking the square root.

## When ICP stops too early

`icp` stops when the cost improves by less than `threshold` from one iteration to the next.
That is the right test when each iteration makes steady progress. On a smooth, curved target,
though, point-to-point ICP corrects a rotation slowly: every closest point slides along the
surface, so the cost falls by tiny amounts while the rotation is still visibly wrong. That is
what happens in step 2: at the default `threshold=1e-5`, the fit stops with the rotation still
about 5 degrees off.

Two fixes, both shown above:

- **Tighten the stopping rule** (`threshold=1e-9`, more `max_iterations`) and check that
  `sqrt(cost)` comes down to the noise level you expect from your scanner.
- **Switch to [`icp_point_to_plane`][ordito.registration.icp_point_to_plane].** It measures each
  residual along the target's normal, so sliding along the surface costs nothing and the fit
  moves straight to the answer. It is the usual choice for scans of smooth surfaces, the same
  trade Open3D makes between `TransformationEstimationPointToPlane` and `PointToPoint`. With a mesh
  target it computes the normals itself; pass `target_normals=` only when the target is a bare
  point cloud (`target_faces=None`).

## When the starting pose is far off

ICP only finds the *nearest* good alignment. If the scan starts rotated by a large angle, the
closest-point pairs are mostly wrong, and the fit can settle in the wrong place. Give it a better
start with `initial=` (a `wp.mat44`), from anything you know about the capture setup or from a
`procrustes` fit over a few hand-picked landmark pairs.
