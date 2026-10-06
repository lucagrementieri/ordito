# Point cloud to watertight surface

A scanner gives you points, not triangles. ordito has two reconstruction algorithms, and they
answer different questions:

- [`reconstruction.screened_poisson`][ordito.reconstruction.screened_poisson] fits an **implicit**
  surface: it solves for a smooth function that is zero on the surface, then extracts that level
  set. The result is always closed (watertight) and smooths over noise, but its vertices are new
  points, not the input samples.
- [`reconstruction.ball_pivoting`][ordito.reconstruction.ball_pivoting] builds an
  **interpolating** mesh: it rolls a small ball over the cloud and adds a triangle wherever the
  ball rests on three points. Its vertices *are* input points, but where the ball cannot rest
  (sparse or uneven sampling) it leaves a hole.

Both need a normal at every point, to tell the inside of the surface from the outside. A real
scan rarely comes with normals, so the recipe estimates them first.

```python
import numpy as np

import ordito as od

# Simulate a scanner: 20 000 points sampled at random from a sphere, with no normals attached.
vertices, faces = od.creation.icosphere(subdivisions=3)
points, face_index = od.sample.sample_surface(vertices, faces, 20_000, seed=0)


def estimate_normals(points):
    """A normal per point from its 16 nearest neighbours (a local plane fit, by PCA)."""
    neighbor_idx, _ = od.neighbors.query_nearest(points, points, k=16)
    return od.points.estimate_normals(points, neighbor_idx)


normals = estimate_normals(points)

# This is a simulation, so the estimate can be checked against the true face normals.
true_normals = od.Trimesh(vertices, faces).face_normals.numpy()[face_index.numpy()]
agreement = np.sum(normals.numpy() * true_normals, axis=1)  # cosine of the angle between them
print(f"median cosine: {np.median(agreement):.4f}, flipped: {np.sum(agreement < 0)}")  # ~0.9996, 0

# Screened Poisson: closed whatever the sampling looks like.
poisson_vertices, poisson_faces = od.reconstruction.screened_poisson(points, normals, depth=7)
print("screened Poisson:", poisson_faces.shape[0] // 3, "faces")  # ~127 000
print("  watertight:", od.validation.is_watertight(poisson_vertices, poisson_faces))  # True


def ball_pivot(points, normals):
    bpa_vertices, bpa_faces = od.reconstruction.ball_pivoting(points, normals)
    _, hole_offsets = od.boundary.boundary_loops_with_offsets(bpa_vertices, bpa_faces)
    n_holes = hole_offsets.shape[0] - 1  # offsets carry one entry per loop, plus a final total
    print(f"  {bpa_vertices.shape[0]} of {points.shape[0]} points used, {n_holes} holes")


# Ball pivoting on the random sample: random points clump and leave gaps the ball falls through.
print("ball pivoting, random sample:")
ball_pivot(points, normals)  # ~19 870 of 20 000 points used, ~1 300 holes

# The same density, evenly spaced (blue noise: no two points closer than `radius`).
even_points, _ = od.sample.sample_surface_blue_noise(vertices, faces, radius=0.02, seed=0)
print("ball pivoting, even sample:")
ball_pivot(even_points, estimate_normals(even_points))  # every point used, ~2 holes
```

Three things to notice in the output:

- **Screened Poisson is closed regardless of the sampling**, but its vertices are its own: far
  more of them than input points at `depth=7`, none of them exactly an input point.
- **Ball pivoting keeps the input points as vertices**, but it is only as good as the sampling.
  Random sampling clumps points in some places and leaves gaps in others, and every gap wider than
  the ball is a hole.
- **Even sampling fixes most of it.** With the same number of points spread evenly, the ball
  rolls over nearly the whole surface. Real scanners are closer to the even case than to random
  sampling, but they leave their own gaps (occlusions, grazing angles), which is why screened
  Poisson is the usual default.

## Estimating normals for a real scan

[`points.estimate_normals`][ordito.points.estimate_normals] fits a plane to each point's
neighbourhood, which fixes the normal's *line* but not its *sign*: a plane has two sides. By
default it orients every normal away from the centroid of the whole cloud. That is right for a
closed, roughly convex object like this sphere, and wrong for anything with deep concavities (the
inside of a mug, the space between two fingers). Two keywords cover the common capture setups:

- `camera_location=wp.vec3(...)`: point every normal toward the scanner position, the right choice
  for a single-view scan.
- `orient_reference=wp.vec3(...)`: point every normal along one direction (for example `+z` for a
  terrain scan).

For a general multi-view cloud of a complicated object, consistent global orientation is a hard
problem of its own, which this function does not attempt. `k=16` neighbours is a common default;
raise it for noisy data (a smoother fit) and lower it where the surface has fine detail.

## Choosing between the two

- **Use `screened_poisson`** for noisy or unevenly sampled clouds, and whenever you need a closed
  surface (to measure a volume, 3D-print, or test inside/outside). Its `depth` sets the resolution
  of the grid the surface is fitted on: each extra level gives about four times the faces and
  costs about eight times as much. `method="adaptive"` refines only near the samples instead of
  everywhere. See the function's docstring for the remaining parameters.
- **Use `ball_pivoting`** when output vertices must be input points (for example, to carry
  per-point colours or labels over unchanged) and the cloud is dense and evenly spaced. Its
  `radius` defaults to a guess from the typical spacing between neighbouring points. If the
  sampling is uneven, thin it out evenly first, as above, rather than hoping a larger ball bridges
  the gaps: on the random sample, no single radius brought the hole count anywhere near the even
  sample's.
