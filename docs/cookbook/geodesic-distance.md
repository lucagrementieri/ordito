# Geodesic distance fields

The geodesic distance between two points on a surface is the length of the shortest path that
stays on the surface. [`heat.heat_geodesic`][ordito.heat.heat_geodesic] computes it from a set
of source vertices to every vertex with the heat method of Crane, Weischedel and Wardetzky
(2013). It takes three steps, all on the device:

1. Let heat diffuse from the sources for a short time `t` (one sparse linear solve).
2. Normalize the gradient of the heat on every triangle. Heat flows away from the sources, so the
   negated, normalized gradient points along the shortest paths, with unit length.
3. Find the distance field whose gradient best matches that direction field (a second sparse
   solve: a Poisson equation).

A sphere makes the result easy to check, because the exact answer is known: on the unit sphere,
the geodesic distance between two points is the angle between them, `arccos(p · q)`.

```python
import numpy as np
import warp as wp

import ordito as od

vertices, faces = od.creation.icosphere(subdivisions=4)  # 2 562 vertices on the unit sphere
sources = wp.array([0], dtype=wp.int32, device=vertices.device)

distance = od.heat.heat_geodesic(vertices, faces, sources)  # wp.array[wp.float64], one per vertex

# Exact geodesic distance on the unit sphere: the angle to the source vertex.
points = vertices.numpy()
exact = np.arccos(np.clip(points @ points[0], -1.0, 1.0))

error = np.abs(distance.numpy() - exact)
print(f"farthest vertex: {distance.numpy().max():.3f} (exact {exact.max():.3f}, pi = {np.pi:.3f})")
print(f"mean error: {error.mean():.4f}, worst: {error.max():.4f}")  # ~0.013 and ~0.03
```

The heat method is an *approximation*: its accuracy depends on the mesh resolution and on the
diffusion time `t`. The default `t` is the squared mean edge length, the value the method's
authors recommend; libigl's `igl.heat_geodesics_*` and potpourri3d's
`MeshHeatMethodDistanceSolver` compute the same approximation. In exchange it is cheap: two
sparse solves, instead of a shortest-path search over the surface for every source.

## Many source sets on one mesh

Most of the cost is building the sparse operators (the Laplacian and mass matrix behind both
solves), and they depend on the mesh alone, not on the sources. When you need distances from many
different source sets on the *same* mesh (one per frame of an animation, one per candidate in a
sampling loop), build them once with [`heat.heat_operators`][ordito.heat.heat_operators] and pass
them back in:

```python
operators = od.heat.heat_operators(vertices, faces)

for seed in (0, 100, 200):
    sources = wp.array([seed], dtype=wp.int32, device=vertices.device)
    distance = od.heat.heat_geodesic(vertices, faces, sources, operators=operators)
    print(seed, f"{distance.numpy().max():.3f}")  # about pi from every seed
```

Several sources in one array give the distance to the *nearest* of them:
`wp.array([0, 100], dtype=wp.int32, device=...)`.

## Beyond scalar distance

The same machinery extends to tangent vectors, in [`ordito.heat`][ordito.heat]:

- [`transport_tangent_vectors`][ordito.heat.transport_tangent_vectors] parallel-transports a
  tangent vector from each source to every vertex: the direction a traveller would be facing
  after walking the shortest path while never turning.
- [`log_map`][ordito.heat.log_map] gives every vertex's position in the source vertex's tangent
  plane: the direction and distance of the shortest path from the source. It unrolls a
  neighbourhood of the source into a flat disk, which is useful for local parametrization or
  stamping a decal onto a surface.
- [`extend_scalar`][ordito.heat.extend_scalar] spreads values given at a few vertices to every
  vertex: each vertex takes roughly the value of its geodesically nearest source, blending
  smoothly where two sources compete.
- [`heat_signed_distance`][ordito.heat.heat_signed_distance] is the signed heat method: the
  distance to a set of oriented curves on the surface, positive inside the region a
  counter-clockwise curve encloses and negative outside it.

These solve a vector-valued diffusion, so their reusable operators come from
[`heat.vector_heat_operators`][ordito.heat.vector_heat_operators] rather than `heat_operators`,
passed back through the same `operators=` keyword.
