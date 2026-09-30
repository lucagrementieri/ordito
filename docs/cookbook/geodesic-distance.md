# Geodesic distance fields

[`heat.heat_geodesic`][triwarp.heat.heat_geodesic] approximates geodesic distance over a mesh
surface with Crane et al.'s heat method: diffuse heat briefly from a set of source vertices,
normalize the resulting gradient into a unit direction field, then integrate that field back into
a distance by solving a second sparse system. Both solves run on-device via conjugate gradient.

```python
import numpy as np
import warp as wp

import triwarp as tw

vertices, faces = tw.creation.icosphere(subdivisions=3)
sources = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=vertices.device)

distance = tw.heat.heat_geodesic(vertices, faces, sources)
print(distance.numpy().min(), distance.numpy().max())  # 0.0 3.09... (a bit under pi, as expected
# on a unit-radius sphere)
```

The result is an *approximation* — typically a few percent off the true geodesic distance,
matching `igl::heat_geodesics` — which is the right trade for how much cheaper it is than an exact
shortest-path search over the whole mesh.

## Reusing the operators across many source sets

Assembling the two sparse operators the heat method needs (the Laplacian for diffusion, and the
Poisson system for integration) is the expensive part; solving with a different source set is
comparatively cheap. If you need distance from many different sources on the *same* mesh — one
per frame of an animation, one per candidate seed in a sampling loop — precompute the operators
once with [`heat.heat_operators`][triwarp.heat.heat_operators] and pass them back in:

```python
operators = tw.heat.heat_operators(vertices, faces)

for seed_index in range(vertices.shape[0]):
    sources = wp.array(np.array([seed_index], dtype=np.int32), device=vertices.device)
    distance = tw.heat.heat_geodesic(vertices, faces, sources, operators=operators)
    ...
```

## Beyond scalar distance

The same heat-method machinery extends to vector- and frame-valued fields, in
[`triwarp.heat`][triwarp.heat]:

- [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors] — parallel-transports a
  tangent vector field outward from a source set, the vector generalization of geodesic distance.
- [`log_map`][triwarp.heat.log_map] — the geodesic log map from a source point: for every vertex,
  the direction and distance of the shortest path back to the source, in the source's own tangent
  frame. Useful for unrolling a local neighbourhood into a flat disk (e.g. for texture painting).
- [`heat_signed_distance`][triwarp.heat.heat_signed_distance] — the signed heat method, for a
  distance field that also carries an inside/outside sign.

All three share the same `operators=` precomputation pattern above.
