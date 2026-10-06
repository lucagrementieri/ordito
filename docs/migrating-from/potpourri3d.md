# Migrating from potpourri3d

potpourri3d (the Python binding for geometry-central) is the reference for the heat-method
family — vector heat, parallel transport, log maps, and signed heat — and ordito's
[`ordito.heat`][ordito.heat] module mirrors that scope. The main structural difference:
potpourri3d builds a stateful `*Solver` object per mesh and reuses it across calls; ordito
separates that into an explicit, cacheable operator bundle
([`heat.heat_operators`][ordito.heat.heat_operators] /
[`heat.vector_heat_operators`][ordito.heat.vector_heat_operators]) you pass back into each
solve.

```python
import potpourri3d as pp3d  # before
import ordito as od  # after
```

## Distance and vector heat

| potpourri3d | ordito |
|---|---|
| `pp3d.MeshHeatMethodDistanceSolver(V, F)` then `.compute_distance(source)` | [`heat.heat_operators(vertices, faces)`][ordito.heat.heat_operators] then [`heat.heat_geodesic(vertices, faces, sources, operators=...)`][ordito.heat.heat_geodesic] |
| `pp3d.MeshVectorHeatSolver(V, F)` then `.extend_scalar(sources, values)` | [`heat.vector_heat_operators`][ordito.heat.vector_heat_operators] then [`heat.extend_scalar`][ordito.heat.extend_scalar] |
| `.transport_tangent_vector(source, vector)` | [`heat.transport_tangent_vectors`][ordito.heat.transport_tangent_vectors] |
| `.compute_log_map(source)` | [`heat.log_map`][ordito.heat.log_map] |
| `pp3d.SignedHeatSolver(V, F)` then `.compute_distance(curve)` | [`heat.heat_signed_distance`][ordito.heat.heat_signed_distance] |
| `pp3d.edges(V, F)` | [`edges.faces_to_edges`][ordito.edges.faces_to_edges] / [`edges.edges_unique`][ordito.edges.edges_unique] |
| `pp3d.GeodesicTracer(V, F).trace_geodesic_from_vertex(...)` | [`geodesic_walk`][ordito.geodesic_walk] (straightest geodesics: walk a given distance in a given direction, batched over many walks) |
| Tangent frames used internally by the solvers | [`tangent_space.vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames] / [`face_tangent_frames`][ordito.tangent_space.face_tangent_frames] (public and directly inspectable in ordito) |

## What's different, not just renamed

- **Operators are explicit, not hidden inside a solver object.** Where potpourri3d builds a
  solver per mesh and calls methods on it, ordito returns the operators as a plain value you pass
  back into each call (see
  [Many source sets on one mesh](../cookbook/geodesic-distance.md#many-source-sets-on-one-mesh)).
- **The defaults discretize differently.** potpourri3d's solvers default to `use_robust=True`,
  which mollifies the mesh and builds an intrinsic Delaunay Laplacian; ordito defaults to the
  plain cotangent Laplacian (`use_robust=False`). To compare the two outputs, construct the
  potpourri3d solver with `use_robust=False`, or pass `use_robust=True` to
  [`heat.heat_geodesic`][ordito.heat.heat_geodesic].
- **Tangent-space quantities are gauge-dependent in both libraries.** A tangent frame from
  [`tangent_space.vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames] agrees with
  potpourri3d's own frames only up to a rotation about the normal — compare transport results
  through a gauge-invariant combination (e.g. the angle between two transported vectors), never
  their raw 2D tangent components, exactly as you would with potpourri3d's own output.
- **Isolines come back as points, not mesh elements.** potpourri3d's `marching_triangles`
  reports each isoline point as a geometry-central element index plus barycentric coordinates;
  ordito's [`intersection.marching_triangles`][ordito.intersection.marching_triangles] returns 3D
  positions directly.
