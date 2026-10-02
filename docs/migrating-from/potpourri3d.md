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
| Tangent frames used internally by the solvers | [`tangent_space.vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames] / [`face_tangent_frames`][ordito.tangent_space.face_tangent_frames] (public and directly inspectable in ordito) |

## What's different, not just renamed

- **Operators are explicit, not hidden inside a solver object.** potpourri3d's `use_robust` and
  `use_intrinsic_delaunay` mollification flags are constructor arguments to a solver you keep
  around; ordito exposes the equivalent behind
  [`heat.heat_operators`][ordito.heat.heat_operators]'s own keyword, and the operators it
  returns are a plain value you can cache yourself (see
  [Geodesic distance fields](../cookbook/geodesic-distance.md#reusing-the-operators-across-many-source-sets)
  in the cookbook) — nothing solver-shaped to construct once and mutate.
- **Tangent-space quantities are gauge-dependent in both libraries.** A tangent frame from
  [`tangent_space.vertex_tangent_frames`][ordito.tangent_space.vertex_tangent_frames] agrees with
  potpourri3d's own frames only up to a rotation about the normal — compare transport results
  through a gauge-invariant combination (e.g. the angle between two transported vectors), never
  their raw 2D tangent components, exactly as you would with potpourri3d's own output.
- **`log_map`'s barycentric output is decoded directly as a ordito array**, not as
  potpourri3d's geometry-central element-index pairs — see
  [`heat.log_map`][ordito.heat.log_map]'s own docstring for the returned convention.
