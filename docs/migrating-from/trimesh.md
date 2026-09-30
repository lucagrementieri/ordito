# Migrating from trimesh

triwarp's object API is deliberately [trimesh](https://trimesh.org)-shaped —
[`triwarp.mesh.Trimesh`][triwarp.mesh.Trimesh] mirrors `trimesh.Trimesh`'s property names wherever
the underlying quantity is the same — so most of this migration is a mechanical swap of `tm` for
`tw`, plus the one structural difference that runs through the whole library: everything is a
`wp.array`, not a `np.ndarray`.

```python
import trimesh as tm  # before
import triwarp as tw  # after
```

## Mesh construction and I/O

| trimesh | triwarp |
|---|---|
| `tm.load(path)` | [`io.load_mesh(path)`][triwarp.io.load_mesh] → a `warp.Mesh`, or [`io.load_mesh_data(path)`][triwarp.io.load_mesh_data] for a dict of every attribute the file carries |
| `tm.Trimesh(vertices, faces)` | [`Trimesh(vertices, faces)`][triwarp.mesh.Trimesh] — `vertices` a `wp.array[wp.vec3]`, `faces` a **flat** `wp.array[wp.int32]` of length `3 * n_faces` (not `(n_faces, 3)`) |
| `tm.creation.icosphere(subdivisions)` | [`creation.icosphere(subdivisions)`][triwarp.creation.icosphere] |
| `tm.creation.box(extents)` | [`creation.box(extents)`][triwarp.creation.box] |
| `tm.creation.icosahedron()` | [`creation.icosahedron()`][triwarp.creation.icosahedron] |

## `Trimesh` properties (same names, same meaning)

| trimesh | triwarp |
|---|---|
| `mesh.area` | `mesh.area` |
| `mesh.is_watertight` | `mesh.is_watertight` |
| `mesh.euler_number` | `mesh.euler_characteristic` |
| `mesh.volume` | `mesh.volume` |
| `mesh.center_mass` | `mesh.center_mass` |
| `mesh.moment_inertia` | `mesh.moment_inertia` |
| `mesh.face_normals` | `mesh.face_normals` |
| `mesh.vertex_normals` | `mesh.vertex_normals` |
| `mesh.edges_unique` | `mesh.edges_unique` |
| `mesh.face_adjacency` | `mesh.face_adjacency` |
| `mesh.bounds`, `mesh.extents` | `mesh.bounds`, `mesh.extents` |
| `mesh.body_count` | `mesh.body_count` |

Every one of these is lazily computed and cached on first access, exactly like trimesh's own
`@caching.cache_decorator` properties — the difference is what comes back is a `wp.array`
(or a Python scalar for a reduction like `area`/`volume`), not a NumPy array.

## Free functions (module-level, for pipelines that don't want the object)

| trimesh | triwarp |
|---|---|
| `trimesh.grouping.unique_rows` | [`grouping.unique_rows`][triwarp.grouping.unique_rows] |
| `trimesh.triangles.area` | [`triangles.face_normals_and_areas`][triwarp.triangles.face_normals_and_areas] (returns normals and areas together) |
| `trimesh.sample.sample_surface(mesh, count)` | [`sample.sample_surface(vertices, faces, count)`][triwarp.sample.sample_surface] |
| `trimesh.proximity.closest_point(mesh, points)` | [`proximity.closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh] |
| `trimesh.proximity.signed_distance` | [`proximity.signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh] |
| `mesh.ray.intersects_location` | [`ray.intersects_location`][triwarp.ray.intersects_location] |
| `trimesh.registration.icp` | [`registration.icp`][triwarp.registration.icp] |
| `trimesh.repair.fix_normals` | [`repair.make_normals_outward`][triwarp.repair.make_normals_outward] |
| `trimesh.repair.fill_holes` | [`holes.fill_min_weight`][triwarp.holes.fill_min_weight], or [`repair.make_solid`][triwarp.repair.make_solid] for the full repair pipeline |
| `trimesh.util.concatenate` | [`combine.concatenate`][triwarp.combine.concatenate] |
| `mesh.split()` | [`combine.split`][triwarp.combine.split] |
| `mesh.subdivide()` | [`remesh.subdivide`][triwarp.remesh.subdivide] |
| `mesh.simplify_quadric_decimation` | [`remesh.quadric_decimate`][triwarp.remesh.quadric_decimate] |

## What's different, not just renamed

- **Face buffers are flat.** `mesh.faces` in trimesh is `(n_faces, 3)`; the triwarp equivalent is
  a flat `(3 * n_faces,)` buffer. Convert once at the boundary:
  `faces_flat = wp.array(faces_np.reshape(-1), dtype=wp.int32)`.
- **No in-place mutation.** `mesh.remove_duplicate_faces()` and similar trimesh calls mutate the
  object; triwarp functions always return new arrays (e.g.
  [`repair.resolve_duplicated_faces`][triwarp.repair.resolve_duplicated_faces]).
- **`isotropic_remesh` is a superset of trimesh's `subdivide_to_size`.** trimesh's crack-free
  `remesh.subdivide_to_size` only splits; triwarp's
  [`remesh.isotropic_remesh`][triwarp.remesh.isotropic_remesh] also collapses, flips, and smooths
  toward a uniform target length — pass `collapse=False, swap=False, smooth=False` to get
  split-only behavior matching trimesh's function, or use
  [`remesh.subdivide_to_size`][triwarp.remesh.subdivide_to_size] directly, which triwarp also
  ships.
