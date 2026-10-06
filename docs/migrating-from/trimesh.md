# Migrating from trimesh

ordito's object API is deliberately [trimesh](https://trimesh.org)-shaped —
[`ordito.mesh.Trimesh`][ordito.mesh.Trimesh] mirrors `trimesh.Trimesh`'s property names wherever
the underlying quantity is the same — so most of this migration is a mechanical swap of `tm` for
`od`, plus the one structural difference that runs through the whole library: everything is a
`wp.array`, not a `np.ndarray`.

```python
import trimesh as tm  # before
import ordito as od  # after
```

## Mesh construction and I/O

| trimesh | ordito |
|---|---|
| `tm.load(path)` | [`io.load_mesh(path)`][ordito.io.load_mesh] → a `warp.Mesh`, or [`io.load_mesh_data(path)`][ordito.io.load_mesh_data] for a dict of every attribute the file carries |
| `tm.Trimesh(vertices, faces)` | [`Trimesh(vertices, faces)`][ordito.mesh.Trimesh] — `vertices` a `wp.array[wp.vec3]`, `faces` a **flat** `wp.array[wp.int32]` of length `3 * n_faces` (not `(n_faces, 3)`) |
| `tm.creation.icosphere(subdivisions)` | [`creation.icosphere(subdivisions)`][ordito.creation.icosphere] |
| `tm.creation.box(extents)` | [`creation.box(extents)`][ordito.creation.box] |
| `tm.creation.icosahedron()` | [`creation.icosahedron()`][ordito.creation.icosahedron] |

## `Trimesh` properties (same names, same meaning)

| trimesh | ordito |
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

| trimesh | ordito |
|---|---|
| `trimesh.grouping.unique_rows` | [`grouping.unique_rows`][ordito.grouping.unique_rows] |
| `trimesh.triangles.area` / `mesh.area_faces` | [`triangles.face_normals_and_areas`][ordito.triangles.face_normals_and_areas] (returns normals and areas together) |
| `trimesh.sample.sample_surface(mesh, count)` | [`sample.sample_surface(vertices, faces, count)`][ordito.sample.sample_surface] |
| `trimesh.proximity.closest_point(mesh, points)` | [`proximity.closest_point_on_mesh`][ordito.proximity.closest_point_on_mesh] |
| `trimesh.proximity.signed_distance` | [`proximity.signed_distance_on_mesh`][ordito.proximity.signed_distance_on_mesh] (**opposite sign**: negative inside, where trimesh is positive inside) |
| `mesh.ray.intersects_location` | [`ray.intersects_location`][ordito.ray.intersects_location] |
| `trimesh.registration.icp` | [`registration.icp`][ordito.registration.icp] |
| `trimesh.repair.fix_normals` | [`repair.make_normals_outward`][ordito.repair.make_normals_outward] |
| `trimesh.repair.fill_holes` | [`holes.fill_min_weight`][ordito.holes.fill_min_weight], or [`repair.make_solid`][ordito.repair.make_solid] for the full repair pipeline |
| `trimesh.util.concatenate` | [`combine.concatenate`][ordito.combine.concatenate] |
| `mesh.split()` | [`combine.split`][ordito.combine.split] |
| `mesh.subdivide()` | [`remesh.subdivide`][ordito.remesh.subdivide] |
| `mesh.simplify_quadric_decimation` | [`remesh.quadric_decimate`][ordito.remesh.quadric_decimate] |

## What's different, not just renamed

- **Face buffers are flat.** `mesh.faces` in trimesh is `(n_faces, 3)`; the ordito equivalent is
  a flat `(3 * n_faces,)` buffer. Convert once at the boundary:
  `faces = wp.array(mesh_tm.faces.reshape(-1), dtype=wp.int32)`.
- **Signed distance has the opposite sign.** trimesh reports points inside the mesh as positive;
  ordito (like Open3D and most SDF code) reports them as negative. Negate one side when porting.
- **No in-place mutation.** `mesh.remove_duplicate_faces()` and similar trimesh calls mutate the
  object; ordito functions always return new arrays (e.g.
  [`repair.resolve_duplicated_faces`][ordito.repair.resolve_duplicated_faces]).
- **`isotropic_remesh` is a superset of trimesh's `subdivide_to_size`.** trimesh's crack-free
  `remesh.subdivide_to_size` only splits; ordito's
  [`remesh.isotropic_remesh`][ordito.remesh.isotropic_remesh] also collapses, flips, and smooths
  toward a uniform target length. For trimesh's split-only behaviour, use
  [`remesh.subdivide_to_size`][ordito.remesh.subdivide_to_size], which matches trimesh's
  crack-free `subdivide_to_size`.
