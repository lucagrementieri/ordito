# Cleaning and remeshing a rough mesh

A mesh coming from a 3D scan or a download rarely arrives watertight: it usually has a hole where
the scanner couldn't see, and often carries a stray disconnected fragment of debris alongside the
real surface. This recipe builds a small stand-in for exactly that — a sphere with a hole and an
unrelated speck of geometry glued on — and runs it through the repair-then-remesh pipeline you'd
point at a real one.

```python
import numpy as np
import warp as wp

import ordito as od

vertices, faces = od.creation.icosphere(subdivisions=3)

# Punch a hole: drop every face whose centroid lands in a small polar cap.
centroids = od.triangles.face_centroids(vertices, faces).numpy()
keep_mask = wp.array(centroids[:, 2] < 0.9, dtype=wp.bool, device=vertices.device)
holed_vertices, holed_faces = od.selection.submesh_from_face_mask(vertices, faces, keep_mask)

# Glue on a small, unrelated fragment -- the kind of stray debris a scanner can pick up.
debris_vertices, debris_faces = od.creation.tetrahedron(device=vertices.device)
debris_vertices = wp.array(
    debris_vertices.numpy() * 0.05 + np.array([3.0, 0.0, 0.0], dtype=np.float32),
    dtype=wp.vec3,
    device=vertices.device,
)
broken_vertices, broken_faces = od.combine.concatenate(
    [(holed_vertices, holed_faces), (debris_vertices, debris_faces)]
)

print("faces before repair:", broken_faces.shape[0] // 3)  # 1222
print("watertight before:  ", od.validation.is_watertight(broken_vertices, broken_faces))  # False

# One call: drop the small disconnected component, close the remaining hole, and clean up
# whatever degeneracies or self-intersections that leaves behind.
repaired_vertices, repaired_faces = od.repair.make_solid(
    broken_vertices, broken_faces, keep_largest=True
)
print("faces after repair: ", repaired_faces.shape[0] // 3)  # 1242
print(
    "watertight after:   ", od.validation.is_watertight(repaired_vertices, repaired_faces)
)  # True

# Now that the surface is solid, drive every edge length toward a uniform target -- splitting
# long edges, collapsing short ones, flipping toward ideal vertex valence, and tangentially
# smoothing and reprojecting the result back onto the repaired surface.
remeshed_vertices, remeshed_faces = od.remesh.isotropic_remesh(
    repaired_vertices, repaired_faces, target_length=0.2
)
print("faces after remesh: ", remeshed_faces.shape[0] // 3)  # ~700 (varies slightly on CUDA)
print(
    "watertight remeshed:", od.validation.is_watertight(remeshed_vertices, remeshed_faces)
)  # True
```

## What `make_solid` actually does

[`repair.make_solid`][ordito.repair.make_solid] is not one opaque operation: it runs smaller
public functions in a deliberate order, and when only one stage applies to your mesh you can call
that stage directly.

1. **Validate the faces.** Every index must name an existing vertex, or it raises `ValueError`:
   everything after this indexes vertices without a bounds check.
2. **Connectivity repair**, the cleanup most mesh *loaders* do without telling you:
   [`remove_unreferenced_vertices`][ordito.repair.remove_unreferenced_vertices],
   [`make_winding_consistent`][ordito.repair.make_winding_consistent] and
   [`split_non_manifold_vertices`][ordito.repair.split_non_manifold_vertices].
3. **Choose what to keep.** `keep_largest=True` (the default) runs
   [`remove_small_components`][ordito.repair.remove_small_components], which is what drops the
   debris above. `join_components=True` runs
   [`holes.join_closest_components`][ordito.holes.join_closest_components] instead, for input
   that is *meant* to be one surface but arrived in several pieces.
4. **Fill every hole** with [`holes.fill_min_weight`][ordito.holes.fill_min_weight].
5. **Clean up, until nothing changes** (at most `max_iter` rounds):
   [`remove_degenerate_faces`][ordito.repair.remove_degenerate_faces],
   [`collapse_small_triangles`][ordito.repair.collapse_small_triangles] and
   [`fix_self_intersections`][ordito.repair.fix_self_intersections]. Cutting out a
   self-intersection can open new holes or split off a fragment, so the small-component filter
   runs again inside the loop.
6. **Fill once more, last.** This closes the holes stage 5 opened. Nothing runs after it, because
   a cleanup pass after a fill can delete the sliver that closed a small hole and reopen it.

`make_solid` has no success flag: check the result with
[`validation.is_watertight`][ordito.validation.is_watertight], as the recipe does.

## Tuning the remesh

`target_length` can be a single float (as above) or a per-vertex `wp.array[wp.float32]` sizing
field, for a mesh that should stay coarse in flat regions and fine near detail. `feature_angle`
(default 30°) controls which edges are treated as creases and preserved rather than smoothed away;
see [`remesh.isotropic_remesh`][ordito.remesh.isotropic_remesh] for the full set of per-stage
toggles (`split`, `collapse`, `swap`, `smooth`, `reproject`) and `max_deviation` for bounding how
far the reprojection is allowed to move a vertex from the original surface.
