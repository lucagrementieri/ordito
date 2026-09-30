# Getting started

## Install

```bash
uv add triwarp
```

or with `pip`:

```bash
pip install triwarp
```

Requires Python ≥ 3.11 and `warp-lang` ≥ 1.17. A CUDA-capable GPU is recommended but not
required — every function also runs on Warp's CPU backend, same code, same results. Mesh file
I/O via [meshio](https://github.com/nschloe/meshio) is an optional extra:
`pip install triwarp[io]`.

### Platform support

triwarp ships as a pure-Python `py3-none-any` wheel; everything platform-specific lives in
`warp-lang`, which brings its own compiled backend. What differs between the rows below is how
much of it is *verified*, not how much of it is meant to work.

| Platform | Status |
|---|---|
| **Linux + CUDA** | The developed and measured target. Every performance figure in these docs is this configuration, and the full test suite plus the eleven-library comparison suite run here on every release. |
| **Linux, CPU only** | Fully supported and correctness-tested on every change, via Warp's CPU backend. Not performance-tuned: the CPU backend runs a launch grid far less parallel than a GPU does, and tuning decisions in triwarp are made on the CUDA measurement. |
| **macOS** | Untested. `warp-lang` supports macOS on the CPU backend (there is no CUDA, and Metal is not a Warp backend), so triwarp is expected to work, but nothing verifies it. |
| **Windows** | Untested. Same position as macOS: `warp-lang` supports it, triwarp is not checked against it. |

The untested rows are a statement about coverage, not a warning — if triwarp works or fails for
you on either, a short [issue](https://github.com/lucagrementieri/triwarp/issues) saying so is
what would move them up the table.

## The five-minute mental model

triwarp has no scene graph, no viewer, and no mandatory mesh object. A mesh is two
[`warp.array`](https://nvidia.github.io/warp/modules/runtime.html#arrays) buffers — vertex
positions and a flat triangle index buffer — and every function is a plain call that takes
arrays and returns arrays:

```python
import triwarp as tw

vertices, faces = tw.creation.icosphere(subdivisions=4)
# vertices: wp.array[wp.vec3], shape (2562,)
# faces:    wp.array[wp.int32], shape (15360,) -- flat (3 * n_faces,), not (n_faces, 3)
```

That flat, `(3 * n_faces,)` face layout (rather than `(n_faces, 3)`) is the one convention that
differs from most mesh libraries you may already know, and it's worth internalizing early —
[Concepts](concepts.md) explains why, and [Migrating from another library](migrating-from/index.md)
maps the functions you already reach for onto their triwarp equivalents.

An optional [`Trimesh`][triwarp.mesh.Trimesh] class wraps that same pair of arrays and caches
derived quantities (normals, adjacency, boundary loops, manifoldness predicates) the first time
each is asked for:

```python
mesh = tw.Trimesh(vertices, faces)
print(mesh.area)  # 12.551353454589844
print(mesh.is_watertight)  # True
print(mesh.euler_characteristic)  # 2
```

Nothing about the free functions requires it — reach for `Trimesh` when an object is more
convenient than threading `(vertices, faces)` through every call, and drop back to arrays when a
pipeline benefits from staying explicit about what's being computed.

## A first pipeline

A realistic input is rarely as clean as a freshly generated icosphere. The snippet below builds a
stand-in for one — punches a hole in the sphere, then bolts on a small piece of unrelated
debris — and runs it through the same repair-then-remesh pipeline you'd point at a raw scan:

```python
import numpy as np
import warp as wp

import triwarp as tw

vertices, faces = tw.creation.icosphere(subdivisions=3)

# Punch a hole (drop faces whose centroid sits in a small polar cap)...
centroids = tw.triangles.face_centroids(vertices, faces).numpy()
keep_mask = wp.array(centroids[:, 2] < 0.9, dtype=wp.bool, device=vertices.device)
holed_vertices, holed_faces = tw.selection.submesh_from_face_mask(vertices, faces, keep_mask)

# ...and glue on an unrelated speck of debris, the way a scanner might pick up a stray fragment.
debris_vertices, debris_faces = tw.creation.tetrahedron(device=vertices.device)
debris_vertices = wp.array(
    debris_vertices.numpy() * 0.05 + np.array([3.0, 0.0, 0.0], dtype=np.float32),
    dtype=wp.vec3,
    device=vertices.device,
)
broken_vertices, broken_faces = tw.combine.concatenate(
    [(holed_vertices, holed_faces), (debris_vertices, debris_faces)]
)
print(tw.validation.is_watertight(broken_vertices, broken_faces))  # False

# One call: drop small components, close the hole, and clean up what's left.
repaired_vertices, repaired_faces = tw.repair.make_solid(
    broken_vertices, broken_faces, keep_largest=True
)
print(tw.validation.is_watertight(repaired_vertices, repaired_faces))  # True

# Feature-preserving isotropic remeshing (split / collapse / flip / smooth / reproject).
remeshed_vertices, remeshed_faces = tw.remesh.isotropic_remesh(
    repaired_vertices, repaired_faces, target_length=0.2
)
```

Every intermediate value here is a `wp.array` on whatever device the input arrived on (CUDA if
one is present) — nothing crossed back to the host between `submesh_from_face_mask`,
`concatenate`, `make_solid`, and `isotropic_remesh`. That's the "device follows the data" design
principle in practice; see [Concepts](concepts.md) for the rest of it.

## Where to next

- **[Concepts](concepts.md)** — the four ideas worth understanding before writing anything
  nontrivial: arrays in/arrays out, the device follows the data, host syncs are budgeted, and
  everything is measured rather than assumed.
- **[Cookbook](cookbook/index.md)** — task-oriented recipes: cleaning a scan, point clouds to
  watertight surfaces, geodesic distance fields, aligning two scans.
- **[Migrating from another library](migrating-from/index.md)** — already know trimesh, libigl,
  Open3D, MeshLab, potpourri3d, or PyTorch3D? Start here for a function-by-function map.
- **[API Reference](api/triwarp/creation.md)** — every public module, generated from its own docstrings.
