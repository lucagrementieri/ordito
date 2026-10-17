# Getting started

## Install

=== "pip"

    ```bash
    pip install ordito
    ```

=== "uv"

    ```bash
    uv add ordito
    ```

Requires Python ≥ 3.11 and `warp-lang` ≥ 1.18. A CUDA-capable GPU is recommended but not required —
every function also runs on Warp's CPU backend, with the same code. Warp's PyPI wheels are
built with CUDA 13, so the GPU path needs an NVIDIA R580-series or newer driver and a Turing
(`sm_75`) or newer GPU; Warp publishes CUDA 12 wheels (`+cu12`) on its GitHub releases for older
drivers. Mesh file I/O via [meshio](https://github.com/nschloe/meshio) is an optional extra:
`pip install "ordito[io]"`.

### Platform support

ordito ships as a pure-Python `py3-none-any` wheel; everything platform-specific lives in
`warp-lang`, which brings its own compiled backend. What differs between the rows below is how
much of it is *verified*, not how much of it is meant to work.

| Platform | Status |
|---|---|
| **Linux + CUDA** | The developed and measured target. Every performance figure in these docs is this configuration, and the full test suite plus the eleven-library comparison suite run here on every release. |
| **Linux, CPU only** | Fully supported and correctness-tested on every change, via Warp's CPU backend. Not performance-tuned: the CPU backend runs a launch grid far less parallel than a GPU does, and tuning decisions in ordito are made on the CUDA measurement. |
| **macOS** | Untested. `warp-lang` supports macOS on the CPU backend (there is no CUDA, and Metal is not a Warp backend), so ordito is expected to work, but nothing verifies it. |
| **Windows** | Untested. Same position as macOS: `warp-lang` supports it, ordito is not checked against it. |

The untested rows are a statement about coverage, not a warning — if ordito works or fails for
you on either, a short [issue](https://github.com/lucagrementieri/ordito/issues) saying so is
what would move them up the table.

## The five-minute mental model

ordito has no scene graph, no viewer, and no mandatory mesh object. A mesh is two
[`warp.array`](https://nvidia.github.io/warp/modules/runtime.html#arrays) buffers — vertex
positions and a flat triangle index buffer — and every function is a plain call that takes
arrays and returns arrays:

```python
import ordito as od

vertices, faces = od.creation.icosphere(subdivisions=4)
# vertices: wp.array[wp.vec3], shape (2562,)
# faces:    wp.array[wp.int32], shape (15360,) -- flat (3 * n_faces,), not (n_faces, 3)
```

That flat, `(3 * n_faces,)` face layout (rather than `(n_faces, 3)`) is the one convention that
differs from most mesh libraries you may already know, and it's worth internalizing early —
[Concepts](concepts.md) explains why, and [Migrating from another library](migrating-from/index.md)
maps the functions you already reach for onto their ordito equivalents.

An optional [`Trimesh`][ordito.mesh.Trimesh] class wraps that same pair of arrays and caches
derived quantities (normals, adjacency, boundary loops, manifoldness predicates) the first time
each is asked for:

```python
mesh = od.Trimesh(vertices, faces)
print(f"{mesh.area:.4f}")  # 12.5514
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

import ordito as od

vertices, faces = od.creation.icosphere(subdivisions=3)

# Punch a hole (drop faces whose centroid sits in a small polar cap)...
centroids = od.triangles.face_centroids(vertices, faces).numpy()
keep_mask = wp.array(centroids[:, 2] < 0.9, dtype=wp.bool, device=vertices.device)
holed_vertices, holed_faces = od.selection.submesh_from_face_mask(vertices, faces, keep_mask)

# ...and glue on an unrelated speck of debris, the way a scanner might pick up a stray fragment.
debris_vertices, debris_faces = od.creation.tetrahedron(device=vertices.device)
debris_vertices = wp.array(
    debris_vertices.numpy() * 0.05 + np.array([3.0, 0.0, 0.0], dtype=np.float32),
    dtype=wp.vec3,
    device=vertices.device,
)
broken_vertices, broken_faces = od.combine.concatenate(
    [(holed_vertices, holed_faces), (debris_vertices, debris_faces)]
)
print(od.validation.is_watertight(broken_vertices, broken_faces))  # False

# One call: drop small components, close the hole, and clean up what's left.
repaired_vertices, repaired_faces = od.repair.make_solid(
    broken_vertices, broken_faces, keep_largest=True
)
print(od.validation.is_watertight(repaired_vertices, repaired_faces))  # True

# Feature-preserving isotropic remeshing (split / collapse / flip / smooth / reproject).
remeshed_vertices, remeshed_faces = od.remesh.isotropic_remesh(
    repaired_vertices, repaired_faces, target_length=0.2
)
```

Every intermediate value here is a `wp.array` on whatever device the input arrived on (CUDA if
one is present) — nothing crossed back to the host between `submesh_from_face_mask`,
`concatenate`, `make_solid`, and `isotropic_remesh`. That's the "device follows the data" design
principle in practice; see [Concepts](concepts.md) for the rest of it.

## Getting data in and out

ordito computes on Warp arrays, and everything else in Python speaks NumPy, files or PyTorch
tensors. The conversions happen at the edges of a pipeline, once each way.

**From NumPy, or any library that hands you NumPy arrays** (trimesh, Open3D, libigl, PyVista):
[`io.mesh_from_numpy`][ordito.io.mesh_from_numpy] accepts `(n_vertices, 3)` positions and
`(n_faces, 3)` faces of any dtype and returns a [`Trimesh`][ordito.mesh.Trimesh]. Going back,
`.numpy()` copies an array to the host, and `reshape(-1, 3)` restores the row layout of faces:

```python
import numpy as np
import trimesh

import ordito as od

mesh_tm = trimesh.creation.icosphere(subdivisions=2)
mesh = od.io.mesh_from_numpy(mesh_tm.vertices, mesh_tm.faces)
print(mesh.n_vertices, mesh.n_faces)  # 162 320

faces_rows = mesh.faces.numpy().reshape(-1, 3)  # (n_faces, 3), as trimesh stores them
print(np.array_equal(faces_rows, mesh_tm.faces))  # True
```

**From a file**: install the I/O extra (`pip install "ordito[io]"`) and call
[`io.load_mesh_data`][ordito.io.load_mesh_data], which reads any format
[meshio](https://github.com/nschloe/meshio) does (OBJ, PLY, STL, OFF, VTK, glTF, ...) and returns
the vertices, faces and whatever per-vertex attributes the file carries. ordito has no writer of
its own; meshio writes the arrays back out:

```python
import pathlib
import tempfile

import meshio

import ordito as od

vertices, faces = od.creation.torus(major_radius=1.0, minor_radius=0.3)
with tempfile.TemporaryDirectory() as folder:
    path = pathlib.Path(folder) / "torus.ply"
    cells = [("triangle", faces.numpy().reshape(-1, 3))]
    meshio.write(path, meshio.Mesh(vertices.numpy(), cells))
    data = od.io.load_mesh_data(path)
print(data["vertices"].shape == vertices.shape, data["faces"].shape == faces.shape)  # True True
```

**From and to PyTorch**, without a copy: `wp.from_torch` wraps a tensor's memory as a Warp array,
and `wp.to_torch` does the reverse. The two views share storage, so a training loop can call
ordito on its tensors directly:

```python
import torch
import warp as wp

import ordito as od

points_torch = torch.rand(1_000, 3, device=str(wp.get_device()))
points = wp.from_torch(points_torch, dtype=wp.vec3)  # no copy: shape (1_000,) of vec3
lower, upper = od.bounds.aabb(points)
print(all(0.0 <= value <= 1.0 for value in (*lower, *upper)))  # True
```

## Where to next

- **[Concepts](concepts.md)** — the four ideas worth understanding before writing anything
  nontrivial: arrays in/arrays out, the device follows the data, host syncs are budgeted, and
  everything is measured rather than assumed.
- **[Examples](examples/index.md)** — a gallery of runnable examples with the images they
  produce: cleaning a scan, point clouds to watertight surfaces, geodesic distance fields,
  aligning two scans, and many more.
- **[Migrating from another library](migrating-from/index.md)** — already know trimesh, libigl,
  Open3D, MeshLab, potpourri3d, or PyTorch3D? Start here for a function-by-function map.
- **[API Reference](reference.md)** — a task index and a map of every module; each module page
  documents every function with a runnable example.
