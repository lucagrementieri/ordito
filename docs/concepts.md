# Concepts

Four ideas shape every function in ordito. None of them are optional design flourishes — each
one is a constraint the whole library is built against, and understanding them up front saves
having to reverse-engineer them from a confusing signature later.

## Arrays in, arrays out

The core API is free functions over [`warp.array`](https://nvidia.github.io/warp/modules/runtime.html#arrays)
buffers. There is no mandatory mesh object and no hidden state:

```python
vertices, faces = od.creation.icosphere(subdivisions=3)
areas = od.triangles.face_areas(vertices, faces)  # a plain wp.array[wp.float32], nothing else
```

A mesh is always the same two things — a `wp.array[wp.vec3]` of vertex positions and a
**flat** `wp.array[wp.int32]` of length `3 * n_faces` — never a `(n_faces, 3)` array and never an
object with methods. That flat layout is deliberate: it's the shape a Warp kernel indexes without
a stride computation, and it's the one convention every function in the package agrees on, so a
buffer produced by one function is always a valid argument to the next.

The optional [`Trimesh`][ordito.mesh.Trimesh] class is a thin wrapper around exactly this pair,
with derived quantities (normals, adjacency, boundary loops) computed lazily and cached on first
access. It exists for convenience, not because the free functions need it — every one of
`Trimesh`'s properties is also a public function you can call directly on raw arrays.

**Why not NumPy?** A `wp.array` is a real device buffer — on CUDA, an actual allocation on the
GPU; the round trip through `.numpy()` is a real host readback with a real cost. Passing NumPy
arrays as the primary interface would mean paying that cost at *every* function boundary rather
than only where a caller actually needs the values back on the host (to print them, save them, or
hand them to a plotting library). `wp.array(numpy_array, dtype=...)` and `warp_array.numpy()` are
the two directions across that boundary; reach for them exactly as often as your pipeline needs
data on the host and no more.

## The device follows the data

There is no global device switch. Every kernel launch and every allocation inherits the device of
its own input arrays:

```python
import warp as wp

cpu_vertices, cpu_faces = od.creation.icosphere(subdivisions=2, device="cpu")
gpu_vertices, gpu_faces = od.creation.icosphere(subdivisions=2, device="cuda:0")

od.triangles.face_areas(cpu_vertices, cpu_faces)  # runs on the CPU backend
od.triangles.face_areas(gpu_vertices, gpu_faces)  # runs on cuda:0
```

A function that allocates a *new* array without being given one to place it on (a primitive
constructor like `od.creation.icosphere`) takes a `device=` keyword and defaults to Warp's current
device — `wp.set_device("cuda:0")` (or a `wp.ScopedDevice`) sets that default for a whole block of
code, same as any other Warp program.

This is what makes chaining functions cheap: a pipeline of ten calls with no explicit `device=`
anywhere runs entirely on one device, because each function's output already carries the device
its input arrived on, and the next call reads it from there.

## Host syncs are budgeted

A device-to-host readback (`.numpy()`, or reading a single scalar off a device array) is a real
synchronization point — the GPU has to finish everything queued before it, and the value has to
physically cross the PCIe bus. Wrappers avoid these wherever the computation doesn't need one, and
where one genuinely is unavoidable (typically because a buffer's *size* depends on a value only
the device knows, like how many triangles survived a filter), the function's docstring says so.

Some functions expose a keyword that lets a caller skip the inference behind such a readback,
when the caller already knows the bound:

```python
# Without n_vertices, the function reads back a count to size an internal table.
adjacency = od.adjacency.face_adjacency(faces)

# If you already know how many vertices the mesh has, pass it and skip that readback.
adjacency = od.adjacency.face_adjacency(faces, n_vertices=vertices.shape[0])
```

This matters most inside a loop — a saved readback is trivial once, and adds up over a thousand
iterations of a solver or a remeshing pass.

## Measured, not assumed

Every public module has both a test file, comparing its output against an established CPU
geometry-processing library (trimesh, libigl, Open3D, MeshLab, potpourri3d, PyTorch3D, and
others — see [`ordito.validation`][ordito.validation]-style parity in the test suite), and a
benchmark file. A performance change lands only with a before/after measurement behind it, and a
correctness change is asserted on actual values, never on shape or "did it run" alone. The
[Performance](performance.md) page and [Benchmarks](https://github.com/lucagrementieri/ordito/tree/main/benchmarks)
in the repository are where that discipline is visible from the outside — every number quoted
there is reproducible with the same commands the library's own test suite uses.

## Where to next

- **[Cookbook](cookbook/index.md)** — these four ideas applied to real tasks.
- **[Migrating from another library](migrating-from/index.md)** — how these conventions map onto
  the API you already know.
