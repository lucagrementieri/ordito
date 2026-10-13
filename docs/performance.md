# Performance

This page starts with what to do in your own code to get the speed ordito is built for, then
explains where that speed comes from and how to reproduce the published comparisons.

## Practical tips

**The first call into a module compiles its kernels.** Warp compiles each kernel module to
native code the first time a function needs it on a device, and caches the result on disk
(`~/.cache/warp` by default; `wp.config.kernel_cache_dir` moves it). Right after installing or
upgrading, the first call of a function takes about a second for each kernel module it touches
that has not been compiled yet, a few seconds for a pipeline such as remeshing. Every later call,
in this process or any later one, loads the cached binary in milliseconds. When you time something, time the second call. In a service or a container, warm
the cache once at build time by running the pipeline on a small input, and keep the cache
directory.

**Keep data on the device between calls.** Every function returns arrays on the device its
inputs came from, so a chain of calls never leaves the GPU. A `.numpy()`, `float(...)` or
`print(...)` in the middle of a loop makes the host wait for the GPU each time; move it after the
loop.

**Pass what you already know.** Some functions read a value back from the device only to size an
output (the vertex count, from the largest face index). Where you know it, pass it: many
topology functions take `n_vertices=` (see [Concepts](concepts.md#host-syncs-are-budgeted)).

**Reuse a [`Trimesh`][ordito.mesh.Trimesh] for repeated work on one mesh.** Its derived
quantities are computed once and cached. The sparse operators behind the heat method and the
parametrizations are kept on the mesh too, and from the second solve on a sparse Cholesky
factorization of them, so a loop over many source sets on one mesh pays the setup once (see
[Geodesic distance fields](cookbook/geodesic-distance.md#many-source-sets-on-one-mesh)).

**Small meshes do not need a GPU.** Each call costs tens of microseconds of launch overhead
whatever the mesh size, so on a mesh of a few hundred triangles the CPU device can be as fast.
The GPU wins as soon as there is enough work to fill it, from tens of thousands of elements up.

## Why the GPU path is fast

- **Every function is array-in, array-out on a device buffer**, so a pipeline of several calls
  never round-trips through the host between them (see
  [Concepts: the device follows the data](concepts.md#the-device-follows-the-data)). A CPU
  library built around NumPy pays a Python-loop or a memory-layout cost between every step that
  isn't itself vectorized; ordito pays it once, at the boundary where you actually need values
  back on the host.
- **One kernel launch does the work of a whole loop.** A per-triangle or per-vertex computation —
  a normal, an area, a Laplacian entry, a nearest-neighbour query — launches as a single batched
  kernel over every element at once, rather than as a Python (or even a vectorized-but-still-
  single-threaded) loop.
- **Sparse linear solves stay on the device.** Cotangent Laplacians, harmonic and LSCM
  parametrization, the heat method and screened-Poisson reconstruction assemble their operators
  on the GPU and run their conjugate-gradient iterations there. On CUDA the iteration loop is
  recorded once as a CUDA graph that tests convergence on the device, so the host does not wait
  on every iteration.
- **The same code runs on Warp's CPU backend when no GPU is present.** The test suite runs on
  both devices, so correctness does not depend on which one you use, only speed does. (Results can
  differ in the last bits: floating-point sums on a GPU may add terms in a different order.)

## How to reproduce a number

Every timing this project publishes — in its README, in a release's changelog, or on the
[Benchmarks](benchmarks.md) page — comes from `benchmarks/`, built on
[pytest-benchmark](https://pytest-benchmark.readthedocs.io) and run against nine established
geometry-processing libraries. Eight run on the CPU; PyTorch3D also has CUDA kernels, so it is
timed on both devices and is the suite's only GPU-against-GPU comparison. Anyone can run the same
comparison:

```bash
git clone https://github.com/lucagrementieri/ordito
cd ordito
uv sync --group bench
uv run pytest benchmarks/ --benchmark-json=results.json
```

See [`benchmarks/README.md`](https://github.com/lucagrementieri/ordito/tree/main/benchmarks) in
the repository for the full methodology: which axis each benchmark group sweeps, which libraries
are compared for which operation, and why a handful of comparisons are excluded (a different
algorithm answering a related-but-not-identical question, a parameter one library's API doesn't
expose — every exclusion is written down at the point it's made, not left implicit).

See [Benchmarks](benchmarks.md) for a curated set of these comparisons rendered as charts, each
library named by its own logo.

## Where ordito is not the fastest option

A speed claim means something only next to the library it is compared with, the mesh (its size,
but also its number of components, its boundary length and its triangle shapes) and the hardware.
Not every operation benefits from a GPU. A few functions have little parallel work at realistic
sizes, such as a sequential dynamic program over one small boundary loop, and a fast
single-threaded C++ implementation beats them. The [Benchmarks](benchmarks.md) page shows those
cases alongside the others.
