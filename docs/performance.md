# Performance

ordito exists because mesh processing is embarrassingly parallel and almost every CPU-bound
geometry library leaves that parallelism on the table. This page explains *why* the GPU path is
fast in the shapes that matter, and how to reproduce every number this project publishes —
it deliberately does not repeat this project's own internal, hardware-pinned engineering notes
verbatim: a number is only worth quoting here alongside the mesh, the library, and the hardware it
was measured against, and it is kept current rather than copied once and left to go stale.

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
  parametrization, the heat method, and screened-Poisson reconstruction all assemble their
  operators and run their conjugate-gradient solves without a single host readback in the loop —
  the CPU only sees a value once the solve has actually converged.
- **The same code runs on Warp's CPU backend when no GPU is present.** There is no separate
  code path to fall back to — correctness doesn't depend on which device you happen to be running
  on, only speed does.

## How to reproduce a number

Every timing this project publishes — in its README, in a release's changelog, or on the
[Benchmarks](benchmarks.md) page — comes from `benchmarks/`, built on
[pytest-benchmark](https://pytest-benchmark.readthedocs.io) and run against nine established CPU
geometry-processing libraries plus PyTorch3D's own CUDA kernels (the one reference with a GPU path
of its own, and so the suite's only GPU-against-GPU comparison). Anyone can run the same
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

## What "fast" means here

A speed claim on this project is only meaningful next to three things: **which library** it beats,
**on what mesh** (size and shape both matter — connected-component count, boundary length, and
triangle aspect ratio move the needle as often as raw triangle count does), and **on what
hardware**. A bare "GPU-accelerated" adjective is not a number, and a number with no mesh or
hardware attached is not reproducible. Every claim in this project's own documentation follows
that discipline; hold any number quoted elsewhere about ordito to the same standard before
trusting it.

## Where ordito is not the fastest option

Not every operation benefits from a GPU. A handful of ordito's own functions are launch-bound
rather than compute-bound at realistic mesh sizes — a short, sequential DP over one small
boundary loop, or a per-mesh CPU seed step a parallel algorithm can't usefully replace — and lose
to a fast single-threaded C++ implementation there. This isn't hidden: where it's true, it's
measured and stated rather than smoothed over, exactly like every other number on this page.
