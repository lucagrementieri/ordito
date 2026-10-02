<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/lucagrementieri/ordito/main/docs/assets/logo-lockup-dark.svg">
  <img src="https://raw.githubusercontent.com/lucagrementieri/ordito/main/docs/assets/logo-lockup.svg" alt="ordito" height="72">
</picture>

[![PyPI](https://img.shields.io/pypi/v/ordito)](https://pypi.org/project/ordito/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/ordito/)
[![Docs](https://img.shields.io/badge/docs-lucagrementieri.github.io%2Fordito-blue)](https://lucagrementieri.github.io/ordito/)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/lucagrementieri/208b2ea8a3b764ca7c10d82195533738/raw/ordito-coverage.json)](#development)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green)](#license)
[![Status](https://img.shields.io/badge/status-beta-yellow)](#status)

**ordito puts mesh processing on the GPU.** Geometry, topology, repair, remeshing, spatial
queries, discrete differential operators, geodesics, point-cloud reconstruction, and
registration — as plain array-in / array-out functions backed by
[NVIDIA Warp](https://github.com/NVIDIA/warp) kernels, with a
[trimesh](https://trimesh.org)-inspired API. Every function runs on CUDA when a GPU is available
and falls back to CPU otherwise — same code, same results.

## Highlights

- **One dependency.** The runtime depends on `warp-lang` alone; NumPy is only needed to move
  data in and out.
- **Broad coverage.** 50 public modules and over 480 functions spanning primitives, topology,
  repair, remeshing, spatial queries, discrete differential operators, geodesics, sampling,
  surface reconstruction, and registration.
- **Stays on the device.** Functions take Warp arrays and return Warp arrays, so pipelines
  compose without host round-trips; device placement follows the inputs.
- **Optional `Trimesh` object.** A frozen mesh class with lazily cached derived quantities
  (normals, adjacency, boundary loops, manifoldness and watertightness predicates, BVH) for
  when an object API is more convenient than free functions.
- **Fully typed** (`py.typed`), documented per module at
  <https://lucagrementieri.github.io/ordito/>, and validated function-by-function against eleven
  reference implementations — nine established geometry-processing libraries plus SciPy and
  NumPy (see below).

## Learn more

- **[Getting started](https://lucagrementieri.github.io/ordito/getting-started/)** — install, the
  mental model, and a first repair-then-remesh pipeline.
- **[Cookbook](https://lucagrementieri.github.io/ordito/cookbook/)** — task-oriented recipes:
  cleaning a scan, point clouds to watertight surfaces, geodesic distance fields, aligning two
  scans.
- **[Migrating from another library](https://lucagrementieri.github.io/ordito/migrating-from/)**
  — already know trimesh, libigl, Open3D, MeshLab, potpourri3d, or PyTorch3D? Start here for a
  function-by-function map.
- **[Performance](https://lucagrementieri.github.io/ordito/performance/)** — why the GPU path is
  fast, and how to check any speed claim yourself.

## One GPU library instead of nine

Mesh processing in Python has long meant stitching together several excellent — but mostly
CPU-bound and stylistically different — libraries. ordito consolidates the functionality it needs
from each of them behind a single GPU-accelerated API:

| Replaces | For | In ordito |
|---|---|---|
| [trimesh](https://github.com/mikedh/trimesh) | Mesh bookkeeping: edges, adjacency, boundary, validation, primitives, sampling, proximity | `edges`, `adjacency`, `boundary`, `validation`, `creation`, `sample`, `proximity`, the `Trimesh` class |
| [libigl](https://libigl.github.io/) ([Python bindings](https://github.com/libigl/libigl-python-bindings)) | Discrete differential geometry: cotangent Laplacians, mass matrices, curvature, parametrization, exact/heat geodesics | `laplacian`, `energies`, `curvature`, `parametrization`, `heat` |
| [Open3D](https://www.open3d.org/) | Point clouds, registration, and surface reconstruction: ICP, screened Poisson, ball pivoting | `points`, `registration`, `reconstruction` |
| [MeshLab](https://www.meshlab.net/) ([PyMeshLab](https://github.com/cnr-isti-vislab/PyMeshLab)) | Mesh editing filters: isotropic remeshing, decimation, smoothing, hole filling, uniform resampling | `remesh`, `smoothing`, `holes`, `repair` |
| [PyVista](https://pyvista.org/) (VTK) | The VTK toolkit: feature edges, cell-quality metrics, contouring and clipping, point location, arc-length and polyline measures, voxelization | `edges`, `triangles`, `intersection`, `levelset`, `proximity`, `polyline`, `voxels` |
| [MeshLib](https://meshlib.io/) | Minimum-weight hole filling and stitching, self-intersection repair, voxel offsets and booleans, projection queries, spike and outlier detection | `holes`, `repair`, `levelset`, `voxels`, `proximity`, `points` |
| [PyMeshFix](https://github.com/pyvista/pymeshfix) (MeshFix / TMesh) | The repair pipeline end to end: a broken digitised surface in, one watertight solid out | `repair`, `holes`, `validation` |
| [potpourri3d](https://github.com/nmwsharp/potpourri3d) (geometry-central) | The heat-method family: vector heat, parallel transport, log maps, signed distance, tangent frames | `heat`, `tangent_space` |
| [PyTorch3D](https://pytorch3d.org/) | Batched neighbour and Chamfer primitives, the mesh regularization losses, cubify and marching cubes | `metrics`, `neighbors`, `energies`, `registration`, `levelset`, `voxels` |

[SciPy](https://scipy.org/)'s `spatial.KDTree` and `sparse.csgraph` (covered by `neighbors`,
`graph` and `proximity`) and NumPy's array primitives (`array`, `reduce`, `grouping`, `linalg`)
round the set out to the eleven reference implementations the suite measures against.

These libraries are not runtime dependencies — they are **test oracles**. Every ordito
function ships with a regression test comparing its output against the corresponding reference
implementation, and a parity gate in the test suite fails the build if a benchmarked
implementation pair is neither value-tested nor explicitly exempted with a written reason. The
benchmark suite spans 351 groups and 587 `(group, library)` pairs: 557 are claimed by a value
test, 30 carry a written and categorised exemption, and none are left uncovered. When ordito and
a reference disagree by definition rather than tolerance, the test says so and documents the
measured difference.

## Install

```bash
uv add ordito
```

or with `pip`:

```bash
pip install ordito
```

Requires Python ≥ 3.11 and `warp-lang` ≥ 1.17. A CUDA-capable GPU is recommended but not
required — every function also runs on Warp's CPU backend. Mesh file I/O via
[meshio](https://github.com/nschloe/meshio) is an optional extra: `pip install ordito[io]`.

Tested on **Linux**, with and without CUDA. The wheel is pure Python and `warp-lang` supports
macOS and Windows, so ordito is expected to work there, but neither is verified — see
[Platform support](https://lucagrementieri.github.io/ordito/getting-started/#platform-support).

## Quickstart

```python
import ordito as od

# Parametric primitives allocate directly on the default device (CUDA if available).
vertices, faces = od.creation.icosphere(subdivisions=4)

# Optional object API: derived quantities are computed on first access and cached.
mesh = od.Trimesh(vertices, faces)
print(mesh.area)  # 12.551...
print(mesh.is_watertight)  # True
print(mesh.euler_characteristic)  # 2
```

Everything the `Trimesh` class offers is also available as free functions over raw Warp
arrays, so pipelines stay on the device end to end:

```python
import numpy as np
import warp as wp

import ordito as od

vertices, faces = od.creation.icosphere(subdivisions=4)

# Geodesic distance from vertex 0 via the heat method (two sparse CG solves, on-device).
sources = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=vertices.device)
distance = od.heat.heat_geodesic(vertices, faces, sources)

# Uniform, area-weighted surface sampling.
points, face_ids = od.sample.sample_surface(vertices, faces, 10_000, seed=0)

# Feature-preserving isotropic remeshing (split / collapse / flip / smooth / reproject).
remeshed_vertices, remeshed_faces = od.remesh.isotropic_remesh(vertices, faces, target_length=0.05)
```

Interop with NumPy is a `wp.array(...)` / `.numpy()` pair away, and
`od.mesh.Trimesh.from_warp_mesh` / `mesh.warp_mesh` bridge to Warp's own `wp.Mesh` BVH for ray
and proximity queries. For a pipeline that starts from a realistically messy input — a hole to
close, stray debris to drop — see the
[Cookbook](https://lucagrementieri.github.io/ordito/cookbook/).

## What's inside

| Area | Modules | Highlights |
|---|---|---|
| **Primitives** | `creation` | Boxes, spheres, platonic solids, capsules, tori; extrusion, revolution, sweeps, 2D polygon triangulation |
| **Structure & topology** | `mesh`, `vertices`, `edges`, `triangles`, `halfedge`, `adjacency`, `boundary`, `selection`, `validation`, `homology`, `tangent_space` | Edge/face adjacency, half-edge structure, boundary loops, manifold/watertight/orientability predicates, submesh selection, homology generators, tangent frames |
| **Measures & shape descriptors** | `measures`, `curvature`, `bounds`, `visibility` | Volume, centroid, and inertia integrals; discrete mean/Gaussian curvature and principal directions; axis-aligned and oriented bounding boxes; ambient occlusion, shape diameter, thickness |
| **Editing & repair** | `transform`, `repair`, `holes`, `combine`, `remesh`, `smoothing`, `seams`, `levelset` | Rigid transforms, hole filling, winding repair, isotropic remeshing, quadric and clustering decimation, Loop and midpoint subdivision, Laplacian/Taubin smoothing, seam cutting, level-set offsets and marching cubes |
| **Spatial queries** | `proximity`, `ray`, `neighbors`, `intersection`, `metrics` | Closest point, signed distance, winding number, ray casting, BVH and hash-grid neighbor queries, triangle-triangle intersection, chamfer and Hausdorff distance (differentiable via `wp.Tape`) |
| **Point clouds, voxels & reconstruction** | `points`, `sample`, `voxels`, `reconstruction`, `registration` | Poisson-disk and blue-noise sampling, farthest-point and voxel down-sampling, screened Poisson reconstruction, ball pivoting, marching cubes, Delaunay triangulation, Procrustes and ICP (point-to-point, point-to-plane) |
| **Operators & solvers** | `laplacian`, `energies`, `linalg`, `interpolation`, `parametrization` | Cotangent Laplacian, mass matrices, discrete energies, sparse conjugate-gradient solvers (Jacobi and multigrid-preconditioned), harmonic, LSCM, and ARAP parametrization |
| **Geodesics & heat methods** | `geodesic_walk`, `heat` | Direct combinatorial surface walks, heat-method geodesic distance, vector heat / parallel transport / log maps, signed heat method |
| **Curves** | `polyline` | Polyline resampling, simplification, and measures |
| **Attributes & I/O** | `texture`, `io` | Per-vertex/face attribute handling, meshio-backed mesh loading |
| **Arrays & infrastructure** | `array`, `reduce`, `grouping`, `graph`, `typing`, `constants` | GPU sort/scan/unique/group primitives, reductions, graph algorithms (BFS, connected components), typed array aliases |

50 public modules in total — the full API reference, generated per module, lives at
<https://lucagrementieri.github.io/ordito/>.

## Design

- **Arrays in, arrays out.** The core API is free functions over `wp.array` buffers — no
  mandatory mesh object, no hidden state. Faces are a flat `int32` index buffer, vertices a
  `wp.vec3` array, the same convention everywhere.
- **The device follows the data.** Every kernel launch and allocation inherits the device of
  its inputs; there are no global device switches.
- **Host syncs are budgeted.** Wrappers avoid device-to-host readbacks, and where one is
  unavoidable the reason is documented; bounds a caller already knows can be passed in to skip
  the inference.
- **Measured, not assumed.** Every public module has both a test file and a benchmark file;
  performance work lands only with a before/after measurement, and correctness is asserted on
  values, never just shapes.

## Status

ordito is pre-1.0 (`0.x`): the test suite is extensive (over 3,400 tests per device, an
eleven-library parity gate with no uncovered pair) and the library is safe to build on, but a
public signature may still shift a positional argument to a keyword or gain a required parameter
between minor versions until 1.0. Released versions are recorded in
[CHANGELOG.md](CHANGELOG.md).

## Development

```bash
git clone https://github.com/lucagrementieri/ordito
cd ordito
uv sync --all-groups            # runtime + dev + test + docs + bench dependencies

uv run pytest                   # regression tests against the reference libraries
uv run ruff format ordito tests && uv run ruff check ordito tests
uv run basedpyright             # type checking (0 errors expected)

uv run zensical serve           # preview the docs locally

uv run pytest --device=cpu --cov=ordito        # coverage, the way CI measures it
```

The coverage badge measures the Python wrapper layer on the CPU backend, which is what CI can
run, and is refreshed by the `main` build. It deliberately excludes `ordito/kernels/`: a
`@wp.kernel` body is compiled from its AST and executed on the device, never called as Python, so
coverage.py reports every line of a constantly-running kernel as unexecuted. Device-side behaviour
is covered instead by running the suite on both backends (`uv run python -m tests.devices`), which
is a two-process job and needs a GPU.

The test environment installs the reference stack (trimesh, libigl, Open3D, potpourri3d,
PyMeshLab, PyVista, MeshLib, PyMeshFix, PyTorch3D, SciPy, and more) so the comparison suite runs
in full.
Benchmarks live in `benchmarks/` and use `pytest-benchmark`; PyTorch3D is the one reference with
CUDA kernels of its own, so it is also the suite's only GPU-against-GPU comparison.

Bug reports, reproductions and pull requests are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for the development loop, the bar a pull request has to clear,
and where the internal engineering reference lives.

## Links

- **Documentation:** <https://lucagrementieri.github.io/ordito/>
- **Source:** <https://github.com/lucagrementieri/ordito>
- **Issues:** <https://github.com/lucagrementieri/ordito/issues>
- **NVIDIA Warp:** <https://nvidia.github.io/warp/>

## Citation

If ordito is useful in your research, please cite it:

```bibtex
@software{grementieri_ordito,
  author  = {Grementieri, Luca},
  title   = {{ordito}: GPU-accelerated triangular mesh processing on NVIDIA Warp},
  year    = {2026},
  url     = {https://github.com/lucagrementieri/ordito},
  version = {0.1.0}
}
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in ordito by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
