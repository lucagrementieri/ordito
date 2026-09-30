# triwarp

**triwarp** is a GPU-first triangular mesh processing library built on
[NVIDIA Warp](https://github.com/NVIDIA/warp). It provides mesh geometry, connectivity, queries,
editing, and solvers as plain array-in / array-out functions backed by Warp kernels, with a
[trimesh](https://trimesh.org)-inspired API. Every function runs on CUDA when a GPU is available
and falls back to CPU otherwise — same code, same results.

## Highlights

- **One dependency.** The runtime depends on `warp-lang` alone; NumPy is only needed to move data
  in and out.
- **Broad coverage.** 50 public modules and over 480 functions spanning primitives, topology,
  repair, remeshing, spatial queries, discrete differential operators, geodesics, sampling,
  surface reconstruction, and registration.
- **Stays on the device.** Functions take Warp arrays and return Warp arrays, so pipelines compose
  without host round-trips; device placement follows the inputs.
- **Optional `Trimesh` object.** A frozen mesh class with lazily cached derived quantities
  (normals, adjacency, boundary loops, manifoldness and watertightness predicates, BVH) for when
  an object API is more convenient than free functions.
- **Fully typed** (`py.typed`) and validated function-by-function against eleven reference
  implementations — nine established geometry-processing libraries plus SciPy and NumPy, all
  listed below.

## Install

```bash
uv add triwarp
```

or with `pip`:

```bash
pip install triwarp
```

Requires Python ≥ 3.11 and `warp-lang` ≥ 1.17. A CUDA-capable GPU is recommended but not required.
Mesh file I/O via [meshio](https://github.com/nschloe/meshio) is an optional extra:
`pip install triwarp[io]`.

## Quickstart

```python
import triwarp as tw

# Parametric primitives allocate directly on the default device (CUDA if available).
vertices, faces = tw.creation.icosphere(subdivisions=4)

# Optional object API: derived quantities are computed on first access and cached.
mesh = tw.Trimesh(vertices, faces)
print(mesh.area)                  # 12.551353454589844
print(mesh.is_watertight)         # True
print(mesh.euler_characteristic)  # 2

# Feature-preserving isotropic remeshing (split / collapse / flip / smooth / reproject).
remeshed_vertices, remeshed_faces = tw.remesh.isotropic_remesh(
    vertices, faces, target_length=0.05
)
```

Continue with **[Getting started](getting-started.md)** for the full walkthrough (including a
repair-then-remesh pipeline on a realistically broken mesh), or jump straight to a task:

- **[Concepts](concepts.md)** — the design principles worth knowing before writing anything
  nontrivial.
- **[Cookbook](cookbook/index.md)** — task-oriented recipes: cleaning a scan, point clouds to
  watertight surfaces, geodesic distance fields, aligning two scans.
- **[Migrating from another library](migrating-from/index.md)** — already know trimesh, libigl,
  Open3D, MeshLab, potpourri3d, or PyTorch3D? Start here.
- **[Performance](performance.md)** — why the GPU path is fast, and how to check any number
  yourself.
- **[Benchmarks](benchmarks.md)** — a curated set of triwarp-vs-reference comparisons, rendered as
  charts, each library named by its own logo.

## One GPU library instead of nine

Mesh processing in Python has long meant stitching together several excellent — but mostly
CPU-bound and stylistically different — libraries. triwarp consolidates the functionality it needs
from each of them behind a single GPU-accelerated API:

| Replaces | For | In triwarp |
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

These libraries are not runtime dependencies — they are **test oracles**. Every triwarp
function ships with a regression test comparing its output against the corresponding reference
implementation, and a parity gate in the test suite fails the build if a benchmarked
implementation pair is neither value-tested nor explicitly exempted with a written reason. The
benchmark suite spans 351 groups and 587 `(group, library)` pairs: 557 are claimed by a value
test, 30 carry a written and categorised exemption, and none are left uncovered. When triwarp and
a reference disagree by definition rather than tolerance, the test says so and documents the
measured difference.

## Development

```bash
git clone https://github.com/lucagrementieri/triwarp
cd triwarp
uv sync --all-groups            # runtime + dev + test + docs + bench dependencies

uv run pytest                   # regression tests against the reference libraries
uv run ruff format triwarp tests && uv run ruff check triwarp tests
uv run basedpyright             # type checking (0 errors expected)

uv run zensical serve           # preview the docs locally
```

See the [source repository](https://github.com/lucagrementieri/triwarp) for contribution
guidelines and the full development setup.
