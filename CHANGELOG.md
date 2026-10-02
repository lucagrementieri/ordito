# Changelog

All notable changes to ordito are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and ordito adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a pre-1.0 caveat: while the
version is `0.x`, a **minor** bump may change a public signature, add or remove a public module,
or raise the `warp-lang` floor, and a **patch** bump is limited to bug fixes and documentation.

ordito is pre-1.0. The public API is safe to build on — it is covered by an extensive regression
suite and an eleven-library parity gate — but until 1.0 a public signature may move a positional
argument to a keyword or gain a required parameter between minor versions. 1.0 will be the point
at which public function signatures become a commitment: module, name and positional argument
order stable, with change limited to additive keyword arguments.

Entries are written by hand rather than generated from commit messages: the git history is mostly
internal engineering (kernel-shape refactors, measurement rounds, review passes) that is not a
user-facing change, and a generated log would bury the handful of entries that are.

Every level-2 heading below is a released version, and `release.yml` cuts the GitHub Release body
from the one matching its tag — so prose sections do not belong between them.

## [Unreleased]

The first public release. Everything below is new, because there is no prior tag to diff against.

### Added

- **Primitives** (`creation`) — boxes, spheres, icospheres, platonic solids, capsules, tori,
  cones and cylinders; extrusion, revolution and sweeps; 2D polygon triangulation; parametric
  surfaces.
- **Structure and topology** (`mesh`, `vertices`, `edges`, `triangles`, `halfedge`, `adjacency`,
  `boundary`, `selection`, `validation`, `homology`, `tangent_space`) — edge and face adjacency,
  a half-edge structure, boundary loops, manifold / watertight / orientability predicates,
  submesh selection, homology generators and tangent frames.
- **Measures and shape descriptors** (`measures`, `curvature`, `bounds`, `visibility`) — volume,
  centroid and inertia integrals; discrete mean and Gaussian curvature with principal directions;
  axis-aligned and oriented bounding boxes; ambient occlusion, shape diameter and thickness.
- **Editing and repair** (`transform`, `repair`, `holes`, `combine`, `remesh`, `smoothing`,
  `seams`, `levelset`) — rigid transforms, minimum-weight hole filling, winding repair, isotropic
  remeshing, quadric and clustering decimation, Loop and midpoint subdivision, Laplacian and
  Taubin smoothing, seam cutting, level-set offsets and marching cubes.
- **Spatial queries** (`proximity`, `ray`, `neighbors`, `intersection`, `metrics`) — closest
  point, signed distance, winding number, ray casting, BVH and hash-grid neighbour queries,
  triangle-triangle intersection, and Chamfer and Hausdorff distance differentiable through
  `wp.Tape`.
- **Point clouds, voxels and reconstruction** (`points`, `sample`, `voxels`, `reconstruction`,
  `registration`) — Poisson-disk and blue-noise sampling, farthest-point and voxel down-sampling,
  screened Poisson reconstruction, ball pivoting, Delaunay triangulation, and Procrustes and ICP
  (point-to-point and point-to-plane) registration.
- **Operators and solvers** (`laplacian`, `energies`, `linalg`, `interpolation`,
  `parametrization`) — the cotangent Laplacian, mass matrices, discrete energies, sparse
  conjugate-gradient solvers with Jacobi and algebraic-multigrid preconditioners, and harmonic,
  LSCM and ARAP parametrization.
- **Geodesics and heat methods** (`geodesic_walk`, `heat`) — combinatorial surface walks,
  heat-method geodesic distance, vector heat, parallel transport, log maps and the signed heat
  method.
- **Curves** (`polyline`) — resampling, simplification and measures.
- **Attributes and I/O** (`texture`, `io`) — per-vertex and per-face attribute handling, and
  meshio-backed loading behind the optional `io` extra.
- **Arrays and infrastructure** (`array`, `reduce`, `grouping`, `graph`, `typing`, `constants`) —
  GPU sort, scan, unique and grouping primitives, reductions, graph algorithms and typed array
  aliases.
- An optional `Trimesh` object API with lazily cached derived quantities, over the same free
  functions.
- Inline type information (`py.typed`) for the whole public surface.
- `ordito.__version__`, read from the installed distribution metadata and resolved lazily, so it
  costs nothing for a caller who never asks for it.

### Notes

- Requires Python 3.11 or newer and `warp-lang` 1.17 or newer. The 1.17 floor is load-bearing:
  `neighbors` and `proximity` call `wp.bvh_query_sphere`, which does not exist before it. CI
  installs and smoke-tests the wheel on 3.11, 3.12, 3.13 and 3.14.
- The only runtime dependencies are `warp-lang` and `numpy`; `numpy` adds nothing to an install,
  since `warp-lang` already requires it unconditionally.
- Every function runs on CUDA when a GPU is present and on Warp's CPU backend otherwise. The
  documented performance figures are measured on Linux with CUDA; see
  [Platform support](https://lucagrementieri.github.io/ordito/getting-started/#platform-support).

[Unreleased]: https://github.com/lucagrementieri/ordito/commits/main
