# Changelog

All notable changes to ordito are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and ordito adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a pre-1.0 caveat: while the
version is `0.x`, a **minor** bump may change a public signature, add or remove a public module,
or raise the `warp-lang` floor, and a **patch** bump is limited to bug fixes and documentation.

ordito is pre-1.0. The public API is safe to build on — it is covered by an extensive regression
suite and a nine-library parity gate — but until 1.0 a public signature may move a positional
argument to a keyword or gain a required parameter between minor versions. 1.0 will be the point
at which public function signatures become a commitment: module, name and positional argument
order stable, with change limited to additive keyword arguments.

Entries are written by hand rather than generated from commit messages: the git history is mostly
internal engineering (kernel-shape refactors, measurement rounds, review passes) that is not a
user-facing change, and a generated log would bury the handful of entries that are.

Every level-2 heading below is a released version, and `release.yml` cuts the GitHub Release body
from the one matching its tag — so prose sections do not belong between them.

## [Unreleased]

## [0.1.0] - 2026-10-27

The first public release.

### Added

- GPU geometry processing on NVIDIA Warp: 52 public modules covering mesh creation, topology and
  validation, measures and curvature, repair, remeshing and smoothing, spatial queries and
  distances, point clouds and reconstruction, registration, Laplacians and sparse solvers
  (CG, multigrid, Cholesky), parametrization, geodesics and heat methods, and polylines.
- An optional `Trimesh` object API with cached derived quantities, over the same functions.
- Opt-in bit-reproducible results on CUDA (`ordito.config.set_deterministic`).
- Inline type information (`py.typed`).

### Notes

- Requires Python 3.11+ and `warp-lang` 1.18+; the only other runtime dependency is `numpy`.
  Mesh I/O needs the optional `io` extra (meshio).
- Runs on CUDA when available and on Warp's CPU backend otherwise; see
  [Platform support](https://lucagrementieri.github.io/ordito/getting-started/#platform-support).

[Unreleased]: https://github.com/lucagrementieri/ordito/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lucagrementieri/ordito/releases/tag/v0.1.0
