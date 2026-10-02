# Contributing to ordito

Thanks for your interest. Bug reports, reproductions, documentation fixes and new geometry
functions are all welcome.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md), and by contributing you
agree that your work is dual licensed under MIT and Apache-2.0, matching the project (see
[README](README.md#license)).

## Before you start

- **Bugs**: open an [issue](https://github.com/lucagrementieri/ordito/issues) with the mesh (or a
  script that builds one), the device it happened on (`cpu` or `cuda`), and your `warp-lang`
  version. A wrong *answer* is more useful to report than a slow one — see below for why speed
  reports need a specific shape.
- **New functions**: open an issue first if it is more than a small addition. ordito deliberately
  does not add an axis, parameter or mode without a call site that needs it, so a proposal lands
  faster when it names the use case.
- **Out of scope**: rendering and visualization. ordito computes geometry; it has no viewer,
  rasterizer or camera model, and adding one is not planned. Use PyVista or Open3D for display.

## Development setup

```bash
git clone https://github.com/lucagrementieri/ordito
cd ordito
uv sync --all-groups        # runtime + dev + test + docs + bench
```

`uv sync --all-groups` installs the eleven reference implementations the comparison suite needs.
It is a large environment — it builds PyTorch3D from source against a CUDA torch — and the build
is cached, so the first sync is slow and later ones are not. A bare `uv sync` will *uninstall*
the test dependencies, because there is no `default-groups` setting; always pass the groups.

## The gate

Run all of these before opening a pull request. They are the same checks CI runs, so a local pass
is a CI pass:

```bash
uv run ruff format ordito tests benchmarks
uv run ruff check ordito tests benchmarks
uv run basedpyright                            # 0 errors expected
uv run pytest                                  # one device
```

CI additionally measures coverage on its CPU run and fails below a floor, so if you are adding or
moving code in the wrapper layer, check it the way CI will:

```bash
uv run pytest --device=cpu --cov=ordito --cov-report=term
```

The measurement covers `ordito/*.py` and deliberately not `ordito/kernels/`. A `@wp.kernel`
body is compiled from its AST rather than called, so coverage.py records every line of it as
unexecuted no matter how often the kernel runs — including it would report a number that means
nothing. New wrapper code therefore needs its Python-visible branches exercised: the argument
validation, the `Literal` menu arms, the empty-input early returns.

Two more that need a CUDA machine and are the maintainer's release gate rather than a per-PR
requirement — run them if you have the hardware:

```bash
uv run python -m tests.devices                 # both devices, as two processes
uv run python -m tests.parity                  # the cross-suite parity matrix
```

If you touch docstrings or cross-references:

```bash
uv run zensical build --strict
```

## The bar for a pull request

- **Every public module has both a test file and a benchmark file.** A new public module needs
  `tests/test_<module>.py` and `benchmarks/test_<module>.py`; a mechanical check fails the test
  run otherwise.
- **Correctness is asserted on values, not shapes.** A new geometry function needs a regression
  test comparing its output against a CPU reference implementation — trimesh by default. Where a
  reference computes the same quantity, the test compares the two outputs; where none does, an
  invariant-only test is the honest answer and its docstring says so. Asserting only that the
  output has the right shape or is finite is the *benchmark's* job, and a parity gate in the test
  suite fails the build if a benchmarked pair is neither value-tested nor explicitly exempted
  with a written reason.
- **Performance changes need a measurement.** Never restructure code for speed without a
  benchmark group timing the current implementation first, and report old and new measured
  back to back in one session — saved baselines drift by more than most wins. A change that was
  measured and rejected is a result worth recording, not a dead end to delete.
- **Docstrings are NumPy style** and every public function needs one; `ruff`'s `D` rules enforce
  this. Keep hardware-specific numbers out of public docstrings — the documentation site
  publishes them as though they were part of the contract.

## The engineering reference

[`AGENTS.md`](AGENTS.md) (the same file as `.claude/CLAUDE.md`) is the authoritative internal
reference for anyone doing nontrivial work here: the Warp kernel conventions, the launch and
allocation cost model, the measured platform quirks, and the reasoning behind decisions that
look arbitrary from outside. It is long, and it is worth reading the relevant section before
writing a kernel or proposing an optimization — most plausible ideas in this repository have
already been built and measured, and several of them lost.

It is written for contributors, not users: it is pinned to specific hardware and a specific Warp
version, and it records reversals as well as conclusions. Nothing in it belongs in user-facing
documentation verbatim.

## Reporting a performance problem

A speed report is actionable when it names the function, the mesh and its size, the device, and
what you compared against. "Slow on a large mesh" cannot be acted on; "`remesh.isotropic_remesh`
on a 2 M-face scan takes 40 s on an RTX 4090 where PyMeshLab takes 12 s" can. If a function is
unexpectedly slow the *first* thing to check is whether Warp is recompiling a kernel module —
a stall at 0 % GPU utilization is almost always a compile, not a slow kernel.

## Releasing

Maintainer-only, and documented separately in [`.github/RELEASING.md`](.github/RELEASING.md):
pushing a `v*` tag is what builds, publishes to PyPI and cuts a GitHub Release. Contributors
never need to touch it — do not bump `pyproject.toml`'s version in a pull request, since the
release checklist owns that step and a bump in a PR only creates a conflict.
