# ordito benchmarks

Performance benchmarks comparing `ordito` against the ten CPU references **trimesh**, **libigl
(`igl`)**, **open3d**, **scipy**, **numpy**, **potpourri3d** (geometry-central), **pymeshlab**
(MeshLab / VCGlib), **pyvista** (VTK), **meshlib** and **pymeshfix** (MeshFix / TMesh) — plus
**pytorch3d**, the one reference with CUDA kernels of its own, which therefore takes two rows and
gives the suite its only GPU-against-GPU comparison. Built on
[pytest-benchmark](https://pytest-benchmark.readthedocs.io).

The authoritative list is `LIBRARIES` in [`conftest.py`](conftest.py); count it there rather than
here if the two ever disagree.

These are **not** collected by the normal test run (`pytest`'s `testpaths` is `tests/`); run them
by pointing pytest at this directory.

## Install

```bash
uv sync --group bench
```

## The rule: one group measures one axis, at 2–3 points

Face count is the *only* cost driver for a minority of functions in this package. Far more often
the driver is something a triangle count cannot express: how many connected components there are,
how many boundary loops and how long, the diameter of the adjacency graph, triangle aspect ratio,
vertex valence, ray depth, collision density, or a parameter that is not a mesh property at all.

So every benchmark group names one axis and takes 2–3 points along it. What is held *fixed* is as
much the point as what varies — that is what makes a timing spread attributable to a cause rather
than to "a different mesh". A group whose spread is ~1× is telling you its axis does not drive that
function, which is a useful answer too.

Two examples of what this buys, both measured on an RTX 5090:

| group | control | perturbed | spread |
|---|---|---|---|
| `combine.concatenate`, F = 81 920 fixed | 8 pieces **0.23 ms** | 512 pieces **5.7 ms** | **25×** |
| `homology_generators`, F ≈ 90 000 fixed | genus 0 **2.66 ms** | genus 64 **3.13 ms** | **1.2×** |

Neither is visible to a face-count sweep. `concatenate` came down from 48× (0.39 → 19.0 ms) by
collapsing its per-piece index renumbering into one launch; what is left is one `wp.copy` per input
buffer, which Warp cannot batch — there is no gather across separate allocations — so the residual
slope is real and bounded below by the piece count. The genus row is the other outcome an axis can
report: it read **2.1×** while 128 generator loops were traced one at a time on the host, and
flattened to 1.2× once the tracing moved to the device — the axis is what showed which half of the
algorithm the genus actually drove.

Four more entries have already been retired from this table by the fixes they prompted, and all
four are kept in their modules as worked examples of what the axis rule is for:

- **`combine.split`** headed it at **262×** (2.6 ms → 669 ms, a 3× loss to trimesh at a thousand
  components). Commit `f57d3f0` batched its per-component compaction; it now runs 2.5 / 3.1 /
  9.4 ms against trimesh's 36.6 / 37.1 / 183 ms and open3d's 68.1 / 76.4 / 290 ms — a 3.7× spread
  and a 19-31× win at every point.
- **`validation.face_orientation_bits`** was **83×** (7.9 ms → 653 ms, an 83× loss to trimesh's
  `fix_winding`) while it propagated its Z2 bits one launch per graph level. Solving them with a
  parity-carrying union-find instead makes it depth-independent: **0.92 ms and 0.77 ms**, a win
  over trimesh at both ends. `repair.make_winding_consistent` inherited the fix.
- **`holes.fill_min_weight`** was *inverted*: 512 three-vertex holes cost 376 ms
  against 273 ms for two 512-vertex rims, i.e. the trivial case cost more than the `B³` one,
  because each hole paid its own readbacks, chord pass and span launches. Batching the interval DP
  across loops took it to **4.2 ms**, and dropped the two-rim point to 157 ms by running both rims
  in the same launches.
- **`polyline.triangulate_polygon`** was **23×** across its resolution points (6.1 ms → 141 ms on a
  1024-point star) and an 80× loss to trimesh. The ear clipper is parallel; what was linear in the
  ring size was its *round count*, because competing ears were ranked by raw ring index and a star
  ring makes that rank suppress all but one ear per round. Ranking by a hash of the index gives
  **2.8 / 4.6 ms** and 30 rounds instead of 1 022.

## Mesh registries

Two registries, in [`meshes.py`](meshes.py), for two different questions.

### Scan meshes — the pure-`N` axis

Real scan meshes, read once with `meshio` (the same loader `ordito.io` uses). Place the files in
`benchmarks/data/` (gitignored, local-only). Used **only** by the throughput groups: `triangles`,
`edges`, `reduce`, `grouping`, the `laplacian` entry kernels, `array.index_bound` /
`mean_vertex_normals`, and the parts of `remesh` / `proximity` / `texture` / `neighbors` that
genuinely scale with size.

| category | faces | meshes |
|---|---|---|
| `small` | `<10k` | — |
| `medium` | `<100k` | bunny_decimated, bunny |
| `large` | `<1M` | dragon |
| `extralarge` | `<10M` | happy_buddha |
| `huge` | `≥10M` | lucy |

### Feature meshes — one control, one property each

Generated with `trimesh.creation` (never with `ordito.creation`: a benchmark input must not depend
on the code under test). `sphere_med` = `icosphere(6)` is the control — **V = 40 962, F = 81 920**,
one watertight component, uniform valence 6, graph diameter ≈ 130, no boundary, aspect ratio ≈ 1.4.
Every other mesh perturbs exactly one of those while pinning `V` and/or `F` to it.

| name | V | F | pinned | perturbed property |
|---|---|---|---|---|
| `sphere_small` | 2 562 | 5 120 | — | clean small `N` |
| **`sphere_med`** | 40 962 | 81 920 | — | **control** |
| `sphere_large` | 163 842 | 327 680 | — | clean large `N` |
| `parts_64` | 41 088 | 81 920 | **F** | components 1 → 64 |
| `parts_1024` | 43 008 | 81 920 | **F** | components 1 → 1 024 |
| `open_parts_4` | 40 968 | 81 916 | **F** | 4 *open* shells, 1 rim each |
| `open_parts_16` | 40 992 | 81 904 | **F** | 16 *open* shells, 1 rim each |
| `open_parts_64` | 41 088 | 81 856 | **F** | 64 *open* shells, 1 rim each |
| `ribbon_long` | 40 962 | 40 960 | **V** | graph diameter 130 → 20 481 |
| `fan_hub` | 40 962 | 81 920 | **V, F** | max valence 6 → 40 960 |
| `rim_long` | 131 072 | 131 072 | — | 2 boundary loops × 65 536 |
| `holes_many` | 40 962 | 81 408 | **V** | 512 boundary loops × 3 |
| `rim_short` | 1 024 | 1 024 | — | 2 loops × 512 (the `B³` DP scale) |
| `saddle_small` | 4 624 | 8 978 | — | small disk patch |
| `saddle` | 17 689 | 34 848 | — | well-conditioned disk patch |
| `saddle_graded` | 17 689 | 34 848 | **V, F, connectivity** | worst aspect ratio 1.6 → 4 719 |
| `hemisphere` | 20 737 | 41 088 | — | curved disk, 1 rim of 384 |
| `shells_8` | 20 496 | 40 960 | — | ray crossings 2 → 16 |
| `tangle_2` | 20 484 | 40 960 | — | self-intersection density |

Ten parametric builders produce them, and [`test_meshes.py`](test_meshes.py) asserts every
recorded count, the topology each mesh is chosen for, and the cross-mesh pinning invariants. **Run
it after touching `meshes.py`** — a silent change in `trimesh.creation` would otherwise leave the
benchmarks green while comparing meshes that differ in more than one way.

### Axes

`AXES` in [`meshes.py`](meshes.py) names each comparison; the control comes first, so a results
table reads left-to-right as "baseline, then the perturbation".

| axis | meshes | holds fixed |
|---|---|---|
| `scale` | sphere_small → med → large | — (the igl-safe `N` sweep) |
| `components` | sphere_med, parts_64, parts_1024 | `F` |
| `open_components` | open_parts_4, _16, _64 | `F` (the joins axis: closed shells are a no-op) |
| `diameter` | sphere_med, ribbon_long | `V`, triangle quality |
| `valence` | sphere_med, fan_hub | `V` and `F` |
| `loops` | sphere_med, rim_long, holes_many | — (loop length vs loop count) |
| `loops_dp` | rim_short, holes_many | — (same contrast, at `B³`-survivable size) |
| `quality` | saddle, saddle_graded | `V`, `F`, connectivity |
| `patch` | saddle_small, saddle, hemisphere | disk topology |
| `polyline` | saddle_small, saddle, rim_long | — (longest loop 268 → 528 → 65 536) |
| `depth` | sphere_med, shells_8 | — |
| `overlap` | sphere_med, tangle_2 | — |
| `resolution` | *no meshes* | — (`test_creation`; see below) |

Select one with `@pytest.mark.benchaxis("components")`. `@pytest.mark.benchmeshes(...)` names meshes
directly for a one-off; both bypass `--size`, and both are a `UsageError` on a mesh-free benchmark.
A benchmark with neither marker gets the scan sweep.

Where a group's driver is a *parameter* rather than a mesh, it is a second
`pytest.mark.parametrize` layer with named ids and lands as extra rows in the same table:
`screened_poisson(depth)`, `rasterize_*(resolution)`, `blue_noise(radius)`,
`expand_vertex_mask(hops)`, `winding_number(|queries|)`, `polyline_simplify(tol)`,
`icp(initial misalignment)`, `solve_spd_columns(check_every)`, `neighbors(k / leaf_size /
grid_bins)`, and the defect counts in `test_repair`.

### Mesh-free benchmarks

`ordito.creation` has no input mesh, so [`test_creation.py`](test_creation.py) takes the
`bench_lib` fixture instead of `bench_case` and is parametrized over libraries alone; its axis is
`resolution`, expressed as a plain `pytest.mark.parametrize` on `sections` / `subdivisions` /
`face_count`. Supporting that needed two harness changes: `BenchCase` derives from a mesh-free
`BenchLibrary` base, and this package overrides `pytest_benchmark_group_stats` — the plugin's own
version indexes `bench["params"]["mesh_name"]` directly and raises `KeyError` as soon as a mesh-free
case is collected under the default `group,param:mesh_name` grouping.

## Run

```bash
# Default: ordito-cuda on all meshes; CPU references on scan meshes up to 'large'.
# ordito-cpu is off by default when CUDA is available (pass --device=both to add it).
uv run pytest benchmarks/

# One module, GPU only:
uv run pytest benchmarks/test_combine.py --device=cuda

# Quick CPU-only smoke on the medium scan meshes -- but see the warning below: on a box with a
# GPU this inflates the ordito-cpu rows, and the harness says so.
CUDA_VISIBLE_DEVICES="" uv run pytest benchmarks/test_edges.py --device=cpu --size=medium

# Both ordito targets, correctly: two processes, the CPU one with CUDA hidden.
uv run python benchmarks/devices.py
```

### Never time `ordito-cpu` in a CUDA-initialised process

**Warp's CPU work costs more once CUDA has been initialised in the process**, and the charge behaves
like a per-launch one, so the factor scales with launch count rather than with work:

| measured (`ordito-cpu`) | CUDA visible | `CUDA_VISIBLE_DEVICES=""` | ratio |
|---|---|---|---|
| `marching_cubes` 64 / 128 | 41.67 / 351.78 ms | 34.64 / 338.34 ms | **1.20x / 1.04x** |
| `edges_unique` [bunny_decimated] | 11.22 ms | 8.76 ms | 1.28x |
| its three siblings | 11.9-12.4 ms | 8.5-8.8 ms | 1.40-1.41x |
| `heat_signed_distance`, one call | 50.57 s | **1.40 s** | ~36x |
| `tests/test_heat_signed.py --device=cpu` | 166.77 s | **8.53 s** | 19.6x |

The spread is the point: **a launch-light row can be inside its own noise** (`marching_cubes`'s 1.04x
sits under the 1.64x that `meshlib-128` itself moved between those two runs), while an iterative
solver pays the whole factor. So the groups to distrust are the solver-heavy ones — heat, laplacian,
linalg, smoothing, parametrization — not every `ordito-cpu` row ever recorded.

Unchanged by `warp.config.launch_array_access_mode` (`RELAXED` 50.34 s, `CHECKED` 49.77 s on the
solver), so it is CUDA *presence* and not the launch guard. So a `ordito-cpu` row taken this way is
not a slow number, it is a **wrong** one — and it reads as ordito losing to CPU references it
actually beats. `--device=cpu` and `--device=both` on a GPU box both emit a `UserWarning` saying so.
The default `--device=auto` is unaffected: it selects `ordito-cpu` only when there is no CUDA
device.

`CUDA_VISIBLE_DEVICES` must be set before the process starts, so this can only be fixed by a second
process — `benchmarks/devices.py` runs the CUDA pass (with every reference) and then a CUDA-hidden
`-k ordito-cpu` pass. The narrowing matters: the CPU references are included in *every* pass, so an
unnarrowed second pass would time them twice and duplicate the rows. `tests/devices.py` is the same
idea for the test suite.

A full default run is **~18 minutes** on an RTX 5090 (measured: 1 097 s across all 33 modules, one
process each, 999 cases plus 139 skipped). The four slowest modules are `test_reconstruction`
(235 s), `test_laplacian` (95 s), `test_proximity` (85 s, the `O(Q x F)` winding number) and
`test_smoothing` (70 s).

The **pymeshlab** rows add roughly **six minutes** on top of that, spread over 26 modules; the
largest single contributions are `test_heat_distance` (+40 s, its heat solver at `setup=full` rebuilds the
factorization every round), `test_proximity` (+3 s even capped at `bunny`), `test_combine` (+18 s, the
1 630 ms-per-call component split on `parts_1024`) and `test_curvature` (+9 s). Three groups are
explicitly capped for that reason — see the pymeshlab hazards below.

That measurement predates the six modules added with the potpourri3d port
(`test_halfedge`, `test_tangent_space`, `test_intersection`, `test_geodesic_walk`, `test_heat_vector`,
`test_laplacian`, `test_heat_signed`) and the potpourri3d rows in the existing ones. The three Phase-1 modules add ~10 s;
`test_heat_vector` and `test_geodesic_walk` add ~40 s between them (the reference factors two sparse systems
per case, and traces one ray per call); `test_heat_signed` adds ~15 s, most of it the reference's
1 s-per-call solver; and the potpourri3d rows add most of their cost to
`test_heat_distance`, now 74 s for the module with its `heat_geodesic` group carrying three libraries at two
setup points each.

Only the first of those was not measuring ordito, and it has since been **removed**: **190 of
`test_reconstruction`'s 199 timed seconds were open3d's CPU `create_from_point_cloud_poisson`**
(7.5 s a call at depth 9 on `bunny`), against 9.0 s for every ordito case in the module combined,
and across the whole suite the two CPU screened-Poisson references cost **6 322 s — 73 % of the
run** — to re-establish a ratio ordito had already won 15–25×. `screened_poisson` is now a
**ordito-only** group; the correctness comparison against open3d and pymeshlab lives in
`tests/test_reconstruction.py`, where it runs at a size a test can afford. The module now costs
~107 s. That episode is also the general lesson: before trimming anything in a slow module, read the
per-library split rather than the module total.

By default the harness prints **one comparison table per (function, mesh)** — each table lists the
libraries and any parameter points side by side — via `--benchmark-group-by=group,param:mesh_name`,
set automatically. Pass your own `--benchmark-group-by=...` to override.

## Flags

| flag | default | meaning |
|---|---|---|
| `--device` | `auto` | `ordito` target(s): `auto`/`cpu`/`cuda`/`both`. `auto` = cuda if available, else cpu. The CPU references (trimesh / igl / open3d / scipy / numpy / potpourri3d / pymeshlab / pyvista / meshlib / pymeshfix) always run, and so does `pytorch3d-cuda` whenever the installed pytorch3d has a working CUDA extension (pytorch3d has no host row at all) — this flag selects among *ordito's* targets, not the references'. **`cpu`/`both` on a GPU box inflate the `ordito-cpu` rows and warn** — use `benchmarks/devices.py`. Registered in the repo-root `conftest.py`, shared with `tests/`. |
| `--size` | `all` | comma-separated size categories for the **scan** sweep (`small,medium,large,extralarge,huge`). Naming a size also lifts the CPU cap for it. Has no effect on axis-driven groups. |
| `--cpu-max-size` | `large` | CPU-bound libraries (every reference, plus `ordito-cpu`) skip scan meshes larger than this unless the size is named in `--size`. |

## Notes

- **The CUDA mempool release threshold is 0, and the sync inside the timed region interacts with
  it.** Warp leaves `cudaMemPoolAttrReleaseThreshold` at 0, so the driver reclaims every free pool
  block at each synchronization point — and this harness deliberately syncs *inside* every timed
  callable (next bullet). A wrapper that allocates per call therefore re-faults its working set every
  round. Measured interleaved in one process on `creation.uv_sphere`: **0.489 ms at threshold 0
  against 0.363 ms at 8 GB (1.35x)**, and the threshold-0 number *drifts upward with pool history*
  (0.489 / 0.508 / 0.598 after 0 / 512 MB / 4 GB of allocate-free churn) where the raised one stays
  flat (0.363 / 0.393 / 0.372).

  It is deliberately **not** set, for two reasons. It buys only sub-millisecond rows: on the rows
  that matter it measures **1.01x** (`quadric_decimate[saddle_graded]`, 203.7 → 202.6 ms) and
  **1.00x** (`heat_geodesic` amortized), because those allocate large buffers few times rather than
  small buffers many times. And a library that raises a *process-global* CUDA pool policy is making a
  memory-footprint decision on its caller's behalf. If you are chasing a sub-millisecond row, set
  `wp.set_mempool_release_threshold(wp.get_device("cuda:0"), 8 << 30)` in your own process and
  re-measure before concluding anything about the wrapper.
- **GPU timing** is captured correctly: `wp.synchronize_device` runs inside the timed region and
  one warm-up round covers Warp kernel JIT compilation.
- **`BenchCase.run(fn, rounds=...)`** lowers the repeat count from the default 10 for the groups
  whose single call runs into hundreds of milliseconds — the hole-filling DP, `split` on a thousand
  components, an isotropic remesh, a Poisson reconstruction, the float64 CG solves. Ten rounds of
  those would dominate the suite's wall clock and their spread is wide enough that the extra samples
  buy nothing.
- **There is a ~340 µs host-side floor on every ordito wrapper call** (allocation plus Warp's
  launch path; ~75 µs of it is the NumPy prologue). `test_creation::test_box` is the probe that
  measures it — `box` is a 12-triangle constant table, so it measures nothing else. Read absolute
  numbers against that floor: a group sitting at it across its whole axis is reporting launch
  overhead, not an algorithm. **Read the floor from a full-suite run**: whichever group runs first
  in a module absorbs that library's one-time initialization (52 ms for ordito, 102 ms for trimesh,
  measured on `test_creation.py` alone).

  **A full-suite ratio matrix (605 comparisons) found 24 rows that are entirely this floor**, and
  each now says so in its own docstring rather than reading as a defect: the `creation` revolution
  primitives and Platonic tables (`creation`'s module docstring covers them collectively),
  `icosphere` since its connectivity became closed-form, `array.index_bound`,
  `bounds.aabb`, `points.point_plane_distance` and `remesh.cluster_decimate` at
  `bunny_decimated`. **Every one of them inverts further along its own axis** — `cluster_decimate`
  wins 76x at `dragon`, `n_vertices` 259x at `lucy` — so a floor row is a statement about the input
  size, not about the code, and optimizing one would mean removing allocations from correct code for
  microseconds nobody experiences. Below roughly 10³ elements, prefer to fix the *input size* of the
  benchmark over the function it measures.

  **A second population sits at 2-10x that floor and says the same thing, which is why a loss table
  sorted by gap keeps promoting it.** Round 7 carried **96 loss rows between 0.3 and 5.0 ms**, 95.9
  ms between them, and attributing them one at a time -- wall / device / launch count at the
  benchmark's own operating point, `pytest -p attrib_plugin` -- found a launch count in the tens and
  a host share of 77-91 %, i.e. the floor above multiplied by a small integer:

  | row | wall | device | host | kernels |
  |---|---|---|---|---|
  | `boundary_edges[sphere_med]` | 0.404 ms | 0.038 | 91 % | 7 |
  | `loop_perimeters[bunny]` | 0.196 | 0.034 | 82 % | 10 |
  | `triangulate_polyline[sphere_small, 64]` | 1.176 | 0.271 | 77 % | 22 |
  | `fillable_loop_mask[bunny]` | 1.287 | 0.217 | 83 % | 35 |
  | `remove_degree3_vertices[bunny]` | 2.545 | 0.333 | 87 % | 75 |

  So the rule for this band is **attribute before optimizing, and expect a floor row**. The
  exception is worth knowing because it is what the attribution is *for*: the same sweep found one
  row whose 51 % host share hid an algorithm rather than a floor -- `loop_perimeters[dragon]` at
  **412** launches, 2.505 ms of its 2.633 spent re-packing a buffer that was already packed -- and
  nothing about its ratio, its size or its host share distinguished it from the five above. The
  launch count did. See `test_boundary.py::test_loop_perimeters` for the fix and
  `test_holes.py::test_fillable_loop_mask` for a row where the same stage was measured and was
  **not** the cost.
- **A CPU reference can be the whole cost of a module.** Two were, before being capped: trimesh's
  `discrete_mean_curvature_measure` (one `cKDTree` ball query per point) took **40 s a call** on
  `sphere_med` and was 95% of `test_curvature`, and `tm.points.fit_line` took **22 s a call** on
  `bunny` and was 91% of `test_points`. Both are now capped at the smallest input that still
  establishes the ratio. When a module's wall clock looks wrong, get `--durations=10` before
  trimming anything — the reference is a more likely culprit than the code under test.
- **Read the per-table unit.** pytest-benchmark picks µs or ms *per table*, so a number copied out
  of one group is not comparable to one from another without checking the `Name (time in ...)`
  header. Cross-group ratios quoted in this repo's docstrings come from direct measurement scripts
  for exactly that reason.
- **A `--device=cpu` crash in unrelated code used to mean "run one module per process".** It was
  never the Warp CPU backend, which is what that advice claimed: the two causes were ordito's own
  out-of-bounds scatter in `successor_cycles` and a `wp.launch` missing `device=`, which ran a CUDA
  kernel over host arrays and corrupted the heap when they were freed. Both are fixed and both now
  have a standing guard — `tests/conftest.py`'s `STRICT` launch mode and `api_conventions` check 15
  — so a one-module-per-process loop is a diagnostic, not a routine:
  ```bash
  for f in benchmarks/test_*.py; do uv run pytest "$f" --device=cpu -q; done
  ```
- **trimesh references** rebuild their `tm.Trimesh` *inside* the timed callable wherever the
  reference reads a cached property, so every round measures real work. open3d references that
  mutate idempotently do the same; `BenchCase.mesh_o3d` is only shared where the reference either
  does not touch its input or recomputes unconditionally.
- `edges_unique*` ordito calls pass `n_vertices=` to avoid a host sync skewing GPU numbers.
- Every benchmark carries an explicit `benchlibs` marker, so adding a library kind to `LIBRARIES`
  never silently generates cases for modules that have no branch for it.
- **pymeshlab rows build their `MeshSet` inside the timed callable** unless the filter is verified
  geometry-preserving, because almost every MeshLab filter mutates `current_mesh()` in place. See the
  pymeshlab hazards section for the build cost and the shared-MeshSet whitelist.
- **`test_texture` uses projected, non-injective UVs** (vertex `xy` normalized to the unit square)
  because the scan meshes carry no atlas and computing one would dominate the measurement. The
  rasterizer cost is the number of (triangle, covered pixel) pairs, which a projection reproduces
  faithfully; the module docstring explains why that is sound for timing but not for parity.

### What a prebuilt `wp.Mesh` saves (measured, RTX 5090)

`proximity.closest_point_on_mesh`, `proximity.signed_distance_on_mesh`,
`validation.face_self_intersecting_mask` and `validation.is_watertight` take an optional
`mesh: wp.Mesh | None = None`, and `Trimesh.is_watertight` passes its cached `warp_mesh` through it.
Measured back to back in one process, both orders, values asserted equal — the saving is the clone
plus the BVH build, and it is **flat in the query count**:

| n_faces | 1 query | 1 000 | 10 000 | 100 000 |
|---|---|---|---|---|
| 1 280 | 0.149 ms (35%) | 0.119 (22%) | 0.149 (21%) | 0.132 (13%) |
| 20 480 | 0.228 ms (37%) | 0.217 (13%) | 0.217 (8.5%) | 0.144 (2.7%) |
| 81 920 | 0.273 ms (32%) | 0.285 (8.9%) | 0.264 (5.2%) | 0.182 (1.7%) |

So it is a constant ~0.15–0.29 ms, which is a third of a single-query call and noise against a
100 000-query one. **The regime it pays in is repeated queries against one mesh**, which is exactly
the `Trimesh` case: `warp_mesh` is a `_CachedProperty`, so the second query onward pays nothing.
It is *not* a way to make one large query faster — there the BVH build has already vanished into the
traversal.

`signed_distance_on_mesh(sign_mode="winding")` refuses a supplied mesh rather than accepting one:
it needs `wp.Mesh(support_winding_number=True)` and `wp.Mesh` exposes no way to read that flag back,
so an unflagged mesh would silently degrade to ray parity. `intersection.mesh_with_mesh` has no such
keyword either, for a different reason — it picks the *smaller* of its two meshes as the BVH target
at runtime, so a caller cannot know which one to build.

### Measured hazards — do not re-enable these without reading the numbers

Clean synthetic meshes **unblocked libigl** where it previously had to be skipped:
`igl.principal_curvature` segfaults on every scan mesh (they have non-manifold vertices and its
vertex-ring walk assumes manifoldness — a hard crash, not an exception) but runs on `sphere_med` in
0.48 s; `igl.heat_geodesics_precompute` raises `Precomputation failed.` on every scan mesh but
succeeds on all sixteen feature meshes; `igl.harmonic` / `igl.lscm` cannot factor the scan meshes'
cotangent systems but handle every mesh on the `patch` axis. So those three comparisons are now
drawn on the same meshes ordito is measured on rather than on two tiny saddle patches.

`potpourri3d` needs the same treatment for a different reason, and the boundary is not
manifoldness but *which* geometry-central mesh class a call builds. Measured:

| call | input | result |
|---|---|---|
| `cotan_laplacian`, `face_areas`, `vertex_areas` | anything | fine — pure numpy/scipy, no mesh built |
| `MeshHeatMethodDistanceSolver`, `marching_triangles` | non-manifold edges | fine — `SurfaceMesh` tolerates them |
| `pp3d.edges` | a mesh with an **unreferenced vertex** | `RuntimeError: GC_SAFETY_ASSERT FAILURE … unreferenced vertex`, on **every scan mesh** |
| `MeshVectorHeatSolver` | non-manifold | `RuntimeError: GC_SAFETY_ASSERT FAILURE … manifold_surface_mesh` |
| `MeshFastMarchingDistanceSolver` | non-manifold | `RuntimeError: handling of nonmanifold mesh not yet implemented` |

So potpourri3d rows live on the synthetic axes, which are manifold and fully referenced by
construction. Where the group's own axis is the scan sweep, the comparison moves to a sibling group
on the `scale` axis rather than being skipped — `edges_unique_manifold` is that case — the same move
that unblocked the libigl comparisons above.

Two more hazards remain and are encoded as explicit skips:

| call | mesh | measured | handling |
|---|---|---|---|
| `igl.principal_curvature` | `fan_hub` | **110 s** (ring walk is quadratic in valence) | `fan_hub` never reaches igl curvature |
| `igl.principal_curvature` | `sphere_large` | ~2 s/call | igl capped at `sphere_med` |
| `open3d.is_watertight` | `sphere_med` | **13.6 s** (brute-force self-intersection scan) | `rounds=1` |
| `open3d.is_watertight` | `tangle_2` | 3.5 s — *faster on the harder mesh*, because the scan early-exits on the first hit | `rounds=1` |
| `pp3d.marching_triangles` | `sphere_med`, 1 072-curve field | 932 ms/call | `rounds=3` |
| `MeshVectorHeatSolver` construction | `sphere_large` | 373 ms/call | `rounds=3` |

Also: never construct a `wp.Mesh` with zero triangles on CUDA (it corrupts device state silently);
`test_meshes.py` asserts every feature mesh is non-empty.

## Reference coverage

Every entry in `LIBRARIES` — `trimesh`, `igl`, `open3d`, `scipy`, `numpy`, `potpourri3d`,
`pymeshlab`, `pyvista`, `meshlib`, `pymeshfix` and `pytorch3d` — is used for
**every** benchmarked function that has a genuine equivalent — the point is to have independent
implementations to spot outliers against, not only to fill gaps. `open3d` is marked `cpu_bound`
even though the installed wheel is a CUDA build: the legacy `open3d.pipelines` / `open3d.geometry`
APIs used here are CPU-only (only `open3d.t` has GPU kernels). `scipy` covers the k-nearest and
radius searches (`spatial.KDTree`) and the graph traversals (`sparse.csgraph`), which is where
trimesh itself delegates. `potpourri3d` is CPU-only (geometry-central) and is the **only** reference
for the heat-method family, tangent spaces and isocontours. `pymeshlab` is CPU-only (MeshLab /
VCGlib) and is the **broadest** — it reaches 26 modules, more than any other single reference.
`pyvista` is CPU-only, single-threaded VTK 9.6 and is the newest; it reaches 47 groups across 26
modules (see its section below). **A group carries `pyvista` or `vedo`, never both** — they wrap the
same VTK, so two rows would double-count one implementation; where a group would take both, `vedo`
gets `noparity(..., oracle="pyvista")`. `meshlib` is the only **multi-threaded** CPU reference, so a
`ordito-cuda` vs `meshlib` ratio is a fair fight where the other CPU ratios are not; `pymeshfix` is
the narrowest and deepest — nine bound algorithms, all repair — and every one of its rows prices its
loader, which is why rows exist only where the operation is ≥ ~30 % of the round. `pytorch3d` is the
only reference with **CUDA kernels of its own**, and it is registered for those alone — one
`pytorch3d-cuda` row and no host row; see its section below for why its GPU ratio is a *crossover*
rather than a bar, and why every one of its rows has to synchronize torch's stream.

#### Known coverage gaps

Recorded so they are not rediscovered as surprises. A gap here is a *benchmark* gap: the function is
public and tested, only untimed.

| function | reference that exists | why there is no group yet |
|---|---|---|
| `visibility.volumetric_obscurance` | `compute_scalar_by_volumetric_obscurance` | it shares `ambient_occlusion`'s kernel and differs only in a per-hit `exp(-tau * t)` factor, so a ordito-only group would re-measure that group's axis. MeshLab's filter *is* a real second reference, so the row is worth adding — with `tau` as its axis, not `rays`. |
| `measures.volume` | `get_geometric_measures["mesh_volume"]`, `trimesh.Trimesh.volume`, `igl.moments[0]` | untimed since before it moved out of `triangles.py`. Its device pass is now `triangles.face_signed_volumes` plus one `reduce.sum`, and the `moments` row already prices the same integrals at three readbacks against this one's — so the row would isolate the readback count, which is the axis this module is about. Worth adding. |
| `measures.euler_characteristic` | `get_geometric_measures["genus"]`, `trimesh.Trimesh.euler_number` | untimed since before it moved out of `validation.py`. It is a `unique_1d` plus an `edges_unique`, so it prices the grouping machinery rather than anything of its own, which is the argument against — but it is also the only whole-mesh integer with no row at all. |
| `triangles.face_signed_volumes` | none | deliberately none: it is the per-face primitive `measures.volume` reduces, and a group over it would measure the same launch twice. Time it through a `volume` row instead. |
| `energies.lscm_hessian`, `energies.vector_area_matrix` | none with a binding | libigl exposes neither on its own — `lscm_hessian` only as `igl.lscm`'s second return, and `vector_area_matrix` not at all (`tests/test_energies.py` derives it as `(-repdiag(L, 2) - Q) / 2`). Their cost is measured through the `lscm` group in `test_parametrization`, which is also the only way a caller reaches them. |

### The parity gate: a benchmarked reference must be a tested reference

Nothing in this directory asserts that two timed implementations compute the same thing — every
assert here is a shape, `isfinite` or count guard whose only job is to prove the timed work was not
optimised away. `tests/test_parity.py` closes that loop and **fails the default `pytest` run** when a
benchmarked `(group, library)` pair is neither tested nor exempted. Run
`uv run python -m tests.parity` for the full matrix.

The `benchmark(group=...)` name is the join key, which makes group names a **cross-suite API**:
renaming one breaks every `parity` marker in `tests/` that cites it.

Either a test in `tests/` claims the pair:

```python
@pytest.mark.parity("faces_to_edges", "trimesh")     # one marker per group, all its libraries listed
```

…or the benchmark function declares, at its own call site and in prose, why the two results are not
comparable:

```python
@pytest.mark.noparity("pymeshlab", oracle="trimesh", reason="D2 ...measured number... trimesh is "
                      "the oracle for this group, in tests/test_x.py::test_y.")
```

`reason` is mandatory and is checked for substance (≥ 40 chars, ≥ 6 words, no `"see above"`-class
filler). `oracle=` names the library that *is* the oracle, and the gate then requires **that** pair to
be covered — so an exemption is a checked claim rather than an escape hatch. Both markers take string
literals only; a computed argument is invisible to the static scan and is rejected. Twenty-seven of the
346 pairs are exempt; the category rubric (D1–D6) is in `.claude/CLAUDE.md` §6.

**Exemptions are for results that cannot be compared, not comparisons that are awkward.** Three rows
here were heading for exemptions on an 8%-of-displacement Laplacian disagreement until MeshLab's
actual smoothing stencil was recovered numerically; they are now exact class-B asserts. "Does strictly
less", "the output shape differs" and "the ordering differs" are transforms, not exemptions.

### pymeshlab

MeshLab exposes 281 filters. 61 of them map onto something ordito already has, and they are what
these rows are drawn from. Eight of the groups they land in **had no reference of any kind** before:

| module | group | filter | what it settled |
|---|---|---|---|
| `test_linalg` | `min_quad_with_fixed` | `compute_scalar_by_scalar_harmonic_field_per_vertex` | its direct solve is **flat** (39.0 / 39.6 ms) across the `quality` axis where ordito's CG goes 33.7 → 83.5, so the whole spread is iteration count |
| `test_selection` | `expand_vertex_mask` / `shrink_vertex_mask` | `apply_selection_dilatation` / `..._erosion` | slopes of 7.3× and 8.0× over 1 → 8 hops, confirming a hop is constant work |
| `test_proximity` | `signed_distance_on_mesh` | `compute_scalar_by_distance_from_another_mesh_per_vertex` | its per-query cost **grows** with the face count (20 / 63 / 768 µs) where ordito's BVH does not |
| `test_smoothing` | `filter_taubin`, `filter_humphrey` (new groups) | `apply_coord_taubin_smoothing`, `apply_coord_hc_laplacian_smoothing` | two ordito filters that were unbenchmarked |
| `test_remesh` | `isotropic_remesh` | `meshing_isotropic_explicit_remeshing` | 3.3× across `quality` against ordito's **flat** 123 ms — the fixed-pass loop is not adapting |
| `test_reduce` | `median` | `get_scalar_statistics_per_vertex` | six statistics in one call, 2.5 / 11 / 110× |
| `test_vertices` | `average_onto_vertices` | `compute_scalar_transfer_face_to_vertex` | also *faster* on `fan_hub` — valence is not a hot spot, independently |
| `test_repair` | `remove_non_manifold_faces` | `meshing_repair_non_manifold_edges` | nothing else in the set removes non-manifold faces at all |

The rest are second or third independent implementations:

| module | pymeshlab reference |
|---|---|
| `test_boundary` | `compute_selection_from_mesh_border` (`boundary_edges` only — no loop ordering) |
| `test_combine` | `generate_splitting_by_connected_components` — one call for label *and* compaction, and a **39× spread** across the `components` axis against ordito's 3.7× |
| `test_convex` | `generate_convex_hull` (qhull a third time, so it prices the wrapper rather than the algorithm) |
| `test_creation` | `create_cube`, `create_sphere` (an *icosphere*, so it pairs with `icosphere` where open3d's pairs with `uv_sphere`), `create_torus`, `create_annulus`, `create_cone` |
| `test_curvature` | `compute_curvature_principal_directions_per_vertex(method='Quadric Fitting')` and `compute_scalar_by_discrete_curvature_per_vertex` — the only reference that survives the whole `scale` axis, where trimesh is capped at `sphere_small` |
| `test_metrics` | `get_hausdorff_distance` (one-directional, so both directions are timed) |
| `test_edges` | `get_geometric_measures()['avg_edge_length']` |
| `test_heat_distance` | `compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex` — the fourth heat-method implementation, and the only one whose amortized path is just "call it twice"; plus `..._geodesic_distance_from_given_point_...` as a non-PDE alternative |
| `test_graph` | `compute_selection_by_small_disconnected_components_per_face(nbfaceratio=0.0)` |
| `test_holes` | `meshing_close_holes` (ear clipping, so **1.7×** across `loops_dp` against ordito's 37× — the price of *not* running a `B³` DP) |
| `test_parametrization` | `compute_texcoord_parametrization_harmonic` / `..._least_squares_conformal_maps` — both wrap **libigl's own code**, so they price MeshLab's wrapper rather than a third algorithm |
| `test_points` | `compute_normal_for_point_clouds(k=)`, `compute_matrix_by_fitting_to_plane` |
| `test_reconstruction` | `generate_surface_reconstruction_ball_pivoting` (VCGlib's original BPA). `..._screened_poisson` was timed here too and is **removed** — see the wall-clock note above |
| `test_registration` | `compute_matrix_by_icp_between_meshes` — the only reference the mesh-target `icp` group has |
| `test_remesh` | `meshing_surface_subdivision_midpoint(threshold=)` in `subdivide_to_size`, landing on the identical output face count |
| `test_sample` | `generate_sampling_poisson_disk(radius=)` — the only blue-noise reference that takes a *radius*, so the radius sweep maps for the first time |
| `test_smoothing` | `apply_coord_laplacian_smoothing_scale_dependent` (Desbrun's, = `filter_mut_dif_laplacian`) and `apply_coord_laplacian_smoothing` |
| `test_triangles` | `compute_normal_per_face` |
| `test_measures` | `get_geometric_measures` (`shell_barycenter` = `surface_centroid`) |
| `test_validation` | `get_topological_measures` + `compute_selection_by_self_intersections_per_face` — the same composition open3d does in **13.8 s** and MeshLab in 140.8 ms; plus `compute_selection_by_non_manifold_per_vertex` |
| `test_vertices` | `compute_normal_per_vertex` at `weightmode='Simple Average'` and `'By Area'` — one filter covering both normal groups |

#### Ported gaps: rows where pymeshlab is the *source*, not the reference

The 24 filters MeshLab exposed that ordito did not have were ported. Their benchmark rows read
the other way round — the pymeshlab filter is the thing being caught up with, and in every case
the port is the newer code:

| group | module | pymeshlab source filter | measured |
|---|---|---|---|
| `face_quality` | `test_triangles` | `compute_scalar_by_aspect_ratio_per_face` | 34 µs against 445 µs, flat across `quality` on both sides |
| `outlier_probability` | `test_points` | `compute_selection_point_cloud_outliers` (LoOP) | 2.6 ms against 0.66 ms at `sphere_small` — the k-NN table dominates, see `test_neighbors` |
| `platonic_solids`, `grid`, `sphere_cap` | `test_creation` | `create_{tetrahedron,octahedron,dodecahedron,grid,sphere_cap}` | fixed-cost ties on the tables; the cap is flat in resolution where MeshLab's per-vertex loop is not |
| `filter_scalar_laplacian` | `test_smoothing` | `apply_scalar_smoothing_per_vertex` | 0.41 ms against 7.4 ms on `bunny_decimated` |
| `shortest_path_envelope` | `test_graph` | `apply_scalar_saturation_per_vertex` | 0.17 ms against 5.3 ms on a spike-seeded Lipschitz projection |
| `transfer_onto_vertices` | `test_vertices` | `transfer_attributes_per_vertex` | closest-point plus barycentric blend, against a serial closest-point walk |
| `cluster_decimate` | `test_remesh` | `meshing_decimation_clustering` | exact agreement with `open3d.simplify_vertex_clustering` on face *and* vertex count |
| `ambient_occlusion`, `shape_diameter` | `test_visibility` | `compute_scalar_ambient_occlusion`, `..._shape_diameter_function_per_vertex` | **9.3 ms against 662 ms** on `bunny` at 64 rays — the largest ratio in the suite |
| `flip_by_objective` | `test_remesh` | `meshing_edge_flip_by_planar_optimization` | 2.0 ms against 28.6 ms on `saddle_graded` |
| `face_defective_mask`, `flip_t_vertices` | `test_validation`, `test_repair` | `compute_selection_bad_faces`, `meshing_remove_t_vertices` | 8.9 ms against 24.2 ms on `saddle_graded` |
| `crease_edges`, `cut_along_edges` | `test_seams` | `compute_selection_crease_per_edge`, `meshing_cut_along_crease_edges` | 1.3 ms against 51.6 ms on `sphere_med`, and **24 output vertices against 32** on a cut cube |
| `uv_seam_edges` | `test_seams` | `compute_selection_by_texture_seams_per_vertex` | **0.94 ms against 300 ms** on `sphere_large`; the filter returns only the seam *vertex set*, unioned with the boundary, so it does strictly less than the ordito row (which also splits boundaries out and finds foldovers) |
| `filter_normals`, `filter_two_step`, `filter_sharpen` | `test_smoothing` | `apply_normal_smoothing_per_face`, `apply_coord_two_steps_smoothing`, `apply_coord_unsharp_mask` | 8.3 ms against 124 ms; 0.34 ms against 51.8 ms |
| `resample_uniform` | `test_reconstruction` | `generate_resampled_uniform_mesh` | 13.5 ms against 292 ms on `bunny` at a 1% cell |
| `quadric_decimate` | `test_remesh` | `meshing_decimation_quadric_edge_collapse` | **the one port that is slower**: ~253 ms on `saddle_graded` at `target_ratio=0.1` against igl's 80 — see below |

Two of those rows are worth reading as findings rather than ratios:

- **`quadric_decimate` trades speed for quality.** Its Hausdorff error at 512 faces from a subdiv-4
  icosphere is 0.0133 against `igl.decimate`'s 0.0250 and Open3D's 0.0236, and it is ~3× slower than
  igl (253 ms on `saddle_graded` at `target_ratio=0.1` against 80). The cost is the *pass count*, not
  the per-pass work: one hashed-key independent set commits roughly `candidates / 50` collapses, and
  each pass rebuilds the edge list, adjacency, quadrics and two radix sorts. A serial queue pays none
  of that per collapse. Committing **several** independent sets against one rebuild closed 1.3–1.8× of
  the gap (`saddle_graded` at 0.1: 461 → 253 ms); the rest is structural. Full numbers in
  `test_remesh.py`'s `_QUADRIC_RATIOS` comment.
- **`cut_along_edges` is faster when it cuts *more*.** Cutting every interior edge of `sphere_med`
  costs 1.34 ms against 1.63 ms for a quarter of them, because the corner graph then has no edges and
  the connected-components pass converges immediately instead of contracting `3F` nodes into `V`. So
  the components contraction is the cost in that group, not the ray/key work.

#### Hazards, all measured

- **The MeshSet build is the floor under every row**, at ~0.47 µs/vertex: 0.91 ms on 2 562 vertices,
  4.55 ms on 10 242, **17.1 ms on `bunny`**'s 35 947. Any row cheaper than that is reporting the
  build. It is a large share of the mutating rows — 16.4 of the 17.8 ms a clean
  `meshing_remove_duplicate_faces` costs, 35% of a ten-step Laplacian on `bunny` — so subtract it
  before quoting a ratio.
- **Almost every filter mutates `current_mesh()` in place**, so the MeshSet goes *inside* the timed
  callable (`BenchCase.new_meshset_pml`), the same precedent as the trimesh rebuilds and
  potpourri3d's in-callable solver construction. `BenchCase.meshset_pml` is the shared exception, for
  filters verified geometry-preserving: the `compute_scalar_*` / `compute_normal_*` family, and the
  selection filters — whose cost is *independent of how much is selected* (120 consecutive dilatations
  carrying `sphere_med` from 0.9% to 86% selected cost 0.86–1.21 ms each), which is what lets
  `test_selection` seed once outside the timed region and still read a clean per-hop cost.
- **`compute_curvature_principal_directions_per_vertex` and
  `meshing_decimation_quadric_edge_collapse` default to `autoclean=True`** and delete unreferenced
  vertices under you, so they cannot share a MeshSet even though they only write attributes.
- **Length parameters take a wrapper type.** `ml.PercentageValue(1)` is 1% of the bbox diagonal;
  `ml.PureValue(x)` is absolute (this version has no `AbsoluteValue`). Every row that sizes work by
  length passes `PureValue` fed from `bench_case.mean_edge` so both sides get the identical parameter.
- **Four filters have parameters that silently measure nothing**, and one of them is a value *we*
  chose. `meshing_close_holes(maxholesize=30)` closes *zero* of `rim_short`'s two 512-edge rims
  (0.66 ms); `get_hausdorff_distance(samplenum=8)` samples eight points out of the whole cloud;
  `generate_sampling_poisson_disk(radius=0%)` autoguesses a radius instead of using yours. And
  `generate_surface_reconstruction_ball_pivoting(clustering=0)` reconstructs **nothing** — 0 faces
  against 1 277 at MeshLab's 20% default, returning *faster* for it (9.6 ms against 2.7 ms), so the
  row read as a ordito win while timing a failure. The clustering fraction is a seed-triangle
  spacing floor, not the optional merge-nearby-vertices post-pass it looks like. Each row overrides
  these and asserts the reference produced output. **Probe a parameter's zero before assuming it means
  "off".**
- **Pass counts are conventions, and mismatching them silently doubles a row's work.** MeshLab's
  `apply_coord_taubin_smoothing(stepsmoothnum=n)` runs `n` lambda-mu **pairs** where ordito and
  trimesh do one half-step per `iterations` and alternate, so the row passes `_ITERATIONS // 2`;
  passing `_ITERATIONS` to both timed MeshLab doing twice the passes. The mapping is pinned exactly
  (5e-08) in `tests/test_smoothing.py::test_filter_taubin_matches_pymeshlab`.
- **MeshLab has two different uniform coordinate umbrellas.** `apply_coord_laplacian_smoothing` and
  `apply_coord_unsharp_mask` weight each neighbour by its shared-face count and include the vertex
  itself once (`1/(2d+1)` self, `2/(2d+1)` per neighbour on a closed mesh); `apply_coord_taubin_
  smoothing` uses the plain 1-ring mean. Neither is documented; both were recovered by solving
  least-squares for the per-vertex stencil over 12 random position sets on one connectivity (residual
  2e-16). The difference from the 1-ring mean is 8% of the displacement.
- **`compute_matrix_by_icp_between_meshes` does not move the vertices.** It writes the source layer's
  *transformation matrix*, so `vertex_matrix()` reads back byte-identical to the input and a converged
  registration looks like a total no-op; the answer is `transform_matrix()` /
  `transformed_vertex_matrix()`.
- **Non-manifold input is a hard boundary, not a slow path** — the same one that shapes the libigl and
  potpourri3d rows. `meshing_surface_subdivision_midpoint` raises `Mesh has some not 2 manifold faces,
  subdivision surfaces require manifoldness` on **every scan mesh**, so its reference row lives on the
  `scale` axis with `subdivide_to_size`; `generate_polyline_from_planar_section` raises
  `Failed to apply filter` on every scan mesh at any plane, and `test_intersection`'s groups are *all*
  on the scan sweep, so that row does not exist at all.
- **Two more preconditions raise rather than degrade.**
  `compute_texcoord_parametrization_{harmonic,lscm}` reject a **closed** mesh (a boundary loop is
  required); `compute_matrix_by_fitting_to_plane` raises `Cannot compute rotation: there is no
  selection` unless something is selected; and `compute_matrix_by_icp_between_meshes` needs **both**
  layers to carry faces.
- **`harm_function` is a no-op in pymeshlab 2025.7.** `compute_texcoord_parametrization_harmonic`
  returns **bit-identical** texture coordinates at `harm_function=1`, `2` and `3` (max deviation
  exactly 0.0) and identical timings, where libigl's own `k=2` costs 4.3× its `k=1`. So the harmonic
  order axis does not map and the pymeshlab row appears at `k=1` only — a row tracking ordito's
  `k=2` would be silently reporting the `k=1` solve.
- **Selection morphology is face-based.** `apply_selection_dilatation` / `..._erosion` dilate the
  *face* set (VCGlib's loose vertex-from-face / face-from-vertex pair); handing them a vertex
  selection simply clears it. Seeding needs `compute_selection_by_condition_per_face`, or
  `compute_selection_transfer_vertex_to_face(inclusive=False)` from a vertex seed — `inclusive=True`,
  the default, selects only faces whose *every* vertex is selected.
- **`generate_surface_reconstruction_vcg` was tried and rejected**: it returns **zero faces** on an
  oriented point cloud at every voxel size probed (`PureValue` 0.02 / 0.05 / 0.10), reporting
  `Mesh Saved 'plymcout.ply': 0 vertices, 0 faces`. It reconstructs from all *visible* layers via a
  temporary `.vmi` file and did not produce geometry from a single cloud.
- **Three more silent-no-op parameters, found while porting bucket B.**
  `compute_scalar_by_shape_diameter_function_per_vertex`'s `cone_amplitude` is a **no-op**: the output
  is byte-identical at 90 and 120 degrees. `generate_resampled_uniform_mesh`'s `offset` as a
  `PercentageValue` runs from *full erosion* at 0% to full dilation at 100%, so its own 50% default is
  the **zero** offset and `PercentageValue(0)` erodes a unit sphere to radius 0.30 — pass
  `PureValue(0.0)` for an absolute zero. And `apply_normal_smoothing_per_face` exposes no parameters at
  all, so its row is a single pass against ordito's twenty.
- **Two filters disagree with the port in ways that are not tolerances.**
  `apply_scalar_smoothing_per_vertex` smooths a *boundary* vertex along the boundary curve alone
  (dividing by 2 rather than by its degree), so the oracle in `tests/test_smoothing.py` runs on closed
  fixtures only. `apply_coord_two_steps_smoothing` at its own four defaults moves a noisy cube
  **further** from clean than the noise was (RMS 0.0244 against 0.0156) because its fitting step rounds
  the corners in — so that comparison asserts crease preservation, which both pass, and a one-sided
  RMS bound.
- **`bunny_decimated` is not edge-manifold** (150 edges with three or more faces), which is why
  `test_seams`' cut *and* `uv_seam_edges` groups run on the synthetic icospheres: both are defined
  through halfedge twins, and ordito and MeshLab's `meshing_cut_along_crease_edges` alike reject
  such a mesh outright rather than degrading.
- **The registry meshes carry no UVs at all**, so `uv_seam_edges` synthesizes a per-corner spherical
  atlas (seamed at the `±π` longitude wrap) and feeds the *same* array to both sides through
  MeshLab's `w_tex_coords_matrix`. A continuous atlas would leave the seam set empty and a
  per-triangle one would make every interior edge a seam, so neither would time the compaction.
- **Several filters print to stdout regardless of `verbose=False`** (ICP's `Found N pairs`, the point-
  cloud normal estimator's `UG 34 34 34`, the VCG reconstructor's whole volume report). pytest's
  fd-level capture absorbs it; a bare script will not.
- **Three groups are capped** because the reference, not ordito, is the cost:
  `signed_distance_on_mesh` at `bunny` (768 µs/query on dragon = 85 s a row),
  `get_geometric_measures` at `bunny` (1.12 s a call on dragon, in *two* modules), and the
  `apply_coord_*` smoothers at `bunny` alongside the trimesh rows.

### libigl

igl is the reference whose input convention matches the harness most closely: `BenchCase.vertices_np`
is `float64` `(n, 3)` and `BenchCase.faces_np` is `int64` `(n_faces, 3)`, which is exactly what every
binding wants, so an igl row is a `benchlibs` entry plus an `elif bench_case.kind == "igl":` branch
with no conversion layer. Every bound function is **pure** — arrays in, arrays out — so unlike
pymeshlab there is no per-call construction cost to amortize and no in-place mutation to defend
against. The exceptions are the stateful solver objects (`HeatGeodesicsData`, `ARAPData`, `SLIMData`,
`min_quad_with_fixed_data`, `AABB`), which cache a factorization and therefore go **inside** the timed
callable, for the same reason potpourri3d's solvers do.

Only **150** of the ~493 top-level C++ headers are bound, so confirm a name exists in the installed
wheel before planning a row around it. Two that do not, and will be looked for because
`ordito.repair` carries functions named after them: **`collapse_small_triangles` and
`resolve_duplicated_faces` are not bound** (`AttributeError`), so neither group can have an igl row.

#### Coverage: 74 pairs over 25 modules

igl reaches **74** of the matrix's 273 `(group, library)` pairs — **the most-covered reference in
the suite** (trimesh and pymeshlab are next at 71 each), up from 27 before the coverage pass and 53
before the twelve igl-parity ports landed. Nine of its groups did not exist before
the coverage pass: `sample_surface`, `face_angles`, `vertex_defects`,
`face_connected_component_labels`, `ears`, `is_edge_manifold`, `remove_unreferenced_vertices`,
`unique_faces`, plus the `icosahedron` case of `platonic_solids` — and the port pass added the
operator family in `test_energies` (`k_harmonic`, `hessian_energy`, `curved_hessian_energy`,
`crouzeix_raviart_cotmatrix` / `_massmatrix`), all five on the `scale` axis because igl's
Crouzeix-Raviart entry points assume edge-manifold input.

`test_voxels` added two more: `igl.grid` for the corner lattice and
`igl.unique_sparse_voxel_corners` for the deduplicated cell corners. The second has a trap worth
recording — its subscripts are packed with radix `2 ** depth + 1`, so the documented `depth=0` makes
every coordinate alias and it silently returns a handful of corners for hundreds of cells. It also
reads column 0 of a subscript as *y* and numbers a cell's corners in yxz binary-counting order, both
of which are named transforms in the parity test.

Where the margins sit, on medians:

| group | igl | ordito-cuda | ratio |
|---|---|---|---|
| `face_angles` | 12.5 ms | 0.044 ms | **282x** |
| `ears` | 161–399 ms | 0.72–0.84 ms | **200–500x** |
| `bvh_from_points` | 198 ms | 0.22 ms | 896x (but see the GC hazard below) |
| `query_nearest_bvh_k1` | 250 ms | 0.65 ms | 385x |
| `sample_surface` (100k) | 55.2 ms | 0.25 ms | 218x |
| `vertex_defects` | 13.9–14.7 ms | 0.19–0.22 ms | 62–78x |
| `is_edge_manifold` | 32–53 ms | 1.15–1.18 ms | 28–48x |
| `face_orientation_bits` | 62 / 17.3 ms | 1.99 / 1.00 ms | 31x / 17x |
| `signed_distance_on_mesh` | 150–520 ms | 4.0–6.6 ms | 10–100x |
| `blue_noise` | 334 / 1 592 ms | 55 / 140 ms | 6.1x / 11.4x |
| `face_adjacency` | 16.5 ms | 0.65 ms | 25x |
| `face_connected_component_labels` | 15.2 ms | 1.27 ms | 12x |
| `make_winding_consistent` | 22.8 ms | 1.02 ms | 22x |
| `remove_unreferenced_vertices` | 0.73 ms | 0.30 ms | 2.5x |
| `crouzeix_raviart_cotmatrix` † | 3.6–84.8 ms | 0.37–0.99 ms | 9.6–86x |
| `curved_hessian_energy` † | 24.8–1 578 ms | 2.3–20.3 ms | 11–78x |
| `crouzeix_raviart_massmatrix` † | 0.15–13.0 ms | 0.18–0.21 ms | **0.73x**–72x |
| `hessian_energy` † | 17.2–1 137 ms | 1.3–22.3 ms | 13–51x |
| `k_harmonic` (k=2) † | 1.7–162.6 ms | 0.96–4.6 ms | 1.7–35x |

† Operator-family rows are the `scale` axis (`sphere_small` → `sphere_med` → `sphere_large`); the
ranges span it, and the ratio grows with size on every one of them because igl's side is Eigen
sparse products where ordito's is a fixed number of launches.

**One igl row now loses one axis point**: `crouzeix_raviart_massmatrix` at `sphere_small` reads igl
0.150 ms against ordito 0.206 ms (1.37x). ordito's side is flat at 0.18–0.21 ms across the axis's
16x size range — a pure host launch/alloc floor (one scatter kernel plus `bsr_diag`), not kernel
time — and the same row is a 72x win at `sphere_large`. Every other igl row ordito wins, and one of
them used to be the exception: `blue_noise`
was a **0.80x loss** (igl 78.4 ms against 98.5) while `sample_surface_blue_noise` ran Bridson
active-list dart throwing, the algorithm `igl.blue_noise` implements. After the rewrite to
randomized-priority selection the same pair reads 4.1x at that radius, and the margin *grows as the
radius falls* — 5.4x at the 2k-sample radius, 9.0x at half of it, 24x on `bunny_decimated` — because
igl's serial cost is per accepted sample where ordito's is per round. igl also returns 2–7% *fewer*
samples at the same radius and has the tightest coverage of the three references (1.073 r against
ordito's 1.103), so neither side is trading quality for speed. That row is the clearest argument in
the suite for keeping a reference that a port was originally written from.

Two rows read the other way round and are worth knowing before quoting them: `face_adjacency_unshared`
and `face_adjacency` time the **same** `igl.triangle_triangle_adjacency` call, because igl computes
both in one pass — so the `unshared` row is an upper bound rather than like-for-like — and
`chamfer_points_to_mesh`'s igl row is the *forward half only*, a lower bound.

#### Hazards, all measured

- **An out-of-range face index is a SIGSEGV, not an exception.** `igl.cotmatrix(V, F)` with one entry
  of `F` set to `len(V) + 500` kills the interpreter with **exit code 139** and no traceback; igl
  bounds-checks nothing. So every igl row is handed the *unreduced* `(V, F)` pair from `BenchCase`,
  never a `remove_unreferenced`-style reduced `V` with the original `F`. This is the same failure class
  as `igl.principal_curvature` on the scan meshes' non-manifold vertices, above.
- **F-only functions size their output by `F.max() + 1`, not by `len(V)`.** Measured on an
  `icosphere(2)` padded with five unreferenced trailing vertices (167 V, 162 referenced):
  `igl.adjacency_matrix`, `igl.vertex_components` and `igl.is_vertex_manifold` return **162** rows
  where `igl.cotmatrix` and `igl.gaussian_curvature` return **167**. The two families disagree with
  each other and only the `(V, F)` family matches ordito. This is not academic: **`bunny` has 1 113
  unreferenced vertices** (`igl.remove_unreferenced` returns 34 834 of 35 947), so
  `igl.connected_components(igl.adjacency_matrix(F))` reports **1 114 components** on it — 1 113
  isolated vertices plus the mesh. Every F-only parity assert is therefore class B with the transform
  named, and must run on a fixture where the distinction is visible.
- **`igl.octree` is priced by the garbage collector, not by the tree.** It returns its per-cell point
  lists as a Python `list` of **151 233 nested lists** on `bunny`'s 35 947 points, so it allocates
  ~150k list objects a call: **40.6–44.6 ms with `gc` disabled against 90.6–176.9 ms with it enabled**,
  measured back to back. In-harness that shows up as a `Min` of 39 ms against a `Median` of 198 ms in
  one ten-round row — so **quote igl's medians, never its minima**, and read `query_nearest_bvh_*`'s
  build-included igl numbers as pricing the binding as much as the search. This is the `*_lists`
  hazard below, except it lands on the only structure igl exposes for k-NN, so there is no array form
  to switch to; it is also why the igl k-NN rows are capped a size below scipy's.
- **Nine call signatures are not what the docs suggest, and two of them fail silently.** Each of these
  failed on the first plausible attempt:

| function | the trap | the working call |
|---|---|---|
| `exact_geodesic` | **Returns an empty array** rather than raising when the trailing arguments are omitted: `VS/FS/VT/FT` all default to `array([])`, so a 4-argument call binds `vt` to `FS` | `igl.exact_geodesic(V, F, VS, FS, VT, FT)`, face arrays explicitly `np.array([], dtype=np.int64)` |
| `knn` | Seven positional arguments; the octree is built over the *data* cloud, not the queries | `igl.knn(queries, points, k, *igl.octree(points)[:4])` |
| `in_element` | Takes a live `igl.AABB`; there is no 3-argument overload | `aabb = igl.AABB(); aabb.init(V, Ele); igl.in_element(V, Ele, Q, aabb)` |
| `crouzeix_raviart_cotmatrix` / `..._massmatrix` | Need `(V, F, E, EMAP)`, not `(V, F)` | `uem = igl.unique_edge_map(F)`, then `uem[1]` as `E` and `uem[2].ravel()` as `EMAP` |
| `slim_precompute` | **Returns** the `SLIMData` rather than taking one, and demands Fortran order with **int32** faces, unlike every other binding | `data = igl.slim_precompute(np.asfortranarray(V), F.astype(np.int32), V_init, ...)` |
| `isolines_intrinsic` | Takes `(F, S, vals)`; the `uE/EMAP/uEC/uEE` quintet the C++ header shows is not in the binding | `igl.isolines_intrinsic(F, S, vals)` |
| `average_onto_vertices` | Its `S` is a per-face **scalar** `(n_faces,)`, not a per-face vector | `igl.average_onto_vertices(V, F, face_scalars)` |
| `cut_mesh` | `C` is a **bool** `(n_faces, 3)` per-corner cut mask, not an edge list | `igl.cut_mesh(V, F, cuts_bool)` |

- **The `*_lists` variants cannot be timed as rows.** `adjacency_list`,
  `triangle_triangle_adjacency_lists`, `vertex_triangle_adjacency_lists` and `unique_edge_map_lists`
  are the same computation as their array siblings with a `list[list[int]]` result, so a row would
  price nanobind's list construction (`triangle_triangle_adjacency_lists` is 321 ms on `bunny` against
  4.25 ms for the array form). Use the array forms. `igl.is_border_vertex` returns a Python
  `list[bool]` and has no array form, so its 3.4 ms is read as an upper bound.

### open3d — hazards, all found while writing parity asserts

The first two fail *silently* rather than raising:

- **A tensor mesh must be held in a name.** Chaining
  `o3d.t.geometry.TriangleMesh.from_legacy(x).fill_holes()` lets the temporary be collected, and the
  result's `vertex["positions"]` then reads freed memory — observed as 2052.1 and 4.4e-41 in the
  leading rows. Bind the intermediate before calling the filter
  (`tests.conversions.trimesh_to_open3d_t` exists so the binding is structural).
- **`fill_holes` winds its cap against the rest of the mesh.** A raw signed volume of its output is
  therefore meaningless: −1.06 on a hemisphere whose true sealed volume is 2.02. Run
  `trimesh.repair.fix_winding` (or ordito's `make_winding_consistent`) before any volume or
  orientation read. Its cap triangulation is otherwise a valid `B − 2` fill over the existing
  vertices, matching ordito's and MeshLab's counts exactly.
- **k-NN distances come back squared**, from both the legacy `KDTreeFlann` and the batched
  `o3d.core.nns` searches — take the square root before any `allclose`. And use `o3d.core.nns`
  for anything batched: the legacy tree's only query is per-point Python (62 ms against 10 ms at
  20 000 queries on a 36k cloud), while `nns.knn_search` matches `scipy.spatial.KDTree` indices
  byte-for-byte on a tie-free cloud.
- **`KDTreeFlann.search_radius_vector_3d` is exclusive at exactly `r`** (a point at distance
  exactly 1.0 is not returned at radius 1.0) where ordito's ball queries are inclusive. Random
  clouds never tie, so only constructed fixtures can see the difference — construct accordingly.
- **`get_volume` validates before it integrates**, and the validation is the full brute-force
  `IsWatertight` composition: 13.8 s on a watertight 82k-face sphere whose divergence integral is
  microseconds. Never put it inside a timed row that claims to measure volume.

### pyvista

VTK through pyvista 0.48 / VTK 9.6, and the registration cost nothing: `pyvista>=0.48` was already in
the `test` dependency group. `PolyData`'s callable surface is **142 distinct filter names** across its
three mixins; 28 of them map onto something ordito already has, and those 28 carry **47 group rows**
across 26 modules (one filter often answers several groups — `cell_quality` alone covers
`face_quality` and `face_angles`, and `find_closest_cell` answers both `closest_point_on_mesh` and
`polyline_point_distance`). Every one is single-threaded CPU VTK, so `pyvista` is `cpu_bound` and the
heavy filters carry explicit caps.

The seven newest rows were added for a different reason from the forty that preceded them, and it is
worth saying which: each is a group whose *only* other reference is `meshlib` or `scipy`, and MeshLib
is the one reference here under a non-commercial licence. They take the count of groups that depend
on MeshLib alone from 22 to 16.

Two mechanics that differ from the other references:

- **The `PolyData` is shared across rows** (`bench_case.mesh_pv`), which is safe because pyvista
  caches nothing — a repeat `cell_quality` or `decimate` recomputes in full (measured 10.1 / 7.2 ms
  and 210 / 201 ms) — and almost every filter returns a new object (`inplace=False` throughout).
  That is the open3d situation rather than the pymeshlab one. Two exceptions build inside the timed
  callable: `edge_mask`, which writes `point_ind` into its input, and any row that passes
  `inplace=True` (none do).
- **The build is cheap and goes through `from_regular_faces`**: 0.057 µs/vertex on 41k vertices,
  ~8× cheaper than pymeshlab's 0.47 µs/vertex, so no row below ~0.2 ms is reporting its own build.
  The padded `[3, i, j, k]` constructor is 13× slower (2.34 ms against 0.184 ms on 41k vertices) and
  `tests.conversions.trimesh_to_pyvista` exists to keep every call on the fast path.

| module | groups | VTK entry points |
|---|---|---|
| `test_triangles` | `face_quality`, `face_angles`, `face_centroids`, `face_normals_and_areas` | `cell_quality` (Verdict), `cell_centers`, `compute_normals` + `compute_cell_sizes` |
| `test_vertices` | `mean_vertex_normals`, `vertex_defects` | `compute_normals` (point), `curvature('gaussian')` |
| `test_laplacian` | `face_gradients` | `compute_derivative(gradient=True)` |
| `test_curvature` | `principal_curvature` (exempt) | `curvature('maximum')` — algebra on VTK's own K and H |
| `test_edges` / `test_seams` / `test_boundary` | `edges_unique`, `crease_edges`, `boundary_edges` | `extract_all_edges`, `extract_feature_edges` with one class on |
| `test_validation` / `test_adjacency` | `is_edge_manifold`, `face_connected_component_labels` | `is_manifold`, `connectivity('all')` |
| `test_interpolation` | `average_onto_vertices`, `average_onto_faces` | `cell_data_to_point_data`, `point_data_to_cell_data` |
| `test_reduce` | `weighted_sum` (new group) | `integrate_data` |
| `test_proximity` | `winding_number`, `signed_distance_on_mesh` | `select_interior_points`, `compute_implicit_distance` |
| `test_heat_distance` | `heat_geodesic` | `geodesic_distance` (Dijkstra over edges — a bound, not the same quantity) |
| `test_bounds` / `test_measures` | `aabb`, `enclosing_diagonal`, `oriented_bounding_box`, `moments` | `bounds`, `length`, `oriented_bounding_box` (PCA), `volume` |
| `test_points` | `principal_axes`, `fit_plane`, `fit_line` (exempt) | `principal_axes`, `fit_plane_to_points`, `fit_line_to_points` |
| `test_registration` | `icp_mesh` | `align(return_matrix=True)` |
| `test_remesh` / `test_smoothing` | `quadric_decimate`, `filter_laplacian_integration` (exempt), `filter_taubin` (exempt) | `decimate`, `smooth`, `smooth_taubin` |
| `test_repair` / `test_voxels` | `remove_duplicated_vertices`, `voxelize_mesh` | `clean`, `voxelize_binary_mask` |
| `test_reconstruction` | `delaunay_triangulation` (new group) | `delaunay_2d` |
| `test_polyline` | `polyline_length`, `polyline_point_distance`, `polyline_triangulate` | `compute_arc_length`, `find_closest_cell`, `triangulate_contours` — all three on a **single-cell** polyline |
| `test_proximity` | `closest_point_on_mesh`, `containing_faces_2d` | `find_closest_cell`, `find_containing_cell` (`vtkStaticCellLocator`, batched) |
| `test_selection` | `region_boundary_edges` | `extract_cells` + `extract_feature_edges(boundary_edges=True)` — a superset, minus the mesh rim |
| `test_intersection` | `mesh_with_mesh` | `intersection` (`vtkIntersectionPolyDataFilter`) |
| `test_intersection` / `test_interpolation` / `test_creation` | `clip_mesh_with_field`, `interpolate_from_points`, `parametric_surface`, `super_ellipsoid`, `super_toroid`, `random_hills` (exempt) | `clip_scalar`, `DataSet.interpolate`, the 21 `Parametric*` surfaces |

**Where pyvista is the strongest reference in the set**, which is what the registration was for:
`compute_implicit_distance` is an exact SDF sharing ordito's sign convention (correlation
1.0000000, max abs difference 1.29e-07, sign agreement 1.000 — vedo's `signed_distance` is a
point-cloud estimator at 0.956 / 0.375 by comparison); `select_interior_points` agrees with
`ray.contains_points` on 1.000 of 2 000 queries; and `decimate` is the **best** of the four
decimation references on sphere deviation, ahead of ordito by 1.14–1.53× (`tests/test_remesh.py`
carries the numbers). Three more from the 2026-08-21 pass: `find_closest_cell` agrees with
`igl.point_mesh_squared_distance` to **4.4e-16** on both distance and point, making it the most
accurate closest-point reference registered; `find_containing_cell` is the **only** working
point-location oracle besides scipy (`igl.in_element` returns batch-size-dependent answers and
aborts on a 200-point Delaunay); and `intersection` returns the unordered segment soup
`mesh_with_mesh` returns, so it pins that curve's *value* where MeshLib's contour linker can only
bound it (36 = 36 segments, bit-identical total length).

**Three caps and one skip come from that pass**, all measured per call: `find_closest_cell` at
10 000 queries is 160 / 376 / 906 ms on `bunny_decimated` / `bunny` / `dragon` (capped at `bunny`);
`intersection` is 134 / 577 ms on the same two (the group's own `bunny` cap covers it); and the
polyline locator collapses on a long single cell — 24.8 ms at 4 096 queries against a 268-segment
loop, **4 963.9 ms** against `rim_long`'s 65 536-segment one and 104 s at 65 536 queries, so
`rim_long` is skipped for `pyvista` in `polyline_point_distance` by name.

**Costs measured on an 81 920-face mesh**, which is where the caps come from: `slice` 7.9 ms,
`curvature('gaussian')` 8.4 ms, `cell_quality` 9.4 ms, `clean` 10.1 ms, `compute_normals` 20.3 ms,
`extract_all_edges` 20.7 ms, `geodesic_distance` 32.5 ms, `smooth(20)` 39.6 ms,
`select_interior_points` (2 000 queries) 124.9 ms, `decimate(0.5)` 159.3 ms,
`voxelize_binary_mask(64³)` **502 ms**. The last three plus `vtkFeatureEdges`, `vtkDijkstra` and
`vtkImplicitPolyDataDistance` are capped at `bunny`; the cheap rows run to `dragon`.

**Five exemptions**, each with its measured disagreement in the `noparity` reason:
`principal_curvature` (D1 — `curvature('maximum')` is exactly `H ± √(H² − K)` from VTK's own two
curvatures, max abs difference 0.0, so it is algebra rather than an estimator), `fit_line` (D2 — it
returns the first *principal* axis where `points.fit_line` is trimesh's σ-weighted major axis,
|dot| 0.802 on an ordinary cloud), `filter_laplacian_integration` and `filter_taubin` (D2 —
different algorithms, see `.claude/CLAUDE.md` §6), and `random_hills` (D5 — VTK draws its own
amplitudes and variances from its own generator, so no seed pairs the two).

**Four filters that look like references and are not**, measured 2026-08-21 so no row is written
against them: `multi_ray_trace` is **trimesh + embree**, not VTK (it imports trimesh and calls
`intersects_location`), so a `pyvista` row on the `intersects_*` groups would be a trimesh row under
another name — VTK's own `ray_trace` is independent and exact (face agreement 1.0000, hit point
2.80e-07) but costs 398.6 µs *per ray*, so it stays a test oracle; `validate_mesh().intersecting_faces`
is an **intra-cell** check and reads 0 on two interpenetrating spheres where ordito flags 92 faces
(`inverted_faces` likewise reads 0 on ten reversed faces, and the degeneracy field that fires is
`zero_size`); `collision` is a two-mesh filter and reports 2 600 hits for a 320-cell mesh against its
own copy, so it cannot see a self-intersection; and `sample()` validates only 476 of 2 562 probes
against `transfer_onto_vertices` (7.20e-08 where valid), with `snap_to_closest_point=True` snapping
to the nearest source *vertex* and making it worse (0.256).

Modules with **no** VTK equivalent: `heat/*` beyond the graph-distance bound, `tangent_space`,
`homology`, `energies`, `parametrization` beyond `delaunay_2d`, `texture`, `linalg`, `array`,
`grouping`, `geodesic_walk`, `convex`, `halfedge` and `visibility` (`silhouette` is absent from all
142 filter names — it is vedo's, not pyvista's). Structurally out of scope: every `pyvista.core`
dataset *type* that is not a surface (`ImageData`, `RectilinearGrid`, `StructuredGrid`,
`UnstructuredGrid`, `MultiBlock`, `Table`), `delaunay_3d` and everything downstream, `streamlines`,
and all of `pyvista.plotting`.

### potpourri3d

| module | potpourri3d reference |
|---|---|
| `test_heat_distance` | `MeshHeatMethodDistanceSolver` (`use_robust=False`, the same discretization as ordito's), and `MeshFastMarchingDistanceSolver` as a different algorithm for the same task |
| `test_intersection` | `marching_triangles` — the only reference for isocontours of an arbitrary vertex field |
| `test_geodesic_walk` | `GeodesicTracer.trace_geodesic_from_vertex` — one ray per call, so its row is linear in the ray count by construction |
| `test_heat_signed` | `MeshSignedHeatSolver.compute_distance` — requires every curve segment inside one face, which is why the source curves are edge paths |
| `test_heat_vector` | `MeshVectorHeatSolver.{extend_scalar,transport_tangent_vectors,compute_log_map}` (`use_intrinsic_delaunay=False`) |
| `test_tangent_space` | `MeshVectorHeatSolver.get_tangent_frames` (`use_intrinsic_delaunay=False`) — an *upper bound*: the frames only come out of the solver's construction, which also factors two sparse systems |
| `test_laplacian` | `cotan_laplacian` (a numpy/scipy build, not C++) and `vertex_areas` (the barycentric lumped mass diagonal) |
| `test_triangles` | `face_areas` |
| `test_edges` | `edges` — in the `edges_unique_manifold` group only, see the hazard table |

`test_laplacian` uses **libigl** (`cotmatrix_intrinsic`) rather than potpourri3d, which exposes
mollification only inside its heat solver. `test_halfedge` and `test_validation`'s subject matter has no
reference anywhere.

Two things shape every potpourri3d row:

- **Its solvers cache their factorizations**, so the solver is constructed **inside** the timed
  callable — that is where geometry-central does the work ordito's per-call assembly does. Timing
  only `compute_*` would compare a back-substitution against a full iterative solve. Where ordito
  has its own reusable precompute the amortized case is measured explicitly instead of argued about:
  `heat_geodesic` carries a `setup=full`/`setup=amortized` parameter layer, and on ordito's side the
  amortized row is a real API path (`heat_operators` passed back through `heat_geodesic`).
- **Its defaults do more work than ordito's.** `MeshHeatMethodDistanceSolver(use_robust=True)` and
  `MeshVectorHeatSolver(use_intrinsic_delaunay=True)` mollify and flip to an intrinsic Delaunay
  triangulation first. Every row here passes `False` so both sides discretize the same triangulation;
  the robust path becomes a parameter layer when ordito grows one.

Measured against it on an RTX 5090: `marching_triangles` is **25x** faster at `sphere_med` and 59x at
`sphere_large`; `heat_geodesic` is 2.8x faster at `saddle` but **1.2x slower** at `saddle_graded`,
where ordito's CG pays for the conditioning and geometry-central's direct solve does not; the
amortized solve is **19x slower** than potpourri3d's back-substitution at `sphere_small`, which is
the clearest statement in the suite of what an iterative solver costs per extra source set.

`heat_signed_distance` is **13x** faster than the reference on `sphere_med` (81 ms against 1 060 ms),
and its own axis answers a design question rather than a competitive one: a 370-segment source curve
costs *less* than a 6-segment one (71 vs 81 ms), because a source spread over the surface converges in
fewer conjugate-gradient iterations. Pinning the level set costs 4.7x an unconstrained solve
(80.6 against 17.3 ms).

The tangent-space and geodesic-walk groups repeat both halves of that story. `trace_rays` is **flat
at 1.15 ms from 1 to 4 096 rays** against potpourri3d's 63 → 77 ms (55x → 67x), because one thread
traces one ray and the reference's API traces one ray per call. `log_map` is 6.8x faster at `saddle`
and only 2.6x at `saddle_graded` — ordito's vector solve pays **2.7x** for the aspect ratio there,
the reference's factorization nothing. `heat_signed_distance_conditioning` shows the same at **3.1x**
(40.3 → 125.8 ms against a flat 525 → 519), which is the worst of the three because that method runs
three solves. `robust_laplacian` runs 9.5-22x faster than
`igl.cotmatrix_intrinsic` on the scan meshes, and `mollify_intrinsic` sits at 0.28 ms on `bunny`,
which is its two host readbacks and essentially nothing else. `intrinsic_delaunay` runs
1.15 / 1.20 / 2.18 ms over the `scale` axis against `igl.intrinsic_delaunay_cotmatrix`'s
7.2 / 58.9 / 260 ms — read with the caveat that igl's call assembles the matrix too, but the
crossover is real: one round of the parallel flip loop finding nothing to do costs about a
millisecond.

| module | open3d reference |
|---|---|
| `test_creation` | `create_box`, `create_sphere` (a UV sphere, so it pairs with `uv_sphere`), `create_cylinder`, `create_cone`, `create_torus`, `create_tetrahedron` / `create_octahedron` / `create_icosahedron` (no dodecahedron) |
| `test_registration` | `TransformationEstimationPointToPoint.compute_transformation`, `registration_icp` (point-to-point, point-to-plane, `TukeyLoss`) |
| `test_reconstruction` | `create_from_point_cloud_ball_pivoting` (`create_from_point_cloud_poisson` was timed here and is **removed** — 73 % of the suite's wall clock) |
| `test_remesh` | `subdivide_midpoint` (`subdivide` only) |
| `test_smoothing` | `filter_smooth_laplacian` (`novol` only), `filter_smooth_taubin`, `filter_sharpen` — all three exempt (D2): the first two re-derive inverse-distance weights every pass, the third multiplies the residual by the vertex degree |
| `test_sample` | `sample_points_poisson_disk` |
| `test_combine` | `cluster_connected_triangles` + `select_by_index` (`split` only) |
| `test_holes` | `open3d.t.geometry.TriangleMesh.fill_holes` |
| `test_repair` | `remove_duplicated_triangles`, `remove_duplicated_vertices`, `remove_unreferenced_vertices` |
| `test_validation` | `is_watertight`, `is_edge_manifold` (same `allow_boundary_edges` switch), `is_vertex_manifold` (connectivity-based: agrees with ordito exactly on edge-manifold input, passes vertices on a non-manifold edge that the fan definition fails) |
| `test_vertices` | `compute_vertex_normals` |
| `test_bounds` | `get_axis_aligned_bounding_box` (`aabb`), `get_minimal_oriented_bounding_box` (hull-based, trimesh's algorithm family — not the PCA `get_oriented_bounding_box`, which minimizes nothing), `AxisAlignedBoundingBox.get_point_indices_within_bounding_box` (`points_in_aabb`), `PointCloud.crop` (`crop_points`) and `TriangleMesh.crop` (`crop_mesh`) — the three query rows, all of them crossovers rather than ratios, and the only reference of the ten that answers the query half at all |
| `test_points` | `PointCloud.estimate_normals` (`KDTreeSearchParamKNN`), `remove_statistical_outlier`, `remove_radius_outlier` (**nondeterministic** — a shared `KDTreeFlann` across an OpenMP loop; three keep sets over eight reps, so the correctness comparison queries that tree serially instead), `remove_duplicated_points`, `farthest_point_down_sample` (its `SelectByIndex` sorts, so only the selected *set* is comparable) |
| `test_metrics` | `PointCloud.compute_point_cloud_distance` (the non-differentiable Chamfer / Hausdorff cases) |
| `test_convex` | `compute_convex_hull` (exact qhull vs the approximate support sweep) |
| `test_voxels` | `VoxelGrid.create_from_triangle_mesh_within_bounds`, `create_from_point_cloud`, `PointCloud.voxel_down_sample`, `check_if_included` — the same four answers, from a `std::unordered_map<Eigen::Vector3i>` on one core |
| `test_neighbors` | `o3d.core.nns.NearestNeighborSearch.knn_search` / `fixed_radius_search` — the batched tensor queries, **not** the legacy `KDTreeFlann` per-query Python loop (62 ms against 10 ms at 20k queries); plus `PointCloud.compute_nearest_neighbor_distance`, which is serial C++ rather than a Python loop |
| `test_proximity` | `o3d.t.geometry.RaycastingScene.compute_signed_distance` (Embree; parity-ray sign, same convention as ordito's `"parity"` mode to 1.8e-7) |
| `test_triangles` | `compute_triangle_normals` (unit normals, unlike MeshLab's raw cross product) |

Modules with **no** open3d equivalent, and why, are documented in each module's docstring:
`test_edges` (no general edge list), `test_boundary` (no loop ordering), `test_grouping` and
`test_reduce` (array primitives), `test_parametrization` (no harmonic/LSCM/ARAP), `test_linalg` (no
iterative solver), `test_laplacian` (no cotangent or mass matrix — the smoothing filters build their
weights inline), `test_curvature` (no curvature estimation at all), `test_intersection` (no plane
section), `test_texture` (stores UVs but has no bake or resample), `test_polyline` (`LineSet` is
unordered segments with no length/resample/simplify), `test_heat_distance` (no geodesic distance),
`test_selection` (no selection morphology), `test_mesh` (no caching container), `test_graph` (no
traversal over an abstract CSR), plus the individual functions noted inline. One function is *tested*
against open3d and deliberately **not** benchmarked: `points.point_finite_mask` is a single `wp.map`
over a three-component `isfinite`, so a group would report the ~11 us launch floor and nothing else,
while `remove_non_finite_points` would be timing its copy of the surviving cloud. `bounds`'
`points_in_aabb_mask` / `points_in_obb_mask` are declined for the same reason from the other side:
each is its group's row *minus* the `flatnonzero` that sizes the index buffer, so a row would
re-time one `wp.map` under a second name. Two rows were
*rejected on measurement* rather than absence and carry the numbers in their module docstrings:
`test_measures`' `get_volume` validates before it integrates (13.8 s of `IsWatertight` for a
microsecond integral — it would re-time the watertightness row under another name), and
`test_sample`'s `sample_points_uniformly` returns a bare cloud with no face index, so asserting it
against the area law would transform the reference.

Modules with no baseline from **any** reference are `test_texture` and `test_halfedge` (plus
`stitch*` in `test_holes` and the two transport groups in `test_tangent_space`); each docstring says
which reference was considered and why it is not apples-to-apples. Those are before/after
self-comparisons.

Four modules were on that list and no longer are, so a reader who remembers it should re-read the
docstrings rather than the list:

- **`test_reduce`** carries `numpy` on all ten of its groups and always did — it was on the list by
  mistake.
- **`test_array`** likewise now carries `numpy` on six, on the argument `test_reduce` had already
  made at length: a device primitive returning a Python value pays a launch and a readback NumPy
  never pays, so the question each row answers is *where the crossover sits*, and the split is by
  **return type** rather than by size.
- **`test_linalg`** carries `pymeshlab` on `min_quad_with_fixed` and now `igl`, which binds
  `min_quad_with_fixed` itself.
- **`test_polyline`** carries `meshlib` and `pyvista` across six of its nine groups. MeshLib's
  `Polyline3` is a complete polyline library and VTK reaches the same operations through a
  single-cell `PolyData`; `polyline_radius` and `polyline_angles` are the two genuine declines and
  say so at the group.

**`test_texture` is the one that is still open on the evidence rather than settled.** Neither
trimesh, libigl nor open3d has a UV-space rasterizer or sampler, which is what the module docstring
says — but **pymeshlab does, in both directions**: `transfer_attributes_to_texture_per_vertex`,
`compute_texmap_from_color` and `generate_sampling_texel` bake, and
`compute_color_from_texture_per_vertex` and `transfer_texture_to_color_per_vertex` sample back. That
is five groups with no oracle against a library that answers all five. The parameter surface is large
and the output is an image rather than an array, so it is scoped as its own piece of work rather than
five rows bolted on; what is *not* true is that no library does it.

Where the reference is not algorithmically identical, the module docstring says so — `test_repair`
(open3d's dedup is orientation-sensitive), `test_sample` (count- vs radius-parametrized),
`test_holes` and `test_smoothing` (different algorithms for the same task), `test_convex`
(approximate vs exact).

### pytorch3d — the only GPU reference, and the only crossover

| module | pytorch3d reference |
|---|---|
| `test_neighbors` | `ops.knn_points` (`k1` / `k7`, both backends), `ops.ball_query` (both backends) |
| `test_metrics` | `loss.chamfer_distance` (the convention `ordito.metrics` documents), `loss.point_mesh_face_distance` |
| `test_registration` | `ops.corresponding_points_alignment`, `ops.iterative_closest_point` |
| `test_laplacian` | `ops.laplacian`, `ops.norm_laplacian`, `ops.cot_laplacian` (which carries `mass_matrix`'s row from the same call) |
| `test_energies` | `loss.mesh_edge_loss`, `loss.mesh_normal_consistency`, `loss.mesh_laplacian_smoothing` — the three regularizers `ordito.energies` grew for it |
| `test_points` | `ops.sample_farthest_points`, `ops.estimate_pointcloud_normals` |
| `test_voxels` | `ops.add_points_features_to_volume_densities_features`, and `torch.nn.functional.grid_sample` for the gather half |
| `test_triangles` | `ops.mesh_face_areas_normals` — the only reference in that group that answers *both* halves in one call, like ordito |
| `test_vertices` | `Meshes.verts_normals_packed` |
| `test_edges` | `Meshes.edges_packed`, `Meshes.faces_packed_to_edges_packed` |
| `test_remesh` | `ops.SubdivideMeshes` |
| `test_levelset` | `ops.marching_cubes.marching_cubes` — the fifth implementation of that case table, and the only one needing no lattice-convention fix |
| `test_creation` | `utils.ico_sphere`, `utils.torus` |
| `test_combine` | `structures.join_meshes_as_scene` |
| `test_sample` | `ops.sample_points_from_meshes` |
| `test_reduce` | `ops.utils.wmean` |
| `test_array` | `ops.packed_to_padded` / `padded_to_packed` |

**It is the only reference with CUDA kernels of its own, and it is registered for those alone** —
a single `pytorch3d-cuda` row, the suite's only GPU-against-GPU comparison, and no host row at
all (see below). That row is gated on a probe rather than on
`torch.cuda.is_available()`, which is the wrong question: a build that compiled the CPU extension
only (the normal state of a runner with no `CUDA_HOME`) reports CUDA available and then raises
`RuntimeError: Not compiled with GPU support.` on every kernel, and a wheel whose arch list stops
short of the device reports available and fails with *"no kernel image is available"*.
`_pytorch3d_cuda_available()` launches a two-point `knn_points` and believes the result.

**Every `-cuda` row synchronizes torch's stream, not Warp's.** `wp.synchronize_device`
synchronizes Warp's and says nothing about torch's, so a row synchronized the Warp way would time
the launch rather than the kernel — a number that is silently far too good. `BenchCase.run`
branches on `kind == "pytorch3d"`.

**Its `_C` carries no spatial structure on either device**, so the GPU ratio is a *crossover* and
not a bar. Measured on this box, min of 12 interleaved reps on a uniform cloud, each side building
whatever index it builds inside the timed call:

| (p3d / ordito-cuda) | 20 000 points | 200 000 points |
|---|---|---|
| `knn_points(K=8)` | 2.31 / 3.29 ms **0.70x** | 74.65 / 0.87 ms **85.5x** |
| `chamfer_distance` | 3.39 / 5.36 ms **0.63x** | 108.12 / 1.19 ms **90.5x** |
| `ball_query(r=0.15)` | 4.97 / 0.31 ms 16.1x | 179.52 / 2.12 ms 84.6x |
| `sample_farthest_points(K=1000)` | 14.35 / 5.43 ms 2.64x | 104.93 / 35.99 ms 2.92x |

pytorch3d **wins** the k-NN and chamfer rows at 20 000 points: brute force with perfect coalescing
beats a BVH descent while the whole problem still fits the device's bandwidth. So those groups need
the point count as an *axis*; a one-size row reports whichever side of the crossover it happened to
land on.

**Half of that swing is ordito's, and the row must not be read as a statement about brute force.**
pytorch3d is a clean quadratic over the sweep — 0.63, 2.26, 5.59, 20.18, 73.82 ms at 5 k / 20 k /
50 k / 100 k / 200 k — while `query_nearest` is **non-monotonic**: 0.837, 3.262, 7.567, **0.472**,
0.801 ms, a **16x drop** between 50 k and 100 k on the same box and the same extent. So the honest
reading of the 20 000-point row is "ordito is 7x off its own 100 000-point cost here", not
"pytorch3d is faster".

Two things about that claim were wrong for a long time and are corrected here, both by re-running
the sweep **interleaved across size and backend in one pre-warmed process** (§13's rule — the
original was a sequential sweep, which is exactly the shape that manufactures this artifact, so it
had to be re-measured before it could be trusted):

- **It is a `k >= 8` effect and does not exist at `k = 1`.** At `k = 1` the same sweep is flatly
  monotonic — 0.229, 0.236, 0.249, 0.269, 0.346 ms — and per query it *falls* 45.8 → 1.7 ns, which
  is a warm cache filling up rather than anything pathological. The published figures were a `k = 8`
  sweep (they reproduce it to ~3 %: 0.82/0.837, 3.25/3.262, 7.38/7.567, 0.46/0.472, 0.75/0.801) and
  had been read onto the `k1` rows.
- **It is hash-grid-specific, so it is not "a stage the two structures share".** The `bvh` backend
  is monotonic over the identical sweep at `k = 8` — 0.695, 0.774, 0.854, 1.368, 2.023 ms — and
  therefore **beats the hash grid by 1.2x / 4.2x / 8.9x** at 5 k / 20 k / 50 k before losing 2.9x /
  2.5x at 100 k / 200 k. The earlier "identical to three digits between `bvh` and `hashgrid`
  (0.82/0.82, 3.25/3.25, 7.38/7.18)" does not reproduce at all.

What survives is the mechanism and it is now located rather than shared: the hash grid's cell width
is `initial_radius`, and where a query's true `k`-th distance runs past
`_knn_widest_grid_radius(cell, n)` that row abandons the walk for an **exact linear scan** of the
whole cloud. 50 000² tests in 7.567 ms is 3.3e11 tests/s, which is that scan and not a search. The
uniform-density estimate has no distribution behind it, so the *tail* of the true `k`-th distance is
what trips the cutover, and how much of the cloud is in that tail is what moves between 50 k and
100 k. `backend="bvh"` has no cell width and is the available workaround at moderate `k`;
`query_nearest`'s own note documents the choice.

**There is deliberately no `-cpu` row.** pytorch3d earns its seat on the CUDA kernels, and its
host path is a reference implementation rather than a tuned one: threaded
(`torch.get_num_threads()` is 24 here) but with the same absence of a spatial index, which makes
it Θ(N²) — measured rather than assumed: `knn_points` 299.8 / 1 189.5 / 4 562.8 ms and
`chamfer_distance` 557.8 / 2 319.1 / 9 077.1 ms at 10 k / 20 k / 40 k self-queries, 3.84-4.16x per
doubling against the 4x a quadratic predicts. Extrapolating puts `bunny` at ~3.5 s per `knn` round
and `dragon` at **~9 minutes per round**. Round 9 ran two such rows uncapped and lost 1 h 43 min of
wall clock and two whole modules' JSON to them, because `--benchmark-json` is written at session
end — and the eleven CPU baselines already say everything a quadratic host row would. So pytorch3d
contributes its `-cuda` row or no row: where the extension is missing the reference is simply
absent, with no host fallback.

**Two structural facts shape almost every row.** `Meshes` and `Pointclouds` are **immutable** and
every `ops.*` / `loss.*` entry point is pure, so one container serves many calls and there is no
freshness rule at all — the opposite end of the scale from a `ml.MeshSet`. But they **memoize** their
derived quantities (`verts_normals_packed`, `edges_packed`, `faces_packed_to_edges_packed`,
`laplacian_packed`, `faces_areas_packed`), so a row naming one of those must build the container
*inside* the timed callable or it reports a cache hit — measured 0.287 ms of build against a 3.693 ms
normals derivation on a 40 962-vertex mesh, so the build is 8 % of that round on the host and 34 % on
CUDA, and the row says which. Rows that name a pure `ops` function warm the accessors outside
instead.

**Two hard limits, both of which raise rather than degrade.**
`ops.sample_points_from_meshes` draws its face index with `torch.multinomial`, whose category limit
is 2²⁴, so `lucy`'s 28 055 742 faces raise `RuntimeError: number of categories cannot exceed 2^24`
and are skipped (`happy_buddha`'s 1 087 716 are fine). And `ops.packed_to_padded` /
`padded_to_packed` bounds-check nothing: a mismatched `max_size` or `total_size` corrupts the heap
and the process dies in `malloc` much later, in unrelated code, so the sizes are read off the
buffers.

**No exemptions, and four groups tested but deliberately not timed** — which is unusual for a
reference this broad and is worth reading as a statement about the *inputs* rather than the answers.
Each of the four carries a `benchmarked=False` parity claim with its measured reason:

- **`filter_taubin`**: `ops.taubin_smoothing` rebuilds its inverse-distance operator from the
  current positions every half-pass, and `filter_taubin(recompute=True)` matches it to **2.4e-07**
  — but that path measures **~23x** the group's default row (0.49 → 11.06 ms at 10 passes), so a
  row would race ten sparse assemblies against one under a single name. The default *does* disagree
  (1.04e-02, the size of the displacement), which is why the pair is not a D2 exemption but a
  configuration mismatch.
- **`closest_point_on_edges`**: `loss.point_mesh_edge_distance` reads `Meshes.edges_packed`, the
  mesh's whole undirected edge set, and takes no edge subset — so it cannot be handed the crease set
  both rows of that group share. A row would price 19.6x more edges on `bunny_decimated` and 76.9x
  on `bunny`.
- **`to_boxes`**: `ops.cubify` culls and compacts, where that group is pinned to
  `cull_internal=False` so trimesh's `multibox` and VTK's glyph filter are like-for-like unwelded
  rows. The comparable pair was measured anyway and is not interesting: 2.238 ms against
  `to_boxes(cull_internal=True)`'s 1.850 on 63 568 voxels.
- **`load_mesh`**: `ordito.io` has no benchmark group at all and deliberately so — it is a meshio
  round trip, so a row would time meshio's PLY parser against pytorch3d's, neither of which is
  ordito code.
