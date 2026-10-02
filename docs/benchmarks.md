# Benchmarks

Every chart on this page is generated, not hand-drawn: `benchmarks/plot_comparison.py` reads
`pytest-benchmark`'s own `--benchmark-json` output (via `benchmarks/aggregate.py`, the same module
that produces the plain-text loss tables used during development) and renders one pair of PNGs per
comparison, each competing library named by its own logo.

**How to read one of these:** bars are sorted fastest-to-slowest, top to bottom; ordito's own bar
is the accent color, every reference library is the same muted gray (the logo and label carry
which library it is — color is reserved for "is this ordito"); the axis switches to a log scale
whenever the fastest and slowest library in that cell differ by more than 15x, stated in the
chart's own subtitle so it is never ambiguous; and every bar carries its exact value as a direct
label, so nothing depends on reading the axis precisely.

**These are rendered from a full run, with every reference library enabled.** The development
sweep skips any reference already measured losing by more than 2x and ranking third-or-worse for a
given cell — it recovers about 43 of the suite's 50 minutes of timed regions and costs nothing a
loss table cares about, since a library already 100x behind does not become more informative on
the next mesh size. But it is exactly the wrong input for these charts, because the rows it drops
are the ones ordito beats most widely: in the current run the default sweep has **15** cells with
three or more reference libraries, while the same suite with `--bench-all-libs` has **206**. The
charts below come from the full run, so a chart with five bars means five libraries really do
compute that quantity.

**Nothing here is cherry-picked in the sense of hiding a loss.** Three of the ten cells are a tie
or a defeat, and they were chosen to be informative rather than token — the `k=1` nearest-neighbour
cell sits directly below the `k=7` one it loses to, which is the honest way to show where the
crossover is.

**Reproduce any of these yourself:**

```bash
git clone https://github.com/lucagrementieri/ordito
cd ordito
uv sync --group bench

# One JSON per module into a directory; --bench-all-libs is what makes the charts multi-library.
# Budget about an hour on an RTX 5090 -- roughly 1.8x the default sweep, which skips the
# slowest reference rows.
mkdir -p /tmp/bench-json
for f in benchmarks/test_*.py; do
  uv run pytest "$f" --device=cuda --bench-all-libs \
    --benchmark-json="/tmp/bench-json/$(basename "$f" .py).json" -q
done

uv run python benchmarks/plot_comparison.py /tmp/bench-json --out /tmp/charts --hero
```

`--hero` renders the ten curated cells on this page; drop it to render every comparable cell, or
pass `--group <name>` for one group across every mesh.

See [`benchmarks/README.md`](https://github.com/lucagrementieri/ordito/tree/main/benchmarks) for
the full methodology — which axis each group sweeps, which libraries are compared for which
operation, and why a handful of comparisons are excluded.

---

## Proximity: `signed_distance_on_mesh`

Signed distance from a point set to a triangle mesh, against five independent implementations —
the widest field of any cell on this page, and the single best illustration of what the library
is for.

![signed_distance_on_mesh benchmark](assets/benchmarks/signed_distance_on_mesh-bunny-sign_mode-parity-light.png#only-light)
![signed_distance_on_mesh benchmark](assets/benchmarks/signed_distance_on_mesh-bunny-sign_mode-parity-dark.png#only-dark)

??? note "Exact values"
    | Library | Median |
    |---|---|
    | ordito (CUDA) | 2.68 ms |
    | MeshLib | 13.73 ms |
    | Open3D | 30.98 ms |
    | libigl | 103.81 ms |
    | PyVista | 276.93 ms |
    | MeshLab | 658.63 ms |

This is `sign_mode="parity"`, the default. ordito also offers `sign_mode="winding"`, which is
exact on meshes with holes where ray parity misclassifies; there the only comparable reference is
libigl, and ordito is 44x ahead of it.

## Inside / outside: `winding_number`

![winding_number benchmark](assets/benchmarks/winding_number-bunny_decimated-n_queries-10000-light.png#only-light)
![winding_number benchmark](assets/benchmarks/winding_number-bunny_decimated-n_queries-10000-dark.png#only-dark)

??? note "Exact values"
    | Library | Median |
    |---|---|
    | ordito (CUDA) | 1.24 ms |
    | MeshLib | 54.54 ms |
    | libigl | 60.12 ms |
    | PyVista | 150.50 ms |

## Spatial queries: `query_ball`

A radius (ball) query against a prebuilt BVH — one of the primitives most of the rest of the
library is built on. SciPy's `cKDTree` and PyTorch3D's `ball_query` are both here, so this spans a
CPU tree, a GPU brute-force kernel and two spatial indices.

![query_ball_bvh benchmark](assets/benchmarks/query_ball_bvh-bunny-radius_scale-4.0-light.png#only-light)
![query_ball_bvh benchmark](assets/benchmarks/query_ball_bvh-bunny-radius_scale-4.0-dark.png#only-dark)

??? note "Exact values"
    | Library | Median |
    |---|---|
    | ordito (CUDA) | 769.9 µs |
    | Open3D | 8.11 ms |
    | PyTorch3D (CUDA) | 134.22 ms |
    | SciPy | 134.31 ms |

## Spatial queries: `query_nearest`, k = 7

The same cloud and the same 20 000 queries, asking for seven neighbours instead of a ball. Five
implementations, including the one other GPU library in the suite.

![query_nearest_bvh_k7 benchmark](assets/benchmarks/query_nearest_bvh_k7-bunny-light.png#only-light)
![query_nearest_bvh_k7 benchmark](assets/benchmarks/query_nearest_bvh_k7-bunny-dark.png#only-dark)

??? note "Exact values"
    | Library | Median |
    |---|---|
    | ordito (CUDA) | 595.7 µs |
    | PyTorch3D (CUDA) | 3.79 ms |
    | Open3D | 12.96 ms |
    | SciPy | 27.10 ms |
    | libigl | 240.78 ms |

## The same query at k = 1, where ordito does *not* win

Drop `k` from 7 to 1 and MeshLib's projector edges ahead. This is the most useful chart on the
page: the two cells differ only in `k`, so the comparison isolates one variable, and it shows that
ordito's advantage here comes from amortizing a traversal across many neighbours rather than from
the traversal being faster per se. At `k=1` there is almost nothing to amortize and a good serial
C++ projector is the better tool.

![query_nearest_bvh_k1 benchmark](assets/benchmarks/query_nearest_bvh_k1-bunny-light.png#only-light)
![query_nearest_bvh_k1 benchmark](assets/benchmarks/query_nearest_bvh_k1-bunny-dark.png#only-dark)

??? note "Exact values"
    | Library | Median |
    |---|---|
    | MeshLib | 377.4 µs |
    | ordito (CUDA) | 419.5 µs |
    | PyTorch3D (CUDA) | 2.80 ms |
    | Open3D | 11.70 ms |
    | SciPy | 17.63 ms |
    | libigl | 204.36 ms |

At 419 µs against 377 µs this is a 1.1x difference on a sub-millisecond call, which is close
enough to the fixed per-call overhead of a GPU launch that it should be read as a tie rather than
a ranking.

## Point-cloud reconstruction: `ball_pivoting`

![ball_pivoting benchmark](assets/benchmarks/ball_pivoting-bunny-light.png#only-light)
![ball_pivoting benchmark](assets/benchmarks/ball_pivoting-bunny-dark.png#only-dark)

??? note "Exact values"
    | Library | Median |
    |---|---|
    | ordito (CUDA) | 45.77 ms |
    | MeshLab | 136.25 ms |
    | Open3D | 458.55 ms |

## Point-cloud reconstruction: `screened_poisson`

![screened_poisson benchmark](assets/benchmarks/screened_poisson-bunny_decimated-depth-7-method-dense-light.png#only-light)
![screened_poisson benchmark](assets/benchmarks/screened_poisson-bunny_decimated-depth-7-method-dense-dark.png#only-dark)

??? note "Exact values"
    | Library | Median |
    |---|---|
    | ordito (CUDA) | 20.42 ms |
    | MeshLib | 120.45 ms |

Only two bars, and the reason is worth stating. Open3D and MeshLab both wrap Kazhdan's CPU
solver, and their rows were deliberately removed from the benchmark: between them they accounted
for **73 % of the entire suite's runtime** while measuring a reference ordito had already beaten
by 15-25x, and at `dragon` scale they ran 93 minutes without completing a single round. The
correctness comparison against them is still made, in `tests/test_reconstruction.py`, at a size a
test can afford. MeshLib is kept here because it is a genuinely *different* implicit
reconstructor and is cheap enough to time.

## Measures: `oriented_bounding_box`

A minimum-volume oriented bounding box over the largest scan mesh in the suite. Worth including
because the four references disagree with each other by three orders of magnitude, which is a
useful reminder that "the reference implementation" is rarely one thing.

![oriented_bounding_box benchmark](assets/benchmarks/oriented_bounding_box-dragon-light.png#only-light)
![oriented_bounding_box benchmark](assets/benchmarks/oriented_bounding_box-dragon-dark.png#only-dark)

??? note "Exact values"
    | Library | Median |
    |---|---|
    | ordito (CUDA) | 2.41 ms |
    | PyVista | 40.16 ms |
    | Open3D | 113.89 ms |
    | trimesh | 161.41 ms |
    | libigl | 2.590 s |

All four references are timed on a minimizing entry point, not on a PCA box — Open3D's row is
`get_minimal_oriented_bounding_box` rather than `get_oriented_bounding_box`, whose PCA-of-hull
result minimizes nothing and would not be the same question. So the spread here is genuinely
speed, not a difference in what is being computed.

## Sampling: `blue_noise`

Poisson-disk / blue-noise surface sampling, against four references.

![blue_noise benchmark](assets/benchmarks/blue_noise-bunny-radius_scale-0.5-light.png#only-light)
![blue_noise benchmark](assets/benchmarks/blue_noise-bunny-radius_scale-0.5-dark.png#only-dark)

??? note "Exact values"
    | Library | Median |
    |---|---|
    | ordito (CUDA) | 4.08 ms |
    | MeshLib | 41.92 ms |
    | MeshLab | 271.63 ms |
    | Open3D | 1.048 s |
    | libigl | 1.069 s |

## Editing: `quadric_decimate`, an honest loss

![quadric_decimate benchmark](assets/benchmarks/quadric_decimate-saddle_graded-target_ratio-0.1-light.png#only-light)
![quadric_decimate benchmark](assets/benchmarks/quadric_decimate-saddle_graded-target_ratio-0.1-dark.png#only-dark)

??? note "Exact values"
    | Library | Median |
    |---|---|
    | MeshLib | 31.36 ms |
    | ordito (CUDA) | 46.06 ms |
    | PyVista | 55.12 ms |
    | libigl | 75.98 ms |
    | Open3D | 84.58 ms |
    | MeshLab | 506.60 ms |

MeshLib's serial C++ decimator is faster than ordito's on this particular mesh — a small, badly
graded saddle, which is deliberately the worst-conditioned input in the suite. The crossover is
real and this chart doesn't hide which side of it a small input lands on: at the gentler
`target_ratio=0.5` on this same mesh the two are level (16.74 ms against MeshLib's 17.14), and
ordito's GPU decimator pulls ahead at larger mesh sizes, where there is enough parallelism to
fill the device.

---

*Generated on an NVIDIA GeForce RTX 5090, warp-lang 1.17.0, from a full `--bench-all-libs`
benchmark run at commit `21ef41d`. Numbers on this page are refreshed at release time, not
continuously — see [Performance](performance.md) for how to read a speed claim, and the reproduce
command above to check any of these yourself against current code.*
