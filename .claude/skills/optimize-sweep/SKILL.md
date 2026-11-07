---
name: optimize-sweep
description: Run a measured optimization sweep over ordito, or over one submodule of it. Collapse multi-launch functions into fewer kernels, move host work to the device, remove redundant copies, transfers, readbacks and re-validation, fuse reductions into their producers, tighten sorting, grouping and packing, extract duplicated kernel code as shared @wp.func helpers, generalize the tricks found in one function to every other function where they pay, and expose caller-supplied hints that skip work. The scope is the whole repository, one submodule, named functions, the last N commits or the last N days. Covers host overhead (launch count and cost, allocations, readbacks, graph replay, CPU one-launch loops), device work (band-proportional work, structure reuse, contention, precision filters), memory (flag-and-recompute compaction, chunking, early release, buffer rotation, bitsets, narrow dtypes, sparse storage) and algorithm replacement. The benchmarks census is the counter; an interleaved A/B on an idle GPU is the clock. Use whenever the user asks for an optimization sweep, a speed-up pass, a kernel fusion pass, a census reduction, to "reduce launches/allocations/readbacks/syncs", or a review of recent kernels for speed.
---

# Optimization sweep

A sweep delivers three things:

1. **Staged commits on `main`.** Each one holds one idea, verified (byte-identical
   where the old output was deterministic) and priced. A change that moves host traffic
   re-records both census budgets in the same commit.
2. **Decline comments at the sites.** Every idea tried and lost gets one, with its number (§9).
3. **A closing report.** What landed with its ratio, every decision taken on the user's behalf
   with its reason, and what was left. Leftovers go into §16 as open items, never only in chat.

`.claude/CLAUDE.md` is the rulebook. A sweep leans on §2.2 (the deterministic setting), §2.4
(fusion means extraction), §3.10 (readbacks), §9 (measure first), §13 (cost model), §14 (kernel
shape verdicts, including the refuted list) and §15 (measurement traps). Before proposing anything
for a component in scope, read its section of `reference/component-status.md`.

---

## 0. Scope and preflight

- **Fix the scope first and state it in one line.** The scope is one of:
  - **a submodule:** `ordito/<module>.py` with its `ordito/kernels/<module>.py` (and any
    `kernels/algorithms/` package it owns). Given as `heat`, `ordito.heat` or a path. Only its
    public functions and their private helpers are candidates, and only its benchmark file is
    counted and timed. Shared libraries it calls (`kernels/array.py`, `reduce.py`, `scatter.py`,
    `predicates.py`, `triangles.py`, `_launch.py`) may change only when a change in scope needs
    them. Anything else noticed outside the scope goes into the report as a lead, not into a
    commit;
  - **named functions:** the same rules, narrowed to those functions;
  - **recent work:** the functions touched by the last N commits
    (`git log -N --name-only -- ordito/`), or by the last N days (`--since`);
  - **the whole repository:** cut into module batches, with a report after each batch. The user
    asks "why is it taking so long?" when nothing has been reported for an hour.

  With no scope given, ask for one. Do not default to the whole repository.
- **The GPU is shared until proven idle.** Run `nvidia-smi` and look for other processes before
  any timing, and again before trusting a result. If the GPU is busy, decide on census counts
  (they are deterministic) and queue the clock A/B. Never report a ratio measured beside another
  job. Another Claude session may be benchmarking. Do not spawn parallel agents that time on the
  same GPU.
- **Baseline worktree:** `git worktree add -q --detach $SCRATCH/base HEAD` (§15; never `stash`).
- **Baseline census for the scope:**
  ```bash
  uv run python -m pytest benchmarks/test_<module>.py -p benchmarks.census -k ordito-cuda -q \
      --census-budget /tmp/base.json --census-update --census-report /tmp/base.jsonl
  ```
  `--census-report` lists each row's launch sequence (kernel, dim, issuing line). That list is
  where the candidates come from: read it instead of grepping for launches. For a submodule
  scope, add `-k <function>` to narrow it further.

## 1. Find the candidates

**First decide where the time goes.** At the benchmark sizes most ordito calls are host-bound: a
launch costs ~12 µs of host time whatever its `dim` (§13.1), so a call issuing 40 launches spends
about half a millisecond before the GPU does any real work. Compare the census counts multiplied
by the §13.1 prices with the device time (`wp.timing_begin`; see §15 for what it misses).

- **Host-bound:** the time follows launches, allocations and readbacks. Work through A first.
- **Device-bound:** the time follows the mesh size. Work through B and C.
- **Memory-bound or out of memory on a large mesh:** the peak sets the limit. Work through C.
- **Check both ends.** A call can be host-bound on `sphere_small` and device-bound on `dragon`.

Work from the largest scale to the smallest: try a better algorithm (D) before fusing its kernels
(A), and do not micro-tune kernels that D would delete. Each technique below names a site where it
already landed, so read that commit before writing your own version.

### A. Host overhead: fewer and cheaper calls into Warp

1. **Collapse a function into one launch.** A function that launches several kernels, or calls
   ordito wrappers that each launch, often runs as one kernel. Examples: `creation`'s "one launch
   per shape", `subdivide` writing vertices and faces in one launch, `transform_mesh`.
2. **Fuse producer and consumer.** Fuse consecutive same-`dim` launches where the consumer reads
   the producer at its own index; every pair measured so far was faster (§14). Also:
   - fold a `wp.map` into the kernel that reads its output (`slice_mesh_with_plane` forms its
     plane dots inside the clip kernels);
   - form a quantity where it is read, instead of in a table built for it (`face_defective_mask`
     forms its normals in the decision pass);
   - let one launch serve both directions of a symmetric quantity (symmetric Chamfer and
     Hausdorff);
   - clear carried state in the kernel that reads it, not in a fill before it (blue-noise per-cell
     summaries, atomic slots that are never re-cleared).
3. **Finish in the last block.** Fold a reduction into the kernel that produces its terms (§2.2
   `store_block_*`). A small finalizer then runs in the block that draws the last ticket instead
   of in a launch of its own: `centroid` divides in the block commit, `fit_plane` runs its SVD in
   the last block, an ICP round closes in its accumulation's last block, and `quadric_decimate`'s
   scans finish in their last block.
4. **Make each launch cheaper.**
   - Launch through `_launch.launch`.
   - Bundle a looped launch with a dozen or more arguments into a `@wp.struct` (§2.8).
   - Hoist a `wp.map` kernel out of a loop with `return_kernel=True` (§3.5).
   - Prefer `_launch.view` to a slice.
   - Unwrap Warp-typed values at Python scope with `int()` / `float()` (§13.1: 13-370x).
   - Launch a generic kernel through its overload table (§2.5).
5. **Never recompile.** Declare overloads and `wp.map` signatures (§2.5, §3.5), cast through
   concrete kernels and never `array_cast` (§3.6), and give a kernel launched at its own width a
   module of its own (§12.4). A sudden slowdown is a rebuild until proven otherwise (§15).
6. **Allocate less.**
   - Carve same-lifetime scratch out of one allocation per dtype (the "carve" commits across
     repair, holes and smoothing; radix sort keys and payload from one `sort_buffers` allocation).
   - Write counts straight into the offsets' buffer.
   - Write a CSR's empty bounds instead of zero-filling the offsets.
   - Allocate a pass's buffers once, outside the loop (`filter_two_step`).
   - Drop a `fill_` or `zero_` where every element is written anyway (§3.3).
   - Remove `wp.clone` / `wp.copy` of a buffer nobody mutates, and prefix slices that get cloned
     (§3.4).
7. **Read back less often.**
   - Each readback either carries its reason comment (§3.10) or goes.
   - Merge small reads into one buffer read once (`_common.read_values`; `screened_poisson`
     "reads its small values together").
   - Read totals off a scan's tail.
   - Take a size the host already knows from the host (`concatenate` reads its offsets off the
     host).
   - In an idempotent loop, read the stop condition every few rounds (the flip loop replays
     several rounds a read; `fast_marching_distance` replays two round pairs between reads).
   - Drive a fixed-width loop from counts kept on the device.
   - Skip a readback that only infers a bound the caller knows (step 5 below).
8. **Replay what repeats.** Record a launch sequence that recurs identically and replay it as a
   CUDA graph. Examples: `fill_min_weight`'s span sweep, the octree Gauss-Seidel wave group,
   `quadric_decimate`'s first-pass round loop. Keep a loop counter on the device so the rounds
   become identical. Read §14's capture verdict first: capture loses on once-through and short
   loops, and on batches replayed past a fixpoint.
9. **On the CPU device, run the round loop as one launch (§14).** CG, Chebyshev, Cholesky panels,
   ear clipping, the fill and stitch DPs, the fan flood, the Poisson V-cycle and the convex hull
   all took this form. Independent ranges of host-sequential work run on host threads
   (`geodesic_walk`'s edge-flip shortening, the contested rows' builder).
10. **Remove re-validation by inner calls.** A public wrapper calling another public wrapper
    repeats `require_same_device`, `ensure_ndim` and the dtype checks: call the private core
    (`join_closest_components` skips its own self-validation).
11. **Move host work to the device.** This covers NumPy over a `.numpy()` readback, Python loops
    over elements, and host-side sizing (the oriented box scored and ranked on the device, the
    `holes` host work, the parametric lattice). Small host metadata math stays on the host
    (§3.8).

### B. Device work: touch less, contend less

12. **Do work in proportion to the answer, not to the domain.**
    - Run only over the band, front or active set: the octree extraction runs over the surface
      band, `marching_cubes` counts only the nodes the surface crosses, `fast_marching_distance`
      runs from front lists, and the octree divergence skips windows with no normal data.
    - Launch only for the items that need the pass (the fill DP's span launches cover only loops
      longer than the span).
    - Rank only over the listed nodes (a long successor table).
13. **Reuse derived structures across stages and iterations.**
    - Every edge classifier shares one halfedge-mates pairing.
    - `isotropic_remesh` carries its edge grouping across iterations, and the flip loop keeps
      its adjacency across rounds.
    - Heat and Poisson share one Cholesky analysis per operator, and the vector heat system reuses
      the scalar bundle's cotangents, rings and twins.
    - A repeated assembly builds its pattern once (`mesh_operator_pattern`).
    - Take a structure the caller already holds as a keyword (`halfedge_mates(vertex_faces=)`).
14. **Do not store what you can stream or recompute.** `screened_poisson` sums its iso-value
    samples without storing them; mesh-to-mesh metrics need no closest-point buffer. Recomputing
    a neighbour's value from positions instead of reading a table (gather-form fusion) wins on
    small and grid-ordered meshes but loses 1.7-2.1x on scrambled meshes of about a million
    faces: price it there (§14).
15. **Sort, group and pack.**
    - Write keys, not triplets (`csr_key`, §3.7); the Hessian energies build their pattern and
      gather entries with no triplet stored.
    - Narrow the sort to its radix (`end_bit`, a 32-bit dissection sort).
    - Emit at each sorted run instead of building a unique table (`unique_rows` writes each
      class's first row in the run emit).
    - Classify with one scan of one-hot `vec{K}i` (`split_faces_along_field`).
    - Fold a compaction into its emit (§14).
    - Order for locality (Hilbert order for the winding-number hierarchy, Morton trees).
16. **Contend less.**
    - A single-address atomic serializes the launch: partition across lanes, then make one
      guarded commit per block (§13.2).
    - Append through a warp-aggregated `append_slot`, and skip a max-bid already beaten.
    - Commit once a warp (`polyline_simplify`'s span maxima).
    - Give a crowded group a block (voxel mean and sum pooling).
    - Pre-sum across a warp that shares a group (`warp_lanes_sum`).
    - Split long runs (§14 fixed-order group sums).
17. **Filter in `float32`, decide in `float64`.** `float64` runs at 1/64 rate (§13.2). Put a
    `float32` test with a rigorous bound ahead of the exact one (`orient3d_filtered_f32`;
    `fast_marching_distance` filters its unfold offers in `float32`).
18. **Fit the launch shape to the work (§14).**
    - Use register blocking for `(item, slice)` reductions.
    - Use one block per item only when the outer dimension starves the device.
    - Give narrow rounds one block (the fan flood's narrow rounds).
    - Gate a tree walk's width on the query count.
    - Issue the loads of a latency-bound row branch-free.

### C. Memory: fewer bytes written, held and read

Every byte written to a table and read back costs bandwidth, and a large transient forces the
mempool to grow. Memory and time therefore usually move together: the plane slices cut their
transient 749 -> 107 MiB and their time 3.5 -> 0.8 ms on lucy, and the Hessian energies cut their
peak 6.3 GB -> 0.7 GB and their time by half on dragon. **Find the stage that sets the peak
first**; a saving anywhere else does not move the peak.

M1. **Do not materialize a per-domain table for a sparse answer.** Write a flag per item, scan it,
    and let the compaction recompute each kept item's payload from the shared per-item
    `@wp.func` (`mesh_with_plane` and `marching_triangles` stored a segment pair for every face;
    now they store a flag). Append kept cells instead of masking the whole box (every voxel
    build: `fill_cavities` 2.67 -> 1.39 ms, `dilate` 13.9 -> 2.4 ms).
M2. **Build the pattern first, then gather.** Do not store triplets that a sort will merge (§3.7;
    the Hessian energies). Radix sort buffers are double-width and often set the call's peak:
    narrow the keys (`end_bit`), or drop a sort that only groups (`repair`'s `method="meshlab"`
    lets its pairing sort go: lucy peak 3.6 -> 2.6 GB).
M3. **Stream in chunks.** A table that grows as items × rays or items × neighbours is processed a
    chunk at a time, sized so that a small input is one chunk (`shape_diameter`: lucy peak
    13.8 GB -> 1.3 GB, same time; geodesic balls fitted chunk by chunk).
M4. **Release before the peak.** Free a buffer whose last reader runs before the peak stage: drop
    the Python reference, and carve it apart from the long-lived scratch. Examples:
    `smooth_region`'s gathered terms (dragon 322 -> 240 MiB), `triangulate_point_cloud`'s
    neighbour table (506 -> 394 MiB), the octree backend's solve buffers before the extraction.
M5. **Rotate and share buffers instead of holding more.**
    - Ping-pong iterates that are read only at their own index (Chebyshev keeps two iterates,
      not three; the polynomial CG rotates its `w` and `u`; the squared V-cycle keeps three
      vectors, not five).
    - Borrow a probe's vectors for the escalated solve.
    - Share solve buffers between methods (dense and adaptive Poisson).
    - Store a slot in another table whose reader looks at it only at its own index (the
      winding band's slots ride in its ambiguity marks).
M6. **Size to the content, not to a capacity.** Cut a matrix to its stored entries
    (`multigrid_preconditioner` dragon 227 -> 68 MiB, lucy 6.8 -> 2.1 GB). Size front lists
    and hashes for what a run holds (`ball_pivoting`). Never size off `nnz` (§3.7).
M7. **Derive what you can instead of storing it.**
    - Store an undirected key, form the endpoints from it, and carry the direction in a sign
      (`ball_pivoting`'s edge table: dragon 771 -> 643 MiB).
    - Form a lattice node's position in its thread instead of storing a `vec3` per node (16 -> 4
      bytes a node).
    - Index by the item that reads the value (the octree's vertex partners are indexed by
      vertex).
M8. **Use bits and narrow types.**
    - Bitsets with a popcount rank in place of `int32` flags plus a scan: octree created edges
      (222 MiB at depth 8), `split_edges`, halfedge flags, rank-packed brick indices.
    - Verdict words a 32nd of the table (intersection's broad phase).
    - `int8` corner sources, and symmetric storage (10 doubles for a quadric instead of 16).
    - `float32` where the precision allows (large V-cycle levels).

    Shared helpers live in `kernels/array.py`. Extend them rather than writing a local copy.
M9. **Store sparse data sparsely.** Store lattices and cell tables by brick, and pack per-sample
    tables by occupied rank (`screened_poisson(method="octree")`: bunny depth 9 8.3 -> 5.2 GB
    with the fused residual). Pack variable-length rows before assembling them, instead of
    padding an `(n, k, 3)` table (`triangulate_point_cloud`'s fans).
M10. **Leave nothing alive after the call.** Look for a Python frame, closure, pool or cache that
     keeps arrays alive past the call (the lazy-import frame held 26.9 MiB; §12.4). A cache that
     holds memory across calls is rejected (§13.1).

**Measure the peak** as the mempool high-water mark, `wp.get_mempool_used_mem_high(device)`
after a synchronize. Measure in a fresh process per arm, since the mark only rises. Measure on
the largest mesh the row supports (dragon, lucy, or a high octree depth), because memory problems
appear only there. Raise the release threshold (`wp.set_mempool_release_threshold`) when timing
memory changes, so the arms do not pay different pool trims. Report the peak beside the time.

**When memory and time pull apart,** the user decides. Take a memory saving that costs time only
when it lets an input run that otherwise fails with out-of-memory, or when it costs nothing
measurable. Record the trade either way: an `(E, 2)` edge list took 0.55-0.64x the peak but
1.3-2.4x the time, and was declined.

### D. Algorithm: the large wins

19. **Replace the algorithm.** The largest ratios in the history came from changing the method,
    not from tuning it:
    - a narrow-band brick solver replacing `warp.fem` (4-460x);
    - the exact hierarchical winding number over a Morton tree (5-94x);
    - sparse Cholesky's symbolic analysis on the device (3.2 s -> 25 ms on dragon);
    - direct solves where an operator repeats, and multigrid V-cycles whose sweep is fused with the
      mat-vec;
    - a dataflow wavefront for serial-order sweeps;
    - `marching_cubes` as one count/scan/emit (2.4-3.5x).

    For the most expensive rows in scope, look at the literature and at MeshLib's implementation
    (usually close to the state of the art). Propose the bold option with its expected ratio and
    its risk to outputs. A change of algorithm that changes outputs needs the user's go-ahead.
20. **Choose per size or per device when the optimum moves.** Gate on a measured threshold
    (direct solves from 5 000 vertices, LSCM from 12 000, the multigrid coarsest level dense when
    its key product fits). Use a per-device algorithm only when the parallel form does
    asymptotically more work (§14).

**Before proposing any item, check for a recorded decline.** Grep the call site, the constant
comments, the benchmark docstring, §14's refuted list and `component-status.md`. A decline is
re-opened only with new evidence: the code it measured has changed since, or the context differs
(a new size, a new Warp). Say which. "It was reverted once" is not a reason to skip an idea when
the surrounding code has changed; the user expects a retry then.

Write the candidate list with the expected saving per item: launches, readbacks and allocations,
and the share of the call they represent at small and large sizes. Then start, unless the user
asked for a plan first.

## 2. Implement one item at a time

- **Fusion means extraction (§2.4).** The statements shared by the fused kernel and any surviving
  sibling become one `@wp.func`, in the module that owns the quantity.
- **The deterministic setting (§2.2).** With the setting off, the path keeps the same kernels,
  launches and allocations, and its cost must not rise, on the CPU too. Reproducibility work goes
  only into the on branch. A new reduction or scan routes through the shared forms rather than
  inventing a local strategy.
- **Output equality.** On the CPU, with the setting on, compare old and new outputs byte for byte.
  A reordering of float operations is acceptable when the user allowed it; say so in the commit.
  Decision arrays (masks, counts, indices) must be identical.
- **Every Part I rule still applies:** checks 13, 15, 22 and 23, overload registration (§2.5),
  `wp.map` declaration tables (§3.5) and helper ordering (§5). After introducing a new construct,
  grep `tests/api_conventions.py` for AST checks it might hide from (§4.5).

## 3. Price it

- **Peak memory beside time.** For a change that touches buffers, record the high-water mark on
  the largest mesh as well (C, "Measure the peak"). A peak that rises is a regression even when
  the time holds.
- **Counts first.** Rerun the census for the touched module and diff against `/tmp/base.json`.
  Every count that moved must move the way the change said it would.
- **Clock second.** Run an interleaved A/B against the baseline worktree, one process per arm
  group:
  - at two or more sizes, including a scrambled large mesh (the §14 gather-form trap);
  - reporting the minimum beside the median;
  - on CUDA, which is the target, and on the CPU, where it must not regress beyond noise. Split a
    per-device constant where the optima differ (§13.2).
- **A share that falls as the mesh grows is a decline** (§15). Write it at the site with the
  numbers, then move on.
- **A fusion that changes a slot or box index gets a debug-mode run** (§14).
- **Do not run the full all-libraries benchmark sweep or regenerate the figures** unless asked. A
  sweep prices with targeted rows only.

## 4. Generalize what worked

Each landed change is a technique, not only a fix. Before closing the scope, name the technique
behind every landed item in one line, for example "dense flag table -> bitset with popcount
rank", "per-block float atomic -> warp-aggregated lane sum", "readback of a bound -> caller
keyword", "sort-to-group -> one-hot scan", "map + reduce -> producer commits its block sum".
Then, for each technique:

1. **Make it a shared helper** if it is not one yet: a `@wp.func`, or a Python-scope helper in
   `_launch.py` / `array.py`, placed where its quantity lives (§3.1). A local copy of the trick
   inside one kernel module is the finding to fix first.
2. **Search the whole repository for the same shape**, whatever the sweep's scope: the code
   pattern (grep the old spelling, e.g. `wp.zeros(..., dtype=wp.int32)` used as flags, or an
   `atomic_add` on one slot), and the census report for the same launch sequence.
3. **Rank the other sites** by the count or time they would save, and price each one as in §3. A
   technique that pays in one function can lose in another (a different size, a different access
   pattern): record the decline at that site and move on.
4. **For a submodule scope, sites outside it go into the report** with their expected saving, as
   a list the user can approve. Apply them only after approval, or when the user asked for
   generalization across the repository.

Where several sites had already solved one problem in different ways (two strategies for one
reproducible sum, two bit-rank helpers), pick the best, priced, and route every site through it.

## 5. Caller-supplied hints

Some functions read back or recompute something the caller often already has: `n_vertices`,
sorted edges, adjacency, a built `wp.Mesh` / BVH, a value cached on `Trimesh`. For those, add an
optional keyword (§3.10):

- Measure the share it saves at small and large sizes. Leave it out if it is below about 2 % at
  every size, unless the user asked for the API addition anyway.
- The keyword is trusted, not checked. The docstring states the bound the value must satisfy, and
  that the answer is identical either way.
- Pass it from every in-repo caller that knows it, and from `Trimesh` where the value is cached.
- When a private helper gains the keyword because an internal caller needed it, ask whether the
  public function should expose it too. The answer is usually yes.

## 6. Gates and commit

Run these for every commit:

- `ruff format` / `ruff check --fix` / `basedpyright` (0 errors);
- the touched modules' tests on CUDA, plus the CPU through `uv run python -m tests.devices` when
  the change is device-dependent (§7.2);
- `tests/test_api_conventions.py`, then a re-record of both census budgets when counts moved
  (`benchmarks/README.md`).

Commit messages follow the repo's gitmoji style (`:zap: <module>: <what>, <ratio>`).

**Signing:**

- **Sign when GPG is unlocked.** Commit normally, so the commit is signed.
- **If GPG is locked, keep working.** When the signed commit fails (pinentry times out, `gpg
  failed to sign the data`), commit the same change with `git commit --no-gpg-sign` and carry on.
  Do not stop the sweep or wait for the user. Try a signed commit again at the next commit, since
  the user may have unlocked GPG in the meantime.
- **Tell the user about every unsigned commit:** after each batch, list the unsigned hashes.
- **Never push an unsigned commit.** Before any push, check the range:
  ```bash
  git log --format='%G? %h %s' origin/main..HEAD
  ```
  Every line must start with `G`. If any line does not, stop and ask the user to unlock GPG.
  Then re-sign the range:
  ```bash
  git rebase --exec 'git commit --amend --no-edit -S' origin/main
  ```
  Check the range again, and push only once every commit is signed.
- Push only when asked. At the end, offer to remove the worktrees and merged branches the sweep
  created.

## 7. The closing report

Write it in plain prose:

- **Landed:** each commit (by hash, unsigned ones marked) with its count deltas, its peak memory
  where buffers moved, and its CUDA/CPU ratios at the
  sizes measured.
- **Generalized:** each technique, the helper it now lives in, and every site it was applied to
  or declined at, with numbers. For a submodule scope, also list the out-of-scope sites waiting
  for approval.
- **Decisions taken on the user's behalf:** API additions or removals, allowlist entries added,
  rule bends, changed defaults. Give one line of reason each; the user asks for them when they are
  missing.
- **Declined:** each idea with its number, and where its decline comment lives.
- **Left open:** what was not attempted and why, recorded in §16 in the same commit as the work.
  Results go to `component-status.md`; §16 lists only open work.
