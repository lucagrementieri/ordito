# Role: NVIDIA Warp Expert

You are an expert in NVIDIA Warp (`wp`) working on **ordito**, a GPU geometry-processing library.
Follow every rule below when writing kernels, `@wp.func` helpers, Python-scope wrappers, tests and
benchmarks.

- **Part I (§1-§11)** — rules: how code in this repository must be written.
- **Part II (§12-§16)** — a digest of what was measured on this hardware (RTX 5090, warp-lang
  1.19.0): one line a verdict. The full record, with numbers, dates and sites, is
  `reference/measured-facts.md` under the same section numbers. **Read the matching section there
  before proposing an optimization or diagnosing a slowdown**: most plausible ideas have already
  been built and priced, and several lost.
- Also read on demand: `reference/reference-libraries.md` (traps of each test/benchmark reference
  library) and `reference/component-status.md` (per-component measured facts and refuted plans).
- Record a new measured finding in `reference/measured-facts.md` (or the component's section of
  `component-status.md`); add one line here only when it changes what to do, and a rule that
  follows from it goes into Part I. A measured decline is a result: do not re-propose it without
  new evidence.
- Section numbers are cited from code comments (check 24): never renumber or remove a heading.
- A Warp version claim is spelled `Warp 1.19` (the word "Warp" immediately before the number); the
  staleness check (§4.5, check 9) reads only that anchored form.

---

# PART I — RULES

---

## 1. Kernel syntax and semantics

### 1.1 Decorators

- `@wp.kernel` for launched entry points; `@wp.func` for helpers.
- A `@wp.kernel` must not return a value: annotate `-> None` and write into output arrays.
- Every argument of a `@wp.kernel` and `@wp.func` must be explicitly typed.
- Call `wp.tid()` only inside a `@wp.kernel`; pass the index to helpers.
- A `@wp.func` may return `tuple[T1, T2, ...]`.

### 1.2 Type standards

- Subscript-style array annotations: `wp.array[wp.vec3]` (check 14).
- In `ordito/kernels/` only: `wp.array2d[T]` / `array3d` / `array4d` with multi-index `wp.tid()`. Wrappers use `ordito.typing` aliases (§3.2).
- No bare `bool` / `int` / `float` annotation in a kernel or `@wp.func` signature (check 18); kernel factories are ordinary Python and keep them.
- Do not call `wp.constant()` (an identity). The typed constructor is what matters: `TOLERANCE_MERGE = wp.float32(1e-8)`. A plain Python float is `float32` in kernels and mixing it with `float64` is a parse error; use `wp.float64(...)`.
- Prefer dtype-generic `@wp.func`s (`wp.Float` / `wp.Scalar`, `Any` for vectors; the `kernels/predicates.py` convention). Limits: §12.3. `Any` is also generic over rank and dimension, so do not pin a helper to one; use rank-free reductions (`wp.min(wp.abs(wp.get_diag(r))) < tol`, since a matrix has no `.shape` in kernels).

### 1.3 Casts and conversions

- Cast `wp.tid()` when used as an index: `f = wp.int32(wp.tid())` (check 22; multi-index unpacks cannot carry a cast).
- `wp.int32` / `wp.float32` are the only cast spelling; never bare `int(...)` / `float(...)` (check 16). `float(...)` fails inside `wp.Float`-generic functions; use `type(x)(...)` where a function is or could be generic.
- A cast to the type a value already has is noise; delete it, same-type constructors such as `wp.vec3(*(a - b))` included. The tid cast is the one kept exception.
- A cast of a bare literal is never redundant: `wp.int32(0)` is mutable, `0` is a constant (§1.6).
- `wp.cast(expr, T)` reinterprets bits between same-size types only; widen or narrow with the constructor. Whole-array conversion: §3.6.

### 1.4 Kernel-scope restrictions

Not supported inside kernels and `@wp.func`s: lambdas, comprehensions, sets, dicts, `list.append()`, `eval()`, recursion, exceptions, Python tuples for initialization (use `wp.vec3(1.0, 2.0, 3.0)`).

- Small fixed collections are vector types; larger ones are `wp.types.vector(length=N)`, not `wp.zeros(shape=N, ...)` (§2.9).
- A tuple cannot be indexed at runtime; a vector can: unpack into locals first (`a, b, c = corner_triple(faces, f)`, then `wp.vec3i(a, b, c)`). A starred call over a `@wp.func` returning a tuple (`wp.vec3i(*corner_triple(faces, f))`) fails to parse, and a warm kernel cache hides it until the module is next built.
- A variable defined only inside an `if` branch is uninitialized when the branch is skipped; initialize first.

### 1.5 Arithmetic and conditional spellings

- Integer `//` floors while `/` and `%` truncate (C++ sign of the dividend), so they differ on a negative dividend. Check the dividend's sign before relying on either.
- Spell integer division `//`, never `/` (check 17). Do not respell `//` as `/` for speed.
- A conditional value is `wp.where(cond, a, b)`, not a ternary (check 20). It evaluates both arms; use an `if` when one must not run.

### 1.6 Mutable loop accumulators

- A variable initialized to a bare literal (`count = 0.0`, `n = 0`) is a compile-time constant and cannot be mutated in a dynamic loop (an error, or silently the initial value). Declare accumulators with the typed constructor: `count = wp.float32(0.0)`, `n = wp.int32(0)`.
- Never let `ruff check --fix` strip the constructor (it rewrites `int(0)` to `0`); legacy spellings carry `# noqa: UP018, RUF046`.
- An accumulator initialized from a **module-level constant** (`best = wp.float64(_DBL_MAX)`) *is* that constant's codegen variable: the loop rewrites it and a later comparison with the constant reads `best < best`. Initialize from a computed expression: `wp.float64(_DBL_MAX) + wp.float64(0.0)`.
- Never change a compile-time constant's type in kernel scope (an unrolled `for side in range(6)` variable or a module constant passed to `wp.uint64(...)`): the cast rewrites the constant's variable where it stands and earlier reads see it unset (release builds hide it, `-O0` reads garbage). Cast a variable derived from it (`f = wp.int32(side)`, then `wp.uint64(f)`), or give the other type its own module constant.
- When a branch never fires, read the generated `.cpp`.

---

## 2. Kernel architecture and launches

### 2.1 Launch discipline

- `wp.launch(kernel=..., dim=..., inputs=[...], device=...)`. Always forward `device` from the input arrays: a memory-safety rule (§3.9, check 15).
- Prefix output arguments `out_` and put them after all inputs (check 13). Exempt (`_KERNEL_OUTPUT_ALLOWLIST`): in-place arguments and scratch / persistent-state buffers, named for what they hold. A read-only input never wears `out_`.
- Derive the face count as `f = faces.shape[0] // 3`.
- Every launch runs on Warp's current stream, which the caller owns (`docs/streams.md`); ordito never picks one. A graph is recorded through `_common.scoped_capture`, never `wp.ScopedCapture`, and a `warp.optim.linear` solver is called inside `with _common.recordable_stream(device):` (check 33). ordito is not safe to call from two threads at once (§12.1).

### 2.2 A lane-parallel kernel strides by `wp.block_dim()`, or it stays lane-free

A kernel whose lanes cooperate (`wp.tile_sum` / `tile_min` / `tile_max`, a `wp.tile_bvh_query_aabb` walk) is correct on both devices only when its lanes partition a sequence the block already owns, with the stride taken from `wp.block_dim()`, never from a kernel argument or constant.

- `launch_tiled` runs one lane per block on CPU, where `wp.block_dim()` reads 1; a one-element tile is not a bug, an arg-strided sum is (§12.2).
- Where lanes would partition the outer work (a whole-array reduction with no per-item dimension), keep a constant stride on CUDA only with a lane-free `_sliced` sibling for CPU (`_common.prefers_tiled_reduction`), or stay lane-free on both.
- A lane-strided reduction launches at `dim=kernel_reduce.blocks_1d(n)`, never `dim=n / TILE_1D`. Assemble a wide accumulator before the single commit (§13.2).

**Determinism** is one global setting, `ordito.config` (`set_deterministic`, `with deterministic(True):`), off by default in the library and benchmarks, on for every test (`tests/conftest.py`). It also fixes output order (an atomic-slot fill becomes a stable sort).

- New code obeys it by routing through the shared forms, which read it at the call:
    - a float block commit: `kernels/reduce.store_block_sum` / `store_block_total` into a `_launch.block_partials` buffer, never a bare per-block `atomic_add`; integer commits keep `commit_block_*`;
    - a float scan: `_launch.array_scan`;
    - a per-group float sum: `kernels/scatter.scatter_add` / `scatter_add_vector` / `scatter_add_matrix` with a `_launch.group_sums` buffer (`launch_group_sums`, `finish_group_sums`); a consuming kernel reads through `kernels/reduce.read_group_sum` from `group_sums(fused=True)`. Public forms: `reduce.scatter_sum`, `reduce.segment_sum`. Each call names its off-mode pooling: `ARRIVAL_POOL_PEERS` for keys hot across the launch (component / region labels), `ARRIVAL_POOL_UNIFORM` for sorted or clustered keys;
    - a per-vertex float scatter (`add_corner_triple`, normals through `_common.scatter_normal_sums`) and `vertex_face_adjacency` obey it too.
- A vector or matrix group sum shares one exponent across its components, so a small component beside a large one is lost; components of unrelated scale sum as scalar groups (slot `group * n + c`).
- Never add a per-call `deterministic` keyword or thread the mode through signatures.
- **A reproducibility fix is gated on the setting, never unconditional.** Off, the code is the fast arrival-order path it was (same kernels, launches, allocations, dtypes). "Costs nothing measurable" does not justify making it unconditional. A reproducibility test claims the on mode only.
- **Determinism is the requirement, not a particular summation order.** Matching another form's order, or the CPU's, is never a reason to keep a slow shape; compare against a reference at tolerance.
- Keep a site's existing form where §14 prices it faster.

### 2.3 Block-per-item is an occupancy trade, not a shape choice

Use one block per item only when the outer per-item dimension alone would starve the device. A kernel that already has a slice dimension does not qualify; collapsing it into lanes loses 2-8x. Compute the launch's thread count and measure at the large end first, since the small end flatters. Verdicts: §14.

- A thread-per-item serial walk launched over few items gives each item a warp through the shared pair, never a hand-written mapping: `dim = m * _launch.walk_lanes(m, device)` (one on the CPU) and `item = kernels/array.walk_item(wp.int32(wp.tid()), lanes)`, returning on `-1`.
- A serial walk counted in one launch and written in another walks once: the counting pass stages its points (`kernels/array.stage_point`, from `first_staged_chunk`), the writing pass copies them (`copy_staged`) and walks again only on `False`; `_common.staging_pool` allocates the pool. Price it first: it pays where the walk is the device time, not on a host-bound call.

### 2.4 Fusing a kernel: extract the shared part as a `@wp.func` in the same commit

A fusion is done when the code it duplicated has a name. Whenever a kernel is fused, split, specialised or given a second variant, the same commit must:

1. Name each shared statement run (corner load, window computation, emit protocol, guard sequence) as one `@wp.func` both kernels call.
2. Put it where the quantity lives: predicates in `kernels/predicates.py`, per-face quantities in `kernels/triangles.py`, index/sort/search helpers in `kernels/array.py`, scatters in `kernels/scatter.py` (§3.1).
3. Say in a comment what the kernels still differ by. Prefer one kernel with a warp-uniform int selector (§2.7) when the difference is a parameter; keep two kernels naming each other when they are two algorithms.

- A comment naming the other copy ("mirrors X") is the finding: extract a shared `@wp.func`.
- A duplicated decision rule is a correctness hazard even when the control flow differs; extract it as a `@wp.func` returning the classification (a sentinel for "reject").
- Factor the family, not the pair: write the generic form and check the other rank and precision.
- One predicate, one spelling per module: `wp.length(d) < r` and `wp.length_sq(d) < r * r` differ in float32.
- A green suite does not prove an extraction was behaviour-neutral; read the diff and gate on the decision arrays a kernel writes.

### 2.5 A generic kernel registers its overloads at import (`wp.overload`)

A `@wp.kernel` generic over a dtype (`wp.Scalar`, `wp.Float`, `wp.Int`, `Any`) must register its concrete overloads at import, in a `_register_overloads()` at the bottom of the file. Otherwise the first launch at each new dtype silently recompiles every kernel in the module.

- Register what the wrapper's dispatch can reach; for independent generic arguments register the cross product, not a diagonal.
- Registration is not compilation: never call `wp.load_module` / `wp.force_load` at import.
- The test only checks that some overload exists; a missing dtype shows up as a slowdown (§15).
- Keep the `wp.Kernel` that `wp.overload` returns and launch through it: a module builds a dtype-keyed `kernels/array.OverloadTable` and the wrapper writes `kernel_laplacian.COTMATRIX_TRIPLETS[cot_entries.dtype, dtype]`. A new generic kernel adds a table, not a bare `wp.overload` call.

### 2.6 Backward passes, and in-place `@wp.func` parameters (`wp.ref[T]`)

Only `ordito/kernels/metrics.py` is differentiated. Every other kernel module sets `wp.set_module_options({"enable_backward": False})` right after its imports; a new module copies the line.

- `@wp.func` helpers may take `wp.ref[T]` parameters to mutate caller storage (`update_argmin` in `kernels/array.py`); `T` must be concrete (§12.3).
- Any kernel calling a `wp.ref` helper must be `@wp.kernel(enable_backward=False)`; the module option is not consulted at parse time.
- Never use `wp.ref` in `kernels/metrics.py`.

### 2.7 Function-valued parameters and kernel factories

- A `@wp.func` may take `fn: wp.Function`, bound at compile time per call site (user `@wp.func`s and simple builtins, not tile intrinsics).
- `wp.launch` cannot pass a `wp.Function`. For runtime selection pass an int and branch in a dispatch `@wp.func` (`registration.robust_weight`, the `ACCEL_HASHGRID` / `ACCEL_BVH` pattern).
- Name a factory kernel with `wp.kernel(f, name=...)` (a valid C++ identifier), not by mutating `__name__`.
- A factory avoids generic-launch dispatch cost. Set `__annotations__` after defining the body, and give `dtype` no generic default:
  ```python
  def _factory(name, dtype):
      def _k(values: wp.array[wp.Scalar], out: wp.array[wp.Scalar]) -> None: ...

      _k.__annotations__["values"] = wp.array[dtype]
      _k.__annotations__["out"] = wp.array[dtype]
      return wp.kernel(_k, name=name)
  ```

### 2.8 `@wp.struct` argument bundles

Each launch argument costs ~1 µs of host time (§13.1). Bundle the invariant tables of a kernel with a dozen or more arguments, launched in a Python loop, into a `@wp.struct` built once in the wrapper.

- Do not bundle a kernel whose launch is captured.
- After introducing a struct, re-check static checks that match AST node types (check 13, §4.5), and the `+cu12` miscompile scan (§12.4).

### 2.9 Per-thread row storage

`wp.zeros(shape=K, dtype=T)` in a kernel is not register storage (pointer-addressed, 2x slower). Use `wp.types.vector(length=K, dtype=T)` (the `vec5d` pattern in `kernels/curvature.py`), which wins up to K near 64.

- Never pass the vector to a `@wp.func` (it spills); inline it per call site.
- Any runtime index spills it: read the k-th element with an unrolled `for slot in range(K): if slot == k - 1:`, closing over `K` in a factory (one kernel per bucket, unique `name`, `KNN_ROW_BUCKETS`).
- A register insertion sort needs an explicit `placed` flag; carrying the displaced element drops a neighbour on equal distances.

### 2.10 Every launch width and lazy kernel is declared for `ordito.precompile`

`python -m ordito.precompile` builds what `precompile.kernel_variants` lists without running a
workload, so a module a launch reaches only at run time must be declared.

- A launch at a `block_dim` other than its module's default declares the width with
  `kernels/array.declare_block_dims`: in a `_declare_block_dims()` at the bottom of the kernel
  module whose constant names it, or beside the wrapper constant that does. A width chosen at run
  time declares every value it can take; name a literal width as a constant first. A kernel
  launched at its own width lives in its own module (`module="unique"`, §12.4).
- A kernel created on first use (a factory, `cast_kernel`, an `int_vector_scan_kernels` dtype, a
  `wp.map` op at an undeclared signature) is created in a function decorated with
  `kernels/array.declare_lazy_kernels`, and lives in a module of its own (`module="unique"`).
- The census (§15) fails on a load the list misses. CI's CPU run sees modules only: re-run the
  CUDA census after adding a width.

---

## 3. Python-scope wrappers

### 3.1 Module layout and imports

- Every kernel lives in a `kernels/` sub-module, imported with an alias:
  `from ordito.kernels import triangles as kernel_triangles`.
- A top-level kernel module is named for the public module it backs (check 7); exceptions are
  `kernels/predicates.py`, `kernels/scatter.py` and the `kernels/algorithms/` sub-packages.
- `kernels/array.py`, `predicates.py`, `triangles.py` and `scatter.py` are also shared libraries.
  Put a helper reached from a second module in the one it belongs to; never make unrelated modules
  import an algorithm module to reach geometry.
- `ordito/__init__.py` is lazy (PEP 562 `__getattr__`) and must stay so.
- The radix sort is stable, so with an identity payload each class's first occurrence starts its
  run: a grouping that needs representatives reads them off the emit pass
  (`grouping._unique_keys_core`), not `first_occurrence_indices`.
- The three searches in `kernels/array.py` differ silently: `binary_search_index` is
  `searchsorted(side="right")` (`slot + 1` on an exact hit), `binary_search_index_left` is
  `side="left"` (for a lookup add `index < n and values[index] == v`),
  `binary_search_sorted_contains` is membership only.
- A radix sort of packed keys with a known radix passes `end_bit` and the producer writes keys
  straight into the sort's double-width buffer (`unique_1d(max_value=)`).

### 3.2 Typing (`ordito.typing`)

Import as `import ordito.typing as odt`; do not re-export typing symbols from `ordito/__init__.py`.

- Wrappers do not use `wp.array2d[dtype]`; use the `odt.Array2dInt32` / `Array2dFloat32` /
  `Array2dFloat64` / `Array1dInt32` aliases and the `IntArray` / `FloatArray` / `ScalarArray`
  unions. Kernels keep `wp.array2d[dtype]`.
- Validate with `odt.ensure_ndim` and `odt.as_array2d` / `as_array3d`, not `isinstance`.

### 3.3 Allocation and returns

- Allocate 2D/3D outputs with `odt.empty_2d((rows, cols), wp.int32, device=...)` / `odt.empty_3d`
  and return rank-2 results as `odt.as_array2d(arr, wp.int32)`.
- 1D outputs: `wp.empty(n, dtype=..., device=input.device)`; `odt.empty_1d` only in `reduce`,
  `metrics` and `neighbors`, whose signatures carry the rank.
- At Python scope a rank-1 array's length is `.size`. Keep `.shape[0]` in kernels, for rank-2 row
  counts, and for rank-free or `Any` operands.
- Always pass `device=` to every allocation, including in `_*.py` (check 10).
- Size buffers for their final use at allocation; never allocate-then-grow (the producer allocates
  the `n + 1` terminated form, `counts_to_offsets`).
- Warp raises on a zero-length slice; guard a trailing `fill_` with `if stop > start`.
- A buffer whose initial value matters is allocated holding it (`wp.zeros`,
  `wp.full(n, value, dtype=..., device=...)`), never `wp.empty` then `fill_` / `zero_`.
  `wp.empty` is right where every element is written before it is read.

### 3.4 Python-scope gather indexing (prefer over trivial gather kernels)

`src[indices]` yields a `wp.indexedarray`. Do not add per-element gather kernels when this suffices.

- Materialize a gather through `array.gather`, not `wp.copy(dst, src[indices])` (faster on CUDA).
- Index arrays must be 1D and **contiguous**: Warp silently ignores a view's stride, so
  `payload[edges[:, 0]]` returns the flat buffer's leading entries. A prefix slice (`arr[:n]`) is
  safe; a column or step slice must be `wp.copy`'d to a dense buffer first. This is why
  `kernels/edges.py:edge_lengths` stays a kernel.
- When converting a gather, verify values: the corrupt version is faster.
- Indexed assignment is unsupported; keep a small kernel (`mark_membership_mask`).

### 3.5 Elementwise ops at Python scope (`wp.map`)

Do not write a `@wp.kernel` whose body is only `out[i] = f(in[i], ...)`. Keep the op as a named
`@wp.func` in `kernels/` (or a builtin such as `wp.neg`) and call `wp.map(op, *inputs, out=...)`.

- Named `@wp.func` only, never lambdas. Always pass `out=` (in-place: `out=<an input>`;
  multi-output: `out=[a, b]`).
- `wp.map` infers a bare Python `int` as `wp.int32`: cast explicitly, e.g. `wp.uint64(bvh.id)`.
- In per-iteration loops, hoist the kernel with `wp.map(..., return_kernel=True)` and `wp.launch`
  it (`smoothing.py`).
- Each call signature (dtype, rank, a length-1 array, an `indexedarray`) forks the generated
  `map_<op>` module and recompiles it. An op reached at more than one signature declares them at
  import (check 23) in `_declare_map_kernels()` at the bottom of the owning module; Warp builtins
  in `kernels/array.py` (`declare_map_signatures`). Declare with `return_kernel=True` on a
  zero-length CPU array. Derive a table by instrumenting `wp.map` over a suite run.
- Never declare via `ordito/__init__.py`, `wp.load_module` or `wp.force_load`.
- Still a real kernel: thread index as data, whole-array arguments, scatters, row-indexed outputs.

### 3.6 Dtype conversion at Python scope

`wp.cast` is kernel scope only. Convert arrays with `od.array.astype(src, dtype)` (new array) or
`od.array.copyto(dst, src)` (into a buffer you hold). Never call `wp.utils.array_cast` from
`ordito/` (check 30). A kernel created lazily goes in its own module
(`wp.kernel(..., module="unique")`), or it recompiles every kernel in the existing one.

### 3.7 Sparse: assemble from keys, never through `bsr_from_triplets`; and `nnz` is a stale capacity

No `ordito/` module calls `warp.sparse.bsr_from_triplets` / `bsr_set_from_triplets` (check 27);
tests may, for an independent input. Build CSR in `ordito/array.py`, with no host readback:

- **Sparsity follows from the mesh: write keys, not triplets.** A producer writes one
  `kernels/array.csr_key(row, col, n_rows, n_cols)` per contribution (sentinel `n_rows * n_cols`
  for an empty slot) into `array.csr_key_buffers`; `array.csr_from_keys` sorts and returns
  `offsets`, `columns` and each entry's first sorted position; a per-row value kernel forms the
  entries. `laplacian._mesh_operator_pattern` is the model; a caller assembling repeatedly over
  fixed faces builds it once with `laplacian.mesh_operator_pattern` and passes `pattern=`.
- **A genuinely unordered coordinate list:** `array.csr_from_triplets` (`prune_numerical_zeros`
  drops entries whose sum is zero).
- **A CSR the producer can write directly** is written directly and wrapped by `array.bsr_from_csr`.
- An operator summing many contributions into each entry builds its pattern first (keys for the
  entries' pairs, `csr_from_keys`, one count readback to cut `columns`), then gathers each entry's
  contributions (`energies.hessian_energy` / `curved_hessian_energy`). Look for the factorization
  the contributions share before tuning a row gather (§14).

**Never size a buffer, slice or launch `dim` off `matrix.nnz`.** After a triplet build it is the
triplet count handed in, duplicates included, and only `nnz_sync()` repairs it; use `nnz_sync()`
or read `offsets[nrow]`. The failure is silent (an unwritten `wp.empty` tail feeds garbage into
real entries).

- A writer leaving triplet slots unwritten points them out of range, not at `(0, 0, 0.0)`.
- Any Warp `*_count` / `.nnz` field is a capacity until proven otherwise
  (`Volume.get_voxel_count()`; use `get_active_stats().voxel_count`).
- A key assembly may return a handful of spare slots in `columns` / `values`; leave them.
- A scanned `int32` count that sizes buffers is checked for a wrap through
  `_common.scan_count_checked`: a total past `2**32` wraps to a small positive count. Any per-row
  count summing neighbours' row lengths grows as the square of the largest valence.

More `warp.sparse` behaviour: §12.5.

### 3.8 NumPy at Python scope is sanctioned; leaking it through the API is not

A device round trip costs far more than small host arithmetic. **Do not open a "remove NumPy"
pass.** Host metadata math and host-sequential algorithms stay in NumPy; a closed-form parallel
map whose output scales with a resolution parameter (`creation.icosphere`, `creation.grid`) is a
kernel. Three things are defects:

- **A public signature or return that names `np.ndarray`** (outside `ordito/io.py`). Return
  `wp.mat33d` / `wp.vec3`; annotate duck-typed inputs `Sequence[Sequence[float]]`.
- **NumPy standing in for a Warp Python-scope equivalent that needs no host buffer** (`wp.full`,
  `arr[k:].fill_()`, `wp.determinant`, `wp.inverse`, `wp.svd3`), unless pricing says otherwise:
  Python-scope builtins are dispatch calls and often slower than NumPy or `math` (§13.1). Prefer
  `math.pi` / `float("nan")` over `np` constants.
- **NumPy reducing a full `.numpy()` readback** (`.min()`, `.any()`, `.sum(axis=0)`). Use
  `ordito.reduce` or `wp.utils.array_sum`; decide on the CUDA measurement, and keep the host path
  where the buffer never scales with the mesh.

In `ordito/`, delete a `.tolist()` immediately splatted into a Warp vector or matrix constructor
(keep it in `tests/`). `arr.numpy().tolist()` equals `arr.list()` only for rank 1.

### 3.9 Device checks and the launch-device memory-safety rule

Every public function with two or more device-bearing arguments calls
`_common.require_same_device(**named)` as its first statement, passing every array, mesh, BVH, hash
grid, `wp.Volume` or list/tuple of those, including `X | None` arguments. It raises `RuntimeError`;
document it with a `Raises` entry (check 11 cannot see the delegation). Internal code stays
trusting and forwards `device=` from an input array.

`wp.launch` does not raise on a device mismatch: it corrupts the host heap or segfaults.
`tests/conftest.py` sets `LaunchArrayAccessMode.STRICT` (keep the suite green under it), but
`STRICT` misses an omitted `device=` on a CUDA run; check 15 scans every launch.

### 3.10 Readbacks

- An array read back to the host is `ordito._common.read_array(arr)`, never `arr.numpy()` (check 32;
  docstring examples keep `.numpy()`).
- Every readback (`read_array`, `int(<device value>)`) carries a comment naming why it is
  unavoidable. When the caller can supply the bound a readback infers, expose a keyword
  (`face_adjacency(n_vertices=...)`) and pass it from every in-repo caller that knows it.
- Several adjacent small values: `_common.read_values(arr, start, count)`. A tail value:
  `_common.read_scalar(arr, index=-1)`, never a hand-rolled spelling (a pinned scratch races).
- Several small device values go into one buffer and are read once (`points.fit_plane`).

---

## 4. Evolving the public API

### 4.1 Naming

- Name a function after what it returns, in NumPy vocabulary, never after the Warp call it wraps.
- A mask is named `<element>_<property>_mask`, element first.
- A wrapper whose whole device side is one kernel nobody else launches shares that kernel's name. Reject filler affixes (`init_`, `do_`, `compute_`, `run_`, `make_`, `kernel_`, `_impl`, `_inner`); informative ones stay (plural for batched, `_pass` / `_step`, `finalize_`).
- A tuning choice is a keyword, not a name (`backend="hashgrid" | "bvh"`); use `None` for "not passed". Keep the discriminator in the benchmark group name and test that merged paths agree.
- One operation family, one module; place a function by what it computes, not where the reference library keeps it.

### 4.2 Signatures and returns

- A packed buffer and its offsets are returned and accepted values first: `(values, offsets)`, any per-item array last. A transposed unpack type-checks and indexes garbage.
- Offsets are always the `n + 1` total-terminated form (`indptr`; empty is `[0]`). Build by allocating `n + 1`, writing counts into `offsets[1:]` and scanning in place. Never read an item count as `offsets.shape[0]`.
- A variable-length-item function has a packed sibling `<name>_with_offsets` (the list form is that plus `array.split`); one taking the packed form is `<name>_from_offsets`.
- A function that rebuilds a mesh's connectivity reports where its output came from through one keyword, `return_provenance: bool = False`, appending an `interpolation.Provenance` to its returns (`Literal` overloads for both arms). Never a bespoke `return_index` / `return_operator`; a caller chains edits with `Provenance.then`. Off, the call issues exactly what it did; on, build it from what the edit already holds through `Provenance.from_edge_points`, `from_rows` or `from_clusters`, never by a geometric search. `tests/comparisons.assert_provenance_describes` is the check.
- No public signature may name `np.ndarray`, outside `ordito/io.py`. A guard must encode a real limitation.
- A `Literal` menu argument is validated at the public boundary and raises `ValueError` naming the argument, value and options; a delegating wrapper documents it and its test calls through the wrapper.
- When mirroring a NumPy function, mirror its positional signature and make `device` keyword-only.
- No speculative generality: add a parameter only when an in-repo caller needs it; remove unread ones.
- Inverse pairs cross-reference each other and have a round-trip test.
- Every public `ordito/<module>.py` has `tests/test_<module>.py` and `benchmarks/test_<module>.py` (check 4). Every public function and method has a benchmark row (check 31).

### 4.3 Docstring, signature and body must agree

- A documented `Raises` must be reachable; a direct `raise` needs a `Raises` block (check 11). Delegating to a shared guard means documenting its `Raises`.
- A documented validation must be performed.
- When a comment and the body disagree, decide which is load-bearing first; usually the comment is right.
- After a rank/shape/dtype rule change, check the paths bypassing the helper (early returns, `@overload` stubs, `Returns` prose).
- Tests parametrize over the boundary value of the parameter the contract turns on.

### 4.4 Moving or renaming

A move or rename of a public function carries everything derived from it in the same commit:

1. Its exclusive kernels (shared kernels stay and are imported).
2. Its tests, into `tests/test_<destination>.py`, keeping §5 order.
3. Its benchmark rows, into `benchmarks/test_<destination>.py`.
4. Its `benchmark(group=...)` name and every `parity` / `noparity` marker citing it; `uv run python -m tests.parity` must show the same pair count.
5. Its `docs/SUMMARY.md` and `docs/reference.md` entries (check 28) and every cross-reference (`zensical build --strict`).

Renaming a keyword argument also needs a grep of the old name alone (`**splat` `TypedDict` fields) and of fenced docstring examples (check 12). After any rename run the whole suite.

### 4.5 The mechanical gate: `tests/api_conventions.py`

Thirty-three checks fail the default `pytest` run. Read an allowlist entry's reason before adding one; prefer fixing the code.

1. Summary line naming a reference library (§6).
2. `*_mask` producer not returning `wp.array[wp.bool]`.
3. Module summary ending in `(Warp)` / `on NVIDIA Warp`.
4. Module without both a `tests/` and a `benchmarks/` file.
5. Private name reached across a module boundary (a new `_*.py` for one helper dodges it).
6. One public name exported by two modules.
7. `kernels/<name>.py` without `ordito/<name>.py`, or the reverse.
8. Private helper defined above its first caller (§5).
9. Comment or docstring blaming a Warp version older than installed (anchored `Warp 1.19` spelling).
10. Allocation with no `device=` (§3.3).
11. Public function that raises with no `Raises` block (§4.3).
12. Fenced ```python docstring example that does not run on its own, or prints other than its trailing comment says (§6).
13. Kernel `out_` prefix and position (§2.1).
14. Non-subscript array annotations (§1.2).
15. `wp.launch` with no `device=` (§3.9).
16. Bare `int(...)` / `float(...)` in kernel scope (§1.3).
17. `/` on declared-integer operands (§1.5).
18. Bare `bool` / `int` / `float` kernel annotation (§1.2).
19. Reference-library comparison test with no class label (§7.4).
20. Kernel-scope ternary instead of `wp.where` (§1.5).
21. MeshLib or promesh name under `ordito/` (§7.6); the quoted method value `"meshlib"` is exempt.
22. Bare single-index `wp.tid()` (§1.3).
23. `wp.map`'d `@wp.func` with no declaration table (§3.5).
24. `.claude/CLAUDE.md` cross-reference to a nonexistent section, or a bare chapter number where that chapter has numbered subsections.
25. `!!!` admonition inside a numpydoc item-list section (§6).
26. Warp-typed module constant used as a Python-scope operand or slice bound (§13.1).
27. `bsr_from_triplets` / `bsr_set_from_triplets` under `ordito/` (§3.7).
28. Public module missing from, duplicated in, or stale in `docs/SUMMARY.md` or `docs/reference.md` (§6).
29. Array docstring entry opening with `Length-`, `Shape ``(`, `Flat` or `Rank-N ``(` (§6).
30. `warp.utils.array_cast` under `ordito/` (§3.6).
31. Public function or method that no `benchmarks/` row names (`od.<module>.<name>`), outside `_BENCHMARK_EXEMPT`.
32. `.numpy()` readback under `ordito/` outside `_common.py` (§3.10).
33. `wp.ScopedCapture` outside `_common.py`, or a `warp.optim.linear` solver call outside `with recordable_stream(...)` (§2.1).

- A new Warp construct (first `@wp.struct`, `wp.ref`, tile intrinsic, even a plain `@wp.func` extraction) can silently hide buffers from a check that pattern-matches AST nodes; after introducing one, grep `tests/api_conventions.py` for such checks. §2.4's extraction wins over the check.
- Write checks with a staleness half: an allowlist entry that matches nothing fails.

---

## 5. Function ordering within a module

Source order is the rendered docs order.

- Wrapper modules (`ordito/*.py`): docstring, imports, constants / aliases, functions. Group public functions thematically, most important first; the simple form precedes its variants.
- Private helpers (check 8): one caller means immediately after it; several means after the last; cross-cutting utilities go in a trailing section. Never above the first caller.
- Kernel modules (`ordito/kernels/*.py`) invert the rule: a `@wp.func` must textually precede every kernel or `@wp.func` calling it. Mirror the wrapper's public order, place helpers immediately before their kernel, and a shared helper before its first user.
- Tests (`tests/test_<module>.py`) mirror the wrapper module's public-function order.

---

## 6. Documentation (Zensical + mkdocstrings)

```bash
uv run zensical build --strict --clean  # validate; --clean after docstring edits
uv run zensical serve                   # preview; restart to see docstring changes
```

- API pages come from `api-autonav`, one per public module. A new module needs a line in `docs/SUMMARY.md` and in `docs/reference.md` (module table, and the task index when it answers a common task) (check 28). Never use `gen-files`: Zensical ignores unsupported plugins silently.
- mkdocstrings templates are overridden in `docs/.overrides/mkdocstrings/`; layout CSS is `docs/stylesheets/extra.css`. Check rendering changes in the built HTML, not only the exit status.
- A build finishing in about 0.1 s with an empty `site/` is inotify exhaustion (exits 0); raise `fs.inotify.max_user_watches`.

### Docstring content

- The summary line says what the function returns, never the C++ call it wraps (checks 1, 3); attribution goes in `Notes` or `See Also`.
- An array entry (every `Parameters` / `Returns` / `Yields` / `Attributes` entry, and a `Trimesh` property summary) opens with its shape as a code span, then its meaning: ``` ``(3 * n_faces,)`` flat triangle index buffer. ``` A rank-1 length is a 1-tuple; no `Length-`, `Shape`, `Flat`, `Rank-1` (check 29). Spell sizes in full (`n_faces`); define a free size (`m`, `k`) on first use.
- Every public function and class ends its docstring with an `Examples` section: one fenced ```` ```python ```` block (never `>>>`) that imports what it uses, builds its own input (`od.creation`, a small literal) and sets no `device=`. It runs in a fresh namespace on each device, and a top-level `print(x)  # value` must print exactly `value` (`tests/doc_examples.py`), so print shapes, counts, booleans and floats rounded to the digits CPU and CUDA agree on. Guide pages under `docs/` follow the same print rule.
- No measured timing (ms, ratios, launch or byte counts) in a public docstring; state the claim in caller terms.
- No development-history narrative in `ordito/*.py` ("measured X, declined Y", "reverted", internal section references). Keep the behavioural fact; reasoning belongs in `kernels/`, `benchmarks/`, `tests/` or the reference files, which may state a ratio or crossover but not a wall-clock figure.

### Docstring syntax

NumPy-style, with mkdocs-autorefs syntax; Sphinx roles (`:func:` etc.) render as literal text.

| Target | Syntax | Example |
|---|---|---|
| Internal (`ordito.*`) | `` [`short_name`][fully.qualified.path] `` | `` [`face_adjacency`][ordito.graph.face_adjacency] `` |
| External with inventory (`trimesh`, `numpy`, `scipy`, stdlib) | `` [`fully.qualified.name`][] `` | `` [`trimesh.grouping.group_rows`][] `` |
| External without inventory (`warp`, `igl`) | double-backtick code span, no link | ``` ``warp.sparse.BsrMatrix`` ``` |
| Shapes, literals, C++ names, paths | plain code span | ``` ``(n_vertices,)`` ``` |

- Resolve internal refs to the fully-qualified path, even within the same module.
- Use `!!! note`, only in free prose (description, `Notes`, `Examples`), never inside an item-list section (check 25).
- After editing docstrings, `grep -rnE ':(func|attr|meth|class|data|mod):\`' ordito/` must return nothing.

---

## 7. Testing

Every new geometry function MUST have regression tests comparing against a CPU reference
implementation. `trimesh` is the default; §7.6 covers the other nine libraries.

### 7.1 Conventions

- Test file `tests/test_<module>.py`; import `trimesh.<module> as tm` and `ordito.<module> as od`.
- Use the `device` fixture from `tests/conftest.py`. A test names `device` only if it touches the
  device directly (`reportUnusedParameter` enforces it); one reaching it through a mesh fixture is
  parametrized anyway.
- pytest alone, never `unittest` (ruff `TID251`): `monkeypatch`, plain `assert`, `pytest.raises`,
  `pytest.warns`. A test that counts or intercepts launches or allocations patches both
  `ordito._launch.*` and the `wp.*` fallback.
- `np.random.default_rng(seed)` with a fixed seed per test.
- Upload a NumPy mesh with `conversions.numpy_to_warp(vertices_np, faces_np, device)`, never a
  local helper (`numpy_to_warp_uv` for `wp.vec2`, `warp_to_trimesh` for the inverse). A triangle
  soup `(n, 3, 3)` becomes an indexed mesh first: reshape to `(-1, 3)` with
  `np.arange(n * 3, dtype=np.int32)` as faces.
- Call `.numpy()` on Warp outputs inline, before the NumPy comparison; bind no intermediate.
- `np.allclose(..., rtol=1e-5, atol=1e-5)` for floats; `np.array_equal` for bool and int.
- Variable suffixes name the library: `_np`, `_tm`, `_wp`, `_igl`, `_pp`, `_pml`, `_o3d`, `_pv`,
  `_ml`, `_pmf`, `_p3d`, `_m3`. Avoid `got` / `exp`.
- A NumPy vector to a `wp.vec3` argument in a test: `wp.vec3(*array_np.tolist())`.

### 7.2 Devices and the two-process runner

`--device={auto,cpu,cuda,both}` selects devices for this process (`auto` is cuda if available).

- While developing: `uv run python -m pytest tests/test_<module>.py -q --device=cuda`.
- Both-device coverage is a two-process job: `uv run python -m tests.devices` (a CUDA pass, then a
  CPU pass with `CUDA_VISIBLE_DEVICES=""`), because CPU work is ~36x slower once CUDA is
  initialised. Never use `--device=both` on a GPU box for CPU coverage. Use it when a change is
  device-dependent by construction (`launch_tiled` kernel, `prefers_tiled_reduction`, a
  device-gated constant).
- CPU passes run under `pytest-xdist` (`--cpu-workers`, default 8); a CUDA run stays one process.
  A test must not depend on another test having run in its process.
- CPU is the deterministic oracle for byte-for-byte A/B; on CUDA any reduction-order change is
  noise.
- A test costing more than about 15 s on CPU wears `@pytest.mark.slow_cpu(<seconds>)`.
- A `launch_tiled` kernel's test pins `"cpu"` explicitly in a parametrize (the fixture returns
  `cuda:0` whenever CUDA exists). `python -m tests.devices --cpu-blocks` adds a pass running CUDA
  tile code on the CPU.

### 7.3 Mesh fixtures

Reuse `tests/conftest.py` fixtures (each returns `(mesh_tm, mesh_wp)`):

| Fixture | Use when |
|---------|----------|
| `icosahedron`, `icosphere`, `icosphere_coarse` | Easy closed solids; edge-case and sign tests only |
| `unit_box` | Exact 90° creases; crease/seam/dihedral tests |
| `cave_cube` | Hollow / multi-component shell |
| `hemisphere`, `half_torus` | Easy open surfaces |
| `boy_surface` | Closed, non-orientable, χ = 1 |
| `mobius` | Non-orientable with one boundary loop, χ = 0 |
| `bohemian_dome` | Closed genus 1, self-intersecting |
| `saddle_graded` | **Default open fixture** (`OPEN_MESHES`): graded disk, needles of aspect ~4 800, one rim; `saddle_graded_large_arrays()` for size-dependent tests |
| `sphere_irregular` | **Default closed fixture** (`CLOSED_MESHES`): degree 3-12, aspect up to ~300, mostly obtuse, scrambled order, rotated and off-origin |
| `sphere_irregular_hollow` | Solid with a cavity, two components |
| `torus_irregular`, `torus_irregular_holes` | Genus 1 with needles; the second adds three holes (`BOUNDARY_MESHES`) |
| `sphere_irregular_cap`, `sphere_irregular_band`, `convex_irregular_cap`, `convex_irregular_band` | Cut bodies with planar rims (bumpy / convex) |
| `sphere_well_shaped`, `sphere_well_shaped_open`, `torus_well_shaped` | Same surfaces with no needles (aspect ~6), for premises the needles break |
| `sphere_round`, `torus_round` | Exact unit sphere / torus in a known frame (`conftest.round_frame_coordinates`): analytic oracles only |

- Import the shared sets `CLOSED_MESHES`, `OPEN_MESHES`, `MESHES`, `BOUNDARY_MESHES`; keep a local
  list only for a genuinely different set, with a comment saying why.
- Keep the hardest fixture per class, not many easy ones. A test about topology names the fixture
  that has it. A test whose premise the hard fixtures break (Delaunay, convex, planar rim,
  `float32`-length paths, hard-coded counts) names a well-shaped one, with the reason beside it.
- The scrambled fixtures sit off the origin: probe at `center_mass`; analytic oracles use `sphere_round`.
- Do not hand-roll `wp.Mesh(...)` unless the case needs a bespoke degenerate mesh. Never construct
  a zero-triangle `wp.Mesh` on CUDA (§12.1); a single-triangle mesh reaches an "empty mesh" guard.
- Parametrize over fixtures with `request.getfixturevalue(mesh_name)`; allocate queries on
  `mesh_wp.device`.
- Do not use `trimesh.slice_plane` output as a reference input; assert the hole count first.

### 7.4 The parity gate: a benchmarked reference must be a tested reference

`tests/test_parity.py` fails when a benchmarked `(group, library)` pair is neither tested nor
exempted. The `benchmark(group=...)` name is the key. Both markers are stackable and take string
literals only; `uv run python -m tests.parity` prints the matrix.

```python
# tests/test_edges.py -- this test proves ordito agrees with trimesh for that group
@pytest.mark.parity("faces_to_edges", "trimesh")

# benchmarks/test_curvature.py -- timed, but the results are not comparable
@pytest.mark.noparity("pymeshlab", oracle="trimesh", reason="...why not comparable, with numbers")
```

- Where a reference computes the same quantity, one test must compare the two outputs; an invariant
  does not discharge it. Check that the reference really has the quantity (§7.6).
- Where no reference computes it, an invariant-only test is the honest answer. Its docstring says
  "Not a library comparison: <why none exists>" and what the invariant excludes.
- Classify every comparison in the docstring, written as `Class A`:
    - **A**: direct `np.allclose` / `np.array_equal`. The default. An answer equal up to how its
      elements are numbered is Class A too, through `ordito.equality`: `meshes_allclose`,
      `faces_equal`, `points_allclose`, `rows_equal` (sort each row first for undirected edges),
      `partitions_equal`. `atol` is a Euclidean distance and must stay under half the reference's
      closest point pair (`saddle_graded`: 1.4e-5).
    - **B**: equal after a named transform, still at `1e-5` (index map, unit fix, projection,
      `lexsort`, sign or gauge fix).
    - **C**: a derived scalar, set distance or statistic. Must name the bug class it excludes and
      record a mutation probe and its margin (threshold at least 3x the measured agreement).
      `fraction_within` bounds must fail under shuffling one side.
    - **D**: exemption, only for: not an independent implementation (`oracle=` required); a
      different algorithm with a measured disagreement; a parameter the reference lacks; an answer
      not observable in isolation; stochastic with no invariant; inputs where ordito is undefined.
      Not admissible: "awkward", "tolerance would be loose", any class-B situation.
- Check 19 accepts four labels: `Class [ABCD]`, `Not a library comparison`,
  `Ordito against ordito`, `Not a parity assert`.
- Never a parity assert: shape-only or `isfinite`-only; ordito compared with itself; a threshold a
  constant output would pass. A boolean assert is parametrized over inputs giving both answers.
  Ordito-against-ordito is legitimate for pinning two entry points where only one has an oracle.
- Check the comparison is not vacuous on its fixture: assert the reference's answer is non-empty,
  has its expected count, and is not constant (`np.unique(labels_np).shape[0] > 1`,
  `np.ptp(reference) > 1e-3`).
- A guard test bites only if you identify which assertion fails under the mutation.
  Mutation-probe boundary predicates with exact ties.

### 7.5 Shared helpers: check both modules before writing a private one

- `tests/comparisons.py` holds the comparison helpers (`lexsort_rows`, `assert_unordered_rows_equal`,
  `canonical_labels`, `same_partition`, `fraction_within`, `symmetric_chamfer`, `chamfer_two_sided`,
  `hausdorff_two_sided`, `sparse_allclose`, and more); `tests/conversions.py` the converters for
  every library. List them before writing a private one.
- Compare a sparse operator with `sparse_allclose`, never densified `toarray()` comparisons.
- `symmetric_chamfer` takes meshes, `chamfer_two_sided` drawn clouds; prefer the mean form over
  Hausdorff for distributional claims.
- `lexsort` is unusable on float coordinates with ties. For positions use a `cKDTree` match with a
  bijection check, or `hausdorff_two_sided`; keep `lexsort_rows` for integer index rows.

### 7.6 The ten reference libraries

- All ten (trimesh, igl, potpourri3d, pymeshlab, open3d, pyvista, meshlib, pymeshfix, pytorch3d,
  manifold3d) are hard test dependencies: import them plainly, never via `pytest.importorskip`.
  Aliases are pinned in ruff (`import trimesh as tm`, `import igl`, `import potpourri3d as pp3d`,
  `import pymeshlab as ml`, `import open3d as o3d`, `import pyvista as pv`, `from meshlib import
  mrmeshpy as mm` / `mrmeshnumpy as mn`, `import pymeshfix`, `import pytorch3d.ops as p3d_ops`,
  `import manifold3d as m3`).
- Threading decides how a ratio reads: trimesh, igl, pyvista and pymeshfix are single-threaded;
  meshlib, manifold3d (TBB) and pytorch3d-cpu are multi-threaded; only pytorch3d-cuda runs on the
  GPU.
- `manifold3d` is the library `ordito.boolean` ports (Apache-2.0, code may derive from it): its
  answers are compared array for array, `to_mesh()` order included. Its wheel can round one
  `float32` ulp away from its own source; its CSG is lazy (time a row through `num_vert()`).
- Licensing: nothing under `ordito/` may name MeshLib or promesh (check 21), except the method
  value `"meshlib"`; pymeshfix and pymeshlab (GPL) may be named in prose, but no ordito code may
  derive from their source.
- Before writing a test or benchmark against a library, read its section in
  `reference/reference-libraries.md` (traps that fail silently).

### 7.7 Where the reference put the answer

A reference that seems to disagree more often had its result read from the wrong place, or was
handed something other than what you thought.

- Ask what the reference was handed and whether it finished the job before reading a ratio.
- A reference's warning is usually about our input. Read the reference's source at the warned
  line, solve for the input that makes it fire, then assert that property on the buffer we passed
  (for example `face_nondegenerate_mask`), not `-W error::RuntimeWarning`.

## 8. Tooling and local validation

Ruff and basedpyright (configured in `pyproject.toml`) are the authority for style and typing. Run
them and the tests after any change to `ordito/` and before considering work done.

```bash
uv run ruff format ordito tests       # 100 columns, skip-magic-trailing-comma
uv run ruff check --fix ordito tests
uv run basedpyright                   # gate stays at 0 errors
uv sync --all-groups                  # full environment; a bare `uv sync` uninstalls test deps
```

### Ruff

- Ruff is the only linter and formatter; do not override it inline. No blanket `# noqa` for a
  selected rule; prefer a scoped `per-file-ignores`. The one standing exception is §1.6's kernel
  accumulator (`# noqa: UP018, RUF046`).
- `reference/` is excluded from lint and typing; never edit vendored code to satisfy a tool.
- `F811` misses a duplicate whose first binding is used: after a module merge, scan for repeated
  top-level names and confirm the test count is unchanged.

### basedpyright

- `ordito/kernels/` is excluded: do not make kernels type-clean or add `# pyright: ignore` there.
- Config is `strict` less the five `reportUnknown*` rules. Read the config, not this prose.
- No local Warp stubs: typing gaps are closed in `ordito/typing.py` (`odt`: `Kernel`, `BsrMatrix`,
  `CsrMatrix`, `has_blocks`, `as_dense`, typed `odt.bsr_*` views); the rest take a `cast`. Call
  `odt.bsr_*`, never `wps.bsr_*`, on an ordito-typed matrix.
- Type variables: a union argument solves no `TypeVar`; a function returning its argument's own type
  uses a `TypeVar` bounded by `wp.array` or an overload set. A dtype parameter is
  `dtype: type[odt.Block] = wp.float32`. Never put `float32` / `float64` overload arms ahead of a
  generic one. Forwarded-only values are `object`, not `Any`.
- Rank `Any` is the one deliberate `Any`: a parameter takes the `Any`-ranked `odt.ArrayNd*` family,
  a return keeps the `Literal`-ranked `odt.Array1d*` / `Array2d*` aliases. Allocate through
  `_launch.empty` / `odt.empty_1d`, not bare `wp.empty`; narrow a slice with `odt.as_dense`.
- Order of preference: a correct annotation or `odt` helper, then `cast`, then a scoped
  `# pyright: ignore[rule]` with its reason (in `_launch.py`'s hot path an ignore beats a `cast`).
- `reportPossiblyUnboundVariable` is never suppressed: initialize to `None` before the branch and
  `assert x is not None` at the use.
- `reportUnusedParameter` covers `tests/` and `benchmarks/`: a protocol-signature parameter is
  `_`-prefixed; a `parametrize` value used only as a label goes in `pytest.param(..., id=)`.
- An error is almost always a real bug: resolve it, do not widen the disabled-rule list.
- `tests/` and `benchmarks/` are checked at the same bar (`tests.conversions.warp_empty`,
  `tests/typings/pymeshlab`, `assert odt.has_blocks(...)`; a private helper exercised on purpose
  takes a scoped `# pyright: ignore[reportPrivateUsage]`).

### Dependencies

- Test and reference dependencies live under `[dependency-groups] test`; use `uv add --group test <pkg>`.
- `plans/` is gitignored.

### Version control

- Commit directly on `main` unless the user asks for a branch. Committing is an explicit request:
  finish the work, run the gates, commit when asked.
- An A/B against a prior revision uses a detached worktree, never `git stash` and never a branch
  checkout in the live tree (§15).

### Coverage

- `ordito/kernels/` is omitted from coverage on purpose. Never write tests to raise a
  kernel-module figure, and do not raise the floor to chase `device.is_cuda` branches.
- Coverage gates wrapper branches only; it says nothing about vacuous comparisons (§7.4).

## 9. Performance work: measure before you change

- Something suddenly slow is a Warp rebuild until proven otherwise (§15).
- A benchmark lands before the optimization; never restructure for speed without one timing the
  current implementation.
- Before proposing an optimization, grep the call site, constants, benchmark docstring and
  `reference/measured-facts.md` for a written decline.
- Attribute a cost with one direct measurement at the benchmark's own operating point and at more
  than one size. A share that falls as the input grows is a decline.
- Check whether the function graph-captures before reading device time (§15).
- Attribute a change only with an interleaved A/B in one session, using a detached worktree. A
  probe that instruments what it measures carries a do-nothing control arm.
- CUDA is the target: decide on the CUDA number, measure both devices, and keep the CPU path
  correct. Split a tuning constant per device when the optima differ (§13.2).
- A decline is a result: write it at the site with the number, and resolve every site the finding
  named.
- Count the launches a numerical-method change adds per iteration.
- Host traffic is gated: `-p benchmarks.census` counts every ordito row's launches, allocations,
  memsets, copies, readbacks, scans and sorts, and CI fails when a CPU count rises above
  `benchmarks/census_budget_cpu.json`. A change that moves them re-records both budgets
  (`--census-update`; `benchmarks/README.md`) in the same commit. On a shared GPU, decide a
  candidate on these counts first and on a quiet-box A/B second; `--census-report` dumps each
  row's launch sequence for finding fusion candidates.

## 10. Warp API reference mirrors

Warp function lists are mirrored under `reference/warp_api/`: `builtins.md` (kernel-scope
builtins), `warp.md` (Python scope), `sparse.md`, `utils.md`, `fem_linalg.md`.

- Before using an unfamiliar Warp builtin, sparse or utils function, `grep` these files to confirm
  name, signature and scope. `uv run reference/warp_api/warp_version.py` compares the version
  stamps with the installed `warp-lang`; `REGENERATE.md` explains how to re-extract.
- A name in `dir(wp)` missing from the mirrors is usually hidden on purpose. Introspect first:
  `warp._src.context.builtin_functions[name]` exposes `.hidden`, `.doc` and `.input_types`.
- Then check the quantity (storage class, precision), not the name. Adoption verdicts: §12.6.

## 11. Running long commands: never poll with `until`

- Benchmark suites and probes run for minutes. Run the command in the background and read its own
  output file when it re-invokes you; do not write a wait loop around it.
- `until ! pgrep -f <name>; do ...; done` never terminates (`pgrep -f` matches the polling shell's
  own command line), and `pkill -f` kills the shell the same way.
- If a poll is unavoidable, test a sentinel file the process writes on exit and bound the iteration
  count.
- Do not run a timing probe while `pytest` or `zensical` is running.

---

# PART II — MEASURED FACTS (digest)

Each line is a verdict; the numbers, dates, sites and declines behind it are in
`reference/measured-facts.md` under the same section number. Read that section before acting.

---

## 12. The Warp platform: bugs, quirks and version status

### 12.1 Memory safety — every failure here is silent

- A `wp.launch` without `device=` corrupts the host heap (it resolves to `cuda:0` while CPU arrays are freed under the kernel).
- An out-of-bounds kernel write on CPU is glibc heap corruption; range-check at the write. Debug mode (-O0) drops FMA, so exact-float comparisons move.
- A zero-triangle `wp.Mesh` corrupts CUDA allocator state (error 700 on the next allocation). Never build one on CUDA, tests included.
- Python-scope gather ignores a non-contiguous index view's stride (§3.4); `wp.copy(..., count=0)` copies the whole source; `wp.copy` into pinned memory is async with no event (use `_common.read_scalar`). `.numpy()` is not a sync on a CPU array.
- CPU work is ~36x slower once CUDA is initialised in the process (§7.2).
- **Streams:** a capture on the legacy default stream fails (`unknown stream`), and a blocking stream's recording is broken by any thread's legacy-stream work, so ordito records on a private non-blocking stream (`_common.scoped_capture`). Warp's current stream is one field per device for the whole process, not per thread: two threads in Warp at once corrupt each other; a process-wide lock around every call is safe (`docs/streams.md`).

### 12.2 `wp.launch_tiled` runs one lane per block on the CPU

- On CPU `wp.block_dim()` reads 1. `wp.config.enable_cpu_blocks` runs every lane but is 1.5-340x slower: a test oracle (`--cpu-blocks`), not a production path.
- Probe a lane-constructed tile (`wp.tile(x)` from per-lane values), not `tile_load`, or you will delete a correctness branch.
- Warp has no grid-wide barrier, no per-cell `HashGrid` entry point and no builtin node-by-node BVH traversal; node reads take `wp.func_native` over `wp::BVH` (`kernels/proximity.py`; re-probe the layout at each upgrade).
- `wp.tile_bvh_query_aabb` overruns its shared result buffer on leaves holding more than 4 primitives (every builder makes a depth-limit node a leaf whatever its size), dropping primitives and faulting inside the builtin. Walk with it only a query that meets no such leaf (`kernels/proximity.flag_overfull_leaf`); the rest take a walk that cannot drop. Still bound-check every candidate.
- BVH `max_nodes` is a capacity and `num_nodes` is stale on a CUDA LBVH; read the count through `kernels/proximity.bvh_node_count`, never either field alone.
- A multi-launch `capture_while` body replays differently on CPU.

### 12.3 Numerics, casts and `wp.ref`

- Generic type-vars do not instantiate inside `wp.ref[...]`; `@wp.func` cannot be overloaded by name; `arr[i], arr[j] = ...` swaps raise. `wp.Scalar` does not instantiate for `wp.bool` (write the concrete bool kernel). Scalar arguments must match the input precision (`a.dtype(tol)`); literals in a generic `@wp.func` need `type(x)(...)`.
- FMA makes CPU and CUDA differ: never assume bit agreement, never substitute an algebraic identity inside a sign test that feeds a branch, and snap a distance within its own rounding to zero before branching on its sign (`predicates.plane_distances_d`). Do not disable FMA globally; a containment test that must not depend on the kernel around it is built unfused (`module_options={"fuse_fp": False}` on a `module="unique"` kernel), and a byte-identical or reference-matching rounding is pinned in a `wp.func_native` (`__fmaf_rn`, `__fmul_rn`).
- Exact orientation exists (`predicates.orient3d_sign` / `_f32` / `_from`, `orient2d_sign`): use it, not a tolerance, wherever a sign decides topology.
- Sum cross products about a point of the loop, never the origin (`predicates.newell_term`). Form area vectors and cotangents where they do not cancel (`predicates.triangle_area_vector`); Heron needs sorted sides; float32 lengths cannot represent a needle.
- A degeneracy tolerance carries the units of the quantity it bounds (relative, not an absolute `1e-12` on a length^4). Heat fields scale as 1/scale², so absolute tolerances on them are wrong.
- Gate an angle near pi on the angle (`predicates.vector_angle`), not on its `float32` cosine.
- `mesh_query_point_no_sign` is inexact (~2e-5; its face need not be the nearest). NaN breaks `searchsorted(side="right")`.
- Kernel-scope unary minus is `T(0.0) - x`: a negative zero is the literal `wp.float64(-0.0)`.
- `wp.svd3` on a `float32` covariance is ~6e-5 off on the middle eigenvalue; decompose as `wp.mat33d`. Moving `wp.svd3` into another kernel changes its CUDA result by ulps.
- **Float `atomic_add` and CUB's float scan are not run-to-run reproducible on CUDA**; integer ones, float `atomic_min` / `atomic_max`, `wp.utils.array_sum` and `bsr_mv` are. Hence §2.2's forms. NaN in `atomic_min` is device-dependent. A last-block ticket is taken on an `int32`, or on one component of a vector counter.
- Constant-cast and module-constant accumulator rewrites: §1.6.
- At Python scope `wp.int32` arithmetic costs ~10 µs and `//` raises; unwrap with `int()` (check 26).

### 12.4 Compilation, module hashing and import cost

- A generic kernel's lazy overload rebuilds its whole module; `wp.map` forks per call signature; Warp's own generic kernels do too (`array_cast`, `warp.sparse`): use `array.copyto` or `kernels/linalg.register_warp_overload`.
- `@wp.kernel` builds an `Adjoint` at import, so the lazy `__init__` is load-bearing; another module's top-level `import warp.fem` cancels a deferral (import it inside the function).
- Backward passes dominate cold compile (§2.6). Only host readbacks block graph capture.
- A tile `shape=` must be a plain integer; wrap typed constants in `int()` on the host.
- A launch's `block_dim` is a compile-time parameter of its module: a second width rebuilds the whole module, so a kernel launched at its own width lives in its own module (§2.10).
- **NVRTC 12.9 (`+cu12` Warp) miscompiles some `@wp.struct` kernel parameters for sm_120** (CUDA error 700). CI runs `cu13` and cannot see it. The two Poisson iso-value term sums take `kernels/reduce.NVRTC12_STRUCT_TERM_OPTIONS`; after touching a struct-parameter kernel, run the PTX scan described in the reference file.
- **MathDx objects live in a cache no toolkit keys**: a kernel linking `tile_cholesky` / `tile_lower_solve` / `tile_matmul` lives in a module of its own and loads through `_common.load_mathdx_kernel`.
- A cold CPU build of all kernel modules is ~200 s on 1.19 (vectorization); `cpu_compiler_flags` is per module and changes the hash.
- `ordito.precompile` builds ahead of time with no GPU (`wp.compile_aot_module`), one module default `block_dim` at a time; details in the reference file.

### 12.5 `warp.sparse` and `wp.Volume`

`nnz` is a capacity (§3.7).

- `bsr_mm` returns a structural superset (explicit zeros) that inflates chained products; `bsr_mm`, `bsr_axpy` and `bsr_set_transpose` read the `nnz` field.
- `bsr_mv` takes one vector. Unwritten `(0, 0, 0.0)` triplets all accumulate on one entry (9-32x): point them out of range.
- `TiledDot` takes an O(n) per-block reduction under `batch_offsets` with `batch_count > 1`; `linalg._BatchedCg` avoids it.
- `cg` at `check_every=0` records a graph per call; it resolves an omitted `atol` to `tol`, so a tiny `||b||` returns zero iterations. Always pass `atol=0.0`.
- `allocate_by_voxels` and `Nanogrid` work on CPU; an empty point set raises. A one-`atomic_cas` table beats `allocate_by_voxels` 1.3-3.2x.
- `get_voxels()` order is leaf-major: lexsort before comparing. NanoVDB centres voxels on integers: pass `translation = origin + 0.5 * voxel_size` or cells are off by one.
- `get_voxel_count()` is a capacity; use `get_active_stats().voxel_count`. `rebuild()` does not pay.

### 12.6 Warp builtins: adoption verdicts

- **Adopted:** `mesh_query_point_sign_winding_number` (needs `support_winding_number=True`, else it silently returns parity; O(faces) a query on long thin triangles); `bvh_query_sphere` (not for a radius equal to the grid cell); BVH leaf size 1 for ball queries; `mesh_get_bvh`; `volume_index_to_world`; `delaunay_edge_flip` above 2**19 points; `sparse_marching_cubes` above a lattice-size gate on closed input; `wp.Mesh.points` assignment as a refit (`Trimesh.with_vertices(refit=True)`).
- **Rejected:** `intersect_tri_tri` (not scale-invariant); `closest_point_edge_edge` (float32, worse); `sample_unit_hemisphere_surface`; `norm_huber`; `tile_arange`; `volume_voxel_count` (capacity); `dense_chol` family; `tri_tri_adjacency` for `halfedge_twins` (test oracle only); `swept_volume_mesh`.

### 12.7 Upgrade discipline

The bugs above (zero-triangle mesh, CPU one-lane tiles, `tile_bvh` overrun, `wp.ref` generics, `bsr_mm` superset, stale `nnz`) all persist on 1.19; re-probe them and every tuning constant at each upgrade (the `upgrade-warp` skill). Verification defaults to a false pass: `compute-sanitizer` that instrumented nothing prints 0 errors (expect ~10x slowdown), and `capture_while` sites skip the graph path when `wp.is_conditional_graph_supported()` is `False`. Gates: full suite, `basedpyright` 0 errors, `zensical build --strict`, unchanged `tests.parity` pair count.

---

## 13. The cost model (RTX 5090)

### 13.1 Host-side, per call

Measure `n` calls between two syncs and divide; a sync inside the loop drains the mempool (up to 14x wrong). The lever is allocating less; a cache holding memory across calls is rejected.

| primitive | cost |
|---|---|
| `wp.launch`, flat in `dim` | ~12 µs (`_launch.launch` 4.5-6 µs) |
| each launch argument | ~1 µs; generic kernel +12 µs |
| `wp.empty` / `wp.zeros` / `wp.copy` | 6 / 9 / 4.5 µs |
| `array_cast` / `array_sum` | 21 µs / 40 µs (avoid) |
| cached `wp.map` | ~11 µs above the kernel |
| readback | ~0.1 ms queued, 14 µs isolated |
| replayed graph kernel | 1.2 µs |
| small `.numpy()` / `_common.read_values` | ~27 µs / ~9 µs |
| `wp.array` slice / `_launch.view` | 4.2 / 1.05 µs |
| small-pattern native fill (`_launch.pattern`) / 2-element upload | 8.2 / 11.7 µs |

- Warp-typed values at Python scope are slow (`wp.int32` arithmetic, Warp-typed slice bounds, `wp.length`, vector arithmetic: 4-40 µs against under 1 µs in plain Python). Unwrap with `int()` / `float()` (check 26).
- A device reduction costs 0.1-0.3 ms flat; a readback wins below ~200 k `int32` elements (1 M `bool`).
- Small constant tables cost more passed by value as vector arguments than uploaded. The creation builders are host-bound (NumPy / Python per-op overhead).

### 13.2 Device-side access and tuning constants

- `wp.tile_sum(wp.tile(x))[0]` is a block-wide barrier (~0.13 µs at 32 lanes, ~0.35 µs at 64-128).
- A single-address atomic serializes the launch: lane partition, `wp.tile_sum`, one guarded commit per block at `kernel_reduce.blocks_1d(n)`. Do not convert a conditional atomic.
- A CG round's cost is its block count, not its mat-vec (`CG_CSR_TARGET_BLOCKS = 512`, `CG_TARGET_BLOCKS = 2048` for the `float32` Poisson grids); re-probe at an upgrade.
- An append-only list whose order decides nothing takes `kernels/array.append_slot`; a read before `atomic_min` (`array.atomic_min_unless_below`) pays only where many threads hit one slot.
- **`float64` runs at 1/64 of the `float32` rate**: put a `float32` filter with a rigorous bound ahead of a latency-bound chain (`predicates.orient3d_filtered_f32`); take constant divisions out of a wide integrand's loop (`reduce.store_folded_sum`); table a long row's `float64` transcendentals across the device first.
- A fused term sum's fold width follows its term's latency: a dependent-load term takes a narrower `TermSum` (`tiles_per_block`), a fold over CSR rows `kernels/reduce.spanned_blocks_1d`.
- A tile kernel whose reduced extent is under one tile loses ~49x; clamp with `reduce.block_chunk`.
- Pack same-dtype quantities into one vector tile (`preserve_type=True`) for one barrier.
- Tuning constants are per-device (CUDA long strided slices, CPU short: `ITEMS_PER_SLICE_CUDA = 128`, `_CPU = 32`). Sweep both devices and more than two values.
- Reproducibility costs 2-15 µs a launch or scan: over 10 % of a 15-100 µs call, which is why the setting can be switched off.

---

## 14. Kernel-shape verdicts

Each bullet names a shape that was built and priced. Grep `reference/measured-facts.md` §14 for
the helper or site named before building anything similar.

**Occupancy and launch shape**

- **Block-per-item** wins only when the outer dimension alone starves the device (§2.3; `obscurance`, `shape_diameter`, `farthest_point_sample`). A one-walk least tangent radius was refuted.
- **Register-blocking outer items** wins for `(item, slice)` reductions; choose W per launch so the grid keeps about `1 << 17` threads (`RegisterBlockedTable.slice_count`); a target past a kernel's optimum loses.
- A `float64` rank-k update register-blocks only on wide launches, in narrow (32-lane) blocks (`kernels/cholesky.update_panel_tiles`).
- **Pointer-chasing tree walks** run faster with fewer warps resident, gated on query count (`kernels/neighbors.DESCENT_WIDE_BLOCK_DIM`); check a new walk at two widths, and time it on the device (5x by loop spelling).
- **Cooperative BVH walks** pay only with few concurrent queries (`ball_pivoting`'s pivot search); do not re-run the tree-wide sweep.
- A lone serial hash-grid walk splits across a warp (`kernels/neighbors.warp_grid_chunk`) only where few walks run; many short walks side by side lose.
- A walk with costly candidates leaves its traversal loop with one candidate a lane (`kernels/proximity.nearest_offset_within`).
- A counted-then-written walk with costly candidates decides each candidate once, a thread a candidate (`kernels/boolean.decide_edge_candidates`); where the walk is the cost it stages instead (§2.3), declined where the walk is no share of the call.
- A per-point `float64` neighbourhood chain runs a warp a point, a neighbour a lane, its serial part (`svd3`) in a pass of its own.
- A one-thread serial walk of dependent loads on CUDA data runs on the CPU device (`boolean`'s collapses, long polygons); move only used prefixes.
- Independent per-item iterations collapse into one in-thread loop (`max_tangent_sphere`), kept out of a kernel holding another long walk.
- **Tile solves** win for K >= 16-32 only; the dense K = 5, 6 solves stay.
- **Shapes that lose:** a line-per-thread lattice pass absorbing a cell-per-thread one; a serial walk storing gathered values instead of indices; a per-face edge marker instead of a per-edge one.

**Host loops, capture and readbacks**

- **Graph capture** wins when a launch sequence repeats identically (break-even ~2 replayed rounds), loses on once-through and short loops. The body may not allocate. Replays past a fixpoint cost full device time: replay one round a read unless each launch returns first on an idle flag. A captured path needs a plain-loop CPU sibling.
- **A CPU round loop runs as one launch**: each round's body is a `@wp.func`, and a CPU-only kernel runs every round in one thread. Declined for the Cholesky triangular solve per level and `halfedge_mates` pairing.
- **Readbacks:** a device reduction wins on CUDA for mesh-sized arrays and loses on CPU; keep the readback for bool masks and small buffers. A per-pass readback is cheaper than an extra pass. Read a validation flag as early as its producer allows. In idempotent loops read the stop count every few rounds.
- **Per-device algorithm choice** only when the parallel form does asymptotically more work.
- **Producer-consumer fusion:** always fuse consecutive same-`dim` launches where the consumer reads the producer only at its own index. Price the fused region; a fusion that changes a slot or box index needs a debug-mode run. Fold-on-read loses on dependent chases.
- **Gather-form fusion** wins below ~100 k faces and loses 1.7-2.1x on scrambled ~1 M-face meshes: price it on a scrambled large mesh.
- Two block reductions of different widths share one launch at `dim = max(blocks)`.

**Hub rows and long rings** (`fan_hub` is the fixture)

- A row-a-thread solver step takes long rows (`kernels/linalg.CSR_LONG_ROW_ENTRIES`) by pieces (`linalg._LongRows`, `CSR_LONG_ROW_PIECE`), by deferring them to its block (`csr_row_long`, no list or readback), or, where there is no readback to carry a flag, as a warp (`kernels/smoothing.warp_long_operator_row`, `reduce.warp_walk`). A heavy row body stays inline in the kernel, not behind a `@wp.func`.
- A one-block pass lets the warp walk long rows only past a gate (`conjugate_gradient.ONE_BLOCK_WARP_ROWS_FROM`).
- A long linked walk ranks by rulers in a block (`kernels/halfedge.rank_long_rings`); a ring kernel takes long rings from the list `vertex_one_rings` read back (`_common.note_long_rings`).
- A sort that only groups rows under a small key is a ring compare (`repair._exact_value_classes`); a hub sends the call back to the sort, and the CPU keeps its sort for duplicated faces.

**Latency-bound kernels**

- A latency-bound row loads branch-free at clamped addresses (`kernels/array.row_chunk_slots` / `row_chunk_entries`); it loses on wide bandwidth-bound launches and on ~6-entry operator rows below `2**18` rows; the CPU keeps the entry-a-step walk, bound per kernel. Price against a do-nothing kernel first.
- A thread a row writing its segment in turn collapses past L2: a thread a slot (`kernels/array.outer_product_slot`) on CUDA, the row walk on CPU.
- A row gather in a triplet sort's order loses to the regrouped sum (`energies.*_values`).
- A one-block round loop costs its candidates, not its ring (`ear_clip_tail`).
- A relaxation to a fixpoint takes block rounds only over spatial patches (`_common.hilbert_point_order`).
- An exact maximum over a box query prunes by a bound every candidate attains (`kernels/polyline.ec_tree_ear_cost`).
- **Tree searches take scattered queries in Hilbert order** (`_common.search_order`, `kernels/array.ordered_item`) from 2^18 queries (2^16 when the sampled test finds them scrambled); answers unchanged. Rays take it only when their origins sit on the surface; declined in `ray`.
- A CPU skip in front of a few atomics commits a selected zero instead (select, never multiply by a 0/1 weight).

**Solvers and sweeps**

- A single-level polynomial preconditioner wins on long solves; a Chebyshev multigrid smoother loses. A small synthetic system can mislead.
- The factored triangular solve does not run in one block on CUDA, nor behind a device-wide spin barrier; the CPU runs it as one serial launch.
- **Blocked wavefront** wins for a 2-D DP with local dependencies and small per-step work; it loses where levels are already wide (the fill DP).
- **Serial-order sweeps as a dataflow wavefront** (Gauss-Seidel, `filter_laplacian(method="pyvista")`): one launch a wave, bit-identical to the serial loop; batched waves per readback; capturing the batch lost; the volume-constrained path counts its waves once. The CPU runs the serial loop.
- **Refuted, do not re-propose:** one launch a Jacobi CG round; a single-block BFS drain; a one-block hole-fill DP; merged-level blocked interval DPs; voxel aggregation for multigrid; micro-tuned Bridson blue noise; two-stream overlap of sequential branches; `cg(check_every=0)` as a lever.

**Sorting, counting, scatters**

- An order statistic is a radix select on CUDA (`kernels/reduce.RADIX_SELECT_PASS`), size-gated; the CPU sorts.
- A count of distinct edges is an edge set homed at the lower vertex (`grouping.edge_key_slot`), not a sort.
- A crowded count histogram pools a warp's peers (`reduce.peer_counted_add`).
- **Per-class compaction:** scan a one-hot `wp.vec{K}i` (or a `vec2i` of `(flag, count)`); `_launch.array_scan` routes `int32` vectors of 3+ components to a tile scan where it wins.
- **Fixed-order group sums:** never a stable sort plus one thread per run on huge groups; chunk long runs or use the exact fixed-point sum. The shared exact group sum wins on few hot groups and loses on many; the sites that keep their own form (exact `float32` into `float64` atomics, `arap`'s sorted run walk, the Poisson chunked splat, the cluster heavy-run sum) carry the decline.
- Off-mode pooling: peers only on keys hot across the launch (§2.2). Per-vertex float scatters pool a warp's peers (`kernels/scatter.peer_summed_add*`) only where the kernel's work is the scatter; edge slots lose.
- A corner scatter reads its three corners off the flat `faces` buffer, never by a loop over an `(n, k)` view with a dynamic trip count (CPU 1.45-2.1x).
- A kernel whose work is the scatter keeps its on-mode terms out of line on the CPU (`kernels/scatter.sole_scatter_add*`); not for a kernel with work of its own.
- Bellman-Ford over a CSR with duplicate entries pays per stored entry.
- **Rasterization:** one `atomic_min` owner pass per face (sliced over lanes on CUDA), then one thread per pixel; the kernels build unfused.
- **CPU on Warp 1.19:** SLP vectorization took CPU rows to a 0.88 geomean; a module-wide flag to undo its few losses (MLS curvature) was declined.

## 15. Benchmark and measurement traps

- **A sudden slowdown is a Warp rebuild until proven otherwise:** look for `Module hash changed, recompiling` at `LOG_DEBUG` and an idle GPU (§2.5, §3.5). Next suspect host-side per-element Python.
- **Attribute against one number:** time the candidate directly (subtraction projections are optimistic by 3-10x), at more than one size and at the benchmark's own parameter and fixture. Compute the candidate's share before reading a ratio near 1.0; pin the iteration count before timing an iterative solver.
- **A/B:** a detached worktree for the B arm, never `git stash`, `sed` sweeps or the live tree (another session may edit it). The editable finder outranks `PYTHONPATH`: drop it and assert the submodule `__file__`. Interleave A and B in one loop, report the minimum beside the median, batch launches and sync once.
- **The GPU is shared:** check `nvidia-smi` before and after every timing; if busy, measure counts, not clocks. A CUDA error 700 under memory pressure from another job is pressure until it reproduces on an idle GPU.
- **Harness hazards:** cap every group's mesh size first. Harness and hand-probe numbers are not comparable; decide on the harness number. A median at `rounds=3` can hide a one-off cost. Run `benchmarks/test_meshes.py` after touching the mesh registry.
- **A row that moves only inside a whole-module sweep depends on what ran before it**: rerun it alone before attributing it.
- **CPU confounds:** a kernel's code placement (its address modulo 64 moves with any edit to its module) and the heap's history (count `ru_minflt` per call in both arms).
- **Generated kernel variants need their own `@wp.func` names** (they otherwise all compile against the last one defined), and a do-nothing control from the same generator.
- **`wp.timing_begin` sees no graph-replayed kernels** (re-run with capture disabled) and `TIMING_KERNEL` omits memcpys (`TIMING_ALL` to price a fusion replacing a `wp.copy`).
- **Census:** counts come from monkeypatching Warp, not grepping. Every group has a census row on `sphere_small` (`CENSUS_MESH`) or the mesh `@pytest.mark.censusmesh(name)` names; `censuskeep` / `censusskip` adjust the size axis; a row that skips leaves its group outside the gate. A count must not depend on what ran before the row: a new process-wide cache that changes what a call issues is checked by running the row alone and inside its file.

---

## 16. Open items

Per-component facts, design choices and measured declines live in `reference/component-status.md`;
read the component's section there before optimizing or diagnosing it. This section lists only
what is still open. Re-measure before acting on any of it, and delete an item when it closes.

- **Mesh visuals, atlas packing** (2026-11-24): `Trimesh.__add__` of two meshes with different
  materials drops the material with a warning where trimesh packs the images into one atlas.
- **A Class C pair an option could make Class A**, by porting the reference's algorithm:
  `offset_mesh(method="meshlib")` against MeshLib's `offsetMesh`, whose extraction is ported
  (`marching_cubes(method="openvdb")`, Class A) and whose OpenVDB `meshToLevelSet` band field is
  not. Scoped, with source availability and licensing, in `reference/component-status.md` (16.0b).
