# Role: NVIDIA Warp Expert

You are an expert in NVIDIA Warp (`wp`) working on **ordito**, a GPU geometry-processing library.
Follow every rule below when writing kernels, `@wp.func` helpers, Python-scope wrappers, tests and
benchmarks.

- **Part I (§1-§11)** — rules: how code in this repository must be written.
- **Part II (§12-§16)** — measured facts on this hardware (RTX 5090, warp-lang 1.18.0): platform
  bugs, cost model, kernel-shape verdicts, measurement traps, component status. Read the relevant
  section before proposing an optimization or diagnosing a slowdown; most plausible ideas here have
  already been built and priced, and several lost.
- Deeper material lives outside this file and is read on demand: `reference/reference-libraries.md`
  (traps of each test/benchmark reference library) and `reference/component-status.md` (per-component
  measured facts and refuted plans).
- Record a new measured finding in the section that owns it (Part II here, or the reference files);
  a rule that follows from it goes into Part I. A measured decline is a result: do not re-propose it
  without new evidence.
- A Warp version claim is spelled `Warp 1.18` (the word "Warp" immediately before the number); the
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
- Do not call `wp.constant()` (§12.4). The typed constructor is what matters: `TOLERANCE_MERGE = wp.float32(1e-8)`. A plain Python float is `float32` in kernels and mixing it with `float64` is a parse error; use `wp.float64(...)`.
- Prefer dtype-generic `@wp.func`s (`wp.Float` / `wp.Scalar`, `Any` for vectors; the `kernels/predicates.py` convention). Limits: §12.3. `Any` is also generic over rank and dimension, so do not pin a helper to one; use rank-free reductions (`wp.min(wp.abs(wp.get_diag(r))) < tol`, since a matrix has no `.shape` in kernels).

### 1.3 Casts and conversions

- Cast `wp.tid()` when used as an index: `f = wp.int32(wp.tid())` (check 22; multi-index unpacks cannot carry a cast).
- `wp.int32` / `wp.float32` are the only cast spelling; never bare `int(...)` / `float(...)` (check 16). `float(...)` fails inside `wp.Float`-generic functions; use `type(x)(...)` where a function is or could be generic (§12.3).
- A cast to the type a value already has is noise; delete it. This includes same-type constructors such as `wp.vec3(*(a - b))`, which check 16 does not see. The tid cast is the one kept exception.
- A cast of a bare literal is never redundant: `wp.int32(0)` is mutable, `0` is a constant (§1.6).
- `wp.cast(expr, T)` reinterprets bits between same-size types only; widen or narrow with the constructor. Whole-array conversion: §3.6.

### 1.4 Kernel-scope restrictions

Not supported inside kernels and `@wp.func`s: lambdas, comprehensions, sets, dicts, `list.append()`, `eval()`, recursion, exceptions, Python tuples for initialization (use `wp.vec3(1.0, 2.0, 3.0)`).

- Small fixed collections are vector types; larger ones are `wp.types.vector(length=N)`, not `wp.zeros(shape=N, ...)` (§2.9).
- A tuple cannot be indexed at runtime; a vector can (`wp.vec3i(*corner_triple(faces, f))`). Prefer an index form to a slice.
- A variable defined only inside an `if` branch is uninitialized when the branch is skipped; initialize first.

### 1.5 Arithmetic and conditional spellings

- `%` follows C++11 (sign of the dividend). Since Warp 1.18 integer `//` floors while `/` and `%` truncate, so they differ on a negative dividend. Check the dividend's sign before relying on either.
- Spell integer division `//`, never `/` (check 17; the scan types operands by declaration only). Do not respell `//` as `/` for speed.
- A conditional value is `wp.where(cond, a, b)`, not a ternary (check 20). It evaluates both arms; use an `if` when one must not run.

Checks 16, 17, 18, 20 and 22 are one family: every spelling is legal and generates identical code, so only a scan holds the line.

### 1.6 Mutable loop accumulators

A variable initialized to a bare literal (`count = 0.0`, `n = 0`) is a compile-time constant and cannot be mutated in a dynamic loop (an error, or silently the initial value). Declare accumulators with the typed constructor: `count = wp.float32(0.0)`, `n = wp.int32(0)`.

Never let `ruff check --fix` strip the constructor (it rewrites `int(0)` to `0`); legacy spellings carry `# noqa: UP018, RUF046`.

---

## 2. Kernel architecture and launches

### 2.1 Launch discipline

- `wp.launch(kernel=..., dim=..., inputs=[...], device=...)`. Always forward `device` from the input arrays: a memory-safety rule (§3.9, check 15).
- Prefix output arguments `out_` and put them after all inputs (check 13). Exempt (`_KERNEL_OUTPUT_ALLOWLIST`): in-place arguments and scratch / persistent-state buffers, which are named for what they hold. A read-only input never wears `out_`.
- Derive the face count as `f = faces.shape[0] // 3`.

### 2.2 A lane-parallel kernel strides by `wp.block_dim()`, or it stays lane-free

A kernel whose lanes cooperate (`wp.tile_sum` / `tile_min` / `tile_max` per lane, a `wp.tile_bvh_query_aabb` walk) is correct on both devices only when its lanes partition a sequence the block already owns, with the stride taken from `wp.block_dim()`, never from a kernel argument or constant.

- `launch_tiled` runs one lane per block on CPU, where `wp.block_dim()` reads 1; a one-element tile is not a bug, an arg-strided sum is. Do not conclude tiles do not work on CPU (§12.2).
- Where lanes would partition the outer work (a whole-array reduction with no per-item dimension, e.g. `metrics.chamfer_*_tiled`), keep a constant stride on CUDA only with a lane-free `_sliced` sibling for CPU (`_device.prefers_tiled_reduction`), or stay lane-free on both.
- For a lane-strided reduction, tune the fold width: launch `dim=kernel_reduce.blocks_1d(n)` so each block owns `ITEMS_PER_BLOCK_1D` elements, never `dim=n / TILE_1D`. Assemble a wide accumulator before the single commit (§13.2).
- A float block commit goes through `kernels/reduce.store_block_sum` / `store_block_total` into a `_launch.block_partials` buffer, never a bare per-block `atomic_add` (arrival order, §12.3); integer commits keep `commit_block_*`. A float scan goes through `_launch.array_scan`. Both read the global setting `ordito.config` (`set_deterministic`, the scoped `with deterministic(True):`; off by default in the library and the benchmarks, on for every test via `tests/conftest.py`) at the call, so a new reduction or float scan obeys it by routing through them. The same flag fixes output order (an atomic-slot fill becomes a stable sort); there is no separate order setting. Never add a per-call `deterministic` keyword or thread the mode through signatures; a caller that reads more than the result slots tells the two buffer forms apart by the buffer's size. Benchmarks run with the setting off for the whole session (`benchmarks/conftest.py`, `benchmarks/README.md`).
- A new per-group float sum goes through `kernels/scatter.scatter_add` / `scatter_add_vector` / `scatter_add_matrix` with a `_launch.group_sums` buffer (`launch_group_sums`, `finish_group_sums`): the buffer picks the mode, a plain `atomic_add` with the setting off and the exact fixed-point sum of `kernels/reduce` with it on (identical on CPU and CUDA). A kernel that consumes the sums reads them through `kernels/reduce.read_group_sum` from `group_sums(fused=True)` (no finish launch and no result buffer when on; repair's component sums). The public forms are `reduce.scatter_sum` and `reduce.segment_sum`. Keep a site's existing form where §14 prices it faster.
- A per-vertex float scatter (`add_corner_triple`) obeys the same setting. `float32` values sum exactly into `float64` whatever the order (`scatter_sum_exact`, `scatter_face_areas_exact`); `float64` values are written per corner and summed by `kernels/scatter.sum_sorted_corner_values` over one stable sort of the faces, with the atomic scatter kept for the off mode. `vertex_face_adjacency` obeys it too (ascending rows from a stable corner sort; the atomic scatter when off), so a consumer emitting triplets in its row order is reproducible through it.
- **A reproducibility fix is gated on the setting, never unconditional.** With it off (the default) the code is the fast arrival-order path it was: the same kernels, launches, allocations and buffer dtypes, non-reproducible by design. With it on, the output is bit-identical run to run. "Costs nothing measurable" or "exact anyway" does not justify making a fix unconditional. A reproducibility test claims the on mode only (`tests/conftest.py` sets it).

### 2.3 Block-per-item is an occupancy trade, not a shape choice

Use one block per item only when the outer per-item dimension alone would starve the device. A kernel that already has a slice dimension does not qualify; collapsing it into lanes loses 2-8x. Compute the launch's thread count and measure at the large end first, since the small end flatters. Verdicts: §14.

### 2.4 Fusing a kernel: extract the shared part as a `@wp.func` in the same commit

A fusion is done when the code it duplicated has a name. Whenever a kernel is fused, split, specialised or given a second variant, the same commit must:

1. Name each shared statement run (corner load, window computation, emit protocol, guard sequence) as one `@wp.func` both kernels call.
2. Put it where the quantity lives: predicates in `kernels/predicates.py`, per-face quantities in `kernels/triangles.py`, index/sort/search helpers in `kernels/array.py`, scatters in `kernels/scatter.py` (§3.1).
3. Say in a comment what the kernels still differ by. Prefer one kernel with a warp-uniform int selector (§2.7, the `ACCEL_HASHGRID` / `ACCEL_BVH` pattern) when the difference is a parameter; keep two kernels naming each other when they are two algorithms.

Further rules:

- A comment naming the other copy ("mirrors X") is the finding: extract a shared `@wp.func`.
- A duplicated decision rule is a correctness hazard even when the control flow differs; extract it as a `@wp.func` returning the classification (a sentinel for "reject").
- Factor the family, not the pair: write the generic form and check the other rank and precision.
- One predicate, one spelling per module: `wp.length(d) < r` and `wp.length_sq(d) < r * r` differ in float32.
- A green suite does not prove an extraction was behaviour-neutral; read the diff and gate on the decision arrays a kernel writes.

### 2.5 A generic kernel registers its overloads at import (`wp.overload`)

A `@wp.kernel` generic over a dtype (`wp.Scalar`, `wp.Float`, `wp.Int`, `Any`) must register its concrete overloads at import, in a `_register_overloads()` at the bottom of the file. Otherwise the first launch at each new dtype recompiles every kernel in the module, silently.

- Register what the wrapper's dispatch can reach; for independent generic arguments register the cross product, not a diagonal.
- Registration is not compilation: never call `wp.load_module` / `wp.force_load` at import.
- `test_generic_kernels_register_their_overloads` only checks that some overload exists; a missing dtype shows up as a slowdown (§15).
- `wp.map` has the same chain: §3.5 (check 23).
- Keep the `wp.Kernel` that `wp.overload` returns and launch through it. A module builds a dtype-keyed [`OverloadTable`][ordito.kernels.array.OverloadTable] and the wrapper writes `kernel_laplacian.COTMATRIX_TRIPLETS[cot_entries.dtype, dtype]`. A new generic kernel adds a table, not a bare `wp.overload` call.

### 2.6 Backward passes, and in-place `@wp.func` parameters (`wp.ref[T]`)

Only `ordito/kernels/metrics.py` is differentiated. Every other kernel module sets `wp.set_module_options({"enable_backward": False})` right after its imports; a new module copies the line (`test_only_the_taped_kernel_module_compiles_backward_passes`).

- `@wp.func` helpers may take `wp.ref[T]` parameters to mutate caller storage (`update_argmin` in `kernels/array.py`); `T` must be concrete (§12.3).
- Any kernel calling a `wp.ref` helper must be `@wp.kernel(enable_backward=False)`; the module option is not consulted at parse time.
- Never use `wp.ref` in `kernels/metrics.py`.

### 2.7 Function-valued parameters and kernel factories

- A `@wp.func` may take `fn: wp.Function`, bound at compile time per call site (user `@wp.func`s and simple builtins, not tile intrinsics).
- `wp.launch` cannot pass a `wp.Function`. For runtime selection pass an int and branch in a dispatch `@wp.func` (`registration.robust_weight`).
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

- Do not bundle a kernel whose launch is captured (§13.1).
- After introducing a struct, re-check static checks that match AST node types (check 13, §4.5).

### 2.9 Per-thread row storage

`wp.zeros(shape=K, dtype=T)` in a kernel is not register storage (it is pointer-addressed, 2x slower). Use `wp.types.vector(length=K, dtype=T)` (the `vec5d` pattern in `kernels/curvature.py`), which wins up to K near 64 before spilling.

- Never pass the vector to a `@wp.func` (it spills); inline it per call site.
- Any runtime index spills it: read the k-th element with an unrolled `for slot in range(K): if slot == k - 1:`, closing over `K` in a factory (one kernel per bucket, unique `name`, `KNN_ROW_BUCKETS`).
- A register insertion sort needs an explicit `placed` flag; carrying the displaced element drops a neighbour on equal distances.

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
- The radix sort is stable under both deterministic settings, so with an identity payload each
  class's first occurrence sits at the start of its run: a grouping that needs representatives
  reads them off the emit pass (`grouping._unique_keys_core`), not `first_occurrence_indices`.
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
  counts (`.size` is rows x cols), and for rank-free or `Any` operands.
- Always pass `device=` to every allocation, including in `_*.py` (check 10).
- Size buffers for their final use at allocation; never allocate-then-grow (the producer allocates
  the `n + 1` terminated form, `counts_to_offsets`).
- Warp raises on a zero-length slice; guard a trailing `fill_` with `if stop > start`.
- A buffer whose initial value matters is allocated holding it (`wp.zeros`,
  `wp.full(n, value, dtype=..., device=...)`, at rank 2
  `odt.as_array2d(wp.full((rows, cols), value, ...), dtype)`), never `wp.empty` then
  `fill_` / `zero_`. `wp.empty` is right where every element is written before it is read.

### 3.4 Python-scope gather indexing (prefer over trivial gather kernels)

`src[indices]` yields a `wp.indexedarray`; materialize it with `wp.copy(dst, view)` when callers
need `.reshape()` or a dense `wp.array`. Do not add per-element gather kernels when this suffices.

- Index arrays must be 1D: flatten, gather, `wp.copy`, `.reshape(shape)`.
- The index array must be contiguous: Warp silently ignores a view's stride, so
  `payload[edges[:, 0]]` returns the flat buffer's leading entries. A prefix slice (`arr[:n]`) is
  safe; a column or step slice must be `wp.copy`'d to a dense buffer first, and a prefix slice must
  not be cloned. This is why `kernels/edges.py:edge_lengths` stays a kernel (§12.1).
- When converting a gather, verify values: the corrupt version is faster.
- Indexed assignment is unsupported; keep a small kernel (`mark_membership_mask`).

### 3.5 Elementwise ops at Python scope (`wp.map`)

Do not write a `@wp.kernel` whose body is only `out[i] = f(in[i], ...)`. Keep the op as a named
`@wp.func` in `kernels/` (or a builtin such as `wp.neg`, `wp.normalize`) and call
`wp.map(op, *inputs, out=...)`.

- Named `@wp.func` only, never lambdas (the cache keys on the unqualified name plus dtypes).
- Always pass `out=` (in-place: `out=<an input>`; multi-output: `out=[a, b]`).
- `wp.map` infers a bare Python `int` as `wp.int32`: cast explicitly, e.g. `wp.uint64(bvh.id)`.
- In per-iteration loops, hoist the kernel with `wp.map(..., return_kernel=True)` and `wp.launch`
  it (`smoothing.py`; §13.1).
- An op reached at more than one call signature must declare those signatures at import
  (check 23), because each signature forks the generated `map_<op>` module and recompiles it (as in
  §2.5). Tables live in `_declare_map_kernels()` at the bottom of the owning module; Warp builtins
  are declared in `kernels/array.py` (`declare_map_signatures`). Declare with
  `return_kernel=True` on a zero-length CPU array (this also declares the CUDA overload).
- A length-1 array, an `indexedarray` and the rank also fork the module, not only dtype; derive a
  table by instrumenting `wp.map` over a suite run (check 23 asserts only that one exists).
- Never declare via `ordito/__init__.py`, `wp.load_module` or `wp.force_load`.
- Still a real kernel: thread index as data, whole-array arguments, scatters, row-indexed outputs.

### 3.6 Dtype conversion at Python scope

`wp.cast` is kernel scope only. Convert arrays with `od.array.astype(src, dtype)` (new array) or
`od.array.copyto(dst, src)` (into a buffer you hold).
Never call `wp.utils.array_cast` from `ordito/` (check 30): its generic kernel lives in Warp's own
module, so every new dtype pair rebuilds that module (§12.4). A kernel created lazily must go in its
own module (`wp.kernel(..., module="unique")`), or it recompiles every kernel in the existing one.

### 3.7 Sparse: assemble from keys, never through `bsr_from_triplets`; and `nnz` is a stale capacity

No `ordito/` module calls `warp.sparse.bsr_from_triplets` / `bsr_set_from_triplets` (check 27); it
allocates scratch several times the matrix. Tests may call it for an independent input. Build CSR
in `ordito/array.py`, with no host readback:

- **Sparsity follows from the mesh: write keys, not triplets.** A producer writes one
  `kernels/array.csr_key(row, col, n_rows, n_cols)` per contribution (sentinel `n_rows * n_cols` for
  an empty slot) into [`csr_key_buffers`][ordito.array.csr_key_buffers];
  [`csr_from_keys`][ordito.array.csr_from_keys] sorts and returns `offsets`, `columns` and each
  entry's first sorted position; a per-row value kernel forms the entries.
  `laplacian._mesh_operator_pattern` is the model. A caller assembling repeatedly over fixed faces
  builds the pattern once with `laplacian.mesh_operator_pattern` and passes `pattern=`.
- **A genuinely unordered coordinate list:** [`csr_from_triplets`][ordito.array.csr_from_triplets].
  `prune_numerical_zeros` drops entries whose sum is zero (Warp drops zero-valued triplets).
- **A CSR the producer can write directly** is written directly and wrapped by
  [`bsr_from_csr`][ordito.array.bsr_from_csr].

Never size a buffer, slice or launch `dim` off `matrix.nnz`.

- After a triplet build `nnz` is the triplet count handed in, duplicates included. Use
  `matrix.nnz_sync()` (one readback) or read `offsets[nrow]`.
- `nnz` is a cache that only `nnz_sync()` repairs (`bsr_mv`, `.numpy()` do not), so a `.nnz` read is
  order-dependent (tests build two operators: one to measure, one to hand over).
- The failure is silent: an allocation sized by `nnz` leaves an unwritten `wp.empty` tail that
  feeds garbage into real entries.
- A writer leaving triplet slots unwritten must point them out of range, not at `(0, 0, 0.0)`
  (§12.5).
- Any Warp `*_count` / `.nnz` field is a capacity until proven otherwise
  (`Volume.get_voxel_count()`; use `get_active_stats().voxel_count`).

More `warp.sparse` behaviour: §12.5.

### 3.8 NumPy at Python scope is sanctioned; leaking it through the API is not

NumPy is a core dependency of `warp-lang` and a device round trip costs far more than small host
arithmetic. **Do not open a "remove NumPy" pass.** Host metadata math and host-sequential algorithms
stay in NumPy; a closed-form parallel map with independent elements whose output scales with a
resolution parameter (`creation.icosphere`, `creation.grid`) is a kernel.

Three things are defects:

- **A public signature or return that names `np.ndarray`** (outside `ordito/io.py`). Return
  `wp.mat33d` / `wp.vec3`; annotate duck-typed inputs `Sequence[Sequence[float]]`.
- **NumPy standing in for a Warp Python-scope equivalent, unless pricing says otherwise.** Python-scope
  builtins are dispatch calls and often slower than NumPy or `math` (§13.1); the rule is to avoid a
  host round trip and `np.ndarray` in public signatures. `wp.full`, `arr[k:].fill_()`,
  `wp.determinant`, `wp.inverse` and `wp.svd3` need no host buffer. Prefer `math.pi` /
  `float("nan")` over `np` constants.
- **NumPy reducing a full `.numpy()` readback** (`.min()`, `.max()`, `.any()`, `.sum(axis=0)`). Use
  `ordito.reduce` or `wp.utils.array_sum`. Decide on the CUDA measurement (§9, §14), but keep the
  host path where the buffer never scales with the mesh.

Not defects to fix the other way: delete a `.tolist()` immediately splatted into a Warp vector or
matrix constructor in `ordito/` (keep it in `tests/`, §7.1, and where it is a `list[...]` return or
plain bookkeeping); `arr.numpy().tolist()` equals `arr.list()` only for rank 1, since `.list()`
flattens.

### 3.9 Device checks and the launch-device memory-safety rule

Every public function with two or more device-bearing arguments calls
`_device.require_same_device(**named)` as its first statement, passing every array, mesh, BVH, hash
grid, `wp.Volume` or list/tuple of those, including `X | None` arguments. It raises `RuntimeError`;
document it with a `Raises` entry (§4.3; check 11 cannot see the delegation). Internal code stays
trusting and forwards `device=` from an input array (§2.1).

`wp.launch` does not raise on a device mismatch: it corrupts the host heap or segfaults.

- `tests/conftest.py` sets `LaunchArrayAccessMode.STRICT`; keep the suite green under it.
- `STRICT` misses an omitted `device=` on a CUDA run; check 15 scans every launch. Always forward
  `device=`.
- `.numpy()` is not a sync on a CPU array (§12.1).

### 3.10 Readbacks

- Every `.numpy()` / `int(<device value>)` readback carries a comment naming why it is unavoidable.
  When the caller can supply the bound a readback infers, expose a keyword
  (`face_adjacency(n_vertices=...)`) and pass it from every in-repo caller that knows it (§14).
- Several adjacent small values: `ordito._device.read_values(arr, start, count)`. A tail value:
  `ordito._device.read_scalar(arr, index=-1)`, never a hand-rolled spelling (a pinned scratch races,
  §12.1).
- Several small device values go into one buffer and are read once (`points.fit_plane`); a tail
  read of a mesh-sized array stays `read_scalar`.

---

## 4. Evolving the public API

### 4.1 Naming

- Name a function after what it returns, in NumPy vocabulary, never after the Warp call it wraps.
- A mask is named `<element>_<property>_mask`, element first.
- A wrapper whose whole device side is one kernel nobody else launches shares that kernel's name. Reject filler affixes (`init_`, `do_`, `compute_`, `run_`, `make_`, `kernel_`, `_impl`, `_inner`); informative ones stay (plural for batched, `_pass` / `_step`, `finalize_`, a shared kernel library's vocabulary, one ingredient).
- A tuning choice is a keyword, not a name (`backend="hashgrid" | "bvh"`); use `None` for "not passed". Keep the discriminator in the benchmark group name and test that merged paths agree.
- One operation family, one module; place a function by what it computes, not where the reference library keeps it.

### 4.2 Signatures and returns

- A packed buffer and its offsets are returned and accepted values first: `(values, offsets)`, any per-item array last. A transposed unpack type-checks and indexes garbage.
- Offsets are always the `n + 1` total-terminated form (`indptr`; empty is `[0]`). Build by allocating `n + 1`, writing counts into `offsets[1:]` and scanning in place. Never read an item count as `offsets.shape[0]` (CPU heap corruption, §12.1).
- A variable-length-item function has a packed sibling `<name>_with_offsets` (the list form is that plus `array.split`); one taking the packed form is `<name>_from_offsets`.
- No public signature may name `np.ndarray`, outside `ordito/io.py` (§3.8). A guard must encode a real limitation.
- A `Literal` menu argument is validated at the public boundary and raises `ValueError` naming the argument, value and options; a delegating wrapper documents it and its test calls through the wrapper.
- When mirroring a NumPy function, mirror its positional signature and make `device` keyword-only.
- No speculative generality: add a parameter only when an in-repo caller needs it; remove unread ones.
- Inverse pairs cross-reference each other and have a round-trip test.
- Every public `ordito/<module>.py` has `tests/test_<module>.py` and `benchmarks/test_<module>.py` (check 4); tests live in the file mirroring the function's module (§5). Every public function and method has a benchmark row (check 31).

### 4.3 Docstring, signature and body must agree

- A documented `Raises` must be reachable; a direct `raise` needs a `Raises` block (check 11). Delegating to a shared guard means documenting its `Raises` (`_device.require_same_device`, §3.9).
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

Renaming a keyword argument also needs a grep of the old name alone (`**splat` `TypedDict` fields) and of fenced docstring examples (check 12). After any rename re-derive the residual set and run the whole suite.

### 4.5 The mechanical gate: `tests/api_conventions.py`

Thirty-one checks fail the default `pytest` run. Read an allowlist entry's reason before adding one; prefer fixing the code.

1. Summary line naming a reference library (§6).
2. `*_mask` producer not returning `wp.array[wp.bool]`.
3. Module summary ending in `(Warp)` / `on NVIDIA Warp`.
4. Module without both a `tests/` and a `benchmarks/` file.
5. Private name reached across a module boundary (a new `_*.py` for one helper dodges it).
6. One public name exported by two modules.
7. `kernels/<name>.py` without `ordito/<name>.py`, or the reverse.
8. Private helper defined above its first caller (§5).
9. Comment or docstring blaming a Warp version older than installed (anchored `Warp 1.18` spelling).
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
31. Public function or method that no `benchmarks/` row names (`od.<module>.<name>`; a method by name), outside `_BENCHMARK_EXEMPT` (§4.2).

Two rules about the gate itself:

- A new Warp construct (first `@wp.struct`, `wp.ref`, tile intrinsic, or even a plain `@wp.func` extraction) can silently hide buffers from a check that pattern-matches AST nodes; after introducing one, grep `tests/api_conventions.py` for such checks. §2.4's extraction wins over the check.
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

Docs use Zensical with mkdocstrings, configured in `mkdocs.yml`.

```bash
uv run zensical build --strict --clean  # validate; --clean after docstring edits
uv run zensical serve                   # preview; restart to see docstring changes
```

- API pages come from `api-autonav`, one per public module. The nav is `docs/SUMMARY.md`; a new module needs a line there and in the API overview `docs/reference.md` (module table, and the task index when it answers a common task) (check 28). Never use `gen-files`: Zensical ignores unsupported plugins silently.
- mkdocstrings templates are overridden in `docs/.overrides/mkdocstrings/` (backlinks render as one collapsed line); layout CSS is `docs/stylesheets/extra.css`. Check rendering changes in the built HTML, not only the build's exit status.
- A build finishing in about 0.1 s with an empty `site/` is inotify exhaustion (exits 0); raise `fs.inotify.max_user_watches`.

### Docstring content

- The summary line says what the function returns, never the C++ call it wraps (checks 1, 3); attribution goes in `Notes` or `See Also`.
- An array entry (every `Parameters` / `Returns` / `Yields` / `Attributes` entry, and a `Trimesh` property summary) opens with its shape as a code span, then its meaning: ``` ``(3 * n_faces,)`` flat triangle index buffer. ```
    - A rank-1 length is a 1-tuple (``` ``(n_vertices + 1,)`` ```); no `Length-`, `Shape`, `Flat`, `Rank-1`, or shape buried mid-sentence (check 29).
    - Spell sizes in full (`n_faces`); define a free size (`m`, `k`) on first use.
- Every public function and class ends its docstring with an `Examples` section: one fenced ```` ```python ```` block (never `>>>`) that imports what it uses, builds its own input (`od.creation`, a small literal) and sets no `device=`. It runs in a fresh namespace on each device, and a top-level `print(x)  # value` must print exactly `value` (`tests/doc_examples.py`), so print shapes, counts, booleans and floats rounded to the digits CPU and CUDA agree on. The guide pages under `docs/` follow the same print rule (`tests/test_docs_examples.py`).
- No measured timing (ms, ratios, launch or byte counts) in a public docstring; state the claim in caller terms.
- No development-history narrative in `ordito/*.py` ("measured X, declined Y", "reverted", "round N", internal section references). Keep the behavioural fact and delete the rest; reasoning belongs in `kernels/`, `benchmarks/`, `tests/` or Part II.
- `kernels/`, `benchmarks/` and `tests/` may state a ratio or crossover but not a wall-clock figure.

### Docstring syntax

Docstrings are NumPy-style. Use mkdocs-autorefs syntax; Sphinx roles (`:func:` etc.) render as literal text.

| Target | Syntax | Example |
|---|---|---|
| Internal (`ordito.*`) | `` [`short_name`][fully.qualified.path] `` | `` [`face_adjacency`][ordito.graph.face_adjacency] `` |
| External with inventory (`trimesh`, `numpy`, `scipy`, stdlib) | `` [`fully.qualified.name`][] `` | `` [`trimesh.grouping.group_rows`][] `` |
| External without inventory (`warp`, `igl`) | double-backtick code span, no link | ``` ``warp.sparse.BsrMatrix`` ``` |
| Shapes, literals, C++ names, paths | plain code span | ``` ``(n_vertices,)`` ``` |

- Resolve internal refs to the fully-qualified path, even within the same module.
- Use `!!! note`, only in free prose (description, `Notes`, `Examples`), never inside an item-list section (check 25).
- After editing docstrings, `grep -rnE ':(func|attr|meth|class|data|mod):\`' ordito/` must return nothing (`kernels/` included).

---

## 7. Testing

Every new geometry function MUST have regression tests comparing against a CPU reference
implementation. `trimesh` is the default; §7.6 covers the other eight libraries.

### 7.1 Conventions

- Test file `tests/test_<module>.py`; import `trimesh.<module> as tm` and `ordito.<module> as od`.
- Use the `device` fixture from `tests/conftest.py`. A test names `device` only if it touches the
  device directly; a test that reaches it through a mesh fixture does not. `pytest_generate_tests`
  parametrizes every test whose fixture closure reaches `device`. `reportUnusedParameter` (§8)
  enforces this.
- Tests use pytest alone, never `unittest` (ruff `TID251`). Patch with `monkeypatch`; assert with
  plain `assert`, `pytest.raises` and `pytest.warns`. A test that counts or intercepts launches or
  allocations patches both `ordito._launch.*` and the `wp.*` fallback.
- Use `np.random.default_rng(seed)` with a fixed seed per test.
- Upload a NumPy mesh with `conversions.numpy_to_warp(vertices_np, faces_np, device)`, never a
  local helper (`numpy_to_warp_uv` for `wp.vec2`, `warp_to_trimesh` for the inverse). A triangle
  soup `(n, 3, 3)` becomes an indexed mesh first: reshape to `(-1, 3)` and pass
  `np.arange(n * 3, dtype=np.int32)` as faces.
- Call `.numpy()` on Warp outputs inline, before the NumPy comparison; bind no intermediate.
- `np.allclose(..., rtol=1e-5, atol=1e-5)` for floats; `np.array_equal` for bool and int.
- Variable suffixes name the library: `_np`, `_tm`, `_wp`, `_igl`, `_pp`, `_pml`, `_o3d`, `_pv`, `_ml`,
  `_pmf`, `_p3d`. Avoid `got` / `exp`.
- A NumPy vector to a `wp.vec3` argument in a test: `wp.vec3(*array_np.tolist())`. Production code
  in `ordito/` drops the `.tolist()` (§3.8).

### 7.2 Devices and the two-process runner

`--device={auto,cpu,cuda,both}` selects devices for this process (`auto` is cuda if available).

- Both-device coverage is a two-process job: `uv run python -m tests.devices` (a CUDA pass, then a
  CPU pass with `CUDA_VISIBLE_DEVICES=""`), because Warp's CPU work is much slower once CUDA is
  initialised (§12.1). Never use `--device=both` on a GPU box for CPU coverage.
- CPU runs are parallel, CUDA runs are not. The CPU passes of `tests.devices` run under
  `pytest-xdist` (`--cpu-workers`, default 8; CI's `pytest-cpu` uses `-n 4`): a CPU kernel runs on
  one thread, so workers scale until the slowest tests set the wall clock, and xdist matched the
  serial run's pass, skip and coverage counts exactly. A CUDA run stays one process, or the
  workers compete for the device's memory. A test must therefore not depend on another test
  having run in its process.
- While developing, run `uv run python -m pytest tests/test_<module>.py -q --device=cuda` only.
- Use `tests.devices` when a change is device-dependent by construction (`launch_tiled` kernel (§12.2),
  `_device.prefers_tiled_reduction`, device-gated constant (§13.2)).
- CPU is the deterministic oracle for byte-for-byte A/B; on CUDA any reduction-order change is
  noise.
- A test costing more than about 15 s on CPU wears `@pytest.mark.slow_cpu(<seconds>)`.
- A `launch_tiled` kernel's test pins `"cpu"` explicitly in a parametrize, because the fixture
  returns `cuda:0` whenever CUDA exists (§12.2).
- `python -m tests.devices --cpu-blocks` adds a pass running CUDA tile code on the CPU.

### 7.3 Mesh fixtures

Reuse `tests/conftest.py` fixtures (each returns `(mesh_tm, mesh_wp)`):

| Fixture | Use when |
|---------|----------|
| `icosahedron`, `icosphere`, `icosphere_coarse` | Easy closed solids; edge-case and sign tests only |
| `unit_box` | Sharp features: exact 90° creases; crease/seam/dihedral tests |
| `cave_cube` | Hollow / multi-component shell |
| `hemisphere`, `half_torus` | Easy open surfaces |
| `boy_surface` | Closed, non-orientable, χ = 1 |
| `mobius` | Non-orientable with one boundary loop, χ = 0 |
| `bohemian_dome` | Closed genus 1, self-intersecting |
| `saddle_graded` | **Default open fixture** (`OPEN_MESHES`): graded disk, needles of aspect ~4 800, one rim; `saddle_graded_large_arrays()` is the larger form for size-dependent tests |
| `sphere_irregular` | **Default closed fixture** (`CLOSED_MESHES`): degree 3-12, aspect up to ~300, mostly obtuse faces, scrambled order, rotated and off-origin |
| `sphere_irregular_hollow` | Solid with a cavity, two components |
| `torus_irregular`, `torus_irregular_holes` | Genus 1 with needles; the second adds three holes (`BOUNDARY_MESHES`) |
| `sphere_irregular_cap`, `sphere_irregular_band`, `convex_irregular_cap`, `convex_irregular_band` | Cut bodies with planar rims (bumpy / convex) |
| `sphere_well_shaped`, `sphere_well_shaped_open`, `torus_well_shaped` | Same surfaces with no needles (aspect ~6), for premises the needles break |
| `sphere_round`, `torus_round` | Exact unit sphere / torus in a known frame (`conftest.round_frame_coordinates`): analytic oracles only |

- Shared sets in `tests/conftest.py`: `CLOSED_MESHES`, `OPEN_MESHES`, `MESHES` (their sum) and
  `BOUNDARY_MESHES`. Import them; keep a local list only for a genuinely different set, with a
  comment saying why.
- Keep the hardest fixture per class, not many easy ones. A test about topology names the fixture
  that has it (several components, another χ, several rims). A test whose premise the hard
  fixtures break (Delaunay, convex, planar rim, no ears, `float32`-length paths, hard-coded counts)
  names a well-shaped one, with the reason beside the list.
- The scrambled fixtures sit off the origin: probe at `center_mass`; analytic oracles use `sphere_round`.
- Do not hand-roll `wp.Mesh(...)` unless the case needs a bespoke degenerate mesh. Never construct
  a zero-triangle `wp.Mesh` on CUDA (§12.1); a single-triangle mesh reaches an "empty mesh" guard.
- Parametrize over fixtures with `request.getfixturevalue(mesh_name)`; allocate queries on
  `mesh_wp.device`.
- Do not use `trimesh.slice_plane` output as a reference input; assert the hole count first.

### 7.4 The parity gate: a benchmarked reference must be a tested reference

`tests/test_parity.py` fails the default `pytest` run when a benchmarked `(group, library)` pair is
neither tested nor exempted. The `benchmark(group=...)` name is the key; renaming one breaks every
marker citing it. Both markers are stackable and take string literals only.

```python
# tests/test_edges.py -- this test proves ordito agrees with trimesh for that group
@pytest.mark.parity("faces_to_edges", "trimesh")

# benchmarks/test_curvature.py -- timed, but the results are not comparable
@pytest.mark.noparity("pymeshlab", oracle="trimesh", reason="...why not comparable, with numbers")
```

`uv run python -m tests.parity` prints the full matrix.

- Where a reference computes the same quantity, one test must compare the two outputs; an invariant
  does not discharge it. Check that the reference really has the quantity (§7.6). Invariant checks
  may share the same test.
- Where no reference computes the quantity, an invariant-only test is the honest answer. Its
  docstring says "Not a library comparison: <why none exists>" and what the invariant excludes.
- Classify every comparison in the docstring, written as `Class A`:
    - **A**: direct `np.allclose` / `np.array_equal`. The default. An answer equal up to how its
      elements are numbered is Class A too, compared through `ordito.equality`: `meshes_allclose`
      (vertex numbering, face order and face rotation), `faces_equal` (face order and rotation
      over a shared numbering), `points_allclose` (point order), `rows_equal` (row order of an
      integer table; sort each row first for undirected edges) and `partitions_equal` (label
      names). Numbering is not part of the answer. Upload the reference with the
      `tests/conversions` helpers; `atol` is a Euclidean distance and must stay under half the
      reference's closest point pair (`saddle_graded`: 1.4e-5).
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
  Ordito-against-ordito is legitimate for pinning two entry points where only one has an oracle
  (say which).
- Check the comparison is not vacuous on its fixture: assert the reference's answer is non-empty,
  has its expected count, and is not constant (`np.unique(labels_np).shape[0] > 1`,
  `np.ptp(reference) > 1e-3`). A docstring saying the vacuous branch is "covered separately" is a
  claim to grep for.
- A guard test bites only if you identify which assertion fails under the mutation.
- Mutation-probe boundary predicates with exact ties, and parametrize over the boundary value of
  the parameter the contract turns on.

### 7.5 Shared helpers: check both modules before writing a private one

- `tests/comparisons.py` holds the comparison helpers (`lexsort_rows`, `assert_unordered_rows_equal`,
  `canonical_labels`, `same_partition`, `fraction_within`, `symmetric_chamfer`, `chamfer_two_sided`,
  `hausdorff_two_sided`, `sparse_allclose`, and more): list it before writing a private one.
- `tests/conversions.py` holds the converters for every library; use them to build the reference
  and to read back.
- Compare a sparse operator with `sparse_allclose`, never densified `toarray()` comparisons.
- `symmetric_chamfer` takes meshes, `chamfer_two_sided` drawn clouds; prefer the mean form over
  Hausdorff for distributional claims.
- `lexsort` is unusable on float coordinates with ties. For positions use a `cKDTree` match with a
  bijection check, or `hausdorff_two_sided`; keep `lexsort_rows` for integer index rows.

### 7.6 The nine reference libraries

- All nine (trimesh, igl, potpourri3d, pymeshlab, open3d, pyvista, meshlib, pymeshfix, pytorch3d)
  are hard test dependencies: import them plainly, never via `pytest.importorskip`. Aliases are
  pinned in ruff (`import trimesh as tm`, `import igl`, `import potpourri3d as pp3d`, `import
  pymeshlab as ml`, `import open3d as o3d`, `import pyvista as pv`, `from meshlib import mrmeshpy
  as mm` / `mrmeshnumpy as mn`, `import pymeshfix`, `import pytorch3d.ops as p3d_ops`).
- Threading decides how a ratio reads: trimesh, igl, pyvista and pymeshfix are single-threaded;
  meshlib and pytorch3d-cpu are multi-threaded; only pytorch3d-cuda runs on the GPU.
- Licensing: nothing under `ordito/` may name MeshLib or promesh (check 21), except an option
  reproducing MeshLib, whose method value is `"meshlib"`; pymeshfix and pymeshlab
  (GPL) may be named in prose, but no ordito code may derive from their source.
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
- Import aliases and docstrings (`D`) are enforced.
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
  a return keeps the `Literal`-ranked `odt.Array1d*` / `Array2d*` aliases.
    - Allocate through `_launch.empty` / `odt.empty_1d`, not bare `wp.empty` (it binds the first
      overload arm).
    - A slice is narrowed with `odt.as_dense`; a gather (`src[indices]`) is materialized with
      `wp.copy` (§3.4).
- Order of preference: a correct annotation or `odt` helper, then `cast`, then a scoped
  `# pyright: ignore[rule]` with its reason (in `_launch.py`'s hot path an ignore beats a `cast`).
- `reportPossiblyUnboundVariable` is an error and is never suppressed: initialize to `None` before
  the branch and `assert x is not None` at the use.
- `reportUnnecessaryTypeIgnoreComment` and `reportUnusedParameter` are on (the latter covers
  `tests/` and `benchmarks/`). A protocol-signature parameter is `_`-prefixed; a `parametrize`
  value used only as a label goes in `pytest.param(..., id=)`.
- An error is almost always a real bug: resolve it, do not widen the disabled-rule list.
- `tests/` and `benchmarks/` are checked at the same bar. Test-side gaps: `tests.conversions.warp_empty`,
  `tests/typings/pymeshlab`, `assert odt.has_blocks(...)`. A private helper exercised on purpose takes
  a scoped `# pyright: ignore[reportPrivateUsage]`.

### Dependencies

- Test and reference dependencies live under `[dependency-groups] test`; use `uv add --group test <pkg>`.
- `plans/` is gitignored.

### Version control

- Commit directly on `main` unless the user asks for a branch. Committing is an explicit request:
  finish the work, run the gates, commit when asked.
- An A/B against a prior revision uses a detached worktree, never `git stash` and never a branch
  checkout in the live tree (§15).

### Coverage

- `ordito/kernels/` is omitted from coverage on purpose (kernel bodies are never run as Python).
  Never write tests to raise a kernel-module figure.
- The badge is the CPU wrapper layer. Do not raise the floor to chase `device.is_cuda` branches;
  they belong to §7.2's two-process job.
- Coverage gates wrapper branches only; it says nothing about vacuous comparisons (§7.4).

## 9. Performance work: measure before you change

Numbers live in §13 (cost model), §14 (kernel shapes) and §15 (measurement traps); component status
is in `reference/component-status.md`.

- Something suddenly slow is a Warp rebuild until proven otherwise (§15).
- A benchmark lands before the optimization; never restructure for speed without one timing the
  current implementation.
- Attribute a cost with one direct measurement at the benchmark's own operating point and at more
  than one size (§15). A share that falls as the input grows is a decline.
- Check whether the function graph-captures before reading device time (§15).
- Before proposing an optimization, grep the call site, constants and benchmark docstring for a
  written decline (§15).
- Attribute a change only with an interleaved A/B in one session, using a detached worktree
  (§15). A probe that instruments what it measures carries a do-nothing control arm.
- CUDA is the target: decide on the CUDA number, measure both devices, and keep the CPU path
  correct (§14). Split a tuning constant per device when the optima differ (§13.2).
- A decline is a result: write it at the site with the number, and resolve every site the finding
  named.
- Count the launches a numerical-method change adds per iteration (§14).
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

- Benchmark suites and probes run for minutes. Do not write a wait loop around them: run the
  command in the background and read its own output file when it re-invokes you.
- `until ! pgrep -f <name>; do ...; done` never terminates (`pgrep -f` matches the polling shell's
  own command line), and `pkill -f` kills the shell the same way.
- If a poll is unavoidable, test a sentinel file the process writes on exit and bound the iteration
  count.
- Do not run a timing probe while `pytest` or `zensical` is running.

---

# PART II — MEASURED FACTS

---

## 12. The Warp platform: bugs, quirks and version status

### 12.1 Memory safety — every failure here is silent

- A `wp.launch` without `device=` corrupts the host heap: it resolves to `cuda:0` while CPU arrays are freed under the running kernel. Always forward `device=` (§2.1).
- An out-of-bounds kernel write on CPU is glibc heap corruption; range-check at the write. Debug mode (-O0) raises register use and drops FMA, so exact-float comparisons move.
- A zero-triangle `wp.Mesh` corrupts CUDA allocator state (error 700 on the next allocation; NVIDIA/warp#1765). Never build one on CUDA, tests included.
- Python-scope gather ignores a non-contiguous index view's stride (§3.4); `wp.copy(..., count=0)` copies the whole source; `wp.copy` into pinned memory is async with no event (use `_device.read_scalar`, safe for scalars only).
- CPU work is ~36x slower once CUDA is initialised in the process, hence two-process coverage (§7.2).

### 12.2 `wp.launch_tiled` runs one lane per block on the CPU

On CPU, `launch_tiled` runs one thread per block and `wp.block_dim()` reads 1. The experimental `wp.config.enable_cpu_blocks` (NVIDIA/warp#1638) runs every lane but is 1.5-340x slower: a test oracle (`--cpu-blocks`), not a production path.

- Probe a lane-constructed tile (`wp.tile(x)` from per-lane values), not `tile_load`, or you will delete a correctness branch.
- Stride by `wp.block_dim()`. Partitioned kernels keep a lane-free `_sliced` sibling behind `_device.prefers_tiled_reduction` (§2.2).
- Warp has no grid-wide barrier, no per-cell `HashGrid` entry point and no builtin node-by-node BVH traversal; node reads take `wp.func_native` over `wp::BVH` (`kernels/proximity.py`, re-probe the layout at each upgrade).
- `wp.tile_bvh_query_aabb` overruns its `5 * block_dim` shared result buffer exactly when a round meets leaves holding more than 4 primitives, and every Warp BVH builder makes a node at depth 32 a leaf whatever its size (lucy's mesh BVH: 0.9 M such leaves, up to 28 faces; the bunnies none). The overrun drops primitives and reads past the buffer *inside* the builtin, which faults (CUDA error 700) once it is large, so no bound check on the returned index can catch it. Walk with it only a query that meets no such leaf; `kernels/proximity.flag_overfull_leaf` flags those leaves in a node bitset, and the rest take a walk that cannot drop (`mesh_to_mesh_distance`: its own `wp.tile_stack` walk; `ball_pivoting`: per-lane slabs of the box through the serial `wp.bvh_query_aabb`). Point clouds hit it too (dragon, happy_buddha, lucy). Still bound-check every candidate (`0 <= c < n`). `max_nodes` is the *capacity*: a CUDA LBVH builds all `2n - 1` nodes and never writes `num_nodes` (`wp_bvh_create_device` copies an uninitialised stack `wp::BVH`, so the field is stale stack contents: usually 0, once a large value that sent the `ball_pivoting` bitset pass past `node_lowers`, CUDA error 700 found by `compute-sanitizer`), a top-down tree (every CPU `wp.Mesh`, or `bvh_constructor="sah"` on CUDA) only a prefix counted in `num_nodes` (657 of 1999 measured), the rest uninitialised. The host builder writes `num_nodes` but not `constructor_type`. `kernels/proximity.bvh_node_count` therefore selects on `constructor_type` on CUDA and reads `num_nodes` on the CPU, clamped to `max_nodes`; never trust either field alone. A node-bitset pass runs wherever blocks have several lanes (`prefers_tiled_reduction`), where the tile walk exists.
- A multi-launch `capture_while` body replays differently on CPU, and Python-level ping-pong cannot help a captured loop.

### 12.3 Numerics, casts and `wp.ref`

- Generic type-vars do not instantiate inside `wp.ref[...]`; use `wp.ref[wp.float32]`. `@wp.func` cannot be overloaded by name and `arr[i], arr[j] = ...` swaps raise, so the shared argmin/argmax helpers are concrete float32 and float64 sites keep loops.
- `wp.Scalar` does not instantiate for `wp.bool`; write the concrete bool kernel rather than widening to int32 (~2x). Scalar arguments must match the input precision (`a.dtype(tol)`); literals in a generic `@wp.func` need `type(x)(...)`.
- Sum cross products about a point of the loop, never the origin (`predicates.newell_term`).
- FMA fusion makes a degenerate triangle's area ~1e-8 on CUDA and 0 on CPU. Do not disable it; write degeneracy tests at small scale and never assume CPU/CUDA bit agreement.
- Form area vectors and cotangents where they do not cancel (`predicates.triangle_area_vector` crosses at the largest angle); Heron needs sorted sides. Float32 lengths cannot represent a needle (the intrinsic path warns).
- Never substitute an algebraic identity inside a sign test that feeds a branch; CUDA differs, CPU does not. Exactly coplanar float32 corners give float64 plane distances of ~1e-19 with mixed signs on CUDA (FMA) and exact zeros on CPU: snap a distance within its own rounding to zero before branching on its sign (`predicates.plane_distances_d`; without it two disjoint coplanar triangles "intersected" on CUDA).
- `mesh_query_point_no_sign` is inexact (~2e-5, the point worse than the distance; its face need not be the nearest, so an exact match re-searches the faces within that distance, `kernels/registration.landmark_match`); `length < r` and `length_sq < r*r` differ in float32 (§2.4); NaN breaks `searchsorted(side="right")` (§3.1).
- Gate an angle near pi on the angle (`atan2` form, `predicates.vector_angle`), not on its `float32` cosine: the cosine is flat there, so every threshold above ~179.98° rounded to `cos == -1` and the fold gate of `face_defective_mask` / `remove_folded_faces` never flagged an exactly folded-back pair (fixed 2026-10-26; MeshLib, which tests `dot <= cos`, is inclusive at the threshold where ordito is strict).
- Float32 storage sets a noise floor no tolerance reaches; heat fields scale as 1/scale², so absolute tolerances on them are wrong.
- The backward pass of an empty dynamic `range` runs one iteration past the array: a differentiated thread whose loop can draw nothing must return first.
- **Float atomics and CUB's float scan are not run-to-run reproducible on CUDA**; integer ones are exact. One `atomic_add` per block changed `procrustes`, `moments`, `gram_matrix` and `icp` in 30 of 30 calls at 1 M items; `array_scan` (decoupled look-back) changed every 1.3 M-element `float32` scan, and with it every seeded CDF draw. Reproducible forms: `_launch.array_scan` takes a fixed-order tile scan for `float32` on CUDA (`wp.tile_scan_*` takes no `float64`), and `store_block_*` stores a row per block that the block drawing the last ticket folds after a `__threadfence` (`wp.func_native`). A vector or matrix counter is one atomic per component, so a ticket is read off one component (`last_ticket_vector` / `_matrix`): comparing whole values let no block, or two, see the last ticket. `wp.utils.array_sum` (CUB reduce) and `bsr_mv` repeat bit for bit. Per-vertex scatters too: `hessian_energy`, `curved_hessian_energy` and `vertex_defects` differed on every CUDA call (~1e-13 relative on the energies) until they took the §2.2 forms; so did `vertex_face_adjacency`'s row order.
- Moving `wp.svd3` into another kernel (a dim=1 finalizer fused into a scatter's last block) changes its CUDA result by ulps, and flipped a degenerate 2-point plane normal; CPU stays byte-identical. A last-block ticket on a matrix counter is 9 float atomics per block on one address (~18 µs at 14 k blocks): take the ticket on an `int32`.
- **Exact orientation exists** (`predicates.orient3d_sign` / `_f32` / `_from`, `orient2d_sign`): exact for `float32` coordinates plus one 28-bit point, FMA-proof, the same sign on both devices. Use it, not a tolerance, wherever a sign decides topology.
- `int(x)` and `wp.int32(x)` generate identical code, but `float()` fails to parse inside `wp.Float`-generic functions; use `type(x)(...)`. `//` floors since 1.18; `%` truncates (§1.5). At Python scope `wp.int32` arithmetic costs ~10 µs and `//` raises; unwrap with `int()` (check 26).

### 12.4 Compilation, module hashing and import cost

- A generic kernel's lazy overload instantiation rebuilds its whole module, and `wp.map` forks per call signature (§2.5, §3.5). Warp's own generic kernels do too (`array_cast`, `warp.sparse`): use `array.copyto` or `kernels/linalg.register_warp_overload`.
- `@wp.kernel` builds an `Adjoint` at import, so the lazy `__init__` is load-bearing; another module's top-level `import warp.fem` silently cancels a deferral.
- Backward passes dominate cold compile (§2.6). Only host readbacks block graph capture.
- `@wp.func` is a real function that nvcc inlines; verify "free" by comparing `_forward` SASS from the module cache.
- A kernel-scope `not` on a bool lowers as a select; share sentinel-returning predicates, not bools.
- `wp.constant(x)` is an identity; a tile `shape=` must be a plain integer; wrap typed constants in `int()` on the host.
- A launch's `block_dim` is a compile-time parameter of the module launched (`Module.load(device, block_dim)`, with matching `__launch_bounds__`): a kernel launched at a second width builds its whole module again, and one of its kernels may not build at that width (`kernels/neighbors` at 128 failed). A kernel launched at its own width lives in its own module (`module="unique"`; `kernels/validation.closed_signed_volume`, `kernels/neighbors.query_nearest_via_bvh`).

### 12.5 `warp.sparse` and `wp.Volume`

`nnz` is a capacity (§3.7).

- `bsr_mm` returns a structural superset (explicit zeros) that inflates chained products; `bsr_mm`, `bsr_axpy` and `bsr_set_transpose` read the `nnz` field.
- `bsr_mv` takes one vector. Unwritten `(0, 0, 0.0)` triplets all accumulate on one entry (9-32x): point them out of range.
- `TiledDot` takes an O(n) per-block reduction under `batch_offsets` with `batch_count > 1`; `linalg._BatchedCg` avoids it.
- `cg` at `check_every=0` records a graph per call; it resolves an omitted `atol` to `tol`, so a tiny `||b||` returns zero iterations. Always pass `atol=0.0`.
- `allocate_by_voxels` and `Nanogrid` work on CPU; an empty point set raises. A one-`atomic_cas` table beats `allocate_by_voxels` 1.3-3.2x.
- `get_voxels()` order is leaf-major: lexsort before comparing. NanoVDB centres voxels on integers: pass `translation = origin + 0.5 * voxel_size` or cells are off by one.
- `get_voxel_count()` is a capacity; use `get_active_stats().voxel_count`. `rebuild()` does not pay. Import `warp.fem` inside the function.

### 12.6 Warp builtins: adoption verdicts

- **Adopted:** `mesh_query_point_sign_winding_number` (exact on holes; silently returns parity unless the mesh was built with `support_winding_number=True`); `bvh_query_sphere` (wins when the radius is large against a BVH, equals `length_sq <= r*r`, still beats a box walk on 1.18 though slower there; not for a radius equal to the grid cell); BVH leaf size 1 for ball queries and size-gated for k-NN; `mesh_get_bvh`; `volume_index_to_world`; `delaunay_edge_flip` as the flip start above 2**19 points; `sparse_marching_cubes` above a lattice-size gate on closed, consistently wound input.
- **Rejected:** `intersect_tri_tri` (not scale-invariant); `closest_point_edge_edge` (float32, worse); `sample_unit_hemisphere_surface` (variance); `norm_huber` (norm, not weight); `tile_arange`; `volume_voxel_count` (capacity); `dense_chol` family (hidden, 2x loss); `tri_tri_adjacency` for `halfedge_twins` (cannot carry validation, quadratic on hubs; test oracle only); `swept_volume_mesh` (no caller).

### 12.7 Upgrade discipline

The bugs above (zero-triangle mesh, CPU one-lane tiles, `tile_bvh` overrun, `wp.ref` generics, `bsr_mm` superset, stale `nnz`) all persist on 1.18; re-probe them and every tuning constant (§9) at each upgrade. Verification defaults to a false pass: `compute-sanitizer` that instrumented nothing prints 0 errors (expect ~10x slowdown), and `capture_while` sites skip the graph path when `wp.is_conditional_graph_supported()` is `False`. Gates: full suite, `basedpyright` 0 errors, `zensical build --strict`, unchanged `tests.parity` pair count.

---

## 13. The cost model (RTX 5090)

### 13.1 Host-side, per call

Measure `n` calls between two syncs and divide; a sync inside the loop drains the mempool (up to 14x wrong). The lever is allocating less; a cache holding memory across calls is rejected (reference/component-status.md).

| primitive | cost |
|---|---|
| `wp.launch`, flat in `dim` | ~12 µs (`_launch.launch` 4.5-6 µs) |
| each launch argument | ~1 µs; generic kernel +12 µs |
| `wp.empty` / `wp.zeros` / `wp.copy` | 6 / 9 / 4.5 µs |
| `array_cast` / `array_sum` | 21 µs / 40 µs (avoid) |
| cached `wp.map` | ~11 µs above the kernel |
| readback | ~0.1 ms queued, 14 µs isolated |
| replayed graph kernel | 1.2 µs |
| small `.numpy()` readback / `_device.read_values` | ~27 µs (builds a host `wp.array`) / ~9 µs |
| `wp.array` slice / `_launch.view` | 4.2 / 1.05 µs |
| small-pattern native fill (`_launch.pattern`) / 2-element upload | 8.2 / 11.7 µs |

Small constant tables passed by value as `wp.types.vector` kernel arguments cost more host time than uploading them. The creation builders are host-bound: NumPy and Python per-op overhead (1-3 µs each) dominates, not the upload or the launch.

Warp-typed values at Python scope are slow: `wp.int32` arithmetic, slices with Warp-typed bounds, `wp.length` and vector arithmetic cost 4-40 µs against under 1 µs in plain Python (13-370x). Unwrap with `int()` / `float()` (check 26); `wp.cross` is the exception. A device reduction costs 0.1-0.3 ms flat; a readback wins below ~200 k `int32` elements (1 M `bool`).

### 13.2 Device-side access and tuning constants

- `wp.tile_sum(wp.tile(x))[0]` is a block-wide reduction and a barrier (~0.13 µs at 32 lanes, ~0.35 µs at 64-128). A cooperative one-block rewrite wins only when a level's work exceeds ~0.6 µs of barriers (§14).
- A single-address atomic serializes the launch. Use lane partition, `wp.tile_sum` and one guarded commit per block, launched at `kernel_reduce.blocks_1d(n)`; the block count is the tuning variable. Do not convert a conditional atomic.
- An append-only list whose order decides nothing takes `kernels/array.append_slot` (one `atomicAdd` per warp), and a max-bid onto few addresses first skips a bid below the current value: together they took `convex_hull`'s first pass over a 14 M cloud 4.38 -> 0.62 ms.
- **`float64` runs at 1/64 of the `float32` rate on this device**, so a launch of a few warps chaining `float64` tests is latency-bound: put a `float32` filter with a rigorous bound ahead (`predicates.orient3d_filtered_f32`, `orient3d_sign_from`; `convex_hull`'s flip round 2.5x).
- Pack same-dtype quantities into one vector tile (`preserve_type=True`) for one barrier. Float sums that must reproduce use a fixed-order two-stage sum.
- A tile kernel whose reduced extent is under one tile loses ~49x; `tile_chunk` reports what remains, so clamp with `reduce.block_chunk`.
- Tuning constants are per-device: CUDA wants long strided slices, CPU short (`ITEMS_PER_SLICE_CUDA = 128`, `_CPU = 32`). Sweep both devices and more than two values, split the constant if the optima differ, and re-probe after an upgrade (§9).
- **Reproducibility prices (§12.3):** the last-block commit adds 2-5 µs of device time to a launch (fence, ticket, the last block's fold, latency-bound: fold its rows with interleaved accumulators), against 3-8 µs of host for a separate fold launch. The fixed-order scan is two launches and an allocation above one 2048-element chunk (+10-15 µs, 3x CUB's device time at 5 M) and cheaper than CUB below. On a 15-100 µs call either crosses 10 %, which is why the setting can be switched off (`ordito.config`, §2.2); on solves and loops it is noise.

---

## 14. Kernel-shape verdicts

- **Block-per-item** wins when the outer dimension alone starves the device and loses when a slice dimension already fills it (§2.3); such kernels use register blocking instead (below). Models: `obscurance`, `shape_diameter`, `farthest_point_sample` as one persistent block. Any-hit tracing suits binary occlusion; a point-major layout wins only on very large clouds. A cuBQL mesh pays only when the caller builds once and traces many rays (`Trimesh.mesh_for_rays` decides). `max_tangent_sphere` answers inside a tolerance window; an "exact" variant is not the true answer on a mesh.
- **Cooperative BVH walks:** the tiled walk beats the hash grid only while concurrent queries are few (a few thousand for a ball of the cell width); the serial BVH does not. Only `ball_pivoting`'s pivot search qualified, so do not re-run the tree-wide sweep. A thread-per-query BVH launch is usually load-imbalanced, not under-pruned: histogram per-thread candidates first. Walk time varies up to 5x with loop spelling, so time a new or fused walk on the device.
- **CUDA graph capture** wins when a launch sequence repeats identically (break-even about two replayed rounds) and loses on once-through loops and short fixed-count loops. A chain differing only in a loop counter becomes repeatable once the counter lives on the device (record a group of 8). A `capture_while` body may not allocate (`array_scan` does) and loses to a batched host loop when its per-iteration overhead exceeds the sync it removes. **A batch of replays per readback runs whole past the fixpoint**, and a replayed round costs the device what an issued one does: on device-bound rounds the idle tail costs as much as the work (the flip loop's headroom batch: 26.3 vs 15.6 ms on dragon), so replay one round a read and record only loops that run long (`_FLIP_CAPTURE_FROM_ROUND`). A fixed-width collapse pass replayed lost too: 36 nodes, device-bound at about the host cost of issuing it. A captured path needs a plain-loop CPU sibling, which is also the byte-identity reference.
- **Tile solves** win for K >= 16-32 with few systems and a large matrix; ordito's dense solves (K = 5, 6) lose, so "rewrite them as tiles" is refuted.
- **Readbacks:** a device reduction wins on CUDA for mesh-sized arrays and loses on CPU; decide on CUDA, measure both. Keep the readback for bool masks and small non-scaling buffers (the real axis is often the loop count). In device loops a per-pass readback is cheaper than an extra pass, so trading passes for fewer syncs loses; `cg(check_every=0)` is not a lever in the harness.
- **Per-device algorithm choice** only when the parallel form does asymptotically more work (pointer-doubled `polyline_downsample`, CUDA only above a size gate). `prefers_tiled_reduction` is a correctness branch, not a performance one.
- **Solvers:** a single-level polynomial preconditioner wins on long solves; a Chebyshev multigrid smoother loses (no single `(degree, interval)` is robust). A small synthetic system can mislead.
- **Refuted, do not re-propose:** a single-block BFS drain, a one-block hole-fill DP and merged-level blocked interval DPs (the grid must stay full); voxel aggregation for multigrid; micro-tuned Bridson blue noise; two-stream overlap of sequential branches.
- **Producer-consumer fusion:** always fuse consecutive same-`dim` launches where the consumer reads the producer only at its own index; every pair measured was faster. Price the fused region, not the call containing it, and cross-check kernel and allocation counts. Also fold a map into its consumer, merge two launches of one kernel into a wider one, and walk-and-reduce neighbour lists in one thread per query. A fusion that changes a slot or box index needs a debug-mode run. Inline single-caller helpers it leaves behind; fold-on-read loses on dependent chases.
- **Blocked wavefront:** a 2-D DP with local directional dependencies and small per-step work wins as one block per tile and one launch per tile-diagonal, with `wp.tile_sum` as the barrier. It loses where levels are already wide (the fill DP). Test it above 32 lanes and pin it byte for byte to the simple schedule.
- **Register-blocking outer items:** giving each thread W outer items from registers wins for `(item, slice)` reductions bound by streaming the cloud once per item. Choose W per launch so the grid keeps about `1 << 17` threads; a fixed W loses on small inputs.
- **Gather-form fusion** (recomputing a neighbour's per-face value from positions instead of reading a table) wins below ~100 k faces and on grid-ordered meshes but loses 1.7-2.1x on scrambled ~1 M-face meshes (`cotmatrix`, `filter_normals`). Price it on a scrambled large mesh.
- **Independent per-item iterations collapse into one in-thread loop:** a per-round launch plus convergence readback over items that converge independently (`max_tangent_sphere`) became one kernel, 1.1-1.2x device time and no per-round sync. Keep that loop out of a kernel holding another long walk (1.1-1.6x, registers).
- **Occupancy for pointer-chasing tree walks:** a thread-per-query BVH descent that chases nodes through L1 runs faster with fewer warps resident. The `k = 1` point-tree descent (40 registers) at 1024-thread blocks was 0.75-0.87x its 256-wide launch from about two blocks per SM, and 1.04-1.67x below one (too few blocks), so its width is gated on the query count (`kernels/neighbors.DESCENT_WIDE_BLOCK_DIM`). Fewer registers do not make a walk faster by themselves: at 256 the same descent lost 1.1-1.2x to Warp's 53-register mesh query on the same tree, and a line-for-line port of that query lost the same. Check a new walk at two widths before reading its per-node code.
- **Shapes that lose:** a line-per-thread lattice pass absorbing a cell-per-thread one (`fill_orthographic`, 0.4-0.85x); a single-thread serial walk storing gathered values instead of computed indices (1.5-3x, polyline downsample); a per-face edge marker instead of a per-edge one plus a child-count launch (1.2x, remesh split).
- **Rasterization:** one `atomic_min` owner pass per face, then one thread per pixel to resolve, beats a pre-filled image plus a per-face scatter 1.5-4.9x.
- **Per-class compaction:** scan a one-hot `wp.vec{K}i` per item; the last entry holds all K counts (one read) and each item's in-class rank is the step from the previous entry. A compaction folds into its emit when the emit follows the size readback anyway. A `vec2i` scan of `(flag, count)` gives a compaction rank and CSR offsets with both totals in one read.
- **`store_block_sum` / `store_block_total` count blocks from the partials buffer**, so two reductions of different widths share one launch at `dim = max(blocks)`. `0 + block_sum(x)` equals the one-block commit bit for bit, so "frame then reduce" fuses whenever `blocks_1d(n) == 1`.
- **Readback cadence in idempotent loops:** rounds past a fixpoint are no-ops (but not free on the device: see CUDA graph capture), so read the stop count every few rounds: the MIS undecided count after round 7 and then every 2 (multigrid setup readbacks halved), flip rounds `K = (budget - flips_since_build) // (n_faces / 2) + 1` per read (`delaunay_triangulation` 39 to 19). A deterministic composite skips a stage while the mesh is still that idempotent stage's own last output (`make_solid`).
- **Bellman-Ford over a CSR with duplicate entries** gives the same distances but pays per stored entry: 0.5x at 328 k faces (`geodesic_path` completion graph from vertex-face rows).
- **Fixed-order group sums (§2.2) have three measured forms.** A stable sort by group plus one thread per run serializes on huge groups: 37x the atomic splat on a coarse Poisson level, 40x on coarse `cluster_decimate` cells. Split long runs instead: fixed-size chunks summed in two passes (`screened_poisson`'s splat), or runs above a length cap handed to a block-strided sum (`cluster_decimate`, cap 128). Where a group sum feeds only a threshold or sign, an exact `int64` fixed-point sum (one pass for each group's largest exponent, one integer-atomic pass) needs no sort and stays fully parallel; it beat the single-address float32 atomics it replaced (`make_volume(multibody=True)` 0.50x wall at 1.3 M faces).
- **The shared exact group sum (`scatter_add` + `_launch.group_sums`) against each per-site form, CUDA, setting on (new/old):** it wins on few hot groups (one group: 0.03-0.05x the float atomic; repair's component sums 0.11-0.26x large) and loses on many groups (1.8-2.6x the off-mode atomic at 10 M terms). It loses to the exact `float32` into `float64` atomics (1.5-4.3x; a second pass or word cannot beat one pass), to `arap`'s amortized sorted run walk on scrambled large meshes (1.6-2.0x; 0.74-0.95x ordered), to the Poisson chunked splat at res 256 (1.3-1.4x; 0.56x small) and to the cluster heavy-run sum at 2 M vertices (1.2-1.8x; 0.52x at 100 k). Those sites keep their forms; each carries the decline. What paid inside it: integer scaling of the mantissa (2x over a `float64` multiply), skipping a zero word (one atomic per `float32` term), warp-uniform pre-aggregation (30-40x on one hot slot), per-group fields (1.9x on vec4 lattices); `__match_any_sync` aggregation lost 1.5x. The off-mode branch takes the same warp-uniform pre-sum (`kernels/reduce.warp_lanes_sum`, in `float64`, plain atomic otherwise and always on CPU, so CPU bits are unchanged), off new/old: one group of 10 M terms 0.04x, `remove_small_components` / `make_volume(multibody=True)` 0.22-0.27x on 1 and 64 components at 5-21 M faces, `segment_sum` over 1 000 segments 0.02x; at par (0.96-1.02x) with every face its own component, on 1 M random groups and at 20 k items. A `float32` lane sum for `float32` values measured 0.94-1.00x, not worth a second native. On the CPU device the off path cost 1.15-1.6x a plain atomic per term, and not for the dead pre-sum test (1.02x): the inlined on-mode terms made a kernel whose work is the scatter too large for the CPU compiler to inline into its per-thread loop (a call per thread, varying with the module), and every extra per-term load costs 1.1-1.2x in that latency-bound loop. `scatter_sum` / `segment_sum` now take `kernels/scatter.sole_scatter_add*`, whose on-mode terms run out of line on the CPU only (`kernels/reduce.device_is_cpu` folds, so CUDA inlines them), and every off commit skips the negative-index wrap (`kernels/scatter.atomic_add_at`, the load the mode test adds): 1.03-1.05x a plain atomic, `scatter_sum` off 0.75-0.83x its old CPU time at 10 M terms, CUDA 0.98-1.02x off and on, CPU on and repair's sums at par, bits unchanged. Not for a kernel with work of its own, which is never inlined into that loop: there six out-of-line calls a face cost the on mode 1.26-1.40x (spilled registers). `robust_laplacian`'s rounding rows are now one body over both modes (`kernels/laplacian.intrinsic_rounding_rows`), its mode test hoisted to once a face and its on terms one out-of-line call a face on the CPU (per-term inlined terms ran the CPU off mode 1.07-1.08x the separate off kernel it replaced; per-term calls the on mode 1.26-1.40x): new/old off CPU 0.87x, CUDA host 0.82x at 5 k faces and device 0.89-1.00x; on CPU 0.95-0.98x, CUDA 1.00x. The consumer-fused finish (`read_group_sum`) cut repair's finish-plus-flip device time 0.124 to 0.053 ms at 5.2 M faces; a `@wp.struct` argument now packs in `_launch._Block` (7.8 to 6.5 µs a launch, against 6.0 with no struct).

## 15. Benchmark and measurement traps

- **A sudden slowdown is a Warp rebuild until proven otherwise:** look for `Module hash changed, recompiling` at `LOG_DEBUG` and an idle GPU (§2.5, §3.5). Next suspect host-side per-element Python; fix it on the device, not with a smaller mesh.
- **Attribute against one number:** time the candidate directly (subtraction projections are optimistic by 3-10x), at more than one size (the trend's sign decides) and at the benchmark's own parameter and fixture. Compute the candidate's share before reading a ratio near 1.0, and pin the iteration count before timing an iterative solver.
- **Harness hazards:** cap every group's mesh size first (one pathological row costs the whole JSON). Harness and hand-probe numbers are not comparable; decide on the harness number. A row can depend on mempool state left by its setup. A median at `rounds=3` can hide a one-off cost, so compare it with the minimum. Run `benchmarks/test_meshes.py` after touching the mesh registry.
- **Plan items:** grep the target function, constant comment and benchmark docstring for a recorded decline first; stage-profile before optimizing a stage.
- **A/B:** use a detached worktree, never `git stash` or `sed` sweeps. The editable finder outranks `PYTHONPATH`, so drop it and assert the submodule `__file__`. Check the box is quiet first (otherwise measure counts, not clocks); a failing `nvidia-smi` does not mean CUDA is unusable.
- **Timing hygiene:** interleave A and B in one loop and report the minimum beside the median; baselines drift 10-30 %, so re-run both arms back to back. Batch launches and sync once. Use one pytest process per module. Re-run a failing configuration in a fresh process before blaming the product.
- **`wp.timing_begin` sees no graph-replayed kernels**, so every CG solve reads host-bound. Re-run with capture disabled and carry the device total back. It misses `capture_while` passes too, and `TIMING_KERNEL` omits memcpys: price a fusion that replaces a `wp.copy` with `TIMING_ALL`.
- **Every group has a census row:** scan-mesh rows (`bunny` to `lucy`) run only at `--size medium`, so every scan sweep also runs its ordito rows on `sphere_small` (`CENSUS_MESH`, `benchmarks/conftest.py`), the rows the gate counts; a test with nothing to do on a closed smooth sphere names another small feature mesh with `@pytest.mark.censusmesh(name)` (`saddle_small` for rims, `tangle_torus_small` for creases). A census row that skips leaves its group outside the gate. The census keeps only the smallest mesh of each size axis a test sweeps (`benchmarks/census.py`, `_SIZE_AXES`): of 120 size-axis tests, 91 counted the same at every size and 20 more only ran more rounds (CG iterations, Cholesky panels); the other branches on size are kept with `@pytest.mark.censuskeep` (k-NN deferral, chunked geodesic ball, multi-block polyline reduction, longest-edge split selection). Feature-axis meshes are different inputs and all stay, except a row `@pytest.mark.censusskip` names whose counts equal a cheaper row's by construction (`thickness_interior`'s `max_sphere` on `sphere_med`: in-kernel iterations, 75x `shells_8`'s time for its exact counts). `screened_poisson` keeps depth 7 only. `sample_surface_poisson_disk`'s counts follow its round count, which varies with the setting off, so its row is exempt. The stitch rows assert their band size, since a band of two parallel half-strips once stitched nothing.
- **The uncapped sweep predicts only from failures.** A row whose cell timed out or ran out of memory on a smaller scan mesh is recorded as predicted and not run; a crash predicts nothing (shape, not size). Extrapolating a median by face count is declined: replayed over the 2026-10-23 sweep it skipped measured rows (pyvista grazing `mesh_collision_pairs` on lucy predicted 99 min linearly, took 56 s; its cost follows contacts). Reference rows on the CPU take no warm-up call; device rows do (`conftest.BenchLibrary.run`).
- **Census** host calls by monkeypatching Warp (`Function.__call__`, `wp.array.numpy`, `wp.zeros`, `wp.empty`), not by grepping. Read the slope across two iteration counts, and watch for capture hiding loops.
- **A census count must not depend on what ran before the row.** Two sources made the gate flaky, both now neutralized in `benchmarks/census.py`: Warp pools a `BsrMatrix`'s pinned `nnz` transfer buffer and returns it from a `weakref.finalize`, so cyclic-GC timing decided whether a call allocated (`multigrid_preconditioner` 149-154 allocations between runs; the counted call now runs after `gc.collect()` with the collector off); and `linalg._cached_solver` records a pooled state on the second operator of a shape and replays from the third, so one warm call left the counted call recording or replaying by session history (`heat_geodesic[sphere_med-full]` 18 or 79 launches; two warm calls now). A third crossed rows: `linalg._SHAPES_SEEN` remembers operator shapes process-wide, so a row whose shape an earlier file had solved took a pooled state and paid one `refresh` launch a call (seven `linalg` / `diffuse_tangent_field` rows, recorded with `test_heat.py`'s history baked in); the census now empties the pool before each row. A new process-wide cache that changes what a call issues needs the same check: run the row alone and inside its file, and compare. The census runs under xdist (`-n 4 --dist loadfile` in CI, 2.7x on four cores): the gate passing under a different file-to-process assignment is that check for the whole suite, and the serial and xdist runs agreed on every count of all 1 172 rows.

---

## 16. ordito component status

The full per-component detail is in `reference/component-status.md`; read the component's section there before optimizing or diagnosing it. Its sections: method notes; where time goes; `import ordito`; `reconstruction`; `remesh`/`repair`/`creation`/`bounds`; `array`/`graph`/`polyline`/`intersection`/`boundary`/`levelset`; `proximity`/`metrics`/`neighbors`; `sample`; `smoothing`/`laplacian`/`energies`; sparse assembly; `heat`; `validation`/`adjacency`/`halfedge`; `holes`; `points`/`voxels`; `registration`; preconditioners; the CG solver and `ordito.cholesky`.

### Open defects and traps

- **`screened_poisson(method="octree")` reproduces open3d's single-threaded mesh, up to rounding ties.** Same vertex and triangle counts and 97-99 % of the triangles identical on crowns and the fixtures (`test_poisson_octree_matches_open3d`, CPU and CUDA). The rest are minimal-area splits open3d's own `float32` rounding decides (`MAT.inl`: two splits agreeing to a few `1e-6`; the same surface), plus a handful of far-field vertices where the surface brushes the cube's walls (0.07 cell at most; open3d's wall evaluation was not traced). open3d's default thread count is not a reference: its borrowed face segments then arrive in thread order and 5-6 % of its triangles change from run to run. Further parity needs bit-identical vertex positions, i.e. open3d's float summation order in the solve: not a parallel design.
- **Reproducibility is a property of the deterministic setting (§2.2).** With it off (the library default) every float reduction, scan, scatter and atomic-slot order is arrival-ordered by design, so a CUDA difference there is not a regression; gate any A/B on the CPU device. With it on, every float sum, scan and slot order the setting governs repeats on CUDA (the list is `ordito/config.py`'s docstring; a new one joins it). `log_map`'s angle at the cut locus repeats run to run but is meaningless: it differs between devices and builds, so compare off the antipode.
- **The deterministic setting has no known arrival-order residue on CUDA.** The last ones closed: `quadric_decimate`'s hubs (more than 64 faces: `gather_hub_quadrics`, exact two-word sums in persistent blocks) and branching rims (three or more boundary edges: planes summed in ascending neighbour order), and `robust_laplacian`'s rounding rows (`intrinsic_rounding_rows_exact`); both are in `ordito/config.py`'s list, as is `isotropic_remesh`. A float64 atomic is not exact over a group: a new fix takes one of the §14 group-sum forms.
- **A settled CG iterate is not a solved one.** On ill-conditioned systems (graded meshes) the settle rule stops on round-off. `solve_spd_settled` verifies the componentwise backward error and falls back to a Cholesky factorization held on a caller-owned object. Never trust a residual or a settle test alone for the heat family.
- **A CG that hits its cap or stalls does not warn when `check_every=0`.** `min_quad_with_fixed` therefore runs the verified path (iterate, check backward error, factor from zero on failure). `harmonic` at `k >= 3` uses the direct solver.
- **Never size anything off `matrix.nnz` or a Warp `*_count`**; both are capacities (§3.7). `fem.integrate` / `bsr_axpy` on stale `nnz` once exhausted memory and surfaced as CUDA error 700.
- **`--cpu-blocks` runs the CUDA-only branches on the CPU, and two of them assumed CUDA facts** (fixed 2026-10-26). `mesh_to_mesh_distance` segfaulted in `mark_overfull_leaves`, not in the straggler walks: the node-bitset pass climbed parents from nodes past the CPU SAH tree's built prefix (`bvh_node_count` returned `max_nodes`), and a garbage parent sent its `atomic_or` outside the bitset, corrupting the heap. Behind it, the straggler launch was sized `16 * sm_count` blocks, and the CPU reports `sm_count == 0`, so the tiled pass never ran and stragglers kept the capped pass's partial answer. A branch gated on `prefers_tiled_reduction` must not read device properties the CPU leaves zero, nor BVH nodes past `num_nodes`. The full `--cpu-blocks` pass now completes. Its one failure, `test_icp_point_to_plane_mesh_matches_meshlib[cpu-saddle_graded]` (converged RMS 5.7e-6 against a 4e-6 bound), was the bound: the cost reaches the same floor (~4e-12) as on the plain CPU, and the RMS measures a sliding direction of the saddle the point-to-plane objective barely constrains, which the reduction order moves (0.8e-6 at 200 iterations; plain CPU 2.9e-7 at 50). The bound is now the comparison's own 1e-5.
- **Fail-silently families:** a wrong binary search on an argsort payload returns a valid index of the wrong element (§3.1); `NaN` poisons `side="right"` search; an `int32` scan of large triplet counts wraps (use `int64`); an out-of-bounds write on CPU corrupts the heap later.
- **Heat method on obtuse-heavy meshes is wrong by the method's own error** (and so is potpourri3d's). `geodesic_path` strands on spurious minima and then finishes along mesh edges with a warning. `extend_scalar` warns when the field leaves the source range; `filter_mut_dif_laplacian` and `robust_laplacian` warn about ill-conditioned input. These warnings are the contract, not noise.
- **A degenerate-face threshold that is absolute (`TOLERANCE_MERGE`) removes marching-cubes slivers and opens the mesh**; `marching_cubes(edge_margin=)` is the fix, and every extraction of a level set passes `kernels/levelset.lattice_edge_margin` (`screened_poisson`, `resample_uniform` and `offset_mesh` through `signed_distance_level_set(edge_margin=)`). Lattice nodes landing exactly on the level (a unit box on a 0.05 grid) is the trigger; Warp's `sparse_marching_cubes` has no margin, so a positive margin takes the dense path.
- **Mean-curvature flow pinches thin parts by design**; the plain flow's rebuilt cotangent system then needs orders of magnitude more CG iterations (bunny: 41 to 59 416). `filter_implicit_fairing(conformalized=True)` keeps the input's stiffness within one call (matches libigl's conformalized flow); repeated single-pass calls are always the plain flow.
- **`principal_curvature` overestimating at the default `radius=5` is the quadric fit, not a defect**: it equals `igl.principal_curvature(..., useKring=False)` to 1e-6 (torus tube 3.22 vs exact 2.86; libigl's k-ring default reads 3.12).
- **The grid k-NN's start radius is measured from 1 M queries** (`kernel_neighbors.knn_start_sample`): the volume estimate is about 31 spacings on a surface cloud, and every point of an over-wide first walk is inserted into the row. The cell width is not the lever, and a pre-pass costs one row's serial latency (0.1-0.3 ms) whatever its size, hence the gate.
- **`wp.HashGrid` folds cell coordinates modulo its bins**, so a walk running past one period of the table reads cells a period away. A `k > 1` k-NN grid is sized so no walk from inside the cloud wraps (`neighbors._knn_unfolded_grid`: the bins grow, else the cell); at the old `extent / grid_bins` floor a solid cloud's face rows took the opposite face's points as their `k`-th distance and were deferred with a cloud-wide ball (1 M cube, `k = 2`: 278 ms instead of 1.3). Declined: widening the cell (0.84-0.91x on small-`k` `lucy`) and a far-candidate filter in the walk (0.39-0.53x at `k = 1` off the cloud, whose far rows the folded points hand over early).

### Measured declines: do not re-propose

- Single-block or persistent-block rewrites of BFS, the hole-fill DP or other level-synchronous work: no, a block cannot beat a full grid and Warp has no grid-wide barrier. Blocked wavefront tiling wins only where levels were under-occupied (stitch DP).
- Blocked interval DP for hole fill: no, a plain span level already fills the device.
- Chebyshev multigrid smoother, multigrid or Chebyshev preconditioning for the heat diffusions, block CG over columns, batching CG convergence checks: no, interval robustness, componentwise accuracy or iteration growth.
- `cg(check_every=0)` and `wpl.cg` for ordito solves: no, a fresh conditional graph per call; `_BatchedCg` is the path.
- cuDSS or any one-shot direct backend; per-call factorization caches: no, plan cost dominates and the owner rejects memory held across calls. Factorizations live on caller-owned objects.
- `warp.geometry.tri_tri_adjacency` for `halfedge_twins`: no, quadratic in the largest vertex bucket and cannot carry the validation. Own bucketed pairing is used on CUDA.
- Hard triangle-quality veto, accumulated quadrics, boundary-segment placement or heavier boundary weights in `quadric_decimate`: no, each worsened mean deviation or stalled; the face-quality guard is measured and left off by owner decision.
- Reinstating `warp.fem` for Poisson: no, the narrow-band brick backend replaced it.
- Fusing `sparse_cholesky`'s panel-step launches (re-factoring the diagonal block per row group, lookahead factor in the update or rows launch): no, the factor is device-bound and a tile factor in a launch slows every block of it; a float64 16 x 16 `tile_cholesky` is the floor (5.9 us at 64 lanes, 11.8 at 256).
- A mixed-precision panel factor (float32 `tile_cholesky` + float64 Newton correction of the inverse): no, one step saves 6 % of the factor kernel and loses 3-6 digits at condition 1e3-1e6; two steps are slower than float64.

### Component notes

- **`import ordito` is lazy (PEP 562)**; the guarding test imports in a subprocess and checks that no kernel module loads.
- **Solver state:** one recorded state per operator in a weak-keyed cache; the state must not hold its key. Returned device arrays alias the state and are overwritten by the next solve. Pooled states refresh on every use.
- **CG launch shape:** two launches a round (Chronopoulos-Gear); small systems (at most 1 024 rows) run as one block per column; screened-Poisson storage stays `float32` on purpose, with `float64` block-partial folds.
- **`Trimesh` owns factorizations** (`heat_solver`, `fixed_vertex_solver`; `release_factorizations()`); nothing is factored automatically except on the second heat solve, on a failed settle check (which also factors that mesh's Poisson solves, sharing the heat system's analysis), or for a `k == 2` harmonic / `lscm` system from 4 000 vertices. A negative operator is factored over one analysis (`sparse_cholesky(negated=None)`).
- **`benchmarks/test_cholesky.py::sparse_cholesky_solve` times no solve**: its `solution` is reused across rounds, so from round 2 the warm start passes the residual test at once (a ~30 us replay of copies and one SpMV). Do not read a ratio off it.
- **Poisson:** `dense` multigrid-PCG is the default (`solver_tolerance=1e-5`, non-convergence warns); `method="adaptive"` is the narrow-band backend, which always runs one FAS tau-correction cycle after its cascade (the cascade alone does not converge to the documented system) and lands a few thousandths of a cell from the dense surface; `dense` stays the default for robustness (unchecked band rounds, parity pinned to dense), not accuracy. Where the lattice outruns the samples `adaptive` resolves the same sample-scale dimples dense does. Error is not monotone in depth.
- **Key-sorted CSR everywhere:** structural builds write keys, not triplets; a repeated assembly over fixed faces passes `pattern=`.
- **Orders ordito reproduces itself:** `edges_unique` rows ascend by `(max, min)` key; collapse lock keys, new-vertex numbering and homology tie-breaks read row indices. Do not change `edges_unique` order casually. This is already trimesh's order (`hashable_rows` packs column 0 into the low bits), so `edges_unique`, `face_adjacency`, `subdivide` and `oriented_boundary_edges` equal trimesh's as returned (`boundary_edges` is trimesh's `edges_sorted` rows). igl has no single order to adopt instead: `unique_edge_map` sorts by `(min, max)` keeping first-seen direction, `upsample` numbers midpoints in face-walk discovery order.
- **`quadric_decimate`:** memoryless quadrics gathered in ascending face order (reproducible on CUDA), boundary planes added once per edge, relaxed endpoint-only independence, cost-bucketed lock key, default `feature_angle=180`. Thin parts and slits are kept by vetoing a collapse whose endpoints face opposite ways unless it removes a sliver (mean ratio < 0.05); cost ties move a cell's `i2o` and quality up to 1.4x, so judge a change over several tie perturbations.
- **`ball_pivoting`'s claim key is the *ordered* source pair, seeder first** (`proposal_key`). The
  seed scan truncates at `MAX_SEED_NEIGHBORS`, so two points can each seed with the other as
  partner; an unordered key tied them, both won every vertex claim and both committed, often as the
  same triangle with the same winding, which `resolve_duplicated_faces` rejects (happy_buddha at
  `1.5 x` mean edge, 142-149 points within `2 r`; a 400-point random plane at `r = 4` spacings, 10
  of 10 seeds). The direction bit sits below the unordered pack, so pivot priorities are unchanged.
  `test_ball_pivoting_seeds_no_triangle_twice` asserts on the raw wave-loop soup.
- **Tuned constants live next to their code and are device-gated** (`_LINK_ON_DEVICE_FROM`, `EAR_ONE_BLOCK_MAX`, `RDP_ONE_BLOCK_MAX`, `_DOWNSAMPLE_DOUBLING_FROM`, `_KNN_DEFER_MIN_POINTS`, `_NEAREST_LEAF_SIZES`, `_GEODESIC_BALL_CHUNK`, `PACK_SEGMENTS_KERNEL_FROM`); re-probe after a Warp upgrade.

### Conventions adopted to turn Class B parity into Class A (2026-10-27)

Each costs nothing measurable unless stated; the tests it freed compare buffers directly.

- **Tangent gauge = geometry-central's.** Polar angle 0 is `vertex.halfedge()`: the outgoing
  boundary halfedge on a rim, else the outgoing halfedge of the highest-index face (the ring's
  largest halfedge, `kernels/tangent_space.ring_reference_slot`); `basis_x` is the fan fit
  `fan_anchor` the geodesic walk already used. Frames, transport angles, the connection Laplacian
  and transported vectors equal potpourri3d's (6e-7, 1.7e-6 rad, 1.2e-4 on entries up to 279,
  5e-4°). The old ring-start gauge was 7-76° off in the median. The factorized transport moved
  from 5.5e-6 to 1.5e-5 of CG's (`test_reused_operators_give_the_same_transport`, bound 5e-5):
  round-off of the same, unitarily equivalent system.
- **A per-face flip writes `np.fliplr` of the row** (`kernels/repair.write_face_winding`), as
  trimesh (`fix_winding`, `fix_normals`) and igl (`bfs_orient`) do: `make_winding_consistent`
  equals `FF`, `make_normals_outward` equals `fix_normals`. MeshLab's flip swaps a corner pair that
  varies per face, so its comparison stays Class B. `thicken_mesh`'s inner layer swaps corners 0
  and 1, MeshLib's: closed shells equal `makeThickMesh` row for row; an open shell's band numbering
  is MeshLib's half-edge construction and stays a bijection.
- **`remove_duplicated_vertices` numbers survivors by first occurrence**, the order of Open3D, VTK,
  MeshLab and MeshLib (and an already-welded mesh comes back unchanged): a flag, a scan and a
  relabel after the key sort, 1.12-1.20x the key-order weld (flat host cost).
- **`estimate_normals(orientation="neighbors")`** is pytorch3d's SHOT sign rule (at least half the
  neighbour offsets project positively), equal to `disambiguate_directions=True` to 3e-6.
- **Already equal, tests only:** `face_adjacency_angles` / `_projections` / `_convex` and
  `Trimesh.face_adjacency` (trimesh's row order), `subdivide` (trimesh's buffers), `crop_mesh`
  (Open3D's), `truncated_prisms` (trimesh's).
- **Left Class B, by reason:** an output needing more than renumbering (a mesh, point set, row
  set or partition equal up to numbering is Class A through `ordito.equality`, 2026-10-28); an eigenvector sign (`fit_line`, `fit_plane`,
  Open3D / MeshLib normals); a serial queue or ear clipper (flips, `sweep_polygon` caps, MeshLab's
  flip corner); a reference that returns another object (masks, dicts, bitsets).
- **`laplacian.face_weighted_laplacian`** is MeshLab's coordinate-smoothing stencil (neighbours
  weighted by the faces on their edge, the vertex once; a rim vertex averages along the rim, itself
  twice), one row kernel over the `cotmatrix` pattern whose runs are those face counts
  (`kernels/laplacian.pattern_row_on_rim` is shared with the heat method's pin). As
  `laplacian_operator=` it makes `filter_laplacian` and `filter_sharpen` equal MeshLab's filters
  (1.4e-6 at 10 passes). **`filter_sharpen(method="graph_laplacian")`** is Open3D's sharpener
  (`v + w sum (v - v_j)`, repeated), equal to 2.4e-7 at 3 steps; its benchmark group is
  `filter_sharpen_graph_laplacian`.
- **Heat methods are named by library: `method="potpourri3d" | "meshlab" | "igl"`** on
  `heat_operators`, `HeatSolver` and `Trimesh.heat_solver` (default `"potpourri3d"`:
  geometry-central's discretization, what the vector heat method and potpourri3d use); `heat_geodesic(method="meshlab" | "igl")`, default
  `"meshlab"`, runs on `mesh.heat_solver(method=method)`. MeshLab = barycentric mass, Neumann, `h`
  over every face's edges (`kernels/heat.upper_edge_length_sum_and_count(face_edges=)` weights each
  unique edge by its run, the faces on it): Class A to 9.5e-7 of the range. igl = Voronoi mass
  (`kernels/triangles.face_voronoi_areas`, formed in `float64`, rounded to `float32` so the exact
  corner scatter holds), unique-edge `h`, the Neumann heat averaged with the rim-pinned one
  (`HeatSolver.diffuse_pinned`: rim rows identity, rim columns zero, its own factorization slot;
  skipped after one readback on a closed mesh): Class A to 9.1e-7. `log_map`'s radius is the
  distance at geometry-central's timestep, not the MeshLab default's (they differ on open meshes).
- **geometry-central's (potpourri3d's) heat geodesic factors `L + 1e-6 I`**, not the singular
  Laplacian: rebuilt in SciPy with that term it equals potpourri3d to 1e-14, and without it the gap
  is exactly ordito's (1.9e-5 / 6.7e-5 of the range). The term stops the method converging, since
  `L`'s smallest non-zero eigenvalue falls with resolution: on unit icospheres, max error against
  the exact distance, ordito / potpourri3d, 0.051 / 0.051 at 642 vertices, 0.019 / 0.020 at 10 k,
  0.014 / 0.018 at 41 k, 0.011 / 0.027 at 164 k. Not adopted: that comparison stays Class C.
- **`filter_sharpen`'s default operator is `face_weighted_laplacian`**, so the default is MeshLab's
  `apply_coord_unsharp_mask` (Class A); the plain 1-ring mean it used before matched no library.
- **`estimate_normals` defaults to `orientation="neighbors"`** (pytorch3d's SHOT rule, Class A);
  `"centroid"` (outward from the centroid, ordito's own, what a reconstruction needs) is explicit at
  every in-repo caller and in the guides. `camera_location` is MeshLab's `flipflag` toward `viewpos`
  and `orient_reference` Open3D's `orient_normals_to_align_with_direction`, both Class A.
- **Dropped as a partial port:** `flip_by_objective(objective="curvature")` (a dihedral rule where
  MeshLab's filter decides on a vertex-curvature measure, `curvtype`). `polyline_upsample`'s
  `"floor"` was dropped and then restored by owner decision (2026-10-28): its 2026-10-26 entry
  below stands.
- **`cluster_decimate(method="open3d" | "meshlab")`, `contraction="average" | "closest"`
  (2026-10-28).** Both libraries keep a face and its reversed twin and merge only same-winding
  repeats; ordito merged by sorted triple, so on a rotated sphere Open3D returned 408 / 94 faces to
  ordito's 402 / 93 (the unrotated icosphere the old test used never made a twin). The face table
  now keys the triple rotated to its smallest cell (`kernels/array.rotate_min_first`), faces are
  written that way (Open3D's rows) and ordered by that key. `"meshlab"` is vcg's
  `Clustering::Init` grid (box grown by one cell a side, `floor(extent / c)` stretched cells per
  axis, per-axis `inverse_size`) and its `AverageColorCell` mean over face *corners*; the corner
  items reuse the vertex scatter and the ordered sort (`corner_items`, `4 * 3 n_faces` staging).
  Equal to `meshing_decimation_clustering` (faces as sets, vertices 4e-6) and, with `"closest"`, to
  `generate_sampling_clustered_vertex('Closest to center')` exactly; `"open3d"` equals
  `simplify_vertex_clustering` (1e-6). Benchmark group `cluster_decimate_meshlab`. Not adopted:
  keeping cells whose faces all collapsed (both libraries do) and Open3D's first-occurrence vertex
  numbering; both return faces in hash order, so the face comparison stays Class B.
- **`oriented_bounding_box` keeps `refine_iterations=8` by default** although `0` is the Class A
  one (igl's sampled search, the same candidates): unrefined, a turned `1 x 2 x 4` box comes back
  23 % too large at 4096 rotations; refined it is exact, and on stretched clouds it reaches the
  hull-face minimizers' volume (trimesh, Open3D) to a few parts in 1e5, the frames differing from
  Open3D's by about 0.25° (a flat minimum). No library refines; the owner decides the default.

### Class C comparisons upgraded, and the ones that cannot be (2026-10-28)

Every option below is opt-in; each default is unchanged in outputs and launch counts. Options that
reproduce a library are named for it (`"potpourri3d"`, `"meshlab"`, `"meshlib"`; check 21 exempts
the quoted `"meshlib"` value only), and their kernels describe behaviour, not the reference source.

- **Mislabelled, already Class A:** the three ICP-vs-Open3D tests (transformed sources within
  8e-6 / 3e-6 / 7.3e-7). **Test-side only:** MeshLib's sky-view factor equals `1 - ambient_occlusion`
  bit for bit when handed ordito's Fibonacci lattice in the `+z` bundle frame (`any_perpendicular`
  gives `(-l1, l0, l2)`) and the origins stepped `1e-4` of the scene diagonal; and
  `signed_distance_level_set` on `igl.voxel_grid`'s lattice is `igl.offset_surface`'s mesh
  (levels 0, -0.05, 0.05, vertices matched by the lattice edge each lies on; at 0.1 one vertex is
  4.8e-5 off). `resample_uniform` reaches igl's lattice under `method="igl"` (second pass, below).
- **`filter_two_step(method="meshlab")`:** MeshLab's normal pass averages each face with the faces
  of *lower index* around its corners (weights `(c - sigma)^2`, `sigma = max(cos, 0)`), so its
  answer depends on face order; that, not the fit, rounds a cube's corners inward (the whole ring
  reads RMS 0.0112, the default 0.0114, MeshLab 0.0244). 2.5e-7 on the cube, 1.4e-5 on a
  scrambled icosphere (CUDA).
- **`filter_normals` / `filter_two_step(method="meshlib")`:** Zhang et al.'s Mumford-Shah
  denoiser; the face system takes each pair's weight from the *higher-numbered* face's perimeter
  (the reference's LDLT reads one triangle of a non-symmetric matrix: 8.9e-8 that way, 9.6e-4 for
  the full solve). Within 1.3e-6. Batched CG at tol 1e-10 on the face system; the vertex system
  factored once.
- **`heat_signed_distance(method="potpourri3d")`:** an edge (Crouzeix-Raviart) vector field at
  `t = (h/2)^2`, sourced on the curve's edges; `L_CR` is Hermitian PSD, so CG/Cholesky apply.
  Within 1e-7; ~2.5x the default on CUDA. The old Class C test also handed potpourri3d an open
  curve (closing it: 4.4 % -> 3.7 % of the diagonal).
- **`log_map(method="potpourri3d")`:** the outward-ball radial field and the transported reference
  solved as one paired system. Class A on `saddle_graded` and `sphere_well_shaped` (1.5e-5 worst);
  6e-5 to 1.8e-4 on the needle fixtures, from the shared connection's `float32` transport angles
  (a `float64` prototype rounded there moves 4.4e-5). Fixing that changes the default's system.
- **`cluster_decimate(method="meshlib", contraction="closest")`:** exact `float32` box,
  `ceil(extent / voxel)` cells capped at 1024 an axis, the maximum clamped into the last cell,
  closest vertex by an unfused (`__fmul_rn`/`__fadd_rn`) squared distance: with an FMA, CUDA
  picked 4 of 128 cell representatives differently on ties at the centre. Exact set equality.
- **`relax_approx(method="meshlib")`:** balls of edge-path length (`float32` sums, inclusive
  radius) rebuilt each pass, fit in `float64` in its own kernel (`float32` `svd3` tilted a plane
  by 4e-4; a precision switch inside the default kernel moved it by an ulp at the clamp). Within
  4.8e-7, quadric exact; 2.1x the default at 1 pass, 5-6x at 3. Planar passes on
  `torus_irregular_holes` amplify ulps into different balls (2e-3 at 5 passes).
- **Measured declines:** hole-fill ties against MeshLib need a `float64` DP, MeshLib's triangle
  terms bit for bit, descending `k`, *and* MeshLib's smallest-EdgeId start vertex (not
  reproducible in general); stitch ties follow `std::priority_queue` order. Earcut's order is
  serial (and trimesh here falls through to manifold3d, `mapbox_earcut` not installed). MeshLab's
  ambient occlusion is Embree with no vendored source. The rest of Class C are independent RNGs,
  serial priority queues, or comparisons against the truth.

### Class D exemptions upgraded, and the ones that stay (2026-10-28)

Seven of the 28 Class D items (25 `noparity` rows, 3 Class D tests) became Class A; four were
measurement artefacts, so only tests changed. vcglib / MeshLab source is GPL and not vendored:
it was read for the rule only, and each kernel implements the rule, not the source.

- **Measurement artefacts, test-only:** MeshLib's `relax` is `filter_laplacian(lamb=force,
  volume_constraint=False)` over `laplacian(symmetric=True)` (6e-8, open and closed); the recorded
  "cosine 0.45, minimum -1.0" was the volume constraint left on. MeshLab's
  `apply_normal_smoothing_per_face` is one `filter_normals(threshold=180)` pass (3e-7 closed); a
  rim face counts its own normal once more per rim edge (MeshLab's neighbour across a boundary
  edge is the face itself; emulated to 3e-16; the default is unchanged, and the test pins the rim
  from both sides). MeshLab's erosion is `shrink_vertex_mask` exactly, vertex and face selections
  at every hop, once the start is `compute_selection_transfer_vertex_to_face(inclusive=True)`;
  `inclusive=False` lags one hop, which was the recorded 51 / 39 / 25.
- **`filter_humphrey(method="meshlab")`:** vcg's HC adds the neighbours' mean correction where
  Vollmer et al. subtract it (`q - beta b + (1 - beta) L b`, `alpha = 0`, `beta = 0.5`, one pass
  a call, symmetric 1-ring): 2e-16 in NumPy, Class A in ordito. A sign in the existing update kernel
  (`1 * L b` is exact, so the default keeps its bits). Group `filter_humphrey_meshlab`.
- **`remove_non_manifold_faces(method="meshlab")`:** vcg's `RemoveNonManifoldFace` visits the
  faces on a crowded edge in ascending area and deletes one while an edge of it still has more
  than two live faces. Run as rounds: a face decides once no smaller undecided face shares a
  crowded edge (deciders in one round share none), two launches and one readback a round, integer
  atomics only. The key is the area of the *sorted* corner triple, so a face and its copies tie
  bit for bit on both devices and the index decides (the natural-order area made CUDA and CPU keep
  different windings). MeshLab's own tie between a face and its reversed copy is its unstable
  sort's, so that input compares as unordered triples; fins and same-winding copies are Class A
  after `meshing_remove_unreferenced_vertices` (MeshLab leaves deleted fins' apexes). The default
  drops every face on a crowded edge, which no library does. Group `remove_non_manifold_faces_meshlab`.
- **`face_defective_mask(normal_criterion="meshlab")`:** `nfratio` is the largest dihedral to
  any edge-neighbour, `max_angle` that `neighbor_normals` already formed: one radians threshold
  on `face_defective`. Equal selections on a reversed icosphere, `sphere_irregular` and
  `saddle_graded` at three thresholds each. The consensus default stays: it finds the culprit.
- **Then to A, B or C, test-only (same day):** `mesh_collision_pairs` / pyvista is Class A in
  general position (`vtkCollisionDetectionFilter`'s all-contacts pairs are the crossing pairs;
  the old "2 600 hits" was a mesh against its identical copy, where VTK counts coplanar twins).
  `annulus` / pymeshlab is Class B: MeshLab's disk is ordito's top cap, vertices to 3e-7, each
  quad split along the other diagonal. `fill_min_weight` / trimesh: trimesh fills only 3- and
  4-edge holes; a triangle hole is Class A, quads Class B through the same quad merge. Class C:
  VTK's in-place smoother at `relaxation_factor=0.01` (gap / displacement 0.0040, linear in the
  factor, bound 0.015, the face-weighted stencil reads 0.070); MeshLab's pointwise curvatures
  against the ball measure over `pi r^2` on a 64 x 32 exact torus at two mean edges (median gap
  1.5 % of the range, bound 5 %; the 400-point `torus_round` is too coarse, 3-5x margin); MeshLib's
  distance volume on an `icosphere(5)` cloud at depth 6, outer shell after origin plus half a voxel
  (`gridToMesh` puts voxel centres on integers), worst gap 0.285 voxel, bound 0.9; the mutations
  (no half shift, one-voxel move) clear it by only 1.2-1.3x. `symmetric_chamfer` at its default
  sample count could not tell these surfaces apart (its floor is the sample spacing):
  `symmetric_surface_distance` measures against the surface.
- **Stay Class D, by reason:** stochastic (`random_hills`, `icp_mesh` / pymeshlab); serial
  queues and a gap the row reports (`isotropic_remesh` / pymeshlab); different quantities
  (`fit_line` / pyvista: `pv.fit_line_to_points` is the centred principal axis, which
  `principal_axes` already compares in its own group, so a `fit_line(centered=)` would duplicate
  it -- declined; `filter_mut_dif_laplacian` / open3d and pymeshlab). Class D tests kept: igl's
  arbitrary joined pair in `split_non_manifold_vertices` (ordito's split is order-independent) and
  MeshLib's inclusive fold threshold at exactly 90 / 180.

### Second pass over the non-A pairs (2026-10-28)

- **Double tests of wrappers, now Class A or B:** pymeshlab's harmonic and LSCM filters are igl
  with the libigl tutorial boundary (the longest loop from its first vertex onto the circle by
  arc length; pins `loop[0]`, `loop[len // 2]` at `(0, 0)`, `(1, 0)`), 3e-8 on `saddle_graded`:
  `test_harmonic_matches_pymeshlab`, `test_lscm_matches_pymeshlab` (Class A).
  `Trimesh.vertex_normals` is `vertex_normals(weighting="angle")` (Class A).
- **`principal_curvature(method="pyvista")`** is VTK's `curvature("maximum" / "minimum")`:
  `H +- sqrt(H^2 - K)` (`H` alone where negative) over `vtkCurvatures`' 1-ring `K` (three times
  the defect over the incident area) and `H` (each edge of exactly two faces adds
  `3 |e| theta / (A_f + A_g)` to both ends, `theta` signed by `(n_f x n_g) . e` with `e` walked by
  the lower face, half the mean over the vertex's such edges); zero directions. One thread per
  vertex over its face row and the halfedge mates, in `float64`: Class A at `1e-5` of the field's
  scale on every fixture, `saddle_graded` included (2.1e-7; a `float32` composition of ordito's
  tables lost 0.03 there to a needle's 1e-6 lumped area). Group `principal_curvature_pyvista` (the
  pyvista row left `principal_curvature`).
- **`curvature.vertex_curvatures`** (new) is Meyer et al.'s pointwise `(mean, gaussian)` as vcg's
  `UpdateCurvature::MeanAndGaussian` forms it (MeshLab's
  `compute_scalar_by_discrete_curvature_per_vertex`): the mixed area (non-obtuse: the cotangent
  form; obtuse: a quarter of the area at the obtuse corner, an eighth at the others), the defect
  minus, at a vertex with one rim, the angle between its rim edges, `H = +-|L x| / A` signed by
  the area-weighted normal, zero where `A <= DBL_EPSILON`; a face with an exactly zero angle keeps
  its area and nothing else. One thread per vertex in `float64`, no atomics. MeshLab rounds its
  angles and areas to `float`, which no transcription reproduced (a `float` emulation of the
  rounding read no closer), so the test is Class B: off the rims `K` to 6.4e-6 and `H` to 3.8e-6
  of the field's scale; on rims `|H|` to 1.45e-5 (bound 5e-5), its sign a near-tie (the Laplacian
  lies almost in the tangent plane). `saddle_graded` is excluded: MeshLab's `float` over a needle's
  1e-6 area moves `H` by 8.5 % of the field. Group `vertex_curvatures`.
  `principal_curvature(method="meyer")` is the pair those imply (`extreme_curvatures`, the rule
  VTK's method shares), zero directions, one kernel over the same per-vertex walk
  (`kernels/curvature.meyer_curvatures`); Class B against MeshLab's `H` and `K` off the rims, where
  the sign of `H` decides which value is the larger. Group `principal_curvature_meyer` (its pymeshlab row
  times the `H` / `K` filter, the same work as `vertex_curvatures`').
- **`resolve_duplicated_faces(method="cancel" | "igl" | "open3d" | "meshlab")`.** The owner's
  raise stays the default and is the right default: a surplus of two same-winding copies is
  geometry the caller has duplicated, and igl's release build silently deletes all of it (an
  assert in debug), opening a hole. `"igl"` drops such a group (Class A with the group planted);
  `"open3d"` keeps the first face per rotation-invariant, winding-sensitive key in input order
  (Class A, `remove_duplicated_triangles` row for row); `"meshlab"` keeps the first face per
  vertex set in input order. MeshLab's survivor is whichever its unstable `std::sort` leaves last
  (measured: varies group to group on opposite-winding pairs), so that pair stays Class B on sorted
  rows. Groups `resolve_duplicated_faces_open3d`, `_meshlab`.
- **`filter_windowed_sinc`** is VTK's `vtkWindowedSincPolyDataFilter` (`pv.smooth_taubin`):
  Nuttall window, the Newton-searched pass-band offset (warns when it fails, always at one
  iteration), stencils from the `O1` / `O2` analysis without feature edges (ring; rim pair under
  `edge_angle`; fixed on a crowded edge, one or more than two rim edges, 255+ faces or
  unreferenced). A fixed vertex is scaled by the filter's response at zero, as VTK does. Class A
  (4e-8 / 1.2e-7) on four fixtures, both `rim_smoothing` settings; group `filter_windowed_sinc`
  (the pyvista row left `filter_taubin`).
- **`resample_uniform(method="igl")`** builds `igl::voxel_grid(V, offset, s, 1)`'s lattice on the
  host (`s = round(longest_extent / voxel_size)`) and returns the extraction as marched: Class A
  against `igl.offset_surface` at offsets 0 and -0.05 through `meshes_allclose`.
- **`rasterize_attribute` / `rasterize_discrete_attribute(subpixel_bits=)`** snap the UVs to
  `1 / 2 ** bits` of a pixel (`wp.round`) before rasterizing: at 8 bits the NVIDIA OpenGL image
  equals ordito's at the standard tolerance (3.9e-7 in NumPy; 6 / 10 bits read 1.8e-3 / 3.4e-4 at
  64 px). The discrete labels then agree everywhere but exact ties.
- **`filter_two_step(method="meshlab")` runs in `float64`** (pymeshlab holds `double`): with
  MeshLab handed the same `float32`-rounded input, 3.0e-8 on the cube and 5.7e-8 on the scrambled
  icosphere, both devices (was 1.4e-5 on CUDA, a 1.4x margin; the unrounded input alone moves
  MeshLab 3.8e-6). Price, CUDA, best of 5 against HEAD in a detached worktree: 1.61x at 20 k faces,
  1.20x at 328 k. The default stays `float32`.
- **`icp(method="pyvista")`** is `vtkIterativeClosestPointTransform` as `align` runs it: every
  `n // 100`-th point a landmark (100-199 of them), the source first translated onto the target's
  centroid (both over every point), Kabsch on the landmarks' matches composed each round (VTK's
  Horn quaternion is the same rotation; its Jacobi solve moves the first round 3e-9), stop at the
  cap or once a fit moves the landmarks by an RMS of at most `threshold` (VTK's `MeanDistance`,
  RMS mode, measured from VTK's own matrices). One block, one launch, `float64`
  (`kernels/registration.landmark_icp`). Class A to 6.3e-7 (CPU), mesh and cloud targets, at a
  3-iteration cap and converged; group `icp_mesh_pyvista` (the pyvista row left `icp_mesh`). The
  match must be the *exact* nearest face: `mesh_query_point_no_sign` returned a face a few 1e-5
  farther than the nearest on `saddle_graded`'s needles, one landmark switched faces at round
  110, and the converged answer missed VTK's by 1.4e-4 (stopping at 153 rounds, VTK at 117);
  taking the `float64` minimum over the faces whose boxes meet that face's distance closes it.

### Open items from the 2026-10-28 Class C upgrades

Each is a measured limit of an option above; re-measure before acting on it.

- **`log_map(method="potpourri3d")` is Class A only on well-shaped fixtures.** On the needle
  fixtures it is 6e-5 (`sphere_irregular`, CUDA; 1.8e-4 CPU) and 1.5e-4 (`torus_irregular`) from
  potpourri3d: the connection Laplacian it shares with the default stores its transport angles in
  `float32` (rounding only those angles in a `float64` prototype moves the map 4.4e-5).
  **Proposed:** form the transport in `float64` for both methods, which changes the default's
  vector system and its parity numbers; price it and re-run every vector-heat comparison.
- **`relax_approx(method="meshlib")` costs 2.1x the default at 1 pass and 5.2-5.9x at 3**
  (41 k / 164 k vertices, CUDA): one edge-path walk is about two of the default's breadth-first
  walks, and it rebuilds every pass where the default walks once. **Proposed:** a cheaper walk
  (the label-correcting scan is `O(k^2)` per vertex), priced on the benchmark row. Separately,
  5 planar passes on `torus_irregular_holes` drift 2e-3 from MeshLib though each pass agrees to
  4.8e-7: one-ulp position differences change the next pass's balls. Not reproducible without
  MeshLib's own float order; the test excludes that case and says so.
- **`signed_distance_level_set` against `igl.offset_surface` holds at three of four levels.** At
  `iso = 0.1` the counts and faces agree but one vertex is 4.8e-5 off. Cause (traced 2026-10-28,
  both devices): a near-tangent lattice edge, both nodes within 1.8e-4 of the level, where any
  field or node error is amplified by `h / |df|` ~ 225. Ordito's field there is 2e-7 from igl's,
  but a `float64` field alone does not close it (probed with igl's own double distance as the
  exact field): at the nodes the `float32` `bounds` place (1.2e-7 from igl's) the vertex is still
  2.8e-5 off, a second 4.7e-5 (now 4.1e-5); with `float32` nodes 2.0e-5 / 3.7e-5. Only igl's
  `float64` lattice reproduces them. Both vertices lie on the same surface to ~1e-7, slid along
  it. `0.05` is closed (2026-10-28): two crossings there sit 5e-6 of a cell from one node, nearer
  each other than the tolerance, so the test matches vertices by the lattice edge each lies on, not
  by nearest neighbour (a bijection at all four levels, faces equal, worst vertex 2.6e-5 CUDA /
  1.9e-5 CPU). The test runs `0`, `-0.05` and `0.05`. **Proposed:** for 0.1, a `float64`
  path through the whole extraction (`float64` `bounds`, a public API change; `float64` node
  positions; each node's distance recomputed in `float64` on the face Warp's search finds;
  `float64` interpolation in `marching_cubes`; a route for inputs on the `float32`-only
  `sparse_marching_cubes`), opt-in or priced first (`float64` runs at 1/64 rate, §13.2, under
  `offset_mesh`, `resample_uniform`, `screened_poisson`, `fix_self_intersections`).
- **`heat_signed_distance(method="potpourri3d")` is ~2.5x the default on CUDA** (ring 25.3 vs
  9.8 ms, band 9.3 vs 5.2) with more launches (CPU census 2448 / 958 vs 1749 / 809): the edge
  system's assembly and a longer settle on a system ~3x larger (the operator is already held per
  `HeatSolver`). Not yet stage-profiled.
- **`filter_two_step(method="meshlib")` issues ~410 launches against the default's 127** (CUDA
  census; 1780-1875 on the CPU), and 21 sorts and scans for the vertex factorization's analysis
  per call: the face-system CG and 20 vertex solves. Unpriced against MeshLib beyond the small
  saddles (15 ms vs ~400 ms).
- **Private access added across a module boundary.** The heat half is closed (2026-10-28):
  `HeatSolver.diffuse_tangent_pair` and `HeatSolver.diffuse_edge_field` are public, the solver
  owns the Crouzeix-Raviart system (no weak-keyed module cache), and `log_map` /
  `heat_signed_distance` call them with no `reportPrivateUsage` ignore. Still open: `smoothing`
  reaches `neighbors._EdgePathBallWalk` (check 5 allowlist, plus a scoped `reportUnusedClass`
  ignore), the same shape as the two allowlisted `_GeodesicBallWalk` uses; a public chunked walk
  would retire all three entries at once.
- **Declined, recorded so it is not re-proposed without new evidence:** hole-fill and stitch ties
  against MeshLib (a `float64` DP, MeshLib's triangle-term bits, descending `k` and its start
  vertex; stitch follows its heap order); earcut's serial ear order; MeshLab's Embree ambient
  occlusion (no vendored source).

### Open items from the 2026-10-28 review of the non-A pairs

Each line names the function, the reference and why the comparison is not Class A. The plan is to
go algorithm by algorithm and match the reference wherever its answer is determined.

**A different quantity, or a check against the true answer (Class C).** The comparison is honest
as it stands; an upgrade means adding the reference's own algorithm as an option.

- `heat_geodesic` against `igl.exact_geodesic` (the heat method's own error), MeshLib's
  `computeSurfaceDistances` (fast marching) and pyvista's Dijkstra edge distance (a two-sided
  bound): an upgrade is a fast-marching option (MeshLib, MeshLab and potpourri3d time one).
- `geodesic_path`: an inequality against `igl.exact_geodesic` and a per-cent of potpourri3d's
  `EdgeFlipGeodesicSolver` on a sphere; an upgrade is an edge-flip shortening option.
- `trace_from_vertex` (a trimesh distance bound, no oracle); `shorten_loop` against MeshLib's
  `findShortestEquivalentLoops` (a minimum over a different path space).
- `heat_signed_distance` (default) and its level-set constraint against potpourri3d (correlation
  plus mean error; `method="potpourri3d"` is Class A); `log_map` angles (median 13.2 degrees, two
  discretization errors, `Open items from the 2026-10-22 parity audit`); the cut-locus transport
  (no reference).
- `ambient_occlusion`, `volumetric_obscurance` and `shape_diameter` against pymeshlab (rank
  correlations: MeshLab traces Embree / GPU direction sets, not vendored); `shape_diameter`
  against MeshLib's single ray; `max_tangent_sphere` against MeshLib's shrinking sphere (median
  relative difference: the same method, a different iteration).
- `statistical_outlier_mask` against MeshLib's `findOutliers` (a connectivity criterion);
  `convex_subset_mask` against qhull (soundness plus recall); `voxelize_mesh` against MeshLib's
  narrow-band distance field; `principal_axes` on a degenerate spectrum (one axis defined).
- `screened_poisson` (`dense`) against open3d / pymeshlab (Kazhdan's octree; `method="octree"` is
  the reproduction) and MeshLib's distance volume (another reconstructor).
- `principal_curvature(frame_independent=True)` against igl (a bound; `False` is Class A).
  (`icp_mesh` against pyvista is closed: `icp(method="pyvista")`, `Second pass over the non-A
  pairs`.)
- The defaults of `filter_normals` and `filter_two_step` against MeshLib's and MeshLab's crease
  measures: different algorithms whose library-named methods are Class A.
- Truth checks with no library answer: `resample_uniform` on a sphere's offset; `sample_volume`'s
  first moment and containment.

**Same algorithm, small numeric gap.** Each has its own entry: `log_map(method="potpourri3d")`
(`float32` transport angles) and `relax_approx(method="meshlib")` (planar drift) in
`Open items from the 2026-10-28 Class C upgrades`; `signed_distance_level_set` against
`igl.offset_surface` at iso 0.1 (the lattice's `float32` placement), same section; `screened_poisson(method="octree")` on an
open cloud (`test_poisson_octree_matches_open3d_on_an_open_cloud`: far-field vertices at the
cube's walls, open3d's wall evaluation untraced; on closed clouds the residual ties are open3d's
own `float32` rounding, `Open defects`); `triangulate_point_cloud` without normals
(`Open item from the 2026-10-27 triangulation port`). `filter_two_step(method="meshlab")` is
closed (`float64`, `Second pass over the non-A pairs`).

**Same algorithm, different settings.** `resample_uniform` against pymeshlab
(`generate_resampled_uniform_mesh`: vcg's resampler, its own grid and sign rule; worst sample 1.69
voxels against ordito's 0.48) and MeshLib (`rebuildMesh` decimates after marching); `offset_mesh`
against MeshLib's `offsetMesh` and pymeshlab's uniform resampler at a matched cell size. Each needs
the reference's grid placement, sign rule and post-processing as an option; igl's is done
(`resample_uniform(method="igl")`).

**A rule the reference adds.**

- `rasterize_attribute` against OpenGL: closed (`subpixel_bits=8`).
- MeshLab's pointwise curvature: closed by `curvature.vertex_curvatures` (Class B, its `float`
  rounding; `Second pass over the non-A pairs`). The ball measures `discrete_gaussian_curvature` /
  `discrete_mean_curvature` keep their Class C comparison with it.
- `filter_laplacian` against `vtkSmoothPolyDataFilter`: VTK updates in place, vertex by vertex
  (Gauss-Seidel in index order), so its answer depends on that order; reproducing it is a serial
  sweep (Class C at `relaxation_factor=0.01`).

**MeshLib's "nice" operations (Class C, surface or volume statistics).** `fill_smooth` and
`refill_region`'s refinement against `fillHoleNicely` / `patchMesh`, `stitch_smooth` against
`stitchHolesNicely`, `bridge_edges_smooth` against `makeSmoothBridge` (a different curve
construction) and `smooth_region_boundary` against `smoothRegionBoundary`. Each is a
minimum-weight fill followed by MeshLib's own subdivision schedule (a serial priority queue) and
Laplacian fairing; a port would follow that schedule.

### Open item from the 2026-10-27 octree memory work: a sparse solve

`screened_poisson(method="octree")` at depth 8 now peaks at 484 MiB, down from 2 036 (`7498cad1`
to `59e49998`: `created` a bitset, `int8` corner sources, per-cell passes over the surface band,
`partners` per vertex, the solve's buffers released before the extraction; bit-identical on both
devices, 0.99-1.13x faster). The peak by phase is splat 341, constraints 443, cascade 367,
extraction 484 MiB. **Proposed: make the solve sparse, then the extraction's `owners`.** The
constraints phase is the solver's own dense storage: a normal field over every cell of every depth
(219 MiB) plus the per-depth lattices; storing them over the occupied cells only is a solver
redesign, not an extraction change. The last dense extraction table, the per-edge `owners`
(222 MiB), would take the extraction to about 260 MiB, but alone it buys only about 40 MiB (8 %),
since the constraints phase then sets the peak; it pays once the solve is sparse. Gate either step
on byte-identical output against the dense form (CPU hash A/B at depths 6-8, the sphere at depth 8
pairs 16 vertices) and on time at depth 8.

### Open item from the 2026-10-27 triangulation port: the unoriented path

`triangulate_point_cloud` with trusted normals now assembles its fans as MeshLib's
`triangulatePointCloud` does (oriented candidate classes in its order, the serial builder on the
contested rows, `findHoleComplicatingFaces` rounds, holes by the gap; `4cef27d6`,
`reference/component-status.md`). **Without normals it is unchanged**: runs of two or three fans
whatever their winding, `remove_non_manifold_faces` dropping both of two conflicting faces, then
`_face_away_from` winding each component. MeshLib orients the *fans* first
(`autoOrientLocalTriangulations`: fans face away from the box centre, then a heap flood from the
farthest points flips each neighbour fan to agree) and only then takes the oriented classes and
runs the same builder. **Proposed:** port that orientation, then route the unoriented path through
`_assemble_trusted_fans`. The flood is a priority-ordered serial walk, so price it like the builder
(serial only where fans disagree, or on the CPU device), and gate on
`test_torus_without_normals_matches_meshlib_reference` / `test_open_cloud_without_normals_faces_like_meshlib`
tightening from Class C toward face-set equality.

### Open items from the 2026-10-23 uncapped sweep: ordito on large meshes

The sweep (`.benchmarks/merged-2026-10-23-final/incomplete.jsonl`, ordito-cuda rows) left ordito
with no time on three large-mesh rows; all three are fixed. `ball_pivoting` on happy_buddha (component
notes, the claim key), `relax_approx` on lucy (streamed geodesic balls,
`reference/component-status.md`, "k-NN and ball queries") and `smooth_region` on lucy (a V-cycle
form for large regions, `reference/component-status.md`, smoothing). Cotangent weights, which the
benchmark row does not time, take that form too: dragon 10 948 -> 339 CG iterations (6.4 -> 0.58 s),
lucy 153 s where the polynomial did not finish. **The bare `V D² V` count grew with the region
because squaring a V-cycle squares its error on the smooth modes, not because of the rim rows**
(measured, SciPy/pyamg on icospheres: with exact `L⁻¹` the rim rows cost 26 -> 52 iterations over
16x the size, against 7-9 without them; a V-cycle squared grew 137 -> 1 095 with or without them,
while the same V-cycle alone held 17-21 on `L`). Wrapping each V-cycle in a Chebyshev iteration,
`C = p(V L) V`, keeps `C D² C` near `L⁻¹ D² L⁻¹`, but costs `degree` V-cycles a side: it pays only
on the largest regions (degree 3 from 500 k free vertices cotangent, 2.5 M unit; lucy 153 -> 99 s
and 48 -> 42 s; dragon 2x slower at any degree above 1, since ordito's l1 V-cycle at 8 sweeps
leaves little error to remove -- 338 iterations where pyamg's stand-in needed 5 606). Lucy's
cotangent count is still 730; the exact preconditioner's floor grows too (26 -> 52), so some growth
is inherent to the least-squares rows.

### Open items from the 2026-10-22 parity audit

The audit of every Class B/C/D test (commits `310fb0e8`, `03226107`) left these open. Each declined item gives the reason the agent recorded, then why it might not hold; re-measure before acting either way.

**Re-measured or decided 2026-10-26 (confirmed declines; the public changes are the owner's call):**

- **`resolve_duplicated_faces` keeps raising `ValueError` on a duplicate group it cannot orient** (owner decision, 2026-10-26); igl silently drops the group, and `method="igl"` now does too (2026-10-28, `Second pass over the non-A pairs`). `ears`' `ear_opp` keeps igl's meaning with no release note (owner decision).
- **`flip_t_vertices` takes MeshLab's `meshing_remove_t_vertices` rule** (owner decision, 2026-10-26): a face is a sliver when its *longest edge over the height onto it* exceeds `threshold`, and that longest edge is the one flipped, when the flip raises the worse height-over-longest-edge of the pair (`flip_by_objective(objective="t_vertex")`'s `aspect_threshold` likewise). The old `aspect_ratio` gate (about `(base / height)^2 / 8` on a sliver, so its 40 was MeshLab's ~18, any edge of a sliver) was the whole 130-234-face gap; with the edge-key claim order and convexity guard kept the gap is 4 / 36 faces on `sphere_irregular` (thresholds 40 / 20), 0 / 6 on `torus_irregular` and 0 / 0 on `saddle_graded`. The sphere's 4 are MeshLab flips of non-convex quads that fold it to a 178.3° dihedral; the rest is flip order (a serial ascending sweep of the rule does no better). No in-repo caller of the `t_vertex` objective but `flip_t_vertices`; the repair guide's output is unchanged.
- **`fix_degeneracies` / `collapse_small_triangles` vs MeshLib: confirmed order (2026-10-26).** On the saddle MeshLib keeps 986 vertices and ordito 1020 of 1 156. Most of the count gap is exact ties on the mirror-symmetric rows (the endpoint-cost comparison keeps its first endpoint): jittering the input by 1e-7 / 1e-6 / 1e-5 of the diagonal gives ordito 990 / 994 / 999 against MeshLib 986 / 986 / 1000; not a cap (`max_iter=100` changes nothing). The kept sets still differ by 59-62 vertices (face IoU 0.86-0.87), and ordito's collapse lock key is hashed, not length-ranked (a length rank picked the same collapses on this saddle, `kernels/remesh.collapse_candidates`), so the residual is MeshLib's serial heap order. `collapse_small_triangles` differs by one vertex each way (the `13-45-49` chain); keeping the largest index instead makes it 15, so no survivor convention closes it.
- **`polyline_upsample(method=)`** (owner decision, 2026-10-26): `"floor"` (default, `max(floor(L / step), 1)` pieces, spacing in `[step, 2 step)`) or `"bisect"` (the smallest power of two whose pieces are at most `step`, MeshLib's `subdividePolyline`): `test_upsample_polyline_bisect_matches_meshlib` is Class A over six steps, and the `polyline_upsample` benchmark row runs `"bisect"` so ordito and MeshLib emit the same points. The old docstring's "matching the source implementation" named an internal PyTorch port (`fd484ce4`), nothing a user could check. `polyline_smooth_upsample` keeps `"floor"` (shared `_upsample` body, `bisect=False`).
- **`log_map` angles vs potpourri3d differ by a median 13.2°: confirmed Class C (2026-10-26).** The tangent gauge is now shared (component notes, "Conventions adopted"), so no source rotation separates the two; it was never the gap. Conventions tested on `saddle_graded`: the source rotation as the Euclidean angle between the two `basis_x` (13.19°; the intrinsic offset 13.18°), mirroring (64.6°), the radial direction expressed intrinsically (<0.1°). Against the radial direction of `igl.exact_geodesic`'s distance, read ordito's way, ordito's angles are a median 4.5° off and potpourri3d's 11.0° from the test's source; 6.6-9.5° against 12.7-18.4° on `sphere_irregular`; 2.4° against 0.4° from the saddle's middle, where the two agree to 2.5°. The gap is two radial-direction discretization errors, potpourri3d's usually the larger.
- **`split_faces_along_field`'s "high side first" is a mask, not an order**: it returns one mesh and a `values >= isovalue` mask, the side `clip_mesh_with_field` keeps; only the pyvista test pairs VTK's `(below, above)` tuple against it. Inverting the mask would invert that shared convention for no reference's sake. Callers: 0 in `ordito/`, 5 in `tests/test_intersection.py`, 1 benchmark row, 1 guide example (`docs/examples/cutting.md`) and its own docstring.
- **Boundary loop direction: keep it.** ordito's direction *is* igl's (`boundary_loop`, `boundary_loop_all`: exact `array_equal`); trimesh's outline is direction-agnostic; only MeshLib's `getLeftRing` walks the other way (it walks the hole). Flipping would touch 15 library consumers (every filler, `extend_hole`, the stitchers, `_cap_section_loops`, `map_vertices_to_circle` and the maps chained on it, `heat_signed_distance`, `loop_directed_areas`), 9 docstring outputs, 20 tests and 3 guide pages; 11 consumers, 60 tests and every benchmark are agnostic.
- **Dense `0..k-1` component labels: priced, not adopted.** igl's label is exactly the rank of ordito's (smallest face / node id), now asserted in both igl tests. A dense output costs a root-flag launch, an `int32` scan and a gather: 1.9x the call at 328 k faces (host-bound), 1.03x at 5.2 M.
- **icp_mesh / pymeshlab stays `noparity`**: 16 fresh processes, two working directories, never converged (RMS 0.043-0.049 at 0.15 rad, 0.1035-0.1066 at 0.30) and moved run to run.
- **polyharmonic `k = 3` keeps `1e-4`** (CPU 6.8e-5, CUDA 9.5e-6) and the 100 k circle map keeps `1e-4` (CPU 3.0e-5, CUDA 2.3e-6).

**Verification after the audit and the 2026-10-26 fixes:** the two-process `python -m tests.devices --cpu-blocks` (CUDA 4 835, CPU 4 732, CPU blocks 4 732 passed; no failure), both census budgets strict, `zensical build --strict`, `basedpyright` 0 errors, 664 parity pairs (637 covered, 0 uncovered).
