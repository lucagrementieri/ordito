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
- A Warp version claim is spelled `Warp 1.17` (the word "Warp" immediately before the number); the
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
- Do not call `wp.constant()` (§12.6). The typed constructor is what matters: `TOLERANCE_MERGE = wp.float32(1e-8)`. A plain Python float is `float32` in kernels and mixing it with `float64` is a parse error; use `wp.float64(...)`.
- Prefer dtype-generic `@wp.func`s (`wp.Float` / `wp.Scalar`, `Any` for vectors; the `kernels/predicates.py` convention). Limits: §12.4. `Any` is also generic over rank and dimension, so do not pin a helper to one; use rank-free reductions (`wp.min(wp.abs(wp.get_diag(r))) < tol`, since a matrix has no `.shape` in kernels).

### 1.3 Casts and conversions

- Cast `wp.tid()` when used as an index: `f = wp.int32(wp.tid())` (check 22; multi-index unpacks cannot carry a cast).
- `wp.int32` / `wp.float32` are the only cast spelling; never bare `int(...)` / `float(...)` (check 16). `float(...)` fails inside `wp.Float`-generic functions; use `type(x)(...)` where a function is or could be generic (§12.5).
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

### 2.3 Block-per-item is an occupancy trade, not a shape choice

Use one block per item only when the outer per-item dimension alone would starve the device. A kernel that already has a slice dimension does not qualify; collapsing it into lanes loses 2-8x. Compute the launch's thread count and measure at the large end first, since the small end flatters. Verdicts: §14.1.

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
- `test_generic_kernels_register_their_overloads` only checks that some overload exists; a missing dtype shows up as a slowdown (§15.1).
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
module, so every new dtype pair rebuilds that module (§12.6). A kernel created lazily must go in its
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
  (§12.7).
- Any Warp `*_count` / `.nnz` field is a capacity until proven otherwise
  (`Volume.get_voxel_count()`; use `get_active_stats().voxel_count`).

More `warp.sparse` behaviour: §12.7.

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
  `ordito.reduce` or `wp.utils.array_sum`. Decide on the CUDA measurement (§9, §14.5), but keep the
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
  (`face_adjacency(n_vertices=...)`) and pass it from every in-repo caller that knows it (§14.6).
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
- Every public `ordito/<module>.py` has `tests/test_<module>.py` and `benchmarks/test_<module>.py` (check 4); tests live in the file mirroring the function's module (§5).

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
5. Its `docs/SUMMARY.md` entry (check 28) and every cross-reference (`zensical build --strict`).

Renaming a keyword argument also needs a grep of the old name alone (`**splat` `TypedDict` fields) and of fenced docstring examples (check 12). After any rename re-derive the residual set and run the whole suite.

### 4.5 The mechanical gate: `tests/api_conventions.py`

Thirty checks fail the default `pytest` run. Read an allowlist entry's reason before adding one; prefer fixing the code.

1. Summary line naming a reference library (§6).
2. `*_mask` producer not returning `wp.array[wp.bool]`.
3. Module summary ending in `(Warp)` / `on NVIDIA Warp`.
4. Module without both a `tests/` and a `benchmarks/` file.
5. Private name reached across a module boundary (a new `_*.py` for one helper dodges it).
6. One public name exported by two modules.
7. `kernels/<name>.py` without `ordito/<name>.py`, or the reverse.
8. Private helper defined above its first caller (§5).
9. Comment or docstring blaming a Warp version older than installed (anchored `Warp 1.17` spelling).
10. Allocation with no `device=` (§3.3).
11. Public function that raises with no `Raises` block (§4.3).
12. Fenced ```python docstring example that does not run.
13. Kernel `out_` prefix and position (§2.1).
14. Non-subscript array annotations (§1.2).
15. `wp.launch` with no `device=` (§3.9).
16. Bare `int(...)` / `float(...)` in kernel scope (§1.3).
17. `/` on declared-integer operands (§1.5).
18. Bare `bool` / `int` / `float` kernel annotation (§1.2).
19. Reference-library comparison test with no class label (§7.4).
20. Kernel-scope ternary instead of `wp.where` (§1.5).
21. MeshLib or promesh name under `ordito/` (§7.6).
22. Bare single-index `wp.tid()` (§1.3).
23. `wp.map`'d `@wp.func` with no declaration table (§3.5).
24. `.claude/CLAUDE.md` cross-reference to a nonexistent section, or a bare chapter number where that chapter has numbered subsections.
25. `!!!` admonition inside a numpydoc item-list section (§6).
26. Warp-typed module constant used as a Python-scope operand or slice bound (§13.1).
27. `bsr_from_triplets` / `bsr_set_from_triplets` under `ordito/` (§3.7).
28. Public module missing from, duplicated in, or stale in `docs/SUMMARY.md` (§6).
29. Array docstring entry opening with `Length-`, `Shape ``(`, `Flat` or `Rank-N ``(` (§6).
30. `warp.utils.array_cast` under `ordito/` (§3.6).

Two rules about the gate itself:

- A new Warp construct (first `@wp.struct`, `wp.ref`, tile intrinsic, or even a plain `@wp.func` extraction) can silently hide buffers from a check that pattern-matches AST nodes; after introducing one, grep `tests/api_conventions.py` for such checks. §2.4's extraction wins over the check.
- Write checks with a staleness half: an allowlist entry that matches nothing fails.

---

## 5. Function ordering within a module

Source order is the rendered docs order.

### Python wrapper modules (`ordito/*.py`)

- Layout: docstring, imports, constants / aliases, functions.
- Group public functions thematically, most important first; the simple form precedes its variants.
- Private helpers (check 8): one caller means immediately after it; several means after the last; cross-cutting utilities go in a trailing section. Never above the first caller.

### Kernel modules (`ordito/kernels/*.py`)

The rule inverts: a `@wp.func` must textually precede every kernel or `@wp.func` calling it. Mirror the wrapper's public order, place helpers immediately before their kernel, and a shared helper before its first user.

### Tests (`tests/test_<module>.py`)

Mirror the wrapper module's public-function order.

---

## 6. Documentation (Zensical + mkdocstrings)

Docs use Zensical with mkdocstrings, configured in `mkdocs.yml`.

```bash
uv run zensical build --strict --clean  # validate; --clean after docstring edits
uv run zensical serve                   # preview; restart to see docstring changes
```

- API pages come from `api-autonav`, one per public module. The nav is `docs/SUMMARY.md`; a new module needs a line (check 28). Never use `gen-files`: Zensical ignores unsupported plugins silently.
- A build finishing in about 0.1 s with an empty `site/` is inotify exhaustion (exits 0); raise `fs.inotify.max_user_watches`.

### Docstring content

- The summary line says what the function returns, never the C++ call it wraps (checks 1, 3); attribution goes in `Notes` or `See Also`.
- An array entry (every `Parameters` / `Returns` / `Yields` / `Attributes` entry, and a `Trimesh` property summary) opens with its shape as a code span, then its meaning: ``` ``(3 * n_faces,)`` flat triangle index buffer. ```
    - A rank-1 length is a 1-tuple (``` ``(n_vertices + 1,)`` ```); no `Length-`, `Shape`, `Flat`, `Rank-1`, or shape buried mid-sentence (check 29).
    - Spell sizes in full (`n_faces`); define a free size (`m`, `k`) on first use.
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
- While developing, run `uv run python -m pytest tests/test_<module>.py -q --device=cuda` only.
- Use `tests.devices` when a change is device-dependent by construction (`launch_tiled` kernel (§12.2),
  `_device.prefers_tiled_reduction`, device-gated constant (§13.3)).
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
    - **A**: direct `np.allclose` / `np.array_equal`. The default.
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
- Licensing: nothing under `ordito/` may name MeshLib or promesh (check 21); pymeshfix and pymeshlab
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
  checkout in the live tree (§15.6).

### Coverage

- `ordito/kernels/` is omitted from coverage on purpose (kernel bodies are never run as Python).
  Never write tests to raise a kernel-module figure.
- The badge is the CPU wrapper layer. Do not raise the floor to chase `device.is_cuda` branches;
  they belong to §7.2's two-process job.
- Coverage gates wrapper branches only; it says nothing about vacuous comparisons (§7.4).

## 9. Performance work: measure before you change

Numbers live in §13 (cost model), §14 (kernel shapes) and §15 (measurement traps); component status
is in `reference/component-status.md`.

- Something suddenly slow is a Warp rebuild until proven otherwise (§15.1).
- A benchmark lands before the optimization; never restructure for speed without one timing the
  current implementation.
- Attribute a cost with one direct measurement at the benchmark's own operating point and at more
  than one size (§15.2, §15.3). A share that falls as the input grows is a decline.
- Check whether the function graph-captures before reading device time (§15.10).
- Before proposing an optimization, grep the call site, constants and benchmark docstring for a
  written decline (§15.5).
- Attribute a change only with an interleaved A/B in one session, using a detached worktree
  (§15.6, §15.7). A probe that instruments what it measures carries a do-nothing control arm.
- CUDA is the target: decide on the CUDA number, measure both devices, and keep the CPU path
  correct (§14.5). Split a tuning constant per device when the optima differ (§13.3).
- A decline is a result: write it at the site with the number, and resolve every site the finding
  named.
- Count the launches a numerical-method change adds per iteration (§14.8).

## 10. Warp API reference mirrors

Warp function lists are mirrored under `reference/warp_api/`: `builtins.md` (kernel-scope
builtins), `warp.md` (Python scope), `sparse.md`, `utils.md`, `fem_linalg.md`.

- Before using an unfamiliar Warp builtin, sparse or utils function, `grep` these files to confirm
  name, signature and scope. `uv run reference/warp_api/warp_version.py` compares the version
  stamps with the installed `warp-lang`; `REGENERATE.md` explains how to re-extract.
- A name in `dir(wp)` missing from the mirrors is usually hidden on purpose. Introspect first:
  `warp._src.context.builtin_functions[name]` exposes `.hidden`, `.doc` and `.input_types`.
- Then check the quantity (storage class, precision), not the name. Adoption verdicts: §12.8.

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
- Warp has no grid-wide barrier, no per-cell `HashGrid` entry point and no node-by-node BVH traversal.
- `wp.tile_bvh_query_aabb` returns out-of-range indices when a round overruns its result buffer (unchanged on 1.18). Bound-check every candidate (`0 <= c < n`); dropped primitives are lost, so each caller needs a second sound bound.
- A multi-launch `capture_while` body replays differently on CPU, and Python-level ping-pong cannot help a captured loop.

### 12.3 `wp.ref[T]` requires concrete types

Generic type-vars do not instantiate inside `wp.ref[...]`; use `wp.ref[wp.float32]`. `@wp.func` cannot be overloaded by name and `arr[i], arr[j] = ...` swaps raise, so the shared argmin/argmax helpers are concrete float32 and float64 sites keep loops.

### 12.4 Numerics and precision

- `wp.Scalar` does not instantiate for `wp.bool`; write the concrete bool kernel rather than widening to int32 (~2x). Scalar arguments must match the input precision (`a.dtype(tol)`); literals in a generic `@wp.func` need `type(x)(...)`.
- Sum cross products about a point of the loop, never the origin (`predicates.newell_term`).
- FMA fusion makes a degenerate triangle's area ~1e-8 on CUDA and 0 on CPU. Do not disable it; write degeneracy tests at small scale and never assume CPU/CUDA bit agreement.
- Form area vectors and cotangents where they do not cancel (`predicates.triangle_area_vector` crosses at the largest angle); Heron needs sorted sides. Float32 lengths cannot represent a needle (the intrinsic path warns).
- Never substitute an algebraic identity inside a sign test that feeds a branch; CUDA differs, CPU does not.
- `mesh_query_point_no_sign` is inexact (~2e-5, the point worse than the distance); `length < r` and `length_sq < r*r` differ in float32 (§2.4); NaN breaks `searchsorted(side="right")` (§3.1).
- Float32 storage sets a noise floor no tolerance reaches; heat fields scale as 1/scale², so absolute tolerances on them are wrong.
- The backward pass of an empty dynamic `range` runs one iteration past the array: a differentiated thread whose loop can draw nothing must return first.

### 12.5 Kernel-scope cast semantics

`int(x)` and `wp.int32(x)` generate identical code, but `float()` fails to parse inside `wp.Float`-generic functions; use `type(x)(...)`. `//` floors since 1.18; `%` truncates (§1.5). At Python scope `wp.int32` arithmetic costs ~10 µs and `//` raises; unwrap with `int()` (check 26).

### 12.6 Compilation, module hashing and import cost

- A generic kernel's lazy overload instantiation rebuilds its whole module, and `wp.map` forks per call signature (§2.5, §3.5). Warp's own generic kernels do too (`array_cast`, `warp.sparse`): use `array.copyto` or `kernels/linalg.register_warp_overload`.
- `@wp.kernel` builds an `Adjoint` at import, so the lazy `__init__` is load-bearing; another module's top-level `import warp.fem` silently cancels a deferral.
- Backward passes dominate cold compile (§2.6). Only host readbacks block graph capture.
- `@wp.func` is a real function that nvcc inlines; verify "free" by comparing `_forward` SASS from the module cache.
- A kernel-scope `not` on a bool lowers as a select; share sentinel-returning predicates, not bools.
- `wp.constant(x)` is an identity; a tile `shape=` must be a plain integer; wrap typed constants in `int()` on the host.

### 12.7 `warp.sparse`

`nnz` is a capacity (§3.7). Also:

- `bsr_mm` returns a structural superset (explicit zeros) that inflates chained products; `bsr_mm`, `bsr_axpy` and `bsr_set_transpose` read the `nnz` field.
- `bsr_mv` takes one vector. Unwritten `(0, 0, 0.0)` triplets all accumulate on one entry (9-32x): point them out of range.
- `TiledDot` takes an O(n) per-block reduction under `batch_offsets` with `batch_count > 1`; `linalg._BatchedCg` avoids it.
- `cg` at `check_every=0` records a graph per call; it resolves an omitted `atol` to `tol`, so a tiny `||b||` returns zero iterations. Always pass `atol=0.0`.

### 12.8 Warp builtins: adoption verdicts

- **Adopted:** `mesh_query_point_sign_winding_number` (exact on holes; silently returns parity unless the mesh was built with `support_winding_number=True`); `bvh_query_sphere` (wins when the radius is large against a BVH, equals `length_sq <= r*r`, still beats a box walk on 1.18 though slower there; not for a radius equal to the grid cell); BVH leaf size 1 for ball queries and size-gated for k-NN; `mesh_get_bvh`; `volume_index_to_world`; `delaunay_edge_flip` as the flip start above 2**19 points; `sparse_marching_cubes` above a lattice-size gate on closed, consistently wound input.
- **Rejected:** `intersect_tri_tri` (not scale-invariant); `closest_point_edge_edge` (float32, worse); `sample_unit_hemisphere_surface` (variance); `norm_huber` (norm, not weight); `tile_arange`; `volume_voxel_count` (capacity); `dense_chol` family (hidden, 2x loss); `tri_tri_adjacency` for `halfedge_twins` (cannot carry validation, quadratic on hubs; test oracle only); `swept_volume_mesh` (no caller).

### 12.9 `wp.Volume` as a voxel-set container

- `allocate_by_voxels` and `Nanogrid` work on CPU; an empty point set raises. A one-`atomic_cas` table beats `allocate_by_voxels` 1.3-3.2x.
- `get_voxels()` order is leaf-major: lexsort before comparing. NanoVDB centres voxels on integers: pass `translation = origin + 0.5 * voxel_size` or cells are off by one.
- `get_voxel_count()` is a capacity; use `get_active_stats().voxel_count`. `rebuild()` does not pay. Import `warp.fem` inside the function.

### 12.10 Upgrade discipline

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

Warp-typed values at Python scope are slow: `wp.int32` arithmetic, slices with Warp-typed bounds, `wp.length` and vector arithmetic cost 4-40 µs against under 1 µs in plain Python (13-370x). Unwrap with `int()` / `float()` (check 26); `wp.cross` is the exception. A device reduction costs 0.1-0.3 ms flat; a readback wins below ~200 k `int32` elements (1 M `bool`).

### 13.2 Device-side and memory access

- `wp.tile_sum(wp.tile(x))[0]` is a block-wide reduction and a barrier (~0.13 µs at 32 lanes, ~0.35 µs at 64-128). A cooperative one-block rewrite wins only when a level's work exceeds ~0.6 µs of barriers (§14.9).
- A single-address atomic serializes the launch. Use lane partition, `wp.tile_sum` and one guarded commit per block, launched at `kernel_reduce.blocks_1d(n)`; the block count is the tuning variable. Do not convert a conditional atomic.
- Pack same-dtype quantities into one vector tile (`preserve_type=True`) for one barrier. Float sums that must reproduce use a fixed-order two-stage sum.
- A tile kernel whose reduced extent is under one tile loses ~49x; `tile_chunk` reports what remains, so clamp with `reduce.block_chunk`.

### 13.3 Tuning constants are per-device

CUDA wants long strided slices, CPU short (`ITEMS_PER_SLICE_CUDA = 128`, `_CPU = 32`). Sweep both devices and more than two values, split the constant if the optima differ, and re-probe after an upgrade (§9).

---

## 14. Kernel-shape verdicts

### 14.1 Block-per-item

- Block-per-item wins when the outer dimension alone starves the device, and loses when a slice dimension already fills it (§2.3). Models: `obscurance`, `shape_diameter`, and `farthest_point_sample` as one persistent block.
- Any-hit tracing suits binary occlusion; closest-hit is kept where the distance is used.
- A point-major layout wins only on very large clouds, and not for `shape_diameter`.
- A cuBQL mesh speeds up ray bundles when the caller builds once and traces many rays; `Trimesh.mesh_for_rays` decides rent-or-buy.
- `max_tangent_sphere` returns an answer inside a tolerance window; an "exact" variant is not the true one on a mesh, and the contract choice is the owner's.
- Kernels that already carry a slice dimension keep the arg-strided form; they use register blocking instead (§14.12).

### 14.2 Cooperative BVH walks

- The tiled BVH walk wins when concurrent queries are few (a few thousand for a ball of the cell width) and loses past that crossover, where the hash grid wins.
- The serial BVH is not better than the hash grid, so the win comes from the tiled traversal.
- Only `ball_pivoting`'s pivot search could take it; do not re-run the tree-wide sweep.
- A thread-per-query BVH launch is usually load-imbalanced, not under-pruned; histogram the per-thread candidate count first.
- Walk time varies up to 5x with loop spelling, so time a new or fused walk on the device against the one it replaces.

### 14.3 CUDA graph capture

- Capture wins when a launch sequence repeats identically, about two replayed rounds to break even. It loses on a once-through loop, and on a fixed-count loop of a few launches.
- A chain that differs only in a loop counter becomes repeatable when the counter lives on the device; record a group of 8 and replay it.
- A `capture_while` body may not allocate (`array_scan` does), and it loses to a batched host loop when its per-iteration overhead exceeds the sync it removes.
- A captured path needs a plain-loop CPU sibling, which is also the byte-identity reference.

### 14.4 Tile solves

- Tile solves win for K >= 16-32 when systems are few and the matrix is large; they lose for many small systems. ordito's dense solves (K = 5, 6) are on the losing side, so "rewrite them as tiles" is refuted.

### 14.5 Readback versus device reduction

- A device reduction wins on CUDA for mesh-sized arrays and loses on CPU. Decide on CUDA, but measure both.
- Keep the readback for bool masks and for small, non-scaling buffers; the real axis is often the loop count, not the array size.

### 14.6 Readbacks inside device loops

- A per-pass readback is cheaper than an extra loop pass, so trading passes for fewer syncs loses; `cg(check_every=0)` is not a lever in the harness. Price one iteration before removing a sync.

### 14.7 Per-device algorithm choice

- Branch on device only when the parallel form does asymptotically more work (pointer-doubled `polyline_downsample`, CUDA only above a size gate). `prefers_tiled_reduction` is a correctness branch, not a performance one.

### 14.8 Solvers

- A single-level polynomial preconditioner wins on long solves. A Chebyshev multigrid smoother loses because no single `(degree, interval)` is robust. A small synthetic system can mislead.

### 14.9 Refuted rewrites: do not re-propose

- A single-block BFS drain, a one-block hole-fill DP, and merged-level blocked interval DPs lose; the grid must stay full.
- Voxel aggregation for multigrid, micro-tuned Bridson blue noise, and two-stream overlap of sequential branches did not pay.

### 14.10 Producer-consumer fusion

- Always fuse consecutive same-`dim` launches where the consumer reads the producer only at its own index; every pair measured was faster.
- Price the fused region, not the call that contains it, and cross-check with kernel and allocation counts.
- Fold a map into its consumer, merge two launches of one kernel into a wider one, and walk-and-reduce neighbour lists in one thread per query.
- Fusion that changes a slot or box index needs a debug-mode run.
- Inline single-caller helpers the fusion leaves behind. Fold-on-read loses on dependent chases.

### 14.11 Blocked wavefront

- A tiled wavefront wins for a 2-D DP with local directional dependencies and small per-step work: one block per tile, one launch per tile-diagonal, a `wp.tile_sum` as barrier. It loses when levels are already wide (the fill DP).
- Test it at more than 32 lanes, and pin it byte for byte to the simple schedule.

### 14.12 Register-blocking outer items

- Giving each thread W outer items from registers wins for `(item, slice)` reductions bound by streaming the cloud once per item. Choose W per launch so the grid stays at about `1 << 17` threads; a fixed W loses on small inputs.

## 15. Benchmark and measurement traps

### 15.1 Sudden slowdown

- A sudden slowdown is a Warp rebuild until proven otherwise: look for `Module hash changed, recompiling` at `LOG_DEBUG` and an idle GPU (§2.5, §3.5). The second candidate is host-side per-element Python; fix it on the device, not with a smaller mesh.

### 15.2 Attribute against one number

- Time the candidate directly; subtraction projections are optimistic by 3-10x. Price at more than one size, since the trend's sign decides.
- Compute the candidate's share before reading a ratio near 1.0; isolate regions under a few percent.
- Pin the iteration count before timing an iterative solver.

### 15.3 Operating point

- Profile at the benchmark's own parameter and fixture; a win at one point can lose at another.

### 15.4 Harness hazards

- Cap every group's mesh size first, since one pathological row costs the whole JSON.
- Harness and hand-probe numbers are not comparable.
- A row can depend on mempool state left by its setup.
- A median at `rounds=3` can hide a one-off cost; compare it with the minimum.
- Run `benchmarks/test_meshes.py` after touching the mesh registry.

### 15.5 Plan items

- Grep the target function, constant comment and benchmark docstring for a recorded decline before measuring a plan item. Stage-profile before optimizing a stage.

### 15.6 A/B without `git stash`

- Use a detached worktree, never `git stash` or `sed` sweeps. The editable finder outranks `PYTHONPATH`, so drop it and assert the submodule `__file__`.
- Check the box is quiet first; when it is not, measure counts instead of clocks.
- A failing `nvidia-smi` does not mean CUDA is unusable.

### 15.7 Timing hygiene

- Interleave A and B in one loop and report the minimum beside the median. Baselines drift 10-30 %, so re-run both arms back to back.
- Batch launches and sync once.
- Use one pytest process per module for A/B.

### 15.8 Probe contamination

- Re-run a failing configuration in a fresh process before blaming the product.

### 15.9 Harness number

- Decide on the harness number, not the isolated one.

### 15.10 `wp.timing_begin`

- `wp.timing_begin` sees no graph-replayed kernels, so every CG solve reads host-bound. Re-run with capture disabled and carry the device total back.

### 15.11 Census

- Census host calls by monkeypatching Warp (`Function.__call__`, `wp.array.numpy`, `wp.zeros`, `wp.empty`), not by grepping. Read the slope across two iteration counts, and watch for capture hiding loops.

---

## 16. ordito component status

The full per-component detail is in `reference/component-status.md`; read the component's section there before optimizing or diagnosing it. Its sections: method notes; where time goes; `import ordito`; `reconstruction`; `remesh`/`repair`/`creation`/`bounds`; `array`/`graph`/`polyline`/`intersection`/`boundary`/`levelset`; `proximity`/`metrics`/`neighbors`; `sample`; `smoothing`/`laplacian`/`energies`; sparse assembly; `heat`; `validation`/`adjacency`/`halfedge`; `holes`; `points`/`voxels`; `registration`; preconditioners; the CG solver and `ordito.cholesky`.

### Open defects and traps

- **Not reproducible on CUDA (gate any A/B on the CPU device):** `ball_pivoting` face buffer on large meshes and with the default auto radius; `fill_min_weight` on a degenerate rim (float atomics in the rim normal); `face_flip_mask` on a non-orientable mesh; `fix_self_intersections`; `isotropic_remesh`; `sample_surface_blue_noise` on large meshes; heat solves on `saddle_graded`; `log_map` at the cut locus. A CUDA difference on these is not a regression.
- **A settled CG iterate is not a solved one.** On ill-conditioned systems (graded meshes) the settle rule stops on round-off. `solve_spd_settled` verifies the componentwise backward error and falls back to a Cholesky factorization held on a caller-owned object. Never trust a residual or a settle test alone for the heat family.
- **A CG that hits its cap or stalls does not warn when `check_every=0`.** `min_quad_with_fixed` therefore runs the verified path (iterate, check backward error, factor from zero on failure). `harmonic` at `k >= 3` uses the direct solver.
- **Never size anything off `matrix.nnz` or a Warp `*_count`**; both are capacities (§3.7). `fem.integrate` / `bsr_axpy` on stale `nnz` once exhausted memory and surfaced as CUDA error 700.
- **Fail-silently families:** a wrong binary search on an argsort payload returns a valid index of the wrong element (§3.1); `NaN` poisons `side="right"` search; an `int32` scan of large triplet counts wraps (use `int64`); an out-of-bounds write on CPU corrupts the heap later.
- **Heat method on obtuse-heavy meshes is wrong by the method's own error** (and so is potpourri3d's). `geodesic_path` strands on spurious minima and then finishes along mesh edges with a warning. `extend_scalar` warns when the field leaves the source range; `filter_mut_dif_laplacian` and `robust_laplacian` warn about ill-conditioned input. These warnings are the contract, not noise.
- **A degenerate-face threshold that is absolute (`TOLERANCE_MERGE`) removes marching-cubes slivers and opens the mesh**; `marching_cubes(edge_margin=)` is the fix and `screened_poisson` passes it.
- **Open:** off-cloud k-NN queries on the hash grid are slow (use `backend="bvh"`); `cholesky._estimate` undercounts memory about 2x; `quadric_decimate` collapses thin parts onto their mid-surface; FAS tau-correction for the Poisson band backend is unbuilt.

### Measured declines: do not re-propose

- Single-block or persistent-block rewrites of BFS, the hole-fill DP or other level-synchronous work: no, a block cannot beat a full grid and Warp has no grid-wide barrier. Blocked wavefront tiling wins only where levels were under-occupied (stitch DP).
- Blocked interval DP for hole fill: no, a plain span level already fills the device.
- Chebyshev multigrid smoother, multigrid or Chebyshev preconditioning for the heat diffusions, block CG over columns, batching CG convergence checks: no, interval robustness, componentwise accuracy or iteration growth.
- `cg(check_every=0)` and `wpl.cg` for ordito solves: no, a fresh conditional graph per call; `_BatchedCg` is the path.
- cuDSS or any one-shot direct backend; per-call factorization caches: no, plan cost dominates and the owner rejects memory held across calls. Factorizations live on caller-owned objects.
- `warp.geometry.tri_tri_adjacency` for `halfedge_twins`: no, quadratic in the largest vertex bucket and cannot carry the validation. Own bucketed pairing is used on CUDA.
- Hard triangle-quality veto, accumulated quadrics, boundary-segment placement or heavier boundary weights in `quadric_decimate`: no, each worsened mean deviation or stalled; the face-quality guard is measured and left off by owner decision.
- Reinstating `warp.fem` for Poisson: no, the narrow-band brick backend replaced it.

### Component notes

- **`import ordito` is lazy (PEP 562)**; the guarding test imports in a subprocess and checks that no kernel module loads.
- **Solver state:** one recorded state per operator in a weak-keyed cache; the state must not hold its key. Returned device arrays alias the state and are overwritten by the next solve. Pooled states refresh on every use.
- **CG launch shape:** two launches a round (Chronopoulos-Gear); small systems (at most 1 024 rows) run as one block per column; screened-Poisson storage stays `float32` on purpose, with `float64` block-partial folds.
- **`Trimesh` owns factorizations** (`heat_solver`, `fixed_vertex_solver`; `release_factorizations()`); nothing is factored automatically except on the second heat solve or on a failed settle check.
- **Poisson:** `dense` multigrid-PCG is the default (`solver_tolerance=1e-5`, non-convergence warns); `method="adaptive"` is the narrow-band backend. Error is not monotone in depth.
- **Key-sorted CSR everywhere:** structural builds write keys, not triplets; a repeated assembly over fixed faces passes `pattern=`.
- **Orders ordito reproduces itself:** `edges_unique` rows ascend by `(max, min)` key; collapse lock keys, new-vertex numbering and homology tie-breaks read row indices. Do not change `edges_unique` order casually.
- **`quadric_decimate`:** memoryless quadrics gathered in ascending face order (reproducible on CUDA), boundary planes added once per edge, relaxed endpoint-only independence, cost-bucketed lock key, default `feature_angle=180`.
- **Tuned constants live next to their code and are device-gated** (`_LINK_ON_DEVICE_FROM`, `EAR_ONE_BLOCK_MAX`, `RDP_ONE_BLOCK_MAX`, `_DOWNSAMPLE_DOUBLING_FROM`, `_KNN_DEFER_MIN_POINTS`, `_NEAREST_LEAF_SIZES`, `_GEODESIC_BALL_CHUNK`, `PACK_SEGMENTS_KERNEL_FROM`); re-probe after a Warp upgrade.
