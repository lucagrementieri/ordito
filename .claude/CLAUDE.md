# Role: NVIDIA Warp Expert

You are an expert in NVIDIA Warp (`wp`) working on **ordito**, a GPU geometry-processing library.
Follow every rule below when writing kernels, `@wp.func` helpers, Python-scope wrappers, tests and
benchmarks.

The file has two parts:

- **Part I — Rules (§1-§11).** How code in this repository must be written.
- **Part II — Measured facts (§12-§16).** Platform bugs, the cost model, kernel-shape verdicts,
  measurement traps and component status on this hardware and Warp version. Read the relevant
  section **before** proposing an optimization or diagnosing a slowdown: most plausible ideas here
  have been built and priced, and several lost.

**Part II is the only knowledge base.** There is no separate notes or memory store. Write a new
measured finding (a ratio, a refuted idea, a platform quirk, a reference library's trap) into the
Part II section that owns it, next to the numbers it revises; a rule that follows from it goes into
Part I with a cross-reference.

- **A measured decline is a result.** Write the number at the site and record it in Part II. Do
  not re-derive it; do not re-propose it without new evidence.
- **A Warp version claim is spelled `Warp 1.17`** (the word "Warp" immediately before the number).
  The staleness check (§4.5, check 9) reads only that anchored form, because measured ratios such
  as `within 1.25x` share the bare `1.N` shape. Installed: **warp-lang 1.18.0** on an **RTX 5090**.

## Contents

**Part I — Rules**

1. [Kernel syntax and semantics](#1-kernel-syntax-and-semantics)
2. [Kernel architecture and launches](#2-kernel-architecture-and-launches)
3. [Python-scope wrappers](#3-python-scope-wrappers)
4. [Evolving the public API](#4-evolving-the-public-api) (naming, signatures, the 30 checks)
5. [Function ordering within a module](#5-function-ordering-within-a-module)
6. [Documentation](#6-documentation-zensical--mkdocstrings)
7. [Testing](#7-testing) (parity gate, the nine reference libraries)
8. [Tooling and local validation](#8-tooling-and-local-validation)
9. [Performance work: measure before you change](#9-performance-work-measure-before-you-change)
10. [Warp API reference mirrors](#10-warp-api-reference-mirrors)
11. [Running long commands](#11-running-long-commands-never-poll-with-until)

**Part II — Measured facts**

12. [The Warp platform](#12-the-warp-platform-bugs-quirks-and-version-status)
13. [The cost model (RTX 5090)](#13-the-cost-model-rtx-5090)
14. [Kernel-shape verdicts](#14-kernel-shape-verdicts)
15. [Benchmark and measurement traps](#15-benchmark-and-measurement-traps)
16. [ordito component status](#16-ordito-component-status) (open defects, refuted plans and
    rules, by component)

---

# PART I — RULES

---

## 1. Kernel syntax and semantics

### 1.1 Decorators

- `@wp.kernel` for entry points launched via `wp.launch()`; `@wp.func` for helpers.
- **`@wp.kernel` functions MUST NOT return a value.** Annotate `-> None`; write into output arrays.
- **Every argument of `@wp.kernel` and `@wp.func` MUST be explicitly typed.**
- `wp.tid()` may be called **only inside `@wp.kernel`**; pass the index to a `@wp.func` as an
  argument.
- A `@wp.func` may return a tuple; declare `tuple[T1, T2, ...]`. The generic forms parse on
  Warp 1.17 (`tuple[Any, Any]`, `tuple[wp.Float, wp.Float]`, mixed), so a generic helper is no
  reason to omit the annotation.
- A `@wp.func` may read `values.shape[1]` and may carry default argument values. `wp.launch`
  accepts `None` for an array argument (null descriptor), legal as long as no thread indexes it.

### 1.2 Type standards

- Subscript-style array annotations: `wp.array[wp.vec3]`, `wp.array[wp.int32]`. Check 14 scans the
  whole package, because only in an *annotation* position is `wp.array(dtype=T)` stale.
- In **`ordito/kernels/` only**: `wp.array2d[T]` / `wp.array3d[T]` / `wp.array4d[T]` with
  matching multi-index `wp.tid()` unpacking. Python wrappers use `ordito.typing` aliases (§3.2).
- Vector/matrix types: `wp.vec2/3/4`, `wp.mat22/33/44`, `wp.quat`; `wp.indexedarray[...]` for
  indexed access.
- **No bare `bool` / `int` / `float` annotation** in a kernel or `@wp.func` signature (check 18).
  A kernel *factory* is ordinary Python and keeps `row_size: int` / `name: str`.
- **`wp.constant()` is not what makes a module-level value visible in kernel scope.** On
  Warp 1.17 it is `return x` after an `is_value(x)` check (§12.6): any module-level global that
  evaluates to a scalar/vector/matrix resolves from kernel scope, wrapped or not.
  `ordito/constants.py` carries no `wp.constant()` calls; do not add one. What is load-bearing is
  the typed constructor: `TOLERANCE_MERGE = wp.float32(1e-8)`. A plain Python float is treated as
  `wp.float32` in kernel arithmetic, and mixing it with a `float64` variable is a parse error; use
  `wp.float64(...)` where 64-bit precision is required.
- **Prefer dtype-generic `@wp.func`s** (`wp.Float` / `wp.Scalar` for scalars, `Any` for vectors;
  the `kernels/predicates.py` convention) while dispatch stays readable. Limits: §12.4.
- **The generic axis is also rank and dimension: `Any` is generic over both.** A helper pinned to
  either breeds copies as a precision-pinned one does. A matrix has no readable `.shape` in kernel
  scope (`r.shape[0]` is a parse-time error), so the rank-free singularity test is a reduction over
  the diagonal, `wp.min(wp.abs(wp.get_diag(r))) < tol` (`linalg.solve_normal_equations`); a
  barycentric solve needs no per-dimension copy because `wp.length_sq` / `wp.dot` ignore the
  ambient dimension. **When two bodies differ only in a size, look for the one statement that
  names it and ask whether Warp has a reduction for it.**

### 1.3 Casts and conversions

- Cast `wp.tid()` explicitly when used as an index: `f = wp.int32(wp.tid())`. **`wp.int32` /
  `wp.float32` are the only cast spelling** — never bare `int(...)` / `float(...)` in a kernel or
  `@wp.func` body (check 16). They are the same builtins (Warp writes `#define int(x)
  cast_int(x)` into every module header), except that **`float(...)` is a hard compile error inside
  a `wp.Float`-generic function** (`Input types must be the same, got ['float64', 'float32']`).
  Where the enclosing function is or could be generic, use `type(x)(...)`. Detail: §12.5.
- **A cast to the type a value already has is noise; delete it.** No cast is load-bearing on a
  bare `wp.tid()` (passes to a `wp.int32` parameter, a `wp.Scalar` generic, a kernel-scope slice),
  on `wp.int32` array elements, or on a `wp.int32` argument.
- **The tid cast is the one kept exception**, because it declares the index type the kernel is
  written against. **Check 22** enforces it: no bare single-index `wp.tid()`. It reads
  single-`Name` assignment targets only; a multi-index `i, j = wp.tid()` cannot carry a cast
  (`test_bare_tid_scan_ignores_multi_index_unpacks` pins that).
- A cast of a bare **literal** is never redundant: `wp.int32(0)` is a mutable dynamic variable,
  `0` a compile-time constant that freezes the enclosing loop (§1.6).
- **A same-type constructor is the same noise and check 16 does not see it**:
  `wp.vec3(*(a - b))` rebuilds a `wp.vec3`. When deleting no-op conversions, grep the
  `wp.vecN(*(...))` / `wp.matNM(*(...))` splat too. (Python-scope construction from host sequences
  is not this defect.)
- `wp.cast(expr, T)` is a bit reinterpretation **between same-size types only**; a width change
  fails at NVRTC time (`source and destination must have the same size`). Widening or narrowing a
  scalar is the constructor (`wp.float32(x)`, `wp.float64(i)`). Whole-array conversion is §3.6.

### 1.4 Kernel-scope restrictions

Not supported inside `@wp.kernel` / `@wp.func`:

- Lambdas, list comprehensions, sets, dicts, `list.append()`, `eval()`, recursion, exceptions.
- Python tuples for initialization; use typed constructors (`wp.vec3(1.0, 2.0, 3.0)`).
- Small fixed collections: vector types. Larger: `wp.zeros(shape=N, dtype=T)` (stack-allocated),
  but that is a **2x loss** against `wp.types.vector(length=N)` (§2.9).
- **A tuple cannot be subscripted by a runtime index; a vector can.** A `tuple[T, T, T]` return
  only unpacks into names. The spelling that keeps both is `wp.vec3i(*corner_triple(faces, f))`.
  The rule is *no slice where an index form exists*, not "no slices".
- **Variables defined only inside an `if` branch** are accessible afterwards but uninitialized if
  the branch was not taken; always initialize before branching.
- `wp.asin()` / `wp.acos()` auto-clamp to [-1, 1].

### 1.5 Arithmetic and conditional spellings

- **`%` follows C++11** (sign of result = sign of dividend), still on Warp 1.18.
- **Since Warp 1.18 integer `//` floors (like CPython) while `/` and `%` still truncate**
  (NVIDIA/warp GH-1918). `[-8, -7, -1, 0, 1, 7, 8]` gives `[-3, -3, -1, 0, 0, 2, 2]` under `//`
  and `[-2, -2, 0, 0, 0, 2, 2]` under `/`, on both devices. So on a **negative** dividend `//` and
  `/` are different operations, and `(a // b) * b + a % b == a` **fails**: `%` is no longer
  `//`'s remainder. Warp 1.17 truncated both. An audit at the upgrade found no kernel-scope `//`
  whose dividend can be negative where the result is used (`twin // 3` sites are guarded by
  `twin >= 0`; `remesh`'s collapse budget is only read when `>= 1`); the `-(-n // c)` ceiling
  idiom exists only at Python scope, where it always floored.
- **The floor costs a sign correction on CUDA**: `floordiv_signed` is `a / b` plus a
  remainder-sign fix-up, about 2 SASS instructions per site per entry even for a constant divisor
  (408 vs 400 on a two-site probe). Unsigned `//` is the plain division. **Priced tree-wide and
  declined**: every kernel module compiled against a copy of Warp whose `floordiv_signed` truncates
  differs by 440 `_forward` SASS instructions in 1 036 k (0.04 %), in 66 of 886 kernels; the
  largest relative growth (`multigrid.dense_solve`, +25 %) is one `//` outside the loop of a
  single-launch coarsest-level solve. Do not respell `//` as `/` or unsigned operands for speed.
- **Spell integer division `//`** (check 17). `/` on two `int32`s only truncates because the
  operands are integers, so a reader must recover the types. Dividends here are non-negative
  indices, where the conventions coincide; the hazard is a negative dividend, where `//` and `/`
  now differ. The scan types an operand only **by declaration**
  (annotated `wp.int*` / `wp.uint*` / `wp.Int` parameter, element of an array of such dtype,
  integer module `wp.constant`, integer literal, `.shape[...]`, `wp.tid()`, integer constructor, or
  an integer-preserving expression over those), because a scan that misfires on float division
  gets switched off.
- **Kernel-scope `and` / `or` short-circuit (Warp 1.17)**: `codegen.emit_BoolOp` guards each later
  operand behind an `if`, so `i + 1 < n and a[i + 1] == v` never reads past the end. A comment or
  nested-`if` shape justified by "`and` does not short-circuit" is stale.
- **Spell the remainder `%`**, not `i - (i // stride) * stride`.
- **A conditional value is `wp.where(cond, a, b)`, not a Python ternary** (check 20). Both lower
  identically. `wp.where` evaluates both arms eagerly where a ternary short-circuits; every
  kernel-scope ternary found had both arms already evaluated, so a future site where they are not
  needs a decision, not a mechanical conversion.

Checks 16, 17, 18, 20 and 22 are **one family**: every spelling is *legal* and generates identical
code, so the compiler and the suite cannot see the defect; only a scan holds the line.

### 1.6 Mutable loop accumulators

A variable initialized to a **bare numeric literal** (`count = 0.0`, `n = 0`) is a compile-time
constant and cannot be mutated inside a dynamic `for` / `while` loop. Warp raises `WarpCodegenError:
"Error mutating a constant ... inside a dynamic loop"` or, in some module contexts, **silently keeps
the initial value** (`mean / count` then divides by zero).

Declare mutable accumulators with the typed constructor: `count = wp.float32(0.0)`,
`n = wp.int32(0)`. (`float(0.0)` / `int(0)` work identically, §12.5, but §1.3 means the tree writes
`wp.float32` / `wp.int32`.)

**Never let a ruff auto-fix strip the constructor off a kernel-scope accumulator.** Where the legacy
`int(0)` spelling survives, two rules fire: `float(0.0)` trips only `UP018`, but `int(0)` also trips
`RUF046`, and `# noqa: UP018` alone does not suppress it; `ruff check --fix` rewrites it to `0` and
the kernel fails to compile on the next launch.

---

## 2. Kernel architecture and launches

### 2.1 Launch discipline

- `wp.launch(kernel=..., dim=..., inputs=[...], device=...)`. **Always forward the `device` from the
  input arrays**; this is a memory-safety rule (§3.9). Check 15 scans every launch site.
- Array slicing works inside kernels (`faces[f * 3 : (f + 1) * 3]`), but prefer an index form
  (§1.4).
- **Prefix output argument names with `out_` and put them at the end of the signature**, after all
  inputs (check 13). Two exemption classes (`_KERNEL_OUTPUT_ALLOWLIST`): **in-place** arguments
  (the same buffer is input and result) and **scratch / persistent-state** buffers (cursors,
  stacks, open-addressing tables, `ball_pivoting`'s front); name those for what they hold. A
  read-only input never wears `out_`, even when a producer kernel wrote it.
- Derive the face count as `f = faces.shape[0] // 3` from the flat face index array.

### 2.2 A lane-parallel kernel strides by `wp.block_dim()`, or it stays lane-free

A kernel whose lanes cooperate (`wp.tile_sum` / `tile_min` / `tile_max` over one value per lane, a
`wp.tile_bvh_query_aabb` walk) is correct on **both** devices exactly when its lanes partition a
sequence **the block already owns**, with the stride taken from `wp.block_dim()` — never from a
kernel argument or a module constant.

`wp.launch_tiled` runs exactly **one lane per block on the CPU device** through Warp 1.17 whatever
`block_dim=` is passed, and `wp.block_dim()` reads `1` there, so the single lane walks the whole
sequence and the one-element tile it reduces is correct. **A one-element tile is not the bug.** An
arg-strided sum is wrong on **both** devices: on CPU the lane walks only its stride's share, on
CUDA it is right only when the wrapper happens to pass `n_slices == block_dim`.

**Do not read a CPU failure of a tiled kernel as "tiles do not work on the CPU backend."** That
reading converted four kernels away from tile reductions for nothing. Platform detail and the probe
that separates the two cases: §12.2.

Where the lanes would partition the **outer** work the grid is over (a whole-array reduction with
no per-item dimension: `measures.centroid_tiled`, `metrics.chamfer_*_tiled`) there is no
block-owned sequence and no `wp.block_dim()`. Such a kernel either keeps a *constant* stride and
launches on CUDA only, with a lane-free `_sliced` sibling for CPU (the
`_device.prefers_tiled_reduction` pair), or stays lane-free on both. A single portable kernel is not
available. A strided loop is one token from portable; a block-index partition is a rewrite.

Models: `kernels/visibility.py::obscurance` (one block per point, lanes over its ray bundle,
3.2-11.8x against one thread per point) and `kernels/measures.py::centroid_tiled`.

**A lane-strided reduction's tuning variable is the fold width, not lane redundancy.** Launch it
`dim=kernel_reduce.blocks_1d(n)` so each block owns `ITEMS_PER_BLOCK_1D` elements, never
`dim=n / TILE_1D`: one atomic per block reaches the constant-slot accumulator, so the block count is
the contended quantity. Two kernels that already had one atomic per block gained **4.1-10.7x** from
the fold alone; removing the 64-fold redundant lane arithmetic was worth 1.02-1.13x.
Wide-accumulator caveat and commit spelling (assemble the summed vector/matrix first;
`wp.atomic_add` reads trailing indices as array dimensions): §13.2.

### 2.3 Block-per-item is an occupancy trade, not a shape choice

**Eligibility is an occupancy question; getting it backwards costs 2-8x.** An outer per-item
dimension is necessary, not sufficient: the form pays only when that dimension *alone* would starve
the device. `obscurance` qualified (`dim = n_points`, a few thousand threads, under 3 % of the
device, no second dimension). A kernel that already carries a **slice dimension** does not qualify:
that dimension fills the device, and collapsing it into `block_dim` lanes throws the occupancy away.

`points.hull_support_extremes` converted both ways: at 5 000 points (13 x 40 threads)
block-per-direction is **2.3x faster**; at 200 000 (13 x 1 563 threads) it is **0.12-0.60x**, 13
blocks on 170 SMs. `proximity.winding_number_tiled` shrinks from 2.16x at 4 096 queries to **1.01x**
at 65 536, a decline per §9.

**Before proposing this rewrite, compute the launch's current thread count and measure at the large
end first**; the small end shows a flattering 2x. Four kernels keep the arg-strided form
deliberately, each with its number. Win/loss table: §14.1.

### 2.4 Fusing a kernel: extract the shared part as a `@wp.func` in the same commit

Fusing two launches is welcome (removes a launch, a global-memory round trip and often a buffer),
but a fused kernel is written by *copying* the prologue of the kernel it absorbs. **A fusion is done
when the code it duplicated has a name.** Whenever a kernel is fused, split, specialised or given a
second variant (tiled/serial, float32/float64), the same commit must:

1. **Name the shared run.** Any consecutive statement run shared with the source kernel (corner
   load, window computation, emit protocol, guard sequence) becomes one `@wp.func` both call.
   `@wp.func` calls cost nothing at runtime (nvcc inlines, §12.6).
2. **Put it where the *quantity* lives**: a general geometric predicate in `kernels/predicates.py`,
   a per-face quantity in `kernels/triangles.py`, an index/sort/search helper in `kernels/array.py`,
   a scatter in `kernels/scatter.py` (§3.1). A helper left in an algorithm module makes unrelated
   modules import an algorithm to reach geometry.
3. **Say in a comment what the two kernels still differ by.** Where the difference is a
   *parameter*, prefer one kernel with a warp-uniform int selector (§2.7; the `ACCEL_HASHGRID` /
   `ACCEL_BVH` pattern in `kernels/neighbors.py`); where it is genuinely two algorithms, keep two
   kernels and have each name the other.

**Merge on identity of meaning, not of tokens.** Two bodies that compute the same quantity are one
function; two that agree because a one-line kernel has only one shape are two functions.
`remesh.compute_midpoints` and `triangles.face_centroids` normalise identically and stay apart.

**A comment that names the other copy ("mirrors X", "as in X") is the finding, not the fix.** A
cross-reference about *code* is a claim only a shared `@wp.func` can keep true: extract instead.
Re-count the run from the code, not the comment.

**A clean duplicate scan is not a clean file.** Scans key on α-renamed statement *text*, so two
bodies expressing one decision through different control flow (a `reject` flag against an early
`return`) look unrelated. **A duplicated *decision rule* is a live correctness hazard** and
outranks longer arithmetic runs: extract it as a `@wp.func` returning the classification (a
sentinel for "reject", the `_resolve_flip_quad_guarded` convention).

**Factor the family, not the pair.** A helper pinned to one rank or precision breeds the copies it
was meant to prevent (`corner_triple` covered flat 3-stride buffers while its rank-2 sibling stayed
nameless at eight sites; `laplacian.squared_edge_lengths`, hardcoded `wp.vec3` / `float32`, left
`float64` sites in `energies.py` open-coding it). Write the generic form (§1.2) and check the other
rank and precision before declaring the run named.

**One predicate, one spelling per module — a correctness rule.** `wp.length(d) < r` and
`wp.length_sq(d) < r * r` differ in float32 (10 of 200k rows disagree at the boundary). Pick one
per predicate and say which; speed is flat (0.997-1.003x, §13.2).

**A green suite does not prove a `@wp.func` extraction was behaviour-neutral**: reordering an
expression changes `float32` results without moving a `1e-5` comparison. Read the diff. Prefer
gating on the *decision* arrays a kernel writes, and on the combinatorial object (triangulation,
face buffer) where a reference's answer is one.

Watch the signature: a fused kernel inherits the union of two argument lists, ~1.0 µs of host time
each (§2.8, §13.1). **Price the fused region itself, never the call containing it** (§14.10): every
pair measured is faster fused (1.04-3.46x).

**A helper with one caller is a name, not an abstraction: inline it, then look again.** When a
fusion absorbs its own producer, the single-caller helper hides redundancy between the halves
(inlining exposed a doubled vertex load, a doubled grid index and a twice-computed `dbl_area`).

### 2.5 A generic kernel registers its overloads at import (`wp.overload`)

**A `@wp.kernel` generic over a dtype (`wp.Scalar`, `wp.Float`, `wp.Int`, `Any`) must have its
concrete overloads registered at import**, in a `_register_overloads()` called at the bottom of the
file. Warp instantiates an overload lazily on the first launch at each new dtype, and a module's
hash covers the set of **instantiated** overloads, so that launch recompiles **every kernel in the
module**. Nothing fails; it only costs (206 module loads in a full suite against 87 after
registration; one test file went from ~9 minutes to 1.4 s).

- **The chain is order-dependent**: a different test selection or call order re-pays it.
- **Register what the wrapper's dispatch can reach**, not every dtype the template admits (§4.2): a
  public `dtype=` keyword documented "float32 or float64", the key dtypes `sortable_dtype` maps,
  etc.; say so in a comment. Where two generic arguments are independent, register the **cross
  product**, not a diagonal: a diagonal registration made the first off-diagonal call recompile for
  over a minute (`energies.crouzeix_raviart_cotmatrix_triplets`; `laplacian.py`'s sibling is the
  model).
- **Registration is not compilation**: `wp.overload` builds the `Adjoint` only (milliseconds of
  import). Do **not** `wp.load_module` / `wp.force_load` at import.
- **`block_dim` forks the hash independently of dtype** but takes a bounded two or three values
  (a tiled launch's, Warp's 256 default, 1 on CPU); leave it alone. Collapsing them was measured for
  `reduce` and declined.
- `tests/test_api_conventions.py::test_generic_kernels_register_their_overloads` fails when a
  generic kernel has **no** overload registered. Nothing checks that a dtype *set* is complete; a
  missing dtype is diagnosed from the clock (§15.1).
- **`wp.map` has the same chain**, forked per *signature*: §3.5 (check 23), §12.6.
- **Keep the `wp.Kernel` that `wp.overload` returns and launch through it.** Every generic launch
  otherwise runs ~12 µs of `infer_argument_types` (§13.1; 2.17x on one launch, 1.09-1.58x end to end
  across 20 wrappers). A module's registration builds a dtype-keyed
  [`OverloadTable`][ordito.kernels.array.OverloadTable] and the wrapper writes
  `kernel_laplacian.COTMATRIX_TRIPLETS[cot_entries.dtype, dtype]`. All generic launch sites are
  converted; a new generic kernel adds a table, not a bare `wp.overload` call. A dtype the table
  lacks raises a `KeyError` naming the kernel. `kernels/reduce.py` has **no** generic kernels:
  its factories bake the dtype in.

### 2.6 Backward passes, and in-place `@wp.func` parameters (`wp.ref[T]`)

**Only `ordito/kernels/metrics.py` is differentiated** (the Chamfer losses, through a caller's
`wp.Tape`). Every other kernel module sets `wp.set_module_options({"enable_backward": False})`
right after its imports, because Warp compiles an adjoint for every kernel by default and the
adjoints dominate the build: a cold compile of all 55 kernel modules took 25.4 s with them and
10.6 s without (`energies` 11.5 s -> 1.5 s: `curved_hessian_triplets`' backward is 55 k SASS
instructions against an 8 k forward). The forward SASS is unchanged or smaller (38 of 886 kernels
shrink, mostly `reduce`'s tile reductions). A new kernel module copies the line;
`test_only_the_taped_kernel_module_compiles_backward_passes` fails when one does not, and fails if
`metrics` ever sets it (a taped kernel without an adjoint silently records no gradient). A
`@wp.func` compiles under its *calling* module's options, so `metrics`' taped kernels still get
adjoints of the helpers they import. Generated `wp.map` modules keep Warp's default.

`@wp.func` helpers may declare `wp.ref[T]` parameters to mutate caller-owned storage (locals, array
elements, struct fields), e.g. `update_argmin` in `kernels/array.py`.

- **Any kernel calling a `wp.ref` helper must be `@wp.kernel(enable_backward=False)`** — the
  per-kernel flag. A module-level `wp.set_module_options({"enable_backward": False})` is NOT
  consulted at kernel-parse time and the module still fails to compile. Inference-neutral.
- **Never use `wp.ref` in `ordito/kernels/metrics.py`**: the chamfer kernels are differentiated
  via `wp.Tape`.
- **`wp.ref[T]` requires a concrete `T`**; generics do not instantiate inside it (§12.3).

### 2.7 Function-valued parameters and kernel factories

- A `@wp.func` may take `fn: wp.Function` parameters; the target is bound at **compile time** per
  call site (user `@wp.func`s and simple builtins like `wp.min` are valid; tile intrinsics,
  variadic and LTO builtins are not).
- **`wp.launch` cannot pass a `wp.Function` as a kernel argument.** For runtime selection pass an
  **int/enum argument** and branch over `wp.Function` targets in a dispatch `@wp.func` (warp-uniform
  branch, one compiled module; `registration.robust_weight`).
- Builtins with no Python-scope handle (tile intrinsics) can parameterize kernel factories by
  pulling the `Function` from `warp._src.context.builtin_functions["tile_max"]` and
  closure-capturing it (inline at codegen, templated on the tile dtype; `kernels/reduce.py`). Give
  each factory instantiation a unique kernel `name`.
- **Name a factory kernel with `wp.kernel(f, name=...)`**, not by mutating `__name__` /
  `__qualname__`. `name=` (Warp 1.17, NVIDIA/warp#1561) sets the registration key and the native
  entry-point base; it must be a valid C++ identifier.
- **A factory also avoids a generic kernel's ~12 µs launch dispatch** (same for `Any`, `wp.Float`,
  `wp.Scalar`; more with several generic parameters, §13.1). Write `__annotations__` after defining
  the body:
  ```python
  def _factory(name, dtype):
      def _k(values: wp.array[wp.Scalar], out: wp.array[wp.Scalar]) -> None: ...

      _k.__annotations__["values"] = wp.array[dtype]
      _k.__annotations__["out"] = wp.array[dtype]
      return wp.kernel(_k, name=name)
  ```
  **A factory whose `dtype` parameter has a generic default is a factory nobody specialised**: give
  it no default.
- **Where the body should stay generic, the §2.5 table gets the same launch cost with no
  restructuring.** Use a factory when the *body* needs specialising (axis, storage class, captured
  builtin), the table when only the dtype does.
- **`if wp.static(flag):` prunes the untaken branch even when the branches bind locals of
  different types** (`wp.types.vector(length=K)()` vs `wp.zeros(shape=K)`, `values[i, k]` vs
  `values[k, i]`), so one body can generate axis- or storage-specialised kernels with no runtime
  branch.

### 2.8 `@wp.struct` argument bundles

**A `wp.launch` argument costs ~1.0 µs of host time, linearly, on both devices** (§13.1). If a
kernel is launched inside a Python loop and carries a dozen or more arguments, bundle the
invariant tables into a `@wp.struct` built **once** in the wrapper (subscript-style field
annotations): 43 -> 18 µs per launch for a 25-argument kernel, flat in `dim`, 2.6 µs per bundle
construction. Rebuilding per launch gives the win back.

- **Width alone does not qualify a kernel; the launch count around it does.** Count launches per
  call and divide. `remesh.objective_flip_candidates` (14 arguments) is declined: a flip round
  rebuilds the face adjacency around its one launch and converges in 2 rounds, bounding the saving
  at under 2 %.
- **Do not bundle a kernel whose launch is captured**: a replayed launch costs ~1.17 µs whatever
  its argument list (§13.1).
- Do **not** open a tree-wide pass: only 18 of 440 kernels take >= 12 arguments (§4.2).
- **A new construct can switch off a static check**: moving buffers into a `@wp.struct` removed
  them from check 13's view until it walked `ast.Attribute` (§4.5).

Examples: `holes._fill_dp` (16 arguments, once per span), `reconstruction._bpa_wave` (27, per wave).
Gains and the graph-capture comparison: §13.1, §14.3.

### 2.9 Per-thread row storage

**`wp.zeros(shape=K, dtype=T)` in a kernel is not fast per-thread storage.** Warp wraps the stack
buffer in an `array_t`, so accesses go through a pointer and nvcc never promotes it to registers:
a **2x loss** against a global-memory row at k=32.

Use **`wp.types.vector(length=K, dtype=T)`** (the `vec5d` pattern in `kernels/curvature.py`):
register-resident, 1.09x at k=1, 1.5x at k=7, 1.9x at k=16, 2.9x at k=30, **7.8x at k=64**,
collapsing to 1.66x at 96 where it spills (the row costs `2 * K` registers).

**Keeping it in registers constrains the code shape:**

- Never pass the vector to a `@wp.func` (takes its address and spills it); insert inline,
  duplicated per call site.
- **Any runtime index spills the whole vector**: read the k-th element with an unrolled
  `for slot in range(K): if slot == k - 1:`. `K` is closure-captured in a kernel factory so
  `range(K)` unrolls; one kernel per bucket, unique `name` each (`KNN_ROW_BUCKETS = (1, 4, 8, 16,
  32, 64)`).
- **A register insertion sort needs an explicit `placed` flag.** Carrying the displaced element
  with `if carry < row[slot]` breaks on a run of **equal** distances, dropping a neighbour (1 row
  in 20 000 at k=32, with the distance array byte-identical, so invisible to a distance test).

Cost of admission: `kernels/neighbors.py`'s cold-cache compile roughly tripled for 12 generated
kernels.

---
## 3. Python-scope wrappers

### 3.1 Module layout and imports

- Every kernel lives in a `kernels/` sub-module, imported with an alias:
  `from ordito.kernels import triangles as kernel_triangles`.
- **A top-level kernel module is named exactly for the public module it backs**:
  `ordito/kernels/<module>.py` ↔ `ordito/<module>.py`, one-to-one (check 7). Exceptions are the
  kernel-side libraries that back no single public module — `kernels/predicates.py` and
  `kernels/scatter.py`. Sub-packages (`kernels/algorithms/`) mirror a folder and are exempt. A
  kernel module with no public counterpart, or a public module whose kernels live under another
  name, is a defect: fix the name, do not document the exception.
- **Backing a public module and serving as a shared library are not exclusive.** Four modules do
  both, and being imported across the tree is what they are for:

  | Module | Holds |
  |---|---|
  | `kernels/array.py` | index/sort/cast/search `@wp.func`s (`sort3`, `cross2`, `to_vec3d`, `binary_search_index`) |
  | `kernels/predicates.py` | precision-generic geometric predicates |
  | `kernels/triangles.py` | per-face corner/quality/gradient `@wp.func`s |
  | `kernels/scatter.py` | scatter/accumulate kernels |

  The defect is placement: a general geometric predicate living in a module that owns an
  **algorithm**, so unrelated modules import the algorithm to reach the geometry. When a helper is
  reached from a second module, ask which of the four it belongs in before adding the import.
- **`ordito/__init__.py` is lazy (PEP 562 `__getattr__`) and must stay that way.** `@wp.kernel`
  builds an `Adjoint` at import time, so an eager `__init__` makes `import ordito as od` a
  whole-package pull (§16.2).
- **Three searches in `kernels/array.py` are not interchangeable**, and picking wrong is silent:
  `binary_search_index` is `searchsorted(side="right")` (returns `slot + 1` on an exact hit),
  `binary_search_index_left` is `side="left"` (the exact slot for a present key; for a lookup
  combine with `index < n and values[index] == v`), and `binary_search_sorted_contains` is
  membership only (§16.5).
- **A radix sort of packed keys whose radix is known passes `end_bit`**, and the producer writes
  the keys straight into the sort's double-width buffer. `wp.utils.radix_sort_pairs` otherwise
  orders every bit, at a cost mostly fixed per digit pass: 2.1x on the sort at 24 k edge keys
  (§13.1). The permutation is identical (stable sort), so results are byte-identical on both
  devices. `unique_1d(max_value=)`, `adjacency.sorted_face_edge_keys` and
  `unique_faces(max_index=)` carry the bound; the pair radix (`INDEX_RADIX_PAIR`) needs all 64 bits
  and gains nothing.

### 3.2 Typing (`ordito.typing`)

Import once per wrapper module as `import ordito.typing as odt`. Do not re-export typing symbols
from `ordito/__init__.py`.

**Why not `wp.array2d` in wrappers?** At runtime every buffer is `warp.array`; `wp.array2d[dtype]`
is a static helper that type checkers do not treat as a real array (missing `.shape`, bad
assignability from `wp.empty`), and `isinstance(x, wp.array2d)` is always `False`. Use
**`wp.array[dtype, Literal[ndim]]`** through the aliases:

| Alias | Meaning |
|-------|---------|
| `odt.Array2dInt32` | `(rows, cols)` `int32` |
| `odt.Array2dFloat32`, `odt.Array2dFloat64` | `(rows, cols)` `float32` / `float64` |
| `odt.Array1dInt32` | 1D `int32` |
| `odt.IntArray`, `odt.FloatArray`, `odt.ScalarArray` | 1D or 2D unions (e.g. `reduce.py`) |

Kernels in `ordito/kernels/` keep `wp.array2d[dtype]`. Optional 2D arguments:
`edges_sorted_wp: odt.Array2dInt32 | None = None`.

**Runtime checks, not `isinstance`:**

- `odt.ensure_ndim(arr_wp, 2, dtype=wp.int32)` validates rank and dtype on inputs.
- `odt.as_array2d(arr_wp, wp.int32)` checks, then narrows the return type for Pyright (overloaded
  for `wp.int32` / `wp.float32` / `wp.float64`); `odt.as_array3d` covers `wp.float32` / `wp.bool`.

### 3.3 Allocation and returns

- Wrappers accept `wp.array[T]` for 1D buffers; use `odt.Array2dInt32`, `odt.Array2dFloat32`, etc.
  for rank-2 results, returned as `odt.as_array2d(arr, wp.int32)`.
- **2D outputs**: `odt.empty_2d((rows, cols), wp.int32, device=...)`; `odt.empty_3d` at rank 3.
  `empty_1d` / `empty_2d` / `empty_3d` are one allocator at three ranks: each carries `dtype` into
  its return type through a `TypeVar`, so every Warp element type works at every rank (same for
  `as_array2d` / `as_array3d`). **Do not add an overload for "my dtype is not accepted"**; a real
  need would be a runtime restriction and belongs in the shared `_empty_ranked` body.
- **1D outputs**: `wp.empty(n, dtype=..., device=input.device)` when the kernel writes every
  element. `odt.empty_1d(n, dtype, device=...)` is for the modules whose signatures carry the rank
  (`reduce`, `metrics`, `neighbors`, where `k=1` or `axis=` collapses a rank). Elsewhere it buys
  nothing and, because `NDim` is **invariant**, breaks assignment to `wp.array[dtype]` (§8).
- **A rank-1 array's length is `.size`, not `.shape[0]` — at Python scope only.** `wp.array.size`
  and `.ndim` are typed `int` (`.shape[i]` and `BsrMatrix.nrow` / `.ncol` read `Unknown` through
  Warp's stubs) and are plain Python `int`s at runtime, so `int(x.size)` is noise. Limits:
    - **Kernel scope has no `.size`** (`WarpCodegenAttributeError`); `kernels/` keeps `a.shape[0]`.
    - **On a rank-2 array `.size` is rows × cols**; a wrong swap type-checks and runs. A table's
      row count stays `.shape[0]`.
    - **`torch.Tensor.size` is a method**; the rule is for Warp and NumPy arrays.
    - A parameter annotated rank-free (`odt.ArrayNd*`, `wp.array[Any]`) or reached at both ranks
      keeps `.shape[0]`; so do sites whose operand type is `Any` / `Unknown` / rank-2.
- **Always pass `device=` to every allocation** (`wp.zeros` / `empty` / `ones` / `full` / `array`)
  — check 10, scanning all of `ordito/` including `_*.py`. Without it the buffer lands on Warp's
  *current* device, which the suite cannot see because a test runs with its arrays' device
  current.
- Empty-mesh early return: `return odt.empty_2d((0, 2), wp.int32, device=faces_wp.device)`.
- **Size buffers for their final use at allocation time.** No allocate-then-grow. A consumer
  needing an `n + 1` sentinel-terminated form gets it from the *producer*, which allocates `n + 1`
  and returns a view (`counts_to_offsets`); a helper that only patches another function's output
  convention is a smell to fix at the producer.
- **Warp raises on a zero-length slice** (`RuntimeError: Invalid indexing in slice: 20:20:1`), so a
  trailing-mask `fill_` needs an `if stop > start` guard where NumPy silently no-ops.
- **A buffer whose initial value matters is allocated holding it — never `wp.empty` then
  `fill_` / `zero_`.** Use `wp.zeros`, `wp.full(n, value, dtype=..., device=...)`, and at rank 2
  `odt.as_array2d(wp.full((rows, cols), value, ...), dtype)` (deliberately no `odt.full_2d`, §4.2).
  This is legibility, not speed: the two-call form measures 0.96-1.00x.
    - `wp.empty` is the rule where **every** element is written before it is read.
    - Where branches initialize differently, allocate inside each branch.
    - A *partial* write into a buffer another writer already filled (running counter, padded
      triplet index, mask head written by a kernel) is not this pattern. The scan keys on a
      whole-buffer `fill_` / `zero_` on a name assigned from `wp.empty` / `odt.empty_*` within a
      few lines; run a second pass keyed on a *slice* target too.
    - There is no Python-scope scalar write to pair with `_device.read_scalar`: `arr[k] = v` raises
      `TypeError`, and `arr[k : k + 1].fill_(v)` is the primitive. Usually the *allocation* should
      carry the value instead.

### 3.4 Python-scope gather indexing (prefer over trivial gather kernels)

At Python scope `view = src[indices]` yields a `wp.indexedarray`. Materialize a dense `wp.array`
with `wp.copy(dst, view)` when callers need `.reshape()` or a guaranteed `wp.array` return (see
`selection.py`'s face gather, `array.py`'s `isin`).

- **2D index arrays**: Warp requires **1D** index arrays. Flatten (`elements.flatten()`), gather,
  `wp.copy`, `.reshape(original_shape)`. `.flatten()` on rank-3/4 arrays returns a contiguous
  rank-1 view (why `array.isin` accepts any rank), but **raises** `RuntimeError` on a
  non-contiguous view rather than copying.
- **The index array must be CONTIGUOUS.** Warp reads the index buffer as contiguous and **silently
  ignores a view's stride** (Warp 1.17): `payload[edges[:, 0]]` returns the flattened buffer's
  leading entries (`[0, 10, 1, 11, …]`), not column 0. A contiguous *prefix* slice (`arr[:n]`) is
  safe; a column (`arr[:, k]`) or step slice (`arr[::2]`) is not — `wp.copy` it to a dense buffer
  first. **Both halves are load-bearing**: cloning a prefix slice is a wasted copy (1.50x when
  dropped). `adjacency.face_adjacency` and `reconstruction._seed_candidates` clone a `[:, 0]`
  column and are the case the rule exists for. This is why `kernels/edges.py:edge_lengths` stays a
  kernel (§12.1).
- **When converting a gather, verify values, not just speed**: the corrupt version reads a
  contiguous prefix and is measurably *faster*.
- **Indexed assignment** (`arr[indices] = value`) is not supported at Python scope; keep a small
  kernel for scatter / mask marking (`mark_membership_mask` in `kernels/array.py`).

Do not add per-element gather kernels when `[]` plus `wp.copy` suffices. Probe tests live in
`tests/test_*_indexing_probe.py`; `test_array_indexing_probe.py` pins the stride hazard as "the
gather returns the flat prefix", so a Warp release that honours the stride fails it and this rule
is revisited.

### 3.5 Elementwise ops at Python scope (`wp.map`)

Do not write a `@wp.kernel` whose body is only `out[i] = f(in[i], ...)`. Keep the op as a named
`@wp.func` in `kernels/` (or use a builtin such as `wp.neg`, `wp.add`, `wp.div`, `wp.normalize`)
and call **`wp.map(op, *inputs, out=...)`**. The generated kernel is cached (per process and on
disk) with GPU time identical to a hand-written one; a cached call costs ~11 µs extra host time.

- **Named `@wp.func` only, never lambdas**: the cache is keyed by the *unqualified* function name
  plus input dtypes (same-named ops collide), and lambdas re-derive each call.
- **Always pass `out=`** so allocation stays in the wrapper. In-place is `out=<an input>`;
  multi-output funcs (`tuple[...]` return) take `out=[a, b]`.
- Scalars mix freely with arrays; the device is inferred from the array inputs. **But `wp.map`
  infers a bare Python `int` as `wp.int32`**, not the func's declared dtype (unlike `wp.launch`):
  a `bvh.id` for a `wp.uint64` parameter fails at codegen. Cast explicitly:
  `wp.map(fn, wp.uint64(bvh.id), ...)`.
- **Slice views** work as inputs and outputs (`wp.map(segment_length, polyline[:-1], polyline[1:],
  out=lengths)`; `out=dst[o : o + n]`; CSR row degrees from `offsets[:-1]`, `offsets[1:]`).
  **Python-scope gather composes**: `wp.map(pred, table[indices], out=mask)`.
- In **per-iteration wrapper loops** hoist the kernel with `wp.map(..., return_kernel=True)` and
  `wp.launch` it in the loop (see `smoothing.py`). A cached `wp.map` costs **1.78-1.86x the launch
  it wraps (~11 µs)**, flat in array size: the same host-side resolution a generic kernel pays
  (§13.1). Hoist where the loop body is cheap; a map inside a solve loop is a fraction of a percent
  (`linalg._multigrid_hierarchy`, the ARAP and CG loops and the remesh pass loops are left alone).
- **An op reached at more than one call signature must have those signatures declared at import**
  (check 23), for the reason of `wp.overload` in §2.5. `wp.map` names its module
  `map_<unqualified op name>` and each signature forks that module's hash, so an op at three
  signatures builds its module three times (2.1x cold-cache compile, 1.18x warm over the eight
  longest chains). **Declaration is not compilation**: `return_kernel=True` on a zero-length host
  array is sub-millisecond. Tables live in `_declare_map_kernels()` at the bottom of the module
  that owns the op; Warp *builtins* are declared in `kernels/array.py` (`declare_map_signatures`)
  because one generated module is shared across wrappers and every declaration must run before the
  first launch from any of them.
- **The fork axis is not only the dtype.** `warp._src.utils.map` keys on `(is_array,
  type(input).__name__, dtype, ndim, broadcast_mask)` per input, `broadcast_mask` being
  `tuple(d == 1 for d in shape)` — so a **length-1** array forks a module (12 ops fork on that
  axis alone; it is the normal path of every reduction-into-a-scalar wrapper), as does an
  **`indexedarray`** from a gather, as does the rank. Derive a table by instrumenting `wp.map`
  over a suite run and recording Warp's own key. Check 23 fails when a module needs a table and
  has none; the completeness gate is the load census (one `map_*` load per `(module, device,
  block_dim)` is the floor). Zero-length **CPU** arrays suffice to declare a CUDA overload.
- Never declare map kernels in `ordito/__init__.py` (§3.1), and never via `wp.load_module` /
  `wp.force_load` (eager compile).
- Still a real kernel: ops needing the thread index as *data* (`init_range`, `seed_orientation`),
  whole arrays as uniform arguments (binary-search tables), scatters, multi-element/row-indexed
  outputs.

### 3.6 Dtype conversion at Python scope

`wp.cast(expr, TargetType)` is kernel / `@wp.func` scope only; there is no `wp.cast` on whole
arrays. Convert with **`od.array.astype(src, dtype)`** (new array) or **`od.array.copyto(dst,
src)`** (into a buffer you hold; a slice view writes into its parent). Do not add a
`bool_to_int32`-style kernel.

**Never call `wp.utils.array_cast` from `ordito/`** (check 30). Its kernel is `Any`-generic and
lives in Warp's own `warp.utils` module, so every new dtype pair rebuilds that whole module (§2.5's
mechanism, in a module ordito cannot register overloads for; measured in §12.6), and it costs 1.8x
a launch besides (§13.1). `copyto` launches `kernel_array.cast_kernel(source, target)`: the hot
pairs are `ASTYPE` entries built with `kernels/array`, any other pair is built on first use in a
module of its own (`wp.kernel(..., module="unique")`), so building it rehashes nothing.
**A kernel created lazily must go in its own module** for the same reason: added to an
already-loaded module it changes the hash and recompiles every kernel there.

### 3.7 Sparse: assemble from keys, never through `bsr_from_triplets`; and `nnz` is a stale capacity

**No `ordito/` module calls `warp.sparse.bsr_from_triplets` / `bsr_set_from_triplets`
(check 27).** It sorts every triplet on a full-width key and allocates scratch several times the
matrix (9.4 GB per `cotmatrix` call at `lucy` for a matrix under 1 GB; a drained mempool makes
those bytes a cost, §13.1, §16.9). CSR is built two ways, both in `ordito/array.py`, both with no
host readback:

- **The sparsity follows from the mesh: write keys, not triplets.** A producer writes one
  `kernels/array.csr_key(row, col, n_rows, n_cols)` per contribution (sentinel `n_rows * n_cols`
  for an empty slot) into [`csr_key_buffers`][ordito.array.csr_key_buffers] with a payload that
  locates the value; [`csr_from_keys`][ordito.array.csr_from_keys] sorts only the bits the shape
  needs and returns `offsets`, `columns` and each entry's first sorted position; a per-row value
  kernel forms each entry from its contributors in producer order and may write the diagonal from
  the row it just formed. `laplacian._mesh_operator_pattern` is the model, shared by `cotmatrix`,
  `connection_laplacian`, `graph_laplacian` and `laplacian` (given faces): below
  `_UNDIRECTED_PATTERN_FROM_FACES` one sort of six directed keys per face plus diagonal slots,
  above it one sort of three undirected keys per face plus a stable 32-bit sort of the unique
  edges by larger endpoint for each row's transposed half — the identical matrix either way,
  pinned by a forced-threshold test. **A caller assembling an operator repeatedly over fixed faces
  builds the pattern once** with the public `laplacian.mesh_operator_pattern` and passes
  `pattern=` (`filter_implicit_fairing`, `filter_taubin(recompute=True)`,
  `vector_heat_operators`; §16.9).
- **The input is genuinely an unordered coordinate list:
  [`csr_from_triplets`][ordito.array.csr_from_triplets].** Unpruned it equals
  `bsr_from_triplets` (summed in triplet order, out-of-range triplets dropped; bit-identical on
  CUDA, equal to rounding on CPU). **Pruning differs on purpose**: `prune_numerical_zeros` drops
  every entry whose *sum* is zero, where Warp drops zero-valued triplets before summing and keeps
  an entry whose triplets cancel. So the two differ only in stored zeros (exact on CUDA, within
  float32 rounding on CPU). Zero-valued triplets are still skipped before the sort. Used by the
  energies, `laplacian(edges=...)`, `graph.edges_to_csr`, `index_sparse`, the Loop operator and
  the implicit smoothing system. The generic builder is only at parity with Warp (0.96-1.13x);
  the wins are the structural builds.
- **A CSR the producer can write directly is written directly** and wrapped by
  [`bsr_from_csr`][ordito.array.bsr_from_csr] (multigrid tentative prolongator; the empty matrix).

**Never size a buffer, slice or launch dim off `matrix.nnz`.** After a triplet build the `nnz`
field holds the *triplet count it was handed*, duplicates included: an upper bound (1.85x the true
count on `laplacian.cotmatrix`). Use **`matrix.nnz_sync()`** (one host readback) or read
`offsets[nrow]` (as `energies.k_harmonic` does).

- **`nnz` is a *cache*, and `nnz_sync()` repairs it in place**; nothing else syncs it (`bsr_mv`,
  `values.numpy()`, `offsets.numpy()` leave it stale). Whether a `.nnz` read is correct depends on
  whether unrelated earlier code synced that matrix: order-dependent, and a trap for tests too. A
  guard that measures the capacity and then hands the *same* matrix to the function under test
  has already repaired it and passes against the broken code. Build two operators, one to measure
  and one to hand over (`test_filter_laplacian_implicit_duplicate_built_operator`).
- **The failure is silent and not a Warp bug.** Sizing a `triplet_buffers` allocation by `nnz`
  leaves the tail `[nnz_sync(), nnz)` unwritten; since the buffers are `wp.empty`, the gap reaches
  the next `bsr_from_triplets` as **uninitialized triplets**. Out-of-range indices are dropped
  silently (huge or negative, no exception, no CUDA fault), so most garbage vanishes; entries whose
  garbage index lands in `[0, nrow)` accumulate a garbage value into a **real** entry (eleven
  orders of magnitude off on `_build_implicit_system`). **This is what the old "`bsr_mm` is
  nondeterministic on CUDA" claim really was; `bsr_mm` is sound — do not reintroduce it.**
- A matrix built from *duplicate-free* triplets (`laplacian.laplacian`) has
  `nnz == nnz_sync()`, so a probe on the default operator proves nothing; test on a
  `cotmatrix`-shaped input. Where a triplet writer legitimately leaves slots unwritten, `wp.zeros`
  is correct — but unwritten `(0, 0, 0.0)` triplets cost 31.8x (§12.7); point unwritten slots out
  of range instead.
- **A Warp object's `*_count` / `.nnz` field is a *capacity* until proven otherwise.**
  `wp.Volume.get_voxel_count()` and `wp.volume_voxel_count` report allocated capacity, so
  `ordito.voxels` uses `Volume.get_active_stats().voxel_count`. Before sizing a buffer, slice or
  `dim` off such a field, probe it against a construction with a known true count. The wrong
  reading is an *upper* bound, so nothing raises and the tail is garbage.

More `warp.sparse` behaviour (`bsr_mm`'s structural superset, `bsr_compress`, operands read from
the stale `nnz` field): §12.7.

### 3.8 NumPy at Python scope is sanctioned; leaking it through the API is not

`warp-lang` carries an unconditional `Requires-Dist: numpy`, `import warp` loads it eagerly, and
`wp.array(list, dtype=...)` ends in `np.asarray`. NumPy is a declared core dependency; deleting
`import numpy as np` from a wrapper only moves the call into Warp. **Do not open a "remove NumPy"
pass.** A census of 488 host-side sites under `ordito/` found only lattice/template index
arithmetic convertible (`parametric_surface`, §16.4); the rest is settled by one measurement: a
minimum device round trip (`wp.launch(dim=1)` plus one readback) is ~43 µs against 0.7-6 µs of
host arithmetic for a 4x4 matmul, 3x3 determinant or 3x3 SVD (a 26-60x loss).

Host-side metadata math (offset scans, launch dims, per-loop sizes, small candidate tables) and
host-*sequential* algorithms (patience sorting in `combine`, DP traceback in `holes`, `lexsort`
Delaunay in `reconstruction`, `argsort` + `searchsorted` chain linking in `intersection`, most
procedural templates in `creation`) stay in NumPy: porting them buys Python loops.

**The exception is a closed-form parallel *map* whose output scales with a resolution parameter**
(no dependence between elements). `creation.icosphere` and `creation.grid` are kernels; `grid` was
78 % NumPy prologue and is 33x faster (112x at the top of its axis) and bit-identical, because a
`float64` kernel followed by the same `float32` store rounds as the host build did. Decide by
whether elements depend on each other, not by module.

**The crossover, per operation class** (result staying *on the device*; ratio numpy/device, >1 =
device wins). Use it to price a conversion:

| operation | 1e3 | 1e4 | 1e5 | 1e6 | crossover |
|---|---|---|---|---|---|
| `cumsum` / scan | 0.38x | **2.6x** | 26x | 251x | ~5 k |
| `unique` | 0.18x | **2.8x** | 36x | 542x | ~5 k |
| gather | 0.08x | 0.67x | **6.3x** | 58x | ~30 k |
| sort | 0.03x | 0.27x | **3.0x** | 31x | ~50 k |
| sum -> scalar | 0.07x | 0.16x | 0.96x | **10.3x** | ~100 k |
| elementwise | 0.02x | 0.09x | 0.48x | **14.5x** | ~200 k |
| flatnonzero | 0.01x | 0.06x | 0.45x | **4.9x** | ~200 k |
| min/max -> scalars | 0.02x | 0.03x | 0.12x | **2.9x** | ~500 k |

Readback question separately (mask on the device, caller needs a host bool):
`mask.numpy().any()` against `reduce.any` crosses over at ~0.5 M elements (§12.4). `np.any`
short-circuits, so its cost depends on the data; an all-False mask is the worst case and the one a
convergence loop hits; timing a half-True mask makes numpy look 100x better than it is.

**Measured NumPy share of real public calls**: nine of eleven probed sit at 0.6-6 %, flat or
falling across a 256x face range (§9: a decline). The two above 10 % were
`creation.uv_sphere` / `capsule` (`_revolve_kept_template`, under the device floor for profiles up
to 64 points, a wash below ~256, declined) and `bounds.oriented_bounding_box`, where NumPy was not
the cost (§16.4).

Three things are still defects:

- **A public signature or return that names `np.ndarray`**, which forces the dependency on the
  caller. Return `wp.mat33d` / `wp.vec3` (`measures.moments` returns `wp.mat33d`; `wp.mat33` would
  discard the `float64` digits); annotate duck-typed `np.asanyarray` inputs
  `Sequence[Sequence[float]]`. The one exception is `ordito/io.py` (meshio hands back
  `np.ndarray`; NumPy-in is `mesh_from_numpy`'s purpose).
- **NumPy standing in for a Warp Python-scope equivalent that exists — but price it first.** A
  Warp builtin at Python scope is a builtin *dispatch* (§13.1): `wp.length` is 13-14x
  `np.linalg.norm`, `wp.min`/`wp.max` on a `vec3` 37x `np.minimum`, `wp.inverse` 3.2x
  `np.linalg.inv`, `vec3 - vec3` 17x, while `wp.cross` beats `np.cross` by 1.3x. The rule is really
  about not forcing a host round trip and not naming `np.ndarray` in a public signature.
  `wp.full`, `arr[k:].fill_()`, `wp.array([wp.mat44(...)])`, `wp.determinant`, `wp.inverse`,
  `wp.transpose` and `wp.svd3` work at Python scope with no host buffer; `arr.list()[0]` gives a
  row-indexable `wp.mat44`. `math.pi` / `float("nan")` / `float("inf")` beat `np.pi` / `np.nan` /
  `np.inf`. **Trap: `wp.svd3` is not `np.linalg.svd` of a non-square matrix**
  (`creation._align_vectors` takes the SVD of a `(3, 1)`; its free rotation about the axis is a
  gauge the trimesh comparison pins element-wise).
- **NumPy reducing a full `.numpy()` readback** (`.min()`, `.max()`, `.any()`, `.sum(axis=0)`) is
  a §9 defect: the whole array crossed the bus for one scalar. Use `ordito.reduce` (or
  `wp.utils.array_sum`, which reduces a `wp.vec3d` array componentwise) and check whether a kernel
  already exists (`holes._mean_rim_edge_length` read back the whole vertex buffer while
  `_loop_perimeters` already computed the answer). **Decide on the CUDA measurement and accept the
  CPU regression** (§9), but keep the host path where the buffer never scales with the mesh (a
  `k`-element argument check, a per-segment offset list). The seven-site A/B and its four
  rejections: §14.5.

Two redundancies that are not defects to "fix" the other way:

- **A `.tolist()` immediately splatted into a Warp vector/matrix constructor is noise — delete
  it.** `wp.vec3(*x.tolist())`, `wp.mat33(*x.ravel().tolist())`, `wp.mat44(*x.flatten().tolist())`
  and `wp.mat33d(...)` construct identically from the raw NumPy array (float32 and float64,
  contiguous and non-contiguous views); `math.dist` likewise takes raw arrays. **Keep** a
  `.tolist()` that *is* the return value of a `list[...]`-typed public signature, and one used for
  plain-Python bookkeeping such as a dict key. `tests/` keeps the `wp.vec3(*array_np.tolist())`
  spelling (§7.1); do not carry it into `ordito/`.
- **`arr.numpy().tolist()` is `arr.list()` only for a rank-1 array** (same cost; `.list()` calls
  `.numpy()`). **`.list()` unconditionally flattens**, so an `(n, 2)` output becomes one flat
  `2n` list and silently breaks row iteration. Restrict the swap to a genuinely `ndim == 1`
  buffer (offsets/index/mask arrays, `list[wp.array]` elements, an indexed row of a 2D array).

### 3.9 Device checks and the launch-device memory-safety rule

**Every public function that accepts two or more device-bearing arguments calls
`_device.require_same_device(**named)` as its first statement**, passing every array, mesh, BVH,
hash grid, `wp.Volume` or list/tuple of those it received — including an `X | None = None`
precomputed-cache argument (the helper skips `None`). It raises `RuntimeError` naming the two
mismatched arguments and their devices. Document it with a `Raises` entry like a direct `raise`
(§4.3); check 11 cannot see through the delegation.

The public boundary checks once through this one helper and everything behind it stays trusting:
internal call sites forward `device=` from an input array (§2.1), the test harness runs under
`STRICT`, and internal wrappers pass arrays they validated or produced. An external caller can
build a mismatch by accident (a mesh loaded to CPU, one built with a GPU default), and the
failures below are silent corruption or a bare segfault, not a `ValueError`. The check's cost is
unmeasurable next to a `wp.launch` (§13.1). `RuntimeError` (not `ValueError`) because no value is
bad — each array is valid on its own device — and it is the type PyTorch reserves for this class.

**`wp.launch` does not raise on a device mismatch** (since Warp 1.14, which removed the
unconditional same-device check, NVIDIA/warp GH-1461). The default
`wp.config.launch_array_access_mode` is `RELAXED` and validates nothing. On this box both failures
are silent:

- **CPU arrays, CUDA launch** (a launch that omitted `device=`, resolving to `cuda:0`): the GPU
  reads host arrays over HMM (`is_cpu_memory_access_from_gpu_supported` is `True` here) and
  computes the **right answer**; the launch is asynchronous, so when the host arrays are freed
  while the kernel still runs, the heap is corrupted and the process aborts in `malloc` later.
- **CUDA arrays, CPU launch**: immediate `SIGSEGV`, no Python exception (GH-1693).

Consequences:

- **`tests/conftest.py` sets `LaunchArrayAccessMode.STRICT`**, the only mode that rejects a
  genuine cross-device argument (`CHECKED` validates addressability, which HMM provides, and still
  corrupts). The full suite passes under `STRICT`; keep it so.
- **`STRICT` alone misses the motivating bug**: on a CUDA run an omitted `device=` resolves to
  `cuda:0`, which *is* the arrays' device, so nothing is rejected and the corruption waits for a
  CPU run. `test_launches_name_their_device` (check 15) scans every launch site statically; the two
  guards cover different halves.
- **§2.1's "always forward the `device`" is a memory-safety rule.**
- **`.numpy()` is not a sync on a CPU array** (zero-copy view; on CUDA it synchronizes), so "I read
  the result and it was correct" proves nothing about whether the kernel finished.

Root-cause detail and the bisection techniques: §12.1.

### 3.10 Readbacks

- **Budget the host-device syncs.** Every `.numpy()` / `int(<device value>)` readback in a wrapper
  carries a comment naming why it is unavoidable. When the caller can supply the bound the
  readback infers, expose a keyword (`face_adjacency(n_vertices=...)`,
  `hash_indices_rows(validate=False)`) **and pass it from every in-repo caller that knows it**; an
  escape hatch nothing uses is not an optimization.
- **A readback costs ~0.1 ms queued; an extra device pass costs 0.9-2.4 ms.** Trading one readback
  for an extra pass is usually a loss (§13.1, §14.6).
- **Several adjacent small values: `ordito._device.read_values(arr, start, count)`** (cached
  pageable scratch, one offset copy). A slice view plus `.numpy()` of three `int32`s measured
  26 µs against 10 for `read_scalar`.
- **A tail read uses `ordito._device.read_scalar(arr, index=-1)`**, not a hand-rolled spelling:
  the fast path is device-split and a pinned scratch is a **race** (§12.1). It takes any index and
  dtype, so `arr[k : k + 1].numpy()[0]` and `arr.numpy()[k]` are both it, spelled slower.
- **That is a single-value rule and it inverts at two.** A readback's cost is almost all fixed, so
  one `.numpy()` of a small buffer beats two `read_scalar` calls. A function returning several
  small device values writes them into *one* buffer and reads it once (`points.fit_plane` /
  `principal_axes`, `measures.moments`, `registration.icp_point_to_plane`'s scalar accumulator;
  1.27-3.00x). It reverses once the buffer is large enough for the copy to matter, so a *tail*
  read of a mesh-sized array stays `read_scalar`.

---
## 4. Evolving the public API

### 4.1 Naming

- **Name a function after what it returns, in NumPy vocabulary — never after the Warp call it
  wraps** (`sort_pairs` became `sort_and_argsort`: it is a sort *and* an argsort).
- **A mask is named `<element>_<property>_mask`, element first**, so typing `od.validation.face_`
  lists every per-face predicate. A convention followed regardless of fit is not worth having:
  `radius_outlier_mask` / `statistical_outlier_mask`, `half_space_mask`, `uv_seam_vertex_mask` and
  `convex_subset_mask` / `convex_superset_mask` deliberately keep their names. Check 2 scans the
  dtype; nothing enforces the order (a review question).
- **Two public names that differ by one character are a defect even when both are correct**
  (`boundary_loop` / `boundary_loops` became `longest_boundary_loop` / `boundary_loops`). Look for
  this whenever a plural is added next to a singular.
- **A wrapper whose whole device side is one kernel nobody else launches shares that kernel's
  name.** Reject the filler affixes `init_` / `do_` / `compute_` / `run_` / `make_` / `kernel_` /
  `_impl` / `_inner`: they stop `grep <name>` from finding both halves. The rule is about *filler*;
  the question is **"does the kernel's name carry information the wrapper's name does not"**.
  Informative differences stay: a plural (`trace_from_face` → `trace_from_faces`, the kernel is
  batched), `_pass` / `_step` (one iteration of the wrapper's loop), `finalize_` (second half of a
  two-stage reduction), a shared kernel library's vocabulary (§3.1's four:
  `vertices.vertex_defects` launches `scatter.scatter_sum_scalar`), and a kernel computing one
  *ingredient* (`remesh.subdivide` → `compute_midpoints`).
- **A tuning choice is a keyword, not a name.** `neighbors` exposes `query_ball` /
  `query_ball_count` / `query_ball_with_offsets` / `query_nearest` with
  `backend="hashgrid" | "bvh"` and an `accelerator=` that infers it, because the kernels already
  branch on `ACCEL_HASHGRID` / `ACCEL_BVH`.
    - **A default that must be distinguishable from "not passed" is `None`** (`backend=None` means
      `hashgrid` without an `accelerator`); only an *explicit* mismatch raises.
    - **Keep the discriminator in the benchmark group name** (`query_ball_bvh`); the group name is
      the parity key, so all markers move in the same commit.
    - **A merge needs a ordito-against-ordito test that the paths agree**
      (`test_the_two_backends_agree`).
    - Stays split: `query_bvh_ball` / `query_bvh_box` (BVH-only), and `metrics.chamfer_*` /
      `hausdorff_*` (different algorithms over different inputs, not one algorithm with a knob).
- **One operation family, one module.** A family split across two modules is a defect; the suite
  disagreeing with the split (a benchmark file holding rows from both) is the signal.
- **Place a function by what it computes and what machinery it shares, not by where the reference
  library keeps it** (`triangles.volume` / `moments` / `centroid` are whole-mesh reductions and do
  not belong in a per-triangle module). Names keep the trimesh / igl spelling where that is the
  field's vocabulary. A *component* of another module's solver stays with it; check with an
  import/call scan.

### 4.2 Signatures and returns

- **A packed buffer and its offsets are returned, and accepted, values first**: `(values,
  offsets)`, with any per-item third array last (`(ring_halfedges, offsets, is_boundary)`). Both
  halves are usually `wp.array[wp.int32]`, so **a transposed unpack type-checks, runs, and indexes
  garbage** — a transposed hand-built pair (`descend_field(vertex_faces=…)`) segfaulted the CPU
  backend several launches later. Check the pair order before blaming Warp.
- **Offsets are always the `n + 1` total-terminated form** (scipy's `indptr`): item `i` is
  `values[offsets[i] : offsets[i + 1]]`, `offsets[-1] == len(values)`, empty is `[0]`, no separate
  sizes array. `array.split` enforces it and raises on a length-`n` array. Build one by allocating
  `n + 1`, writing counts into `offsets[1:]` and scanning in place. **The hazard is an item count
  read as `offsets.shape[0]`**: it runs one item past the end, which on CPU is host-heap
  corruption (§12.1), so gate convention changes with `tests.devices`, never CUDA alone.
- **Every function returning a list of variable-length items has a packed sibling
  `<name>_with_offsets`** (device arrays: values, offsets, then any per-item array); the list form
  is that plus `array.split`, so the two cannot drift. A function that *takes* the packed form is
  `<name>_from_offsets`. A function that only slices an already-packed result into a list *is*
  `array.split`; call it rather than wrapping it (`geodesic_walk`'s tracers return the packed
  form only). `_batched` is not a suffix in this package. `array.split` also accepts host-sequence
  offsets.
- **No public signature or return type may name `np.ndarray`**, outside `ordito/io.py` (§3.8).
- **A guard must encode a real limitation.** Where the implementation is rank- or dtype-agnostic,
  drop the `ensure_ndim` cap and widen the annotation.
- **A `Literal`-typed menu argument is validated at the public boundary and raises `ValueError`
  naming the argument, the offending value and the options.** The annotation is a hint: nothing
  rejects an off-menu value arriving through a variable or `**kwargs`.
    - `Literal[True]` / `Literal[False]` `return_*` pairs and rank literals are not menus.
    - Where the options live in a table, derive the message from it (`list(...)` for an ordered
      table, `sorted(...)` for a `frozenset`); for an `if` / `elif` chain spell the names out.
    - A delegating wrapper does not repeat the check; it documents the `ValueError` (§4.3), and
      its *test* must call through the wrapper.
    - **No static check guards this, deliberately** (a strict one flags every delegating wrapper,
      nearly half the sites). The gate is a **probe**: call each menu entry point with an
      off-menu value and read the exception type.
- **When a function mirrors a NumPy one, mirror its positional signature and make `device`
  keyword-only** (`array.arange(start, stop=None, step=1, dtype=wp.int32, *, device)`).
    - A required keyword-only `device` breaks every positional call site; **re-derive the residual
      set from the new name** (§4.4) — imports via `from ordito.array import arange` fail at
      runtime, not at lint time.
    - Keep the specialised kernel for the common case: `arange` dispatches to the zero-argument
      kernel when `start == 0 and step == 1`, `arange_affine` otherwise (two fewer launch args).
- **No speculative generality.** Add an axis, parameter or mode only when an in-repo call site
  needs it. A parameter the body never reads is removed, not kept as a documented no-op (a caller
  then gets a `TypeError`, not a silent no-op).
- **No near-duplicate wrappers.** Two public functions that are one algorithm with different
  returns share one private helper (`concatenate` / `pack_1d_arrays` behind `_pack_segments`).
- **Inverse and dual pairs cross-reference each other and have a round-trip test**
  (`flatnonzero` / `indices_to_mask`). Bidirectional `See Also` is required for inverse pairs and
  simple/advanced variants, not for hub→spoke references.
- **Coverage is per module.** Every public `ordito/<module>.py` has `tests/test_<module>.py` and
  `benchmarks/test_<module>.py` (check 4); a function's tests live in the file mirroring *its*
  module (§5).

### 4.3 Docstring, signature and body must agree

- A documented `Raises` must be reachable, and a function with a direct `raise` needs a `Raises`
  block (check 11). A function delegating validation to a shared guard documents the guard's
  `Raises` (every public function with two or more device-bearing arguments documents
  `_device.require_same_device`'s `RuntimeError`, §3.9); check 11 does not scan those.
- A documented validation must actually be performed, or the claim goes.
- Annotations must cover every rank and dtype the docstring claims and the body supports.
- **When a comment and the body disagree, decide which is load-bearing before "fixing" it — the
  usual answer is the comment.** (`tangent_space.any_perpendicular` compares only `|n[0]|` with
  `|n[1]|`, which is correct for its purpose; "correcting" it would move the tangent frame at
  every z-dominant normal.) Say in the commit which one you changed.
- **When a private helper's docstring states a rule the public surface must obey, the defect is in
  the paths that bypass the helper** — early returns, the `@overload` stubs and the `Returns` prose
  are further statements of the rule that no test reads (`neighbors._shape_nearest` collapses to
  rank-1 at `k == 1`; two early returns of `query_nearest` skipped it). Route bypassing paths
  back through the helper, and **after changing a rank/shape/dtype rule grep the early returns and
  `@overload` stubs**: basedpyright cannot see a self-consistent wrong overload.
- **A test that hardcodes one value of the parameter the contract turns on tests the one case that
  cannot fail** (`test_query_nearest_empty` pinned `k = 2`). Parametrize over the boundary value.

### 4.4 Moving or renaming

**Moving or renaming a public function moves everything derived from it — in the same commit.**
Five artifacts, every time:

1. **Its kernels, if exclusively its.** A kernel referenced only by the moved function moves to
   the destination's `kernels/` module; one shared with a function that stays does not (the new
   module imports it; kernel-to-kernel imports are normal). Decide with an AST scan of which
   wrappers reference each `kernel_<mod>.<name>`, not by reading.
2. **Its tests**, into `tests/test_<destination>.py`, keeping §5 source order.
3. **Its benchmark rows**, into `benchmarks/test_<destination>.py`.
4. **Its `benchmark(group=...)` name**, when named after the function or old module. The group name
   is the parity key: update every `parity` / `noparity` marker citing it in the same commit, and
   `uv run python -m tests.parity` must show the same pair count before and after.
5. **Its docs entry** in `docs/SUMMARY.md` (check 28) and every `[`name`][ordito.old.path]`
   cross-reference — `zensical build --strict` finds the missed ones.

**Renaming a *keyword argument* has its own artifact list; a call-site scan sees none of it:**

1. A `TypedDict` field feeding a `**splat` (`registration._TargetIndex.bvh` →
   `query_nearest(..., **target_index)`): grep the old keyword name on its own.
2. A keyword pass that ran after the function rename and no longer matches the merged names:
   sweep "merged name still carrying the old keyword" as a separate final pass.
3. A fenced ```python docstring example: check 12's runtime `exec` finds it; `ast.parse` and
   grep do not.

After any mechanical rename **re-derive the residual set from the new names**; a rename is not
done until the whole suite has run.

### 4.5 The mechanical gate: `tests/api_conventions.py`

**Thirty checks**; they fail the default `pytest` run. Each carries a written allowlist —
read the reason before adding an entry, and prefer fixing the code. The gate does not replace
review: it cannot tell whether a *new* name is a good one.

**Public surface of `ordito/` (excluding `kernels/`):**

1. A module/function summary line naming a reference library (§6). Allowlist: `mesh.py`'s
   "mirrors `trimesh.Trimesh`" alone.
2. A `*_mask` producer not returning `wp.array[wp.bool]`.
3. A module summary ending in `(Warp)` / `on NVIDIA Warp`.
4. A module without both a `tests/` and a `benchmarks/` file named for it.
5. A private name reached across a module boundary. It reads only public wrapper modules and
   `_`-prefixed *names*, so a plain-named helper in `ordito/_thing.py` (like `_device.py`) is
   invisible — **but a new `_*.py` holding one helper is a module created to dodge the check**: a
   shared operation wants a home. When a private cross-module helper must stop being public, ask
   whether it is really one operation (`adjacency.require_paired_adjacency` is the shared *rule*;
   the *derivation* is an inline `face_adjacency(return_edges=True)` per caller). A name moved into
   a `_*.py` breaks every `[`x`][ordito.mod.x]` reference under `--strict` (private modules
   generate no page), and a newly public validator needs its own `Raises` block and accepting-case
   tests.
6. One public name exported by two modules.
7. A top-level `kernels/<name>.py` without `ordito/<name>.py`, or the reverse.
8. A private helper defined above its first caller (§5).

**Cross-tree:**

9. (`kernels/`, `tests/`, `benchmarks/` too) A comment or docstring blaming a Warp version older
   than the installed `warp-lang`; reads only the anchored `Warp 1.17` spelling.
   `_WARP_VERSION_ALLOWLIST` holds deliberate history and the next upgrade's re-verification
   notes. **A stale `pytest.skip` is worse than a stale comment**: it deletes a branch, and on a
   CUDA box that is the branch nobody runs.
10. An allocation with no `device=` (§3.3); scans all of `ordito/` including `_*.py`.
11. A public function that raises with no `Raises` block (§4.3).
12. A fenced ```python docstring example that does not run: `tests/test_api_conventions.py`
    `exec`s the extracted blocks against a mesh fixture (both motivating defects were runtime
    ones). Blocks holding a bare `...` are outlines and skip. A name the fixture lacks raises
    `NameError` — extend `example_namespace`, do not weaken the test.
13. Kernel `out_` prefix and position (§2.1); in-place and scratch/persistent-state buffers are in
    `_KERNEL_OUTPUT_ALLOWLIST`.
14. Subscript-style array annotations, whole package (§1.2).
15. A `wp.launch` with no `device=` (§3.9).
16. Cast spelling: bare `int(...)` / `float(...)` in a kernel or `@wp.func` (§1.3).
17. `/` on declared-integer operands (§1.5).
18. Bare `bool` / `int` / `float` annotation in a kernel signature (§1.2).
19. (`tests/`) A test comparing against a reference library with no class label (§7.4). It keys
    on `ast.Assert` (not the function body, which would flag every mesh fixture unpack), accepts
    **all four** label phrases, leaves `_np` out of its suffix list, checks only that a label is
    *present*, and also closes "comparison with no docstring" (ruff ignores `D103`).
20. Kernel-scope ternary instead of `wp.where` (§1.5).
21. A MeshLib **or promesh** name anywhere under `ordito/` — a licensing guard (§7.6).
22. A bare single-index `wp.tid()` (§1.3); multi-index unpacks are not read.
23. A kernel module whose `@wp.func` is `wp.map`'d from several sites with no declaration table
    (§3.5). It asserts a table *exists*, not that it is complete; completeness is the load
    census (§15.1).
24. A `.claude/CLAUDE.md` cross-reference naming a nonexistent section, **or a bare chapter number
    where that chapter is subdivided** (a staleness rule: one number can stand for several
    sections). Chapters 5, 6, 8, 9, 10 and 11 carry no numbered `###` heading, so a bare number is
    accepted for them; the check reads this from the file's headings, keeping
    `_CLAUDE_CHAPTER_ALLOWLIST` empty. Scans `ordito/`, `tests/`, `benchmarks/`; matches
    `AGENTS.md` (a symlink); abstains if the file is absent. Cannot see a reference split across
    sentences, nor a bare `section N` belonging to a paper (`kernels/remesh.py` cites "Liepa 2003,
    section 3").
25. A `!!!` admonition inside a numpydoc item-list section (`Parameters`, `Returns`, `Yields`,
    `Receives`, `Raises`, `Warns`, `Attributes`, `See Also`; §6). No allowlist machinery: every
    admonition has a correct home in `Notes` or the leading description.
26. A Warp-typed module constant (`wp.int32(0)`) used at Python scope as an arithmetic operand or
    slice bound (§13.1). Scope is narrow on purpose: wrapper layer and *module scope* in
    `kernels/` only, constants only (not `wp.length` calls), looks *through* `wp.constant`
    (`wp.constant(7)` is a plain `int`; only `wp.constant(wp.int32(7))` is Warp-typed). Empty
    allowlist; a constant forwarded to `wp.launch(inputs=[...])` or wrapped in `int(CONST)` is
    fine. Vector arithmetic and explicit builtins are covered by the runtime census (§15.11).
27. A call to `warp.sparse.bsr_from_triplets` / `bsr_set_from_triplets` under `ordito/` (§3.7).
    Empty allowlist; tests may call it to build an independent input.
28. A public module missing from `docs/SUMMARY.md`, listed twice, or listed after removal (§6).
    Abstains when `docs/` is absent.
29. An array entry (`Parameters` / `Returns` / `Yields` / `Attributes`, or a property's summary)
    whose description opens with `Length-`, `Shape ``(`, `Flat` or `Rank-N ``(` instead of its
    shape (§6). Empty allowlist. Reads the opening only: a shape buried later, a redundant dtype
    or an abbreviated size name is review's.
30. A call to `warp.utils.array_cast` under `ordito/` (§3.6). Empty allowlist; tests may call it.

Checks 16, 17, 18, 20, 22 and 26 are one family (§1.5): legal spellings with identical codegen,
held only by a scan.

**Two rules about the gate itself:**

- **A new Warp construct can silently switch off a static check that predates it.** Bundling
  buffers into a `@wp.struct` removed them from check 13's view (it resolved store targets to a
  bare `ast.Name`; fixed by also walking `ast.Attribute`). **After introducing a construct the
  package has not used before** (first `@wp.struct`, first `wp.ref`, first tile intrinsic), grep
  `tests/api_conventions.py` for checks that pattern-match on AST node types. **A plain `@wp.func`
  extraction does it too**: moving a kernel's writes into a helper removes that buffer from
  check 13's view, caught only by the staleness half reporting an unmatched allowlist entry.
  §2.4's extraction wins over the check: the entry comes out and the convention goes unenforced
  for that parameter.
- **Write checks with a staleness half** (an allowlist entry that matches nothing fails).

---

## 5. Function ordering within a module

`mkdocstrings` uses `members_order: source`, so **source order is the rendered docs order**.

### Python wrapper modules (`ordito/*.py`)

- Layout: module docstring → imports → module constants / type aliases → functions.
- **Group public functions thematically, then order groups by importance and expected frequency
  of use**: primary entry points first, niche or low-level variants last. Related functions are
  consecutive (`expand_vertex_mask` / `shrink_vertex_mask`); the simple form precedes its
  advanced variants (`query_ball` before `query_ball_with_offsets` / `query_ball_count`).
- **Stepdown rule for private helpers** (check 8): a private function called by exactly one public
  function goes **immediately after** it; one shared by several functions goes after its **last**
  caller; cross-cutting utilities go in a trailing "private helpers" section. A private helper
  never appears above its first caller. When reordering, consider *every* private helper (moving
  one below a callee it uses strands the callee), and verify a "pure move" by comparing the
  multiset of definitions, since `@overload` stubs repeat names.

### Kernel modules (`ordito/kernels/*.py`)

The stepdown rule **inverts**: Warp resolves `@wp.func` references at decoration time, so a
`@wp.func` **must textually precede** every kernel or `@wp.func` calling it. Order kernels to
mirror the wrapper module's public-function order, place each kernel's helpers immediately
**before** it, and put a helper shared by several kernels before its **first** user.

### Tests (`tests/test_<module>.py`)

Mirror the wrapper module's public-function order.

---

## 6. Documentation (Zensical + mkdocstrings)

Docs are built with **Zensical** + **mkdocstrings** (`python` handler, `docstring_style: numpy`).
Zensical reads **`mkdocs.yml`** unchanged (`theme: name: material`).

```bash
uv run zensical build --strict          # validate: exits 1 on a broken cross-reference
uv run zensical serve                   # preview locally
```

**API pages come from Zensical's native `api-autonav` (0.0.66+)** configured in `mkdocs.yml`: one
page per public module at `api/ordito/<module>/`, with `ordito.kernels`, the `_*.py` modules and
the package root excluded. No pre-build step; never use `mkdocs-gen-files`. **The nav is the
committed `docs/SUMMARY.md`**, which shelves pages by theme; the plugin adds no page the nav does
not name, so a new module needs a line there (check 28). Runtime facts, each of which fails
quietly:

- **Zensical ignores an unsupported plugin entry silently.** Never register `gen-files`: the build
  exits 0 and generates nothing; only `--strict` reports the unresolved cross-references.
- **Backlinks are on (`backlinks: tree`)**: each cross-linked object gets a "Referenced by" block,
  which is the consumer list a hub's `See Also` should not carry by hand. `ordito.typing` opts
  out via `api-autonav`'s `module_options` (its aliases are linked from every return annotation).
- **`--strict` is the cross-reference backstop** §4.4 leans on; the four external inventories
  still resolve under it.
- **`zensical serve` watches `docs/` but not `ordito/*.py`**, and mkdocstrings caches the module
  in-process: restart `serve` to see a docstring change.
- **There is no `exclude_docs:` equivalent.** The `assets/benchmarks/*.md` sidecar tables are kept
  out of search by `search: exclude: true` front matter (emitted by
  `benchmarks/plot_comparison.py`); they are still built as unlinked pages. `draft: true` is a
  no-op.
- **A build that finishes in ~0.1 s and leaves `site/` empty is inotify exhaustion, and it exits
  0.** Zensical adds an inotify watch per file read and silently drops files when
  `inotify_add_watch` fails with `ENOSPC` (`fs.inotify.max_user_watches`, 65 536 here; IDE
  watchers use it up); `--strict` still reports `No issues found`, so the gate checks nothing.
  Confirm with `strace -f -e inotify_add_watch zensical build`. Fix: `sudo sysctl
  fs.inotify.max_user_watches=524288` or close watchers; without root, an `LD_PRELOAD` shim that
  fakes a descriptor on `ENOSPC` restores a full build (compile with `gcc -shared -fPIC -o
  fakewatch.so fakewatch.c -ldl`, run `LD_PRELOAD=./fakewatch.so zensical build --strict`):

  ```c
  #define _GNU_SOURCE
  #include <dlfcn.h>
  #include <errno.h>
  #include <stdint.h>
  static int next_fake = 1 << 28;
  int inotify_add_watch(int fd, const char *path, uint32_t mask) {
      static int (*real)(int, const char *, uint32_t) = 0;
      if (!real) real = dlsym(RTLD_NEXT, "inotify_add_watch");
      int wd = real(fd, path, mask);
      if (wd < 0 && errno == ENOSPC) { errno = 0; return next_fake++; }
      return wd;
  }
  ```

  **Pass `--clean` after a docstring change**: `build` reuses `.cache/` and can re-emit the
  previous docstring, which `--strict` also passes.
- **`literate-nav`, `section-index` and `api-autonav` are implemented natively**; none is
  installed and their `plugins:` entries are read as configuration.

### Docstring content

**The one-line summary says what the function returns, never which C++ call it wraps**:
mkdocstrings renders it as the function's index entry. Attribution stays, one line down in `Notes`
or `See Also`. Enforced by checks 1 and 3.

**An array's entry opens with its shape as a code span, then says what it means**:
``` ``(3 * n_faces,)`` flat triangle index buffer. ```, ``` ``(m, 2)`` unique undirected vertex
pairs, ``m <= 3 * n_faces``. ```, ``` ``(n_faces,)`` component labels on ``faces.device``. ```
Every `Parameters` / `Returns` / `Yields` / `Attributes` entry of an array, sparse matrix or array
pair, and a `Trimesh` property's summary. mkdocstrings' table puts the description beside the
rendered annotation, so the shape is the one fact the annotation lacks and is read first.

- **One spelling**: a rank-1 length is a 1-tuple (``` ``(n_vertices + 1,)`` ```), never
  `Length-``X```, `Shape ``(…)```, `Flat …`, `Rank-1 ``(n,)```, nor a shape buried mid-sentence
  (`Smoothed ``(n,)`` positions`). A shape that is not the item's own (`cell ``(0, 0, 0)```) stays.
- **Dtype only where the annotation does not fix it** (rank-free or union annotations, a bare
  `wp.array`, "any float dtype"): the rendered annotation already says `wp.int32`.
- **Size names in full** (`n_faces`, `n_vertices`, `3 * n_faces`), never `f` or `V`; a free size
  (`m`, `k`, `n_unique`) is defined on first use, by the entry it sizes or a short clause.

Check 29 bans the four late openings; the rest is review.

**No measured timing belongs in a public function's docstring** — no millisecond figure, speedup
ratio, launch or byte count: they are facts about one box, Warp version and mesh. Keep the
*claim* in caller terms ("roughly doubles the call", "a host readback serialises the device
pipeline").

**No development-history narrative in `ordito/*.py`, in a docstring or a comment**: "measured X,
declined Y", "reverted", "round N", "probe"/"sweep" as methodology, cross-references to internal
doc sections. The public wrapper layer documents behaviour for a caller; keep the behavioural or
correctness fact (convention, sign rule, aliasing warning, what raises and when) and cut the
narrative. **Delete it, don't relocate it**: a decline's reasoning belongs in `kernels/`,
`benchmarks/`, `tests/` (§9: a measured decline is written at the site) or Part II, not in a
public wrapper. Zero tolerance: a fresh hit in a new or edited public function is a regression.

**The same holds for absolute numbers everywhere.** `kernels/`, `benchmarks/` and `tests/` may
carry a measured *conclusion* (ratio, crossover, share, "flat across a 256x range") but not a
wall-clock figure tied to one box and one mesh (`0.287 ms on a 40 962-vertex mesh`); Part II is
the one place for a raw cost model. Prose narrating how a change came about ("an earlier version
of this docstring", "round N") is development log; a regression test's "X used to be a bug, this
pins it" is the legitimate exception.

The docstring scan is an `ast` walk over `ordito/`'s public functions matching
`\d[\d.,]*\s*(ms|us|µs|ns|GB|MB|kB)\b` or `\b\d+(\.\d+)?x\b`; key on the *quantity*, not on prose
words like `measured` or `faster`. There is no comment-scanning counterpart; review catches
comments.

### Docstring syntax

Docstrings are **NumPy-style** (`Parameters`/`Returns`/`Raises`/`See Also`); cross-references use
**mkdocs-autorefs** syntax, not Sphinx roles (`:func:`, `:attr:`, `:meth:`, `:class:`, `:data:`,
`:mod:` render as literal text).

| Target | Syntax | Example |
|---|---|---|
| Internal (`ordito.*`) | `` [`short_name`][fully.qualified.path] `` | `` [`face_adjacency`][ordito.graph.face_adjacency] `` |
| External **with** inventory (`trimesh`, `numpy`, `scipy`, stdlib) | `` [`fully.qualified.name`][] `` | `` [`trimesh.grouping.group_rows`][] `` |
| External **without** inventory (`warp`, `igl`) | double-backtick code span, no link | ``` ``warp.sparse.BsrMatrix`` ``` |
| Shapes, literals, C++ names, paths | plain code span | ``` ``(n_vertices,)`` ``` |

- Resolve internal refs to the **fully-qualified path**, even within the same module (numpydoc's
  auto-linking of bare `See Also` names does not carry over).
- Before adding an external inventory to `mkdocs.yml`, verify `curl -I <url>/objects.inv`.
- RST admonitions do not exist in Markdown; use `!!! note`. **An admonition goes in free prose
  (leading description, `Notes`, `Examples`), never inside `Parameters` / `Returns` / `Raises` /
  `See Also` or any item-list section** (check 25): griffe reads each entry's first line as a
  *name*, so it renders as an exception type; `--strict` cannot see it.
- Module-level constants / type aliases without a docstring are linkable only because
  `show_if_no_docstring: true` is set; re-check `ordito/constants.py` and `ordito/typing.py`
  cross-refs before removing it.
- Every public function has a docstring (an undocumented one renders with an empty description).
- After editing docstrings run `grep -rnE ':(func|attr|meth|class|data|mod):\`' ordito/` — it
  must return nothing (`kernels/` counts too).

---
## 7. Testing

Every new geometry function MUST have regression tests comparing against a CPU reference
implementation. `trimesh` is the default; §7.6 lists the eight others and what each one binds.

### 7.1 Conventions

- Test file `tests/test_<module>.py`; import pattern:
  ```python
  import trimesh.<module> as tm
  import ordito.<module> as od
  ```
- Use the `device` fixture from `tests/conftest.py`. **A test takes `device` only if it touches
  the device directly** (builds device data itself); a test that reaches it through a mesh fixture
  does not name it. `pytest_generate_tests` parametrizes every test whose fixture closure reaches
  `device` (including via `request.getfixturevalue`), so `icosphere` alone runs on both devices.
  `reportUnusedParameter` (§8) enforces this.
- **Tests use pytest alone — never `unittest`.** Patch with the `monkeypatch` fixture
  (`monkeypatch.setattr(module, "NAME", value)`; `with monkeypatch.context() as patch:` for a
  patch scoped to a block), assert with plain `assert` and `pytest.raises` / `pytest.warns`.
  `monkeypatch` undoes every patch at teardown even when the test fails, and one spelling keeps a
  patch findable by one grep. Ruff's `TID251` bans `unittest` (`pyproject.toml`,
  `flake8-tidy-imports.banned-api`). **A test that counts or intercepts launches or allocations
  patches both paths**: the wrapper layer calls `ordito._launch.*`, which falls back to the `wp.*`
  function (and whose `launch_tiled` routes through its own `launch`), so patching `wp.launch`
  alone sees only the fallback (`test_successor_cycles_validates_before_launching`).
- Reproducible random data: `np.random.default_rng(seed)` with a fixed integer seed per test.
- **Upload a NumPy mesh with `conversions.numpy_to_warp(vertices_np, faces_np, device)`**, never a
  local helper (six private copies once existed). `numpy_to_warp_uv` is the `wp.vec2` sibling,
  `warp_to_trimesh` the inverse. Triangle soup `(n, 3, 3)` becomes an indexed mesh first:
  ```python
  vertices_wp, faces_wp = numpy_to_warp(
      tri_np.reshape(-1, 3), np.arange(tri_np.shape[0] * 3, dtype=np.int32), device
  )
  ```
- Call `.numpy()` on Warp outputs **inline**, before passing to NumPy comparisons; bind no
  intermediate variable for it.
- `np.allclose(got, exp, rtol=1e-5, atol=1e-5)` for floats; `np.array_equal` for bool/int.
- Variable suffixes by library: `_np` (NumPy/SciPy), `_tm` (trimesh), `_wp` (Warp), `_igl`, `_pp`
  (potpourri3d), `_pml` (pymeshlab), `_o3d` (open3d), `_pv` (pyvista), `_ml` (meshlib), `_pmf`
  (pymeshfix), `_p3d` (pytorch3d). Avoid `got` / `exp`; use clear names.
- NumPy 1D vector to a `wp.vec3` argument at Python scope: `wp.vec3(*array_np.tolist())`. (This
  is the test-side spelling only; `ordito/` production code drops the `.tolist()`, §3.8.)

### 7.2 Devices and the two-process runner

The `device` fixture is parametrized; `--device={auto,cpu,cuda,both}` selects for *this process*
(`auto` = cuda if available, like `benchmarks/conftest.py`; ids carry `[cpu]` / `[cuda0]`).

- **Both-device coverage is a two-process job: `uv run python -m tests.devices`** — a CUDA pass,
  then a CPU pass with `CUDA_VISIBLE_DEVICES=""`. Warp's CPU work is ~36x slower once CUDA is
  initialised in the process (§12.1). **Never use `--device=both` on a GPU box for CPU coverage.**
- **While developing, run `--device=cuda` only** (`uv run python -m pytest tests/test_<module>.py
  -q --device=cuda`). The CPU pass is ~3x the wall clock and CI runs it on every push.
- **Reach for `tests.devices` when the change is device-dependent by construction**: a
  `launch_tiled` kernel (§12.2), a `_device.prefers_tiled_reduction` branch, a device-gated
  constant (§13.3), anything taking `device=` from a dependency, cooperating lanes. The defect
  class "CUDA present *and* arrays on host" (e.g. a `warp.fem` ambient-device leak) is reached by
  neither single-device run; `wp.ScopedDevice(device)` is the fix when a dependency picks the
  device.
- **CPU is the deterministic oracle for byte-for-byte A/B**: float atomics serialize there, while
  on CUDA any reduction-order change is noise (§16.12). A measurement use, not a per-edit gate.
- **A test costing more than ~15 s on CPU wears `@pytest.mark.slow_cpu(<measured seconds>)`**,
  skipped on `cpu` unless `--device=both` (the runner's `--slow-cpu` asks for it). Only where the
  *device* is the cost and the claim is device-independent (four `screened_poisson` tests, §16.3);
  a test slow on both devices gets a smaller input.
- **A `launch_tiled` kernel's test pins `"cpu"` explicitly in a parametrize** — the fixture
  returns `cuda:0` whenever CUDA exists, leaving the CPU path unexercised (§12.2).
- **`--cpu-blocks` runs the CUDA tile code on the CPU** (`python -m tests.devices --cpu-blocks`
  adds it as a third pass): Warp's experimental `enable_cpu_blocks` gives a CPU block all its
  lanes, so every `launch_tiled` kernel and tiled reduction executes its CUDA path on the
  deterministic device (§12.2). Opt-in, for a change to a tiled kernel or to lane partitioning;
  each CPU block costs a fixed price per launch (whole suite 329 s against the plain CPU pass's
  119 s; green on first run, 2026-10-05).
- **When two devices differ but neither is wrong, compare both to a common oracle**, not to each
  other (`heat_signed_distance`: the cross-device gap is 5x below either device's discretization
  error, so no guard).

### 7.3 Mesh fixtures

Reuse `tests/conftest.py` fixtures (each returns `(mesh_tm, mesh_wp)`):

| Fixture | Use when |
|---------|----------|
| `icosahedron` | Default watertight solid, 12 vertices; inside/outside, sampling, sign tests |
| `icosphere`, `icosphere_coarse` | Closed, curved: `subdivisions=3` (642 v) and `2` (162 v) |
| `unit_box` | Sharp features: 12 exact 90° creases, 6 flat diagonals; crease/seam/dihedral tests |
| `cave_cube` | Hollow / non-convex shell (boolean difference) |
| `hemisphere`, `half_torus` | Curved open surfaces |
| `boy_surface` | Closed, watertight, **non-orientable**, χ = 1: the `False` branch of `is_orientable` / `face_orientation_bits`, `make_winding_consistent`'s impossible case |
| `mobius` | Non-orientable with a boundary: same predicates, one 78-edge loop, χ = 0 |
| `bohemian_dome` | Closed genus 1 that self-intersects: `homology_generators` at genus 1, `is_self_intersecting` on a closed input |

The last three (built by `creation.parametric_surface`) are the only non-orientable / odd-χ
inputs; a boolean predicate asserted only on orientable fixtures tests one branch.
`parametric_surface` builds thirteen more surfaces that are not fixtures yet: add one rather than
hand-rolling a degenerate mesh.

- **Do not** call `tm.creation.box()` or hand-roll `wp.Mesh(...)` unless the case needs a bespoke
  degenerate mesh. **Never construct a zero-triangle `wp.Mesh` on CUDA** (corrupts the allocator,
  §12.1).
- Prefer `icosahedron` / `cave_cube` over a simple cube. Parametrize over fixtures with
  `request.getfixturevalue(mesh_name)`. Use `mesh_wp.device` for query allocations when a mesh
  fixture is in scope.
- Edge-case tests (empty points/faces, single-triangle) may use minimal inline buffers; a
  **single-triangle** mesh is the safe way to reach an "empty mesh" guard on CUDA.
- **Fixture sets are shared**: `CLOSED_MESHES`, `OPEN_MESHES`, `MESHES = CLOSED_MESHES +
  OPEN_MESHES` in `tests/conftest.py`. Import them; keep a local list only for a genuinely
  different set, with a comment why (`test_adjacency.py` drops `cave_cube`: coplanar box faces make
  every adjacency angle 0 or pi/2).
- **`trimesh.slice_plane` output is a poor reference input**: a hemisphere from `icosphere(2)`
  reports 17 boundary loops in MeshLib (1 after `merge_vertices()`) and gains vertices in
  pymeshfix; duplicated rim vertices also made `boundary_loops`' pinch handling device-dependent.
  Use the conftest fixtures (`hemisphere` merges vertices) and **assert the hole count** before a
  per-hole comparison.

### 7.4 The parity gate: a benchmarked reference must be a tested reference

`benchmarks/` asserts only shapes and finiteness. `tests/test_parity.py` fails the default
`pytest` run when a benchmarked `(group, library)` pair is neither tested nor exempted. The
benchmark's `benchmark(group=...)` name is the key (a cross-suite API: renaming one breaks every
`parity` marker citing it). Both markers are stackable and take string **literals** only.

```python
# tests/test_edges.py -- "this test proves ordito agrees with trimesh for that group"
@pytest.mark.parity("faces_to_edges", "trimesh")

# benchmarks/test_curvature.py -- "timed, but the results are not comparable"
@pytest.mark.noparity("pymeshlab", oracle="trimesh", reason="MeshLab computes the Meyer/Desbrun "
                      "pointwise 1-ring operator, not the Cohen-Steiner/Morvan ball measure, so "
                      "its absolute value is not comparable; measured 0.982 correlation with a 7% "
                      "offset. trimesh is the oracle for this group.")
```

`uv run python -m tests.parity` prints the full matrix.

**Where a reference computes the same quantity, one test must compare the two outputs.** An
invariant (watertight, symmetric, idempotent) does not discharge it. **Check, do not assume**,
that the reference has the quantity (§7.6 lists what each binds; several plausible names are
absent). **Invariant checks are welcome in the same test** as the comparison (one test carries the
claim and the parity marker); split them out only when they need an input the comparison cannot
use.

**Where no reference computes the quantity, an invariant-only test is the honest answer**, and its
docstring says *"Not a library comparison: <why none exists>"* plus what the invariant excludes
(`halfedge_twins`, `homology_generators`' counting identity, `geodesic_walk` arc lengths). That is
a label beside A-D, not a class-D exemption (D is for a *benchmarked* pair that is incomparable).

**Classify every comparison in the docstring:**

- **A** — direct `np.allclose` / `np.array_equal`. The default.
- **B** — equal after a *named* transform, still at `1e-5`: a dict index, a unit fix
  (`igl.doublearea / 2`), a reduction, a projection (`igl.boundary_loop` is the longest loop),
  `lexsort` for unordered rows, a sign or gauge fix. Most apparent non-equivalence lands here;
  "does strictly less" / "upper bound" benchmark caveats are about cost, not value.
- **C** — a derived scalar, set distance or statistic (no correspondence exists). Must name the bug
  class it excludes and record a mutation probe and its margin (threshold ≥ 3x from measured
  agreement). `fraction_within` bounds must be shown to fail under shuffling one side.
- **D** — exemption. Only for: not an independent implementation (`oracle=` required); a
  different algorithm with a measured disagreement; a parameter the reference lacks; an answer not
  observable in isolation; stochastic with no invariant; input classes where ordito is undefined.
  **Not** admissible: "awkward", "tolerance would be loose", any class-B situation.

**Write the label as `Class A`** (capital). Check 19 gates it. **Four phrases are labels in good
standing**: `Class [ABCD]`, `Not a library comparison`, `Ordito against ordito`,
`Not a parity assert`; do not reword them to fit a narrower grep.

**Never a parity assert:** shape-only or `isfinite`-only; ordito compared with itself; a
threshold a constant output would pass. A boolean assert must be parametrized over inputs
producing both answers. **Ordito-against-ordito** is a legitimate test for one job: pinning two
entry points where only one has an oracle (mask vs index form, precomputed vs deriving path, CPU
vs CUDA); say which carries the oracle.

**Check the comparison is not vacuous on its fixture** (the gate cannot):

- **An empty answer**: `test_ears` compared `igl.ears` with `boundary.ears` where both found
  none (`[] == []`); making it non-vacuous exposed `ordito_opp == (igl_opp + 1) % 3`.
- **A constant answer is as vacuous**, and the docstring asserted the non-vacuity that was absent:
  a connected-component test said "several components" and produced 1; a curvature test "over
  every vertex" ran on a regular icosahedron (one value, spread 0.0), so a permutation or index
  swap passed. **Make the claim an assert**: `assert np.unique(labels_np).shape[0] > 1`,
  `assert np.ptp(reference) > 1e-3`.
- **Assert the reference's answer is non-empty / has its expected count** before comparing;
  "already the oracle elsewhere in tests/" is no evidence the comparison is live.
- **A docstring saying the vacuous branch is "covered separately" is a claim to grep for.**
  `test_face_nondegenerate_mask` named zero-area tests that did not exist; it is now parametrized
  over a clean and a degenerate arm with the expected count carrying the claim.

**A guard test "bites" only if you identify WHICH assertion fails under the mutation.** A k-NN
tie-break test "verified" by deleting a carry flag failed only its index assert (distances stayed
identical, and the docstring disclaims tied-neighbour identity), so it pinned a disclaimed
convention. If a test compares ordito with itself, ask what an external oracle would say;
**reproduce a plan's failure-mode claim before building a test around it.**

**A Class C threshold with large headroom does not bite.** The ≥ 3x rule is a floor against
flakiness, not a ceiling; four probed tests sat ≥ 10x above their own agreement. The probe that
measures what a threshold distinguishes re-runs the *reference* on a deliberately wrong input,
not a perturbed ordito side. Some cannot tighten (`heat_geodesic`'s 5 % bar: a third is genuine
discretization error in both methods; the two `heat_signed_distance` correlations have 1.19x
error-bound headroom on `hemisphere`).

**A rank correlation is scale-invariant, an error bound is not — a Class C test carrying both
carries two guards; say which catches what.** On `heat_signed_distance`: shuffling one side fails
the error bound but leaves correlation 0.73; negating fails the correlation only; scaling by 1.5
fails the error bound and leaves the correlation exactly unchanged.

**Mutation-probe boundary predicates.** A `greedy_downsample_mask` branch test did not bite when
`>=` became `>` because no spacing put two points exactly one step apart; "random plus a ties
case" does not reach an exact tie, a spacing of 0.25 with step 1.0 (exact in float32) does.

**A test hardcoding one value of the parameter the contract turns on tests the case that cannot
fail** (`test_query_nearest_empty` pinned `k = 2`, where the rank collapse is a no-op): parametrize
over the boundary value, not a comfortable one.

### 7.5 Shared helpers — check both modules before writing a private one

Reuse `tests/comparisons.py`: `lexsort_rows`, `assert_unordered_rows_equal`, `undirected_edges`,
`edge_multiplicity`, `euler_characteristic`, `open_edge_count`, `canonical_labels`,
`same_partition`, `canonical_winding`, `assert_same_up_to_sign`,
`assert_cyclic_permutation_equal`, `assert_same_loop_set`, `trimesh_outline_loops`,
`fraction_within`, `symmetric_chamfer`, `chamfer_two_sided`, `symmetric_surface_distance`,
`hausdorff_two_sided`, `hausdorff_surface_two_sided`.

And `tests/conversions.py`: `numpy_to_warp`, `numpy_to_warp_uv`, `points_to_warp`,
`points_to_warp_uv`, `trimesh_to_warp`, `warp_to_trimesh`, `trimesh_to_open3d`,
`points_to_open3d`, `open3d_to_trimesh`, `trimesh_to_open3d_t`, `trimesh_to_pymeshlab`,
`warp_to_pymeshlab`, `points_to_pymeshlab`, `trimesh_to_pyvista`, `points_to_pyvista`,
`pyvista_edges_to_indices`, `numpy_to_meshlib`, `trimesh_to_meshlib`, `warp_to_meshlib`,
`points_to_meshlib`, `meshlib_to_trimesh`, `numpy_to_meshlib_bitset`, `meshlib_scalars_to_numpy`,
`meshlib_indices_to_numpy`, `meshlib_bitset_to_numpy`, `numpy_to_pymeshfix`,
`trimesh_to_pymeshfix`, `warp_to_pymeshfix`, `pymeshfix_to_numpy`,
`pymeshfix_intersecting_faces`, `pymeshfix_face_remap`, `points_to_torch`, `numpy_to_pytorch3d`,
`trimesh_to_pytorch3d`, `warp_to_pytorch3d`, `points_to_pytorch3d`, `pytorch3d_to_numpy`,
`faces_igl`, `mesh_igl`.

- **`points_to_warp` and `warp_to_trimesh` are the most re-rolled** (the bare-cloud upload was
  written 403 times in four spellings). Authors reach for the shared helper when *building* the
  reference and hand-write the readback; use the helper both ways.
- **`canonical_labels`** is the label-packing transform every component comparison needs: ordito
  names a component by a representative element, igl/scipy number `0..k-1` in their own orders,
  VTK's `RegionId` in a third; only the *partition* is shared.
- **`symmetric_chamfer(mesh_a, mesh_b)` takes two meshes** and samples them itself;
  `chamfer_two_sided(points_a, points_b)` takes two already-drawn clouds (use it to compare two
  *samplers*) and the mesh one raises on point arrays. **Prefer the mean form over
  `hausdorff_two_sided` for distributional claims**: two samplings of `icosphere(2)` vs one scaled
  1.15 separate 6.8x by the mean and 1.4x by Hausdorff. `symmetric_chamfer` has a sampling noise
  floor (a mesh against itself is not 0); a threshold must clear it.
- **`lexsort` is unusable on float coordinates with ties** (float32 ties differ in float64 and
  order differently). For **positions** use a `cKDTree` nearest-neighbour match with a bijection
  check, or `hausdorff_two_sided`; keep `lexsort_rows` for integer index rows.

### 7.6 The nine reference libraries

**All nine are hard test dependencies: import them plainly, never via `pytest.importorskip`** (this
covers every package in `[dependency-groups] test`). A broken import must fail, not make tests
vanish — and a static `parity` marker would keep passing over the skip. Where a *runtime*
precondition cannot be declared (an EGL driver for `moderngl`), keep a skip that names the driver,
in the fixture that needs it, never at module scope over the import.

Aliases (pinned in ruff): `import trimesh as tm`, `import igl`, `import potpourri3d as pp3d`,
`import pymeshlab as ml`, `import open3d as o3d`, `import pyvista as pv`,
`from meshlib import mrmeshpy as mm` / `mrmeshnumpy as mn`, `import pymeshfix` /
`from pymeshfix import _meshfix`, `import pytorch3d.ops as p3d_ops` / `.loss` / `.structures`.

**Threading decides how a ratio reads.** Single-threaded: trimesh, igl, pyvista, pymeshfix.
Multi-threaded: `meshlib` (100+ OS threads), `pytorch3d-cpu`. GPU: `pytorch3d-cuda` only. A
`ordito-cpu` row loses to a threaded reference on any parallel op regardless of algorithm (§9:
decide on the CUDA number).

There are nine libraries and ten subsections: `promesh` is not a reference (nothing installs it).

#### libigl (`igl`)

Input convention matches ordito's (`float64` `(n, 3)` vertices, `int64` faces); bound functions
are pure except the stateful solver objects (`HeatGeodesicsData`, `ARAPData`,
`min_quad_with_fixed_data`, `AABB`), which cache a factorization and must be constructed
**inside** a timed callable.

- **An out-of-range face index is a SIGSEGV** (exit 139, no traceback): igl bounds-checks nothing.
  Never pass a reduced `V` with the original `F`. `igl.principal_curvature` on a non-manifold
  vertex crashes likewise; `heat_geodesics_precompute` / `harmonic` / `lscm` raise instead.
- **Three bound functions are memory-unsafe on ordinary input; check *values*.** `igl.loop`
  aborts on a 5-vertex mesh with three faces on one edge, SIGSEGVs on `bunny_decimated`, and on
  `bunny` returns a `NaN` row per unreferenced vertex. **`igl.in_element` is unusable** (misses
  interior queries, batch-size-dependent, aborts on a 200-point Delaunay): use
  `scipy.spatial.Delaunay.find_simplex` or pyvista's `find_containing_cell`. **`igl.upsample`
  corrupts the process heap on the scan meshes** (SIGSEGV lands later, in unrelated code;
  `--benchmark-json` is written at session end so it once destroyed a whole module's rows; worse
  with other libraries co-resident; compacting unreferenced vertices makes it worse). Safe on
  `icosahedron`, so a tested reference there, never a benchmarked one.
- **F-only functions size output by `F.max() + 1`, not `len(V)`** (`adjacency_matrix`,
  `vertex_components`, `is_vertex_manifold`), where `cotmatrix` / `gaussian_curvature` use
  `len(V)`: they disagree on meshes with unreferenced vertices; an F-only comparison is class B
  with the transform named. `igl.connected_components(igl.adjacency_matrix(F))` counts each
  isolated vertex as a component.
- **Signatures that fail silently.** `igl.exact_geodesic(V, F, vs, vt)` returns an *empty array*
  (a 4-argument call binds `vt` to `FS`): pass all six, face arrays as
  `np.array([], dtype=np.int64)`. `igl.knn(P, V, k, *igl.octree(V)[:4])` takes seven positionals;
  `in_element` needs a live `igl.AABB`; `crouzeix_raviart_*` need `(V, F, E, EMAP)` from
  `unique_edge_map(F)`; `average_onto_vertices`' `S` is a per-face *scalar*; `cut_mesh`'s `C` is a
  per-corner **bool** mask.
- **`collapse_small_triangles` and `resolve_duplicated_faces` are not bound** (only ~150 of ~493
  headers are): confirm a name exists in the wheel before planning a comparison. A hand port of
  the C++ in a test file is a legitimate *test* oracle (`test_metrics.py`, `test_polyline.py`,
  `test_seams.py`), never a benchmark row.

**Licensing:** libigl's core is MPL2, but everything under `reference/libigl/include/igl/copyleft/`
is **GPL** (CGAL booleans, `progressive_hulls`, `quadprog`, tetgen, `copyleft/marching_cubes`). No
ordito code may derive from that subtree; read the MPL2 top-level `marching_cubes.h`.

#### potpourri3d (`pp3d`)

Heat-method family, tangent spaces and isocontours (pybind11 over geometry-central; `float64`
vertices, `int32` faces).

- **Construct solvers with `use_robust=False` / `use_intrinsic_delaunay=False`** so both sides
  discretize the same triangulation (defaults mollify and flip to intrinsic Delaunay).
- **Tangent-space quantities are gauge-dependent**: frames agree up to a rotation about the
  normal; compare gauge-invariant combinations (holonomy around a face), never 2D components or
  single connection phases.
- **Barycentric output must be decoded.** `marching_triangles` returns `(element_index, coords)`
  in geometry-central's numbering: decode edges via `pp3d.edges(V, F)`, dispatching on
  `len(coords)` (0 vertex, 1 edge, 2 face). Closed curves repeat their first point; open ones do
  not.
- **It rejects some inputs with `RuntimeError`**: `MeshVectorHeatSolver` / `GeodesicTracer` / fast
  marching need a manifold mesh; `pp3d.edges` needs every vertex referenced. Pick fixtures
  accordingly.
- **A zero cotangent weight** (both opposite angles right, i.e. every diagonal-split quad grid:
  `cave_cube`, `half_torus`) erases that edge's phase from `get_connection_laplacian()`, so it
  cannot be an oracle there (`tests/test_tangent.py`).

#### pymeshlab (`ml`)

Broadest reference (281 filters). Build via `trimesh_to_pymeshlab` / `warp_to_pymeshlab`. Use it
where it is a *better* oracle than the incumbent. Every trap **fails green**:

- **Almost every filter mutates `current_mesh()` in place** (`apply_coord_*`, `meshing_*`,
  `compute_*_per_vertex`; `generate_*` pushes a new mesh). One MeshSet serves one filter call; build
  a fresh one per comparison, inside the timed callable (exception: selection filters, whose cost
  is independent of selection size). `compute_curvature_principal_directions_per_vertex` and
  `meshing_decimation_quadric_edge_collapse` default to `autoclean=True` and delete unreferenced
  vertices.
- **`get_*` returns a dict; `compute_*` / `meshing_*` / `apply_*` return `None`** (or stats): read
  the answer off `current_mesh()` (`vertex_matrix`, `face_matrix`, `vertex_normal_matrix`,
  `vertex_scalar_array`, `face_scalar_array`, `vertex_selection_array`, `face_selection_array`,
  `vertex_curvature_principal_dir{1,2}_matrix`, `edge_matrix`). Selections are bool arrays.
- **Length parameters take a wrapper**: `ml.PercentageValue(1)` = 1 % of bbox diagonal;
  `ml.PureValue(x)` = absolute length (no `AbsoluteValue`).
- **`harm_function` is a no-op** (`compute_texcoord_parametrization_harmonic` is bit-identical at
  1, 2, 3): never map ordito's `k` onto it.
- **Four defaults silently measure nothing**: `meshing_close_holes(maxholesize=30)` closes zero
  512-edge rims; `get_hausdorff_distance(samplenum=8)`; `generate_sampling_poisson_disk(radius=0%)`
  autoguesses; `generate_surface_reconstruction_ball_pivoting(clustering=0)` reconstructs nothing
  (and is *faster* for it). Assert on the returned dict and that the reference produced output.
- **`get_hausdorff_distance` `maxdist` defaults to a bbox percentage and returns `inf`** for
  farther pairs: pass `maxdist=ml.PureValue(1e6)`. Its `min` is a sound upper bound, tight
  whenever a sample lands on the witness (always for a convex polytope pair): a registered oracle
  for `mesh_to_mesh_distance` on that basis.
- **Silent no-op parameters, and one that runs backwards**: SDF `cone_amplitude` is identical at
  90 and 120 degrees; `apply_normal_smoothing_per_face` and `apply_scalar_smoothing_per_vertex`
  expose no parameters; `generate_resampled_uniform_mesh`'s `offset` as `PercentageValue` runs from
  full erosion (0 %) to full dilation (100 %), so its 50 % default is the **zero** offset (pass
  `PureValue(0.0)`). Probe a parameter before building an axis on it.
- **Non-manifold input raises**: `meshing_surface_subdivision_midpoint` and
  `generate_polyline_from_planar_section` fail on every scan mesh. Other preconditions:
  `compute_texcoord_parametrization_{harmonic,lscm}` need a boundary;
  `compute_matrix_by_fitting_to_plane` needs `set_selection_all()`;
  `compute_matrix_by_icp_between_meshes` needs both layers to carry faces;
  `face_face_adjacency_matrix()` raises `MissingComponentException` unless FF was requested;
  `generate_boolean_*` takes `first_mesh` / `second_mesh`.
- **Selection morphology is face-based**: `apply_selection_dilatation` / `_erosion` dilate *face*
  sets (a vertex selection is cleared). Bridge from a vertex seed with
  `compute_selection_transfer_vertex_to_face(inclusive=False)` (`True`, the default, selects only
  faces with every vertex selected). Dilate maps to `expand_vertex_mask`; **erode does not** map to
  `shrink_vertex_mask`.
- **`generate_surface_reconstruction_vcg` returns 0 faces** at every voxel size probed: rejected.
- **Two filters differ by definition**: `apply_scalar_smoothing_per_vertex` averages a boundary
  vertex over its two boundary neighbours alone (oracle on closed fixtures only);
  `apply_coord_two_steps_smoothing` at defaults moves a noisy cube further from clean (its fit
  step rounds corners).
- `MeshSet(verbose=False)` is default but ICP, the point-cloud normal estimator and the VCG
  reconstructor print anyway (pytest fd capture absorbs it).
- **`face_normal_matrix()` after `compute_normal_per_face()` is unnormalised** (`2 * area`): it
  checks normals *and* areas.
- **Pass counts are conventions**: Taubin `stepsmoothnum` counts lambda-mu *pairs* (ordito /
  trimesh do one half-step per iteration: `2 * stepsmoothnum`); `get_scalar_statistics_per_vertex`
  `"med"` is the element at `n // 2 - 1`, one below the middle, for both parities.
- **Two undocumented uniform umbrellas**: `apply_coord_laplacian_smoothing` and
  `apply_coord_unsharp_mask` weight each neighbour by shared-face count and include the vertex once
  (`1/(2d+1)` self, `2/(2d+1)` per neighbour, closed mesh); `apply_coord_taubin_smoothing` uses the
  plain 1-ring mean. 8 % displacement difference. **General technique: one pass of a linear
  filter is a linear map, so its stencil is solvable by least squares** over random position sets.
- **MeshLab writes a layer transform, not vertices**: `compute_matrix_by_icp_between_meshes`
  leaves `vertex_matrix()` unchanged; read `transform_matrix()` / `transformed_vertex_matrix()`.

**Licensing:** pymeshlab is GPL. `ordito/` may name it in prose; no ordito code may derive from
its source.

#### open3d (`o3d`)

Build meshes with `trimesh_to_open3d` (reusable, unlike a MeshSet), clouds with
`points_to_open3d`, tensor meshes with `trimesh_to_open3d_t`; never chain off an unbound
`from_legacy(...)`. The wheel is CUDA-built but legacy `open3d.geometry` / `open3d.pipelines` are
CPU-only; only `open3d.t` has GPU kernels.

- **Legacy `remove_*` / `orient_*` / `filter_*` mutate in place** (build inside the timed
  callable); pure `compute_*` / `get_*` / `is_*` recompute unconditionally and can share a mesh.
  **`get_volume` validates first** with the brute-force `IsWatertight` (seconds) and *raises* on
  non-watertight input: never benchmark it as "volume".
- **Tensor meshes must be held in a name**: `from_legacy(x).fill_holes()` lets the temporary be
  collected and reads freed memory (garbage floats, no exception).
- **`fill_holes` winds its cap against the rest of the mesh**; `trimesh.repair.fix_winding` before
  taking a signed volume.
- **k-NN distances come back squared** from `KDTreeFlann` and `o3d.core.nns`. Use
  `o3d.core.nns.NearestNeighborSearch` for batches (indices match `scipy.spatial.KDTree` on a
  tie-free cloud); the legacy tree is a per-point Python loop. `KDTreeFlann` radius search is
  **exclusive at exactly `r`**, ordito's ball queries are inclusive: only a constructed fixture
  exposes it.
- **`is_vertex_manifold` tests connectivity, not a fan**: three faces on one edge pass it and fail
  ordito's and igl's definition. Agreement is exact on edge-manifold input: restrict to that class
  and pin the divergence. `is_edge_manifold` shares `allow_boundary_edges` semantics.
- **Smoothing filters re-derive inverse-distance weights every pass** (`filter_smooth_laplacian`,
  `filter_smooth_taubin`): they match ordito's fixed assembled operator at one iteration and
  diverge by ten; Taubin's count is lambda-mu *pairs*. `filter_sharpen` adds the *unnormalized*
  residual (displacement is ordito's times the degree). All three are D exemptions.
- **Platonic solids come in rotated frames and odd scales**: octahedron matches exactly,
  tetrahedron is rotated, icosahedron is the raw `(0, ±1, ±φ)` table; compare rigid-motion
  invariants at unit circumradius. No `create_dodecahedron`.
- **`RaycastingScene.compute_signed_distance` shares ordito's convention** (negative inside, no
  negation, unlike trimesh). `compute_closest_points` diverges at equidistant-face ties: compare
  *distances*.
- **`get_oriented_bounding_box` is PCA of the hull** (minimizes nothing); comparable is
  `get_minimal_oriented_bounding_box`.
- **`remove_radius_outlier` is nondeterministic** (shared `KDTreeFlann` across an omp loop whose
  radius search is not thread-safe: three keep sets over eight runs), so not a class-A oracle.
  Evaluate its published rule (`count > nb_points`, self counted) through
  `search_radius_vector_3d` in a loop, which reproduces `points.radius_outlier_mask` exactly; the
  filter keeps only its benchmark row. No other `remove_*` shares the defect.
- **A down-sampler's output order is its own**: every legacy selection routes through
  `SelectByIndex`, emitting survivors in ascending index order, destroying
  `farthest_point_down_sample`'s greedy sequence: compare only the set. Where order is the claim,
  transcribe the C++ loop (strict `>` arg-max, lowest index wins ties). `num_samples=0` returns an
  empty cloud; `compute_nearest_neighbor_distance` reports **`0.0`** for < 2 points (honest answer
  `inf`).
- Benchmark gotchas: `remove_duplicated_triangles` mutates, returns `self`, idempotent;
  `compute_vertex_normals` overwrites but never caches; `select_by_index` takes **vertex** indices;
  `Vector3dVector` rejects non-writeable arrays (use `np.array(...)`, not
  `np.ascontiguousarray(...)`, on a trimesh `TrackedArray`); `fill_holes` is tensor-API only.

#### pyvista (`pv`)

The **VTK** reference (VTK 9.6 through pyvista). Build with `trimesh_to_pyvista` /
`points_to_pyvista`; one `PolyData` serves many comparisons (nothing cached). No renderer needed.
**Invoke probes from the repo root**: a `uv run` inside `reference/pyvista` builds a second
virtualenv. Licences MIT / BSD-3 (no subtree to avoid).

- **Float64 in, sometimes float32 out**: point storage and `regular_faces` (`(n, 3)` `int64`) are
  exact. `compute_normals`' `Normals`, `ray_trace` hits, `fit_plane_to_points(return_meta=True)`
  and `texture_map_to_*` are **float32**; `multi_ray_trace`, `principal_axes`, `curvature`,
  `compute_implicit_distance` are float64. Check dtype per row; use `atol` for differences of
  large numbers (the residual is ordito's float32 vertex buffer).
- **One filter of twelve mutates**: `edge_mask` writes `point_ind` into its input. Filters with
  `inplace` default `False`; never pass `True`.
- **An extraction renumbers its points**: `extract_feature_edges` returns only touched points in
  its own order (even on a unit box, all 8 reordered); go through
  `tests.conversions.pyvista_edges_to_indices`. `extract_cells` / `extract_points` / `threshold` /
  `clip_box` return `UnstructuredGrid` (maps in `vtkOriginalPointIds` / `vtkOriginalCellIds`);
  `split_bodies` / `bounding_box` / `oriented_bounding_box` return `MultiBlock` unless
  `as_composite=False`; `remove_points` / `collision` / `ray_trace` / `contour_banded` return
  **tuples**.
- **`cell_quality`: 12 usable measures of 28 on triangles, two naming inversions, one constant.**
  pyvista `radius_ratio` = ordito `aspect_ratio` (ordito's `radius_ratio` is its reciprocal);
  `shape` = ordito `mean_ratio`, `aspect_frobenius` = 1 / it; `condition` duplicates
  `aspect_frobenius`; `min_angle` / `max_angle` are **degrees**; `distortion` is constant **1.0**
  (a threshold test passes on anything); the 16 inapplicable measures return `-1.0` rather than
  raising. Decoding is in `tests/test_triangles.py`.
- **`is_manifold` is `n_open_edges == 0`, and `n_open_edges` counts boundary *plus* non-manifold
  edges**: `is_manifold` maps to `is_edge_manifold(allow_boundary_edges=False)`; the count does not
  map to `boundary_edges` (3 faces on one edge: 7 vs 6). **`DataSet.center` is the bbox centre**;
  `bounding_sphere` returns `(radius, center)` and is genuinely near-minimal (it coincides with the
  AABB-centre sphere on centrally symmetric meshes, which hid the wrong reading; differs on a
  hemisphere).
- **`curvature('maximum')` / `('minimum')` are algebra**: `H ± √(H² − K)` from VTK's own Gauss and
  mean curvature (difference 0.0 in float64, complex where `H² < K`), not an independent
  principal-curvature implementation. `curvature('gaussian')` is the angle defect over barycentric
  lumped area and a genuine oracle.
- **Surface operators return surface quantities**: `compute_derivative`'s gradient is tangential
  (mean `2/3 · e_x` for `f = x` on the unit sphere; matches `face_gradients` after
  `average_onto_vertices`) and stays on **points** for a point field even with
  `preference='cell'`.
- **Both smoothers are different algorithms**: `smooth` moves vertices along incident edges under
  VTK's convergence test (diverges with iteration count vs a fixed operator); `smooth_taubin` is
  windowed-sinc, warns *"An optimal offset ... could not be found"* on ordinary input, counts
  lambda-mu pairs. Both D exemptions.
- **Several answers are empty, constant or unchanged, so assert non-vacuity**:
  `clip_surface(pv.Sphere(radius=0.6))` returns 0 cells on `icosphere(3)`;
  `extract_values(0.010, scalars='area')` 0 cells (matches *exactly*; a range is `ranges=`);
  `edge_mask(30)` all-`False` on a smooth sphere (use a box); `integrate_data` of a symmetric
  point field ~0; `validate_mesh().coincident_points` **empty** on ten duplicated vertices (`clean`
  is the dedup oracle; a zero-area triangle lands in its `zero_size`, not `degenerate_faces`);
  `lines_from_points` gives one two-point cell per segment (so `compute_arc_length` restarts every
  segment and `decimate_polyline` is a no-op); `tube` / `ribbon` emit **strips** (`n_faces == 0`:
  `.triangulate()`); `extrude(capping=True)` leaves 16 open edges.
- **Where the reference put the answer**: `align(return_matrix=True)` returns `(aligned_mesh, 4x4)`
  and *does* move points (unlike MeshLab); `geodesic` puts the ordered path in
  `vtkOriginalPointIds` (Euclidean length = `geodesic_distance`); `sample` marks misses with
  `vtkValidPointMask` and `vtkGhostType`; `voxelize_binary_mask` writes a **point** array `mask` on
  a cell-centred grid (*solid*: its set is contained in ordito's `mode="solid"`). Deprecated in
  0.48.4: module-level `pv.voxelize` / `pv.voxelize_volume` (hard `DeprecationError`),
  `select_enclosed_points` (→ `select_interior_points`, array `selected_points`),
  `extract_geometry` (→ `extract_surface(algorithm=None)`), `n_faces_strict` (→ `n_faces`).
- **`multi_ray_trace` is trimesh + embree, not VTK** (same call as trimesh's own): a pyvista row
  for `intersects_*` groups would be a trimesh row in disguise. VTK's own `ray_trace` is
  independent but one ray per call (100x off embree): test oracle only.
- **`validate_mesh()`'s cell fields are per *cell*, not per mesh**: `intersecting_faces` is "two
  faces of a 3D cell", identically empty on triangle meshes (0 on interpenetrating icospheres and
  `bohemian_dome`); `inverted_faces` reads 0 with ten reversed faces. The firing field is
  **`zero_size`**, and `clean()` **keeps** those faces at every tolerance (detector, no filter).
  A degeneracy comparison needs a *scale-aware* input (an exactly collinear float64 face survives
  ordito's float32 altitude test, §12.4). `collision` is two-mesh and cannot see a
  self-intersection (thousands of hits for a mesh against its own copy).
- **`compute_implicit_distance` needs polygons**: on a line-set VTK logs *"No polygons to evaluate
  function!"* per query and returns a far-off field. Polyline distance goes through
  `find_closest_cell` on a **single-cell** polyline; its locator collapses on a long cell (fine at
  a few thousand queries vs a few hundred segments, minutes at 65 536 of each).
- **`find_containing_cell` is the working point-location oracle** (batched, `-1` outside, exactly
  equals `Delaunay.find_simplex` on 10 000 queries): igl's `in_element` failure does not
  generalize. `find_closest_cell` is the most accurate closest-point reference (float64-exact vs
  `igl.point_mesh_squared_distance`), but its **cell id is not comparable** (28 % of exterior
  queries differ from igl on shared edges): compare distance, at Warp's
  `mesh_query_point_no_sign` floor (§12.4).
- **Two filters answer a different question than their name**: `sample()` interpolates only where
  the query lands *inside* a source cell, and `snap_to_closest_point=True` snaps to the nearest
  *vertex* (worse); `delaunay_2d(edge_source=loop)` does not clip to the loop (covers more area
  than the polygon): the polygon-fill oracle is `triangulate_contours` (zero Steiner points,
  matches `polyline.triangulate_polyline`'s `n - 2` count and area to nine digits).
- **Parametric surfaces arrive open, and `clean` defaults differ per surface**:
  `surface_from_para(clean=False)` is the base default (all 21 have 156 or 236 boundary edges;
  raw `ParametricMobius()` is a disk) while pyvista overrides to `clean=True` on 9: **always pass
  `clean=True`**. **`klein` is not a Klein bottle** (welds to two boundary loops, reads
  orientable): only `figure8_klein` is closed non-orientable χ = 0. The mirror matches the wheel
  for `core/utilities/parametric_objects.py`; domains / periodicity live in VTK's
  `vtkParametric*` (`JoinU` / `JoinV` / `TwistU` / `TwistV`), not pyvista.

#### meshlib (`mm` / `mn`)

Build with `trimesh_to_meshlib` / `numpy_to_meshlib` / `warp_to_meshlib`, clouds with
`points_to_meshlib`, read back with `meshlib_to_trimesh`; never hand-roll `mn.meshFromFacesVerts`.
It is the **only multi-threaded CPU reference** (a fair fight for `ordito-cuda`), binds a real
Liepa/Klincsek `fillHole` with a 12-metric family and a two-loop `stitchHoles`, and is the only
oracle for `repair.collapse_small_triangles`. `meshlib.mrcudapy` is deliberately **not** used (it
would make rows incomparable).

- **Almost every free function mutates its `Mesh` in place and returns something else** (`relax`,
  `fillHole(s)`, `decimateMesh`, `remesh`, `subdivideMesh`, `fixMeshDegeneracies`,
  `filterCreaseEdges`, `denoiseNormals`, `smoothRegionBoundary`, `expand`, `shrink`: a status
  `bool`, count, `EdgeId` or new-faces `FaceBitSet`). One `mm.Mesh` serves one mutating call
  (`BenchCase.new_mesh_ml()` is a method). `marchingCubes` returns a fresh `Mesh`, so its input is
  cacheable.
- **`meshFromFacesVerts` takes faces *first*** (reverse of every other converter) **and sizes the
  vertex buffer by `F.max() + 1`**: a trailing unreferenced vertex is dropped, an interior one is
  kept but excluded from `numValidVerts`. Never pass a compacted `V` with the original `F`; never
  assume `getNumpyVerts(...).shape[0] == len(V)`.
- **`pack()` is mandatory before reading topology back**: after a decimation leaving 120 of 320
  faces, `getNumpyFaces` returns `last_valid_face_id + 1` rows, mostly `[0, 0, 0]`, silently.
  `meshlib_to_trimesh` packs by default.
- **`np.asarray` on a scalar container gives a 0-d `object` array** (`VertScalars`, `FaceScalars`,
  `UndirectedEdgeScalars`; `mn.toNumpyArray` accepts only `VertCoords` / `FaceNormals` /
  `std_vector_Vector3_float`). Use `conversions.meshlib_scalars_to_numpy`; for containers of **ids**
  use `meshlib_indices_to_numpy` (the scalar reader raises: `VertId` has `__index__` not
  `__float__`).
- **A returned bitset is only as long as its highest set bit, unpredictably**: `mn.getNumpyBitSet`
  reads at the set's own length (`findSelfCollidingTrianglesBS` returns a short array on a colliding
  pair and an **empty** one on a clean mesh), breaking `np.array_equal` by shape. Use
  `conversions.meshlib_bitset_to_numpy(bitset, size)`.
- **Every bitset converts in bulk, both directions; never a per-bit loop.** `TypedBitSet` derives
  from `MR::BitSet`, so `mn.getNumpyBitSet` upcasts any. Load with `BitSet.fromBlocks` on packed
  `uint64` blocks: `np.packbits(flags, bitorder="little")` (`numpy_to_meshlib_bitset`; wrap in the
  typed set). `bitorder="little"` is required; `fromBlocks` rejects a NumPy `uint64` array (pass
  `.tolist()`) and rounds size up to 64-bit blocks (`resize` back); a `VoxelBitSet` is `x`
  fastest, i.e. `dense.ravel(order="F")`. **"MeshLib binds no converter" means probe the base
  class.**
- **`(*args, **kwargs)` is an overload set `inspect` cannot see** (docstrings stripped): read real
  signatures by calling with one junk argument and reading pybind11's `TypeError`.
- **Overloads of one name can have opposite output conventions**: `expand(topology, region:
  FaceBitSet, hops)` returns `None` and **mutates `region`**, `expand(topology, f: FaceId, hops)`
  **returns** a new set (same for `shrink`); `stitchHoles(mesh, a, b, params)` takes named hole
  edges, `stitchHoles(mesh, params)` finds them (only argument count differs); `relax` takes a
  `PointCloud` or a `Mesh` with different params; one `getAllComponents` returns a
  `(components, count)` tuple. Resolve the overload and assert result type/count.
- **`findOutliers`' default mask segfaults on a cloud without normals** (`OutlierTypeMask.All`
  includes `AwayNormal`); the other three modes are fine.
- **A projector stores a raw pointer to its mesh/cloud, so a temporary segfaults**:
  `PointsToMeshProjector.updateMeshData(build_a_mesh())`, `PointsProjector.setPointCloud(...)`.
  Bind to a name outliving every query. `mm.MeshPart` keeps a real Python reference (safe over a
  temporary); **probe the refcount (`sys.getrefcount`), do not infer the lifetime.**
  `findProjections`' `upDistLimitSq`: pass MeshLib's `FLT_MAX`, `math.inf` segfaults.
- **The AABB tree is lazily built and cached on the `Mesh`**: first-vs-second `findProjection`
  differs 10-100x (most on the process's first mesh while the thread pool spins up). State whether
  the build is inside the timed callable; build outside and pre-warm with one throwaway query. A
  mutating call invalidates the tree. A `PointCloud` caches its point tree likewise
  (`cloud.invalidateCaches()` in `setup`).
- **Per-vertex free functions are per-*vertex*** (`discreteGaussianCurvature`, `sumAngles` take one
  `VertId`): use the batched `mn.getNumpyGaussianCurvature(mesh)` (bit-identical, 10-100x faster).
  No per-vertex Python loop in a benchmark row.
- **`getNumpyVerts` is float64 but storage is float32** (comparison floor ~1e-7).
  `computePerFaceNormals` is **normalized** (pymeshlab's `face_normal_matrix()` is not).
- **Parameter conventions that read as disagreement**: `sampleHalfSphere()` spans `z` from -1 to +1
  (not a half sphere; halves a sky-view answer); `InSphereSearchSettings.maxRadius` defaults to
  **1** whatever the scale (pass half the smallest bbox side); `makeUVSphere`'s
  `verticalResolution` counts interior latitude **rings**, pairing with
  `creation.uv_sphere(count=(v + 2, h))` (then identical vertex for vertex);
  `leftCotan(e)` is the **plain** cotangent by directed edge's left face (vs
  `laplacian.cotmatrix_entries`' *half* cotangent by `(face, corner)`; `cotan(ue)` sums both);
  `MarchingCubesParams.origin` addresses the voxel **centre** (march with
  `origin = lower - voxel / 2`); **`findNClosestPointsPerPoint` returns a heap**, ids exactly
  scipy's `k` nearest but ~91 % in distance order and the nearest **last** (use `numNei=1` for one
  neighbour).
- **`computeRayThicknessAtVertices` takes its direction from the *pseudonormal***: pair it with
  `visibility.thickness(method="ray", normals=angle_weighted_vertex_normals(...))` (five orders
  worse against area-weighted). The thickness functions take no query set, which keeps them out of
  groups whose input is a subsample.
- **Detectors are reliable oracles; several mutators are not.** `mm.eliminateTunnels` leaves the
  mesh byte-identical on every configuration probed while `detectTunnelFaces` /
  `detectBasisTunnels` answer correctly; `inflate(mesh, verts, InflateSettings)` takes the
  **unselected** vertices as Dirichlet (selecting all collapses `icosphere(3)` to the origin; a
  region solves a different problem). `findSpikeVertices` and `findInnerVertsOfDegree` are exact
  class-A oracles. **Run a mutator once and assert it changed the mesh**; else fall back to the
  invariant (χ for tunnels, volume monotonicity and normal alignment for inflation) and say what
  was probed.
- **`mm.localFixSelfIntersections` needs a *single-component* input**: on two welded spheres it
  returns its input unchanged at every configuration; on a single-component self-intersecting torus
  it mutates and still does not clear (doubles the colliding count). `mm.fixSelfIntersections`
  (voxel path) has no such limit. **Apply the same detector to both outputs**, and a benchmark
  asserting `numValidFaces() > 0` passes a no-op. **The fix was the fixture, not a skip** (give the
  group a single-component input when a reference declines an input rather than being wrong).
  Behaviour is fixture-dependent in both directions (clears on a trimesh torus, worsens on a
  MeshLib torus): probe the specific input.
- **Pairings not guessable from names**: `computePerVertNormals` matches
  `vertices.area_weighted_vertex_normals`; `computePerVertPseudoNormals` matches
  `angle_weighted_vertex_normals` (each to float32 rounding, orders of magnitude from the other's
  partner). `mn.getNumpyGaussianCurvature` is the pointwise **angle defect**: it pairs with
  `vertices.vertex_defects`, **not** `curvature.discrete_gaussian_curvature` (the
  Cohen-Steiner/Morvan ball measure).

**Licensing: MeshLib is not open source.** The wheel and `reference/MeshLib` are under AMV
Consulting's *"NON-COMMERCIAL & education"* agreement (terminable, non-transferable, commercial
licence required otherwise, bar on modifying or transferring), which restricts *use*, while ordito
ships `MIT OR Apache-2.0`.

**Nothing under `ordito/` may name MeshLib at all**: not the library, a function (`fillHole`,
`positionVertsSmoothly`, `triangleAspectRatio`) or a source file (`MRMeshDelone.cpp`,
`MRTriMath.h`). Describe what the code **computes** or name the algorithm in the literature's
vocabulary ("the Liepa/Klincsek interval DP", "the Delone empty-circumcircle test", "circum-radius
over twice the in-radius"). Read `reference/MeshLib` for an operation's *interface*, never to port
its body. It stays a test/benchmark dependency. **Check 21 enforces it**: it keys on `meshlib` /
`mrmeshpy` / `mrmeshnumpy`, on an `MR<CamelCase>` prefix generically, and on MR-less C++
identifiers (`FanOptimizer`, `buildLocalTriangulation`, `positionVertsSmoothly`,
`calcQueueElement_`, `updateBorderQueueElement_`). Add to the pattern when a new symbol is found;
never narrow it.

#### pymeshfix (`_pmf`)

nanobind over Attene's MeshFix / TMesh. Build every `PyTMesh` via `numpy_to_pymeshfix` /
`trimesh_to_pymeshfix` / `warp_to_pymeshfix`; read back with `pymeshfix_to_numpy`,
`pymeshfix_intersecting_faces`, `pymeshfix_face_remap` (the raw calls are unsafe three ways).

It is the **narrowest deep** reference: 19 `PyTMesh` members, **nine algorithms**
(`fill_small_boundaries`, `select_intersecting_triangles`, `strong_degeneracy_removal`,
`strong_intersection_removal`, `clean`, `remove_smallest_components`, `join_closest_components`,
`fix_connectivity`, module-level `clean_from_arrays`), all *repair*. It is the only reference that
goes arrays of a broken surface in to a single watertight solid out. No curvature, geodesic,
parametrization, registration, reconstruction, decimation, remeshing, point-cloud, boolean,
proximity or signed-distance entry point; `cutAndStitch`, `iterativeEdgeSwaps`, `loopSubdivision`,
`isInnerPoint`, `openToDisk`, `marchIntersections.cpp` are in headers and **unbound**.

- **`load_array` is already a repair, and it renumbers.** It runs the connectivity fix and Euler
  update: trailing *or* interior unreferenced vertices are dropped; an exactly duplicated face is
  **kept** and its non-manifold edges cut (both counts *rise*), a *reversed* duplicate is refused
  and vertices still cut; coincident *referenced* vertices are **not** merged; one backwards face
  is rewound, *every* face backwards is left alone (consistent is not outward). Scan meshes gain or
  lose vertices/faces on load; a non-orientable closed surface is cut along its
  orientation-reversing seam (two coincident sheets), so `select_intersecting_triangles`
  over-reports there. **Every comparison must be index-free** (positions, canonically sorted rows,
  sets, counts); where a face index is unavoidable use `pymeshfix_face_remap` (checks the load,
  refuses rather than guessing). `load_array` is itself an oracle for
  `remove_unreferenced_vertices`, `make_winding_consistent`, partly `split_non_manifold_vertices`.
- **One `PyTMesh` serves one load and one mutating call** (a second `load_array` raises; every
  algorithm mutates in place).
- **`select_intersecting_triangles` returns a mostly-uninitialised array**: `(n, 3)` `int32` with
  the `n` indices in the **flat** prefix and `2n` entries of heap garbage. The only defined read is
  `out.ravel()[: out.shape[0]]`.
- **`tris_per_cell` and `justproper` are no-ops**: pass both explicitly so a wheel honouring either
  fails a test; build no ordito flag around `justproper`.
- **`nbe` is inclusive, both docstrings say otherwise, pymeshlab's is exclusive**:
  `fill_small_boundaries(nbe, ...)` fills loops of **at most** `nbe` edges (`nbe = 0` = all);
  pymeshlab fills the same rim at `maxholesize = nbe + 1`. `holes.fill_small(max_edges=...)`
  follows pymeshfix (precedence rule below); a pymeshlab comparison passes `max_edges + 1`.
- **The "MeshFix could not fix everything" stderr line is printed on *success*** (inverted;
  `set_quiet` does not suppress it). Read the boolean or the mesh.
- **`remove_smallest_components` ranks by face count** (not area/diameter), returns the number
  removed, and always reduces to one component: the rule
  `repair.remove_small_components(keep_largest=True)` defaults to.
- **The output face buffer is a reordering even when nothing was repaired**: vertices byte-identical
  in float64 and the triangle set identical under `np.sort(rows, axis=1)` plus lexsort, but rows
  and starting corners differ. Never compare face buffers positionally.
- **`n_boundaries` is a property in 0.18.1 and `boundaries()` raises** (`n_points` / `n_faces`
  too); pre-0.17 examples fail with `TypeError: 'int' object is not callable`.
- **`strong_degeneracy_removal` measures in `double`, stricter than ordito's `float32`**: exactly
  collinear vertices are removed by both; the same strip offset by `1e-9` is removed by ordito,
  kept by pymeshfix. Compare on an *exactly* degenerate fixture and pin the near-degenerate class.
- **`strong_intersection_removal` is a different algorithm from
  `repair.fix_self_intersections(method="local")`**: on a self-intersecting torus ordito cuts and
  refills each sheet (two closed components) where pymeshfix removes far more (one). Neither
  benchmarked nor a parity claim. The comparable level is `repair.make_solid` vs
  `clean_from_arrays` (float32 rounding, identical counts on `bunny_decimated`).
- **Reproducing `clean_from_arrays` needs the loader's repair as an explicit first stage**
  (`remove_unreferenced_vertices` + `make_winding_consistent` + `split_non_manifold_vertices`),
  else the result has χ = 2, one component, and is **not watertight**. The component filter must
  run *inside* the intersection loop as well as before it; **nothing geometric may run after the
  final fill** (a 3-vertex rim fills with a sliver, a degeneracy pass deletes it and reopens the
  rim, for ever).
- `trimesh.slice_plane` output is a poor input (§7.3).

**Benchmark rule: the load is most of the row.** The build must be inside the timed callable
(`BenchCase.new_tmesh_pmf()`). **Create a `pymeshfix` row only where the operation is ≥ ~30 % of
the round, stating the share in the group docstring**; the intersection family and
`clean_from_arrays` qualify; hole-fill and component-removal comparisons carry
`pytest.mark.parity(<group>, "pymeshfix", benchmarked=False, reason=...)` with the ratio. Cap at
`bunny`.

**Licensing: pymeshfix is GPL-3.0**, and the TMesh headers under `reference/pymeshfix/src/` are
GPLv3 *or* a commercial agreement with IMATI-GE/CNR. Different from MeshLib:

| | MeshLib | pymeshfix |
|---|---|---|
| May `ordito/` name it? | **No** | **Yes**, in `Notes` / `See Also` |
| May `ordito/` derive from its source? | No | **No** |
| May `tests/` and `benchmarks/` import it? | Yes | Yes |
| Why | proprietary, restricts *use* | copyleft, restricts *distribution of derivatives* |

Read `reference/pymeshfix/src/` for interface and parameters, never the body; cite the **paper**
(Attene, *"A lightweight approach to repairing digitized polygon meshes"*, Visual Computer 26,
2010; Liepa, *"Filling holes in meshes"*, SGP 2003 §3; Barequet & Sharir 1995). Keep
`grep -rnE 'MeshFix|Basic_TMesh|TMesh|_meshfix' ordito/` empty (prose `pymeshfix` is allowed, C++
symbols are not).

> **Precedence rule: where pymeshfix and MeshLib both answer a question and differ, ordito's
> default is pymeshfix's answer and MeshLib's is reachable by a flag** — not the reverse. Where
> only MeshLib answers, nothing changes.

A function whose behaviour is pinned against MeshLib alone has its specification in a proprietary
binary nobody may read. It currently bites `holes.fill_small`, which takes a **perimeter**
threshold because MeshLib's `fillHoles` does, where pymeshfix and pymeshlab take a
**boundary-edge count**.

#### pytorch3d (`p3d_ops` / `p3d_loss` / `p3d_structures`)

Build via `trimesh_to_pytorch3d` / `numpy_to_pytorch3d` / `warp_to_pytorch3d`; clouds via
`points_to_pytorch3d` (`loss` container) or `points_to_torch` (bare batched tensor for `ops`); read
back with `pytorch3d_to_numpy`. It is the **only reference with CUDA kernels**, so two `LIBRARIES`
rows (`pytorch3d-cpu` / `pytorch3d-cuda`) and the one GPU-vs-GPU comparison. Nothing in the shipped
package may import torch (a 2.5 GB install).

- **Everything is batched; the wrap fails silently.** `knn_points(p, q)` on a bare `(P, 3)` reads
  `(N=P, P1=3, D)` and compares three points at full speed (a *faster* wrong answer).
  Pass `x[None]`,
  read `result[0]`; assert the reference's output **shape** before its values.
- **Every neighbour / Chamfer distance is squared** (`knn_points().dists`, `ball_query().dists`,
  `loss.chamfer_distance`, both `point_mesh_*` scalars): take the square root (0.0 after, host;
  1.19e-07 on CUDA).
- **`torch.cuda.is_available()` is the wrong probe for the CUDA extension**: a wheel whose arch
  list stops short returns `True` then fails every kernel ("no kernel image"); a CPU-only build
  (no `CUDA_HOME`, the normal CI state) raises `RuntimeError: Not compiled with GPU support.`.
  `benchmarks/conftest.py` gates `-cuda` on launching a two-point `knn_points`.
- **`Meshes` / `Pointclouds` are immutable *caching* containers** and every `ops.*` / `loss.*` is
  pure, so one object serves many comparisons; but `verts_normals_packed`, `edges_packed`,
  `faces_packed_to_edges_packed`, `laplacian_packed`, `faces_areas_packed` memoize on first request:
  a *benchmark* row naming one must build the container **inside** the timed callable. The answer
  lives on the `*_packed()` accessors.
- **It does not cast for you, and one entry point casts anyway**: a float64 `Meshes` stays float64
  (hence float32 in the converters, else ordito looks ~1e-7 wrong);
  `ops.mesh_face_areas_normals` returns **float32** regardless. Faces are int64.
- **`corresponding_points_alignment` / `iterative_closest_point` are row-vector** (`s·X·R + T =
  Y`): `R` is the transpose of `registration.procrustes`' linear block divided by scale (2.5e-07);
  `T` needs no transform. Compare the converged transform and rmse, never iteration count.
- **`ops.cot_laplacian` mixes conventions**: off-diagonal is **twice** ordito's half-cotangent
  table, diagonal identically **0.0** (`laplacian.cotmatrix` assembles the row sum); its second
  return is `1 / inv_areas == 3 * M_ii`. Both cancel in `mesh_laplacian_smoothing`'s ratios.
  `ops.laplacian` writes **-1** on the diagonal (`laplacian.laplacian(equal_weight=True)` writes 0);
  `ops.norm_laplacian` **is** `laplacian(equal_weight=False)` before row normalization (same
  formula and `eps = 1e-12`, 1.49e-08 after dividing by the row sum). **It never coalesces**: the
  `cotmatrix` ratio is a scope mismatch (§16.6).
- **`ops.marching_cubes` does not exist** at `pytorch3d.ops` (only `pytorch3d.ops.marching_cubes`).
  With `return_local_coords=False` it emits lattice indices, which is `levelset.marching_cubes`'
  default: no convention fix needed.
- **`ops.cubify`'s three `align` modes are one uniform scale and translation apart**: on a 6³
  occupancy sphere `"topleft"` / `"corner"` / `"center"` give the identical face buffer with bounds
  `[-0.6, 1.0]`, `[-0.667, 0.667]`, `[-0.8, 0.8]`, all reachable via
  `voxels.from_cells(cells, voxel_size, origin)`. It also **compacts**.
- **`packed_to_padded` / `padded_to_packed` bounds-check nothing and corrupt the heap** on a
  mismatched `max_size` / `total_size` (process dies in `malloc`/`free` later). Their `first_idxs`
  are starting indices (= `array.pack_1d_arrays`' `offsets`, `[0, 3, 8]` for lengths `(3, 5, 2)`);
  read sizes off the buffers.
- **`sample_points_from_meshes` caps faces at 2²⁴** (`torch.multinomial` category limit):
  `lucy`'s 28 055 742 raise `RuntimeError`; `happy_buddha`'s 1 087 716 are fine.
- **`add_points_features_to_volume_densities_features` is `[-1, 1]` local space, `[z, y, x]`
  storage, `rescale_features=True`**: lattice is the **transpose** of `voxels.splat_onto_grid`'s,
  `align_corners=True` is ordito's `bounds`, and the default divides by `density.clamp(min_weight)`
  (average, not accumulation). Lined up, 0.0 on host, 4.8e-07 on CUDA.
- **`mesh_normal_consistency` counts pairs**: an edge with `k` faces contributes `C(k, 2)` terms;
  `adjacency.face_adjacency` keeps only exactly-two-face edges. Equal on edge-manifold input
  (0.0155947 over 480 pairs), 0.777 vs 0.0 on three faces sharing an edge.
- **`ops.taubin_smoothing` rebuilds and row-normalizes its operator every half-pass**: a fixed
  operator sits 4.1e-03 / 6.5e-03 / 9.9e-03 away at 1 / 3 / 10 iterations;
  `smoothing.filter_taubin(recompute=True)` closes it to 2.4e-07 at ~23x the cost. `num_iter`
  counts lambda-mu **pairs**.
- **`ico_sphere` is not the rotated-frame hazard**: same `(±0.5257, ±0.8507, 0)` table and
  subdivision as `creation.icosphere`; the 5.8e-05 residual is pytorch3d's four-decimal table.
  Match by nearest vertex with a bijection check at 1e-4. `utils.torus` takes the **minor** radius
  first and builds vertices in a Python double loop (needs a real parameter mapping; its column is
  a per-vertex Python floor).
- **Every `laplacian_matrices` entry point builds a COO tensor, warning once per process**
  (*"Memory errors (e.g. SEGFAULT) will occur ..."*, from `ATen/Context.cpp`): a statement about
  torch's global state, so §7.7's "solve for the input" does not apply. `tests/conftest.py` calls
  `torch.sparse.check_sparse_tensor_invariants.enable()` (beside `STRICT`; a malformed tensor from
  a future upstream-`main` pin should raise by invariant, not SIGSEGV). `BenchLibrary.run` calls
  `disable()` on a `pytorch3d` row: the checks cost 1.04-1.07x of a sparse construction and those
  ops build theirs inside the timed call.

**CPU rows are Θ(N²) on any neighbour query** (no spatial structure on either device):
`knn_points` 299.8 / 1 189.5 / 4 562.8 ms and `chamfer_distance` 557.8 / 2 319.1 / 9 077.1 ms at
10 k / 20 k / 40 k self-queries (3.84-4.16x per doubling; ~3.5 s per `knn` round on `bunny`, ~9
minutes on `dragon`): a `pytorch3d-cpu` neighbour or chamfer row is capped at a feature mesh.
**The GPU ratio is a crossover, not a bar**: brute force with perfect coalescing beats a BVH
descent while the problem fits the bandwidth (2.31 vs ordito 3.29 ms at 20 000 points,
`knn_points`, 0.70x; 74.65 vs 0.87 at 200 000, 85x), so neighbour and chamfer groups need the
point count as an **axis**.

**Every CUDA row must synchronize torch's stream** (`wp.synchronize_device` syncs Warp's only):
`BenchCase.run` branches on `kind == "pytorch3d"`. Put `torch.cuda.empty_cache()` in that teardown
(torch never frees device memory; 16 ordito rows once failed to allocate 65 368 bytes on a 32 GB
card after an uncapped pytorch3d row, §15.4).

**Licensing:** pytorch3d and torch are BSD-3: `ordito/` may name it and derive from it with
attribution; it stays a test/benchmark dependency.

#### promesh — not a reference library, not citable from `ordito/`

`reference/promesh/` is a bare source drop with **no `LICENSE`, `COPYING` or `pyproject.toml`**,
not a published package (`import promesh` fails); it can never be tested or benchmarked. Treat it
as a design mirror only, like reading MeshLib for an *interface*.

**Nothing under `ordito/` may name it**, the mirror image of MeshLib's rule: its terms are
unknown, so a "port of" comment cites terms nobody has checked. Describe the algorithm instead
(the gap-bridging problem of **Barequet & Sharir (1995)**, minimal-perimeter heuristic with a
longest-increasing-subsequence monotonicity correction). **Check 21 covers both libraries in one
scan** (additionally keying on `promesh` and `triangulate_boundaries`), and its test pins the
*replacement* wording as a negative so the pattern is never widened onto the literature's
vocabulary. `reference/promesh/deformation.py` attributes its mollification helper to
`kentechx/HoleFillingPy` (**MIT**), which is why `kernels/laplacian.triangle_inequality_slack`
cites that upstream, not promesh.

### 7.7 Where the reference put the answer

A reference that looks like it *disagrees* more often had its result read from the wrong place, or
was handed something other than what you thought.

- **Ask what a reference was handed, and whether it finished the job, before reading a ratio**
  (`chamfer_backward`: the *inputs* differed; `cot_laplacian`: the *output* is a different object,
  an uncoalesced COO tensor without diagonal, §16.6).
- **A reference's zero is not always "off", and its warning is usually about *our* input.** A
  `RuntimeWarning: invalid value encountered in divide` from `trimesh/triangles.py:659` (dividing
  by `d1 - d3`, the squared length of edge AB: a zero-length edge) was misread as reference
  numerics; `screened_poisson(point_weight=0.0)` returned 27-64 zero-area triangles every run, and
  the warning's intermittency only reflected whether a query's nearest triangle was one of them.
  **Read the reference's source at the warned line, solve for what input makes it fire, then
  assert that property on the buffer we passed** (`face_nondegenerate_mask`, not
  `-W error::RuntimeWarning`): it fires every run.
## 8. Tooling and local validation

Ruff and basedpyright are configured in `pyproject.toml` and are the authority for style and
typing — do not hand-roll equivalents or add competing tools. Run them and the tests after any
change to `ordito/` and before considering work done.

### Ruff — lint + format

Ruff is the **only** linter and formatter. Config lives in `[tool.ruff]`; do not override it inline.

```bash
uv run ruff format ordito tests      # format (100-col, skip-magic-trailing-comma)
uv run ruff check --fix ordito tests # lint + autofix
```

- Respect `[tool.ruff.lint]` rule families and per-file ignores; no blanket `# noqa` for a selected
  rule. Prefer a scoped `per-file-ignores` entry over inline suppression. The one standing
  exception is §1.6's kernel accumulator (`# noqa: UP018, RUF046`).
- Import conventions are enforced (`[tool.ruff.lint.flake8-import-conventions.aliases]`; aliases in
  §7.6). Docstrings are enforced (`D`): every public function needs one (§6).
- `reference/` is excluded from lint and type-checking — never edit vendored code to satisfy a
  tool.
- **`F811` is a dead-code check, not a collision check.** A duplicate `def` trips it only if the
  first binding is unused; a duplicate whose first binding *is* used is invisible, and the last
  definition wins for every call site including those above it. Duplicate module-level constants
  are never linted. **After any module merge, run an AST scan for repeated top-level
  `FunctionDef` / `ClassDef` / `Assign` names and confirm the test count is unchanged.**

### basedpyright — type checking

basedpyright is the configured type checker (what the IDE runs). Config: `[tool.basedpyright]`.

```bash
uv run basedpyright
```

- **`ordito/kernels/` is excluded**: the kernel DSL is not modelled by any stubs. Do not make
  kernels type-clean or add `# pyright: ignore` there.
- **`strict`, less the five `reportUnknown*` rules.** Warp's Python-scope stubs leave most of
  `wp.array` untyped (`.device`, `.shape`, `.numpy()`, slicing), so every expression over an
  array is partially `Unknown`: 4 844 of strict's 4 967 errors on `ordito/` when it was adopted
  (`arr.device` alone ~2 000). Every other strict rule is on, including `reportArgumentType`,
  `reportAttributeAccessIssue`, `reportOperatorIssue` and `reportPrivateUsage` on `ordito/`.
  Read the config, not this prose; it carries the counts.
- **No local Warp stubs — decided, not overlooked.** A generated `typings/warp/_src` stub set was
  built and removed. A `stubPath` never reaches a derived library (its checker reads Warp's own
  stubs), so ordito would be checked against types its users do not see. Shipping it as a PEP 561
  partial `warp-stubs` package works only with Warp's `__init__.pyi` copied in verbatim (without it
  every Warp name reads `Unknown`), needs a second PyPI distribution, and pins private `warp._src`
  internals to one Warp minor. The gaps that can be closed in one place are closed in
  `ordito/typing.py`; the rest take a `cast` at the site.
- **`ordito/typing.py` is where Warp's typing gaps are closed**, so a call site does not need
  `Any`:
    - `odt.Kernel` (`wp.Kernel | Callable[..., None]`): `wp.kernel` has no return annotation, so
      a decorated kernel reads as the Python function it wraps.
    - `odt.BsrMatrix[Block]`, a `TYPE_CHECKING`-only subclass of `wps.BsrMatrix` declaring the
      storage fields `bsr_matrix_t` generates at runtime (`nrow`, `offsets`, `values`, ...), with
      `Block` the element dtype of `values`. Warp parameterizes by a phantom
      `BlockType[Rows, Cols, Scalar]` no function returns. `odt.CsrMatrix` (`float32 | float64`)
      and `odt.SparseMatrix` (`CsrMatrix | BsrMatrix[mat22d]`) are the solver-facing unions;
      `odt.has_blocks(matrix, dtype)` is a `TypeIs` narrowing both branches.
    - **Typed views of Warp functions** (`odt.bsr_mm`, `bsr_mv`, `bsr_axpy`, `bsr_diag`, ...,
      `normalize`, `cross`, `dot`, `transform_point`): the Warp function itself bound once through
      `cast` to a `Protocol` spelling its signature over concrete types (zero call cost). Protocol
      `__call__` parameters are exempt from `reportUnusedParameter`, which a `TYPE_CHECKING` `def`
      redeclaration is not. Call `odt.bsr_*`, never `wps.bsr_*`, on a ordito-typed matrix.
    - `odt.as_dense` (slice narrowing, below) and `odt.vec3_floats` (Warp types `vec3[i]` as
      `vec_t | bool | float32 | int32`; at Python scope it is a `float`).
- **Type variables versus unions — measured, and the root of most remaining casts.**
    - A **union argument solves neither a plain nor a constrained `TypeVar`**
      (`BsrMatrix[float32] | BsrMatrix[float64]` into `BsrMatrix[B]` is an error). A function
      accepting "any of these" takes the union alias; one that returns its argument's own type
      uses a **`TypeVar` bounded by `wp.array`** (`ArrayT` in `_launch.clone`, `odt.ensure_ndim`),
      which does accept a union and returns it; or it is an **overload set**, across whose arms a
      union argument expands (`linalg._owned_copy`).
    - **A dtype parameter is `dtype: type[odt.Block] = wp.float32`** returning `BsrMatrix[Block]`
      / `wp.array[Block]`: the checker solves from the default when omitted, exactly when given,
      and to `Unknown` (assignable) for a runtime `type`. The constrained `wp.Float` would bind a
      runtime `type` to `float`.
    - **Never add `float32` / `float64` overload arms ahead of a generic one**: an `Unknown` or
      `Any` argument binds the first arm silently (a `float64` builder then reads as `float32`).
    - A predicate that also checks other conditions narrows with `TypeGuard` (positive branch
      only; `linalg._one_block_eligible`), never `TypeIs`.
    - Heterogeneous values that are only forwarded, hashed or duck-typed are `object`, not `Any`
      (`require_same_device(**named: object)`, cache keys `tuple[object, ...]`); a sequence
      parameter is `Sequence[object]` (a `list` is invariant).
- **Rank `Any` is the one deliberate `Any`.** `NDim` is invariant, so `wp.array[wp.float32]`
  (`array[float32, int]`) and `odt.Array1dFloat32` are not assignable in either direction, and a
  `TypeVar` used once in a signature is itself an error. Convention: **accept wide, return
  narrow** — a *parameter* takes the `Any`-ranked `odt.ArrayNd*` family, a *return* keeps the
  `Literal`-ranked `odt.Array1d*` / `Array2d*` aliases; the dtype still discriminates.
    - **`wp.empty` binds silently to an overload's first arm** (`Unknown` satisfies every arm).
      Allocate through `_launch.empty` / `odt.empty_1d`, which carry the dtype.
    - **Narrowings are two shapes.** A Python-scope *slice* is always a dense `wp.array`:
      `odt.as_dense` narrows it with a real `isinstance` (`wp.indexedarray` is not a subclass). A
      gather (`src[indices]`) *is* an `indexedarray` (§3.4) and is materialized with `wp.copy`.
      `reportUnnecessaryCast` is every cast's staleness check.
- **Order of preference:** a correct annotation or helper in `odt` > `cast` > a scoped
  `# pyright: ignore[rule]` with its reason. In `_launch.py`'s hot path an ignore beats a `cast`
  (a `cast` is a function call per launch); that module also carries one file-level
  `reportPrivateUsage=false`, since driving Warp internals is its purpose. Cross-module private
  uses that api_conventions check 5 allowlists carry the matching scoped ignore.
- **`reportPossiblyUnboundVariable` is an error** — it catches §1.4's conditional-scope gotcha. On a
  *correlated* condition initialize to `None` before the branch and `assert x is not None` at the
  use (`ordito/registration.py`). Never suppress it.
- **Two rules above strict are ON:** `reportUnnecessaryTypeIgnoreComment` and
  `reportUnusedParameter` (which covers `tests/` and `benchmarks/`, enforcing §7.1's `device`
  rule). A protocol-signature parameter is `_`-prefixed (`matvec`'s `_y`); a `parametrize` value
  used only as a label goes in `pytest.param(..., id=)` / `ids=`.
- **Measured and declined:** `reportUnreachable` flags exactly §4.2's off-menu `Literal` guards;
  `reportImportCycles` flags §3.1's lazy `__init__`; `reportUninitializedInstanceVariable`
  misreads `__slots__`.
- The gate stays at **0 errors**. An error is almost always a real possibly-unbound bug or a
  missing dependency — resolve it, do not widen the disabled-rule list.

### Full environment

Test/reference dependencies live under `[dependency-groups] test`, **not** `[project]
dependencies` (ordito itself needs only `warp-lang`). There is no `[tool.uv] default-groups`, so a
bare `uv sync` does **not** install them and uninstalls trimesh/pytest:

```bash
uv sync --all-groups
```

- `uv add <pkg>` targets main `dependencies`; use `uv add --group test <pkg>`. Prerelease pins need
  `--prerelease allow`; trimesh is pinned `>=5.0.0rc1` because the crack-free
  `trimesh.remesh.subdivide_to_size` (the reference for `ordito.remesh.subdivide_to_size`) exists
  only in the 5.0.0rc prereleases (stable 4.x is the T-junction "soup" variant).
- **basedpyright's `include` covers `ordito`, `tests` and `benchmarks`, so a dev-only env is
  unusable** (hundreds of `reportMissingImports`). Run `uv sync --all-groups` first, or
  `uv run basedpyright ordito` (an explicit path overrides `include`; this is CI's fast
  `typecheck` job). The wide gate runs in `pytest-cpu`, the only job with all nine reference
  libraries.
- `tests/` and `benchmarks/` are checked **at the same bar as `ordito/`**, conceding only
  `reportMissingTypeStubs` (no reference library ships stubs) through a per-directory
  `executionEnvironments` block. **An execution environment's `root` re-bases import resolution**,
  so each entry needs `extraPaths = ["."]`. Test-side typing gaps have one home each:
    - `tests.conversions.warp_empty(shape, dtype, device)`: Warp leaves `wp.empty`'s `dtype`
      unannotated, so the checker types it `type[float]` from the default.
    - `tests/typings/pymeshlab/__init__.pyi` (the top-level `stubPath`, so it serves `benchmarks/`
      too): pymeshlab's names come from a compiled `import *` and its filters are bound at runtime,
      so the stub declares the four classes with dynamic members.
    - A matrix a test builds with `warp.sparse` directly is `wps.BsrMatrix`, not `odt.BsrMatrix`:
      narrow with `assert odt.has_blocks(matrix, dtype)` or `isinstance(matrix, odt.BsrMatrix)`.
    - A private helper exercised on purpose takes a scoped `# pyright: ignore[reportPrivateUsage]`;
      a file whose purpose is testing a module's internals carries one file-level
      `# pyright: reportPrivateUsage=false` with its reason.
  A fixture parameter is annotated with its fixture's return type.
- Keep `# type: ignore` directives to the few verified load-bearing by
  `reportUnnecessaryTypeIgnoreComment`. **Set the rule set first, then delete what the checker
  flags** — the unnecessary count is a function of the rules.
- **`plans/` is gitignored** — plan documents are local working notes and never appear in a commit.

### Version control

**Commit directly on `main` unless the user asks for a branch** (single-developer repository; no
review queue). Branch only when the user names one or a change is speculative and expected to be
thrown away. Committing is still an explicit request: finish the work, run the gates (§8, §12.10),
commit when asked. An A/B against a prior revision uses a **detached worktree**, never `git stash`
and never a branch checkout in the live tree (§15.6).

### Coverage — `pytest --cov`, and `kernels/` is omitted on purpose

`pytest-cov` is configured in `pyproject.toml`'s `[tool.coverage.run]`. CI's CPU job measures it,
publishes the figure to the README badge, *then* enforces the floor (gating first would abort a
regressing `main` build before the badge updated).

- **`ordito/kernels/` is omitted**: a `@wp.kernel` / `@wp.func` body is never called as Python, so
  coverage.py reports a kernel that runs on every test as unexecuted (§12.6). Never "fix" a low
  kernel-module figure by writing tests at it.
- **The badge is the CPU wrapper layer.** No GPU on a runner, so every `device.is_cuda` branch and
  everything behind `_device.prefers_tiled_reduction` is unreachable there. Do not raise the floor
  to chase those lines; they are §7.2's two-process `tests.devices` job.
- **There is no `exclude_also`**: `@overload` stubs write `...` on the `def` line (which executes at
  import) and coverage.py already excludes `if TYPE_CHECKING:` blocks; both patterns moved the
  statement count by exactly 0. Verify an exclusion by diffing the statement count, not by reading
  the regex.

Coverage gates the *wrapper* layer's branches (validation, `Literal` menus (§4.2), empty-input
early returns). It says nothing about whether a comparison is vacuous — that is §7.4.

---

## 9. Performance work: measure before you change

The methodology is here; the *numbers* are Part II (§13 cost model, §14 kernel-shape verdicts,
§15 measurement traps, §16 component status).

- **Something suddenly slow is a Warp rebuild until proven otherwise.** Check before profiling,
  before believing a kernel got slower, before deleting a test for being slow (§15.1).
- **A benchmark lands before the optimization does.** Never restructure for speed without a
  `benchmarks/test_<module>.py` group timing the *current* implementation. A belief about where
  the cost sits is a hypothesis until that group prints a number.
- **Attribute a cost with one measurement of the thing itself.** A projection from subtracting
  measurements of *different* things has been optimistic by 3-10x every time (§15.2). Attribute at
  the *benchmark's own* operating point and fixture (§15.3), and at more than one size: **a share
  that falls as the input grows is a decline, not a small win.**
- **Read device time before diagnosing** — most big losses are 92-99 % host-side launch and
  allocation cost (§16.1) — **but first ask whether the function graph-captures**:
  `wp.timing_begin` cannot see replayed kernels and reports a device-bound function as ~100 % host
  (§15.10). That covers ordito's own capture sites and every CG solve (`warp.optim.linear`
  captures by default). Disable capture, measure there, carry the device total back.
- **Before proposing an optimization, read what *calls* the thing**: the decline may already be
  written at the call site, in a constant's comment or in the benchmark's docstring (§15.5). Grep
  them for a number first.
- **Attribute a change only with a back-to-back A/B in one session** (saved baselines drift ±10 %,
  ±30 % under 100 µs). Use a **detached git worktree, never `git stash`** (§15.6). **Interleave A
  and B in one loop and read the `min` alongside the median** (§15.7).
- **A probe that instruments the thing it measures must carry a do-nothing control arm** (same
  wrapper, no substitution), and instrument both arms identically or not at all (§16.0).
- **Verify values, not just timing**: the wrong implementation is often the faster one because it
  reads less (§3.4). Every perf change keeps its parity / regression test green, so a function
  about to be optimized needs one first.
- **CUDA is the target; decide on the CUDA number.** A CPU regression is an acceptable price for a
  GPU win (Warp's CPU reductions run ~1 lane per block while NumPy is vectorized C, so a readback
  plus NumPy would veto nearly every reduction). Still measure both (the CPU path must stay
  correct, and the ratio belongs in the comment) and decline a change that wins nowhere: "no gain
  on CUDA" is the reason, not "slower on CPU" (§14.5). The exception is an algorithm that does
  asymptotically more work to expose parallelism (§14.7). **Sweep both devices before writing a
  tuning number**; if the optima differ by more than a tolerance, split the constant (§13.3).
  **Re-probe a tuning constant after a Warp upgrade** (a block-size reading of "flat between 32
  and 128" on Warp 1.16 did not hold on Warp 1.17).
- **A decline is a result: write it at the site, with the number, and resolve every site the
  finding named.** A finding naming five sites is closed when all five are converted *or*
  annotated (model: `kernels/neighbors.py`'s `wp.length_sq` decline).
- **A written decline can expire because a neighbour got faster** (a level-synchronous RDP took
  `polyline_simplify` from 84 ms to 0.68 ms and inverted a module's ordering). Re-read a module's
  declines after a big win in it.
- **A mechanism validated on one member of a fixture pair built to isolate a variable must be
  measured on the other before it ships** (`saddle`/`saddle_graded`, `sphere`/`tangle`,
  closed/open, uniform/graded). A block CG won on two well-conditioned saddles and was a 0.386x
  loss on the graded one (§16.16). §7.4's vacuity rule applied to a benchmark.
- **A decline written in one place is not applied everywhere, and the source is not the census.**
  Take censuses of properties Warp computes from the runtime: an AST scan found 39 of 71 generic
  kernels, `wp.get_module(name).kernels` found all 71 (§16.0).
- **Count the launches a numerical-method change adds per iteration before proposing it.** A cycle
  here is launch-bound; a smoother that buys iterations with launches loses (§14.8).

---

## 10. Warp API reference mirrors

Authoritative Warp function lists are mirrored under `reference/warp_api/`:

| File | Scope |
|------|-------|
| `reference/warp_api/builtins.md` | Built-ins usable inside `@wp.kernel` / `@wp.func` (`wp.<name>`) |
| `reference/warp_api/warp.md` | `warp` module API at Python scope |
| `reference/warp_api/sparse.md` | `warp.sparse` BSR/CSR matrix API |
| `reference/warp_api/utils.md` | `warp.utils` Python-scope utilities |
| `reference/warp_api/fem_linalg.md` | `warp.fem.linalg` linear-algebra utilities |

BEFORE using an unfamiliar Warp builtin, sparse or utils function, `grep` these files to confirm
name, signature and scope. Each file stamps the Warp version and source URL it was transcribed
from (fetch the URL when the one-line description is insufficient). `uv run
reference/warp_api/warp_version.py` compares every stamp against the installed `warp-lang`;
`reference/warp_api/REGENERATE.md` records how to re-extract them.

**A name in `dir(wp)` missing from the mirrors is usually hidden on purpose.** `dir(wp)` exposes
~510 names against ~137 the package uses; browsing the rest for adoption candidates is how the
`wp.dense_chol` / `dense_subs` / `dense_solve` family gets proposed. Introspect first:

```python
from warp._src.context import builtin_functions

f = builtin_functions["dense_chol"]  # a Function, the same handle §2.7's factories capture
print(f.hidden, f.doc, f.input_types)  # True  'WIP'  {n: int32, A: array(ndim=1, float32), ...}
```

**Then check the *quantity*, not the name**: a matching builtin is sometimes a regression.
Check the storage class and precision (a `float32`-only builtin cannot serve the `float64` half of
a dispatch). Where a builtin fits, the argument is often single-source-of-truth rather than speed
(`wp.volume_index_to_world` is perf-neutral, 1.08x at 200k voxels and 1.005x at 2M; adopt it for
the convention, and **name the split in the module docstring** if only some sites can convert).
Measured adoptions and rejections: §12.8. **When proposing a geometry builtin, sweep the *scale*
axis and degenerate cases**, not only unit-scale random inputs (§12.8's defects all hide there).

---

## 11. Running long commands: never poll with `until`

Benchmark suites and measurement probes run for minutes. **Do not write a wait loop around them.** A
backgrounded command re-invokes you when it exits, with its exit code and output file path. Launch
it, do something else, read the output file on notification. A foreground command that outruns its
timeout is also moved to the background and notified the same way.

- **`until ! pgrep -f <name>; do sleep 5; done` never terminates**: `pgrep -f` matches every
  process's full command line, including the polling shell's own, so the condition is permanently
  true.
- **A watcher is never the record.** The command's own stdout file is. Do not launch a second
  command to learn whether the first finished.
- **When a poll is genuinely unavoidable** (external state the harness cannot see), test a
  **sentinel file** the process writes on exit and bound the iteration count. The bracket trick
  (`pgrep -f '[p]robe'`) is not sufficient: the harness runs commands as `zsh -c '… eval '\''<your
  command>'\''…'`, so the pattern appears literally in the wrapper's command line and `pgrep -f`
  matches that shell (`pgrep -af '[p]ytest'` returned exactly its own wrapper). Read a hit's command
  line before believing the process is real. The same self-match kills the shell on `pkill -f`.
- **Do not chain short sleeps** to approximate a long wait. Pass a longer `timeout`, or background.
- **Do not run a timing probe while `pytest` or `zensical` is running** (one reconstruction read
  527-752 ms under contention, 295-400 ms quiet).

---
---

# PART II — MEASURED FACTS

Everything below was measured on this box (**RTX 5090**, 170 SMs) against the Warp version stamped
at each item. Read the relevant section before proposing an optimization, diagnosing a slowdown, or
re-deriving a number.

---

## 12. The Warp platform: bugs, quirks and version status

### 12.1 Memory safety — every failure here is silent

- **A `wp.launch` with `device=` omitted corrupts the host heap.** It resolves to the default
  device (`cuda:0` whenever CUDA is present) while the arrays sit on the CPU; HMM lets the GPU read
  host memory, so results are numerically correct, but the launch is asynchronous and the host
  arrays are freed while the kernel still reads them. A CUDA array frees with stream-ordered
  `cudaFreeAsync`, a CPU array's storage frees immediately and unordered, so the CPU side is what
  corrupts. A `wp.synchronize()` after the launch is the fix; always forward `device=` (§2.1).
- **An out-of-bounds kernel write on the CPU device is glibc heap corruption** (a CPU Warp array is
  host heap). Range-check at the write, not in a downstream call
  (`connected_component_parity_from_edges` still does not).
- **Neither of the above is "the Warp CPU backend corrupts the heap"**; both were application bugs.
  Bisection: `wp.config.mode = "debug"` compiles kernel-side bounds checks, but **a clean debug run
  is evidence about timing, not correctness**; a **component-swap bisection** (substitute one
  library component at a time into a clean pipeline) finds launch-ordering bugs line-level bisection
  cannot.
    - **A debug build is `--device-debug` (-O0) and changes two things a release suite relies on.**
      Registers: the one-block ear-clip, RDP and CG kernels go from 39-56 per thread to 163-255,
      so a 512- or 1 024-lane block fails with CUDA error 701 (too many resources);
      `_launch.launch_tiled` caps `block_dim` at `DEBUG_MAX_BLOCK_DIM = 256` in debug mode (255
      registers x 256 lanes always fits; those kernels stride by `wp.block_dim()`). FMA: -G does
      not contract despite `--fmad=true`, so an exact-float comparison against an optimized
      reference moves by an ulp (`test_marching_cubes_matches_pytorch3d`, 9.5e-7 at lattice
      coordinates 8-16, reproduced in release with `fuse_fp=False` on `warp._src.marching_cubes`).
      Read that failure as expected under debug, not as a defect.
- **A `wp.Mesh` with zero triangles silently corrupts CUDA allocator state.** The constructor
  succeeds; the *next* unrelated CUDA allocation fails with a spurious OOM and cascades into `CUDA
  error 700`. Only `indices.shape[0] == 0` matters. Safe on `cpu`. Never construct one on CUDA,
  **including in tests** (use a single-triangle mesh to reach an `n_faces < 2` guard).
  `ordito.mesh.Trimesh.warp_mesh` would hit this for a zero-face mesh — a known latent issue.
- **Python-scope gather silently ignores a non-contiguous index view's stride** (rule and
  measurement: §3.4); a `wp.map` over the same view inherits the corruption while reading faster.
- **`wp.copy(dst, src, count=0)` copies the *whole* source**, and `wp.utils.array_cast` inherits it
  (`count == 0` means "not passed"). Check a data-derived `count` for zero at the call site and
  answer the empty case without calling `wp.copy`. Same shape as §3.3's zero-length-slice raise and
  §3.7's capacity-versus-count rule.
- **`wp.copy` into a *pinned* host buffer is an async memcpy with no event**, so reading right after
  races and returns the *previous* value (CUDA blocks the host only on a *pageable* destination).
  `ordito._device.read_scalar` is the safe, device-split spelling. **It caches one scratch buffer
  per dtype, safe only for scalars**: for a vector/matrix dtype `.numpy()[0]` is a view onto the
  shared scratch, so sequential reads alias; copy before returning.
- **Warp's CPU work runs ~36x slower in a process where CUDA has been initialised** (CUDA's mere
  presence, not the launch-access guard or GPU contention); `CUDA_VISIBLE_DEVICES` must be set
  before the process starts. Hence both-device coverage is a two-process job (§7.2).

### 12.2 `wp.launch_tiled` runs one lane per block on the CPU

Still true by default on **Warp 1.18**: `wp.launch_tiled(kernel, dim=[...], block_dim=64)`
executes **one thread per block** on the CPU backend; `wp.tid()`'s lane index is always 0. Warp 1.18
adds the experimental opt-in `wp.config.enable_cpu_blocks = True` (NVIDIA/warp#1638), under which
the CPU runs every lane and `wp.block_dim()` reports the launch's value; it is off by default and
documented as substantially slower, so the `_sliced` siblings stay. Priced on a 256-block
lane-strided sum: ~18 µs per block at 64 lanes and ~110 µs at 256 (65 k elements: 0.09 ms default,
4.7 ms at 64 lanes, 29 ms at 256), a 1.5-340x loss, so **not a production path**. Its use is the
test oracle: `pytest --cpu-blocks` (and `python -m tests.devices --cpu-blocks`, a third pass) sets
it, `_device.prefers_tiled_reduction` then answers `True` on the CPU too, and every `launch_tiled`
kernel and tiled reduction runs its CUDA code path on the deterministic CPU device. That is the
one run that executes §2.2's partition hazard with more than one lane on a device whose float
atomics serialize.

- **The obvious probe says "fixed", and that is the trap.** `wp.tile_load` reads a whole tile out
  of an array, is lane-independent and was never affected; only `wp.tile(x)` built from *per-lane*
  values collapses. **Probe the lane-constructed tile**, or you will delete a correctness branch.
- Silent consequences: a block reduction over a lane-constructed tile returns only the leader's
  contribution, and a kernel relying on lanes covering a tile (`idx = tile_i * TILE_1D + t`) touches
  every 64th element. `tests/conftest.py`'s `device` fixture returns `cuda:0` when CUDA is
  available, so this is invisible to a default run; it needs an explicit `"cpu"` parametrize (§7.2).
- **The fix is to stride by `wp.block_dim()`** (the launch's `block_dim` on CUDA, **1** on CPU,
  free on CUDA): the single CPU lane covers every element and the tile reduction degenerates to a
  one-element tile that returns that lane's answer. **It does not generalize to *partitioned*
  kernels** (`f = i * TILE_1D + t` at `dim = n_blocks` drops most of the range on CPU); those keep a
  lane-free `_sliced` sibling behind `_device.prefers_tiled_reduction`.
- **The correctness boundary is narrow**: only a partition stride that is not `wp.block_dim()` is
  wrong, on *either* device (§2.2). A one-element CPU tile is harmless, and tile reductions work
  inside a `@wp.func`. A blanket "no tile reductions on CPU" comment is itself a defect.
- Upstream: **NVIDIA/warp#1480** (CPU/GPU tile parity), **NVIDIA/warp#1638** (efficient CPU block
  execution), cited in `_device.prefers_tiled_reduction`'s docstring. The branch is removable only
  when CPU blocks run more than one logical thread.
- **Warp exposes no grid-wide barrier** (no cooperative groups, no `__threadfence`); a spin-wait
  over `atomic_*` needs all blocks co-resident and a fence Warp cannot spell. That is the ceiling
  on every level-synchronous rewrite (§14.9).
- **`wp.HashGrid` has no per-cell entry point** (only `hash_grid_query` / `hash_grid_query_next`, a
  sequential per-thread iterator, and `hash_grid_point_id`), so the cell *walk* cannot be split
  across lanes. The walk is 70-73 % of per-candidate cost, capping a cooperative search that keeps
  the grid at ~1.37x. **Never propose a warp-per-edge search that keeps the hash grid.**
- **`wp.tile_bvh_query_aabb` returns out-of-range primitive indices once a traversal round overruns
  its result buffer**, silently. A round appends each hit with an unconditional atomic increment
  but guards only the *write* against a fixed `result_buffer_capacity` (`WP_TILE_BLOCK_DIM * 5`,
  160 at `block_dim=32`); past it a lane reads uninitialised shared memory as a primitive index,
  and the documented `>= 0` test passes about half the garbage. A query box grown by a distance
  bound on a large mesh triggers it (confirmed with `compute-sanitizer --tool memcheck` on
  `proximity.mesh_to_mesh_distance`'s straggler pass).
    - **Bound-check the index at every `tile_bvh_query_next` site**, `candidate >= 0 and candidate
      < n` (output-neutral: an out-of-range index is never a BVH primitive).
    - The guard stops the corruption but **cannot restore the dropped primitives**; a guarded walk
      may return an incomplete candidate set (upstream's half). Both ordito callers survive
      because each has a second sound bound (the global running minimum; the pivot's acceptance
      test).
    - **Unchanged on Warp 1.18**: `native/tile_bvh.h`'s CUDA traversal is byte-identical to 1.17's
      (the diff touches only the CPU path), and its node stack drops pushes past
      `64 * BVH_QUERY_STACK_SIZE` the same silent way. A leaf-1 BVH cannot reach the overrun (a
      round appends at most one primitive per lane, 32 under the 160 capacity), which is why a
      uniform-cloud probe returns exact sets on both versions; the trigger needs multi-primitive
      leaves, as `wp.mesh_get_bvh`'s.
- **Warp exposes no node-by-node BVH traversal** (only `bvh_query_aabb` / `_ray` / `_sphere` /
  `bvh_get_group_root`), so a BVH-pair wavefront means writing our own hierarchy.
- **A `wp.capture_while` body issuing several launches does not replay as one unit on both
  devices.** Unrolling two Bellman-Ford passes (ping-ponging two buffers) into the body of
  `graph.shortest_path_envelope` was 1.26-1.45x on CUDA but CPU relaxed markedly fewer nodes per
  round (it records through `ScopedCapture`'s APIC recording; `wp.is_conditional_graph_supported()`
  is a *machine* query and returns `True` even for a CPU-device array). Reverted. **A Python-level
  ping-pong cannot help a captured loop** (the body is recorded once; rebinding names takes effect
  at record time): a double-buffered iteration inside `capture_while` keeps its copy.

### 12.3 `wp.ref[T]` requires concrete types

Last probed on Warp 1.16.

- **Generic type-vars do not instantiate inside `wp.ref[...]`**: `wp.ref[wp.Scalar]`,
  `wp.ref[wp.Int]`, `wp.ref[typing.Any]` fail ("Couldn't find function overload") even for float32.
  `wp.Any` does not exist. Concrete `wp.ref[wp.float32]` / `wp.ref[wp.int32]` work.
- **No `@wp.func` name-overloading**: a second `def foo` shadows at Python scope and `wp.overload()`
  is kernels-only, so one helper name cannot serve both float32 and float64.
- **Tuple-assignment swap** (`a, b = b, a`) works only for simple local variables (like `sort3`);
  `arr[i], arr[j] = ...` raises *"Multiple return functions can only assign to simple variables"*.

Consequence: the shared argmin/argmax/swap helpers in `kernels/array.py` are concrete `float32`
(`update_argmin`, `update_argmax`, `update_argmax_lowest_index`, `update_argmax_vec3`,
`update_argmin_pair`); float64 sites and int32 array-element swaps keep hand-written loops.

### 12.4 Numerics and precision

- **`wp.Float` covers `float16` / `float32` / `float64` from one generic definition.
  `wp.Scalar` does NOT instantiate for `wp.bool`** (`TypeError: Function <name> does not support the
  provided argument types warp._src.types.bool`; it works for `int8` / `int32` / `float32`), so a
  "works on masks and numbers" wrapper needs a dtype branch.
    - **Do not widen a mask to `int32` first; write the concrete bool kernel.** `astype(mask,
      wp.int32)` costs nine bytes of traffic per mask byte and ~2x on the whole call. There is no
      `wp.tile_load` of a `bool` array, so the kernel's lanes stride by `wp.block_dim()` (§2.2) and
      fold their registers with one `wp.tile` reduction. `counts_to_offsets(astype(mask, ...))` is
      not the same defect (`wp.utils.array_scan` cannot read `wp.bool`), and `kernels/array.
      bool_flags` writes the 0/1 bytes at half the cost of an `array_cast`.
    - The readback-versus-device-reduction crossover for a `wp.bool` mask is ~0.5 M elements
      (§13.1); the two `repair.py` sites carrying a readback decline sit under it.
- **Scalar arguments must be constructed at the input's precision**: `wp.map(f, a_float64,
  wp.float32(tol))` will not resolve the generic — use `a.dtype(tol)`. Literals *inside* a generic
  `@wp.func` need `type(x)(...)`.
- **There is no `wp.any` / `wp.all` over vector components** and no generic vector annotation
  beyond `Any`: per-component predicates stay as per-type funcs behind a dtype-keyed dispatch
  (`is_close_vec3`).
- **FMA fusion makes a degenerate triangle's area ~1e-8 on CUDA and exactly 0 on CPU.** Warp's
  `fuse_fp` is ON by default and fuses `a*b - c*d` into `fma(a,b,-(c*d))`, so `triangle_cross` of a
  face with a repeated vertex is `0` on CPU and ~`1e-8` on CUDA, flipping it to "nondegenerate"
  against the absolute `TOLERANCE_MERGE = 1e-8` at unit scale. **Decision: do NOT disable
  `fuse_fp`** (it would lose FMA across the whole triangles kernel module, and `@wp.func`s compile
  under each *calling* module's options, so every module using `triangle_cross` would need it).
  Write degeneracy tests with **scale-aware inputs** (vertices ~1e-2) and never assume CPU/GPU
  bit-agreement on zero-area faces.
- **The same contraction makes `-orient2d(p)` and `orient2d(mirror_y(p))` different predicates on
  CUDA** (a near-collinear vertex flips its convex/reflex verdict). `kernels/polyline.mirror_y`
  evaluates the mirrored loop explicitly. **Never substitute an algebraic identity inside a sign
  test that feeds a branch**: it is bit-exact on CPU and not on CUDA, so a CPU byte-identity gate
  cannot see it.
- **`wp.mesh_query_point_no_sign` + `wp.mesh_eval_position` is not an exact closest-point query**:
  it disagrees with a float64 oracle by up to ~2e-5 absolute (always the *smaller* distance) and
  is a fixed point. A bound built on it is exact only against Warp's own query
  (`proximity.closest_point_on_mesh`). **Test such a bound** by asserting exactness against the same
  Warp query and an *improvement ratio* against an independent oracle, not `independent_distance <=
  bound`.
- **`wp.length(d) < r` and `wp.length_sq(d) < r*r` are not the same predicate in float32** (10 of
  200k rows disagree at the boundary, the same 10 on both devices). Rule: §2.4.
- **`NaN` breaks a binary search, and how depends on the convention.** `searchsorted(side="right")`
  advances into the float radix sort's trailing `NaN` block and returns the array length for every
  finite value; `side="left"`'s `wp.lower_bound` is correct for every finite value and lands `NaN`
  at slot 0. Neither raises. For a sorted-unique table (`NaN` only in the last slot) use the left
  search plus `if index >= n or values[index] != value: index = n - 1` (§16.5).
- **float32 storage sets a noise floor no solver tolerance reaches.** The float32
  `tangent_space.halfedge_transport_angles` sets the noise floor of everything built on
  `laplacian.connection_laplacian`, and it overlaps real signal, so report a resolved/unresolved
  mask instead of a cutoff (`heat.transport_tangent_vectors` returns `(transported, resolved)`).
  **"Solver noise or input precision?", cheapest first:** (1) sweep the solver tolerance (flat =
  not convergence); (2) inject noise into the suspected input and check linearity; (3) redo in
  float64 but round the result back to the shipped dtype (unchanged = storage, so an internal-only
  fix cannot work).
- **Diffused heat fields scale as 1/scale² with the mesh coordinates**; an absolute tolerance on one
  is a silent wrong answer on a rescaled mesh.
- **The backward pass of an *empty* dynamic `range` is not empty (Warp 1.17).** The adjoint walks
  `iter_reverse(range(start, end, step))` (`warp/native/range.h`), `start + int((end - start - 1) /
  step) * step` with C++ truncation, which for `start >= end` and `end - start - 1 > -step` is one
  iteration at `start`, past the array. The forward is untouched; the gradient is wrong (11x on
  `metrics.chamfer_nn_term_sliced` at 1 point over 37 slices) and the CPU device corrupts the heap.
  **A differentiated thread whose strided loop can draw nothing must return before the loop**
  (both sliced chamfer kernels open with `if j >= n: return`, pinned by
  `test_sliced_chamfer_terms_grad_with_more_slices_than_points`). Only `metrics` is taped.

### 12.5 Kernel-scope cast semantics

Measured on Warp 1.16 (probe scripts must be files — Warp refuses `exec()`-defined kernels).

- **`int(x)` and `wp.int32(x)` are the same operation**: identical C++ apart from the call name;
  `wp::int(x)` compiles only because Warp writes an unconditional `#define int(x) cast_int(x)` /
  `#define float(x) cast_float(x)` into every module header. No cost either way. The one difference
  is in Warp's type system: `int`'s `value_type` is Python `int`, `float`'s is Python `float`, where
  `int32` / `float32` carry Warp types.
- **That difference is fatal exactly once**: `total / float(count)` inside a `wp.Float`-generic
  `@wp.func` fails to parse (`Input types must be the same, got ['float64', 'float32']`). Every
  `float()` in a kernel commits its function to never being generic.
- **No cast on a thread index is load-bearing**: a bare `wp.tid()` passes to a `wp.int32` parameter,
  a `wp.Scalar`-generic parameter and a kernel-scope slice.
- **`//` is CPython's `//` since Warp 1.18, `%` is not** (§1.5).
- **At *Python* scope the same spellings behave oppositely.** `wp.int32(x)` is a constructor for a
  `warp._src.types.int32` whose arithmetic routes through Warp's builtin dispatch at ~10 µs per
  operation (§13.1). `//` on it **raises `TypeError`** (`%` works since Warp 1.18); `wp.float32(x)` does **not** round
  (`scalar_base.__init__` is `self.value = x`); `int(x)` / `float(x)` are the *unwrap* (~0.08 µs),
  which is the fix. Check 26 (§4.5) scans for it.

Rules: §1.3, §1.5, §1.6.

### 12.6 Compilation, module hashing and import cost

- **A generic kernel's lazy overload instantiation rebuilds its whole module** (§2.5: 206 → 87
  module loads, suite 1 033 s → 29 s). **`wp.map` forks its generated module per call *signature***
  on axes wider than the dtype (§3.5: 182 → 143 loads, cold cache 2.1x).
- **Warp's own generic kernels fork Warp's modules the same way** (2026-10-05, Warp 1.18).
  `wp.utils.array_cast` was 17 of the 23 `Module hash changed, recompiling` lines of a CUDA suite
  run, one per new dtype pair, each landing on whichever test reached the pair first (the "slow"
  test moved between runs). Routed through `array.copyto` (§3.6): 0, and the CUDA pass 53.8 →
  49.9 s. The sixth non-`warp.sparse` one was a *test* calling `wp.map(wp.mul, ...)` over `vec3`,
  which forked the `map_mul` module the package declares at import: **a test's own `wp.map` of a
  builtin shares the library's generated module**, so tests build such inputs on the host. The last
  five were `warp.sparse`'s generic `_bsr_*` / `bsr_mv` kernels meeting a second dtype: now 0.
  **Warp's own generic kernels take `wp.overload` too**: `kernels/linalg.register_warp_overload`
  derives the concrete signature from the kernel's template (`kernel.adj.arg_types`, `Any` and
  generic arrays set to one scalar, per-argument overrides for a mixed-precision copy).
  `register_bsr_mv_overloads` registers both `bsr_mv` kernels ordito's heavy-row CG launches at
  float32 and float64 at import (the factories are `functools.cache`d, so they are called as
  `bsr_mv` calls them: `block_cols=` by keyword, the tile positionally), and `tests/conftest.py`
  registers the three the suite's own reference builds reach (`bsr_from_triplets`,
  `bsr_transposed`, `bsr_copy`). `test_heavy_row_rounds_launch_only_the_bsr_mv_overloads_
  registered_at_import` fails if a Warp release moves the signature (probed with the registration
  removed: both arms fail on CUDA, the first on CPU, where both arms use one kernel). **Census them with `wp.config.log_level = wp.LOG_DEBUG` under
  `pytest -s`**: pytest captures Warp's log otherwise and the grep finds nothing.
- **`import ordito` is expensive because `@wp.kernel` builds an `Adjoint` at import time for every
  decorated kernel**, and importing one submodule imports the parent package first. Fixed by a
  PEP 562 lazy `__init__` (§16.2).
- **A deferral one module makes can be silently cancelled by another module's top-level import —
  check `sys.modules`, not the comment.** `ordito/reconstruction.py` defers `import warp.fem`, but
  two kernel modules loaded the package eagerly for two `@wp.func`s. `warp.fem`'s public shim pulls
  in `adaptivity`, `dirichlet`, `domain`, `field.*`, `geometry.*`; the tree imports
  `warp._src.fem.linalg` directly, with a comment naming the probed Warp version (an upgrade moving
  `warp._src` fails loudly at import).
- **Only host readbacks block a CUDA graph capture** (last probed on Warp 1.16). `array_scan`,
  `bsr_from_triplets`, `radix_sort_pairs` and a nested `wp.capture_while` capture and replay.
  A readback that sizes an output (`edges_unique`, `flatnonzero`, `remove_unreferenced_vertices`
  did) raises `CUDA error 906` inside a capture; leave the count in a device `state` array.
  **`warp.Graph` retains modules but not arrays**: keep a reference list for anything allocated
  inside a capture (defensive, not proven).
- **`block_dim=32` takes the `warp_count == 1` fast path** in `tile_reduce_impl` (ballot plus warp
  shuffle, no cross-warp shared round trip) — the whole 126-vs-369 ns gap in §13.2.
- **"Clear the kernel cache before the first run" is a no-op across an upgrade**: Warp namespaces
  the cache by version (`~/.cache/warp/1.17.0`). `wp.config.kernel_cache_dir` reads `None` until
  init.
- **Cold compile is dominated by adjoints, not by inlining** (census 2026-10-05, Warp 1.18, every
  kernel module loaded into an empty cache): 25.4 s with backward passes, 10.6 s without (§2.6,
  now the tree's setting); `energies` 11.5 s and `reduce` 2.8 s were the top two, `neighbors` only
  0.4 s because its register-row bucket kernels are factory-built on first use. Warp 1.18's
  `@wp.func(inline=False)` was therefore not pursued: the large duplicated bodies are the bucket
  kernels' unrolled rows, which §2.9 requires inline, and no other module's compile is near the
  adjoint cost removed.
- **`wp.config.verbose = True` is deprecated in Warp 1.17** (stderr noise); use
  `wp.config.log_level = wp.LOG_DEBUG`.
- **coverage.py cannot see a kernel body** (the Python function is never called; the tracer
  reports kernels as unexecuted), so `ordito/kernels/` is omitted (§8). Over the whole CPU suite
  (Warp 1.17), line coverage: wrapper layer 96.02 % vs `kernels/` 24.29 % (including kernels would
  publish a meaningless 59.40 %). With branch coverage, which ships, the wrapper layer is 93.79 %,
  against CI's `coverage report --fail-under=90` (a regression alarm, not a target).
- **`wp.map` leaves unparseable filenames**: `warp._src.utils.map` `exec`s its kernel and names the
  module `f"{basename}:{lineno}"` (e.g. `ordito/points.py:153`), so coverage.py emits one
  `couldnt-parse` warning per generated module (72 over a full run, no lines). Omitting `*.py:*`
  removes them and changes no count. It reproduces only on a whole-suite run. Configured in
  `pyproject.toml`.
- **A `@wp.func` is not inlined at codegen: Warp emits a real `static CUDA_CALLABLE` function and
  nvcc inlines it.** "Costs nothing" held for every extraction measured (byte-identical or smaller
  SASS) but is a claim to verify. A tuple return emits a `wp::copy` per component, also elided.
- **Proving a `@wp.func` extraction cost-neutral is free, needs no clock, and suits a busy box
  (§15.6).** Warp caches the `.cu` and the compiled code per module under
  `~/.cache/warp/<version>/wp_<module>_<hash>/`. Recipe: load the module on `cuda:0` in each arm
  (new tree and a detached worktree), read the hash off
  `list(warp._src.context.get_module(name).hashers.values())[0].get_hash()`, and count instructions
  per `.visible .entry` in the PTX (strip the per-kernel `_<8 hex>_cuda_kernel_` infix; read only
  `_forward` entries).
    - **Since Warp 1.18 (CUDA 13.4 NVRTC) the cache holds an `.sm120.cubin`, not `.ptx`**: run
      `nvdisasm -c` on the cubin directly; the PTX steps below apply only to a PTX-emitting build.
    - **Go to SASS**: ptxas folds most of what a PTX diff reports (four kernels that moved in PTX
      were identical in SASS). Pipeline: `ptxas -arch=sm_120 -O3` plus `nvdisasm -c` from
      `/usr/local/cuda-12.8/bin` (not on `PATH`). That ptxas is one PTX version behind Warp's NVRTC
      (`Unsupported .version 8.8`); rewrite the `.version` line to `8.7`.
    - **Count `nvdisasm -c` lines with address regex `/*[0-9a-f]{4,}*/`, not `{4}`**, or the count
      saturates at 4 096 (addresses gain a fifth hex digit at `0x10000`). Two different kernels
      reporting exactly 4 096 is the tell; a truncating counter fails toward "identical".
    - **Compare `_forward` entries; `_backward` ones move** even when the extraction is free
      forward (shifts of −320 to +32). Not a cost: only `kernels/metrics.py` is differentiated
      (§2.6).
- **Warp lowers a kernel-scope `not` on a bool as a *select***, so a boolean `@wp.func` imposes a
  polarity on callers and the complement costs a select per call: on `voxels.count_box_faces`'
  unrolled six-row loop, sharing the probe as `-> wp.bool` cost 48 SASS instructions, sharing it as
  `-> wp.int32` (the grid slot) with each caller's own `< 0` / `>= 0` is free, and counting the
  complement (`6 - covered`) was worse (+88). **A shared predicate over a sentinel returns the
  sentinel, not a boolean** (`voxels.cell_slot` / `point_slot`).
- **`wp.constant(x)` is `return x` after an `is_value(x)` check on Warp 1.17**: an identity, not a
  declaration. Any module-level global evaluating to a static value resolves from kernel scope;
  `ordito/constants.py` does not call it (§1.2). It fixes no dtype; it only raises `TypeError` at
  the definition site for a non-scalar/vector/matrix value.
- **A tile `shape=` must be a plain integer**: `wp.constant(256)` works for `wp.tile_load` /
  `wp.tile_zeros`; `wp.constant(wp.int32(n))` fails at parse time (`AttributeError` naming the
  kernel). Check 17 (§1.5) recognises an integer `wp.constant` only in the typed form, so a
  constant that is also a tile shape is invisible to it. A typed constant is also unusable in *host*
  arithmetic (`(n + c - 1) // c` raises), so wrappers read `int(...)`.

### 12.7 `warp.sparse`

The `nnz`-is-a-capacity rule is §3.7. Further behaviours, all silent:

- **`bsr_mm` returns a structural *superset* of the product**: extra entries exactly zero,
  interspersed in column order rather than trailing capacity (not NVIDIA/warp#1769; still present
  on Warp 1.17). The zeros matter once the result is multiplied again: an explicit zero at `(i, c)`
  makes column `c` "see" row `i`, so a Galerkin product `PᵀAP` inherits every reachable aggregate
  (measured 47x pattern blowup, operator complexity 2.19 instead of 1.03).
- **`bsr_set_transpose`, `bsr_mm` and `bsr_axpy` read the `nnz` *field*, never `nnz_sync()`**, so a
  stale count carries garbage into the consumer.
- **`bsr_mv` takes one vector**: a cycle over several right-hand sides pays one launch per vector
  per mat-vec; a hand-written batched CSR kernel amortizes it (the multigrid V-cycle's).
- **Unwritten `(0, 0, 0.0)` triplets in a `wp.zeros` buffer all accumulate on entry `(0, 0)`, and
  `bsr_from_triplets`' duplicate accumulation costs O(duplicates on the hottest address)**, not
  O(triplets): 9-32x. A conditional triplet writer must point unwritten slots *out of range*
  (`rows.fill_(n_rows)`, silently dropped). §3.7 is right about correctness and wrong about cost;
  price a conditional emit by its collision count. Padding must be a hole, not a value (§13.2).
- **`warp.optim.linear`'s `TiledDot` silently picks an O(n) per-block reduction whenever
  `batch_offsets` is set with `batch_count > 1`** (normally an O(log n) tiled tree), and
  `use_bounded_tree` requires `batch_count == 1`, so a batched solve cannot reach it. Two dots per
  CG iteration can dominate at moderate `n`. `linalg._BatchedCg` uses a two-stage per-column tree
  (1.10-1.50x end to end); one unbatched `cg` per column duplicates every other kernel and loses.
- **`warp.optim.linear.cg` at `check_every=0` records, instantiates and launches a fresh
  conditional graph on every call** (`_run_capturable_loop` opens a `wp.ScopedCapture`), so each
  solve pays §14.3's "record, replay once" row: 2.07-2.10 ms through `wpl.cg`, 1.33-1.36 ms through
  a fresh `_BatchedCg`, 0.61-0.63 ms replaying a persistent one (heat solve, 2 562-10 242
  vertices, flat in size). No scalar or `wp.mat22d` solve in the tree reaches `wpl.cg` now; a solve
  keeps one recorded state per operator (§16.16).
- **`warp.optim.linear.cg` resolves an omitted `atol` to `atol := tol`**, turning a relative
  tolerance into an absolute floor (criterion `max(atol, tol * ‖b‖)`); once `‖b‖` falls below the
  floor the solve returns **zero iterations** and the initial guess as "converged" (`heat.log_map`
  at extreme mesh scale). Every ordito direct `wpl.cg` call passes `atol=0.0` explicitly alongside
  `tol=`; `_BatchedCg` computes its threshold with `atol_sq = 0.0` already.

### 12.8 Warp builtins: adoption verdicts

**Adopted:**

- **`wp.mesh_query_point_sign_winding_number`** → `proximity.signed_distance_on_mesh(...,
  sign_mode="winding")`: exact where ray parity misclassifies on meshes with holes, at memory and
  runtime cost (parity stays the default). Traps: `support_winding_number=True` is required or the
  builtin **silently returns the ray-parity answer**. Through Warp 1.17 `wp.Mesh` did not retain
  the flag, so only a function that built its own mesh could guarantee it; Warp 1.18 exposes
  `wp.Mesh.support_winding_number` (GH-1824), and `signed_distance_on_mesh` / `signed_distance_grid`
  accept a supplied mesh exactly when it reads `True` (a plain Python attribute set by the
  constructor, so a caller who assigns it afterwards defeats the check). Warp's public API exposes
  only the thresholded *sign*; the value (`wp::mesh_query_winding_number`, an order-2 Barnes-Hut
  walk) is reachable only through `wp.func_native`. `proximity.winding_number` needs neither: it is
  exact through its own patch hierarchy (§16.6).
- **`wp.bvh_query_sphere`** (Warp 1.17), in `kernels/neighbors.py` and
  `proximity.py::closest_point_on_edges`: a win where the enumeration radius is large relative to
  an existing BVH. **It is bit-exactly `wp.length_sq(d) <= r*r`**, so adopting it means the squared
  predicate everywhere, including the hash-grid branch. **Do not convert a probe whose radius
  equals the hash-grid cell width** (the grid's best case: a small well-centred query reaches its
  cell by address arithmetic, a BVH pays a root-to-leaf descent); the same conversion in
  `ball_pivoting.ball_is_empty` was a loss.
- **Warp 1.18 made `wp.bvh_query_sphere` 1.7-2.2x slower on surface point clouds** (packed-leaf
  rewrite, GH-1840/1843): `dragon` vertices, self queries, r = 2 mean edges, 0.551 -> 1.197 ms with
  identical counts; 1.35x slower on a uniform volumetric cloud. The box walk is unchanged on surface
  clouds (3.2x faster on the uniform one). ordito's BVH ball queries lost a median 0.70x
  (`query_ball_count`) / 0.81x (`_with_offsets`); the hash grid did not move, so the `hashgrid`
  default stands with a wider margin (it wins 49 of 54 public ball cells on 1.18 against 27 on
  1.17, where the BVH won 20). Independently of the version, the BVH leaf size wants opposite things per query (sweep over 8 k
  to 14 M points, public calls, build included): **ball queries are fastest at leaf 1 at every
  size** (1.07-1.35x over leaf 4, growing with the radius; every larger leaf is slower), while
  **k-NN's optimum grows with size and `k`** (8-16 at 8 k points, 16-32 at 0.4-1 M, 32-64 from
  4 M; up to 1.59x over leaf 4). A gate of 16 below ~100 k points, 32 to ~2 M and 64 above stays
  within 0.97x of best everywhere measured, where a fixed 16 or 32 drops to 0.85x. The leaf size
  does not matter to the default hash-grid k-NN (within 1.03x). **Adopted** (2026-10-05): the ball
  queries default to `leaf_size=1`, `query_nearest` / `query_weighted_nearest` to `leaf_size=None`,
  that size gate (`neighbors._NEAREST_LEAF_SIZES`; the weighted query stays within 0.92x of its best
  leaf under it, where 4 fell to 0.73x at `lucy`); `bvh_from_points` keeps 4 for a tree shared by
  both families. Harness against the leaf-4 defaults: `query_ball_bvh` 1.25-1.29x,
  `query_nearest_bvh_k64` 1.23-1.34x, `_k7` 1.00-1.14x, `query_weighted_nearest` 0.97-1.16x;
  distances identical on both devices (tied `k`-th indices may differ on CUDA, as they may anyway).
  Re-run the backend sweep if upstream fixes the sphere walk: on 1.17 it beat the grid off-surface.
- **The sphere walk still beats the box walk at every ball site on 1.18**, despite the regression.
  Every `bvh_query_sphere` enumeration (ball count / collect, both k-NN row kernels, the weighted
  query, `ball_mean_curvature`, `closest_point_on_edges`) swapped for the circumscribed cube plus
  the exact `in_ball` test, harness A/B at the benchmark points: ball queries 1.00-1.02x at 2 mean
  edges and 0.88-0.91x at 4, k-NN 0.69-1.00x, weighted 0.75-0.92x, `discrete_mean_curvature`
  0.70-0.82x, `closest_point_on_edges` 0.76x at `bunny_decimated` but 1.04-1.09x at `happy_buddha`
  / `lucy` (too mixed for a size gate). The raw-kernel coin toss (box ahead at 1-2 mean edges) does
  not survive into the public calls. Numbers at `kernels/neighbors.py`'s ball-broad-phase comment,
  `kernels/curvature.ball_mean_curvature` and `kernels/proximity.py`'s edge search.
- **`wp.bvh_query_sphere` as a broad phase over *bounds***: `curvature.discrete_mean_curvature` won
  2-4x over a cube `wp.bvh_query_aabb`. **`wp.bvh_query_aabb`'s traversal costs several times
  `wp.bvh_query_sphere`'s per candidate on the same BVH, even at equal candidate count** (measured
  on Warp 1.17; on 1.18 the per-candidate gap narrowed, and the public calls still favour the
  sphere, above), so a cube broad phase is the wrong query when the real predicate is a ball. **`ball_pivoting`'s pivot
  search is blocked on Warp**: its tiled box walk is the win (§14.2) and Warp 1.17 has no
  `tile_bvh_query_sphere`. The exact in-loop substitute (reject `|c - mp| > 2r` before the
  prefilter: a candidate on a radius-`r` ball whose chord holds the edge midpoint is within `2r`
  of it) measured 1.00x on `bunny_decimated` / `bunny` with identical faces, so the cost is the
  box traversal, not the candidates it returns; reverted.
- **`wp.mesh_get_bvh`** (Warp 1.17): `proximity.mesh_to_mesh_distance` builds one structure over
  mesh B instead of two.
- **`wp.volume_index_to_world`**: adopted for the convention (perf-neutral).
- **`warp.geometry.delaunay_edge_flip`** (Warp 1.18) as the *start* of
  `reconstruction.delaunay_triangulation`'s flip phase, from `_NATIVE_DELAUNAY_FLIP_FROM = 2**19`
  points on CUDA, with ordito's float64 flip loop finishing from its output. Alone it is not exact
  (at 1 M random points it leaves two edges that fail the in-circle test under exact rational
  arithmetic, where the loop leaves none) but about twice as fast as the loop from 0.5 M points
  (31 vs 64 ms at 1 M, 84 vs 163 at 2 M); the loop then repairs what it left, so the faces are the
  loop's own (identical at 0.1-2 M, pinned by `test_delaunay_native_flip_start_changes_nothing`).
  Whole call 0.90x / 1.04x / 1.08x / 1.10x at 0.1 / 0.5 / 1 / 2 M points: the single-threaded seed
  (283 of 347 ms at 1 M) is most of it. Also a Class A test oracle from the same seed at 200 and
  20 000 points (`test_delaunay_matches_warp_edge_flip_from_the_same_seed`).
- **`warp.geometry.sparse_marching_cubes`** (Warp 1.18) behind
  `levelset.signed_distance_level_set`, the one extraction `offset_mesh`, `resample_uniform` and
  `fix_self_intersections(method="voxel")` share: a Lipschitz octree brackets the level set and
  only the kept cells' corners are queried. Gated on lattice
  size (`levelset._SPARSE_LEVEL_SET_FROM_NODES = 2**21`: 0.84-0.93x on `dragon` at 0.6-0.7 M nodes,
  1.95x at 2.4 M, 4.5x at 9 M; 5.7-11.8x on a 20 k-face sphere at 10-37 M) **and on a closed,
  consistently wound input**. On an open or non-orientable one the winding-signed distance jumps
  away from the surface, across the region a hole spans, so it is not 1-Lipschitz and the octree
  drops cells the dense lattice keeps (a fifth of a hemisphere's offset faces). The closedness test
  is two key sorts, winding first (0.98x at worst, `lucy`). Equal to the dense surface up to buffer
  order except at a node whose distance rounds onto the level: the two extractions place a node one
  rounding apart, so a lattice-aligned input (an axis-aligned box) triangulates such ties
  differently; `test_offset_mesh_sparse_extraction_matches_the_dense_lattice` rotates `cave_cube`
  for that reason. Its octree reads back one count per level. Sparse against dense, forced by the
  gate constant: `resample_uniform` on `happy_buddha` 0.28x / 0.42x / 1.04x / 2.86x at 0.03 /
  0.17 / 1.1 / 7.9 M nodes, on a 20 k-face sphere 2.75x at 1.9 M and 7.0x at 13.5 M;
  `fix_self_intersections(method="voxel")` on the tangled tori 0.41-0.54x at `diag / 128`,
  1.28-1.46x at 256, 4.85-4.95x at 512, identical counts (a closed self-intersecting input is still
  1-Lipschitz: its winding sign changes only on the surface). The defaults of all three stay under
  the gate on the benchmark meshes (flat in the harness, 0.98-1.04x), so the gain is at finer cells.
  `kernels/reconstruction.lattice_points` went with the move (`grid_points` samples the lattice).

**Rejected on measured evidence — do not re-propose without new data:**

- **`wp.intersect_tri_tri` for `intersection.triangles_intersect_sat`**: its epsilon is absolute on
  unnormalized plane distances, so the verdict moves with mesh scale (bad at very small and very
  large scale, both precisions) where the SAT is scale-invariant; `mesh_with_mesh` is public on
  arbitrary meshes.
- **`wp.closest_point_edge_edge` for `remesh._segments_dist_sq_d`**: float32-only and measurably
  worse relative error on near-parallel segments, which the existing float64 port exists for.
- **`wp.sample_unit_hemisphere_surface`** would replace `visibility`'s low-discrepancy Fibonacci
  lattice with Monte Carlo at the same ray count (variance where there was none).
- **`wp.norm_huber`** is the Huber *norm*; `registration.robust_weight` needs the IRLS *weight*
  `ρ'(r)/r`.
- **`wp.tile_arange`** cannot express `array.arange` (bounds read at codegen, so a runtime start is
  a parse error) and a range fill has no reuse for a tile to exploit (§13.1).
- **`wp.volume_voxel_count`** is a capacity (§3.7).
- **`dense_chol` / `dense_subs` / `dense_solve`** are `hidden: True` / `doc: "WIP"` and take
  `wp.array[float32]` where the caller holds a `wp.spatial_matrix` in registers: a 2x loss (§2.9).
- **`warp.geometry.tri_tri_adjacency`** (Warp 1.18) for `halfedge.halfedge_twins`: a different
  contract (it pairs an edge whose two faces are wound against each other, which `halfedge_twins`
  rejects; on non-manifold input the two disagree), so a test oracle on edge-manifold, consistently
  wound fixtures (`test_halfedge_twins_matches_warp_tri_tri_adjacency`, Class B). It is faster at
  scale (0.426 vs 0.493 ms at `dragon`, 9.7 vs 17.9 ms at `lucy`, slower below 0.1 M faces), but it
  cannot replace the default path: every in-repo caller runs `halfedge_twins` with `validate=True`,
  whose rejections ride the pairing sort, and Warp's output cannot carry them (on an edge with three
  faces a third halfedge reads as a boundary, so the defect is invisible without counting the
  run, which is the sort). A lead for §16.11's sort, not a substitute.
  **Also refuted for the `validate=False` path alone** (2026-10-05, built and reverted): Warp's
  pairing is one thread per vertex scanning that vertex's bucket (every halfedge whose *lower*
  endpoint it is) pairwise, so it is quadratic in the largest bucket: a cone fan with its hub at a
  low index cost 38 / 147 / 600 / 3 000 ms at 512 / 1 024 / 2 048 / 4 096 spokes against the
  sort's 0.4-0.9 ms. Scan meshes are safe (largest bucket 22 on `dragon`, 24 on `lucy`), where it
  was 0.44-0.58x up to 0.33 M faces and 1.53x / 1.75x / 2.2x at `dragon` / `happy_buddha` / `lucy`.
  A bucket-size guard needs a readback. **What was adopted instead is its counting sort with a
  different owner and matcher** (§16.11): ordito's own buckets beat both, on validated calls too.
- **`warp.geometry.swept_volume_mesh`** (Warp 1.18): no ordito counterpart and no caller (§4.2).

### 12.9 `wp.Volume` as a voxel-set container

`ordito/voxels.py` carries the detail. On Warp 1.16+, `allocate_by_voxels` and `fem.Nanogrid` work
on **CPU** with identical counts and byte-identical `get_voxels()` row order across devices, so a
volume-backed module need not be CUDA-only. An empty point set raises `RuntimeError` (still guard).

- **`Volume.allocate_by_voxels(world_points, voxel_size, translation)` deduplicates** and beats a
  hand-rolled `unique_rows` dedup by ~2.3x on both devices, **but loses to a one-`atomic_cas` cell
  table** (1.3-3.2x on `voxel_down_sample`, byte-identical, §16.13), which reproduces
  `get_voxels()`' leaf-major order with a 36-bit in-tile sort plus a root-tile pass.
- **`volume_lookup_index(grid, i, j, k)` is the `k`-th row of `get_voxels()`**, `-1` when absent:
  grid and cell array share one numbering, so a per-voxel payload is a `wp.array(n_voxels)`; O(1)
  membership.
- **NanoVDB centres voxels on integers**: to make its index equal a corner-aligned cell index
  (`floor((p-origin)/s)`) pass `translation = origin + 0.5*voxel_size`, or every cell is +1 on every
  axis, silently.
- **`get_voxels()` order is leaf-major**, deterministic but not lexicographic: lexsort before
  comparing to a reference.
- **`voxel_points` may be integer** (contiguous `(n,3)` int32 or `vec3i`, read as index-space
  cells).
- **`point_mask` (int32, 0 = ignore) filters during the build**, so no compaction pass is needed;
  a fully-masked one-point build is the only legal EMPTY volume (zero-length input raises).
- **Rebuildable volumes do not pay**: `rebuild()` into a reserved topology is only marginally
  faster than a fresh `allocate_by_voxels`. Traps if used: `get_voxel_count()` returns the reserved
  *capacity* (use `get_active_stats().voxel_count`, §3.7); the four `max_*` capacities **cascade**
  (passing only `max_active_voxels` under-reserves the others, runs out of device memory and leaves
  the CUDA context throwing illegal-memory-access): pass all four.
- **`warp.fem.Nanogrid(volume)` derives topology ordito would hand-write** (verified exact):
  `.vertex_grid` is the deduplicated corner lattice; `boundary_side_index()` + `side_position` +
  `side_normal` enumerate outward faces; `side_inner_cell_index` over boundary sides is the
  6-connected surface-voxel set. Import `warp.fem` **inside the function** (§12.6).

### 12.10 Upgrade discipline and the workaround table

Status of every version-stamped workaround (last full re-probe on Warp 1.16, Warp 1.17 and
Warp 1.18 deltas noted):

| Workaround | Status |
|---|---|
| `warp.optim.linear.cg` returns NaN on CPU | **FIXED in 1.16** (cpu and cuda:0 agree to 1.7e-10); `_device.require_cuda` and its call sites deleted |
| CPU backend corrupts the process heap | **FIXED / never was Warp** (§12.1) |
| `bsr_mm` nondeterministic on a chained triple product | **REFUTED — never a Warp bug** (§3.7) |
| `bsr_compress` illegal memory access (#1769) | **FIXED in 1.17** (§12.7) |
| `bsr_mm` structural superset | **still present on 1.17**, not #1769 (§12.7) |
| `wp.launch_tiled` one lane per block on CPU | **still the default on 1.18**; the experimental `wp.config.enable_cpu_blocks = True` (#1638) runs every lane (a lane-constructed `tile_sum` over 4 blocks of 64 correct, `block_dim()` reads 64), "substantially slower" per the changelog (§12.2) |
| fast launcher's tid-extent guard | **renamed on 1.18**: `adj.scalar_tid_extent_limit_candidate` became `tid_extent_limit_candidate` and every `wp.tid()` axis is bounded by 2³¹, not only the leading one; `_launch._extent` defers any oversized axis to `wp.launch` |
| `_launch.array_scan`'s native fast path | **must check the status on 1.18**: the CUDA `wp_array_scan_*_device` natives now return `False` on failure (GH-1894: an OOM scratch allocation used to scan a null buffer and leave the output partly unwritten) and `wp.utils.array_scan` raises; the fast path called the native directly and dropped the status, so it now raises `RuntimeError(runtime.get_error_string())` too. The radix-sort natives still return nothing |
| `.nnz` after `bsr_mm` / `bsr_axpy` / `bsr_transposed` / `bsr_copy` | unchanged on 1.18 despite "no eager block-count transfer" (GH-1792): identical values to 1.17 on both devices (a capacity, §3.7) |
| empty `wp.Mesh` corrupts the CUDA allocator | **still broken on 1.18** (NVIDIA/warp#1765 open; 10/10 subprocesses at 0 and 3 points die on the next alloc with error 700; on 1.18 the constructor itself now logs OOM / radix-sort / error-700 lines to stderr but still raises nothing and returns a valid id; a one-triangle control passes, §12.1) |
| `wp.ref[wp.Scalar]` generics | **still broken** — `WarpCodegenError` at kernel parse (§12.3) |
| radix key dtypes | unchanged: int32/int64/uint32/uint64/float32/float64, 4- or 8-byte values; `segmented_sort_pairs` takes int32/float32 keys only |
| no sparse triangular solve | unchanged |
| generic `Any`-typed `@wp.func` wrappers around tile intrinsics | a wrapper over a *tile* argument still fails (NVRTC "more than one instance of overloaded function"); use §2.7's builtin-capture factory. A wrapper over a per-lane *value* works on Warp 1.17 (`reduce.block_sum` / `block_min` / `block_max` at int32/int64/uint64/float32/float64, `vec2i`, `vec3d`, `mat33`, `spatial_matrix`, a 25-wide vector, both devices) |
| `@wp.kernel(grid_stride=False)` | benchmarked as noise (±5 %, sign flips) — not adopted |
| `tile_bvh_query_aabb` result-buffer overrun | **still present on 1.18** (`tile_bvh.h`'s CUDA traversal byte-identical to 1.17); the `candidate < n` guards stay (§12.2) |

**Three things that make an upgrade's verification honest, all of which default to a *false pass*:**

- **`compute-sanitizer` is at `/usr/local/cuda-12.8/bin/compute-sanitizer`, not on `PATH`**, and
  needs `--target-processes all` (through `uv` it hops two processes). **A run that instrumented
  nothing also prints `ERROR SUMMARY: 0 errors`; the proof is the ~10x slowdown** — time the same
  suite both ways.
- **Both `wp.capture_while` sites sit behind `wp.is_conditional_graph_supported()`**; where it
  returns `False` the CUDA-graph path is skipped and tests pass having tested only the fallback.
  Confirm it is `True` with a pytest plugin that monkeypatches `wp.capture_while` /
  `wp.capture_launch` and counts calls.
- **Re-running our own repro validates the repro, not the upstream bug** (how the `bsr_mm`
  misattribution survived two re-probes). When a workaround rests on an unconfirmed upstream bug,
  suspect the repro.

**Gates for an upgrade**, all four: the full suite, `basedpyright` 0 errors, `zensical build
--strict` clean, `tests.parity` with an unchanged pair count. The `pyproject.toml` specifier stays
`warp-lang>=1.15` where no newer-only API is used (the pin lives in `uv.lock`). **Also re-probe any
tuning constant** (§9). mkdocstrings renders ordito's source-level annotations, so Warp's 1.16
change to `repr()` of array annotations never reached the docs.

**Two probe-fixture traps that faked a "DIFFERS" for a whole family:** drawing random input
**inside** a per-device callable hands the two devices different problems (hoist every `rng` draw
to host scope); and `slice_plane` without `merge_vertices()` leaves duplicated boundary vertices,
so harmonic / tutte / arap were pinned to different boundaries (§7.3).

---
## 13. The cost model (RTX 5090)

### 13.1 Host-side, per call

**Measure `n` calls between two syncs and divide.** A per-call cost taken with a sync *inside* the
loop is up to 14x wrong: Warp leaves the CUDA mempool release threshold at 0, so every sync drains
the pool and the next allocation is cold. A drained pool's cost grows with the allocation's size:
raising the threshold is 1.00-1.02x at small sizes and 1.4-1.7x on every multi-allocation call
from `dragon` (0.87 M faces) up (`face_adjacency`, `edges_unique`, `cotmatrix`, `crease_edges`),
2.9x on the two-allocation `face_normals_and_areas` at `lucy`. ordito keeps Warp's default
(the setting is process-wide; re-decided on those numbers); the lever is allocating less (§16.9).
A scratch cache that keeps buffers reserved between calls is rejected for the same reason (§16.9).

| primitive (correct regime) | cost |
|---|---|
| `wp.launch` / `wp.launch_tiled` | **11.8 / 12.2 µs**, independent of `dim` |
| `_launch.launch`, hot path (generated per-kernel packer) | **4.5-4.9 µs** at 2-5 arguments, 6.1-6.3 at 12; the native `wp_cuda_launch_kernel` call alone is **2.1-2.3 µs** whatever its ctypes prototype |
| `wp.empty` | 6.1 µs, flat in size (`wp.empty(0)` 2.0) |
| `wp.zeros` / `wp.full` | 9.3 µs |
| `arr.fill_` / `arr.zero_` | 3.2 / 2.5 µs |
| `wp.copy` | 4.5 µs |
| `wp.clone` | 13.3 µs |
| `wp.array(numpy)` | 16.1 µs |
| a slice view | 3.0 µs |
| **`wp.utils.array_cast`** | **20.8-22.1 µs — 1.8x a plain launch** |
| `wp.utils.array_scan` | 7.1 µs |
| **`wp.utils.array_sum`** | **39.8 µs — 3.4x a plain launch** |
| `wp.utils.radix_sort_pairs` | 15.6 (int32) / 18.6 (int64) µs at `n = 1`; 64.3 / 86.1 at 61 440 |
| `radix_sort_pairs(..., end_bit=b)`, `uint64` keys, graph-replayed | **43.4 vs 89.5 µs** at 24 576 keys / 24 bits; 87.8 / 120.7 at 491 520 / 34 bits; 310 / 367 at 3 M / 42 bits |
| a cached `wp.map` call (Python overhead above the kernel) | ~11 µs (1.78-1.86x the launch it wraps) |
| a host readback | ~0.1 ms *queued*, **14.3 µs isolated** (§14.6) |
| `wp.synchronize_device` | 1.1 µs |
| a **replayed** kernel in a captured chain | **1.17 µs**, linear from 1 to 12 kernels |

The launch and allocation rows were re-measured on Warp 1.17 (20-25 % above the earlier version;
all terms moved together). The bold rows change decisions: `array_sum` is 3.4x a launch,
`array_cast` is 1.8x a launch for a copy a plain kernel does identically, and an isolated readback
is 14.3 µs, so a function that already synchronized does not pay 0.1 ms for a second read.

A host-cost model of `allocations x per-call cost + kernels x 9.7 µs` accounts for 84-122 % of a
wrapper's measured host time (seven wrappers, 0.3-2.9 ms).

**Launch marshalling is ~1.0 µs per argument**, linear, identical on both devices (14.8 µs at 2
arguments, 41.1 at 28). The "~32 µs per launch" quoted elsewhere is the mean kernel (5.1
arguments), not a constant. A `@wp.struct` bundle saves ~25 µs flat at every `dim` for a
25-argument kernel; building the bundle costs ~2.6 µs. Rule and eligibility: §2.8. Only 18 of 440
kernels take at least 12 arguments.

**A generic kernel costs a further ~12 µs of host overload resolution per launch** (roughly double
a concrete one); `wp.Float`, `wp.Scalar` and `Any` cost the same, and the cost scales with how many
parameters are generic. `wp.launch` runs `infer_argument_types` over the whole argument list before
the overload lookup. Launch through the concrete `wp.Kernel` that `wp.overload()` returns (§2.5):
typically 1.1-1.6x end to end. A missing dtype then raises instead of silently rebuilding.

**A Warp-typed value is not a Python number: its `+ - *` are ~370x a Python float's.**
`scalar_base.__add__` is `warp.add(self, y)`, Warp's Python-scope builtin dispatch
(`inspect.signature().bind()` per operand).

| operation at Python scope | Warp | plain / NumPy | ratio |
|---|---|---|---|
| `wp.int32` / `wp.float32` `+` `-` `*` | **9.9-10.1 µs** | 0.027 µs | ~370x |
| `wp.int32` `<` `==` | 0.11-0.17 µs | 0.028 µs | 6x — immaterial |
| `wp.int32(x)` construct, `int(x)` unwrap | 0.08-0.10 µs | — | free |
| `wp.int32 // %`, `wp.zeros(wp.int32(n))` | **`TypeError`** | — | fails loudly |
| `arr[K : K + 1]`, both bounds Warp-typed | **39.4 µs** | 3.16 µs | **12.4x** |
| `arr[K : 2]` / `arr[0 : K]`, one bound | 27.3 / 15.7 µs | | 8.6x / 5.0x |
| `wp.length(v)` | 9.05 µs | 0.63-0.71 µs | 13-14x |
| `wp.min` / `wp.max` on `vec3` | 10.73 µs | 0.29 µs | 37x |
| `wp.inverse(mat44)` | 9.01 µs | 2.85 µs | 3.2x |
| `wp.normalize(v)` | 8.41 µs | — | |
| **`wp.cross(a, b)`** | 9.31 µs | **12.37 µs** | **0.75x — Warp wins** |
| `vec3 - vec3` | 4.07 µs | 0.24 µs | 17x |

- **Only `+ - *`, slicing and explicit builtins are silent.** `//`, `%` and
  `wp.zeros(wp.int32(n))` raise; comparison and construction are free. This is why check 26 (§4.5)
  is a clean scan.
- **A slice is three dispatches** (`__getitem__` forms `stop - start` and `int(strides) * start`
  itself), so a partially typed slice still costs 5-8x; read `arr[K:]` and `arr[:K]` too.
- **`wp.constant(x)` does not make a value Warp-typed and `wp.float32(x)` does not round it**
  (`wp.constant(7)` is a plain `int`, §12.6; `scalar_base.__init__` is `self.value = x`).
  Wrapping a host intermediate buys nothing and costs a dispatch per later operation.
- **Vector arithmetic takes a different path**: `vec_t.__sub__` runs a Python component loop, and is
  invisible to the `Function.__call__` census (§15.11).
- **Before arithmetic on a `wp.constant` at Python scope, wrap it in `int()` or derive a plain
  value once at import.** `kernels/array.LOOP_CONDITION_VIEW` is a plain `slice` derived from the
  constant so the two cannot drift (§12.6 has the other half: a typed constant raises in host
  `//`).

**Cheap replacements, measured.** `float(wp.length(upper - lower))` -> `math.dist(lower, upper)`
is 4.9x (a `wp.vec3` indexes to a native `float`). `wp.min(a, b)`, `wp.max(c, d)` -> componentwise
`wp.vec3(min(...), ...)` is 7.6x and byte-identical. Where the NumPy buffer is already in hand the
win is larger (`enclosing_diagonal`: one `np.linalg.norm`, 13.4x). Quote the end-to-end number
beside the expression's: `bounds.aabb_union` measures 3.55x rather than 7.6x (`require_same_device`
and the call itself). `math.dist` computes in float64 where `wp.length` is float32 (~2e-08
relative); gate such a change on both devices, not by byte-comparison.

**This partly contradicts §3.8's "NumPy standing in for a Warp equivalent" defect**: for `length`,
`min`/`max`, `inverse` and vector arithmetic the measurement runs the other way by 3-37x, because
Warp's route to a host operation is a builtin dispatch. §3.8 stands for not forcing a NumPy round
trip or naming `np.ndarray` in a public signature; `wp.cross` is the genuine case where Warp wins.
**Decide per operation from the table.**

**Packing costs are host constants, flat in the data** (1.46 ms at 0.07 MB and 2.42 ms at 268 MB):

| operation | per segment |
|---|---|
| `wp.copy` (`pack_1d_arrays`, `concatenate`) | **6.02 µs** |
| `wp.clone` (`split(copy=True)`) | **15.06 µs** (~10 µs allocation + ~6 µs copy) |
| a `wp.array` slice view (`split(copy=False)`) | **3.63 µs** |
| one whole-buffer `wp.copy`, 48 903 to 2 614 242 elements | **0.010-0.015 ms**, flat |

- **A `wp.array` view can be stamped instead of sliced, and `array.split` does.**
  `object.__new__(wp.array)` plus a copy of one *fresh* template slice's `__dict__` with `ptr`,
  `shape`, `size`, `capacity` patched is the same object (`_ref` keeps the base alive) at ~0.4 µs a
  segment against 3-3.5 (7-9x). The template must be made inside the call (a reused view may carry
  a cached `ctype` holding its own pointer). A trailing empty segment stamps safely;
  `requires_grad`, non-contiguous and subclass bases fall back to real slicing.
  `test_segment_views_match_real_slices` pins the layout key by key, so a Warp release that
  renames an attribute fails there.
- **`require_same_device` scans unlabelled first** and builds labels only on a mismatch (one
  f-string per list element was half of `pack_1d_arrays` at 2 048 segments).
- **Warp has an array-of-arrays**: a `@wp.struct` may carry a `wp.array` field, and a `wp.array` of
  that struct is a descriptor table a kernel indexes as `segments[s].data[k]`. One launch over it
  replaces the copy loop and is flat in segment count and total size, so the choice is a threshold,
  `array.PACK_SEGMENTS_KERNEL_FROM = 32`: a loss below 8, **1.89x at 32, 14.9x at 256, 150x at
  4 096**. A "Warp has no X" comment is a claim to probe (§10), not a fact to inherit.
    - **Build the descriptor vectorized**: one NumPy structured array through the struct's
      `numpy_dtype()`, uploaded once; instance-by-instance construction gives the win back.
    - **One kernel serves every dtype**: declare the field `wp.array[wp.int32]`, length in 4-byte
      *words*, and alias the destination the same way (`wp.array(ptr=..., dtype=wp.int32,
      shape=...)`; `wp.array.view` refuses a dtype of a different size). Verified byte-identical for
      `int32`, `float32`, `float64`, `vec3`, `uint64`. An itemsize that is not a multiple of 4
      keeps the loop.
- **`split(copy=True)` has a contract floor**: its outputs are `n_segments` separate arrays, so the
  allocations are the return value; one buffer plus views would pin the whole buffer while any
  segment lives, the opposite of `copy=True` (`copy=False` exists for views). It fills the
  allocations with one `unpack_segment_words` launch from 32 segments (2.5x). Declined spellings
  (raw-pointer view, shared output buffer, dropping `src_offset=` / `count=`) are at their sites in
  `ordito/array.py`. **The NumPy crossover is a segment SIZE (~98 kB), independent of the
  segment count.**
- `state.assign([0, 1, 0])` is a third of the cost of three `wp.zeros(1)` / `wp.ones(1)` buffers,
  so packing a loop's state into one word is a saving as well as a convention. A seed launch takes
  one row view, not a flattened prefix (views cost ~3 µs).

**A device reduction costs ~0.10-0.32 ms flat on CUDA regardless of `n`** (launch + 4-byte read),
while a readback scales with bytes: the crossover is where the copy exceeds ~0.15 ms (~200 k
`int32`/`float32`, 1 M `bool`, 16 k `vec3d`). Under ~100 k elements both forms sit at the ~18 µs
launch floor.

### 13.2 Device-side and memory access

| primitive, in-kernel, amortized over 100k iterations | cost |
|---|---|
| dependent L2 load (160 KB / 640 KB working set) | **118 / 147 ns** |
| dependent load, 64 MB working set | 678 ns |
| 15 *independent* loads from one thread | **524 ns total** (~35 ns each) |
| `wp.tile_sum(wp.tile(x))[0]`, `block_dim` 32 / 64 / 128 | **126 / 325 / 369 ns** |
| `wp.tile_scan_exclusive(wp.tile(x))` + `untile`, any width to 32 | **353 ns** |

**These decide whether a cooperative (one-block, barrier-synchronized) rewrite of a serial kernel
can win, before writing it.** A correct level-synchronous round needs at least 2 barriers plus a
prefix scan (~600 ns); if the level's own work is under that, the rewrite loses (§14.9).

- **`wp.tile_sum(wp.tile(x))[0]` is a block-wide broadcast reduction**: correct on every lane, it
  syncs before and after, so it doubles as a full block barrier (as does `tile_scan_exclusive`).
- **Tile ops are legal inside a dynamic `while` loop** on CUDA. Keep the condition block-uniform
  (a global every lane reads after a barrier, or registers every lane updates identically);
  `__syncthreads` in divergent flow is UB.
- **A single thread's throughput is ~35 ns per independent memory op**, so a serial pointer-chasing
  kernel is usually throughput-bound, not latency-bound.
- **A `wp.hash_grid_query` cell probe costs ~600 linear-scan point tests** (a hash plus two
  dependent uncoalesced loads). Once the search radius outgrows a couple of cell widths an exact
  O(n) scan is cheaper than widening the walk; the break-even span scales as `n^(1/3)`, which is
  why `_knn_widest_grid_radius` is `n`-aware.
- **`launch_tiled` with every lane walking the whole chunk and lane 0 doing the atomics beats one
  thread per chunk**, though it looks 64x redundant: the loads broadcast, while one thread per
  chunk stops coalescing. The redundant arithmetic is free in memory-bound reductions
  (`accumulate_procrustes_moments`).
- **Dropping a `sqrt` from a ball query's narrow phase is flat** (declined, and `wp.length` vs
  `wp.length_sq` are not the same predicate, §12.4).
- **A single-address `float64` `wp.atomic_add` serializes the launch**: a global dot needs a
  two-stage reduction.
- **`reduce.sum` / `mean` / `weighted_sum` on floats are a fixed-order two-stage sum** (2026-10-02,
  `reduce._sum_in_fixed_order`, `kernels/reduce.SUM1D_PARTIALS`): each block stores its partial
  in its own slot and the same kernel folds the partials until one block remains, so the result is
  bit-identical across runs and processes on CUDA (one atomic per block summed in arrival order
  moved the last bit of `mean_edge_length[bunny]` between processes). One launch up to 1 024
  elements as before, one more per factor of 1 024 after. Integer sums keep the atomic (exact in
  any order); internal device-resident float folds (`smoothing`'s `adil_sum`,
  `points._point_sum`) still commit atomically.
- **A `wp.tile(v, preserve_type=True)` keeps a vector tile** and `wp.tile_sum` reduces it
  componentwise in **one** tree, bit-identical to one reduction per component, for vectors,
  matrices and `wp.types.vector(length=N)`. Pack several same-dtype quantities into one vector for
  one barrier (`reduce.block_sum`; 25 barriers to 1 in `accumulate_procrustes_moments`, 44 to 3 in
  `accumulate_point_to_plane`). A plain `wp.tile(vec3)` decomposes to a scalar tile.
  `wp.array.view(wp.float32)` on a vec3 array gives a zero-copy `(n, 3)` view for `reduce.minmax`.

**Single-slot atomic reductions.** A single-slot atomic reduction serializes on one address, so its
cost is linear in the launch; lane-partition + `wp.tile_sum` + one guarded commit is ~1x at a few
thousand elements and 10-30x at a million, with no CPU/CUDA branch (lanes stride a block-owned
chunk by `wp.block_dim()`, §2.2).

- **Do not convert a *conditional* atomic** (compaction cursor, change flag, rare-event counter):
  contention scales with hits, not the launch.
- **The recurrence is one atomic per block: the variable is the block count, not the lane
  redundancy.** Removing the 64-fold redundant lane arithmetic recovers almost nothing; re-launching
  at the reduce module's fold width is worth several-x at large sizes. The shape is greppable: a
  `launch_tiled` at `dim` sized to the *item* count (not `blocks_1d`) with a `lane == 0` commit.
- **A kernel that partitions the *outer* work at a constant stride (`_sliced`-paired) is a
  different case**: the quantity the fold reduces is `blocks x accumulator slots` and needs to be
  ~1e5 or more before there is anything to win. Compute that product first.
- **A wide accumulator needs the fold more.** All `wp.tile_sum`s run *outside* the `lane == 0`
  guard (block collective); only the commit is inside. Assemble the whole vector/matrix from the
  tile sums first, since `wp.atomic_add` reads trailing indices as array dimensions.
- **Flattening a reduction to its launch floor can make a neighbouring fusion a large win and expire
  a written decline**: re-read the declines about the launches either side of a big win (§9).
- **Padding must be a hole, not a value.** Padded rows pointed at a dummy valid index collide on one
  entry and serialize (as with §12.7's zero-padded triplets); send them out of range.

**A fused fold's tuning variable is the fold width, and a kernel that also writes per element must
keep its per-element dimension in the grid.** Launching a fused count fold at `blocks_1d` grain
(`ITEMS_PER_BLOCK_1D = 1 024`) collapses the grid: `homology.dual_candidate_mask` at 140 964 edges
gets 137 blocks (under one per SM on 170) against 2 188 at one tile per block; the wrong fold cost
0.92-0.95x. The narrow form launches `dim = ceil(n / TILE_1D)` with `tile_chunk(n, chunk, TILE_1D)`.
A pure reduction takes the wide fold; a reduction fused onto a map does not (§2.3's occupancy rule).
Narrow-fold counterpart: rows per block when the fold commits `float64` to constant slots; 256 won
at 41 k and 1 M rows where 1 024 starved the grid and one row per lane contended 15 625 blocks.

**Three tiling antipatterns, all measured in `kernels/reduce.py`:**

1. **A `wp.tile_load` kernel below one tile is a large loss** (~49x on one case): when the *reduced
   extent* is under the tile width every lane redundantly walks the same short row. Fall back to one
   plain thread per output row. **The dispatch key is the reduced extent, not the axis**: the same
   fallback on the other axis is a large loss (too few outputs to fill the device serially). It
   recurs for 2-D tables with a short trailing dimension; gate on the trailing extent and
   contiguity (flattening a non-contiguous view raises).
2. **One atomic per 64-element tile does not scale** (~4.9x lost at large sizes). Fold several tiles
   into a register before the atomic; the swept fold width never loses across the range.
3. **`tile_chunk` reports what is left to the end of the array, not the block's share; clamping is
   the caller's job.** Tile-load factories satisfy it through a fixed `TILES_PER_BLOCK_1D` loop. A
   loop bounded by `remaining` (`for k in range(t, remaining, wp.block_dim())`) must write
   `remaining = wp.min(remaining, ITEMS_PER_BLOCK_1D)` or block 0 walks the whole array. It fails
   loudly for sums and **silently for min/max/any/all** (idempotent re-reads). The clamped form
   has one spelling, `reduce.block_chunk(n, block, width)` (`block_chunk_1d` is it at the fold
   width); a kernel owning its own rows per block calls it rather than re-deriving the clamp. Changing a kernel's
   per-block contract silently breaks any other module that launches it with its own `dim`:
   `grep` for direct launches first, and reuse the exported block-count helper.

### 13.3 Tuning constants are per-device

`ITEMS_PER_SLICE` (elements per thread in the lane-free strided-slice reductions): **CUDA wants
long slices, CPU short** — `ITEMS_PER_SLICE_CUDA = 128` / `ITEMS_PER_SLICE_CPU = 32` behind
`_device.items_per_slice(device)`. It is not one number within a device either: 32 suits reductions
into a few accumulators, a *per-query* reduction wants ~128 (`proximity.ITEMS_PER_QUERY_SLICE`)
because the query dimension already fills the device and a short slice only multiplies atomics.

- **Adding a device-dispatched path: re-ask which device still reaches each branch and re-sweep the
  constants that branch reads.** `ITEMS_PER_SLICE` was tuned against two call sites that stopped
  reaching the sliced form on CUDA.
- **A constant whose sweep sampled only two values has not been shown to be a two-bracket
  problem**: a finer re-probe found a third bracket between the two points, costing a real loss at
  the size the old crossover was tuned around. Sweep values not tried the first time.
- Re-probe tuning constants after a Warp upgrade (§9); sweep both devices and split the constant if
  the optima differ.

---

## 14. Kernel-shape verdicts

### 14.1 Block-per-item (one block per item, lanes stride the inner sequence)

**It wins where the outer dimension alone starves the device and loses where it does not** (the
criterion and table are §2.3). Three shapes take it: `kernels/visibility.py::obscurance` (block per
point, lanes over its ray bundle, 3.2-11.8x, `block_dim=64`), `shape_diameter` beside it, and
`points.farthest_point_sample` as **one persistent block** (`launch_tiled(dim=(1,))`, lanes
striding the cloud, `wp.tile_max` over a packed key as both argmax and barrier, identical
tie-break; `block_dim` 1024 on large clouds, 256 on small).

**Binary occlusion traces any-hit, in its own kernel** (2026-10-04, R30-1, `occlusion`):
`mesh_query_ray_anyhit` stops at the first hit, 1.06-1.14x on bunny / dragon / lucy at 64 / 256 rays,
values identical on both devices; `obscurance` (`tau > 0`) keeps the closest hit. **A point-major
layout** (`occlusion_point_major`: 32 Morton-consecutive points x 4 ray groups per 128-lane block,
per-point sums one `block_sum` of a 32-vector) adds 1.28x / 1.19x at `lucy` (14 M points), values
identical, from `visibility._OCCLUSION_POINT_MAJOR_FROM = 4 M` points on CUDA (sweep of vertex
subsets: 0.96x at 1 M, 1.02x at 4 M). Reading the per-point sums off a lane-sized tile instead was
0.89x. **Not for `shape_diameter`**: its inward bundle measured 0.65x point-major (scattered scratch
rows) and 0.98x with a ray-major scratch, though an *outward* closest-hit bundle gains 1.25x there
(price the actual rays). **Nor does its `(n, n_rays)` scratch cost anything** (15 GB at `lucy` and
256 rays): register-held distances (one unrolled kernel per rays-per-lane bucket) were 0.97-0.98x.

**The caller's BVH builder sets the bundle cost, and the answer does not move** (2026-10-06,
Warp 1.18). The bundle kernels trace the caller's `wp.Mesh`; built with `bvh_constructor="cubql"`
instead of the default LBVH, `ambient_occlusion` / `shape_diameter` at 64 rays are 1.50x / 1.94x at
`dragon` (34.9 / 55.0 -> 23.1 / 28.4 ms) and 2.51x / 2.18x at `lucy` (1 606 / 2 708 -> 641 /
1 243 ms), outputs bit-identical, and every builder matches a brute-force `float64` ray oracle ray
for ray (bunny, dragon). `sah` at leaf 1-2 is 1.37-1.45x at `dragon`, 1.9-2.3x at `lucy`; LBVH leaf 1
1.11-1.16x; leaf 8 0.78x. The build is the price: 2 / 37 ms default against 80 ms / **3.9 s**
`cubql` and 0.4 / 13 s `sah` at `dragon` / `lucy`, so building one inside a single call barely
breaks even at 256 rays; the win is a caller who builds once and traces many rays. The functions
take a caller-built mesh, so the module only documents it (owner's decision, option c; a
`Trimesh` property for a traced mesh is open). Probe trap: with `normals=None` the closest-face
normal at a vertex is a tie the BVH breaks, so comparing builders without explicit normals reads
as answers changing by up to 0.88.

**`max_tangent_sphere`: the shrink loop's answer is a tolerance window, and "exact" is not
"true" on a mesh** (2026-10-06, probes only, nothing shipped). `thickness_interior[sphere_med]`'s
422 ms is 12 rounds whose first five cannot prune (every inward ray hits the antipode, so the seed
centre is equidistant from all 81 920 faces). The loop stops anywhere in `[r_true, r*]`, `r* = tol +
R(p + tol n)` with the absolute `TOLERANCE_PLANAR`; at a convex vertex `r_true = 0` (incident faces
fold into any ball), so on a sphere it returns 0.007 of the ray thickness, as trimesh does. Two exact
alternatives were built (closed-form touch radius per face -- vertex, edge and interior cases --
galloped over `bvh_query_sphere`, exact by nesting of tangent balls; matches a `float64` oracle to
2.6e-5) and compared on `2r / thickness(method="ray")` (the ray is the ball's diameter chord, so
`<= 1`, never violated): **faces through the point excluded** 0.67 / 0.30 / 0.21 median on
icosphere(4) / `bunny_decimated` / `dragon`, because an inscribed polyhedron's second ring sags
into the ball by the same order as the ball's curvature over one edge, so it does not converge
under refinement; **vertices only** (Ma et al. 2012's shrinking ball on the samples) 0.97 / 0.36 /
0.28, the smooth answer on the sphere but blind to a face interior between samples; the current
loop 0.007 / 0.049 / 0.32 (its 1e-5 window exceeds small features at `dragon`'s scale). Cost vs the
loop: faces 6.3x / 0.36x / 0.37x, vertices 0.99x / 0.39x / 0.68x -- a ball hugging the surface pulls
up to ~34 000 face boxes into the verification query. A warm start from `r*` under a 128-face
budget is 390x on `sphere_med` and level on scans but answers inside the same window. The contract choice is the owner's.

**FPS wins where the hole DP lost**: its per-iteration work is `n` distances, so one SM suffices and
what it removes is two replayed kernels of launch latency per dependent round.

**Four kernels keep the arg-strided form deliberately**, each annotated with its number: they
already carry a *slice* dimension (`points.hull_support_extremes` is 2.3x at 5 000 points and a
**2-8x loss** at 200 000). What those kernels did take is the opposite move: several outer items
per thread from registers (§14.12).

### 14.2 Cooperative BVH walks

**`tile_bvh_query_aabb` beats the hash grid in the narrow-query regime and regresses above the
crossover**: ~4-8 k concurrent queries for a ball of the cell width, 16-65 k for twice it (one
cloud/radius/query set, all walkers agreeing exactly):

| walker | ball of `r` | ball of `2r` |
|---|---|---|
| `wp.HashGrid` serial (incumbent) | 70-83 µs | 359-394 µs |
| `wp.Bvh` **serial** | 132-148 µs (**worse**) | 317-399 µs |
| `wp.Bvh` **tiled** | **29.5-67.6 µs** | **40-99 µs** |
| hashed cell grid, cooperative | 8.6-12.2 µs | 18.3-28.1 µs |

The serial BVH being worse than the hash grid shows the win is the tiled traversal, not the
structure; it spends 32 lanes on a query the device could already saturate, which is the crossover.

- **Do not re-run the tree-wide sweep**: exactly one of three hash-grid usages could take it.
  `neighbors.query_*` runs past the crossover, `poisson_fem.refinement_oracle` has no block to
  cooperate over (`warp.fem` owns the launch shape), `ball_pivoting`'s pivot search qualified on its
  few-hundred-edge front. Its empty-ball test stays a per-lane serial hash-grid query (each lane
  tests a different ball).
- **A bespoke cell list bounds what an index could be worth**: 1.3-1.7x faster single-threaded
  (Warp's iterator pays for generality), 7-30x with lanes splitting the cells. A *dense* grid cannot
  ship (cell count grows as `n^1.5` for a surface cloud); store the packed cell key per entry and
  compare exactly. Build one only if the query is still the bottleneck.
- **A thread-per-query BVH launch is usually load-imbalanced, not under-pruned: histogram the
  per-thread candidate count first.** On a scan mesh against a translated copy, 0.16 % of
  candidates survive the box prune, 98.2 % of faces return no candidate and 0.5 % of faces carry
  half the traversal. The fix is a capped thread pass appending stragglers to a work list, then one
  warp per straggler. Declined with it: `block_dim` (256 wins at every value from 32) and tightening
  the query margin (the vertex bound equals the answer to 16 digits). The imbalance need not
  persist at scale (§16.6).
- **A BVH walk's device time varies up to 5x with how its loop is spelled, with no resource
  difference** (identical candidates, 37-38 registers, same 33 KB shared stack, no spills):
  `mesh_query_aabb` stored box, no cap 97 µs; same with a literal `c < 32` cap 453 µs;
  `bvh_query_aabb` 522 µs whether the box was stored or formed in the kernel;
  `mesh_aabb_collect` (runtime cap) 98 µs stored box but 455 µs with the box formed from the face
  corners. It is code-generation variance: **time a new or fused walk kernel against the one it
  replaces, on the device** (§16.11). `mesh_to_mesh_distance`'s capped walk with `mesh_query_aabb`
  or precomputed grown boxes measured 0.93-1.08x, so the variance is absent there.

### 14.3 CUDA graph capture

**Capture pays on a launch sequence that repeats identically; a `@wp.struct` bundle pays on one that
runs once.** Recording costs at least what issuing costs:

| arm | vs loose arguments |
|---|---|
| `@wp.struct` bundle | **1.88x** |
| capture-and-replay-**once** | **0.84x — a loss** |
| replay of an already-recorded sequence | **4.29x** |

Do not capture a once-through Python loop (packing reproduces the 0.84x: segment pointers change
every call, so the recording is never reused).

**A sequence that is not repeated can often be made repeated.** A chain of launches differing only
in a loop counter becomes identical once the counter lives on the device: record a *group* once,
replay it to cover the chain, with a `dim=1` kernel stepping the counter as the graph's last node
(replays serialize on the stream). Hole-fill span sweep: **510 launches 6.45 ms issued against 1.10
ms as an 8-launch group replayed 64 times, 5.9x**, recording included.

- **The grid is fixed across replays**, sized for the first group and over-covering later ones.
  Free only where surplus threads exit on a guard they would reach anyway (a span kernel is flat in
  unused grid width, 0.91-1.02x).
- **The group size is a shallow optimum** (recording costs one launch per span; a small group pays
  a counter kernel per replay): 8 won at every rim length; 64 and up lose.
- **Upper bound: device work per launch.** Once kernels cover their own launches the host was never
  critical and replay only adds the counter kernel (a 2-4 % loss).
- Where capture pays it pays large; `quadric_decimate`'s pass is flat in mesh size, so replaying
  every pass at the pass-0 width costs ~1.01x. A captured wrapper chain pays its Python once.
- **Break-even is ~2 replayed rounds** (§16.14); a recording costs more than an issued round.
- `wp.capture_begin` is CUDA-only, so a captured path needs the plain loop as its CPU sibling
  (also the byte-identity reference). §12.6: what blocks a capture; §15.10: attribution.
- **A `wp.capture_while` body may not allocate, and `wp.utils.array_scan` does** (CUB scratch):
  recording raises `Conditional body graph contains an unsupported operation (memory
  allocation)`, while the same scan in a plain `ScopedCapture` is legal. `fill_`, `zero_` and
  `radix_sort_pairs` record into a conditional body. A loop whose round contains a scan is recorded
  as a plain graph and replayed from the host with one 4-byte termination read per round (1.48x on
  a 50-round flip call, §16.4).
- **`wp.capture_while` is slower than a batched host loop** where the per-iteration
  conditional-graph overhead exceeds the sync it removes (ball pivoting batches 8 waves per
  readback). Nesting one inside a capture is fine.
- **DECLINED (2026-10-02): recording a fixed-count smoothing loop** (two ping-pong passes
  recorded once, replayed `iterations // 2` times; `filter_neighborhood_average`, byte-identical):
  0.89-0.90x at 4 iterations, 0.99-1.02x at the default 10, 1.16-1.27x at 50 on
  `bunny_decimated` / `bunny`, 0.99-1.00x on `dragon` at every count. `_launch`'s cached launcher
  issues a launch in ~6 µs, so a replayed one saves too little to repay the recording below a few
  dozen passes. The same shape covers `filter_laplacian`, `filter_taubin`, `filter_humphrey` and
  the other one-to-four-launch smoothing loops (an AST scan for readback-free launch loops lists
  them). Open, unmeasured: `wp.capture_if` (unused) for a stage that spends a host readback
  deciding.

### 14.4 Tile solves: the crossover is K >= 16-32

Batched K x K SPD solves, one thread per system (register Cholesky on a `wp.matrix`) against one
block per system (`wp.tile_cholesky` + `wp.tile_cholesky_solve`), values cross-checked, harness
floor (~14 µs) subtracted:

| K | N | per-thread | tiled | |
|---|---|---|---|---|
| 6 | 65536 | 8.5 | 38.5 | tiles **4.5x slower** |
| 8 | 65536 | 17.3 | 53.6 | tiles 3.1x slower |
| 16 | 1024 | 27.7 | 7.2 | tiles 3.9x faster |
| 16 | 65536 | 54.1 | 100.6 | tiles 1.9x slower |
| 32 | 1024 | 585.6 | 15.6 | tiles 38x faster |
| 64 | 1024 | 3989 | 85 | tiles 47x faster |

Tiles win only when **occupancy-starved** (few systems, large matrix). ordito's only dense solves
are K=6 (N=1) and K=5 (N=n_vertices), both on the wrong side: **"rewrite the small dense solves as
tiles" is refuted.**

### 14.5 NumPy readback vs device reduction: 3 of 7 converted

Interleaved A/B on the scan meshes on both devices (A readback + NumPy, B device reduction); the CPU
axis rejected 4 of 7, so §9's rule: decide on CUDA, measure both.

- **Converted** (win on both): reusing an existing perimeter kernel instead of a per-loop
  `.numpy()`; `.numpy().sum(axis=0)` -> `wp.utils.array_sum` (reduces a `wp.vec3d` array
  componentwise, no kernel); one weighted-branch iso-value reduction.
- **Rejected (NumPy is faster)**: edge-range validation (14.7x CUDA win, **20x CPU loss**);
  `bool(mask.numpy().any())` (a bool array is 1 byte per element, the copy never clears the launch
  cost); a `.sum()` + `.min()` pair that re-uploads; a uniform iso-value branch.
- Method: sequential A-then-B timing gives non-monotonic garbage, and pre-allocating B's scratch
  outside the timed callable flatters it. Grep the whole package for a private helper's callers
  (basedpyright cannot see cross-module callers, §8).
- **The axis is often not the one it looks like**: the per-loop readback's axis was the *loop
  count*; on CPU `vertices.numpy()` is a zero-copy view, and the cost was thousands of Python
  iterations with one `.numpy()` each.

### 14.6 Readbacks inside device loops

A per-pass readback costs ~0.1 ms while one extra loop pass costs 0.9-2.4 ms, so **any scheme that
trades redundant passes for fewer syncs loses**: checking every 2 passes is +5 % to +30 % with
bit-identical output; `cg(check_every=25/50)` is +0 % to +6% (overshoot adds real iterations).

- **`cg(check_every=0)` is not a lever in the harness**: isolated probe 2.1-2.3x, harness
  1.05-1.23x and **0.94x** on the conditioning rows (§15.9). It also changes `cg`'s return to
  device arrays and needs conditional-graph support. Do not re-try.
- **Batching convergence checks loses whenever an extra iteration runs a real kernel** (BVH query,
  whole-graph hook): keep per-iteration 4-8-byte readbacks and fuse several flags into one buffer.
- **Price one loop iteration before removing a host sync**: only remove it if the host could run
  ahead. A readback's profile self-time is the queue depth in front of it, not its own cost.

### 14.7 Per-device algorithm choice (rare; the criterion is asymptotic work)

A function branches on the *device* only when the parallel form does **asymptotically more work**
to expose parallelism, so a backend running a launch grid as one serial loop gets no GPU win.

- **`_device.prefers_tiled_reduction`** is a *correctness* branch (§12.2), not performance.
- **`polyline_downsample`'s pointer-doubled greedy walk** (`_DOWNSAMPLE_DOUBLING_FROM = 2048`, CUDA
  only): the kept set is the orbit of point 0 under "next point at least `step` further along";
  build that successor function for every point and pointer-double it. On CPU it loses at every
  size (30x at 65 536: `n log n` against `n`, and Warp's CPU walk is faster than CUDA's). Serial
  over the jumped form: 0.72x at 512 points, 0.89x at 1 024, 1.15x at 2 048, 2.59x at 8 192, 15.4x
  at 65 536. `double_greedy_orbit` fuses a round into one launch and chases up to
  `POINTER_JUMP_MAX_HOPS = 16` pointers per launch.
    - **Exactness is the claim**: the successor search evaluates the same float32 predicate the
      walk does (`cum[mid] - step` would not), so masks agree bit for bit. No large-`n` NumPy
      oracle exists (float64 sequential `cumsum` vs Warp's float32 tree scan differ by the order of
      the gaps between decisions); exactness is pinned ordito-against-ordito and the large-`n`
      test is invariants only.

### 14.8 Solvers: the cycle is launch-bound

**A smoother that buys iterations with launches loses inside a V-cycle.** A single-level
polynomial *preconditioner* is the opposite trade and wins (§16.15). A Chebyshev multigrid smoother
was built and reverted: against damped Jacobi over five systems Jacobi wins four, the one loss is
1.03x and the worst Chebyshev cell is a **2.07x regression**.

- The launch argument overstates the case ~4x (one V-cycle apply is 34 launches, the smoother 9, so
  Chebyshev raises the cycle's launch count 1.13x against a ~1.03x iteration decrease). **What
  refutes it is interval robustness**: no single `(degree, interval)` is best everywhere and `rho/5`
  is a cliff of up to 17x. Do not quote "the cycle is launch-bound" as the reason.
- Traps: a small synthetic system lied (a 576-unknown grid Laplacian gave a 7x iteration reduction
  that vanished on real meshes); `sweeps` is a default argument frozen at import, so patching the
  module constant changes nothing.
- Direct GPU factorization and conditioning-flatness evidence: §16.16.

### 14.9 Refuted, with the code written — do not re-propose

**Read §14.11 first**: two entries below are single-block rewrites of a whole problem and lose;
keeping the grid full while trading a launch for a block barrier is the biggest win in the package.
The distinguishing variable is whether the grid stays full.

- **A single-block cooperative BFS drain.** Byte-identical, a large loss: the serial drain is
  memory-throughput bound on one thread (software pipelining and register-vector batched loads did
  not help), and on a narrow non-growing frontier the per-level work is below the barrier cost of a
  correct cooperative round, so thousands of levels put a multi-ms floor on synchronization.
  **A per-level dispatch cost only matters if the level body runs** (on a ribbon graph the parallel
  loop emitted 6 nodes of 40 962 and the rest was the serial drain). `graph.bfs` is deleted; the
  lesson stands.
- **A persistent one-block-per-loop tiled kernel for the Liepa hole-fill DP.** Byte-identical, loses
  at every rim size, worse as the rim grows (total work outgrows one SM). Same conclusion as BFS
  from the other direction (too much work per level for a block); only a grid-wide barrier would
  beat both, and Warp exposes none (§12.2).
    - **The *blocked* interval DP is refuted by arithmetic** (§16.12): a tile-diagonal holds
      `B/tile` tiles where a plain span level already launches `B - span` blocks, and that width
      fills the device. **Merging `C` dependency levels into one kernel caps the block count at
      ~`B/C` and multiplies work by ~`C/2`, so it pays only where the unmerged levels were
      themselves under-occupied.** Apply that test before citing §14.11 for a new DP. What removed
      the fill sweep's launch cost was making launches cheaper (§14.3).
- **Tile solves at ordito's problem size** (§14.4). **A Chebyshev smoother** (§14.8).
- **Voxel aggregation for the multigrid hierarchy**: the geometric aggregation blows operator
  complexity up far more than the algebraic one, and a cheaper unsmoothed variant does not fix the
  convergence-rate dependence on mesh resolution.
- **Micro-optimizations of the old Bridson blue-noise propose kernel** (superseded, §16.7): cheaper
  random permutations added bias or overhead; dropping either pruning pass was a large loss (both
  load-bearing); an eager shuffle made lazy was a wash.
- **A `wp.Stream`-overlap rewrite for "independent-but-sequential" pairs, tree-wide.** Five
  candidates built with two streams joined by `wait_stream`: **0.87-1.07x**, never repeatably a win.
  Branches are host-launch-dominated (little device time for a second stream to hide behind, and
  `ScopedStream` costs about as much), and a CG solve's periodic host readback completes before the
  next Python call issues, so sequential Python calls cannot overlap through streams alone. Check
  the device/wall split first. For one operator with several right-hand sides use
  `linalg.solve_spd_columns`'s `_BatchedCg`; merging Krylov subspaces was removed (§16.16).

### 14.10 Producer-consumer fusion: always fuse

**Two consecutive launches at the same `dim` where the second reads the first's output only at its
own thread index are fusible, and the fused kernel is faster: every pair measured, at every size,
1.04-3.46x** (one launch, one allocation and one global-memory round trip fewer).

**The 2026-10-01 tree-wide audit** (every consecutive launch pair in every wrapper function: 217
functions, 413 pairs) found the earlier 89-pair backlog worked through. About 375 pairs are
blocked: ~150 because the consumer needs the producer's complete output (atomics, neighbour reads,
reductions, union-find rounds, claim/commit), ~140 by host work between (scan, sort, readback,
readback-sized allocation), ~55 by exclusive branches, ~20 because both sit in a replayed graph.
About 14 carried a recorded decline. Of ~60 fusible pairs, 41 landed (byte-identical on CPU) and
the rest are declined at their sites with the number. The scan is an `ast` walk mapping argument
expressions onto kernel parameters (resolve kernels through the wrapper module's namespace at
runtime, so `OverloadTable` entries and factories resolve); re-derive it rather than re-reading call
sites. Its traps: locality is necessary and not sufficient; it does not follow `@wp.func` calls,
does not see intervening host work or replayed graphs entered through a helper
(`run_device_loop`), mislabels a scatter-then-per-vertex pair at different `dim`s as a candidate,
and a claim/commit independent-set pair is never fusible (commit must see every claim). Read both
kernel bodies and the wrapper lines between.

**The 2026-10-04 re-audit** (406 pairs after rounds 28-30) found nine fusible pairs and ten
independent ones; the rest split as above. Landed, each byte-identical on CPU:

| Fusion | Gain |
|---|---|
| `grouping.hash_vector_rows`: round and pack in one kernel (`round_pack_vec3`) | 1.13-1.59x, `remove_duplicated_vertices` 1.03-1.05x |
| `filter_mut_dif_laplacian`: the mean's sum folded into `mut_dif_adil_pass` (256-row fold), `adil` recomputed by the step | 1.07-1.16x small meshes, flat at `dragon` / `lucy` |
| `remesh._classify`: feature counts kept, every reader applies `finalize_vertex_codes` | `isotropic_remesh` 1.01-1.04x |
| `split_mesh_with_plane`: the emit labels its own children (`plane_face_above`) | 1.02-1.06x |
| `clip_mesh_with_field(cap=True)`: `on_level_set` read per corner, no per-vertex mask | 1.05x small meshes, flat large |
| `procrustes(return_cost=True)` and `icp`'s rounds: every lane forms the fit (`fit_transform_and_accumulate_cost`) | 1.04-1.07x, `icp` 1.00-1.02x |

Declined with numbers at the site: the oriented-box loss table in the seed block (0.90-1.01x), the
loop frame inside `triangulate_rings` (0.90-0.97x), the shell centroid per tetrahedron thread
(1.00-1.05x), and `zero_at_boundary_edges` on `cr_gradient_rows`' grid (1.00-1.01x). The last is
the only grid-share measured. It saved one launch, about 1 % of a 0.5 ms call, and read flat.
That prices the other nine grid-shares: each saves one launch, which is at most 1.4 % of its call
(`subdivide_loop(return_operator=True)` at 0.37 ms) and under 0.5 % for the rest, so none was
built. `repair._dilate_face_mask`'s ping-pong over `dilate_vertex_mask` trades a launch for a
zeroed vertex buffer at the default `max_expand=1` (§13.1: 4.5 against 9.3 us), so it was not built
either.

The audit's verdicts that generalize:

- **The biggest wins were never the pair the scan flagged but what it exposed**: dropping a
  `bsr_copy` once the row walk wrote the scaled values (`SquaredLaplacianPreconditioner` 1.8-1.9x),
  dropping an `(n_a, n_b)` cost matrix once the per-row argmin computed its own costs (`stitch`
  3.15x at 16 384-vertex rims; the old matrix needed 17 GB at 65 536), and an MIS round going
  4 -> 2 launches by forming keys from state on read (aggregation 1.35x).
- **Removing a 2-D tabulation can lose its parallelism**: one thread per row walking the costs was
  0.79x at 4 096-vertex rims; one block per row with lanes striding by `wp.block_dim()` and
  `reduce.block_argmin` was 1.24x there and bit-identical (§2.2, §2.3).
- **Forming a table entry on read costs a load per read**: on a *dependent* chase (pointer-jumping
  ranking) the inline init lost 0.96x; on a serial per-loop walk (`geodesic_walk`'s repeat flags)
  folding the owner labels lost 0.86x. Fold on read only where the read is independent.
- **An argument added to a launch repeated per iteration can outweigh the launch it removes**
  (~1 µs per argument, §13.1): declined at `filter_humphrey`-style fit loops.
- **An F6 grid-share (two independent launches, one grid) is worth 1.00-1.08x on the call**; take it
  where the branch is per block or per thread range and the bodies already exist as `@wp.func`s.
- **A fusion that changes a slot or box index needs a bounds-checked run** (`wp.config.mode =
  "debug"`): `array.atomic_min_packed_box` takes a *box index*, and passing the element offset 6
  wrote past a 12-slot buffer. On CUDA it was silent and the A/B read identical (the probe's queries
  sat inside the mesh box, so the unwritten box was never the extreme); on CPU it corrupted the
  JIT's state and one run in four failed with `Failed to find forward kernel ... for device 'cpu'`.
  Make the probe fixture exercise every slot a change writes.

- **Measure the region, not the call that contains it** (§15.2). Whole-call A/B gave 1.02x and a
  25-row harness sweep 0.987-1.041x where the region itself is 1.04-2.65x and never slower. A cell
  whose region is a fraction of a percent of the call cannot vote.
- **Two deterministic cross-checks settle it faster than a clock** (§15.6): the kernel count is
  `base - 1` per fused site (no solver iteration moved; a fusion shifts answers by 1-3 ulps) and
  the allocation count is lower by one buffer per site, so sub-1.0 cells are provably noise.
- **A fusion can remove one of two intermediate buffers and still win** (`filter_humphrey` 1.79x).
- **When the middle stage blocks the obvious fusion, look across the loop boundary**
  (`filter_normals`: normalization fused with the *next* pass's seed, peeling the first seed).
- **Two launches of the *same* kernel merge into one wider launch: the biggest win available**
  (a cap kernel run twice per solid becomes one `dim=(2, n_cap)` launch, 3.4x). Grep for a kernel
  launched twice in a row with different scalar arguments.
- **A neighbour list consumed only by a per-query reduction is a fusion too: walk and reduce in
  one thread per query.** `query_ball_with_offsets` / `query_bvh_ball` is a count launch, a scan,
  a readback and a fill launch; a consumer that only sums, counts or thresholds over each query's
  list folds it during the walk instead (`interpolation.interpolate_from_points_in_ball` was the
  model). `discrete_gaussian_curvature` 1.85-4.81x, `discrete_mean_curvature` 1.00-1.71x (the
  per-candidate form's parallelism did not matter), `radius_outlier_mask` 1.10-2.04x, at 2 and 4
  mean edge lengths from `bunny_decimated` to `dragon` (§16.6). A list reused across launches or
  loop rounds (`sample_surface_poisson_disk`, `relax_approx`) is not this shape.
- **A scatter, then an elementwise map, then a consumer reading the map's output by index: the
  consumer applies the map as it reads.** The map is usually a reciprocal, a defect or a cast
  (`energies.curved_hessian_energy`'s kappa and inverse mass, `laplacian_smoothing_loss`'s scales,
  `mass_matrix_entries`' area table and float64 cast: 1.46-2.79x). A *zero-on-boundary* step
  survives the move when it zeroes the map's *input* to a value the map sends to zero. A map of a
  per-face quantity into a whole-mesh total is one lane-strided fold (`measures.volume`
  1.72-2.00x). A map whose result is returned to the caller (`vertex_defects`,
  `average_onto_vertices`) has no consumer to move into.
- **Declined:** launches in different branches (not sequential); pairs inside a captured loop
  (a replayed launch is ~1.17 µs, §14.3).

**A fused kernel is not finished until the duplication inside it is gone.** Inline any `@wp.func`
the fusion leaves with a single caller, then look for what that exposes (a doubled vertex load, a
doubled grid index, a quantity computed twice by two spellings: 1.15x -> 1.29x, and a tie -> 1.13x
at the large end). The fix is a local, not a new `@wp.func`; share a helper only when two *kernels*
repeat a run. Afterwards: deleting the last launcher leaves dead kernels that still cost import time
(§12.6) and may stale the `wp.map` bookkeeping (check 23's allowlist); move the deleted kernel's
prose (sign conventions, preconditions, non-merge decisions) onto what survives.

**One near-duplicate is deliberate.** The scalar-field CSR row walk does not call `operator_row`,
which promotes a float32 weight to float64 for a `wp.vec3d` accumulator:

- *Keep float32 storage, accumulate float64* does not compile (`wp.float64(w) * f32_value` is a hard
  parse error, §12.4).
- *Carry the field in float64* lets one `Any`-generic helper serve both (seed the accumulator with
  the first term; kernel scope cannot spell "zero of `Any`'s type") but measures **0.63-0.86x**:
  the field is one of four streams, so doubling its width costs bandwidth no saved launch repays.
- Two kernels differing only in dtype are not always mergeable; when the merge costs memory traffic,
  duplicate the loop and write down why (the generic form was also reverted on §4.2: one
  instantiation).

### 14.11 Blocked wavefront: trade a launch for a block barrier, keeping the grid full

**A dependency chain of `N` steps over a *grid* does not need `N` launches.** For a 2-D DP whose
cell reads only cells above and to the left, tile it into `T x T` squares and launch one
tile-diagonal at a time: `2 * ceil(N / T)` launches, every other step a block-local barrier.
`stitch_loops_min_weight`'s band DP: **2 049 launches to 65, 11.0-13.3x on the DP sweep, up to
6.94x on the public call**, byte-identical (§16.12).

**It is the opposite of §14.9's single-block rewrites: one block per tile and as many tiles as the
diagonal holds, so concurrency is unchanged and only synchronisation got cheaper.** Ask which one a
proposal is before citing either precedent. Conditions (all met by the band DP):

- **The dependency is local and directional** (`(i, j)` reads `(i-1, j)` and `(i, j-1)` only), so a
  tile-diagonal's tiles are mutually independent.
- **Per-step work is small next to a launch**: ~12 µs launch against ~126 ns for a 32-lane
  `wp.tile_sum` barrier, ~95x per step.
- **Warp exposes no barrier, so a block-collective reduction is it.** `wp.tile_sum(wp.tile(x))[0]`
  synchronises and orders the global writes the next step reads. **Probe by removing it**: it fails
  only at 64 lanes and above (one warp needs none), so a test at the shipped 32 cannot see a
  missing barrier.
- **The schedule must be provably value-neutral**: keep the cell body in one `@wp.func` both kernels
  call and pin the fast schedule to the simple one byte-for-byte (§2.4), on **one** set of inputs
  (rebuilding per arm compares different problems wherever upstream uses a float atomic).

**Where to look next:** a Python loop issuing one launch per step whose `dim` is a slice of a 2-D
table. **Check the untiled schedule's occupancy first**: the fill DP has the same shape and tiling
it is refuted (a span level already launches `B - span` blocks). The stitch DP won because an
anti-diagonal of O(1)-work cells was under-occupied before tiling. Where levels are already wide,
record the launches instead (§14.3).

---

### 14.12 Register-block the outer items of a strided slice reduction

**A `(item, slice)` kernel that streams the whole cloud once per outer item is bound by that
traffic, not its arithmetic: give each thread `W` items and dot every point it loads against all
of them from registers** (a `wp.types.matrix(shape=(W, 3))` of directions / queries, a
`wp.types.vector(length=W)` per accumulator, the inner `for d in range(W)` unrolled from a factory
closure). Max, min and an argmax with a strict `>` over ascending indices are exact, and a per-item
running sum keeps its face order, so the outputs are the one-item kernel's (byte-identical on both
devices for the extremes and argmax; per-thread sums identical, so CPU is exact and CUDA moves only
by the float-atomic commit order). Measured (2026-10-02):

| kernel | W | gain |
|---|---|---|
| `points.hull_support_extremes` (642 directions) | 8 | 2.5-3.2x at 0.4-0.5 M points, 4.6-10.8x at 14 M (with §16.13's slice filter); 16 is slower than 8 |
| `bounds.oriented_box_extents` (candidate frames) | 4 | `oriented_bounding_box` 1.2x at 0.4-0.5 M, 4.26x at 14 M |
| `proximity.winding_number_tiled` (queries) | 4 | 1.7-2.0x at 16 k faces, 4.2x at 69 k, 1.9x at 0.87 M |
| `visibility.support_argmax_sliced` (deferred queries) | 4 | `max_tangent_sphere(inwards=False)` 1.30x at 0.87 M faces |

- **Grouping divides the grid, so `W` is chosen per launch**: the widest of `(8|4, 2, 1)` whose
  `ceil(n_items / W) * n_slices` still reaches `1 << 17` threads, one rule for all four
  (`kernels/array.RegisterBlockedTable.launch_shape`, `REGISTER_BLOCK_MIN_THREADS`; each module
  keeps its own `*_WIDTHS`). A fixed `W = 8` cost a 36 k-point cloud 0.82x.
- **A matrix row held from a global load inside the loop** (`axes[k] * point`) was left to the
  compiler's loop-invariant motion and measured as fast as an explicit register copy; check the
  device time if a new site does the same.

## 15. Benchmark and measurement traps

### 15.1 A sudden slowdown is a Warp rebuild until proven otherwise

Before profiling, believing a kernel got slower, or deleting a test for being slow: a
single-digit-second operation that now takes tens of seconds, or a test file whose cost appears and
disappears with *which* tests you select, is almost always a module recompile from an unregistered
generic-kernel overload (§2.5) or an undeclared `wp.map` signature (§3.5), up to 1 200x apart
between a fresh selection and an identical repeat. Two cheap confirmations:

```bash
# 1. The compile is single-threaded nvcc, so the GPU is idle while the clock runs.
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader   # 0 % during the stall
# 2. Warp says so outright.
uv run python -c "import warp as wp; wp.config.log_level = wp.LOG_DEBUG; ..." 2>&1 \
    | grep -E "Module hash changed, recompiling|took .* ms  \(compiled\)"
```

Any `Module hash changed, recompiling: <module>` line for a `ordito.kernels.*` or `map_*` module is
the defect; a *second* line for the same module in one run means the chain is still forking. Do not
read it as "this test is inherently slow" and cap its input or delete it.

**The second candidate is host-side per-element Python** (GPU idle too, milliseconds-not-minutes
rules out launch overhead): a benchmark ran over 20 minutes at 0 % GPU because a `validate=True`
path built Python `set` comprehensions over a `.numpy()` array. **When a benchmark group stalls,
grep the timed function for a Python `for` / `set` / comprehension over anything derived from
`faces` or `vertices`.** The fix is a device scan, **not a smaller benchmark mesh** (the largest row
exists to catch this).

### 15.2 Attribute against one number

A projection built by subtracting measurements of *different* things is a hypothesis, optimistic by
3-10x every time here; the one that held came from a single directly-attributed number. Mixed-up
quantities seen: a fallback pass's cost from two per-span metrics, an amortized cost read from a
warm unrepresentative solve, a memset count that initialized kernel inputs rather than padding.

- **Price the candidate directly**: time the exact call in isolation, or count launches and
  multiply. A *warm* repeat measures a different regime than the cold one inside the real call (CG
  especially); `wp.empty` vs `wp.zeros` differ in what they remove.
- **Price at more than one size: the sign of the trend is the decision.** One readback's share
  *grows* with mesh size (worth landing); another's *falls* (§9: a decline).
- **The rule fails in both directions.** Measuring a candidate *through* a call it barely occupies
  makes enclosing noise the verdict: a `heat` fusion was declined on whole-call 1.02x / 1.006x and
  a 25-row sweep of 0.987-1.041x, while the region itself is 1.04-2.65x and never slower (0.14 % of
  the call). **Before reading a ratio near 1.0 as a verdict, compute the candidate's share of what
  you measured**; under a few percent, isolate the region instead (§14.10).
- **A loop with a convergence break reports the break point, not the change.** An
  `icp_point_to_plane` hoist read 1.86x end to end and was worth 1.02-1.06x because the arms
  stopped at different iterations. **Pin the iteration count before timing an iterative solver**
  (`threshold=0.0`; for point-to-point ICP `threshold=-inf`). The call is not bit-reproducible once
  converged (float32 atomics), so a value gate compares a *trajectory*.

### 15.3 Attribute at the benchmarked operating point

**A profiled share is a share at one point on the parameter axis and can reverse an optimization's
sign.** A blue-noise inversion (one thread per *accepted* point instead of per *alive* point) won at
the natural dense radius and lost at the benchmarked sparser radius (too few accepted points per
round starves the device). Grep the benchmark for how it derives its parameter and profile *there*.
**The same holds for the fixture**: a `mesh_to_mesh_distance` fix read as a decline on an
interpenetrating pair and was a genuine win on the benchmark's disjoint copies. Read the fixture,
not just the call.

### 15.4 Benchmark-harness hazards

- **`benchmarks/test_meshes.py` is a gate the default `pytest` run does not collect.** It checks the
  registry against a topology table; a mesh registered without its row fails there with a bare
  `KeyError` while everything else stays green. After touching `benchmarks/meshes.py`, run
  `pytest benchmarks/test_meshes.py`.
- **`--benchmark-json` is written at session end, so one pathological row costs the whole module.**
  Put a `skip_larger_than`-style cap first in every group, before any library-specific branch; make
  sure it covers meshes outside the reference's own registry; free any foreign library's device
  memory in its teardown outside the timed region; and keep a bounded per-module wall-clock cap in
  the runner.
- **Never benchmark Poisson reconstruction on very large clouds against a reference that does not
  scale** (one run exceeded 90 minutes without completing a row). When a module's wall clock looks
  wrong, get the per-library split first: the slow side is often the reference.
- **A harness number and an isolated hand-probe number for the same call are not comparable** (the
  harness syncs and cold-pools differently): compare probe to probe or harness to harness, and label
  which a quoted number is.
- **A row's cost can depend on whether its *setup* left a large mempool chunk mapped.** A CUDA
  mempool chunk is released at a sync only once nothing in it is live, so an input allocated
  inside a big chunk keeps that chunk's free remainder warm for the timed call's scratch.
  `group[dragon]` read 0.74x after `edges_unique` moved from a hash table to a sort: the timed
  call is unchanged and its sort identical (0.155 ms), but HEAD's inverse sat in the hash table's
  chunk, so `group`'s two 21 MB buffers allocated in 0.034 ms against 0.094 cold. HEAD handed a
  value-identical *fresh* copy is as slow as the new tree; `edges_unique_inverse` + `group`
  together is 2.0x faster. **When an unchanged function regresses in the harness, time its
  allocation step with and without a fresh copy of its input** before suspecting the code.
- **A median at low sample count (`rounds=3`) can misrepresent its samples**: a one-off cost (a Warp
  module load) in two of three samples inflates the median while the floor never moved. Run the
  aggregate script's "suspect" check (median far above its own minimum, on ordito's and the
  reference's cells alike) and re-measure a flagged cell before building against it.

### 15.5 A plan item may be refuted by its own target

A measured decline is prose next to the code, so a table- or diff-driven plan cannot see it: all
three items in one plan round were refuted by text already in the target file (a constant's
counterexample, a benchmark docstring's scope caveat, a threshold's tuning basis). **Before
measuring a plan item, grep the target function, its constant's comment and its benchmark docstring
for a number.** When a "nothing separates these" note exists, check which *kind* of quantity it
ruled out: several predictors tried properties of the *solve* (size, iteration count, convergence
extrapolation) and all failed; the separating property was of the *operator* (§16.15).

Two refutations from the host-cost sweep, both wrong-looking-right:

- **Narrowing edge/row keys to 32 bits to halve the radix sort.** `radix_sort_pairs` is 1.3-4.1x
  cheaper for `int32` than `int64` in isolation, but end to end
  `unique_1d(return_inverse=True)` on `uint32` keys is **0.91-1.01x** vs `uint64`: `uint32` is
  outside `_unique_hash`'s native `(int32, int64)` set (extra `bitcast_to_int` copy) and the call is
  host-bound. The lever was the reduction *around* the packing (§16.4).
- **Replacing `map_sorted_inverse`'s binary search with a hash-slot lookup.** Stage-profiled it is
  **under 5 %** of `unique_1d` best case, against an extra `n`-sized `int32` buffer and a wider
  `hash_insert`. Not built. Stage-profile before optimizing a stage; the radix sort's host floor is
  the bigger share.

### 15.6 A/B without `git stash`

**Do not `git stash` to compare against pre-change code**: the working tree is shared, and a stash
cycle can revert in-flight edits from outside the session. Use a detached worktree:

```bash
git worktree add -q --detach $SCRATCH/baseline HEAD
```

**`uv sync` installs ordito through a MetaPathFinder that outranks `sys.path` and
`PYTHONPATH`**, so `PYTHONPATH=$SCRATCH/baseline python probe.py` silently imports the
working tree. Drop the finder:

```python
sys.meta_path = [f for f in sys.meta_path if "editable" not in getattr(f, "__module__", "")]
sys.path.insert(0, BASELINE)
```

- **Verify which tree loaded** before trusting a number: assert the *submodule's* `__file__` (the
  lazy `__init__` makes `ordito.__file__` prove nothing). Do not `uv run` from inside the worktree
  (it builds a second virtualenv); use `.venv/bin/python`.
- **A `sed`-based in-place sweep of a tuning constant is the same hazard as `git stash`.** A
  constant baked into a kernel cannot be swept in one process anyway (fixed at codegen); put each
  value in a detached worktree and drive it with the main venv's interpreter.
- **Check the box is yours before any clock reading**: `nvidia-smi` utilization and a process
  listing cost nothing (foreign processes once produced a 4x swing at fixed parameters). Interleave
  A and B inside one process where possible. **When the box is not quiet, measure a non-clock
  quantity**: iteration, launch, level and candidate counts, `nnz`, whether a reference mutated its
  input.
- **A failing `nvidia-smi` is not evidence CUDA is unusable.** NVML is a separate library from the
  driver API Warp calls; a host with `Failed to initialize NVML: Driver/library version mismatch`
  can still launch kernels with the full CUDA suite green. `nvidia-smi` is the box-is-quiet check
  only; availability is `wp.get_cuda_device_count()` plus one real launch.
- **torch reports the same mismatch as the suite's only warning on a CUDA run**
  (`UserWarning: Can't initialize NVML` from `torch.cuda.__init__`; a CUDA run ends in `2 warnings`,
  CPU in none). It is the box, no code change silences it, reloading the NVIDIA kernel modules (or
  rebooting) is the remedy. Do not filter it.

### 15.7 Timing hygiene

- **Interleave A and B in one loop and report the `min` beside the median**: timing all of A then B
  lets GPU clock state decide the winner (non-monotonic ratios reversed into a clean trend once
  interleaved with the GPU pre-warmed).
- **Saved baselines drift +-10 % (+-30 % under 100 µs) between sessions**; flagged deltas in that
  range appear on unchanged code. Re-run both arms back-to-back.
- **A reference library's own column is the control that licenses a cross-session comparison**
  (unchanged within noise proves the drift belongs to ordito's side). Re-run the whole group
  including reference rows; rewrite a stale table rather than annotate it.
- **Do not `wp.synchronize_device` around each launch** of a microsecond kernel (measures sync
  latency). Batch K launches, sync once, divide.
- **The device/wall timer's per-launch synchronization inflates wall badly** (once >10x). Take
  launch counts and device totals from `wp.timing_begin(cuda_filter=wp.TIMING_KERNEL |
  wp.TIMING_MEMSET, synchronize=True)` and wall from a separate un-instrumented loop.
- **A/B one pytest process per module, never several modules in one process.** A 15-module
  single-process A/B put ~24 cells under 0.93x (`face_adjacency_angles[dragon]` 0.47x twice), all
  level when run alone: arm timing depends on the allocator and cache state preceding modules leave.
  A row whose code did not change but whose *module* did is an order effect until an isolated probe
  says otherwise.

### 15.8 Probe-process contamination

**A loop-over-configurations probe is not the experiment the code under test runs.** One process
looping over cloud sizes and depths produced an apparent nondeterministic backend bug (same
configuration correct, then erroring, between succeeding neighbours); each configuration in a fresh
process was correct every time. **Re-run a failing configuration in a fresh process before
attributing a multi-config failure to the product**, and prefer the real tests to ad hoc
re-derivations. Conversely, a probe that *passes* in one process says nothing about a suite running
hundreds of allocations first, and holding several very large structures at once has produced a CUDA
allocator error that was the probe's own footprint.

### 15.9 Decide on the harness number, not the isolated one

`check_every=0` for the heat-family CG read 2.1-2.3x isolated and 1.05-1.23x in the harness
(0.94x on `saddle_graded`, §14.6). Same shape as §15.4's probe-versus-harness factor:
**decide on the harness number.**

### 15.10 `wp.timing_begin` is blind to graph-replayed kernels

**This invalidates the device/wall split for every function that graph-captures**, in the expensive
direction (a device-bound function reads host-bound, pointing at launch elimination when kernels
are the cost). Measured: 20 loose launches report 20 kernels; the same 20 replayed from a capture
report **zero**. **The reach is wide**: `warp.optim.linear`'s solvers capture their iteration by
default, so every CG solve (heat, parametrization, smoothing, `min_quad_with_fixed`, `solve_spd*`)
hides its dominant kernels.

- **Measure correctly by re-running with capture disabled** and reading `timing_begin` there: the
  kernel set is unchanged, so the device total transfers back (only the wall does not).
  `warp.optim.linear` takes `use_cuda_graph=False`; ordito's `wp.capture_while` sites fall back to
  direct execution when `wp.is_conditional_graph_supported` returns `False`, so monkeypatch that.
  Instrument both arms identically (§9); the uncaptured arm is the measurement, not the baseline.
- **When a device/wall split contradicts a tolerance or input-size sweep on the same function, trust
  the sweep**: it cannot be fooled by where kernels were issued from.
- **Second-order traps that make host time look larger**: a `.numpy()` readback's wall time is the
  queue depth in front of it (§14.6), and once enough launches are pending the driver's launch
  queue fills and `wp.launch` itself blocks, a device-bound signature that reads like marshalling
  cost.

### 15.11 Census the host calls by monkeypatching Warp, not by grepping

Per-call cost that depends on runtime types (a `wp.int32` or an `int`?) and on how often a line runs
is invisible to a static scan. Two monkeypatch censuses answer both without a benchmark; they
produced §13.1's dispatch table.

- **Census 1, Python-scope builtin dispatch**: patch `warp._src.context.Function.__call__`, walk out
  of `/warp/_src/` frames to the first ordito frame, count by `file:line`. Record the operator
  dunder (the first `/warp/_src/types.py` frame's `co_name`): `via=__mul__` is arithmetic on a
  Warp-typed value (a defect), `via=None` is an explicit `wp.length(...)` call (a judgement).
  Run under `pytest tests -q`.
- **Census 2, host-call counts by call site**: patch `wp.array.__getitem__`, `wp.array.numpy`,
  `wp.zeros`, `wp.empty`, `warp._src.utils.map`, attribute the same way. Suite-wide on CUDA:
  216 911 `wp.empty`, 52 816 slices, 44 755 readbacks, 34 433 `wp.zeros`, 6 566 `wp.map` (~2.4 s of
  a 68 s run at §13.1's prices).
- **Read the slope, not the count**: run at two iteration counts and difference the per-site counts
  to isolate per-iteration repeats (found `linalg._dot_finalize`'s 2.00 slices per CG iteration and
  `filter_implicit_fairing`'s 6 column views per pass, and refuted statically derived claims for
  `filter_taubin`, `filter_humphrey` and the blue-noise rounds).
- **Caveats, each of which produced a wrong reading**:
    - **A census only sees what runs: a two-device job** (§7.2); a slice inside
      `if not device.is_cuda:` is visible only on the CPU pass.
    - **It can misattribute a Warp-internal call to the ordito caller** (the frame walk stops at
      the first non-Warp frame; `wp.MarchingCubes.extract_*` internals are charged to the
      `ordito/levelset.py` line). Read the source line before believing a hit.
    - **Vector arithmetic is invisible to census 1** (§13.1): a clean census does not mean no
      Warp-typed arithmetic.
    - **A capture hides a loop from census 2 as it hides kernels from `wp.timing_begin`** (§15.10):
      with `check_every == 0` on CUDA, `_BatchedCg._iteration` runs once at record time, so counts
      are per *solve*. Take the slope on the path you mean to price.

---
## 16. ordito component status

Open defects, refuted plans, open leads and per-component constants, by area. **Check here before
opening work on a component.** What shipped is in the code; what is recorded here is what a reader
would otherwise re-derive. General platform and cost facts live in §12-§15 and are only pointed
to.

### 16.0 Method notes for component work

- **Take a census from the runtime, not the parser**, for any property Warp computes rather than
  the source spells. `[k for k, v in wp.get_module(name).kernels.items() if v.is_generic]` finds
  generic kernels that an AST scan of annotations misses (a factory whose `dtype` has a generic
  default).
- **A written decline is not a landed one**: grep for the sites that should have obeyed it.
- **A probe that instruments the thing it measures carries a do-nothing control arm** (§9);
  monkeypatching `wp.launch` added a frame to one arm and reported losses that were artifacts.
  A probe that fakes a readback fakes every readback that reaches it: gate it on a deterministic
  property of the result before reading its clock.
- **Gate a schedule change byte-for-byte on the CPU oracle** (§7.2) when tie-breaks are by index.
  CUDA differences that also appear baseline-against-baseline are float-atomic or atomic-cursor
  order, not the change. Build both arms from **one** set of inputs.
- **A whole-call A/B cannot resolve a region worth a fraction of a percent of the call** and has
  read the sign backwards repeatedly (§15.2). Verdicts that held came from deterministic counts
  and from timing the changed kernels alone. A row the change does not touch is the control that
  licenses the rest.
- **A count census prices the host and is blind to work that grows with the data.** A count-only
  change needs a clock at the largest mesh (a chain with fewer launches, allocations and
  readbacks measured 0.19x at `lucy` because it deduplicated every face, not the survivors).
  **Census allocation *sizes*, not counts**, when a few-launch call is host-slow at large meshes
  (§13.1, §16.9).
- **A cell that reads slower with untouched code is an order effect until an isolated run says
  otherwise**: re-run it alone, one pytest process per module (§15.7). Many sub-0.9x cells in
  multi-module A/Bs were level alone.
- **Measure at a fixed iteration count** for loop-with-convergence-break calls (`threshold=-inf`
  for ICP): a plateau's last-bit jitter stops the arms at different iterations.
- **A duplicated decision rule is the finding a de-duplication pass is for** (§2.4): two copies of
  "did this column step" disagreed at the round-off floor (`cg_close_round` vs
  `cg_step_scalars`) and the loop stayed alive; the fix is a call to one `@wp.func`.
- **A circular-import objection is answered by moving the helper to the lower module**
  (`sorted_run_start` moved from `grouping` to `array` so `csr_run_start` builds on it), not by
  keeping the copy.
- **Unread parameters are removed, not kept as documented no-ops.** A "Not read" parameter is a
  signature promising a lever that does not exist; the edge-table parameters of the `boundary`,
  `validation` and `adjacency` entry points, `submesh_from_face_indices(unique_indices=)` and
  the tests that pinned "passing it changes nothing" are gone. `is_volume(edges=)` and
  `edges_unique(edges_sorted=)` are read and stay.
- **A fusion that removes a launch is not free on the device for a BVH walk** (§14.2): a box
  formed in the kernel from the face corners was 4.5x slower than the identical walk reading a
  stored box (`face_aabb_bounds` launches first, 4 us). Time a new walk kernel against its
  predecessor's device time.
- **A shared helper that returns trimmed views is not free to callers that pass `n`** (~14 us of
  view construction): such callers launch the underlying kernel themselves.
- **Turning a loop-bound `if` into `wp.where` can move CPU results by ~2e-7**
  (`accumulate_radius_frame`): the CPU oracle is sensitive to control-flow shape.
- **Declined on speed: `halfedge.opposite_edge` for the `(e + 1) % 3` reads in the pattern-key
  kernels.** `e` is an unrolled literal there and folds at compile time; the helper computes the
  modulo at runtime in device-bound kernels.
- **Bulk edits**: `ast.parse` every changed file after a bulk docstring edit (a rewrap folded
  function bodies into docstrings); name files explicitly in `ruff format`; give concurrent
  probe helpers unique names and assert the *submodule's* `__file__` (the lazy `__init__` makes
  `ordito.__file__` meaningless); `pkill -f <probe>` kills the shell running it (§11); re-run
  every gate against the final tree after an interrupted pass.

### 16.1 Where ordito's time goes (losses, floors, wins)

**The whole mid-level surface is host-bound.** 44 public functions timed over a 256x face range
came out flat within 1.25x and none above 3x. Two consequences:

- **There is usually no kernel to make faster.** The currency is the *count* of Warp API calls;
  §13.1 is the price list. Micro-optimising wrapper Python is not a lever; removing Warp calls
  is.
- **Flatness across mesh size is the measurement to take first**, before any profiler: two
  timings, no instrumentation, immune to graph capture (§15.10).

Most of the biggest losses are 92-99 % host-side launch and allocation cost, and almost
everything in the 0.3-20 ms band is 66-97 % host at high launch counts. **Tiling cannot touch a
launch floor; only launch elimination can.** Device-bound exceptions (`ambient_occlusion`,
`lscm`, `query_nearest_bvh_k1`) are so because the grid is too narrow to fill the machine.

!!! warning "Only valid for a function that does **not** graph-capture"
    `wp.timing_begin` reports zero kernels for graph-replayed work (§15.10) and
    `warp.optim.linear` captures by default, so any CG-backed or `wp.capture_while` function
    reads ~100 % host whatever it is. Check for capture first. Uncaptured truth: the
    quadric-decimation pass was 21.6 ms device of 43.8, not "99 % host"; hole fill 5.2 of 6.6.

- **Change ratio tracks how host-bound the call was** (§15.2): pipelines collapsing to a few
  launches win 2-4x, mid-level calls 1.2-1.6x, device-bound calls (Chamfer, Hausdorff,
  `max_tangent_sphere`, `mesh_to_mesh_distance`) stay 1.00-1.02x even when their launch count
  falls by a third.
- **A row attributed "launch-bound" may be device-bound.** `transport_tangent_vectors` /
  `vector_heat_scale` are 71 % device (two `TiledDot` kernels are 42 % of the solve, §12.7);
  `delaunay_triangulation` is the one genuinely host row, its host half being the designed
  single-thread CPU seed. `linalg.assemble_interior_system` spends its device time in one
  `_bsr_accumulate_triplet_values` because it routes an already-sorted CSR through
  `bsr_from_triplets`.
- **The 0.3-5 ms band is mostly closed or floor**: per-segment floors (`split_array`,
  `concatenate_arrays`, `pack_1d_arrays`, §13.1), `is_watertight` (scope mismatch),
  `cluster_decimate`, `remove_degree3_vertices`, `lscm`. What remains at the top is list
  returns, device-bound traversals and settle solves bound by their round count.
- **Before optimizing a traversal, read what its caller unpacks.** `homology_generators`' BFS
  ran four kernels a level; its one consumer used `order` for `.shape[0]` only. An atomic-min
  parent tree keeps it deterministic, halves the level and drops the column-sorted-adjacency
  requirement; the basis differs from the FIFO tree (total loop length ~+10 % at genus 64 for
  the same generator count). `graph.bfs`, `bfs_from_edges` and `bfs_multi_source` were deleted
  (no consumer; `kernels/algorithms/bfs.py` keeps `per_source_bfs_collect`). Deleting an entry
  point is the honest close, not a win.
- **A published attribution can outlive its fix**: `combine.split`'s "needs batching" is stale;
  it issues the same launch count as `split_batched` and what remains is the per-view cost of
  the returned arrays, which is the return value.
- **Where ordito wins big (context for reading a loss)**: `screened_poisson` 15-25x over
  open3d, `combine.split` on few-component meshes 34-120x, `winding_number` 13-18x over igl,
  `cluster_decimate` 9x at `dragon`, `polyline_simplify` two orders at `rim_long`, `procrustes`
  ~7x over trimesh (launch-latency bound; only even with open3d). Small-input fixed overhead of
  reductions and scans costs tens of us; large meshes win 10-260x. A third implementation makes
  an outlier legible: trimesh alone is slow enough that a 3x regression still looks like a win.

### 16.2 `import ordito`

`ordito/__init__.py` resolves each submodule (and `Trimesh`, a class) through a PEP 562
`__getattr__` and caches it. Cause: `@wp.kernel` builds an `Adjoint` at import time for every
decorated kernel and importing a submodule imports the parent package first (§12.6).

**The guarding test runs `import ordito` in a subprocess and asserts zero kernel modules are
pulled in** — in-process the answer is always "all of them". Do not simplify it. The laziness
deliberately changes nothing else: overload registration still runs before the first launch
through its module, and nothing calls `wp.load_module` / `wp.force_load` at import.

### 16.3 `reconstruction`

- **`screened_poisson`'s `dense` solve is over a `2^depth`-cubed node grid whatever the cloud
  size**, so each level costs ~8x and the CPU test depth is one lower than CUDA's. **Error is not
  monotone in depth** (past some depth the octree resolves sampling noise), so never assert
  "finer depth reduces error".
- **`screened_poisson` drops zero-area triangles before orienting** (`remove_degenerate_faces`;
  `point_weight=0.0` used to emit 27-64 a run, §7.7). Deliberately not the full
  `_clean_reconstruction` tail (welding manufactures non-manifold edges; dedup would move the
  default path's output); pinned by `test_reconstruction.py`'s `point_weight=0` arm.
- **Dense solve preconditioner: a geometric multigrid V-cycle** (`_PoissonMultigrid`, kernels
  `poisson_mg_*`). Nested node grids; level `l`'s operator is `2 ** l * L_l + screen * (P^T)^l
  W` (the 7-point rediscretization of the Galerkin product; its smooth-mode ratio to `P^T L P`
  tends to 2 in 3-D, 1.57 already at 9^3); restriction `P^T`; one damped-Jacobi sweep at 6/7 a
  side, four on the 3^3 coarsest grid; `float32`, matrix-free, symmetric. **13-16 iterations to
  `1e-5` at every level**, true residual following (6.1e-6 at 257^3, below the Jacobi floor).
  Two sweeps, damping 2/3 or eight coarsest sweeps each cost 5-15 % more for no fewer
  iterations. The zero-start pre-smoothing sweep rides in the launch that produced `b`; a cycle
  is four launches a level.
- **`solver_tolerance` defaults to `1e-5`**, above both backends' `float32` floors (multigrid
  path ~2-6e-6; Jacobi 4.8e-6 / 1.0e-5 / 2.2e-5 / 4.0e-5 at 33^3 / 65^3 / 129^3 / 257^3;
  `warp.fem` system 1.2e-6). The old cap of `solver_iterations = 100` at `1e-6` left the dense
  fine levels two orders short (5.0e-3 at 257^3) and the surface 1.0e-3 of the bbox diagonal
  off the converged one. **Non-convergence warns on both backends and devices** (`Warns`
  section): the dense path reads its iteration count once per level (residual only on a level
  that used its whole budget); the adaptive path calls `solve_spd` at its host-scalar cadence.
  `test_poisson_dense_solve_converges_in_few_iterations` fails if the V-cycle is swapped for
  Jacobi. `solve_spd(check_every=0)` does not warn on CPU (documented contract).
- Solver kernels of the dense and adaptive backends: §16.16. `method="adaptive"` runs Warp's own
  `warp.fem` mat-vec.
- **`ball_pivoting` design choices counter to the obvious guess — do not re-try them.**
  `wp.capture_while` is *slower* than a batched host loop for this loop shape (§14.3; 8 waves
  per readback); **the hash-grid cell width must be the *ball* radius, not the wider pivot
  neighbourhood** (a wider cell makes the far more frequent empty-ball query enumerate many more
  points than it needs); the persistent front is what fixed the old watertightness limitation,
  so the "overlapping sheets" caveat belongs to the per-wave rebuild and must not be reinstated.
  The pivot search uses the tiled BVH walk; the empty-ball test stays a per-lane serial
  hash-grid query (§12.8, §14.2). Graph-capturing the 8-wave batch is declined (likely
  device-bound; needs a device/wall split first).
    - **Deterministic order**: nondeterminism was integer arrival order (an atomic slot reused as
      *priority*). A globally fixed key alone *starves* the front (the same proposals win the
      same contested vertices each wave), so the key is salted with a hash of the wave counter.
      No uniformly sampled closed sphere can test this (exact Euler triangulation leaves wave
      order nothing to decide); only an irregular-spacing fixture exposes it.
    - **The default `radius=0` auto-radius is nondeterministic**; the reproducibility claim is
      scoped to an explicit `radius=`. Do not "fix" the reduction without a caller that needs it.
    - **Even with an explicit `radius=`, the CUDA face buffer is not reproducible on `bunny`**:
      650-2 500 of 211 038 entries differ run to run (face count stable), measured 2026-10-01 on
      the unchanged tree. `bunny_decimated` reproduces. Gate a ball-pivoting A/B on the CPU device.
    - **The seed kernel must not re-prove failures**: an attempt's outcome depends only on the
      stored neighbour list (`ball_is_empty` never reads `point_used`), so a point whose failed
      walk saw its *whole* neighbourhood can never succeed and returns at once
      (`SEED_EXHAUSTED`); a truncated walk is not remembered (used neighbours let the next walk
      past the break). A deferral cache was sound and bought nothing.
    - The mean spacing for the auto radius folds sum and count in one `float64` pair pass
      (`_mean_positive_finite`).
    - **An intermittent `CUDA error 700` with clean memory tools was a host-side front-buffer
      swap desync**: when the memory tools come back clean on a memory-shaped symptom, instrument
      the control flow instead.
- **`delaunay_triangulation`'s seed keeps the hull in a ring** (2026-10-03,
  `kernels/reconstruction.lexicographic_triangulation`): one pass per insertion computes every
  boundary orientation once (exactness rests on evaluating all of them: a degenerate visible set
  is decided exactly as the array sweep decided it), emits and finds the arc in the same pass, and
  the kept arc becomes the boundary by moving the ring's start, copying only when it wraps past
  logical 0 (rare: the visible arc surrounds the last insertion). The lex order is a device radix
  sort of a packed sortable key (`array.ordered_float_bits`, `-0.0` onto `+0.0`, `NaN` above
  `+inf`), `numpy.lexsort`'s order exactly. Seed 6.8 -> 4.4 ms at 20 000 points, call 1.50x (1.12x
  at 2 000); byte-identical on both devices over random, grid, duplicate, collinear, signed-zero
  and cocircular inputs. Not proposed: an O(1) linked-list hull walk (it cannot evaluate every
  orientation, so degenerate inputs could differ).
- **`triangulate_point_cloud`**: the candidate grouping is one stable sort emitting runs of 2-3
  in key order, which *is* `resolve_duplicated_faces`' output order, so
  `_clean_reconstruction(deduplicate=False)` skips that stage (every face reaching it has a
  distinct vertex set).
- **`warp.fem` gotchas, each a plausible wrong answer rather than an error**: an `ImplicitField`
  func must have no return annotation; `allocate_by_voxels` is voxel-*centered* (the extraction
  lattice needs a half-voxel translation or a spurious surface component appears); solve in
  *index space* so the screening-versus-stiffness balance matches the dense calibration; a
  point-source weak form rings when cells are much smaller than the sample spacing (cap the grid
  depth rather than rounding it); `warp.fem` owns its launch shape, so it cannot take a block
  walk (§14.2).
- **Slab-chunked marching cubes is not viable**: `wp.MarchingCubes` is crack-free only *within*
  one grid, so welding independently computed slabs leaves non-manifold seams.
- **Where a dense solve's time goes (bunny, Warp 1.18, 2026-10-06)**: depth 9 is 228-250 ms, of
  which the 513³ level (splat, setup, MG-PCG) is 185 ms, levels 5-8 31 ms and the extraction
  27 ms; device time 203 ms, bandwidth-bound at ~12 ms a finest round, 12-16 rounds a level.
  Capping every level's rounds is a cascadic schedule with no code change: 4 rounds 2.15x at
  0.014 cells mean surface error against the converged output, `solver_tolerance=1e-3` 1.39x at
  0.0005 -- against a discretization scale of 0.13 cells (depth 9 vs 8). Extraction is now
  `levelset.marching_cubes`' port (§16.5), 1.03-1.04x on the whole call at depths 7-9 (the
  standalone 2.9x is mostly allocation the warm pool already hides here).
- **A narrow-band brick hierarchy was prototyped, not adopted** (awaiting a decision). Finest
  data only, every coarser level's right-hand side and screening its `Pᵀ` restriction (the MG
  operator `2^(L-l) L_l + s W_l`), a dense depth-7 base, 8³ bricks near the samples above it,
  Dirichlet ghosts `P x_coarse`: 10-19x at depth 9 (bunny, dragon), 3.5-5x at depth 8, and depth
  10 runs (80 ms on dragon; dense is out of memory there). Error 0.01-0.04 cells against dense,
  shrinking with band width, not rounds. **Independently splatted coarse levels cannot supply the
  Dirichlet values**: the converged depth-8 and depth-9 dense solutions are not one scale apart
  (ratio 0.5-0.67 near the surface, 0.33 far, screening is not level-normalized), which put 0.15
  cells of error in a band that grew with more rounds.
- **OPEN DEFECT: the default dense output is not watertight at depth 9.** The extraction is
  watertight; `remove_degenerate_faces` then opens 80-110 boundary edges on bunny,
  bunny_decimated and dragon (depth 8 bunny: 0), contrary to the docstring's "the default
  `point_weight` emits no degenerate face". The level set grazes lattice nodes often enough at
  513³ to emit zero-area triangles.
- **OPEN DEFECT: `method="adaptive"` on `dragon` at depth 9 dies with CUDA error 700** inside
  its CG (`linalg._row_path`'s read), reproduced on HEAD in a fresh process (Warp 1.18,
  2026-10-06). Depth 7 is fine. The adaptive backend's time at depth 9 on bunny (400 ms) is
  `fem.adaptive_nanogrid_from_field` 182 ms, two `fem.integrate` 62 ms, the CG 39 ms, the lattice
  resample 17 ms and the extraction 27 ms.

### 16.4 `remesh`, `repair`, `creation`, `bounds`

#### Tests and fixtures

- **A green parity test is evidence only if its fixture can express the difference** (§7.4). The
  remesh concentration test ran on a uniform icosphere where area weights equal uniform ones; it
  is now parametrized over a uniform sphere **and** a graded patch, and disabling any of split /
  collapse / smooth fails it. A graded regular grid is the pathological remesh input
  (valence-perfect, Delaunay, a fixed point of the unweighted Laplacian), so three of five stages
  are blind to its anisotropy.
- **A tangential step to a convex combination of the one ring cannot fold a *convex* link**, so
  the smooth pass's fold veto needs a deliberately non-convex link to be reached. The weighting
  is pinned by a hand-computable NumPy oracle.
- **A collapse anti-oscillation test must walk *both* rings** when the placement is free.
- **An independent-set kernel needs a test on how many winners a round produces**, not only on
  the validity of the ones it commits (locking on the raw edge index is spatially monotone:
  nearly every round had one winner among tens of thousands).
- **`isotropic_remesh` is not byte-gateable** (atomic-order drift in normal and ring
  accumulation tips split/collapse decisions, even the face count). Gate the exactly
  reproducible contained stage (`_valence_flip_pass`) instead.
- **Edge-length equilibrium is ~1.0x target only for a "nice" ratio of the input edge**
  (midpoint-split quantization); coarser-than-input targets plateau lower.
- **`isotropic_remesh` hands its edge groupings between stages, and a collapse pass reads back
  once** (2026-10-02, R28-5). The split pass that finds nothing to split grouped the faces it
  returns (`_subdivide_to_size`'s fourth return), which the first collapse pass takes; a flip pass
  that flipped regroups from its own `_FlipTopology` sort. A collapse pass issues its compaction
  before reading the commit count, so one copy of a three-word buffer carries the stop decision
  and both sizes (a pass that committed nothing discards a compaction of an identity remap).
  Together 1.10-1.12x `saddle`, 1.06-1.10x `saddle_graded`, level `hemisphere`; CPU
  byte-identical over four stage-toggle arms. Stage profile (`saddle`, three iterations, synced):
  the 15 collapse passes are 8.1 of 14.2 ms, ~60 Warp calls each. **Not built, priced:** patching
  the edge table across passes (the per-pass regroup is ~0.11 ms of ~8 launches; a patch still
  sorts or merges the changed keys at these sizes, so it saves launches only at meshes far above
  the benchmark's); fusing the ring and vertex-face CSR builds into one count/scan/fill (~5 calls
  a pass, ~3.5 % of the call, at the price of a remesh-private copy of two shared builders).

#### Collapse, independent sets, valence flips

- **The isotropic collapse and the quadric collapse share one fold veto** (a duplicated decision
  rule diverging was exactly that gap). The adjacency build the veto needs is free: a rejected
  collapse removes downstream work that pays for it.
- **A min-key parallel independent set needs a key that is spatially incoherent *and*
  injective.** A hash alone lets two candidates collide and both commit, corrupting the mesh; the
  key packs the hash and the raw index into separate halves of a 64-bit key. A guard bolted onto
  one of several callers is the tell that the key is wrong.
- **`_valence_flip_pass` reads valence off the sorted key buffer its own edge rebuild produced.**
  Trap: the natural "runs of exactly two" marker undercounts (flags manifold-interior edges only,
  dropping boundary edges); the marker matching `edges_unique`'s row set is the any-length run
  start.
- **`_collapse_pass` builds rings with `edges_to_neighbor_lists` (no sort)** because every
  consumer counts, exits early or takes a row minimum: check the *consumers* before dropping a
  row order.
- **A count the producing kernel already knows is a conditional `wp.atomic_add`**, not a
  reduction over its output mask (§13.2).
- **Hoisting per-pass scratch is a loss when the loop usually runs once**: hoist against the
  *expected* pass count.
- **A whole-buffer `fill_` / `zero_` whose length equals the `dim` of an adjacent launch that
  does not read it belongs inside that launch.** Counter-case: `sample._dart_throw_blue_noise`,
  whose buffers are cell-sized while its launch is over a shrinking alive count.
- **Quadric decimation (`quadric_decimate`)**: the pass is flat in mesh size, so replaying every
  pass at the pass-0 width costs ~1.01x; a pass is re-recorded at the live width once it has
  halved and freed 250 000 faces of width (a loss on small meshes, hence the gate). A graph
  replay of many small nodes is not free device time (~3 us gap per node: 0.88 ms replayed vs
  ~0.58 ms of kernel time over 109 nodes), so **the lever is fewer nodes (fusion), not fewer
  syncs**. A winner's rank in the cost order is read off two bitmasks (popcount plus a word
  scan), exact against the old stable sort of `winner ? cost : +inf` including `inf` / `NaN`;
  six per-round radix sorts were half of a replay's busy time. Scans inside the pass are a
  private allocation-free chunked scan (`remesh._ExclusiveScan`), which a `capture_while` body
  needs and `wp.utils.array_scan` is not (§14.3). **Declined: `capture_while` for the pass**
  (42 replays 36.7 ms back to back vs 37.3 ms with per-pass readbacks: ~2 %); its
  allocations are dozens of buffers, not only scan scratch.
- **`remove_degree3_vertices`**: an interior degree-3 vertex is three faces whose opposite edges
  close a directed 3-cycle, and the cycle *is* the replacement; a telescoping `x - y` sum over
  the opposite edges is an exact "closed link" flag (0 for any closed link, non-zero for one
  boundary fan, also mod 2^32). Selection and emit fuse; pinched input is accepted (never a
  candidate) rather than raising. Validation runs inside pass 0. Replacements are appended to one
  fixed-capacity face buffer (`n + n // 2` rows; each removal nets two faces fewer), so per-pass
  compaction and the tail `remove_unreferenced_vertices` are one final launch. The kernel counts
  the rim vertices it turns into candidates and a zero stops the loop. **The MIS rewrite is
  declined** (degree-3 candidates share no edge on any benchmark mesh, so pass 0 takes them
  all). **Recording the passes lost** (0.86x at 2 passes, 0.87-0.96x at 3; §16.14's ~2-replay
  break-even). The hash-only count is declined.
- **Flip loop keeps its adjacency instead of regrouping.** Only the first *flipping* round
  regroups (a key launch, a radix sort, a mark, a scan, an emit); that regroup also builds the
  incremental state later (recorded) rounds run on. Three things make it byte-identical:
    - **Claims rank candidates by edge key, not row index** (`kernel_remesh.flip_priority`,
      `uint64` `atomic_min`). A regroup emitted rows in ascending key order, so the smallest row
      index *was* the smallest key; winners are the same whatever the row order.
    - **Rows are rebuilt from maintained halfedges, not patched.** `commit_flips` keeps
      halfedge -> row and row -> halfedges maps current; `refresh_flip_rows` rewrites every row
      through the same `write_flip_row` the emit uses. Patching a row in place races (a side
      edge's two faces can belong to two committing flips). Moving a halfedge is race-free: each
      flip writes only the column holding its own face's halfedge. `valence` rides the commit.
      The claim tables are re-armed by `refresh_flip_rows`.
    - **The duplicate-edge guard reads a hashed key set** (`grouping.hash_find_or_insert`,
      `key_set_remove` tombstone) once the sort is stale, the sorted keys before
      (`kernel_remesh.edge_key_exists`, warp-uniform selector). Tombstones accumulate;
      `_FlipTopology.flipped` rebuilds once the flips since the last build could fill the table
      past three quarters. `edges_unique` re-sorts only.
  **An incremental structure that replaces a per-round rebuild should be built by the rebuild
  it replaces, the first time one is needed**, not up front: the eager version cost short calls
  (`subdivide_region_to_size` 0.86-0.93x: +3 launches, +11 allocations). An issued round
  refreshes only if it flipped; inactive state is passed as `None` / the sort buffer.
  `_FLIP_CAPTURE_FROM_ROUND = 1`: round 0 is issued and the graph recorded from round 1
  (recording costs more than an issued round; starting at round 3 gave 0.89-0.92x on
  2-4-round calls; round 0 vs 1 differ 0.94-1.03x elsewhere). Flip rounds replay as a **plain
  graph** (a scan allocates, §14.3). The interior-edge count cannot change after round 0, so
  later regroups skip its readback. `flip_priority` keys and the sort use `end_bit` (§13.1).
    - **REFUTED: the flip loop under `wp.capture_while`** (device-side running total and
      tombstone budget): readbacks fell 102 -> 4 on a 100-round call, but a conditional round
      costs more than a host-replayed one and recording costs a couple of dozen reads:
      `delaunay_triangulation` 0.86-0.97x, `remove_t_vertices[saddle_graded]` 0.86x; only a
      100-round few-flip call gained (1.08-1.13x).
    - `icosphere(4)` jittered by 0.3 of an edge oscillates at one flip a round to the iteration
      cap (pre-existing cycle). The incremental-state test uses `icosphere(3)` at that jitter
      (79, 28, 9, 1 flips past the plain round), the smallest fixture that reaches the
      incremental rounds (at jitter 0.02 every flip is in the plain round and a no-op `refresh`
      passes).
    - A flip round's cost is device: a 64-bit radix sort of every edge key ~86 us (68 us at
      32-bit keys, only below 65 536 vertices: ~1.5 % of the call, not taken) and
      `delone_flip_candidates` ~59 us (its `float64` Delone predicate; narrowing it changes
      which edges flip; not taken). `subdivide_region_to_size`: 4.1 of 6.3 ms is its three flip
      passes at ~0.24 ms a replayed round; per-pass region-edge scatter and long-edge test are
      one launch (`mark_long_region_edges`); `_FlipTopology.edges_unique()` reads the keys off
      the flip sort instead of a whole-mesh `edges_unique`.
- **`remove_degree3_vertices` reads no output count at the end** (2026-10-03): pass 0 counts the
  input's referenced vertices (`record_degree3_halfedge` returns its arrival slot) and each
  removal unreferences exactly its centre, so both output sizes are known; the state views come
  from one `array.split`. 1.11x `bunny`, flat at `dragon` / `happy_buddha`; CPU byte-identical
  (CUDA's face order varies run to run on the unchanged tree too). `boundary_edges` /
  `oriented_boundary_edges` pass the vertex count as the key radix (`end_bit` sort): 1.22-1.35x.
- **A fast path is checked against the general engine, not asserted, and the fallback must be
  measured live.** `_revolve_regular` returns `None` when the surviving triangles are not the
  regular pattern (10 of 27 probed configurations). Its profile cap
  (`_REVOLVE_REGULAR_MAX_PROFILE`) is set by an asymmetry: the win when the screen succeeds is
  flat, the loss when it declines grows with the profile. On a decline the template is computed
  twice; threading it through `revolve` is not worth it (its signature may not name
  `np.ndarray`, §3.8). **Specialising a general routine means calling its *predicate***:
  `cone` / `cylinder` / `revolve_uniform` ask `_revolve_kept_template` which triangles survive,
  keeping byte-identity at degenerate section counts and its absolute area tolerance.
- **`creation.parametric_surface`'s lattice has a size gate** (`_PARAMETRIC_LATTICE_DEVICE_FROM
  = 625` samples: host and device tie at 400, device leads from 900) with both implementations
  kept. Every resolution the suite uses is below the old gate, so its test monkeypatches the
  threshold both ways and compares bit-for-bit. Two orderings are easy to get wrong, called out
  at the kernel: the u-seam re-canonicalisation after a v-twist is **unmasked**, and all four
  pole masks are snapshots of the post-wrap state taken *before* any collapse; the pole anchor is
  statically derivable.
- **`extrude_polygon` / `sweep_polygon` wall a full `n - 2` triangulation from the ring edges**
  (the derived boundary is the fallback and still raises the non-simple-ring `ValueError`).
- **`creation.icosphere` computes each vertex by descending the refinement** from its base face
  (integer barycentric weights `(w0, w1, w2)` over the triangle size `s`; a corner child keeps
  `w_k - s/2`, the centre child takes `s/2 - w_k`; midpoints lerp from the lower global index),
  bit-identical to the per-level build on both devices. One launch for every level is O(N L): 1.8-3.0x
  up to six levels and **0.30x at nine**. Shipped: six levels in the first launch, one launch per
  level after (`_ICOSPHERE_LEVELS_PER_LAUNCH = 6`): 1.8-3.0x through six levels, 1.53x at eight,
  0.97x at nine (10 M faces, device-bound). Five or seven in the first launch lose at seven to nine.
  A thread whose target is a side's midpoint computes only that edge vertex (0.94x -> 0.97x at nine).
- **`creation.sphere_cap`'s ring inverse**: a thread recovers its ring from its vertex index by
  solving `3 r^2 - 3 r + 1 <= v`; ring starts are `1 + 3 r (r - 1)` **except ring 0**, the lone
  apex at slot 0. Getting it wrong wound every apex triangle around its neighbour and only a
  byte-identity gate saw it.
- **`shorten_loop_with_offsets` replays recorded sweeps** (2026-10-02, `geodesic_walk._LoopSweep`):
  the loops live in fixed buffers with headroom (`_SHORTEN_LOOP_GROWTH = 0.5`, at least
  `_SHORTEN_LOOP_SLACK = 1024`), the loop state in eight device words
  (`kernels/geodesic_walk.SHORTEN_*`: parity, stop, consecutive-unchanged, accepted, overflow,
  length, any-change, cap), the owning loop is a binary search of the offsets (no owner-label
  pass), and a sweep is six launches plus two scans with no readback. Sweep 0 is issued, then
  `_SHORTEN_LOOP_GRAPH_SWEEPS = 4` are recorded and replayed with one state read per replay. A
  sweep whose rewrite would not fit is undone (its rewrite and compaction skip, its count is not
  taken) and re-run in buffers twice the needed size (`test_shorten_loop_regrows_its_buffers_and_
  matches` forces it with zero headroom). Byte-identical loops and sweep counts on both devices
  (`handles_1` / `handles_64` / `tangle_torus_small`, 5 / 20 / 52 sweeps); `handles_64` 3.85 ->
  1.42 ms (2.7x), `handles_1` 1.13x; `remove_tunnels[handles_64]` 12.9 -> 10.3 ms (1.26x).
- **`remove_tunnels`**: vertex-disjoint non-trivial loops can still be *dependent* (23 disjoint
  generators on `handles_64` bound a piece of the slab and cutting all split it). Fix: label the
  faces of the cut mesh (`kernels/repair.sever_barrier_pairs`, a binary search of the host-sorted
  loop keys in the same launch); every component after the first is a dependent loop, and a
  longest-first spanning forest over the component graph leaves exactly those uncut. The unit
  test carries the genus-64 slab itself (no smaller variant produces a dependent family). The
  cost was a host scan (`_independent_loops` found each loop's sides by a NumPy scan of the
  whole face array per loop) and a split -> pack round trip with per-loop readbacks:
  `kernels/polyline.packed_closed_loop_lengths` measures every loop in one launch
  **bit-identically to `polyline_length`** (one block per loop, same lane stride and
  `wp.tile_sum` fold). Exactness is the constraint: the sort keeps equal lengths in basis order
  and the regular fixture has many ties. `_cycle_length`'s per-loop readback stays (a host length
  would reorder exact ties).
- **A region-sized version of a region question**: a kept rim edge is an input boundary edge
  exactly when no *deleted* face contains it, so `delete_region_keep_boundary` uses the deleted-
  face rule (1.15-1.17x). It needs `boundary_loops_batched` to walk pinched rims as a successor
  graph over boundary halfedges (`kernels/halfedge.next_boundary_halfedge`), after which every
  consecutive pair any walk returns is a boundary edge on *any* mesh (twins are true opposites
  or `-1`; a rotation enters a face only through the twin of its incoming halfedge, so rotations
  never merge and `successor_cycles` never gets colliding input). A mesh that is not
  edge-manifold can *lose* a loop, never invent an edge
  (`test_boundary_loops_never_invent_an_edge`). **A runtime guard for this was built, bit and
  measured flat (0.97-1.04x); when a runtime check costs the win it protects and a proof covers
  it, move it to a test.** An empty deletion exits early from the kept face count with no
  readback. The unit test's "interior" region had itself been pinched. `holes._EdgeTable`'s
  inversion (sort the rim's keys, one face pass probes them) is exact and never slower.
  Not taken: `delete_region_keep_boundary`'s four readbacks have launches between every pair.
- **`fix_self_intersections(local)`**: per-round dilation is two face-mask hops (no
  `edges_unique` rebuild). Stage profile: `refine_and_smooth_region` is 18.6 of 23.8 ms, of which
  `_solve_region_smooth` 8.1 ms and `subdivide_region_to_size` 6.7. **The single-use captured CG
  there is not waste**: forcing the host-check path is 1.7-2.2x slower at every cadence.
  `refine_and_smooth_region` builds one `edges_unique` for the boundary mask, both regions and
  `exclude_fully_selected_components`. `fix_self_intersections` is not bit-reproducible on CUDA
  (float atomics).
- **`repair` details**: `make_winding_consistent` takes its flips from
  `validation.face_flip_mask` (§16.11). **Each connected component keeps its lowest-indexed
  face's winding** (the parity union-find's root is the component's smallest face id at parity 0),
  so on orientable input the result is a deterministic function of `faces` on both devices,
  documented as the contract and pinned by
  `test_make_winding_consistent_keeps_each_components_lowest_face`; only a non-orientable input
  varies run to run on CUDA (§16.11). `remove_degenerate_and_non_manifold_faces` filters in the input's numbering
  and compacts once. Declined: reading the orientation sign off halfedge origins instead of
  `pair_flip_sign`'s corner search (they disagree on a face with a repeated vertex, changing
  `is_orientable`). `_vertex_scale_attribute` lifts the region faces' keys by `base**2` so one
  `end_bit` sort puts the outside edges first in `edges_unique` order; the empty-outside fallback
  is free (the smallest key is already lifted).

#### bounds

- **A host helper that reproduces a kernel's arithmetic is a duplicated decision rule across the
  host/device boundary.** `_spiral_frames` reproduced the candidate-axis spiral in NumPy and was
  never bit-identical (the host narrowed after building the matrix, the kernel narrows the
  quaternion first), so refinement descended from a frame fractionally different from the scored
  one.
- **A search whose answer can tie is tested on the *achieved objective*, not the returned frame**
  (two of 105 oriented-box configurations pick a different frame under a 1-ulp reduction
  difference: a rotationally symmetric needle has three candidates at the exact minimum).
- **A guard whose only failure mode is an out-of-bounds read needs the check, not a test**: the
  oriented-box chain padding is not gated by any value test, so the kernel clamps the index at
  the read (§12.1).
- **Declined: a global farthest-first oriented-box seed** (less than half the faithful walk's
  cost, better volumes on average, worse on one shape by enough to eat half the margin
  `test_oriented_bounding_box_refinement_is_monotone` documents); needs its own evaluation
  across `MESHES`.
- **REFUTED: reducing a concatenation of two clouds instead of seeding one accumulator twice**
  (`bounds.enclosing_diagonal`): a loss at every size from 1 000 to 4 000 000 points (the union
  is an allocation and a copy to remove one flat launch). The negated upper corner in that
  reduction exists so one `wp.full(6, inf)` seeds both ends and every update is an `atomic_min`;
  unpacking it would need a two-value seed (§3.3). Both launches take `dim` from
  `kernel_reduce.chunks_1d` (for the *unfolded* `TILE_1D` kernels; `blocks_1d` is wrong by a
  factor of `TILES_PER_BLOCK_1D`).

#### Host vs device helpers, general

- **A numpy reduction over a short trailing axis is a pathological shape**: `sides.prod(axis=1)`
  on `(n, 3)` is an order of magnitude dearer than `s[:,0]*s[:,1]*s[:,2]`, and a strided
  difference on a stride-24 view ten times its contiguous cost. Check before pricing any host
  table op by element count.
- **Producer-then-reduce has three mechanisms**: fold the reduction into its producer; express a
  *closed* polyline by **wrapping the index** rather than a whole-buffer copy (once the reduction
  is fused that copy *is* the call); and where a readback must stay because a Python loop
  branches on it, drop the reduction around it. ~33 sites match an `ast` scan; most spend 1-4 %
  of the call and are declined. A lane-strided fold plus `wp.tile_sum` is not the sequential
  sum `reduce.sum` performs below `TILE_1D` (the more accurate arm, but it breaks a test asserting
  bit-identity with a sequential reference: check for an exact comparison before fusing). A scan
  keyed on line proximity over-counts (mutually exclusive branches).
- **`edges_unique` from faces is one radix sort of the corner keys** (2026-10-02): the keys and
  the identity payload written into the sort's buffers (`adjacency.face_edge_keys_and_order`), a
  run-start mark scanned in the payload's upper half, and `kernels/edges.emit_sorted_unique_edges`
  (shared with `remesh._FlipTopology.edges_unique`) writing rows and the corner map -- the
  identical ascending-key output. 1.3-1.5x at 16-69 k faces, 2.0x at 0.87 M, 2.85x at 28 M (59 ->
  21 ms; the ~2n-slot hash table was gigabytes of random atomics there).
- **`edges_unique` is a host-bound substrate under ~57 call sites.** Deduplicated rows are
  recoverable from packed keys (`array.unpack_edge_key`); `validate` defaults `True` with every
  internal caller passing `False`; `array.index_bound` takes both ends from one `reduce.minmax`
  via `require_non_negative=`. A grep for `validate=False` on one line undercounts (half the
  converted sites wrap across lines).
- **A two-column index row needs no bound**: `hash_indices_rows` packs `sum(digit[i] * radix **
  i)`, so any radix above every entry is injective, and every `int32` reinterpreted as `uint32`
  fits two columns in a `uint64` (`constants.INDEX_RADIX_PAIR`); packing stays monotone
  lexicographic. Ask what else a bound is before widening it: in `validation.is_vertex_manifold`
  it is the *length* of the per-vertex mask.
- **A heavy per-element integrand wants a chunk width chosen on the host**:
  `measures._moment_chunk_faces` doubles the chunk until the grid is no wider than 1 280 blocks
  (a runtime width measures identical to a `wp.constant` one).
- **A function returning several small device values writes them into one buffer and reads it
  once** (§3.10).
- **A narrow fold's width is rows per block when the fold commits `float64` to constant slots**
  (`points.neighbor_distance_moments`): one row per lane was 1.4x the unfused pair at 1 M rows
  (15 625 blocks contending on two slots), 1 024 rows starved the grid at 41 k, 256 won at both
  (§13.2).

### 16.5 `array`, `graph`, `polyline`, `intersection`, `boundary`, `levelset`

#### array, graph

- **The three searches in `kernels/array.py` are not interchangeable and picking wrong is
  silent** (§3.1): a wrong search on an argsort payload returns a valid index of the wrong
  element (seeds land on neighbours, a flood fill returns plausibly too much). When a binary
  search feeds an argsort payload, verify one lookup by hand against NumPy. When a fix does not
  change a symptom, suspect two causes.
- **`side="right"` is poisoned by `NaN`** (§12.4): `unique_1d(return_inverse=True)` searches
  `side="left"` and falls back to the last slot only on the `NaN` branch. Registering the float
  overloads (§2.5) is what surfaced a dtype nothing had launched.
- **`array.isin` takes a caller-supplied `max_index`, not `assume_unique`** (neither strategy
  dedups). The bound is range-guarded inside the kernel: an over-tight bound is an out-of-bounds
  gather, i.e. host-heap corruption on CPU (§12.1).
- `flatnonzero` uses an inclusive scan plus a single tail read; `scatter_index_where` expects
  the inclusive scan and writes at `inclusive[i] - 1`.
- **A compaction is one kernel over an int32 flag scanned in place**, not a bool mask +
  `flatnonzero` + one gather per output (`intersection`'s segment producers,
  `polyline_simplify`). Unused optional outputs pass `None` and test `shape[0]` (a null
  descriptor's shape reads 0 on both devices). `mask_to_compact_ranks` scans in place, so
  `counts_to_offsets(astype(mask, wp.int32))` is strictly worse. A mask-then-scan compaction is
  deterministic where an atomic cursor is not (`boundary.ears` ascends on CUDA too).
- **Where values are known-bounded indices, a mask is a cheaper `unique_1d`**: zeroed mask,
  membership scatter, `flatnonzero` (vs hash table, compaction, radix sort, two readbacks).
- **`astype` is generic and pays generic dispatch**: `kernel_array.ASTYPE` launches concrete
  kernels for the four dtype pairs that are 97.9 % of calls and falls back for the rest.
- **A helper that densifies or indexes *the whole mesh* under a consumer whose answer is a small
  subset of it** is the shape to look for (cost flat in mesh size while the answer is not):
  `marching_triangles` packs its own sorted vertex-pair key instead of a mesh-wide edge id.
    - **REFUTED: replacing that densification with a sorted join** (`argsort` starts,
      `searchsorted` ends): slower at every size (two sort-class passes vs one; `searchsorted`
      of `n` into `n` is `n log n` dependent cache-missing probes). Dense labels make the join
      O(1) per element.
- **`unique_1d` deduplicates every integer dtype by one radix sort, not its hash table**
  (2026-10-02, `grouping._unique_sorted`): keys and the identity payload straight into the sort's
  double-width buffers, run starts marked into the payload's free upper half and scanned in place,
  one emit writing the values, the inverse, the run starts (counts) and each run's first
  occurrence -- the stable sort's `order` at the run start, so `unique_rows` / `unique_faces` need
  no `first_occurrence_indices` scatter. Against the hash path, every regime measured wins: 1.1-1.5x
  at 1 k-30 k values, 2.0-2.6x at 1 M, and at 30 M 1.5x with eight distinct values (a hot slot
  serializes the hash probes), 3.1-5.8x with `n / 3` distinct, 2.8x on random 64-bit keys;
  `unique_faces` 1.15x at 69 k faces, 1.45x at 0.87 M, 3.3x at 28 M. The values are a fresh
  `n_unique` buffer, no longer a prefix view of the sort scratch. Floats (`NaN` grouping) and the
  sub-32-bit dtypes keep the hash table, as does `hashed_occurrence_counts`.
- `read_scalar` copies with `src_offset=` from a contiguous rank-1 source (safe: the destination
  is pageable, §12.1).
- **`intersection._link_segments` is vectorized NumPy pointer doubling** (Wyllie, extended to
  open chains). The device port was declined (most remaining cost is one non-portable sort; it
  would regress the common single-contour case). **In NumPy, multi-hop chasing loses** (`h - 1`
  gathers a round: 1.3-2.7x slower than doubling), so the host doubling keeps doubling.
- **`graph.shortest_path_envelope`** keeps its double-buffer copy inside `capture_while` (§12.2).
- **`_closed_successor_cycles`, not the general `successor_cycles`, ranks boundary-walk cycles**:
  a window table `(succ^W, window minimum, hops to its first occurrence)` merged with a strict
  `<` gives every node its cycle's start and distance in one set of rounds; one `vec2i` scan
  places every cycle. **Boundary walks on a mesh that is not edge-manifold are not pure cycles**
  (`bunny_decimated`: 71 of 123 boundary edges lie in loops), so the successor table is
  `-1`-filled and a chase that reaches a non-tail drops the node (an uninitialised table
  segfaulted the CPU device). Pointer jumping chases up to `POINTER_JUMP_MAX_HOPS = 16` pointers a
  launch; the clock is flat from 8 to 64 hops (`rim_long` 0.633 / 0.512 / 0.463 / 0.454 / 0.451 /
  0.443 ms at 2 / 4 / 8 / 16 / 32 / 64). Three multi-hop window rankings (`boundary`,
  `intersection`, `graph.jump_rank`) share one merge rule; the window-minimum half is
  `kernels/array.merge_window_minimum` (strict `<`, first occurrence), called by
  `boundary.closed_cycle_jump` and `intersection.link_rank_round` (`jump_rank` carries no
  minimum), cost-neutral (0.98-1.03x on `boundary_loops[rim_long]` and `marching_triangles`).
  `graph.pointer_jump_schedule` is shared by `marching_triangles`' link.
- **`boundary.boundary_loops` is a floor row.** Its largest piece is the pointer-doubling loop
  and both mechanisms for removing it are refuted: a CUDA graph cannot be reused across calls
  (round count varies, §14.3) and a single persistent block is §14.9's shape. Its gate must
  include `mobius` (the non-orientable branch) or it is vacuous (§7.4).
- **Boundary walks**: `boundary_loops_batched` walks a pinched rim as a successor graph over
  boundary halfedges (gated on the degree flag it already reads); the dart walk's mirror filter
  and the pinched walk's gather are on the device, and the pinched walk builds its twins off
  `_BoundaryHalfedges`' own mates. Its seam/pinch census is **one zeroed buffer** holding the
  flags, two bits and the degree table, read through `read_values` (a census in a counting
  launch costs closed meshes two zeroed allocations and a 26 us slice read: 0.83x). Loops are the
  boundary of the surface with each pinch vertex split once per fan; a loop can pass a pinch
  vertex twice where two holes touch. The boundary family, `region_boundary_edges` and
  `crease_edges` classify off `halfedge.halfedge_mates` and keep their ascending-key rows through
  `halfedge.key_ordered_halfedges` (§16.11); a mesh-wide hash for the seam lost 0.74x at `dragon`.
- **Tree-wide the floor under every `holes` entry point is `boundary_loops_batched`** (§16.12).

#### marching triangles, polyline

- **`marching_triangles` links on the device from `_LINK_ON_DEVICE_FROM = 1024` segments, CUDA
  only**; on the CPU device it loses at every size (0.89-1.0x). The compaction writes the `2n`
  crossing keys straight into a radix sort (`end_bit`); adjacent equal keys pair segments (0.08-
  0.11 ms flat vs `np.unique`'s 0.11-1.14 ms at 1.5 k-26 k); multi-hop ranking carries `(ahead,
  lowest, first-min offset, steps)`; one weight scan lays curves out in the host's order; a
  malformed pairing falls back to the host link, keeping its `ValueError`. The host link reads
  keys and endpoints in one buffer and uploads every host result (offsets, closed flags, points)
  as one byte buffer addressed by typed views. **`edges_unique`'s row order is `(max, min)`
  lexicographic** (`hash_indices_rows` packs column 0 as the low digit), so a region sort meant to
  reproduce its numbering keys `max * base + min`. `split_faces_along_field` sorts only the cut
  faces' crossed halfedges and emits every class in one launch.
- **`split_mesh_with_plane` numbers only the crossed edges** (2026-10-03): the child counts'
  scan sizes the crossed-halfedge buffer exactly (`row - f` is a face's first slot), those keys
  alone are sorted into `edges_unique`'s order, and `emit_plane_split_faces` runs
  `remesh.write_split_children` (`emit_size_faces`' templates, now a shared `@wp.func`) on the
  slots. Byte-identical on both devices (22 configurations, tolerance-band and on-vertex planes):
  2.09x / 2.47x / 5.52x at `bunny` / `dragon` / `lucy`.
- **`clip_mesh_with_field(cap=True)` follows `vtkClipClosedSurface`** (2026-10-03): the input
  vertices are returned as given, only the crossing points are merged (exact-position classes,
  `points.point_duplicate_first`, over the crossings alone), and only the section is sealed --
  boundary edges with both ends on the level set chained by `graph.successor_cycles` (a section
  that runs into an input boundary is a chain and stays open, as does the input's boundary).
  `kernels/intersection.canonical_edge_crossing` interpolates from the endpoint first by
  *position* (index on a tie), so a coincident duplicate edge's crossings are bitwise equal too;
  that moved slice / split / clip crossings by at most an ulp (one quad-diagonal tie flipped in
  two of 66 split cases). **The weld it replaced was a defect, not a convention**:
  `remove_duplicated_vertices(epsilon=0)` buckets at ~2.4e-4 relative, so it merged crossings into
  nearby input vertices -- `icosphere(4)` at the 0.81 height quantile came back with 136 open and
  12 non-manifold edges, `dragon` at the median added no cap at all and returned 4 331 non-manifold
  edges -- and it sealed every pre-existing hole. Now face counts and volumes equal
  `clip_closed_surface` on every closed case probed, and open / touching-parts inputs equal the raw
  VTK class (pyvista refuses any input with an open edge; on the open, non-manifold scans the raw
  class is itself unreliable). The fill is real work now: `dragon` 1.26 -> 61 ms (3 710 cap
  faces), `lucy` 671 ms; `bunny` 2.71x faster (its median section runs into holes: three chains,
  nothing to cap).
- **A planar section loop is ear-clipped, not min-weight filled** (2026-10-04,
  `intersection._cap_section_loops`): the min-weight DP was 86-99 % of a capped clip (`O(B^3)`,
  issued one span a launch: 10 103 launches for `lucy`'s one 10 060-vertex loop). A loop within
  `_CAP_PLANAR_TOLERANCE = 1e-5` of its extent from its best-fit plane is projected **on the host
  in `float64`** and handed to `polyline.triangulate_polygon` (first version; since the frame fix
  below it calls `polyline_triangulate` on the device, the host keeps only the planarity test);
  non-planar loops and any loop whose clip returns fewer than `B - 2` triangles keep the DP.
  Capped rows against the DP, final form: 1.6x / 2.7x / 2.8x / 3.2x / 18.6x (`bunny_decimated` /
  `bunny` / `dragon` / `happy_buddha` / `lucy`, 1 100 -> 59 ms); same face counts, open edges and
  cap area to six digits as the DP, no new duplicated directed edge, fewer slivers than the DP
  (below). The pyvista parity test **did not catch a flipped cap winding** (its volume check
  passed); `test_clip_mesh_with_field_planar_cap_matches_the_min_weight_cap` does (winding
  consistency against the forced-DP cap).
    - **The round loop's ear test walks a grid, not the ring** (2026-10-04, R30-0b,
      `kernels/polyline.ear_grade_grid`): a uniform grid over **every** ring vertex (two cells a
      point, cell keys radix-sorted, starts filled per sorted run), and per triangle row only the
      cells its x extent over the row's slab crosses, padded by `EAR_GRID_PAD_ULPS = 64` float32
      ulps. Only a reflex vertex can lie in an ear of a simple polygon, but which corners read as
      reflex is their float32 rounding, so a reflex-only grid would not be byte-identical; this one
      tests exactly the vertices the ring walk does. The box comes free with the turning-angle pass
      (`RING_EXTENT`, `atomic_max` of `p - p0` / `p0 - p` into the zeroed sums, one readback). Faces
      identical to the ring walk on every ring probed (random stars to 65 536, the scan sections):
      the round loop 2.6x `dragon`'s 1 739-loop, 20x `lucy`'s 10 060-loop (180 -> 8.8 ms), random
      star 23x at 65 536 (94 -> 4.0 ms). Walking the triangle's **box** instead of its row
      crossings, with a ring-walk fallback past `n / 8` cells, was a 0.82x loss on the star (long
      thin diagonal ears). `EAR_ONE_BLOCK_MAX` fell from 1 024 to 512 on both devices (below).
      The pad is a margin no random ring reaches (deleting it changes no face); walking the
      mirrored triangle's cells instead of the raw one's is what the single-block test catches.
    - **Short loops share one launch** (`polyline.polyline_triangulate_from_offsets`, public):
      one block per loop runs the frame (`accumulate_loop_frames`, the single-loop arithmetic bit
      for bit), the projection, the turning/reflex sums, then the fan or `ear_clip_ring`
      (`triangulate_rings`); ear ranks hash the index *within* the loop (`ear_selected(start=)`),
      so each loop gets its alone-triangulation; loops past `EAR_ONE_BLOCK_MAX` keep the round
      loop. The cap ear-clips every loop this way while the host tests planarity. Not taken: a
      packed **round** loop for several long loops (`dragon`'s three over 512 points cost 4.5 ms in
      sequence against ~2.2 for the longest; nothing else has two).
    - **The section is found among the faces touching it**: an edge with both ends on the level
      set lies only in faces with two such corners, so `oriented_boundary_edges` runs on those
      (`section_face_flags`, scan, `emit_section_faces`) rather than the mesh (10.6 -> 0.2 ms at
      `lucy`), and `successor_cycles` on the section's vertices numbered compactly (2.0 -> 0.7 ms).
      Capped clip after R30-0a/b: 2.07 / 3.07 -> 3.2 / 9.3 / 5.2 / 19.6 ms (`bunny_decimated` /
      `bunny` / `dragon` / `happy_buddha` / `lucy`), from 1 099 ms at `lucy` at round-30 HEAD.
    - **`polyline_triangulate` frame and slivers, fixed 2026-10-04.** Its plane frame was summed
      in `float32` by per-block atomics, uncentred: two distinct frames in 10 runs on a 10 060-point
      ring and a *partial* triangulation (10 056 of 10 058) in 9 of 10. `accumulate_loop_frame` is
      now one block (`LOOP_FRAME_BLOCK_DIM`), `float64` sums relative to the ring's first point
      (both sums are translation-invariant), stored without atomics, and the projection subtracts
      the centre in `float64`. That alone still stalled at 10 056: two ring vertices projected to
      the same `float32` point (a zero-length side), never strictly convex and blocking every
      neighbouring ear on the boundary. `kernels/polyline.ear_grade` now grades ears
      `EAR_NONE` / `EAR_THIN` / `EAR_GOOD`: a corner on a neighbour, or one whose height over its
      cut is within `EAR_THIN_ULPS = 4` float32 ulps of its coordinates (`corner_is_thin`), is thin
      and competes only in a round with no good ear; the same rule sends a ring with such a corner
      past the convex fan (whose triangles at vertex 0 would be exactly those slivers). Thin
      corners skip the containment walk unless the previous round found no good ear
      (`EAR_CHECK_THIN`). Slivers (height at most one ulp of the ring's scale) on the scan
      sections, `bunny` / `dragon` / `happy_buddha` / `lucy`: ear clip before 0 / 12 / 13 / 216,
      after **0 / 0 / 1 / 10**, the min-weight fill 0 / 1 / 2 / 40. Swept: 1 ulp left 8 / 6 / 91, 16
      left 0 / 1 / 17, 64 left 1 / 3 / 70. The cost is rounds, not arithmetic: a run of
      near-collinear points is retired from its ends, so `lucy`'s ring takes 205 rounds against 122
      (`polyline_triangulate` 30 -> 41 ms there); the 1 024-point star rows are unchanged (1.00x,
      after moving the coincident-neighbour test into the reject branch and gating the walk: each
      added term in `ear_clip_block` cost ~0.045 ms on its own, a code-shape cost). Rejected: a
      float64 longest-edge flip pass on the cap (left 61 at `lucy`, 25-80 ms of host Python) and
      `remesh.flip_to_delaunay` (fixed point after one call, 5 faces still inverted). Pinned by
      `test_triangulate_polygon_ring_with_a_repeated_corner_is_complete` /
      `_leaves_no_rounding_level_slivers` (each fails on its own rule's removal, both clip paths)
      and `test_polyline_triangulate_is_reproducible_far_from_the_origin`. A rounding-level
      sliver's orientation is set by rounding, so a float64 re-projection reads some as
      "inverted" against the float32 ring the clipper saw: check against that ring before calling
      a face a fold.
- **DECLINED: a record buffer for small `marching_triangles` level sets** (2026-10-03): 0.84-0.95x
  on `sphere_med`, flat on `sphere_small`; the number is at `kernels/intersection.
  marching_triangles_segments`.
  `marching_triangles_segments` re-zeros at the isovalue as it reads.
- **`closed=True` wraps the index** in every `polyline_*` closure (`upsample`, `smooth_upsample`,
  `downsample`, `resample`, `point_distance`, `simplify`, `radius`): no `polyline_close` copy,
  closure decided per thread, tables sized for `n` segments with a zero-length closing slot when
  the input already repeats its first point (`seam_repeats_first`), so every prefix is the
  `n - 1`-segment one bit for bit. `loop_point` (one select) replaces `wrap_index` (two modulos)
  wherever the index is at most `n`. **Evaluate a per-thread predicate before a long loop, not
  after** (the same closure test after `distance_to_segments`' 65 536-iteration loop cost 13 %
  device time even when not taken). The `allclose` tolerances are kernel-scope constants
  (`constants.ALLCLOSE_*_CONSTANT`), removing two launch arguments per closure kernel.
  `polyline_resample(closed=True)` on empty input returns empty.
- **A starved brute-force grid whose reduction is min or max slices its inner loop over a second
  grid dimension and commits with `atomic_min`** (order-free, byte-identical;
  `polyline.distance_to_segment_slices`). **The lanes must be the outer items**:
  `dim=(slice, query)` broadcasts each segment load across the warp; `(query, slice)` measured
  2-6x slower at large slice counts.
- **`polyline_downsample`'s pointer-doubled walk** and its crossover (2 048, CUDA only): §14.7.
- **One-block loops** (crossover ~1 024 items, same reason as §16.16's one-block CG: a graph
  recorded per call is most of a small round loop): `triangulate_polygon`'s `ear_clip_block` (1
  024 lanes, `block_sum` barriers) up to `EAR_ONE_BLOCK_MAX = 512` on both devices since the round
  loop's grid ear test (2026-10-04): the block walks the ring per corner, so against the grid it is
  2.5x at 64 corners, level at 512, 0.6x at 1 024 on CUDA; on CPU level at ~640, 0.5x at 1 024,
  0.1x at 4 096 (it was every size on CPU while both walked the ring); `polyline_simplify`'s
  `rdp_simplify_block` up to `RDP_ONE_BLOCK_MAX = 4096` (CUDA 2.0-2.3x to 1 024 points, 1.28-
  1.30x at 4 096, 0.86-0.96x at 8 192; an RDP round is four streaming passes rather than an
  O(ring) test per corner; CPU wins at every size). Removing a step barrier **hangs** the ear
  loop on CUDA and fails `test_simplify_block_and_round_loop_match_reference` on the 1 500-point
  walk. The orientation test in the ear loop reads input coordinates (not centred-and-rotated:
  §12.4's sign-test hazard); the near-collinear ring's faces differ from before, both valid, and
  the new one covers exactly trimesh's region (`test_triangulate_polygon_near_collinear`). No
  reference pins the diagonals. `rdp_split_spans`' `span_lo`/`span_hi` and the ear loop's
  `init_ring_slot` / `clip_ear` are `@wp.func`s (check 13's allowlist entries removed, §4.5).
- **`polyline_radius` is one fused pass**; `reduction="mean"` differs from the old route by 9.4e-8
  relative (summation order). **A per-thread closure predicate beats a `dim=1` flag launch on
  CUDA and loses at 20 000 points on CPU** (0.87x, accepted per §9).
- **Bridge validation needs no edge table**: one face-parallel census of each queried pair's
  undirected and directed matches answers "on the rim, wound as its face" (both counts 1) and
  "edge exists" (`holes.bridge_edge_census`; the boundary-edge build it replaced was 11x on
  `lucy`).

#### levelset

- Marching-cubes slab chunking is not viable (§16.3); pytorch3d's `marching_cubes` needs no
  convention fix at all (§7.6).
- **`marching_cubes` is a port of `IsoSurfaceMarchingCubes.extract`, byte-identical including
  buffer order** (2026-10-06, `kernels/levelset.marching_cubes_counts` / `_emit`). Warp's
  extraction allocates 44 bytes of scratch a node (two zeroed `3n` count buffers, their scans,
  a `(nx, ny, nz, 3)` edge-to-vertex table, the cell counts and scan: 11x the field, ~6 GB at
  513³), runs two scans and two host reads. The port counts each node's crossing edges and its
  cell's triangles into one `vec2i`, scans it once in place and reads one total (8 bytes a node);
  the emit pass recovers an edge's vertex from its owner node's scanned count, recomputing which
  of the owner's higher-axis edges cross. Vertex numbering (edge's lower node row-major, then
  axis), triangle order, the spacing (`(upper - lower) / cells` in float32) and the lerp are
  Warp's, so the outputs compare equal as bytes on both devices: tori, anisotropic world bounds,
  exact ties at `iso`, reversed bounds, 2³ lattices, every Poisson / adaptive / `resample_uniform`
  lattice on bunny, bunny_decimated and dragon at depths 7-9. **NaN is the trap**: Warp's case
  code reads NaN as below `iso` while its edge test never crosses, so a face can name an edge
  with no vertex and Warp writes `-1`; the port reproduces the `-1`
  (`test_marching_cubes_is_warps_extraction_bit_for_bit`'s NaN arm bites on its removal). CUDA,
  min of interleaved runs: 2.98x / 2.37x / 2.79x / 2.9x at 64³ / 128³ / 257³ / 513³ (513³:
  26.9 -> 9.2 ms wall, 9.5 -> 2.8 ms device); CPU 1.31x / 1.58x / 1.59x at 64³-257³. No size
  gate: it wins at every size. Warp's public `sparse_marching_cubes_from_cells` over the crossing
  cells was the first candidate (7.7 ms at 513³, same faces and vertex set) but is not bit-exact:
  it rebuilds corner positions from the cells' minimum subscript, sorts every corner's int64 code
  and reads back three times.

### 16.6 `proximity`, `metrics`, `neighbors`

#### k-NN and ball queries

- **`geodesic_ball`'s per-source BFS stops once an in-ball neighbour finds the queue full**
  (2026-10-02, `kernels/algorithms/bfs.per_source_bfs_collect`, when `min_count` fits the
  capacity): the remaining dequeues could only count drops, so the returned rows are identical
  (CPU byte-identical over 18 radius / `min_count` cases) and the warning's number now counts
  clipped *sources*. `relax_approx` (3 % of the diagonal, 48 M drops a call at `dragon`)
  1.21-1.24x at 0.4-0.5 M vertices; `geodesic_ball` at 5 mean edges is flat (it rarely clips).
  The count is exactly one per clipped source (visited-table and backfill drops included), and
  the warning says "`k` of `n` neighborhoods exceeded the fixed capacity"; the dropped-neighbour
  count it used to print is only known by walking the tail the early exit skips.
  **The exact count must cost nothing per insert** (2026-10-02): the first version compared
  `visited_n` before and after every visited-table insert and cost `geodesic_ball` 1.33x and
  `relax_approx` 1.59x at `dragon` / `happy_buddha`. A dropped insert now returns
  `_VISITED_MAX_FILL + 1`, which the `>=` bound still reads as full, and the source commits once
  on `clipped or visited_n > _VISITED_MAX_FILL`: outputs and warnings identical (76 arrays, both
  devices), 1.43-1.48x / 1.62-1.65x against the per-insert form and faster than before it
  (48 vs 52.5 ms on `geodesic_ball[dragon]`). Bookkeeping in a BFS's innermost loop is a
  codegen-variance hazard (§14.2): carry the signal in a value the loop already updates.
  Extracting the two phases' neighbour distance as `bfs.bfs_center_distance` is cost-neutral
  (35.8 vs 35.8 ms at `dragon`, byte-identical on both devices).
- **`geodesic_ball`'s chunk is its occupancy** (2026-10-02, `neighbors._GEODESIC_BALL_CHUNK`):
  the per-source walk is latency-bound and one launch holds a chunk of sources, so at `1 << 15`
  it ran under a tenth of the device's threads. `1 << 16`: 1.51x `bunny` (one launch instead of
  two), 1.26-1.28x `happy_buddha` / `dragon` at five mean edges, 1.09-1.13x at 3 % of the diagonal;
  `1 << 17` and `1 << 18` lose (0.84-0.70x at 3 %: the ~6.6 kB-a-source pools outgrow L2). CPU flat
  (0.97-1.04x). Outputs are per-source, so identical; `test_geodesic_ball_chunks_agree` forces
  the multi-chunk path no fixture reaches.
- **The `1 << 16` chunk is not the best chunk everywhere, and the refill is not why** (2026-10-03,
  R29-0). `principal_curvature[sphere_large r=3]` reads 0.85x against `1 << 15`, but the visited
  pool's `fill_` is below the timer's resolution and the cost is elsewhere: the walk kernel itself
  (1.73 -> 2.32 ms device over the call: a radius-3 walk touches ~2 KB of its rows, and 64 k
  concurrent sources overflow the 96 MB L2 where 32 k fit) plus allocating the doubled pools in a
  drained mempool (+0.6 ms host). Chunk sweeps disagree by mesh: `sphere_large` is fastest at 24-32
  k at both radii (4.15-4.42 vs 5.59 ms at r=3), `bunny` / `dragon` at 64 k (tail-dominated,
  irregular walks; `dragon` r=8 75.9 vs 88.9 ms at 32 k), so no single constant or size rule
  serves both. **REFUTED by pricing: restoring each visited row on exit** to drop the refill (the
  plan's R29-0): the inserted set is not recorded (out-of-ball neighbours are inserted but kept
  nowhere), so a restore re-walks every queued vertex's adjacency, more than the memset it removes.
  **REFUTED: a locality-preserving visited hash** (output-neutral, the drop rule is count-based):
  an 8-slot-grouped Fibonacci hash (consecutive ids share a sector) helps `sphere_large` at 64 k
  (4.85 vs 5.59 ms r=3, 20.5 vs 21.1 r=8) and loses on the scans (`bunny` r=8 0.87x, `dragon`
  0.97x r=3, 0.83x r=8): linear probing over runs of consecutive ids lengthens the chains.
- **DECLINED: a smaller visited row for small balls** (2026-10-02, R28-7). A source whose walk stays
  under a smaller row's fill bound inserts the same set, so a 512-slot row is exact for every
  source below 384 inserts (all of `bunny`'s at five mean edges, 98.9-99.4 % of `dragon` /
  `happy_buddha`'s, ~none at 3 % of the diagonal, where every ball fills the 512 queue). Three exact
  forms, outputs identical to HEAD on both devices, all slower than the plain 1024-slot row at the
  1 << 16 chunk (37.2 / 83.0 ms `dragon`, 45.5 / 105.9 `happy_buddha`, 2.34 `bunny`):
  - **retry pass** (a source that fills the small row stops and runs again in a full row): the
    retried sources are the longest walks and a launch costs its longest walk whatever its thread
    count (a 4 096-source probe launch took 1.0 ms, as long as the 31 851 sources after it), so
    the retry launch is a second tail per chunk. Small-row launches alone were 1.21x on `bunny`
    (kernel 1.31 -> 1.08 ms). With the retry test after every insert the walk ran 3.4x slower
    (§14.2); moved into the loop conditions, 1.2-1.5x slower on the large meshes;
  - **grow in place** (the row's base, mask and fill become loop-carried values, the entries are
    rehashed into a full row when a dequeue could overflow): 42.3 / 94.3, 54.9 / 119.9, 2.28 ms;
  - **two phases** (one loop per row, constant fill bounds, a switch between them): 42.5 / 102.6,
    51.7 / 130.2, 2.27 ms. The second loop's code costs the walk more than the small row saves.
  The walk is latency- and tail-bound, and anything added to its loop is paid by every source.
- **`geodesic_ball` was never nondeterministic; its radius was** (closed 2026-10-02). The
  "differs in some processes" reading came from `r = 5 * mean_edge_length`, whose float32 sum
  committed one `atomic_add` per block and moved in its last bit between processes and calls;
  a vertex pair exactly `r` apart (`bunny` 9560 / 15147) then flipped in or out. Fixed at the sum
  (§13.2's deterministic float sum). **Before calling a kernel nondeterministic, check its scalar
  inputs are bit-stable across processes** (`repr(float(...))` of every derived radius).

- **`query_hashgrid_nearest`'s cost is *cubic* in how far `initial_radius` under-estimates the
  answer distance**, because that one scalar sets both the cell width and the seed. The default
  estimator inverts the *target* cloud's density, wrong whenever the clouds are displaced; the
  cost is in the cell walk, not the linear-scan fallback. The public lever is `initial_radius=`.
- **REFUTED: seeding the *forward* pass from a query-prefix probe, gated on size.** Sweeping size
  alone looks like a gateable cliff; fixing size and varying separation shows the real variable
  is answer-distance over point spacing, and not monotonically. Vary the fixture's own free
  parameters before trusting a gate.
- **REFUTED: an automatic cell-width trigger for `neighbors._knn_cell_size`.** A brute-force probe
  over a cloud subsample recovers the missing displacement term (subtract the subsample's own
  spacing or the estimate over-widens 3-5x) and is worth 2.0-11.4x through the middle band, but
  doubles the on-surface call (the benchmarked point) and at large displacement the wide cell is
  a 1.6-2.2x loss (the linear scan is then right). **Bar: a probe under ~0.05 ms ships.** Method
  note: `knn_sorted_insert` binary-searches its row's **whole length**, so a probe handing it a
  row wider than `k` gets silent garbage that reads like a ordito defect. The block-cooperative
  lead is refuted twice (the grid walk cannot be lane-split, §12.2; the linear-scan fallback is
  uniformly expensive where it costs).
- **`grid_bins=None` default**: the smallest multiple of 32 whose cube holds two bins a point,
  clamped to 128-384 (128 below ~1 M points; `lucy`'s 14 M points occupy 4.6 M cells which a
  128^3 table folds onto 2.1 M buckets). `interpolate_from_points`' radius path reduces each
  query's Gaussian mean during the grid walk (`interpolate_from_points_in_ball`, same `in_ball`
  predicate and visit order as `ball_collect`, byte-identical), building no ball-query CSR. The
  only CUDA difference from before is tied neighbour indices at finer bins (a tie's identity is
  unspecified). A shared ball-walk helper is declined (4-line loop, predicate already shared).
- **`query_nearest`'s non-monotonic drop is `k >= 8` and hash-grid-specific**: once a row's true
  k-th distance exceeds the grid's widest radius it falls back to an exact O(n) scan. A row whose
  query sits on a point of the cloud (nearest found within `DEFER_NEAR = 0.5` initial radii) is
  marked `DEFERRED_ROW` and finished by the BVH row kernel (`only_deferred=1`), for `k <= 64` at
  `n >= _KNN_DEFER_MIN_POINTS = 8192`. The grid then wins every self-query cloud, 1.1-3.6x, and
  in-tree self-queries dropped `backend="bvh"`. Price: one readback per `k > 1` default query
  (0.89-0.97x on uniform surface clouds). **REFUTED: deferring off-cloud rows too** (a small ball
  in the empty space inside a surface cloud overlaps many interior BVH nodes: `knn_far` 1.42 ->
  3.04 ms on `bunny_decimated`). **REFUTED: widening the deferral** (`DEFER_NEAR` swept 0.5 / 1 /
  2 / 4 / everything): losing cells do not move, the far set gets worse up to 2.1x; a hand-set
  `initial_radius` of 2-4 spacings does not help. An off-surface row doubles its radius to the
  widest walk before any finish runs, so a fix must decide *before* the walk. One measured loss
  is kept: `interpolate_from_points`' `k` path on queries one to two spacings off the cloud, 0.42x
  on `bunny_decimated` (vs 1.6-6.6x on far queries); no backend dominates (grid beats `bvh` in 24
  of 34 cells by up to 3.6x). **Re-probed on Warp 1.18 with the size-gated leaf** (2026-10-05, `lucy`
  vertex subsets, self queries): deferring is 0.82-0.87x at `k = 7` up to 8 k points and 4.1-13.6x
  from 16 k; at `k = 64` 0.77x at 2 k and 1.13-1.81x at 4-16 k. The 8192 threshold stands. Off-cloud
  queries are the open defect: the default grid `k = 7` took 133 / 285 ms on 1 M / 2 M-point `lucy`
  subsets with queries two spacings off, against 11 / 23 ms through `backend="bvh"`.
- **`k`, not `n`, is what is still slow in the k-NN path**: insertion cost is super-linear in `k`
  because the candidate row lives in global memory (§2.9's register row targets it).
- **A tie exact in float32 is not a tie in a float64 oracle**, so an exact set compare pins the
  adjacency's column order rather than the answer.
- **The deterministic cheap adjacency**: the unsorted builder was declined for years because its
  atomic row order varies, yet the ball's answer never changed (same counts and sets; curvature
  moved ~1e-6 on a field of range 4 at ill-conditioned vertices the docstring disclaims). Canonical
  rows (`array.sort_segments`, one launch) keep the gain. "It broke a bit-exactness test" and "it
  changed the answer" are different findings; only the second justifies a revert. The per-row
  shell sort is O(d^1.3) on one thread: a win to a hub degree of a few hundred, a large loss
  past it (use `edges_to_csr`); a shell sort because a segment's width is data, with no host
  branch to `segmented_sort_pairs`.

#### Nearest point, collapsed-point mesh

- **A `k = 1` query is a closest-point query.** A `wp.Mesh` whose triangles each collapse onto
  one point (`(i, i, i)`; public `neighbors.mesh_from_points`) answers
  `mesh_query_point_no_sign` with the nearest *point* exactly, at the BVH's pruned-descent cost
  regardless of displacement (187 -> 4.2 ms on `dragon` shifted 5 %). It loses on queries on the
  cloud (the tree build: 0.74 -> 1.76 ms), so the grid kernel defers to it at `k = 1`: a row that
  would scan is marked `DEFERRED_ROW` and counted, one 4-byte read decides whether to build the
  tree. `bvh_constructor="lbvh"` is the only viable one (`sah` / `median` 30x slower);
  `bvh_leaf_size=1` is 1.4-1.7x Warp's default for this mesh. Distances are byte-identical to the
  BVH walk. **Load-bearing Warp detail: `mesh.h`'s sliver cull `|n| / sum|e|^2 < 1e-6` reads NaN
  on a collapsed triangle, which compares false, so it is tested not culled** —
  `test_mesh_from_points_answers_the_nearest_point` fails if a release changes that. Used by
  `query_nearest(k=1, backend="bvh")` without an accelerator, both ICPs' cloud targets and
  Chamfer/Hausdorff cloud searches. Bringing the deferral radius in (0.25-0.5 of the scan
  cutover) is a trade: it defers the odd outlier of an on-surface set, whose one row pays for the
  whole tree (`bunny` on-surface 0.35 -> 0.50 ms); kept at the cutover.
- **`query_nearest(out=)` declined at 0.97x**: the two `(m, 1)` views a rank-1 `out` needs plus
  checks cost more than the allocation they replace.
- Merging `nearest_point_via_mesh` with `query_nearest_via_mesh` is declined (a launch argument
  on a launch-bound `k = 1` path, 0.96-0.97x). Two optional arguments on the shared k-NN kernels
  measured 0.96-0.97x on their other callers at 2 562 queries; the ICP step is a variant of its
  own (`neighbors.query_bvh_nearest_after_step`, sharing `_bvh_row_search`, a `wp.ref`-row
  `@wp.func` per bucket).
- **`max_t` / `max_dist` unbounded is bit-identical to the scene diagonal** (`mesh_query_ray` /
  `_anyhit` start `min_t = max_t`; `mesh_query_point_*` square the limit, `inf^2 = inf`, and only
  prune with it); computing the diagonal cost two launches and a readback. 330/330 outputs
  identical on both devices, grazing rays included. **Three kinds of limit stay**: one that also
  sizes something else (`visibility`'s ray-origin offset, `closest_point_on_edges`' initial
  radius); one inside an iteration whose state can outgrow it (`max_tangent_sphere`'s shrink
  query, where a miss *is* the answer); and one that prunes queries outside the geometry
  (`containing_faces_2d`: identical answer, 1.1-1.2x slower).
- A caller-supplied `edges_sorted` is never cheaper than packing keys from `faces`
  (`sorted_face_edge_keys`); the parameters are removed (§16.0). A `Trimesh` property that built
  `edges_sorted` cold was pure overhead. `is_watertight(mesh=)` takes a factory and builds the BVH
  only once both manifold tests pass. **Declined: an early exit on the edge test** (saves an open
  mesh a third, costs a closed one 3-8 %: the vertex test's launches stop overlapping the sort).

#### winding_number: exact hierarchy (boundary caps)

- **`winding_number` was a brute force, `O(queries x faces)`**, compute-bound at ~2e11 solid angles
  a second (4.4 us a query on `dragon`); no tree existed. Above a size gate it now walks an
  **exact** hierarchy (Jacobson, Kavan, Sorkine-Hornung 2013): a patch the query is outside the
  convex region of contributes the solid angle of a fan closing its boundary (patch plus fan is
  a closed 2-chain, winding 0 outside the region), so a query costs the perimeters of the sibling
  patches on its path, about `sqrt(n_faces)` solid angles (2.1 k on 16 k faces, 17.5 k on 0.87 M,
  53 k on 28 M). Holds for any soup: an uncancelled halfedge only costs work. Interleaved against
  HEAD (2026-10-06, build included): `bunny` 10 k 5.5x, `dragon` 10 k 22x, `happy_buddha` 10 k 28x,
  `bunny_decimated` 100 k 3.0x, `bunny` 100 k 18x, `lucy` 10 k 94x (2.67 s -> 28 ms);
  `bunny_decimated` 10 k stays on the tiled sum (1.00x). Its float64 error (<= 1.7e-6) is below
  the tiled float32 sum's on every scan mesh (2.1e-5 at `lucy`); sums in int64 fixed point, so it
  is bit-reproducible. On the CPU device 1.3-1.7x at 16 queries, 18x at 1 k (`bunny`); the gate
  is per device.
- **The apex must lie inside the region the query is outside of.** With a 14-DOP region and the
  box centre as apex, one query in 10 000 (Hilbert-ordered `bunny`) was off by exactly one turn:
  the fan left the DOP. The apex is the mean of the patch corners (float64 sums: a float32 sum of
  millions of corners drifts), and the region is grown by 1e-6 of its coordinates for the
  float32 rounding of the apex. `test_winding_number_tree_fans_from_inside_the_patch_region`
  pins it.
- **Declined, measured at the site** (`kernels/proximity.py`): a Barnes-Hut far field over the
  caps (order 1/2, beta 2-8: no gain at any useful accuracy, 3-30x slower when forced, because
  siblings next to the query are never well separated); cap edges as vertex indices (0.30x);
  a complex-product phase sum instead of `atan2` (0.81x); one block per lattice cell (16x slower);
  Morton-sorted queries (1.04-1.08x). Warp's native order-2 walk needs accuracy 8 for 1.8e-5 and is
  slower than the exact hierarchy from accuracy 6. Open: a Hilbert face order (1.08-1.23x on the
  walk); `signed_distance_on_mesh(sign_mode="winding")` taking its sign from the hierarchy (exact,
  but it still needs the closest point; Warp's accuracy-2 sign disagreed with the exact one at 0
  of 99 k `offset_mesh` lattice nodes on `dragon`, so it is speed, not a defect).
- **Block width 32**: best on 69 k faces (1.5x over 64 at 100 k queries), within 1.07x of 64 on
  ~1 M faces, and the width `kernels/proximity` already launches with (no new module variant).

#### mesh_to_mesh_distance, metrics

- **Warp 1.18 sped `mesh_to_mesh_distance` up with no ordito change** (its tiled box walk and
  the mesh BVH go through the rewritten traversal): 1.6-1.8x on the bunnies, 1.06-1.15x on
  `dragon` / `happy_buddha`, flat at `lucy`, identical distances, against a 1.17 worktree. The
  `ball_pivoting` pivot search, the other tiled box walk, is flat (1.01x, identical faces), so
  §14.2's crossover was not re-swept: its hash-grid side did not move and surface-cloud box walks
  did not either.
- **The `metrics` backward-search gate (`_GRID_BACKWARD_MIN_POINTS`) was not re-measured on 1.18**:
  it chooses between the collapsed-point mesh and the seeded hash grid, and neither moved (0.451 vs
  0.454 ms and the grid's control row flat).

- **A distance *bound* only seeds a prune limit, so a subsample is exactly as sound.** The bound
  phase (sample A by *face corners*, stride capped at 128 faces; cap each sampled closest-point
  query's `max_dist` with a brute-force 1 024 x 1 024 corner-pair seed; one `atomic_min`, one
  `read_scalar`) paid, and the bound is kept on the device in a slot the walk reads (the relax-
  and-floor rule is monotone, so the minimum of the publications is the publication of the
  minimum). **REFUTED: a per-face lower-bound filter** (prunes 0-50 % of faces, walk 0.93-1.10x;
  the pruned faces were cheap). Correctness: an unreferenced vertex of A near B used to give a
  bound *below* the answer (not a surface point); the published global best is floored at the
  smallest positive float32 (at exact zero every other zero-gap candidate was pruned and CUDA
  returned whichever face published first: CUDA now returns the lowest face index, as CPU did).
  Uses `wp.mesh_get_bvh` (§12.8).
    - **An optimization that loosens a bound stresses everything it feeds**: it reached §12.2's
      `tile_bvh_query_aabb` overrun deterministically and exposed a CUDA-only bug (the straggler
      re-walk *overwrote* the first pass's correct answer). **The CPU device, which runs no
      capped second pass, was the oracle** (§7.2). A serial uncapped straggler pass was tried and
      reverted (§14.2's load imbalance is real).
    - **A thread-per-query BVH launch is load-imbalanced, not under-pruned** (§14.2). The walk's
      time is sensitive to loop spelling by up to 5x (§14.2): its `mesh_query_aabb` variants
      measured 0.93-1.08x (closed lead).
- **Seed the *backward* search of a symmetric distance query from the forward half's own answer**
  (one distance scale), capped at the target's density estimate: `min(seed, knn_initial_radius)`
  (an uncapped seed at 5 % displacement made cells so wide that certified rows scanned thousands
  of points: 59.9 vs 7.6 ms). `backend="bvh"` at those sites is a loss at every size tested.
  On CUDA past 262 144 points `metrics` picks the backward search from the forward answer: the
  seeded hash grid when the largest forward distance is at most half the density radius **and**
  at least 75 % of backward queries are some point's forward answer (one `atomic_exch` a point),
  else the collapsed-point mesh. **A small forward maximum does not make a pair coincident** (a
  cloud covering half its partner has one); the hit fraction separates cleanly (>= 0.89 vs
  <= 0.5). The grid's best seed is 0.1-0.3 of `knn_initial_radius`. The readback, not the search,
  is the cost below ~250 k points (it stops the backward half's issue overlapping the forward
  half). Metrics compute the forward maximum once and reuse it for `"max"` Chamfer.
- **DECLINED: the `cotmatrix` loss to pytorch3d is a scope mismatch** (`cot_laplacian` returns an
  uncoalesced COO tensor with duplicates unsummed and no diagonal; coalesced to the same job
  ordito is ahead at every size). The timing is incomparable, not the result, so it stays a live
  parity comparison.
- **Apply the same detector to *both* sides' output before comparing costs**: the suite's largest
  reported loss for several rounds was a reference call that no-opped on that fixture.
- **Read the host half of a device profile before accepting a device-side attribution**: a
  `.numpy()[k]` on an array that scales with the mesh has a share that *grows* with the mesh.

#### curvature

- **REFUTED: a streaming Givens QR for `curvature.principal_curvature`'s quadric fit.** Against
  the exact rational least-squares solution it is ~430 000x less order-sensitive and 2-4x slower;
  exactly one vertex of 544 has a normal-equation error worth caring about (2 % of the field
  median, the regime the docstring disclaims). Normal equations do one `wp.outer`; QR annihilates
  five entries each needing a `float64` `sqrt`; the penalty grows with neighbourhood size, which
  is backwards.
- **REFUTED with it: symmetric Jacobi scaling of those normal equations** (free, measures a hair
  worse): the ill-conditioning is genuine near-rank-deficiency of the neighbourhood in the tangent
  plane, not column scaling (the fit already divides local coordinates by the ring radius).
- `curvature.discrete_mean_curvature` uses `wp.bvh_query_sphere` as a broad phase (§12.8), now
  walked and summed by one thread per query (`ball_mean_curvature`); `discrete_gaussian_curvature`
  walks the hash grid the same way and forms each vertex's defect from its angle sum as it reads it
  (§14.10). Both sum per query in walk order where the scatter added by atomics, so values move by
  float32 rounding (≤ 1.2e-6 relative of the field's range).

### 16.7 `sample`

- **`sample_surface` / `sample_volume` draw `lower_bound(cdf, randf * total)` over the unnormalized
  prefix sum** instead of normalizing the cdf by a map first (1.18-1.24x on `sample_surface`). Not
  bit-identical to the normalized draw, and not meant to be: of 2 M draws on `bunny`, 351 move to
  the adjacent face; against an exact float64 inverse transform of the same uniforms the normalized
  form disagrees on 8 045 and this one on 7 930 (the float32 prefix scan dominates both); per-face
  chi2/dof is 0.9873 for both. Seeded output is reproducible within a version and device, which is
  all the contract and the tests claim. The draw must stay inline in the kernel: `randf` advances
  `state` in place, and a `@wp.func` taking `state` advances a copy.
- **Reach for randomized-priority selection whenever a GPU port needs a maximal-packing / MIS-shaped
  result.** `sample_surface_blue_noise` is randomized-priority parallel dart throwing, not
  Bridson: every pool point draws a priority, a point is accepted when no smaller-priority point
  still in play lies within `r`, and everything within `r` of an acceptance is discarded. The
  tie-break-free correctness argument (the later of any too-close pair was already discarded) is
  the serial algorithm's own distribution.
- **The cover pass is load-bearing for *termination*, not just speed**: a wider shell or looser
  cover radius can stop the loop converging, because the accept step will not take a point while
  a smaller-priority alive point still covers it. Its byte-identity gate is cheap insurance.
- **The round count is stable at 4-6 whatever the cloud** (five mesh shapes, 55x pool range;
  logarithmic growth as MIS theory predicts); the loop around the two shell-scan kernels is near
  its launch floor.
- **Attribute at the benchmarked parameter** (§15.3): inverting the propose kernel to one thread
  per *accepted* point won at the dense natural radius and lost at the benchmarked sparser one
  (reverted). Micro-optimisations of the old Bridson propose kernel are null results (cheaper
  permutations biased or cost more; dropping either pruning pass was a large loss; lazy shuffle a
  wash).
- **A *second consecutive* readback is far cheaper than §13.1's queued figure** (§15.2).
- **Run the loop in cell-sorted index space**: the pool is already radix-sorted on its cell key,
  so that permutation *is* the cell order; permuting payloads once makes a cell's members the
  contiguous run its offsets name, every read stride 1. The tie-break stays byte-identical
  because the comparison still reads the original pool index, on the priority-tie branch only.
- **The dart throw's setup writes every buffer the loop starts from** (2026-10-02): the permuted
  points (no `gather`), the `ALIVE` state (no `wp.zeros`) and the identity work list (no
  `arange`) ride `dart_point_setup`; the accepted set is int32 flags written through `bucket`
  (a permutation, so no zeroed mask), scanned in the spent survivor-position buffer, and one
  `sample.emit_kept_samples` launch writes points and faces (no `flatnonzero`, no two gathers).
  The Poisson-disk elimination's tail takes the same kernel over its `alive` flags.
  `sample_surface_blue_noise` 1.06-1.09x (`bunny`, `dragon` at half radius), Poisson-disk flat;
  byte-identical on CPU and on CUDA at `bunny`. **At `dragon` the CUDA output is not
  reproducible on the unchanged tree** (19 552 / 19 550 / 19 540 points over three runs of
  HEAD): gate an A/B of this sampler on the CPU device.
- `sample.apply_deletions` clears `alive` in the pass that subtracts the deleted points'
  contributions (one `wp.map` per elimination round); `homology.forest_link` clears
  `candidate[e]` for an accepted edge instead of writing an `in_forest` mask. Both are races only
  on paper (each thread writes its own slot and every reader of a written slot is excluded by a
  second test); a kernel clearing state it has just read needs an allowlist entry for check 13.

### 16.8 `smoothing`, `laplacian`, `energies` (operators and smoothers)

- **REMOVED: batching several CG iterations per conditional-graph test.** The overshoot is a
  fixed number of launches whose share is set by solve length: long solves win a little, short
  ones lose a lot (+13 % best, -40 % worst). Its sweep could not express the failure mode (three
  of six cells never reached the captured loop: read the launch count beside the ratio). A batched
  body is right for the settle loop (§16.16) and wrong for `CG_CHECK_EVERY`.
- **REMOVED: block conjugate gradient over two columns (`_BlockCg2`).** A wash to +3 % on
  well-conditioned systems; iteration count up 2.14x on ill-conditioned ones (the columns'
  directions go nearly parallel). A Jacobi-diagonal-spread predictor separates the cases by three
  orders of magnitude but is worth at most +3 %. **A mechanism whose best case is a wash needs
  removing, not a better gate.** It shipped after two well-conditioned saddles and never saw
  `saddle_graded` (§9's fixture-pair rule).
  `test_two_column_solve_costs_no_more_iterations_than_its_worst_column` is the deterministic
  guard (well- and ill-conditioned shift); `_BatchedCg` satisfies it by construction.
- **Every direct-factorization reference is flat across the conditioning axis and ordito's CG is
  not**: a conditioning regression is an iteration-count problem, not an assembly problem.
  `isotropic_remesh` inverts it (ordito flat, reference tripling) because it runs a fixed
  `iterations` x five launches: flatness is a fixed work budget, not insensitivity.
- **A readback census parametrized by the loop count is the tell for a per-pass host sync**
  (patch `wp.array.numpy`, call at two iteration counts; a fixed base plus one per pass is the
  signature). ~25 lexical readbacks in loops are mostly genuine host *branches*; check whether the
  host branches on the value (`registration.icp_*`'s early exit is the counter-example).
- **When moving a host-side `if` into a kernel, check what the *skipped* branch did with
  arguments it never evaluated.** An identity is only an identity for finite operands (a
  face-less mesh's centre is `NaN`; a "skip" as scale `1.0` propagates it; every fixture has
  faces). `wp.utils.array_sum` writes nothing for empty input, so its `out=` must be `wp.zeros`.
  `filter_mut_dif_laplacian`'s correction writes *unconditionally* (the host version applied an
  offset of exactly zero): answer the skip-versus-identity question from the code replaced. Its
  three volumes are summed by `wp.utils.array_sum`, deliberately the same reduction (not
  `measures.volume`'s tiled one), because the correction is a difference of nearly equal volumes.
- **`smoothing.inflate`'s inherited volume constraint is load-bearing** (it restores the volume
  the smoothing half-step removed); exposing it as a keyword was declined (§4.2). **It is device
  resident** (2026-10-03, `smoothing._VolumeConstraint`, shared with `filter_laplacian`): the
  anchored volume (`MESH_SIGNED_VOLUME` on the `float64` copy) and the first four moment integrals
  (`kernels/measures.centroid_integrals`, `moment_integrals`' chunking and per-face arithmetic
  through `tetrahedron_first_integrals`, so its totals are bit-identical on CPU) go into one state
  buffer; `volume_rescale_parameters` forms the scale and the `float32`-rounded centre once per
  pass, so there is no readback and one `float64` cube root a pass rather than one a vertex (the
  per-vertex `pow` was 1.95 ms of a `lucy` pass; the ten-integral `moments` kernel 6.9 ms, the
  four-integral one 2.6). `inflate`'s pass is written out over buffers allocated once, normals
  normalized as the displacement reads their accumulator (`vertices.normalized_accumulated_row`).
  CPU byte-identical (explicit, fixed-point and BiCGSTAB paths): `inflate` 2.06x / 1.73x / 1.48x
  and default `filter_laplacian` 1.17x / 1.49x / 1.39x at `bunny` / `dragon` / `lucy`. Remaining
  `lucy` device time is the per-pass `face_signed_volumes` + `array_sum` over a 224 MB per-face
  buffer; a tiled fold would drop the buffer but change the reduction order (outputs at rounding):
  not taken.
- **`filter_spikes` is three launches a pass** (2026-10-03): corner angles scattered as they are
  formed (`vertices.scatter_corner_angles`), spikes marked and counted while re-zeroing the sums
  (`smoothing.mark_spikes`), and only the spikes averaged into a reused buffer from `float32`
  positions widened on read (`operator_row` is generic over `vec3` / `vec3d` fields). **Few spikes
  need only their rows**: below `_FEW_SPIKES_RATIO = 64` vertices a spike, a pass sorts the spike
  rows' neighbour keys and averages each row from them (`average_spike_rows`, the symmetric
  operator's weights, normalization order and arithmetic), instead of building the whole-mesh
  operator (32.8 ms at `lucy`, which flattens 3 vertices). Both paths byte-identical to the old
  call on both devices; `test_filter_spikes_spike_rows_match_the_whole_operator` forces each way
  (bites on a dropped dedupe). 2.74x `bunny`, 1.80x `dragon`, 2.10x `happy_buddha`, 10.6x `lucy`.
- **`relax_keep_volume` builds sorted neighbour lists, not a CSR from triplets** (2026-10-03):
  `edges_to_neighbor_lists(sort_rows=True)` is exactly `edges_to_csr`'s structure without sorting
  every edge twice (4.1 vs 23.7 ms at `lucy`); the relaxation family reads a `None` region as all
  vertices (`kernels/smoothing.in_region`) instead of allocating an all-`True` mask. 1.44x /
  1.47x / 1.34x at `bunny` / `dragon` / `lucy`, byte-identical. The census found no other
  `edges_unique` -> `edges_to_csr` site needing only structure (`graph.py`'s carries lengths).
- **`equalize_triangle_areas` sums its ring in `float32`** (2026-10-04, R30-3,
  `kernels/smoothing.equal_area_position`): the quadratic form needs only edges and the right-hand
  side is taken relative to the free vertex, so the sums hold ring-scale differences; the 3x3 (or
  tangent 2x2) solve stays `float64`, and a system whose determinant is under
  `EQUAL_AREA_F32_CONDITION = 1e-4` of its trace's power is redone by the old `float64`
  accumulation (`equal_area_position_f64`; 1e-6 let a dragon ring through 3e-6 off). The row is
  summed in **ascending face order** (`next_row_entry`, an O(row) rescan, free): vertex-face rows
  are unordered sets, and a `float32` sum in slot order made results differ run to run (the
  region test's bit-identity assert caught it); a `float64` accumulation of the `float32` terms
  instead cost 3x. Ten passes: 1.44x bunny, 2.9x dragon, 3.2x lucy (75.7 -> 23.6 ms), 1.8x with
  `no_shrinkage` (its normals pass). One pass agrees with the `float64` path to the positions'
  rounding; ten `no_shrinkage` passes fling ~500 dragon vertices past 0.01 on either path, and
  those differ (they also differ run to run on the `float64` path).
- **A `for` loop around a single-column solver, in a module whose siblings call the batched one,
  is the textual tell for a multi-column solve**: three position components share one operator.
- **A helper returning the same sentinel for two different "nothing to do" cases is a defect
  waiting for the rarer one** (`filter_implicit_fairing`'s Dirichlet path ran the unconstrained
  solve when pinning the boundary of a single triangle or small patch).
- **Conditional-emit triplet writers point unwritten slots out of range, not to zero** (§12.7);
  `holes.fill_smooth` reaches it through a shared helper.
- **`filter_laplacian(implicit_time_integration=True)`**: the uniform operator is built from
  directed `mesh.edges` (trimesh's convention), so on a mesh with a boundary `(1 + lamb) I -
  lamb L` is **not symmetric**. CG on it was wrong on open meshes at larger `lamb` (1.3e4 off on
  `hemisphere`, 1.3e7 on `saddle_small` at `lamb = 5`; 3.3 s on `saddle_graded`); the parity test
  ran on the closed icosahedron only. Now a fixed-point iteration `x' = (b + lamb L x) /
  (1 + lamb)`, strictly diagonally dominant, contracting by `q = lamb ||L||_inf / (1 + lamb)`
  whatever the symmetry: the step count bounding the error at `CG_TOLERANCE` is known up front
  (22 at `lamb = 0.5`), each step is one fused `vec3d` launch with no reduction or readback, and a
  pass is recorded once and replayed. Matches trimesh to 1e-7 closed and open at `lamb = 0.5` and
  `5`; past 400 steps (`lamb` above ~16) or for a non-contraction it solves the assembled system
  with `wpl.bicgstab` (4e-6 at `lamb = 50`).
  `test_filter_laplacian_implicit_on_open_meshes` covers both arms. An unreferenced vertex stays
  put (as `operator_row` makes it on the explicit path). The Chebyshev preconditioner is
  deliberately *not* used there: CG on a non-symmetric system has no guarantee under any
  preconditioner.
- **`filter_implicit_fairing` / `filter_taubin(recompute=True)`** build the pattern once and pass
  it (`pattern=`, §3.7, §16.9); `filter_taubin(recompute=True)` at 4 passes went 45 / 50 / 1 -> 19
  / 16 / 0 launches / allocations / readbacks.
- **`smooth_region_boundary` assembles the band, not the mesh.** The free set and pattern are
  fixed across passes, so the first pass extracts the band's Dirichlet system and later passes
  rewrite values and right-hand side from new half-cotangents
  (`kernels/smoothing.band_dirichlet_values`, which forms the band faces' cotangents itself);
  keeping the operator object keeps the solver state (whose graph replays). The pattern comes from
  the vertex-face rings (`band_pattern`: a count pass and a fill pass of one kernel), not a
  mesh-wide `-cotmatrix` (§16.5's densify-the-whole-mesh shape). The rim mask calls
  `selection.exclude_fully_selected_components` (a duplicated decision rule), which labels over
  `faces_to_edges` rather than `edges_unique` (labels are the component's smallest vertex id
  whatever the union order). `test_smooth_region_boundary_later_passes_match_a_rebuild` fails
  under a sign mutation of the pinned term; later passes are bit-identical on CPU, 9e-8 on CUDA.
  The pooled operator state must refresh on every use (§16.16).
- **Region solves assemble directly** (no `bsr_*` call left in either region solve;
  `SquaredLaplacianPreconditioner.from_factors`): one incidence build per region pair; the
  fixed-rim values and `M^T M` are written over region-own patterns. **`bsr_mm` sums most product
  entries along the row and the rest in its triplet sort's order, which is not stable on CPU, so
  it is reproducible only to one rounding per entry** (54 of 1 000 `M^T M` rows differ by one
  rounding). A face repeating a vertex no longer contributes a self-loop to the unit weights.
  Declined: matrix-free `M^T M` in the one-block solve (ceiling under 0.3 ms). **`smooth_region`
  solves normal equations `MᵀM` (fourth order)** with free rows `M_ff = D⁻¹ L`; the *exact*
  `M_ff⁻¹ M_ff⁻ᵀ` converges in 21-30 iterations on every system (Jacobi 421-5 000, the `MᵀM`
  V-cycle 153-313), so the one ring of rim rows does not spoil the spectral equivalence;
  approximating `M_ff⁻¹` is the design question (§16.15).
- **Fixed-rim solves in `fix_self_intersections` / `refill_region` / `fill_smooth`** assemble a
  *new* system every call, so no per-operator state is hit; the recording is not what they pay
  (fixed-rim Jacobi solve: 0.17 ms recording + 1.18 ms for 52 rounds; squared-Laplacian: 0.85 +
  2.33 ms for 21, of a 22 ms call; the polynomial's 682 tiny `chebyshev_step` launches are 3.8 ms
  of 9.3 ms device). Small systems take the one-block solve (§16.16).
- **`laplacian` module notes**: `robust_laplacian`'s negative-weight residue is entirely
  *boundary* edges, not Delaunay violations (a boundary edge has one opposite angle); the only
  remedy is Steiner points, which the contract forbids (CLOSED). The interior half is fixed by
  `intrinsic_delaunay`'s multi-edge support (a flip may create a second, geometrically distinct
  edge between already-adjacent vertices). **Quoting a `min()` over a set whose members have two
  causes can size a fix by 100x the wrong number.** `laplacian.face_half_cotangents` is the shared
  run. The energies' sandwich products and hessians stay on `csr_from_triplets` (§16.9).
  **`intrinsic_delaunay` derives its initial twins from the halfedge mates** (2026-10-03 from
  one sort, 2026-10-05 from `halfedge.halfedge_mates`: `kernels/remesh.pair_intrinsic_twins` over
  each pair's lower halfedge, the `edge_pair_topology` + first-match `local_corner` rule), not a
  whole `_FlipTopology` rebuild (row tables, quad table, a claim hash of `>= 4m` slots):
  `robust_laplacian` 1.21x / 1.15x / 1.11x, byte-identical twins, faces and lengths on both
  devices (degenerate and fin inputs included). The mates (bucketed at scale) add 1.05-1.08x at
  `dragon` / `happy_buddha`, byte-identical; flat below the bucket gate. Declined: a device-side `mollify_intrinsic` for this caller (its two reads are ~3 ms
  of an 80 ms `lucy` call). `curved_hessian_energy` validates edge-manifoldness and zeroes the
  boundary from `internal_angles_and_sums`' halfedge cursor (`energies.zero_at_boundary_edges`)
  instead of a second halfedge sort, `is_edge_manifold` and `boundary_vertex_indices` (1.02-1.05x:
  the 20 ms is the triplet assembly).
- **Mean edge length for `heat_operators`' timestep** and related: §16.10.

### 16.9 Sparse assembly: key-sorted CSR and mesh operator patterns

Rules and semantics are §3.7 (check 27); this records the measured consequences.

- **A drained mempool is a size-proportional cost** (§13.1): few-launch rows on large meshes
  (`crease_edges[lucy]` 19.0 ms wall vs 2.5 ms device; `cotmatrix[dragon]` 2.95 vs 0.41) pay it
  per allocation; raising the release threshold was 1.5-1.6x at `dragon` / `lucy`, flat at
  `bunny`. ordito does not set it (process-wide); the lever is fewer and smaller allocations.
  **Where the cost lands (nsys, `lucy`, 2026-10-02)**: not in the allocations (1.2-3.2 ms of
  `cudaMallocAsync` a call) but in the *sync after the call*, which returns the pool's freed pages:
  the closing `cuCtxSynchronize` outlasts the last kernel by 5.1 / 6.8 / 6.0 / 12.1 / 7.3 ms on
  `face_adjacency` / `edges_unique` / `crease_edges` / `cotmatrix` / `voxelize_mesh`, a third of
  each wall. With the threshold raised the wall *is* the device time (17.6 -> 11.3, 37.6 -> 23.9,
  18.8 -> 10.2 ms); the device time is CUB's onesweep radix passes first (8.2 of 11.2 ms on
  `face_adjacency`). Warp's timer sees neither half, so these rows read "host-bound" with no
  Python to remove. The remaining lever is bytes freed per call.
  **REJECTED by the owner (2026-10-04): an ordito-owned scratch cache** (R30-2: keep the radix
  sort's double buffers and the sorted-key scratch resident between calls, in any form --
  explicit release, scoped context manager or byte cap). It would speed the large-mesh rows
  1.3-1.7x only by holding memory a real caller would not expect held, so the benchmark win is
  not a fair one. Do not re-propose it, nor any other mechanism whose gain comes from keeping
  memory reserved across calls (raising the pool's release threshold included).
- **`cotmatrix` allocated 9.4 GB a call at `lucy`** for a matrix under 1 GB (three 1.28 GB
  12-triplet buffers and 5.2 GB of `bsr_from_triplets` scratch). It now emits six off-diagonal
  triplets per face plus one diagonal slot per vertex (the tail's rows prefilled with
  `n_vertices`, so an unreferenced vertex's slot is out of range and its row stays empty), and
  `kernels/laplacian.cotmatrix_diagonal` writes each diagonal as minus its row's off-diagonal sum.
  Pattern and off-diagonals are bit-identical to the old build (an off-diagonal's terms and order
  are unchanged); the `float64` diagonal is bit-identical and the `float32` one within 3.2e-7
  relative (terms summed per edge first). The duplicate-built-operator tests still reach a
  duplicate build.
- **Not transferable to `connection_laplacian`**: its diagonal is `sum w * I` and `w` is not
  stably recoverable from a rotated off-diagonal block (the decode divides by `cos rho`); it needs
  a scalar weight source, and a `float64` atomic sum would make the diagonal nondeterministic on
  CUDA.
- **Warp's `prune_numerical_zeros` prunes triplets, not entries**; `csr_from_triplets` prunes the
  assembled entries instead, sums each run at its start in the launch that flags it, and still
  skips zero triplets before the sort. Priced against Warp at 0.2 M / 4 M / 40 M triplets:
  1.06x / 1.00x / 1.10x pruned, 1.12x / 0.99x / 1.11x unpruned (a wash), so the semantics were
  chosen for the answer. Gate: 360 random builds per device, every entry we keep equals Warp's,
  every entry Warp keeps that we drop is zero there (exactly on CUDA, at most 9.5e-7 on CPU).
  `csr_from_triplets` is bit-identical to `bsr_from_triplets` on 160 random scalar builds on both
  devices. **The generic builder is only at parity with Warp (0.96-1.13x); the big wins are the
  structural builds** (`cotmatrix` 2.4-3.95x at `dragon` / `happy_buddha` / `lucy`, 1.16-1.6x at
  the bunnies; `graph_laplacian` 3.0-5.2x; `laplacian(equal_weight)` 1.4-2.5x; `robust_laplacian`
  1.5-1.7x).
- **Core (`array.csr_from_keys`)**: one `uint64` key per contribution, radix sort over only the
  bits `n_rows * n_cols` needs, run starts scanned in the payload buffer's own upper half, offsets
  filled by each row's first run so empty rows need no pass. A mesh operator's per-row value
  kernel forms each entry from its contributors in producer order (off-diagonals and patterns
  bit-identical to the triplet build; diagonals now a row sum).
- **The undirected mesh-pattern build (above ~0.5 M faces, `_UNDIRECTED_PATTERN_FROM_FACES`)**:
  one sort of `3 * n_faces` keys plus a 32-bit sort of the unique edges by larger endpoint for the
  transposed half, instead of one sort of `6 * n_faces + n_vertices`. The sorts alone are
  1.70-1.88x at `dragon` / `lucy` and 0.72x at the bunnies; whole `cotmatrix` 1.35-1.45x above
  0.87 M faces and 0.55-0.78x below 0.33 M. The second sort's 32-bit keys live in the first sort's
  upper-half key scratch (a `wp.array(ptr=...)` alias) and each entry's run end is read off the
  sorted keys, not stored (`lucy` 6.6 -> 5.3 GB). A degenerate face's self-edge, which the
  directed build keeps as its keys, is carried as a flagged diagonal slot so both builds agree. It
  is three launches more than the directed build (one exclusive scan of the whole `(3, n)` tallies
  replaces the row-count kernel and two scans). The two builds are pinned to each other by a
  forced-threshold test.
- **`laplacian.MeshOperatorPattern`** (public `laplacian.mesh_operator_pattern`) is reusable across
  assemblies over fixed faces (§3.7); `graph_laplacian` and `laplacian` (given faces) are one
  pattern and one row kernel; `laplacian_equal_weight` is directed, one key per halfedge.
  All 50 operator matrices (5 meshes, 2 precisions, 5 operators) are bit-identical to the triplet
  builds on both devices. The `laplacian_inverse_distance` benchmark row passes precomputed
  `edges=`, so it measures `csr_from_triplets` vs Warp (0.97-1.13x), not the structural build.
- **Sites left on `csr_from_triplets` and why**: energy sandwich products and hessians (each input
  row scatters over many output rows), the Loop operator (even rows are unsorted rings; a per-row
  sort is §16.6's hub hazard), `edges_to_csr` / `index_sparse` (the input *is* a coordinate list),
  the implicit smoothing system (solve-bound, 0.97-1.04x).
- **`wps.bsr_zeros` then field assignment wastes three allocations and a memset per matrix**;
  `wps.bsr_matrix_t(dtype)()` plus field assignment and `notify_nnz_changed` works on both
  devices (`linalg._bsr_over`). `heat_operators` writes its systems over the Laplacian's own
  pattern (§16.10).
- **The rest of the `lucy` allocation census is functional**: `crease_edges`' 1.9 GB is the radix
  sort's double-width keys and payload (48-bit key), `cluster_decimate`'s hash tables and face
  buffer, `mesh_to_mesh_distance`'s 963 MB of per-face and per-vertex buffers.

### 16.10 `heat` (diffusion, settle test, far field)

- **`heat_geodesic` was wrong far from its sources past a dozen rings, and the same defect sat
  under every heat-diffusion entry point.** On `icosphere(5)` from one source the worst error was
  2.34 against the great-circle distance (igl 0.019). A CG iterate after `k` rounds is a
  degree-`k` polynomial in the operator applied to the source indicator, **exactly zero more than
  `k` rings away**, and the well-conditioned heat solve met its tolerance at ~30 rounds whatever
  the mesh, so 91.6 % of vertices got no heat. **The fix is a stopping rule, not a tolerance**
  (`linalg.solve_spd_settled` at `heat._HEAT_*`, one continuous settle solve, §16.16): stop once the reached count stops
  growing *and* the change is under `1e-6`, or three chunks past full reach; the far field needs
  80-150 rounds past reach to settle and entries near `1e-300` never settle relative to
  themselves. `extend_scalar`, `transport_tangent_vectors`, `log_map`, `heat_signed_distance` and
  `diffuse_tangent_field` go through it. The graded saddle stops at reach + 192 = 336 rounds.
  Outputs vs earlier chunked runs move by 1e-15 (spheres) to 4e-9 (bunny extension) relative
  (restarted 64-round chunks never passed the vector systems' settle test). **Every igl
  comparison used to run on `icosphere(3)` or smaller**; the far-field test
  (`test_heat_geodesic_matches_igl_far_from_the_sources`, `icosphere(5)`, one and three sources,
  bound `5e-3` of range) is 0.01-0.13 % on spheres, 0.8 % `bunny`, 1.1 % `bunny_decimated`, 2 %
  `hemisphere` from three sources. `extend_scalar` vs potpourri3d is 0.0000 on `icosphere(5)`;
  `log_map`'s radius 0.8 % / 2.8 % (`icosphere(5)` / `bunny`) where it was 30 % / 26 %. The
  correctness cost is rounds: `extend_scalar` 0.33-0.35x, `transport_tangent_vectors` 0.17-0.42x
  vs the unsettled version.
- **igl averages a Neumann and a Dirichlet heat solve on a boundary mesh; ordito keeps Neumann
  only, on the measurement** (vs `igl.exact_geodesic` on `half_torus`: Neumann 0.93 % mean / 3.1 %
  max, igl's average 1.15 % / 4.9 %; tied on `hemisphere`; potpourri3d and pymeshlab are Neumann
  too and the averaged field read *below* the Euclidean distance).
- **REFUTED: the Jacobi-Chebyshev polynomial in the heat diffusions**: 1.3-1.8x faster on spheres and
  **wrong on `bunny`** at every chunk and settle setting (distance 0.68-0.90 of range off igl's,
  scalar extension divergent to 1e5): obtuse triangles give the heat system positive off-diagonal
  entries and the polynomial's interval does not cover the far field's decay.
- **`float32` is out for the heat method**: the implicit step decays by a near-constant factor per
  ring, so the far field falls below `float32`'s range (~1e-38, 1e-45 subnormal) within a few dozen
  rings (~1e-300 on the big meshes); the normalized gradient needs the *direction*, which
  `float64` carries to the antipode. The Poisson step has no such range problem but is one solve.
- **Three kernel-scope precision fixes the far field exposed** (a diffused value near `1e-300`
  squares to zero): `predicates.stable_length` / `stable_normalize` divide by the largest component
  first (used by `triangles.face_unit_gradient`, the transport and log-map kernels); the transport
  `resolved` mask compares the diffused magnitude against the *diffused indicator* at
  `_RESOLVED_FRACTION = 1e-4` (the `cave_cube` antipode sits at 1.4e-7 of its own indicator from
  round-off); `extend_scalar` divides only where the indicator is non-zero (`divide_nonzero`:
  obtuse triangles make `bunny`'s indicator negative at two vertices).
- **Independent diffusions stack into one settle solve** (`linalg.block_diag`, public, cached per
  set of operators so the stack's state replays; `kernels/linalg.stack_block_diagonal` reads each
  block's count off its own last offset, never `nnz`): transport solves `[vector; heat; heat]`,
  `log_map` `[vector; heat]`. **A single CG over a block-diagonal stack is not two independent
  CGs** (`alpha` / `beta` span every block): iterates differ at a given round and agree only once
  settled (1e-7 on the saddle and spheres); on `saddle_graded`, which never settles, `log_map`'s
  distance error vs `igl.exact_geodesic` moved 9.4 -> 9.7 % mean at the same 336 rounds. **Stack
  only systems that settle together.**
- **A fresh operator of a seen shape takes a pooled state** (§16.16): zero `ScopedCapture`s per
  call (was 2-3).
- **`heat_operators` / `vector_heat_operators` write their systems over the Laplacian's own
  pattern** (`shifted_system_values`, `linalg.bsr_with_values`): `cotmatrix` and
  `connection_laplacian` keep every triplet, so every referenced vertex's row holds a diagonal and
  the pattern merge is one launch. The three scalar operators share one pattern. The Poisson
  polynomial is built on the **first apply** (`transport_tangent_vectors` / `extend_scalar` never
  run that solve). **The timestep** reads the mean unique-edge length off the Laplacian's strict
  upper triangle (one entry per edge) into device sums, and `shifted_system_values` forms the mean
  itself: no readback. `t` moves by ~1e-7 relative (`float64` accumulation) and a zero-length
  self-edge of a degenerate face no longer enters the mean. **The scalar and vector solvers read
  the same kernel off two operators with one sparsity, so their `t` is bit-identical** (the
  agreement `log_map` depends on). A test pinned `t` to `mean_unique_edge_length` at 1e-12; it now
  recovers `t` from the system and checks the convention against trimesh at 1e-6.
  `vector_heat_operators` builds its lumped mass once. The Chebyshev interval is fitted on the
  device (§16.15), so building either polynomial does not drain the queued diffusion.
- **Normalization and signed distance**: `heat_geodesic`'s normalization is two launches with no
  readback (`source_and_global_sums`, `shift_and_orient`); `heat_signed_distance` folds the unit
  field into `vertex_field_divergence` (which also accumulates `-div`), its `"none"` shift is
  `heat_geodesic`'s device-side one without orientation, and curve segments are derived per thread
  (binary search into the offsets). `log_map` folds two maps into `log_map_from_angles`;
  `transport_tangent_vectors` folds the extension's division into `transported_and_resolved`.
- **`log_map` is not reproducible at the cut locus on CUDA** (one `icosphere(5)` vertex at radius
  ~pi differed by 3e-2 across builds and 3e-3 between runs); the docstring calls that angle
  arbitrary. Compare off the antipode.
- **DECLINED: a native `wp.mat22d` operator in place of the scalar expansion** (mat-vec alone,
  `float32` values: 0.97x at `sphere_med`, 1.10x at `sphere_large`, ~2 us of a ~50 us round).
  **The settle solves read a `float32` copy of their operator** (`_BatchedCg`'s `narrow_values`;
  `multigrid.csr_row_dot` accumulates at `x`'s precision and widens a narrower value as it reads):
  distance moves 2.1-2.3e-8 of range, transport angles 1.6-9.7e-5 degrees at most, resolved masks
  identical. Only 1.06-1.08x at `sphere_large` (flat below): most of the working set sits in the
  96 MB L2, so halving value bytes halves less than the bytes model says.
- **The Poisson solve's Chebyshev degree is at its optimum**: degree 2 / 4 / 8 / 12 against Jacobi
  at 2 562, 40 962, 163 842 rows: 12 wins or ties (8 is 3 % ahead at 2 562), so the small
  `heat_geodesic` row's 231 polynomial launches are this method's floor.
- **Tangent-field quantities are gauge-dependent and the float32 transport-angle storage sets a
  noise floor** (§12.4, §7.6).

### 16.11 `validation`, `adjacency`, `halfedge`, connected components

- **A pre-hooked forest is compressed before the hook from `ECL_COMPRESS_FROM = 1 << 21` nodes**
  (`connected_components.ecl_compress`: every node lowered by `atomic_min` to its current root, an
  ancestor, so no root moves). The pre-hook strings a large mesh's wandering numbering into long
  descending chains that every hook's finds walk: `face_connected_component_labels[lucy]` 43 -> 23
  ms (the union-find alone 28 -> 5.1), flat at 0.87 M faces. All six pre-hook / hook sites take it
  (`adjacency`, `graph`, `validation` x2, `selection` x2); each is pinned by a forced-threshold
  test.
- **An edge-parallel union-find replaces a CSR build whenever the answer is "smallest id per
  component".** `ecl_hook_edges` hooks the larger root under the smaller, so every label is the
  component's minimum node id whatever order the unions ran in, equal to the CSR path's labels.
  It drops `bsr_from_triplets` and its sort from every caller (`face_connected_component_labels`,
  `boundary_loops_batched`, `vertex_manifold_mask`, `successor_cycles`). **It needs its pre-hook**:
  from a bare identity forest the CAS hooks race (`rim_long`'s sequential-id cycles grow parent
  chains as long as the cycle; `fan_hub`'s 40 960 spokes retry against one root: `boundary_loops
  [rim_long]` 3.6 -> 26.4 ms, `is_vertex_manifold[fan_hub]` 2.0 -> 9.1). One `wp.atomic_min(parents,
  max(a, b), min(a, b))` per edge first (`ecl_init_parent_edges`; shared pair versions
  `ecl_prehook_pair` / `ecl_hook_pair` in `connected_components`) changes no root and fixes both
  (1.6 / 0.73 ms), for one launch on an ordinary mesh. "Changes no root" was true of the answer and
  false of the cost (§9's fixture-pair rule). Edges formed in the thread generalise: the
  vertex-manifold corner graph, the orientation parity graph, `faces_left_of_contour` (hooks faces
  across twins; no dual-edge list, cursor, trim or readback) and `cut_along_edges` (union-find over
  halfedges; the root-flag scan's ascending ranks *are* `unique_1d`'s sorted inverse).
  libigl's "every vertex below `max(faces)` is referenced" half runs in corner space.
- **A union-find flatten must never write into `parents`.** Writing labels over `parents` to save
  an allocation was byte-identical on CPU and wrong on CUDA (1-140 entries per mesh): another
  thread's path halving writes `parents[child] = grandparent` from a stale read over a stored root.
  A serial device cannot show it.
- **`face_connected_component_labels` hooks straight off the sorted keys**
  (`sorted_pair_prehook` / `sorted_pair_hook`; no compaction, readback or edge table). The first
  cut wrote a `(3n, 2)` table and read **0.80x at `lucy`** (the key staging copy another 0.07): a
  saving that removes a readback and adds `O(n)` traffic shrinks with the mesh and can invert;
  measure the top of the axis.
- **Solve parity through the halfedge mates, not an adjacency table**:
  `validation.face_flip_mask` runs the parity union-find straight off `halfedge.halfedge_mates`
  (`make_winding_consistent`).
- **A halfedge's mate answers "is this edge manifold"**: 1, 2 or 3+ is the mate code (`-1`, a
  partner, `<= -2`), so a per-face or per-halfedge manifold mask is one kernel after the pairing,
  not `unique_1d` plus inverse plus counts plus a gather. The twin rule is one `@wp.func`,
  `kernels/halfedge.mate_twin_defect`, which the bucketed and sorted twin builds and
  `remove_degree3_vertices`' input check call (the sorted-run spelling,
  `sorted_halfedge_run_class`, is gone with it). `face_adjacency(edges_paired=
  True)`: every edge on exactly two faces makes the adjacency the sort permutation read two to a
  row. `is_watertight` / `is_volume` / `is_vertex_manifold` / `is_edge_manifold` are a handful of
  launches with `n_vertices=`. `is_winding_consistent` / `is_orientable` / `face_flip_mask` /
  `repair.make_winding_consistent` take `n_vertices=` too (trusted, not checked) for the same
  `end_bit` sort: the sort 1.3-1.6x, the calls 1.15-1.41x from `bunny_decimated` to `happy_buddha`,
  answers unchanged (`Trimesh`, `make_solid` and `make_normals_outward` pass it).
  **`face_flip_mask` on a non-orientable mesh is not reproducible on CUDA** (the parity hooks race:
  31-68 of a few thousand bits differ between two unbounded runs on `boy` / `mobius` with half the
  faces flipped; CPU is byte-identical), so a ordito-against-ordito gate compares the flip mask
  on orientable input only.
- **`adjacency.sorted_face_edge_keys(faces, *, n_vertices=None)`** names the run four modules
  repeated: pack straight into the sort's double-width buffer (no staging copy, 670 MB at `lucy`)
  and sort only the bits a known radix needs (§13.1, §3.1); `face_adjacency(edges_paired=)`,
  `halfedge_twins`, `face_connected_component_labels` and `validation`'s sorted keys use it. Three
  internal sorters launch the shared `adjacency.face_edge_keys_and_order` kernel themselves
  (views cost ~14 us, §16.0). `_pair_halfedges` and the edge-manifold predicates use
  `adjacency.face_edge_keys` rather than `faces_to_edges` + `hash_indices_rows`. The radix sort's
  **upper half is free scratch after the sort**: seam flags, the adjacency emit's scan and the
  component forest live in `order[n:]`, one allocation fewer each (up to 336 MB at `lucy`).
  `hashed_occurrence_counts` gives an order-free share count on `unique_1d`'s own table
  (`is_watertight`); `is_vertex_manifold` gained `n_vertices=`. `edges_unique` takes
  `max_value`-bounded `unique_1d` (§13.1); the pair radix (`INDEX_RADIX_PAIR`) needs all 64 bits.
- **Hash tables versus sorts is a fact about the table shape, not a rule.** `point_duplicate_mask`
  is one `atomic_cas` table of `int32` point indices at >= 2n slots, bitwise compare, `atomic_min`
  to the class's first index (`-0.0 == +0.0` and identical NaN rows merge): 2 launches / 3
  allocations / 0 readbacks vs 14 / 21 / 2, **8-12x to 1 M points, 33x at 4 M**. **REFUTED: a
  hash-only validate for `halfedge_twins`** (directed-edge uniqueness through
  `hashed_occurrence_counts`): 1.26x on bunny, **0.59x on dragon** (random atomics into a 2^23-slot
  table lose to the radix sort); the region-boundary hash lost 0.74x at `dragon` and the sort
  shipped. Probe each table shape.
- **A derivation already paid for is often recomputed one call later**: `metrics` reduced the same
  array twice; `triangulate_point_cloud` sorted and hashed the same rows twice; `isotropic_remesh`
  built an edge grouping its next stage rebuilt; `voxel_down_sample` probed slots pooling had just
  probed. Grep a wrapper for two calls fed the same arguments before anything cleverer.
- **`halfedge_twins` pairs through per-vertex edge buckets on CUDA** (2026-10-05,
  `halfedge._pair_bucketed_halfedges`, whenever `n_vertices` is given, which every in-repo caller
  does): a counting sort of the halfedges by their edge's lower-*degree* endpoint (ties to the lower
  index; corner counts are one `count_occurrences` launch), then one thread per halfedge scanning
  its bucket for the run the key sort would give -- same twins, same two defect
  counts, each defect counted once by its run's lowest halfedge. Warp's `tri_tri_adjacency` has the
  bucketing but keys the *lower index* and matches one thread per vertex, a quadratic cliff on any
  hub (§12.8); the degree owner keeps a hub's edges in its spokes' buckets and the per-halfedge
  matcher makes a large bucket parallel work. Against the radix sort, unvalidated: 1.04-1.07x up
  to 82 k faces, 1.24x at 0.33 M, 2.35x / 2.23x / 2.58x at `dragon` / `happy_buddha` / `lucy`
  (0.36 -> 0.15 ms at `dragon`, ahead of Warp's 0.23), 1.26x on an 800 k-face fan with its
  400 000-spoke hub numbered first; validated (one readback either way) 1.02-1.35x below 0.33 M and
  2.06-2.15x at 0.87-1.1 M. Tables and counts identical everywhere probed, non-manifold
  `bunny_decimated` / `lucy` included. **The CPU keeps the sort**: the buckets lose 0.29-0.57x
  there at every size (`_BUCKETED_PAIRING_ON_CPU`, which the equivalence tests force on).
  `n_vertices` now sizes buffers on this path, so it must bound every index (it already did for
  `vertex_one_rings`). Pinned by `test_bucketed_twins_match_the_sorted_twins_on_*` and
  `test_bucketed_twins_on_a_hub`; double-counting a run or matching on the halfedge alone fails
  them.
- **`halfedge.halfedge_mates` is the order-free pairing the edge-classifying consumers read**
  (2026-10-05): per halfedge, the other halfedge of an edge carrying exactly two (either
  direction), `-1` alone, `-2 - lowest` for three or more, so a pair is acted on by its lower
  halfedge and a non-manifold edge once. Built by the edge buckets on CUDA from
  `_BUCKETED_MATES_FROM_HALFEDGES = 1 << 19` halfedges with `n_vertices` given, else by the key
  sort plus one pass over its runs (`sorted_halfedge_mates`). Every order-free edge classifier
  reads it: `face_connected_component_labels` (union-find hooks off `mate > h`),
  `edge_manifold_mask`, `is_vertex_manifold` / `vertex_manifold_mask` / `is_watertight` (corner
  graph), `is_winding_consistent` / `is_volume`, `is_orientable` / `face_flip_mask` (parity
  union-find), `face_defective_mask`'s neighbour pass, `remove_degree3_vertices`' input check, the
  sorted twin build and `boundary`'s pinched-rim twins; `combine.split` and `holes._JoinRim` now
  pass `n_vertices`. Against the sorted-run kernels on CUDA at `dragon` / `happy_buddha` /
  `lucy`: labels 1.81x / 1.83x / 1.90x, `edge_manifold_mask` 2.25x / 2.27x / 2.25x,
  `is_vertex_manifold` 1.58x / 1.78x / 1.83x, `vertex_manifold_mask` 1.75x / 1.86x / 1.83x,
  `is_winding_consistent` 1.99x / 2.36x / 2.25x, `is_volume` 2.0x / 2.1x / 2.26x,
  `is_orientable` 1.42x / 1.60x / 1.89x, `face_flip_mask` 1.48x / 1.66x / 1.92x,
  `face_defective_mask` 1.67x / 1.68x / 1.68x, `remove_degree3_vertices` 1.37x / 1.52x (`lucy`
  is non-manifold). **The price is the extra pass on the sort path, accepted by the owner
  (2026-10-05)**: on CUDA up to 70 k faces most calls are flat and the mask 0.81-0.85x (~13 us,
  one launch and one buffer), `halfedge_twins` without `n_vertices` 0.93-0.98x at every size; on
  the CPU 0.81-1.05x (worst `is_winding_consistent` / `is_volume` 0.82-0.88x). Keeping a
  sorted-run consumer below the gate would remove it at the cost of two kernels per consumer.
  The sort's free upper half cannot hold the mates from `halfedge.py` (`sorted_face_edge_keys`
  returns trimmed views). On the CPU the union-finds and `face_defective_mask`'s float atomics
  now run in halfedge order rather than sorted order: labels are unchanged (minimum ids), the
  neighbour sums move at rounding, and a non-orientable flip mask (best effort) may differ. Pinned by `test_halfedge_mates_*` (NumPy grouping oracle on both builders,
  defects whose non-manifold edge does not start at halfedge 0, a hub), which a bare `-2` code
  fails.
- **Order is a contract only where code reproduces it, so an ordered edge list keeps its order
  and sorts only its survivors** (2026-10-05). Every list built from the edge-key sort comes out
  in ascending `(max, min)` key order (`constants.py`'s packing), and no reference comparison
  reads that order (all sort rows or compare sets); what reads it is ordito itself:
  `_FlipTopology.edges_unique` and `_DecimationBuffers._group_edges` rebuild it byte for byte,
  `intersection.unique_edge_order_key`, `creation.ring_boundary_edge`, `holes._JoinRim` and the
  remesh region sort reproduce it, and the remesh collapse lock key (`scramble_index(row)`), the
  split / subdivide new-vertex numbering and `homology`'s lowest-edge tie-break read the row
  index. So `edges_unique`'s order stays. Within a class each edge appears once in these lists,
  so sorting only the selected halfedges reproduces the order exactly:
  `halfedge.key_ordered_halfedges` takes the flag scan and either compacts it through the key
  sort's permutation (`halfedge_mates(return_key_order=True)` on the sort path: nothing sorted)
  or sorts only the flagged halfedges (bucket path), several classes in one sort with the class
  above the key. Taken by `boundary_edges` / `oriented_boundary_edges` / `boundary_loops*` /
  `boundary_vertex_indices` / `ears` (`_BoundaryHalfedges` is now the mates), the pinched walk's
  twins, `region_boundary_edges` and `crease_edges`; byte-identical to the sort on both devices
  (120 CUDA / 96 CPU arrays: scan meshes, a 327 k-face sphere with 512 holes, `mobius`, `boy`, a
  pinched rim, a fin). CUDA against HEAD at `dragon` / `happy_buddha` / `lucy`: `boundary_edges`
  1.4x / 2.0x / 2.1x, `boundary_loops_with_offsets` 1.56x / 1.95x / 1.58x,
  `boundary_vertex_indices` 1.8x / 2.05x / 2.23x, `crease_edges` 1.33x / 1.45x / 2.0x,
  `region_boundary_edges` 1.55x / 1.7x / 2.07x on a half-mesh region (1.22x / 1.3x / 1.58x on a
  striped one whose seam is half the edges); the 327 k-face sphere 1.1-1.38x. **Prices, accepted
  under §9**: below the bucket gate the mates pass and the compaction launch cost 15-25 us
  (0.83-0.90x at 16-69 k faces); calls without `n_vertices` stay on the sort and pay the extra
  pass (`ears` 0.89-0.98x, `region_boundary_edges` without it 0.94-0.96x at scale); CPU
  0.79-1.0x (`crease_edges(include_boundary=True)` 1.09x). **Declined: `edges_unique` and
  `face_adjacency`** keep the full sort: they keep about 1.5 halfedges per face, whose sort costs
  about half the full one, and mates plus that sort is the full sort again (sort 0.341 vs mates
  0.158 ms at `dragon`, 11.65 vs 5.10 at `lucy`). A lowest-halfedge numbering would drop that
  sort but move every reproduction above and every remesh / homology answer.
- **`halfedge_twins` / `vertex_one_rings` gained `validate=`** (pass-0-only validation in
  `remove_degree3_vertices`); `is_edge_manifold` / `edge_manifold_mask` share
  `adjacency.face_edge_keys`. **Face-hop vertex morphology** (`expand_vertex_mask` /
  `shrink_vertex_mask`) marks all three corners of any face with a selected corner, 2.3-23x
  without an edge table and flat-or-faster with one (public `unique_edges=` removed).
  `crease_edges` is one `crease_flags` kernel on `predicates.vector_angle`, the predicate
  `face_adjacency_angles` uses. `face_self_intersecting_mask` runs its broad phase as one
  fixed-stride table with no count pass or readback (**fusing the narrow phase into the walk stays
  declined: 0.91x on `sphere_med`, 0.27x on `tangle_2`**); the two-mesh broad phase is the same
  (`intersection.collect_face_box_candidates`, moved from `validation`, with
  `candidate_slot_query` as shared prologue). The fixed-stride table pays where the consumer is
  cheap (verdict or mask: 1.16-1.24x); where it is heavy `float64` work the sparse slots roughly
  double that kernel, so the segment compaction launches over the kept count (`wp.lower_bound`
  on the scan). Storing per-slot segments to skip the recompute was 0.8x (a 26 MB `wp.empty`).
  `mesh_collision_pairs` / `mesh_with_mesh` run the narrow phase inside the segment kernel;
  `mesh_with_plane` computes plane dots in-kernel. A box predicate with an optional rotation must
  **branch, not multiply by the identity** (`0 * inf` is `nan`).
- **`remove_unreferenced_vertices`** is one mark, scan and compaction; `kernels/triangles.
  sort_face_indices` is deleted. A union-find pass's readback-free forms keep the deterministic
  property: a mask-then-scan compaction is deterministic where an atomic cursor is not.

### 16.12 `holes`

- **A packed engine fed by the split form pays the loop count twice**: `ordito.holes` builds every
  stage on `_PackedLoops`; two of the three places deriving rims from a mesh took `boundary_loops`
  (`split` over the batched buffer) then concatenated the views back (37.3 ms vs `_hole_loops`'
  2.15 ms on an 8 192-rim sphere). Both go through `_boundary_loops_packed` (2.0-2.1x at 512 rims,
  15-16x at 8 192). The tell is a wrapper calling the split form and packing the result. The third,
  `_single_boundary_loops`, keeps the split: `stitch*` requires exactly one rim per mesh. The
  benchmark hoists `boundary_loops` out of the timed callable (a hoist that removes a cost also
  removes an axis; say so in the row's docstring).
- **`fillable_loop_mask`'s two pinch tests are one per-vertex occurrence count** (`count > 1` is
  exactly the union of "visited twice by one rim" and "held by two rims"; the range guard is fused
  in because the histogram is indexed by the value read, so an out-of-range rim answers `False`
  where the host form raised `IndexError`). Its chord test runs over face corners (idempotent and
  symmetric, no `edges_unique`) and the pinch test and table scatter are one launch; the whole
  predicate is readback-free. `np.unique` alone was 85 % of the old long-rim call (§3.8).
- **The min-weight DP's traceback runs on the device, one thread per rim**: byte-identical (the DFS
  takes `(k, j)` before `(i, k)` as the Python stack did). A rim can emit fewer than `B - 2`
  triangles (forbidden chord or pinch leaves `prev = -1`), so each rim writes into its own padded
  block and reports a count; the single readback is the packed total which also tells whether any
  rim fell short (the compaction launch runs only then). Its stack is caller-allocated scratch, one
  slot per packed rim vertex (depth never passes `B - 1`). The rim pass writes positions and keys
  it already loads; the traceback writes into the output's tail and reads one shortfall scalar.
  **Not taken: sizing `_EdgeTable` from `len(vertices)`** (§16.4's deliberate negative-index guard,
  re-found as a "redundant" reduction).
- **`fill_min_weight`'s triangulation is not bit-reproducible on a degenerate rim, and never was**:
  `loop_rim_metrics` accumulates each rim's Newell normal with float32 `wp.atomic_add`, so on a rim
  with exact metric ties (pinched, or two rims of equal perimeter under `preserve_largest_hole`)
  which triangulation wins flips (2 distinct results over 30 launches; `_PackedLoops.perimeters`
  gave 15 distinct values over 30 calls on a cylinder). **Gate any A/B on this module on the CPU
  device**; do not read a CUDA difference on a degenerate fixture as a regression.
- **The span sweep is a chain of launches whose count is a floor — but they no longer have to be
  issued.** Invariant tables ride in the `HoleFillTables` bundle (§2.8), and the sweep is
  **recorded once and replayed** in groups of 8 with a device-side counter (§14.3); the launch
  count is unchanged, what changed is the price of each. "Launch-bound" overstated the ceiling at
  the long-rim point (6.06 ms wall vs 5.13 ms device on the 510-launch `rim_short` sweep:
  removing *every* launch was worth ~16 %). Per-span device time is nearly flat across a 127x
  work range (7.1 us at span 2 against 10.4 at span 500, block 128; a null kernel at that grid is
  3.0): device time is mostly per-kernel fixed cost, so the sweep responds to the *number* of
  kernels; `block_dim` 32/64/128/256 sweeps to 128 and stays there.
    - **REFUTED (arithmetic): a blocked interval DP.** Tiles on one tile-diagonal are mutually
      independent, but a diagonal holds `B/tile` tiles against the `B - span` blocks a plain span
      level launches, and that width fills the device. Merging `C` levels caps the block count at
      ~`B/C` and multiplies work by ~`C/2` (§14.9, §14.11).
    - **REFUTED: transposed mirrors of the DP tables so the apex loop's second read coalesces**
      (byte-identical, a loss): the tables are L2-resident, so the 32x transaction argument prices
      bandwidth this kernel does not pay, and the two extra stores and launch arguments were not
      hidden. Do not re-propose without a rim whose tables exceed L2.
    - **REFUTED by a free PTX diff: collapsing `triangle_fill_metric`'s duplicated geometry.** ~40 %
      of its flops recur in *different* basic blocks, but nvcc already eliminates them (hand-fusing
      changes three `sub.f32` of ~1 100): a PTX op-count diff costs no GPU time (§12.6).
    - **The persistent one-block-per-loop DP is refuted** (§14.9).
- **The fill DP's apex loop is a branch and bound, and its column read is a mirror**
  (2026-10-02). Every term `apex_cost` adds after the children is non-negative and both combines
  are monotone, so `combine(dp[i,k], dp[k,j])` is an exact `float32` lower bound: an apex whose
  children reach the lane's best is skipped, and after each lane's first apex one `block_min`
  shares the block's best, skipped with a strict `>` (a tie could still win on its smaller `k`).
  With the metric mostly gone the strided column read `dp[k, j]` was the sweep's cost, so every
  store also writes the free lower triangle at `(j, i)` (`store_dp`; `init_dp_base` seeds the
  mirrored rim edges) and the child is read there, contiguous in `k`. Byte-identical on 81 CUDA /
  54 CPU fills across all nine metrics; `refill_region(dp_only)` 1.15x at 0.87 M faces, 1.37x at
  1.1 M, 1.57x at 28 M (a 2 765-vertex rim), flat on short rims (launch-bound). The earlier
  "a transposed mirror is a net loss" was a *separate* table while the metric dominated.
- **The stitch band DP is a blocked wavefront** (§14.11): `stitch_dp_tile` gives one **block** a
  `32 x 32` square and launches one tile-diagonal at a time (2 049 launches -> 65 at two 1 024-
  vertex rims). DP sweep 11.0-13.3x on CUDA, 2.4-7.8x on CPU; `stitch_loops_min_weight` 1.76x at
  64-rims up to 6.94x at 1 024, byte-identical. There is no crossover (ahead at every rim from 10
  vertices, 7.1x CUDA / 11.6x CPU, to 2 048). The host side was never the lever (`came.numpy()`
  0.27 ms for 4.2 MB; the band traceback is sequential).
    - **`wp.tile_sum(wp.tile(x))[0]` is the barrier** (Warp exposes none, §10, §13.2). Probed by
      removing it: the 2-D wavefront fails 8/8 runs at 64, 128 and 256 lanes and passes at 32.
      **32 lanes is the optimum and also the width at which the barrier cannot be tested** (one
      warp needs none; `warp_count == 1` fast path, §12.6), hence `test_stitch_dp_tile_matches_
      diagonal` parametrizes `tile` over 32 **and** 64. A tuning constant that makes a correctness
      mechanism unobservable needs the test to cover a value it does not ship.
    - **The tiled schedule wins on CPU too, so it is the default on both** (unlike the fill DP's
      tiled engine: there the lanes are the win and CPU has one; here the win is the launch count).
      Its own optimum is 64 on CPU (4 % apart, inside §13.3's tolerance, not device-split).
      `stitch_dp_diag` stays as the reference schedule (`_run_stitch_dp(tiled=False)`). A
      `@wp.struct` type cannot be annotated (the decorator rebinds the name to a `Struct` *value*):
      a helper taking a bundle names `warp._src.codegen.StructInstance`.
- **Every other NumPy site in the module was priced and kept**: the `stitch_loops` monotonicity
  correction (LIS over the association array), the band traceback and `bridge_edges_smooth`'s
  Hermite strip are host-sequential or fixed-size (§3.8); `_PackedLoops`' cumsums and uploads are
  0.10-0.16 ms even at 8 192 rims.
- **`join_closest_components` edits its rim instead of regrouping per join** (2026-10-02,
  `holes._JoinRim`): a join appends a two-triangle patch, so the boundary loses the two bridged
  rows and gains the patch's two chords (tails unchanged) and two components merge; the table is
  kept in `oriented_boundary_edges`' row order (ascending undirected key, larger index the high
  digit), which is the pairing's tie-break, so every round sees the rows, order and label
  equalities a regroup would. Byte-identical faces on both devices over `open_parts_*` and
  `max_joins` / `max_distance` arms; `test_join_closest_components_rim_tracks_the_regrouped_boundary`
  bites on an unsorted insert and on a dropped diagonal. Each round is one upload, a block-per-row
  pairing (`CLOSEST_PAIR_BLOCK = 64`, a one-thread-per-row walk of ~200 members was a 34 us
  dependent chain) and one five-integer read; the patches are appended once. `open_parts_4 / 16 /
  64`: 1.51 -> 1.27, 6.15 -> 2.80, 25.9 -> 8.8 ms (2.9x at 64). The exact Kruskal-on-the-host batch
  (R28-2) was not needed for identity and is not built: per-round tie-breaks read row indices a
  bridge renumbers, so a batched pick must reproduce the table anyway. The rim's row count is
  constant across joins (two rows out, two chords in), so the table, partner, answer and best-key
  buffers are allocated once and `closest_pair_rows` re-arms the key as it decodes it, read
  through `read_values`: a further 1.45-1.49x on `open_parts_64` (9.25 -> 6.2 ms) and 1.36x on
  `open_parts_16`, identical faces on both devices.
- **VOID: sweeping `refill_region`'s `min_area` retry only to the failing loops' widest rim**
  (2026-10-03, R29-8): on the benchmark's cap region every rim fails the primary metric at
  `dragon` and `lucy`, the 2 765-vertex `lucy` rim included (and the longest rim on `bunny`), so
  the retry needs the full width.
- **The floor under every entry point is `boundary_loops_batched`, a flat ~2 ms** whatever the
  mesh (closed, §16.5): `fill_fan[holes_many]` is 2.2 ms of which 2.1 is that call. Bridge
  validation: §16.5.

### 16.13 `points`, `voxels`, clustering

- **`voxel_down_sample` builds no `wp.Volume`** (`voxels._VoxelTable`): a point-index CAS table
  keyed on the cell, the distinct cells sorted into `Volume.get_voxels()`' order (a 36-bit in-tile
  key, plus a stable root-tile pass when cells can span more than one 4096-cell root tile), then
  pooling probes through the volume's own float32 transform reproduced bit for bit. Byte-identical
  on both devices for every pooling and an explicit origin. Segments are read off the sorted
  buckets (no histogram or scan). `Volume.allocate_by_voxels` itself beats a hand-rolled dedup
  ~2.3x but loses to a one-`atomic_cas` table 1.3-3.2x (§12.9).
- **`cluster_decimate` clusters and deduplicates with two `atomic_cas` tables** and sorts only what
  sets the output order: the kept clusters' cell keys (`end_bit` from the cell bound) and the
  distinct surviving faces' ranked triples. The face table keys the sorted cluster triple and lowers
  each class to its smallest face index with `atomic_min`, so the first occurrence's winding is
  kept exactly as `unique_faces` did. Cell keys are packed in the cell kernel;
  `resolve_voxel_grid(return_cell_bound=True)` skips the cell hash's validating reduction;
  `unique_faces(max_index=)` skips its validating reduction and both counts come from one 2-slot
  buffer the compaction kernel publishes into a prefix view of a worst-case buffer.
  `test_cluster_decimate_matches_a_numpy_transcription` pins the numbering on a two-sheet shell
  (inner sheet reversed): a single sphere never welds two faces onto one triple, and keeping the
  largest index instead of the smallest fails its four coarse arms. **A two-pass argmin with an
  index tie-break is one `uint64` `atomic_min`** on the non-negative float's bits above the index
  (`contraction="closest"`, same winner and lowest-index tie). **Compact then `unique_faces`, not
  deduplicate every face** (§16.0's 0.19x at `lucy`). On CUDA `average` moves with its float
  atomics.
- **The convex masks' threshold sweeps walk only the slices that can reach a threshold**
  (2026-10-02). `hull_support_extremes` also stores each `(direction, slice)` extreme, widened by
  `1e-6 * max sum |d_i p_i|` over the slice because the sweeps recompute the dot in other code and
  FMA contraction can move it a ULP (`SUPPORT_TIE_SLACK`'s reason), and `support_indices` /
  `mark_hull_support` launch per `(direction, slice)` and return unless the slice's bound reaches
  the threshold. From `SUPPORT_SLICE_FILTER_FROM = 1 << 17` points; below it the one-thread-per-point
  grid is kept (0.70-0.91x at 36 k). With §14.12's direction blocking: `convex_superset_mask`
  2.6-2.7x at 0.4-0.5 M points and 7.6x at 14 M (subdivision 3), `convex_subset_mask` 2.5-3.2x and
  10.8x (256 directions), byte-identical on both devices. Pinned by
  `test_convex_masks_slice_filter_matches_exhaustive` on an off-origin cloud.
- **`statistical_outlier_mask`** reads back one value (a `float64` device threshold; can move an
  ulp; identical on 30 masks). **`outlier_probability` forms its normalizer on the device, and
  `wp.utils.array_inner(a, a)` returns an `np.float32`**, so the host's `value / n` was a `float32`
  division under NEP 50 and the device port must reproduce it or sit 1 ulp off. **`crop_points` /
  `points_in_*` flag, scan in place, compact.** `fit_plane` / `principal_axes` fold the centroid
  division into their consumers. `neighbor_distance_moments` fold width: §16.4.
- `point_duplicate_mask`: §16.11.

### 16.14 `registration` (ICP)

- **Both ICP loops share one correspondence search** (`icp_match`, warp-uniform cloud/mesh) and one
  stopping rule (`continue_icp_loop`); normals are read from the target's table at the index.
  Target normals need not be unit: both the fit and the MAD scale read through one
  `kernels/registration.target_unit_normal` (the scale had read the raw table, so non-unit normals
  put it on residuals scaled by each normal's length;
  `test_robust_scale_ignores_the_length_of_the_target_normals` and
  `test_icp_point_to_plane_accepts_non_unit_target_normals` fail on the raw read).
- **Point-to-plane ICP's pinned loop runs on the device** (`point_to_plane_round`, three launches a
  round under `_device.run_device_loop`, recorded once per call; iteration 0 on the host for the
  MAD scale). **Point-to-point `icp` likewise**: five launches a round (the correspondence pass
  moves each point by the kept transform in registers; `point_to_point_round` keeps a fit only if
  it carried weight and runs the host's `float64` stop test). **A recording (~70-85 us host plus
  ~19 us to launch) pays back after about two replayed rounds of 30-55 us gap**: a 1-2-round call is
  0.82x, a pinned 10-round call 1.07-1.46x. REFUTED: issuing K host rounds before recording (K = 2
  won 4-8 % on 2-round calls and lost up to 37 % on weightless ones).
- **Record while the device runs, then read, then launch** (`_device.record_device_loop`, which
  `run_device_loop` calls). Reading round 0's outcome before recording costs every gated call 2-8 %.
  A call weightless at its seed still records a loop of zero rounds (failure path only, 0.94x); a
  weightless round writing an **identity step** lets the closing pass report `inf` with no readback
  (bit-identical except an exact `-0.0`).
- **The MAD scale runs on the device without compaction**: `+inf` for excluded residuals, one radix
  sort, a prefix search, medians exact against `reduce.median`.
- **Tukey ICP stopped after 2 iterations at 4.8 degrees of error** with an explicit `robust_scale`
  smaller than the starting residuals: the loop tested `sum w r^2`, which a redescending kernel
  makes *rise* while points re-enter it. It now tests the biweight loss `2 rho`, monotone in
  `|r|` (`none` / `huber` / MAD-scale Tukey bit-identical);
  `test_icp_point_to_plane_tukey_converges_from_outside_its_kernel` fails on the old loop.
- **`cost` scores the returned transform** (correspondences searched again at the returned pose;
  at `max_iterations=0` the initial pose's; was one step behind `matrix`, `inf` at 0 iterations).
  This is Open3D's convention (`RegistrationICP`'s `GetRegistrationResultAndCorrespondences`);
  `test_icp_point_to_plane_cost_matches_open3d_evaluation` pins it via `evaluate_registration` +
  point-to-plane `compute_rmse` (`cost == rmse^2 * n`). Open3D's *reported* quantity differs (a
  point-to-point `inlier_rmse` plus `fitness`, whatever the estimator), hence the test scores
  through the estimator. Point-to-point `icp` keeps trimesh's convention (fit residual of the
  returned matrix against the correspondences it was fitted to). **Both are "the returned pose";
  they differ in which correspondences, and each follows its reference. Neither is a bug to
  align.** Both keep their costs in the moments accumulator.
- **A loop with a convergence break reports the break point** (§15.2): a hoist read 1.86x and was
  worth 1.02-1.06x; pin the iteration count. A twist-norm stop cannot be shown on sphere fixtures
  (rotation is a free gauge) and would change a public contract: declined. The normal equations
  accumulate through float32 atomics, so the plateau wobbles: gate on a trajectory.
  Point-to-plane against a cloud gathers correspondences inside the accumulation; it applies its
  step inside the next nearest search (1.03-1.10x at 20 fixed iterations).
- **Declined**: reading a cloud target through the nearest index inside the Procrustes kernels
  instead of a gather copy, and ping-ponging the accumulator to fold its memset (both 0.98-1.02x at
  2 562 and 40 962 points: the loop is paced by its per-iteration readback, which a removed launch
  hides behind); a `wp.mat22d`-style structural merge belongs to §16.10.
- **Cloud targets and mesh targets**: an unbounded `max_dist` is used on a mesh target (§16.6);
  `mesh_from_points` backs cloud targets (§16.6). Both ICP loops already run under
  `record_device_loop` with no per-iteration readback, so no readback lead remains there.

### 16.15 Preconditioners (Jacobi-Chebyshev, squared-Laplacian, adaptive, multigrid)

- **A single-level Jacobi-Chebyshev preconditioner wins on long Laplacian solves**
  (`linalg.chebyshev_preconditioner`, `preconditioner="chebyshev"`, `z = p(D⁻¹A) D⁻¹ r` at
  `CHEBYSHEV_DEGREE = 12`, one fused launch per step; `CHEBYSHEV_INTERVAL = 80`). The earlier claim
  that a single-level Chebyshev preconditioner is worse counted mat-vecs; a solve here is bound by
  launches. Measured gains: `lscm` 2.8-3.4x, `harmonic` k=1 1.4-1.5x, heat solves 1.4-3.3x,
  `arap` 1.8-2.7x on saddles. Losses: `arap[hemisphere 10]` 0.83x and `heat_geodesic[sphere_small]`
  0.96x at adoption.
    - **Opt-in per call site, and must stay so.** Fixed cost ~0.23 ms of setup plus the extra
      launches recorded into the solve's graph is 0.5-0.9 ms, so any *short* solve loses:
      `min_quad_with_fixed` at 50 % pinned 0.59-0.80x at every size from 576 to 17 689 unknowns, the
      heat system (`M - tL`, near-diagonal) 0.7-0.9x, near-converged repeated solves
      (`spd_column_solver_amortized[x50]`) 0.56x, hole-chain fixed-rim patches 0.84-0.91x. **The
      axis is the solve's length, not its size**: at 1 % pinned it wins from 576 unknowns up
      (1.3-2.1x). Column-solver defaults stay `"diag"`; opt-ins are the heat Poisson solves,
      `lscm`, `harmonic` at k=1, `arap`, and both `filter_implicit_fairing` solves. Making it the
      global default took a suite 40 minutes instead of 3 (use a per-process timeout in A/Bs).
    - **It needs a symmetric operator** (a polynomial in `D⁻¹A` is not symmetric when `A` is not;
      Jacobi always is): CG ran to its cap at 100 s a call on `filter_laplacian`'s asymmetric
      implicit system. That site is not converted (§16.8).
    - **`chebyshev_step` computed the wrong polynomial, in both preconditioners.** The
      semi-iteration's first iterate is `source / theta`; the second step read `source` unscaled.
      Still *a* polynomial, so the squared-Laplacian preconditioner (`theta ~ 1`) preconditioned
      fine and every solve test passed; at a Gershgorin bound of 2.2 (`theta ~ 1.1`) it went
      negative inside its interval (CG 14 000 iterations vs 46). Fixed with a `previous_scale`
      argument; pinned by `test_chebyshev_preconditioner_applies_the_chebyshev_polynomial`, a
      closed-form eigenbasis oracle. **A polynomial preconditioner's tests must check the
      polynomial, not the solve.**
    - **Take the Gershgorin *lower* bound too**: discs of `D⁻¹A` are centred on 1 with radius `r =
      max_i sum_j |A_ij| / |A_ii|`, so a diagonally dominant operator has spectrum in `[1 - r, 1 +
      r]`, far tighter than the `80 / n` lower end a Laplacian needs. `upper` is `1 + r` (not `1 +
      max(r, 1)`); `r` is floored at `1e-3`.
    - **Scale rows in the step, not in a copy** (`row_scaled` flag of `chebyshev_step`; `wps.
      bsr_copy` was 0.34 of a 0.57 ms setup). **Intervals are fitted on the device**
      (`kernels/linalg.chebyshev_steps`, `chebyshev_step` reads coefficients from an array): no
      Gershgorin readback (it drained the queued diffusion in `heat_geodesic`), at a coefficient
      load per step of 2-11 % on long solves when read before the row dot and ~0-3 % after: **read
      the coefficients after the mat-vec** (`lscm[hemisphere]` 0.97x the one cell below 1.0). The
      Jacobi diagonal and Gershgorin ratio come from one row walk (`jacobi_dominance_rows`);
      `jacobi_preconditioner` builds Warp's inverse diagonal on first apply.
    - **Degree and interval**: the iteration count is monotone in the degree once the recurrence is
      right; 12 sits on the flat part of wall time from 2.5k to 20k unknowns; a lower end at 5-20 /
      n costs up to 2x the iterations, 80-160 / n is flat. Folding the Gershgorin max into the row
      kernels is declined (a block-folded row walk gives each lane 16 rows where the row-per-thread
      kernel has one, costing more at large `n`).
- **A squared-Laplacian polynomial preconditions `smooth_region`'s `MᵀM`**
  (`linalg.squared_laplacian_preconditioner`, degree 12, `a = 40 / n`; the best lower end falls as
  `1/n`: 0.04 at 1 000 unknowns, 0.005 at 9 000). Three approximations of `M_ff⁻¹` were built:
  a V-cycle on `L_ff` (77 / 126 iterations, but **13-14 ms hierarchy setup**, half of
  `bunny_decimated`'s call), a Neumann series (no setup, weak: `D⁻¹L`'s spectrum reaches 2 where
  `(1 - λ)ᵏ` does not decay), and the Chebyshev polynomial (shipped: `smooth_region[bunny]` 69.5 ->
  17.0 ms).
    - **The upper end must be the Gershgorin bound, not 2**: clamped cotangent weights go negative
      on a regular grid's near-right triangles (spectrum at 2.18) and a polynomial fitted to `[a,
      2]` explodes (the uniform `saddle` took 2 824 iterations, 3x slower than before); with `b = 1
      + max_i Σ|L_ij| / L_ii` it is 153. The unit test's negative-weight arm reads 340 vs 37 under
      the mutation.
    - **The bound is `max_i sum_j |L_ij| / D_i` for general positive `D`** (`kernels/linalg.
      scaled_row_abs_sums`), with the lower end scaled with it: the `1 + max(dominance(L), 1)` form
      is `D⁻¹L`'s bound only when `D = diag(L)`, and `harmonic(k=2)` (`L M⁻¹ L` is `L D⁻² L` with
      `D = sqrt(M)`; `parametrization._solve_biharmonic`) diverged on it. `harmonic(k=2)` went 179
      / 381 / 265 multigrid rounds -> 52 / 160 / 224 (2.2-3.7x).
    - `SquaredLaplacianPreconditioner.from_factors` builds it from assembled factors (§16.8).
- **`preconditioner="adaptive"`**: Jacobi under `CG_CHEBYSHEV_PROBE_ITERATIONS = 150`, escalating
  warm-started to Chebyshev; `min_quad_with_fixed`'s default (`[saddle_graded pin1pct]` 39.0 -> 14.6
  ms, now a win against igl's 26.7; pin50pct flat). **Refuted for `arap`**: its warm-started steps
  are long, not short, so every step paid the probe and a readback (0.37-0.63x); `arap` keeps
  `"chebyshev"` unconditionally. `_AdaptiveCg` keeps its states across calls and therefore must not
  use pooled states (§16.16).
- **Smoother verdicts are §14.8** (a Chebyshev multigrid smoother is refuted; interval robustness).
- **The coarsest level is a checked LU inverse under a one-thread BLAS** (2026-10-02,
  `linalg._symmetric_inverse`). Every coarse level probed is non-singular (nullity 0, condition
  88-333 at n = 118-331), where `np.linalg.inv` equals `pinv(hermitian=True)` to 1e-14 relative at
  0.55-0.6x its cost; an infinity-norm condition above `1e9` (or `LinAlgError`) falls back to
  `pinv`, and both are symmetrized. **OpenBLAS's 48-thread default costs twice**: the small
  factorization is slower threaded, and its workers keep spinning after it returns and slow the
  launches that follow (cProfile: the setup's `_launch.launch` cumulative 21 vs 12 ms over ten
  calls). `openblas_set_num_threads_local(1)` (thread-local, OpenBLAS >= 0.3.27, found in
  `numpy.libs/` by `ctypes`; a no-op where absent) around the factorization recovers both:
  `multigrid_preconditioner` 1.52x / 1.88x / 1.75x on `saddle_graded` / `saddle` / `sphere_med`
  (11.0 -> 7.2, 10.2 -> 5.5, 15.4 -> 8.9 ms), CG iteration counts identical on both devices,
  residuals equal to ~1e-13. `inv` alone was 1.07-1.36x, the thread limit the rest. Prefer it to
  the process-wide `OPENBLAS_NUM_THREADS` (a six-module A/B of ordito rows found nothing else it
  helps). The level transposes are `array.csr_transpose` (one sort of `nnz` keys, bit-identical to
  `bsr_transposed`, ~0.4 ms of Warp host time a level removed).
- **The multigrid hierarchy's *setup* is the blocker** (several sparse-op calls per coarsening level
  at Warp's fixed per-call cost, not the aggregation algorithm): every losing case loses by exactly
  that; with a free setup all would win. **No `bsr_mm` runs in a well-coarsened level**: the smoothed
  prolongator `(I - ωD⁻¹A)P0` is `A`'s entries re-keyed by their column's aggregate plus the
  identity's, one `csr_from_triplets` (`smoothed_prolongator_triplets`; 1.9-4.0x on the
  prolongator), and `Pᵀ(AP)` is each fine row's outer product of its `P` and `AP` rows as triplets
  (`galerkin_triplets`; 1.14-1.8x on the Galerkin product, no transpose multiplied). Setup
  1.23-1.51x on the `bunny_decimated` to `dragon` cotangent Laplacians, setup plus a `1e-8` solve
  1.15-1.28x below `dragon`, CG iteration counts identical; patterns equal except 4 of 1.08 M
  coarse entries that cancel exactly in one summation order. Every hierarchy matrix is
  `nnz_sync`ed (the triplet build leaves `nnz` at the triplet count). `AP` is the same shape
  (`product_triplets`, one triplet per `A_ij P_jl`: 1.7x on the product at the bunnies, 1.04x at
  `dragon`'s 9 M triplets; setup a further 1.11-1.19x, residuals identical); the direct triple
  product (`nnz(A) · deg(P)²`, ~75 M at `dragon`) is too large to assemble.
  **The triplet products need a budget, and their counts need 64 bits.** A level whose coarsening
  stalls goes nearly dense and the per-row products turn quadratic in row length (`icosphere(6)`
  at `theta = 0.2`: 1.04 G triplets; `dragon`: 8-80 G per level), where `bsr_mm`'s memory follows
  its output. Past `_MULTIGRID_TRIPLET_FACTOR = 16` times the operator's entries plus rows (cap
  2^30) a product falls back to `bsr_mm`. The counts and their scan are `int64`: an `int32` scan
  wrapped to a positive 440 M on `dragon` and passed the budget test, an out-of-bounds write
  (CUDA error 700). The default levels emit 1-3x their operator. The `"auto"` gate's decision ("will the hierarchy pay for itself") belongs
  on a property of the *operator*, not of problem size or an extrapolated iteration count (both
  tried, both wrong): off-diagonal dominance (`CG_MULTIGRID_DOMINANCE`) separates the cases. **Never
  route a new caller through `"auto"` without re-measuring on its own systems** (an operator with a
  favourable dominance on one mesh can read unfavourably on another of identical connectivity).
  **DECLINED (2026-10-02): an operator-chosen strength threshold.** Interleaved, setup plus a
  `1e-8` solve: `theta = 0` beats the default 0.05 by 1.0-1.3x on isotropic cotangent systems
  (`saddle`, `sphere_med`, `hemisphere`, `bunny_decimated`) and on a 3x-stretched saddle (1.5x),
  but loses 1.8x on `saddle_graded` (316 iterations vs 51) and 1.07-1.17x on a 10x-stretched
  one; `theta >= 0.2` stalls the coarsening. No operator statistic separates the cases (the
  weak-entry fraction and the median `|A_ij| / sqrt(A_ii A_jj)` both interleave: 0.051 on the 3x
  saddle that prefers 0, 0.075 on the graded one that needs 0.05). 0.05 never loses badly, so it
  stays. Iteration counts in a sequential theta sweep are trustworthy, its clocks are not (each
  later theta read slower on unchanged hierarchies; §15.7). The connection-Laplacian
  operator cannot reach the gate (the hierarchy is scalar-CSR-only). The smoothed-aggregation
  hierarchy serves `harmonic` at `k >= 2`, `"auto"` and `"multigrid"`.
    - The multigrid damping is folded into the inverse diagonal on the device
      (`kernels/algorithms/multigrid.damped_inverse_diagonal`): no level reads its spectral radius
      back; every consumer's arithmetic is unchanged, byte-identical on CPU; on CUDA the device
      `pow` moves `omega` in its last bit. Flat on the clock (the removed sync was queued behind the
      aggregation's count readback). The growth is no `wp.utils.array_inner` pass either (2026-10-02):
      the last power step stores per-block `|y|^2` partials and every `damped_inverse_diagonal`
      block folds them (fixed order; pre-folded past 4 096 partials), and the final spread hop
      counts the aggregate sizes: two operations fewer a level, bit-identical diagonals and labels
      at 40 k-2.6 M rows, the setup region 1.03x at 655 k and level at 40 k. Blocks own 256 rows,
      not the `reduce` fold width of 1 024 (40 blocks at 40 k rows read ~2 % slower). The root
      flags are scanned in place, the first spread hop is written into the MIS loop's spent
      `next_state`, the damped diagonal into the power iteration's spent `x`, and the undecided
      counter is cleared by each round's first launch: three allocations and a memset per round
      fewer, byte-identical preconditioner applies on both devices, flat on the clock.
      `_JacobiChebyshev` reads the diagonal and ratio from one row
      walk. **DECLINED: capturing the multigrid MIS loop** to drop its per-round readback (ceiling
      ~2 % of the preconditioner build vs a medium rewrite with a documented reverted precedent,
      §12.2, and a device-infinite-loop hazard where the round cap is a host `range`; the odd
      `state` ping-pong is removable since `mis_decide` reads only `state[i]`). Fusing the
      V-cycle's matvec into its Jacobi sweep is declined (inside a captured loop, §14.10). Voxel
      aggregation is refuted (§14.9).
- **A direct GPU factorization (cuDSS) wins only where the multigrid gate already fires**, and is
  not installed. The cost is the symbolic plan, not the numeric work (flat in conditioning), so it
  loses badly wherever CG converges quickly. **Do not add it as a one-shot backend.**
  **DECLINED (2026-10-02, measured): plan reuse for ARAP's loop** (nvmath-python 1.0 /
  `nvidia-cudss-cu12` 0.8 in a throwaway env, factorized once and re-solved per iteration for
  both columns): per-iteration solves are 1.5-2.8x faster than the batched CG (0.25-0.32 vs
  0.47-0.70 ms), but plan plus factorization is 25-90 ms against a whole default 10-iteration
  `arap` of 2-9 ms (0.04-0.23x on `saddle_small` / `saddle` / `hemisphere`), break-even ~150
  iterations; and it would make ordito depend on more than `warp-lang`. Traps if opened: AMD reordering beats the defaults without the
  MT layer; a one-shot `direct_solver` re-creates the handle; `reset_operands(a=new)` drops the
  plan; `libcudss.so` is not on the loader path (preload it with `ctypes.CDLL(...,
  RTLD_GLOBAL)` from the wheel's `nvidia/cu12/lib`, found through `importlib.metadata`); systems with **empty rows** (unreferenced free
  vertices) are singular for a direct solver where CG leaves them at the initial guess. A host
  launch count cannot bound the solve's share (the CG is graph-captured, §15.10).

### 16.16 The CG solver (`_BatchedCg`, Chronopoulos-Gear, one-block, settle, precision)

#### Routing and state

- **Every scalar `float64` solve runs `_BatchedCg`** (one column too); `solve_spd` does whenever
  its preconditioner is `None` or one `linalg` built *for that matrix* (`jacobi_preconditioner`,
  `chebyshev_preconditioner`, tagged with a weakref to their matrix); any other `LinearOperator`
  still goes to Warp. `warp.optim.linear.cg` records a fresh conditional graph per call (§12.7);
  no scalar or `wp.mat22d` solve reaches `wpl.cg` any more. Every `wpl.cg` call passes `atol=0.0`
  (§12.7).
- **A solve keeps one recorded state per operator** (`_cached_solver` in a
  `weakref.WeakKeyDictionary`, per `(operator, configuration)`), so a hoisted operator replays its
  recorded loop. **The state must not hold its own key**: a `_BatchedCg`, `_JacobiChebyshev` or
  multigrid level holding the matrix keeps the weak entry alive for ever. The cached state is built
  over `_storage_alias(matrix)` (a second `BsrMatrix` sharing the arrays), which holds every pointer
  the graph recorded and not the key; the key carries the arrays' identities, so replaced storage
  gets a new state. `test_solver_cache_does_not_outlive_its_operator` pins the lifetime with
  `gc.collect()`.
- **The graph writes `x` through a pointer taken at record time**, so a state that outlives one
  call owns its solution buffer and copies in and out; the right-hand side is read outside the
  graph. Returned device arrays alias the state and are overwritten by the next solve. The guard
  is `test_spd_column_solver_reads_a_rewritten_rhs_on_every_call` (re-solving the *same* rhs
  passes a stale replay).
- **Values rewritten in place are not detected, and need not be for correctness** (the mat-vec
  reads them live; only the preconditioner goes stale: a rate change for Jacobi, used deliberately
  by `smooth_region_boundary`; a Chebyshev or multigrid state with a stale interval is a
  definiteness risk, so no in-repo caller rewrites under one).
- **A fresh operator of a seen shape takes a pooled state** (`_cached_solver(..., pooled=True)`,
  `_BatchedCg.refresh`, `kernels/linalg.refresh_pooled_operator`): the pooled state owns a copy of
  its operator and one launch copies the next operator in and re-derives the narrowed values, the
  Jacobi diagonal and the Chebyshev ratios (refit on the device) so the recorded graph replays.
  **Refreshed on every use**, not only when the operator changes (an owning state cannot see
  in-place rewrites, and two live operators of one shape may alternate). The first operator of a
  shape keeps an aliasing state (a hoisted operator pays no copy). Only immediate-use callers pool.
  The squared-Laplacian operator pools too. `test_pooled_solver_state_follows_each_operator` bites a
  no-op refresh and a skipped refit (the refit is a rate, so only the step comparison sees it).
- **A `wp.mat22d` operator is solved as its scalar expansion** (`_scalar_expansion`, cached per
  operator; `kernels/linalg.expand_block_csr_2x2`), with `wp.vec2d` operands viewed as `float64`.
  Exact: Warp's blocked Jacobi inverts each diagonal block's diagonal *coefficients*, which is
  scalar Jacobi on the expansion.
- **Sibling solvers**: `linalg._BatchedCg` works around `TiledDot`'s O(n) per-block reduction under
  `batch_offsets` (§12.7); `fix_self_intersections`' captured CG is not waste (§16.4).

#### The iteration: two launches a round

- **A round is two launches: the Chronopoulos-Gear iteration** (1989). One launch does the
  mat-vec with *all three* dots in one `wp.vec3d` reduction (`cg_matvec_dots`); one does the
  update with the Jacobi apply (`cg_update`). `alpha` is recovered from `r.u` and `w.u` (`w = A u`)
  and `s = A p` is carried by recurrence. **13 us a round** against a floor of ~4.3 us for a bare
  `capture_while` round plus ~1 us per replayed node (the conditional test is the largest single
  part). Folding the dots' second stage into the consumer blocks (three nodes) took a round only
  17 -> 16 us: each fold is a block-wide `tile_sum` of ~0.4 us on the critical path and two chained
  reductions stay two.
    - `wp.tile(v, preserve_type=True)` reduces `wp.vec2d` / `vec3d` in one pass (§13.2).
    - **Narrower blocks are worse**: `block_dim` 32 / 64 / 128 / 256 cost 25 / 17 / 13 / 13 us a
      round (each lane walks several rows serially; the row walk is the latency).
    - **Iteration counts equal standard CG's on 72 of 82 solves**, the rest within 1.8 % either way
      (2 956 -> 3 009 on the graded saddle's 3 000-round solve, 386 -> 371 on `lscm`); no stability
      loss on any fixture.
    - The stopping test comes first in `cg_update` (the triple describes the `r` the update starts
      from), so the loop runs one detect-only round past convergence; the reported count is the
      rounds that stepped. Recurrence scalars are double-buffered (`*_new` written by the update,
      carried to `*_old` by the next mat-vec) because every block of the update reads them. Setup is
      two launches (`cg_initial`: `r = b - A x`, `u`, `p = s = 0`, `||b||^2` partials, and the
      initial-guess copy; then `cg_seed`). Under the Jacobi-Chebyshev polynomial the update writes
      `D^-1 r` straight into the polynomial's input.
    - **A solve's prologue replays with its loop**: `cg_seed`, the preconditioner's first apply and
      the settle reset are recorded ahead of `capture_while` in the same graph, so a cached solve
      issues only `cg_initial` (Chebyshev `solve_spd` 13 -> 1 launches, 2.6-2.9x on a short solve).
      The loop state and count share the `float64` scalar buffer's head; per-column scalars live in
      one `(10, n_columns)` buffer (threshold and `r.r` adjacent), so a host result is two reads
      (`host_result`) and `_cg_residual_and_tolerance` reads an adjacent pair in one copy
      (`_read_pair`). `cg_step_scalars` (step only when `r.z != 0` and `p.Ap != 0`; `!= 0` rather
      than `> 0` because a negative-definite `min_quad_with_fixed` system uses it) keeps `tol=0`
      from reading `0/0`. Named runs: `cg_advance`, `cg_threshold`, `multigrid.chebyshev_update`.
- **Folding a round's dots in every block past `CG_FOLD_MAX_BLOCKS = 256` tiles is a loss**, though
  the unfolded path's third node looks like the cost: on transport's stacked 70 756-row solve (277
  tiles) a round is 10.4 us `cg_matvec_dots` + 7.2 us `cg_coefficients` + 5.1 us `cg_update`;
  raising the cap to 512 / 1 024 or doubling the span until the column fits under 256 blocks
  measured 0.90-0.97x on transport, 0.82-0.85x `transport_scale[sphere_med]`, 0.65x
  `[sphere_large]`, 0.84-0.86x `heat_geodesic[sphere_large]` (redundant per-block folds and
  per-thread `float64` divisions cost more than the node). An empty in-graph launch is 1.1 us; a
  `float64` block reduction adds 0.9-1.5 us and a `wp.vec3d` one 2.2-3.5 us.
- **Folding `p.Ap` partials into `csr_matvec` is declined**: the kernel is shared with the V-cycle
  at four sites that want no partials, and making it tiled would expose them to the ragged-tail
  hazard `cg_dot_partials` documents (a long tail cost several times a short one at nearly the same
  size: 1.5x win -> 0.82x loss).
- **Batching host checks**: `CG_CHECK_EVERY_FALLBACK` cadence; §14.6. `cg(check_every=0)`: §14.6.
- **The block-per-row `bsr_mv` is CUDA-only** (fixed 2026-10-05). `_row_path` passed
  `tile_size=_HEAVY_ROW_TILE` on every device, where `bsr_mv`'s own heuristic tiles only on CUDA;
  on the CPU the `launch_tiled` kernel runs one lane per block (§12.2), forms `A u` from one lane's
  share, and both precisions returned **zero iterations and a zero solution** with no error (rows
  over `_HEAVY_ROW_TILED_ENTRIES` on a column too long to fold; no fixture reached it until the
  `bsr_mv` registration guard forced the path). The CPU now keeps the lane-per-row kernel, pinned
  by that test's `block_per_row` arm (fails with the device check removed).

#### Settle solves (heat)

- **The heat diffusions run one continuous solve with a device-side settle test**
  (`linalg.solve_spd_settled`, `_BatchedCg(settle=...)`, kernels `cg_settle_change` /
  `cg_settle_decide`): the recorded loop body is 16 rounds plus two launches that compare the
  iterate with the one 16 rounds earlier and may clear the loop condition; nothing is read back.
  Round counts equal the `W=16` column of the need measurement exactly (saddle 304, `sphere_med`
  256, `sphere_small` 112 / 96). The settle solves are bound by their round count (§16.1).
- **A settle solve reads a `float32` copy of its operator** (§16.10).

#### One-block solves

- **Small solves run as one launch, one block per column** (`cg_one_block`, gated at
  `CG_ONE_BLOCK_MAX_ROWS = 1024`, Jacobi or the squared-Laplacian polynomial). Against an
  already-recorded `_BatchedCg` (its best case): 1.5-2.1x at 642 rows; a loss from ~1 200 rows at
  one column and ~1 900 at three. Iteration counts identical (52, 21 on `fix_self_intersections`'
  two region solves). The crossover (~1 024 items) is the same as the ear loop's (§16.5): a graph
  recorded per call is most of a small round loop and one block of barriers undercuts it until the
  work outgrows one SM.
    - **The single block is `float64`-bound, not barrier-bound**: one SM of this GeForce part runs
      `float64` at 1/64 of `float32`, and a 1 000-row mat-vec is ~13 000 double ops (~2.7 us of one
      SM's FP64 pipe). So the squared-Laplacian polynomial is applied **in `float32`** inside the
      one-block kernel (a preconditioner need only be a fixed positive-definite map; operator,
      residual and answer stay `float64`): 3.98 -> 2.16 ms (1.84x) against 3.17 ms with the
      polynomial in `float64`, iterations unchanged. The batched path keeps `float64` throughout.
    - The one-block gate's per-row schedule has no other axis: a warp per row needs warp shuffles
      and Warp exposes only block-wide tile reductions; block width swept 128-1024 lanes is the
      schedule. `_cg_one_block` writes each column's round count into its float results, so a host
      caller allocates no integer buffer and reads once.

#### Precision, storage and the screened-Poisson solves

- **The CG kernels are generic over the vectors' *storage* precision** (`OverloadTable`s
  `CG_INITIAL` / `CG_MATVEC_DOTS` / `CG_ROUND_DOTS` / `CG_UPDATE` at `float32` and `float64`); the
  cross-block dot folds, `alpha` and `beta` are always `float64`. "It is `float32`" was not a
  reason to leave `reconstruction`'s solves on `wpl.cg`. Storage stays `float32` for the
  screened-Poisson grids on purpose (the dense top level is 17 M nodes at depth 8 and 134 M at 9,
  bound by bytes). The dense grid is matrix-free (a CSR of the 7-point stencil is more memory than
  the solve), so it has its own mat-vec + dots and initial-residual kernels in
  `kernels/reconstruction.py`, sharing `screened_laplacian_row`, `cg_round_terms`,
  `cg_publish_round_dots` and `cg_update` with the CSR path, driven by `_device.run_device_loop`.
    - **Deriving a scalar from device values in every thread is `float64` division at scale**:
      `cg_update` reading `alpha` / `beta` off the fold and dividing per thread was 0.53 -> 1.0 ms
      at 17 M entries (FP64 at a small fraction of FP32 on this part). Past `CG_FOLD_MAX_BLOCKS` a
      `cg_coefficients` launch derives them once per column (it replaces the old finalize).
    - **The block count is the reduction's cost** (§13.2): 66 000 one-tile blocks ran the stencil
      mat-vec with its dots at 190 us a round; `cg_layout` spans a long column's blocks over powers
      of two of tiles to land near `CG_TARGET_BLOCKS = 2048` (80 us). **Only the reducing kernel
      wants it**: the same span made the pure-stream update 1.4x slower, so the update always
      launches one tile a block.
    - **On L2-resident levels the element arithmetic is the cost, the reductions' included.** At 2.1
      M nodes (8.6 MB a vector against 96 MB of L2), widening every entry to `float64` made the
      level slower than Warp's (10.9 vs 9.4 ms). Element updates run at the storage precision (what
      `wpl.cg` does) and so do the in-block dot sums (`cg_round_terms`); only the few thousand block
      partials are folded in `float64` (`cg_widen`) and the scalars derived there. Per level on
      `bunny`, depth 8: 1.8 / 2.4 / 4.9 / 77 ms (`float64` in-block sums: 2.0 / 3.2 / 6.6 / 79). On
      `float64` storage every conversion is the identity, so the mesh solves are unchanged bit for
      bit.
    - **`float64` reductions buy no accuracy on a `float32` system; `float32` storage sets the
      floor.** True relative residual `||b - Ax|| / ||b||`, recomputed in `float64` on the host, is
      identical to 3-4 digits between `float32` in-block sums, `float64` sums and Warp's
      all-`float32` `cg`: at a 100-iteration cap (4.2e-4 / 2.2e-3 / 2.9e-3 / 5.0e-3 by level) and
      with the cap lifted (8.90e-5 at `tol=1e-4`; at `tol=1e-6` all three stall at the same 4.5e-6
      / 9.3e-6 / 2.0e-5, the `float32` storage limit). Revisit only for a `float32` system whose
      tolerance is below that floor.
    - **A lane-per-row mat-vec fused with a block reduction loses from ~16 entries a row** (the
      barrier holds every lane until the block's longest row finishes). 400 000-row sweep, fused vs
      `bsr_mv`'s lane-per-row kernel: 0.042 / 0.064 ms at 8 a row, tie at 16, 0.22 / 0.13 at 32,
      0.82 / 0.64 at 128 (`bsr_mv`'s 64-lane tiled kernel 0.31 there). A long column of rows over
      `CG_HEAVY_ROW_ENTRIES = 16` calls `bsr_mv` and reduces in `cg_round_dots`; a *short* one keeps
      the fused launch even with heavy rows (below the fold threshold a round is launch-bound:
      `smooth_region`'s 20-a-row three-column systems read 0.91-0.94x on the split path). **Decide
      on the true count, never `nnz`** (the `warp.fem` system's `nnz` was 197 M against a true 44
      M, and `bsr_mv`'s tile heuristic reads the capacity).
- **`float32` CG for the mesh solves: measured, and it is the tolerance, not the precision, that
  moves the clock.** Jacobi-CG on `icosphere` heat and Poisson systems at equal tolerance takes the
  same iteration count in either precision; `float32` is 1.03-1.15x faster to 41 k unknowns
  (launch-bound) and 1.3-1.4x at 164 k, while loosening `1e-10` to `1e-5` alone saves 1.5-2x in
  either. **It is not taken for the mesh solves**: `float32` storage floors the relative residual
  near `1e-6`, so it cannot run at the shipped `1e-8` / `1e-10`; the switch is really "loosen and
  narrow" (100x the solution error). On 1 %-pinned cotangent Dirichlet systems (636 to 40 553
  unknowns) `float64` Jacobi-Chebyshev at the *shipped* `1e-8` is faster than either (1.2-2.6 ms vs
  3.1 at 40 k) and 100x more accurate. The lever is the preconditioner, not the width.
